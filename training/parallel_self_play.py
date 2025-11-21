import torch
import numpy as np
from typing import List, Optional, Dict
import random
import multiprocessing as mp
import os
import tempfile
import logging
import sys
import time
import gc

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

BATCH_SIZE = 16  # Optimized for RX 7900 XTX - better GPU throughput

_WORKER_MODEL = None
_WORKER_DEVICE = None

def _worker_init(model_path, device):
    """Initialize worker ONCE with model loaded."""
    global _WORKER_MODEL, _WORKER_DEVICE

    # Disable threading to avoid CPU contention
    os.environ['NUMBA_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # ROCm optimizations for AMD GPUs
    os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
    os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:512'

    # Load model ONCE per worker
    if model_path and os.path.exists(model_path):
        from training.training_types import TrainingConfig
        from training.optim_train import ChessTrainer

        config = TrainingConfig(device=device)
        trainer = ChessTrainer(config)
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            trainer.model.load_state_dict(checkpoint)

        _WORKER_MODEL = trainer.model.to(device).eval()
        _WORKER_DEVICE = device

        logger.info(f"Worker initialized with model on {device}")

def _game_worker_batch(args):
    """
    Fixed game worker with NO global state dependencies and guaranteed cleanup.
    Each worker runs in complete isolation with local memory management.
    """

    import torch
    import numpy as np
    import random
    import sys
    import os
    import time
    import gc

    # Local imports only - ensure fresh modules
    from training.training_types import TrainingExample, TrainingConfig
    from game3d.common.coord_utils import coord_to_idx
    from game3d.common.shared_types import (
        VOLUME, SIZE, Color, Result, PieceType, COORD_DTYPE, INDEX_DTYPE,
        FLOAT_DTYPE, POLICY_DIM, N_TOTAL_PLANES
    )
    from game3d.movement.movepiece import Move
    from game3d.board.board import Board
    from game3d.game.factory import start_game_state
    from game3d.game3d import OptimizedGame3D
    from training.opponents import create_opponent, OpponentBase
    from training.optim_train import ChessTrainer
    from game3d.game.terminal import is_fifty_move_draw, is_threefold_repetition

    game_id, _, device, opponent_types, epsilon, max_moves = args

    # Use the SHARED model (no loading!)
    model = _WORKER_MODEL
    if model is None:
        raise RuntimeError("Worker model not initialized")
    worker_logger = logging.getLogger(f"worker.{game_id}")
    state_cache = {}  # Local per-worker cache (eliminates global StateTensorPool)

    try:
        worker_logger.info(f"Worker {game_id} starting...")

        model = _WORKER_MODEL
        if model is None:
            raise RuntimeError("Worker model not initialized")

        # Initialize game
        initial_state = start_game_state()
        game = OptimizedGame3D(board=initial_state.board, cache=initial_state.cache_manager)

        opponents = {
            Color.WHITE: create_opponent(opponent_types[0], Color.WHITE),
            Color.BLACK: create_opponent(opponent_types[1], Color.BLACK),
        }

        # Safety limits
        MAX_MOVES_PER_GAME = max_moves
        error_count = 0
        move_count = 0
        examples = []
        start_time = time.perf_counter()

        # Main game loop
        while move_count < MAX_MOVES_PER_GAME and not game.is_game_over():

            # Draw detection
            if is_fifty_move_draw(game.state) or is_threefold_repetition(game.state):
                break

            # Get legal moves
            legal_moves = game.state.legal_moves
            if legal_moves.size == 0:
                break

            # Model inference
            state_array = game.state.board.array()
            state_tensor = torch.from_numpy(state_array).float().unsqueeze(0).to(device)

            with torch.no_grad():
                if device == 'cuda':
                    with torch.amp.autocast(device_type=device):
                        from_logits, to_logits, value_pred = model(state_tensor)
                else:
                    from_logits, to_logits, value_pred = model(state_tensor)

            from_probs = torch.softmax(from_logits, dim=-1).cpu().numpy()[0]
            to_probs = torch.softmax(to_logits, dim=-1).cpu().numpy()[0]
            value_pred_scalar = float(value_pred.cpu().numpy()[0, 0])

            # Process moves
            from_coords = legal_moves[:, :3]
            to_coords = legal_moves[:, 3:6]

            # Get occupancy data
            occ_cache = game.state.cache_manager.occupancy_cache
            from_colors, from_types = occ_cache.batch_get_attributes(from_coords)
            to_colors, to_types = occ_cache.batch_get_attributes(to_coords)

            # Filter valid moves (piece belongs to current player)
            valid_mask = from_colors == game.state.color
            if not np.any(valid_mask):
                worker_logger.warning("No valid moves for current player")
                break

            valid_moves = legal_moves[valid_mask]
            n_valid = len(valid_moves)

            # Create policy targets DIRECTLY - NO GLOBAL POOLING
            from_indices = coord_to_idx(valid_moves[:, :3])
            to_indices = coord_to_idx(valid_moves[:, 3:6])

            # Allocate fresh arrays (memory is cheap compared to crashes)
            from_target = np.zeros(POLICY_DIM, dtype=np.float32)
            to_target = np.zeros(POLICY_DIM, dtype=np.float32)

            # Fill sparse targets
            from_target[from_indices] = from_probs[from_indices]
            to_target[to_indices] = to_probs[to_indices]

            # Normalize
            from_sum = from_target.sum()
            if from_sum > 0:
                from_target /= from_sum

            to_sum = to_target.sum()
            if to_sum > 0:
                to_target /= to_sum

            # Create examples
            player_sign = 1.0 if game.state.color == Color.WHITE.value else -1.0

            # Create SINGLE example for this position
            # We copy arrays to ensure they persist after the game state changes
            ex = TrainingExample(
                state_tensor=state_array.copy(),
                from_target=from_target.copy(),
                to_target=to_target.copy(),
                value_target=value_pred_scalar,
                move_count=move_count,
                player_sign=player_sign,
                game_id=f"{game_id}_{move_count}"
            )
            examples.append(ex)

            # Choose move
            move_scores = from_probs[from_indices] + to_probs[to_indices]

            if random.random() < epsilon:
                chosen_idx = random.randint(0, n_valid - 1)
            else:
                opponent = opponents[game.state.color]
                if isinstance(opponent, OpponentBase):
                    rewards = opponent.batch_reward(game.state, valid_moves)
                    move_scores += 0.5 * rewards

                chosen_idx = int(np.argmax(move_scores))

            # Execute move
            chosen_move = valid_moves[chosen_idx]
            submit_move = Move(chosen_move[:3], chosen_move[3:6])
            receipt = game.submit_move(submit_move)

            if not receipt.is_legal:
                error_count += 1
                if error_count >= 3:
                    break
                continue

            game._state = receipt.new_state
            game._state._legal_moves_cache = None  # Force cache refresh
            move_count += 1

            # Observe move for opponent learning
            opponent = opponents[game.state.color]
            if isinstance(opponent, OpponentBase):
                opponent.observe(game.state, submit_move)

        # Assign final outcomes
        final_result = game.result()
        outcome = 0.0
        if final_result == Result.WHITE_WIN:
            outcome = 1.0
        elif final_result == Result.BLACK_WIN:
            outcome = -1.0

        for ex in examples:
            ex.value_target = outcome * ex.player_sign

        worker_logger.info(f"Game complete: {move_count} moves, {len(examples)} examples")
        return examples

    except Exception as e:
        worker_logger.error(f"Worker failed: {e}", exc_info=True)
        return []

    finally:
        # AGGRESSIVE cleanup - CRITICAL to prevent memory leaks between tasks
        state_cache.clear()

        # Clear game caches
        if 'game' in locals():
            if hasattr(game.state, 'cache_manager'):
                game.state.cache_manager.occupancy_cache.clear()
                game.state.cache_manager.move_cache.clear()

        # GPU memory cleanup
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

        # Force garbage collection
        for _ in range(3):
            gc.collect()

def generate_training_data_parallel(
    model,
    num_games: int = 10,
    device: str = "cuda",
    opponent_types: Optional[List[str]] = None,
    epsilon: float = 0.1,
    num_parallel: int = 4,  # Optimized for 6-core Ryzen 5600X (4 workers * 3 threads = 12)
    max_moves: int = 100_000,
):
    # Save model ONCE
    model_path = None
    if hasattr(model, 'state_dict'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            torch.save(model.state_dict(), tmp.name)
            model_path = tmp.name

    all_examples = []
    pool = None

    try:
        # Create pool with model initialization
        pool = mp.Pool(
            processes=num_parallel,
            initializer=_worker_init,
            initargs=(model_path, device),
            maxtasksperchild=5  # Optimized: faster cleanup, prevents memory accumulation
        )

        # Process games
        worker_args = [
            (f"{i}", None, device, opponent_types, epsilon, max_moves)  # None for model_path
            for i in range(num_games)
        ]

        for result in pool.imap_unordered(_game_worker_batch, worker_args):
            if isinstance(result, list) and result:
                all_examples.extend(result)

    except Exception as e:
        logger.critical(f"Pool execution failed: {e}")
        raise

    finally:
        if pool is not None:
            # GENTLE shutdown: let workers finish cleanup
            pool.close()  # No new tasks
            pool.join()  # Wait up to 30s for cleanup

        # Cleanup temp file
        if model_path and os.path.exists(model_path):
            try:
                os.unlink(model_path)
            except:
                pass

        # Final GPU cleanup from main process
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    return all_examples

def generate_training_data(**kwargs):
    """Alias for generate_training_data_parallel."""
    return generate_training_data_parallel(**kwargs)
