"""Simplified parallel self-play with per-worker models."""
import torch
import numpy as np
from typing import List, Optional
import random
import multiprocessing as mp
import logging
import sys
import time
import gc

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _game_worker_permodel(args):
    """
    Worker that runs game logic with its own model instance.
    Each worker loads the model independently - no IPC needed.
    """
    import torch
    import numpy as np
    import random
    import time
    import gc
    
    # Local imports
    from training.training_types import TrainingExample
    from game3d.common.coord_utils import coord_to_idx
    from game3d.common.shared_types import (
        POLICY_DIM, N_PIECE_TYPES, Color, Result
    )
    from game3d.movement.movepiece import Move
    from game3d.game.factory import start_game_state
    from game3d.game3d import OptimizedGame3D
    from training.opponents import create_opponent, OpponentBase
    from game3d.game.terminal import get_draw_reason, result as get_result

    (
        game_id, 
        model_checkpoint_path, 
        device, 
        opponent_types, 
        epsilon, 
        max_moves,
        model_size
    ) = args
    
    worker_logger = logging.getLogger(f"worker.{game_id}")
    
    try:
        worker_logger.info(f"Worker {game_id} starting, loading model...")
        
        # Import the base model class (without compilation)
        from models.graph_transformer import GraphTransformer3D
        
        # Model configuration based on size
        configs = {
            "small": {
                "dim": 384,
                "depth": 8,
                "heads": 6,
                "dim_head": 64,
                "ff_mult": 4,
                "dropout": 0.1,
                "use_gradient_checkpointing": True
            },
            "default": {
                "dim": 512,
                "depth": 12,
                "heads": 8,
                "dim_head": 64,
                "ff_mult": 4,
                "dropout": 0.1,
                "use_gradient_checkpointing": True
            },
            "large": {
                "dim": 896,
                "depth": 20,
                "heads": 14,
                "dim_head": 64,
                "ff_mult": 4,
                "dropout": 0.1,
                "use_gradient_checkpointing": True
            },
            "huge": {
                "dim": 1024,
                "depth": 24,
                "heads": 16,
                "dim_head": 64,
                "ff_mult": 4,
                "dropout": 0.1,
                "use_gradient_checkpointing": True
            }
        }
        
        if model_size not in configs:
            raise ValueError(f"Unknown model size: {model_size}")
        
        config = configs[model_size]
        
        # Create base model WITHOUT torch.compile
        model = GraphTransformer3D(**config)
        model = model.to(device)
        
        # Load weights from checkpoint
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle torch.compile wrapper if present in saved checkpoint
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        # Load the state dict into the uncompiled model
        model.load_state_dict(state_dict)
        
        # NOW compile the model after weights are loaded
        if device == 'cuda' and hasattr(torch, 'compile'):
            try:
                model = torch.compile(
                    model,
                    mode='default',
                    fullgraph=False,
                    dynamic=False,
                )
                worker_logger.info(f"Worker {game_id} model compiled with torch.compile")
            except Exception as e:
                worker_logger.warning(f"Could not compile model: {e}")
        
        model.eval()
        
        worker_logger.info(f"Worker {game_id} model loaded successfully")

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

            # Get legal moves
            legal_moves = game.state.legal_moves
            if legal_moves.size == 0:
                break

            # Model inference - Direct call, no queues!
            state_array = game.state.board.array()
            state_tensor = torch.from_numpy(state_array).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                if device == 'cuda':
                    with torch.amp.autocast('cuda'):
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
            
            # Filter valid moves (piece belongs to current player)
            valid_mask = from_colors == game.state.color
            if not np.any(valid_mask):
                worker_logger.warning("No valid moves for current player")
                break

            valid_moves = legal_moves[valid_mask]
            n_valid = len(valid_moves)

            # Create policy targets
            from_indices = coord_to_idx(valid_moves[:, :3])
            to_indices = coord_to_idx(valid_moves[:, 3:6])

            from_target = np.zeros(POLICY_DIM, dtype=np.float32)
            to_target = np.zeros(POLICY_DIM, dtype=np.float32)

            from_target[from_indices] = from_probs[from_indices]
            to_target[to_indices] = to_probs[to_indices]

            # Normalize
            from_sum = from_target.sum()
            if from_sum > 0: from_target /= from_sum
            
            to_sum = to_target.sum()
            if to_sum > 0: to_target /= to_sum

            # Create example
            player_sign = 1.0 if game.state.color == Color.WHITE.value else -1.0
            
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
            game._state._legal_moves_cache = None
            move_count += 1

            # Observe move
            opponent = opponents[game.state.color]
            if isinstance(opponent, OpponentBase):
                opponent.observe(game.state, submit_move)

        # Validation and Stats
        valid_examples = []
        for ex in examples:
            try:
                if ex.validate():
                    valid_examples.append(ex)
            except Exception:
                continue

        # Stats logging
        duration = time.perf_counter() - start_time
        game_result = get_result(game.state)
        
        result_str = "UNKNOWN"
        if game_result == Result.WHITE_WIN: result_str = "WHITE WIN"
        elif game_result == Result.BLACK_WIN: result_str = "BLACK WIN"
        elif game_result == Result.DRAW: result_str = f"DRAW ({get_draw_reason(game.state)})"
            
        # Get material counts from occupancy cache
        occ_cache = game.state.cache_manager.occupancy_cache
        coords, piece_types, colors = occ_cache.get_all_occupied_vectorized()
        white_mat = np.sum(colors == Color.WHITE)
        black_mat = np.sum(colors == Color.BLACK)
        
        # Check priest status
        white_priests = "Y" if occ_cache.has_priest(Color.WHITE) else "N"
        black_priests = "Y" if occ_cache.has_priest(Color.BLACK) else "N"
        
        worker_logger.info(
            f"GAME FINISHED: {game_id} | {result_str} | {move_count} moves | "
            f"{duration:.1f}s | Mat: W={white_mat}/B={black_mat} | Priests: W={white_priests} B={black_priests}"
        )

        # Assign outcomes
        final_result = game.result()
        outcome = 0.0
        if final_result == Result.WHITE_WIN: outcome = 1.0
        elif final_result == Result.BLACK_WIN: outcome = -1.0

        for ex in valid_examples:
            ex.value_target = outcome * ex.player_sign

        return valid_examples

    except Exception as e:
        worker_logger.error(f"Worker failed: {e}", exc_info=True)
        return []

    finally:
        # Cleanup
        if 'game' in locals() and hasattr(game.state, 'cache_manager'):
            game.state.cache_manager.occupancy_cache.clear()
            game.state.cache_manager.move_cache.clear()
        if 'model' in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_training_data_parallel(
    model_checkpoint_path: str,
    num_games: int = 10,
    device: str = "cuda",
    opponent_types: Optional[List[str]] = None,
    epsilon: float = 0.1,
    num_parallel: int = 4,
    max_moves: int = 100_000,
    model_size: str = "default",
):
    """
    Generate training data using parallel self-play with per-worker models.
    
    Args:
        model_checkpoint_path: Path to model checkpoint file
        num_games: Number of games to play
        device: Device to use for inference ('cuda' or 'cpu')
        opponent_types: List of [white_opponent, black_opponent] types
        epsilon: Exploration rate for move selection
        num_parallel: Number of parallel workers
        max_moves: Maximum moves per game
        model_size: Model size string for loading ('small', 'default', 'large', 'huge')
    
    Returns:
        List of training examples from all games
    """
    all_examples = []
    
    # Default opponent types
    if opponent_types is None:
        opponent_types = ["random", "random"]
    
    logger.info(f"Starting parallel self-play: {num_games} games, {num_parallel} workers")
    logger.info(f"Model checkpoint: {model_checkpoint_path}")
    logger.info(f"Device: {device}, Model size: {model_size}")

    pool = None
    try:
        # Create worker pool
        pool = mp.Pool(processes=num_parallel)

        worker_args = []
        for i in range(num_games):
            gid = f"{i}"
            args = (
                gid, 
                model_checkpoint_path, 
                device, 
                opponent_types, 
                epsilon, 
                max_moves,
                model_size
            )
            worker_args.append(args)

        # Collect results
        logger.info(f"Launching {num_games} game workers...")
        for result in pool.imap_unordered(_game_worker_permodel, worker_args):
            if isinstance(result, list) and result:
                all_examples.extend(result)
                logger.info(f"Game completed: {len(result)} examples collected")

    except Exception as e:
        logger.critical(f"Pool execution failed: {e}", exc_info=True)
        raise

    finally:
        # Cleanup
        if pool is not None:
            pool.close()
            pool.join()
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"Self-play complete: {len(all_examples)} total examples from {num_games} games")
    return all_examples


def generate_training_data(**kwargs):
    """Alias for generate_training_data_parallel."""
    return generate_training_data_parallel(**kwargs)
