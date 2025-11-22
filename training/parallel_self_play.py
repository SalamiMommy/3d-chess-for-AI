import torch
import numpy as np
from typing import List, Optional, Dict, Tuple
import random
import multiprocessing as mp
import os
import tempfile
import logging
import sys
import time
import gc
import queue
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

BATCH_SIZE = 32  # Larger batch size for better GPU utilization
TIMEOUT = 0.01   # Short timeout for batch accumulation

@dataclass
class InferenceRequest:
    game_id: str
    state_tensor: np.ndarray  # Numpy array to be pickleable

@dataclass
class InferenceResult:
    from_logits: np.ndarray
    to_logits: np.ndarray
    value_pred: float

class RemoteModel:
    """Mimics the model interface but communicates with the InferenceServer."""
    def __init__(self, input_queue: mp.Queue, output_queues: Dict[str, mp.Queue], game_id: str):
        self.input_queue = input_queue
        self.output_queues = output_queues # Dictionary of queues, but we only need ours
        self.game_id = game_id
        # We need to know WHICH queue is ours. 
        # In this design, we'll pass a dict of queues to the worker, and the worker picks its own.
        
    def __call__(self, state_tensor: torch.Tensor):
        # Convert to numpy for transmission
        state_np = state_tensor.cpu().numpy()
        
        # Send request
        req = InferenceRequest(self.game_id, state_np)
        self.input_queue.put(req)
        
        # Wait for response
        # Note: output_queues is a dict shared via Manager, or we pass just the specific queue?
        # Passing the specific queue is safer/faster if possible.
        # Let's assume self.output_queues is the specific queue for this worker.
        
        try:
            result = self.output_queues.get(timeout=30.0) # Long timeout for safety
            if isinstance(result, Exception):
                raise result
        except queue.Empty:
            raise RuntimeError(f"Timeout waiting for inference result for game {self.game_id}")
            
        # Convert back to tensors (on CPU) to match expected interface
        # The worker code expects tensors on 'device', but since we are lightweight, 
        # we can keep them on CPU or move to device if needed.
        # The original code did: from_probs = torch.softmax(from_logits, dim=-1).cpu().numpy()[0]
        # So it expects logits.
        
        # We return TENSORS because the worker code does torch.softmax on them.
        # But wait, the worker code does:
        # with torch.no_grad():
        #    if device == 'cuda': ...
        #    from_logits, ... = model(state_tensor)
        # from_probs = torch.softmax(from_logits, ...).cpu().numpy()
        
        # We can return CPU tensors.
        return (
            torch.from_numpy(result.from_logits),
            torch.from_numpy(result.to_logits),
            torch.tensor([[result.value_pred]]) # Shape (1, 1)
        )

def _inference_server_loop(model, input_queue: mp.Queue, output_queues: Dict[str, mp.Queue], stop_event: threading.Event, device: str):
    """
    Runs in a THREAD in the main process.
    Aggregates requests, runs batch inference, distributes results.
    """
    logger.info("Inference Server started")
    model.eval()
    
    while not stop_event.is_set():
        batch_requests = []
        start_wait = time.perf_counter()
        
        # 1. Collect batch
        try:
            # Blocking get for first item
            req = input_queue.get(timeout=0.1)
            batch_requests.append(req)
            
            # Non-blocking get for rest up to BATCH_SIZE or TIMEOUT
            while len(batch_requests) < BATCH_SIZE:
                if time.perf_counter() - start_wait > TIMEOUT:
                    break
                try:
                    req = input_queue.get_nowait()
                    batch_requests.append(req)
                except queue.Empty:
                    break
                    
        except queue.Empty:
            continue # Loop again and check stop_event
            
        if not batch_requests:
            continue
            
        # 2. Prepare batch
        try:
            # Stack states
            # state_tensor shape: (1, C, D, H, W) -> stack -> (B, C, D, H, W)
            states = np.concatenate([r.state_tensor for r in batch_requests], axis=0)
            states_tensor = torch.from_numpy(states).float().to(device)
            
            # 3. Run inference
            with torch.no_grad():
                if device == 'cuda':
                    with torch.amp.autocast(device_type=device):
                        from_logits, to_logits, value_pred = model(states_tensor)
                else:
                    from_logits, to_logits, value_pred = model(states_tensor)
            
            # Move to CPU numpy
            from_logits_np = from_logits.float().cpu().numpy()
            to_logits_np = to_logits.float().cpu().numpy()
            value_pred_np = value_pred.float().cpu().numpy()
            
            # 4. Distribute results
            for i, req in enumerate(batch_requests):
                res = InferenceResult(
                    from_logits=from_logits_np[i:i+1], # Keep (1, ...) shape
                    to_logits=to_logits_np[i:i+1],
                    value_pred=float(value_pred_np[i, 0])
                )
                
                if req.game_id in output_queues:
                    output_queues[req.game_id].put(res)
                else:
                    logger.error(f"Output queue not found for game {req.game_id}")
                    
        except Exception as e:
            logger.error(f"Inference server error: {e}", exc_info=True)
            # Send error to all waiting workers
            for req in batch_requests:
                if req.game_id in output_queues:
                    output_queues[req.game_id].put(e)

    logger.info("Inference Server stopped")

def _game_worker_batch(args):
    """
    Worker that runs the game logic.
    Communicates with Inference Server for model evaluations.
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

    game_id, _, device, opponent_types, epsilon, max_moves, input_queue, my_output_queue = args
    
    worker_logger = logging.getLogger(f"worker.{game_id}")
    
    # Setup Remote Model
    # We wrap the queue communication in a callable that looks like the model
    class WorkerRemoteModel:
        def __init__(self, in_q, out_q, gid):
            self.in_q = in_q
            self.out_q = out_q
            self.gid = gid
            
        def __call__(self, state_tensor):
            # state_tensor is (1, ...)
            state_np = state_tensor.cpu().numpy()
            self.in_q.put(InferenceRequest(self.gid, state_np))
            
            res = self.out_q.get()
            if isinstance(res, Exception):
                raise res
                
            return (
                torch.from_numpy(res.from_logits),
                torch.from_numpy(res.to_logits),
                torch.tensor([[res.value_pred]])
            )

    model = WorkerRemoteModel(input_queue, my_output_queue, game_id)

    try:
        worker_logger.info(f"Worker {game_id} starting...")

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

            # Model inference
            state_array = game.state.board.array()
            state_tensor = torch.from_numpy(state_array).float().unsqueeze(0) # CPU tensor
            
            # Call remote model
            # Note: We don't need autocast here, the server handles it
            from_logits, to_logits, value_pred = model(state_tensor)

            from_probs = torch.softmax(from_logits, dim=-1).numpy()[0]
            to_probs = torch.softmax(to_logits, dim=-1).numpy()[0]
            value_pred_scalar = float(value_pred.numpy()[0, 0])

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
            
        board_arr = game.state.board.array()
        white_mat = np.sum(board_arr[:N_PIECE_TYPES])
        black_mat = np.sum(board_arr[N_PIECE_TYPES:])
        
        worker_logger.info(
            f"GAME FINISHED: {game_id} | {result_str} | {move_count} moves | "
            f"{duration:.1f}s | Mat: {white_mat:.0f}/{black_mat:.0f}"
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
        gc.collect()

def generate_training_data_parallel(
    model,
    num_games: int = 10,
    device: str = "cuda",
    opponent_types: Optional[List[str]] = None,
    epsilon: float = 0.1,
    num_parallel: int = 4,
    max_moves: int = 100_000,
):
    all_examples = []
    
    # 1. Setup Queues
    # Use Manager to create queues that can be shared with workers
    # Actually, simple mp.Queue is enough if passed to Process
    manager = mp.Manager()
    input_queue = manager.Queue()
    output_queues = {} # Local dict to hold queues
    
    # We need a way to map game_id to a queue. 
    # We can create a dict of queues, but passing a dict of mp.Queues is tricky.
    # Better: Create a list of queues, one per worker/game.
    # Since we spawn num_games tasks, we might have more games than workers.
    # But we process them in a pool.
    # Wait, if we use pool.imap_unordered, we don't know which worker gets which game ID easily beforehand 
    # if we just use 0..N.
    # But we can create one queue per GAME ID.
    
    # Create queues for all games
    game_queues = {}
    for i in range(num_games):
        gid = f"{i}"
        game_queues[gid] = manager.Queue()
        output_queues[gid] = game_queues[gid]

    # 2. Start Inference Server (Thread)
    stop_event = threading.Event()
    server_thread = threading.Thread(
        target=_inference_server_loop,
        args=(model, input_queue, output_queues, stop_event, device),
        daemon=True
    )
    server_thread.start()

    pool = None
    try:
        # 3. Start Workers
        # We don't need _worker_init anymore because workers don't load models
        pool = mp.Pool(processes=num_parallel)

        worker_args = []
        for i in range(num_games):
            gid = f"{i}"
            # Pass ONLY the specific output queue for this game to avoid pickling the whole dict
            # Actually, passing the whole dict of queues might be heavy if num_games is huge.
            # But for batch training (e.g. 24 games), it's fine.
            # However, mp.Queue objects are not picklable directly if not from Manager?
            # Manager queues are picklable (proxies).
            
            args = (
                gid, None, device, opponent_types, epsilon, max_moves,
                input_queue, game_queues[gid]
            )
            worker_args.append(args)

        # 4. Collect results
        for result in pool.imap_unordered(_game_worker_batch, worker_args):
            if isinstance(result, list) and result:
                all_examples.extend(result)

    except Exception as e:
        logger.critical(f"Pool execution failed: {e}")
        raise

    finally:
        # 5. Cleanup
        stop_event.set()
        if server_thread.is_alive():
            server_thread.join(timeout=2.0)
            
        if pool is not None:
            pool.close()
            pool.join()
            
        # Clear queues
        try:
            while not input_queue.empty(): input_queue.get_nowait()
        except: pass
        
        manager.shutdown()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_examples

def generate_training_data(**kwargs):
    """Alias for generate_training_data_parallel."""
    return generate_training_data_parallel(**kwargs)
