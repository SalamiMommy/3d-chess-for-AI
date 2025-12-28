"""
Client-Server Parallel Self-Play Architecture for 3D Chess.

Solves VRAM exhaustion by centralizing the model in one process (InferenceServer)
and having lightweight GameWorkers communicate via queues.
"""
import torch
import numpy as np
from typing import List, Optional, Tuple, Dict
import random
import multiprocessing as mp
from queue import Empty
import logging
import time
import gc
from tqdm import tqdm
from training.training_types import TrainingExample

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    import setproctitle
except ImportError:
    setproctitle = None


# =============================================================================
# INFERENCE SERVER
# =============================================================================

def _inference_server_loop(
    model_checkpoint_path: str,
    device: str,
    model_size: str,
    request_queue: mp.Queue,
    response_queues: Dict[int, mp.Queue],
    stop_event: mp.Event,
    batch_size: int
):
    """
    Main loop for the inference server.
    Loads model ONCE and processes batches of requests from workers.
    """
    # Import locally to avoid pollution
    from models.graph_transformer import create_optimized_model
    import torch

    server_logger = logging.getLogger("InferenceServer")
    if setproctitle:
        setproctitle.setproctitle("3dchess: server")
    server_logger.info(f"Starting Server on {device} (Batch Size: {batch_size})")


    try:
        # 1. Load Model (Single Copy in VRAM)
        # create_optimized_model usually applies torch.compile inside it!
        model = create_optimized_model(model_size)
        model = model.to(device)

        # Load weights
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Clean up state_dict keys (remove _orig_mod prefix from checkpoint if present)
        # This ensures we have "clean" keys: "layers.0..." instead of "_orig_mod.layers.0..."
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # FIX: Load weights into the correct object
        # If create_optimized_model compiled the model, 'model' is an OptimizedModule wrapper.
        # It expects keys with "_orig_mod.", but we have clean keys.
        # Solution: Load into the underlying uncompiled model.
        if hasattr(model, '_orig_mod'):
            server_logger.info("Detected compiled model (OptimizedModule). Loading weights into underlying _orig_mod.")
            model._orig_mod.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

        model.eval()

        # Optimization: Apply compilation ONLY if not already compiled
        if not hasattr(model, '_orig_mod') and hasattr(torch, 'compile'):
            try:
                # Use 'reduce-overhead' for faster dispatch in loop
                server_logger.info("Compiling model with mode='reduce-overhead'...")
                model = torch.compile(model, mode='reduce-overhead')
            except Exception as e:
                server_logger.warning(f"Compilation failed, running eager: {e}")
        else:
            server_logger.info("Model already compiled (or compilation unavailable), skipping re-compile.")

        # 2. Setup Memory Context
        if device == 'cuda':
            # Reserve appropriate memory fraction for this single process
            # Leaving 10% for system overhead
            torch.cuda.set_per_process_memory_fraction(0.9)
            torch.cuda.empty_cache()

        # Use Inference Mode for maximum efficiency (better than no_grad)
        inference_ctx = torch.inference_mode()

        server_logger.info("Server Ready. Listening...")

        # 3. Main Loop
        first_batch = True
        with inference_ctx:
            while not stop_event.is_set():
                requests = []

                # A. Dynamic Batching
                try:
                    # Blocking wait for first item (efficiency: sleeps CPU when idle)
                    # Timeout allows checking stop_event periodically
                    first_req = request_queue.get(timeout=1.0)
                    requests.append(first_req)

                    # Opportunistic fetch for rest of batch
                    while len(requests) < batch_size:
                        try:
                            req = request_queue.get_nowait()
                            requests.append(req)
                        except Empty:
                            break # No more immediate requests

                except Empty:
                    continue # Loop back to check stop_event

                if not requests:
                    continue

                # B. Batch Processing
                # Unpack requests: [(worker_id, state_numpy), ...]
                worker_ids = [r[0] for r in requests]
                states_np = np.stack([r[1] for r in requests])

                # Tensor conversion
                states = torch.from_numpy(states_np).float().to(device, non_blocking=True)

                # Inference
                if first_batch:
                    server_logger.info("Processing first batch (this may trigger compilation)...")

                if device == 'cuda':
                    with torch.amp.autocast('cuda'):
                        from_logits, to_logits, values = model(states)
                else:
                    from_logits, to_logits, values = model(states)
                
                if first_batch:
                    server_logger.info("First batch processed.")
                    first_batch = False

                # Move results to CPU
                # Use non_blocking=False to ensure synchronization before sending
                from_probs_batch = torch.softmax(from_logits, dim=-1).cpu().numpy()
                to_probs_batch = torch.softmax(to_logits, dim=-1).cpu().numpy()
                values_batch = values.cpu().numpy()

                # C. Distribution
                for i, w_id in enumerate(worker_ids):
                    response = (
                        from_probs_batch[i],
                        to_probs_batch[i],
                        float(values_batch[i, 0])
                    )
                    try:
                        response_queues[w_id].put(response)
                    except Exception as e:
                        server_logger.error(f"Failed to send to worker {w_id}: {e}")

    except Exception as e:
        server_logger.critical(f"Server Crashed: {e}", exc_info=True)
    finally:
        server_logger.info("Server shutting down.")


# =============================================================================
# GAME WORKER (LIGHTWEIGHT)
# =============================================================================

def _game_worker_client(args):
    """
    Lightweight worker. Does NOT load model.
    Sends states to Server, receives predictions.
    """
    import numpy as np
    import random
    import time

    # Local imports to avoid pickling issues
    from training.training_types import TrainingExample
    from game3d.common.coord_utils import coord_to_idx
    from game3d.common.shared_types import POLICY_DIM, Color, Result, COORD_DTYPE
    from game3d.movement.movepiece import Move
    from game3d.main_game import OptimizedGame3D, InvalidMoveError
    from training.opponents import create_opponent, OpponentBase
    from game3d.game.terminal import get_draw_reason, result as get_result
    from game3d.game.factory import start_game_state

    (
        game_id,
        worker_id,
        opponent_types,
        epsilon,
        max_moves,
        request_queue,   # Shared Queue -> Server
        response_queue   # Private Queue <- Server
    ) = args

    worker_logger = logging.getLogger(f"worker.{worker_id}")
    if setproctitle:
        setproctitle.setproctitle(f"3dchess: worker {worker_id}")


    try:
        # Initialize game (Low memory footprint)
        initial_state = start_game_state()
        game = OptimizedGame3D(board=initial_state.board, cache=initial_state.cache_manager)

        opponents = {
            Color.WHITE: create_opponent(opponent_types[0], Color.WHITE),
            Color.BLACK: create_opponent(opponent_types[1], Color.BLACK),
        }

        move_count = 0
        examples = []
        error_count = 0
        
        # --- GAME LOOP ---
        while move_count < max_moves and not game.is_game_over():

            legal_moves = game.state.legal_moves
            if legal_moves.size == 0:
                break

            # 1. REQUEST PREDICTION
            state_array = game.state.board.array()

            # Send to server
            request_queue.put((worker_id, state_array))

            # Wait for response (Blocking)
            try:
                # 600s timeout to allow for compilation on first batch
                from_probs, to_probs, value_pred_scalar = response_queue.get(timeout=600.0)
            except Empty:
                raise RuntimeError("Server response timeout - server likely crashed or is overloaded")

            # 2. PROCESS PREDICTION
            from_coords = legal_moves[:, :3].astype(COORD_DTYPE)
            to_coords = legal_moves[:, 3:6].astype(COORD_DTYPE)

            # Get occupancy data
            occ_cache = game.state.cache_manager.occupancy_cache
            from_colors, _ = occ_cache.batch_get_attributes(from_coords)

            # Filter valid moves
            valid_mask = from_colors == game.state.color
            if not np.any(valid_mask):
                break

            valid_moves = legal_moves[valid_mask]
            n_valid = len(valid_moves)

            # Create policy targets
            valid_from_coords = valid_moves[:, :3]
            valid_to_coords = valid_moves[:, 3:6]
            from_indices = coord_to_idx(valid_from_coords)
            to_indices = coord_to_idx(valid_to_coords)

            from_target = np.zeros(POLICY_DIM, dtype=np.float32)
            to_target = np.zeros(POLICY_DIM, dtype=np.float32)

            # Extract probabilities for valid moves
            from_target[from_indices] = from_probs[from_indices]
            to_target[to_indices] = to_probs[to_indices]

            # Normalize
            from_sum = from_target.sum()
            if from_sum > 0: from_target /= from_sum
            to_sum = to_target.sum()
            if to_sum > 0: to_target /= to_sum

            # Create Example
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

            # 3. SELECT MOVE
            move_scores = from_probs[from_indices] + to_probs[to_indices]

            if random.random() < epsilon:
                chosen_idx = random.randint(0, n_valid - 1)
            else:
                opponent = opponents[game.state.color]
                if isinstance(opponent, OpponentBase):
                    rewards = opponent.batch_reward(game.state, valid_moves)
                    move_scores += 0.5 * rewards

                chosen_idx = int(np.argmax(move_scores))

            chosen_move = valid_moves[chosen_idx]
            submit_move = Move(
                chosen_move[:3].astype(COORD_DTYPE),
                chosen_move[3:6].astype(COORD_DTYPE)
            )

            try:
                receipt = game.submit_move(submit_move)
            except (InvalidMoveError, ValueError) as e:
                worker_logger.warning(f"Worker {worker_id}: Caught invalid move error: {e}. Invalidating cache and retrying.")
                # Force cache invalidation to clear stale moves
                game.state.cache_manager.move_cache.invalidate()
                game.state._legal_moves_cache = None
                
                error_count += 1
                if error_count >= 3: break
                continue

            if not receipt.is_legal:
                error_count += 1
                if error_count >= 3: break
                continue

            game._state = receipt.new_state
            game._state._legal_moves_cache = None
            move_count += 1

            # FIX: Check if game is over after move execution
            if receipt.is_game_over:
                worker_logger.debug(f"Worker {worker_id}: Game over detected via receipt (result={receipt.result})")
                break

            # Observe
            opponent = opponents[game.state.color]
            if isinstance(opponent, OpponentBase):
                opponent.observe(game.state, submit_move)

        # --- GAME END ---
        valid_examples = []
        for ex in examples:
            if ex.validate(): valid_examples.append(ex)

        game_result = get_result(game.state)
        final_result = game.result()

        # Determine Outcome
        outcome = 0.0
        if final_result == Result.WHITE_WIN: outcome = 1.0
        elif final_result == Result.BLACK_WIN: outcome = -1.0
        else: outcome = -0.05 # Draw penalty

        for ex in valid_examples:
            if final_result in [Result.WHITE_WIN, Result.BLACK_WIN]:
                ex.value_target = outcome * ex.player_sign
            else:
                ex.value_target = -0.05 # Draw penalty for both

        return valid_examples

    except Exception as e:
        worker_logger.error(f"Worker {worker_id} failed: {e}", exc_info=True)
        return []


def _pool_worker_shim(packed_args):
    """
    Shim to handle Worker ID assignment inside the Pool.
    """
    game_args, id_queue = packed_args
    # Acquire ID (blocks if none available)
    worker_id = id_queue.get()

    try:
        # Unpack game args
        (gid, opps, eps, moves, req_q, resp_qs) = game_args

        # Get specific response queue
        my_resp_q = resp_qs[worker_id]

        # Construct full args for client
        client_args = (
            gid,
            worker_id,
            opps,
            eps,
            moves,
            req_q,
            my_resp_q
        )

        return _game_worker_client(client_args)
    finally:
        # Release ID so another task can run on this process (or another process)
        # Note: In a Pool, processes reuse this shim, so we must release the ID.
        id_queue.put(worker_id)


# =============================================================================
# SELF-PLAY ENGINE (PERSISTENT)
# =============================================================================

class SelfPlayEngine:
    """
    Persistent engine that manages the worker pool and inference server.
    Avoids expensive process spawning every iteration.
    """
    def __init__(self, num_parallel: int = 4, device: str = "cuda"):
        self.num_parallel = num_parallel
        self.device = device
        self.logger = logging.getLogger("SelfPlayEngine")
        
        # 1. Setup Multiprocessing Context
        # Use 'spawn' context for CUDA compatibility
        self.ctx = mp.get_context('spawn')
        self.manager = self.ctx.Manager()
        
        # 2. Setup Queues (Persistent)
        self.request_queue = self.manager.Queue()
        self.response_queues = {i: self.manager.Queue() for i in range(num_parallel)}
        
        # 3. Setup ID Queue (for Pool workers to claim persistent slot IDs)
        self.id_queue = self.manager.Queue()
        for i in range(num_parallel):
            self.id_queue.put(i)
            
        # 4. Setup Worker Pool (Persistent)
        # We start the pool once. It will stay alive waiting for tasks.
        self.logger.info(f"Initializing Worker Pool with {num_parallel} processes...")
        self.pool = self.ctx.Pool(processes=num_parallel)
        
        # 5. Server State
        self.server_process = None
        self.stop_event = self.ctx.Event()

    def update_model(self, model_checkpoint_path: str, model_size: str = "default"):
        """
        Restart the inference server with new model weights.
        """
        self.logger.info(f"Updating model to: {model_checkpoint_path}")
        
        # Stop existing server if any
        if self.server_process and self.server_process.is_alive():
            self.logger.info("Stopping old inference server...")
            self.stop_event.set()
            self.server_process.join(timeout=5)
            if self.server_process.is_alive():
                self.server_process.terminate()
            self._drain_queues()
            
        # Reset stop event for new server
        self.stop_event.clear()
        
        # Start new server
        self.logger.info("Starting new inference server...")
        self.server_process = self.ctx.Process(
            target=_inference_server_loop,
            args=(
                model_checkpoint_path,
                self.device,
                model_size,
                self.request_queue,
                self.response_queues,
                self.stop_event,
                self.num_parallel
            ),
            name="InferenceServer"
        )
        self.server_process.start()
        
        # Wait specifically for startup
        time.sleep(2)
        if not self.server_process.is_alive():
            raise RuntimeError("Inference server failed to start immediately.")

    def generate_games(
        self,
        num_games: int,
        opponent_types: Optional[List[str]] = None,
        epsilon: float = 0.1,
        max_moves: int = 100_000
    ) -> List[TrainingExample]:
        """
        Dispatch game tasks to the persistent pool.
        """
        if opponent_types is None:
            opponent_types = ["piece_capture", "piece_capture"]
            
        self.logger.info(f"Dispatching {num_games} games to pool...")
        
        game_args = []
        for i in range(num_games):
            gid = f"{int(time.time())}_{i}"
            args = (
                gid,
                opponent_types,
                epsilon,
                max_moves,
                self.request_queue,
                self.response_queues
            )
            game_args.append(args)
            
        all_examples = []
        
        # Use imap_unordered to process results as they come in
        try:
            # Note: we pass id_queue separately to the shim
            packed_args = [(a, self.id_queue) for a in game_args]
            
            iterator = self.pool.imap_unordered(_pool_worker_shim, packed_args)
            
            for res in tqdm(iterator, total=num_games, desc="Self-Play Games"):
                if res:
                    all_examples.extend(res)
                    
            return all_examples
            
        except Exception as e:
            self.logger.error(f"Error during game generation: {e}")
            raise

    def _drain_queues(self):
        """Helper to clear queues on server restart."""
        try:
            while not self.request_queue.empty():
                try: self.request_queue.get_nowait()
                except: break
            for q in self.response_queues.values():
                while not q.empty():
                    try: q.get_nowait()
                    except: break
        except:
            pass

    def shutdown(self):
        """
        Full cleanup of pool, server, and manager.
        """
        self.logger.info("Shutting down SelfPlayEngine...")
        
        # 1. Stop Server
        self.stop_event.set()
        if self.server_process:
            self.server_process.join(timeout=5)
            if self.server_process.is_alive():
                self.server_process.terminate()
        
        # 2. Stop Pool
        if self.pool:
            self.pool.terminate()
            self.pool.join()
            
        # 3. Stop Manager
        try:
            self.manager.shutdown()
        except:
            pass
            
        self.logger.info("Shutdown complete.")


def generate_training_data_parallel(
    model_checkpoint_path: str,
    num_games: int,
    device: str,
    model_size: str = "default",
    opponent_types: Optional[List[str]] = None
) -> List[TrainingExample]:
    """
    Convenience wrapper for SelfPlayEngine to generate training data.
    Useful for one-off calls (e.g., from simple training loops).
    """
    # Use fewer processes for simple wrapper calls to avoid system overload
    num_workers = min(4, mp.cpu_count() - 1)
    
    engine = SelfPlayEngine(num_parallel=num_workers, device=device)
    try:
        engine.update_model(model_checkpoint_path, model_size=model_size)
        return engine.generate_games(num_games, opponent_types)
    finally:
        engine.shutdown()
