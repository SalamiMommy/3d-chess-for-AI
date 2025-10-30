# parallel_self_play.py - TRUE parallel self-play with threading
"""
Parallel self-play with GPU batching and multi-threaded CPU game engines.
Uses thread pool for game logic and synchronized GPU batching.
"""
import torch
import numpy as np
from typing import List, Optional, Dict, Tuple
import random
from collections import defaultdict
import threading
import queue
import time
from dataclasses import dataclass

from game3d.game.gamestate import GameState
from game3d.common.enums import Color, Result
from game3d.movement.movepiece import Move
from game3d.game3d import OptimizedGame3D
from training.types import TrainingExample
from game3d.common.coord_utils import coord_to_idx, Coord
from game3d.common.constants import VOLUME
from game3d.board.board import Board
from game3d.game.terminal import (
    is_fifty_move_draw,
    is_threefold_repetition,
)
from training.opponents import create_opponent, OpponentBase
from game3d.game.factory import start_game_state


@dataclass
class InferenceRequest:
    """Request for GPU inference."""
    game_id: int
    state_array: np.ndarray
    response_queue: queue.Queue


@dataclass
class InferenceResponse:
    """Response from GPU inference."""
    game_id: int
    from_logits: torch.Tensor
    to_logits: torch.Tensor
    value_pred: torch.Tensor


class GPUInferenceServer:
    """Dedicated thread for batched GPU inference."""

    def __init__(self, model: torch.nn.Module, device: str, batch_size: int):
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.request_queue = queue.Queue(maxsize=batch_size * 2)
        self.running = False
        self.thread = None
        self.total_inferences = 0
        self.total_batches = 0

    def start(self):
        """Start the inference server thread."""
        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the inference server."""
        self.running = False
        if self.thread:
            self.thread.join()

    def request_inference(self, game_id: int, state_array: np.ndarray, response_queue: queue.Queue):
        """Submit inference request (called from game threads)."""
        request = InferenceRequest(game_id, state_array, response_queue)
        self.request_queue.put(request)

    def _inference_loop(self):
        """Main inference loop - processes batches."""
        while self.running:
            # Collect batch
            batch_requests = []
            timeout = 0.01  # 10ms timeout for responsive shutdown

            try:
                # Get first request (blocking with timeout)
                first_request = self.request_queue.get(timeout=timeout)
                batch_requests.append(first_request)

                # Collect more requests up to batch_size (non-blocking)
                while len(batch_requests) < self.batch_size:
                    try:
                        request = self.request_queue.get_nowait()
                        batch_requests.append(request)
                    except queue.Empty:
                        break

            except queue.Empty:
                continue

            if not batch_requests:
                continue

            # Process batch on GPU
            try:
                states = np.stack([req.state_array for req in batch_requests], axis=0)
                states_tensor = torch.from_numpy(states).float().to(self.device)

                with torch.no_grad():
                    from_logits, to_logits, value_pred = self.model(states_tensor)

                # Send responses back to game threads
                for i, request in enumerate(batch_requests):
                    response = InferenceResponse(
                        game_id=request.game_id,
                        from_logits=from_logits[i],
                        to_logits=to_logits[i],
                        value_pred=value_pred[i],
                    )
                    request.response_queue.put(response)

                self.total_inferences += len(batch_requests)
                self.total_batches += 1

            except Exception as e:
                print(f"[GPU SERVER] Inference error: {e}")
                # Send error responses
                for request in batch_requests:
                    request.response_queue.put(None)


class GameWorker:
    """Worker thread that plays a single game."""

    def __init__(
        self,
        game_id: int,
        inference_server: GPUInferenceServer,
        opponent_types: List[str],
        epsilon: float,
        max_moves: int,
    ):
        self.game_id = game_id
        self.inference_server = inference_server
        self.opponent_types = opponent_types
        self.epsilon = epsilon
        self.max_moves = max_moves
        self.response_queue = queue.Queue(maxsize=1)
        self.examples = []
        self.move_count = 0
        self.finished = False

    def run(self) -> List[TrainingExample]:
        """Run the game to completion."""
        # Initialize game
        initial_state = start_game_state()
        game = OptimizedGame3D(board=initial_state.board, cache=initial_state.cache_manager)

        opponents = {
            Color.WHITE: create_opponent(self.opponent_types[0], Color.WHITE),
            Color.BLACK: create_opponent(self.opponent_types[1], Color.BLACK),
        }

        # Initialize position tracking
        if not hasattr(game.state, '_position_counts'):
            game.state._position_counts = defaultdict(int)
        game.state._position_counts[game.state.zkey] = 1

        error_count = 0

        while self.move_count < self.max_moves and not game.is_game_over():
            try:
                # Check termination
                if is_fifty_move_draw(game.state) or is_threefold_repetition(game.state):
                    break

                # Get legal moves
                legal_moves = game.state.legal_moves()
                if not legal_moves:
                    break

                # Request GPU inference
                state_array = game.state.to_array()
                self.inference_server.request_inference(
                    self.game_id, state_array, self.response_queue
                )

                # Wait for response
                response = self.response_queue.get()
                if response is None:
                    error_count += 1
                    if error_count >= 3:
                        break
                    continue

                # Process response
                from_logits = response.from_logits
                to_logits = response.to_logits
                value_pred = response.value_pred

                # Create training example
                example = self._create_training_example(
                    game, from_logits, to_logits, value_pred, legal_moves
                )
                self.examples.append(example)

                # Choose and apply move
                chosen_move = self._choose_move(
                    game, opponents, from_logits, to_logits, legal_moves
                )

                if chosen_move is None:
                    break

                # Update opponent
                opponent = opponents[game.state.color]
                opponent.observe(game.state, chosen_move)

                # Apply move
                receipt = game.submit_move(chosen_move)
                if not receipt.is_legal:
                    error_count += 1
                    if error_count >= 3:
                        break
                    continue

                # Update tracking
                game.state._position_counts[game.state.zkey] += 1
                self.move_count += 1
                error_count = 0

                if self.move_count % 100 == 0:
                    print(f"[GAME {self.game_id}] Move {self.move_count}")

            except Exception as e:
                print(f"[GAME {self.game_id}] Error at move {self.move_count}: {e}")
                error_count += 1
                if error_count >= 3:
                    break

        # Assign outcomes
        self._assign_outcomes(game)
        self.finished = True

        print(f"[GAME {self.game_id}] Completed: {len(self.examples)} examples, {self.move_count} moves")
        return self.examples

    def _create_training_example(
        self,
        game: OptimizedGame3D,
        from_logits: torch.Tensor,
        to_logits: torch.Tensor,
        value_pred: torch.Tensor,
        legal_moves: List[Move],
    ) -> TrainingExample:
        """Create training example from current state."""
        from_logits_cpu = from_logits.cpu()
        to_logits_cpu = to_logits.cpu()

        # PRE-COMPUTE all indices at once (vectorized)
        from_indices = np.array([coord_to_idx(mv.from_coord) for mv in legal_moves if mv], dtype=np.int32)
        to_indices = np.array([coord_to_idx(mv.to_coord) for mv in legal_moves if mv], dtype=np.int32)

        # Vectorized logit computation
        from_logits_np = from_logits_cpu.numpy()
        to_logits_np = to_logits_cpu.numpy()

        move_probs = from_logits_np[from_indices] + to_logits_np[to_indices]

        # Softmax (vectorized)
        max_logit = np.max(move_probs)
        exp_logit = np.exp(move_probs - max_logit)
        sum_exp = np.sum(exp_logit)

        if sum_exp == 0 or np.isnan(sum_exp):
            move_probs_soft = np.ones_like(move_probs) / len(move_probs)
        else:
            move_probs_soft = exp_logit / sum_exp

        # Vectorized target creation
        from_target_np = np.zeros(729, dtype=np.float32)
        to_target_np = np.zeros(729, dtype=np.float32)

        # Use np.add.at for fast accumulation
        np.add.at(from_target_np, from_indices, move_probs_soft)
        np.add.at(to_target_np, to_indices, move_probs_soft)

        # Normalize
        from_sum = np.sum(from_target_np)
        to_sum = np.sum(to_target_np)
        if from_sum > 0:
            from_target_np /= from_sum
        if to_sum > 0:
            to_target_np /= to_sum

        player_sign = 1.0 if game.state.color == Color.WHITE else -1.0
        state_array = game.state.to_array()

        return TrainingExample(
            state_tensor=state_array,
            from_target=from_target_np,
            to_target=to_target_np,
            value_target=value_pred.item(),
            move_count=self.move_count,
            player_sign=player_sign
        )

    def _choose_move(
        self,
        game: OptimizedGame3D,
        opponents: Dict[Color, OpponentBase],
        from_logits: torch.Tensor,
        to_logits: torch.Tensor,
        legal_moves: List[Move],
    ) -> Optional[Move]:
        """Choose move using opponent logic - OPTIMIZED."""
        if random.random() < self.epsilon:
            return random.choice(legal_moves) if legal_moves else None

        if not legal_moves:
            return None

        # Vectorize everything
        from_indices = np.array([coord_to_idx(mv.from_coord) for mv in legal_moves], dtype=np.int32)
        to_indices = np.array([coord_to_idx(mv.to_coord) for mv in legal_moves], dtype=np.int32)

        from_logits_np = from_logits.cpu().numpy()
        to_logits_np = to_logits.cpu().numpy()

        move_probs = from_logits_np[from_indices] + to_logits_np[to_indices]

        # Softmax
        max_logit = np.max(move_probs)
        exp_logit = np.exp(move_probs - max_logit)
        sum_exp = np.sum(exp_logit)

        if sum_exp == 0 or np.isnan(sum_exp):
            move_probs_soft = np.ones_like(move_probs) / len(move_probs)
        else:
            move_probs_soft = exp_logit / sum_exp

        # Vectorized rewards (if opponent supports it)
        color = game.state.color
        opponent = opponents[color]

        # Try batch rewards if available
        if hasattr(opponent, 'batch_reward'):
            rewards_np = opponent.batch_reward(game.state, legal_moves)
        else:
            rewards_np = np.array([opponent.reward(game.state, mv) for mv in legal_moves])

        alpha = 0.5
        scores = move_probs_soft + alpha * rewards_np

        best_idx = int(np.argmax(scores))
        return legal_moves[best_idx]

    def _assign_outcomes(self, game: OptimizedGame3D):
        """Assign final outcomes to examples."""
        if game.is_game_over():
            result = game.result()
            if result == Result.WHITE_WON:
                final_outcome = 1.0
            elif result == Result.BLACK_WON:
                final_outcome = -1.0
            else:
                final_outcome = 0.0
        else:
            final_outcome = 0.0

        for ex in self.examples:
            ex.value_target = final_outcome * ex.player_sign


class ParallelSelfPlayGenerator:
    """Generate training data with true parallel execution."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        num_parallel_games: int = 8,
        temperature: float = 1.5,
        opponent_types: Optional[List[str]] = None,
        epsilon: float = 0.1,
        batch_size: int = 8,
    ):
        self.device = device
        self.num_parallel_games = num_parallel_games
        self.temperature = temperature
        self.epsilon = epsilon

        # Opponent setup
        self.opponent_types = opponent_types or ["adversarial", "center_control"]
        if len(self.opponent_types) == 1:
            self.opponent_types = [self.opponent_types[0], self.opponent_types[0]]
        else:
            self.opponent_types = self.opponent_types[:2]

        # Start GPU inference server
        self.inference_server = GPUInferenceServer(model, device, batch_size)
        self.inference_server.start()

        print(f"[PARALLEL] Initialized with {num_parallel_games} parallel games")
        print(f"[GPU SERVER] Batch size: {batch_size}")
        print(f"[DEVICE] AI model on {device}, game logic on CPU threads")

    def generate_games_parallel(self, max_moves: int = 100_000) -> List[TrainingExample]:
        """Generate multiple games in parallel with true threading."""
        workers = [
            GameWorker(
                game_id=i,
                inference_server=self.inference_server,
                opponent_types=self.opponent_types,
                epsilon=self.epsilon,
                max_moves=max_moves,
            )
            for i in range(self.num_parallel_games)
        ]

        print(f"\n[PARALLEL] Starting {self.num_parallel_games} game threads")
        start_time = time.time()

        # Create threads
        threads = []
        results_queue = queue.Queue()

        def worker_wrapper(worker):
            try:
                examples = worker.run()
                results_queue.put(examples)
            except Exception as e:
                print(f"[GAME {worker.game_id}] Fatal error: {e}")
                import traceback
                traceback.print_exc()
                results_queue.put([])

        # Start all threads
        for worker in workers:
            thread = threading.Thread(target=worker_wrapper, args=(worker,))
            thread.start()
            threads.append(thread)

        # Wait for all threads
        for thread in threads:
            thread.join()

        elapsed = time.time() - start_time

        # Collect results
        all_examples = []
        for _ in range(self.num_parallel_games):
            examples = results_queue.get()
            all_examples.extend(examples)

        print(f"\n[PARALLEL] All games completed in {elapsed:.2f}s")
        print(f"[PARALLEL] Total examples: {len(all_examples)}")
        print(f"[PARALLEL] Games per second: {self.num_parallel_games / elapsed:.2f}")
        print(f"[GPU SERVER] Total batches: {self.inference_server.total_batches}")
        print(f"[GPU SERVER] Total inferences: {self.inference_server.total_inferences}")
        print(f"[GPU SERVER] Avg batch size: {self.inference_server.total_inferences / max(1, self.inference_server.total_batches):.2f}")

        return all_examples

    def __del__(self):
        """Clean up inference server."""
        if hasattr(self, 'inference_server'):
            self.inference_server.stop()


def generate_training_data_parallel(
    model: torch.nn.Module,
    num_games: int = 10,
    max_moves: int = 100_000,
    device: str = "cuda",
    opponent_types: Optional[List[str]] = None,
    epsilon: float = 0.1,
    num_parallel: int = 8,
) -> List[TrainingExample]:
    """Generate training data with parallel game execution."""

    all_examples = []
    rounds = (num_games + num_parallel - 1) // num_parallel

    for round_idx in range(rounds):
        print(f"\n{'='*80}")
        print(f"Parallel round {round_idx + 1}/{rounds}")
        print(f"{'='*80}")

        games_this_round = min(num_parallel, num_games - round_idx * num_parallel)

        generator = ParallelSelfPlayGenerator(
            model=model,
            device=device,
            num_parallel_games=games_this_round,
            opponent_types=opponent_types,
            epsilon=epsilon,
            batch_size=games_this_round,
        )

        examples = generator.generate_games_parallel(max_moves)
        all_examples.extend(examples)

        # Clean up
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Round {round_idx + 1} completed: {len(examples)} examples from {games_this_round} games")

    print(f"\n{'='*80}")
    print(f"Total examples generated: {len(all_examples)}")
    print(f"{'='*80}")

    return all_examples
