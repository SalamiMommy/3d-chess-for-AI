# parallel_self_play.py - TRUE parallel self-play with multiprocessing
"""
Parallel self-play with GPU batching and multi-process CPU game engines.
Uses process pool for game logic and synchronized GPU batching.
"""
import torch
import numpy as np
from typing import List, Optional, Dict, Tuple
import random
from collections import defaultdict
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
import os
import tempfile
import threading

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
from training.optim_train import TrainingConfig, ChessTrainer


class SharedModelInference:
    """Shared model that handles batched inference for multiple games."""

    def __init__(self, model_path: str, device: str, batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.request_queue = mp.Queue()
        self.result_queues = {}  # game_id -> queue
        self.lock = threading.Lock()

        # Load model once
        config = TrainingConfig(device=self.device)
        self.trainer = ChessTrainer(config)
        if model_path:
            state_dict = torch.load(model_path, map_location=self.device)
            self.trainer.model.load_state_dict(state_dict)
        self.model = self.trainer.model.eval()

        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def _inference_loop(self):
        """Main inference loop running in a separate thread."""
        while self.running:
            try:
                # Collect batch of requests
                batch_requests = []
                start_time = time.time()

                # Get first request with timeout
                try:
                    first_request = self.request_queue.get(timeout=0.1)
                    batch_requests.append(first_request)

                    # Collect more requests quickly
                    while len(batch_requests) < self.batch_size and time.time() - start_time < 0.05:
                        try:
                            request = self.request_queue.get_nowait()
                            batch_requests.append(request)
                        except queue.Empty:
                            break
                except queue.Empty:
                    continue

                if not batch_requests:
                    continue

                # Process batch
                states = np.stack([req['state_array'] for req in batch_requests])
                states_tensor = torch.from_numpy(states).float().to(self.device)

                with torch.no_grad():
                    from_logits, to_logits, value_pred = self.model(states_tensor)

                # Send results back
                for i, req in enumerate(batch_requests):
                    result = {
                        'from_logits': from_logits[i].cpu(),
                        'to_logits': to_logits[i].cpu(),
                        'value_pred': value_pred[i].cpu()
                    }
                    self.result_queues[req['game_id']].put(result)

            except Exception as e:
                print(f"[INFERENCE] Error: {e}")
                # Put None in result queues to indicate error
                for req in batch_requests:
                    self.result_queues[req['game_id']].put(None)

    def request_inference(self, game_id: int, state_array: np.ndarray):
        """Request inference for a game state."""
        with self.lock:
            if game_id not in self.result_queues:
                self.result_queues[game_id] = mp.Queue()

        self.request_queue.put({
            'game_id': game_id,
            'state_array': state_array
        })

        # Wait for result
        result = self.result_queues[game_id].get()
        return result

    def stop(self):
        """Stop the inference thread."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)


def _game_worker(args):
    """Game worker function that runs a single game."""
    game_id, model_path, device, opponent_types, epsilon, max_moves = args

    try:
        # Create shared inference for this process
        inference = SharedModelInference(model_path, device)

        # Initialize game
        initial_state = start_game_state()
        game = OptimizedGame3D(board=initial_state.board, cache=initial_state.cache_manager)

        opponents = {
            Color.WHITE: create_opponent(opponent_types[0], Color.WHITE),
            Color.BLACK: create_opponent(opponent_types[1], Color.BLACK),
        }

        # Initialize position tracking
        if not hasattr(game.state, '_position_counts'):
            game.state._position_counts = defaultdict(int)
        game.state._position_counts[game.state.zkey] = 1

        examples = []
        move_count = 0
        error_count = 0

        while move_count < max_moves and not game.is_game_over():
            try:
                # Check termination
                if is_fifty_move_draw(game.state) or is_threefold_repetition(game.state):
                    break

                # Get legal moves
                legal_moves = game.state.legal_moves()
                if not legal_moves:
                    break

                # Request inference
                state_array = game.state.to_array()
                result = inference.request_inference(game_id, state_array)

                if result is None:
                    error_count += 1
                    if error_count >= 3:
                        break
                    continue

                # Process response
                from_logits = result['from_logits']
                to_logits = result['to_logits']
                value_pred = result['value_pred']

                # Create training example
                example = _create_training_example(
                    game, from_logits, to_logits, value_pred, legal_moves, move_count
                )
                examples.append(example)

                # Choose and apply move
                chosen_move = _choose_move(
                    game, opponents, from_logits, to_logits, legal_moves, epsilon
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
                move_count += 1
                error_count = 0

                if move_count % 100 == 0:
                    print(f"[GAME {game_id}] Move {move_count}")

            except Exception as e:
                print(f"[GAME {game_id}] Error at move {move_count}: {e}")
                error_count += 1
                if error_count >= 3:
                    break

        # Assign outcomes
        _assign_outcomes(game, examples)

        inference.stop()
        print(f"[GAME {game_id}] Completed: {len(examples)} examples, {move_count} moves")
        return examples

    except Exception as e:
        print(f"[GAME {game_id}] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return []


def _create_training_example(
    game: OptimizedGame3D,
    from_logits: torch.Tensor,
    to_logits: torch.Tensor,
    value_pred: torch.Tensor,
    legal_moves: List[Move],
    move_count: int,
) -> TrainingExample:
    """Create training example from current state."""
    from_logits_np = from_logits.numpy()
    to_logits_np = to_logits.numpy()

    # PRE-COMPUTE all indices at once (vectorized)
    from_indices = np.array([coord_to_idx(mv.from_coord) for mv in legal_moves if mv], dtype=np.int32)
    to_indices = np.array([coord_to_idx(mv.to_coord) for mv in legal_moves if mv], dtype=np.int32)

    # Vectorized logit computation
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
        move_count=move_count,
        player_sign=player_sign
    )


def _choose_move(
    game: OptimizedGame3D,
    opponents: Dict[Color, OpponentBase],
    from_logits: torch.Tensor,
    to_logits: torch.Tensor,
    legal_moves: List[Move],
    epsilon: float,
) -> Optional[Move]:
    """Choose move using opponent logic - OPTIMIZED."""
    if random.random() < epsilon:
        return random.choice(legal_moves) if legal_moves else None

    if not legal_moves:
        return None

    # Vectorize everything
    from_indices = np.array([coord_to_idx(mv.from_coord) for mv in legal_moves], dtype=np.int32)
    to_indices = np.array([coord_to_idx(mv.to_coord) for mv in legal_moves], dtype=np.int32)

    from_logits_np = from_logits.numpy()
    to_logits_np = to_logits.numpy()

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


def _assign_outcomes(game: OptimizedGame3D, examples: List[TrainingExample]):
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

    for ex in examples:
        ex.value_target = final_outcome * ex.player_sign


class ParallelSelfPlayGenerator:
    """Generate training data with true parallel execution using processes."""

    def __init__(
        self,
        model_path: Optional[str],
        device: str = "cuda",
        num_parallel_games: int = 8,
        temperature: float = 1.5,
        opponent_types: Optional[List[str]] = None,
        epsilon: float = 0.1,
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

        self.model_path = model_path

        print(f"[PARALLEL] Initialized with {num_parallel_games} parallel games")
        print(f"[DEVICE] AI model on {device}, game logic on CPU processes")

    def generate_games_parallel(self, max_moves: int = 100_000) -> List[TrainingExample]:
        """Generate multiple games in parallel with true multiprocessing."""
        print(f"\n[PARALLEL] Starting {self.num_parallel_games} game processes")
        start_time = time.time()

        # Use multiprocessing Pool with limited processes to avoid resource contention
        num_processes = min(self.num_parallel_games, mp.cpu_count() // 2)  # Don't overload CPU
        print(f"[PARALLEL] Using {num_processes} processes out of {mp.cpu_count()} available CPUs")

        with mp.get_context('spawn').Pool(processes=num_processes) as pool:
            # Prepare arguments for each worker
            worker_args = [
                (
                    i,  # game_id
                    self.model_path,  # model_path
                    self.device,  # device
                    self.opponent_types,  # opponent_types
                    self.epsilon,  # epsilon
                    max_moves,  # max_moves
                )
                for i in range(self.num_parallel_games)
            ]

            # Run games in parallel with chunks to reduce overhead
            chunk_size = max(1, len(worker_args) // num_processes)
            results = pool.map(_game_worker, worker_args, chunksize=chunk_size)

        elapsed = time.time() - start_time

        # Collect results
        all_examples = []
        for examples in results:
            all_examples.extend(examples)

        print(f"\n[PARALLEL] All games completed in {elapsed:.2f}s")
        print(f"[PARALLEL] Total examples: {len(all_examples)}")
        print(f"[PARALLEL] Games per second: {self.num_parallel_games / elapsed:.2f}")

        return all_examples

    def cleanup(self):
        """Cleanup method."""
        pass


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

        # Save current model state to temp file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            torch.save(model.state_dict(), temp_file.name)
            model_path = temp_file.name

        generator = ParallelSelfPlayGenerator(
            model_path=model_path,
            device=device,
            num_parallel_games=games_this_round,
            opponent_types=opponent_types,
            epsilon=epsilon,
        )

        try:
            examples = generator.generate_games_parallel(max_moves)
            all_examples.extend(examples)
        finally:
            # Ensure cleanup happens even if there's an error
            generator.cleanup()
            os.unlink(model_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Round {round_idx + 1} completed: {len(examples)} examples from {games_this_round} games")

    print(f"\n{'='*80}")
    print(f"Total examples generated: {len(all_examples)}")
    print(f"{'='*80}")

    return all_examples
