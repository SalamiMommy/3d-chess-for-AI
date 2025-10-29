# self_play.py - UPDATED to properly handle numpy/tensor conversion
"""Self-play data generation with termination via terminal.py."""
import torch
import numpy as np
from typing import List, Optional, Dict
import random
from collections import defaultdict
from game3d.game.gamestate import GameState
from game3d.common.enums import Color, Result, PieceType
from game3d.movement.movepiece import Move
from game3d.game3d import OptimizedGame3D
from training.types import TrainingExample
from game3d.common.coord_utils import coord_to_idx, idx_to_coord, Coord
from game3d.common.piece_utils import find_king
from game3d.common.constants import SIZE, VOLUME
from game3d.board.board import Board
from game3d.game.terminal import (
    is_check,
    is_stalemate,
    is_insufficient_material,
    is_fifty_move_draw,
    is_threefold_repetition,
)
from training.opponents import (
    create_opponent,
    AVAILABLE_OPPONENTS,
    OpponentBase,
)
import multiprocessing as mp
import gc

from game3d.game.factory import start_game_state

class MoveEncoder:
    def coord_to_index(self, coord: Coord) -> int:
        return coord_to_idx(coord)

def move_to_index(from_coord: Coord, to_coord: Coord) -> int:
    return coord_to_idx(from_coord) * VOLUME + coord_to_idx(to_coord)

def index_to_move(idx: int) -> tuple[Coord, Coord]:
    to_idx = idx % VOLUME
    from_idx = idx // VOLUME
    return idx_to_coord(from_idx), idx_to_coord(to_idx)

class SelfPlayGenerator:
    """Generate training data with SINGLE cache manager reused across all games."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        temperature: float = 1.5,
        opponent_types: Optional[List[str]] = None,
        epsilon: float = 0.1
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.temperature = temperature
        self.epsilon = epsilon
        self.move_encoder = MoveEncoder()

        # Cache for NUMPY arrays (internal representation)
        self._state_array_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # CRITICAL: Use factory to create initial game state with cache manager
        initial_game_state = start_game_state()
        self._shared_cache_manager = initial_game_state.cache_manager
        print(f"[CACHE MANAGER] Using factory-created cache manager ID: {id(self._shared_cache_manager)}")

        # Opponent setup
        self.opponent_types = opponent_types or ["adversarial", "center_control"]
        if len(self.opponent_types) == 1:
            self.opponent_types = [self.opponent_types[0], self.opponent_types[0]]
        else:
            self.opponent_types = self.opponent_types[:2]

        self.opponents = {
            Color.WHITE: create_opponent(self.opponent_types[0], Color.WHITE),
            Color.BLACK: create_opponent(self.opponent_types[1], Color.BLACK),
        }

    def _get_state_tensor_for_model(self, game_state: GameState) -> torch.Tensor:
        """Get state tensor - converts from internal numpy array to tensor for AI ONLY."""
        state_hash = game_state.board.byte_hash()
        cache_key = (state_hash, game_state.color)

        if cache_key in self._state_array_cache:
            self._cache_hits += 1
            array = self._state_array_cache[cache_key]
        else:
            self._cache_misses += 1
            # Use internal numpy representation
            array = game_state.to_array()
            self._state_array_cache[cache_key] = array

            # Clean cache if too large
            if len(self._state_array_cache) > 1000:
                keys_to_remove = list(self._state_array_cache.keys())[:200]
                for key in keys_to_remove:
                    del self._state_array_cache[key]

        # ONLY CONVERT TO TENSOR HERE - for AI model consumption
        tensor = torch.from_numpy(array).to(self.device).unsqueeze(0)
        return tensor

    def _choose_move_with_opponent(self, game: OptimizedGame3D, policy_logits, legal_moves) -> Move:
        """Choose a move using opponent reward logic and policy probabilities."""
        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        if not legal_moves:
            print("[ERROR] _choose_move_with_opponent called with empty legal_moves")
            return None

        legal_moves = [mv for mv in legal_moves if mv is not None]
        if not legal_moves:
            return None

        color = game.state.color
        opponent: OpponentBase = self.opponents[color]

        from_logits, to_logits = policy_logits
        move_encoder = self.move_encoder

        move_probs = []
        for mv in legal_moves:
            f_idx = move_encoder.coord_to_index(mv.from_coord)
            t_idx = move_encoder.coord_to_index(mv.to_coord)
            logit = from_logits[0, f_idx] + to_logits[0, t_idx]
            move_probs.append(logit.item())

        move_probs_np = np.array(move_probs, dtype=np.float32)
        max_logit = np.max(move_probs_np)
        exp_logit = np.exp(move_probs_np - max_logit)
        sum_exp = np.sum(exp_logit)
        if sum_exp == 0 or np.isnan(sum_exp):
            move_probs_soft = np.ones_like(move_probs_np) / len(move_probs_np)
        else:
            move_probs_soft = exp_logit / sum_exp

        rewards = [opponent.reward(game.state, mv) for mv in legal_moves]
        rewards_np = np.array(rewards)

        alpha = 0.5
        scores = move_probs_soft + alpha * rewards_np

        best_idx = int(np.argmax(scores))
        return legal_moves[best_idx]

    def generate_game(self, max_moves: int = 100_000) -> List[TrainingExample]:
        """Generate ONE game - termination logic delegated to terminal.py."""

        # CRITICAL: Use factory pattern to create new board with shared cache manager
        board = Board.empty()
        board.init_startpos()

        # Rebuild the shared cache manager for this new board
        self._shared_cache_manager.rebuild(board, Color.WHITE)

        game = OptimizedGame3D(board=board, cache=self._shared_cache_manager)
        examples = []
        move_count = 0
        max_consecutive_errors = 3
        consecutive_errors = 0

        # Initialize position tracking in game state for terminal.py to use
        if not hasattr(game.state, '_position_counts'):
            game.state._position_counts = defaultdict(int)
        game.state._position_counts[game.state.zkey] = 1

        while move_count < max_moves and not game.is_game_over():
            try:
                # Check termination conditions via terminal.py
                if is_fifty_move_draw(game.state):
                    print(f"[GAME END] Fifty-move rule at move {move_count}")
                    break

                if is_threefold_repetition(game.state):
                    print(f"[GAME END] Threefold repetition at move {move_count}")
                    break

                # ONLY convert to tensor here for AI model
                state_tensor = self._get_state_tensor_for_model(game.state)
                with torch.no_grad():
                    from_logits, to_logits, value_pred = self.model(state_tensor)
                    policy_logits = (from_logits, to_logits)

                legal_moves = game.state.legal_moves()
                if not legal_moves:
                    print(f"[WARNING] No legal moves at move {move_count}, ending game")
                    break

                chosen_move = self._choose_move_with_opponent(game, policy_logits, legal_moves)
                if chosen_move is None:
                    print(f"[ERROR] No move chosen at move {move_count}")
                    break

                opponent = self.opponents[game.state.color]
                opponent.observe(game.state, chosen_move)

                # Prepare targets - using numpy internally
                from_indices = []
                to_indices = []
                move_probs = []

                for mv in legal_moves:
                    if mv is None:
                        continue
                    f_idx = self.move_encoder.coord_to_index(mv.from_coord)
                    t_idx = self.move_encoder.coord_to_index(mv.to_coord)
                    logit = policy_logits[0][0, f_idx] + policy_logits[1][0, t_idx]
                    move_probs.append(logit.item())
                    from_indices.append(f_idx)
                    to_indices.append(t_idx)

                move_probs_np = np.array(move_probs, dtype=np.float32)
                max_logit = np.max(move_probs_np)
                exp_logit = np.exp(move_probs_np - max_logit)
                sum_exp = np.sum(exp_logit)
                if sum_exp == 0 or np.isnan(sum_exp):
                    move_probs_soft = np.ones_like(move_probs_np) / len(move_probs_np)
                else:
                    move_probs_soft = exp_logit / sum_exp

                from_target_np = np.zeros(729, dtype=np.float32)
                to_target_np = np.zeros(729, dtype=np.float32)

                for i in range(len(legal_moves)):
                    prob = move_probs_soft[i]
                    from_target_np[from_indices[i]] += prob
                    to_target_np[to_indices[i]] += prob

                if np.sum(from_target_np) > 0:
                    from_target_np /= np.sum(from_target_np)
                if np.sum(to_target_np) > 0:
                    to_target_np /= np.sum(to_target_np)

                player_sign = 1.0 if game.state.color == Color.WHITE else -1.0

                # Store as numpy arrays internally, convert to tensors only in dataset
                state_array = game.state.to_array()  # Internal numpy representation

                examples.append(
                    TrainingExample(
                        state_tensor=state_array,  # Store as numpy
                        from_target=from_target_np,  # Store as numpy
                        to_target=to_target_np,  # Store as numpy
                        value_target=value_pred.item(),
                        move_count=move_count,
                        player_sign=player_sign
                    )
                )

                # Verify cache manager is still the same
                if game.state.cache_manager is not self._shared_cache_manager:
                    print(f"[ERROR] Cache manager changed! Expected {id(self._shared_cache_manager)}, got {id(game.state.cache_manager)}")
                    game.state.cache_manager = self._shared_cache_manager
                    game.state.board.cache_manager = self._shared_cache_manager

                receipt = game.submit_move(chosen_move)
                if not receipt.is_legal:
                    raise ValueError(receipt.message)

                # Update position count for threefold repetition tracking
                game.state._position_counts[game.state.zkey] += 1

                if move_count % 100 == 0:
                    print(f"[MOVE {move_count}] {chosen_move} ({game.state.color.name} to move)")
                    print(f"  Halfmove clock: {game.state.halfmove_clock}")
                    print(f"  Unique positions: {len(game.state._position_counts)}")

                move_count += 1
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                print(f"[ERROR] Move {move_count} failed: {e}")
                import traceback
                traceback.print_exc()

                if consecutive_errors >= max_consecutive_errors:
                    print(f"[ERROR] Too many consecutive errors ({consecutive_errors}), ending game")
                    break

        # Print final statistics
        total_requests = self._cache_hits + self._cache_misses
        if total_requests > 0:
            hit_rate = self._cache_hits / total_requests
            print(f"[Cache] Final hit rate: {hit_rate:.2%} ({self._cache_hits}/{total_requests})")

        print(f"\n[GAME SUMMARY]")
        print(f"  Total moves: {move_count}")
        print(f"  Game over: {game.is_game_over()}")
        print(f"  Halfmove clock: {game.state.halfmove_clock}")
        print(f"  Unique positions: {len(game.state._position_counts)}")
        print(f"  Cache manager ID: {id(self._shared_cache_manager)}")

        # Assign final outcomes
        if game.is_game_over():
            result = game.result()
            print(f"[GAME OVER] Move {move_count}, Result: {result}")

            # Debug why game ended
            if is_stalemate(game.state):
                print("  - Reason: Stalemate")
            if is_check(game.state):
                print("  - Reason: Check")
            if is_insufficient_material(game.state):
                print("  - Reason: Insufficient material")
            if is_fifty_move_draw(game.state):
                print(f"  - Reason: 50-move rule (halfmove_clock: {game.state.halfmove_clock})")
            if is_threefold_repetition(game.state):
                print("  - Reason: Threefold repetition")

            legal_moves = game.state.legal_moves()
            print(f"  - Legal moves available: {len(legal_moves)}")

            if result == Result.WHITE_WON:
                final_outcome = 1.0
                print("Game result: WHITE wins")
            elif result == Result.BLACK_WON:
                final_outcome = -1.0
                print("Game result: BLACK wins")
            else:
                final_outcome = 0.0
                print("Game result: DRAW")
        else:
            # Game didn't finish naturally - assign draw
            final_outcome = 0.0
            print("Game result: INCOMPLETE (treated as DRAW)")

        for ex in examples:
            ex.value_target = final_outcome * ex.player_sign

        return examples

def play_game(model: torch.nn.Module, max_moves: int = 100_000, device: str = "cuda",
             opponent_types: Optional[List[str]] = None, epsilon: float = 0.1) -> List[TrainingExample]:
    """Self-play a single game - creates ONE generator (ONE cache manager)."""
    generator = SelfPlayGenerator(model, device=device, opponent_types=opponent_types, epsilon=epsilon)
    return generator.generate_game(max_moves)

def generate_training_data(model: torch.nn.Module, num_games: int = 10, max_moves: int = 100_000,
                          device: str = "cuda", opponent_types: Optional[List[str]] = None,
                          epsilon: float = 0.1) -> List[TrainingExample]:
    """Generate examples from multiple games - ONE generator reused for all games."""
    generator = SelfPlayGenerator(model, device=device, opponent_types=opponent_types, epsilon=epsilon)

    all_examples = []
    for game_idx in range(num_games):
        print(f"\n{'='*80}")
        print(f"Generating game {game_idx + 1}/{num_games}")
        print(f"{'='*80}")

        game_examples = generator.generate_game(max_moves)
        all_examples.extend(game_examples)

        print(f"Game {game_idx + 1} completed: {len(game_examples)} examples")
        gc.collect()

    print(f"\n{'='*80}")
    print(f"Total examples generated: {len(all_examples)}")
    print(f"{'='*80}")
    return all_examples

def parallel_generate_game(args):
    model, max_moves, device, opponent_types, epsilon = args
    return play_game(model, max_moves, device, opponent_types, epsilon)

def generate_training_data_parallel(model: torch.nn.Module, num_games: int = 10, max_moves: int = 100_000,
                                  device: str = "cuda", opponent_types: Optional[List[str]] = None,
                                  epsilon: float = 0.1, num_processes: int = 4) -> List[TrainingExample]:
    with mp.Pool(num_processes) as pool:
        args = [(model, max_moves, device, opponent_types, epsilon) for _ in range(num_games)]
        all_games = pool.map(parallel_generate_game, args)

    all_examples = []
    for game_examples in all_games:
        all_examples.extend(game_examples)

    return all_examples
