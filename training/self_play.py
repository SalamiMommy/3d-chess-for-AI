# self_play.py
"""Self-play data generation - FIXED to reuse single cache manager."""
import torch
import numpy as np
from typing import List, Optional
import random
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
)
from training.opponents import (
    create_opponent,
    AVAILABLE_OPPONENTS,
    OpponentBase,
)
import multiprocessing as mp
import gc

# ADD THIS IMPORT
from game3d.game.factory import start_game_state  # ADD THIS LINE

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
        device: str = "cpu",
        temperature: float = 1.0,
        opponent_types: Optional[List[str]] = None,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.temperature = temperature
        self.move_encoder = MoveEncoder()
        self._state_tensor_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # CRITICAL: Use factory to create initial game state with cache manager
        initial_game_state = start_game_state()  # This creates the cache manager via factory
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

    def _get_state_tensor(self, game_state: GameState) -> torch.Tensor:
        """Get state tensor with caching."""
        state_hash = game_state.board.byte_hash()
        cache_key = (state_hash, game_state.color)

        if cache_key in self._state_tensor_cache:
            self._cache_hits += 1
            return self._state_tensor_cache[cache_key].to(self.device)

        self._cache_misses += 1
        state_tensor = game_state.to_tensor(device=self.device).unsqueeze(0)
        self._state_tensor_cache[cache_key] = state_tensor.cpu()

        if len(self._state_tensor_cache) > 1000:
            keys_to_remove = list(self._state_tensor_cache.keys())[:200]
            for key in keys_to_remove:
                del self._state_tensor_cache[key]

        return state_tensor

    def _choose_move_with_opponent(self, game: OptimizedGame3D, policy_logits, legal_moves) -> Move:
        """Choose a move using opponent reward logic and policy probabilities."""
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
            move_probs.append(logit)

        move_probs_tensor = torch.tensor(move_probs)
        move_probs_tensor = torch.softmax(move_probs_tensor, dim=0)

        rewards = [opponent.reward(game.state, mv) for mv in legal_moves]
        rewards_np = np.array(rewards)

        alpha = 0.5
        scores = move_probs_tensor.cpu().numpy() + alpha * rewards_np

        best_idx = int(np.argmax(scores))
        return legal_moves[best_idx]

    def generate_game(self, max_moves: int = 100_000) -> List[TrainingExample]:
        """Generate ONE game with proper reset."""

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

        while move_count < max_moves and not game.is_game_over():
            try:
                state_tensor = self._get_state_tensor(game.state)
                with torch.no_grad():
                    from_logits, to_logits, value_pred = self.model(state_tensor)
                    policy_logits = (from_logits, to_logits)

                legal_moves = game.state.legal_moves()
                if not legal_moves:
                    print(f"[WARNING] No legal moves at move {move_count}, ending game")
                    break

                chosen_move = self._choose_move_with_opponent(game, policy_logits, legal_moves)

                # Prepare targets
                from_indices = []
                to_indices = []
                move_probs = []

                for mv in legal_moves:
                    if mv is None:
                        continue
                    f_idx = self.move_encoder.coord_to_index(mv.from_coord)
                    t_idx = self.move_encoder.coord_to_index(mv.to_coord)
                    logit = policy_logits[0][0, f_idx] + policy_logits[1][0, t_idx]
                    move_probs.append(logit)
                    from_indices.append(f_idx)
                    to_indices.append(t_idx)

                move_probs_tensor = torch.stack(move_probs)
                move_probs_tensor = torch.softmax(move_probs_tensor, dim=0)

                from_target = torch.zeros(729, device=self.device)
                to_target = torch.zeros(729, device=self.device)

                for i in range(len(legal_moves)):
                    prob = move_probs_tensor[i]
                    from_target[from_indices[i]] += prob
                    to_target[to_indices[i]] += prob

                if from_target.sum() > 0:
                    from_target /= from_target.sum()
                if to_target.sum() > 0:
                    to_target /= to_target.sum()

                player_sign = 1.0 if game.state.color == Color.WHITE else -1.0

                examples.append(
                    TrainingExample(
                        state_tensor=state_tensor.squeeze(0).cpu(),
                        from_target=from_target.cpu(),
                        to_target=to_target.cpu(),
                        value_target=value_pred.item(),
                        move_count=move_count,
                        player_sign=player_sign
                    )
                )

                # Verify cache manager is still the same
                if game.state.cache_manager is not self._shared_cache_manager:
                    print(f"[ERROR] Cache manager changed! Expected {id(self._shared_cache_manager)}, got {id(game.state.cache_manager)}")
                    # Force it back
                    game.state.cache_manager = self._shared_cache_manager
                    game.state.board.cache_manager = self._shared_cache_manager

                receipt = game.submit_move(chosen_move)
                if not receipt.is_legal:
                    raise ValueError(receipt.message)

                if move_count % 100 == 0:
                    print(f"[MOVE {move_count}] {chosen_move} ({game.state.color.name} to move)")

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
            final_outcome = 0.0
            print("Game result: UNFINISHED")

        for ex in examples:
            ex.value_target = final_outcome * ex.player_sign

        return examples

def play_game(model: torch.nn.Module, max_moves: int = 100_000, device: str = "cpu", opponent_types: Optional[List[str]] = None) -> List[TrainingExample]:
    """Self-play a single game - creates ONE generator (ONE cache manager)."""
    generator = SelfPlayGenerator(model, device=device, opponent_types=opponent_types)
    return generator.generate_game(max_moves)

def generate_training_data(model: torch.nn.Module, num_games: int = 10, max_moves: int = 100_000, device: str = "cpu", opponent_types: Optional[List[str]] = None) -> List[TrainingExample]:
    """Generate examples from multiple games - ONE generator reused for all games."""
    # CRITICAL: Create ONE generator that reuses cache manager across all games
    generator = SelfPlayGenerator(model, device=device, opponent_types=opponent_types)

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
    model, max_moves, device, opponent_types = args
    return play_game(model, max_moves, device, opponent_types)

def generate_training_data_parallel(model: torch.nn.Module, num_games: int = 10, max_moves: int = 100_000, device: str = "cpu", opponent_types: Optional[List[str]] = None, num_processes: int = 4) -> List[TrainingExample]:
    with mp.Pool(num_processes) as pool:
        args = [(model, max_moves, device, opponent_types) for _ in range(num_games)]
        all_games = pool.map(parallel_generate_game, args)

    all_examples = []
    for game_examples in all_games:
        all_examples.extend(game_examples)

    return all_examples
