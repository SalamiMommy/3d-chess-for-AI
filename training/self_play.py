"""Self-play data generation for 3D chess training with custom opponents."""
import builtins, sys, traceback, inspect, linecache

_real_get = builtins.__getattribute__

import torch
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
import random
from game3d.game.gamestate import GameState
from game3d.common.enums import Color, Result, PieceType
from game3d.movement.movepiece import Move
from game3d.game3d import OptimizedGame3D  # Import the game controller
from training.types import TrainingExample  # Shared dataclass
from game3d.common.coord_utils import coord_to_idx, idx_to_coord, Coord
from game3d.common.constants import SIZE, VOLUME
from game3d.game.factory import new_board_with_manager

# IMPORT OPPONENT MODULE
from training.opponents import (
    create_opponent,
    AVAILABLE_OPPONENTS,
    OpponentBase,
)

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
    """Generate training data through self-play with robust cache handling and custom opponents."""

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
        # Add cache for state tensors
        self._state_tensor_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Choose opponent types for each color (alternate if multiple provided)
        self.opponent_types = opponent_types or ["adversarial", "center_control"]
        if len(self.opponent_types) == 1:
            self.opponent_types = [self.opponent_types[0], self.opponent_types[0]]
        elif len(self.opponent_types) == 2:
            pass
        else:
            # More than 2: cycle through
            self.opponent_types = self.opponent_types[:2]

        # Instantiate opponents
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

        # Cache on CPU to save GPU memory
        self._state_tensor_cache[cache_key] = state_tensor.cpu()

        # Limit cache size
        if len(self._state_tensor_cache) > 1000:
            # Remove oldest 20% of entries
            keys_to_remove = list(self._state_tensor_cache.keys())[:200]
            for key in keys_to_remove:
                del self._state_tensor_cache[key]

        return state_tensor

    def _choose_move_with_opponent(self, game: OptimizedGame3D, policy_logits, legal_moves) -> Move:
        """Choose a move using opponent reward logic and policy probabilities."""
        # Add validation
        if not legal_moves:
            print("[ERROR] _choose_move_with_opponent called with empty legal_moves")
            return None

        if any(mv is None for mv in legal_moves):
            print("[ERROR] _choose_move_with_opponent received None moves")
            legal_moves = [mv for mv in legal_moves if mv is not None]
            if not legal_moves:
                return None

        color = game.state.color
        opponent: OpponentBase = self.opponents[color]

        # Compute policy probabilities
        move_encoder = self.move_encoder
        from_logits, to_logits = policy_logits

        move_scores = []
        move_probs = []
        for mv in legal_moves:
            f_idx = move_encoder.coord_to_index(mv.from_coord)
            t_idx = move_encoder.coord_to_index(mv.to_coord)
            logit = from_logits[0, f_idx] + to_logits[0, t_idx]
            move_probs.append(logit)

        move_probs_tensor = torch.stack(move_probs)
        move_probs_tensor = torch.softmax(move_probs_tensor, dim=0)

        # Compute reward for each move (use current state)
        rewards = []
        for mv in legal_moves:
            reward = opponent.reward(game.state, mv)
            rewards.append(reward)

        rewards_np = np.array(rewards)
        # Mix policy and reward: score = policy_prob + alpha * reward
        alpha = 0.5  # Can be tuned
        scores = move_probs_tensor.cpu().numpy() + alpha * rewards_np

        # Pick move with highest score (or sample stochastically)
        best_idx = int(np.argmax(scores))
        chosen_move = legal_moves[best_idx]
        return chosen_move

    def _try_one_move(self, game: OptimizedGame3D) -> list:
        """Ask the engine for the next move and decorate every exception
        with the piece type that was being processed."""
        try:
            return game.state.legal_moves()
        except NameError as exc:
            # ----  find out which piece was on the square  ----
            sq = game.state.board.active_king_coord   # fallback
            for idx in range(VOLUME):
                c = idx_to_coord(idx)
                pc = game.state.cache.piece_cache.get(c)
                if pc is not None:
                    sq = c
                    break
            pc = game.state.cache.piece_cache.get(sq)
            pt = PieceType(pc.ptype).name if pc else "Unknown"
            new_msg = f"Piece {pt} on {sq} triggered: {exc}"
            print(f"[ERROR] Move {game.state.fullmove_number} failed: {new_msg}")
            raise NameError(new_msg) from exc
        except Exception as exc:
            # Handle the EffectsCache error specifically
            if "EffectsCache' object has no attribute 'piece_cache'" in str(exc):
                print(f"[ERROR] EffectsCache error detected at move {game.state.fullmove_number}")
                print(f"[ERROR] Game state cache type: {type(game.state.cache)}")
                print(f"[ERROR] Game state effects type: {type(game.state.effects)}")

                # Try to recover by using the main cache manager instead of EffectsCache
                try:
                    # Save the original cache
                    original_cache = game.state.cache

                    # Try to get the main cache manager from the effects cache
                    if hasattr(game.state.effects, 'cache'):
                        game.state.cache = game.state.effects.cache
                        print("[ERROR] Attempting to recover by using effects.cache")
                        return game.state.legal_moves()
                    else:
                        print("[ERROR] Cannot recover - no backup cache available")
                        return []
                except Exception as recovery_exc:
                    print(f"[ERROR] Recovery failed: {recovery_exc}")
                    return []
            else:
                # Handle other exceptions as before
                sq = game.state.board.active_king_coord
                pc = game.state.cache.piece_cache.get(sq)
                pt = PieceType(pc.ptype).name if pc else "Unknown"
                new_msg = f"Piece {pt} on {sq} triggered: {exc}"
                print(f"[ERROR] Move {game.state.fullmove_number} failed: {new_msg}")
                raise type(exc)(new_msg) from exc

    def generate_game(self, max_moves: int = 1_000_000) -> List[TrainingExample]:
        from game3d.board.board import Board
        from game3d.game3d import OptimizedGame3D
        from game3d.common.enums import Color

        # 1. create ONE board and ONE cache
        board = new_board_with_manager(Color.WHITE)
        cache = board.cache_manager

        # DIAGNOSTIC: Verify board has pieces
        occupied = list(board.list_occupied())
        print(f"[DIAGNOSTIC] Board has {len(occupied)} pieces")
        if len(occupied) == 0:
            print("[ERROR] Board is empty! Cannot generate game.")
            return []

        # DIAGNOSTIC: Verify cache is properly initialized
        print(f"[DIAGNOSTIC] Cache type: {type(cache)}")
        print(f"[DIAGNOSTIC] Cache has piece_cache: {hasattr(cache, 'piece_cache')}")

        game = OptimizedGame3D(board=board, cache=cache)
        game.toggle_debug_turn_info(False)
        examples = []
        move_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 3

        # Pre-warm the cache with initial state
        initial_tensor = self._get_state_tensor(game.state)

        print(f"[GAME] Starting new game with max_moves={max_moves}")

        while not game.is_game_over() and move_count < max_moves:
            try:
                # Get current state tensor (with caching)
                state_tensor = self._get_state_tensor(game.state)

                # Verify the state is valid
                if game.state is None:
                    print(f"[ERROR] Game state is None at move {move_count}")
                    break

                # Check cache efficiency periodically
                if move_count % 50 == 0:
                    total_requests = self._cache_hits + self._cache_misses
                    if total_requests > 0:
                        hit_rate = self._cache_hits / total_requests
                        print(f"[Cache] Hit rate: {hit_rate:.2%} ({self._cache_hits}/{total_requests})")

                with torch.no_grad():
                    from_logits, to_logits, value_pred = self.model(state_tensor)
                    from_logits = from_logits / self.temperature
                    to_logits = to_logits / self.temperature

                # Get legal moves from the game (these should be cached)
                legal_moves = self._try_one_move(game)
                if not legal_moves:
                    print(f"[DEBUG] No legal moves found at move {move_count}")
                    print(f"[DEBUG] Game state: in_check={game.state.is_check()}, game_over={game.is_game_over()}")
                    break

                # Verify moves are valid before processing
                if any(mv is None for mv in legal_moves):
                    print(f"[ERROR] Found None moves in legal_moves at move {move_count}")
                    legal_moves = [mv for mv in legal_moves if mv is not None]
                    if not legal_moves:
                        print(f"[ERROR] No valid moves after filtering None values")
                        break

                # Use opponent logic to choose move
                try:
                    chosen_move = self._choose_move_with_opponent(game, (from_logits, to_logits), legal_moves)
                    if chosen_move is None:
                        print(f"[ERROR] chosen_move is None at move {move_count}")
                        break
                except Exception as e:
                    print(f"[ERROR] Failed to choose move at move {move_count}: {e}")
                    import traceback
                    traceback.print_exc()
                    break

                # Compute marginal from and to targets (using policy only for training)
                move_encoder = self.move_encoder
                from_indices = []
                to_indices = []
                move_probs = []
                for mv in legal_moves:
                    f_idx = move_encoder.coord_to_index(mv.from_coord)
                    t_idx = move_encoder.coord_to_index(mv.to_coord)
                    logit = from_logits[0, f_idx] + to_logits[0, t_idx]
                    move_probs.append(logit)
                    from_indices.append(f_idx)
                    to_indices.append(t_idx)

                move_probs_tensor = torch.stack(move_probs)
                move_probs_tensor = torch.softmax(move_probs_tensor, dim=0)

                from_target = torch.zeros(729, device=self.device)
                to_target   = torch.zeros(729, device=self.device)
                for i in range(len(legal_moves)):
                    prob = move_probs_tensor[i]
                    from_target[from_indices[i]] += prob
                    to_target[to_indices[i]] += prob
                if from_target.sum() > 0:
                    from_target /= from_target.sum()
                if to_target.sum() > 0:
                    to_target /= to_target.sum()

                # Player sign for this example
                player_sign = 1.0 if game.state.color == Color.WHITE else -1.0

                examples.append(
                    TrainingExample(
                        state_tensor=state_tensor.squeeze(0).cpu(),
                        from_target=from_target.cpu(),
                        to_target=to_target.cpu(),
                        value_target=value_pred.item(),  # Will be overwritten with final outcome
                        move_count=move_count,
                        player_sign=player_sign  # Store per example
                    )
                )
                print(f"[SP] About to submit move on board-id={id(game.state.board)} "
                    f"cache_manager={game.state.board.cache_manager}")
                # Submit the move to the game
                receipt = game.submit_move(chosen_move)
                if not receipt.is_legal:
                    raise ValueError(receipt.message)

                move_count += 1
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                print(f"[ERROR] Move {move_count} failed: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    print(f"[ERROR] Too many consecutive errors ({consecutive_errors}), ending game")
                    break

        # Print cache statistics
        total_requests = self._cache_hits + self._cache_misses
        if total_requests > 0:
            hit_rate = self._cache_hits / total_requests
            print(f"[Cache] Final hit rate: {hit_rate:.2%} ({self._cache_hits}/{total_requests})")

        # Print game summary
        print(f"\n[GAME SUMMARY]")
        print(f"  Total moves: {move_count}")
        print(f"  Game over: {game.is_game_over()}")
        print(f"  Turn: {game.state.color.name}")
        print(f"  Halfmove clock: {game.state.halfmove_clock}")

        # Print piece counts
        piece_counts = {
            Color.WHITE: {ptype: 0 for ptype in PieceType},
            Color.BLACK: {ptype: 0 for ptype in PieceType}
        }
        for _, piece in game.state.board.list_occupied():
            piece_counts[piece.color][piece.ptype] += 1
        print(f"  White pieces: {piece_counts[Color.WHITE]}")
        print(f"  Black pieces: {piece_counts[Color.BLACK]}")

        # Check if black king is missing
        if piece_counts[Color.BLACK][PieceType.KING] == 0:
            print("[WARNING] Black king is missing - this may indicate a bug")

        # Assign final outcomes
        if game.is_game_over():
            result = game.result()
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
            print("Game result: UNFINISHED (max moves reached or crashed)")

        for ex in examples:
            ex.value_target = final_outcome * ex.player_sign  # Use per-example player_sign

        return examples

def play_game(model: torch.nn.Module, max_moves: int = 1_000_000, device: str = "cpu", opponent_types: Optional[List[str]] = None) -> List[TrainingExample]:
    """Self-play a single game using custom opponents."""
    generator = SelfPlayGenerator(model, device=device, opponent_types=opponent_types)
    return generator.generate_game(max_moves)

def generate_training_data(model: torch.nn.Module, num_games: int = 10, max_moves: int = 1_000_000, device: str = "cpu", opponent_types: Optional[List[str]] = None) -> List[TrainingExample]:
    """Generate examples from multiple self-play games using custom opponents."""
    all_examples = []
    for game_idx in range(num_games):
        print(f"Generating game {game_idx + 1}/{num_games}")
        game_examples = play_game(model, max_moves, device, opponent_types=opponent_types)
        all_examples.extend(game_examples)
    print(f"Total examples generated: {len(all_examples)}")
    return all_examples
