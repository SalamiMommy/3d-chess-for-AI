"""Self-play data generation for 3D chess training."""
import builtins, sys, traceback, inspect, linecache

_real_get = builtins.__getattribute__

def _trap(name):
    if name == 'cache':
        # 1. native Python stack
        traceback.print_stack(file=sys.stdout)
        # 2. deepest frame we can reach
        frame = inspect.currentframe()
        while frame.f_back: frame = frame.f_back
        filename = frame.f_code.co_filename
        lineno   = frame.f_lineno
        line     = linecache.getline(filename, lineno).strip()
        print('\n>>> DEEPEST FRAME <<<')
        print('file :', filename)
        print('line :', lineno)
        print('text :', line)
        print('locals-keys:', list(frame.f_locals.keys())[:20])
        # 3. do NOT raise here – let the original NameError propagate
        #    so the *real* exception is still shown afterwards
    return _real_get(name)

builtins.__getattribute__ = _trap

import torch
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
import random
from game3d.game.gamestate import GameState
from game3d.pieces.enums import Color, Result
from game3d.cache.manager import OptimizedCacheManager
from game3d.movement.movepiece import Move
from game3d.game3d import OptimizedGame3D  # Import the game controller
from training.types import TrainingExample  # Shared dataclass
from game3d.common.common import SIZE, VOLUME, coord_to_idx, idx_to_coord, Coord
from game3d.pieces.enums import PieceType

class MoveEncoder:
    def coord_to_index(self, coord: Coord) -> int:
        return coord_to_idx(coord)

def move_to_index(from_coord: Coord, to_coord: Coord) -> int:
    return coord_to_idx(from_coord) * VOLUME + coord_to_idx(to_coord)

def index_to_move(idx: int) -> tuple[Coord, Coord]:
    to_idx = idx % VOLUME
    from_idx = idx // VOLUME
    return idx_to_coord(from_idx), idx_to_coord(to_idx)

# In self_play.py
class SelfPlayGenerator:
    """Generate training data through self-play with robust cache handling."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        temperature: float = 1.0,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.temperature = temperature
        self.move_encoder = MoveEncoder()
        # Add cache for state tensors
        self._state_tensor_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

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

    def generate_game(self, max_moves: int = 200) -> List[TrainingExample]:
        """Generate a single game with robust error handling using OptimizedGame3D."""
        game = OptimizedGame3D()
        game.toggle_debug_turn_info(False)
        examples = []
        move_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 3

        # Pre-warm the cache with initial state
        initial_tensor = self._get_state_tensor(game.state)

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
                # legal_moves = game.state.legal_moves()
                legal_moves = self._try_one_move(game)
                if not legal_moves:
                    break

                # Compute move logits
                move_logits = []
                from_indices = []
                to_indices = []
                for mv in legal_moves:
                    f_idx = self.move_encoder.coord_to_index(mv.from_coord)
                    t_idx = self.move_encoder.coord_to_index(mv.to_coord)
                    logit = from_logits[0, f_idx] + to_logits[0, t_idx]
                    move_logits.append(logit)
                    from_indices.append(f_idx)
                    to_indices.append(t_idx)

                move_logits = torch.stack(move_logits)
                move_probs = torch.softmax(move_logits, dim=0)

                # Sample move
                chosen_idx = torch.multinomial(move_probs, num_samples=1).item()
                chosen_move = legal_moves[chosen_idx]

                # Compute marginal from and to targets
                from_target = torch.zeros(729, device=self.device)
                to_target   = torch.zeros(729, device=self.device)
                for i in range(len(legal_moves)):
                    prob = move_probs[i]
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
                    break

        # Print cache statistics
        total_requests = self._cache_hits + self._cache_misses
        if total_requests > 0:
            hit_rate = self._cache_hits / total_requests
            print(f"[Cache] Final hit rate: {hit_rate:.2%} ({self._cache_hits}/{total_requests})")

        # Assign final outcomes
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
            ex.value_target = final_outcome * ex.player_sign  # Use per-example player_sign

        return examples

    def _try_one_move(self, game: OptimizedGame3D) -> list:
        """Ask the engine for the next move and decorate every exception
        with the piece type that was being processed."""
        try:
            return game.state.legal_moves()          # <-- the call that explodes
        except NameError as exc:
            # ----  find out which piece was on the square  ----
            sq = game.state.board.active_king_coord   # fallback
            for idx in range(VOLUME):
                c = idx_to_coord(idx)
                pc = game.state.board.piece_at(c)
                if pc is not None:
                    sq = c
                    break
            pc = game.state.board.piece_at(sq)
            pt = PieceType(pc.ptype).name if pc else "Unknown"
            new_msg = f"Piece {pt} on {sq} triggered: {exc}"
            print(f"[ERROR] Move {game.state.fullmove_number} failed: {new_msg}")  # <-- NEW
            raise NameError(new_msg) from exc
        except Exception as exc:
            sq = game.state.board.active_king_coord
            pc = game.state.board.piece_at(sq)
            pt = PieceType(pc.ptype).name if pc else "Unknown"
            new_msg = f"Piece {pt} on {sq} triggered: {exc}"
            print(f"[ERROR] Move {game.state.fullmove_number} failed: {new_msg}")  # <-- NEW
            raise type(exc)(new_msg) from exc

def play_game(model: torch.nn.Module, max_moves: int = 200, device: str = "cuda") -> List[TrainingExample]:
    """Self-play a single game."""
    generator = SelfPlayGenerator(model, device=device)
    return generator.generate_game(max_moves)

def generate_training_data(model: torch.nn.Module, num_games: int = 10, max_moves: int = 200, device: str = "cuda") -> List[TrainingExample]:
    """Generate examples from multiple self-play games."""
    all_examples = []
    for game_idx in range(num_games):
        print(f"Generating game {game_idx + 1}/{num_games}")
        game_examples = play_game(model, max_moves, device)
        all_examples.extend(game_examples)
    print(f"Total examples generated: {len(all_examples)}")
    return all_examples
