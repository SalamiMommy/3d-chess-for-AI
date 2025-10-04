# moveeffects.py - Updated to use move_utils
from __future__ import annotations
from typing import List, Tuple, Optional, Set, TYPE_CHECKING
import time

from game3d.board.board import Board
from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.cache.manager import OptimizedCacheManager, get_cache_manager
from game3d.pieces.piece import Piece

# Import shared utilities
from .move_utils import (
    apply_hole_effects,
    apply_bomb_effects,
    apply_trailblaze_effect,
    reconstruct_trailblazer_path,
    extract_enemy_slid_path
)
from .zobrist import compute_zobrist

if TYPE_CHECKING:
    from .gamestate import GameState

# Keep only archery-specific function here
# moveeffects.py - Add archery function
def apply_archery_attack(game_state: 'GameState', target_sq: Tuple[int, int, int]) -> 'GameState':
    """Apply archery attack to create new game state."""
    from .performance import track_operation_time

    with track_operation_time(game_state._metrics, 'total_make_move_time'):
        game_state._metrics.make_move_calls += 1

        new_board = game_state.board.clone()

        # Remove piece at target square if exists
        if game_state.cache.piece_cache.get(target_sq) is not None:
            new_board.set_piece(target_sq, None)

        # Create archery move
        archery_move = Move(
            from_coord=target_sq,
            to_coord=target_sq,
            is_capture=True,
            metadata={
                "is_archery": True,
                "archer_player": game_state.color,
                "target_square": target_sq,
                "timestamp": time.time()
            }
        )

        # Create new cache for new state
        new_cache = get_cache_manager(new_board, game_state.color.opposite())

        from .gamestate import GameState as GS

        # Create new state
        new_state = GS(
            board=new_board,
            color=game_state.color.opposite(),
            cache=new_cache,
            history=game_state.history + (archery_move,),
            halfmove_clock=game_state.halfmove_clock + 1,
            game_mode=game_state.game_mode,
            turn_number=game_state.turn_number + 1,
        )

        # Recompute zobrist key
        new_state._zkey = compute_zobrist(new_board, new_state.color)
        return new_state

def apply_hive_moves(game_state: 'GameState', moves: List[Move]) -> 'GameState':
    """Apply a series of hive moves to create new game state."""
    from .performance import track_operation_time

    with track_operation_time(game_state._metrics, 'total_make_move_time'):
        game_state._metrics.make_move_calls += 1

        current_state = game_state
        for move in moves:
            # Validate each move in the sequence
            if not current_state.is_legal_move(move):
                raise ValueError(f"Hive move {move} is illegal in current position")

            # Apply move
            current_state = current_state.make_move(move)

        return current_state
