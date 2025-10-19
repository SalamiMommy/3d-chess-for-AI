from __future__ import annotations
from typing import List, Tuple, Optional, Set, TYPE_CHECKING
import time

from game3d.board.board import Board
from game3d.common.enums import Color, PieceType
from game3d.movement.movepiece import Move

from game3d.pieces.piece import Piece
from game3d.movement.movepiece import MOVE_FLAGS
from .move_utils import (
    apply_hole_effects,
    apply_bomb_effects,
    apply_trailblaze_effect,
    reconstruct_trailblazer_path,
    extract_enemy_slid_path
)
from .zobrist import compute_zobrist
from game3d.cache.manager import get_cache_manager  # Added missing import

if TYPE_CHECKING:
    from .gamestate import GameState

# Keep only archery-specific function here
# moveeffects.py - Add archery function
def apply_archery_attack(game_state: 'GameState', target_sq: Tuple[int, int, int]) -> 'GameState':

    from .performance import track_operation_time

    with track_operation_time(game_state._metrics, 'total_make_move_time'):
        game_state._metrics.make_move_calls += 1

        new_board = game_state.board.clone()

        # Remove piece at target square if exists (optimized: check first)
        if new_board.get_piece(target_sq) is not None:  # Use board method for consistency
            new_board.set_piece(target_sq, None)

        # Create archery move
        archery_move = Move(
            from_coord=target_sq,
            to_coord=target_sq,
            flags=MOVE_FLAGS['CAPTURE'],
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
            history=game_state.history + (archery_move,),  # Use list for history to avoid tuple allocations if mutable
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

        # Optimization: Validate all moves first in batch to avoid partial applications
        for move in moves:
            if not game_state.is_legal_move(move):
                raise ValueError(f"Hive move {move} is illegal in current position")

        # Optimization: Use a single board clone and apply moves incrementally instead of full make_move each time
        new_board = game_state.board.clone()
        new_cache = get_cache_manager(new_board, game_state.color.opposite())  # Shared cache for efficiency
        current_state = game_state  # Start from original

        for move in moves:
            # Apply incrementally (assuming board.apply_move is efficient)
            new_board.apply_move(move)  # Apply to cloned board
            # Update cache incrementally if possible (add call if method exists)
            new_cache.update_after_move(move)  # Assuming such a method; implement if needed

        # Create final state once
        new_state = current_state.__class__(  # Use class for instantiation
            board=new_board,
            color=game_state.color.opposite(),
            cache=new_cache,
            history=(*game_state.history, *moves),  # Batch append
            halfmove_clock=game_state.halfmove_clock + len(moves),  # Assume increments per move
            game_mode=game_state.game_mode,
            turn_number=game_state.turn_number + 1,
        )
        return new_state
