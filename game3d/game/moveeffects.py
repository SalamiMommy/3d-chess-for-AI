# moveeffects.py - FIXED
# moveeffects.py
from __future__ import annotations
from typing import List, Tuple, Optional, Set, TYPE_CHECKING
import time  # Added missing import

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

if TYPE_CHECKING:
    from .gamestate import GameState

def apply_archery_attack(game_state: 'GameState', target_sq: Tuple[int, int, int]) -> 'GameState':
    from .performance import track_operation_time

    with track_operation_time(game_state._metrics, 'total_make_move_time'):
        game_state._metrics.make_move_calls += 1

        new_board = game_state.board.clone()

        # Remove piece at target square if exists
        piece = game_state.cache_manager.occupancy.get(target_sq)
        if piece is not None:
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

        # KEY FIX: Reuse existing cache manager with updated board
        existing_cache_manager = game_state.cache_manager
        existing_cache_manager.board = new_board
        new_board.cache_manager = existing_cache_manager

        # Update occupancy in existing cache manager
        existing_cache_manager.occupancy.set_position(target_sq, None)

        from .gamestate import GameState as GS

        # Create new state with SAME cache manager
        new_state = GS(
            board=new_board,
            color=game_state.color.opposite(),
            cache_manager=existing_cache_manager,  # REUSE manager
            history=game_state.history + (archery_move,),
            halfmove_clock=game_state.halfmove_clock + 1,
            game_mode=game_state.game_mode,
            turn_number=game_state.turn_number + 1,
        )

        # Zobrist will be incrementally updated by cache manager
        return new_state

def apply_hive_moves(game_state: 'GameState', moves: List[Move]) -> 'GameState':
    """Apply a series of hive moves reusing the same cache manager."""
    from .performance import track_operation_time

    with track_operation_time(game_state._metrics, 'total_make_move_time'):
        game_state._metrics.make_move_calls += 1

        # Optimization: Validate all moves first
        for move in moves:
            legal_moves = game_state.legal_moves()
            if move not in legal_moves:
                raise ValueError(f"Hive move {move} is illegal")

        # Use existing cache manager
        existing_cache_manager = game_state.cache_manager
        current_board = game_state.board

        # Apply all moves incrementally to the same cache manager
        for move in moves:
            existing_cache_manager.apply_move(move, game_state.color)
            current_board = existing_cache_manager.board  # Board gets updated by apply_move

        # Create final state with SAME cache manager
        from .gamestate import GameState as GS
        new_state = GS(
            board=current_board,
            color=game_state.color.opposite(),
            cache_manager=existing_cache_manager,  # REUSE manager
            history=(*game_state.history, *moves),
            halfmove_clock=game_state.halfmove_clock + len(moves),
            game_mode=game_state.game_mode,
            turn_number=game_state.turn_number + 1,
        )
        return new_state
