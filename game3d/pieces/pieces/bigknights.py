"""Big Knights - Extended knight pieces with (3,1) and (3,2) leap patterns."""
from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import (
    COORD_DTYPE, Color, PieceType, Result,
    KNIGHT31, KNIGHT32
)
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Piece-specific movement vectors - Knight31 (3,1,1) leap pattern
KNIGHT31_MOVEMENT_VECTORS = np.array([
    (3, 1, 1), (3, 1, -1), (3, -1, 1), (3, -1, -1),
    (-3, 1, 1), (-3, 1, -1), (-3, -1, 1), (-3, -1, -1),
    (1, 3, 1), (1, 3, -1), (1, -3, 1), (1, -3, -1),
    (-1, 3, 1), (-1, 3, -1), (-1, -3, 1), (-1, -3, -1),
    (1, 1, 3), (1, 1, -3), (1, -1, 3), (1, -1, -3),
    (-1, 1, 3), (-1, 1, -3), (-1, -1, 3), (-1, -1, -3),
], dtype=COORD_DTYPE)

# Piece-specific movement vectors - Knight32 (3,2,2) leap pattern
KNIGHT32_MOVEMENT_VECTORS = np.array([
    (3, 2, 2), (3, 2, -2), (3, -2, 2), (3, -2, -2),
    (-3, 2, 2), (-3, 2, -2), (-3, -2, 2), (-3, -2, -2),
    (2, 3, 2), (2, 3, -2), (2, -3, 2), (2, -3, -2),
    (-2, 3, 2), (-2, 3, -2), (-2, -3, 2), (-2, -3, -2),
    (2, 2, 3), (2, 2, -3), (2, -2, 3), (2, -2, -3),
    (-2, 2, 3), (-2, 2, -3), (-2, -2, 3), (-2, -2, -3),
], dtype=COORD_DTYPE)

def generate_knight31_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    pos_arr = pos.astype(COORD_DTYPE)

    # Validate position
    if not in_bounds_vectorized(pos_arr.reshape(1, 3))[0]:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Use jump engine with piece-specific vectors
    jump_engine = get_jump_movement_generator()
    return jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=KNIGHT31_MOVEMENT_VECTORS,
        allow_capture=True,
        piece_type=PieceType.KNIGHT31
    )

def generate_knight32_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    pos_arr = pos.astype(COORD_DTYPE)

    # Validate position
    if not in_bounds_vectorized(pos_arr.reshape(1, 3))[0]:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Use jump engine with piece-specific vectors
    jump_engine = get_jump_movement_generator()
    return jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=KNIGHT32_MOVEMENT_VECTORS,
        allow_capture=True,
        piece_type=PieceType.KNIGHT32
    )

@register(PieceType.KNIGHT31)
def knight31_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_knight31_moves(state.cache_manager, state.color, pos)

@register(PieceType.KNIGHT32)
def knight32_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_knight32_moves(state.cache_manager, state.color, pos)

__all__ = [
    'KNIGHT31_MOVEMENT_VECTORS',
    'KNIGHT32_MOVEMENT_VECTORS',
    'generate_knight31_moves',
    'generate_knight32_moves'
]
