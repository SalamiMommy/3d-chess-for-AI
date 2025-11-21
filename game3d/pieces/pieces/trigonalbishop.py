"""Trigonal Bishop - 8 space-diagonal movement vectors."""
from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import COORD_DTYPE, SIZE_MINUS_1, Color, PieceType
from game3d.common.registry import register
from game3d.movement.slider_engine import get_slider_movement_generator
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Piece-specific movement vectors - 8 true 3D diagonals
TRIGONAL_BISHOP_VECTORS = np.array([
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
    [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
], dtype=COORD_DTYPE)

def generate_trigonal_bishop_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray,
    max_steps: int = SIZE_MINUS_1
) -> np.ndarray:
    """Generate trigonal bishop moves from numpy-native position array."""
    pos_arr = pos.astype(COORD_DTYPE)

    # Validate position using vectorized bounds check
    if not in_bounds_vectorized(pos_arr.reshape(1, 3))[0]:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Use slider engine for movement generation
    slider_engine = get_slider_movement_generator(cache_manager)
    moves = slider_engine.generate_slider_moves_array(
        color=color,
        pos=pos_arr,
        directions=TRIGONAL_BISHOP_VECTORS,
        max_distance=max_steps,
    )

    return moves

@register(PieceType.TRIGONALBISHOP)
def trigonal_bishop_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Registered dispatcher for Trigonal Bishop moves."""
    return generate_trigonal_bishop_moves(state.cache_manager, state.color, pos)

__all__ = ['TRIGONAL_BISHOP_VECTORS', 'generate_trigonal_bishop_moves', 'trigonal_bishop_move_dispatcher']
