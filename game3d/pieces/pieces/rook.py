"""Rook movement generator - orthogonal slider with 6 directions."""
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import COORD_DTYPE, Color, PieceType
from game3d.common.registry import register
from game3d.movement.slider_engine import get_slider_movement_generator
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import in_bounds_vectorized
from game3d.movement.movementmodifiers import get_range_modifier

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Piece-specific movement vectors - 6 orthogonal directions
ROOK_MOVEMENT_VECTORS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
], dtype=COORD_DTYPE)

def generate_rook_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray,
    max_steps: int = 8
) -> np.ndarray:
    """Generate rook moves from numpy-native position array."""
    pos_arr = pos.astype(COORD_DTYPE)

    # Validate position using vectorized bounds check
    if not in_bounds_vectorized(pos_arr.reshape(1, 3))[0]:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Use centralized slider generator with piece-specific vectors
    moves = get_slider_movement_generator(cache_manager).generate_slider_moves_array(
        color=color,
        pos=pos_arr,
        directions=ROOK_MOVEMENT_VECTORS,
        max_distance=max_steps,
    )

    return moves

@register(PieceType.ROOK)
def rook_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Registered dispatcher for Rook moves."""
    modifier = get_range_modifier(state, pos)
    max_steps = max(1, 8 + modifier)
    return generate_rook_moves(state.cache_manager, state.color, pos, max_steps)

__all__ = ['ROOK_MOVEMENT_VECTORS', 'generate_rook_moves', 'rook_move_dispatcher']
