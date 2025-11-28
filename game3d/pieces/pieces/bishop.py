"""Bishop movement generator - diagonal slider."""
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.coord_utils import CoordinateUtils
from game3d.common.shared_types import COORD_DTYPE, PieceType, MAX_STEPS_SLIDER
from game3d.common.registry import register
from game3d.movement.slider_engine import get_slider_movement_generator
from game3d.movement.movepiece import Move
from game3d.movement.movementmodifiers import get_range_modifier

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager

# Piece-specific movement vectors - 12 diagonal directions for 3D
BISHOP_MOVEMENT_VECTORS = np.array([
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
    [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
    [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]
], dtype=COORD_DTYPE)

def generate_bishop_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray,
    max_steps: int = MAX_STEPS_SLIDER,
    ignore_occupancy: bool = False
) -> np.ndarray:
    """Generate bishop moves using numpy-native operations."""
    pos_arr = pos.astype(COORD_DTYPE)

    # Validate position
    if not CoordinateUtils.in_bounds(pos_arr):
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Use slider engine with piece-specific movement vectors
    slider_engine = get_slider_movement_generator()
    return slider_engine.generate_slider_moves_array(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=BISHOP_MOVEMENT_VECTORS,
        max_distance=max_steps,
        ignore_occupancy=ignore_occupancy
    )

@register(PieceType.BISHOP)
def bishop_move_dispatcher(state: 'GameState', pos: np.ndarray, ignore_occupancy: bool = False) -> np.ndarray:
    """Dispatcher for bishop moves - receives numpy array position."""
    modifier = get_range_modifier(state, pos)
    max_steps = max(1, MAX_STEPS_SLIDER + modifier)
    return generate_bishop_moves(state.cache_manager, state.color, pos, max_steps, ignore_occupancy)

__all__ = ['BISHOP_MOVEMENT_VECTORS', 'generate_bishop_moves']
