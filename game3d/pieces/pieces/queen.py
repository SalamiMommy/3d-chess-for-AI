"""Queen movement generator - combines rook and bishop movement."""
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.coord_utils import CoordinateUtils
from game3d.common.shared_types import COORD_DTYPE, PieceType
from game3d.common.registry import register
from game3d.movement.slider_engine import get_slider_movement_generator
from game3d.movement.movepiece import Move
from game3d.movement.movementmodifiers import get_range_modifier

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager

# Piece-specific movement vectors - queen combines orthogonal + diagonal
QUEEN_MOVEMENT_VECTORS = np.array([
    # Orthogonal directions (6)
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    # Diagonal directions (12)
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
    [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
    [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]
], dtype=COORD_DTYPE)

def generate_queen_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray,
    max_steps: int = 8
) -> np.ndarray:
    """Generate queen moves using numpy-native operations."""
    pos_arr = pos.astype(COORD_DTYPE)

    # Validate position
    if not CoordinateUtils.in_bounds(pos_arr):
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Use integrated slider generator with queen-specific vectors
    slider_engine = get_slider_movement_generator()
    return slider_engine.generate_slider_moves_array(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=QUEEN_MOVEMENT_VECTORS,
        max_distance=max_steps,
    )

@register(PieceType.QUEEN)
def queen_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Dispatcher for queen moves - receives numpy array position."""
    modifier = get_range_modifier(state, pos)
    max_steps = max(1, 8 + modifier)
    return generate_queen_moves(state.cache_manager, state.color, pos, max_steps)

__all__ = ['QUEEN_MOVEMENT_VECTORS', 'generate_queen_moves']
