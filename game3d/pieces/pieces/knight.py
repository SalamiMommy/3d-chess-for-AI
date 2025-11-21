"""Knight movement generator - 3D leaper with 24 movement vectors."""
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import COORD_DTYPE, Color, PieceType
from game3d.common.registry import register
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Piece-specific movement vectors - 24 knight movement patterns
KNIGHT_MOVEMENT_VECTORS = np.array([
    [1, 2, 0], [2, 1, 0], [-1, 2, 0], [-2, 1, 0],
    [1, -2, 0], [2, -1, 0], [-1, -2, 0], [-2, -1, 0],
    [0, 1, 2], [0, 2, 1], [0, -1, 2], [0, -2, 1],
    [0, 1, -2], [0, 2, -1], [0, -1, -2], [0, -2, -1],
    [1, 0, 2], [2, 0, 1], [-1, 0, 2], [-2, 0, 1],
    [1, 0, -2], [2, 0, -1], [-1, 0, -2], [-2, 0, -1]
], dtype=COORD_DTYPE)

def generate_knight_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    pos_arr = pos.astype(COORD_DTYPE)

    # Use centralized bounds checking from coord_utils
    if not in_bounds_vectorized(pos_arr.reshape(1, 3))[0]:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Use jump engine with piece-specific movement vectors
    jump_engine = get_jump_movement_generator(cache_manager)
    return jump_engine.generate_jump_moves(
        color=color,
        pos=pos_arr,
        directions=KNIGHT_MOVEMENT_VECTORS,
        allow_capture=True,
    )

@register(PieceType.KNIGHT)
def knight_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_knight_moves(state.cache_manager, state.color, pos)

__all__ = ['KNIGHT_MOVEMENT_VECTORS', 'generate_knight_moves']
