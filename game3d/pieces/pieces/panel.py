"""Panel movement generator - teleport on same x/y/z plane + king moves."""
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import COORD_DTYPE, SIZE, SIZE_MINUS_1
from game3d.common.shared_types import Color, PieceType
from game3d.common.registry import register
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Piece-specific movement vectors - 3x3 panels at distance 2 (unbuffed) and 3 (buffed)

def _create_panel_vectors(distance):
    vectors = []
    r = [-1, 0, 1]
    
    # X faces
    for x in [-distance, distance]:
        for y in r:
            for z in r:
                vectors.append([x, y, z])
                
    # Y faces
    for y in [-distance, distance]:
        for x in r:
            for z in r:
                vectors.append([x, y, z])
                
    # Z faces
    for z in [-distance, distance]:
        for x in r:
            for y in r:
                vectors.append([x, y, z])
                
    return np.array(vectors, dtype=COORD_DTYPE)

PANEL_MOVEMENT_VECTORS = _create_panel_vectors(2)
BUFFED_PANEL_MOVEMENT_VECTORS = _create_panel_vectors(3)

def generate_panel_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    pos_arr = pos.astype(COORD_DTYPE)

    # Validate position only for single piece input
    if pos_arr.ndim == 1:
        if not in_bounds_vectorized(pos_arr.reshape(1, 3))[0]:
            return np.empty((0, 6), dtype=COORD_DTYPE)

    # Use jump generator with piece-specific vectors
    jump_engine = get_jump_movement_generator()
    return jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=PANEL_MOVEMENT_VECTORS,
        allow_capture=True,
        piece_type=None,
        buffed_directions=BUFFED_PANEL_MOVEMENT_VECTORS
    )

@register(PieceType.PANEL)
def panel_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_panel_moves(state.cache_manager, state.color, pos)

__all__ = ['PANEL_MOVEMENT_VECTORS', 'generate_panel_moves']
