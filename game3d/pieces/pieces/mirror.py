# game3d/movement/pieces/mirror.py - Mirror piece implementation
"""
Mirror-Teleporter: Teleports to mirrored position across board center.
"""

from __future__ import annotations
import numpy as np
from numba import njit
from typing import List, TYPE_CHECKING
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, SIZE_MINUS_1, SIZE, SIZE_SQUARED, BOOL_DTYPE
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

@njit(cache=True)
def _generate_mirror_moves_kernel(
    positions: np.ndarray,
    flattened_occ: np.ndarray,
    color: int
) -> np.ndarray:
    """
    Generate mirror moves for a batch of positions.
    """
    n = positions.shape[0]
    moves = np.empty((n, 6), dtype=COORD_DTYPE)
    count = 0
    
    for i in range(n):
        sx, sy, sz = positions[i]
        
        # Calculate mirrored target
        tx = SIZE_MINUS_1 - sx
        ty = SIZE_MINUS_1 - sy
        tz = SIZE_MINUS_1 - sz
        
        # Skip if target is same as start (center of board)
        if sx == tx and sy == ty and sz == tz:
            continue
            
        # Check bounds (should always be in bounds if start is in bounds, but good to be safe)
        if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
            idx = tx + SIZE * ty + SIZE_SQUARED * tz
            occ = flattened_occ[idx]
            
            # Check occupancy: must be empty (0) or enemy (!= color)
            if occ == 0 or occ != color:
                moves[count, 0] = sx
                moves[count, 1] = sy
                moves[count, 2] = sz
                moves[count, 3] = tx
                moves[count, 4] = ty
                moves[count, 5] = tz
                count += 1
                
    return moves[:count]

def generate_mirror_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate mirror teleport moves to mirrored position."""
    pos_arr = pos.astype(COORD_DTYPE)
    
    # Handle single input
    if pos_arr.ndim == 1:
        pos_arr = pos_arr.reshape(1, 3)
        
    # Get flattened occupancy
    flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
    
    # Generate moves using kernel
    return _generate_mirror_moves_kernel(pos_arr, flattened_occ, color)

@register(PieceType.MIRROR)
def mirror_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Dispatch mirror piece moves."""
    return generate_mirror_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_mirror_moves"]
