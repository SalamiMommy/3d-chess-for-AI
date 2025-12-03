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
    color: int,
    mirror_all_axes: bool
) -> np.ndarray:
    """
    Generate mirror moves for a batch of positions.
    
    Args:
        positions: Starting positions
        flattened_occ: Flattened occupancy array
        color: Piece color
        mirror_all_axes: If True, mirror x,y,z. If False, only mirror z.
    """
    n = positions.shape[0]
    moves = np.empty((n, 6), dtype=COORD_DTYPE)
    count = 0
    
    for i in range(n):
        sx, sy, sz = positions[i]
        
        # Calculate mirrored target based on buff status
        if mirror_all_axes:
            # Buffed: mirror all three axes
            tx = SIZE_MINUS_1 - sx
            ty = SIZE_MINUS_1 - sy
            tz = SIZE_MINUS_1 - sz
        else:
            # Unbuffed: only mirror z axis
            tx = sx
            ty = sy
            tz = SIZE_MINUS_1 - sz
        
        # Skip if target is same as start
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

from game3d.pieces.pieces.kinglike import generate_king_moves

def generate_mirror_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate mirror moves: King-like moves + Teleport (z-mirror unbuffed, xyz-mirror buffed)."""
    pos_arr = pos.astype(COORD_DTYPE)
    
    # Handle single input
    if pos_arr.ndim == 1:
        pos_arr = pos_arr.reshape(1, 3)
        
    moves_list = []

    # 1. King-like movement (handles buffs)
    king_moves = generate_king_moves(cache_manager, color, pos_arr, piece_type=PieceType.MIRROR)
    if king_moves.size > 0:
        moves_list.append(king_moves)

    # 2. Mirror Teleport
    # Check if buffed
    buffed_squares = cache_manager.consolidated_aura_cache._buffed_squares
    x, y, z = pos_arr[0]
    is_buffed = buffed_squares[x, y, z]
    
    # Get flattened occupancy
    flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
    
    # Generate moves using kernel (mirror all axes if buffed, only z if not)
    teleport_moves = _generate_mirror_moves_kernel(pos_arr, flattened_occ, color, is_buffed)
    if teleport_moves.size > 0:
        moves_list.append(teleport_moves)

    if not moves_list:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    return np.concatenate(moves_list, axis=0)

@register(PieceType.MIRROR)
def mirror_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Dispatch mirror piece moves."""
    return generate_mirror_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_mirror_moves"]
