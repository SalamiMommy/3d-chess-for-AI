# game3d/movement/pieces/friendlytp.py - OPTIMIZED NUMBA VERSION
"""
Friendly-Teleporter – teleport to any empty neighbour of a friendly piece
PLUS normal 1-step King moves.
"""

from __future__ import annotations
import numpy as np
from numba import njit, prange
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import Color, PieceType, Result, get_empty_coord_batch
from game3d.common.shared_types import COORD_DTYPE, SIZE, SIZE_SQUARED, BOOL_DTYPE
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS, BUFFED_KING_MOVEMENT_VECTORS

@njit(cache=True, fastmath=True)
def _generate_friendlytp_moves_fused(
    teleporter_positions: np.ndarray,
    friendly_coords: np.ndarray,
    flattened_occ: np.ndarray,
    directions: np.ndarray,
    occ_3d: np.ndarray,
    my_color: int
) -> np.ndarray:
    """
    Fused kernel:
    1. Finds all valid network squares (empty neighbors of friendly pieces)
    2. Generates moves from teleporters to these squares
    3. Generates standard King moves (1 step) for teleporters
    """
    n_tps = teleporter_positions.shape[0]
    n_friendly = friendly_coords.shape[0]
    n_dirs = directions.shape[0]
    
    # 1. Identify network squares using a boolean mask
    network_mask = np.zeros(SIZE * SIZE_SQUARED, dtype=BOOL_DTYPE)
    network_count = 0
    
    for i in range(n_friendly):
        fx, fy, fz = friendly_coords[i]
        
        for j in range(n_dirs):
            dx, dy, dz = directions[j]
            tx, ty, tz = fx + dx, fy + dy, fz + dz
            
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                idx = tx + SIZE * ty + SIZE_SQUARED * tz
                
                # Must be empty and not already marked
                if flattened_occ[idx] == 0 and not network_mask[idx]:
                    network_mask[idx] = True
                    network_count += 1
    
    # Calculate max moves
    # Teleports: n_tps * network_count
    # King moves: n_tps * 26 (upper bound)
    max_moves = n_tps * (network_count + 26)
    moves = np.empty((max_moves, 6), dtype=COORD_DTYPE)
    
    count = 0
    
    # 2. Generate Network Teleports
    if network_count > 0:
        # Iterate through mask to find targets
        for idx in range(SIZE * SIZE_SQUARED):
            if network_mask[idx]:
                # Decode index
                z = idx // SIZE_SQUARED
                rem = idx % SIZE_SQUARED
                y = rem // SIZE
                x = rem % SIZE
                
                for i in range(n_tps):
                    sx, sy, sz = teleporter_positions[i]
                    
                    # Skip self-teleport
                    if sx == x and sy == y and sz == z:
                        continue
                        
                    moves[count, 0] = sx
                    moves[count, 1] = sy
                    moves[count, 2] = sz
                    moves[count, 3] = x
                    moves[count, 4] = y
                    moves[count, 5] = z
                    count += 1

    # 3. Generate King Moves
    for i in range(n_tps):
        sx, sy, sz = teleporter_positions[i]
        
        for j in range(n_dirs):
            dx, dy, dz = directions[j]
            tx, ty, tz = sx + dx, sy + dy, sz + dz
            
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                # Check occupancy
                target_color = occ_3d[tx, ty, tz]
                if target_color != my_color: # Empty or Enemy (Capture)
                    moves[count, 0] = sx
                    moves[count, 1] = sy
                    moves[count, 2] = sz
                    moves[count, 3] = tx
                    moves[count, 4] = ty
                    moves[count, 5] = tz
                    count += 1

    return moves[:count]

def generate_friendlytp_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate friendly teleporter moves: king walks + network teleports."""
    pos_arr = pos.astype(COORD_DTYPE)
    
    # Handle single input by reshaping
    if pos_arr.ndim == 1:
        pos_arr = pos_arr.reshape(1, 3)
        
    if pos_arr.shape[0] == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)
    
    # Get all friendly pieces
    friendly_coords = cache_manager.occupancy_cache.get_positions(color)
    flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
    occ_3d = cache_manager.occupancy_cache._occ
    
    # Use single fused kernel for everything
    return _generate_friendlytp_moves_fused(
        pos_arr, 
        friendly_coords, 
        flattened_occ,
        KING_MOVEMENT_VECTORS,
        occ_3d,
        color
    )

@register(PieceType.FRIENDLYTELEPORTER)
def friendlytp_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_friendlytp_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_friendlytp_moves"]
