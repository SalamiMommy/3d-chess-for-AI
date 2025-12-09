# game3d/movement/pieces/archer.py - FULLY NUMPY-NATIVE
"""
Unified Archer dispatcher
- 1-radius sphere  → walk (normal king-like move)
- 2-radius surface → shoot (archery capture, no movement)
"""

from __future__ import annotations
import numpy as np
from numba import njit, prange
from typing import TYPE_CHECKING
from game3d.common.shared_types import *
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING: pass

from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS, BUFFED_KING_MOVEMENT_VECTORS

# Archery directions (2-radius surface only) - optimized numpy construction
coords = np.mgrid[-2:3, -2:3, -2:3].reshape(3, -1).T
distances = np.sum(coords * coords, axis=1)
_ARCHERY_DIRECTIONS = coords[distances == 4].astype(COORD_DTYPE)

@njit(cache=True, fastmath=True)
def _generate_archer_shots_kernel(
    starts: np.ndarray,
    directions: np.ndarray,
    flattened_occ: np.ndarray,
    color: int
) -> np.ndarray:
    """
    Numba kernel to generate archer shots.
    Filters bounds and enemy occupancy inline to avoid large intermediate arrays.
    """
    n_starts = starts.shape[0]
    n_dirs = directions.shape[0]
    
    # Pre-calculate max possible moves to allocate once
    max_moves = n_starts * n_dirs
    moves = np.empty((max_moves, 6), dtype=COORD_DTYPE)
    
    count = 0
    
    for i in range(n_starts):
        sx, sy, sz = starts[i]
        
        for j in range(n_dirs):
            dx, dy, dz = directions[j]
            tx, ty, tz = sx + dx, sy + dy, sz + dz
            
            # Bounds check
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                # Occupancy check
                idx = tx + SIZE * ty + SIZE_SQUARED * tz
                occ = flattened_occ[idx]
                
                # Must be enemy
                if occ != 0 and occ != color:
                    moves[count, 0] = sx
                    moves[count, 1] = sy
                    moves[count, 2] = sz
                    moves[count, 3] = tx
                    moves[count, 4] = ty
                    moves[count, 5] = tz
                    count += 1
                    
    return moves[:count]



__all__ = []

