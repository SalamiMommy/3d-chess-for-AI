# swapper.py - FULLY NUMPY-NATIVE
"""
Swapper == King-steps ∪ friendly-swap teleport
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from numba import njit

from game3d.common.shared_types import Color, PieceType, Result, get_empty_coord_batch, SWAPPER, COORD_DTYPE, SIZE

if TYPE_CHECKING: pass

from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS, BUFFED_KING_MOVEMENT_VECTORS

@njit(cache=True)
def _generate_swap_moves_kernel(
    swapper_positions: np.ndarray,
    friendly_positions: np.ndarray,
    friendly_types: np.ndarray
) -> np.ndarray:
    """Fused kernel to generate swap moves (swapper -> friendly piece)."""
    n_swappers = swapper_positions.shape[0]
    n_friendly = friendly_positions.shape[0]
    
    # Max moves = n_swappers * (n_friendly - 1)
    # But we allocate safely
    max_moves = n_swappers * n_friendly
    moves = np.empty((max_moves, 6), dtype=COORD_DTYPE)
    
    count = 0
    for i in range(n_swappers):
        sx, sy, sz = swapper_positions[i]
        
        for j in range(n_friendly):
            fx, fy, fz = friendly_positions[j]
            
            # Skip self (cannot swap with self)
            if sx == fx and sy == fy and sz == fz:
                continue
            
            # ✅ CRITICAL FIX: Prevent swapping with Wall entirely
            # Wall requires 2x2 space and moving a single part fragments it.
            # Note: Use integer value (24) for PieceType.WALL in Numba context
            if friendly_types[j] == 24:  # PieceType.WALL.value
                continue
                
            moves[count, 0] = sx
            moves[count, 1] = sy
            moves[count, 2] = sz
            moves[count, 3] = fx
            moves[count, 4] = fy
            moves[count, 5] = fz
            count += 1
            
    return moves[:count]

__all__ = []
