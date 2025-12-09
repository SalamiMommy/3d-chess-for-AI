"""Slower piece - king-like mover with 2-sphere enemy debuff aura."""
from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.shared_types import Color, PieceType, SLOWER, COORD_DTYPE, RADIUS_2_OFFSETS
from game3d.common.coord_utils import in_bounds_vectorized
from numba import njit
from game3d.common.shared_types import SIZE, SIZE_SQUARED

if TYPE_CHECKING: pass

def get_debuffed_squares(
    cache_manager: 'OptimizedCacheManager',
    effect_color: int,
):
    """
    Get squares within 2-sphere of friendly SLOWER pieces that affect enemies.
    Returns array of shape (N, 3) containing affected square coordinates.
    """
    # ✅ OPTIMIZATION: Use vectorized get_positions and batch_get_attributes_unsafe
    all_coords = cache_manager.occupancy_cache.get_positions(effect_color)
    if all_coords.size == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)
        
    _, piece_types = cache_manager.occupancy_cache.batch_get_attributes_unsafe(all_coords)
    slower_mask = piece_types == PieceType.SLOWER
    effect_pieces = all_coords[slower_mask]

    if effect_pieces.shape[0] == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)

    # ✅ OPTIMIZATION: Use fused Numba kernel
    flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
    return _get_slower_debuff_squares_fast(
        effect_pieces, flattened_occ, effect_color, RADIUS_2_OFFSETS
    )

@njit(cache=True)
def _get_slower_debuff_squares_fast(
    slower_positions: np.ndarray,
    flattened_occ: np.ndarray,
    effect_color: int,
    radius_offsets: np.ndarray
) -> np.ndarray:
    """Fused kernel to find enemy pieces within radius 2 of slower pieces.
    
    Replaces:
    1. Expansion (slower + offsets)
    2. Bounds checking
    3. Occupancy lookup
    4. Enemy filtering
    5. Deduplication (via boolean mask)
    """
    # Use a boolean map to track unique squares
    # Since board is small (9x9x9 = 729), a boolean map is efficient.
    affected_mask = np.zeros(SIZE_SQUARED * SIZE, dtype=np.bool_)
    
    n_slower = slower_positions.shape[0]
    n_offsets = radius_offsets.shape[0]
    
    for i in range(n_slower):
        sx, sy, sz = slower_positions[i]
        
        for j in range(n_offsets):
            dx, dy, dz = radius_offsets[j]
            tx, ty, tz = sx + dx, sy + dy, sz + dz
            
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                idx = tx + SIZE * ty + SIZE_SQUARED * tz
                occ = flattened_occ[idx]
                
                # Check for enemy piece (not empty, not friendly)
                if occ != 0 and occ != effect_color:
                    affected_mask[idx] = True
                    
    # Count unique affected squares
    count = 0
    for i in range(SIZE_SQUARED * SIZE):
        if affected_mask[i]:
            count += 1
            
    if count == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)
            
    # Collect coordinates
    out = np.empty((count, 3), dtype=COORD_DTYPE)
    idx_out = 0
    for i in range(SIZE_SQUARED * SIZE):
        if affected_mask[i]:
            # Decode index
            z = i // SIZE_SQUARED
            rem = i % SIZE_SQUARED
            y = rem // SIZE
            x = rem % SIZE
            out[idx_out, 0] = x
            out[idx_out, 1] = y
            out[idx_out, 2] = z
            idx_out += 1
            
    return out

__all__ = ['get_debuffed_squares']

