# geomancer.py - OPTIMIZED NUMBA VERSION
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numba import njit, prange

from game3d.common.shared_types import *
from game3d.common.coord_utils import in_bounds_vectorized, CoordinateUtils

if TYPE_CHECKING: pass

# Pre-compute geomancy offsets - Radius 3 (unbuffed, Cheb dist >= 2)
_OFFSETS_R3 = np.asarray(RADIUS_3_OFFSETS, dtype=COORD_DTYPE)
_CHEB_DIST_R3 = np.max(np.abs(_OFFSETS_R3), axis=1)
_GEOMANCY_MASK_R3 = _CHEB_DIST_R3 >= 2
GEOMANCY_OFFSETS = _OFFSETS_R3[_GEOMANCY_MASK_R3]

# Pre-compute geomancy offsets - Radius 4 (buffed, Cheb dist >= 2)
# Generate radius 4 offsets manually since RADIUS_4_OFFSETS doesn't exist
_OFFSETS_R4 = np.array([
    [-4, -1, 0],
    [-4, 0, -1],
    [-4, 0, 0],
    [-4, 0, 1],
    [-4, 1, 0],
    [-3, -2, -2],
    [-3, -2, 2],
    [-3, 2, -2],
    [-3, 2, 2],
    [-2, -3, -2],
    [-2, -3, 2],
    [-2, -2, -3],
    [-2, -2, 3],
    [-2, 2, -3],
    [-2, 2, 3],
    [-2, 3, -2],
    [-2, 3, 2],
    [-1, -4, 0],
    [-1, 0, -4],
    [-1, 0, 4],
    [-1, 4, 0],
    [0, -4, -1],
    [0, -4, 0],
    [0, -4, 1],
    [0, -1, -4],
    [0, -1, 4],
    [0, 0, -4],
    [0, 0, 4],
    [0, 1, -4],
    [0, 1, 4],
    [0, 4, -1],
    [0, 4, 0],
    [0, 4, 1],
    [1, -4, 0],
    [1, 0, -4],
    [1, 0, 4],
    [1, 4, 0],
    [2, -3, -2],
    [2, -3, 2],
    [2, -2, -3],
    [2, -2, 3],
    [2, 2, -3],
    [2, 2, 3],
    [2, 3, -2],
    [2, 3, 2],
    [3, -2, -2],
    [3, -2, 2],
    [3, 2, -2],
    [3, 2, 2],
    [4, -1, 0],
    [4, 0, -1],
    [4, 0, 0],
    [4, 0, 1],
    [4, 1, 0],
], dtype=COORD_DTYPE)
_CHEB_DIST_R4 = np.max(np.abs(_OFFSETS_R4), axis=1)
_GEOMANCY_MASK_R4 = _CHEB_DIST_R4 >= 2
BUFFED_GEOMANCY_OFFSETS = _OFFSETS_R4[_GEOMANCY_MASK_R4]

@njit(cache=True, fastmath=True)
def _block_candidates_numba(
    geomancer_coords: np.ndarray, 
    flattened_occ: np.ndarray,
    offsets: np.ndarray
) -> np.ndarray:
    """
    Fused kernel to find empty squares that geomancers can block.
    Replaces: broadcasting + bounds check + occupancy check + unique.
    """
    n_geom = geomancer_coords.shape[0]
    n_offsets = offsets.shape[0]
    
    # Use boolean mask for deduplication (SIZE^3 is small enough)
    mask = np.zeros(SIZE * SIZE_SQUARED, dtype=BOOL_DTYPE)
    
    for i in range(n_geom):
        gx, gy, gz = geomancer_coords[i]
        
        for j in range(n_offsets):
            dx, dy, dz = offsets[j]
            tx, ty, tz = gx + dx, gy + dy, gz + dz
            
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                idx = tx + SIZE * ty + SIZE_SQUARED * tz
                if flattened_occ[idx] == 0:
                    mask[idx] = True
                    
    # Collect results
    count = 0
    for i in range(SIZE * SIZE_SQUARED):
        if mask[i]:
            count += 1
            
    if count == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)
        
    out = np.empty((count, 3), dtype=COORD_DTYPE)
    idx_out = 0
    for i in range(SIZE * SIZE_SQUARED):
        if mask[i]:
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

def block_candidates_numpy(
    cache_manager: 'OptimizedCacheManager',
    mover_color: 'Color',
):
    """
    Return empty squares that <mover_color> may block via geomancy this turn.
    Returns array of shape (N, 3).
    """
    # Get all geomancer coordinates
    # We can filter by type efficiently if we had a type-specific list, 
    # but for now we filter from all pieces of color.
    pieces = cache_manager.occupancy_cache.get_positions(mover_color)
    if pieces.shape[0] == 0:
        return get_empty_coord_batch()

    # Filter for geomancers
    # Use batch_get_types_only for speed
    types = cache_manager.occupancy_cache.batch_get_types_only(pieces)
    geomancer_mask = (types == PieceType.GEOMANCER.value)
    
    if not np.any(geomancer_mask):
        return get_empty_coord_batch()
        
    geomancer_coords = pieces[geomancer_mask]

    # Run fused kernel
    flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
    offsets = np.asarray(RADIUS_3_OFFSETS, dtype=COORD_DTYPE)
    
    return _block_candidates_numba(geomancer_coords, flattened_occ, offsets)

@njit(cache=True, fastmath=True)
def _generate_geomancy_moves_kernel(
    starts: np.ndarray,
    flattened_occ: np.ndarray,
    offsets: np.ndarray
) -> np.ndarray:
    """Fused kernel for geomancy moves (radius 2/3 placement) for a batch of pieces."""
    n_starts = starts.shape[0]
    n_offsets = offsets.shape[0]
    
    # Max moves = n_starts * n_offsets
    max_moves = n_starts * n_offsets
    moves = np.empty((max_moves, 6), dtype=COORD_DTYPE)
    count = 0
    
    for i in range(n_starts):
        sx, sy, sz = starts[i]
        
        for j in range(n_offsets):
            dx, dy, dz = offsets[j]
            tx, ty, tz = sx + dx, sy + dy, sz + dz
            
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                idx = tx + SIZE * ty + SIZE_SQUARED * tz
                if flattened_occ[idx] == 0:
                    moves[count, 0] = sx
                    moves[count, 1] = sy
                    moves[count, 2] = sz
                    moves[count, 3] = tx
                    moves[count, 4] = ty
                    moves[count, 5] = tz
                    count += 1
                
    return moves[:count]

    """Generate geomancer moves: radius-1 king moves + radius-2/3 (unbuffed) or radius-2/4 (buffed) geomancy placement moves."""
    start = pos.astype(COORD_DTYPE)
    
    # Handle single input
    if start.ndim == 1:
        start = start.reshape(1, 3)

    # Generate king moves for piece movement within radius 1
    king_moves = generate_king_moves(cache_manager, color, start, piece_type=PieceType.GEOMANCER)
    
    # Check if buffed
    buffed_squares = cache_manager.consolidated_aura_cache._buffed_squares
    x, y, z = start[0]
    is_buffed = buffed_squares[x, y, z]
    
    # Use appropriate offsets based on buff status
    offsets = BUFFED_GEOMANCY_OFFSETS if is_buffed else GEOMANCY_OFFSETS
    
    # Generate geomancy moves (radius 2/3 unbuffed, or 2/4 buffed)
    flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
    geom_moves = _generate_geomancy_moves_kernel(start, flattened_occ, offsets)
    
    if geom_moves.shape[0] == 0:
        return king_moves
        
    if king_moves.shape[0] == 0:
        return geom_moves
        
    return np.concatenate((king_moves, geom_moves))

__all__ = ['block_candidates_numpy']

