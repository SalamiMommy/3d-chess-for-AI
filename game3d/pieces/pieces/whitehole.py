# whitehole.py - OPTIMIZED NUMBA VERSION
"""White-Hole — moves like a Speeder and pushes enemies 1 step away."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numba import njit, prange

from game3d.common.shared_types import (
    Color, PieceType, Result, WHITEHOLE,
    BOOL_DTYPE, COORD_DTYPE, SIZE, SIZE_SQUARED, WHITEHOLE_PUSH_RADIUS
)
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING: pass

@njit(cache=True, fastmath=True, parallel=True)
def _push_candidates_numba(
    enemy_coords: np.ndarray,
    whitehole_coords: np.ndarray,
    flattened_occ: np.ndarray
) -> np.ndarray:
    """
    Fused kernel to find enemies pushed by whiteholes.
    Replaces broadcasting (N, 1, 3) - (1, M, 3) with parallel iteration.
    """
    n_enemies = enemy_coords.shape[0]
    n_holes = whitehole_coords.shape[0]
    
    # Store results: (enemy_idx, push_x, push_y, push_z)
    valid_mask = np.zeros(n_enemies, dtype=BOOL_DTYPE)
    push_targets = np.empty((n_enemies, 3), dtype=COORD_DTYPE)
    
    for i in prange(n_enemies):
        ex, ey, ez = enemy_coords[i]
        
        # Find closest whitehole
        min_dist = 999999
        closest_idx = -1
        
        for j in range(n_holes):
            hx, hy, hz = whitehole_coords[j]
            
            # Chebyshev distance
            dx = abs(ex - hx)
            dy = abs(ey - hy)
            dz = abs(ez - hz)
            dist = max(dx, max(dy, dz))
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = j
            elif dist == min_dist:
                pass
                
        if closest_idx != -1 and min_dist <= WHITEHOLE_PUSH_RADIUS:
            # Calculate push
            hx, hy, hz = whitehole_coords[closest_idx]
            
            # Direction: sign(enemy - hole) (Push AWAY)
            dx = ex - hx
            dy = ey - hy
            dz = ez - hz
            
            # Sign
            sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
            sy = 1 if dy > 0 else (-1 if dy < 0 else 0)
            sz = 1 if dz > 0 else (-1 if dz < 0 else 0)
            
            px, py, pz = ex + sx, ey + sy, ez + sz
            
            # Check bounds
            if 0 <= px < SIZE and 0 <= py < SIZE and 0 <= pz < SIZE:
                # Check occupancy (must be empty)
                idx = px + SIZE * py + SIZE_SQUARED * pz
                if flattened_occ[idx] == 0:
                    valid_mask[i] = True
                    push_targets[i, 0] = px
                    push_targets[i, 1] = py
                    push_targets[i, 2] = pz

    # Collect results
    count = 0
    for i in range(n_enemies):
        if valid_mask[i]:
            count += 1
            
    if count == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)
        
    out = np.empty((count, 6), dtype=COORD_DTYPE)
    idx_out = 0
    for i in range(n_enemies):
        if valid_mask[i]:
            out[idx_out, 0] = enemy_coords[i, 0]
            out[idx_out, 1] = enemy_coords[i, 1]
            out[idx_out, 2] = enemy_coords[i, 2]
            out[idx_out, 3] = push_targets[i, 0]
            out[idx_out, 4] = push_targets[i, 1]
            out[idx_out, 5] = push_targets[i, 2]
            idx_out += 1
            
    return out

def push_candidates_vectorized(
    cache_manager: 'OptimizedCacheManager',
    controller: Color
) -> np.ndarray:
    """Find enemies to push away from whiteholes - optimized with Numba."""
    # Get all friendly pieces
    all_coords = cache_manager.occupancy_cache.get_positions(controller)
    if all_coords.size == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)
    
    # Filter for whiteholes
    types = cache_manager.occupancy_cache.batch_get_types_only(all_coords)
    whitehole_mask = (types == PieceType.WHITEHOLE.value)
    
    if not np.any(whitehole_mask):
        return np.empty((0, 6), dtype=COORD_DTYPE)
    
    whitehole_coords = all_coords[whitehole_mask]

    # Get enemy coordinates
    enemy_color = controller.opposite()
    enemy_coords = cache_manager.occupancy_cache.get_positions(enemy_color)
    
    if enemy_coords.size == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # ✅ CRITICAL FIX: Filter out Walls AND Kings (immune to physics)
    # Walls are 2x2 structures and cannot be moved point-wise by physics
    # Kings cannot be forcibly moved by physics effects
    enemy_types = cache_manager.occupancy_cache.batch_get_types_only(enemy_coords)
    immune_mask = (enemy_types != PieceType.WALL.value) & (enemy_types != PieceType.KING.value)
    
    if not np.any(immune_mask):
        return np.empty((0, 6), dtype=COORD_DTYPE)
        
    enemy_coords = enemy_coords[immune_mask]

    # Run fused kernel
    flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
    return _push_candidates_numba(enemy_coords, whitehole_coords, flattened_occ)

__all__ = ['push_candidates_vectorized']

