"""Speeder – king-like mover + 1-sphere friendly buff."""

from __future__ import annotations
from typing import List, Set, TYPE_CHECKING
import numpy as np

from game3d.common.shared_types import *
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING: pass

def buffed_squares(
    cache_manager: 'OptimizedCacheManager',
    effect_color: int,
):
    """Get coordinates within 1-sphere of friendly Speeder pieces."""
    # Get all friendly pieces
    all_coords = cache_manager.occupancy_cache.get_positions(effect_color)
    if all_coords.size == 0:
        return set()
        
    # Filter for Speeders
    # ✅ OPTIMIZATION: Use unsafe access (coords from get_positions are valid)
    _, piece_types = cache_manager.occupancy_cache.batch_get_attributes_unsafe(all_coords)
    speeder_mask = piece_types == PieceType.SPEEDER
    speeder_coords = all_coords[speeder_mask]

    if speeder_coords.shape[0] == 0:
        return set()

    # Broadcast all Speeder positions with RADIUS_1_OFFSETS
    aura_coords = speeder_coords[:, np.newaxis, :] + RADIUS_1_OFFSETS
    aura_coords = aura_coords.reshape(-1, 3)

    # Vectorized bounds check
    in_bounds_mask = in_bounds_vectorized(aura_coords)
    valid_coords = aura_coords[in_bounds_mask]
    
    if valid_coords.shape[0] == 0:
        return set()

    # ✅ OPTIMIZATION: Vectorized color check using unsafe access
    # We only check valid_coords which are already bounds-checked
    colors, _ = cache_manager.occupancy_cache.batch_get_attributes_unsafe(valid_coords)
    
    # Filter for friendly pieces
    friendly_mask = (colors == effect_color)
    friendly_coords = valid_coords[friendly_mask]

    # Convert to set of bytes
    return {c.tobytes() for c in friendly_coords}

__all__ = ['buffed_squares']

