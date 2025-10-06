"""
Network Teleporter — teleport to any empty square adjacent to any friendly piece.
Optimized using existing caches and batch processing.
"""

from __future__ import annotations

from typing import List, Set
import numpy as np
import torch

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.cache.manager import CacheManager
from game3d.common.common import in_bounds

# Precomputed neighbor directions (26 directions in 3D space)
_NEIGHBOR_DIRECTIONS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == dy == dz == 0)
], dtype=np.int8)

def generate_network_teleport_moves(
    cache: CacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List['Move']:
    """
    Generate all legal network-teleport moves from (x, y, z) using batch processing.

    Optimizations:
    - Avoid Python-level loops for neighbor finding and deduplication.
    - Use numpy arrays for all geometry math and occupancy checks.
    - Avoid set and list conversions except at the very end.
    - Short-circuit early if no friendly pieces, no empty targets, etc.
    - Batch handoff to jump movement generator.
    """
    start = (x, y, z)

    # Fast batch: Get all friendly piece positions as a numpy array
    friendly_positions = np.empty((0, 3), dtype=np.int8)
    if hasattr(cache.piece_cache, "iter_color"):
        # Use list comprehension for direct numpy construction
        friendly_positions = np.array(
            [coord for coord, _ in cache.piece_cache.iter_color(color)],
            dtype=np.int8
        )
    if friendly_positions.shape[0] == 0:
        return []

    # Occupancy mask – direct reference, don't copy
    occupancy_mask = cache.occupancy.mask

    # Batch neighbor generation: vectorized addition
    # friendly_positions: (N, 3), _NEIGHBOR_DIRECTIONS: (26, 3)
    # Output: (N, 26, 3)
    all_neighbors = friendly_positions[:, np.newaxis, :] + _NEIGHBOR_DIRECTIONS
    neighbors_flat = all_neighbors.reshape(-1, 3)

    # Filter out-of-bounds positions (logical AND, vectorized)
    valid_mask = (
        (neighbors_flat[:, 0] >= 0) & (neighbors_flat[:, 0] < 9) &
        (neighbors_flat[:, 1] >= 0) & (neighbors_flat[:, 1] < 9) &
        (neighbors_flat[:, 2] >= 0) & (neighbors_flat[:, 2] < 9)
    )
    valid_neighbors = neighbors_flat[valid_mask]

    if valid_neighbors.shape[0] == 0:
        return []

    # Batch occupancy check: not occupied (mask is True if occupied)
    z_idx, y_idx, x_idx = valid_neighbors.T
    empty_mask = ~occupancy_mask[z_idx, y_idx, x_idx]
    empty_neighbors = valid_neighbors[empty_mask]

    if empty_neighbors.shape[0] == 0:
        return []

    # Deduplicate targets with numpy.unique (avoid Python sets)
    unique_targets = np.unique(empty_neighbors, axis=0)
    if unique_targets.shape[0] == 0:
        return []

    # Exclude the starting position (can't teleport to self)
    not_self_mask = ~np.all(unique_targets == np.array(start), axis=1)
    unique_targets = unique_targets[not_self_mask]
    if unique_targets.shape[0] == 0:
        return []

    # Compute direction vectors for jump generator (batch)
    start_array = np.array(start, dtype=np.int8)
    directions = unique_targets - start_array

    # Hand off to jump movement generator (batch, fast)
    gen = get_integrated_jump_movement_generator(cache)
    return gen.generate_jump_moves(
        color=color,
        pos=start,
        directions=directions,
        allow_capture=False,
    )
