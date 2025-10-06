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
    """Generate all legal network-teleport moves from (x, y, z) using batch processing."""
    start = (x, y, z)

    # Get all friendly piece positions as a tensor for batch processing
    friendly_positions = np.array([coord for coord, _ in cache.piece_cache.iter_color(color)], dtype=np.int8)

    if len(friendly_positions) == 0:
        return []

    # Get occupancy mask directly
    occupancy_mask = cache.occupancy.mask

    # Precompute all possible neighbor positions at once
    all_neighbors = friendly_positions[:, np.newaxis, :] + _NEIGHBOR_DIRECTIONS
    all_neighbors = all_neighbors.reshape(-1, 3)  # Flatten to (N*26, 3)

    # Filter out-of-bounds positions
    valid_mask = (
        (all_neighbors[:, 0] >= 0) & (all_neighbors[:, 0] < 9) &
        (all_neighbors[:, 1] >= 0) & (all_neighbors[:, 1] < 9) &
        (all_neighbors[:, 2] >= 0) & (all_neighbors[:, 2] < 9)
    )
    valid_neighbors = all_neighbors[valid_mask]

    # Check occupancy for all valid neighbors at once
    z_indices, y_indices, x_indices = valid_neighbors.T
    empty_mask = ~occupancy_mask[z_indices, y_indices, x_indices]
    empty_neighbors = valid_neighbors[empty_mask]

    # Remove duplicates more efficiently
    unique_targets = np.unique(empty_neighbors, axis=0)

    if len(unique_targets) == 0:
        return []

    # Convert to directions for the jump engine
    start_array = np.array(start)
    directions = unique_targets - start_array

    # Hand off to the existing generator
    gen = get_integrated_jump_movement_generator(cache)
    return gen.generate_jump_moves(
        color=color,
        pos=start,
        directions=directions,
        allow_capture=False,  # network teleport never captures
    )

    if not unique_targets:
        return []

    # --- 4. Convert to directions for the jump engine ---
    start_array = np.array(start, dtype=np.int8)
    target_array = np.array(list(unique_targets), dtype=np.int8)
    directions = target_array - start_array  # Shape: (M, 3)

    # --- 5. Hand off to the existing generator ---
    gen = get_integrated_jump_movement_generator(cache)
    return gen.generate_jump_moves(
        color=color,
        pos=start,
        directions=directions,
        allow_capture=False,  # network teleport never captures
    )
