# networkteleportmovement.py
"""
Network Teleporter — teleport to any empty square adjacent to any friendly piece.
Optimized using existing caches and incremental updates.
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
    Generate all legal network-teleport moves from (x, y, z) using cached targets.

    Optimizations:
    - Cache network teleport targets per color and update incrementally
    - Only recalculate targets when cache is dirty
    - Use numpy arrays for efficient operations
    - Batch handoff to jump movement generator
    """
    start = (x, y, z)

    # Initialize cache attributes if not present
    if not hasattr(cache, '_network_teleport_targets'):
        cache._network_teleport_targets = {Color.WHITE: set(), Color.BLACK: set()}
        cache._network_teleport_dirty = {Color.WHITE: True, Color.BLACK: True}

    # Recalculate targets if cache is dirty
    if cache._network_teleport_dirty[color]:
        _update_network_teleport_targets(cache, color)
        cache._network_teleport_dirty[color] = False

    # Get cached targets for this color
    targets = cache._network_teleport_targets[color]

    # Remove the starting position if present
    if start in targets:
        targets_without_start = targets - {start}
    else:
        targets_without_start = targets

    if not targets_without_start:
        return []

    # Convert to numpy array for direction computation
    targets_array = np.array(list(targets_without_start), dtype=np.int8)
    start_array = np.array(start, dtype=np.int8)
    directions = targets_array - start_array

    # Hand off to jump movement generator
    gen = get_integrated_jump_movement_generator(cache)
    return gen.generate_jump_moves(
        color=color,
        pos=start,
        directions=directions,
        allow_capture=False,
    )

def _update_network_teleport_targets(cache: CacheManager, color: Color) -> None:
    """
    Update the cached network teleport targets for the given color.
    Uses vectorized operations for efficiency.
    """
    targets = set()

    # Get all friendly pieces for color
    friendly_positions = []
    if hasattr(cache.piece_cache, "iter_color"):
        friendly_positions = [coord for coord, _ in cache.piece_cache.iter_color(color)]

    if not friendly_positions:
        cache._network_teleport_targets[color] = targets
        return

    # Convert to numpy array for vectorized operations
    friendly_positions = np.array(friendly_positions, dtype=np.int8)

    # Generate all neighbors
    all_neighbors = friendly_positions[:, np.newaxis, :] + _NEIGHBOR_DIRECTIONS
    neighbors_flat = all_neighbors.reshape(-1, 3)

    # Filter out-of-bounds
    valid_mask = (
        (neighbors_flat[:, 0] >= 0) & (neighbors_flat[:, 0] < 9) &
        (neighbors_flat[:, 1] >= 0) & (neighbors_flat[:, 1] < 9) &
        (neighbors_flat[:, 2] >= 0) & (neighbors_flat[:, 2] < 9)
    )
    valid_neighbors = neighbors_flat[valid_mask]

    if valid_neighbors.shape[0] > 0:
        # Check occupancy
        occupancy_mask = cache.occupancy.mask
        z_idx, y_idx, x_idx = valid_neighbors.T
        empty_mask = ~occupancy_mask[z_idx, y_idx, x_idx]
        empty_neighbors = valid_neighbors[empty_mask]

        # Deduplicate
        unique_targets = np.unique(empty_neighbors, axis=0)
        targets = {tuple(coord) for coord in unique_targets}

    cache._network_teleport_targets[color] = targets
