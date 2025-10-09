# friendlytp.py - OPTIMIZED VERSION
"""Friendly-Teleporter – teleport to any empty neighbour of a friendly piece
PLUS normal 1-step King moves – all in one jump-batch."""

from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING
from game3d.pieces.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move, convert_legacy_move_args

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ----------------------------------------------------------
# 1. Pre-computed constants
# ----------------------------------------------------------
_KING_DIRS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

# Pre-compute bounds check mask (once at module load)
_BOUNDS_MASK = 0xFFFFFFF8  # ~7 for 9x9x9 board

# ----------------------------------------------------------
# 2. Vectorized network direction builder
# ----------------------------------------------------------
def _build_network_directions(cache, color, x, y, z) -> np.ndarray:
    """
    OPTIMIZED: Return (N,3) int8 array of (target - start) for every valid teleport.
    Uses vectorized operations to eliminate Python loops.
    """
    start = np.array([x, y, z], dtype=np.int16)

    # Get occupancy mask once
    occ_mask = cache.occupancy.mask  # 9×9×9 bool

    # Get all friendly piece coordinates as numpy array
    friendly_coords = []
    for coord, _ in cache.piece_cache.iter_color(color):
        friendly_coords.append(coord)

    if not friendly_coords:
        return np.empty((0, 3), dtype=np.int8)

    # Convert to numpy array: (N_pieces, 3)
    friendly_arr = np.array(friendly_coords, dtype=np.int16)

    # Broadcast addition: (N_pieces, 1, 3) + (1, 26, 3) = (N_pieces, 26, 3)
    # This computes all 26 neighbors for all friendly pieces in one operation
    neighbors = friendly_arr[:, np.newaxis, :] + _KING_DIRS[np.newaxis, :, :]

    # Flatten to (N_pieces * 26, 3)
    neighbors_flat = neighbors.reshape(-1, 3)

    # Vectorized bounds check using bitwise operations
    # All coordinates must be in [0, 8]
    out_of_bounds = (
        (neighbors_flat[:, 0] < 0) | (neighbors_flat[:, 0] > 8) |
        (neighbors_flat[:, 1] < 0) | (neighbors_flat[:, 1] > 8) |
        (neighbors_flat[:, 2] < 0) | (neighbors_flat[:, 2] > 8)
    )

    # Filter out-of-bounds coordinates
    neighbors_valid = neighbors_flat[~out_of_bounds]

    if len(neighbors_valid) == 0:
        return np.empty((0, 3), dtype=np.int8)

    # Vectorized occupancy check using advanced indexing
    # Note: occ_mask is [z, y, x] indexed
    occupied = occ_mask[neighbors_valid[:, 2], neighbors_valid[:, 1], neighbors_valid[:, 0]]

    # Filter occupied squares
    empty_neighbors = neighbors_valid[~occupied]

    if len(empty_neighbors) == 0:
        return np.empty((0, 3), dtype=np.int8)

    # Remove duplicates using unique (much faster than Python set)
    unique_targets = np.unique(empty_neighbors, axis=0)

    # Remove start position if present
    start_mask = np.all(unique_targets == start, axis=1)
    unique_targets = unique_targets[~start_mask]

    if len(unique_targets) == 0:
        return np.empty((0, 3), dtype=np.int8)

    # Compute direction vectors: target - start
    directions = (unique_targets - start).astype(np.int8)

    return directions

# ----------------------------------------------------------
# 3. Unified generator – one batch to the jump engine
# ----------------------------------------------------------
def generate_network_teleport_with_king_moves(
    cache, color: Color, x: int, y: int, z: int
) -> List[Move]:
    """Generate both teleport and king moves in a single batch."""
    start = (int(x), int(y), int(z))

    # 1. Get teleport directions (now vectorized)
    tel_dirs = _build_network_directions(cache, color, x, y, z)

    # 2. King directions (already numpy array)
    king_dirs = _KING_DIRS.astype(np.int8)

    # 3. Concatenate and deduplicate using numpy
    if len(tel_dirs) > 0:
        all_dirs = np.unique(np.vstack((tel_dirs, king_dirs)), axis=0)
    else:
        all_dirs = king_dirs

    # 4. Single jump-batch generation
    jump = get_integrated_jump_movement_generator(cache)
    raw_moves = jump.generate_jump_moves(
        color=color,
        pos=start,
        directions=all_dirs,
        allow_capture=True,
    )

    # 5. Mark teleports (if needed for metadata)
    # Create set of teleport direction tuples for fast lookup
    tel_set = {tuple(d) for d in tel_dirs} if len(tel_dirs) > 0 else set()

    moves = []
    start_arr = np.array(start, dtype=np.int16)

    for m in raw_moves:
        mv = convert_legacy_move_args(
            from_coord=start,
            to_coord=m.to_coord,
            is_capture=m.is_capture
        )

        # Check if this is a teleport move
        if tel_set:
            direction = tuple(np.array(mv.to_coord, dtype=np.int16) - start_arr)
            if direction in tel_set:
                mv.metadata["is_teleport"] = True

        moves.append(mv)

    return moves

# ----------------------------------------------------------
# 4. Dispatcher – state-first
# ----------------------------------------------------------
@register(PieceType.FRIENDLYTELEPORTER)
def friendlytp_move_dispatcher(state: "GameState", x: int, y: int, z: int) -> List[Move]:
    """Dispatcher for friendly teleporter piece."""
    return generate_network_teleport_with_king_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_network_teleport_with_king_moves", "friendlytp_move_dispatcher"]
