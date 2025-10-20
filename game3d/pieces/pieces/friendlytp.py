# friendlytp.py - FULL RE-WRITE using common.in_bounds_vectorised
"""Friendly-Teleporter – teleport to any empty neighbour of a friendly piece
PLUS normal 1-step King moves – all in one jump-batch."""

from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move, convert_legacy_move_args
from game3d.common.coord_utils import in_bounds_vectorised  # central helper

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ----------------------------------------------------------
# 1.  Pre-computed constants
# ----------------------------------------------------------
_KING_DIRS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

# ----------------------------------------------------------
# 2.  Vectorised network-direction builder
# ----------------------------------------------------------
def _build_network_directions(cache, color: Color, x: int, y: int, z: int) -> np.ndarray:
    # Force int conversion to prevent np.int64 propagation
    start = np.array([int(x), int(y), int(z)], dtype=np.int16)

    # Collect friendly pieces
    friendly_coords = np.array(
        [coord for coord, _ in cache.piece_cache.iter_color(color)],
        dtype=np.int16,
    )
    if friendly_coords.size == 0:
        return np.empty((0, 3), dtype=np.int8)

    # All 26 neighbors of every friendly piece
    neighbours = (
        friendly_coords[:, np.newaxis, :] + _KING_DIRS[np.newaxis, :, :]
    ).reshape(-1, 3)

    # Keep only in-bounds squares
    neighbours = neighbours[in_bounds_vectorised(neighbours)]
    neighbours = neighbours.astype(np.int16)

    # Keep only empty squares
    occ_mask = (cache.occupancy._occ != 0)  # Use updated occupancy logic: derive mask from _occ
    x_, y_, z_ = neighbours[:, 0], neighbours[:, 1], neighbours[:, 2]

    # CRITICAL FIX: Clip coordinates before indexing to prevent OOB
    x_ = np.clip(x_, 0, 8)
    y_ = np.clip(y_, 0, 8)
    z_ = np.clip(z_, 0, 8)

    empty_mask = ~occ_mask[z_, y_, x_]
    empty_neighbours = neighbours[empty_mask]

    # Remove duplicate targets
    unique_targets = np.unique(empty_neighbours, axis=0)

    # Remove start square
    unique_targets = unique_targets[~np.all(unique_targets == start, axis=1)]

    # Final safety filter
    safety_mask = in_bounds_vectorised(unique_targets)
    unique_targets = unique_targets[safety_mask]

    # Convert to directions and validate bounds
    directions = (unique_targets - start).astype(np.int8)

    # CRITICAL: Ensure directions don't exceed board dimensions
    directions = directions[np.all(np.abs(directions) <= 8, axis=1)]

    # Ensure directions don't lead to OOB from start
    dest_coords = start + directions
    valid_mask = np.all((dest_coords >= 0) & (dest_coords < 9), axis=1)
    directions = directions[valid_mask]

    return directions

# ----------------------------------------------------------
# 3.  Unified generator – one batch to the jump engine
# ----------------------------------------------------------
def generate_network_teleport_with_king_moves(
    cache, color: Color, x: int, y: int, z: int
) -> List[Move]:
    """Generate both teleport and king moves in a single batch."""
    start = (int(x), int(y), int(z))  # FIX: Force int

    # 1.  teleport directions (vectorised and bounds-safe)
    tel_dirs = _build_network_directions(cache, color, x, y, z)

    # 2.  king directions (already numpy array)
    king_dirs = _KING_DIRS.astype(np.int8)

    # 3.  concatenate and deduplicate using numpy
    if len(tel_dirs) > 0:
        all_dirs = np.unique(np.vstack((tel_dirs, king_dirs)), axis=0)
    else:
        all_dirs = king_dirs

    # 4.  single jump-batch generation
    jump = get_integrated_jump_movement_generator(cache)
    raw_moves = jump.generate_jump_moves(
        color=color,
        pos=start,
        directions=all_dirs,
        allow_capture=True,
    )

    # 5.  mark teleports (if needed for metadata)
    tel_set = {tuple(d) for d in tel_dirs} if len(tel_dirs) > 0 else set()
    moves = []
    start_arr = np.array(start, dtype=np.int16)

    for m in raw_moves:
        mv = convert_legacy_move_args(
            from_coord=start,
            to_coord=m.to_coord,
            is_capture=m.is_capture
        )

        # check if this is a teleport move
        if tel_set:
            direction = tuple(np.array(mv.to_coord, dtype=np.int16) - start_arr)
            if direction in tel_set:
                mv.metadata["is_teleport"] = True

        moves.append(mv)

    return moves

# ----------------------------------------------------------
# 4.  Dispatcher – state-first
# ----------------------------------------------------------
@register(PieceType.FRIENDLYTELEPORTER)
def friendlytp_move_dispatcher(state: "GameState", x: int, y: int, z: int) -> List[Move]:
    """Dispatcher for friendly teleporter piece."""
    # FIX: Force int to ensure consistent typing
    return generate_network_teleport_with_king_moves(state.cache, state.color, int(x), int(y), int(z))

__all__ = ["generate_network_teleport_with_king_moves", "friendlytp_move_dispatcher"]
