# swapper.py – FIXED VERSION
"""Swapper == King-steps ∪ friendly-swap teleport"""
from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.coord_utils import in_bounds_vectorised

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ---------- 1. King 1-step vectors (26 directions) ----------
_KING_DIRS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

# ---------- 2. Friendly-swap teleport directions ----------
def _friendly_swap_directions(cache, color, x, y, z) -> np.ndarray:
    """
    FIXED: Return numpy array of (target - start) for every friendly piece.
    Uses proper bounds checking and coordinate ordering.
    """
    start = np.array((x, y, z), dtype=np.int16)

    # Get all friendly pieces except the current one
    friends = []
    for coord, _ in cache.piece_cache.iter_color(color):
        if coord != (x, y, z):
            friends.append(coord)

    if not friends:
        return np.empty((0, 3), dtype=np.int8)

    # Convert to numpy array: (N_pieces, 3) in [x, y, z] order
    friends_arr = np.array(friends, dtype=np.int16)

    # Vectorized bounds check (should already be valid, but defensive)
    valid_mask = in_bounds_vectorised(friends_arr)
    friends_arr = friends_arr[valid_mask]

    if len(friends_arr) == 0:
        return np.empty((0, 3), dtype=np.int8)

    # Compute direction vectors: target - start
    directions = (friends_arr - start).astype(np.int8)

    return directions

# ---------- 3. Unified generator ----------
def generate_swapper_moves(cache, color, x, y, z) -> List[Move]:
    """
    Generate all swapper moves:
    1. King walks (26 directions, 1 step each)
    2. Friendly teleport-swaps (to any friendly piece location)
    """
    jump = get_integrated_jump_movement_generator(cache)
    pos = (x, y, z)

    # 1. King walks (standard 1-step moves)
    moves = jump.generate_jump_moves(color=color, pos=pos, directions=_KING_DIRS)

    # 2. Friendly teleport-swaps
    swap_dirs = _friendly_swap_directions(cache, color, x, y, z)
    if swap_dirs.size > 0:
        swaps = jump.generate_jump_moves(
            color=color,
            pos=pos,
            directions=swap_dirs,
            allow_capture=False,  # we never capture, we *swap*
        )
        # Mark them so the board knows to perform a swap later
        for m in swaps:
            m.metadata["is_swap"] = True
        moves += swaps

    return moves

# ---------- 4. Dispatcher ----------
@register(PieceType.SWAPPER)
def swapper_move_dispatcher(state: "GameState", x: int, y: int, z: int) -> List[Move]:
    """Dispatcher for swapper piece."""
    return generate_swapper_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_swapper_moves", "swapper_move_dispatcher"]
