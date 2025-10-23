# game3d/movement/pieces/echo.py
"""
Echo – 2-sphere surface projected ±3 in every axis.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.cache_utils import ensure_int_coords
from game3d.common.coord_utils import in_bounds_vectorised

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# --------------------------------------------------------------------------- #
#  Echo directions (2-sphere surface anchored 3 steps away)                  #
# --------------------------------------------------------------------------- #
# 8 anchor offsets (±2, ±2, ±2)
_ANCHORS = np.array([
    (dx, dy, dz)
    for dx in (-2, 2)
    for dy in (-2, 2)
    for dz in (-2, 2)
], dtype=np.int8)

# 32 radius-2 bubble offsets
_BUBBLE = np.array([
    (dx, dy, dz)
    for dx in range(-2, 3)
    for dy in range(-2, 3)
    for dz in range(-2, 3)
    if dx*dx + dy*dy + dz*dz <= 4 and (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

# 256 raw jump vectors
_ECHO_DIRECTIONS = (_ANCHORS[:, None, :] + _BUBBLE[None, :, :]).reshape(-1, 3)

def generate_echo_moves(
    cache: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate echo moves: jumps to 2-sphere surface anchored 3 steps away."""
    x, y, z = ensure_int_coords(x, y, z)
    start = np.array([x, y, z], dtype=np.int16)

    # Filter valid directions
    targets = start + _ECHO_DIRECTIONS
    valid_mask = in_bounds_vectorised(targets)
    safe_dirs = _ECHO_DIRECTIONS[valid_mask]

    # Additional safety check
    if len(safe_dirs) > 0:
        dest_check = start + safe_dirs
        final_mask = np.all((dest_check >= 0) & (dest_check < 9), axis=1)
        safe_dirs = safe_dirs[final_mask]

    jump_gen = get_integrated_jump_movement_generator(cache)
    moves = jump_gen.generate_jump_moves(
        color=color,
        pos=(x, y, z),
        directions=safe_dirs,
        allow_capture=True,
    )

    return moves

@register(PieceType.ECHO)
def echo_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_echo_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_echo_moves"]
