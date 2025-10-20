# game3d/movement/pieces/echo.py
"""Echo – 2-sphere surface projected ±3 in every axis – self-contained."""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.common.coord_utils import in_bounds_vectorised  # ← re-use central helper

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# 8 anchor offsets (±3, ±3, ±3)
_ANCHORS = np.array([(dx, dy, dz)
                     for dx in (-2, 2)
                     for dy in (-2, 2)
                     for dz in (-2, 2)], dtype=np.int8)

# 32 radius-2 bubble offsets (dx²+dy²+dz² ≤ 4, origin excluded)
_BUBBLE = np.array([(dx, dy, dz)
                    for dx in range(-2, 3)
                    for dy in range(-2, 3)
                    for dz in range(-2, 3)
                    if dx*dx + dy*dy + dz*dz <= 4 and (dx, dy, dz) != (0, 0, 0)],
                   dtype=np.int8)

# 256 raw jump vectors
_ECHO_DIRS = (_ANCHORS[:, None, :] + _BUBBLE[None, :, :]).reshape(-1, 3)

def generate_echo_moves(cache, color: Color, x: int, y: int, z: int) -> List[Move]:
    """Echo jumps to any square on the 2-sphere surface anchored 3 steps away."""
    # compute target squares
    start_pos = np.array([x, y, z], dtype=np.int16)
    targets = start_pos + _ECHO_DIRS

    # keep only directions whose target is inside the 9×9×9 board
    good = in_bounds_vectorised(targets)  # 1-D bool array
    safe_dirs = _ECHO_DIRS[good]

    # Additional safety check: ensure no direction component exceeds board bounds
    # This prevents intermediate calculations from going OOB
    if len(safe_dirs) > 0:
        dest_check = start_pos + safe_dirs
        final_good = np.all((dest_check >= 0) & (dest_check < 9), axis=1)
        safe_dirs = safe_dirs[final_good]

    return get_integrated_jump_movement_generator(cache).generate_jump_moves(
        piece_name='echo',
        color=color,
        pos=(x, y, z),
        directions=safe_dirs,
        allow_capture=True
    )

@register(PieceType.ECHO)
def echo_move_dispatcher(state: GameState, x: int, y: int, z: int):
    return generate_echo_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_echo_moves"]
