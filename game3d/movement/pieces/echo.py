# game3d/movement/piecemoves/echomoves.py
"""Echo – 2-sphere surface projected ±3 in every axis – self-contained."""
from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING

from game3d.pieces.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move
if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# 8 anchor offsets (±3, ±3, ±3)
_ANCHORS = np.array([(dx, dy, dz) for dx in (-3, 3) for dy in (-3, 3) for dz in (-3, 3)], dtype=np.int8)

# 32 radius-2 bubble offsets (dx²+dy²+dz² ≤ 4, origin excluded)
_BUBBLE = np.array([
    (dx, dy, dz)
    for dx in range(-2, 3)
    for dy in range(-2, 3)
    for dz in range(-2, 3)
    if dx*dx + dy*dy + dz*dz <= 4 and (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

# 256 final jump vectors
_ECHO_DIRS = (_ANCHORS[:, None, :] + _BUBBLE[None, :, :]).reshape(-1, 3)

def generate_echo_moves(cache, color: Color, x: int, y: int, z: int) -> List:
    """Echo jumps to any square on the 2-sphere surface anchored 3 steps away."""
    return get_integrated_jump_movement_generator(cache).generate_jump_moves(
        color=color, pos=(x, y, z), directions=_ECHO_DIRS
    )

@register(PieceType.ECHO)
def echo_move_dispatcher(state: State, x: int, y: int, z: int):
    return generate_echo_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_echo_moves"]
