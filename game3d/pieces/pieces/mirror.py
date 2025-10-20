# mirrorteleportmoves.py
"""Mirror-Teleporter – single jump to (8-x, 8-y, 8-z) + armour filter."""

from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move, convert_legacy_move_args
from game3d.common.coord_utils import in_bounds

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ----------------------------------------------------------
# 1.  Mirror-teleport direction (single vector)
# ----------------------------------------------------------
def _mirror_target(x: int, y: int, z: int) -> tuple[int, int, int]:
    return (8 - x, 8 - y, 8 - z)

# ----------------------------------------------------------
# 2.  Generator – occupancy pre-filter, single jump batch
# ----------------------------------------------------------
def generate_mirror_teleport_moves(cache, color: Color, x: int, y: int, z: int) -> List[Move]:
    start = (x, y, z)
    target = _mirror_target(x, y, z)
    if start == target:
        return []

    tx, ty, tz = target
    if not in_bounds((tx, ty, tz)):
        return []

    if cache.occupancy.is_occupied(tx, ty, tz):
        victim = cache.occupancy.get(target)
        if victim is None or victim.color == color:
            return []

    dirs = np.array([[tx - x, ty - y, tz - z]], dtype=np.int8)
    jump = get_integrated_jump_movement_generator(cache)
    return jump.generate_jump_moves(color=color, pos=start, directions=dirs, allow_capture=True)
# ----------------------------------------------------------
# 3.  Dispatcher – state-first
# ----------------------------------------------------------
@register(PieceType.MIRROR)
def mirror_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_mirror_teleport_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_mirror_teleport_moves"]
