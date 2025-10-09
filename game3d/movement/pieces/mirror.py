# mirrorteleportmoves.py
"""Mirror-Teleporter – single jump to (8-x, 8-y, 8-z) + armour filter."""

from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING
from game3d.pieces.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ----------------------------------------------------------
# 1.  Mirror-teleport direction (single vector)
# ----------------------------------------------------------
def _mirror_target(x: int, y: int, z: int) -> tuple[int, int, int]:
    return (8 - x, 8 - y, 8 - z)

# ----------------------------------------------------------
# 3.  Generator – occupancy pre-filter, single jump batch
# ----------------------------------------------------------
def generate_mirror_teleport_moves(cache, color: Color, x: int, y: int, z: int) -> List[Move]:
    start = (x, y, z)
    target = _mirror_target(x, y, z)

    if start == target:          # no-op
        return []

    # pre-filter: must be empty OR enemy (non-armoured)
    occ_mask = cache.occupancy.mask
    tx, ty, tz = target
    if not in_bounds(tx, ty, tz):
        return []

    if occ_mask[tz, ty, tx]:
        victim = cache.piece_cache.get(target)
        if victim is None or victim.color == color:
            return []                       # blocked or immune
        # legal capture – keep
    # else: empty – keep

    # single direction → jump engine
    dirs = np.array([[tx - x, ty - y, tz - z]], dtype=np.int8)

    jump = get_integrated_jump_movement_generator(cache)
    return jump.generate_jump_moves(
        color=color,
        pos=start,
        directions=dirs,
        allow_capture=True,
    )

# ----------------------------------------------------------
# 4.  Dispatcher – state-first
# ----------------------------------------------------------
@register(PieceType.MIRROR)
def mirror_move_dispatcher(state: State, x: int, y: int, z: int) -> List[Move]:
    return generate_mirror_teleport_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_mirror_teleport_moves"]
