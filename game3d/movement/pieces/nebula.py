# nebulamoves.py
"""Nebula – teleport to any square within radius-3 sphere, armour-filtered, single batch."""

from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING
from game3d.pieces.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds
from game3d.cache.manager import OptimizedCacheManager

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ----------------------------------------------------------
# 1.  Pre-computed radius-3 sphere (122 offsets)
# ----------------------------------------------------------
_NEBULA_DIRS = np.array([
    (dx, dy, dz)
    for dx in range(-3, 4)
    for dy in range(-3, 4)
    for dz in range(-3, 4)
    if dx*dx + dy*dy + dz*dz <= 9 and (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)          # shape (122, 3)


# ----------------------------------------------------------
# 3.  Generator – occupancy pre-filter, single jump batch
# ----------------------------------------------------------
def generate_nebula_moves(cache: OptimizedCacheManager, color: Color, x: int, y: int, z: int) -> List[Move]:
    start = (x, y, z)
    occ_mask = cache.occupancy.mask          # 9×9×9 bool
    targets = []

    for dx, dy, dz in _NEBULA_DIRS:
        tx, ty, tz = x + dx, y + dy, z + dz
        if not in_bounds(tx, ty, tz):
            continue
        if occ_mask[tz, ty, tx]:
            victim = cache.piece_cache.get((tx, ty, tz))
            if victim and victim.color != color:
                targets.append((tx, ty, tz))
        else:
            targets.append((tx, ty, tz))

    if not targets:
        return []

    # vectorised batch
    tarr = np.array(targets, dtype=np.int16)
    directions = tarr - np.array(start, dtype=np.int16)

    jump = get_integrated_jump_movement_generator(cache)
    return jump.generate_jump_moves(
        color=color,
        pos=start,
        directions=directions.astype(np.int8),
        allow_capture=True,
    )

# ----------------------------------------------------------
# 4.  Dispatcher – in-file
# ----------------------------------------------------------
@register(PieceType.NEBULA)
def nebula_move_dispatcher(state: State, x: int, y: int, z: int) -> List[Move]:
    return generate_nebula_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_nebula_moves"]
