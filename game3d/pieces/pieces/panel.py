# panelmoves.py — Panel: teleport to any empty square on the same x OR y OR z plane
#                 plus normal 1-step King moves – all in one batch
from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ----------------------------------------------------------
# 1.  Same-plane offsets (x=const, y=const, z=const) – 24 directions
# ----------------------------------------------------------
_PANEL_DIRS = np.array([
    (dx, dy, dz)
    for dx in range(-8, 9)
    for dy in range(-8, 9)
    for dz in range(-8, 9)
    if (dx == 0 or dy == 0 or dz == 0)
       and not (dx == dy == dz == 0)
], dtype=np.int8)

# ----------------------------------------------------------
# 2.  26 King directions (inline – no import)
# ----------------------------------------------------------
_KING_DIRS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

# ----------------------------------------------------------
# 3.  Generator – occupancy-filtered, deduped, single jump batch
# ----------------------------------------------------------
def generate_panel_moves(cache, color: Color, x: int, y: int, z: int) -> List[Move]:
    start = (x, y, z)
    own_code = 1 if color == Color.WHITE else 2

    targets = set()

    # 3a. panel leaps (same-plane)
    for dx, dy, dz in _PANEL_DIRS:
        tx, ty, tz = x + dx, y + dy, z + dz
        if not in_bounds((tx, ty, tz)):
            continue
        if cache.occupancy.is_occupied(tx, ty, tz):
            victim = cache.occupancy.get((tx, ty, tz))
            if victim and victim.color != color:
                targets.add((tx, ty, tz))
        else:
            targets.add((tx, ty, tz))

    # 3b. king walks
    for dx, dy, dz in _KING_DIRS:
        tx, ty, tz = x + dx, y + dy, z + dz
        if not in_bounds((tx, ty, tz)):
            continue
        if cache.occupancy.is_occupied(tx, ty, tz):
            victim = cache.occupancy.get((tx, ty, tz))
            if victim and victim.color != color:
                targets.add((tx, ty, tz))
        else:
            targets.add((tx, ty, tz))

    if not targets:
        return []

    # 3c. vectorised batch
    tarr = np.array(list(targets), dtype=np.int16)
    directions = tarr - np.array(start, dtype=np.int16)

    jump = get_integrated_jump_movement_generator(cache)
    return jump.generate_jump_moves(
        color=color,
        pos=start,
        directions=directions.astype(np.int8),
        allow_capture=True,
    )

# ----------------------------------------------------------
# 5.  Dispatcher
# ----------------------------------------------------------
@register(PieceType.PANEL)
def panel_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_panel_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_panel_moves"]
