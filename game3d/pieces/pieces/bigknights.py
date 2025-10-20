# knightmoves.py — (3,1) and (3,2) knight jumps in one file
from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import convert_legacy_move_args
from game3d.common.common import in_bounds

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ----------------------------------------------------------
#  (3,1) and (3,2) offsets — generated once at import
# ----------------------------------------------------------
_KNIGHT31_DIRS = np.array([
    (dx, dy, dz)
    for dx in (-3, 3)
    for dy in (-1, 1)
    for dz in (-1, 1)
], dtype=np.int8)   # 8 directions

_KNIGHT32_DIRS = np.array([
    (dx, dy, dz)
    for dx in (-3, 3)
    for dy in (-2, 2)
    for dz in (-2, 2)
], dtype=np.int8)   # 8 directions

# ----------------------------------------------------------
#  Unified generator – colour-blind directions
# ----------------------------------------------------------
def _generate_knight_leaps(
    cache, color: Color, x: int, y: int, z: int, dirs: np.ndarray
) -> List[Move]:
    start = (x, y, z)
    own_code = 1 if color == Color.WHITE else 2

    # pre-filter targets before building directions
    targets = []
    for dx, dy, dz in dirs:
        tx, ty, tz = x + dx, y + dy, z + dz
        if not in_bounds((tx, ty, tz)):
            continue
        if cache.occupancy.is_occupied(tx, ty, tz):               # occupied – may be capture
            victim = cache.occupancy.get((tx, ty, tz))
            if victim and victim.color != color:
                targets.append((tx, ty, tz))
        else:                                   # empty – always legal
            targets.append((tx, ty, tz))

    if not targets:
        return []

    # vectorised batch to jump engine
    tarr = np.array(targets, dtype=np.int16)
    start_arr = np.array(start, dtype=np.int16)
    directions = tarr - start_arr

    jump = get_integrated_jump_movement_generator(cache)
    return jump.generate_jump_moves(
        color=color,
        pos=start,
        directions=directions.astype(np.int8),
        allow_capture=True,
    )

# ----------------------------------------------------------
#  Dispatchers – both piece types in the same file
# ----------------------------------------------------------
@register(PieceType.KNIGHT31)
def knight31_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return _generate_knight_leaps(state.cache, state.color, x, y, z, _KNIGHT31_DIRS)

@register(PieceType.KNIGHT32)
def knight32_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return _generate_knight_leaps(state.cache, state.color, x, y, z, _KNIGHT32_DIRS)

__all__ = []
