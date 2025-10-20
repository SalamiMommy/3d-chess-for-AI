# game3d/movement/movetypes/knight.py
"""
Unified Knight movement generator – 2+1 leaper, share-square aware.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.enums import Color, PieceType
from game3d.common.coord_utils import in_bounds
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import convert_legacy_move_args

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ------------------------------------------------------------------
# 24 classical knight offsets (2,1,0) permutations
# ------------------------------------------------------------------
KNIGHT_OFFSETS = np.array([
    (1, 2, 0), (1, -2, 0), (-1, 2, 0), (-1, -2, 0),
    (2, 1, 0), (2, -1, 0), (-2, 1, 0), (-2, -1, 0),
    (1, 0, 2), (1, 0, -2), (-1, 0, 2), (-1, 0, -2),
    (2, 0, 1), (2, 0, -1), (-2, 0, 1), (-2, 0, -1),
    (0, 1, 2), (0, 1, -2), (0, -1, 2), (0, -1, -2),
    (0, 2, 1), (0, 2, -1), (0, -2, 1), (0, -2, -1),
], dtype=np.int8)

# ------------------------------------------------------------------
# Public generator – same architecture as bigknights
# ------------------------------------------------------------------
def generate_knight_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    cache = state.cache
    color = state.color
    start = (x, y, z)

    # 1. Collect all legal targets (empty or enemy)
    targets: list[tuple[int, int, int]] = []
    for dx, dy, dz in KNIGHT_OFFSETS:
        tx, ty, tz = x + dx, y + dy, z + dz
        if not in_bounds((tx, ty, tz)):
            continue
        occ = cache.occupancy.get((tx, ty, tz))
        if occ is None or occ.color != color:        # empty or enemy
            targets.append((tx, ty, tz))

    if not targets:                                  # early exit
        return []

    # 2. Vectorise and hand over to the integrated jump engine
    tarr = np.array(targets, dtype=np.int16)
    directions = tarr - np.array(start, dtype=np.int16)

    jump = get_integrated_jump_movement_generator(cache)
    return jump.generate_jump_moves(
        color=color,
        pos=start,
        directions=directions.astype(np.int8),
        allow_capture=True,
    )

# ------------------------------------------------------------------
# Dispatcher registration
# ------------------------------------------------------------------
@register(PieceType.KNIGHT)
def knight_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_knight_moves(state, x, y, z)

__all__ = ["generate_knight_moves"]
