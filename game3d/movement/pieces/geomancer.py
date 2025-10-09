# geomancermoves.py
"""Geomancer – 1-sphere walk OR 3-sphere surface block (empty squares only)."""

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
# 1.  26 King directions (1-radius walk)
# ----------------------------------------------------------
_KING_DIRS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

# ----------------------------------------------------------
# 2.  3-sphere surface directions (dx²+dy²+dz² ≈ 9)
# ----------------------------------------------------------
_GEOMANCY_DIRS = np.array([
    (dx, dy, dz)
    for dx in range(-4, 5)
    for dy in range(-4, 5)
    for dz in range(-4, 5)
    if 8 <= dx*dx + dy*dy + dz*dz <= 10 and (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

# ----------------------------------------------------------
# 3.  Generator – two distinct move kinds
# ----------------------------------------------------------
def generate_geomancer_moves(cache: OptimizedCacheManager, color: Color, x: int, y: int, z: int) -> List[Move]:
    start = (x, y, z)
    occ_mask = cache.occupancy.mask
    moves: List[Move] = []

    jump = get_integrated_jump_movement_generator(cache)

    # 3a. Normal 1-radius walk (King moves)
    king_moves = jump.generate_jump_moves(
        color=color,
        pos=start,
        directions=_KING_DIRS,
        allow_capture=True,
    )
    moves.extend(king_moves)

    # 3b. 3-sphere surface → effect move (no movement, no capture, must be EMPTY)
    effect_targets = []
    for dx, dy, dz in _GEOMANCY_DIRS:
        tx, ty, tz = x + dx, y + dy, z + dz
        if not in_bounds((tx, ty, tz)):
            continue
        if occ_mask[tz, ty, tx]:              # occupied → skip
            continue
        effect_targets.append((tx, ty, tz))

    if effect_targets:
        # vectorised batch: self-move to self, flagged as geomancy
        tarr = np.array(effect_targets, dtype=np.int16)
        directions = tarr - np.array(start, dtype=np.int16)

        effect_moves = jump.generate_jump_moves(
            color=color,
            pos=start,
            directions=directions.astype(np.int8),
            allow_capture=False,            # no captures
        )
        # mark them so the board knows to apply the block
        for mv in effect_moves:
            mv.metadata["is_geomancy_effect"] = True
            mv.metadata["geomancy_target"] = mv.to_coord   # real square to block
            mv.to_coord = start                            # piece does NOT move
        moves.extend(effect_moves)

    return moves

# ----------------------------------------------------------
# 4.  Dispatcher – in-file
# ----------------------------------------------------------
@register(PieceType.GEOMANCER)
def geomancer_move_dispatcher(state: State, x: int, y: int, z: int) -> List[Move]:
    return generate_geomancer_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_geomancer_moves"]
