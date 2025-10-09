# game3d/movement/piecemoves/orbitalmoves.py
"""Orbiter – Manhattan-distance 4 jumps – self-contained."""
from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING

from game3d.pieces.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move
if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# 66 Manhattan-4 offsets
_ORBITAL_DIRS = np.array([
    (dx, dy, dz)
    for dx in range(-4, 5)
    for dy in range(-4, 5)
    for dz in range(-4, 5)
    if abs(dx) + abs(dy) + abs(dz) == 4
], dtype=np.int8)

def generate_orbital_moves(cache, color: Color, x: int, y: int, z: int) -> List:
    """Orbiter jumps to any square exactly 4 Manhattan away."""
    return get_integrated_jump_movement_generator(cache).generate_jump_moves(
        color=color, pos=(x, y, z), directions=_ORBITAL_DIRS
    )

@register(PieceType.ORBITER)
def orbital_move_dispatcher(state: State, x: int, y: int, z: int):
    return generate_orbital_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_orbital_moves"]
