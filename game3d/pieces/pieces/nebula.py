# game3d/movement/pieces/nebula.py
"""
Nebula – teleport to any square within radius-3 sphere.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.common.cache_utils import ensure_int_coords

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# --------------------------------------------------------------------------- #
#  Radius-3 sphere directions (122 directions)                               #
# --------------------------------------------------------------------------- #
_NEBULA_DIRECTIONS = np.array([
    (dx, dy, dz)
    for dx in range(-3, 4)
    for dy in range(-3, 4)
    for dz in range(-3, 4)
    if dx*dx + dy*dy + dz*dz <= 9 and (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

def generate_nebula_moves(
    cache: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate nebula moves: teleport within radius-3 sphere."""
    x, y, z = ensure_int_coords(x, y, z)

    jump_gen = get_integrated_jump_movement_generator(cache_manager)
    moves = jump_gen.generate_jump_moves(
        color=color,
        pos=(x, y, z),
        directions=_NEBULA_DIRECTIONS,
        allow_capture=True,
    )

    return moves

@register(PieceType.NEBULA)
def nebula_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_nebula_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_nebula_moves"]
