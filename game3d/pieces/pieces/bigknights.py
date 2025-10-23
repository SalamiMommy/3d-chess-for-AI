# game3d/movement/pieces/bigknights.py
"""
Knight31 and Knight32 – (3,1) and (3,2) knight jumps.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.cache_utils import ensure_int_coords

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# --------------------------------------------------------------------------- #
#  Knight31 directions (3,1) leaps                                           #
# --------------------------------------------------------------------------- #
_KNIGHT31_DIRECTIONS = np.array([
    (dx, dy, dz)
    for dx in (-3, 3)
    for dy in (-1, 1)
    for dz in (-1, 1)
], dtype=np.int8)

# --------------------------------------------------------------------------- #
#  Knight32 directions (3,2) leaps                                           #
# --------------------------------------------------------------------------- #
_KNIGHT32_DIRECTIONS = np.array([
    (dx, dy, dz)
    for dx in (-3, 3)
    for dy in (-2, 2)
    for dz in (-2, 2)
], dtype=np.int8)

def generate_knight31_moves(
    cache: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate Knight31 moves."""
    x, y, z = ensure_int_coords(x, y, z)

    jump_gen = get_integrated_jump_movement_generator(cache)
    moves = jump_gen.generate_jump_moves(
        color=color,
        pos=(x, y, z),
        directions=_KNIGHT31_DIRECTIONS,
        allow_capture=True,
    )

    return moves

def generate_knight32_moves(
    cache: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate Knight32 moves."""
    x, y, z = ensure_int_coords(x, y, z)

    jump_gen = get_integrated_jump_movement_generator(cache)
    moves = jump_gen.generate_jump_moves(
        color=color,
        pos=(x, y, z),
        directions=_KNIGHT32_DIRECTIONS,
        allow_capture=True,
    )

    return moves

@register(PieceType.KNIGHT31)
def knight31_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_knight31_moves(state.cache, state.color, x, y, z)

@register(PieceType.KNIGHT32)
def knight32_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_knight32_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_knight31_moves", "generate_knight32_moves"]
