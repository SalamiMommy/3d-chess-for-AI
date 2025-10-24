# game3d/movement/piecemoves/kinglikemoves.py - FIXED
"""Exports king-like move generators and registers them with the dispatcher."""

from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.common.cache_utils import ensure_int_coords

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

# 26 one-step vectors (unchanged)
KING_DIRECTIONS_3D = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

# Base move generator
def generate_king_moves(
    cache_manager: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate king moves using single cache manager."""
    x, y, z = ensure_int_coords(x, y, z)
    pos = (x, y, z)

    jump_gen = get_integrated_jump_movement_generator(cache_manager)
    return jump_gen.generate_jump_moves(
        color=color,
        pos=pos,
        directions=KING_DIRECTIONS_3D,
        allow_capture=True,  # ADDED: kings can capture
    )

# Aliases for each piece type
generate_priest_moves = generate_king_moves

# Register dispatchers for each piece type
@register(PieceType.PRIEST)
def priest_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_king_moves(state.cache_manager, state.color, x, y, z)

@register(PieceType.KING)
def king_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_king_moves(state.cache_manager, state.color, x, y, z)

# Exports
__all__ = [
    'generate_king_moves',
    'generate_priest_moves'
]
