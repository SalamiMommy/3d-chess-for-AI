"""
XZ-Queen: 8 slider rays in XZ-plane + full 3-D king hop (26 directions, 1 step).
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.slidermovement import generate_moves
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# 8 directions confined to the XZ-plane (Y fixed)
_XZ_SLIDER_DIRS = np.array([
    (1, 0, 0), (-1, 0, 0),
    (0, 0, 1), (0, 0, -1),
    (1, 0, 1), (1, 0, -1),
    (-1, 0, 1), (-1, 0, -1)
], dtype=np.int8)

# 26 one-step king directions (3-D)
_KING_3D_DIRS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

def generate_xz_queen_moves(
    cache_manager: 'OptimizedCacheManager',  # FIXED: Consistent parameter name
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Slider rays (XZ-plane, 8 dirs, 8 steps) + king hop (26 dirs, 1 step)."""
    slider_moves = generate_moves(
        piece_type='xz_queen',
        pos=(x, y, z),
        color=color,
        max_distance=8,
        directions=_XZ_SLIDER_DIRS,
        cache_manager=cache_manager,  # FIXED: Use parameter
    )

    king_moves = generate_moves(
        piece_type='xz_queen_kinghop',
        pos=(x, y, z),
        color=color,
        max_distance=1,
        directions=_KING_3D_DIRS,
        cache_manager=cache_manager,  # FIXED: Use parameter
    )

    return slider_moves + king_moves

@register(PieceType.XZQUEEN)
def xz_queen_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_xz_queen_moves(state.cache_manager, state.color, x, y, z)  # FIXED: Use cache_manager

__all__ = ['generate_xz_queen_moves']
