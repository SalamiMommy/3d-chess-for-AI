"""
Trigonal-Bishop — 8 space-diagonal rays (consolidated).
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

# 8 true 3-D diagonals
_TRIGONAL_DIRS = np.array([
    ( 1,  1,  1), ( 1,  1, -1), ( 1, -1,  1), ( 1, -1, -1),
    (-1,  1,  1), (-1,  1, -1), (-1, -1,  1), (-1, -1, -1)
], dtype=np.int8)

def generate_trigonal_bishop_moves(
    cache_manager: 'OptimizedCacheManager',  # FIXED: Consistent parameter name
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Space-diagonal slider up to board edge."""
    return generate_moves(
        piece_type='trigonalbishop',
        pos=(x, y, z),
        color=color,
        max_distance=8,
        directions=_TRIGONAL_DIRS,
        cache_manager=cache_manager,  # FIXED: Use parameter
    )

@register(PieceType.TRIGONALBISHOP)
def trigonal_bishop_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_trigonal_bishop_moves(state.cache_manager, state.color, x, y, z)  # FIXED: Use cache_manager

__all__ = ['generate_trigonal_bishop_moves']
