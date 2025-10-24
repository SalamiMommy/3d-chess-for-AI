# game3d/movement/piecemoves/bishopmoves.py
"""
3-D Bishop – pure slider on every 2-axis and 3-axis diagonal.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.slidermovement import generate_moves
from game3d.common.cache_utils import ensure_int_coords

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# 2-axis and 3-axis diagonal directions (20 unique unit vectors)
BISHOP_DIRECTIONS = np.array([
    # 2-D diagonals (12)
    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
    (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
    (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    # 3-D pure diagonal (8)
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
], dtype=np.int8)

def generate_bishop_moves(
    cache_manager: 'OptimizedCacheManager',  # FIXED: Consistent parameter name
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Return all bishop slides up to 8 steps along diagonals."""
    x, y, z = ensure_int_coords(x, y, z)
    return generate_moves(
        piece_type='bishop',
        pos=(x, y, z),
        color=color,
        max_distance=8,
        directions=BISHOP_DIRECTIONS,
        cache_manager=cache_manager,  # FIXED: Use parameter
    )

@register(PieceType.BISHOP)
def bishop_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_bishop_moves(state.cache_manager, state.color, x, y, z)  # FIXED: Use cache_manager

__all__ = ["generate_bishop_moves"]
