"""
Trigonal-Bishop — 8 space-diagonal rays (consolidated).
Exports:
  generate_trigonal_bishop_moves(cache, color, x, y, z) -> list[Move]
  (decorated) trigonal_bishop_dispatcher(state, x, y, z) -> list[Move]
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.slidermovement import get_slider_generator
from game3d.movement.movepiece import Move
if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager as CacheManager
    from game3d.game.gamestate import GameState

# 8 true 3-D diagonals
_TRIGONAL_DIRS = np.array([
    ( 1,  1,  1), ( 1,  1, -1), ( 1, -1,  1), ( 1, -1, -1),
    (-1,  1,  1), (-1,  1, -1), (-1, -1,  1), (-1, -1, -1)
], dtype=np.int8)

def generate_trigonal_bishop_moves(
    cache: CacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List:
    """Space-diagonal slider up to board edge."""
    return get_slider_generator().generate_moves(
        piece_type='trigonal_bishop',
        pos=(x, y, z),
        board_occupancy=cache.occupancy.mask,
        color=color.value,
        max_distance=8
    )

@register(PieceType.TRIGONALBISHOP)
def trigonal_bishop_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List:
    return generate_trigonal_bishop_moves(state.cache, state.color, x, y, z)

__all__ = ['generate_trigonal_bishop_moves']
