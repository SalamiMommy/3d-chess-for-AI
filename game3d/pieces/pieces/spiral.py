"""
Spiral-Slider — 6 counter-clockwise spiral rays (consolidated).
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

# same 6×8 offset table you already use
_SPIRAL_OFFS = np.vstack([
    off + np.array(dir_ax) * (i + 1)
    for dir_ax, offsets in [
        (( 1, 0, 0), [(0, 0, 0), (0, 1, 0), (-1, 1, 0), (-1, 0, 0),
                      (-1, -1, 0), (0, -1, 0), (1, -1, 0), (1, 0, 0)]),
        ((-1, 0, 0), [(0, 0, 0), (0, -1, 0), (-1, -1, 0), (-1, 0, 0),
                      (-1, 1, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
        (( 0, 1, 0), [(0, 0, 0), (-1, 0, 0), (-1, 0, 1), (0, 0, 1),
                      (1, 0, 1), (1, 0, 0), (1, 0, -1), (0, 0, -1)]),
        (( 0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1),
                       (-1, 0, 1), (-1, 0, 0), (-1, 0, -1), (0, 0, -1)]),
        (( 0, 0, 1), [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                      (-1, 1, 0), (-1, 0, 0), (-1, -1, 0), (0, -1, 0)]),
        (( 0, 0, -1), [(0, 0, 0), (-1, 0, 0), (-1, 1, 0), (0, 1, 0),
                       (1, 1, 0), (1, 0, 0), (1, -1, 0), (0, -1, 0)]),
    ]
    for i, off in enumerate(offsets)
])

def generate_spiral_moves(
    cache_manager: 'OptimizedCacheManager',  # FIXED: Consistent parameter name
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    return generate_moves(
        piece_type='spiral',
        pos=(x, y, z),
        color=color,
        max_distance=16,
        directions=_SPIRAL_OFFS,
        cache_manager=cache_manager,  # FIXED: Use parameter
    )

@register(PieceType.SPIRAL)
def spiral_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_spiral_moves(state.cache_manager, state.color, x, y, z)  # FIXED: Use cache_manager

__all__ = ['generate_spiral_moves']
