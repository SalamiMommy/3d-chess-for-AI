"""
Spiral-Slider — 6 counter-clockwise spiral rays (consolidated).
Exports:
  generate_spiral_moves(cache, color, x, y, z) -> list[Move]
  (decorated) spiral_dispatcher(state, x, y, z) -> list[Move]
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

def generate_spiral_moves(cache: CacheManager,
                          color: Color,
                          x: int, y: int, z: int) -> List[Move]:
    return get_slider_generator().generate_moves(
        piece_type='spiral',
        pos=(x, y, z),
        color=color.value,
        max_distance=32,
        cache_manager=cache,          # ← REQUIRED keyword-only argument
        directions=_SPIRAL_OFFS
    )

@register(PieceType.SPIRAL)
def spiral_move_dispatcher(state, x: int, y: int, z: int) -> List[Move]:
    return generate_spiral_moves(state.cache, state.color, x, y, z)

__all__ = ['generate_spiral_moves']
