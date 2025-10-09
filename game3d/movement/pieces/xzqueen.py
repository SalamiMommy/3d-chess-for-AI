"""
XZ-Queen: 8 slider rays in XZ-plane + full 3-D king hop (26 directions, 1 step).
Exports:
  generate_xz_queen_moves(cache, color, x, y, z) -> list[Move]
  (decorated) xz_queen_dispatcher(state, x, y, z) -> list[Move]
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.slidermovement import get_slider_generator
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move
if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager as CacheManager
    from game3d.game.gamestate import GameState

# 8 directions confined to the XZ-plane (Y fixed)
_XZ_SLIDER_DIRS = np.array([
    (1, 0, 0), (-1, 0, 0),
    (0, 0, 1), (0, 0, -1),
    (1, 0, 1), (1, 0, -1),
    (-1, 0, 1), (-1, 0, -1)
], dtype=np.int8)

# same 26 king directions
_KING_3D_DIRS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

def generate_xz_queen_moves(
    cache: CacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List:
    """Slider rays (XZ-plane) + full 3-D king hop."""
    pos = (x, y, z)
    slider_engine = get_slider_generator()
    slider_moves = slider_engine.generate_moves(
        piece_type='xz_queen',
        pos=pos,
        board_occupancy=cache.occupancy.mask,
        color=color.value,
        max_distance=8
    )

    jump_gen = get_integrated_jump_movement_generator(cache)
    king_moves = jump_gen.generate_jump_moves(
        color=color,
        pos=pos,
        directions=_KING_3D_DIRS,
        allow_capture=True
    )

    return slider_moves + king_moves

@register(PieceType.XZQUEEN)
def xz_queen_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List:
    return generate_xz_queen_moves(state.cache, state.color, x, y, z)

__all__ = ['generate_xz_queen_moves']
