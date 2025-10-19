# game3d/movement/piecemoves/bishopmoves.py
"""
3-D Bishop – pure slider on every 2-axis diagonal.
One file → movement + registration + slider integration.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.slidermovement import generate_moves

# --------------------------------------------------------------------------- #
#  2-axis diagonal directions (13 unique unit vectors)                        #
# --------------------------------------------------------------------------- #
BISHOP_DIRECTIONS: np.ndarray = np.array([
    # 2-D diagonals (4)
    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
    (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
    (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    # 3-D pure diagonal (4) – optional, delete if you want strict 2-axis
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
], dtype=np.int8)

# --------------------------------------------------------------------------- #
#  Core generator – talks directly to the slider kernel                       #
# --------------------------------------------------------------------------- #
def generate_bishop_moves(cache: OptimizedCacheManager,
                          color: Color,
                          x: int, y: int, z: int) -> List[Move]:
    """Return all bishop slides up to 8 steps along 2-axis diagonals."""
    return generate_moves(
        piece_type='bishop',
        pos=(x, y, z),
        color=color.value,
        max_distance=8,
        directions=BISHOP_DIRECTIONS,
        occupancy=cache.occupancy._occ,    # ← pass the 3-D mask instead of the cache
    )

# --------------------------------------------------------------------------- #
#  Dispatcher registration – done here, no external file needed               #
# --------------------------------------------------------------------------- #
@register(PieceType.BISHOP)
def bishop_move_dispatcher(state, x: int, y: int, z: int) -> List[Move]:
    return generate_bishop_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_bishop_moves"]
