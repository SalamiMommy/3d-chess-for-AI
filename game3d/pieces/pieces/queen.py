# game3d/movement/piecemoves/queenmoves.py
"""3-D Queen – one-pass slider over orthogonal + 2-axis + 3-axis diagonals."""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.slidermovement import generate_moves
from game3d.movement.movepiece import Move

# --------------------------------------------------------------------------- #
#  Union of the three direction groups (23 unique unit vectors)               #
# --------------------------------------------------------------------------- #
QUEEN_DIRECTIONS = np.concatenate((
    np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=np.int8),                # 6 orthogonal
    np.array([(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0), (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),             # 12 diagonal-2D
              (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)], dtype=np.int8),
    np.array([(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)], dtype=np.int8)  # 8 diagonal-3D
))

def generate_queen_moves(cache: OptimizedCacheManager,
                         color: Color,
                         x: int, y: int, z: int) -> List[Move]:
    """Single-call slider over all 23 queen directions."""
    return generate_moves(
        piece_type='queen',
        pos=(x, y, z),
        color=color.value,
        max_distance=8,
        directions=QUEEN_DIRECTIONS,
        occupancy=cache.occupancy._occ,    # ← pass the 3-D mask instead of the cache
    )

@register(PieceType.QUEEN)
def queen_move_dispatcher(state, x: int, y: int, z: int) -> List[Move]:
    return generate_queen_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_queen_moves"]
