# game3d/movement/piecemoves/bishopmoves.py
"""
3-D Bishop – pure slider on every 2-axis diagonal.
One file → movement + registration + slider integration.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.pieces.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager

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
def generate_bishop_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Return all bishop slides up to 8 steps along 2-axis diagonals."""
    occ, _ = cache.occupancy.export_arrays()          # 9×9×9 int8 occupancy
    pos = (x, y, z)
    moves: List[Move] = []

    for dx, dy, dz in BISHOP_DIRECTIONS:
        for step in range(1, 9):                        # sliders can go 8 squares
            nx, ny, nz = x + step * dx, y + step * dy, z + step * dz
            if not in_bounds(nx, ny, nz):
                break
            occupant = occ[nz, ny, nx]
            if occupant == 0:                           # empty
                moves.append(Move(pos, (nx, ny, nz), flags=0))
            elif occupant != color.value:               # enemy
                moves.append(Move(pos, (nx, ny, nz), flags=MOVE_FLAGS['CAPTURE']))
                break
            else:                                       # friendly
                break
    return moves

# --------------------------------------------------------------------------- #
#  Dispatcher registration – done here, no external file needed               #
# --------------------------------------------------------------------------- #
@register(PieceType.BISHOP)
def bishop_move_dispatcher(state: "GameState", x: int, y: int, z: int) -> List[Move]:
    return generate_bishop_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_bishop_moves"]   # kept for imports
