"""3-D King move generation — now powered by the integrated jump engine."""
from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager as CacheManager

# ------------------------------------------------------------------
#  26 one-step vectors (unchanged)
# ------------------------------------------------------------------
KING_DIRECTIONS_3D = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

# ------------------------------------------------------------------
#  Move generator
# ------------------------------------------------------------------
def generate_king_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """
    Generate all legal king moves (single-step, no castling).

    Delegates final-square legality to the integrated jump generator:
    - off-board          → discarded
    - friendly piece     → discarded
    - enemy king w/ priests → discarded
    - wall               → discarded
    """
    pos = (x, y, z)

    jump_gen = get_integrated_jump_movement_generator(cache)
    return jump_gen.generate_jump_moves(
        color=color,
        pos=pos,
        directions=KING_DIRECTIONS_3D,
       
    )
