"""Knight (3,2,1) leaper — 3-D knight using the integrated jump engine."""
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
#  48 unique (3,2,1) permutations with all sign combinations
# ------------------------------------------------------------------
VECTORS_32 = np.array([
    # (3,2,1) family
    (3, 2, 1), (3, 2, -1), (3, -2, 1), (3, -2, -1),
    (-3, 2, 1), (-3, 2, -1), (-3, -2, 1), (-3, -2, -1),
    # (3,1,2) family
    (3, 1, 2), (3, 1, -2), (3, -1, 2), (3, -1, -2),
    (-3, 1, 2), (-3, 1, -2), (-3, -1, 2), (-3, -1, -2),
    # (2,3,1) family
    (2, 3, 1), (2, 3, -1), (2, -3, 1), (2, -3, -1),
    (-2, 3, 1), (-2, 3, -1), (-2, -3, 1), (-2, -3, -1),
    # (1,3,2) family
    (1, 3, 2), (1, 3, -2), (1, -3, 2), (1, -3, -2),
    (-1, 3, 2), (-1, 3, -2), (-1, -3, 2), (-1, -3, -2),
    # (2,1,3) family
    (2, 1, 3), (2, 1, -3), (2, -1, 3), (2, -1, -3),
    (-2, 1, 3), (-2, 1, -3), (-2, -1, 3), (-2, -1, -3),
    # (1,2,3) family
    (1, 2, 3), (1, 2, -3), (1, -2, 3), (1, -2, -3),
    (-1, 2, 3), (-1, 2, -3), (-1, -2, 3), (-1, -2, -3),
], dtype=np.int8)

# ------------------------------------------------------------------
#  Move generator
# ------------------------------------------------------------------
def generate_knight32_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate all legal (3,2,1) knight leaps."""
    pos = (x, y, z)

    jump_gen = get_integrated_jump_movement_generator(cache)
    return jump_gen.generate_jump_moves(
        color=color,
        pos=pos,
        directions=VECTORS_32,
       
    )
