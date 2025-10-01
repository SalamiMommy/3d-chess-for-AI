"""Knight (3,1,1) leaper — 3-D knight using the integrated jump engine."""
from __future__ import annotations

from typing import List
import numpy as np

from game3d.pieces.enums import PieceType, Color
 
from game3d.movement.movepiece import Move
from game3d.cache.manager import OptimizedCacheManager
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator

# ------------------------------------------------------------------
#  24 unique (3,1,1) permutations with all sign combinations
# ------------------------------------------------------------------
VECTORS_31 = np.array([
    # (3,1,1) family
    (3, 1, 1), (3, 1, -1), (3, -1, 1), (3, -1, -1),
    (-3, 1, 1), (-3, 1, -1), (-3, -1, 1), (-3, -1, -3),
    # (1,3,1) family
    (1, 3, 1), (1, 3, -1), (1, -3, 1), (1, -3, -1),
    (-1, 3, 1), (-1, 3, -1), (-1, -3, 1), (-1, -3, -1),
    # (1,1,3) family
    (1, 1, 3), (1, 1, -3), (1, -1, 3), (1, -1, -3),
    (-1, 1, 3), (-1, 1, -3), (-1, -1, 3), (-1, -1, -3),
], dtype=np.int8)

# ------------------------------------------------------------------
#  Move generator
# ------------------------------------------------------------------
def generate_knight31_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate all legal (3,1,1) knight leaps."""
    pos = (x, y, z)

      

    jump_gen = get_integrated_jump_movement_generator(cache)
    return jump_gen.generate_jump_moves(
        color=color,
        position=pos,
        directions=VECTORS_31,
        allow_capture=True,
    )
