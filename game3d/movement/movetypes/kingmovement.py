"""3-D King move generator — reusable movement module (no registration)."""
from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.enums import Color
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.cache_utils import ensure_int_coords

if TYPE_CHECKING:
    from game3d.movement.movepiece import Move
    from game3d.cache.manager import OptimizedCacheManager

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
#  Public helper – identical signature to kinglike.py
# ------------------------------------------------------------------
def generate_king_moves(
    cache_manager: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Return all legal one-step king moves for the given square."""
    x, y, z = ensure_int_coords(x, y, z)
    jump_gen = get_integrated_jump_movement_generator(cache_manager)
    return jump_gen.generate_jump_moves(
        color=color,
        pos=(x, y, z),
        directions=KING_DIRECTIONS_3D,
    )

# ------------------------------------------------------------------
#  Convenience aliases for pieces that want to re-export them
# ------------------------------------------------------------------
generate_priest_moves = generate_king_moves

__all__ = [
    "generate_king_moves",
    "generate_priest_moves",
]
