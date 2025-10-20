"""3-D King move generator — reusable movement module (no registration)."""
from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.enums import Color
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator

if TYPE_CHECKING:
    from game3d.movement.movepiece import Move
    # Any object that quacks like a CacheManager
    from game3d.game.gamestate import GameState  # or your real state type

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
    cache_manager,  # any cache manager with .occupancy
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Return all legal one-step king moves for the given square."""
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
