"""Panel piece — 6 orthogonal 3×3 walls 2 steps away, computed via the jump engine."""
from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager as CacheManager

# 54 offsets: six 3×3 walls, 2 units away along each axis
# game3d/movement/movetypes/panelmovement.py  (top portion)

_PANEL_DIRECTIONS: np.ndarray = np.array([
    # +X wall
    *[( 2, wy, wz) for wy in (-1, 0, 1) for wz in (-1, 0, 1)],
    # +Y wall
    *[(wx,  2, wz) for wx in (-1, 0, 1) for wz in (-1, 0, 1)],
    # +Z wall
    *[(wx, wy,  2) for wx in (-1, 0, 1) for wy in (-1, 0, 1)],
    # -X wall
    *[(-2, wy, wz) for wy in (-1, 0, 1) for wz in (-1, 0, 1)],
    # -Y wall
    *[(wx, -2, wz) for wx in (-1, 0, 1) for wz in (-1, 0, 1)],
    # -Z wall
    *[(wx, wy, -2) for wx in (-1, 0, 1) for wy in (-1, 0, 1)]
], dtype=np.int8)

def generate_panel_moves(
    cache: CacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List['Move']:
    """Generate all legal Panel moves from (x, y, z) using the jump engine."""
    gen = get_integrated_jump_movement_generator(cache)
    return gen.generate_jump_moves(
        color=color,
        pos=(x, y, z),
        directions=_PANEL_DIRECTIONS,
       
                  # keep GPU path when available
    )

# Optional helper for external consumers
def get_panel_directions() -> np.ndarray:
    return _PANEL_DIRECTIONS.copy()
