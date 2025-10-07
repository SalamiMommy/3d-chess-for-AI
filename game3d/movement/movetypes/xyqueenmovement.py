"""3D XY-Queen movement logic — 2-D queen in XY-plane via slidermovement."""
from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds
from game3d.movement.movetypes.slidermovement import get_slider_generator
if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager as CacheManager

# 8 directions in XY-plane (Z fixed)
XY_QUEEN_DIRECTIONS = np.array([
    (1, 0, 0), (-1, 0, 0),   # X axis
    (0, 1, 0), (0, -1, 0),   # Y axis
    (1, 1, 0), (1, -1, 0),   # XY diagonals
    (-1, 1, 0), (-1, -1, 0)
], dtype=np.int8)

def generate_xy_queen_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate all legal XY-QUEEN moves from (x, y, z)."""
    engine = get_slider_generator()  # Fixed: no arguments needed
    return engine.generate_moves(    # Fixed: changed method name
        piece_type='xy_queen',      # Added piece_type
        pos=(x, y, z),
        board_occupancy=cache.occupancy.mask
,  # Added board_occupancy
        color=color.value if isinstance(color, Color) else color,  # Convert to int
        max_distance=8,              # Changed from max_steps
    )
