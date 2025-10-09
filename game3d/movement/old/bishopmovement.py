# game3d/movement/movetypes/bishopmovement.py
"""3D Bishop move generation — now symmetry-aware via slidermovement engine."""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds
from game3d.movement.movetypes.slidermovement import get_slider_generator
if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager as CacheManager
# --------------------------------------------------------------------------- #
#  Public API — signature unchanged                                          #
# --------------------------------------------------------------------------- #
def generate_bishop_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate all legal bishop moves from (x, y, z) with symmetry optimisation."""
    engine = get_slider_generator()

    # Use the existing occupancy data directly instead of rebuilding
    occ, ptype = cache.occupancy.export_arrays()

    # Create a color-aware occupancy array more efficiently
    board_occupancy = np.zeros((9, 9, 9), dtype=np.int8)

    # Use vectorized operations instead of loops
    occupied_mask = occ > 0
    board_occupancy[occupied_mask] = occ[occupied_mask]

    col_val = color.value if isinstance(color, Color) else color

    return engine.generate_moves(
        piece_type='bishop',
        pos=(x, y, z),
        board_occupancy=board_occupancy,
        color=col_val,
        max_distance=8
    )
