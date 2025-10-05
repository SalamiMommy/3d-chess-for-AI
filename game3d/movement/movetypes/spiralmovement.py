"""Counter-clockwise 3-D spiral slider — 6 fixed-offset rays via slidermovement."""

from typing import List, Tuple
import numpy as np
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.slidermovement import get_slider_generator
from game3d.cache.manager import OptimizedCacheManager

# --------------------------------------------------------------------------- #
#  Pre-computed spiral offsets (same data, now NumPy)
# --------------------------------------------------------------------------- #
_SPIRAL_VECTORS: List[Tuple[Tuple[int, int, int], np.ndarray]] = [
    (( 1, 0, 0), np.array([(0, 0, 0), (0, 1, 0), (-1, 1, 0), (-1, 0, 0),
                            (-1, -1, 0), (0, -1, 0), (1, -1, 0), (1, 0, 0)], dtype=np.int8)),
    ((-1, 0, 0), np.array([(0, 0, 0), (0, -1, 0), (-1, -1, 0), (-1, 0, 0),
                            (-1, 1, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)], dtype=np.int8)),
    (( 0, 1, 0), np.array([(0, 0, 0), (-1, 0, 0), (-1, 0, 1), (0, 0, 1),
                            (1, 0, 1), (1, 0, 0), (1, 0, -1), (0, 0, -1)], dtype=np.int8)),
    (( 0, -1, 0), np.array([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1),
                             (-1, 0, 1), (-1, 0, 0), (-1, 0, -1), (0, 0, -1)], dtype=np.int8)),
    (( 0, 0, 1), np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                            (-1, 1, 0), (-1, 0, 0), (-1, -1, 0), (0, -1, 0)], dtype=np.int8)),
    (( 0, 0, -1), np.array([(0, 0, 0), (-1, 0, 0), (-1, 1, 0), (0, 1, 0),
                             (1, 1, 0), (1, 0, 0), (1, -1, 0), (0, -1, 0)], dtype=np.int8)),
]

# Build one flat array of absolute step vectors for the slidermovement engine
SPIRAL_DIRECTIONS = np.vstack([off + np.array(dir_ax) * (i + 1)
                               for dir_ax, offsets in _SPIRAL_VECTORS
                               for i, off in enumerate(offsets)])

def generate_spiral_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int,
    y: int,
    z: int
) -> List[Move]:
    engine = get_slider_generator()  # Fixed: removed cache argument
    return engine.generate_moves(    # Fixed: changed method name
        piece_type='spiral',         # Added piece_type
        pos=(x, y, z),
        board_occupancy=cache.occupancy.mask
,  # Added board_occupancy
        color=color.value if isinstance(color, Color) else color,  # Convert to int
        max_distance=16,              # Changed from max_steps
    )
