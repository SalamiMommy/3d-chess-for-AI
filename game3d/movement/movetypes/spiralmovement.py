"""Counter-clockwise 3-D spiral slider — 6 fixed-offset rays via slidermovement."""

from typing import List, Tuple
import numpy as np
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.slidermovement import get_integrated_movement_generator
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
    engine = get_integrated_movement_generator(cache)
    return engine.generate_sliding_moves(
        color=color,
        piece_type=PieceType.SPIRAL,   # <-- NEW
        position=(x, y, z),
        directions=SPIRAL_DIRECTIONS,
        max_steps=1,
        allow_capture=True,
        allow_self_block=False,
        use_symmetry=True,
        use_amd=True
    )
