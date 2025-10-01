"""YZ-Zig-Zag Slider — zig-zag rays via slidermovement engine."""

import numpy as np
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.slidermovement import get_integrated_movement_generator
from game3d.cache.manager import OptimizedCacheManager

# ---------------------------------------------------------------------------
#  Build the 24 zig-zag step vectors once (6 planes × 4 directions)
#  Each segment is 3 steps, then axis flip.
#  We pre-compute the absolute squares visited.
# ---------------------------------------------------------------------------
def _build_zigzag_directions() -> np.ndarray:
    dirs = []
    for plane, fixed in [('YZ', 0), ('XZ', 1), ('XY', 2)]:        # which coord stays
        for pri, sec in [(1, -1), (-1, 1)]:                       # primary / secondary signs
            seq = []
            curr = np.array([0, 0, 0], dtype=np.int8)
            move_axis = 0                                           # 0 = primary, 1 = secondary
            for seg in range(3):                                    # 3 segments → 9 steps
                step_vec = np.zeros(3, dtype=np.int8)
                if plane == 'YZ':
                    step_vec[1 + move_axis] = pri if move_axis == 0 else sec
                elif plane == 'XZ':
                    step_vec[0 if move_axis == 0 else 2] = pri if move_axis == 0 else sec
                else:                                               # XY
                    step_vec[move_axis] = pri if move_axis == 0 else sec
                for _ in range(3):
                    curr = curr + step_vec
                    seq.append(curr.copy())
                move_axis ^= 1                                      # flip axis
            dirs.extend(seq)
    return np.array(dirs, dtype=np.int8)

ZIGZAG_DIRECTIONS = _build_zigzag_directions()

# ---------------------------------------------------------------------------
# Public API — drop-in replacement
# ---------------------------------------------------------------------------
def generate_yz_zigzag_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate all zig-zag moves via slidermovement engine."""
    engine = get_integrated_movement_generator(cache)
    return engine.generate_sliding_moves(
        color=color,
        piece_type=PieceType.YZZIGZAG,   # <-- NEW
        position=(x, y, z),
        directions=ZIGZAG_DIRECTIONS,
        max_steps=1,
        allow_capture=True,
        allow_self_block=False,
        use_symmetry=True,
        use_amd=True
    )
