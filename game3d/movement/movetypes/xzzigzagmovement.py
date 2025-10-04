"""XZ-Zig-Zag Slider — zig-zag rays via slidermovement engine."""

import numpy as np
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.slidermovement import get_slider_generator
from game3d.cache.manager import OptimizedCacheManager

# ---------------------------------------------------------------------------
#  Pre-build every square visited by every zig-zag ray
#  (same logic as YZ-Zig-Zag, just different fixed axis)
# ---------------------------------------------------------------------------
def _build_xz_zigzag_vectors() -> np.ndarray:
    vecs = []
    for plane, fixed_axis in [('XZ', 1), ('XY', 2), ('YZ', 0)]:
        for pri, sec in [(1, -1), (-1, 1)]:
            seq = []
            curr = np.zeros(3, dtype=np.int8)
            move_primary = True
            for seg in range(3):                    # 3 segments → 9 steps
                step = np.zeros(3, dtype=np.int8)
                if plane == 'XZ':
                    step[0 if move_primary else 2] = pri if move_primary else sec
                elif plane == 'XY':
                    step[0 if move_primary else 1] = pri if move_primary else sec
                else:                               # YZ
                    step[1 if move_primary else 2] = pri if move_primary else sec
                for _ in range(3):
                    curr = curr + step
                    seq.append(curr.copy())
                move_primary ^= 1
            vecs.extend(seq)
    return np.array(vecs, dtype=np.int8)

XZ_ZIGZAG_DIRECTIONS = _build_xz_zigzag_vectors()

# ---------------------------------------------------------------------------
# Public drop-in replacement
# ---------------------------------------------------------------------------
def generate_xz_zigzag_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate all XZ-zig-zag moves via slidermovement engine."""
    engine = get_slider_generator(cache)
    return engine.generate(
        color=color,
        ptype=PieceType.XZZIGZAG,   # <-- NEW
        pos=(x, y, z),
        directions=XZ_ZIGZAG_DIRECTIONS,
        max_steps=1,
       
        
        
        
    )
