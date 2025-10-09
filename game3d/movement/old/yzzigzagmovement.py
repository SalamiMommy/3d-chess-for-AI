"""YZ-Zig-Zag Slider — zig-zag rays via slidermovement engine."""
from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds
from game3d.movement.movetypes.slidermovement import get_slider_generator, generate_slider_moves_kernel
if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager as CacheManager

# ---------------------------------------------------------------------------
#  Pre-build every square visited by every zig-zag ray
#  (aligned to match XZ implementation for consistency)
# ---------------------------------------------------------------------------
def _build_yz_zigzag_vectors() -> np.ndarray:
    vecs = []
    for plane, fixed_axis in [('YZ', 0), ('XZ', 1), ('XY', 2)]:
        for pri, sec in [(1, -1), (-1, 1)]:
            seq = []
            curr = np.zeros(3, dtype=np.int8)
            move_primary = True
            for seg in range(3):                    # 3 segments → 9 steps
                step = np.zeros(3, dtype=np.int8)
                if plane == 'YZ':
                    step[1 if move_primary else 2] = pri if move_primary else sec
                elif plane == 'XZ':
                    step[0 if move_primary else 2] = pri if move_primary else sec
                else:                               # XY
                    step[0 if move_primary else 1] = pri if move_primary else sec
                for _ in range(3):
                    curr = curr + step
                    seq.append(curr.copy())
                move_primary ^= 1
            vecs.extend(seq)
    return np.array(vecs, dtype=np.int8)

YZ_ZIGZAG_DIRECTIONS = _build_yz_zigzag_vectors()

# ---------------------------------------------------------------------------
# Custom zigzag generator that uses the slider kernel directly
# (reused from XZ for consistency; could share a global if needed)
# ---------------------------------------------------------------------------
class ZigzagMovementGenerator:
    __slots__ = ("_move_cache", "_cache_hits", "_cache_misses")

    def __init__(self):
        self._move_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def generate_moves(
        self,
        pos: tuple[int, int, int],
        board_occupancy: np.ndarray,
        color: int,
        directions: np.ndarray,
        max_distance: int = 16
    ) -> List[Move]:
        """Generate zigzag moves using the slider kernel directly."""
        # Create cache key
        cache_key = (pos, board_occupancy.tobytes(), color, directions.tobytes())

        # Check cache
        if cache_key in self._move_cache:
            self._cache_hits += 1
            return self._move_cache[cache_key]

        self._cache_misses += 1

        # Generate moves using Numba kernel
        raw_moves = generate_slider_moves_kernel(
            pos, directions, board_occupancy, color, max_distance
        )

        # Convert to Move objects
        moves = []
        for nx, ny, nz, is_capture in raw_moves:
            moves.append(Move(
                from_coord=pos,
                to_coord=(nx, ny, nz),
                flags=MOVE_FLAGS['CAPTURE'] if is_capture else 0
                # Removed captured_piece=None (unnecessary and causes attribute issues)
            ))

        # Cache result
        if len(self._move_cache) > 5000:  # Prevent unbounded growth
            self._move_cache.clear()
        self._move_cache[cache_key] = moves

        return moves

# Global instance for reuse
_global_zigzag_gen = None

def get_zigzag_generator():
    """Get or create global zigzag generator instance."""
    global _global_zigzag_gen
    if _global_zigzag_gen is None:
        _global_zigzag_gen = ZigzagMovementGenerator()
    return _global_zigzag_gen

# ---------------------------------------------------------------------------
# Public drop-in replacement
# ---------------------------------------------------------------------------
def generate_yz_zigzag_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate all YZ-zig-zag moves via slidermovement engine."""
    engine = get_zigzag_generator()

    # Get the occupancy array with color codes from the piece cache
    piece_occ = cache.piece_cache.export_arrays()[0]  # This is a 9x9x9 array with 0,1,2

    return engine.generate_moves(
        pos=(x, y, z),
        board_occupancy=piece_occ,
        color=color.value if isinstance(color, Color) else color,
        directions=YZ_ZIGZAG_DIRECTIONS,
        max_distance=16
    )
