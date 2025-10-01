"""Reflecting Bishop — wall-bouncing diagonal rays via slidermovement engine."""

from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from numba import njit, prange
from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.slidermovement import (
    get_integrated_movement_generator,
    IntegratedSlidingMovementGenerator,_build_compact
)
from game3d.cache.manager import OptimizedCacheManager

if TYPE_CHECKING:
    from game3d.cache.transposition import CompactMove

# --------------------------------------------------------------------------- #
#  Geometry  – 8 diagonal directions
# --------------------------------------------------------------------------- #
BISHOP_DIRECTIONS = np.array(
    [(dx, dy, dz) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)],
    dtype=np.int8,
)

# --------------------------------------------------------------------------- #
#  NumPy bounce kernel – runs inside slidermovement engine
# --------------------------------------------------------------------------- #
@njit(cache=True, inline="always")
def _occ_code(occ: np.ndarray, x: int, y: int, z: int) -> int:
    return occ[x, y, z]

MAX_BOUNCES = 3
BOARD_SIZE  = 9
MAX_STEPS   = 24          # we promised 24 steps max

@njit(parallel=True, fastmath=True, nogil=False, cache=True)  # OPTIMIZED: Enabled parallel=True for potential perf gain on larger dir sets
def _reflecting_slide_kernel(
    starts: np.ndarray,          # (D,3)  int8
    dirs: np.ndarray,            # (D,3)  int8
    occ: np.ndarray,             # (9,9,9) uint8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    D = starts.shape[0]
    # pre-allocate with *literal* shape – Numba is happy
    coords = np.empty((D, MAX_STEPS, 3), np.int8)
    valid  = np.zeros((D, MAX_STEPS), np.bool_)
    hit    = np.zeros((D, MAX_STEPS), np.uint8)

    for d in prange(D):
        sx, sy, sz = starts[d]
        dx, dy, dz = dirs[d]
        bounces = 0
        for s in range(MAX_STEPS):        # 0-based now
            tx = sx + dx
            ty = sy + dy
            tz = sz + dz

            out = (tx < 0) | (tx >= BOARD_SIZE) | (ty < 0) | (ty >= BOARD_SIZE) | (tz < 0) | (tz >= BOARD_SIZE)
            if out:
                if bounces >= MAX_BOUNCES:
                    break
                if tx < 0 or tx >= BOARD_SIZE:
                    dx = -dx
                if ty < 0 or ty >= BOARD_SIZE:
                    dy = -dy
                if tz < 0 or tz >= BOARD_SIZE:
                    dz = -dz
                bounces += 1
                continue

            coords[d, s] = (tx, ty, tz)
            valid[d, s]  = True
            code = _occ_code(occ, tx, ty, tz)
            if code:
                hit[d, s] = code
                break
            sx, sy, sz = tx, ty, tz
    return coords, valid, hit

# --------------------------------------------------------------------------- #
#  Plug the new kernel into the engine
# --------------------------------------------------------------------------- #
class ReflectingBishopGenerator(IntegratedSlidingMovementGenerator):
    """Bouncing-ray kernel."""

    def _compute_compact(
        self,
        color: Color,
        piece_type: PieceType,
        pos: Tuple[int, int, int],
        dirs: np.ndarray,
        max_steps: int,
        allow_capture: bool,
        use_symmetry: bool,          # ← NEW
        use_amd: bool,               # ← NEW
    ) -> List['CompactMove']:
        occ, _ = self.cache.piece_cache.export_arrays()
        starts = np.repeat(np.array([pos], dtype=np.int8), len(dirs), axis=0)
        coords, valid, hit = _reflecting_slide_kernel(starts, dirs, occ)
        return _build_compact(color, piece_type, pos, coords, valid, hit, allow_capture)

    # Re-use parent _build_compact / _expand_compact unchanged
    def _build_compact(self, color, piece_type, pos, coords, valid, hit, allow_capture):  # FIXED: Added piece_type to signature and call
        return _build_compact(color, piece_type, pos, coords, valid, hit, allow_capture)

# --------------------------------------------------------------------------- #
#  Singleton helper
# --------------------------------------------------------------------------- #
def get_reflecting_bishop_generator(cache: OptimizedCacheManager) -> ReflectingBishopGenerator:
    if not hasattr(cache, "_reflecting_bishop_gen"):
        cache._reflecting_bishop_gen = ReflectingBishopGenerator(cache)
    return cache._reflecting_bishop_gen

# --------------------------------------------------------------------------- #
#  Public drop-in replacement
# --------------------------------------------------------------------------- #
def generate_reflecting_bishop_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int,
    y: int,
    z: int
) -> List[Move]:
    engine = get_reflecting_bishop_generator(cache)
    return engine.generate_sliding_moves(
        color=color,
        piece_type=PieceType.REFLECTOR,
        position=(x, y, z),
        directions=BISHOP_DIRECTIONS,
        max_steps=24,
        allow_capture=True,
        use_symmetry=True,
        use_amd=True
    )
