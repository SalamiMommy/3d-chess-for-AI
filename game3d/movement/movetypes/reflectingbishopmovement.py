# game3d/movement/movetypes/reflectingbishopmovement.py
from __future__ import annotations

from typing import List, Tuple
import numpy as np
from numba import njit
from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.slidermovement import get_slider_generator

# --------------------------------------------------------------------------- #
#  8 diagonal directions
# --------------------------------------------------------------------------- #
BISHOP_DIRS = np.array(
    [(dx, dy, dz) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)],
    dtype=np.int8,
)

# --------------------------------------------------------------------------- #
#  bouncing kernel – returns packed uint64 moves
# --------------------------------------------------------------------------- #
@njit(fastmath=True, boundscheck=False, cache=True)
def _bounce_kernel(
    occ: np.ndarray,          # flat uint8[729]
    start_idx: int,
    dirs: np.ndarray,         # (N,3) int8
    max_steps: int,
    buf: np.ndarray,          # uint64[128]  pre-allocated
) -> int:                     # number of moves written
    ptr = 0
    for d in range(len(dirs)):
        dx, dy, dz = dirs[d]
        sx = sy = sz = start_idx
        bounces = 0
        for _ in range(max_steps):
            tx = sx + dx
            ty = sy + dy
            tz = sz + dz
            out = (tx < 0) | (tx >= 9) | (ty < 0) | (ty >= 9) | (tz < 0) | (tz >= 9)
            if out:
                if bounces >= 3:
                    break
                # mirror direction
                if tx < 0 or tx >= 9:
                    dx = -dx
                if ty < 0 or ty >= 9:
                    dy = -dy
                if tz < 0 or tz >= 9:
                    dz = -dz
                bounces += 1
                continue

            to_idx = tx * 81 + ty * 9 + tz
            packed = (start_idx << 21) | to_idx
            code = occ[to_idx]
            if code == 0:                               # empty
                buf[ptr] = packed
                ptr += 1
            else:                                       # blocked
                if code & 0x80:                         # enemy
                    buf[ptr] = packed | (1 << 42)       # capture flag
                    ptr += 1
                break
            sx, sy, sz = tx, ty, tz
    return ptr

# --------------------------------------------------------------------------- #
#  public generator
# --------------------------------------------------------------------------- #
class ReflectingBishopGenerator:
    __slots__ = ("cache", "_buf")

    def __init__(self, cache):
        self.cache = cache
        self._buf = np.empty(128, dtype=np.uint64)

    def generate(self, color: Color, pos: Tuple[int, int, int]) -> List[Move]:
        occ = self.cache.piece_cache.get_flat_occupancy()
        f_idx = pos[0] * 81 + pos[1] * 9 + pos[2]
        n = _bounce_kernel(occ, f_idx, BISHOP_DIRS, 24, self._buf)

        moves: List[Move] = [None] * n
        for i in range(n):
            packed = self._buf[i]
            fr_idx = (packed >> 21) & 0x1FF
            to_idx = packed & 0x1FF
            moves[i] = Move(
                from_coord=(fr_idx // 81, (fr_idx // 9) % 9, fr_idx % 9),
                to_coord=(to_idx // 81, (to_idx // 9) % 9, to_idx % 9),
                is_capture=bool(packed & (1 << 42)),
                captured_piece=None,
            )
        return moves

# --------------------------------------------------------------------------- #
#  singleton helper
# --------------------------------------------------------------------------- #
def get_reflecting_bishop_generator(cache) -> ReflectingBishopGenerator:
    if not hasattr(cache, "_reflecting_bishop_gen"):
        cache._reflecting_bishop_gen = ReflectingBishopGenerator(cache)
    return cache._reflecting_bishop_gen

# --------------------------------------------------------------------------- #
#  drop-in dispatcher
# --------------------------------------------------------------------------- #
def generate_reflecting_bishop_moves(
    cache, color: Color, x: int, y: int, z: int
) -> List[Move]:
    return get_reflecting_bishop_generator(cache).generate(color, (x, y, z))
