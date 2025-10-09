# reflectingbishopmoves.py
"""Reflecting-Bishop – diagonal slider that bounces off walls (max 3 reflections)."""

from __future__ import annotations

import numpy as np
from numba import njit
from typing import List, TYPE_CHECKING
from game3d.pieces.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds
from game3d.movement.movepiece import convert_legacy_move_args
if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager

# ----------------------------------------------------------
# 8 pure-diagonal unit directions
# ----------------------------------------------------------
_BISHOP_DIRS = np.array(
    [(dx, dy, dz) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)],
    dtype=np.int8,
)

# ----------------------------------------------------------
# Bouncing kernel – walks one ray, reflects ≤3×, returns
# packed (from_idx, to_idx, is_capture) into uint64 buffer
# ----------------------------------------------------------
@njit(fastmath=True, boundscheck=False, cache=True)
def _bounce_ray(
    occ_flat,          # flat uint8[729]  (x + y*9 + z*81)
    start_x, start_y, start_z,
    dx, dy, dz,
    max_bounces: int,
    buf,               # uint64[256]  pre-allocated output
    color_code: int,   # 1 white  2 black
) -> int:              # number of moves written
    ptr = 0
    sx, sy, sz = start_x, start_y, start_z
    bounces = 0
    start_idx = start_x + start_y * 9 + start_z * 81

    for _ in range(24):          # 24 steps is plenty on 9×9×9
        tx = sx + dx
        ty = sy + dy
        tz = sz + dz

        # ---- wall reflection ----
        out = (tx < 0) | (tx >= 9) | (ty < 0) | (ty >= 9) | (tz < 0) | (tz >= 9)
        if out:
            if bounces >= max_bounces:
                break
            # mirror offending component(s)
            if tx < 0 or tx >= 9:
                dx = -dx
            if ty < 0 or ty >= 9:
                dy = -dy
            if tz < 0 or tz >= 9:
                dz = -dz
            bounces += 1
            continue

        # ---- occupancy test ----
        to_idx = tx + ty * 9 + tz * 81
        code = occ_flat[to_idx]
        if code == 0:                                    # empty
            packed = (start_idx << 21) | to_idx
            buf[ptr] = packed
            ptr += 1
        else:                                            # blocked
            if code != color_code:                       # enemy
                packed = (start_idx << 21) | to_idx | (1 << 42)
                buf[ptr] = packed
                ptr += 1
            break
        sx, sy, sz = tx, ty, tz
    return ptr

# ----------------------------------------------------------
# Generator class – thin Python wrapper
# ----------------------------------------------------------
class _ReflectingBishopGen:
    __slots__ = ("cache", "_buf")

    def __init__(self, cache: OptimizedCacheManager):
        self.cache = cache
        self._buf = np.empty(256, dtype=np.uint64)

    def generate(self, color: Color, pos: tuple[int, int, int]) -> List[Move]:
        occ = self.cache.piece_cache.get_flat_occupancy()
        x, y, z = pos
        color_code = 1 if color == Color.WHITE else 2
        moves: List[Move] = []

        for dx, dy, dz in _BISHOP_DIRS:
            n = _bounce_ray(occ, x, y, z, dx, dy, dz, 3, self._buf, color_code)
            for i in range(n):
                packed = self._buf[i]
                fr_idx = (packed >> 21) & 0x3FFFF
                to_idx = packed & 0x3FFFF
                is_cap = bool(packed & (1 << 42))

                from_coord = (fr_idx % 9, (fr_idx // 9) % 9, fr_idx // 81)
                to_coord   = (to_idx % 9, (to_idx // 9) % 9, to_idx // 81)

                moves.append(convert_legacy_move_args(
                    from_coord=from_coord,
                    to_coord=to_coord,
                    is_capture=is_cap
                ))
        return moves

# ----------------------------------------------------------
# Singleton helper
# ----------------------------------------------------------
def _get_gen(cache: OptimizedCacheManager) -> _ReflectingBishopGen:
    # double-checked pattern
    gen = cache._reflecting_bishop_gen
    if gen is None:
        gen = _ReflectingBishopGen(cache)
        cache._reflecting_bishop_gen = gen
    return gen
# ----------------------------------------------------------
# Public API + dispatcher
# ----------------------------------------------------------
def generate_reflecting_bishop_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    return _get_gen(cache).generate(color, (x, y, z))

@register(PieceType.REFLECTOR)
def reflector_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    gen = _get_gen(state.cache)
    assert gen is not None, "Reflector generator is None – cache helper broken?"
    return gen.generate(state.color, (x, y, z))

__all__ = ["generate_reflecting_bishop_moves"]
