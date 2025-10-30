# reflector.py - OPTIMIZED WITH BATCHING
"""Reflecting-Bishop – diagonal slider that bounces off walls (max 3 reflections)."""

from __future__ import annotations

import numpy as np
from numba import njit
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import in_bounds
from game3d.movement.movepiece import Move
from game3d.common.cache_utils import ensure_int_coords

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

# 8 pure-diagonal unit directions
_BISHOP_DIRS = np.array(
    [(dx, dy, dz) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)],
    dtype=np.int8,
)

@njit(fastmath=True, boundscheck=False, cache=True)
def _bounce_ray(
    occ_flat,
    start_x, start_y, start_z,
    dx, dy, dz,
    max_bounces: int,
    buf,
    color_code: int,
) -> int:
    ptr = 0
    sx, sy, sz = start_x, start_y, start_z
    bounces = 0
    start_idx = start_x + start_y * 9 + start_z * 81

    for _ in range(24):
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
        if code == 0:
            packed = (start_idx << 21) | to_idx
            buf[ptr] = packed
            ptr += 1
        else:
            if code != color_code:
                packed = (start_idx << 21) | to_idx | (1 << 42)
                buf[ptr] = packed
                ptr += 1
            break
        sx, sy, sz = tx, ty, tz
    return ptr

class _ReflectingBishopGen:
    __slots__ = ("cache_manager", "_buf")

    def __init__(self, cache_manager: 'OptimizedCacheManager'):
        self.cache_manager = cache_manager
        self._buf = np.empty(256, dtype=np.uint64)

    def generate(self, color: Color, pos: tuple[int, int, int]) -> List[Move]:
        occ_array = self.cache_manager.get_occupancy_array_readonly()
        occ_flat = occ_array.reshape(-1)
        x, y, z = pos
        color_code = 1 if color == Color.WHITE else 2

        # OPTIMIZED: Collect all moves for batch processing
        to_coords_list = []
        capture_flags = []

        for dx, dy, dz in _BISHOP_DIRS:
            n = _bounce_ray(occ_flat, x, y, z, dx, dy, dz, 3, self._buf, color_code)
            for i in range(n):
                packed = self._buf[i]
                to_idx = packed & 0x3FFFF
                is_cap = bool(packed & (1 << 42))

                to_coord = (to_idx % 9, (to_idx // 9) % 9, to_idx // 81)
                to_coords_list.append(to_coord)
                capture_flags.append(is_cap)

        if not to_coords_list:
            return []

        # Use batch creation
        to_coords_array = np.array(to_coords_list, dtype=np.int8)
        captures_array = np.array(capture_flags, dtype=bool)
        return Move.create_batch(from_coord=pos, to_coords=to_coords_array, captures=captures_array)

def _get_gen(cache_manager: 'OptimizedCacheManager') -> _ReflectingBishopGen:
    """Use the cache manager's existing generator instead of creating new one."""
    if hasattr(cache_manager, '_reflecting_bishop_gen') and cache_manager._reflecting_bishop_gen is not None:
        return cache_manager._reflecting_bishop_gen

    cache_manager._reflecting_bishop_gen = _ReflectingBishopGen(cache_manager)
    return cache_manager._reflecting_bishop_gen

def generate_reflecting_bishop_moves(
    cache_manager: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return _get_gen(cache_manager).generate(color, (x, y, z))

@register(PieceType.REFLECTOR)
def reflector_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    gen = _get_gen(state.cache_manager)
    return gen.generate(state.color, (x, y, z))

__all__ = ["generate_reflecting_bishop_moves"]
