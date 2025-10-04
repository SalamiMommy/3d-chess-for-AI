# game3d/movement/movetypes/slidermovement.py
from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
import numpy as np
import threading  # ADDED: Missing import
from numba import njit, prange, typeof
from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.cache.manager import CacheManager

# ----------  constants  ----------
_SQ = 9
_CUBE = _SQ * _SQ * _SQ
_RAYS = 13 * 9 * 9
_TO_MASK   = 0x1FF
_FROM_MASK = 0x1FF << 21
_CAPT_BIT  = 1 << 42
_PROM_BIT  = 1 << 43

# ----------  low-level  ----------
@njit(inline="always")
def _coord_to_idx(x: int, y: int, z: int) -> int:
    return x * 81 + y * 9 + z

@njit(inline="always")
def _idx_to_coord(idx: int) -> Tuple[int, int, int]:
    x = idx // 81
    y = (idx // 9) % 9
    z = idx % 9
    return x, y, z

# ----------  ray blocker cache  ----------
_ray_blocker = np.full(_RAYS, -1, dtype=np.int16)
_ray_id_map  = np.zeros((_CUBE, 27), dtype=np.int16)
_ray_offset  = np.zeros((_CUBE, 27), dtype=np.int8)

# ----------  optimized single direction slide  ----------
@njit(fastmath=True, boundscheck=False, cache=True)
def _slide_single_direction(
    occ: np.ndarray,
    from_sq: int,
    dx: int, dy: int, dz: int,
    max_steps: int,
    move_buffer: np.ndarray,
    buffer_offset: int
) -> int:
    """Slide in one direction, return new buffer offset"""
    x, y, z = _idx_to_coord(from_sq)
    step = 0

    ray_dir = (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)
    ray_id = _ray_id_map[from_sq, ray_dir] if 0 <= ray_dir < 27 else -1
    blocker = _ray_blocker[ray_id] if 0 <= ray_id < _RAYS else -1

    while step < max_steps:
        step += 1
        tx, ty, tz = x + step * dx, y + step * dy, z + step * dz

        if not (0 <= tx < 9 and 0 <= ty < 9 and 0 <= tz < 9):
            break

        to_sq = _coord_to_idx(tx, ty, tz)

        if blocker != -1 and to_sq == blocker:
            if occ[to_sq]:
                piece_code = occ[to_sq]
                packed = (from_sq << 21) | to_sq
                move_buffer[buffer_offset] = packed | _CAPT_BIT
                buffer_offset += 1
            break

        piece_code = occ[to_sq]
        packed = (from_sq << 21) | to_sq

        if piece_code == 0:
            move_buffer[buffer_offset] = packed
            buffer_offset += 1
        else:
            move_buffer[buffer_offset] = packed | _CAPT_BIT
            buffer_offset += 1
            break

    return buffer_offset

@njit(fastmath=True, boundscheck=False, cache=True)
def _batch_slide_complex(
    occ: np.ndarray,
    from_sq: int,
    directions: np.ndarray,
    max_steps: int,
    move_buffer: np.ndarray
) -> int:
    """Generate moves in all complex directions, return count"""
    buffer_ptr = 0

    for i in range(directions.shape[0]):
        dx = directions[i, 0]
        dy = directions[i, 1]
        dz = directions[i, 2]

        buffer_ptr = _slide_single_direction(
            occ, from_sq, dx, dy, dz, max_steps, move_buffer, buffer_ptr
        )

        if buffer_ptr >= move_buffer.shape[0]:
            break

    return buffer_ptr

@njit(cache=True)
def _update_blocker(occ: np.ndarray, sq: int, add: bool):
    """Maintain _ray_blocker when a piece lands/leaves sq."""
    x, y, z = _idx_to_coord(sq)

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue

                ray_dir = (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)
                ray_id = _ray_id_map[sq, ray_dir]
                if ray_id == -1:
                    continue

                if add:
                    if _ray_blocker[ray_id] == -1:
                        _ray_blocker[ray_id] = sq
                else:
                    if _ray_blocker[ray_id] == sq:
                        next_blocker = -1
                        step = 1

                        while step < 9:
                            tx, ty, tz = x + step * dx, y + step * dy, z + step * dz

                            if not (0 <= tx < 9 and 0 <= ty < 9 and 0 <= tz < 9):
                                break

                            check_sq = _coord_to_idx(tx, ty, tz)
                            if occ[check_sq]:
                                next_blocker = check_sq
                                break

                            step += 1

                        _ray_blocker[ray_id] = next_blocker

class SliderGenerator:
    __slots__ = ("cache", "_move_buffer", "_buffer_lock")

    def __init__(self, cache: CacheManager):
        self.cache = cache
        self._move_buffer = np.empty(2000, dtype=np.uint64)
        self._buffer_lock = threading.Lock()

    def generate(
        self,
        color: Color,
        ptype: PieceType,
        pos: Tuple[int, int, int],
        directions: np.ndarray,
        max_steps: int = 8,
    ) -> List[Move]:
        with self._buffer_lock:
            occ = self.cache.piece_cache.get_flat_occupancy()
            from_sq = _coord_to_idx(*pos)

            move_count = _batch_slide_complex(
                occ, from_sq, directions, max_steps, self._move_buffer
            )

            moves = [None] * move_count
            for i in range(move_count):
                packed = self._move_buffer[i]

                from_idx = (packed >> 21) & _TO_MASK
                to_idx = packed & _TO_MASK
                is_cap = bool(packed & _CAPT_BIT)

                from_coord = _idx_to_coord(from_idx)
                to_coord = _idx_to_coord(to_idx)

                moves[i] = Move(
                    from_coord=from_coord,
                    to_coord=to_coord,
                    is_capture=is_cap,
                    captured_piece=None,
                )

            return moves

    def update_blocker(self, sq: Tuple[int, int, int], add: bool):
        """Update blocker cache when piece moves."""
        sq_idx = _coord_to_idx(*sq)
        occ = self.cache.piece_cache.get_flat_occupancy()
        _update_blocker(occ, sq_idx, add)

def get_slider_generator(cache: CacheManager) -> SliderGenerator:
    """Cached singleton - avoid repeated initialization"""
    if not hasattr(cache, "_slider_gen"):
        cache._slider_gen = SliderGenerator(cache)
    return cache._slider_gen
