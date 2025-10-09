# game3d/movement/movetypes/slidermovement.py
"""Optimized slider movement generation for 3D chess
Now supports disk-backed precomputed move rays, fallback to original vectorized kernel."""
from __future__ import annotations
import os
from typing import TYPE_CHECKING, List, Tuple, Optional
import numpy as np
from numba import njit, prange
from functools import lru_cache
from itertools import product

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move, MOVE_FLAGS, convert_legacy_move_args
from game3d.common.common import coord_to_idx, in_bounds

# ---------- direction tables ----------
SLIDER_DIRECTIONS = {
    'orthogonal': np.array([
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
    ], dtype=np.int8),
    'diagonal_2d': np.array([
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
    ], dtype=np.int8),
    'diagonal_3d': np.array([
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
    ], dtype=np.int8),
}

QUEEN_DIRS = np.vstack([SLIDER_DIRECTIONS['orthogonal'],
                        SLIDER_DIRECTIONS['diagonal_2d'],
                        SLIDER_DIRECTIONS['diagonal_3d']])
ROOK_DIRS   = SLIDER_DIRECTIONS['orthogonal']
BISHOP_DIRS = np.vstack([SLIDER_DIRECTIONS['diagonal_2d'],
                         SLIDER_DIRECTIONS['diagonal_3d']])

_PRECOMPUTED_DIR = os.path.join(os.path.dirname(__file__), "precomputed")

# ---------- helpers ----------
@lru_cache(maxsize=4096)
def _get_direction_set(piece_type: str) -> np.ndarray:
    if piece_type == 'queen':
        return QUEEN_DIRS
    elif piece_type == 'rook':
        return ROOK_DIRS
    elif piece_type == 'bishop':
        return BISHOP_DIRS
    elif piece_type in {'xy_queen', 'xz_queen', 'yz_queen'}:
        plane = piece_type.split('_')[0]
        if plane == 'xy':
            return np.array([(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                             (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)], dtype=np.int8)
        elif plane == 'xz':
            return np.array([(1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                             (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)], dtype=np.int8)
        else:  # yz
            return np.array([(0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
                             (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=np.int8)
    else:
        return QUEEN_DIRS

def load_precomputed_rays(piece_name: str):
    path = os.path.join(_PRECOMPUTED_DIR, f"{piece_name}_rays.npy")
    if not os.path.isfile(path):
        return None
    arr = np.load(path, allow_pickle=True)
    return arr if arr.shape[0] == 729 else None

# ---------- generator ----------
class OptimizedSliderMovementGenerator:
    def __init__(self):
        self._move_cache: dict[tuple, list[Move]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._precomputed_rays: dict[str, np.ndarray | None] = {}

    # ------------- public API -------------
    def generate_moves(
        self,
        piece_type: str,
        pos: tuple[int, int, int],
        color: int,
        max_distance: int = 8,
        *,
        cache_manager: "OptimizedCacheManager",
        directions: np.ndarray | None = None,
    ) -> list[Move]:
        """
        Generate all slider moves for a piece.
        Occupancy is read from cache_manager.occupancy.mask
        """
        # 1.  fast path – disk-backed rays
        rays = self._get_rays(piece_name=piece_type, pos=pos)
        if rays is not None:
            return self._ray_walk(rays, pos, color, cache_manager, directions)

        # 2.  fallback – vectorised kernel + in-memory cache
        occ = cache_manager.occupancy.mask
        cache_key = (piece_type, pos, occ.tobytes(), color, max_distance,
                     directions.tobytes() if directions is not None else b"")
        if cache_key in self._move_cache:
            self._cache_hits += 1
            return self._move_cache[cache_key]

        self._cache_misses += 1
        if directions is None:
            directions = _get_direction_set(piece_type)

        raw = generate_slider_moves_kernel(pos, directions, occ, color, max_distance)
        moves = [convert_legacy_move_args(from_coord=pos,
                                          to_coord=(nx, ny, nz),
                                          is_capture=is_cap)
                 for nx, ny, nz, is_cap in raw]

        # keep cache bounded
        if len(self._move_cache) > 10_000:
            self._move_cache.clear()
        self._move_cache[cache_key] = moves
        return moves

    # ------------- internal helpers -------------
    def _get_rays(self, piece_name: str, pos: tuple[int, int, int]):
        if piece_name not in self._precomputed_rays:
            self._precomputed_rays[piece_name] = load_precomputed_rays(piece_name)
        rays = self._precomputed_rays[piece_name]
        if rays is None:
            return None
        idx = coord_to_idx(pos)
        return rays[idx]          # shape: [num_dirs][variable_ray_len]

    def _ray_walk(self,
                  rays: np.ndarray,
                  pos: tuple[int, int, int],
                  color: int,
                  cache_manager: "OptimizedCacheManager",
                  directions: np.ndarray | None) -> list[Move]:
        """Walk pre-computed rays."""
        occ = cache_manager.occupancy.mask
        if directions is None:
            directions = _get_direction_set("queen")  # fallback – not used for indexing
        moves: list[Move] = []

        for dir_idx, ray in enumerate(rays):
            for sq in ray:
                x, y, z = sq
                if not (0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9):
                    break
                occ_val = occ[z, y, x]          # [z,y,x] ordering
                if occ_val == 0:                # empty
                    moves.append(Move(pos, (x, y, z), flags=0))
                elif occ_val != color:          # capture (enemy)
                    moves.append(Move(pos, (x, y, z), flags=MOVE_FLAGS['CAPTURE']))
                    break
                else:                           # own piece blocks
                    break
        return moves

    # ------------- stats -------------
    def get_cache_stats(self):
        total = self._cache_hits + self._cache_misses
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': self._cache_hits / max(1, total),
            'cache_size': len(self._move_cache),
        }

# ---------- global instance ----------
_global_slider_gen = OptimizedSliderMovementGenerator()

def get_slider_generator() -> OptimizedSliderMovementGenerator:
    return _global_slider_gen

# ---------- hot numba kernel ----------
@njit(cache=True, fastmath=True, parallel=True)
def generate_slider_moves_kernel(
    pos: tuple[int, int, int],
    directions: np.ndarray,
    occupancy: np.ndarray,
    color: int,
    max_distance: int = 8,
) -> list[tuple[int, int, int, bool]]:
    px, py, pz = pos
    n_dirs = directions.shape[0]
    move_coords = np.empty((n_dirs, max_distance, 3), dtype=np.int32)
    is_capture = np.zeros((n_dirs, max_distance), dtype=np.bool_)
    move_counts = np.zeros(n_dirs, dtype=np.int32)

    for d_idx in prange(n_dirs):
        dx, dy, dz = directions[d_idx]
        count = 0
        for step in range(1, max_distance + 1):
            nx = px + step * dx
            ny = py + step * dy
            nz = pz + step * dz
            if not (0 <= nx < 9 and 0 <= ny < 9 and 0 <= nz < 9):
                break
            occ = occupancy[nz, ny, nx]
            if occ == 0:                       # empty
                move_coords[d_idx, count] = (nx, ny, nz)
                is_capture[d_idx, count] = False
                count += 1
            elif occ != color:                 # enemy
                move_coords[d_idx, count] = (nx, ny, nz)
                is_capture[d_idx, count] = True
                count += 1
                break
            else:                              # own piece
                break
        move_counts[d_idx] = count

    # flatten to Python list
    moves: list[tuple[int, int, int, bool]] = []
    for d_idx in range(n_dirs):
        for i in range(move_counts[d_idx]):
            x, y, z = move_coords[d_idx, i]
            moves.append((x, y, z, is_capture[d_idx, i]))
    return moves

# ---------- util for incremental update ----------
def dirty_squares_slider(
    mv: Move,
    mover: Color,
    cache_manager: "OptimizedCacheManager",
) -> set[tuple[int, int, int]]:
    """
    Return every square on the 26 rays that pass through from- or to-square.
    Used by move-generation caches to know which pieces may have discovered slides.
    """
    dirty: set[tuple[int, int, int]] = set()
    for centre in (mv.from_coord, mv.to_coord):
        for dx, dy, dz in product((-1, 0, 1), repeat=3):
            if dx == dy == dz == 0:
                continue
            for step in range(1, 9):
                c = (centre[0] + step * dx,
                     centre[1] + step * dy,
                     centre[2] + step * dz)
                if not in_bounds(c):
                    break
                dirty.add(c)
    return dirty
