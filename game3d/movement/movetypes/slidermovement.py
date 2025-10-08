# game3d/movement/movetypes/slidermovement.py
"""Optimized slider movement generation for 3D chess
Now supports disk-backed precomputed move rays, fallback to original vectorized kernel."""

import os
import numpy as np
from numba import njit, prange
from typing import List, Tuple, Set, Optional
from functools import lru_cache
from itertools import product

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import MOVE_FLAGS
from game3d.movement.movepiece import Move
from game3d.common.common import coord_to_idx
from game3d.cache.manager import OptimizedCacheManager
# Precompute all slider directions at module level
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
ROOK_DIRS = SLIDER_DIRECTIONS['orthogonal']
BISHOP_DIRS = np.vstack([SLIDER_DIRECTIONS['diagonal_2d'],
                         SLIDER_DIRECTIONS['diagonal_3d']])

_PRECOMPUTED_DIR = os.path.join(os.path.dirname(__file__), "precomputed")

def load_precomputed_rays(piece_name: str):
    """Load precomputed move rays for a piece type from disk, or return None if unavailable."""
    path = os.path.join(_PRECOMPUTED_DIR, f"{piece_name}_rays.npy")
    if not os.path.isfile(path):
        return None
    arr = np.load(path, allow_pickle=True)
    if arr.shape[0] != 729:
        return None
    return arr

@lru_cache(maxsize=4096)
def _get_direction_set(piece_type: str) -> np.ndarray:
    if piece_type == 'queen':
        return QUEEN_DIRS
    elif piece_type == 'rook':
        return ROOK_DIRS
    elif piece_type == 'bishop':
        return BISHOP_DIRS
    elif piece_type in ['xz_queen', 'xy_queen', 'yz_queen']:
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

class OptimizedSliderMovementGenerator:
    """High-performance slider movement generator with disk-backed rays."""

    def __init__(self):
        self._move_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._precomputed_rays = {}  # piece_name: rays array

    def _get_rays(self, piece_type, pos):
        if piece_type not in self._precomputed_rays:
            rays = load_precomputed_rays(piece_type)
            self._precomputed_rays[piece_type] = rays
        rays = self._precomputed_rays[piece_type]
        if rays is None:
            return None
        idx = coord_to_idx(pos)
        return rays[idx]  # shape: [num_dirs][variable ray length]

    def generate_moves(
        self,
        piece_type: str,
        pos: Tuple[int, int, int],
        board_occupancy: np.ndarray,
        color: int,
        max_distance: int = 8
    ) -> List['Move']:
        rays = self._get_rays(piece_type, pos)
        if rays is not None:
            # Use rays: for each direction, walk ray until blocked
            occ = board_occupancy
            moves = []
            directions = _get_direction_set(piece_type)
            for dir_idx, ray in enumerate(rays):
                for sq in ray:
                    x, y, z = sq
                    if not (0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9):
                        break
                    occ_val = occ[z, y, x]  # occupancy indexed [z,y,x]
                    if occ_val == 0:  # empty
                        moves.append(Move(pos, (x, y, z), flags=0))
                    elif occ_val != color:  # enemy
                        moves.append(Move(pos, (x, y, z), flags=MOVE_FLAGS['CAPTURE']))
                        break
                    else:  # friendly
                        break
            return moves

        # Fallback: use kernel
        cache_key = (piece_type, pos, board_occupancy.tobytes(), color)
        if cache_key in self._move_cache:
            self._cache_hits += 1
            return self._move_cache[cache_key]
        self._cache_misses += 1
        directions = _get_direction_set(piece_type)
        raw_moves = generate_slider_moves_kernel(
            pos, directions, board_occupancy, color, max_distance
        )
        moves = []
        for nx, ny, nz, is_capture in raw_moves:
            move = Move(pos, (nx, ny, nz), flags=MOVE_FLAGS['CAPTURE'] if is_capture else 0)
            moves.append(move)
        if len(self._move_cache) > 10000:
            self._move_cache.clear()
        self._move_cache[cache_key] = moves
        return moves

    def get_cache_stats(self):
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(1, total)
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._move_cache)
        }

_global_slider_gen = None

def get_slider_generator():
    global _global_slider_gen
    if _global_slider_gen is None:
        _global_slider_gen = OptimizedSliderMovementGenerator()
    return _global_slider_gen

@njit(cache=True, fastmath=True, parallel=True)
def generate_slider_moves_kernel(
    pos: Tuple[int, int, int],
    directions: np.ndarray,
    occupancy: np.ndarray,
    color: int,
    max_distance: int = 8
) -> List[Tuple[int, int, int, bool]]:
    px, py, pz = pos
    n_dirs = directions.shape[0]
    move_coords = np.empty((n_dirs, max_distance, 3), dtype=np.int32)
    is_capture_flags = np.zeros((n_dirs, max_distance), dtype=np.bool_)
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
            if occ == 0:
                move_coords[d_idx, count] = (nx, ny, nz)
                is_capture_flags[d_idx, count] = False
                count += 1
            elif occ != color:
                move_coords[d_idx, count] = (nx, ny, nz)
                is_capture_flags[d_idx, count] = True
                count += 1
                break
            else:
                break
        move_counts[d_idx] = count
    total_moves = np.sum(move_counts)
    moves = []
    for d_idx in range(n_dirs):
        for i in range(move_counts[d_idx]):
            x, y, z = move_coords[d_idx, i]
            is_capture = is_capture_flags[d_idx, i]
            moves.append((x, y, z, is_capture))
    return moves

def dirty_squares_slider(
    mv: Move,
    mover: Color,
    cache_manager: OptimizedCacheManager
) -> set[Tuple[int,int,int]]:
    """
    Sliding pieces can be *discovered* when the move opens/closes a ray.
    We return every square on the 26 rays that pass through from- or to-square.
    """
    dirty: set[Tuple[int,int,int]] = set()

    for centre in (mv.from_coord, mv.to_coord):
        for dx,dy,dz in product((-1,0,1), repeat=3):
            if dx==dy==dz==0:
                continue
            for step in range(1,9):
                c = (centre[0]+step*dx, centre[1]+step*dy, centre[2]+step*dz)
                if not in_bounds(*c):
                    break
                dirty.add(c)
    # 3-ring around move is enough – sliders rarely exceed 8 steps
    return dirty
