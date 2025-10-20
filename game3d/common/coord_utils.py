# game3d/common/coord_utils.py
# ------------------------------------------------------------------
# Coordinate utilities – optimised drop-in replacements
# ------------------------------------------------------------------
from __future__ import annotations
import torch
import math
from typing import Tuple, List, Optional, Union, Set
import numpy as np
from numba import njit

from game3d.common.constants import SIZE, _COORD_TO_IDX, _IDX_TO_COORD, RADIUS_2_OFFSETS

Coord = Tuple[int, int, int]

# ------------------------------------------------------------------
# Fast bounds check – branch-free, no Python-level tuple unpacking
# ------------------------------------------------------------------
_UPPER = SIZE - 1  # 8

@njit
def in_bounds(c: Coord) -> bool:
    return (0 <= c[0] < SIZE) and (0 <= c[1] < SIZE) and (0 <= c[2] < SIZE)

def in_bounds_vectorised(coords: np.ndarray) -> np.ndarray:
    """
    Return a 1-D bool array: True for every row whose x,y,z are all in 0-8.
    coords: int array shape (N, 3)  or  (3,)
    """
    coords = np.atleast_2d(coords)          # (3,)  →  (1,3)
    return np.all((coords >= 0) & (coords < SIZE), axis=1)

@njit(cache=True)
def in_bounds_scalar(x: int, y: int, z: int) -> bool:
    return 0 <= x < SIZE and 0 <= y < SIZE and 0 <= z < SIZE

def filter_valid_coords(coords: np.ndarray, log_oob: bool = True, clamp: bool = False) -> np.ndarray:
    # 1. optionally coerce edge values
    if clamp:
        coords = np.clip(coords, 0, SIZE - 1)

    # 2. build mask (True ⇒ keep)
    valid_mask = np.all((coords >= 0) & (coords < SIZE), axis=1)

    # 3. logging
    if log_oob and not np.all(valid_mask):
        bad = coords[~valid_mask]
        # print(f"[WARNING] Filtered {len(bad)} OOB coords. Sample: {bad[:3]}")

    # 4. return only valid rows
    return coords[valid_mask]

#  NEW – real functions that the rest of the code can call
def coord_to_idx(coord: Coord) -> int:
    """9×9×9 linear index:  x + 9*y + 81*z   (0 … 728)"""
    # we reuse the pre-built dict for speed
    return _COORD_TO_IDX[coord]

def idx_to_coord(idx: int) -> Coord:
    """Inverse of the above."""
    return _IDX_TO_COORD[idx]

def clip_coords(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clamp coordinates to [0, 8] to prevent index errors."""
    return np.clip(x, 0, 8), np.clip(y, 0, 8), np.clip(z, 0, 8)

@njit(cache=True)
def add_coords(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Component-wise addition of two 3-D coordinates."""
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]

@njit(cache=True)
def subtract_coords(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Component-wise subtraction of two 3-D coordinates."""
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]

def scale_coord(coord: Tuple[int, int, int], scalar: int) -> Tuple[int, int, int]:
    return (coord[0] * scalar, coord[1] * scalar, coord[2] * scalar)

# ------------------------------------------------------------------
# Distance helpers – kept as one-liners, already fast
# ------------------------------------------------------------------
manhattan_distance = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
chebyshev_distance = lambda a, b: max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))
euclidean_distance_squared = lambda a, b: (a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2

# Add the missing euclidean_distance function
def euclidean_distance(a: Coord, b: Coord) -> float:
    """Calculate the Euclidean distance between two coordinates."""
    return math.sqrt(euclidean_distance_squared(a, b))

# Add the missing manhattan function (alias for manhattan_distance)
def manhattan(a: Coord, b: Coord) -> int:
    """Calculate the Manhattan distance between two coordinates."""
    return manhattan_distance(a, b)

# ------------------------------------------------------------------
# Ray generation (keep — useful for some pieces)
# ------------------------------------------------------------------
def generate_ray(
    start: Coord,
    direction: Tuple[int, int, int],
    max_steps: Optional[int] = None
) -> List[Coord]:
    ray = []
    current = start
    steps = 0
    while True:
        current = add_coords(current, direction)
        if not in_bounds(current):
            break
        if max_steps is not None and steps >= max_steps:
            break
        ray.append(current)
        steps += 1
    return ray

# ------------------------------------------------------------------
# Path utilities (fixed)
# ------------------------------------------------------------------
def reconstruct_path(start: Coord, end: Coord, include_start: bool = False, include_end: bool = True, as_set: bool = False) -> Union[List[Coord], Set[Coord]]:
    """Reconstruct path from start to end, optionally as set, including/excluding endpoints."""
    if start == end:
        return set() if as_set else []
    fx, fy, fz = start
    tx, ty, tz = end
    dx = tx - fx
    dy = ty - fy
    dz = tz - fz
    max_delta = max(abs(dx), abs(dy), abs(dz))
    if max_delta == 0:
        return [] if not as_set else set()
    step_x = dx // max_delta if dx != 0 else 0
    step_y = dy // max_delta if dy != 0 else 0
    step_z = dz // max_delta if dz != 0 else 0
    path = set() if as_set else []
    x, y, z = fx, fy, fz
    if include_start:
        if as_set:
            path.add((x, y, z))
        else:
            path.append((x, y, z))
    for _ in range(max_delta):
        x += step_x
        y += step_y
        z += step_z
        if _ < max_delta - 1 or include_end:
            if as_set:
                path.add((x, y, z))
            else:
                path.append((x, y, z))
    return path

# NEW: Batch version for multiple ends
def reconstruct_paths_batch(start: Coord, ends: np.ndarray, include_start: bool = False, include_end: bool = True, as_sets: bool = False) -> List[Union[List[Coord], Set[Coord]]]:
    paths = []
    for end in ends:
        paths.append(reconstruct_path(start, tuple(end), include_start, include_end, as_sets))
    return paths

# Add the missing get_path_squares function
def get_path_squares(start: Coord, end: Coord) -> List[Coord]:
    """Get all squares on the path from start to end, including both endpoints."""
    if start == end:
        return [start]

    diff = subtract_coords(end, start)
    non_zero = [d for d in diff if d != 0]
    if not non_zero:
        return [start]

    # Check if straight line: orthogonal or diagonal
    abs_vals = [abs(d) for d in non_zero]
    is_orthogonal = len(non_zero) == 1
    is_diagonal = len(set(abs_vals)) == 1

    if not (is_orthogonal or is_diagonal):
        # Not a straight line, just return start and end
        return [start, end]

    step = tuple(0 if d == 0 else (1 if d > 0 else -1) for d in diff)
    distance = max(abs_vals)

    path = [start]
    for i in range(1, distance + 1):
        sq = add_coords(start, scale_coord(step, i))
        if in_bounds(sq):
            path.append(sq)
        else:
            break

    return path

def get_between_squares(start: Coord, end: Coord) -> List[Coord]:
    """Get squares strictly between start and end on a straight line."""
    if start == end:
        return []

    diff = subtract_coords(end, start)
    non_zero = [d for d in diff if d != 0]
    if not non_zero:
        return []

    # Check if straight line: orthogonal or diagonal
    abs_vals = [abs(d) for d in non_zero]
    is_orthogonal = len(non_zero) == 1
    is_diagonal = len(set(abs_vals)) == 1

    if not (is_orthogonal or is_diagonal):
        return []

    step = tuple(0 if d == 0 else (1 if d > 0 else -1) for d in diff)
    distance = max(abs_vals)

    between = []
    for i in range(1, distance):
        sq = add_coords(start, scale_coord(step, i))
        if in_bounds(sq):
            between.append(sq)
        else:
            break
    return between

# ------------------------------------------------------------------
# Geometric utilities (fixed space_diag)
# ------------------------------------------------------------------
def get_layer_coords(layer_type: str, index: int) -> List[Coord]:
    coords = []
    if layer_type == 'x':
        coords = [(index, y, z) for y in range(SIZE) for z in range(SIZE)]
    elif layer_type == 'y':
        coords = [(x, index, z) for x in range(SIZE) for z in range(SIZE)]
    elif layer_type == 'z':
        coords = [(x, y, index) for x in range(SIZE) for y in range(SIZE)]
    elif layer_type == 'xy_diag':
        coords = [(i, i, index) for i in range(SIZE)]
        coords += [(SIZE-1-i, i, index) for i in range(SIZE) if i != SIZE-1-i]
    elif layer_type == 'xz_diag':
        coords = [(i, index, i) for i in range(SIZE)]
        coords += [(SIZE-1-i, index, i) for i in range(SIZE) if i != SIZE-1-i]
    elif layer_type == 'yz_diag':
        coords = [(index, i, i) for i in range(SIZE)]
        coords += [(index, SIZE-1-i, i) for i in range(SIZE) if i != SIZE-1-i]
    elif layer_type == 'space_diag':
        # All 4 main space diagonals (ignore index)
        coords = [(i, i, i) for i in range(SIZE)]
        coords += [(i, i, SIZE-1-i) for i in range(SIZE)]
        coords += [(i, SIZE-1-i, i) for i in range(SIZE)]
        coords += [(SIZE-1-i, i, i) for i in range(SIZE)]
    return [c for c in coords if in_bounds(c)]

def get_distance_layers(start: Coord, max_distance: int) -> List[List[Coord]]:
    layers = [[] for _ in range(max_distance + 1)]
    for x in range(SIZE):
        for y in range(SIZE):
            for z in range(SIZE):
                dist = chebyshev_distance(start, (x, y, z))
                if dist <= max_distance:
                    layers[dist].append((x, y, z))
    return layers

def get_aura_squares(center: Tuple[int, int, int], radius: int = 2) -> Set[Tuple[int, int, int]]:
    """Compute aura squares around center using precomputed offsets, with bounds check."""
    if radius != 2:
        raise ValueError("Only radius=2 supported for precompute")
    cx, cy, cz = center
    aura = set()
    for dx, dy, dz in RADIUS_2_OFFSETS:
        sq = (cx + dx, cy + dy, cz + dz)
        if in_bounds(sq):  # Use existing njit in_bounds
            aura.add(sq)
    return aura

def validate_directions(start: Coord, directions: np.ndarray, board_size: int = 9) -> np.ndarray:
    """Filter directions that would lead to out-of-bounds coordinates."""
    start_arr = np.array(start, dtype=np.int16)
    dest_coords = start_arr + directions
    valid_mask = np.all((dest_coords >= 0) & (dest_coords < board_size), axis=1)
    return directions[valid_mask]

def _clip_coords(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clamp coordinates to [0, 8] to prevent index errors."""
    return np.clip(x, 0, 8), np.clip(y, 0, 8), np.clip(z, 0, 8)
