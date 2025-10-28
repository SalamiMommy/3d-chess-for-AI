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

from game3d.common.constants import SIZE, _COORD_TO_IDX, _IDX_TO_COORD, RADIUS_2_OFFSETS, SIZE_SQUARED, SIZE_MINUS_1

Coord = Tuple[int, int, int]

# ------------------------------------------------------------------
# Fast bounds check – split for numba compatibility
# ------------------------------------------------------------------
_UPPER = SIZE_MINUS_1  # 8

@njit(cache=True, inline='always')
def in_bounds_scalar(x: int, y: int, z: int) -> bool:
    """Ultra-fast bounds check with early exits."""
    if x < 0 or x >= SIZE:
        return False
    if y < 0 or y >= SIZE:
        return False
    return 0 <= z < SIZE

@njit(cache=True)
def in_bounds_numpy_batch(coords: np.ndarray) -> np.ndarray:
    """Batch bounds checking for numpy arrays - numba compatible."""
    if coords.ndim == 1:
        return np.all((coords >= 0) & (coords < SIZE))

    # Manual implementation for 2D arrays since numba's np.all doesn't support axis parameter
    n = coords.shape[0]
    result = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        result[i] = np.all((coords[i] >= 0) & (coords[i] < SIZE))
    return result

def in_bounds_torch_batch(coords: torch.Tensor) -> torch.Tensor:
    """Batch bounds checking for torch tensors."""
    if coords.ndim == 1:
        return torch.all((coords >= 0) & (coords < SIZE))
    return torch.all((coords >= 0) & (coords < SIZE), dim=1)

def in_bounds(c: Union[Coord, np.ndarray, torch.Tensor]) -> Union[bool, np.ndarray, torch.Tensor]:
    """Bounds checking - supports scalar and batch mode."""
    if isinstance(c, torch.Tensor):
        return in_bounds_torch_batch(c)
    elif isinstance(c, np.ndarray):
        return in_bounds_numpy_batch(c)
    else:
        # Scalar mode - tuple or list
        return in_bounds_scalar(c[0], c[1], c[2])

def in_bounds_vectorised(coords: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Return a 1-D bool array: True for every row whose x,y,z are all in 0-8.
    coords: int array shape (N, 3)  or  (3,) - supports scalar and batch mode.
    """
    if isinstance(coords, torch.Tensor):
        return in_bounds_torch_batch(coords)
    else:
        return in_bounds_numpy_batch(coords)

def filter_valid_coords(coords: np.ndarray) -> np.ndarray:
    """Optimized coordinate filtering."""
    if coords.size == 0:
        return coords

    # Single bounds check - all operations vectorized
    valid = ((coords >= 0) & (coords < SIZE)).all(axis=1)

    # Early return if all valid
    if valid.all():
        return coords

    return coords[valid]

# NEW – real functions that the rest of the code can call
def coord_to_idx(coord: Union[Coord, torch.Tensor]) -> Union[int, torch.Tensor]:
    """9×9×9 linear index:  x + 9*y + 81*z   (0 … 728) - supports scalar and batch mode."""
    if isinstance(coord, torch.Tensor) and coord.ndim > 1:
        # Batch mode: [N, 3] -> [N]
        x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
        return x + SIZE * y + SIZE_SQUARED * z
    else:
        # Scalar mode
        if isinstance(coord, torch.Tensor):
            x, y, z = coord.tolist()
        else:
            x, y, z = coord
        return x + SIZE * y + SIZE_SQUARED * z

def idx_to_coord(idx: Union[int, torch.Tensor]) -> Union[Coord, torch.Tensor]:
    """Inverse of the above - supports scalar and batch mode."""
    if isinstance(idx, torch.Tensor) and idx.ndim > 0:
        # Batch mode: [N] -> [N, 3]
        z = idx // (SIZE_SQUARED)
        rem = idx % (SIZE_SQUARED)
        y = rem // SIZE
        x = rem % SIZE
        return torch.stack([x, y, z], dim=1)
    else:
        # Scalar mode
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        z, rem = divmod(idx, SIZE_SQUARED)
        y, x = divmod(rem, SIZE)
        return (x, y, z)

def clip_coords(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], z: Union[np.ndarray, torch.Tensor]) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Clamp coordinates to [0, 8] to prevent index errors - supports scalar and batch mode."""
    if isinstance(x, torch.Tensor):
        return torch.clamp(x, 0, 8), torch.clamp(y, 0, 8), torch.clamp(z, 0, 8)
    else:
        return np.clip(x, 0, 8), np.clip(y, 0, 8), np.clip(z, 0, 8)

@njit(cache=True)
def add_coords(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Component-wise addition of two 3-D coordinates."""
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]

def add_coords_batch(a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Batch coordinate addition - supports scalar and batch mode."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a + b
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a + b
    else:
        # Mixed types - convert to common type
        if isinstance(a, torch.Tensor):
            b = torch.tensor(b, dtype=a.dtype, device=a.device)
            return a + b
        else:
            a = np.array(a)
            b = np.array(b)
            return a + b

@njit(cache=True)
def subtract_coords(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Component-wise subtraction of two 3-D coordinates."""
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]

def subtract_coords_batch(a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Batch coordinate subtraction - supports scalar and batch mode."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a - b
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a - b
    else:
        # Mixed types
        if isinstance(a, torch.Tensor):
            b = torch.tensor(b, dtype=a.dtype, device=a.device)
            return a - b
        else:
            a = np.array(a)
            b = np.array(b)
            return a - b

@njit(cache=True)
def scale_coord(coord: Tuple[int, int, int], scalar: int) -> Tuple[int, int, int]:
    return (coord[0] * scalar, coord[1] * scalar, coord[2] * scalar)

def scale_coord_batch(coord: Union[np.ndarray, torch.Tensor], scalar: Union[int, np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Batch coordinate scaling - supports scalar and batch mode."""
    if isinstance(coord, torch.Tensor):
        if isinstance(scalar, torch.Tensor):
            return coord * scalar.unsqueeze(-1)
        else:
            return coord * scalar
    else:
        if isinstance(scalar, np.ndarray):
            return coord * scalar[:, np.newaxis]
        else:
            return coord * scalar

# ------------------------------------------------------------------
# Distance helpers – optimized with numba
# ------------------------------------------------------------------
@njit(cache=True)
def manhattan_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

@njit(cache=True)
def manhattan_distance_numpy_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batch Manhattan distance for numpy arrays - numba compatible."""
    if a.ndim == 1 and b.ndim == 1:
        return np.sum(np.abs(a - b))
    else:
        # Ensure 2D for consistent broadcasting
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        return np.sum(np.abs(a - b), axis=1).astype(np.int32)

def manhattan_distance_batch(
    a: Union[np.ndarray, torch.Tensor],
    b: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """Batch Manhattan distance - supports scalar and batch mode."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.ndim == 1 and b.ndim == 1:
            return torch.sum(torch.abs(a - b))
        else:
            return torch.sum(torch.abs(a - b), dim=1)
    else:
        return manhattan_distance_numpy_batch(np.asarray(a), np.asarray(b))

@njit(cache=True)
def chebyshev_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))

@njit(cache=True)
def chebyshev_distance_numpy_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batch Chebyshev distance for numpy arrays - numba compatible."""
    if a.ndim == 1 and b.ndim == 1:
        return np.max(np.abs(a - b))
    else:
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        return np.max(np.abs(a - b), axis=1)

def chebyshev_distance_batch(a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Batch Chebyshev distance - supports scalar and batch mode."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.ndim == 1 and b.ndim == 1:
            return torch.max(torch.abs(a - b))
        else:
            return torch.max(torch.abs(a - b), dim=1)[0]
    else:
        return chebyshev_distance_numpy_batch(np.asarray(a), np.asarray(b))

@njit(cache=True)
def euclidean_distance_squared(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
    return (a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2

@njit(cache=True)
def euclidean_distance_squared_numpy_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batch Euclidean distance squared for numpy arrays - numba compatible."""
    if a.ndim == 1 and b.ndim == 1:
        return np.sum((a - b)**2)
    else:
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        return np.sum((a - b)**2, axis=1)

def euclidean_distance_squared_batch(a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Batch Euclidean distance squared - supports scalar and batch mode."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.ndim == 1 and b.ndim == 1:
            return torch.sum((a - b)**2)
        else:
            return torch.sum((a - b)**2, dim=1)
    else:
        return euclidean_distance_squared_numpy_batch(np.asarray(a), np.asarray(b))

@njit(cache=True)
def euclidean_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    """Calculate the Euclidean distance between two coordinates."""
    return math.sqrt(euclidean_distance_squared(a, b))

def euclidean_distance_batch(a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Batch Euclidean distance - supports scalar and batch mode."""
    squared = euclidean_distance_squared_batch(a, b)
    if isinstance(squared, torch.Tensor):
        return torch.sqrt(squared.float())
    else:
        return np.sqrt(squared.astype(float))

# Aliases for backward compatibility
manhattan = manhattan_distance

# ------------------------------------------------------------------
# Ray generation (optimized)
# ------------------------------------------------------------------
def generate_ray(
    start: Union[Coord, torch.Tensor],
    direction: Union[Tuple[int, int, int], torch.Tensor],
    max_steps: Optional[int] = None
) -> Union[List[Coord], List[List[Coord]]]:
    """Generate ray - supports scalar and batch mode."""
    if isinstance(start, torch.Tensor) and start.ndim > 1:
        # Batch mode
        batch_size = start.shape[0]
        if isinstance(direction, torch.Tensor) and direction.ndim > 1:
            directions = direction
        else:
            directions = direction.unsqueeze(0).repeat(batch_size, 1)

        rays = []
        for i in range(batch_size):
            single_start = tuple(start[i].tolist())
            single_dir = tuple(directions[i].tolist())
            rays.append(generate_ray(single_start, single_dir, max_steps))
        return rays

    # Scalar mode
    if isinstance(start, torch.Tensor):
        start = tuple(start.tolist())
    if isinstance(direction, torch.Tensor):
        direction = tuple(direction.tolist())

    # Pre-allocate list with estimated capacity
    if max_steps is None:
        max_steps = SIZE * 3  # Conservative estimate

    ray = []
    current = list(start)  # Convert to list for faster updates
    dx, dy, dz = direction

    for steps in range(1, max_steps + 1):
        current[0] += dx
        current[1] += dy
        current[2] += dz

        if not in_bounds_scalar(current[0], current[1], current[2]):
            break

        ray.append(tuple(current))

    return ray

# ------------------------------------------------------------------
# Path utilities (optimized)
# ------------------------------------------------------------------
def reconstruct_path(start: Union[Coord, torch.Tensor], end: Union[Coord, torch.Tensor], include_start: bool = False, include_end: bool = True, as_set: bool = False) -> Union[List[Coord], Set[Coord], List[List[Coord]], List[Set[Coord]]]:
    """Reconstruct path from start to end - supports scalar and batch mode."""
    if isinstance(start, torch.Tensor) and start.ndim > 1:
        # Batch mode
        batch_size = start.shape[0]
        if isinstance(end, torch.Tensor) and end.ndim > 1:
            ends = end
        else:
            ends = end.unsqueeze(0).repeat(batch_size, 1)

        paths = []
        for i in range(batch_size):
            single_start = tuple(start[i].tolist())
            single_end = tuple(ends[i].tolist())
            paths.append(reconstruct_path(single_start, single_end, include_start, include_end, as_set))
        return paths

    # Scalar mode
    if isinstance(start, torch.Tensor):
        start = tuple(start.tolist())
    if isinstance(end, torch.Tensor):
        end = tuple(end.tolist())

    if start == end:
        return set() if as_set else []

    fx, fy, fz = start
    tx, ty, tz = end
    dx, dy, dz = tx - fx, ty - fy, tz - fz

    max_delta = max(abs(dx), abs(dy), abs(dz))
    if max_delta == 0:
        return set() if as_set else []

    step_x = dx // max_delta
    step_y = dy // max_delta
    step_z = dz // max_delta

    if as_set:
        path = set()
        if include_start:
            path.add(start)
        x, y, z = fx, fy, fz
        for i in range(max_delta):
            x += step_x
            y += step_y
            z += step_z
            if i < max_delta - 1 or include_end:
                path.add((x, y, z))
        return path
    else:
        # List version - pre-allocate
        path_size = max_delta
        if include_start:
            path_size += 1
        if not include_end:
            path_size -= 1

        path = []
        if include_start:
            path.append(start)

        x, y, z = fx, fy, fz
        for i in range(max_delta):
            x += step_x
            y += step_y
            z += step_z
            if i < max_delta - 1 or include_end:
                path.append((x, y, z))
        return path

# Optimized batch version
def reconstruct_paths_batch(start: Union[Coord, torch.Tensor], ends: Union[np.ndarray, torch.Tensor], include_start: bool = False, include_end: bool = True, as_sets: bool = False) -> List[Union[List[Coord], Set[Coord]]]:
    """Batch path reconstruction - supports scalar and batch mode."""
    if isinstance(ends, torch.Tensor):
        ends_list = [tuple(end.tolist()) for end in ends]
    else:
        ends_list = [tuple(end) for end in ends]

    if isinstance(start, torch.Tensor):
        start = tuple(start.tolist())

    return [reconstruct_path(start, end, include_start, include_end, as_sets) for end in ends_list]

# Optimized path functions
def get_path_squares(start: Union[Coord, torch.Tensor], end: Union[Coord, torch.Tensor]) -> Union[List[Coord], List[List[Coord]]]:
    """Get all squares on the path from start to end - supports scalar and batch mode."""
    if isinstance(start, torch.Tensor) and start.ndim > 1:
        # Batch mode
        batch_size = start.shape[0]
        if isinstance(end, torch.Tensor) and end.ndim > 1:
            ends = end
        else:
            ends = end.unsqueeze(0).repeat(batch_size, 1)

        paths = []
        for i in range(batch_size):
            single_start = tuple(start[i].tolist())
            single_end = tuple(ends[i].tolist())
            paths.append(get_path_squares(single_start, single_end))
        return paths

    # Scalar mode
    if isinstance(start, torch.Tensor):
        start = tuple(start.tolist())
    if isinstance(end, torch.Tensor):
        end = tuple(end.tolist())

    if start == end:
        return [start]

    dx, dy, dz = subtract_coords(end, start)
    abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)

    # Check if straight line
    non_zero = [d for d in (abs_dx, abs_dy, abs_dz) if d != 0]
    if not non_zero:
        return [start]

    is_orthogonal = len(non_zero) == 1
    is_diagonal = len(non_zero) == len(set(non_zero)) and len(set(non_zero)) <= 1

    if not (is_orthogonal or is_diagonal):
        return [start, end]

    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    step_z = 0 if dz == 0 else (1 if dz > 0 else -1)

    distance = max(abs_dx, abs_dy, abs_dz)
    path = [start]

    # Pre-allocate path list
    for i in range(1, distance + 1):
        sq = (start[0] + i * step_x, start[1] + i * step_y, start[2] + i * step_z)
        if in_bounds_scalar(sq[0], sq[1], sq[2]):
            path.append(sq)
        else:
            break

    return path

def get_between_squares(start: Union[Coord, torch.Tensor], end: Union[Coord, torch.Tensor]) -> Union[List[Coord], List[List[Coord]]]:
    """Get squares strictly between start and end - supports scalar and batch mode."""
    if isinstance(start, torch.Tensor) and start.ndim > 1:
        # Batch mode
        batch_size = start.shape[0]
        if isinstance(end, torch.Tensor) and end.ndim > 1:
            ends = end
        else:
            ends = end.unsqueeze(0).repeat(batch_size, 1)

        between_list = []
        for i in range(batch_size):
            single_start = tuple(start[i].tolist())
            single_end = tuple(ends[i].tolist())
            between_list.append(get_between_squares(single_start, single_end))
        return between_list

    # Scalar mode
    if isinstance(start, torch.Tensor):
        start = tuple(start.tolist())
    if isinstance(end, torch.Tensor):
        end = tuple(end.tolist())

    if start == end:
        return []

    dx, dy, dz = subtract_coords(end, start)
    abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)

    non_zero = [d for d in (abs_dx, abs_dy, abs_dz) if d != 0]
    if not non_zero:
        return []

    # Check if straight line
    is_orthogonal = len(non_zero) == 1
    is_diagonal = len(non_zero) == len(set(non_zero)) and len(set(non_zero)) <= 1

    if not (is_orthogonal or is_diagonal):
        return []

    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    step_z = 0 if dz == 0 else (1 if dz > 0 else -1)

    distance = max(abs_dx, abs_dy, abs_dz)
    between = []

    # Only add intermediate squares
    for i in range(1, distance):
        sq = (start[0] + i * step_x, start[1] + i * step_y, start[2] + i * step_z)
        if in_bounds_scalar(sq[0], sq[1], sq[2]):
            between.append(sq)
        else:
            break

    return between

# ------------------------------------------------------------------
# Geometric utilities (optimized)
# ------------------------------------------------------------------
def get_layer_coords(layer_type: str, index: Union[int, torch.Tensor]) -> Union[List[Coord], List[List[Coord]]]:
    """Get layer coordinates - supports scalar and batch mode."""
    if isinstance(index, torch.Tensor) and index.ndim > 0:
        # Batch mode
        return [get_layer_coords(layer_type, idx.item()) for idx in index]

    # Scalar mode
    # Precompute ranges
    rng = range(SIZE)

    if layer_type == 'x':
        return [(index, y, z) for y in rng for z in rng]
    elif layer_type == 'y':
        return [(x, index, z) for x in rng for z in rng]
    elif layer_type == 'z':
        return [(x, y, index) for x in rng for y in rng]
    elif layer_type == 'xy_diag':
        coords = [(i, i, index) for i in rng]
        # Avoid duplicates in center
        coords += [(SIZE-1-i, i, index) for i in rng if i != SIZE-1-i]
        return coords
    elif layer_type == 'xz_diag':
        coords = [(i, index, i) for i in rng]
        coords += [(SIZE-1-i, index, i) for i in rng if i != SIZE-1-i]
        return coords
    elif layer_type == 'yz_diag':
        coords = [(index, i, i) for i in rng]
        coords += [(index, SIZE-1-i, i) for i in rng if i != SIZE-1-i]
        return coords
    elif layer_type == 'space_diag':
        # All 4 main space diagonals (ignore index)
        coords = [(i, i, i) for i in rng]
        coords += [(i, i, SIZE-1-i) for i in rng]
        coords += [(i, SIZE-1-i, i) for i in rng]
        coords += [(SIZE-1-i, i, i) for i in rng]
        return [c for c in coords if in_bounds(c)]
    return []

def get_distance_layers(start: Union[Coord, torch.Tensor], max_distance: Union[int, torch.Tensor]) -> Union[List[List[Coord]], List[List[List[Coord]]]]:
    """Get distance layers - supports scalar and batch mode."""
    if isinstance(start, torch.Tensor) and start.ndim > 1:
        # Batch mode
        batch_size = start.shape[0]
        if isinstance(max_distance, torch.Tensor) and max_distance.ndim > 0:
            max_distances = max_distance
        else:
            max_distances = torch.tensor([max_distance] * batch_size)

        layers_list = []
        for i in range(batch_size):
            single_start = tuple(start[i].tolist())
            single_max_dist = max_distances[i].item()
            layers_list.append(get_distance_layers(single_start, single_max_dist))
        return layers_list

    # Scalar mode
    if isinstance(start, torch.Tensor):
        start = tuple(start.tolist())
    if isinstance(max_distance, torch.Tensor):
        max_distance = max_distance.item()

    # Pre-allocate layers
    layers = [[] for _ in range(max_distance + 1)]
    sx, sy, sz = start

    # Use more efficient iteration
    for x in range(SIZE):
        dx = abs(x - sx)
        for y in range(SIZE):
            dy = abs(y - sy)
            for z in range(SIZE):
                dist = max(dx, abs(y - sy), abs(z - sz))
                if dist <= max_distance:
                    layers[dist].append((x, y, z))
    return layers

def get_aura_squares(center: Union[Tuple[int, int, int], torch.Tensor], radius: Union[int, torch.Tensor] = 2) -> Union[Set[Tuple[int, int, int]], List[Set[Tuple[int, int, int]]]]:
    """Compute aura squares around center - supports scalar and batch mode."""
    if isinstance(center, torch.Tensor) and center.ndim > 1:
        # Batch mode
        batch_size = center.shape[0]
        if isinstance(radius, torch.Tensor) and radius.ndim > 0:
            radii = radius
        else:
            radii = torch.tensor([radius] * batch_size)

        auras = []
        for i in range(batch_size):
            single_center = tuple(center[i].tolist())
            single_radius = radii[i].item()
            auras.append(get_aura_squares(single_center, single_radius))
        return auras

    # Scalar mode
    if isinstance(center, torch.Tensor):
        center = tuple(center.tolist())
    if isinstance(radius, torch.Tensor):
        radius = radius.item()

    if radius != 2:
        raise ValueError("Only radius=2 supported for precompute")

    cx, cy, cz = center
    aura = set()

    # Pre-compute bounds checks
    for dx, dy, dz in RADIUS_2_OFFSETS:
        x, y, z = cx + dx, cy + dy, cz + dz
        if in_bounds_scalar(x, y, z):
            aura.add((x, y, z))
    return aura

def validate_directions(start: Union[Coord, torch.Tensor], directions: Union[np.ndarray, torch.Tensor], board_size: int = 9) -> Union[np.ndarray, torch.Tensor]:
    """Filter directions that would lead to out-of-bounds coordinates - supports scalar and batch mode."""
    if directions.size == 0:
        return directions

    if isinstance(start, torch.Tensor) and isinstance(directions, torch.Tensor):
        start_arr = start
        if start_arr.ndim == 1:
            start_arr = start_arr.unsqueeze(0)
        if directions.ndim == 2:
            directions = directions.unsqueeze(0)

        dest_coords = start_arr + directions
        valid_mask = torch.all((dest_coords >= 0) & (dest_coords < board_size), dim=-1)
        return directions[valid_mask]
    else:
        start_arr = np.array(start, dtype=np.int16)
        dest_coords = start_arr + directions
        valid_mask = np.all((dest_coords >= 0) & (dest_coords < board_size), axis=1)
        return directions[valid_mask]

def validate_and_sanitize_coord(coord: Union[Coord, torch.Tensor]) -> Optional[Coord]:
    """Comprehensive coordinate validation and sanitization - scalar mode only."""
    if isinstance(coord, torch.Tensor):
        coord = tuple(coord.tolist())

    if not isinstance(coord, tuple) or len(coord) != 3:
        return None

    x, y, z = coord
    if not all(isinstance(c, int) for c in (x, y, z)):
        return None

    if not in_bounds_scalar(x, y, z):
        return None

    return (x, y, z)  # Already in bounds, no need to reclip

def batch_validate_coords(coords: Union[List[Coord], torch.Tensor]) -> Union[List[Coord], torch.Tensor]:
    """Batch validate multiple coordinates - supports scalar and batch mode."""
    if isinstance(coords, torch.Tensor):
        # Tensor batch validation
        valid_mask = in_bounds_vectorised(coords)
        return coords[valid_mask]
    else:
        # List batch validation
        SIZE_1 = SIZE_MINUS_1
        return [c for c in coords if isinstance(c, tuple) and len(c) == 3 and
                all(isinstance(coord, int) for coord in c) and
                0 <= c[0] <= SIZE_1 and 0 <= c[1] <= SIZE_1 and 0 <= c[2] <= SIZE_1]
