"""Optimized 3-D geometry helpers with enhanced performance and additional utilities."""

from __future__ import annotations
import math
from typing import Iterable, Tuple, List, Set, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import struct
import array

from .constants import SIZE_X, SIZE_Y, SIZE_Z, X, Y, Z

# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

Coord = Tuple[int, int, int]

class Direction3D(Enum):
    """Pre-defined 3D directions for optimized ray casting."""
    UP = (0, 0, 1)
    DOWN = (0, 0, -1)
    NORTH = (0, 1, 0)
    SOUTH = (0, -1, 0)
    EAST = (1, 0, 0)
    WEST = (-1, 0, 0)

    # Diagonal directions
    NORTHEAST = (1, 1, 0)
    NORTHWEST = (-1, 1, 0)
    SOUTHEAST = (1, -1, 0)
    SOUTHWEST = (-1, -1, 0)

    # 3D diagonals
    UP_NORTH = (0, 1, 1)
    UP_SOUTH = (0, -1, 1)
    UP_EAST = (1, 0, 1)
    UP_WEST = (-1, 0, 1)
    DOWN_NORTH = (0, 1, -1)
    DOWN_SOUTH = (0, -1, -1)
    DOWN_EAST = (1, 0, -1)
    DOWN_WEST = (-1, 0, -1)

@dataclass(slots=True, frozen=True)
class Sphere3D:
    """Represents a 3D sphere for efficient geometric calculations."""
    center: Coord
    radius: float
    radius_squared: float

    def __post_init__(self):
        object.__setattr__(self, 'radius_squared', self.radius * self.radius)

@dataclass(slots=True)
class Ray3D:
    """Optimized 3D ray with pre-calculated direction vectors."""
    origin: Coord
    direction: Coord
    direction_normalized: Coord
    max_steps: int

    def __post_init__(self):
        # Normalize direction vector
        length = math.sqrt(sum(d*d for d in self.direction))
        if length > 0:
            self.direction_normalized = tuple(d / length for d in self.direction)

# ==============================================================================
# BOUNDS AND CLAMPING
# ==============================================================================

def in_bounds(c: Coord) -> bool:
    """Fast bounds checking with early termination."""
    return (0 <= c[X] < SIZE_X and
            0 <= c[Y] < SIZE_Y and
            0 <= c[Z] < SIZE_Z)

def in_bounds_batch(coords: List[Coord]) -> List[bool]:
    """Batch bounds checking for multiple coordinates."""
    return [
        (0 <= c[X] < SIZE_X and 0 <= c[Y] < SIZE_Y and 0 <= c[Z] < SIZE_Z)
        for c in coords
    ]

def clamp(c: Coord) -> Coord:
    """Fast coordinate clamping with boundary checks."""
    return (max(0, min(c[X], SIZE_X - 1)),
            max(0, min(c[Y], SIZE_Y - 1)),
            max(0, min(c[Z], SIZE_Z - 1)))

def clamp_batch(coords: List[Coord]) -> List[Coord]:
    """Batch coordinate clamping."""
    return [
        (max(0, min(c[X], SIZE_X - 1)),
         max(0, min(c[Y], SIZE_Y - 1)),
         max(0, min(c[Z], SIZE_Z - 1)))
        for c in coords
    ]

# ==============================================================================
# VECTOR OPERATIONS
# ==============================================================================

def add(a: Coord, b: Coord) -> Coord:
    """Fast vector addition."""
    return (a[X] + b[X], a[Y] + b[Y], a[Z] + b[Z])

def sub(a: Coord, b: Coord) -> Coord:
    """Fast vector subtraction."""
    return (a[X] - b[X], a[Y] - b[Y], a[Z] - b[Z])

def scale(v: Coord, k: Union[int, float]) -> Coord:
    """Fast vector scaling."""
    return (int(v[X] * k), int(v[Y] * k), int(v[Z] * k))

def dot_product(a: Coord, b: Coord) -> int:
    """Vector dot product."""
    return a[X] * b[X] + a[Y] * b[Y] + a[Z] * b[Z]

def cross_product(a: Coord, b: Coord) -> Coord:
    """Vector cross product."""
    return (a[Y] * b[Z] - a[Z] * b[Y],
            a[Z] * b[X] - a[X] * b[Z],
            a[X] * b[Y] - a[Y] * b[X])

def magnitude(v: Coord) -> float:
    """Vector magnitude."""
    return math.sqrt(v[X]*v[X] + v[Y]*v[Y] + v[Z]*v[Z])

def normalize(v: Coord) -> Coord:
    """Normalize vector to unit length."""
    mag = magnitude(v)
    if mag == 0:
        return (0, 0, 0)
    return (v[X] / mag, v[Y] / mag, v[Z] / mag)

# ==============================================================================
# DISTANCE CALCULATIONS
# ==============================================================================

def manhattan(a: Coord, b: Coord) -> int:
    """Fast Manhattan distance."""
    return abs(a[X] - b[X]) + abs(a[Y] - b[Y]) + abs(a[Z] - b[Z])

def euclidean(a: Coord, b: Coord) -> float:
    """Euclidean distance between coordinates."""
    dx = a[X] - b[X]
    dy = a[Y] - b[Y]
    dz = a[Z] - b[Z]
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def euclidean_squared(a: Coord, b: Coord) -> int:
    """Squared Euclidean distance (avoids sqrt for comparisons)."""
    dx = a[X] - b[X]
    dy = a[Y] - b[Y]
    dz = a[Z] - b[Z]
    return dx*dx + dy*dy + dz*dz

def chebyshev(a: Coord, b: Coord) -> int:
    """Chebyshev distance (chess king moves)."""
    return max(abs(a[X] - b[X]), abs(a[Y] - b[Y]), abs(a[Z] - b[Z]))

# ==============================================================================
# ENHANCED RAY CASTING
# ==============================================================================

def ray_iter(start: Coord, step: Coord, max_steps: int = SIZE_X) -> Iterable[Coord]:
    """Optimized ray iteration with early termination."""
    for n in range(1, max_steps + 1):
        c = add(start, scale(step, n))
        if not in_bounds(c):
            break
        yield c

def ray_iter_optimized(ray: Ray3D) -> Iterable[Coord]:
    """Optimized ray iteration using Ray3D object."""
    for n in range(1, ray.max_steps + 1):
        c = add(ray.origin, scale(ray.direction, n))
        if not in_bounds(c):
            break
        yield c

def ray_all_directions(center: Coord, max_distance: int = 1) -> Dict[Direction3D, List[Coord]]:
    """Get rays in all directions from center."""
    rays = {}
    for direction in Direction3D:
        rays[direction] = list(ray_iter(center, direction.value, max_distance))
    return rays

# ==============================================================================
# SPHERE AND CIRCLE OPERATIONS
# ==============================================================================

def sphere_surface(center: Coord, radius: float, tolerance: float = 0.1) -> Set[Coord]:
    """Get coordinates on sphere surface with given radius."""
    surface = set()
    radius_squared = radius * radius
    radius_int = int(radius) + 1

    # Check cubic region around center
    for dx in range(-radius_int, radius_int + 1):
        for dy in range(-radius_int, radius_int + 1):
            for dz in range(-radius_int, radius_int + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue

                target = add(center, (dx, dy, dz))
                if not in_bounds(target):
                    continue

                distance_squared = dx*dx + dy*dy + dz*dz
                if abs(distance_squared - radius_squared) <= tolerance * radius:
                    surface.add(target)

    return surface

def sphere_volume(center: Coord, radius: float) -> Set[Coord]:
    """Get all coordinates within sphere volume."""
    volume = set()
    radius_squared = radius * radius
    radius_int = int(radius) + 1

    for dx in range(-radius_int, radius_int + 1):
        for dy in range(-radius_int, radius_int + 1):
            for dz in range(-radius_int, radius_int + 1):
                target = add(center, (dx, dy, dz))
                if not in_bounds(target):
                    continue

                if dx*dx + dy*dy + dz*dz <= radius_squared:
                    volume.add(target)

    return volume

def circle_perimeter(center: Coord, radius: float, plane: str = 'xy') -> Set[Coord]:
    """Get coordinates on circle perimeter in specified plane."""
    perimeter = set()
    radius_squared = radius * radius
    radius_int = int(radius) + 1

    for dx in range(-radius_int, radius_int + 1):
        for dy in range(-radius_int, radius_int + 1):
            if plane == 'xy':
                target = add(center, (dx, dy, 0))
                distance_squared = dx*dx + dy*dy
            elif plane == 'xz':
                target = add(center, (dx, 0, dy))
                distance_squared = dx*dx + dy*dy
            else:  # 'yz'
                target = add(center, (0, dx, dy))
                distance_squared = dx*dx + dy*dy

            if not in_bounds(target):
                continue

            if abs(distance_squared - radius_squared) <= 0.1:
                perimeter.add(target)

    return perimeter

# ==============================================================================
# SYMMETRY AND TRANSFORMATIONS
# ==============================================================================

def identity() -> List[Coord]:
    """Identity transformation matrix."""
    return [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

def apply_matrix(mat: List[Coord], v: Coord) -> Coord:
    """Optimized matrix multiplication with 3×3 matrix."""
    return (mat[0][0] * v[X] + mat[0][1] * v[Y] + mat[0][2] * v[Z],
            mat[1][0] * v[X] + mat[1][1] * v[Y] + mat[1][2] * v[Z],
            mat[2][0] * v[X] + mat[2][1] * v[Y] + mat[2][2] * v[Z])

def rotate_around_axis(center: Coord, point: Coord, axis: Coord, angle_degrees: float) -> Coord:
    """Rotate point around axis through center."""
    # Translate to origin
    translated = sub(point, center)

    # Normalize axis
    axis_norm = normalize(axis)

    # Convert angle to radians
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Rodrigues' rotation formula
    axis_cross = cross_product(axis_norm, translated)
    axis_dot = dot_product(axis_norm, translated)

    rotated = add(
        add(scale(translated, cos_a), scale(axis_cross, sin_a)),
        scale(axis_norm, axis_dot * (1 - cos_a))
    )

    # Translate back
    return add(rotated, center)

def reflect_across_plane(center: Coord, point: Coord, normal: Coord) -> Coord:
    """Reflect point across plane through center with normal."""
    # Vector from center to point
    to_point = sub(point, center)

    # Project onto normal
    proj_length = dot_product(to_point, normal) / dot_product(normal, normal)
    proj = scale(normal, proj_length)

    # Reflect
    reflected = sub(to_point, scale(proj, 2))

    return add(center, reflected)

# ==============================================================================
# INDEXING AND ENCODING
# ==============================================================================

def coord_to_idx(c: Coord) -> int:
    """Fast coordinate to index conversion."""
    return c[X] + c[Y] * SIZE_X + c[Z] * SIZE_X * SIZE_Y

def idx_to_coord(idx: int) -> Coord:
    """Fast index to coordinate conversion."""
    x = idx % SIZE_X
    y = (idx // SIZE_X) % SIZE_Y
    z = idx // (SIZE_X * SIZE_Y)
    return (x, y, z)

def coord_to_idx_batch(coords: List[Coord]) -> List[int]:
    """Batch coordinate to index conversion."""
    return [c[X] + c[Y] * SIZE_X + c[Z] * SIZE_X * SIZE_Y for c in coords]

def idx_to_coord_batch(indices: List[int]) -> List[Coord]:
    """Batch index to coordinate conversion."""
    return [
        (idx % SIZE_X, (idx // SIZE_X) % SIZE_Y, idx // (SIZE_X * SIZE_Y))
        for idx in indices
    ]

# ==============================================================================
# NEIGHBORHOOD OPERATIONS
# ==============================================================================

def get_neighbors_6(center: Coord) -> List[Coord]:
    """Get 6-connected neighbors (face-sharing)."""
    neighbors = []
    for direction in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        neighbor = add(center, direction)
        if in_bounds(neighbor):
            neighbors.append(neighbor)
    return neighbors

def get_neighbors_26(center: Coord) -> List[Coord]:
    """Get 26-connected neighbors (including edges and corners)."""
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbor = add(center, (dx, dy, dz))
                if in_bounds(neighbor):
                    neighbors.append(neighbor)
    return neighbors

def get_neighbors_18(center: Coord) -> List[Coord]:
    """Get 18-connected neighbors (face and edge sharing)."""
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                # Exclude pure corner connections (all coordinates change)
                if abs(dx) + abs(dy) + abs(dz) <= 2:
                    neighbor = add(center, (dx, dy, dz))
                    if in_bounds(neighbor):
                        neighbors.append(neighbor)
    return neighbors

# ==============================================================================
# PERFORMANCE OPTIMIZATIONS
# ==============================================================================

# Pre-calculated direction vectors
DIRECTION_VECTORS = {
    'cardinal': [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)],
    'diagonal_2d': [(1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0)],
    'diagonal_3d': [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1), (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)],
}

# Pre-calculated distance tables
def precompute_distance_table(max_distance: int = 10) -> Dict[int, List[Coord]]:
    """Precompute coordinates within specific distances from origin."""
    distance_table = {}
    origin = (0, 0, 0)

    for distance in range(max_distance + 1):
        coords = []
        for x in range(-distance, distance + 1):
            for y in range(-distance, distance + 1):
                for z in range(-distance, distance + 1):
                    if manhattan(origin, (x, y, z)) == distance:
                        coords.append((x, y, z))
        distance_table[distance] = coords

    return distance_table

# Global distance table for common operations
DISTANCE_TABLE = precompute_distance_table(15)

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def bounding_box(coords: List[Coord]) -> Tuple[Coord, Coord]:
    """Get bounding box of coordinates."""
    if not coords:
        return ((0, 0, 0), (0, 0, 0))

    min_coord = (min(c[X] for c in coords), min(c[Y] for c in coords), min(c[Z] for c in coords))
    max_coord = (max(c[X] for c in coords), max(c[Y] for c in coords), max(c[Z] for c in coords))
    return (min_coord, max_coord)

def center_of_mass(coords: List[Coord]) -> Coord:
    """Calculate center of mass for coordinates."""
    if not coords:
        return (0, 0, 0)

    total_x = sum(c[X] for c in coords)
    total_y = sum(c[Y] for c in coords)
    total_z = sum(c[Z] for c in coords)
    n = len(coords)

    return (total_x // n, total_y // n, total_z // n)

def farthest_point(coords: List[Coord], reference: Coord) -> Coord:
    """Find farthest coordinate from reference point."""
    if not coords:
        return reference

    max_distance = -1
    farthest = coords[0]

    for coord in coords:
        distance = euclidean_squared(coord, reference)
        if distance > max_distance:
            max_distance = distance
            farthest = coord

    return farthest

# ==============================================================================
# BATCH OPERATIONS
# ==============================================================================

def batch_manhattan_distance(coords_a: List[Coord], coords_b: List[Coord]) -> List[int]:
    """Calculate Manhattan distances between coordinate pairs."""
    return [manhattan(a, b) for a, b in zip(coords_a, coords_b)]

def batch_euclidean_distance(coords_a: List[Coord], coords_b: List[Coord]) -> List[float]:
    """Calculate Euclidean distances between coordinate pairs."""
    return [euclidean(a, b) for a, b in zip(coords_a, coords_b)]

def filter_by_distance(center: Coord, coords: List[Coord], max_distance: int, distance_type: str = 'manhattan') -> List[Coord]:
    """Filter coordinates by maximum distance from center."""
    if distance_type == 'manhattan':
        return [c for c in coords if manhattan(center, c) <= max_distance]
    elif distance_type == 'euclidean':
        max_distance_squared = max_distance * max_distance
        return [c for c in coords if euclidean_squared(center, c) <= max_distance_squared]
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")

# ==============================================================================
# PERFORMANCE MONITORING
# ==============================================================================

def get_memory_usage() -> int:
    """Estimate memory usage of precomputed tables."""
    total = 0

    # Distance table
    for distance_list in DISTANCE_TABLE.values():
        total += len(distance_list) * 12  # 3 ints * 4 bytes each

    # Direction vectors
    for direction_list in DIRECTION_VECTORS.values():
        total += len(direction_list) * 12

    return total

def benchmark_operations(iterations: int = 100000) -> Dict[str, float]:
    """Benchmark geometric operations."""
    import time

    test_coord = (5, 5, 5)
    test_coords = [(i, i, i) for i in range(10)]

    results = {}

    # Benchmark single operations
    start = time.perf_counter()
    for _ in range(iterations):
        in_bounds(test_coord)
    results['in_bounds'] = (time.perf_counter() - start) / iterations * 1e6  # microseconds

    # Benchmark batch operations
    start = time.perf_counter()
    for _ in range(iterations // 10):
        in_bounds_batch(test_coords)
    results['in_bounds_batch'] = (time.perf_counter() - start) / (iterations // 10) * 1e6

    return results
