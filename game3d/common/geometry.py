"""Game-agnostic 3-D geometry helpers."""

import math
from typing import Iterable, Tuple, List
from .constants import SIZE_X, SIZE_Y, SIZE_Z, X, Y, Z

Coord = Tuple[int, int, int]

# ---------- bounds ----------
def in_bounds(c: Coord) -> bool:
    return 0 <= c[X] < SIZE_X and 0 <= c[Y] < SIZE_Y and 0 <= c[Z] < SIZE_Z

def clamp(c: Coord) -> Coord:
    return (max(0, min(c[X], SIZE_X - 1)),
            max(0, min(c[Y], SIZE_Y - 1)),
            max(0, min(c[Z], SIZE_Z - 1)))

# ---------- vectors ----------
def add(a: Coord, b: Coord) -> Coord:
    return (a[X] + b[X], a[Y] + b[Y], a[Z] + b[Z])

def sub(a: Coord, b: Coord) -> Coord:
    return (a[X] - b[X], a[Y] - b[Y], a[Z] - b[Z])

def scale(v: Coord, k: int) -> Coord:
    return (v[X] * k, v[Y] * k, v[Z] * k)

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[X] - b[X]) + abs(a[Y] - b[Y]) + abs(a[Z] - b[Z])

# ---------- rays ----------
def ray_iter(start: Coord, step: Coord, max_steps: int = SIZE_X) -> Iterable[Coord]:
    """Yield coordinates along a ray (excludes start)."""
    for n in range(1, max_steps + 1):
        c = add(start, scale(step, n))
        if not in_bounds(c):
            break
        yield c

# ---------- symmetry group (48 rotations / reflections) ----------
def identity() -> List[Coord]:
    return [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

def apply_matrix(mat: List[Coord], v: Coord) -> Coord:
    """Multiply 3×3 matrix with column vector v."""
    return (mat[0][0] * v[X] + mat[0][1] * v[Y] + mat[0][2] * v[Z],
            mat[1][0] * v[X] + mat[1][1] * v[Y] + mat[1][2] * v[Z],
            mat[2][0] * v[X] + mat[2][1] * v[Y] + mat[2][2] * v[Z])

# ---------- indexing ----------
def coord_to_idx(c: Coord) -> int:
    """Flatten (x,y,z) → int in [0,VOLUME)."""
    return c[X] + c[Y] * SIZE_X + c[Z] * SIZE_X * SIZE_Y

def idx_to_coord(idx: int) -> Coord:
    x = idx % SIZE_X
    y = (idx // SIZE_X) % SIZE_Y
    z = idx // (SIZE_X * SIZE_Y)
    return (x, y, z)
