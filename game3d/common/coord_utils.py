"""
Centralised coordinate validation / sanitisation for the 9×9×9 board.

Usage
-----
from game3d.common.coord_utils import sanitize_coord, sanitize_many

# --- inside any public API ------------------------------------------
try:
    coord = sanitize_coord(raw_coord)          # raises if seriously broken
except ValueError as e:
    log.error("Bad coordinate: %s", e)
    return []

# --- bulk path / tensor scan ----------------------------------------
coords = sanitize_many(raw_coords)           # silently drops bad ones
"""
from __future__ import annotations
import math
from typing import Iterable, List, Tuple, overload

Coord = Tuple[int, int, int]

BOARD_SIZE: int = 9          # single source of truth
_MAX: int = BOARD_SIZE - 1   # 8

# ------------------------------------------------------------------ #
# fast inline helpers – keep them @njit-friendly
# ------------------------------------------------------------------ #
def _in_range(c: int) -> bool:
    return 0 <= c < BOARD_SIZE

def _is_valid(coord: Coord) -> bool:
    return _in_range(coord[0]) and _in_range(coord[1]) and _in_range(coord[2])

# ------------------------------------------------------------------ #
# public API
# ------------------------------------------------------------------ #
def sanitize_coord(coord: Coord, *, clamp: bool = False) -> Coord:
    """
    Return a clean coordinate tuple.

    Parameters
    ----------
    coord : (int, int, int)
    clamp : bool
        False  -> raise ValueError if any component is out of range
        True   -> silently clip to 0–8

    Returns
    -------
    (int, int, int)  – guaranteed 0 ≤ x,y,z ≤ 8
    """
    if clamp:
        x = max(0, min(coord[0], _MAX))
        y = max(0, min(coord[1], _MAX))
        z = max(0, min(coord[2], _MAX))
        return x, y, z

    if not _is_valid(coord):
        raise ValueError(
            f"Coordinate {coord} out of range for board size {BOARD_SIZE}"
        )
    return coord

@overload
def sanitize_many(coords: Iterable[Coord], *, clamp: bool = False) -> List[Coord]: ...

def sanitize_many(coords: Iterable[Coord], *, clamp: bool = False) -> List[Coord]:
    """
    Bulk version.  Always returns a list; invalid coordinates are either
    clamped or skipped (when clamp=False).
    """
    out: List[Coord] = []
    for c in coords:
        try:
            out.append(sanitize_coord(c, clamp=clamp))
        except ValueError:
            continue
    return out

def coord_to_idx(coord: Coord) -> int:
    """9×9×9 linear index – safe version."""
    x, y, z = sanitize_coord(coord)        # never clamp silently here
    return x + BOARD_SIZE * y + BOARD_SIZE * BOARD_SIZE * z

def idx_to_coord(idx: int) -> Coord:
    """Inverse of above – always returns a valid tuple."""
    if not 0 <= idx < BOARD_SIZE ** 3:
        raise ValueError(f"Index {idx} out of range")
    z, rem = divmod(idx, BOARD_SIZE * BOARD_SIZE)
    y, x = divmod(rem, BOARD_SIZE)
    return x, y, z

# ------------------------------------------------------------------ #
# geometric helpers that *produce* coordinates
# ------------------------------------------------------------------ #
def add_coords(a: Coord, b: Coord) -> Coord:
    """Component-wise addition – result is validated."""
    return sanitize_coord((a[0] + b[0], a[1] + b[1], a[2] + b[2]), clamp=True)

def subtract_coords(a: Coord, b: Coord) -> Coord:
    """Component-wise subtraction – result is validated."""
    return sanitize_coord((a[0] - b[0], a[1] - b[1], a[2] - b[2]), clamp=True)

def scale_coord(c: Coord, factor: int) -> Coord:
    return sanitize_coord((c[0] * factor, c[1] * factor, c[2] * factor), clamp=True)

def get_aura_squares(center: Coord, radius: int = 2) -> List[Coord]:
    """Return *only* in-bounds squares around center."""
    center = sanitize_coord(center)          # defensive
    aura: List[Coord] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == dy == dz == 0:
                    continue
                sq = sanitize_coord((center[0] + dx, center[1] + dy, center[2] + dz),
                                    clamp=True)
                aura.append(sq)
    return aura
