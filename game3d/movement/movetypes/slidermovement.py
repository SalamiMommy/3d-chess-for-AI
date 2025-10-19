# slidermovement.py (updated imports)
"""Slider movement – kernel-only, no pre-computed rays."""
from __future__ import annotations
import numpy as np
from numba import njit, prange
from typing import List

from game3d.common.enums import Color
from game3d.movement.movepiece import Move, MOVE_FLAGS, convert_legacy_move_args
from game3d.common.common import in_bounds, in_bounds_scalar  # UPDATED: Add scalar for njit
# ------------------------------------------------------------------
# 1.  Public generator (single entry-point)
# ------------------------------------------------------------------
def generate_moves(
    piece_type: str,                       # still accepted for logging / future use
    pos: tuple[int, int, int],
    color: int,
    max_distance: int = 8,
    *,
    directions: np.ndarray,                # ← REQUIRED now
    occupancy: np.ndarray,                 # ← pass the current mask directly
) -> List[Move]:
    """Generate every slider move by running the Numba kernel once."""
    raw = generate_slider_moves_kernel(
        pos=pos,
        directions=directions,
        occupancy=occupancy,
        color=color,
        max_distance=max_distance,
    )
    return [
        convert_legacy_move_args(
            from_coord=pos,
            to_coord=(nx, ny, nz),
            is_capture=is_cap,
        )
        for nx, ny, nz, is_cap in raw
    ]


# ------------------------------------------------------------------
# 2.  Hot Numba kernel (unchanged)
# ------------------------------------------------------------------
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
            # UPDATED: Use in_bounds_scalar for njit
            if not in_bounds_scalar(nx, ny, nz):
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
# ------------------------------------------------------------------
# 3.  Convenience re-exports (keep existing callers happy)
# ------------------------------------------------------------------
def dirty_squares_slider(mv, mover, cache_manager):
    """Same helper as before—unchanged."""
    from itertools import product
    dirty = set()
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
