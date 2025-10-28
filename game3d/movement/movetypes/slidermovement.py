# slidermovement.py - FIXED (parameter name consistency)
"""Slider movement – use only public cache manager API."""
from __future__ import annotations
import numpy as np
from numba import njit, prange
from typing import List, TYPE_CHECKING

from game3d.common.enums import Color
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import in_bounds_scalar, in_bounds
from game3d.common.cache_utils import ensure_int_coords

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager

def generate_moves(
    piece_type: str,
    pos: tuple[int, int, int],
    color: Color,
    max_distance: int = 8,
    *,
    directions: np.ndarray,
    cache_manager: 'OptimizedCacheManager',  # FIXED: Consistent parameter name
) -> List[Move]:
    """Generate every slider move using cache manager's public API."""
    pos = ensure_int_coords(*pos)

    # Use public API to get occupancy array
    occupancy = cache_manager.get_occupancy_array_readonly()

    raw = generate_slider_moves_kernel(
        pos=pos,
        directions=directions,
        occupancy=occupancy,
        color=color.value,  # Convert to int for kernel
        max_distance=max_distance,
    )

    return [
        Move.create_simple(
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
            if not in_bounds_scalar(nx, ny, nz):
                break
            occ = occupancy[nz, ny, nx]

            # CRITICAL FIX: Enhanced occupancy logic
            if occ == 0:                       # empty
                move_coords[d_idx, count] = (nx, ny, nz)
                is_capture[d_idx, count] = False
                count += 1
            elif occ != color:                 # enemy
                move_coords[d_idx, count] = (nx, ny, nz)
                is_capture[d_idx, count] = True
                count += 1
                break
            else:                              # own piece - stop here
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
# 3.  Backward compatibility wrapper
# ------------------------------------------------------------------
def generate_moves_with_occupancy(
    piece_type: str,
    pos: tuple[int, int, int],
    color: int,
    max_distance: int = 8,
    *,
    directions: np.ndarray,
    occupancy: np.ndarray,
) -> List[Move]:
    """Backward compatibility wrapper for existing callers."""
    # Create a mock cache manager for compatibility
    class MockCacheManager:
        def __init__(self, occupancy):
            self.occupancy = type('Occupancy', (), {'_occ': occupancy})()

    mock_cache = MockCacheManager(occupancy)
    return generate_moves(
        piece_type=piece_type,
        pos=pos,
        color=color,
        max_distance=max_distance,
        directions=directions,
        cache_manager=mock_cache
    )

# ------------------------------------------------------------------
# 4.  Convenience re-exports
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
