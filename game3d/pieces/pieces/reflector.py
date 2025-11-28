# reflector.py - REFACTORED to be numpy-native like bishop.py
"""Reflecting-Bishop – diagonal slider that bounces off walls (max 3 reflections)."""

import numpy as np
from numba import njit
from typing import List, TYPE_CHECKING
from game3d.common.shared_types import COORD_DTYPE, SIZE, Color, PieceType
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import CoordinateUtils, ensure_coords

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

# 8 pure-diagonal directions using consistent coordinate dtype
_REFLECTOR_DIRS = np.array(
    [[dx, dy, dz] for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)],
    dtype=COORD_DTYPE,
)

@njit(cache=True, fastmath=True, boundscheck=False)
def _trace_reflector_ray(
    occupancy: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    max_bounces: int,
    color_code: int,
    ignore_occupancy: bool = False
) -> tuple:
    """
    Trace a single reflecting ray and return target coordinates with capture flags.
    """
    # Pre-allocate buffers for this ray (max 24 squares per ray on 9x9x9 board)
    coords = np.empty((24, 3), dtype=np.int32)
    captures = np.empty(24, dtype=np.bool_)
    count = 0

    pos = origin.copy()
    dir_vec = direction.copy()
    bounces = 0

    for _ in range(24):  # Maximum path length before termination
        next_pos = pos + dir_vec

        # Boundary check with bounce logic
        out_x = next_pos[0] < 0 or next_pos[0] >= SIZE
        out_y = next_pos[1] < 0 or next_pos[1] >= SIZE
        out_z = next_pos[2] < 0 or next_pos[2] >= SIZE
        is_out_of_bounds = out_x or out_y or out_z

        if is_out_of_bounds:
            if bounces >= max_bounces:
                break

            # Reflect direction components that hit boundaries
            if out_x:
                dir_vec[0] = -dir_vec[0]
            if out_y:
                dir_vec[1] = -dir_vec[1]
            if out_z:
                dir_vec[2] = -dir_vec[2]

            bounces += 1
            continue

        # Check target square occupancy
        flat_idx = (next_pos[0] +
                   next_pos[1] * SIZE +
                   next_pos[2] * SIZE * SIZE)
        occupant = occupancy[flat_idx]

        if occupant == 0:
            # Empty square - quiet move
            coords[count] = next_pos
            captures[count] = False
            count += 1
        else:
            # Occupied square
            if ignore_occupancy:
                # Treat as a move (capture logic irrelevant for raw moves, but we mark it)
                coords[count] = next_pos
                captures[count] = (occupant != color_code)
                count += 1
                # CONTINUE RAY
            else:
                # Capture if enemy piece
                if occupant != color_code:
                    coords[count] = next_pos
                    captures[count] = True
                    count += 1
                # Stop ray after encountering any piece
                break

        pos = next_pos

    return coords[:count], captures[:count]

def generate_reflecting_bishop_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray,
    max_bounces: int = 3,
    ignore_occupancy: bool = False
) -> np.ndarray:
    """
    Generate all legal moves for a reflecting bishop piece.
    Uses numpy-native operations and follows the same pattern as bishop.py.
    """
    # Validate and normalize input position - ENSURE 1D!
    origin = ensure_coords(pos).astype(COORD_DTYPE).squeeze()
    if not CoordinateUtils.in_bounds(origin):
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Get flattened occupancy array for fast vectorized lookups
    occupancy_flat = cache_manager.occupancy_cache.get_flattened_occupancy()

    # Map color enum to internal occupancy code (1=WHITE, 2=BLACK)
    friendly_code = 1 if color == Color.WHITE else 2

    # Accumulate results from all 8 diagonal directions
    direction_coords = []

    for direction in _REFLECTOR_DIRS:
        coords, captures = _trace_reflector_ray(
            occupancy=occupancy_flat,
            origin=origin,
            direction=direction,
            max_bounces=max_bounces,
            color_code=friendly_code,
            ignore_occupancy=ignore_occupancy
        )

        if len(coords) > 0:
            direction_coords.append(coords)

    # Early exit if no moves generated
    if not direction_coords:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Consolidate results from all directions
    all_to_coords = np.concatenate(direction_coords, axis=0)

    # Create move array: [from_x, from_y, from_z, to_x, to_y, to_z]
    n_moves = all_to_coords.shape[0]
    move_array = np.empty((n_moves, 6), dtype=COORD_DTYPE)
    move_array[:, 0:3] = origin
    move_array[:, 3:6] = all_to_coords

    return move_array

@register(PieceType.REFLECTOR)
def reflector_move_dispatcher(state: 'GameState', pos: np.ndarray, ignore_occupancy: bool = False) -> np.ndarray:
    """
    Registered dispatcher for REFLECTOR piece type.
    Delegates to numpy-native move generation.
    """
    return generate_reflecting_bishop_moves(
        cache_manager=state.cache_manager,
        color=state.color,
        pos=pos,
        max_bounces=3,
        ignore_occupancy=ignore_occupancy
    )

__all__ = ["generate_reflecting_bishop_moves"]
