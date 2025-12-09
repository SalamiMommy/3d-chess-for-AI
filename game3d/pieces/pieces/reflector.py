# reflector.py - REFACTORED to be numpy-native like bishop.py
"""Reflecting-Bishop – diagonal slider that bounces off walls (max 3 reflections)."""

import numpy as np
from numba import njit
from typing import List, TYPE_CHECKING
from game3d.common.shared_types import COORD_DTYPE, SIZE, SIZE_SQUARED, Color, PieceType
from game3d.common.coord_utils import CoordinateUtils, ensure_coords

if TYPE_CHECKING: pass

from game3d.pieces.pieces.bishop import BISHOP_MOVEMENT_VECTORS

# 12 diagonal directions (same as Bishop)
_REFLECTOR_DIRS = BISHOP_MOVEMENT_VECTORS

@njit(cache=True, fastmath=True, boundscheck=False)
def _trace_reflector_rays_batch(
    occupancy: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    max_bounces: int,
    color_code: int,
    ignore_occupancy: bool = False
):
    """
    Trace reflecting rays for a batch of pieces.
    """
    n_pieces = origins.shape[0]
    n_dirs = directions.shape[0]
    
    # Estimate max moves: n_pieces * n_dirs * max_path_len (24)
    # This is an upper bound, we'll slice it at the end
    max_moves = n_pieces * n_dirs * 24
    moves = np.empty((max_moves, 6), dtype=COORD_DTYPE)
    count = 0

    for i in range(n_pieces):
        origin = origins[i]
        
        for j in range(n_dirs):
            direction = directions[j]
            
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
                           next_pos[2] * SIZE_SQUARED)
                occupant = occupancy[flat_idx]

                if occupant == 0:
                    # Empty square - quiet move
                    moves[count, 0] = origin[0]
                    moves[count, 1] = origin[1]
                    moves[count, 2] = origin[2]
                    moves[count, 3] = next_pos[0]
                    moves[count, 4] = next_pos[1]
                    moves[count, 5] = next_pos[2]
                    count += 1
                else:
                    # Occupied square
                    if ignore_occupancy:
                        moves[count, 0] = origin[0]
                        moves[count, 1] = origin[1]
                        moves[count, 2] = origin[2]
                        moves[count, 3] = next_pos[0]
                        moves[count, 4] = next_pos[1]
                        moves[count, 5] = next_pos[2]
                        count += 1
                        # CONTINUE RAY
                    else:
                        # Capture if enemy piece
                        if occupant != color_code:
                            moves[count, 0] = origin[0]
                            moves[count, 1] = origin[1]
                            moves[count, 2] = origin[2]
                            moves[count, 3] = next_pos[0]
                            moves[count, 4] = next_pos[1]
                            moves[count, 5] = next_pos[2]
                            count += 1
                        # Stop ray after encountering any piece
                        break

                pos = next_pos

    return moves[:count]

    """
    Generate all legal moves for a reflecting bishop piece.
    Uses numpy-native operations and follows the same pattern as bishop.py.
    """
    # Validate and normalize input position
    pos_arr = pos.astype(COORD_DTYPE)
    
    # Handle single input
    if pos_arr.ndim == 1:
        pos_arr = pos_arr.reshape(1, 3)
        
    if pos_arr.shape[0] == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Get flattened occupancy array for fast vectorized lookups
    occupancy_flat = cache_manager.occupancy_cache.get_flattened_occupancy()

    # Map color enum to internal occupancy code (1=WHITE, 2=BLACK)
    friendly_code = 1 if color == Color.WHITE else 2

    # Run batch kernel
    return _trace_reflector_rays_batch(
        occupancy=occupancy_flat,
        origins=pos_arr,
        directions=_REFLECTOR_DIRS,
        max_bounces=max_bounces,
        color_code=friendly_code,
        ignore_occupancy=ignore_occupancy
    )

__all__ = []

