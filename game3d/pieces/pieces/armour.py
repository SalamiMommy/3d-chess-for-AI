"""ARMOUR piece movement and protection logic - fully numpy native."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numba import jit

from game3d.common.shared_types import *
from game3d.common.shared_types import Color, PieceType, Result

if TYPE_CHECKING: pass

from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS, BUFFED_KING_MOVEMENT_VECTORS

def is_armour_protected(sq: np.ndarray, cache_manager: 'OptimizedCacheManager') -> bool:
    """Check if square contains an ARMOUR piece."""
    sq_arr = np.atleast_2d(sq).astype(COORD_DTYPE)
    if sq_arr.shape[0] == 0:
        return False
    pieces = cache_manager.occupancy.get_batch(sq_arr)
    return bool(pieces and pieces[0] is not None and pieces[0]["piece_type"] == ARMOUR_TYPE)

def batch_is_armour_protected(
    coords: np.ndarray,
    cache_manager: 'OptimizedCacheManager'
):
    """Vectorized armour protection check."""
    coords_arr = np.atleast_2d(coords).astype(COORD_DTYPE)
    if coords_arr.shape[0] == 0:
        return np.array([], dtype=BOOL_DTYPE)
    _, types = cache_manager.occupancy.batch_get_attributes(coords_arr)
    return types == ARMOUR_TYPE

def get_armoured_squares(
    cache_manager: 'OptimizedCacheManager',
    controller: Color
):
    """Get all squares occupied by ARMOUR pieces of given color - fully vectorized."""
    # Get all pieces of the controller color - avoid list() conversion
    coords_list = []
    types_list = []
    for coord, piece in cache_manager.get_pieces_of_color(controller):
        coords_list.append(coord)
        types_list.append(piece["piece_type"])

    if not coords_list:
        return get_empty_coord_batch()

    # Vectorized extraction of coordinates and types
    coords_arr = np.array(coords_list, dtype=COORD_DTYPE)
    types_arr = np.array(types_list, dtype=PIECE_TYPE_DTYPE)

    # Vectorized filtering for ARMOUR pieces
    armour_mask = (types_arr == ARMOUR_TYPE)

    if not np.any(armour_mask):
        return get_empty_coord_batch()

    return coords_arr[armour_mask]

@jit(nopython=True)
def _batch_check_armour_protection_numba(
    coords: np.ndarray,
    armoured_squares: np.ndarray
) -> np.ndarray:
    """Numba-compiled core for batch armour protection check - optimized with flat indices."""
    n = coords.shape[0]
    m = armoured_squares.shape[0]
    
    if n == 0 or m == 0:
        return np.zeros(n, dtype=BOOL_DTYPE)
    
    # Convert coordinates to flat indices for O(1) lookup
    # Using SIZE constant from shared_types
    SIZE_SQUARED = SIZE * SIZE
    coord_flat = coords[:, 0] + SIZE * coords[:, 1] + SIZE_SQUARED * coords[:, 2]
    armour_flat = armoured_squares[:, 0] + SIZE * armoured_squares[:, 1] + SIZE_SQUARED * armoured_squares[:, 2]
    
    # Vectorized membership test - O(n+m) instead of O(n*m)
    result = np.zeros(n, dtype=BOOL_DTYPE)
    for i in range(n):
        for j in range(m):
            if coord_flat[i] == armour_flat[j]:
                result[i] = True
                break
    
    return result

def batch_check_armour_protection(
    coords: np.ndarray,
    armoured_squares: np.ndarray
) -> np.ndarray:
    """Check if coordinates are protected by armour - fully vectorized."""
    coords_arr = np.atleast_2d(coords).astype(COORD_DTYPE)
    armoured_squares_arr = np.atleast_2d(armoured_squares).astype(COORD_DTYPE)

    if coords_arr.shape[0] == 0 or armoured_squares_arr.shape[0] == 0:
        return np.zeros(coords_arr.shape[0], dtype=BOOL_DTYPE)

    # Use numba-optimized function for performance
    return _batch_check_armour_protection_numba(coords_arr, armoured_squares_arr)

__all__ = []

