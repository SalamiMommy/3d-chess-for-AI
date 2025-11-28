"""ARMOUR piece movement and protection logic - fully numpy native."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numba import jit

from game3d.common.shared_types import (
    COORD_DTYPE, INDEX_DTYPE, BOOL_DTYPE, COLOR_DTYPE, PIECE_TYPE_DTYPE, FLOAT_DTYPE,
    SIZE, ARMOUR as ARMOUR_TYPE, get_empty_coord_batch
)
from game3d.common.registry import register
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.common.shared_types import Color, PieceType, Result


if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

# ARMOUR movement vectors - orthogonal movement (6 directions) - numpy native
ARMOUR_MOVEMENT_VECTORS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
], dtype=COORD_DTYPE)

@jit(nopython=True, cache=True)
def _coords_match(coord1: np.ndarray, coord2: np.ndarray) -> bool:
    """Fast coordinate comparison."""
    return np.array_equal(coord1, coord2)

@jit(nopython=True, cache=True)
def _is_armour_at_coord(coord: np.ndarray, armour_coords: np.ndarray) -> bool:
    """Check if coordinate matches any armour position using vectorized comparison."""
    return np.any(np.all(armour_coords == coord, axis=1))

def generate_armour_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate legal one-step armour moves."""
    pos_arr = pos.astype(COORD_DTYPE)

    jump_engine = get_jump_movement_generator()
    return jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=ARMOUR_MOVEMENT_VECTORS,
        allow_capture=True,
    )

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
) -> np.ndarray:
    """Vectorized armour protection check."""
    coords_arr = np.atleast_2d(coords).astype(COORD_DTYPE)
    if coords_arr.shape[0] == 0:
        return np.array([], dtype=BOOL_DTYPE)
    _, types = cache_manager.occupancy.batch_get_attributes(coords_arr)
    return types == ARMOUR_TYPE

def get_armoured_squares(
    cache_manager: 'OptimizedCacheManager',
    controller: Color
) -> np.ndarray:
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

@register(PieceType.ARMOUR)
def armour_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Dispatcher for ARMOUR piece moves."""
    return generate_armour_moves(state.cache_manager, state.color, pos)

__all__ = [
    "ARMOUR_MOVEMENT_VECTORS",
    "generate_armour_moves",
    "is_armour_protected",
    "batch_is_armour_protected",
    "get_armoured_squares",
    "batch_check_armour_protection",
    "armour_move_dispatcher"
]
