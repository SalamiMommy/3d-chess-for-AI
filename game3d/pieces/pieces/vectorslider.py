"""Vector-Slider piece implementation with 152 primitive directions."""
from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.common.registry import register
from game3d.movement.slider_engine import get_slider_movement_generator
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Optimized 152 primitive directions using numpy operations
def _vector_dirs_numpy() -> np.ndarray:
    """Generate primitive direction vectors using numpy operations."""
    # Create coordinate grids using meshgrid for vectorized calculation
    dx_range = np.arange(-3, 4, dtype=COORD_DTYPE)
    dy_range = np.arange(-3, 4, dtype=COORD_DTYPE)
    dz_range = np.arange(-3, 4, dtype=COORD_DTYPE)

    # Create all combinations using meshgrid
    dx_grid, dy_grid, dz_grid = np.meshgrid(dx_range, dy_range, dz_range, indexing='ij')

    # Flatten and create direction vectors
    dx_flat = dx_grid.ravel()
    dy_flat = dy_grid.ravel()
    dz_flat = dz_grid.ravel()

    # Remove (0,0,0) - no movement
    mask = (dx_flat != 0) | (dy_flat != 0) | (dz_flat != 0)
    dx_flat = dx_flat[mask]
    dy_flat = dy_flat[mask]
    dz_flat = dz_flat[mask]

    # Calculate GCD for each direction vector and filter to primitive directions
    directions = []
    for dx, dy, dz in zip(dx_flat, dy_flat, dz_flat):
        # Calculate GCD of absolute values
        abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)
        g = np.gcd.reduce([abs_dx, abs_dy, abs_dz])

        # Preserve original signs and apply GCD reduction
        dx_reduced = dx // g if g > 0 else dx
        dy_reduced = dy // g if g > 0 else dy
        dz_reduced = dz // g if g > 0 else dz

        # Ensure we have primitive directions within bounds
        if abs(dx_reduced) <= 3 and abs(dy_reduced) <= 3 and abs(dz_reduced) <= 3:
            directions.append([dx_reduced, dy_reduced, dz_reduced])

    # Convert to numpy array and ensure proper dtype
    directions_array = np.array(directions, dtype=COORD_DTYPE)

    # Remove duplicates by using unique rows
    unique_dirs = np.unique(directions_array, axis=0)
    return unique_dirs

# Generate and cache the vector directions
VECTOR_DIRECTIONS = _vector_dirs_numpy()

def generate_vector_slider_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate vector slider moves from numpy-native position array."""
    pos_arr = pos.astype(COORD_DTYPE)

    # Use the slider engine
    slider_engine = get_slider_movement_generator()
    return slider_engine.generate_slider_moves_array(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=VECTOR_DIRECTIONS,
        max_distance=8
    )

@register(PieceType.VECTORSLIDER)
def vectorslider_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Registered dispatcher for Vector Slider moves."""
    return generate_vector_slider_moves(state.cache_manager, state.color, pos)

__all__ = ['VECTOR_DIRECTIONS', 'generate_vector_slider_moves', 'vectorslider_move_dispatcher']
