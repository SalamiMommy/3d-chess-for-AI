"""
Face-Cone-Slider — 6 conical rays (fully numpy native).
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.shared_types import COORD_DTYPE, PieceType, SIZE_MINUS_1
from game3d.common.registry import register
from game3d.movement.slider_engine import get_slider_movement_generator
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager

def _generate_cone_vectors_numpy() -> np.ndarray:
    """Generate cone direction vectors using fully vectorized numpy operations."""
    # Cone axes: 6 primary directions
    cone_axes = np.array([
        [1, 0, 0], [-1, 0, 0],  # X-axis cones
        [0, 1, 0], [0, -1, 0],  # Y-axis cones
        [0, 0, 1], [0, 0, -1]   # Z-axis cones
    ], dtype=COORD_DTYPE)

    # Vectorized generation of all possible directions
    dy_range = np.arange(-SIZE_MINUS_1, SIZE_MINUS_1 + 1, dtype=COORD_DTYPE)
    dz_range = np.arange(-SIZE_MINUS_1, SIZE_MINUS_1 + 1, dtype=COORD_DTYPE)

    # Create all combinations using meshgrid for vectorization
    dy_mesh, dz_mesh = np.meshgrid(dy_range, dz_range, indexing='ij')

    # Flatten for processing
    dy_flat = dy_mesh.ravel()
    dz_flat = dz_mesh.ravel()

    # Pre-allocate arrays for directions
    all_directions = []

    for i, cone_axis in enumerate(cone_axes):
        px, py, pz = cone_axis

        if px != 0:  # X-cone - expand in YZ plane
            # Vectorized max expansion calculation
            max_expansion = np.maximum(1, np.maximum(np.abs(dy_flat), np.abs(dz_flat)))
            dx = px * max_expansion
            directions = np.column_stack([dx, dy_flat, dz_flat])
        elif py != 0:  # Y-cone - expand in XZ plane
            max_expansion = np.maximum(1, np.maximum(np.abs(dy_flat), np.abs(dz_flat)))
            dy_expanded = py * max_expansion
            directions = np.column_stack([dy_flat, dy_expanded, dz_flat])
        else:  # Z-cone - expand in XY plane
            max_expansion = np.maximum(1, np.maximum(np.abs(dy_flat), np.abs(dz_flat)))
            dz_expanded = pz * max_expansion
            directions = np.column_stack([dy_flat, dz_flat, dz_expanded])

        # Remove zero vectors
        non_zero_mask = np.any(directions != 0, axis=1)
        directions = directions[non_zero_mask]

        # Vectorized GCD calculation for primitive direction normalization
        abs_dirs = np.abs(directions)
        g = np.gcd.reduce(abs_dirs, axis=1)
        g[g == 0] = 1  # Avoid division by zero

        primitive_dirs = directions // g[:, np.newaxis]
        all_directions.append(primitive_dirs)

    # Combine all directions and remove duplicates
    all_dirs_array = np.vstack(all_directions)
    unique_dirs = np.unique(all_dirs_array, axis=0)

    return unique_dirs.astype(COORD_DTYPE)

# Piece-specific movement vectors - precomputed cone directions
FACE_CONE_MOVEMENT_VECTORS = _generate_cone_vectors_numpy()

def generate_face_cone_slider_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray,
    max_steps: int = SIZE_MINUS_1
) -> np.ndarray:
    """Generate face-cone slider moves using numpy-native operations."""
    pos_arr = pos.astype(COORD_DTYPE)

    # Use integrated slider generator with cone-specific vectors
    slider_engine = get_slider_movement_generator()
    return slider_engine.generate_slider_moves_array(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=FACE_CONE_MOVEMENT_VECTORS,
        max_distance=max_steps,
    )

@register(PieceType.CONESLIDER)
def face_cone_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Dispatcher for face-cone slider moves - receives numpy array position."""
    return generate_face_cone_slider_moves(
        cache_manager=state.cache_manager,
        color=state.color,
        pos=pos,
        max_steps=SIZE_MINUS_1
    )

__all__ = ['FACE_CONE_MOVEMENT_VECTORS', 'generate_face_cone_slider_moves']
