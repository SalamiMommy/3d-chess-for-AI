"""Black-Hole piece - fully numpy native with vectorized operations."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from game3d.common.registry import register
from game3d.pieces.pieces.kinglike import generate_king_moves
from game3d.movement.movepiece import Move
from game3d.common.shared_types import (
    COORD_DTYPE, COLOR_DTYPE, Color, PieceType, BLACKHOLE, BLACKHOLE_PULL_RADIUS
)
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Black-hole movement vectors (same as king - 26 directions) - numpy native
BLACKHOLE_MOVEMENT_VECTORS = np.array([
    [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],  # YZ plane
    [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],  # XZ plane
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],  # XY plane
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],  # 3D corners (+X)
    [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],  # 3D corners (-X)
    [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0],  # Z and Y axes
    [1, 0, 0], [-1, 0, 0]  # X axis
], dtype=COORD_DTYPE)


def _toward_numpy(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Move one Chebyshev step from pos toward target. Returns (3,) array."""
    diff = target - pos
    direction = np.sign(diff).astype(COORD_DTYPE)
    return pos + direction

def generate_blackhole_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate blackhole moves (king-like single steps)."""
    return generate_king_moves(cache_manager, color, pos)

def suck_candidates_vectorized(
    cache_manager: 'OptimizedCacheManager',
    controller: Color,
) -> np.ndarray:
    """Find enemies to pull toward blackholes - fully vectorized numpy operations."""
    # Get all friendly pieces
    all_coords = cache_manager.occupancy_cache.get_positions(controller)
    if all_coords.size == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)
    
    # Filter for blackholes
    _, piece_types = cache_manager.occupancy_cache.batch_get_attributes(all_coords)
    blackhole_mask = piece_types == PieceType.BLACKHOLE

    if not np.any(blackhole_mask):
        return np.empty((0, 6), dtype=COORD_DTYPE)

    blackhole_coords = all_coords[blackhole_mask]

    # Get enemy coordinates
    enemy_color = controller.opposite()
    enemy_coords = cache_manager.occupancy_cache.get_positions(enemy_color)

    if enemy_coords.size == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Vectorized distance calculation: all enemies vs all blackholes
    # Shape: (num_enemies, num_blackholes, 3)
    diff = enemy_coords[:, np.newaxis, :] - blackhole_coords[np.newaxis, :, :]
    cheb_distances = np.max(np.abs(diff), axis=2)  # Shape: (num_enemies, num_blackholes)

    # Vectorized range checking
    in_range = cheb_distances <= BLACKHOLE_PULL_RADIUS  # Shape: (num_enemies, num_blackholes)

    # Find closest valid blackhole for each enemy using vectorized operations
    # Use large penalty for out-of-range distances
    penalty = 1e6
    penalized_distances = cheb_distances + penalty * ~in_range
    closest_indices = np.argmin(penalized_distances, axis=1)  # Shape: (num_enemies,)

    # Vectorized pull calculation for all enemies
    closest_holes = blackhole_coords[closest_indices]  # Shape: (num_enemies, 3)

    # Calculate pull positions for all enemies
    diff_vectors = closest_holes - enemy_coords
    pull_directions = np.sign(diff_vectors).astype(COORD_DTYPE)
    pull_positions = enemy_coords + pull_directions  # Shape: (num_enemies, 3)

    # Vectorized bounds and occupancy checking
    valid_bounds = in_bounds_vectorized(pull_positions)

    # Batch occupancy check for all pull positions
    valid_occupancy = np.array([
        cache_manager.occupancy_cache.get(pos) is None for pos in pull_positions
    ])

    # Filter valid pulls using vectorized operations
    valid_pulls_mask = valid_bounds & valid_occupancy & np.any(in_range, axis=1)

    if not np.any(valid_pulls_mask):
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Create final pull pairs using vectorized concatenation
    valid_enemies = enemy_coords[valid_pulls_mask]
    valid_pulls = pull_positions[valid_pulls_mask]

    return np.hstack([valid_enemies, valid_pulls])


@register(PieceType.BLACKHOLE)
def blackhole_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Dispatch blackhole moves with numpy-native coordinates."""
    return generate_blackhole_moves(state.cache_manager, state.color, pos)


__all__ = ["generate_blackhole_moves", "suck_candidates_vectorized", "BLACKHOLE_MOVEMENT_VECTORS"]
