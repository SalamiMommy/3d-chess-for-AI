# whitehole.py - FULLY NUMPY-NATIVE (NO TUPLES ANYWHERE)
"""White-Hole — moves like a Speeder and pushes enemies 1 step away."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from game3d.common.shared_types import Color, PieceType, Result, WHITEHOLE
from game3d.common.registry import register
from game3d.pieces.pieces.kinglike import generate_king_moves
from game3d.movement.movepiece import Move
from game3d.common.shared_types import COORD_DTYPE, WHITEHOLE_PUSH_RADIUS
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState


def _away_numpy(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """1 Chebyshev step from pos away from target. Returns (3,) COORD_DTYPE."""
    diff = pos - target
    direction = np.sign(diff).astype(COORD_DTYPE)
    # np.sign already yields 0 where diff == 0, so np.where is redundant
    return pos + direction


def generate_whitehole_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """White-Hole moves exactly like a Speeder (king single steps)."""
    return generate_king_moves(cache_manager, color, pos, piece_type=PieceType.WHITEHOLE)


def push_candidates_vectorized(
    cache_manager: 'OptimizedCacheManager',
    controller: Color,
) -> np.ndarray:
    """
    Fully vectorized, tuple-free version.
    Returns (N, 6) int8 array: [ex, ey, ez, px, py, pz].
    """
    # Get all friendly pieces
    all_coords = cache_manager.occupancy_cache.get_positions(controller)
    if all_coords.size == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)
    
    # Filter for whiteholes
    # ✅ OPTIMIZATION: Use unsafe access (coords from get_positions are valid)
    _, piece_types = cache_manager.occupancy_cache.batch_get_attributes_unsafe(all_coords)
    whitehole_mask = piece_types == PieceType.WHITEHOLE
    
    if not np.any(whitehole_mask):
        return np.empty((0, 6), dtype=COORD_DTYPE)
    
    whitehole_coords = all_coords[whitehole_mask]

    # Get enemy coordinates
    enemy_color = controller.opposite()
    enemy_coords = cache_manager.occupancy_cache.get_positions(enemy_color)
    
    if enemy_coords.size == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    if enemy_coords.shape[0] == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Vectorized distance calculation: all enemies vs all whiteholes
    # Shape: (num_enemies, num_whiteholes, 3)
    diff = enemy_coords[:, np.newaxis, :] - whitehole_coords[np.newaxis, :, :]
    cheb_distances = np.max(np.abs(diff), axis=2)  # Shape: (num_enemies, num_whiteholes)

    # Vectorized range checking
    in_range = cheb_distances <= WHITEHOLE_PUSH_RADIUS  # Shape: (num_enemies, num_whiteholes)

    # Find closest valid whitehole for each enemy using vectorized operations
    # Use large penalty for out-of-range distances
    penalty = 1e6
    penalized_distances = cheb_distances + penalty * ~in_range
    closest_indices = np.argmin(penalized_distances, axis=1)  # Shape: (num_enemies,)

    # Vectorized push calculation for all enemies
    closest_holes = whitehole_coords[closest_indices]  # Shape: (num_enemies, 3)

    # Calculate push positions for all enemies
    # Push AWAY from hole: direction is sign(enemy - hole)
    diff_vectors = enemy_coords - closest_holes
    push_directions = np.sign(diff_vectors).astype(COORD_DTYPE)
    push_positions = enemy_coords + push_directions  # Shape: (num_enemies, 3)

    # Vectorized bounds and occupancy checking
    valid_bounds = in_bounds_vectorized(push_positions)

    # ✅ OPTIMIZATION: Vectorized occupancy check using unsafe access
    # We only check occupancy for positions that are in bounds
    # Initialize valid_occupancy as False
    valid_occupancy = np.zeros(push_positions.shape[0], dtype=bool)
    
    if np.any(valid_bounds):
        # Only check in-bounds positions
        bounds_indices = np.where(valid_bounds)[0]
        check_pos = push_positions[bounds_indices]
        
        colors, _ = cache_manager.occupancy_cache.batch_get_attributes_unsafe(check_pos)
        # Valid if empty (color == 0)
        valid_occupancy[bounds_indices] = (colors == 0)

    # Filter valid pushes using vectorized operations
    valid_pushes_mask = valid_bounds & valid_occupancy & np.any(in_range, axis=1)

    if not np.any(valid_pushes_mask):
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Create final push pairs using vectorized concatenation
    valid_enemies = enemy_coords[valid_pushes_mask]
    valid_pushes = push_positions[valid_pushes_mask]

    return np.hstack([valid_enemies, valid_pushes])


@register(PieceType.WHITEHOLE)
def whitehole_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Registered dispatcher for White-Hole moves."""
    return generate_whitehole_moves(state.cache_manager, state.color, pos)


__all__ = ["generate_whitehole_moves", "push_candidates_vectorized"]
