"""Freezer piece - king-like movement with freeze aura."""
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import (
    COORD_DTYPE, RADIUS_2_OFFSETS, Color, PieceType
)
from game3d.common.registry import register
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Piece-specific movement vectors - king-like movement
# Converted to numpy-native using meshgrid for better performance
dx_vals, dy_vals, dz_vals = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
all_coords = np.stack([dx_vals.ravel(), dy_vals.ravel(), dz_vals.ravel()], axis=1)
# Remove the (0, 0, 0) origin
origin_mask = np.all(all_coords != 0, axis=1)
FREEZER_MOVEMENT_VECTORS = all_coords[origin_mask].astype(COORD_DTYPE)

def generate_freezer_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    pos_arr = pos.astype(COORD_DTYPE)

    # Use jump generator with piece-specific vectors
    jump_engine = get_jump_movement_generator()
    return jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=FREEZER_MOVEMENT_VECTORS,
        allow_capture=True,
        piece_type=PieceType.FREEZER
    )

def get_all_frozen_squares_numpy(
    cache_manager: 'OptimizedCacheManager',
    controller: Color,
) -> np.ndarray:
    """Get all enemy squares frozen by controller's freezers. Returns (N, 3) array."""
    if isinstance(controller, np.ndarray):
        controller = Color(int(controller.item()))
    enemy_color = controller.opposite()

    # Get all friendly pieces
    all_coords = cache_manager.occupancy_cache.get_positions(controller)
    if all_coords.size == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)
    
    # Filter for freezers
    _, piece_types = cache_manager.occupancy_cache.batch_get_attributes(all_coords)
    freezer_mask = piece_types == PieceType.FREEZER
    freezers = all_coords[freezer_mask]

    if freezers.shape[0] == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)

    # Calculate all potential freeze squares at once using precomputed offsets
    freeze_offsets = RADIUS_2_OFFSETS  # 2-sphere freeze radius
    expanded_freezers = freezers[:, np.newaxis]  # Shape: (n_freezers, 1, 3)
    expanded_offsets = freeze_offsets[np.newaxis, :, :]  # Shape: (1, n_offsets, 3)

    # Broadcast and calculate all freeze positions
    freeze_candidates = expanded_freezers + expanded_offsets  # Shape: (n_freezers, n_offsets, 3)
    freeze_candidates = freeze_candidates.reshape(-1, 3)  # Shape: (n_freezers * n_offsets, 3)

    # Filter in-bounds coordinates
    valid_mask = in_bounds_vectorized(freeze_candidates)
    bounded_targets = freeze_candidates[valid_mask]

    if bounded_targets.shape[0] == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)

    # Check for enemy pieces at freeze positions
    frozen_coords = []
    for coord in bounded_targets:
        target = cache_manager.occupancy_cache.get(coord)
        if target is not None and target["color"] == enemy_color:
            frozen_coords.append(coord)

    if not frozen_coords:
        return np.empty((0, 3), dtype=COORD_DTYPE)

    # Remove duplicates
    frozen = np.array(frozen_coords, dtype=COORD_DTYPE)
    return np.unique(frozen, axis=0)

@register(PieceType.FREEZER)
def freezer_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Generate legal freezer moves from position."""
    return generate_freezer_moves(state.cache_manager, state.color, pos)

__all__ = ["FREEZER_MOVEMENT_VECTORS", "generate_freezer_moves", "get_all_frozen_squares_numpy"]
