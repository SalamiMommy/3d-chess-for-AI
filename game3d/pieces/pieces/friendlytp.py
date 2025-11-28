# game3d/movement/pieces/friendlytp.py - FULLY NUMPY-NATIVE
"""
Friendly-Teleporter – teleport to any empty neighbour of a friendly piece
PLUS normal 1-step King moves.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.shared_types import Color, PieceType, Result, get_empty_coord_batch
from game3d.common.shared_types import COORD_DTYPE, SIZE
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# King directions (1-step moves) - converted to numpy-native using meshgrid
dx_vals, dy_vals, dz_vals = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
all_coords = np.stack([dx_vals.ravel(), dy_vals.ravel(), dz_vals.ravel()], axis=1)
# Remove the (0, 0, 0) origin
origin_mask = np.all(all_coords != 0, axis=1)
_KING_DIRECTIONS = all_coords[origin_mask].astype(COORD_DTYPE)

def generate_friendlytp_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate friendly teleporter moves: king walks + network teleports."""
    start = pos.astype(COORD_DTYPE)

    # Get teleport directions
    teleport_dirs = _build_network_directions(cache_manager, color, pos)

    # Combine all directions
    if teleport_dirs.shape[0] > 0:
        all_dirs = np.unique(np.vstack((_KING_DIRECTIONS, teleport_dirs)), axis=0)
    else:
        all_dirs = _KING_DIRECTIONS

    # Generate all moves
    jump_engine = get_jump_movement_generator()
    moves = jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=start,
        directions=all_dirs,
        allow_capture=True,
        piece_type=PieceType.FRIENDLYTELEPORTER
    )

    # Note: Metadata annotation removed as it was already being lost when
    # generator.py converted Move objects to arrays. The teleport functionality
    # works without the metadata flag.
    return moves

def _build_network_directions(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Build directions to empty squares adjacent to friendly pieces."""
    start = pos.astype(COORD_DTYPE)

    # ✅ FIXED: Use occupancy_cache.get_positions() instead of get_pieces_of_color()
    friendly_coords = cache_manager.occupancy_cache.get_positions(color)

    # Filter out the start position
    if friendly_coords.shape[0] > 0:
        not_start_mask = ~np.all(friendly_coords == start, axis=1)
        friendly_arr = friendly_coords[not_start_mask]
    else:
        friendly_arr = friendly_coords

    if friendly_arr.shape[0] == 0:
        return get_empty_coord_batch()

    # Get all neighbors of friendly pieces
    neighbours = (friendly_arr[:, np.newaxis, :] + _KING_DIRECTIONS[np.newaxis, :, :]).reshape(-1, 3)

    # Filter: in-bounds and empty
    valid_mask = in_bounds_vectorized(neighbours)
    neighbours = neighbours[valid_mask]

    if neighbours.shape[0] == 0:
        return get_empty_coord_batch()

    # ✅ OPTIMIZED: Use batch_get_colors_only instead of loop-based get()
    # Old: [cache_manager.occupancy_cache.get(sq) is None for sq in neighbours]
    # New: Direct batch array operation
    colors = cache_manager.occupancy_cache.batch_get_colors_only(neighbours)
    empty_mask = (colors == 0)
    empty_neighbours = neighbours[empty_mask]

    if empty_neighbours.shape[0] == 0:
        return get_empty_coord_batch()

    # Remove duplicates and start position
    unique_targets = np.unique(empty_neighbours, axis=0)
    start_mask = np.all(unique_targets == start, axis=1)
    unique_targets = unique_targets[~start_mask]

    # Convert to directions
    directions = (unique_targets - start).astype(COORD_DTYPE)

    # Final validation
    dest_coords = start + directions
    valid_mask = np.all((dest_coords >= 0) & (dest_coords < SIZE), axis=1)
    return directions[valid_mask]

@register(PieceType.FRIENDLYTELEPORTER)
def friendlytp_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_friendlytp_moves(state.cache_manager, state.color, pos)

__all__ = ["_KING_DIRECTIONS", "generate_friendlytp_moves"]
