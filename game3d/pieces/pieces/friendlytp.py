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

    # ✅ OPTIMIZATION: Skip np.unique - just concatenate directions
    # Duplicates are rare and jump_engine handles them efficiently via occupancy check
    if teleport_dirs.shape[0] > 0:
        all_dirs = np.vstack((_KING_DIRECTIONS, teleport_dirs))
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
    # Shape: (N_friendly, 26, 3) -> (N_friendly * 26, 3)
    neighbours = (friendly_arr[:, np.newaxis, :] + _KING_DIRECTIONS[np.newaxis, :, :]).reshape(-1, 3)

    # Filter: in-bounds
    # ✅ OPTIMIZATION: Use optimized in_bounds
    valid_mask = in_bounds_vectorized(neighbours)
    neighbours = neighbours[valid_mask]

    if neighbours.shape[0] == 0:
        return get_empty_coord_batch()

    # ✅ OPTIMIZED: Use batch_is_occupied_unsafe for faster boolean check
    # We only need to know if it's empty (color == 0), so is_occupied check is sufficient
    # batch_is_occupied returns True if occupied, False if empty
    is_occupied = cache_manager.occupancy_cache.batch_is_occupied_unsafe(neighbours)
    empty_neighbours = neighbours[~is_occupied]

    if empty_neighbours.shape[0] == 0:
        return get_empty_coord_batch()

    # ✅ OPTIMIZATION: Fast deduplication using boolean mask on 1D indices
    # Convert to flat indices for O(1) deduplication
    flat_indices = (empty_neighbours[:, 0] + SIZE * empty_neighbours[:, 1] + SIZE * SIZE * empty_neighbours[:, 2])
    
    # Use a boolean mask for deduplication (since max index is small: 9*9*9 = 729)
    # This is much faster than np.unique or set() for dense coordinates
    unique_mask = np.zeros(SIZE * SIZE * SIZE, dtype=np.bool_)
    unique_mask[flat_indices] = True
    
    # Also mask out the start position to avoid self-teleport
    start_idx = start[0] + SIZE * start[1] + SIZE * SIZE * start[2]
    unique_mask[start_idx] = False
    
    # Get unique flat indices
    unique_flat = np.flatnonzero(unique_mask)
    
    if unique_flat.shape[0] == 0:
        return get_empty_coord_batch()
        
    # Convert back to coordinates
    # z = idx // SIZE_SQUARED, y = (idx % SIZE_SQUARED) // SIZE, x = idx % SIZE
    z = unique_flat // (SIZE * SIZE)
    rem = unique_flat % (SIZE * SIZE)
    y = rem // SIZE
    x = rem % SIZE
    
    unique_targets = np.column_stack((x, y, z)).astype(COORD_DTYPE)

    # Convert to directions
    directions = (unique_targets - start).astype(COORD_DTYPE)

    # Final validation (should be redundant if logic is correct, but safe to keep for now)
    # dest_coords = start + directions
    # valid_mask = np.all((dest_coords >= 0) & (dest_coords < SIZE), axis=1)
    # return directions[valid_mask]
    
    return directions

@register(PieceType.FRIENDLYTELEPORTER)
def friendlytp_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_friendlytp_moves(state.cache_manager, state.color, pos)

__all__ = ["_KING_DIRECTIONS", "generate_friendlytp_moves"]
