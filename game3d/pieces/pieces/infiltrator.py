# game3d/movement/pieces/infiltrator.py - FULLY NUMPY-NATIVE
"""
Infiltrator – king moves + teleport to squares in front of enemy pawns.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.shared_types import Color, PieceType, Result, get_empty_coord_batch
from game3d.common.shared_types import (
    COORD_DTYPE, INDEX_DTYPE, BOOL_DTYPE, COLOR_DTYPE, PIECE_TYPE_DTYPE
)
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

def generate_infiltrator_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate infiltrator moves: king walks + pawn-front teleports."""
    start = pos.astype(COORD_DTYPE)

    # Get teleport directions
    teleport_dirs = _get_pawn_front_directions(cache_manager, color, pos)

    # ✅ OPTIMIZATION: Skip np.unique - just concatenate directions
    # Duplicates are rare and jump_engine handles them efficiently via occupancy check
    if teleport_dirs.shape[0] > 0:
        all_dirs = np.vstack((teleport_dirs, _KING_DIRECTIONS))
    else:
        all_dirs = _KING_DIRECTIONS

    # Generate all moves using jump movement
    jump_engine = get_jump_movement_generator()
    moves = jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=start,
        directions=all_dirs,
        allow_capture=True,
        piece_type=PieceType.INFILTRATOR
    )

    return moves

def _get_pawn_front_directions(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Get directions to empty squares in front of enemy pawns - vectorized version."""
    start = pos.astype(COORD_DTYPE)
    enemy_color = Color(color).opposite()

    # Get all enemy piece positions from the occupancy cache
    enemy_coords = cache_manager.occupancy_cache.get_positions(enemy_color)

    if enemy_coords.shape[0] == 0:
        return get_empty_coord_batch()

    # Get piece types for enemy positions (vectorized)
    # ✅ OPTIMIZATION: Use unsafe variant - enemy_coords from get_positions are always valid
    _, piece_types = cache_manager.occupancy_cache.batch_get_attributes_unsafe(enemy_coords)

    # Filter to only pawn coordinates
    pawn_mask = piece_types == PieceType.PAWN.value
    pawn_coords = enemy_coords[pawn_mask]

    if pawn_coords.shape[0] == 0:
        return get_empty_coord_batch()

    # Front direction depends on enemy color
    dz = 1 if enemy_color == Color.BLACK else -1
    front_directions = np.array([0, 0, dz], dtype=COORD_DTYPE)

    # Calculate all front squares at once
    front_squares = pawn_coords + front_directions

    # Check bounds and emptiness in vectorized way
    valid_mask = in_bounds_vectorized(front_squares)
    valid_front_squares = front_squares[valid_mask]

    if valid_front_squares.shape[0] == 0:
        return get_empty_coord_batch()

    # Filter to empty squares using the occupancy cache
    empty_mask = ~cache_manager.occupancy_cache.batch_is_occupied_unsafe(valid_front_squares)
    empty_front_squares = valid_front_squares[empty_mask]

    if empty_front_squares.shape[0] == 0:
        return get_empty_coord_batch()

    # Calculate directions from start to targets
    directions = (empty_front_squares - start).astype(COORD_DTYPE)

    return directions

@register(PieceType.INFILTRATOR)
def infiltrator_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_infiltrator_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_infiltrator_moves"]
