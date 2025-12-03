"""Speeder – king-like mover + 1-sphere friendly buff."""

from __future__ import annotations
from typing import List, Set, TYPE_CHECKING
import numpy as np

from game3d.common.shared_types import (
    COORD_DTYPE, COLOR_DTYPE, RADIUS_1_OFFSETS,
    SPEEDER, SLOWER, Color, PieceType, Result
)
from game3d.common.registry import register
from game3d.pieces.pieces.kinglike import generate_king_moves
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState




def generate_speeder_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate king-like moves for the Speeder piece."""
    x, y, z = pos[0], pos[1], pos[2]
    return generate_king_moves(cache_manager, color, pos, piece_type=PieceType.SPEEDER)


def buffed_squares(
    cache_manager: 'OptimizedCacheManager',
    effect_color: int,
) -> Set[bytes]:
    """Get coordinates within 1-sphere of friendly Speeder pieces."""
    # Get all friendly pieces
    all_coords = cache_manager.occupancy_cache.get_positions(effect_color)
    if all_coords.size == 0:
        return set()
        
    # Filter for Speeders
    # ✅ OPTIMIZATION: Use unsafe access (coords from get_positions are valid)
    _, piece_types = cache_manager.occupancy_cache.batch_get_attributes_unsafe(all_coords)
    speeder_mask = piece_types == PieceType.SPEEDER
    speeder_coords = all_coords[speeder_mask]

    if speeder_coords.shape[0] == 0:
        return set()

    # Broadcast all Speeder positions with RADIUS_1_OFFSETS
    aura_coords = speeder_coords[:, np.newaxis, :] + RADIUS_1_OFFSETS
    aura_coords = aura_coords.reshape(-1, 3)

    # Vectorized bounds check
    in_bounds_mask = in_bounds_vectorized(aura_coords)
    valid_coords = aura_coords[in_bounds_mask]
    
    if valid_coords.shape[0] == 0:
        return set()

    # ✅ OPTIMIZATION: Vectorized color check using unsafe access
    # We only check valid_coords which are already bounds-checked
    colors, _ = cache_manager.occupancy_cache.batch_get_attributes_unsafe(valid_coords)
    
    # Filter for friendly pieces
    friendly_mask = (colors == effect_color)
    friendly_coords = valid_coords[friendly_mask]

    # Convert to set of bytes
    return {c.tobytes() for c in friendly_coords}


@register(PieceType.SPEEDER)
def speeder_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Dispatch Speeder moves from GameState."""
    return generate_speeder_moves(state.cache_manager, state.color, pos)


__all__ = ['generate_speeder_moves', 'buffed_squares']
