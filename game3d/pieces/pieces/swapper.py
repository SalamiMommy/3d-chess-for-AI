# swapper.py - FULLY NUMPY-NATIVE
"""
Swapper == King-steps ∪ friendly-swap teleport
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.shared_types import Color, PieceType, Result, get_empty_coord_batch, SWAPPER
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.common.shared_types import COORD_DTYPE

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Swapper-specific movement vectors - king-like movement for walk and swap
# Converted to numpy-native using meshgrid for better performance
dx_vals, dy_vals, dz_vals = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
all_coords = np.stack([dx_vals.ravel(), dy_vals.ravel(), dz_vals.ravel()], axis=1)
# Remove the (0, 0, 0) origin
origin_mask = np.all(all_coords != 0, axis=1)
_SWAPPER_MOVEMENT_VECTORS = all_coords[origin_mask].astype(COORD_DTYPE)

def generate_swapper_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> List[Move]:
    jump_engine = get_jump_movement_generator()
    moves_list = []

    # 1. King walks
    king_moves = jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=pos.astype(COORD_DTYPE),
        directions=_SWAPPER_MOVEMENT_VECTORS,
        allow_capture=True,
        piece_type=PieceType.SWAPPER
    )
    if king_moves.size > 0:
        moves_list.append(king_moves)

    # 2. Friendly swaps
    swap_dirs = _get_friendly_swap_directions(cache_manager, color, pos)
    if swap_dirs.shape[0] > 0:
        swap_moves = jump_engine.generate_jump_moves(
            cache_manager=cache_manager,
        color=color,
            pos=pos.astype(COORD_DTYPE),
            directions=swap_dirs,
            allow_capture=False,  # Swaps don't capture
        )
        if swap_moves.size > 0:
            moves_list.append(swap_moves)

    if not moves_list:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    return np.concatenate(moves_list, axis=0)

def _get_friendly_swap_directions(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Get directions to all friendly pieces (excluding self)."""
    friendly_coords = np.array([
        coord for coord in cache_manager.occupancy_cache.get_positions(color)
        if not np.array_equal(coord, pos)
    ], dtype=COORD_DTYPE)

    if friendly_coords.shape[0] == 0:
        return get_empty_coord_batch()

    directions = (friendly_coords - pos).astype(COORD_DTYPE)
    return directions

@register(PieceType.SWAPPER)
def swapper_move_dispatcher(state: 'GameState', pos: np.ndarray) -> List[Move]:
    return generate_swapper_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_swapper_moves"]
