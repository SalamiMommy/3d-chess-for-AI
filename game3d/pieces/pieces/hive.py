"""Hive piece movement generator with optimized 3D king-like single steps."""

from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.shared_types import (
    COORD_DTYPE, COLOR_DTYPE, PIECE_TYPE_DTYPE,
    HIVE, Color, PieceType, Result
)
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

# Precomputed hive movement vectors - all 26 3D king-like directions
# Converted to numpy-native using meshgrid for better performance
dx_vals, dy_vals, dz_vals = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
all_coords = np.stack([dx_vals.ravel(), dy_vals.ravel(), dz_vals.ravel()], axis=1)
# Remove the (0, 0, 0) origin
origin_mask = np.all(all_coords != 0, axis=1)
HIVE_DIRECTIONS_3D = all_coords[origin_mask].astype(COORD_DTYPE)

def generate_hive_moves(
    cache_manager: 'OptimizedCacheManager',
    color: COLOR_DTYPE,
    pos: np.ndarray,
) -> np.ndarray:
    engine = get_jump_movement_generator(cache_manager)
    return engine.generate_jump_moves(
        color=color,
        pos=pos.astype(COORD_DTYPE),
        directions=HIVE_DIRECTIONS_3D,
        allow_capture=True
    )

@register(PieceType.HIVE)
def hive_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_hive_moves(state.cache_manager, state.color, pos)

def get_movable_hives(state: 'GameState', color: COLOR_DTYPE) -> List[Move]:
    from game3d.movement.generator import generate_legal_moves_for_piece

    hives: List[Move] = []
    for coord, piece in state.cache_manager.get_pieces_of_color(color):
        if piece["piece_type"] == HIVE:
            hives.extend(generate_legal_moves_for_piece(state, coord))
    return hives

def apply_multi_hive_move(state: 'GameState', move: Move) -> 'GameState':
    """Apply hive move without flipping turn - enables multiple hive moves per turn."""
    new_state = state.make_move(move)
    # Preserve current player's turn for additional hive moves
    object.__setattr__(new_state, "color", state.color)
    new_state._clear_caches()
    return new_state



__all__ = [
    "generate_hive_moves",
    "get_movable_hives",
    "apply_multi_hive_move",
    "HIVE_DIRECTIONS_3D"
]
