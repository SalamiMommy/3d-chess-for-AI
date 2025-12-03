"""King movement generator - 26-direction king/priest piece."""
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import (
    COORD_DTYPE, COLOR_DTYPE, PIECE_TYPE_DTYPE, COLOR_WHITE, COLOR_BLACK
)
from game3d.common.shared_types import Color, PieceType
from game3d.common.registry import register
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Piece-specific movement vectors - 26 directions (3x3x3 - 1)
# Converted to numpy-native using meshgrid for better performance
dx_vals, dy_vals, dz_vals = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
all_coords = np.stack([dx_vals.ravel(), dy_vals.ravel(), dz_vals.ravel()], axis=1)
# Remove the (0, 0, 0) origin
origin_mask = np.any(all_coords != 0, axis=1)
KING_MOVEMENT_VECTORS = all_coords[origin_mask].astype(COORD_DTYPE)

# Buffed King: 5x5x5 cube (Chebyshev distance 2)
dx_vals_b, dy_vals_b, dz_vals_b = np.meshgrid(
    [-2, -1, 0, 1, 2], 
    [-2, -1, 0, 1, 2], 
    [-2, -1, 0, 1, 2], 
    indexing='ij'
)
all_coords_b = np.stack([dx_vals_b.ravel(), dy_vals_b.ravel(), dz_vals_b.ravel()], axis=1)
origin_mask_b = np.any(all_coords_b != 0, axis=1)
BUFFED_KING_MOVEMENT_VECTORS = all_coords_b[origin_mask_b].astype(COORD_DTYPE)

def generate_king_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray,
    piece_type: PieceType = None
) -> np.ndarray:
    pos_arr = pos.astype(COORD_DTYPE)

    # Validate position using consolidated utils
    if pos_arr.ndim == 1:
        if not in_bounds_vectorized(pos_arr.reshape(1, 3))[0]:
            return np.empty((0, 6), dtype=COORD_DTYPE)
    # For batch input (ndim=2), we assume valid coordinates from cache/generator

    # Use jump generator with piece-specific vectors
    jump_engine = get_jump_movement_generator()
    return jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=KING_MOVEMENT_VECTORS,
        allow_capture=True,
        piece_type=piece_type,
        buffed_directions=BUFFED_KING_MOVEMENT_VECTORS
    )

# Dispatchers for king and priest (same movement pattern)
@register(PieceType.PRIEST)
def priest_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_king_moves(state.cache_manager, state.color, pos, piece_type=PieceType.PRIEST)

@register(PieceType.KING)
def king_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_king_moves(state.cache_manager, state.color, pos, piece_type=PieceType.KING)

__all__ = ['KING_MOVEMENT_VECTORS', 'generate_king_moves']
