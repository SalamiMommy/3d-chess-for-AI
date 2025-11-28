"""Slower piece - king-like mover with 2-sphere enemy debuff aura."""
from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.shared_types import Color, PieceType, SLOWER, COORD_DTYPE, RADIUS_2_OFFSETS
from game3d.common.registry import register
from game3d.pieces.pieces.kinglike import generate_king_moves
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Piece-specific movement vectors - same as king (26 directions)
# Converted to numpy-native using meshgrid for better performance
dx_vals, dy_vals, dz_vals = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
all_coords = np.stack([dx_vals.ravel(), dy_vals.ravel(), dz_vals.ravel()], axis=1)
# Remove the (0, 0, 0) origin
origin_mask = np.all(all_coords != 0, axis=1)
SLOWER_MOVEMENT_VECTORS = all_coords[origin_mask].astype(COORD_DTYPE)

def generate_slower_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate king-like single-step moves for slower piece."""
    return generate_king_moves(cache_manager, color, pos, piece_type=PieceType.SLOWER)

def get_debuffed_squares(
    cache_manager: 'OptimizedCacheManager',
    effect_color: int,
) -> np.ndarray:
    """
    Get squares within 2-sphere of friendly SLOWER pieces that affect enemies.
    Returns array of shape (N, 3) containing affected square coordinates.
    """
    effect_pieces = np.array([
        coord for coord, piece in cache_manager.get_pieces_of_color(effect_color)
        if piece["piece_type"] == PieceType.SLOWER
    ], dtype=COORD_DTYPE)

    if effect_pieces.shape[0] == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)

    # Vectorized processing: combine all SLOWER auras and process together
    aura_coords = effect_pieces[:, np.newaxis, :] + RADIUS_2_OFFSETS
    aura_coords = aura_coords.reshape(-1, 3)

    # Filter to bounds - vectorized
    valid_mask = in_bounds_vectorized(aura_coords)
    valid_coords = aura_coords[valid_mask]

    if valid_coords.shape[0] == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)

    # Check each valid coordinate for enemy pieces
    affected_coords = []
    for sq in valid_coords:
        target = cache_manager.get_piece(sq)
        if target is not None and target["color"] != effect_color:
            affected_coords.append(sq)

    if not affected_coords:
        return np.empty((0, 3), dtype=COORD_DTYPE)

    # Remove duplicates and return
    affected = np.array(affected_coords, dtype=COORD_DTYPE)
    return np.unique(affected, axis=0)


@register(PieceType.SLOWER)
def slower_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Dispatch slower move generation with numpy-native coordinates."""
    return generate_slower_moves(state.cache_manager, state.color, pos)


__all__ = ["generate_slower_moves", "get_debuffed_squares", "SLOWER_MOVEMENT_VECTORS"]
