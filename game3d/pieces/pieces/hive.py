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
    engine = get_jump_movement_generator()
    return engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=pos.astype(COORD_DTYPE),
        directions=HIVE_DIRECTIONS_3D,
        allow_capture=True
    )

@register(PieceType.HIVE)
def hive_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_hive_moves(state.cache_manager, state.color, pos)

def get_movable_hives(state: 'GameState', color: COLOR_DTYPE, exclude_positions: set = None) -> np.ndarray:
    from game3d.movement.generator import generate_legal_moves_for_piece

    movable_hives: List[np.ndarray] = []
    exclude_positions = exclude_positions or set()

    # Get all pieces of this color
    coords = state.cache_manager.occupancy_cache.get_positions(color)
    if coords.size == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)

    # Get their attributes (colors and piece types)
    colors, piece_types = state.cache_manager.occupancy_cache.batch_get_attributes(coords)

    # Filter for HIVE pieces
    for i in range(len(coords)):
        if piece_types[i] == HIVE:
            # Skip if this hive has already moved this turn
            coord = coords[i]
            pos_tuple = tuple(coord.tolist())
            if pos_tuple not in exclude_positions:
                # Generator now handles shape normalization
                moves = generate_legal_moves_for_piece(state, coord)
                if moves.size > 0:
                    movable_hives.append(coord)

    # Return array of unique hive coordinates
    return np.stack(movable_hives, axis=0) if movable_hives else np.empty((0, 3), dtype=COORD_DTYPE)

def apply_multi_hive_move(state: 'GameState', move: Move) -> 'GameState':
    """Apply hive move without flipping turn - enables multiple hive moves per turn."""
    # Track this hive as having moved
    from_pos_tuple = tuple(move.from_coord.tolist())
    state._moved_hive_positions.add(from_pos_tuple)
    state._pending_hive_moves.append(move)
    
    # Convert Move object to array format [from_x, from_y, from_z, to_x, to_y, to_z]
    move_array = np.concatenate([move.from_coord, move.to_coord])
    new_state = state.make_move_vectorized(move_array)
    
    # Preserve current player's turn for additional hive moves
    object.__setattr__(new_state, "color", state.color)
    
    # ✅ CRITICAL FIX: Reset turn counters to prevent double increment
    # The turn number should only increment when the turn actually switches (in game3d.py)
    object.__setattr__(new_state, "turn_number", state.turn_number)
    object.__setattr__(new_state, "halfmove_clock", state.halfmove_clock)
    
    # Carry over the hive tracking to the new state
    new_state._moved_hive_positions = state._moved_hive_positions.copy()
    new_state._pending_hive_moves = state._pending_hive_moves.copy()
    
    new_state._clear_caches()
    return new_state



__all__ = [
    "generate_hive_moves",
    "get_movable_hives",
    "apply_multi_hive_move",
    "HIVE_DIRECTIONS_3D"
]
