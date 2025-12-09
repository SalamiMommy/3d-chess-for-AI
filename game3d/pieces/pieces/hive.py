"""Hive piece movement generator with optimized 3D king-like single steps."""

from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.shared_types import *
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING: pass
from game3d.movement.generator import generate_legal_moves_for_piece

from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS, BUFFED_KING_MOVEMENT_VECTORS

def get_movable_hives(state: 'GameState', color: COLOR_DTYPE, exclude_positions: set = None) -> np.ndarray:

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
    # Capture current state before move to preserve turn
    original_color = state.color
    
    # Convert Move object to array format [from_x, from_y, from_z, to_x, to_y, to_z]
    move_array = np.concatenate([move.from_coord, move.to_coord])
    
    # Execute the move (this validates the move, so we must NOT have added to _moved_hive_positions yet)
    new_state = state.make_move_vectorized(move_array)
    
    # Track this hive as having moved AFTER successful execution
    # We track the DESTINATION because that is where the hive is now located
    to_pos_tuple = tuple(move.to_coord.tolist())
    new_state._moved_hive_positions = state._moved_hive_positions.copy()
    new_state._moved_hive_positions.add(to_pos_tuple)
    new_state._pending_hive_moves = state._pending_hive_moves.copy()
    new_state._pending_hive_moves.append(move)
    
    # Preserve current player's turn for additional hive moves
    # We must explicitly set it back to the original color because make_move flips it
    object.__setattr__(new_state, "color", original_color)
    
    # ✅ CRITICAL FIX: Reset turn counters to prevent double increment
    # The turn number should only increment when the turn actually switches (in game3d.py)
    object.__setattr__(new_state, "turn_number", state.turn_number)
    object.__setattr__(new_state, "halfmove_clock", state.halfmove_clock)
    
    new_state._clear_caches()
    return new_state

__all__ = []

