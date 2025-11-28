"""Jump Movement Engine - RAW MOVE GENERATION ONLY.

NO validation here. Just generates candidate moves.
Validation happens in generator.py.
"""

import numpy as np
import os
from numba import njit, prange
from typing import TYPE_CHECKING, Optional, Dict, Any

from game3d.common.shared_types import (
    COORD_DTYPE, BOOL_DTYPE, COLOR_DTYPE, SIZE, MAX_COORD_VALUE,
    PieceType, compute_board_index, SIZE_SQUARED
)
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.cache.manager import UnifiedCacheManager

# Module-level cache for precomputed moves
_PRECOMPUTED_MOVES: Dict[int, np.ndarray] = {}
_PRECOMPUTED_LOADED = False

def _load_precomputed_moves():
    """Load precomputed move tables from disk."""
    global _PRECOMPUTED_LOADED
    if _PRECOMPUTED_LOADED:
        return

    # Path to precomputed directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    precomputed_dir = os.path.join(current_dir, "precomputed")
    
    if not os.path.exists(precomputed_dir):
        # Silent fail or log warning - we don't want to crash if files are missing
        # print(f"Warning: Precomputed directory not found at {precomputed_dir}")
        _PRECOMPUTED_LOADED = True
        return

    # Map PieceType name to file suffix
    # We iterate through PieceType members
    for piece_type in PieceType:
        name = piece_type.name
        filename = f"moves_{name}.npy"
        path = os.path.join(precomputed_dir, filename)
        
        if os.path.exists(path):
            try:
                # Load the object array (jagged array of moves)
                moves_array = np.load(path, allow_pickle=True)
                _PRECOMPUTED_MOVES[piece_type.value] = moves_array
            except Exception as e:
                print(f"Failed to load precomputed moves for {name}: {e}")
    
    _PRECOMPUTED_LOADED = True


@njit(cache=True, fastmath=True, parallel=True)
def _generate_jump_targets(pos: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Generate jump target coordinates (no validation)."""
    targets = pos + offsets

    # Simple bounds filter
    valid_mask = ((targets[:, 0] >= 0) & (targets[:, 0] < SIZE) &
                  (targets[:, 1] >= 0) & (targets[:, 1] < SIZE) &
                  (targets[:, 2] >= 0) & (targets[:, 2] < SIZE))

    valid_targets = targets[valid_mask]

    # CRITICAL: Ensure we always return a valid array (even if empty)
    if valid_targets.shape[0] == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)

    return valid_targets


class JumpMovementEngine:
    """
    Raw jump move generation (Knight, King, etc.).

    NO validation - just generates candidate moves.
    Generator will validate them.
    """

    def __init__(self):
        # Ensure precomputed moves are loaded
        global _PRECOMPUTED_LOADED
        if not _PRECOMPUTED_LOADED:
            _load_precomputed_moves()
            _PRECOMPUTED_LOADED = True


    def generate_jump_moves(
            self,
            cache_manager: 'UnifiedCacheManager',
            color: COLOR_DTYPE,
            pos: np.ndarray,
            directions: np.ndarray,          # (N,3) jump offsets
            allow_capture: bool = True,
            allow_zero_direction: bool = False,  # Allow (0,0,0) self-targeting moves
            piece_type: Optional[PieceType] = None # Optional piece type for precomputed lookup
        ) -> np.ndarray:
            """Generate jump moves as numpy array [from_x, from_y, from_z, to_x, to_y, to_z]."""
            
            # Filter out zero-direction vectors unless explicitly allowed (e.g., for BOMB self-detonation)
            if not allow_zero_direction:
                zero_mask = ~((directions[:, 0] == 0) & (directions[:, 1] == 0) & (directions[:, 2] == 0))
                directions = directions[zero_mask]
                
                if directions.shape[0] == 0:
                    return np.empty((0, 6), dtype=COORD_DTYPE)
            
            # Check for Speeder buff
            # We need to find the ConsolidatedAuraCache to check for buffs
            # This is done dynamically to avoid circular imports
            is_buffed = False
            if hasattr(cache_manager, '_effect_cache_instances'):
                for cache in cache_manager._effect_cache_instances:
                    if cache.__class__.__name__ == 'ConsolidatedAuraCache':
                        # Check if this specific piece is buffed
                        # batch_is_buffed expects (N, 3)
                        pos_reshaped = pos.reshape(1, 3)
                        buff_status = cache.batch_is_buffed(pos_reshaped, color)
                        if buff_status.size > 0 and buff_status[0]:
                            is_buffed = True
                        break
            
            targets = None
            
            # Try to use precomputed moves if available and not buffed
            if not is_buffed and piece_type is not None and piece_type in _PRECOMPUTED_MOVES:
                # Calculate flat index
                # pos is (3,) array
                flat_idx = pos[0] + SIZE * pos[1] + SIZE_SQUARED * pos[2]
                
                # Retrieve targets from precomputed array
                # _PRECOMPUTED_MOVES[piece_type] is an object array of arrays
                try:
                    targets = _PRECOMPUTED_MOVES[piece_type][flat_idx]
                except IndexError:
                    # Fallback if index out of bounds (shouldn't happen with valid pos)
                    pass

            # Fallback to vector calculation if no precomputed moves or buffed
            if targets is None:
                current_directions = directions
                if is_buffed:
                    # Apply buff: increase length of longest directional vector(s) by 1
                    # Create a copy to avoid modifying the original static array
                    current_directions = directions.copy()
                    
                    # Vectorized application of the rule
                    abs_dirs = np.abs(current_directions)
                    max_vals = np.max(abs_dirs, axis=1, keepdims=True)
                    
                    # Identify components that are equal to the max value (longest components)
                    # and add 1 in the direction of the sign
                    mask = (abs_dirs == max_vals)
                    sign = np.sign(current_directions)
                    
                    # Add sign to the components where mask is True
                    # We use += to modify in place
                    # If sign is 0 (shouldn't happen for jump vectors usually, but good to be safe), it adds 0
                    current_directions = current_directions + (sign * mask).astype(COORD_DTYPE)

                targets = _generate_jump_targets(pos, current_directions)

            if targets.shape[0] == 0:
                return np.empty((0, 6), dtype=COORD_DTYPE)

            flattened = cache_manager.occupancy_cache.get_flattened_occupancy()
            idxs = targets[:, 0] + SIZE * targets[:, 1] + SIZE * SIZE * targets[:, 2]
            occs = flattened[idxs]

            if allow_capture:
                valid_mask = (occs != color)
            else:
                valid_mask = (occs == 0)

            if not np.any(valid_mask):
                return np.empty((0, 6), dtype=COORD_DTYPE)

            valid_targets = targets[valid_mask]
            
            if valid_targets.shape[0] == 0:
                return np.empty((0, 6), dtype=COORD_DTYPE)

            # Construct (N, 6) array
            n_moves = valid_targets.shape[0]
            moves = np.empty((n_moves, 6), dtype=COORD_DTYPE)
            
            # Fill from_coord
            moves[:, 0] = pos[0]
            moves[:, 1] = pos[1]
            moves[:, 2] = pos[2]
            
            # Fill to_coord
            moves[:, 3:6] = valid_targets
            
            return moves



def get_jump_movement_generator() -> JumpMovementEngine:
    """Backwards compatibility alias for JumpMovementEngine constructor."""
    return JumpMovementEngine()

# Update the __all__ export list
__all__ = ['JumpMovementEngine', 'get_jump_movement_generator']
