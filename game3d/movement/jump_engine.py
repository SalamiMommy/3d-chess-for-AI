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


@njit(cache=True, fastmath=True, inline='always')
def _generate_jump_targets(pos: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Generate jump target coordinates with inlined bounds checking.
    
    PERFORMANCE: inline='always' eliminates function call overhead.
    Uses mask-based filtering (thread-safe) instead of parallel loop with shared counter.
    """
    targets = pos + offsets

    # Vectorized bounds filter (thread-safe, no race conditions)
    valid_mask = ((targets[:, 0] >= 0) & (targets[:, 0] < SIZE) &
                  (targets[:, 1] >= 0) & (targets[:, 1] < SIZE) &
                  (targets[:, 2] >= 0) & (targets[:, 2] < SIZE))

    valid_targets = targets[valid_mask]

    # Return empty array if no valid targets
    if valid_targets.shape[0] == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)

    return valid_targets


@njit(cache=True, fastmath=True, parallel=False)
def _filter_jump_targets_by_occupancy(
    targets: np.ndarray,
    flattened_occ: np.ndarray,
    allow_capture: bool,
    player_color: int
) -> np.ndarray:
    """Filter targets by occupancy in a single fused pass.
    
    OPTIMIZATION: Eliminates intermediate arrays (idxs, occs) and combines
    index calculation + occupancy lookup + filtering in one pass.
    Better CPU cache locality and reduced memory allocations.
    """
    n = targets.shape[0]
    if n == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)
    
    # Pre-allocate mask
    mask = np.empty(n, dtype=np.bool_)
    
    # Single pass: calculate index, lookup occupancy, apply filter
    for i in range(n):
        idx = targets[i, 0] + SIZE * targets[i, 1] + SIZE_SQUARED * targets[i, 2]
        occ = flattened_occ[idx]
        
        if allow_capture:
            mask[i] = (occ != player_color)
        else:
            mask[i] = (occ == 0)
    
    # Filter targets (numba handles this efficiently)
    return targets[mask]


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
            
            # ✅ OPTIMIZED: Direct buff check using cached array
            is_buffed = False
            aura_cache = getattr(cache_manager, 'consolidated_aura_cache', None)
            
            if aura_cache is not None:
                # Direct array access - O(1) lookup instead of method call overhead
                x, y, z = pos[0], pos[1], pos[2]
                is_buffed = aura_cache._buffed_squares[x, y, z]
            
            targets = None
            
            # Try to use precomputed moves if available and not buffed
            if not is_buffed and piece_type is not None and piece_type.value in _PRECOMPUTED_MOVES:
                # ✅ OPTIMIZATION: Use cached SIZE_SQUARED constant
                flat_idx = pos[0] + SIZE * pos[1] + SIZE_SQUARED * pos[2]
                
                # Retrieve targets from precomputed array
                try:
                    targets = _PRECOMPUTED_MOVES[piece_type.value][flat_idx]
                except IndexError:
                    # Fallback if index out of bounds (shouldn't happen with valid pos)
                    pass

            # Fallback to vector calculation if no precomputed moves or buffed
            if targets is None:
                current_directions = directions
                if is_buffed:
                    # Apply buff: increase length of longest directional vector(s) by 1
                    current_directions = directions.copy()
                    
                    # Vectorized application of the rule
                    abs_dirs = np.abs(current_directions)
                    max_vals = np.max(abs_dirs, axis=1, keepdims=True)
                    
                    # Identify components that are equal to the max value (longest components)
                    mask = (abs_dirs == max_vals)
                    sign = np.sign(current_directions)
                    
                    # Add sign where mask is True
                    current_directions = current_directions + (sign * mask).astype(COORD_DTYPE)

                targets = _generate_jump_targets(pos, current_directions)
                
                # Early exit if no valid targets
                if targets.shape[0] == 0:
                    return np.empty((0, 6), dtype=COORD_DTYPE)

            # ✅ OPTIMIZATION: Use optimized batch_is_occupied_unsafe
            # This leverages the adaptive dispatching in OccupancyCache
            is_occupied = cache_manager.occupancy_cache.batch_is_occupied_unsafe(targets)
            
            if allow_capture:
                # Valid if: NOT occupied OR (occupied AND color != self.color)
                # But we need to check color if occupied.
                # Actually, filtering by occupancy first is faster if we expect many empty squares.
                
                # Let's use the fused logic from before but via cache API if possible?
                # No, batch_is_occupied_unsafe only returns boolean.
                # If we want capture logic, we need colors for occupied squares.
                
                # Optimized approach:
                # 1. Identify occupied squares
                # 2. For occupied, check color
                
                # However, the previous _filter_jump_targets_by_occupancy was doing it all in one pass.
                # Can we do better?
                # batch_get_attributes_unsafe returns (colors, types).
                
                # If we use batch_get_colors_only (which we should add/use), we can do:
                # colors = cache.batch_get_colors_only(targets)
                # valid = (colors == 0) | (colors != color)
                
                # Let's check if batch_get_colors_only exists and is optimized.
                # It exists in OccupancyCache.
                
                # Use batch_get_colors_only which is optimized
                target_colors = cache_manager.occupancy_cache.batch_get_colors_only(targets)
                
                # Valid move if:
                # 1. Empty (color == 0)
                # 2. Enemy (color != self.color)
                # Combined: target_colors != self.color (since self.color is never 0)
                valid_mask = (target_colors != color)
                
            else:
                # Only empty squares allowed
                # batch_is_occupied_unsafe is perfect here
                is_occupied = cache_manager.occupancy_cache.batch_is_occupied_unsafe(targets)
                valid_mask = ~is_occupied

            valid_targets = targets[valid_mask]
            
            n_moves = valid_targets.shape[0]
            if n_moves == 0:
                return np.empty((0, 6), dtype=COORD_DTYPE)

            # ✅ OPTIMIZATION: Construct move array more efficiently
            moves = np.empty((n_moves, 6), dtype=COORD_DTYPE)
            moves[:, :3] = pos  # Broadcasts pos to all rows
            moves[:, 3:6] = valid_targets
            
            return moves



def get_jump_movement_generator() -> JumpMovementEngine:
    """Backwards compatibility alias for JumpMovementEngine constructor."""
    return JumpMovementEngine()

# Update the __all__ export list
__all__ = ['JumpMovementEngine', 'get_jump_movement_generator']
