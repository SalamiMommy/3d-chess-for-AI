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
    # Filter targets (numba handles this efficiently)
    return targets[mask]


@njit(cache=True, fastmath=True)
def _generate_and_filter_jump_moves(
    pos: np.ndarray,
    directions: np.ndarray,
    flattened_occ: np.ndarray,
    allow_capture: bool,
    player_color: int
) -> np.ndarray:
    """Generate and filter jump moves in a single pass.
    
    Fuses:
    1. Target calculation (pos + direction)
    2. Bounds checking
    3. Occupancy/Capture filtering
    
    Avoids allocating intermediate 'targets' and 'mask' arrays.
    """
    n_dirs = directions.shape[0]
    # Allocate max possible size
    moves = np.empty((n_dirs, 6), dtype=COORD_DTYPE)
    count = 0
    
    px, py, pz = pos[0], pos[1], pos[2]
    
    for i in range(n_dirs):
        dx, dy, dz = directions[i]
        
        # Target
        tx, ty, tz = px + dx, py + dy, pz + dz
        
        # Bounds check
        if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
            # Occupancy check
            idx = tx + SIZE * ty + SIZE_SQUARED * tz
            occ = flattened_occ[idx]
            
            is_valid = False
            if occ == 0:
                is_valid = True
            elif allow_capture:
                if occ != player_color:
                    is_valid = True
            
            if is_valid:
                moves[count, 0] = px
                moves[count, 1] = py
                moves[count, 2] = pz
                moves[count, 3] = tx
                moves[count, 4] = ty
                moves[count, 5] = tz
                count += 1
                
    return moves[:count]


@njit(cache=True, fastmath=True)
def _generate_and_filter_jump_moves_batch(
    positions: np.ndarray,
    directions: np.ndarray,
    flattened_occ: np.ndarray,
    allow_capture: bool,
    player_color: int
) -> np.ndarray:
    """Generate and filter jump moves for a BATCH of positions.
    
    Fuses:
    1. Target calculation (pos + direction) for all positions
    2. Bounds checking
    3. Occupancy/Capture filtering
    
    Returns:
        (K, 6) array of moves [fx, fy, fz, tx, ty, tz]
    """
    n_pos = positions.shape[0]
    n_dirs = directions.shape[0]
    max_moves = n_pos * n_dirs
    
    moves = np.empty((max_moves, 6), dtype=COORD_DTYPE)
    count = 0
    
    for i in range(n_pos):
        px, py, pz = positions[i]
        
        for j in range(n_dirs):
            dx, dy, dz = directions[j]
            
            # Target
            tx, ty, tz = px + dx, py + dy, pz + dz
            
            # Bounds check
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                # Occupancy check
                idx = tx + SIZE * ty + SIZE_SQUARED * tz
                occ = flattened_occ[idx]
                
                is_valid = False
                if occ == 0:
                    is_valid = True
                elif allow_capture:
                    if occ != player_color:
                        is_valid = True
                
                if is_valid:
                    moves[count, 0] = px
                    moves[count, 1] = py
                    moves[count, 2] = pz
                    moves[count, 3] = tx
                    moves[count, 4] = ty
                    moves[count, 5] = tz
                    count += 1
                    
    return moves[:count]


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
            
            # Filter out zero-direction vectors unless explicitly allowed
            if not allow_zero_direction:
                zero_mask = ~((directions[:, 0] == 0) & (directions[:, 1] == 0) & (directions[:, 2] == 0))
                directions = directions[zero_mask]
                
                if directions.shape[0] == 0:
                    return np.empty((0, 6), dtype=COORD_DTYPE)

            # ✅ BATCH PROCESSING PATH
            if pos.ndim == 2:
                if pos.shape[0] == 0:
                    return np.empty((0, 6), dtype=COORD_DTYPE)
                
                # Check buffs
                aura_cache = getattr(cache_manager, 'consolidated_aura_cache', None)
                flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
                
                if aura_cache is not None:
                    # Check if ANY piece in batch is buffed
                    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
                    buff_mask = aura_cache._buffed_squares[x, y, z]
                    
                    if np.any(buff_mask):
                        moves_list = []
                        
                        # 1. Process unbuffed pieces
                        unbuffed_pos = pos[~buff_mask]
                        if unbuffed_pos.shape[0] > 0:
                            moves_list.append(_generate_and_filter_jump_moves_batch(
                                unbuffed_pos, directions, flattened_occ, allow_capture, color
                            ))
                            
                        # 2. Process buffed pieces
                        buffed_pos = pos[buff_mask]
                        if buffed_pos.shape[0] > 0:
                            # Calculate buffed directions (increase length of longest component)
                            current_directions = directions.copy()
                            abs_dirs = np.abs(current_directions)
                            max_vals = np.max(abs_dirs, axis=1, keepdims=True)
                            mask = (abs_dirs == max_vals)
                            sign = np.sign(current_directions)
                            buffed_directions = current_directions + (sign * mask).astype(COORD_DTYPE)
                            
                            moves_list.append(_generate_and_filter_jump_moves_batch(
                                buffed_pos, buffed_directions, flattened_occ, allow_capture, color
                            ))
                            
                        return np.concatenate(moves_list) if moves_list else np.empty((0, 6), dtype=COORD_DTYPE)

                # Fast path: No buffs, use batch kernel
                # print(f"DEBUG: Calling batch kernel with pos shape {pos.shape}")
                return _generate_and_filter_jump_moves_batch(
                    pos, directions, flattened_occ, allow_capture, color
                )

            # ✅ SINGLE PIECE PATH (Legacy/Single)
            # ---------------------------------------------------------
            
            # Direct buff check
            is_buffed = False
            aura_cache = getattr(cache_manager, 'consolidated_aura_cache', None)
            
            if aura_cache is not None:
                x, y, z = pos[0], pos[1], pos[2]
                is_buffed = aura_cache._buffed_squares[x, y, z]
            
            targets = None
            
            # Try to use precomputed moves if available and not buffed
            if not is_buffed and piece_type is not None and piece_type.value in _PRECOMPUTED_MOVES:
                flat_idx = pos[0] + SIZE * pos[1] + SIZE_SQUARED * pos[2]
                
                try:
                    targets = _PRECOMPUTED_MOVES[piece_type.value][flat_idx]
                except IndexError:
                    pass

            # Fallback to vector calculation if no precomputed moves or buffed
            if targets is None:
                current_directions = directions
                if is_buffed:
                    current_directions = directions.copy()
                    abs_dirs = np.abs(current_directions)
                    max_vals = np.max(abs_dirs, axis=1, keepdims=True)
                    mask = (abs_dirs == max_vals)
                    sign = np.sign(current_directions)
                    current_directions = current_directions + (sign * mask).astype(COORD_DTYPE)

                flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
                return _generate_and_filter_jump_moves(
                    pos, current_directions, flattened_occ, allow_capture, color
                )

            # Use optimized batch_is_occupied_unsafe for precomputed targets
            flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
            
            valid_targets = _filter_jump_targets_by_occupancy(
                targets, flattened_occ, allow_capture, color
            )
            
            n_moves = valid_targets.shape[0]
            if n_moves == 0:
                return np.empty((0, 6), dtype=COORD_DTYPE)

            moves = np.empty((n_moves, 6), dtype=COORD_DTYPE)
            moves[:, :3] = pos  # Broadcasts pos to all rows
            moves[:, 3:6] = valid_targets
            
            return moves



def get_jump_movement_generator() -> JumpMovementEngine:
    """Backwards compatibility alias for JumpMovementEngine constructor."""
    return JumpMovementEngine()

# Update the __all__ export list
__all__ = ['JumpMovementEngine', 'get_jump_movement_generator']
