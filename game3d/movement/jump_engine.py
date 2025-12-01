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





@njit(cache=True, fastmath=True, parallel=False)
def _filter_jump_targets_by_occupancy(
    targets: np.ndarray,
    occ: np.ndarray,
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
        tx, ty, tz = targets[i, 0], targets[i, 1], targets[i, 2]
        occ_val = occ[tx, ty, tz]
        
        if allow_capture:
            mask[i] = (occ_val != player_color)
        else:
            mask[i] = (occ_val == 0)
    
    # Filter targets (numba handles this efficiently)
    return targets[mask]


@njit(cache=True, fastmath=True)
def _generate_and_filter_jump_moves(
    pos: np.ndarray,
    directions: np.ndarray,
    occ: np.ndarray,
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
            occ_val = occ[tx, ty, tz]
            
            is_valid = False
            if occ_val == 0:
                is_valid = True
            elif allow_capture:
                if occ_val != player_color:
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


@njit(cache=True, fastmath=True, parallel=True)
def _generate_and_filter_jump_moves_batch(
    positions: np.ndarray,
    directions: np.ndarray,
    occ: np.ndarray,
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
    
    # Pass 1: Count
    counts = np.zeros(n_pos, dtype=np.int32)
    
    for i in prange(n_pos):
        px, py, pz = positions[i]
        count = 0
        
        for j in range(n_dirs):
            dx, dy, dz = directions[j]
            tx, ty, tz = px + dx, py + dy, pz + dz
            
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                occ_val = occ[tx, ty, tz]
                is_valid = False
                if occ_val == 0:
                    is_valid = True
                elif allow_capture:
                    if occ_val != player_color:
                        is_valid = True
                
                if is_valid:
                    count += 1
        counts[i] = count
        
    # Pass 2: Offsets
    total_moves = np.sum(counts)
    offsets = np.zeros(n_pos, dtype=np.int32)
    current_offset = 0
    for i in range(n_pos):
        offsets[i] = current_offset
        current_offset += counts[i]
        
    # Pass 3: Fill
    moves = np.empty((total_moves, 6), dtype=COORD_DTYPE)
    
    for i in prange(n_pos):
        write_idx = offsets[i]
        px, py, pz = positions[i]
        
        for j in range(n_dirs):
            dx, dy, dz = directions[j]
            tx, ty, tz = px + dx, py + dy, pz + dz
            
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                occ_val = occ[tx, ty, tz]
                is_valid = False
                if occ_val == 0:
                    is_valid = True
                elif allow_capture:
                    if occ_val != player_color:
                        is_valid = True
                
                if is_valid:
                    moves[write_idx, 0] = px
                    moves[write_idx, 1] = py
                    moves[write_idx, 2] = pz
                    moves[write_idx, 3] = tx
                    moves[write_idx, 4] = ty
                    moves[write_idx, 5] = tz
                    write_idx += 1
                    
    return moves


@njit(cache=True, fastmath=True, parallel=True)
def _generate_jump_moves_batch_unified(
    positions: np.ndarray,
    directions: np.ndarray,
    buffed_squares: np.ndarray,
    occ: np.ndarray,
    allow_capture: bool,
    player_color: int
) -> np.ndarray:
    """Generate moves for a batch with integrated buff handling.
    
    Fuses:
    1. Buff check per piece
    2. Direction calculation (normal vs buffed)
    3. Target calculation
    4. Bounds checking
    5. Occupancy/Capture filtering
    
    Args:
        positions: (N, 3) array of piece positions
        directions: (M, 3) array of base jump offsets
        buffed_squares: (SIZE, SIZE, SIZE) boolean array of buffed status
        occ: (SIZE, SIZE, SIZE) occupancy array
        allow_capture: Whether capturing is allowed
        player_color: Color of the moving player
        
    Returns:
        (K, 6) array of moves [fx, fy, fz, tx, ty, tz]
    """
    n_pos = positions.shape[0]
    n_dirs = directions.shape[0]
    
    # Pass 1: Count
    counts = np.zeros(n_pos, dtype=np.int32)
    
    for i in prange(n_pos):
        px, py, pz = positions[i]
        is_buffed = buffed_squares[px, py, pz]
        count = 0
        
        for j in range(n_dirs):
            dx, dy, dz = directions[j]
            
            if is_buffed:
                adx, ady, adz = abs(dx), abs(dy), abs(dz)
                max_val = max(adx, max(ady, adz))
                
                sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
                sy = 1 if dy > 0 else (-1 if dy < 0 else 0)
                sz = 1 if dz > 0 else (-1 if dz < 0 else 0)
                
                dx = dx + (sx if adx == max_val else 0)
                dy = dy + (sy if ady == max_val else 0)
                dz = dz + (sz if adz == max_val else 0)
            
            tx, ty, tz = px + dx, py + dy, pz + dz
            
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                occ_val = occ[tx, ty, tz]
                is_valid = False
                if occ_val == 0:
                    is_valid = True
                elif allow_capture:
                    if occ_val != player_color:
                        is_valid = True
                
                if is_valid:
                    count += 1
        counts[i] = count
        
    # Pass 2: Offsets
    total_moves = np.sum(counts)
    offsets = np.zeros(n_pos, dtype=np.int32)
    current_offset = 0
    for i in range(n_pos):
        offsets[i] = current_offset
        current_offset += counts[i]
        
    # Pass 3: Fill
    moves = np.empty((total_moves, 6), dtype=COORD_DTYPE)
    
    for i in prange(n_pos):
        write_idx = offsets[i]
        px, py, pz = positions[i]
        is_buffed = buffed_squares[px, py, pz]
        
        for j in range(n_dirs):
            dx, dy, dz = directions[j]
            
            if is_buffed:
                adx, ady, adz = abs(dx), abs(dy), abs(dz)
                max_val = max(adx, max(ady, adz))
                
                sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
                sy = 1 if dy > 0 else (-1 if dy < 0 else 0)
                sz = 1 if dz > 0 else (-1 if dz < 0 else 0)
                
                dx = dx + (sx if adx == max_val else 0)
                dy = dy + (sy if ady == max_val else 0)
                dz = dz + (sz if adz == max_val else 0)
            
            tx, ty, tz = px + dx, py + dy, pz + dz
            
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                occ_val = occ[tx, ty, tz]
                is_valid = False
                if occ_val == 0:
                    is_valid = True
                elif allow_capture:
                    if occ_val != player_color:
                        is_valid = True
                
                if is_valid:
                    moves[write_idx, 0] = px
                    moves[write_idx, 1] = py
                    moves[write_idx, 2] = pz
                    moves[write_idx, 3] = tx
                    moves[write_idx, 4] = ty
                    moves[write_idx, 5] = tz
                    write_idx += 1
                    
    return moves


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
                # Use 3D occupancy directly to avoid copy
                occ = cache_manager.occupancy_cache._occ
                
                if aura_cache is not None:
                    # Use unified kernel that handles buffs per-piece
                    # This avoids Python-side "any" checks and array allocations
                    buffed_squares = aura_cache._buffed_squares
                    return _generate_jump_moves_batch_unified(
                        pos, directions, buffed_squares, 
                        occ, allow_capture, color
                    )

                # Fast path: No buffs, use standard batch kernel
                return _generate_and_filter_jump_moves_batch(
                    pos, directions, occ, allow_capture, color
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
            occ = cache_manager.occupancy_cache._occ
            
            if targets is None:
                current_directions = directions
                if is_buffed:
                    # For single piece, just calculating here is fine, or we could use the unified kernel too.
                    # But let's keep it simple and consistent with legacy for now unless we want to unify everything.
                    # Actually, calculating buffed directions in Python for single piece is fast enough.
                    current_directions = directions.copy()
                    abs_dirs = np.abs(current_directions)
                    max_vals = np.max(abs_dirs, axis=1, keepdims=True)
                    mask = (abs_dirs == max_vals)
                    sign = np.sign(current_directions)
                    current_directions = current_directions + (sign * mask).astype(COORD_DTYPE)

                return _generate_and_filter_jump_moves(
                    pos, current_directions, occ, allow_capture, color
                )

            # Use optimized batch_is_occupied_unsafe for precomputed targets
            valid_targets = _filter_jump_targets_by_occupancy(
                targets, occ, allow_capture, color
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
