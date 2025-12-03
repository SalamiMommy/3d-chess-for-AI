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

# Module-level cache for precomputed moves (piece_type -> variant -> data)
# Variants are 'unbuffed' and 'buffed'
_PRECOMPUTED_MOVES: Dict[int, Dict[str, np.ndarray]] = {}
# Flattened arrays for Numba (piece_type -> variant -> flat_moves_array)
_PRECOMPUTED_MOVES_FLAT: Dict[int, Dict[str, np.ndarray]] = {}
# Offsets for Numba (piece_type -> variant -> offsets_array of size SIZE^3 + 1)
_PRECOMPUTED_OFFSETS: Dict[int, Dict[str, np.ndarray]] = {}
_PRECOMPUTED_LOADED = False

def _load_precomputed_moves():
    """Load precomputed move tables from disk for both buffed and unbuffed variants."""
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

    # Iterate through PieceType members and load both buffed and unbuffed variants
    for piece_type in PieceType:
        name = piece_type.name
        pt_val = piece_type.value
        
        # Initialize nested dictionaries for this piece type
        _PRECOMPUTED_MOVES[pt_val] = {}
        _PRECOMPUTED_MOVES_FLAT[pt_val] = {}
        _PRECOMPUTED_OFFSETS[pt_val] = {}
        
        # Load both variants: 'unbuffed' and 'buffed'
        for variant in ['unbuffed', 'buffed']:
            # Try loading flat arrays first (preferred for Numba)
            flat_filename = f"moves_{name}_{variant}_flat.npy"
            offsets_filename = f"moves_{name}_{variant}_offsets.npy"
            flat_path = os.path.join(precomputed_dir, flat_filename)
            offsets_path = os.path.join(precomputed_dir, offsets_filename)
            
            if os.path.exists(flat_path) and os.path.exists(offsets_path):
                try:
                    flat_moves = np.load(flat_path)
                    offsets = np.load(offsets_path)
                    
                    _PRECOMPUTED_MOVES_FLAT[pt_val][variant] = flat_moves
                    _PRECOMPUTED_OFFSETS[pt_val][variant] = offsets
                    
                    # Also load the object array for single-piece path compatibility
                    legacy_filename = f"moves_{name}_{variant}.npy"
                    legacy_path = os.path.join(precomputed_dir, legacy_filename)
                    if os.path.exists(legacy_path):
                        _PRECOMPUTED_MOVES[pt_val][variant] = np.load(legacy_path, allow_pickle=True)
                    
                except Exception as e:
                    print(f"Failed to load {variant} precomputed moves for {name}: {e}")
    
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
def _generate_jump_moves_batch_precomputed(
    positions: np.ndarray,
    flat_moves: np.ndarray,
    offsets: np.ndarray,
    occ: np.ndarray,
    allow_capture: bool,
    player_color: int
) -> np.ndarray:
    """Generate jump moves using precomputed tables.
    
    Args:
        positions: (N, 3) array of piece positions
        flat_moves: (TotalMoves, 3) array of all precomputed targets
        offsets: (SIZE^3 + 1,) array of offsets into flat_moves
        occ: (SIZE, SIZE, SIZE) occupancy array
        allow_capture: Whether capturing is allowed
        player_color: Color of the moving player
        
    Returns:
        (K, 6) array of moves [fx, fy, fz, tx, ty, tz]
    """
    n_pos = positions.shape[0]
    
    # Pass 1: Count valid moves
    counts = np.zeros(n_pos, dtype=np.int32)
    
    for i in prange(n_pos):
        px, py, pz = positions[i]
        
        # Calculate flat index for precomputed lookup
        # Note: This assumes standard packing: x + y*SIZE + z*SIZE^2
        flat_idx = px + SIZE * py + SIZE_SQUARED * pz
        
        start = offsets[flat_idx]
        end = offsets[flat_idx + 1]
        
        count = 0
        for k in range(start, end):
            tx, ty, tz = flat_moves[k, 0], flat_moves[k, 1], flat_moves[k, 2]
            
            # Bounds are guaranteed by precomputed table
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
    write_offsets = np.zeros(n_pos, dtype=np.int32)
    current_offset = 0
    for i in range(n_pos):
        write_offsets[i] = current_offset
        current_offset += counts[i]
        
    # Pass 3: Fill
    moves = np.empty((total_moves, 6), dtype=COORD_DTYPE)
    
    for i in prange(n_pos):
        write_idx = write_offsets[i]
        px, py, pz = positions[i]
        flat_idx = px + SIZE * py + SIZE_SQUARED * pz
        
        start = offsets[flat_idx]
        end = offsets[flat_idx + 1]
        
        for k in range(start, end):
            tx, ty, tz = flat_moves[k, 0], flat_moves[k, 1], flat_moves[k, 2]
            
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


@njit(cache=True, fastmath=True, parallel=False)
def _generate_and_filter_jump_moves_batch_serial(
    positions: np.ndarray,
    directions: np.ndarray,
    occ: np.ndarray,
    allow_capture: bool,
    player_color: int
) -> np.ndarray:
    """Serial version of _generate_and_filter_jump_moves_batch."""
    n_pos = positions.shape[0]
    n_dirs = directions.shape[0]
    
    # Pass 1: Count
    counts = np.zeros(n_pos, dtype=np.int32)
    
    for i in range(n_pos):
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
    
    for i in range(n_pos):
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


@njit(cache=True, fastmath=True, parallel=False)
def _generate_jump_moves_batch_precomputed_serial(
    positions: np.ndarray,
    flat_moves: np.ndarray,
    offsets: np.ndarray,
    occ: np.ndarray,
    allow_capture: bool,
    player_color: int
) -> np.ndarray:
    """Serial version of _generate_jump_moves_batch_precomputed."""
    n_pos = positions.shape[0]
    
    # Pass 1: Count valid moves
    counts = np.zeros(n_pos, dtype=np.int32)
    
    for i in range(n_pos):
        px, py, pz = positions[i]
        # Calculate flat index for precomputed lookup
        flat_idx = px + SIZE * py + SIZE_SQUARED * pz
        
        start = offsets[flat_idx]
        end = offsets[flat_idx + 1]
        
        count = 0
        for k in range(start, end):
            tx, ty, tz = flat_moves[k, 0], flat_moves[k, 1], flat_moves[k, 2]
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
    write_offsets = np.zeros(n_pos, dtype=np.int32)
    current_offset = 0
    for i in range(n_pos):
        write_offsets[i] = current_offset
        current_offset += counts[i]
        
    # Pass 3: Fill
    moves = np.empty((total_moves, 6), dtype=COORD_DTYPE)
    
    for i in range(n_pos):
        write_idx = write_offsets[i]
        px, py, pz = positions[i]
        flat_idx = px + SIZE * py + SIZE_SQUARED * pz
        
        start = offsets[flat_idx]
        end = offsets[flat_idx + 1]
        
        for k in range(start, end):
            tx, ty, tz = flat_moves[k, 0], flat_moves[k, 1], flat_moves[k, 2]
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
        color: int,
        pos: np.ndarray,
        directions: np.ndarray,
        allow_capture: bool = True,
        piece_type: Optional[PieceType] = None,
        buffed_directions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate jump moves for a piece or batch of pieces.
        
        Args:
            cache_manager: The cache manager instance
            color: The color of the piece(s)
            pos: (3,) or (N, 3) array of positions
            directions: (M, 3) array of jump directions (unbuffed)
            allow_capture: Whether capturing is allowed
            piece_type: Optional PieceType for precomputed optimization
            buffed_directions: Optional (K, 3) array of jump directions when buffed.
                               If None, buffed pieces use standard directions (no buff effect).
        """
        # Handle single piece input
        if pos.ndim == 1:
            # Check for buff
            is_buffed = False
            # OPTIMIZATION: Direct access, try/except is faster than hasattr for likely attribute
            try:
                # Direct access to boolean array
                x, y, z = pos[0], pos[1], pos[2]
                is_buffed = cache_manager.consolidated_aura_cache._buffed_squares[x, y, z]
            except AttributeError:
                pass
            
            # Select directions and variant
            dirs_to_use = directions
            variant = 'unbuffed'
            if is_buffed:
                variant = 'buffed'
                if buffed_directions is not None:
                    dirs_to_use = buffed_directions
            
            # Use precomputed if available for the appropriate variant
            if piece_type is not None:
                pt_val = piece_type.value
                # OPTIMIZATION: Check dict directly without 'in' first if possible, but dict lookup is fast
                if pt_val in _PRECOMPUTED_MOVES:
                    moves_map = _PRECOMPUTED_MOVES[pt_val]
                    if variant in moves_map:
                        # Legacy object array path - fast for single piece
                        moves_list = moves_map[variant]
                        idx = compute_board_index(pos[0], pos[1], pos[2])
                        candidates = moves_list[idx]
                        
                        if candidates is None or len(candidates) == 0:
                            return np.empty((0, 6), dtype=COORD_DTYPE)
                            
                        # Filter by occupancy
                        occ = cache_manager.occupancy_cache._occ
                        
                        valid_targets = _filter_jump_targets_by_occupancy(
                            candidates, occ, allow_capture, color
                        )
                        
                        if valid_targets.shape[0] == 0:
                            return np.empty((0, 6), dtype=COORD_DTYPE)
                            
                        # Construct result
                        n = valid_targets.shape[0]
                        result = np.empty((n, 6), dtype=COORD_DTYPE)
                        result[:, 0] = pos[0]
                        result[:, 1] = pos[1]
                        result[:, 2] = pos[2]
                        result[:, 3:] = valid_targets
                        return result

            # Fallback to runtime generation
            occ = cache_manager.occupancy_cache._occ
            return _generate_and_filter_jump_moves(
                pos, dirs_to_use, occ, allow_capture, color
            )

        # Batch input
        n_pos = pos.shape[0]
        if n_pos == 0:
            return np.empty((0, 6), dtype=COORD_DTYPE)

        occ = cache_manager.occupancy_cache._occ
        
        # Get buff status for all positions
        # OPTIMIZATION: Assume consolidated_aura_cache exists or handle gracefully
        try:
            # (N,) boolean array
            x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
            is_buffed_batch = cache_manager.consolidated_aura_cache._buffed_squares[x, y, z]
            has_buffs = np.any(is_buffed_batch)
        except AttributeError:
            is_buffed_batch = None
            has_buffs = False
            
        # OPTIMIZATION: Fast path for all unbuffed (very common)
        if not has_buffs:
            # All unbuffed
            # Try precomputed optimization for unbuffed variant
            if piece_type is not None:
                pt_val = piece_type.value
                if pt_val in _PRECOMPUTED_MOVES_FLAT:
                    flat_map = _PRECOMPUTED_MOVES_FLAT[pt_val]
                    if 'unbuffed' in flat_map:
                        flat_moves = flat_map['unbuffed']
                        offsets = _PRECOMPUTED_OFFSETS[pt_val]['unbuffed']
                        
                        # Use serial for small batches, parallel for large
                        if n_pos < 300:
                            return _generate_jump_moves_batch_precomputed_serial(
                                pos, flat_moves, offsets, occ, allow_capture, color
                            )
                        else:
                            return _generate_jump_moves_batch_precomputed(
                                pos, flat_moves, offsets, occ, allow_capture, color
                            )
            
            # Standard runtime generation
            if n_pos < 300:
                return _generate_and_filter_jump_moves_batch_serial(
                    pos, directions, occ, allow_capture, color
                )
            else:
                return _generate_and_filter_jump_moves_batch(
                    pos, directions, occ, allow_capture, color
                )

        # Mixed buffed/unbuffed
        buffed_indices = np.flatnonzero(is_buffed_batch)
        unbuffed_indices = np.flatnonzero(~is_buffed_batch)
        
        results = []
        
        # 1. Process Unbuffed Pieces
        if unbuffed_indices.size > 0:
            unbuffed_pos = pos[unbuffed_indices]
            
            # Try precomputed optimization for unbuffed variant
            used_precomputed = False
            if piece_type is not None:
                pt_val = piece_type.value
                if pt_val in _PRECOMPUTED_MOVES_FLAT and 'unbuffed' in _PRECOMPUTED_MOVES_FLAT[pt_val]:
                    flat_moves = _PRECOMPUTED_MOVES_FLAT[pt_val]['unbuffed']
                    offsets = _PRECOMPUTED_OFFSETS[pt_val]['unbuffed']
                    
                    # Use serial for small batches, parallel for large
                    if unbuffed_indices.size < 300:
                        moves = _generate_jump_moves_batch_precomputed_serial(
                            unbuffed_pos, flat_moves, offsets, occ, allow_capture, color
                        )
                    else:
                        moves = _generate_jump_moves_batch_precomputed(
                            unbuffed_pos, flat_moves, offsets, occ, allow_capture, color
                        )
                    results.append(moves)
                    used_precomputed = True
            
            if not used_precomputed:
                # Standard runtime generation
                if unbuffed_indices.size < 300:
                    moves = _generate_and_filter_jump_moves_batch_serial(
                        unbuffed_pos, directions, occ, allow_capture, color
                    )
                else:
                    moves = _generate_and_filter_jump_moves_batch(
                        unbuffed_pos, directions, occ, allow_capture, color
                    )
                results.append(moves)
                
        # 2. Process Buffed Pieces
        if buffed_indices.size > 0:
            buffed_pos = pos[buffed_indices]
            
            # Determine directions to use
            dirs_to_use = directions
            if buffed_directions is not None:
                dirs_to_use = buffed_directions
            
            # Try precomputed optimization for buffed variant
            used_precomputed = False
            if piece_type is not None and buffed_directions is None:
                pt_val = piece_type.value
                if pt_val in _PRECOMPUTED_MOVES_FLAT and 'buffed' in _PRECOMPUTED_MOVES_FLAT[pt_val]:
                    flat_moves = _PRECOMPUTED_MOVES_FLAT[pt_val]['buffed']
                    offsets = _PRECOMPUTED_OFFSETS[pt_val]['buffed']
                    
                    if buffed_indices.size < 300:
                        moves = _generate_jump_moves_batch_precomputed_serial(
                            buffed_pos, flat_moves, offsets, occ, allow_capture, color
                        )
                    else:
                        moves = _generate_jump_moves_batch_precomputed(
                            buffed_pos, flat_moves, offsets, occ, allow_capture, color
                        )
                    results.append(moves)
                    used_precomputed = True
            
            if not used_precomputed:
                # Runtime generation with buffed directions
                if buffed_indices.size < 300:
                    moves = _generate_and_filter_jump_moves_batch_serial(
                        buffed_pos, dirs_to_use, occ, allow_capture, color
                    )
                else:
                    moves = _generate_and_filter_jump_moves_batch(
                        buffed_pos, dirs_to_use, occ, allow_capture, color
                    )
                results.append(moves)
            
        # Combine results
        if not results:
            return np.empty((0, 6), dtype=COORD_DTYPE)
        elif len(results) == 1:
            final_moves = results[0]
        else:
            final_moves = np.vstack(results)
            
        # ✅ CRITICAL: Defensive bounds validation
        if final_moves.size > 0:
            valid_mask = (
                (final_moves[:, 3] >= 0) & (final_moves[:, 3] < SIZE) &
                (final_moves[:, 4] >= 0) & (final_moves[:, 4] < SIZE) &
                (final_moves[:, 5] >= 0) & (final_moves[:, 5] < SIZE)
            )
            if not np.all(valid_mask):
                import warnings
                n_invalid = np.sum(~valid_mask)
                warnings.warn(
                    f"Jump engine filtered {n_invalid} out-of-bounds moves. "
                    f"This indicates a bug in move generation.",
                    RuntimeWarning
                )
                final_moves = final_moves[valid_mask]
                
        return final_moves


def get_jump_movement_generator() -> JumpMovementEngine:
    """Backwards compatibility alias for JumpMovementEngine constructor."""
    return JumpMovementEngine()

# Update the __all__ export list
__all__ = ['JumpMovementEngine', 'get_jump_movement_generator']
