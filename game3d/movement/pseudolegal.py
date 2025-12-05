# pseudolegal.py
"""Raw pseudo-legal move generation - NO VALIDATION.

This module ONLY generates raw moves by calling piece dispatchers.
It does NOT validate moves or apply any filtering.

ARCHITECTURAL CONTRACT:
- All functions return numpy arrays of shape (N,6) with dtype=COORD_DTYPE
- No validation is performed here
- Validation happens in generator.py
- Filtering happens in turnmove.py
"""

from __future__ import annotations
import numpy as np
import logging
from typing import TYPE_CHECKING
from numba import njit, prange

from game3d.common.shared_types import COORD_DTYPE, PieceType, MOVE_DTYPE, Color
from game3d.common.registry import get_piece_dispatcher
from game3d.pieces.pieces.pawn import generate_pawn_moves

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

logger = logging.getLogger(__name__)

# Parallel execution imports
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib not available - parallel move generation disabled")

# Parallelization threshold: only parallelize when piece count exceeds this
PARALLEL_THRESHOLD = 8


class MoveContractViolation(TypeError):
    """Raised when dispatcher returns non-native arrays."""
    pass


# =============================================================================
# COORDINATE KEY UTILITIES - NUMBA COMPILED
# =============================================================================

@njit(cache=True, fastmath=True)
def coord_to_key(coords: np.ndarray) -> np.ndarray:
    """
    Convert (N,3) coordinates to integer keys using bit packing.
    PACKING: 9 bits for x + 9 bits for y + 9 bits for z = 27 bits total
    """
    n = coords.shape[0]
    keys = np.empty(n, dtype=np.int32)

    for i in prange(n):
        # Pack coordinates: x in bits 0-8, y in bits 9-17, z in bits 18-26
        keys[i] = (coords[i, 0]) | (coords[i, 1] << 9) | (coords[i, 2] << 18)

    return keys


@njit(cache=True, fastmath=True, parallel=True)
def extract_piece_moves_from_batch(
    batch_moves: np.ndarray,
    piece_coord: np.ndarray
) -> np.ndarray:
    """
    VECTORIZED: Extract moves for a specific piece from batch.

    Args:
        batch_moves: (N, 6) array of all moves
        piece_coord: (3,) array of piece coordinate

    Returns:
        (M, 6) array of moves for this piece
    """
    # Create mask for this piece's moves
    mask = (batch_moves[:, 0] == piece_coord[0]) & \
           (batch_moves[:, 1] == piece_coord[1]) & \
           (batch_moves[:, 2] == piece_coord[2])

    # Count matches
    n_matches = np.sum(mask)

    if n_matches == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Extract matching moves
    result = np.empty((n_matches, 6), dtype=COORD_DTYPE)
    write_idx = 0

    for i in range(batch_moves.shape[0]):
        if mask[i]:
            result[write_idx] = batch_moves[i]
            write_idx += 1

    return result


@njit(cache=True, fastmath=True)
def group_indices_by_type(types: np.ndarray, indices: np.ndarray, max_type: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group indices by type using counting sort (O(N)).
    
    Args:
        types: (N,) array of piece types
        indices: (N,) array of original indices
        max_type: Maximum piece type value
        
    Returns:
        sorted_indices: (N,) array of indices sorted by type
        offsets: (max_type + 1,) array of start offsets for each type
        counts: (max_type + 1,) array of counts for each type
    """
    n = types.shape[0]
    
    # Count per type
    counts = np.zeros(max_type + 1, dtype=np.int32)
    for i in range(n):
        counts[types[i]] += 1
        
    # Offsets
    offsets = np.zeros(max_type + 1, dtype=np.int32)
    current = 0
    for i in range(max_type + 1):
        offsets[i] = current
        current += counts[i]
        
    # Fill
    sorted_indices = np.empty(n, dtype=np.int32)
    current_offsets = offsets.copy()
    
    for i in range(n):
        t = types[i]
        pos = current_offsets[t]
        sorted_indices[pos] = indices[i]
        current_offsets[t] += 1
        
    return sorted_indices, offsets, counts

# =============================================================================
# RAW MOVE GENERATION - NO VALIDATION
# =============================================================================




def generate_pseudolegal_moves_batch(
    state: "GameState",
    batch_coords: np.ndarray,
    debuffed_coords: np.ndarray = None,
    ignore_occupancy: bool = False
) -> np.ndarray:
    """
    Generate raw pseudo-legal moves for a batch of pieces.
    
    This function ONLY generates moves by calling piece dispatchers.
    NO validation, NO filtering is performed.
    
    Args:
        state: Game state
        batch_coords: (N, 3) array of piece coordinates
        debuffed_coords: Optional (M, 3) array of debuffed squares
        ignore_occupancy: If True, sliders generate moves through pieces (for pin detection)
        
    Returns:
        (K, 6) array of raw moves [from_x, from_y, from_z, to_x, to_y, to_z]
        
    Raises:
        MoveContractViolation: If dispatcher violates contract
    """
    moves_list = []

    # Convert debuffed coords to keys for fast lookup
    debuffed_keys = set()
    if debuffed_coords is not None and debuffed_coords.size > 0:
        keys = coord_to_key(debuffed_coords)
        debuffed_keys = set(keys)

    # ✅ OPTIMIZATION 1: Batch pre-fetch all piece info instead of individual get() calls
    # Use UNSAFE fast path since coords come from get_positions() which always returns valid coords
    if batch_coords.shape[0] > 0:
        colors_batch, types_batch = state.cache_manager.occupancy_cache.batch_get_attributes_unsafe(batch_coords)
    else:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # ✅ OPTIMIZATION 2: Pre-compute coord keys for all coordinates
    coord_keys = coord_to_key(batch_coords)
    
    # ✅ OPTIMIZATION 3: Group pieces by type using Numba counting sort
    # This replaces argsort (O(N log N)) with counting sort (O(N))
    # and avoids allocating intermediate arrays.
    
    # Filter valid pieces (not empty)
    valid_mask = colors_batch != 0
    valid_indices = np.flatnonzero(valid_mask)
    
    if valid_indices.size == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)
        
    valid_types = types_batch[valid_indices]
    
    # Handle debuffs
    if debuffed_coords is not None and debuffed_coords.size > 0:
        # Vectorized check for debuffs
        debuffed_keys_arr = np.array(list(debuffed_keys), dtype=np.int32)
        valid_keys = coord_keys[valid_indices]
        is_debuffed = np.isin(valid_keys, debuffed_keys_arr)
        
        # Update types to PAWN where debuffed
        valid_types[is_debuffed] = PieceType.PAWN.value
        
    # Use Numba counting sort to group indices by type
    # Max piece type is around 40, so this is very efficient
    from game3d.common.shared_types import N_PIECE_TYPES
    sorted_indices, offsets, counts = group_indices_by_type(valid_types, valid_indices, N_PIECE_TYPES)
    
    # ✅ OPTIMIZATION 4: Process each piece type directly
    from game3d.common.registry import get_piece_dispatcher_fast
    
    # Iterate only over types that are present
    present_types = np.flatnonzero(counts)
    
    for piece_type in present_types:
        count = counts[piece_type]
        if count == 0: continue
            
        start = offsets[piece_type]
        end = start + count
        indices = sorted_indices[start:end]
        
        # Get dispatcher
        if piece_type == PieceType.PAWN.value:
            dispatcher = lambda s, p: generate_pawn_moves(s.cache_manager, s.color, p)
        else:
            dispatcher = get_piece_dispatcher_fast(piece_type)
            
        if not dispatcher:
             continue

        # Process batch
        coords = batch_coords[indices]
        
        # Try batch dispatch
        try:
             # OPTIMIZATION: Avoid try-except overhead if ignore_occupancy is False (default)
             if ignore_occupancy:
                 try:
                     raw_moves = dispatcher(state, coords, ignore_occupancy=True)
                 except TypeError:
                     # Dispatcher doesn't accept ignore_occupancy (e.g. Jump pieces)
                     # Just call without it
                     raw_moves = dispatcher(state, coords)
             else:
                 # If False, just call without argument (assumes default is False or not needed)
                 # This avoids TypeError for pieces that don't accept the argument (Knights, etc.)
                 raw_moves = dispatcher(state, coords)
                 
             # Assume valid numpy array return (skip checks for speed)
             if raw_moves.size > 0:
                 moves_list.append(raw_moves)
                 
        except Exception:
             # Fallback to sequential if batch fails
             for coord in coords:
                 try:
                     if ignore_occupancy:
                         try:
                             r = dispatcher(state, coord, ignore_occupancy=True)
                         except TypeError:
                             r = dispatcher(state, coord)
                     else:
                         r = dispatcher(state, coord)
                 except TypeError:
                     # Last resort fallback
                     r = dispatcher(state, coord)
                 
                 if r.size > 0:
                     moves_list.append(r)

    return np.concatenate(moves_list, axis=0) if moves_list else np.empty((0, 6), dtype=COORD_DTYPE)


def generate_pseudolegal_moves_for_piece(
    state: "GameState",
    coord: np.ndarray,
    debuffed_coords: np.ndarray = None,
    ignore_occupancy: bool = False
) -> np.ndarray:
    """
    Generate raw pseudo-legal moves for a single piece.
    
    This is a convenience wrapper around generate_pseudolegal_moves_batch
    for single-piece move generation.
    
    Args:
        state: Game state
        coord: (3,) or (1, 3) array of piece coordinate
        debuffed_coords: Optional (M, 3) array of debuffed squares
        ignore_occupancy: If True, sliders generate moves through pieces
        
    Returns:
        (N, 6) array of raw moves
    """
    # Ensure coord is 2D (1, 3)
    if coord.ndim == 1:
        coord = coord.reshape(1, 3)
    
    return generate_pseudolegal_moves_batch(state, coord, debuffed_coords, ignore_occupancy)


# =============================================================================
# CACHE REFRESH LOGIC
# =============================================================================

def _cache_piece_moves(cache_manager, batch_coords: np.ndarray, batch_moves: np.ndarray, color: int, is_raw: bool = False) -> None:
    """
    Groups batch moves by source coordinate and stores them in piece cache.
    Sorts moves by key for efficient grouping. Complexity: O(M log M).
    """
    if batch_moves.size == 0:
        # If no moves generated, we still need to mark pieces as having 0 moves
        # But we need to know WHICH pieces had 0 moves.
        # This function assumes batch_moves contains moves.
        # For pieces with 0 moves, we handle them in the caller loop or by checking coverage.
        return

    move_sources = batch_moves[:, :3]
    move_keys = coord_to_key(move_sources)

    # Group moves by sorting keys
    sort_idx = np.argsort(move_keys)
    sorted_moves = batch_moves[sort_idx]
    sorted_keys = move_keys[sort_idx]

    unique_keys, start_indices = np.unique(sorted_keys, return_index=True)
    end_indices = np.append(start_indices[1:], sorted_keys.size)

    for i in range(unique_keys.size):
        if is_raw:
            cache_manager.move_cache.store_piece_raw_moves(
                color, unique_keys[i], sorted_moves[start_indices[i]:end_indices[i]]
            )
        else:
            cache_manager.move_cache.store_piece_moves(
                color, unique_keys[i], sorted_moves[start_indices[i]:end_indices[i]]
            )

def refresh_pseudolegal_cache(state: "GameState") -> None:
    """
    Refresh the pseudolegal move cache for the current player.
    
    1. Identifies pieces that need updates (missing or invalidated).
    2. Generates RAW and PSEUDOLEGAL moves for them.
    3. Updates piece-level cache.
    4. Reconstructs and updates color-level cache.
    
    ✅ OPTIMIZED: Uses np.isin() for O(N log M) affected check instead of O(N*M)
    """
    cache_manager = state.cache_manager
    color = state.color
    
    # 1. Check if color-level cache is already valid
    if cache_manager.move_cache.get_pseudolegal_moves(color) is not None:
        return

    # 2. Identify pieces needing regeneration
    affected_pieces = cache_manager.move_cache.get_affected_pieces(color)
    all_coords = cache_manager.occupancy_cache.get_positions(color)
    
    if all_coords.size == 0:
        empty_moves = np.empty((0, 6), dtype=COORD_DTYPE)
        cache_manager.move_cache.store_raw_moves(color, empty_moves)
        cache_manager.move_cache.store_pseudolegal_moves(color, empty_moves)
        return

    coord_keys = coord_to_key(all_coords)
    
    # ✅ OPTIMIZED: Vectorized affected check using np.isin() - O(N log M) instead of O(N*M)
    is_affected_mask = np.isin(coord_keys, affected_pieces) if affected_pieces.size > 0 else np.zeros(len(coord_keys), dtype=bool)
    
    # ✅ OPTIMIZED: Batch check for missing cache entries
    # Pre-compute which pieces need regeneration in a single pass
    needs_regen_mask = is_affected_mask.copy()
    
    # Check missing entries using batch approach with single lock acquisition
    # We can't easily vectorize dictionary lookups, but we can minimize locking overhead
    move_cache = cache_manager.move_cache
    color_idx = 0 if color == Color.WHITE else 1
    
    # Check coverage efficiently
    with move_cache._lock:
        # Get all keys currently in cache for this color
        # Optimized: Extract directly to list then numpy array
        # Note: keys() returns (color_idx, coord_key) tuples
        keys_in_moves_cache = np.array(
            [k for c, k in move_cache._piece_moves_cache.keys() if c == color_idx], 
            dtype=np.int32
        )
        keys_in_raw_moves_cache = np.array(
            [k for c, k in move_cache._piece_raw_moves_cache.keys() if c == color_idx],
            dtype=np.int32
        )
        
        # Verify coverage using vectorized set operations
        # We need keys that are in BOTH caches
        # Since we just need to identify missing ones
        
        # Check 1: Is key in pseudolegal moves cache?
        in_pseudo = np.isin(coord_keys, keys_in_moves_cache)
        
        # Check 2: Is key in raw moves cache?
        in_raw = np.isin(coord_keys, keys_in_raw_moves_cache)
        
        # If missing in either, needs regen (OR if already marked from affected)
        # needs_regen_mask is already True for affected items
        # We Update it: needs_regen_mask |= (~in_pseudo | ~in_raw)
        missing_mask = (~in_pseudo) | (~in_raw)
        needs_regen_mask = needs_regen_mask | missing_mask
    
    # 3. Regenerate moves for affected pieces
    if np.any(needs_regen_mask):
        regenerate_coords = all_coords[needs_regen_mask]
        regenerate_keys = coord_keys[needs_regen_mask]
        debuffed_coords = cache_manager.consolidated_aura_cache.get_debuffed_squares(color)
        
        # A. Generate RAW moves (Ignore Occupancy)
        raw_moves_batch = generate_pseudolegal_moves_batch(state, regenerate_coords, debuffed_coords, ignore_occupancy=True)
        
        # B. Generate PSEUDOLEGAL moves (Respect Occupancy)
        pseudo_moves_batch = generate_pseudolegal_moves_batch(state, regenerate_coords, debuffed_coords, ignore_occupancy=False)
        
        # C. Update Piece Caches with single lock acquisition
        empty = np.empty((0, 6), dtype=COORD_DTYPE)
        
        with move_cache._lock:
            for key in regenerate_keys:
                int_key = int(key)
                piece_id = (color_idx, int_key)
                # Store empty moves first to handle pieces with 0 moves
                move_cache._piece_raw_moves_cache[piece_id] = empty
                move_cache._piece_moves_cache[piece_id] = empty
            
        # Now overwrite with actual moves
        if raw_moves_batch.size > 0:
            _cache_piece_moves(cache_manager, regenerate_coords, raw_moves_batch, color, is_raw=True)
            
        if pseudo_moves_batch.size > 0:
            _cache_piece_moves(cache_manager, regenerate_coords, pseudo_moves_batch, color, is_raw=False)

    # 4. ✅ OPTIMIZED: Batch reconstruct Color-level arrays from Piece cache
    # Single lock acquisition + list comprehension is faster than manual loops
    with move_cache._lock:
         # Use list comprehension for speed
         all_raw_moves = [
             move_cache._piece_raw_moves_cache.get((color_idx, int(key)), np.empty((0, 6), dtype=COORD_DTYPE))
             for key in coord_keys
         ]
         all_pseudo_moves = [
             move_cache._piece_moves_cache.get((color_idx, int(key)), np.empty((0, 6), dtype=COORD_DTYPE))
             for key in coord_keys
         ]
            
    final_raw = np.concatenate(all_raw_moves, axis=0) if all_raw_moves else np.empty((0, 6), dtype=COORD_DTYPE)
    final_pseudo = np.concatenate(all_pseudo_moves, axis=0) if all_pseudo_moves else np.empty((0, 6), dtype=COORD_DTYPE)
    
    # 5. Store in Color-level cache
    cache_manager.move_cache.store_raw_moves(color, final_raw)
    cache_manager.move_cache.store_pseudolegal_moves(color, final_pseudo)
    
    # Clear affected status
    cache_manager.move_cache.clear_affected_pieces(color)

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'generate_pseudolegal_moves_batch',
    'generate_pseudolegal_moves_for_piece',
    'refresh_pseudolegal_cache',
    'coord_to_key',
    'extract_piece_moves_from_batch',
    'MoveContractViolation',
]
