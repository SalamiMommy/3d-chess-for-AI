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

from game3d.common.shared_types import COORD_DTYPE, PieceType
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
    
    # ✅ OPTIMIZATION 3: Group pieces by type to amortize dispatcher lookups
    # This reduces get_piece_dispatcher calls from N to unique_types
    
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
        # valid_types is a copy, so we can modify it
        valid_types[is_debuffed] = PieceType.PAWN.value
        
    # Sort by type to group them
    sort_idx = np.argsort(valid_types)
    sorted_indices = valid_indices[sort_idx]
    sorted_types = valid_types[sort_idx]
    
    # Find unique types and their start indices
    unique_types, start_indices = np.unique(sorted_types, return_index=True)
    
    # ✅ OPTIMIZATION 4: Process each piece type directly
    # Inlined logic to avoid helper function overhead and list construction
    from game3d.common.registry import get_piece_dispatcher_fast
    
    for i in range(len(unique_types)):
        piece_type = unique_types[i]
        start = start_indices[i]
        end = start_indices[i+1] if i + 1 < len(unique_types) else len(sorted_indices)
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
             # Try to pass ignore_occupancy
             try:
                 raw_moves = dispatcher(state, coords, ignore_occupancy=ignore_occupancy)
             except TypeError:
                 raw_moves = dispatcher(state, coords)
                 
             # Assume valid numpy array return (skip checks for speed)
             if raw_moves.size > 0:
                 moves_list.append(raw_moves)
                 
        except Exception:
             # Fallback to sequential if batch fails
             for coord in coords:
                 try:
                     r = dispatcher(state, coord, ignore_occupancy=ignore_occupancy)
                 except TypeError:
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
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'generate_pseudolegal_moves_batch',
    'generate_pseudolegal_moves_for_piece',
    'coord_to_key',
    'extract_piece_moves_from_batch',
    'MoveContractViolation',
]
