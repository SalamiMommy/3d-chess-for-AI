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

def _process_piece_type(
    state: "GameState",
    piece_type: int,
    indices: list,
    batch_coords: np.ndarray,
    dispatcher,
    ignore_occupancy: bool
) -> np.ndarray:
    """Process all pieces of a single type (helper for parallelization).
    
    Args:
        state: Game state
        piece_type: Piece type ID
        indices: List of indices in batch_coords for this piece type
        batch_coords: (N, 3) array of all piece coordinates
        dispatcher: Move generation function for this piece type
        ignore_occupancy: Whether to ignore occupancy
        
    Returns:
        (M, 6) array of moves for all pieces of this type
    """
    # Try batch processing for all types
    # Extract all coords for this piece type
    coords = batch_coords[indices]
    
    try:
        # Call dispatcher with batch coordinates
        # Try to pass ignore_occupancy first
        try:
            raw_moves = dispatcher(state, coords, ignore_occupancy=ignore_occupancy)
        except TypeError:
            # Dispatcher doesn't support ignore_occupancy
            raw_moves = dispatcher(state, coords)
        
        if not isinstance(raw_moves, np.ndarray):
            raise MoveContractViolation(
                f"Dispatcher for piece type {piece_type} returned {type(raw_moves)}. "
                f"Must return numpy array of shape (N, 6) with integer dtype."
            )
            
        if raw_moves.dtype != COORD_DTYPE:
            raw_moves = raw_moves.astype(COORD_DTYPE, copy=False)
            
        if raw_moves.ndim != 2 or raw_moves.shape[1] != 6:
            raise MoveContractViolation(
                f"Dispatcher returned shape {raw_moves.shape}. Expected (N, 6)."
            )
            
        return raw_moves
        
    except Exception as e:
        # Fallback to sequential if batch fails
        # This handles cases where dispatcher doesn't support batch input
        # or raises an error during batch processing
        # logger.debug(f"Batch generation failed for type {piece_type}: {e}. Falling back to sequential.")
        pass

    moves_list = []
    
    for i in indices:
        coord = batch_coords[i]
        
        # Try to pass ignore_occupancy, fallback if not supported
        try:
            raw_moves = dispatcher(state, coord, ignore_occupancy=ignore_occupancy)
        except TypeError:
            # Dispatcher doesn't support ignore_occupancy
            raw_moves = dispatcher(state, coord)
        
        # ENFORCE CONTRACT: Crash immediately if dispatcher violates contract
        if not isinstance(raw_moves, np.ndarray):
            raise MoveContractViolation(
                f"Dispatcher for piece type {piece_type} returned {type(raw_moves)}. "
                f"Must return numpy array of shape (N, 6) with integer dtype."
            )
        
        # Ensure correct dtype without copying if already correct
        if raw_moves.dtype != COORD_DTYPE:
            raw_moves = raw_moves.astype(COORD_DTYPE, copy=False)
        
        # Ensure correct shape
        if raw_moves.ndim != 2 or raw_moves.shape[1] != 6:
            raise MoveContractViolation(
                f"Dispatcher returned shape {raw_moves.shape}. Expected (N, 6)."
            )
        
        moves_list.append(raw_moves)
    
    return np.concatenate(moves_list, axis=0) if moves_list else np.empty((0, 6), dtype=COORD_DTYPE)


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
    from collections import defaultdict
    pieces_by_type = defaultdict(list)
    
    for i, coord in enumerate(batch_coords):
        color = colors_batch[i]
        piece_type = types_batch[i]
        
        # Skip empty squares (color == 0)
        if color == 0:
            continue
        
        # Check if piece is debuffed using pre-computed keys
        is_debuffed = coord_keys[i] in debuffed_keys
        
        # Use special type code for debuffed pieces
        effective_type = PieceType.PAWN.value if is_debuffed else piece_type
        pieces_by_type[effective_type].append(i)
    
    # ✅ OPTIMIZATION 4: Process each piece type with parallel execution
    from game3d.common.registry import get_piece_dispatcher_fast
    
    # DISABLED: Parallelization conflicts with Numba's internal parallelization (prange)
    # Numba functions with parallel=True use their own threading layer which conflicts
    # with joblib threads. Since Numba already parallelizes the heavy operations,
    # we don't need joblib parallelization on top.
    use_parallel = False
    
    # Original threshold-based logic kept for future use if needed
    # total_pieces = sum(len(indices) for indices in pieces_by_type.items())
    # use_parallel = (JOBLIB_AVAILABLE and 
    #                total_pieces > PARALLEL_THRESHOLD and 
    #                len(pieces_by_type) > 1)
    
    if use_parallel:
        # PARALLEL PATH: Currently disabled due to Numba threading conflicts
        # If re-enabled, must use prefer="processes" to avoid Numba workqueue conflicts
        def process_wrapper(piece_type, indices):
            """Wrapper to get dispatcher and process piece type."""
            if piece_type == PieceType.PAWN.value:
                dispatcher = lambda s, p: generate_pawn_moves(s.cache_manager, s.color, p)
            else:
                dispatcher = get_piece_dispatcher_fast(piece_type)
            
            if not dispatcher:
                raise RuntimeError(f"No dispatcher registered for piece type {piece_type}")
            
            return _process_piece_type(state, piece_type, indices, batch_coords, 
                                      dispatcher, ignore_occupancy)
        
        # Use prefer="processes" to avoid Numba threading conflicts
        # Note: This has overhead, so threshold should be high (50+ pieces)
        results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(process_wrapper)(piece_type, indices)
            for piece_type, indices in pieces_by_type.items()
        )
        
        moves_list.extend(results)
        
    else:
        # SEQUENTIAL PATH: Fallback for small batches or when joblib unavailable
        for piece_type, indices in pieces_by_type.items():
            # Get dispatcher once per type instead of once per piece
            if piece_type == PieceType.PAWN.value:
                dispatcher = lambda s, p: generate_pawn_moves(s.cache_manager, s.color, p)
            else:
                dispatcher = get_piece_dispatcher_fast(piece_type)
            
            if not dispatcher:
                raise RuntimeError(f"No dispatcher registered for piece type {piece_type}")
            
            # Process all pieces of this type sequentially
            piece_moves = _process_piece_type(state, piece_type, indices, batch_coords,
                                             dispatcher, ignore_occupancy)
            if piece_moves.size > 0:
                moves_list.append(piece_moves)

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
