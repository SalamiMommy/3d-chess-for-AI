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
    debuffed_coords: np.ndarray = None
) -> np.ndarray:
    """
    Generate raw pseudo-legal moves for a batch of pieces.
    
    This function ONLY generates moves by calling piece dispatchers.
    NO validation, NO filtering is performed.
    
    Args:
        state: Game state
        batch_coords: (N, 3) array of piece coordinates
        debuffed_coords: Optional (M, 3) array of debuffed squares
        
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

    for i, coord in enumerate(batch_coords):
        piece_info = state.cache_manager.occupancy_cache.get(coord.reshape(1, 3))
        if piece_info is None:
            # logger.debug(f"Skipping empty affected coordinate {coord}")  # Expected in incremental updates
            continue

        piece_type = piece_info["piece_type"]

        # Check if piece is debuffed
        coord_key = coord_to_key(coord.reshape(1, -1))[0]
        is_debuffed = coord_key in debuffed_keys
        
        if is_debuffed:
            # Use Pawn movement for debuffed pieces
            dispatcher = lambda s, p: generate_pawn_moves(s.cache_manager, s.color, p)
        else:
            dispatcher = get_piece_dispatcher(piece_type)

        if not dispatcher:
            raise RuntimeError(f"No dispatcher registered for piece type {piece_type}")

        # CRITICAL: DISPATCHER MUST RETURN NUMPY ARRAY OR FAIL FAST
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


def generate_pseudolegal_moves_for_piece(
    state: "GameState",
    coord: np.ndarray,
    debuffed_coords: np.ndarray = None
) -> np.ndarray:
    """
    Generate raw pseudo-legal moves for a single piece.
    
    This is a convenience wrapper around generate_pseudolegal_moves_batch
    for single-piece move generation.
    
    Args:
        state: Game state
        coord: (3,) or (1, 3) array of piece coordinate
        debuffed_coords: Optional (M, 3) array of debuffed squares
        
    Returns:
        (N, 6) array of raw moves
    """
    # Ensure coord is 2D (1, 3)
    if coord.ndim == 1:
        coord = coord.reshape(1, 3)
    
    return generate_pseudolegal_moves_batch(state, coord, debuffed_coords)


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
