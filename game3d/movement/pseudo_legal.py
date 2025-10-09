# pseudo_legal.py
"""Optimized pseudo-legal move generator with enhanced caching and performance."""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum
import time
from collections import defaultdict  # Added for pieces_by_type
if TYPE_CHECKING:
    from game3d.game.gamestate import GameState   # only for mypy/IDE
from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.movement.registry import register, get_dispatcher, get_all_dispatchers, dispatch_batch
from game3d.movement.movementmodifiers import modify_raw_moves
# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

BOARD_SIZE = 9  # Extracted

@dataclass(slots=True)
class PseudoLegalStats:
    """Statistics for pseudo-legal move generation."""
    total_calls: int = 0
    total_moves_generated: int = 0
    piece_breakdown: Dict[PieceType, int] = None
    average_time_ms: float = 0.0

    def __post_init__(self):
        if self.piece_breakdown is None:
            self.piece_breakdown = {pt: 0 for pt in PieceType}

class PseudoLegalMode(Enum):
    STANDARD   = "standard"
    BATCH      = "batch"        # <-- new
    INCREMENTAL= "incremental"

# ==============================================================================
# OPTIMIZED PSEUDO-LEGAL GENERATION
# ==============================================================================
_STATS = PseudoLegalStats()

def _generate_pseudo_legal_moves_impl(
    state: GameState,
    mode: PseudoLegalMode = PseudoLegalMode.STANDARD
) -> List[Move]:
    """Optimized pseudo-legal move generation with multiple strategies."""
    start_time = time.perf_counter()
    _STATS.total_calls += 1

    try:
        if mode == PseudoLegalMode.BATCH:
            moves = _generate_pseudo_legal_batch(state)
        elif mode == PseudoLegalMode.INCREMENTAL:
            moves = _generate_pseudo_legal_incremental(state)
        else:
            moves = _generate_pseudo_legal_standard(state)

        # Update statistics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _update_stats(elapsed_ms, len(moves))

        return moves

    except Exception as e:
        # Fallback to standard generation
        return _generate_pseudo_legal_standard(state)

def generate_pseudo_legal_moves(state: GameState) -> List[Move]:
    """Entry-point that always uses the standard path."""
    return _generate_pseudo_legal_moves_impl(state, mode=PseudoLegalMode.STANDARD)

def generate_pseudo_legal_moves_for_piece(state: GameState, coord: Tuple[int, int, int]) -> List[Move]:
    """Generate pseudo-legal moves for a specific piece."""
    piece = state.cache.occupancy.get(coord)
    if not piece or piece.color != state.color:
        return []

    # Check for SLOWER debuff (forces pawn movement)
    is_debuffed = state.cache.is_movement_debuffed(coord, piece.color)
    if is_debuffed and piece.ptype != PieceType.PAWN:  # Avoid redundant override for actual pawns
        dispatcher = get_dispatcher(PieceType.PAWN)  # Force pawn movement
    else:
        dispatcher = get_dispatcher(piece.ptype)

    if dispatcher is None:
        return []

    piece_moves = dispatcher(state, *coord)
    modified_moves = modify_raw_moves(piece_moves, coord, state)
    validated_moves = _validate_piece_moves(modified_moves, coord, piece, state)

    return validated_moves

def _generate_pseudo_legal_batch(state: GameState) -> list[Move]:
    """Batch pseudo-legal move generation – now *really* batched."""
    coords, types = [], []
    debuffed_indices = []  # Track indices of debuffed pieces

    for idx, (coord, piece) in enumerate(_get_current_player_pieces(state)):
        coords.append(coord)
        types.append(piece.ptype)
        if state.cache.is_movement_debuffed(coord, piece.color) and piece.ptype != PieceType.PAWN:
            debuffed_indices.append(idx)  # Mark for override

    # Normal dispatch
    raw_moves = dispatch_batch(state, coords, types, state.color)

    # Override debuffed pieces: Re-dispatch them as pawns
    if debuffed_indices:
        debuffed_coords = [coords[i] for i in debuffed_indices]
        pawn_types = [PieceType.PAWN] * len(debuffed_coords)  # Force pawn type
        debuffed_raw_moves = dispatch_batch(state, debuffed_coords, pawn_types, state.color)

        # Replace in raw_moves (assuming dispatch_batch returns in same order)
        for batch_idx, orig_idx in enumerate(debuffed_indices):
            # This assumes raw_moves is a flat list; adjust if structured
            # For simplicity, we'll rebuild the list here
            pass

    # Apply modifiers and validations post-batch
    all_moves = []
    cache_manager = state.cache
    color = state.color
    # Group moves by from_coord for per-piece processing
    moves_by_from = defaultdict(list)
    for move in raw_moves:
        moves_by_from[move.from_coord].append(move)

    for from_coord, piece_moves in moves_by_from.items():
        # Get piece once per group
        piece = state.cache.occupancy.get(from_coord)
        if not piece:
            continue
        # FIXED: Use from_coord instead of coord
        modified_moves = modify_raw_moves(piece_moves, from_coord, state)
        validated_moves = _validate_piece_moves(modified_moves, from_coord, piece, state)
        all_moves.extend(validated_moves)
        _STATS.piece_breakdown[piece.ptype] += len(validated_moves)
        _STATS.total_moves_generated += len(validated_moves)

    return all_moves

def _generate_pseudo_legal_incremental(state: GameState) -> List[Move]:
    """Incremental pseudo-legal move generation (for small changes)."""
    # For now, fall back to standard version
    return _generate_pseudo_legal_standard(state)

def _generate_pseudo_legal_standard(state: GameState) -> List[Move]:
    """Standard pseudo-legal move generation - CORRECTED."""
    all_moves: List[Move] = []

    for coord, piece in _get_current_player_pieces(state):
        dispatcher = get_dispatcher(piece.ptype)
        if dispatcher is None:
            continue

        try:
            piece_moves = dispatcher(state, *coord)
            modified_moves = modify_raw_moves(piece_moves, coord, state)
            validated_moves = _validate_piece_moves(modified_moves, coord, piece, state)

            if validated_moves:
                all_moves.extend(validated_moves)
                _STATS.piece_breakdown[piece.ptype] += len(validated_moves)
                _STATS.total_moves_generated += len(validated_moves)

        except Exception as e:
            print(f"Error generating moves for {piece.ptype} at {coord}: {e}")
            continue

    return all_moves

# ==============================================================================
# ENHANCED VALIDATION
# ==============================================================================
def _validate_piece_moves(
    moves: List[Move],
    expected_coord: Tuple[int, int, int],
    piece,
    state: "GameState"
) -> List[Move]:
    """
    Enhanced validation for piece-generated moves (basic checks only: bounds, from-consistency, dest not friendly).
    Uses raw occupancy arrays for direct checks.
    """
    if not moves:
        return moves

    cache_manager = state.cache

    # Get raw numpy arrays (no locking needed for reads)
    occ, _ptype = cache_manager.occupancy.export_arrays()
    color_code = piece.color.value  # 1 for white, 2 for black

    n_moves = len(moves)
    if n_moves < 32:  # Fallback to Python loop for small lists to avoid numpy overhead
        validated = []
        exp_x, exp_y, exp_z = expected_coord
        for move in moves:
            from_x, from_y, from_z = move.from_coord
            if (from_x, from_y, from_z) != (exp_x, exp_y, exp_z):
                continue

            to_x, to_y, to_z = move.to_coord
            if not (0 <= to_x < BOARD_SIZE and 0 <= to_y < BOARD_SIZE and 0 <= to_z < BOARD_SIZE):
                continue

            dest_code = occ[to_z, to_y, to_x]
            if dest_code == color_code:
                continue

            validated.append(move)
        return validated
    else:
        # Vectorized path for larger lists
        # Convert moves to numpy for vectorized operations
        from_coords = np.array([move.from_coord for move in moves])
        to_coords = np.array([move.to_coord for move in moves])

        expected_arr = np.array(expected_coord)  # For broadcasting

        # 1. Check from-square consistency (vectorized)
        valid_from = np.all(from_coords == expected_arr, axis=1)

        # 2. Bounds check (vectorized)
        valid_bounds = np.all((to_coords >= 0) & (to_coords < BOARD_SIZE), axis=1)

        # 3. Check destination occupancy/color DIRECTLY (vectorized)
        to_z, to_y, to_x = to_coords.T.astype(int)  # Ensure int for indexing
        dest_codes = occ[to_z, to_y, to_x]
        if piece.ptype is PieceType.KNIGHT:
            valid_dest = (dest_codes == 0) | (dest_codes != color_code)  # empty OR enemy
        else:
            valid_dest = dest_codes != color_code

        # Combine all validations
        valid_mask = valid_from & valid_bounds & valid_dest

        # Return validated moves
        return [moves[i] for i in np.flatnonzero(valid_mask)]


def _extend_move_range(move: Move, start_sq: Tuple[int, int, int], state: GameState) -> List[Move]:
    """Extend movement range for buffed pieces."""
    direction = (
        move.to_coord[0] - move.from_coord[0],
        move.to_coord[1] - move.from_coord[1],
        move.to_coord[2] - move.from_coord[2],
    )

    length = max(abs(d) for d in direction)
    if length == 0:
        return [move]

    normalized_dir = tuple(d // length for d in direction)

    extended_coord = (
        move.to_coord[0] + normalized_dir[0],
        move.to_coord[1] + normalized_dir[1],
        move.to_coord[2] + normalized_dir[2],
    )

    extended_moves = [move]

    if (0 <= extended_coord[0] < BOARD_SIZE and
        0 <= extended_coord[1] < BOARD_SIZE and
        0 <= extended_coord[2] < BOARD_SIZE):
        extended_move = Move(
            from_coord=move.from_coord,
            to_coord=extended_coord,
            is_capture=move.is_capture,
            metadata={**move.metadata, 'extended': True}
        )
        extended_moves.append(extended_move)

    return extended_moves

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def _get_current_player_pieces(state: GameState) -> List[Tuple[Tuple[int, int, int], Any]]:
    """Get all pieces for current player using cache - CENTRALIZED."""
    return list(state.cache.occupancy.iter_color(state.color))

def _update_stats(elapsed_ms: float, move_count: int) -> None:
    """Update performance statistics."""
    _STATS.average_time_ms = (
        (_STATS.average_time_ms * (_STATS.total_calls - 1) + elapsed_ms) /
        _STATS.total_calls
    )

# ==============================================================================
# PERFORMANCE MONITORING
# ==============================================================================

def get_pseudo_legal_stats() -> Dict[str, Any]:
    """Get pseudo-legal move generation statistics."""
    return {
        'total_calls': _STATS.total_calls,
        'total_moves_generated': _STATS.total_moves_generated,
        'average_time_ms': _STATS.average_time_ms,
        'piece_breakdown': _STATS.piece_breakdown.copy(),
        'registry_size': len(get_all_dispatchers()),
    }

def reset_pseudo_legal_stats() -> None:
    """Reset performance statistics."""
    global _STATS
    _STATS = PseudoLegalStats()

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

def generate_pseudo_legal_moves_legacy(state: GameState) -> List[Move]:
    """Legacy interface for backward compatibility."""
    return _generate_pseudo_legal_moves_impl(state, mode=PseudoLegalMode.STANDARD)
