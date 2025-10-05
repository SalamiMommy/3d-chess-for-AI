# generator.py
from __future__ import annotations
from typing import Callable, List, Dict, Any, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel validation
from threading import local
from game3d.pieces.enums import PieceType, Color
if TYPE_CHECKING:
    from game3d.game.gamestate import GameState   # only for mypy/IDE
from game3d.movement.movepiece import Move
from game3d.common.common import Coord, in_bounds
from game3d.movement.registry import register, get_dispatcher, get_all_dispatchers
from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
from game3d.attacks.check import king_in_check, get_check_summary
from game3d.movement.validation import (  # NEW IMPORT
    is_basic_legal, leaves_king_in_check, leaves_king_in_check_optimized,
    resolves_check, batch_check_validation, validate_move_batch
)
import sys
from functools import wraps

def recursion_limit(max_depth):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            depth = getattr(wrapper, 'depth', 0)
            if depth >= max_depth:
                # Fallback to simpler implementation
                return _generate_legal_moves_fallback(*args, **kwargs)
            wrapper.depth = depth + 1
            try:
                result = func(*args, **kwargs)
            finally:
                wrapper.depth = depth
            return result
        wrapper.depth = 0
        return wrapper
    return decorator


def _generate_legal_moves_fallback(state: GameState) -> List[Move]:
    """Simple fallback move generation when recursion is detected."""
    legal_moves = []
    current_color = state.color

    for coord, piece in state.board.list_occupied():
        if piece.color != current_color:
            continue

        # Only generate basic moves without complex validation
        dispatcher = get_dispatcher(piece.ptype)
        if dispatcher:
            moves = dispatcher(state, coord[0], coord[1], coord[2])
            for move in moves:
                # Only basic bounds and destination checks
                if (0 <= move.to_coord[0] < 9 and
                    0 <= move.to_coord[1] < 9 and
                    0 <= move.to_coord[2] < 9):
                    # Fix: Use occupancy cache instead of piece_cache
                    dest_piece = state.cache.occupancy.get(move.to_coord)
                    if not dest_piece or dest_piece.color != current_color:
                        legal_moves.append(move)
    return legal_moves
# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================
X, Y, Z = 0, 1, 2
BOARD_SIZE = 9

# REMOVED: _is_between function (moved to validation.py)

@dataclass(slots=True)
class MoveGenStats:
    """Statistics for move generation performance."""
    total_calls: int = 0
    piece_specific_calls: Dict[PieceType, int] = None
    average_time_ms: float = 0.0
    total_moves_filtered: int = 0
    freeze_filtered: int = 0
    check_filtered: int = 0

    def __post_init__(self):
        if self.piece_specific_calls is None:
            self.piece_specific_calls = {pt: 0 for pt in PieceType}

class MoveGenMode(Enum):
    """Move generation modes for different optimization levels."""
    STANDARD = "standard"
    BATCH = "batch"
    PARALLEL = "parallel"  # For multi-threading in validation

# ==============================================================================
# OPTIMIZED LEGAL MOVE GENERATION
# ==============================================================================
_STATS = MoveGenStats()

_thread_local = local()

def _generate_legal_moves_impl(
    state: GameState,
    mode: MoveGenMode = MoveGenMode.STANDARD
) -> List[Move]:
    # Check for recursion
    if getattr(_thread_local, 'in_move_generation', False):
        return _generate_legal_moves_fallback(state)

    _thread_local.in_move_generation = True
    try:
        """Internal implementation - NO RECURSION for large board games."""
        start_time = time.perf_counter()
        _STATS.total_calls += 1

        # Choose generation strategy based on mode and board complexity
        if mode == MoveGenMode.BATCH:
            moves = _generate_legal_moves_batch(state)
        elif mode == MoveGenMode.PARALLEL:
            # For very dense boards, batch might be better than parallel due to overhead
            if len([p for p in state.board.list_occupied()]) > 400:
                moves = _generate_legal_moves_batch(state)
            else:
                moves = _generate_legal_moves_parallel(state)
        else:  # STANDARD mode
            moves = _generate_legal_moves_standard(state)

        # Update performance statistics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _update_stats(elapsed_ms, len(moves))

        return moves

    except Exception as e:
        # For large boards, fallback to most reliable method
        print(f"Error in {mode.value} move generation: {e}, falling back to batch mode")
        return _generate_legal_moves_batch(state)
    finally:
        _thread_local.in_move_generation = False

def generate_legal_moves(state: GameState) -> List[Move]:
    """Entry-point that always uses the standard path."""
    return _generate_legal_moves_impl(state, mode=MoveGenMode.STANDARD)

def _generate_legal_moves_batch(state: GameState) -> List[Move]:
    """Batch move generation - CORRECTED."""
    # Validate coordinates first
    for coord, piece in state.board.list_occupied():
        assert len(coord) == 3 and all(0 <= c < 9 for c in coord), coord

    cache_manager = state.cache
    legal_moves = []

    current_pieces = []
    for coord, piece in state.board.list_occupied():
        if piece.color == state.color:
            current_pieces.append((coord, piece))

    pieces_by_type: Dict[PieceType, List[Tuple[Coord, Any]]] = defaultdict(list)
    for coord, piece in current_pieces:
        pieces_by_type[piece.ptype].append((coord, piece))

    for piece_type, pieces in pieces_by_type.items():
        dispatcher = get_dispatcher(piece_type)
        if dispatcher:
            for coord, piece in pieces:
                # Unpack coordinates properly
                moves = dispatcher(state, coord[0], coord[1], coord[2])

                # Apply movement modifiers
                moves = _apply_movement_modifiers(moves, coord, state, cache_manager)

                # Filter legal moves
                legal_moves.extend(_filter_legal_moves(moves, state))

    return legal_moves

def _generate_legal_moves_parallel(state: GameState) -> List[Move]:
    """Parallel legal move generation with freeze and check filtering."""
    # Get pseudo-legal moves
    pseudo_legal_moves = generate_pseudo_legal_moves(state)

    if not pseudo_legal_moves:
        return []

    # Pre-filter frozen pieces
    freeze_cache = state.cache.effects["freeze"]
    color = state.color

    # Filter out frozen pieces first (fast operation)
    unfrozen_moves = [
        mv for mv in pseudo_legal_moves
        if not freeze_cache.is_frozen(mv.from_coord, color)
    ]

    _STATS.freeze_filtered += len(pseudo_legal_moves) - len(unfrozen_moves)

    if not unfrozen_moves:
        return []

    # Batch check validation in parallel
    legal_moves = batch_check_validation(unfrozen_moves, state)  # UPDATED

    return legal_moves

def _generate_legal_moves_standard(state: GameState) -> List[Move]:
    """Standard legal move generation with optimized filtering."""
    pseudo_moves = generate_pseudo_legal_moves(state)
    return _filter_legal_moves(pseudo_moves, state)  # Now uses batch version

# ==============================================================================
# MOVEMENT MODIFIERS
# ==============================================================================
def _apply_movement_modifiers(
    moves: List[Move],
    start_sq: Tuple[int, int, int],
    state: GameState,
    cache_manager=None
) -> List[Move]:
    """Apply movement buffs/debuffs - CORRECTED."""
    if cache_manager is None:
        cache_manager = state.cache
    if not moves:
        return moves

    modified_moves = []

    for move in moves:
        if cache_manager.is_movement_buffed(start_sq, state.color):  # Fixed
            extended_moves = _extend_move_range(move, start_sq, state)
            modified_moves.extend(extended_moves)
        elif (hasattr(cache_manager, "is_movement_debuffed") and  # Added hasattr check
              cache_manager.is_movement_debuffed(start_sq, state.color)):  # Fixed
            restricted_move = _restrict_move_range(move, start_sq, state)
            if restricted_move:
                modified_moves.append(restricted_move)
        else:
            modified_moves.append(move)

    return modified_moves

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

    if (0 <= extended_coord[0] < 9 and
        0 <= extended_coord[1] < 9 and
        0 <= extended_coord[2] < 9):
        extended_move = Move(
            from_coord=move.from_coord,
            to_coord=extended_coord,
            is_capture=move.is_capture,
            metadata={**move.metadata, 'extended': True}
        )
        extended_moves.append(extended_move)

    return extended_moves

def _restrict_move_range(move: Move, start_sq: Tuple[int, int, int], state: GameState) -> Optional[Move]:
    """Restrict movement range for debuffed pieces."""
    distance = max(
        abs(move.to_coord[0] - move.from_coord[0]),
        abs(move.to_coord[1] - move.from_coord[1]),
        abs(move.to_coord[2] - move.from_coord[2])
    )

    if distance <= 1:
        return move

    return None

# ==============================================================================
# LEGAL MOVE FILTERING
# ==============================================================================
def _filter_legal_moves(moves: List[Move], state: GameState) -> List[Move]:
    """Optimized batch legal move filtering with incremental validation."""
    if not moves:
        return moves

    # Get position state ONCE for all moves
    check_summary = get_check_summary(state.board, state.cache)
    legal_moves = []

    # Pre-compute attacked squares for efficiency
    attacked_squares = check_summary[f'attacked_squares_{state.color.opposite().name.lower()}']
    king_pos = check_summary[f'{state.color.name.lower()}_king_position']
    in_check = check_summary[f'{state.color.name.lower()}_check']

    for move in moves:
        # Basic legality check
        if not is_basic_legal(move, state):  # UPDATED
            continue

        # Fast check for king moves
        if move.from_coord == king_pos:
            if move.to_coord in attacked_squares:
                continue
        # If in check, only allow moves that resolve the check
        elif in_check:
            if not resolves_check(move, state, check_summary):  # UPDATED
                continue

        legal_moves.append(move)

    return legal_moves

# REMOVED: _is_basic_legal, _leaves_king_in_check_optimized, _leaves_king_in_check,
# _blocks_check, _along_pin_line, _batch_check_validation, _validate_move_batch functions

# ==============================================================================
# PIECE-SPECIFIC OPTIMIZATIONS
# ==============================================================================
def get_max_steps(piece_type: PieceType, start_sq: Tuple[int, int, int], state: GameState) -> int:
    """Get maximum steps for piece movement - CORRECTED."""
    cache_manager = state.cache

    base_steps = {
        PieceType.KING: 1,
        PieceType.QUEEN: 8,
        PieceType.ROOK: 8,
        PieceType.BISHOP: 8,
        PieceType.KNIGHT: 1,
        PieceType.PAWN: 2,
        PieceType.HIVE: 1,
        PieceType.ARCHER: 1,
        PieceType.PRIEST: 1,
        PieceType.WHITE_HOLE: 1,
        PieceType.BLACK_HOLE: 1,
        PieceType.WALL: 1,  # Added WALL with 0 steps
    }

    max_steps = base_steps.get(piece_type, 3)

    if cache_manager.is_movement_buffed(start_sq, state.color):  # Fixed
        max_steps += 1
    elif (hasattr(cache_manager, "is_movement_debuffed") and  # Added hasattr check
          cache_manager.is_movement_debuffed(start_sq, state.color)):  # Fixed
        max_steps = max(1, max_steps - 1)

    return max_steps

# ==============================================================================
# SPECIALIZED GENERATORS
# ==============================================================================

def generate_legal_moves_excluding_checks(state: GameState) -> List[Move]:
    """Generate moves without check validation (for performance)."""
    pseudo_moves = generate_pseudo_legal_moves(state)

    # Only apply basic filters
    freeze_cache = state.cache.effects["freeze"]
    color = state.color

    return [
        mv for mv in pseudo_moves
        if not freeze_cache.is_frozen(mv.from_coord, color)
    ]

def generate_legal_moves_for_piece(state: GameState, coord: Tuple[int, int, int]) -> List[Move]:
    """Generate legal moves only for a specific piece."""
    piece = state.cache.piece_cache.get(coord) if hasattr(state.cache, 'piece_cache') else state.cache.occupancy.get(coord)
    if not piece or piece.color != state.color:
        return []

    dispatcher = get_dispatcher(piece.ptype)
    if not dispatcher:
        return []

    moves = dispatcher(state, coord[0], coord[1], coord[2])
    return _filter_legal_moves(moves, state)

def generate_legal_captures(state: GameState) -> List[Move]:
    """Generate only legal capturing moves."""
    all_legal = generate_legal_moves(state)
    return [mv for mv in all_legal if mv.is_capture]

def generate_legal_non_captures(state: GameState) -> List[Move]:
    """Generate only legal non-capturing moves."""
    all_legal = generate_legal_moves(state)
    return [mv for mv in all_legal if not mv.is_capture]

# ==============================================================================
# STATISTICS AND MONITORING
# ==============================================================================

def _update_stats(elapsed_ms: float, move_count: int) -> None:
    """Update performance statistics."""
    _STATS.average_time_ms = (
        (_STATS.average_time_ms * (_STATS.total_calls - 1) + elapsed_ms) /
        _STATS.total_calls
    )

def get_move_generation_stats() -> Dict[str, Any]:
    """Get move generation performance statistics."""
    return {
        'total_calls': _STATS.total_calls,
        'average_time_ms': _STATS.average_time_ms,
        'piece_specific_calls': _STATS.piece_specific_calls.copy(),
        'registry_size': len(get_all_dispatchers()),
        'total_moves_filtered': _STATS.total_moves_filtered,
        'freeze_filtered': _STATS.freeze_filtered,
        'check_filtered': _STATS.check_filtered,
    }

def reset_move_gen_stats() -> None:
    """Reset performance statistics."""
    global _STATS
    _STATS = MoveGenStats()

# REMOVED: _resolves_check function (moved to validation.py)

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

def generate_legal_moves_legacy(state: GameState) -> List[Move]:
    """Legacy interface for backward compatibility."""
    return _generate_legal_moves_impl(state, mode=MoveGenMode.STANDARD)
