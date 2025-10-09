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
from game3d.movement.pseudo_legal import generate_pseudo_legal_moves, generate_pseudo_legal_moves_for_piece
from game3d.attacks.check import king_in_check, get_check_summary
from game3d.movement.validation import (
    leaves_king_in_check,
    resolves_check,
    filter_legal_moves
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
    pseudo_moves = generate_pseudo_legal_moves(state)
    return filter_legal_moves(pseudo_moves, state)
# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================
X, Y, Z = 0, 1, 2
BOARD_SIZE = 9

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
    """Batch move generation - CORRECTED. Relies on pseudo-legal for iteration and basics."""
    pseudo_moves = generate_pseudo_legal_moves(state)
    return filter_legal_moves(pseudo_moves, state)  # UPDATED: use validation function

def _generate_legal_moves_parallel(state: GameState) -> List[Move]:
    """Parallel legal move generation with check filtering (freeze/modifiers handled in pseudo)."""
    pseudo_legal_moves = generate_pseudo_legal_moves(state)

    if not pseudo_legal_moves:
        return []

    # Use filter_legal_moves instead of batch_check_validation
    legal_moves = filter_legal_moves(pseudo_legal_moves, state)

    return legal_moves

def _generate_legal_moves_standard(state: GameState) -> List[Move]:
    """Standard legal move generation with optimized filtering."""
    pseudo_moves = generate_pseudo_legal_moves(state)
    return filter_legal_moves(pseudo_moves, state)  # UPDATED: use validation function

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
    return generate_pseudo_legal_moves(state)  # Modifiers/freeze already applied

def generate_legal_moves_for_piece(state: GameState, coord: Tuple[int, int, int]) -> List[Move]:
    """Generate legal moves only for a specific piece."""
    pseudo_moves = generate_pseudo_legal_moves_for_piece(state, coord)
    return filter_legal_moves(pseudo_moves, state)  # UPDATED: use validation function

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

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

def generate_legal_moves_legacy(state: GameState) -> List[Move]:
    """Legacy interface for backward compatibility."""
    return _generate_legal_moves_impl(state, mode=MoveGenMode.STANDARD)
