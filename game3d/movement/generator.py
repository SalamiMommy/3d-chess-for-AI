from __future__ import annotations
from typing import Callable, List, Dict, Any, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict
from game3d.pieces.enums import PieceType, Color
if TYPE_CHECKING:
    from game3d.game.gamestate import GameState   # only for mypy/IDE
import game3d.game.gamestate as _gs              # module object at runtime
from game3d.movement.movepiece import Move
from game3d.common.common import X, Y, Z, Coord, in_bounds
from game3d.movement.registry import register, get_dispatcher, get_all_dispatchers

# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

@dataclass(slots=True)
class MoveGenStats:
    """Statistics for move generation performance."""
    total_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    piece_specific_calls: Dict[PieceType, int] = None
    average_time_ms: float = 0.0

    def __post_init__(self):
        if self.piece_specific_calls is None:
            self.piece_specific_calls = {pt: 0 for pt in PieceType}

class MoveGenMode(Enum):
    """Move generation modes for different optimization levels."""
    STANDARD = "standard"
    CACHED = "cached"
    BATCH = "batch"
    PARALLEL = "parallel"  # For future multi-threading

# ==============================================================================
# ENHANCED REGISTRY SYSTEM
# ==============================================================================


_MOVE_CACHE: Dict[int, List[Move]] = {}  # Simple cache for repeated positions
_STATS = MoveGenStats()

def generate_legal_moves(state: GameState) -> List[Move]:
    """Entry-point that always uses the cached path."""
    return _generate_legal_moves_impl(state, mode=MoveGenMode.CACHED, use_cache=True)

def _generate_legal_moves_impl(
    state: GameState,
    mode: MoveGenMode = MoveGenMode.CACHED,
    use_cache: bool = True
) -> List[Move]:
    """Internal implementation – can be safely imported elsewhere."""
    start_time = time.perf_counter()
    _STATS.total_calls += 1

    try:
        if mode == MoveGenMode.CACHED:
            moves = _generate_legal_moves_cached(state)
        elif mode == MoveGenMode.BATCH:
            moves = _generate_legal_moves_batch(state)
        else:
            moves = _generate_legal_moves_standard(state)

        # Update statistics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _update_stats(elapsed_ms, len(moves))

        return moves

    except Exception as e:
        # Fallback to standard generation
        return _generate_legal_moves_standard(state)

def _generate_legal_moves_cached(state: GameState) -> List[Move]:
    """Cached move generation with position hashing - CORRECTED."""
    # Use state.color consistently
    cache_key = hash((state.board.byte_hash(), state.color))

    if cache_key in _MOVE_CACHE:
        _STATS.cache_hits += 1
        return _MOVE_CACHE[cache_key].copy()

    _STATS.cache_misses += 1

    moves = _generate_legal_moves_standard(state)
    _MOVE_CACHE[cache_key] = moves.copy()

    if len(_MOVE_CACHE) > 1000:
        _cleanup_move_cache()

    return moves

def _generate_legal_moves_batch(state: GameState) -> List[Move]:
    """Batch move generation - CORRECTED."""
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
                # Pass state to dispatcher
                moves = dispatcher(state, coord[X], coord[Y], coord[Z])

                # Apply movement modifiers
                moves = _apply_movement_modifiers(moves, coord, state, cache_manager)

                # Filter legal moves
                legal_moves.extend(_filter_legal_moves(moves, state))

    return legal_moves

def _generate_legal_moves_standard(state: GameState) -> List[Move]:
    # Lazy import to break circular dependency
    from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
    pseudo_moves = generate_pseudo_legal_moves(state)
    return _filter_legal_moves(pseudo_moves, state)

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
    """Filter pseudo-legal moves to ensure they don't leave king in check."""
    if not moves:
        return moves

    legal_moves = []

    for move in moves:
        if _is_basic_legal(move, state):
            if not _leaves_king_in_check(move, state):
                legal_moves.append(move)

    return legal_moves

def _is_basic_legal(move: Move, state: GameState) -> bool:
    """Basic legality checks - CORRECTED."""
    if not (0 <= move.to_coord[0] < 9 and
            0 <= move.to_coord[1] < 9 and
            0 <= move.to_coord[2] < 9):
        return False

    dest_piece = state.cache.piece_cache.get(move.to_coord)
    if dest_piece and dest_piece.color == state.color:  # Fixed from state.current
        return False

    return True

def _leaves_king_in_check(move: Move, state: GameState) -> bool:
    # Lazy import to break circular dependency
    from game3d.attacks.check import king_in_check

    temp_state = state.clone()
    temp_state.make_move(move)
    return king_in_check(temp_state.board, state.color, state.color.opposite(), temp_state.cache)

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
        'cache_hits': _STATS.cache_hits,
        'cache_misses': _STATS.cache_misses,
        'cache_hit_rate': _STATS.cache_hits / max(1, _STATS.total_calls),
        'average_time_ms': _STATS.average_time_ms,
        'piece_specific_calls': _STATS.piece_specific_calls.copy(),
        'registry_size': len(get_all_dispatchers()),
        'cache_size': len(_MOVE_CACHE),
    }

def clear_move_cache() -> None:
    """Clear move generation cache."""
    _MOVE_CACHE.clear()
    _STATS.cache_hits = 0
    _STATS.cache_misses = 0

def _cleanup_move_cache() -> None:
    """Cleanup old cache entries (LRU-style)."""
    if len(_MOVE_CACHE) <= 500:
        return

    keys_to_remove = list(_MOVE_CACHE.keys())[:len(_MOVE_CACHE) - 500]
    for key in keys_to_remove:
        del _MOVE_CACHE[key]

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

def generate_legal_moves_legacy(state: GameState) -> List[Move]:
    """Legacy interface for backward compatibility."""
    return _generate_legal_moves_impl(state, mode=MoveGenMode.STANDARD)


