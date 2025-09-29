from __future__ import annotations
from typing import Callable, List, Dict, Any, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel validation
from game3d.pieces.enums import PieceType, Color
if TYPE_CHECKING:
    from game3d.game.gamestate import GameState   # only for mypy/IDE
import game3d.game.gamestate as _gs              # module object at runtime
from game3d.movement.movepiece import Move
from game3d.common.common import X, Y, Z, Coord, in_bounds
from game3d.movement.registry import register, get_dispatcher, get_all_dispatchers
from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
from game3d.attacks.check import king_in_check, get_check_summary

# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

BOARD_SIZE = 9

@dataclass(slots=True)
class MoveGenStats:
    """Statistics for move generation performance."""
    total_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
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
    CACHED = "cached"
    BATCH = "batch"
    PARALLEL = "parallel"  # For multi-threading in validation

# ==============================================================================
# ENHANCED CACHING SYSTEM
# ==============================================================================

class LegalMoveCache:
    """Optimized cache for legal moves with validation."""

    __slots__ = ("_cache", "_last_state_hash", "_last_color", "_validation_count")

    def __init__(self):
        self._cache: Dict[int, List[Move]] = {}
        self._last_state_hash: int = 0
        self._last_color: Color = Color.WHITE
        self._validation_count: int = 0

    def get(self, state_hash: int, color: Color) -> Optional[List[Move]]:
        """Get cached legal moves if valid."""
        if state_hash == self._last_state_hash and color == self._last_color:
            return self._cache.get(state_hash)
        return None

    def store(self, state_hash: int, color: Color, moves: List[Move]) -> None:
        """Store legal moves in cache."""
        self._last_state_hash = state_hash
        self._last_color = color
        self._validation_count = len(moves)
        self._cache[state_hash] = moves.copy()

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._last_state_hash = 0
        self._last_color = Color.WHITE
        self._validation_count = 0

# Global cache instance
_LEGAL_CACHE = LegalMoveCache()
_STATS = MoveGenStats()

# ==============================================================================
# OPTIMIZED LEGAL MOVE GENERATION
# ==============================================================================
def _generate_legal_moves_impl(
    state: GameState,
    mode: MoveGenMode = MoveGenMode.CACHED,
    use_cache: bool = True
) -> List[Move]:
    """Internal implementation – can be safely imported elsewhere."""
    start_time = time.perf_counter()
    _STATS.total_calls += 1

    try:
        if mode == MoveGenMode.CACHED and use_cache:
            moves = _generate_legal_moves_cached(state)
        elif mode == MoveGenMode.BATCH:
            moves = _generate_legal_moves_batch(state)
        elif mode == MoveGenMode.PARALLEL:
            moves = _generate_legal_moves_parallel(state)
        else:
            moves = _generate_legal_moves_standard(state)

        # Update statistics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _update_stats(elapsed_ms, len(moves))

        return moves

    except Exception as e:
        # Fallback to standard generation
        return _generate_legal_moves_standard(state)

def generate_legal_moves(state: GameState) -> List[Move]:
    """Entry-point that always uses the cached path."""
    return _generate_legal_moves_impl(state, mode=MoveGenMode.CACHED, use_cache=True)

def _generate_legal_moves_cached(state: GameState) -> List[Move]:
    """Cached move generation with position hashing - CORRECTED."""
    # Use state.color consistently
    state_hash = hash((state.board.byte_hash(), state.color, state.halfmove_clock))

    if state_hash in _LEGAL_CACHE._cache:
        _STATS.cache_hits += 1
        return _LEGAL_CACHE._cache[state_hash].copy()

    _STATS.cache_misses += 1

    moves = _generate_legal_moves_standard(state)
    _LEGAL_CACHE.store(state_hash, state.color, moves)

    if len(_LEGAL_CACHE._cache) > 1000:
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

def _generate_legal_moves_parallel(state: GameState) -> List[Move]:
    """Parallel legal move generation with freeze and check filtering."""
    # Get pseudo-legal moves
    pseudo_legal_moves = generate_pseudo_legal_moves(state)

    if not pseudo_legal_moves:
        return []

    # Pre-filter frozen pieces
    freeze_cache = state.cache._effect["freeze"]
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
    legal_moves = _batch_check_validation(unfrozen_moves, state)

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

    # Group moves by validation type for efficiency
    king_moves = []
    capture_moves = []
    regular_moves = []

    # First pass: basic legality and categorization
    for move in moves:
        if not _is_basic_legal(move, state):
            continue

        # Categorize by move type for optimized validation order
        king_pos = check_summary[f'{state.color.name.lower()}_king_position']
        if move.from_coord == king_pos:
            king_moves.append(move)
        elif move.is_capture:
            capture_moves.append(move)
        else:
            regular_moves.append(move)

    # Validate in order of likelihood to be legal (most restrictive first)
    # This reduces the number of expensive check validations

    # Regular moves (usually most numerous, often legal)
    for move in regular_moves:
        if not _leaves_king_in_check_optimized(move, state, check_summary):
            legal_moves.append(move)

    # Capture moves (need to check if they resolve checks)
    for move in capture_moves:
        if not _leaves_king_in_check_optimized(move, state, check_summary):
            legal_moves.append(move)

    # King moves (always need special validation)
    for move in king_moves:
        if not _leaves_king_in_check_optimized(move, state, check_summary):
            legal_moves.append(move)

    return legal_moves

def _is_basic_legal(move: Move, state: GameState) -> bool:
    """Basic legality checks - CORRECTED."""
    if not (0 <= move.to_coord[0] < 9 and
            0 <= move.to_coord[1] < 9 and
            0 <= move.to_coord[2] < 9):
        return False

    dest_piece = state.cache.piece_cache.get(move.to_coord)
    if dest_piece and dest_piece.color == state.color:
        return False

    return True

def _leaves_king_in_check_optimized(move: Move, state: GameState, check_summary: Dict[str, Any]) -> bool:
    """Optimized check validation using pre-computed position state."""

    king_color = state.color
    king_pos = check_summary[f'{king_color.name.lower()}_king_position']
    if not king_pos:
        return False

    # Case 1: Moving the king - check destination safety
    if move.from_coord == king_pos:
        attacked_squares = check_summary[f'attacked_squares_{king_color.opposite().name.lower()}']
        return move.to_coord in attacked_squares

    # Case 2: In check - must block or capture
    if check_summary[f'{king_color.name.lower()}_check']:
        # For now, fall back to full validation for complex check scenarios
        # You can optimize this further with incremental check resolution
        return _leaves_king_in_check(move, state)  # Your existing method

    # Case 3: Check for discovered attacks (pinned pieces)
    if state.cache.is_pinned(move.from_coord):
        pin_direction = state.cache.get_pin_direction(move.from_coord)
        if not _along_pin_line(move, pin_direction):
            return True

    # Fast case: no immediate check concerns
    return False

def _leaves_king_in_check(move: Move, state: GameState) -> bool:
    # Lazy import to break circular dependency
    from game3d.attacks.check import king_in_check

    temp_state = state.clone()
    temp_state.make_move(move)
    return king_in_check(temp_state.board, state.color, state.color.opposite(), temp_state.cache)


def _blocks_check(move: Move, king_pos: Coord, checker_pos: Coord) -> bool:
    """Check if move blocks the check ray."""
    # Simple ray intersection test
    return _is_between(move.to_coord, king_pos, checker_pos)

def _along_pin_line(move: Move, pin_direction: Tuple[int, int, int]) -> bool:
    """Check if move stays along pin line."""
    move_direction = (
        move.to_coord[0] - move.from_coord[0],
        move.to_coord[1] - move.from_coord[1],
        move.to_coord[2] - move.from_coord[2]
    )

    # Normalize directions for comparison
    def normalize_direction(dx, dy, dz):
        length = max(abs(dx), abs(dy), abs(dz))
        if length == 0:
            return (0, 0, 0)
        return (dx // length, dy // length, dz // length)

    return normalize_direction(*move_direction) == normalize_direction(*pin_direction)

def _batch_check_validation(moves: List[Move], state: GameState) -> List[Move]:
    """Batch validation of moves for check avoidance using parallelism."""
    legal_moves = []

    # Process moves in batches for better cache utilization
    batch_size = 50  # Configurable batch size

    for i in range(0, len(moves), batch_size):
        batch = moves[i:i + batch_size]
        batch_results = _validate_move_batch(batch, state)
        legal_moves.extend(batch_results)

    return legal_moves

def _validate_move_batch(moves: List[Move], state: GameState) -> List[Move]:
    legal_batch = []
    with ThreadPoolExecutor(max_workers=4) as executor:  # Configurable
        # NOTE: Assumes state is not mutated during threading for compatibility
        futures = [executor.submit(_leaves_king_in_check, move, state) for move in moves]
        for future, move in zip(as_completed(futures), moves):
            if not future.result():
                legal_batch.append(move)
            else:
                _STATS.check_filtered += 1
    return legal_batch

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
    freeze_cache = state.cache._effect["freeze"]
    color = state.color

    return [
        mv for mv in pseudo_moves
        if not freeze_cache.is_frozen(mv.from_coord, color)
    ]

def generate_legal_moves_for_piece(state: GameState, coord: Tuple[int, int, int]) -> List[Move]:
    """Generate legal moves only for a specific piece."""
    # Get all legal moves
    all_legal = generate_legal_moves(state)

    # Filter for specific piece
    return [mv for mv in all_legal if mv.from_coord == coord]

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
        'cache_hits': _STATS.cache_hits,
        'cache_misses': _STATS.cache_misses,
        'cache_hit_rate': _STATS.cache_hits / max(1, _STATS.total_calls),
        'average_time_ms': _STATS.average_time_ms,
        'piece_specific_calls': _STATS.piece_specific_calls.copy(),
        'registry_size': len(get_all_dispatchers()),
        'cache_size': len(_LEGAL_CACHE._cache),
        'total_moves_filtered': _STATS.total_moves_filtered,
        'freeze_filtered': _STATS.freeze_filtered,
        'check_filtered': _STATS.check_filtered,
    }

def clear_move_cache() -> None:
    """Clear move generation cache."""
    _LEGAL_CACHE.clear()
    _STATS.cache_hits = 0
    _STATS.cache_misses = 0

def _cleanup_move_cache() -> None:
    """Cleanup old cache entries (LRU-style)."""
    if len(_LEGAL_CACHE._cache) <= 500:
        return

    keys_to_remove = list(_LEGAL_CACHE._cache.keys())[:len(_LEGAL_CACHE._cache) - 500]
    for key in keys_to_remove:
        del _LEGAL_CACHE._cache[key]

def reset_move_gen_stats() -> None:
    """Reset performance statistics."""
    global _STATS
    _STATS = MoveGenStats()

# ==============================================================================
# ENHANCED CACHING STRATEGIES
# ==============================================================================

class IncrementalLegalCache:
    """Incremental cache for legal moves (future enhancement)."""

    def __init__(self):
        self.base_moves: Dict[Tuple[int, int, int], List[Move]] = {}
        self.delta_moves: Dict[str, List[Move]] = {}

    def update_from_delta(self, move: Move, state: GameState) -> None:
        """Update cache incrementally based on last move."""
        # This would implement true incremental updating
        # For now, placeholder for future enhancement
        pass

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

def generate_legal_moves_legacy(state: GameState) -> List[Move]:
    """Legacy interface for backward compatibility."""
    return _generate_legal_moves_impl(state, mode=MoveGenMode.STANDARD)
