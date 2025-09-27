#qame3d/movement/generator.py  (keep the registrar, swap the implementation)
from __future__ import annotations
from typing import Callable, List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from game3d.pieces.enums import PieceType, Color
from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move
from game3d.movement.pseudo_legal import generate_pseudo_legal_moves  # NEW
from game3d.cache.manager import get_cache_manager
from game3d.common.common import X, Y, Z
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

_REGISTRY: Dict[PieceType, Callable[[GameState, int, int, int], List[Move]]] = {}
_MOVE_CACHE: Dict[int, List[Move]] = {}  # Simple cache for repeated positions
_STATS = MoveGenStats()

def register(pt: PieceType):
    """Enhanced decorator with performance tracking."""
    def _decorator(fn: Callable[[GameState, int, int, int], List[Move]]) -> Callable:
        def _wrapped(state: GameState, x: int, y: int, z: int) -> List[Move]:
            _STATS.piece_specific_calls[pt] += 1
            return fn(state, x, y, z)

        _REGISTRY[pt] = _wrapped
        return _wrapped
    return _decorator

def get_dispatcher(pt: PieceType) -> Optional[Callable[[GameState, int, int, int], List[Move]]]:
    """Enhanced lookup with statistics."""
    return _REGISTRY.get(pt)

# ==============================================================================
# OPTIMIZED MOVE GENERATION
# ==============================================================================

def generate_legal_moves(state: GameState) -> List[Move]:
    """Entry-point that always uses the cached path."""
    return _generate_legal_moves_impl(state, mode=MoveGenMode.CACHED, use_cache=True)

def _generate_legal_moves_impl(
    state: GameState,
    mode: MoveGenMode = MoveGenMode.CACHED,
    use_cache: bool = True
) -> List[Move]:
    """Internal implementation – can be safely imported elsewhere."""
    """Optimized legal move generation with multiple strategies."""
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
    """Cached move generation with position hashing."""
    # Create cache key from board state
    cache_key = hash((state.board.byte_hash(), state.current))

    if cache_key in _MOVE_CACHE:
        _STATS.cache_hits += 1
        cached_moves = _MOVE_CACHE[cache_key]

        # Validate cache freshness
        if _is_cache_valid(state, cached_moves):
            return cached_moves.copy()

    _STATS.cache_misses += 1

    # Generate moves and cache them
    moves = _generate_legal_moves_standard(state)
    _MOVE_CACHE[cache_key] = moves.copy()

    # Cache size management - keep only recent entries
    if len(_MOVE_CACHE) > 1000:  # Configurable limit
        _cleanup_move_cache()

    return moves

def _generate_legal_moves_batch(state: GameState) -> List[Move]:
    """Batch move generation for better performance."""
    cache_manager = get_cache_manager()
    legal_moves = []

    # Get all occupied squares for current player
    current_pieces = []
    for coord, piece in state.board.list_occupied():
        if piece.color == state.current:
            current_pieces.append((coord, piece))

    # Process pieces in batches by type
    pieces_by_type: Dict[PieceType, List[Tuple[Coord, PieceType]]] = {}
    for coord, piece in current_pieces:
        if piece.ptype not in pieces_by_type:
            pieces_by_type[piece.ptype] = []
        pieces_by_type[piece.ptype].append((coord, piece.ptype))

    # Generate moves for each piece type
    for piece_type, pieces in pieces_by_type.items():
        dispatcher = get_dispatcher(piece_type)
        if dispatcher:
            for coord, _ in pieces:
                moves = dispatcher(state, coord[X], coord[Y], coord[Z])

                # Apply movement buffs/debuffs
                moves = _apply_movement_modifiers(moves, coord, state, cache_manager)

                # Filter legal moves
                legal_moves.extend(_filter_legal_moves(moves, state))

    return legal_moves

def _generate_legal_moves_standard(state: GameState) -> List[Move]:
    """Standard legal move generation (original implementation)."""
    from game3d.movement.legal import generate_legal_moves as _legal_gen
    return _legal_gen(state)

# ==============================================================================
# MOVEMENT MODIFIERS
# ==============================================================================

def _apply_movement_modifiers(
    moves: List[Move],
    start_sq: Tuple[int, int, int],
    state: GameState,
    cache_manager
) -> List[Move]:
    """Apply movement buffs/debuffs to generated moves."""
    if not moves:
        return moves

    modified_moves = []

    for move in moves:
        # Check for movement buffs
        if cache_manager.is_movement_buffed(start_sq, state.current):
            # Extend movement range for buffed pieces
            extended_moves = _extend_move_range(move, start_sq, state)
            modified_moves.extend(extended_moves)

        # Check for movement debuffs
        elif cache_manager.is_movement_debuffed(start_sq, state.current):
            # Restrict movement range for debuffed pieces
            restricted_move = _restrict_move_range(move, start_sq, state)
            if restricted_move:
                modified_moves.append(restricted_move)

        else:
            modified_moves.append(move)

    return modified_moves

def _extend_move_range(move: Move, start_sq: Tuple[int, int, int], state: GameState) -> List[Move]:
    """Extend movement range for buffed pieces."""
    # Calculate direction vector
    direction = (
        move.to_coord[0] - move.from_coord[0],
        move.to_coord[1] - move.from_coord[1],
        move.to_coord[2] - move.from_coord[2],
    )

    # Normalize direction
    length = max(abs(d) for d in direction)
    if length == 0:
        return [move]

    normalized_dir = tuple(d // length for d in direction)

    # Extend by one step in the same direction
    extended_coord = (
        move.to_coord[0] + normalized_dir[0],
        move.to_coord[1] + normalized_dir[1],
        move.to_coord[2] + normalized_dir[2],
    )

    # Create extended move if valid
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
    # For debuffed pieces, only allow moves within 1 square
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
        # Quick check for basic legality
        if _is_basic_legal(move, state):
            # Deeper check for check avoidance
            if not _leaves_king_in_check(move, state):
                legal_moves.append(move)

    return legal_moves

def _is_basic_legal(move: Move, state: GameState) -> bool:
    """Basic legality checks."""
    # Check if destination is valid
    if not (0 <= move.to_coord[0] < 9 and
            0 <= move.to_coord[1] < 9 and
            0 <= move.to_coord[2] < 9):
        return False

    # Check if destination has friendly piece
    dest_piece = state.board.piece_at(move.to_coord)
    if dest_piece and dest_piece.color == state.current:
        return False

    return True

def _leaves_king_in_check(move: Move, state: GameState) -> bool:
    """Check if move would leave king in check."""
    # Simulate move (clone state)
    temp_state = state.clone()
    temp_state.make_move(move)
    return temp_state.is_check()

# ==============================================================================
# PIECE-SPECIFIC OPTIMIZATIONS
# ==============================================================================

def get_max_steps(piece_type: PieceType, start_sq: Tuple[int, int, int], state: GameState) -> int:
    """Get maximum steps for piece movement with buff/debuff consideration."""
    cache_manager = get_cache_manager()

    base_steps = {
        PieceType.KING: 1,
        PieceType.QUEEN: 8,
        PieceType.ROOK: 8,
        PieceType.BISHOP: 8,
        PieceType.KNIGHT: 1,
        PieceType.PAWN: 2,
        PieceType.HIVE: 2,
        PieceType.ARCHER: 1,  # Archers don't move far
        PieceType.PRIEST: 3,
        PieceType.WHITE_HOLE: 1,
        PieceType.BLACK_HOLE: 1,
    }

    max_steps = base_steps.get(piece_type, 3)

    # Apply movement buffs/debuffs
    if cache_manager.is_movement_buffed(start_sq, state.current):
        max_steps += 1
    elif cache_manager.is_movement_debuffed(start_sq, state.current):
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
        'registry_size': len(_REGISTRY),
        'cache_size': len(_MOVE_CACHE),
    }

def clear_move_cache() -> None:
    """Clear move generation cache."""
    _MOVE_CACHE.clear()
    _STATS.cache_hits = 0
    _STATS.cache_misses = 0

def _cleanup_move_cache() -> None:
    """Cleanup old cache entries (LRU-style)."""
    if len(_MOVE_CACHE) <= 500:  # Keep recent entries
        return

    # Remove oldest entries (simplified - could use proper LRU)
    keys_to_remove = list(_MOVE_CACHE.keys())[:len(_MOVE_CACHE) - 500]
    for key in keys_to_remove:
        del _MOVE_CACHE[key]

def _is_cache_valid(state: GameState, cached_moves: List[Move]) -> bool:
    """Validate if cached moves are still valid for current state."""
    # Simple validation - check if move count matches expected
    # More sophisticated validation could be implemented
    return len(cached_moves) > 0

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

def generate_legal_moves_legacy(state: GameState) -> List[Move]:
    """Legacy interface for backward compatibility."""
    return generate_legal_moves(state, mode=MoveGenMode.STANDARD)

# Maintain original function signature
def generate_legal_moves(state: GameState) -> List[Move]:
    """Full legal moves (pseudo-legal → legality filter)."""
    return generate_legal_moves(state, mode=MoveGenMode.CACHED)
