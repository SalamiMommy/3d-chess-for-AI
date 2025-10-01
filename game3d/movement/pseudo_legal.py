"""Optimized pseudo-legal move generator with enhanced caching and performance."""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict  # Added for pieces_by_type
if TYPE_CHECKING:
    from game3d.game.gamestate import GameState   # only for mypy/IDE
from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.movement.registry import register, get_dispatcher, get_all_dispatchers, dispatch_batch
# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

BOARD_SIZE = 9  # Extracted

@dataclass(slots=True)
class PseudoLegalStats:
    """Statistics for pseudo-legal move generation."""
    total_calls: int = 0
    total_moves_generated: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    piece_breakdown: Dict[PieceType, int] = None
    average_time_ms: float = 0.0

    def __post_init__(self):
        if self.piece_breakdown is None:
            self.piece_breakdown = {pt: 0 for pt in PieceType}

class PseudoLegalMode(Enum):
    STANDARD   = "standard"
    CACHED     = "cached"
    BATCH      = "batch"        # <-- new
    INCREMENTAL= "incremental"

# ==============================================================================
# ENHANCED CACHING SYSTEM
# ==============================================================================

class PseudoLegalCache:
    """Optimized cache for pseudo-legal moves."""

    __slots__ = ("_cache", "_last_board_hash", "_last_color", "_move_count")

    def __init__(self):
        self._cache: Dict[int, List[Move]] = {}
        self._last_board_hash: int = 0
        self._last_color: Color = Color.WHITE
        self._move_count: int = 0

    def get(self, board_hash: int, color: Color) -> Optional[List[Move]]:
        """Get cached moves if available and valid."""
        if board_hash == self._last_board_hash and color == self._last_color:
            return self._cache.get(board_hash)
        return None

    def store(self, board_hash: int, color: Color, moves: List[Move]) -> None:
        """Store moves in cache."""
        self._last_board_hash = board_hash
        self._last_color = color
        self._move_count = len(moves)
        self._cache[board_hash] = moves.copy()

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._last_board_hash = 0
        self._last_color = Color.WHITE
        self._move_count = 0

    def is_valid(self, board_hash: int, color: Color) -> bool:
        """Check if cache entry is valid."""
        return (board_hash == self._last_board_hash and
                color == self._last_color and
                board_hash in self._cache)

# Global cache instance
_PSEUDO_LEGAL_CACHE = PseudoLegalCache()
_STATS = PseudoLegalStats()

# ==============================================================================
# OPTIMIZED PSEUDO-LEGAL GENERATION
# ==============================================================================
def _generate_pseudo_legal_moves_impl(
    state: GameState,
    mode: PseudoLegalMode = PseudoLegalMode.CACHED,
    use_cache: bool = True
) -> List[Move]:
    """Optimized pseudo-legal move generation with multiple strategies."""
    start_time = time.perf_counter()
    _STATS.total_calls += 1

    try:
        if mode == PseudoLegalMode.CACHED and use_cache:
            moves = _generate_pseudo_legal_cached(state)
        elif mode == PseudoLegalMode.BATCH:
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
    """Entry-point that always uses the cached path."""
    return _generate_pseudo_legal_moves_impl(state, mode=PseudoLegalMode.CACHED, use_cache=True)

def _generate_pseudo_legal_cached(state: GameState) -> List[Move]:
    """Cached pseudo-legal move generation."""
    board_hash = state.board.byte_hash()

    # Check cache
    cached_moves = _PSEUDO_LEGAL_CACHE.get(board_hash, state.color)
    if cached_moves is not None:
        _STATS.cache_hits += 1
        return cached_moves.copy()

    _STATS.cache_misses += 1

    # Generate and cache
    moves = _generate_pseudo_legal_standard(state)
    _PSEUDO_LEGAL_CACHE.store(board_hash, state.color, moves)

    return moves

def _generate_pseudo_legal_batch(state: GameState) -> list[Move]:
    """Batch pseudo-legal move generation – now *really* batched."""
    coords, types = [], []

    for coord, piece in state.board.list_occupied():
        if piece.color == state.color:
            coords.append(coord)
            types.append(piece.ptype)

    return dispatch_batch(state, coords, types, state.color)

def _generate_pseudo_legal_incremental(state: GameState) -> List[Move]:
    """Incremental pseudo-legal move generation (for small changes)."""
    # This would use incremental updates from cache
    # For now, fall back to cached version
    return _generate_pseudo_legal_cached(state)

def _generate_pseudo_legal_standard(state: GameState) -> List[Move]:
    """Standard pseudo-legal move generation - CORRECTED."""
    all_moves: List[Move] = []

    for coord, piece in state.board.list_occupied():
        if piece.color != state.color:  # This is correct, not state.current
            continue

        dispatcher = get_dispatcher(piece.ptype)
        if dispatcher is None:
            continue

        try:
            piece_moves = dispatcher(state, *coord)
            validated_moves = _validate_piece_moves(piece_moves, coord, piece, state)

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
    state: GameState
) -> List[Move]:
    """Enhanced validation for piece-generated moves."""
    if not moves:
        return moves

    validated_moves = []

    for move in moves:
        # Basic validation
        if not _is_move_valid(move, expected_coord, piece, state):
            continue

        # Additional validation based on piece type
        if not _is_move_legal_for_piece_type(move, piece, state):
            continue

        validated_moves.append(move)

    return validated_moves

def _is_move_valid(
    move: Move,
    expected_coord: Tuple[int, int, int],
    piece,
    state: GameState
) -> bool:
    """Basic move validation."""
    # Check coordinate consistency
    if move.from_coord != expected_coord:
        return False

    # Check bounds
    to_x, to_y, to_z = move.to_coord
    if not (0 <= to_x < BOARD_SIZE and 0 <= to_y < BOARD_SIZE and 0 <= to_z < BOARD_SIZE):
        return False

    # Check destination piece
    dest_piece = state.cache.piece_cache.get(move.to_coord)
    if dest_piece and dest_piece.color == piece.color:
        return False

    return True

def _is_move_legal_for_piece_type(move: Move, piece, state: GameState) -> bool:
    """Piece-type specific move validation - CORRECTED."""
    cache_manager = state.cache

    # Check freeze effects first (highest priority)
    if cache_manager.is_frozen(move.from_coord, piece.color):
        return False

    # Check movement buffs
    if cache_manager.is_movement_buffed(move.from_coord, piece.color):
        pass  # Allow extended movement

    # Check movement debuffs with proper hasattr check
    elif (hasattr(cache_manager, "is_movement_debuffed") and
          cache_manager.is_movement_debuffed(move.from_coord, piece.color)):
        distance = max(
            abs(move.to_coord[0] - move.from_coord[0]),
            abs(move.to_coord[1] - move.from_coord[1]),
            abs(move.to_coord[2] - move.from_coord[2])
        )
        if distance > 1:
            return False

    return True

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def _get_current_player_pieces(state: GameState) -> List[Tuple[Tuple[int, int, int], Any]]:
    """Get all pieces for current player - CORRECTED."""
    current_pieces = []

    for coord, piece in state.board.list_occupied():
        if piece.color == state.color:  # This is correct
            current_pieces.append((coord, piece))

    return current_pieces

def _estimate_move_count(state: GameState) -> int:
    """Estimate number of moves for pre-allocation."""
    # Rough estimate based on piece count
    piece_count = sum(1 for _, piece in state.board.list_occupied() if piece.color == state.color)
    return piece_count * 15  # Average 15 moves per piece

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
        'cache_hits': _STATS.cache_hits,
        'cache_misses': _STATS.cache_misses,
        'cache_hit_rate': _STATS.cache_hits / max(1, _STATS.total_calls),
        'average_time_ms': _STATS.average_time_ms,
        'piece_breakdown': _STATS.piece_breakdown.copy(),
        'registry_size': len(get_all_dispatchers()),
        'cache_size': len(_PSEUDO_LEGAL_CACHE._cache),
    }

def clear_pseudo_legal_cache() -> None:
    """Clear pseudo-legal move cache."""
    _PSEUDO_LEGAL_CACHE.clear()
    _STATS.cache_hits = 0
    _STATS.cache_misses = 0

def reset_pseudo_legal_stats() -> None:
    """Reset performance statistics."""
    global _STATS
    _STATS = PseudoLegalStats()

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

def generate_pseudo_legal_moves_legacy(state: GameState) -> List[Move]:
    """Legacy interface for backward compatibility."""
    return _generate_pseudo_legal_moves_impl(state, mode=PseudoLegalMode.STANDARD, use_cache=False)
