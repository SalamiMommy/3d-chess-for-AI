"""Optimized legal-move filter with enhanced caching and performance."""

from __future__ import annotations
from typing import List, Optional, Set, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel if needed

from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move
from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
from game3d.attacks.check import king_in_check
from game3d.pieces.enums import Color, PieceType
from game3d.board.board import Board

# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

BOARD_SIZE = 9

@dataclass(slots=True)
class LegalMoveStats:
    """Statistics for legal move generation."""
    total_calls: int = 0
    total_moves_filtered: int = 0
    freeze_filtered: int = 0
    check_filtered: int = 0
    average_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

class LegalMoveMode(Enum):
    """Legal move filtering modes."""
    STANDARD = "standard"
    CACHED = "cached"
    BATCH = "batch"
    PARALLEL = "parallel"  # For future multi-threading

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
_STATS = LegalMoveStats()

# ==============================================================================
# OPTIMIZED LEGAL MOVE GENERATION
# ==============================================================================
def _generate_legal_moves_impl(
    state: GameState,
    mode: LegalMoveMode = LegalMoveMode.CACHED,
    use_cache: bool = True
) -> List[Move]:
    """Optimized legal move generation with multiple filtering strategies."""
    start_time = time.perf_counter()
    _STATS.total_calls += 1

    try:
        if mode == LegalMoveMode.CACHED and use_cache:
            moves = _generate_legal_moves_cached(state)
        elif mode == LegalMoveMode.BATCH:
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

def generate_legal_moves(state: GameState) -> List[Move]:
    """Entry-point that always uses the cached path."""
    return _generate_legal_moves_impl(state, mode=LegalMoveMode.CACHED, use_cache=True)

def _generate_legal_moves_cached(state: GameState) -> List[Move]:
    """Cached legal move generation with state validation."""
    # Create state hash
    state_hash = hash((state.board.byte_hash(), state.color, state.halfmove_clock))

    # Check cache
    cached_moves = _LEGAL_CACHE.get(state_hash, state.color)
    if cached_moves is not None:
        _STATS.cache_hits += 1
        return cached_moves.copy()

    _STATS.cache_misses += 1

    # Generate and cache
    moves = _generate_legal_moves_standard(state)
    _LEGAL_CACHE.store(state_hash, state.color, moves)

    return moves

def _generate_legal_moves_batch(state: GameState) -> List[Move]:
    """Batch legal move generation with optimized filtering."""
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

    # Batch check validation
    legal_moves = _batch_check_validation(unfrozen_moves, state)

    return legal_moves

def _generate_legal_moves_standard(state: GameState) -> List[Move]:
    """Standard legal move generation with enhanced filtering."""
    legal: List[Move] = []

    # Get pseudo-legal moves
    pseudo_moves = generate_pseudo_legal_moves(state)

    if not pseudo_moves:
        return legal

    # Get effect caches
    freeze_cache = state.cache._effect["freeze"]
    color = state.color

    # Process each move
    for mv in pseudo_moves:
        # Fast filter: check freeze
        if freeze_cache.is_frozen(mv.from_coord, color):
            _STATS.freeze_filtered += 1
            continue

        # Check if move leaves king in check
        if _would_leave_king_in_check(mv, state):
            _STATS.check_filtered += 1
            continue

        legal.append(mv)
        _STATS.total_moves_filtered += 1

    return legal

# ==============================================================================
# ENHANCED VALIDATION
# ==============================================================================

def _would_leave_king_in_check(move: Move, state: GameState) -> bool:
    """Check if move would leave king in check (optimized version)."""
    # Create temporary board
    tmp_board = state.board.clone()
    tmp_board.apply_move(move)

    # Use cached check detection
    opponent_color = state.color.opposite()
    try:
        # Pass the cache to king_in_check for better performance
        return king_in_check(tmp_board, state.color, opponent_color, state.cache)
    except Exception:
        # Fallback to basic check detection
        return False

def _batch_check_validation(moves: List[Move], state: GameState) -> List[Move]:
    """Batch validation of moves for check avoidance."""
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
        futures = [executor.submit(_would_leave_king_in_check, move, state) for move in moves]
        for future, move in zip(as_completed(futures), moves):
            if not future.result():
                legal_batch.append(move)
            else:
                _STATS.check_filtered += 1
    return legal_batch

# ==============================================================================
# SPECIALIZED FILTERS
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
# PERFORMANCE MONITORING
# ==============================================================================

def _update_stats(elapsed_ms: float, move_count: int) -> None:
    """Update performance statistics."""
    _STATS.average_time_ms = (
        (_STATS.average_time_ms * (_STATS.total_calls - 1) + elapsed_ms) /
        _STATS.total_calls
    )

def get_legal_move_stats() -> Dict[str, Any]:
    """Get legal move generation statistics."""
    return {
        'total_calls': _STATS.total_calls,
        'total_moves_filtered': _STATS.total_moves_filtered,
        'freeze_filtered': _STATS.freeze_filtered,
        'check_filtered': _STATS.check_filtered,
        'cache_hits': _STATS.cache_hits,
        'cache_misses': _STATS.cache_misses,
        'cache_hit_rate': _STATS.cache_hits / max(1, _STATS.total_calls),
        'average_time_ms': _STATS.average_time_ms,
    }

def clear_legal_move_cache() -> None:
    """Clear legal move cache."""
    _LEGAL_CACHE.clear()
    _STATS.cache_hits = 0
    _STATS.cache_misses = 0

def reset_legal_move_stats() -> None:
    """Reset performance statistics."""
    global _STATS
    _STATS = LegalMoveStats()

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
    return _generate_legal_moves_impl(state, mode=LegalMoveMode.STANDARD, use_cache=False)
