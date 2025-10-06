# game3d/attacks/check.py
from __future__ import annotations
from typing import Protocol, Optional, runtime_checkable, Dict, Set, Tuple, List, Any
from dataclasses import dataclass
from enum import Enum
import weakref
from typing import Iterable
from threading import local

# -----------  bring in the constants that live in common  -----------
from game3d.common.common import (
    Coord,
    in_bounds,
    N_PIECE_TYPES,
    N_COLOR_PLANES,
    N_TOTAL_PLANES,
    PIECE_SLICE,
    COLOR_SLICE,
)
# --------------------------------------------------------------------

from game3d.pieces.piece import Piece
from game3d.pieces.enums import PieceType, Color, Result
from game3d.movement.movepiece import MOVE_FLAGS

# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================
_thread_local = local()

@runtime_checkable
class BoardProto(Protocol):
    def list_occupied(self) -> Iterable[Tuple[Coord, Piece]]: ...
    def piece_at(self, coord: Coord) -> Optional[Piece]: ...

@dataclass(slots=True)
class CheckCache:
    """Optimized cache for check detection results."""
    king_position: Optional[Tuple[int, int, int]] = None
    priests_alive: bool = False
    last_board_hash: int = 0
    last_player: Color = Color.WHITE

class CheckStatus(Enum):
    """Check status enumeration."""
    SAFE = 0
    CHECK = 1
    CHECKMATE = 2
    STALEMATE = 3

# ==============================================================================
# OPTIMIZED CHECK DETECTION USING MOVE CACHE
# ==============================================================================
def _get_cache_from_board(board: BoardProto) -> Optional[Any]:
    """Try to get cache manager from board if available."""
    if hasattr(board, 'cache_manager'):
        return board.cache_manager
    return None

def _any_priest_alive(board: BoardProto, king_color: Color | None = None, cache=None) -> bool:
    """Check if any priest is alive for the given color."""
    # Fast path: if we're already in move generation, use simple check
    if getattr(_thread_local, 'in_move_generation', False):
        return _simple_priest_check(board, king_color)

    # Ensure we have a cache
    if cache is None:
        cache = _get_cache_from_board(board)

    # Use cache if available
    if cache and hasattr(cache, 'move'):
        # Check if we have priest counts cached
        if hasattr(cache.move, '_priest_count'):
            if king_color is None:
                return any(count > 0 for count in cache.move._priest_count.values())
            return cache.move._priest_count.get(king_color, 0) > 0

    # Fallback to simple check
    return _simple_priest_check(board, king_color)

def _simple_priest_check(board: BoardProto, king_color: Color | None = None) -> bool:
    """Simple priest check that doesn't generate moves."""
    for coord, piece in board.list_occupied():
        if piece.ptype == PieceType.PRIEST:
            if king_color is None or piece.color == king_color:
                return True
    return False

def _find_king_position(board: BoardProto, king_color: Color, cache=None) -> Optional[Tuple[int, int, int]]:
    """Find king position with caching."""
    # Ensure we have a cache
    if cache is None:
        cache = _get_cache_from_board(board)

    # Try move cache first
    if cache and hasattr(cache, 'move'):
        king_pos = cache.move._king_pos.get(king_color)
        if king_pos:
            # Verify the king is still there
            piece = board.piece_at(king_pos)
            if piece and piece.color == king_color and piece.ptype == PieceType.KING:
                return king_pos

    # Standard search
    for coord, piece in board.list_occupied():
        if piece.color == king_color and piece.ptype == PieceType.KING:
            # Update cache if available
            if cache and hasattr(cache, 'move'):
                cache.move._king_pos[king_color] = coord
            return coord
    return None

def _get_attacked_squares_from_move_cache(
    board: BoardProto,
    attacker_color: Color,
    cache=None
) -> Set[Tuple[int, int, int]]:
    """Get all squares attacked by attacker_color using move cache."""
    # Ensure we have a cache
    if cache is None:
        cache = _get_cache_from_board(board)

    if cache and hasattr(cache, 'move'):
        # Use the move cache to get attacked squares
        return cache.move.get_attacked_squares(attacker_color)

    # Fallback to calculation (should rarely happen)
    return _calculate_attacked_squares_fallback(board, attacker_color, cache)

def _calculate_attacked_squares_fallback(
    board: BoardProto,
    attacker_color: Color,
    cache=None
) -> Set[Tuple[int, int, int]]:
    """Fallback calculation of attacked squares when move cache isn't available."""
    attacked = set()

    # Get all pieces of the attacker color
    for coord, piece in board.list_occupied():
        if piece.color != attacker_color:
            continue

        # Generate moves for this piece
        moves = _generate_piece_moves(board, coord, piece, cache)
        for move in moves:
            attacked.add(move.to_coord)

    _stats.attack_calculations += 1
    return attacked

def _generate_piece_moves(
    board: BoardProto,
    coord: Coord,
    piece: Piece,
    cache=None
) -> List['Move']:
    """Generate moves for a single piece."""
    from game3d.movement.registry import get_dispatcher

    dispatcher = get_dispatcher(piece.ptype)
    if dispatcher is None:
        return []

    # Create minimal state for move generation
    from game3d.game.gamestate import GameState
    tmp_state = GameState.__new__(GameState)
    tmp_state.board = board
    tmp_state.color = piece.color

    # Try to use cache if available
    if cache is not None:
        tmp_state.cache = cache
    else:
        # Create a minimal cache if none provided
        from game3d.cache.manager import get_cache_manager
        tmp_state.cache = get_cache_manager(board, piece.color)

    try:
        return dispatcher(tmp_state, *coord)
    except Exception:
        return []

def square_attacked_by(
    board: BoardProto,
    current_player: Color,
    square: Tuple[int, int, int],
    attacker_color: Color,
    cache=None
) -> bool:
    """Optimized square attack detection using move cache."""
    # Ensure we have a cache
    if cache is None:
        cache = _get_cache_from_board(board)

    # Use attacked squares from move cache
    attacked_squares = _get_attacked_squares_from_move_cache(board, attacker_color, cache)
    return square in attacked_squares

def king_in_check(
    board: BoardProto,
    current_player: Color,
    king_color: Color,
    cache=None
) -> bool:
    """Check if king is in check using move cache."""
    # Ensure we have a cache
    if cache is None:
        cache = _get_cache_from_board(board)

    # Fast path: if priests are alive, no check
    if _any_priest_alive(board, king_color, cache):
        return False

    # Find king position
    king_pos = _find_king_position(board, king_color, cache)
    if king_pos is None:
        return False

    # Check if king is attacked using move cache
    return square_attacked_by(board, current_player, king_pos, king_color.opposite(), cache)

def get_check_status(
    board: BoardProto,
    current_player: Color,
    king_color: Color,
    cache=None
) -> CheckStatus:
    """Get detailed check status including checkmate detection."""
    # Ensure we have a cache
    if cache is None:
        cache = _get_cache_from_board(board)

    # Quick check for priests
    if _any_priest_alive(board, king_color, cache):
        return CheckStatus.SAFE

    # Find king
    king_pos = _find_king_position(board, king_color, cache)
    if king_pos is None:
        return CheckStatus.SAFE

    # Check if in check
    if not square_attacked_by(board, current_player, king_pos, king_color.opposite(), cache):
        return CheckStatus.SAFE

    # In check - determine if checkmate
    # This would require legal move generation to check for escape moves
    # For now, we'll return CHECK and let the caller determine checkmate
    return CheckStatus.CHECK

def get_all_pieces_in_check(
    board: BoardProto,
    current_player: Color,
    cache=None
) -> List[Tuple[Tuple[int, int, int], Color]]:
    """Get all pieces that are in check (useful for multi-king variants)."""
    # Ensure we have a cache
    if cache is None:
        cache = _get_cache_from_board(board)

    pieces_in_check = []

    # Check each color
    for color in [Color.WHITE, Color.BLACK]:
        if king_in_check(board, current_player, color, cache):
            king_pos = _find_king_position(board, color, cache)
            if king_pos:
                pieces_in_check.append((king_pos, color))

    return pieces_in_check

def batch_king_check_detection(
    boards: List[BoardProto],
    players: List[Color],
    king_colors: List[Color],
    cache=None
) -> List[bool]:
    """Batch check detection for multiple positions."""
    results = []

    for board, player, king_color in zip(boards, players, king_colors):
        results.append(king_in_check(board, player, king_color, cache))

    return results

def get_check_summary(
    board: BoardProto,
    cache=None
) -> Dict[str, any]:
    """Get comprehensive check information."""
    # Ensure we have a cache
    if cache is None:
        cache = _get_cache_from_board(board)

    summary = {
        'white_check': False,
        'black_check': False,
        'white_priests_alive': False,
        'black_priests_alive': False,
        'white_king_position': None,
        'black_king_position': None,
        'attacked_squares_white': set(),
        'attacked_squares_black': set(),
    }

    # Check priests
    summary['white_priests_alive'] = _any_priest_alive(board, Color.WHITE, cache)
    summary['black_priests_alive'] = _any_priest_alive(board, Color.BLACK, cache)

    # Find kings
    summary['white_king_position'] = _find_king_position(board, Color.WHITE, cache)
    summary['black_king_position'] = _find_king_position(board, Color.BLACK, cache)

    # Get attacked squares from move cache
    summary['attacked_squares_white'] = _get_attacked_squares_from_move_cache(board, Color.WHITE, cache)
    summary['attacked_squares_black'] = _get_attacked_squares_from_move_cache(board, Color.BLACK, cache)

    # Check king status
    if summary['white_king_position'] and not summary['white_priests_alive']:
        summary['white_check'] = summary['white_king_position'] in summary['attacked_squares_black']

    if summary['black_king_position'] and not summary['black_priests_alive']:
        summary['black_check'] = summary['black_king_position'] in summary['attacked_squares_white']

    return summary

# ==============================================================================
# PERFORMANCE MONITORING
# ==============================================================================

class CheckDetectorStats:
    """Statistics for check detection performance."""

    def __init__(self):
        self.total_checks = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.king_searches = 0
        self.priest_scans = 0
        self.attack_calculations = 0

    def get_stats(self) -> Dict[str, int]:
        return {
            'total_checks': self.total_checks,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(1, self.total_checks),
            'king_searches': self.king_searches,
            'priest_scans': self.priest_scans,
            'attack_calculations': self.attack_calculations,
        }

# Global stats instance
_stats = CheckDetectorStats()

def get_check_detector_stats() -> Dict[str, int]:
    """Get check detector performance statistics."""
    return _stats.get_stats()

def reset_check_detector_stats() -> None:
    """Reset performance statistics."""
    global _stats
    _stats = CheckDetectorStats()
