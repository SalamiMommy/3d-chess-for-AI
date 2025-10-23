# game3d/attacks/check.py - CORRECTED KEY FUNCTIONS
"""
This shows the corrected versions of key functions that were accessing caches directly.
The full file is too long, but these are the main corrections needed.
"""

from __future__ import annotations
from typing import Protocol, Optional, Dict, Set, Tuple, List, Any
from dataclasses import dataclass
from enum import Enum

from game3d.common.coord_utils import Coord
from game3d.pieces.piece import Piece
from game3d.common.enums import PieceType, Color

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
def _get_cache_from_board(board) -> Optional[Any]:
    """Try to get cache manager from board if available."""
    if hasattr(board, 'cache_manager'):
        return board.cache_manager
    return None

def _any_priest_alive(
    board,
    king_color: Color | None = None,
    cache: Any | None = None,
) -> bool:
    """
    Return True if *king_color* still has at least one priest on the board.
    If *king_color* is None -> True if ANY priest exists.
    Delegates to the manager's occupancy cache counters.
    """
    if cache is None:
        cache = _get_cache_from_board(board)

    if cache is None:
        # Ultra-defensive fallback
        return False

    # CORRECTED: Use manager method
    if king_color is None:
        return cache.any_priest_alive()
    return cache.has_priest(king_color)

def _find_king_position(
    board,
    king_color: Color,
    cache=None,
) -> Optional[Coord]:
    """
    Fast king lookup using the occupancy cache through manager.
    Returns (x, y, z) or None.
    """
    if cache is None:
        cache = _get_cache_from_board(board)

    if cache is None:
        return None

    # CORRECTED: Access through manager
    return cache.occupancy.find_king(king_color)

def _get_attacked_squares_from_move_cache(
    board,
    attacker_color: Color,
    cache=None
) -> Set[Tuple[int, int, int]]:
    """Get all squares attacked by attacker_color using move cache through manager."""
    if cache and hasattr(cache, 'move'):
        # CORRECTED: Access through manager
        return cache.move.get_attacked_squares(attacker_color)

    # Fallback
    return _calculate_attacked_squares_fallback(board, attacker_color, cache)

def _calculate_attacked_squares_fallback(
    board,
    attacker_color: Color,
    cache=None
) -> Set[Tuple[int, int, int]]:
    """Fallback calculation of attacked squares when move cache isn't available."""
    attacked: Set[Tuple[int, int, int]] = set()

    if cache is None:
        cache = _get_cache_from_board(board)

    if cache is None:
        return attacked

    # CORRECTED: Access through manager
    for coord, piece in cache.occupancy.iter_color(attacker_color):
        moves = _generate_piece_moves(board, coord, piece, cache)
        for move in moves:
            attacked.add(move.to_coord)

    return attacked

def _generate_piece_moves(
    board,
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

    # CORRECTED: Cache is the manager
    if cache is not None:
        tmp_state.cache = cache
    else:
        from game3d.cache.manager import get_cache_manager
        tmp_state.cache = get_cache_manager(board, piece.color)

    try:
        return dispatcher(tmp_state, *coord)
    except Exception:
        return []

def square_attacked_by(
    board,
    current_player: Color,
    square: Tuple[int, int, int],
    attacker_color: Color,
    cache=None
) -> bool:
    """Optimized square attack detection using move cache through manager."""
    if cache is None:
        cache = _get_cache_from_board(board)

    # CORRECTED: Use manager method
    attacked_squares = _get_attacked_squares_from_move_cache(board, attacker_color, cache)
    return square in attacked_squares

def king_in_check(
    board,
    current_player: Color,
    king_color: Color,
    cache=None
) -> bool:
    """Check if king is in check using move cache through manager."""
    if cache is None:
        cache = _get_cache_from_board(board)

    # Fast path: if priests are alive, no check
    if _any_priest_alive(board, king_color, cache):
        return False

    # Find king position
    king_pos = _find_king_position(board, king_color, cache)
    if king_pos is None:
        return False

    # Check if king is attacked
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

# ---------------------------------------------------------------------------
#  1.  ANY PRIEST ALIVE  (was line 46)
# ---------------------------------------------------------------------------
def _any_priest_alive(board, king_color: Color | None = None, cache: Any | None = None) -> bool:
    if cache is None:
        cache = _get_cache_from_board(board)

    # CORRECT - through manager
    if cache is not None:
        if king_color is None:
            return cache.any_priest_alive()
        return cache.has_priest(king_color)

    # Ultra-defensive fallback (should never fire)
    occ = getattr(board, "occupancy", None)
    if occ is not None:
        for _, piece in occ.iter_color(king_color or Color.WHITE):
            if piece.ptype == PieceType.PRIEST:
                return True
    return False


# ---------------------------------------------------------------------------
#  2.  FIND KING POSITION  (was line 67)
# ---------------------------------------------------------------------------
def _find_king_position(board, king_color: Color, cache=None) -> Optional[Coord]:
    if cache is None:
        cache = _get_cache_from_board(board)

    # CORRECT - through manager's occupancy
    if cache is not None:
        return cache.occupancy.find_king(king_color)
    return None
# ---------------------------------------------------------------------------
#  3.  FALLBACK ATTACKED-SQUARES  (was line 111)
# ---------------------------------------------------------------------------
def _calculate_attacked_squares_fallback(
    board: BoardProto,
    attacker_color: Color,
    cache=None
) -> Set[Tuple[int, int, int]]:
    """Fallback calculation of attacked squares when move cache isn't available."""
    attacked: Set[Tuple[int, int, int]] = set()

    # Get all pieces of the attacker colour via occupancy cache
    occ = getattr(board, "occupancy", None)
    if occ is None:                      # should never happen
        return attacked

    for coord, piece in occ.iter_color(attacker_color):
        moves = _generate_piece_moves(board, coord, piece, cache)
        for move in moves:
            attacked.add(move.to_coord)

    _stats.attack_calculations += 1
    return attacked


# ---------------------------------------------------------------------------
#  4.  GET CHECK SUMMARY  (was line 215)
# ---------------------------------------------------------------------------
def get_check_summary(
    board,
    cache=None
) -> Dict[str, Any]:
    """Get comprehensive check information."""
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

    if cache is None:
        return summary

    # CORRECTED: Use manager methods
    summary['white_priests_alive'] = cache.has_priest(Color.WHITE)
    summary['black_priests_alive'] = cache.has_priest(Color.BLACK)

    # CORRECTED: Access through manager
    summary['white_king_position'] = cache.occupancy.find_king(Color.WHITE)
    summary['black_king_position'] = cache.occupancy.find_king(Color.BLACK)

    # Attacked squares from move cache (through manager)
    summary['attacked_squares_white'] = _get_attacked_squares_from_move_cache(board, Color.WHITE, cache)
    summary['attacked_squares_black'] = _get_attacked_squares_from_move_cache(board, Color.BLACK, cache)

    # Check flags
    wk = summary['white_king_position']
    bk = summary['black_king_position']
    if wk and not summary['white_priests_alive']:
        summary['white_check'] = wk in summary['attacked_squares_black']
    if bk and not summary['black_priests_alive']:
        summary['black_check'] = bk in summary['attacked_squares_white']

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
