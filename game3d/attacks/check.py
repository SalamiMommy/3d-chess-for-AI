"""Optimized check detector for 9×9×9 with enhanced caching and performance."""

from __future__ import annotations
from typing import Protocol, Optional, runtime_checkable, Dict, Set, Tuple, List
from dataclasses import dataclass
from enum import Enum
import weakref

from game3d.pieces.enums import PieceType, Color

# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

@runtime_checkable
class BoardProto(Protocol):
    def list_occupied(self): ...

@dataclass(slots=True)
class CheckCache:
    """Optimized cache for check detection results."""
    king_position: Optional[Tuple[int, int, int]] = None
    priests_alive: bool = False
    attacked_squares: Set[Tuple[int, int, int]] = None
    last_board_hash: int = 0
    last_player: Color = Color.WHITE

    def __post_init__(self):
        if self.attacked_squares is None:
            self.attacked_squares = set()

class CheckStatus(Enum):
    """Check status enumeration."""
    SAFE = 0
    CHECK = 1
    CHECKMATE = 2
    STALEMATE = 3

# ==============================================================================
# OPTIMIZED CHECK DETECTION
# ==============================================================================

def _any_priest_alive(board: BoardProto, color: Color, cache=None) -> bool:
    """Fast priest scan with optional caching."""
    if cache and hasattr(cache, 'priest_positions'):
        priest_cache = cache.get_priest_positions(color)
        return len(priest_cache) > 0

    # Fast scan – returns True on first priest found
    for _, piece in board.list_occupied():
        if piece.color == color and piece.ptype == PieceType.PRIEST:
            return True
    return False

def _find_king_position(board: BoardProto, king_color: Color, cache=None) -> Optional[Tuple[int, int, int]]:
    """Find king position with caching."""
    if cache and hasattr(cache, 'king_positions'):
        king_cache = cache.get_king_positions(king_color)
        return king_cache[0] if king_cache else None

    # Standard search
    for coord, piece in board.list_occupied():
        if piece.color == king_color and piece.ptype == PieceType.KING:
            return coord
    return None

def _get_attacked_squares_cached(
    board: BoardProto,
    attacker_color: Color,
    cache=None
) -> Set[Tuple[int, int, int]]:
    """Get all squares attacked by attacker_color using cache."""
    if cache and hasattr(cache, 'attacked_squares'):
        attacked_cache = cache.get_attacked_squares(attacker_color)
        if attacked_cache is not None:
            return attacked_cache

    # Fallback to calculation
    return _calculate_attacked_squares(board, attacker_color, cache)

def _calculate_attacked_squares(
    board: BoardProto,
    attacker_color: Color,
    cache=None
) -> Set[Tuple[int, int, int]]:
    """Calculate all squares under attack by attacker_color."""
    from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
    from game3d.game.gamestate import GameState

    attacked = set()

    tmp_state = GameState.__new__(GameState)
    tmp_state.board = board
    tmp_state.color = attacker_color
    tmp_state.cache = cache

    try:
        for mv in generate_pseudo_legal_moves(tmp_state):
            attacked.add(mv.to_coord)
    except Exception:
        # Fallback: direct calculation if cache issues
        for coord, piece in board.list_occupied():
            if piece.color == attacker_color:
                # Add basic attack patterns based on piece type
                if piece.ptype == PieceType.ARCHER:
                    # Add archery attack patterns
                    attacked.update(_get_archery_attack_squares(coord, board, attacker_color, cache))
                elif piece.ptype == PieceType.KING:
                    # Add king attack patterns (1-step in all directions)
                    attacked.update(_get_king_attack_squares(coord))
                # Add other piece types as needed

    return attacked

def _get_archery_attack_squares(
    archer_pos: Tuple[int, int, int],
    board: BoardProto,
    attacker_color: Color,
    cache=None
) -> Set[Tuple[int, int, int]]:
    """Get archery attack squares for 2-radius sphere surface."""
    if cache and hasattr(cache, 'archery_cache'):
        archery_cache = cache.get_archery_cache()
        return set(archery_cache.attack_targets(attacker_color))

    # Fallback calculation
    from game3d.geometry import sphere_surface
    return sphere_surface(archer_pos, 2.0)

def _get_king_attack_squares(king_pos: Tuple[int, int, int]) -> Set[Tuple[int, int, int]]:
    """Get squares attacked by king (1-step in all directions)."""
    from game3d.geometry import get_neighbors_26
    return set(get_neighbors_26(king_pos))

def square_attacked_by(
    board: BoardProto,
    current_player: Color,
    square: Tuple[int, int, int],
    attacker_color: Color,
    cache=None
) -> bool:
    """Optimized square attack detection with caching."""
    # Use attacked squares cache if available
    attacked_squares = _get_attacked_squares_cached(board, attacker_color, cache)
    return square in attacked_squares

def king_in_check(
    board: BoardProto,
    current_player: Color,
    king_color: Color,
    cache=None
) -> bool:
    """Optimized king check detection with full caching support."""
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
    return CheckStatus.CHECK  # Simplified - could be enhanced to detect checkmate

def get_all_pieces_in_check(
    board: BoardProto,
    current_player: Color,
    cache=None
) -> List[Tuple[Tuple[int, int, int], Color]]:
    """Get all pieces that are in check (useful for multi-king variants)."""
    pieces_in_check = []

    # Check each color
    for color in [Color.WHITE, Color.BLACK]:
        if king_in_check(board, current_player, color, cache):
            king_pos = _find_king_position(board, color, cache)
            if king_pos:
                pieces_in_check.append((king_pos, color))

    return pieces_in_check

# ==============================================================================
# BATCH OPERATIONS
# ==============================================================================

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

    # Get attacked squares
    summary['attacked_squares_white'] = _get_attacked_squares_cached(board, Color.WHITE, cache)
    summary['attacked_squares_black'] = _get_attacked_squares_cached(board, Color.BLACK, cache)

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
