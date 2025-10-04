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
    N_PIECE_TYPES,      #  ← this was missing
    N_COLOR_PLANES,     #  (and the other new constants, just in case)
    N_TOTAL_PLANES,
    PIECE_SLICE,
    COLOR_SLICE,
)
# --------------------------------------------------------------------

from game3d.pieces.piece import Piece
from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
from game3d.pieces.enums import PieceType, Color, Result
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
def _any_priest_alive(board: BoardProto, king_color: Color | None = None, cache=None) -> bool:
    # Check if we're already in move generation
    if getattr(_thread_local, 'in_move_generation', False):
        # Use a simpler check that doesn't generate moves
        return _simple_priest_check(board, king_color)

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
    # Fallback cache
    if cache is None:
        from game3d.cache.manager import get_cache_manager
        tmp_cache = get_cache_manager(board, attacker_color)
    else:
        tmp_cache = cache

    # 👇 Import GameState ONLY when needed
    from game3d.game.gamestate import GameState
    tmp_state = GameState(board, attacker_color, tmp_cache)

    try:
        from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
        pseudo_moves = generate_pseudo_legal_moves(tmp_state)
    except Exception:
        pseudo_moves = []

    attacked = set()
    for move in pseudo_moves:
        attacked.add(move.to_coord)

    if cache and hasattr(cache, 'store_attacked_squares'):
        cache.store_attacked_squares(attacker_color, attacked)

    _stats.attack_calculations += 1
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

def _is_king_attacked_directly(
    board: BoardProto,
    king_pos: Tuple[int, int, int],
    attacker_color: Color,
    cache=None
) -> bool:
    """Check if king is attacked without generating moves."""
    for coord, piece in board.list_occupied():
        if piece.color != attacker_color:
            continue

        # Check if this piece can attack the king
        if _can_piece_attack_square(piece, coord, king_pos, board):
            return True
    return False

def king_in_check(
    board: BoardProto,
    current_player: Color,
    king_color: Color,
    cache=None
) -> bool:
    # Fast path: if priests are alive, no check
    if _any_priest_alive(board, king_color, cache):
        return False

    # Find king position
    king_pos = _find_king_position(board, king_color, cache)
    if king_pos is None:
        return False

    # Use direct attack calculation instead of move generation
    return _is_king_attacked_directly(board, king_pos, king_color.opposite(), cache)

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

def _can_piece_attack_square(piece: Piece, from_coord: Coord, to_coord: Coord, board: BoardProto) -> bool:
    """Check if a non-sliding piece can attack a square."""
    from game3d.movement.movepiece import Move

    # Create a dummy move
    dummy_move = Move(from_coord=from_coord, to_coord=to_coord, is_capture=True)

    # Get the dispatcher for this piece type
    from game3d.movement.registry import get_dispatcher
    dispatcher = get_dispatcher(piece.ptype)
    if dispatcher is None:
        return False

    # Create temporary state
    from game3d.game.gamestate import GameState
    tmp_state = GameState.__new__(GameState)
    tmp_state.board = board
    tmp_state.color = piece.color
    # Use minimal cache or None
    tmp_state.cache = getattr(board, 'cache', None)

    try:
        # Generate moves for this piece
        moves = dispatcher(tmp_state, *from_coord)
        # Check if our target square is in the generated moves
        for move in moves:
            if move.to_coord == to_coord:
                return True
    except Exception:
        pass

    return False

def _can_sliding_piece_attack_king(piece: Piece, from_coord: Coord, king_pos: Coord,
                                  board: BoardProto, piece_cache) -> bool:
    """Check if a sliding piece can attack the king along a clear ray."""
    # Check if king and piece are on same line
    if not _is_on_same_ray(from_coord, king_pos):
        return False

    # Check if ray is clear
    if not _is_ray_clear(from_coord, king_pos, board, piece_cache):
        return False

    # If ray is clear and on same line, piece can attack
    return True

def _is_on_same_ray(from_coord: Coord, to_coord: Coord) -> bool:
    """Check if two coordinates are on the same orthogonal, diagonal, or space diagonal ray."""
    fx, fy, fz = from_coord
    tx, ty, tz = to_coord

    dx = tx - fx
    dy = ty - fy
    dz = tz - fz

    # Same square
    if dx == dy == dz == 0:
        return False

    # Orthogonal: only one non-zero delta
    if (dx != 0) + (dy != 0) + (dz != 0) == 1:
        return True

    # Diagonal: two non-zero deltas with same absolute value
    non_zero = [d for d in (dx, dy, dz) if d != 0]
    if len(non_zero) == 2 and abs(non_zero[0]) == abs(non_zero[1]):
        return True

    # Space diagonal: all three non-zero with same absolute value
    if len(non_zero) == 3 and abs(dx) == abs(dy) == abs(dz):
        return True

    return False

def _is_ray_clear(from_coord: Coord, to_coord: Coord, board: BoardProto, piece_cache) -> bool:
    """Check if the ray between from_coord and to_coord is clear of pieces."""
    fx, fy, fz = from_coord
    tx, ty, tz = to_coord

    dx = tx - fx
    dy = ty - fy
    dz = tz - fz

    # Get step directions
    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    step_z = 0 if dz == 0 else (1 if dz > 0 else -1)

    # Number of steps
    steps = max(abs(dx), abs(dy), abs(dz))

    # Check each square along the ray (excluding endpoints)
    x, y, z = fx, fy, fz
    for _ in range(steps - 1):
        x += step_x
        y += step_y
        z += step_z

        if piece_cache is not None:
            piece = piece_cache.get((x, y, z))
        else:
            piece = board.piece_at((x, y, z))

        if piece is not None:
            return False

    return True

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

def _simple_priest_check(board: BoardProto, king_color: Color | None = None) -> bool:
    """Simple priest check that doesn't generate moves."""
    for coord, piece in board.list_occupied():
        if piece.ptype == PieceType.PRIEST:
            if king_color is None or piece.color == king_color:
                return True
    return False

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
