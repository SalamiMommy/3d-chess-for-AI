from __future__ import annotations
from typing import Protocol, Optional, Dict, List, Any, NamedTuple, TYPE_CHECKING, Union
from enum import IntEnum
import numpy as np
from numba import njit, prange

from game3d.common.shared_types import (
    PieceType, Color, SIZE, COLOR_WHITE, COLOR_BLACK, COLOR_EMPTY,
    COORD_DTYPE, BOOL_DTYPE, INDEX_DTYPE, MAX_COORD_VALUE, MIN_COORD_VALUE
)
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.attacks.movepiece import Move
    from game3d.cache.manager import CacheManager

Coord = np.ndarray  # Shape: (3,)

class CheckStatus(IntEnum):
    """Check status for game state evaluation."""
    SAFE = 0
    CHECK = 1
    CHECKMATE = 2
    STALEMATE = 3

class KingInCheckInfo(NamedTuple):
    """Information about king in check."""
    king_coord: np.ndarray
    color: int

def _get_priest_count(board, king_color: Optional[Color] = None, cache: Any = None) -> int:
    """Get count of priests for the given king color."""
    cache = cache or getattr(board, 'cache_manager', None)
    if cache is None:
        from game3d.cache.manager import get_cache_manager
        cache = get_cache_manager(board, king_color or Color.WHITE)

    if cache is not None and hasattr(cache, 'has_priest'):
        # Check if any priest exists for this color
        return 0 if not cache.has_priest(king_color) else 1
    return 0


def _find_king_position(board, king_color: int, cache=None) -> Optional[np.ndarray]:
    """Find king position using cache manager's occupancy cache exclusively."""
    cache = cache or getattr(board, 'cache_manager', None)

    # Use cache manager's occupancy cache - this is the primary and only method
    if cache is not None and hasattr(cache, 'occupancy_cache') and hasattr(cache.occupancy_cache, 'find_king'):
        king_pos = cache.occupancy_cache.find_king(king_color)
        if king_pos is not None:
            return king_pos.astype(COORD_DTYPE)
    
    # If cache manager or occupancy cache is not available, raise an error
    # This ensures check detection always uses the optimized cache-based approach
    if cache is None:
        raise RuntimeError("Cache manager not available for king position lookup")
    elif not hasattr(cache, 'occupancy_cache'):
        raise RuntimeError("Cache manager does not have occupancy cache for king position lookup")
    elif not hasattr(cache.occupancy_cache, 'find_king'):
        raise RuntimeError("Occupancy cache does not have find_king method for king position lookup")
    
    return None


def _get_attacked_squares_from_move_cache(board, attacker_color: int, cache=None) -> np.ndarray:
    """Get attacked squares using move cache - calculates from cached moves."""
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)

    # Use move cache for optimal attack calculation
    if cache and hasattr(cache, 'move_cache'):
        cached_moves = cache.move_cache.get_cached_moves(attacker_color)
        if cached_moves is None or len(cached_moves) == 0:
            return mask

        # Extract destination coordinates from cached moves (MOVE_DTYPE: [from_x, from_y, from_z, to_x, to_y, to_z])
        for move in cached_moves:
            # Moves are numpy arrays with columns: [from_x, from_y, from_z, to_x, to_y, to_z]
            to_x, to_y, to_z = int(move[3]), int(move[4]), int(move[5])
            if (0 <= to_x < SIZE and 0 <= to_y < SIZE and 0 <= to_z < SIZE):
                # Use (x, y, z) indexing as per occupancycache.py architecture
                mask[to_x, to_y, to_z] = True

        return mask

    return mask


def _generate_piece_moves(board, coord: np.ndarray, piece: np.ndarray, cache=None) -> np.ndarray:
    """Generate moves for a piece using cache manager.

    This function leverages the move cache for optimal performance.
    Returns numpy array of moves in MOVE_DTYPE format.
    """
    cache = cache or getattr(board, 'cache_manager', None)

    if cache is None or not hasattr(cache, 'move_cache'):
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Get cached moves from move cache
    cached_moves = cache.move_cache.get_cached_moves(piece["color"])
    if cached_moves is None or len(cached_moves) == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Filter moves for this specific piece (moves are numpy arrays: [from_x, from_y, from_z, to_x, to_y, to_z])
    piece_moves = []
    for move in cached_moves:
        # Compare from coordinates (columns 0, 1, 2)
        if np.array_equal(move[:3], coord):
            piece_moves.append(move)

    return np.array(piece_moves) if piece_moves else np.empty((0, 6), dtype=COORD_DTYPE)

def square_attacked_by(board, current_player: Color, square: np.ndarray, attacker_color: int, cache=None) -> bool:
    """Check if a square is attacked by a specific color."""
    square = square.astype(COORD_DTYPE)

    # Vectorized bounds checking
    if not in_bounds_vectorized(square.reshape(1, -1))[0]:
        return False

    attacked_mask = _get_attacked_squares_from_move_cache(board, attacker_color, cache)
    x, y, z = square[0], square[1], square[2]
    # Use (x, y, z) indexing as per occupancycache.py architecture
    return bool(attacked_mask[x, y, z])

def king_in_check(board, current_player: Color, king_color: int, cache=None) -> bool:
    """Check if king is in check - only when king has 0 priests."""
    # Skip check if king has any priests
    if _get_priest_count(board, king_color, cache) > 0:
        return False

    king_pos = _find_king_position(board, king_color, cache)
    if king_pos is None:
        return False
    return square_attacked_by(board, current_player, king_pos, 1 - king_color, cache)

def get_check_status(board, current_player: Color, king_color: int, cache=None) -> CheckStatus:
    """Get check status - only when king has 0 priests."""
    cache = cache or getattr(board, 'cache_manager', None)

    # Skip check if king has any priests
    if _get_priest_count(board, king_color, cache) > 0:
        return CheckStatus.SAFE

    king_pos = _find_king_position(board, king_color, cache)
    if king_pos is None or not square_attacked_by(board, current_player, king_pos, 1 - king_color, cache):
        return CheckStatus.SAFE

    return CheckStatus.CHECK

def get_all_pieces_in_check(board, current_player: Color, cache=None) -> List[KingInCheckInfo]:
    """Get all kings in check with vectorized operations."""
    cache = cache or getattr(board, 'cache_manager', None)

    pieces_in_check = []

    # Vectorized check for both colors
    for color in (Color.WHITE, Color.BLACK):
        if king_in_check(board, current_player, color, cache):
            king_pos = _find_king_position(board, color, cache)
            if king_pos is not None:
                pieces_in_check.append(KingInCheckInfo(king_coord=king_pos, color=color))

    return pieces_in_check

def batch_king_check_detection(boards: List, players: List[Color], king_colors: List[Color], cache=None) -> List[bool]:
    """Batch check multiple boards for king safety."""
    if not boards or not players or not king_colors:
        return []

    # Ensure all lists have the same length
    min_length = min(len(boards), len(players), len(king_colors))
    if min_length == 0:
        return []

    results = []
    for i in range(min_length):
        result = king_in_check(boards[i], players[i], king_colors[i], cache)
        results.append(result)

    return results

def get_check_summary(board, cache=None) -> Dict[str, Any]:
    """Get comprehensive check status summary."""
    cache = cache or getattr(board, 'cache_manager', None)

    summary = {
        'white_check': False,
        'black_check': False,
        'white_priests_alive': False,
        'black_priests_alive': False,
        'white_king_position': None,
        'black_king_position': None,
        'attacked_mask_white': np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE),
        'attacked_mask_black': np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE),
    }

    # Get priest counts
    white_priests = _get_priest_count(board, Color.WHITE, cache)
    black_priests = _get_priest_count(board, Color.BLACK, cache)

    summary['white_priests_alive'] = white_priests > 0
    summary['black_priests_alive'] = black_priests > 0

    # Get king positions using cache manager exclusively
    summary['white_king_position'] = _find_king_position(board, Color.WHITE, cache)
    summary['black_king_position'] = _find_king_position(board, Color.BLACK, cache)

    # Get attacked squares
    summary['attacked_mask_white'] = _get_attacked_squares_from_move_cache(board, Color.WHITE, cache)
    summary['attacked_mask_black'] = _get_attacked_squares_from_move_cache(board, Color.BLACK, cache)

    # Determine check status
    wk = summary['white_king_position']
    bk = summary['black_king_position']

    # Check white king safety (only when no priests)
    if wk is not None and white_priests == 0:
        wk_coords = wk.astype(COORD_DTYPE)
        # Use (x, y, z) indexing as per occupancycache.py architecture
        summary['white_check'] = bool(summary['attacked_mask_black'][wk_coords[0], wk_coords[1], wk_coords[2]])

    # Check black king safety (only when no priests)
    if bk is not None and black_priests == 0:
        bk_coords = bk.astype(COORD_DTYPE)
        # Use (x, y, z) indexing as per occupancycache.py architecture
        summary['black_check'] = bool(summary['attacked_mask_white'][bk_coords[0], bk_coords[1], bk_coords[2]])

    return summary

# Update __all__ exports
__all__ = [
    'CheckStatus', 'KingInCheckInfo',
    'king_in_check', 'get_check_status', 'get_all_pieces_in_check',
    'batch_king_check_detection', 'get_check_summary'
]
