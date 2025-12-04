from __future__ import annotations
from typing import Protocol, Optional, Dict, List, Any, NamedTuple, TYPE_CHECKING, Union
from enum import IntEnum
import numpy as np
from numba import njit, prange
import logging

logger = logging.getLogger(__name__)

from game3d.common.shared_types import (
    PieceType, Color, SIZE, COLOR_WHITE, COLOR_BLACK, COLOR_EMPTY,
    COORD_DTYPE, BOOL_DTYPE, INDEX_DTYPE, MAX_COORD_VALUE, MIN_COORD_VALUE,
    MOVE_DTYPE
)
from game3d.common.coord_utils import in_bounds_vectorized, CoordinateUtils

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

def move_would_leave_king_in_check(game_state: 'GameState', move: np.ndarray, cache=None) -> bool:
    """
    Centralized function to check if a move would leave the player's king in check.

    This is the SINGLE SOURCE OF TRUTH for move safety checking. It:
    1. Checks if player has priests (all moves are safe if true)
    2. Simulates the move on a temporary board state
    3. Checks if the king is in check after the move
    4. Reverts the simulation

    Args:
        game_state: Current game state
        move: Move array [from_x, from_y, from_z, to_x, to_y, to_z]
        cache: Optional cache manager

    Returns:
        True if move would leave king in check, False otherwise
    """
    cache = cache or getattr(game_state, 'cache_manager', None)
    if cache is None:
        raise RuntimeError("Cache manager required for move safety check")

    occ_cache = cache.occupancy_cache
    player_color = game_state.color

    # ✅ CENTRALIZED PRIEST CHECK: All moves are safe if player has priests
    if occ_cache.has_priest(player_color):
        logger.warning(f"move_would_leave_king_in_check called for {Color(player_color).name} despite having priests! Call stack should be checked.")
        return False

    # Get move data
    from_coord, to_coord = move[:3], move[3:]
    
    # ✅ OPTIMIZATION: Use get_fast for speed (coords from move array are trusted)
    if hasattr(occ_cache, 'get_fast'):
        ptype, color = occ_cache.get_fast(from_coord)
        if ptype == 0: # Empty
             return False
        
        captured_ptype, captured_color = occ_cache.get_fast(to_coord)
        captured_data = None if captured_ptype == 0 else {'piece_type': captured_ptype, 'color': captured_color}
        
        piece_data = {'piece_type': ptype, 'color': color}
    else:
        piece_data = occ_cache.get(from_coord)
        if piece_data is None:
            return False  # Invalid move (no piece at source)
        captured_data = occ_cache.get(to_coord)

    # Simulate move
    # ✅ OPTIMIZATION: Use set_position_fast if available
    # BUT skip for King moves as it doesn't update king cache
    is_king_move = (piece_data['piece_type'] == PieceType.KING)

    if hasattr(occ_cache, 'set_position_fast') and not is_king_move:
        occ_cache.set_position_fast(from_coord, 0, 0) # Clear source
        occ_cache.set_position_fast(to_coord, piece_data['piece_type'], piece_data['color'])
    else:
        occ_cache.set_position(from_coord, None)
        occ_cache.set_position(to_coord, np.array([piece_data['piece_type'], piece_data['color']]))

    try:
        # Check if king is in check using INCREMENTAL delta updates
        # This only regenerates moves for pieces affected by the simulated move
        king_pos = occ_cache.find_king(player_color)
        if king_pos is None:
             is_check = True # Assume worst if king missing
        else:
             # ✅ OPTIMIZED: Use incremental delta updates instead of full regeneration
             # This is 10-20x faster than the slow path
             is_check = square_attacked_by_incremental(
                 game_state.board,
                 king_pos,
                 Color(player_color).opposite().value, # Opponent color
                 cache,
                 from_coord,
                 to_coord
             )
    finally:
        # Always revert move (even if exception occurs)
        if hasattr(occ_cache, 'set_position_fast') and not is_king_move:
            occ_cache.set_position_fast(from_coord, piece_data['piece_type'], piece_data['color'])
            if captured_data:
                occ_cache.set_position_fast(to_coord, captured_data['piece_type'], captured_data['color'])
            else:
                occ_cache.set_position_fast(to_coord, 0, 0)
        else:
            occ_cache.set_position(from_coord, np.array([piece_data['piece_type'], piece_data['color']]))
            if captured_data:
                occ_cache.set_position(to_coord, np.array([captured_data['piece_type'], captured_data['color']]))
            else:
                occ_cache.set_position(to_coord, None)

    return is_check

def _get_priest_count(board, king_color: Optional[Color] = None, cache: Any = None) -> int:
    """Get count of priests for the given king color."""
    cache = cache or getattr(board, 'cache_manager', None)
    if cache is None:
        from game3d.cache.manager import get_cache_manager
        cache = get_cache_manager(board, king_color or Color.WHITE)

    # Check if cache is OccupancyCache or CacheManager
    if hasattr(cache, 'has_priest'):
        return 0 if not cache.has_priest(king_color) else 1
    elif hasattr(cache, 'occupancy_cache') and hasattr(cache.occupancy_cache, 'has_priest'):
        return 0 if not cache.occupancy_cache.has_priest(king_color) else 1
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


def _get_attacked_squares_from_move_cache(board, attacker_color: int, cache=None) -> Optional[np.ndarray]:
    """Get attacked squares using move cache - calculates from RAW/pseudolegal moves.
    
    Uses raw moves instead of legal moves because:
    - For check detection, we need ALL squares an enemy can attack
    - Legal moves are filtered for king safety, which would miss attacking squares
    - A piece can attack a square even if moving there would leave its own king in check
    
    Returns None if cache miss (caller should fallback to slow check).
    """
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)

    # Use move cache for optimal attack calculation
    if cache and hasattr(cache, 'move_cache'):
        # ✅ UPDATED: Use get_pseudolegal_moves() to get occupancy-aware attacks
        cached_moves = cache.move_cache.get_pseudolegal_moves(attacker_color)
        if cached_moves is None:
            return None # Cache miss

        if len(cached_moves) == 0:
            return mask

        # Extract destination coordinates from cached moves (MOVE_DTYPE: [from_x, from_y, from_z, to_x, to_y, to_z])
        for move in cached_moves:
            # Moves are numpy arrays with columns: [from_x, from_y, from_z, to_x, to_y, to_z]
            to_x, to_y, to_z = int(move[3]), int(move[4]), int(move[5])
            if (0 <= to_x < SIZE and 0 <= to_y < SIZE and 0 <= to_z < SIZE):
                # Use (x, y, z) indexing as per occupancycache.py architecture
                mask[to_x, to_y, to_z] = True

        return mask

    return None # No cache available


def square_attacked_by(board, current_player: Color, square: np.ndarray, attacker_color: int, cache=None, use_move_cache=True) -> bool:
    """Check if a square is attacked by a specific color."""
    square = square.astype(COORD_DTYPE)

    # Vectorized bounds checking
    # ✅ OPTIMIZATION: Use scalar check for single coordinate
    if hasattr(CoordinateUtils, 'in_bounds_scalar'):
        if not CoordinateUtils.in_bounds_scalar(square[0], square[1], square[2]):
            return False
    elif not in_bounds_vectorized(square.reshape(1, -1))[0]:
        return False

    # Use move cache if allowed and available
    if use_move_cache and cache and hasattr(cache, 'move_cache'):
        attacked_mask = _get_attacked_squares_from_move_cache(board, attacker_color, cache)
        if attacked_mask is not None:
            x, y, z = square[0], square[1], square[2]
            return bool(attacked_mask[x, y, z])
    
    # Fallback: Fast inverse check (was slow dynamic check)
    from game3d.attacks.fast_attack import square_attacked_by_fast
    return square_attacked_by_fast(board, square, attacker_color, cache)

def square_attacked_by_incremental(
    board,
    square: np.ndarray,
    attacker_color: int,
    cache,
    from_coord: np.ndarray,
    to_coord: np.ndarray
) -> bool:
    """
    Check if square is attacked using incremental delta updates with lazy regeneration.
    
    ✅ OPTIMIZATION: Uses piece cache for O(1) move lookups instead of regenerating.
    Only regenerates moves for pieces that are actually affected (typically 1-5 pieces).
    
    This is ~10-20x faster than _square_attacked_by_slow because:
    1. Uses cached moves for unaffected pieces (O(1) lookup)
    2. Only regenerates for affected pieces (typically 1-5 vs 16+)
    3. Uses lazy attack mask computation (only if needed)
    
    Algorithm:
    1. Identify pieces affected by the move (from/to squares)
    2. Get OLD moves for affected pieces from piece cache (O(1))
    3. Regenerate moves ONLY for affected pieces with current occupancy
    4. Create attack mask with delta applied
    5. Check if target square is in mask
    
    Args:
        board: Game board
        square: Target square to check (3,)
        attacker_color: Color of attacking pieces
        cache: Cache manager
        from_coord: Source coordinate of simulated move (3,)
        to_coord: Destination coordinate of simulated move (3,)
        
    Returns:
        True if square is attacked, False otherwise
    """
    if not cache or not hasattr(cache, 'move_cache'):
        # Fallback if cache not available
        # No cached moves, use fast path
        from game3d.attacks.fast_attack import square_attacked_by_fast
        return square_attacked_by_fast(board, square, attacker_color, cache)
    
    # Identify pieces affected by the move
    affected_ids, affected_coords, affected_keys = cache.move_cache.get_pieces_affected_by_move(
        from_coord, to_coord, attacker_color
    )
    
    # ✅ OPTIMIZATION: Use piece cache directly for OLD moves (O(1) per piece)
    # No need to extract from base_moves array
    old_affected_moves_list = []
    
    if len(affected_ids) > 0:
        color_idx = 0 if attacker_color == Color.WHITE else 1
        
        with cache.move_cache._lock:
            for piece_id in affected_ids:
                if piece_id in cache.move_cache._piece_moves_cache:
                    old_moves = cache.move_cache._piece_moves_cache[piece_id]
                    if old_moves.size > 0:
                        old_affected_moves_list.append(old_moves)
    
    old_affected_moves = np.concatenate(old_affected_moves_list, axis=0) if old_affected_moves_list else np.empty((0, 6), dtype=MOVE_DTYPE)
    
    # Regenerate moves for affected pieces with current occupancy
    # NOTE: The occupancy cache has already been updated by the caller
    from game3d.game.gamestate import GameState
    from game3d.movement.pseudolegal import generate_pseudolegal_moves_batch
    
    # Create temporary game state for move generation
    temp_state = GameState(board, attacker_color, cache)
    
    # Generate new moves for affected pieces
    new_affected_moves = generate_pseudolegal_moves_batch(
        temp_state, affected_coords, np.empty((0, 3), dtype=COORD_DTYPE), ignore_occupancy=False
    ) if affected_coords.size > 0 else np.empty((0, 6), dtype=MOVE_DTYPE)
    
    # ✅ LAZY ATTACK MASK: Only create if there are delta changes OR no cached base moves
    # Get base moves from cache
    base_moves = cache.move_cache.get_pseudolegal_moves(attacker_color)
    
    if base_moves is None:
        # No cached moves, use slow path
        # No cached moves, use fast path
        from game3d.attacks.fast_attack import square_attacked_by_fast
        return square_attacked_by_fast(board, square, attacker_color, cache)
    
    # If no affected pieces, just use the cached attack mask directly
    if len(affected_ids) == 0:
        # Create attack mask from base moves (fast path - no regeneration needed)
        attack_mask = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)
        if base_moves.size > 0:
            tx, ty, tz = base_moves[:, 3], base_moves[:, 4], base_moves[:, 5]
            valid = (tx >= 0) & (tx < SIZE) & (ty >= 0) & (ty < SIZE) & (tz >= 0) & (tz < SIZE)
            attack_mask[tx[valid], ty[valid], tz[valid]] = True
        
        x, y, z = square[0], square[1], square[2]
        return bool(attack_mask[x, y, z])
    
    # Create attack mask from base moves
    attack_mask = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)
    if base_moves.size > 0:
        tx, ty, tz = base_moves[:, 3], base_moves[:, 4], base_moves[:, 5]
        valid = (tx >= 0) & (tx < SIZE) & (ty >= 0) & (ty < SIZE) & (tz >= 0) & (tz < SIZE)
        attack_mask[tx[valid], ty[valid], tz[valid]] = True
    
    # Apply delta: remove old affected moves and add new ones
    cache.move_cache.remove_moves_from_mask(attack_mask, old_affected_moves)
    cache.move_cache.add_moves_to_mask(attack_mask, new_affected_moves)
    
    # Check if target square is attacked
    x, y, z = square[0], square[1], square[2]
    return bool(attack_mask[x, y, z])


def _square_attacked_by_slow(board, square: np.ndarray, attacker_color: int, cache=None) -> bool:
    """Check if square is attacked by generating moves dynamically (slow)."""
    # We need to generate moves for all opponent pieces given the CURRENT occupancy.
    # We can't use the move cache because it might be stale.
    
    # Get opponent positions from occupancy cache (which is updated)
    if cache and hasattr(cache, 'occupancy_cache'):
        attacker_positions = cache.occupancy_cache.get_positions(attacker_color)
    else:
        # Fallback if no cache (shouldn't happen in this context)
        return False

    # Import here to avoid circular imports
    from game3d.movement.pseudolegal import generate_pseudolegal_moves_batch
    
    # We need a GameState.
    from game3d.game.gamestate import GameState
    
    # Hack: Create a dummy GameState
    dummy_state = GameState(board, attacker_color, cache)
    
    # Generate ALL raw moves for attacker
    # This is expensive but correct.
    moves = generate_pseudolegal_moves_batch(dummy_state, attacker_positions)
    
    # Check if any move targets the square
    # moves is (N, 6) array: [fx, fy, fz, tx, ty, tz]
    if moves.size == 0:
        return False

    target_x, target_y, target_z = square[0], square[1], square[2]
    
    hits = (moves[:, 3] == target_x) & (moves[:, 4] == target_y) & (moves[:, 5] == target_z)
    return np.any(hits)

def king_in_check(board, current_player: Color, king_color: int, cache=None) -> bool:
    """Check if king is in check - only when king has 0 priests."""
    # Skip check if king has any priests
    if _get_priest_count(board, king_color, cache) > 0:
        return False

    king_pos = _find_king_position(board, king_color, cache)
    if king_pos is None:
        # King is missing and no priests -> Treated as Check (Checkmate)
        return True
    return square_attacked_by(board, current_player, king_pos, Color(king_color).opposite().value, cache)

def get_check_status(board, current_player: Color, king_color: int, cache=None) -> CheckStatus:
    """Get check status - only when king has 0 priests."""
    cache = cache or getattr(board, 'cache_manager', None)

    # Skip check if king has any priests
    if _get_priest_count(board, king_color, cache) > 0:
        return CheckStatus.SAFE

    king_pos = _find_king_position(board, king_color, cache)
    if king_pos is None or not square_attacked_by(board, current_player, king_pos, Color(king_color).opposite().value, cache):
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

    # ✅ OPTIMIZATION: If both sides have priests, no check detection is needed
    if summary['white_priests_alive'] and summary['black_priests_alive']:
        return summary

    # Get king positions using cache manager exclusively - only if needed
    if not summary['white_priests_alive']:
        summary['white_king_position'] = _find_king_position(board, Color.WHITE, cache)
    
    if not summary['black_priests_alive']:
        summary['black_king_position'] = _find_king_position(board, Color.BLACK, cache)

    # Get attacked squares - only if needed for check detection
    # We need black's attacks to check if white king is in check
    if not summary['white_priests_alive']:
        mask = _get_attacked_squares_from_move_cache(board, Color.BLACK, cache)
        if mask is not None:
            summary['attacked_mask_black'] = mask
        else:
            # Fallback to slow check for King only?
            # For summary, we might want the full mask, but generating it slowly is expensive.
            # We'll just leave it empty and rely on king_in_check for the boolean status.
            pass
        
    # We need white's attacks to check if black king is in check
    if not summary['black_priests_alive']:
        mask = _get_attacked_squares_from_move_cache(board, Color.WHITE, cache)
        if mask is not None:
            summary['attacked_mask_white'] = mask
        else:
            pass

    # Determine check status
    wk = summary['white_king_position']
    bk = summary['black_king_position']

    # Check white king safety (only when no priests)
    if wk is not None and white_priests == 0:
        wk_coords = wk.astype(COORD_DTYPE)
        # Use (x, y, z) indexing as per occupancycache.py architecture
        # If mask is available, use it
        if np.any(summary['attacked_mask_black']):
             summary['white_check'] = bool(summary['attacked_mask_black'][wk_coords[0], wk_coords[1], wk_coords[2]])
        else:
             # Fallback to precise check
             summary['white_check'] = square_attacked_by(board, Color.WHITE, wk, Color.BLACK.value, cache)

    # Check black king safety (only when no priests)
    if bk is not None and black_priests == 0:
        bk_coords = bk.astype(COORD_DTYPE)
        # Use (x, y, z) indexing as per occupancycache.py architecture
        if np.any(summary['attacked_mask_white']):
            summary['black_check'] = bool(summary['attacked_mask_white'][bk_coords[0], bk_coords[1], bk_coords[2]])
        else:
            summary['black_check'] = square_attacked_by(board, Color.BLACK, bk, Color.WHITE.value, cache)

    return summary

# Update __all__ exports
__all__ = [
    'CheckStatus', 'KingInCheckInfo',
    'king_in_check', 'get_check_status', 'get_all_pieces_in_check',
    'batch_king_check_detection', 'get_check_summary',
    'square_attacked_by', 'square_attacked_by_incremental', 'move_would_leave_king_in_check'
]

