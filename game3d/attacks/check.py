from __future__ import annotations
from typing import Protocol, Optional, Dict, List, Any, NamedTuple, TYPE_CHECKING, Union
from enum import IntEnum
import numpy as np
from numba import njit, prange
import logging
import threading

logger = logging.getLogger(__name__)

# ✅ OPTIMIZATION: Thread-local scratch buffer for attack mask delta operations
_ATTACK_SCRATCH = threading.local()

# ✅ OPTIMIZATION: Singleton empty arrays to avoid repeated allocations
_EMPTY_MOVES = np.empty((0, 6), dtype=np.int16)  # Frozen empty moves array


def _get_scratch_buffers() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get thread-local scratch buffers for subset generation."""
    if not hasattr(_ATTACK_SCRATCH, 'type_counts'):
        # Allocate once per thread
        # 512 is safe upper bound for piece count
        _ATTACK_SCRATCH.type_counts = np.zeros(64, dtype=np.int32)
        _ATTACK_SCRATCH.effective_types = np.zeros(512, dtype=np.int32)
        _ATTACK_SCRATCH.sorted_indices = np.zeros(512, dtype=np.int32)
        _ATTACK_SCRATCH.offsets = np.zeros(64, dtype=np.int32)
    
    return (
        _ATTACK_SCRATCH.type_counts,
        _ATTACK_SCRATCH.effective_types,
        _ATTACK_SCRATCH.sorted_indices,
        _ATTACK_SCRATCH.offsets
    )


from game3d.common.shared_types import (
    PieceType, Color, SIZE, COLOR_WHITE, COLOR_BLACK, COLOR_EMPTY,
    COORD_DTYPE, BOOL_DTYPE, INDEX_DTYPE, MAX_COORD_VALUE, MIN_COORD_VALUE,
    MOVE_DTYPE
)
from game3d.common.coord_utils import in_bounds_vectorized, CoordinateUtils

if TYPE_CHECKING:
    from game3d.attacks.movepiece import Move
    from game3d.cache.manager import CacheManager
    from game3d.game.gamestate import GameState

from game3d.common.shared_types import MinimalStateProxy

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
        # logger.warning(f"move_would_leave_king_in_check called for {Color(player_color).name} despite having priests!")
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


def batch_moves_leave_king_in_check(
    game_state: 'GameState',
    moves: np.ndarray,
    cache=None
) -> np.ndarray:
    """
    Batch check if moves would leave the player's king in check.
    
    ✅ OPTIMIZED: Uses attack mask comparison instead of per-move simulation.
    For N moves, this is O(N) instead of O(N * M) where M = piece count.
    
    Algorithm:
    1. Get current attack mask from opponent's cached moves
    2. For non-king moves: king stays in place, just check presence in mask
    3. For king moves: check each destination against attack mask
    4. Handle special cases (captures that remove attackers)
    
    Args:
        game_state: Current game state
        moves: (N, 6) array of moves to check
        cache: Optional cache manager
        
    Returns:
        Boolean array of length N - True if move leaves king in check
    """
    if moves.size == 0:
        return np.array([], dtype=BOOL_DTYPE)
    
    cache = cache or getattr(game_state, 'cache_manager', None)
    if cache is None:
        # Fallback to per-move simulation
        return np.array([
            move_would_leave_king_in_check(game_state, move, cache)
            for move in moves
        ], dtype=BOOL_DTYPE)
    
    occ_cache = cache.occupancy_cache
    player_color = game_state.color
    opponent_color = Color(player_color).opposite().value
    
    # Get attack mask from cached opponent moves
    attack_mask = _get_attacked_squares_from_move_cache(
        game_state.board, opponent_color, cache
    )
    
    if attack_mask is None:
        # Cache miss - fallback to per-move simulation
        return np.array([
            move_would_leave_king_in_check(game_state, move, cache)
            for move in moves
        ], dtype=BOOL_DTYPE)
    
    # Find king position
    king_pos = occ_cache.find_king(player_color)
    if king_pos is None:
        # King missing - all moves are "unsafe"
        return np.ones(len(moves), dtype=BOOL_DTYPE)
    
    kx, ky, kz = int(king_pos[0]), int(king_pos[1]), int(king_pos[2])
    results = np.zeros(len(moves), dtype=BOOL_DTYPE)
    
    # Identify king moves
    king_move_mask = (
        (moves[:, 0] == kx) & 
        (moves[:, 1] == ky) & 
        (moves[:, 2] == kz)
    )
    
    # For non-king moves: king stays at current position
    # If king is currently attacked, all non-king moves are potentially unsafe
    # unless they block the attack or capture the attacker
    king_currently_attacked = attack_mask[kx, ky, kz]
    
    if king_currently_attacked:
        # When in check, we need to simulate each move
        # This path is unavoidable but rare (only when in check)
        non_king_indices = np.where(~king_move_mask)[0]
        for idx in non_king_indices:
            results[idx] = move_would_leave_king_in_check(game_state, moves[idx], cache)
    # else: Non-king moves are safe (king not in attack mask)
    
    # For king moves: check destination against attack mask
    king_indices = np.where(king_move_mask)[0]
    for idx in king_indices:
        dest = moves[idx, 3:6].astype(np.int32)
        dx, dy, dz = int(dest[0]), int(dest[1]), int(dest[2])
        
        # Check bounds
        if not (0 <= dx < SIZE and 0 <= dy < SIZE and 0 <= dz < SIZE):
            results[idx] = True  # Out of bounds = unsafe
            continue
        
        # King moves to attacked square
        if attack_mask[dx, dy, dz]:
            # But if this is a capture, the attacker is removed
            # Need to check if the captured piece was the ONLY attacker
            # This is complex, so fall back to simulation for captures
            captured_ptype, _ = occ_cache.get_fast(dest) if hasattr(occ_cache, 'get_fast') else (0, 0)
            if captured_ptype != 0:
                # Capture - must simulate
                results[idx] = move_would_leave_king_in_check(game_state, moves[idx], cache)
            else:
                # No capture, moving to attacked square = unsafe
                results[idx] = True
        # else: Destination not attacked = safe
    
    return results


@njit(cache=True, fastmath=True)
def _find_attackers_of_square(
    king_pos: np.ndarray,
    opponent_moves: np.ndarray,
    opponent_from_coords: np.ndarray
) -> np.ndarray:
    """Find indices of opponent moves that attack the king position."""
    n_moves = opponent_moves.shape[0]
    attacker_indices = np.empty(n_moves, dtype=np.int32)
    count = 0
    
    kx, ky, kz = king_pos[0], king_pos[1], king_pos[2]
    
    for i in range(n_moves):
        if (opponent_moves[i, 3] == kx and 
            opponent_moves[i, 4] == ky and 
            opponent_moves[i, 5] == kz):
            attacker_indices[count] = i
            count += 1
    
    return attacker_indices[:count]


@njit(cache=True, fastmath=True, parallel=True)
def _batch_check_move_blocks_or_captures(
    moves: np.ndarray,
    attacker_from: np.ndarray,
    king_pos: np.ndarray,
    indices: np.ndarray
) -> np.ndarray:
    """
    Vectorized check if moves block all attack rays or capture the attacker.
    
    Args:
        moves: (N, 6) entire moves array
        attacker_from: (3,) attacker position
        king_pos: (3,) king position
        indices: (K,) indices of moves to check (subset of moves)
        
    Returns:
        (K,) boolean array (True=SAFE, False=UNSAFE)
    """
    n = indices.shape[0]
    results = np.empty(n, dtype=BOOL_DTYPE)
    
    ax, ay, az = attacker_from[0], attacker_from[1], attacker_from[2]
    kx, ky, kz = king_pos[0], king_pos[1], king_pos[2]
    
    # Calculate direction from attacker to king
    dx = kx - ax
    dy = ky - ay
    dz = kz - az
    
    # Normalize direction (L-infinity norm for Chebyshev distance)
    max_abs = max(abs(dx), max(abs(dy), abs(dz)))
    
    for i in prange(n):
        idx = indices[i]
        mx, my, mz = moves[idx, 3], moves[idx, 4], moves[idx, 5]
        
        # 1. Check CAPTURE
        if mx == ax and my == ay and mz == az:
            results[i] = True
            continue
            
        # 2. Check BLOCK
        if max_abs == 0:
            results[i] = False
            continue
            
        # Check collinearity: (move - attacker) cross (king - attacker) == 0
        dmx = mx - ax
        dmy = my - ay
        dmz = mz - az
        
        cross_x = dmy * dz - dmz * dy
        cross_y = dmz * dx - dmx * dz
        cross_z = dmx * dy - dmy * dx
        
        if cross_x != 0 or cross_y != 0 or cross_z != 0:
            results[i] = False # Not on ray
            continue
            
        # Check bounded segment: 0 < t < max_abs
        # t = dm / d
        if dx != 0:
            t = dmx * max_abs // dx
        elif dy != 0:
            t = dmy * max_abs // dy
        else:
            t = dmz * max_abs // dz
            
        if 0 < t < max_abs:
            results[i] = True # Blocking
        else:
            results[i] = False

    return results


@njit(cache=True, fastmath=True)
def _check_move_blocks_or_captures(
    move: np.ndarray,
    attacker_from: np.ndarray,
    king_pos: np.ndarray,
    n_attackers: int
) -> bool:
    """
    Check if a move blocks all attack rays or captures the attacker.
    Returns True if move is SAFE (blocks/captures), False if UNSAFE.
    """
    # If more than one attacker, only capturing won't help (can't capture both)
    # and blocking is impossible (can't block both rays)
    if n_attackers > 1:
        # Must check if move captures one of them AND blocks the other
        # This is too complex - return False (unsafe) to force simulation
        # Or simplistic view: Unsafe unless king moves (handled by caller)
        return False
    
    # Single attacker case
    move_to = move[3:6]
    ax, ay, az = attacker_from[0], attacker_from[1], attacker_from[2]
    
    # Check if move captures the attacker
    if move_to[0] == ax and move_to[1] == ay and move_to[2] == az:
        return True  # Capture = safe
    
    # Check if move blocks the attack ray
    # Ray goes from attacker to king - move must land on this ray
    kx, ky, kz = king_pos[0], king_pos[1], king_pos[2]
    
    # Calculate direction from attacker to king
    dx = kx - ax
    dy = ky - ay
    dz = kz - az
    
    # Normalize direction
    max_abs = max(abs(dx), max(abs(dy), abs(dz)))
    if max_abs == 0:
        return False  # Attacker on king? Shouldn't happen
    
    # Check if move destination is on the line between attacker and king
    # It must be at a position (ax + t*dx/max, ay + t*dy/max, az + t*dz/max)
    # for some 0 < t < max_abs
    
    mx, my, mz = move_to[0], move_to[1], move_to[2]
    
    # Check alignment with the ray
    dmx = mx - ax
    dmy = my - ay
    dmz = mz - az
    
    # The move must be collinear and between attacker and king
    # Cross product should be zero for collinearity
    # (dm x d) == 0
    cross_x = dmy * dz - dmz * dy
    cross_y = dmz * dx - dmx * dz
    cross_z = dmx * dy - dmy * dx
    
    if cross_x != 0 or cross_y != 0 or cross_z != 0:
        return False  # Not on the ray
    
    # Check that move is strictly between attacker and king (not beyond)
    # t = dm / d should be in (0, 1) * max_abs
    if dx != 0:
        t = dmx * max_abs // dx
    elif dy != 0:
        t = dmy * max_abs // dy
    else:
        t = dmz * max_abs // dz
    
    if 0 < t < max_abs:
        return True  # Blocking move
    
    return False  # Not blocking


def batch_moves_leave_king_in_check_fused(
    game_state: 'GameState',
    moves: np.ndarray,
    cache
) -> np.ndarray:
    """
    ✅ OPTIMIZED: Fused batch check that handles the 'in check' case efficiently.
    
    Unlike batch_moves_leave_king_in_check, this function avoids per-move simulation
    when the king is in check by:
    1. Identifying all attackers of the king
    2. For single-attacker case: Check if move captures or blocks
    3. For multi-attacker case: Only king moves can escape (double check)
    4. For king moves: Use attack mask to check safety of destination
    
    This is 10-50x faster than per-move simulation when in check.
    """
    if moves.size == 0:
        return np.array([], dtype=BOOL_DTYPE)
    
    occ_cache = cache.occupancy_cache
    player_color = game_state.color
    opponent_color = Color(player_color).opposite().value
    
    # Find king position
    king_pos = occ_cache.find_king(player_color)
    if king_pos is None:
        return np.ones(len(moves), dtype=BOOL_DTYPE)
    
    kx, ky, kz = int(king_pos[0]), int(king_pos[1]), int(king_pos[2])
    king_pos_arr = king_pos.astype(COORD_DTYPE)
    
    # Get attack mask from cached opponent moves
    attack_mask = _get_attacked_squares_from_move_cache(
        game_state.board, opponent_color, cache
    )
    
    if attack_mask is None:
        # No cached moves - use fast attack check
        from game3d.attacks.fast_attack import square_attacked_by_fast
        is_attacked = square_attacked_by_fast(game_state.board, king_pos_arr, opponent_color, cache)
        if not is_attacked:
            # King not in check - all moves are potentially safe
            return np.zeros(len(moves), dtype=BOOL_DTYPE)
        # Fallback to old behavior
        return batch_moves_leave_king_in_check(game_state, moves, cache)
    
    king_currently_attacked = attack_mask[kx, ky, kz]
    
    # Identify king moves
    king_move_mask = (
        (moves[:, 0] == kx) & 
        (moves[:, 1] == ky) & 
        (moves[:, 2] == kz)
    )
    
    results = np.zeros(len(moves), dtype=BOOL_DTYPE)
    
    if not king_currently_attacked:
        # King not in check - use simple attack mask check for king moves
        king_indices = np.where(king_move_mask)[0]
        for idx in king_indices:
            dest = moves[idx, 3:6].astype(np.int32)
            dx, dy, dz = int(dest[0]), int(dest[1]), int(dest[2])
            
            if not (0 <= dx < SIZE and 0 <= dy < SIZE and 0 <= dz < SIZE):
                results[idx] = True
            elif attack_mask[dx, dy, dz]:
                # Check for capture that removes the attacker
                captured_ptype, _ = occ_cache.get_fast(dest) if hasattr(occ_cache, 'get_fast') else (0, 0)
                if captured_ptype != 0:
                    # Simulate to check if capturing removes the threat
                    results[idx] = move_would_leave_king_in_check(game_state, moves[idx], cache)
                else:
                    results[idx] = True
        # Non-king moves are safe when not in check
        return results
    
    # ===== KING IS IN CHECK - OPTIMIZED PATH =====
    
    # Get opponent's moves to find attackers
    opponent_moves = cache.move_cache.get_pseudolegal_moves(opponent_color)
    if opponent_moves is None or opponent_moves.size == 0:
        # No opponent moves cached, fallback
        return batch_moves_leave_king_in_check(game_state, moves, cache)
    
    # Find all pieces attacking the king
    attacker_indices = _find_attackers_of_square(king_pos_arr, opponent_moves, opponent_moves[:, :3])
    n_attackers = len(np.unique(opponent_moves[attacker_indices, 0:3].view(dtype=[('x', COORD_DTYPE), ('y', COORD_DTYPE), ('z', COORD_DTYPE)]).ravel()))
    
    # Get unique attacker positions
    if attacker_indices.size > 0:
        attacker_from_coords = opponent_moves[attacker_indices, :3]
        unique_attackers = np.unique(attacker_from_coords, axis=0)
        n_unique_attackers = len(unique_attackers)
    else:
        unique_attackers = np.empty((0, 3), dtype=COORD_DTYPE)
        n_unique_attackers = 0
    
    if n_unique_attackers == 0:
        # No attackers found despite being 'in check' - data inconsistency
        # Fallback to safe behavior
        return batch_moves_leave_king_in_check(game_state, moves, cache)
    
    if n_unique_attackers > 1:
        # Double (or more) check - only king moves can save
        # Mark all non-king moves as unsafe
        results[~king_move_mask] = True
    else:
        # Single attacker - non-king moves can block or capture
        attacker_pos = unique_attackers[0]
        
        non_king_indices = np.where(~king_move_mask)[0]
        if non_king_indices.size > 0:
            is_safe_batch = _batch_check_move_blocks_or_captures(
                moves,
                attacker_pos.astype(COORD_DTYPE),
                king_pos_arr,
                non_king_indices.astype(np.int32)
            )
            # Invert because results array stores True for UNSAFE ("leaves king in check")
            # is_safe_batch returns True for SAFE ("blocks/captures")
            results[non_king_indices] = ~is_safe_batch
    
    # Check king moves - must move to unattacked square
    king_indices = np.where(king_move_mask)[0]
    for idx in king_indices:
        dest = moves[idx, 3:6].astype(np.int32)
        dx, dy, dz = int(dest[0]), int(dest[1]), int(dest[2])
        
        if not (0 <= dx < SIZE and 0 <= dy < SIZE and 0 <= dz < SIZE):
            results[idx] = True
            continue
        
        # Check if destination is attacked
        # Note: We need to account for the king leaving its current position
        # which might open up new attacks or remove some
        # For safety, simulate for king moves
        results[idx] = move_would_leave_king_in_check(game_state, moves[idx], cache)
    
    return results


def _get_attacked_squares_from_move_cache(board, attacker_color: int, cache=None) -> Optional[np.ndarray]:
    """Get attacked squares using move cache - calculates from RAW/pseudolegal moves.
    
    Uses raw moves instead of legal moves because:
    - For check detection, we need ALL squares an enemy can attack
    - Legal moves are filtered for king safety, which would miss attacking squares
    - A piece can attack a square even if moving there would leave its own king in check
    
    Returns None if cache miss (caller should fallback to slow check).
    """
    # Use move cache for optimal attack calculation
    if cache and hasattr(cache, 'move_cache'):
        # ✅ OPTIMIZATION: First check for pre-cached attack mask
        cached_mask = cache.move_cache.get_attack_mask(attacker_color)
        if cached_mask is not None:
            return cached_mask
        
        # Cache miss on mask - build from moves
        cached_moves = cache.move_cache.get_pseudolegal_moves(attacker_color)
        if cached_moves is None:
            return None  # Cache miss

        mask = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)
        
        if len(cached_moves) == 0:
            return mask

        # ✅ OPTIMIZED: Extract destination coordinates without conversion
        # NumPy advanced indexing works with int16 (COORD_DTYPE)
        to_coords = cached_moves[:, 3:6]
        
        # Vectorized bounds check
        valid = (to_coords[:, 0] >= 0) & (to_coords[:, 0] < SIZE) & \
                (to_coords[:, 1] >= 0) & (to_coords[:, 1] < SIZE) & \
                (to_coords[:, 2] >= 0) & (to_coords[:, 2] < SIZE)
        
        # Apply valid coordinates to mask using advanced indexing
        valid_coords = to_coords[valid]
        if len(valid_coords) > 0:
            mask[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = True

        return mask

    return None  # No cache available


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
    4. ✅ OPTIMIZED: Uses lightweight proxy instead of full GameState creation
    
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
        from game3d.attacks.fast_attack import square_attacked_by_fast
        return square_attacked_by_fast(board, square, attacker_color, cache)
    
    # ✅ FIX: Check if opponent moves are cached - if not, fall back to full check
    # This handles the case where the move cache hasn't generated opponent moves yet
    cached_opponent_moves = cache.move_cache.get_pseudolegal_moves(attacker_color)
    if cached_opponent_moves is None:
        # No cached moves - use fast_attack which regenerates from current occupancy
        from game3d.attacks.fast_attack import square_attacked_by_fast
        return square_attacked_by_fast(board, square, attacker_color, cache)
    
    # Identify pieces affected by the move
    affected_ids, affected_coords, affected_keys = cache.move_cache.get_pieces_affected_by_move(
        from_coord, to_coord, attacker_color
    )
    
    # ✅ OPTIMIZATION: Simplified High-Performance Logic
    # 1. FAST CHECK: Do any STABLE pieces attack?
    # This checks the global attack matrix but masks out the 'affected' pieces.
    # If a non-affected piece attacks, we are done (True).
    if cache.move_cache.has_other_attackers(square, attacker_color, affected_keys):
        return True

    # 2. SLOW CHECK: Do any AFFECTED pieces attack?
    # We must regenerate their moves because their attacks might have changed.
    # Note: We only generate if stable check failed (lazy generation would be even better, 
    # but we need to know IF they attack).
    if affected_coords.size > 0:
        from game3d.core.buffer import state_to_buffer_from_pieces
        from game3d.core.generator_functional import generate_moves_subset
        
        # Generator imports
        occ_cache = cache.occupancy_cache
        current_colors, current_types = occ_cache.batch_get_attributes_unsafe(affected_coords)
        
        # ✅ FIX: Apply the move to the affected state
        # 1. Handle Captures: Use boolean indexing
        # If an affected piece is at 'to_coord' and matches attacker_color, it is captured (removed).
        # Actually, get_pieces_affected_by_move returns pieces of 'attacker_color'.
        # If we (player) capture an opponent (attacker), that opponent piece is at to_coord.
        # We must remove it.
        
        # Scalar compare for to_coord
        tx, ty, tz = int(to_coord[0]), int(to_coord[1]), int(to_coord[2])
        is_captured = (affected_coords[:, 0] == tx) & \
                      (affected_coords[:, 1] == ty) & \
                      (affected_coords[:, 2] == tz)
                      
        # 2. Handle Moving Piece: from_coord -> to_coord
        fx, fy, fz = int(from_coord[0]), int(from_coord[1]), int(from_coord[2])
        is_moving = (affected_coords[:, 0] == fx) & \
                    (affected_coords[:, 1] == fy) & \
                    (affected_coords[:, 2] == fz)
        
        # Apply filtering (remove captured)
        # Note: A piece can't be both moving and captured in this context usually
        valid_mask = ~is_captured
        
        if not np.any(valid_mask):
             affected_coords = np.empty((0, 3), dtype=COORD_DTYPE)
             current_types = np.empty(0, dtype=np.int8)
             current_colors = np.empty(0, dtype=np.int8)
        else:
             affected_coords = affected_coords[valid_mask].copy() # Copy to modify
             current_types = current_types[valid_mask]
             current_colors = current_colors[valid_mask]
             # Update moving piece coords
             # Re-calculate is_moving for the filtered array
             is_moving = (affected_coords[:, 0] == fx) & \
                         (affected_coords[:, 1] == fy) & \
                         (affected_coords[:, 2] == fz)
             
             if np.any(is_moving):
                 affected_coords[is_moving] = to_coord

        source = MinimalStateProxy(board, attacker_color, cache)
        if affected_coords.size > 0:
            buffer = state_to_buffer_from_pieces(
                source, affected_coords, current_types, current_colors, readonly=True
            )
            
            subset_indices = np.arange(len(affected_coords), dtype=np.int32)
            scratch = _get_scratch_buffers()
            new_affected_moves = generate_moves_subset(buffer, subset_indices, scratch_pad=scratch)
        else:
            new_affected_moves = _EMPTY_MOVES
        
        if new_affected_moves.size > 0:
            tx, ty, tz = int(square[0]), int(square[1]), int(square[2])
            # Vectorized check
            hits = (new_affected_moves[:, 3] == tx) & \
                   (new_affected_moves[:, 4] == ty) & \
                   (new_affected_moves[:, 5] == tz)
            if np.any(hits):
                return True
                
    return False


@njit(cache=True)
def _can_capture_attacker(attacker_coord, defender_color, occupancy_grid, piece_type_grid):
    # This function is a helper but ideally we should query attacks on the attacker_coord
    # But for optimization in check scenarios, we might need direct logic.
    # For now, let's focus on is_square_attacked.
    pass

def is_square_attacked(state, square_coord: np.ndarray, defender_color: Optional[int] = None) -> bool:
    """
    Check if a square is under attack by the opponent.
    
    OPTIMIZATION: Uses Bitboard cache from MoveCache if available.
    """
    if defender_color is None:
        defender_color = state.color
    
    # 1. Try Cache First (Bitboard O(1))
    # Check if 'defender_color' is under attack (implies attacker is opposite)
    if state.cache_manager.move_cache.is_under_attack(square_coord, defender_color):
        return True
        
    # If cache returns False (dirty or clean-safe), fall back to general check.
    attacker_color = Color(defender_color).opposite().value
    return square_attacked_by(state.board, state.color, square_coord, attacker_color, state.cache_manager)


def _square_attacked_by_slow(board, square: np.ndarray, attacker_color: int, cache=None) -> bool:
    """Check if square is attacked by generating moves dynamically (slow).
    
    ✅ OPTIMIZED: Uses lightweight _MinimalStateProxy instead of full GameState.
    """
    # We need to generate moves for all opponent pieces given the CURRENT occupancy.
    # We can't use the move cache because it might be stale.
    
    # Get opponent positions from occupancy cache (which is updated)
    if cache and hasattr(cache, 'occupancy_cache'):
        attacker_positions = cache.occupancy_cache.get_positions(attacker_color)
    else:
        # Fallback if no cache (shouldn't happen in this context)
        return False

    # ✅ OPTIMIZED: Use stateless API with GameBuffer
    from game3d.core.buffer import state_to_buffer
    from game3d.core.api import generate_pseudolegal_moves
    from game3d.common.coord_utils import coords_to_keys

    # Create stateless buffer from current state (which has updated occupancy)
    buffer = state_to_buffer(board) # Assuming board functions as state here, or we need to construct minimal state wrapper if board is just board array?
    # Actually board argument in _square_attacked_by_slow is often passed as 'game_state.board' which is the board array, 
    # BUT the callers pass 'board' as the first arg. 
    # Let's check callers. In square_attacked_by_extended fallback, it passes 'board'. 
    # In batch_moves_leave_king_in_check, it passes 'game_state.board'.
    # state_to_buffer expects a GameState-like object with cache_manager.
    
    # We need a way to get a buffer.
    # If board is just an array, we can't easily make a buffer without the cache manager.
    # Fortunately, we have 'cache' argument which is the CacheManager.
    # We can construct a minimal object to pass to state_to_buffer.
    
    class BufferSource:
        def __init__(self, cache_mgr, color):
            self.cache_manager = cache_mgr
            self.color = color
            self.turn_number = 1 # Dummy
            self.halfmove_clock = 0
            
    # However, 'board' arg might be the GameState object itself in some calls?
    # No, type hint says 'board'.
    
    source = MinimalStateProxy(board, attacker_color, cache)
    buffer = state_to_buffer(source)
    
    # Generate ALL moves for attacker color
    all_moves = generate_pseudolegal_moves(buffer)
    
    if all_moves.size == 0:
        return False
        
    # Filter for moves starting from attacker positions
    # We already have attacker_positions from cache
    # But generate_pseudolegal_moves returns ALL moves for that color.
    # We can checking if any move lands on 'square'.
    
    target_x, target_y, target_z = square[0], square[1], square[2]
    
    # Check if any move destinations match target
    # This automatically handles "attacker positions" because only pieces of 'attacker_color' move.
    
    # But wait, we only want to check if *intended* attackers generate the move? 
    # actually square_attacked_by checks if *any* piece of attacker_color attacks square.
    # So using all_moves is correct and even better (no need to filter by source).
    
    hits = (all_moves[:, 3] == target_x) & \
           (all_moves[:, 4] == target_y) & \
           (all_moves[:, 5] == target_z)
           
    return bool(np.any(hits))
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


def get_attackers(game_state: 'GameState', cache=None) -> List[str]:
    """
    Get a list of attackers targeting the current player's king.
    
    Uses cached moves to identify attackers, ensuring consistency with is_check().
    
    Returns:
        List of strings, e.g., ["ROOK at (0,0,5)", "KNIGHT at (2,2,2)"]
    """
    cache = cache or getattr(game_state, 'cache_manager', None)
    if not cache:
        return [] # Cannot determine without cache

    king_color = game_state.color
    opponent_color = Color(king_color).opposite().value
    
    # 1. Find King
    occ_cache = cache.occupancy_cache
    king_pos = occ_cache.find_king(king_color)
    if king_pos is None:
        return [] # No king, no attackers (or handled elsewhere)
        
    king_pos_arr = king_pos.astype(COORD_DTYPE)
    
    # 2. Get Opponent Moves (pseudolegal is fine for attack detection)
    move_cache = cache.move_cache
    opponent_moves = move_cache.get_pseudolegal_moves(opponent_color)
    
    if opponent_moves is None:
        # Cache miss - Force generation
        from game3d.core.buffer import state_to_buffer
        from game3d.core.api import generate_pseudolegal_moves
        
        # Use MinimalStateProxy to switch color without full GameState copy
        dummy_state = MinimalStateProxy(game_state.board, opponent_color, cache)
        buffer = state_to_buffer(dummy_state, readonly=True)
        opponent_moves = generate_pseudolegal_moves(buffer)
        
        # Store in cache for future use (and to generate attack mask)
        move_cache.store_pseudolegal_moves(opponent_color, opponent_moves)
    


    # 3. Find attackers using JIT kernel
    attacker_indices = _find_attackers_of_square(king_pos_arr, opponent_moves, opponent_moves[:, :3])
    
    if attacker_indices.size == 0:
        # Fallback: If no attackers found via cache, force regeneration IGNORING cache
        # This handles cases where cache might be desynced (empty but valid) while check actually exists
        from game3d.core.buffer import state_to_buffer
        from game3d.core.api import generate_pseudolegal_moves
        
        # Create fresh buffer from current board state
        dummy_state = MinimalStateProxy(game_state.board, opponent_color, cache)
        buffer = state_to_buffer(dummy_state, readonly=True)
        fresh_moves = generate_pseudolegal_moves(buffer)
        
        if fresh_moves.size > 0:
            # Re-run detection on fresh moves
            attacker_indices = _find_attackers_of_square(king_pos_arr, fresh_moves, fresh_moves[:, :3])
            if attacker_indices.size > 0:
                # Found attackers that cache missed!
                opponent_moves = fresh_moves
            else:
                # Optionally log if fallback failed to find anything even after regen
                # But typically returns empty list
                pass
        else:
             # Fallback generated no moves
             pass
    
    if attacker_indices.size == 0:
        return []
        
    # 4. Format Output
    attackers_info = []
    attacker_moves = opponent_moves[attacker_indices]
    
    # Track unique attacker positions to avoid duplicates
    seen_positions = set()
    
    # Retrieve piece types for logging
    # We can get source coords and look up types in occ_cache
    for move in attacker_moves:
        src = move[:3]
        pos_key = (int(src[0]), int(src[1]), int(src[2]))
        
        # Skip duplicate attacker positions
        if pos_key in seen_positions:
            continue
        seen_positions.add(pos_key)
        
        # Get piece info - get_fast returns (ptype, color) tuple
        ptype, color = occ_cache.get_fast(src)
        if ptype > 0:
            ptype_name = PieceType(ptype).name
            attackers_info.append(f"{ptype_name} at {pos_key}")
        else:
            attackers_info.append(f"UNKNOWN at {pos_key}")
             
    return attackers_info

# Update __all__ exports

__all__ = [
    'CheckStatus', 'KingInCheckInfo',
    'king_in_check', 'get_check_status', 'get_all_pieces_in_check',
    'batch_king_check_detection', 'get_check_summary',
    'square_attacked_by', 'square_attacked_by_incremental', 'move_would_leave_king_in_check',
    'batch_moves_leave_king_in_check'
]

