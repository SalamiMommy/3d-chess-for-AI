"""State utilities for 9x9x9 chess engine - FIXED CACHE SYNC."""
from collections import defaultdict
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
import logging

logger = logging.getLogger(__name__)

# Import shared types and constants
from game3d.common.shared_types import (
    INDEX_DTYPE, FLOAT_DTYPE, BOOL_DTYPE, SIZE, N_TOTAL_PLANES,
    VECTORIZATION_THRESHOLD, MAX_BATCH_SIZE, DEFAULT_BATCH_SIZE,
    Color, MOVE_DTYPE
)

# Type aliases for better readability
StateArray = NDArray
CoordArray = NDArray

def create_new_state(original_state, new_board, new_color, move=None,
                    increment_turn=True, reuse_cache=True):
    """
    Create new game state with PROPER CACHE SYNCHRONIZATION.

    CRITICAL FIX: Cache manager must be synchronized with new board state.
    """
    # === STEP 1: Handle Cache Manager ===
    if reuse_cache:
        cache_manager = original_state.cache_manager
        if not hasattr(cache_manager, 'zobrist_cache'):
            raise ValueError("Invalid cache manager provided")

        # Update cache manager with new board reference
        cache_manager._board = new_board
        cache_manager._board_generation = getattr(new_board, 'generation', 0)

        # 🔥 CACHE-ONLY FIX: Trust incremental updates, no rebuild needed
        # Cache is already correct from incremental updates in make_move()
        # Rebuilding from board would be wasteful and error-prone

        # Recompute Zobrist from current cache state
        zkey = cache_manager._compute_initial_zobrist(new_color)
        cache_manager._zkey = zkey
    else:
        # Create fresh cache manager if not reusing
        from game3d.cache.manager import OptimizedCacheManager
        cache_manager = OptimizedCacheManager(board=new_board, color=new_color)
        zkey = cache_manager._zkey

    # === STEP 2: Handle Move History ===
    new_history = original_state.history
    if move is not None:
        # Convert move to structured array for history
        if isinstance(move, np.ndarray) and move.dtype != MOVE_DTYPE:
            # Assume move is (6,) or (1, 6) coordinate array
            move_struct = np.zeros(1, dtype=MOVE_DTYPE)
            move_flat = move.flatten()
            move_struct['from_x'] = move_flat[0]
            move_struct['from_y'] = move_flat[1]
            move_struct['from_z'] = move_flat[2]
            move_struct['to_x'] = move_flat[3]
            move_struct['to_y'] = move_flat[4]
            move_struct['to_z'] = move_flat[5]
            # Flags would need to be passed or inferred, defaulting to 0
            new_history = np.concatenate((new_history, move_struct))
        else:
            new_history = np.concatenate((new_history, np.atleast_1d(move)))

    # === STEP 3: Handle Turn Counters ===
    turn_number = original_state.turn_number
    halfmove_clock = original_state.halfmove_clock

    if increment_turn:
        turn_number += 1
        halfmove_clock += 1

    # === STEP 4: Create New State ===
    from game3d.game.gamestate import GameState
    new_state = GameState(
        board=new_board,
        color=new_color,
        cache_manager=cache_manager,
        history=new_history,
        halfmove_clock=halfmove_clock,
        turn_number=turn_number,
    )

    # 🔥 CRITICAL FIX: Ensure Zobrist is synchronized
    new_state._zkey = zkey

    # === STEP 5: Handle Position Counts (Threefold Repetition) ===
    if hasattr(original_state, '_position_counts'):
        new_state._position_counts = original_state._position_counts.copy()
        new_state._position_keys = original_state._position_keys.copy()
    new_state._update_position_counts(new_state.zkey, 1)

    # === STEP 6: Verify Synchronization (Debug Mode) ===
    if logger.isEnabledFor(logging.DEBUG):
        # Verify king positions are cached correctly
        for color in [Color.WHITE, Color.BLACK]:
            king_pos = cache_manager.occupancy_cache.find_king(color)
            if king_pos is None:
                logger.warning(f"King for color {color} not found after state creation")
            else:
                logger.debug(f"King for color {color} cached at {king_pos}")

    return new_state


# =============================================================================
# BATCH STATE OPERATIONS - VECTORIZED FOR PERFORMANCE
# =============================================================================

def create_batch_states(original_states, new_boards, new_colors, moves=None,
                       increment_turns=None, reuse_caches=True):
    """
    Create multiple new game states in batch for vectorized operations.

    🔥 FIXED: Each state now properly synchronizes its cache.
    """
    n_states = len(original_states)

    # Ensure numpy-native operations
    new_boards = np.asarray(new_boards, dtype=FLOAT_DTYPE)
    new_colors = np.asarray(new_colors, dtype=np.uint8)

    # Handle optional parameters with defaults
    if increment_turns is None:
        increment_turns = np.ones(n_states, dtype=BOOL_DTYPE)
    if moves is None:
        moves = [None] * n_states

    if reuse_caches is True:
        reuse_caches = np.ones(n_states, dtype=BOOL_DTYPE)
    elif isinstance(reuse_caches, bool):
        reuse_caches = np.full(n_states, reuse_caches, dtype=BOOL_DTYPE)

    # Batch validation
    if n_states > MAX_BATCH_SIZE:
        # Process in chunks to avoid memory issues
        chunk_size = DEFAULT_BATCH_SIZE
        result_states = []

        for i in range(0, n_states, chunk_size):
            end_idx = min(i + chunk_size, n_states)
            chunk_states = create_batch_states(
                original_states[i:end_idx],
                new_boards[i:end_idx],
                new_colors[i:end_idx],
                moves[i:end_idx],
                increment_turns[i:end_idx],
                reuse_caches[i:end_idx]
            )
            result_states.extend(chunk_states)

        return result_states

    # 🔥 FIXED: Use create_new_state for each state to ensure proper sync
    new_states = []

    for i in range(n_states):
        new_state = create_new_state(
            original_state=original_states[i],
            new_board=new_boards[i],
            new_color=int(new_colors[i]),
            move=moves[i],
            increment_turn=bool(increment_turns[i]),
            reuse_cache=bool(reuse_caches[i])
        )
        new_states.append(new_state)

    return new_states


# =============================================================================
# STATE VALIDATION AND COMPARISON
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def validate_state_arrays(board, color, turn_number=None, halfmove_clock=None):
    """
    Validate state arrays using numpy-native operations and shared_types constants.
    """
    # Board shape validation
    expected_shape = (N_TOTAL_PLANES, SIZE, SIZE, SIZE)
    if board.shape != expected_shape:
        return False

    # Board dtype validation
    if board.dtype != FLOAT_DTYPE:
        return False

    # Color validation using shared_types
    valid_colors = {Color.EMPTY.value, Color.WHITE.value, Color.BLACK.value}
    if color not in valid_colors:
        return False

    # Optional numeric validations
    if turn_number is not None and (turn_number < 0 or not isinstance(turn_number, (int, np.integer))):
        return False

    if halfmove_clock is not None and (halfmove_clock < 0 or not isinstance(halfmove_clock, (int, np.integer))):
        return False

    return True


def compare_states_batch(state1_list, state2_list):
    """Compare multiple state pairs in batch using vectorized operations."""
    if len(state1_list) != len(state2_list):
        raise ValueError("State lists must have the same length")

    n_states = len(state1_list)
    if n_states == 0:
        return np.array([], dtype=BOOL_DTYPE)

    # Stack all boards for batch comparison
    boards1 = np.stack([state1.board.array() for state1 in state1_list], axis=0)
    boards2 = np.stack([state2.board.array() for state2 in state2_list], axis=0)

    # Vectorized board comparison
    boards_equal = np.all(boards1 == boards2, axis=(1, 2, 3, 4))

    # Vectorized attribute comparison
    colors1 = np.array([state1.color for state1 in state1_list], dtype=np.uint8)
    colors2 = np.array([state2.color for state2 in state2_list], dtype=np.uint8)
    colors_equal = colors1 == colors2

    turns1 = np.array([state1.turn_number for state1 in state1_list], dtype=INDEX_DTYPE)
    turns2 = np.array([state2.turn_number for state2 in state2_list], dtype=INDEX_DTYPE)
    turns_equal = turns1 == turns2

    halfmoves1 = np.array([state1.halfmove_clock for state1 in state1_list], dtype=INDEX_DTYPE)
    halfmoves2 = np.array([state2.halfmove_clock for state2 in state2_list], dtype=INDEX_DTYPE)
    halfmoves_equal = halfmoves1 == halfmoves2

    # Combine all comparisons
    results = boards_equal & colors_equal & turns_equal & halfmoves_equal

    return results.astype(BOOL_DTYPE)


def get_state_differences(state1, state2):
    """Get detailed differences between two states."""
    differences = {}

    # Board differences
    board1 = state1.board.array()
    board2 = state2.board.array()
    if not np.array_equal(board1, board2):
        board_diff = board1 != board2
        differences['board'] = {
            'changed_planes': np.where(np.any(board_diff, axis=(1, 2, 3)))[0],
            'total_changed_positions': np.sum(board_diff)
        }

    # Color differences
    if state1.color != state2.color:
        differences['color'] = {'from': state1.color, 'to': state2.color}

    # Turn number differences
    if state1.turn_number != state2.turn_number:
        differences['turn_number'] = {'from': state1.turn_number, 'to': state2.turn_number}

    # Halfmove clock differences
    if state1.halfmove_clock != state2.halfmove_clock:
        differences['halfmove_clock'] = {'from': state1.halfmove_clock, 'to': state2.halfmove_clock}

    return differences


# =============================================================================
# UTILITY FUNCTIONS FOR OPTIMIZED STATE MANAGEMENT
# =============================================================================

def get_state_hash_batch(states):
    """Get Zobrist hash values for multiple states in batch."""
    n_states = len(states)
    if n_states == 0:
        return np.array([], dtype=INDEX_DTYPE)

    hashes = np.array([state.zkey for state in states], dtype=INDEX_DTYPE)
    return hashes


def is_state_valid_batch(states):
    """Validate multiple states in batch."""
    n_states = len(states)
    if n_states == 0:
        return np.array([], dtype=BOOL_DTYPE)

    results = np.ones(n_states, dtype=BOOL_DTYPE)

    for i, state in enumerate(states):
        try:
            # Check board shape
            board = state.board.array()
            expected_shape = (N_TOTAL_PLANES, SIZE, SIZE, SIZE)
            if board.shape != expected_shape:
                results[i] = False
                continue

            # Check color validity
            if state.color not in [Color.EMPTY.value, Color.WHITE.value, Color.BLACK.value]:
                results[i] = False
                continue

            # Check cache manager
            if not hasattr(state, 'cache_manager') or state.cache_manager is None:
                results[i] = False
                continue

        except Exception as e:
            logger.warning(f"State validation failed for index {i}: {e}")
            results[i] = False

    return results


def get_state_metrics_batch(states):
    """Get performance metrics for multiple states in batch."""
    n_states = len(states)
    if n_states == 0:
        return {
            'turn_numbers': np.array([], dtype=INDEX_DTYPE),
            'halfmove_clocks': np.array([], dtype=INDEX_DTYPE),
            'colors': np.array([], dtype=np.uint8),
            'hashes': np.array([], dtype=INDEX_DTYPE),
            'board_nonzero_counts': np.array([], dtype=INDEX_DTYPE)
        }

    # Extract attributes in vectorized manner
    turn_numbers = np.array([state.turn_number for state in states], dtype=INDEX_DTYPE)
    halfmove_clocks = np.array([state.halfmove_clock for state in states], dtype=INDEX_DTYPE)
    colors = np.array([state.color for state in states], dtype=np.uint8)
    hashes = np.array([state.zkey for state in states], dtype=INDEX_DTYPE)

    # Vectorized board non-zero count calculation
    boards = np.stack([state.board.array() for state in states], axis=0)
    board_nonzero_counts = np.count_nonzero(boards, axis=(1, 2, 3, 4)).astype(INDEX_DTYPE)

    return {
        'turn_numbers': turn_numbers,
        'halfmove_clocks': halfmove_clocks,
        'colors': colors,
        'hashes': hashes,
        'board_nonzero_counts': board_nonzero_counts
    }


# Module exports for optimized state utilities
__all__ = [
    # Core state creation
    'create_new_state',

    # Batch operations (vectorized)
    'create_batch_states',
    'compare_states_batch',
    'get_state_hash_batch',
    'is_state_valid_batch',
    'get_state_metrics_batch',

    # Validation and comparison
    'validate_state_arrays',
    'get_state_differences',

    # Constants and types
    'INDEX_DTYPE', 'FLOAT_DTYPE', 'BOOL_DTYPE', 'SIZE', 'N_TOTAL_PLANES',
    'VECTORIZATION_THRESHOLD', 'MAX_BATCH_SIZE', 'DEFAULT_BATCH_SIZE',
    'StateArray', 'CoordArray',
]
