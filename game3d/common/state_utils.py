"""State utilities for 9x9x9 chess engine - optimized for numpy-native operations."""
from collections import defaultdict
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

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
    """Create new game state with proper position handling."""
    if reuse_cache:
        cache_manager = original_state.cache_manager
        if not hasattr(cache_manager, 'zobrist_cache'):
            raise ValueError("Invalid cache manager provided")
        cache_manager.board = new_board
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

    turn_number = original_state.turn_number
    halfmove_clock = original_state.halfmove_clock

    if increment_turn:
        turn_number += 1
        halfmove_clock += 1

    from game3d.game.gamestate import GameState
    new_state = GameState(
        board=new_board,
        color=new_color,
        cache_manager=cache_manager,
        history=new_history,
        halfmove_clock=halfmove_clock,
        turn_number=turn_number,
    )

    # Handle position counts for threefold repetition
    if hasattr(original_state, '_position_counts'):
        new_state._position_counts = original_state._position_counts.copy()
        new_state._position_keys = original_state._position_keys.copy()
    new_state._update_position_counts(new_state.zkey, 1)

    return new_state


# =============================================================================
# BATCH STATE OPERATIONS - VECTORIZED FOR PERFORMANCE
# =============================================================================

def create_batch_states(original_states, new_boards, new_colors, moves=None,
                       increment_turns=None, reuse_caches=True):
    """
    Create multiple new game states in batch for vectorized operations.
    
    Args:
        original_states: List of original GameState objects
        new_boards: List or array of new board states (N, N_TOTAL_PLANES, SIZE, SIZE, SIZE)
        new_colors: Array-like of new colors (N,)
        moves: Optional list of moves corresponding to each state
        increment_turns: Optional boolean array indicating turn increment (N,)
        reuse_caches: Boolean or array-like for cache reuse (N,)
    
    Returns:
        List of new GameState objects
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
    
    # Optimized batch state creation - vectorized approach
    new_states = []
    
    # Pre-extract arrays for better memory access patterns
    colors_array = new_colors.astype(np.uint8)
    increment_array = increment_turns.astype(BOOL_DTYPE)
    cache_array = reuse_caches.astype(BOOL_DTYPE)
    
    # Batch process with minimal overhead
    for i in range(n_states):
        # Direct attribute access for performance
        orig_state = original_states[i]
        new_board = new_boards[i]
        
        # Optimized state creation with reduced overhead
        if cache_array[i]:
            cache_manager = orig_state.cache_manager
            if hasattr(cache_manager, 'zobrist_cache'):
                cache_manager.board = new_board
        # Optimized history handling
        move = moves[i]
        if move is not None:
            new_history = orig_state.history + (move,)
        # Optimized turn handling
        if increment_array[i]:
            turn_number = orig_state.turn_number + 1
            halfmove_clock = orig_state.halfmove_clock + 1
        from game3d.game.gamestate import GameState
        new_state = GameState(
            board=new_board,
            color=int(colors_array[i]),
            cache_manager=cache_manager,
            history=new_history,
            halfmove_clock=halfmove_clock,
            turn_number=turn_number,
        )

        # Optimized position counts
        if hasattr(orig_state, '_position_counts'):
            new_state._position_counts = orig_state._position_counts.copy()
            new_state._position_keys = orig_state._position_keys.copy()
        new_state._update_position_counts(new_state.zkey, 1)
        new_states.append(new_state)
    
    return new_states


# =============================================================================
# STATE VALIDATION AND COMPARISON
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def validate_state_arrays(board, color, turn_number=None, halfmove_clock=None):
    """
    Validate state arrays using numpy-native operations and shared_types constants.
    
    Args:
        board: Board state array (N_TOTAL_PLANES, SIZE, SIZE, SIZE)
        color: Color value (should match Color enum values)
        turn_number: Optional turn number for validation
        halfmove_clock: Optional halfmove clock for validation
    
    Returns:
        bool: True if all validations pass
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


@njit(cache=True, fastmath=True, parallel=True)
def compare_states_batch(state1_list, state2_list):
    """
    Compare multiple state pairs in batch using vectorized operations with stacked arrays.
    
    Args:
        state1_list: List of first states to compare
        state2_list: List of second states to compare
    
    Returns:
        NDArray: Boolean array indicating state equality (N,)
    """
    if len(state1_list) != len(state2_list):
        raise ValueError("State lists must have the same length")
    
    n_states = len(state1_list)
    if n_states == 0:
        return np.array([], dtype=BOOL_DTYPE)
    
    # Optimized vectorized comparison using stacked arrays
        # Stack all boards for batch comparison
        boards1 = np.stack([state1.board for state1 in state1_list], axis=0)
        boards2 = np.stack([state2.board for state2 in state2_list], axis=0)
        
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
    """
    Get detailed differences between two states.
    
    Args:
        state1: First state
        state2: Second state
    
    Returns:
        dict: Dictionary with difference information
    """
    differences = {}
    
    # Board differences
    if not np.array_equal(state1.board, state2.board):
        board_diff = state1.board != state2.board
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

@njit(cache=True, fastmath=True, parallel=True)
def get_state_hash_batch(states):
    """
    Get Zobrist hash values for multiple states in batch using vectorized operations.
    
    Args:
        states: List of GameState objects
    
    Returns:
        NDArray: Array of hash values (N,)
    """
    n_states = len(states)
    if n_states == 0:
        return np.array([], dtype=INDEX_DTYPE)
    
    # Optimized vectorized hash extraction
        # Direct array extraction for better performance
        hashes = np.array([state.zkey for state in states], dtype=INDEX_DTYPE)
        return hashes
@njit(cache=True, fastmath=True, parallel=True)
def is_state_valid_batch(states):
    """
    Validate multiple states in batch using vectorized operations.
    
    Args:
        states: List of GameState objects
    
    Returns:
        NDArray: Boolean array indicating validity (N,)
    """
    n_states = len(states)
    if n_states == 0:
        return np.array([], dtype=BOOL_DTYPE)
    
    # Vectorized batch validation for better performance
        # Extract all state attributes in vectorized manner
        boards = np.stack([state.board for state in states], axis=0)
        colors = np.array([state.color for state in states], dtype=np.uint8)
        turn_numbers = np.array([state.turn_number for state in states], dtype=INDEX_DTYPE)
        halfmove_clocks = np.array([state.halfmove_clock for state in states], dtype=INDEX_DTYPE)
        
        # Vectorized validation
        results = np.ones(n_states, dtype=BOOL_DTYPE)
        
        # Board shape validation (vectorized)
        expected_shape = (N_TOTAL_PLANES, SIZE, SIZE, SIZE)
        shape_valid = np.all(boards.shape[1:] == expected_shape[1:], axis=0)
        results &= shape_valid
        
        # Board dtype validation (vectorized)
        dtype_valid = boards.dtype == FLOAT_DTYPE
        results &= dtype_valid
        
        # Color validation (vectorized)
        valid_colors = np.array([Color.EMPTY.value, Color.WHITE.value, Color.BLACK.value], dtype=np.uint8)
        color_valid = np.isin(colors, valid_colors)
        results &= color_valid
        
        # Turn number validation (vectorized)
        turn_valid = (turn_numbers >= 0) & np.isin(turn_numbers.dtype, [np.int32, np.int64, int])
        results &= turn_valid
        
        # Halfmove clock validation (vectorized)
        halfmove_valid = (halfmove_clocks >= 0) & np.isin(halfmove_clocks.dtype, [np.int32, np.int64, int])
        results &= halfmove_valid
        
        # Cache manager validation (vectorized)
        cache_valid = np.array([
            hasattr(state, 'cache_manager') and state.cache_manager is not None 
            for state in states
        ], dtype=BOOL_DTYPE)
        results &= cache_valid
        
        return results
        
@njit(cache=True, fastmath=True, parallel=True)
def get_state_metrics_batch(states):
    """
    Get performance metrics for multiple states in batch using vectorized operations.
    
    Args:
        states: List of GameState objects
    
    Returns:
        dict: Dictionary with metric arrays
    """
    n_states = len(states)
    if n_states == 0:
        return {
            'turn_numbers': np.array([], dtype=INDEX_DTYPE),
            'halfmove_clocks': np.array([], dtype=INDEX_DTYPE),
            'colors': np.array([], dtype=np.uint8),
            'hashes': np.array([], dtype=INDEX_DTYPE),
            'board_nonzero_counts': np.array([], dtype=INDEX_DTYPE)
        }
    
    # Optimized vectorized metrics extraction
        # Extract attributes in vectorized manner
        turn_numbers = np.array([state.turn_number for state in states], dtype=INDEX_DTYPE)
        halfmove_clocks = np.array([state.halfmove_clock for state in states], dtype=INDEX_DTYPE)
        colors = np.array([state.color for state in states], dtype=np.uint8)
        hashes = np.array([state.zkey for state in states], dtype=INDEX_DTYPE)
        
        # Vectorized board non-zero count calculation
        boards = np.stack([state.board for state in states], axis=0)
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
