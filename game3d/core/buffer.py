
"""
Stateless Game Buffer Definition.
This module defines the raw data structure that holds the entire game state.
"""

import numpy as np
import threading
from typing import NamedTuple, Tuple
from numba import njit
from game3d.common.shared_types import (
    COORD_DTYPE, PIECE_TYPE_DTYPE, COLOR_DTYPE,
    HASH_DTYPE, INDEX_DTYPE, BOOL_DTYPE,
    SIZE, N_PIECE_TYPES, MAX_HISTORY_SIZE
)

# ✅ OPTIMIZATION #4: Thread-local buffer pool for array reuse
_BUFFER_POOL = threading.local()
_SUBSET_POOL = threading.local()  # ✅ OPTIMIZATION #5: Separate pool for subset operations
_POOL_MAX_PIECES = 1024 # Increased to cover full board (729 squares)


class GameBuffer(NamedTuple):
    """
    Immutable Game State Buffer.
    Passed by value (reference to arrays) to Numba functions.
    """
    # 1. Sparse Representation (good for iterating pieces)
    # Shapes: (N, 3), (N,), (N,) - where N is max pieces or current pieces
    occupied_coords: np.ndarray  
    occupied_types: np.ndarray
    occupied_colors: np.ndarray
    occupied_count: int  # Number of active pieces

    # 2. Dense Representation (good for O(1) lookup and kernel compatibility)
    # Split arrays to match existing kernel signatures (occ, ptype)
    board_type: np.ndarray   # (SIZE, SIZE, SIZE), dtype=PIECE_TYPE_DTYPE (int8)
    board_color: np.ndarray  # (SIZE, SIZE, SIZE), dtype=COLOR_DTYPE (uint8)
    
    # 3. Pre-flattened board_color for kernel compatibility
    # Indexed as: idx = x + SIZE*y + SIZE*SIZE*z
    board_color_flat: np.ndarray  # (SIZE^3,), dtype=COLOR_DTYPE

    # 4. Cached Aura Maps (Boolean)
    is_buffed: np.ndarray   # (SIZE, SIZE, SIZE)
    is_debuffed: np.ndarray # (SIZE, SIZE, SIZE)
    is_frozen: np.ndarray   # (SIZE, SIZE, SIZE)

    # 5. Game Metadata
    # [0]: active_color (1=White, 2=Black)
    # [1]: halfmove_clock
    # [2]: turn_number
    # [3]: en_passant_available (0 or 1)
    # [4]: white_king_idx
    # [5]: black_king_idx
    # [6]: white_priest_count
    # [7]: black_priest_count
    meta: np.ndarray

    # 6. History / Zobrist
    zkey: int
    history: np.ndarray  # Circular buffer of last hashes
    history_count: int   # Number of moves in history


@njit(cache=True)
def create_empty_buffer(max_pieces: int = 256) -> GameBuffer:
    """Create a new initialized GameBuffer."""
    occupied_coords = np.zeros((max_pieces, 3), dtype=COORD_DTYPE)
    occupied_types = np.zeros(max_pieces, dtype=PIECE_TYPE_DTYPE)
    occupied_colors = np.zeros(max_pieces, dtype=COLOR_DTYPE)
    
    board_type = np.zeros((SIZE, SIZE, SIZE), dtype=PIECE_TYPE_DTYPE)
    board_color = np.zeros((SIZE, SIZE, SIZE), dtype=COLOR_DTYPE)
    board_color_flat = np.zeros(SIZE * SIZE * SIZE, dtype=COLOR_DTYPE)
    
    # Empty Aura Maps
    is_buffed = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)
    is_debuffed = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)
    is_frozen = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)
    
    # Metadata: White(1) starts, clock 0, turn 1
    meta = np.zeros(10, dtype=INDEX_DTYPE)
    meta[0] = 1 # Active Color
    meta[2] = 1 # Turn Number
    
    history = np.zeros(MAX_HISTORY_SIZE, dtype=HASH_DTYPE)
    
    return GameBuffer(
        occupied_coords,
        occupied_types,
        occupied_colors,
        0, # count
        board_type,
        board_color,
        board_color_flat,
        is_buffed,
        is_debuffed,
        is_frozen,
        meta,
        0, # zkey
        history,
        0 # history_count
    )

def state_to_buffer(state, readonly: bool = False) -> GameBuffer:
    """
    Convert a GameState to a GameBuffer for stateless processing.
    
    Args:
        state: GameState instance
        readonly: If True, reuse existing arrays from cache without copying.
                 USE ONLY for immediate read-only operations (e.g. move generation).
        
    Returns:
        GameBuffer with data copied from state
    """
    from game3d.core.hashing import compute_hash_from_buffer
    
    # Get piece data from cache
    cache = state.cache_manager.occupancy_cache
    
    # ✅ OPTIMIZATION: Zero-Copy Setup
    # We must setup the pool BEFORE calling export_buffer_data to pass the buffers in.
    
    occupied_coords = None
    occupied_types = None
    occupied_colors = None
    n_pieces = 0
    
    if readonly:
        # Get or create thread-local pooled arrays
        if not hasattr(_BUFFER_POOL, 'coords'):
            _BUFFER_POOL.coords = np.zeros((_POOL_MAX_PIECES, 3), dtype=COORD_DTYPE)
            _BUFFER_POOL.types = np.zeros(_POOL_MAX_PIECES, dtype=PIECE_TYPE_DTYPE)
            _BUFFER_POOL.colors = np.zeros(_POOL_MAX_PIECES, dtype=COLOR_DTYPE)
            _BUFFER_POOL.meta = np.zeros(10, dtype=INDEX_DTYPE)
            _BUFFER_POOL.history = np.zeros(MAX_HISTORY_SIZE, dtype=HASH_DTYPE)
            
        # Pass pooled arrays to export to avoid internal allocation
        occ_grid, ptype_grid, all_coords, all_types, all_colors = cache.export_buffer_data(
            out_coords=_BUFFER_POOL.coords,
            out_types=_BUFFER_POOL.types,
            out_colors=_BUFFER_POOL.colors
        )
        
        n_pieces = all_coords.shape[0] # Valid count
        
        # Use simple references (since all_coords is a view of _BUFFER_POOL.coords)
        # We want the FULL buffer for GameBuffer struct usually, but views are fine
        # as long as we pass n_pieces correctly.
        occupied_coords = _BUFFER_POOL.coords
        occupied_types = _BUFFER_POOL.types
        occupied_colors = _BUFFER_POOL.colors
        
    else:
        # Normal allocation path (copying)
        occ_grid, ptype_grid, all_coords, all_types, all_colors = cache.export_buffer_data()
        n_pieces = all_coords.shape[0]
        
        max_pieces = max(512, n_pieces)
        occupied_coords = np.zeros((max_pieces, 3), dtype=COORD_DTYPE)
        occupied_types = np.zeros(max_pieces, dtype=PIECE_TYPE_DTYPE)
        occupied_colors = np.zeros(max_pieces, dtype=COLOR_DTYPE)
        
        if n_pieces > 0:
            occupied_coords[:n_pieces] = all_coords
            occupied_types[:n_pieces] = all_types
            occupied_colors[:n_pieces] = all_colors

    # Full board arrays
    if readonly:
        # ✅ OPTIMIZATION: Zero-copy access for immediate use
        board_color = occ_grid
        board_type = ptype_grid
    else:
        # Explicit copy for buffer safety (immutability)
        board_color = occ_grid.copy()
        board_type = ptype_grid.copy()
    
    # Dense flattened color board - ✅ OPTIMIZED: Use cached view
    board_color_flat = cache.get_flat_occ_view()
    
    # ✅ OPTIMIZATION: Fetch cached Aura Maps
    # These are boolean arrays (SIZE, SIZE, SIZE) for the ACTIVE color
    is_buffed, is_debuffed, is_frozen = state.cache_manager.aura_cache.get_maps(state.color)
    
    # Metadata
    meta = np.zeros(10, dtype=INDEX_DTYPE)
    meta[0] = state.color  # Active Color
    meta[1] = getattr(state, 'halfmove_clock', 0)
    meta[2] = state.turn_number
    
    # Find king indices
    # We can iterate the small sparse arrays
    white_king_idx = -1
    black_king_idx = -1
    
    # Numba optimized search on small array
    # Or simple python loop over n_pieces (N ~ 30)
    for i in range(n_pieces):
        if all_types[i] == 6:  # PieceType.KING
            if all_colors[i] == 1: # White
                white_king_idx = i
            else:
                black_king_idx = i
                
    meta[4] = white_king_idx
    meta[5] = black_king_idx
    meta[6] = cache.get_priest_count(1) # White Priests
    meta[7] = cache.get_priest_count(2) # Black Priests
    
    # History
    history = np.zeros(MAX_HISTORY_SIZE, dtype=HASH_DTYPE)
    history_count = 0
    
    # Create buffer
    buffer = GameBuffer(
        occupied_coords,
        occupied_types,
        occupied_colors,
        n_pieces,
        board_type,
        board_color,
        board_color_flat,
        is_buffed,
        is_debuffed,
        is_frozen,
        meta,
        0,  # zkey placeholder
        history,
        history_count
    )
    
    # Return buffer with correct hash
    return GameBuffer(
        buffer.occupied_coords,
        buffer.occupied_types,
        buffer.occupied_colors,
        buffer.occupied_count,
        buffer.board_type,
        buffer.board_color,
        buffer.board_color_flat,
        buffer.is_buffed,
        buffer.is_debuffed,
        buffer.is_frozen,
        buffer.meta,
        0, # zkey not computed here to save time
        buffer.history,
        buffer.history_count
    )

def state_to_buffer_with_indices(state, indices: list[int], coords: np.ndarray, 
                                types: np.ndarray, colors: np.ndarray, 
                                readonly: bool = True) -> GameBuffer:
    """
    Construct a GameBuffer containing only a subset of pieces (for incremental generation).
    
    Args:
        state: GameState reference (for meta/board access)
        indices: List of indices to include from the provided arrays
        coords: Full sparse coordinate array
        types: Full sparse type array
        colors: Full sparse color array
        readonly: Whether to share board arrays (default True)
    """
    n_subset = len(indices)
    max_pieces = max(512, n_subset)
    
    subset_coords = np.zeros((max_pieces, 3), dtype=COORD_DTYPE)
    subset_types = np.zeros(max_pieces, dtype=PIECE_TYPE_DTYPE)
    subset_colors = np.zeros(max_pieces, dtype=COLOR_DTYPE)
    
    if n_subset > 0:
        # Fancy indexing to extract subset
        idx_arr = np.array(indices, dtype=INDEX_DTYPE)
        subset_coords[:n_subset] = coords[idx_arr]
        subset_types[:n_subset] = types[idx_arr]
        subset_colors[:n_subset] = colors[idx_arr]
        
    # Full board arrays (Context)
    occ_cache = state.cache_manager.occupancy_cache
    if readonly:
        board_color = occ_cache._occ
        board_type = occ_cache._ptype
    else:
        board_color = occ_cache._occ.copy()
        board_type = occ_cache._ptype.copy()
        
    board_color_flat = occ_cache.get_flat_occ_view()  # ✅ OPTIMIZED: Use cached view
    
    # Fetch Aura Maps
    is_buffed, is_debuffed, is_frozen = state.cache_manager.aura_cache.get_maps(state.color)

    # Meta
    meta = np.zeros(10, dtype=INDEX_DTYPE)
    meta[0] = state.color
    meta[1] = getattr(state, 'halfmove_clock', 0)
    meta[2] = state.turn_number
    meta[6] = occ_cache.get_priest_count(1)
    meta[7] = occ_cache.get_priest_count(2)
    
    # We don't strictly need King indices for move generation usually, 
    # unless specific move logic depends on it. 
    # generate_moves only asks for active_color from meta[0].
    
    return GameBuffer(
        subset_coords,
        subset_types,
        subset_colors,
        n_subset,
        board_type,
        board_color,
        board_color_flat,
        is_buffed,
        is_debuffed,
        is_frozen,
        meta,
        0, # No hash needed
        np.zeros(MAX_HISTORY_SIZE, dtype=HASH_DTYPE),
        0
    )

def state_to_buffer_from_pieces(state, coords: np.ndarray, types: np.ndarray, 
                               colors: np.ndarray, readonly: bool = True) -> GameBuffer:
    """
    Construct a GameBuffer from raw piece arrays (DIRECT construction).
    
    ✅ OPTIMIZED: Uses thread-local buffer pool to avoid allocations.
    
    Optimized for incremental updates where we already have the specific
    coordinates and attributes of the pieces we want to simulate.
    avoids the O(N) scan of 'export_buffer_data' + filtering.
    
    Args:
        state: GameState reference (for meta/board access)
        coords: (K, 3) Array of coordinates for the subset
        types: (K,) Array of piece types
        colors: (K,) Array of piece colors
        readonly: Whether to share board arrays (default True)
    """
    n_subset = coords.shape[0]
    
    # ✅ OPTIMIZATION: Use thread-local pooled arrays
    if not hasattr(_SUBSET_POOL, 'coords'):
        _SUBSET_POOL.coords = np.zeros((_POOL_MAX_PIECES, 3), dtype=COORD_DTYPE)
        _SUBSET_POOL.types = np.zeros(_POOL_MAX_PIECES, dtype=PIECE_TYPE_DTYPE)
        _SUBSET_POOL.colors = np.zeros(_POOL_MAX_PIECES, dtype=COLOR_DTYPE)
        _SUBSET_POOL.meta = np.zeros(10, dtype=INDEX_DTYPE)
        _SUBSET_POOL.history = np.zeros(MAX_HISTORY_SIZE, dtype=HASH_DTYPE)
    
    if n_subset <= _POOL_MAX_PIECES:
        # Reuse pooled arrays (zero-copy for read-only)
        subset_coords = _SUBSET_POOL.coords
        subset_types = _SUBSET_POOL.types
        subset_colors = _SUBSET_POOL.colors
        meta = _SUBSET_POOL.meta
        history = _SUBSET_POOL.history
        
        # Fill with data
        if n_subset > 0:
            subset_coords[:n_subset] = coords
            subset_types[:n_subset] = types
            subset_colors[:n_subset] = colors
    else:
        # Fallback for very large piece counts
        max_pieces = max(512, n_subset)
        subset_coords = np.zeros((max_pieces, 3), dtype=COORD_DTYPE)
        subset_types = np.zeros(max_pieces, dtype=PIECE_TYPE_DTYPE)
        subset_colors = np.zeros(max_pieces, dtype=COLOR_DTYPE)
        meta = np.zeros(10, dtype=INDEX_DTYPE)
        history = np.zeros(MAX_HISTORY_SIZE, dtype=HASH_DTYPE)
        
        if n_subset > 0:
            subset_coords[:n_subset] = coords
            subset_types[:n_subset] = types
            subset_colors[:n_subset] = colors
        
    # Full board arrays (Context)
    occ_cache = state.cache_manager.occupancy_cache
    if readonly:
        board_color = occ_cache._occ
        board_type = occ_cache._ptype
    else:
        board_color = occ_cache._occ.copy()
        board_type = occ_cache._ptype.copy()
        
    board_color_flat = occ_cache.get_flat_occ_view()  # ✅ OPTIMIZED: Use cached view
    
    # Fetch Aura Maps
    is_buffed, is_debuffed, is_frozen = state.cache_manager.aura_cache.get_maps(state.color)

    # Meta
    meta[0] = state.color
    meta[1] = getattr(state, 'halfmove_clock', 0)
    meta[2] = state.turn_number
    meta[6] = occ_cache.get_priest_count(1)
    meta[7] = occ_cache.get_priest_count(2)
    
    return GameBuffer(
        subset_coords,
        subset_types,
        subset_colors,
        n_subset,
        board_type,
        board_color,
        board_color_flat,
        is_buffed,
        is_debuffed,
        is_frozen,
        meta,
        0, # No hash needed
        history,
        0
    )

