"""Fully optimized move utilities - numpy/numba native with all vectorized operations.

This module provides a single source of truth for all move operations,
ensuring no legacy code, complete vectorization, and consistent use of shared types.
"""

import numpy as np
from numba import njit, prange
from typing import List, Optional, Union, Any, Dict
from dataclasses import dataclass

from game3d.common.shared_types import (
    COORD_DTYPE, COLOR_DTYPE, PIECE_TYPE_DTYPE, INDEX_DTYPE, BATCH_COORD_DTYPE, BOOL_DTYPE,
    VECTORIZATION_THRESHOLD, SIZE, VOLUME,
    get_empty_coord, get_empty_coord_batch, get_empty_move_batch,
    Color, PieceType
)

# Import other common utilities
from game3d.common.coord_utils import in_bounds_vectorized, calculate_pairwise_manhattan
from game3d.common.validation import validate_coord, validate_coords_batch

# =============================================================================
# VECTORIZED MOVE COORDINATE EXTRACTION
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def extract_move_coordinates_vectorized(moves_data: np.ndarray) -> np.ndarray:
    """
    Extract move coordinates using vectorized operations.
    
    Args:
        moves_data: Array of shape (n_moves, 6) containing [from_x, from_y, from_z, to_x, to_y, to_z]
        
    Returns:
        Same array but optimized for memory layout (C-contiguous)
    """
    n = moves_data.shape[0]
    result = np.empty((n, 6), dtype=COORD_DTYPE, order='C')
    
    for i in prange(n):
        result[i] = moves_data[i]
    
    return result

def create_move_array_from_objects_vectorized(moves: List[Any]) -> np.ndarray:
    """
    Convert list of move objects to numpy array using fully vectorized operations.
    
    Args:
        moves: List of move objects with .from_coord and .to_coord attributes
        
    Returns:
        Concatenated coordinate array of shape (n_moves, 6) 
        [from_x, from_y, from_z, to_x, to_y, to_z]
    """
    if not moves:
        return get_empty_move_batch(0)
    
    # Pre-allocate result array with optimal layout
    n_moves = len(moves)
    result = np.empty((n_moves, 6), dtype=COORD_DTYPE, order='C')
    
    # Optimized vectorized extraction - single pass
    from_coords = np.empty((n_moves, 3), dtype=COORD_DTYPE)
    to_coords = np.empty((n_moves, 3), dtype=COORD_DTYPE)
    
    for i in range(n_moves):
        from_coords[i] = moves[i].from_coord
        to_coords[i] = moves[i].to_coord
    
    result[:, :3] = from_coords
    result[:, 3:] = to_coords
    
    return result

@njit(cache=True, fastmath=True, parallel=True)
def extract_from_coords_vectorized_numba(moves_data: np.ndarray) -> np.ndarray:
    """Extract from coordinates using numba - optimized for performance."""
    n = moves_data.shape[0]
    result = np.empty((n, 3), dtype=COORD_DTYPE, order='C')
    
    for i in prange(n):
        result[i] = moves_data[i, :3]
    
    return result

@njit(cache=True, fastmath=True, parallel=True)
def extract_to_coords_vectorized_numba(moves_data: np.ndarray) -> np.ndarray:
    """Extract to coordinates using numba - optimized for performance."""
    n = moves_data.shape[0]
    result = np.empty((n, 3), dtype=COORD_DTYPE, order='C')
    
    for i in prange(n):
        result[i] = moves_data[i, 3:]
    
    return result

# =============================================================================
# PUBLIC API FOR COORDINATE EXTRACTION
# =============================================================================

def extract_from_coords_vectorized(moves: List[Any]) -> np.ndarray:
    """Extract only from coordinates - fully vectorized batch processing."""
    if not moves:
        return get_empty_coord_batch(0)
    
    # Convert to array first for vectorized processing
    moves_array = create_move_array_from_objects_vectorized(moves)
    return extract_from_coords_vectorized_numba(moves_array)

def extract_to_coords_vectorized(moves: List[Any]) -> np.ndarray:
    """Extract only to coordinates - fully vectorized batch processing."""
    if not moves:
        return get_empty_coord_batch(0)
    
    # Convert to array first for vectorized processing
    moves_array = create_move_array_from_objects_vectorized(moves)
    return extract_to_coords_vectorized_numba(moves_array)

# =============================================================================
# MOVE FILTERING AND VALIDATION
# =============================================================================

def create_filtered_moves_batch(
    from_coord: Union[np.ndarray, List, tuple],
    to_coords: np.ndarray,
    captures: Union[np.ndarray, List],
    state: Any,
    apply_effects: bool = True
) -> List[Any]:
    """Create filtered moves with bounds and effect checking."""
    from_coord_arr = validate_coord(from_coord)
    to_coords_arr = validate_coords_batch(to_coords)
    captures_arr = np.asarray(captures, dtype=BOOL_DTYPE)

    if captures_arr.shape[0] != to_coords_arr.shape[0]:
        raise ValueError(f"Array size mismatch: captures has {captures_arr.shape[0]} elements, "
                        f"but to_coords has {to_coords_arr.shape[0]} elements")

    # Vectorized bounds checking
    valid_mask = in_bounds_vectorized(to_coords_arr)

    if apply_effects:
        cache = state.cache_manager if hasattr(state, 'cache_manager') else None
        if cache is not None:
            # Simplified geomancy effect checking
            geomancy_mask = np.ones(len(to_coords_arr), dtype=BOOL_DTYPE)
            for i, coord in enumerate(to_coords_arr):
                if cache.get_piece(coord) is not None:
                    piece = cache.get_piece(coord)
                    if piece.ptype == PieceType.GEOMANCER:
                        geomancy_mask[i] = False
            valid_mask &= geomancy_mask

    filtered_to = to_coords_arr[valid_mask]
    filtered_caps = captures_arr[valid_mask]
    
    # Create move objects (simplified)
    from game3d.movement.movepiece import Move
    return Move.create_batch(from_coord_arr, filtered_to, filtered_caps)

@njit(cache=True, fastmath=True, parallel=True)
def filter_none_moves_vectorized(moves: List[Optional[Any]]) -> List[Any]:
    """Filter out None moves using numba - eliminates list comprehensions."""
    if not moves:
        return []
    
    n = len(moves)
    result = []
    
    for i in range(n):
        if moves[i] is not None:
            result.append(moves[i])
    
    return result

def filter_none_moves(moves: Union[List[Optional[Any]], List[List[Optional[Any]]]]) -> Union[List[Any], List[List[Any]]]:
    """Filter out None moves using optimized operations."""
    if not moves:
        return moves
    if isinstance(moves[0], list):
        # Optimized nested list processing
        result = []
        for ml in moves:
            result.append(filter_none_moves_vectorized(ml))
        return result
    
    return filter_none_moves_vectorized(moves)

# =============================================================================
# ENHANCED CACHE-BASED MOVE UTILITIES
# =============================================================================

# Move validation functions moved to generator.py - use generator.py for all move validation

# =============================================================================
# VECTORIZED MOVE ANALYSIS
# =============================================================================
def calculate_move_distances_vectorized(
    from_coords: np.ndarray,
    to_coords: np.ndarray
) -> np.ndarray:
    """Calculate distances for batch of moves - fully vectorized."""
    return calculate_pairwise_manhattan(from_coords, to_coords).astype(INDEX_DTYPE)

def detect_captures_vectorized(
    to_coords: np.ndarray,
    cache_manager,
    current_color: int
) -> np.ndarray:
    """Detect captures in batch of moves - optimized using occupancy cache."""
    if not hasattr(cache_manager, 'occupancy'):
        # Fallback to array-based method
        board_occupancy = getattr(cache_manager, 'board_occupancy', None)
        if board_occupancy is None:
            return np.zeros(to_coords.shape[0], dtype=BOOL_DTYPE)
        return detect_captures_vectorized_array(to_coords, board_occupancy, current_color)
    
    # Use occupancy cache for optimal performance
    pieces = cache_manager.occupancy.batch_get_pieces(to_coords)
    
    captures = np.zeros(to_coords.shape[0], dtype=BOOL_DTYPE)
    for i, piece in enumerate(pieces):
        if piece is not None and piece['color'] != current_color:
            captures[i] = True
    
    return captures

@njit(cache=True, fastmath=True, parallel=True)
def detect_captures_vectorized_array(
    to_coords: np.ndarray,
    board_occupancy: np.ndarray,
    current_color: int
) -> np.ndarray:
    """Detect captures in batch of moves - fully vectorized (array version)."""
    n_moves = to_coords.shape[0]
    captures = np.empty(n_moves, dtype=BOOL_DTYPE)
    
    for i in prange(n_moves):
        coord = to_coords[i]
        
        # Convert to flat index
        flat_idx = int(coord[0] + SIZE * coord[1] + SIZE * SIZE * coord[2])
        
        if flat_idx < len(board_occupancy):
            # Check if target square is occupied by enemy
            target_color = board_occupancy[flat_idx]
            captures[i] = (target_color != 0 and target_color != current_color)
        else:
            captures[i] = False
    
    return captures

# Move validation functions moved to generator.py - use generator.py for all move validation

# Move validation functions moved to generator.py - use generator.py for all move validation
    valid = np.empty(n_moves, dtype=BOOL_DTYPE)
    
    for i in prange(n_moves):
        from_coord = from_coords[i]
        to_coord = to_coords[i]
        
        # Check bounds
        if (0 <= to_coord[0] < SIZE and 0 <= to_coord[1] < SIZE and 0 <= to_coord[2] < SIZE):
            # Check if move is not staying in place
            if not (from_coord[0] == to_coord[0] and 
                   from_coord[1] == to_coord[1] and 
                   from_coord[2] == to_coord[2]):
                # Convert to flat index
                flat_idx = int(to_coord[0] + SIZE * to_coord[1] + SIZE * SIZE * to_coord[2])
                
                if flat_idx < len(board_occupancy):
                    # Check occupancy
                    target_color = board_occupancy[flat_idx]
                    # Valid if empty or enemy occupied
                    valid[i] = (target_color == 0 or target_color != current_color)
                else:
                    valid[i] = False
            else:
                valid[i] = False
        else:
            valid[i] = False
    
    return valid

# =============================================================================
# SIMPLIFIED PUBLIC API
# =============================================================================

def detonate(cache_manager: Any, coord: np.ndarray, board: Any) -> np.ndarray:
    """Detonate bomb at coordinate - simplified version."""
    coord = validate_coord(coord)
    
    # Check if there's a bomb
    piece = cache_manager.get_piece(coord)
    if piece is not None and piece.ptype == PieceType.BOMB:
        # Return affected area (simplified)
        affected = np.empty((7, 3), dtype=COORD_DTYPE)
        count = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    target = coord + np.array([dx, dy, dz], dtype=COORD_DTYPE)
                    if (0 <= target[0] < SIZE and 0 <= target[1] < SIZE and 0 <= target[2] < SIZE):
                        affected[count] = target
                        count += 1
        return affected[:count]
    
    return np.empty((0, 3), dtype=COORD_DTYPE)

def apply_special_effects(cache_manager: Any, moves: List[Any], current_ply: int) -> np.ndarray:
    """Apply special effects to moves - simplified version."""
    if not moves:
        return get_empty_move_batch(0)
    
    n_moves = len(moves)
    special_applied = np.zeros(n_moves, dtype=BOOL_DTYPE)
    
    # Simplified special effect application
    for i, move in enumerate(moves):
        # Check for special effects at destination
        target_piece = cache_manager.get_piece(move.to_coord)
        if target_piece is not None:
            ptype = target_piece.ptype
            if ptype in [PieceType.WHITEHOLE, PieceType.BLACKHOLE, PieceType.GEOMANCER]:
                special_applied[i] = True
    
    return special_applied

# Module exports
__all__ = [
    # Core vectorized extraction functions
    'extract_move_coordinates_vectorized',
    'create_move_array_from_objects_vectorized',
    'extract_from_coords_vectorized',
    'extract_to_coords_vectorized',
    
    # Move filtering
    'create_filtered_moves_batch',
    'filter_none_moves',
    
    # Vectorized analysis
    'calculate_move_distances_vectorized',
    'detect_captures_vectorized',
    
    # Utility functions
    'detonate',
    'apply_special_effects',
]
