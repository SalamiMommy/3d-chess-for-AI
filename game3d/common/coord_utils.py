"""Coordinate utilities - fully optimized numpy/numba native implementation.

This module provides a single source of truth for all coordinate operations,
ensuring all functions are vectorized and batched with no legacy code.
"""

import numpy as np
from numba import njit, prange
from typing import Union, Optional, Any

from game3d.common.shared_types import (
    SIZE, VOLUME, SIZE_SQUARED, SIZE_MINUS_1, MAX_COORD_VALUE,
    COORD_DTYPE, BATCH_COORD_DTYPE, INDEX_DTYPE, BOOL_DTYPE,
    VECTORIZATION_THRESHOLD
)

# =============================================================================
# CORE COORDINATE UTILITIES - SINGLE SOURCE OF TRUTH
# =============================================================================

class CoordinateUtils:
    """Centralized coordinate utilities - fully vectorized and batched."""
    
    @staticmethod
    def coord_to_idx(coords: np.ndarray, cache_manager: Optional[Any] = None) -> np.ndarray:
        """Convert coordinates to flat indices - fully vectorized with cache integration.
        
        Args:
            coords: Coordinates to convert
            cache_manager: Cache manager for memory pool optimization
            
        Returns:
            Flat indices array
        """
        coords = np.asarray(coords, dtype=BATCH_COORD_DTYPE)
        coords = np.atleast_2d(coords)
        
        # Use memory pool for allocation if available (reduces memory overhead)
        if cache_manager is not None and hasattr(cache_manager, '_memory_pool'):
            memory_pool = cache_manager._memory_pool
            if hasattr(memory_pool, 'allocate_array'):
                # Use memory pool for efficient array allocation
                try:
                    n_coords = coords.shape[0]
                    result = memory_pool.allocate_array((n_coords,), INDEX_DTYPE)
                    # Fill with calculation
                    if coords.shape[0] > VECTORIZATION_THRESHOLD:
                        result[:] = CoordinateUtils._coord_to_idx_batch_numba(coords)
                    else:
                        result[:] = (coords[:, 0] + SIZE * coords[:, 1] + SIZE_SQUARED * coords[:, 2])
                    return result.astype(INDEX_DTYPE)
                except (AttributeError, TypeError):
                    # Fall back to standard allocation if memory pool optimization fails
                    pass
        
        # Standard calculation with efficient numpy allocation
        if coords.shape[0] > VECTORIZATION_THRESHOLD:
            return CoordinateUtils._coord_to_idx_batch_numba(coords)
        
        return (coords[:, 0] + SIZE * coords[:, 1] + SIZE_SQUARED * coords[:, 2]).astype(INDEX_DTYPE)

    @staticmethod
    def idx_to_coord(indices: np.ndarray, cache_manager: Optional[Any] = None) -> np.ndarray:
        """Convert flat indices to coordinates - fully vectorized with cache integration.
        
        Args:
            indices: Flat indices to convert
            cache_manager: Cache manager for memory pool optimization
            
        Returns:
            Coordinate array
        """
        indices = np.asarray(indices, dtype=INDEX_DTYPE)
        indices = np.atleast_1d(indices)
        
        # Use memory pool for allocation if available (reduces memory overhead)
        if cache_manager is not None and hasattr(cache_manager, '_memory_pool'):
            memory_pool = cache_manager._memory_pool
            if hasattr(memory_pool, 'allocate_array'):
                # Use memory pool for efficient array allocation
                try:
                    n_indices = indices.shape[0]
                    result = memory_pool.allocate_array((n_indices, 3), BATCH_COORD_DTYPE)
                    # Fill with calculation
                    if indices.shape[0] > VECTORIZATION_THRESHOLD:
                        result[:] = CoordinateUtils._idx_to_coord_batch_numba(indices)
                    else:
                        z = indices // SIZE_SQUARED
                        remainder = indices % SIZE_SQUARED
                        y = remainder // SIZE
                        x = remainder % SIZE
                        
                        result[:, 0] = x
                        result[:, 1] = y
                        result[:, 2] = z
                    return result
                except (AttributeError, TypeError):
                    # Fall back to standard allocation if memory pool optimization fails
                    pass
        
        # Standard calculation with efficient numpy allocation
        if indices.shape[0] > VECTORIZATION_THRESHOLD:
            return CoordinateUtils._idx_to_coord_batch_numba(indices)
        
        z = indices // SIZE_SQUARED
        remainder = indices % SIZE_SQUARED
        y = remainder // SIZE
        x = remainder % SIZE
        
        coords = np.empty((indices.shape[0], 3), dtype=BATCH_COORD_DTYPE, order='C')
        coords[:, 0] = x
        coords[:, 1] = y
        coords[:, 2] = z
        return coords

    @staticmethod
    @njit(cache=True, fastmath=True, inline='always')
    def coord_to_idx_scalar(x: int, y: int, z: int) -> int:
        """Scalar coordinate to index conversion."""
        return x + SIZE * y + SIZE_SQUARED * z

    @staticmethod
    @njit(cache=True, fastmath=True, parallel=True)
    def _coord_to_idx_batch_numba(coords: np.ndarray) -> np.ndarray:
        """Numba coordinate to index conversion."""
        n = coords.shape[0]
        result = np.empty(n, dtype=INDEX_DTYPE)
        
        for i in prange(n):
            result[i] = coords[i, 0] + SIZE * coords[i, 1] + SIZE_SQUARED * coords[i, 2]
        
        return result

    @staticmethod
    @njit(cache=True, fastmath=True, parallel=True)
    def _idx_to_coord_batch_numba(indices: np.ndarray) -> np.ndarray:
        """Numba index to coordinate conversion."""
        n = indices.shape[0]
        result = np.empty((n, 3), dtype=BATCH_COORD_DTYPE)
        
        for i in prange(n):
            idx = indices[i]
            result[i, 0] = idx % SIZE
            result[i, 1] = (idx // SIZE) % SIZE
            result[i, 2] = idx // SIZE_SQUARED
        
        return result

    @staticmethod
    def in_bounds(coords: np.ndarray) -> np.ndarray:  # Change return type to np.ndarray only
        """Bounds checking for coordinates - fully vectorized."""
        coords = np.asarray(coords, dtype=BATCH_COORD_DTYPE)
        coords = np.atleast_2d(coords)

        if coords.shape[0] > VECTORIZATION_THRESHOLD:
            return CoordinateUtils._bounds_check_batch_numba(coords)

        result = ((coords >= 0).all(axis=1)) & ((coords <= SIZE_MINUS_1).all(axis=1))

        return result

    @staticmethod
    @njit(cache=True, fastmath=True, parallel=True)
    def _bounds_check_batch_numba(coords: np.ndarray) -> np.ndarray:
        """Numba bounds checking."""
        n = coords.shape[0]
        result = np.empty(n, dtype=BOOL_DTYPE)
        
        for i in prange(n):
            x, y, z = coords[i, 0], coords[i, 1], coords[i, 2]
            result[i] = (0 <= x <= SIZE_MINUS_1 and 0 <= y <= SIZE_MINUS_1 and 0 <= z <= SIZE_MINUS_1)
        
        return result

# =============================================================================
# PUBLIC API FUNCTIONS - SINGLE IMPLEMENTATIONS
# =============================================================================
def in_bounds_vectorized(coords: np.ndarray) -> np.ndarray:
    """Vectorized bounds checking for coordinate arrays."""
    return CoordinateUtils.in_bounds(coords)

def get_neighbors_vectorized(coords: np.ndarray) -> np.ndarray:
    """Get valid neighboring coordinates - fully vectorized for batch input."""
    coords = np.asarray(coords, dtype=BATCH_COORD_DTYPE)
    coords = np.atleast_2d(coords)
    
    # Lazy import to avoid circular dependency
    from game3d.pieces.pieces.rook import ROOK_MOVEMENT_VECTORS
    
    # Use precomputed orthogonal direction vectors for 6-connected neighbors
    directions = ROOK_MOVEMENT_VECTORS
    
    # Vectorized neighbor generation
    n_coords = coords.shape[0]
    n_dirs = directions.shape[0]
    
    # Create all possible neighbor coordinates
    neighbor_coords = coords[:, np.newaxis, :] + directions[np.newaxis, :, :]
    neighbor_coords = neighbor_coords.reshape(-1, 3)
    
    # Use vectorized bounds checking
    valid_mask = in_bounds_vectorized(neighbor_coords)
    
    return neighbor_coords[valid_mask]

# =============================================================================
# ENHANCED VECTORIZED OPERATIONS
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def calculate_distances_euclidean_vectorized(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """Calculate Euclidean distances between coordinate batches."""
    n1, n2 = coords1.shape[0], coords2.shape[0]
    distances = np.empty((n1, n2), dtype=np.float32)
    
    for i in prange(n1):
        for j in range(n2):
            diff = coords1[i] - coords2[j]
            distances[i, j] = np.sqrt(float(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]))
    
    return distances

@njit(cache=True, fastmath=True, parallel=True)
def calculate_distances_manhattan_vectorized(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """Calculate Manhattan distances between coordinate batches."""
    n1, n2 = coords1.shape[0], coords2.shape[0]
    distances = np.empty((n1, n2), dtype=INDEX_DTYPE)
    
    for i in prange(n1):
        for j in range(n2):
            diff = coords1[i] - coords2[j]
            distances[i, j] = abs(diff[0]) + abs(diff[1]) + abs(diff[2])
    
    return distances

# =============================================================================
# PUBLIC API FOR DISTANCE AND GEOMETRIC OPERATIONS
# =============================================================================

def calculate_distances_euclidean(coords1: np.ndarray, coords2: np.ndarray, cache_manager: Optional[Any] = None) -> np.ndarray:
    """Calculate Euclidean distances between coordinate batches - fully vectorized with cache integration."""
    coords1 = np.asarray(coords1, dtype=BATCH_COORD_DTYPE)
    coords2 = np.asarray(coords2, dtype=BATCH_COORD_DTYPE)
    coords1 = np.atleast_2d(coords1)
    coords2 = np.atleast_2d(coords2)
    
    # Use memory pool for result allocation if available
    if cache_manager is not None and hasattr(cache_manager, '_memory_pool'):
        memory_pool = cache_manager._memory_pool
        if hasattr(memory_pool, 'allocate_array'):
            try:
                n1, n2 = coords1.shape[0], coords2.shape[0]
                result = memory_pool.allocate_array((n1, n2), np.float32)
                # Calculate using existing vectorized function
                result[:] = calculate_distances_euclidean_vectorized(coords1, coords2)
                return result
            except (AttributeError, TypeError):
                # Fall back to standard allocation
                pass
    
    return calculate_distances_euclidean_vectorized(coords1, coords2)

def calculate_distances_manhattan(coords1: np.ndarray, coords2: np.ndarray, cache_manager: Optional[Any] = None) -> np.ndarray:
    """Calculate Manhattan distances between coordinate batches - fully vectorized with cache integration."""
    coords1 = np.asarray(coords1, dtype=BATCH_COORD_DTYPE)
    coords2 = np.asarray(coords2, dtype=BATCH_COORD_DTYPE)
    coords1 = np.atleast_2d(coords1)
    coords2 = np.atleast_2d(coords2)
    
    # Use memory pool for result allocation if available
    if cache_manager is not None and hasattr(cache_manager, '_memory_pool'):
        memory_pool = cache_manager._memory_pool
        if hasattr(memory_pool, 'allocate_array'):
            try:
                n1, n2 = coords1.shape[0], coords2.shape[0]
                result = memory_pool.allocate_array((n1, n2), INDEX_DTYPE)
                # Calculate using existing vectorized function
                result[:] = calculate_distances_manhattan_vectorized(coords1, coords2)
                return result
            except (AttributeError, TypeError):
                # Fall back to standard allocation
                pass
    
    return calculate_distances_manhattan_vectorized(coords1, coords2)

def ensure_coords(coords: np.ndarray, dtype: np.dtype = COORD_DTYPE) -> np.ndarray:
    """Ensure coordinates are properly formatted numpy arrays.
    
    Delegates to the consolidated validation.ensure_coords for consistency.
    """
    from game3d.common.validation import ensure_coords as validation_ensure_coords
    result = validation_ensure_coords(coords)
    
    # Apply dtype conversion if different from validation result
    if dtype != COORD_DTYPE and result.dtype != dtype:
        result = result.astype(dtype)
    
    return result

# Module-level convenience functions
def coord_to_idx(coords: np.ndarray, cache_manager: Optional[Any] = None) -> np.ndarray:
    """Convert coordinates to flat indices - module-level convenience function with cache integration."""
    return CoordinateUtils.coord_to_idx(coords, cache_manager)

def idx_to_coord(indices: np.ndarray, cache_manager: Optional[Any] = None) -> np.ndarray:
    """Convert flat indices to coordinates - module-level convenience function with cache integration."""
    return CoordinateUtils.idx_to_coord(indices, cache_manager)

# Aliases for compatibility
coords_to_flat_batch = coord_to_idx
flat_to_coords_vectorized = idx_to_coord

def create_coord(x: Union[int, np.ndarray], y: Optional[Union[int, np.ndarray]] = None, 
                z: Optional[Union[int, np.ndarray]] = None) -> np.ndarray:
    """Create a coordinate array from x, y, z values."""
    if y is None and z is None:
        # Assume input is already a coordinate array
        coords = np.asarray(x, dtype=COORD_DTYPE)
        if coords.shape == (3,):
            return coords
        else:
            raise ValueError(f"Expected 3D coordinate, got shape {coords.shape}")
    else:
        if y is None or z is None:
            raise ValueError("Both y and z must be provided if any are provided")
        return np.array([x, y, z], dtype=COORD_DTYPE)

# =============================================================================
# PAIRWISE DISTANCE OPERATIONS (for move_utils compatibility)
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def calculate_pairwise_manhattan_vectorized(
    from_coords: np.ndarray,
    to_coords: np.ndarray
) -> np.ndarray:
    """Calculate Manhattan distances between corresponding coordinate pairs."""
    n = from_coords.shape[0]
    distances = np.empty(n, dtype=INDEX_DTYPE)

    for i in prange(n):
        diff = from_coords[i] - to_coords[i]
        distances[i] = abs(diff[0]) + abs(diff[1]) + abs(diff[2])

    return distances

def calculate_pairwise_manhattan(
    from_coords: np.ndarray,
    to_coords: np.ndarray,
    cache_manager: Optional[Any] = None
) -> np.ndarray:
    """Calculate Manhattan distances between corresponding coordinate pairs."""
    from_coords = np.asarray(from_coords, dtype=BATCH_COORD_DTYPE)
    to_coords = np.asarray(to_coords, dtype=BATCH_COORD_DTYPE)
    from_coords = np.atleast_2d(from_coords)
    to_coords = np.atleast_2d(to_coords)

    if from_coords.shape[0] != to_coords.shape[0]:
        raise ValueError(f"Coordinate count mismatch: {from_coords.shape[0]} vs {to_coords.shape[0]}")

    # Memory pool optimization
    if cache_manager is not None and hasattr(cache_manager, '_memory_pool'):
        memory_pool = cache_manager._memory_pool
        if hasattr(memory_pool, 'allocate_array'):
            try:
                n = from_coords.shape[0]
                result = memory_pool.allocate_array((n,), INDEX_DTYPE)
                result[:] = calculate_pairwise_manhattan_vectorized(from_coords, to_coords)
                return result
            except (AttributeError, TypeError):
                pass

    return calculate_pairwise_manhattan_vectorized(from_coords, to_coords)

# Module exports - fully vectorized version with cache integration
__all__ = [
    # Core coordinate operations - fully vectorized
    'in_bounds_vectorized', 'get_neighbors_vectorized', 'ensure_coords',
    'coord_to_idx', 'idx_to_coord', 'create_coord',
    'coords_to_flat_batch', 'flat_to_coords_vectorized',
    
    # Distance calculations - fully vectorized with cache integration
    'calculate_distances_euclidean', 'calculate_distances_manhattan',
    
    # Core class
    'CoordinateUtils'
]

# =============================================================================
# PATH CALCULATION UTILITIES
# =============================================================================

@njit(cache=True, fastmath=True)
def _get_path_between_numba(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """
    Calculate squares strictly between start and end coordinates (exclusive).
    Works for orthogonal and diagonal paths.
    Returns empty array if not a straight line or adjacent.
    """
    diff = end - start
    dist = np.abs(diff)
    max_dist = np.max(dist)
    
    # Check if it's a valid straight line (all non-zero diffs must be equal magnitude)
    # e.g. (2, 0, 0) -> valid
    # e.g. (2, 2, 0) -> valid
    # e.g. (2, 1, 0) -> invalid (knight-like or irregular)
    
    non_zero = dist > 0
    if np.sum(non_zero) == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE) # Same square
        
    if not np.all(dist[non_zero] == max_dist):
        return np.empty((0, 3), dtype=COORD_DTYPE) # Not a straight line
        
    if max_dist <= 1:
        return np.empty((0, 3), dtype=COORD_DTYPE) # Adjacent
        
    # Calculate step direction (-1, 0, 1)
    step = np.sign(diff)
    
    # Generate path
    n_steps = max_dist - 1
    path = np.empty((n_steps, 3), dtype=COORD_DTYPE)
    
    for i in range(n_steps):
        path[i] = start + step * (i + 1)
        
    return path

def get_path_between(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """
    Get coordinates strictly between start and end.
    Returns empty array if move is not a straight line or is adjacent.
    """
    start = np.asarray(start, dtype=COORD_DTYPE)
    end = np.asarray(end, dtype=COORD_DTYPE)
    return _get_path_between_numba(start, end)

def coords_to_keys(coords: np.ndarray) -> np.ndarray:
    """Convert coordinates to cache keys using bit packing: x | (y << 9) | (z << 18)."""
    return coords[:, 0] | (coords[:, 1] << 9) | (coords[:, 2] << 18)

def get_adjacent_squares(coord: np.ndarray) -> np.ndarray:
    """Get the 6 direct (orthogonal) neighbors of a single coordinate."""
    # Ensure coord is 2D for get_neighbors_vectorized, which is optimized for batch input
    coord_batch = np.atleast_2d(coord)
    # get_neighbors_vectorized uses ROOK_MOVEMENT_VECTORS for 6-connected neighbors (orthogonal)
    return get_neighbors_vectorized(coord_batch)

__all__.append('get_path_between')
