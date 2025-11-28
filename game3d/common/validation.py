# consolidated_validation.py
"""Consolidated validation functions for coordinate and data validation."""

from __future__ import annotations

import numpy as np
from typing import Any, Union, List, Optional, Dict, TYPE_CHECKING, Callable
from numpy.typing import NDArray
import warnings
import functools

# Direct imports for constants, types, and utilities
# Note: Not importing from coord_utils to avoid circular import
from game3d.common.shared_types import (
    SIZE, VOLUME,
    COORD_DTYPE, INDEX_DTYPE, BOOL_DTYPE, COLOR_DTYPE, PIECE_TYPE_DTYPE, FLOAT_DTYPE,
    get_empty_coord, get_empty_bool_array, get_empty_index_array,
    format_coord_error, format_batch_error, format_bounds_error
)
from game3d.common.coord_utils import in_bounds_vectorized

import logging
logger = logging.getLogger(__name__)

# ==============================================================================
# CACHING AND PERFORMANCE OPTIMIZATION
# ==============================================================================

def _validate_coord_shape(coords: np.ndarray, name: str) -> None:
    """Consolidated shape validation for coordinate arrays."""
    if coords.ndim == 1:
        if coords.shape[0] != 3:
            raise ValidationError(f"{name} must have exactly 3 elements, got {coords.shape[0]}")
    elif coords.ndim == 2:
        if coords.shape[1] != 3:
            raise ValidationError(f"{name} must have exactly 3 columns, got {coords.shape[1]}")

def _validate_array_size(coords: np.ndarray, name: str, min_size: int = 0, max_size: Optional[int] = None) -> None:
    """Consolidated size validation for coordinate arrays."""
    size = coords.size if coords.ndim > 0 else 1
    
    if size < min_size:
        raise ValidationError(f"{name} must have at least {min_size} elements, got {size}")
    
    if max_size is not None and size > max_size:
        raise ValidationError(f"{name} cannot have more than {max_size} elements, got {size}")

def cache_array_conversion(func: Callable) -> Callable:
    """Decorator to cache array conversions and reduce redundant operations."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Simple caching for frequent calls (no LRU cache to avoid memory overhead)
        cache_key = (id(args[0]) if args else id(kwargs.get('coords', None)))
        if not hasattr(wrapper, '_cache'):
            wrapper._cache = {}
        
        if cache_key in wrapper._cache:
            return wrapper._cache[cache_key]
        
        result = func(*args, **kwargs)
        # Limit cache size to prevent memory bloat
        if len(wrapper._cache) > 1000:
            wrapper._cache.clear()
        wrapper._cache[cache_key] = result
        return result
    return wrapper

# ==============================================================================
# PRIMARY EXCEPTION CLASS
# ==============================================================================

class ValidationError(Exception):
    """Primary exception for all validation failures in the consolidated module."""
    pass

# ==============================================================================
# CORE VALIDATION FUNCTION 1: validate_coord
# ==============================================================================
def validate_coord(coord: Any) -> NDArray[COORD_DTYPE]:
    """Validate and normalize a single 3D coordinate - NO CONVERSIONS."""
    if coord is None:
        raise ValidationError("Coordinate cannot be None")

    # DIRECT NUMPY PATH ONLY - reject Python lists
    if isinstance(coord, (int, np.integer)):
        # Single scalar - assume x-coordinate, default y=z=0
        coord_arr = np.array([coord, 0, 0], dtype=COORD_DTYPE)
    elif isinstance(coord, np.ndarray):
        if coord.shape == (3,):
            coord_arr = coord.astype(COORD_DTYPE, copy=False)
        else:
            raise ValidationError(f"Invalid coordinate shape: {coord.shape}")
    else:
        raise ValidationError(f"Invalid coordinate format: {type(coord)}")

    return coord_arr
# ==============================================================================
# CORE VALIDATION FUNCTION 2: validate_coords_batch
# ==============================================================================

def validate_coords_batch(coords: Any) -> NDArray[COORD_DTYPE]:
    """Validate and normalize a batch of 3D coordinates.

    Consolidates batch coordinate validation from multiple files, handling both
    single coordinates (reshaped to batch) and actual batches.

    Args:
        coords: Input coordinates (single coord or array of coordinates)

    Returns:
        Validated coordinate batch as numpy array with dtype=int32 and shape (N, 3)

    Raises:
        ValidationError: If coordinate batch format, shape, or values are invalid
        TypeError: If coordinates contain non-numeric data

    Examples:
        >>> validate_coords_batch([[1, 2, 3], [4, 5, 6]])
        array([[1, 2, 3],
               [4, 5, 6]], dtype=int32)
        >>> validate_coords_batch([7, 8, 9])  # Single coord becomes batch
        array([[7, 8, 9]], dtype=int32)
    """
    if coords is None:
        return np.empty((0, 3), dtype=COORD_DTYPE)

    # Convert to numpy array
    coords_arr = np.asarray(coords, dtype=COORD_DTYPE)

    # Handle empty array - use centralized utility
    if coords_arr.size == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)

    # Handle single coordinate case (1D with 3 elements)
    if coords_arr.ndim == 1:
        if coords_arr.shape[0] != 3:
            raise ValidationError(
                f"Single coordinate must have exactly 3 elements, got {coords_arr.shape[0]}"
            )
        # Reshape to (1, 3) batch format and validate bounds
        if not in_bounds_vectorized(coords_arr.reshape(1, 3))[0]:
            raise ValidationError(format_bounds_error(coords_arr.reshape(1, 3)))
        return coords_arr.reshape(1, 3).astype(COORD_DTYPE)

    # Validate batch shape
    if coords_arr.ndim != 2:
        raise ValidationError(
            f"Coordinate batch must be 1D or 2D array, got {coords_arr.ndim}D"
        )

    if coords_arr.shape[1] != 3:
        raise ValidationError(
            f"Coordinate batch must have exactly 3 columns, got {coords_arr.shape[1]}"
        )

    # Use vectorized bounds checking
    valid_mask = in_bounds_vectorized(coords_arr)
    if not np.all(valid_mask):
        invalid_coords = coords_arr[~valid_mask]
        raise ValidationError(
            f"Coordinate batch contains {np.sum(~valid_mask)} out-of-bounds coordinates in range [0, {SIZE-1}]"
        )

    return coords_arr.astype(COORD_DTYPE)

# ==============================================================================
# CORE VALIDATION FUNCTION 3: validate_coords_bounds
# ==============================================================================

def validate_coords_bounds(coords: Union[NDArray, List, tuple]) -> NDArray[BOOL_DTYPE]:
    """Validate coordinates against board bounds.

    Consolidates bounds checking logic from coord_utils.py and other files,
    providing consistent boundary validation.

    Args:
        coords: Coordinates to validate (single or batch)

    Returns:
        Boolean array indicating which coordinates are within bounds

    Examples:
        >>> validate_coords_bounds([[1, 2, 3], [8, 8, 8]])
        array([ True, False])
    """
    # Convert to numpy array without strict validation
    coords_arr = np.asarray(coords, dtype=COORD_DTYPE)

    # Handle empty array - use centralized utility
    if coords_arr.size == 0:
        return get_empty_bool_array(0)

    # Handle single coordinate case
    if coords_arr.ndim == 1:
        if coords_arr.shape[0] != 3:
            # Invalid format, return all False
            return get_empty_bool_array(1)
        coords_arr = coords_arr.reshape(1, 3)

    # Validate batch shape
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        # Invalid format, return appropriate False array
        if coords_arr.ndim == 2:
            return get_empty_bool_array(coords_arr.shape[0])
    # Use vectorized bounds checking from shared_types
    return in_bounds_vectorized(coords_arr)

# ==============================================================================
# CORE VALIDATION FUNCTION 4: validate_coordinate_array
# ==============================================================================

def validate_coordinate_array(
    coords: Any,
    name: str = "coords",
    allow_single: bool = True,
    allow_batch: bool = True,
    min_size: int = 1,
    max_size: Optional[int] = None,
    strict_bounds: bool = True
) -> NDArray[COORD_DTYPE]:
    """Flexible coordinate array validation with comprehensive options.

    This is the primary coordinate validation function that handles most use cases,
    consolidating logic from coord_utils.py and other files.

    Args:
        coords: Input coordinate array
        name: Variable name for error messages
        allow_single: Whether to allow single coordinates (3,) or (1, 3)
        allow_batch: Whether to allow batch coordinates (N, 3)
        min_size: Minimum number of coordinates required
        max_size: Maximum number of coordinates allowed (None for no limit)
        strict_bounds: Whether to enforce board bounds checking

    Returns:
        Validated coordinate array with consistent dtype=int32

    Raises:
        ValidationError: If validation fails

    Examples:
        >>> validate_coordinate_array([1, 2, 3])
        array([[1, 2, 3]], dtype=int32)
        >>> validate_coordinate_array([[1, 2, 3], [4, 5, 6]], min_size=2)
        array([[1, 2, 3],
               [4, 5, 6]], dtype=int32)
    """
    if coords is None:
        raise ValidationError(f"{name} cannot be None")

    if not isinstance(coords, np.ndarray):
        coords = np.asarray(coords)

    # Handle empty arrays - use centralized utility
    if coords.size == 0:
        if min_size == 0:
            return np.empty((0, 3), dtype=COORD_DTYPE)
    # Single coordinate handling
    if coords.ndim == 1:
        if coords.shape[0] != 3:
            raise ValidationError(
                f"{name} single coordinate must have exactly 3 elements, got {coords.shape[0]}"
            )
        if not allow_single:
            raise ValidationError(f"{name} single coordinates not allowed")
        return coords.reshape(1, 3).astype(COORD_DTYPE)

    # Batch coordinate handling
    if coords.ndim != 2:
        raise ValidationError(f"{name} must be 1D or 2D array, got {coords.ndim}D")

    if coords.shape[1] != 3:
        raise ValidationError(f"{name} must have exactly 3 columns, got {coords.shape[1]}")

    n_coords = coords.shape[0]
    if n_coords < min_size:
        raise ValidationError(
            f"{name} must have at least {min_size} coordinates, got {n_coords}"
        )

    if max_size is not None and n_coords > max_size:
        raise ValidationError(
            f"{name} cannot have more than {max_size} coordinates, got {n_coords}"
        )

    # Bounds checking if requested
    if strict_bounds:
        bounds_valid = validate_coords_bounds(coords)
        if not np.all(bounds_valid):
            invalid_coords = coords[~bounds_valid]
            raise ValidationError(
                f"{name} contains out-of-bounds coordinates"
            )

    return coords.astype(COORD_DTYPE)

# ==============================================================================
# CORE VALIDATION FUNCTION 5: validate_index_array
# ==============================================================================

def validate_index_array(
    indices: Any,
    name: str = "indices",
    min_value: int = 0,
    max_value: Optional[int] = None,
    min_size: int = 0,
    max_size: Optional[int] = None
) -> NDArray[INDEX_DTYPE]:
    """Validate array of indices with value and size constraints.

    Consolidates index validation from coord_utils.py and other files.

    Args:
        indices: Input index array
        name: Variable name for error messages
        min_value: Minimum valid index value
        max_value: Maximum valid index value (None for VOLUME-1)
        min_size: Minimum array size
        max_size: Maximum array size (None for no limit)

    Returns:
        Validated index array with dtype=int32

    Raises:
        ValidationError: If validation fails

    Examples:
        >>> validate_index_array([0, 1, 2])
        array([0, 1, 2], dtype=int32)
        >>> validate_index_array([728], max_value=727)
        array([728], dtype=int32)
    """
    if indices is None:
        raise ValidationError(f"{name} cannot be None")

    if not isinstance(indices, np.ndarray):
        indices = np.asarray(indices)

    # Handle empty arrays - use centralized utility
    if indices.size == 0:
        if min_size == 0:
            return get_empty_index_array(0)
    # Handle scalar input
    if indices.ndim == 0:
        indices = np.array([indices])

    # Size validation
    if indices.size < min_size:
        raise ValidationError(
            f"{name} must have at least {min_size} elements, got {indices.size}"
        )

    if max_size is not None and indices.size > max_size:
        raise ValidationError(
            f"{name} cannot have more than {max_size} elements, got {indices.size}"
        )

    # Value range validation
    if max_value is None:
        max_value = VOLUME - 1  # Default to board volume - 1

    if np.any(indices < min_value) or np.any(indices > max_value):
        invalid_mask = (indices < min_value) | (indices > max_value)
        invalid_values = indices[invalid_mask]
        raise ValidationError(
            f"{name} values out of range [{min_value}, {max_value}]: "
            f"{name} contains invalid index values"
        )

    return indices.astype(INDEX_DTYPE)

# ==============================================================================
# CORE VALIDATION FUNCTION 6: validate_not_none
# ==============================================================================

def validate_not_none(value: Any, name: str = "value") -> Any:
    """Validate that a value is not None.

    Consolidates None validation from cache_utils.py.

    Args:
        value: Value to validate
        name: Name of the parameter for error messages

    Returns:
        The validated value

    Raises:
        ValidationError: If value is None

    Examples:
        >>> validate_not_none([1, 2, 3])
        [1, 2, 3]
        >>> validate_not_none(None, "coordinate")
        Traceback (most recent call last):
            ...
        ValidationError: coordinate cannot be None
    """
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    return value

# ==============================================================================
# CORE VALIDATION FUNCTION 7: validate_move_basic - NOW COMPREHENSIVE MOVE VALIDATOR
# ==============================================================================
def validate_move_basic(
    game_state: Any,
    move: Union[Any, np.ndarray],
    expected_color: Optional[int] = None
) -> Union[bool, NDArray[BOOL_DTYPE]]:
    """COMPREHENSIVE MOVE VALIDATION - SINGLE NUMPY PATH."""
    if not hasattr(game_state, 'cache_manager'):
        raise ValidationError("game_state must have cache_manager attribute")

    cache = game_state.cache_manager
    if cache is None:
        return False

    expected_color = expected_color or getattr(game_state, 'color', None)
    if expected_color is None:
        raise ValidationError("expected_color must be provided")

    # BATCH VALIDATION - keep as numpy array throughout
    if isinstance(move, np.ndarray):
        # Numpy array format: (N, 6)
        if move.ndim != 2 or move.shape[1] != 6:
            return get_empty_bool_array(move.shape[0] if move.size > 0 else 0)

        from_coords = move[:, :3]
        to_coords = move[:, 3:6]

        # Vectorized bounds checking
        from_bounds_valid = in_bounds_vectorized(from_coords)
        to_bounds_valid = in_bounds_vectorized(to_coords)
        bounds_valid = from_bounds_valid & to_bounds_valid

        # For ownership check, only process coordinates that pass bounds validation
        valid_from_coords = from_coords[bounds_valid]

        if len(valid_from_coords) > 0:
            # Get colors for valid coordinates only
            valid_colors, _ = cache.occupancy_cache.batch_get_attributes(valid_from_coords)

            # Create ownership array for all moves, default False
            piece_ownership_valid = get_empty_bool_array(len(move))

            # Set ownership for valid coordinates
            valid_indices = np.where(bounds_valid)[0]
            piece_ownership_valid[valid_indices] = (valid_colors == expected_color) & (valid_colors != 0)

            # Combine validations
            return bounds_valid & piece_ownership_valid
        else:
            # No valid coordinates, return bounds check only
            return bounds_valid

    # SINGLE MOVE VALIDATION
    if hasattr(move, 'from_coord') and hasattr(move, 'to_coord'):
        from_coord = move.from_coord
        to_coord = move.to_coord
    else:
        return False

    # Validate bounds
    if not in_bounds_vectorized(np.array([from_coord]))[0] or not in_bounds_vectorized(np.array([to_coord]))[0]:
        return False

    # Check piece ownership
    from_colors, _ = cache.occupancy_cache.batch_get_attributes(np.array([from_coord]))
    return (from_colors[0] == expected_color) and (from_colors[0] != 0)
# ==============================================================================
# CORE VALIDATION FUNCTION 8: validate_move - PUBLIC API
# ==============================================================================
def validate_move(game_state: Any, move: Union[Any, np.ndarray]) -> bool:
    """PUBLIC API: Validate a single move.

    This is the ONLY public interface for single move validation.
    Delegates to validate_move_basic for actual validation.
    """
    result = validate_move_basic(game_state, move)

    # Handle both Python bool and numpy bool scalars
    if isinstance(result, (bool, np.bool_)):
        return bool(result)

    # Handle array results (batch case)
    return result[0]
# ==============================================================================
# CORE VALIDATION FUNCTION 9: validate_moves - PUBLIC API
# ==============================================================================

def validate_moves(game_state: Any, moves: Union[List[Any], np.ndarray]) -> NDArray[BOOL_DTYPE]:
    """PUBLIC API: Validate multiple moves.
    
    This is the ONLY public interface for batch move validation.
    Delegates to validate_move_basic for actual validation.
    """
    return validate_move_basic(game_state, moves)

# ==============================================================================
# CORE VALIDATION FUNCTION 10: filter_valid_moves - UTILITY
# ==============================================================================

def filter_valid_moves(game_state: Any, moves: Union[List[Any], np.ndarray]) -> Union[List[Any], np.ndarray]:
    """Filter moves to only valid ones.
    
    Returns moves that pass validation using numpy-optimized filtering.
    """
    if isinstance(moves, np.ndarray):
        if moves.ndim == 2 and moves.shape[1] == 6:
            # Move array format
            validity = validate_moves(game_state, moves)
            return moves[validity]
    elif isinstance(moves, list):
        # List of Move objects
        validity = validate_moves(game_state, moves)
        if isinstance(validity, np.ndarray):
            # Use NumPy boolean indexing instead of list comprehension
            valid_indices = np.where(validity)[0]
            return [moves[i] for i in valid_indices]
        elif validity:
            return moves
    
    return moves
# ==============================================================================
# CORE VALIDATION FUNCTION 8: validate_array
# ==============================================================================

def validate_array(
    array: Any,
    name: str = "array",
    dtype: Optional[type] = None,
    ndim: Optional[int] = None,
    shape: Optional[tuple] = None,
    min_size: int = 0,
    max_size: Optional[int] = None,
    allow_none: bool = False
) -> NDArray:
    """General array validation with comprehensive options.

    This function provides a unified interface for validating various types of arrays,
    consolidating validation patterns from multiple files.

    Args:
        array: Input array to validate
        name: Variable name for error messages
        dtype: Expected numpy dtype (None for any numeric type)
        ndim: Expected number of dimensions (None for any)
        shape: Expected shape (None for any)
        min_size: Minimum number of elements
        max_size: Maximum number of elements (None for no limit)
        allow_none: Whether to allow None values

    Returns:
        Validated numpy array

    Raises:
        ValidationError: If validation fails

    Examples:
        >>> validate_array([1, 2, 3], dtype=np.int32)
        array([1, 2, 3], dtype=int32)
        >>> validate_array(np.array([4, 5, 6]), shape=(3,))
        array([4, 5, 6])
    """
    if array is None:
        if allow_none:
            return None
    if not isinstance(array, np.ndarray):
            array = np.asarray(array)
    # Dimension validation
    if ndim is not None and array.ndim != ndim:
        raise ValidationError(
            f"{name} must have {ndim} dimensions, got {array.ndim}"
        )

    # Shape validation
    if shape is not None and array.shape != shape:
        raise ValidationError(
            f"{name} must have shape {shape}, got {array.shape}"
        )

    # Size validation
    size = array.size
    if size < min_size:
        raise ValidationError(
            f"{name} must have at least {min_size} elements, got {size}"
        )

    if max_size is not None and size > max_size:
        raise ValidationError(
            f"{name} cannot have more than {max_size} elements, got {size}"
        )

    # Dtype validation
    if dtype is not None and not np.issubdtype(array.dtype, dtype):
            array = array.astype(dtype)
    return array

def validate_move_bounds_with_error(from_coord: np.ndarray, to_coord: np.ndarray) -> Optional[str]:
    """Validate move coordinates are within bounds and return error message."""
    from_coords = from_coord.reshape(1, 3)
    to_coords = to_coord.reshape(1, 3)

    try:
        from_valid = in_bounds_vectorized(from_coords)[0]
        to_valid = in_bounds_vectorized(to_coords)[0]
    except Exception as e:
        logger.critical(f"Coordinate validation crashed: {e}", exc_info=True)
        return f"Coordinate validation error: {e}"

    if not from_valid and not to_valid:
        return f"Both source {from_coord} and target {to_coord} coordinates are out of bounds"
    elif not from_valid:
        return f"Source coordinate {from_coord} is out of bounds"
    elif not to_valid:
        return f"Target coordinate {to_coord} is out of bounds"

    return None


def validate_move_ownership_with_error(game_state: Any, from_coord: np.ndarray, expected_color: int) -> Optional[str]:
    """Validate piece exists and belongs to correct player, returning error message."""
    try:
        from_coord_batch = from_coord.reshape(1, 3)
        colors, types = game_state.cache_manager.occupancy_cache.batch_get_attributes(from_coord_batch)

        if types[0] == 0:  # COLOR_EMPTY
            return f"No piece at source coordinate {from_coord}"

        if colors[0] != expected_color:
            return f"Piece at {from_coord} belongs to opponent"

    except Exception as e:
        logger.critical(f"Occupancy cache lookup failed: {e}", exc_info=True)
        return "Cache system failure during piece lookup"

    return None


def validate_hive_move_allowed(game_state: Any, from_coord: np.ndarray, piece_type: int) -> Optional[str]:
    """Validate hive hasn't already moved this turn."""
    from game3d.common.shared_types import PieceType

    if piece_type == PieceType.HIVE and game_state.has_hive_moved(from_coord):
        return f"Hive at {from_coord} has already moved this turn"
    return None

# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Primary exception class
    'ValidationError',

    # Core validation functions (10 total)
    'validate_coord',
    'validate_coords_batch',
    'validate_coords_bounds',
    'validate_coordinate_array',
    'validate_index_array',
    'validate_not_none',
    'validate_move_basic',
    'validate_move',  # PUBLIC API
    'validate_moves',  # PUBLIC API
    'filter_valid_moves',  # UTILITY
    'validate_array',

    # Standalone validation functions
    'ensure_coords', 'ensure_single_coord', 'flat_to_coord',

    # Constants and types
    'SIZE', 'VOLUME',
    'COORD_DTYPE', 'INDEX_DTYPE', 'BOOL_DTYPE', 'COLOR_DTYPE', 'PIECE_TYPE_DTYPE', 'FLOAT_DTYPE',
    'WHITE', 'BLACK',
]

# Removed performance aliases to avoid confusion

# ==============================================================================
# STANDALONE VALIDATION FUNCTIONS
# ==============================================================================

def ensure_coords(coords: Union[np.ndarray, list, tuple, int]) -> np.ndarray:
    """
    Ensure coordinates are in proper (N,3) numpy format.
    
    This is the single, canonical implementation that replaces duplicate
    ensure_coords functions across the codebase.
    """
    if isinstance(coords, (list, tuple)):
        if len(coords) == 3:
            return np.array(coords, dtype=COORD_DTYPE).reshape(1, 3)
        elif all(isinstance(c, (list, tuple)) and len(c) == 3 for c in coords):
            return np.array(coords, dtype=COORD_DTYPE)
    
    elif isinstance(coords, np.ndarray):
        if coords.ndim == 0:
            # Single scalar - convert from flat index
            from .coord_utils import idx_to_coord
            return idx_to_coord(int(coords)).reshape(1, 3)
        elif coords.ndim == 1:
            if coords.shape[0] == 3:
                return coords.astype(COORD_DTYPE).reshape(1, 3)
            elif coords.shape[0] == 1 and coords.dtype == COORD_DTYPE:
                return coords.reshape(1, 3)
        elif coords.ndim == 2:
            if coords.shape[1] == 3:
                return coords.astype(COORD_DTYPE)
    
    elif isinstance(coords, (int, np.integer)):
        from .coord_utils import idx_to_coord
        return idx_to_coord(int(coords)).reshape(1, 3)
    
    raise ValueError(f"Invalid coordinate format: {type(coords)}, shape: {getattr(coords, 'shape', 'N/A')}")

def ensure_single_coord(coord: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """Ensure coordinate is in (3,) format."""
    coord = ensure_coords(coord)
    return coord[0] if coord.size > 0 else get_empty_coord()

def flat_to_coord(flat_idx: int) -> np.ndarray:
    """Convert flat index to coordinate."""
    from game3d.common.coord_utils import idx_to_coord
    return idx_to_coord(flat_idx)
