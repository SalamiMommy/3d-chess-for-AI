# game3d/board/symmetry.py
"""Numpy-native symmetry transformations for 3D chess board - pure array operations."""

import numpy as np
from numba import njit, prange
from typing import Tuple, Union, Optional
import logging

from game3d.common.shared_types import (
    SIZE, N_TOTAL_PLANES, N_PIECE_TYPES, N_COLOR_PLANES,
    COORD_DTYPE, INDEX_DTYPE, HASH_DTYPE, Color, COLOR_DTYPE, PIECE_TYPE_DTYPE,
    PIECE_SLICE
)

logger = logging.getLogger(__name__)

# =============================================================================
# TRANSFORMATION MATRICES
# =============================================================================

# Pre-computed 3D rotation matrices for cube symmetries (10 unique transformations)
# Each is a 3x3 integer matrix representing rotations around x, y, z axes
TRANSFORM_MATRICES = np.array([
    # 0: identity
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    # 1: rotate_x_90 (90° around x-axis)
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    # 2: rotate_x_270 (270° around x-axis)
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    # 3: rotate_x_180 (180° around x-axis)
    [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
    # 4: rotate_y_90 (90° around y-axis)
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    # 5: rotate_y_270 (270° around y-axis)
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    # 6: rotate_y_180 (180° around y-axis)
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
    # 7: rotate_z_90 (90° around z-axis)
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    # 8: rotate_z_270 (270° around z-axis)
    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    # 9: rotate_z_180 (180° around z-axis)
    [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
], dtype=INDEX_DTYPE)

N_SYMMETRIES = len(TRANSFORM_MATRICES)

# Center point for transformations (4,4,4 for 9x9x9 board)
TRANSFORM_CENTER = np.array([(SIZE - 1) // 2] * 3, dtype=np.float32)


# =============================================================================
# VECTORIZED COORDINATE TRANSFORMATIONS
# =============================================================================

@njit(cache=True, nogil=True)
def transform_coordinate_numba(coord: np.ndarray, transform_idx: int) -> np.ndarray:
    """
    Apply 3D rotation to a single coordinate using numpy operations.

    Args:
        coord: Shape (3,) array [x, y, z]
        transform_idx: Which transformation to apply (0-9)

    Returns:
        Transformed coordinate as shape (3,) array
    """
    # Convert to float for precise rotation
    coord_float = coord.astype(np.float32)

    # Center the coordinate around origin
    centered = coord_float - TRANSFORM_CENTER

    # Apply rotation matrix (3x3 @ (3,) -> (3,))
    matrix = TRANSFORM_MATRICES[transform_idx]
    rotated_x = matrix[0, 0] * centered[0] + matrix[0, 1] * centered[1] + matrix[0, 2] * centered[2]
    rotated_y = matrix[1, 0] * centered[0] + matrix[1, 1] * centered[1] + matrix[1, 2] * centered[2]
    rotated_z = matrix[2, 0] * centered[0] + matrix[2, 1] * centered[1] + matrix[2, 2] * centered[2]

    # Translate back to board coordinates
    result_x = rotated_x + TRANSFORM_CENTER[0]
    result_y = rotated_y + TRANSFORM_CENTER[1]
    result_z = rotated_z + TRANSFORM_CENTER[2]

    # Round and clip to valid board coordinates
    result = np.empty(3, dtype=COORD_DTYPE)
    result[0] = int(round(result_x))
    result[1] = int(round(result_y))
    result[2] = int(round(result_z))

    # Safety: ensure in bounds (should not be needed but guards against rounding errors)
    if result[0] < 0: result[0] = 0
    if result[0] >= SIZE: result[0] = SIZE - 1
    if result[1] < 0: result[1] = 0
    if result[1] >= SIZE: result[1] = SIZE - 1
    if result[2] < 0: result[2] = 0
    if result[2] >= SIZE: result[2] = SIZE - 1

    return result


@njit(cache=True, nogil=True, parallel=True)
def transform_coordinates_batch(coords: np.ndarray, transform_idx: int) -> np.ndarray:
    """
    Apply transformation to batch of coordinates in parallel.

    Args:
        coords: Shape (N, 3) array of coordinates
        transform_idx: Which transformation to apply

    Returns:
        Shape (N, 3) array of transformed coordinates
    """
    n_coords = coords.shape[0]
    result = np.empty_like(coords)

    for i in prange(n_coords):
        result[i] = transform_coordinate_numba(coords[i], transform_idx)

    return result


# =============================================================================
# BOARD ARRAY TRANSFORMATION
# =============================================================================
@njit(cache=True, nogil=True)
def transform_board_array(board_array: np.ndarray, transform_idx: int) -> np.ndarray:
    transformed = np.zeros_like(board_array)

    for plane in range(N_TOTAL_PLANES):
        for z in range(SIZE):
            for y in range(SIZE):
                for x in range(SIZE):
                    value = board_array[plane, z, y, x]
                    if value != 0:
                        # ✅ FIX 1: Create array in correct order directly
                        coord_xyz = np.array([x, y, z], dtype=COORD_DTYPE)

                        transformed_xyz = transform_coordinate_numba(coord_xyz, transform_idx)

                        # ✅ FIX 2: Extract components directly instead of fancy indexing
                        tz, ty, tx = transformed_xyz[2], transformed_xyz[1], transformed_xyz[0]

                        if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                            transformed[plane, tz, ty, tx] = value

    return transformed
# =============================================================================
# SYMMETRY MANAGER CLASS
# =============================================================================


class SymmetryManager:
    """Numpy-native symmetry manager for 3D chess boards."""

    def __init__(self, cache_manager=None):
        """
        Initialize symmetry manager.

        Args:
            cache_manager: Optional cache manager for integration
        """
        self.cache_manager = cache_manager
        self.transform_matrices = TRANSFORM_MATRICES
        self.n_symmetries = N_SYMMETRIES

        # For this set of rotations, each transform is its own inverse
        self.inverse_transforms = np.arange(N_SYMMETRIES, dtype=INDEX_DTYPE)
        
        # Canonical form cache: hash -> (canonical_array, transform_idx)
        self._canonical_cache = {}
        self._cache_size_limit = 10000
        self._cache_hits = 0
        self._cache_misses = 0

    def get_canonical_form(self, board) -> Tuple[np.ndarray, int]:
        """
        Find canonical form of board by testing all symmetry transformations.
        The canonical form is the one with the smallest Zobrist hash.

        Args:
            board: Board object with .array() method OR 4D numpy array

        Returns:
            Tuple of:
            - canonical_board_array: Shape (N_TOTAL_PLANES, SIZE, SIZE, SIZE)
            - transform_index: Integer 0-9 indicating which transformation yields canonical form
        """
        # Extract board array from any board-like object
        if hasattr(board, 'array'):
            board_array = board.array()
        elif isinstance(board, np.ndarray):
            board_array = board
        else:
            raise TypeError(f"Board must provide array() or be ndarray, got {type(board)}")

        # Validate shape
        if board_array.shape != (N_TOTAL_PLANES, SIZE, SIZE, SIZE):
            raise ValueError(
                f"Board array must be shape {(N_TOTAL_PLANES, SIZE, SIZE, SIZE)}, "
                f"got {board_array.shape}"
            )

        # ✅ OPTIMIZATION 1: Early exit for empty board
        if np.sum(board_array) == 0:
            return board_array, 0
        
        # ✅ OPTIMIZATION 2: Early exit for single-piece boards
        # Single piece has no meaningful symmetry advantage
        piece_planes = board_array[PIECE_SLICE]
        n_pieces = np.count_nonzero(piece_planes)
        if n_pieces == 1:
            return board_array, 0

        # ✅ OPTIMIZATION 3: Check cache using hash as key
        from game3d.cache.caches.zobrist import compute_zobrist
        board_hash = compute_zobrist(board_array, Color.WHITE)
        
        if board_hash in self._canonical_cache:
            self._cache_hits += 1
            cached_canonical, cached_transform = self._canonical_cache[board_hash]
            # Return copy to avoid mutation issues
            return cached_canonical.copy(), cached_transform

        self._cache_misses += 1

        # Initialize with original board
        min_hash = board_hash
        min_transform = 0
        min_array = board_array

        # Test all symmetry transformations
        for i in range(1, self.n_symmetries):
            # Apply transformation
            transformed = transform_board_array(board_array, i)

            # Compute hash (use WHITE perspective for consistency)
            hash_val = compute_zobrist(transformed, Color.WHITE)

            # Keep the smallest hash (canonical form)
            if hash_val < min_hash:
                min_hash = hash_val
                min_transform = i
                min_array = transformed

        # ✅ OPTIMIZATION 4: Cache the result
        # Implement simple LRU by removing oldest entry when full
        if len(self._canonical_cache) >= self._cache_size_limit:
            # Remove first (oldest) entry
            first_key = next(iter(self._canonical_cache))
            del self._canonical_cache[first_key]
        
        self._canonical_cache[board_hash] = (min_array.copy(), min_transform)

        return min_array, min_transform


    def transform_move(self, move: 'CompactMove', transform_idx: int) -> 'CompactMove':
        """
        Transform a move's coordinates according to symmetry.

        Args:
            move: CompactMove object to transform
            transform_idx: Which transformation to apply (0-9)

        Returns:
            New CompactMove with transformed coordinates
        """
        # Import here to avoid circular imports
        from game3d.cache.caches.transposition import CompactMove

        from_coord = move.from_coord
        to_coord = move.to_coord

        # Transform coordinates
        transformed_from = transform_coordinate_numba(from_coord, transform_idx)
        transformed_to = transform_coordinate_numba(to_coord, transform_idx)

        # Create new move with transformed coordinates
        return CompactMove(
            transformed_from,
            transformed_to,
            move.get_piece_type(),
            move.is_capture,
            move.get_captured_type(),
            move.is_promotion
        )

    def invert_move_transform(self, move: 'CompactMove', transform_idx: int) -> 'CompactMove':
        """
        Transform a move back from canonical to original orientation.

        Args:
            move: CompactMove in canonical orientation
            transform_idx: Transform index used to create canonical form

        Returns:
            CompactMove in original board orientation
        """
        # Since our transforms are self-inverse, we can reuse transform_move
        # For non-self-inverse transforms, use: inverse_idx = self.inverse_transforms[transform_idx]
        return self.transform_move(move, transform_idx)

    def get_transform_matrix(self, transform_idx: int) -> np.ndarray:
        """
        Get the 3x3 rotation matrix for a transform index.

        Args:
            transform_idx: Integer 0-9

        Returns:
            Shape (3, 3) rotation matrix
        """
        if not (0 <= transform_idx < self.n_symmetries):
            raise ValueError(f"Transform index must be 0-{self.n_symmetries-1}, got {transform_idx}")

        return self.transform_matrices[transform_idx].copy()

    def get_inverse_transform(self, transform_idx: int) -> int:
        """
        Get the inverse transform index.

        Note: For rotation matrices, the inverse is the transpose.
        Our rotation matrices are orthogonal, so inverse = transpose.
        Many are self-inverse (transpose = original).

        Args:
            transform_idx: Integer 0-9

        Returns:
            Integer inverse transform index
        """
        if not (0 <= transform_idx < self.n_symmetries):
            raise ValueError(f"Transform index must be 0-{self.n_symmetries-1}, got {transform_idx}")

        return int(self.inverse_transforms[transform_idx])
