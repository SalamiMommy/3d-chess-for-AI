# game3d/board/symmetry.py
"""Numpy-native symmetry transformations for 3D chess board - pure array operations."""

import numpy as np
from numba import njit, prange
from typing import Tuple, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from game3d.cache.caches.transposition import CompactMove
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
@njit(cache=True, nogil=True, parallel=True)
def transform_board_array(board_array: np.ndarray, transform_idx: int) -> np.ndarray:
    """
    Apply symmetry transformation to entire board array.
    
    ✅ OPTIMIZED: 
    - Parallel execution via prange on z-axis
    - Inlined matrix multiplication (avoids per-iteration array allocation)
    - Pre-fetch matrix outside loops
    """
    transformed = np.zeros_like(board_array)
    matrix = TRANSFORM_MATRICES[transform_idx]
    
    # Center point (4 for 9x9x9 board)
    center = (SIZE - 1) // 2

    for plane in range(N_TOTAL_PLANES):
        for z in prange(SIZE):  # ✅ Parallelize outer spatial loop
            for y in range(SIZE):
                for x in range(SIZE):
                    value = board_array[plane, z, y, x]
                    if value != 0:
                        # ✅ OPTIMIZED: Inline matrix multiplication
                        # Center coordinates around origin
                        cx = x - center
                        cy = y - center  
                        cz = z - center
                        
                        # Apply rotation matrix (inline, no array allocation)
                        rx = matrix[0, 0] * cx + matrix[0, 1] * cy + matrix[0, 2] * cz
                        ry = matrix[1, 0] * cx + matrix[1, 1] * cy + matrix[1, 2] * cz
                        rz = matrix[2, 0] * cx + matrix[2, 1] * cy + matrix[2, 2] * cz
                        
                        # Translate back and convert to int
                        tx = int(rx + center)
                        ty = int(ry + center)
                        tz = int(rz + center)

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
        
        # ✅ OPT 3.3: Hash-only canonical cache for faster lookups
        # Maps: original_hash -> (canonical_hash, transform_idx)
        self._hash_canonical_cache = {}
        self._hash_cache_size_limit = 50000  # Larger limit for hash-only cache

    # ✅ OPT 3.3: Fast canonical hash lookup
    def get_canonical_hash(self, board, active_color: int) -> Tuple[int, int]:
        """
        Get canonical Zobrist hash without reconstructing the full board array.
        
        ✅ OPTIMIZED: Returns (canonical_hash, transform_idx) using cached sparse
        coordinates. Much faster than get_canonical_form when only hash is needed.
        
        Use case: TT probing needs only the hash, not the full board array.
        
        Args:
            board: Game board object
            active_color: Active player color
            
        Returns:
            (canonical_hash, transform_idx) tuple
        """
        from game3d.cache.caches.zobrist import compute_zobrist_sparse
        from game3d.common.shared_types import Color
        
        # Get sparse representation
        if hasattr(board, 'cache_manager') and hasattr(board.cache_manager, 'occupancy_cache'):
            cache = board.cache_manager.occupancy_cache
            _, _, coords, types, colors = cache.export_buffer_data()
        else:
            # Fallback to full form
            _, transform_idx = self.get_canonical_form(board)
            from game3d.cache.caches.zobrist import compute_zobrist
            canonical_hash = compute_zobrist(board, active_color)
            return int(canonical_hash), transform_idx
        
        if coords.shape[0] == 0:
            return 0, 0
        
        # Compute original hash
        original_hash = compute_zobrist_sparse(coords, types, colors, active_color)
        
        # Check hash-only cache
        cache_key = (int(original_hash), active_color)
        if cache_key in self._hash_canonical_cache:
            self._cache_hits += 1
            return self._hash_canonical_cache[cache_key]
        
        self._cache_misses += 1
        
        # Find minimum hash across symmetries
        min_hash = original_hash
        min_transform = 0
        
        for i in range(1, self.n_symmetries):
            t_coords = transform_coordinates_batch(coords, i)
            t_hash = compute_zobrist_sparse(t_coords, types, colors, active_color)
            if t_hash < min_hash:
                min_hash = t_hash
                min_transform = i
        
        # Cache result
        if len(self._hash_canonical_cache) >= self._hash_cache_size_limit:
            # Simple FIFO eviction
            keys_to_remove = list(self._hash_canonical_cache.keys())[:1000]
            for k in keys_to_remove:
                del self._hash_canonical_cache[k]
        
        result = (int(min_hash), min_transform)
        self._hash_canonical_cache[cache_key] = result
        return result

    # ✅ OPT 3.3: Get all symmetry-equivalent hashes for TT probe
    def get_symmetry_equivalent_hashes(self, board, active_color: int) -> np.ndarray:
        """
        Get Zobrist hashes for all symmetry-equivalent positions.
        
        ✅ OPTIMIZED: Returns array of 10 hashes (one per symmetry) for
        parallel TT probing. Allows finding any symmetric position in TT.
        
        Use case: When probing TT, check all symmetric variants in one pass.
        
        Args:
            board: Game board object
            active_color: Active player color
            
        Returns:
            (10,) int64 array of symmetry-equivalent hashes
        """
        from game3d.cache.caches.zobrist import compute_zobrist_sparse
        
        hashes = np.empty(self.n_symmetries, dtype=np.int64)
        
        # Get sparse representation
        if hasattr(board, 'cache_manager') and hasattr(board.cache_manager, 'occupancy_cache'):
            cache = board.cache_manager.occupancy_cache
            _, _, coords, types, colors = cache.export_buffer_data()
        else:
            # Fallback: only return original hash
            from game3d.cache.caches.zobrist import compute_zobrist
            hashes[0] = compute_zobrist(board, active_color)
            hashes[1:] = hashes[0]  # Duplicate for consistency
            return hashes
        
        if coords.shape[0] == 0:
            return np.zeros(self.n_symmetries, dtype=np.int64)
        
        # Compute hash for each symmetry
        for i in range(self.n_symmetries):
            if i == 0:
                t_coords = coords
            else:
                t_coords = transform_coordinates_batch(coords, i)
            hashes[i] = compute_zobrist_sparse(t_coords, types, colors, active_color)
        
        return hashes

    def get_canonical_form(self, board) -> Tuple[np.ndarray, int]:
        """
        Find canonical form of board by testing all symmetry transformations.
        The canonical form is the one with the smallest Zobrist hash.
        """
        # 1. Early exits
        if isinstance(board, np.ndarray):
            if np.sum(board) == 0: return board, 0
        
        # 2. Determine input type and get initial hash
        coords = None
        types = None
        colors = None
        board_array = None
        board_hash = 0
        
        from game3d.cache.caches.zobrist import compute_zobrist, compute_zobrist_sparse
        from game3d.common.shared_types import Color, PIECE_SLICE, N_PIECE_TYPES
        
        is_sparse_ready = False
        
        if hasattr(board, 'cache_manager') and hasattr(board.cache_manager, 'occupancy_cache'):
             # Fast path: Use existing sparse cache
             cache = board.cache_manager.occupancy_cache
             # Unpack all sparse data directly
             occ_grid, ptype_grid, raw_coords, raw_types, raw_colors = cache.export_buffer_data()
             
             coords = raw_coords
             types = raw_types
             colors = raw_colors
             
             board_array = cache._occ # Reference for shape validation or fallback
             
             # Calculate hash sparse
             board_hash = compute_zobrist_sparse(coords, types, colors, Color.WHITE)
             is_sparse_ready = True
             
             # We can get the dense array lazily if needed
             if hasattr(board, 'array'):
                 board_array = board.array()
             else:
                 # Fallback if somehow sparse but no array method? Unlikely.
                 pass
                 
        elif isinstance(board, np.ndarray):
             # Dense input
             board_array = board
             # Calculate hash dense
             board_hash = compute_zobrist(board_array, Color.WHITE)
             is_sparse_ready = False
             
        elif hasattr(board, 'array'):
             # Generic object with array()
             board_array = board.array()
             board_hash = compute_zobrist(board_array, Color.WHITE)
             is_sparse_ready = False
        else:
             raise TypeError(f"Invalid board input: {type(board)}")

        # 3. Check Cache
        if board_hash in self._canonical_cache:
            self._cache_hits += 1
            cached_canonical, cached_transform = self._canonical_cache[board_hash]
            return cached_canonical.copy(), cached_transform

        self._cache_misses += 1
        
        # 4. Prepare Sparse Data (if not ready)
        if not is_sparse_ready:
             if board_array is None:
                 raise ValueError("Board array missing on cache miss")
                 
             # Check single piece
             piece_planes = board_array[PIECE_SLICE]
             if np.count_nonzero(piece_planes) <= 1:
                  return board_array, 0
                  
             # Extract sparse data from dense array
             occupied_mask = piece_planes > 0
             occ_3d = np.any(occupied_mask, axis=0) # (9,9,9)
             occupied_indices = np.where(occ_3d) # (z, y, x)
             
             if len(occupied_indices[0]) == 0:
                 return board_array, 0
                 
             z_idx, y_idx, x_idx = occupied_indices
             
             # Stack coords: (x, y, z)
             coords = np.column_stack((x_idx, y_idx, z_idx)).astype(COORD_DTYPE)
             
             # Get attributes
             occupied_values = board_array[:, z_idx, y_idx, x_idx]
             plane_indices = np.argmax(occupied_values, axis=0)
             
             is_white = plane_indices < N_PIECE_TYPES
             colors = np.where(is_white, Color.WHITE, Color.BLACK).astype(COLOR_DTYPE)
             types = np.where(is_white, 
                             plane_indices + 1, 
                             plane_indices - N_PIECE_TYPES + 1).astype(PIECE_TYPE_DTYPE)

        # 5. Sparse Symmetry Search
        min_hash = board_hash
        min_transform = 0
        
        n_sym = self.n_symmetries
        for i in range(1, n_sym):
            t_coords = transform_coordinates_batch(coords, i)
            t_hash = compute_zobrist_sparse(t_coords, types, colors, Color.WHITE)
            
            if t_hash < min_hash:
                min_hash = t_hash
                min_transform = i

        # 6. Reconstruct canonical array
        if min_transform == 0:
             canonical_array = board_array
        else:
             # If we started with sparse object and didn't get board_array yet
             if board_array is None and hasattr(board, 'array'):
                 board_array = board.array()
                 
             canonical_array = transform_board_array(board_array, min_transform)

        # 7. Cache result
        if len(self._canonical_cache) >= self._cache_size_limit:
            first_key = next(iter(self._canonical_cache))
            del self._canonical_cache[first_key]
        
        self._canonical_cache[board_hash] = (canonical_array.copy(), min_transform)

        return canonical_array, min_transform


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
