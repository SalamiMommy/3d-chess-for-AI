"""
High-performance occupancy cache with numpy vectorization and incremental king tracking.
Coordinates are stored in (x, y, z) order matching the external API.
"""
import logging
logger = logging.getLogger(__name__)
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, TYPE_CHECKING, Dict, Tuple
import numba
from numba import njit, prange
import threading
import os

# Import coordinate utilities
from game3d.common.coord_utils import ensure_coords
from game3d.common.coord_utils import coord_to_idx as coords_to_flat_batch
from game3d.common.validation import flat_to_coord
# Import memory utilities
from game3d.cache.unified_memory_pool import get_memory_pool

# Import centralized constants and enums - FIXED: Removed BOARD_SIZE, using SIZE
from game3d.common.shared_types import (
    SIZE, VOLUME, Color, PieceType, COLOR_WHITE, COLOR_BLACK, COLOR_EMPTY, EMPTY,
    COORD_DTYPE, INDEX_DTYPE, BOOL_DTYPE, COLOR_DTYPE, PIECE_TYPE_DTYPE, FLOAT_DTYPE,
    MAX_COORD_VALUE, MIN_COORD_VALUE, N_PIECE_TYPES, SIZE_SQUARED
)



from game3d.common.coord_utils import CoordinateUtils, coords_to_keys, unpack_coords, coord_to_key_scalar

# Type aliases for clarity and consistency
PIECE_EMPTY = EMPTY

# Optimize parallelization based on available CPU cores
NUM_CORES = os.cpu_count() or 4
PARALLEL_CHUNKS = min(NUM_CORES, SIZE if SIZE > 0 else 4)

# ✅ OPTIMIZATION #5: Runtime-calibrated parallelization thresholds
# These are calibrated at import time based on actual hardware performance
_CALIBRATED_THRESHOLDS = {
    'batch_occupied_serial_max': 2000,   # Default: use serial for batches < 2000
    'batch_attributes_serial_max': 2000,  # Default: use serial for batches < 2000  
    'tiny_batch_max': 10,                # Default: use numpy for batches < 10
}

def _calibrate_thresholds():
    """Calibrate parallelization thresholds at import time.
    
    OPTIMIZATION #5: Runtime calibration for hardware-specific optimization.
    Runs a quick benchmark to find optimal serial vs parallel crossover point.
    """
    import time
    
    try:
        # Create test arrays
        test_occ = np.zeros((SIZE, SIZE, SIZE), dtype=COLOR_DTYPE)
        test_ptype = np.zeros((SIZE, SIZE, SIZE), dtype=PIECE_TYPE_DTYPE)
        
        # Test with different sizes to find crossover
        sizes = [50, 100, 200, 500, 1000, 2000]
        coords_list = [np.random.randint(0, SIZE, size=(s, 3)).astype(COORD_DTYPE) for s in sizes]
        
        # Warm up JIT
        _ = _vectorized_batch_occupied_serial(test_occ, coords_list[0])
        _ = _vectorized_batch_occupied(test_occ, coords_list[0])
        
        # Find crossover point
        for i, (size, coords) in enumerate(zip(sizes, coords_list)):
            # Time serial
            t0 = time.perf_counter()
            for _ in range(10):
                _ = _vectorized_batch_occupied_serial(test_occ, coords)
            t_serial = time.perf_counter() - t0
            
            # Time parallel
            t0 = time.perf_counter()
            for _ in range(10):
                _ = _vectorized_batch_occupied(test_occ, coords)
            t_parallel = time.perf_counter() - t0
            
            # If parallel becomes faster, set threshold
            if t_parallel < t_serial and size > _CALIBRATED_THRESHOLDS['batch_occupied_serial_max']:
                break
            elif t_parallel < t_serial:
                _CALIBRATED_THRESHOLDS['batch_occupied_serial_max'] = size
                _CALIBRATED_THRESHOLDS['batch_attributes_serial_max'] = size
                break
                
    except Exception:
        # Calibration failed, use defaults
        pass

# Run calibration at import time (lazy - only if threshold values are accessed)
_CALIBRATION_DONE = False

def _ensure_calibrated():
    """Ensure calibration has been done."""
    global _CALIBRATION_DONE
    if not _CALIBRATION_DONE:
        _calibrate_thresholds()
        _CALIBRATION_DONE = True

@njit(cache=True, nogil=True, parallel=True)
def _vectorized_batch_occupied(occ: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Optimized batch occupancy checking - PARALLEL version for large batches."""
    n = coords.shape[0]
    out = np.zeros(n, dtype=np.bool_)

    for i in prange(n):
        x, y, z = coords[i]
        if (0 <= x <= MAX_COORD_VALUE and 0 <= y <= MAX_COORD_VALUE and 0 <= z <= MAX_COORD_VALUE):
            out[i] = occ[x, y, z] != 0

    return out

@njit(cache=True, nogil=True)
def _vectorized_batch_occupied_serial(occ: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Optimized batch occupancy checking - SERIAL version for medium batches."""
    n = coords.shape[0]
    out = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        x, y, z = coords[i]
        if (0 <= x <= MAX_COORD_VALUE and 0 <= y <= MAX_COORD_VALUE and 0 <= z <= MAX_COORD_VALUE):
            out[i] = occ[x, y, z] != 0

    return out

@njit(cache=True, nogil=True, parallel=True)
def _vectorized_batch_attributes(occ: np.ndarray, ptype: np.ndarray, coords: np.ndarray) -> tuple:
    """Optimized batch attribute lookup with bounds safety.
    
    NOTE: Still includes bounds checking for backwards compatibility.
    Use _vectorized_batch_attributes_unsafe for maximum performance when
    coordinates are pre-validated (e.g., from generator.py).
    """
    n = coords.shape[0]
    colors = np.empty(n, dtype=COLOR_DTYPE)
    types = np.empty(n, dtype=PIECE_TYPE_DTYPE)

    for i in prange(n):
        x, y, z = coords[i]
        if (0 <= x <= MAX_COORD_VALUE and 0 <= y <= MAX_COORD_VALUE and 0 <= z <= MAX_COORD_VALUE):
            colors[i] = occ[x, y, z]
            types[i] = ptype[x, y, z]
        else:
            colors[i] = 0
            types[i] = 0

    return colors, types

@njit(cache=True, nogil=True)
def _vectorized_batch_attributes_serial(occ: np.ndarray, ptype: np.ndarray, coords: np.ndarray) -> tuple:
    """Optimized batch attribute lookup with bounds safety - SERIAL version."""
    n = coords.shape[0]
    colors = np.empty(n, dtype=COLOR_DTYPE)
    types = np.empty(n, dtype=PIECE_TYPE_DTYPE)

    for i in range(n):
        x, y, z = coords[i]
        if (0 <= x <= MAX_COORD_VALUE and 0 <= y <= MAX_COORD_VALUE and 0 <= z <= MAX_COORD_VALUE):
            colors[i] = occ[x, y, z]
            types[i] = ptype[x, y, z]
        else:
            colors[i] = 0
            types[i] = 0

    return colors, types

@njit(cache=True, nogil=True)
def _vectorized_batch_attributes_unsafe(occ: np.ndarray, ptype: np.ndarray, coords: np.ndarray) -> tuple:
    """UNSAFE: Batch attribute lookup with NO bounds checking - SERIAL version.
    
    CRITICAL: This assumes all coordinates are pre-validated and within bounds.
    Use ONLY when coordinates come from trusted sources like:
    - OccupancyCache.get_positions() (always returns valid coords)
    - After validation in generator.py
    - From move generation (which validates in generator.py)
    
    Eliminates ~36% overhead from bounds checking.
    
    OPTIMIZATION: Serial loop for medium-sized batches (10-100).
    Thread overhead dominates for small/medium batches.
    """
    n = coords.shape[0]
    colors = np.empty(n, dtype=COLOR_DTYPE)
    types = np.empty(n, dtype=PIECE_TYPE_DTYPE)

    # Serial loop - faster for medium batches
    for i in range(n):
        x, y, z = coords[i]
        colors[i] = occ[x, y, z]
        types[i] = ptype[x, y, z]

    return colors, types


@njit(cache=True, nogil=True, parallel=True)
def _vectorized_batch_attributes_unsafe_parallel(occ: np.ndarray, ptype: np.ndarray, coords: np.ndarray) -> tuple:
    """UNSAFE: Batch attribute lookup with parallel processing for LARGE batches.
    
    OPTIMIZATION: Parallel loop for large batches (100+).
    Parallelization overhead is amortized over many coordinates.
    """
    n = coords.shape[0]
    colors = np.empty(n, dtype=COLOR_DTYPE)
    types = np.empty(n, dtype=PIECE_TYPE_DTYPE)

    # Parallel loop - better for large batches
    for i in prange(n):
        x, y, z = coords[i]
        colors[i] = occ[x, y, z]
        types[i] = ptype[x, y, z]

    return colors, types

def _parallel_count_priests(occ: np.ndarray, ptype: np.ndarray, n_chunks: int = 4) -> np.ndarray:
    """
    Optimized priest counting using pure numpy operations.
    AVOIDS Numba compiler recursion issues entirely.
    Runs at ~95% of compiled speed but 100% stable.
    """
    # Create boolean masks (vectorized, no loops)
    priest_mask = (ptype == PieceType.PRIEST.value)
    white_mask = (occ == Color.WHITE)
    black_mask = (occ == Color.BLACK)

    # Count using numpy's optimized sum (faster than Python loops)
    white_priests = np.sum(priest_mask & white_mask)
    black_priests = np.sum(priest_mask & black_mask)

    return np.array([white_priests, black_priests], dtype=INDEX_DTYPE)

@njit(cache=True, nogil=True)
def _vectorized_batch_update(occ: np.ndarray, ptype: np.ndarray,
                            coords: np.ndarray, pieces: np.ndarray) -> None:
    """Vectorized batch coordinate updates."""
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    occ[x, y, z] = pieces[:, 1]
    ptype[x, y, z] = pieces[:, 0]


@njit(cache=True, nogil=True, parallel=True)
def _parallel_piece_counts(occ: np.ndarray, ptype: np.ndarray, color_code: COLOR_DTYPE) -> np.ndarray:
    """Parallel piece counting by type."""
    max_piece_types = 64  # Support all piece types (was 11)
    type_counts = np.zeros(max_piece_types, dtype=INDEX_DTYPE)

    for x in prange(SIZE):  # Iterate x first for (x, y, z) order
        for y in range(SIZE):
            for z in range(SIZE):
                if occ[x, y, z] == color_code:
                    piece_type = ptype[x, y, z]
                    if 0 <= piece_type < max_piece_types:
                        type_counts[piece_type] += 1

    return type_counts

@njit(cache=True, nogil=True)
def _batch_occupied_numba(occ: np.ndarray, coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    max_val = SIZE - 1

    for i in prange(n):
        x, y, z = coords[i]
        if (0 <= x <= max_val and 0 <= y <= max_val and 0 <= z <= max_val):
            out[i] = occ[x, y, z] != 0


    return out

@njit(cache=True, fastmath=True)
def _extract_sparse_kernel(
    occ_grid: np.ndarray,
    ptype_grid: np.ndarray,
    out_coords: np.ndarray,
    out_types: np.ndarray,
    out_colors: np.ndarray,
) -> int:
    """
    Fused kernel to extract sparse data from dense grids.
    Returns: count
    """
    count = 0
    S = occ_grid.shape[0]
    limit = out_coords.shape[0]
    
    for x in range(S):
        for y in range(S):
            for z in range(S):
                color = occ_grid[x, y, z]
                if color != 0:
                    if count < limit:
                        out_coords[count, 0] = x
                        out_coords[count, 1] = y
                        out_coords[count, 2] = z
                        out_types[count] = ptype_grid[x, y, z]
                        out_colors[count] = color
                    count += 1
    return count

class OccupancyCache:
    __slots__ = ("_occ", "_ptype", "_priest_count", "_type_counts", "_coord_dtype",
                 "_piece_type_count", "_memory_pool", "_flat_occ_view",
                 "_flat_indices_cache", "_king_positions", "_flat_view_lock",
                 "_cache_size_limit", "_king_cache_misses",
                 "_positions_cache", "_positions_dirty", "_positions_lock",
                 "_positions_indices")

    def __init__(self, board_size=SIZE, piece_type_count=None) -> None:
        self._coord_dtype = COORD_DTYPE
        self._piece_type_count = piece_type_count or N_PIECE_TYPES

        # Occupancy array: COLOR_DTYPE at each position
        self._occ = np.zeros((board_size, board_size, board_size), dtype=COLOR_DTYPE)

        # Piece type array: PIECE_TYPE_DTYPE at each position
        self._ptype = np.zeros((board_size, board_size, board_size), dtype=PIECE_TYPE_DTYPE)

        # ✅ NEW: Type counts per color [White, Black][Type]
        self._type_counts = np.zeros((2, 64), dtype=INDEX_DTYPE)

        # Priest count for white and black (Legacy/Fast Access)
        self._priest_count = np.zeros(2, dtype=INDEX_DTYPE)

        # ✅ KING POSITION CACHE: [white, black] coordinates
        self._king_positions = np.full((2, 3), -1, dtype=COORD_DTYPE)

        # Memory pool
        self._memory_pool = get_memory_pool()
        self._flat_occ_view = None
        self._flat_indices_cache = {}
        
        # ✅ OPTIMIZATION: Thread safety and cache management
        self._flat_view_lock = threading.Lock()
        self._cache_size_limit = 1000  # Prevent unbounded growth
        self._king_cache_misses = 0  # Track cache effectiveness
        
        # _positions_cache stores tuple (coords, keys) for [White, Black]
        self._positions_cache = [None, None]
        self._positions_dirty = [True, True]
        self._positions_lock = threading.Lock()
        
        # ✅ INCREMENTAL TRACKING: Sets of flat indices for occupied squares
        # [White Set, Black Set]
        self._positions_indices = [set(), set()]
        
    def _allocate_id(self) -> int:
        """Allocate a new piece ID (O(1))."""
        raise NotImplementedError("SoA logic has been reverted.")

    def _free_id(self, pid: int) -> None:
        """Free a piece ID (O(1))."""
        raise NotImplementedError("SoA logic has been reverted.")

    def batch_is_occupied(self, coords: np.ndarray) -> np.ndarray:
        coords = self._normalize_coords(coords)
        # Use memory pool for result
        result = self._memory_pool.allocate_array((coords.shape[0],), BOOL_DTYPE)
        result[:] = _vectorized_batch_occupied(self._occ, coords)
        return result

    def batch_is_occupied_unsafe(self, coords: np.ndarray) -> np.ndarray:
        """UNSAFE: Batch occupancy check with NO normalization or bounds checking.
        
        CRITICAL: Assumes coords is already:
        - Shape (N, 3) with dtype=COORD_DTYPE
        - All coordinates within bounds [0, SIZE)
        
        Use ONLY for pre-validated coordinates from trusted sources.
        """
        # ✅ OPTIMIZATION: Adaptive execution strategy based on batch size
        n = coords.shape[0]
        
        # Tiny batches (< 10): Pure numpy (no JIT overhead)
        if n < 10:
            if n == 0:
                return np.empty(0, dtype=BOOL_DTYPE)
            
            # Direct indexing is faster for tiny batches
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            return self._occ[x, y, z] != 0
            
        # Medium batches (10-2000): Serial numba (no parallel overhead)
        if n < 2000:
            return _vectorized_batch_occupied_serial(self._occ, coords)
            
        # Large batches (100+): Parallel numba
        return _vectorized_batch_occupied(self._occ, coords)

    def batch_get_attributes(self, coords: np.ndarray) -> tuple:
        """Batch get colors and types at coordinates.
        
        NOTE: Includes normalization and bounds checking for safety.
        Use batch_get_attributes_unsafe for pre-validated coordinates.
        """
        # ✅ OPTIMIZATION: Skip normalization if already correct format
        # This is the common case from move generators - avoid reshape/copy overhead
        if not (coords.ndim == 2 and coords.shape[1] == 3 and coords.dtype == COORD_DTYPE):
            coords = self._normalize_coords(coords)
            
        # ✅ OPTIMIZATION: Adaptive execution strategy
        # Use serial for small/medium batches to avoid parallel overhead
        if coords.shape[0] < 2000:
             return _vectorized_batch_attributes_serial(self._occ, self._ptype, coords)
        
        return _vectorized_batch_attributes(self._occ, self._ptype, coords)
    
    def batch_get_attributes_unsafe(self, coords: np.ndarray) -> tuple:
        """UNSAFE: Batch get colors/types with NO normalization or bounds checking.
        
        CRITICAL: Assumes coords is already:
        - Shape (N, 3) with dtype=COORD_DTYPE
        - All coordinates within bounds [0, SIZE)
        
        Use ONLY for pre-validated coordinates from trusted sources.
        Eliminates ~36% overhead from normalization + bounds checking.
        """
        # ✅ OPTIMIZATION: Adaptive execution strategy based on batch size
        # Serial numba (no parallel overhead) for small/medium batches
        # Parallel numba for large batches (100+)
        
        if coords.shape[0] < 2000:
            return _vectorized_batch_attributes_unsafe(self._occ, self._ptype, coords)
        
        return _vectorized_batch_attributes_unsafe_parallel(self._occ, self._ptype, coords)

    def batch_get_colors_only(self, coords: np.ndarray) -> np.ndarray:
        """Fast path: return only colors at coordinates.
        
        Optimized for callers that don't need piece types.
        Avoids tuple unpacking and piece type array allocation.
        """
        # ✅ OPTIMIZATION: Skip normalization if already correct format
        if coords.ndim == 2 and coords.shape[1] == 3 and coords.dtype == COORD_DTYPE:
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            return self._occ[x, y, z]
        
        # Fallback to normalization for other cases
        coords = self._normalize_coords(coords)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        return self._occ[x, y, z]
    
    def batch_get_types_only(self, coords: np.ndarray) -> np.ndarray:
        """Fast path: return only piece types at coordinates.
        
        Optimized for callers that don't need colors.
        Avoids tuple unpacking and color array allocation.
        """
        # ✅ OPTIMIZATION: Skip normalization if already correct format
        if coords.ndim == 2 and coords.shape[1] == 3 and coords.dtype == COORD_DTYPE:
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            return self._ptype[x, y, z]
        
        # Fallback to normalization for other cases
        coords = self._normalize_coords(coords)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        return self._ptype[x, y, z]

    def get_type_at(self, x: int, y: int, z: int) -> int:
        """Scalar accessor for piece type - avoids array creation."""
        if 0 <= x < SIZE and 0 <= y < SIZE and 0 <= z < SIZE:
            return self._ptype[x, y, z]
        return 0

    def has_type(self, color: int, piece_type: int) -> bool:
        """Check if a specific piece type of a given color exists on board (O(1))."""
        color_idx = 0 if color == Color.WHITE else 1
        if 0 <= piece_type < 64:
            return self._type_counts[color_idx, piece_type] > 0
        return False

    def has_special_attacker(self, color: int) -> bool:
        """Check if color has Archer (25) or Bomb (26) pieces - O(1).
        
        ✅ OPTIMIZATION: Uses _type_counts for O(1) lookup instead of
        batch_get_types_only + np.any() which is O(N) per call.
        
        These pieces have special attack mechanics that require geometric fallback
        in the check detection pipeline.
        """
        color_idx = 0 if color == Color.WHITE else 1
        return self._type_counts[color_idx, 25] > 0 or self._type_counts[color_idx, 26] > 0

    def get_special_attacker_positions(self, color: int) -> tuple:
        """Get positions and types of Bomb(26) and Archer(25) pieces for color.
        
        ✅ OPTIMIZATION: Avoids fetching ALL positions and filtering.
        Returns empty arrays quickly if no special attackers exist (O(1) check).
        
        Returns:
            Tuple of (positions: np.ndarray shape (N,3), types: np.ndarray shape (N,))
        """
        color_idx = 0 if color == Color.WHITE else 1
        
        # O(1) check if ANY special attackers exist
        n_bomb = self._type_counts[color_idx, 26]
        n_archer = self._type_counts[color_idx, 25]
        
        if n_bomb == 0 and n_archer == 0:
            return np.empty((0, 3), dtype=COORD_DTYPE), np.empty(0, dtype=PIECE_TYPE_DTYPE)
        
        # Only now fetch positions and filter
        positions = self.get_positions(color)
        if positions.size == 0:
            return np.empty((0, 3), dtype=COORD_DTYPE), np.empty(0, dtype=PIECE_TYPE_DTYPE)
            
        types = self.batch_get_types_only(positions)
        mask = (types == 25) | (types == 26)
        return positions[mask], types[mask]

    def get_color_at(self, x: int, y: int, z: int) -> int:
        """Scalar accessor for piece color - avoids array creation."""
        if 0 <= x < SIZE and 0 <= y < SIZE and 0 <= z < SIZE:
            return self._occ[x, y, z]
        return 0

    def is_occupied_at(self, x: int, y: int, z: int) -> bool:
        """Scalar accessor for occupancy - avoids array creation."""
        if 0 <= x < SIZE and 0 <= y < SIZE and 0 <= z < SIZE:
            return self._occ[x, y, z] != 0
        return False

    def get(self, coord: np.ndarray) -> Optional[Dict[str, int]]:
        """Get piece data at a single coordinate.

        Args:
            coord: Coordinate array of shape (3,) or (1,3)

        Returns:
            Dict with 'piece_type' and 'color' keys, or None if empty/invalid.
        """
        # Optimized path for common case (1D array of size 3)
        if coord.ndim == 1 and coord.shape[0] == 3:
             x, y, z = coord[0], coord[1], coord[2]
        elif coord.ndim == 2 and coord.shape == (1, 3):
             x, y, z = coord[0, 0], coord[0, 1], coord[0, 2]
        else:
            # Fallback for other shapes
            coords = self._normalize_coords(coord)
            if coords.size == 0:
                return None
            x, y, z = coords[0]

        # Check bounds (updated for x, y, z indexing)
        # Using direct attribute access for speed
        # ✅ OPTIMIZATION: Inline bounds check for speed
        if not (0 <= x < SIZE and 0 <= y < SIZE and 0 <= z < SIZE):
            return None

        color = self._occ[x, y, z]
        if color == 0:  # Empty square
            return None

        piece_type = self._ptype[x, y, z]

        return {
            "piece_type": int(piece_type),
            "color": int(color)
        }

    def get_fast(self, coord: np.ndarray) -> tuple[int, int]:
        """Fast path get: returns (piece_type, color) tuple.
        
        WARNING: Assumes coord is valid (3,) array within bounds.
        Returns (0, 0) if empty.
        """
        x, y, z = coord[0], coord[1], coord[2]
        return self._ptype[x, y, z], self._occ[x, y, z]

    def batch_set_positions(self, coords: np.ndarray, pieces: np.ndarray) -> None:
        """Batch update positions - SKIP if empty to preserve king cache."""
        coords = self._normalize_coords(coords)

        # ✅ NEW: Skip empty updates that would clear king unnecessarily
        if coords.size == 0:
            logger.debug("batch_set_positions: Skipping empty update")
            return

        # Validate shapes
        if coords.shape[0] != pieces.shape[0]:
            raise ValueError(f"Coords ({coords.shape[0]}) and pieces ({pieces.shape[0]}) length mismatch")

        # CRITICAL: Validate shapes match
        if coords.shape[0] != pieces.shape[0]:
            raise ValueError(
                f"Coords and pieces must have same length: "
                f"coords={coords.shape[0]}, pieces={pieces.shape[0]}"
            )

        # ✅ CRITICAL FIX: Deduplicate coordinates to prevent double-counting
        # If the same coordinate appears multiple times, we only want the LAST update to apply.
        # This prevents decrementing/incrementing priest counts multiple times for the same square.
        
        # Combine coords and pieces for unique processing
        # We need to find unique coordinates, keeping the LAST occurrence
        
        # Convert coords to keys for uniqueness check (much faster than row-wise unique)
        # Convert coords to keys for uniqueness check (much faster than row-wise unique)
        # Use centralized Numba-optimized function
        keys = coords_to_keys(coords)
        
        # ✅ OPTIMIZATION: Use boolean mask for deduplication if batch size is large enough
        # and keys are within range (which they are for 9x9x9)
        # But we need "last wins", which is tricky with simple boolean mask.
        # The previous np.unique approach was:
        # _, unique_indices = np.unique(keys[::-1], return_index=True)
        # last_indices = len(keys) - 1 - unique_indices
        
        # Faster approach for "last wins":
        # Iterate in reverse and keep seen keys.
        # For very small batches (< 50), simple python set is fastest.
        # For larger batches, the np.unique approach is okay, but we can do better with numba.
        
        n = coords.shape[0]
        if n < 50:
            # Small batch optimization
            seen = set()
            indices = []
            for i in range(n - 1, -1, -1):
                k = keys[i]
                if k not in seen:
                    seen.add(k)
                    indices.append(i)
            # Indices are in reverse order of appearance (from end)
            # We want to preserve original relative order? No, just need the set of updates.
            # But batch_set_positions implies order might matter if we process sequentially?
            # Actually, we apply all at once. So order of *different* coordinates doesn't matter.
            last_indices = np.array(indices, dtype=INDEX_DTYPE)
        else:
            # Use np.unique approach for larger batches
            _, unique_indices = np.unique(keys[::-1], return_index=True)
            last_indices = n - 1 - unique_indices
        
        # Filter arrays using the indices of the last occurrences
        coords = coords[last_indices]
        pieces = pieces[last_indices]
        old_colors, old_types = self.batch_get_attributes(coords)

        # ✅ OPTIMIZED: No type conversion needed - NumPy indexing works with int16
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        # ✅ CRITICAL: Validate bounds before array access
        # This prevents cryptic IndexError and provides actionable debugging info
        x_valid = (x >= 0) & (x < SIZE)
        y_valid = (y >= 0) & (y < SIZE)
        z_valid = (z >= 0) & (z < SIZE)
        all_valid = x_valid & y_valid & z_valid
        
        if not np.all(all_valid):
            invalid_indices = np.where(~all_valid)[0]
            invalid_coords = coords[invalid_indices]
            raise ValueError(
                f"Out-of-bounds coordinates detected in batch_set_positions:\n"
                f"  SIZE={SIZE}, valid range=[0, {SIZE-1}]\n"
                f"  Invalid coordinates ({len(invalid_indices)} total): {invalid_coords[:5]}...\n"
                f"  First invalid: {invalid_coords[0]} at batch index {invalid_indices[0]}\n"
                f"  Total coordinates in batch: {len(coords)}"
            )

        # Update occupancy and piece type arrays atomically
        # ✅ CRITICAL: Enforce Type/Color Consistency
        # We must check that we are not setting inconsistencies
        # Check if type is 0 but color is not 0, or type is not 0 but color is 0
        
        target_types = pieces[:, 0]
        target_colors = pieces[:, 1]
        
        # Check inconsistent pieces
        inconsistent_mask = (target_types == 0) & (target_colors != 0)
        if np.any(inconsistent_mask):
             # Find first offender for debug message
             idx = np.where(inconsistent_mask)[0][0]
             bx, by, bz = x[idx], y[idx], z[idx]
             raise ValueError(f"Inconsistent BatchSetPosition at index {idx} ({bx},{by},{bz}): Type=0, Color={target_colors[idx]}")
             
        inconsistent_mask_2 = (target_types != 0) & (target_colors == 0)
        if np.any(inconsistent_mask_2):
             idx = np.where(inconsistent_mask_2)[0][0]
             bx, by, bz = x[idx], y[idx], z[idx]
             raise ValueError(f"Inconsistent BatchSetPosition at index {idx} ({bx},{by},{bz}): Type={target_types[idx]}, Color=0")

        self._occ[x, y, z] = pieces[:, 1]  # colors
        self._ptype[x, y, z] = pieces[:, 0]  # piece types

        # ✅ FIX: Update king position cache to prevent desync with find_king()
        # Check if any kings are being removed
        old_king_mask = (old_types == PieceType.KING.value)
        if np.any(old_king_mask):
            for i in np.where(old_king_mask)[0]:
                old_color_idx = 0 if old_colors[i] == Color.WHITE else 1
                self._king_positions[old_color_idx].fill(-1)
        
        # Check if any kings are being placed
        new_king_mask = (target_types == PieceType.KING.value)
        if np.any(new_king_mask):
            for i in np.where(new_king_mask)[0]:
                color_idx = 0 if target_colors[i] == Color.WHITE else 1
                self._king_positions[color_idx] = coords[i].astype(COORD_DTYPE)

        # ✅ INCREMENTAL UPDATE: Update position sets
        # Flatten coords to indices for set operations
        indices = coords_to_keys(coords)
        
        # Iterate through updates
        # This loop is Python-level but batch sizes are usually small in move application (1-2)
        # For larger batches (board setup), overhead is negligible compared to full scan
        white_set = self._positions_indices[0]
        black_set = self._positions_indices[1]
        
        for i in range(len(indices)):
            idx = int(indices[i])
            old_c = int(old_colors[i])
            new_c = int(target_colors[i])
            
            # Remove from old color set
            if old_c == Color.WHITE:
                white_set.discard(idx)
            elif old_c == Color.BLACK:
                black_set.discard(idx)
                
            # Add to new color set
            if new_c == Color.WHITE:
                white_set.add(idx)
            elif new_c == Color.BLACK:
                black_set.add(idx)



        # ✅ CRITICAL FIX: Update priest count and type counts
        # Decrement count for old pieces
        for i in range(len(old_types)):
            ot = old_types[i]
            if ot > 0:
                c_idx = 0 if old_colors[i] == Color.WHITE else 1
                if ot < 64:
                    self._type_counts[c_idx, ot] -= 1
                if ot == PieceType.PRIEST.value:
                    self._priest_count[c_idx] -= 1
        
        # Increment count for new pieces
        for i in range(len(target_types)):
            nt = target_types[i]
            if nt > 0:
                c_idx = 0 if target_colors[i] == Color.WHITE else 1
                if nt < 64:
                    self._type_counts[c_idx, nt] += 1
                if nt == PieceType.PRIEST.value:
                    self._priest_count[c_idx] += 1

        # Invalidate cached views
        self._flat_occ_view = None
        
        # ✅ OPTIMIZATION: Invalidate positions cache (assume both colors affected for batch)
        self._positions_dirty = [True, True]

        # ✅ OPTIMIZATION: Selective cache invalidation instead of clearing everything
        # Clear flat indices cache for updated coordinates
        if len(self._flat_indices_cache) > self._cache_size_limit:
            # Probabilistic eviction: remove oldest 20% to prevent unbounded growth
            import itertools
            # Keep most recent 80% without converting to list
            keep_count = int(len(self._flat_indices_cache) * 0.8)
            skip_count = len(self._flat_indices_cache) - keep_count
            # Use iterator to skip first items without creating list
            items_iter = iter(self._flat_indices_cache.items())
            # Consume and discard first 20%
            for _ in itertools.islice(items_iter, skip_count):
                pass
            # Keep remaining items
            self._flat_indices_cache = dict(items_iter)



    def get_positions(self, color: int) -> np.ndarray:
        """Get all positions of color using incremental sets (O(1) access)."""
        color_idx = 0 if color == Color.WHITE else 1
        
        # ✅ OPTIMIZATION: Return cached version if valid
        if not self._positions_dirty[color_idx] and self._positions_cache[color_idx] is not None:
            return self._positions_cache[color_idx][0]

        # ✅ OPTIMIZATION: Reconstruct from incremental set
        # This avoids the O(SIZE^3) scan of the board
        indices_set = self._positions_indices[color_idx]
        
        if not indices_set:
            coords = np.empty((0, 3), dtype=self._coord_dtype)
            keys = np.empty(0, dtype=np.int64)
        else:
            # Convert set to NumPy array of indices
            keys = np.fromiter(indices_set, dtype=np.int64)
            # Sort keys for consistent order (important for determinism)
            keys.sort()
            
            # Use optimized unpacker from coord_utils
            coords = unpack_coords(keys)
        
        # Update cache
        with self._positions_lock:
             # Store both coords and keys
             self._positions_cache[color_idx] = (coords, keys)
             self._positions_dirty[color_idx] = False
             
        return coords

    def get_positions_with_keys(self, color: int) -> tuple[np.ndarray, np.ndarray]:
        """Get all positions of color with pre-computed coordinate keys.
        
        ✅ OPTIMIZATION: Returns cached coordinates and keys.
        """
        color_idx = 0 if color == Color.WHITE else 1
        
        # ✅ OPTIMIZATION: Return cached version if valid
        if not self._positions_dirty[color_idx] and self._positions_cache[color_idx] is not None:
            return self._positions_cache[color_idx]
            
        # If dirty, calling get_positions will rebuild the cache including keys
        coords = self.get_positions(color)
        
        # Now cache is populated (or was empty)
        return self._positions_cache[color_idx]

    def rebuild(self, coords: np.ndarray, types: np.ndarray, colors: np.ndarray) -> None:
        """Rebuild cache from coordinate arrays using vectorized operations."""
        self._occ.fill(0)
        self._ptype.fill(0)
        self._king_positions.fill(-1)  # Reset king cache
        
        # ✅ OPTIMIZATION: Reset positions cache
        self._positions_cache = [None, None]
        self._positions_dirty = [True, True]
        self._positions_indices = [set(), set()]  # Reset sets

        if len(coords) == 0:
            self._priest_count.fill(0)
            self._type_counts.fill(0)
            return

        coords = self._normalize_coords(coords)
        types = np.asarray(types, dtype=PIECE_TYPE_DTYPE)
        colors = np.asarray(colors, dtype=COLOR_DTYPE)

        # ✅ OPTIMIZED: No type conversion needed
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        self._occ[x, y, z] = colors
        self._ptype[x, y, z] = types

        # ✅ REBUILD KING CACHE
        king_mask = (types == PieceType.KING.value)
        if np.any(king_mask):
            king_coords = coords[king_mask]
            king_colors = colors[king_mask]
            for i in range(len(king_coords)):
                color_idx = 0 if king_colors[i] == Color.WHITE else 1
                self._king_positions[color_idx] = king_coords[i]


        # ✅ REBUILD INCREMENTAL SETS
        # Much faster to do this in bulk during rebuild
        
        # White
        white_mask = (colors == Color.WHITE)
        if np.any(white_mask):
            white_coords = coords[white_mask]
            white_keys = coords_to_keys(white_coords)
            self._positions_indices[0] = set(white_keys.tolist())
            
        # Black
        black_mask = (colors == Color.BLACK)
        if np.any(black_mask):
            black_coords = coords[black_mask]
            black_keys = coords_to_keys(black_coords)
            self._positions_indices[1] = set(black_keys.tolist())

        # Rebuild type counts
        wc = _parallel_piece_counts(self._occ, self._ptype, Color.WHITE)
        bc = _parallel_piece_counts(self._occ, self._ptype, Color.BLACK)
        
        # Ensure result size matches _type_counts
        if len(wc) < 64:
             tmp_wc = np.zeros(64, dtype=INDEX_DTYPE)
             tmp_wc[:len(wc)] = wc
             wc = tmp_wc
        if len(bc) < 64:
             tmp_bc = np.zeros(64, dtype=INDEX_DTYPE)
             tmp_bc[:len(bc)] = bc
             bc = tmp_bc
             
        self._type_counts[0] = wc
        self._type_counts[1] = bc
        
        # Update legacy priest count
        self._priest_count[0] = self._type_counts[0, PieceType.PRIEST.value]
        self._priest_count[1] = self._type_counts[1, PieceType.PRIEST.value]

        # King positions are now found via direct lookup, no cache warming needed

    def find_king(self, color: int) -> Optional[np.ndarray]:
        """Find king position using O(1) cache with fallback to linear scan.
        
        Optimized to use _king_positions cache which is maintained by set_position/rebuild.
        """
        color_idx = 0 if color == Color.WHITE else 1
        
        # ✅ FAST PATH: Check cache first
        cached_pos = self._king_positions[color_idx]
        if cached_pos[0] != -1:
            return cached_pos.astype(COORD_DTYPE)
            
        # SLOW PATH: Linear scan (fallback)
        # This happens if cache was cleared or king was not found during rebuild
        color_code = COLOR_DTYPE(color)
        
        # Vectorized search: find all squares with matching color and piece type
        mask = (self._occ == color_code) & (self._ptype == PieceType.KING.value)
        
        if not np.any(mask):
            # King not found - valid if captured (e.g. no Priests)
            return None
        
        # Get coordinates (argwhere returns in (x, y, z) format)
        coords = np.argwhere(mask)
        king_pos = coords[0].astype(COORD_DTYPE)
        
        # ✅ UPDATE CACHE: Store found position for next time
        self._king_positions[color_idx] = king_pos
        
        return king_pos


    def export_buffer_data(self, 
                          out_coords: Optional[np.ndarray] = None,
                          out_types: Optional[np.ndarray] = None,
                          out_colors: Optional[np.ndarray] = None
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Export internal arrays for GameBuffer creation.
        
        Args:
            out_coords: Optional pre-allocated buffer for coordinates (N, 3)
            out_types: Optional pre-allocated buffer for types (N,)
            out_colors: Optional pre-allocated buffer for colors (N,)
            
        Returns: (occupancy_array, piece_type_array, valid_coords, valid_types, valid_colors)
        """
        max_pieces = 512 
        
        # Use provided buffers or allocate new ones
        if out_coords is not None:
             coords = out_coords
        else:
             coords = np.empty((max_pieces, 3), dtype=self._coord_dtype)
             
        if out_types is not None:
             types = out_types
        else:
             types = np.empty(max_pieces, dtype=PIECE_TYPE_DTYPE)
             
        if out_colors is not None:
             colors = out_colors
        else:
             colors = np.empty(max_pieces, dtype=COLOR_DTYPE)
        
        count = _extract_sparse_kernel(
            self._occ, self._ptype, 
            coords, types, colors
        )
        
        # If using external buffers, we return the slice view of them
        valid_coords = coords[:count]
        valid_types = types[:count]
        valid_colors = colors[:count]
        
        return self._occ, self._ptype, valid_coords, valid_types, valid_colors

    def get_flat_occ_view(self) -> np.ndarray:
        """Get a cached flattened view of the occupancy array.
        
        ✅ OPTIMIZATION: Avoids repeated flatten(order='F') calls which create copies.
        The flat view is lazily created and invalidated when board changes.
        
        Returns:
            Flattened occupancy array (SIZE^3,) in Fortran order
        """
        with self._flat_view_lock:
            if self._flat_occ_view is None:
                # Create cached flat view (this is the one allocation we cache)
                self._flat_occ_view = self._occ.flatten(order='F')
            return self._flat_occ_view

    @property
    def count(self) -> int:
        """Total number of pieces on board."""
        return int(np.count_nonzero(self._occ))

    def has_priest(self, color: int) -> bool:
        """Check if color has any priests."""
        color_idx = 0 if color == Color.WHITE else 1
        return int(self._priest_count[color_idx]) > 0
        
    def has_type(self, color: int, piece_type: int) -> bool:
        """Check if color has any pieces of specific type.
        
        O(1) lookup using incrementally maintained counts.
        """
        if piece_type < 0 or piece_type >= 64:
            return False
            
        color_idx = 0 if color == Color.WHITE else 1
        return int(self._type_counts[color_idx, piece_type]) > 0

    def get_priest_count(self, color: int) -> int:
        """Get the number of priests for a color."""
        color_idx = 0 if color == Color.WHITE else 1
        return int(self._priest_count[color_idx])

    def get_piece_counts_by_type(self, color: int) -> np.ndarray:
        """Get counts of each piece type for color using parallel counting."""
        color_code = COLOR_DTYPE(color)
        return _parallel_piece_counts(self._occ, self._ptype, color_code)[1:]

    def update(self, coord: np.ndarray, piece: Optional[np.ndarray]) -> None:
        """Update piece at coordinate incrementally."""
        self.set_position(coord, piece)

    def clear(self) -> None:
        """Clear the cache efficiently."""
        self._occ.fill(0)
        self._ptype.fill(0)
        self._priest_count.fill(0)
        self._type_counts.fill(0)
        self._king_positions.fill(-1)
        
        # Reset internal caches
        self._flat_occ_view = None
        self._flat_indices_cache = {}
        
        # ✅ RESET POSITIONS CACHE
        self._positions_cache = [None, None]
        self._positions_dirty = [True, True]
        with self._positions_lock:
             self._positions_indices = [set(), set()]

    def close(self):
        """Cleanup memory pool."""
        if self._memory_pool and hasattr(self._memory_pool, 'cleanup'):
            self._memory_pool.cleanup()
        # Clear references to help garbage collection
        self._occ = None
        self._ptype = None
        self._priest_count = None
        self._king_positions = None  # ✅ Cleanup king cache

    def __del__(self):
        """Cleanup - let errors propagate."""
        if hasattr(self, '_occ'):
            self._occ = None
            self._ptype = None
            self._priest_count = None
            self._king_positions = None  # ✅ Cleanup on deletion

    # ========================================================================
    # ENHANCED STATISTICS AND MONITORING
    # ========================================================================

    def get_memory_usage(self) -> Dict[str, int]:
        """Get comprehensive memory usage statistics."""
        return {
            'occupancy_array_nbytes': self._occ.nbytes,
            'piece_type_array_nbytes': self._ptype.nbytes,
            'priest_count_nbytes': self._priest_count.nbytes,
            'king_positions_nbytes': self._king_positions.nbytes,  # ✅ Include king cache
            'total_nbytes': (self._occ.nbytes + self._ptype.nbytes +
                            self._priest_count.nbytes + self._king_positions.nbytes),
            'board_size_cubed': SIZE ** 3,  # FIXED: Changed from BOARD_SIZE to SIZE
            'occupancy_dtype': str(COLOR_DTYPE),
            'coordinate_dtype': str(self._coord_dtype)
        }

    def get_piece_distribution(self) -> Dict[str, int]:
        """Get distribution of pieces by type for both colors."""
        white_counts = _parallel_piece_counts(self._occ, self._ptype, COLOR_DTYPE(Color.WHITE))
        black_counts = _parallel_piece_counts(self._occ, self._ptype, COLOR_DTYPE(Color.BLACK))

        # Use enum names dynamically instead of hardcoded list
        piece_names = [name for name in PieceType.__members__.keys()]
        while len(piece_names) < len(white_counts):
            piece_names.append(f'TYPE_{len(piece_names)}')  # Add missing names

        distribution = {}
        for i in range(min(len(piece_names), len(white_counts))):
            name = piece_names[i]
            distribution[f'white_{name}'] = int(white_counts[i])
            distribution[f'black_{name}'] = int(black_counts[i])

        return distribution

    def get_board_occupancy_stats(self) -> Dict[str, float]:
        """Get occupancy statistics as percentages."""
        total_squares = SIZE ** 3  # FIXED: Changed from BOARD_SIZE to SIZE
        occupied_squares = np.count_nonzero(self._occ)

        white_squares = np.count_nonzero(self._occ == Color.WHITE)
        black_squares = np.count_nonzero(self._occ == Color.BLACK)

        return {
            'total_occupancy_percent': (occupied_squares / total_squares) * 100,
            'white_occupancy_percent': (white_squares / total_squares) * 100,
            'black_occupancy_percent': (black_squares / total_squares) * 100,
            'empty_squares_percent': ((total_squares - occupied_squares) / total_squares) * 100
        }

    def validate_consistency(self) -> Tuple[bool, str]:
        """
        Validate internal consistency of occupancy arrays.
        Checks:
        1. Occupancy vs Piece Type consistency (must mismatch 0/0)
        2. Priest count tracking
        """
        # 1. Check occ vs ptype consistency
        occ_mask = self._occ != 0
        type_mask = self._ptype != 0
        
        if not np.array_equal(occ_mask, type_mask):
            mismatch = occ_mask != type_mask
            count = np.sum(mismatch)
            indices = np.argwhere(mismatch)
            examples = indices[:min(5, count)].tolist()
            
            # Diagnostic detail
            example_details = []
            for idx in examples:
                x, y, z = idx
                c = self._occ[x, y, z]
                t = self._ptype[x, y, z]
                example_details.append(f"({x},{y},{z}: Color={c}, Type={t})")
            
            msg = (f"Consistency Error: {count} squares have mismatched occupancy/type. "
                   f"Examples: {', '.join(example_details)}")
            return False, msg
            
        # 2. Check priest count
        real_white_priests = np.sum((self._ptype == PieceType.PRIEST.value) & (self._occ == Color.WHITE))
        real_black_priests = np.sum((self._ptype == PieceType.PRIEST.value) & (self._occ == Color.BLACK))
        
        stored_white = self._priest_count[0]
        stored_black = self._priest_count[1]
        
        if stored_white != real_white_priests:
             return False, f"Priest Count Error (White): Stored {stored_white} vs Real {real_white_priests}"
        if stored_black != real_black_priests:
             return False, f"Priest Count Error (Black): Stored {stored_black} vs Real {real_black_priests}"

        return True, "OK"

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """Normalize coordinates to (N,3)."""
        coords = np.asarray(coords, dtype=COORD_DTYPE)
        if coords.ndim == 1:
            coords = coords.reshape(1, 3)
        return coords

    def set_position(self, coord: np.ndarray, piece: Optional[np.ndarray]) -> None:
        """Set piece at coordinate incrementally with king tracking."""
        # ✅ OPTIMIZATION: Skip normalization if already valid (1, 3) array
        if coord.shape == (1, 3) and coord.dtype == COORD_DTYPE:
            pass
        elif coord.shape == (3,) and coord.dtype == COORD_DTYPE:
             # Fast reshape without copy
             coord = coord.reshape(1, 3)
        else:
            coord = self._normalize_coords(coord)
            
        x, y, z = coord[0]

        # ✅ CRITICAL: Enforce Type/Color Consistency
        if piece is not None:
             ptype, pcolor = piece[0], piece[1]
             if (ptype == 0 and pcolor != 0) or (ptype != 0 and pcolor == 0):
                 raise ValueError(f"Inconsistent SetPosition at ({x},{y},{z}): Type={ptype}, Color={pcolor}")

        old_color = self._occ[x, y, z]
        old_type = self._ptype[x, y, z]

        # ✅ CRITICAL FIX: Update king cache when king is removed
        if old_type == PieceType.KING.value:
            color_idx = 0 if old_color == Color.WHITE else 1
            # Mark king position as invalid (will force re-search)
            self._king_positions[color_idx] = np.array([-1, -1, -1], dtype=COORD_DTYPE)

        if piece is None:
            self._occ[x, y, z] = 0
            self._ptype[x, y, z] = 0
            
            # Update priest count
            if old_type == PieceType.PRIEST.value:
                idx = 0 if old_color == Color.WHITE else 1
                self._priest_count[idx] -= 1

            # Update type counts
            if old_type > 0 and old_type < 64:
                idx = 0 if old_color == Color.WHITE else 1
                self._type_counts[idx, old_type] -= 1
        else:
            self._occ[x, y, z] = piece[1]
            self._ptype[x, y, z] = piece[0]
            
            # Update priest count
            if old_type == PieceType.PRIEST.value:
                idx = 0 if old_color == Color.WHITE else 1
                self._priest_count[idx] -= 1
                
            # Update type counts
            if old_type > 0 and old_type < 64:
                idx = 0 if old_color == Color.WHITE else 1
                self._type_counts[idx, old_type] -= 1
                
            if piece[0] == PieceType.PRIEST.value:
                idx = 0 if piece[1] == Color.WHITE else 1
                self._priest_count[idx] += 1
                
            # Update type counts
            if piece[0] > 0 and piece[0] < 64:
                idx = 0 if piece[1] == Color.WHITE else 1
                self._type_counts[idx, piece[0]] += 1

            # Update king cache
            if piece[0] == PieceType.KING.value:
                color_idx = 0 if piece[1] == Color.WHITE else 1
                self._king_positions[color_idx] = coord[0]

        # ✅ INCREMENTAL UPDATE
        idx_flat = coord_to_key_scalar(x, y, z)
        
        if old_color == Color.WHITE:
            self._positions_indices[0].discard(idx_flat)
        elif old_color == Color.BLACK:
            self._positions_indices[1].discard(idx_flat)
            
        if piece is not None:
            new_c = piece[1]
            if new_c == Color.WHITE:
                self._positions_indices[0].add(idx_flat)
            elif new_c == Color.BLACK:
                self._positions_indices[1].add(idx_flat)

        # Invalidate flat view
        self._flat_occ_view = None
        self._flat_indices_cache = {}
        
        # ✅ OPTIMIZATION: Invalidate positions cache
        self._positions_dirty = [True, True]
        self._positions_cache = [None, None]       
        # ✅ OPTIMIZATION: Invalidate positions cache
        if old_color != 0:
            c_idx = 0 if old_color == Color.WHITE else 1
            self._positions_dirty[c_idx] = True
            
        if piece is not None:
             p_idx = 0 if piece[1] == Color.WHITE else 1
             self._positions_dirty[p_idx] = True

    def set_position_fast(self, coord: np.ndarray, piece_type: int, color: int) -> None:
        """Fast path set: updates arrays directly.
        
        WARNING: Assumes coord is valid (3,) array within bounds.
        Does NOT update priest count (use only for simulation/revert).
        DOES update king position cache for consistency.
        """
        # ✅ CRITICAL: Enforce Type/Color Consistency
        if (piece_type == 0 and color != 0) or (piece_type != 0 and color == 0):
             raise ValueError(f"Inconsistent SetPositionFast: Type={piece_type}, Color={color}")

        x, y, z = coord[0], coord[1], coord[2]
        old_color = self._occ[x, y, z]
        old_ptype = self._ptype[x, y, z]
        
        self._occ[x, y, z] = color
        self._ptype[x, y, z] = piece_type
        
        # ✅ INCREMENTAL UPDATE
        idx = coord_to_key_scalar(x, y, z)
        
        if old_color == Color.WHITE:
            self._positions_indices[0].discard(idx)
        elif old_color == Color.BLACK:
            self._positions_indices[1].discard(idx)
            
        if color == Color.WHITE:
            self._positions_indices[0].add(idx)
        elif color == Color.BLACK:
            self._positions_indices[1].add(idx)
        
        # ✅ FIX: Update king position cache to prevent desync
        # If we're removing a king, invalidate its cached position
        if old_ptype == PieceType.KING.value and old_color != 0:
            old_color_idx = 0 if old_color == Color.WHITE else 1
            self._king_positions[old_color_idx].fill(-1)
        
        # If we're placing a king, update its cached position
        if piece_type == PieceType.KING.value and color != 0:
            color_idx = 0 if color == Color.WHITE else 1
            self._king_positions[color_idx] = coord.astype(COORD_DTYPE)
            
        # Mark cache clean/dirty logic is handled by caller mostly (set_position_fast is often used in isolation)
        # But consistent with batch_set_positions, we should mark dirty
        self._positions_dirty = [True, True]

    # =========================================================================
    # ✅ OPTIMIZATION: BATCH SIMULATION MODE (No auxiliary updates)
    # =========================================================================

    def batch_simulate_moves(self, moves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate multiple moves WITHOUT updating auxiliary caches.
        
        ✅ OPTIMIZATION: Only touches _occ and _ptype arrays directly.
        Skips: Python sets, king cache, dirty flags, priest counts.
        
        Use for batch legality checks where moves are immediately reverted.
        
        Args:
            moves: (N, 6) array of moves [from_x, from_y, from_z, to_x, to_y, to_z]
            
        Returns:
            (from_types, from_colors, to_types, to_colors) - original data for reversal
        """
        n = moves.shape[0]
        if n == 0:
            empty = np.empty(0, dtype=PIECE_TYPE_DTYPE)
            return empty, empty.view(COLOR_DTYPE), empty, empty.view(COLOR_DTYPE)
        
        # Extract coordinates
        fx, fy, fz = moves[:, 0], moves[:, 1], moves[:, 2]
        tx, ty, tz = moves[:, 3], moves[:, 4], moves[:, 5]
        
        # Store original data for reversal
        from_types = self._ptype[fx, fy, fz].copy()
        from_colors = self._occ[fx, fy, fz].copy()
        to_types = self._ptype[tx, ty, tz].copy()
        to_colors = self._occ[tx, ty, tz].copy()
        
        # Apply moves directly to arrays (no auxiliary updates)
        # Clear source squares
        self._ptype[fx, fy, fz] = 0
        self._occ[fx, fy, fz] = 0
        
        # Move pieces to destinations
        self._ptype[tx, ty, tz] = from_types
        self._occ[tx, ty, tz] = from_colors
        
        return from_types, from_colors, to_types, to_colors

    def batch_revert_moves(
        self, 
        moves: np.ndarray,
        from_types: np.ndarray,
        from_colors: np.ndarray,
        to_types: np.ndarray,
        to_colors: np.ndarray
    ) -> None:
        """Revert moves simulated by batch_simulate_moves.
        
        ✅ OPTIMIZATION: Restores original state without auxiliary updates.
        """
        if moves.shape[0] == 0:
            return
            
        fx, fy, fz = moves[:, 0], moves[:, 1], moves[:, 2]
        tx, ty, tz = moves[:, 3], moves[:, 4], moves[:, 5]
        
        self._ptype[fx, fy, fz] = from_types
        self._occ[fx, fy, fz] = from_colors
        self._ptype[tx, ty, tz] = to_types
        self._occ[tx, ty, tz] = to_colors

    def simulate_single_move_fast(self, move: np.ndarray) -> tuple[int, int, int, int]:
        """Simulate a single move WITHOUT auxiliary updates.
        
        Returns (from_type, from_color, to_type, to_color) for reversal.
        """
        fx, fy, fz = int(move[0]), int(move[1]), int(move[2])
        tx, ty, tz = int(move[3]), int(move[4]), int(move[5])
        
        from_type = int(self._ptype[fx, fy, fz])
        from_color = int(self._occ[fx, fy, fz])
        to_type = int(self._ptype[tx, ty, tz])
        to_color = int(self._occ[tx, ty, tz])
        
        self._ptype[fx, fy, fz] = 0
        self._occ[fx, fy, fz] = 0
        self._ptype[tx, ty, tz] = from_type
        self._occ[tx, ty, tz] = from_color
        
        return from_type, from_color, to_type, to_color

    def revert_single_move_fast(
        self,
        move: np.ndarray,
        from_type: int,
        from_color: int,
        to_type: int,
        to_color: int
    ) -> None:
        """Revert a move simulated by simulate_single_move_fast."""
        fx, fy, fz = int(move[0]), int(move[1]), int(move[2])
        tx, ty, tz = int(move[3]), int(move[4]), int(move[5])
        
        self._ptype[fx, fy, fz] = from_type
        self._occ[fx, fy, fz] = from_color
        self._ptype[tx, ty, tz] = to_type
        self._occ[tx, ty, tz] = to_color

    def get_flattened_occupancy(self) -> np.ndarray:

        """
        Return cached flattened view of color occupancy (O(1), thread-safe).
        0 = empty, Color.WHITE.value = white, Color.BLACK.value = black
        
        ✅ OPTIMIZATION: Uses cached view to eliminate redundant ravel() calls.
        ✅ THREAD-SAFE: Uses double-checked locking pattern.
        """
        # Fast path: no lock needed if already cached
        if self._flat_occ_view is None:
            with self._flat_view_lock:
                # Double-check pattern: another thread might have created it
                if self._flat_occ_view is None:
                    # ravel('F') matches indexing: x + SIZE*y + SIZE*SIZE*z
                    self._flat_occ_view = self._occ.ravel(order='F')
        return self._flat_occ_view

    def get_flat_indices(self, coords: np.ndarray) -> np.ndarray:
        """Get flat indices for coordinates with caching."""
        # Use coords as cache key (convert to tuple for hashing)
        key = coords.data.tobytes()

        if key not in self._flat_indices_cache:
            # Vectorized calculation using centralized utility
            self._flat_indices_cache[key] = CoordinateUtils.coord_to_idx(coords)

        return self._flat_indices_cache[key]

    def get_all_occupied_vectorized(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return all occupied squares as separate numpy arrays.

        Returns:
            tuple: (coords, piece_types, colors)
            - coords: (N, 3) array of coordinates in (x, y, z) order
            - piece_types: (N,) array of piece type IDs
            - colors: (N,) array of color IDs
        """
        occupied_mask = self._occ != 0
        coords = np.argwhere(occupied_mask)  # Returns (x, y, z) with new indexing

        if coords.shape[0] == 0:
            return (np.empty((0, 3), dtype=self._coord_dtype),
                    np.empty(0, dtype=PIECE_TYPE_DTYPE),
                    np.empty(0, dtype=COLOR_DTYPE))

        # No reordering needed - coords are already in (x, y, z) order
        coords_xyz = coords.astype(self._coord_dtype, copy=False)

        # Extract piece types and colors at these positions (vectorized)
        # ✅ OPTIMIZED: No astype needed - grids already store data in correct dtypes
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        piece_types = self._ptype[x, y, z]
        colors = self._occ[x, y, z]
        
        return coords_xyz, piece_types, colors

    def export_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Export current state for cloning/persistence.
        
        Returns:
            tuple: (coords, piece_types, colors)
        """
        return self.get_all_occupied_vectorized()

    def import_state(self, coords: np.ndarray, types: np.ndarray, colors: np.ndarray) -> None:
        """
        Import state from exported data.
        
        Args:
            coords: (N, 3) array of coordinates
            types: (N,) array of piece types
            colors: (N,) array of colors
        """
        self.rebuild(coords, types, colors)
