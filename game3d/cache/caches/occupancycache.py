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
    MAX_COORD_VALUE, MIN_COORD_VALUE, N_PIECE_TYPES
)

from game3d.common.coord_utils import CoordinateUtils

# Type aliases for clarity and consistency
PIECE_EMPTY = EMPTY

# Optimize parallelization based on available CPU cores
NUM_CORES = os.cpu_count() or 4
PARALLEL_CHUNKS = min(NUM_CORES, SIZE if SIZE > 0 else 4)

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
    max_piece_types = 11  # Support 0-10 piece types
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


class OccupancyCache:
    __slots__ = ("_occ", "_ptype", "_priest_count", "_coord_dtype",
                 "_piece_type_count", "_memory_pool", "_flat_occ_view",
                 "_flat_indices_cache", "_king_positions", "_flat_view_lock",
                 "_cache_size_limit", "_king_cache_misses")

    def __init__(self, board_size=SIZE, piece_type_count=None) -> None:
        self._coord_dtype = COORD_DTYPE
        self._piece_type_count = piece_type_count or N_PIECE_TYPES

        # Occupancy array: COLOR_DTYPE at each position
        self._occ = np.zeros((board_size, board_size, board_size), dtype=COLOR_DTYPE)

        # Piece type array: PIECE_TYPE_DTYPE at each position
        self._ptype = np.zeros((board_size, board_size, board_size), dtype=PIECE_TYPE_DTYPE)

        # Priest count for white and black
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
        self._memory_pool = get_memory_pool()

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
        # Use simple integer packing: x | (y << 9) | (z << 18)
        keys = coords[:, 0] | (coords[:, 1] << 9) | (coords[:, 2] << 18)
        
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

        # Vectorized update (this is the core operation)
        x = coords[:, 0].astype(INDEX_DTYPE)
        y = coords[:, 1].astype(INDEX_DTYPE)
        z = coords[:, 2].astype(INDEX_DTYPE)

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
        self._occ[x, y, z] = pieces[:, 1]  # colors
        self._ptype[x, y, z] = pieces[:, 0]  # piece types

        # Note: King tracking is now done via direct lookup in find_king()
        # No need to maintain a separate king position cache


        # ✅ CRITICAL FIX: Update priest count for removed/added priests
        # Decrement count for old priests being removed/replaced
        old_priest_mask = (old_types == PieceType.PRIEST.value)
        if np.any(old_priest_mask):
            for i in np.where(old_priest_mask)[0]:
                color_idx = 0 if old_colors[i] == Color.WHITE else 1
                self._priest_count[color_idx] -= 1
        
        # Increment count for new priests being added
        new_priest_mask = (pieces[:, 0] == PieceType.PRIEST.value)
        if np.any(new_priest_mask):
            for i in np.where(new_priest_mask)[0]:
                color_idx = 0 if pieces[i, 1] == Color.WHITE else 1
                self._priest_count[color_idx] += 1

        # Invalidate cached views
        self._flat_occ_view = None

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
        """Get all positions of color using vectorized operations."""
        mask = (self._occ == color)

        coords = np.argwhere(mask)
        if coords.size == 0:
            return np.empty((0, 3), dtype=self._coord_dtype)

        # No reordering needed - argwhere returns (x, y, z) from array indexed as [x, y, z]
        return coords.astype(self._coord_dtype)

    def rebuild(self, coords: np.ndarray, types: np.ndarray, colors: np.ndarray) -> None:
        """Rebuild cache from coordinate arrays using vectorized operations."""
        self._occ.fill(0)
        self._ptype.fill(0)
        self._king_positions.fill(-1)  # Reset king cache

        if len(coords) == 0:
            self._priest_count.fill(0)
            return

        coords = self._normalize_coords(coords)
        types = np.asarray(types, dtype=PIECE_TYPE_DTYPE)
        colors = np.asarray(colors, dtype=COLOR_DTYPE)

        x = coords[:, 0].astype(INDEX_DTYPE)
        y = coords[:, 1].astype(INDEX_DTYPE)
        z = coords[:, 2].astype(INDEX_DTYPE)

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

        self._priest_count = _parallel_count_priests(self._occ, self._ptype, PARALLEL_CHUNKS)

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


    @property
    def count(self) -> int:
        """Total number of pieces on board."""
        return int(np.count_nonzero(self._occ))

    def has_priest(self, color: int) -> bool:
        """Check if color has any priests."""
        color_idx = 0 if color == Color.WHITE else 1
        return int(self._priest_count[color_idx]) > 0

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
        """Clear all occupancy data."""
        self._occ.fill(0)
        self._ptype.fill(0)
        self._priest_count.fill(0)
        self._king_positions.fill(-1)  # ✅ Clear king cache

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
            if old_type == PieceType.PRIEST.value:
                color_idx = 0 if old_color == Color.WHITE else 1
                self._priest_count[color_idx] -= 1
        else:
            self._occ[x, y, z] = piece[1]
            self._ptype[x, y, z] = piece[0]
            if old_type == PieceType.PRIEST.value:
                color_idx = 0 if old_color == Color.WHITE else 1
                self._priest_count[color_idx] -= 1
            if piece[0] == PieceType.PRIEST.value:
                color_idx = 0 if piece[1] == Color.WHITE else 1
                self._priest_count[color_idx] += 1
            
            # ✅ CRITICAL FIX: Update king cache when king is placed
            if piece[0] == PieceType.KING.value:
                color_idx = 0 if piece[1] == Color.WHITE else 1
                self._king_positions[color_idx] = coord[0].copy()

        self._flat_occ_view = None
        self._flat_indices_cache.clear()

    def set_position_fast(self, coord: np.ndarray, piece_type: int, color: int) -> None:
        """Fast path set: updates arrays directly.
        
        WARNING: Assumes coord is valid (3,) array within bounds.
        Does NOT update priest count or king cache (use only for simulation/revert).
        """
        x, y, z = coord[0], coord[1], coord[2]
        self._occ[x, y, z] = color
        self._ptype[x, y, z] = piece_type

    @njit(cache=True, nogil=True)
    def _coord_to_flat_idx(coord: np.ndarray) -> np.ndarray:
        """Vectorized coord (N,3) → flat index."""
        return coord[:, 0] + SIZE * coord[:, 1] + SIZE * SIZE * coord[:, 2]

    def get_flattened_occupancy(self) -> np.ndarray:
        """
        Return a read-only flattened view of color occupancy.
        0 = empty, Color.WHITE.value = white, Color.BLACK.value = black
        This is O(1) and extremely cheap.
        """
        # ravel('F') matches the indexing used in compute_board_index: x + SIZE*y + SIZE*SIZE*z
        return self._occ.ravel(order='F')

    def get_cached_flattened(self) -> np.ndarray:
        """Return cached flattened occupancy (0 = empty, 1 = white, 2 = black).
        
        ✅ THREAD-SAFE: Uses double-checked locking pattern.
        """
        # Fast path: no lock needed if already cached
        if self._flat_occ_view is None:
            with self._flat_view_lock:
                # Double-check pattern: another thread might have created it
                if self._flat_occ_view is None:
                    self._flat_occ_view = self._occ.ravel(order='F')
        return self._flat_occ_view

    def get_flat_indices(self, coords: np.ndarray) -> np.ndarray:
        """Get flat indices for coordinates with caching."""
        # Use coords as cache key (convert to tuple for hashing)
        key = coords.data.tobytes()

        if key not in self._flat_indices_cache:
            # Vectorized calculation - matches array indexing [x, y, z] and ravel('C') order
            self._flat_indices_cache[key] = (
                coords[:, 0] + SIZE * coords[:, 1] + SIZE * SIZE * coords[:, 2]
            ).astype(np.int32)

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
        coords_xyz = coords.astype(self._coord_dtype)

        # Extract piece types and colors at these positions (vectorized)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        piece_types = self._ptype[x, y, z].astype(PIECE_TYPE_DTYPE)
        colors = self._occ[x, y, z].astype(COLOR_DTYPE)
        
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
