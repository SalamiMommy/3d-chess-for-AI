# trailblazecache_optimized.py – FULLY NUMPY-NATIVE IMPLEMENTATION
from __future__ import annotations
import numpy as np
from numba import njit, prange
from typing import Optional, Dict, Any, List
import threading

from game3d.common.shared_types import (
    COORD_DTYPE, INDEX_DTYPE, BOOL_DTYPE, SIZE, VOLUME, MAX_COORD_VALUE,
    Color, TRAILBLAZER, INT8_DTYPE
)
from game3d.common.coord_utils import coord_to_idx, idx_to_coord, in_bounds_vectorized
from game3d.cache.cache_protocols import CacheListener

# =============================================================================
# STRUCTURED DTYPES FOR NUMPY-NATIVE STORAGE
# =============================================================================

# Maximum trail length per trailblazer (3 moves * 3 squares max = 9 squares)
# But we need to store the history of moves.
# Let's store up to 3 past moves. Each move has a path.
# Max path length for a move is MAX_TRAILBLAZER_DISTANCE (3).
# So we need storage for 3 * 3 = 9 squares total.
MAX_TRAIL_HISTORY = 3
MAX_SQUARES_PER_MOVE = 3
TOTAL_TRAIL_CAPACITY = MAX_TRAIL_HISTORY * MAX_SQUARES_PER_MOVE

# Trail data storage: One entry per potential trailblazer position
TRAIL_DATA_DTYPE = np.dtype([
    ('flat_idx', INDEX_DTYPE),           # Trailblazer position (flat index)
    ('trail_coords', COORD_DTYPE, (TOTAL_TRAIL_CAPACITY, 3)),  # Pre-allocated trail coordinates
    ('trail_lengths', INDEX_DTYPE, (MAX_TRAIL_HISTORY,)), # Length of each history segment
    ('history_ptr', INDEX_DTYPE),        # Circular buffer pointer
    ('active', BOOL_DTYPE)               # Whether this entry is active
])

# =============================================================================
# NUMBA-OPTIMIZED CORE OPERATIONS
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def _batch_trailblazer_check_numba(
    coords: np.ndarray,
    trailblazer_positions: np.ndarray
) -> np.ndarray:
    """Vectorized trailblazer membership check using flat indices."""
    if trailblazer_positions.size == 0 or coords.size == 0:
        return np.zeros(coords.shape[0], dtype=BOOL_DTYPE)

    # Convert all coordinates to flat indices for O(1) comparison
    n = coords.shape[0]
    result = np.zeros(n, dtype=BOOL_DTYPE)
    flat_coords = np.empty(n, dtype=INDEX_DTYPE)

    for i in prange(n):
        x, y, z = coords[i]
        flat_coords[i] = x + SIZE * y + SIZE * (SIZE * z)

    # Vectorized membership test
    for i in prange(n):
        result[i] = np.any(trailblazer_positions == flat_coords[i])

    return result

@njit(cache=True, fastmath=True, parallel=True)
def _batch_trail_squares_check_numba(
    trail_coords: np.ndarray,
    query_coords: np.ndarray
) -> np.ndarray:
    """Vectorized trail membership check."""
    if trail_coords.size == 0 or query_coords.size == 0:
        return np.zeros(query_coords.shape[0], dtype=BOOL_DTYPE)

    n_queries = query_coords.shape[0]
    result = np.zeros(n_queries, dtype=BOOL_DTYPE)

    for i in prange(n_queries):
        # Check if query coord matches any trail coord
        diff = trail_coords - query_coords[i]
        result[i] = np.any((diff[:, 0] == 0) & (diff[:, 1] == 0) & (diff[:, 2] == 0))

    return result

@njit(cache=True, nogil=True)
def _coords_to_flat_scalar(x: int, y: int, z: int) -> int:
    """Scalar coordinate to flat index conversion."""
    return x + SIZE * y + SIZE * (SIZE * z)

# =============================================================================
# OPTIMIZED TRAILBLAZE CACHE
# =============================================================================

class TrailblazeCache(CacheListener):
    """Fully numpy-native trailblaze cache with zero Python loops."""

    __slots__ = (
        "_manager", "_coord_dtype", "_memory_pool",
        "_trail_data", "_trailblazer_flat_positions", "_position_mask",
        "_victim_counters", "_lock", "_next_slot", "_max_trailblazers"
    )

    def __init__(self, cache_manager: Optional["OptimizedCacheManager"] = None) -> None:
        self._manager = cache_manager
        self._coord_dtype = COORD_DTYPE
        self._memory_pool = getattr(cache_manager, '_memory_pool', None)

        # Pre-allocate array for up to 100 trailblazers (adjust as needed)
        self._max_trailblazers = 100

        # Structured array: One entry per trailblazer
        self._trail_data = np.zeros(self._max_trailblazers, dtype=TRAIL_DATA_DTYPE)
        self._trail_data['active'] = False  # All inactive initially

        # Fast lookup: flat indices of active trailblazer positions
        self._trailblazer_flat_positions = np.empty(0, dtype=INDEX_DTYPE)

        # Boolean mask for O(1) position checks: shape (VOLUME,)
        self._position_mask = np.zeros(VOLUME, dtype=BOOL_DTYPE)
        
        # Victim counters: shape (VOLUME,) - stores counter value (0-3) per square
        self._victim_counters = np.zeros(VOLUME, dtype=INT8_DTYPE)

        # Thread safety
        self._lock = threading.RLock()

        # Next available slot index
        self._next_slot = 0

    # ========================================================================
    # CACHE LISTENER INTERFACE - VECTORIZED
    # ========================================================================

    def get_priority(self) -> int:
        """Return priority for update order (lower = higher priority)."""
        return 3  # Medium priority for trailblazer effects

    def on_occupancy_changed(self, changed_coords: np.ndarray, pieces: np.ndarray) -> None:
        """Handle single coordinate change - delegated to batch handler."""
        if changed_coords.size == 0:
            return

        # Reshape single coord to batch
        coords_2d = changed_coords.reshape(-1, 3)
        self.on_batch_occupancy_changed(coords_2d, pieces)

    def on_batch_occupancy_changed(self, coords: np.ndarray, pieces: np.ndarray) -> None:
        """Batch occupancy change handler - fully vectorized."""
        if coords.size == 0:
            return

        with self._lock:
            # Ensure proper shape
            coords = self._ensure_coords(coords)

            # Convert to flat indices for fast lookup
            flat_coords = self._coords_to_flat(coords)

            # Check which coords are trailblazers
            is_trailblazer_mask = self._is_trailblazer_flat(flat_coords)

            # Check which are being removed (piece type 0 or empty)
            removed_mask = is_trailblazer_mask & (pieces[:, 0] == 0)

            # Batch remove trails for removed trailblazers
            if np.any(removed_mask):
                self._remove_trail_batch(flat_coords[removed_mask])

    # ========================================================================
    # BATCH QUERY OPERATIONS
    # ========================================================================

    def batch_is_trailblazer(self, coords: np.ndarray) -> np.ndarray:
        """Vectorized trailblazer check."""
        coords = self._ensure_coords(coords)
        flat_coords = self._coords_to_flat(coords)
        return self._is_trailblazer_flat(flat_coords)

    def check_trail_intersection(self, path_coords: np.ndarray) -> bool:
        """Check if any coordinate in path intersects with ANY active trail."""
        if path_coords.size == 0:
            return False
            
        # Get all active trail squares
        all_trails = self._get_all_active_trail_squares()
        if all_trails.size == 0:
            return False
            
        return np.any(_batch_trail_squares_check_numba(all_trails, path_coords))

    def get_intersecting_squares(self, path_coords: np.ndarray) -> np.ndarray:
        """Get subset of path_coords that intersect with trails."""
        if path_coords.size == 0:
            return np.empty((0, 3), dtype=self._coord_dtype)
            
        all_trails = self._get_all_active_trail_squares()
        if all_trails.size == 0:
            return np.empty((0, 3), dtype=self._coord_dtype)
            
        mask = _batch_trail_squares_check_numba(all_trails, path_coords)
        return path_coords[mask]

    # ========================================================================
    # COUNTER OPERATIONS
    # ========================================================================

    def increment_counter(self, coord: np.ndarray) -> bool:
        """
        Increment counter at coordinate.
        Returns True if counter reaches 3 (capture condition).
        """
        coords = self._ensure_coords(coord)
        flat_idx = self._coords_to_flat(coords)[0]
        
        with self._lock:
            self._victim_counters[flat_idx] += 1
            return self._victim_counters[flat_idx] >= 3

    def get_counter(self, coord: np.ndarray) -> int:
        """Get counter value at coordinate."""
        coords = self._ensure_coords(coord)
        flat_idx = self._coords_to_flat(coords)[0]
        return int(self._victim_counters[flat_idx])

    def clear_counter(self, coord: np.ndarray) -> None:
        """Reset counter at coordinate."""
        coords = self._ensure_coords(coord)
        flat_idx = self._coords_to_flat(coords)[0]
        with self._lock:
            self._victim_counters[flat_idx] = 0

    def move_counter(self, from_coord: np.ndarray, to_coord: np.ndarray) -> None:
        """Move counter value from one square to another (piece moved)."""
        from_c = self._ensure_coords(from_coord)
        to_c = self._ensure_coords(to_coord)
        
        from_idx = self._coords_to_flat(from_c)[0]
        to_idx = self._coords_to_flat(to_c)[0]
        
        with self._lock:
            val = self._victim_counters[from_idx]
            if val > 0:
                self._victim_counters[to_idx] = val
                self._victim_counters[from_idx] = 0

    # ========================================================================
    # TRAIL MANAGEMENT
    # ========================================================================

    def add_trail(self, trailblazer_pos: np.ndarray, squares: np.ndarray) -> None:
        """Add trail for a specific trailblazer at trailblazer_pos."""
        if squares.size == 0:
            return
            
        with self._lock:
            coords = self._ensure_coords(trailblazer_pos)
            flat_idx = self._coords_to_flat(coords)[0]
            
            # Check if we already have an entry for this trailblazer
            mask = self._trail_data['flat_idx'] == flat_idx
            if np.any(mask):
                entry_idx = np.where(mask)[0][0]
            else:
                # Find free slot
                if self._next_slot >= self._max_trailblazers:
                    # Try to find inactive slot
                    inactive = np.where(~self._trail_data['active'])[0]
                    if inactive.size > 0:
                        entry_idx = inactive[0]
                    else:
                        # Cache full - ignore
                        return
                else:
                    entry_idx = self._next_slot
                    self._next_slot += 1
            
            # Activate entry
            self._trail_data[entry_idx]['flat_idx'] = flat_idx
            self._trail_data[entry_idx]['active'] = True
            
            # Manage circular buffer history
            ptr = self._trail_data[entry_idx]['history_ptr']
            
            # Store new trail segment
            n_squares = min(squares.shape[0], MAX_SQUARES_PER_MOVE)
            start_idx = ptr * MAX_SQUARES_PER_MOVE
            
            # Clear old data in this segment
            self._trail_data[entry_idx]['trail_coords'][start_idx:start_idx+MAX_SQUARES_PER_MOVE] = 0
            
            # Write new data
            self._trail_data[entry_idx]['trail_coords'][start_idx:start_idx+n_squares] = squares[:n_squares]
            self._trail_data[entry_idx]['trail_lengths'][ptr] = n_squares
            
            # Advance pointer
            self._trail_data[entry_idx]['history_ptr'] = (ptr + 1) % MAX_TRAIL_HISTORY
            
            self._rebuild_fast_lookups()

    def _get_all_active_trail_squares(self) -> np.ndarray:
        """Get all active trail squares across all trailblazers."""
        active_mask = self._trail_data['active']
        if not np.any(active_mask):
            return np.empty((0, 3), dtype=self._coord_dtype)
            
        all_squares = []
        active_indices = np.where(active_mask)[0]
        
        for idx in active_indices:
            lengths = self._trail_data[idx]['trail_lengths']
            coords = self._trail_data[idx]['trail_coords']
            
            for i in range(MAX_TRAIL_HISTORY):
                l = lengths[i]
                if l > 0:
                    start = i * MAX_SQUARES_PER_MOVE
                    all_squares.append(coords[start:start+l])
                    
        if not all_squares:
            return np.empty((0, 3), dtype=self._coord_dtype)
            
        return np.vstack(all_squares)

    # ========================================================================
    # INTERNAL BATCH OPERATIONS
    # ========================================================================

    def _remove_trail_batch(self, flat_indices: np.ndarray) -> int:
        """Batch trail removal - fully vectorized."""
        if flat_indices.size == 0:
            return 0

        removed_count = 0

        # Process each index to remove
        for idx in flat_indices:
            # Find matching entries
            mask = self._trail_data['flat_idx'] == idx
            if not np.any(mask):
                continue

            entry_idx = np.where(mask)[0][0]

            if self._trail_data[entry_idx]['active']:
                # Mark as inactive
                self._trail_data[entry_idx]['active'] = False
                self._trail_data[entry_idx]['trail_lengths'][:] = 0
                removed_count += 1

        # Rebuild fast lookup structures
        self._rebuild_fast_lookups()

        return removed_count

    def _rebuild_fast_lookups(self) -> None:
        """Rebuild vectorized lookup structures."""
        active_mask = self._trail_data['active']
        active_count = np.sum(active_mask)

        # Rebuild flat positions array
        self._trailblazer_flat_positions = self._trail_data['flat_idx'][active_mask].copy()

        # Rebuild position mask
        self._position_mask.fill(False)
        if active_count > 0:
            self._position_mask[self._trailblazer_flat_positions] = True

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def clear(self) -> None:
        """Clear all trail data."""
        with self._lock:
            self._trail_data['active'] = False
            self._trail_data['trail_lengths'][:] = 0
            self._trailblazer_flat_positions = np.empty(0, dtype=INDEX_DTYPE)
            self._position_mask.fill(False)
            self._victim_counters.fill(0)
            self._next_slot = 0

    def get_trailblazer_positions(self) -> np.ndarray:
        """Get all trailblazer coordinates."""
        with self._lock:
            if self._trailblazer_flat_positions.size == 0:
                return np.empty((0, 3), dtype=self._coord_dtype)

            # Convert flat indices back to coordinates
            return idx_to_coord(self._trailblazer_flat_positions)

    def get_trail_mask(self) -> np.ndarray:
        """Get boolean mask of trailblazer positions."""
        # Return 3D mask view of the 1D position mask
        return self._position_mask.reshape(SIZE, SIZE, SIZE)

    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        with self._lock:
            active_count = np.sum(self._trail_data['active'])
            total_trail_coords = np.sum(self._trail_data['trail_lengths'])

            return {
                'trailblazer_count': int(active_count),
                'total_trail_coords': int(total_trail_coords),
                'position_cache_size': len(self._trailblazer_flat_positions) * self._trailblazer_flat_positions.itemsize,
                'mask_size': self._position_mask.nbytes,
                'trail_data_size': self._trail_data.nbytes,
                'counters_size': self._victim_counters.nbytes
            }

    # ========================================================================
    # MOVE INTERFACE (REQUIRED BY CACHE LISTENER)
    # ========================================================================

    def apply_move(self, mv, mover, current_ply, board) -> None:
        """Trails are unaffected by piece movement."""
        pass

    def undo_move(self, mv, mover, current_ply, board) -> None:
        """Trails are unaffected by undo."""
        pass

    def current_trail_squares(self, controller, board) -> np.ndarray:
        """Get all trail squares for controller."""
        # This seems to be a legacy method or one used by other parts.
        # We return all active trails.
        return self._get_all_active_trail_squares()

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _ensure_coords(self, coords: np.ndarray) -> np.ndarray:
        """Ensure coordinates are (N, 3) numpy arrays."""
        coords = np.asarray(coords, dtype=self._coord_dtype)

        if coords.ndim == 1 and coords.shape[0] == 3:
            return coords.reshape(1, 3)
        elif coords.ndim == 2 and coords.shape[1] == 3:
            return coords
        else:
            raise ValueError(f"Invalid coordinate shape: {coords.shape}")

    def _coords_to_flat(self, coords: np.ndarray) -> np.ndarray:
        """Batch coordinate to flat index conversion."""
        # Use memory pool if available
        if self._memory_pool is not None and hasattr(self._memory_pool, 'allocate_array'):
            try:
                n = coords.shape[0]
                result = self._memory_pool.allocate_array((n,), INDEX_DTYPE)
                for i in range(n):
                    x, y, z = coords[i]
                    result[i] = _coords_to_flat_scalar(int(x), int(y), int(z))
                return result
            except (AttributeError, TypeError):
                pass

        # Fallback to direct calculation
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        return x + SIZE * y + SIZE * SIZE * z

    def _is_trailblazer_flat(self, flat_coords: np.ndarray) -> np.ndarray:
        """Check if flat indices are trailblazers."""
        if flat_coords.size == 0 or self._trailblazer_flat_positions.size == 0:
            return np.zeros(flat_coords.shape[0], dtype=BOOL_DTYPE)

        return np.isin(flat_coords, self._trailblazer_flat_positions)

__all__ = ['TrailblazeCache']
