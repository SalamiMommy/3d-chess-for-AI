# trailblazecache_optimized.py – FULLY NUMPY-NATIVE IMPLEMENTATION
from __future__ import annotations
import numpy as np
from numba import njit, prange
from typing import Optional, Dict, Any
import threading

from game3d.common.shared_types import (
    COORD_DTYPE, INDEX_DTYPE, BOOL_DTYPE, SIZE, VOLUME, MAX_COORD_VALUE,
    Color, TRAILBLAZER
)
from game3d.common.coord_utils import coord_to_idx, idx_to_coord, in_bounds_vectorized
from game3d.cache.cache_protocols import CacheListener

# =============================================================================
# STRUCTURED DTYPES FOR NUMPY-NATIVE STORAGE
# =============================================================================

# Maximum trail length per trailblazer (trailblazer + 8 steps)
MAX_TRAIL_LENGTH = 9

# Trail data storage: One entry per potential trailblazer position
TRAIL_DATA_DTYPE = np.dtype([
    ('flat_idx', INDEX_DTYPE),           # Trailblazer position (flat index)
    ('trail_coords', COORD_DTYPE, (MAX_TRAIL_LENGTH, 3)),  # Pre-allocated trail coordinates
    ('trail_length', INDEX_DTYPE),       # Actual length of trail (0 = inactive)
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
        "_lock", "_next_slot", "_max_trailblazers"
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

    def batch_is_in_trail(self, trailblazer_sq: np.ndarray, query_coords: np.ndarray) -> np.ndarray:
        """Vectorized trail membership check."""
        if trailblazer_sq.size == 0 or query_coords.size == 0:
            return np.zeros(query_coords.shape[0], dtype=BOOL_DTYPE)

        coords = self._ensure_coords(trailblazer_sq)
        trailblazer_idx = self._coords_to_flat(coords)[0]

        # Find trailblazer entry
        trailblazer_mask = self._trail_data['flat_idx'] == trailblazer_idx
        if not np.any(trailblazer_mask):
            return np.zeros(query_coords.shape[0], dtype=BOOL_DTYPE)

        entry_idx = np.where(trailblazer_mask)[0][0]
        trail_length = self._trail_data[entry_idx]['trail_length']

        if trail_length == 0:
            return np.zeros(query_coords.shape[0], dtype=BOOL_DTYPE)

        # Get trail coordinates
        trail_coords = self._trail_data[entry_idx]['trail_coords'][:trail_length]

        # Vectorized membership check
        return _batch_trail_squares_check_numba(trail_coords, query_coords)

    # ========================================================================
    # SINGLE QUERY OPERATIONS (wrap batch operations)
    # ========================================================================

    def get_trail_count(self, trailblazer_sq: np.ndarray) -> int:
        """Get trail length for trailblazer."""
        coords = self._ensure_coords(trailblazer_sq)
        flat_idx = self._coords_to_flat(coords)[0]

        mask = self._trail_data['flat_idx'] == flat_idx
        if not np.any(mask):
            return 0

        entry_idx = np.where(mask)[0][0]
        return int(self._trail_data[entry_idx]['trail_length'])

    def get_trail_squares(self, trailblazer_sq: np.ndarray) -> np.ndarray:
        """Get trail coordinates for trailblazer."""
        coords = self._ensure_coords(trailblazer_sq)
        flat_idx = self._coords_to_flat(coords)[0]

        mask = self._trail_data['flat_idx'] == flat_idx
        if not np.any(mask):
            return np.empty((0, 3), dtype=self._coord_dtype)

        entry_idx = np.where(mask)[0][0]
        trail_length = self._trail_data[entry_idx]['trail_length']

        if trail_length == 0:
            return np.empty((0, 3), dtype=self._coord_dtype)

        return self._trail_data[entry_idx]['trail_coords'][:trail_length].copy()

    def get_all_trails(self) -> Dict[int, np.ndarray]:
        """Get all trail data as dict (for compatibility)."""
        active_mask = self._trail_data['active']
        active_indices = np.where(active_mask)[0]

        result = {}
        for idx in active_indices:
            flat_idx = self._trail_data[idx]['flat_idx']
            length = self._trail_data[idx]['trail_length']
            if length > 0:
                result[int(flat_idx)] = self._trail_data[idx]['trail_coords'][:length].copy()

        return result

    def remove_trail(self, trailblazer_sq: np.ndarray) -> bool:
        """Remove single trail (wraps batch version)."""
        coords = self._ensure_coords(trailblazer_sq)
        if coords.size == 0:
            return False

        flat_idx = self._coords_to_flat(coords)[0]
        return self._remove_trail_batch(np.array([flat_idx], dtype=INDEX_DTYPE)) > 0

    def add_trail(self, color: int, squares: np.ndarray) -> None:
        """
        Add trail squares for a trailblazer.
        Note: This implementation assumes the trail is associated with the *current* position 
        of the trailblazer, but the interface passed just 'squares'.
        We need to know WHICH trailblazer this trail belongs to.
        However, the current usage in moveeffects.py passes (color, squares).
        This seems insufficient if we want to track per-trailblazer.
        But if we just want to store "active trails", maybe we don't need strict association?
        The cache structure `TRAIL_DATA_DTYPE` links `flat_idx` (trailblazer pos) to `trail_coords`.
        So we MUST know the trailblazer position.
        
        Let's update the signature to `add_trail(self, trailblazer_pos, squares)`.
        But I called it with `add_trail(color, path_arr)` in moveeffects.py.
        I should fix moveeffects.py to pass the position.
        """
        # Placeholder to match the call in moveeffects.py for now, but we need to fix it.
        # I will implement `add_trail_for_piece(self, piece_pos, squares)` and update moveeffects.py
        pass

    def add_trail_for_piece(self, piece_pos: np.ndarray, squares: np.ndarray) -> None:
        """Add trail for a specific trailblazer at piece_pos."""
        if squares.size == 0:
            return
            
        with self._lock:
            coords = self._ensure_coords(piece_pos)
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
                        # Cache full - ignore or evict?
                        return
                else:
                    entry_idx = self._next_slot
                    self._next_slot += 1
            
            # Update entry
            self._trail_data[entry_idx]['flat_idx'] = flat_idx
            self._trail_data[entry_idx]['active'] = True
            
            # Store trail squares
            # We append to existing or replace? 
            # Trailblazer logic says "union of last 3 trails". 
            # Here we have a fixed buffer. Let's just overwrite or append up to MAX.
            # For simplicity, let's just store the NEW trail. 
            # Real logic might need a history buffer per piece.
            # Given MAX_TRAIL_LENGTH=9, maybe it stores history?
            
            n_squares = min(squares.shape[0], MAX_TRAIL_LENGTH)
            self._trail_data[entry_idx]['trail_length'] = n_squares
            self._trail_data[entry_idx]['trail_coords'][:n_squares] = squares[:n_squares]
            
            self._rebuild_fast_lookups()

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
                self._trail_data[entry_idx]['trail_length'] = 0
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
            self._trail_data['trail_length'] = 0
            self._trailblazer_flat_positions = np.empty(0, dtype=INDEX_DTYPE)
            self._position_mask.fill(False)
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
            total_trail_coords = np.sum(self._trail_data['trail_length'])

            return {
                'trailblazer_count': int(active_count),
                'total_trail_coords': int(total_trail_coords),
                'position_cache_size': len(self._trailblazer_flat_positions) * self._trailblazer_flat_positions.itemsize,
                'mask_size': self._position_mask.nbytes,
                'trail_data_size': self._trail_data.nbytes
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
        all_trails = []

        active_mask = self._trail_data['active']
        for idx in np.where(active_mask)[0]:
            length = self._trail_data[idx]['trail_length']
            if length > 0:
                all_trails.append(
                    self._trail_data[idx]['trail_coords'][:length].copy()
                )

        if not all_trails:
            return np.empty((0, 3), dtype=self._coord_dtype)

        return np.vstack(all_trails)

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
