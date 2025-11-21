"""Unified memory pool for cache operations with optimized numpy operations."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import threading
from collections import defaultdict

# Import standardized dtypes from shared_types
from game3d.common.shared_types import (
    COORD_DTYPE, BATCH_COORD_DTYPE, INDEX_DTYPE, BOOL_DTYPE, FLOAT_DTYPE, INT8_DTYPE,
    COLOR_DTYPE, PIECE_TYPE_DTYPE, COORD_OFFSET_DTYPE, SIZE, VOLUME
)

# Memory pool configuration
MAX_POOL_SIZE = 50
DEFAULT_COORD_POOL_SIZE = 100
DEFAULT_BOOL_POOL_SIZE = 75
DEFAULT_INDEX_POOL_SIZE = 80

# Pool size ratios based on typical usage patterns
POOL_SIZE_RATIOS = {
    'coords_3': 1.0,     # High usage
    'coords_1': 0.8,     # Medium-high usage  
    'bool': 0.6,         # Medium usage
    'index': 0.8,        # Medium-high usage
    'count': 0.4,        # Lower usage
    'float': 0.3,        # Lower usage
    'piece': 0.7,        # Medium usage
    'occupancy': 0.9,    # High usage
    'piece_type': 0.9,   # High usage
    'board_3d': 1.0,     # High usage
    'board_4d': 0.2,     # Very low usage
    'blocked': 0.5,      # Low-medium usage
}


class UnifiedMemoryPool:
    """Unified memory pool for efficient array reuse."""
    
    def __init__(self, max_pool_size: int = MAX_POOL_SIZE, max_memory_bytes: int = 2 * 1024**3):
        self.max_pool_size = max_pool_size
        self._lock = threading.RLock()
        
        # ✅ OPTIMIZATION: Global memory tracking
        self._max_memory_bytes = max_memory_bytes  # Default 2GB per pool
        self._total_pooled_bytes = 0
        
        # Initialize memory pools for different array types
        self._pools = {
            'coords_3': [],  # (N, 3) coordinate arrays
            'coords_1': [],  # (N,) coordinate arrays
            'bool': [],      # Boolean/int8 arrays
            'index': [],     # Index arrays
            'count': [],     # Count arrays
            'float': [],     # Float arrays
            'piece': [],     # Piece [type, color] arrays
            'occupancy': [], # (SIZE, SIZE, SIZE) occupancy arrays
            'piece_type': [], # (SIZE, SIZE, SIZE) piece type arrays
            'board_3d': [],  # 3D board arrays
            'board_4d': [],  # 4D board arrays
            'blocked': [],   # Blocked indices/expiry arrays
        }
        
        # Simple statistics tracking
        self._hits = defaultdict(int)
        self._misses = defaultdict(int)
        self._total_allocated = defaultdict(int)

    def allocate_array(self, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Generic allocation method compatible with other pool interfaces."""
        # Determine pool key based on shape and dtype
        key = None
        
        if dtype == COORD_DTYPE:
            if len(shape) == 2 and shape[1] == 3:
                key = 'coords_3'
            elif len(shape) == 1:
                key = 'coords_1'
        elif dtype == BOOL_DTYPE:
            key = 'bool'
        elif dtype == INDEX_DTYPE:
            key = 'index'
        elif dtype == COLOR_DTYPE:
            if shape == (SIZE, SIZE, SIZE):
                key = 'occupancy' # Default to occupancy for 3D color arrays
            elif len(shape) == 2 and shape[1] == 2:
                key = 'piece'
        elif dtype == PIECE_TYPE_DTYPE:
             if shape == (SIZE, SIZE, SIZE):
                key = 'piece_type'
        elif dtype == INT8_DTYPE or dtype == np.uint16:
            key = 'count'
        elif dtype == FLOAT_DTYPE or dtype == np.float32:
             if len(shape) == 4:
                 key = 'board_4d'
             else:
                 key = 'float'
        elif dtype == COORD_OFFSET_DTYPE:
            key = 'blocked'

        if key:
            return self._get_array(key, shape, dtype)
        
        # Fallback
        return np.empty(shape, dtype=dtype)
    
    def get_coords_3(self, shape: Tuple[int, int], dtype: np.dtype = COORD_DTYPE) -> np.ndarray:
        """Get or create (N, 3) coordinate array."""
        return self._get_array('coords_3', shape, dtype)
    
    def get_coords_1(self, shape: Tuple[int], dtype: np.dtype = COORD_DTYPE) -> np.ndarray:
        """Get or create (N,) coordinate array."""
        return self._get_array('coords_1', shape, dtype)
    
    def get_bool_array(self, shape: Tuple[int], dtype: np.dtype = BOOL_DTYPE) -> np.ndarray:
        """Get or create boolean array."""
        return self._get_array('bool', shape, dtype)
    
    def get_index_array(self, shape: Tuple[int], dtype: np.dtype = INDEX_DTYPE) -> np.ndarray:
        """Get or create index array."""
        return self._get_array('index', shape, dtype)
    
    def get_count_array(self, shape: Tuple[int]) -> np.ndarray:
        """Get or create count array (uint16)."""
        return self._get_array('count', shape, np.uint16)
    
    def get_piece_array(self, shape: Tuple[int, int]) -> np.ndarray:
        """Get or create piece array [type, color]."""
        return self._get_array('piece', shape, COLOR_DTYPE)
    
    def get_occupancy_array(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Get or create occupancy array (SIZE, SIZE, SIZE)."""
        return self._get_array('occupancy', shape, COLOR_DTYPE)
    
    def get_piece_type_array(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Get or create piece type array (SIZE, SIZE, SIZE)."""
        return self._get_array('piece_type', shape, PIECE_TYPE_DTYPE)
    
    def get_board_3d(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Get or create 3D board array."""
        return self._get_array('board_3d', shape, COLOR_DTYPE)
    
    def get_board_4d(self, shape: Tuple[int, int, int, int]) -> np.ndarray:
        """Get or create 4D board array."""
        return self._get_array('board_4d', shape, FLOAT_DTYPE)
    
    def get_blocked_array(self, shape: Tuple[int]) -> np.ndarray:
        """Get or create blocked indices array."""
        return self._get_array('blocked', shape, COORD_OFFSET_DTYPE)
    
    def _get_array(self, key: str, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Get array from pool or create new one."""
        with self._lock:
            if self._pools[key]:
                arr = self._pools[key].pop()
                if arr.shape == shape and arr.dtype == dtype:
                    self._hits[key] += 1
                    return arr
            
            self._misses[key] += 1
            self._total_allocated[key] += 1
            return np.empty(shape, dtype=dtype, order='C')
    
    def release(self, arr: np.ndarray, pool_type: Optional[str] = None) -> None:
        if arr.size == 0:
            return

        with self._lock:
            if pool_type is None:
                pool_type = self._determine_pool_type(arr)

            if pool_type and pool_type in self._pools:
                # ✅ OPTIMIZATION: Check global memory limit before adding
                arr_bytes = arr.nbytes
                
                # Enforce per-pool limits
                if len(self._pools[pool_type]) >= self.max_pool_size:
                    return  # Pool full, don't add
                
                # Check global memory limit
                if self._total_pooled_bytes + arr_bytes > self._max_memory_bytes:
                    # Evict from largest pools first
                    self._trim_to_fit(arr_bytes)
                
                # CLEAR ARRAY BEFORE RETURNING TO POOL
                arr.fill(0)
                self._pools[pool_type].append(arr)
                self._total_pooled_bytes += arr_bytes
    
    def _determine_pool_type(self, arr: np.ndarray) -> Optional[str]:
        """Determine appropriate pool type for array."""
        # Check for structured arrays first
        if arr.dtype.names is not None:
            if any('from_' in name or 'to_' in name for name in arr.dtype.names):
                return None  # Don't pool move structs
        
        # Determine pool type based on dtype and shape
        if arr.dtype == COORD_DTYPE:
            return 'coords_3' if arr.ndim == 2 and arr.shape[1] == 3 else 'coords_1'
        elif arr.dtype == BOOL_DTYPE:
            return 'bool'
        elif arr.dtype == INDEX_DTYPE:
            return 'index'
        elif arr.dtype == COLOR_DTYPE:
            if arr.shape == (SIZE, SIZE, SIZE):
                return 'board_3d' if arr.ndim == 3 else 'occupancy'
            elif arr.ndim == 2 and arr.shape[1] == 2:
                return 'piece'
        elif arr.dtype == PIECE_TYPE_DTYPE:
            if arr.shape == (SIZE, SIZE, SIZE):
                return 'piece_type'
        elif arr.dtype == INT8_DTYPE:
            return 'count'
        elif arr.dtype == FLOAT_DTYPE:
            return 'board_4d' if arr.ndim == 4 else 'float'
        elif arr.dtype == COORD_OFFSET_DTYPE:
            return 'blocked'
        
        return None
    
    def _trim_to_fit(self, needed_bytes: int) -> None:
        """Trim pools to make room for new array.
        
        OPTIMIZATION: Smart eviction - remove from largest pools first.
        """
        # Calculate how much we need to free
        bytes_to_free = (self._total_pooled_bytes + needed_bytes) - self._max_memory_bytes
        
        if bytes_to_free <= 0:
            return
        
        # Sort pools by total size (largest first)
        pool_sizes = []
        for pool_type, pool in self._pools.items():
            if pool:
                pool_bytes = sum(arr.nbytes for arr in pool)
                pool_sizes.append((pool_type, pool_bytes, len(pool)))
        
        pool_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Evict from largest pools until we have enough space
        freed = 0
        for pool_type, pool_bytes, pool_count in pool_sizes:
            if freed >= bytes_to_free:
                break
            
            pool = self._pools[pool_type]
            # Remove oldest 30% from this pool
            remove_count = max(1, len(pool) // 3)
            for _ in range(remove_count):
                if pool:
                    removed_arr = pool.pop(0)  # Remove oldest (FIFO)
                    freed += removed_arr.nbytes
                    self._total_pooled_bytes -= removed_arr.nbytes
                    
                if freed >= bytes_to_free:
                    break
    
    def clear_pool(self, pool_type: str) -> None:
        """Clear specific pool."""
        with self._lock:
            if pool_type in self._pools:
                # Subtract bytes from total before clearing
                for arr in self._pools[pool_type]:
                    self._total_pooled_bytes -= arr.nbytes
                self._pools[pool_type].clear()
    
    def clear_all(self) -> None:
        """Clear all pools."""
        with self._lock:
            for pool_type in self._pools:
                self._pools[pool_type].clear()
            self._total_pooled_bytes = 0  # Reset total tracking
    
    def clear(self) -> None:
        """Clear all pools (alias for clear_all)."""
        self.clear_all()
    
    def trim_pools(self, target_size: int = None) -> None:
        """Trim pools to target size to free memory."""
        if target_size is None:
            target_size = self.max_pool_size // 2
        
        with self._lock:
            for pool_type, pool in self._pools.items():
                # Use pool-specific ratios for better memory management
                pool_target = int(target_size * POOL_SIZE_RATIOS.get(pool_type, 0.5))
                while len(pool) > pool_target:
                    pool.pop()
    
    def optimize_pool_sizes(self) -> None:
        """Optimize pool sizes based on usage statistics."""
        with self._lock:
            for pool_type in self._pools:
                pool_ops = self._hits[pool_type] + self._misses[pool_type]
                if pool_ops > 0:
                    hit_rate = self._hits[pool_type] / pool_ops
                    # Increase pool size for high-performing pools
                    if hit_rate > 0.8:
                        target_size = min(len(self._pools[pool_type]) * 2, self.max_pool_size)
                    else:
                        target_size = min(len(self._pools[pool_type]) // 2, self.max_pool_size // 2)
                    
                    # Trim pool to optimized size
                    pool = self._pools[pool_type]
                    while len(pool) > target_size:
                        pool.pop()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            total_hits = sum(self._hits.values())
            total_misses = sum(self._misses.values())
            total_ops = total_hits + total_misses
            hit_rate = total_hits / total_ops if total_ops > 0 else 0.0
            
            pool_stats = {}
            for pool_type, pool in self._pools.items():
                pool_hits = self._hits[pool_type]
                pool_misses = self._misses[pool_type]
                pool_ops = pool_hits + pool_misses
                pool_hit_rate = pool_hits / pool_ops if pool_ops > 0 else 0.0
                
                pool_stats[pool_type] = {
                    'size': len(pool),
                    'hits': pool_hits,
                    'misses': pool_misses,
                    'hit_rate': pool_hit_rate,
                    'total_allocated': self._total_allocated[pool_type]
                }
            
            return {
                'total_hits': total_hits,
                'total_misses': total_misses,
                'overall_hit_rate': hit_rate,
                'pool_stats': pool_stats
            }
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        usage = {}
        for pool_type, pool in self._pools.items():
            total_bytes = sum(arr.nbytes for arr in pool)
            usage[f'{pool_type}_nbytes'] = total_bytes
            usage[f'{pool_type}_count'] = len(pool)
        
        usage['total_nbytes'] = sum(arr.nbytes for pool in self._pools.values() for arr in pool)
        usage['total_arrays'] = sum(len(pool) for pool in self._pools.values())
        
        return usage
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._hits.clear()
            self._misses.clear()
            self._total_allocated.clear()


# Global unified memory pool instance
_UNIFIED_MEMORY_POOL = UnifiedMemoryPool()


# Convenience functions
def get_memory_pool() -> UnifiedMemoryPool:
    """Get the global unified memory pool instance."""
    return _UNIFIED_MEMORY_POOL


def get_coords_3(shape: Tuple[int, int], dtype: np.dtype = COORD_DTYPE) -> np.ndarray:
    """Get coordinate array (N, 3)."""
    return _UNIFIED_MEMORY_POOL.get_coords_3(shape, dtype)


def get_coords_1(shape: Tuple[int], dtype: np.dtype = COORD_DTYPE) -> np.ndarray:
    """Get coordinate array (N,)."""
    return _UNIFIED_MEMORY_POOL.get_coords_1(shape, dtype)


def get_bool_array(shape: Tuple[int], dtype: np.dtype = BOOL_DTYPE) -> np.ndarray:
    """Get boolean array."""
    return _UNIFIED_MEMORY_POOL.get_bool_array(shape, dtype)


def get_index_array(shape: Tuple[int], dtype: np.dtype = INDEX_DTYPE) -> np.ndarray:
    """Get index array."""
    return _UNIFIED_MEMORY_POOL.get_index_array(shape, dtype)


def get_piece_array(shape: Tuple[int, int]) -> np.ndarray:
    """Get piece array [type, color]."""
    return _UNIFIED_MEMORY_POOL.get_piece_array(shape)


def get_occupancy_array(shape: Tuple[int, int, int]) -> np.ndarray:
    """Get occupancy array (SIZE, SIZE, SIZE)."""
    return _UNIFIED_MEMORY_POOL.get_occupancy_array(shape)


def get_piece_type_array(shape: Tuple[int, int, int]) -> np.ndarray:
    """Get piece type array (SIZE, SIZE, SIZE)."""
    return _UNIFIED_MEMORY_POOL.get_piece_type_array(shape)


def get_board_3d(shape: Tuple[int, int, int]) -> np.ndarray:
    """Get 3D board array."""
    return _UNIFIED_MEMORY_POOL.get_board_3d(shape)


def get_board_4d(shape: Tuple[int, int, int, int]) -> np.ndarray:
    """Get 4D board array."""
    return _UNIFIED_MEMORY_POOL.get_board_4d(shape)


def get_blocked_array(shape: Tuple[int]) -> np.ndarray:
    """Get blocked indices array."""
    return _UNIFIED_MEMORY_POOL.get_blocked_array(shape)


def release_array(arr: np.ndarray, pool_type: Optional[str] = None) -> None:
    """Release array back to pool."""
    _UNIFIED_MEMORY_POOL.release(arr, pool_type)


def get_pool_stats() -> Dict[str, Any]:
    """Get pool statistics."""
    return _UNIFIED_MEMORY_POOL.get_stats()


def clear_all_pools() -> None:
    """Clear all pools."""
    _UNIFIED_MEMORY_POOL.clear_all()


def optimize_pool_sizes() -> None:
    """Optimize pool sizes based on usage statistics."""
    _UNIFIED_MEMORY_POOL.optimize_pool_sizes()


def get_pool_memory_usage() -> Dict[str, int]:
    """Get detailed memory usage statistics."""
    return _UNIFIED_MEMORY_POOL.get_memory_usage()


def reset_pool_stats() -> None:
    """Reset pool statistics."""
    _UNIFIED_MEMORY_POOL.reset_stats()


# Export main interface
__all__ = [
    'UnifiedMemoryPool',
    'get_memory_pool',
    'get_coords_3',
    'get_coords_1',
    'get_bool_array',
    'get_index_array',
    'get_piece_array',
    'get_occupancy_array',
    'get_piece_type_array',
    'get_board_3d',
    'get_board_4d',
    'get_blocked_array',
    'release_array',
    'get_pool_stats',
    'clear_all_pools',
    'optimize_pool_sizes',
    'get_pool_memory_usage',
    'reset_pool_stats',
]
