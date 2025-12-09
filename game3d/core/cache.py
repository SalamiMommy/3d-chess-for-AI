
"""
Simple Move Cache for Stateless Generator.
Uses Zobrist key for lookup.
"""

import numpy as np
from typing import Optional, Dict
from game3d.common.shared_types import COORD_DTYPE, HASH_DTYPE

class StatelessMoveCache:
    """
    Simple LRU-style move cache keyed by Zobrist hash.
    """
    
    def __init__(self, max_entries: int = 1024):
        self.max_entries = max_entries
        self._cache: Dict[int, np.ndarray] = {}
        self._access_order: list = []
    
    def get(self, zkey: int) -> Optional[np.ndarray]:
        """Retrieve cached moves for a position."""
        if zkey in self._cache:
            # Move to end of access order (LRU)
            if zkey in self._access_order:
                self._access_order.remove(zkey)
            self._access_order.append(zkey)
            return self._cache[zkey]
        return None
    
    def store(self, zkey: int, moves: np.ndarray) -> None:
        """Store moves for a position."""
        if zkey in self._cache:
            self._cache[zkey] = moves
            return
            
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_entries:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        self._cache[zkey] = moves
        self._access_order.append(zkey)
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
    
    def __len__(self) -> int:
        return len(self._cache)


# Global cache instance
_move_cache = StatelessMoveCache()

def get_cached_moves(zkey: int) -> Optional[np.ndarray]:
    """Get cached moves if available."""
    return _move_cache.get(zkey)

def cache_moves(zkey: int, moves: np.ndarray) -> None:
    """Cache moves for a position."""
    _move_cache.store(zkey, moves.copy())

def clear_cache() -> None:
    """Clear the move cache."""
    _move_cache.clear()
