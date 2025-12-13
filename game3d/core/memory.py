"""
Thread-local memory management for high-performance move generation.
Avoids significant allocation overhead by reusing numpy arrays in hot paths.
"""

import threading
import numpy as np
from typing import Tuple, Optional
from game3d.common.shared_types import INDEX_DTYPE

class ThreadLocalBufferPool:
    """
    Manages thread-local scratch buffers for move generation.
    Eliminates malloc/free overhead in tight loops.
    """
    _local = threading.local()

    @classmethod
    def get_scratch_pad(cls) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get thread-local scratch pad buffers.
        Returns a tuple of (type_counts, effective_types, sorted_indices, offsets).
        
        Buffers are reused across calls within the same thread.
        Resetting logic (e.g. .fill(0)) is the responsibility of the caller if needed,
        though overwriting is preferred for performance.
        """
        if not hasattr(cls._local, 'scratch_pad'):
            # Provide generous sizes to avoid reallocation
            # 64 types max
            # 1024 pieces max per subset (covering most rigorous cases)
            type_counts = np.zeros(64, dtype=INDEX_DTYPE)
            effective_types = np.zeros(1024, dtype=INDEX_DTYPE)
            sorted_indices = np.zeros(1024, dtype=INDEX_DTYPE)
            offsets = np.zeros(64, dtype=INDEX_DTYPE)
            
            cls._local.scratch_pad = (type_counts, effective_types, sorted_indices, offsets)
            
        return cls._local.scratch_pad

    @classmethod
    def clear(cls):
        """Clear thread-local storage (mainly for testing)."""
        if hasattr(cls._local, 'scratch_pad'):
            del cls._local.scratch_pad
