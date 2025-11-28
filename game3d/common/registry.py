# game3d/common/registry.py - SIMPLE FUNCTIONAL REGISTRY
"""Simple registry for piece dispatchers - maps PieceType to dispatcher functions."""
from typing import Callable, Optional, Dict, Any
import numpy as np
from game3d.common.shared_types import PieceType

# Simple dictionary mapping piece type -> dispatcher function
_dispatcher_registry: Dict[int, Callable] = {}

# ✅ OPTIMIZATION: Pre-allocated array for O(1) dispatcher lookup
# Indexed by piece_type (max 50 types), stores function references
_dispatcher_array: Optional[np.ndarray] = None
_dispatcher_array_dirty = True  # Flag to rebuild array when registry changes

def register(piece_type: int):
    """Decorator to register a piece dispatcher function."""
    def _decorator(fn: Callable):
        global _dispatcher_array_dirty
        _dispatcher_registry[piece_type] = fn
        _dispatcher_array_dirty = True  # Mark array for rebuild
        return fn
    return _decorator

def get_piece_dispatcher(piece_type: int) -> Optional[Callable]:
    """Retrieve the dispatcher function for a piece type.
    
    ✅ OPTIMIZED: Uses cached array for O(1) lookup without dict overhead.
    """
    global _dispatcher_array, _dispatcher_array_dirty
    
    # Rebuild dispatcher array if registry was modified
    if _dispatcher_array_dirty or _dispatcher_array is None:
        _rebuild_dispatcher_array()
    
    # Fast array lookup (bounds check first)
    if 0 <= piece_type < len(_dispatcher_array):
        dispatcher = _dispatcher_array[piece_type]
        return dispatcher if dispatcher is not None else None
    
    return None

def get_piece_dispatcher_fast(piece_type: int) -> Optional[Callable]:
    """Ultra-fast dispatcher lookup - assumes array is warm.
    
    WARNING: Only use in hot loops where you know the array is initialized.
    Skips the dirty check for maximum performance.
    """
    global _dispatcher_array
    
    if _dispatcher_array is None:
        _rebuild_dispatcher_array()
    
    # Direct array access without bounds check (assumes valid piece_type)
    return _dispatcher_array[piece_type] if piece_type < len(_dispatcher_array) else None

def _rebuild_dispatcher_array():
    """Rebuild the dispatcher array from the registry."""
    global _dispatcher_array, _dispatcher_array_dirty
    
    if not _dispatcher_registry:
        _dispatcher_array = np.empty(50, dtype=object)
        _dispatcher_array.fill(None)
        _dispatcher_array_dirty = False
        return
    
    # Determine max piece type to size the array
    max_piece_type = max(_dispatcher_registry.keys())
    array_size = max(max_piece_type + 1, 50)  # At least size 50
    
    # Create object array to hold function references
    _dispatcher_array = np.empty(array_size, dtype=object)
    _dispatcher_array.fill(None)
    
    # Populate array
    for piece_type, dispatcher in _dispatcher_registry.items():
        _dispatcher_array[piece_type] = dispatcher
    
    _dispatcher_array_dirty = False

def is_piece_registered(piece_type: int) -> bool:
    """Check if a piece type has a registered dispatcher."""
    return piece_type in _dispatcher_registry

def get_all_registered_types() -> np.ndarray:
    """Get numpy array of all registered piece types."""
    if not _dispatcher_registry:
        return np.array([], dtype=np.int32)
    return np.fromiter(_dispatcher_registry.keys(), dtype=np.int32, count=len(_dispatcher_registry))

def clear_registry():
    """Clear all registrations (useful for testing)."""
    global _dispatcher_registry, _dispatcher_array_dirty
    _dispatcher_registry.clear()
    _dispatcher_array_dirty = True

__all__ = ['register', 'get_piece_dispatcher', 'get_piece_dispatcher_fast', 
           'is_piece_registered', 'get_all_registered_types', 'clear_registry']
