# game3d/common/registry.py - SIMPLE FUNCTIONAL REGISTRY
"""Simple registry for piece dispatchers - maps PieceType to dispatcher functions."""
from typing import Callable, Optional, Dict
from game3d.common.shared_types import PieceType

# Simple dictionary mapping piece type -> dispatcher function
_dispatcher_registry: Dict[int, Callable] = {}

def register(piece_type: int):
    """Decorator to register a piece dispatcher function."""
    def _decorator(fn: Callable):
        _dispatcher_registry[piece_type] = fn
        return fn
    return _decorator

def get_piece_dispatcher(piece_type: int) -> Optional[Callable]:
    """Retrieve the dispatcher function for a piece type."""
    return _dispatcher_registry.get(piece_type)

def is_piece_registered(piece_type: int) -> bool:
    """Check if a piece type has a registered dispatcher."""
    return piece_type in _dispatcher_registry

def get_all_registered_types() -> 'np.ndarray':
    """Get numpy array of all registered piece types."""
    import numpy as np
    if not _dispatcher_registry:
        return np.array([], dtype=np.int32)
    return np.fromiter(_dispatcher_registry.keys(), dtype=np.int32, count=len(_dispatcher_registry))

def clear_registry():
    """Clear all registrations (useful for testing)."""
    global _dispatcher_registry
    _dispatcher_registry.clear()

__all__ = ['register', 'get_piece_dispatcher', 'is_piece_registered', 'get_all_registered_types', 'clear_registry']
