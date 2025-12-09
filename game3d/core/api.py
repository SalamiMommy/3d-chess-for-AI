
"""
High-Level Stateless Move Generation API.
Wraps generator_functional and attacks with caching.
"""

import numpy as np
from typing import Optional

from game3d.core.buffer import GameBuffer
from game3d.core.generator_functional import generate_moves as _generate_pseudolegal
from game3d.core.attacks import filter_legal_moves as _filter_legal
from game3d.core.cache import get_cached_moves, cache_moves, clear_cache

def generate_legal_moves(buffer: GameBuffer, use_cache: bool = True) -> np.ndarray:
    """
    Generate all legal moves for the current position.
    
    Args:
        buffer: GameBuffer representing current position
        use_cache: If True, use move cache for fast lookups
        
    Returns:
        Array of legal moves (N, 6)
    """
    zkey = buffer.zkey
    
    # Check cache
    if use_cache:
        cached = get_cached_moves(zkey)
        if cached is not None:
            return cached
    
    # Generate fresh
    pseudo = _generate_pseudolegal(buffer)
    legal = _filter_legal(buffer, pseudo)
    
    # Store in cache
    if use_cache:
        cache_moves(zkey, legal)
    
    return legal

def generate_pseudolegal_moves(buffer: GameBuffer) -> np.ndarray:
    """
    Generate all pseudo-legal moves (no check filtering).
    
    Args:
        buffer: GameBuffer representing current position
        
    Returns:
        Array of pseudo-legal moves (N, 6)
    """
    return _generate_pseudolegal(buffer)

def invalidate_cache() -> None:
    """Clear the move cache."""
    clear_cache()

def generate_pseudolegal_moves_subset(buffer: GameBuffer, indices: np.ndarray) -> np.ndarray:
    """Generate moves for a subset of pieces."""
    from game3d.core.generator_functional import generate_moves_subset
    return generate_moves_subset(buffer, indices)

__all__ = ['generate_legal_moves', 'generate_pseudolegal_moves', 'invalidate_cache', 'generate_pseudolegal_moves_subset']
