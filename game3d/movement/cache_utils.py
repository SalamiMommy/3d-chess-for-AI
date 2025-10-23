# game3d/movement/cache_utils.py
"""Standardized cache access patterns for movement generators"""
import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING
from game3d.common.coord_utils import in_bounds
from game3d.pieces.piece import Piece
from game3d.common.enums import Color

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager

def get_piece(cache_manager: 'OptimizedCacheManager', coord: Tuple[int, int, int]) -> Optional[Piece]:
    """Standardized piece access with bounds checking"""
    if not in_bounds(coord):
        return None
    return cache_manager.occupancy.get(coord)

def is_occupied(cache_manager: 'OptimizedCacheManager', coord: Tuple[int, int, int]) -> bool:
    """Standardized occupancy check"""
    return get_piece(cache_manager, coord) is not None

def ensure_int_coords(x, y, z) -> Tuple[int, int, int]:
    """Ensure coordinates are Python ints to prevent numpy type issues"""
    return (int(x), int(y), int(z))

def get_occupancy_array(cache_manager: 'OptimizedCacheManager') -> np.ndarray:
    """Get occupancy array for movement generators with proper access pattern."""
    return cache_manager.occupancy._occ

def is_frozen(cache_manager: 'OptimizedCacheManager', coord: Tuple[int, int, int], color: Color) -> bool:
    """Standardized freeze check"""
    return cache_manager.is_frozen(coord, color)

def is_movement_debuffed(cache_manager: 'OptimizedCacheManager', coord: Tuple[int, int, int], color: Color) -> bool:
    """Standardized debuff check"""
    return cache_manager.is_movement_debuffed(coord, color)

def is_geomancy_blocked(cache_manager: 'OptimizedCacheManager', coord: Tuple[int, int, int], current_ply: int) -> bool:
    """Standardized geomancy block check"""
    return cache_manager.is_geomancy_blocked(coord, current_ply)
