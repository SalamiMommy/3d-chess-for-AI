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

def get_cache_manager(board_or_state):
    """Unified cache manager access."""
    if hasattr(board_or_state, 'cache_manager'):
        return board_or_state.cache_manager
    if hasattr(board_or_state, 'cache'):
        return board_or_state.cache
    return None

def validate_cache_integrity(game_state: 'GameState') -> bool:
    """Validate that all caches are in sync after move."""
    try:
        # Check occupancy cache vs board
        for coord, piece in game_state.board.list_occupied():
            cached_piece = game_state.cache_manager.occupancy.get(coord)
            if cached_piece != piece:
                print(f"Cache desync at {coord}: board={piece}, cache={cached_piece}")
                return False

        # Check Zobrist hash
        computed_hash = compute_zobrist(game_state.board, game_state.color)
        if computed_hash != game_state.zkey:
            print(f"Zobrist desync: computed={computed_hash:#x}, cached={game_state.zkey:#x}")
            return False

        return True
    except Exception as e:
        print(f"Cache validation failed: {e}")
        return False

def batch_process_effect_updates(
    cache_manager: "OptimizedCacheManager",
    updates: List[Tuple[Coord, Optional[Piece]]],
    effect_types: List[str]
) -> None:
    """Standardized batch processing for effect caches."""
    # Update occupancy first
    cache_manager.batch_set_pieces(updates)

    # Update effect caches
    for effect_type in effect_types:
        cache = cache_manager._get_cache_by_name(effect_type)
        if cache and hasattr(cache, 'batch_update'):
            cache.batch_update(updates)

def synchronize_zobrist_after_move(
    cache_manager: "OptimizedCacheManager",
    move: "Move",
    from_piece: Piece,
    captured_piece: Optional[Piece],
    new_color: Color
) -> None:
    """Standardized Zobrist hash synchronization after moves."""
    new_hash = cache_manager._zobrist.update_hash_move(
        cache_manager._current_zobrist_hash,
        move,
        from_piece,
        captured_piece
    )
    cache_manager.sync_zobrist(new_hash)

def is_occupied_safe(cache_manager: 'OptimizedCacheManager', coord: Tuple[int, int, int]) -> bool:
    """Safe occupancy check with bounds validation."""
    if not in_bounds(coord):
        return False
    return cache_manager.get_piece(coord) is not None
