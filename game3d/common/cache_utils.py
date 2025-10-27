# game3d/movement/cache_utils.py
"""Standardized cache access patterns for movement generators"""
import numpy as np
import torch
from typing import Tuple, Optional, TYPE_CHECKING, List, Union, Dict
from game3d.common.coord_utils import in_bounds
from game3d.pieces.piece import Piece
from game3d.common.enums import Color

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.move import Move

def get_piece(cache_manager: 'OptimizedCacheManager', coord: Union[Tuple[int, int, int], torch.Tensor]) -> Union[Optional[Piece], List[Optional[Piece]]]:
    """Standardized piece access with bounds checking - supports scalar and batch mode"""
    if torch.is_tensor(coord) and coord.ndim > 1:
        # Batch mode: coord is [N, 3] tensor
        coords_list = [tuple(coord[i].tolist()) for i in range(coord.shape[0])]
        return [get_piece(cache_manager, c) for c in coords_list]
    else:
        # Scalar mode
        if torch.is_tensor(coord):
            coord = tuple(coord.tolist())
        if not in_bounds(coord):
            return None
        return cache_manager.occupancy.get(coord)

def is_occupied(cache_manager: 'OptimizedCacheManager', coord: Union[Tuple[int, int, int], torch.Tensor]) -> Union[bool, torch.Tensor]:
    """Standardized occupancy check - optimized by checking bounds first - supports scalar and batch mode"""
    if torch.is_tensor(coord) and coord.ndim > 1:
        # Batch mode: coord is [N, 3] tensor
        results = []
        for i in range(coord.shape[0]):
            single_coord = tuple(coord[i].tolist())
            results.append(is_occupied(cache_manager, single_coord))
        return torch.tensor(results, dtype=torch.bool)
    else:
        # Scalar mode
        if torch.is_tensor(coord):
            coord = tuple(coord.tolist())
        return in_bounds(coord) and cache_manager.occupancy.get(coord) is not None

def ensure_int_coords(x, y, z) -> Union[Tuple[int, int, int], torch.Tensor]:
    """Ensure coordinates are Python ints to prevent numpy type issues - supports scalar and batch mode"""
    if torch.is_tensor(x) and x.ndim > 0:
        # Batch mode
        if x.ndim == 1 and len(x) == 3:
            # Single coordinate as tensor, convert to tuple
            return tuple(x.int().tolist())
        else:
            # Batch of coordinates, ensure all are int tensors
            return torch.stack([x.int(), y.int(), z.int()], dim=-1)
    else:
        # Scalar mode
        return int(x), int(y), int(z)

def get_occupancy_array(cache_manager: 'OptimizedCacheManager') -> np.ndarray:
    """Get occupancy array for movement generators with proper access pattern."""
    return cache_manager.occupancy._occ

def is_frozen(cache_manager: 'OptimizedCacheManager', coord: Union[Tuple[int, int, int], torch.Tensor], color: Union[Color, torch.Tensor]) -> Union[bool, torch.Tensor]:
    """Standardized freeze check - optimized with direct cache access - supports scalar and batch mode"""
    if torch.is_tensor(coord) and coord.ndim > 1:
        # Batch mode
        results = []
        for i in range(coord.shape[0]):
            single_coord = tuple(coord[i].tolist())
            single_color = color[i] if torch.is_tensor(color) else color
            results.append(is_frozen(cache_manager, single_coord, single_color))
        return torch.tensor(results, dtype=torch.bool)
    else:
        # Scalar mode
        if torch.is_tensor(coord):
            coord = tuple(coord.tolist())
        if torch.is_tensor(color):
            color = Color(color.item())

        return cache_manager._frozen_cache.get(coord, color) if hasattr(cache_manager, '_frozen_cache') else cache_manager.is_frozen(coord, color)

def is_movement_debuffed(cache_manager: 'OptimizedCacheManager', coord: Union[Tuple[int, int, int], torch.Tensor], color: Union[Color, torch.Tensor]) -> Union[bool, torch.Tensor]:
    """Standardized debuff check - optimized with direct cache access - supports scalar and batch mode"""
    if torch.is_tensor(coord) and coord.ndim > 1:
        # Batch mode
        results = []
        for i in range(coord.shape[0]):
            single_coord = tuple(coord[i].tolist())
            single_color = color[i] if torch.is_tensor(color) else color
            results.append(is_movement_debuffed(cache_manager, single_coord, single_color))
        return torch.tensor(results, dtype=torch.bool)
    else:
        # Scalar mode
        if torch.is_tensor(coord):
            coord = tuple(coord.tolist())
        if torch.is_tensor(color):
            color = Color(color.item())

        return cache_manager._debuff_cache.get(coord, color) if hasattr(cache_manager, '_debuff_cache') else cache_manager.is_movement_debuffed(coord, color)

def is_geomancy_blocked(cache_manager: 'OptimizedCacheManager', coord: Union[Tuple[int, int, int], torch.Tensor], current_ply: Union[int, torch.Tensor]) -> Union[bool, torch.Tensor]:
    """Standardized geomancy block check - optimized with direct cache access - supports scalar and batch mode"""
    if torch.is_tensor(coord) and coord.ndim > 1:
        # Batch mode
        results = []
        for i in range(coord.shape[0]):
            single_coord = tuple(coord[i].tolist())
            single_ply = current_ply[i] if torch.is_tensor(current_ply) else current_ply
            results.append(is_geomancy_blocked(cache_manager, single_coord, single_ply))
        return torch.tensor(results, dtype=torch.bool)
    else:
        # Scalar mode
        if torch.is_tensor(coord):
            coord = tuple(coord.tolist())
        if torch.is_tensor(current_ply):
            current_ply = current_ply.item()

        return cache_manager._geomancy_cache.is_blocked(coord, current_ply) if hasattr(cache_manager, '_geomancy_cache') else cache_manager.is_geomancy_blocked(coord, current_ply)

def get_cache_manager(board_or_state):
    """Unified cache manager access - optimized attribute checking"""
    # Single hasattr check per possible attribute
    if hasattr(board_or_state, 'cache_manager'):
        return board_or_state.cache_manager
    return board_or_state.cache if hasattr(board_or_state, 'cache') else None

def batch_process_effect_updates(
    cache_manager: "OptimizedCacheManager",
    updates: List[Tuple[Tuple[int, int, int], Optional[Piece]]],
    effect_types: List[str]
) -> None:
    """Standardized batch processing for effect caches - optimized batch operations"""
    # Update occupancy first using single batch operation
    cache_manager.batch_set_pieces(updates)

    # Update effect caches - pre-filter valid caches
    valid_effect_caches = []
    for effect_type in effect_types:
        cache = cache_manager._get_cache_by_name(effect_type)
        if cache and hasattr(cache, 'batch_update'):
            valid_effect_caches.append(cache)

    # Batch update all valid caches
    for cache in valid_effect_caches:
        cache.batch_update(updates)

def synchronize_zobrist_after_move(
    cache_manager: "OptimizedCacheManager",
    move: "Move",
    from_piece: Piece,
    captured_piece: Optional[Piece],
    new_color: Color
) -> None:
    """Standardized Zobrist hash synchronization after moves - optimized with direct hash update"""
    # Use direct zobrist update if available
    if hasattr(cache_manager, '_zobrist'):
        new_hash = cache_manager._zobrist.update_hash_move(
            cache_manager._current_zobrist_hash,
            move,
            from_piece,
            captured_piece
        )
        cache_manager.sync_zobrist(new_hash)
    else:
        # Fallback to full recomputation
        from game3d.cache.zobrist import compute_zobrist
        board = getattr(cache_manager, 'board', None)
        if board:
            new_hash = compute_zobrist(board, new_color)
            cache_manager.sync_zobrist(new_hash)

def is_occupied_safe(cache_manager: 'OptimizedCacheManager', coord: Union[Tuple[int, int, int], torch.Tensor]) -> Union[bool, torch.Tensor]:
    """Safe occupancy check with bounds validation - optimized single bounds check - supports scalar and batch mode"""
    if torch.is_tensor(coord) and coord.ndim > 1:
        # Batch mode
        results = []
        for i in range(coord.shape[0]):
            single_coord = tuple(coord[i].tolist())
            results.append(is_occupied_safe(cache_manager, single_coord))
        return torch.tensor(results, dtype=torch.bool)
    else:
        # Scalar mode
        if torch.is_tensor(coord):
            coord = tuple(coord.tolist())
        return in_bounds(coord) and cache_manager.get_piece(coord) is not None


def validate_single_cache_manager(state: 'GameState') -> bool:
    """
    Validate that only ONE cache manager exists and is used consistently.
    Returns True if valid, raises exception if inconsistent.
    """
    cache_manager = state.cache_manager

    # Check board reference
    if state.board.cache_manager is not cache_manager:
        raise RuntimeError(
            f"CACHE DESYNC: Board has different cache manager!\n"
            f"State cache manager: {id(cache_manager)}\n"
            f"Board cache manager: {id(state.board.cache_manager)}"
        )

    # Check occupancy cache
    if cache_manager.occupancy._manager_id != id(cache_manager):
        raise RuntimeError(
            f"CACHE DESYNC: Occupancy cache has wrong manager ID!\n"
            f"Expected: {id(cache_manager)}\n"
            f"Got: {cache_manager.occupancy._manager_id}"
        )

    # Check effect caches
    for cache_name, cache in [
        ('aura', cache_manager.aura_cache),
        ('trailblaze', cache_manager.trailblaze_cache),
        ('geomancy', cache_manager.geomancy_cache),
        ('attacks', cache_manager.attacks_cache)
    ]:
        if hasattr(cache, '_cache_manager') and cache._cache_manager is not cache_manager:
            raise RuntimeError(
                f"CACHE DESYNC: {cache_name} cache has different manager!\n"
                f"Expected: {id(cache_manager)}\n"
                f"Got: {id(cache._cache_manager)}"
            )

    # Check move cache
    if cache_manager._move_cache and hasattr(cache_manager._move_cache, '_cache_manager'):
        if cache_manager._move_cache._cache_manager is not cache_manager:
            raise RuntimeError(
                f"CACHE DESYNC: Move cache has different manager!\n"
                f"Expected: {id(cache_manager)}\n"
                f"Got: {id(cache_manager._move_cache._cache_manager)}"
            )

    print(f"[VALIDATION] ✓ Single cache manager verified: {id(cache_manager)}")
    return True

def batch_filter_active_coords(
    cache_manager: 'OptimizedCacheManager',
    coords: np.ndarray,
    color: Color,
    effects: List[str] = ['frozen', 'debuffed']  # e.g., ['frozen', 'geomancy']
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Batch-fetch effect masks and filter active coords.
    Returns: filtered_coords, {'frozen': mask, 'debuffed': mask, ...}
    """
    masks = {}
    for effect in effects:
        if effect == 'frozen':
            masks[effect] = cache_manager.batch_get_frozen_status(coords, color)
        elif effect == 'debuffed':
            masks[effect] = cache_manager.batch_get_debuffed_status(coords, color)
        # Add geomancy, etc., as needed
    combined_mask = np.ones(len(coords), dtype=bool)  # Start with all active
    for mask in masks.values():
        combined_mask &= ~mask  # Invert for "active"
    return coords[combined_mask], masks
