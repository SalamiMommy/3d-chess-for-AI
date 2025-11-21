# geomancer.py - FULLY NUMPY NATIVE (NO MOVE_POOL)
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from game3d.common.shared_types import (
    Color, PieceType, Result,
    COORD_DTYPE, INDEX_DTYPE, BOOL_DTYPE, COLOR_DTYPE, PIECE_TYPE_DTYPE, FLOAT_DTYPE,
    RADIUS_3_OFFSETS, get_empty_coord_batch
)
from game3d.common.registry import register
from game3d.pieces.pieces.kinglike import generate_king_moves
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.coord_utils import in_bounds_vectorized, CoordinateUtils

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState


def block_candidates_numpy(
    cache_manager: 'OptimizedCacheManager',
    mover_color: 'Color',
) -> np.ndarray:
    """
    Return empty squares that <mover_color> may block via geomancy this turn.
    Returns array of shape (N, 3).
    """
    # Get all geomancer coordinates using vectorized filtering
    pieces = cache_manager.occupancy_cache.get_positions(mover_color)
    if pieces.shape[0] == 0:
        return get_empty_coord_batch()

    # Filter for geomancers only
    geomancer_coords = []
    for coord in pieces:
        piece_data = cache_manager.occupancy_cache.get(coord)
        if piece_data and piece_data["piece_type"] == PieceType.GEOMANCER:
            geomancer_coords.append(coord)

    geomancer_coords = np.array(geomancer_coords, dtype=COORD_DTYPE)
    if geomancer_coords.shape[0] == 0:
        return get_empty_coord_batch()

    offsets = np.asarray(RADIUS_3_OFFSETS, dtype=COORD_DTYPE)

    # VECTORIZED: (G, 1, 3) + (1, O, 3) → (G, O, 3)
    all_targets = geomancer_coords[:, np.newaxis, :] + offsets[np.newaxis, :, :]

    # Flatten for batch processing: (G*O, 3)
    flat_targets = all_targets.reshape(-1, 3)

    # Vectorized bounds check
    valid_mask = in_bounds_vectorized(flat_targets)
    bounded_targets = flat_targets[valid_mask]

    if bounded_targets.shape[0] == 0:
        return get_empty_coord_batch()

    # Vectorized occupancy check using occupancy_cache directly
    empty_mask = np.array([
        cache_manager.occupancy_cache.get(target) is None for target in bounded_targets
    ], dtype=BOOL_DTYPE)

    empty_targets = bounded_targets[empty_mask]

    if empty_targets.shape[0] == 0:
        return get_empty_coord_batch()

    return np.unique(empty_targets, axis=0)


def generate_geomancer_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate geomancer moves: normal king moves + geomancy placement moves."""
    start = pos.astype(COORD_DTYPE)

    # Generate king moves for piece movement (returns np.ndarray)
    king_moves = generate_king_moves(cache_manager, color, start)
    
    move_arrays = []
    if king_moves.size > 0:
        move_arrays.append(king_moves)

    # VECTORIZED: Generate all radius-3 targets at once
    offsets = np.asarray(RADIUS_3_OFFSETS, dtype=COORD_DTYPE)
    targets = start + offsets

    # Vectorized bounds and occupancy filtering
    valid_mask = in_bounds_vectorized(targets)
    valid_targets = targets[valid_mask]

    if valid_targets.shape[0] == 0:
        return king_moves

    # Vectorized occupancy check using occupancy_cache directly
    empty_mask = np.array([
        cache_manager.occupancy_cache.get(t) is None for t in valid_targets
    ], dtype=BOOL_DTYPE)

    geom_targets = valid_targets[empty_mask]

    if geom_targets.shape[0] == 0:
        return king_moves

    # Create geomancy moves as numpy array
    n_geom = geom_targets.shape[0]
    geom_moves = np.empty((n_geom, 6), dtype=COORD_DTYPE)
    geom_moves[:, 0:3] = start
    geom_moves[:, 3:6] = geom_targets
    
    move_arrays.append(geom_moves)
    
    return np.concatenate(move_arrays)


@register(PieceType.GEOMANCER)
def geomancer_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_geomancer_moves(state.cache_manager, state.color, pos)


__all__ = ["generate_geomancer_moves", "block_candidates_numpy"]
