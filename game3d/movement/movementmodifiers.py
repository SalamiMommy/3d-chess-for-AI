# movementmodifiers.py - CONSOLIDATED AUTODETECT VERSION
"""Centralized movement modifier application with autodetection of scalar vs batch mode."""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any, Set, Union
from dataclasses import dataclass
from enum import Enum
import time

import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.common.coord_utils import add_coords, euclidean_distance, manhattan, get_path_squares, in_bounds_vectorised, filter_valid_coords
from game3d.common.constants import SIZE
from game3d.common.move_utils import extend_move_range, rebuild_moves_from_directions, extract_directions_and_steps_vectorized
from game3d.common.piece_utils import get_player_pieces
from game3d.movement.movepiece import Move
from game3d.common.debug_utils import StatsTracker, measure_time_ms, fallback_mode

@dataclass(slots=True)
class MovementEffectStats:
    """Standalone MovementEffectStats without inheritance"""
    total_calls: int = 0
    average_time_ms: float = 0.0
    buffs_applied: int = 0
    debuffs_applied: int = 0
    directions_filtered: int = 0
    geomancy_blocks: int = 0
    wall_captures_prevented: int = 0

    def update_average(self, elapsed_ms: float) -> None:
        self.total_calls += 1
        self.average_time_ms = (
            (self.average_time_ms * (self.total_calls - 1) + elapsed_ms) / self.total_calls
            if self.total_calls > 0 else elapsed_ms
        )

    def get_stats(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def reset(self) -> None:
        self.total_calls = 0
        self.average_time_ms = 0.0
        self.buffs_applied = 0
        self.debuffs_applied = 0
        self.directions_filtered = 0
        self.geomancy_blocks = 0
        self.wall_captures_prevented = 0

_STATS = MovementEffectStats()

class MovementEffectType(Enum):
    DEBUFF_RANGE = "debuff_range"
    GEOMANCY_BLOCK = "geomancy_block"
    WALL_CAPTURE_RESTRICTION = "wall_capture_restriction"
    FREEZE = "freeze"
    SLOW = "slow"

def apply_movement_effects(
    state,
    starts: np.ndarray,
    raw_directions_batch: List[List[Tuple[int, int, int]]],
    max_steps_list: List[int],
    piece_types: Optional[List[PieceType]] = None,
    cache_manager=None,
    *,
    current_ply: int,
) -> Tuple[List[List[Tuple[int, int, int]]], List[int]]:
    """Batch apply movement effects to multiple pieces."""
    with measure_time_ms() as elapsed_ms:
        _STATS.total_calls += 1

        if cache_manager is None:
            raise ValueError("cache_manager is required")

        batch_size = len(starts)
        all_filtered_directions = []
        all_current_max_steps = []

        frozen_mask = cache_manager.batch_get_frozen_status(starts, state.color)

        for i in range(batch_size):
            start = tuple(starts[i])
            raw_directions = raw_directions_batch[i]
            max_steps = max_steps_list[i]

            if frozen_mask[i]:
                all_filtered_directions.append([])
                all_current_max_steps.append(0)
                continue

            directions = raw_directions.copy()
            current_max_steps = max_steps

            if cache_manager.is_movement_debuffed(start, state.color):
                current_max_steps = max(1, current_max_steps - 1)
                _STATS.debuffs_applied += 1

            directions = _filter_geomancy_blocked_directions_batch(
                directions, start, current_max_steps, cache_manager, state, current_ply
            )

            directions = _filter_wall_capture_restrictions_batch(
                directions, start, current_max_steps, cache_manager, state
            )

            all_filtered_directions.append(directions)
            all_current_max_steps.append(current_max_steps)

        _STATS.update_average(elapsed_ms())
        return all_filtered_directions, all_current_max_steps

def _filter_geomancy_blocked_directions_batch(
    directions: List[Tuple[int, int, int]],
    start: Tuple[int, int, int],
    max_steps: int,
    cache_manager,
    state,
    current_ply: int,
) -> List[Tuple[int, int, int]]:
    if not directions:
        return []

    dir_arr = np.array(directions)
    end_arr = np.array(start) + dir_arr * max_steps
    in_b = in_bounds_vectorised(end_arr)
    valid_ends = end_arr[in_b]

    blocked = np.full(len(end_arr), False)
    if len(valid_ends) > 0:
        blocked_valid = cache_manager.batch_get_geomancy_blocked(valid_ends, current_ply)
        blocked[in_b] = blocked_valid

    valid_mask = in_b & (~blocked)
    _STATS.geomancy_blocks += len(directions) - np.sum(valid_mask)
    return dir_arr[valid_mask].tolist()

def _filter_wall_capture_restrictions_batch(
    directions: List[Tuple[int, int, int]],
    start: Tuple[int, int, int],
    max_steps: int,
    cache_manager,
    state,
) -> List[Tuple[int, int, int]]:
    from game3d.pieces.pieces.wall import can_capture_wall_batch

    piece = cache_manager.occupancy_cache.get(start)
    if not (piece and piece.ptype == PieceType.WALL):
        return directions

    dir_array = np.array(directions)
    target_sqs = np.array([add_coords(start, tuple(d)) for d in dir_array])
    valid_mask = can_capture_wall_batch(target_sqs.tolist(), start)
    return dir_array[valid_mask].tolist()

# ================================
# UNIFIED MODIFIER ENTRY POINT
# ================================

def modify_raw_moves_unified(
    from_coords: Union[Tuple[int, int, int], np.ndarray],
    to_coords_or_batch: Union[np.ndarray, List[np.ndarray]],
    captures_or_batch: Union[np.ndarray, List[np.ndarray]],
    color_or_colors: Union[Color, np.ndarray],
    cache_manager,
    debuffed_or_mask: Union[bool, np.ndarray] = False,
    *,
    current_ply: int,
) -> Union[List[Move], List[List[Move]]]:
    """
    Unified movement modifier that autodetects scalar vs batch mode.

    Scalar mode (single piece):
        from_coords: (x, y, z)
        to_coords_or_batch: np.ndarray of shape (N, 3)
        captures_or_batch: np.ndarray of shape (N,)
        color_or_colors: Color
        debuffed_or_mask: bool

    Batch mode (multiple pieces):
        from_coords: np.ndarray of shape (B, 3)
        to_coords_or_batch: List[np.ndarray] (length B)
        captures_or_batch: List[np.ndarray] (length B)
        color_or_colors: np.ndarray of shape (B,) — assumed same color
        debuffed_or_mask: np.ndarray of shape (B,)

    Returns:
        Scalar: List[Move]
        Batch: List[List[Move]]
    """
    # Autodetect mode
    if isinstance(from_coords, tuple):
        # Scalar mode
        return _modify_scalar(
            from_coord=from_coords,
            to_coords=to_coords_or_batch,
            captures=captures_or_batch,
            color=color_or_colors,
            cache_manager=cache_manager,
            debuffed=debuffed_or_mask,
            current_ply=current_ply,
        )
    else:
        # Batch mode
        return _modify_batch(
            from_coords=from_coords,
            to_coords_batch=to_coords_or_batch,
            captures_batch=captures_or_batch,
            colors=color_or_colors,
            cache_manager=cache_manager,
            debuffed_mask=debuffed_or_mask,
            current_ply=current_ply,
        )

def _modify_scalar(
    from_coord: Tuple[int, int, int],
    to_coords: np.ndarray,
    captures: np.ndarray,
    color: Color,
    cache_manager,
    debuffed: bool,
    current_ply: int,
) -> List[Move]:
    if len(to_coords) == 0:
        return []

    # 1. Bounds filter
    to_coords = filter_valid_coords(to_coords)
    captures = captures[:len(to_coords)]
    if len(to_coords) == 0:
        return []

    # 2. Frozen check (skip if frozen)
    if cache_manager.is_frozen(from_coord, color):
        return []

    # 3. Slow debuff: reduce range
    if debuffed:
        from_arr = np.array(from_coord)
        distances = np.max(np.abs(to_coords - from_arr), axis=1)
        new_max = max(1, int(np.max(distances)) - 1) if len(distances) > 0 else 1
        keep_mask = distances <= new_max
        to_coords = to_coords[keep_mask]
        captures = captures[keep_mask]
        if len(to_coords) == 0:
            return []

    # 4. Geomancy block
    blocked_mask = np.array([
        not cache_manager.is_geomancy_blocked(tuple(c), current_ply) for c in to_coords
    ], dtype=bool)
    to_coords = to_coords[blocked_mask]
    captures = captures[blocked_mask]
    if len(to_coords) == 0:
        return []

    # 5. Wall capture restriction
    piece = cache_manager.occupancy_cache.get(from_coord)
    if piece and piece.ptype == PieceType.WALL:
        from game3d.pieces.pieces.wall import can_capture_wall_batch
        valid_mask = can_capture_wall_batch(to_coords.tolist(), from_coord)
        to_coords = to_coords[valid_mask]
        captures = captures[valid_mask]

    if len(to_coords) == 0:
        return []

    return Move.create_batch(
        from_coord,
        np.array(to_coords, dtype=np.int32),
        np.array(captures, dtype=bool),
        debuffed=debuffed
    )

def _modify_batch(
    from_coords: np.ndarray,
    to_coords_batch: List[np.ndarray],
    captures_batch: List[np.ndarray],
    colors: np.ndarray,
    cache_manager,
    debuffed_mask: np.ndarray,
    current_ply: int,
) -> List[List[Move]]:
    batch_size = len(from_coords)
    if batch_size == 0:
        return []

    # Ensure all arrays have consistent shapes, even when empty
    to_coords_batch = [arr if arr.ndim == 2 and arr.shape[1] == 3 else np.empty((0, 3), dtype=np.int32)
                      for arr in to_coords_batch]
    captures_batch = [arr if arr.ndim == 1 else np.empty(0, dtype=bool)
                     for arr in captures_batch]

    # Precompute global masks
    try:
        all_to_coords = np.vstack(to_coords_batch) if any(len(arr) > 0 for arr in to_coords_batch) else np.empty((0, 3))
        all_captures = np.concatenate(captures_batch) if any(len(arr) > 0 for arr in captures_batch) else np.empty(0, dtype=bool)
    except ValueError as e:
        # Fallback: process each piece individually if concatenation fails
        return _modify_batch_fallback(from_coords, to_coords_batch, captures_batch, colors,
                                    cache_manager, debuffed_mask, current_ply)

    if len(all_to_coords) == 0:
        return [[] for _ in range(batch_size)]

    bounds_valid = in_bounds_vectorised(all_to_coords)
    frozen_mask = cache_manager.batch_get_frozen_status(from_coords, colors[0])
    geomancy_blocked = cache_manager.batch_get_geomancy_blocked(all_to_coords, current_ply)

    all_modified_moves = []
    move_idx = 0

    for i in range(batch_size):
        to_coords = to_coords_batch[i]
        captures = captures_batch[i]
        from_coord = tuple(from_coords[i])
        debuffed = debuffed_mask[i]

        if len(to_coords) == 0:
            all_modified_moves.append([])
            continue

        n = len(to_coords)
        local_bounds = bounds_valid[move_idx:move_idx + n]
        local_geomancy_ok = ~geomancy_blocked[move_idx:move_idx + n]
        move_idx += n

        if frozen_mask[i]:
            all_modified_moves.append([])
            continue

        valid_mask = local_bounds & local_geomancy_ok
        valid_to = to_coords[valid_mask]
        valid_cap = captures[valid_mask]

        if len(valid_to) == 0:
            all_modified_moves.append([])
            continue

        # Apply slow debuff
        if debuffed:
            from_arr = np.array(from_coord)
            distances = np.max(np.abs(valid_to - from_arr), axis=1)
            new_max = max(1, int(np.max(distances)) - 1) if len(distances) > 0 else 1
            keep = distances <= new_max
            valid_to = valid_to[keep]
            valid_cap = valid_cap[keep]

        if len(valid_to) == 0:
            all_modified_moves.append([])
            continue

        # Wall restriction (per piece)
        piece = cache_manager.occupancy_cache.get(from_coord)
        if piece and piece.ptype == PieceType.WALL:
            from game3d.pieces.pieces.wall import can_capture_wall_batch
            valid_mask = can_capture_wall_batch(valid_to.tolist(), from_coord)
            valid_to = valid_to[valid_mask]
            valid_cap = valid_cap[valid_mask]

        if len(valid_to) > 0:
            moves = Move.create_batch(
                from_coord,
                np.array(valid_to, dtype=np.int32),
                np.array(valid_cap, dtype=bool),
                debuffed=debuffed
            )
            all_modified_moves.append(moves)
        else:
            all_modified_moves.append([])

    return all_modified_moves


def _modify_batch_fallback(
    from_coords: np.ndarray,
    to_coords_batch: List[np.ndarray],
    captures_batch: List[np.ndarray],
    colors: np.ndarray,
    cache_manager,
    debuffed_mask: np.ndarray,
    current_ply: int,
) -> List[List[Move]]:
    """Fallback implementation that processes each piece individually"""
    batch_size = len(from_coords)
    all_modified_moves = []

    for i in range(batch_size):
        from_coord = tuple(from_coords[i])
        to_coords = to_coords_batch[i] if i < len(to_coords_batch) else np.empty((0, 3))
        captures = captures_batch[i] if i < len(captures_batch) else np.empty(0, dtype=bool)
        color = colors[i] if i < len(colors) else colors[0]
        debuffed = debuffed_mask[i] if i < len(debuffed_mask) else False

        # Ensure proper array shapes
        if to_coords.ndim != 2 or to_coords.shape[1] != 3:
            to_coords = np.empty((0, 3), dtype=np.int32)
        if captures.ndim != 1:
            captures = np.empty(0, dtype=bool)

        # Use scalar processing for this single piece
        moves = _modify_scalar(
            from_coord=from_coord,
            to_coords=to_coords,
            captures=captures,
            color=color,
            cache_manager=cache_manager,
            debuffed=debuffed,
            current_ply=current_ply
        )
        all_modified_moves.append(moves)

    return all_modified_moves
# ================================
# BACKWARD-COMPATIBLE WRAPPERS
# ================================

def modify_raw_moves(
    from_coord: Tuple[int, int, int],
    to_coords: np.ndarray,
    captures: np.ndarray,
    color: Color,
    cache_manager,
    debuffed: bool = False,
    *,
    current_ply: int,
) -> List[Move]:
    return modify_raw_moves_unified(
        from_coords=from_coord,
        to_coords_or_batch=to_coords,
        captures_or_batch=captures,
        color_or_colors=color,
        cache_manager=cache_manager,
        debuffed_or_mask=debuffed,
        current_ply=current_ply,
    )

def modify_raw_moves_batch(
    from_coords: np.ndarray,
    to_coords_batch: List[np.ndarray],
    captures_batch: List[np.ndarray],
    colors: np.ndarray,
    cache_manager,
    debuffed_mask: np.ndarray,
    current_ply: int,
) -> List[List[Move]]:
    return modify_raw_moves_unified(
        from_coords=from_coords,
        to_coords_or_batch=to_coords_batch,
        captures_or_batch=captures_batch,
        color_or_colors=colors,
        cache_manager=cache_manager,
        debuffed_or_mask=debuffed_mask,
        current_ply=current_ply,
    )

# ================================
# HELPER FUNCTIONS (unchanged)
# ================================

def batch_validate_frozen_status(
    cache_manager,
    coords: np.ndarray,
    color: Color
) -> np.ndarray:
    return np.array([
        not cache_manager.is_frozen(tuple(coord), color)
        for coord in coords
    ], dtype=bool)

def batch_validate_debuffed_status(
    cache_manager,
    coords: np.ndarray,
    color: Color
) -> np.ndarray:
    return np.array([
        cache_manager.is_movement_debuffed(tuple(coord), color)
        for coord in coords
    ], dtype=bool)
