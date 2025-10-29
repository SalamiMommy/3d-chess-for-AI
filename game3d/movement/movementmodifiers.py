# movementmodifiers.py - NUMPY VERSION
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
    starts: np.ndarray,  # Changed to np.ndarray (N,3)
    raw_directions_batch: List[np.ndarray],  # Each (M,3)
    max_steps_list: np.ndarray,
    piece_types: Optional[List[PieceType]] = None,
    cache_manager=None,
    *,
    current_ply: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Batch apply movement effects to multiple pieces - fully vectorized."""
    with measure_time_ms() as elapsed_ms:
        _STATS.total_calls += 1

        if cache_manager is None:
            raise ValueError("cache_manager is required")

        batch_size = starts.shape[0]
        if batch_size == 0:
            return [], []

        # Vectorize: Flatten all directions and steps across batch
        all_raw_directions = np.vstack(raw_directions_batch)
        all_max_steps = []
        direction_indices = []  # To reconstruct per-piece
        cumsum = 0
        for dirs, max_steps in zip(raw_directions_batch, max_steps_list):
            n_dirs = len(dirs)
            all_max_steps.extend([max_steps] * n_dirs)
            direction_indices.extend([cumsum + i for i in range(n_dirs)])
            cumsum += n_dirs

        if len(all_raw_directions) == 0:
            return [[] for _ in range(batch_size)], np.zeros(batch_size, dtype=int)

        dir_arr = all_raw_directions  # (total_dirs, 3)
        max_steps_arr = np.array(all_max_steps)  # (total_dirs,)

        # Per-piece starts: Reconstruct with indices
        piece_indices = np.repeat(np.arange(batch_size), [len(dirs) for dirs in raw_directions_batch])
        starts_per_dir = starts[piece_indices]  # (total_dirs, 3)

        end_arr = starts_per_dir + dir_arr * max_steps_arr[:, np.newaxis]  # Vectorized ends
        in_bounds = in_bounds_vectorised(end_arr)
        valid_ends = end_arr[in_bounds]

        # Vectorized filters
        frozen_mask = cache_manager.batch_get_frozen_status(starts, state.color)
        active_pieces = ~frozen_mask  # (batch_size,)

        # Debuff: Apply per-piece, then broadcast
        debuffed_max_steps = max_steps_arr.copy()
        debuffed_pieces = cache_manager.batch_get_debuffed_status(starts, state.color)
        debuff_active = debuffed_pieces[piece_indices]
        debuffed_max_steps[debuff_active] = np.maximum(1, debuffed_max_steps[debuff_active] - 1)
        _STATS.debuffs_applied += np.sum(debuff_active)

        # Geomancy: Vectorized block check
        if len(valid_ends) > 0:
            blocked = cache_manager.batch_get_geomancy_blocked(valid_ends, current_ply)
            # Map back to full mask
            full_blocked = np.full(len(end_arr), False)
            full_blocked[in_bounds] = blocked
        else:
            full_blocked = np.full(len(end_arr), False)

        # Wall: Only for wall pieces - vectorized per direction
        if piece_types is not None:
            wall_pieces = np.array([pt == PieceType.WALL for pt in piece_types])
        else:
            wall_pieces = np.full(batch_size, False)

        valid_wall_mask = np.ones(len(dir_arr), dtype=bool)
        if np.any(wall_pieces):
            wall_indices = np.where(wall_pieces)[0]
            wall_mask = np.isin(piece_indices, wall_indices)
            if np.any(wall_mask):
                from game3d.pieces.pieces.wall import can_capture_wall_batch
                wall_starts = starts_per_dir[wall_mask]
                wall_dirs = dir_arr[wall_mask]
                wall_max_steps = debuffed_max_steps[wall_mask]
                wall_ends = wall_starts + wall_dirs * wall_max_steps[:, np.newaxis]
                wall_valid = can_capture_wall_batch(wall_ends.tolist(), tuple(starts[wall_indices[0]]))  # Assume single start for batch
                valid_wall_mask[wall_mask] = wall_valid
                _STATS.wall_captures_prevented += np.sum(~wall_valid)

        # Combined mask: in_bounds & ~blocked & active_pieces[piece_idx] & valid_wall_mask
        piece_active_mask = active_pieces[piece_indices]
        final_mask = in_bounds & ~full_blocked & piece_active_mask & valid_wall_mask

        # Reconstruct per-piece
        filtered_directions = [[] for _ in range(batch_size)]
        current_max_steps_list = np.zeros(batch_size, dtype=int)
        for i in range(batch_size):
            piece_mask = piece_indices == i
            piece_dirs = dir_arr[piece_mask & final_mask]
            filtered_directions[i] = piece_dirs
            current_max_steps_list[i] = max_steps_list[i] if active_pieces[i] else 0

        _STATS.geomancy_blocks += np.sum(full_blocked)
        _STATS.directions_filtered += np.sum(~final_mask)
        _STATS.update_average(elapsed_ms())
        return filtered_directions, current_max_steps_list

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

    piece = cache_manager.occupancy.get(start)
    if piece.ptype != PieceType.WALL:
        return directions

    end_arr = np.array(start) + np.array(directions) * max_steps
    valid_mask = can_capture_wall_batch(end_arr.tolist(), start)
    return [d for i, d in enumerate(directions) if valid_mask[i]]

def modify_raw_moves_unified(
    from_coords: Union[Tuple[int, int, int], np.ndarray],
    to_coords_or_batch: Union[np.ndarray, List[np.ndarray]],
    captures_or_batch: Union[np.ndarray, List[np.ndarray]],
    color_or_colors: Union[Color, np.ndarray],
    cache_manager,
    debuffed_or_mask: Union[bool, np.ndarray],
    current_ply: int,
) -> Union[List[Move], List[List[Move]]]:
    """
    Unified function that handles both scalar and batch mode based on input types.
    Autodetects mode and dispatches to appropriate implementation.
    """
    # Detect if we're in batch mode
    is_batch = (isinstance(from_coords, np.ndarray) and
                from_coords.ndim == 2 and
                from_coords.shape[1] == 3 and
                isinstance(to_coords_or_batch, list) and
                len(to_coords_or_batch) > 0 and
                isinstance(to_coords_or_batch[0], np.ndarray))

    if is_batch:
        # Batch mode
        return _modify_batch(
            from_coords=from_coords,
            to_coords_batch=to_coords_or_batch,
            captures_batch=captures_or_batch,
            colors=color_or_colors,
            cache_manager=cache_manager,
            debuffed_mask=debuffed_or_mask,
            current_ply=current_ply
        )
    else:
        # Scalar mode - convert to batch of size 1 and extract result
        from_coords_batch = np.array([from_coords])
        to_coords_batch = [np.array(to_coords_or_batch)]
        captures_batch = [np.array(captures_or_batch)]
        colors_batch = np.array([color_or_colors.value])
        debuffed_mask = np.array([debuffed_or_mask])

        batch_result = _modify_batch(
            from_coords=from_coords_batch,
            to_coords_batch=to_coords_batch,
            captures_batch=captures_batch,
            colors=colors_batch,
            cache_manager=cache_manager,
            debuffed_mask=debuffed_mask,
            current_ply=current_ply
        )

        return batch_result[0] if batch_result else []


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
    all_modified_moves = []

    # FIX: Ensure all arrays in to_coords_batch have consistent shape
    to_coords_batch_fixed = []
    captures_batch_fixed = []

    for i, (to_coords, captures) in enumerate(zip(to_coords_batch, captures_batch)):
        # Ensure to_coords has shape (n, 3) even when empty
        if len(to_coords) == 0:
            to_coords_batch_fixed.append(np.empty((0, 3), dtype=np.int32))
            captures_batch_fixed.append(np.empty(0, dtype=bool))
        else:
            # Ensure proper shape and dtype
            if to_coords.ndim == 1 and to_coords.shape[0] == 3:
                # Single coordinate case - reshape to (1, 3)
                to_coords_batch_fixed.append(to_coords.reshape(1, -1).astype(np.int32))
                captures_batch_fixed.append(np.array([captures], dtype=bool) if np.isscalar(captures) else captures.astype(bool))
            else:
                to_coords_batch_fixed.append(to_coords.astype(np.int32))
                captures_batch_fixed.append(captures.astype(bool))

    to_coords_batch = to_coords_batch_fixed
    captures_batch = captures_batch_fixed

    # Mega-vectorized preprocessing
    lengths = [len(tc) for tc in to_coords_batch]

    # Only proceed if we have any coordinates to process
    if sum(lengths) == 0:
        return [[] for _ in range(batch_size)]

    cum_lengths = np.cumsum([0] + lengths)
    all_to_coords = np.vstack(to_coords_batch)
    all_captures = np.concatenate(captures_batch)

    bounds_valid = in_bounds_vectorised(all_to_coords)
    geomancy_blocked = cache_manager.batch_get_geomancy_blocked(all_to_coords, current_ply)
    frozen_mask = cache_manager.batch_get_frozen_status(from_coords, colors[0])  # Assume same color

    # Check for no debuffs/geomancy to skip
    has_debuff = np.any(debuffed_mask)
    has_geomancy = np.any(geomancy_blocked)

    if not has_debuff and not has_geomancy:
        move_idx = 0
        for i in range(batch_size):
            n = lengths[i]
            if n == 0:
                all_modified_moves.append([])
                continue

            valid_to = all_to_coords[move_idx:move_idx + n]
            valid_cap = all_captures[move_idx:move_idx + n]
            # Single create_batch per piece (vectorized internally)
            moves = Move.create_batch(tuple(from_coords[i]), valid_to, valid_cap, debuffed=debuffed_mask[i])
            all_modified_moves.append(moves)
            move_idx += n
        return all_modified_moves

    move_idx = 0
    for i in range(batch_size):
        from_coord = tuple(from_coords[i])
        to_coords = to_coords_batch[i]
        captures = captures_batch[i]
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

# FIX: Add the missing _modify_scalar function
def _modify_scalar(
    from_coord: Tuple[int, int, int],
    to_coords: np.ndarray,
    captures: np.ndarray,
    color: Color,
    cache_manager,
    debuffed: bool,
    current_ply: int,
) -> List[Move]:
    """Scalar implementation for single piece processing."""
    if len(to_coords) == 0:
        return []

    # Check if piece is frozen
    if cache_manager.is_frozen(from_coord, color):
        return []

    # Filter out-of-bounds coordinates
    bounds_valid = in_bounds_vectorised(to_coords)
    to_coords = to_coords[bounds_valid]
    captures = captures[bounds_valid]

    if len(to_coords) == 0:
        return []

    # Filter geomancy-blocked coordinates
    geomancy_blocked = cache_manager.batch_get_geomancy_blocked(to_coords, current_ply)
    valid_mask = ~geomancy_blocked
    to_coords = to_coords[valid_mask]
    captures = captures[valid_mask]

    if len(to_coords) == 0:
        return []

    # Apply slow debuff
    if debuffed:
        from_arr = np.array(from_coord)
        distances = np.max(np.abs(to_coords - from_arr), axis=1)
        new_max = max(1, int(np.max(distances)) - 1) if len(distances) > 0 else 1
        keep = distances <= new_max
        to_coords = to_coords[keep]
        captures = captures[keep]

    if len(to_coords) == 0:
        return []

    # Wall restriction
    piece = cache_manager.occupancy_cache.get(from_coord)
    if piece and piece.ptype == PieceType.WALL:
        from game3d.pieces.pieces.wall import can_capture_wall_batch
        valid_mask = can_capture_wall_batch(to_coords.tolist(), from_coord)
        to_coords = to_coords[valid_mask]
        captures = captures[valid_mask]

    if len(to_coords) == 0:
        return []

    return Move.create_batch(from_coord, to_coords, captures, debuffed=debuffed)

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
