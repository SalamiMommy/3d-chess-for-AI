# movementmodifiers.py
"""Optimized central place to apply **all** movement buffs / debuffs to raw sliders."""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass
from enum import Enum
import time

import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.common.common import (
    add_coords,
    euclidean_distance,
    manhattan,
    get_path_squares,
    in_bounds_vectorised,
    SIZE,
    extend_move_range,
    extract_directions_and_steps_vectorized,
    StatsTracker,
    measure_time_ms,
    get_player_pieces,
    rebuild_moves_from_directions,
    fallback_mode,
    filter_valid_coords,
)
from game3d.movement.movepiece import Move

BOARD_SIZE = SIZE


@dataclass(slots=True)
class MovementEffectStats(StatsTracker):
    buffs_applied: int = 0
    debuffs_applied: int = 0
    directions_filtered: int = 0
    geomancy_blocks: int = 0
    wall_captures_prevented: int = 0


_STATS = MovementEffectStats()


class MovementEffectType(Enum):
    DEBUFF_RANGE = "debuff_range"
    GEOMANCY_BLOCK = "geomancy_block"
    WALL_CAPTURE_RESTRICTION = "wall_capture_restriction"
    FREEZE = "freeze"
    SLOW = "slow"


EFFECT_PRIORITIES = {
    MovementEffectType.FREEZE: 100,
    MovementEffectType.GEOMANCY_BLOCK: 90,
    MovementEffectType.WALL_CAPTURE_RESTRICTION: 80,
    MovementEffectType.DEBUFF_RANGE: 60,
    MovementEffectType.SLOW: 50,
}


def apply_movement_effects(
    state,
    start: Tuple[int, int, int],
    raw_directions: List[Tuple[int, int, int]],
    max_steps: int,
    piece_type: Optional[PieceType] = None,
    cache_manager=None,
    *,
    current_ply: int,
) -> Tuple[List[Tuple[int, int, int]], int]:
    with measure_time_ms() as elapsed_ms:
        _STATS.total_calls += 1

        if cache_manager is None:
            raise ValueError("cache_manager is required")

        # Frozen square ⇒ no moves
        x, y, z = start
        if cache_manager.move._frozen_bitmap[z, y, x]:
            _STATS.debuffs_applied += 1
            return [], 0

        directions = raw_directions.copy()
        current_max_steps = max_steps

        sorted_effects = sorted(EFFECT_PRIORITIES.items(), key=lambda x: x[1], reverse=True)
        for effect_type, _ in sorted_effects:
            if effect_type == MovementEffectType.FREEZE:
                continue
            elif effect_type == MovementEffectType.SLOW:
                if _has_slow_effect(cache_manager, start, state.color):
                    current_max_steps = max(1, current_max_steps - 1)
                    _STATS.debuffs_applied += 1
            elif effect_type == MovementEffectType.GEOMANCY_BLOCK:
                directions = _filter_geomancy_blocked_directions(
                    directions, start, current_max_steps, cache_manager, state, current_ply
                )
            elif effect_type == MovementEffectType.WALL_CAPTURE_RESTRICTION:
                directions = _filter_wall_capture_restrictions(
                    directions, start, current_max_steps, cache_manager, state
                )
                _STATS.wall_captures_prevented += len(raw_directions) - len(directions)

        _STATS.update_average(elapsed_ms)

        return directions, current_max_steps


def _has_slow_effect(cache_manager, start: Tuple[int, int, int], color: Color) -> bool:
    return cache_manager.is_movement_debuffed(start, color)


def _filter_geomancy_blocked_directions(
    directions: List[Tuple[int, int, int]],
    start: Tuple[int, int, int],
    max_steps: int,
    cache_manager,
    state,
    current_ply: int,
) -> List[Tuple[int, int, int]]:
    """Remove directions whose landing square is geomancy-blocked."""
    dir_arr = np.array(directions)
    end_arr = np.array(start) + dir_arr * max_steps

    in_b = in_bounds_vectorised(end_arr)

    # Only query blocked for in-bounds ends to avoid IndexError
    valid_ends = end_arr[in_b]
    blocked = np.full(len(end_arr), False)  # Default: not blocked
    if len(valid_ends) > 0:
        blocked_valid = np.array([
            cache_manager.is_geomancy_blocked(tuple(e), current_ply)
            for e in valid_ends
        ])
        blocked[in_b] = blocked_valid

    valid_mask = in_b & (~blocked)
    _STATS.geomancy_blocks += len(directions) - np.sum(valid_mask)
    return dir_arr[valid_mask].tolist()

def _filter_wall_capture_restrictions(
    directions: List[Tuple[int, int, int]],
    start: Tuple[int, int, int],
    max_steps: int,
    cache_manager,
    state,
) -> List[Tuple[int, int, int]]:
    """Drop every direction that would capture a Wall from a square
    that is *not* behind that Wall."""
    from game3d.pieces.pieces.wall import can_capture_wall

    victim = cache_manager.occupancy.get(start)
    if not (victim and victim.ptype == PieceType.WALL):
        return directions  # Not a wall – nothing to restrict

    filtered = []
    for d in directions:
        target_sq = add_coords(start, d)
        if can_capture_wall(attacker_sq=target_sq, wall_anchor=start):
            filtered.append(d)  # Behind the wall – capture allowed
    return filtered


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
    to_coords = filter_valid_coords(to_coords)
    captures = captures[:len(to_coords)]
    if len(to_coords) == 0:
        return []

    # 1.  Bounds and frozen filter -----------------------------------------
    to_coords = filter_valid_coords(to_coords)          # ndarray (N,3)
    # Defensive check
    assert np.all(to_coords >= 0) and np.all(to_coords < 9), f"Out-of-bounds after filter: {to_coords}"
    captures  = captures[:len(to_coords)]
    if len(to_coords) == 0:
        return []

    frozen_mask = ~np.array([cache_manager.is_frozen(tuple(c), color) for c in to_coords])

    # keep only un-frozen targets
    to_coords = [tuple(int(c) for c in row) for row in to_coords[frozen_mask]]
    captures  = captures[frozen_mask]

    if not to_coords:
        return []

    # 2.  Slow debuff -------------------------------------------------------
    if cache_manager.is_movement_debuffed(from_coord, color):
        new_max = max(1, int(np.max(np.abs(to_coords - np.array(from_coord)))) - 1)
        # keep only moves whose Chebyshev distance <= new_max
        keep = [max(abs(x - from_coord[0]), abs(y - from_coord[1]), abs(z - from_coord[2])) <= new_max
                for x, y, z in to_coords]
        to_coords = [c for c, k in zip(to_coords, keep) if k]
        captures = [c for c, k in zip(captures, keep) if k]

    if not to_coords:
        return []

    # 3.  Geomancy block ----------------------------------------------------
    blocked = [cache_manager.is_geomancy_blocked(c, current_ply) for c in to_coords]

    # 4.  Wall-capture restriction -----------------------------------------
    piece = cache_manager.occupancy.get(from_coord)
    if piece and piece.ptype is PieceType.WALL:
        from game3d.pieces.pieces.wall import can_capture_wall
        filtered = []
        filt_cap = []
        for c, cap in zip(to_coords, captures):
            if can_capture_wall(attacker_sq=c, wall_anchor=from_coord):
                filtered.append(c)
                filt_cap.append(cap)
        to_coords, captures = filtered, filt_cap

    # 6.  Build Move objects -------------------------------------------------
    return Move.create_batch(from_coord,
                             np.array(to_coords, dtype=np.int32),
                             np.array(captures, dtype=bool),
                             debuffed=debuffed)


def _extend_move_range(m: "Move", start: Tuple[int, int, int], state) -> List["Move"]:
    return extend_move_range(m, start)
