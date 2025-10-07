"""Optimized central place to apply **all** movement buffs / debuffs to raw sliders."""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass
from enum import Enum
import time

from game3d.pieces.enums import Color, PieceType
from game3d.cache.manager import get_cache_manager
from game3d.common.common import add_coords
from game3d.geometry import euclidean_distance, manhattan
from game3d.geometry import get_path_squares  # Moved to top

# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

BOARD_SIZE = 9  # Extracted constant

@dataclass(slots=True)
class MovementEffectStats:
    """Statistics for movement effect application."""
    total_calls: int = 0
    buffs_applied: int = 0
    debuffs_applied: int = 0
    directions_filtered: int = 0
    geomancy_blocks: int = 0
    wall_captures_prevented: int = 0
    average_time_ms: float = 0.0

class MovementEffectType(Enum):
    """Types of movement effects."""
    BUFF_RANGE = "buff_range"
    DEBUFF_RANGE = "debuff_range"
    BLOCK_DIAGONAL = "block_diagonal"
    GEOMANCY_BLOCK = "geomancy_block"
    WALL_CAPTURE_RESTRICTION = "wall_capture_restriction"
    FREEZE = "freeze"
    SLOW = "slow"

# ==============================================================================
# ENHANCED MOVEMENT EFFECTS SYSTEM
# ==============================================================================

_STATS = MovementEffectStats()

# Effect priority order (higher priority effects applied first)
EFFECT_PRIORITIES = {
    MovementEffectType.FREEZE: 100,      # Highest - prevents all movement
    MovementEffectType.GEOMANCY_BLOCK: 90,
    MovementEffectType.WALL_CAPTURE_RESTRICTION: 80,
    MovementEffectType.BLOCK_DIAGONAL: 70,
    MovementEffectType.DEBUFF_RANGE: 60,
    MovementEffectType.SLOW: 50,
    MovementEffectType.BUFF_RANGE: 10,    # Lowest - applied last
}

def apply_movement_effects(
    state,
    start: Tuple[int, int, int],
    raw_directions: List[Tuple[int, int, int]],
    max_steps: int,
    piece_type: Optional[PieceType] = None,
    cache_manager: Optional = None
) -> Tuple[List[Tuple[int, int, int]], int]:
    """
    Optimized movement effects application - CORRECTED.
    Removed duplicate definition at end of file.
    """
    start_time = time.perf_counter()
    _STATS.total_calls += 1

    if cache_manager is None:
        cache_manager = state.cache

    try:
        # Quick freeze check (highest priority)
        if cache_manager.is_frozen(start, state.color):
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
            elif effect_type == MovementEffectType.BLOCK_DIAGONAL:
                if _has_diagonal_block(cache_manager, start, state.color):
                    directions = _filter_diagonal_directions(directions)
                    _STATS.directions_filtered += len(raw_directions) - len(directions)
            elif effect_type == MovementEffectType.GEOMANCY_BLOCK:
                directions = _filter_geomancy_blocked_directions(
                    directions, start, current_max_steps, cache_manager, state
                )
            elif effect_type == MovementEffectType.WALL_CAPTURE_RESTRICTION:
                directions = _filter_wall_capture_directions(
                    directions, start, state, cache_manager
                )
            elif effect_type == MovementEffectType.BUFF_RANGE:
                if cache_manager.is_movement_buffed(start, state.color):
                    current_max_steps += 1
                    _STATS.buffs_applied += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _update_stats(elapsed_ms)

        return directions, current_max_steps

    except Exception as e:
        return _fallback_movement_effects(start, raw_directions, max_steps, cache_manager, state)

def _has_slow_effect(cache_manager, start: Tuple[int, int, int], color: Color) -> bool:
    """Check if piece has slow effect."""
    return (hasattr(cache_manager, "is_movement_slowed") and
            cache_manager.is_movement_slowed(start, color))

def _has_diagonal_block(cache_manager, start: Tuple[int, int, int], color: Color) -> bool:
    """Check if diagonal movement is blocked."""
    return (hasattr(cache_manager, "is_diagonal_blocked") and
            cache_manager.is_diagonal_blocked(start, color))

def _filter_diagonal_directions(directions: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """Filter out diagonal directions (keep axis-aligned)."""
    return [d for d in directions if sum(1 for coord in d if coord != 0) <= 1]  # Updated to strictly axis-aligned

def _filter_geomancy_blocked_directions(
    directions: List[Tuple[int, int, int]],
    start: Tuple[int, int, int],
    max_steps: int,
    cache_manager,
    state
) -> List[Tuple[int, int, int]]:
    """Filter directions that would land on geomancy-blocked squares."""
    if not (hasattr(cache_manager, "is_geomancy_blocked") and
            hasattr(state, "halfmove_clock")):
        return directions

    filtered_directions = []

    for direction in directions:
        # Assume direction is unit vector; validate magnitude
        dir_magnitude = sum(abs(c) for c in direction)
        if dir_magnitude != 1:
            continue  # Skip non-unit directions

        # Calculate end position
        end_pos = add_coords(start, (direction[0] * max_steps, direction[1] * max_steps, direction[2] * max_steps))

        # Check bounds
        if not all(0 <= coord < BOARD_SIZE for coord in end_pos):
            continue

        # Check if end position is geomancy-blocked
        if not cache_manager.is_geomancy_blocked(end_pos, state.halfmove_clock):
            filtered_directions.append(direction)
        else:
            _STATS.geomancy_blocks += 1

    return filtered_directions

def _filter_wall_capture_directions(
    directions: List[Tuple[int, int, int]],
    start: Tuple[int, int, int],
    state,
    cache_manager
) -> List[Tuple[int, int, int]]:
    """Filter directions based on wall capture restrictions."""
    # Check if there's a wall at the current position
    victim = state.cache.piece_cache.get(start)
    if victim is None or victim.ptype != PieceType.WALL:
        return directions

    # Check wall capture restrictions
    if (hasattr(cache_manager, "can_capture_wall") and
        not cache_manager.can_capture_wall(start, start, state.color)):
        _STATS.wall_captures_prevented += 1
        return []  # Can't capture wall, no valid directions

    return directions

def _fallback_movement_effects(
    start: Tuple[int, int, int],
    raw_directions: List[Tuple[int, int, int]],
    max_steps: int,
    cache_manager,
    state
) -> Tuple[List[Tuple[int, int, int]], int]:
    """Fallback to original implementation - CORRECTED."""
    directions = raw_directions.copy()

    # 1. range buffs / debuffs
    if cache_manager.is_movement_buffed(start, state.color):  # Fixed state.color
        max_steps += 1

    if (hasattr(cache_manager, "is_movement_slowed") and
        cache_manager.is_movement_slowed(start, state.color)):  # Fixed state.color
        max_steps = max(1, max_steps - 1)

    # 2. direction filters
    if (hasattr(cache_manager, "is_diagonal_blocked") and
        cache_manager.is_diagonal_blocked(start, state.color)):  # Fixed state.color
        directions = [d for d in directions if sum(1 for coord in d if coord != 0) <= 1]

    # 4. geomancy blocked squares
    if hasattr(cache_manager, "is_geomancy_blocked") and hasattr(state, "halfmove_clock"):
        filtered_dirs = []
        for d in directions:
            end_pos = add_coords(start, (d[0] * max_steps, d[1] * max_steps, d[2] * max_steps))
            if all(0 <= coord < BOARD_SIZE for coord in end_pos):
                if not cache_manager.is_geomancy_blocked(end_pos, state.halfmove_clock):
                    filtered_dirs.append(d)
        directions = filtered_dirs

    # 5. wall capture rules
    victim = state.cache.piece_cache.get(start)
    if victim is not None and victim.ptype == PieceType.WALL:
        if (hasattr(cache_manager, "can_capture_wall") and
            not cache_manager.can_capture_wall(start, start, state.color)):  # Fixed state.color
            return [], max_steps

    return directions, max_steps

# ==============================================================================
# SPECIALIZED EFFECT HANDLERS
# ==============================================================================
def apply_archery_effects(
    start: Tuple[int, int, int],
    targets: List[Tuple[int, int, int]],
    state,
    cache_manager: Optional = None
) -> List[Tuple[int, int, int]]:
    """Apply archery-specific movement effects - CORRECTED."""
    if cache_manager is None:
        cache_manager = state.cache

    valid_targets = []

    for target in targets:
        distance = euclidean_distance(start, target)
        if abs(distance - 2.0) > 0.1:
            continue

        if _has_clear_line_of_sight(start, target, state):
            valid_targets.append(target)

    return valid_targets

def _has_clear_line_of_sight(
    start: Tuple[int, int, int],
    target: Tuple[int, int, int],
    state
) -> bool:
    """Check if there's clear line of sight for archery."""
    # Simplified line of sight check
    # In full implementation, this would check for blocking pieces
    path_squares = get_path_squares(start, target)

    for sq in path_squares[1:-1]:  # Exclude start and end
        if state.cache.piece_cache.get(sq) is not None:
            return False

    return True

def apply_hive_effects(
    moves: List[Tuple[int, int, int]],
    state,
    cache_manager: Optional = None
) -> List[Tuple[int, int, int]]:
    """Apply hive-specific movement effects - CORRECTED."""
    if cache_manager is None:
        cache_manager = state.cache

    valid_moves = []

    for move in moves:
        # Fixed hasattr check
        if (not hasattr(cache_manager, "is_movement_blocked_for_hive") or
            not cache_manager.is_movement_blocked_for_hive(move, state.color)):
            valid_moves.append(move)

    return valid_moves

# ==============================================================================
# PERFORMANCE MONITORING
# ==============================================================================

def _update_stats(elapsed_ms: float) -> None:
    """Update performance statistics."""
    _STATS.average_time_ms = (
        (_STATS.average_time_ms * (_STATS.total_calls - 1) + elapsed_ms) /
        _STATS.total_calls
    )

def get_movement_effects_stats() -> Dict[str, Any]:
    """Get movement effects performance statistics."""
    return {
        'total_calls': _STATS.total_calls,
        'buffs_applied': _STATS.buffs_applied,
        'debuffs_applied': _STATS.debuffs_applied,
        'directions_filtered': _STATS.directions_filtered,
        'geomancy_blocks': _STATS.geomancy_blocks,
        'wall_captures_prevented': _STATS.wall_captures_prevented,
        'average_time_ms': _STATS.average_time_ms,
        'effect_breakdown': {
            'buff_range': _STATS.buffs_applied,
            'debuff_range': _STATS.debuffs_applied,
            'diagonal_blocks': _STATS.directions_filtered,
            'geomancy_blocks': _STATS.geomancy_blocks,
            'wall_restrictions': _STATS.wall_captures_prevented,
        }
    }

def reset_movement_effects_stats() -> None:
    """Reset performance statistics."""
    global _STATS
    _STATS = MovementEffectStats()

# ==============================================================================
# BATCH OPERATIONS
# ==============================================================================

def apply_movement_effects_batch(
    state,
    positions: List[Tuple[int, int, int]],
    directions_list: List[List[Tuple[int, int, int]]],
    max_steps_list: List[int],
    cache_manager: Optional = None
) -> List[Tuple[List[Tuple[int, int, int]], int]]:
    """Apply movement effects to multiple positions in batch."""
    if cache_manager is None:
        cache_manager = state.cache

    results = []

    for start, directions, max_steps in zip(positions, directions_list, max_steps_list):
        result = apply_movement_effects(state, start, directions, max_steps, piece_type=None, cache_manager=cache_manager)
        results.append(result)

    return results

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

def apply_movement_effects_legacy(
    state,
    start: Tuple[int, int, int],
    raw_directions: List[Tuple[int, int, int]],
    max_steps: int,
) -> Tuple[List[Tuple[int, int, int]], int]:
    """Legacy interface for backward compatibility."""
    return apply_movement_effects(state, start, raw_directions, max_steps)
