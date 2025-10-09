"""Optimized central place to apply **all** movement buffs / debuffs to raw sliders."""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass
from enum import Enum
import time

from game3d.pieces.enums import Color, PieceType
from game3d.common.common import add_coords, euclidean_distance, manhattan, get_path_squares
from game3d.movement.movepiece import Move
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
    Optimized movement effects application.
    """
    start_time = time.perf_counter()
    _STATS.total_calls += 1

    if cache_manager is None:
        # Lazy import
        from game3d.cache.manager import get_cache_manager
        cache_manager = get_cache_manager(state.board, state.color)

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
    """Check if piece has slow effect using movement debuff cache."""
    return cache_manager.is_movement_debuffed(start, color)

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
    """Fallback to original implementation."""
    directions = raw_directions.copy()

    # 1. range buffs / debuffs
    if cache_manager.is_movement_buffed(start, state.color):
        max_steps += 1

    if cache_manager.is_movement_debuffed(start, state.color):
        max_steps = max(1, max_steps - 1)

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
            not cache_manager.can_capture_wall(start, start, state.color)):
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
    """Apply archery-specific movement effects."""
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
    """Apply hive-specific movement effects."""
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


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def modify_raw_moves(
    raw_moves: List["Move"],
    start: Tuple[int, int, int],
    state,
    cache_manager=None
) -> List["Move"]:
    """
    Central entry-point used by pseudo_legal.py.
    Applies *all* movement effects (buff, debuff, freeze, slow, geomancy, wall-capture, hive, archery, …) to the *raw* Move list returned
    by the piece dispatcher.
    Returns a new list of Moves that are safe to pass to the final
    bounds/friendly-colour filter.
    """
    # Lazy import get_cache_manager
    from game3d.cache.manager import get_cache_manager

    if cache_manager is None:
        cache_manager = state.cache

    # 1.  Early exit if the square is frozen
    if cache_manager.is_frozen(start, state.color):
        return []

    # 2.  Range modifiers (buff / debuff / slow)
    max_steps = _extract_max_steps(raw_moves)          # local helper below
    directions = _extract_directions(raw_moves, start)  # local helper below
    directions, max_steps = apply_movement_effects(
        state, start, directions, max_steps,
        piece_type=None, cache_manager=cache_manager
    )

    # 3.  Re-build the move list with the new directions / range
    moves = _rebuild_moves(raw_moves, start, directions, max_steps)

    # 4.  Piece-specific post-filters
    piece = cache_manager.occupancy.get(start)
    if piece is None:
        return []

    # 4a.  Hive
    if piece.ptype is PieceType.HIVE:
        moves = [_ for _ in moves if not cache_manager.is_movement_blocked_for_hive(
                 _.to_coord, state.color)]

    # 4b.  Archery (range-2 slide that needs LOS)
    if piece.ptype is PieceType.ARCHER:
        moves = [_ for _ in moves
                 if euclidean_distance(start, _.to_coord) == 2.0 and
                    _has_clear_line_of_sight(start, _.to_coord, state)]

    # 5.  Range-buff extension (if still needed)
    if cache_manager.is_movement_buffed(start, state.color):
        extended: List["Move"] = []
        for m in moves:
            extended.extend(_extend_move_range(m, start, state))
        moves = extended

    return moves


# ----------  tiny local helpers  ---------- #

def _extract_max_steps(moves: List["Move"]) -> int:
    """Largest coordinate delta found in the list."""
    if not moves:
        return 0
    return max(max(abs(a - b) for a, b in zip(m.from_coord, m.to_coord))
               for m in moves)

def _extract_directions(moves: List["Move"], start: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """Unique unit directions present in the raw move list."""
    dirs = set()
    for m in moves:
        dx, dy, dz = (b - a for a, b in zip(start, m.to_coord))
        norm = max(abs(dx), abs(dy), abs(dz))
        if norm:
            dirs.add((dx // norm, dy // norm, dz // norm))
    return list(dirs)

def _rebuild_moves(
    raw: List["Move"],
    start: Tuple[int, int, int],
    directions: List[Tuple[int, int, int]],
    max_steps: int
) -> List["Move"]:
    """Rebuild Move objects that respect the new directions / range."""
    rebuilt: List["Move"] = []
    for d in directions:
        for step in range(1, max_steps + 1):
            to_coord = tuple(a + step * b for a, b in zip(start, d))
            # preserve capture flag from any original raw move that ends here
            capture = any(m.to_coord == to_coord and m.is_capture for m in raw)
            rebuilt.append(Move.create_simple(start, to_coord, is_capture=capture))
    return rebuilt

def _extend_move_range(m: "Move", start: Tuple[int, int, int], state) -> List["Move"]:
    """Extend move range for buffed pieces."""
    extended = []
    direction = tuple((b - a) for a, b in zip(start, m.to_coord))
    norm = max(abs(d) for d in direction)
    if norm == 0:
        return [m]

    unit_dir = tuple(d // norm for d in direction)

    # Add one more step in the same direction
    next_step = tuple(a + b for a, b in zip(m.to_coord, unit_dir))

    # Check if next step is within board bounds
    if all(0 <= c < BOARD_SIZE for c in next_step):
        # Preserve capture flag from original move
        extended.append(Move.create_simple(start, next_step, is_capture=m.is_capture))

    return [m] + extended
