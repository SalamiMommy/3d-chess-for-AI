# debug_utils.py - CONSOLIDATED UTILITIES VERSION (FIXED)
# ------------------------------------------------------------------
# Debugging and validation utilities - CONSOLIDATED UTILITIES VERSION
# Uses shared_types and coord_utils_consolidated for consistency
# ------------------------------------------------------------------
from __future__ import annotations
import numpy as np
import time
from typing import List, Dict, Any, TYPE_CHECKING, Optional
from dataclasses import dataclass, field
from functools import wraps
from contextlib import contextmanager

# FIXED: Import from consolidated validation modules
from game3d.common.shared_types import (
    Coord, CoordBatch, BoolArray,
    COORD_DTYPE, BATCH_COORD_DTYPE, BOOL_DTYPE,
    SIZE, MAX_COORD_VALUE, MIN_COORD_VALUE, N_DIMENSIONS, MS_TO_S, MIN_CALLS, ZERO_VALUE, get_empty_coord_batch,
    # From enums.py
    Piece, PieceType
)
from game3d.common.validation import (
    validate_coordinate_array,
    validate_coords_batch, 
    validate_coords_bounds,
    validate_coord
)
from game3d.common.coord_utils import ensure_coords
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.attacks.movepiece import Move

# Coord is already imported from constants





def validate_coord(c: Coord) -> bool:
    """Validate a coordinate array (shape (3,), dtype=int16) - USES CONSOLIDATED VALIDATION."""
    # Use consolidated validation function from validation module
    validated_coord = validate_coords_batch(c.reshape(1, N_DIMENSIONS) if c.ndim == 1 else c)
    return True


def coord_to_string(c: Coord) -> str:
    """Convert coordinate array to string."""
    return f"({c[0]}, {c[1]}, {c[2]})"


def batch_coords_to_array(coords: List[Coord]) -> np.ndarray:
    """Convert list of coordinate arrays to (N, 3) array - VECTORIZED for performance."""
    if not coords:
        return get_empty_coord_batch(0)

    # VECTORIZED: Process all coordinates at once instead of looping
    # Use ensure_coords for validation and normalization
    validated_coords = [ensure_coords(coord) for coord in coords]
    
    # Stack all validated coordinates
    coord_arrays = []
    for validated_coord in validated_coords:
        if validated_coord.ndim == 1:
            coord_arrays.append(validated_coord)
        else:
            coord_arrays.append(validated_coord[0])
    
    return np.stack(coord_arrays, axis=0)


def array_to_batch_coords(arr: np.ndarray) -> List[Coord]:
    """Convert (N, 3) array to list of coordinate arrays - Returns numpy arrays, not lists."""
    if arr.size == ZERO_VALUE:
        return []
    # Return list of array views for better performance
    return list(arr)


@contextmanager
def measure_time_ms():
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * MS_TO_S


@dataclass(slots=True)
class StatsTracker:
    total_calls: int = ZERO_VALUE
    average_time_ms: float = 0.0

    def update_average(self, elapsed_ms: float) -> None:
        self.total_calls += 1
        if self.total_calls == MIN_CALLS:
            self.average_time_ms = elapsed_ms
        else:
            self.average_time_ms = (
                (self.average_time_ms * (self.total_calls - 1) + elapsed_ms) / self.total_calls
            )

    def get_stats(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def reset(self) -> None:
        self.total_calls = ZERO_VALUE
        self.average_time_ms = 0.0


@dataclass(slots=True)
class MoveStatsTracker:
    total_calls: int = ZERO_VALUE
    average_time_ms: float = 0.0
    total_moves_generated: int = ZERO_VALUE
    total_moves_filtered: int = ZERO_VALUE
    freeze_filtered: int = ZERO_VALUE
    check_filtered: int = ZERO_VALUE
    piece_breakdown: Dict[PieceType, int] = field(default_factory=lambda: {pt: 0 for pt in PieceType})

    def update_average(self, elapsed_ms: float) -> None:
        self.total_calls += 1
        if self.total_calls == MIN_CALLS:
            self.average_time_ms = elapsed_ms
        else:
            self.average_time_ms = (
                (self.average_time_ms * (self.total_calls - 1) + elapsed_ms) / self.total_calls
            )

    def get_stats(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def reset(self) -> None:
        self.total_calls = ZERO_VALUE
        self.average_time_ms = 0.0
        self.total_moves_generated = ZERO_VALUE
        self.total_moves_filtered = ZERO_VALUE
        self.freeze_filtered = ZERO_VALUE
        self.check_filtered = ZERO_VALUE
        self.piece_breakdown = {pt: ZERO_VALUE for pt in PieceType}


def track_time(tracker: MoveStatsTracker):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with measure_time_ms() as elapsed:
                result = func(*args, **kwargs)
            tracker.update_average(elapsed())
            return result
        return wrapper
    return decorator


class GeneratorBase:
    def __init__(self, mode_enum, default_mode, stats_tracker: MoveStatsTracker):
        self.mode_enum = mode_enum
        self.default_mode = default_mode
        self.stats = stats_tracker

    def generate(self, state: 'GameState', *, mode: str | None = None) -> List['Move']:
        effective_mode = mode if mode is not None else self.default_mode.value
        return self._impl(state, mode=effective_mode)

    def _impl(self, state: 'GameState', mode: str) -> List['Move']:
        raise NotImplementedError


def log_oob(coords: np.ndarray):
    """Log out-of-bound coordinates - FIXED: Use consolidated bounds checking."""
    if coords is None:
        return
    if isinstance(coords, np.ndarray) and coords.size == 0:
        return

    # FIXED: Use consolidated validation and bounds checking
    if coords.ndim == 1:
        coords = coords[np.newaxis, :]

    # Use consolidated in_bounds_vectorized from shared_types
    valid_mask = in_bounds_vectorized(coords)
    bad = coords[~valid_mask]

    if len(bad) > 0:
        print(f"[OOB] {len(bad)} invalid coords: {bad[:3]}")


@dataclass(slots=True)
class UndoSnapshot:
    original_board_array: np.ndarray
    original_halfmove_clock: int
    original_turn_number: int
    original_zkey: int
    moving_piece: Piece
    captured_piece: Optional[Piece] = None
    original_aura_state: Any = None
    original_trailblaze_state: Any = None
    original_geomancy_state: Any = None

    def __post_init__(self):
        # Ensure board array is truly a copy
        if not self.original_board_array.flags['OWNDATA']:
            self.original_board_array = self.original_board_array.copy()


@dataclass(slots=True)
class CacheStats:
    hits: int = ZERO_VALUE
    misses: int = ZERO_VALUE
    collisions: int = ZERO_VALUE
    size: int = ZERO_VALUE
    memory_usage_mb: float = 0.0
    last_rebuild: float = 0.0

    def get_stats_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class CacheStatsMixin:
    def __init__(self):
        self._stats = CacheStats()
        self._performance_tracker = StatsTracker()

    def get_cache_stats(self) -> Dict[str, Any]:
        base_stats = self._stats.get_stats_dict()
        base_stats.update(self._performance_tracker.get_stats())
        return base_stats


# Module exports
__all__ = [
    # Validation functions
    'validate_coord', 'coord_to_string',

    # Coordinate conversion utilities
    'batch_coords_to_array', 'array_to_batch_coords',

    # Time measurement
    'measure_time_ms',

    # Stats tracking
    'StatsTracker', 'MoveStatsTracker', 'track_time',

    # Generator base class
    'GeneratorBase',

    # OOB logging
    'log_oob',

    # Snapshot and stats classes
    'UndoSnapshot', 'CacheStats', 'CacheStatsMixin',
]
