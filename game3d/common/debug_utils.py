# game3d/common/debug_utils.py
# ------------------------------------------------------------------
# Debugging and validation utilities
# ------------------------------------------------------------------
from __future__ import annotations
import torch
import time
import numpy as np
from typing import List, Tuple, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from functools import wraps
from contextlib import contextmanager

from game3d.common.coord_utils import Coord, in_bounds_vectorised
from game3d.common.enums import PieceType
from game3d.pieces.piece import Piece

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.movement.movepiece import Move

def fallback_mode(default_mode):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log error (replace print with logging)
                print(f"Error in mode: {e}, falling back to {default_mode}")
                import traceback
                traceback.print_exc()
                # Assume args[0] is state, kwargs.get('mode') or args[1] is mode
                mode_arg_idx = 1 if len(args) > 1 else None
                if mode_arg_idx:
                    args = list(args)
                    args[mode_arg_idx] = default_mode
                else:
                    kwargs['mode'] = default_mode
                return func(*args, **kwargs)
            return wrapper
    return decorator

def validate_coord(c: Coord) -> bool:
    return isinstance(c, tuple) and len(c) == 3 and all(isinstance(i, int) for i in c) and in_bounds_vectorised(np.array([c]))[0]

def coord_to_string(c: Coord) -> str:
    return f"({c[0]}, {c[1]}, {c[2]})"

def batch_coords_to_tensor(coords: List[Coord]) -> torch.Tensor:
    return torch.tensor(coords, dtype=torch.long) if coords else torch.empty((0, 3), dtype=torch.long)

def tensor_to_batch_coords(tensor: torch.Tensor) -> List[Coord]:
    return [tuple(row.tolist()) for row in tensor]

@contextmanager
def measure_time_ms():
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000

@dataclass(slots=True)
class StatsTracker:
    total_calls: int = 0
    average_time_ms: float = 0.0
    # Add common fields like total_moves_generated, etc., as needed

    def update_average(self, elapsed_ms: float) -> None:
        self.average_time_ms = (
            (self.average_time_ms * (self.total_calls - 1) + elapsed_ms) / self.total_calls
            if self.total_calls > 0 else elapsed_ms
        )

    def get_stats(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def reset(self) -> None:
        self.total_calls = 0
        self.average_time_ms = 0.0

@dataclass(slots=True)
class MoveStatsTracker:
    total_calls: int = 0
    average_time_ms: float = 0.0
    total_moves_generated: int = 0
    total_moves_filtered: int = 0
    freeze_filtered: int = 0
    check_filtered: int = 0
    piece_breakdown: Dict[PieceType, int] = field(default_factory=lambda: {pt: 0 for pt in PieceType})

    def update_average(self, elapsed_ms: float) -> None:
        self.average_time_ms = (
            (self.average_time_ms * (self.total_calls - 1) + elapsed_ms) / self.total_calls
            if self.total_calls > 0 else elapsed_ms
        )

    def get_stats(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def reset(self) -> None:
        self.total_calls = 0
        self.average_time_ms = 0.0
        self.total_moves_generated = 0
        self.total_moves_filtered = 0
        self.freeze_filtered = 0
        self.check_filtered = 0
        self.piece_breakdown = {pt: 0 for pt in PieceType}

# NEW: Track time decorator/context
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

# NEW: Abstract GeneratorBase
class GeneratorBase:
    def __init__(self, mode_enum, default_mode, stats_tracker: MoveStatsTracker):
        self.mode_enum = mode_enum
        self.default_mode = default_mode
        self.stats = stats_tracker

    def generate(self, state: GameState, *, mode: str | None = None) -> List[Move]:
        # if caller supplied mode use it, otherwise fall back to default
        effective_mode = mode if mode is not None else self.default_mode.value
        return self._impl(state, mode=effective_mode)

    def _impl(self, state: GameState, mode: str) -> List[Move]:
        raise NotImplementedError

# NEW: For logging OOB
def log_oob(coords: np.ndarray):
    bad = coords[~in_bounds_vectorised(coords)]
    if len(bad) > 0:
        print(f"[OOB] {len(bad)} invalid coords: {bad[:3]}")

# NEW: UndoSnapshot dataclass
@dataclass(slots=True)
class UndoSnapshot:
    original_board_tensor: torch.Tensor
    original_halfmove_clock: int
    original_turn_number: int
    original_zkey: int
    moving_piece: Piece
    captured_piece: Optional[Piece] = None
    # ADD the missing fields that are being used in turnmove.py
    original_aura_state: Any = None
    original_trailblaze_state: Any = None
    original_geomancy_state: Any = None


@dataclass(slots=True)
class CacheStats:
    """Standardized cache statistics."""
    hits: int = 0
    misses: int = 0
    collisions: int = 0
    size: int = 0
    memory_usage_mb: float = 0.0
    last_rebuild: float = 0.0

    def get_stats_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class CacheStatsMixin:
    """Mixin for standardized cache statistics."""
    def __init__(self):
        self._stats = CacheStats()
        self._performance_tracker = StatsTracker()

    def get_cache_stats(self) -> Dict[str, Any]:
        base_stats = self._stats.get_stats_dict()
        base_stats.update(self._performance_tracker.get_stats())
        return base_stats
