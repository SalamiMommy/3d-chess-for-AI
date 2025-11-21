# performance.py - FIXED VERSION
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from contextlib import contextmanager
from functools import wraps
import time

# Import shared type definitions from shared_types
from game3d.common.shared_types import (
    COORD_DTYPE, INDEX_DTYPE, BOOL_DTYPE, COLOR_DTYPE, PIECE_TYPE_DTYPE, FLOAT_DTYPE, MIN_CALLS
)

@dataclass
class PerformanceMetrics:
    """Performance tracking for GameState operations."""
    make_move_calls: int = 0
    undo_move_calls: int = 0
    legal_moves_calls: int = 0
    zobrist_computations: int = 0
    total_make_move_time: float = 0.0
    total_undo_move_time: float = 0.0
    total_legal_moves_time: float = 0.0
    custom_time: float = 0.0

    def average_make_move_time(self) -> float:
        """Calculate average make_move execution time."""
        return self.total_make_move_time / max(MIN_CALLS, self.make_move_calls)

    def average_undo_move_time(self) -> float:
        """Calculate average undo_move execution time."""
        return self.total_undo_move_time / max(MIN_CALLS, self.undo_move_calls)

    def average_legal_moves_time(self) -> float:
        """Calculate average legal_moves execution time."""
        return self.total_legal_moves_time / max(MIN_CALLS, self.legal_moves_calls)

    def average_custom_time(self) -> float:
        """Calculate average custom execution time across all operations."""
        total_calls = self.make_move_calls + self.undo_move_calls + self.legal_moves_calls
        return self.custom_time / max(MIN_CALLS, total_calls)



@contextmanager
def track_operation_time(metrics: PerformanceMetrics, metric_attr: str):
    """Context manager for tracking operation timing.
    
    Args:
        metrics: PerformanceMetrics instance to update
        metric_attr: String name of the attribute to update with timing
    """
    start_time = time.monotonic()
    try:
        yield
    finally:
        end_time = time.monotonic()
        duration = end_time - start_time
        setattr(metrics, metric_attr, getattr(metrics, metric_attr, 0.0) + duration)
def track_performance(func):
    """Decorator for tracking function performance execution time.
    
    This decorator measures the execution time of methods and updates
    the PerformanceMetrics custom_time attribute. It's designed to be
    used on GameState methods that have a '_metrics' attribute.
    
    Args:
        func: The function to wrap with performance tracking
        
    Returns:
        Wrapper function that tracks execution time
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.monotonic()
        result = func(self, *args, **kwargs)
        return result
    return wrapper
