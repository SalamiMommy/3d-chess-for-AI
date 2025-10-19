# performance.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from contextlib import contextmanager
import time

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

    def average_make_move_time(self) -> float:
        return self.total_make_move_time / max(1, self.make_move_calls)

    def average_undo_move_time(self) -> float:
        return self.total_undo_move_time / max(1, self.undo_move_calls)

    def average_legal_moves_time(self) -> float:
        return self.total_legal_moves_time / max(1, self.legal_moves_calls)

@contextmanager
def track_operation_time(metrics: PerformanceMetrics, metric_attr: str):
    """Context manager for tracking operation timing."""
    start_time = time.monotonic()  # More precise
    try:
        yield
    finally:
        duration = time.monotonic() - start_time
        setattr(metrics, metric_attr, getattr(metrics, metric_attr) + duration)

def track_performance(func):
    """Decorator for tracking function performance."""
    def wrapper(self, *args, **kwargs):
        start_time = time.monotonic()
        try:
            return func(self, *args, **kwargs)
        finally:
            duration = time.monotonic() - start_time
            # Add custom tracking (e.g., log or metric update)
            if hasattr(self, '_metrics'):
                self._metrics.custom_time += duration  # Assume extension in Metrics
    return wrapper
