# game3d/cache/managerperformance.py

"""Optimized performance monitoring with reduced overhead."""

import time
import gc
import threading
import os
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class CacheEventType(Enum):
    MOVE_APPLIED = "move_applied"
    MOVE_UNDONE = "move_undone"
    TT_HIT = "tt_hit"
    TT_MISS = "tt_miss"
    TT_COLLISION = "tt_collision"
    CACHE_CLEARED = "cache_cleared"
    CACHE_ERROR = "cache_error"

@dataclass
class CacheEvent:
    """Event for cache performance monitoring."""
    __slots__ = ("event_type", "timestamp", "data")
    event_type: CacheEventType
    timestamp: float
    data: Dict[str, Any]

class CachePerformanceMonitor:
    """Lightweight performance monitoring for the cache system."""

    def __init__(self, enable_monitoring: bool = True, max_events: int = 1000):  # Reduced from 10000
        self.enable_monitoring = enable_monitoring
        self.events: List[CacheEvent] = []
        self.max_events = max_events
        self.start_time = time.perf_counter()

        # Performance counters - use simple counters instead of full event tracking
        self.tt_hits = 0
        self.tt_misses = 0
        self.tt_collisions = 0
        self.move_applications = 0
        self.move_undos = 0
        self.cache_clears = 0

        # Timing statistics - keep smaller samples
        self.move_apply_times = np.zeros(100, dtype=np.float32)  # Fixed size array
        self.move_undo_times = np.zeros(100, dtype=np.float32)
        self.legal_move_generation_times = np.zeros(100, dtype=np.float32)
        self._time_index = 0

    def record_event(self, event_type: CacheEventType, data: Dict[str, Any] = None):
        if not self.enable_monitoring:
            return

        # Only record critical events to reduce overhead
        if event_type in [CacheEventType.TT_HIT, CacheEventType.TT_MISS,
                         CacheEventType.CACHE_ERROR]:
            event = CacheEvent(
                event_type=event_type,
                timestamp=time.perf_counter() - self.start_time,
                data=data or {}
            )
            # Use circular buffer for events
            if len(self.events) >= self.max_events:
                self.events.pop(0)
            self.events.append(event)

        # Update counters
        if event_type == CacheEventType.TT_HIT:
            self.tt_hits += 1
        elif event_type == CacheEventType.TT_MISS:
            self.tt_misses += 1
        elif event_type == CacheEventType.TT_COLLISION:
            self.tt_collisions += 1
        elif event_type == CacheEventType.MOVE_APPLIED:
            self.move_applications += 1
        elif event_type == CacheEventType.MOVE_UNDONE:
            self.move_undos += 1
        elif event_type == CacheEventType.CACHE_CLEARED:
            self.cache_clears += 1

    def record_move_apply_time(self, duration: float):
        self.move_apply_times[self._time_index % 100] = duration
        self._time_index += 1

    def record_move_undo_time(self, duration: float):
        self.move_undo_times[self._time_index % 100] = duration

    def record_legal_move_generation_time(self, duration: float):
        self.legal_move_generation_times[self._time_index % 100] = duration

    def get_performance_stats(self) -> Dict[str, Any]:
        total_tt_accesses = self.tt_hits + self.tt_misses
        tt_hit_rate = self.tt_hits / max(1, total_tt_accesses)

        # Use numpy for faster mean calculation
        valid_apply_times = self.move_apply_times[self.move_apply_times != 0]
        valid_undo_times = self.move_undo_times[self.move_undo_times != 0]
        valid_legal_times = self.legal_move_generation_times[self.legal_move_generation_times != 0]

        avg_move_apply_time = np.mean(valid_apply_times) if len(valid_apply_times) > 0 else 0
        avg_move_undo_time = np.mean(valid_undo_times) if len(valid_undo_times) > 0 else 0
        avg_legal_gen_time = np.mean(valid_legal_times) if len(valid_legal_times) > 0 else 0

        return {
            'tt_hits': self.tt_hits,
            'tt_misses': self.tt_misses,
            'tt_hit_rate': tt_hit_rate,
            'tt_collisions': self.tt_collisions,
            'move_applications': self.move_applications,
            'move_undos': self.move_undos,
            'cache_clears': self.cache_clears,
            'avg_move_apply_time_ms': avg_move_apply_time * 1000,
            'avg_move_undo_time_ms': avg_move_undo_time * 1000,
            'avg_legal_gen_time_ms': avg_legal_gen_time * 1000,
            'total_events': len(self.events),
        }

    def get_optimization_suggestions(self) -> List[str]:
        suggestions = []
        stats = self.get_performance_stats()

        if stats['tt_hits'] + stats['tt_misses'] > 1000:
            if stats['tt_hit_rate'] < 0.6:
                suggestions.append("Transposition table hit rate is low. Consider increasing size beyond 6GB.")

        if stats['avg_legal_gen_time_ms'] > 3.0:
            suggestions.append("Legal move gen slow. Ensure parallelism and vectorization are active.")

        return suggestions

class MemoryManager:
    """Lightweight memory management without background monitoring."""

    def __init__(self, config, move_cache_ref):
        self.config = config
        self.move_cache_ref = move_cache_ref
        self.last_gc_time = 0
        self._gc_count = 0

        # Disable background monitoring thread to reduce overhead
        # Memory checks will be done on-demand

    def check_and_gc_if_needed(self) -> None:
        """Lightweight memory check without psutil overhead."""
        current_time = time.time()

        # Only check every N operations to reduce overhead
        self._gc_count += 1
        if self._gc_count < 100:  # Check every 100 operations
            return

        self._gc_count = 0

        # Simple GC based on time since last GC
        if current_time - self.last_gc_time > self.config.gc_cooldown:
            gc.collect()
            self.last_gc_time = current_time

    def compress_caches(self) -> None:
        """Optional compression - disabled by default to reduce overhead."""
        pass  # Disable compression for performance
