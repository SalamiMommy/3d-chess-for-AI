# game3d/cache/managerperformance.py

"""Optimized performance monitoring with reduced overhead."""

import time
import gc
import os
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
from game3d.common.shared_types import FLOAT_DTYPE

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

    def __init__(self, enable_monitoring: bool = True, max_events: int = 1000, time_buffer_size: int = 1000):  # Increased buffer
        self.enable_monitoring = enable_monitoring
        self.events: List[CacheEvent] = []
        self.max_events = max_events
        self.start_time = time.perf_counter()

        # Performance counters
        self.tt_hits = 0
        self.tt_misses = 0
        self.tt_collisions = 0
        self.move_applications = 0
        self.move_undos = 0
        self.cache_clears = 0

        # Timing statistics - Using standardized float dtype
        self.move_apply_times = np.zeros(time_buffer_size, dtype=FLOAT_DTYPE)
        self.move_undo_times = np.zeros(time_buffer_size, dtype=FLOAT_DTYPE)
        self.legal_move_generation_times = np.zeros(time_buffer_size, dtype=FLOAT_DTYPE)
        self._time_index = 0
        self._buffer_size = time_buffer_size

        # Configurable thresholds for suggestions
        self.tt_hit_rate_threshold = 0.6
        self.legal_gen_time_threshold_ms = 3.0

    def record_event(self, event_type: CacheEventType, data: Dict[str, Any] = None):
        """Record a cache event with error handling."""
        if not self.enable_monitoring:
            return

        # Type checking
        if not isinstance(event_type, CacheEventType):
            return  # Silently skip invalid event types

        critical_events = {CacheEventType.TT_HIT, CacheEventType.TT_MISS, CacheEventType.CACHE_ERROR}  # Set for faster lookup
        if event_type in critical_events:
            event = CacheEvent(
                event_type=event_type,
                timestamp=time.perf_counter() - self.start_time,
                data=data or {}
            )
            if len(self.events) >= self.max_events:
                self.events.pop(0)
            self.events.append(event)

        # Update counters using dictionary lookup for better performance
        counter_map = {
            CacheEventType.TT_HIT: 'tt_hits',
            CacheEventType.TT_MISS: 'tt_misses',
            CacheEventType.TT_COLLISION: 'tt_collisions',
            CacheEventType.MOVE_APPLIED: 'move_applications',
            CacheEventType.MOVE_UNDONE: 'move_undos',
            CacheEventType.CACHE_CLEARED: 'cache_clears',
        }

        if event_type in counter_map:
            setattr(self, counter_map[event_type], getattr(self, counter_map[event_type]) + 1)

    def record_move_apply_time(self, duration: float):
        """Record move apply timing with error handling."""
        if duration < 0 or np.isnan(duration) or np.isinf(duration):
            return  # Skip invalid values
        self.move_apply_times[self._time_index] = duration
        self._time_index = (self._time_index + 1) % self._buffer_size

    def record_move_undo_time(self, duration: float):
        """Record move undo timing with error handling."""
        if duration < 0 or np.isnan(duration) or np.isinf(duration):
            return  # Skip invalid values
        self.move_undo_times[self._time_index] = duration
        self._time_index = (self._time_index + 1) % self._buffer_size

    def record_legal_move_generation_time(self, duration: float):
        """Record legal move generation timing with error handling."""
        if duration < 0 or np.isnan(duration) or np.isinf(duration):
            return  # Skip invalid values
        self.legal_move_generation_times[self._time_index] = duration
        self._time_index = (self._time_index + 1) % self._buffer_size

    def get_performance_stats(self) -> Dict[str, Any]:
        total_tt_accesses = self.tt_hits + self.tt_misses
        # Use np.where for clearer conditional logic
        tt_hit_rate = np.where(
            total_tt_accesses > 0,
            self.tt_hits / total_tt_accesses,
            0.0
        )

        # Use boolean masks directly to avoid creating copies
        apply_mask = self.move_apply_times != 0
        undo_mask = self.move_undo_times != 0
        legal_mask = self.legal_move_generation_times != 0

        # Count valid entries
        apply_count = np.count_nonzero(apply_mask)
        undo_count = np.count_nonzero(undo_mask)
        legal_count = np.count_nonzero(legal_mask)

        # Compute averages only if we have valid data
        avg_move_apply_time = np.mean(self.move_apply_times[apply_mask]) if apply_count > 0 else 0
        avg_move_undo_time = np.mean(self.move_undo_times[undo_mask]) if undo_count > 0 else 0
        avg_legal_gen_time = np.mean(self.legal_move_generation_times[legal_mask]) if legal_count > 0 else 0

        return {
            'tt_hits': self.tt_hits,
            'tt_misses': self.tt_misses,
            'tt_hit_rate': float(tt_hit_rate),  # Ensure it's a Python float for JSON serialization
            'tt_collisions': self.tt_collisions,
            'move_applications': self.move_applications,
            'move_undos': self.move_undos,
            'cache_clears': self.cache_clears,
            'avg_move_apply_time_ms': avg_move_apply_time * 1000,
            'avg_move_undo_time_ms': avg_move_undo_time * 1000,
            'avg_legal_gen_time_ms': avg_legal_gen_time * 1000,
            'total_events': len(self.events),
            'valid_timing_entries': {
                'apply_count': int(apply_count),
                'undo_count': int(undo_count),
                'legal_gen_count': int(legal_count),
            }
        }

    def get_optimization_suggestions(self) -> List[str]:
        suggestions = []
        stats = self.get_performance_stats()

        if stats['tt_hits'] + stats['tt_misses'] > 1000:
            if stats['tt_hit_rate'] < self.tt_hit_rate_threshold:
                suggestions.append("Transposition table hit rate is low. Consider increasing size beyond 6GB.")

        if stats['avg_legal_gen_time_ms'] > self.legal_gen_time_threshold_ms:
            suggestions.append("Legal move gen slow. Ensure parallelism and vectorization are active.")

        return suggestions

class MemoryManager:
    """Lightweight memory management without background monitoring."""

    def __init__(self, config, move_cache_ref):
        self.config = config
        self.move_cache_ref = move_cache_ref
        self.last_gc_time = 0
        self._gc_count = 0

    def check_and_gc_if_needed(self) -> None:
        """Lightweight memory check without psutil overhead."""
        current_time = time.time()

        self._gc_count += 1
        if self._gc_count < 100:
            return

        self._gc_count = 0

        if current_time - self.last_gc_time > self.config.gc_cooldown:
            gc.collect()
            self.last_gc_time = current_time
