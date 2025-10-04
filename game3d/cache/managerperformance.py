# game3d/cache/managerperformance.py

"""Performance monitoring and memory management for the cache manager."""

import time
import gc
import psutil
import threading
import os
import numpy as np
import pickle
import zlib
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
    """Advanced performance monitoring for the cache system."""

    def __init__(self, enable_monitoring: bool = True, max_events: int = 10000):
        self.enable_monitoring = enable_monitoring
        self.events: List[CacheEvent] = []
        self.max_events = max_events
        self.start_time = time.time()

        # Performance counters
        self.tt_hits = 0
        self.tt_misses = 0
        self.tt_collisions = 0
        self.move_applications = 0
        self.move_undos = 0
        self.cache_clears = 0

        # Timing statistics (keep last 1000)
        self.move_apply_times = []
        self.move_undo_times = []
        self.legal_move_generation_times = []

    def record_event(self, event_type: CacheEventType, data: Dict[str, Any] = None):
        if not self.enable_monitoring:
            return
        event = CacheEvent(
            event_type=event_type,
            timestamp=time.time() - self.start_time,
            data=data or {}
        )
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)

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
        self.move_apply_times.append(duration)
        if len(self.move_apply_times) > 1000:
            self.move_apply_times.pop(0)

    def record_move_undo_time(self, duration: float):
        self.move_undo_times.append(duration)
        if len(self.move_undo_times) > 1000:
            self.move_undo_times.pop(0)

    def record_legal_move_generation_time(self, duration: float):
        self.legal_move_generation_times.append(duration)
        if len(self.legal_move_generation_times) > 1000:
            self.legal_move_generation_times.pop(0)

    def get_performance_stats(self) -> Dict[str, Any]:
        total_tt_accesses = self.tt_hits + self.tt_misses
        tt_hit_rate = self.tt_hits / max(1, total_tt_accesses)

        avg_move_apply_time = np.mean(self.move_apply_times) if self.move_apply_times else 0
        avg_move_undo_time = np.mean(self.move_undo_times) if self.move_undo_times else 0
        avg_legal_gen_time = np.mean(self.legal_move_generation_times) if self.legal_move_generation_times else 0

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

        # Only warn about TT hit rate after meaningful usage
        if stats['tt_hits'] + stats['tt_misses'] > 1000:  # ← ADD THIS CHECK
            if stats['tt_hit_rate'] < 0.6:
                suggestions.append("Transposition table hit rate is low. Consider increasing size beyond 6GB.")

        if stats['tt_collisions'] > stats['tt_hits'] * 0.01:
            suggestions.append("High TT collision rate. Verify Zobrist hash quality.")

        if stats['avg_move_apply_time_ms'] > 0.5:
            suggestions.append("Move apply time high. Profile occupancy/piece cache rebuilds.")

        if stats['avg_legal_gen_time_ms'] > 3.0:
            suggestions.append("Legal move gen slow. Ensure parallelism and vectorization are active.")

        return suggestions

class MemoryManager:
    """Handles memory monitoring and compression."""

    def __init__(self, config, move_cache_ref):
        self.config = config
        self.move_cache_ref = move_cache_ref  # Weak ref or direct ref to move_cache
        self.last_gc_time = 0
        self.mem_monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.mem_monitor_thread.start()

    def _monitor_memory(self) -> None:
        """Background thread: GC as soon as RAM ≥ 90 % (increased from 85%)."""
        process = psutil.Process(os.getpid())
        last_check = 0
        cached_mem = None
        while True:
            try:
                current_time = time.time()
                # Cache memory checks to reduce overhead - check every 30 seconds instead of 1 second
                if cached_mem is None or current_time - last_check > 30.0:
                    cached_mem = psutil.virtual_memory()
                    last_check = current_time

                # System-wide physical memory check - increased threshold from 85% to 90%
                used_percent = cached_mem.percent
                if used_percent >= 90.0:  # Increased from 85%
                    print(f"[CacheManager] RAM {used_percent:.1f}% ≥ 90 % → forcing GC")
                    gc.collect()
                    self.compress_caches()
                    self.last_gc_time = current_time

                # Private bytes check (secondary) - increased threshold from 40GB to 50GB
                priv_gb = process.memory_info().rss / (1024 ** 3)
                if (priv_gb > 50 and  # Increased from 40GB
                    current_time - self.last_gc_time > self.config.gc_cooldown):
                    print(f"[CacheManager] Private bytes {priv_gb:.1f} GB → GC")
                    gc.collect()
                    self.compress_caches()
                    self.last_gc_time = current_time

            except Exception as e:
                print(f"[CacheManager] monitor error: {e}")

            time.sleep(30.0)  # Increased from 5s to 30s

    def check_and_gc_if_needed(self) -> None:
        process = psutil.Process(os.getpid())
        mem_usage_gb = process.memory_info().rss / (1024 ** 3)
        current_time = time.time()
        if mem_usage_gb > self.config.mem_threshold_gb and (current_time - self.last_gc_time > self.config.gc_cooldown):
            print(f"[CacheManager] Inline GC trigger: {mem_usage_gb:.2f}GB > {self.config.mem_threshold_gb}GB")
            gc.collect()
            self.compress_caches()
            self.last_gc_time = current_time

    def compress_caches(self) -> None:
        """Compress big caches like simple move cache using zlib."""
        if not self.move_cache_ref:
            return
        simple_cache = self.move_cache_ref._simple_move_cache
        if simple_cache and len(simple_cache) > 10000:  # Threshold
            compressed = {}
            for key, moves in simple_cache.items():
                pickled = pickle.dumps(moves, protocol=5)
                compressed[key] = zlib.compress(pickled, level=1)
            self.move_cache_ref._simple_move_cache = compressed
            print(f"[CacheManager] Compressed {len(compressed)} simple move cache entries.")
            # Note: movecache.py needs update to decompress on access (check if bytes, then pickle.loads(zlib.decompress()))
