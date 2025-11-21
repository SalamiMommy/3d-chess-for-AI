# game3d/cache/managerconfig.py

"""Configuration settings for the cache manager."""

import os
from dataclasses import dataclass, field
from game3d.common.shared_types import SIZE, VOLUME

@dataclass
class ManagerConfig:
    """Configuration for cache manager resources and thresholds."""
    total_ram_budget_gb: int = 2
    main_tt_size_mb_fraction: float = 0.5  # Main TT fraction (sums to 1.0 with sym)
    sym_tt_size_mb_fraction: float = 0.5   # Symmetry TT fraction
    mem_threshold_gb: int = 2
    mem_check_interval: int = 1800  # Seconds
    gc_cooldown: int = 300  # Seconds
    max_workers: int | None = field(default=None)  # None to use auto-fallback
    cache_stats_interval: int = 1000
    enable_parallel: bool = False
    enable_vectorization: bool = True
    enable_disk_cache: bool = False
    
    # Board-specific configuration using centralized constants
    SIZE: int = SIZE
    board_volume: int = VOLUME
    max_cache_entries_per_volume: int = max(1000, VOLUME // 2)

    def __post_init__(self):
        if self.total_ram_budget_gb < 1:
            raise ValueError("total_ram_budget_gb must be at least 1")
        total_mb = min(self.total_ram_budget_gb * 1024, 1024)
        self.main_tt_size_mb = int(total_mb * self.main_tt_size_mb_fraction)
        self.sym_tt_size_mb = int(total_mb * self.sym_tt_size_mb_fraction)
        if self.max_workers is None:
            self.max_workers = max(1, (os.cpu_count() or 6) - 1)
