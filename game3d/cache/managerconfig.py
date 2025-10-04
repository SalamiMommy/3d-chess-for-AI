# game3d/cache/managerconfig.py

"""Configuration settings for the cache manager."""

from dataclasses import dataclass

@dataclass
class ManagerConfig:
    """Configuration for cache manager resources and thresholds."""
    total_ram_budget_gb: int = 45
    main_tt_size_mb_fraction: float = 0.73
    sym_tt_size_mb_fraction: float = 0.27
    mem_threshold_gb: int = 50
    mem_check_interval: int = 1800  # Seconds
    gc_cooldown: int = 300  # Seconds
    max_workers: int = 8  # Adjusted based on CPU
    cache_stats_interval: int = 1000
    enable_parallel: bool = True
    enable_vectorization: bool = True

    def __post_init__(self):
        import os
        self.max_workers = min(self.max_workers, os.cpu_count() or 6)
        total_mb = self.total_ram_budget_gb * 1024
        self.main_tt_size_mb = int(total_mb * self.main_tt_size_mb_fraction)
        self.sym_tt_size_mb = int(total_mb * self.sym_tt_size_mb_fraction)
