# game3d/cache/parallelmanager.py

"""Parallel processing management for the cache system."""

import os
from concurrent.futures import ThreadPoolExecutor

class ParallelManager:
    """Manages parallel execution and vectorization flags."""

    def __init__(self, config):
        self.config = config
        max_workers = config.max_workers or max(1, (os.cpu_count() or 6) - 1)  # Optimized fallback
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def shutdown(self, wait: bool = False):
        self.executor.shutdown(wait=wait)
