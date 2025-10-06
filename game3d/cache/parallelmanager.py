# game3d/cache/parallelmanager.py

"""Parallel processing management for the cache system."""

from concurrent.futures import ThreadPoolExecutor

class ParallelManager:
    """Manages parallel execution and vectorization flags."""

    def __init__(self, config):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)

    def shutdown(self):
        self.executor.shutdown(wait=False)
