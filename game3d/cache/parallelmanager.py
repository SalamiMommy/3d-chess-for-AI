# game3d/cache/parallelmanager.py

"""Parallel processing management for the cache system."""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any
from contextlib import contextmanager

# Import centralized constants from shared_types
from ..common.shared_types import CPU_COUNT_FALLBACK, MIN_WORKERS


class ParallelManager:
    """Manages parallel execution with proper resource management and error handling."""
    
    _cached_cpu_count: Optional[int] = None
    _cached_calculation: Optional[int] = None

    def __init__(self, config: Any):
        """
        Initialize ParallelManager with validated configuration.
        
        Args:
            config: Configuration object with max_workers attribute
            
        Raises:
            TypeError: If config is None or lacks max_workers attribute
            ValueError: If max_workers is negative
        """
        self._validate_config(config)
        self.config = config
        self._executor: Optional[ThreadPoolExecutor] = None
        self._calculate_worker_count()

    def _validate_config(self, config: Any) -> None:
        """
        Validate configuration object and max_workers parameter.
        
        Args:
            config: Configuration object to validate
            
        Raises:
            TypeError: If config is None
            AttributeError: If config lacks max_workers attribute
            ValueError: If max_workers is negative
        """
        if config is None:
            raise TypeError("Configuration object cannot be None")
        
        if not hasattr(config, 'max_workers'):
            raise AttributeError("Configuration object must have 'max_workers' attribute")
        
        if config.max_workers is not None and config.max_workers < 0:
            raise ValueError(f"max_workers cannot be negative, got: {config.max_workers}")

    def _calculate_worker_count(self) -> int:
        """
        Calculate optimal number of worker threads with caching and fallback.

        Returns:
            int: Number of worker threads to use

        Note:
            This method caches the result to avoid redundant system calls
        """
        # Return cached value if available
        if self._cached_calculation is not None:
            return self._cached_calculation

        config_workers = self.config.max_workers

        # Handle explicit configuration
        if config_workers is not None:
            if config_workers == 0:
                # Use minimum workers for zero (treat as None)
                worker_count = MIN_WORKERS
            else:
                # Use explicitly configured value
                worker_count = config_workers
        else:
            # Auto-calculate optimal worker count
            worker_count = self._get_optimal_worker_count()

        # Cache the result
        self._cached_calculation = worker_count
        return worker_count

    def _get_optimal_worker_count(self) -> int:
        """
        Get optimal worker count based on system resources.
        
        Returns:
            int: Optimal number of workers for the current system
            
        Note:
            Uses cached CPU count to avoid repeated system calls
        """
        cpu_count = self._get_cached_cpu_count()
        
        # Default strategy: use CPU count - 1, with minimum of MIN_WORKERS
        # This leaves one CPU available for system operations
        optimal_count = max(MIN_WORKERS, cpu_count - 1) if cpu_count > 1 else MIN_WORKERS
        
        return optimal_count

    def _get_cached_cpu_count(self) -> int:
        """
        Get CPU count with fallback and caching.
        
        Returns:
            int: CPU count with fallback to CPU_COUNT_FALLBACK
            
        Note:
            Caches the result to avoid redundant os.cpu_count() calls
        """
        if self._cached_cpu_count is not None:
            return self._cached_cpu_count
        
        cpu_count = os.cpu_count()
        if cpu_count is None or cpu_count <= 0:
            cpu_count = CPU_COUNT_FALLBACK
        
        # Cache the result
        self._cached_cpu_count = cpu_count
        return cpu_count

    def _initialize_executor(self) -> None:
        """
        Initialize the ThreadPoolExecutor with calculated worker count.
        
        Raises:
            RuntimeError: If executor is already initialized
        """
        if self._executor is not None:
            raise RuntimeError("ThreadPoolExecutor is already initialized")
        
        worker_count = self._calculate_worker_count()
        
        self._executor = ThreadPoolExecutor(max_workers=worker_count)

    def get_executor(self) -> ThreadPoolExecutor:
        """
        Get or create the ThreadPoolExecutor instance.
        
        Returns:
            ThreadPoolExecutor: The initialized executor instance
            
        Note:
            Creates executor on first access if not already initialized
        """
        if self._executor is None:
            self._initialize_executor()
        
        return self._executor

    def submit(self, fn, *args, **kwargs):
        """
        Submit a function to the thread pool for execution.
        
        Args:
            fn: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Future: The submitted task's future object
        """
        executor = self.get_executor()
        return executor.submit(fn, *args, **kwargs)

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """
        Submit multiple function calls to the thread pool.
        
        Args:
            fn: Function to execute
            *iterables: Iterables of arguments
            timeout: Optional timeout for the operation
            chunksize: Size of chunks for batch processing
            
        Returns:
            Generator: Results of the function calls
        """
        executor = self.get_executor()
        return executor.map(fn, *iterables, timeout=timeout, chunksize=chunksize)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """
        Shutdown the ThreadPoolExecutor with proper cleanup.
        
        Args:
            wait: If True, wait for all pending tasks to complete
            cancel_futures: If True, cancel all pending futures
        """
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            self._executor = None

    def force_shutdown(self) -> None:
        """
        Force immediate shutdown of all pending tasks.
        
        Note:
            This method cancels all pending futures without waiting.
            Use with caution as it may leave work incomplete.
        """
        self.shutdown(wait=False, cancel_futures=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.shutdown(wait=True)
        return False

    def __del__(self):
        """Destructor with graceful cleanup."""
        if self._executor is not None:
            # Use gentle shutdown in destructor
            self.shutdown(wait=False)
            # Suppress exceptions during garbage collection
            pass

    @property
    def worker_count(self) -> int:
        """Get the current worker count."""
        return self._calculate_worker_count()

    @property
    def is_running(self) -> bool:
        """Check if the executor is currently running."""
        return self._executor is not None

    def reset_cache(self) -> None:
        """
        Reset cached calculations to allow recalculation.
        
        Note:
            This is mainly useful for testing or when system resources change
        """
        self._cached_calculation = None
        self._cached_cpu_count = None
