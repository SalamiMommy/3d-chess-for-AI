# Optimized performance utilities for 3D chess engine
import time
import numpy as np
from contextlib import contextmanager
from typing import Any, Optional, Union
from numba import njit, prange
from ..common.shared_types import MS_TO_S

def _safe_attribute_update(obj: Any, attr_name: str, value: Union[int, float]) -> bool:
    """
    Safely update an attribute on an object with consolidated error handling.
    
    This consolidates the redundant hasattr → getattr → setattr pattern
    to reduce overhead in high-frequency operations.
    
    Args:
        obj: Object to update
        attr_name: Name of the attribute
        value: Value to add to the attribute
        
    Returns:
        bool: True if successful, False otherwise
    """
    if hasattr(obj, attr_name):
        current_value = getattr(obj, attr_name)
        setattr(obj, attr_name, current_value + value)
        return True
    return False

def _safe_increment_counter(obj: Any, attr_name: str, increment: int = 1) -> bool:
    """
    Safely increment a counter attribute on an object.
    
    Args:
        obj: Object to update
        attr_name: Name of the counter attribute
        increment: Value to add to the counter (default: 1)
        
    Returns:
        bool: True if successful, False otherwise
    """
    return _safe_attribute_update(obj, attr_name, increment)

def create_timing_context(start_time: Optional[float] = None) -> float:
    """
    Create a standardized timing context.
    
    Args:
        start_time: Optional custom start time
        
    Returns:
        float: The start time to use for timing
    """
    return start_time if start_time is not None else time.perf_counter()

def calculate_elapsed_ms(start_time: float) -> float:
    """
    Calculate elapsed time in milliseconds using MS_TO_S constant.
    
    Args:
        start_time: Start time from perf_counter()
        
    Returns:
        float: Elapsed time in milliseconds
    """
    return (time.perf_counter() - start_time) * MS_TO_S

@contextmanager
def track_operation(metrics: Optional[Any] = None, operation_name: str = "operation", start_time: Optional[float] = None):
    """
    Unified performance tracking with proper type hints and consolidated error handling.
    
    Args:
        metrics: Optional object to store performance metrics
        operation_name: Name of the operation being tracked
        start_time: Optional custom start time for testing
        
    Raises:
        TypeError: If operation_name is not a string
    """
    if not isinstance(operation_name, str):
        raise TypeError(f"operation_name must be string, got {type(operation_name)}")

    start_time = create_timing_context(start_time)
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = (end_time - start_time) * MS_TO_S
        # Update metrics here if needed
        pass
# Future batch operation support
# NOTE: Removed @njit decorator - this function uses Python callables and time.perf_counter()
# which are not fully Numba-compatible. This is a utility function for benchmarking only.
def batch_measure_performance(operations: list, metrics: Any) -> np.ndarray:
    """
    Measure performance of multiple operations using numpy for batch processing.
    
    NOTE: This function cannot be JIT-compiled due to dynamic callable operations.
    
    Args:
        operations: List of operation callables
        metrics: Object to store aggregated performance metrics
        
    Returns:
        np.ndarray: Array of elapsed times for each operation
    """
    if not operations:
        return np.empty(0, dtype=np.float32)
        
    elapsed_times = []
    
    for operation in operations:
        start = time.perf_counter()
        operation()
        elapsed = calculate_elapsed_ms(start)
        elapsed_times.append(elapsed)
    
    elapsed_array = np.array(elapsed_times, dtype=np.float32)
    
    # Update batch metrics if available
    if metrics is not None:
        _safe_attribute_update(metrics, "total_batch_time", float(np.sum(elapsed_array)))
        _safe_increment_counter(metrics, "batch_operations", len(operations))
    
    return elapsed_array