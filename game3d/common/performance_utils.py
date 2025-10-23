@contextmanager
def track_operation(metrics, operation_name):
    """Unified performance tracking."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start_time) * 1000
        if hasattr(metrics, operation_name):
            setattr(metrics, operation_name, getattr(metrics, operation_name) + elapsed)
        # Also update call counts
        call_attr = f"{operation_name}_calls"
        if hasattr(metrics, call_attr):
            setattr(metrics, call_attr, getattr(metrics, call_attr) + 1)
