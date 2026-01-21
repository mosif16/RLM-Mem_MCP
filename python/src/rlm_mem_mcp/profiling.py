"""
Profiling hooks for RLM v2.8+ performance monitoring.

Provides simple decorators to measure latency per phase.
"""

import time
import functools
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)


def profile_latency(phase_name: str = "operation"):
    """
    Decorator to profile latency of a function.

    Usage:
        @profile_latency("rg_search")
        def rg_search(...):
            ...

    Logs: "[LATENCY] rg_search: 45.3ms"
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info(f"[LATENCY] {phase_name}: {elapsed_ms:.1f}ms")
        return wrapper
    return decorator


def profile_memory(phase_name: str = "operation"):
    """
    Decorator to profile memory usage of a function.

    Usage:
        @profile_memory("parallel_scan")
        def parallel_scan(...):
            ...

    Logs: "[MEMORY] parallel_scan: 45.3MB peak"
    (Note: Requires psutil for accurate measurement)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                import psutil
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
            except ImportError:
                mem_before = None

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if mem_before is not None:
                    try:
                        mem_after = psutil.Process().memory_info().rss / 1024 / 1024
                        mem_delta = mem_after - mem_before
                        logger.info(f"[MEMORY] {phase_name}: {mem_delta:+.1f}MB")
                    except:
                        pass
        return wrapper
    return decorator


class LatencyTracker:
    """
    Context manager to track latency for a code block.

    Usage:
        with LatencyTracker("analysis_phase"):
            # code to measure
            ...
    """
    def __init__(self, phase_name: str = "operation"):
        self.phase_name = phase_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        logger.info(f"[LATENCY] {self.phase_name}: {elapsed_ms:.1f}ms")


# v2.8: Profiling hooks for common operations
def enable_profiling(log_level=logging.INFO):
    """Enable detailed profiling output."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def disable_profiling():
    """Disable detailed profiling output."""
    logging.getLogger(__name__).setLevel(logging.WARNING)
