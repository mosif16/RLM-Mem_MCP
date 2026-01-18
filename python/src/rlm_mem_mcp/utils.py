"""
Utility functions and decorators for RLM-Mem MCP Server.

Includes:
- Performance metrics decorator
- Memory usage monitoring
- Timing utilities
"""

import asyncio
import functools
import gc
import resource
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar, ParamSpec
from collections import defaultdict

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class PerformanceMetrics:
    """Collected performance metrics for a function call."""
    function_name: str
    elapsed_ms: float
    memory_delta_bytes: int = 0
    success: bool = True
    error: str | None = None
    call_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function_name,
            "elapsed_ms": self.elapsed_ms,
            "memory_delta_bytes": self.memory_delta_bytes,
            "success": self.success,
            "error": self.error,
        }


class MetricsCollector:
    """Collects and aggregates performance metrics."""

    def __init__(self):
        self._metrics: dict[str, list[PerformanceMetrics]] = defaultdict(list)
        self._max_entries_per_function = 1000

    def record(self, metrics: PerformanceMetrics) -> None:
        """Record a performance metric."""
        entries = self._metrics[metrics.function_name]
        entries.append(metrics)

        # Trim old entries if needed
        if len(entries) > self._max_entries_per_function:
            self._metrics[metrics.function_name] = entries[-self._max_entries_per_function:]

    def get_stats(self, function_name: str | None = None) -> dict[str, Any]:
        """Get aggregated statistics."""
        if function_name:
            entries = self._metrics.get(function_name, [])
            return self._aggregate_entries(function_name, entries)

        # Aggregate all functions
        result = {}
        for name, entries in self._metrics.items():
            result[name] = self._aggregate_entries(name, entries)
        return result

    def _aggregate_entries(self, name: str, entries: list[PerformanceMetrics]) -> dict[str, Any]:
        if not entries:
            return {"function": name, "call_count": 0}

        times = [e.elapsed_ms for e in entries]
        success_count = sum(1 for e in entries if e.success)

        return {
            "function": name,
            "call_count": len(entries),
            "success_rate": success_count / len(entries),
            "avg_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "p50_ms": sorted(times)[len(times) // 2],
            "p95_ms": sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else max(times),
            "p99_ms": sorted(times)[int(len(times) * 0.99)] if len(times) >= 100 else max(times),
        }

    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()


# Global metrics collector
_global_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return _global_collector


def get_memory_usage() -> int:
    """Get current memory usage in bytes."""
    try:
        # Try to get RSS (Resident Set Size) on Unix
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Note: ru_maxrss is in KB on Linux, bytes on macOS
        if sys.platform == "darwin":
            return usage.ru_maxrss  # Already in bytes on macOS
        return usage.ru_maxrss * 1024  # Convert KB to bytes on Linux
    except AttributeError:
        # resource module not available (e.g., Windows)
        return 0
    except Exception as e:
        # Log unexpected errors but don't crash
        print(f"Warning: get_memory_usage failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 0


def performance_metrics(
    collector: MetricsCollector | None = None,
    track_memory: bool = False,
    log_slow_calls_ms: float | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to collect performance metrics for a function.

    Args:
        collector: MetricsCollector to use (default: global collector)
        track_memory: Whether to track memory usage delta
        log_slow_calls_ms: Log calls slower than this threshold

    Example:
        @performance_metrics(track_memory=True, log_slow_calls_ms=1000)
        async def process_data(data: str) -> str:
            ...
    """
    collector = collector or _global_collector

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        func_name = f"{func.__module__}.{func.__qualname__}"

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                start_time = time.perf_counter()
                start_memory = get_memory_usage() if track_memory else 0
                error = None
                success = True

                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error = str(e)
                    success = False
                    raise
                finally:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    memory_delta = (get_memory_usage() - start_memory) if track_memory else 0

                    metrics = PerformanceMetrics(
                        function_name=func_name,
                        elapsed_ms=elapsed_ms,
                        memory_delta_bytes=memory_delta,
                        success=success,
                        error=error,
                    )
                    collector.record(metrics)

                    if log_slow_calls_ms and elapsed_ms > log_slow_calls_ms:
                        print(
                            f"SLOW: {func_name} took {elapsed_ms:.1f}ms (threshold: {log_slow_calls_ms}ms)",
                            file=sys.stderr
                        )

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                start_time = time.perf_counter()
                start_memory = get_memory_usage() if track_memory else 0
                error = None
                success = True

                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error = str(e)
                    success = False
                    raise
                finally:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    memory_delta = (get_memory_usage() - start_memory) if track_memory else 0

                    metrics = PerformanceMetrics(
                        function_name=func_name,
                        elapsed_ms=elapsed_ms,
                        memory_delta_bytes=memory_delta,
                        success=success,
                        error=error,
                    )
                    collector.record(metrics)

                    if log_slow_calls_ms and elapsed_ms > log_slow_calls_ms:
                        print(
                            f"SLOW: {func_name} took {elapsed_ms:.1f}ms (threshold: {log_slow_calls_ms}ms)",
                            file=sys.stderr
                        )

            return sync_wrapper

    return decorator


class MemoryMonitor:
    """
    Monitor memory usage with configurable limits.

    Example:
        monitor = MemoryMonitor(max_bytes=1_000_000_000)  # 1GB limit

        if monitor.check_limit():
            print("Memory limit exceeded!")
            gc.collect()
    """

    def __init__(
        self,
        max_bytes: int = 1_000_000_000,  # 1GB default
        warning_threshold: float = 0.8,  # Warn at 80%
    ):
        self.max_bytes = max_bytes
        self.warning_threshold = warning_threshold
        self._last_warning_time: float = 0
        self._warning_interval: float = 60  # Warn at most once per minute

    def get_usage(self) -> int:
        """Get current memory usage in bytes."""
        return get_memory_usage()

    def get_usage_percent(self) -> float:
        """Get memory usage as percentage of limit."""
        usage = self.get_usage()
        return usage / self.max_bytes if self.max_bytes > 0 else 0

    def check_limit(self) -> bool:
        """Check if memory limit is exceeded."""
        return self.get_usage() >= self.max_bytes

    def check_warning(self) -> bool:
        """Check if memory usage is above warning threshold."""
        usage_pct = self.get_usage_percent()

        if usage_pct >= self.warning_threshold:
            now = time.time()
            if now - self._last_warning_time >= self._warning_interval:
                self._last_warning_time = now
                return True

        return False

    def get_stats(self) -> dict[str, Any]:
        """Get memory usage statistics."""
        usage = self.get_usage()
        return {
            "current_bytes": usage,
            "max_bytes": self.max_bytes,
            "usage_percent": usage / self.max_bytes if self.max_bytes > 0 else 0,
            "warning_threshold": self.warning_threshold,
            "limit_exceeded": usage >= self.max_bytes,
        }

    def force_gc(self) -> int:
        """Force garbage collection and return bytes freed (approximate)."""
        before = self.get_usage()
        gc.collect()
        after = self.get_usage()
        return max(0, before - after)


# Global memory monitor
_global_memory_monitor: MemoryMonitor | None = None


def get_memory_monitor(max_bytes: int | None = None) -> MemoryMonitor:
    """Get or create the global memory monitor."""
    global _global_memory_monitor

    if _global_memory_monitor is None:
        _global_memory_monitor = MemoryMonitor(max_bytes or 1_000_000_000)

    return _global_memory_monitor


def timed_block(name: str):
    """
    Context manager for timing code blocks.

    Example:
        with timed_block("data_processing"):
            process_data()
    """
    class TimedBlock:
        def __init__(self, block_name: str):
            self.name = block_name
            self.start_time: float = 0
            self.elapsed_ms: float = 0

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            _global_collector.record(PerformanceMetrics(
                function_name=f"block:{self.name}",
                elapsed_ms=self.elapsed_ms,
            ))

    return TimedBlock(name)
