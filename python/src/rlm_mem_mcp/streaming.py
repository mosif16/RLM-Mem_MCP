"""
Streaming results support for RLM v2.9+.

Provides progressive result delivery as findings are discovered.
v2.8: Initial framework and API design
v2.9: Full implementation with streaming execution
"""

from typing import Callable, Generator, Any
from dataclasses import dataclass


@dataclass
class StreamingResult:
    """A result yielded progressively during streaming."""
    phase: str          # e.g., "security", "quality"
    finding_count: int  # Number of findings so far
    total_found: int    # Total unique findings
    current_finding: Any = None
    is_final: bool = False


def stream_analysis(func: Callable) -> Callable:
    """
    Decorator to enable streaming results from analysis function.

    Usage:
        @stream_analysis
        def find_secrets():
            yield StreamingResult(phase="secrets", finding_count=1, total_found=1)
            ...

    v2.8: Stub implementation (collects all results first)
    v2.9: Stream results as discovered (progressive delivery)
    """
    def wrapper(*args, **kwargs) -> Generator[StreamingResult, None, None]:
        # v2.8: Stub - collect all, then yield
        results = []
        try:
            result = func(*args, **kwargs)
            if result:
                results.append(result)
        except Exception as e:
            results.append(StreamingResult(
                phase="error",
                finding_count=0,
                total_found=0,
                current_finding=str(e),
                is_final=True
            ))

        # Yield collected results
        for i, result in enumerate(results):
            result.is_final = (i == len(results) - 1)
            yield result

    return wrapper


class StreamingAnalyzer:
    """
    Streaming analysis executor for progressive result delivery.

    v2.8: Framework structure
    v2.9: Full streaming implementation
    """
    def __init__(self):
        self.findings = []
        self.current_phase = None

    def update_finding(self, phase: str, finding: Any):
        """Record a finding and yield progress."""
        self.current_phase = phase
        self.findings.append(finding)
        return StreamingResult(
            phase=phase,
            finding_count=len(self.findings),
            total_found=len(set(str(f) for f in self.findings)),
            current_finding=finding
        )

    def finish(self):
        """Mark analysis as complete."""
        return StreamingResult(
            phase=self.current_phase or "complete",
            finding_count=len(self.findings),
            total_found=len(self.findings),
            is_final=True
        )
