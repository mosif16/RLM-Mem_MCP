"""
Progress callbacks for long-running analyses in RLM v2.9+.

Provides hooks to report progress during analysis execution.
v2.8: Initial framework and callback types
v2.9: Full implementation with progress event firing
"""

from typing import Callable, Optional
from dataclasses import dataclass
from enum import Enum


class ProgressEventType(Enum):
    """Types of progress events during analysis."""
    STARTED = "started"
    PHASE_STARTED = "phase_started"
    FINDING = "finding"
    PHASE_COMPLETED = "phase_completed"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProgressEvent:
    """A progress update event."""
    type: ProgressEventType
    phase: str = ""
    message: str = ""
    current: int = 0
    total: int = 0
    percentage: float = 0.0


class ProgressCallback:
    """
    Base class for progress callbacks.

    v2.8: Framework structure
    v2.9: Called during analysis execution
    """
    def on_progress(self, event: ProgressEvent) -> None:
        """Handle a progress event."""
        raise NotImplementedError


class LoggingProgressCallback(ProgressCallback):
    """Logs progress events."""

    def on_progress(self, event: ProgressEvent) -> None:
        import logging
        logger = logging.getLogger(__name__)

        if event.type == ProgressEventType.STARTED:
            logger.info(f"[PROGRESS] Analysis started: {event.phase}")
        elif event.type == ProgressEventType.PHASE_STARTED:
            logger.info(f"[PROGRESS] Phase started: {event.phase}")
        elif event.type == ProgressEventType.FINDING:
            logger.debug(f"[PROGRESS] Finding in {event.phase}: {event.message}")
        elif event.type == ProgressEventType.PHASE_COMPLETED:
            logger.info(f"[PROGRESS] Phase completed: {event.phase} ({event.current}/{event.total})")
        elif event.type == ProgressEventType.COMPLETED:
            logger.info(f"[PROGRESS] Analysis completed: {event.current} findings")
        elif event.type == ProgressEventType.ERROR:
            logger.error(f"[PROGRESS] Error in {event.phase}: {event.message}")


class ProgressTracker:
    """
    Tracks progress and fires callbacks.

    v2.8: Stub implementation
    v2.9: Called during analysis
    """
    def __init__(self, callback: Optional[ProgressCallback] = None):
        self.callback = callback or LoggingProgressCallback()
        self.current_phase = ""
        self.findings_count = 0

    def on_analysis_started(self, phase: str = "analysis"):
        """Called when analysis starts."""
        self.current_phase = phase
        self.callback.on_progress(ProgressEvent(
            type=ProgressEventType.STARTED,
            phase=phase
        ))

    def on_phase_started(self, phase: str):
        """Called when a phase starts."""
        self.current_phase = phase
        self.callback.on_progress(ProgressEvent(
            type=ProgressEventType.PHASE_STARTED,
            phase=phase
        ))

    def on_finding(self, message: str, total: int = 0):
        """Called when a finding is made."""
        self.findings_count += 1
        self.callback.on_progress(ProgressEvent(
            type=ProgressEventType.FINDING,
            phase=self.current_phase,
            message=message,
            current=self.findings_count,
            total=total
        ))

    def on_phase_completed(self, total: int = 0):
        """Called when a phase completes."""
        self.callback.on_progress(ProgressEvent(
            type=ProgressEventType.PHASE_COMPLETED,
            phase=self.current_phase,
            current=self.findings_count,
            total=total
        ))

    def on_analysis_completed(self):
        """Called when analysis completes."""
        self.callback.on_progress(ProgressEvent(
            type=ProgressEventType.COMPLETED,
            current=self.findings_count
        ))

    def on_error(self, message: str):
        """Called when an error occurs."""
        self.callback.on_progress(ProgressEvent(
            type=ProgressEventType.ERROR,
            phase=self.current_phase,
            message=message
        ))
