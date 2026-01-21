"""
Result Aggregation for RLM Processing.

Provides result structures and aggregation utilities:
- Progress events for streaming updates
- Chunk and final result structures
- Result merging and deduplication
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProgressEvent:
    """Structured progress event for streaming updates."""
    event_type: str  # "start", "file_collected", "chunk_analyzed", "finding_verified", "complete", "error"
    message: str
    progress_percent: float
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "message": self.message,
            "progress_percent": self.progress_percent,
            "details": self.details,
        }


@dataclass
class ChunkResult:
    """Result from processing a single chunk."""

    chunk_id: int
    content_preview: str  # First 200 chars
    response: str
    token_count: int
    relevance_score: float
    processing_time_ms: int


@dataclass
class RLMResult:
    """Final result from RLM processing."""

    query: str
    scope: str  # Description of what was processed
    response: str
    chunk_results: list[ChunkResult] = field(default_factory=list)
    total_tokens_processed: int = 0
    total_api_calls: int = 0
    cache_hits: int = 0
    processing_time_ms: int = 0
    truncated: bool = False
    error: str | None = None


def merge_chunk_results(results: list[ChunkResult], max_tokens: int = 4000) -> str:
    """
    Merge multiple chunk results into a single response.

    Args:
        results: List of chunk results to merge
        max_tokens: Maximum tokens for merged result

    Returns:
        Merged response string
    """
    if not results:
        return "No results to merge."

    # Sort by relevance score (highest first)
    sorted_results = sorted(results, key=lambda r: r.relevance_score, reverse=True)

    merged_parts = []
    total_chars = 0
    max_chars = max_tokens * 4  # Rough estimate

    for result in sorted_results:
        if total_chars + len(result.response) > max_chars:
            break
        merged_parts.append(result.response)
        total_chars += len(result.response)

    return "\n\n---\n\n".join(merged_parts)


def deduplicate_findings(findings: list[dict]) -> list[dict]:
    """
    Remove duplicate findings based on file:line key.

    Args:
        findings: List of finding dictionaries with 'file' and 'line' keys

    Returns:
        Deduplicated list of findings
    """
    seen = set()
    unique = []

    for finding in findings:
        key = (finding.get('file', ''), finding.get('line', 0))
        if key not in seen:
            seen.add(key)
            unique.append(finding)

    return unique


def aggregate_scanner_results(results: list[Any]) -> dict:
    """
    Aggregate results from multiple scanner runs.

    Args:
        results: List of ToolResult objects from scanners

    Returns:
        Aggregated statistics and findings
    """
    total_findings = 0
    high_confidence = 0
    medium_confidence = 0
    low_confidence = 0
    all_findings = []

    for result in results:
        if hasattr(result, 'count'):
            total_findings += result.count
        if hasattr(result, 'high_confidence'):
            high_confidence += len(result.high_confidence)
        if hasattr(result, 'findings'):
            for f in result.findings:
                confidence = getattr(f, 'confidence', None)
                if confidence:
                    if confidence.value == 'HIGH':
                        high_confidence += 1
                    elif confidence.value == 'MEDIUM':
                        medium_confidence += 1
                    else:
                        low_confidence += 1
                all_findings.append(f)

    return {
        'total_findings': total_findings,
        'high_confidence': high_confidence,
        'medium_confidence': medium_confidence,
        'low_confidence': low_confidence,
        'findings': all_findings,
    }


def format_findings_markdown(findings: list[Any], title: str = "Findings") -> str:
    """
    Format findings list as markdown.

    Args:
        findings: List of Finding objects
        title: Section title

    Returns:
        Markdown formatted string
    """
    if not findings:
        return f"## {title}\n\nNo findings."

    lines = [f"## {title}", f"**Count:** {len(findings)}", ""]

    for i, finding in enumerate(findings, 1):
        file_path = getattr(finding, 'file', 'unknown')
        line_num = getattr(finding, 'line', 0)
        code = getattr(finding, 'code', '')
        message = getattr(finding, 'message', '')
        confidence = getattr(finding, 'confidence', None)

        conf_str = f"[{confidence.value}]" if confidence else ""

        lines.append(f"### {i}. {file_path}:{line_num} {conf_str}")
        if message:
            lines.append(f"**Issue:** {message}")
        if code:
            lines.append(f"```\n{code[:200]}\n```")
        lines.append("")

    return "\n".join(lines)
