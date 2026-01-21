"""
Common types and utilities for RLM analysis tools.

Contains:
- Enums: Confidence, Severity
- Dataclasses: Finding, AnalysisContext, ToolResult
- Confidence calculation logic
- Performance optimization helpers
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Confidence(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    FILTERED = "FILTERED"  # L11: For false positives


class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class Finding:
    """A single finding from a search tool."""
    file: str
    line: int
    code: str
    issue: str
    confidence: Confidence = Confidence.MEDIUM
    severity: Severity = Severity.MEDIUM
    fix: str = ""
    category: str = ""

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "code": self.code,
            "issue": self.issue,
            "confidence": self.confidence.value,
            "severity": self.severity.value,
            "fix": self.fix,
            "category": self.category,
        }

    def __str__(self) -> str:
        return f"**{self.file}:{self.line}** [{self.confidence.value}] - {self.issue}\n```\n{self.code}\n```"


@dataclass
class AnalysisContext:
    """L11: Context for confidence calculation."""
    in_dead_code: bool = False
    in_test_file: bool = False
    line_verified: bool = True
    is_comment: bool = False
    pattern_match_only: bool = True
    has_semantic_verification: bool = False
    multiple_indicators: bool = False
    # v2.7: Constant detection to reduce false positives
    is_constant_declaration: bool = False  # let/const/static let patterns
    is_immutable_collection: bool = False  # Frozen arrays, immutable sets
    is_compile_time_only: bool = False     # Bundle.main, FileManager.default patterns


def calculate_confidence(finding: Finding, context: AnalysisContext) -> Confidence:
    """
    L11: Apply standardized confidence criteria.

    Scoring:
    - Start at 100 (HIGH)
    - Deduct for uncertainty factors
    - Boost for verification factors
    """
    score = 100

    # Deduct for uncertainty
    if context.pattern_match_only:
        score -= 20
    if context.in_dead_code:
        score -= 50
    if context.in_test_file:
        score -= 30
    if context.is_comment:
        score -= 40
    if not context.line_verified:
        score -= 15

    # Boost for verification
    if context.has_semantic_verification:
        score += 20
    if context.multiple_indicators:
        score += 15

    # Deduct for constant/immutable patterns
    if context.is_constant_declaration:
        score -= 35
    if context.is_immutable_collection:
        score -= 25
    if context.is_compile_time_only:
        score -= 20

    # Map to confidence level
    if score >= 80:
        return Confidence.HIGH
    elif score >= 50:
        return Confidence.MEDIUM
    elif score >= 20:
        return Confidence.LOW
    else:
        return Confidence.FILTERED


@dataclass
class ToolResult:
    """Result from a structured tool."""
    tool_name: str
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""
    files_scanned: int = 0
    errors: list[str] = field(default_factory=list)
    # Optional: file line counts for validation
    _file_line_counts: dict[str, int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.findings)

    @property
    def high_confidence(self) -> list[Finding]:
        return [f for f in self.findings if f.confidence == Confidence.HIGH]

    @property
    def by_severity(self) -> dict[str, list[Finding]]:
        result: dict[str, list[Finding]] = {}
        for f in self.findings:
            sev = f.severity.value
            if sev not in result:
                result[sev] = []
            result[sev].append(f)
        return result

    def validate_line_numbers(self, file_line_counts: dict[str, int]) -> 'ToolResult':
        """
        Validate and filter findings with invalid line numbers.

        Args:
            file_line_counts: Dict mapping filepath to total line count

        Returns:
            New ToolResult with invalid line numbers filtered or corrected
        """
        validated_findings = []
        invalid_count = 0

        for finding in self.findings:
            max_lines = file_line_counts.get(finding.file, 0)

            # Try partial path match if exact match fails
            if max_lines == 0:
                for path, count in file_line_counts.items():
                    if finding.file in path or path.endswith(finding.file):
                        max_lines = count
                        break

            if max_lines > 0 and finding.line > max_lines:
                # Invalid line number - either skip or mark as low confidence
                invalid_count += 1
                # Create a corrected finding with clamped line number
                corrected = Finding(
                    file=finding.file,
                    line=max_lines,  # Clamp to last line
                    code=f"{finding.code} [LINE CORRECTED: was {finding.line}, max {max_lines}]",
                    issue=finding.issue,
                    confidence=Confidence.LOW,  # Lower confidence due to correction
                    severity=finding.severity,
                    fix=finding.fix,
                    category=finding.category,
                )
                validated_findings.append(corrected)
            else:
                validated_findings.append(finding)

        # Create new result with validated findings
        result = ToolResult(
            tool_name=self.tool_name,
            findings=validated_findings,
            summary=self.summary,
            files_scanned=self.files_scanned,
            errors=self.errors.copy(),
            _file_line_counts=file_line_counts,
        )

        if invalid_count > 0:
            result.errors.append(f"{invalid_count} findings had invalid line numbers (corrected)")

        return result

    def to_markdown(self) -> str:
        """Format results as markdown."""
        if not self.findings:
            return f"## {self.tool_name}\n\nNo findings. Scanned {self.files_scanned} files."

        lines = [
            f"## {self.tool_name}",
            "",
            f"**Summary**: {self.summary}",
            f"**Findings**: {self.count} ({len(self.high_confidence)} high confidence)",
            "",
        ]

        # Group by confidence
        for conf in [Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW]:
            conf_findings = [f for f in self.findings if f.confidence == conf]
            if conf_findings:
                lines.append(f"### {conf.value} Confidence ({len(conf_findings)})")
                for finding in conf_findings:
                    lines.append(f"\n{finding}")
                lines.append("")

        return "\n".join(lines)

    def to_json(self) -> dict:
        """Export as JSON-serializable dict."""
        return {
            "tool": self.tool_name,
            "summary": self.summary,
            "files_scanned": self.files_scanned,
            "findings_count": self.count,
            "high_confidence_count": len(self.high_confidence),
            "findings": [f.to_dict() for f in self.findings],
            "errors": self.errors,
        }


def get_optimal_workers(max_limit: int = 16, min_limit: int = 4) -> int:
    """
    Calculate optimal number of worker threads based on CPU count.

    v2.8: Dynamic thread pool sizing for better parallelism.

    Formula: max(min_limit, min(max_limit, os.cpu_count() or min_limit))

    Examples:
    - 2 CPU system: 4 workers (enforces minimum)
    - 4 CPU system: 4 workers
    - 8 CPU system: 8 workers
    - 16+ CPU system: 16 workers (enforces maximum)

    Args:
        max_limit: Maximum workers to allow (default: 16)
        min_limit: Minimum workers to enforce (default: 4)

    Returns:
        Optimal worker count for current system
    """
    import os
    cpu_count = os.cpu_count() or min_limit
    return max(min_limit, min(max_limit, cpu_count))


def get_optimal_batch_size(pattern_count: int, data_size_mb: int = 100) -> int:
    """
    v2.8: Calculate optimal batch size for parallel_rg_search based on data and pattern complexity.

    Formula: min(patterns, 16, max(2, data_size_mb // 50))
    - Adapts to data size: Smaller batches for large data
    - Respects pattern count: Don't batch more than needed
    - Minimum 2: Always batch at least 2 patterns
    - Maximum 16: Don't exceed worker count

    Args:
        pattern_count: Number of patterns to search
        data_size_mb: Estimated data size in MB (default: 100)

    Returns:
        Recommended batch size (2-16)
    """
    # Heuristic: 50MB per batch is reasonable
    size_based = max(2, data_size_mb // 50)
    return min(pattern_count, 16, max(2, size_based))
