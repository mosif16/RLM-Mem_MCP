"""
Structured Output System for RLM Processing

Provides JSON-based structured output for:
1. Findings with validated file:line references
2. Progress events during analysis
3. Machine-parseable results for verification

This reduces hallucinations by forcing structured output that can be validated.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Literal
from enum import Enum


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class FindingCategory(str, Enum):
    SECURITY = "security"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    ARCHITECTURE = "architecture"
    STYLE = "style"


@dataclass
class StructuredFinding:
    """A single finding with all required fields."""
    file: str
    line: int
    category: str
    title: str
    description: str
    confidence: str = "MEDIUM"
    code_snippet: str = ""
    severity: str = "medium"  # high, medium, low
    dead_code: bool = False
    dead_code_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        confidence_note = ""
        if self.dead_code:
            confidence_note = f" - {self.dead_code_reason}"

        parts = [
            f"**{self.file}:{self.line}** [{self.severity.upper()}] [Confidence: {self.confidence}{confidence_note}]",
            f"_{self.title}_: {self.description}",
        ]
        if self.code_snippet:
            parts.append(f"```\n{self.code_snippet}\n```")
        return "\n".join(parts)


@dataclass
class StructuredProgress:
    """Progress event during analysis."""
    event: str  # "start", "chunk", "verify", "complete", "error"
    message: str
    current: int = 0
    total: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredResult:
    """Complete structured result from RLM analysis."""
    query: str
    findings: list[StructuredFinding] = field(default_factory=list)
    summary: str = ""
    files_analyzed: int = 0
    chunks_processed: int = 0
    verification_stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "files_analyzed": self.files_analyzed,
            "chunks_processed": self.chunks_processed,
            "verification_stats": self.verification_stats,
            "errors": self.errors,
            "warnings": self.warnings,
            "finding_count": len(self.findings),
            "by_severity": self._count_by_severity(),
            "by_confidence": self._count_by_confidence(),
        }

    def _count_by_severity(self) -> dict[str, int]:
        counts = {"high": 0, "medium": 0, "low": 0}
        for f in self.findings:
            sev = f.severity.lower()
            if sev in counts:
                counts[sev] += 1
        return counts

    def _count_by_confidence(self) -> dict[str, int]:
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for f in self.findings:
            conf = f.confidence.upper()
            if conf in counts:
                counts[conf] += 1
        return counts

    def to_markdown(self) -> str:
        """Convert to markdown report."""
        parts = [
            f"## Analysis Results",
            "",
            f"**Query:** {self.query}",
            f"**Files analyzed:** {self.files_analyzed}",
            f"**Findings:** {len(self.findings)}",
            "",
        ]

        if self.errors:
            parts.append("### Errors")
            for err in self.errors:
                parts.append(f"- {err}")
            parts.append("")

        if self.warnings:
            parts.append("### Warnings")
            for warn in self.warnings[:5]:
                parts.append(f"- {warn}")
            if len(self.warnings) > 5:
                parts.append(f"- ... and {len(self.warnings) - 5} more")
            parts.append("")

        if self.findings:
            # Group by severity
            by_severity = {"high": [], "medium": [], "low": []}
            for f in self.findings:
                sev = f.severity.lower()
                if sev in by_severity:
                    by_severity[sev].append(f)

            for severity in ["high", "medium", "low"]:
                findings = by_severity[severity]
                if findings:
                    parts.append(f"### {severity.upper()} Severity ({len(findings)})")
                    parts.append("")
                    for f in findings[:20]:  # Limit per category
                        parts.append(f.to_markdown())
                        parts.append("")
                    if len(findings) > 20:
                        parts.append(f"... and {len(findings) - 20} more")
                        parts.append("")
        else:
            parts.append("No findings matched the query.")

        if self.summary:
            parts.append("### Summary")
            parts.append(self.summary)

        return "\n".join(parts)


class StructuredOutputParser:
    """
    Parse LLM output into structured findings.

    Handles various output formats and extracts findings into StructuredFinding objects.
    """

    # Pattern for file:line references
    FILE_LINE_PATTERN = re.compile(
        r'[*`]*([^\s*`:\[\]]+\.[a-zA-Z]{1,10}):(\d+)[*`]*'
    )

    # Pattern for confidence levels
    CONFIDENCE_PATTERN = re.compile(
        r'\[Confidence:\s*(HIGH|MEDIUM|LOW)(?:\s*-\s*([^\]]+))?\]',
        re.IGNORECASE
    )

    # Pattern for severity
    SEVERITY_PATTERN = re.compile(
        r'\[(HIGH|MEDIUM|LOW|CRITICAL)\]',
        re.IGNORECASE
    )

    def parse_response(self, response: str, default_category: str = "general") -> list[StructuredFinding]:
        """
        Parse LLM response into structured findings.

        Extracts file:line references with context and creates StructuredFinding objects.
        """
        findings = []
        lines = response.split('\n')

        current_finding = None
        current_snippet = []
        in_code_block = False

        for i, line in enumerate(lines):
            # Check for code block markers
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block
                    if current_finding and current_snippet:
                        current_finding.code_snippet = '\n'.join(current_snippet)
                    current_snippet = []
                in_code_block = not in_code_block
                continue

            if in_code_block:
                current_snippet.append(line)
                continue

            # Check for file:line reference
            file_match = self.FILE_LINE_PATTERN.search(line)
            if file_match:
                # Save previous finding if exists
                if current_finding:
                    findings.append(current_finding)

                file_path = file_match.group(1)
                line_num = int(file_match.group(2))

                # Extract confidence
                conf_match = self.CONFIDENCE_PATTERN.search(line)
                confidence = conf_match.group(1).upper() if conf_match else "MEDIUM"
                dead_code_reason = conf_match.group(2) if conf_match and conf_match.group(2) else ""

                # Extract severity
                sev_match = self.SEVERITY_PATTERN.search(line)
                severity = sev_match.group(1).lower() if sev_match else "medium"
                if severity == "critical":
                    severity = "high"

                # Get description from surrounding context
                description = self._extract_description(lines, i)

                current_finding = StructuredFinding(
                    file=file_path,
                    line=line_num,
                    category=default_category,
                    title=self._extract_title(line),
                    description=description,
                    confidence=confidence,
                    severity=severity,
                    dead_code="dead" in dead_code_reason.lower() or "#if" in dead_code_reason.lower(),
                    dead_code_reason=dead_code_reason,
                )
                current_snippet = []

        # Don't forget last finding
        if current_finding:
            findings.append(current_finding)

        return findings

    def _extract_title(self, line: str) -> str:
        """Extract a short title from the line."""
        # Remove file:line reference and confidence markers
        cleaned = self.FILE_LINE_PATTERN.sub('', line)
        cleaned = self.CONFIDENCE_PATTERN.sub('', cleaned)
        cleaned = self.SEVERITY_PATTERN.sub('', cleaned)
        # Remove markdown formatting
        cleaned = re.sub(r'[*_`#]', '', cleaned)
        return cleaned.strip()[:100] or "Finding"

    def _extract_description(self, lines: list[str], current_idx: int) -> str:
        """Extract description from surrounding lines."""
        description_parts = []
        # Look at next few lines for description
        for j in range(current_idx + 1, min(current_idx + 4, len(lines))):
            line = lines[j].strip()
            if not line or line.startswith('```') or self.FILE_LINE_PATTERN.search(line):
                break
            if not line.startswith('#'):
                description_parts.append(line)
        return ' '.join(description_parts)[:500]


# JSON Schema for structured output (can be used with Claude's JSON mode)
FINDING_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "File path (must exist in analyzed files)"},
                    "line": {"type": "integer", "minimum": 1, "description": "Line number (must be valid)"},
                    "category": {"type": "string", "enum": ["security", "quality", "performance", "architecture"]},
                    "title": {"type": "string", "maxLength": 100},
                    "description": {"type": "string", "maxLength": 500},
                    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "code_snippet": {"type": "string", "description": "Actual code from the file"},
                },
                "required": ["file", "line", "category", "title", "confidence", "severity"],
            }
        },
        "summary": {"type": "string", "maxLength": 1000},
    },
    "required": ["findings"],
}


def create_structured_prompt_suffix() -> str:
    """
    Create a prompt suffix that instructs the LLM to output structured findings.
    """
    return """

## OUTPUT FORMAT (REQUIRED)

You MUST format each finding as:

**FILE:LINE** [SEVERITY] [Confidence: LEVEL]
Title: Brief issue description
```
actual code snippet from the file
```
Description: What the issue is and why it matters.

Where:
- FILE = exact file path from the file list
- LINE = verified line number (use verify_line() first!)
- SEVERITY = HIGH, MEDIUM, or LOW
- LEVEL = HIGH (verified), MEDIUM (likely), or LOW (uncertain/dead code)

Example:
**src/auth/login.py:42** [HIGH] [Confidence: HIGH]
Title: Hardcoded API credential
```
API_SECRET = "sk-live-abc123..."
```
Description: Production API key hardcoded in source. Should use environment variable.

BEFORE reporting any finding:
1. Use verify_line(file, line, pattern) to confirm the code exists
2. Use check_dead_code(file) to see if it's in an #if false block
3. Assign appropriate confidence based on verification results
"""
