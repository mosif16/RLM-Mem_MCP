"""
Code quality scanners for RLM tools (v2.9).

Contains:
- find_todos: Find TODO, FIXME, HACK, XXX comments
- find_long_functions: Find functions exceeding line limit
- find_complex_functions: Find high cyclomatic complexity
- find_code_smells: Find common code smells
- find_dead_code: Find potentially unused code
"""

import re
from typing import TYPE_CHECKING

from ..common_types import Finding, Confidence, Severity, ToolResult
from ..scan_patterns import QUALITY_PATTERNS

if TYPE_CHECKING:
    from ..scan_base import ScannerBase


class QualityScanner:
    """Code quality scanners."""

    def __init__(self, base: "ScannerBase"):
        self.base = base

    def find_todos(self) -> ToolResult:
        """Find TODO, FIXME, HACK, XXX comments."""
        result = ToolResult(tool_name="TODO Scanner", files_scanned=len(self.base.files))

        for pattern, severity_str in QUALITY_PATTERNS['todo']:
            severity = Severity[severity_str] if isinstance(severity_str, str) else Severity.INFO
            matches = self.base._search_pattern(pattern)

            for filepath, line_num, line, match in matches:
                tag = match.group(1)
                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=f"{tag} comment",
                    confidence=Confidence.HIGH,
                    severity=severity,
                    category="todo",
                ))

        result.summary = f"Found {len(result.findings)} TODO/FIXME comments"
        return result

    def find_long_functions(self, max_lines: int = 30) -> ToolResult:
        """
        Find functions that are too long.

        Args:
            max_lines: Maximum acceptable function length (default: 30)
        """
        result = ToolResult(tool_name=f"Long Function Scanner (>{max_lines} lines)", files_scanned=len(self.base.files))

        # Function start patterns by language
        func_patterns = QUALITY_PATTERNS['long_function']

        for filepath in self.base.files:
            lines = self.base._get_file_lines(filepath)
            if not lines:
                continue

            # Find applicable pattern
            pattern = None
            for p, ext in func_patterns:
                if filepath.endswith(ext):
                    pattern = p
                    break

            if not pattern:
                continue

            # Find functions and their lengths
            func_starts = []
            for i, line in enumerate(lines):
                match = re.match(pattern, line)
                if match:
                    func_starts.append((i, match.group(1)))

            # Calculate lengths
            for i, (start_line, func_name) in enumerate(func_starts):
                end_line = func_starts[i + 1][0] if i + 1 < len(func_starts) else len(lines)
                length = end_line - start_line

                if length > max_lines:
                    # Determine severity based on length
                    if length > 100:
                        severity = Severity.HIGH
                    elif length > 60:
                        severity = Severity.MEDIUM
                    else:
                        severity = Severity.LOW

                    result.findings.append(Finding(
                        file=filepath,
                        line=start_line + 1,
                        code=f"Function '{func_name}' is {length} lines long",
                        issue=f"Function too long ({length} lines)",
                        confidence=Confidence.HIGH,
                        severity=severity,
                        fix=f"Refactor into smaller functions (aim for <{max_lines} lines)",
                        category="long_function",
                    ))

        result.summary = f"Found {len(result.findings)} functions over {max_lines} lines"
        return result

    def find_complex_functions(self, max_complexity: int = 10) -> ToolResult:
        """
        Find functions with high cyclomatic complexity.

        Counts decision points (if, elif, for, while, and, or, except, case).

        Args:
            max_complexity: Maximum acceptable complexity (default: 10)
        """
        result = ToolResult(tool_name=f"Complexity Scanner (>{max_complexity})", files_scanned=len(self.base.files))

        # Decision point patterns
        decision_patterns = [
            r'\bif\b',
            r'\belif\b',
            r'\bfor\b',
            r'\bwhile\b',
            r'\band\b',
            r'\bor\b',
            r'\bexcept\b',
            r'\bcase\b',
            r'\?\s*:',  # Ternary operator
        ]

        for filepath in self.base.files:
            if not any(filepath.endswith(ext) for ext in ['.py', '.swift', '.js', '.ts', '.go', '.rs']):
                continue

            lines = self.base._get_file_lines(filepath)
            if not lines:
                continue

            content = '\n'.join(lines)

            # Simple function detection
            func_pattern = r'(?:def|func|function|fn)\s+(\w+)'
            for match in re.finditer(func_pattern, content):
                func_name = match.group(1)
                func_start = match.start()

                # Find function end (next function or end of file)
                next_func = re.search(func_pattern, content[match.end():])
                func_end = match.end() + next_func.start() if next_func else len(content)
                func_content = content[func_start:func_end]

                # Count complexity
                complexity = 1  # Base complexity
                for dp_pattern in decision_patterns:
                    complexity += len(re.findall(dp_pattern, func_content))

                if complexity > max_complexity:
                    severity = Severity.HIGH if complexity > 20 else Severity.MEDIUM

                    # Get line number
                    line_num = content[:func_start].count('\n') + 1

                    result.findings.append(Finding(
                        file=filepath,
                        line=line_num,
                        code=f"Function '{func_name}' has complexity {complexity}",
                        issue=f"High cyclomatic complexity ({complexity})",
                        confidence=Confidence.MEDIUM,
                        severity=severity,
                        fix="Simplify by extracting methods or using polymorphism",
                        category="complexity",
                    ))

        result.summary = f"Found {len(result.findings)} complex functions"
        return result

    def find_code_smells(self) -> ToolResult:
        """Find common code smells."""
        result = ToolResult(tool_name="Code Smell Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Magic numbers
            (r'(?<![\w.])\b(?!0|1|2|10|100|1000)\d{2,}\b(?!\s*[=<>])', "Magic number - use named constant", Severity.LOW),
            # Deeply nested code (4+ levels)
            (r'^(\s{16,}|\t{4,})\S', "Deeply nested code - consider refactoring", Severity.MEDIUM),
            # Multiple return statements
            (r'return\s+[^;]+;.*return\s+[^;]+;.*return', "Multiple returns - consider early exit pattern", Severity.LOW),
            # Long parameter list
            (r'def\s+\w+\s*\([^)]{100,}\)', "Long parameter list - consider parameter object", Severity.LOW),
            # Empty except/catch
            (r'(?:except|catch)\s*[^:]*:\s*(?:pass|\.\.\.|\{\s*\})', "Empty exception handler", Severity.HIGH),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if self.base._is_in_comment(line, filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    category="code_smell",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} code smells"
        return result

    def find_dead_code(self) -> ToolResult:
        """Find potentially unused/dead code."""
        result = ToolResult(tool_name="Dead Code Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Unreachable code after return
            (r'return\s+[^;]+;\s*\n\s*[^}]', "Code after return statement", Severity.MEDIUM),
            # Commented out code blocks
            (r'(?://|#)\s*(?:def|func|class|if|for|while)\s+\w+', "Commented out code", Severity.LOW),
            # Unused imports (basic detection)
            (r'^import\s+(\w+)(?!.*\1)', "Potentially unused import", Severity.LOW),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    category="dead_code",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} potential dead code issues"
        return result


def create_quality_scanner(base: "ScannerBase") -> QualityScanner:
    """Factory function to create a QualityScanner."""
    return QualityScanner(base)
