"""
Security scanners for RLM tools (v2.9).

Contains:
- find_secrets: Detect hardcoded secrets, API keys, credentials
- find_sql_injection: Detect SQL injection vulnerabilities
- find_command_injection: Detect command injection vulnerabilities
- find_xss_vulnerabilities: Detect XSS vulnerabilities (JS/TS)
- find_python_security: Python-specific security issues
"""

import re
from typing import TYPE_CHECKING

from ..common_types import Finding, Confidence, Severity, ToolResult
from ..scan_patterns import (
    SECRET_PATTERNS,
    SQL_INJECTION_PATTERNS,
    COMMAND_INJECTION_PATTERNS,
    XSS_PATTERNS,
    SANITIZER_PATTERNS,
)

if TYPE_CHECKING:
    from ..scan_base import ScannerBase


class SecurityScanner:
    """Security-focused scanners for detecting vulnerabilities."""

    def __init__(self, base: "ScannerBase"):
        """
        Initialize with a ScannerBase instance.

        Args:
            base: ScannerBase instance providing file access and utilities
        """
        self.base = base

    def find_secrets(self) -> ToolResult:
        """
        Find hardcoded secrets in the codebase.

        Searches for:
        - API keys and tokens
        - AWS credentials
        - Private keys
        - Database connection strings
        - Hardcoded passwords
        """
        result = ToolResult(tool_name="Secret Scanner", files_scanned=len(self.base.files))

        # Skip patterns for false positives
        skip_patterns = [
            r'example',
            r'placeholder',
            r'your[_-]?api[_-]?key',
            r'xxx+',
            r'\*+',
            r'<.*>',
            r'process\.env',
            r'os\.environ',
            r'getenv',
        ]

        for pattern, issue, severity_str in SECRET_PATTERNS:
            severity = Severity[severity_str] if isinstance(severity_str, str) else severity_str
            matches = self.base._search_pattern(pattern, case_insensitive=True)

            for filepath, line_num, line, match in matches:
                # Skip if in comment
                if self.base._is_in_comment(line, filepath):
                    continue

                # Skip if matches false positive patterns
                if any(re.search(skip, line, re.IGNORECASE) for skip in skip_patterns):
                    continue

                # Skip test files for lower severity secrets
                if severity != Severity.CRITICAL and self.base._is_test_file(filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Move to environment variables or secure vault",
                    category="secrets",
                    base_confidence=Confidence.HIGH,
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} potential secrets"
        return result

    def find_sql_injection(self) -> ToolResult:
        """
        Find SQL injection vulnerabilities.

        Searches for:
        - String concatenation in SQL queries
        - f-string interpolation in SQL (Python)
        - Template literal interpolation in SQL (JavaScript)
        """
        result = ToolResult(tool_name="SQL Injection Scanner", files_scanned=len(self.base.files))

        for pattern, issue, severity_str in SQL_INJECTION_PATTERNS:
            severity = Severity[severity_str] if isinstance(severity_str, str) else severity_str
            matches = self.base._search_pattern(pattern, case_insensitive=True)

            for filepath, line_num, line, match in matches:
                if self.base._is_in_comment(line, filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Use parameterized queries or prepared statements",
                    category="sql_injection",
                    base_confidence=Confidence.HIGH,
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} potential SQL injection vulnerabilities"
        return result

    def find_command_injection(self) -> ToolResult:
        """
        Find command injection vulnerabilities.

        Searches for:
        - Shell command execution with string concatenation
        - Unsafe use of os.system, subprocess, exec, spawn
        """
        result = ToolResult(tool_name="Command Injection Scanner", files_scanned=len(self.base.files))

        for pattern, issue, severity_str in COMMAND_INJECTION_PATTERNS:
            severity = Severity[severity_str] if isinstance(severity_str, str) else severity_str
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
                    fix="Use subprocess with shell=False and list arguments",
                    category="command_injection",
                    base_confidence=Confidence.HIGH,
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} potential command injection vulnerabilities"
        return result

    def _is_sanitized(self, line: str, filepath: str, line_num: int) -> bool:
        """Check if a line uses sanitization that makes innerHTML/etc safe."""
        # Check current line for sanitizer patterns
        for sanitizer_pattern in SANITIZER_PATTERNS:
            if re.search(sanitizer_pattern, line, re.IGNORECASE):
                return True

        # Check surrounding context (look back 5 lines for sanitizer usage)
        lines = self.base._get_file_lines(filepath)
        if lines:
            start_line = max(0, line_num - 6)
            context = '\n'.join(lines[start_line:line_num])
            for sanitizer_pattern in SANITIZER_PATTERNS:
                if re.search(sanitizer_pattern, context, re.IGNORECASE):
                    return True

        return False

    def _get_xss_confidence(self, line: str, filepath: str, line_num: int) -> Confidence:
        """Determine confidence level for XSS finding based on context."""
        # Check for sanitization
        if self._is_sanitized(line, filepath, line_num):
            return Confidence.FILTERED

        # Check if it's a static string (less concerning)
        if re.search(r'\.innerHTML\s*=\s*["\'][^"\']*["\']', line):
            return Confidence.LOW

        # Check if it's in a test file
        if self.base._is_test_file(filepath):
            return Confidence.LOW

        # Check for common false positive patterns
        false_positive_patterns = [
            r'\.innerHTML\s*=\s*["\']<',  # Static HTML string
            r'\.innerHTML\s*=\s*["\']$',  # Empty string assignment
            r'\.innerHTML\s*=\s*``',       # Empty template literal
        ]
        for fp_pattern in false_positive_patterns:
            if re.search(fp_pattern, line):
                return Confidence.LOW

        return Confidence.HIGH

    def find_xss_vulnerabilities(self) -> ToolResult:
        """
        Find XSS vulnerabilities in JavaScript/TypeScript.

        Searches for:
        - innerHTML with variables
        - document.write
        - dangerouslySetInnerHTML

        Filters out:
        - Sanitized content (escapeHtml, DOMPurify, etc.)
        - Static string assignments
        - Test files (lower confidence)
        """
        result = ToolResult(tool_name="XSS Vulnerability Scanner", files_scanned=len(self.base.files))

        for pattern, issue, severity_str in XSS_PATTERNS:
            severity = Severity[severity_str] if isinstance(severity_str, str) else severity_str
            matches = self.base._search_pattern(pattern)

            for filepath, line_num, line, match in matches:
                # Only check JS/TS files
                if not any(filepath.endswith(ext) for ext in ['.js', '.ts', '.jsx', '.tsx', '.vue']):
                    continue

                if self.base._is_in_comment(line, filepath):
                    continue

                # Determine confidence based on sanitization context
                confidence = self._get_xss_confidence(line, filepath, line_num)

                # Skip filtered findings (sanitized)
                if confidence == Confidence.FILTERED:
                    continue

                # Adjust issue message for lower confidence
                adjusted_issue = issue
                if confidence == Confidence.LOW:
                    adjusted_issue = f"{issue} (verify - may be sanitized)"
                elif confidence == Confidence.MEDIUM:
                    adjusted_issue = f"{issue} (context unclear)"

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=adjusted_issue,
                    confidence=confidence,
                    severity=severity,
                    fix="Use textContent or proper sanitization (escapeHtml, DOMPurify)",
                    category="xss",
                ))

        result.summary = f"Found {len(result.findings)} potential XSS vulnerabilities"
        return result

    def find_python_security(self) -> ToolResult:
        """
        Find Python-specific security issues.

        Searches for:
        - pickle.loads with untrusted data
        - yaml.load without SafeLoader
        - Mutable default arguments
        - Bare except clauses
        """
        result = ToolResult(tool_name="Python Security Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Dangerous deserialization
            (r'''pickle\.loads?\(''', "pickle.load - unsafe deserialization", Severity.HIGH),
            (r'''yaml\.load\([^)]*(?!Loader\s*=\s*(?:yaml\.)?SafeLoader)''', "yaml.load without SafeLoader", Severity.HIGH),
            (r'''yaml\.load\([^,)]+\)(?!\s*#.*[Ss]afe)''', "yaml.load without SafeLoader", Severity.HIGH),

            # Mutable default argument
            (r'''def\s+\w+\([^)]*=\s*\[\]''', "Mutable default argument (list)", Severity.MEDIUM),
            (r'''def\s+\w+\([^)]*=\s*\{\}''', "Mutable default argument (dict)", Severity.MEDIUM),

            # Bare except
            (r'''except\s*:''', "Bare except clause", Severity.LOW),

            # eval/exec with variables
            (r'''eval\s*\([^)]*[+]''', "eval with string concatenation", Severity.CRITICAL),
            (r'''exec\s*\([^)]*[+]''', "exec with string concatenation", Severity.CRITICAL),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern, file_filter=".py")
            for filepath, line_num, line, match in matches:
                if self.base._is_in_comment(line, filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Use safe alternatives (SafeLoader, immutable defaults)",
                    category="python_security",
                    base_confidence=Confidence.HIGH,
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} Python security issues"
        return result


def create_security_scanner(base: "ScannerBase") -> SecurityScanner:
    """Factory function to create a SecurityScanner."""
    return SecurityScanner(base)
