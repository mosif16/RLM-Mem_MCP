"""
Structured Search Tools for RLM REPL

These tools provide preconfigured search patterns that agents can call directly
instead of writing raw Python code. Each tool returns structured output.

Usage in REPL:
    results = find_secrets()
    results = find_sql_injection()
    results = find_force_unwraps()
    results = map_architecture()

All tools return a ToolResult with:
    - findings: List of structured findings
    - summary: One-line summary
    - confidence: Overall confidence level
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum


class Confidence(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


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
class ToolResult:
    """Result from a structured tool."""
    tool_name: str
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""
    files_scanned: int = 0
    errors: list[str] = field(default_factory=list)

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
                lines.append("")
                for f in conf_findings[:20]:  # Limit per category
                    lines.append(str(f))
                    if f.fix:
                        lines.append(f"*Fix: {f.fix}*")
                    lines.append("")
                if len(conf_findings) > 20:
                    lines.append(f"*... and {len(conf_findings) - 20} more*")
                    lines.append("")

        return "\n".join(lines)


class StructuredTools:
    """
    Preconfigured search tools for common analysis patterns.

    Each tool encapsulates:
    - Search pattern (regex)
    - Context extraction
    - Confidence assessment
    - Structured output
    """

    def __init__(self, content: str, file_index: dict[str, tuple[int, int]] | None = None):
        """
        Initialize with content to search.

        Args:
            content: The full content (prompt variable)
            file_index: Optional precomputed {filepath: (start_pos, end_pos)}
        """
        self.content = content
        self.file_index = file_index or self._build_file_index()
        self.files = list(self.file_index.keys())

    def _build_file_index(self) -> dict[str, tuple[int, int]]:
        """Build index of file positions in content."""
        index = {}
        pattern = r'### File: ([^\n]+)\n'

        matches = list(re.finditer(pattern, self.content))
        for i, match in enumerate(matches):
            filepath = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(self.content)
            index[filepath] = (start, end)

        return index

    def _get_file_content(self, filepath: str) -> str | None:
        """Get content of a specific file."""
        if filepath not in self.file_index:
            # Try partial match
            for fp in self.file_index:
                if filepath in fp or fp.endswith(filepath):
                    start, end = self.file_index[fp]
                    return self.content[start:end]
            return None
        start, end = self.file_index[filepath]
        return self.content[start:end]

    def _get_file_lines(self, filepath: str) -> list[str] | None:
        """Get lines of a specific file."""
        content = self._get_file_content(filepath)
        if content is None:
            return None
        return content.split('\n')

    def _search_pattern(
        self,
        pattern: str,
        file_filter: str | None = None,
        exclude_pattern: str | None = None,
    ) -> list[tuple[str, int, str, re.Match]]:
        """
        Search for regex pattern across all files.

        Returns: List of (filepath, line_num, line_content, match)
        """
        results = []

        for filepath, (start, end) in self.file_index.items():
            # Apply file filter
            if file_filter and not filepath.endswith(file_filter):
                continue

            file_content = self.content[start:end]
            lines = file_content.split('\n')

            for line_num, line in enumerate(lines, 1):
                # Skip if matches exclude pattern
                if exclude_pattern and re.search(exclude_pattern, line):
                    continue

                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    results.append((filepath, line_num, line.strip(), match))

        return results

    def _is_in_comment(self, line: str, filepath: str) -> bool:
        """Check if a line is a comment."""
        stripped = line.strip()

        # Python/Shell comments
        if stripped.startswith('#'):
            return True

        # C-style comments
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            return True

        # Swift/Kotlin doc comments
        if stripped.startswith('///'):
            return True

        return False

    def _is_in_string(self, line: str, match_pos: int) -> bool:
        """Check if match position is inside a string literal."""
        # Simple heuristic: count quotes before position
        before = line[:match_pos]
        single_quotes = before.count("'") - before.count("\\'")
        double_quotes = before.count('"') - before.count('\\"')

        return (single_quotes % 2 == 1) or (double_quotes % 2 == 1)

    def _is_code_file(self, filepath: str) -> bool:
        """Check if file is actual code (not docs/config/tests)."""
        # Documentation files - skip for security scans
        doc_extensions = {'.md', '.markdown', '.rst', '.txt', '.adoc', '.html', '.htm'}
        # Config files that shouldn't have security issues flagged
        config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'}

        lower = filepath.lower()
        ext = '.' + lower.rsplit('.', 1)[-1] if '.' in lower else ''

        if ext in doc_extensions:
            return False

        # Skip README, CHANGELOG, etc.
        basename = filepath.rsplit('/', 1)[-1].lower()
        if basename in {'readme.md', 'changelog.md', 'contributing.md', 'license', 'license.md'}:
            return False

        return True

    def _is_test_file(self, filepath: str) -> bool:
        """Check if file is a test file."""
        lower = filepath.lower()
        return any(x in lower for x in ['test', 'spec', 'mock', 'fixture', '__tests__'])

    def _is_swift_safe_try(self, line: str, context_lines: list[str] | None = None) -> bool:
        """
        Check if try! is safe (compile-time constant like regex).

        Safe patterns:
        - NSRegularExpression with string literal
        - Static/constant initialization
        - Decoder/Encoder with fixed types
        """
        # Regex initialization is safe - pattern is compile-time constant
        if 'NSRegularExpression' in line or 'Regex(' in line:
            return True

        # JSONDecoder/Encoder with type literal is usually safe
        if ('JSONDecoder' in line or 'JSONEncoder' in line) and '.decode(' in line:
            # Only safe if decoding a known type, not user data
            if '.self' in line:  # Type.self pattern
                return True

        # PropertyListDecoder/Encoder
        if 'PropertyListDecoder' in line or 'PropertyListEncoder' in line:
            return True

        # Static let with try! is often intentional for constants
        if line.strip().startswith('static let') and 'try!' in line:
            return True

        # FileManager operations that always succeed
        if 'FileManager' in line and any(op in line for op in [
            'urls(for:', 'containerURL', 'documentDirectory'
        ]):
            return True

        # Bundle.main operations
        if 'Bundle.main' in line and any(op in line for op in [
            'path(forResource:', 'url(forResource:', 'bundleIdentifier'
        ]):
            return True

        return False

    def _is_safe_force_unwrap(self, line: str, match: "re.Match") -> bool:
        """
        Check if force unwrap is a common safe pattern.

        Safe patterns:
        - documentsDirectory.first! (always exists on iOS)
        - Bundle.main.bundleIdentifier! (always exists)
        - UIApplication.shared access
        - Storyboard instantiation with known identifiers
        """
        matched_text = match.group(0) if hasattr(match, 'group') else ""

        # Documents directory always exists
        if 'documentsDirectory' in line and '.first!' in line:
            return True  # Safe but could be flagged as MEDIUM confidence instead

        # Bundle identifier always exists
        if 'bundleIdentifier!' in line:
            return True

        # UIApplication.shared is always available
        if 'UIApplication.shared' in line:
            return True

        # URL(string:)! with hardcoded string literal is common (though not great)
        if 'URL(string:' in line and ('http' in line.lower() or '"/' in line):
            return True  # Often intentional for known URLs

        # Storyboard with hardcoded identifier
        if 'instantiateViewController' in line and 'withIdentifier' in line:
            return True

        # NSCoder required init patterns
        if 'required init' in line and 'coder' in line.lower():
            return True

        # fatalError is intentional crash
        if 'fatalError' in line:
            return True

        return False

    def _get_context_lines(self, filepath: str, line_num: int, before: int = 3, after: int = 3) -> list[str]:
        """Get surrounding lines for context."""
        lines = self._get_file_lines(filepath)
        if not lines:
            return []

        start = max(0, line_num - before - 1)
        end = min(len(lines), line_num + after)
        return lines[start:end]

    # ===== SECURITY TOOLS =====

    def find_secrets(self) -> ToolResult:
        """
        Find hardcoded secrets: API keys, passwords, tokens.

        Searches for:
        - API key patterns (sk-, api_key, apikey)
        - Password assignments
        - Token/secret assignments
        - Private keys
        """
        result = ToolResult(tool_name="Hardcoded Secrets Scanner", files_scanned=len(self.files))

        patterns = [
            # API keys with prefixes
            (r'''['"](sk-[a-zA-Z0-9]{20,})['"]''', "API Key (sk- prefix)", Severity.CRITICAL),
            (r'''['"](api[_-]?key[_-]?[a-zA-Z0-9]{10,})['"]''', "API Key pattern", Severity.HIGH),

            # Assignments
            (r'''(?:api[_-]?key|apikey)\s*[=:]\s*['"]((?!<|{|\$)[^'"]{8,})['"]\s*''', "API Key assignment", Severity.HIGH),
            (r'''(?:password|passwd|pwd)\s*[=:]\s*['"]((?!<|{|\$)[^'"]{4,})['"]\s*''', "Password assignment", Severity.CRITICAL),
            (r'''(?:secret|token)\s*[=:]\s*['"]((?!<|{|\$)[^'"]{8,})['"]\s*''', "Secret/Token assignment", Severity.HIGH),

            # AWS patterns
            (r'''AKIA[0-9A-Z]{16}''', "AWS Access Key ID", Severity.CRITICAL),

            # Private keys
            (r'''-----BEGIN (?:RSA |EC )?PRIVATE KEY-----''', "Private Key", Severity.CRITICAL),

            # Bearer tokens
            (r'''['"](Bearer\s+[a-zA-Z0-9\-_.]+)['"]\s*''', "Bearer Token", Severity.HIGH),
        ]

        for pattern, issue, severity in patterns:
            matches = self._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                # Skip non-code files (markdown, docs, etc.)
                if not self._is_code_file(filepath):
                    continue

                # Skip if in comment
                if self._is_in_comment(line, filepath):
                    continue

                # Determine confidence based on file type
                if self._is_test_file(filepath):
                    confidence = Confidence.LOW
                else:
                    confidence = Confidence.HIGH

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=issue,
                    confidence=confidence,
                    severity=severity,
                    fix="Use environment variables or secrets manager",
                    category="secrets",
                ))

        result.summary = f"Found {len(result.findings)} potential hardcoded secrets"
        return result

    def find_sql_injection(self) -> ToolResult:
        """
        Find SQL injection vulnerabilities.

        Searches for:
        - String concatenation in queries
        - f-strings in SQL
        - .format() in SQL
        - % formatting in SQL
        """
        result = ToolResult(tool_name="SQL Injection Scanner", files_scanned=len(self.files))

        patterns = [
            # f-string in SQL
            (r'''f['"](SELECT|INSERT|UPDATE|DELETE|DROP).*\{''', "f-string in SQL query", Severity.CRITICAL),

            # String concatenation
            (r'''(SELECT|INSERT|UPDATE|DELETE).*\+\s*[a-zA-Z_]''', "String concat in SQL", Severity.HIGH),
            (r'''[a-zA-Z_]\s*\+.*(SELECT|INSERT|UPDATE|DELETE)''', "String concat in SQL", Severity.HIGH),

            # .format() in SQL
            (r'''['"](SELECT|INSERT|UPDATE|DELETE).*['"]\.format\(''', ".format() in SQL query", Severity.HIGH),

            # % formatting
            (r'''['"](SELECT|INSERT|UPDATE|DELETE).*%s.*['"].*%''', "% formatting in SQL query", Severity.HIGH),

            # execute with string concat
            (r'''\.execute\([^)]*\+''', "execute() with concatenation", Severity.HIGH),
            (r'''\.execute\(f['"']''', "execute() with f-string", Severity.CRITICAL),
        ]

        # Only scan actual code files (Python, JS, etc.)
        code_extensions = {'.py', '.js', '.ts', '.php', '.rb', '.java', '.go', '.rs'}

        for pattern, issue, severity in patterns:
            matches = self._search_pattern(pattern, exclude_pattern=r'#|//|/\*|\*')
            for filepath, line_num, line, match in matches:
                # Skip non-code files
                if not self._is_code_file(filepath):
                    continue

                # Only check files that could have SQL
                ext = '.' + filepath.rsplit('.', 1)[-1].lower() if '.' in filepath else ''
                if ext not in code_extensions:
                    continue

                if self._is_in_comment(line, filepath):
                    continue

                confidence = Confidence.LOW if self._is_test_file(filepath) else Confidence.HIGH

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=issue,
                    confidence=confidence,
                    severity=severity,
                    fix="Use parameterized queries",
                    category="sql_injection",
                ))

        result.summary = f"Found {len(result.findings)} potential SQL injection points"
        return result

    def find_command_injection(self) -> ToolResult:
        """
        Find command injection vulnerabilities.

        Searches for:
        - os.system() with variables
        - subprocess with shell=True
        - eval/exec with external input
        """
        result = ToolResult(tool_name="Command Injection Scanner", files_scanned=len(self.files))

        patterns = [
            # os.system with variable
            (r'''os\.system\([^)]*[a-zA-Z_][a-zA-Z0-9_]*''', "os.system() with variable", Severity.CRITICAL),
            (r'''os\.system\(f['"']''', "os.system() with f-string", Severity.CRITICAL),

            # subprocess with shell=True
            (r'''subprocess\.\w+\([^)]*shell\s*=\s*True''', "subprocess with shell=True", Severity.HIGH),

            # os.popen
            (r'''os\.popen\([^)]*[a-zA-Z_]''', "os.popen() with variable", Severity.HIGH),

            # eval/exec
            (r'''eval\([^)]*[a-zA-Z_][a-zA-Z0-9_]*''', "eval() with variable", Severity.CRITICAL),
            (r'''exec\([^)]*[a-zA-Z_][a-zA-Z0-9_]*''', "exec() with variable", Severity.CRITICAL),
        ]

        for pattern, issue, severity in patterns:
            matches = self._search_pattern(pattern, file_filter=".py")
            for filepath, line_num, line, match in matches:
                if self._is_in_comment(line, filepath):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=issue,
                    confidence=Confidence.HIGH,
                    severity=severity,
                    fix="Use subprocess with shell=False and list args",
                    category="command_injection",
                ))

        result.summary = f"Found {len(result.findings)} potential command injection points"
        return result

    # ===== iOS/SWIFT TOOLS =====

    def find_force_unwraps(self) -> ToolResult:
        """
        Find force unwraps in Swift code.

        Searches for:
        - variable! (excluding !=)
        - try! (excluding safe patterns like regex compilation)
        - as!

        Filters out safe patterns:
        - NSRegularExpression/Regex with compile-time constant patterns
        - Static let with try! (intentional constants)
        - JSONDecoder/Encoder with type literals
        """
        result = ToolResult(tool_name="Force Unwrap Scanner", files_scanned=len(self.files))

        patterns = [
            # variable! anywhere (standalone, property access, array index, etc) but not !=
            # Match: optionalValue!, self.property!, array![0], etc.
            # Skip: x != y, result != nil
            (r'''[a-zA-Z_][a-zA-Z0-9_]*!(?!=)''', "Force unwrap", Severity.MEDIUM, False),

            # try! - needs safe pattern check
            (r'''\btry!\s+''', "Force try (try!)", Severity.HIGH, True),

            # as!
            (r'''\bas!\s+''', "Force cast (as!)", Severity.MEDIUM, False),

            # Implicitly unwrapped optional declaration (: Type!)
            (r''':\s*[A-Z][a-zA-Z0-9_<>,\s]*!(?!=)\s*[=\n{]''', "Implicitly unwrapped optional", Severity.LOW, False),
        ]

        for pattern, issue, severity, check_safe in patterns:
            matches = self._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self._is_in_comment(line, filepath):
                    continue

                # Skip matches inside string literals ("Great news!", etc.)
                match_pos = match.start() if hasattr(match, 'start') else line.find('!')
                if self._is_in_string(line, match_pos):
                    continue

                # For try!, check if it's a safe pattern (regex, static constants, etc.)
                if check_safe and self._is_swift_safe_try(line):
                    continue

                # Skip common safe force unwrap patterns
                if self._is_safe_force_unwrap(line, match):
                    continue

                # Check for #if false blocks
                confidence = Confidence.HIGH
                file_start = self.file_index.get(filepath, (0, 0))[0]
                if '#if false' in self.content[:file_start]:
                    confidence = Confidence.LOW

                # Lower confidence for safe-looking patterns even if not filtered
                if 'static let' in line or 'static var' in line:
                    confidence = Confidence.LOW
                    issue = f"{issue} (static constant - likely intentional)"

                # Lower confidence for common safe patterns that we didn't filter
                if any(safe in line for safe in ['.first!', '.last!', 'Bundle.main', 'FileManager.default']):
                    confidence = Confidence.MEDIUM

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=issue,
                    confidence=confidence,
                    severity=severity,
                    fix="Use if-let, guard-let, or nil coalescing (unless compile-time constant)",
                    category="force_unwrap",
                ))

        result.summary = f"Found {len(result.findings)} force unwraps"
        return result

    def find_retain_cycles(self) -> ToolResult:
        """
        Find potential retain cycles in Swift.

        Searches for:
        - Closures with self without [weak self]
        - Delegates not marked weak
        """
        result = ToolResult(tool_name="Retain Cycle Scanner", files_scanned=len(self.files))

        patterns = [
            # Closure with self. but no [weak self]
            (r'''\{\s*(?!\[(?:weak|unowned)\s+self\]).*\bself\.''', "Closure captures self strongly", Severity.MEDIUM),

            # Delegate property without weak
            (r'''var\s+\w*[Dd]elegate\w*\s*:\s*(?!weak)''', "Delegate not marked weak", Severity.HIGH),

            # @escaping closure with self
            (r'''@escaping.*\{[^}]*\bself\b''', "@escaping closure captures self", Severity.MEDIUM),
        ]

        for pattern, issue, severity in patterns:
            matches = self._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self._is_in_comment(line, filepath):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=issue,
                    confidence=Confidence.MEDIUM,
                    severity=severity,
                    fix="Add [weak self] or [unowned self] to closure",
                    category="retain_cycle",
                ))

        result.summary = f"Found {len(result.findings)} potential retain cycles"
        return result

    def find_main_thread_violations(self) -> ToolResult:
        """
        Find potential main thread violations in Swift.

        Searches for:
        - UI updates without @MainActor
        - Missing DispatchQueue.main
        """
        result = ToolResult(tool_name="Main Thread Violation Scanner", files_scanned=len(self.files))

        patterns = [
            # Task with UI property updates
            (r'''Task\s*\{[^}]*\.(text|isHidden|alpha|frame|bounds)\s*=''', "UI update in Task without @MainActor", Severity.HIGH),

            # DispatchQueue.global with UI
            (r'''DispatchQueue\.global.*\{[^}]*\.(text|isHidden|alpha)\s*=''', "UI update on background queue", Severity.CRITICAL),

            # async function modifying UI
            (r'''func\s+\w+\s*\([^)]*\)\s*async[^{]*\{[^}]*\.(text|isHidden)\s*=''', "UI update in async function", Severity.MEDIUM),
        ]

        for pattern, issue, severity in patterns:
            matches = self._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self._is_in_comment(line, filepath):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=issue,
                    confidence=Confidence.MEDIUM,
                    severity=severity,
                    fix="Use @MainActor or DispatchQueue.main.async",
                    category="thread_safety",
                ))

        result.summary = f"Found {len(result.findings)} potential main thread violations"
        return result

    def find_cloudkit_issues(self) -> ToolResult:
        """
        Find CloudKit error handling issues.

        Searches for:
        - Fire-and-forget CloudKit operations (no await, no completion)
        - Missing CKError-specific handling in catch blocks

        NOTE: Does NOT flag:
        - try await calls (async/await with do/catch is valid)
        - Operations with completion handlers
        - Operations inside do/catch blocks
        """
        result = ToolResult(tool_name="CloudKit Error Handling Scanner", files_scanned=len(self.files))

        for filepath, (start, end) in self.file_index.items():
            if not filepath.endswith('.swift'):
                continue

            file_content = self.content[start:end]

            # Only check files that use CloudKit
            if 'CloudKit' not in file_content and 'CKRecord' not in file_content:
                continue

            lines = file_content.split('\n')

            # Track if we're inside a do block (for context)
            in_do_block = False
            do_block_depth = 0

            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()

                # Track do/catch blocks
                if stripped.startswith('do {') or stripped == 'do {':
                    in_do_block = True
                    do_block_depth = 1
                elif in_do_block:
                    do_block_depth += line.count('{') - line.count('}')
                    if do_block_depth <= 0:
                        in_do_block = False

                # Skip comments
                if stripped.startswith('//') or stripped.startswith('/*'):
                    continue

                # Skip lines with proper error handling
                if any(eh in line for eh in ['try await', 'try?', 'try!', 'completionHandler', 'completionBlock', 'completion:']):
                    continue

                # Check for fire-and-forget patterns
                if '.save(' in line or '.fetch(' in line or '.delete(' in line:
                    if 'database' in line.lower() or 'container' in line.lower():
                        # Only flag if NOT in do block and no await/completion
                        if not in_do_block and 'await' not in line:
                            result.findings.append(Finding(
                                file=filepath,
                                line=line_num,
                                code=line[:200],
                                issue="CloudKit operation may lack error handling",
                                confidence=Confidence.LOW,  # Low confidence - needs manual verification
                                severity=Severity.MEDIUM,
                                fix="Verify error handling via completion handler or do/catch",
                                category="cloudkit",
                            ))

                # Check for catch blocks that only print
                if 'catch' in stripped and 'print(' in stripped:
                    # Look for CKError handling in surrounding context
                    context = '\n'.join(lines[max(0, line_num-5):min(len(lines), line_num+5)])
                    if 'CKError' not in context and 'CloudKit' in file_content:
                        result.findings.append(Finding(
                            file=filepath,
                            line=line_num,
                            code=line[:200],
                            issue="Catch block may need CKError-specific handling",
                            confidence=Confidence.LOW,
                            severity=Severity.LOW,
                            fix="Consider handling specific CKError cases (serverRecordChanged, zoneBusy, etc.)",
                            category="cloudkit",
                        ))

        result.summary = f"Found {len(result.findings)} potential CloudKit issues (verify manually)"
        return result

    def find_deprecated_apis(self, min_severity: str = "LOW") -> ToolResult:
        """
        Find deprecated iOS API usage.

        Args:
            min_severity: Minimum severity to report. Options: "LOW", "MEDIUM", "HIGH", "CRITICAL"
                         Use "MEDIUM" to skip cosmetic deprecations like foregroundColor.

        Searches for:
        - Deprecated UIKit APIs
        - Deprecated SwiftUI APIs
        - Deprecated Foundation APIs
        """
        # Map string to Severity enum for filtering
        severity_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3, "INFO": -1}
        min_sev_value = severity_order.get(min_severity.upper(), 0)

        result = ToolResult(tool_name=f"Deprecated API Scanner (>={min_severity})", files_scanned=len(self.files))

        deprecated_patterns = [
            # UIKit deprecated - HIGH priority
            (r'''\bUIWebView\b''', "UIWebView is deprecated, use WKWebView", Severity.HIGH),
            (r'''DispatchQueue\.main\.sync\s*\{''', "Potential deadlock with sync on main", Severity.HIGH),

            # UIKit deprecated - MEDIUM priority (should fix eventually)
            (r'''UIApplication\.shared\.keyWindow''', "keyWindow is deprecated in iOS 13+, use window scene", Severity.MEDIUM),
            (r'''UIAlertView\(''', "UIAlertView is deprecated, use UIAlertController", Severity.MEDIUM),
            (r'''UIActionSheet\(''', "UIActionSheet is deprecated, use UIAlertController", Severity.MEDIUM),
            (r'''@ObservedObject\s+var\s+\w+\s*=\s*\w+\(''', "@ObservedObject with default value - use @StateObject", Severity.MEDIUM),

            # Cosmetic deprecations - LOW priority (still works, fix when convenient)
            (r'''\.statusBarStyle\s*=''', "Setting statusBarStyle directly is deprecated", Severity.LOW),
            (r'''\.beginAnimations\(''', "beginAnimations is deprecated, use UIView.animate", Severity.LOW),
            (r'''\.characters\.count''', ".characters is deprecated, use .count directly", Severity.LOW),
            (r'''\.foregroundColor\(''', "foregroundColor deprecated iOS 17+, use foregroundStyle (cosmetic)", Severity.LOW),
            (r'''\.accentColor\(''', "accentColor is deprecated, use tint (cosmetic)", Severity.LOW),
        ]

        for pattern, issue, severity in deprecated_patterns:
            # Skip if below minimum severity threshold
            if severity_order.get(severity.value, 0) < min_sev_value:
                continue

            matches = self._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self._is_in_comment(line, filepath):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=issue,
                    confidence=Confidence.HIGH,
                    severity=severity,
                    fix="Update to modern API equivalent",
                    category="deprecated_api",
                ))

        result.summary = f"Found {len(result.findings)} deprecated API usages (>={min_severity})"
        return result

    def find_swiftdata_issues(self) -> ToolResult:
        """
        Find SwiftData race condition and threading issues.

        Searches for:
        - ModelContext accessed from wrong actor
        - Missing @MainActor on SwiftData views
        - Concurrent context access
        """
        result = ToolResult(tool_name="SwiftData Race Condition Scanner", files_scanned=len(self.files))

        patterns = [
            # ModelContext without actor isolation
            (r'''let\s+context\s*=\s*ModelContext''', "ModelContext creation - ensure actor isolation", Severity.MEDIUM),

            # Accessing modelContext in background
            (r'''Task\s*\{[^}]*modelContext''', "modelContext in Task - may need @MainActor", Severity.HIGH),
            (r'''DispatchQueue\.global[^}]*modelContext''', "modelContext on background queue", Severity.CRITICAL),

            # Missing @MainActor on views with @Query
            (r'''@Query[^@]*struct\s+\w+:\s*View''', "@Query view - verify @MainActor", Severity.MEDIUM),

            # ModelActor without proper isolation
            (r'''@ModelActor[^{]*\{[^}]*(?!isolated)''', "@ModelActor - check isolated access", Severity.MEDIUM),

            # Passing model objects across actors
            (r'''await\s+\w+\.\w+\([^)]*@Model''', "Passing @Model object across actors", Severity.HIGH),

            # Background save without proper context
            (r'''\.save\(\)[^}]*Task\s*\{''', "save() before background Task - race condition risk", Severity.HIGH),
        ]

        for pattern, issue, severity in patterns:
            matches = self._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                # Only flag if file uses SwiftData
                file_content = self._get_file_content(filepath) or ""
                if 'SwiftData' not in file_content and '@Model' not in file_content and 'ModelContext' not in file_content:
                    continue

                if self._is_in_comment(line, filepath):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=issue,
                    confidence=Confidence.MEDIUM,
                    severity=severity,
                    fix="Use @ModelActor for background work, ensure @MainActor for UI contexts",
                    category="swiftdata",
                ))

        result.summary = f"Found {len(result.findings)} SwiftData threading issues"
        return result

    def find_weak_self_issues(self) -> ToolResult:
        """
        Find missing [weak self] in closures (multi-line aware).

        Uses a two-pass approach:
        1. Find closure openings (.sink {, Timer..., etc.)
        2. Check if closure has self. usage without [weak self] capture

        Searches for:
        - Combine sinks without [weak self]
        - Timer closures without [weak self]
        - NotificationCenter observers
        - Escaping closures
        """
        result = ToolResult(tool_name="Weak Self Scanner", files_scanned=len(self.files))

        # Closure-opening patterns (we'll check for self. in following lines)
        closure_starters = [
            (r'''\.sink\s*\{''', "Combine sink", Severity.MEDIUM),
            (r'''\.sink\s*\(\s*receiveValue:\s*\{''', "Combine sink receiveValue", Severity.MEDIUM),
            (r'''Timer\.scheduledTimer[^{]*\{''', "Timer closure", Severity.HIGH),
            (r'''Timer\.publish[^{]*\.sink\s*\{''', "Timer publisher sink", Severity.HIGH),
            (r'''NotificationCenter[^{]*addObserver[^{]*\{''', "NotificationCenter observer", Severity.HIGH),
            (r'''\.publisher\(for:[^{]*\.sink\s*\{''', "NotificationCenter publisher sink", Severity.MEDIUM),
            (r'''DispatchQueue\.[^{]*\.async[^{]*\{''', "DispatchQueue async", Severity.LOW),
            (r'''\.asyncAfter[^{]*\{''', "asyncAfter", Severity.MEDIUM),
            (r'''URLSession[^{]*\{''', "URLSession completion", Severity.MEDIUM),
        ]

        for filepath in self.files:
            if not filepath.endswith('.swift'):
                continue

            lines = self._get_file_lines(filepath)
            if not lines:
                continue

            file_content = self._get_file_content(filepath) or ""

            # Skip struct-only files (no retain cycles possible)
            if 'class ' not in file_content and 'actor ' not in file_content:
                continue

            for pattern, issue_type, severity in closure_starters:
                for line_num, line in enumerate(lines, 1):
                    if not re.search(pattern, line, re.IGNORECASE):
                        continue

                    # Check if this line already has [weak self]
                    if '[weak self]' in line or '[unowned self]' in line:
                        continue

                    # Look at next 10 lines for self. usage
                    has_self_usage = False
                    for i in range(line_num, min(line_num + 10, len(lines))):
                        check_line = lines[i]
                        if re.search(r'\bself\.', check_line):
                            has_self_usage = True
                            break
                        # Stop at closing brace that ends the closure
                        if check_line.strip() == '}':
                            break

                    if has_self_usage:
                        result.findings.append(Finding(
                            file=filepath,
                            line=line_num,
                            code=line.strip()[:200],
                            issue=f"{issue_type} captures self without [weak self]",
                            confidence=Confidence.HIGH,
                            severity=severity,
                            fix="Add [weak self] and use guard let self else { return }",
                            category="weak_self",
                        ))

        result.summary = f"Found {len(result.findings)} potential [weak self] issues"
        return result

    # ===== PYTHON TOOLS =====

    def find_python_security(self) -> ToolResult:
        """
        Find Python-specific security issues.

        Searches for:
        - pickle.loads with untrusted data
        - yaml.load without SafeLoader
        - Mutable default arguments
        """
        result = ToolResult(tool_name="Python Security Scanner", files_scanned=len(self.files))

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
        ]

        for pattern, issue, severity in patterns:
            matches = self._search_pattern(pattern, file_filter=".py")
            for filepath, line_num, line, match in matches:
                if self._is_in_comment(line, filepath):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=issue,
                    confidence=Confidence.HIGH,
                    severity=severity,
                    fix="Use safe alternatives (SafeLoader, immutable defaults)",
                    category="python_security",
                ))

        result.summary = f"Found {len(result.findings)} Python security issues"
        return result

    # ===== JAVASCRIPT TOOLS =====

    def find_xss_vulnerabilities(self) -> ToolResult:
        """
        Find XSS vulnerabilities in JavaScript/TypeScript.

        Searches for:
        - innerHTML with variables
        - document.write
        - dangerouslySetInnerHTML
        """
        result = ToolResult(tool_name="XSS Vulnerability Scanner", files_scanned=len(self.files))

        patterns = [
            # innerHTML
            (r'''\.innerHTML\s*=\s*[^'"<]''', "innerHTML with variable", Severity.HIGH),
            (r'''\.innerHTML\s*=\s*`''', "innerHTML with template literal", Severity.HIGH),

            # document.write
            (r'''document\.write\(''', "document.write() usage", Severity.MEDIUM),

            # React dangerouslySetInnerHTML
            (r'''dangerouslySetInnerHTML\s*=''', "dangerouslySetInnerHTML usage", Severity.MEDIUM),

            # outerHTML
            (r'''\.outerHTML\s*=''', "outerHTML with variable", Severity.HIGH),
        ]

        for pattern, issue, severity in patterns:
            matches = self._search_pattern(pattern, file_filter=None)  # .js, .ts, .jsx, .tsx
            for filepath, line_num, line, match in matches:
                if not any(filepath.endswith(ext) for ext in ['.js', '.ts', '.jsx', '.tsx', '.vue']):
                    continue
                if self._is_in_comment(line, filepath):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=issue,
                    confidence=Confidence.HIGH,
                    severity=severity,
                    fix="Use textContent or proper sanitization",
                    category="xss",
                ))

        result.summary = f"Found {len(result.findings)} potential XSS vulnerabilities"
        return result

    # ===== ARCHITECTURE TOOLS =====

    def map_architecture(self) -> ToolResult:
        """
        Map the codebase architecture.

        Returns:
        - File categories
        - Entry points
        - Module structure
        """
        result = ToolResult(tool_name="Architecture Mapper", files_scanned=len(self.files))

        categories = {
            "entry_points": [],
            "views_ui": [],
            "models": [],
            "services": [],
            "utilities": [],
            "tests": [],
            "config": [],
        }

        for filepath in self.files:
            lower = filepath.lower()

            # Categorize
            if any(x in lower for x in ['main.', 'index.', 'app.', '__main__', 'cli.']):
                categories["entry_points"].append(filepath)
            elif any(x in lower for x in ['view', 'controller', 'screen', 'component', 'page']):
                categories["views_ui"].append(filepath)
            elif any(x in lower for x in ['model', 'entity', 'schema', 'type']):
                categories["models"].append(filepath)
            elif any(x in lower for x in ['service', 'manager', 'provider', 'handler', 'api']):
                categories["services"].append(filepath)
            elif any(x in lower for x in ['util', 'helper', 'common', 'shared']):
                categories["utilities"].append(filepath)
            elif any(x in lower for x in ['test', 'spec', '_test', '.test']):
                categories["tests"].append(filepath)
            elif any(x in lower for x in ['config', 'setting', '.json', '.yaml', '.env']):
                categories["config"].append(filepath)

        # Create findings for each category
        for category, files in categories.items():
            if files:
                result.findings.append(Finding(
                    file=category,
                    line=0,
                    code="\n".join(files[:20]),
                    issue=f"{len(files)} files",
                    confidence=Confidence.HIGH,
                    severity=Severity.INFO,
                    category="architecture",
                ))

        result.summary = f"Mapped {len(self.files)} files into {len([c for c in categories.values() if c])} categories"
        return result

    def find_imports(self, module_name: str) -> ToolResult:
        """
        Find all imports of a specific module.

        Args:
            module_name: Name of module to find imports for
        """
        result = ToolResult(tool_name=f"Import Scanner ({module_name})", files_scanned=len(self.files))

        patterns = [
            (rf'''import\s+{re.escape(module_name)}''', "import statement"),
            (rf'''from\s+{re.escape(module_name)}''', "from import"),
            (rf'''require\(['"]{re.escape(module_name)}['"]''', "require()"),
        ]

        for pattern, issue in patterns:
            matches = self._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=issue,
                    confidence=Confidence.HIGH,
                    severity=Severity.INFO,
                    category="imports",
                ))

        result.summary = f"Found {len(result.findings)} imports of {module_name}"
        return result

    def find_todos(self) -> ToolResult:
        """
        Find TODO, FIXME, HACK, XXX comments.
        """
        result = ToolResult(tool_name="TODO Scanner", files_scanned=len(self.files))

        patterns = [
            (r'''#\s*(TODO|FIXME|HACK|XXX)[:.]?\s*(.*)''', Severity.INFO),
            (r'''//\s*(TODO|FIXME|HACK|XXX)[:.]?\s*(.*)''', Severity.INFO),
            (r'''/\*\s*(TODO|FIXME|HACK|XXX)[:.]?\s*(.*)''', Severity.INFO),
        ]

        for pattern, severity in patterns:
            matches = self._search_pattern(pattern)
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

    # ===== QUALITY TOOLS =====

    def find_long_functions(self, max_lines: int = 50) -> ToolResult:
        """
        Find functions that are too long.

        Args:
            max_lines: Maximum acceptable function length
        """
        result = ToolResult(tool_name=f"Long Function Scanner (>{max_lines} lines)", files_scanned=len(self.files))

        # Function start patterns by language
        func_patterns = [
            (r'''^\s*def\s+(\w+)\s*\(''', '.py'),  # Python
            (r'''^\s*(?:async\s+)?func\s+(\w+)\s*\(''', '.swift'),  # Swift
            (r'''^\s*(?:async\s+)?function\s+(\w+)\s*\(''', '.js'),  # JavaScript
            (r'''^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+(\w+)\s*\(''', '.java'),  # Java
        ]

        for filepath in self.files:
            lines = self._get_file_lines(filepath)
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
                # End is next function start or end of file
                end_line = func_starts[i + 1][0] if i + 1 < len(func_starts) else len(lines)
                length = end_line - start_line

                if length > max_lines:
                    result.findings.append(Finding(
                        file=filepath,
                        line=start_line + 1,
                        code=f"def {func_name}(...): # {length} lines",
                        issue=f"Function too long ({length} lines > {max_lines})",
                        confidence=Confidence.HIGH,
                        severity=Severity.MEDIUM if length < 100 else Severity.HIGH,
                        fix=f"Refactor into smaller functions",
                        category="long_function",
                    ))

        result.summary = f"Found {len(result.findings)} functions over {max_lines} lines"
        return result

    # ===== iOS SECURITY SCANNERS =====

    def find_insecure_storage(self) -> ToolResult:
        """
        Find sensitive data stored insecurely (UserDefaults instead of Keychain).

        Searches for:
        - Passwords/tokens/secrets in UserDefaults
        - API keys in UserDefaults
        - Credentials not using Keychain
        """
        result = ToolResult(tool_name="Insecure Storage Scanner", files_scanned=len(self.files))

        # Patterns for sensitive data in UserDefaults
        sensitive_keys = [
            'password', 'passwd', 'token', 'apikey', 'api_key', 'secret',
            'credential', 'auth', 'bearer', 'access_token', 'refresh_token',
            'private_key', 'session', 'jwt', 'oauth'
        ]

        # Build pattern for UserDefaults with sensitive keys
        key_pattern = '|'.join(sensitive_keys)
        patterns = [
            # UserDefaults.standard.set with sensitive key
            (rf'''UserDefaults[^)]*\.set\([^)]*(?i)({key_pattern})''',
             "Sensitive data in UserDefaults - use Keychain", Severity.CRITICAL),

            # UserDefaults storing with sensitive key name
            (rf'''UserDefaults[^)]*forKey:\s*["'](?i).*({key_pattern})''',
             "Sensitive key in UserDefaults - use Keychain", Severity.CRITICAL),

            # @AppStorage with sensitive key
            (rf'''@AppStorage\s*\(\s*["'](?i).*({key_pattern})''',
             "@AppStorage with sensitive data - use Keychain", Severity.CRITICAL),

            # Storing in plist/file (not encrypted)
            (rf'''\.write\([^)]*(?i)({key_pattern})[^)]*toFile:''',
             "Sensitive data written to file - encrypt or use Keychain", Severity.HIGH),
        ]

        for pattern, issue, severity in patterns:
            matches = self._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self._is_in_comment(line, filepath):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=issue,
                    confidence=Confidence.HIGH,
                    severity=severity,
                    fix="Use Keychain Services or Security framework for sensitive data",
                    category="insecure_storage",
                ))

        result.summary = f"Found {len(result.findings)} insecure storage patterns"
        return result

    def find_input_sanitization_issues(self) -> ToolResult:
        """
        Find user input that may not be properly sanitized.

        Searches for:
        - Text field input used directly in operations
        - URL construction from user input
        - String interpolation with user input in sensitive contexts
        """
        result = ToolResult(tool_name="Input Sanitization Scanner", files_scanned=len(self.files))

        patterns = [
            # User input directly in URL construction
            (r'''URL\(string:\s*(?:text|input|query|search|user)''',
             "User input in URL - sanitize and validate", Severity.MEDIUM),

            # TextField bound to variable used elsewhere without validation
            (r'''TextField[^)]*\$(\w+)[^}]*\}[^}]*\1[^}]*(?:URL|request|query)''',
             "TextField input may reach sensitive operation", Severity.MEDIUM),

            # String interpolation in shell/command context
            (r'''Process\(\)[^}]*arguments.*\\?\(\w+\)''',
             "Variable in Process arguments - validate input", Severity.HIGH),

            # SQL-like string construction (Core Data predicates)
            (r'''NSPredicate\(format:\s*["'][^"']*%@[^"']*["']\s*,\s*(?:text|input|search)''',
             "User input in NSPredicate - use format arguments properly", Severity.MEDIUM),

            # Webview loading user-provided URL
            (r'''(?:loadRequest|load)\([^)]*URL[^)]*(?:text|input|user)''',
             "User input in WebView URL - validate and sanitize", Severity.HIGH),

            # JavaScript evaluation with user input
            (r'''evaluateJavaScript\([^)]*(?:text|input|user|\\\()''',
             "User input in JavaScript evaluation - XSS risk", Severity.CRITICAL),
        ]

        for pattern, issue, severity in patterns:
            matches = self._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self._is_in_comment(line, filepath):
                    continue
                if self._is_test_file(filepath):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=issue,
                    confidence=Confidence.MEDIUM,
                    severity=severity,
                    fix="Validate and sanitize user input before use",
                    category="input_sanitization",
                ))

        result.summary = f"Found {len(result.findings)} potential input sanitization issues"
        return result

    def find_missing_jailbreak_detection(self) -> ToolResult:
        """
        Check if app has jailbreak detection (important for finance apps).

        Note: This scanner checks for PRESENCE of jailbreak detection,
        not absence. Returns findings if detection is missing.
        """
        result = ToolResult(tool_name="Jailbreak Detection Scanner", files_scanned=len(self.files))

        # Look for common jailbreak detection patterns
        jailbreak_patterns = [
            r'cydia://',
            r'/Applications/Cydia\.app',
            r'/Library/MobileSubstrate',
            r'/bin/bash',
            r'/usr/sbin/sshd',
            r'canOpenURL.*cydia',
            r'jailbroken',
            r'jailbreak',
            r'isJailbroken',
        ]

        has_jailbreak_detection = False
        detection_locations = []

        for pattern in jailbreak_patterns:
            matches = self._search_pattern(pattern, file_filter=".swift")
            if matches:
                has_jailbreak_detection = True
                for filepath, line_num, line, match in matches[:3]:  # Just note a few
                    detection_locations.append(f"{filepath}:{line_num}")

        if not has_jailbreak_detection:
            # Check if this looks like a finance/banking app
            finance_indicators = ['payment', 'banking', 'finance', 'wallet', 'transaction', 'money']
            is_finance_app = False

            for indicator in finance_indicators:
                if self._search_pattern(indicator):
                    is_finance_app = True
                    break

            if is_finance_app:
                result.findings.append(Finding(
                    file="(project-wide)",
                    line=0,
                    code="No jailbreak detection found",
                    issue="Finance app without jailbreak detection",
                    confidence=Confidence.MEDIUM,
                    severity=Severity.MEDIUM,
                    fix="Consider adding jailbreak detection for sensitive operations",
                    category="security",
                ))

        result.summary = f"Jailbreak detection: {'Present' if has_jailbreak_detection else 'Not found'}"
        return result

    # ===== RUN ALL SECURITY =====

    def run_security_scan(self) -> list[ToolResult]:
        """Run all security-related scans."""
        return [
            self.find_secrets(),
            self.find_sql_injection(),
            self.find_command_injection(),
            self.find_xss_vulnerabilities(),
            self.find_python_security(),
            self.find_insecure_storage(),
            self.find_input_sanitization_issues(),
        ]

    # ===== RUN ALL QUALITY =====

    def run_quality_scan(self) -> list[ToolResult]:
        """Run all quality-related scans."""
        return [
            self.find_long_functions(),
            self.find_todos(),
        ]

    # ===== RUN ALL iOS =====

    def run_ios_scan(self) -> list[ToolResult]:
        """Run all iOS/Swift scans including new checks."""
        return [
            self.find_force_unwraps(),
            self.find_retain_cycles(),
            self.find_main_thread_violations(),
            self.find_weak_self_issues(),
            self.find_cloudkit_issues(),
            self.find_deprecated_apis(),
            self.find_swiftdata_issues(),
            self.find_insecure_storage(),
            self.find_input_sanitization_issues(),
            self.find_missing_jailbreak_detection(),
        ]


def create_tools(content: str) -> StructuredTools:
    """Factory function to create tools instance."""
    return StructuredTools(content)
