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


def calculate_confidence(finding: Finding, context: AnalysisContext) -> Confidence:
    """
    L11: Apply standardized confidence criteria.

    Scoring:
    - Start at 100 (HIGH)
    - Deduct for uncertainty factors
    - Boost for verification factors
    - Map final score to confidence level
    """
    score = 100

    # Deductions
    if context.in_dead_code:
        score -= 50  # Dead code = likely false positive

    if context.in_test_file:
        score -= 30  # Test code = intentional patterns

    if not context.line_verified:
        score -= 20  # Can't verify exact line

    if context.is_comment:
        score -= 40  # Commented code

    if context.pattern_match_only:
        score -= 10  # No semantic verification

    # Boosts
    if context.has_semantic_verification:
        score += 20  # LLM verified this finding

    if context.multiple_indicators:
        score += 15  # Multiple patterns match

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

    def _get_file_line_count(self, filepath: str) -> int:
        """Get the total number of lines in a file."""
        lines = self._get_file_lines(filepath)
        return len(lines) if lines else 0

    def _validate_line_number(self, filepath: str, line_num: int) -> tuple[bool, int]:
        """
        Validate that a line number is within file bounds.

        Args:
            filepath: Path to the file
            line_num: Line number to validate (1-indexed)

        Returns:
            (is_valid, actual_max_lines) tuple
            - is_valid: True if line_num is within bounds
            - actual_max_lines: The actual number of lines in the file
        """
        max_lines = self._get_file_line_count(filepath)
        if max_lines == 0:
            return False, 0
        return 1 <= line_num <= max_lines, max_lines

    def _clamp_line_number(self, filepath: str, line_num: int) -> int:
        """
        Clamp a line number to valid range for the file.

        If line_num exceeds file length, returns the last valid line.
        """
        max_lines = self._get_file_line_count(filepath)
        if max_lines == 0:
            return line_num  # Can't validate, return as-is
        return min(max(1, line_num), max_lines)

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

    # =========================================================================
    # PUBLIC FILE ACCESS TOOLS
    # These are safe to expose in the REPL for direct file reading
    # =========================================================================

    def read_file(self, path: str) -> str:
        """
        Read the content of a specific file.

        This is the SAFE way to read files in the REPL sandbox.
        Files must have been collected during the initial scan.

        Args:
            path: File path (can be partial - will match if path ends with this)

        Returns:
            File content as string, or error message if not found

        Example:
            content = read_file("server.py")
            content = read_file("src/utils/helpers.ts")
        """
        content = self._get_file_content(path)
        if content is None:
            # Provide helpful error with suggestions
            similar = [f for f in self.files if path.lower() in f.lower()][:5]
            if similar:
                suggestion = "\n".join(f"  - {f}" for f in similar)
                return f"[ERROR] File not found: '{path}'\n\nDid you mean:\n{suggestion}\n\nUse list_files() to see all available files."
            return f"[ERROR] File not found: '{path}'\n\nUse list_files() to see all {len(self.files)} available files."
        return content

    def list_files(self, pattern: str | None = None) -> str:
        """
        List all available files that were collected for analysis.

        Args:
            pattern: Optional filter pattern (case-insensitive substring match)

        Returns:
            Formatted list of files with sizes

        Example:
            list_files()              # All files
            list_files(".swift")      # Only Swift files
            list_files("ViewModel")   # Files containing 'ViewModel'
        """
        files_to_show = self.files
        if pattern:
            pattern_lower = pattern.lower()
            files_to_show = [f for f in self.files if pattern_lower in f.lower()]

        if not files_to_show:
            if pattern:
                return f"No files matching '{pattern}'. Total files available: {len(self.files)}"
            return "No files available."

        lines = [f"## Available Files ({len(files_to_show)}" + (f" matching '{pattern}')" if pattern else ")")]
        lines.append("")

        # Group by extension for better readability
        by_ext: dict[str, list[str]] = {}
        for f in files_to_show:
            ext = f.split('.')[-1] if '.' in f else 'other'
            if ext not in by_ext:
                by_ext[ext] = []
            by_ext[ext].append(f)

        for ext in sorted(by_ext.keys()):
            lines.append(f"### .{ext} ({len(by_ext[ext])} files)")
            for f in sorted(by_ext[ext]):
                size = self.file_index[f][1] - self.file_index[f][0]
                lines.append(f"  - {f} ({size:,} chars)")
            lines.append("")

        return "\n".join(lines)

    def search_in_file(self, path: str, pattern: str) -> list[dict]:
        """
        Search for a regex pattern within a specific file.

        Args:
            path: File path to search in
            pattern: Regex pattern to search for

        Returns:
            List of matches with line numbers

        Example:
            matches = search_in_file("server.py", r"def \\w+")
            for m in matches:
                print(f"Line {m['line']}: {m['code']}")
        """
        lines = self._get_file_lines(path)
        if lines is None:
            return [{"error": f"File not found: {path}"}]

        results = []
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    results.append({
                        "file": path,
                        "line": i,
                        "code": line.strip(),
                        "match": regex.search(line).group(0)
                    })
        except re.error as e:
            return [{"error": f"Invalid regex: {e}"}]

        return results

    def get_file_info(self, path: str) -> dict:
        """
        Get metadata about a specific file.

        Args:
            path: File path

        Returns:
            Dict with file info (path, size, line_count, extension)
        """
        content = self._get_file_content(path)
        if content is None:
            return {"error": f"File not found: {path}"}

        lines = content.split('\n')
        return {
            "path": path,
            "size_chars": len(content),
            "line_count": len(lines),
            "extension": path.split('.')[-1] if '.' in path else None,
            "first_line": lines[0].strip() if lines else "",
        }

    def _search_pattern(
        self,
        pattern: str,
        file_filter: str | None = None,
        exclude_pattern: str | None = None,
        validate_lines: bool = True,
    ) -> list[tuple[str, int, str, re.Match]]:
        """
        Search for regex pattern across all files.

        Args:
            pattern: Regex pattern to search for
            file_filter: Optional file extension filter (e.g., ".swift")
            exclude_pattern: Optional pattern to exclude matches
            validate_lines: If True, validate line numbers against file length

        Returns: List of (filepath, line_num, line_content, match)
        """
        results = []

        for filepath, (start, end) in self.file_index.items():
            # Apply file filter
            if file_filter and not filepath.endswith(file_filter):
                continue

            file_content = self.content[start:end]
            lines = file_content.split('\n')
            max_lines = len(lines)

            for line_num, line in enumerate(lines, 1):
                # Skip if matches exclude pattern
                if exclude_pattern and re.search(exclude_pattern, line):
                    continue

                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Validate line number against actual file length
                    if validate_lines and line_num > max_lines:
                        # This shouldn't happen in normal operation, but safeguard
                        continue

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
        """
        Check if match position is inside a string literal.

        Handles:
        - Double-quoted strings: "Hello!"
        - Single-quoted strings: 'Hello!'
        - Swift string interpolation: "Value: \\(x)"
        - Multi-line string indicators: triple quotes
        """
        # Find all string regions in the line
        in_string = False
        string_char = None
        i = 0

        while i < len(line) and i < match_pos:
            char = line[i]

            # Check for escape sequences
            if i > 0 and line[i-1] == '\\':
                i += 1
                continue

            # Toggle string state
            if char in ('"', "'"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None

            i += 1

        return in_string

    def _extract_unwrapped_var(self, line: str, match) -> str | None:
        """Extract the variable name being force unwrapped."""
        # Get the matched text
        if hasattr(match, 'group'):
            matched = match.group(0)
        else:
            return None

        # For patterns like "variable!", extract "variable"
        if matched.endswith('!'):
            var_name = matched[:-1].strip()
            # Handle property access like self.property!
            if '.' in var_name:
                var_name = var_name.split('.')[-1]
            return var_name if var_name.isidentifier() else None

        return None

    def _is_guarded_unwrap(self, filepath: str, line_num: int, var_name: str) -> bool:
        """
        Check if a variable is guarded by if-let/guard-let in preceding lines.

        Looks back up to 15 lines for patterns like:
        - if let varName = ...
        - guard let varName = ...
        - if varName != nil
        - guard varName != nil
        """
        lines = self._get_file_lines(filepath)
        if not lines:
            return False

        # Look back up to 15 lines
        start_line = max(0, line_num - 15)

        guard_patterns = [
            rf'\bif\s+let\s+{re.escape(var_name)}\s*=',
            rf'\bguard\s+let\s+{re.escape(var_name)}\s*=',
            rf'\bif\s+{re.escape(var_name)}\s*!=\s*nil',
            rf'\bguard\s+{re.escape(var_name)}\s*!=\s*nil',
            rf'\bif\s+let\s+\w+\s*=\s*{re.escape(var_name)}\b',  # if let x = varName
            rf'\bguard\s+let\s+\w+\s*=\s*{re.escape(var_name)}\b',  # guard let x = varName
        ]

        for i in range(start_line, line_num - 1):
            if i < len(lines):
                check_line = lines[i]
                for pattern in guard_patterns:
                    if re.search(pattern, check_line):
                        return True

        return False

    def _is_string_literal_pattern(self, line: str) -> bool:
        """
        Check if the line contains a force unwrap inside a string literal.

        Specifically handles SwiftUI and common Swift patterns where ! appears
        in user-facing text, not as a force unwrap operator.

        Patterns detected:
        - Text("Hello!")
        - Label("Warning!", systemImage: ...)
        - print("Error!")
        - String literals with ! inside quotes
        """
        # Common SwiftUI/UIKit text patterns where ! is in string content
        text_patterns = [
            r'Text\s*\(\s*"[^"]*![^"]*"\s*\)',           # Text("Hello!")
            r'Label\s*\(\s*"[^"]*![^"]*"',               # Label("Warning!", ...)
            r'print\s*\(\s*"[^"]*![^"]*"\s*\)',          # print("Error!")
            r'NSLog\s*\(\s*@?"[^"]*![^"]*"',             # NSLog("Error!")
            r'assertionFailure\s*\(\s*"[^"]*![^"]*"',    # assertionFailure("...")
            r'preconditionFailure\s*\(\s*"[^"]*![^"]*"', # preconditionFailure("...")
            r'fatalError\s*\(\s*"[^"]*![^"]*"',          # fatalError("...")
            r'\.alert\s*\(\s*"[^"]*![^"]*"',             # .alert("Warning!", ...)
            r'Button\s*\(\s*"[^"]*![^"]*"',              # Button("Click me!", ...)
            r'NavigationLink\s*\(\s*"[^"]*![^"]*"',      # NavigationLink("Go!", ...)
            r'\.navigationTitle\s*\(\s*"[^"]*![^"]*"',   # .navigationTitle("Hello!")
            r'LocalizedStringKey\s*\(\s*"[^"]*![^"]*"',  # LocalizedStringKey("...")
            r'NSLocalizedString\s*\(\s*"[^"]*![^"]*"',   # NSLocalizedString("...")
            r'String\s*\(\s*localized:\s*"[^"]*![^"]*"', # String(localized: "...")
        ]

        for pattern in text_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True

        # Check if the only ! in the line is inside quotes
        # Find all ! positions
        exclamation_positions = [i for i, c in enumerate(line) if c == '!']

        for pos in exclamation_positions:
            # Skip != operator
            if pos + 1 < len(line) and line[pos + 1] == '=':
                continue

            # Check if this ! is inside a string
            if self._is_in_string(line, pos):
                continue

            # Found a ! outside a string that's not !=
            return False

        # All ! are either inside strings or are !=
        return True if exclamation_positions else False

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

                # Skip lines where ! only appears in string literal contexts (SwiftUI Text, etc.)
                if self._is_string_literal_pattern(line):
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

                # Check if this unwrap is guarded by if-let/guard-let in preceding lines
                var_name = self._extract_unwrapped_var(line, match)
                if var_name and self._is_guarded_unwrap(filepath, line_num, var_name):
                    confidence = Confidence.LOW
                    issue = f"{issue} (appears guarded - verify manually)"

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

    def find_task_cancellation_issues(self) -> ToolResult:
        """
        Find Task { } blocks without proper cancellation handling.

        Searches for:
        - Task { } without Task.isCancelled or Task.checkCancellation()
        - Long-running operations in Task without cancellation checks
        - Task.detached without cancellation handling
        """
        result = ToolResult(tool_name="Task Cancellation Scanner", files_scanned=len(self.files))

        for filepath in self.files:
            if not filepath.endswith('.swift'):
                continue

            lines = self._get_file_lines(filepath)
            if not lines:
                continue

            file_content = self._get_file_content(filepath) or ""

            # Find Task { blocks and check for cancellation handling
            in_task_block = False
            task_start_line = 0
            task_depth = 0
            task_has_cancellation = False
            task_code_lines: list[str] = []

            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()

                # Detect Task { or Task.detached {
                if re.search(r'\bTask\s*(\.\s*detached\s*)?\{', stripped):
                    in_task_block = True
                    task_start_line = line_num
                    task_depth = 1
                    task_has_cancellation = False
                    task_code_lines = [line]
                    continue

                if in_task_block:
                    task_code_lines.append(line)
                    task_depth += line.count('{') - line.count('}')

                    # Check for cancellation handling
                    if any(pattern in line for pattern in [
                        'Task.isCancelled',
                        'Task.checkCancellation',
                        'isCancelled',
                        'checkCancellation',
                        'withTaskCancellationHandler'
                    ]):
                        task_has_cancellation = True

                    # End of Task block
                    if task_depth <= 0:
                        # Check if this Task has long-running operations
                        task_content = '\n'.join(task_code_lines)
                        has_long_operation = any(op in task_content for op in [
                            'await', 'for ', 'while ', 'sleep', 'URLSession',
                            'fetch', 'load', 'download', 'upload'
                        ])

                        if has_long_operation and not task_has_cancellation:
                            result.findings.append(Finding(
                                file=filepath,
                                line=task_start_line,
                                code=task_code_lines[0].strip()[:200],
                                issue="Task with async operation lacks cancellation handling",
                                confidence=Confidence.MEDIUM,
                                severity=Severity.MEDIUM,
                                fix="Add Task.checkCancellation() or check Task.isCancelled in loops",
                                category="task_cancellation",
                            ))

                        in_task_block = False
                        task_code_lines = []

        result.summary = f"Found {len(result.findings)} Task blocks without cancellation handling"
        return result

    def find_mainactor_issues(self) -> ToolResult:
        """
        Find missing @MainActor annotations on UI-related code.

        Searches for:
        - ObservableObject classes without @MainActor
        - @Published properties updated from async contexts
        - ViewModel classes without @MainActor
        """
        result = ToolResult(tool_name="@MainActor Scanner", files_scanned=len(self.files))

        for filepath in self.files:
            if not filepath.endswith('.swift'):
                continue

            lines = self._get_file_lines(filepath)
            if not lines:
                continue

            file_content = self._get_file_content(filepath) or ""

            # Skip if no ObservableObject or ViewModel patterns
            if 'ObservableObject' not in file_content and 'ViewModel' not in file_content.lower():
                continue

            has_main_actor = '@MainActor' in file_content

            # Find class declarations
            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()

                # Check for ObservableObject without @MainActor
                if 'ObservableObject' in line and 'class ' in line:
                    # Look back a few lines for @MainActor
                    context_start = max(0, line_num - 5)
                    context = '\n'.join(lines[context_start:line_num])

                    if '@MainActor' not in context:
                        result.findings.append(Finding(
                            file=filepath,
                            line=line_num,
                            code=stripped[:200],
                            issue="ObservableObject without @MainActor - UI updates may cause issues",
                            confidence=Confidence.MEDIUM,
                            severity=Severity.MEDIUM,
                            fix="Add @MainActor to class or ensure @Published updates are on main thread",
                            category="mainactor",
                        ))

                # Check for ViewModel pattern without @MainActor
                if re.search(r'class\s+\w*ViewModel', line, re.IGNORECASE):
                    context_start = max(0, line_num - 5)
                    context = '\n'.join(lines[context_start:line_num])

                    if '@MainActor' not in context and 'ObservableObject' not in context:
                        result.findings.append(Finding(
                            file=filepath,
                            line=line_num,
                            code=stripped[:200],
                            issue="ViewModel class may need @MainActor for thread safety",
                            confidence=Confidence.LOW,
                            severity=Severity.LOW,
                            fix="Consider adding @MainActor if class updates UI state",
                            category="mainactor",
                        ))

                # Check for @Published in async function without @MainActor class
                if '@Published' in line and not has_main_actor:
                    # Look for async functions that modify this
                    var_match = re.search(r'@Published\s+var\s+(\w+)', line)
                    if var_match:
                        var_name = var_match.group(1)
                        # Search for async function modifying this var
                        for later_line_num, later_line in enumerate(lines[line_num:], line_num + 1):
                            if f'{var_name} =' in later_line or f'{var_name}.' in later_line:
                                # Check if inside async context
                                context_start = max(0, later_line_num - 10)
                                async_context = '\n'.join(lines[context_start:later_line_num])
                                if 'func ' in async_context and 'async' in async_context:
                                    if '@MainActor' not in async_context and 'MainActor.run' not in later_line:
                                        result.findings.append(Finding(
                                            file=filepath,
                                            line=later_line_num,
                                            code=later_line.strip()[:200],
                                            issue=f"@Published var '{var_name}' modified in async context without @MainActor",
                                            confidence=Confidence.HIGH,
                                            severity=Severity.HIGH,
                                            fix="Wrap in MainActor.run { } or add @MainActor to function/class",
                                            category="mainactor",
                                        ))
                                        break  # Only report first occurrence

        result.summary = f"Found {len(result.findings)} potential @MainActor issues"
        return result

    def find_keychain_issues(self) -> ToolResult:
        """
        Find Keychain security issues in iOS/macOS code.

        Searches for:
        - SecItemAdd/Update without accessibility settings
        - SecItemCopyMatching result not checked
        - Biometric auth without fallback
        - Hardcoded Keychain service/account names
        """
        result = ToolResult(tool_name="Keychain Security Scanner", files_scanned=len(self.files))

        patterns = [
            # SecItemAdd without kSecAttrAccessible
            (r'''SecItemAdd\s*\([^)]+\)''',
             "SecItemAdd - verify kSecAttrAccessible is set", Severity.MEDIUM, "keychain_accessibility"),

            # SecItemUpdate without checking result
            (r'''SecItemUpdate\s*\([^)]+\)\s*(?!\s*==|\s*!=|\s*,\s*&)''',
             "SecItemUpdate result not checked", Severity.HIGH, "keychain_error"),

            # SecItemCopyMatching without result check
            (r'''SecItemCopyMatching\s*\([^)]+\)\s*\n\s*(?!if|guard|switch|let\s+status)''',
             "SecItemCopyMatching - verify status is checked", Severity.MEDIUM, "keychain_error"),

            # LAContext without fallback handling
            (r'''LAContext\(\)\.evaluatePolicy[^}]*\{[^}]*(?!fallback|password|deviceOwnerAuthentication)''',
             "Biometric auth may need fallback for devices without biometrics", Severity.LOW, "biometric"),

            # Hardcoded Keychain identifiers (potential issue for obfuscation)
            (r'''kSecAttrService[^,]*["'][a-zA-Z0-9_.]+["']''',
             "Hardcoded Keychain service name - consider obfuscation for sensitive apps", Severity.LOW, "keychain_hardcode"),

            # Storing sensitive data without Keychain (should use Keychain)
            (r'''(?:password|token|secret|apiKey|api_key)\s*=\s*["'][^"']+["']''',
             "Sensitive value may be hardcoded - use Keychain for runtime secrets", Severity.HIGH, "hardcoded_secret"),
        ]

        for pattern, issue, severity, category in patterns:
            matches = self._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self._is_in_comment(line, filepath):
                    continue

                # Check context for kSecAttrAccessible
                confidence = Confidence.MEDIUM
                if "kSecAttrAccessible" in line:
                    continue  # Already has accessibility

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=issue,
                    confidence=confidence,
                    severity=severity,
                    fix="Use Keychain with proper kSecAttrAccessible setting",
                    category=category,
                ))

        result.summary = f"Found {len(result.findings)} Keychain security issues"
        return result

    def find_cloudkit_sync_issues(self) -> ToolResult:
        """
        Find CloudKit sync pattern issues.

        Searches for:
        - CKModifyRecordsOperation without proper error handling
        - Missing server change token persistence
        - Subscription setup without error recovery
        - Operations without QoS settings
        """
        result = ToolResult(tool_name="CloudKit Sync Scanner", files_scanned=len(self.files))

        patterns = [
            # CKModifyRecordsOperation without completion handlers
            (r'''CKModifyRecordsOperation\s*\([^)]*\)(?![^}]*(?:perRecordSaveBlock|modifyRecordsResultBlock|completionBlock))''',
             "CKModifyRecordsOperation may be missing result handlers", Severity.HIGH, "cloudkit_handler"),

            # fetchDatabaseChanges without token persistence
            (r'''fetchDatabaseChanges[^}]*serverChangeToken[^}]*(?!UserDefaults|save|store|persist|NSUbiquitousKeyValueStore)''',
             "Server change token should be persisted", Severity.MEDIUM, "cloudkit_token"),

            # CKOperation without qualityOfService
            (r'''CK\w+Operation\s*\([^)]*\)[^}]*(?!qualityOfService)''',
             "CKOperation without explicit QoS - may affect performance", Severity.LOW, "cloudkit_qos"),

            # Missing CKError handling
            (r'''\.perform\([^)]*\)\s*\{[^}]*(?!CKError|error\s*!=\s*nil|if\s+let\s+error)''',
             "CloudKit operation may be missing error handling", Severity.MEDIUM, "cloudkit_error"),

            # Record conflicts not handled
            (r'''CKModifyRecordsOperation[^}]*savePolicy\s*=\s*\.ifServerRecordUnchanged[^}]*(?!serverRecordChanged|CKError)''',
             "Using ifServerRecordUnchanged without conflict handling", Severity.HIGH, "cloudkit_conflict"),

            # Batch size potentially too large
            (r'''CKModifyRecordsOperation\s*\(\s*recordsToSave:\s*\w+[^)]*\)(?![^}]*\.{0,50}chunked|\.{0,50}batch)''',
             "Large batch operations should be chunked (400 record limit)", Severity.LOW, "cloudkit_batch"),
        ]

        for pattern, issue, severity, category in patterns:
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
                    fix="Implement proper CloudKit error handling and sync patterns",
                    category=category,
                ))

        result.summary = f"Found {len(result.findings)} CloudKit sync issues"
        return result

    def find_stateobject_issues(self) -> ToolResult:
        """
        Find @ObservedObject used where @StateObject should be used.

        In SwiftUI, @StateObject should be used for object creation,
        @ObservedObject for objects passed in from parent.
        """
        result = ToolResult(tool_name="@StateObject Scanner", files_scanned=len(self.files))

        patterns = [
            # @ObservedObject with inline initialization
            (r'@ObservedObject\s+(?:private\s+)?var\s+\w+\s*=\s*\w+\(',
             "@ObservedObject with inline init - use @StateObject", Severity.HIGH),

            # @ObservedObject with default value
            (r'@ObservedObject\s+(?:private\s+)?var\s+\w+:\s*\w+\s*=\s*',
             "@ObservedObject with default value - use @StateObject", Severity.HIGH),
        ]

        for pattern, issue, severity in patterns:
            matches = self._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self._is_in_comment(line, filepath):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line.strip()[:200],
                    issue=issue,
                    confidence=Confidence.HIGH,
                    severity=severity,
                    fix="Change @ObservedObject to @StateObject for owned objects",
                    category="stateobject",
                ))

        result.summary = f"Found {len(result.findings)} @StateObject issues"
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

    # Known sanitizer functions/libraries that make innerHTML safe
    SANITIZER_PATTERNS = [
        r'escapeHtml\s*\(',          # escapeHtml() function
        r'escape\s*\(',              # escape() function
        r'sanitize\s*\(',            # sanitize() function
        r'DOMPurify\.sanitize',      # DOMPurify library
        r'xss\s*\(',                 # xss() sanitizer
        r'htmlEncode\s*\(',          # htmlEncode() function
        r'encodeHTML\s*\(',          # encodeHTML() function
        r'safeHTML\s*\(',            # safeHTML() function
        r'purify\s*\(',              # purify() function
        r'sanitizeHtml\s*\(',        # sanitizeHtml() function
        r'createTextNode',           # Safe DOM method
        r'textContent\s*=',          # Safe property (often used in context)
        r'\.innerText\s*=',          # Safe property
        r'validator\.escape',        # validator.js escape
        r'he\.encode',               # he library encode
        r'entities\.encode',         # entities library
    ]

    def _is_sanitized(self, line: str, filepath: str, line_num: int) -> bool:
        """
        Check if a line uses sanitization that makes innerHTML/etc safe.

        Checks:
        1. Direct sanitizer call on the same line
        2. Sanitizer call in the variable assignment (look back a few lines)
        3. Known safe patterns (textContent assignment followed by innerHTML read)
        """
        # Check current line for sanitizer patterns
        for sanitizer_pattern in self.SANITIZER_PATTERNS:
            if re.search(sanitizer_pattern, line, re.IGNORECASE):
                return True

        # Check surrounding context (look back 5 lines for sanitizer usage)
        lines = self._get_file_lines(filepath)
        if lines:
            start_line = max(0, line_num - 6)
            context = '\n'.join(lines[start_line:line_num])
            for sanitizer_pattern in self.SANITIZER_PATTERNS:
                if re.search(sanitizer_pattern, context, re.IGNORECASE):
                    return True

        return False

    def _get_xss_confidence(self, line: str, filepath: str, line_num: int) -> Confidence:
        """
        Determine confidence level for XSS finding based on context.

        Returns:
        - FILTERED: If sanitization is clearly present
        - LOW: If sanitization might be present (ambiguous)
        - MEDIUM: If no sanitization but context unclear
        - HIGH: Clear vulnerability without sanitization
        """
        # Check for sanitization
        if self._is_sanitized(line, filepath, line_num):
            return Confidence.FILTERED

        # Check if it's a static string (less concerning)
        if re.search(r'\.innerHTML\s*=\s*["\'][^"\']*["\']', line):
            return Confidence.LOW  # Static string, not a variable

        # Check if it's in a test file
        if self._is_test_file(filepath):
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

    # ===== TYPESCRIPT ANALYSIS TOOLS =====

    def analyze_typescript_imports(self, filepath: str | None = None) -> ToolResult:
        """
        Analyze TypeScript/JavaScript imports and exports to build dependency graph.

        Args:
            filepath: Optional specific file to analyze. If None, analyzes all TS/JS files.

        Returns:
            ToolResult with import/export relationships
        """
        result = ToolResult(tool_name="TypeScript Import Analyzer", files_scanned=len(self.files))

        # Import patterns
        import_patterns = [
            # ES6 imports
            (r'''import\s+\{([^}]+)\}\s+from\s+['"]([@\w./\-]+)['"]''', 'named'),
            (r'''import\s+(\w+)\s+from\s+['"]([@\w./\-]+)['"]''', 'default'),
            (r'''import\s+\*\s+as\s+(\w+)\s+from\s+['"]([@\w./\-]+)['"]''', 'namespace'),
            (r'''import\s+['"]([@\w./\-]+)['"]''', 'side_effect'),

            # CommonJS
            (r'''(?:const|let|var)\s+\{([^}]+)\}\s*=\s*require\(['"]([@\w./\-]+)['"]\)''', 'cjs_named'),
            (r'''(?:const|let|var)\s+(\w+)\s*=\s*require\(['"]([@\w./\-]+)['"]\)''', 'cjs_default'),

            # Dynamic imports
            (r'''import\(['"]([@\w./\-]+)['"]\)''', 'dynamic'),
        ]

        # Export patterns
        export_patterns = [
            (r'''export\s+\{([^}]+)\}''', 'named_export'),
            (r'''export\s+default\s+(\w+)''', 'default_export'),
            (r'''export\s+(?:const|let|var|function|class|interface|type)\s+(\w+)''', 'declaration_export'),
            (r'''module\.exports\s*=\s*\{([^}]+)\}''', 'cjs_export'),
            (r'''module\.exports\s*=\s*(\w+)''', 'cjs_default_export'),
        ]

        target_files = [filepath] if filepath else self.files
        import_map = {}  # file -> list of (import_type, imported_items, source)
        export_map = {}  # file -> list of (export_type, exported_items)

        for fp in target_files:
            if not any(fp.endswith(ext) for ext in ['.ts', '.tsx', '.js', '.jsx', '.mjs']):
                continue

            lines = self._get_file_lines(fp)
            if not lines:
                continue

            file_content = '\n'.join(lines)
            imports = []
            exports = []

            # Find imports
            for pattern, import_type in import_patterns:
                for match in re.finditer(pattern, file_content):
                    if import_type == 'side_effect':
                        imports.append((import_type, '*', match.group(1)))
                    elif import_type == 'dynamic':
                        imports.append((import_type, 'dynamic', match.group(1)))
                    else:
                        imports.append((import_type, match.group(1).strip(), match.group(2)))

            # Find exports
            for pattern, export_type in export_patterns:
                for match in re.finditer(pattern, file_content):
                    exports.append((export_type, match.group(1).strip()))

            if imports:
                import_map[fp] = imports
            if exports:
                export_map[fp] = exports

            # Report findings
            for import_type, items, source in imports:
                result.findings.append(Finding(
                    file=fp,
                    line=1,  # Would need more work to get exact line
                    code=f"import {{{items}}} from '{source}'",
                    issue=f"Imports from {source}",
                    confidence=Confidence.HIGH,
                    severity=Severity.INFO,
                    category="import",
                ))

        result.summary = f"Analyzed {len(import_map)} files with imports, {len(export_map)} with exports"
        return result

    def trace_websocket_flow(self) -> ToolResult:
        """
        Trace WebSocket message flow through TypeScript/JavaScript code.

        Tracks:
        - WebSocket creation and connection
        - Message handlers (onmessage, addEventListener)
        - Send operations
        - Message type definitions
        - Data flow from receive to process
        """
        result = ToolResult(tool_name="WebSocket Flow Tracer", files_scanned=len(self.files))

        # WebSocket patterns
        ws_patterns = [
            # WebSocket creation
            (r'''new\s+WebSocket\s*\(['"](wss?://[^'"]+)['"]''', "WebSocket connection", Severity.INFO),
            (r'''new\s+WebSocket\s*\(([^)]+)\)''', "WebSocket creation (dynamic URL)", Severity.LOW),

            # Socket.io
            (r'''io\s*\(\s*['"](wss?://[^'"]+)['"]''', "Socket.io connection", Severity.INFO),
            (r'''socket\.on\s*\(\s*['"]([\w:]+)['"]''', "Socket event listener", Severity.INFO),
            (r'''socket\.emit\s*\(\s*['"]([\w:]+)['"]''', "Socket emit", Severity.INFO),

            # Message handlers
            (r'''\.onmessage\s*=\s*(?:async\s*)?\(?(\w*)''', "onmessage handler", Severity.INFO),
            (r'''\.addEventListener\s*\(\s*['"](message|open|close|error)['"]''', "WebSocket event listener", Severity.INFO),

            # Send operations
            (r'''\.send\s*\(\s*JSON\.stringify\s*\(([^)]+)\)''', "WebSocket send (JSON)", Severity.INFO),
            (r'''\.send\s*\(([^)]+)\)''', "WebSocket send", Severity.INFO),

            # Message parsing
            (r'''JSON\.parse\s*\(\s*(?:event|e|msg|message)\.data\s*\)''', "Message parsing", Severity.INFO),

            # Type definitions (TypeScript)
            (r'''interface\s+(\w*[Mm]essage\w*)\s*\{''', "Message type definition", Severity.INFO),
            (r'''type\s+(\w*[Mm]essage\w*)\s*=''', "Message type alias", Severity.INFO),
        ]

        # Track WebSocket-related files and handlers
        ws_files = set()
        handlers = []
        connections = []
        message_types = []

        for pattern, issue, severity in ws_patterns:
            matches = self._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if not any(filepath.endswith(ext) for ext in ['.ts', '.tsx', '.js', '.jsx']):
                    continue

                ws_files.add(filepath)

                # Categorize finding
                if "connection" in issue.lower() or "creation" in issue.lower():
                    connections.append((filepath, line_num, match.group(1) if match.groups() else ""))
                elif "handler" in issue.lower() or "listener" in issue.lower():
                    handlers.append((filepath, line_num, match.group(1) if match.groups() else ""))
                elif "type" in issue.lower():
                    message_types.append((filepath, line_num, match.group(1) if match.groups() else ""))

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:150],
                    issue=issue,
                    confidence=Confidence.HIGH,
                    severity=severity,
                    category="websocket",
                ))

        # Build flow summary
        if connections or handlers:
            flow_summary = f"WebSocket Flow: {len(connections)} connections, {len(handlers)} handlers, {len(message_types)} types in {len(ws_files)} files"
            result.summary = flow_summary

            # Add flow trace finding
            if len(ws_files) > 0:
                trace = "WebSocket Message Flow:\n"
                if connections:
                    trace += f"  Connections: {', '.join(f'{c[0]}:{c[1]}' for c in connections[:5])}\n"
                if handlers:
                    trace += f"  Handlers: {', '.join(f'{h[0]}:{h[1]}' for h in handlers[:5])}\n"
                if message_types:
                    trace += f"  Types: {', '.join(t[2] for t in message_types[:5])}"

                result.findings.insert(0, Finding(
                    file="[FLOW TRACE]",
                    line=0,
                    code=trace,
                    issue="WebSocket message flow summary",
                    confidence=Confidence.HIGH,
                    severity=Severity.INFO,
                    category="websocket_flow",
                ))
        else:
            result.summary = "No WebSocket usage found"

        return result

    def build_call_graph(self, entry_point: str | None = None) -> ToolResult:
        """
        Build a function call graph for TypeScript/JavaScript.

        Traces function calls and method invocations to understand code flow.

        Args:
            entry_point: Optional function name to start tracing from
        """
        result = ToolResult(tool_name="Call Graph Builder", files_scanned=len(self.files))

        # Function definition patterns
        func_def_patterns = [
            # Named functions
            (r'''function\s+(\w+)\s*\(([^)]*)\)''', 'function'),
            # Arrow functions assigned to const/let
            (r'''(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>''', 'arrow'),
            # Class methods
            (r'''(?:async\s+)?(\w+)\s*\(([^)]*)\)\s*\{''', 'method'),
            # TypeScript methods with access modifiers
            (r'''(?:public|private|protected)\s+(?:async\s+)?(\w+)\s*\(([^)]*)\)''', 'ts_method'),
        ]

        # Call patterns
        call_patterns = [
            r'''(\w+)\s*\(''',  # Direct function call
            r'''this\.(\w+)\s*\(''',  # this.method() call
            r'''await\s+(\w+)\s*\(''',  # await function()
            r'''\.(\w+)\s*\(''',  # method call on object
        ]

        functions = {}  # name -> (file, line, params)
        calls = {}  # caller -> list of callees

        for filepath in self.files:
            if not any(filepath.endswith(ext) for ext in ['.ts', '.tsx', '.js', '.jsx']):
                continue

            lines = self._get_file_lines(filepath)
            if not lines:
                continue

            current_function = None

            for line_num, line in enumerate(lines, 1):
                # Check for function definitions
                for pattern, func_type in func_def_patterns:
                    match = re.search(pattern, line)
                    if match:
                        func_name = match.group(1)
                        params = match.group(2) if len(match.groups()) > 1 else ""
                        functions[func_name] = (filepath, line_num, params, func_type)
                        current_function = func_name

                        result.findings.append(Finding(
                            file=filepath,
                            line=line_num,
                            code=f"{func_type}: {func_name}({params})",
                            issue=f"Function definition: {func_name}",
                            confidence=Confidence.HIGH,
                            severity=Severity.INFO,
                            category="function_def",
                        ))
                        break

                # Track function calls within current function
                if current_function:
                    for call_pattern in call_patterns:
                        for match in re.finditer(call_pattern, line):
                            callee = match.group(1)
                            # Skip common built-ins and noise
                            if callee in ['if', 'for', 'while', 'switch', 'catch', 'console', 'log', 'error', 'warn']:
                                continue
                            if current_function not in calls:
                                calls[current_function] = set()
                            calls[current_function].add(callee)

        # Build call graph summary
        if entry_point and entry_point in functions:
            # Trace from entry point
            visited = set()
            trace = []

            def trace_calls(func_name, depth=0):
                if func_name in visited or depth > 5:
                    return
                visited.add(func_name)
                indent = "  " * depth
                if func_name in functions:
                    fp, ln, params, ftype = functions[func_name]
                    trace.append(f"{indent}{func_name}() @ {fp}:{ln}")
                else:
                    trace.append(f"{indent}{func_name}() [external]")

                if func_name in calls:
                    for callee in sorted(calls[func_name]):
                        trace_calls(callee, depth + 1)

            trace_calls(entry_point)
            trace_str = "\n".join(trace)

            result.findings.insert(0, Finding(
                file="[CALL GRAPH]",
                line=0,
                code=trace_str[:500],
                issue=f"Call graph from {entry_point}",
                confidence=Confidence.MEDIUM,
                severity=Severity.INFO,
                category="call_graph",
            ))

        result.summary = f"Found {len(functions)} functions, {sum(len(c) for c in calls.values())} calls"
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

    def find_long_functions(self, max_lines: int = 30) -> ToolResult:
        """
        Find functions that are too long.

        Args:
            max_lines: Maximum acceptable function length (default: 30, lowered from 50 for better sensitivity)
        """
        result = ToolResult(tool_name=f"Long Function Scanner (>{max_lines} lines)", files_scanned=len(self.files))

        # Function start patterns by language
        func_patterns = [
            (r'''^\s*def\s+(\w+)\s*\(''', '.py'),  # Python
            (r'''^\s*(?:async\s+)?func\s+(\w+)\s*\(''', '.swift'),  # Swift
            (r'''^\s*(?:async\s+)?function\s+(\w+)\s*\(''', '.js'),  # JavaScript
            (r'''^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(''', '.js'),  # JS arrow functions
            (r'''^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(''', '.ts'),  # TS arrow functions
            (r'''^\s*(?:async\s+)?function\s+(\w+)\s*\(''', '.ts'),  # TypeScript
            (r'''^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+(\w+)\s*\(''', '.java'),  # Java
            (r'''^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)''', '.rs'),  # Rust
            (r'''^\s*func\s+(\w+)''', '.go'),  # Go
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
                    # Determine severity based on length thresholds
                    if length > 100:
                        severity = Severity.HIGH
                    elif length > 60:
                        severity = Severity.MEDIUM
                    else:
                        severity = Severity.LOW

                    result.findings.append(Finding(
                        file=filepath,
                        line=start_line + 1,
                        code=f"def {func_name}(...): # {length} lines",
                        issue=f"Function too long ({length} lines > {max_lines})",
                        confidence=Confidence.HIGH,
                        severity=severity,
                        fix=f"Refactor into smaller functions",
                        category="long_function",
                    ))

        result.summary = f"Found {len(result.findings)} functions over {max_lines} lines"
        return result

    def find_complex_functions(self, max_complexity: int = 10) -> ToolResult:
        """
        Find functions with high cyclomatic complexity.

        Approximates complexity by counting:
        - if/elif/else statements
        - for/while loops
        - try/except blocks
        - and/or operators
        - ternary operators

        Args:
            max_complexity: Maximum acceptable complexity score (default: 10)
        """
        result = ToolResult(tool_name=f"Complexity Scanner (>{max_complexity})", files_scanned=len(self.files))

        # Complexity indicators by pattern
        complexity_patterns = [
            (r'\bif\b', 1),
            (r'\belif\b', 1),
            (r'\belse\b', 1),
            (r'\bfor\b', 1),
            (r'\bwhile\b', 1),
            (r'\btry\b', 1),
            (r'\bexcept\b', 1),
            (r'\bcatch\b', 1),
            (r'\bcase\b', 1),
            (r'\b(and|or|&&|\|\|)\b', 1),
            (r'\?.*:', 1),  # Ternary
            (r'\bguard\b', 1),  # Swift guard
            (r'\bswitch\b', 1),
        ]

        func_start_patterns = [
            (r'^\s*def\s+(\w+)\s*\(', '.py'),
            (r'^\s*(?:async\s+)?func\s+(\w+)\s*\(', '.swift'),
            (r'^\s*(?:async\s+)?function\s+(\w+)\s*\(', '.js'),
            (r'^\s*(?:async\s+)?function\s+(\w+)\s*\(', '.ts'),
        ]

        for filepath in self.files:
            lines = self._get_file_lines(filepath)
            if not lines:
                continue

            # Find applicable function pattern
            func_pattern = None
            for p, ext in func_start_patterns:
                if filepath.endswith(ext):
                    func_pattern = p
                    break

            if not func_pattern:
                continue

            # Find functions
            func_starts = []
            for i, line in enumerate(lines):
                match = re.match(func_pattern, line)
                if match:
                    func_starts.append((i, match.group(1)))

            # Calculate complexity for each function
            for i, (start_line, func_name) in enumerate(func_starts):
                end_line = func_starts[i + 1][0] if i + 1 < len(func_starts) else len(lines)
                func_content = '\n'.join(lines[start_line:end_line])

                # Count complexity
                complexity = 1  # Base complexity
                for pattern, weight in complexity_patterns:
                    matches = re.findall(pattern, func_content, re.IGNORECASE)
                    complexity += len(matches) * weight

                if complexity > max_complexity:
                    result.findings.append(Finding(
                        file=filepath,
                        line=start_line + 1,
                        code=f"{func_name}(): complexity={complexity}",
                        issue=f"High cyclomatic complexity ({complexity} > {max_complexity})",
                        confidence=Confidence.MEDIUM,
                        severity=Severity.MEDIUM if complexity < 20 else Severity.HIGH,
                        fix="Break into smaller functions, reduce nesting, simplify conditionals",
                        category="complexity",
                    ))

        result.summary = f"Found {len(result.findings)} functions with complexity > {max_complexity}"
        return result

    def find_code_smells(self) -> ToolResult:
        """
        Find common code smells and anti-patterns.

        Searches for:
        - Magic numbers (non-obvious numeric literals)
        - Deep nesting (4+ levels)
        - God classes (too many methods)
        - Long parameter lists
        - Duplicate code patterns
        """
        result = ToolResult(tool_name="Code Smell Scanner", files_scanned=len(self.files))

        # Magic numbers (excluding 0, 1, 2, -1, common values)
        magic_number_pattern = r'(?<![a-zA-Z_])(?<![\d.])([3-9]\d{2,}|[1-9]\d{3,})(?![\d.])'

        # Long parameter lists (5+ params)
        long_params_pattern = r'def\s+\w+\s*\([^)]*,\s*[^)]*,\s*[^)]*,\s*[^)]*,\s*[^)]*\)'

        # Deep nesting indicators (4+ indent levels = 16+ spaces or 4+ tabs)
        deep_nesting_pattern = r'^(\s{16,}|\t{4,})\S'

        for filepath in self.files:
            lines = self._get_file_lines(filepath)
            if not lines:
                continue

            # Skip test files for some checks
            is_test = self._is_test_file(filepath)

            for line_num, line in enumerate(lines, 1):
                # Magic numbers (skip tests and constants)
                if not is_test and not self._is_in_comment(line, filepath):
                    magic_matches = re.findall(magic_number_pattern, line)
                    for magic in magic_matches:
                        # Skip if it looks like a year, port, or common constant
                        if int(magic) in [80, 443, 8080, 3000, 5000, 8000, 1000, 1024, 2048, 4096]:
                            continue
                        result.findings.append(Finding(
                            file=filepath,
                            line=line_num,
                            code=line.strip()[:100],
                            issue=f"Magic number: {magic}",
                            confidence=Confidence.LOW,
                            severity=Severity.LOW,
                            fix="Extract to named constant",
                            category="magic_number",
                        ))

                # Deep nesting
                if re.match(deep_nesting_pattern, line):
                    result.findings.append(Finding(
                        file=filepath,
                        line=line_num,
                        code=line.strip()[:100],
                        issue="Deep nesting (4+ levels)",
                        confidence=Confidence.MEDIUM,
                        severity=Severity.MEDIUM,
                        fix="Extract nested logic into helper functions",
                        category="deep_nesting",
                    ))

            # Check for long parameter lists
            file_content = '\n'.join(lines)
            for match in re.finditer(long_params_pattern, file_content):
                line_num = file_content[:match.start()].count('\n') + 1
                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=match.group(0)[:100],
                    issue="Long parameter list (5+ params)",
                    confidence=Confidence.MEDIUM,
                    severity=Severity.LOW,
                    fix="Consider using a config object or builder pattern",
                    category="long_params",
                ))

        result.summary = f"Found {len(result.findings)} code smells"
        return result

    def find_dead_code(self) -> ToolResult:
        """
        Find potentially dead/unreachable code.

        Searches for:
        - Code after return/throw/break/continue
        - Unused variables (basic detection)
        - Commented-out code blocks
        - #if false / #if DEBUG blocks (Swift/ObjC)
        """
        result = ToolResult(tool_name="Dead Code Scanner", files_scanned=len(self.files))

        patterns = [
            # Code after return
            (r'^\s*return\s+[^;]*;\s*\n\s*[a-zA-Z]', "Code after return statement", Severity.MEDIUM),

            # Swift #if false
            (r'#if\s+false', "#if false block (dead code)", Severity.LOW),

            # Commented-out code (looks like real code)
            (r'//\s*(if|for|while|func|def|class|return|var|let|const)\s+', "Commented-out code", Severity.LOW),

            # Python pass in non-stub context
            (r'^\s+pass\s*$', "Pass statement (potential placeholder)", Severity.LOW),

            # Unreachable after throw/raise
            (r'(throw|raise)\s+[^;]*\n\s*[a-zA-Z]', "Code after throw/raise", Severity.MEDIUM),
        ]

        for pattern, issue, severity in patterns:
            matches = self._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if self._is_test_file(filepath):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:100],
                    issue=issue,
                    confidence=Confidence.LOW,
                    severity=severity,
                    fix="Remove or refactor dead code",
                    category="dead_code",
                ))

        result.summary = f"Found {len(result.findings)} potential dead code areas"
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
        # Note: Case-insensitive matching is handled by _search_pattern with re.IGNORECASE
        key_pattern = '|'.join(sensitive_keys)
        patterns = [
            # UserDefaults.standard.set with sensitive key
            (rf'''UserDefaults[^)]*\.set\([^)]*({key_pattern})''',
             "Sensitive data in UserDefaults - use Keychain", Severity.CRITICAL),

            # UserDefaults storing with sensitive key name
            (rf'''UserDefaults[^)]*forKey:\s*["'].*({key_pattern})''',
             "Sensitive key in UserDefaults - use Keychain", Severity.CRITICAL),

            # @AppStorage with sensitive key
            (rf'''@AppStorage\s*\(\s*["'].*({key_pattern})''',
             "@AppStorage with sensitive data - use Keychain", Severity.CRITICAL),

            # Storing in plist/file (not encrypted)
            (rf'''\.write\([^)]*({key_pattern})[^)]*toFile:''',
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

    # ===== PERSISTENCE/STATE TOOLS =====

    def find_persistence_patterns(self) -> ToolResult:
        """
        Find data persistence patterns in the codebase.

        Searches for:
        - localStorage/sessionStorage (JavaScript)
        - UserDefaults/Keychain (iOS)
        - CoreData/SwiftData/Realm (iOS)
        - SharedPreferences (Android)
        - File I/O operations
        - Database operations
        """
        result = ToolResult(tool_name="Persistence Pattern Scanner", files_scanned=len(self.files))

        patterns = [
            # JavaScript/Web persistence
            (r'''localStorage\.(get|set|remove)Item''', "localStorage usage", Severity.INFO, ".js"),
            (r'''sessionStorage\.(get|set|remove)Item''', "sessionStorage usage", Severity.INFO, ".js"),
            (r'''indexedDB\.open''', "IndexedDB usage", Severity.INFO, ".js"),
            (r'''\.setItem\(['"]\w+['"]''', "Storage setItem", Severity.INFO, None),

            # iOS persistence
            (r'''UserDefaults\.(standard\.)?set''', "UserDefaults write", Severity.INFO, ".swift"),
            (r'''UserDefaults\.(standard\.)?(?:string|integer|bool|data|object)\(forKey:''', "UserDefaults read", Severity.INFO, ".swift"),
            (r'''@AppStorage\s*\(['"]\w+['"]''', "@AppStorage usage", Severity.INFO, ".swift"),
            (r'''NSKeyedArchiver\.archive''', "NSKeyedArchiver usage", Severity.MEDIUM, ".swift"),
            (r'''NSKeyedUnarchiver\.unarchive''', "NSKeyedUnarchiver usage", Severity.MEDIUM, ".swift"),
            (r'''FileManager\.(default\.)?(?:createFile|write|contents)''', "FileManager I/O", Severity.INFO, ".swift"),

            # CoreData/SwiftData
            (r'''NSManagedObjectContext''', "CoreData context", Severity.INFO, ".swift"),
            (r'''@FetchRequest''', "CoreData @FetchRequest", Severity.INFO, ".swift"),
            (r'''ModelContext''', "SwiftData ModelContext", Severity.INFO, ".swift"),
            (r'''@Model\s+class''', "SwiftData @Model", Severity.INFO, ".swift"),
            (r'''@Query\s+var''', "SwiftData @Query", Severity.INFO, ".swift"),

            # Realm
            (r'''Realm\(\)''', "Realm database", Severity.INFO, ".swift"),
            (r'''realm\.write''', "Realm write transaction", Severity.INFO, ".swift"),

            # Python persistence
            (r'''pickle\.(dump|load)''', "Pickle serialization", Severity.MEDIUM, ".py"),
            (r'''json\.(dump|load)s?\(''', "JSON serialization", Severity.INFO, ".py"),
            (r'''sqlite3\.connect''', "SQLite connection", Severity.INFO, ".py"),
            (r'''with open\([^)]+,\s*['"](w|a|wb|ab)['"]''', "File write operation", Severity.INFO, ".py"),

            # General database
            (r'''\.execute\(['"](INSERT|UPDATE|DELETE)''', "SQL write operation", Severity.INFO, None),
            (r'''\.query\(['"](SELECT|INSERT|UPDATE)''', "Database query", Severity.INFO, None),
        ]

        for pattern, issue, severity, file_filter in patterns:
            matches = self._search_pattern(pattern, file_filter=file_filter)
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
                    category="persistence",
                ))

        result.summary = f"Found {len(result.findings)} persistence patterns"
        return result

    def find_state_mutations(self) -> ToolResult:
        """
        Find state management and mutation patterns.

        Searches for:
        - React useState/useReducer
        - SwiftUI @State/@Binding/@Published
        - Redux/MobX patterns
        - Observable patterns
        - Event listeners/handlers
        """
        result = ToolResult(tool_name="State Mutation Scanner", files_scanned=len(self.files))

        patterns = [
            # React state
            (r'''useState\s*[<(]''', "React useState", Severity.INFO, None),
            (r'''useReducer\s*\(''', "React useReducer", Severity.INFO, None),
            (r'''setState\s*\(''', "React setState", Severity.INFO, None),
            (r'''useContext\s*\(''', "React useContext", Severity.INFO, None),

            # Redux
            (r'''dispatch\s*\(\s*\{?\s*type:''', "Redux dispatch", Severity.INFO, None),
            (r'''createSlice\s*\(''', "Redux createSlice", Severity.INFO, None),
            (r'''useSelector\s*\(''', "Redux useSelector", Severity.INFO, None),
            (r'''createStore\s*\(''', "Redux createStore", Severity.INFO, None),

            # SwiftUI state
            (r'''@State\s+(?:private\s+)?var''', "SwiftUI @State", Severity.INFO, ".swift"),
            (r'''@Binding\s+var''', "SwiftUI @Binding", Severity.INFO, ".swift"),
            (r'''@Published\s+var''', "Combine @Published", Severity.INFO, ".swift"),
            (r'''@ObservedObject\s+var''', "SwiftUI @ObservedObject", Severity.INFO, ".swift"),
            (r'''@StateObject\s+var''', "SwiftUI @StateObject", Severity.INFO, ".swift"),
            (r'''@EnvironmentObject\s+var''', "SwiftUI @EnvironmentObject", Severity.INFO, ".swift"),
            (r'''@Environment\s*\(''', "SwiftUI @Environment", Severity.INFO, ".swift"),
            (r'''@Observable\s+class''', "Observation @Observable", Severity.INFO, ".swift"),

            # Combine/RxSwift
            (r'''\.sink\s*\{''', "Combine sink subscription", Severity.INFO, ".swift"),
            (r'''\.assign\s*\(to:''', "Combine assign", Severity.INFO, ".swift"),
            (r'''PassthroughSubject''', "Combine PassthroughSubject", Severity.INFO, ".swift"),
            (r'''CurrentValueSubject''', "Combine CurrentValueSubject", Severity.INFO, ".swift"),

            # Event listeners
            (r'''addEventListener\s*\(['"]\w+['"]''', "Event listener", Severity.INFO, ".js"),
            (r'''\.on\(['"]\w+['"]''', "Event handler (.on)", Severity.INFO, None),
            (r'''NotificationCenter\.default\.addObserver''', "NotificationCenter observer", Severity.INFO, ".swift"),

            # MobX
            (r'''@observable''', "MobX @observable", Severity.INFO, None),
            (r'''@action''', "MobX @action", Severity.INFO, None),
            (r'''makeAutoObservable''', "MobX makeAutoObservable", Severity.INFO, None),

            # Vue
            (r'''ref\s*\(''', "Vue ref()", Severity.INFO, ".vue"),
            (r'''reactive\s*\(''', "Vue reactive()", Severity.INFO, ".vue"),
            (r'''computed\s*\(''', "Vue computed()", Severity.INFO, ".vue"),
        ]

        for pattern, issue, severity, file_filter in patterns:
            matches = self._search_pattern(pattern, file_filter=file_filter)
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
                    category="state",
                ))

        result.summary = f"Found {len(result.findings)} state mutation patterns"
        return result

    def run_persistence_scan(self, min_confidence: str = "LOW") -> list[ToolResult]:
        """
        Run all persistence and state management scans.

        Returns findings about:
        - Data persistence (localStorage, UserDefaults, databases)
        - State management (React, SwiftUI, Redux, etc.)
        """
        return [
            self.find_persistence_patterns(),
            self.find_state_mutations(),
        ]

    # ===== RUN ALL SECURITY =====

    def run_security_scan(
        self,
        min_confidence: str = "MEDIUM",
        include_quality: bool = False
    ) -> list[ToolResult]:
        """
        Run all security-related scans.

        Args:
            min_confidence: Minimum confidence level ("LOW", "MEDIUM", "HIGH")
                           Default is MEDIUM to filter noise.
            include_quality: If True, include code quality checks.
        """
        all_results = [
            self.find_secrets(),
            self.find_sql_injection(),
            self.find_command_injection(),
            self.find_xss_vulnerabilities(),
            self.find_python_security(),
            self.find_insecure_storage(),
            self.find_input_sanitization_issues(),
        ]

        if include_quality:
            all_results.extend(self.run_quality_scan(min_confidence="LOW"))

        if min_confidence.upper() != "LOW":
            all_results = self._filter_by_confidence(all_results, min_confidence)

        return all_results

    # ===== RUN ALL QUALITY =====

    def run_quality_scan(self, min_confidence: str = "LOW", include_all: bool = True) -> list[ToolResult]:
        """
        Run all quality-related scans (style, maintainability, complexity).

        Includes:
        - Long functions (>30 lines, lowered threshold)
        - Complex functions (cyclomatic complexity >10)
        - Code smells (magic numbers, deep nesting, long params)
        - Dead code detection
        - TODO/FIXME comments

        Note: Quality scans are excluded from iOS/security scans by default.
        Use include_quality=True to include them, or run separately.

        Args:
            min_confidence: Minimum confidence level ("LOW", "MEDIUM", "HIGH")
            include_all: If True, run all quality checks. If False, run only basic checks.
        """
        all_results = [
            self.find_long_functions(),  # Lowered to 30 lines threshold
            self.find_todos(),
        ]

        # Add comprehensive quality checks if requested
        if include_all:
            all_results.extend([
                self.find_complex_functions(),
                self.find_code_smells(),
                self.find_dead_code(),
            ])

        if min_confidence.upper() != "LOW":
            all_results = self._filter_by_confidence(all_results, min_confidence)

        return all_results

    # ===== RUN ALL iOS =====

    def run_ios_scan(
        self,
        min_confidence: str = "MEDIUM",
        include_quality: bool = False
    ) -> list[ToolResult]:
        """
        Run all iOS/Swift scans including new checks.

        Args:
            min_confidence: Minimum confidence level to include ("LOW", "MEDIUM", "HIGH")
                           Default is MEDIUM to filter noise. Use "LOW" for comprehensive scan.
            include_quality: If True, include code quality checks (long functions, TODOs).
                           Default is False to focus on bugs/security issues.
        """
        all_results = [
            # Security & crash issues (always included)
            self.find_force_unwraps(),
            self.find_retain_cycles(),
            self.find_main_thread_violations(),
            self.find_weak_self_issues(),
            self.find_cloudkit_issues(),
            self.find_cloudkit_sync_issues(),  # New: CloudKit sync patterns
            self.find_swiftdata_issues(),
            self.find_insecure_storage(),
            self.find_keychain_issues(),  # New: Keychain security
            self.find_input_sanitization_issues(),
            self.find_missing_jailbreak_detection(),
            self.find_task_cancellation_issues(),
            self.find_mainactor_issues(),
            self.find_stateobject_issues(),
        ]

        # Only include style/quality checks if explicitly requested
        if include_quality:
            all_results.append(self.find_deprecated_apis())
            all_results.extend(self.run_quality_scan(min_confidence="LOW"))

        # Filter by confidence
        if min_confidence.upper() != "LOW":
            all_results = self._filter_by_confidence(all_results, min_confidence)

        return all_results

    def _filter_by_confidence(
        self,
        results: list[ToolResult],
        min_confidence: str
    ) -> list[ToolResult]:
        """
        Filter results to only include findings at or above minimum confidence.

        Args:
            results: List of ToolResult objects
            min_confidence: "LOW", "MEDIUM", or "HIGH"

        Returns:
            Filtered list with updated summaries
        """
        confidence_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        min_level = confidence_order.get(min_confidence.upper(), 0)

        filtered_results = []
        for result in results:
            filtered_findings = [
                f for f in result.findings
                if confidence_order.get(f.confidence.value, 0) >= min_level
            ]

            if filtered_findings or not result.findings:
                # Create new result with filtered findings
                new_result = ToolResult(
                    tool_name=result.tool_name,
                    findings=filtered_findings,
                    summary=f"{result.summary} (filtered: >={min_confidence})",
                    files_scanned=result.files_scanned,
                    errors=result.errors,
                )
                filtered_results.append(new_result)

        return filtered_results


def create_tools(content: str) -> StructuredTools:
    """Factory function to create tools instance."""
    return StructuredTools(content)
