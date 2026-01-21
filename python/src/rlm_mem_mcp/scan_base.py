"""
Base class and utilities for RLM structured scanning tools (v2.9).

Contains:
- ScannerBase: Base class with common scanning utilities
- File indexing and content extraction
- Pattern matching helpers
- Comment detection
"""

import re
from typing import Optional, Iterator, Tuple

from .common_types import Finding, Confidence, Severity, ToolResult, AnalysisContext, calculate_confidence
from .scan_patterns import SAFE_CONSTANT_PATTERNS, COMPILE_TIME_PATTERNS


class ScannerBase:
    """
    Base class for structured search tools.

    Provides common functionality:
    - File indexing and content extraction
    - Pattern matching with context
    - Comment detection
    - Confidence calculation with constant detection
    """

    def __init__(self, content: str, file_index: Optional[dict[str, tuple[int, int]]] = None):
        """
        Initialize scanner with content to search.

        Args:
            content: The full content (prompt variable)
            file_index: Optional precomputed {filepath: (start_pos, end_pos)}
        """
        self.content = content
        self.file_index = file_index or self._build_file_index()
        self.files = list(self.file_index.keys())
        self._file_contents_cache: dict[str, str] = {}

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

    def _strip_markdown_fences(self, content: str) -> str:
        """
        Strip markdown code fences from file content.

        The content format is:
        ```language
        actual content
        ```
        """
        if not content:
            return content

        lines = content.split('\n')

        # Remove opening fence (```language)
        if lines and lines[0].strip().startswith('```'):
            lines = lines[1:]

        # Remove closing fence (```)
        while lines and (lines[-1].strip() == '```' or lines[-1].strip() == ''):
            if lines[-1].strip() == '```':
                lines = lines[:-1]
                break
            lines = lines[:-1]

        return '\n'.join(lines)

    def _get_file_content(self, filepath: str) -> Optional[str]:
        """Get content of a specific file (with markdown fences stripped)."""
        # Check cache first
        if filepath in self._file_contents_cache:
            return self._file_contents_cache[filepath]

        raw_content = None

        if filepath not in self.file_index:
            # Try partial match
            for fp in self.file_index:
                if filepath in fp or fp.endswith(filepath):
                    start, end = self.file_index[fp]
                    raw_content = self.content[start:end]
                    break
        else:
            start, end = self.file_index[filepath]
            raw_content = self.content[start:end]

        if raw_content is None:
            return None

        # Strip markdown fences to get actual file content
        result = self._strip_markdown_fences(raw_content)
        self._file_contents_cache[filepath] = result
        return result

    def _get_file_lines(self, filepath: str) -> Optional[list[str]]:
        """Get lines of a specific file (with markdown fences stripped)."""
        content = self._get_file_content(filepath)
        if content is None:
            return None
        return content.split('\n')

    def _get_file_line_count(self, filepath: str) -> int:
        """Get the total number of lines in a file."""
        lines = self._get_file_lines(filepath)
        return len(lines) if lines else 0

    def _is_test_file(self, filepath: str) -> bool:
        """Check if a file is a test file."""
        lower = filepath.lower()
        return any(x in lower for x in ['test', 'spec', '_test', '.test', 'mock', 'fixture', '__tests__'])

    def _is_in_comment(self, line: str, filepath: str) -> bool:
        """
        Check if the match is inside a comment.

        Handles:
        - // single line comments
        - # single line comments
        - /* */ block comments (basic)
        """
        stripped = line.strip()

        # Single-line comments
        if stripped.startswith('//') or stripped.startswith('#'):
            return True

        # Block comment start
        if stripped.startswith('/*') or stripped.startswith('"""') or stripped.startswith("'''"):
            return True

        # Swift/Kotlin doc comments
        if stripped.startswith('///') or stripped.startswith('/**'):
            return True

        return False

    def _is_constant_pattern(self, line: str, filepath: str) -> bool:
        """
        Check if the line matches safe constant patterns.

        v2.7: Reduces false positives by detecting immutable/compile-time values.
        """
        # Determine language from file extension
        ext = filepath.split('.')[-1].lower() if '.' in filepath else ''
        lang_map = {
            'swift': 'swift',
            'js': 'javascript',
            'ts': 'javascript',
            'jsx': 'javascript',
            'tsx': 'javascript',
            'py': 'python',
            'rs': 'rust',
        }
        lang = lang_map.get(ext, '')

        # Check language-specific constant patterns
        if lang and lang in SAFE_CONSTANT_PATTERNS:
            for pattern in SAFE_CONSTANT_PATTERNS[lang]:
                if re.search(pattern, line):
                    return True

        # Check compile-time patterns (all languages)
        for pattern in COMPILE_TIME_PATTERNS:
            if re.search(pattern, line):
                return True

        return False

    def _search_pattern(
        self,
        pattern: str,
        file_filter: Optional[str] = None,
        case_insensitive: bool = False
    ) -> Iterator[Tuple[str, int, str, re.Match]]:
        """
        Search for a pattern across all files.

        Args:
            pattern: Regex pattern to search for
            file_filter: Optional file extension filter (e.g., ".swift", ".py")
            case_insensitive: Whether to ignore case

        Yields:
            (filepath, line_num, line, match) tuples
        """
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error:
            return

        for filepath in self.files:
            # Apply file filter
            if file_filter and not filepath.endswith(file_filter):
                continue

            lines = self._get_file_lines(filepath)
            if not lines:
                continue

            for line_num, line in enumerate(lines, 1):
                match = regex.search(line)
                if match:
                    yield filepath, line_num, line, match

    def _create_finding(
        self,
        filepath: str,
        line_num: int,
        line: str,
        issue: str,
        severity: Severity = Severity.MEDIUM,
        fix: str = "",
        category: str = "",
        base_confidence: Confidence = Confidence.MEDIUM
    ) -> Optional[Finding]:
        """
        Create a Finding with proper confidence calculation.

        Automatically adjusts confidence based on:
        - Whether line is in a comment
        - Whether line is a constant pattern
        - Whether file is a test file
        """
        # Build analysis context
        context = AnalysisContext(
            in_dead_code=False,
            in_test_file=self._is_test_file(filepath),
            line_verified=True,
            is_comment=self._is_in_comment(line, filepath),
            pattern_match_only=True,
            has_semantic_verification=False,
            multiple_indicators=False,
            is_constant_declaration=self._is_constant_pattern(line, filepath),
            is_immutable_collection=False,
            is_compile_time_only=any(re.search(p, line) for p in COMPILE_TIME_PATTERNS),
        )

        # Skip if it's a comment
        if context.is_comment:
            return None

        finding = Finding(
            file=filepath,
            line=line_num,
            code=line.strip()[:200],
            issue=issue,
            confidence=base_confidence,
            severity=severity,
            fix=fix,
            category=category,
        )

        # Calculate adjusted confidence
        finding.confidence = calculate_confidence(finding, context)

        # Filter out very low confidence findings
        if finding.confidence == Confidence.FILTERED:
            return None

        return finding

    def read_file(self, path: str) -> str:
        """
        Read the content of a specific file.

        This is the SAFE way to read files in the REPL sandbox.
        Files must have been collected during the initial scan.

        Args:
            path: File path (can be partial - will match if path ends with this)

        Returns:
            File content as string, or error message if not found
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

    def list_files(self, pattern: Optional[str] = None) -> str:
        """
        List all available files that were collected for analysis.

        Args:
            pattern: Optional filter pattern (case-insensitive substring match)

        Returns:
            Formatted list of files with sizes
        """
        files_to_show = self.files
        if pattern:
            pattern_lower = pattern.lower()
            files_to_show = [f for f in self.files if pattern_lower in f.lower()]

        if not files_to_show:
            if pattern:
                return f"No files matching '{pattern}'. Total files available: {len(self.files)}"
            return "No files available."

        result = [f"Found {len(files_to_show)} files:"]
        for f in files_to_show[:100]:
            result.append(f"  {f}")
        if len(files_to_show) > 100:
            result.append(f"  ... and {len(files_to_show) - 100} more")

        return "\n".join(result)
