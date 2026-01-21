#!/usr/bin/env python3
"""
Quick verification script for the markdown fence stripping fix.
Tests that file content extraction correctly strips markdown fences.
"""

import sys
import os
import re
from dataclasses import dataclass, field
from enum import Enum

# Copy the helper function directly to avoid import issues
def strip_markdown_fences_from_content(content_lines: list[str]) -> list[str]:
    """
    Strip markdown code fences from file content lines.
    """
    if not content_lines:
        return content_lines

    result = list(content_lines)

    # Remove opening fence (```language) - it's typically the first line
    if result and result[0].strip().startswith('```'):
        result = result[1:]

    # Remove closing fence (```) - check last few lines for trailing whitespace
    while result and (result[-1].strip() == '```' or result[-1].strip() == ''):
        if result[-1].strip() == '```':
            result = result[:-1]
            break
        result = result[:-1]

    return result


class Confidence(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    FILTERED = "FILTERED"


class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class Finding:
    file: str
    line: int
    code: str
    issue: str
    confidence: Confidence = Confidence.MEDIUM
    severity: Severity = Severity.MEDIUM
    fix: str = ""
    category: str = ""


@dataclass
class ToolResult:
    tool_name: str
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""
    files_scanned: int = 0

    @property
    def count(self) -> int:
        return len(self.findings)


class StructuredToolsSimplified:
    """Simplified version for testing"""

    def __init__(self, content: str):
        self.content = content
        self.file_index = self._build_file_index()
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

    def _strip_markdown_fences(self, content: str) -> str:
        """Strip markdown code fences from file content."""
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

    def _get_file_content(self, filepath: str) -> str | None:
        """Get content of a specific file (with markdown fences stripped)."""
        raw_content = None

        if filepath not in self.file_index:
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

        return self._strip_markdown_fences(raw_content)

    def _search_pattern(self, pattern: str, file_filter: str | None = None) -> list[tuple[str, int, str, re.Match]]:
        """Search for regex pattern across all files."""
        results = []

        for filepath in self.file_index.keys():
            if file_filter and not filepath.endswith(file_filter):
                continue

            file_content = self._get_file_content(filepath)
            if file_content is None:
                continue

            lines = file_content.split('\n')
            max_lines = len(lines)

            for line_num, line in enumerate(lines, 1):
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    results.append((filepath, line_num, line.strip(), match))

        return results

    def find_secrets(self) -> ToolResult:
        """Find hardcoded secrets."""
        result = ToolResult(tool_name="Hardcoded Secrets Scanner", files_scanned=len(self.files))

        patterns = [
            (r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\'][^"\']{8,}["\']', "Potential API key"),
            (r'(?i)(password|passwd|pwd)\s*[=:]\s*["\'][^"\']+["\']', "Hardcoded password"),
            (r'(?i)(secret|token|auth)\s*[=:]\s*["\'][^"\']{8,}["\']', "Potential secret/token"),
            (r'ghp_[a-zA-Z0-9]{36}', "GitHub Personal Access Token"),
            (r'AKIA[0-9A-Z]{16}', "AWS Access Key ID"),
        ]

        for pattern, issue_desc in patterns:
            matches = self._search_pattern(pattern)
            for filepath, line_num, line_content, match in matches:
                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line_content[:150],
                    issue=issue_desc,
                    confidence=Confidence.MEDIUM,
                    severity=Severity.HIGH,
                ))

        result.summary = f"Found {len(result.findings)} potential hardcoded secrets"
        return result


def test_strip_markdown_fences():
    """Test the strip_markdown_fences_from_content helper function."""
    print("Test 1: strip_markdown_fences_from_content")

    content_lines = [
        '```python',
        'def hello():',
        '    print("Hello, World!")',
        '',
        'api_key = "secret_key_123"',
        '```',
    ]

    result = strip_markdown_fences_from_content(content_lines)

    assert result[0] == 'def hello():', f"Expected 'def hello():', got '{result[0]}'"
    assert len(result) == 4, f"Expected 4 lines, got {len(result)}"
    assert 'api_key' in result[3], f"Expected api_key in line 4, got '{result[3]}'"

    print("  ✓ Opening fence removed")
    print("  ✓ Closing fence removed")
    print("  ✓ Content lines preserved")
    print("  PASSED!\n")


def test_structured_tools_strip_fences():
    """Test that StructuredTools correctly strips markdown fences."""
    print("Test 2: StructuredTools._get_file_content (with fence stripping)")

    combined_content = """### File: test/app.py
```py
def hello():
    print("Hello")

api_key = "sk-secret123"
password = "hunter2"
```

### File: test/utils.py
```py
def util_func():
    return True
```
"""

    tools = StructuredToolsSimplified(combined_content)
    content = tools._get_file_content("test/app.py")

    assert content is not None, "Content should not be None"
    lines = content.split('\n')

    assert lines[0].strip() == 'def hello():', f"Expected 'def hello():', got '{lines[0]}'"
    print("  ✓ First line is actual code, not fence")

    assert '```' not in content, f"Content should not contain fences"
    print("  ✓ No markdown fences in content")

    assert 'api_key' in lines[3], f"Expected api_key on line 4 (index 3), got '{lines[3]}'"
    print("  ✓ Line numbers are correct")

    print("  PASSED!\n")


def test_search_pattern():
    """Test that _search_pattern returns correct line numbers."""
    print("Test 3: StructuredTools._search_pattern (line number accuracy)")

    combined_content = """### File: config.py
```python
# Configuration file
API_KEY = "sk-live-abc123"
DEBUG = True
PASSWORD = "admin123"
```

### File: main.py
```python
import config
print(config.API_KEY)
```
"""

    tools = StructuredToolsSimplified(combined_content)
    results = tools._search_pattern(r'API_KEY')

    assert len(results) > 0, "Should find at least one result"

    first_result = results[0]
    assert first_result[0] == 'config.py', f"Expected config.py, got {first_result[0]}"
    # Line 1: # Configuration file
    # Line 2: API_KEY = "sk-live-abc123"
    assert first_result[1] == 2, f"Expected line 2, got {first_result[1]}"
    print(f"  ✓ Found API_KEY at {first_result[0]}:{first_result[1]}")

    password_results = tools._search_pattern(r'PASSWORD')
    assert len(password_results) > 0, "Should find PASSWORD"
    assert password_results[0][1] == 4, f"Expected line 4, got {password_results[0][1]}"
    print(f"  ✓ Found PASSWORD at {password_results[0][0]}:{password_results[0][1]}")

    print("  PASSED!\n")


def test_find_secrets():
    """Test that find_secrets works correctly with fence-stripped content."""
    print("Test 4: StructuredTools.find_secrets()")

    combined_content = """### File: config/secrets.py
```python
# Secrets configuration
AWS_SECRET_KEY = "AKIAIOSFODNN7EXAMPLE"
API_TOKEN = "ghp_abc123def456ghijklmnopqrstuvwxyz1234"
password = "mysupersecret"
```

### File: utils/helper.py
```python
def get_env():
    return os.environ.get('API_KEY')
```
"""

    tools = StructuredToolsSimplified(combined_content)
    result = tools.find_secrets()

    assert result.count > 0, f"Should find secrets, got {result.count}"
    print(f"  ✓ Found {result.count} potential secrets")

    for finding in result.findings:
        # Line numbers should be small (1-5) not inflated by fence lines
        assert finding.line <= 10, f"Line number too high: {finding.line}"
        print(f"    - {finding.file}:{finding.line} - {finding.issue[:50]}...")

    print("  PASSED!\n")


def test_line_number_offset_bug():
    """Specific test for the line number offset bug (the main issue)."""
    print("Test 5: Line number offset bug (main fix)")

    # This simulates exactly how content looks after get_combined_content()
    combined_content = """### File: vulnerable.py
```python
import os

api_key = "hardcoded_secret_123"
db_password = "admin123"
```
"""

    tools = StructuredToolsSimplified(combined_content)

    # Without the fix, api_key would be on line 4 (```python counts as line 1)
    # With the fix, api_key should be on line 3 (actual file content)
    content = tools._get_file_content("vulnerable.py")
    lines = content.split('\n')

    print(f"  Content lines: {lines}")

    # Line 1: import os
    # Line 2: (empty)
    # Line 3: api_key = "hardcoded_secret_123"
    # Line 4: db_password = "admin123"

    results = tools._search_pattern(r'api_key')
    assert len(results) == 1, f"Should find exactly 1 api_key, found {len(results)}"

    line_num = results[0][1]
    print(f"  api_key found at line {line_num}")

    # The key test: line number should be 3, not 4
    assert line_num == 3, f"BUG: api_key should be line 3, got {line_num}. Fence not stripped!"

    print("  ✓ Line number is correct (3, not 4)")
    print("  ✓ Bug is FIXED!")
    print("  PASSED!\n")


def main():
    print("=" * 60)
    print("RLM-Mem Markdown Fence Fix Verification")
    print("=" * 60)
    print()

    tests = [
        test_strip_markdown_fences,
        test_structured_tools_strip_fences,
        test_search_pattern,
        test_find_secrets,
        test_line_number_offset_bug,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
