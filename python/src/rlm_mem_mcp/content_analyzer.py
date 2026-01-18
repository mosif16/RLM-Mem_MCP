"""
Content Analyzer for RLM Processing

Provides utilities to:
1. Detect dead/conditional code blocks (#if false, #if DEBUG, etc.)
2. Verify line number references against actual content
3. Detect function implementation status (body vs signature only)
4. Assign confidence levels to findings

These improvements address false positive issues in RLM analysis.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class Confidence(Enum):
    """Confidence level for findings."""
    HIGH = "high"      # Verified in active code
    MEDIUM = "medium"  # Found but context unclear
    LOW = "low"        # Possible false positive (dead code, etc.)


@dataclass
class DeadCodeRegion:
    """Represents a region of dead/conditional code."""
    file_path: str
    start_line: int
    end_line: int
    condition: str  # e.g., "#if false", "#if DEBUG"
    language: str


@dataclass
class LineVerification:
    """Result of verifying a line number reference."""
    is_valid: bool
    actual_content: str | None
    expected_pattern: str | None
    in_dead_code: bool
    confidence: Confidence
    reason: str


@dataclass
class ImplementationStatus:
    """Status of a function/method implementation."""
    is_implemented: bool
    has_body: bool
    body_lines: int
    is_stub: bool  # Returns NotImplemented, pass, ...
    confidence: Confidence
    reason: str


# Patterns for conditional compilation across languages
CONDITIONAL_PATTERNS = {
    # Swift/Objective-C
    "swift": [
        (r'#if\s+false\b', r'#endif\b'),
        (r'#if\s+DEBUG\b', r'#endif\b'),
        (r'#if\s+!RELEASE\b', r'#endif\b'),
        (r'#if\s+NEVER\b', r'#endif\b'),
        (r'#if\s+0\b', r'#endif\b'),
    ],
    # C/C++/Objective-C preprocessor
    "c": [
        (r'#if\s+0\b', r'#endif\b'),
        (r'#if\s+false\b', r'#endif\b'),
        (r'#ifdef\s+NEVER\b', r'#endif\b'),
        (r'#if\s+defined\s*\(\s*NEVER\s*\)', r'#endif\b'),
    ],
    # Python (less common but exists)
    "python": [
        (r'if\s+False\s*:', r'(?=\nif\s|^\S|\Z)'),  # if False: block
        (r'if\s+0\s*:', r'(?=\nif\s|^\S|\Z)'),
    ],
    # JavaScript/TypeScript
    "javascript": [
        (r'if\s*\(\s*false\s*\)\s*\{', r'\}'),
        (r'if\s*\(\s*0\s*\)\s*\{', r'\}'),
    ],
}

# Stub patterns that indicate unimplemented functions
STUB_PATTERNS = [
    r'return\s+NotImplemented\b',
    r'raise\s+NotImplementedError',
    r'TODO\s*:?\s*implement',
    r'FIXME\s*:?\s*implement',
    r'pass\s*$',
    r'\.\.\.(\s*#.*)?$',  # Ellipsis in Python
    r'throw\s+new\s+Error\s*\(\s*["\']not\s+implemented',
    r'fatalError\s*\(\s*["\']not\s+implemented',
]


def detect_language(file_path: str) -> str:
    """Detect language from file extension."""
    ext = file_path.lower().rsplit('.', 1)[-1] if '.' in file_path else ''

    lang_map = {
        'swift': 'swift',
        'm': 'swift',  # Objective-C uses similar preprocessor
        'mm': 'swift',
        'h': 'c',
        'c': 'c',
        'cpp': 'c',
        'cc': 'c',
        'cxx': 'c',
        'hpp': 'c',
        'py': 'python',
        'pyw': 'python',
        'js': 'javascript',
        'jsx': 'javascript',
        'ts': 'javascript',
        'tsx': 'javascript',
        'mjs': 'javascript',
    }

    return lang_map.get(ext, 'unknown')


def find_dead_code_regions(content: str, file_path: str) -> list[DeadCodeRegion]:
    """
    Find all dead/conditional code regions in a file.

    Args:
        content: File content with line numbers (format: "1: line\\n2: line\\n...")
                 or raw content
        file_path: Path to the file (for language detection)

    Returns:
        List of DeadCodeRegion objects (deduplicated)
    """
    language = detect_language(file_path)
    regions = []
    seen_regions: set[tuple[int, int]] = set()  # Track (start, end) to deduplicate

    # Get patterns for this language
    patterns = CONDITIONAL_PATTERNS.get(language, [])

    # Also check universal patterns
    universal_patterns = [
        (r'#if\s+false\b', r'#endif\b'),
        (r'#if\s+0\b', r'#endif\b'),
    ]
    patterns = list(patterns) + universal_patterns

    # Split into lines for line number tracking
    lines = content.split('\n')

    for start_pattern, end_pattern in patterns:
        # Find all start markers
        for i, line in enumerate(lines):
            # Strip line number prefix if present
            line_content = re.sub(r'^\d+:\s*', '', line)

            start_match = re.search(start_pattern, line_content, re.IGNORECASE)
            if start_match:
                # Find matching end
                nesting = 1
                end_line = i

                for j in range(i + 1, len(lines)):
                    check_line = re.sub(r'^\d+:\s*', '', lines[j])

                    # Check for nested starts (same type)
                    if re.search(start_pattern, check_line, re.IGNORECASE):
                        nesting += 1

                    # Check for end
                    if re.search(end_pattern, check_line, re.IGNORECASE):
                        nesting -= 1
                        if nesting == 0:
                            end_line = j
                            break

                # Deduplicate: only add if we haven't seen this region
                region_key = (i + 1, end_line + 1)
                if region_key not in seen_regions:
                    seen_regions.add(region_key)
                    regions.append(DeadCodeRegion(
                        file_path=file_path,
                        start_line=i + 1,  # 1-indexed
                        end_line=end_line + 1,
                        condition=start_match.group(0),
                        language=language
                    ))

    return regions


def is_line_in_dead_code(line_num: int, regions: list[DeadCodeRegion]) -> tuple[bool, str | None]:
    """
    Check if a line number falls within a dead code region.

    Returns:
        (is_dead, condition) tuple
    """
    for region in regions:
        if region.start_line <= line_num <= region.end_line:
            return True, region.condition
    return False, None


def verify_line_reference(
    content: str,
    file_path: str,
    line_num: int,
    expected_pattern: str | None = None,
    dead_code_regions: list[DeadCodeRegion] | None = None
) -> LineVerification:
    """
    Verify that a line number reference is valid and contains expected content.

    Args:
        content: File content (with or without line numbers)
        file_path: Path to file
        line_num: Line number to verify (1-indexed)
        expected_pattern: Optional regex pattern expected on that line
        dead_code_regions: Pre-computed dead code regions (computed if None)

    Returns:
        LineVerification result
    """
    lines = content.split('\n')

    # Handle line-numbered format
    actual_lines = []
    for line in lines:
        stripped = re.sub(r'^\d+:\s*', '', line)
        actual_lines.append(stripped)

    # Check if line exists
    if line_num < 1 or line_num > len(actual_lines):
        return LineVerification(
            is_valid=False,
            actual_content=None,
            expected_pattern=expected_pattern,
            in_dead_code=False,
            confidence=Confidence.LOW,
            reason=f"Line {line_num} out of range (file has {len(actual_lines)} lines)"
        )

    actual_content = actual_lines[line_num - 1]

    # Check dead code
    if dead_code_regions is None:
        dead_code_regions = find_dead_code_regions(content, file_path)

    in_dead_code, condition = is_line_in_dead_code(line_num, dead_code_regions)

    # Verify pattern if provided
    pattern_matches = True
    if expected_pattern:
        pattern_matches = bool(re.search(expected_pattern, actual_content, re.IGNORECASE))

    # Determine confidence
    if in_dead_code:
        confidence = Confidence.LOW
        reason = f"Line is in dead code block ({condition})"
    elif not pattern_matches and expected_pattern:
        confidence = Confidence.LOW
        reason = f"Line content doesn't match expected pattern"
    else:
        confidence = Confidence.HIGH
        reason = "Verified in active code"

    return LineVerification(
        is_valid=pattern_matches,
        actual_content=actual_content,
        expected_pattern=expected_pattern,
        in_dead_code=in_dead_code,
        confidence=confidence,
        reason=reason
    )


def check_implementation_status(
    content: str,
    function_name: str,
    file_path: str
) -> ImplementationStatus:
    """
    Check if a function/method is actually implemented (not just a stub).

    Args:
        content: File content
        function_name: Name of function to check
        file_path: Path for language detection

    Returns:
        ImplementationStatus result
    """
    language = detect_language(file_path)
    lines = content.split('\n')

    # Find function definition
    func_patterns = {
        'python': rf'def\s+{re.escape(function_name)}\s*\(',
        'javascript': rf'(?:function\s+{re.escape(function_name)}|{re.escape(function_name)}\s*[=:]\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))',
        'swift': rf'func\s+{re.escape(function_name)}\s*[(<]',
        'c': rf'(?:\w+\s+)+{re.escape(function_name)}\s*\(',
    }

    pattern = func_patterns.get(language, rf'{re.escape(function_name)}\s*\(')

    func_start = None
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            func_start = i
            break

    if func_start is None:
        return ImplementationStatus(
            is_implemented=False,
            has_body=False,
            body_lines=0,
            is_stub=False,
            confidence=Confidence.MEDIUM,
            reason=f"Function '{function_name}' not found"
        )

    # Extract function body (simplified - looks for next function or significant dedent)
    body_lines = []
    in_body = False
    brace_count = 0

    for i in range(func_start, min(func_start + 100, len(lines))):
        line = lines[i]

        if '{' in line:
            brace_count += line.count('{')
            in_body = True
        if '}' in line:
            brace_count -= line.count('}')

        if in_body:
            body_lines.append(line)
            if brace_count == 0 and i > func_start:
                break
        elif language == 'python' and i > func_start:
            # Python: check for dedent
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                break
            body_lines.append(line)

    # Check for stub patterns
    body_text = '\n'.join(body_lines)
    is_stub = any(re.search(p, body_text, re.IGNORECASE | re.MULTILINE) for p in STUB_PATTERNS)

    # Count non-empty, non-comment lines
    real_lines = [l for l in body_lines if l.strip() and not l.strip().startswith(('#', '//', '/*', '*'))]

    has_real_body = len(real_lines) > 2  # More than just signature + one line

    if is_stub:
        return ImplementationStatus(
            is_implemented=False,
            has_body=True,
            body_lines=len(body_lines),
            is_stub=True,
            confidence=Confidence.HIGH,
            reason="Function contains stub/NotImplemented pattern"
        )

    if not has_real_body:
        return ImplementationStatus(
            is_implemented=False,
            has_body=len(body_lines) > 0,
            body_lines=len(body_lines),
            is_stub=False,
            confidence=Confidence.MEDIUM,
            reason="Function body appears minimal/empty"
        )

    return ImplementationStatus(
        is_implemented=True,
        has_body=True,
        body_lines=len(body_lines),
        is_stub=False,
        confidence=Confidence.HIGH,
        reason="Function has substantial implementation"
    )


def annotate_content_with_dead_code(content: str, file_path: str) -> str:
    """
    Annotate content to mark dead code regions.

    Adds markers like "[DEAD CODE START: #if false]" and "[DEAD CODE END]"
    to help LLMs recognize inactive code.

    Args:
        content: File content
        file_path: Path for language detection

    Returns:
        Annotated content
    """
    regions = find_dead_code_regions(content, file_path)

    if not regions:
        return content

    lines = content.split('\n')
    result_lines = []

    # Track which lines have annotations
    start_annotations = {r.start_line: r.condition for r in regions}
    end_annotations = {r.end_line for r in regions}

    for i, line in enumerate(lines):
        line_num = i + 1

        if line_num in start_annotations:
            result_lines.append(f"[DEAD CODE START: {start_annotations[line_num]}]")

        result_lines.append(line)

        if line_num in end_annotations:
            result_lines.append("[DEAD CODE END]")

    return '\n'.join(result_lines)


def generate_confidence_guidance() -> str:
    """
    Generate guidance text for LLMs about confidence levels.

    Returns:
        Guidance text to include in prompts
    """
    return """
## Confidence Levels (REQUIRED for all findings)

Assign a confidence level to EVERY finding:

**HIGH confidence** - Use when:
- Code is in active (non-conditional) blocks
- Line numbers verified against actual content
- Function has real implementation (not stub)
- Pattern clearly matches the issue type

**MEDIUM confidence** - Use when:
- Code context is unclear
- Cannot verify if code path is reachable
- Function exists but implementation status uncertain
- Pattern partially matches

**LOW confidence** - Use when:
- Code is in #if false, #if DEBUG, or similar blocks
- Line number couldn't be verified
- Function appears to be a stub or unimplemented
- Finding based on signatures only, not actual implementation

Format: Include `[Confidence: HIGH/MEDIUM/LOW]` with each finding.

Example:
```
**src/auth.py:42** [Confidence: HIGH]
Issue: SQL injection via string concatenation
```python
query = "SELECT * FROM users WHERE id=" + user_id
```

**src/legacy.swift:156** [Confidence: LOW - in #if false block]
Issue: Hardcoded API key (DEAD CODE - not compiled)
```swift
let apiKey = "sk-12345"
```
```
"""
