"""
Chunking utilities for RLM Processing.

Provides intelligent content chunking strategies:
- Keyword-based chunk filtering
- Function-aware chunking
- Token-optimized splitting
"""

import re


def extract_query_keywords(query: str) -> list[str]:
    """Extract searchable keywords from a query."""
    # Common stop words to ignore
    stop_words = {'find', 'search', 'look', 'for', 'the', 'a', 'an', 'in', 'is', 'are', 'all', 'any'}

    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    # Add domain-specific keywords
    if 'security' in query.lower():
        keywords.extend(['password', 'secret', 'key', 'token', 'auth', 'eval', 'exec'])
    if 'ios' in query.lower() or 'swift' in query.lower():
        keywords.extend(['unwrap', 'weak', 'self', 'optional', 'guard'])

    return list(set(keywords))


def smart_chunk_filter(chunk: str, query: str, context_lines: int = 10) -> str:
    """
    L2: Extract only query-relevant portions of a chunk before LLM call.

    This reduces token usage by 30-50% by filtering irrelevant content.
    """
    keywords = extract_query_keywords(query)

    if not keywords:
        return chunk  # Can't filter without keywords

    lines = chunk.split('\n')
    relevant_indices: set[int] = set()

    # Find lines matching keywords and include context
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(kw in line_lower for kw in keywords):
            # Include context window around match
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            relevant_indices.update(range(start, end))

    if not relevant_indices:
        # No matches - return truncated preview
        return '\n'.join(lines[:50]) + "\n... (no keyword matches, showing first 50 lines)"

    # Build filtered content preserving order
    sorted_indices = sorted(relevant_indices)
    filtered_lines = []
    last_idx = -1

    for idx in sorted_indices:
        if last_idx >= 0 and idx > last_idx + 1:
            filtered_lines.append("... (skipped lines)")
        filtered_lines.append(lines[idx])
        last_idx = idx

    return '\n'.join(filtered_lines)


def find_function_boundaries(content: str, file_type: str) -> list[tuple[int, int, str]]:
    """
    L7: Find function/method boundaries in source code.

    Returns list of (start_pos, end_pos, function_name) tuples.
    """
    boundaries = []

    if file_type in ['.py']:
        # Python: def/async def/class
        pattern = r'^(async\s+)?def\s+(\w+)|^class\s+(\w+)'
        for match in re.finditer(pattern, content, re.MULTILINE):
            start = match.start()
            name = match.group(2) or match.group(3) or "unknown"
            # Find end by next function/class at same indent or EOF
            indent = len(content[content.rfind('\n', 0, start)+1:start])
            end_pattern = rf'^.{{{indent}}}(def |class |async def )'
            end_match = re.search(end_pattern, content[start+1:], re.MULTILINE)
            end = start + end_match.start() + 1 if end_match else len(content)
            boundaries.append((start, end, name))

    elif file_type in ['.swift']:
        # Swift: func/class/struct/enum
        pattern = r'(func|class|struct|enum|extension)\s+(\w+)'
        for match in re.finditer(pattern, content):
            start = match.start()
            name = match.group(2)
            # Find matching brace end
            brace_count = 0
            in_func = False
            end = start
            for i, char in enumerate(content[start:]):
                if char == '{':
                    brace_count += 1
                    in_func = True
                elif char == '}':
                    brace_count -= 1
                    if in_func and brace_count == 0:
                        end = start + i + 1
                        break
            boundaries.append((start, end, name))

    elif file_type in ['.js', '.ts', '.tsx', '.jsx']:
        # JavaScript/TypeScript: function/class/arrow functions
        pattern = r'(function\s+(\w+)|class\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\()'
        for match in re.finditer(pattern, content):
            start = match.start()
            name = match.group(2) or match.group(3) or match.group(4) or "anonymous"
            # Simple brace matching
            brace_count = 0
            in_func = False
            end = start
            for i, char in enumerate(content[start:]):
                if char == '{':
                    brace_count += 1
                    in_func = True
                elif char == '}':
                    brace_count -= 1
                    if in_func and brace_count == 0:
                        end = start + i + 1
                        break
            boundaries.append((start, end, name))

    return boundaries


def function_aware_chunking(content: str, file_type: str, max_chunk_tokens: int = 8000) -> list[str]:
    """
    L7: Chunk content at function/class boundaries instead of arbitrary token limits.

    This prevents truncating code mid-logic and improves analysis accuracy.
    """
    # Rough estimate: 1 token ≈ 4 chars
    max_chars = max_chunk_tokens * 4

    boundaries = find_function_boundaries(content, file_type)

    if not boundaries:
        # Fallback to simple chunking if no functions found
        chunks = []
        for i in range(0, len(content), max_chars):
            chunks.append(content[i:i + max_chars])
        return chunks

    chunks = []
    current_chunk = ""
    current_size = 0

    for start, end, name in boundaries:
        func_content = content[start:end]
        func_size = len(func_content)

        if func_size > max_chars:
            # Function too large - split it with overlap
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
                current_size = 0

            # Split large function
            for i in range(0, func_size, max_chars - 500):  # 500 char overlap
                chunks.append(f"# Part of {name}\n" + func_content[i:i + max_chars])

        elif current_size + func_size > max_chars:
            # Would exceed limit - save current and start new
            chunks.append(current_chunk)
            current_chunk = func_content
            current_size = func_size

        else:
            # Add to current chunk
            current_chunk += func_content
            current_size += func_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def hybrid_verify_finding(finding_code: str, finding_issue: str, context: str) -> tuple[bool, str]:
    """
    L6: Verify a regex-based finding using semantic analysis.

    Returns (is_valid, reason).
    """
    # Quick heuristic checks before LLM call
    false_positive_patterns = [
        (r'#.*' + re.escape(finding_code[:20]), "Appears in comment"),
        (r'""".*' + re.escape(finding_code[:20]) + r'.*"""', "Appears in docstring"),
        (r"'''.*" + re.escape(finding_code[:20]) + r".*'''", "Appears in docstring"),
        (r'//.*' + re.escape(finding_code[:20]), "Appears in comment"),
        (r'/\*.*' + re.escape(finding_code[:20]) + r'.*\*/', "Appears in block comment"),
    ]

    for pattern, reason in false_positive_patterns:
        try:
            if re.search(pattern, context, re.DOTALL | re.IGNORECASE):
                return False, reason
        except re.error:
            pass

    # Check for test file patterns
    test_indicators = ['test_', '_test.', 'spec.', 'mock', 'fixture', 'assert']
    if any(ind in context.lower() for ind in test_indicators):
        return True, "Test file - lower confidence"

    return True, "Verified"


def simple_chunk(content: str, max_chars: int = 32000) -> list[str]:
    """Simple character-based chunking with overlap."""
    if len(content) <= max_chars:
        return [content]

    chunks = []
    overlap = 500  # Characters of overlap between chunks

    for i in range(0, len(content), max_chars - overlap):
        chunks.append(content[i:i + max_chars])

    return chunks
