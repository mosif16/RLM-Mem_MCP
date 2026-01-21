"""
REPL Analyzers for code analysis.

Provides analyzer functions for Swift-specific analysis and
general file analysis using LLM.
"""

import re
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .repl_state import REPLState

from .repl_security import strip_markdown_fences_from_content


class REPLAnalyzers:
    """Factory for creating REPL analyzer functions."""

    def __init__(
        self,
        state: "REPLState",
        llm_query_func: Callable[[str, int], str],
        extract_with_lines_func: Callable[[str, int], str]
    ):
        self.state = state
        self.llm_query = llm_query_func
        self.extract_with_lines = extract_with_lines_func

    def create_swift_analyzer_function(self) -> Callable:
        """Create a Swift-specific issue finder."""
        prompt = self.state.prompt

        def find_swift_issues(file_path: str, issue_types: list[str] | None = None) -> list[dict]:
            """
            Find Swift-specific issues in a file.

            Args:
                file_path: Path to the Swift file
                issue_types: Optional list of issue types to check:
                    - "retain_cycle": Closures missing [weak self]
                    - "force_unwrap": Force unwraps (!)
                    - "actor_isolation": @MainActor and Sendable issues
                    - "swiftui": SwiftUI lifecycle issues

            Returns:
                List of issues with file, line, type, description, confidence
            """
            # Extract the file content
            pattern = f"### File: [^\\n]*{re.escape(file_path.split('/')[-1])}[^\\n]*\\n"
            match = re.search(pattern, prompt)
            if not match:
                return [{"error": f"File not found: {file_path}"}]

            start = match.end()
            end_match = re.search(r"\n### File:", prompt[start:])
            end = start + end_match.start() if end_match else len(prompt)
            content = prompt[start:end]

            issues = []
            # Strip markdown fences from content
            lines = strip_markdown_fences_from_content(content.split('\n'))

            # Default to all issue types
            if not issue_types:
                issue_types = ["retain_cycle", "force_unwrap", "actor_isolation", "swiftui"]

            for line_num, line in enumerate(lines, 1):
                # Retain cycle detection
                if "retain_cycle" in issue_types:
                    if re.search(r'\{\s*(?!\[(?:weak|unowned)\s+self\]).*\bself\.', line):
                        if not re.search(r'\[weak\s+self\]|\[unowned\s+self\]', line):
                            issues.append({
                                "file": file_path,
                                "line": line_num,
                                "type": "retain_cycle",
                                "code": line.strip(),
                                "description": "Closure captures self - consider [weak self]",
                                "confidence": "MEDIUM"
                            })

                # Force unwrap detection (excluding !=)
                if "force_unwrap" in issue_types:
                    if re.search(r'\w!(?!=)\s*[.\[]', line) or re.search(r'\btry!\s+', line):
                        issues.append({
                            "file": file_path,
                            "line": line_num,
                            "type": "force_unwrap",
                            "code": line.strip(),
                            "description": "Force unwrap or try! - handle optionals safely",
                            "confidence": "HIGH"
                        })

                # Actor isolation
                if "actor_isolation" in issue_types:
                    if re.search(r'DispatchQueue\.main\.async', line):
                        issues.append({
                            "file": file_path,
                            "line": line_num,
                            "type": "actor_isolation",
                            "code": line.strip(),
                            "description": "Consider @MainActor instead of DispatchQueue.main",
                            "confidence": "LOW"
                        })

                # SwiftUI lifecycle
                if "swiftui" in issue_types:
                    if re.search(r'@ObservedObject\s+var\s+\w+\s*=', line):
                        issues.append({
                            "file": file_path,
                            "line": line_num,
                            "type": "swiftui",
                            "code": line.strip(),
                            "description": "@ObservedObject with default - use @StateObject",
                            "confidence": "HIGH"
                        })

            return issues

        return find_swift_issues

    def create_file_analyzer_function(self) -> Callable:
        """Create a general file analyzer using sub-LLM with intelligent chunking."""
        llm_query = self.llm_query
        extract_with_lines = self.extract_with_lines

        # Maximum chars per chunk for sub-LLM context (conservative to leave room for prompts)
        MAX_CHUNK_CHARS = 6000
        # Overlap between chunks to maintain context
        CHUNK_OVERLAP_LINES = 20

        def chunk_content(content: str, max_chars: int = MAX_CHUNK_CHARS) -> list[tuple[str, int, int]]:
            """
            Split content into overlapping chunks for analysis.

            Args:
                content: Line-numbered content to chunk
                max_chars: Maximum characters per chunk

            Returns:
                List of (chunk_content, start_line, end_line) tuples
            """
            if len(content) <= max_chars:
                # Content fits in one chunk
                lines = content.split('\n')
                return [(content, 1, len(lines))]

            chunks = []
            lines = content.split('\n')
            current_chunk_lines = []
            current_chunk_chars = 0
            chunk_start_line = 1

            for i, line in enumerate(lines, 1):
                line_with_newline = line + '\n'

                if current_chunk_chars + len(line_with_newline) > max_chars and current_chunk_lines:
                    # Save current chunk
                    chunk_content_str = '\n'.join(current_chunk_lines)
                    chunk_end_line = i - 1
                    chunks.append((chunk_content_str, chunk_start_line, chunk_end_line))

                    # Start new chunk with overlap
                    overlap_start = max(0, len(current_chunk_lines) - CHUNK_OVERLAP_LINES)
                    current_chunk_lines = current_chunk_lines[overlap_start:]
                    chunk_start_line = i - len(current_chunk_lines)
                    current_chunk_chars = sum(len(l) + 1 for l in current_chunk_lines)

                current_chunk_lines.append(line)
                current_chunk_chars += len(line_with_newline)

            # Don't forget the last chunk
            if current_chunk_lines:
                chunk_content_str = '\n'.join(current_chunk_lines)
                chunks.append((chunk_content_str, chunk_start_line, len(lines)))

            return chunks

        def analyze_file(file_path: str, analysis_type: str = "security") -> str:
            """
            Analyze a file using the sub-LLM for semantic understanding.

            For large files, automatically chunks content and aggregates results.

            Args:
                file_path: Path to the file to analyze
                analysis_type: Type of analysis - "security", "quality", "architecture"

            Returns:
                Analysis results from sub-LLM (aggregated if chunked)
            """
            content = extract_with_lines(file_path)
            if not content or "not found" in content.lower():
                return f"File not found: {file_path}"

            # Get chunks (may be just one for small files)
            chunks = chunk_content(content)

            prompts_template = {
                "security": """Analyze this code for security issues:
{content}

Look for: hardcoded secrets, injection vulnerabilities, authentication issues, data exposure.
For each issue: specify exact line number, code snippet, severity (HIGH/MEDIUM/LOW).
Note: This is lines {start_line}-{end_line} of the file.""",

                "quality": """Analyze this code for quality issues:
{content}

Look for: complex functions, missing error handling, code duplication, unclear naming.
For each issue: specify exact line number and recommendation.
Note: This is lines {start_line}-{end_line} of the file.""",

                "architecture": """Analyze this code for architectural issues:
{content}

Look for: tight coupling, missing abstractions, violation of SOLID principles.
Describe the file's role and any concerns.
Note: This is lines {start_line}-{end_line} of the file."""
            }

            template = prompts_template.get(analysis_type, prompts_template["security"])

            if len(chunks) == 1:
                # Single chunk - simple case
                chunk_content_str, start_line, end_line = chunks[0]
                prompt = template.format(
                    content=chunk_content_str,
                    start_line=start_line,
                    end_line=end_line
                )
                return llm_query(prompt)

            # Multiple chunks - analyze each and aggregate
            all_findings = []
            for i, (chunk_content_str, start_line, end_line) in enumerate(chunks, 1):
                prompt = template.format(
                    content=chunk_content_str,
                    start_line=start_line,
                    end_line=end_line
                )
                chunk_result = llm_query(prompt)

                # Only include if there are actual findings
                if chunk_result and "no " not in chunk_result.lower()[:50]:
                    all_findings.append(f"### Chunk {i} (lines {start_line}-{end_line}):\n{chunk_result}")

            if not all_findings:
                return f"No {analysis_type} issues found in {file_path}"

            # Aggregate findings
            if len(all_findings) == 1:
                return all_findings[0]

            aggregated = f"## Analysis of {file_path} ({len(chunks)} chunks analyzed)\n\n"
            aggregated += "\n\n".join(all_findings)

            # If many findings, ask LLM to deduplicate
            if len(all_findings) > 2:
                dedup_prompt = f"""Deduplicate and organize these findings from analyzing {file_path}:

{aggregated}

Remove duplicates (same line, same issue). Keep all unique findings with their line numbers."""

                return llm_query(dedup_prompt, max_tokens=4000)

            return aggregated

        return analyze_file


def create_repl_analyzers(
    state: "REPLState",
    llm_query_func: Callable[[str, int], str],
    extract_with_lines_func: Callable[[str, int], str]
) -> REPLAnalyzers:
    """Factory function to create REPL analyzers."""
    return REPLAnalyzers(state, llm_query_func, extract_with_lines_func)
