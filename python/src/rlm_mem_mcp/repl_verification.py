"""
REPL Verification helpers for validating findings.

Provides verification functions that help validate code analysis results
for accuracy and confidence.
"""

import re
from typing import Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .repl_state import REPLState

from .repl_security import strip_markdown_fences_from_content
from .content_analyzer import (
    find_dead_code_regions,
    is_line_in_dead_code,
    verify_line_reference,
    check_implementation_status,
    DeadCodeRegion,
)
from .result_verifier import (
    QueryVerificationResult,
    verify_query_results,
)


class REPLVerification:
    """Factory for creating REPL verification helper functions."""

    def __init__(self, state: "REPLState"):
        self.state = state

    def _extract_file_list(self) -> list[str]:
        """Extract list of actual file paths from the prompt content."""
        files = re.findall(r'### File: ([^\n]+)', self.state.prompt)
        return files

    def create_verify_line_function(self) -> Callable[[str, int, str | None], dict]:
        """Create the verify_line helper function for line verification."""
        state = self.state

        def verify_line(filepath: str, line_num: int, expected_pattern: str | None = None) -> dict:
            """
            Verify that a line number reference is valid and contains expected content.

            Args:
                filepath: Path to the file
                line_num: Line number to verify (1-indexed)
                expected_pattern: Optional regex pattern expected on that line

            Returns:
                dict with 'is_valid', 'actual_content', 'in_dead_code', 'confidence', 'reason'
            """
            # Find the file content
            content = None
            actual_filepath = None

            parts = state.prompt.split("### File:")
            for part in parts[1:]:
                lines = part.split("\n")
                if not lines:
                    continue
                file_path = lines[0].strip()
                if filepath in file_path or file_path.endswith(filepath):
                    # Strip markdown fences from content
                    content_lines = strip_markdown_fences_from_content(lines[1:])
                    content = "\n".join(content_lines)
                    actual_filepath = file_path
                    break

            if content is None:
                return {
                    'is_valid': False,
                    'actual_content': None,
                    'in_dead_code': False,
                    'confidence': 'LOW',
                    'reason': f"File not found: {filepath}"
                }

            # Get or compute dead code regions for this file
            if actual_filepath not in state.dead_code_regions:
                state.dead_code_regions[actual_filepath] = find_dead_code_regions(content, actual_filepath)

            # Use the content analyzer
            result = verify_line_reference(
                content,
                actual_filepath,
                line_num,
                expected_pattern,
                state.dead_code_regions[actual_filepath]
            )

            return {
                'is_valid': result.is_valid,
                'actual_content': result.actual_content,
                'in_dead_code': result.in_dead_code,
                'confidence': result.confidence.value,
                'reason': result.reason
            }

        return verify_line

    def create_check_dead_code_function(self) -> Callable[[str], list[dict]]:
        """Create the check_dead_code helper function for dead code detection."""
        state = self.state

        def check_dead_code(filepath: str) -> list[dict]:
            """
            Check if a file has dead code regions (#if false, #if DEBUG, etc.)

            Args:
                filepath: Path to the file

            Returns:
                List of dead code regions with start_line, end_line, condition
            """
            # Find the file content
            content = None
            actual_filepath = None

            parts = state.prompt.split("### File:")
            for part in parts[1:]:
                lines = part.split("\n")
                if not lines:
                    continue
                file_path = lines[0].strip()
                if filepath in file_path or file_path.endswith(filepath):
                    # Strip markdown fences from content
                    content_lines = strip_markdown_fences_from_content(lines[1:])
                    content = "\n".join(content_lines)
                    actual_filepath = file_path
                    break

            if content is None:
                return []

            # Get or compute dead code regions
            if actual_filepath not in state.dead_code_regions:
                state.dead_code_regions[actual_filepath] = find_dead_code_regions(content, actual_filepath)

            # Convert to dicts for REPL
            return [
                {
                    'start_line': r.start_line,
                    'end_line': r.end_line,
                    'condition': r.condition,
                    'language': r.language
                }
                for r in state.dead_code_regions[actual_filepath]
            ]

        return check_dead_code

    def create_is_implemented_function(self) -> Callable[[str, str], dict]:
        """Create the is_implemented helper function for checking function implementation."""
        state = self.state

        def is_implemented(filepath: str, function_name: str) -> dict:
            """
            Check if a function is actually implemented (not just a stub).

            Args:
                filepath: Path to the file
                function_name: Name of function to check

            Returns:
                dict with 'is_implemented', 'has_body', 'is_stub', 'body_lines', 'confidence', 'reason'
            """
            # Find the file content
            content = None
            actual_filepath = None

            parts = state.prompt.split("### File:")
            for part in parts[1:]:
                lines = part.split("\n")
                if not lines:
                    continue
                file_path = lines[0].strip()
                if filepath in file_path or file_path.endswith(filepath):
                    # Strip markdown fences from content
                    content_lines = strip_markdown_fences_from_content(lines[1:])
                    content = "\n".join(content_lines)
                    actual_filepath = file_path
                    break

            if content is None:
                return {
                    'is_implemented': False,
                    'has_body': False,
                    'is_stub': False,
                    'body_lines': 0,
                    'confidence': 'LOW',
                    'reason': f"File not found: {filepath}"
                }

            # Use the content analyzer
            status = check_implementation_status(content, function_name, actual_filepath)

            return {
                'is_implemented': status.is_implemented,
                'has_body': status.has_body,
                'is_stub': status.is_stub,
                'body_lines': status.body_lines,
                'confidence': status.confidence.value,
                'reason': status.reason
            }

        return is_implemented

    def create_batch_verify_function(self) -> Callable[[list[dict]], list[dict]]:
        """Create the batch_verify helper function for efficient multi-finding verification."""
        state = self.state

        def batch_verify(findings: list[dict]) -> list[dict]:
            """
            Verify multiple findings at once (more efficient than one-by-one).

            Args:
                findings: List of dicts with 'file', 'line', and optionally 'pattern'

            Returns:
                List of verification results with 'is_valid', 'actual_content', 'in_dead_code', 'confidence', 'reason'
            """
            results = []

            # Group by file for efficiency
            from collections import defaultdict
            by_file = defaultdict(list)
            for i, f in enumerate(findings):
                by_file[f.get('file', '')].append((i, f))

            # Process each file once
            parts = state.prompt.split("### File:")

            for filepath, file_findings in by_file.items():
                # Find file content
                content = None
                actual_filepath = None

                for part in parts[1:]:
                    lines = part.split("\n")
                    if not lines:
                        continue
                    file_path = lines[0].strip()
                    if filepath in file_path or file_path.endswith(filepath):
                        # Strip markdown fences from content
                        content_lines_raw = strip_markdown_fences_from_content(lines[1:])
                        content = "\n".join(content_lines_raw)
                        actual_filepath = file_path
                        break

                if content is None:
                    # File not found - mark all findings for this file as invalid
                    for idx, finding in file_findings:
                        results.append({
                            'index': idx,
                            'is_valid': False,
                            'actual_content': None,
                            'in_dead_code': False,
                            'confidence': 'LOW',
                            'reason': f"File not found: {filepath}"
                        })
                    continue

                # Get dead code regions once per file
                if actual_filepath not in state.dead_code_regions:
                    state.dead_code_regions[actual_filepath] = find_dead_code_regions(content, actual_filepath)

                dead_regions = state.dead_code_regions[actual_filepath]
                content_lines = content.split('\n')

                # Verify each finding in this file
                for idx, finding in file_findings:
                    line_num = finding.get('line', 0)
                    pattern = finding.get('pattern')

                    if line_num < 1 or line_num > len(content_lines):
                        results.append({
                            'index': idx,
                            'is_valid': False,
                            'actual_content': None,
                            'in_dead_code': False,
                            'confidence': 'LOW',
                            'reason': f"Line {line_num} out of range (file has {len(content_lines)} lines)"
                        })
                        continue

                    actual_content = content_lines[line_num - 1]
                    in_dead_code, condition = is_line_in_dead_code(line_num, dead_regions)

                    # Check pattern if provided
                    pattern_matches = True
                    if pattern:
                        pattern_matches = bool(re.search(pattern, actual_content, re.IGNORECASE))

                    # Determine confidence
                    if in_dead_code:
                        confidence = 'LOW'
                        reason = f"Line is in dead code block ({condition})"
                    elif not pattern_matches and pattern:
                        confidence = 'LOW'
                        reason = "Pattern not found on line"
                    else:
                        confidence = 'HIGH'
                        reason = "Verified in active code"

                    results.append({
                        'index': idx,
                        'is_valid': pattern_matches,
                        'actual_content': actual_content,
                        'in_dead_code': in_dead_code,
                        'confidence': confidence,
                        'reason': reason
                    })

            # Sort by original index
            results.sort(key=lambda x: x.get('index', 0))
            return results

        return batch_verify

    def create_verify_results_function(self) -> Callable[[str, str], QueryVerificationResult]:
        """Create the verify_results function for the REPL guardrail."""
        content = self.state.prompt
        files = self._extract_file_list()

        def verify_results(query: str, results: str) -> QueryVerificationResult:
            """
            Verify results before setting FINAL_ANSWER.

            This is the GUARDRAIL that ensures results are:
            1. Aligned with query intent
            2. Specific (file:line, code, confidence)

            Args:
                query: The original search query
                results: The results to verify (from tool.to_markdown())

            Returns:
                QueryVerificationResult with status, guidance

            Example:
                result = find_secrets()
                verification = verify_results("find secrets", result.to_markdown())
                if verification.status.value != "FAILED":
                    FINAL_ANSWER = result.to_markdown()
                else:
                    print(verification.guidance)  # Fix issues and retry
            """
            return verify_query_results(query, results, content, files)

        return verify_results


def create_repl_verification(state: "REPLState") -> REPLVerification:
    """Factory function to create REPL verification helpers."""
    return REPLVerification(state)
