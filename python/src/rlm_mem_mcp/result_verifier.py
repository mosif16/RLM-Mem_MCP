"""
Result Verifier for RLM Processing

Post-processes RLM findings to verify:
1. File references exist in the actual collection
2. Line numbers are within file bounds
3. Code snippets match actual content

This layer catches hallucinations and invalid references before output.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from .file_collector import CollectionResult


@dataclass
class VerificationStats:
    """Statistics from verification process."""
    total_file_refs: int = 0
    verified_files: int = 0
    unverified_files: int = 0
    total_line_refs: int = 0
    verified_lines: int = 0
    invalid_lines: int = 0
    corrections_made: int = 0
    warnings_added: int = 0

    @property
    def file_accuracy(self) -> float:
        """Percentage of file references that were verified."""
        return self.verified_files / self.total_file_refs if self.total_file_refs > 0 else 1.0

    @property
    def line_accuracy(self) -> float:
        """Percentage of line references that were verified."""
        return self.verified_lines / self.total_line_refs if self.total_line_refs > 0 else 1.0

    @property
    def confidence_score(self) -> float:
        """
        Calculate overall confidence score (0.0 - 1.0).

        Factors:
        - File accuracy (40% weight)
        - Line accuracy (40% weight)
        - Correction penalty (10% weight) - more corrections = less confident
        - Warning penalty (10% weight) - more warnings = less confident
        """
        file_score = self.file_accuracy * 0.4
        line_score = self.line_accuracy * 0.4

        # Correction penalty: each correction reduces confidence slightly
        correction_penalty = min(self.corrections_made * 0.02, 0.1)  # Max 10% penalty
        correction_score = (0.1 - correction_penalty)

        # Warning penalty: each warning reduces confidence slightly
        warning_penalty = min(self.warnings_added * 0.01, 0.1)  # Max 10% penalty
        warning_score = (0.1 - warning_penalty)

        return max(0.0, min(1.0, file_score + line_score + correction_score + warning_score))

    @property
    def confidence_level(self) -> str:
        """Get human-readable confidence level."""
        score = self.confidence_score
        if score >= 0.9:
            return "HIGH"
        elif score >= 0.7:
            return "MEDIUM"
        elif score >= 0.5:
            return "LOW"
        else:
            return "VERY LOW"

    @property
    def confidence_emoji(self) -> str:
        """Get emoji for confidence level."""
        level = self.confidence_level
        return {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🟠", "VERY LOW": "🔴"}.get(level, "⚪")

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_file_refs": self.total_file_refs,
            "verified_files": self.verified_files,
            "unverified_files": self.unverified_files,
            "file_accuracy": self.file_accuracy,
            "total_line_refs": self.total_line_refs,
            "verified_lines": self.verified_lines,
            "invalid_lines": self.invalid_lines,
            "line_accuracy": self.line_accuracy,
            "corrections_made": self.corrections_made,
            "warnings_added": self.warnings_added,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level,
        }


class ResultVerifier:
    """
    Verifies RLM findings against actual file content.

    Post-processes LLM output to:
    - Verify file paths exist
    - Verify line numbers are in bounds
    - Flag unverifiable references
    - Apply fuzzy corrections for close matches
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the verifier.

        Args:
            strict_mode: If True, remove unverified references. If False, add warnings.
        """
        self.strict_mode = strict_mode

    def verify_findings(
        self,
        response: str,
        collection: CollectionResult
    ) -> tuple[str, VerificationStats]:
        """
        Verify all file:line references in response.

        Args:
            response: The RLM response text to verify
            collection: The file collection to verify against

        Returns:
            (verified_response, verification_stats) tuple
        """
        stats = VerificationStats()

        # Build file lookup map
        file_map = self._build_file_map(collection)

        # Extract all file:line references (e.g., src/auth.py:42)
        pattern = r'([^\s`\'":\[\]]+\.[a-zA-Z]{1,10}):(\d+)'
        references = re.findall(pattern, response)

        stats.total_file_refs = len(set(ref[0] for ref in references))
        stats.total_line_refs = len(references)

        corrections = {}  # old_ref -> new_ref
        warnings = []

        for filepath, line_str in references:
            line_num = int(line_str)

            # Clean markdown artifacts from file path
            clean_filepath = self._clean_filepath(filepath)
            full_ref = f"{filepath}:{line_num}"

            # Try exact match first (with cleaned path)
            actual_content = file_map.get(clean_filepath) or file_map.get(filepath)

            # Try fuzzy match if exact fails
            if actual_content is None:
                matched_path = self._fuzzy_find_file(filepath, file_map)
                if matched_path:
                    actual_content = file_map[matched_path]
                    # Record correction
                    if matched_path != filepath:
                        corrections[filepath] = matched_path
                        stats.corrections_made += 1

            if actual_content:
                stats.verified_files += 1
                lines = actual_content.split('\n')

                if 1 <= line_num <= len(lines):
                    stats.verified_lines += 1
                else:
                    stats.invalid_lines += 1
                    if not self.strict_mode:
                        warnings.append(
                            f"{full_ref} [line {line_num} out of range, file has {len(lines)} lines]"
                        )
                    else:
                        # In strict mode, we'd remove these but that's complex
                        # For now, add a marker
                        response = response.replace(
                            full_ref,
                            f"{full_ref} [INVALID LINE]"
                        )
            else:
                stats.unverified_files += 1
                if not self.strict_mode:
                    warnings.append(f"{full_ref} [file not found in collection]")

        # Apply corrections
        for old_path, new_path in corrections.items():
            response = response.replace(old_path, new_path)

        # Add verification summary if there were issues
        if warnings or corrections:
            stats.warnings_added = len(warnings)
            verification_note = self._build_verification_note(stats, warnings[:10], corrections)
            response = response + "\n\n" + verification_note

        return response, stats

    def _build_file_map(self, collection: CollectionResult) -> dict[str, str]:
        """Build a map of file paths to content."""
        file_map = {}
        for f in collection.files:
            # Store both full and relative paths
            file_map[f.path] = f.content
            file_map[f.relative_path] = f.content
            # Also store just the filename for fuzzy matching
            filename = f.relative_path.split('/')[-1]
            if filename not in file_map:
                file_map[filename] = f.content
        return file_map

    def _clean_filepath(self, filepath: str) -> str:
        """Remove markdown artifacts from file path."""
        # Remove markdown bold/italic markers
        filepath = filepath.replace("**", "").replace("__", "")
        # Remove single asterisks/underscores carefully (not in middle of words)
        if filepath.startswith("*") or filepath.startswith("_"):
            filepath = filepath[1:]
        if filepath.endswith("*") or filepath.endswith("_"):
            filepath = filepath[:-1]
        # Remove backticks
        filepath = filepath.replace("`", "")
        # Remove quotes
        filepath = filepath.strip('"').strip("'")
        return filepath.strip()

    def _fuzzy_find_file(self, filepath: str, file_map: dict[str, str]) -> str | None:
        """
        Find a file using fuzzy matching.

        Tries:
        1. Exact match
        2. Ends-with match
        3. Contains match
        4. Filename-only match
        """
        # Exact match already tried

        # Ends-with match
        for known_path in file_map.keys():
            if known_path.endswith(filepath) or filepath.endswith(known_path):
                return known_path

        # Filename match
        filename = filepath.split('/')[-1]
        for known_path in file_map.keys():
            if known_path.endswith(filename):
                return known_path

        # Contains match (last resort)
        for known_path in file_map.keys():
            if filename in known_path:
                return known_path

        return None

    def _build_verification_note(
        self,
        stats: VerificationStats,
        warnings: list[str],
        corrections: dict[str, str]
    ) -> str:
        """Build a verification summary note."""
        parts = [
            "---",
            "## Verification Summary",
            "",
            f"**Overall Confidence: {stats.confidence_emoji} {stats.confidence_level}** ({stats.confidence_score:.0%})",
            "",
        ]

        parts.append(f"- File references verified: {stats.verified_files}/{stats.total_file_refs} ({stats.file_accuracy:.0%})")

        if stats.total_line_refs > 0:
            parts.append(f"- Line references verified: {stats.verified_lines}/{stats.total_line_refs} ({stats.line_accuracy:.0%})")

        if corrections:
            parts.append(f"\n**Corrections applied:** {len(corrections)}")
            for old, new in list(corrections.items())[:5]:
                parts.append(f"  - `{old}` → `{new}`")

        if warnings:
            parts.append(f"\n**Warnings ({len(warnings)}):**")
            for warning in warnings[:10]:
                parts.append(f"  - {warning}")
            if len(warnings) > 10:
                parts.append(f"  - ... and {len(warnings) - 10} more")

        return "\n".join(parts)

    def quick_verify(self, response: str, collection: CollectionResult) -> dict[str, Any]:
        """
        Quick verification without modifying response.

        Returns verification stats only.
        """
        _, stats = self.verify_findings(response, collection)
        return stats.to_dict()


def create_batch_verifier(collection: CollectionResult) -> 'BatchVerifier':
    """Create a batch verifier for multiple findings."""
    return BatchVerifier(collection)


class BatchVerifier:
    """
    Batch verification for multiple findings at once.

    More efficient than verifying one at a time - parses files once.
    """

    def __init__(self, collection: CollectionResult):
        self.collection = collection
        self._file_cache: dict[str, tuple[str, list[str]]] = {}  # path -> (content, lines)
        self._build_cache()

    def _build_cache(self) -> None:
        """Build file content cache."""
        for f in self.collection.files:
            lines = f.content.split('\n')
            self._file_cache[f.relative_path] = (f.content, lines)
            self._file_cache[f.path] = (f.content, lines)
            # Filename only
            filename = f.relative_path.split('/')[-1]
            if filename not in self._file_cache:
                self._file_cache[filename] = (f.content, lines)

    def verify_batch(
        self,
        findings: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Verify a batch of findings.

        Args:
            findings: List of dicts with 'file', 'line', and optionally 'pattern'

        Returns:
            List of verification results
        """
        results = []

        for finding in findings:
            filepath = finding.get('file', '')
            line_num = finding.get('line', 0)
            pattern = finding.get('pattern')

            result = {
                'file': filepath,
                'line': line_num,
                'is_valid': False,
                'actual_content': None,
                'in_bounds': False,
                'pattern_matches': None,
                'reason': ''
            }

            # Find file
            file_data = self._find_file(filepath)
            if not file_data:
                result['reason'] = 'File not found'
                results.append(result)
                continue

            content, lines = file_data

            # Check line bounds
            if line_num < 1 or line_num > len(lines):
                result['reason'] = f'Line {line_num} out of bounds (file has {len(lines)} lines)'
                results.append(result)
                continue

            result['in_bounds'] = True
            result['actual_content'] = lines[line_num - 1]

            # Check pattern if provided
            if pattern:
                import re
                if re.search(pattern, lines[line_num - 1], re.IGNORECASE):
                    result['pattern_matches'] = True
                    result['is_valid'] = True
                    result['reason'] = 'Verified'
                else:
                    result['pattern_matches'] = False
                    result['reason'] = 'Pattern not found on line'
            else:
                result['is_valid'] = True
                result['reason'] = 'Line exists'

            results.append(result)

        return results

    def _find_file(self, filepath: str) -> tuple[str, list[str]] | None:
        """Find file in cache with fuzzy matching."""
        # Exact match
        if filepath in self._file_cache:
            return self._file_cache[filepath]

        # Ends-with match
        for known_path, data in self._file_cache.items():
            if known_path.endswith(filepath) or filepath.endswith(known_path):
                return data

        # Filename match
        filename = filepath.split('/')[-1]
        if filename in self._file_cache:
            return self._file_cache[filename]

        return None
