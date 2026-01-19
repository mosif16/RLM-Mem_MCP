"""
Result Verifier for RLM Processing

This module implements a 3-layer verification system:

Layer 1: POST-PROCESSING VERIFICATION (existing)
- File references exist in the actual collection
- Line numbers are within file bounds
- Code snippets match actual content

Layer 2: QUERY-RESULT ALIGNMENT (NEW)
- Did the search actually find what was asked?
- Does the intent match the results?

Layer 3: SPECIFICITY REQUIREMENTS (NEW)
- Are findings concrete with file:line, code, confidence?
- Fail-fast with guidance if requirements not met

This acts as a GUARDRAIL - results must pass verification before returning.
"""

import re
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from .file_collector import CollectionResult


# ============================================================================
# LAYER 2 & 3: QUERY-RESULT ALIGNMENT & SPECIFICITY VERIFICATION
# ============================================================================

class VerificationStatus(Enum):
    """Status of verification check."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"


@dataclass
class AlignmentCheck:
    """Result of checking query-result alignment."""
    query_intent: str
    results_address_intent: bool
    confidence: float  # 0.0 to 1.0
    explanation: str
    missing_aspects: list[str] = field(default_factory=list)


@dataclass
class SpecificityCheck:
    """Result of checking a single specificity requirement."""
    requirement: str
    passed: bool
    details: str
    examples: list[str] = field(default_factory=list)


@dataclass
class QueryVerificationResult:
    """Complete verification result for query-result alignment."""
    status: VerificationStatus
    alignment: AlignmentCheck
    specificity_checks: list[SpecificityCheck]
    overall_score: float  # 0.0 to 1.0
    guidance: str
    improved_query_suggestion: str = ""

    def to_markdown(self) -> str:
        """Format verification result as markdown."""
        status_emoji = {
            VerificationStatus.PASSED: "✅",
            VerificationStatus.FAILED: "❌",
            VerificationStatus.WARNING: "⚠️"
        }

        lines = [
            f"## Verification {status_emoji[self.status]} {self.status.value}",
            "",
            f"**Overall Score:** {self.overall_score:.0%}",
            "",
            "### Query Alignment",
            f"- **Intent:** {self.alignment.query_intent}",
            f"- **Addressed:** {'Yes' if self.alignment.results_address_intent else 'No'} ({self.alignment.confidence:.0%} confidence)",
            f"- **Explanation:** {self.alignment.explanation}",
        ]

        if self.alignment.missing_aspects:
            lines.append(f"- **Missing:** {', '.join(self.alignment.missing_aspects)}")

        lines.extend(["", "### Specificity Checks"])

        for check in self.specificity_checks:
            status = "✅" if check.passed else "❌"
            lines.append(f"- {status} **{check.requirement}:** {check.details}")
            if not check.passed and check.examples:
                lines.append(f"  - Issues: {', '.join(check.examples[:3])}")

        if self.guidance:
            lines.extend(["", "### Guidance", self.guidance])

        if self.improved_query_suggestion:
            lines.extend(["", "### Suggested Query", f"```", self.improved_query_suggestion, "```"])

        return "\n".join(lines)

    def should_fail_fast(self) -> bool:
        """Determine if we should fail fast based on verification."""
        if self.status == VerificationStatus.FAILED:
            return True
        # Also fail if critical specificity checks failed
        critical = ["File References", "Code Snippets"]
        for check in self.specificity_checks:
            if check.requirement in critical and not check.passed:
                return True
        return False


class QueryResultVerifier:
    """
    Verification agent that validates RLM search results BEFORE returning.

    This is the NEW guardrail layer that ensures:
    1. Results align with query intent
    2. Results meet specificity requirements (file:line, code, confidence)
    """

    # Keywords that indicate different query intents
    INTENT_KEYWORDS = {
        "security": ["security", "vulnerability", "exploit", "injection", "xss", "secret",
                     "password", "token", "auth", "credential", "sensitive"],
        "ios": ["swift", "ios", "swiftui", "uikit", "force unwrap", "retain cycle",
                "weak self", "@mainactor", "cloudkit", "swiftdata"],
        "quality": ["quality", "refactor", "long function", "complexity", "todo",
                    "fixme", "code smell", "dead code"],
        "architecture": ["architecture", "structure", "module", "dependency", "import",
                         "layer", "pattern", "design"],
        "performance": ["performance", "memory", "leak", "slow", "optimize", "cache",
                        "latency", "bottleneck"],
        "crash": ["crash", "force unwrap", "nil", "fatal", "exception", "error handling"],
        "persistence": ["persist", "storage", "localstorage", "userdefaults", "coredata",
                        "state", "save", "load", "cache", "database", "realm"],
        "state": ["state", "mutation", "redux", "observable", "binding", "@state",
                  "@published", "viewmodel"],
    }

    def __init__(self, content: str, files: list[str]):
        """
        Initialize verifier with codebase context.

        Args:
            content: The full prompt content
            files: List of actual file paths in the codebase
        """
        self.content = content
        self.files = set(files)
        self.file_line_counts = self._compute_file_line_counts()

    def _compute_file_line_counts(self) -> dict[str, int]:
        """Compute line counts for each file."""
        counts = {}
        parts = self.content.split("### File:")
        for part in parts[1:]:
            lines = part.split("\n")
            if lines:
                filepath = lines[0].strip()
                counts[filepath] = len(lines) - 1
        return counts

    def _detect_query_intent(self, query: str) -> tuple[str, list[str]]:
        """Detect the intent of the query."""
        query_lower = query.lower()
        detected_intents = []

        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                detected_intents.append(intent)

        if not detected_intents:
            return "general", ["relevant findings"]

        primary_intent = detected_intents[0]

        # Map intent to expected finding types
        expected_findings = {
            "security": ["vulnerabilities", "secrets", "injection points", "auth issues"],
            "ios": ["force unwraps", "retain cycles", "thread issues", "SwiftUI problems"],
            "quality": ["long functions", "code smells", "TODOs", "complexity issues"],
            "architecture": ["module structure", "dependencies", "layering issues"],
            "performance": ["memory issues", "slow operations", "inefficiencies"],
            "crash": ["force unwraps", "nil handling", "fatal errors"],
            "persistence": ["storage patterns", "state persistence", "data flow"],
            "state": ["state mutations", "bindings", "observable patterns"],
        }

        return primary_intent, expected_findings.get(primary_intent, ["relevant findings"])

    def _check_alignment(self, query: str, results: str) -> AlignmentCheck:
        """Check if results align with query intent."""
        primary_intent, expected_findings = self._detect_query_intent(query)
        results_lower = results.lower()

        found_expected = []
        missing_expected = []

        for expected in expected_findings:
            terms = expected.lower().replace(" ", "_").split("_")
            if any(term in results_lower for term in terms):
                found_expected.append(expected)
            else:
                missing_expected.append(expected)

        # Check for "no findings"
        no_findings_indicators = [
            "no findings", "no issues", "nothing found", "no relevant",
            "0 findings", "clean", "no vulnerabilities"
        ]
        is_no_findings = any(ind in results_lower for ind in no_findings_indicators)

        # Check for vague responses
        vague_indicators = [
            "could potentially", "might be", "possibly", "may have",
            "consider checking", "generally speaking", "in theory"
        ]
        is_vague = sum(1 for v in vague_indicators if v in results_lower) >= 2

        if is_no_findings:
            confidence = 0.7
            addresses_intent = True
            explanation = "Search completed with no findings - verify search was comprehensive"
        elif is_vague:
            confidence = 0.3
            addresses_intent = False
            explanation = "Results are too vague and lack specific findings"
        elif not found_expected:
            confidence = 0.4
            addresses_intent = False
            explanation = f"Results don't address the {primary_intent} intent"
        else:
            coverage = len(found_expected) / max(len(expected_findings), 1)
            confidence = 0.6 + (coverage * 0.4)
            addresses_intent = confidence >= 0.6
            explanation = f"Results address {len(found_expected)}/{len(expected_findings)} expected types"

        return AlignmentCheck(
            query_intent=primary_intent,
            results_address_intent=addresses_intent,
            confidence=confidence,
            explanation=explanation,
            missing_aspects=missing_expected if not addresses_intent else []
        )

    def _extract_file_references(self, results: str) -> list[tuple[str, int]]:
        """Extract file:line references from results."""
        references = []
        patterns = [
            r'([a-zA-Z0-9_/\-\.]+\.[a-zA-Z]{1,5}):(\d+)',
            r'\*\*([a-zA-Z0-9_/\-\.]+\.[a-zA-Z]{1,5}):(\d+)\*\*',
            r'`([a-zA-Z0-9_/\-\.]+\.[a-zA-Z]{1,5}):(\d+)`',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, results)
            for filepath, line_str in matches:
                try:
                    references.append((filepath, int(line_str)))
                except ValueError:
                    continue
        return references

    def _extract_code_snippets(self, results: str) -> list[str]:
        """Extract code snippets from results."""
        snippets = []
        code_block_pattern = r'```[a-z]*\n(.*?)\n```'
        snippets.extend(re.findall(code_block_pattern, results, re.DOTALL))

        inline_pattern = r'`([^`]{10,})`'
        inline_matches = re.findall(inline_pattern, results)
        snippets.extend([m for m in inline_matches if any(c in m for c in ['(', '{', '=', '.'])])

        return snippets

    def _check_specificity(self, results: str) -> list[SpecificityCheck]:
        """Check all specificity requirements."""
        checks = []

        file_refs = self._extract_file_references(results)

        # 1. File references
        has_files = len(file_refs) > 0
        checks.append(SpecificityCheck(
            requirement="File References",
            passed=has_files,
            details=f"Found {len(file_refs)} file:line references" if has_files else "No file references",
            examples=[] if has_files else ["Use 'filename.swift:42' format"]
        ))

        # 2. Line numbers
        has_lines = any(line > 0 for _, line in file_refs)
        checks.append(SpecificityCheck(
            requirement="Line Numbers",
            passed=has_lines,
            details=f"Line numbers in {len([r for r in file_refs if r[1] > 0])} refs" if has_lines else "No line numbers",
            examples=[] if has_lines else ["Specify exact line numbers"]
        ))

        # 3. Code snippets
        snippets = self._extract_code_snippets(results)
        has_snippets = len(snippets) > 0
        checks.append(SpecificityCheck(
            requirement="Code Snippets",
            passed=has_snippets,
            details=f"Found {len(snippets)} code snippets" if has_snippets else "No code snippets",
            examples=[] if has_snippets else ["Include actual code showing the issue"]
        ))

        # 4. Confidence levels
        confidence_pattern = r'\b(HIGH|MEDIUM|LOW)\b.*confidence|\[.*?(HIGH|MEDIUM|LOW).*?\]'
        has_confidence = bool(re.search(confidence_pattern, results, re.IGNORECASE))
        checks.append(SpecificityCheck(
            requirement="Confidence Levels",
            passed=has_confidence,
            details="Confidence levels indicated" if has_confidence else "No confidence levels",
            examples=[] if has_confidence else ["Mark as [HIGH], [MEDIUM], or [LOW]"]
        ))

        # 5. Files exist
        invalid_files = []
        for filepath, _ in file_refs:
            exists = any(filepath in f or f.endswith(filepath) for f in self.files)
            if not exists:
                invalid_files.append(filepath)

        files_valid = len(invalid_files) == 0 or len(file_refs) == 0
        checks.append(SpecificityCheck(
            requirement="Files Exist",
            passed=files_valid,
            details="All files exist" if files_valid else f"{len(invalid_files)} not found",
            examples=invalid_files[:3]
        ))

        # 6. Lines valid
        invalid_lines = []
        for filepath, line_num in file_refs:
            for actual_file in self.files:
                if filepath in actual_file or actual_file.endswith(filepath):
                    max_lines = self.file_line_counts.get(actual_file, 0)
                    if line_num > max_lines > 0:
                        invalid_lines.append(f"{filepath}:{line_num}")
                    break

        lines_valid = len(invalid_lines) == 0
        checks.append(SpecificityCheck(
            requirement="Lines Valid",
            passed=lines_valid,
            details="All lines valid" if lines_valid else f"{len(invalid_lines)} invalid",
            examples=invalid_lines[:3]
        ))

        return checks

    def _generate_guidance(
        self, alignment: AlignmentCheck, specificity_checks: list[SpecificityCheck], query: str
    ) -> tuple[str, str]:
        """Generate guidance and improved query suggestion."""
        issues = []
        suggestions = []

        if not alignment.results_address_intent:
            issues.append(f"Results don't address '{alignment.query_intent}' intent")
            if alignment.missing_aspects:
                suggestions.append(f"Search for: {', '.join(alignment.missing_aspects[:2])}")

        failed_checks = [c for c in specificity_checks if not c.passed]
        for check in failed_checks:
            issues.append(f"{check.requirement}: {check.details}")

        if not any(c.passed for c in specificity_checks if c.requirement == "File References"):
            suggestions.append("Use structured tools like find_secrets() that return file:line")

        if not any(c.passed for c in specificity_checks if c.requirement == "Code Snippets"):
            suggestions.append("Include actual code demonstrating each finding")

        if not any(c.passed for c in specificity_checks if c.requirement == "Confidence Levels"):
            suggestions.append("Use result.to_markdown() for automatic confidence levels")

        guidance = ""
        if issues:
            guidance = "**Issues:**\n" + "\n".join(f"- {i}" for i in issues)
            if suggestions:
                guidance += "\n\n**Suggestions:**\n" + "\n".join(f"- {s}" for s in suggestions)
        else:
            guidance = "Results meet all verification requirements."

        improved_query = ""
        if failed_checks:
            improved_query = f"{query}\n\nREQUIRED for each finding:\n"
            improved_query += "1. File path and line (file.swift:42)\n"
            improved_query += "2. Actual code snippet\n"
            improved_query += "3. Confidence (HIGH/MEDIUM/LOW)\n"
            improved_query += "4. Specific fix"

        return guidance, improved_query

    def verify(self, query: str, results: str) -> QueryVerificationResult:
        """
        Verify that results meet quality standards.

        This is called BEFORE returning FINAL_ANSWER.
        If verification fails, returns guidance on what to fix.
        """
        if not results or not results.strip():
            return QueryVerificationResult(
                status=VerificationStatus.FAILED,
                alignment=AlignmentCheck("unknown", False, 0.0, "No results", ["any findings"]),
                specificity_checks=[SpecificityCheck("Results Present", False, "Empty", ["Run search first"])],
                overall_score=0.0,
                guidance="No results provided. Run a search tool:\n```python\nresult = find_secrets()\nFINAL_ANSWER = result.to_markdown()\n```",
                improved_query_suggestion=query
            )

        alignment = self._check_alignment(query, results)
        specificity_checks = self._check_specificity(results)

        alignment_score = alignment.confidence if alignment.results_address_intent else alignment.confidence * 0.5
        specificity_score = sum(1 for c in specificity_checks if c.passed) / len(specificity_checks)
        overall_score = (alignment_score * 0.4) + (specificity_score * 0.6)

        critical_checks = ["File References", "Code Snippets"]
        critical_failed = any(not c.passed for c in specificity_checks if c.requirement in critical_checks)

        if overall_score >= 0.7 and not critical_failed:
            status = VerificationStatus.PASSED
        elif overall_score >= 0.5 and alignment.results_address_intent:
            status = VerificationStatus.WARNING
        else:
            status = VerificationStatus.FAILED

        guidance, improved_query = self._generate_guidance(alignment, specificity_checks, query)

        return QueryVerificationResult(
            status=status,
            alignment=alignment,
            specificity_checks=specificity_checks,
            overall_score=overall_score,
            guidance=guidance,
            improved_query_suggestion=improved_query if status == VerificationStatus.FAILED else ""
        )


def verify_query_results(query: str, results: str, content: str, files: list[str]) -> QueryVerificationResult:
    """
    Convenience function to verify results before returning.

    This is the main entry point for the REPL guardrail.

    Example:
        >>> verification = verify_query_results(query, results, prompt, files)
        >>> if verification.should_fail_fast():
        ...     print(verification.guidance)
        ...     # Re-run with better approach
        >>> else:
        ...     FINAL_ANSWER = results
    """
    verifier = QueryResultVerifier(content, files)
    return verifier.verify(query, results)


# ============================================================================
# LAYER 1: POST-PROCESSING VERIFICATION (existing functionality)
# ============================================================================


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
