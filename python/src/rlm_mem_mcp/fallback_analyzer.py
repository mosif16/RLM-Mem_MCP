"""
Fallback Analyzer for RLM Processing

When the REPL code execution fails (syntax errors, sandbox restrictions, etc.),
this provides a graceful degradation to pattern-based analysis.

This uses simple regex patterns to find common issues without LLM code execution.
It's less powerful than full RLM but always produces results.
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FallbackFinding:
    """A finding from fallback analysis."""
    category: str
    file_path: str
    line_number: int
    line_content: str
    severity: str  # "high", "medium", "low"
    description: str


@dataclass
class FallbackResult:
    """Result from fallback analysis."""
    query: str
    findings: list[FallbackFinding] = field(default_factory=list)
    files_analyzed: int = 0
    patterns_checked: int = 0
    fallback_reason: str = ""

    def to_markdown(self) -> str:
        """Convert to markdown output."""
        parts = [
            "## ⚠️ FALLBACK MODE - Pattern-Based Analysis",
            "",
            "```",
            "╔══════════════════════════════════════════════════════════════╗",
            "║  RLM REPL execution failed - using regex pattern matching    ║",
            f"║  Reason: {self.fallback_reason[:50]:<50} ║",
            "║                                                              ║",
            "║  LIMITATIONS:                                                ║",
            "║  • Surface-level pattern matching only                       ║",
            "║  • No semantic understanding of code flow                    ║",
            "║  • May miss context-dependent issues                         ║",
            "║  • Cannot detect architectural problems                      ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "```",
            "",
            f"**Files analyzed:** {self.files_analyzed}",
            f"**Patterns checked:** {self.patterns_checked}",
            f"**Findings:** {len(self.findings)}",
            "",
        ]

        if not self.findings:
            parts.append("No issues found matching the search patterns.")
            return "\n".join(parts)

        # Group by category
        by_category: dict[str, list[FallbackFinding]] = {}
        for f in self.findings:
            if f.category not in by_category:
                by_category[f.category] = []
            by_category[f.category].append(f)

        for category, findings in by_category.items():
            parts.append(f"### {category} ({len(findings)} findings)")
            parts.append("")

            for f in findings[:20]:  # Limit to 20 per category
                parts.append(f"**{f.file_path}:{f.line_number}** [{f.severity.upper()}]")
                parts.append(f"```")
                parts.append(f.line_content.strip()[:100])
                parts.append(f"```")
                parts.append(f"_{f.description}_")
                parts.append("")

            if len(findings) > 20:
                parts.append(f"... and {len(findings) - 20} more in this category")
                parts.append("")

        parts.append("---")
        parts.append("")
        parts.append("### 💡 For faster results, consider native tools:")
        parts.append("```bash")
        parts.append("# Force unwraps (Swift)")
        parts.append("rg '!\\.' --type swift")
        parts.append("")
        parts.append("# Retain cycles")
        parts.append("rg 'self\\.' --type swift -B2 | rg -v 'weak|unowned'")
        parts.append("")
        parts.append("# Hardcoded secrets")
        parts.append("rg -i '(api.?key|secret|password|token)\\s*=' --type swift")
        parts.append("```")
        parts.append("")
        parts.append("*RLM is best for semantic analysis across large codebases (1M+ lines).*")
        parts.append("*For pattern matching, native grep/ripgrep is faster and more precise.*")

        return "\n".join(parts)


# Pattern categories for common security/quality issues
SECURITY_PATTERNS = {
    "Hardcoded Secrets": [
        (r'(?:api_key|apikey|api-key)\s*[:=]\s*["\'][^"\']{10,}["\']', "high", "Possible hardcoded API key"),
        (r'(?:password|passwd|pwd)\s*[:=]\s*["\'][^"\']+["\']', "high", "Possible hardcoded password"),
        (r'(?:secret|token)\s*[:=]\s*["\'][^"\']{10,}["\']', "high", "Possible hardcoded secret/token"),
        (r'(?:private_key|privatekey)\s*[:=]', "high", "Possible hardcoded private key"),
        (r'sk-[a-zA-Z0-9]{20,}', "high", "OpenAI API key pattern"),
        (r'ghp_[a-zA-Z0-9]{30,}', "high", "GitHub personal access token"),
        (r'xox[baprs]-[a-zA-Z0-9-]+', "high", "Slack token pattern"),
    ],
    "SQL Injection": [
        (r'["\']SELECT\s+.*\s+FROM\s+.*["\']\s*\+', "high", "String concatenation in SQL query"),
        (r'f["\']SELECT\s+.*\{', "high", "F-string in SQL query"),
        (r'execute\s*\(\s*["\'].*\%s', "medium", "Possible SQL with string interpolation"),
        (r'\.format\s*\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE)', "high", "Format string in SQL"),
    ],
    "Command Injection": [
        (r'os\.system\s*\([^)]*\+', "high", "os.system with string concatenation"),
        (r'subprocess\.\w+\s*\([^)]*shell\s*=\s*True', "high", "subprocess with shell=True"),
        (r'exec\s*\([^)]*\+', "high", "exec with string concatenation"),
        (r'eval\s*\([^)]*\+', "high", "eval with string concatenation"),
    ],
    "Path Traversal": [
        (r'open\s*\([^)]*\+', "medium", "open() with string concatenation"),
        (r'\.\./', "low", "Relative path traversal pattern"),
    ],
}

SWIFT_PATTERNS = {
    "Force Unwrap": [
        # CRITICAL: Use (?!=) negative lookahead to exclude != (not-equals)
        # Pattern !(?!=) means: match ! NOT followed by =
        # This prevents false positives like "a != b" being flagged
        (r'\w!(?!=)\s*\.', "medium", "Force unwrap before method call"),
        (r'\w!(?!=)\s*\[', "medium", "Force unwrap before subscript"),
        (r'\)!(?!=)\s*\.', "medium", "Force unwrap on function result"),
        (r'\bas!\s+\w+', "medium", "Force cast"),
        # EXCLUDE try! on compile-time regex patterns (safe and intentional)
        # Only flag try! that's NOT followed by NSRegularExpression, Regex, or #/.../#
        (r'\btry!\s+(?!NSRegularExpression|Regex|#/)', "high", "Force try (excluding compile-time regex)"),
    ],
    "Retain Cycles": [
        # Only flag Combine publishers - these are the real retain cycle risks
        # REMOVED: Task closures - Tasks don't create retain cycles like regular closures
        # REMOVED: Generic closure pattern - too many false positives
        (r'\.sink\s*\{(?!\s*\[weak)', "medium", "Combine sink without [weak self]"),
        (r'\.receive\s*\(on:.*\)\s*\{(?!\s*\[weak)', "medium", "Combine receive without [weak self]"),
        (r'\.store\s*\(in:.*\)(?!.*\[weak)', "low", "Combine store - verify subscription lifecycle"),
    ],
    "Actor Isolation": [
        # Only flag clear concurrency issues, not informational patterns
        # REMOVED: @MainActor async (informational, not an issue)
        # REMOVED: Task with @MainActor (correct usage)
        (r'DispatchQueue\.main\.async(?!.*@MainActor)', "low", "DispatchQueue.main.async - consider @MainActor for modern code"),
        (r'nonisolated.*\bself\.', "medium", "nonisolated accessing self - potential race condition"),
    ],
    "Sendable Issues": [
        # Only flag explicit @unchecked Sendable (code smell) and compiler warnings
        # REMOVED: Generic class pattern - way too broad, most classes don't cross isolation
        (r'@unchecked\s+Sendable', "medium", "@unchecked Sendable - verify thread safety manually"),
        (r'closure.*captures.*non-Sendable', "high", "Closure captures non-Sendable type"),
    ],
    "SwiftUI Lifecycle": [
        # Only flag real bugs, not stylistic concerns
        # REMOVED: @State with initializer - Date(), UUID() etc are perfectly fine
        # REMOVED: .task without weak self - SwiftUI handles cancellation automatically
        (r'\.onAppear\s*\{[^}]*\.onAppear', "medium", "Nested onAppear - may cause multiple calls"),
        (r'@ObservedObject\s+var\s+\w+\s*=', "high", "@ObservedObject with default value - use @StateObject"),
    ],
    # REMOVED: "Missing Weak Self" category entirely - too many false positives
    "Hardcoded URLs": [
        (r'URL\s*\(\s*string:\s*["\']https?://', "low", "Hardcoded URL"),
    ],
    "Print Statements": [
        (r'^\s*print\s*\(', "low", "Debug print statement"),
        (r'NSLog\s*\(', "low", "NSLog statement"),
        # REMOVED: Debug-only print pattern (it's acceptable, why flag it?)
    ],
}

# SwiftData-specific patterns (iOS 17+)
# SIGNIFICANTLY REDUCED - Only flag real bugs, not expected SwiftData behavior
SWIFTDATA_PATTERNS = {
    "SwiftData Model Issues": [
        # Missing @Model macro - this is a real bug
        (r'class\s+\w+.*:\s*.*PersistentModel', "medium", "Class conforms to PersistentModel but may be missing @Model macro"),
        # Relationship without inverse - can cause CloudKit sync issues
        (r'@Relationship\s*\([^)]*\)\s*(?!.*inverse)', "medium", "@Relationship without inverse - may cause sync issues"),
        # REMOVED: @Query outside view - it works fine in ViewModels
        # REMOVED: ModelContext in Task - SwiftData handles this correctly in most cases
        # REMOVED: Insert without save - autosave IS the expected behavior
    ],
    "SwiftData Migration Issues": [
        # Only flag if there's evidence of schema changes without versioning
        # REMOVED: Generic "no migration plan" - too noisy, lightweight is fine for most apps
        (r'NSManagedObjectModel.*merge', "medium", "Merging models - ensure migration path is tested"),
    ],
    "SwiftData Performance": [
        # REMOVED: FetchDescriptor without predicate - sometimes you need all records
        # REMOVED: Missing fetchLimit - most queries legitimately need all results
        # Only flag obvious performance issues
        (r'for\s+\w+\s+in\s+[^{]+\{[^}]*modelContext\.save', "medium", "save() inside loop - consider batching"),
    ],
}

# CloudKit-specific patterns
# REDUCED - Only flag things that are likely actual bugs
CLOUDKIT_PATTERNS = {
    "CloudKit Sync Issues": [
        # Only flag missing error handling on critical operations
        # REMOVED: Most patterns were informational, not bugs
        (r'CKError\.Code\.serverRecordChanged(?!.*retry|merge)', "medium", "serverRecordChanged without retry/merge logic"),
    ],
    # REMOVED: CloudKit Container Issues - hardcoded identifiers are often intentional
    # REMOVED: CloudKit Subscription Issues - too informational
    # REMOVED: CloudKit Record Issues - CKAsset usage is normal, not a warning
    # REMOVED: CloudKit Sharing Issues - too broad
}

# Core Data patterns (for projects not yet migrated to SwiftData)
# REDUCED - Only flag clear threading bugs
COREDATA_PATTERNS = {
    "Core Data Thread Safety": [
        # Only flag clear thread safety violations
        # REMOVED: viewContext.perform - that's the correct way to use it
        (r'newBackgroundContext\(\)(?!.*perform)', "medium", "Background context without perform block - thread unsafe"),
        (r'DispatchQueue\.[^}]*managedObject', "medium", "Passing managed object across threads - use objectID instead"),
    ],
    # REMOVED: Core Data Performance - fetchBatchSize and faults are optimization details
    # REMOVED: Core Data Migration - too informational
}

# File extensions that should be analyzed for each pattern category
FILE_TYPE_FILTERS = {
    # Swift-specific patterns (reduced set - only real bugs)
    "Force Unwrap": {".swift"},
    "Retain Cycles": {".swift"},  # Now only Combine-specific
    "Actor Isolation": {".swift"},
    "Sendable Issues": {".swift"},
    "SwiftUI Lifecycle": {".swift"},
    "Hardcoded URLs": {".swift", ".m", ".mm"},
    "Print Statements": {".swift", ".m", ".mm"},
    # SwiftData patterns (iOS 17+) - reduced
    "SwiftData Model Issues": {".swift"},
    "SwiftData Migration Issues": {".swift"},
    "SwiftData Performance": {".swift"},
    # CloudKit patterns - minimal
    "CloudKit Sync Issues": {".swift"},
    # Core Data patterns - only thread safety
    "Core Data Thread Safety": {".swift", ".m", ".mm"},
    # Cross-language security patterns
    "SQL Injection": {".py", ".js", ".ts", ".java", ".go", ".rb", ".php"},
    "Command Injection": {".py", ".js", ".ts", ".java", ".go", ".rb", ".php", ".sh"},
    "Hardcoded Secrets": None,  # None = check all files
    "Path Traversal": {".py", ".js", ".ts", ".java", ".go", ".rb", ".php"},
    # Quality patterns
    "TODO/FIXME": None,  # Check all files
    "Debug Code": {".js", ".ts", ".jsx", ".tsx", ".py"},
    "Empty Catch": {".py", ".js", ".ts", ".java", ".swift", ".go"},
}

QUALITY_PATTERNS = {
    "TODO/FIXME": [
        (r'//\s*TODO:', "low", "TODO comment"),
        (r'//\s*FIXME:', "medium", "FIXME comment"),
        (r'//\s*HACK:', "medium", "HACK comment"),
        (r'#\s*TODO:', "low", "TODO comment"),
        (r'#\s*FIXME:', "medium", "FIXME comment"),
    ],
    "Debug Code": [
        (r'console\.log\s*\(', "low", "console.log statement"),
        (r'debugger\s*;', "medium", "debugger statement"),
        (r'breakpoint\s*\(\s*\)', "medium", "breakpoint() call"),
    ],
    "Empty Catch": [
        (r'except\s*:', "medium", "Bare except clause"),
        (r'catch\s*\{\s*\}', "medium", "Empty catch block"),
        (r'catch\s*\([^)]*\)\s*\{\s*\}', "medium", "Empty catch block"),
    ],
}


class FallbackAnalyzer:
    """
    Pattern-based analyzer for when RLM REPL fails.

    Uses regex patterns to find common issues without code execution.
    Always produces results, even if less comprehensive than full RLM.
    """

    def __init__(self):
        self.all_patterns: dict[str, list[tuple[str, str, str]]] = {}
        self.all_patterns.update(SECURITY_PATTERNS)
        self.all_patterns.update(SWIFT_PATTERNS)
        self.all_patterns.update(SWIFTDATA_PATTERNS)
        self.all_patterns.update(CLOUDKIT_PATTERNS)
        self.all_patterns.update(COREDATA_PATTERNS)
        self.all_patterns.update(QUALITY_PATTERNS)

    def analyze(
        self,
        content: str,
        query: str,
        fallback_reason: str = "Code execution failed"
    ) -> FallbackResult:
        """
        Analyze content using pattern matching.

        Args:
            content: The concatenated file content (### File: format)
            query: The user's query (used to select relevant patterns)
            fallback_reason: Why we're using fallback

        Returns:
            FallbackResult with findings
        """
        result = FallbackResult(
            query=query,
            fallback_reason=fallback_reason
        )

        # Parse files from content
        files = self._parse_files(content)
        result.files_analyzed = len(files)

        # Select relevant patterns based on query
        patterns_to_use = self._select_patterns(query)
        result.patterns_checked = sum(len(p) for p in patterns_to_use.values())

        # Analyze each file
        for file_path, file_content in files.items():
            # Get file extension
            file_ext = self._get_extension(file_path)
            lines = file_content.split('\n')

            for category, patterns in patterns_to_use.items():
                # Check if this category applies to this file type
                allowed_extensions = FILE_TYPE_FILTERS.get(category)
                if allowed_extensions is not None and file_ext not in allowed_extensions:
                    continue  # Skip this category for this file type

                for pattern, severity, description in patterns:
                    try:
                        regex = re.compile(pattern, re.IGNORECASE)

                        for line_num, line in enumerate(lines, 1):
                            if regex.search(line):
                                result.findings.append(FallbackFinding(
                                    category=category,
                                    file_path=file_path,
                                    line_number=line_num,
                                    line_content=line,
                                    severity=severity,
                                    description=description,
                                ))
                    except re.error:
                        # Skip invalid regex
                        pass

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        result.findings.sort(key=lambda f: severity_order.get(f.severity, 3))

        return result

    def _parse_files(self, content: str) -> dict[str, str]:
        """Parse ### File: format into dict."""
        files = {}
        parts = content.split("### File:")

        for part in parts[1:]:  # Skip first empty part
            lines = part.split("\n")
            if lines:
                file_path = lines[0].strip()
                # Clean markdown artifacts from file paths
                file_path = self._clean_file_path(file_path)
                if file_path:  # Skip empty paths
                    file_content = "\n".join(lines[1:])
                    files[file_path] = file_content

        return files

    def _clean_file_path(self, file_path: str) -> str:
        """Remove markdown artifacts and normalize file path."""
        # Remove markdown bold/italic markers
        file_path = file_path.replace("**", "").replace("__", "")
        file_path = file_path.replace("*", "").replace("_", "")
        # Remove backticks
        file_path = file_path.replace("`", "")
        # Remove leading/trailing whitespace and quotes
        file_path = file_path.strip().strip('"').strip("'")
        # Remove any trailing colons or punctuation
        file_path = file_path.rstrip(":")
        return file_path

    def _get_extension(self, file_path: str) -> str:
        """Get file extension from path."""
        if '.' in file_path:
            return '.' + file_path.rsplit('.', 1)[-1].lower()
        return ''

    def _select_patterns(self, query: str) -> dict[str, list[tuple[str, str, str]]]:
        """Select relevant patterns based on query keywords."""
        query_lower = query.lower()

        # Keywords to pattern categories
        keyword_map = {
            ("security", "vulnerability", "vulnerabilities", "injection", "secret", "password", "key"):
                [SECURITY_PATTERNS],
            ("swift", "ios", "xcode", "unwrap", "force"):
                [SWIFT_PATTERNS],
            ("swiftdata", "swift data", "@model", "@query", "modelcontext", "persistentmodel"):
                [SWIFTDATA_PATTERNS],
            ("cloudkit", "cloud kit", "ckrecord", "ckcontainer", "icloud", "sync"):
                [CLOUDKIT_PATTERNS],
            ("coredata", "core data", "nsmanagedobject", "nsfetchrequest", "nspersistent"):
                [COREDATA_PATTERNS],
            ("data", "persistence", "database", "storage"):
                [SWIFTDATA_PATTERNS, CLOUDKIT_PATTERNS, COREDATA_PATTERNS],
            ("quality", "todo", "fixme", "debug", "clean"):
                [QUALITY_PATTERNS],
            ("audit", "review", "all", "comprehensive"):
                [SECURITY_PATTERNS, SWIFT_PATTERNS, SWIFTDATA_PATTERNS, CLOUDKIT_PATTERNS, COREDATA_PATTERNS, QUALITY_PATTERNS],
        }

        selected = {}

        for keywords, pattern_dicts in keyword_map.items():
            if any(kw in query_lower for kw in keywords):
                for pd in pattern_dicts:
                    selected.update(pd)

        # Default to security if nothing matched
        if not selected:
            selected.update(SECURITY_PATTERNS)

        return selected

    def quick_scan(self, content: str) -> dict[str, int]:
        """
        Quick scan for issue counts without full analysis.

        Returns category -> count mapping.
        """
        files = self._parse_files(content)
        counts: dict[str, int] = {}

        for file_path, file_content in files.items():
            for category, patterns in self.all_patterns.items():
                for pattern, _, _ in patterns:
                    try:
                        matches = len(re.findall(pattern, file_content, re.IGNORECASE))
                        if matches:
                            counts[category] = counts.get(category, 0) + matches
                    except re.error:
                        pass

        return counts
