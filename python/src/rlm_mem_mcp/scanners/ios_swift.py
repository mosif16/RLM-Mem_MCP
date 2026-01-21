"""
iOS/Swift scanners for RLM tools (v2.9).

Contains scanners for Swift-specific issues:
- Force unwraps and optionals
- Retain cycles and memory management
- Concurrency issues (async/await, actors)
- SwiftUI performance
- Accessibility and localization
"""

import re
from typing import TYPE_CHECKING

from ..common_types import Finding, Confidence, Severity, ToolResult
from ..scan_patterns import FORCE_UNWRAP_PATTERNS, RETAIN_CYCLE_PATTERNS, SAFE_TRY_PATTERNS

if TYPE_CHECKING:
    from ..scan_base import ScannerBase


class iOSSwiftScanner:
    """iOS and Swift-specific scanners."""

    def __init__(self, base: "ScannerBase"):
        self.base = base

    def find_force_unwraps(self) -> ToolResult:
        """
        Find force unwraps (!) in Swift code.

        v2.7: Enhanced to filter safe patterns like Bundle.main, static lets.
        """
        result = ToolResult(tool_name="Force Unwrap Scanner", files_scanned=len(self.base.files))

        # Patterns that indicate safe force unwraps
        safe_patterns = [
            r'Bundle\.main',
            r'UIApplication\.shared',
            r'FileManager\.default',
            r'static\s+let',
            r'@IBOutlet',
            r'@IBAction',
            r'fatalError',
            r'preconditionFailure',
        ]

        pattern = r'(\w+)!(?!\s*=)(?!\.)(?!["\'])'
        matches = self.base._search_pattern(pattern, file_filter=".swift")

        for filepath, line_num, line, match in matches:
            if self.base._is_in_comment(line, filepath):
                continue

            # Skip safe patterns
            if any(re.search(sp, line) for sp in safe_patterns):
                continue

            # Skip != comparisons
            if '!=' in line:
                continue

            finding = self.base._create_finding(
                filepath=filepath,
                line_num=line_num,
                line=line,
                issue="Force unwrap (!) - may crash at runtime",
                severity=Severity.MEDIUM,
                fix="Use guard let, if let, or nil coalescing (??)",
                category="force_unwrap",
            )
            if finding:
                result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} force unwraps"
        return result

    def find_retain_cycles(self) -> ToolResult:
        """Find potential retain cycles in closures."""
        result = ToolResult(tool_name="Retain Cycle Scanner", files_scanned=len(self.base.files))

        patterns = [
            (r'''\{\s*(?!\[(?:weak|unowned)\s+self\])[^}]*\bself\.[^}]*\}''',
             "Closure capturing self without weak/unowned", Severity.MEDIUM),
            (r'''Timer\.[^}]*\{[^}]*self\.[^}]*\}''',
             "Timer closure without weak self", Severity.HIGH),
            (r'''NotificationCenter[^}]*\{[^}]*self\.[^}]*\}''',
             "NotificationCenter observer without weak self", Severity.HIGH),
            (r'''DispatchQueue[^}]*\{[^}]*self\.[^}]*\}''',
             "DispatchQueue closure referencing self", Severity.LOW),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self.base._is_in_comment(line, filepath):
                    continue

                # Skip if weak/unowned is already present
                if '[weak self]' in line or '[unowned self]' in line:
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Add [weak self] or [unowned self] to closure",
                    category="retain_cycle",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} potential retain cycles"
        return result

    def find_weak_self_issues(self) -> ToolResult:
        """Find weak self without guard let unwrapping."""
        result = ToolResult(tool_name="Weak Self Scanner", files_scanned=len(self.base.files))

        # Pattern: [weak self] followed by self? usage without guard
        pattern = r'\[weak\s+self\][^}]*self\?\.'
        matches = self.base._search_pattern(pattern, file_filter=".swift")

        for filepath, line_num, line, match in matches:
            if self.base._is_in_comment(line, filepath):
                continue

            # Check if there's a guard let self nearby
            lines = self.base._get_file_lines(filepath)
            if lines:
                context_start = max(0, line_num - 3)
                context = '\n'.join(lines[context_start:line_num + 2])
                if 'guard let self' in context or 'guard let `self`' in context:
                    continue

            finding = self.base._create_finding(
                filepath=filepath,
                line_num=line_num,
                line=line,
                issue="weak self used with optional chaining - consider guard let",
                severity=Severity.LOW,
                fix="Use 'guard let self = self else { return }' for cleaner code",
                category="weak_self",
            )
            if finding:
                result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} weak self issues"
        return result

    def find_mainactor_issues(self) -> ToolResult:
        """Find @MainActor annotation issues."""
        result = ToolResult(tool_name="MainActor Scanner", files_scanned=len(self.base.files))

        patterns = [
            # UI updates without @MainActor
            (r'''(?<!@MainActor[^{]*)\b(?:UIView|UILabel|UIButton|UITableView)\.[^=]*=''',
             "UI update may need @MainActor", Severity.MEDIUM),
            # ObservableObject without @MainActor
            (r'''class\s+\w+\s*:\s*ObservableObject(?![^{]*@MainActor)''',
             "ObservableObject should use @MainActor", Severity.MEDIUM),
            # @Published without @MainActor on class
            (r'''@Published\s+(?:var|let)''',
             "@Published property - ensure class has @MainActor", Severity.LOW),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self.base._is_in_comment(line, filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Add @MainActor annotation for UI-related code",
                    category="mainactor",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} @MainActor issues"
        return result

    def find_async_await_issues(self) -> ToolResult:
        """Find async/await related issues."""
        result = ToolResult(tool_name="Async/Await Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Missing await
            (r'''(?<!await\s)Task\s*\{''', "Task without await - fire and forget?", Severity.LOW),
            # Blocking call in async context
            (r'''async\s+func[^}]*Thread\.sleep''',
             "Thread.sleep in async function - use Task.sleep", Severity.HIGH),
            # DispatchQueue in async function
            (r'''async\s+func[^}]*DispatchQueue\.main\.async''',
             "DispatchQueue in async function - use @MainActor", Severity.MEDIUM),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self.base._is_in_comment(line, filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Review async/await usage",
                    category="async_await",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} async/await issues"
        return result

    def find_sendable_issues(self) -> ToolResult:
        """Find Sendable conformance issues."""
        result = ToolResult(tool_name="Sendable Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Class without Sendable used in async context
            (r'''class\s+\w+(?![^{]*:\s*[^{]*Sendable)''',
             "Class may need Sendable conformance for concurrency", Severity.LOW),
            # @unchecked Sendable
            (r'''@unchecked\s+Sendable''',
             "@unchecked Sendable bypasses safety checks", Severity.MEDIUM),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self.base._is_in_comment(line, filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Add Sendable conformance or use actors",
                    category="sendable",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} Sendable issues"
        return result

    def find_swiftui_performance_issues(self) -> ToolResult:
        """Find SwiftUI performance issues."""
        result = ToolResult(tool_name="SwiftUI Performance Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Heavy computation in body
            (r'''var\s+body[^}]*\.filter\(''',
             "filter() in body - move to computed property or onAppear", Severity.MEDIUM),
            (r'''var\s+body[^}]*\.sorted\(''',
             "sorted() in body - move to computed property or onAppear", Severity.MEDIUM),
            # State in ForEach
            (r'''ForEach[^}]*@State''',
             "@State inside ForEach loop - use item properties instead", Severity.HIGH),
            # Large inline closures
            (r'''\.onAppear\s*\{[^}]{500,}\}''',
             "Large onAppear closure - extract to method", Severity.LOW),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self.base._is_in_comment(line, filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Optimize SwiftUI view performance",
                    category="swiftui_performance",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} SwiftUI performance issues"
        return result

    def find_memory_management_issues(self) -> ToolResult:
        """Find memory management issues in Swift."""
        result = ToolResult(tool_name="Memory Management Scanner", files_scanned=len(self.base.files))

        patterns = [
            # unowned with optional (crash risk)
            (r'''unowned\s+var\s+\w+\s*:\s*\w+\?''',
             "unowned with optional type - will crash if nil", Severity.CRITICAL),
            # Strong reference in NotificationCenter
            (r'''NotificationCenter\.default\.addObserver\s*\([^)]*self[^)]*\{(?!\s*\[weak)''',
             "NotificationCenter observer may retain self", Severity.HIGH),
            # Timer without invalidation pattern
            (r'''Timer\.scheduledTimer[^}]+\}''',
             "Timer may not be invalidated - potential memory leak", Severity.MEDIUM),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self.base._is_in_comment(line, filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Review memory management",
                    category="memory_management",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} memory management issues"
        return result

    def find_error_handling_issues(self) -> ToolResult:
        """Find error handling issues in Swift."""
        result = ToolResult(tool_name="Error Handling Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Empty catch block
            (r'''catch\s*\{[\s\n]*\}''', "Empty catch block - errors are silently ignored", Severity.HIGH),
            # catch with only print
            (r'''catch\s*\{[\s\n]*print\s*\([^)]+\)[\s\n]*\}''',
             "Catch block only prints - consider proper error handling", Severity.MEDIUM),
            # try? without nil handling
            (r'''try\?\s+\w+[^\n]*\n(?!\s*(?:guard|if|else))''',
             "try? without nil check - error is silently ignored", Severity.MEDIUM),
            # fatalError in production code
            (r'''fatalError\s*\(\s*"[^"]*"\s*\)''',
             "fatalError in code - will crash in production", Severity.HIGH),
            # Force try
            (r'''\btry!\s+\w''',
             "try! can crash at runtime - use do-catch", Severity.HIGH),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self.base._is_in_comment(line, filepath):
                    continue

                # Skip test files for fatalError
                if 'fatalError' in line and self.base._is_test_file(filepath):
                    continue

                # Skip safe try! patterns
                if 'try!' in line and any(re.search(safe, line) for safe in SAFE_TRY_PATTERNS):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Implement proper error handling",
                    category="error_handling",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} error handling issues"
        return result

    def find_accessibility_issues(self) -> ToolResult:
        """Find accessibility issues in SwiftUI."""
        result = ToolResult(tool_name="Accessibility Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Image without accessibility label
            (r'''Image\s*\([^)]+\)(?![^}]*\.accessibilityLabel)''',
             "Image without accessibility label", Severity.MEDIUM),
            # Button without accessibility hint
            (r'''Button\s*\([^)]*\)\s*\{[^}]*\}(?![^}]*\.accessibilityHint)''',
             "Button may need accessibility hint", Severity.LOW),
            # Icon-only tab items
            (r'''\.tabItem\s*\{[^}]*Image[^}]*\}(?![^}]*Text)''',
             "Tab item with only image - add label", Severity.MEDIUM),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self.base._is_in_comment(line, filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Add accessibility modifiers",
                    category="accessibility",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} accessibility issues"
        return result

    def find_localization_issues(self) -> ToolResult:
        """Find hardcoded strings that should be localized."""
        result = ToolResult(tool_name="Localization Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Text with hardcoded string
            (r'''Text\s*\(\s*"[A-Z][a-z]+(?:\s+[a-z]+)+"\s*\)(?!.*LocalizedStringKey)''',
             "Hardcoded string in Text - should be localized", Severity.MEDIUM),
            # Button label with hardcoded string
            (r'''Button\s*\(\s*"[A-Z][a-z]+(?:\s+[a-z]+)*"\s*[,)]''',
             "Hardcoded button label - should be localized", Severity.MEDIUM),
            # Alert with hardcoded strings
            (r'''\.alert\s*\(\s*"[A-Z][a-z]+''',
             "Hardcoded alert title - should be localized", Severity.MEDIUM),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern, file_filter=".swift")
            for filepath, line_num, line, match in matches:
                if self.base._is_in_comment(line, filepath):
                    continue

                # Skip test files
                if self.base._is_test_file(filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Use String(localized:) or LocalizedStringKey",
                    category="localization",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} localization issues"
        return result


def create_ios_swift_scanner(base: "ScannerBase") -> iOSSwiftScanner:
    """Factory function to create an iOSSwiftScanner."""
    return iOSSwiftScanner(base)
