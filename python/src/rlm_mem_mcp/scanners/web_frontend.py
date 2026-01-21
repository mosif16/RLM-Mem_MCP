"""
Web/Frontend scanners for RLM tools (v2.9).

Contains:
- find_react_issues: React-specific issues
- find_vue_issues: Vue.js issues
- find_angular_issues: Angular issues
- find_dom_security: DOM security issues
- find_a11y_issues: Accessibility issues
- find_css_issues: CSS/styling issues
"""

import re
from typing import TYPE_CHECKING

from ..common_types import Finding, Confidence, Severity, ToolResult
from ..scan_patterns import REACT_PATTERNS

if TYPE_CHECKING:
    from ..scan_base import ScannerBase


class WebFrontendScanner:
    """Web and frontend-specific scanners."""

    def __init__(self, base: "ScannerBase"):
        self.base = base

    def find_react_issues(self) -> ToolResult:
        """Find React-specific issues."""
        result = ToolResult(tool_name="React Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Missing key in list
            (r'''\.map\s*\([^)]*\)\s*=>.*<\w+(?![^>]*key=)''',
             "Missing key prop in list rendering", Severity.HIGH),
            # useEffect with empty deps and state setter
            (r'''useEffect\s*\(\s*\(\)\s*=>\s*\{[^}]*set\w+[^}]*\}\s*,\s*\[\s*\]\s*\)''',
             "useEffect with empty deps may cause infinite loop", Severity.HIGH),
            # Inline function in JSX
            (r'''onClick\s*=\s*\{\s*\(\)\s*=>''',
             "Inline arrow function in JSX prop (re-renders)", Severity.LOW),
            # Direct state mutation
            (r'''state\.\w+\s*=''',
             "Direct state mutation - use setState", Severity.HIGH),
            # Missing dependency in useCallback/useMemo
            (r'''use(?:Callback|Memo)\s*\([^)]*\]\s*\)''',
             "Check useCallback/useMemo dependencies", Severity.LOW),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if not any(filepath.endswith(ext) for ext in ['.jsx', '.tsx', '.js', '.ts']):
                    continue
                if self.base._is_in_comment(line, filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Review React best practices",
                    category="react",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} React issues"
        return result

    def find_vue_issues(self) -> ToolResult:
        """Find Vue.js specific issues."""
        result = ToolResult(tool_name="Vue Scanner", files_scanned=len(self.base.files))

        patterns = [
            # v-if with v-for (performance)
            (r'''v-for=.*v-if=|v-if=.*v-for=''',
             "v-if with v-for on same element - use computed property", Severity.MEDIUM),
            # Missing key in v-for
            (r'''v-for=[^>]*(?!:key|v-bind:key)>''',
             "Missing :key in v-for", Severity.HIGH),
            # Direct data mutation in Vuex
            (r'''this\.\$store\.state\.\w+\s*=''',
             "Direct Vuex state mutation - use mutations", Severity.HIGH),
            # Arrow function in methods
            (r'''methods:\s*\{[^}]*\w+:\s*\([^)]*\)\s*=>''',
             "Arrow function in methods loses 'this' binding", Severity.MEDIUM),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if not filepath.endswith('.vue'):
                    continue
                if self.base._is_in_comment(line, filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Review Vue.js best practices",
                    category="vue",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} Vue issues"
        return result

    def find_angular_issues(self) -> ToolResult:
        """Find Angular specific issues."""
        result = ToolResult(tool_name="Angular Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Subscribe without unsubscribe
            (r'''\.subscribe\s*\(''',
             "Observable subscription - ensure unsubscribe in ngOnDestroy", Severity.MEDIUM),
            # Missing trackBy in ngFor
            (r'''\*ngFor=[^>]*(?!;trackBy)''',
             "Missing trackBy in *ngFor - may cause performance issues", Severity.LOW),
            # Using any type
            (r''':\s*any\b''',
             "Using 'any' type - consider specific type", Severity.LOW),
            # Direct DOM manipulation
            (r'''document\.getElementById|document\.querySelector''',
             "Direct DOM manipulation - use ViewChild/Renderer2", Severity.MEDIUM),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if not any(filepath.endswith(ext) for ext in ['.ts', '.component.ts']):
                    continue
                if self.base._is_in_comment(line, filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Review Angular best practices",
                    category="angular",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} Angular issues"
        return result

    def find_dom_security(self) -> ToolResult:
        """Find DOM-related security issues."""
        result = ToolResult(tool_name="DOM Security Scanner", files_scanned=len(self.base.files))

        patterns = [
            # innerHTML assignment
            (r'''\.innerHTML\s*=''', "innerHTML assignment - potential XSS", Severity.HIGH),
            # outerHTML assignment
            (r'''\.outerHTML\s*=''', "outerHTML assignment - potential XSS", Severity.HIGH),
            # document.write
            (r'''document\.write\(''', "document.write - potential XSS", Severity.HIGH),
            # eval usage
            (r'''\beval\s*\(''', "eval usage - potential code injection", Severity.CRITICAL),
            # Function constructor
            (r'''new\s+Function\s*\(''', "Function constructor - potential code injection", Severity.HIGH),
            # location manipulation
            (r'''location\.href\s*=\s*[^'"<]''', "Dynamic location.href - potential open redirect", Severity.MEDIUM),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if not any(filepath.endswith(ext) for ext in ['.js', '.ts', '.jsx', '.tsx', '.vue', '.html']):
                    continue
                if self.base._is_in_comment(line, filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Use safe DOM methods or proper sanitization",
                    category="dom_security",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} DOM security issues"
        return result

    def find_a11y_issues(self) -> ToolResult:
        """Find accessibility (a11y) issues."""
        result = ToolResult(tool_name="Accessibility Scanner", files_scanned=len(self.base.files))

        patterns = [
            # img without alt
            (r'''<img[^>]*(?!alt=)[^>]*>''', "Image without alt attribute", Severity.HIGH),
            # Button without type
            (r'''<button(?![^>]*type=)[^>]*>''', "Button without type attribute", Severity.LOW),
            # onClick on non-interactive element
            (r'''<(?:div|span)[^>]*onClick''', "onClick on non-interactive element - use button", Severity.MEDIUM),
            # Missing label for input
            (r'''<input[^>]*(?!aria-label|aria-labelledby)[^>]*(?!id=)[^>]*>''',
             "Input may be missing associated label", Severity.MEDIUM),
            # Positive tabindex
            (r'''tabindex=["'][1-9]''', "Positive tabindex disrupts tab order", Severity.LOW),
            # Empty link
            (r'''<a[^>]*>\s*</a>''', "Empty link - missing accessible text", Severity.HIGH),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if not any(filepath.endswith(ext) for ext in ['.html', '.jsx', '.tsx', '.vue']):
                    continue
                if self.base._is_in_comment(line, filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Improve accessibility following WCAG guidelines",
                    category="a11y",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} accessibility issues"
        return result

    def find_css_issues(self) -> ToolResult:
        """Find CSS/styling issues."""
        result = ToolResult(tool_name="CSS Scanner", files_scanned=len(self.base.files))

        patterns = [
            # !important usage
            (r'''!\s*important''', "!important usage - consider specificity", Severity.LOW),
            # Hardcoded colors (potential theme issue)
            (r'''(?:color|background):\s*#[0-9a-fA-F]{3,6}''',
             "Hardcoded color - consider CSS variables", Severity.LOW),
            # Fixed pixel values for font
            (r'''font-size:\s*\d+px''', "Fixed font-size - consider rem/em for accessibility", Severity.LOW),
            # z-index war
            (r'''z-index:\s*[0-9]{4,}''', "Very high z-index - consider z-index scale", Severity.LOW),
            # Vendor prefix without standard
            (r'''-webkit-[^:]+:(?![^}]*[^-]webkit)''',
             "Vendor prefix may need standard property", Severity.LOW),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if not any(filepath.endswith(ext) for ext in ['.css', '.scss', '.less', '.sass']):
                    continue
                if self.base._is_in_comment(line, filepath):
                    continue

                finding = self.base._create_finding(
                    filepath=filepath,
                    line_num=line_num,
                    line=line,
                    issue=issue,
                    severity=severity,
                    fix="Review CSS best practices",
                    category="css",
                )
                if finding:
                    result.findings.append(finding)

        result.summary = f"Found {len(result.findings)} CSS issues"
        return result


def create_web_frontend_scanner(base: "ScannerBase") -> WebFrontendScanner:
    """Factory function to create a WebFrontendScanner."""
    return WebFrontendScanner(base)
