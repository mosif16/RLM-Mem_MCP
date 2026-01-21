"""
Batch scan orchestration for RLM tools (v2.9).

Contains:
- run_security_scan: All security-related scans
- run_ios_scan: All iOS/Swift scans
- run_quality_scan: All code quality scans
- run_web_scan: All web/frontend scans
- run_rust_scan: All Rust scans
- run_node_scan: All Node.js scans
- run_frontend_scan: Combined web + node scans
- run_backend_scan: Combined node + security scans
"""

from typing import TYPE_CHECKING, Callable

from ..common_types import ToolResult, Confidence
from ..parallel_execution import parallel_scan

if TYPE_CHECKING:
    from ..scan_base import ScannerBase


class BatchScanner:
    """Orchestrates batch scanning across multiple scanner types."""

    def __init__(self, base: "ScannerBase"):
        self.base = base
        # Scanner instances will be created lazily
        self._security_scanner = None
        self._ios_scanner = None
        self._quality_scanner = None
        self._web_scanner = None
        self._rust_scanner = None
        self._node_scanner = None
        self._architecture_scanner = None

    def _get_security_scanner(self):
        """Lazy-load security scanner."""
        if self._security_scanner is None:
            from .security import create_security_scanner
            self._security_scanner = create_security_scanner(self.base)
        return self._security_scanner

    def _get_ios_scanner(self):
        """Lazy-load iOS scanner."""
        if self._ios_scanner is None:
            from .ios_swift import create_ios_swift_scanner
            self._ios_scanner = create_ios_swift_scanner(self.base)
        return self._ios_scanner

    def _get_quality_scanner(self):
        """Lazy-load quality scanner."""
        if self._quality_scanner is None:
            from .quality import create_quality_scanner
            self._quality_scanner = create_quality_scanner(self.base)
        return self._quality_scanner

    def _get_web_scanner(self):
        """Lazy-load web frontend scanner."""
        if self._web_scanner is None:
            from .web_frontend import create_web_frontend_scanner
            self._web_scanner = create_web_frontend_scanner(self.base)
        return self._web_scanner

    def _get_rust_scanner(self):
        """Lazy-load Rust scanner."""
        if self._rust_scanner is None:
            from .rust import create_rust_scanner
            self._rust_scanner = create_rust_scanner(self.base)
        return self._rust_scanner

    def _get_node_scanner(self):
        """Lazy-load Node scanner."""
        if self._node_scanner is None:
            from .node import create_node_scanner
            self._node_scanner = create_node_scanner(self.base)
        return self._node_scanner

    def _get_architecture_scanner(self):
        """Lazy-load architecture scanner."""
        if self._architecture_scanner is None:
            from .architecture import create_architecture_scanner
            self._architecture_scanner = create_architecture_scanner(self.base)
        return self._architecture_scanner

    def _filter_by_confidence(
        self,
        results: list[ToolResult],
        min_confidence: str
    ) -> list[ToolResult]:
        """
        Filter results to only include findings at or above minimum confidence.

        Args:
            results: List of ToolResult objects
            min_confidence: "LOW", "MEDIUM", or "HIGH"

        Returns:
            Filtered list with updated summaries
        """
        confidence_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        min_level = confidence_order.get(min_confidence.upper(), 0)

        filtered_results = []
        for result in results:
            filtered_findings = [
                f for f in result.findings
                if confidence_order.get(f.confidence.value if hasattr(f.confidence, 'value') else str(f.confidence), 0) >= min_level
            ]

            if filtered_findings or not result.findings:
                new_result = ToolResult(
                    tool_name=result.tool_name,
                    findings=filtered_findings,
                    summary=f"{result.summary} (filtered: >={min_confidence})",
                    files_scanned=result.files_scanned,
                    errors=result.errors,
                )
                filtered_results.append(new_result)

        return filtered_results

    def _run_scan_functions(
        self,
        scan_functions: list[Callable],
        parallel: bool = True
    ) -> list[ToolResult]:
        """Run a list of scan functions, optionally in parallel."""
        if parallel:
            return parallel_scan(self.base, scan_functions)
        else:
            return [func() for func in scan_functions]

    def run_security_scan(
        self,
        min_confidence: str = "MEDIUM",
        include_quality: bool = False,
        parallel: bool = True,
    ) -> list[ToolResult]:
        """
        Run all security-related scans.

        Args:
            min_confidence: Minimum confidence level ("LOW", "MEDIUM", "HIGH")
            include_quality: If True, include code quality checks.
            parallel: If True, run scans in parallel (2-4x faster).
        """
        security = self._get_security_scanner()

        scan_functions = [
            security.find_secrets,
            security.find_sql_injection,
            security.find_command_injection,
            security.find_xss_vulnerabilities,
            security.find_python_security,
        ]

        all_results = self._run_scan_functions(scan_functions, parallel)

        if include_quality:
            all_results.extend(self.run_quality_scan(min_confidence="LOW", parallel=parallel))

        if min_confidence.upper() != "LOW":
            all_results = self._filter_by_confidence(all_results, min_confidence)

        return all_results

    def run_ios_scan(
        self,
        min_confidence: str = "MEDIUM",
        include_quality: bool = False,
        parallel: bool = True,
    ) -> list[ToolResult]:
        """
        Run all iOS/Swift scans.

        Args:
            min_confidence: Minimum confidence level ("LOW", "MEDIUM", "HIGH")
            include_quality: If True, include code quality checks.
            parallel: If True, run scans in parallel (2-4x faster).
        """
        ios = self._get_ios_scanner()

        scan_functions = [
            ios.find_force_unwraps,
            ios.find_retain_cycles,
            ios.find_weak_self_issues,
            ios.find_async_await_issues,
            ios.find_sendable_issues,
            ios.find_mainactor_issues,
            ios.find_swiftui_performance_issues,
            ios.find_memory_management_issues,
            ios.find_error_handling_issues,
        ]

        all_results = self._run_scan_functions(scan_functions, parallel)

        if include_quality:
            quality_funcs = [
                ios.find_accessibility_issues,
                ios.find_localization_issues,
            ]
            all_results.extend(self._run_scan_functions(quality_funcs, parallel))
            all_results.extend(self.run_quality_scan(min_confidence="LOW", parallel=parallel))

        if min_confidence.upper() != "LOW":
            all_results = self._filter_by_confidence(all_results, min_confidence)

        return all_results

    def run_quality_scan(
        self,
        min_confidence: str = "LOW",
        include_all: bool = True,
        parallel: bool = True
    ) -> list[ToolResult]:
        """
        Run all quality-related scans.

        Args:
            min_confidence: Minimum confidence level ("LOW", "MEDIUM", "HIGH")
            include_all: If True, run all quality checks.
            parallel: If True, run scans in parallel (2-4x faster).
        """
        quality = self._get_quality_scanner()

        scan_functions = [
            quality.find_long_functions,
            quality.find_todos,
        ]

        if include_all:
            scan_functions.extend([
                quality.find_complex_functions,
                quality.find_code_smells,
                quality.find_dead_code,
            ])

        all_results = self._run_scan_functions(scan_functions, parallel)

        if min_confidence.upper() != "LOW":
            all_results = self._filter_by_confidence(all_results, min_confidence)

        return all_results

    def run_web_scan(
        self,
        min_confidence: str = "MEDIUM",
        parallel: bool = True
    ) -> list[ToolResult]:
        """
        Run all web/frontend scans.

        Args:
            min_confidence: Minimum confidence level ("LOW", "MEDIUM", "HIGH")
            parallel: If True, run scans in parallel (2-4x faster).
        """
        web = self._get_web_scanner()
        security = self._get_security_scanner()

        scan_functions = [
            web.find_react_issues,
            web.find_vue_issues,
            web.find_angular_issues,
            web.find_dom_security,
            web.find_a11y_issues,
            web.find_css_issues,
            security.find_xss_vulnerabilities,
        ]

        all_results = self._run_scan_functions(scan_functions, parallel)

        if min_confidence.upper() != "LOW":
            all_results = self._filter_by_confidence(all_results, min_confidence)

        return all_results

    def run_rust_scan(
        self,
        min_confidence: str = "MEDIUM",
        parallel: bool = True
    ) -> list[ToolResult]:
        """
        Run all Rust scans.

        Args:
            min_confidence: Minimum confidence level ("LOW", "MEDIUM", "HIGH")
            parallel: If True, run scans in parallel (2-4x faster).
        """
        rust = self._get_rust_scanner()

        scan_functions = [
            rust.find_unsafe_blocks,
            rust.find_unwrap_usage,
            rust.find_rust_concurrency_issues,
            rust.find_rust_error_handling,
            rust.find_rust_clippy_patterns,
        ]

        all_results = self._run_scan_functions(scan_functions, parallel)

        if min_confidence.upper() != "LOW":
            all_results = self._filter_by_confidence(all_results, min_confidence)

        return all_results

    def run_node_scan(
        self,
        min_confidence: str = "MEDIUM",
        parallel: bool = True
    ) -> list[ToolResult]:
        """
        Run all Node.js scans.

        Args:
            min_confidence: Minimum confidence level ("LOW", "MEDIUM", "HIGH")
            parallel: If True, run scans in parallel (2-4x faster).
        """
        node = self._get_node_scanner()

        scan_functions = [
            node.find_callback_hell,
            node.find_promise_issues,
            node.find_node_security,
            node.find_require_issues,
            node.find_node_async_issues,
        ]

        all_results = self._run_scan_functions(scan_functions, parallel)

        if min_confidence.upper() != "LOW":
            all_results = self._filter_by_confidence(all_results, min_confidence)

        return all_results

    def run_frontend_scan(
        self,
        min_confidence: str = "MEDIUM",
        parallel: bool = True
    ) -> list[ToolResult]:
        """
        Run combined frontend scan (web + node).

        Args:
            min_confidence: Minimum confidence level ("LOW", "MEDIUM", "HIGH")
            parallel: If True, run scans in parallel (2-4x faster).
        """
        return self.run_web_scan(min_confidence, parallel) + self.run_node_scan(min_confidence, parallel)

    def run_backend_scan(
        self,
        min_confidence: str = "MEDIUM",
        parallel: bool = True
    ) -> list[ToolResult]:
        """
        Run combined backend scan (node + security).

        Args:
            min_confidence: Minimum confidence level ("LOW", "MEDIUM", "HIGH")
            parallel: If True, run scans in parallel (2-4x faster).
        """
        return self.run_node_scan(min_confidence, parallel) + self.run_security_scan(min_confidence, parallel=parallel)


def create_batch_scanner(base: "ScannerBase") -> BatchScanner:
    """Factory function to create a BatchScanner."""
    return BatchScanner(base)
