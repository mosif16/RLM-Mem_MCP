"""
Rust scanners for RLM tools (v2.9).

Contains:
- find_unsafe_blocks: Unsafe Rust code analysis
- find_unwrap_usage: Unwrap/panic pattern detection
- find_rust_concurrency_issues: Data race and concurrency issues
- find_rust_error_handling: Error handling patterns
- find_rust_clippy_patterns: Clippy-style lint detection
"""

import re
from typing import TYPE_CHECKING

from ..common_types import Finding, Confidence, Severity, ToolResult

if TYPE_CHECKING:
    from ..scan_base import ScannerBase


class RustScanner:
    """Rust-specific scanners for safety and quality issues."""

    def __init__(self, base: "ScannerBase"):
        self.base = base

    def find_unsafe_blocks(self) -> ToolResult:
        """
        Find unsafe Rust code blocks and analyze their usage.

        Searches for:
        - unsafe blocks and functions
        - Raw pointer operations
        - FFI boundaries
        - Transmute usage
        """
        result = ToolResult(tool_name="Rust Unsafe Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Unsafe blocks and functions
            (r'\bunsafe\s*\{', "unsafe block - review for memory safety", Severity.MEDIUM),
            (r'\bunsafe\s+fn\b', "unsafe function - document safety requirements", Severity.MEDIUM),
            (r'\bunsafe\s+impl\b', "unsafe impl - ensure trait contract is upheld", Severity.MEDIUM),
            (r'\bunsafe\s+trait\b', "unsafe trait - document safety invariants", Severity.MEDIUM),
            # Raw pointers
            (r'\*(?:const|mut)\s+\w+', "Raw pointer type", Severity.LOW),
            (r'\.as_ptr\s*\(\)', "Converting to raw pointer", Severity.LOW),
            (r'\.as_mut_ptr\s*\(\)', "Converting to mutable raw pointer", Severity.MEDIUM),
            (r'\bfrom_raw\s*\(', "Creating from raw pointer - ensure pointer validity", Severity.HIGH),
            # Dangerous operations
            (r'std::mem::transmute', "transmute is extremely unsafe - prefer safe alternatives", Severity.CRITICAL),
            (r'std::mem::forget', "mem::forget prevents Drop - potential resource leak", Severity.HIGH),
            (r'std::mem::uninitialized', "uninitialized memory is UB - use MaybeUninit", Severity.CRITICAL),
            (r'std::mem::zeroed', "zeroed memory may be invalid for type", Severity.HIGH),
            # FFI
            (r'extern\s+"C"\s*\{', "FFI boundary - ensure correct ABI", Severity.MEDIUM),
            (r'#\[no_mangle\]', "no_mangle for FFI - ensure correct signature", Severity.LOW),
            # Union access
            (r'\bunion\s+\w+', "Union type - all field access is unsafe", Severity.MEDIUM),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if not filepath.endswith('.rs'):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:150],
                    issue=issue,
                    confidence=Confidence.HIGH,
                    severity=severity,
                    category="rust_unsafe",
                ))

        result.summary = f"Found {len(result.findings)} unsafe Rust patterns"
        return result

    def find_unwrap_usage(self) -> ToolResult:
        """
        Find .unwrap() and .expect() usage that may panic.

        Searches for:
        - .unwrap() calls
        - .expect() calls
        - Indexing that may panic
        """
        result = ToolResult(tool_name="Rust Unwrap Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Unwrap variants
            (r'\.unwrap\s*\(\)', ".unwrap() may panic - handle error properly", Severity.MEDIUM),
            (r'\.unwrap_or_default\s*\(\)', ".unwrap_or_default() - ensure default is appropriate", Severity.LOW),
            (r'\.expect\s*\(["\']', ".expect() may panic - ensure invariant is documented", Severity.LOW),
            # Unchecked indexing
            (r'\[\s*\w+\s*\](?!\s*(?:\.get|\.as_ref))', "Direct indexing may panic - consider .get()", Severity.LOW),
            # Unreachable/unimplemented
            (r'\bunreachable!\s*\(', "unreachable! - ensure this path is truly unreachable", Severity.LOW),
            (r'\bunimplemented!\s*\(', "unimplemented! - stub that will panic", Severity.MEDIUM),
            (r'\btodo!\s*\(', "todo! - incomplete code that will panic", Severity.HIGH),
            # Panic
            (r'\bpanic!\s*\(', "Explicit panic - consider Result/Option", Severity.MEDIUM),
            (r'\.unwrap_err\s*\(\)', ".unwrap_err() panics on Ok - ensure this is tested", Severity.MEDIUM),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if not filepath.endswith('.rs'):
                    continue

                # Skip test files for unwrap (common and acceptable)
                if '/tests/' in filepath or '_test.rs' in filepath or '#[test]' in line:
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:150],
                    issue=issue,
                    confidence=Confidence.MEDIUM,
                    severity=severity,
                    fix="Use ? operator or match/if let for proper error handling",
                    category="rust_unwrap",
                ))

        result.summary = f"Found {len(result.findings)} unwrap/panic patterns"
        return result

    def find_rust_concurrency_issues(self) -> ToolResult:
        """
        Find Rust concurrency issues and potential data races.

        Searches for:
        - Arc without Mutex for interior mutability
        - Mutex poisoning risks
        - Deadlock patterns
        - Unsafe Send/Sync implementations
        """
        result = ToolResult(tool_name="Rust Concurrency Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Arc without Mutex
            (r'Arc<(?!.*Mutex|RwLock|Atomic).*RefCell', "Arc<RefCell<T>> is not thread-safe - use Mutex", Severity.CRITICAL),
            (r'Arc<(?!.*Mutex|RwLock|Atomic).*Cell<', "Arc<Cell<T>> is not thread-safe - use Atomic*", Severity.CRITICAL),
            # Mutex patterns
            (r'\.lock\s*\(\)\s*\.unwrap\s*\(\)', "Mutex lock unwrap - handle poisoned mutex", Severity.MEDIUM),
            (r'Mutex::new\([^)]*Mutex', "Nested Mutex - potential deadlock", Severity.HIGH),
            # Unsafe Send/Sync
            (r'unsafe\s+impl\s+Send', "Manual Send impl - ensure thread safety", Severity.HIGH),
            (r'unsafe\s+impl\s+Sync', "Manual Sync impl - ensure thread safety", Severity.HIGH),
            # Static mut
            (r'static\s+mut\s+', "static mut is unsafe - prefer atomics or Mutex", Severity.HIGH),
            # Channel patterns
            (r'\.send\s*\([^)]+\)\s*\.unwrap\s*\(\)', "Channel send unwrap - receiver may be dropped", Severity.MEDIUM),
            # Thread spawn without join
            (r'thread::spawn\s*\([^)]+\)(?!.*\.join)', "Thread spawned without join - may outlive caller", Severity.LOW),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if not filepath.endswith('.rs'):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:150],
                    issue=issue,
                    confidence=Confidence.HIGH,
                    severity=severity,
                    category="rust_concurrency",
                ))

        result.summary = f"Found {len(result.findings)} Rust concurrency issues"
        return result

    def find_rust_error_handling(self) -> ToolResult:
        """
        Find Rust error handling patterns and issues.

        Searches for:
        - Improper Result handling
        - Missing error propagation
        - Error type issues
        """
        result = ToolResult(tool_name="Rust Error Handling Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Discarding Results
            (r'let\s+_\s*=\s*\w+\s*\?\s*;', "Propagating then discarding Result", Severity.LOW),
            (r'(?<!let\s)(?<!return\s)\w+\([^)]*\)\s*;(?=\s*//.*(?:Result|Option))', "Ignoring Result/Option return value", Severity.MEDIUM),
            # String errors
            (r'Result<.*,\s*String>', "Using String for errors - consider custom error type", Severity.LOW),
            (r'Result<.*,\s*&\'static\s+str>', "Using &str for errors - consider custom error type", Severity.LOW),
            # Box<dyn Error>
            (r'Box<dyn\s+(?:std::)?error::Error', "Box<dyn Error> loses error context - consider thiserror/anyhow", Severity.LOW),
            # Panic in Result context
            (r'\.map_err\s*\([^)]*panic', "Panic in map_err - defeats purpose of Result", Severity.HIGH),
            # Ok-wrapping anti-pattern
            (r'Ok\s*\(\s*\(\s*\)\s*\)', "Ok(()) - consider returning more useful value", Severity.LOW),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if not filepath.endswith('.rs'):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:150],
                    issue=issue,
                    confidence=Confidence.MEDIUM,
                    severity=severity,
                    category="rust_errors",
                ))

        result.summary = f"Found {len(result.findings)} Rust error handling issues"
        return result

    def find_rust_clippy_patterns(self) -> ToolResult:
        """
        Find patterns that Clippy would flag.

        Manual detection of common Clippy lints for when Clippy isn't available.
        """
        result = ToolResult(tool_name="Rust Clippy Pattern Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Redundant clone
            (r'\.clone\s*\(\)\s*\.clone\s*\(\)', "Redundant double clone", Severity.LOW),
            (r'\.to_string\s*\(\)\s*\.to_string\s*\(\)', "Redundant double to_string", Severity.LOW),
            # Useless conversions
            (r'\.into\s*\(\)\s*\.into\s*\(\)', "Redundant double into()", Severity.LOW),
            (r'String::from\s*\(\s*"[^"]*"\s*\)\.as_str\s*\(\)', "Useless String::from().as_str()", Severity.LOW),
            # Inefficient patterns
            (r'\.iter\s*\(\)\s*\.map\s*\([^)]+\)\s*\.collect::<Vec', "iter().map().collect() - consider .iter().map() directly", Severity.LOW),
            (r'\.len\s*\(\)\s*==\s*0', ".len() == 0 - use .is_empty()", Severity.LOW),
            (r'\.len\s*\(\)\s*>\s*0', ".len() > 0 - use !.is_empty()", Severity.LOW),
            # Option/Result patterns
            (r'\.is_some\s*\(\)\s*\{[^}]*\.unwrap\s*\(\)', "is_some() then unwrap() - use if let Some(x)", Severity.MEDIUM),
            (r'\.is_ok\s*\(\)\s*\{[^}]*\.unwrap\s*\(\)', "is_ok() then unwrap() - use if let Ok(x)", Severity.MEDIUM),
            (r'match\s+\w+\s*\{\s*Some\s*\(\s*x\s*\)\s*=>\s*Some\s*\(', "match Some(x) => Some() - use .map()", Severity.LOW),
            # Single char patterns
            (r'\.starts_with\s*\(\s*"[^"]"\s*\)', "starts_with single char - use char literal", Severity.LOW),
            (r'\.ends_with\s*\(\s*"[^"]"\s*\)', "ends_with single char - use char literal", Severity.LOW),
            # Loop patterns
            (r'for\s+\w+\s+in\s+0\s*\.\.\s*\w+\.len\s*\(\)', "for i in 0..x.len() - use for item in &x", Severity.LOW),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if not filepath.endswith('.rs'):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:150],
                    issue=issue,
                    confidence=Confidence.MEDIUM,
                    severity=severity,
                    fix="Run `cargo clippy` for full lint analysis",
                    category="rust_clippy",
                ))

        result.summary = f"Found {len(result.findings)} Clippy-style issues"
        return result


def create_rust_scanner(base: "ScannerBase") -> RustScanner:
    """Factory function to create a RustScanner."""
    return RustScanner(base)
