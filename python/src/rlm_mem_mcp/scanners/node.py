"""
Node.js scanners for RLM tools (v2.9).

Contains:
- find_callback_hell: Deeply nested callback detection
- find_promise_issues: Promise anti-patterns
- find_node_security: Node.js security vulnerabilities
- find_require_issues: require/import problems
- find_node_async_issues: Async/await issues
"""

import re
from typing import TYPE_CHECKING

from ..common_types import Finding, Confidence, Severity, ToolResult

if TYPE_CHECKING:
    from ..scan_base import ScannerBase


class NodeScanner:
    """Node.js-specific scanners for quality and security issues."""

    def __init__(self, base: "ScannerBase"):
        self.base = base

    def find_callback_hell(self) -> ToolResult:
        """
        Find deeply nested callbacks (callback hell).

        Searches for:
        - Deeply nested callbacks
        - Pyramid of doom patterns
        """
        result = ToolResult(tool_name="Callback Hell Scanner", files_scanned=len(self.base.files))

        for filepath in self.base.files:
            if not any(filepath.endswith(ext) for ext in ['.js', '.mjs', '.cjs', '.ts']):
                continue

            lines = self.base._get_file_lines(filepath)
            if not lines:
                continue

            # Track nesting depth
            callback_depth = 0
            callback_start_line = 0

            for line_num, line in enumerate(lines, 1):
                # Count callback indicators
                callbacks_opened = len(re.findall(r'(?:function\s*\(|=>\s*\{|\(\s*(?:err|error|e)\s*(?:,|\)\s*=>))', line))
                callbacks_closed = line.count('});') + line.count('})') + line.count('};')

                callback_depth += callbacks_opened - callbacks_closed

                if callbacks_opened > 0 and callback_start_line == 0:
                    callback_start_line = line_num

                if callback_depth >= 4:  # 4+ levels of nesting
                    result.findings.append(Finding(
                        file=filepath,
                        line=callback_start_line or line_num,
                        code=line[:150],
                        issue=f"Callback nesting depth {callback_depth} - consider async/await or Promises",
                        confidence=Confidence.HIGH,
                        severity=Severity.MEDIUM,
                        fix="Refactor to use async/await or Promise chains",
                        category="callback_hell",
                    ))
                    callback_depth = 0
                    callback_start_line = 0

                if callback_depth <= 0:
                    callback_depth = 0
                    callback_start_line = 0

        result.summary = f"Found {len(result.findings)} callback hell patterns"
        return result

    def find_promise_issues(self) -> ToolResult:
        """
        Find Promise anti-patterns and issues.

        Searches for:
        - Unhandled rejections
        - Promise constructor anti-pattern
        - Missing error handling
        """
        result = ToolResult(tool_name="Promise Issues Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Unhandled promises
            (r'new\s+Promise\s*\([^)]+\)(?!\s*\.(then|catch|finally))', "Promise without .catch() - may have unhandled rejection", Severity.MEDIUM),
            (r'\.then\s*\([^)]+\)(?!\s*\.(catch|finally))', ".then() without .catch() - add error handling", Severity.MEDIUM),
            # Promise constructor anti-pattern
            (r'new\s+Promise\s*\([^)]*resolve\s*\([^)]*await', "Promise constructor with await - unnecessary wrapper", Severity.LOW),
            (r'new\s+Promise\s*\([^)]*resolve\s*\(\s*\w+\s*\)\s*\)', "Promise wrapping non-promise - use Promise.resolve()", Severity.LOW),
            # async function without await
            (r'async\s+(?:function\s+\w+|\w+\s*=\s*async)\s*\([^)]*\)\s*\{(?![\s\S]*await)', "async function without await", Severity.LOW),
            # Common mistakes
            (r'await\s+\w+\.forEach\s*\(', "await with forEach doesn't work as expected - use for...of", Severity.HIGH),
            (r'\.catch\s*\(\s*\(\s*\)\s*=>\s*\{\s*\}\s*\)', "Empty catch block - at least log the error", Severity.MEDIUM),
            # Promise.all pitfalls
            (r'Promise\.all\s*\(\s*\[(?![\s\S]*\.map)', "Promise.all with static array - ensure all are promises", Severity.LOW),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if not any(filepath.endswith(ext) for ext in ['.js', '.mjs', '.cjs', '.ts', '.tsx', '.jsx']):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:150],
                    issue=issue,
                    confidence=Confidence.MEDIUM,
                    severity=severity,
                    category="promise",
                ))

        result.summary = f"Found {len(result.findings)} Promise issues"
        return result

    def find_node_security(self) -> ToolResult:
        """
        Find Node.js security vulnerabilities.

        Searches for:
        - child_process without sanitization
        - Path traversal risks
        - Unsafe require/import
        - Prototype pollution risks
        """
        result = ToolResult(tool_name="Node.js Security Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Command injection
            (r'exec\s*\(\s*(?:`[^`]*\$|["\'][^"\']*\+)', "Command injection risk - exec with user input", Severity.CRITICAL),
            (r'execSync\s*\(\s*(?:`[^`]*\$|["\'][^"\']*\+)', "Command injection risk - execSync with user input", Severity.CRITICAL),
            (r'spawn\s*\(\s*(?:req\.|user|input)', "Potential command injection in spawn", Severity.HIGH),
            # Path traversal
            (r'(?:readFile|writeFile|readdir)\s*\([^)]*(?:req\.|user|input)', "Path traversal risk - validate file paths", Severity.HIGH),
            (r'path\.join\s*\([^)]*(?:req\.|user|input|\.\.))', "Path traversal risk - sanitize path input", Severity.HIGH),
            (r'__dirname\s*\+\s*(?:req\.|user|input)', "Path traversal risk - use path.join with validation", Severity.HIGH),
            # Eval and similar
            (r'\beval\s*\(', "eval is dangerous - avoid completely", Severity.CRITICAL),
            (r'new\s+Function\s*\(', "new Function is like eval - avoid", Severity.CRITICAL),
            (r'vm\.runInContext\s*\([^)]*(?:req\.|user|input)', "vm.runInContext with user input - sandbox escape risk", Severity.CRITICAL),
            # Prototype pollution
            (r'Object\.assign\s*\(\s*\{\s*\}[^)]*(?:req\.|user|input)', "Object.assign with user input - prototype pollution risk", Severity.HIGH),
            (r'\[(?:req\.|user|input)[^\]]*\]\s*=', "Dynamic property assignment - prototype pollution risk", Severity.MEDIUM),
            # Unsafe deserialization
            (r'JSON\.parse\s*\([^)]*(?:req\.|user|input)', "JSON.parse with untrusted input - validate schema", Severity.MEDIUM),
            (r'(?:serialize|unserialize)\s*\(', "Serialization may be unsafe - use JSON", Severity.HIGH),
            # SQL in template literals
            (r'(?:query|execute)\s*\(\s*`[^`]*\$\{', "SQL injection risk - use parameterized queries", Severity.CRITICAL),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if not any(filepath.endswith(ext) for ext in ['.js', '.mjs', '.cjs', '.ts']):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:150],
                    issue=issue,
                    confidence=Confidence.HIGH,
                    severity=severity,
                    category="node_security",
                ))

        result.summary = f"Found {len(result.findings)} Node.js security issues"
        return result

    def find_require_issues(self) -> ToolResult:
        """
        Find require/import issues in Node.js.

        Searches for:
        - Dynamic requires
        - Circular dependency indicators
        - Deprecated module usage
        """
        result = ToolResult(tool_name="Node.js Require Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Dynamic require
            (r'require\s*\(\s*(?![\'"]).+\)', "Dynamic require - may cause bundling issues", Severity.MEDIUM),
            (r'require\s*\(\s*`', "Template literal require - unpredictable", Severity.MEDIUM),
            (r'import\s*\(\s*(?![\'"]).+\)', "Dynamic import - ensure proper error handling", Severity.LOW),
            # Deprecated Node.js modules
            (r'require\s*\(\s*["\'](?:domain|sys)["\']', "Deprecated Node.js module", Severity.MEDIUM),
            (r'require\s*\(\s*["\']punycode["\']', "punycode is deprecated in Node.js", Severity.LOW),
            # Common issues
            (r'require\.cache\s*\[', "Manipulating require.cache - may cause issues", Severity.MEDIUM),
            (r'delete\s+require\.cache', "Clearing require cache - ensure this is intentional", Severity.MEDIUM),
            # Missing file extensions
            (r'require\s*\(\s*["\']\.\/[^"\']+(?<!\.[jt]sx?|\.json|\.node)["\']', "require without file extension - may fail", Severity.LOW),
            # Conditional requires (harder to analyze)
            (r'if\s*\([^)]+\)\s*\{[^}]*require\s*\(', "Conditional require - may cause issues with bundlers", Severity.LOW),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if not any(filepath.endswith(ext) for ext in ['.js', '.mjs', '.cjs', '.ts']):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:150],
                    issue=issue,
                    confidence=Confidence.MEDIUM,
                    severity=severity,
                    category="node_require",
                ))

        result.summary = f"Found {len(result.findings)} require/import issues"
        return result

    def find_node_async_issues(self) -> ToolResult:
        """
        Find async/await issues specific to Node.js.

        Searches for:
        - Missing await
        - Sequential awaits that could be parallel
        - Event emitter memory leaks
        """
        result = ToolResult(tool_name="Node.js Async Scanner", files_scanned=len(self.base.files))

        patterns = [
            # Missing await
            (r'(?<!await\s)(?:readFile|writeFile|mkdir|rmdir|unlink|rename|copyFile)\s*\(', "Async fs operation without await", Severity.MEDIUM),
            (r'\.json\s*\(\)(?!\s*\.then|\s*;?\s*$)', ".json() without await - returns Promise", Severity.MEDIUM),
            (r'fetch\s*\([^)]+\)(?!\s*\.then|await)', "fetch without await/then", Severity.MEDIUM),
            # Sequential awaits
            (r'await\s+\w+\([^)]*\)\s*;\s*\n\s*await\s+\w+\([^)]*\)\s*;\s*\n\s*await\s+', "Sequential awaits - consider Promise.all()", Severity.LOW),
            # Event emitter issues
            (r'\.on\s*\([^)]+\)(?![\s\S]{0,100}\.removeListener|\.off)', "Event listener without cleanup", Severity.MEDIUM),
            (r'\.addListener\s*\([^)]+\)(?![\s\S]{0,100}\.removeListener)', "addListener without removeListener", Severity.MEDIUM),
            (r'emitter\.setMaxListeners\s*\(\s*0\s*\)', "Unlimited listeners - potential memory leak", Severity.HIGH),
            # Async in constructor
            (r'constructor\s*\([^)]*\)\s*\{[^}]*await\s', "await in constructor - use factory function instead", Severity.HIGH),
            # process.exit in async
            (r'async\s+\w+[^{]*\{[^}]*process\.exit', "process.exit in async function - may skip cleanup", Severity.MEDIUM),
        ]

        for pattern, issue, severity in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                if not any(filepath.endswith(ext) for ext in ['.js', '.mjs', '.cjs', '.ts']):
                    continue

                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:150],
                    issue=issue,
                    confidence=Confidence.MEDIUM,
                    severity=severity,
                    category="node_async",
                ))

        result.summary = f"Found {len(result.findings)} Node.js async issues"
        return result


def create_node_scanner(base: "ScannerBase") -> NodeScanner:
    """Factory function to create a NodeScanner."""
    return NodeScanner(base)
