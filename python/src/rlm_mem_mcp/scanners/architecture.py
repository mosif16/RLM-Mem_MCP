"""
Architecture analysis scanners for RLM tools (v2.9).

Contains:
- map_architecture: Map codebase structure
- find_imports: Find module imports
- analyze_typescript_imports: Analyze TS/JS dependencies
- build_call_graph: Build function call graph
"""

import re
from typing import TYPE_CHECKING, Optional

from ..common_types import Finding, Confidence, Severity, ToolResult

if TYPE_CHECKING:
    from ..scan_base import ScannerBase


class ArchitectureScanner:
    """Architecture analysis tools."""

    def __init__(self, base: "ScannerBase"):
        self.base = base

    def map_architecture(self, detailed: bool = True) -> ToolResult:
        """
        Map the codebase architecture with detailed output.

        Args:
            detailed: If True, extract key classes/functions from each file

        Returns:
            ToolResult with file categories and key symbols
        """
        result = ToolResult(tool_name="Architecture Mapper", files_scanned=len(self.base.files))

        categories: dict[str, list[dict]] = {
            "entry_points": [],
            "views_ui": [],
            "models": [],
            "services": [],
            "utilities": [],
            "tests": [],
            "config": [],
            "other": [],
        }

        imports_by_file: dict[str, list[str]] = {}

        for filepath in self.base.files:
            lower = filepath.lower()
            file_info: dict = {"path": filepath, "classes": [], "functions": [], "imports": []}

            # Extract key symbols if detailed mode
            if detailed:
                content = self.base._get_file_content(filepath)
                if content:
                    ext = filepath.split('.')[-1].lower() if '.' in filepath else ''

                    # Extract classes/structs
                    if ext == 'swift':
                        file_info["classes"] = re.findall(r'(?:class|struct|enum|protocol|actor)\s+(\w+)', content)
                        file_info["functions"] = re.findall(r'func\s+(\w+)\s*\(', content)
                        file_info["imports"] = re.findall(r'import\s+(\w+)', content)
                    elif ext == 'py':
                        file_info["classes"] = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
                        file_info["functions"] = re.findall(r'^(?:async\s+)?def\s+(\w+)', content, re.MULTILINE)
                        file_info["imports"] = [m[0] or m[1] for m in re.findall(r'^(?:from\s+(\S+)|import\s+(\S+))', content, re.MULTILINE)]
                    elif ext in ('js', 'ts', 'tsx', 'jsx'):
                        file_info["classes"] = re.findall(r'(?:class|interface)\s+(\w+)', content)
                        file_info["functions"] = re.findall(r'(?:function|const|let|var)\s+(\w+)\s*[=\(]', content)
                        file_info["imports"] = re.findall(r"(?:import|require)\s*\(?['\"]([^'\"]+)['\"]", content)

                    imports_by_file[filepath] = file_info["imports"]

            # Categorize file
            if any(x in lower for x in ['main.', 'index.', 'app.', '__main__', 'cli.', 'entrypoint']):
                categories["entry_points"].append(file_info)
            elif any(x in lower for x in ['view', 'controller', 'screen', 'component', 'page', 'widget']):
                categories["views_ui"].append(file_info)
            elif any(x in lower for x in ['model', 'entity', 'schema', 'types', 'dto']):
                categories["models"].append(file_info)
            elif any(x in lower for x in ['service', 'manager', 'provider', 'handler', 'api', 'client', 'repository']):
                categories["services"].append(file_info)
            elif any(x in lower for x in ['util', 'helper', 'common', 'shared', 'extension', 'utils']):
                categories["utilities"].append(file_info)
            elif any(x in lower for x in ['test', 'spec', '_test', '.test', 'mock', 'fixture']):
                categories["tests"].append(file_info)
            elif any(x in lower for x in ['config', 'setting', '.json', '.yaml', '.env', '.plist']):
                categories["config"].append(file_info)
            else:
                categories["other"].append(file_info)

        # Create findings for each category
        for category, files in categories.items():
            if not files:
                continue

            code_lines = []
            for f in files[:30]:
                path_line = f["path"]
                if f.get("classes"):
                    path_line += f"  [Classes: {', '.join(f['classes'][:5])}]"
                if f.get("functions") and len(f.get("classes", [])) < 3:
                    funcs = [fn for fn in f['functions'][:5] if not fn.startswith('_')]
                    if funcs:
                        path_line += f"  [Funcs: {', '.join(funcs)}]"
                code_lines.append(path_line)

            if len(files) > 30:
                code_lines.append(f"... and {len(files) - 30} more files")

            result.findings.append(Finding(
                file=category,
                line=0,
                code="\n".join(code_lines),
                issue=f"{len(files)} files",
                confidence=Confidence.HIGH,
                severity=Severity.INFO,
                category="architecture",
                fix=f"Category: {category.replace('_', ' ').title()}"
            ))

        # Add dependency summary
        if imports_by_file:
            all_imports: dict[str, int] = {}
            for imports in imports_by_file.values():
                for imp in imports:
                    all_imports[imp] = all_imports.get(imp, 0) + 1

            top_imports = sorted(all_imports.items(), key=lambda x: -x[1])[:15]
            if top_imports:
                import_summary = "\n".join(f"{imp}: {count} files" for imp, count in top_imports)
                result.findings.append(Finding(
                    file="dependencies",
                    line=0,
                    code=import_summary,
                    issue=f"Top {len(top_imports)} imports/dependencies",
                    confidence=Confidence.HIGH,
                    severity=Severity.INFO,
                    category="architecture",
                    fix="Most frequently imported modules"
                ))

        total_categories = len([c for c in categories.values() if c])
        total_classes = sum(len(f.get("classes", [])) for files in categories.values() for f in files)
        total_functions = sum(len(f.get("functions", [])) for files in categories.values() for f in files)

        result.summary = f"Mapped {len(self.base.files)} files into {total_categories} categories. Found {total_classes} classes, {total_functions} functions."
        return result

    def find_imports(self, module_name: str) -> ToolResult:
        """
        Find all imports of a specific module.

        Args:
            module_name: Name of module to find imports for
        """
        result = ToolResult(tool_name=f"Import Scanner ({module_name})", files_scanned=len(self.base.files))

        patterns = [
            (rf'''import\s+{re.escape(module_name)}''', "import statement"),
            (rf'''from\s+{re.escape(module_name)}''', "from import"),
            (rf'''require\(['"]{re.escape(module_name)}['"]''', "require()"),
        ]

        for pattern, issue in patterns:
            matches = self.base._search_pattern(pattern)
            for filepath, line_num, line, match in matches:
                result.findings.append(Finding(
                    file=filepath,
                    line=line_num,
                    code=line[:200],
                    issue=issue,
                    confidence=Confidence.HIGH,
                    severity=Severity.INFO,
                    category="imports",
                ))

        result.summary = f"Found {len(result.findings)} imports of {module_name}"
        return result

    def analyze_typescript_imports(self, filepath: Optional[str] = None) -> ToolResult:
        """Analyze TypeScript/JavaScript imports and exports."""
        result = ToolResult(tool_name="TypeScript Import Analyzer", files_scanned=len(self.base.files))

        import_patterns = [
            (r'''import\s+\{([^}]+)\}\s+from\s+['"]([@\w./\-]+)['"]''', 'named'),
            (r'''import\s+(\w+)\s+from\s+['"]([@\w./\-]+)['"]''', 'default'),
            (r'''import\s+\*\s+as\s+(\w+)\s+from\s+['"]([@\w./\-]+)['"]''', 'namespace'),
            (r'''import\s+['"]([@\w./\-]+)['"]''', 'side_effect'),
        ]

        target_files = [filepath] if filepath else self.base.files
        import_map = {}

        for fp in target_files:
            if not any(fp.endswith(ext) for ext in ['.ts', '.tsx', '.js', '.jsx', '.mjs']):
                continue

            content = self.base._get_file_content(fp)
            if not content:
                continue

            imports = []
            for pattern, import_type in import_patterns:
                for match in re.finditer(pattern, content):
                    if import_type == 'side_effect':
                        imports.append((import_type, '*', match.group(1)))
                    else:
                        imports.append((import_type, match.group(1).strip(), match.group(2)))

            if imports:
                import_map[fp] = imports
                for import_type, items, source in imports:
                    result.findings.append(Finding(
                        file=fp,
                        line=1,
                        code=f"import {{{items}}} from '{source}'",
                        issue=f"Imports from {source}",
                        confidence=Confidence.HIGH,
                        severity=Severity.INFO,
                        category="import",
                    ))

        result.summary = f"Analyzed {len(import_map)} files with imports"
        return result

    def build_call_graph(self, entry_point: Optional[str] = None) -> ToolResult:
        """Build a function call graph for TypeScript/JavaScript."""
        result = ToolResult(tool_name="Call Graph Builder", files_scanned=len(self.base.files))

        func_def_patterns = [
            (r'''function\s+(\w+)\s*\(([^)]*)\)''', 'function'),
            (r'''(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>''', 'arrow'),
            (r'''(?:async\s+)?(\w+)\s*\(([^)]*)\)\s*\{''', 'method'),
        ]

        call_patterns = [
            r'''(\w+)\s*\(''',
            r'''this\.(\w+)\s*\(''',
            r'''await\s+(\w+)\s*\(''',
        ]

        functions = {}
        calls = {}

        for filepath in self.base.files:
            if not any(filepath.endswith(ext) for ext in ['.ts', '.tsx', '.js', '.jsx']):
                continue

            lines = self.base._get_file_lines(filepath)
            if not lines:
                continue

            current_function = None

            for line_num, line in enumerate(lines, 1):
                for pattern, func_type in func_def_patterns:
                    match = re.search(pattern, line)
                    if match:
                        func_name = match.group(1)
                        params = match.group(2) if len(match.groups()) > 1 else ""
                        functions[func_name] = (filepath, line_num, params, func_type)
                        current_function = func_name

                        result.findings.append(Finding(
                            file=filepath,
                            line=line_num,
                            code=f"{func_type}: {func_name}({params})",
                            issue=f"Function definition: {func_name}",
                            confidence=Confidence.HIGH,
                            severity=Severity.INFO,
                            category="function_def",
                        ))
                        break

                if current_function:
                    for call_pattern in call_patterns:
                        for match in re.finditer(call_pattern, line):
                            callee = match.group(1)
                            if callee in ['if', 'for', 'while', 'switch', 'catch', 'console', 'log']:
                                continue
                            if current_function not in calls:
                                calls[current_function] = set()
                            calls[current_function].add(callee)

        result.summary = f"Found {len(functions)} functions, {sum(len(c) for c in calls.values())} calls"
        return result


def create_architecture_scanner(base: "ScannerBase") -> ArchitectureScanner:
    """Factory function to create an ArchitectureScanner."""
    return ArchitectureScanner(base)
