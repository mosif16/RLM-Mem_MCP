"""
Security and validation for REPL sandbox execution.

Provides code validation, syntax repair, and security checks for
sandboxed Python code execution.
"""

import ast
import re
from typing import Optional

# Dangerous attribute names that could be used for sandbox escapes
FORBIDDEN_ATTRIBUTES = frozenset({
    '__class__', '__base__', '__bases__', '__subclasses__',
    '__mro__', '__globals__', '__builtins__', '__code__',
    '__reduce__', '__reduce_ex__', '__getattribute__',
    '__setattr__', '__delattr__', '__init_subclass__',
    '__set_name__', '__repr__', '__str__', '__bytes__',
    '__dict__', '__closure__', '__func__', '__self__',
    '__module__', '__qualname__', '__annotations__',
    '__wrapped__', 'gi_frame', 'gi_code', 'f_globals',
    'f_locals', 'f_builtins', 'co_code', 'func_globals',
})

# Forbidden function names
FORBIDDEN_CALLS = frozenset({
    'eval', 'exec', 'compile', 'open', 'input', '__import__',
    'getattr', 'setattr', 'delattr', 'globals', 'locals',
    'vars', 'dir', 'breakpoint', 'exit', 'quit',
})

# Modules that are pre-loaded in the REPL environment (safe to "import")
PRELOADED_MODULES = frozenset({'re'})

# Available built-in functions in the sandbox (for documentation)
AVAILABLE_BUILTINS = [
    "len", "str", "int", "float", "bool", "list", "dict", "tuple", "set",
    "range", "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "min", "max", "sum", "any", "all", "abs", "round", "print",
    "isinstance", "ord", "chr", "repr", "frozenset"
]


class UnsafeCodeError(Exception):
    """Raised when code contains potentially unsafe constructs."""
    pass


def strip_markdown_fences_from_content(content_lines: list[str]) -> list[str]:
    """
    Strip markdown code fences from file content lines.

    When files are combined with get_combined_content(), each file's content
    is wrapped in markdown fences like:
        ```language
        actual content
        ```

    This function removes those fences to get the actual file content.

    Args:
        content_lines: Lines of content that may include markdown fences

    Returns:
        Lines with markdown fences removed
    """
    if not content_lines:
        return content_lines

    result = list(content_lines)

    # Remove opening fence (```language) - it's typically the first line
    if result and result[0].strip().startswith('```'):
        result = result[1:]

    # Remove closing fence (```) - check last few lines for trailing whitespace
    while result and (result[-1].strip() == '```' or result[-1].strip() == ''):
        if result[-1].strip() == '```':
            result = result[:-1]
            break
        result = result[:-1]

    return result


def attempt_syntax_repair(code: str, error: SyntaxError) -> tuple[Optional[str], str]:
    """
    Attempt to automatically repair common syntax errors in LLM-generated code.

    Args:
        code: The code with syntax error
        error: The SyntaxError that was raised

    Returns:
        (repaired_code, repair_description) - repaired_code is None if repair failed
    """
    error_msg = str(error).lower()
    repairs = []

    # 1. Remove markdown code fences (very common LLM mistake)
    if code.startswith("```") or "```python" in code or "```\n" in code:
        # Remove opening fence
        code = re.sub(r'^```(?:python|py)?\s*\n?', '', code)
        # Remove closing fence
        code = re.sub(r'\n?```\s*$', '', code)
        repairs.append("Removed markdown code fences")

    # 2. Handle unterminated triple-quoted strings
    if "unterminated triple-quoted string" in error_msg or "unterminated string" in error_msg:
        # Count triple quotes to see if we're missing closers
        triple_double = code.count('"""')
        triple_single = code.count("'''")

        # If odd number of triple double quotes, add closing one
        if triple_double % 2 == 1:
            code = code.rstrip() + '\n"""'
            repairs.append('Added missing """ at end')

        # If odd number of triple single quotes, add closing one
        if triple_single % 2 == 1:
            code = code.rstrip() + "\n'''"
            repairs.append("Added missing ''' at end")

    # 3. Handle unterminated regular strings
    if "eol while scanning string literal" in error_msg or "unterminated string literal" in error_msg:
        lines = code.split('\n')
        if error.lineno and error.lineno <= len(lines):
            line = lines[error.lineno - 1]
            # Simple heuristic: if line has odd number of quotes, add one
            if line.count('"') % 2 == 1:
                lines[error.lineno - 1] = line + '"'
                code = '\n'.join(lines)
                repairs.append(f'Added missing " at line {error.lineno}')
            elif line.count("'") % 2 == 1:
                lines[error.lineno - 1] = line + "'"
                code = '\n'.join(lines)
                repairs.append(f"Added missing ' at line {error.lineno}")

    # 4. Handle missing colons after def/if/for/while/class/try/except/with
    if "expected ':'" in error_msg or "invalid syntax" in error_msg:
        lines = code.split('\n')
        if error.lineno and error.lineno <= len(lines):
            line = lines[error.lineno - 1]
            # Check if line looks like a statement that needs a colon
            needs_colon_pattern = r'^(\s*)(def|if|elif|else|for|while|class|try|except|finally|with|async\s+def|async\s+for|async\s+with)\b.*[^:]\s*$'
            if re.match(needs_colon_pattern, line):
                lines[error.lineno - 1] = line.rstrip() + ':'
                code = '\n'.join(lines)
                repairs.append(f'Added missing : at line {error.lineno}')

    # 5. Handle missing parentheses in print (Python 2 to 3 migration)
    if "missing parentheses in call to 'print'" in error_msg:
        # Convert print x to print(x)
        code = re.sub(r'\bprint\s+([^(\n]+)', r'print(\1)', code)
        repairs.append("Added parentheses to print statements")

    # 6. Handle f-string with backslash (common error)
    if "f-string expression part cannot include a backslash" in error_msg:
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if 'f"' in line or "f'" in line:
                if '\\n' in line or '\\t' in line:
                    lines[i] = f"# FIXME: f-string with backslash needs refactoring\n{line}"
                    repairs.append(f"Marked problematic f-string at line {i+1}")
        code = '\n'.join(lines)

    # 7. Handle indentation errors
    if "unexpected indent" in error_msg or "expected an indented block" in error_msg:
        lines = code.split('\n')
        if error.lineno and error.lineno <= len(lines):
            if "expected an indented block" in error_msg:
                # Add a pass statement if block is empty
                prev_line = lines[error.lineno - 2] if error.lineno > 1 else ""
                if prev_line.rstrip().endswith(':'):
                    indent = len(prev_line) - len(prev_line.lstrip()) + 4
                    lines.insert(error.lineno - 1, ' ' * indent + 'pass  # Auto-added')
                    code = '\n'.join(lines)
                    repairs.append(f'Added pass statement at line {error.lineno}')

    # 8. Handle incomplete list/dict/tuple (missing closing bracket)
    if "unexpected eof" in error_msg or "eof while scanning" in error_msg:
        open_parens = code.count('(') - code.count(')')
        open_brackets = code.count('[') - code.count(']')
        open_braces = code.count('{') - code.count('}')

        if open_parens > 0:
            code = code.rstrip() + ')' * open_parens
            repairs.append(f'Added {open_parens} missing )')
        if open_brackets > 0:
            code = code.rstrip() + ']' * open_brackets
            repairs.append(f'Added {open_brackets} missing ]')
        if open_braces > 0:
            code = code.rstrip() + '}' * open_braces
            repairs.append(f'Added {open_braces} missing }}')

    # 9. Remove trailing ellipsis (LLM sometimes adds ... to indicate continuation)
    if code.rstrip().endswith('...') and '...' not in error_msg:
        code = code.rstrip()[:-3].rstrip()
        if not code.endswith(':'):
            repairs.append("Removed trailing ellipsis")

    if repairs:
        return code, "; ".join(repairs)

    # Handle f-string with unmatched braces
    if "f-string" in error_msg:
        if code.count('{') > code.count('}'):
            code = code + '}'
            repairs.append('Added missing } for f-string')
        elif code.count('}') > code.count('{'):
            code = '{' + code
            repairs.append('Added missing { for f-string')
        if repairs:
            return code, "; ".join(repairs)

    # Handle missing colons at end of control structures
    if "expected ':'" in error_msg or "expected ':':" in error_msg:
        lines = code.split('\n')
        if error.lineno and error.lineno <= len(lines):
            line = lines[error.lineno - 1].rstrip()
            control_keywords = ['if ', 'elif ', 'else', 'for ', 'while ', 'def ', 'class ', 'try', 'except', 'finally', 'with ']
            for kw in control_keywords:
                if line.lstrip().startswith(kw) and not line.endswith(':'):
                    lines[error.lineno - 1] = line + ':'
                    return '\n'.join(lines), f'Added missing : at line {error.lineno}'

    # Handle incomplete code blocks (truncated by LLM)
    if error.lineno and error.lineno >= len(code.split('\n')) - 2:
        open_parens = code.count('(') - code.count(')')
        open_brackets = code.count('[') - code.count(']')
        open_braces = code.count('{') - code.count('}')

        if open_parens > 0:
            code = code + ')' * open_parens
            repairs.append(f'Added {open_parens} missing )')
        if open_brackets > 0:
            code = code + ']' * open_brackets
            repairs.append(f'Added {open_brackets} missing ]')
        if open_braces > 0:
            code = code + '}' * open_braces
            repairs.append(f'Added {open_braces} missing }}')

        if repairs:
            return code, "; ".join(repairs)

    return None, "Could not auto-repair"


class CodeValidator(ast.NodeVisitor):
    """
    AST validator to detect potentially dangerous code patterns.

    Blocks:
    - Access to dunder attributes (__class__, __globals__, etc.)
    - Dangerous function calls (eval, exec, open, etc.)
    - Import statements (except pre-loaded modules like 're')
    - Attribute chains that could escape sandbox

    Pre-loaded modules: re
    """

    def __init__(self):
        self.errors: list[str] = []
        self.syntax_error: Optional[SyntaxError] = None
        self.import_hints: list[str] = []  # Helpful hints for import errors

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check for forbidden attribute access."""
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self.errors.append(f"Forbidden attribute access: '{node.attr}'")
        elif node.attr.startswith('_') and not node.attr.startswith('__'):
            # Allow single underscore but warn about double underscore
            pass
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Check for forbidden name access."""
        if node.id in FORBIDDEN_CALLS:
            self.errors.append(f"Forbidden name: '{node.id}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check for forbidden function calls."""
        # Check direct calls like eval(), exec()
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                self.errors.append(f"Forbidden function call: '{node.func.id}'")
        # Check method calls
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_CALLS:
                self.errors.append(f"Forbidden method call: '{node.func.attr}'")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Block import statements (except pre-loaded modules)."""
        for alias in node.names:
            if alias.name in PRELOADED_MODULES:
                # This module is pre-loaded - add helpful hint instead of error
                self.import_hints.append(
                    f"Note: '{alias.name}' is already available (no import needed). "
                    f"Just use it directly: {alias.name}.findall(...)"
                )
            else:
                self.errors.append(
                    f"Import not allowed: 'import {alias.name}'. "
                    f"Pre-loaded modules: {', '.join(sorted(PRELOADED_MODULES))}. "
                    f"Available builtins: {', '.join(AVAILABLE_BUILTINS[:10])}..."
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Block from...import statements (except pre-loaded modules)."""
        if node.module in PRELOADED_MODULES:
            self.import_hints.append(
                f"Note: '{node.module}' is already available (no import needed). "
                f"Just use it directly: {node.module}.findall(...)"
            )
        else:
            self.errors.append(
                f"Import not allowed: 'from {node.module} import ...'. "
                f"Pre-loaded modules: {', '.join(sorted(PRELOADED_MODULES))}."
            )

    def validate(self, code: str) -> tuple[bool, list[str], list[str]]:
        """
        Validate code for safety.

        Returns:
            (is_safe, errors, hints) tuple
        """
        self.errors = []
        self.import_hints = []
        self.syntax_error = None

        try:
            tree = ast.parse(code)
            self.visit(tree)
        except SyntaxError as e:
            self.syntax_error = e
            self.errors.append(f"Syntax error: {e}")

        return len(self.errors) == 0, self.errors, self.import_hints


def validate_code(code: str) -> tuple[bool, list[str], Optional[SyntaxError], list[str]]:
    """
    Validate Python code for sandbox safety.

    Args:
        code: Python source code to validate

    Returns:
        (is_safe, errors, syntax_error, hints) tuple
        - is_safe: True if code passed validation
        - errors: List of error messages
        - syntax_error: SyntaxError if one occurred (for repair attempts)
        - hints: Helpful hints (e.g., "re is already available")
    """
    validator = CodeValidator()
    is_safe, errors, hints = validator.validate(code)
    return is_safe, errors, validator.syntax_error, hints
