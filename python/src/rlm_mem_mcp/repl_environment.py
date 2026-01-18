"""
REPL Environment for RLM Processing

This is the CORE of the RLM technique from arXiv:2512.24601.

From the paper (see Figure 1):
- `prompt` variable contains the full content (NOT in LLM context)
- LLM writes code like: print(prompt[:100])
- LLM splits content: part1, part2 = prompt.split("Chapter 2")
- LLM queries sub-LLM: result = llm_query(f"Find X in: {part1}")
- Results stored in variables (pre_cata, post_cata, etc.)
- Final answer via: print(FINAL_ANSWER)

This preserves ALL data - nothing is summarized or lost.

Improvements (based on user feedback):
- Dead code detection (#if false, #if DEBUG, etc.)
- Confidence levels for findings (HIGH/MEDIUM/LOW)
- Line number verification before reporting
- Implementation detection (body vs signature)
"""

import re
import ast
from dataclasses import dataclass, field
from typing import Any, Callable
from io import StringIO
import sys

from openai import OpenAI

from .config import RLMConfig
from .content_analyzer import (
    find_dead_code_regions,
    is_line_in_dead_code,
    verify_line_reference,
    check_implementation_status,
    annotate_content_with_dead_code,
    generate_confidence_guidance,
    DeadCodeRegion,
    Confidence,
)


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


class UnsafeCodeError(Exception):
    """Raised when code contains potentially unsafe constructs."""
    pass


def attempt_syntax_repair(code: str, error: SyntaxError) -> tuple[str | None, str]:
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

    # Handle unterminated triple-quoted strings
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

        if repairs:
            return code, "; ".join(repairs)

    # Handle unterminated regular strings (EOL while scanning)
    if "eol while scanning string literal" in error_msg or "unterminated string literal" in error_msg:
        lines = code.split('\n')
        if error.lineno and error.lineno <= len(lines):
            line = lines[error.lineno - 1]
            # Simple heuristic: if line has odd number of quotes, add one
            if line.count('"') % 2 == 1:
                lines[error.lineno - 1] = line + '"'
                return '\n'.join(lines), f'Added missing " at line {error.lineno}'
            elif line.count("'") % 2 == 1:
                lines[error.lineno - 1] = line + "'"
                return '\n'.join(lines), f"Added missing ' at line {error.lineno}"

    # Handle f-string with unmatched braces
    if "f-string" in error_msg:
        # Try to balance braces in f-strings
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
            # Check if it's a control structure missing colon
            control_keywords = ['if ', 'elif ', 'else', 'for ', 'while ', 'def ', 'class ', 'try', 'except', 'finally', 'with ']
            for kw in control_keywords:
                if line.lstrip().startswith(kw) and not line.endswith(':'):
                    lines[error.lineno - 1] = line + ':'
                    return '\n'.join(lines), f'Added missing : at line {error.lineno}'

    # Handle incomplete code blocks (truncated by LLM)
    # If the error is near the end and we have unclosed structures
    if error.lineno and error.lineno >= len(code.split('\n')) - 2:
        # Check for unclosed parentheses/brackets
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


# Modules that are pre-loaded in the REPL environment (safe to "import")
PRELOADED_MODULES = frozenset({'re'})

# Available built-in functions in the sandbox (for documentation)
AVAILABLE_BUILTINS = [
    "len", "str", "int", "float", "bool", "list", "dict", "tuple", "set",
    "range", "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "min", "max", "sum", "any", "all", "abs", "round", "print",
    "isinstance", "ord", "chr", "repr", "frozenset"
]


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
        self.syntax_error: SyntaxError | None = None
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


def validate_code(code: str) -> tuple[bool, list[str], SyntaxError | None, list[str]]:
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


@dataclass
class REPLState:
    """State of the REPL environment."""

    # The full content - stored as `prompt` variable (paper terminology)
    prompt: str = ""

    # All variables created during execution
    variables: dict[str, Any] = field(default_factory=dict)

    # History of all code executed
    code_history: list[str] = field(default_factory=list)

    # History of all outputs
    output_history: list[str] = field(default_factory=list)

    # Sub-LLM call results (preserved, NOT summarized)
    llm_responses: dict[str, str] = field(default_factory=dict)

    # Counter for auto-naming llm_query results
    query_counter: int = 0

    # Final answer (set when FINAL_ANSWER is assigned)
    final_answer: str | None = None

    # Dead code regions detected in the content (for validation)
    dead_code_regions: dict[str, list[DeadCodeRegion]] = field(default_factory=dict)


class RLMReplEnvironment:
    """
    Python REPL environment for RLM processing.

    Matches the paper's architecture exactly:
    - `prompt`: Variable containing full content
    - `llm_query(text)`: Function to call sub-LLM
    - `FINAL_ANSWER`: Variable to set the final answer
    - All Python string operations available
    """

    # System prompt matching the paper's approach - with anti-hallucination measures
    SYSTEM_PROMPT = '''You are an RLM (Recursive Language Model) analyzing REAL code via a Python REPL.

## CRITICAL: This is REAL content, NOT a simulation

The `prompt` variable contains ACTUAL source code files. You must:
- Extract and analyze REAL code from the prompt
- Report ONLY findings that exist in the actual content
- Include EXACT file paths, LINE NUMBERS, and code snippets from the prompt
- NEVER generate example/simulated/hypothetical findings
- ALWAYS assign a CONFIDENCE LEVEL to each finding

## Sandbox Environment - IMPORTANT

**Pre-loaded (no import needed):**
- `re` - Regular expressions (just use `re.findall(...)` directly)
- `prompt` - The full source code content

**Available builtins:**
len, str, int, float, bool, list, dict, tuple, set, range, enumerate, zip,
map, filter, sorted, reversed, min, max, sum, any, all, abs, round, print,
isinstance, ord, chr, repr, frozenset

**NOT available (will error):**
- `import` statements (except `import re` which is auto-handled)
- `open()`, `eval()`, `exec()`, `__import__()`
- File I/O operations
- Network operations

**Variable scope:**
- Variables persist across code blocks (e.g., `files` defined in one block is available in the next)
- ALWAYS define variables before using them in the SAME code block
- If you get NameError, check spelling and ensure the variable was defined in a previous successful execution

**Helper functions provided:**

1. `llm_query(text)` - Query a sub-LLM. Pass ACTUAL code snippets from prompt.
   The sub-LLM will analyze ONLY what you send it.

2. `extract_with_lines(filepath)` - Extract file content WITH line numbers.
   Returns formatted string: "1: first line\\n2: second line\\n..."

3. `verify_line(filepath, line_num, expected_pattern)` - VERIFY a line exists and contains expected content.
   Returns: dict with 'is_valid', 'actual_content', 'in_dead_code', 'confidence', 'reason'
   USE THIS before reporting any finding to avoid false positives!

4. `check_dead_code(filepath)` - Check if file has dead code regions (#if false, #if DEBUG, etc.)
   Returns: list of dead code regions with start_line, end_line, condition

5. `is_implemented(filepath, function_name)` - Check if a function is actually implemented (not a stub).
   Returns: dict with 'is_implemented', 'has_body', 'is_stub', 'confidence', 'reason'

6. `FINAL_ANSWER` - Set this to your final response with REAL findings.

## Content

`prompt` contains {char_count:,} chars of REAL source code (files concatenated).
Format: "### File: path/to/file.ext\\n<actual code>\\n### File: ..."

## CONFIDENCE LEVELS (REQUIRED)

Every finding MUST have a confidence level:

**[Confidence: HIGH]** - Use when:
- Code is in active (non-conditional) blocks
- Line verified with verify_line()
- Function has real implementation (check with is_implemented())

**[Confidence: MEDIUM]** - Use when:
- Code context unclear
- Cannot verify reachability
- Implementation status uncertain

**[Confidence: LOW]** - Use when:
- Code is in #if false, #if DEBUG blocks (check with check_dead_code())
- Line couldn't be verified
- Function is a stub or unimplemented
- Finding based on signature only

## CRITICAL OUTPUT FORMAT

All findings MUST include:
- **File:Line** format: `path/to/file.py:42`
- **[Confidence: HIGH/MEDIUM/LOW]** with reason if LOW
- **Code snippet**: The actual code from the file
- **Context**: What the issue is

Example finding format:
```
**src/auth.py:156** [Confidence: HIGH]
Issue: Hardcoded API key
```python
API_KEY = "sk-1234567890abcdef"  # Line 156
```

**src/legacy.swift:42** [Confidence: LOW - in #if false block]
Issue: Potential hardcoded secret (DEAD CODE - not compiled)
```swift
let key = "test-key"  # Line 42
```
```

## Required Workflow

1. **First, discover what files exist**:
   ```python
   # Find all actual file paths in the content
   files = re.findall(r'### File: ([^\\n]+)', prompt)
   print(f"Found {{len(files)}} files:")
   for f in files[:20]:
       print(f"  - {{f}}")
   ```

2. **Check for dead code regions BEFORE analyzing**:
   ```python
   # Check each file for conditional compilation blocks
   for f in files[:10]:
       dead_regions = check_dead_code(f)
       if dead_regions:
           print(f"{{f}}: {{len(dead_regions)}} dead code regions")
           for r in dead_regions:
               print(f"  Lines {{r['start_line']}}-{{r['end_line']}}: {{r['condition']}}")
   ```

3. **Extract files WITH LINE NUMBERS**:
   ```python
   # Use the helper to get line-numbered content
   content = extract_with_lines("path/to/file.py")
   print(content[:2000])  # Shows "1: line1\\n2: line2\\n..."
   ```

4. **VERIFY findings before reporting**:
   ```python
   # Always verify line references!
   result = verify_line("src/auth.py", 42, r"API_KEY|secret|password")
   if result['is_valid'] and not result['in_dead_code']:
       print(f"CONFIRMED: Line 42 contains: {{result['actual_content']}}")
       confidence = "HIGH"
   elif result['in_dead_code']:
       print(f"WARNING: Line 42 is in dead code block")
       confidence = "LOW"
   else:
       print(f"NOT FOUND: {{result['reason']}}")
   ```

5. **Check implementation status for function findings**:
   ```python
   # Don't report "function returns NotImplemented" if it's actually implemented
   status = is_implemented("src/api.py", "process_request")
   if status['is_stub']:
       print(f"WARNING: Function is a stub - {{status['reason']}}")
   elif status['is_implemented']:
       print(f"Function is fully implemented with {{status['body_lines']}} lines")
   ```

6. **Compile findings with confidence levels**:
   ```python
   FINAL_ANSWER = f"""## Analysis of {{len(files)}} files

   ### High Confidence Findings
   {{high_confidence_findings}}

   ### Medium Confidence Findings
   {{medium_confidence_findings}}

   ### Low Confidence Findings (verify manually)
   {{low_confidence_findings}}

   Files analyzed: {{', '.join(analyzed_files)}}"""
   ```

## Anti-Hallucination Rules

- ONLY report file paths that appear in `prompt`
- ONLY quote code that exists in `prompt`
- ALWAYS verify line numbers with verify_line() before reporting
- ALWAYS check for dead code with check_dead_code()
- ALWAYS assign confidence levels (HIGH/MEDIUM/LOW)
- If a finding is in dead code (#if false, etc.), mark as LOW confidence
- If no issues found, say "No issues found" - don't invent examples
- Every finding MUST have format: `filepath:line_number [Confidence: X]`

## Query
{query}

Start by discovering files, checking for dead code regions, then use verification helpers before reporting findings.'''

    def __init__(self, config: RLMConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/mosif16/RLM-Mem_MCP",
                "X-Title": "RLM-Mem MCP Server"
            }
        )
        self.state = REPLState()

    def initialize(self, content: str) -> None:
        """Initialize the REPL with content stored as `prompt` variable."""
        self.state = REPLState(prompt=content)
        self.state.variables["prompt"] = content
        self.state.variables["FINAL_ANSWER"] = None

    def _create_llm_query_function(self) -> Callable[[str], str]:
        """Create the llm_query function for sub-LLM calls."""

        # Get actual file paths from prompt for validation
        actual_files = set(self._extract_file_list())

        def llm_query(text: str, max_tokens: int = 4000) -> str:
            """
            Query a sub-LLM with content.

            Args:
                text: The prompt/content to send to sub-LLM
                max_tokens: Maximum response tokens

            Returns:
                The sub-LLM's response (stored in full, not summarized)
            """
            self.state.query_counter += 1
            query_id = f"llm_response_{self.state.query_counter}"

            # Enhanced prompt to prevent hallucination
            enhanced_text = f"""IMPORTANT: You are analyzing REAL code.
- Only report what you actually find in the provided content
- Include exact file paths, line references, and code snippets
- If nothing relevant is found, say "No relevant findings"
- NEVER generate example/hypothetical/simulated output

{text}"""

            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": enhanced_text}]
                )

                result = response.choices[0].message.content if response.choices else ""

                # Validation: Check for signs of hallucinated/simulated output
                hallucination_markers = [
                    "example:", "for example,", "hypothetically",
                    "let's say", "imagine", "suppose", "would be like",
                    "simulated", "demonstration", "sample output"
                ]
                result_lower = result.lower()
                for marker in hallucination_markers:
                    if marker in result_lower:
                        result = f"[WARNING: Response may contain simulated content]\n\n{result}"
                        break

                # Store FULL response (paper: variables as buffers)
                self.state.llm_responses[query_id] = result
                self.state.variables[query_id] = result

                return result

            except Exception as e:
                error_msg = f"llm_query error: {type(e).__name__}: {e}"
                self.state.llm_responses[query_id] = error_msg
                return error_msg

        return llm_query

    def _extract_file_list(self) -> list[str]:
        """Extract list of actual file paths from the prompt content."""
        files = re.findall(r'### File: ([^\n]+)', self.state.prompt)
        return files

    def _strip_preloaded_imports(self, code: str) -> str:
        """
        Remove import statements for pre-loaded modules.

        This allows LLM-generated code with `import re` to work,
        since `re` is already available in the sandbox.
        """
        lines = code.split('\n')
        filtered_lines = []

        for line in lines:
            stripped = line.strip()
            # Check for `import re` or `import re as ...`
            skip = False
            for module in PRELOADED_MODULES:
                if stripped == f"import {module}" or stripped.startswith(f"import {module} as "):
                    skip = True
                    break
                if stripped.startswith(f"from {module} import"):
                    skip = True
                    break

            if not skip:
                filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def _create_extract_with_lines_function(self) -> Callable[[str], str]:
        """Create the extract_with_lines helper function for line-numbered extraction."""

        def extract_with_lines(filepath: str, max_lines: int = 500) -> str:
            """
            Extract file content with line numbers.

            Args:
                filepath: Path to extract (partial match supported)
                max_lines: Maximum lines to return (default 500)

            Returns:
                Line-numbered content: "1: first line\\n2: second line\\n..."
            """
            # Find the file in prompt
            parts = self.state.prompt.split("### File:")

            for part in parts[1:]:  # Skip first empty part
                lines = part.split("\n")
                if not lines:
                    continue

                file_path = lines[0].strip()

                # Match if filepath is contained in or equals file_path
                if filepath in file_path or file_path.endswith(filepath):
                    # Get file content (skip the filename line)
                    content_lines = lines[1:]

                    # Add line numbers
                    numbered_lines = []
                    for i, line in enumerate(content_lines[:max_lines], 1):
                        numbered_lines.append(f"{i}: {line}")

                    result = "\n".join(numbered_lines)

                    if len(content_lines) > max_lines:
                        result += f"\n... ({len(content_lines) - max_lines} more lines)"

                    return result

            return f"File not found: {filepath}"

        return extract_with_lines

    def _create_verify_line_function(self) -> Callable[[str, int, str | None], dict]:
        """Create the verify_line helper function for line verification."""

        def verify_line(filepath: str, line_num: int, expected_pattern: str | None = None) -> dict:
            """
            Verify that a line number reference is valid and contains expected content.

            Args:
                filepath: Path to the file
                line_num: Line number to verify (1-indexed)
                expected_pattern: Optional regex pattern expected on that line

            Returns:
                dict with 'is_valid', 'actual_content', 'in_dead_code', 'confidence', 'reason'
            """
            # Find the file content
            content = None
            actual_filepath = None

            parts = self.state.prompt.split("### File:")
            for part in parts[1:]:
                lines = part.split("\n")
                if not lines:
                    continue
                file_path = lines[0].strip()
                if filepath in file_path or file_path.endswith(filepath):
                    content = "\n".join(lines[1:])
                    actual_filepath = file_path
                    break

            if content is None:
                return {
                    'is_valid': False,
                    'actual_content': None,
                    'in_dead_code': False,
                    'confidence': 'LOW',
                    'reason': f"File not found: {filepath}"
                }

            # Get or compute dead code regions for this file
            if actual_filepath not in self.state.dead_code_regions:
                self.state.dead_code_regions[actual_filepath] = find_dead_code_regions(content, actual_filepath)

            # Use the content analyzer
            result = verify_line_reference(
                content,
                actual_filepath,
                line_num,
                expected_pattern,
                self.state.dead_code_regions[actual_filepath]
            )

            return {
                'is_valid': result.is_valid,
                'actual_content': result.actual_content,
                'in_dead_code': result.in_dead_code,
                'confidence': result.confidence.value,
                'reason': result.reason
            }

        return verify_line

    def _create_check_dead_code_function(self) -> Callable[[str], list[dict]]:
        """Create the check_dead_code helper function for dead code detection."""

        def check_dead_code(filepath: str) -> list[dict]:
            """
            Check if a file has dead code regions (#if false, #if DEBUG, etc.)

            Args:
                filepath: Path to the file

            Returns:
                List of dead code regions with start_line, end_line, condition
            """
            # Find the file content
            content = None
            actual_filepath = None

            parts = self.state.prompt.split("### File:")
            for part in parts[1:]:
                lines = part.split("\n")
                if not lines:
                    continue
                file_path = lines[0].strip()
                if filepath in file_path or file_path.endswith(filepath):
                    content = "\n".join(lines[1:])
                    actual_filepath = file_path
                    break

            if content is None:
                return []

            # Get or compute dead code regions
            if actual_filepath not in self.state.dead_code_regions:
                self.state.dead_code_regions[actual_filepath] = find_dead_code_regions(content, actual_filepath)

            # Convert to dicts for REPL
            return [
                {
                    'start_line': r.start_line,
                    'end_line': r.end_line,
                    'condition': r.condition,
                    'language': r.language
                }
                for r in self.state.dead_code_regions[actual_filepath]
            ]

        return check_dead_code

    def _create_is_implemented_function(self) -> Callable[[str, str], dict]:
        """Create the is_implemented helper function for checking function implementation."""

        def is_implemented(filepath: str, function_name: str) -> dict:
            """
            Check if a function is actually implemented (not just a stub).

            Args:
                filepath: Path to the file
                function_name: Name of function to check

            Returns:
                dict with 'is_implemented', 'has_body', 'is_stub', 'body_lines', 'confidence', 'reason'
            """
            # Find the file content
            content = None
            actual_filepath = None

            parts = self.state.prompt.split("### File:")
            for part in parts[1:]:
                lines = part.split("\n")
                if not lines:
                    continue
                file_path = lines[0].strip()
                if filepath in file_path or file_path.endswith(filepath):
                    content = "\n".join(lines[1:])
                    actual_filepath = file_path
                    break

            if content is None:
                return {
                    'is_implemented': False,
                    'has_body': False,
                    'is_stub': False,
                    'body_lines': 0,
                    'confidence': 'LOW',
                    'reason': f"File not found: {filepath}"
                }

            # Use the content analyzer
            status = check_implementation_status(content, function_name, actual_filepath)

            return {
                'is_implemented': status.is_implemented,
                'has_body': status.has_body,
                'is_stub': status.is_stub,
                'body_lines': status.body_lines,
                'confidence': status.confidence.value,
                'reason': status.reason
            }

        return is_implemented

    def _create_safe_globals(self) -> dict[str, Any]:
        """Create a restricted execution environment."""
        # Allow safe builtins only - REMOVED dangerous ones:
        # - getattr, setattr, delattr: can bypass attribute restrictions
        # - type: can be used for metaclass attacks
        # - hasattr: internally uses getattr
        # - eval, exec, compile: code execution
        # - open, input: I/O operations
        # - __import__: module imports
        safe_builtins = {
            # Type constructors (safe)
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "frozenset": frozenset,
            # Iteration helpers (safe)
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            # Aggregation (safe)
            "min": min,
            "max": max,
            "sum": sum,
            "any": any,
            "all": all,
            "abs": abs,
            "round": round,
            # Output (safe)
            "print": print,
            # Type checking (safe - isinstance doesn't allow attribute access)
            "isinstance": isinstance,
            # String operations
            "ord": ord,
            "chr": chr,
            "repr": repr,
        }

        return {
            "__builtins__": safe_builtins,
            "prompt": self.state.prompt,
            "llm_query": self._create_llm_query_function(),
            "extract_with_lines": self._create_extract_with_lines_function(),
            "verify_line": self._create_verify_line_function(),
            "check_dead_code": self._create_check_dead_code_function(),
            "is_implemented": self._create_is_implemented_function(),
            "re": re,  # Regex support
            "FINAL_ANSWER": None,
            **self.state.variables
        }

    def execute_code(self, code: str, allow_repair: bool = True) -> tuple[str, bool]:
        """
        Execute Python code in the sandboxed REPL environment.

        Args:
            code: Python code to execute
            allow_repair: If True, attempt to repair common syntax errors

        Returns:
            (output, success) tuple
        """
        # SECURITY: Validate code with AST analysis BEFORE execution
        is_safe, validation_errors, syntax_error, hints = validate_code(code)

        # If we have hints (e.g., "re is already available"), include them
        hint_output = ""
        if hints:
            hint_output = "\n".join(f"[HINT] {h}" for h in hints) + "\n\n"

        if not is_safe:
            # Attempt syntax repair if it's a syntax error
            if syntax_error and allow_repair:
                repaired_code, repair_desc = attempt_syntax_repair(code, syntax_error)
                if repaired_code:
                    # Re-validate repaired code
                    is_repaired_safe, repaired_errors, _, repaired_hints = validate_code(repaired_code)
                    if is_repaired_safe:
                        # Recursively execute repaired code (but don't allow further repair)
                        output, success = self.execute_code(repaired_code, allow_repair=False)
                        if success:
                            return f"[Auto-repaired: {repair_desc}]\n{output}", True

            # Build detailed error message with the failed code shown
            error_msg = "Code execution failed.\n\n"
            error_msg += "**Errors:**\n" + "\n".join(f"  - {e}" for e in validation_errors)

            if syntax_error:
                error_msg += "\n\n**Syntax Error Details:**\n"
                error_msg += f"  Line {syntax_error.lineno}: {syntax_error.msg}\n"
                error_msg += "\n**Hint:** Check that all strings (especially triple-quoted ones) are properly closed."

            # Show the failed code for debugging (truncated if long)
            code_preview = code[:500] + "..." if len(code) > 500 else code
            error_msg += f"\n\n**Failed code:**\n```python\n{code_preview}\n```"

            # Add environment info to help LLM understand what's available
            error_msg += "\n\n**Available in sandbox:**\n"
            error_msg += f"  - Pre-loaded modules: {', '.join(sorted(PRELOADED_MODULES))}\n"
            error_msg += f"  - Builtins: {', '.join(AVAILABLE_BUILTINS[:15])}...\n"
            error_msg += "  - Special: prompt, llm_query(), extract_with_lines(), verify_line(), check_dead_code(), is_implemented()"

            self.state.output_history.append(error_msg)
            return hint_output + error_msg, False

        # If we have hints but code is valid, show hints before output
        if hint_output:
            # Strip the import statement that generated the hint and re-execute
            # This allows code with `import re` to work (since re is already available)
            code_without_imports = self._strip_preloaded_imports(code)
            if code_without_imports != code:
                return self.execute_code(code_without_imports, allow_repair=False)

        self.state.code_history.append(code)

        exec_globals = self._create_safe_globals()

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            exec(code, exec_globals)
            output = sys.stdout.getvalue()

            # Update variables from execution
            for key, value in exec_globals.items():
                if not key.startswith("_") and key not in ("prompt", "llm_query", "re"):
                    if not callable(value):
                        self.state.variables[key] = value

            # Check if FINAL_ANSWER was set
            if exec_globals.get("FINAL_ANSWER") is not None:
                self.state.final_answer = str(exec_globals["FINAL_ANSWER"])
                self.state.variables["FINAL_ANSWER"] = self.state.final_answer

            self.state.output_history.append(output)
            return output, True

        except NameError as e:
            # Provide helpful context for undefined variable errors
            error_output = f"**NameError:** {e}\n\n"
            error_output += "**Available variables:**\n"
            user_vars = [k for k in self.state.variables.keys()
                        if not k.startswith('_') and k not in ('prompt', 'FINAL_ANSWER')]
            if user_vars:
                error_output += f"  - {', '.join(user_vars[:20])}"
                if len(user_vars) > 20:
                    error_output += f" ... and {len(user_vars) - 20} more"
            else:
                error_output += "  (none yet - define variables before using them)"
            error_output += "\n\n**Tip:** Make sure to define variables before using them in the same code block."
            self.state.output_history.append(error_output)
            return error_output, False

        except Exception as e:
            # Generic error with code context
            error_output = f"**{type(e).__name__}:** {e}"
            self.state.output_history.append(error_output)
            return error_output, False

        finally:
            sys.stdout = old_stdout

    async def run_rlm_session(
        self,
        query: str,
        max_iterations: int = 25  # Increased from 15 to handle complex queries
    ) -> str:
        """
        Run a complete RLM session.

        The orchestrating LLM (Sonnet) writes code iteratively.
        Sub-queries use Haiku 4.5 via llm_query().

        Args:
            query: The question to answer
            max_iterations: Maximum code execution rounds

        Returns:
            The final answer
        """
        system = self.SYSTEM_PROMPT.format(
            query=query,
            char_count=len(self.state.prompt)
        )

        # Extract actual file list to ground the LLM in reality
        actual_files = self._extract_file_list()
        file_list_preview = "\n".join(f"  - {f}" for f in actual_files[:30])
        if len(actual_files) > 30:
            file_list_preview += f"\n  ... and {len(actual_files) - 30} more files"

        # Initial message showing REAL file paths (grounds the LLM)
        initial_msg = f"""REPL ready. `prompt` contains {len(self.state.prompt):,} characters of REAL source code.

## ACTUAL FILES IN PROMPT ({len(actual_files)} files):
{file_list_preview}

## Content Preview (first 500 chars):
```
{self.state.prompt[:500]}
```

These are REAL files. Analyze them to answer: {query}

Start by exploring the actual content with Python code."""

        messages = [{"role": "user", "content": initial_msg}]

        consecutive_failures = 0
        max_consecutive_failures = 3

        for iteration in range(max_iterations):
            # Orchestrating LLM generates code
            response = self.client.chat.completions.create(
                model=self.config.aggregator_model,
                max_tokens=6000,  # Increased from 4000 to reduce truncation
                messages=[
                    {"role": "system", "content": system},
                    *messages
                ]
            )

            assistant_msg = response.choices[0].message.content if response.choices else ""

            # Detect truncated response (ends mid-code block)
            if assistant_msg.count("```") % 2 == 1:
                # Odd number of backticks = unclosed code block = truncated
                messages.append({"role": "assistant", "content": assistant_msg})
                messages.append({
                    "role": "user",
                    "content": "Your response was truncated mid-code. Please complete the code block and continue."
                })
                continue

            # Extract code blocks
            code_blocks = self._extract_code_blocks(assistant_msg)

            if not code_blocks:
                # No code - check if we have a final answer
                if self.state.final_answer:
                    return self.state.final_answer

                # Check if assistant is asking for clarification or done
                if any(phrase in assistant_msg.lower() for phrase in ["final_answer", "no issues found", "analysis complete", "in conclusion"]):
                    return assistant_msg

                # No code and no answer - prompt to continue
                messages.append({"role": "assistant", "content": assistant_msg})
                messages.append({
                    "role": "user",
                    "content": "Please write Python code to analyze the content, or set FINAL_ANSWER with your findings."
                })
                continue

            # Execute code blocks
            outputs = []
            any_success = False
            for code in code_blocks:
                output, success = self.execute_code(code)
                outputs.append(output if output else "(no output)")
                if success:
                    any_success = True

                # Check if FINAL_ANSWER was set
                if self.state.final_answer:
                    return self.state.final_answer

            # Track consecutive failures for early termination
            if not any_success:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    # Too many failures - return partial results
                    partial = self.get_partial_results()
                    if partial and partial != "No partial results accumulated.":
                        return f"## Analysis Incomplete\n\n**Reason:** Code execution failed {consecutive_failures} times consecutively.\n\n**Partial findings (before failure):**\n\n{partial}"
                    return "Analysis failed - code execution errors. Try a more specific query.\n\n**Tip:** Check the error messages above for what went wrong."
            else:
                consecutive_failures = 0

            # Continue conversation
            execution_results = "\n---\n".join(outputs)
            messages.append({"role": "assistant", "content": assistant_msg})

            # Add progress hint if we're getting close to max iterations
            progress_hint = ""
            if iteration >= max_iterations - 5:
                progress_hint = f" (Iteration {iteration+1}/{max_iterations} - please wrap up soon)"

            messages.append({
                "role": "user",
                "content": f"Output:\n```\n{execution_results}\n```\n\nContinue or set FINAL_ANSWER.{progress_hint}"
            })

        # Return best available answer
        if self.state.final_answer:
            return self.state.final_answer

        # No final answer - gather partial results
        partial = self.get_partial_results()
        if partial and partial != "No partial results accumulated.":
            return f"## Analysis Reached Iteration Limit ({max_iterations})\n\n**Partial findings:**\n\n{partial}"

        # Fall back to last output
        if self.state.output_history:
            return self.state.output_history[-1]

        return "No answer generated - try a more specific query."

    def _extract_code_blocks(self, text: str) -> list[str]:
        """Extract Python code blocks from text."""
        pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches if m.strip()]

    def get_all_responses(self) -> dict[str, str]:
        """Get all stored sub-LLM responses (full, not summarized)."""
        return self.state.llm_responses.copy()

    def get_execution_history(self) -> list[tuple[str, str]]:
        """Get (code, output) history."""
        return list(zip(self.state.code_history, self.state.output_history))

    def get_partial_results(self) -> str:
        """
        Gather all accumulated findings for partial result reporting.

        Returns a formatted string with:
        - Sub-LLM responses (the actual analysis results)
        - Successful execution outputs
        - Variables that might contain findings
        """
        parts = []

        # Include sub-LLM responses (these are the actual findings)
        if self.state.llm_responses:
            parts.append("## Sub-LLM Analysis Results\n")
            for query_id, response in self.state.llm_responses.items():
                # Skip error responses and empty results
                if "error:" in response.lower() or not response.strip():
                    continue
                # Skip "no findings" type responses
                if any(phrase in response.lower() for phrase in ["no relevant", "no issues", "nothing found", "no findings"]):
                    continue
                parts.append(f"### {query_id}\n{response}\n")

        # Include successful outputs that look like findings
        finding_keywords = ["found", "issue", "vulnerability", "error", "warning", "file:", "line"]
        for output in self.state.output_history:
            if not output or "(no output)" in output:
                continue
            # Skip error messages
            if output.startswith("Code execution failed") or output.startswith("Error:"):
                continue
            # Include if it looks like it contains findings
            if any(kw in output.lower() for kw in finding_keywords):
                if output not in parts:  # Avoid duplicates
                    parts.append(output)

        # Include any variables that look like findings
        for var_name, value in self.state.variables.items():
            if var_name.startswith("_") or var_name in ("prompt", "FINAL_ANSWER"):
                continue
            if isinstance(value, str) and len(value) > 50:
                # Check if it looks like findings
                if any(kw in value.lower() for kw in finding_keywords):
                    parts.append(f"\n## Variable: {var_name}\n{value[:2000]}")

        if not parts:
            return "No partial results accumulated."

        return "\n\n---\n\n".join(parts)

