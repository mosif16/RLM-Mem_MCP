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
from .fallback_analyzer import FallbackAnalyzer


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
    if "eol while scanning string literal" in error_msg:
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
        # Replace \n inside f-strings with a workaround
        # This is a simplified fix - complex cases may still fail
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if 'f"' in line or "f'" in line:
                # Extract f-string parts and fix backslashes
                # Simple approach: suggest using a variable
                if '\\n' in line or '\\t' in line:
                    lines[i] = f"# FIXME: f-string with backslash needs refactoring\n{line}"
                    repairs.append(f"Marked problematic f-string at line {i+1}")
        code = '\n'.join(lines)

    # 7. Handle indentation errors
    if "unexpected indent" in error_msg or "expected an indented block" in error_msg:
        lines = code.split('\n')
        if error.lineno and error.lineno <= len(lines):
            # Try to fix by normalizing indentation
            # This is a heuristic - may not always work
            repaired = False
            if "expected an indented block" in error_msg:
                # Add a pass statement if block is empty
                prev_line = lines[error.lineno - 2] if error.lineno > 1 else ""
                if prev_line.rstrip().endswith(':'):
                    indent = len(prev_line) - len(prev_line.lstrip()) + 4
                    lines.insert(error.lineno - 1, ' ' * indent + 'pass  # Auto-added')
                    code = '\n'.join(lines)
                    repairs.append(f'Added pass statement at line {error.lineno}')
                    repaired = True

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
        if not code.endswith(':'):  # Don't break valid code
            repairs.append("Removed trailing ellipsis")

    if repairs:
        return code, "; ".join(repairs)

    return None, "Could not auto-repair"


class CodeValidator(ast.NodeVisitor):
    """
    AST validator to detect potentially dangerous code patterns.

    Blocks:
    - Access to dunder attributes (__class__, __globals__, etc.)
    - Dangerous function calls (eval, exec, open, etc.)
    - Import statements
    - Attribute chains that could escape sandbox
    """

    def __init__(self):
        self.errors: list[str] = []
        self.syntax_error: SyntaxError | None = None

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
        """Block import statements."""
        names = ', '.join(alias.name for alias in node.names)
        self.errors.append(f"Import statements not allowed: 'import {names}'")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Block from...import statements."""
        self.errors.append(f"Import statements not allowed: 'from {node.module} import ...'")

    def validate(self, code: str) -> tuple[bool, list[str]]:
        """
        Validate code for safety.

        Returns:
            (is_safe, errors) tuple
        """
        self.errors = []
        self.syntax_error = None

        try:
            tree = ast.parse(code)
            self.visit(tree)
        except SyntaxError as e:
            self.syntax_error = e
            self.errors.append(f"Syntax error: {e}")

        return len(self.errors) == 0, self.errors


def validate_code(code: str) -> tuple[bool, list[str], SyntaxError | None]:
    """
    Validate Python code for sandbox safety.

    Args:
        code: Python source code to validate

    Returns:
        (is_safe, errors, syntax_error) tuple - syntax_error is set if a SyntaxError occurred
    """
    validator = CodeValidator()
    is_safe, errors = validator.validate(code)
    return is_safe, errors, validator.syntax_error


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

    @staticmethod
    def sanitize_query(query: str) -> str:
        """
        Sanitize user query to prevent format string issues and clean markdown.

        This prevents:
        - Curly braces breaking .format() calls
        - Markdown formatting confusing the LLM code generation
        - Special characters causing syntax errors in generated code
        """
        # Escape curly braces to prevent .format() issues
        query = query.replace("{", "{{").replace("}", "}}")

        # Convert markdown formatting to plain text equivalents
        import re
        # Remove bold/italic markers
        query = re.sub(r'\*\*([^*]+)\*\*', r'\1', query)  # **bold** -> bold
        query = re.sub(r'\*([^*]+)\*', r'\1', query)      # *italic* -> italic
        query = re.sub(r'__([^_]+)__', r'\1', query)      # __bold__ -> bold
        query = re.sub(r'_([^_]+)_', r'\1', query)        # _italic_ -> italic

        # Convert markdown headers to plain text
        query = re.sub(r'^#+\s*', '', query, flags=re.MULTILINE)  # # Header -> Header

        # Remove backticks but keep content
        query = re.sub(r'`([^`]+)`', r'\1', query)  # `code` -> code
        query = re.sub(r'```[^\n]*\n?', '', query)  # Remove code fence markers

        # Normalize whitespace
        query = re.sub(r'\n{3,}', '\n\n', query)  # Max 2 newlines

        return query.strip()

    # System prompt matching the paper's approach - with anti-hallucination measures
    SYSTEM_PROMPT = '''You are an RLM (Recursive Language Model) analyzing REAL code via a Python REPL.

## CRITICAL: This is REAL content, NOT a simulation

The `prompt` variable contains ACTUAL source code files. You must:
- Extract and analyze REAL code from the prompt
- Report ONLY findings that exist in the actual content
- Include EXACT file paths, LINE NUMBERS, and code snippets from the prompt
- NEVER generate example/simulated/hypothetical findings
- ALWAYS assign a CONFIDENCE LEVEL to each finding

## CRITICAL: Code Execution Rules

**ALWAYS initialize variables before using them.** Each code block runs independently.

WRONG (will cause NameError):
```python
findings.append(x)  # NameError: 'findings' not defined!
```

CORRECT:
```python
findings = []  # Initialize first!
findings.append(x)
```

**Common variables you MUST initialize:**
- `findings = []` before appending findings
- `output = ""` before building output strings
- `results = []` before collecting results
- `files = []` before storing file lists

## Environment

1. `prompt` - Contains {char_count:,} chars of REAL source code (files concatenated)
   Format: "### File: path/to/file.ext\\n<actual code>\\n### File: ..."

2. `llm_query(text)` - Query a sub-LLM. Pass ACTUAL code snippets from prompt.
   The sub-LLM will analyze ONLY what you send it.

3. `extract_with_lines(filepath)` - Extract file content WITH line numbers.
   Returns formatted string: "1: first line\\n2: second line\\n..."

4. `verify_line(filepath, line_num, expected_pattern)` - VERIFY a line exists and contains expected content.
   Returns: dict with 'is_valid', 'actual_content', 'in_dead_code', 'confidence', 'reason'
   USE THIS before reporting any finding to avoid false positives!

5. `check_dead_code(filepath)` - Check if file has dead code regions (#if false, #if DEBUG, etc.)
   Returns: list of dead code regions with start_line, end_line, condition

6. `is_implemented(filepath, function_name)` - Check if a function is actually implemented (not a stub).
   Returns: dict with 'is_implemented', 'has_body', 'is_stub', 'confidence', 'reason'

7. `batch_verify(findings)` - EFFICIENT batch verification for multiple findings.
   Pass a list of {{'file': path, 'line': num, 'pattern': regex}} dicts.
   Returns: list of verification results (more efficient than calling verify_line repeatedly!)
   USE THIS when you have 5+ findings to verify.

8. `find_swift_issues(file_path, issue_types)` - Swift-specific issue finder.
   issue_types: ["retain_cycle", "force_unwrap", "actor_isolation", "swiftui"]
   Returns: list of issues with file, line, type, description, confidence
   Example: `issues = find_swift_issues("PaywallView.swift", ["retain_cycle", "force_unwrap"])`

9. `analyze_file(file_path, analysis_type)` - Deep analysis using sub-LLM.
   analysis_type: "security", "quality", or "architecture"
   Returns: Detailed analysis from sub-LLM
   Example: `analysis = analyze_file("AuthManager.swift", "security")`

10. `search_pattern(pattern, file_filter)` - Fast regex search across all files.
    file_filter: Optional extension filter like ".swift"
    Returns: list of {{'file', 'line', 'content', 'match'}} dicts
    Example: `matches = search_pattern(r'api.?key', '.swift')`

11. `FINAL_ANSWER` - Set this to your final response with REAL findings.

## PRE-INITIALIZED VARIABLES (ready to use)

These variables are already initialized - you can append to them directly:
- `findings = []` - Append your findings here
- `results = []` - Append results here
- `issues = []` - Append issues here
- `files = []` - Store file paths here
- `output = ""` - Build output strings here

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
                    # Original markers
                    "example:", "for example,", "hypothetically",
                    "let's say", "imagine", "suppose", "would be like",
                    "simulated", "demonstration", "sample output",
                    # Extended markers for better detection
                    "might look like", "could be something like",
                    "typical implementation", "usually looks like",
                    "generic", "placeholder", "dummy",
                    "here's what", "here is what", "would typically",
                    "commonly seen", "a common pattern", "pseudo",
                    "illustrative", "conceptual", "theoretical"
                ]
                result_lower = result.lower()
                has_hallucination_warning = False
                for marker in hallucination_markers:
                    if marker in result_lower:
                        result = f"[WARNING: Response may contain simulated content]\n\n{result}"
                        has_hallucination_warning = True
                        break

                # Validate line number references - flag suspiciously high or invalid ones
                line_refs = re.findall(r':(\d+)\b', result)
                invalid_lines = []
                for line_ref in line_refs:
                    line_num = int(line_ref)
                    # Flag suspiciously high line numbers (most files < 5000 lines)
                    if line_num > 5000:
                        invalid_lines.append(line_num)

                if invalid_lines and not has_hallucination_warning:
                    result = f"[VERIFY: Contains high line numbers that may be inaccurate: {invalid_lines[:5]}]\n\n{result}"

                # Check for file paths that don't exist in actual files
                file_refs = re.findall(r'([a-zA-Z0-9_/\-\.]+\.[a-z]{1,5}):', result)
                unverified_files = []
                for file_ref in file_refs:
                    # Check if this file path exists in our actual files (fuzzy match)
                    if not any(file_ref in actual_file or actual_file.endswith(file_ref) for actual_file in actual_files):
                        unverified_files.append(file_ref)

                if unverified_files and len(unverified_files) <= 5:
                    unique_unverified = list(set(unverified_files))[:3]
                    result += f"\n\n[NOTE: These file references could not be verified: {unique_unverified}]"

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
        import re
        files = re.findall(r'### File: ([^\n]+)', self.state.prompt)
        return files

    def _build_categorized_file_index(self, files: list[str]) -> str:
        """
        Build a categorized file index for better LLM context.

        Categories files by type to help LLM find relevant files quickly.
        """
        categories = {
            "Views/UI": [],
            "Models/Data": [],
            "Services/Managers": [],
            "Extensions/Widgets": [],
            "Tests": [],
            "Config": [],
            "Other": [],
        }

        for f in files:
            f_lower = f.lower()
            categorized = False

            # Check for extensions/widgets first (high priority)
            if "extension" in f_lower or "widget" in f_lower:
                categories["Extensions/Widgets"].append(f)
                categorized = True
            elif "view" in f_lower or "controller" in f_lower or "screen" in f_lower or "ui" in f_lower:
                categories["Views/UI"].append(f)
                categorized = True
            elif "model" in f_lower or "entity" in f_lower or "schema" in f_lower:
                categories["Models/Data"].append(f)
                categorized = True
            elif "service" in f_lower or "manager" in f_lower or "provider" in f_lower or "handler" in f_lower:
                categories["Services/Managers"].append(f)
                categorized = True
            elif "test" in f_lower or "spec" in f_lower or "_test" in f_lower:
                categories["Tests"].append(f)
                categorized = True
            elif any(ext in f for ext in [".json", ".yaml", ".yml", ".toml", ".plist", ".xml", ".env"]):
                categories["Config"].append(f)
                categorized = True

            if not categorized:
                categories["Other"].append(f)

        # Build formatted output
        parts = []
        for category, cat_files in categories.items():
            if cat_files:
                parts.append(f"\n### {category} ({len(cat_files)} files)")
                # Show up to 25 files per category
                for f in cat_files[:25]:
                    parts.append(f"  - {f}")
                if len(cat_files) > 25:
                    parts.append(f"  ... and {len(cat_files) - 25} more in this category")

        return "\n".join(parts)

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

    def _create_batch_verify_function(self) -> Callable[[list[dict]], list[dict]]:
        """Create the batch_verify helper function for efficient multi-finding verification."""

        def batch_verify(findings: list[dict]) -> list[dict]:
            """
            Verify multiple findings at once (more efficient than one-by-one).

            Args:
                findings: List of dicts with 'file', 'line', and optionally 'pattern'

            Returns:
                List of verification results with 'is_valid', 'actual_content', 'in_dead_code', 'confidence', 'reason'
            """
            results = []

            # Group by file for efficiency
            from collections import defaultdict
            by_file = defaultdict(list)
            for i, f in enumerate(findings):
                by_file[f.get('file', '')].append((i, f))

            # Process each file once
            parts = self.state.prompt.split("### File:")

            for filepath, file_findings in by_file.items():
                # Find file content
                content = None
                actual_filepath = None

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
                    # File not found - mark all findings for this file as invalid
                    for idx, finding in file_findings:
                        results.append({
                            'index': idx,
                            'is_valid': False,
                            'actual_content': None,
                            'in_dead_code': False,
                            'confidence': 'LOW',
                            'reason': f"File not found: {filepath}"
                        })
                    continue

                # Get dead code regions once per file
                if actual_filepath not in self.state.dead_code_regions:
                    self.state.dead_code_regions[actual_filepath] = find_dead_code_regions(content, actual_filepath)

                dead_regions = self.state.dead_code_regions[actual_filepath]
                content_lines = content.split('\n')

                # Verify each finding in this file
                for idx, finding in file_findings:
                    line_num = finding.get('line', 0)
                    pattern = finding.get('pattern')

                    if line_num < 1 or line_num > len(content_lines):
                        results.append({
                            'index': idx,
                            'is_valid': False,
                            'actual_content': None,
                            'in_dead_code': False,
                            'confidence': 'LOW',
                            'reason': f"Line {line_num} out of range (file has {len(content_lines)} lines)"
                        })
                        continue

                    actual_content = content_lines[line_num - 1]
                    in_dead_code, condition = is_line_in_dead_code(line_num, dead_regions)

                    # Check pattern if provided
                    pattern_matches = True
                    if pattern:
                        pattern_matches = bool(re.search(pattern, actual_content, re.IGNORECASE))

                    # Determine confidence
                    if in_dead_code:
                        confidence = 'LOW'
                        reason = f"Line is in dead code block ({condition})"
                    elif not pattern_matches and pattern:
                        confidence = 'LOW'
                        reason = "Pattern not found on line"
                    else:
                        confidence = 'HIGH'
                        reason = "Verified in active code"

                    results.append({
                        'index': idx,
                        'is_valid': pattern_matches,
                        'actual_content': actual_content,
                        'in_dead_code': in_dead_code,
                        'confidence': confidence,
                        'reason': reason
                    })

            # Sort by original index
            results.sort(key=lambda x: x.get('index', 0))
            return results

        return batch_verify

    def _create_swift_analyzer_function(self) -> Callable:
        """Create a Swift-specific issue finder."""
        prompt = self.state.prompt
        llm_query = self._create_llm_query_function()

        def find_swift_issues(file_path: str, issue_types: list[str] | None = None) -> list[dict]:
            """
            Find Swift-specific issues in a file.

            Args:
                file_path: Path to the Swift file
                issue_types: Optional list of issue types to check:
                    - "retain_cycle": Closures missing [weak self]
                    - "force_unwrap": Force unwraps (!)
                    - "actor_isolation": @MainActor and Sendable issues
                    - "swiftui": SwiftUI lifecycle issues

            Returns:
                List of issues with file, line, type, description, confidence
            """
            # Extract the file content
            pattern = f"### File: [^\\n]*{re.escape(file_path.split('/')[-1])}[^\\n]*\\n"
            match = re.search(pattern, prompt)
            if not match:
                return [{"error": f"File not found: {file_path}"}]

            start = match.end()
            end_match = re.search(r"\n### File:", prompt[start:])
            end = start + end_match.start() if end_match else len(prompt)
            content = prompt[start:end]

            issues = []
            lines = content.split('\n')

            # Default to all issue types
            if not issue_types:
                issue_types = ["retain_cycle", "force_unwrap", "actor_isolation", "swiftui"]

            for line_num, line in enumerate(lines, 1):
                # Retain cycle detection
                if "retain_cycle" in issue_types:
                    if re.search(r'\{\s*(?!\[(?:weak|unowned)\s+self\]).*\bself\.', line):
                        if not re.search(r'\[weak\s+self\]|\[unowned\s+self\]', line):
                            issues.append({
                                "file": file_path,
                                "line": line_num,
                                "type": "retain_cycle",
                                "code": line.strip(),
                                "description": "Closure captures self - consider [weak self]",
                                "confidence": "MEDIUM"
                            })

                # Force unwrap detection (excluding !=)
                if "force_unwrap" in issue_types:
                    if re.search(r'\w!(?!=)\s*[.\[]', line) or re.search(r'\btry!\s+', line):
                        issues.append({
                            "file": file_path,
                            "line": line_num,
                            "type": "force_unwrap",
                            "code": line.strip(),
                            "description": "Force unwrap or try! - handle optionals safely",
                            "confidence": "HIGH"
                        })

                # Actor isolation
                if "actor_isolation" in issue_types:
                    if re.search(r'DispatchQueue\.main\.async', line):
                        issues.append({
                            "file": file_path,
                            "line": line_num,
                            "type": "actor_isolation",
                            "code": line.strip(),
                            "description": "Consider @MainActor instead of DispatchQueue.main",
                            "confidence": "LOW"
                        })

                # SwiftUI lifecycle
                if "swiftui" in issue_types:
                    if re.search(r'@ObservedObject\s+var\s+\w+\s*=', line):
                        issues.append({
                            "file": file_path,
                            "line": line_num,
                            "type": "swiftui",
                            "code": line.strip(),
                            "description": "@ObservedObject with default - use @StateObject",
                            "confidence": "HIGH"
                        })

            return issues

        return find_swift_issues

    def _create_file_analyzer_function(self) -> Callable:
        """Create a general file analyzer using sub-LLM with intelligent chunking."""
        llm_query = self._create_llm_query_function()
        extract_with_lines = self._create_extract_with_lines_function()

        # Maximum chars per chunk for sub-LLM context (conservative to leave room for prompts)
        MAX_CHUNK_CHARS = 6000
        # Overlap between chunks to maintain context
        CHUNK_OVERLAP_LINES = 20

        def chunk_content(content: str, max_chars: int = MAX_CHUNK_CHARS) -> list[tuple[str, int, int]]:
            """
            Split content into overlapping chunks for analysis.

            Args:
                content: Line-numbered content to chunk
                max_chars: Maximum characters per chunk

            Returns:
                List of (chunk_content, start_line, end_line) tuples
            """
            if len(content) <= max_chars:
                # Content fits in one chunk
                lines = content.split('\n')
                return [(content, 1, len(lines))]

            chunks = []
            lines = content.split('\n')
            current_chunk_lines = []
            current_chunk_chars = 0
            chunk_start_line = 1

            for i, line in enumerate(lines, 1):
                line_with_newline = line + '\n'

                if current_chunk_chars + len(line_with_newline) > max_chars and current_chunk_lines:
                    # Save current chunk
                    chunk_content = '\n'.join(current_chunk_lines)
                    chunk_end_line = i - 1
                    chunks.append((chunk_content, chunk_start_line, chunk_end_line))

                    # Start new chunk with overlap
                    overlap_start = max(0, len(current_chunk_lines) - CHUNK_OVERLAP_LINES)
                    current_chunk_lines = current_chunk_lines[overlap_start:]
                    chunk_start_line = i - len(current_chunk_lines)
                    current_chunk_chars = sum(len(l) + 1 for l in current_chunk_lines)

                current_chunk_lines.append(line)
                current_chunk_chars += len(line_with_newline)

            # Don't forget the last chunk
            if current_chunk_lines:
                chunk_content = '\n'.join(current_chunk_lines)
                chunks.append((chunk_content, chunk_start_line, len(lines)))

            return chunks

        def analyze_file(file_path: str, analysis_type: str = "security") -> str:
            """
            Analyze a file using the sub-LLM for semantic understanding.

            For large files, automatically chunks content and aggregates results.

            Args:
                file_path: Path to the file to analyze
                analysis_type: Type of analysis - "security", "quality", "architecture"

            Returns:
                Analysis results from sub-LLM (aggregated if chunked)
            """
            content = extract_with_lines(file_path)
            if not content or "not found" in content.lower():
                return f"File not found: {file_path}"

            # Get chunks (may be just one for small files)
            chunks = chunk_content(content)

            prompts_template = {
                "security": """Analyze this code for security issues:
{content}

Look for: hardcoded secrets, injection vulnerabilities, authentication issues, data exposure.
For each issue: specify exact line number, code snippet, severity (HIGH/MEDIUM/LOW).
Note: This is lines {start_line}-{end_line} of the file.""",

                "quality": """Analyze this code for quality issues:
{content}

Look for: complex functions, missing error handling, code duplication, unclear naming.
For each issue: specify exact line number and recommendation.
Note: This is lines {start_line}-{end_line} of the file.""",

                "architecture": """Analyze this code for architectural issues:
{content}

Look for: tight coupling, missing abstractions, violation of SOLID principles.
Describe the file's role and any concerns.
Note: This is lines {start_line}-{end_line} of the file."""
            }

            template = prompts_template.get(analysis_type, prompts_template["security"])

            if len(chunks) == 1:
                # Single chunk - simple case
                chunk_content_str, start_line, end_line = chunks[0]
                prompt = template.format(
                    content=chunk_content_str,
                    start_line=start_line,
                    end_line=end_line
                )
                return llm_query(prompt)

            # Multiple chunks - analyze each and aggregate
            all_findings = []
            for i, (chunk_content_str, start_line, end_line) in enumerate(chunks, 1):
                prompt = template.format(
                    content=chunk_content_str,
                    start_line=start_line,
                    end_line=end_line
                )
                chunk_result = llm_query(prompt)

                # Only include if there are actual findings
                if chunk_result and "no " not in chunk_result.lower()[:50]:
                    all_findings.append(f"### Chunk {i} (lines {start_line}-{end_line}):\n{chunk_result}")

            if not all_findings:
                return f"No {analysis_type} issues found in {file_path}"

            # Aggregate findings
            if len(all_findings) == 1:
                return all_findings[0]

            aggregated = f"## Analysis of {file_path} ({len(chunks)} chunks analyzed)\n\n"
            aggregated += "\n\n".join(all_findings)

            # If many findings, ask LLM to deduplicate
            if len(all_findings) > 2:
                dedup_prompt = f"""Deduplicate and organize these findings from analyzing {file_path}:

{aggregated}

Remove duplicates (same line, same issue). Keep all unique findings with their line numbers."""

                return llm_query(dedup_prompt, max_tokens=4000)

            return aggregated

        return analyze_file

    def _create_pattern_search_function(self) -> Callable:
        """Create a pattern search function that returns structured results."""
        prompt = self.state.prompt

        def search_pattern(pattern: str, file_filter: str | None = None) -> list[dict]:
            """
            Search for a regex pattern across all files.

            Args:
                pattern: Regex pattern to search for
                file_filter: Optional file extension filter (e.g., ".swift")

            Returns:
                List of matches with file, line, content
            """
            results = []
            current_file = None
            line_num = 0

            for line in prompt.split('\n'):
                if line.startswith("### File:"):
                    current_file = line.replace("### File:", "").strip()
                    line_num = 0
                    continue

                line_num += 1

                if current_file and (not file_filter or current_file.endswith(file_filter)):
                    try:
                        if re.search(pattern, line, re.IGNORECASE):
                            results.append({
                                "file": current_file,
                                "line": line_num,
                                "content": line.strip()[:200],
                                "match": re.search(pattern, line, re.IGNORECASE).group(0)
                            })
                    except re.error:
                        pass

            return results[:100]  # Limit results

        return search_pattern

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

        # Pre-initialize common variables to prevent NameError
        pre_initialized = {
            "findings": [],
            "results": [],
            "output": "",
            "issues": [],
            "files": [],
            "errors": [],
        }

        # Merge with existing state variables (state takes precedence)
        all_variables = {**pre_initialized, **self.state.variables}

        return {
            "__builtins__": safe_builtins,
            "prompt": self.state.prompt,
            "llm_query": self._create_llm_query_function(),
            "extract_with_lines": self._create_extract_with_lines_function(),
            "verify_line": self._create_verify_line_function(),
            "check_dead_code": self._create_check_dead_code_function(),
            "is_implemented": self._create_is_implemented_function(),
            "batch_verify": self._create_batch_verify_function(),
            # Swift-specific helpers
            "find_swift_issues": self._create_swift_analyzer_function(),
            "analyze_file": self._create_file_analyzer_function(),
            "search_pattern": self._create_pattern_search_function(),
            "re": re,  # Regex support
            "FINAL_ANSWER": None,
            **all_variables
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
        is_safe, validation_errors, syntax_error = validate_code(code)

        if not is_safe:
            # Attempt syntax repair if it's a syntax error
            if syntax_error and allow_repair:
                repaired_code, repair_desc = attempt_syntax_repair(code, syntax_error)
                if repaired_code:
                    # Re-validate repaired code
                    is_repaired_safe, repaired_errors, _ = validate_code(repaired_code)
                    if is_repaired_safe:
                        # Recursively execute repaired code (but don't allow further repair)
                        output, success = self.execute_code(repaired_code, allow_repair=False)
                        if success:
                            return f"[Auto-repaired: {repair_desc}]\n{output}", True

            # Differentiate between syntax errors and security violations
            if syntax_error and all("Syntax error" in e for e in validation_errors):
                error_msg = "Code has syntax errors (LLM generated invalid Python):\n"
                error_msg += "\n".join(f"  - {e}" for e in validation_errors)
                error_msg += "\n\nThis is a code generation issue, not a security issue."
                error_msg += "\nHint: The LLM may have included markdown or special characters in the code."
            elif syntax_error:
                error_msg = "Code validation failed:\n" + "\n".join(f"  - {e}" for e in validation_errors)
                error_msg += "\n\nNote: Some issues are syntax errors, others may be security-related."
            else:
                error_msg = "Code rejected for security reasons:\n" + "\n".join(f"  - {e}" for e in validation_errors)
            self.state.output_history.append(error_msg)
            return error_msg, False

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

        except Exception as e:
            error_output = f"Error: {type(e).__name__}: {e}"
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
        # Sanitize query to prevent format string issues and clean markdown
        safe_query = self.sanitize_query(query)

        system = self.SYSTEM_PROMPT.format(
            query=safe_query,
            char_count=len(self.state.prompt)
        )

        # Extract actual file list to ground the LLM in reality
        actual_files = self._extract_file_list()

        # Build categorized file index for better discovery
        categorized_index = self._build_categorized_file_index(actual_files)

        # Initial message showing REAL file paths (grounds the LLM)
        initial_msg = f"""REPL ready. `prompt` contains {len(self.state.prompt):,} characters of REAL source code.

## CATEGORIZED FILE INDEX ({len(actual_files)} files):
{categorized_index}

## Quick File Search
To find specific files, use: `[f for f in files if 'keyword' in f.lower()]`
Example: `[f for f in files if 'widget' in f.lower() or 'extension' in f.lower()]`

## Important Patterns to Check
- **Extensions/Widgets**: App extensions, widget targets, share extensions
- **Paywall/Subscription**: Payment, subscription, StoreKit files
- **Auth/Security**: Login, authentication, session management

## Content Preview (first 500 chars):
```
{self.state.prompt[:500]}
```

These are REAL files. Analyze them to answer: {safe_query}

Start by exploring the actual content with Python code. Use the categorized index above to find relevant files."""

        messages = [{"role": "user", "content": initial_msg}]

        consecutive_failures = 0
        max_consecutive_failures = 3

        for iteration in range(max_iterations):
            # Log progress to stderr so users see activity
            import sys
            print(f"[REPL] Iteration {iteration + 1}/{max_iterations} - Generating code...", file=sys.stderr)

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
            print(f"[REPL] Iteration {iteration + 1} - Executing {len(code_blocks)} code block(s)...", file=sys.stderr)
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
                    # Too many failures - use fallback analyzer
                    return self._run_fallback_analysis(
                        query,
                        f"Code execution failed {consecutive_failures} times consecutively"
                    )
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
        if self.state.output_history:
            return self.state.output_history[-1]

        # No answer from REPL - use fallback
        return self._run_fallback_analysis(query, "Max iterations reached without result")

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

    def _run_fallback_analysis(self, query: str, reason: str) -> str:
        """
        Run fallback pattern-based analysis when REPL execution fails.

        This provides graceful degradation - always returns useful results
        even when the LLM code execution doesn't work.

        Args:
            query: The original query
            reason: Why we're falling back

        Returns:
            Markdown-formatted analysis results
        """
        fallback = FallbackAnalyzer()
        result = fallback.analyze(
            content=self.state.prompt,
            query=query,
            fallback_reason=reason
        )

        # Build output
        output_parts = [result.to_markdown()]

        # Add any successful LLM sub-responses (these are valuable even if REPL failed)
        if self.state.llm_responses:
            output_parts.append("\n\n---\n## 📋 Partial Analysis (LLM sub-responses before failure)")
            for key, response in list(self.state.llm_responses.items())[:5]:
                if response and len(response) > 50:  # Only include substantive responses
                    output_parts.append(f"\n**{key[:60]}:**")
                    output_parts.append(f"```\n{response[:800]}\n```")

        # Add any partial execution output
        successful_outputs = [o for o in self.state.output_history if o.strip() and "Error:" not in o[:20]]
        if successful_outputs:
            output_parts.append("\n\n---\n## Successful REPL Executions (before failure)")
            for i, output in enumerate(successful_outputs[-3:], 1):
                output_parts.append(f"\n**Execution {i}:**\n```\n{output[:500]}\n```")

        return "\n".join(output_parts)

