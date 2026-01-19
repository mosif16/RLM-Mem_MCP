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
from .structured_tools import StructuredTools, ToolResult, Finding, Confidence as ToolConfidence, Severity
from .result_verifier import (
    QueryResultVerifier,
    QueryVerificationResult,
    VerificationStatus,
    verify_query_results,
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

    # System prompt - emphasizes using STRUCTURED TOOLS and VERIFICATION
    SYSTEM_PROMPT = '''You are an RLM analyzing REAL code via Python REPL with STRUCTURED TOOLS.

## CRITICAL: VERIFICATION REQUIRED

Before setting FINAL_ANSWER, you MUST verify your results:
```python
result = find_secrets()  # or other tool
verification = verify_results("{query}", result.to_markdown())
if verification.status.value == "FAILED":
    print(verification.guidance)  # See what's missing, try again
else:
    FINAL_ANSWER = result.to_markdown()
```

Results that fail verification will be REJECTED. Requirements:
- File:line references (e.g., `auth.py:42`)
- Actual code snippets showing the issue
- Confidence levels (HIGH/MEDIUM/LOW)
- Results must match query intent

## USE STRUCTURED TOOLS (PREFERRED)

Instead of writing raw code, use these preconfigured tools that return structured results:

### Security Tools
```python
result = find_secrets()           # Find hardcoded API keys, passwords, tokens
result = find_sql_injection()     # Find SQL injection vulnerabilities
result = find_command_injection() # Find command injection (os.system, subprocess)
result = find_xss()               # Find XSS vulnerabilities (innerHTML, etc.)
result = find_python_security()   # Find Python issues (pickle, yaml.load, bare except)
```

### iOS/Swift Tools
```python
result = find_force_unwraps()          # Find ! force unwraps (excluding !=)
result = find_retain_cycles()          # Find closures missing [weak self]
result = find_main_thread_violations() # Find UI updates off main thread
```

### Persistence/State Tools
```python
result = find_persistence_patterns()  # Find localStorage, UserDefaults, CoreData usage
result = find_state_mutations()       # Find state management patterns
```

### Quality Tools
```python
result = find_long_functions(max_lines=50)  # Find functions over N lines
result = find_todos()                        # Find TODO/FIXME/HACK comments
```

### Architecture Tools
```python
result = map_architecture()        # Categorize files by type
result = find_imports("module")    # Find all imports of a module
```

### Batch Scans (run multiple tools at once)
```python
results = run_security_scan()  # Runs all security tools
results = run_quality_scan()   # Runs all quality tools
results = run_ios_scan()       # Runs all iOS/Swift tools
results = run_persistence_scan()  # Runs persistence/state tools
```

### Using Results
Each tool returns a `ToolResult` with:
- `result.findings` - List of Finding objects with file:line, code, confidence
- `result.count` - Number of findings
- `result.high_confidence` - Only HIGH confidence findings
- `result.to_markdown()` - Formatted output for FINAL_ANSWER

```python
# CORRECT workflow (with verification)
result = find_secrets()
if result.count > 0:
    output = result.to_markdown()
    verification = verify_results("{query}", output)
    if verification.status.value != "FAILED":
        FINAL_ANSWER = output
    else:
        print(f"Verification failed: {{verification.guidance}}")
else:
    FINAL_ANSWER = "No hardcoded secrets found."
```

## Content Info

- `files` - List of {file_count} file paths (pre-loaded)
- `prompt` - Full content ({char_count:,} chars)

## Quick Reference

| Task | Tool |
|------|------|
| Security audit | `run_security_scan()` |
| iOS/Swift review | `run_ios_scan()` |
| Code quality | `run_quality_scan()` |
| Persistence/state | `run_persistence_scan()` |
| Find secrets | `find_secrets()` |
| Force unwraps | `find_force_unwraps()` |
| State patterns | `find_state_mutations()` |

## Advanced: Custom Search

If no tool matches your needs, use raw search:
```python
matches = search_pattern(r'your_regex', '.py')  # Returns list of matches
for m in matches:
    print(f"{{m['file']}}:{{m['line']}} - {{m['content']}}")
```

Or use `llm_query(text)` for semantic analysis of specific code sections.

## Output

ALWAYS verify before setting FINAL_ANSWER:
```python
result = find_secrets()
output = result.to_markdown()
verification = verify_results("{query}", output)
FINAL_ANSWER = output  # Only if verification passed
```

## Query
{query}

Start by selecting the appropriate tool(s), run them, VERIFY results, then set FINAL_ANSWER.'''

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
        # Cache usage tracking
        self.cache_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_writes": 0,
            "tokens_saved": 0,
            "cost_saved": 0.0,
        }

    def initialize(self, content: str) -> None:
        """Initialize the REPL with content stored as `prompt` variable."""
        self.state = REPLState(prompt=content)
        self.state.variables["prompt"] = content
        self.state.variables["FINAL_ANSWER"] = None

    def _build_cached_message(self, role: str, content: str, use_cache: bool = True) -> dict:
        """
        Build a message with OpenRouter cache control.

        For large content (system prompts, codebase), we add cache_control
        to enable prompt caching. This significantly reduces costs on
        repeated queries against the same codebase.

        Args:
            role: Message role (system, user, assistant)
            content: Message content
            use_cache: Whether to add cache control

        Returns:
            Message dict with optional cache_control
        """
        if not use_cache or not self.config.use_prompt_cache:
            return {"role": role, "content": content}

        # Use multipart content format for cache control
        # Cache control works best on large, static content
        return {
            "role": role,
            "content": [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {
                        "type": "ephemeral",
                        "ttl": self.config.prompt_cache_ttl
                    }
                }
            ]
        }

    def _make_cached_request(
        self,
        messages: list[dict],
        max_tokens: int = 4000,
        model: str | None = None
    ) -> tuple[str, dict]:
        """
        Make an API request with cache tracking.

        Returns:
            (response_text, usage_info)
        """
        self.cache_stats["total_requests"] += 1

        extra_body = {}
        if self.config.track_cache_usage:
            extra_body["usage"] = {"include": True}

        try:
            response = self.client.chat.completions.create(
                model=model or self.config.model,
                max_tokens=max_tokens,
                messages=messages,
                extra_body=extra_body if extra_body else None
            )

            result = response.choices[0].message.content if response.choices else ""

            # Track cache usage if available
            usage_info = {}
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                usage_info = {
                    "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(usage, 'completion_tokens', 0),
                    "total_tokens": getattr(usage, 'total_tokens', 0),
                }

                # Check for cache discount (OpenRouter specific)
                if hasattr(usage, 'cache_discount'):
                    usage_info["cache_discount"] = usage.cache_discount
                    if usage.cache_discount > 0:
                        self.cache_stats["cache_hits"] += 1
                        self.cache_stats["tokens_saved"] += int(usage.cache_discount)

                # Check for cache_read_input_tokens (Anthropic style)
                if hasattr(usage, 'cache_read_input_tokens'):
                    cache_read = getattr(usage, 'cache_read_input_tokens', 0)
                    if cache_read > 0:
                        self.cache_stats["cache_hits"] += 1
                        usage_info["cache_read_tokens"] = cache_read

                # Check for cache_creation_input_tokens
                if hasattr(usage, 'cache_creation_input_tokens'):
                    cache_write = getattr(usage, 'cache_creation_input_tokens', 0)
                    if cache_write > 0:
                        self.cache_stats["cache_writes"] += 1
                        usage_info["cache_write_tokens"] = cache_write

            return result, usage_info

        except Exception as e:
            return f"Error: {e}", {"error": str(e)}

    def get_cache_stats(self) -> dict:
        """Get current cache usage statistics."""
        stats = self.cache_stats.copy()
        if stats["total_requests"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_requests"]
        else:
            stats["cache_hit_rate"] = 0.0
        return stats

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

    def _create_batch_llm_query_function(self) -> Callable[[list[str], int], list[str]]:
        """
        L1: Create batch query function for parallel LLM calls.

        This dramatically reduces latency when multiple sub-queries are needed
        by executing them in parallel rather than sequentially.
        """
        import asyncio
        from openai import AsyncOpenAI

        # Create async client for parallel calls
        async_client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base_url,
        )

        actual_files = set(self._extract_file_list())

        async def _async_single_query(text: str, query_id: str, max_tokens: int) -> str:
            """Execute single async LLM query."""
            enhanced_text = f"""IMPORTANT: You are analyzing REAL code.
- Only report what you actually find in the provided content
- Include exact file paths, line references, and code snippets
- If nothing relevant is found, say "No relevant findings"
- NEVER generate example/hypothetical/simulated output

{text}"""

            try:
                response = await async_client.chat.completions.create(
                    model=self.config.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": enhanced_text}]
                )
                result = response.choices[0].message.content if response.choices else ""
                self.state.llm_responses[query_id] = result
                self.state.variables[query_id] = result
                return result
            except Exception as e:
                error_msg = f"llm_query error: {type(e).__name__}: {e}"
                self.state.llm_responses[query_id] = error_msg
                return error_msg

        async def _async_batch_query(queries: list[str], max_tokens: int) -> list[str]:
            """Execute multiple queries in parallel with semaphore for rate limiting."""
            semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

            async def limited_query(text: str, idx: int) -> str:
                async with semaphore:
                    query_id = f"llm_batch_{self.state.query_counter}_{idx}"
                    return await _async_single_query(text, query_id, max_tokens)

            self.state.query_counter += 1
            tasks = [limited_query(q, i) for i, q in enumerate(queries)]
            return await asyncio.gather(*tasks)

        def llm_batch_query(queries: list[str], max_tokens: int = 4000) -> list[str]:
            """
            Query multiple sub-LLMs in parallel for faster processing.

            L1 Enhancement: Reduces latency by 40-60% for multi-query operations.

            Args:
                queries: List of prompts to send to sub-LLMs
                max_tokens: Maximum response tokens per query

            Returns:
                List of responses (same order as queries)
            """
            import sys
            print(f"[REPL] L1: Executing {len(queries)} queries in parallel...", file=sys.stderr)

            # Run async batch in event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                # If already in async context, create task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(_async_batch_query(queries, max_tokens))
                    )
                    return future.result()
            else:
                return loop.run_until_complete(_async_batch_query(queries, max_tokens))

        return llm_batch_query

    def _extract_file_list(self) -> list[str]:
        """Extract list of actual file paths from the prompt content."""
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

    def _create_verify_results_function(self) -> Callable:
        """Create the verify_results function for the REPL guardrail."""
        content = self.state.prompt
        files = self._extract_file_list()

        def verify_results(query: str, results: str) -> QueryVerificationResult:
            """
            Verify results before setting FINAL_ANSWER.

            This is the GUARDRAIL that ensures results are:
            1. Aligned with query intent
            2. Specific (file:line, code, confidence)

            Args:
                query: The original search query
                results: The results to verify (from tool.to_markdown())

            Returns:
                QueryVerificationResult with status, guidance

            Example:
                result = find_secrets()
                verification = verify_results("find secrets", result.to_markdown())
                if verification.status.value != "FAILED":
                    FINAL_ANSWER = result.to_markdown()
                else:
                    print(verification.guidance)  # Fix issues and retry
            """
            return verify_query_results(query, results, content, files)

        return verify_results

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

        # Initialize structured tools
        tools = StructuredTools(self.state.prompt)

        return {
            "__builtins__": safe_builtins,
            "prompt": self.state.prompt,
            "llm_query": self._create_llm_query_function(),
            "llm_batch_query": self._create_batch_llm_query_function(),  # L1: Parallel queries
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

            # ===== STRUCTURED TOOLS (PREFERRED) =====
            # These return ToolResult objects with structured findings
            "tools": tools,

            # Security tools
            "find_secrets": tools.find_secrets,
            "find_sql_injection": tools.find_sql_injection,
            "find_command_injection": tools.find_command_injection,
            "find_xss": tools.find_xss_vulnerabilities,
            "find_python_security": tools.find_python_security,

            # iOS/Swift tools
            "find_force_unwraps": tools.find_force_unwraps,
            "find_retain_cycles": tools.find_retain_cycles,
            "find_main_thread_violations": tools.find_main_thread_violations,
            "find_weak_self_issues": tools.find_weak_self_issues,
            "find_cloudkit_issues": tools.find_cloudkit_issues,
            "find_deprecated_apis": tools.find_deprecated_apis,
            "find_swiftdata_issues": tools.find_swiftdata_issues,

            # Quality tools
            "find_long_functions": tools.find_long_functions,
            "find_todos": tools.find_todos,

            # Architecture tools
            "map_architecture": tools.map_architecture,
            "find_imports": tools.find_imports,

            # Batch scans
            "run_security_scan": tools.run_security_scan,
            "run_quality_scan": tools.run_quality_scan,
            "run_ios_scan": tools.run_ios_scan,

            # Helper to get file list
            "files": tools.files,

            # Persistence/State tools
            "find_persistence_patterns": tools.find_persistence_patterns,
            "find_state_mutations": tools.find_state_mutations,
            "run_persistence_scan": tools.run_persistence_scan,

            # VERIFICATION TOOL (REQUIRED before FINAL_ANSWER)
            "verify_results": self._create_verify_results_function(),
            "VerificationStatus": VerificationStatus,

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

            # Differentiate between syntax errors and security violations
            if syntax_error and all("Syntax error" in e for e in validation_errors):
                error_msg = "Code has syntax errors (LLM generated invalid Python):\n"
                error_msg += "\n".join(f"  - {e}" for e in validation_errors)
                error_msg += f"\n\nLine {syntax_error.lineno}: {syntax_error.msg}"
                error_msg += "\n\nHint: Check that all strings are properly closed and no markdown formatting is included."
            elif syntax_error:
                error_msg = "Code validation failed:\n" + "\n".join(f"  - {e}" for e in validation_errors)
                error_msg += f"\n\nSyntax Error at Line {syntax_error.lineno}: {syntax_error.msg}"
            else:
                error_msg = "Code rejected for security reasons:\n" + "\n".join(f"  - {e}" for e in validation_errors)

            # Show the failed code for debugging (truncated if long)
            code_preview = code[:500] + "..." if len(code) > 500 else code
            error_msg += f"\n\n**Failed code:**\n```python\n{code_preview}\n```"

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

    def _detect_query_type(self, query: str) -> tuple[str, str]:
        """
        Detect query type for timeout calculation and routing.

        Returns:
            (query_type, complexity) - e.g., ("pattern", "normal") or ("semantic", "deep")
        """
        query_lower = query.lower()

        # Pattern queries - fast, regex-based
        pattern_indicators = [
            "find", "search", "grep", "locate", "list", "count",
            "force unwrap", "secrets", "todo", "fixme", "import"
        ]

        # Semantic queries - need LLM analysis
        semantic_indicators = [
            "analyze", "explain", "understand", "review", "assess",
            "architecture", "design", "refactor", "improve", "why",
            "how does", "what is the purpose", "security audit"
        ]

        # Complexity indicators
        deep_indicators = [
            "thorough", "comprehensive", "all", "every", "complete",
            "deep", "detailed", "full analysis"
        ]

        is_semantic = any(ind in query_lower for ind in semantic_indicators)
        is_deep = any(ind in query_lower for ind in deep_indicators)

        query_type = "semantic" if is_semantic else "pattern"
        complexity = "deep" if is_deep else ("normal" if is_semantic else "simple")

        return query_type, complexity

    async def run_rlm_session(
        self,
        query: str,
        max_iterations: int | None = None  # None = use config default
    ) -> str:
        """
        Run a complete RLM session with dynamic timeout.

        The orchestrating LLM (Sonnet) writes code iteratively.
        Sub-queries use Haiku 4.5 via llm_query().

        Args:
            query: The question to answer
            max_iterations: Maximum code execution rounds (None = config default)

        Returns:
            The final answer
        """
        # Use config defaults if not specified
        if max_iterations is None:
            max_iterations = self.config.max_iterations

        # Detect query type for dynamic timeout
        query_type, complexity = self._detect_query_type(query)
        file_count = len(self._extract_file_list())

        # Calculate dynamic timeout
        timeout_seconds = self.config.calculate_timeout(
            file_count=file_count,
            query_type=query_type,
            complexity=complexity
        )

        import sys
        print(f"[REPL] Query type: {query_type}, complexity: {complexity}", file=sys.stderr)
        print(f"[REPL] Dynamic timeout: {timeout_seconds}s for {file_count} files", file=sys.stderr)
        print(f"[REPL] Max iterations: {max_iterations}", file=sys.stderr)

        # Sanitize query to prevent format string issues and clean markdown
        safe_query = self.sanitize_query(query)

        # Extract actual file list to ground the LLM in reality
        actual_files = self._extract_file_list()

        system = self.SYSTEM_PROMPT.format(
            query=safe_query,
            char_count=len(self.state.prompt),
            file_count=len(actual_files)
        )

        # Build categorized file index for better discovery
        categorized_index = self._build_categorized_file_index(actual_files)

        # Initial message - simplified to emphasize tool usage
        # Build system message with caching for the large codebase content
        # The system prompt + codebase index is static and benefits from caching
        system_content = f"""You are an RLM REPL analyzing code. Use structured tools and VERIFY results before returning.

## Available Files ({len(actual_files)} files, {len(self.state.prompt):,} chars)
{categorized_index}

## Tools Available
- Security: find_secrets(), find_sql_injection(), run_security_scan()
- iOS/Swift: find_force_unwraps(), find_retain_cycles(), run_ios_scan()
- Persistence: find_persistence_patterns(), find_state_mutations(), run_persistence_scan()
- Quality: find_long_functions(), find_todos(), run_quality_scan()

## REQUIRED: Verify before returning
```python
result = <tool>()
verification = verify_results("{safe_query}", result.to_markdown())
if verification.status.value != "FAILED":
    FINAL_ANSWER = result.to_markdown()
```"""

        # User message with the specific query (changes per request)
        user_msg = f"""## Query: {safe_query}

Select the appropriate tool(s), run them, verify results, then set FINAL_ANSWER."""

        # Build messages with cache control on system (static) content
        messages = [
            self._build_cached_message("system", system_content, use_cache=True),
            {"role": "user", "content": user_msg}  # User query changes, don't cache
        ]

        # Log cache status
        if self.config.use_prompt_cache:
            import sys
            print(f"[REPL] Prompt caching enabled (TTL: {self.config.prompt_cache_ttl})", file=sys.stderr)

        consecutive_failures = 0
        max_consecutive_failures = self.config.max_consecutive_failures

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

            # L5: Adaptive retry with prompt modification on failures
            if not any_success:
                consecutive_failures += 1

                # After 2 failures, inject simplified instructions
                if consecutive_failures == 2:
                    print(f"[REPL] Code errors detected, injecting simplified instructions...", file=sys.stderr)
                    messages.append({
                        "role": "user",
                        "content": f"""Your code had errors. Let me help you with simpler approaches.

**RULES FOR SUCCESS:**
1. Use ONLY these pre-built tools (no custom code needed):
   - `result = find_secrets()` - Find hardcoded secrets
   - `result = run_security_scan()` - Run all security checks
   - `result = run_ios_scan()` - Run all iOS/Swift checks
   - `result = find_force_unwraps()` - Find Swift force unwraps

2. Get results with: `FINAL_ANSWER = result.to_markdown()`

3. Do NOT write custom regex or loops - the tools handle everything.

**Example (copy this pattern):**
```python
result = run_security_scan()
if result:
    FINAL_ANSWER = "\\n".join(r.to_markdown() for r in result)
else:
    FINAL_ANSWER = "No security issues found."
```

Try again with the pre-built tools:"""
                    })
                    continue  # Give it another chance with simplified instructions

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

        # Try to return last successful output
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
            output_parts.append("\n\n---\n## Partial Analysis (LLM sub-responses before failure)")
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

