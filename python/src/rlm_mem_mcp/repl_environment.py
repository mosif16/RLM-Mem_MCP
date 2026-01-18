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
"""

import re
import ast
from dataclasses import dataclass, field
from typing import Any, Callable
from io import StringIO
import sys

from openai import OpenAI

from .config import RLMConfig


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

        try:
            tree = ast.parse(code)
            self.visit(tree)
        except SyntaxError as e:
            self.errors.append(f"Syntax error: {e}")

        return len(self.errors) == 0, self.errors


def validate_code(code: str) -> tuple[bool, list[str]]:
    """
    Validate Python code for sandbox safety.

    Args:
        code: Python source code to validate

    Returns:
        (is_safe, errors) tuple
    """
    validator = CodeValidator()
    return validator.validate(code)


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

## Environment

1. `prompt` - Contains {char_count:,} chars of REAL source code (files concatenated)
   Format: "### File: path/to/file.ext\\n<actual code>\\n### File: ..."

2. `llm_query(text)` - Query a sub-LLM. Pass ACTUAL code snippets from prompt.
   The sub-LLM will analyze ONLY what you send it.

3. `extract_with_lines(filepath)` - Extract file content WITH line numbers.
   Returns formatted string: "1: first line\\n2: second line\\n..."

4. `FINAL_ANSWER` - Set this to your final response with REAL findings.

## CRITICAL OUTPUT FORMAT

All findings MUST include:
- **File:Line** format: `path/to/file.py:42`
- **Code snippet**: The actual code from the file
- **Context**: What the issue is

Example finding format:
```
**src/auth.py:156**
Issue: Hardcoded API key
```python
API_KEY = "sk-1234567890abcdef"  # Line 156
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

2. **Extract files WITH LINE NUMBERS**:
   ```python
   # Use the helper to get line-numbered content
   content = extract_with_lines("path/to/file.py")
   print(content[:2000])  # Shows "1: line1\\n2: line2\\n..."
   ```

3. **Analyze with llm_query (include line numbers)**:
   ```python
   # Send line-numbered content for analysis
   content = extract_with_lines("src/main.py")
   result = llm_query(f"""Analyze this code. For each finding, report:
   - Exact line number
   - The code on that line
   - What the issue is

   Code:
   {{content[:15000]}}""")
   print(result)
   ```

4. **Compile findings with file:line references**:
   ```python
   FINAL_ANSWER = f"""## Analysis of {{len(files)}} files

   {{result}}

   Files analyzed: {{', '.join(analyzed_files)}}"""
   ```

## Anti-Hallucination Rules

- ONLY report file paths that appear in `prompt`
- ONLY quote code that exists in `prompt`
- ALWAYS include line numbers (use extract_with_lines helper)
- If no issues found, say "No issues found" - don't invent examples
- Every finding MUST have format: `filepath:line_number`

## Query
{query}

Start by discovering what files are in the prompt, then use extract_with_lines() to analyze them with line numbers.'''

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
        import re
        files = re.findall(r'### File: ([^\n]+)', self.state.prompt)
        return files

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
            "re": re,  # Regex support
            "FINAL_ANSWER": None,
            **self.state.variables
        }

    def execute_code(self, code: str) -> tuple[str, bool]:
        """
        Execute Python code in the sandboxed REPL environment.

        Args:
            code: Python code to execute

        Returns:
            (output, success) tuple
        """
        # SECURITY: Validate code with AST analysis BEFORE execution
        is_safe, validation_errors = validate_code(code)
        if not is_safe:
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
        max_iterations: int = 15
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

        for iteration in range(max_iterations):
            # Orchestrating LLM generates code
            response = self.client.chat.completions.create(
                model=self.config.aggregator_model,
                max_tokens=4000,
                messages=[
                    {"role": "system", "content": system},
                    *messages
                ]
            )

            assistant_msg = response.choices[0].message.content if response.choices else ""

            # Extract code blocks
            code_blocks = self._extract_code_blocks(assistant_msg)

            if not code_blocks:
                # No code - check if we have a final answer
                if self.state.final_answer:
                    return self.state.final_answer
                return assistant_msg

            # Execute code blocks
            outputs = []
            for code in code_blocks:
                output, success = self.execute_code(code)
                outputs.append(output if output else "(no output)")

                # Check if FINAL_ANSWER was set
                if self.state.final_answer:
                    return self.state.final_answer

            # Continue conversation
            execution_results = "\n---\n".join(outputs)
            messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({
                "role": "user",
                "content": f"Output:\n```\n{execution_results}\n```\n\nContinue or set FINAL_ANSWER."
            })

        # Return best available answer
        return self.state.final_answer or self.state.output_history[-1] if self.state.output_history else "No answer generated"

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

