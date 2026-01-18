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

    # System prompt matching the paper's approach
    SYSTEM_PROMPT = '''You are an RLM (Recursive Language Model) with access to a Python REPL.

## Environment

You have access to:
1. `prompt` - A variable containing extremely long content ({char_count:,} chars).
   It's NOT in your context - you must examine it via code.

2. `llm_query(text)` - Function to query a sub-LLM that can handle ~100K chars.
   Use this to analyze portions of the prompt.

3. `FINAL_ANSWER` - Set this variable to your final response.

## Strategy

1. **Explore** the prompt structure:
   ```python
   print(f"Length: {{len(prompt)}}")
   print(prompt[:500])  # Preview
   ```

2. **Split** into manageable parts:
   ```python
   parts = prompt.split("### File:")
   print(f"Found {{len(parts)}} sections")
   ```

3. **Query** relevant sections with llm_query():
   ```python
   section = parts[1][:10000]  # First 10K chars of section
   result = llm_query(f"Find security issues in:\\n{{section}}")
   print(result)
   ```

4. **Store** results in variables (NOT summarized):
   ```python
   findings = []
   findings.append(result)  # Keep FULL response
   ```

5. **Set** final answer:
   ```python
   FINAL_ANSWER = "\\n".join(findings)
   ```

## Query
{query}

Write Python code to answer this. Execute step by step.'''

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

            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,  # Gemini 2.5 Flash
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": text}]
                )

                result = response.choices[0].message.content if response.choices else ""

                # Store FULL response (paper: variables as buffers)
                self.state.llm_responses[query_id] = result
                self.state.variables[query_id] = result

                return result

            except Exception as e:
                error_msg = f"llm_query error: {type(e).__name__}: {e}"
                self.state.llm_responses[query_id] = error_msg
                return error_msg

        return llm_query

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

        # Initial message with preview (NOT full content)
        initial_msg = f"""REPL ready. `prompt` has {len(self.state.prompt):,} characters.

Preview:
```
{self.state.prompt[:800]}...
```

Write Python code to answer: {query}"""

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

