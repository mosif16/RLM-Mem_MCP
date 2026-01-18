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

import anthropic

from .config import RLMConfig


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
        self.client = anthropic.Anthropic(api_key=config.api_key)
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
                response = self.client.messages.create(
                    model=self.config.model,  # Haiku 4.5
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": text}]
                )

                result = response.content[0].text if response.content else ""

                # Store FULL response (paper: variables as buffers)
                self.state.llm_responses[query_id] = result
                self.state.variables[query_id] = result

                return result

            except Exception as e:
                error_msg = f"llm_query error: {e}"
                self.state.llm_responses[query_id] = error_msg
                return error_msg

        return llm_query

    def _create_safe_globals(self) -> dict[str, Any]:
        """Create a restricted execution environment."""
        # Allow safe builtins only
        safe_builtins = {
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "min": min,
            "max": max,
            "sum": sum,
            "any": any,
            "all": all,
            "print": print,
            "isinstance": isinstance,
            "type": type,
            "hasattr": hasattr,
            "getattr": getattr,
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
            # Orchestrating LLM (Sonnet) generates code
            response = self.client.messages.create(
                model=self.config.aggregator_model,
                max_tokens=4000,
                system=system,
                messages=messages
            )

            assistant_msg = response.content[0].text if response.content else ""

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

