"""
REPL Environment for RLM Processing

This is the CORE of the RLM technique from arXiv:2512.24601.

Key insight: The prompt is stored as a VARIABLE in a Python REPL,
NOT fed into the LLM's context. The LLM writes code to:
1. Examine portions of the content (slicing, regex, etc.)
2. Call llm_query() on relevant portions
3. Store results in variables (NOT summarize them)
4. Build up the final answer from stored variables

This preserves ALL data - nothing is summarized or lost.
The LLM can access any part of the content at any time by writing code.
"""

import re
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable
from io import StringIO
import sys

import anthropic

from .config import RLMConfig


@dataclass
class REPLState:
    """State of the REPL environment."""

    # The full content - stored as a variable, NOT in LLM context
    context: str = ""

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


class RLMReplEnvironment:
    """
    Python REPL environment for RLM processing.

    The LLM writes code that executes in this environment.
    The content is stored as `context` variable - accessible but NOT
    consuming the LLM's context window.

    Available to the LLM:
    - `context`: The full content as a string variable
    - `llm_query(prompt)`: Call a sub-LLM with a portion of content
    - All Python string operations (slicing, regex, split, etc.)
    - Variables persist across code executions
    """

    # System prompt explaining the REPL to the LLM
    SYSTEM_PROMPT = '''You are an RLM (Recursive Language Model) operating in a Python REPL environment.

## Available Resources

1. **`context`** - A variable containing the FULL content to analyze (can be very long).
   You must examine it by writing Python code - it's NOT in your context window.

2. **`llm_query(prompt)`** - Function to query a sub-LLM. Use this to analyze portions of context.
   Returns the LLM's response as a string. Example:
   ```python
   chunk = context[0:10000]
   answer = llm_query(f"Find security issues in:\\n{chunk}")
   print(answer)
   ```

3. **Variables** - Any variables you create persist across code blocks.
   Use them to store intermediate results (NOT summaries - keep full responses).

## Your Process

1. First, explore the context structure:
   ```python
   print(f"Total length: {len(context)} chars")
   print(f"Preview: {context[:500]}...")
   ```

2. Identify relevant sections by writing code to search/slice:
   ```python
   # Find all file boundaries
   files = context.split("### File:")
   print(f"Found {len(files)} files")
   ```

3. For each relevant section, use llm_query() to analyze:
   ```python
   findings = []
   for i, file_content in enumerate(relevant_files):
       result = llm_query(f"Analyze for {query}:\\n{file_content}")
       findings.append(result)  # Store FULL response, don't summarize!
   ```

4. Build your final answer from the stored findings:
   ```python
   # Combine all findings (they're all preserved in variables)
   final_answer = "\\n\\n".join(findings)
   print(final_answer)
   ```

## Critical Rules

- NEVER try to read the full context at once - it won't fit in your context
- ALWAYS use code to slice/search the context variable
- STORE llm_query() responses in variables - don't lose them
- The final print() output becomes your answer

## Query to Answer
{query}

Write Python code to answer this query by examining the context variable.'''

    def __init__(self, config: RLMConfig):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key)
        self.state = REPLState()

    def initialize(self, content: str) -> None:
        """Initialize the REPL with content stored as a variable."""
        self.state = REPLState(context=content)
        self.state.variables["context"] = content

    def _create_llm_query_function(self) -> Callable[[str], str]:
        """Create the llm_query function for sub-LLM calls."""

        def llm_query(prompt: str, max_tokens: int = 2000) -> str:
            """
            Query a sub-LLM with a portion of content.

            Args:
                prompt: The prompt including the content portion to analyze
                max_tokens: Maximum tokens for response

            Returns:
                The LLM's response as a string (stored, not summarized)
            """
            # Track the query
            self.state.query_counter += 1
            query_id = f"response_{self.state.query_counter}"

            try:
                response = self.client.messages.create(
                    model=self.config.model,  # Haiku 4.5 for efficiency
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )

                result = response.content[0].text if response.content else ""

                # Store the FULL response (not summarized!)
                self.state.llm_responses[query_id] = result
                self.state.variables[query_id] = result

                return result

            except Exception as e:
                error_msg = f"Error in llm_query: {e}"
                self.state.llm_responses[query_id] = error_msg
                return error_msg

        return llm_query

    def execute_code(self, code: str) -> tuple[str, bool]:
        """
        Execute Python code in the REPL environment.

        Args:
            code: Python code to execute

        Returns:
            (output, success) tuple
        """
        # Record the code
        self.state.code_history.append(code)

        # Create execution environment with all variables
        exec_globals = {
            "context": self.state.context,
            "llm_query": self._create_llm_query_function(),
            "re": re,  # Regex support
            **self.state.variables
        }

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # Execute the code
            exec(code, exec_globals)

            # Get output
            output = sys.stdout.getvalue()

            # Update variables (exclude builtins and functions)
            for key, value in exec_globals.items():
                if not key.startswith("_") and key not in ("context", "llm_query", "re"):
                    if not callable(value) or key.startswith("response_"):
                        self.state.variables[key] = value

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
        max_iterations: int = 10
    ) -> str:
        """
        Run a complete RLM session to answer a query.

        The LLM writes code iteratively to examine content and build an answer.

        Args:
            query: The question to answer
            max_iterations: Maximum code execution iterations

        Returns:
            The final answer
        """
        # Build the system prompt
        system = self.SYSTEM_PROMPT.format(query=query)

        # Initial context info for the LLM (NOT the full content!)
        initial_info = f"""REPL initialized. Context variable contains {len(self.state.context):,} characters.

Preview of context structure:
```
{self.state.context[:1000]}...
```

Write Python code to explore and analyze the context to answer: {query}"""

        messages = [{"role": "user", "content": initial_info}]

        final_output = ""

        for iteration in range(max_iterations):
            # Get code from the LLM
            response = self.client.messages.create(
                model=self.config.aggregator_model,  # Sonnet for orchestration
                max_tokens=4000,
                system=system,
                messages=messages
            )

            assistant_message = response.content[0].text if response.content else ""

            # Extract code blocks
            code_blocks = self._extract_code_blocks(assistant_message)

            if not code_blocks:
                # No more code - LLM is done
                final_output = assistant_message
                break

            # Execute each code block
            all_outputs = []
            for code in code_blocks:
                output, success = self.execute_code(code)
                all_outputs.append(f"```\n{output}\n```" if output else "(no output)")

            # Send results back to LLM
            execution_result = "\n\n".join(all_outputs)
            messages.append({"role": "assistant", "content": assistant_message})
            messages.append({"role": "user", "content": f"Execution results:\n{execution_result}\n\nContinue your analysis or provide the final answer."})

            # Check if we have a final answer
            if "FINAL ANSWER:" in assistant_message or iteration == max_iterations - 1:
                final_output = self._extract_final_answer(assistant_message, all_outputs)
                break

        return final_output

    def _extract_code_blocks(self, text: str) -> list[str]:
        """Extract Python code blocks from LLM response."""
        # Match ```python ... ``` or ``` ... ```
        pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches if m.strip()]

    def _extract_final_answer(self, text: str, outputs: list[str]) -> str:
        """Extract the final answer from LLM response or outputs."""
        # Check for explicit final answer
        if "FINAL ANSWER:" in text:
            return text.split("FINAL ANSWER:")[-1].strip()

        # Use the last substantial output
        for output in reversed(outputs):
            clean = output.replace("```", "").strip()
            if len(clean) > 50:
                return clean

        return text

    def get_all_responses(self) -> dict[str, str]:
        """Get all stored sub-LLM responses (preserved, not summarized)."""
        return self.state.llm_responses.copy()

    def get_variable(self, name: str) -> Any:
        """Get a variable from the REPL state."""
        return self.state.variables.get(name)

    def get_execution_history(self) -> list[tuple[str, str]]:
        """Get history of code executions and their outputs."""
        return list(zip(self.state.code_history, self.state.output_history))
