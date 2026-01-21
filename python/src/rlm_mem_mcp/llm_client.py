"""
LLM Client for RLM REPL Environment (v2.9).

Provides LLM query functionality including:
- Single query execution
- Batch parallel queries
- Response validation and hallucination detection
"""

import re
import asyncio
from typing import Callable, Any, TYPE_CHECKING

from openai import OpenAI, AsyncOpenAI

if TYPE_CHECKING:
    from .config import RLMConfig


class LLMClient:
    """Handles LLM queries for the REPL environment."""

    def __init__(self, config: "RLMConfig"):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base_url,
        )
        self.async_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base_url,
        )

    def create_llm_query_function(
        self,
        state: Any,
        actual_files: set[str]
    ) -> Callable[[str, int], str]:
        """
        Create the llm_query function for sub-LLM calls.

        Args:
            state: REPLState instance to store results
            actual_files: Set of actual file paths for validation

        Returns:
            llm_query function
        """

        def llm_query(text: str, max_tokens: int = 4000) -> str:
            """
            Query a sub-LLM with content.

            Args:
                text: The prompt/content to send to sub-LLM
                max_tokens: Maximum response tokens

            Returns:
                The sub-LLM's response (stored in full, not summarized)
            """
            state.query_counter += 1
            query_id = f"llm_response_{state.query_counter}"

            enhanced_text = self._enhance_prompt(text)

            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": enhanced_text}]
                )

                result = response.choices[0].message.content if response.choices else ""
                result = self._validate_response(result, actual_files)

                # Store FULL response (paper: variables as buffers)
                state.llm_responses[query_id] = result
                state.variables[query_id] = result

                return result

            except Exception as e:
                error_msg = f"llm_query error: {type(e).__name__}: {e}"
                state.llm_responses[query_id] = error_msg
                return error_msg

        return llm_query

    def create_batch_llm_query_function(
        self,
        state: Any,
        actual_files: set[str]
    ) -> Callable[[list[str], int], list[str]]:
        """
        Create batch query function for parallel LLM calls.

        Args:
            state: REPLState instance to store results
            actual_files: Set of actual file paths for validation

        Returns:
            llm_batch_query function
        """

        async def _async_single_query(text: str, query_id: str, max_tokens: int) -> str:
            """Execute single async LLM query."""
            enhanced_text = self._enhance_prompt(text)

            try:
                response = await self.async_client.chat.completions.create(
                    model=self.config.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": enhanced_text}]
                )
                result = response.choices[0].message.content if response.choices else ""
                state.llm_responses[query_id] = result
                state.variables[query_id] = result
                return result
            except Exception as e:
                error_msg = f"llm_query error: {type(e).__name__}: {e}"
                state.llm_responses[query_id] = error_msg
                return error_msg

        async def _async_batch_query(queries: list[str], max_tokens: int) -> list[str]:
            """Execute multiple queries in parallel with semaphore for rate limiting."""
            semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests

            async def limited_query(text: str, idx: int) -> str:
                async with semaphore:
                    query_id = f"llm_batch_{state.query_counter}_{idx}"
                    return await _async_single_query(text, query_id, max_tokens)

            state.query_counter += 1
            tasks = [limited_query(q, i) for i, q in enumerate(queries)]
            return await asyncio.gather(*tasks)

        def llm_batch_query(queries: list[str], max_tokens: int = 4000) -> list[str]:
            """
            Query multiple sub-LLMs in parallel for faster processing.

            Args:
                queries: List of prompts to send to sub-LLMs
                max_tokens: Maximum response tokens per query

            Returns:
                List of responses (same order as queries)
            """
            import sys
            print(f"[REPL] Executing {len(queries)} queries in parallel...", file=sys.stderr)

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(_async_batch_query(queries, max_tokens))
                    )
                    return future.result()
            else:
                return loop.run_until_complete(_async_batch_query(queries, max_tokens))

        return llm_batch_query

    def _enhance_prompt(self, text: str) -> str:
        """Add anti-hallucination instructions to prompt."""
        return f"""IMPORTANT: You are analyzing REAL code.
- Only report what you actually find in the provided content
- Include exact file paths, line references, and code snippets
- If nothing relevant is found, say "No relevant findings"
- NEVER generate example/hypothetical/simulated output

{text}"""

    def _validate_response(self, result: str, actual_files: set[str]) -> str:
        """Validate response for hallucination markers and invalid references."""
        hallucination_markers = [
            "example:", "for example,", "hypothetically",
            "let's say", "imagine", "suppose", "would be like",
            "simulated", "demonstration", "sample output",
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

        # Validate line number references
        line_refs = re.findall(r':(\d+)\b', result)
        invalid_lines = []
        for line_ref in line_refs:
            line_num = int(line_ref)
            if line_num > 5000:
                invalid_lines.append(line_num)

        if invalid_lines and not has_hallucination_warning:
            result = f"[VERIFY: Contains high line numbers that may be inaccurate: {invalid_lines[:5]}]\n\n{result}"

        # Check for file paths that don't exist
        file_refs = re.findall(r'([a-zA-Z0-9_/\-\.]+\.[a-z]{1,5}):', result)
        unverified_files = []
        for file_ref in file_refs:
            if not any(file_ref in actual_file or actual_file.endswith(file_ref) for actual_file in actual_files):
                unverified_files.append(file_ref)

        if unverified_files and len(unverified_files) <= 5:
            unique_unverified = list(set(unverified_files))[:3]
            result += f"\n\n[NOTE: These file references could not be verified: {unique_unverified}]"

        return result


def create_llm_client(config: "RLMConfig") -> LLMClient:
    """Factory function to create an LLMClient."""
    return LLMClient(config)
