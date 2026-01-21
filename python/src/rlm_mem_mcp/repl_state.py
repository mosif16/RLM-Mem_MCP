"""
REPL State management for RLM Processing.

Contains the state dataclass that tracks execution context.
"""

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .content_analyzer import DeadCodeRegion


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
    dead_code_regions: dict[str, list["DeadCodeRegion"]] = field(default_factory=dict)


def create_repl_state(content: str = "") -> REPLState:
    """Factory function to create a new REPL state."""
    state = REPLState(prompt=content)
    state.variables["prompt"] = content
    state.variables["FINAL_ANSWER"] = None
    return state
