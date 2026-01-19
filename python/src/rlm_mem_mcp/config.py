"""
Configuration for RLM-Mem MCP Server

Environment Variables:
- OPENROUTER_API_KEY: Required for RLM processing via OpenRouter
- RLM_MODEL: Model for RLM processing (default: google/gemini-2.5-flash-preview)
- RLM_AGGREGATOR_MODEL: Model for final aggregation (default: google/gemini-2.5-flash-preview)
- RLM_MAX_RESULT_TOKENS: Maximum tokens in result (default: 4000)

OpenRouter:
- Uses OpenAI-compatible API at https://openrouter.ai/api/v1
- Gemini 2.5 Flash is fast and cost-effective (~$0.15/1M input tokens)
"""

import os
from dataclasses import dataclass, field
from typing import Literal, Set
from dotenv import load_dotenv

load_dotenv()


# Default model (OpenRouter model ID)
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"


@dataclass
class RLMConfig:
    """Configuration for RLM processing."""

    # API Configuration (OpenRouter)
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    api_base_url: str = "https://openrouter.ai/api/v1"

    # Use Gemini 2.5 Flash - fast and cost-effective
    model: str = field(default_factory=lambda: os.getenv("RLM_MODEL", DEFAULT_MODEL))
    aggregator_model: str = field(default_factory=lambda: os.getenv("RLM_AGGREGATOR_MODEL", DEFAULT_MODEL))

    # Processing Configuration
    max_result_tokens: int = field(
        default_factory=lambda: int(os.getenv("RLM_MAX_RESULT_TOKENS", "4000"))
    )
    max_chunk_tokens: int = field(
        default_factory=lambda: int(os.getenv("RLM_MAX_CHUNK_TOKENS", "8000"))
    )
    overlap_tokens: int = field(
        default_factory=lambda: int(os.getenv("RLM_OVERLAP_TOKENS", "200"))
    )

    # Dynamic Timeout Configuration
    base_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("RLM_BASE_TIMEOUT", "120"))
    )
    per_file_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("RLM_PER_FILE_TIMEOUT", "2.0"))
    )
    max_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("RLM_MAX_TIMEOUT", "600"))
    )
    semantic_query_multiplier: float = field(
        default_factory=lambda: float(os.getenv("RLM_SEMANTIC_MULTIPLIER", "1.5"))
    )

    # Iteration Limits (configurable per feedback)
    max_iterations: int = field(
        default_factory=lambda: int(os.getenv("RLM_MAX_ITERATIONS", "25"))
    )
    max_consecutive_failures: int = field(
        default_factory=lambda: int(os.getenv("RLM_MAX_FAILURES", "3"))
    )

    def calculate_timeout(
        self,
        file_count: int,
        query_type: str = "pattern",
        complexity: str = "normal"
    ) -> int:
        """
        Calculate dynamic timeout based on task parameters.

        Args:
            file_count: Number of files to process
            query_type: "pattern" (grep-like) or "semantic" (LLM analysis)
            complexity: "simple", "normal", or "deep"

        Returns:
            Timeout in seconds
        """
        # Base timeout
        timeout = self.base_timeout_seconds

        # Add per-file time
        timeout += int(file_count * self.per_file_timeout_seconds)

        # Semantic queries take longer (LLM analysis vs pattern matching)
        if query_type == "semantic":
            timeout = int(timeout * self.semantic_query_multiplier)

        # Complexity multipliers
        complexity_multipliers = {
            "simple": 0.7,
            "normal": 1.0,
            "deep": 1.5,
            "thorough": 2.0,
        }
        timeout = int(timeout * complexity_multipliers.get(complexity, 1.0))

        # Cap at maximum
        return min(timeout, self.max_timeout_seconds)

    # File Collection Configuration
    included_extensions: Set[str] = field(default_factory=lambda: {
        # Code files
        ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
        ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt",
        ".scala", ".cs", ".m", ".mm", ".vue", ".svelte",
        # Config files
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
        # Documentation
        ".md", ".txt", ".rst", ".adoc",
        # Other
        ".sql", ".graphql", ".proto", ".sh", ".bash", ".zsh",
        ".dockerfile", ".makefile", ".cmake",
    })

    skipped_directories: Set[str] = field(default_factory=lambda: {
        ".git", "node_modules", "__pycache__", "venv", ".venv",
        "dist", "build", ".next", "target", "vendor", ".cache",
        ".idea", ".vscode", "coverage", ".nyc_output", "eggs",
        "*.egg-info", ".tox", ".pytest_cache", ".mypy_cache",
        ".ruff_cache", "htmlcov", ".hypothesis",
    })

    max_file_size_bytes: int = 1_000_000  # 1MB per file
    max_total_tokens: int = 500_000  # Maximum tokens to process

    # OpenRouter Prompt Caching Configuration
    # Gemini has implicit caching (automatic), but we can extend TTL
    # Anthropic requires explicit cache_control
    use_prompt_cache: bool = field(
        default_factory=lambda: os.getenv("RLM_USE_PROMPT_CACHE", "true").lower() == "true"
    )
    prompt_cache_ttl: Literal["5m", "1h"] = field(
        default_factory=lambda: os.getenv("RLM_PROMPT_CACHE_TTL", "1h")  # type: ignore
    )
    # Track cache usage for cost monitoring
    track_cache_usage: bool = field(
        default_factory=lambda: os.getenv("RLM_TRACK_CACHE_USAGE", "true").lower() == "true"
    )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.api_key:
            errors.append("OPENROUTER_API_KEY environment variable not set")

        if self.max_result_tokens < 100:
            errors.append("max_result_tokens must be at least 100")

        if self.max_chunk_tokens < 1000:
            errors.append("max_chunk_tokens must be at least 1000")

        return errors


@dataclass
class ServerConfig:
    """Configuration for the MCP server."""

    name: str = "rlm-recursive-memory"
    version: str = "1.0.0"
    description: str = (
        "MCP server implementing Recursive Language Model (RLM) "
        "for processing arbitrarily large inputs with Claude"
    )

    # Server limits
    max_concurrent_operations: int = 3
    operation_timeout_seconds: int = 300  # 5 minutes

    # Response formatting
    include_metadata: bool = True
    truncation_strategy: Literal["head", "tail", "both"] = "both"


def get_config() -> tuple[RLMConfig, ServerConfig]:
    """Get configuration instances."""
    return RLMConfig(), ServerConfig()
