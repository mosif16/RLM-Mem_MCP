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
DEFAULT_MODEL = "google/gemini-2.5-flash"


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

    # Cache Configuration (for Anthropic API prompt caching)
    use_cache: bool = field(default_factory=lambda: os.getenv("RLM_USE_CACHE", "true").lower() == "true")
    cache_ttl: Literal["5m", "1h"] = field(default_factory=lambda: os.getenv("RLM_CACHE_TTL", "5m"))  # type: ignore

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
