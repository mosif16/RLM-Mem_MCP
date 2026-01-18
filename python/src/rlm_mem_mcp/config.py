"""
Configuration for RLM-Mem MCP Server

Environment Variables:
- ANTHROPIC_API_KEY: Required for RLM processing
- RLM_MODEL: Model for RLM processing (default: claude-sonnet-4-5-20241022)
- RLM_BACKEND: API backend (default: anthropic)
- RLM_MAX_RESULT_TOKENS: Maximum tokens in result (default: 4000)
- RLM_USE_CACHE: Enable prompt caching (default: true)
- RLM_CACHE_TTL: Cache TTL - "5m" or "1h" (default: 5m)
"""

import os
from dataclasses import dataclass, field
from typing import Literal, Set
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RLMConfig:
    """Configuration for RLM processing."""

    # API Configuration
    api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("RLM_MODEL", "claude-sonnet-4-5-20241022"))
    backend: str = field(default_factory=lambda: os.getenv("RLM_BACKEND", "anthropic"))

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

    # Cache Configuration (Anthropic Prompt Caching)
    use_cache: bool = field(
        default_factory=lambda: os.getenv("RLM_USE_CACHE", "true").lower() == "true"
    )
    cache_ttl: Literal["5m", "1h"] = field(
        default_factory=lambda: os.getenv("RLM_CACHE_TTL", "5m")  # type: ignore
    )
    min_tokens_for_cache: int = 1024  # Claude Sonnet minimum

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

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.api_key:
            errors.append("ANTHROPIC_API_KEY environment variable not set")

        if self.max_result_tokens < 100:
            errors.append("max_result_tokens must be at least 100")

        if self.max_chunk_tokens < 1000:
            errors.append("max_chunk_tokens must be at least 1000")

        if self.cache_ttl not in ("5m", "1h"):
            errors.append("cache_ttl must be '5m' or '1h'")

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
