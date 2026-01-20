"""
Configuration for RLM-Mem MCP Server

Environment Variables:
- OPENROUTER_API_KEY: Required for RLM processing via OpenRouter
- RLM_MODEL: Model for RLM processing (default: anthropic/claude-haiku-4-5-20250115)
- RLM_AGGREGATOR_MODEL: Model for final aggregation (default: anthropic/claude-haiku-4-5-20250115)
- RLM_MAX_RESULT_TOKENS: Maximum tokens in result (default: 4000)

OpenRouter:
- Uses OpenAI-compatible API at https://openrouter.ai/api/v1
- Claude Haiku 4.5 is fast and cost-effective (~$0.80/1M input tokens)
- Supports prompt caching for 90% cost reduction on repeated content
"""

import os
from dataclasses import dataclass, field
from typing import Literal, Set
from dotenv import load_dotenv

load_dotenv()


# Default model (OpenRouter model ID) - Claude Haiku 4.5 for speed and cost efficiency
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"

# Default file extensions to include
DEFAULT_EXTENSIONS = {
    # Code files
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt",
    ".scala", ".cs", ".m", ".mm", ".vue", ".svelte",
    # iOS/Swift specific
    ".xcstrings",       # Xcode localization (new format)
    ".strings",         # Localization strings
    ".stringsdict",     # Pluralization rules
    ".entitlements",    # App entitlements
    ".xcconfig",        # Build configuration
    ".modulemap",       # Module maps for C/Obj-C interop
    ".metal",           # Metal GPU shaders
    ".intentdefinition", # Siri Intent definitions
    ".xib",             # Interface Builder (legacy)
    ".storyboard",      # Storyboard UI files
    # Config files
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
    # Documentation
    ".md", ".txt", ".rst", ".adoc",
    # Other
    ".sql", ".graphql", ".proto", ".sh", ".bash", ".zsh",
    ".dockerfile", ".makefile", ".cmake",
    # Additional common types
    ".xml", ".html", ".css", ".scss", ".less", ".plist",
}

# Default directories to skip
DEFAULT_SKIP_DIRS = {
    # General
    ".git", "node_modules", "__pycache__", "venv", ".venv",
    "dist", "build", ".next", "target", "vendor", ".cache",
    ".idea", ".vscode", "coverage", ".nyc_output", "eggs",
    "*.egg-info", ".tox", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "htmlcov", ".hypothesis",
    # iOS/Xcode specific
    "DerivedData",      # Xcode build artifacts
    ".build",           # Swift Package Manager build directory
    "Pods",             # CocoaPods dependencies
    "Carthage",         # Carthage dependencies
    "SourcePackages",   # Swift Package Manager cache
    "*.xcodeproj",      # Xcode project bundles (internal files)
    "*.xcworkspace",    # Xcode workspace bundles
    "*.xcassets",       # Asset catalogs (images, colors) - binary
    "*.framework",      # Framework bundles
    "*.app",            # App bundles
    "*.appex",          # App extension bundles
    "*.lproj",          # Localization bundles (use .strings instead)
    "*.dSYM",           # Debug symbols
    "*.ipa",            # App archives
    "ModuleCache",      # Clang module cache
    "Index",            # Xcode index data
}


def _parse_csv_env(env_var: str) -> Set[str]:
    """Parse a comma-separated environment variable into a set."""
    value = os.getenv(env_var, "")
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _get_included_extensions() -> Set[str]:
    """
    Get the set of included file extensions, applying environment overrides.

    Environment variables:
    - RLM_EXTRA_EXTENSIONS: Comma-separated extensions to add (e.g., ".xml,.xsl")
    - RLM_SKIP_EXTENSIONS: Comma-separated extensions to exclude (e.g., ".md,.txt")
    """
    extensions = DEFAULT_EXTENSIONS.copy()

    # Add extra extensions
    extra = _parse_csv_env("RLM_EXTRA_EXTENSIONS")
    extensions.update(extra)

    # Remove skipped extensions
    skip = _parse_csv_env("RLM_SKIP_EXTENSIONS")
    extensions -= skip

    return extensions


def _get_skipped_directories() -> Set[str]:
    """
    Get the set of skipped directories, applying environment overrides.

    Environment variables:
    - RLM_EXTRA_SKIP_DIRS: Comma-separated directories to skip (e.g., "logs,tmp")
    - RLM_INCLUDE_DIRS: Comma-separated directories to include (removes from skip list)
    """
    dirs = DEFAULT_SKIP_DIRS.copy()

    # Add extra directories to skip
    extra = _parse_csv_env("RLM_EXTRA_SKIP_DIRS")
    dirs.update(extra)

    # Remove directories from skip list (force include)
    include = _parse_csv_env("RLM_INCLUDE_DIRS")
    dirs -= include

    return dirs


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
    # RLM_MAX_ITERATIONS: Maximum code execution rounds (default: 25)
    # RLM_MIN_ITERATIONS: Minimum iterations before allowing early exit (default: 3)
    # RLM_MAX_FAILURES: Consecutive failures before fallback (default: 3)
    max_iterations: int = field(
        default_factory=lambda: int(os.getenv("RLM_MAX_ITERATIONS", "25"))
    )
    min_iterations: int = field(
        default_factory=lambda: int(os.getenv("RLM_MIN_ITERATIONS", "3"))
    )
    max_consecutive_failures: int = field(
        default_factory=lambda: int(os.getenv("RLM_MAX_FAILURES", "3"))
    )
    # Force at least one tool execution before accepting "no findings"
    require_tool_execution: bool = field(
        default_factory=lambda: os.getenv("RLM_REQUIRE_TOOL_EXECUTION", "true").lower() == "true"
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
    # Can be customized via environment variables:
    # - RLM_EXTRA_EXTENSIONS: Comma-separated extensions to add (e.g., ".xml,.xsl")
    # - RLM_SKIP_EXTENSIONS: Comma-separated extensions to exclude (e.g., ".md,.txt")
    # - RLM_EXTRA_SKIP_DIRS: Comma-separated directories to skip (e.g., "logs,tmp")
    # - RLM_INCLUDE_DIRS: Comma-separated directories to include (removes from skip list)
    included_extensions: Set[str] = field(default_factory=lambda: _get_included_extensions())

    skipped_directories: Set[str] = field(default_factory=lambda: _get_skipped_directories())

    max_file_size_bytes: int = field(
        default_factory=lambda: int(os.getenv("RLM_MAX_FILE_SIZE", "1000000"))
    )  # 1MB per file default
    max_total_tokens: int = field(
        default_factory=lambda: int(os.getenv("RLM_MAX_TOTAL_TOKENS", "500000"))
    )  # Maximum tokens to process

    # Claude 4.5 Prompt Caching Configuration
    # Anthropic requires explicit cache_control blocks
    # Cache hits cost 10% of normal input tokens (90% savings!)
    use_cache: bool = field(
        default_factory=lambda: os.getenv("RLM_USE_CACHE", "true").lower() == "true"
    )
    use_prompt_cache: bool = field(
        default_factory=lambda: os.getenv("RLM_USE_PROMPT_CACHE", "true").lower() == "true"
    )
    cache_ttl: Literal["5m", "1h"] = field(
        default_factory=lambda: os.getenv("RLM_CACHE_TTL", "1h")  # type: ignore
    )
    prompt_cache_ttl: Literal["5m", "1h"] = field(
        default_factory=lambda: os.getenv("RLM_PROMPT_CACHE_TTL", "1h")  # type: ignore
    )
    # Track cache usage for cost monitoring
    track_cache_usage: bool = field(
        default_factory=lambda: os.getenv("RLM_TRACK_CACHE_USAGE", "true").lower() == "true"
    )
    # Prefilled assistant responses for token efficiency
    use_prefilled_responses: bool = field(
        default_factory=lambda: os.getenv("RLM_USE_PREFILLED", "true").lower() == "true"
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
