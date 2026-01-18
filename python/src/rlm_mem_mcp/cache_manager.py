"""
Cache Manager for RLM-Mem MCP Server

Implements Anthropic's prompt caching for token optimization.
Based on official Anthropic documentation for prompt caching.

Key features:
- Automatic cache breakpoint management
- Cache statistics tracking
- TTL configuration (5m or 1h)
- Minimum token threshold enforcement
"""

import time
from dataclasses import dataclass, field
from typing import Any, Literal
from collections import defaultdict

import tiktoken

from .config import RLMConfig


@dataclass
class CacheBreakpoint:
    """Represents a cache breakpoint in the prompt."""

    position: int  # Position in content list
    token_count: int
    ttl: Literal["5m", "1h"]
    content_hash: str


@dataclass
class CacheStats:
    """Statistics about cache usage."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    tokens_saved: int = 0
    cost_saved_usd: float = 0.0

    # Per-breakpoint stats
    breakpoint_hits: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def record_request(
        self,
        cache_read_tokens: int,
        cache_write_tokens: int,
        input_tokens: int
    ) -> None:
        """Record a request's cache statistics."""
        self.total_requests += 1

        if cache_read_tokens > 0:
            self.cache_hits += 1
            self.tokens_saved += cache_read_tokens
            # Cache reads cost 10% of normal input
            # So we save 90% of the normal cost
            self.cost_saved_usd += (cache_read_tokens / 1_000_000) * 3 * 0.9

        if cache_write_tokens > 0:
            self.cache_writes += 1
            # Cache writes cost 25% more than normal
            # No savings on first write

        if cache_read_tokens == 0 and cache_write_tokens == 0:
            self.cache_misses += 1


class CacheManager:
    """
    Manages prompt caching for Anthropic API calls.

    Implements the caching strategy from Anthropic's documentation:
    - Place static content at the beginning
    - Use cache_control parameter for breakpoints
    - Track cache performance metrics
    """

    # Pricing per million tokens (Claude Sonnet 4.5)
    PRICING = {
        "input": 3.0,
        "cache_read": 0.3,  # 10% of input
        "cache_write_5m": 3.75,  # 125% of input
        "cache_write_1h": 6.0,  # 200% of input
        "output": 15.0,
    }

    # Minimum tokens for caching by model family
    MIN_CACHE_TOKENS = {
        "sonnet": 1024,
        "opus": 4096,
        "haiku": 2048,
    }

    def __init__(self, config: RLMConfig | None = None):
        self.config = config or RLMConfig()
        self.stats = CacheStats()
        self._encoder: tiktoken.Encoding | None = None
        self._breakpoints: list[CacheBreakpoint] = []

    @property
    def encoder(self) -> tiktoken.Encoding:
        """Lazy-load tiktoken encoder."""
        if self._encoder is None:
            self._encoder = tiktoken.encoding_for_model("gpt-4")
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def get_model_family(self, model: str) -> str:
        """Determine model family from model name."""
        model_lower = model.lower()
        if "opus" in model_lower:
            return "opus"
        elif "haiku" in model_lower:
            return "haiku"
        return "sonnet"

    def meets_cache_threshold(self, token_count: int, model: str | None = None) -> bool:
        """Check if content meets minimum token threshold for caching."""
        model = model or self.config.model
        family = self.get_model_family(model)
        min_tokens = self.MIN_CACHE_TOKENS.get(family, 1024)
        return token_count >= min_tokens

    def build_cached_system(
        self,
        content: str | list[dict[str, Any]],
        model: str | None = None
    ) -> list[dict[str, Any]] | str:
        """
        Build a system prompt with appropriate cache control.

        Args:
            content: System prompt content (string or list of blocks)
            model: Model name for threshold calculation

        Returns:
            Content with cache_control if caching is enabled and thresholds are met
        """
        if not self.config.use_cache:
            return content if isinstance(content, str) else content

        # Convert string to block format
        if isinstance(content, str):
            blocks = [{"type": "text", "text": content}]
        else:
            blocks = content

        # Calculate total tokens
        total_text = "".join(
            block.get("text", "") for block in blocks if isinstance(block, dict)
        )
        token_count = self.count_tokens(total_text)

        # Check if we meet the minimum threshold
        if not self.meets_cache_threshold(token_count, model):
            return content if isinstance(content, str) else blocks

        # Add cache control to the last block
        if blocks:
            last_block = blocks[-1].copy() if isinstance(blocks[-1], dict) else {"type": "text", "text": str(blocks[-1])}
            last_block["cache_control"] = {
                "type": "ephemeral",
                "ttl": self.config.cache_ttl,
            }
            blocks = blocks[:-1] + [last_block]

        return blocks

    def build_cached_messages(
        self,
        messages: list[dict[str, Any]],
        cache_at_indices: list[int] | None = None,
        model: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Add cache control to specific message positions.

        Args:
            messages: List of message dicts
            cache_at_indices: Indices where to add cache breakpoints
            model: Model name for threshold calculation

        Returns:
            Messages with cache_control added where appropriate
        """
        if not self.config.use_cache:
            return messages

        result = []
        cache_indices = set(cache_at_indices or [len(messages) - 1])

        for i, msg in enumerate(messages):
            msg_copy = msg.copy()

            if i in cache_indices:
                content = msg_copy.get("content")

                if isinstance(content, str):
                    # Convert to block format for cache control
                    token_count = self.count_tokens(content)
                    if self.meets_cache_threshold(token_count, model):
                        msg_copy["content"] = [{
                            "type": "text",
                            "text": content,
                            "cache_control": {
                                "type": "ephemeral",
                                "ttl": self.config.cache_ttl,
                            }
                        }]
                elif isinstance(content, list) and content:
                    # Add cache control to last block
                    total_text = "".join(
                        block.get("text", "") for block in content
                        if isinstance(block, dict)
                    )
                    token_count = self.count_tokens(total_text)

                    if self.meets_cache_threshold(token_count, model):
                        new_content = content[:-1] + [
                            {
                                **content[-1],
                                "cache_control": {
                                    "type": "ephemeral",
                                    "ttl": self.config.cache_ttl,
                                }
                            }
                        ]
                        msg_copy["content"] = new_content

            result.append(msg_copy)

        return result

    def process_response_usage(self, usage: dict[str, Any]) -> dict[str, Any]:
        """
        Process usage data from API response and update statistics.

        Args:
            usage: Usage dict from API response

        Returns:
            Processed usage with additional calculations
        """
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_write = usage.get("cache_creation_input_tokens", 0)
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        # Update stats
        self.stats.record_request(cache_read, cache_write, input_tokens)

        # Calculate costs
        total_input = cache_read + cache_write + input_tokens

        # Cost calculation
        cache_read_cost = (cache_read / 1_000_000) * self.PRICING["cache_read"]

        cache_write_cost = (cache_write / 1_000_000) * (
            self.PRICING["cache_write_1h"]
            if self.config.cache_ttl == "1h"
            else self.PRICING["cache_write_5m"]
        )

        input_cost = (input_tokens / 1_000_000) * self.PRICING["input"]
        output_cost = (output_tokens / 1_000_000) * self.PRICING["output"]

        # What it would have cost without caching
        baseline_input_cost = (total_input / 1_000_000) * self.PRICING["input"]

        return {
            **usage,
            "total_input_tokens": total_input,
            "estimated_cost_usd": cache_read_cost + cache_write_cost + input_cost + output_cost,
            "baseline_cost_usd": baseline_input_cost + output_cost,
            "savings_usd": max(0, baseline_input_cost - (cache_read_cost + cache_write_cost + input_cost)),
            "cache_hit": cache_read > 0,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics as a dictionary."""
        return {
            "total_requests": self.stats.total_requests,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "cache_writes": self.stats.cache_writes,
            "hit_rate": f"{self.stats.hit_rate:.1%}",
            "tokens_saved": self.stats.tokens_saved,
            "cost_saved_usd": f"${self.stats.cost_saved_usd:.4f}",
        }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self.stats = CacheStats()


def create_cached_content_block(
    text: str,
    ttl: Literal["5m", "1h"] = "5m"
) -> dict[str, Any]:
    """
    Create a content block with cache control.

    Args:
        text: The text content
        ttl: Cache TTL ("5m" or "1h")

    Returns:
        Content block dict with cache_control
    """
    return {
        "type": "text",
        "text": text,
        "cache_control": {
            "type": "ephemeral",
            "ttl": ttl,
        }
    }
