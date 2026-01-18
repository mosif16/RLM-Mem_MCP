"""
RLM Processor for RLM-Mem MCP Server

Implements the Recursive Language Model (RLM) technique from arXiv:2512.24601.

THE KEY INSIGHT (from the paper):
- Content is stored as a VARIABLE in a Python REPL (NOT in LLM context)
- The LLM writes CODE to examine portions of the content
- Sub-LLM responses are stored in VARIABLES (NOT summarized)
- Full data is PRESERVED - the LLM can access any part at any time

This is NOT summarization! The data is kept intact and accessible.

Performance Optimizations:
- AsyncOpenAI client for true async API calls
- Exponential backoff retry for transient failures (429, 503)
- Circuit breaker for repeated failures
- Request timeout protection
- Connection pooling via httpx
- LRU cache for LLM responses
- Incremental token counting
- Early termination on relevance threshold
"""

import asyncio
import hashlib
import time
import random
from dataclasses import dataclass, field
from typing import Any, Callable
from functools import lru_cache
from collections import OrderedDict

from openai import AsyncOpenAI
import httpx
import tiktoken

from .config import RLMConfig
from .cache_manager import CacheManager
from .file_collector import CollectionResult


# Performance tuning constants
LLM_REQUEST_TIMEOUT_SECONDS = 120  # Max time for a single LLM call
MAX_RETRIES = 3  # Max retries for transient failures
INITIAL_RETRY_DELAY = 1.0  # Initial delay for exponential backoff
CIRCUIT_BREAKER_THRESHOLD = 5  # Failures before circuit opens
CIRCUIT_BREAKER_RESET_TIME = 60  # Seconds before circuit resets
LLM_CACHE_MAX_SIZE = 1000  # Max cached LLM responses
EARLY_TERMINATION_THRESHOLD = 0.9  # Stop if relevance exceeds this

# Rate limiting constants
RATE_LIMIT_REQUESTS_PER_MINUTE = 60  # Max requests per minute
RATE_LIMIT_TOKENS_PER_MINUTE = 100_000  # Max tokens per minute


@dataclass
class ChunkResult:
    """Result from processing a single chunk."""

    chunk_id: int
    content_preview: str  # First 200 chars
    response: str
    token_count: int
    relevance_score: float
    processing_time_ms: int


@dataclass
class RLMResult:
    """Final result from RLM processing."""

    query: str
    scope: str  # Description of what was processed
    response: str
    chunk_results: list[ChunkResult] = field(default_factory=list)
    total_tokens_processed: int = 0
    total_api_calls: int = 0
    cache_hits: int = 0
    processing_time_ms: int = 0
    truncated: bool = False
    error: str | None = None


class CircuitBreaker:
    """Circuit breaker pattern for API call protection."""

    def __init__(self, threshold: int = CIRCUIT_BREAKER_THRESHOLD, reset_time: int = CIRCUIT_BREAKER_RESET_TIME):
        self.threshold = threshold
        self.reset_time = reset_time
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.is_open = False

    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.threshold:
            self.is_open = True

    def record_success(self) -> None:
        """Record a success and reset failure count."""
        self.failure_count = 0
        self.is_open = False

    def can_proceed(self) -> bool:
        """Check if we can proceed with a request."""
        if not self.is_open:
            return True

        # Check if reset time has passed
        if time.time() - self.last_failure_time >= self.reset_time:
            self.is_open = False
            self.failure_count = 0
            return True

        return False

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "is_open": self.is_open,
            "failure_count": self.failure_count,
            "threshold": self.threshold,
        }


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Limits both request count and token throughput per minute.
    """

    def __init__(
        self,
        requests_per_minute: int = RATE_LIMIT_REQUESTS_PER_MINUTE,
        tokens_per_minute: int = RATE_LIMIT_TOKENS_PER_MINUTE
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute

        # Sliding window tracking
        self._request_times: list[float] = []
        self._token_usage: list[tuple[float, int]] = []  # (timestamp, tokens)

        # Stats
        self._total_requests = 0
        self._total_tokens = 0
        self._throttle_count = 0

    def _cleanup_old_entries(self, current_time: float) -> None:
        """Remove entries older than 1 minute."""
        cutoff = current_time - 60.0

        # Clean request times
        self._request_times = [t for t in self._request_times if t > cutoff]

        # Clean token usage
        self._token_usage = [(t, tokens) for t, tokens in self._token_usage if t > cutoff]

    def can_proceed(self, estimated_tokens: int = 0) -> tuple[bool, float]:
        """
        Check if a request can proceed under rate limits.

        Args:
            estimated_tokens: Estimated tokens for this request

        Returns:
            (can_proceed, wait_seconds) tuple
        """
        current_time = time.time()
        self._cleanup_old_entries(current_time)

        # Check request count limit
        if len(self._request_times) >= self.requests_per_minute:
            oldest = self._request_times[0]
            wait_time = 60.0 - (current_time - oldest)
            return False, max(0.1, wait_time)

        # Check token limit
        current_tokens = sum(tokens for _, tokens in self._token_usage)
        if current_tokens + estimated_tokens > self.tokens_per_minute:
            if self._token_usage:
                oldest = self._token_usage[0][0]
                wait_time = 60.0 - (current_time - oldest)
                return False, max(0.1, wait_time)

        return True, 0.0

    def record_request(self, tokens_used: int = 0) -> None:
        """Record a completed request."""
        current_time = time.time()
        self._request_times.append(current_time)
        if tokens_used > 0:
            self._token_usage.append((current_time, tokens_used))

        self._total_requests += 1
        self._total_tokens += tokens_used

    async def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Wait if rate limit would be exceeded."""
        can_proceed, wait_time = self.can_proceed(estimated_tokens)
        if not can_proceed:
            self._throttle_count += 1
            await asyncio.sleep(wait_time)

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        current_time = time.time()
        self._cleanup_old_entries(current_time)

        return {
            "requests_last_minute": len(self._request_times),
            "tokens_last_minute": sum(tokens for _, tokens in self._token_usage),
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "throttle_count": self._throttle_count,
            "requests_limit": self.requests_per_minute,
            "tokens_limit": self.tokens_per_minute,
        }


class LLMResponseCache:
    """LRU cache for LLM responses with TTL support."""

    def __init__(self, max_size: int = LLM_CACHE_MAX_SIZE, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _make_key(self, model: str, messages: list[dict], max_tokens: int) -> str:
        """Create cache key from request parameters."""
        content = f"{model}:{max_tokens}:{str(messages)}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, model: str, messages: list[dict], max_tokens: int) -> str | None:
        """Get cached response if available and not expired."""
        key = self._make_key(model, messages, max_tokens)

        if key in self._cache:
            response, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return response
            else:
                # Expired, remove it
                del self._cache[key]

        self._misses += 1
        return None

    def set(self, model: str, messages: list[dict], max_tokens: int, response: str) -> None:
        """Cache a response."""
        key = self._make_key(model, messages, max_tokens)

        # Remove oldest if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (response, time.time())

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "size": len(self._cache),
            "max_size": self.max_size,
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


class RLMProcessor:
    """
    Recursive Language Model processor.

    Implements the RLM technique for processing inputs that exceed
    the model's context window. Instead of sending everything at once,
    we:

    1. Split input into manageable chunks
    2. Assess relevance of each chunk to the query
    3. Process relevant chunks with sub-queries
    4. Aggregate results into a coherent response

    Performance features:
    - AsyncOpenAI for true async API calls
    - Exponential backoff retry with jitter
    - Circuit breaker for failure protection
    - LRU response caching
    - Connection pooling
    """

    # System prompts for different stages
    RELEVANCE_PROMPT = """You are a relevance assessor. Given a query and content chunk,
rate the relevance from 0.0 to 1.0. Only output a single decimal number.

0.0 = Completely irrelevant
0.3 = Marginally relevant
0.5 = Somewhat relevant
0.7 = Quite relevant
1.0 = Highly relevant"""

    CHUNK_ANALYSIS_PROMPT = """You are a precise information extractor analyzing code and text.
Your task is to find information relevant to the given query.

CRITICAL OUTPUT REQUIREMENTS:
1. ALWAYS include file:line format (e.g., `src/utils.py:42`)
2. ALWAYS include the actual code snippet showing the issue
3. ALWAYS assign a CONFIDENCE LEVEL to each finding
4. Format each finding as:
   ```
   **File:** path/to/file.ext:LINE_NUMBER [Confidence: HIGH/MEDIUM/LOW]
   **Issue:** Brief description
   **Code:**
   ```language
   <actual code from the content>
   ```
   ```

CONFIDENCE LEVELS (REQUIRED):
- **HIGH**: Code is in active blocks, line verified, finding clearly matches issue type
- **MEDIUM**: Context unclear, cannot verify reachability, partial pattern match
- **LOW**: Code in #if false/#if DEBUG blocks, line unverified, signature-only (no body)

DEAD CODE DETECTION:
- Check for #if false, #if 0, #if DEBUG, #ifdef NEVER blocks
- Code inside these blocks should be marked [Confidence: LOW - dead code]
- Swift/ObjC: #if false ... #endif
- C/C++: #if 0 ... #endif
- Python: if False: ...

Guidelines:
- Focus ONLY on the provided content
- Extract EXACT code snippets (copy from content, don't paraphrase)
- Count line numbers from file headers (### File: path starts at line 1)
- If content is not relevant, say "No relevant findings" briefly
- Be specific and actionable - readers should find the exact location
- If uncertain about a finding, use MEDIUM or LOW confidence"""

    AGGREGATION_PROMPT = """You are an expert at synthesizing information from multiple sources.
Your task is to combine findings into a coherent, well-organized response.

CRITICAL: PRESERVE ALL SPECIFIC DETAILS
- Keep ALL file:line references (e.g., `src/auth.py:42`)
- Keep ALL code snippets exactly as provided
- Keep ALL confidence levels [Confidence: HIGH/MEDIUM/LOW]
- Keep ALL severity ratings and categories
- Do NOT generalize specific findings into vague summaries

CONFIDENCE HANDLING:
- Group findings by confidence level (HIGH first, then MEDIUM, then LOW)
- LOW confidence findings should be labeled "Verify Manually"
- If a finding is in dead code (#if false, etc.), keep it LOW confidence

Output Format:
## Summary
Brief overview: X high-confidence, Y medium-confidence, Z low-confidence findings

## High Confidence Findings
These findings are verified and in active code.

### [Category]
**file.py:LINE** [Confidence: HIGH] - Description
```language
actual code snippet
```

## Medium Confidence Findings
These findings need additional verification.

### [Category]
**file.py:LINE** [Confidence: MEDIUM] - Description
```language
actual code snippet
```

## Low Confidence Findings (Verify Manually)
These may be false positives (dead code, unverified lines, etc.)

### [Category]
**file.py:LINE** [Confidence: LOW - reason] - Description
```language
actual code snippet
```

Guidelines:
- Group similar findings by category within each confidence level
- Remove exact duplicates only (same file, same line, same issue)
- Preserve unique findings even if similar
- If findings conflict, note the discrepancy with both locations
- Dead code findings (#if false, #if DEBUG) should always be LOW confidence"""

    def __init__(
        self,
        config: RLMConfig | None = None,
        cache_manager: CacheManager | None = None
    ):
        self.config = config or RLMConfig()
        self.cache_manager = cache_manager  # Keep for interface compatibility

        # Create async HTTP client with connection pooling
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(LLM_REQUEST_TIMEOUT_SECONDS),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

        # Create async OpenAI client with pooled connections
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base_url,
            http_client=self._http_client,
            default_headers={
                "HTTP-Referer": "https://github.com/mosif16/RLM-Mem_MCP",
                "X-Title": "RLM-Mem MCP Server"
            }
        )

        self._encoder: tiktoken.Encoding | None = None

        # Performance components
        self._circuit_breaker = CircuitBreaker()
        self._response_cache = LLMResponseCache()
        self._rate_limiter = RateLimiter()
        self._api_call_count = 0
        self._cache_hit_count = 0

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        await self._http_client.aclose()

    @property
    def encoder(self) -> tiktoken.Encoding:
        """Lazy-load tiktoken encoder."""
        if self._encoder is None:
            self._encoder = tiktoken.encoding_for_model("gpt-4")
        return self._encoder

    async def get_encoder_async(self) -> tiktoken.Encoding:
        """Get encoder without blocking event loop."""
        if self._encoder is None:
            self._encoder = await asyncio.to_thread(
                tiktoken.encoding_for_model, "gpt-4"
            )
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self.encoder.encode(text))

    async def count_tokens_async(self, text: str) -> int:
        """Count tokens without blocking event loop."""
        if not text:
            return 0
        encoder = await self.get_encoder_async()
        return await asyncio.to_thread(lambda: len(encoder.encode(text)))

    def split_into_chunks(
        self,
        content: str,
        max_chunk_tokens: int | None = None
    ) -> list[str]:
        """
        Split content into chunks of approximately equal token count.

        Uses intelligent splitting:
        1. Try to split on major boundaries (files, functions, sections)
        2. Fall back to paragraph splitting
        3. Final fallback to token-based splitting
        """
        max_tokens = max_chunk_tokens or self.config.max_chunk_tokens
        total_tokens = self.count_tokens(content)

        if total_tokens <= max_tokens:
            return [content]

        chunks = []

        # Try splitting by file markers first (### File: ...)
        if "### File:" in content:
            return self._split_by_file_markers(content, max_tokens)

        # Try splitting by major section markers
        section_markers = ["\n## ", "\n# ", "\n---\n", "\n\n\n"]
        for marker in section_markers:
            if marker in content:
                sections = content.split(marker)
                if len(sections) > 1:
                    chunks = self._merge_small_sections(
                        sections, marker, max_tokens
                    )
                    if chunks:
                        return chunks

        # Fall back to paragraph splitting
        paragraphs = content.split("\n\n")
        if len(paragraphs) > 1:
            chunks = self._merge_small_sections(paragraphs, "\n\n", max_tokens)
            if chunks:
                return chunks

        # Final fallback: split by token count
        return self._split_by_tokens(content, max_tokens)

    def _split_by_file_markers(
        self,
        content: str,
        max_tokens: int
    ) -> list[str]:
        """Split content that has file markers."""
        chunks = []
        current_chunk = ""
        current_tokens = 0  # Incremental token counting

        # Split on file markers but keep the marker with the content
        parts = content.split("### File:")
        for i, part in enumerate(parts):
            if i == 0 and not part.strip():
                continue

            file_content = ("### File:" + part) if i > 0 else part
            file_tokens = self.count_tokens(file_content)

            if current_tokens + file_tokens <= max_tokens:
                current_chunk += file_content
                current_tokens += file_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if file_tokens > max_tokens:
                    # Split this large file
                    chunks.extend(self._split_by_tokens(file_content, max_tokens))
                    current_chunk = ""
                    current_tokens = 0
                else:
                    current_chunk = file_content
                    current_tokens = file_tokens

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _merge_small_sections(
        self,
        sections: list[str],
        separator: str,
        max_tokens: int
    ) -> list[str]:
        """Merge small sections together until they approach max_tokens."""
        chunks = []
        current_chunk = ""
        current_tokens = 0  # Incremental token counting

        for section in sections:
            section_tokens = self.count_tokens(section)

            if current_tokens + section_tokens <= max_tokens:
                if current_chunk:
                    current_chunk += separator + section
                    current_tokens += section_tokens + self.count_tokens(separator)
                else:
                    current_chunk = section
                    current_tokens = section_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if section_tokens > max_tokens:
                    # Section too large, split it further
                    chunks.extend(self._split_by_tokens(section, max_tokens))
                    current_chunk = ""
                    current_tokens = 0
                else:
                    current_chunk = section
                    current_tokens = section_tokens

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_by_tokens(self, content: str, max_tokens: int) -> list[str]:
        """Split content by token count (fallback method)."""
        chunks = []
        tokens = self.encoder.encode(content)

        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks

    async def _call_llm_with_retry(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int
    ) -> str:
        """
        Call LLM API with retry logic, caching, and circuit breaker.

        Features:
        - LRU response caching
        - Exponential backoff with jitter
        - Circuit breaker for repeated failures
        - Timeout protection
        """
        # Check circuit breaker
        if not self._circuit_breaker.can_proceed():
            raise Exception("Circuit breaker is open - too many recent failures")

        # Check cache first
        cached = self._response_cache.get(model, messages, max_tokens)
        if cached is not None:
            self._cache_hit_count += 1
            return cached

        # Rate limiting - wait if needed
        estimated_tokens = max_tokens + sum(len(m.get("content", "")) // 4 for m in messages)
        await self._rate_limiter.wait_if_needed(estimated_tokens)

        # Retry loop with exponential backoff
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                self._api_call_count += 1

                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=messages
                    ),
                    timeout=LLM_REQUEST_TIMEOUT_SECONDS
                )

                result = response.choices[0].message.content if response.choices else ""

                # Record success, rate limit usage, and cache
                self._circuit_breaker.record_success()
                tokens_used = getattr(response.usage, 'total_tokens', max_tokens) if response.usage else max_tokens
                self._rate_limiter.record_request(tokens_used)
                self._response_cache.set(model, messages, max_tokens, result)

                return result

            except asyncio.TimeoutError:
                last_exception = Exception(f"Request timeout after {LLM_REQUEST_TIMEOUT_SECONDS}s")
                self._circuit_breaker.record_failure()

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Check if retryable error
                if any(code in error_str for code in ["429", "503", "502", "500", "rate"]):
                    self._circuit_breaker.record_failure()

                    if attempt < MAX_RETRIES - 1:
                        # Exponential backoff with jitter
                        delay = INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                        await asyncio.sleep(delay)
                        continue
                else:
                    # Non-retryable error
                    self._circuit_breaker.record_failure()
                    raise

        # All retries exhausted
        raise last_exception or Exception("Unknown error after retries")

    async def process(
        self,
        query: str,
        collection: CollectionResult,
        progress_callback: Callable[[str], None] | None = None
    ) -> RLMResult:
        """
        Process a query against collected content using RLM technique.

        Processing order (TRUE RLM first):
        1. REPL Mode - Content as variable, LLM writes code (TRUE RLM from paper)
        2. Agent SDK Mode - Subagents for parallel processing
        3. Built-in Mode - Sequential chunk processing (fallback)

        Args:
            query: The query/question to answer
            collection: Collected files/content to process
            progress_callback: Optional callback for progress updates

        Returns:
            RLMResult with the processed response
        """
        start_time = time.time()

        # Build scope description
        scope = f"{collection.file_count} files, ~{collection.total_tokens:,} tokens"

        if progress_callback:
            progress_callback(f"Processing {scope}...")

        # Get combined content
        content = collection.get_combined_content(include_headers=True)

        # 1. Try TRUE RLM (REPL-based) - content as variable, LLM writes code
        try:
            from .repl_environment import RLMReplEnvironment

            if progress_callback:
                progress_callback("Using TRUE RLM: Content as variable, LLM writes code...")

            repl = RLMReplEnvironment(self.config)
            repl.initialize(content)

            # Run the REPL session
            response = await repl.run_rlm_session(query)

            # Get all preserved sub-responses (NOT summarized!)
            all_responses = repl.get_all_responses()

            processing_time = int((time.time() - start_time) * 1000)

            return RLMResult(
                query=query,
                scope=scope,
                response=response,
                chunk_results=[],
                total_tokens_processed=collection.total_tokens,
                total_api_calls=len(all_responses),
                cache_hits=0,
                processing_time_ms=processing_time,
                truncated=False,
                error=None
            )

        except ImportError:
            if progress_callback:
                progress_callback("REPL environment not available, using built-in processor...")
        except Exception as e:
            if progress_callback:
                progress_callback(f"REPL mode failed: {e}, using built-in processor...")

        # 2. Fall back to built-in sequential processing
        if progress_callback:
            progress_callback(f"Using built-in processor with {self.config.model}...")

        return await self._process_builtin(
            query, collection, scope, start_time, progress_callback
        )

    async def _process_builtin(
        self,
        query: str,
        collection: CollectionResult,
        scope: str,
        start_time: float,
        progress_callback: Callable[[str], None] | None
    ) -> RLMResult:
        """Process using built-in RLM implementation."""
        result = RLMResult(query=query, scope=scope, response="")

        try:
            # Get combined content
            content = collection.get_combined_content(include_headers=True)

            # Split into chunks
            chunks = self.split_into_chunks(content)

            if progress_callback:
                progress_callback(f"Split into {len(chunks)} chunks")

            # Process chunks (with relevance filtering)
            chunk_results = await self._process_chunks(
                query, chunks, progress_callback
            )

            result.chunk_results = chunk_results
            result.total_tokens_processed = sum(
                self.count_tokens(c) for c in chunks
            )

            # Filter to relevant chunks
            relevant_results = [
                r for r in chunk_results if r.relevance_score >= 0.3
            ]

            if not relevant_results:
                result.response = "No relevant information found for the query."
            elif len(relevant_results) == 1:
                result.response = relevant_results[0].response
            else:
                # Aggregate results
                if progress_callback:
                    progress_callback(f"Aggregating {len(relevant_results)} relevant findings...")

                result.response = await self._aggregate_results(
                    query, relevant_results
                )
                result.total_api_calls += 1

            # Truncate if necessary
            result.response, result.truncated = self._truncate_response(
                result.response
            )

            # Update stats
            result.total_api_calls = self._api_call_count
            result.cache_hits = self._cache_hit_count

        except Exception as e:
            result.error = str(e)
            result.response = f"Error during RLM processing: {e}"

        result.processing_time_ms = int((time.time() - start_time) * 1000)
        return result

    async def _process_chunks(
        self,
        query: str,
        chunks: list[str],
        progress_callback: Callable[[str], None] | None
    ) -> list[ChunkResult]:
        """Process all chunks with relevance assessment in parallel."""
        if progress_callback:
            progress_callback(f"Processing {len(chunks)} chunks in parallel...")

        # Track if we found highly relevant content (for early termination)
        found_highly_relevant = asyncio.Event()

        async def process_single_chunk(i: int, chunk: str) -> ChunkResult:
            """Process a single chunk (relevance + analysis)."""
            start_time = time.time()

            # Assess relevance
            relevance = await self._assess_relevance(query, chunk)

            # Check for early termination signal
            if relevance >= EARLY_TERMINATION_THRESHOLD:
                found_highly_relevant.set()

            # Only process if relevant enough
            if relevance >= 0.3:
                response = await self._analyze_chunk(query, chunk)
            else:
                response = "(Skipped - low relevance)"

            processing_time = int((time.time() - start_time) * 1000)

            return ChunkResult(
                chunk_id=i,
                content_preview=chunk[:200] + "..." if len(chunk) > 200 else chunk,
                response=response,
                token_count=self.count_tokens(chunk),
                relevance_score=relevance,
                processing_time_ms=processing_time,
            )

        # Process all chunks in parallel
        tasks = [process_single_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)

        if progress_callback:
            relevant_count = sum(1 for r in results if r.relevance_score >= 0.3)
            progress_callback(f"Completed: {relevant_count}/{len(chunks)} chunks relevant")

        return list(results)

    async def _assess_relevance(self, query: str, chunk: str) -> float:
        """Assess relevance of a chunk to the query."""
        # For very short chunks, assume relevant
        if self.count_tokens(chunk) < 100:
            return 0.5

        try:
            # Use a smaller preview for relevance assessment
            preview = chunk[:2000] if len(chunk) > 2000 else chunk

            messages = [
                {"role": "system", "content": self.RELEVANCE_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nContent:\n{preview}\n\nRelevance (0.0-1.0):"}
            ]

            text = await self._call_llm_with_retry(
                model=self.config.model,
                messages=messages,
                max_tokens=10
            )

            score = float(text.strip())
            return max(0.0, min(1.0, score))

        except Exception as e:
            # Log the actual error for debugging
            import sys
            print(f"Relevance assessment error: {type(e).__name__}: {e}", file=sys.stderr)
            return 0.5  # Default to moderate relevance on error

    async def _analyze_chunk(self, query: str, chunk: str) -> str:
        """Analyze a chunk for information relevant to the query."""
        try:
            messages = [
                {"role": "system", "content": self.CHUNK_ANALYSIS_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nContent to analyze:\n{chunk}\n\nFindings:"}
            ]

            return await self._call_llm_with_retry(
                model=self.config.model,
                messages=messages,
                max_tokens=1000
            )

        except Exception as e:
            return f"(Error analyzing chunk: {type(e).__name__}: {e})"

    async def _aggregate_results(
        self,
        query: str,
        chunk_results: list[ChunkResult]
    ) -> str:
        """Aggregate findings from multiple chunks."""
        # Build combined findings
        findings = []
        for i, result in enumerate(chunk_results):
            if result.response and "(Skipped" not in result.response:
                findings.append(f"[Source {i + 1}]:\n{result.response}")

        combined = "\n\n---\n\n".join(findings)

        try:
            messages = [
                {"role": "system", "content": self.AGGREGATION_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nFindings from {len(findings)} sources:\n\n{combined}\n\nSynthesized response:"}
            ]

            return await self._call_llm_with_retry(
                model=self.config.aggregator_model,
                messages=messages,
                max_tokens=2000
            )

        except Exception as e:
            # Fall back to just returning combined findings
            return f"Combined findings:\n\n{combined}\n\n(Aggregation error: {type(e).__name__}: {e})"

    def _truncate_response(
        self,
        response: str,
        max_chars: int | None = None
    ) -> tuple[str, bool]:
        """Truncate response if too long."""
        # Convert max tokens to approximate chars (1 token ≈ 4 chars)
        max_chars = max_chars or (self.config.max_result_tokens * 4)

        if len(response) <= max_chars:
            return response, False

        # Keep first half and last quarter
        first_half = max_chars // 2
        last_quarter = max_chars // 4

        truncated = (
            response[:first_half] +
            f"\n\n[... TRUNCATED {len(response) - max_chars} characters ...]\n\n" +
            response[-last_quarter:]
        )

        return truncated, True

    def format_result(self, result: RLMResult) -> str:
        """Format an RLM result for output."""
        parts = [
            "## RLM Analysis Complete",
            "",
            f"**Query:** {result.query}",
            f"**Scope:** {result.scope}",
        ]

        if result.error:
            parts.extend([
                "",
                f"**Error:** {result.error}",
            ])

        parts.extend([
            "",
            "### Result",
            "",
            result.response,
        ])

        if result.truncated:
            parts.extend([
                "",
                "*[Response truncated - use more specific queries for details]*",
            ])

        # Add stats
        parts.extend([
            "",
            "---",
            f"*Processed in {result.processing_time_ms}ms | "
            f"{len(result.chunk_results)} chunks | "
            f"Cache hits: {result.cache_hits}*",
        ])

        return "\n".join(parts)

    def get_stats(self) -> dict[str, Any]:
        """Get processor statistics."""
        return {
            "api_calls": self._api_call_count,
            "cache_hits": self._cache_hit_count,
            "response_cache": self._response_cache.get_stats(),
            "circuit_breaker": self._circuit_breaker.get_status(),
            "rate_limiter": self._rate_limiter.get_stats(),
        }
