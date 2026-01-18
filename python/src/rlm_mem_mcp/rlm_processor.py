"""
RLM Processor for RLM-Mem MCP Server

Implements the Recursive Language Model (RLM) technique from arXiv:2512.24601.
This processor handles:
1. Chunking large inputs
2. Recursive sub-queries
3. Response aggregation
4. Memory of key findings

The key insight from RLM: store large content OUTSIDE the model's context,
then have the model write queries to peek at chunks and aggregate findings.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic
import tiktoken

from .config import RLMConfig
from .cache_manager import CacheManager
from .file_collector import CollectionResult


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

    This approach:
    - Stays within context limits
    - Reduces costs (only process relevant chunks)
    - Enables arbitrarily large inputs
    - Leaves context space for reasoning
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

Guidelines:
- Focus ONLY on the provided content
- Be specific: cite file names, line numbers, function names
- If content is not relevant, say so briefly
- Be concise but thorough"""

    AGGREGATION_PROMPT = """You are an expert at synthesizing information from multiple sources.
Your task is to combine findings into a coherent, well-organized response.

Guidelines:
- Remove redundancy while preserving important details
- Organize by theme or importance
- Cite specific sources when relevant
- Be comprehensive but concise
- If findings conflict, note the discrepancy"""

    def __init__(
        self,
        config: RLMConfig | None = None,
        cache_manager: CacheManager | None = None
    ):
        self.config = config or RLMConfig()
        self.cache_manager = cache_manager or CacheManager(self.config)
        self.client = anthropic.Anthropic(api_key=self.config.api_key)
        self._encoder: tiktoken.Encoding | None = None

        # Try to import the official RLM library if available
        self._rlm_available = False
        try:
            import rlm
            self._rlm_available = True
            self._rlm = rlm
        except ImportError:
            pass

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

        # Split on file markers but keep the marker with the content
        parts = content.split("### File:")
        for i, part in enumerate(parts):
            if i == 0 and not part.strip():
                continue

            file_content = ("### File:" + part) if i > 0 else part
            file_tokens = self.count_tokens(file_content)

            if self.count_tokens(current_chunk) + file_tokens <= max_tokens:
                current_chunk += file_content
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if file_tokens > max_tokens:
                    # Split this large file
                    chunks.extend(self._split_by_tokens(file_content, max_tokens))
                    current_chunk = ""
                else:
                    current_chunk = file_content

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

        for section in sections:
            section_tokens = self.count_tokens(section)
            current_tokens = self.count_tokens(current_chunk)

            if current_tokens + section_tokens <= max_tokens:
                current_chunk += (separator if current_chunk else "") + section
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if section_tokens > max_tokens:
                    # Section too large, split it further
                    chunks.extend(self._split_by_tokens(section, max_tokens))
                    current_chunk = ""
                else:
                    current_chunk = section

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

    async def process(
        self,
        query: str,
        collection: CollectionResult,
        progress_callback: Callable[[str], None] | None = None
    ) -> RLMResult:
        """
        Process a query against collected content using RLM technique.

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

        # Check if we can use the official RLM library
        if self._rlm_available and collection.total_tokens > 50000:
            try:
                return await self._process_with_rlm_library(
                    query, collection, scope, start_time, progress_callback
                )
            except Exception as e:
                if progress_callback:
                    progress_callback(f"RLM library failed, using fallback: {e}")

        # Use our built-in RLM implementation
        return await self._process_builtin(
            query, collection, scope, start_time, progress_callback
        )

    async def _process_with_rlm_library(
        self,
        query: str,
        collection: CollectionResult,
        scope: str,
        start_time: float,
        progress_callback: Callable[[str], None] | None
    ) -> RLMResult:
        """Process using the official RLM library."""
        # This would integrate with the actual RLM library
        # For now, fall back to built-in implementation
        raise NotImplementedError("RLM library integration pending")

    async def _process_builtin(
        self,
        query: str,
        collection: CollectionResult,
        scope: str,
        start_time: float,
        progress_callback: Callable[[str], None] | None
    ) -> RLMResult:
        """Process using built-in RLM implementation."""
        result = RLMResult(query=query, scope=scope)

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
        """Process all chunks with relevance assessment."""
        results = []

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(f"Processing chunk {i + 1}/{len(chunks)}...")

            start_time = time.time()

            # Assess relevance
            relevance = await self._assess_relevance(query, chunk)

            # Only process if relevant enough
            if relevance >= 0.3:
                response = await self._analyze_chunk(query, chunk)
            else:
                response = "(Skipped - low relevance)"

            processing_time = int((time.time() - start_time) * 1000)

            results.append(ChunkResult(
                chunk_id=i,
                content_preview=chunk[:200] + "..." if len(chunk) > 200 else chunk,
                response=response,
                token_count=self.count_tokens(chunk),
                relevance_score=relevance,
                processing_time_ms=processing_time,
            ))

        return results

    async def _assess_relevance(self, query: str, chunk: str) -> float:
        """Assess relevance of a chunk to the query."""
        # For very short chunks, assume relevant
        if self.count_tokens(chunk) < 100:
            return 0.5

        try:
            # Use a smaller preview for relevance assessment
            preview = chunk[:2000] if len(chunk) > 2000 else chunk

            system = self.cache_manager.build_cached_system(
                self.RELEVANCE_PROMPT
            )

            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=10,
                system=system,
                messages=[{
                    "role": "user",
                    "content": f"Query: {query}\n\nContent:\n{preview}\n\nRelevance (0.0-1.0):"
                }]
            )

            # Update cache stats
            self.cache_manager.process_response_usage(dict(response.usage))

            text = response.content[0].text if response.content else "0.5"
            score = float(text.strip())
            return max(0.0, min(1.0, score))

        except Exception:
            return 0.5  # Default to moderate relevance on error

    async def _analyze_chunk(self, query: str, chunk: str) -> str:
        """Analyze a chunk for information relevant to the query."""
        try:
            system = self.cache_manager.build_cached_system(
                self.CHUNK_ANALYSIS_PROMPT
            )

            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=1000,
                system=system,
                messages=[{
                    "role": "user",
                    "content": f"Query: {query}\n\nContent to analyze:\n{chunk}\n\nFindings:"
                }]
            )

            # Update cache stats
            self.cache_manager.process_response_usage(dict(response.usage))

            return response.content[0].text if response.content else ""

        except Exception as e:
            return f"(Error analyzing chunk: {e})"

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
            system = self.cache_manager.build_cached_system(
                self.AGGREGATION_PROMPT
            )

            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=2000,
                system=system,
                messages=[{
                    "role": "user",
                    "content": f"Query: {query}\n\nFindings from {len(findings)} sources:\n\n{combined}\n\nSynthesized response:"
                }]
            )

            # Update cache stats
            self.cache_manager.process_response_usage(dict(response.usage))

            return response.content[0].text if response.content else combined

        except Exception as e:
            # Fall back to just returning combined findings
            return f"Combined findings:\n\n{combined}\n\n(Aggregation error: {e})"

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
