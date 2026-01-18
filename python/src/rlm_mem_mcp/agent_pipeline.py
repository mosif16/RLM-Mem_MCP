"""
Claude Agent SDK Pipeline for RLM Processing

This module implements the RLM (Recursive Language Model) technique using
the Claude Agent SDK with subagents for efficient chunk processing.

Architecture:
- Uses Claude Haiku 4.5 for chunk processing (fast, included in Claude Max)
- Uses Claude Sonnet for final aggregation (when complex synthesis needed)
- Leverages subagents for parallel chunk processing
- Integrates with prompt caching for cost optimization

Claude Max Subscription Benefits:
- Haiku 4.5 calls are included at no additional API cost
- Makes RLM pipeline extremely cost-effective for large inputs
"""

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from .config import RLMConfig
from .file_collector import CollectionResult
from .cache_manager import CacheManager

# Type alias for message handler
MessageHandler = Callable[[str], None]


@dataclass
class AgentPipelineResult:
    """Result from the Agent SDK pipeline."""

    query: str
    scope: str
    response: str
    chunks_processed: int
    relevant_chunks: int
    total_tokens: int
    processing_time_ms: int
    model_used: str
    aggregator_model_used: str
    error: str | None = None


class RLMAgentPipeline:
    """
    RLM Pipeline using Claude Agent SDK with subagents.

    This pipeline:
    1. Splits large content into chunks
    2. Uses Haiku 4.5 subagents to assess relevance and analyze chunks
    3. Uses Sonnet to aggregate findings into coherent response
    4. Leverages prompt caching for repeated system prompts

    Optimized for Claude Max subscription where Haiku calls are included.

    Performance Optimizations:
    - Cached subagent definitions (created once per instance)
    - Batch relevance scoring support
    - File list mode for RAG-style processing
    """

    def __init__(
        self,
        config: RLMConfig | None = None,
        cache_manager: CacheManager | None = None
    ):
        self.config = config or RLMConfig()
        self.cache_manager = cache_manager or CacheManager(self.config)

        # Check if Agent SDK is available
        self._agent_sdk_available = False
        self._query_func = None
        self._ClaudeAgentOptions = None
        self._AgentDefinition = None

        # Cached subagent definitions (created once)
        self._cached_subagent_definitions: dict[str, Any] | None = None

        try:
            from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition
            self._agent_sdk_available = True
            self._query_func = query
            self._ClaudeAgentOptions = ClaudeAgentOptions
            self._AgentDefinition = AgentDefinition
        except ImportError:
            pass

    @property
    def is_available(self) -> bool:
        """Check if Agent SDK is available."""
        return self._agent_sdk_available

    def _create_subagent_definitions(self) -> dict[str, Any]:
        """Create subagent definitions for the pipeline (cached)."""
        # Return cached definitions if available
        if self._cached_subagent_definitions is not None:
            return self._cached_subagent_definitions

        if not self._AgentDefinition:
            return {}

        definitions = {
            # Relevance scorer - fast assessment using Haiku (supports batch)
            "relevance-scorer": self._AgentDefinition(
                description=(
                    "Quickly scores content chunks for relevance to a query. "
                    "Supports BATCH scoring: pass multiple file paths comma-separated. "
                    "Use this to filter which chunks need deeper analysis."
                ),
                prompt="""You are a relevance scorer. Given a query and content/file paths,
output relevance scores from 0.0 to 1.0.

Scoring guide:
- 0.0-0.2: Completely irrelevant
- 0.3-0.4: Marginally relevant
- 0.5-0.6: Somewhat relevant
- 0.7-0.8: Quite relevant
- 0.9-1.0: Highly relevant

For BATCH scoring (multiple files), output one line per file:
file_path: score

For single items, output just the number.

Be fast and concise - this is for filtering, not detailed analysis.""",
                tools=["Read"],
                model="haiku"  # Fast Haiku for scoring
            ),

            # Chunk analyzer - detailed analysis using Haiku
            "chunk-analyzer": self._AgentDefinition(
                description=(
                    "Analyzes individual code/text chunks in detail. "
                    "Use this to extract specific information from relevant chunks."
                ),
                prompt="""You are a precise information extractor analyzing code and text.

Guidelines:
- Focus ONLY on the provided content
- Be specific: cite file names, line numbers, function names
- Extract key findings relevant to the query
- Be concise but thorough
- If content is not relevant, say so briefly

Output structured findings that can be aggregated later.""",
                tools=["Read", "Grep"],
                model="haiku"  # Haiku for efficient chunk processing
            ),

            # Aggregator - synthesis using Sonnet for complex reasoning
            "aggregator": self._AgentDefinition(
                description=(
                    "Combines and synthesizes findings from multiple chunk analyses. "
                    "Use at the end to create a coherent final response."
                ),
                prompt="""You are an expert at synthesizing information from multiple sources.

Guidelines:
- Remove redundancy while preserving important details
- Organize findings by theme, severity, or importance
- Cite specific sources when relevant
- Be comprehensive but concise
- Note any conflicts or discrepancies in findings
- Create a well-structured response

Output a cohesive analysis that addresses the original query.""",
                tools=["Read"],
                model="sonnet"  # Sonnet for complex aggregation
            ),
        }

        # Cache the definitions for future calls
        self._cached_subagent_definitions = definitions
        return definitions

    async def process(
        self,
        query: str,
        collection: CollectionResult,
        progress_callback: MessageHandler | None = None
    ) -> AgentPipelineResult:
        """
        Process a query using the Agent SDK pipeline.

        Args:
            query: The query/question to answer
            collection: Collected files/content to process
            progress_callback: Optional callback for progress updates

        Returns:
            AgentPipelineResult with the processed response
        """
        import time
        start_time = time.time()

        scope = f"{collection.file_count} files, ~{collection.total_tokens:,} tokens"

        if not self._agent_sdk_available:
            return AgentPipelineResult(
                query=query,
                scope=scope,
                response="Claude Agent SDK not available. Install with: pip install claude-agent-sdk",
                chunks_processed=0,
                relevant_chunks=0,
                total_tokens=collection.total_tokens,
                processing_time_ms=0,
                model_used=self.config.model,
                aggregator_model_used=self.config.aggregator_model,
                error="Agent SDK not installed"
            )

        if progress_callback:
            progress_callback(f"Starting Agent SDK pipeline for {scope}...")

        try:
            # Get combined content
            content = collection.get_combined_content(include_headers=True)

            # Build the orchestration prompt
            orchestration_prompt = self._build_orchestration_prompt(query, content, collection)

            if progress_callback:
                progress_callback("Launching agent pipeline with subagents...")

            # Create agent options with subagents
            options = self._ClaudeAgentOptions(
                allowed_tools=["Read", "Grep", "Task"],
                agents=self._create_subagent_definitions(),
                model=self.config.model  # Default to Haiku
            )

            # Run the agent pipeline
            response_parts = []
            async for message in self._query_func(
                prompt=orchestration_prompt,
                options=options
            ):
                if hasattr(message, 'result'):
                    response_parts.append(str(message.result))
                elif hasattr(message, 'content'):
                    response_parts.append(str(message.content))

                if progress_callback and hasattr(message, 'type'):
                    progress_callback(f"Agent: {message.type}")

            final_response = "\n".join(response_parts) if response_parts else "No response generated"

            processing_time = int((time.time() - start_time) * 1000)

            return AgentPipelineResult(
                query=query,
                scope=scope,
                response=final_response,
                chunks_processed=collection.file_count,
                relevant_chunks=collection.file_count,  # Agent SDK handles filtering
                total_tokens=collection.total_tokens,
                processing_time_ms=processing_time,
                model_used=self.config.model,
                aggregator_model_used=self.config.aggregator_model,
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            return AgentPipelineResult(
                query=query,
                scope=scope,
                response=f"Error in Agent SDK pipeline: {str(e)}",
                chunks_processed=0,
                relevant_chunks=0,
                total_tokens=collection.total_tokens,
                processing_time_ms=processing_time,
                model_used=self.config.model,
                aggregator_model_used=self.config.aggregator_model,
                error=str(e)
            )

    def _build_orchestration_prompt(
        self,
        query: str,
        content: str,
        collection: CollectionResult
    ) -> str:
        """Build the orchestration prompt for the agent pipeline."""

        # Use RAG-style approach for large collections
        use_rag_style = collection.file_count > 20 or collection.total_tokens > 50000

        if use_rag_style:
            return self._build_rag_style_prompt(query, collection)
        else:
            return self._build_full_content_prompt(query, content, collection)

    def _build_rag_style_prompt(
        self,
        query: str,
        collection: CollectionResult
    ) -> str:
        """Build RAG-style prompt with file list (for large collections)."""

        file_summaries = collection.get_file_summaries()

        return f"""You are an RLM (Recursive Language Model) orchestrator analyzing a codebase.

## Query
{query}

## Available Files ({collection.file_count} files, ~{collection.total_tokens:,} tokens)

{file_summaries}

## Your Process (RAG-Style)

1. **Select Relevant Files**: Based on the file names and summaries above, identify which
   files are most likely relevant to the query. Use the `relevance-scorer` subagent to
   batch-score files if needed.

2. **Read & Analyze**: Use the Read tool to fetch content of relevant files, then use
   the `chunk-analyzer` subagent to extract detailed findings.

3. **Aggregate Results**: Use the `aggregator` subagent to synthesize findings.

## Batch Relevance Scoring

To efficiently score multiple files, provide a comma-separated list of file paths to evaluate.
Example: "src/auth.py, src/login.py, src/session.py"

## Instructions

1. Review the file list above
2. Identify the 5-10 most likely relevant files for: "{query}"
3. Read and analyze those files
4. Aggregate findings into a comprehensive response

Begin your analysis now."""

    def _build_full_content_prompt(
        self,
        query: str,
        content: str,
        collection: CollectionResult
    ) -> str:
        """Build full content prompt (for smaller collections)."""

        # Truncate content if too large for the prompt
        max_content_chars = 50000  # ~12k tokens
        if len(content) > max_content_chars:
            content_preview = content[:max_content_chars] + "\n\n[... content truncated for orchestration ...]"
        else:
            content_preview = content

        return f"""You are an RLM (Recursive Language Model) orchestrator. Your task is to
analyze the provided content and answer the query by coordinating subagents.

## Query
{query}

## Content Overview
- Total files: {collection.file_count}
- Total tokens: ~{collection.total_tokens:,}
- Content type: Code and text files

## Your Process

1. **Assess Relevance**: For each major section/file, use the `relevance-scorer` subagent
   to quickly determine if it's relevant to the query (score >= 0.3 means relevant).

2. **Analyze Relevant Chunks**: For chunks scoring >= 0.3, use the `chunk-analyzer` subagent
   to extract detailed findings related to the query.

3. **Aggregate Results**: Once all relevant chunks are analyzed, use the `aggregator` subagent
   to synthesize all findings into a coherent, comprehensive response.

## Content to Analyze

{content_preview}

## Instructions

Execute this analysis pipeline:
1. First, identify which sections are most relevant to: "{query}"
2. Analyze those sections in detail
3. Aggregate all findings into a final response

Begin your analysis now."""

    def format_result(self, result: AgentPipelineResult) -> str:
        """Format the pipeline result for output."""
        parts = [
            "## RLM Agent Pipeline Complete",
            "",
            f"**Query:** {result.query}",
            f"**Scope:** {result.scope}",
            f"**Model:** {result.model_used} (chunks), {result.aggregator_model_used} (aggregation)",
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
            "",
            "---",
            f"*Processed in {result.processing_time_ms}ms | "
            f"{result.chunks_processed} chunks | "
            f"{result.relevant_chunks} relevant*",
        ])

        return "\n".join(parts)


# Standalone function for direct use
async def run_rlm_pipeline(
    query: str,
    paths: list[str],
    config: RLMConfig | None = None,
    progress_callback: MessageHandler | None = None
) -> AgentPipelineResult:
    """
    Run the RLM Agent SDK pipeline on specified paths.

    This is a convenience function for direct use without going through MCP.

    Args:
        query: What to find/analyze
        paths: File or directory paths to analyze
        config: Optional RLM configuration
        progress_callback: Optional progress callback

    Returns:
        AgentPipelineResult with findings

    Example:
        result = await run_rlm_pipeline(
            query="Find security vulnerabilities",
            paths=["./src"],
            progress_callback=print
        )
        print(result.response)
    """
    from .file_collector import FileCollector

    config = config or RLMConfig()
    collector = FileCollector(config)
    pipeline = RLMAgentPipeline(config)

    # Collect files
    collection = collector.collect_paths(paths)

    if collection.file_count == 0:
        return AgentPipelineResult(
            query=query,
            scope="0 files",
            response="No matching files found in the specified paths.",
            chunks_processed=0,
            relevant_chunks=0,
            total_tokens=0,
            processing_time_ms=0,
            model_used=config.model,
            aggregator_model_used=config.aggregator_model,
            error="No files found"
        )

    # Run pipeline
    return await pipeline.process(query, collection, progress_callback)


# CLI entry point for testing
if __name__ == "__main__":
    import sys

    async def main():
        if len(sys.argv) < 3:
            print("Usage: python -m rlm_mem_mcp.agent_pipeline <query> <path1> [path2...]")
            sys.exit(1)

        query = sys.argv[1]
        paths = sys.argv[2:]

        print(f"Running RLM Agent Pipeline...")
        print(f"Query: {query}")
        print(f"Paths: {paths}")
        print()

        result = await run_rlm_pipeline(
            query=query,
            paths=paths,
            progress_callback=lambda msg: print(f"  > {msg}")
        )

        pipeline = RLMAgentPipeline()
        print(pipeline.format_result(result))

    asyncio.run(main())
