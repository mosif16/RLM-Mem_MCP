# Source Code Structure & Module Guide

Guide to the RLM-Mem MCP codebase structure and key modules.

## Directory Structure

```
RLM-Mem_MCP/
├── .claude/                              # Claude Code session & documentation
│   ├── docs/                             # Code documentation
│   ├── guides/                           # Usage guides
│   └── reference/                        # API reference
├── python/
│   ├── src/rlm_mem_mcp/
│   │   ├── __init__.py                   # Package initialization
│   │   ├── server.py                     # MCP server entry point & tool handlers
│   │   ├── rlm_processor.py              # Core RLM processing pipeline
│   │   ├── repl_environment.py           # Python REPL sandbox & llm_query()
│   │   ├── file_collector.py             # Async file collection & filtering
│   │   ├── cache_manager.py              # Anthropic-style prompt caching
│   │   ├── memory_store.py               # SQLite persistent memory
│   │   ├── structured_tools.py           # Pre-built analysis tools
│   │   ├── result_verifier.py            # Confidence scoring & verification
│   │   ├── content_analyzer.py           # Content analysis utilities
│   │   ├── config.py                     # Configuration from environment
│   │   ├── utils.py                      # Performance monitoring & helpers
│   │   ├── agent_pipeline.py             # Claude Agent SDK integration
│   │   ├── project_analyzer.py           # Project-level analysis
│   │   ├── fallback_analyzer.py          # Fallback analysis when LLM unavailable
│   │   ├── incremental_cache.py          # Incremental caching
│   │   └── structured_output.py          # Structured output formatting
│   ├── tests/                            # Test suite
│   ├── requirements.txt                  # Python dependencies
│   └── pyproject.toml                    # Project configuration
├── docs/                                 # Archive of documentation
├── .claude/sessions/                     # Claude Code session history (auto-managed)
├── CLAUDE.md                             # Main documentation (this project)
├── README.md                             # Quick start guide
└── .mcp.json                             # MCP configuration
```

---

## Key Modules

### 1. server.py - MCP Server & Tool Handlers

**Responsibility**: Entry point for MCP server, tool definitions, and request routing.

**Key Components**:
- `class RLMServer(Server)` - MCP server implementation
- `handle_call_tool()` - Route tool requests to handlers
- Tool definitions: `rlm_analyze`, `rlm_query_text`, `rlm_memory_store`, etc.

**Important Functions**:
```python
async def handle_call_tool(name, arguments):
    """Route tool requests to appropriate handler"""
    if name == "rlm_analyze":
        return await rlm_processor.process_files_with_query(...)
    elif name == "rlm_query_text":
        return await rlm_processor.process_text_with_query(...)
    # ... other tools
```

**When to Modify**:
- Adding new tools
- Changing tool parameters
- Adding new response formats
- Updating error handling

---

### 2. rlm_processor.py - Core RLM Pipeline

**Responsibility**: Orchestrate the complete RLM analysis pipeline.

**Key Components**:
- `class RLMProcessor` - Main orchestrator
- `process_files_with_query()` - Analyze files
- `process_text_with_query()` - Analyze text
- `_collect_files()` - Async file collection
- `_chunk_content()` - Smart chunking
- `_generate_analysis_code()` - LLM code generation
- `_aggregate_results()` - Result compilation

**Process Flow**:
```
Input → Collect Files → Chunk Content → Generate Code →
Execute in REPL → Aggregate Results → Verify → Output
```

**Key Methods**:
```python
async def process_files_with_query(paths, query, options):
    """Main file analysis method"""
    files = await self._collect_files(paths)
    chunks = self._chunk_content(files)
    code = await self._generate_analysis_code(query)
    results = await self._execute_and_collect(chunks, code)
    findings = await self._aggregate_and_verify(results)
    return findings

async def _collect_files(paths):
    """Collect files from paths using file_collector"""
    return await file_collector.collect_files(paths)

def _chunk_content(files):
    """Split content on semantic boundaries"""
    # Respects function/class/section boundaries
    # Maintains overlap for context
    # Preserves line numbers

async def _aggregate_results(findings):
    """Combine findings from all chunks"""
    # Deduplicate exact matches
    # Apply confidence scoring
    # Verify each finding
    # Sort by severity
```

**When to Modify**:
- Changing chunking strategy
- Modifying code generation prompts
- Adding new verification steps
- Changing result aggregation logic

**Related Modules**:
- `file_collector.py` - File collection
- `repl_environment.py` - Code execution
- `result_verifier.py` - Confidence scoring

---

### 3. repl_environment.py - Execution Sandbox

**Responsibility**: Provide safe Python execution environment with custom functions.

**Key Components**:
- `class REPLEnvironment` - Sandbox executor
- `execute()` - Run code in sandbox
- `llm_query()` - Query LLM for sub-analysis
- Result accumulation in `results` list

**Available in Executed Code**:
```python
prompt          # str - The content being analyzed
context         # dict - Metadata (filename, size, line ranges, etc.)
results         # list - Where findings are accumulated
llm_query()     # async func - Query LLM for portions
```

**Execution Restrictions**:
- No file system access (except pre-loaded `prompt`)
- No network access (except via `llm_query()`)
- 30-second timeout protection
- Memory limits enforced
- Restricted built-ins

**LLM Query Function**:
```python
async def llm_query(
    prompt_text: str,              # Content to analyze
    query: str,                    # What to find
    model: str = None,             # Override model
    confidence: str = "MEDIUM"     # Min confidence threshold
) -> dict:
    """Query LLM for specific portion analysis"""
    return {
        "findings": [...],
        "summary": "...",
        "metadata": {...}
    }
```

**Example Generated Code** (what LLM generates):
```python
# Generated by LLM, executed in REPL
import re

findings = []
for match in re.finditer(r"SELECT.*\+.*WHERE", prompt):
    line_num = prompt[:match.start()].count('\n') + 1
    findings.append({
        "line": line_num,
        "code": match.group(),
        "issue": "SQL injection via string concatenation"
    })

# Accumulate in results list
results.extend(findings)
```

**When to Modify**:
- Changing available functions
- Modifying execution restrictions
- Changing timeout values
- Adding new built-in modules

---

### 4. file_collector.py - Async File Collection

**Responsibility**: Efficiently traverse directories and collect files.

**Key Components**:
- `class FileCollector` - Main collector
- `collect_files()` - Async directory walk
- File filtering by extension and path
- Size and count limits

**Key Methods**:
```python
async def collect_files(
    paths: list[str],              # Root paths to scan
    extensions: set[str],          # File extensions to include
    skip_dirs: set[str],           # Directories to skip
    max_files: int = 1000,         # Maximum files to collect
    max_size_mb: int = 100         # Maximum total size
) -> list[FileInfo]:
    """Collect files from directories"""
    # Async walk filesystem
    # Filter by extension
    # Skip patterns (node_modules, .git, etc.)
    # Respect limits
    # Group by directory
```

**File Grouping Strategy**:
- Related files grouped by directory
- Large files analyzed separately
- Metadata preserved (size, modified time, line count)

**When to Modify**:
- Changing file type support
- Adding skip patterns
- Modifying collection limits
- Changing file grouping

---

### 5. structured_tools.py - Analysis Tools

**Responsibility**: Pre-built analysis functions available in REPL.

**Available Tools**:
- `find_secrets()` - Detect hardcoded credentials
- `find_sql_injection()` - SQL injection patterns
- `find_xss()` - XSS vulnerabilities
- `find_force_unwraps()` - iOS unsafe unwrapping
- `find_ios_memory_issues()` - Memory management
- `map_architecture()` - System architecture
- `analyze_performance()` - Performance issues
- `check_compliance()` - Standards compliance

**Tool Usage in Generated Code**:
```python
# Tool called from LLM-generated code
findings = find_secrets()

# Tool returns ToolResult with findings
for finding in findings.findings:
    results.append({
        "file": finding.file,
        "line": finding.line,
        "code": finding.code,
        "issue": finding.issue,
        "confidence": finding.confidence
    })
```

**Tool Structure**:
```python
class ToolResult:
    tool_name: str              # Name of tool
    findings: list[Finding]     # Results found
    summary: str                # Summary text
    metadata: dict              # Additional metadata
```

**When to Modify**:
- Adding new analysis tools
- Modifying confidence scoring
- Changing pattern detection
- Adding tool parameters

**Related Modules**:
- `result_verifier.py` - Confidence scoring

---

### 6. result_verifier.py - Quality Assurance

**Responsibility**: Verify findings and apply confidence scoring.

**Key Components**:
- `class ResultVerifier` - Main verifier
- `verify_findings()` - Verify and score results
- Confidence scoring algorithm (L11)
- False positive filtering

**Verification Strategies**:
1. **Pattern Verification** - Confirm regex matches
2. **Semantic Verification** - LLM context analysis
3. **False Positive Filtering** - Dead code, test files, comments
4. **Confidence Scoring** - L11 algorithm
5. **Deduplication** - Remove exact duplicates
6. **Completeness Check** - Ensure all chunks processed

**Confidence Scoring Algorithm**:
```
Start Score: 100

Deductions:
- In dead code: -50
- In test file: -30
- Pattern match only: -10
- In comment: -40
- Can't verify line: -20

Boosts:
- Semantic verification: +20
- Multiple indicators: +15

Final Scores:
- ≥80: HIGH
- 50-79: MEDIUM
- 20-49: LOW
- <20: FILTERED
```

**When to Modify**:
- Changing confidence thresholds
- Adding verification strategies
- Modifying deduplication logic
- Adding false positive patterns

---

### 7. cache_manager.py - Prompt Caching

**Responsibility**: Implement Anthropic-style prompt caching.

**Key Components**:
- `class CacheManager` - Cache handler
- `cache_prompt()` - Store cached content
- `get_cached_prompt()` - Retrieve from cache
- TTL management (5m / 1h)

**Caching Strategy**:
- System prompts cached with TTL
- LRU eviction when full
- Cost reduction ~90% on hits
- Automatic TTL refresh on access

**Cache Statistics**:
```python
{
    "hits": int,                    # Cache hits
    "misses": int,                  # Cache misses
    "hit_rate": float,              # Hit percentage
    "cached_prompts": int,          # Active cached items
    "avg_save_per_hit": str         # Estimated cost savings
}
```

**When to Modify**:
- Changing TTL values
- Modifying cache key generation
- Adjusting LRU limits
- Adding new cache strategies

---

### 8. memory_store.py - Persistent Memory

**Responsibility**: Store and retrieve findings persistently.

**Key Components**:
- `class MemoryStore` - SQLite backend
- `store()` - Persist finding
- `recall()` - Retrieve by key
- `search()` - Query by tags

**Database Schema**:
```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    key TEXT UNIQUE,               -- Unique identifier
    value TEXT,                    -- Content
    tags TEXT,                     -- CSV tags for searching
    created_at TIMESTAMP,          -- Creation time
    accessed_count INTEGER,        -- Access count
    last_accessed TIMESTAMP        -- Last access time
)
```

**Operations**:
```python
store(key, value, tags)            # Persist finding
recall(key)                         # Retrieve by key
search(tags)                        # Query by tags
list(tag)                          # List with tag
delete(key)                        # Remove finding
```

**When to Modify**:
- Changing storage schema
- Adding search capabilities
- Modifying retention policy
- Adding expiration logic

---

### 9. config.py - Configuration Management

**Responsibility**: Load and validate configuration from environment.

**Key Configuration**:
```python
OPENROUTER_API_KEY                # API authentication
RLM_MODEL                         # LLM model for analysis
RLM_AGGREGATOR_MODEL              # Model for aggregation
RLM_USE_CACHE                     # Cache enabled?
RLM_CACHE_TTL                     # Cache time-to-live
RLM_MAX_RESULT_TOKENS             # Max result size
RLM_MAX_CHUNK_TOKENS              # Max chunk size
RLM_OVERLAP_TOKENS                # Chunk overlap
RLM_EXTRA_SKIP_DIRS               # Additional skip patterns
RLM_SUPPORTED_EXTENSIONS          # File types to analyze
```

**When to Modify**:
- Adding new configuration options
- Changing default values
- Adding validation
- Supporting new environment formats

---

### 10. utils.py - Utilities & Monitoring

**Responsibility**: Performance monitoring and helper functions.

**Key Features**:
- Performance monitoring decorators
- Token counting utilities
- Text normalization
- Error handling helpers

**Common Utilities**:
```python
@monitor_performance
async def analyze_chunk():
    """Automatically monitor execution time"""
    pass

count_tokens(text)                 # Estimate token count
normalize_text(text)               # Clean text
format_findings(results)           # Format output
```

**When to Modify**:
- Adding new monitoring
- Changing token counting
- Adding text processing
- Modifying error handling

---

## Data Flow

### File Analysis Flow

```
Claude Code
    ↓
server.py (receive tool call)
    ↓
rlm_processor.py
    ├─ file_collector.py (collect files)
    ├─ repl_environment.py (setup sandbox)
    ├─ LLM: generate analysis code
    ├─ repl_environment.py (execute code)
    ├─ structured_tools.py (pre-built analysis)
    ├─ result_verifier.py (verify & score)
    ├─ cache_manager.py (cache results)
    ├─ memory_store.py (optionally persist)
    ↓
server.py (format response)
    ↓
Claude Code
```

### Text Analysis Flow

```
Claude Code (paste large text)
    ↓
server.py (receive rlm_query_text)
    ↓
rlm_processor.py
    ├─ repl_environment.py (store as prompt variable)
    ├─ LLM: generate analysis code
    ├─ repl_environment.py (execute code)
    ├─ result_verifier.py (verify & score)
    ↓
server.py (format response)
    ↓
Claude Code
```

---

## Adding Features

### Adding a New Structured Tool

1. **Define in structured_tools.py**:
```python
def find_my_pattern() -> ToolResult:
    """Find my custom pattern"""
    findings = []
    # Pattern detection logic
    return ToolResult(
        tool_name="find_my_pattern",
        findings=findings,
        summary=f"Found {len(findings)} instances"
    )
```

2. **Make available in REPL**:
   - Already available if defined in structured_tools.py
   - LLM-generated code can call directly

3. **Document in API_TOOLS.md**

### Adding a New Configuration Option

1. **Add to config.py**:
```python
NEW_OPTION = os.getenv("RLM_NEW_OPTION", "default_value")
```

2. **Update environment variable list in CLAUDE.md**

3. **Use in appropriate module**

---

## Testing Guide

### Running Tests
```bash
cd python
pip install -e ".[dev]"
pytest
```

### Test Structure
```
tests/
├── test_integration.py          # End-to-end tests
├── test_benchmark.py            # Performance tests
├── test_stress.py               # Stress tests
└── conftest.py                  # Fixtures & setup
```

---

## Performance Considerations

### Optimization Points
1. **File Collection**: Async I/O, limited by max_files
2. **Chunking**: Respects boundaries, minimizes overlap
3. **Code Generation**: Single LLM call per query
4. **Caching**: 90% cost reduction on hits
5. **Verification**: Parallel processing of findings

### Scaling
- Large codebases: Use narrower paths
- Long analyses: Break into multiple phases
- Cost optimization: Enable caching, use faster models

---

## Troubleshooting Guide

### Performance Issues
- Reduce paths: `["src/"]` instead of `["."]`
- Use faster model: `google/gemini-2-flash`
- Enable caching: `RLM_USE_CACHE=true`

### False Positives
- Increase confidence: `min_confidence: "HIGH"`
- Be more specific in query
- Exclude test directories

### Results Too Brief
- Increase tokens: `RLM_MAX_RESULT_TOKENS=8000`
- Use more capable model
- Ask for more detail in query

---

## See Also

- [CLAUDE.md](../../CLAUDE.md) - Main documentation
- [API Tools Reference](./../reference/API_TOOLS.md)
- [Common Queries](./../guides/COMMON_QUERIES.md)
