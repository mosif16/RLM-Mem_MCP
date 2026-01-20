# RLM-Mem Architecture

Complete technical architecture guide for the RLM-Mem MCP server.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Code (User)                       │
└─────────────────┬───────────────────────────────────────────────┘
                  │ MCP Protocol (JSON-RPC over stdio)
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RLM-Mem MCP Server                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ MCP Resource Handler (server.py)                         │  │
│  │ - Tool definitions (rlm_analyze, rlm_query_text, etc.)   │  │
│  │ - Request routing and validation                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ RLM Processor (rlm_processor.py)                         │  │
│  │ - File collection and chunking                           │  │
│  │ - Query orchestration                                    │  │
│  │ - Result aggregation                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ REPL Environment (repl_environment.py)                  │  │
│  │ - Python execution sandbox                              │  │
│  │ - llm_query() function for sub-queries                  │  │
│  │ - Results accumulation                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Support Layer                                            │  │
│  │ - Cache Manager (prompt caching)                         │  │
│  │ - File Collector (async traversal)                       │  │
│  │ - Memory Store (SQLite persistence)                      │  │
│  │ - Result Verifier (confidence scoring)                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────┬───────────────────────────────────────────────┘
                  │
        ┌─────────┼─────────┬─────────┐
        ▼         ▼         ▼         ▼
    ┌─────┐  ┌────────┐  ┌──────┐  ┌──────┐
    │Cache│  │OpenRouter │Anthropic│SQLite│
    │    │  │   API     │  API    │(Mem) │
    └─────┘  └────────┘  └──────┘  └──────┘
```

---

## Core Components

### 1. MCP Server (server.py)

**Responsibility**: Handle MCP protocol and resource management

**Key Functions**:
- `handle_call_tool()` - Route tool requests to appropriate handler
- Tool definitions for Claude Code
- Error handling and response formatting
- Resource initialization and cleanup

**Flow**:
```
MCP Request → Validation → Tool Handler → Processing → MCP Response
```

**Tools Exposed**:
- `rlm_analyze` - File/directory analysis
- `rlm_query_text` - Text block processing
- `rlm_memory_store` - Persist findings
- `rlm_memory_recall` - Retrieve findings
- `rlm_status` - Server health check

---

### 2. RLM Processor (rlm_processor.py)

**Responsibility**: Orchestrate the TRUE RLM analysis pipeline

**Architecture**:
```
Input (Files/Text)
    ↓
[File Collection] - Async directory walk, filter by extension
    ↓
[Chunking] - Respect boundaries, add overlap
    ↓
[Variable Storage] - Store as `prompt` in REPL
    ↓
[Code Generation] - LLM writes Python to analyze
    ↓
[Execution] - Run in sandboxed REPL environment
    ↓
[Result Aggregation] - Combine findings, apply confidence scoring
    ↓
Output (Findings with metadata)
```

**Key Methods**:
- `process_files_with_query()` - Main analysis method
- `_collect_files()` - Gather files matching extensions
- `_chunk_content()` - Split while respecting boundaries
- `_generate_analysis_code()` - Create Python for analysis
- `_aggregate_results()` - Combine and score findings

**Process Details**:

1. **Collection Phase** (file_collector.py)
   - Async walk directories
   - Filter by extension (configurable)
   - Skip patterns (node_modules, .git, etc.)
   - Respect max file limits

2. **Chunking Phase**
   - Split on semantic boundaries (functions, classes, sections)
   - Maintain overlap for context
   - Preserve line numbers
   - Group related chunks

3. **Storage Phase** (repl_environment.py)
   - Store content in `prompt` variable
   - Store metadata in `context` dict
   - Initialize results list

4. **Code Generation Phase**
   - Convert query to Python code
   - Use structured tools when applicable
   - Handle edge cases and error conditions

5. **Execution Phase**
   - Run in restricted REPL
   - Can call `llm_query()` for sub-queries
   - Accumulate results in `results` list

6. **Aggregation Phase**
   - Compile results from all chunks
   - Apply confidence scoring (structured_tools.py)
   - Verify findings (result_verifier.py)
   - Format for output

---

### 3. REPL Environment (repl_environment.py)

**Responsibility**: Provide execution sandbox with custom functions

**Key Components**:

```python
# Available in executed code:
prompt        # str - File/text content
context       # dict - Metadata about content
results       # list - Accumulate findings
llm_query()   # func - Query LLM for specific portions
```

**LLM Query Function**:
```python
async def llm_query(
    prompt_text: str,     # Content to analyze
    query: str,           # What to find
    model: str = None,    # Override model
    confidence: str = "MEDIUM"  # Min confidence
) -> dict:
    """
    Query LLM for analysis of specific content.
    Returns structured findings.
    """
```

**Execution Restrictions**:
- No access to file system (except via pre-loaded `prompt`)
- No network access (except `llm_query()`)
- Timeout protection (max 30s per chunk)
- Memory limits enforced

**Result Accumulation**:
```python
results.append({
    "file": filename,
    "line": 42,
    "code": code_snippet,
    "issue": description,
    "confidence": "HIGH",
    "severity": "CRITICAL"
})
```

---

### 4. Structured Tools (structured_tools.py)

**Responsibility**: Pre-built analysis functions available in REPL

**Available Tools**:
- `find_secrets()` - Detect hardcoded credentials
- `find_sql_injection()` - SQL injection patterns
- `find_xss()` - XSS vulnerabilities
- `find_force_unwraps()` - iOS unsafe unwrapping
- `map_architecture()` - System architecture
- `analyze_performance()` - Performance issues
- `check_compliance()` - Standards compliance

**Example Usage in Generated Code**:
```python
# Generated code in REPL:
findings = find_secrets()
for f in findings.high_confidence:
    results.append(f.to_dict())
```

**Confidence Scoring** (L11 - Verification Guardrail):
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

Final:
- ≥80: HIGH
- 50-79: MEDIUM
- 20-49: LOW
- <20: FILTERED (false positive)
```

---

### 5. Result Verifier (result_verifier.py)

**Responsibility**: Ensure result quality and accuracy

**Verification Strategies**:
1. **Pattern Verification** - Confirm regex matches
2. **Semantic Verification** - Context analysis
3. **False Positive Filtering** - Dead code, tests, comments
4. **Confidence Scoring** - L11 algorithm
5. **Deduplication** - Remove exact duplicates
6. **Completeness Check** - Ensure all chunks processed

**Verification Flow**:
```
Raw Findings
    ↓
[Pattern Verify] - Re-check regex matches
    ↓
[Semantic Verify] - LLM confirms meaning
    ↓
[Context Analysis] - Check for dead code/tests
    ↓
[Confidence Score] - Apply L11 algorithm
    ↓
[Filter] - Remove false positives
    ↓
[Deduplicate] - Remove duplicates
    ↓
Verified Findings
```

---

### 6. Cache Manager (cache_manager.py)

**Responsibility**: Implement Anthropic-style prompt caching

**Caching Strategy**:
- **System prompts** cached with 5m/1h TTL
- **Reduces cost** by ~90% on cache hits
- **Automatic management** - no manual cache busting needed
- **LRU Response Cache** - stores LLM responses

**Cache Key Generation**:
```python
cache_key = hash(
    model +
    analysis_query +
    file_content_hash
)
```

**TTL Options**:
- `5m` (default) - Active development
- `1h` - Less frequent access

**Statistics**:
```
hits: 45
misses: 12
hit_rate: 0.79 (79%)
avg_save_per_hit: ~$0.03
```

---

### 7. File Collector (file_collector.py)

**Responsibility**: Efficiently traverse directories and collect files

**Features**:
- **Async I/O** - Non-blocking file system access
- **Filtering** - By extension, by path pattern
- **Limits** - Max files, max total size
- **Caching** - Remember traversal results

**Process**:
```python
async def collect_files(
    paths: list[str],
    extensions: set[str],
    skip_dirs: set[str],
    max_files: int = 1000,
    max_size_mb: int = 100
) -> list[File]:
    # Async walk → Filter → Group → Return
```

**File Grouping**:
- Large files analyzed separately
- Related files grouped by directory
- Metadata (size, modified time) preserved

---

### 8. Memory Store (memory_store.py)

**Responsibility**: Persistent finding storage across conversations

**Backend**: SQLite
- Auto-created in `~/.rlm_mem/` directory
- Survives server restarts
- Indexed for fast lookup

**Schema**:
```sql
memories (
    id INTEGER PRIMARY KEY,
    key TEXT UNIQUE,
    value TEXT,
    tags TEXT (CSV),
    created_at TIMESTAMP,
    accessed_count INTEGER,
    last_accessed TIMESTAMP
)
```

**Operations**:
- `store(key, value, tags)` - Persist finding
- `recall(key)` - Retrieve by exact key
- `search(tags)` - Search by tags
- `list(tag)` - List all items with tag
- `delete(key)` - Remove finding

---

## Data Flow Examples

### Example 1: Security Audit

```
User Query: "Find SQL injection vulnerabilities"
    ↓
[Server] Calls rlm_analyze()
    ↓
[Processor] Collects .py, .js files
    ↓
[Processor] Chunks content with 200-token overlap
    ↓
[REPL] Stores content as `prompt` variable
    ↓
[Generator] Creates code:
    results = []
    patterns = [
        r"sql\s*=\s*['\"].*\+",  # String concat
        r"execute\s*\(\s*query",  # Dynamic execute
    ]
    for match in re.finditer(pattern, prompt):
        results.append({
            "line": line_number,
            "code": code_snippet,
            "issue": "SQL injection via string concatenation",
            ...
        })
    ↓
[REPL] Executes code, accumulates findings
    ↓
[Verifier] Checks each finding:
    - Is pattern real? ✓
    - Is it executable? ✓
    - Is it in test file? ✗
    - Is it in dead code? ✗
    - Confidence: HIGH
    ↓
[Aggregator] Combines across chunks, removes duplicates
    ↓
[Output] Returns findings with HIGH confidence
```

### Example 2: Architecture Mapping

```
User Query: "Map architecture with components and data flow"
    ↓
[Processor] Collects all project files
    ↓
[REPL] Stores everything in `prompt`
    ↓
[Generator] Creates code to analyze:
    - Module structure
    - Import relationships
    - Entry points
    - Data flow
    ↓
[Structured Tool] Uses find_architecture()
    ↓
[REPL] Builds dependency graph, traces flows
    ↓
[LLM Query] For complex architectural questions
    ↓
[Aggregator] Synthesizes into coherent architecture diagram
    ↓
[Output] Returns structured architecture document
```

### Example 3: Log Analysis

```
User: "Here's a 50MB log file. Find all errors."
    ↓
[Server] Calls rlm_query_text()
    ↓
[Processor] Stores log in `prompt` (no chunking needed)
    ↓
[Generator] Creates regex for log parsing:
    ERROR_PATTERN = r"\[ERROR\]\s+(.+)"
    for match in re.finditer(ERROR_PATTERN, prompt):
        ...
    ↓
[REPL] Executes, finds ~5000 ERROR entries
    ↓
[Aggregator] Groups by service, time range, error type
    ↓
[Output] Summary: "5000 errors across 3 services"
```

---

## Component Interaction

### Initialization Sequence

```
┌─ MCP Server Starts
│
├─ Load Configuration (config.py)
│  ├─ Read environment variables
│  └─ Build extension/dir filters
│
├─ Initialize Components
│  ├─ Cache Manager (connect to cache)
│  ├─ Memory Store (create SQLite if needed)
│  ├─ File Collector (prepare async executor)
│  └─ Rate Limiter (initialize buckets)
│
├─ Setup MCP Handler
│  ├─ Register tools
│  ├─ Register resources
│  └─ Setup request router
│
└─ Ready for requests
```

### Request Processing Sequence

```
MCP Request Received
    ↓
[Validation] Check parameters
    ↓
[Auth] Verify API keys
    ↓
[Rate Limiting] Check rate limits
    ↓
[Routing] Determine tool handler
    ↓
[Processing] Execute tool
    │   ├─ File collection (if needed)
    │   ├─ REPL setup
    │   ├─ Code generation
    │   ├─ Execution
    │   └─ Result verification
    ↓
[Aggregation] Combine results
    ↓
[Caching] Cache if applicable
    ↓
[Memory] Store if requested
    ↓
[Response] Format as MCP response
    ↓
Return to Claude Code
```

---

## Resilience Patterns

### Circuit Breaker
```
State: CLOSED (normal operation)
    ↓
[3 consecutive failures]
    ↓
State: OPEN (stop requests)
    ↓
[Wait 30 seconds]
    ↓
State: HALF_OPEN (test request)
    ↓
Success? → CLOSED
Failure? → OPEN (reset timer)
```

### Exponential Backoff
```
Attempt 1: Wait 1s
Attempt 2: Wait 2s
Attempt 3: Wait 4s
Attempt 4: Wait 8s
Attempt 5: Wait 16s
Max: 32s, Max attempts: 5
```

### Rate Limiting
```
Token bucket algorithm:
- Bucket size: 1,000,000 tokens/min
- Refill rate: Configured
- On request: Deduct tokens
- If insufficient: Reject (429)
```

---

## Performance Characteristics

### Time Complexity
- File collection: O(n) where n = file count
- Chunking: O(m) where m = total tokens
- Analysis: O(k) where k = chunk count (parallelizable)
- Aggregation: O(r) where r = result count

### Space Complexity
- File content: O(m) tokens in REPL
- Results: O(r) findings stored
- Cache: O(c) cached prompts (LRU limited)
- Memory store: O(s) stored findings (SQLite on disk)

### Real-World Performance
| Operation | Typical Time | Range |
|-----------|--------------|-------|
| Collect 100 files | 50ms | 10-200ms |
| Chunk 100k tokens | 100ms | 50-300ms |
| Analyze 10 chunks | 5-10s | 3-30s |
| Aggregate results | 100ms | 50-500ms |
| **Total** | **5-12s** | **3-35s** |

---

## Security Considerations

### Sandbox Security
- REPL executes in restricted Python environment
- No file system access (except `prompt` variable)
- No network access (except controlled `llm_query()`)
- Timeout protection (30s per chunk)
- Memory limits enforced

### API Security
- API keys stored in environment variables
- No keys in logs or error messages
- Credentials not returned to Claude Code
- Rate limiting prevents abuse

### Data Privacy
- File content never stored persistently (except REPL variable)
- Memory store stores only findings, not source code
- Cache expires automatically
- User can disable memory store

---

## Extension Points

### Adding New Structured Tools

1. Define in `structured_tools.py`:
```python
def find_my_pattern() -> ToolResult:
    """Find my custom pattern."""
    findings = []
    # Implementation
    return ToolResult(
        tool_name="find_my_pattern",
        findings=findings,
        summary=f"Found {len(findings)} instances"
    )
```

2. Use in generated code:
```python
results_tool = find_my_pattern()
for f in results_tool.findings:
    results.append(f.to_dict())
```

### Adding New Models

1. Update `config.py`:
```python
# Add to available models
SUPPORTED_MODELS = {
    "openrouter": ["google/gemini-2.5-flash-lite", "..."],
    "anthropic": ["claude-opus-4.5", "..."],
}
```

2. Use in configuration:
```bash
export RLM_MODEL=anthropic/claude-opus-4.5
```

---

## References

- [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) - Recursive Language Models paper
- [MCP Documentation](https://modelcontextprotocol.io/)
- [Anthropic Prompt Caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

See [API Reference](./API.md) and [Configuration Guide](./CONFIGURATION.md) for implementation details.
