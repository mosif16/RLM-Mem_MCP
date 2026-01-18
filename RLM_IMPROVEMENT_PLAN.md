# RLM MCP Tool - Deep Investigation & Improvement Plan

## Executive Summary

After deep investigation of the RLM-Mem MCP codebase, I've identified **5 root causes** for the reported issues and designed **12 concrete solutions** to raise the tool from 6/10 to 9/10 reliability.

---

## Part 1: Root Cause Analysis

### Issue 1: Files Are Missed (PaywallView.swift, Widget Extensions, etc.)

**Location**: `repl_environment.py:857-874`, `file_collector.py:362-391`

**Root Cause**: The problem is NOT in file collection (which works correctly) but in the **REPL session's file discovery presentation**:

```python
# repl_environment.py:857-859 - Only shows first 30 files!
file_list_preview = "\n".join(f"  - {f}" for f in actual_files[:30])
if len(actual_files) > 30:
    file_list_preview += f"\n  ... and {len(actual_files) - 30} more files"
```

The LLM only sees 30 file paths upfront. If `PaywallView.swift` is file #45, the LLM never sees it unless it explicitly searches.

**Secondary Cause**: No cross-referencing with project metadata (CLAUDE.md, Package.swift, Xcode project files) which explicitly list important files.

---

### Issue 2: Cache Hits = 0

**Location**: `rlm_processor.py:255-312`

**Root Cause**: The `LLMResponseCache` creates keys from `hash(model + max_tokens + messages)`:

```python
# rlm_processor.py:266-268
def _make_key(self, model: str, messages: list[dict], max_tokens: int) -> str:
    content = f"{model}:{max_tokens}:{str(messages)}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]
```

Every analysis generates **unique messages** (different chunks, different sub-queries). The cache only hits on EXACT identical calls, which never happen during normal operation.

**The cache is architecturally wrong for this use case** - it's designed for repeated identical requests, not for semantic similarity.

---

### Issue 3: Deep Dives 14x Slower Than Broad Scans

**Location**: `repl_environment.py:832-964`

**Root Cause**: The REPL session uses iterative LLM reasoning with `max_iterations = 25`. Performance profile:

| Query Type | LLM Iterations | Sub-LLM Calls | Verification Calls |
|------------|----------------|---------------|-------------------|
| Broad scan | 3-5 | 2-4 | 0-2 |
| Deep dive | 15-25 | 10-20 | 20-40 |

Deep dives trigger:
1. More `verify_line()` calls per finding
2. More `check_dead_code()` calls per file
3. More `is_implemented()` calls per function
4. More `llm_query()` sub-calls for detailed analysis

The verification helpers (good for accuracy) create an O(n²) pattern where n = findings.

---

### Issue 4: Hallucinated Code Snippets

**Location**: `repl_environment.py:457-487`, `rlm_processor.py:1085-1113`

**Root Causes**:

1. **No post-verification of LLM output**: Results from `llm_query()` are stored directly without checking if file:line references actually exist.

2. **Aggregation introduces new hallucinations**: The `_aggregate_results()` function sends chunk findings to another LLM call which can "improve" or hallucinate details.

3. **Line counting errors**: LLM estimates line numbers by counting in code blocks rather than using the actual `### File:` markers with line offsets.

4. **Hallucination markers insufficient**:
```python
# repl_environment.py:467-476
hallucination_markers = [
    "example:", "for example,", "hypothetically", ...
]
```
These markers catch obvious cases but miss subtle hallucinations like wrong line numbers or templated code.

---

### Issue 5: Widget Extension Blind Spot

**Location**: `file_collector.py:232-237`

**Root Cause**: The `should_skip_directory()` function checks against patterns:

```python
skipped_directories: Set[str] = {
    ".git", "node_modules", "__pycache__", "venv", ".venv",
    "dist", "build", ".next", "target", "vendor", ...
}
```

The widget extension directory ("thebill widgetExtension") is NOT in skip list. The real issue is:
1. Directory names with spaces may have encoding issues
2. The LLM's regex patterns (`### File: ([^\n]+)`) correctly capture paths but the LLM doesn't search for "widget" or "extension" keywords

---

## Part 2: Solution Designs

### Solution 1: Smart File Discovery with Project Metadata

**Problem**: LLM only sees first 30 files

**Solution**: Add project structure awareness

```python
# New file: python/src/rlm_mem_mcp/project_analyzer.py

class ProjectAnalyzer:
    """Extracts project structure from metadata files."""

    METADATA_FILES = [
        "CLAUDE.md", "README.md",  # Documentation
        "Package.swift", "*.xcodeproj/project.pbxproj",  # iOS
        "package.json", "tsconfig.json",  # JS/TS
        "Cargo.toml", "pyproject.toml",  # Rust/Python
    ]

    def get_key_files(self, collection: CollectionResult) -> list[str]:
        """Extract key files mentioned in project metadata."""
        key_files = []

        for f in collection.files:
            if any(f.relative_path.endswith(m.replace("*", ""))
                   for m in self.METADATA_FILES):
                # Parse and extract referenced files
                key_files.extend(self._extract_file_references(f.content))

        return key_files

    def get_file_categories(self, collection: CollectionResult) -> dict[str, list[str]]:
        """Categorize files by type/purpose."""
        return {
            "views": [f for f in paths if "View" in f or "views" in f.lower()],
            "models": [f for f in paths if "Model" in f or "models" in f.lower()],
            "services": [f for f in paths if "Service" in f or "Manager" in f],
            "tests": [f for f in paths if "test" in f.lower() or "spec" in f.lower()],
            "config": [f for f in paths if any(ext in f for ext in [".json", ".yaml", ".toml"])],
            "extensions": [f for f in paths if "extension" in f.lower() or "widget" in f.lower()],
        }
```

**Integration Point**: Modify `repl_environment.py:850-874` to include categorized file list.

---

### Solution 2: Semantic Cache with Embeddings

**Problem**: Cache only hits on exact matches

**Solution**: Implement semantic similarity caching

```python
# Enhanced cache in rlm_processor.py

class SemanticCache:
    """Cache with semantic similarity matching."""

    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self._cache: dict[str, tuple[str, list[float], float]] = {}  # key -> (response, embedding, timestamp)
        self._embedder = None  # Lazy-load sentence-transformers

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for cache key (query + context summary)."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, small
        return self._embedder.encode(text).tolist()

    def get_similar(self, query: str, context_summary: str) -> str | None:
        """Find cached response with similar query+context."""
        key_text = f"{query}\n---\n{context_summary[:500]}"
        query_embedding = self._get_embedding(key_text)

        best_match = None
        best_similarity = 0.0

        for cached_key, (response, cached_embedding, timestamp) in self._cache.items():
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            if similarity > best_similarity and similarity >= self.threshold:
                best_similarity = similarity
                best_match = response

        return best_match
```

**Trade-off**: Adds ~50ms latency for embedding lookup but can save 10-100x on cache hits.

---

### Solution 3: Batched Verification to Fix O(n²) Performance

**Problem**: Deep dives call verification helpers individually

**Solution**: Batch verification calls

```python
# Enhanced repl_environment.py

def _create_batch_verify_function(self) -> Callable:
    """Create batched verification for multiple findings at once."""

    def batch_verify(findings: list[dict]) -> list[dict]:
        """
        Verify multiple findings in one call.

        Args:
            findings: List of {"file": str, "line": int, "pattern": str}

        Returns:
            List of verification results
        """
        results = []

        # Group by file to minimize file parsing
        by_file = defaultdict(list)
        for f in findings:
            by_file[f["file"]].append(f)

        for filepath, file_findings in by_file.items():
            # Parse file once
            content = self._get_file_content(filepath)
            if not content:
                results.extend([{"is_valid": False, "reason": "File not found"}
                               for _ in file_findings])
                continue

            # Compute dead code regions once per file
            dead_regions = find_dead_code_regions(content, filepath)

            # Verify all lines in this file
            for finding in file_findings:
                result = verify_line_reference(
                    content, filepath, finding["line"],
                    finding.get("pattern"), dead_regions
                )
                results.append(result)

        return results

    return batch_verify
```

**Expected Impact**: Reduces verification time from O(n * m) to O(n + m) where n=findings, m=files.

---

### Solution 4: Post-Processing Verification Layer

**Problem**: LLM outputs aren't verified against actual content

**Solution**: Add mandatory verification pass

```python
# New file: python/src/rlm_mem_mcp/result_verifier.py

class ResultVerifier:
    """Verifies RLM findings against actual file content."""

    def verify_findings(self, response: str, collection: CollectionResult) -> tuple[str, dict]:
        """
        Verify all file:line references in response.

        Returns:
            (verified_response, verification_stats)
        """
        # Extract all file:line references
        references = re.findall(r'([^\s:]+\.[a-z]+):(\d+)', response, re.IGNORECASE)

        verified_count = 0
        invalid_count = 0
        corrections = []

        for filepath, line_str in references:
            line_num = int(line_str)

            # Find actual file content
            actual_content = collection.get_file_content(filepath)
            if not actual_content:
                # Try fuzzy match
                actual_content, matched_path = self._fuzzy_find_file(filepath, collection)
                if matched_path:
                    corrections.append((filepath, matched_path))

            if actual_content:
                lines = actual_content.split('\n')
                if 1 <= line_num <= len(lines):
                    verified_count += 1
                else:
                    invalid_count += 1
                    # Add warning annotation
                    response = response.replace(
                        f"{filepath}:{line_num}",
                        f"{filepath}:{line_num} [UNVERIFIED - line out of range]"
                    )
            else:
                invalid_count += 1
                response = response.replace(
                    f"{filepath}:{line_num}",
                    f"{filepath}:{line_num} [UNVERIFIED - file not found]"
                )

        # Apply corrections
        for old_path, new_path in corrections:
            response = response.replace(old_path, new_path)

        stats = {
            "total_references": len(references),
            "verified": verified_count,
            "invalid": invalid_count,
            "corrections_made": len(corrections),
        }

        return response, stats
```

---

### Solution 5: Improved Query Decomposition

**Problem**: Broad queries produce surface-level results

**Current State**: `rlm_processor.py:553-610` already has decomposition, but it's passive

**Solution**: Make decomposition smarter with project-aware sub-queries

```python
# Enhanced rlm_processor.py

QUERY_DECOMPOSITIONS_V2 = {
    "security": {
        "sub_queries": [
            "Find INJECTION: SQL via string concat, command via subprocess/exec",
            "Find SECRETS: hardcoded API keys, passwords, tokens",
            "Find AUTH issues: missing checks, weak sessions",
            "Find DATA exposure: logging sensitive data, insecure storage",
        ],
        "file_hints": ["auth", "login", "session", "api", "database", "secret"],
    },
    "ios_app": {
        "sub_queries": [
            "Find StoreKit/IAP integration in *Manager.swift and *Service.swift",
            "Find UI components in *View.swift and *ViewController.swift",
            "Find data models in *Model.swift and Core Data files",
            "Find extension targets in *Extension directories",
        ],
        "file_hints": ["View", "Controller", "Manager", "Model", "Extension", "Widget"],
    },
}

async def decompose_with_file_hints(
    self,
    query: str,
    collection: CollectionResult
) -> list[tuple[str, list[str]]]:
    """
    Decompose query into sub-queries with relevant file subsets.

    Returns:
        List of (sub_query, relevant_file_paths) tuples
    """
    analysis = self.analyze_query_quality(query)

    if not analysis["is_broad"]:
        return [(query, collection.get_file_list())]

    decomposition = QUERY_DECOMPOSITIONS_V2.get(
        analysis["query_type"],
        QUERY_DECOMPOSITIONS_V2["security"]
    )

    result = []
    for sub_query in decomposition["sub_queries"]:
        # Find files matching hints for this sub-query
        hints = decomposition["file_hints"]
        relevant_files = [
            f for f in collection.get_file_list()
            if any(hint.lower() in f.lower() for hint in hints)
        ]
        result.append((sub_query, relevant_files or collection.get_file_list()[:50]))

    return result
```

---

### Solution 6: Progress Streaming

**Problem**: 144s with no intermediate output is poor UX

**Solution**: Implement streaming callbacks throughout the pipeline

```python
# Enhanced rlm_processor.py and repl_environment.py

@dataclass
class ProgressEvent:
    """Structured progress event for streaming."""
    event_type: str  # "file_collected", "chunk_analyzed", "finding_verified", etc.
    message: str
    progress_percent: float
    details: dict = field(default_factory=dict)

class StreamingRLMProcessor:
    """RLM processor with streaming progress."""

    async def process_with_streaming(
        self,
        query: str,
        collection: CollectionResult,
        event_callback: Callable[[ProgressEvent], None]
    ) -> RLMResult:
        """Process with streaming progress events."""

        total_steps = collection.file_count + 5  # files + analysis steps
        current_step = 0

        # Step 1: File collection complete
        event_callback(ProgressEvent(
            event_type="collection_complete",
            message=f"Collected {collection.file_count} files",
            progress_percent=10,
            details={"file_count": collection.file_count}
        ))

        # Step 2: Chunking
        chunks = self.split_into_chunks(content)
        event_callback(ProgressEvent(
            event_type="chunking_complete",
            message=f"Split into {len(chunks)} chunks",
            progress_percent=20,
        ))

        # Step 3: Process chunks with per-chunk updates
        for i, chunk in enumerate(chunks):
            result = await self._process_single_chunk(chunk, query)
            current_step += 1
            event_callback(ProgressEvent(
                event_type="chunk_analyzed",
                message=f"Analyzed chunk {i+1}/{len(chunks)}",
                progress_percent=20 + (60 * current_step / len(chunks)),
                details={"chunk_id": i, "relevance": result.relevance_score}
            ))

        # ... continue with streaming for aggregation, verification, etc.
```

---

### Solution 7: Enhanced REPL File Context

**Problem**: LLM only sees 30 files, misses important ones

**Solution**: Provide categorized, searchable file index

```python
# Enhanced repl_environment.py:850-874

def _build_enhanced_file_context(self) -> str:
    """Build comprehensive file context for REPL."""
    files = self._extract_file_list()

    # Categorize files
    categories = {
        "Views/UI": [],
        "Models/Data": [],
        "Services/Managers": [],
        "Extensions/Widgets": [],
        "Tests": [],
        "Config": [],
        "Other": [],
    }

    for f in files:
        f_lower = f.lower()
        if "view" in f_lower or "controller" in f_lower or "ui" in f_lower:
            categories["Views/UI"].append(f)
        elif "model" in f_lower or "entity" in f_lower or "data" in f_lower:
            categories["Models/Data"].append(f)
        elif "service" in f_lower or "manager" in f_lower or "provider" in f_lower:
            categories["Services/Managers"].append(f)
        elif "extension" in f_lower or "widget" in f_lower:
            categories["Extensions/Widgets"].append(f)
        elif "test" in f_lower or "spec" in f_lower:
            categories["Tests"].append(f)
        elif any(ext in f for ext in [".json", ".yaml", ".toml", ".plist"]):
            categories["Config"].append(f)
        else:
            categories["Other"].append(f)

    # Build context string
    context_parts = [f"## File Index ({len(files)} total files)\n"]

    for category, cat_files in categories.items():
        if cat_files:
            context_parts.append(f"\n### {category} ({len(cat_files)} files)")
            for f in cat_files[:15]:  # Show up to 15 per category
                context_parts.append(f"  - {f}")
            if len(cat_files) > 15:
                context_parts.append(f"  ... and {len(cat_files) - 15} more")

    context_parts.append("\n\n**To find files**: Use `[f for f in files if 'keyword' in f.lower()]`")

    return "\n".join(context_parts)
```

---

## Part 3: Prioritized Implementation Plan

### Priority 1: Critical Fixes (Impact: High, Effort: Low)

| # | Fix | File | Effort | Impact |
|---|-----|------|--------|--------|
| 1 | Enhanced file context | `repl_environment.py:850-874` | 2h | High |
| 2 | Post-verification layer | New `result_verifier.py` | 4h | High |
| 3 | Batched verification | `repl_environment.py` | 3h | High |

### Priority 2: Performance Improvements (Impact: High, Effort: Medium)

| # | Fix | File | Effort | Impact |
|---|-----|------|--------|--------|
| 4 | Progress streaming | `rlm_processor.py`, `server.py` | 6h | Medium |
| 5 | Project metadata awareness | New `project_analyzer.py` | 4h | High |
| 6 | Improved decomposition | `rlm_processor.py:553-610` | 4h | Medium |

### Priority 3: Architecture Improvements (Impact: Medium, Effort: High)

| # | Fix | File | Effort | Impact |
|---|-----|------|--------|--------|
| 7 | Semantic cache | `rlm_processor.py:255-312` | 8h | Medium |
| 8 | Parallel chunk processing | `rlm_processor.py:987-1036` | 6h | Medium |
| 9 | Incremental results | `server.py`, MCP protocol | 8h | Low |

---

## Part 4: Quick Wins (Can Implement Immediately)

### Quick Win 1: Fix 30-file limit (5 minutes)

```python
# repl_environment.py:857-859 - Change from 30 to 100, add categories
file_list_preview = "\n".join(f"  - {f}" for f in actual_files[:100])
```

### Quick Win 2: Add widget/extension awareness (10 minutes)

```python
# repl_environment.py - Add to SYSTEM_PROMPT
IMPORTANT_FILE_PATTERNS = """
## File Patterns to Watch For
- *Extension*: App extensions (widgets, share, etc.)
- *Widget*: Widget implementations
- *Manager.swift, *Service.swift: Core business logic
- *View.swift, *ViewController.swift: UI components
"""
```

### Quick Win 3: Better hallucination detection (15 minutes)

```python
# repl_environment.py:467-476 - Add more markers
hallucination_markers = [
    "example:", "for example,", "hypothetically",
    "let's say", "imagine", "suppose", "would be like",
    "simulated", "demonstration", "sample output",
    # NEW markers
    "might look like", "could be something like",
    "typical implementation", "usually looks like",
    "generic", "placeholder", "dummy",
]
```

### Quick Win 4: Verify line numbers exist (20 minutes)

```python
# repl_environment.py - Add to _create_llm_query_function
def llm_query(text: str, max_tokens: int = 4000) -> str:
    # ... existing code ...

    # NEW: Validate line references in response
    response_text = result
    line_refs = re.findall(r':(\d+)\b', response_text)
    for line_ref in line_refs:
        line_num = int(line_ref)
        if line_num > 10000:  # Suspiciously high line number
            response_text = response_text.replace(
                f":{line_ref}",
                f":{line_ref} [VERIFY]"
            )

    return response_text
```

---

## Part 5: Testing Strategy

### Unit Tests to Add

```python
# tests/test_result_verifier.py
def test_verifies_valid_line_references():
    verifier = ResultVerifier()
    response = "Found issue at src/auth.py:42"
    collection = create_mock_collection({"src/auth.py": "line1\n" * 50})

    verified, stats = verifier.verify_findings(response, collection)
    assert stats["verified"] == 1
    assert "[UNVERIFIED]" not in verified

def test_flags_invalid_line_references():
    verifier = ResultVerifier()
    response = "Found issue at src/auth.py:999"
    collection = create_mock_collection({"src/auth.py": "line1\n" * 10})

    verified, stats = verifier.verify_findings(response, collection)
    assert stats["invalid"] == 1
    assert "[UNVERIFIED - line out of range]" in verified
```

### Integration Tests

```python
# tests/test_rlm_accuracy.py
async def test_finds_documented_files():
    """Ensure RLM finds files mentioned in CLAUDE.md."""
    result = await rlm_analyze(
        query="Find PaywallView implementation",
        paths=["./mybill"]
    )

    assert "PaywallView.swift" in result.response
    assert ":line" in result.response  # Has line references

async def test_widget_extension_discovery():
    """Ensure widget extensions are found."""
    result = await rlm_analyze(
        query="Find all widget implementations",
        paths=["./mybill"]
    )

    assert "widgetExtension" in result.response.lower()
```

---

## Conclusion

The RLM tool has solid foundations but suffers from:
1. **Information bottlenecks** (30-file limit, no project awareness)
2. **Wrong caching strategy** (exact match vs semantic similarity)
3. **Missing verification** (LLM output trusted without validation)
4. **Poor progress visibility** (no streaming)

Implementing Priority 1 fixes alone should raise reliability from 6/10 to 8/10. Full implementation of all solutions targets 9/10.

The tool's core architecture (REPL-based RLM from arXiv:2512.24601) is sound - the issues are in the supporting infrastructure, not the fundamental approach.

---

## Part 6: Fallback Mechanism (Added)

### Issue: REPL Execution Failures with No Results

When the REPL code execution fails (syntax errors, sandbox restrictions), the tool previously returned nothing useful after 37+ seconds.

### Solution: `FallbackAnalyzer`

Created `python/src/rlm_mem_mcp/fallback_analyzer.py` with:

1. **Pattern-based analysis** - Uses regex to find common issues without LLM code execution
2. **Graceful degradation** - Always returns results, even if less comprehensive
3. **Query-aware patterns** - Selects relevant patterns based on query keywords

**Pattern Categories:**
- Security: hardcoded secrets, SQL injection, command injection, path traversal
- Swift/iOS: force unwraps, missing weak self, hardcoded URLs
- Quality: TODO/FIXME, debug code, empty catch blocks

**Integration:**
- Triggers after 3 consecutive code execution failures
- Triggers when max iterations reached without result
- Includes partial REPL output and recommendations

**Example Output:**
```markdown
## Fallback Analysis Results

*Note: RLM code execution failed (Code execution failed 3 times). Using pattern-based analysis.*

- Files analyzed: 163
- Patterns checked: 24
- Findings: 12

### Hardcoded Secrets (3 findings)
**src/config.swift:42** [HIGH]
```
let apiKey = "sk-..."
```
_Possible hardcoded API key_
```
