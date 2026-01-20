# RLM-Mem API Tools Reference

Quick reference for all available MCP tools and their parameters.

## Tool Summary

| Tool | Purpose | Speed | Use Case |
|------|---------|-------|----------|
| `rlm_analyze` | Comprehensive file analysis | 5-15s | Large codebases, security audits, architecture |
| `rlm_query_text` | Text processing | 2-5s | Logs, transcripts, documents |
| `rlm_memory_store` | Store findings | Instant | Persist important discoveries |
| `rlm_memory_recall` | Retrieve findings | Instant | Look up past analyses |
| `rlm_status` | Server health | <1s | Check configuration and stats |

---

## rlm_analyze - File/Directory Analysis

### Basic Usage
```python
{
  "query": "Find SQL injection vulnerabilities",
  "paths": ["./src", "./api"]
}
```

### Full Parameters
```javascript
{
  "query": "string (required)",              // What to search for
  "paths": ["string"] (required),            // Which files to analyze
  "scan_mode": "auto|security|ios|quality",  // Optimization mode
  "min_confidence": "HIGH|MEDIUM|LOW",       // Result filtering
  "include_quality": boolean                 // Include code quality checks
}
```

### Query Templates

#### Security Scan
```
"Find (1) SQL injection via string concat (2) hardcoded secrets
 (api_key, sk-, password) (3) eval/exec with user input.
 Report: file:line, code, severity."
```

#### Performance Issues
```
"Find (1) N+1 database queries (2) inefficient loops (3) O(n²) algorithms
 (4) blocking operations in async code. Report: file:line, issue, fix."
```

#### Architecture Mapping
```
"Map architecture: (1) main components (2) entry points (3) data flow
 (4) dependencies (5) design patterns. Format: tree with descriptions."
```

#### iOS/Swift Security
```
"Find (1) force unwraps (!) excluding != (2) closures missing [weak self]
 (3) @ObservedObject with default init. Report: file:line, code, fix."
```

### Response Structure
```javascript
{
  "findings": [
    {
      "file": "src/auth/handler.py",
      "line": 45,
      "code": "query = \"SELECT * FROM users WHERE id=\" + user_id",
      "issue": "SQL injection via string concatenation",
      "confidence": "HIGH",
      "severity": "CRITICAL",
      "fix": "Use parameterized query: cursor.execute('SELECT * FROM users WHERE id=%s', [user_id])"
    }
  ],
  "summary": "Found 3 critical SQL injection vulnerabilities",
  "files_scanned": 156,
  "errors": []
}
```

### Confidence Levels Explained

- **HIGH (80+)**: Multiple verified indicators, semantic confirmation
- **MEDIUM (50-79)**: Pattern matched with some uncertainty
- **LOW (20-49)**: Weak pattern, likely false positive
- **FILTERED**: Ruled out (test file, dead code, etc.)

---

## rlm_query_text - Text Processing

### Basic Usage
```python
{
  "query": "Extract all ERROR entries",
  "text": "[2024-01-19 10:23:45] ERROR [auth] Invalid token...\n..."
}
```

### Full Parameters
```javascript
{
  "query": "string (required)",      // What to extract/analyze
  "text": "string (required)"          // The content to process
}
```

### Query Templates

#### Log Analysis
```
"Extract (1) ERROR/WARN entries with timestamps (2) stack traces
 with root cause (3) error frequency by service.
 Format: timestamp | level | service | message | count"
```

#### Config Extraction
```
"Extract (1) environment variables (2) connection strings
 (3) feature flags (4) API endpoints.
 Format: key = value | location | usage"
```

#### Data Structure Analysis
```
"Extract (1) all unique field names (2) data types (3) nested depth
 (4) required vs optional fields. Format: field: type | presence | depth"
```

#### Meeting Transcript
```
"Extract (1) key decisions (2) action items with owners (3) blockers
 (4) unresolved questions. Format: bullet points with timestamps"
```

### Response Structure
```javascript
{
  "query": "original query",
  "results": {
    "extracted_data": {
      "error_count": 1247,
      "by_service": { "auth": 540, "api": 450, "db": 257 },
      "peak_time": "2024-01-19 10:30-11:00"
    },
    "summary": "Found 1247 errors over 6 hours...",
    "metrics": {
      "total_items": 1247,
      "categories": { "ERROR": 892, "WARN": 355 }
    }
  },
  "processing_time_ms": 1240
}
```

---

## rlm_memory_store - Persist Findings

### Basic Usage
```python
{
  "key": "security_audit_2024-01-19",
  "value": "Found 5 critical SQL injection vulnerabilities...",
  "tags": ["security", "critical", "sql-injection"]
}
```

### Full Parameters
```javascript
{
  "key": "string (required)",           // Unique identifier for this finding
  "value": "string (required)",         // The finding content
  "tags": ["string"] (optional)         // Searchable tags
}
```

### Best Practices
- Use descriptive keys: `"security_audit_phase1_2024-01"` ✓
- Avoid generic keys: `"findings"` ✗
- Use meaningful tags for easy retrieval
- Store summaries, not raw output
- Include dates in keys for tracking

### Example Storages
```python
# Security audit results
{
  "key": "sec_audit_2024_q1",
  "value": "Phase 1 findings: 15 vulnerabilities found...",
  "tags": ["security", "audit", "q1-2024", "critical"]
}

# Architecture decision
{
  "key": "architecture_microservices_decision",
  "value": "Decided to use microservices pattern for...",
  "tags": ["architecture", "decision", "approved"]
}

# Performance baseline
{
  "key": "perf_baseline_v1.0",
  "value": "Response time: 150ms avg, P99: 500ms...",
  "tags": ["performance", "baseline", "v1.0"]
}
```

---

## rlm_memory_recall - Retrieve Findings

### By Key
```python
{
  "key": "security_audit_2024-01-19"
}
```

### By Tags
```python
{
  "search_tags": ["security", "critical"]
}
```

### Response Structure
```javascript
{
  "key": "security_audit_2024-01-19",
  "value": "Found 5 critical vulnerabilities...",
  "tags": ["security", "critical"],
  "created_at": "2024-01-19T10:23:45Z",
  "accessed_count": 3,
  "last_accessed": "2024-01-19T14:15:30Z"
}
```

### Search Examples
```python
# Find all security-related findings
rlm_memory_recall({"search_tags": ["security"]})

# Find high-priority items
rlm_memory_recall({"search_tags": ["critical", "high-priority"]})

# Retrieve specific audit
rlm_memory_recall({"key": "perf_audit_2024_jan"})
```

---

## rlm_status - Server Health

### Usage
```python
{
  # No parameters required
}
```

### Response Structure
```javascript
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "config": {
    "model": "google/gemini-2.5-flash-lite",
    "aggregator_model": "google/gemini-2.5-flash-lite",
    "cache_enabled": true,
    "cache_ttl": "5m",
    "max_chunk_tokens": 8000,
    "max_result_tokens": 4000
  },
  "cache_stats": {
    "hits": 45,
    "misses": 12,
    "hit_rate": 0.79,
    "cached_prompts": 5,
    "avg_save_per_hit": "$0.03"
  },
  "memory": {
    "stored_items": 12,
    "total_size_bytes": 45320
  },
  "rate_limit": {
    "requests_remaining": 95,
    "tokens_remaining": 950000,
    "reset_at": "2024-01-19T10:30:00Z"
  }
}
```

### Interpreting Health Status
- **healthy**: All systems operational, cache hit rate good
- **degraded**: Some issues but still functional, rate limited or cache misses
- **unhealthy**: Service unavailable or circuit breaker open

---

## Scan Modes Comparison

| Mode | Best For | Speed | Accuracy |
|------|----------|-------|----------|
| `auto` | General analysis | Normal | Good |
| `security` | Vulnerability hunting | Fast | Very Good |
| `ios` | iOS/Swift apps | Normal | Excellent |
| `quality` | Code quality issues | Normal | Good |
| `all` | Comprehensive review | Slower | Very Good |

---

## Error Codes and Solutions

| Code | Meaning | Solution |
|------|---------|----------|
| 429 | Rate limited | Retry with exponential backoff or reduce tokens |
| 503 | Service unavailable | Wait 30-60 seconds, circuit breaker will recover |
| Invalid query | Malformed parameters | Check query syntax and parameters |
| File not found | Path doesn't exist | Verify path exists and is accessible |
| Timeout | Analysis took too long | Use faster model or narrow scope |

---

## Rate Limits

- **OpenRouter**: 100 requests/min, 1M tokens/min
- **Anthropic Direct**: 50 requests/min, 500k tokens/min
- **Circuit Breaker**: Stops after 3 consecutive failures, auto-recovers in 30s
- **Automatic Retry**: Exponential backoff with jitter (1s → 32s max)

---

## Performance Tips

1. **Be Specific**: Narrow queries = better results and faster processing
   - Good: "Find SQL injection in authentication"
   - Bad: "Find all problems"

2. **Use Scan Modes**: Pre-configured modes optimize performance
   - Security-focused: `security` mode
   - iOS specific: `ios` mode
   - Code quality: `quality` mode

3. **Filter by Confidence**: Reduce false positives
   - Critical findings only: `min_confidence: "HIGH"`
   - Exploratory: `min_confidence: "LOW"`

4. **Batch Related Queries**: Reuse cached analysis
   - Cache persists for 5 minutes by default
   - Use same paths/models for cache hits

5. **Reduce Scope**: Smaller paths = faster analysis
   - Good: `paths: ["src/api/"]`
   - Slow: `paths: ["."]` (entire codebase)

---

## Query Writing Best Practices

### ✅ Good Queries (Specific)
```
"Find SQL injection by (1) string concatenation in queries
 (2) unparameterized user input (3) dynamic SQL construction.
 Report: file:line, code, severity"
```

### ❌ Bad Queries (Vague)
```
"Find problems"
"Check security"
"Look for vulnerabilities"
```

### ✅ Well-Formatted Queries
- Use numbered lists for multiple criteria
- Specify output format clearly
- Include context about what to exclude
- Mention severity or importance levels

### Example Query Improvement
```
# Before
"Find bugs in the API endpoints"

# After
"Find (1) missing input validation (2) SQL injection via string concat
 (3) XSS via unsanitized output (4) missing authentication checks.
 Report: file:line, endpoint, vulnerability, severity, fix."
```

---

## Advanced Patterns

### Pattern: Iterative Security Audit
```
1. Initial scan: Find all vulnerabilities
   rlm_analyze({query: "security vulnerabilities", paths: ["."]})

2. Store results
   rlm_memory_store({key: "audit_phase1", value: "...", tags: ["security"]})

3. Review and fix issues
   [developer fixes code]

4. Targeted re-scan
   rlm_analyze({query: "same vulnerabilities", paths: ["modified_files"]})

5. Compare
   rlm_memory_recall({key: "audit_phase1"})
   # Compare phase1 results to current scan
```

### Pattern: Compliance Tracking
```
1. Run compliance checks
   rlm_analyze({query: "OWASP A01:2021 violations"})

2. Store baseline
   rlm_memory_store({key: "compliance_2024_baseline"})

3. Monthly checks
   [repeat analysis]
   [store results with date]

4. Track progress
   rlm_memory_recall({search_tags: ["compliance"]})
   # See all compliance records
```

### Pattern: Cost Optimization
```
1. Use caching
   export RLM_USE_CACHE=true

2. Choose fast model
   export RLM_MODEL=google/gemini-2-flash

3. Specific queries
   "Find X, not everything"

4. Narrow scope
   paths: ["src/"] not ["."]

5. Batch related queries
   # Run multiple related queries within 5m window
```

---

## See Also

- [Usage Guide](../guides/USAGE_GUIDE.md) - Practical examples
- [Configuration Guide](../guides/CONFIGURATION.md) - Setup and tuning
- [Architecture Guide](../guides/ARCHITECTURE.md) - How it works internally
- [CLAUDE.md](../../CLAUDE.md) - Main documentation
