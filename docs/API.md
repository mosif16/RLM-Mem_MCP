# RLM-Mem MCP API Reference

Complete reference for all MCP tools available in the RLM-Mem server.

## Tool Overview

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `rlm_analyze` | Analyze files/directories with custom queries | paths, query, options | Structured findings |
| `rlm_query_text` | Process large text blocks directly | text, query | Analysis results |
| `rlm_memory_store` | Persist findings across conversations | key, value, tags | Confirmation |
| `rlm_memory_recall` | Retrieve stored findings | key or tags | Stored data |
| `rlm_status` | Check server health and stats | (none) | Health info |

---

## rlm_analyze

Analyze files or directories using custom search queries.

### Parameters

```javascript
{
  "query": "string (required)",        // Specific analysis query
  "paths": ["string"] (required),      // Files/directories to analyze
  "scan_mode": "string",               // 'auto', 'security', 'ios', 'quality'
  "min_confidence": "string",          // 'HIGH', 'MEDIUM', 'LOW'
  "include_quality": "boolean"         // Include code quality checks
}
```

### Query Patterns

**Security Analysis:**
```
"Find (1) SQL injection via string concat (2) hardcoded secrets
 matching sk-, api_key, password (3) eval/exec with user input.
 Report: file:line, code, severity."
```

**iOS/Swift Specific:**
```
"Find (1) force unwraps (!) excluding != (2) closures missing
 [weak self] (3) @ObservedObject with default value.
 Report: file:line, code, fix."
```

**Python Vulnerabilities:**
```
"Find (1) pickle.loads with untrusted data (2) bare except clauses
 (3) mutable default args. Report: file:line, code."
```

**JavaScript Issues:**
```
"Find (1) innerHTML XSS (2) missing await (3) useEffect missing deps.
 Report: file:line, code."
```

**Architecture Review:**
```
"Map all modules with purpose, entry points, dependencies,
 data flow."
```

### Response Structure

```javascript
{
  "findings": [
    {
      "file": "path/to/file.py",
      "line": 42,
      "code": "source code snippet",
      "issue": "description of issue",
      "confidence": "HIGH|MEDIUM|LOW|FILTERED",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
      "fix": "suggested fix (optional)",
      "category": "category tag"
    }
  ],
  "summary": "Brief summary of findings",
  "files_scanned": 123,
  "errors": []
}
```

### Examples

**Finding SQL Injection:**
```
rlm_analyze({
  "query": "Find SQL injection via string concat, parameterized
            queries not used, dynamic SQL construction",
  "paths": ["./api", "./db"],
  "min_confidence": "HIGH"
})
```

**Security Audit:**
```
rlm_analyze({
  "query": "Find (1) hardcoded API keys (sk-, api_key patterns)
            (2) credentials in config (3) secrets in comments.
            Report: file:line, secret pattern, severity",
  "paths": ["."],
  "scan_mode": "security"
})
```

**Architecture Mapping:**
```
rlm_analyze({
  "query": "Map architecture: (1) main components (2) entry points
            (3) data flow (4) key dependencies (5) design patterns.
            Format: tree structure with descriptions",
  "paths": ["./src", "./api"]
})
```

---

## rlm_query_text

Process large text blocks (logs, transcripts, documents) directly.

### Parameters

```javascript
{
  "query": "string (required)",        // What to extract/analyze
  "text": "string (required)"           // The text content to process
}
```

### Query Patterns

**Log Analysis:**
```
"Extract (1) ERROR/WARN entries with timestamps (2) stack traces
 with root cause (3) error frequency by type.
 Format: timestamp | level | message | count"
```

**Configuration Extraction:**
```
"Extract (1) all environment variables (2) connection strings
 (3) feature flags. Format: key = value with file location"
```

**Transcript Analysis:**
```
"Extract (1) key decisions made (2) action items with owners
 (3) unresolved questions. Format: bullet points with timestamps"
```

**JSON/Data Analysis:**
```
"Extract (1) all unique field names (2) data types per field
 (3) nested structure depth. Format: field: type (count)"
```

### Response Structure

```javascript
{
  "query": "original query",
  "results": {
    "extracted_data": {},
    "summary": "string",
    "metrics": {
      "total_items": 123,
      "categories": { "ERROR": 45, "WARN": 12 }
    }
  },
  "processing_time_ms": 234
}
```

### Examples

**Parse Application Logs:**
```
rlm_query_text({
  "query": "Extract all ERROR and CRITICAL entries with timestamps,
            service names, and error messages. Count by service.",
  "text": "[2024-01-19 10:23:45] ERROR [auth] Invalid token..."
})
```

**Extract Configuration:**
```
rlm_query_text({
  "query": "Extract all database connection strings, API endpoints,
            feature flags, and environment overrides.",
  "text": "DB_HOST=localhost DB_PORT=5432 FEATURE_NEW_UI=true..."
})
```

---

## rlm_memory_store

Persist important findings for later recall.

### Parameters

```javascript
{
  "key": "string (required)",           // Unique identifier
  "value": "string (required)",         // Content to store
  "tags": ["string"] (optional)         // Categorization tags
}
```

### Examples

**Store Security Findings:**
```
rlm_memory_store({
  "key": "security_audit_2024-01-19",
  "value": "Found 3 SQL injection vulnerabilities in user_service.py...",
  "tags": ["security", "sql-injection", "high-priority"]
})
```

**Store Architecture Decision:**
```
rlm_memory_store({
  "key": "architecture_service_layers",
  "value": "Identified 4 main layers: API, Service, Data, Persistence",
  "tags": ["architecture", "design"]
})
```

---

## rlm_memory_recall

Retrieve stored findings.

### Parameters (Option 1: By Key)

```javascript
{
  "key": "security_audit_2024-01-19"
}
```

### Parameters (Option 2: By Tags)

```javascript
{
  "search_tags": ["security", "high-priority"]
}
```

### Response Structure

```javascript
{
  "key": "security_audit_2024-01-19",
  "value": "content",
  "tags": ["security", "sql-injection"],
  "created_at": "2024-01-19T10:23:45Z",
  "accessed_count": 3
}
```

---

## rlm_status

Check server health, cache stats, and configuration.

### Parameters

(None required)

### Response Structure

```javascript
{
  "status": "healthy|degraded|unhealthy",
  "uptime_seconds": 3600,
  "config": {
    "model": "google/gemini-2.5-flash-lite",
    "cache_enabled": true,
    "cache_ttl": "5m",
    "max_chunk_tokens": 8000
  },
  "cache_stats": {
    "hits": 45,
    "misses": 12,
    "hit_rate": 0.79,
    "cached_prompts": 5
  },
  "memory": {
    "stored_items": 12,
    "total_size_bytes": 45320
  },
  "rate_limit": {
    "requests_remaining": 95,
    "reset_at": "2024-01-19T10:30:00Z"
  }
}
```

---

## Confidence Levels

The system uses a 4-level confidence scale:

- **HIGH** (80+): Multiple indicators verified, semantic confirmation
- **MEDIUM** (50-79): Pattern matched, some uncertainty
- **LOW** (20-49): Weak pattern match, likely false positive
- **FILTERED**: Ruled out (dead code, test file, etc.)

### Confidence Scoring

Findings start at 100 points and are adjusted:

**Deductions:**
- In dead code: -50
- In test file: -30
- Pattern match only: -10
- In comment: -40
- Can't verify line: -20

**Boosts:**
- Semantic verification: +20
- Multiple indicators: +15

---

## Error Handling

### Common Error Codes

| Code | Meaning | Action |
|------|---------|--------|
| 429 | Rate limited | Retry with exponential backoff |
| 503 | Service unavailable | Circuit breaker activated |
| Invalid query | Malformed parameters | Check query syntax |
| File not found | Path doesn't exist | Verify path is correct |

### Retry Logic

The server automatically retries with exponential backoff:
- Initial: 1 second
- Max: 32 seconds
- Max attempts: 5

---

## Performance Tips

1. **Be Specific**: Narrow queries = better results and faster processing
   - Good: "Find SQL injection in authentication module"
   - Bad: "Find all problems"

2. **Use Scan Modes**: Pre-configured modes optimize performance
   - `security`: Security-focused scanning
   - `ios`: iOS/Swift specific checks
   - `quality`: Code quality issues

3. **Filter by Confidence**: Reduce false positives
   - `min_confidence: "HIGH"` for critical findings only

4. **Batch Related Queries**: Reuse cached analysis
   - Cache persists for 5 minutes by default

---

## Rate Limits

- **OpenRouter**: 100 requests/min, 1M tokens/min
- **Anthropic Direct**: 50 requests/min, 500k tokens/min
- **Circuit Breaker**: Stops after 3 consecutive failures, auto-recovers

---

## Authentication

Tools require no explicit authentication—the MCP server handles API keys via environment variables:

- `OPENROUTER_API_KEY`: For OpenRouter access
- `ANTHROPIC_API_KEY`: For Anthropic direct access (fallback)

See [Configuration Guide](./CONFIGURATION.md) for setup details.
