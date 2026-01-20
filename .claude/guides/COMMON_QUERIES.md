# Common RLM Queries - Copy & Paste Examples

Ready-to-use query templates for common analysis tasks.

## Security Analysis

### SQL Injection Detection
```python
{
  "query": "Find SQL injection by (1) string concatenation in queries
            (2) unparameterized user input to database (3) dynamic SQL
            construction. Report: file:line, code, vulnerability, severity, fix.",
  "paths": ["./src", "./api"],
  "scan_mode": "security",
  "min_confidence": "HIGH"
}
```

### Hardcoded Secrets
```python
{
  "query": "Find hardcoded secrets: (1) API keys (sk-, api_key patterns)
            (2) passwords in strings (3) AWS keys (AKIA, aws_secret)
            (4) database credentials. Report: file:line, secret_type, risk.",
  "paths": ["."],
  "scan_mode": "security",
  "min_confidence": "HIGH"
}
```

### XSS Vulnerabilities (JavaScript/HTML)
```python
{
  "query": "Find XSS vulnerabilities: (1) innerHTML with user input
            (2) eval/Function with user input (3) unsanitized DOM manipulation
            (4) missing escaping. Report: file:line, code, input_source, fix.",
  "paths": ["./src", "./web"],
  "scan_mode": "security",
  "min_confidence": "MEDIUM"
}
```

### Authentication & Authorization
```python
{
  "query": "Find auth issues: (1) missing authentication checks
            (2) hardcoded credentials (3) weak password validation
            (4) missing authorization checks (5) token bypass vulnerabilities.
            Report: file:line, issue, severity.",
  "paths": ["./auth", "./api"],
  "scan_mode": "security",
  "min_confidence": "HIGH"
}
```

### Insecure Deserialization (Python)
```python
{
  "query": "Find unsafe deserialization: (1) pickle.loads with untrusted data
            (2) yaml.load without safe_load (3) eval/exec with user input
            (4) json.loads with custom objects. Report: file:line, code, fix.",
  "paths": ["."],
  "scan_mode": "security",
  "min_confidence": "HIGH"
}
```

### Cryptography Issues
```python
{
  "query": "Find crypto issues: (1) weak algorithms (MD5, SHA1)
            (2) hardcoded encryption keys (3) random.random for security
            (4) missing HTTPS validation. Report: file:line, issue, severity.",
  "paths": ["."],
  "scan_mode": "security",
  "min_confidence": "MEDIUM"
}
```

### Full Security Audit
```python
{
  "query": "Find (1) SQL injection via string concat (2) XSS via innerHTML
            (3) hardcoded secrets (sk-, api_key, password) (4) eval/exec with
            user input (5) missing auth checks (6) insecure deserialization
            (7) weak crypto. Report: file:line, code, severity, fix.",
  "paths": ["."],
  "scan_mode": "security",
  "min_confidence": "HIGH"
}
```

---

## Performance Analysis

### Database Query Issues
```python
{
  "query": "Find database issues: (1) N+1 queries in loops (2) missing
            indexes (3) unoptimized joins (4) large result sets without
            pagination (5) blocking queries in async code.
            Report: file:line, issue, optimization.",
  "paths": ["./api", "./services"],
  "min_confidence": "MEDIUM",
  "include_quality": True
}
```

### Memory Leaks & Inefficiency
```python
{
  "query": "Find memory issues: (1) unbounded caches (2) accumulating lists
            (3) circular references (4) large objects in loops (5) missing
            cleanup in finally blocks. Report: file:line, issue, fix.",
  "paths": ["."],
  "min_confidence": "MEDIUM",
  "include_quality": True
}
```

### Slow Algorithms
```python
{
  "query": "Find slow algorithms: (1) O(n²) or worse complexity (2) nested
            loops with expensive operations (3) repeated function calls in
            loops (4) inefficient sorting (5) brute force searches.
            Report: file:line, algorithm, complexity, fix.",
  "paths": ["./algorithms", "./core"],
  "min_confidence": "MEDIUM",
  "include_quality": True
}
```

### Blocking Operations in Async Code
```python
{
  "query": "Find blocking operations in async: (1) synchronous file I/O
            (2) blocking database calls (3) requests.get instead of aiohttp
            (4) sleep instead of asyncio.sleep (5) CPU-bound in async context.
            Report: file:line, code, fix.",
  "paths": ["."],
  "min_confidence": "HIGH"
}
```

---

## Code Quality

### Long/Complex Functions
```python
{
  "query": "Find quality issues: (1) functions >100 lines (2) cyclomatic
            complexity >10 (3) deeply nested code (>4 levels) (4) parameter
            lists >5 args (5) code duplication.
            Report: file:line, metric, refactoring_suggestion.",
  "paths": ["./src"],
  "include_quality": True,
  "min_confidence": "MEDIUM"
}
```

### Dead Code
```python
{
  "query": "Find dead code: (1) unused variables (2) unreachable code
            (3) unused imports (4) unused functions (5) deprecated code
            without removal. Report: file:line, code_type, action.",
  "paths": ["."],
  "include_quality": True,
  "min_confidence": "HIGH"
}
```

### Error Handling
```python
{
  "query": "Find error handling issues: (1) bare except clauses (2) caught
            but not handled exceptions (3) missing try-catch (4) swallowed
            exceptions (5) error messages to users. Report: file:line, issue.",
  "paths": ["."],
  "include_quality": True,
  "min_confidence": "MEDIUM"
}
```

### Type Issues (Python)
```python
{
  "query": "Find type issues: (1) missing type hints (2) inconsistent types
            (3) None handling (4) mutable default arguments (5) duck typing
            without documentation. Report: file:line, issue, type_hint.",
  "paths": ["./src"],
  "include_quality": True,
  "min_confidence": "MEDIUM"
}
```

---

## Architecture & Design

### Component Mapping
```python
{
  "query": "Map architecture: (1) main components and modules (2) entry
            points (main.py, __main__.py, etc.) (3) data flow through system
            (4) key dependencies (5) design patterns used (Factory, Strategy,
            etc.). Format as tree structure with descriptions.",
  "paths": ["."],
  "min_confidence": "MEDIUM"
}
```

### Data Flow Analysis
```python
{
  "query": "Trace data flow: (1) user input entry points (2) validation
            points (3) processing pipeline (4) storage/database (5) output.
            For each flow show file:line and transformation. Format: input →
            process → output.",
  "paths": ["./src"],
  "min_confidence": "MEDIUM"
}
```

### Dependency Analysis
```python
{
  "query": "Analyze dependencies: (1) external libraries used (2) circular
            dependencies (3) unused imports (4) version conflicts (5) heavy
            dependencies. Report: library, usage, alternatives.",
  "paths": ["requirements.txt", "package.json", "pyproject.toml", "./src"],
  "min_confidence": "MEDIUM"
}
```

### Design Pattern Usage
```python
{
  "query": "Find design patterns: (1) Factory pattern (2) Singleton pattern
            (3) Observer pattern (4) Strategy pattern (5) Dependency Injection.
            For each pattern show location, purpose, and implementation.",
  "paths": ["./src"],
  "min_confidence": "MEDIUM"
}
```

---

## Logging & Monitoring

### Logging Issues
```python
{
  "query": "Find logging issues: (1) sensitive data in logs (passwords,
            tokens, PII) (2) missing log levels (3) too verbose or too silent
            (4) unstructured logging (5) no error logging in catch blocks.
            Report: file:line, issue.",
  "paths": ["."],
  "min_confidence": "MEDIUM"
}
```

### Error Messages
```python
{
  "query": "Analyze error messages: (1) error messages to end users
            (2) stack traces exposed (3) generic vs helpful messages (4) error
            codes and meanings (5) internationalization needs.
            Report: file:line, message, user_friendly?",
  "paths": ["./api", "./services"],
  "min_confidence": "MEDIUM"
}
```

---

## iOS/Swift Specific

### Swift Safety Issues
```python
{
  "query": "Find Swift safety issues: (1) force unwraps (!) excluding !=
            (2) closures missing [weak self] (3) @ObservedObject with default
            initialization (4) unhandled optionals (5) retain cycles.
            Report: file:line, code, risk, fix.",
  "paths": ["./ios", "./swift"],
  "scan_mode": "ios",
  "min_confidence": "HIGH"
}
```

### Memory Management
```python
{
  "query": "Find memory issues: (1) missing weak references in closures
            (2) unowned without null check (3) temporary strong references
            (4) delegate patterns missing weak (5) circular reference potential.
            Report: file:line, code, risk.",
  "paths": ["./ios"],
  "scan_mode": "ios",
  "min_confidence": "MEDIUM"
}
```

---

## Text Analysis

### Log File Analysis
```javascript
{
  "query": "Extract (1) ERROR and CRITICAL entries with timestamps
            (2) stack traces with root cause (3) error frequency by type/service
            (4) affected services (5) time periods with highest errors.
            Format: timestamp | level | service | message | count",
  "text": "[paste log content here]"
}
```

### Configuration Extraction
```javascript
{
  "query": "Extract (1) all environment variables (2) connection strings
            (3) API endpoints (4) feature flags (5) timeouts and limits.
            Format: key = value | location | description",
  "text": "[paste config content here]"
}
```

### Meeting Transcript Analysis
```javascript
{
  "query": "Extract (1) key decisions made (2) action items with owners
            (3) blockers and risks (4) unresolved questions (5) next steps.
            Format: decision/action | owner | deadline | status",
  "text": "[paste transcript here]"
}
```

### Data Structure Analysis
```javascript
{
  "query": "Analyze JSON/data structure: (1) all unique field names
            (2) data types per field (3) nested structure depth (4) required
            vs optional fields (5) validation rules needed.
            Format: field | type | nesting | presence | constraints",
  "text": "[paste sample JSON/data here]"
}
```

---

## Compliance & Standards

### OWASP Compliance
```python
{
  "query": "Check OWASP Top 10 compliance: (1) A01 - Broken Access Control
            (2) A02 - Cryptographic Failures (3) A03 - Injection (4) A04 -
            Insecure Design (5) A05 - Security Misconfiguration.
            Report: finding | severity | evidence | fix.",
  "paths": ["."],
  "scan_mode": "security",
  "min_confidence": "HIGH"
}
```

### PCI-DSS Compliance
```python
{
  "query": "Check PCI-DSS compliance: (1) secure coding practices
            (2) access control and authentication (3) encryption of data
            (4) network security (5) testing and monitoring.
            Report: requirement | finding | risk | remediation.",
  "paths": ["."],
  "scan_mode": "security",
  "min_confidence": "HIGH"
}
```

### GDPR Compliance
```python
{
  "query": "Check GDPR compliance: (1) PII handling and protection
            (2) data encryption (3) user consent collection (4) data retention
            (5) export/deletion capabilities. Report: issue | affected data.",
  "paths": ["."],
  "min_confidence": "MEDIUM"
}
```

---

## Maintenance & Upgrades

### Outdated Dependencies
```python
{
  "query": "Find outdated dependencies: (1) version numbers in requirements
            (2) CVE vulnerabilities in current versions (3) deprecated libraries
            (4) major version upgrades available (5) breaking changes.
            Report: package | current_version | latest | risk | action.",
  "paths": ["requirements.txt", "package.json", "Gemfile", "go.mod"],
  "min_confidence": "HIGH"
}
```

### Python 2 to 3 Migration Issues
```python
{
  "query": "Find Python 2 legacy code: (1) print statements (2) xrange
            (3) unicode/str confusion (4) dict.iteritems() (5) old-style classes
            (6) basestring. Report: file:line, code, python3_equivalent.",
  "paths": ["."],
  "min_confidence": "HIGH"
}
```

---

## Tips for Using These Queries

1. **Copy the entire block** including parameters
2. **Adjust `paths`** to your specific directories
3. **Adjust `min_confidence`** based on your needs (HIGH for critical, MEDIUM for general)
4. **Modify query text** for your specific needs
5. **Store results** using `rlm_memory_store` for tracking

---

## Query Customization

### Reducing False Positives
- Add `min_confidence: "HIGH"` to filter to high-confidence findings
- Be more specific in your query description
- Exclude test directories: `paths: ["./src", "./api"]` instead of `["."]`

### Increasing Coverage
- Add `min_confidence: "LOW"` to catch more potential issues
- Use broader paths: `paths: ["."]` instead of `["./src"]`
- Add `include_quality: True` for code quality metrics

### Faster Results
- Use `min_confidence: "HIGH"` to skip low-confidence items
- Reduce scope: narrow `paths` list
- Use specific `scan_mode` instead of `auto`

---

## See Also

- [API Tools Reference](./../reference/API_TOOLS.md)
- [CLAUDE.md](../../CLAUDE.md) - Main documentation
