# RLM-Mem MCP Performance Tuning Guide (v2.8)

**Version**: 2.8 | **Date**: 2026-01-21 | **Improvements**: Dynamic parallelism + async concurrency

## Overview

RLM v2.8 introduces automatic performance optimization through:

1. **Dynamic Thread Pool Sizing** (Phase 2)
   - Automatically uses optimal worker count based on CPU cores
   - 2-4x faster batch operations on multi-core systems

2. **Increased Async Concurrency** (Phase 3)
   - Semaphore limit: 5 → 10 concurrent API requests
   - 2x theoretical throughput for parallel LLM queries

## Performance Improvements

### Phase 2: Boost Parallelism

#### Dynamic Worker Calculation

**Formula**: `get_optimal_workers(max_limit=16, min_limit=4)`
```
optimal_workers = max(4, min(16, os.cpu_count()))
```

**Examples**:
| System | CPU Count | Workers | Speedup |
|--------|-----------|---------|---------|
| Laptop | 4-6 | 4 | 1x (baseline) |
| Desktop | 8 | 8 | 2x |
| Workstation | 16+ | 16 | 4x |

#### Affected Functions

1. **`parallel_scan(tools, functions, max_workers=None)`**
   - Runs multiple analysis functions in parallel
   - Old default: 4 workers (fixed)
   - New: Auto-detect from CPU count
   - Usage: Batch security/quality scans

2. **`parallel_rg_search(patterns, paths, max_workers=None)`**
   - Searches multiple patterns simultaneously
   - Old default: 4 workers (fixed)
   - New: Auto-detect from CPU count
   - Usage: Multi-pattern ripgrep searches

#### Configuration

To override auto-detection:

```python
# Use 8 workers explicitly
results = parallel_scan(tools, functions, max_workers=8)

# Use auto-detection (recommended)
results = parallel_scan(tools, functions)  # max_workers=None triggers auto-detect

# Use sequential (debugging)
results = parallel_scan(tools, functions, max_workers=1)
```

### Phase 3: Improve Async Performance

#### Increased Semaphore Concurrency

**Location**: `repl_environment.py` → `_async_batch_query()`

**Change**:
```python
# Old (v2.7 and earlier)
semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

# New (v2.8+)
semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests
```

**Impact**:
- Parallel LLM queries: ~50% latency reduction
- Batch query throughput: 2x improvement
- Example: 50 queries in ~5 rounds instead of ~10 rounds

#### Usage in REPL

```python
# Parallel LLM queries (now faster!)
results = llm_batch_query([
    "Analyze function A",
    "Analyze function B",
    "Analyze function C",
    # ... up to 50+ queries execute 10 at a time
])
```

## Benchmark Results

### Test Setup
- **Files analyzed**: 50-200
- **Patterns searched**: 4-8
- **Hardware**: Varies (auto-adapts)

### Expected Results

| Operation | v2.7 Baseline | v2.8 w/ Dynamic | Speedup |
|-----------|--------------|-----------------|---------|
| `parallel_scan` (8 functions) | ~800ms | ~200ms | 4x |
| `parallel_rg_search` (4 patterns) | ~300ms | ~100ms | 3x |
| `llm_batch_query` (50 queries) | ~120s | ~60s | 2x |
| **rlm_analyze overall** | **baseline** | **~3-5x faster** | **3-5x** |

### Running Benchmarks

```bash
cd python
python test_performance_v28.py
```

Output:
```
=== Phase 2: Dynamic Worker Sizing ===
System CPU count:      8
Optimal workers (4-16): 8
Expected range:        [4, 8]
✓ Dynamic worker sizing test PASSED

=== Phase 2: Parallel Scan Benchmark ===
parallel_scan (1 workers)         |    0.523s |     8 items |     15.30/s |  1 workers
parallel_scan (4 workers)         |    0.156s |     8 items |     51.30/s |  4 workers
parallel_scan (auto)              |    0.127s |     8 items |     63.00/s |  8 workers
✓ Dynamic workers achieved 4.12x speedup over sequential
```

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `RLM_MAX_WORKERS` | auto | Override optimal worker count (4-16) |
| `RLM_SEMAPHORE_LIMIT` | 10 | Max concurrent API requests |

### Examples

```bash
# Use 16 workers explicitly
export RLM_MAX_WORKERS=16
python -m rlm_mem_mcp.server

# Use 4 workers (for CI/testing)
export RLM_MAX_WORKERS=4
python -m rlm_mem_mcp.server

# Default (auto-detect)
python -m rlm_mem_mcp.server
```

## Tuning for Different Workloads

### Small Files (< 100KB)
- Use auto-detection (optimal)
- Minimal threading overhead needed

### Large Codebases (> 1MB)
- Benefits most from Phase 2 dynamic workers
- Expected speedup: 3-4x
- Recommendation: Allow auto-detection

### High-Concurrency Scenarios
- Batch LLM queries benefit from Phase 3
- Example: Analyzing 100+ files
- Expected reduction: 50% latency for batch operations

### Limited Resources (CI/Docker)
- Manually set `RLM_MAX_WORKERS=4`
- Prevents resource exhaustion
- Still benefits from improved semaphore (Phase 3)

## Monitoring Performance

### Logging

```python
from rlm_mem_mcp.structured_tools import get_optimal_workers
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

optimal = get_optimal_workers()
logger.info(f"Using {optimal} workers for parallel operations")
```

### Profiling

Profile `parallel_scan` and `parallel_rg_search`:

```python
import cProfile
import pstats
from io import StringIO

pr = cProfile.Profile()
pr.enable()

# Run analysis
results = parallel_scan(tools, [
    tools.find_secrets,
    tools.find_sql_injection,
    tools.find_xss,
])

pr.disable()
ps = pstats.Stats(pr, stream=StringIO())
ps.sort_stats('cumulative')
ps.print_stats(10)
```

## Troubleshooting

### Issue: Parallel operations slow or not using expected workers

**Diagnosis**:
```python
from rlm_mem_mcp.structured_tools import get_optimal_workers
print(f"Optimal workers: {get_optimal_workers()}")
```

**Solutions**:
1. Verify CPU count: `os.cpu_count()`
2. Check for resource limits: `ulimit -u`
3. Monitor system load: `top`, `htop`

### Issue: Too many threads causing resource exhaustion

**Solution**:
```bash
export RLM_MAX_WORKERS=4
```

### Issue: LLM API rate limiting

**Solution**: Reduce semaphore limit temporarily:

```python
# In repl_environment.py, temporarily change:
semaphore = asyncio.Semaphore(5)  # Instead of 10
```

Then revert after rate limit window passes.

## Future Optimizations (v2.9+)

### Planned Improvements

1. **Batch Size Optimization** (v2.9)
   - Adaptive batching based on pattern complexity
   - Estimated benefit: 15-25% throughput increase
   - Status: Requires profiling data

2. **Pure Async Refactoring** (v2.9)
   - Replace nested `asyncio.run()` with pure async
   - Eliminate ThreadPoolExecutor workaround
   - Estimated benefit: 20-30% latency reduction
   - Status: Complex refactoring, lower priority

3. **NUMA-Aware Scheduling** (v3.0)
   - Optimize for NUMA systems (>64 cores)
   - Thread affinity improvements
   - Estimated benefit: 5-10% on NUMA systems

## References

- **RLM Paper**: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)
- **Python Concurrency**: https://docs.python.org/3/library/concurrent.futures.html
- **Ripgrep**: https://github.com/BurntSushi/ripgrep

## Support

For performance issues or questions:

1. Run `test_performance_v28.py` to check system configuration
2. Check logs for worker count and API concurrency
3. Open issue with benchmark results and system specs

---

**Last Updated**: 2026-01-21
**v2.8 Release**: Focus on dynamic parallelism and async concurrency
**Next Review**: v2.9 planning (batch optimization, pure async refactoring)
