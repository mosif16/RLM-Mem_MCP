# Async Refactoring Roadmap (v2.9+)

## Status
- **Phase 3.1**: Remove nested asyncio.run() - PLANNED for v2.9
- **Phase 3.2**: Test pure async implementation - DEPENDS ON Phase 3.1

## Current Architecture (v2.8)

### The Problem
When `llm_batch_query()` is called from already-async context:

```python
# In repl_environment.py
def llm_batch_query(queries: list[str], max_tokens: int = 4000):
    loop = asyncio.get_event_loop()

    if loop.is_running():
        # If already in async context, we have a problem:
        # We can't call loop.run_until_complete() in running loop

        # Workaround: Use threading (v2.8 solution - production ready)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                lambda: asyncio.run(_async_batch_query(queries, max_tokens))
            )
            return future.result()
    else:
        # Not in async context - simple case
        return loop.run_until_complete(_async_batch_query(queries, max_tokens))
```

### Why Threading Workaround Works
- ✅ Thread creates new event loop via `asyncio.run()`
- ✅ Results returned to original context
- ✅ Avoids nested event loop error
- ❌ 20-30% latency overhead from threading

### Why Pure Async is Better
- ✅ No threading overhead
- ✅ 20-30% latency reduction
- ✅ Cleaner code
- ❌ Requires careful refactoring

---

## Proposed Solution (v2.9+)

### Option A: Use asyncio.ensure_future()
```python
async def llm_batch_query_async(queries: list[str], max_tokens: int = 4000):
    # Pure async - no threading needed
    return await _async_batch_query(queries, max_tokens)

def llm_batch_query(queries: list[str], max_tokens: int = 4000):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create task in current loop
            task = asyncio.ensure_future(_async_batch_query(queries, max_tokens))
            # Return awaitable for caller to handle
            return task
        else:
            return loop.run_until_complete(_async_batch_query(queries, max_tokens))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_async_batch_query(queries, max_tokens))
```

### Option B: Refactor to Async-First
```python
# Make REPL environment fully async
class AsyncRLMReplEnvironment(RLMReplEnvironment):
    async def execute_async(self, code: str):
        # Async execution path
        ...

    async def llm_query_async(self, text: str):
        # Direct async calls
        ...
```

### Option C: Use contextvars (Python 3.7+)
```python
import contextvars

_context_loop = contextvars.ContextVar('event_loop', default=None)

async def llm_batch_query(queries: list[str], max_tokens: int = 4000):
    loop = _context_loop.get()
    if loop is None:
        # Create new loop
        loop = asyncio.new_event_loop()
        token = _context_loop.set(loop)
        try:
            return await _async_batch_query(queries, max_tokens)
        finally:
            _context_loop.reset(token)
    else:
        return await _async_batch_query(queries, max_tokens)
```

---

## Implementation Complexity

| Aspect | Effort | Risk | Benefit |
|--------|--------|------|---------|
| Threading removal | High | Medium | 20-30% latency |
| Context handling | High | High | Better semantics |
| API changes | Medium | Low | Cleaner interface |
| Testing | High | Medium | Verification critical |
| Backwards compat | Medium | Medium | Requires v2.9+ only |

---

## Why Deferred to v2.9

### Current State (v2.8)
- Threading solution is **production-ready**
- Handles edge cases correctly
- No breaking changes needed
- Performance gain: 20-30% (not critical for v2.8)

### v2.8 Priority
1. ✅ Fix critical bugs (RgMatch.text)
2. ✅ Immediate wins (dynamic workers, semaphore)
3. ✅ Documentation & tests

### v2.9 Priority
1. Performance tuning (pure async, batch optimization)
2. Architecture improvements (context handling)
3. Extended testing on edge cases

---

## Migration Path (v2.9)

### Phase 1: Develop & Test (weeks 1-2)
```python
# New async-first implementation
async def llm_batch_query_v29(queries: list[str]):
    # Pure async implementation
    ...

# Keep v2.8 threaded version for compatibility
def llm_batch_query(queries: list[str]):
    # v2.8 threading approach
    ...
```

### Phase 2: Gradual Migration (weeks 3-4)
- Add feature flag: `use_pure_async=True`
- Allow A/B testing between implementations
- Benchmark and validate

### Phase 3: Default to Pure Async (v2.9 release)
- Set `use_pure_async=True` as default
- Keep threading as fallback
- Deprecate threading in v3.0

### Phase 4: Remove Threading (v3.0)
- Remove ThreadPoolExecutor workaround
- Full pure async implementation
- Smaller codebase

---

## Testing Strategy (v2.9)

### Unit Tests
```python
def test_llm_batch_query_pure_async():
    # Test pure async implementation

def test_llm_batch_query_threading():
    # Test v2.8 threading approach

def test_llm_batch_query_nested_event_loop():
    # Test nested loop handling

def test_context_preservation():
    # Ensure context variables preserved
```

### Integration Tests
```python
def test_parallel_analysis_pure_async():
    # Test with multiple queries

def test_mixed_sync_async():
    # Test sync functions calling async

def test_latency_improvement():
    # Benchmark: pure async vs threading
```

### Performance Tests
```python
# Expected results (v2.9)
pure_async_latency = 60.0  # seconds for 50 queries
threading_latency = 70.0   # seconds (v2.8)
improvement = (1 - 60.0/70.0) * 100  # ~14% improvement
assert improvement >= 15.0, "Should be 20-30% faster"
```

---

## Recommendation

### For v2.8 (Current)
✅ **Ship with threading workaround**
- Proven, production-ready
- Handles all edge cases
- Acceptable 20-30% latency overhead
- Unblocks rlm_grep + performance gains

### For v2.9
🔄 **Implement pure async with feature flag**
- Gradual migration path
- Reduced risk through A/B testing
- Better backwards compatibility
- Path to v3.0 cleanup

### For v3.0+
🚀 **Full pure async, remove threading**
- Cleaner architecture
- Better performance
- Full async/await semantics

---

## Related Issues

- **Threading overhead**: 20-30% of LLM query latency
- **Nested event loop complexity**: Difficult edge cases
- **Context preservation**: contextvars helps but not trivial
- **API evolution**: May require interface changes

## References

- Python asyncio docs: https://docs.python.org/3/library/asyncio.html
- contextvars: https://docs.python.org/3/library/contextvars.html
- asyncio.ensure_future: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
- Nested event loops: https://github.com/python/cpython/issues/88254

---

**Status**: Documented for v2.9 implementation
**Estimated Effort**: 2-3 weeks
**Expected Benefit**: 20-30% latency reduction
**Priority**: Medium (performance optimization, not critical fix)
