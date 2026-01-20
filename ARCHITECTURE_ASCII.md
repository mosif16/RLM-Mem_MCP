# RLM-Mem MCP Server - ASCII Architecture

## Complete System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                   CLAUDE CODE (Client)                               │
│                                                                                      │
│   User Query: "Find all security vulnerabilities in this codebase"                  │
└─────────────────────────────────────────┬───────────────────────────────────────────┘
                                          │
                                          │ MCP Protocol (JSON-RPC over stdio)
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              RLM-MEM MCP SERVER                                      │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                            server.py (Entry Point)                            │  │
│  │                                                                               │  │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │  │
│  │   │ rlm_analyze │  │rlm_query_  │  │rlm_memory_ │  │ rlm_status  │         │  │
│  │   │             │  │   text      │  │store/recall│  │             │         │  │
│  │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────────────┘         │  │
│  └──────────┼────────────────┼────────────────┼─────────────────────────────────┘  │
│             │                │                │                                     │
│             ▼                ▼                ▼                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                         SINGLETON INITIALIZATION                              │  │
│  │                                                                               │  │
│  │   RLMConfig ──► CacheManager ──► FileCollector ──► RLMProcessor              │  │
│  │       │                                                   │                   │  │
│  │       └──► SemanticCache ──► ResultVerifier ──► ProjectAnalyzer              │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                          │
│                                          ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                           CORE PROCESSING ENGINE                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                      RLMProcessor (rlm_processor.py)                    │  │  │
│  │  │                                                                         │  │  │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │  │  │
│  │  │  │CircuitBreaker│  │ RateLimiter  │  │TrajectoryLog │                  │  │  │
│  │  │  │  (Fault Tol) │  │(Token Bucket)│  │  (Debugging) │                  │  │  │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘                  │  │  │
│  │  │                                                                         │  │  │
│  │  │  Pipeline: Query ─► Enhance ─► Chunk ─► Filter ─► Execute ─► Aggregate │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                          │
│                                          ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                     TRUE RLM SANDBOX (repl_environment.py)                    │  │
│  │                                                                               │  │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │  │
│  │   │  GLOBALS AVAILABLE IN SANDBOX                                       │    │  │
│  │   │                                                                     │    │  │
│  │   │   prompt      = "...file contents..."     (THE KEY INSIGHT!)       │    │  │
│  │   │   context     = {metadata, file_info}                               │    │  │
│  │   │   results     = []  # accumulator                                   │    │  │
│  │   │   llm_query() = func(portion) -> sub_result                        │    │  │
│  │   └─────────────────────────────────────────────────────────────────────┘    │  │
│  │                                                                               │  │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │  │
│  │   │  STRUCTURED TOOLS (structured_tools.py)                             │    │  │
│  │   │                                                                     │    │  │
│  │   │  SECURITY          iOS/SWIFT           QUALITY         ARCHITECTURE│    │  │
│  │   │  ─────────         ─────────           ───────         ────────────│    │  │
│  │   │  find_secrets()    find_force_unwraps  find_long_funcs map_arch()  │    │  │
│  │   │  find_sql_inj()    find_retain_cycles  find_dead_code  call_graph()│    │  │
│  │   │  find_xss()        find_async_issues   find_todos()    find_imports│    │  │
│  │   │  find_cmd_inj()    find_weak_self      find_complex()              │    │  │
│  │   └─────────────────────────────────────────────────────────────────────┘    │  │
│  │                                                                               │  │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │  │
│  │   │  SECURITY CONTROLS                                                  │    │  │
│  │   │                                                                     │    │  │
│  │   │  CodeValidator (AST)  │  30s Timeout  │  Memory Limits             │    │  │
│  │   │  ✗ No file I/O        │  ✗ No network │  ✗ No subprocess           │    │  │
│  │   └─────────────────────────────────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                          │
│                                          ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                      SUPPORT LAYER                                            │  │
│  │                                                                               │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │  │
│  │  │FileCollector │  │CacheManager  │  │ResultVerifier│  │ MemoryStore  │      │  │
│  │  │              │  │              │  │              │  │              │      │  │
│  │  │ Async I/O    │  │ Prompt Cache │  │ Confidence   │  │ SQLite DB    │      │  │
│  │  │ 50 parallel  │  │ 90% savings  │  │ Scoring L11  │  │ Persistence  │      │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │  │
│  └─────────┼─────────────────┼─────────────────┼─────────────────┼──────────────┘  │
└────────────┼─────────────────┼─────────────────┼─────────────────┼──────────────────┘
             │                 │                 │                 │
             ▼                 ▼                 ▼                 ▼
      ┌──────────┐      ┌──────────────┐  ┌──────────┐      ┌──────────┐
      │FILESYSTEM│      │OPENROUTER/   │  │ RESULT   │      │ SQLITE   │
      │          │      │ANTHROPIC API │  │ OUTPUT   │      │ DATABASE │
      └──────────┘      └──────────────┘  └──────────┘      └──────────┘
```

## TRUE RLM Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              TRUE RLM TECHNIQUE (arXiv:2512.24601)                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   TRADITIONAL SUMMARIZATION (NOT what we do):                                       │
│   ══════════════════════════════════════════                                        │
│                                                                                      │
│       Large Content ──► LLM Summarizes ──► Information LOST ──► Poor Results        │
│           500k              context              ✗                   ✗               │
│          tokens            overflow                                                  │
│                                                                                      │
│   ─────────────────────────────────────────────────────────────────────────────────  │
│                                                                                      │
│   TRUE RLM TECHNIQUE (what we do):                                                  │
│   ════════════════════════════════                                                  │
│                                                                                      │
│       ┌──────────┐     ┌──────────────────┐     ┌──────────────────┐               │
│       │  Large   │     │  Store as        │     │  LLM writes      │               │
│       │ Content  │ ──► │  VARIABLE        │ ──► │  Python CODE     │               │
│       │  500k    │     │  prompt = "..."  │     │  to examine it   │               │
│       └──────────┘     └──────────────────┘     └────────┬─────────┘               │
│                                                          │                          │
│                                                          ▼                          │
│       ┌──────────┐     ┌──────────────────┐     ┌──────────────────┐               │
│       │  ~4k     │     │  Full data       │     │  Execute code    │               │
│       │ Summary  │ ◄── │  PRESERVED       │ ◄── │  in sandbox      │               │
│       │  Output  │     │  (accessible)    │     │  with llm_query  │               │
│       └──────────┘     └──────────────────┘     └──────────────────┘               │
│                                                                                      │
│   KEY INSIGHT: Content never enters LLM context. LLM acts as PROGRAMMER.            │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Request Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              REQUEST PROCESSING PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────────────┘

   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
   │   1.    │    │   2.    │    │   3.    │    │   4.    │    │   5.    │
   │ COLLECT │───►│ ENHANCE │───►│  CHUNK  │───►│ EXECUTE │───►│AGGREGATE│
   │         │    │         │    │         │    │         │    │         │
   └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
        │              │              │              │              │
        ▼              ▼              ▼              ▼              ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
   │FileCol- │    │Add scan │    │Function-│    │REPL env │    │Combine  │
   │lector   │    │mode,    │    │aware    │    │executes │    │results, │
   │async    │    │tech     │    │splitting│    │LLM code │    │dedup,   │
   │reads    │    │context  │    │8k chunks│    │llm_query│    │verify   │
   └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘


   DETAILED FLOW:
   ═════════════

   Step 1: COLLECT
   ┌────────────────────────────────────────────────────────────────────┐
   │  paths: ["./src"]                                                  │
   │       │                                                            │
   │       ▼                                                            │
   │  FileCollector.collect_paths_async()                               │
   │       │                                                            │
   │       ├── Filter by extension (.py, .js, .swift, etc.)            │
   │       ├── Skip directories (node_modules, .git, DerivedData)      │
   │       └── Parallel I/O (50 concurrent reads)                       │
   │       │                                                            │
   │       ▼                                                            │
   │  CollectionResult(files=54, tokens=223k)                           │
   └────────────────────────────────────────────────────────────────────┘

   Step 2: ENHANCE
   ┌────────────────────────────────────────────────────────────────────┐
   │  query: "Find security issues"                                     │
   │       │                                                            │
   │       ▼                                                            │
   │  ProjectAnalyzer.detect_tech_stack()                               │
   │       │                                                            │
   │       ├── Detect: Python, Swift, TypeScript                       │
   │       ├── Add scan_mode context                                    │
   │       └── Build specialized prompt                                 │
   │       │                                                            │
   │       ▼                                                            │
   │  Enhanced: "Find security issues in Python/Swift: SQL injection,  │
   │            force unwraps, XSS, using find_* tools"                │
   └────────────────────────────────────────────────────────────────────┘

   Step 3: CHUNK
   ┌────────────────────────────────────────────────────────────────────┐
   │  223k tokens                                                       │
   │       │                                                            │
   │       ▼                                                            │
   │  function_aware_chunking()                                         │
   │       │                                                            │
   │       ├── Respect function boundaries                              │
   │       ├── Max 8k tokens per chunk                                  │
   │       ├── 200 token overlap                                        │
   │       └── Smart filtering (skip low-relevance)                     │
   │       │                                                            │
   │       ▼                                                            │
   │  39 chunks ready for processing                                    │
   └────────────────────────────────────────────────────────────────────┘

   Step 4: EXECUTE (per chunk)
   ┌────────────────────────────────────────────────────────────────────┐
   │  chunk_content                                                     │
   │       │                                                            │
   │       ▼                                                            │
   │  REPLEnvironment.initialize(chunk_content)                         │
   │       │                                                            │
   │       │    prompt = chunk_content  ◄── THE KEY!                   │
   │       │                                                            │
   │       ▼                                                            │
   │  LLM generates Python code:                                        │
   │  ┌──────────────────────────────────────────────────────────┐     │
   │  │  findings = find_sql_injection(prompt)                   │     │
   │  │  findings += find_secrets(prompt)                        │     │
   │  │  for f in findings:                                      │     │
   │  │      results.append(f)                                   │     │
   │  │  FINAL_ANSWER = results                                  │     │
   │  └──────────────────────────────────────────────────────────┘     │
   │       │                                                            │
   │       ▼                                                            │
   │  execute_code_in_sandbox() ──► chunk_findings                     │
   └────────────────────────────────────────────────────────────────────┘

   Step 5: AGGREGATE
   ┌────────────────────────────────────────────────────────────────────┐
   │  39 chunk results                                                  │
   │       │                                                            │
   │       ▼                                                            │
   │  ResultVerifier.verify_findings()                                  │
   │       │                                                            │
   │       ├── Confidence scoring (L11 algorithm)                       │
   │       ├── Deduplicate across chunks                                │
   │       ├── Filter false positives                                   │
   │       └── Validate line numbers                                    │
   │       │                                                            │
   │       ▼                                                            │
   │  Final JSON: {findings: [...], summary: "...", files_scanned: 54} │
   └────────────────────────────────────────────────────────────────────┘
```

## Confidence Scoring (L11 Algorithm)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           CONFIDENCE SCORING ALGORITHM                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   START: 100 points                                                                 │
│       │                                                                              │
│       ▼                                                                              │
│   ┌───────────────────┐                                                             │
│   │ In dead code?     │──── YES ────► -50 points                                    │
│   └─────────┬─────────┘                                                             │
│             │ NO                                                                     │
│             ▼                                                                        │
│   ┌───────────────────┐                                                             │
│   │ In test file?     │──── YES ────► -30 points                                    │
│   └─────────┬─────────┘                                                             │
│             │ NO                                                                     │
│             ▼                                                                        │
│   ┌───────────────────┐                                                             │
│   │ Pattern match     │──── YES ────► -10 points                                    │
│   │ only (no context)?│                                                             │
│   └─────────┬─────────┘                                                             │
│             │ NO                                                                     │
│             ▼                                                                        │
│   ┌───────────────────┐                                                             │
│   │ In comment?       │──── YES ────► -40 points                                    │
│   └─────────┬─────────┘                                                             │
│             │ NO                                                                     │
│             ▼                                                                        │
│   ┌───────────────────┐                                                             │
│   │ Can't verify line?│──── YES ────► -20 points                                    │
│   └─────────┬─────────┘                                                             │
│             │ NO                                                                     │
│             ▼                                                                        │
│   ┌───────────────────┐                                                             │
│   │ Semantic verified?│──── YES ────► +20 points                                    │
│   └─────────┬─────────┘                                                             │
│             │ NO                                                                     │
│             ▼                                                                        │
│   ┌───────────────────┐                                                             │
│   │ Multiple          │──── YES ────► +15 points                                    │
│   │ indicators?       │                                                             │
│   └─────────┬─────────┘                                                             │
│             │                                                                        │
│             ▼                                                                        │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                         FINAL SCORE CLASSIFICATION                          │   │
│   ├─────────────────────────────────────────────────────────────────────────────┤   │
│   │                                                                             │   │
│   │    Score ≥ 80  ────►  ████████████████████  HIGH      (verified, report)   │   │
│   │                                                                             │   │
│   │    Score 50-79 ────►  ████████████          MEDIUM    (likely real)        │   │
│   │                                                                             │   │
│   │    Score 20-49 ────►  ██████                LOW       (uncertain)          │   │
│   │                                                                             │   │
│   │    Score < 20  ────►  ██                    FILTERED  (false positive)     │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Query Mode System (v2.3)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              QUERY MODE SELECTION (v2.3)                             │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   INPUT: query_mode parameter                                                        │
│       │                                                                              │
│       ▼                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                             │   │
│   │   ┌─────────┐      ┌──────────────────────────────────────────────────┐    │   │
│   │   │  AUTO   │ ───► │  Analyze query complexity:                       │    │   │
│   │   │(default)│      │  • >15 words or custom patterns? → SEMANTIC      │    │   │
│   │   └─────────┘      │  • Standard audit keywords? → SCANNER            │    │   │
│   │                    │  • Quoted literal strings? → LITERAL             │    │   │
│   │                    └──────────────────────────────────────────────────┘    │   │
│   │                                                                             │   │
│   │   ┌─────────┐      ┌──────────────────────────────────────────────────┐    │   │
│   │   │SEMANTIC │ ───► │  TRUE RLM Mode:                                  │    │   │
│   │   │         │      │  • LLM writes custom Python code                 │    │   │
│   │   └─────────┘      │  • Full REPL environment access                  │    │   │
│   │                    │  • Best for: complex feature discovery           │    │   │
│   │                    └──────────────────────────────────────────────────┘    │   │
│   │                                                                             │   │
│   │   ┌─────────┐      ┌──────────────────────────────────────────────────┐    │   │
│   │   │ SCANNER │ ───► │  Pre-built Tools Only:                           │    │   │
│   │   │         │      │  • Uses find_secrets(), find_xss(), etc.         │    │   │
│   │   └─────────┘      │  • Fastest for standard audits                   │    │   │
│   │                    │  • Best for: security, iOS, quality scans        │    │   │
│   │                    └──────────────────────────────────────────────────┘    │   │
│   │                                                                             │   │
│   │   ┌─────────┐      ┌──────────────────────────────────────────────────┐    │   │
│   │   │ LITERAL │ ───► │  Fast Grep Search:                               │    │   │
│   │   │         │      │  • No LLM involved (~40ms)                       │    │   │
│   │   └─────────┘      │  • Direct regex/string matching                  │    │   │
│   │                    │  • Best for: "find files containing X"           │    │   │
│   │                    └──────────────────────────────────────────────────┘    │   │
│   │                                                                             │   │
│   │   ┌─────────┐      ┌──────────────────────────────────────────────────┐    │   │
│   │   │ CUSTOM  │ ───► │  Semantic WITHOUT Pre-built Scanners:            │    │   │
│   │   │         │      │  • LLM interprets query freely                   │    │   │
│   │   └─────────┘      │  • No auto-enhancement or scanner injection      │    │   │
│   │                    │  • Best for: feature tracing, custom analysis    │    │   │
│   │                    └──────────────────────────────────────────────────┘    │   │
│   │                                                                             │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Cost Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              COST COMPARISON                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   SCENARIO: Analyze 500,000 token codebase                                          │
│                                                                                      │
│   ═══════════════════════════════════════════════════════════════════════════════   │
│                                                                                      │
│   METHOD 1: Direct (Traditional)                                                    │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │  500k tokens ──► Load into context ──► FAILS (context overflow)            │   │
│   │                                         or ~$15 with premium context        │   │
│   │                                                                             │   │
│   │  Cost: $15+ or N/A                     Context Used: 500k (100%)           │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│   METHOD 2: RLM via MCP                                                             │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │  500k tokens ──► Store as variable ──► LLM writes code ──► ~4k summary     │   │
│   │                  (NOT in context)       (examines portions)                 │   │
│   │                                                                             │   │
│   │  Cost: $0.10-0.50                      Context Used: ~4k (0.8%)            │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│   METHOD 3: RLM + Prompt Caching                                                    │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │  500k tokens ──► Cache system prompt ──► 90% token savings on cache hits   │   │
│   │                                                                             │   │
│   │  Base:  $0.80/1M input tokens                                               │   │
│   │  Cache: $0.08/1M input tokens (90% savings!)                               │   │
│   │                                                                             │   │
│   │  Cost: $0.05-0.20                      Context Used: ~4k (0.8%)            │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│   ═══════════════════════════════════════════════════════════════════════════════   │
│                                                                                      │
│   SAVINGS:  Traditional: $15  vs  RLM+Cache: $0.10  =  99.3% cost reduction        │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Component Interactions

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           COMPONENT INTERACTION MAP                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│                              ┌──────────────┐                                       │
│                              │  server.py   │                                       │
│                              │  (Entry)     │                                       │
│                              └──────┬───────┘                                       │
│                                     │                                               │
│           ┌─────────────────────────┼─────────────────────────┐                    │
│           │                         │                         │                    │
│           ▼                         ▼                         ▼                    │
│   ┌──────────────┐          ┌──────────────┐          ┌──────────────┐            │
│   │    config    │◄────────►│ RLMProcessor │◄────────►│MemoryStore   │            │
│   │              │          │              │          │              │            │
│   └──────┬───────┘          └──────┬───────┘          └──────┬───────┘            │
│          │                         │                         │                    │
│          │                         │                         │                    │
│          ▼                         ▼                         ▼                    │
│   ┌──────────────┐          ┌──────────────┐          ┌──────────────┐            │
│   │CacheManager  │◄────────►│    REPL      │          │   SQLite     │            │
│   │              │          │ Environment  │          │   Database   │            │
│   └──────┬───────┘          └──────┬───────┘          └──────────────┘            │
│          │                         │                                              │
│          │                         │                                              │
│          ▼                         ▼                                              │
│   ┌──────────────┐          ┌──────────────┐                                      │
│   │  OpenRouter  │          │ Structured   │                                      │
│   │  /Anthropic  │◄─────────│   Tools      │                                      │
│   │     API      │          │              │                                      │
│   └──────────────┘          └──────┬───────┘                                      │
│                                    │                                              │
│                                    ▼                                              │
│                             ┌──────────────┐                                      │
│   ┌──────────────┐          │  Structured  │          ┌──────────────┐            │
│   │FileCollector │─────────►│   Output     │◄─────────│ResultVerifier│            │
│   │              │          │              │          │              │            │
│   └──────┬───────┘          └──────────────┘          └──────────────┘            │
│          │                                                                        │
│          ▼                                                                        │
│   ┌──────────────┐                                                                │
│   │  Filesystem  │                                                                │
│   │              │                                                                │
│   └──────────────┘                                                                │
│                                                                                      │
│   LEGEND:                                                                           │
│   ───────                                                                           │
│   ─────►  Data flow                                                                │
│   ◄────►  Bidirectional                                                            │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## File Structure

```
RLM-Mem_MCP/
│
├── python/
│   └── src/
│       └── rlm_mem_mcp/
│           │
│           ├── server.py ─────────────── MCP server entry, tool routing
│           │
│           ├── rlm_processor.py ──────── Core orchestrator, pipeline
│           │   ├── CircuitBreaker
│           │   ├── RateLimiter
│           │   └── TrajectoryLogger
│           │
│           ├── repl_environment.py ───── Python sandbox, llm_query()
│           │   ├── CodeValidator
│           │   └── REPLState
│           │
│           ├── structured_tools.py ───── 30+ analysis functions
│           │   ├── Security tools
│           │   ├── iOS/Swift tools
│           │   ├── Quality tools
│           │   └── Architecture tools
│           │
│           ├── structured_output.py ──── Finding/Result classes, parser
│           │
│           ├── file_collector.py ─────── Async I/O, filtering
│           │
│           ├── cache_manager.py ──────── Prompt caching, token optimization
│           │
│           ├── result_verifier.py ────── L11 confidence scoring
│           │
│           ├── memory_store.py ───────── SQLite persistence
│           │
│           ├── semantic_cache.py ─────── Response caching
│           │
│           ├── project_analyzer.py ───── Tech stack detection
│           │
│           └── config.py ─────────────── Environment configuration
│
├── CLAUDE.md ─────────────────────────── Project documentation
├── ARCHITECTURE_ASCII.md ─────────────── This file
└── .mcp.json ─────────────────────────── MCP configuration
```

---

*Generated from RLM analysis of 54 files (~223k tokens)*
