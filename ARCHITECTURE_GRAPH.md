# RLM-Mem MCP Server - System Architecture

## High-Level System Flow

```mermaid
flowchart TB
    subgraph Client["Claude Code (Client)"]
        CC[Claude Code CLI]
    end

    subgraph MCP["MCP Protocol Layer"]
        STDIO[JSON-RPC over stdio]
    end

    subgraph Server["RLM-Mem MCP Server"]
        subgraph Entry["Entry Point"]
            SRV[server.py<br/>create_server]
            INIT[get_instances<br/>Singleton Init]
        end

        subgraph Tools["MCP Tools"]
            T1[rlm_analyze]
            T2[rlm_query_text]
            T3[rlm_memory_store]
            T4[rlm_memory_recall]
            T5[rlm_status]
        end

        subgraph Core["Core Processing"]
            PROC[RLMProcessor<br/>Orchestrator]
            REPL[REPLEnvironment<br/>Python Sandbox]
            STRUCT[StructuredTools<br/>Analysis Library]
        end

        subgraph Support["Support Layer"]
            FC[FileCollector<br/>Async I/O]
            CM[CacheManager<br/>Prompt Caching]
            RV[ResultVerifier<br/>Confidence Scoring]
            SC[SemanticCache<br/>Response Cache]
            PA[ProjectAnalyzer<br/>Tech Detection]
            MS[MemoryStore<br/>SQLite Persistence]
        end
    end

    subgraph External["External Services"]
        OR[OpenRouter API]
        AN[Anthropic API]
        DB[(SQLite DB)]
    end

    CC <-->|MCP Protocol| STDIO
    STDIO <--> SRV
    SRV --> INIT
    INIT --> T1 & T2 & T3 & T4 & T5

    T1 & T2 --> PROC
    T3 & T4 --> MS
    T5 --> CM

    PROC --> FC
    PROC --> REPL
    PROC --> CM
    PROC --> RV

    REPL --> STRUCT
    REPL -.->|llm_query| OR
    REPL -.->|llm_query| AN

    FC -.->|Read Files| FS[(Filesystem)]
    MS <--> DB
    CM -.->|Cached Prompts| OR
    CM -.->|Cached Prompts| AN

    style Client fill:#e1f5fe
    style MCP fill:#fff3e0
    style Server fill:#f3e5f5
    style External fill:#e8f5e9
```

## TRUE RLM Processing Pipeline

```mermaid
flowchart LR
    subgraph Input["1. Input"]
        Q[Query]
        P[Paths]
    end

    subgraph Collection["2. Collection"]
        FC2[FileCollector]
        FILTER[Extension Filter]
        SKIP[Skip Dirs Filter]
    end

    subgraph Processing["3. TRUE RLM Processing"]
        direction TB
        STORE["Store as Variable<br/>(prompt = content)"]
        CODE["LLM Writes Python<br/>to examine prompt"]
        EXEC["Execute in Sandbox<br/>with llm_query()"]
        VARS["Results stored<br/>as variables"]
    end

    subgraph Verification["4. Verification"]
        RV2[ResultVerifier]
        CONF[Confidence Scoring]
        DEDUP[Deduplication]
    end

    subgraph Output["5. Output"]
        JSON[Structured JSON]
        MD[Markdown Report]
    end

    Q & P --> FC2
    FC2 --> FILTER --> SKIP
    SKIP --> STORE
    STORE --> CODE
    CODE --> EXEC
    EXEC --> VARS
    VARS -.->|iterate| CODE
    VARS --> RV2
    RV2 --> CONF --> DEDUP
    DEDUP --> JSON & MD

    style Input fill:#bbdefb
    style Collection fill:#c8e6c9
    style Processing fill:#fff9c4
    style Verification fill:#ffccbc
    style Output fill:#d1c4e9
```

## Component Dependency Graph

```mermaid
flowchart TB
    subgraph Config["Configuration"]
        RC[RLMConfig]
        SC2[ServerConfig]
    end

    subgraph Singletons["Singleton Instances"]
        CM2[CacheManager]
        FC3[FileCollector]
        PROC2[RLMProcessor]
        SEC[SemanticCache]
        RV3[ResultVerifier]
        PA2[ProjectAnalyzer]
    end

    subgraph Runtime["Runtime Components"]
        REPL2[REPLEnvironment]
        ST[StructuredTools]
        SO[StructuredOutput]
        CB[CircuitBreaker]
        RL[RateLimiter]
        TL[TrajectoryLogger]
    end

    RC --> CM2
    RC --> FC3
    RC & CM2 --> PROC2

    PROC2 --> REPL2
    PROC2 --> CB
    PROC2 --> RL
    PROC2 --> TL

    REPL2 --> ST
    ST --> SO

    FC3 -.-> PROC2
    SEC -.-> PROC2
    RV3 -.-> PROC2
    PA2 -.-> PROC2

    style Config fill:#ffcdd2
    style Singletons fill:#c5cae9
    style Runtime fill:#dcedc8
```

## Data Flow: rlm_analyze Request

```mermaid
sequenceDiagram
    participant CC as Claude Code
    participant SRV as server.py
    participant PROC as RLMProcessor
    participant FC as FileCollector
    participant REPL as REPLEnvironment
    participant API as OpenRouter/Anthropic
    participant RV as ResultVerifier

    CC->>SRV: rlm_analyze(query, paths)
    SRV->>SRV: get_instances() singleton init
    SRV->>FC: collect_paths_async(paths)
    FC-->>SRV: CollectionResult(files, tokens)

    SRV->>PROC: process_with_decomposition()

    loop For each chunk
        PROC->>PROC: enhance_query()
        PROC->>API: Generate analysis code
        API-->>PROC: Python code
        PROC->>REPL: execute_code_in_sandbox()

        opt LLM sub-query needed
            REPL->>API: llm_query(portion)
            API-->>REPL: sub-result
        end

        REPL-->>PROC: chunk findings
    end

    PROC->>PROC: aggregate_results()
    PROC->>RV: verify_findings()
    RV->>RV: score_confidence()
    RV->>RV: deduplicate()
    RV-->>PROC: verified findings

    PROC-->>SRV: RLMResult
    SRV-->>CC: JSON response
```

## REPL Sandbox Environment

```mermaid
flowchart TB
    subgraph Sandbox["REPLEnvironment Sandbox"]
        subgraph Globals["Available Globals"]
            PROMPT["prompt<br/>(file content)"]
            CTX["context<br/>(metadata)"]
            RES["results<br/>(accumulator)"]
            LLM["llm_query()<br/>(sub-LLM calls)"]
        end

        subgraph Tools["StructuredTools"]
            direction LR
            SEC3["Security<br/>find_secrets<br/>find_sql_injection<br/>find_xss"]
            IOS["iOS/Swift<br/>find_force_unwraps<br/>find_retain_cycles<br/>find_async_issues"]
            QUAL["Quality<br/>find_long_functions<br/>find_dead_code<br/>find_todos"]
            ARCH["Architecture<br/>map_architecture<br/>build_call_graph"]
        end

        subgraph Security["Security Controls"]
            VAL[CodeValidator<br/>AST-based]
            TIME[30s Timeout]
            MEM[Memory Limits]
        end
    end

    subgraph Forbidden["Forbidden Operations"]
        F1[File I/O]
        F2[Network]
        F3[Subprocess]
        F4[Import *]
    end

    Globals --> Tools
    Security -.->|Enforces| Sandbox
    Forbidden -.->|Blocked by| VAL

    style Sandbox fill:#e8f5e9
    style Forbidden fill:#ffcdd2
```

## Confidence Scoring Algorithm (L11)

```mermaid
flowchart TB
    START[Finding: 100 points] --> D1{In dead code?}
    D1 -->|Yes| SUB1["-50 points"]
    D1 -->|No| D2
    SUB1 --> D2{In test file?}
    D2 -->|Yes| SUB2["-30 points"]
    D2 -->|No| D3
    SUB2 --> D3{Pattern only?}
    D3 -->|Yes| SUB3["-10 points"]
    D3 -->|No| D4
    SUB3 --> D4{In comment?}
    D4 -->|Yes| SUB4["-40 points"]
    D4 -->|No| D5
    SUB4 --> D5{Can verify line?}
    D5 -->|No| SUB5["-20 points"]
    D5 -->|Yes| B1
    SUB5 --> B1{Semantic verified?}
    B1 -->|Yes| ADD1["+20 points"]
    B1 -->|No| B2
    ADD1 --> B2{Multiple indicators?}
    B2 -->|Yes| ADD2["+15 points"]
    B2 -->|No| FINAL
    ADD2 --> FINAL[Calculate Final Score]

    FINAL --> C1{Score >= 80}
    C1 -->|Yes| HIGH[HIGH Confidence]
    C1 -->|No| C2{Score >= 50}
    C2 -->|Yes| MEDIUM[MEDIUM Confidence]
    C2 -->|No| C3{Score >= 20}
    C3 -->|Yes| LOW[LOW Confidence]
    C3 -->|No| FILTERED[FILTERED<br/>False Positive]

    style HIGH fill:#c8e6c9
    style MEDIUM fill:#fff9c4
    style LOW fill:#ffccbc
    style FILTERED fill:#ffcdd2
```

## Query Mode System (v2.3)

```mermaid
flowchart TB
    Q[Query Input] --> MODE{query_mode?}

    MODE -->|auto| AUTO[Auto-detect<br/>complexity]
    MODE -->|semantic| SEM[TRUE RLM<br/>LLM writes code]
    MODE -->|scanner| SCAN[Pre-built<br/>scanners only]
    MODE -->|literal| LIT[Fast grep<br/>~40ms]
    MODE -->|custom| CUST[Semantic without<br/>pre-built scanners]

    AUTO --> DETECT{Query complexity}
    DETECT -->|>15 words| SEM
    DETECT -->|standard audit| SCAN
    DETECT -->|quoted strings| LIT

    SEM --> REPL3[REPL Environment]
    SCAN --> ST2[StructuredTools]
    LIT --> GREP[Regex Search]
    CUST --> REPL3

    REPL3 --> OUT[Results]
    ST2 --> OUT
    GREP --> OUT

    style AUTO fill:#bbdefb
    style SEM fill:#c8e6c9
    style SCAN fill:#fff9c4
    style LIT fill:#ffccbc
    style CUST fill:#d1c4e9
```

## File Structure

```
RLM-Mem_MCP/
├── python/
│   ├── src/rlm_mem_mcp/
│   │   ├── server.py           # MCP server entry point
│   │   ├── rlm_processor.py    # Core RLM orchestrator
│   │   ├── repl_environment.py # Python sandbox
│   │   ├── structured_tools.py # Analysis tool library
│   │   ├── structured_output.py# Output formatting
│   │   ├── file_collector.py   # Async file I/O
│   │   ├── cache_manager.py    # Prompt caching
│   │   ├── result_verifier.py  # Confidence scoring
│   │   ├── memory_store.py     # SQLite persistence
│   │   ├── semantic_cache.py   # Response caching
│   │   ├── project_analyzer.py # Tech stack detection
│   │   └── config.py           # Configuration
│   └── tests/
├── CLAUDE.md                   # Project documentation
└── .mcp.json                   # MCP config
```

## Cost Optimization Flow

```mermaid
flowchart LR
    subgraph Traditional["Traditional Approach"]
        T1[500k tokens] --> T2[Load into context]
        T2 --> T3[Context overflow<br/>or ~$15 cost]
    end

    subgraph RLM["RLM Approach"]
        R1[500k tokens] --> R2[Store as variable]
        R2 --> R3[LLM writes code]
        R3 --> R4[Examine portions]
        R4 --> R5[~4k summary<br/>~$0.10-0.50]
    end

    subgraph Caching["With Prompt Caching"]
        C1[First request] --> C2[Cache system prompt]
        C2 --> C3[Subsequent requests]
        C3 --> C4[90% token savings<br/>$0.08/1M vs $0.80/1M]
    end

    style Traditional fill:#ffcdd2
    style RLM fill:#c8e6c9
    style Caching fill:#bbdefb
```

---

*Generated from RLM analysis of 54 files (~223k tokens)*
