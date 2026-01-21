"""
Scanner modules for RLM tools (v2.9).

This package contains specialized scanners organized by domain:
- security: Security vulnerability detection
- ios_swift: iOS/Swift-specific issues
- web_frontend: React, Vue, Angular, DOM, a11y, CSS
- rust: Rust safety and quality patterns
- node: Node.js async, security, and patterns
- quality: Code quality (long functions, complexity, smells)
- architecture: Codebase structure and dependencies
- batch: Orchestrated multi-scanner runs
"""

# Security scanners
from .security import (
    SecurityScanner,
    create_security_scanner,
)

# iOS/Swift scanners
from .ios_swift import (
    iOSSwiftScanner,
    create_ios_swift_scanner,
)

# Web/Frontend scanners
from .web_frontend import (
    WebFrontendScanner,
    create_web_frontend_scanner,
)

# Rust scanners
from .rust import (
    RustScanner,
    create_rust_scanner,
)

# Node.js scanners
from .node import (
    NodeScanner,
    create_node_scanner,
)

# Quality scanners
from .quality import (
    QualityScanner,
    create_quality_scanner,
)

# Architecture scanners
from .architecture import (
    ArchitectureScanner,
    create_architecture_scanner,
)

# Batch scan orchestration
from .batch import (
    BatchScanner,
    create_batch_scanner,
)

__all__ = [
    # Security
    "SecurityScanner",
    "create_security_scanner",
    # iOS/Swift
    "iOSSwiftScanner",
    "create_ios_swift_scanner",
    # Web/Frontend
    "WebFrontendScanner",
    "create_web_frontend_scanner",
    # Rust
    "RustScanner",
    "create_rust_scanner",
    # Node.js
    "NodeScanner",
    "create_node_scanner",
    # Quality
    "QualityScanner",
    "create_quality_scanner",
    # Architecture
    "ArchitectureScanner",
    "create_architecture_scanner",
    # Batch
    "BatchScanner",
    "create_batch_scanner",
]
