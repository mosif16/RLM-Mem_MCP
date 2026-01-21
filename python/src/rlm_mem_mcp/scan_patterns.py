"""
Pattern constants for RLM structured scanning tools (v2.9).

Contains:
- Safe constant patterns by language
- Compile-time patterns for false positive filtering
- Sanitizer patterns for XSS detection
- Common regex patterns for various scanners
"""

from typing import Dict, List

# v2.7: Safe constant patterns library - patterns that indicate immutable/compile-time values
SAFE_CONSTANT_PATTERNS: Dict[str, List[str]] = {
    'swift': [
        r'^\s*let\s+\w+\s*[=:]',           # let declaration (Swift constant)
        r'^\s*static\s+let\s+\w+',         # static let (type-level constant)
        r'^\s*private\s+let\s+\w+',        # private let
        r'^\s*fileprivate\s+let\s+\w+',    # fileprivate let
        r'Bundle\.main\.',                  # Bundle resources (always exist at runtime)
        r'FileManager\.default\.urls',      # System directories (always valid)
        r'UIApplication\.shared\.',         # Singleton (always exists)
        r'ProcessInfo\.processInfo\.',      # System info (always exists)
    ],
    'javascript': [
        r'^\s*const\s+\w+\s*=',            # const declaration
        r'^\s*readonly\s+\w+',             # TypeScript readonly
        r'Object\.freeze\(',               # Frozen object
        r'Object\.seal\(',                 # Sealed object
    ],
    'python': [
        r'^\s*[A-Z][A-Z0-9_]+\s*=',        # ALL_CAPS convention (constants)
        r'@final',                          # Final decorator
        r'Final\[',                         # typing.Final annotation
        r'frozenset\(',                     # Immutable set
        r'tuple\(',                         # Immutable sequence
    ],
    'rust': [
        r'^\s*const\s+\w+',                # const declaration
        r'^\s*static\s+\w+',               # static (compile-time)
        r'Lazy<',                           # Lazy static
    ],
}

COMPILE_TIME_PATTERNS: List[str] = [
    # iOS/Swift compile-time patterns
    r'Bundle\.main',
    r'Bundle\(for:',
    r'FileManager\.default\.urls\(',
    r'UIScreen\.main',
    r'UIDevice\.current',
    r'ProcessInfo\.processInfo',
    r'NSLocale\.current',
    r'TimeZone\.current',
    # JavaScript/Node compile-time patterns
    r'process\.env\.',
    r'__dirname',
    r'__filename',
    r'import\.meta\.',
]

# Known sanitizer functions/libraries that make innerHTML safe
SANITIZER_PATTERNS: List[str] = [
    r'escapeHtml\s*\(',          # escapeHtml() function
    r'escape\s*\(',              # escape() function
    r'sanitize\s*\(',            # sanitize() function
    r'DOMPurify\.sanitize',      # DOMPurify library
    r'xss\s*\(',                 # xss() sanitizer
    r'htmlEncode\s*\(',          # htmlEncode() function
    r'encodeHTML\s*\(',          # encodeHTML() function
    r'safeHTML\s*\(',            # safeHTML() function
    r'purify\s*\(',              # purify() function
    r'sanitizeHtml\s*\(',        # sanitizeHtml() function
    r'createTextNode',           # Safe DOM method
    r'textContent\s*=',          # Safe property (often used in context)
    r'\.innerText\s*=',          # Safe property
    r'validator\.escape',        # validator.js escape
    r'he\.encode',               # he library encode
    r'entities\.encode',         # entities library
]

# Safe try! patterns in Swift (compile-time constants, safe APIs)
SAFE_TRY_PATTERNS: List[str] = [
    r'static\s+let\s+\w+\s*=\s*try!',  # Static constants
    r'NSRegularExpression',
    r'JSONDecoder',
    r'JSONEncoder',
    r'DateFormatter',
    r'NumberFormatter',
]

# Security patterns for secret detection
SECRET_PATTERNS: List[tuple] = [
    # API Keys and tokens
    (r'''(?i)(?:api[_-]?key|apikey)\s*[=:]\s*['"]([\w\-]{20,})['"]''', "API key", "HIGH"),
    (r'''(?i)(?:secret[_-]?key|secretkey)\s*[=:]\s*['"]([\w\-]{20,})['"]''', "Secret key", "HIGH"),
    (r'''(?i)(?:auth[_-]?token|authtoken)\s*[=:]\s*['"]([\w\-]{20,})['"]''', "Auth token", "HIGH"),
    (r'''(?i)(?:access[_-]?token|accesstoken)\s*[=:]\s*['"]([\w\-]{20,})['"]''', "Access token", "HIGH"),

    # AWS patterns
    (r'''AKIA[0-9A-Z]{16}''', "AWS Access Key ID", "CRITICAL"),
    (r'''(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*['"]([^'"]{40})['"]''', "AWS Secret Key", "CRITICAL"),

    # Private keys
    (r'''-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----''', "Private key", "CRITICAL"),
    (r'''-----BEGIN PGP PRIVATE KEY BLOCK-----''', "PGP private key", "CRITICAL"),

    # Database connection strings
    (r'''(?i)(?:mongodb|postgres|mysql|redis)://[^\s'"]+:[^\s'"]+@''', "Database connection string with credentials", "CRITICAL"),

    # Generic password patterns
    (r'''(?i)password\s*[=:]\s*['"][^'"]{8,}['"]''', "Hardcoded password", "HIGH"),
]

# SQL injection patterns
SQL_INJECTION_PATTERNS: List[tuple] = [
    # String concatenation in queries
    (r'''(?:execute|query|raw)\s*\([^)]*\+\s*(?:req\.|request\.|params\.|user)''',
     "SQL query with string concatenation", "CRITICAL"),
    (r'''(?:SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)[^;]*\+\s*\w+''',
     "SQL with string concatenation", "HIGH"),
    # f-strings in SQL (Python)
    (r'''(?:execute|query)\s*\(\s*f[\'"].*\{.*\}''',
     "SQL with f-string interpolation", "HIGH"),
    # Template literals in SQL (JavaScript)
    (r'''(?:query|execute)\s*\(\s*`[^`]*\$\{''',
     "SQL with template literal interpolation", "HIGH"),
]

# Command injection patterns
COMMAND_INJECTION_PATTERNS: List[tuple] = [
    # Shell execution with user input
    (r'''(?:os\.system|os\.popen|subprocess\.call|subprocess\.run|subprocess\.Popen)\s*\([^)]*\+''',
     "Shell command with string concatenation", "CRITICAL"),
    (r'''(?:exec|spawn|execSync|spawnSync)\s*\([^)]*\+''',
     "Command execution with concatenation", "CRITICAL"),
    (r'''(?:child_process|shelljs)''',
     "Shell execution module usage", "MEDIUM"),
]

# XSS vulnerability patterns
XSS_PATTERNS: List[tuple] = [
    (r'''\.innerHTML\s*=\s*[^'"<]''', "innerHTML with variable", "HIGH"),
    (r'''\.innerHTML\s*=\s*`''', "innerHTML with template literal", "HIGH"),
    (r'''document\.write\(''', "document.write() usage", "MEDIUM"),
    (r'''dangerouslySetInnerHTML\s*=''', "dangerouslySetInnerHTML usage", "MEDIUM"),
    (r'''\.outerHTML\s*=''', "outerHTML with variable", "HIGH"),
]

# iOS/Swift force unwrap patterns
FORCE_UNWRAP_PATTERNS: List[tuple] = [
    # Standard force unwrap
    (r'''(\w+)!(?!\s*=)(?!\s*\.)''', "Force unwrap", "MEDIUM"),
    # Force unwrap in optional chaining
    (r'''\?\s*\..*!''', "Force unwrap in optional chain", "MEDIUM"),
    # Implicitly unwrapped optional declaration
    (r'''(?:var|let)\s+\w+\s*:\s*\w+!''', "Implicitly unwrapped optional", "LOW"),
]

# Retain cycle patterns for iOS/Swift
RETAIN_CYCLE_PATTERNS: List[tuple] = [
    # Closure without weak/unowned self
    (r'''\{\s*(?!\[(?:weak|unowned)\s+self\])[^}]*\bself\b[^}]*\}''',
     "Closure capturing self without weak/unowned", "MEDIUM"),
    # Timer without weak self
    (r'''Timer\.[^}]*\{[^}]*self\.[^}]*\}''',
     "Timer closure without weak self", "HIGH"),
    # NotificationCenter without weak self
    (r'''NotificationCenter[^}]*\{[^}]*self\.[^}]*\}''',
     "NotificationCenter observer without weak self", "HIGH"),
]

# React-specific patterns
REACT_PATTERNS: List[tuple] = [
    # Missing dependency in useEffect
    (r'''useEffect\s*\(\s*\(\)\s*=>\s*\{[^}]*\}\s*,\s*\[\s*\]\s*\)''',
     "useEffect with empty deps may be missing dependencies", "LOW"),
    # State update in useEffect without cleanup
    (r'''useEffect\s*\(\s*\(\)\s*=>\s*\{[^}]*setState[^}]*\}[^)]*\)(?!\s*//.*cleanup)''',
     "setState in useEffect may need cleanup", "LOW"),
    # Inline function in JSX (performance)
    (r'''onClick\s*=\s*\{\s*\(\)\s*=>''',
     "Inline arrow function in JSX prop (re-renders)", "LOW"),
]

# Node.js specific patterns
NODE_PATTERNS: List[tuple] = [
    # Callback hell indicator
    (r'''(?:\w+\s*\([^)]*,\s*function\s*\([^)]*\)\s*\{){3,}''',
     "Nested callbacks (callback hell)", "MEDIUM"),
    # Unhandled promise rejection
    (r'''\.then\s*\([^)]*\)(?!\s*\.catch)''',
     "Promise without catch handler", "LOW"),
    # Sync file operations in async context
    (r'''(?:readFileSync|writeFileSync|existsSync)''',
     "Sync file operation (may block event loop)", "LOW"),
]

# Rust specific patterns
RUST_PATTERNS: List[tuple] = [
    # Unsafe blocks
    (r'''unsafe\s*\{''', "Unsafe block", "MEDIUM"),
    # Unwrap usage
    (r'''\.unwrap\(\)''', "unwrap() may panic", "MEDIUM"),
    # Expect without meaningful message
    (r'''\.expect\s*\(\s*["'][^"']*["']\s*\)''', "expect() with message", "LOW"),
]

# Code quality patterns
QUALITY_PATTERNS: Dict[str, List[tuple]] = {
    'todo': [
        (r'''#\s*(TODO|FIXME|HACK|XXX)[:.]?\s*(.*)''', "INFO"),
        (r'''//\s*(TODO|FIXME|HACK|XXX)[:.]?\s*(.*)''', "INFO"),
        (r'''/\*\s*(TODO|FIXME|HACK|XXX)[:.]?\s*(.*)''', "INFO"),
    ],
    'long_function': [
        # Patterns to detect function starts by language
        (r'''^\s*def\s+(\w+)\s*\(''', '.py'),
        (r'''^\s*(?:async\s+)?func\s+(\w+)\s*\(''', '.swift'),
        (r'''^\s*(?:async\s+)?function\s+(\w+)\s*\(''', '.js'),
        (r'''^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(''', '.ts'),
        (r'''^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)''', '.rs'),
        (r'''^\s*func\s+(\w+)''', '.go'),
    ],
}
