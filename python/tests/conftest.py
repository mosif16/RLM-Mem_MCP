"""
Pytest configuration and fixtures for RLM-Mem MCP tests.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio

from rlm_mem_mcp.config import RLMConfig
from rlm_mem_mcp.file_collector import FileCollector, CollectionResult
from rlm_mem_mcp.cache_manager import CacheManager
from rlm_mem_mcp.memory_store import MemoryStore


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def rlm_config() -> RLMConfig:
    """Create a test RLM configuration."""
    return RLMConfig(
        api_key="test-api-key",
        model="claude-3-5-haiku-20241022",
        aggregator_model="claude-sonnet-4-20250514",
        max_chunk_tokens=4000,
        max_result_tokens=8000,
    )


@pytest.fixture
def file_collector(rlm_config: RLMConfig) -> FileCollector:
    """Create a FileCollector instance."""
    return FileCollector(rlm_config)


@pytest.fixture
def cache_manager(rlm_config: RLMConfig) -> CacheManager:
    """Create a CacheManager instance."""
    return CacheManager(rlm_config)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_files(temp_dir: Path) -> dict[str, Path]:
    """Create sample files for testing."""
    files = {}

    # Python file
    py_file = temp_dir / "sample.py"
    py_file.write_text('''
def hello_world():
    """Say hello."""
    print("Hello, World!")

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

class Calculator:
    """Simple calculator."""

    def multiply(self, x: int, y: int) -> int:
        return x * y
''')
    files["python"] = py_file

    # JavaScript file
    js_file = temp_dir / "sample.js"
    js_file.write_text('''
function greet(name) {
    console.log(`Hello, ${name}!`);
}

const add = (a, b) => a + b;

class Calculator {
    multiply(x, y) {
        return x * y;
    }
}

module.exports = { greet, add, Calculator };
''')
    files["javascript"] = js_file

    # Markdown file
    md_file = temp_dir / "README.md"
    md_file.write_text('''
# Sample Project

This is a sample project for testing.

## Features

- Feature 1
- Feature 2
- Feature 3

## Usage

```python
from sample import hello_world
hello_world()
```
''')
    files["markdown"] = md_file

    # JSON config file
    json_file = temp_dir / "config.json"
    json_file.write_text('''
{
    "name": "test-project",
    "version": "1.0.0",
    "settings": {
        "debug": true,
        "timeout": 30
    }
}
''')
    files["json"] = json_file

    # Create nested directory
    nested_dir = temp_dir / "src" / "utils"
    nested_dir.mkdir(parents=True)

    nested_file = nested_dir / "helpers.py"
    nested_file.write_text('''
def format_string(s: str) -> str:
    """Format a string."""
    return s.strip().lower()

def parse_int(s: str) -> int:
    """Parse an integer."""
    return int(s)
''')
    files["nested"] = nested_file

    return files


@pytest.fixture
def large_file(temp_dir: Path) -> Path:
    """Create a large file for testing chunked reading."""
    large_file = temp_dir / "large_file.txt"

    # Create a ~1MB file
    content = "x" * 1000 + "\n"  # ~1KB per line
    large_file.write_text(content * 1000)  # ~1MB

    return large_file


@pytest_asyncio.fixture
async def memory_store(temp_dir: Path) -> AsyncGenerator[MemoryStore, None]:
    """Create a temporary memory store for testing."""
    db_path = temp_dir / "test_memory.db"
    store = MemoryStore(db_path)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def mock_llm_response() -> dict:
    """Create a mock LLM response."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "claude-3-5-haiku-20241022",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from the LLM.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }
