# RLM-Mem MCP Server - Task Runner
# Usage: just <recipe>
# Run `just --list` to see all available recipes

# Configuration
python := "python3"
venv_dir := "python/.venv"
src_dir := "python/src"
project_root := justfile_directory()

# Default recipe - show help
default:
    @just --list

# ==================== Environment Setup ====================

# Create virtual environment and install all dependencies
setup: _create-venv _install-deps
    @echo "✓ Setup complete! Run 'just run' to start the server"

# Create virtual environment
_create-venv:
    @echo "Creating virtual environment..."
    @{{ python }} -m venv {{ venv_dir }}
    @echo "✓ Virtual environment created at {{ venv_dir }}"

# Install dependencies in venv
_install-deps:
    @echo "Installing dependencies..."
    @{{ venv_dir }}/bin/pip install --upgrade pip
    @{{ venv_dir }}/bin/pip install -e "{{ project_root }}/python[dev]"
    @echo "✓ Dependencies installed"

# Clean up virtual environment and caches
clean:
    @echo "Cleaning up..."
    rm -rf {{ venv_dir }}
    rm -rf python/src/rlm_mem_mcp/__pycache__
    rm -rf python/src/*.egg-info
    rm -rf python/.pytest_cache
    rm -rf .ruff_cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    @echo "✓ Cleaned"

# Reinstall everything from scratch
reinstall: clean setup

# ==================== Running the Server ====================

# Run the MCP server (default)
run:
    @echo "Starting RLM-Mem MCP Server..."
    @cd {{ src_dir }} && {{ project_root }}/{{ venv_dir }}/bin/python -m rlm_mem_mcp.server

# Run with debug logging
run-debug:
    @echo "Starting RLM-Mem MCP Server (debug mode)..."
    @cd {{ src_dir }} && RLM_DEBUG=1 {{ project_root }}/{{ venv_dir }}/bin/python -m rlm_mem_mcp.server

# Run with custom API key (usage: just run-with-key sk-xxx)
run-with-key key:
    @echo "Starting RLM-Mem MCP Server with custom API key..."
    @cd {{ src_dir }} && OPENROUTER_API_KEY={{ key }} {{ project_root }}/{{ venv_dir }}/bin/python -m rlm_mem_mcp.server

# ==================== Testing ====================

# Run all tests
test:
    @echo "Running tests..."
    @cd python && {{ project_root }}/{{ venv_dir }}/bin/pytest tests/ -v

# Run tests with coverage
test-cov:
    @echo "Running tests with coverage..."
    @cd python && {{ project_root }}/{{ venv_dir }}/bin/pytest tests/ -v --cov=src/rlm_mem_mcp --cov-report=term-missing

# Run a specific test file (usage: just test-file tests/test_repl.py)
test-file file:
    @echo "Running {{ file }}..."
    @cd python && {{ project_root }}/{{ venv_dir }}/bin/pytest {{ file }} -v

# ==================== Code Quality ====================

# Format code with black
fmt:
    @echo "Formatting code..."
    @{{ venv_dir }}/bin/black python/src python/tests
    @echo "✓ Code formatted"

# Lint code with ruff
lint:
    @echo "Linting code..."
    @{{ venv_dir }}/bin/ruff check python/src python/tests
    @echo "✓ Lint passed"

# Fix lint issues automatically
lint-fix:
    @echo "Fixing lint issues..."
    @{{ venv_dir }}/bin/ruff check --fix python/src python/tests
    @echo "✓ Lint issues fixed"

# Type check with mypy
typecheck:
    @echo "Type checking..."
    @{{ venv_dir }}/bin/mypy python/src/rlm_mem_mcp
    @echo "✓ Type check passed"

# Run all checks (format, lint, typecheck, test)
check: fmt lint typecheck test
    @echo "✓ All checks passed"

# ==================== MCP Configuration ====================

# Generate MCP config for Claude Code (prints to stdout)
mcp-config:
    #!/usr/bin/env bash
    VENV_PYTHON="{{ project_root }}/{{ venv_dir }}/bin/python"
    SRC_DIR="{{ project_root }}/{{ src_dir }}"

    cat << EOF
    {
      "mcpServers": {
        "rlm": {
          "type": "stdio",
          "command": "$VENV_PYTHON",
          "args": ["-m", "rlm_mem_mcp.server"],
          "cwd": "$SRC_DIR",
          "env": {
            "OPENROUTER_API_KEY": "\${OPENROUTER_API_KEY}"
          }
        }
      }
    }
    EOF

# Write MCP config to .mcp.json (uses env var reference)
mcp-config-write:
    #!/usr/bin/env bash
    VENV_PYTHON="{{ project_root }}/{{ venv_dir }}/bin/python"
    SRC_DIR="{{ project_root }}/{{ src_dir }}"

    cat > "{{ project_root }}/.mcp.json" << EOF
    {
      "mcpServers": {
        "rlm": {
          "type": "stdio",
          "command": "$VENV_PYTHON",
          "args": ["-m", "rlm_mem_mcp.server"],
          "cwd": "$SRC_DIR",
          "env": {
            "OPENROUTER_API_KEY": "\${OPENROUTER_API_KEY}"
          }
        }
      }
    }
    EOF
    echo "✓ Written to .mcp.json"
    echo "Note: Set OPENROUTER_API_KEY in your environment or .env file"

# ==================== Environment Variables ====================

# Create .env template
env-template:
    #!/usr/bin/env bash
    if [ -f "{{ project_root }}/.env" ]; then
        echo "⚠ .env already exists. Remove it first or edit manually."
        exit 1
    fi
    cat > "{{ project_root }}/.env" << 'EOF'
    # RLM-Mem MCP Server Configuration
    # Copy this file and fill in your values

    # OpenRouter API Key (required)
    OPENROUTER_API_KEY=sk-or-v1-your-key-here

    # Optional: Override default models
    # RLM_MODEL=anthropic/claude-sonnet-4
    # RLM_AGGREGATOR_MODEL=anthropic/claude-sonnet-4

    # Optional: Debug mode
    # RLM_DEBUG=1
    EOF
    echo "✓ Created .env template"
    echo "Edit .env and add your OPENROUTER_API_KEY"

# Show current environment configuration
env-check:
    @echo "Environment Configuration:"
    @echo "=========================="
    @if [ -f "{{ project_root }}/.env" ]; then \
        echo "✓ .env file exists"; \
        grep -v '^#' "{{ project_root }}/.env" | grep -v '^$$' | sed 's/=.*/=***/' || true; \
    else \
        echo "⚠ No .env file found"; \
    fi
    @echo ""
    @echo "From shell environment:"
    @if [ -n "$OPENROUTER_API_KEY" ]; then \
        echo "✓ OPENROUTER_API_KEY is set"; \
    else \
        echo "⚠ OPENROUTER_API_KEY not set"; \
    fi

# ==================== Development Helpers ====================

# Open Python REPL with project imports
repl:
    @cd {{ src_dir }} && {{ project_root }}/{{ venv_dir }}/bin/python -i -c "print('RLM-Mem MCP REPL'); print('Try: from rlm_mem_mcp import *')"

# Show project info
info:
    @echo "RLM-Mem MCP Server"
    @echo "=================="
    @echo "Project root: {{ project_root }}"
    @echo "Python venv:  {{ venv_dir }}"
    @echo "Source dir:   {{ src_dir }}"
    @echo ""
    @if [ -d "{{ venv_dir }}" ]; then \
        echo "✓ Virtual environment exists"; \
        {{ venv_dir }}/bin/python --version; \
    else \
        echo "⚠ Virtual environment not found. Run 'just setup'"; \
    fi

# Watch for changes and run tests (requires watchexec)
watch-test:
    @which watchexec > /dev/null || (echo "Install watchexec: brew install watchexec" && exit 1)
    watchexec -w python/src -w python/tests -e py -- just test

# ==================== Release ====================

# Build package
build:
    @echo "Building package..."
    @cd python && {{ project_root }}/{{ venv_dir }}/bin/pip install build
    @cd python && {{ project_root }}/{{ venv_dir }}/bin/python -m build
    @echo "✓ Package built in python/dist/"

# Publish to PyPI (requires credentials)
publish: build
    @echo "Publishing to PyPI..."
    @cd python && {{ project_root }}/{{ venv_dir }}/bin/pip install twine
    @cd python && {{ project_root }}/{{ venv_dir }}/bin/twine upload dist/*
