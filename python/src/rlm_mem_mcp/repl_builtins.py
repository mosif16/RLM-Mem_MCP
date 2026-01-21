"""
REPL Builtin helpers for file extraction and content manipulation.

Provides helper functions that are exposed in the REPL sandbox for
file access and content extraction.
"""

import re
from typing import Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .repl_state import REPLState

from .repl_security import strip_markdown_fences_from_content, PRELOADED_MODULES


def build_categorized_file_index(files: list[str]) -> str:
    """
    Build a categorized file index for better LLM context.

    Categories files by type to help LLM find relevant files quickly.

    Args:
        files: List of file paths

    Returns:
        Formatted string with categorized file listing
    """
    categories = {
        "Views/UI": [],
        "Models/Data": [],
        "Services/Managers": [],
        "Extensions/Widgets": [],
        "Tests": [],
        "Config": [],
        "Other": [],
    }

    for f in files:
        f_lower = f.lower()
        categorized = False

        # Check for extensions/widgets first (high priority)
        if "extension" in f_lower or "widget" in f_lower:
            categories["Extensions/Widgets"].append(f)
            categorized = True
        elif "view" in f_lower or "controller" in f_lower or "screen" in f_lower or "ui" in f_lower:
            categories["Views/UI"].append(f)
            categorized = True
        elif "model" in f_lower or "entity" in f_lower or "schema" in f_lower:
            categories["Models/Data"].append(f)
            categorized = True
        elif "service" in f_lower or "manager" in f_lower or "provider" in f_lower or "handler" in f_lower:
            categories["Services/Managers"].append(f)
            categorized = True
        elif "test" in f_lower or "spec" in f_lower or "_test" in f_lower:
            categories["Tests"].append(f)
            categorized = True
        elif any(ext in f for ext in [".json", ".yaml", ".yml", ".toml", ".plist", ".xml", ".env"]):
            categories["Config"].append(f)
            categorized = True

        if not categorized:
            categories["Other"].append(f)

    # Build formatted output
    parts = []
    for category, cat_files in categories.items():
        if cat_files:
            parts.append(f"\n### {category} ({len(cat_files)} files)")
            # Show up to 25 files per category
            for f in cat_files[:25]:
                parts.append(f"  - {f}")
            if len(cat_files) > 25:
                parts.append(f"  ... and {len(cat_files) - 25} more in this category")

    return "\n".join(parts)


def strip_preloaded_imports(code: str) -> str:
    """
    Remove import statements for pre-loaded modules.

    This allows LLM-generated code with `import re` to work,
    since `re` is already available in the sandbox.

    Args:
        code: Python code that may contain unnecessary imports

    Returns:
        Code with pre-loaded module imports stripped
    """
    lines = code.split('\n')
    filtered_lines = []

    for line in lines:
        stripped = line.strip()
        # Check for `import re` or `import re as ...`
        skip = False
        for module in PRELOADED_MODULES:
            if stripped == f"import {module}" or stripped.startswith(f"import {module} as "):
                skip = True
                break
            if stripped.startswith(f"from {module} import"):
                skip = True
                break

        if not skip:
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def extract_file_list(prompt: str) -> list[str]:
    """Extract list of actual file paths from the prompt content."""
    files = re.findall(r'### File: ([^\n]+)', prompt)
    return files


class REPLBuiltins:
    """Factory for creating REPL builtin helper functions."""

    def __init__(self, state: "REPLState"):
        self.state = state

    def create_extract_with_lines_function(self) -> Callable[[str, int], str]:
        """Create the extract_with_lines helper function for line-numbered extraction."""
        state = self.state

        def extract_with_lines(filepath: str, max_lines: int = 500) -> str:
            """
            Extract file content with line numbers.

            Args:
                filepath: Path to extract (partial match supported)
                max_lines: Maximum lines to return (default 500)

            Returns:
                Line-numbered content: "1: first line\\n2: second line\\n..."
            """
            # Find the file in prompt
            parts = state.prompt.split("### File:")

            for part in parts[1:]:  # Skip first empty part
                lines = part.split("\n")
                if not lines:
                    continue

                file_path = lines[0].strip()

                # Match if filepath is contained in or equals file_path
                if filepath in file_path or file_path.endswith(filepath):
                    # Get file content (skip the filename line)
                    content_lines = lines[1:]

                    # Strip markdown fences from content
                    content_lines = strip_markdown_fences_from_content(content_lines)

                    # Add line numbers
                    numbered_lines = []
                    for i, line in enumerate(content_lines[:max_lines], 1):
                        numbered_lines.append(f"{i}: {line}")

                    result = "\n".join(numbered_lines)

                    if len(content_lines) > max_lines:
                        result += f"\n... ({len(content_lines) - max_lines} more lines)"

                    return result

            return f"File not found: {filepath}"

        return extract_with_lines

    def create_pattern_search_function(self) -> Callable[[str, str | None], list[dict]]:
        """Create a pattern search function that returns structured results."""
        prompt = self.state.prompt

        def search_pattern(pattern: str, file_filter: str | None = None) -> list[dict]:
            """
            Search for a regex pattern across all files.

            Args:
                pattern: Regex pattern to search for
                file_filter: Optional file extension filter (e.g., ".swift")

            Returns:
                List of matches with file, line, content
            """
            results = []
            current_file = None
            line_num = 0
            in_fence = False  # Track if we're inside a markdown fence

            for line in prompt.split('\n'):
                if line.startswith("### File:"):
                    current_file = line.replace("### File:", "").strip()
                    line_num = 0
                    in_fence = False  # Reset fence state for new file
                    continue

                # Skip markdown fence lines (```language and ```)
                stripped = line.strip()
                if stripped.startswith('```'):
                    if not in_fence:
                        in_fence = True  # Opening fence
                    else:
                        in_fence = False  # Closing fence
                    continue

                # Only count lines that are inside the code fence (actual file content)
                if in_fence:
                    line_num += 1

                    if current_file and (not file_filter or current_file.endswith(file_filter)):
                        try:
                            if re.search(pattern, line, re.IGNORECASE):
                                results.append({
                                    "file": current_file,
                                    "line": line_num,
                                    "content": line.strip()[:200],
                                    "match": re.search(pattern, line, re.IGNORECASE).group(0)
                                })
                        except re.error:
                            pass

            return results[:100]  # Limit results

        return search_pattern


def create_repl_builtins(state: "REPLState") -> REPLBuiltins:
    """Factory function to create REPL builtins."""
    return REPLBuiltins(state)
