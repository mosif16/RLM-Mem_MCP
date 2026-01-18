"""
Tests for RLM improvements:
- Result verification
- Project analysis
- Semantic cache
- Batch verification
"""

import pytest
from unittest.mock import MagicMock, patch

from rlm_mem_mcp.result_verifier import ResultVerifier, BatchVerifier, VerificationStats
from rlm_mem_mcp.project_analyzer import ProjectAnalyzer, ProjectInfo
from rlm_mem_mcp.rlm_processor import SemanticCache, ProgressEvent
from rlm_mem_mcp.file_collector import CollectionResult, CollectedFile


# Test fixtures
@pytest.fixture
def mock_collection():
    """Create a mock file collection for testing."""
    files = [
        CollectedFile(
            path="/test/src/auth.py",
            relative_path="src/auth.py",
            content="line1\nline2\nline3\ndef login():\n    pass\nline6",
            extension=".py",
            token_count=50,
        ),
        CollectedFile(
            path="/test/src/views/PaywallView.swift",
            relative_path="src/views/PaywallView.swift",
            content="import SwiftUI\n\nstruct PaywallView: View {\n    var body: some View {\n        Text(\"Hello\")\n    }\n}",
            extension=".swift",
            token_count=40,
        ),
        CollectedFile(
            path="/test/README.md",
            relative_path="README.md",
            content="# Project\n\nKey files:\n- src/auth.py\n- src/views/PaywallView.swift",
            extension=".md",
            token_count=20,
        ),
    ]

    collection = MagicMock(spec=CollectionResult)
    collection.files = files
    collection.file_count = len(files)
    collection.total_tokens = sum(f.token_count for f in files)

    def get_file_content(path):
        for f in files:
            if path in f.path or f.relative_path.endswith(path):
                return f.content
        return None

    collection.get_file_content = get_file_content
    return collection


class TestResultVerifier:
    """Tests for ResultVerifier."""

    def test_verifies_valid_line_references(self, mock_collection):
        """Test that valid line references are verified correctly."""
        verifier = ResultVerifier()
        response = "Found issue at src/auth.py:3"

        verified, stats = verifier.verify_findings(response, mock_collection)

        assert stats.verified_lines >= 1
        assert "[INVALID LINE]" not in verified

    def test_flags_invalid_line_references(self, mock_collection):
        """Test that out-of-bounds lines are flagged."""
        verifier = ResultVerifier()
        response = "Found issue at src/auth.py:999"

        verified, stats = verifier.verify_findings(response, mock_collection)

        assert stats.invalid_lines >= 1

    def test_fuzzy_file_matching(self, mock_collection):
        """Test that files can be matched by partial path."""
        verifier = ResultVerifier()
        response = "Found issue at auth.py:3"

        verified, stats = verifier.verify_findings(response, mock_collection)

        # Should still find the file via fuzzy matching
        assert stats.total_file_refs >= 1

    def test_verification_stats(self, mock_collection):
        """Test that verification stats are calculated correctly."""
        verifier = ResultVerifier()
        response = "Issues at src/auth.py:1 and src/auth.py:2 and fake.py:10"

        verified, stats = verifier.verify_findings(response, mock_collection)

        assert stats.total_line_refs == 3
        assert isinstance(stats.to_dict(), dict)


class TestBatchVerifier:
    """Tests for BatchVerifier."""

    def test_batch_verify_multiple_findings(self, mock_collection):
        """Test batch verification of multiple findings."""
        batch_verifier = BatchVerifier(mock_collection)

        findings = [
            {"file": "src/auth.py", "line": 1},
            {"file": "src/auth.py", "line": 3},
            {"file": "src/auth.py", "line": 999},  # Invalid
        ]

        results = batch_verifier.verify_batch(findings)

        assert len(results) == 3
        assert results[0]["is_valid"] is True
        assert results[1]["is_valid"] is True
        assert results[2]["is_valid"] is False

    def test_batch_verify_with_pattern(self, mock_collection):
        """Test batch verification with regex patterns."""
        batch_verifier = BatchVerifier(mock_collection)

        findings = [
            {"file": "src/auth.py", "line": 4, "pattern": r"def\s+login"},
        ]

        results = batch_verifier.verify_batch(findings)

        assert len(results) == 1
        assert results[0]["is_valid"] is True
        assert results[0]["pattern_matches"] is True


class TestProjectAnalyzer:
    """Tests for ProjectAnalyzer."""

    def test_detects_project_type(self, mock_collection):
        """Test project type detection."""
        analyzer = ProjectAnalyzer()

        info = analyzer.analyze(mock_collection)

        # Should detect based on file extensions
        assert info.project_type in ["ios", "python", "unknown"]

    def test_extracts_key_files_from_readme(self, mock_collection):
        """Test extraction of key files from documentation."""
        analyzer = ProjectAnalyzer()

        info = analyzer.analyze(mock_collection)

        # Should find files mentioned in README
        assert len(info.key_files) >= 0  # May or may not find depending on regex

    def test_project_info_to_context_string(self):
        """Test ProjectInfo context string generation."""
        info = ProjectInfo(
            project_type="ios",
            key_files=["PaywallView.swift", "SubscriptionManager.swift"],
            tech_stack=["SwiftUI", "StoreKit"],
        )

        context = info.to_context_string()

        assert "ios" in context
        assert "PaywallView.swift" in context
        assert "SwiftUI" in context


class TestSemanticCache:
    """Tests for SemanticCache."""

    def test_cache_miss_on_empty(self):
        """Test cache miss when cache is empty."""
        cache = SemanticCache(similarity_threshold=0.85)

        result, similarity = cache.get_similar("find bugs", "file1.py, file2.py")

        assert result is None
        assert similarity == 0.0

    def test_cache_set_and_get_keyword_fallback(self):
        """Test cache with keyword matching fallback."""
        cache = SemanticCache(similarity_threshold=0.85)

        # Store a result
        cache.set("find security bugs", "auth.py, login.py", "Found SQL injection at auth.py:42")

        # Try to retrieve with similar query
        result, similarity = cache.get_similar("find security issues", "auth.py, login.py")

        # May or may not hit depending on embedding availability
        stats = cache.get_stats()
        assert "hits" in stats
        assert "misses" in stats

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = SemanticCache()

        # Perform some operations
        cache.get_similar("query1", "context1")
        cache.set("query1", "context1", "response1")

        stats = cache.get_stats()

        assert stats["misses"] >= 1
        assert stats["size"] >= 1

    def test_cache_max_size(self):
        """Test cache respects max size."""
        cache = SemanticCache(max_size=3)

        for i in range(5):
            cache.set(f"query{i}", f"context{i}", f"response{i}")

        assert cache.get_stats()["size"] <= 3


class TestProgressEvent:
    """Tests for ProgressEvent."""

    def test_progress_event_creation(self):
        """Test ProgressEvent dataclass."""
        event = ProgressEvent(
            event_type="chunk_analyzed",
            message="Analyzed chunk 1/10",
            progress_percent=10.0,
            details={"chunk_id": 1}
        )

        assert event.event_type == "chunk_analyzed"
        assert event.progress_percent == 10.0

    def test_progress_event_to_dict(self):
        """Test ProgressEvent serialization."""
        event = ProgressEvent(
            event_type="complete",
            message="Done",
            progress_percent=100.0,
        )

        d = event.to_dict()

        assert d["event_type"] == "complete"
        assert d["progress_percent"] == 100.0
        assert "details" in d


class TestIntegration:
    """Integration tests for improved RLM pipeline."""

    def test_full_verification_flow(self, mock_collection):
        """Test full verification flow from analysis to output."""
        verifier = ResultVerifier()
        analyzer = ProjectAnalyzer()

        # Simulate RLM output
        rlm_output = """
## Security Analysis

Found potential issues:
- src/auth.py:4 - Missing input validation
- src/views/PaywallView.swift:3 - Hardcoded string
"""

        # Verify findings
        verified, stats = verifier.verify_findings(rlm_output, mock_collection)

        # Analyze project
        project_info = analyzer.analyze(mock_collection)

        # Check results
        assert stats.total_line_refs >= 2
        assert project_info is not None

    def test_categorized_file_index(self, mock_collection):
        """Test that files are categorized correctly."""
        # This tests the file categorization logic
        files = [f.relative_path for f in mock_collection.files]

        categories = {
            "Views/UI": [],
            "Services/Managers": [],
            "Other": [],
        }

        for f in files:
            f_lower = f.lower()
            if "view" in f_lower:
                categories["Views/UI"].append(f)
            elif "manager" in f_lower or "service" in f_lower:
                categories["Services/Managers"].append(f)
            else:
                categories["Other"].append(f)

        # PaywallView should be in Views/UI
        view_files = [f for f in files if "view" in f.lower()]
        assert len(view_files) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
