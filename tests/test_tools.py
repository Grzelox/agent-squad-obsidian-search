import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from modules.tools import (
    list_obsidian_documents,
    search_documents_by_name,
    get_document_info,
    get_tools_for_agent,
    format_tool_descriptions,
)


class TestObsidianTools:
    """Test suite for Obsidian function calling tools."""

    def test_list_obsidian_documents_nonexistent_path(self):
        """Test listing documents with nonexistent vault path."""
        result = list_obsidian_documents.invoke({"vault_path": "/nonexistent/path"})
        data = json.loads(result)

        assert "error" in data
        assert "does not exist" in data["error"]

    def test_search_documents_by_name_nonexistent_path(self):
        """Test searching documents with nonexistent vault path."""
        result = search_documents_by_name.invoke(
            {"vault_path": "/nonexistent/path", "search_term": "test"}
        )
        data = json.loads(result)

        assert "error" in data
        assert "does not exist" in data["error"]

    def test_get_document_info_nonexistent_path(self):
        """Test getting document info with nonexistent vault path."""
        result = get_document_info.invoke(
            {"vault_path": "/nonexistent/path", "document_path": "test.md"}
        )
        data = json.loads(result)

        assert "error" in data
        assert "does not exist" in data["error"]

    @patch("modules.tools.Path")
    def test_list_obsidian_documents_success(self, mock_path):
        """Test successful document listing."""
        # Mock the path and file system
        mock_vault = MagicMock()
        mock_vault.exists.return_value = True
        mock_vault.rglob.return_value = []
        mock_path.return_value = mock_vault

        result = list_obsidian_documents.invoke({"vault_path": "/test/vault"})
        data = json.loads(result)

        assert "total_documents" in data
        assert "vault_path" in data
        assert "documents" in data
        assert data["total_documents"] == 0

    @patch("modules.tools.Path")
    def test_search_documents_by_name_success(self, mock_path):
        """Test successful document search by name."""
        # Mock the path and file system
        mock_vault = MagicMock()
        mock_vault.exists.return_value = True
        mock_vault.rglob.return_value = []
        mock_path.return_value = mock_vault

        result = search_documents_by_name.invoke(
            {"vault_path": "/test/vault", "search_term": "test"}
        )
        data = json.loads(result)

        assert "search_term" in data
        assert "total_matches" in data
        assert "vault_path" in data
        assert "documents" in data
        assert data["search_term"] == "test"
        assert data["total_matches"] == 0

    def test_get_tools_for_agent(self):
        """Test getting tools for agent."""
        tools = get_tools_for_agent("/test/vault")

        assert len(tools) == 4
        assert all(hasattr(tool, "name") for tool in tools)
        assert all(hasattr(tool, "description") for tool in tools)

    def test_format_tool_descriptions(self):
        """Test formatting tool descriptions."""
        descriptions = format_tool_descriptions()

        assert isinstance(descriptions, str)
        assert len(descriptions) > 0
        assert "list_obsidian_documents" in descriptions
        assert "search_documents_by_name" in descriptions
        assert "get_document_info" in descriptions

    @patch("modules.tools.Path")
    @patch("builtins.open")
    def test_list_obsidian_documents_with_files(self, mock_open, mock_path):
        """Test listing documents with actual files."""
        # Create mock files
        mock_file1 = MagicMock()
        mock_file1.relative_to.return_value = Path("test1.md")
        mock_file1.stat.return_value.st_size = 100
        mock_file1.stat.return_value.st_mtime = 1234567890

        mock_file2 = MagicMock()
        mock_file2.relative_to.return_value = Path("folder/test2.md")
        mock_file2.stat.return_value.st_size = 200
        mock_file2.stat.return_value.st_mtime = 1234567891

        # Mock vault
        mock_vault = MagicMock()
        mock_vault.exists.return_value = True
        mock_vault.rglob.return_value = [mock_file1, mock_file2]
        mock_path.return_value = mock_vault

        # Mock file content
        mock_open.return_value.__enter__.return_value.read.return_value = (
            "Test content for the document"
        )

        result = list_obsidian_documents.invoke({"vault_path": "/test/vault"})
        data = json.loads(result)

        assert data["total_documents"] == 2
        assert len(data["documents"]) == 2

        # Check that documents are sorted by modification time (newest first)
        assert data["documents"][0]["modified"] >= data["documents"][1]["modified"]

    @patch("modules.tools.Path")
    @patch("builtins.open")
    def test_search_documents_by_name_with_matches(self, mock_open, mock_path):
        """Test searching documents with matching files."""
        # Create mock files
        mock_file1 = MagicMock()
        mock_file1.relative_to.return_value = Path("python_guide.md")
        mock_file1.stem = "python_guide"
        mock_file1.stat.return_value.st_size = 100
        mock_file1.stat.return_value.st_mtime = 1234567890

        mock_file2 = MagicMock()
        mock_file2.relative_to.return_value = Path("javascript_basics.md")
        mock_file2.stem = "javascript_basics"
        mock_file2.stat.return_value.st_size = 200
        mock_file2.stat.return_value.st_mtime = 1234567891

        # Mock vault
        mock_vault = MagicMock()
        mock_vault.exists.return_value = True
        mock_vault.rglob.return_value = [mock_file1, mock_file2]
        mock_path.return_value = mock_vault

        # Mock file content
        mock_open.return_value.__enter__.return_value.read.return_value = "Test content"

        result = search_documents_by_name.invoke(
            {"vault_path": "/test/vault", "search_term": "python"}
        )
        data = json.loads(result)

        assert data["search_term"] == "python"
        assert data["total_matches"] == 1
        assert len(data["documents"]) == 1
        assert "python_guide.md" in data["documents"][0]["path"]

    @patch("modules.tools.Path")
    @patch("builtins.open")
    def test_get_document_info_success(self, mock_open, mock_path):
        """Test getting document info successfully."""
        # Mock the document path
        mock_vault = MagicMock()
        mock_doc_path = mock_vault / "test.md"
        mock_doc_path.exists.return_value = True
        mock_doc_path.stat.return_value.st_size = 100
        mock_doc_path.stat.return_value.st_ctime = 1234567890
        mock_doc_path.stat.return_value.st_mtime = 1234567891

        mock_path.return_value = mock_vault

        # Mock file content
        test_content = """# Test Document

This is a test document with some content.

## Section 1

Some content here.

[[Link to another doc]]

[External link](https://example.com)"""

        mock_open.return_value.__enter__.return_value.read.return_value = test_content

        result = get_document_info.invoke(
            {"vault_path": "/test/vault", "document_path": "test.md"}
        )
        data = json.loads(result)

        assert data["path"] == "test.md"
        assert "content_stats" in data
        assert data["content_stats"]["word_count"] > 0
        assert data["content_stats"]["line_count"] > 0
        assert data["content_stats"]["header_count"] >= 2
        assert "headers" in data
        assert "preview_start" in data


if __name__ == "__main__":
    pytest.main([__file__])
