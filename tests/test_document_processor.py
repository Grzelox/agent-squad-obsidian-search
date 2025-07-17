import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from langchain.schema import Document

from modules.document_processor import ObsidianDocumentProcessor


class TestObsidianDocumentProcessor:
    """Test cases for ObsidianDocumentProcessor class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock()

    @pytest.fixture
    def vault_path(self):
        """Create a test vault path."""
        return Path("/test/vault")

    @pytest.fixture
    def processor(self, vault_path, mock_logger):
        """Create an ObsidianDocumentProcessor instance for testing."""
        return ObsidianDocumentProcessor(vault_path, mock_logger)

    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        return Document(
            page_content="This is a [[test link]] with #hashtag and some content.\n\n\n  Extra whitespace  \n\n",
            metadata={"source": "/test/vault/subfolder/test.md"}
        )

    def test_init(self, vault_path, mock_logger):
        """Test ObsidianDocumentProcessor initialization."""
        processor = ObsidianDocumentProcessor(vault_path, mock_logger)
        
        assert processor.vault_path == vault_path
        assert processor.logger == mock_logger

    @patch('modules.document_processor.DirectoryLoader')
    def test_load_documents_success(self, mock_directory_loader, processor, sample_document):
        """Test successful document loading and processing."""
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [sample_document]
        mock_directory_loader.return_value = mock_loader_instance

        result = processor.load_documents()

        assert len(result) == 1
        assert isinstance(result[0], Document)
        
        mock_directory_loader.assert_called_once_with(
            str(processor.vault_path),
            glob="**/*.md",
            show_progress=True,
        )
        
        mock_loader_instance.load.assert_called_once()
        
        processor.logger.info.assert_called()
        processor.logger.debug.assert_called()

    @patch('modules.document_processor.DirectoryLoader')
    def test_load_documents_empty_vault(self, mock_directory_loader, processor):
        """Test loading documents from empty vault."""
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = []
        mock_directory_loader.return_value = mock_loader_instance

        result = processor.load_documents()

        assert result == []
        processor.logger.info.assert_called()

    @patch('modules.document_processor.DirectoryLoader')
    def test_load_documents_error_handling(self, mock_directory_loader, processor):
        """Test error handling during document loading."""
        mock_directory_loader.side_effect = Exception("Failed to load documents")

        with pytest.raises(Exception, match="Failed to load documents"):
            processor.load_documents()

        processor.logger.error.assert_called_once()

    @patch('modules.document_processor.DirectoryLoader')
    def test_load_documents_multiple_files(self, mock_directory_loader, processor):
        """Test loading multiple documents."""
        docs = [
            Document(page_content="First [[document]]", metadata={"source": "/test/vault/doc1.md"}),
            Document(page_content="Second #document", metadata={"source": "/test/vault/doc2.md"}),
            Document(page_content="Third document", metadata={"source": "/test/vault/subfolder/doc3.md"})
        ]
        
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = docs
        mock_directory_loader.return_value = mock_loader_instance

        result = processor.load_documents()

        # Verify
        assert len(result) == 3
        assert all(isinstance(doc, Document) for doc in result)

    def test_process_document_content_obsidian_links(self, processor, vault_path):
        """Test processing of Obsidian wiki-style links."""
        doc = Document(
            page_content="Check out [[My Note]] and [[Another Note|Alias]]",
            metadata={"source": str(vault_path / "test.md")}
        )

        result = processor._process_document_content(doc)

        assert "My Note" in result.page_content
        assert "Another Note|Alias" in result.page_content
        assert "[[" not in result.page_content
        assert "]]" not in result.page_content

    def test_process_document_content_hashtags(self, processor, vault_path):
        """Test processing of hashtags."""
        doc = Document(
            page_content="This has #important and #work-notes tags",
            metadata={"source": str(vault_path / "test.md")}
        )

        result = processor._process_document_content(doc)

        assert "important" in result.page_content
        assert "work-notes" in result.page_content

    def test_process_document_content_whitespace_cleanup(self, processor, vault_path):
        """Test cleanup of excessive whitespace."""
        doc = Document(
            page_content="Line 1\n\n\n\n\nLine 2\n   \n  \nLine 3",
            metadata={"source": str(vault_path / "test.md")}
        )

        result = processor._process_document_content(doc)

        assert "\n\n\n\n\n" not in result.page_content
        assert result.page_content.count("\n\n") <= result.page_content.count("Line")

    def test_process_document_content_metadata_source_path(self, processor, vault_path):
        """Test that source path is made relative to vault path."""
        doc = Document(
            page_content="Some content",
            metadata={"source": str(vault_path / "subfolder" / "document.md")}
        )

        result = processor._process_document_content(doc)

        expected_relative_path = "subfolder/document.md"
        assert result.metadata["source"] == expected_relative_path

    def test_process_document_content_combined_processing(self, processor, vault_path):
        """Test document processing with all transformations combined."""
        doc = Document(
            page_content="See [[My Note]] about #productivity\n\n\n\nMore content with [[Another Link]]",
            metadata={"source": str(vault_path / "notes" / "test.md")}
        )

        result = processor._process_document_content(doc)

        assert "My Note" in result.page_content
        assert "productivity" in result.page_content
        assert "Another Link" in result.page_content
        assert "[[" not in result.page_content
        assert "]]" not in result.page_content
        assert result.metadata["source"] == "notes/test.md"

    def test_process_document_content_empty_content(self, processor, vault_path):
        """Test processing document with empty content."""
        doc = Document(
            page_content="",
            metadata={"source": str(vault_path / "empty.md")}
        )

        result = processor._process_document_content(doc)

        assert result.page_content == ""
        assert result.metadata["source"] == "empty.md"

    def test_process_document_content_no_obsidian_syntax(self, processor, vault_path):
        """Test processing document without Obsidian syntax."""
        original_content = "This is plain markdown content with no special syntax."
        doc = Document(
            page_content=original_content,
            metadata={"source": str(vault_path / "plain.md")}
        )

        result = processor._process_document_content(doc)

        assert result.page_content == original_content
        assert result.metadata["source"] == "plain.md"

    def test_process_document_content_preserves_other_metadata(self, processor, vault_path):
        """Test that processing preserves other metadata fields."""
        doc = Document(
            page_content="[[Test content]]",
            metadata={
                "source": str(vault_path / "test.md"),
                "custom_field": "custom_value",
                "another_field": 123
            }
        )

        result = processor._process_document_content(doc)

        assert result.metadata["custom_field"] == "custom_value"
        assert result.metadata["another_field"] == 123
        assert result.metadata["source"] == "test.md"