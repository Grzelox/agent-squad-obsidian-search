import pytest
from unittest.mock import Mock, patch
from langchain.schema import Document
from langchain_ollama import OllamaLLM

from modules.document_processor import ObsidianDocumentProcessor
from modules.config import AppConfig


class TestObsidianDocumentProcessor:
    """Test cases for enhanced ObsidianDocumentProcessor class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock()

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock(spec=OllamaLLM)
        llm.model = "test-model"
        return llm

    @pytest.fixture
    def vault_path(self, tmp_path):
        """Create a temporary test vault path."""
        return tmp_path / "test_vault"

    @pytest.fixture
    def processor(self, vault_path, mock_logger):
        """Create an ObsidianDocumentProcessor instance for testing."""
        with patch("modules.document_processor.get_config") as mock_config:
            config = AppConfig()
            config.markdown_mode = "single"
            config.markdown_strategy = "auto"
            config.summarization_enabled = False
            config.summarization_min_words = 500
            config.summarization_max_length = 200
            mock_config.return_value = config
            return ObsidianDocumentProcessor(vault_path, mock_logger)

    @pytest.fixture
    def processor_with_summarization(self, vault_path, mock_logger, mock_llm):
        """Create an ObsidianDocumentProcessor instance with summarization enabled."""
        with patch("modules.document_processor.get_config") as mock_config:
            config = AppConfig()
            config.markdown_mode = "elements"
            config.markdown_strategy = "hi_res"
            config.summarization_enabled = True
            config.summarization_min_words = 100
            config.summarization_max_length = 50
            mock_config.return_value = config
            return ObsidianDocumentProcessor(vault_path, mock_logger, mock_llm)

    @pytest.fixture
    def sample_document(self, vault_path):
        """Create a sample document for testing."""
        return Document(
            page_content="This is a [[test link]] with #hashtag and some content.\n\n\n  Extra whitespace  \n\n",
            metadata={
                "source": str(
                    vault_path / "test.md"
                ),  # Full path that can be made relative
                "category": "NarrativeText",
                "file_size": 1024,
                "file_modified": 1234567890,
            },
        )

    def test_init_basic(self, vault_path, mock_logger):
        """Test basic ObsidianDocumentProcessor initialization."""
        with patch("modules.document_processor.get_config") as mock_config:
            config = AppConfig()
            config.markdown_mode = "single"
            config.markdown_strategy = "auto"
            config.summarization_enabled = False
            config.summarization_min_words = 500
            config.summarization_max_length = 200
            mock_config.return_value = config
            processor = ObsidianDocumentProcessor(vault_path, mock_logger)

        assert processor.vault_path == vault_path
        assert processor.logger == mock_logger
        assert processor.config.markdown_mode == "single"
        assert processor.config.markdown_strategy == "auto"
        assert processor.config.summarization_enabled == False
        assert processor.llm is None

    def test_init_with_summarization(self, vault_path, mock_logger, mock_llm):
        """Test initialization with summarization enabled."""
        with patch("modules.document_processor.get_config") as mock_config:
            config = AppConfig()
            config.markdown_mode = "elements"
            config.markdown_strategy = "hi_res"
            config.summarization_enabled = True
            config.summarization_min_words = 300
            config.summarization_max_length = 100
            mock_config.return_value = config
            processor = ObsidianDocumentProcessor(vault_path, mock_logger, mock_llm)

        assert processor.config.markdown_mode == "elements"
        assert processor.config.markdown_strategy == "hi_res"
        assert processor.config.summarization_enabled == True
        assert processor.config.summarization_min_words == 300
        assert processor.config.summarization_max_length == 100
        assert processor.llm == mock_llm

    def test_load_documents_no_files(self, processor, vault_path):
        """Test loading documents when no markdown files exist."""
        # Create empty vault directory
        vault_path.mkdir(parents=True, exist_ok=True)

        result = processor.load_documents()

        assert result == []
        processor.logger.warning.assert_called_with("No markdown files found in vault")

    @patch("modules.document_processor.UnstructuredMarkdownLoader")
    def test_load_documents_success(
        self, mock_loader_class, processor, vault_path, sample_document
    ):
        """Test successful document loading with UnstructuredMarkdownLoader."""
        # Setup
        vault_path.mkdir(parents=True, exist_ok=True)
        test_file = vault_path / "test.md"
        test_file.write_text("# Test Content")

        # Update sample document to have correct source path
        sample_document.metadata["source"] = str(test_file)

        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [sample_document]
        mock_loader_class.return_value = mock_loader_instance

        # Execute
        result = processor.load_documents()

        # Verify
        assert len(result) == 1
        assert isinstance(result[0], Document)
        mock_loader_class.assert_called_once()
        processor.logger.info.assert_called()

    @patch("modules.document_processor.UnstructuredMarkdownLoader")
    def test_load_documents_with_summarization(
        self, mock_loader_class, processor_with_summarization, vault_path, mock_llm
    ):
        """Test document loading with summarization enabled."""
        # Setup
        vault_path.mkdir(parents=True, exist_ok=True)
        test_file = vault_path / "test.md"
        test_file.write_text("# Test Content")

        # Create a long document that should be summarized
        long_content = " ".join(["word"] * 150)  # 150 words, above 100 word threshold
        long_doc = Document(
            page_content=long_content,
            metadata={
                "source": str(test_file),
                "file_size": 1024,
                "file_modified": 1234567890,
            },
        )

        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [long_doc]
        mock_loader_class.return_value = mock_loader_instance

        # Mock LLM response for summarization
        mock_llm.invoke = Mock(return_value="This is a test summary.")

        with patch.object(
            processor_with_summarization, "_generate_summary"
        ) as mock_summary:
            mock_summary.return_value = "Test summary"

            # Execute
            result = processor_with_summarization.load_documents()

        # Verify - should have original doc + summary doc
        assert len(result) == 2  # original + summary document

        # Check that one is a summary document
        summary_docs = [doc for doc in result if doc.metadata.get("is_summary", False)]
        assert len(summary_docs) == 1
        assert summary_docs[0].metadata["content_type"] == "summary"

    def test_load_single_markdown_file(self, processor, vault_path):
        """Test loading a single markdown file."""
        # Setup
        vault_path.mkdir(parents=True, exist_ok=True)
        test_file = vault_path / "test.md"
        test_file.write_text("# Test Content")

        with patch(
            "modules.document_processor.UnstructuredMarkdownLoader"
        ) as mock_loader_class:
            mock_loader_instance = Mock()
            mock_doc = Document(
                page_content="Test content", metadata={"category": "Title"}
            )
            mock_loader_instance.load.return_value = [mock_doc]
            mock_loader_class.return_value = mock_loader_instance

            # Execute
            result = processor._load_single_markdown_file(test_file)

            # Verify
            assert len(result) == 1
            doc = result[0]
            assert doc.metadata["source"] == "test.md"  # relative path
            assert doc.metadata["file_path"] == str(test_file)
            assert doc.metadata["markdown_mode"] == "single"
            assert doc.metadata["markdown_strategy"] == "auto"
            assert "file_size" in doc.metadata
            assert "file_modified" in doc.metadata

    def test_process_document_content_obsidian_links(self, processor, vault_path):
        """Test processing of Obsidian wiki-style links."""
        doc = Document(
            page_content="Check out [[My Note]] and [[Another Note|Alias]]",
            metadata={"source": str(vault_path / "test.md")},
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
            metadata={"source": str(vault_path / "test.md")},
        )

        result = processor._process_document_content(doc)

        assert "important" in result.page_content
        assert "work-notes" in result.page_content

    def test_count_words(self, processor):
        """Test word counting functionality."""
        text1 = "This is a simple test."
        assert processor._count_words(text1) == 5

        text2 = "Text with #hashtags and [[links]] and **bold**."
        # Should remove markdown formatting for accurate count
        assert processor._count_words(text2) > 0

    def test_generate_summary_if_needed_short_document(
        self, processor_with_summarization
    ):
        """Test that short documents are not summarized."""
        short_doc = Document(
            page_content="Short content",  # Only 2 words, below 100 word threshold
            metadata={"source": "test.md"},
        )

        result = processor_with_summarization._generate_summary_if_needed(short_doc)
        assert result is None

    def test_generate_summary_if_needed_long_document(
        self, processor_with_summarization, mock_llm
    ):
        """Test that long documents are summarized."""
        long_content = " ".join(["word"] * 150)  # 150 words, above threshold
        long_doc = Document(page_content=long_content, metadata={"source": "test.md"})

        with patch.object(
            processor_with_summarization, "_generate_summary"
        ) as mock_summary:
            mock_summary.return_value = "Test summary"

            result = processor_with_summarization._generate_summary_if_needed(long_doc)

            assert result == "Test summary"
            mock_summary.assert_called_once()

    def test_generate_summary(self, processor_with_summarization, mock_llm):
        """Test summary generation."""
        content = "This is a long document that needs to be summarized."

        # Mock the LLM chain
        with patch("modules.document_processor.ChatPromptTemplate") as mock_template:
            mock_chain = Mock()
            mock_chain.invoke.return_value = "Generated summary."
            mock_template.from_messages.return_value.__or__ = Mock(
                return_value=mock_chain
            )

            result = processor_with_summarization._generate_summary(content, "test.md")

            assert result == "Generated summary."

    def test_generate_summary_no_llm(self, processor):
        """Test that summary generation raises error when no LLM available."""
        with pytest.raises(ValueError, match="LLM not available for summarization"):
            processor._generate_summary("content", "test.md")

    def test_create_summary_document(self, processor_with_summarization):
        """Test creation of summary document."""
        original_doc = Document(
            page_content="Original content",
            metadata={"source": "test.md", "word_count": 100},
        )
        summary_text = "This is a summary."

        result = processor_with_summarization._create_summary_document(
            original_doc, summary_text
        )

        assert result.page_content == summary_text
        assert result.metadata["content_type"] == "summary"
        assert result.metadata["is_summary"] == True
        assert result.metadata["original_source"] == "test.md"
        assert result.metadata["source"] == "test.md (summary)"
        assert result.metadata["summary_method"] == "llm_generated"

    def test_load_documents_error_handling(self, processor, vault_path):
        """Test error handling during document loading."""
        # Setup
        vault_path.mkdir(parents=True, exist_ok=True)
        test_file = vault_path / "test.md"
        test_file.write_text("# Test Content")

        with patch.object(processor, "_load_single_markdown_file") as mock_load:
            mock_load.side_effect = Exception("Loading failed")

            # Should not raise exception, but log error and continue
            result = processor.load_documents()

            # Should return empty list due to error
            assert result == []
            processor.logger.error.assert_called()

    def test_elements_mode_processing(self, vault_path, mock_logger):
        """Test processing with elements mode."""
        with patch("modules.document_processor.get_config") as mock_config:
            config = AppConfig()
            config.markdown_mode = "elements"
            config.markdown_strategy = "hi_res"
            config.summarization_enabled = False
            config.summarization_min_words = 500
            config.summarization_max_length = 200
            mock_config.return_value = config
            processor = ObsidianDocumentProcessor(vault_path, mock_logger)

            assert processor.config.markdown_mode == "elements"
            assert processor.config.markdown_strategy == "hi_res"
