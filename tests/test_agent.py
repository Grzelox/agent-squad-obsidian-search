import pytest
from pathlib import Path
from unittest.mock import Mock, patch, call
from langchain.schema import Document

from modules.agent import ObsidianAgent
from modules.config import AppConfig
from modules.prompts import REACT_AGENT_PROMPT_TEMPLATE


class TestObsidianAgent:
    """Test cases for ObsidianAgent class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = AppConfig()
        config.model_name = "test-model"
        config.embedding_model = "test-embedding"
        config.persist_directory = "/test/chroma"
        config.collection_name = "test_collection"
        config.retrieval_k = 3
        return config

    @pytest.fixture
    def default_agent_params(self, mock_config):
        """Default parameters for creating an ObsidianAgent."""
        return {
            "obsidian_vault_path": "/test/vault",
            "model_name": mock_config.model_name,
            "embedding_model": mock_config.embedding_model,
            "persist_directory": mock_config.persist_directory,
            "log_file": mock_config.logs_file,
            "chroma_host": None,
            "chroma_port": None,
            "collection_name": mock_config.collection_name,
            "verbose": False,
            "quiet": False,
        }

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(page_content="Test content 1", metadata={"source": "doc1.md"}),
            Document(page_content="Test content 2", metadata={"source": "doc2.md"}),
        ]

    @pytest.fixture
    def mock_qa_result(self):
        """Create a mock QA chain result."""
        return {
            "answer": "This is a test answer.",
            "context": [
                Document(page_content="Context 1", metadata={"source": "doc1.md"}),
                Document(page_content="Context 2", metadata={"source": "doc2.md"}),
                Document(page_content="Context 3", metadata={"source": "doc1.md"}),
            ],
        }

    @patch("modules.agent.get_config")
    @patch("modules.agent.setup_agent_logger")
    @patch("modules.agent.OllamaEmbeddings")
    @patch("modules.agent.OllamaLLM")
    @patch("modules.agent.ObsidianDocumentProcessor")
    @patch("modules.agent.VectorStoreManager")
    def test_init_with_default_parameters(
        self,
        mock_vector_store_manager,
        mock_document_processor,
        mock_llm,
        mock_embeddings,
        mock_logger_setup,
        mock_get_config,
        mock_config,
        default_agent_params,
    ):
        """Test ObsidianAgent initialization with default parameters."""
        mock_get_config.return_value = mock_config
        mock_logger = Mock()
        mock_logger_setup.return_value = mock_logger

        agent = ObsidianAgent(**default_agent_params)

        # Verify attributes set from parameters
        assert agent.obsidian_vault_path == Path("/test/vault")
        assert agent.model_name == "test-model"
        assert agent.embedding_model == "test-embedding"
        assert agent.persist_directory == "/test/chroma"
        assert agent.collection_name == "test_collection"
        assert agent.verbose is False
        assert agent.quiet is False

        # Verify logger setup
        mock_logger_setup.assert_called_once_with(
            "logs/app.log", str(id(agent)), False, False
        )
        assert agent.logger == mock_logger

        # Verify LLM and embeddings initialization
        mock_embeddings.assert_called_once_with(model="test-embedding")
        mock_llm.assert_called_once_with(model="test-model")

        # Verify service initialization
        mock_document_processor.assert_called_once_with(
            Path("/test/vault"), mock_logger, mock_llm.return_value
        )
        mock_vector_store_manager.assert_called_once_with(
            embeddings=mock_embeddings.return_value,
            persist_directory="/test/chroma",
            logger=mock_logger,
            chroma_host=None,
            chroma_port=None,
            collection_name="test_collection",
        )

        # Verify initial state
        assert agent.qa_chain is None

    @patch("modules.agent.get_config")
    @patch("modules.agent.setup_agent_logger")
    @patch("modules.agent.OllamaEmbeddings")
    @patch("modules.agent.OllamaLLM")
    @patch("modules.agent.ObsidianDocumentProcessor")
    @patch("modules.agent.VectorStoreManager")
    def test_init_with_custom_parameters(
        self,
        mock_vector_store_manager,
        mock_document_processor,
        mock_llm,
        mock_embeddings,
        mock_logger_setup,
        mock_get_config,
        mock_config,
    ):
        """Test ObsidianAgent initialization with custom parameters."""
        mock_get_config.return_value = mock_config
        mock_logger = Mock()
        mock_logger_setup.return_value = mock_logger

        agent = ObsidianAgent(
            obsidian_vault_path="/custom/vault",
            model_name="custom-model",
            embedding_model="custom-embedding",
            persist_directory="/custom/persist",
            log_file="custom.log",
            chroma_host="localhost",
            chroma_port=8000,
            collection_name="custom_collection",
            verbose=True,
            quiet=True,
        )

        # Verify custom parameters override config
        assert agent.model_name == "custom-model"
        assert agent.embedding_model == "custom-embedding"
        assert agent.persist_directory == "/custom/persist"
        assert agent.chroma_host == "localhost"
        assert agent.chroma_port == 8000
        assert agent.collection_name == "custom_collection"
        assert agent.verbose is True
        assert agent.quiet is True

        # Verify logger setup with custom parameters
        mock_logger_setup.assert_called_once_with(
            "custom.log", str(id(agent)), True, True
        )

    @patch("modules.agent.get_config")
    @patch("modules.agent.setup_agent_logger")
    @patch("modules.agent.log_verbose")
    @patch("modules.agent.OllamaEmbeddings")
    @patch("modules.agent.OllamaLLM")
    @patch("modules.agent.ObsidianDocumentProcessor")
    @patch("modules.agent.VectorStoreManager")
    def test_init_logging_calls(
        self,
        mock_vector_store_manager,
        mock_document_processor,
        mock_llm,
        mock_embeddings,
        mock_log_verbose,
        mock_logger_setup,
        mock_get_config,
        mock_config,
        default_agent_params,
    ):
        """Test initialization logging calls."""
        mock_get_config.return_value = mock_config
        mock_logger = Mock()
        mock_logger_setup.return_value = mock_logger

        agent = ObsidianAgent(**default_agent_params)

        # Verify logging calls - persist_directory should fallback to config value
        expected_info_calls = [
            call("Initializing ObsidianAgent with vault: /test/vault"),
            call("Using model: test-model, embedding model: test-embedding"),
            call("Using local ChromaDB at /test/chroma"),
        ]
        mock_logger.info.assert_has_calls(expected_info_calls)

        mock_logger.debug.assert_called_with("Initializing embeddings and LLM")

        expected_verbose_calls = [
            call(mock_logger, "Creating OllamaEmbeddings with model: test-embedding"),
            call(mock_logger, "Creating OllamaLLM with model: test-model"),
        ]
        mock_log_verbose.assert_has_calls(expected_verbose_calls)

    @patch("modules.agent.get_config")
    @patch("modules.agent.setup_agent_logger")
    @patch("modules.agent.OllamaEmbeddings")
    @patch("modules.agent.OllamaLLM")
    @patch("modules.agent.ObsidianDocumentProcessor")
    @patch("modules.agent.VectorStoreManager")
    def test_init_with_remote_chroma(
        self,
        mock_vector_store_manager,
        mock_document_processor,
        mock_llm,
        mock_embeddings,
        mock_logger_setup,
        mock_get_config,
        mock_config,
        default_agent_params,
    ):
        """Test initialization with remote ChromaDB configuration."""
        mock_get_config.return_value = mock_config
        mock_logger = Mock()
        mock_logger_setup.return_value = mock_logger

        # Use default params but override chroma settings
        params = default_agent_params.copy()
        params["chroma_host"] = "remote-host"
        params["chroma_port"] = 9000

        agent = ObsidianAgent(**params)

        # Verify remote ChromaDB logging
        mock_logger.info.assert_any_call("Using remote ChromaDB at remote-host:9000")

    def test_initialize_with_existing_vectorstore(self, default_agent_params):
        """Test agent initialization when existing vector store is found."""
        with (
            patch("modules.agent.get_config"),
            patch("modules.agent.setup_agent_logger"),
            patch("modules.agent.OllamaEmbeddings"),
            patch("modules.agent.OllamaLLM"),
            patch("modules.agent.ObsidianDocumentProcessor"),
            patch("modules.agent.VectorStoreManager"),
            patch("builtins.print") as mock_print,
        ):

            agent = ObsidianAgent(**default_agent_params)

            # Mock vector store manager methods
            agent.vector_store_manager.get_connection_info.return_value = {
                "type": "local",
                "persist_directory": "/test/chroma",
                "collection": "default",
            }
            agent.vector_store_manager.load_existing_vectorstore.return_value = True

            # Mock _setup_qa_chain
            with patch.object(agent, "_setup_qa_chain") as mock_setup_qa:
                agent.initialize(force_rebuild=False)

                # Verify calls
                agent.vector_store_manager.load_existing_vectorstore.assert_called_once()
                agent.vector_store_manager.create_vectorstore.assert_not_called()
                mock_setup_qa.assert_called_once()

                # Verify logging
                agent.logger.info.assert_any_call("Using existing vector store")

                # Verify console output
                mock_print.assert_any_call("Using existing vector store")

    def test_initialize_with_force_rebuild(
        self, default_agent_params, sample_documents
    ):
        """Test agent initialization with force rebuild."""
        with (
            patch("modules.agent.get_config"),
            patch("modules.agent.setup_agent_logger"),
            patch("modules.agent.OllamaEmbeddings"),
            patch("modules.agent.OllamaLLM"),
            patch("modules.agent.ObsidianDocumentProcessor"),
            patch("modules.agent.VectorStoreManager"),
            patch("modules.agent.log_verbose") as mock_log_verbose,
            patch("builtins.print") as mock_print,
        ):

            agent = ObsidianAgent(**default_agent_params)

            # Mock methods
            agent.vector_store_manager.get_connection_info.return_value = {
                "type": "local",
                "persist_directory": "/test/chroma",
            }
            agent.document_processor.load_documents.return_value = sample_documents

            with patch.object(agent, "_setup_qa_chain") as mock_setup_qa:
                agent.initialize(force_rebuild=True)

                # Verify force rebuild skips existing check
                agent.vector_store_manager.load_existing_vectorstore.assert_not_called()
                agent.document_processor.load_documents.assert_called_once()
                agent.vector_store_manager.create_vectorstore.assert_called_once_with(
                    sample_documents
                )
                mock_setup_qa.assert_called_once()

                # Verify verbose logging
                expected_verbose_calls = [
                    call(agent.logger, "Starting document loading process"),
                    call(
                        agent.logger,
                        f"Loaded {len(sample_documents)} documents for vectorization",
                    ),
                ]
                mock_log_verbose.assert_has_calls(expected_verbose_calls)

    def test_initialize_no_existing_vectorstore(
        self, default_agent_params, sample_documents
    ):
        """Test agent initialization when no existing vector store is found."""
        with (
            patch("modules.agent.get_config"),
            patch("modules.agent.setup_agent_logger"),
            patch("modules.agent.OllamaEmbeddings"),
            patch("modules.agent.OllamaLLM"),
            patch("modules.agent.ObsidianDocumentProcessor"),
            patch("modules.agent.VectorStoreManager"),
            patch("builtins.print") as mock_print,
        ):

            agent = ObsidianAgent(**default_agent_params)

            # Mock methods
            agent.vector_store_manager.get_connection_info.return_value = {
                "type": "local",
                "persist_directory": "/test/chroma",
            }
            agent.vector_store_manager.load_existing_vectorstore.return_value = False
            agent.document_processor.load_documents.return_value = sample_documents

            with patch.object(agent, "_setup_qa_chain") as mock_setup_qa:
                agent.initialize(force_rebuild=False)

                # Verify new vector store creation
                agent.vector_store_manager.load_existing_vectorstore.assert_called_once()
                agent.document_processor.load_documents.assert_called_once()
                agent.vector_store_manager.create_vectorstore.assert_called_once_with(
                    sample_documents
                )
                mock_setup_qa.assert_called_once()

                # Verify console output
                mock_print.assert_any_call("Building new vector store...")

    def test_initialize_remote_connection_info(self, default_agent_params):
        """Test initialization with remote connection info display."""
        with (
            patch("modules.agent.get_config"),
            patch("modules.agent.setup_agent_logger"),
            patch("modules.agent.OllamaEmbeddings"),
            patch("modules.agent.OllamaLLM"),
            patch("modules.agent.ObsidianDocumentProcessor"),
            patch("modules.agent.VectorStoreManager"),
            patch("builtins.print") as mock_print,
        ):

            agent = ObsidianAgent(**default_agent_params)

            # Mock remote connection info
            agent.vector_store_manager.get_connection_info.return_value = {
                "type": "remote",
                "url": "http://localhost:8000",
            }
            agent.vector_store_manager.load_existing_vectorstore.return_value = True

            with patch.object(agent, "_setup_qa_chain"):
                agent.initialize()

                # Verify remote connection display
                mock_print.assert_any_call(
                    "Using remote ChromaDB at http://localhost:8000"
                )

    def test_initialize_error_handling(self, default_agent_params):
        """Test error handling during initialization."""
        with (
            patch("modules.agent.get_config"),
            patch("modules.agent.setup_agent_logger"),
            patch("modules.agent.OllamaEmbeddings"),
            patch("modules.agent.OllamaLLM"),
            patch("modules.agent.ObsidianDocumentProcessor"),
            patch("modules.agent.VectorStoreManager"),
        ):

            agent = ObsidianAgent(**default_agent_params)

            # Mock error after connection info setup but before try block
            agent.vector_store_manager.get_connection_info.return_value = {
                "type": "local",
                "persist_directory": "/test/chroma",
            }
            agent.vector_store_manager.load_existing_vectorstore.side_effect = (
                Exception("Connection error")
            )

            with pytest.raises(Exception, match="Connection error"):
                agent.initialize()

            # Verify error logging is called
            agent.logger.error.assert_called_once()

    def test_query_success(self, default_agent_params, mock_qa_result):
        """Test successful query processing."""
        with (
            patch("modules.agent.get_config"),
            patch("modules.agent.setup_agent_logger"),
            patch("modules.agent.OllamaEmbeddings"),
            patch("modules.agent.OllamaLLM"),
            patch("modules.agent.ObsidianDocumentProcessor"),
            patch("modules.agent.VectorStoreManager"),
            patch("modules.agent.log_verbose") as mock_log_verbose,
            patch("builtins.print") as mock_print,
        ):

            agent = ObsidianAgent(**default_agent_params)

            # Mock QA chain
            mock_qa_chain = Mock()
            mock_qa_chain.invoke.return_value = mock_qa_result
            mock_qa_chain.stream.return_value = [mock_qa_result]
            agent.qa_chain = mock_qa_chain

            question = "What is this about?"
            result = agent.query(question)

            # Verify QA chain invocation

            # Verify result structure
            expected_result = {
                "answer": "This is a test answer.",
                "sources": ["doc1.md", "doc2.md"],  # Unique sources
                "source_details": {
                    "original_sources": ["doc1.md", "doc2.md"],
                    "summary_sources": [],
                    "total_chunks": 3,
                    "used_function_calls": False,
                    "tools_used": [],
                },
            }
            assert result == expected_result

            # Verify logging
            agent.logger.info.assert_any_call(f"Received query: {question}")
            agent.logger.info.assert_any_call(
                "Query processed successfully. Found 3 chunks from 2 sources"
            )

            # Verify verbose logging
            expected_verbose_calls = [
                call(agent.logger, f"Query length: {len(question)} characters"),
                call(agent.logger, "Starting semantic search and retrieval process"),
                call(agent.logger, "Retrieved 3 document chunks"),
            ]
            mock_log_verbose.assert_has_calls(expected_verbose_calls)

            # Verify console output
            mock_print.assert_any_call(f"\nProcessing question: {question}")
            mock_print.assert_any_call("Searching knowledge base...")

    def test_query_no_qa_chain(self, default_agent_params):
        """Test query when QA chain is not initialized."""
        with (
            patch("modules.agent.get_config"),
            patch("modules.agent.setup_agent_logger"),
            patch("modules.agent.OllamaEmbeddings"),
            patch("modules.agent.OllamaLLM"),
            patch("modules.agent.ObsidianDocumentProcessor"),
            patch("modules.agent.VectorStoreManager"),
        ):

            agent = ObsidianAgent(**default_agent_params)
            # qa_chain remains None

            with pytest.raises(ValueError, match="QA chain not initialized"):
                agent.query("Test question")

            # Verify error logging
            agent.logger.error.assert_called_once_with("QA chain not initialized")

    def test_query_error_handling(self, default_agent_params):
        """Test error handling during query processing."""
        with (
            patch("modules.agent.get_config"),
            patch("modules.agent.setup_agent_logger"),
            patch("modules.agent.OllamaEmbeddings"),
            patch("modules.agent.OllamaLLM"),
            patch("modules.agent.ObsidianDocumentProcessor"),
            patch("modules.agent.VectorStoreManager"),
        ):

            agent = ObsidianAgent(**default_agent_params)

            # Mock QA chain with error
            mock_qa_chain = Mock()
            mock_qa_chain.invoke.side_effect = Exception("Query processing error")
            mock_qa_chain.stream.side_effect = Exception("Query processing error")
            agent.qa_chain = mock_qa_chain

            with pytest.raises(Exception, match="Query processing error"):
                agent.query("Test question")

            # Verify error logging
            agent.logger.error.assert_called_with(
                "Error processing query: Query processing error"
            )

    @patch("modules.agent.get_config")
    @patch("modules.agent.create_stuff_documents_chain")
    @patch("modules.agent.create_retrieval_chain")
    @patch("modules.agent.ChatPromptTemplate")
    def test_setup_qa_chain_success(
        self,
        mock_prompt_template,
        mock_create_retrieval_chain,
        mock_create_stuff_chain,
        mock_get_config,
        default_agent_params,
    ):
        """Test successful QA chain setup."""
        with (
            patch("modules.agent.setup_agent_logger"),
            patch("modules.agent.OllamaEmbeddings"),
            patch("modules.agent.OllamaLLM"),
            patch("modules.agent.ObsidianDocumentProcessor"),
            patch("modules.agent.VectorStoreManager"),
            patch("builtins.print") as mock_print,
        ):

            # Mock config for retrieval
            config = AppConfig()
            config.retrieval_k = 7
            mock_get_config.return_value = config

            agent = ObsidianAgent(**default_agent_params)

            # Mock vector store and retriever
            mock_vectorstore = Mock()
            mock_retriever = Mock()
            mock_vectorstore.as_retriever.return_value = mock_retriever
            agent.vector_store_manager.get_vectorstore.return_value = mock_vectorstore

            # Mock chain creation
            mock_qa_chain = Mock()
            mock_prompt = Mock()
            mock_question_answer_chain = Mock()

            mock_prompt_template.from_messages.return_value = mock_prompt
            mock_create_stuff_chain.return_value = mock_question_answer_chain
            mock_create_retrieval_chain.return_value = mock_qa_chain

            agent._setup_qa_chain()

            # Verify vectorstore retriever setup
            mock_vectorstore.as_retriever.assert_called_once_with(
                search_type="similarity", search_kwargs={"k": 7}
            )

            # Verify prompt template creation
            expected_system_prompt = (
                "Use the given context to answer the question. "
                "If you don't know the answer, say you don't know. "
                "Use three sentence maximum and keep the answer concise. "
                "Context: {context}"
            )
            mock_prompt_template.from_messages.assert_called_once_with(
                [
                    ("system", expected_system_prompt),
                    ("human", "{input}"),
                ]
            )

            # Verify chain creation
            mock_create_stuff_chain.assert_called_once_with(agent.llm, mock_prompt)
            mock_create_retrieval_chain.assert_called_once_with(
                mock_retriever, mock_question_answer_chain
            )

            # Verify qa_chain assignment
            assert agent.qa_chain == mock_qa_chain

            # Verify logging and console output
            agent.logger.info.assert_any_call("QA chain setup completed successfully")
            mock_print.assert_called_with("QA chain setup complete!")

    def test_setup_qa_chain_no_vectorstore(self, default_agent_params):
        """Test QA chain setup when vector store is not initialized."""
        with (
            patch("modules.agent.get_config"),
            patch("modules.agent.setup_agent_logger"),
            patch("modules.agent.OllamaEmbeddings"),
            patch("modules.agent.OllamaLLM"),
            patch("modules.agent.ObsidianDocumentProcessor"),
            patch("modules.agent.VectorStoreManager"),
        ):

            agent = ObsidianAgent(**default_agent_params)

            # Mock no vector store
            agent.vector_store_manager.get_vectorstore.return_value = None

            with pytest.raises(ValueError, match="Vector store not initialized"):
                agent._setup_qa_chain()

            # Verify error logging
            agent.logger.error.assert_called_once_with("Vector store not initialized")

    @patch("modules.agent.get_config")
    def test_setup_qa_chain_default_retrieval_k(
        self, mock_get_config, default_agent_params
    ):
        """Test QA chain setup with default RETRIEVAL_K value."""
        with (
            patch("modules.agent.setup_agent_logger"),
            patch("modules.agent.OllamaEmbeddings"),
            patch("modules.agent.OllamaLLM"),
            patch("modules.agent.ObsidianDocumentProcessor"),
            patch("modules.agent.VectorStoreManager"),
            patch("modules.agent.create_stuff_documents_chain"),
            patch("modules.agent.create_retrieval_chain"),
            patch("modules.agent.ChatPromptTemplate"),
            patch("builtins.print"),
        ):

            # Mock config with default RETRIEVAL_K
            config = AppConfig()  # Will use default retrieval_k = 3
            mock_get_config.return_value = config

            agent = ObsidianAgent(**default_agent_params)

            # Mock vector store
            mock_vectorstore = Mock()
            agent.vector_store_manager.get_vectorstore.return_value = mock_vectorstore

            agent._setup_qa_chain()

            # Verify default retrieval_k is used
            mock_vectorstore.as_retriever.assert_called_once_with(
                search_type="similarity", search_kwargs={"k": 3}
            )

    def test_setup_qa_chain_error_handling(self, default_agent_params):
        """Test error handling during QA chain setup."""
        with (
            patch("modules.agent.get_config"),
            patch("modules.agent.setup_agent_logger"),
            patch("modules.agent.OllamaEmbeddings"),
            patch("modules.agent.OllamaLLM"),
            patch("modules.agent.ObsidianDocumentProcessor"),
            patch("modules.agent.VectorStoreManager"),
        ):

            agent = ObsidianAgent(**default_agent_params)

            # Mock vectorstore to exist but fail during retriever setup
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.side_effect = Exception("Vectorstore error")
        agent.vector_store_manager.get_vectorstore.return_value = mock_vectorstore

        with pytest.raises(Exception, match="Vectorstore error"):
            agent._setup_qa_chain()

        # Verify error logging
        agent.logger.error.assert_called_with(
            "Error setting up QA chain: Vectorstore error"
        )

    def test_query_unique_sources_extraction(self, default_agent_params):
        """Test that query correctly extracts unique sources from context."""
        with (
            patch("modules.agent.get_config"),
            patch("modules.agent.setup_agent_logger"),
            patch("modules.agent.OllamaEmbeddings"),
            patch("modules.agent.OllamaLLM"),
            patch("modules.agent.ObsidianDocumentProcessor"),
            patch("modules.agent.VectorStoreManager"),
            patch("builtins.print"),
        ):

            agent = ObsidianAgent(**default_agent_params)

            # Mock QA chain with duplicate sources
            mock_qa_result_with_duplicates = {
                "answer": "Test answer",
                "context": [
                    Document(page_content="Content 1", metadata={"source": "doc1.md"}),
                    Document(page_content="Content 2", metadata={"source": "doc2.md"}),
                    Document(
                        page_content="Content 3", metadata={"source": "doc1.md"}
                    ),  # Duplicate
                    Document(page_content="Content 4", metadata={"source": "doc3.md"}),
                    Document(
                        page_content="Content 5", metadata={"source": "doc2.md"}
                    ),  # Duplicate
                ],
            }

            mock_qa_chain = Mock()
            mock_qa_chain.invoke.return_value = mock_qa_result_with_duplicates
            mock_qa_chain.stream.return_value = [mock_qa_result_with_duplicates]
            agent.qa_chain = mock_qa_chain

            result = agent.query("Test question")

            # Verify unique sources (preserving order)
            expected_sources = ["doc1.md", "doc2.md", "doc3.md"]
            assert result["sources"] == expected_sources

            # Verify logging includes all and unique counts
            agent.logger.info.assert_any_call(
                "Query processed successfully. Found 5 chunks from 3 sources"
            )
