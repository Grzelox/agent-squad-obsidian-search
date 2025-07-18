import pytest
from unittest.mock import Mock, patch
from langchain.schema import Document

from modules.vector_store_manager import VectorStoreManager


class TestVectorStoreManager:
    """Test cases for VectorStoreManager class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock()

    @pytest.fixture
    def mock_embeddings(self):
        """Create a mock OllamaEmbeddings instance."""
        return Mock()

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return {
            "COLLECTION_NAME": "test_collection",
            "CHUNK_SIZE": 1000,
            "CHUNK_OVERLAP": 200,
        }

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="This is the first document content",
                metadata={"source": "doc1.md"},
            ),
            Document(
                page_content="This is the second document content",
                metadata={"source": "doc2.md"},
            ),
            Document(
                page_content="This is the third document with much longer content to test chunking behavior",
                metadata={"source": "doc3.md"},
            ),
        ]

    @pytest.fixture
    def sample_chunks(self):
        """Create sample document chunks for testing."""
        return [
            Document(page_content="Chunk 1 content", metadata={"source": "doc1.md"}),
            Document(page_content="Chunk 2 content", metadata={"source": "doc1.md"}),
            Document(page_content="Chunk 3 content", metadata={"source": "doc2.md"}),
        ]

    @patch("modules.vector_store_manager.get_config")
    def test_init_local_mode(
        self, mock_get_config, mock_logger, mock_embeddings, mock_config
    ):
        """Test VectorStoreManager initialization in local mode."""
        mock_get_config.return_value = mock_config
        persist_directory = "/test/persist"

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory=persist_directory,
        )

        assert manager.logger == mock_logger
        assert manager.embeddings == mock_embeddings
        assert manager.persist_directory == persist_directory
        assert manager.chroma_host is None
        assert manager.chroma_port is None
        assert manager.vectorstore is None
        assert not manager.use_remote_client
        assert manager.collection_name == "test_collection"
        mock_logger.info.assert_called()

    @patch("modules.vector_store_manager.get_config")
    def test_init_remote_mode(
        self, mock_get_config, mock_logger, mock_embeddings, mock_config
    ):
        """Test VectorStoreManager initialization in remote mode."""
        mock_get_config.return_value = mock_config

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
            chroma_host="localhost",
            chroma_port=8000,
            collection_name="custom_collection",
        )

        assert manager.chroma_host == "localhost"
        assert manager.chroma_port == 8000
        assert manager.use_remote_client
        assert (
            manager.collection_name == "test_collection"
        )  # From config, not parameter
        mock_logger.info.assert_any_call(
            "VectorStoreManager initialized with remote ChromaDB"
        )
        mock_logger.info.assert_any_call("Remote ChromaDB: localhost:8000")

    @patch("modules.vector_store_manager.get_config")
    def test_create_vectorstore_local_success(
        self,
        mock_get_config,
        mock_logger,
        mock_embeddings,
        mock_config,
        sample_documents,
        sample_chunks,
    ):
        """Test successful vector store creation in local mode."""
        mock_get_config.return_value = mock_config

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
        )

        with (
            patch.object(
                manager, "_split_documents", return_value=sample_chunks
            ) as mock_split,
            patch.object(manager, "_create_local_vectorstore") as mock_create_local,
            patch("builtins.print") as mock_print,
        ):

            manager.create_vectorstore(sample_documents)

            mock_split.assert_called_once_with(sample_documents)
            mock_create_local.assert_called_once_with(sample_chunks)
            mock_logger.info.assert_any_call("Starting vector store creation")
            mock_logger.info.assert_any_call("Vector store created successfully!")
            mock_print.assert_any_call("Creating vector store...")
            mock_print.assert_any_call("Vector store created successfully!")

    @patch("modules.vector_store_manager.get_config")
    def test_create_vectorstore_remote_success(
        self,
        mock_get_config,
        mock_logger,
        mock_embeddings,
        mock_config,
        sample_documents,
        sample_chunks,
    ):
        """Test successful vector store creation in remote mode."""
        mock_get_config.return_value = mock_config

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
            chroma_host="localhost",
            chroma_port=8000,
        )

        with (
            patch.object(
                manager, "_split_documents", return_value=sample_chunks
            ) as mock_split,
            patch.object(manager, "_create_remote_vectorstore") as mock_create_remote,
        ):

            manager.create_vectorstore(sample_documents)

            mock_split.assert_called_once_with(sample_documents)
            mock_create_remote.assert_called_once_with(sample_chunks)

    @patch("modules.vector_store_manager.get_config")
    def test_create_vectorstore_error_handling(
        self,
        mock_get_config,
        mock_logger,
        mock_embeddings,
        mock_config,
        sample_documents,
    ):
        """Test error handling during vector store creation."""
        mock_get_config.return_value = mock_config

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
        )

        with patch.object(
            manager, "_split_documents", side_effect=Exception("Split error")
        ):
            with pytest.raises(Exception, match="Split error"):
                manager.create_vectorstore(sample_documents)

            mock_logger.error.assert_called_once()

    @patch("modules.vector_store_manager.get_config")
    @patch("modules.vector_store_manager.Chroma")
    def test_create_local_vectorstore(
        self,
        mock_chroma_class,
        mock_get_config,
        mock_logger,
        mock_embeddings,
        mock_config,
        sample_chunks,
    ):
        """Test local vector store creation."""
        mock_get_config.return_value = mock_config
        mock_vectorstore = Mock()
        mock_chroma_class.from_documents.return_value = mock_vectorstore

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
        )

        manager._create_local_vectorstore(sample_chunks)

        mock_chroma_class.from_documents.assert_called_once_with(
            documents=sample_chunks,
            embedding=mock_embeddings,
            persist_directory="/test/persist",
        )
        assert manager.vectorstore == mock_vectorstore
        mock_logger.debug.assert_called()

    @patch("modules.vector_store_manager.get_config")
    @patch("modules.vector_store_manager.chromadb")
    @patch("modules.vector_store_manager.Chroma")
    def test_create_remote_vectorstore(
        self,
        mock_chroma_class,
        mock_chromadb,
        mock_get_config,
        mock_logger,
        mock_embeddings,
        mock_config,
        sample_chunks,
    ):
        """Test remote vector store creation."""
        mock_get_config.return_value = mock_config
        mock_client = Mock()
        mock_chromadb.HttpClient.return_value = mock_client
        mock_vectorstore = Mock()
        mock_chroma_class.from_documents.return_value = mock_vectorstore

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
            chroma_host="localhost",
            chroma_port=8000,
        )

        manager._create_remote_vectorstore(sample_chunks)

        mock_chromadb.HttpClient.assert_called_once_with(host="localhost", port=8000)
        mock_chroma_class.from_documents.assert_called_once_with(
            documents=sample_chunks,
            embedding=mock_embeddings,
            client=mock_client,
            collection_name="test_collection",
        )
        assert manager.vectorstore == mock_vectorstore

    @patch("modules.vector_store_manager.get_config")
    def test_create_remote_vectorstore_missing_host_port(
        self, mock_get_config, mock_logger, mock_embeddings, mock_config, sample_chunks
    ):
        """Test remote vector store creation with missing host/port."""
        mock_get_config.return_value = mock_config

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
        )
        manager.use_remote_client = True  # Force remote mode but no host/port

        with pytest.raises(
            ValueError, match="Remote ChromaDB host and port must be specified"
        ):
            manager._create_remote_vectorstore(sample_chunks)

    @patch("modules.vector_store_manager.get_config")
    def test_load_existing_vectorstore_local_success(
        self, mock_get_config, mock_logger, mock_embeddings, mock_config
    ):
        """Test successful loading of existing local vector store."""
        mock_get_config.return_value = mock_config

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
        )

        with patch.object(
            manager, "_load_local_vectorstore", return_value=True
        ) as mock_load:
            result = manager.load_existing_vectorstore()

            assert result is True
            mock_load.assert_called_once()

    @patch("modules.vector_store_manager.get_config")
    def test_load_existing_vectorstore_remote_success(
        self, mock_get_config, mock_logger, mock_embeddings, mock_config
    ):
        """Test successful loading of existing remote vector store."""
        mock_get_config.return_value = mock_config

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
            chroma_host="localhost",
            chroma_port=8000,
        )

        with patch.object(
            manager, "_load_remote_vectorstore", return_value=True
        ) as mock_load:
            result = manager.load_existing_vectorstore()

            assert result is True
            mock_load.assert_called_once()

    @patch("modules.vector_store_manager.get_config")
    def test_load_existing_vectorstore_error_handling(
        self, mock_get_config, mock_logger, mock_embeddings, mock_config
    ):
        """Test error handling when loading existing vector store fails."""
        mock_get_config.return_value = mock_config

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
        )

        with patch.object(
            manager, "_load_local_vectorstore", side_effect=Exception("Load error")
        ):
            result = manager.load_existing_vectorstore()

            assert result is False
            mock_logger.error.assert_called()
            mock_logger.warning.assert_called_with(
                "Will proceed with creating new vector store"
            )

    @patch("modules.vector_store_manager.get_config")
    @patch("modules.vector_store_manager.Chroma")
    @patch("os.path.exists")
    def test_load_local_vectorstore_exists(
        self,
        mock_exists,
        mock_chroma_class,
        mock_get_config,
        mock_logger,
        mock_embeddings,
        mock_config,
    ):
        """Test loading existing local vector store when directory exists."""
        mock_get_config.return_value = mock_config
        mock_exists.return_value = True
        mock_vectorstore = Mock()
        mock_chroma_class.return_value = mock_vectorstore

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
        )

        with patch("builtins.print") as mock_print:
            result = manager._load_local_vectorstore()

            assert result is True
            assert manager.vectorstore == mock_vectorstore
            mock_exists.assert_called_once_with("/test/persist")
            mock_chroma_class.assert_called_once_with(
                persist_directory="/test/persist", embedding_function=mock_embeddings
            )
            mock_print.assert_called_with("Loading existing vector store...")
            mock_logger.info.assert_any_call(
                "Found existing vector store, loading from: /test/persist"
            )

    @patch("modules.vector_store_manager.get_config")
    @patch("os.path.exists")
    def test_load_local_vectorstore_not_exists(
        self, mock_exists, mock_get_config, mock_logger, mock_embeddings, mock_config
    ):
        """Test loading local vector store when directory doesn't exist."""
        mock_get_config.return_value = mock_config
        mock_exists.return_value = False

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
        )

        result = manager._load_local_vectorstore()

        assert result is False
        mock_logger.debug.assert_any_call("No existing vector store found")

    @patch("modules.vector_store_manager.get_config")
    @patch("modules.vector_store_manager.chromadb")
    @patch("modules.vector_store_manager.Chroma")
    def test_load_remote_vectorstore_exists(
        self,
        mock_chroma_class,
        mock_chromadb,
        mock_get_config,
        mock_logger,
        mock_embeddings,
        mock_config,
    ):
        """Test loading existing remote vector store when collection exists."""
        mock_get_config.return_value = mock_config

        # Setup mock client and collections
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_client.list_collections.return_value = [mock_collection]
        mock_chromadb.HttpClient.return_value = mock_client

        mock_vectorstore = Mock()
        mock_chroma_class.return_value = mock_vectorstore

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
            chroma_host="localhost",
            chroma_port=8000,
        )

        with patch("builtins.print") as mock_print:
            result = manager._load_remote_vectorstore()

            assert result is True
            assert manager.vectorstore == mock_vectorstore
            mock_chromadb.HttpClient.assert_called_with(host="localhost", port=8000)
            mock_client.list_collections.assert_called_once()
            mock_chroma_class.assert_called_once_with(
                client=mock_client,
                collection_name="test_collection",
                embedding_function=mock_embeddings,
            )
            mock_print.assert_called_with("Loading existing remote vector store...")

    @patch("modules.vector_store_manager.get_config")
    @patch("modules.vector_store_manager.chromadb")
    def test_load_remote_vectorstore_not_exists(
        self, mock_chromadb, mock_get_config, mock_logger, mock_embeddings, mock_config
    ):
        """Test loading remote vector store when collection doesn't exist."""
        mock_get_config.return_value = mock_config

        mock_client = Mock()
        mock_client.list_collections.return_value = []  # No collections
        mock_chromadb.HttpClient.return_value = mock_client

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
            chroma_host="localhost",
            chroma_port=8000,
        )

        result = manager._load_remote_vectorstore()

        assert result is False
        mock_logger.debug.assert_any_call(
            "Collection 'test_collection' not found on remote server"
        )

    @patch("modules.vector_store_manager.get_config")
    def test_load_remote_vectorstore_missing_host_port(
        self, mock_get_config, mock_logger, mock_embeddings, mock_config
    ):
        """Test loading remote vector store with missing host/port."""
        mock_get_config.return_value = mock_config

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
        )
        manager.use_remote_client = True  # Force remote mode but no host/port

        with pytest.raises(
            ValueError, match="Remote ChromaDB host and port must be specified"
        ):
            manager._load_remote_vectorstore()

    @patch("modules.vector_store_manager.get_config")
    def test_get_vectorstore(
        self, mock_get_config, mock_logger, mock_embeddings, mock_config
    ):
        """Test getting the vector store instance."""
        mock_get_config.return_value = mock_config

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
        )

        # Initially None
        assert manager.get_vectorstore() is None

        # Set a mock vectorstore
        mock_vectorstore = Mock()
        manager.vectorstore = mock_vectorstore
        assert manager.get_vectorstore() == mock_vectorstore

    @patch("modules.vector_store_manager.get_config")
    def test_get_connection_info_local(
        self, mock_get_config, mock_logger, mock_embeddings, mock_config
    ):
        """Test getting connection info for local mode."""
        mock_get_config.return_value = mock_config

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
        )

        info = manager.get_connection_info()

        expected = {
            "type": "local",
            "persist_directory": "/test/persist",
            "collection": "default",
        }
        assert info == expected

    @patch("modules.vector_store_manager.get_config")
    def test_get_connection_info_remote(
        self, mock_get_config, mock_logger, mock_embeddings, mock_config
    ):
        """Test getting connection info for remote mode."""
        mock_get_config.return_value = mock_config

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
            chroma_host="localhost",
            chroma_port=8000,
        )

        info = manager.get_connection_info()

        expected = {
            "type": "remote",
            "host": "localhost",
            "port": 8000,
            "collection": "test_collection",
            "url": "http://localhost:8000",
        }
        assert info == expected

    @patch("modules.vector_store_manager.get_config")
    @patch("modules.vector_store_manager.RecursiveCharacterTextSplitter")
    def test_split_documents(
        self,
        mock_splitter_class,
        mock_get_config,
        mock_logger,
        mock_embeddings,
        mock_config,
        sample_documents,
        sample_chunks,
    ):
        """Test document splitting functionality."""
        mock_get_config.return_value = mock_config

        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = sample_chunks
        mock_splitter_class.return_value = mock_splitter

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
        )

        with patch("builtins.print") as mock_print:
            result = manager._split_documents(sample_documents)

            assert result == sample_chunks
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
            )
            mock_splitter.split_documents.assert_called_once_with(sample_documents)
            mock_print.assert_called_with(
                f"Created {len(sample_chunks)} document chunks"
            )
            mock_logger.info.assert_any_call(
                f"Created {len(sample_chunks)} document chunks"
            )

    @patch("modules.vector_store_manager.get_config")
    @patch("modules.vector_store_manager.RecursiveCharacterTextSplitter")
    def test_split_documents_custom_config(
        self,
        mock_splitter_class,
        mock_get_config,
        mock_logger,
        mock_embeddings,
        sample_documents,
    ):
        """Test document splitting with custom configuration."""
        custom_config = {
            "COLLECTION_NAME": "test_collection",
            "CHUNK_SIZE": 500,
            "CHUNK_OVERLAP": 100,
        }
        mock_get_config.return_value = custom_config

        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = []
        mock_splitter_class.return_value = mock_splitter

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
        )

        manager._split_documents(sample_documents)

        mock_splitter_class.assert_called_once_with(
            chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
        )

    @patch("modules.vector_store_manager.get_config")
    @patch("modules.vector_store_manager.RecursiveCharacterTextSplitter")
    def test_split_documents_default_config(
        self,
        mock_splitter_class,
        mock_get_config,
        mock_logger,
        mock_embeddings,
        sample_documents,
    ):
        """Test document splitting with default configuration when config values are missing."""
        incomplete_config = {
            "COLLECTION_NAME": "test_collection"
        }  # Missing CHUNK_SIZE and CHUNK_OVERLAP
        mock_get_config.return_value = incomplete_config

        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = []
        mock_splitter_class.return_value = mock_splitter

        manager = VectorStoreManager(
            logger=mock_logger,
            embeddings=mock_embeddings,
            persist_directory="/test/persist",
        )

        manager._split_documents(sample_documents)

        mock_splitter_class.assert_called_once_with(
            chunk_size=1000,  # Default value
            chunk_overlap=200,  # Default value
            separators=["\n\n", "\n", " ", ""],
        )
