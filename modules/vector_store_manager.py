import os
import logging
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
import chromadb
from .config import get_config


class VectorStoreManager:
    """Manages vector store operations including creation, loading, and text chunking."""

    def __init__(
        self,
        logger: logging.Logger,
        embeddings: OllamaEmbeddings,
        persist_directory: str,
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None,
        collection_name: str = "default_collection",
    ):
        self.logger = logger
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.vectorstore: Optional[Chroma] = None
        self.use_remote_client = chroma_host is not None

        self.config = get_config()
        self.collection_name = self.config.get("COLLECTION_NAME")

        self.logger.info(
            f"VectorStoreManager initialized with {'remote' if self.use_remote_client else 'local'} ChromaDB"
        )
        if self.use_remote_client:
            self.logger.info(f"Remote ChromaDB: {chroma_host}:{chroma_port}")

    def create_vectorstore(self, documents: List[Document]) -> None:
        """Create or load vector store from documents."""
        self.logger.info("Starting vector store creation")
        print("Creating vector store...")

        try:
            chunks = self._split_documents(documents)

            if self.use_remote_client:
                self._create_remote_vectorstore(chunks)
            else:
                self._create_local_vectorstore(chunks)

            self.logger.info("Vector store created successfully!")
            print("Vector store created successfully!")

        except Exception as e:
            self.logger.error(f"Error creating vector store: {str(e)}")
            raise

    def _create_local_vectorstore(self, chunks: List[Document]) -> None:
        """Create local file-based vector store."""
        self.logger.debug(
            f"Creating local Chroma vector store in: {self.persist_directory}"
        )
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def _create_remote_vectorstore(self, chunks: List[Document]) -> None:
        """Create remote HTTP client-based vector store."""
        if not self.chroma_host or not self.chroma_port:
            raise ValueError("Remote ChromaDB host and port must be specified")

        self.logger.debug(
            f"Creating remote Chroma vector store at: {self.chroma_host}:{self.chroma_port}"
        )

        client = chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)  # type: ignore

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client=client,
            collection_name=self.collection_name,
        )

    def load_existing_vectorstore(self) -> bool:
        """Load existing vector store if it exists."""
        try:
            if self.use_remote_client:
                return self._load_remote_vectorstore()
            else:
                return self._load_local_vectorstore()
        except Exception as e:
            self.logger.error(f"Error loading existing vector store: {str(e)}")
            self.logger.warning("Will proceed with creating new vector store")
            return False

    def _load_local_vectorstore(self) -> bool:
        """Load existing local vector store if it exists."""
        self.logger.debug(
            f"Checking for existing vector store at: {self.persist_directory}"
        )

        if os.path.exists(self.persist_directory):
            self.logger.info(
                f"Found existing vector store, loading from: {self.persist_directory}"
            )
            print("Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
            self.logger.info("Existing vector store loaded successfully")
            return True
        else:
            self.logger.debug("No existing vector store found")
            return False

    def _load_remote_vectorstore(self) -> bool:
        """Load existing remote vector store if it exists."""
        if not self.chroma_host or not self.chroma_port:
            raise ValueError("Remote ChromaDB host and port must be specified")

        self.logger.debug(
            f"Checking for existing remote vector store at: {self.chroma_host}:{self.chroma_port}"
        )

        try:
            client = chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)  # type: ignore

            try:
                collections = client.list_collections()
                collection_exists = any(
                    col.name == self.collection_name for col in collections
                )

                if collection_exists:
                    self.logger.info(
                        f"Found existing remote collection: {self.collection_name}"
                    )
                    print("Loading existing remote vector store...")
                    self.vectorstore = Chroma(
                        client=client,
                        collection_name=self.collection_name,
                        embedding_function=self.embeddings,
                    )
                    self.logger.info("Existing remote vector store loaded successfully")
                    return True
                else:
                    self.logger.debug(
                        f"Collection '{self.collection_name}' not found on remote server"
                    )
                    return False

            except Exception as e:
                self.logger.debug(f"Error checking remote collections: {str(e)}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to connect to remote ChromaDB: {str(e)}")
            raise

    def get_vectorstore(self) -> Optional[Chroma]:
        """Get the current vector store instance."""
        return self.vectorstore

    def get_connection_info(self) -> dict:
        """Get information about the current ChromaDB connection."""
        if self.use_remote_client:
            return {
                "type": "remote",
                "host": self.chroma_host,
                "port": self.chroma_port,
                "collection": self.collection_name,
                "url": f"http://{self.chroma_host}:{self.chroma_port}",
            }
        else:
            return {
                "type": "local",
                "persist_directory": self.persist_directory,
                "collection": "default",
            }

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for better retrieval."""
        self.logger.debug("Initializing text splitter")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("CHUNK_SIZE", 1000),
            chunk_overlap=self.config.get("CHUNK_OVERLAP", 200),
            separators=["\n\n", "\n", " ", ""],
        )

        self.logger.debug(f"Splitting {len(documents)} documents into chunks")
        chunks = text_splitter.split_documents(documents)
        self.logger.info(f"Created {len(chunks)} document chunks")
        print(f"Created {len(chunks)} document chunks")

        return chunks
