import os
import logging
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document


class VectorStoreManager:
    """Manages vector store operations including creation, loading, and text chunking."""

    def __init__(
        self,
        embeddings: OllamaEmbeddings,
        persist_directory: str,
        logger: logging.Logger,
    ):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.logger = logger
        self.vectorstore: Optional[Chroma] = None

    def create_vectorstore(self, documents: List[Document]) -> None:
        """Create or load vector store from documents."""
        self.logger.info("Starting vector store creation")
        print("Creating vector store...")

        try:
            # Split documents into chunks
            chunks = self._split_documents(documents)

            # Create vector store
            self.logger.debug(
                f"Creating Chroma vector store in: {self.persist_directory}"
            )
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )

            self.logger.info("Vector store created successfully!")
            print("Vector store created successfully!")

        except Exception as e:
            self.logger.error(f"Error creating vector store: {str(e)}")
            raise

    def load_existing_vectorstore(self) -> bool:
        """Load existing vector store if it exists."""
        self.logger.debug(
            f"Checking for existing vector store at: {self.persist_directory}"
        )

        if os.path.exists(self.persist_directory):
            try:
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
            except Exception as e:
                self.logger.error(f"Error loading existing vector store: {str(e)}")
                self.logger.warning("Will proceed with creating new vector store")
                return False
        else:
            self.logger.debug("No existing vector store found")
            return False

    def get_vectorstore(self) -> Optional[Chroma]:
        """Get the current vector store instance."""
        return self.vectorstore

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for better retrieval."""
        self.logger.debug("Initializing text splitter")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )

        self.logger.debug(f"Splitting {len(documents)} documents into chunks")
        chunks = text_splitter.split_documents(documents)
        self.logger.info(f"Created {len(chunks)} document chunks")
        print(f"Created {len(chunks)} document chunks")

        return chunks
