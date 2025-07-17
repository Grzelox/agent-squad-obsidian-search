from pathlib import Path
from typing import Dict, Optional, Any
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from .logger import setup_agent_logger, log_verbose
from .document_processor import ObsidianDocumentProcessor
from .vector_store_manager import VectorStoreManager
from .config import get_config


class ObsidianAgent:
    """Main agent that orchestrates document processing, vector store management, and querying."""

    def __init__(
        self,
        obsidian_vault_path: str,
        model_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        persist_directory: Optional[str] = None,
        log_file: str = "obsidian_agent.log",
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None,
        collection_name: Optional[str] = None,
        verbose: bool = False,
        quiet: bool = False,
    ):
        # Load configuration from environment variables
        config = get_config()

        self.obsidian_vault_path = Path(obsidian_vault_path)
        self.model_name = model_name or config.get("MODEL_NAME", "llama3.2")
        self.embedding_model = embedding_model or config.get(
            "EMBEDDING_MODEL", "nomic-embed-text"
        )
        self.persist_directory = persist_directory or config.get(
            "PERSIST_DIRECTORY", "./chroma_db"
        )
        self.log_file = log_file
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.collection_name = collection_name or config.get(
            "COLLECTION_NAME", "obsidian_documents"
        )
        self.verbose = verbose
        self.quiet = quiet

        self.logger = setup_agent_logger(log_file, str(id(self)), verbose, quiet)

        self.logger.info(
            f"Initializing ObsidianAgent with vault: {obsidian_vault_path}"
        )
        self.logger.info(
            f"Using model: {self.model_name}, embedding model: {self.embedding_model}"
        )

        if chroma_host:
            self.logger.info(f"Using remote ChromaDB at {chroma_host}:{chroma_port}")
        else:
            self.logger.info(f"Using local ChromaDB at {self.persist_directory}")

        self.logger.debug("Initializing embeddings and LLM")
        log_verbose(
            self.logger, f"Creating OllamaEmbeddings with model: {self.embedding_model}"
        )
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        log_verbose(self.logger, f"Creating OllamaLLM with model: {self.model_name}")
        self.llm = OllamaLLM(model=self.model_name)

        self.document_processor = ObsidianDocumentProcessor(
            self.obsidian_vault_path, self.logger
        )
        self.vector_store_manager = VectorStoreManager(
            embeddings=self.embeddings,
            persist_directory=self.persist_directory,
            logger=self.logger,
            chroma_host=self.chroma_host,
            chroma_port=self.chroma_port,
            collection_name=self.collection_name,
        )

        self.qa_chain: Optional[Any] = None

    def initialize(self, force_rebuild: bool = False) -> None:
        """Initialize the agent."""
        self.logger.info("=" * 60)
        self.logger.info("STARTING AGENT INITIALIZATION")
        self.logger.info(f"Force rebuild: {force_rebuild}")

        connection_info = self.vector_store_manager.get_connection_info()
        self.logger.info(f"ChromaDB connection: {connection_info}")

        self.logger.info("=" * 60)
        print("Initializing Obsidian AI Agent...")

        if connection_info["type"] == "remote":
            print(f"🔗 Using remote ChromaDB at {connection_info['url']}")
        else:
            print(f"📁 Using local ChromaDB at {connection_info['persist_directory']}")

        try:
            if (
                not force_rebuild
                and self.vector_store_manager.load_existing_vectorstore()
            ):
                self.logger.info("Using existing vector store")
                print("Using existing vector store")
                log_verbose(
                    self.logger, "Loaded existing vector store from persistence layer"
                )
            else:
                self.logger.info("Building new vector store...")
                print("Building new vector store...")
                log_verbose(self.logger, "Starting document loading process")
                documents = self.document_processor.load_documents()
                log_verbose(
                    self.logger, f"Loaded {len(documents)} documents for vectorization"
                )
                self.vector_store_manager.create_vectorstore(documents)

            self._setup_qa_chain()

            self.logger.info("Agent initialization completed successfully")
            print("✓ Agent initialization completed!")

        except Exception as e:
            self.logger.error(f"Error during agent initialization: {str(e)}")
            raise

    def query(self, question: str) -> Dict[str, Any]:
        """Query the knowledge base."""
        self.logger.info(f"Received query: {question}")
        log_verbose(self.logger, f"Query length: {len(question)} characters")

        if not self.qa_chain:
            self.logger.error("QA chain not initialized")
            raise ValueError("QA chain not initialized")

        print(f"\nProcessing question: {question}")
        print("Searching knowledge base...")

        try:
            self.logger.debug("Invoking QA chain")
            log_verbose(self.logger, "Starting semantic search and retrieval process")
            result = self.qa_chain.invoke({"input": question})
            log_verbose(
                self.logger,
                f"Retrieved {len(result.get('context', []))} document chunks",
            )

            all_sources = [doc.metadata["source"] for doc in result["context"]]
            sources = list(dict.fromkeys(all_sources))

            self.logger.info(
                f"Query processed successfully. Found {len(all_sources)} chunks from {len(sources)} unique source documents"
            )
            self.logger.debug(f"All sources: {all_sources}")
            self.logger.debug(f"Unique sources: {sources}")

            response = {
                "answer": result["answer"],
                "sources": sources,
            }

            self.logger.debug(f"Answer length: {len(result['answer'])} characters")
            return response

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise

    def _setup_qa_chain(self) -> None:
        """Set up the question-answering chain."""
        self.logger.info("Setting up QA chain")

        vectorstore = self.vector_store_manager.get_vectorstore()
        if not vectorstore:
            self.logger.error("Vector store not initialized")
            raise ValueError("Vector store not initialized")

        try:
            # Load config for retrieval parameters
            config = get_config()
            retrieval_k = config.get("RETRIEVAL_K", 5)

            self.logger.debug(
                f"Creating retriever with similarity search (k={retrieval_k})"
            )
            retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": retrieval_k}
            )

            self.logger.debug(f"Creating retrieval chain with model: {self.model_name}")
            system_prompt = (
                "Use the given context to answer the question. "
                "If you don't know the answer, say you don't know. "
                "Use three sentence maximum and keep the answer concise. "
                "Context: {context}"
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            self.qa_chain = create_retrieval_chain(retriever, question_answer_chain)

            self.logger.info("QA chain setup completed successfully")
            print("QA chain setup complete!")

        except Exception as e:
            self.logger.error(f"Error setting up QA chain: {str(e)}")
            raise
