from pathlib import Path
from typing import Optional, Any, Dict
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

from .logger import setup_agent_logger, log_verbose
from .document_processor import ObsidianDocumentProcessor
from .vector_store_manager import VectorStoreManager
from .config import get_config
from .tools import get_tools_for_agent
from .prompts import REACT_AGENT_PROMPT_TEMPLATE

# Module-level constant for function call keywords
FUNCTION_CALL_KEYWORDS = [
    "list documents",
    "list files",
    "what documents",
    "what files",
    "documents available",
    "files available",
    "show me files",
    "show me documents",
    "search for documents",
    "search for files",
    "find documents",
    "find files",
    "tell me about",
    "info about",
    "information about",
    "documents in",
    "files in",
    "what's in",
    "available in",
    "document list",
    "file list",
    "inventory",
    "catalog",
    "vault info",
]


class ObsidianAgent:
    """Main agent that orchestrates document processing, vector store management, and querying."""

    def __init__(
        self,
        obsidian_vault_path: Path,
        model_name: str,
        embedding_model: str,
        persist_directory: str,
        log_file: str,
        chroma_host: str,
        chroma_port: int,
        collection_name: str,
        verbose: bool,
        quiet: bool,
    ):
        self.log_file = log_file

        self.obsidian_vault_path = Path(obsidian_vault_path)
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.collection_name = collection_name
        self.verbose = verbose
        self.quiet = quiet
        self.log_verbose = log_verbose

        self.logger = setup_agent_logger(self.log_file, str(id(self)), verbose, quiet)

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
            self.obsidian_vault_path, self.logger, self.llm
        )
        self.vector_store_manager = VectorStoreManager(
            embeddings=self.embeddings,
            persist_directory=self.persist_directory,
            logger=self.logger,
            chroma_host=self.chroma_host,
            chroma_port=self.chroma_port,
            collection_name=self.collection_name,
        )

        self.tools = get_tools_for_agent(str(self.obsidian_vault_path))
        self.logger.info(f"Initialized {len(self.tools)} tools for function calling")

        self.qa_chain: Optional[Any] = None
        self.agent_executor: Optional[Any] = None

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
            print(f"Using remote ChromaDB at {connection_info['url']}")
        else:
            print(f"Using local ChromaDB at {connection_info['persist_directory']}")

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
            self._setup_function_calling_agent()

            self.logger.info("Agent initialization completed successfully")
            print("✓ Agent initialization completed!")

        except Exception as e:
            self.logger.error(f"Error during agent initialization: {str(e)}")
            raise

    def _rag_answer(self, question: str) -> Dict[str, Any]:
        self.log_verbose(self.logger, f"Query length: {len(question)} characters")
        print("Searching knowledge base...")
        self.log_verbose(self.logger, "Starting semantic search and retrieval process")
        streamed_answer = ""
        result = None
        try:
            for chunk in self.qa_chain.stream({"input": question}):
                if "answer" in chunk:
                    print(chunk["answer"], end="", flush=True)
                    streamed_answer += chunk["answer"]
                result = chunk
            print()
        except Exception as e:
            self.logger.error(f"Error during streaming: {str(e)}")
            raise
        if result is None or "context" not in result:
            self.logger.error("No context returned from streaming. Skipping sources.")
            return {
                "answer": streamed_answer,
                "sources": [],
                "source_details": {
                    "original_sources": [],
                    "summary_sources": [],
                    "total_chunks": 0,
                    "used_function_calls": False,
                    "tools_used": [],
                },
            }
        all_sources = []
        summary_sources = []
        original_sources = []
        for doc in result["context"]:
            source_info = {
                "source": doc.metadata["source"],
                "is_summary": doc.metadata.get("is_summary", False),
                "content_type": doc.metadata.get("content_type", "original"),
            }
            all_sources.append(source_info)
            if doc.metadata.get("is_summary", False):
                summary_sources.append(doc.metadata["source"])
            else:
                base_source = doc.metadata["source"]
                original_sources.append(base_source)
        unique_original_sources = list(dict.fromkeys(original_sources))
        unique_summary_sources = list(dict.fromkeys(summary_sources))
        all_unique_sources = list(
            dict.fromkeys([info["source"] for info in all_sources])
        )
        self.log_verbose(
            self.logger, f"Retrieved {len(result.get('context', []))} document chunks"
        )
        self.logger.info(
            f"Query processed successfully. Found {len(all_sources)} chunks from {len(all_unique_sources)} sources"
        )
        if unique_summary_sources:
            self.logger.info(
                f"Retrieved {len(unique_summary_sources)} summary documents"
            )
        if unique_original_sources:
            self.logger.info(
                f"Retrieved {len(unique_original_sources)} original documents"
            )
        response = {
            "answer": streamed_answer,
            "sources": all_unique_sources,
            "source_details": {
                "original_sources": unique_original_sources,
                "summary_sources": unique_summary_sources,
                "total_chunks": len(all_sources),
                "used_function_calls": False,
                "tools_used": [],
            },
        }
        return response

    def query(self, question: str) -> Dict[str, Any]:
        if self.qa_chain is None:
            self.logger.error("QA chain not initialized")
            raise ValueError("QA chain not initialized")
        self.logger.info(f"Received query: {question}")
        print(f"\nProcessing question: {question}")
        print("Analyzing query for function calls...")
        from modules.query_chain import (
            ReActHandler,
            ManualFunctionCallHandler,
            RAGHandler,
        )

        react = ReActHandler(self)
        manual = ManualFunctionCallHandler(self)
        rag = RAGHandler(self)
        react.set_next(manual).set_next(rag)
        try:
            result = react.handle(question)
            if result is not None:
                return result
            raise Exception("No handler could answer the query.")
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
            config = get_config()
            retrieval_k = config.retrieval_k

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

    def _setup_function_calling_agent(self) -> None:
        """Set up the function calling agent."""
        self.logger.info("Setting up function calling agent")

        try:

            react_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        REACT_AGENT_PROMPT_TEMPLATE,
                    ),
                ]
            )

            self.logger.debug("Creating ReAct agent with tools")
            agent = create_react_agent(self.llm, self.tools, react_prompt)

            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=self.verbose,
                handle_parsing_errors=True,
                max_iterations=3,
                return_intermediate_steps=True,
            )

            self.logger.info("Function calling agent setup completed successfully")
            print("Function calling agent setup complete!")

        except Exception as e:
            self.logger.error(f"Error setting up function calling agent: {str(e)}")
            self.logger.warning(
                "Function calling disabled, using regular QA chain only"
            )
            self.agent_executor = None

    def _should_use_function_calls(self, question: str) -> bool:
        """Check if the question should trigger function calls based on keywords."""
        question_lower = question.lower()

        self.logger.info(f"Question to check: '{question_lower}'")
        for keyword in FUNCTION_CALL_KEYWORDS:
            self.logger.info(f"Checking keyword: '{keyword}' in '{question_lower}'")
            if keyword in question_lower:
                self.logger.info(f"Function call keyword detected: '{keyword}'")
                return True

        self.logger.info("No function call keywords detected")
        return False

    def _handle_manual_function_call(self, question: str) -> Dict[str, Any]:
        """Handle function calls manually when the agent doesn't work."""
        question_lower = question.lower()

        try:
            if any(
                keyword in question_lower
                for keyword in [
                    "list documents",
                    "list files",
                    "what documents",
                    "documents available",
                    "what files",
                    "files available",
                    "show me files",
                    "show me documents",
                    "document list",
                    "file list",
                ]
            ):
                self.logger.info("Manually calling list_obsidian_documents")
                print("Using function call: list_obsidian_documents")

                from .tools import list_obsidian_documents

                result = list_obsidian_documents.invoke(
                    {"vault_path": str(self.obsidian_vault_path)}
                )

                return {
                    "answer": f"Here are the available documents in your Obsidian vault:\n\n{result}",
                    "sources": ["Function calls"],
                    "source_details": {
                        "original_sources": [],
                        "summary_sources": [],
                        "total_chunks": 0,
                        "used_function_calls": True,
                        "tools_used": ["list_obsidian_documents"],
                    },
                }

            elif any(
                keyword in question_lower
                for keyword in [
                    "search for",
                    "find documents",
                    "find files",
                    "search documents",
                ]
            ):
                self.logger.info("Manually calling search_documents_by_name")
                print("Using function call: search_documents_by_name")

                search_term = "python"
                if "about" in question_lower:
                    parts = question_lower.split("about")
                    if len(parts) > 1:
                        search_term = parts[1].strip().split()[0]

                from .tools import search_documents_by_name

                result = search_documents_by_name.invoke(
                    {
                        "vault_path": str(self.obsidian_vault_path),
                        "search_term": search_term,
                    }
                )

                return {
                    "answer": f"Here are the search results for '{search_term}':\n\n{result}",
                    "sources": ["Function calls"],
                    "source_details": {
                        "original_sources": [],
                        "summary_sources": [],
                        "total_chunks": 0,
                        "used_function_calls": True,
                        "tools_used": ["search_documents_by_name"],
                    },
                }

            else:
                self.logger.info(
                    "Manual function call fallback to list_obsidian_documents"
                )
                print("Using function call: list_obsidian_documents")

                from .tools import list_obsidian_documents

                result = list_obsidian_documents.invoke(
                    {"vault_path": str(self.obsidian_vault_path)}
                )

                return {
                    "answer": f"Here are the available documents in your Obsidian vault:\n\n{result}",
                    "sources": ["Function calls"],
                    "source_details": {
                        "original_sources": [],
                        "summary_sources": [],
                        "total_chunks": 0,
                        "used_function_calls": True,
                        "tools_used": ["list_obsidian_documents"],
                    },
                }

        except Exception as e:
            self.logger.error(f"Manual function call failed: {str(e)}")
            return {
                "answer": "",
                "sources": [],
                "source_details": {
                    "original_sources": [],
                    "summary_sources": [],
                    "total_chunks": 0,
                    "used_function_calls": False,
                    "tools_used": [],
                },
            }

    def get_document_summaries(self) -> dict:
        """Retrieve all document summaries from the vector store."""
        try:
            vectorstore = self.vector_store_manager.get_vectorstore()
            if not vectorstore:
                return {}

            all_docs = vectorstore.get()
            summaries = {}

            for i, metadata in enumerate(all_docs.get("metadatas", [])):
                if metadata and metadata.get("is_summary", False):
                    original_source = metadata.get("original_source", f"document_{i}")
                    summary_text = (
                        all_docs.get("documents", [])[i]
                        if all_docs.get("documents")
                        else ""
                    )

                    summaries[original_source] = {
                        "summary": summary_text,
                        "word_count": metadata.get("original_word_count", "unknown"),
                        "summary_word_count": metadata.get("word_count", "unknown"),
                        "summary_model": metadata.get("summary_model", "unknown"),
                    }

            self.logger.info(f"Retrieved {len(summaries)} document summaries")
            return summaries

        except Exception as e:
            self.logger.error(f"Error retrieving document summaries: {str(e)}")
            return {}

    def get_summary_for_document(self, document_path: str) -> Optional[str]:
        """Get the summary for a specific document."""
        summaries = self.get_document_summaries()
        return summaries.get(document_path, {}).get("summary")

    def get_summarized_documents_stats(self) -> dict:
        """Get statistics about summarized documents."""
        summaries = self.get_document_summaries()

        if not summaries:
            return {"total_summaries": 0, "avg_word_count": 0, "documents": []}

        word_counts = []
        for info in summaries.values():
            wc = info.get("word_count", 0)
            if isinstance(wc, int):
                word_counts.append(wc)
            elif isinstance(wc, str) and wc.isdigit():
                word_counts.append(int(wc))

        return {
            "total_summaries": len(summaries),
            "avg_word_count": (
                sum(word_counts) // len(word_counts) if word_counts else 0
            ),
            "documents": list(summaries.keys()),
        }


class ObsidianAgentBuilder:
    def __init__(self):
        self._params = {}

    def obsidian_vault_path(self, path):
        self._params["obsidian_vault_path"] = path
        return self

    def model_name(self, model_name):
        self._params["model_name"] = model_name
        return self

    def embedding_model(self, embedding_model):
        self._params["embedding_model"] = embedding_model
        return self

    def persist_directory(self, persist_directory):
        self._params["persist_directory"] = persist_directory
        return self

    def log_file(self, log_file):
        self._params["log_file"] = log_file
        return self

    def chroma_host(self, chroma_host):
        self._params["chroma_host"] = chroma_host
        return self

    def chroma_port(self, chroma_port):
        self._params["chroma_port"] = chroma_port
        return self

    def collection_name(self, collection_name):
        self._params["collection_name"] = collection_name
        return self

    def verbose(self, verbose: bool):
        self._params["verbose"] = verbose
        return self

    def quiet(self, quiet: bool):
        self._params["quiet"] = quiet
        return self

    def from_config(self, config):
        self.model_name(config.model_name)
        self.embedding_model(config.embedding_model)
        self.persist_directory(config.persist_directory)
        self.collection_name(config.collection_name)
        self.log_file(config.logs_file)
        return self

    def build(self):
        return ObsidianAgent(**self._params)
