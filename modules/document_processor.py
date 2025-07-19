import re
from pathlib import Path
from typing import List, Optional
import logging
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.schema import Document
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from .config import get_config


class ObsidianDocumentProcessor:
    """Handles loading and processing of Obsidian markdown documents with enhanced markdown parsing and optional summarization."""

    def __init__(
        self,
        vault_path: Path,
        logger: logging.Logger,
        llm: Optional[OllamaLLM] = None,
    ):
        self.vault_path = vault_path
        self.logger = logger
        self.llm = llm

        self.config = get_config()

        if self.config.summarization_enabled and self.llm:
            self.logger.info(
                f"Document summarization enabled (min words: {self.config.summarization_min_words})"
            )
        elif self.config.summarization_enabled and not self.llm:
            self.logger.warning(
                "Summarization enabled but no LLM provided - summaries will be skipped"
            )

        self.logger.info(
            f"Markdown processing mode: {self.config.markdown_mode}, strategy: {self.config.markdown_strategy}"
        )

    def load_documents(self) -> List[Document]:
        """Load and process Obsidian markdown files with enhanced markdown parsing and optional summarization."""
        self.logger.info(
            f"Starting enhanced markdown document loading from: {self.vault_path}"
        )
        print(
            f"Loading documents from {self.vault_path} with enhanced markdown parsing"
        )

        try:
            md_files = list(self.vault_path.glob("**/*.md"))
            self.logger.info(f"Found {len(md_files)} markdown files")
            print(f"Found {len(md_files)} markdown files")

            if not md_files:
                self.logger.warning("No markdown files found in vault")
                return []

            raw_documents = []
            for i, md_file in enumerate(md_files):
                self.logger.debug(
                    f"Loading {i+1}/{len(md_files)}: {md_file.relative_to(self.vault_path)}"
                )
                try:
                    docs = self._load_single_markdown_file(md_file)
                    raw_documents.extend(docs)
                except Exception as e:
                    self.logger.error(f"Error loading {md_file}: {str(e)}")
                    # Continue with other files

            self.logger.info(
                f"Successfully loaded {len(raw_documents)} document chunks from markdown files"
            )
            print(f"Loaded {len(raw_documents)} document chunks")

            self.logger.debug("Starting Obsidian-specific content processing")
            processed_docs = []
            summary_docs = []
            summarized_count = 0

            for i, doc in enumerate(raw_documents):
                self.logger.debug(
                    f"Processing document {i+1}/{len(raw_documents)}: {doc.metadata.get('source', 'unknown')}"
                )

                processed_doc = self._process_document_content(doc)

                # Add summarization if enabled and LLM is available
                if self.config.summarization_enabled and self.llm:
                    summary = self._generate_summary_if_needed(processed_doc)
                    if summary:
                        processed_doc.metadata["summary"] = summary
                        processed_doc.metadata["word_count"] = self._count_words(
                            processed_doc.page_content
                        )
                        processed_doc.metadata["has_summary"] = True
                        summarized_count += 1
                        self.logger.debug(
                            f"Generated summary for: {processed_doc.metadata.get('source', 'unknown')}"
                        )

                        # Create a separate document for the summary to be vectorized
                        summary_doc = self._create_summary_document(
                            processed_doc, summary
                        )
                        summary_docs.append(summary_doc)
                    else:
                        processed_doc.metadata["has_summary"] = False
                else:
                    processed_doc.metadata["has_summary"] = False

                processed_docs.append(processed_doc)

            all_documents = processed_docs + summary_docs

            self.logger.info(
                f"Document processing completed. Processed {len(processed_docs)} documents"
            )
            if summarized_count > 0:
                self.logger.info(
                    f"Generated summaries for {summarized_count} long documents"
                )
                self.logger.info(
                    f"Created {len(summary_docs)} summary documents for vectorization"
                )
                print(f"Generated summaries for {summarized_count} long documents")
                print(
                    f"📝 Created {len(summary_docs)} summary documents for semantic search"
                )

            return all_documents

        except Exception as e:
            self.logger.error(f"Error loading documents: {str(e)}")
            raise

    def _load_single_markdown_file(self, file_path: Path) -> List[Document]:
        """Load a single markdown file using UnstructuredMarkdownLoader."""
        try:
            loader = UnstructuredMarkdownLoader(
                str(file_path), mode=self.config.markdown_mode, strategy=self.config.markdown_strategy
            )

            documents = loader.load()

            for doc in documents:
                doc.metadata["source"] = str(file_path.relative_to(self.vault_path))
                doc.metadata["file_path"] = str(file_path)
                doc.metadata["markdown_mode"] = self.config.markdown_mode
                doc.metadata["markdown_strategy"] = self.config.markdown_strategy

                doc.metadata["file_size"] = file_path.stat().st_size
                doc.metadata["file_modified"] = file_path.stat().st_mtime

                if self.config.markdown_mode == "elements":
                    pass

            return documents

        except Exception as e:
            self.logger.error(f"Error loading markdown file {file_path}: {str(e)}")
            raise

    def _process_document_content(self, doc: Document) -> Document:
        """Process a single document's content for Obsidian-specific syntax."""
        content = re.sub(
            r"\[\[([^\]]+)\]\]", r"\1", doc.page_content
        )  # Clean up Obsidian links [[link]] -> link
        content = re.sub(
            r"#([a-zA-Z0-9_-]+)", r"\1", content
        )  # Clean up tags #tag -> tag
        content = re.sub(r"\n\s*\n", "\n\n", content)  # Clean up excessive whitespace

        # Update document content and metadata
        doc.page_content = content

        # Only make path relative if it's not already relative
        source_path = doc.metadata.get("source", "")
        if source_path and Path(source_path).is_absolute():
            try:
                doc.metadata["source"] = str(
                    Path(source_path).relative_to(self.vault_path)
                )
            except ValueError:
                # If we can't make it relative, leave it as is
                pass

        return doc

    def _count_words(self, text: str) -> int:
        """Count words in the given text."""
        clean_text = re.sub(r"[#*`\-=]", "", text)
        words = clean_text.split()
        return len(words)

    def _generate_summary_if_needed(self, doc: Document) -> Optional[str]:
        """Generate a summary for the document if it meets the word count threshold."""
        word_count = self._count_words(doc.page_content)

        if word_count < self.config.summarization_min_words:
            self.logger.debug(
                f"Document too short for summary ({word_count} words < {self.config.summarization_min_words})"
            )
            return None

        try:
            self.logger.debug(
                f"Generating summary for document with {word_count} words"
            )
            return self._generate_summary(
                doc.page_content, doc.metadata.get("source", "unknown")
            )
        except Exception as e:
            self.logger.error(
                f"Error generating summary for {doc.metadata.get('source', 'unknown')}: {str(e)}"
            )
            return None

    def _generate_summary(self, content: str, source: str) -> str:
        """Generate a concise summary of the document content using LLM."""
        if not self.llm:
            raise ValueError("LLM not available for summarization")

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at summarizing documents. Create a concise, informative summary "
                    f"of the following text in no more than {self.config.summarization_max_length} words. "
                    "Focus on the main ideas, key points, and important concepts. "
                    "Write in a clear, readable style that captures the essence of the content.",
                ),
                ("human", "Please summarize this document:\n\n{content}"),
            ]
        )

        chain = prompt_template | self.llm
        result = chain.invoke({"content": content})

        # Clean up the summary
        summary = result.strip()

        # Ensure it's within word limit
        words = summary.split()
        if len(words) > self.config.summarization_max_length:
            summary = " ".join(words[: self.config.summarization_max_length]) + "..."

        return summary

    def _create_summary_document(
        self, original_doc: Document, summary: str
    ) -> Document:
        """Create a separate Document object for the summary to be vectorized."""
        summary_doc = Document(
            page_content=summary,
            metadata={
                # Mark this as a summary document
                "content_type": "summary",
                "original_source": original_doc.metadata.get("source", "unknown"),
                "source": f"{original_doc.metadata.get('source', 'unknown')} (summary)",
                "is_summary": True,
                "word_count": len(summary.split()),
                "original_word_count": original_doc.metadata.get(
                    "word_count", "unknown"
                ),
                "summary_method": "llm_generated",
                "summary_model": (
                    getattr(self.llm, "model", "unknown") if self.llm else "unknown"
                ),
            },
        )

        self.logger.debug(
            f"Created summary document for: {original_doc.metadata.get('source', 'unknown')}"
        )
        return summary_doc
