import re
from pathlib import Path
from typing import List
import logging
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document


class ObsidianDocumentProcessor:
    """Handles loading and processing of Obsidian markdown documents."""

    def __init__(self, vault_path: Path, logger: logging.Logger):
        self.vault_path = vault_path
        self.logger = logger

    def load_documents(self) -> List[Document]:
        """Load and process Obsidian markdown files."""
        self.logger.info(f"Starting document loading from: {self.vault_path}")
        print(f"Loading documents from {self.vault_path}")

        try:
            self.logger.debug("Initializing DirectoryLoader for markdown files")
            loader = DirectoryLoader(
                str(self.vault_path),
                glob="**/*.md",
                show_progress=True,
            )

            documents = loader.load()
            self.logger.info(f"Successfully loaded {len(documents)} raw documents")
            print(f"Loaded {len(documents)} documents")

            self.logger.debug("Starting Obsidian-specific content processing")
            processed_docs = []
            for i, doc in enumerate(documents):
                self.logger.debug(
                    f"Processing document {i+1}/{len(documents)}: {doc.metadata.get('source', 'unknown')}"
                )

                processed_doc = self._process_document_content(doc)
                processed_docs.append(processed_doc)

            self.logger.info(
                f"Document processing completed. Processed {len(processed_docs)} documents"
            )
            return processed_docs

        except Exception as e:
            self.logger.error(f"Error loading documents: {str(e)}")
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
        doc.metadata["source"] = str(
            Path(doc.metadata["source"]).relative_to(self.vault_path)
        )

        return doc
