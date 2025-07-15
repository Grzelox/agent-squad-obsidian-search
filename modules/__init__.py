"""
Obsidian AI Search Agent - Modular Components

This package contains the core components of the Obsidian AI Search Agent:
- ObsidianAgent: Main orchestrator class
- ObsidianDocumentProcessor: Document loading and processing
- VectorStoreManager: Vector store operations
- Logger utilities: Logging setup and configuration
"""

from .agent import ObsidianAgent
from .document_processor import ObsidianDocumentProcessor
from .vector_store_manager import VectorStoreManager
from .logger import setup_agent_logger, setup_cli_logger

__all__ = [
    "ObsidianAgent",
    "ObsidianDocumentProcessor",
    "VectorStoreManager",
    "setup_agent_logger",
    "setup_cli_logger",
]
