"""
Obsidian AI Search Agent - Modular Components

This package contains the core components of the Obsidian AI Search Agent:
- ObsidianAgent: Main orchestrator class
- ObsidianDocumentProcessor: Document loading and processing
- VectorStoreManager: Vector store operations
- VaultCopyService: Vault copying operations
- Logger utilities: Logging setup and configuration
- Configuration: Environment-based configuration management
"""

from .agent import ObsidianAgent
from .document_processor import ObsidianDocumentProcessor
from .vector_store_manager import VectorStoreManager
from .vault_copy_service import VaultCopyService
from .logger import setup_cli_logger
from .config import get_config
from .tools import AVAILABLE_TOOLS, get_tools_for_agent, format_tool_descriptions

__all__ = [
    "ObsidianAgent",
    "ObsidianDocumentProcessor",
    "VectorStoreManager",
    "VaultCopyService",
    "setup_cli_logger",
    "get_config",
    "AVAILABLE_TOOLS",
    "get_tools_for_agent",
    "format_tool_descriptions",
]
