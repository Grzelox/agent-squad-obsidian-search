import os
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    """Application configuration with defaults"""

    model_name: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 3

    persist_directory: str = "./chroma_db"
    collection_name: str = "obsidian_documents"
    logs_file: str = "logs/app.log"

    summarization_enabled: bool = True
    summarization_min_words: int = 500
    summarization_max_length: int = 150

    markdown_mode: Literal["single", "elements"] = "single"
    markdown_strategy: Literal["auto", "hi_res", "fast"] = "auto"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        config = cls()

        if model := os.getenv("MODEL_NAME"):
            config.model_name = model
        if embedding := os.getenv("EMBEDDING_MODEL"):
            config.embedding_model = embedding

        if chunk_size := os.getenv("CHUNK_SIZE"):
            config.chunk_size = int(chunk_size)
        if chunk_overlap := os.getenv("CHUNK_OVERLAP"):
            config.chunk_overlap = int(chunk_overlap)
        if retrieval_k := os.getenv("RETRIEVAL_K"):
            config.retrieval_k = int(retrieval_k)

        if persist_dir := os.getenv("PERSIST_DIRECTORY"):
            config.persist_directory = persist_dir
        if collection := os.getenv("COLLECTION_NAME"):
            config.collection_name = collection
        if logs_file := os.getenv("LOGS_FILE"):
            config.logs_file = logs_file

        if summ_enabled := os.getenv("SUMMARIZATION_ENABLED"):
            config.summarization_enabled = summ_enabled.lower() in [
                "true",
                "1",
                "yes",
                "on",
            ]
        if summ_min_words := os.getenv("SUMMARIZATION_MIN_WORDS"):
            config.summarization_min_words = int(summ_min_words)
        if summ_max_length := os.getenv("SUMMARIZATION_MAX_LENGTH"):
            config.summarization_max_length = int(summ_max_length)

        if markdown_mode := os.getenv("MARKDOWN_MODE"):
            if markdown_mode in ["single", "elements"]:
                config.markdown_mode = markdown_mode  # type: ignore
        if markdown_strategy := os.getenv("MARKDOWN_STRATEGY"):
            if markdown_strategy in ["auto", "hi_res", "fast"]:
                config.markdown_strategy = markdown_strategy  # type: ignore

        return config

    def update_from_cli(self, **kwargs) -> None:
        """Update configuration with CLI arguments."""
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

    def get(self, key: str, default=None):
        """Get configuration value with fallback (for backward compatibility)."""
        return getattr(self, key.lower(), default)


_app_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _app_config
    if _app_config is None:
        _app_config = AppConfig.from_env()
    return _app_config  # type: ignore
