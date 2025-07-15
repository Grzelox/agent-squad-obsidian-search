import os

# Default configuration
DEFAULT_CONFIG = {
    "MODEL_NAME": "llama3.2",
    "EMBEDDING_MODEL": "nomic-embed-text",
    "CHUNK_SIZE": 1000,
    "CHUNK_OVERLAP": 200,
    "RETRIEVAL_K": 5,
    "PERSIST_DIRECTORY": "./chroma_db",
}


def get_config():
    """Get configuration from environment variables or defaults."""
    config = {}
    for key, default_value in DEFAULT_CONFIG.items():
        config[key] = os.getenv(key, default_value)
    return config
