import os
from dotenv import load_dotenv

load_dotenv()

CONFIG_KEYS = [
    "MODEL_NAME",
    "EMBEDDING_MODEL", 
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RETRIEVAL_K",
    "PERSIST_DIRECTORY",
    "COLLECTION_NAME",
    "LOGS_FILE",
    "SUMMARIZATION_ENABLED",
    "SUMMARIZATION_MIN_WORDS",
    "SUMMARIZATION_MAX_LENGTH",
]


def get_config():
    config = {}
    for key in CONFIG_KEYS:
        value = os.getenv(key)
        if value is not None and value != "":
            if key in ["CHUNK_SIZE", "CHUNK_OVERLAP", "RETRIEVAL_K", "SUMMARIZATION_MIN_WORDS", "SUMMARIZATION_MAX_LENGTH"]:
                config[key] = int(value)
            elif key == "SUMMARIZATION_ENABLED":
                config[key] = value.lower() in ["true", "1", "yes", "on"]
            else:
                config[key] = value
        elif value == "":
            if key not in ["CHUNK_SIZE", "CHUNK_OVERLAP", "RETRIEVAL_K", "SUMMARIZATION_MIN_WORDS", "SUMMARIZATION_MAX_LENGTH", "SUMMARIZATION_ENABLED"]:
                config[key] = value
    return config
