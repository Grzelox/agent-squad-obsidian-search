# Configuration Guide

The Obsidian AI Search Agent supports configuration via environment variables. Create a `.env` file in the project root to set these values.

## Environment Variables

| Variable | Description | Default | Type |
|----------|-------------|---------|------|
| `MODEL_NAME` | Ollama model for question answering | `llama3.2` | string |
| `EMBEDDING_MODEL` | Ollama embedding model | `nomic-embed-text` | string |
| `CHUNK_SIZE` | Text chunk size for document processing | `1000` | integer |
| `CHUNK_OVERLAP` | Text chunk overlap size | `200` | integer |
| `RETRIEVAL_K` | Number of document chunks to retrieve | `5` | integer |
| `PERSIST_DIRECTORY` | Directory for ChromaDB storage | `./chroma_db` | string |
| `COLLECTION_NAME` | ChromaDB collection name | `obsidian_documents` | string |

## Example .env File

Create a `.env` file in the project root with your desired configuration:

```env
# Model Configuration
MODEL_NAME=llama3.2
EMBEDDING_MODEL=nomic-embed-text

# Document Processing Configuration  
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=5

# Storage Configuration
PERSIST_DIRECTORY=./chroma_db
COLLECTION_NAME=obsidian_documents
```

## Configuration Precedence

1. **CLI Arguments** (highest priority)
2. **Environment Variables** (from `.env` file)
3. **Default Values** (lowest priority)

This means you can set defaults in your `.env` file and override them with CLI arguments when needed.

## Usage Examples

```bash
# Use environment defaults
python main.py -v "/path/to/vault"

# Override specific values via CLI
python main.py -v "/path/to/vault" -m llama2 -e all-minilm --collection-name my-vault

# Environment variables are used for values not specified in CLI
python main.py -v "/path/to/vault" -m custom-model
# (EMBEDDING_MODEL, CHUNK_SIZE, COLLECTION_NAME, etc. will use .env values)
``` 