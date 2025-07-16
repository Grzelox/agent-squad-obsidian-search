# Obsidian AI Search Agent

An intelligent search agent that copies your Obsidian vault, indexes it using vector embeddings, and allows you to query your knowledge base using natural language with locally run LLM models.

## Prerequisites

1. **Python ≥3.13**
2. **Ollama** - Download from [ollama.ai](https://ollama.ai)
3. **Docker** (optional, ChromaDB and ChromaDB Admin Dashboard)

### Install Required Ollama Models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd agent-squad-obsidian-search

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install uv
uv sync
```

## Quick Start

### Local Storage (Simple)
```bash
python main.py -v "/path/to/your/obsidian/vault"

# With verbose logging
python main.py -v "/path/to/your/obsidian/vault" --verbose

# Quiet mode (no logs)
python main.py -v "/path/to/your/obsidian/vault" --quiet
```

### Docker ChromaDB
```bash
# Start ChromaDB container
docker-compose up -d

# Run agent with remote ChromaDB
python main.py -v "/path/to/your/obsidian/vault" --chroma-host localhost --chroma-port 8000

# Run with verbose logging (shows detailed colored output)
python main.py -v "/path/to/your/obsidian/vault" --chroma-host localhost --chroma-port 8000 --verbose

# Run in quiet mode (no logs, clean output)
python main.py -v "/path/to/your/obsidian/vault" --chroma-host localhost --chroma-port 8000 --quiet
```

### Output Modes

**Quiet Mode (`--quiet`)**
- Clean, minimal output with only essential messages
- No log timestamps or technical details
- Perfect for scripts or when you want minimal noise

**Standard Mode (default)**
- **White**: Regular application messages (copying, questions, answers)  
- **🟢 Green**: Standard log messages (INFO, WARNING, ERROR)

**Verbose Mode (`--verbose`)**
- **White**: Regular application messages (copying, questions, answers)
- **🟢 Green**: Standard log messages (INFO, WARNING, ERROR)
- **🔵 Blue**: Detailed debugging info with `[VERBOSE]` prefix

> **Note**: `--verbose` and `--quiet` cannot be used together

## Usage

The agent will:
1. Copy your vault to a working directory (preserves original)
2. Process and index your documents
3. Start an interactive Q&A session

### Key Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `-v, --vault-path` | Path to your Obsidian vault (required) | - |
| `-d, --destination` | Working directory for vault copy | `./vault` |
| `-m, --model` | Ollama model for Q&A | `llama3.2` |
| `-e, --embedding-model` | Embedding model | `nomic-embed-text` |
| `-r, --rebuild` | Force rebuild vector store | `False` |
| `--chroma-host` | ChromaDB host for remote connection | `None` |
| `--chroma-port` | ChromaDB port | `8000` |
| `--verbose` | Enable detailed logging (blue), standard logs are green | `False` |
| `--quiet` | Hide all log messages, show only essential output | `False` |

### Example Queries
- "What are my notes about machine learning?"
- "Summarize my meeting notes from last week"
- "Tell me about the project ideas I've written down"

Type `quit` to exit.

## ChromaDB Options

**Local Storage**: File-based storage in `./chroma_db`
- Simple setup
- No web interface

**Docker ChromaDB**: Remote server with web interface
- Web admin interface at http://localhost:3001
- Requires Docker

## ChromaDB Admin Interface

When using Docker, the setup includes a ChromaDB admin interface that provides a web-based GUI for managing your vector database.

### Container Architecture

The `docker-compose.yml` includes two containers:

1. **ChromaDB Container** (`chromadb`):
   - Runs ChromaDB server on port 8000
   - Stores vector embeddings and metadata
   - Accessible at `http://localhost:8000`

2. **ChromaDB Admin Container**:
   - Provides web interface for ChromaDB
   - Accessible at `http://localhost:3001`
   - Connects to ChromaDB using internal Docker networking

### Docker Container Communication

**Important**: The admin container connects to ChromaDB using the **service name** as hostname:

```bash
CHROMADB_CONN_STR=http://chromadb:8000 
```

### Setup Instructions

1. **Start the containers**:
   ```bash
   docker-compose up -d
   ```

2. **Verify containers are running**:
   ```bash
   docker-compose ps
   ```

3. **Access the admin interface**:
   - Open your browser to `http://localhost:3001`

4. **Run your agent**:
   ```bash
   python main.py -v "/path/to/vault" --chroma-host localhost --chroma-port 8000
   ```