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

### Local Storage
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
3. Optionally generate summaries for long documents (if enabled)
4. Start an interactive Q&A session

### Document Summarization

The agent can automatically generate concise summaries for long documents during indexing:

```bash
# Enable summarization with default settings (500+ words)
python main.py -v "/path/to/vault" --enable-summarization

# Enable summarization with custom word threshold
python main.py -v "/path/to/vault" --enable-summarization --summarization-min-words 300

# Use with remote ChromaDB and summarization
python main.py -v "/path/to/vault" --chroma-host localhost --enable-summarization --verbose
```

### Enhanced Markdown Processing

The agent uses LangChain's `UnstructuredMarkdownLoader` for superior markdown parsing:

```bash
# Use elements mode for structured parsing (splits by headings, paragraphs, etc.)
python main.py -v "/path/to/vault" --markdown-mode elements

# Use high-resolution strategy for better quality parsing
python main.py -v "/path/to/vault" --markdown-strategy hi_res

# Combine enhanced parsing with summarization
python main.py -v "/path/to/vault" --enable-summarization --markdown-mode elements --markdown-strategy hi_res
```

**Markdown Processing Modes:**
- **single**: Treats each file as one document (preserves overall structure)
- **elements**: Splits documents by structure (headings, paragraphs, lists, etc.)

**Markdown Processing Strategies:**
- **auto**: Automatically choose best strategy (default)
- **hi_res**: Higher quality parsing with more computational cost
- **fast**: Faster parsing with good quality

**Benefits of Enhanced Markdown Processing:**
- **Better structure preservation**: Maintains markdown hierarchy and elements
- **Improved metadata**: Enhanced file information and element categorization
- **Flexible parsing**: Choose between document-level or element-level processing
- **Obsidian compatibility**: Better handling of Obsidian-specific markdown features

**Summarization Features:**
- **Automatic detection**: Only documents above the word threshold are summarized
- **Intelligent summaries**: AI-generated summaries focus on key concepts and main ideas
- **Vectorized summaries**: Summaries are embedded and stored in vector database for semantic search
- **Hybrid retrieval**: Search results can include both original content and summary content
- **Special commands**: Use `summaries` and `stats` commands in the interactive session
- **Enhanced search results**: Clear distinction between original and summary content in results
- **Searchable summaries**: Ask questions that might be better answered by summary content

**Environment Variables for Enhanced Processing:**
- `MARKDOWN_MODE`: Processing mode (single/elements, default: single)
- `MARKDOWN_STRATEGY`: Processing strategy (auto/hi_res/fast, default: auto)
- `SUMMARIZATION_ENABLED`: Enable/disable summarization (true/false)
- `SUMMARIZATION_MIN_WORDS`: Minimum words for summarization (default: 500)
- `SUMMARIZATION_MAX_LENGTH`: Maximum words in generated summary (default: 200)

### Function Calling Capabilities

The agent now supports intelligent function calling, allowing it to perform vault management tasks beyond just semantic search:

**Available Functions:**
- **📋 List Documents**: `list_obsidian_documents` - Get complete vault inventory with metadata
- **🔍 Search by Name**: `search_documents_by_name` - Find documents by filename or path
- **📄 Document Info**: `get_document_info` - Get detailed information about specific documents

**Example Function Call Queries:**
- "What documents are available?" → Lists all markdown files with metadata
- "Search for documents about python" → Finds files with "python" in the name
- "Tell me about my README.md file" → Shows detailed file information
- "List all documents in the project folder" → Shows files in specific directories
- "What files were modified recently?" → Returns files sorted by modification date

**How Function Calling Works:**
1. **Smart Detection**: The agent analyzes your question to determine if tools are needed
2. **Function Execution**: If appropriate, calls Python functions to get real-time vault data
3. **Fallback**: For content questions, automatically falls back to semantic search
4. **Hybrid Responses**: Can combine function calls with knowledge base search

**Function Call vs Semantic Search:**
- **Function calls** are used for vault structure, file listing, and metadata queries
- **Semantic search** is used for content-based questions about what's inside documents
- The agent automatically chooses the best approach for each query

**Visual Indicators:**
When function calls are used, you'll see:
```
🔧 Function calls used:
  - list_obsidian_documents
📄 Response generated using function calls
```

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
| `--enable-summarization` | Enable document summarization | `False` |
| `--summarization-min-words` | Min words for summarization | `500` |
| `--markdown-mode` | Markdown parsing mode (single/elements) | `single` |
| `--markdown-strategy` | Markdown parsing strategy (auto/hi_res/fast) | `auto` |
| `--verbose` | Enable detailed logging (blue), standard logs are green | `False` |
| `--quiet` | Hide all log messages, show only essential output | `False` |

### Example Queries

**Content-Based Questions (Semantic Search):**
- "What are my notes about machine learning?"
- "Summarize my meeting notes from last week"
- "Tell me about the project ideas I've written down"
- "What are the main themes in my research?" (may retrieve summary content)
- "Give me an overview of my thoughts on productivity" (benefits from summary search)

**Vault Management Questions (Function Calls):**
- "What documents are available in my vault?"
- "Search for files containing 'python' in the name"
- "Tell me about my README.md file"
- "List all my meeting notes"
- "What files were created recently?"

**Special Commands (when summarization is enabled):**
- Type `summaries` to view all document summaries
- Type `stats` to see summarization statistics

**Enhanced Search Results:**
When summarization is enabled, search results clearly show:
- 📄 **Original content**: Chunks from actual document text
- 🎯 **Summary content**: AI-generated summaries when they're most relevant
- 📊 **Retrieval statistics**: Total chunks from both content types

Type `quit` to exit.

## ChromaDB Options

**Local Storage**: File-based storage in `./chroma_db`
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
