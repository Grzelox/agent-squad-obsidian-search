# Obsidian AI Search Agent

An intelligent search agent that indexes your Obsidian vault and allows you to query your knowledge base using natural language powered by locally run LLM model.

## Prerequisites

### Required Software
1. **Python >3.13**
2. **Ollama** - Download and install from [ollama.ai](https://ollama.ai)

### Required Ollama Models
Make sure these models are installed in Ollama:

```bash
# Install the default models
ollama pull llama3.2
ollama pull nomic-embed-text

# Optional: Install other models you might want to use
ollama pull llama2
ollama pull codellama
```

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd agent-squad-obsidian-search
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install uv package manager
pip install uv

# Install all dependencies from pyproject.toml
uv sync
```


# Run the agent

### Basic Usage
```bash
python main.py -v "/path/to/your/obsidian/vault"
```

### Command Line Options

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--vault-path` | `-v` | Yes | - | Path to your Obsidian vault directory |
| `--model` | `-m` | No | `llama3.2` | Ollama model to use for Q&A |
| `--embedding-model` | `-e` | No | `nomic-embed-text` | Embedding model for document indexing |
| `--rebuild` | `-r` | No | `False` | Force rebuild of vector store |

### Examples

#### 1. Basic Usage with Default Settings
```bash
python main.py -v "/Users/username/Documents/MyVault"
```

#### 2. Using a Different AI Model
```bash
python main.py -v "/Users/username/Documents/MyVault" -m llama2
```

#### 3. Custom Embedding Model
```bash
python main.py -v "/Users/username/Documents/MyVault" -e all-minilm
```

#### 4. Force Rebuild Vector Store
```bash
python main.py -v "/Users/username/Documents/MyVault" -r
```

#### 5. Full Custom Configuration
```bash
python main.py \
  -v "/Users/username/Documents/MyVault" \
  -m codellama \
  -e nomic-embed-text \
  -r
```

## Interactive Usage

Once the agent starts, you'll see:
```
Obsidian AI Agent ready! Type 'quit' to exit.

Ask a question:
```

Example output format:
```
Answer:
[AI response here]

Sources:
  - filename1.md
  - filename2.md
```

### Example Queries
- "What are my notes about machine learning?"
- "Tell me about the project ideas I've written down"
- "What did I learn about Python yesterday?"
- "Summarize my meeting notes from last week"

### Exit Commands
Type any of these to exit:
- `quit`
- `exit` 
- `q`

## Logging

The application creates comprehensive logs in `logs/obsidian_agent.log` that include:

- **Initialization**: Agent setup and configuration
- **Document Processing**: File loading and indexing progress
- **Vector Operations**: Store creation and retrieval operations
- **Query Processing**: User questions and AI responses
- **Error Tracking**: Detailed error information and stack traces

### Monitor Logs in Real-time
```bash
tail -f logs/obsidian_agent.log
```

## How It Works

1. **Document Loading**: Scans your Obsidian vault for `.md` files
2. **Content Processing**: Cleans up Obsidian-specific syntax (links, tags)
3. **Text Chunking**: Splits documents into searchable chunks
4. **Embedding Generation**: Creates vector embeddings using the embedding model
5. **Vector Storage**: Stores embeddings in ChromaDB for fast retrieval
6. **Query Processing**: Uses semantic search + LLM to answer questions
