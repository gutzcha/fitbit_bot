# Fitbit Bot – Conversational Health Assistant

## Overview

This project implements a conversational AI assistant that processes and reasons over Fitbit-style health data. The system is built around a **graph-based orchestration layer** using LangGraph that routes user requests through intent detection, clarification, and data-processing flows.

The assistant can:
- Answer questions about health metrics (steps, heart rate, calories, etc.)
- Provide health coaching and suggestions based on user data
- Access a knowledge base of health information via RAG
- Handle complex queries requiring SQL queries and data analysis
- Support multiple LLM providers (Ollama, Anthropic)

---

## Architecture

The system uses a **multi-stage graph architecture**:

1. **Intent Detection** - Classifies user queries and determines confidence
2. **Clarification** - Requests additional information when needed
3. **Planning** - Creates execution plans for data queries
4. **Execution** - Executes SQL queries and retrieves health knowledge
5. **Suggestion** - Adds coaching suggestions based on user profile and data
6. **Static Responses** - Handles greetings and out-of-scope queries

### Key Components

- **LangGraph**: Orchestrates the conversation flow
- **LangChain**: LLM integration and tool calling
- **Chroma**: Vector database for health knowledge base (RAG)
- **SQLite**: Stores Fitbit health data
- **Streamlit**: Web UI for interacting with the assistant
- **DuckDB**: SQL query engine for data analysis

---

## Project Structure

```
fitbit_bot/
├── app/                      # Streamlit application
│   ├── app.py               # Main UI entry point
│   └── chat_config.json     # Runtime configuration
├── graph/                   # Core graph architecture
│   ├── graph.py             # Main graph builder
│   ├── state.py             # State schema definitions
│   ├── nodes/               # Graph nodes (intent, clarification, static)
│   ├── chains/              # LangChain chains
│   ├── process/             # Data processing subgraph
│   │   ├── agents/          # Planner, Executor, Suggestor agents
│   │   ├── tools/           # SQL tools and RAG retriever
│   │   └── rag_retrievaer/  # RAG graph for knowledge base
│   ├── tools/               # Tool definitions
│   └── prompts/             # Prompt templates
├── dataset/                 # Health data and databases
│   ├── clean/               # Processed CSV files
│   ├── db/                  # SQLite database and vector store
│   ├── health_data/         # Health knowledge base source files
│   └── user_profiles/       # User profile JSON files
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                   # Unit and integration tests
├── main.py                  # CLI entry point
├── pyproject.toml           # Project dependencies (uv)
└── README.md                # This file
```

---

## Prerequisites

- **Python 3.11+**
- **uv** - Fast Python package installer (install from [astral.sh/uv](https://astral.sh/uv))
- **Ollama** (optional, for local LLM) - Install from [ollama.ai](https://ollama.ai)
  - Required models: `mistral:8b` (or similar) and `mxbai-embed-large:335m` for embeddings

---

## Setup Instructions

### 1. Install uv

If you haven't installed `uv` yet:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Navigate to Project

```bash
cd fitbit_bot
```

### 3. Create Virtual Environment and Install Dependencies

```bash
# Create virtual environment and install all dependencies
uv venv

# Activate virtual environment
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# Install project dependencies
uv pip install -e .
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# LLM Provider Configuration
# Options: "ollama" or "anthropic"
PROVIDER=ollama

# Anthropic API (if using Anthropic provider)
ANTHROPIC_API_KEY=your_api_key_here

# LangSmith (optional, for tracing)
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=fitbit-bot
```

### 5. Set Up Ollama (if using local LLM)

If you're using Ollama as your provider:

```bash
# Pull required models
ollama pull mistral:8b
ollama pull mxbai-embed-large:335m
```

### 6. Initialize Knowledge Base (Optional)

If the knowledge base hasn't been populated:

```bash
python health_kb_loader.py
```

This will load health data from `dataset/health_data/` into the Chroma vector database.

---

## Running the Application

### Streamlit Web UI (Recommended)

```bash
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`.

**Features:**
- Interactive chat interface
- Real-time graph execution visualization
- Model provider switching (Ollama ↔ Anthropic)
- User profile display
- Debug trace viewer



## Configuration

Runtime configuration can be adjusted in `app/chat_config.json` or via the Streamlit sidebar:

- **Provider**: Switch between `ollama` and `anthropic`
- **Slow Fallback**: Enable/disable fallback to larger model for uncertain intents
- **Max History Context**: Number of conversation turns to include in context
- **User ID**: two users defined
---

## Key Features

### 1. Intent Classification
- Automatically detects user intent (data query, greeting, out-of-scope, etc.)
- Uses confidence scoring with optional fallback to larger model

### 2. SQL Query Generation
- Generates optimized SQL queries for Fitbit data
- Pre-defined tools for common queries (steps, heart rate, calories, etc.)
- Custom SQL generation for complex queries

### 3. RAG-Powered Knowledge Base
- Retrieves relevant health information from curated knowledge base
- Uses semantic search with embeddings
- Provides citations and source documents

### 4. Personalized Coaching
- Suggests health improvements based on user profile and data
- Considers user goals, preferences, and historical patterns

### 5. Multi-Provider Support
- **Ollama**: Local, privacy-focused, no API costs
- **Anthropic**: Cloud-based, high-quality responses

---

## Data Sources

The system uses several data sources:

1. **Fitbit SQLite Database** (`dataset/db/fitbit.sqlite`)
   - Daily activity, heart rate, steps, calories, weight logs

2. **User Profiles** (`dataset/user_profiles/*.json`)
   - Health goals, body metrics, coaching preferences

3. **Health Knowledge Base** (`dataset/db/health_kb/`)
   - Vector database of health information
   - Source files in `dataset/health_data/`

---

## Testing

Run the test suite:
(mock tests, not operational)
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_chains.py

# Run with coverage
pytest --cov=graph --cov-report=html
```

---

## Development

### Code Formatting

```bash
# Format code
black .

# Sort imports
isort .
```

### Project Dependencies

Dependencies are managed via `pyproject.toml` and installed with `uv`. Key dependencies include:

- `langchain` & `langgraph` - LLM orchestration
- `langchain-anthropic` & `langchain-ollama` - LLM providers
- `langchain-chroma` - Vector database
- `streamlit` - Web UI
- `duckdb` & `duckdb-engine` - SQL query engine
- `pytest` - Testing framework

### Adding New Tools

1. Define tool in `graph/tools/definitions.py` or `graph/process/tools/`
2. Register tool in the execution agent (`graph/process/agents/execution.py`)
3. Update planner prompts if needed

### Extending the Graph

1. Create new node in `graph/nodes/` or `graph/process/nodes/`
2. Add node to graph in `graph/graph.py` or `graph/process/process_graph.py`
3. Define routing logic in conditional edges

---

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Check model availability: `ollama list`
- Verify base URL in environment (default: `http://localhost:11434`)

### Knowledge Base Not Found
- Run `python health_kb_loader.py` to populate the vector database
- Check that `dataset/db/health_kb/` exists and contains data

### Database Connection Errors
- Verify `dataset/db/fitbit.sqlite` exists
- Check file permissions
- Ensure database was properly initialized

### Import Errors
- Activate virtual environment: `.venv\Scripts\Activate.ps1` (Windows) or `source .venv/bin/activate` (macOS/Linux)
- Reinstall dependencies: `uv pip install -e .`

---

## License

This project was created as a home assignment. See assignment PDF for details.

---

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain)
- Uses [uv](https://github.com/astral-sh/uv) for fast package management
- Health data from Fitabase export format
