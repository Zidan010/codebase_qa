# codebase_q&a

> **Agentic Q&A system over the [`psf/requests`](https://github.com/psf/requests) Python HTTP library**
>
> Ask natural language questions about the codebase — architecture, implementation details, API usage, call flows, dependencies — and get structured, cited answers backed by real code evidence.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
  - [CLI Interface](#cli-interface)
  - [Example Queries & Outputs](#example-queries--outputs)
- [Agent Tools](#agent-tools)
- [Chosen Repository](#chosen-repository)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [AI Tool Usage Disclosure](#ai-tool-usage-disclosure)
- [Troubleshooting](#troubleshooting)

---

## Overview

`codebase_q&a` is a **multi-tool agentic Q&A system** built with LangGraph that answers complex questions about the `psf/requests` Python codebase. It goes beyond simple semantic search by reasoning about which tools to use per query, validating retrieved evidence before answering, and exposing its full reasoning trace.

### Key Features

| Feature | Detail |
|---|---|
| **Multi-tool agent** | 7 specialised tools: semantic search, file reading, directory listing, module summarisation, usage finding, dependency mapping, call flow tracing |
| **Intent classifier** | Classifies each query into 7 intent types and routes tool selection accordingly |
| **Retrieval critic** | Validates evidence quality before answering — refuses to hallucinate |
| **Repo context builder** | Builds a high-level repo summary once, reused across all queries |
| **Native multi-turn history** | Last 3 conversation exchanges passed natively to Groq API as message turns — not a flat text dump. Follow-up questions like "What calls that method?" resolve correctly |
| **Reasoning trace** | Every agent decision is logged and displayed |
| **Persistent vector store** | ChromaDB persists to disk — re-indexing skipped on restart |
| **Local embeddings** | `all-MiniLM-L6-v2` downloaded once to `embedding_model/`, never re-downloaded |
| **Scope guard** | Rejects off-topic questions without hallucinating |
| **Structured CLI** | Rich terminal output with colour-coded panels, source citation table, and toggleable trace |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                LangGraph Agent Graph                 │
│                                                      │
│  ┌──────────────────┐                               │
│  │  Repo Context    │  Builds repo summary once,    │
│  │  Builder Node    │  cached across all turns      │
│  └────────┬─────────┘                               │
│           │                                          │
│  ┌────────▼─────────┐                               │
│  │  Intent          │  Classifies query intent,     │
│  │  Classifier Node │  checks scope, resolves       │
│  │                  │  pronouns via history         │
│  └────────┬─────────┘                               │
│           │                                          │
│    ┌──────┴──────┐                                  │
│  out_of_scope  in_scope                             │
│    │              │                                  │
│    ▼    ┌─────────▼────────┐                        │
│ [answer]│  Tool Executor   │◄──────────┐            │
│         │  (LLM selects    │           │            │
│         │   which tools)   │     need_more          │
│         └─────────┬────────┘           │            │
│                   │                    │            │
│         ┌─────────▼────────┐           │            │
│         │ Retrieval Critic │───────────┘            │
│         │ (validates       │                        │
│         │  evidence)       │                        │
│         └─────────┬────────┘                        │
│                   │                                  │
│         ┌─────────▼────────┐                        │
│         │  Answer Node     │  Structured output     │
│         │                  │  with citations        │
│         └──────────────────┘                        │
└─────────────────────────────────────────────────────┘
```

### Multi-Turn Conversation Flow

Conversation history is passed **natively** to the Groq API as a structured messages array — not embedded as text in the prompt. This means the model natively understands follow-up context:

```
Turn 1 → User: "How does Session handle redirects?"
         Agent: "Via resolve_redirects() in sessions.py..."

Turn 2 → User: "What calls that method?"
         Groq sees: [
           {role: user,      content: "How does Session handle redirects?"},
           {role: assistant, content: "Via resolve_redirects()..."},
           {role: user,      content: "What calls that method?"}   ← current
         ]
         → Correctly resolves "that method" = resolve_redirects
```

Last 3 exchanges (6 messages) are passed per call. Each message truncated to 800 chars.

### Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq `llama-3.3-70b-versatile` (free tier) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, cached to `embedding_model/`) |
| Vector Store | ChromaDB (persistent to `data/vectorstore/`) |
| Agent Framework | LangGraph |
| Code Parsing | Python `ast` module (AST-aware chunking) |
| Interface | Rich CLI |
| Repo Cloning | GitPython (shallow clone, latest code only) |

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- Git installed and available in PATH
- Internet connection (for initial clone of `psf/requests` and Groq API calls)
- A free Groq API key ([get one here](https://console.groq.com))

### Step 1 — Clone this repository

```bash
git clone https://github.com/Zidan010/codebase_qa.git
cd codebase_qa
```

### Step 2 — Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` downloads `all-MiniLM-L6-v2` (~90MB) on first run into `embedding_model/`. This is a one-time download — subsequent runs load from disk instantly.

### Step 4 — Configure Groq API key

**Get a free Groq API key:**

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Click **"API Keys"** in the left sidebar
4. Click **"Create API Key"**
5. Copy the key (starts with `gsk_...`)

**Add it to the environment:**

```bash
# Copy the example env file
cp .env.example .env
```

Open `.env` and fill in api key:

```
GROQ_API_KEY=actual_key_here
```

### Step 5 — Run setup

```bash
python main.py --setup
```

This single command will:
1. Clone `psf/requests` into `data/repos/requests/` (shallow clone, ~seconds)
2. Parse all Python, Markdown, and config files using AST-aware chunking
3. Embed all chunks locally using `all-MiniLM-L6-v2`
4. Persist the vector store to `data/vectorstore/`

**Expected output:**

```
  codebase_q&a
  Agentic Q&A over psf/requests  •  Groq llama-3.3-70b  •  LangGraph  •  ChromaDB

Step 1/2: Checking repository...
✓ Clone complete!
  Branch: main  Last commit: cd90742  by aldbr

Step 2/2: Checking vector index...
Loading embedding model: all-MiniLM-L6-v2 (device: cpu)
✓ Vector store ready (0 chunks already indexed)

Starting indexing pipeline for requests

Step 1/3: Building repository structure map...
  ✓ Structure map stored (187 lines)

Step 2/3: Discovering indexable files...
  ✓ Found 74 indexable files

Step 3/3: Parsing and indexing files...
  ✓ Indexed 721 chunks

╭─────────────────────────────────╮
│       Indexing Summary          │
│  Files discovered      74       │
│  Files parsed          62       │
│  Total chunks         721       │
│    └─ function        298       │
│    └─ class            52       │
│    └─ module           38       │
│    └─ plaintext       332       │
│    └─ structure_map     1       │
│  Time elapsed         47.3s     │
╰─────────────────────────────────╯

✓ Indexing complete! Vector store persisted to data/vectorstore/
```

> **Subsequent runs:** Vector store and cloned repo persist on disk. Re-running skips both automatically.

---

## Usage

### CLI Interface

```bash
python main.py
```

Runs setup check (skips if already done) then launches the conversational CLI.

**Go straight to CLI (skip setup check):**

```bash
python main.py --cli
```

**Force fresh re-index:**

```bash
python main.py --reindex
```

**Available commands inside the CLI:**

| Command | Description |
|---|---|
| `/trace` | Toggle reasoning trace display on/off |
| `/tools` | Toggle tools-used panel on/off |
| `/clear` | Clear conversation history (start fresh) |
| `/status` | Show session info — queries, context cache, toggle states |
| `/help` | Show help and example queries |
| `/quit` | Exit |

---

### Example Queries & Outputs

#### Query 1 — Implementation

```
You: How does Session handle HTTP redirects?

⚙️ Intent: implementation  IN SCOPE  Tools called: 2

╭─ Tools Used ──────────────────────────────────────╮
│  🔍 search_code  5 results for 'redirect handling' │
│  📄 read_file    89 lines from requests/sessions.py│
╰───────────────────────────────────────────────────╯

╭─ Reasoning Trace ─────────────────────────────────╮
│ [Repo Context] Built repo summary (661 chars)      │
│ [Intent] 'implementation' | in_scope=True          │
│ [Tools]  Called search_code → 5 results            │
│ [Tools]  Called read_file(sessions.py:149) → 89 ln │
│ [Critic] sufficient=True | confidence=high         │
│ [Answer] Final answer generated.                   │
╰───────────────────────────────────────────────────╯

╭─ Answer ──────────────────────────────────────────╮
│                                                    │
│  Answer                                            │
│  Session handles redirects via resolve_redirects() │
│  in sessions.py. When a response has a 3xx status  │
│  code, Session.send() calls resolve_redirects()    │
│  which is a generator yielding each redirect hop.  │
│                                                    │
│  How It Works                                      │
│  1. Session.send() receives the initial response   │
│  2. Checks resp.is_redirect (defined in models.py) │
│  3. Calls resolve_redirects() if allow_redirects   │
│  4. Each hop prepares a new Request with Location  │
│  5. Enforces max_redirects=30 to prevent loops     │
│                                                    │
│  Confidence                                        │
│  High — resolve_redirects() is directly in        │
│  sessions.py lines 149–234                         │
╰───────────────────────────────────────────────────╯

╭─ Sources ─────────────────────────────────────────╮
│  File                      Lines    Symbol         │
│  requests/sessions.py      149–234  resolve_redir… │
│  requests/sessions.py      467–490  Session.send   │
│  requests/models.py        89–94    is_redirect    │
╰───────────────────────────────────────────────────╯
```

---

#### Query 2 — Follow-up (multi-turn)

```
You (2): What calls that method?

🔄 Intent: call_flow  IN SCOPE  Tools called: 1

╭─ Answer ──────────────────────────────────────────╮
│  resolve_redirects() is called by Session.send()   │
│  in sessions.py. Session.send() invokes it only    │
│  when allow_redirects=True (the default).          │
│                                                    │
│  Call chain:                                       │
│  Session.get/post() → Session.request()            │
│    → Session.send() → resolve_redirects()          │
╰───────────────────────────────────────────────────╯
```

> The agent correctly resolved "that method" = `resolve_redirects` from the prior conversation turn — using native Groq message history, not keyword matching.

---

#### Query 3 — Call Flow

```
You: Trace the call flow of Session.request

🔄 Intent: call_flow  IN SCOPE  Tools called: 1

╭─ Answer ──────────────────────────────────────────╮
│  Call Flow Diagram                                 │
│                                                    │
│  ► Session.request  [sessions.py:458]              │
│      └─ calls: prepare_request                     │
│      └─ calls: send                                │
│      └─ calls: merge_environment_settings          │
│    ↓                                               │
│    → Session.send  [sessions.py:635]               │
│        └─ calls: get_adapter                       │
│        └─ calls: resolve_redirects                 │
│    ↓                                               │
│    → HTTPAdapter.send  [adapters.py:486]           │
╰───────────────────────────────────────────────────╯
```

---

#### Query 4 — Dependencies

```
You: What does requests.sessions depend on?

🔗 Intent: dependency  IN SCOPE  Tools called: 1

╭─ Answer ──────────────────────────────────────────╮
│  Dependencies of requests.sessions                 │
│                                                    │
│  Internal (repo):                                  │
│    from .adapters import HTTPAdapter               │
│    from .auth import AuthBase                      │
│    from .models import PreparedRequest, Response   │
│    from .hooks import default_hooks                │
│    from .exceptions import TooManyRedirects        │
│    from .cookies import cookiejar_from_dict        │
│                                                    │
│  Standard Library:                                 │
│    os, sys, time, datetime, collections            │
│                                                    │
│  Third-Party: none                                 │
╰───────────────────────────────────────────────────╯
```

---

#### Query 5 — Out of Scope

```
You: What is the capital of France?

🚫 Intent: out_of_scope  OUT OF SCOPE  Tools called: 0

╭─ Answer ──────────────────────────────────────────╮
│  This question is outside the scope of this system.│
│                                                    │
│  Reason                                            │
│  The question is about geography, not the          │
│  psf/requests codebase.                            │
│                                                    │
│  What I Can Help With                              │
│  • Architecture: module organisation, patterns     │
│  • API Usage: Session, HTTPAdapter, auth classes   │
│  • Implementation: how redirects, auth work        │
│  • Call Flow: what happens when requests.get() runs│
│  • Dependencies: what each module imports          │
│  • File Structure: where to find functionality     │
╰───────────────────────────────────────────────────╯
```

---

## Agent Tools

| Tool | Description | Triggered By |
|---|---|---|
| `search_code` | Semantic search with metadata filters (`language`, `chunk_type`, `module`, `file_name`) | Most queries — primary retrieval |
| `read_file` | Read exact file content with optional line range | "Show me the code for...", deep dives |
| `list_directory` | Explore folder structure recursively | Architecture, structure questions |
| `summarize_module` | Module summary: purpose, public API, classes, dependencies | Module-level understanding |
| `find_usages` | AST-based exact symbol search across all files | "Where is X used?", "What uses Y?" |
| `get_dependencies` | Import dependency map: internal / stdlib / third-party | "What does X depend on?" |
| `trace_call_flow` | AST-based call chain tracer, N levels deep | "What calls X?", "Trace the flow of Y" |

**Tool selection is agentic** — the LLM decides which tools to call per query based on intent classification. The agent loops: call tools → critic validates → if insufficient, call more tools → generate answer.

---

## Chosen Repository

**[psf/requests](https://github.com/psf/requests)**

> HTTP for Humans. A simple, elegant HTTP library for Python.

Chosen because:
- Clean, well-structured Python codebase (~8,000 lines of source)
- Rich enough for meaningful cross-file reasoning
- Moderate size — fast to clone and index (~45–60 seconds total)
- Familiar enough that answer correctness is easy to verify

The repository is cloned automatically at runtime (`data/repos/requests/`). You do not need to clone it manually.

---

## Project Structure

```
codebase_qa/
├── main.py                     # Single entry point (setup + CLI)
├── cli.py                      # Conversational CLI interface
├── requirements.txt
├── .env.example                # Environment variable template
│
├── core/
│   ├── config.py               # All configuration and constants
│   │
│   ├── ingestion/
│   │   ├── cloner.py           # Runtime git clone with progress display
│   │   ├── parser.py           # AST-aware chunking + structure map builder
│   │   └── indexer.py          # Full ingestion pipeline with skip-if-exists
│   │
│   ├── tools/
│   │   ├── search_code.py      # Semantic search (valid filters: language, chunk_type, module, file_name)
│   │   ├── read_file.py        # File reader with line range + path traversal guard
│   │   ├── list_directory.py   # Directory tree explorer
│   │   ├── summarize_module.py # Module summariser (AST-based)
│   │   ├── find_usages.py      # Symbol usage finder (AST-based, 6 usage types)
│   │   ├── get_dependencies.py # Import classifier (internal/stdlib/third-party)
│   │   └── trace_call_flow.py  # Call chain tracer (AST-based, configurable depth)
│   │
│   ├── agent/
│   │   ├── state.py            # LangGraph TypedDict state schema
│   │   ├── prompts.py          # All LLM prompts (no history placeholders — native history)
│   │   ├── groq_client.py      # Groq wrapper: retry, history sanitisation, JSON parsing
│   │   ├── nodes.py            # 5 node functions: repo_context, intent, tool, critic, answer
│   │   └── graph.py            # LangGraph graph with conditional routing
│   │
│   ├── vectorstore/
│   │   └── chroma_store.py     # ChromaDB: embed, persist, query, filter validation
│   │
│   └── utils/
│       ├── logger.py           # Rich + file logging
│       └── formatter.py        # CLI output renderer (panels, tables, trace)
│
├── data/
│   ├── repos/                  # Cloned psf/requests (gitignored)
│   └── vectorstore/            # ChromaDB files (gitignored)
│
└── embedding_model/            # Cached all-MiniLM-L6-v2 (gitignored, downloaded once)
```

---

## Screenshots

1. `How does Session handle HTTP redirects?` — implementation intent, shows tools + trace + answer
2. `What calls that method?` — follow-up turn, shows multi-turn history working
3. `Trace the call flow of Session.request` — call_flow intent, shows call diagram
4. `What does requests.sessions depend on?` — dependency intent
5. `What is 1+1?` — out-of-scope guard working

---

## AI Tool Usage Disclosure

This project was developed with assistance from **Claude (Anthropic)** for:

- System architecture and LangGraph graph design
- Code generation across all modules
- Prompt engineering (intent classifier, retrieval critic, tool selector)
- AST parsing strategy for code-aware chunking
- Multi-turn conversation history design (native Groq messages array)
- Debugging and test case design

All generated code has been reviewed, understood, and validated by the developer. No hardcoded answers, fake retrieval results, or mock outputs exist anywhere — all answers are generated from real codebase evidence at runtime.

External references:
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [ChromaDB documentation](https://docs.trychroma.com/)
- [Groq API documentation](https://console.groq.com/docs)
- [sentence-transformers documentation](https://www.sbert.net/)

---

## Troubleshooting

**`GROQ_API_KEY is not set`**
→ Copy `.env.example` to `.env` and paste the api key from [console.groq.com](https://console.groq.com)

**`⏳ Groq rate limit reached. Waiting 30s...`**
→ Normal on free tier. The system retries automatically up to 3 times.
→ To reduce LLM calls per query, set `ENABLE_CRITIC = False` in `core/config.py`

**`Loading weights: 100%` appears on every run**
→ This is loading from local disk (`embedding_model/`), not downloading. The speed (`5000+ it/s`) confirms local read. Safe to ignore.

**`Warning: You are sending unauthenticated requests to the HF Hub`**
→ HuggingFace metadata ping — not a download.

**Vector store empty after restart**
→ Check that `data/vectorstore/` exists and is not empty. If missing, run `python main.py --setup`.

**`search_code` returning 0 results**
→ The agent may be using unsupported filter keys. Valid keys are: `language`, `chunk_type`, `module`, `file_name`. These are validated automatically — invalid keys are stripped before querying ChromaDB.

**Slow first query (~3–5 seconds)**
→ The embedding model loads on first query. All subsequent queries are faster.

**`ModuleNotFoundError` on startup**
→ Ensure virtual environment is activated and `pip install -r requirements.txt` completed without errors.



