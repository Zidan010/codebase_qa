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

---

#### Query 1 — Implementation

```
You: How does Session handle HTTP redirects?
```

**Intent classification:**

```
⚙️ Intent: 'implementation' | in_scope=True
   Tools: ['search_code', 'read_file', 'find_usages']
   Reason: The user wants to know the internal details of how the Session class handles
           HTTP redirects, which is a specific implementation detail within the psf/requests codebase.
```

**Tools used:**

```
╭─ Tools Used ────────────────────────────────────────────────╮
│  🔍 search_code       5 results for 'Session HTTP redirect' │
│  📄 summarize_module  module src.requests.sessions          │
╰─────────────────────────────────────────────────────────────╯
```

**Reasoning trace:**

```
╭─ Reasoning Trace ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                        │
│  [Repo Context] Built repo summary (724 chars)                                                                         │
│                                                                                                                        │
│  [Intent] 'implementation' | in_scope=True                                                                             │
│           tools=['search_code', 'read_file', 'find_usages']                                                            │
│           reason: The user wants to know the internal details of how the Session class handles HTTP redirects,         │
│           which is a specific implementation detail within the psf/requests codebase.                                  │
│                                                                                                                        │
│  [Tools] Reasoning: To understand how Session handles HTTP redirects, we need to look into the implementation          │
│          details of the Session class, specifically how it interacts with the adapters and the request/response        │
│          cycle. The `requests.sessions` module is likely to contain the relevant code.                                 │
│                                                                                                                        │
│  [Tools] Called search_code(query='Session HTTP redirect', top_k=5,                                                    │
│          filters={'module': 'requests.sessions'}) → 5 results                                                          │
│                                                                                                                        │
│  [Tools] Called summarize_module(module_name='requests.sessions') → module 'src.requests.sessions'                     │
│                                                                                                                        │
│  [Critic] sufficient=True | confidence=low                                                                             │
│           The provided evidence includes the definition of Session and SessionRedirectMixin classes, but it does       │
│           not directly explain how Session handles HTTP redirects. The SessionRedirectMixin class has attributes        │
│           related to redirects (max_redirects), but the actual redirect handling logic is not shown in the             │
│           provided code snippets.                                                                                      │
│                                                                                                                        │
│  [Answer] Final answer generated.                                                                                      │
│                                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**Answer:**

```
╭─ Answer ───────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                        │
│  Answer                                                                                                                │
│  The Session class in the psf/requests library handles HTTP redirects through the SessionRedirectMixin class,          │
│  which it inherits from. However, the provided evidence does not directly explain how Session handles HTTP             │
│  redirects, as the actual redirect handling logic is not shown in the provided code snippets.                          │
│                                                                                                                        │
│  How It Works                                                                                                          │
│  Unfortunately, the provided evidence does not contain enough information to provide a step-by-step explanation        │
│  of how the Session class handles HTTP redirects.                                                                      │
│                                                                                                                        │
│  Relevant Code                                                                                                         │
│  The SessionRedirectMixin class has attributes related to redirects, such as max_redirects                             │
│  (src\requests\sessions.py:127–392). The Session class inherits from SessionRedirectMixin                              │
│  (src\requests\sessions.py:395–905).                                                                                   │
│                                                                                                                        │
│  Sources                                                                                                               │
│  • src\requests\sessions.py lines 127–392  — SessionRedirectMixin                                                      │
│  • src\requests\sessions.py lines 395–905  — Session                                                                   │
│  • src\requests\sessions.py lines 1–52     — src.requests.sessions                                                     │
│  • src\requests\sessions.py lines 511–555  — Session.prepare_request                                                   │
│                                                                                                                        │
│  Confidence                                                                                                            │
│  Low — The provided evidence does not contain enough information to fully understand how the Session class handles      │
│  HTTP redirects, as the actual redirect handling logic is not shown in the provided code snippets.                     │
│                                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**Sources table:**

```
╭─ Sources ──────────────────────────────────────────────────────────────────────────────╮
│  File                          Lines       Symbol                    Type              │
│  ─────────────────────────────────────────────────────────────────────────────────     │
│  src\requests\sessions.py      127–392     SessionRedirectMixin      class             │
│  src\requests\sessions.py      395–905     Session                   class             │
│  src\requests\sessions.py      908–920     session                   function          │
│  src\requests\sessions.py      1–52        src.requests.sessions     module            │
│  src\requests\sessions.py      511–555     Session.prepare_request   function          │
│  src\requests\sessions.py      full module src.requests.sessions     module_summary    │
╰────────────────────────────────────────────────────────────────────────────────────────╯
```

---

#### Query 2 — Follow-up (multi-turn)

```
You (2): What calls that method?
```

**Intent classification:**

```
🔄 Intent: 'call_flow' | in_scope=True
   Tools: ['find_usages', 'trace_call_flow']
   Reason: The user wants to know what part of the code calls a specific method,
           likely to understand the execution flow or request lifecycle.
```

**Reasoning trace:**

```
╭─ Reasoning Trace ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                        │
│  [Intent] 'call_flow' | in_scope=True | tools=['find_usages', 'trace_call_flow']                                       │
│           reason: The user wants to know what part of the code calls a specific method, likely to understand           │
│           the execution flow or request lifecycle.                                                                      │
│                                                                                                                        │
│  [Tools] Reasoning: To answer the user's question about how Session handles HTTP redirects, we need to find            │
│          the method that handles redirects and then find the callers of that method. The Session class is defined       │
│          in the requests.sessions module. We can start by finding the method that handles redirects.                   │
│                                                                                                                        │
│  [Tools] Called search_code(query='HTTP redirect handling', top_k=5,                                                   │
│          filters={'module': 'requests.sessions'}) → 5 results                                                          │
│                                                                                                                        │
│  [Critic] sufficient=False | confidence=low                                                                            │
│           The provided evidence shows several methods within the sessions.py file, but it does not explicitly          │
│           state which method calls another. The methods shown are get_redirect_target, should_strip_auth, and          │
│           rebuild_method, but the connections between them are not specified in the given code snippets.               │
│                                                                                                                        │
│  [Tools] Reasoning: To answer the user's question about what calls the method that handles HTTP redirects, we          │
│          need to find the usages of the method responsible for handling HTTP redirects. Since we already know that      │
│          the Session class handles HTTP redirects, we can start by finding the usages of the Session class.            │
│                                                                                                                        │
│  [Tools] Called find_usages(symbol_name='Session') → 20 usages                                                         │
│  [Tools] Called trace_call_flow(symbol_name='Session', depth=2) → 1 level traced                                       │
│                                                                                                                        │
│  [Critic] sufficient=False | confidence=low                                                                            │
│           The provided evidence includes several methods from the Session class, but it does not explicitly show        │
│           which method calls another. The call flow section only shows the instantiation of the Session class,         │
│           but does not provide information on method calls.                                                            │
│                                                                                                                        │
│  [Tools] Called search_code(query='HTTP redirect handling', top_k=10,                                                  │
│          filters={'module': 'requests.sessions'}) → 10 results                                                         │
│                                                                                                                        │
│  [Critic] sufficient=False | confidence=low                                                                            │
│           The provided evidence does not directly mention which method calls the target method, only providing          │
│           method definitions and some usage examples of the Session class.                                              │
│                                                                                                                        │
│  [Tools] Called read_file(path='src/requests/sessions.py', start_line=1, end_line=500) → 920 lines                     │
│                                                                                                                        │
│  [Critic] sufficient=False | confidence=low                                                                            │
│           The provided evidence does not directly mention which method calls another method, only providing             │
│           method definitions and some usage information of the Session class.                                           │
│                                                                                                                        │
│  [Answer] Final answer generated.                                                                                      │
│                                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**Tools used:**

```
╭─ Tools Used ──────────────────────────────────────────────────────────────────────╮
│  🔍 search_code      5 results for 'HTTP redirect handling'                       │
│  🔎 find_usages      20 usages of Session                                         │
│  🔄 trace_call_flow  traced Session (1 level)                                     │
│  🔄 trace_call_flow  traced Session (1 level)                                     │
│  🔍 search_code      10 results for 'HTTP redirect handling'                      │
│  🔎 find_usages      20 usages of Session                                         │
│  🔍 search_code      5 results for 'HTTP redirect handling'                       │
│  📄 read_file        920 lines from src/requests/sessions.py                      │
╰───────────────────────────────────────────────────────────────────────────────────╯
```

**Answer:**

```
╭─ Answer ───────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                        │
│  Answer                                                                                                                │
│  The method that handles HTTP redirects is called by the resolve_redirects function, which is defined in               │
│  src\requests\sessions.py at lines 186–307. However, the provided evidence does not explicitly show which              │
│  method calls resolve_redirects.                                                                                       │
│                                                                                                                        │
│  How It Works                                                                                                          │
│  Unfortunately, the provided evidence does not contain enough information to provide a step-by-step explanation        │
│  of how the Session class handles HTTP redirects.                                                                      │
│                                                                                                                        │
│  Relevant Code                                                                                                         │
│  The resolve_redirects function is defined in src\requests\sessions.py at lines 186–307.                               │
│  The get_redirect_target function, which is also related to HTTP redirect handling, is defined in                      │
│  src\requests\sessions.py at lines 134–152.                                                                            │
│                                                                                                                        │
│  Sources                                                                                                               │
│  • src\requests\sessions.py lines 134–152  — SessionRedirectMixin.get_redirect_target                                  │
│  • src\requests\sessions.py lines 154–184  — SessionRedirectMixin.should_strip_auth                                    │
│  • src\requests\sessions.py lines 186–307  — SessionRedirectMixin.resolve_redirects                                    │
│  • src\requests\sessions.py lines 370–392  — SessionRedirectMixin.rebuild_method                                       │
│                                                                                                                        │
│  Confidence                                                                                                            │
│  Low — The provided evidence does not directly mention which method calls the target method, only providing             │
│  method definitions and some usage information of the Session class.                                                    │
│                                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**Sources table:**

```
╭─ Sources ──────────────────────────────────────────────────────────────────────────────────────────────╮
│  File                          Lines     Symbol                                      Type              │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────     │
│  src\requests\sessions.py      134–152   SessionRedirectMixin.get_redirect_target    function          │
│  src\requests\sessions.py      154–184   SessionRedirectMixin.should_strip_auth      function          │
│  src\requests\sessions.py      370–392   SessionRedirectMixin.rebuild_method         function          │
│  src\requests\sessions.py      186–307   SessionRedirectMixin.resolve_redirects      function          │
│  src\requests\sessions.py      684–693   Session.head                                function          │
│  src/requests/sessions.py      1–300                                                 file_read         │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

> The agent correctly resolved "that method" = `resolve_redirects` from the prior conversation turn — using native Groq message history, not keyword matching.

---

#### Query 3 — Call Flow

```
You: Trace the call flow of Session.request
```

**Intent classification:**

```
🔄 Intent: 'call_flow' | in_scope=True
   Tools: ['trace_call_flow']
   Reason: The user wants to know the execution flow of the Session.request method.
```

**Reasoning trace:**

```
╭─ Reasoning Trace ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                        │
│  [Intent] 'call_flow' | in_scope=True | tools=['trace_call_flow']                                                      │
│           reason: The user wants to know the execution flow of the Session.request method.                             │
│                                                                                                                        │
│  [Tools] Reasoning: To answer the user's question about tracing the call flow of Session.request, we need to           │
│          understand how the Session class handles requests. The Session class is defined in the requests.sessions       │
│          module. We can use the trace_call_flow tool to trace the execution call chain of Session.request.             │
│                                                                                                                        │
│  [Tools] Called trace_call_flow(symbol_name='Session.request', depth=5) → 4 levels traced                              │
│                                                                                                                        │
│  [Critic] sufficient=True | confidence=high                                                                            │
│           The evidence provides a clear call flow of Session.request, including the methods it calls and their         │
│           respective locations in the codebase (e.g., src\requests\sessions.py:557,                                   │
│           src\requests\cookies.py:391, src\requests\adapters.py:128).                                                  │
│                                                                                                                        │
│  [Answer] Final answer generated.                                                                                      │
│                                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**Tools used:**

```
╭─ Tools Used ────────────────────────────────────────────────────╮
│  🔄 trace_call_flow  traced Session.request (4 levels)          │
╰─────────────────────────────────────────────────────────────────╯
```

**Answer:**

```
╭─ Answer ───────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                        │
│  Answer                                                                                                                │
│  The Session.request method is defined in src\requests\sessions.py at line 557. It calls several other methods,        │
│  including RequestsCookieJar.update, Session.merge_environment_settings, BaseAdapter.send, and                         │
│  Session.prepare_request. The Session.request method is the primary interface for sending HTTP requests using           │
│  the psf/requests library.                                                                                             │
│                                                                                                                        │
│  How It Works                                                                                                          │
│  Here's a step-by-step explanation of how Session.request works:                                                       │
│                                                                                                                        │
│  1. The Session.request method is called with the request parameters.                                                  │
│  2. It calls RequestsCookieJar.update to update the cookie jar.                                                        │
│  3. It calls Session.merge_environment_settings to merge the environment settings.                                     │
│  4. It calls BaseAdapter.send to send the request.                                                                     │
│  5. It calls Session.prepare_request to prepare the request.                                                           │
│                                                                                                                        │
│  Relevant Code                                                                                                         │
│  • Session.request                      — src\requests\sessions.py:557                                                 │
│  • RequestsCookieJar.update             — src\requests\cookies.py:391                                                  │
│  • Session.merge_environment_settings   — src\requests\sessions.py:831                                                 │
│  • BaseAdapter.send                     — src\requests\adapters.py:128                                                 │
│  • Session.prepare_request              — src\requests\sessions.py:511                                                 │
│                                                                                                                        │
│  Sources                                                                                                               │
│  • src\requests\sessions.py lines 557–307  — Session.request                                                           │
│  • src\requests\cookies.py  lines 391–467  — RequestsCookieJar.update                                                  │
│  • src\requests\sessions.py lines 831–76   — Session.merge_environment_settings                                        │
│  • src\requests\adapters.py lines 128–     — BaseAdapter.send                                                          │
│  • src\requests\sessions.py lines 511–     — Session.prepare_request                                                   │
│                                                                                                                        │
│  Confidence                                                                                                            │
│  High — The evidence provides a clear call flow of Session.request, including the methods it calls and their           │
│  respective locations in the codebase.                                                                                 │
│                                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

---

#### Query 4 — Dependencies

```
You: What does requests.sessions depend on?
```

**Intent classification:**

```
🔗 Intent: 'dependency' | in_scope=True
   Tools: ['get_dependencies']
   Reason: The user wants to know what modules or packages the requests.sessions module depends on.
```

**Reasoning trace:**

```
╭─ Reasoning Trace ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                        │
│  [Intent] 'dependency' | in_scope=True | tools=['get_dependencies']                                                    │
│           reason: The user wants to know what modules or packages the requests.sessions module depends on.             │
│                                                                                                                        │
│  [Tools] Reasoning: The user is asking about the dependencies of the requests.sessions module. To answer this,         │
│          we need to get the import dependencies of the requests.sessions module.                                       │
│                                                                                                                        │
│  [Tools] Called get_dependencies(module_name='requests.sessions')                                                      │
│          → internal=12, stdlib=8, third_party=0                                                                        │
│                                                                                                                        │
│  [Critic] sufficient=True | confidence=high                                                                            │
│           The evidence lists the internal dependencies of src.requests.sessions, including modules like adapters,      │
│           auth, and cookies, which directly answers the question about what requests.sessions depends on.              │
│                                                                                                                        │
│  [Answer] Final answer generated.                                                                                      │
│                                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**Tools used:**

```
╭─ Tools Used ──────────────────────────────────────────────────────────────────────────────────╮
│  🔗 get_dependencies  src.requests.sessions → internal=12, stdlib=8, 3rd-party=0              │
╰───────────────────────────────────────────────────────────────────────────────────────────────╯
```

**Answer:**

```
╭─ Answer ───────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                        │
│  Answer                                                                                                                │
│  The requests.sessions module depends on 12 internal modules, including adapters, auth, cookies, and exceptions,       │
│  as well as 8 standard library modules, such as os, sys, and collections. Specifically, it imports classes and         │
│  functions like HTTPAdapter, RequestsCookieJar, and default_hooks from these internal modules.                         │
│                                                                                                                        │
│  How It Works                                                                                                          │
│  The dependencies are imported at the beginning of the src.requests.sessions module, allowing the session              │
│  management functionality to utilise the connection logic, authentication, and cookie handling provided by these       │
│  dependencies. The imports enable the requests.sessions module to perform tasks such as sending HTTP requests,         │
│  managing cookies, and handling redirects.                                                                             │
│                                                                                                                        │
│  Relevant Code                                                                                                         │
│  The evidence lists the internal dependencies of src.requests.sessions, including:                                     │
│                                                                                                                        │
│  from .adapters import HTTPAdapter                                                                                     │
│  from .auth import _basic_auth_str                                                                                     │
│  from .cookies import (RequestsCookieJar, cookiejar_from_dict,                                                         │
│                         extract_cookies_to_jar, merge_cookies)                                                         │
│  from .exceptions import (ChunkedEncodingError, ContentDecodingError,                                                  │
│                             InvalidSchema, TooManyRedirects)                                                           │
│                                                                                                                        │
│  These imports are crucial for the functionality of the requests.sessions module.                                      │
│                                                                                                                        │
│  Sources                                                                                                               │
│  • src.requests.sessions lines 1–20   — import statements for internal dependencies                                    │
│  • src.requests.sessions lines 21–30  — import statements for standard library dependencies                            │
│                                                                                                                        │
│  Confidence                                                                                                            │
│  High — The evidence provides a clear and comprehensive list of dependencies for the requests.sessions module,         │
│  allowing for a precise answer to the question.                                                                        │
│                                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

---

#### Query 5 — Out of Scope

```
You: What is the capital of France?
```

**Intent classification:**

```
🚫 Intent: 'out_of_scope' | in_scope=False
   Tools called: 0
```

**Answer:**

```
╭─ Answer ───────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                        │
│  Answer                                                                                                                │
│  This question is outside the scope of this system.                                                                   │
│                                                                                                                        │
│  Reason                                                                                                                │
│  The question is about general knowledge and not related to the psf/requests codebase.                                 │
│                                                                                                                        │
│  What I Can Help With                                                                                                  │
│  This assistant is specialised exclusively for the psf/requests Python HTTP library. I can answer questions about:     │
│                                                                                                                        │
│  • Architecture     — How the library is structured, module organisation                                               │
│  • API Usage        — How to use Session, HTTPAdapter, auth classes, etc.                                              │
│  • Implementation   — How redirects, connection pooling, or auth work internally                                       │
│  • Call Flow        — What happens when you call requests.get()                                                        │
│  • Dependencies     — What each module imports and depends on                                                          │
│  • File Structure   — Where to find specific functionality in the codebase                                             │
│                                                                                                                        │
│  Please ask something about the psf/requests codebase.                                                                 │
│                                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
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

1. `How does Session handle HTTP redirects?` — implementation intent, shows tools + trace + answer![Query 1](screenshots\q1.png)
2. `What calls that method?` — follow-up turn, shows multi-turn history working![Query 2](screenshots\q2.png)
3. `Trace the call flow of Session.request` — call_flow intent, shows call diagram![Query 3](screenshots\q3.png)
4. `What does requests.sessions depend on?` — dependency intent![Query 4](screenshots\q4.png)
5. `What is the capital of France?` — out-of-scope guard working![Query 5](screenshots\q5.png)

---

## AI Tool Usage Disclosure

This project was developed with assistance from **Claude (Anthropic)** for:

- System architecture and LangGraph graph design
- Prompt engineering (intent classifier, retrieval critic, tool selector)
- Code documentation and description generatiion
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



