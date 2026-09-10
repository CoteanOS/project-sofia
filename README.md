# Sofia

A local, agentic AI assistant that runs on your own machine. Sofia chats through a
local language model, remembers things across restarts, and can act on the real
world (read and write files, fetch web pages) through pluggable tools. No cloud
account required; cloud models are optional and swappable.

Built in phases, each one leaving a working system. Four of six phases are done.

---

## What it is (the mental model)

A language model is a text-in, text-out box. It has no memory and no hands: it
can't remember past conversations, run code, or touch files. Everything here is
scaffolding around those two limits.

- **No hands** is solved with **tools**: the model emits text asking for a tool,
  our code runs the real thing and feeds the result back.
- **No memory** is solved two ways: the running conversation (short-term) and a
  notebook on disk that survives restarts (long-term).

Every phase below is that idea, extended.

---

## Architecture

```
              you type a request
                     │
                     ▼
   ┌─────────────────────────────────────────┐
   │  agent.py  — the loop                    │
   │  1. retrieve relevant memories           │
   │  2. send chat + tool menu to the model   │
   │  3. model replies: answer, or tool call  │
   │  4. run the tool, feed result back       │
   │  5. repeat until it answers              │
   └─────────────────────────────────────────┘
        │              │                 │
        ▼              ▼                 ▼
   llm.py         memory.py         mcp_tools.py
   (the spine)    (the notebook)    (the tool socket)
        │              │                 │
        ▼              ▼                 ▼
   Ollama /       ChromaDB +        MCP servers:
   any model      nomic-embed       filesystem + fetch
```

---

## The stack

- **Ollama** — serves local models. Currently `gpt-oss:20b` (backbone) and
  `nomic-embed-text` (embeddings).
- **LiteLLM** — one interface to every model backend, local or cloud. Switching
  models is a one-string change.
- **ChromaDB** — the vector store behind long-term memory.
- **MCP (Model Context Protocol)** — standard plug format for tools. Sofia uses
  the filesystem and fetch servers.
- **uv** — Python environment and package manager.
- **Node** — required to run the filesystem MCP server (launched via `npx`).

---

## Files

| File            | What it is                                                            |
|-----------------|-----------------------------------------------------------------------|
| `llm.py`        | The spine. `llm()` talks to any model; `embed()` turns text into vectors. |
| `tools.py`      | Hand-written local tools. Currently just `get_current_time`.           |
| `mcp_tools.py`  | Bridge to MCP servers. Discovers their tools and runs them.            |
| `memory.py`     | Long-term memory. `ingest()` stores text; `retrieve()` finds it by meaning. |
| `agent.py`      | The loop that ties it all together.                                    |
| `.env`          | API keys (blank for local-only). Never committed.                     |
| `.gitignore`    | Keeps `.venv/`, `.env`, `memory_db/`, and caches out of git.          |
| `memory_db/`    | Chroma's on-disk store. Created at runtime.                            |

---

## Setup

Prerequisites: [Ollama](https://ollama.com), [uv](https://docs.astral.sh/uv/),
and Node.js installed.

```bash
# 1. Environment
cd ~/code/sofia
uv venv && source .venv/bin/activate
uv pip install litellm langgraph chromadb python-dotenv fastapi uvicorn mcp

# 2. Models (Ollama must be running)
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

The MCP servers (`@modelcontextprotocol/server-filesystem` and `mcp-server-fetch`)
download themselves on first use through `npx` and `uvx`. No manual install.

---

## Running it

Ollama must be running in the background. Then, in an activated environment:

```bash
cd ~/code/sofia
source .venv/bin/activate

python agent.py "what time is it?"
python agent.py "fetch https://example.com and tell me the main heading"
python agent.py "list the files in ~/code/sofia and tell me which is biggest"
```

Every new terminal needs `source .venv/bin/activate` again; it only lasts for that
window.

### Teaching it something (long-term memory)

```bash
python -c "from memory import ingest; ingest('Some fact worth remembering.', source='facts')"
```

Ingested facts are retrieved automatically before Sofia answers, and they survive
restarts because they live in `memory_db/` on disk.

---

## How each piece works

### The spine (`llm.py`)
One function, `llm(messages, model=...)`, sends a conversation to a model and
returns its reply. The model is chosen by a string like `ollama_chat/gpt-oss:20b`
(local) or `gemini/gemini-2.5-flash` (cloud). Changing that one string moves
Sofia's brain between backends without touching anything else. `embed()` does the
same job for the tiny embedding model that powers memory.

### The loop (`agent.py`)
The core of every agent. It sends the conversation plus a menu of available tools
to the model. If the model replies with plain text, that's the answer. If it
replies asking for a tool, the loop runs that tool, appends the result to the
conversation, and goes around again. A `max_steps` cap prevents runaway loops.

### Memory (`memory.py`)
Text is converted into vectors (lists of numbers representing meaning) and stored
in Chroma. To recall something, the query is turned into a vector too and the
closest stored chunks come back. Before each answer, `agent.py` retrieves the most
relevant memories and prepends them to the model's instructions. This is RAG:
retrieve relevant context, then generate.

### Tools via MCP (`mcp_tools.py`)
Rather than hand-writing every capability, Sofia connects to pre-built MCP servers,
each exposing a bundle of tools. `load_mcp()` connects at startup and translates
each server's tools into the same schema shape the local tools use. `agent.py`
merges local and MCP tools into one menu, and when the model calls one, dispatches
it to the right place. File operations are handled entirely by the filesystem
server (scoped to `~/code`).

---

## Current capabilities

- Chat through a fully local model, offline.
- Swap to any cloud model by changing one string (needs a key in `.env`).
- Long-term memory that persists across restarts.
- 1 local tool (`get_current_time`) plus 15 MCP tools: full file operations
  (read, write, edit, search, move, directory trees) and web fetch.

---

## Roadmap

- **Phase 4 — Orchestration.** Replace the single model with a team: a small fast
  model routes each request to a specialist (planner, coder, researcher). Also
  where tool-heavy jobs get sent to a sharper model.
- **Phase 5 — Interface.** A `sofia "..."` CLI command, and wiring the brain into
  Open WebUI so it inherits the existing chat and voice setup.
- **Phase 6 — Hardening.** Structured logging of every step and tool call, secret
  and filesystem scoping, and pointing the backbone at a bigger machine when one
  arrives.

### Done

- **Phase 0** — the LLM spine.
- **Phase 1** — the agent loop with hand-written tools.
- **Phase 3** — long-term memory via embeddings and Chroma.
- **Phase 2** — real tools via MCP servers.
