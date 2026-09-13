# Sofia

A local, agentic AI assistant that runs on your own machine. Sofia routes each
request to a specialist, acts on the real world through tools (files, web),
remembers across restarts, checks every action against safety guardrails, and can
be summoned by voice. She runs fully local; cloud models are optional and swappable
through one interface. No third-party AI accounts, no telemetry.

Three ways to reach her: a `sofia "..."` command, an OpenAI-compatible server that
plugs into Open WebUI, and a wake-word voice loop.

---

## Mental model

A language model is a text-in, text-out box with no memory and no hands: it can't
remember past conversations, run code, or touch files. Everything here is
scaffolding around those two limits.

- **No hands** is solved with **tools**: the model emits text asking for a tool;
  our code runs the real thing and feeds the result back.
- **No memory** is solved two ways: the running conversation (short-term) and a
  notebook on disk that survives restarts (long-term).
- **One generalist model is mediocre at everything**, so a **router** sends each
  request to a specialist briefed for that job.
- **Acting on the world is risky**, so every tool call passes through **guardrails**.

---

## Architecture

```
   CLI (sofia "...")        Open WebUI            voice loop (wake word)
        |                       |                       |
        |                       v                       |  wake word (openWakeWord)
        |                 server.py  <------------------ |  -> record -> Whisper (STT)
        |               (OpenAI API)   HTTP /v1          |  -> reply -> Piper (TTS)
        +-----------+-----------+------------------------+
                    v
              graph.py - router
     fast keyword pass first; falls back to the sofia-router model
       |        |        |        |         |            |
       v        v        v        v         v            v
     time   assistant  coder  researcher  planner   research_coder
                          |         |
                          v         v
                       run() / run_stream()   the worker loop
                          |
        +-----------------+------------------+
        v                 v                  v
     llm.py            memory.py         mcp_tools.py
     (spine +          (Chroma +         (persistent MCP sessions:
      fast chat)        nomic-embed)      filesystem + fetch)

     every tool call first passes through guardrails.py
```

The router decides *who* works. `run()` / `run_stream()` are the reusable engine
for *how* a worker does the job, with tools, memory, and guardrails in force. The
CLI, the server, and the voice loop are three front-ends onto the same graph.

---

## Models

Custom Ollama models, built from Modelfiles in the repo:

- **sofia-worker** (`Modelfile.worker`) - `qwen3:8b`, `num_ctx 8192`. The general
  agent and specialists. Chosen for tool-calling support and a footprint (~6 GB)
  that fits a 24 GB machine while leaving room for the voice stack. (Larger workers
  were tried - gpt-oss:20b and qwen3:14b were too heavy on 24 GB; gemma3:12b does
  not support tools.)
- **sofia-router** (`Modelfile.router`) - `qwen3:4b-instruct`, `num_ctx 1024`,
  `temperature 0`. Fast classification only.
- **nomic-embed-text** - embeddings for long-term memory.
- **faster-whisper** - speech-to-text for the voice loop.
- **Kokoro (`af_heart`)** - text-to-speech voice for the voice loop.

Rebuild after editing a Modelfile: `ollama create sofia-worker -f Modelfile.worker`.

Model choice is a one-line change (the `FROM` line in the Modelfile, or the model
string in `llm.py`), so the worker can be swapped for a bigger local model or a
cloud model without touching the rest of the code.

---

## The stack

- **Ollama** - serves the local LLMs.
- **LiteLLM** - one interface to any model backend; also a native Ollama fast-chat
  path for low-latency replies.
- **ChromaDB** + **nomic-embed-text** - long-term memory.
- **MCP** - filesystem and fetch servers, providing Sofia's tools (persistent
  sessions).
- **LangGraph** - models the specialist team as a graph.
- **FastAPI / uvicorn** - the OpenAI-compatible server.
- **openWakeWord** + **faster-whisper** + **Kokoro** - the local voice loop (wake
  word, speech-to-text, text-to-speech). Fully local, no accounts.
- **uv** - Python environment. **Node** - runs the filesystem MCP server.

---

## Files

| File            | What it is                                                               |
|-----------------|--------------------------------------------------------------------------|
| `llm.py`        | The spine: `llm()` / `llm_stream()` for any model, a native Ollama fast-chat path, and `embed()`. |
| `tools.py`      | Hand-written local tools (`get_current_time`, Amsterdam time zone).       |
| `mcp_tools.py`  | Bridge to the MCP servers. Sessions are opened once and reused across calls. |
| `memory.py`     | Long-term memory: `ingest()` stores text, `retrieve()` finds it by meaning. |
| `guardrails.py` | The gate every tool call passes: path normalization + containment, secret wall, confirm-on-write, fetch check, audit log. |
| `agent.py`      | The worker loop: `run()` and `run_stream()`, both taking a `system` persona and optional `model`. |
| `graph.py`      | Orchestration: the router and the six specialist routes.                  |
| `cli.py`        | The `sofia "..."` entry point.                                            |
| `server.py`     | OpenAI-compatible API (`/v1/chat/completions`, streaming) for Open WebUI. |
| `voice.py`      | Wake-word voice loop: openWakeWord -> Whisper -> server.py -> Kokoro.      |
| `warm_models.sh`| Pre-loads the worker and router models.                                   |
| `start_sofia.sh`| Warms models, then launches the server on `127.0.0.1:8000`.               |
| `Modelfile.*`   | Definitions for the custom `sofia-worker` and `sofia-router` models.      |
| `.env`          | API keys (blank for local-only). Never committed.                        |
| `memory_db/`    | Chroma's on-disk store (runtime).                                         |
| `sofia.audit.jsonl` | Log of every tool decision (runtime).                                |

Wake-word models (`*.onnx`) and TTS voices are downloaded/trained separately and
are not committed.

---

## The routes

The router first tries a fast keyword pass and only calls the `sofia-router` model
when that is inconclusive. It defaults to `coder` for local file work and reserves
the research routes for genuine external-info needs.

- **time** - date/time questions. Answered instantly from the local tool, no model call.
- **assistant** - ordinary conversation. Streams straight from the worker, no tools.
- **coder** - code, files, debugging. Runs the full tool loop.
- **researcher** - needs current/external info. Runs the loop with fetch.
- **research_coder** - both: research first, then hand the findings to the coder.
- **planner** - architecture, strategy, decomposition. One-shot plan.

---

## Setup

Prerequisites: [Ollama](https://ollama.com), [uv](https://docs.astral.sh/uv/), Node.js.

```bash
git clone <your-repo-url> sofia && cd sofia
uv venv && source .venv/bin/activate
uv pip install litellm langgraph chromadb python-dotenv fastapi uvicorn mcp
uv pip install openwakeword sounddevice faster-whisper numpy requests onnxruntime piper-tts

# base models, then build the custom ones
ollama pull qwen3:8b
ollama pull qwen3:4b-instruct
ollama pull nomic-embed-text
ollama create sofia-worker -f Modelfile.worker
ollama create sofia-router -f Modelfile.router
```

The MCP servers download themselves on first use via `npx` and `uvx`. The voice
loop needs a wake-word `.onnx` in the project root and a Piper voice; both are
fetched separately (see Voice below).

---

## Running it

Ollama must be running. Then any of the three interfaces:

**CLI:**

```bash
sofia "write a fizzbuzz script and save it to ~/code/sofia/fizzbuzz.py"
```

**Server + Open WebUI:**

```bash
./start_sofia.sh          # warms models, serves 127.0.0.1:8000
```

Add it in Open WebUI as an OpenAI-compatible connection, base URL
`http://127.0.0.1:8000/v1`, any non-empty key. "sofia" then appears in the model
dropdown.

**Voice:**

```bash
./start_sofia.sh          # terminal 1: the brain
python voice.py           # terminal 2: the ears
```

Say the wake word, wait for the acknowledgement, and talk. She records until you
stop, transcribes locally, answers through the server, and speaks the reply with
Piper. `THRESHOLD` in `voice.py` tunes wake sensitivity; `SILENCE_RMS` tunes when a
turn ends.

### Teaching it something (long-term memory)

```bash
python -c "from memory import ingest; ingest('Some fact worth remembering.', source='facts')"
```

Facts are retrieved automatically before Sofia answers and survive restarts.

---

## How each piece works

### The spine (`llm.py`)
`llm()` / `llm_stream()` send a conversation to a model chosen by a string like
`ollama_chat/sofia-worker` (local) or `gemini/gemini-2.5-flash` (cloud); changing
that string moves the brain between backends. A native Ollama fast-chat path is used
for low-latency replies. `embed()` powers memory.

### The worker loop (`agent.py`)
`run(task, system=..., model=...)` sends the conversation plus the tool menu to the
model; text is the answer, a tool call gets run (through guardrails) and fed back,
repeating until it answers. `run_stream()` is the streaming version. The `system`
argument is the worker's persona; `model` lets a specialist use a different backend.

### Memory (`memory.py`)
Text becomes vectors stored in Chroma; a query is embedded too and the closest
chunks come back. Relevant memories are prepended before Sofia answers. This is
RAG: retrieve, then generate.

### Tools via MCP (`mcp_tools.py`)
Sofia connects to pre-built MCP servers, each exposing a bundle of tools. The
servers are started once on a background event loop and the sessions are reused for
every tool call. File operations run through the filesystem server, scoped to
`~/code`.

### Guardrails (`guardrails.py`)
Every tool call passes `check()`:
- **Path normalization + containment** - `~` is expanded, relative paths are
  rejected, and any path resolving outside `~/code` is blocked.
- **Secret wall** - `.env`, `.pem`, `.key`, SSH keys, `.git/config` are blocked outright.
- **Confirm on write** - write/edit/move/mkdir prompt for `y/N` before running.
- **Exit-door check** - a fetch URL over 500 chars is blocked as a possible leak.
- **Audit log** - every decision is written to `sofia.audit.jsonl`.

Blocked calls return a `blocked: reason` note to the model instead of crashing.

### Orchestration (`graph.py`)
A fast keyword pass handles the obvious cases; otherwise the `sofia-router` model
classifies the request in one word, with exact matching and a `coder` fallback. The
chosen specialist runs, and for `research_coder` the researcher's findings are handed
to the coder. LangGraph walks the graph with a shared state.

### Voice (`voice.py`)
A standalone loop: openWakeWord listens on the mic, faster-whisper transcribes the
request, it is POSTed to `server.py`, and the reply is spoken with Kokoro. Replies
are stripped of markdown and emoji before speaking. Fully local.

---

## Notes and limitations

- **Speed.** An agentic request that uses a tool takes roughly 20 to 25 seconds on
  a 24 GB machine. This is the cost of the tool loop, not the model: the worker
  makes two inference passes (decide to call a tool, then read the result and
  answer), plus a tool round-trip. This was measured to be about the same on a
  cloud model, slower on a larger local model (a 14B ran near 40 seconds), and
  unchanged by shortening the answer. It is the floor for local agentic tool calls
  on this hardware. The coder loop is capped at 4 steps so a confused model cannot
  spiral into a much longer run.
- **RAM.** On 24 GB, running the worker, Whisper, Kokoro, the server, and a browser
  at once is tight. Keeping the worker small (qwen3:8b, ~6 GB) is the balance that
  works; a larger worker forces a choice between speed and stability. `keep_alive`
  in `llm.py` pins the model warm (fast, holds RAM) or releases it when idle (frees
  RAM, reloads on next use).
- **Custom wake word.** A quickly-trained custom wake word can false-trigger; the
  `THRESHOLD` dial trades false wakes against missed ones. A prebuilt openWakeWord
  model (e.g. `hey_jarvis`) is more robust if the custom one is too sensitive.
- **Model is swappable.** The worker was tested on local and cloud (Gemini) by
  changing one string; cloud works but the free tier rate-limits an agent quickly,
  and a stronger model probes more aggressively, which the guardrails contained.

---

## Roadmap / ideas

- Tune or retrain the wake word for fewer false triggers.
- Optional barge-in (interrupt Sofia while she speaks).
- Run the server (and voice) as a `launchd` service for always-on use.
- Multi-device access over Tailscale (requires adding auth to `server.py`).
- Tool-call rate limit as an extra guardrail.
