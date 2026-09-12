# graph.py — Sofia orchestration

from typing import TypedDict, NotRequired

from langgraph.graph import StateGraph, START, END

from llm import llm, llm_stream
from agent import answer_text, stream_text, run, run_stream
from tools import get_current_time


ROUTER_MODEL = "ollama_chat/sofia-router"


class State(TypedDict):
    task: str
    route: NotRequired[str]
    context: NotRequired[str]
    result: NotRequired[str]


TIME_PHRASES = (
    "what time is it",
    "what's the time",
    "whats the time",
    "current time",
    "tell me the time",
    "what day is it",
    "what's the date",
    "whats the date",
    "what is the date",
    "current date",
    "today's date",
    "todays date",
)


ASSISTANT_SYSTEM = (
    "You are Sofia, a capable local personal AI assistant. "
    "Answer directly and naturally."
)

CODER_SYSTEM = (
    "You are Sofia's senior software engineer. "
    "Use tools to inspect, read, create, and modify files when useful. "
    "Produce working practical output. "
    "Never claim a file was changed unless you actually changed it. "
    "Be efficient with tools: do NOT read a file you have already read, "
    "do NOT list a directory twice, and do NOT call list_allowed_directories. "
    "Once you have the information you need, stop calling tools and answer."
)

RESEARCHER_SYSTEM = (
    "You are Sofia's researcher. "
    "Use available web or fetch tools when current external information "
    "is required. Give a concise answer based on what you find."
)

PLANNER_SYSTEM = (
    "You are Sofia's planning specialist. "
    "Turn the user's request into a concise practical plan."
)


def current_request(task: str) -> str:
    marker = "CURRENT USER REQUEST:\n"

    if marker in task:
        return task.rsplit(marker, 1)[-1].strip()

    return task.strip()


def is_time_request(task: str) -> bool:
    text = current_request(task).lower().strip().rstrip("?!.")
    return any(phrase in text for phrase in TIME_PHRASES)


def router(state: State) -> State:
    task = state["task"]
    current = current_request(task)
    text = current.lower()

    if is_time_request(task):
        print("  [router-fast] -> time")
        return {"route": "time"}

    # Ordinary conversation gets the fast lane:
    # skip the router model entirely.
    agentic_cues = (
        "code",
        "python",
        "javascript",
        "typescript",
        "debug",
        "bug",
        "fix ",
        "refactor",
        "implement",
        "file",
        "folder",
        "repo",
        "git",
        "api",
        "research",
        "search",
        "latest",
        "current",
        "look up",
        "online",
        "plan",
        "roadmap",
        "architecture",
        "strategy",
    )

    if not any(cue in text for cue in agentic_cues):
        print("  [router-fast] -> assistant")
        return {"route": "assistant"}

    msg = llm(
        [
            {
                "role": "system",
                "content": (
                    "Classify the user's request as EXACTLY one of:\n"
                    "assistant\n"
                    "coder\n"
                    "researcher\n"
                    "research_coder\n"
                    "planner\n\n"
                    "assistant = normal conversation or general questions.\n"
                    "coder = anything about code OR local files: reading, "
                    "listing, writing, editing. This is the DEFAULT for file "
                    "and code tasks.\n"
                    "researcher = ONLY when the request needs information from "
                    "the internet (a URL, or words like latest, current, search, "
                    "look up, news). Local files are NOT research.\n"
                    "research_coder = ONLY when it needs BOTH internet info AND "
                    "writing code. Do not pick this for local file tasks.\n"
                    "planner = architecture, strategy, roadmap or decomposition.\n\n"
                    "Reply ONLY with the route word."
                ),
            },
            {"role": "user", "content": task},
        ],
        model=ROUTER_MODEL,
        temperature=0,
    )

    raw = (msg.content or "").strip().lower()

    valid_routes = (
        "research_coder",
        "researcher",
        "assistant",
        "planner",
        "coder",
    )

    # Exact first (the model followed instructions); substring only as fallback.
    word = raw.split()[0].strip(".,!") if raw.split() else ""
    route = next(
        (r for r in valid_routes if r == word),
        next((r for r in valid_routes if r in raw), "coder"),
    )

    print(f"  [router:{ROUTER_MODEL}] -> {route}")

    return {"route": route}


def time_fast(state: State) -> State:
    return {"result": get_current_time()}


def assistant(state: State) -> State:
    return {
        "result": answer_text(
            state["task"],
            system=ASSISTANT_SYSTEM,
        )
    }


def coder(state: State) -> State:
    context = state.get("context", "")
    prompt = state["task"]

    if context:
        prompt += (
            "\n\nResearch/context already gathered:\n"
            + context
        )

    return {
        "result": run(
            prompt,
            system=CODER_SYSTEM,
        )
    }


def researcher(state: State) -> State:
    result = run(
        state["task"],
        system=RESEARCHER_SYSTEM,
    )

    return {
        "context": result,
        "result": result,
    }


def planner(state: State) -> State:
    msg = llm(
        [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": state["task"]},
        ],
        temperature=0.3,
    )

    return {"result": (msg.content or "").strip()}


def after_research(state: State):
    if state.get("route") == "research_coder":
        return "coder"

    return "end"


builder = StateGraph(State)

builder.add_node("router", router)
builder.add_node("time", time_fast)
builder.add_node("assistant", assistant)
builder.add_node("coder", coder)
builder.add_node("researcher", researcher)
builder.add_node("planner", planner)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "time": "time",
        "assistant": "assistant",
        "coder": "coder",
        "researcher": "researcher",
        "research_coder": "researcher",
        "planner": "planner",
    },
)

builder.add_edge("time", END)
builder.add_edge("assistant", END)
builder.add_edge("coder", END)
builder.add_edge("planner", END)

builder.add_conditional_edges(
    "researcher",
    after_research,
    {
        "coder": "coder",
        "end": END,
    },
)

graph = builder.compile()


def stream_task(task):
    route = router({"task": task})["route"]

    if route == "time":
        yield get_current_time()
        return

    if route == "assistant":
        # TRUE immediate streaming: no tools, no buffering.
        yield from stream_text(
            task,
            system=ASSISTANT_SYSTEM,
        )
        return

    if route == "coder":
        yield from run_stream(
            task,
            system=CODER_SYSTEM,
        )
        return

    if route == "researcher":
        yield from run_stream(
            task,
            system=RESEARCHER_SYSTEM,
        )
        return

    if route == "research_coder":
        research = run(
            task,
            system=RESEARCHER_SYSTEM,
        )

        prompt = (
            task
            + "\n\nResearch/context already gathered:\n"
            + research
        )

        yield from run_stream(
            prompt,
            system=CODER_SYSTEM,
        )
        return

    if route == "planner":
        response = llm_stream(
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": task},
            ],
            temperature=0.3,
        )

        for chunk in response:
            if not chunk.choices:
                continue

            content = getattr(
                chunk.choices[0].delta,
                "content",
                None,
            )

            if content:
                yield content
