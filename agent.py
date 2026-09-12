# agent.py — Sofia workers: fast chat + full tool agent

import json
import uuid

from llm import llm, llm_stream, ollama_fast_chat, ollama_fast_stream
from tools import TOOLS, TOOL_SCHEMAS
from memory import retrieve
from mcp_tools import load_mcp, call_mcp_tool
from guardrails import check


DEFAULT_SYSTEM = (
    "You are Sofia, a local assistant. "
    "Use tools when they help; otherwise answer directly."
)


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------

print("Connecting to MCP servers...")
MCP_SCHEMAS, MCP_ROUTING = load_mcp()
ALL_SCHEMAS = TOOL_SCHEMAS + MCP_SCHEMAS
print(f"Ready: {len(TOOLS)} local tools + {len(MCP_SCHEMAS)} MCP tools.\n")


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

MEMORY_CUES = (
    "remember",
    "recall",
    "last time",
    "previous",
    "previously",
    "earlier",
    "before",
    "we discussed",
    "we talked",
    "you know",
    "i told you",
    "i said",
    "you told me",
    "continue",
    "again",
    "same as",
    "like before",
    "my ",
    "our ",
)


def _current_request(text: str) -> str:
    marker = "CURRENT USER REQUEST:\n"

    if marker in text:
        return text.rsplit(marker, 1)[-1].strip()

    return text.strip()


def should_use_memory(text: str) -> bool:
    current = _current_request(text).lower()
    padded = f" {current} "
    return any(cue in padded for cue in MEMORY_CUES)


def _build_messages(user_input, system, use_memory):
    base = system or DEFAULT_SYSTEM

    if use_memory:
        hits = retrieve(_current_request(user_input))
        print(f"  [memory] retrieved {len(hits)}")
    else:
        hits = []
        print("  [memory] skipped")

    if hits:
        remembered = "\n".join(f"- {h}" for h in hits)
        system_prompt = (
            base
            + "\n\nRelevant things you remember:\n"
            + remembered
        )
    else:
        system_prompt = base

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]


# ---------------------------------------------------------------------------
# FAST CHAT — no tool schemas, one model call
# ---------------------------------------------------------------------------

def answer_text(
    user_input,
    model=None,
    system=None,
    use_memory=None,
    temperature=0.7,
):
    if use_memory is None:
        use_memory = should_use_memory(user_input)

    messages = _build_messages(
        user_input,
        system,
        use_memory,
    )

    print("  [fast-chat] native Ollama / think=low")

    return ollama_fast_chat(
        messages,
        think="low",
    )


def stream_text(
    user_input,
    model=None,
    system=None,
    use_memory=None,
    temperature=0.7,
):
    if use_memory is None:
        use_memory = should_use_memory(user_input)

    messages = _build_messages(
        user_input,
        system,
        use_memory,
    )

    print("  [fast-chat] native Ollama streaming / think=low")

    yield from ollama_fast_stream(
        messages,
        think="low",
    )


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _execute_tool(name, args):
    print(f"  [tool] {name}({args})")

    allowed, reason = check(name, args)

    if not allowed:
        return f"BLOCKED: {reason}"

    try:
        if name in TOOLS:
            return TOOLS[name](**args)

        if name in MCP_ROUTING:
            return call_mcp_tool(
                MCP_ROUTING,
                name,
                args,
            )

        return f"ERROR: unknown tool {name}"

    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Full agent — tools + guardrails
# ---------------------------------------------------------------------------

def run(
    user_input,
    model=None,
    system=None,
    max_steps=10,
    use_memory=None,
):
    if use_memory is None:
        use_memory = should_use_memory(user_input)

    messages = _build_messages(
        user_input,
        system,
        use_memory,
    )

    for _ in range(max_steps):
        kwargs = {"tools": ALL_SCHEMAS}

        if model:
            kwargs["model"] = model

        msg = llm(messages, **kwargs)

        messages.append(msg.model_dump())

        if not msg.tool_calls:
            return msg.content or ""

        for call in msg.tool_calls:
            name = call.function.name

            try:
                args = json.loads(
                    call.function.arguments or "{}"
                )
            except json.JSONDecodeError:
                args = {}

            result = _execute_tool(name, args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                }
            )

    return "Stopped: hit max steps."


# ---------------------------------------------------------------------------
# Streaming tool agent
# ---------------------------------------------------------------------------

def run_stream(
    user_input,
    model=None,
    system=None,
    max_steps=10,
    use_memory=None,
):
    if use_memory is None:
        use_memory = should_use_memory(user_input)

    messages = _build_messages(
        user_input,
        system,
        use_memory,
    )

    for _ in range(max_steps):
        kwargs = {"tools": ALL_SCHEMAS}

        if model:
            kwargs["model"] = model

        response = llm_stream(
            messages,
            **kwargs,
        )

        content_parts = []
        tool_parts = {}

        for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            content = getattr(delta, "content", None)

            if content:
                content_parts.append(content)

            delta_tools = getattr(
                delta,
                "tool_calls",
                None,
            )

            if delta_tools:
                for tool_delta in delta_tools:
                    index = getattr(
                        tool_delta,
                        "index",
                        0,
                    ) or 0

                    if index not in tool_parts:
                        tool_parts[index] = {
                            "id": None,
                            "type": "function",
                            "function": {
                                "name": "",
                                "arguments": "",
                            },
                        }

                    target = tool_parts[index]

                    tool_id = getattr(
                        tool_delta,
                        "id",
                        None,
                    )

                    if tool_id:
                        target["id"] = tool_id

                    function = getattr(
                        tool_delta,
                        "function",
                        None,
                    )

                    if function:
                        name = getattr(function, "name", None)
                        arguments = getattr(
                            function,
                            "arguments",
                            None,
                        )

                        if name:
                            target["function"]["name"] += name

                        if arguments:
                            target["function"]["arguments"] += arguments

        if tool_parts:
            tool_calls = []

            for index in sorted(tool_parts):
                call = tool_parts[index]

                if not call["id"]:
                    call["id"] = "call_" + uuid.uuid4().hex

                tool_calls.append(call)

            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(content_parts) or None,
                    "tool_calls": tool_calls,
                }
            )

            for call in tool_calls:
                name = call["function"]["name"]

                try:
                    args = json.loads(
                        call["function"]["arguments"] or "{}"
                    )
                except json.JSONDecodeError:
                    args = {}

                result = _execute_tool(name, args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": str(result),
                    }
                )

            continue

        final_text = "".join(content_parts)

        if final_text:
            yield final_text

        return

    yield "Stopped: hit max steps."


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Hello"
    print(run(query))
