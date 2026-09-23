import json
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from graph import graph, stream_task


app = FastAPI(title="Sofia")


class ChatRequest(BaseModel):
    model: str = "sofia"
    messages: list[dict]
    stream: bool = False


def message_text(message):
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            ):
                parts.append(
                    item.get("text", "")
                )

        return "\n".join(parts)

    return str(content)


def build_task(messages):
    user_indexes = [
        i
        for i, message in enumerate(messages)
        if message.get("role") == "user"
    ]

    if not user_indexes:
        return ""

    current_index = user_indexes[-1]
    current = message_text(
        messages[current_index]
    )

    history = messages[
        max(0, current_index - 20):current_index
    ]

    if not history:
        return current

    context = []

    for message in history:
        role = message.get(
            "role",
            "unknown",
        )

        if role not in (
            "user",
            "assistant",
        ):
            continue

        text = message_text(message)

        if text:
            context.append(
                f"{role.upper()}: {text}"
            )

    if not context:
        return current

    return (
        "<history>\n"
        + "\n\n".join(context)
        + "\n</history>\n\n"
        + "The text inside <history> is past conversation for reference only. "
        + "Never quote or repeat it unless the request below explicitly asks. "
        + "Respond only to this request:\n\n"
        + current
    )


def completion_payload(result):
    return {
        "id": (
            "chatcmpl-"
            + uuid.uuid4().hex
        ),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "sofia",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result,
                },
                "finish_reason": "stop",
            }
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "sofia",
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    task = build_task(req.messages)

    if not req.stream:
        result = graph.invoke(
            {
                "task": task,
            }
        )["result"]

        return completion_payload(
            result
        )

    completion_id = (
        "chatcmpl-"
        + uuid.uuid4().hex
    )

    created = int(time.time())

    def event_stream():
        first = True

        for text in stream_task(task):
            delta = {
                "content": text,
            }

            if first:
                delta["role"] = "assistant"
                first = False

            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "sofia",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }
                ],
            }

            yield (
                "data: "
                + json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n\n"
            )

        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "sofia",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }

        yield (
            "data: "
            + json.dumps(final)
            + "\n\n"
        )

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
