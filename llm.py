# llm.py - one interface to Sofia's model backends

from dotenv import load_dotenv
from litellm import completion, embedding

load_dotenv()

DEFAULT_MODEL = "ollama_chat/sofia-worker"


def llm(messages, model=DEFAULT_MODEL, tools=None, temperature=0.7):
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if tools:
        kwargs["tools"] = tools

    response = completion(**kwargs)
    return response.choices[0].message


def llm_stream(messages, model=DEFAULT_MODEL, tools=None, temperature=0.7):
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }

    if tools:
        kwargs["tools"] = tools

    return completion(**kwargs)


def embed(texts, model="ollama/nomic-embed-text"):
    if isinstance(texts, str):
        texts = [texts]

    response = embedding(
        model=model,
        input=texts,
    )

    return [
        row["embedding"]
        for row in response["data"]
    ]


# ---------------------------------------------------------------------------
# Native Ollama fast-chat path
# ---------------------------------------------------------------------------

import json as _json
import urllib.request as _urllib_request


OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
FAST_CHAT_MODEL = "sofia-worker"


def ollama_fast_chat(messages, think=None):
    body = {
        "model": FAST_CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": -1,
    }
    if think is not None:
        body["think"] = think
    payload = _json.dumps(body).encode()

    request = _urllib_request.Request(
        OLLAMA_CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with _urllib_request.urlopen(request) as response:
        result = _json.loads(response.read())

    return result.get("message", {}).get("content", "")


def ollama_fast_stream(messages, think=None):
    body = {
        "model": FAST_CHAT_MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": -1,
    }
    if think is not None:
        body["think"] = think
    payload = _json.dumps(body).encode()

    request = _urllib_request.Request(
        OLLAMA_CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with _urllib_request.urlopen(request) as response:
        for raw in response:
            obj = _json.loads(raw)

            content = obj.get(
                "message",
                {},
            ).get(
                "content",
                "",
            )

            if content:
                yield content

            if obj.get("done"):
                return
