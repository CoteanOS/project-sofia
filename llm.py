# llm.py — one call to rule them all
import os
from dotenv import load_dotenv
from litellm import completion, embedding

load_dotenv()

# Change this one string to move Sofia's brain between backends.
DEFAULT_MODEL = "ollama_chat/gpt-oss:20b"

def llm(messages, model=DEFAULT_MODEL, tools=None, temperature=0.7):
    """Send messages to a model, return the assistant message object.
    The returned object has .content (text) and .tool_calls (if any)."""
    resp = completion(
        model=model,
        messages=messages,
        tools=tools,
        temperature=temperature,
    )
    return resp.choices[0].message

def embed(texts, model="ollama/nomic-embed-text"):
    """Turn text into vectors for memory. Accepts a string or a list."""
    if isinstance(texts, str):
        texts = [texts]
    resp = embedding(model=model, input=texts)
    return [row["embedding"] for row in resp["data"]]

if __name__ == "__main__":
    backends = [
        "ollama_chat/gpt-oss:20b",             # local
        "gemini/gemini-2.5-flash",             # cloud (skipped without a key)
        "openrouter/qwen/qwen3-coder",         # cloud (skipped without a key)
    ]
    for m in backends:
        try:
            msg = llm([{"role": "user", "content": "Say hi in five words."}], model=m)
            print(f"{m:40s} -> {msg.content}")
        except Exception as e:
            print(f"{m:40s} -> skipped ({e})")
