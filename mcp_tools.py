# mcp_tools.py - persistent MCP sessions.
# Servers are started ONCE on a background event loop and reused for every
# tool call, instead of being relaunched per call. Public API is unchanged:
# load_mcp() -> (schemas, routing);  call_mcp_tool(routing, name, args) -> str
import asyncio
import pathlib
import threading
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVERS = {
    "fs": StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem",
              str(pathlib.Path("~/code").expanduser())],
    ),
    "fetch": StdioServerParameters(
        command="uvx",
        args=["mcp-server-fetch"],
    ),
}

# --- one background event loop, running for the life of the process ---
_loop = asyncio.new_event_loop()
_thread = threading.Thread(target=_loop.run_forever, daemon=True)
_thread.start()

def _run(coro):
    """Submit a coroutine to the background loop and block for its result."""
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()

# Persistent state, all owned by the background loop.
_stack = AsyncExitStack()
_sessions = {}      # server_name -> live ClientSession

async def _open_all():
    schemas, routing = [], {}
    for server_name, params in SERVERS.items():
        read, write = await _stack.enter_async_context(stdio_client(params))
        session = await _stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        _sessions[server_name] = session                 # kept alive, not torn down
        for t in (await session.list_tools()).tools:
            schemas.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.input_schema or {"type": "object", "properties": {}},
                },
            })
            routing[t.name] = server_name
    return schemas, routing

def load_mcp():
    """Start every server once and return (schemas, routing)."""
    return _run(_open_all())

async def _call(server_name, tool_name, args):
    session = _sessions[server_name]                     # reuse the live session
    res = await session.call_tool(tool_name, args)
    parts = [c.text for c in res.content if getattr(c, "text", None)]
    return "\n".join(parts) if parts else "(no text output)"

def call_mcp_tool(routing, tool_name, args):
    return _run(_call(routing[tool_name], tool_name, args))

if __name__ == "__main__":
    schemas, routing = load_mcp()
    print(f"Connected (persistent). {len(schemas)} tools available:\n")
    for s in schemas:
        fn = s["function"]
        print(f"  {fn['name']:24s} [{routing[fn['name']]}]  {fn['description'][:60]}")
