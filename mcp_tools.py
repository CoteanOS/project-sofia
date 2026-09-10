# mcp_tools.py — expose MCP server tools in the same format as tools.py
import asyncio
import pathlib
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Each entry is the command that launches one MCP server.
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

async def _session(params, stack):
    read, write = await stack.enter_async_context(stdio_client(params))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session

async def _list_all():
    schemas, routing = [], {}
    for server_name, params in SERVERS.items():
        async with AsyncExitStack() as stack:
            session = await _session(params, stack)
            tools = (await session.list_tools()).tools
            for t in tools:
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
    """Connect to every server, return (schemas, routing). Call once at startup."""
    return asyncio.run(_list_all())

async def _call(server_name, tool_name, args):
    async with AsyncExitStack() as stack:
        session = await _session(SERVERS[server_name], stack)
        res = await session.call_tool(tool_name, args)
        parts = [c.text for c in res.content if getattr(c, "text", None)]
        return "\n".join(parts) if parts else "(no text output)"

def call_mcp_tool(routing, tool_name, args):
    return asyncio.run(_call(routing[tool_name], tool_name, args))

if __name__ == "__main__":
    schemas, routing = load_mcp()
    print(f"Connected. {len(schemas)} tools available:\n")
    for s in schemas:
        fn = s["function"]
        print(f"  {fn['name']:24s} [{routing[fn['name']]}]  {fn['description'][:60]}")
