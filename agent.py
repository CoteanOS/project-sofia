# agent.py — the loop, now with local tools + MCP tools + memory
import json
from llm import llm
from tools import TOOLS, TOOL_SCHEMAS
from memory import retrieve
from mcp_tools import load_mcp, call_mcp_tool

SYSTEM = "You are Sofia, a local assistant. Use tools when they help; otherwise answer directly."

print("Connecting to MCP servers...")
MCP_SCHEMAS, MCP_ROUTING = load_mcp()          # discover MCP tools once at startup
ALL_SCHEMAS = TOOL_SCHEMAS + MCP_SCHEMAS       # one combined menu for the model
print(f"Ready: {len(TOOLS)} local tools + {len(MCP_SCHEMAS)} MCP tools.\n")

def run(user_input, model=None, max_steps=10):
    hits = retrieve(user_input)
    if hits:
        remembered = "\n".join(f"- {h}" for h in hits)
        system = SYSTEM + f"\n\nRelevant things you remember:\n{remembered}"
    else:
        system = SYSTEM
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]
    for step in range(max_steps):
        kwargs = {"tools": ALL_SCHEMAS}
        if model:
            kwargs["model"] = model
        msg = llm(messages, **kwargs)
        messages.append(msg.model_dump())
        if not msg.tool_calls:
            return msg.content
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            print(f"  [tool] {name}({args})")
            try:
                if name in TOOLS:                       # one of your local tools
                    result = TOOLS[name](**args)
                elif name in MCP_ROUTING:               # one of the MCP servers' tools
                    result = call_mcp_tool(MCP_ROUTING, name, args)
                else:
                    result = f"ERROR: unknown tool {name}"
            except Exception as e:
                result = f"ERROR: {e}"
            messages.append({
                "role": "tool", "tool_call_id": call.id, "content": str(result),
            })
    return "Stopped: hit max steps."

if __name__ == "__main__":
    import sys
    print(run(" ".join(sys.argv[1:]) or "What time is it?"))
