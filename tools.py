# tools.py — local hand-written tools (file ops now handled by the MCP fs server)
import datetime

def get_current_time():
    return datetime.datetime.now().isoformat(timespec="seconds")

TOOLS = {
    "get_current_time": get_current_time,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return the current local date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
