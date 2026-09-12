# tools.py — local hand-written tools (file ops now handled by the MCP fs server)
from datetime import datetime
from zoneinfo import ZoneInfo

def get_current_time():
    now = datetime.now(ZoneInfo("Europe/Amsterdam"))
    return now.strftime("%A, %Y-%m-%d %H:%M:%S %Z")

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
