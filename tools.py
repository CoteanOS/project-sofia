from memory import ingest, retrieve
# tools.py - local hand-written tools (file ops now handled by the MCP fs server)
from datetime import datetime
from zoneinfo import ZoneInfo

def get_current_time():
    now = datetime.now(ZoneInfo("Europe/Amsterdam"))
    return now.strftime("%A, %Y-%m-%d %H:%M:%S %Z")


def remember(fact):
    """Store a durable fact about the user for future conversations."""
    ingest(fact, source="user")
    return f"Saved: {fact}"

def recall(query):
    """Look up what has been remembered about a topic."""
    hits = retrieve(query)
    return "\n".join(hits) if hits else "Nothing remembered about that."

TOOLS = {
    "remember": remember,
    "recall": recall,
    "get_current_time": get_current_time,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "remember",
        "description": "Save a durable fact about the user (their name, preferences, how to address them, anything to recall later). Call this whenever the user says to remember something or states a lasting preference.",
        "parameters": {"type": "object", "properties": {
            "fact": {"type": "string", "description": "The fact to store, as a clear sentence."}
        }, "required": ["fact"]},
    }},
    {"type": "function", "function": {
        "name": "recall",
        "description": "Look up previously remembered facts about a topic.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look up."}
        }, "required": ["query"]},
    }},
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return the current local date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
