# guardrails.py - the single gate every tool call passes through
import json
import time
import pathlib
import fnmatch

AUDIT = pathlib.Path(__file__).resolve().parent / "sofia.audit.jsonl"
# Folder Sofia may read/write. Change to wherever your projects live.
ALLOWED_ROOT = pathlib.Path("~/Documents/GIT").expanduser().resolve()

MUTATING = {
    "write_file",
    "edit_file",
    "move_file",
    "create_directory",
}

SECRET_PATTERNS = [
    "*.env",
    "*.pem",
    "*.key",
    "*id_rsa*",
    "*/.ssh/*",
    "*/.git/config",
]

# MCP filesystem arguments that contain paths
PATH_KEYS = {
    "path",
    "source",
    "destination",
}


def _log(name, args, decision):
    rec = {
        "ts": time.time(),
        "tool": name,
        "args": args,
        "decision": decision,
    }
    with AUDIT.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _normalize_paths(args):
    """
    Expand ~/... to an absolute path.

    Relative filesystem paths are rejected because their real destination
    is ambiguous from the confirmation prompt.
    """
    for key in PATH_KEYS:
        value = args.get(key)

        if not isinstance(value, str):
            continue

        # URLs etc. are not filesystem paths
        if "://" in value:
            continue

        p = pathlib.Path(value).expanduser()

        if not p.is_absolute():
            return False, f"relative path not allowed: {value}"

        resolved = p.resolve(strict=False)

        try:
            resolved.relative_to(ALLOWED_ROOT)
        except ValueError:
            return False, f"path outside allowed root: {resolved}"

        # Mutate the actual arguments passed to MCP.
        args[key] = str(resolved)

    return True, None


def _touches_secret(args):
    for value in args.values():
        s = str(value).lower()
        for pat in SECRET_PATTERNS:
            if fnmatch.fnmatch(s, pat):
                return pat
    return None


def check(name, args):
    """Return (allowed, reason)."""

    # Normalize filesystem paths BEFORE showing/using them.
    ok, error = _normalize_paths(args)
    if not ok:
        _log(name, args, f"BLOCKED path: {error}")
        return False, f"blocked: {error}"

    # Never expose protected files.
    hit = _touches_secret(args)
    if hit:
        _log(name, args, f"BLOCKED secret:{hit}")
        return False, f"blocked: touches a protected path ({hit})"

    # Simple exfiltration guard.
    if name == "fetch":
        if len(str(args.get("url", ""))) > 500:
            _log(name, args, "BLOCKED long-url")
            return False, "blocked: fetch URL suspiciously long (possible data leak)"

    # Anything that modifies disk requires approval.
    if name in MUTATING:
        print(f"\n  [confirm] Sofia wants to run: {name}({args})")
        if input("  Allow? [y/N] ").strip().lower() != "y":
            _log(name, args, "DENIED by user")
            return False, "denied by user"

        _log(name, args, "ALLOWED by user")
        return True, "allowed"

    _log(name, args, "auto-allowed")
    return True, "allowed"
