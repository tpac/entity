"""PreToolUse(Bash) — catches destructive commands before execution.
Fast regex pre-screen stays in client (avoids daemon round-trip for safe commands).
Only calls daemon/brain for destructive commands.
Output: the brain's safety context (matching warnings) rides
hookSpecificOutput.additionalContext; a clean command prints nothing (see
hook_common.emit_hook_output). The brain informs; it never blocks.
"""
import sys, os, re

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import (get_hook_input, daemon_available, daemon_call_raw,
                         daemon_unavailable_error, brain_debug, emit_hook_output, run_hook)

hook_input = get_hook_input()
tool_input = hook_input.get("tool_input", {})
command = tool_input.get("command", "")

if not command:
    sys.exit(0)

# ── Fast regex pre-screen (stays in client — no daemon round-trip for safe commands) ──
DESTRUCTIVE_REGEXES = [
    r"rm\s+(-[rf]+\s+|.*--force)",
    r"git\s+worktree\s+remove",
    r"git\s+reset\s+--hard",
    r"git\s+clean\s+-[fd]",
    r"git\s+checkout\s+--\s",
    r"git\s+push\s+.*--force",
    r"DROP\s+TABLE",
    r"DELETE\s+FROM",
    r"TRUNCATE",
    r"\brmdir\b",
    r"xargs\s+rm",
]

is_destructive = any(re.search(pat, command, re.IGNORECASE) for pat in DESTRUCTIVE_REGEXES)

if not is_destructive:
    brain_debug("bash: safe → %s" % command[:80])
    sys.exit(0)

# ── Destructive command detected — call daemon/brain for safety check ──
brain_debug("bash: DESTRUCTIVE → %s" % command[:120])


def _warn(detail):
    emit_hook_output("PreToolUse", {
        "additionalContext": "⚠️ Destructive command detected. %s — proceed carefully." % detail,
    })


def main():
    if daemon_available():
        resp = daemon_call_raw("hook_pre_bash_safety", {"command": command, "session_id": hook_input.get("session_id", "")}, timeout=7.0)
        if resp.get("ok"):
            emit_hook_output("PreToolUse", resp.get("result", {}).get("json"))
        else:
            _warn("Safety check unavailable")
    else:
        emit_hook_output("PreToolUse", {"additionalContext": daemon_unavailable_error("pre_bash_safety")})


def _fail_open():
    # Safety check itself crashed — still warn; never let our own error go silent.
    _warn("Safety check error")


run_hook("pre_bash_safety", main, on_error=_fail_open)
