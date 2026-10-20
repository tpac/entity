"""PreToolUse(Edit|Write) — the tool moment: memories about a file at its
agent's first edit of it. Thin client: sends hook_pre_edit to the daemon,
which decides whether this edit is a first one and what to show.
Output: hookSpecificOutput.additionalContext when there is something to
surface, otherwise nothing (see hook_common.emit_hook_output).
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import (get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error,
                         brain_debug, emit_hook_output, tool_target_file, run_hook)

hook_input = get_hook_input()
file_path = tool_target_file(hook_input.get("tool_input", {}))

if not file_path:
    sys.exit(0)

# The cue names the file as the session sees it: relative to its working
# directory when the file is under it, so the same file reads the same way on
# every edit.
cwd = hook_input.get("cwd") or ""
path = file_path
if cwd and os.path.isabs(file_path):
    rel = os.path.relpath(file_path, cwd)
    if not rel.startswith(".."):
        path = rel

t0 = time.time()
def main():
    if daemon_available():
        resp = daemon_call_raw("hook_pre_edit", {
            "path": path,
            "session_id": hook_input.get("session_id", ""),
            # Present only on a subagent's tool call (Claude Code).
            "agent_id": hook_input.get("agent_id", ""),
        }, timeout=7.0)
        latency = (time.time() - t0) * 1000
        if resp.get("ok"):
            j = resp.get("result", {}).get("json") or {}
            brain_debug("suggest: %s → %d chars, %dms"
                        % (path, len(j.get("additionalContext", "")), latency))
            emit_hook_output("PreToolUse", j)
    else:
        emit_hook_output("PreToolUse", {"additionalContext": daemon_unavailable_error("pre_edit_suggest")})

run_hook("pre_edit_suggest", main)
