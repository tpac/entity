"""PreToolUse(Edit|Write) — surfaces brain rules/suggestions before edits.
Thin client: sends hook_pre_edit to daemon.
Output: hookSpecificOutput.additionalContext when there is something to
surface, otherwise nothing (see hook_common.emit_hook_output).
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import (get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error,
                         brain_debug, emit_hook_output, tool_target_file, run_hook)

hook_input = get_hook_input()
tool_input = hook_input.get("tool_input", {})
file_path = tool_target_file(tool_input)
tool_name = hook_input.get("tool_name") or ""

if not file_path:
    sys.exit(0)

filename = os.path.basename(file_path)

# Skip non-source files
skip_exts = [".log", ".map", ".lock", ".json"]
if any(filename.endswith(ext) for ext in skip_exts) and filename != "package.json":
    brain_debug("suggest: skipped %s (non-source)" % filename)
    sys.exit(0)

t0 = time.time()
def main():
    if daemon_available():
        resp = daemon_call_raw("hook_pre_edit", {
            "filename": filename,
            "tool_name": tool_name,
            "session_id": hook_input.get("session_id", ""),
        }, timeout=7.0)
        latency = (time.time() - t0) * 1000
        if resp.get("ok"):
            result = resp.get("result", {})
            if "json" in result:
                j = result["json"]
                reason = j.get("reason", "")
                brain_debug("suggest: %s → %d chars, %dms" % (filename, len(reason), latency))
                emit_hook_output("PreToolUse", j)
            else:
                brain_debug("suggest: %s → no rules, %dms" % (filename, latency))
    else:
        emit_hook_output("PreToolUse", {"additionalContext": daemon_unavailable_error("pre_edit_suggest")})

run_hook("pre_edit_suggest", main)
