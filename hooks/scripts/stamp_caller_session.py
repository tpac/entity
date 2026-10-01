"""PreToolUse(mcp__brain__*) — stamps the calling session into a brain tool call.

Codex hands stdio MCP servers no thread identity (openai/codex#19937), so the
proxy alone cannot attribute a call. This hook DOES receive `session_id`: it
signs it (servers.dispatch_common, HMAC-SHA256 under the install's hook secret)
and rewrites the tool input with `_caller_session` + `_caller_sig`. The proxy
(brain_mcp._stamp_caller_session) accepts the pair only when the signature
verifies; the model sees tool arguments but never the secret, so it cannot
forge one. No daemon call — read, sign, print — because it runs on every brain
tool call. Registered in hooks.codex.json only: Claude Code hands the proxy
CLAUDE_CODE_SESSION_ID directly (decision fa0f5f5a).
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, emit_updated_input, log_hook_error, run_hook


def main():
    hook_input = get_hook_input()
    tool_name = hook_input.get("tool_name", "") or ""
    sid = hook_input.get("session_id", "") or ""
    tool_input = hook_input.get("tool_input")
    if not sid or not isinstance(tool_input, dict):
        # updatedInput REPLACES the arguments: without the real ones there is
        # nothing safe to rewrite, so the call proceeds unattributed — loudly,
        # since either gap means the host payload is not the documented shape.
        log_hook_error("stamp_caller_session",
                       "no %s on stdin — this brain call stays unattributed"
                       % ("session_id" if not sid else "dict tool_input"),
                       "tool=%s" % tool_name, level="warning")
        return
    from servers.dispatch_common import CALLER_SESSION_KEY, CALLER_SIG_KEY, sign_caller_session
    updated = dict(tool_input)
    updated[CALLER_SESSION_KEY] = sid
    updated[CALLER_SIG_KEY] = sign_caller_session(sid)
    emit_updated_input("PreToolUse", tool_name, updated)


run_hook("stamp_caller_session", main)
