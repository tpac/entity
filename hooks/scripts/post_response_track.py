"""Stop hook: records the turn's assistant half (S0 traces) and delivers
pending self-messages. The operator's half is written at UserPromptSubmit.
Thin client: sends hook_post_response_track to daemon.
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import (get_hook_input, daemon_available, daemon_call_raw,
                         daemon_unavailable_error, brain_debug, emit_hook_output, run_hook,
                         turn_model, host_tells, closing_message, log_hook_error)

hook_input = get_hook_input()
event_name = hook_input.get("hook_event_name", "")


def _read_reply(text):
    """(reply, contract stamp, closing message or None). What of the turn
    gets in is the host contract's to say (read_reply, the reply side's
    read_prompt): the host's reply text plus what a reply tool closing the
    turn showed the operator. The transcript is read only on a host that
    declares reply tools."""
    from servers.host_contract import (resolve_host, read_reply, reply_tools,
                                       contract_fingerprint, VOCAB_VERSION)
    host = resolve_host({t: True for t in host_tells()})[0]
    closing = closing_message(hook_input) if reply_tools(host) else None
    reply = read_reply(text, (closing or {}).get("tool_calls", ()), host)
    return reply, {"vocab_version": VOCAB_VERSION, "impl_identity": contract_fingerprint()}, closing


def main():
    t0 = time.time()
    if daemon_available():
        text = hook_input.get("last_assistant_message", "") or ""
        try:
            reply, stamp, closing = _read_reply(text)
        except Exception as e:
            # The host's own text still records the turn, and Stop still
            # delivers; the unstamped row marks the reply as unread.
            log_hook_error("track", e, "reply reader failed — recorded the host's text")
            reply, stamp, closing = text, {}, None
        brain_debug("track: event=%s, assistant_msg=%d chars (host text %d)" % (
            event_name or "Stop", len(reply), len(text)))
        resp = daemon_call_raw("hook_post_response_track", dict({
            "hook_event_name": event_name,
            "last_assistant_message": reply,
            "session_id": hook_input.get("session_id", ""),
            # The S0 session stamp — what produced this turn (see hook_common).
            # At Stop the transcript's closing message IS this turn's.
            "model": turn_model(hook_input, closing),
            "tells": host_tells(),
        }, **stamp), timeout=4.0)  # inside the manifest's 5 s Stop timeout
        latency = (time.time() - t0) * 1000
        if resp.get("ok"):
            result = resp.get("result", {})
            # Stop self-message backstop: the daemon signals a block to force the
            # turn to continue so a pending tap is seen. On Stop, only a JSON
            # decision reaches the host — plain stdout is invisible in Claude
            # Code and invalid in Codex — so the daemon's `output` is debug-only.
            emit_hook_output(event_name or "Stop", result)
            brain_debug("track: completed in %dms, decision=%s" % (latency, result.get("decision") or "none"))
        else:
            brain_debug("track: daemon returned ok=false")
    else:
        daemon_unavailable_error("post_response_track")


run_hook("post_response_track", main)
