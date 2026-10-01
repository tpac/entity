"""Stop hook: records the turn's assistant half (S0 traces) and delivers
pending self-messages. The operator's half is written at UserPromptSubmit.
Thin client: sends hook_post_response_track to daemon.
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import (get_hook_input, daemon_available, daemon_call_raw,
                         daemon_unavailable_error, brain_debug, emit_hook_output, run_hook,
                         turn_model, host_tells)

hook_input = get_hook_input()
event_name = hook_input.get("hook_event_name", "")


def main():
    t0 = time.time()
    if daemon_available():
        last_msg = hook_input.get("last_assistant_message", "")
        brain_debug("track: event=%s, assistant_msg=%d chars" % (
            event_name or "Stop", len(last_msg)))
        resp = daemon_call_raw("hook_post_response_track", {
            "hook_event_name": event_name,
            "last_assistant_message": last_msg,
            "session_id": hook_input.get("session_id", ""),
            # The S0 session stamp — what produced this turn (see hook_common).
            # At Stop the transcript's last assistant entry IS this turn's.
            "model": turn_model(hook_input),
            "tells": host_tells(),
        }, timeout=4.0)  # inside the manifest's 5 s Stop timeout
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
