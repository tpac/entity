"""SessionEnd — final save + drops this session's cached context in the daemon.
Thin client: sends hook_session_end to daemon.

Does NOT shut the daemon down. The daemon is shared across all concurrent
sessions for this user and is launchd-managed (KeepAlive). Ending one session
must not tear it down under the others — doing so killed in-flight recalls in
sibling sessions (closed-DB / empty-reply / connection-reset) and forced the
next recall onto a cold, slow daemon. Startup and recovery are owned by
launchd (KeepAlive/RunAtLoad, kickstart -k) and the daemon_client ladder —
ensure_daemon at SessionStart, recover_daemon from hooks — with the
maintenance lock as the off-switch for that ladder. Teardown is only ever
explicit: a signal, or the `shutdown` command. No hook stops it.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_call_raw, run_hook

hook_input = get_hook_input()

def main():
    # Codex caps SessionEnd handlers at 3 s and SIGKILLs the process group past
    # it — a killed hook never reaches a loud failure path. One call, budgeted
    # inside that cap: a transport failure is logged by daemon_call_raw itself,
    # while a busy daemon is NOT an outage — no liveness probe here, because a
    # short probe reads a loaded daemon as down and would trigger recovery
    # against a healthy process shared with other live sessions. The
    # daemon-side work (discard_session_context + save) completes once the
    # request is sent, even if this process is killed before the reply.
    daemon_call_raw("hook_session_end", {"session_id": hook_input.get("session_id", "")}, timeout=2.5)

run_hook("session_end", main)
