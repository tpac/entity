"""Pre-response recall — thin client for the daemon's hook_recall.

The daemon records the operator's turn, runs recall and the Haiku surface, and
formats additionalContext; this script sends the prompt and prints the result,
or nothing at all.
"""
import sys, os, time

_t0 = time.time()
sys.path.insert(0, os.path.dirname(__file__))
from hook_common import (get_hook_input, daemon_available, daemon_call_raw,
                         daemon_unavailable_error, brain_debug, emit_hook_output, run_hook,
                         turn_model, host_tells)
from datetime import datetime as _dt
def _ts(): return _dt.now().strftime("%H:%M:%S.%f")[:-3]
sys.stderr.write("[recall-hook %s] import: %dms\n" % (_ts(), (time.time() - _t0) * 1000))

# Answers with fewer meaningful chars than this register the turn but skip the
# recall + Haiku surface (register_only). A bare answer carries no recall signal.
SHORT_MESSAGE_MAX_LEN = 5

hook_input = get_hook_input()
user_message = hook_input.get("prompt", "")

# Slash / bang / empty (incl. whitespace-only) are genuinely non-conversational —
# slash commands, /watch wakeups, bang. Skip the daemon entirely; they correctly
# read as heartbeats at Stop (no user_message trace, never encoded).
if not user_message.strip() or user_message.startswith("/") or user_message.startswith("!"):
    brain_debug("recall: skipped (slash/bang/empty)")
    sys.exit(0)

# Short real answers ("yes", "ok", "no") ARE conversational but carry no recall
# signal. Register the turn (user_message trace + conversational classification)
# WITHOUT the recall + Haiku surface, via register_only. Skipping them entirely
# would misfile the turn as a heartbeat and lose the operator's words — often
# the highest-signal turns (approvals/decisions).
# Measure stripped length so " ok " counts as 2, not 4. See
# daemon_hooks.hook_recall register-only fast path.
register_only = len(user_message.strip()) < SHORT_MESSAGE_MAX_LEN

# Harness-injected background-task completions arrive through THIS same
# UserPromptSubmit channel — the harness packages a <task-notification> as a
# prompt, so it passes the slash/bang/short gate above and would otherwise run
# the full recall + Haiku surface. That's pure waste here (I'm mid-task with
# full context) AND actively harmful: every surfaced candidate gets marked
# accessed under this session_id, so machine chatter pollutes synaptic fatigue
# and dampens my NEXT real prompt. Route them register_only — keep the
# user_message trace (turn stays conversational, so my substantive response to
# the results is still encoded at Stop) but skip recall + Haiku + fatigue.
# Full-skip (slash/bang path) would misclassify the turn as a heartbeat and
# drop that response from encoding.
if "<task-notification>" in user_message:
    register_only = True


def _first_full_recall_this_session(session_id):
    """True exactly once per session, on its first NON-register_only recall.

    New sessions ride a cold path (Anthropic-side new-session latency: fork
    and fresh sessions timed out at 20s while the daemon finished fine — the
    trace was written, only the injection was lost, 2026-07-27). That first
    call gets a 30s budget; steady state keeps 20s so a hung daemon still
    surfaces fast. Marker file, not daemon state — the timeout must be
    decided before we talk to the daemon at all. Marker creation failure
    degrades to the normal timeout, never blocks the hook.
    """
    if not session_id:
        return False
    import tempfile
    base = os.environ.get('BRAIN_TMP_DIR') or tempfile.gettempdir()
    marker = os.path.join(base, 'brain-hook-first-%s-%s' % (
        os.getuid(), session_id[:8]))
    if os.path.exists(marker):
        return False
    try:
        open(marker, 'w').close()
    except OSError:
        return False
    return True


def main():
    t0 = time.time()
    if not daemon_available():
        # Register-only is best-effort: there's nothing to recall, so a down
        # daemon must fail SILENT — not surface the recall-unavailable banner
        # for a bare "yes". Worst case the turn simply goes unregistered.
        if register_only:
            sys.exit(0)
        emit_hook_output("UserPromptSubmit", {"additionalContext": daemon_unavailable_error("recall")})
        sys.exit(0)

    # Call daemon — it runs recall + the Haiku surface.
    # register_only does no Haiku/recall (just a trace write), so it fails fast
    # rather than carrying the 20s Haiku-tail budget on the prompt path.
    if register_only:
        _recall_timeout = 4.0   # trace write only, no Haiku
    elif _first_full_recall_this_session(hook_input.get("session_id", "")):
        _recall_timeout = 30.0  # new-session cold path (2026-07-27; hooks.json
                                # UserPromptSubmit timeout is 32 to stay above)
    else:
        _recall_timeout = 20.0  # covers Haiku tail latency under load
                                # (node 2340b053).
    resp = daemon_call_raw("hook_recall", {
        "prompt": user_message,
        "session_id": hook_input.get("session_id", ""),
        "register_only": register_only,  # short answers: register turn, skip recall+Haiku
        # The S0 session stamp — what this turn rides on (see hook_common).
        "model": turn_model(hook_input),
        "tells": host_tells(),
    }, timeout=_recall_timeout)

    if not resp.get("ok"):
        # Register-only failure is best-effort too — stay silent rather than
        # surface RECALL FAILED for a turn that had nothing to recall.
        if register_only:
            sys.exit(0)
        err_msg = resp.get("error", "unknown error")
        # daemon_call_raw already logged this failure to hook_errors (single
        # source of truth). We only render the user-facing message here.
        emit_hook_output("UserPromptSubmit", {"additionalContext":
            "[BRAIN]\n⚠️ RECALL FAILED: %s\nThe brain could not search for relevant memories.\n[/BRAIN]" % err_msg})
        sys.exit(0)

    result = resp.get("result", {})
    elapsed = int((time.time() - t0) * 1000)

    # The daemon returns either additionalContext (surface completed) or approve
    # (nothing to surface / surface failed) — emit_hook_output turns approve into silence.
    result_json = result.get("json", {})
    context = result_json.get("additionalContext", "") if isinstance(result_json, dict) else ""
    brain_debug("recall: daemon returned %s in %dms" % (
        "context (%d chars)" % len(context) if context else "no context", elapsed))
    sys.stderr.write("[recall-hook %s] total: %dms\n" % (_ts(), (time.time() - _t0) * 1000))
    emit_hook_output("UserPromptSubmit", result_json)
    os._exit(0)  # Fast exit — skip Python cleanup


run_hook("recall", main)
