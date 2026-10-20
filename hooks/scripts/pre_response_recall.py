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
                         turn_model, host_tells, log_hook_error)
from datetime import datetime as _dt
def _ts(): return _dt.now().strftime("%H:%M:%S.%f")[:-3]
sys.stderr.write("[recall-hook %s] import: %dms\n" % (_ts(), (time.time() - _t0) * 1000))

hook_input = get_hook_input()
user_message = hook_input.get("prompt", "")

# WHAT of a prompt gets in, who each piece is from and what recall reads is
# the host contract's to say (host_contract.read_prompt) — the hook only asks,
# inside main() so a gated session never pays the contract import. A prompt
# with no pieces (a repeating command such as a /watch re-arm, a shell escape,
# empty, the session's last command again) skips the daemon entirely and reads
# as a heartbeat at Stop; a skill the operator types is a prompt like any
# other. A prompt with nothing to recall
# on — a short answer, a harness notice — is still registered (register_only):
# skipping it would misfile the turn as a heartbeat and drop my reply from
# encoding, while recalling on it would mark every surfaced candidate accessed
# and dampen the next real prompt.


def _session_marker(session_id, moment):
    """This session's marker file for `moment`, or '' without a session id.
    Marker file, not daemon state: every moment is decided before we talk
    to the daemon at all."""
    from servers.brain_constants import session_marker_path
    return session_marker_path('hook-' + moment, session_id)


def _mark(marker):
    """Create `marker`; False when it cannot be written, which degrades to
    the normal path and never blocks the hook."""
    try:
        open(marker, 'w').close()
    except OSError:
        return False
    return True


def _last_command(session_id):
    """The session's last slash command — read_prompt's `last_command` — or
    '' when none is kept."""
    marker = _session_marker(session_id, 'command')
    try:
        with open(marker) as f:
            return f.read().strip()
    except OSError:
        return ''


def _keep_last_command(session_id, intake):
    """Keep the command this prompt opened with, or forget the last one when
    the operator spoke without one. A prompt of the harness's alone (a task
    notification) leaves it, so a scheduled command stays one turn even when
    its own background work reports in between. A failed write is logged:
    every repeat would then be recorded again."""
    marker = _session_marker(session_id, 'command')
    if not marker or not (intake.command or any(p[0] == 'operator' for p in intake.pieces)):
        return
    try:
        with open(marker, 'w') as f:
            f.write(intake.command)
    except OSError as e:
        log_hook_error('recall', e, 'could not keep the last command for the repeat check')


def _first_full_recall_this_session(session_id):
    """True exactly once per session, on its first NON-register_only recall.

    New sessions ride a cold path (Anthropic-side new-session latency: fork
    and fresh sessions timed out at 20s while the daemon finished fine — the
    trace was written, only the injection was lost, 2026-07-27). That first
    call gets a 30s budget; steady state keeps 20s so a hung daemon still
    surfaces fast.
    """
    marker = _session_marker(session_id, 'first')
    return bool(marker) and not os.path.exists(marker) and _mark(marker)


# How long the reply waits for the setup check after the daemon answers. The
# check runs beside the daemon call and normally finishes first (brain node
# b72a3bd2); this bounds a slow Codex.
SETUP_NOTICE_WAIT_S = 3.0


def _start_setup_check(host):
    """Start the host's setup check for the first operator prompt of a chat;
    returns the getter for its notice ('' when there is none).

    The adapter the host contract names (hooks/adapters/) asks the host whether
    Entity's hooks await approval. This hook carries the answer because it is
    running: a hook awaiting approval cannot report itself, and nothing here
    depends on the model noticing an absence (brain node 941811c5).
    """
    from servers.host_contract import HOST_CONTRACT
    adapter = (HOST_CONTRACT.get(host) or {}).get('adapter')
    marker = _session_marker(hook_input.get("session_id", ""), 'setup')
    if not adapter or not marker or os.path.exists(marker):
        return lambda: ''
    import importlib, threading
    # This install's root: the adapter must judge the hooks the host loaded
    # from here, whichever tree BRAIN_SERVER_DIR names.
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    found = []

    def check():
        try:
            adapters = os.path.join(root, 'hooks', 'adapters')
            if adapters not in sys.path:
                sys.path.insert(0, adapters)
            module = importlib.import_module(os.path.splitext(adapter)[0])
            found.append(module.prompt_notice(root))
        except Exception as e:
            log_hook_error("recall", e, "setup check (%s)" % adapter, level="warning")

    worker = threading.Thread(target=check, daemon=True)
    worker.start()

    def notice():
        worker.join(SETUP_NOTICE_WAIT_S)
        if worker.is_alive():
            return ''  # Codex is still answering: the next prompt asks again
        _mark(marker)  # answered or failed (logged): once per chat
        return found[0] if found else ''
    return notice


def _emit(payload, notice):
    """Emit the recall payload, with the setup notice after its context."""
    text = notice()
    if text and (payload is None or isinstance(payload, dict)):
        payload = dict(payload or {})
        context = payload.get("additionalContext") or ""
        payload["additionalContext"] = context + "\n\n" + text if context else text
    emit_hook_output("UserPromptSubmit", payload)


def main():
    t0 = time.time()
    from servers.host_contract import (resolve_host, read_prompt, read_turn,
                                       session_unattended, contract_fingerprint,
                                       VOCAB_VERSION)
    from servers.trace_contract import HOST_TURN_FIELD
    host = resolve_host({t: True for t in host_tells()})[0]
    session_id = hook_input.get("session_id", "")
    intake = read_prompt(user_message, host, session_unattended(host, os.environ),
                         _last_command(session_id))
    _keep_last_command(session_id, intake)
    if not intake.pieces:
        brain_debug("recall: skipped (repeating or repeated command/shell escape/empty)")
        sys.exit(0)
    register_only = not intake.recall_text
    # Only the operator can act on the setup notice; a prompt that holds no
    # operator text leaves it for the next one that does.
    speaks = intake.pieces[0][0] == 'operator'
    notice = _start_setup_check(host) if speaks else (lambda: '')
    if not daemon_available():
        # Register-only is best-effort: there's nothing to recall, so a down
        # daemon must fail SILENT — not surface the recall-unavailable banner
        # for a bare "yes". Worst case the turn simply goes unregistered.
        if register_only:
            _emit({}, notice)
            sys.exit(0)
        _emit({"additionalContext": daemon_unavailable_error("recall")}, notice)
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
        "register_only": register_only,  # nothing to recall on: register the turn, skip recall+Haiku
        # The contract's reading of the prompt: the rows the daemon writes,
        # what recall reads, and which contract read it (stamped on each row).
        "pieces": [list(p) for p in intake.pieces],
        "recall_text": intake.recall_text,
        "vocab_version": VOCAB_VERSION,
        "impl_identity": contract_fingerprint(),
        # The host's name for the turn this prompt arrived in — a message
        # absorbed mid-turn carries the running turn's (host_contract.read_turn).
        HOST_TURN_FIELD: read_turn(hook_input, host),
        # The S0 session stamp — what this turn rides on (see hook_common).
        "model": turn_model(hook_input),
        "tells": host_tells(),
    }, timeout=_recall_timeout)

    if not resp.get("ok"):
        # Register-only failure is best-effort too — stay silent rather than
        # surface RECALL FAILED for a turn that had nothing to recall.
        if register_only:
            _emit({}, notice)
            sys.exit(0)
        err_msg = resp.get("error", "unknown error")
        # daemon_call_raw already logged this failure to hook_errors (single
        # source of truth). We only render the user-facing message here.
        _emit({"additionalContext":
            "[BRAIN]\n⚠️ RECALL FAILED: %s\nThe brain could not search for relevant memories.\n[/BRAIN]" % err_msg}, notice)
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
    _emit(result_json, notice)
    os._exit(0)  # Fast exit — skip Python cleanup


# The prompt reader or the recall path raised: run_hook has logged it (hook
# errors + stderr). Say so in the session too — otherwise this prompt simply
# goes unrecorded and unrecalled with nothing anyone sees.
READER_FAILED = ("[BRAIN]\n⚠️ PROMPT INTAKE FAILED: the brain could not read this prompt, "
                 "so it was not recorded or recalled on. The error is in the brain's "
                 "hook errors (dashboard → Logs → Hook).\n[/BRAIN]")


def _on_error():
    emit_hook_output("UserPromptSubmit", {"additionalContext": READER_FAILED})


run_hook("recall", main, on_error=_on_error)
