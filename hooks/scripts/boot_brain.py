"""SessionStart hook — boots brain context via daemon.

Uses daemon for all brain access (single model, single process).
Falls back to direct Brain() ONLY if daemon is completely unavailable.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import db_path, daemon_available, daemon_call, run_hook, get_unsurfaced_hook_errors, mark_hook_errors_surfaced

# Session ID from hook input — extracted by boot-brain.sh, passed as env var.
# Backfilled from CLAUDE_CODE_SESSION_ID when the payload omits it, the same
# backfill hook_common.get_hook_input gives every other hook.
_hook_session_id = (os.environ.get("BRAIN_HOOK_SESSION_ID", "")
                    or os.environ.get("CLAUDE_CODE_SESSION_ID", ""))
# cwd from hook input — fed to the daemon as session identity (cwd/branch live
# on the session object, surfaced in peek). Daemon never introspects it itself.
_hook_cwd = os.environ.get("BRAIN_HOOK_CWD", "")


def _boot_via_daemon():
    """Boot context through daemon — the normal path.

    New-vs-resume is decided by the daemon from the SESSION OBJECT
    (ctx.boot_time), not a global `last_booted_session` brain_meta key. That
    global guard was last-writer-wins across parallel streams: with multiple live
    sessions, every session but the most-recent booter failed the check, so its
    resume was treated as a fresh boot and the daemon zeroed its accumulated
    session state (back when the Scribe cadence was a stored counter, this is what
    starved it). We now always call reset_session; the daemon preserves session
    state on resume and signals it with status='resumed'.
    """
    # Reset/resume session activity with Claude's session_id (not a random UUID).
    # Feed cwd so the daemon can stamp session identity (cwd + derived branch).
    reset_args = {"session_id": _hook_session_id} if _hook_session_id else {}
    if _hook_cwd:
        reset_args["cwd"] = _hook_cwd
    reset_result = daemon_call("reset_session", reset_args)
    if isinstance(reset_result, dict) and reset_result.get("status") == "resumed":
        print("[BRAIN] (session resumed — brain already loaded)", file=sys.stderr)
        return True

    _sid = _hook_session_id

    # Get formatted boot context (retry once if daemon just started).
    # session_id lets the daemon build this session's Frame; the daemon
    # stamps the boot record's project and counterpart from the session.
    boot_args = {"session_id": _sid} if _sid else {}
    result = daemon_call("context_boot", boot_args)
    if not result:
        import time
        time.sleep(1)
        result = daemon_call("context_boot", boot_args)
    if isinstance(result, str) and result:
        print(result)
        return True

    return False


def _boot_via_direct():
    """Fallback: direct Brain() — only if daemon is completely dead."""
    server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
    if server_dir:
        parent = os.path.dirname(server_dir)
        if parent not in sys.path:
            sys.path.insert(0, parent)

    try:
        from servers.brain import Brain
    except ImportError as e:
        print("brain: Failed to import: " + str(e), file=sys.stderr)
        return

    try:
        brain = Brain(db_path)
    except Exception as e:
        print("brain: Failed to init: " + str(e), file=sys.stderr)
        return

    try:
        # Match the daemon path: feed Claude's session_id + cwd so the fallback
        # boot stamps the same session identity (cwd/branch) the daemon would.
        if _hook_session_id:
            brain.reset_session_activity(session_id=_hook_session_id, cwd=_hook_cwd)
        # No session id: with one, render_boot delivers the session's due
        # items (marks them delivered), and this process can be killed by the
        # hook timeout before it prints. Undelivered, they wait for the next
        # daemon boot.
        print(brain.format_boot_context())
    finally:
        brain.close()


# ── Main ──
def main():
    if daemon_available():
        if not _boot_via_daemon():
            print("[brain-boot] Daemon returned empty context, falling back to direct", file=sys.stderr)
            _boot_via_direct()
    else:
        print("[brain-boot] Daemon not available, using direct Brain()", file=sys.stderr)
        _boot_via_direct()


def _surface_unsurfaced_errors():
    """Boot-surfacing (#1): proactively show hook errors that accumulated since
    the last session. Resilient by design — runs even if boot failed, because
    errors matter most exactly when the brain is too broken to boot. Reads the
    hook_errors sink directly (no daemon needed), prints a banner Claude relays
    to the operator, then marks them surfaced. Never breaks boot."""
    try:
        errs = get_unsurfaced_hook_errors(limit=5)
        if not errs:
            return
        lines = ["[BRAIN] ⚠️ %d unsurfaced hook error(s) since last session:" % len(errs)]
        for e in errs:
            lines.append("  • [%s] %s: %s (%s)" % (
                e.get("level", "error"), e.get("hook_name", "?"),
                (e.get("error") or "")[:120], (e.get("created_at") or "")[:19]))
        lines.append("  → full detail in the dashboard errors view.")
        print("\n".join(lines))
        mark_hook_errors_surfaced([e["id"] for e in errs])
    except Exception as e:
        # Must never break boot — but don't swallow silently either (that's the
        # bug this whole change fought). stderr is the last-resort channel when
        # the hook_errors sink itself may be the thing that's failing.
        print("[brain-boot] error-surfacing failed: %s" % e, file=sys.stderr)


run_hook("boot_brain", main)
_surface_unsurfaced_errors()
