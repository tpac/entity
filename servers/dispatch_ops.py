"""Daemon dispatch — lifecycle / config / admin handlers.

ping, save, session reset, config, the eval-entity
triggers (s2_force / drain_embeddings), the eval escape hatch, and the
one-off revision-history drop.
"""

import json
import os
import time

from .daemon_config import _CODE_FINGERPRINT, _PROCESS_STARTED_AT, REPO_ROOT
from .scales.s1.encode_contract import ENCODE_EVERY


def _handle_ping(brain, args, graph_changes):
    import threading as _t
    result = {"status": "alive", "pid": os.getpid(),
              "code_fingerprint": _CODE_FINGERPRINT,
              "source_dir": REPO_ROOT,
              # Which brain this daemon is actually writing — lets callers
              # detect db-path divergence (daemon_client._db_dir_changed) the
              # way source_dir lets them detect code staleness.
              "db_dir": os.path.dirname(os.path.abspath(brain.db_path)),
              # When this brain became an eval brain, or None: an eval launcher
              # proves its arm is marked instead of assuming it.
              "eval_brain_since": brain.eval_brain_since(),
              # How long THIS PROCESS has been up (not the current serving run —
              # see _PROCESS_STARTED_AT). The dashboard health tab renders it.
              "uptime_seconds": int(time.time() - _PROCESS_STARTED_AT),
              # The Scribe cadence this daemon read at start — the dashboard's
              # encode chip shows it instead of copying the env knob.
              "encode_every": ENCODE_EVERY,
              "threads": _t.active_count()}
    if args.get("thread_detail"):
        result["thread_list"] = [
            {"name": t.name, "daemon": t.daemon, "alive": t.is_alive()}
            for t in _t.enumerate()
        ]
    return {"ok": True, "result": result}


def _handle_get_config(brain, args, graph_changes):
    return {"ok": True, "result": {"value": brain.get_config(
        args.get("key", ""), args.get("default", ""))}}


def _handle_save(brain, args, graph_changes):
    brain.save()
    return {"ok": True, "result": {"status": "saved"}}


def _handle_reset_session(brain, args, graph_changes):
    sid = args.get("session_id", "")
    # cwd is fed from the boot hook (Claude side); the daemon never introspects
    # it. reset_session_activity stamps it as session identity + derives branch,
    # and returns True when it RESUMED an existing session (counters preserved).
    is_resume = brain.reset_session_activity(session_id=sid, cwd=args.get("cwd", ""))
    return {"ok": True, "result": {
        "status": "resumed" if is_resume else "reset",
        "session_id": sid,
    }}


def _handle_set_config(brain, args, graph_changes):
    brain.set_config(args.get("key", ""), args.get("value", ""))
    return {"ok": True, "result": {"status": "set"}}


def _handle_s2_force(brain, args, graph_changes):
    """Run one S2 cycle NOW through Brain.run_s2() — the one door, which owns
    single-flight (a concurrent cycle returns skipped, never queues).

    The deterministic S2 trigger for eval entities: a harness drives encodes
    over MCP/TCP and then fires S2 explicitly instead of re-implementing the
    poll's schedule (the drift class that produced the s2_every_n counter).
    Deliberate act on a live brain — multi-minute LLM run; the poll's idle
    gates are scheduling policy and are intentionally not applied here, and
    the run is forced: a unit that batches work (the aspect classifier's
    batch gate) runs whatever is waiting.

    A completed run stamps the scheduler state the way the poll does, through
    the same owner (brain.mark_s2_ran): without the stamp, last_run_ts still
    reads an hour+ old after a forced cycle and the poll's next tick fires a
    second full cycle over the graph S2 just consolidated. Stamped AFTER
    completion — during the run the poll bails on s2_running — but the encode
    count is snapshotted BEFORE it, so runs finishing during the cycle accrue
    toward the next one instead of being consumed by this one.

    Args (optional):
        units: list of S2 unit NAMEs (e.g. ["aspect_integration"]) — run only
            those, in the coordinator's fixed order. An unknown or empty
            selection fails the command (ValueError) rather than running nothing.

    A partial cycle (the door reports `partial`) does NOT stamp the scheduler.
    Stamping would push the poll's next full cycle out by the min interval and
    consume the encode runs the unrun units still owe work on. The forced unit
    then runs again in that next full cycle, doing whatever work its own gating
    finds by then. Listing every unit is a full cycle and stamps."""
    units = (args or {}).get("units")
    encode_runs = brain.activity.encode_runs_since_maintenance
    outcome = brain.run_s2(units=units, force=True)
    if not outcome.get("skipped") and not outcome.get("partial"):
        brain.mark_s2_ran(time.time(), encode_runs)
    return {"ok": True, "result": outcome}


def _handle_drain_embeddings(brain, args, graph_changes):
    """Run one synchronous embed-worker tick (node/edge vectors, trace
    embeddings, coverage sweep, access marks) via embed_queue.drain_now.

    The deterministic drain for eval entities: after a remember/encode over
    MCP/TCP, everything written is embedded before the next recall — instead
    of the harness re-implementing the background loop (the embedding-drain
    defect class: nodes written but not yet embedded, so invisible to recall)."""
    from servers.embed_queue import drain_now
    return {"ok": True, "result": drain_now(brain)}


def _handle_eval(brain, args, graph_changes):
    """Evaluate a Python expression against the live brain — DEBUG ONLY.

    This is effectively arbitrary code execution: the safe_builtins sandbox is
    weak (the `brain` object alone exposes the full DB, and __builtins__
    restriction is bypassable). Acceptable ONLY because the daemon binds to
    127.0.0.1 and is single-user. Do not expose the daemon on a non-loopback
    interface without gating or removing this handler.
    """
    code = args.get("code", "")
    if not code:
        return {"ok": False, "error": "No code provided"}
    safe_builtins = {"str": str, "int": int, "len": len, "list": list, "dict": dict,
                     "bool": bool, "float": float, "round": round, "sorted": sorted,
                     "min": min, "max": max, "sum": sum, "abs": abs, "type": type,
                     "isinstance": isinstance, "range": range, "enumerate": enumerate,
                     "zip": zip, "map": map, "filter": filter, "print": print,
                     "True": True, "False": False, "None": None}
    local_vars = {"brain": brain, "json": json}
    result = eval(code, {"__builtins__": safe_builtins}, local_vars)
    try:
        json.dumps(result)
    except (TypeError, ValueError):
        result = str(result)
    return {"ok": True, "result": result}


def _handle_drop_sys_revision_history(brain, args, graph_changes):
    """Stage 1A migration: drop legacy _sys_revision_history KV blobs.

    Revision history moved to trace events (event_type='delta',
    ref_type='node_revised'). The legacy _sys_revision_history JSON blob in
    node_metadata_kv is no longer written (see brain_remember.py:revise) and
    is never read by anything (audit confirmed).

    Per Stage 1A spec: drop the data, no retroactive trace conversion.

    Args:
        commit: bool — if False (default), reports count without deleting.

    Returns: {commit, count_found, count_deleted}.
    Idempotent — safe to re-run.
    """
    commit = bool(args.get('commit', False))

    count = brain.conn.execute(
        "SELECT COUNT(*) FROM node_metadata_kv WHERE key = '_sys_revision_history'"
    ).fetchone()[0]

    deleted = 0
    if commit and count > 0:
        cursor = brain.conn.execute(
            "DELETE FROM node_metadata_kv WHERE key = '_sys_revision_history'")
        brain.conn.commit()
        deleted = cursor.rowcount

    graph_changes.append("DROP_REVHISTORY: found=%d deleted=%d (commit=%s)" % (
        count, deleted, commit))

    return {"ok": True, "result": {
        'commit': commit,
        'count_found': count,
        'count_deleted': deleted,
    }}


def _handle_gate(brain, args, graph_changes):
    """The brain's run switch (servers/brain_gate.py) — status / pause /
    resume / ignore / unignore. Writes the user env file, not a brain
    database: is_write=False at the daemon layer. Every action answers with
    the resulting status so the caller sees what the file now says and, for
    each project in args.projects, the gate's verdict for a session there
    (the dashboard asks for its live sessions' projects; the matcher's `!`
    include rule stays here, in its owner)."""
    from .brain_gate import status, set_paused, add_ignore, remove_ignore, verdicts
    action = (args.get("action") or "status").strip()
    pattern = args.get("pattern") or ""
    try:
        if action == "pause":
            set_paused(True)
        elif action == "resume":
            set_paused(False)
        elif action == "ignore":
            add_ignore(pattern)
        elif action == "unignore":
            remove_ignore(pattern)
        elif action != "status":
            return {"ok": False, "error": "gate: unknown action %r" % action}
    except (ValueError, OSError) as e:
        return {"ok": False, "error": "gate: %s" % e}
    result = status()
    result["action"] = action
    projects = args.get("projects")
    if isinstance(projects, list) and projects:
        result["verdicts"] = {p: v._asdict()
                              for p, v in verdicts([str(p) for p in projects]).items()}
    if action == "pause":
        result["note"] = ("paused — hooks in every session go silent from their next "
                          "invocation; this and the other explicit tools keep working")
    return {"ok": True, "result": result}


def _handle_set_api_key(brain, args, graph_changes):
    """The operator's Anthropic key into the user env file — the dashboard's
    /setup page posts here, so the daemon is that file's one writer (the gate
    switches above, this key here). The value is a secret: validated, never
    logged, never echoed back. The daemon re-reads the file on every
    LLM-availability check (scales.dispatch.resolve_api_key), so the key is
    live at the next call with no restart; an existing line is REPLACED —
    the form is explicit operator intent."""
    from .brain_constants import user_env_file, write_env_file_key
    key = (args.get("api_key") or "").strip()
    if not key.startswith("sk-") or not 10 <= len(key) <= 300:
        return {"ok": False, "error": "that doesn't look like an Anthropic API key (expected sk-...)"}
    if any(c.isspace() for c in key):
        return {"ok": False, "error": "the key contains whitespace — check the paste"}
    path = user_env_file()
    try:
        write_env_file_key(path, "ANTHROPIC_API_KEY", key)
    except OSError as e:
        return {"ok": False, "error": "could not write %s: %s" % (path, e.strerror or e)}
    return {"ok": True, "result": {"key_present": True, "env_file": path}}
