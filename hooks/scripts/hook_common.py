"""Common setup for all brain hook Python scripts.

Eliminates repeated boilerplate: path setup, Brain import, input parsing,
daemon connection helpers, error logging. Every hook .py file imports this.
"""
import sys, os, re, json, traceback, sqlite3
from datetime import datetime, timezone

# ── Path setup ──
server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db") if db_dir else ""

if server_dir:
    parent = os.path.dirname(server_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)
else:
    # BRAIN_SERVER_DIR is exported by resolve-brain-db.sh, which only the
    # `bash <script>.sh` hooks source — a hook invoked as a bare `python3
    # <script>.py` inherits none of it, and `import servers.*` then raises
    # ModuleNotFoundError inside helpers that swallow exceptions. Derive the
    # plugin root from this file's own location instead of trusting the env:
    # hooks/scripts/hook_common.py → <plugin root>.
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.isdir(os.path.join(_root, "servers")) and _root not in sys.path:
        sys.path.insert(0, _root)


def _get_hook_name():
    """Infer the calling hook name from the call stack."""
    import inspect
    for frame_info in inspect.stack():
        fname = os.path.basename(frame_info.filename)
        if fname.endswith(".py") and fname != "hook_common.py":
            return fname.replace(".py", "")
    return "unknown_hook"


# ── Debug mode ──
_debug_mode_cache = None


def is_debug_mode():
    """Check if brain debug mode is on.

    Resolution order:
      1. BRAIN_DEBUG env var (when the operator exports it)
      2. brain_meta.debug_enabled in brain.db (persistent config)

    Debug mode shows all brain activity to Claude: recalls, injections,
    errors, encoding, telemetry. Toggle with brain.set_config('debug_enabled', '1'|'0').
    """
    global _debug_mode_cache
    if _debug_mode_cache is not None:
        return _debug_mode_cache

    # Env var (fast path)
    env = os.environ.get("BRAIN_DEBUG", "")
    if env:
        _debug_mode_cache = env == "1"
        return _debug_mode_cache

    # Read from brain_meta
    if db_path and os.path.isfile(db_path):
        try:
            conn = sqlite3.connect(db_path, timeout=10)
            row = conn.execute(
                "SELECT value FROM brain_meta WHERE key = 'debug_enabled'"
            ).fetchone()
            conn.close()
            _debug_mode_cache = row is not None and row[0] == "1"
            return _debug_mode_cache
        except Exception:
            pass

    _debug_mode_cache = False
    return False


def brain_debug(msg):
    """Log hook activity to brain_logs.db for later inspection.

    Rows go to the debug_log table with event_type='hook_debug'. They are
    forensic — read on demand via the `query_logs` MCP tool (source='debug'),
    not pushed into Claude's context. The dashboard's Logs tab filters to
    event_type IN ('error','warning'), so these rows do not appear there.

    Only logs when debug mode is on. Falls back to stderr if DB write fails.
    """
    if not is_debug_mode():
        return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    source = _get_hook_name()
    logs_db = os.path.join(db_dir, "brain_logs.db") if db_dir else ""
    if logs_db and os.path.isdir(db_dir):
        try:
            conn = sqlite3.connect(logs_db, timeout=10)
            conn.execute(
                "INSERT INTO debug_log (session_id, event_type, source, metadata, created_at) "
                "VALUES (?, 'hook_debug', ?, ?, ?)",
                ('current', source, json.dumps({"message": msg}), ts),
            )
            conn.commit()
            conn.close()
            return
        except Exception:
            pass
    # Fallback: stderr (won't reach Claude but at least not lost)
    print("[BRAIN DEBUG] %s" % msg, file=sys.stderr)


def brain_error(msg):
    """Print error visible to Claude. Always prints (not gated by debug).

    Errors should never be silent — this is the lesson from 1,740 blind recalls.
    """
    print("[BRAIN ERROR] %s" % msg, file=sys.stderr)


def log_hook_error(source, error, context="", level="error"):
    """Log a hook error to brain_logs.db AND stderr.

    This is the ONLY place hook errors should be logged. Never swallow silently.
    Uses direct SQLite (not Brain) so it works even when Brain fails to import.

    Not every caller is inside a live `except`: a failure handed over as a
    RETURN VALUE (daemon_client.send_command's transport classes, the
    daemon-down handlers) has no sys.exc_info() to read, and those rows used to
    persist an empty traceback column. Falling back to format_stack() records
    the CALLER chain — which is the part worth reading anyway; the exception's
    own frames are usually one socket call the error string already names.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if sys.exc_info()[2]:
        tb = traceback.format_exc()
    else:
        # INNERMOST frames, not the whole stack: format_stack() is oldest-first,
        # so the column's 2000-char truncation would keep the interpreter's
        # bootstrap and cut the frames that name what actually failed. -1 drops
        # this function.
        tb = "".join(traceback.format_stack()[-9:-1])
    msg = "hook_error [%s] %s: %s" % (source, error, context)

    # Always print to stderr — never silent
    print("brain: %s" % msg, file=sys.stderr)
    if tb and tb.strip() != "NoneType: None":
        print("  traceback: %s" % tb.strip()[:500], file=sys.stderr)

    # Try to log to brain_logs.db
    logs_db = os.path.join(db_dir, "brain_logs.db") if db_dir else ""
    if logs_db and os.path.isdir(db_dir):
        try:
            conn = sqlite3.connect(logs_db, timeout=10)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hook_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    hook_name TEXT NOT NULL,
                    level TEXT NOT NULL DEFAULT 'error',
                    error TEXT NOT NULL,
                    context TEXT DEFAULT '',
                    traceback TEXT DEFAULT '',
                    surfaced INTEGER DEFAULT 0
                )
            """)
            conn.execute(
                "INSERT INTO hook_errors (created_at, hook_name, level, error, context, traceback) VALUES (?, ?, ?, ?, ?, ?)",
                (ts, source, level, str(error), context[:500], tb[:2000]),
            )
            # Prune to the newest 400: schema.LOG_TABLES['hook_errors']['keep_rows'],
            # repeated because this path must work when servers/ cannot import.
            conn.execute("DELETE FROM hook_errors WHERE id NOT IN (SELECT id FROM hook_errors ORDER BY id DESC LIMIT 400)")
            conn.commit()
            conn.close()
        except Exception:
            pass  # Last resort — stderr was already printed


def run_hook(name, fn, on_error=None):
    """Single error boundary for a hook script — the standard every hook runs through.

    Runs fn() (the hook body). If it raises, logs ONCE to hook_errors (via
    log_hook_error, the canonical hook-side sink) and invokes on_error() for any
    fail-safe output the hook still owes. Most hooks owe none: exit 0 with no
    stdout reads as "nothing to report" on every host (see emit_hook_output),
    so a crashed PreToolUse hook simply doesn't block the tool.

    Catches Exception only — SystemExit/KeyboardInterrupt propagate, so a hook's
    own `sys.exit(0)` skip-path (and Ctrl-C) still work. Never re-raises: a
    crashing hook must not break the host. Does NOT impose an exit code — each
    hook owns its own output contract (approve/block/additionalContext/none),
    which is why output lives in fn()/on_error(), not here.
    """
    try:
        fn()
    except Exception as e:
        log_hook_error(name, e, "hook exception")
        if on_error is not None:
            try:
                on_error()
            except Exception as e2:
                log_hook_error(name, e2, "hook on_error fallback also failed")


def get_unsurfaced_hook_errors(limit=10):
    """Read unsurfaced hook errors from brain_logs.db. Returns list of dicts."""
    logs_db = os.path.join(db_dir, "brain_logs.db") if db_dir else ""
    if not logs_db or not os.path.isfile(logs_db):
        return []
    try:
        conn = sqlite3.connect(logs_db, timeout=10)
        # Check if table exists
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='hook_errors'").fetchall()
        if not tables:
            conn.close()
            return []
        rows = conn.execute(
            "SELECT id, created_at, hook_name, level, error, context FROM hook_errors WHERE surfaced = 0 ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [{"id": r[0], "created_at": r[1], "hook_name": r[2], "level": r[3], "error": r[4], "context": r[5]} for r in rows]
    except Exception:
        return []


def mark_hook_errors_surfaced(error_ids):
    """Mark hook errors as surfaced so they are not shown again."""
    logs_db = os.path.join(db_dir, "brain_logs.db") if db_dir else ""
    if not logs_db or not error_ids:
        return
    try:
        conn = sqlite3.connect(logs_db, timeout=10)
        placeholders = ",".join("?" * len(error_ids))
        conn.execute("UPDATE hook_errors SET surfaced = 1 WHERE id IN (%s)" % placeholders, error_ids)
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_hook_input():
    """Parse HOOK_INPUT from environment (set by bash shim).

    Backfills `session_id` from CLAUDE_CODE_SESSION_ID (the env var Claude Code
    exports in every hook process) when the payload omits it. Some events —
    notably WorktreeCreate — don't carry session_id in their JSON, and the daemon
    is a separate launchd process that can't read the hook's env; without this the
    daemon would fall back to the last-writer-wins singleton and mis-attribute
    per-session state to the wrong stream."""
    try:
        data = json.loads(os.environ.get("HOOK_INPUT", "{}"))
    except Exception as e:
        log_hook_error(_get_hook_name(), e, "Failed to parse HOOK_INPUT", level="warning")
        return {}
    if isinstance(data, dict) and not data.get("session_id"):
        env_sid = os.environ.get("CLAUDE_CODE_SESSION_ID", "")
        if env_sid:
            data["session_id"] = env_sid
    return data


# ── Host + model (the S0 session stamp) ──
# What a turn rides on, fed to the daemon by the UserPromptSubmit / Stop hooks
# and stamped onto the turn's S0 rows (trace_contract.S0_SESSION_STAMP_FIELDS).
# Two hosts, two sources: Codex puts `model` on every hook payload; Claude Code
# does not, but every assistant entry in the transcript carries message.model.

# Zero-import mirror of host_contract.all_tell_env_vars(). TestHookMirrors
# requires exact equality: the hot tool hook must not import servers to probe.
HOST_TELL_ENV_VARS = (
    'CLAUDE_CODE_SESSION_ID', 'CODEX_INTERNAL_ORIGINATOR_OVERRIDE', 'PLUGIN_DATA',
)


def host_tells():
    """Declared environment tells present in this process — names, no values."""
    return [name for name in HOST_TELL_ENV_VARS if os.environ.get(name)]


# Tail window for the transcript read: assistant entries land on every tool
# call, so the last one is always within a few KB of the end — a full
# readlines() over a session-long transcript (tens of MB) on the prompt hot
# path is the cost this avoids.
_TRANSCRIPT_TAIL_BYTES = 256 * 1024


def turn_model(hook_input):
    """The model this turn ran on, or '' when the payload doesn't say.

    Codex: the payload's `model`. Claude Code: the LAST main-thread assistant
    entry in the transcript — at Stop that is this turn's model; at
    UserPromptSubmit it is the previous turn's (the model the operator was
    talking to when the prompt arrived; '' on a session's first prompt, and the
    Stop hook fills it). Sidechain entries (subagents, which may run on a
    different model) are skipped — they are not the operator's turn.

    The tail window is the normal read; when it holds no main-thread assistant
    entry (one oversized tool result, or a subagent's sidechain, can push the
    last one out of the window) the whole file is scanned once — rare, and
    the alternative is an unstamped turn plus a spurious model-unset error."""
    m = hook_input.get("model")
    if isinstance(m, str) and m.strip():
        return m.strip()
    path = hook_input.get("transcript_path")
    if not path:
        return ""
    path = os.path.expanduser(path)
    try:
        size = os.path.getsize(path)
        model = _last_main_assistant_model(path, max(0, size - _TRANSCRIPT_TAIL_BYTES))
        if not model and size > _TRANSCRIPT_TAIL_BYTES:
            model = _last_main_assistant_model(path, 0)
        return model
    except OSError:
        return ""


def _last_main_assistant_model(path, start):
    """message.model of the last non-sidechain assistant entry at or after byte
    `start` of the transcript, or ''. Raises OSError on a read failure."""
    with open(path, "rb") as f:
        f.seek(start)
        lines = f.read().decode("utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except ValueError:
            continue   # the first line of a tail window is usually a partial record
        if not isinstance(entry, dict) or entry.get("type") != "assistant" \
                or entry.get("isSidechain"):
            continue
        msg = entry.get("message")
        model = msg.get("model") if isinstance(msg, dict) else None
        if isinstance(model, str) and model.strip():
            return model.strip()
    return ""


# ── Host tool-input shapes ──
# Codex's file tool is apply_patch: its PreToolUse/PostToolUse input is
# {"command": <patch text>} with no file_path — the target sits in the patch
# header. Claude Code sends file_path directly. One resolver for every hook that
# keys on the edited file, so the two hosts never diverge per script.
_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.M)


def tool_target_file(tool_input):
    """The file an Edit/Write/apply_patch call targets, or "" when there is none.
    A multi-file patch resolves to its first file — the daemon's rule lookup is
    single-file."""
    if not isinstance(tool_input, dict):
        return ""
    path = tool_input.get("file_path", "")
    if path:
        return path
    m = _PATCH_FILE_RE.search(tool_input.get("command", "") or "")
    return m.group(1).strip() if m else ""


# ── Hook stdout contract ──
# The ONLY writers of a hook script's stdout live here — emit_hook_output for
# what the brain has to SAY, emit_updated_input for the one tool input it
# REWRITES. Two hosts read it — Claude Code and Codex — against strict per-event
# JSON schemas, so the daemon protocol ({decision, reason} / {additionalContext})
# is translated to the host wire in exactly one place. The brain informs, it
# never gates: a block is honored only on Stop, where it is how a pending
# self-message is delivered (the host continues the turn with the reason as its
# next prompt). Printing `{"decision":"approve"}` is never right — Codex rejects
# it as invalid output (hook run FAILED), and on Claude Code it meant "allow,
# skip the permission prompt", a decision a memory plugin must not make for its
# user.

_BLOCKING_EVENTS = ("Stop",)


def _as_text(value):
    """Coerce a payload field to text. The daemon sends strings; anything else
    is a daemon-side bug we still want to SEE (as text), not a crash that
    swallows the whole output."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return value.strip()


def emit_hook_output(hook_event_name, payload):
    """Write this hook's stdout for `hook_event_name` from a decision payload.

    `payload` keys (any subset, or None): `decision` ("approve" | "block"),
    `reason`, `additionalContext`.

      nothing / approve with no text       → no output ("no opinion" on both hosts)
      additionalContext, or approve+reason  → {"hookSpecificOutput": {…, "additionalContext"}}
                                             (Stop: Codex's Stop schema has no
                                             context channel — dropped, logged)
      block on Stop                         → {"decision": "block", "reason"}
      block on any other event, an unknown  → logged to hook_errors (warning);
      decision, a non-dict payload            the text is still emitted as context

    Both hosts reject a block without a non-empty reason, so one is supplied.
    Flushes stdout so a caller may os._exit() right after.
    """
    def _warn(msg):
        log_hook_error(_get_hook_name(), msg, "emit_hook_output(%s)" % hook_event_name, level="warning")

    if payload is not None and not isinstance(payload, dict):
        _warn("non-dict hook payload (%s) — nothing to emit" % type(payload).__name__)
        payload = {}
    payload = payload or {}
    decision = payload.get("decision")
    reason = _as_text(payload.get("reason"))
    context = _as_text(payload.get("additionalContext"))
    if decision and decision not in ("approve", "block"):
        _warn("unknown hook decision %r — emitted as context, not a decision" % (decision,))
        decision = None
    if decision == "block" and hook_event_name not in _BLOCKING_EVENTS:
        _warn("block on %s — the brain informs, it never gates; emitted as context" % hook_event_name)
        decision = None
    out = None
    if decision == "block":
        out = {"decision": "block", "reason": reason or "blocked by a brain hook"}
    else:
        text = context or reason
        if text:
            if hook_event_name == "Stop":
                _warn("Stop context dropped (%d chars): Codex's Stop schema has no additionalContext channel" % len(text))
            else:
                out = {"hookSpecificOutput": {
                    "hookEventName": hook_event_name,
                    "additionalContext": text,
                }}
    if out is not None:
        sys.stdout.write(json.dumps(out))
        sys.stdout.flush()


def strip_caller_stamp(tool_input):
    """A tool input without the proxy-bound identity pair. The stamp hook
    rewrites the brain's tool arguments with `_caller_session` + `_caller_sig`
    (servers.dispatch_common owns both keys); anything that RECORDS a tool
    input — the PostToolUse trace — must drop them first, or a valid pair lands
    in a trace the model can recall and replay as another stream's identity.
    Keyed on the shared `_caller_` prefix rather than an import: this runs on
    every tool call."""
    if not isinstance(tool_input, dict):
        return tool_input
    return {k: v for k, v in tool_input.items() if not k.startswith("_caller_")}


def emit_updated_input(hook_event_name, tool_name, updated_input):
    """PreToolUse only, brain tools only: hand the host a rewritten tool input.

    Both hosts apply `permissionDecision: "allow"` + `updatedInput` the same
    way — the object REPLACES the tool's arguments before the call runs — and
    neither applies the rewrite without the `allow`. That `allow` is the brain
    permitting its OWN MCP tools, the one tool family a memory plugin may
    approve for itself, so this writer refuses any other `tool_name`
    (servers.dispatch_common.is_brain_tool) — a widened matcher cannot turn it
    into an auto-approve of a user's tool. Any other event, tool, or a non-dict
    input emits nothing and is logged. Flushes stdout so a caller may os._exit()
    right after.
    """
    def _warn(msg):
        log_hook_error(_get_hook_name(), msg, "emit_updated_input(%s)" % hook_event_name, level="warning")

    if hook_event_name != "PreToolUse":
        _warn("updatedInput is a PreToolUse-only shape — nothing emitted")
        return
    from servers.dispatch_common import is_brain_tool
    if not is_brain_tool(tool_name):
        _warn("updatedInput refused for %r — the brain rewrites only its own MCP tools; "
              "check the PreToolUse matcher" % (tool_name,))
        return
    if not isinstance(updated_input, dict):
        _warn("non-dict updatedInput (%s) — nothing emitted" % type(updated_input).__name__)
        return
    sys.stdout.write(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "updatedInput": updated_input,
    }}))
    sys.stdout.flush()


def daemon_unavailable_error(hook_name=None):
    """Return a CRITICAL error when daemon is unreachable.

    Does three things:
    1. Returns CRITICAL additionalContext that Claude MUST relay to user
    2. Logs to dashboard DB directly (no daemon needed)
    3. Attempts to restart daemon via ensure_daemon()

    Exception — unconfigured install (no brain.db anywhere AND no daemon):
    that is not a crash, it's the brain never having come up. Logs BRAIN OFFLINE
    and exits 1 (real, non-blocking hook error) WITHOUT attempting recovery —
    boot creates brains, recovery must not, and recover-storms on machines
    with nothing to recover are how first installs get flaky.
    """
    name = hook_name or _get_hook_name()

    if not (db_path and os.path.isfile(db_path)):
        log_hook_error(
            name,
            "BRAIN OFFLINE - no brain.db at %s and the daemon is not answering"
            % (db_dir or "unresolved"),
            "plugin enabled but the brain never came up: recall+encode+traces dead. "
            "Likely causes in order: stale BRAIN_DB_DIR override in "
            "~/.config/brain/env; first install interrupted; daemon crash "
            "before bind. (A missing ANTHROPIC_API_KEY does NOT cause this — "
            "boot creates the brain and daemon without one.)",
            level="critical")
        sys.exit(1)

    # 1. Format CRITICAL message for Claude
    msg = (
        "[CRITICAL — RELAY TO USER IMMEDIATELY]\n"
        "⚠️ Brain daemon is unreachable. Hook: %s.\n"
        "Brain memories are safe (on disk). Recall and encoding are disabled.\n"
        "Attempting automatic restart.\n"
        "DO NOT continue without informing the user.\n"
        "[/CRITICAL]"
    ) % name

    print("[brain] CRITICAL: %s — daemon unavailable" % name, file=sys.stderr)

    # 2. Persist to brain_logs.db directly (SQLite, no daemon needed) so the
    #    outage shows in the dashboard + query_logs even while the daemon is
    #    down.
    try:
        log_hook_error("DAEMON_DOWN", "daemon unreachable (detected by %s)" % name,
                       "recall + encoding disabled until restart; relayed to operator",
                       level="critical")
    except Exception:
        pass

    # 3. A hung daemon keeps the port bound, so launchd's crash-respawn never
    # fires and the daemon can't SIGTERM itself. Hand off to the single
    # recovery path — it kills the corpse + lets launchd respawn, guarded by
    # maintenance-lock + cooldown + circuit breaker. (The MCP health monitor
    # calls the same function; shared state keeps them from fighting.)
    try:
        from servers.daemon_client import recover_daemon
        recover_daemon(db_path)
    except Exception:
        pass

    return msg


# ── Daemon helpers ──
# No host/port here: servers.daemon_config owns the address (env-first, formula
# as fallback) and daemon_client owns the wire.


def daemon_available(timeout=2.0):
    """Liveness check — True only if the daemon answers a ping.

    Delegates to daemon_client.is_daemon_responsive(), the single source of
    truth. A hung "corpse" that holds the port but never replies reads as
    unavailable here, so recovery can fire; a slow-but-alive daemon still
    answers and reads as available."""
    try:
        from servers.daemon_client import is_daemon_responsive
        return is_daemon_responsive(timeout=timeout)
    except Exception:
        return False


def daemon_call(cmd, args=None, timeout=10.0):
    """Send a command to the daemon and return the result.
    Returns the result dict on success, empty dict on failure.

    The quiet sibling of daemon_call_raw: same wire (daemon_client owns it),
    no logging — for callers that treat a daemon-down as a non-event.
    """
    try:
        from servers.daemon_client import send_command
        resp = send_command(cmd, args, timeout=timeout)
        return resp.get("result", {}) if resp.get("ok") else {}
    except Exception:
        return {}


def daemon_call_raw(cmd, args=None, timeout=10.0):
    """Send a command to the daemon and return the full response.
    Returns the raw response dict including 'ok' field.

    The WIRE belongs to daemon_client.send_command; this wrapper owns the
    hook-side OBSERVABILITY — the [BRAIN ERROR] line Claude sees and the
    hook_errors row the dashboard and boot read. Failure classes come from
    send_command's `transport` key, never from matching its prose.

    On error: always prints [BRAIN ERROR] to stderr so Claude sees it.
    """
    try:
        from servers.daemon_client import send_command
        resp = send_command(cmd, args, timeout=timeout)
    except Exception as e:
        # The client module itself is unreachable (import/path failure) — a
        # hook must degrade, never crash the session.
        brain_error("%s: %s" % (cmd, e))
        log_hook_error(cmd, e, "daemon_call_raw failed")
        return {"ok": False, "error": str(e)}

    transport = resp.get("transport")
    if transport == "timeout":
        brain_error("%s timed out after %.0fs" % (cmd, timeout))
        log_hook_error(cmd, "timeout", "%.0fs" % timeout)
        return {"ok": False, "error": "timeout"}
    if transport == "refused":
        brain_error("%s: daemon not running" % cmd)
        return {"ok": False, "error": "daemon not running"}
    if transport:
        # Empty read, garbled JSON, socket reset. Logged, not just returned:
        # a failure that reaches Claude but never reaches hook_errors is
        # invisible in the dashboard and at boot.
        err = resp.get("error", "unknown error")
        brain_error("%s: %s" % (cmd, err))
        log_hook_error(cmd, err, "daemon_call_raw failed")
        return {"ok": False, "error": err}

    if not resp.get("ok"):
        err = resp.get("error", "unknown error")
        brain_error("%s failed: %s" % (cmd, err))
        log_hook_error(cmd, err, "daemon returned ok=false")

    return resp
