"""
brain — Daemon Client

Client-side functions for communicating with the brain daemon.
Used by hook scripts and brain_mcp.py.

Restarter coordination: ensure_daemon() serializes concurrent callers on the
STARTUP lock (get_startup_lock_path) — first caller converges the daemon,
the rest block and wake up to a running one. The daemon's own singleton
identity is a DIFFERENT lock (get_lock_path), held by a serving daemon for
its whole life; the client only ever probes it non-blockingly (the
two-writer guard before a direct spawn). One inode per question.
"""

import fcntl
import json
import os
import socket
import sys
import time
from typing import Any, Dict, Optional

from .daemon_config import (
    _CODE_FINGERPRINT, _IS_WORKTREE, REPO_ROOT, DAEMON_HOST, CLIENT_STAMP_KEY,
    get_daemon_addr, get_pid_path, get_lock_path,
    get_startup_lock_path,
    get_recovery_state_path, is_maintenance_mode, resolve_db_dir,
    LAUNCHD_THROTTLE_INTERVAL_S,
)
from .daemon_launch import (
    kickstart, kill_daemon, manages, pid_file_age_s, port_is_occupied,
    release_supersedes, spawn_detached_daemon,
)


def send_command(cmd: str, args: Optional[Dict[str, Any]] = None,
                 timeout: float = 10.0, port: Optional[int] = None,
                 role: Optional[str] = None) -> Dict[str, Any]:
    """Send a command to the running daemon via TCP. Returns response dict.

    The ONE implementation of the daemon's wire protocol (connect, JSON +
    newline, read to the newline, parse) — hooks, the MCP server and the
    restart CLI all route through it rather than hand-rolling a socket.

    A failure of the WIRE carries a `transport` key classifying it:
    `timeout`, `refused`, `reset`, `empty` (accepted then closed without
    replying) or `protocol` (garbled read, parse failure). `refused` and
    `reset` mean the daemon never read the request: nothing listened, or the
    request sat unread in the listen queue when the socket closed (a reload or
    shutdown), and a socket closed with unread data resets. A daemon that read
    the request and then closed reads as `empty`. Callers that must react
    differently per class read that key — never the prose in `error`, which is
    for humans and free to change. It is absent on success AND on a
    daemon-level ok=false, which is the daemon answering, not the wire failing.

    `port` addresses a daemon other than this process's own: DAEMON_PORT is
    fixed from the environment at import, so one process driving several eval
    arms names each arm's port. The host stays DAEMON_HOST (loopback).

    `role` (CLIENT_ROLE_HOOK / CLIENT_ROLE_MCP) stamps the request with this
    copy's build identity — role, code fingerprint, root — beside cmd/args, for
    the daemon's stale-copy check (dispatch_common.check_client_code). Callers
    that are not an installed copy's hook or proxy pass none.
    """
    addr = (DAEMON_HOST, port) if port is not None else get_daemon_addr()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect(addr)
        req = {"cmd": cmd, "args": args or {}}
        if role:
            req[CLIENT_STAMP_KEY] = {"role": role, "fp": _CODE_FINGERPRINT,
                                     "root": REPO_ROOT}
        msg = json.dumps(req) + "\n"
        sock.sendall(msg.encode('utf-8'))

        # Read to the newline the daemon terminates every reply with.
        # Accumulate in a list: `data += chunk` reallocates the whole buffer
        # each iteration, which is quadratic on the multi-megabyte replies the
        # graph queries return.
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break

        text = b"".join(chunks).decode('utf-8').strip()
        if not text:
            # Accepted the connection, then closed it without writing. Named
            # explicitly because letting json.loads("") throw yields the
            # cryptic "Expecting value: line 1 column 1 (char 0)".
            return {"ok": False, "transport": "empty",
                    "error": "daemon returned empty response "
                             "(connection closed before reply)"}
        resp = json.loads(text)
        if not isinstance(resp, dict):
            # Valid JSON that is not an object — a foreign listener on the
            # port. Raised rather than returned so it lands in the `protocol`
            # class below: this function's contract is "returns a response
            # dict", and every caller was otherwise left to discover
            # separately that it sometimes doesn't.
            raise ValueError("daemon returned non-object response: %r" % (resp,))
        return resp

    except socket.timeout:
        return {"ok": False, "transport": "timeout", "error": "Timeout"}
    except ConnectionRefusedError:
        return {"ok": False, "transport": "refused",
                "error": "Connection refused — daemon may be dead"}
    except ConnectionResetError as e:
        return {"ok": False, "transport": "reset", "error": str(e)}
    except Exception as e:
        return {"ok": False, "transport": "protocol", "error": str(e)}
    finally:
        sock.close()


def is_daemon_running() -> bool:
    """Check if the daemon is running."""
    pid_path = get_pid_path()
    if not os.path.exists(pid_path):
        return False

    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        try:
            os.unlink(pid_path)
        except Exception:
            pass
        return False


def _can_connect(timeout: float = 2.0) -> dict:
    """Try to ping the daemon. Returns ping response or empty dict."""
    try:
        return send_command("ping", timeout=timeout)
    except Exception:
        return {}


def is_daemon_responsive(timeout: float = 2.0) -> bool:
    """True only if the daemon answers a ping (liveness).

    Distinguishes a live daemon from a hung "corpse" that holds the port but
    services nothing — which a bare TCP connect or a PID-exists check would
    wrongly report as alive. This is the liveness signal; readiness (e.g. a
    cold-start recall that's slow but alive) is a separate concern and must
    NOT be treated as down."""
    return _can_connect(timeout=timeout).get("ok", False)


def log_hook_error(hook_name: str, error: str, context: str = "",
                   level: str = "error", db_dir: Optional[str] = None) -> None:
    """Append a row to brain_logs.db's hook_errors table without the daemon,
    which is often the thing that is down. The dashboard errors panel,
    query_logs and the next boot read that table either way. The SQL lives in
    LogsDAL; `db_dir` defaults to the resolved brain dir. Never raises: a
    failure to log is reported on stderr."""
    try:
        import sqlite3
        from .dal_logs import LogsDAL
        db_dir = db_dir or resolve_db_dir()
        if not (db_dir and os.path.isdir(db_dir)):
            return
        conn = sqlite3.connect(os.path.join(db_dir, "brain_logs.db"), timeout=3)
        try:
            LogsDAL(conn).log_hook_error(hook_name, error, context=context, level=level)
        finally:
            conn.close()
    except Exception as e:
        sys.stderr.write("[brain-daemon] could not log %s to hook_errors: %s\n" % (hook_name, e))


def _restart_evidence() -> str:
    """What the corpse test reads, taken just before a client forces a restart."""
    age = pid_file_age_s()
    if age is not None:
        return "PID file %.0fs old" % age
    return "no PID file, port %s" % ("held" if port_is_occupied() else "free")


def record_forced_restart(action: str, evidence: Optional[str] = None,
                          db_path: Optional[str] = None,
                          caller: Optional[str] = None) -> None:
    """One hook_errors row per restart a client forces: what it did, what it
    saw, and which process did it. daemon.log shows only the SIGTERM that
    follows, and the caller's stderr (a hook's or an MCP proxy's) is not kept,
    so without this row a daemon killed mid-start cannot be traced to whoever
    killed it. `evidence` defaults to reading it now; pass it when the act
    that follows would change it. launchd-install.sh calls this for its
    bootouts and names itself as `caller`."""
    log_hook_error(
        "DAEMON_RESTART_FORCED", action,
        context="%s; caller %s" % (
            evidence if evidence is not None else _restart_evidence(),
            caller or "%s pid %d" % (os.path.basename(sys.argv[0]) or "?", os.getpid())),
        level="error",
        db_dir=os.path.dirname(db_path) if db_path else None)


# Liveness-grace budgets (WALL-CLOCK seconds) for the daemon (re)start decision.
# A bare 2s ping cannot tell "slow / relaunching" from "dead": a recall runs
# 10-17s, and a launchd relaunch is gated by ThrottleInterval (10s — see
# com.brain.daemon.plist) + an embedder reload (~4-6s). The budget must clear
# that window, or a busy/recovering daemon is mistaken for a corpse and
# (re)started — SIGKILLing it, resetting the throttle, corrupting an in-flight
# hook (the closed-DB boot failure of 2026-06-06). Bounded by WALL-CLOCK, not a
# ping count: a hung corpse whose socket blocks the full ping timeout each probe
# must not blow the budget (a count of 40 × 2s pings would be ~100s, not ~20s).
# Derived from the plist throttle (single-sourced in daemon_config) so the two
# worlds can't drift: the budget must clear ThrottleInterval + an embedder reload.
_EMBEDDER_RELOAD_BUDGET_S = 10.0                                          # cold embedder + Anthropic client warmup
_GRACE_DEADLINE_S = LAUNCHD_THROTTLE_INTERVAL_S + _EMBEDDER_RELOAD_BUDGET_S   # 20.0 — confirm-down / wait out a relaunch
_KICKSTART_DEADLINE_S = _GRACE_DEADLINE_S + 5.0                               # 25.0 — wait for our own kickstart to come back


def _await_responsive(deadline_s: float, ping_timeout: float = 2.0) -> dict:
    """Poll the daemon until it answers a ping or `deadline_s` of WALL-CLOCK time
    elapses; returns the last ping response ({} if never responsive).

    Bounded by real time, not a ping count — a hung corpse whose socket blocks
    the full ping_timeout per probe can't exceed deadline_s + one ping. The
    robust liveness gate the boot (re)start decision uses instead of a bare ping.
    NOT used on the recovery hot path (recover_daemon stays a single fast ping)."""
    deadline = time.monotonic() + max(0.0, deadline_s)
    resp = _can_connect(timeout=ping_timeout)
    while not resp.get("ok") and time.monotonic() < deadline:
        time.sleep(0.5)
        resp = _can_connect(timeout=ping_timeout)
    return resp


def _is_daemon_source(resp: dict) -> bool:
    """Whether THIS checkout is the daemon's launch source — the only checkout
    that may manage its lifecycle.

    NOT an identity statement. The brain (the one shared brain.db) is the same
    for every checkout, so every session is Anchor. This is purely machinery:
    the daemon's CODE is launchd-pinned to its source dir, so only the source
    checkout can converge a (re)start. A linked worktree / 2nd clone is a pure
    client — restarting from there can't change the daemon's code, it only churns.

    Signal: the daemon's reported `source_dir` vs this checkout's REPO_ROOT. When
    the daemon is DOWN (no source_dir to compare) we fall back to the
    linked-worktree heuristic: a linked worktree is never the source, but a
    second full *clone* (its .git is a directory, not a file) is indistinguishable
    from the source here and is treated AS source — acceptable because a kickstart
    from a clone still converges via launchd's pinned source, and the manual kill
    path (_relaunch_daemon) re-checks before ever killing the shared daemon.

    One exception makes a different tree the source: a daemon launchd does
    not manage (Linux, or macOS before the LaunchAgent is bootstrapped) whose
    tree this checkout superseded — its launcher is gone, or this checkout is
    a newer release of the same plugin (a plugin update installs a NEW
    version dir beside the old). Nothing else would ever move that daemon.
    Under launchd the plist names the owner and install-daemon-service.sh
    re-points it by the same rule (launchd-install.sh); acting here as well
    would restart the daemon back onto the plist's tree every session."""
    daemon_src = resp.get("result", {}).get("source_dir", "")
    if daemon_src:
        return _runs_this_tree(resp) or _superseded(resp)
    return not _IS_WORKTREE


def _runs_this_tree(resp: dict) -> bool:
    daemon_src = resp.get("result", {}).get("source_dir", "")
    return bool(daemon_src) and os.path.realpath(daemon_src) == os.path.realpath(REPO_ROOT)


def _superseded(resp: dict) -> bool:
    """The exception in _is_daemon_source: the daemon runs a tree this
    checkout superseded, and launchd does not manage it."""
    daemon_src = resp.get("result", {}).get("source_dir", "")
    if not daemon_src or _runs_this_tree(resp):
        return False
    launcher = os.path.join(daemon_src, "hooks", "scripts", "brain-daemon")
    if os.access(launcher, os.X_OK) and not release_supersedes(REPO_ROOT, daemon_src):
        return False
    # Asked last: a launchctl fork, and only a definitive "not managed" says
    # yes (manages() reads an indeterminate answer as managed).
    return not manages()


def _code_changed(resp: dict) -> bool:
    """True iff a restart is warranted: this checkout is the daemon's source AND
    the daemon runs a different code fingerprint. Conservative — an unknown
    fingerprint never restarts, and a non-source checkout is never "changed"
    (it can't converge a restart — the non-convergent churn of 2026-06-06)."""
    if not _is_daemon_source(resp):
        return False
    daemon_fp = resp.get("result", {}).get("code_fingerprint", "")
    # Import-time constant, not a fresh _code_fingerprint(): this process is
    # short-lived (hook/CLI) so its code can't change underneath it, and
    # recomputing re-reads every servers/**/*.py (several MB) on each call.
    return bool(_CODE_FINGERPRINT != "unknown" and daemon_fp and daemon_fp != _CODE_FINGERPRINT)


def _db_dir_changed(resp: dict, db_path: str) -> bool:
    """True iff a restart is warranted for DATA divergence: this checkout is
    the daemon's source AND the daemon reports a `db_dir` different from the
    dir this session resolved (D-13 step 4b — the split-brain killer: after
    the user adopts a new brain location, a daemon launched off the old
    plist-baked path would keep writing the old brain forever).

    Same treatment as _code_changed: source-checkout only (a kickstart from a
    non-source checkout is the same non-convergent churn), and conservative —
    a daemon that doesn't report db_dir (pre-step-4 code) never restarts on
    this signal; the code-fingerprint check already covers that daemon.

    Convergence: kickstart relaunches via brain-daemon, which re-runs the
    resolution ladder (step 4a), against a plist install-daemon-service.sh
    re-materialized at boot (step 4c) — so the restarted daemon lands on the
    same dir this session resolved."""
    if not _is_daemon_source(resp):
        return False
    daemon_db_dir = resp.get("result", {}).get("db_dir", "")
    if not daemon_db_dir or not db_path:
        return False
    session_db_dir = os.path.dirname(os.path.abspath(db_path))
    return os.path.realpath(daemon_db_dir) != os.path.realpath(session_db_dir)


def _singleton_free_within(deadline_s: float) -> bool:
    """Whether the daemon's SINGLETON flock is (or within `deadline_s` becomes)
    free. A dying daemon releases it in _cleanup after closing its DB
    connections, so free means no live writer holds brain.db."""
    deadline = time.monotonic() + deadline_s
    while True:
        try:
            with open(get_lock_path(), 'w') as probe:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(probe, fcntl.LOCK_UN)
            return True
        except (IOError, OSError):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.5)


def ensure_daemon(db_path: str) -> bool:
    """Ensure the daemon is running AND on current code. Returns True if ready.

    launchd owns process CREATION (com.brain.daemon: KeepAlive, RunAtLoad).
    This function only PINGS and, when convergence is needed: a healthy daemon
    on stale code is asked to reload in place (the `restart` command — see
    BrainDaemon._exec_reload); only corpses and db-dir divergence route
    through `launchctl kickstart -k` (daemon_launch.kickstart). It never
    kills + Popens its own process alongside launchd.

    Doing both was the Errno-48 storm of 2026-06-04: N sessions booted at once,
    each saw stale code, and each independently killed + respawned while
    launchd's KeepAlive ALSO respawned — several processes raced to bind the
    port. Now every (re)start decision is serialized under the STARTUP lock
    (get_startup_lock_path — a different inode from the daemon's singleton
    flock, which a serving daemon holds for life) and re-checked after
    acquiring it, so N concurrent callers (re)start at most once. Direct Popen
    survives ONLY as the no-launchd fallback (a fresh
    install where the LaunchAgent isn't bootstrapped) — there's no KeepAlive to
    race there.

    Maintenance mode: if the maintenance lock file exists, skip startup.
    Used during DB operations (VACUUM, schema changes, bulk deletes).
    """
    if is_maintenance_mode():
        sys.stderr.write("[brain-daemon] Maintenance mode active — skipping startup\n")
        return False

    def _needs_restart(r: dict) -> bool:
        # Stale CODE and diverged DATA both mean "not converged" — the MOVE
        # differs (code → in-place reload; data → kickstart via the ladder).
        # A superseded tree goes even when its code matches this one: the
        # launcher and resolver a reload runs are still the old release's.
        if _superseded(r):
            return True
        if _code_changed(r):
            return True
        if _db_dir_changed(r, db_path):
            sys.stderr.write(
                "[brain-daemon] db-path divergence: daemon on %s, session "
                "resolved %s — restarting\n"
                % (r.get("result", {}).get("db_dir", "?"),
                   os.path.dirname(os.path.abspath(db_path))))
            return True
        return False

    resp = _can_connect()

    # A non-source checkout (linked worktree / 2nd clone) is a PURE CLIENT of the
    # shared daemon. Identity is shared — the one brain.db makes every checkout
    # Anchor — but the daemon's CODE is launchd-pinned to its source, so a
    # (re)start from here can't converge to this checkout's code; it would only
    # churn and SIGKILL in-flight work. So a non-source checkout never restarts:
    # up → done; down → wait out launchd's relaunch (KeepAlive owns recovery).
    if not _is_daemon_source(resp):
        if resp.get("ok"):
            return True
        return _await_responsive(_GRACE_DEADLINE_S).get("ok", False)

    # ── Source checkout: owns lifecycle (kickstart converges; stale code reloads). ──
    # Fast path: running, responsive, and on current code → nothing to do.
    if resp.get("ok") and not _needs_restart(resp):
        return True
    # Otherwise (down, or up-but-stale) fall through to the locked (re)start.

    # Serialize the (re)start through the RESTARTER lock — NOT the daemon's
    # singleton flock (get_lock_path). The daemon holds that one for its whole
    # serving life, so blocking on it here hangs the boot hook whenever a
    # healthy-but-stale daemon serves, and holding it through a (re)start
    # makes the successor's own start() exit as a duplicate. See
    # get_startup_lock_path's docstring — one inode per question.
    lock_path = get_startup_lock_path()
    lock_fd = None
    try:
        lock_fd = open(lock_path, 'w')
        sys.stderr.write("[brain-daemon] Acquiring startup lock...\n")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)  # Blocks until acquired
        sys.stderr.write("[brain-daemon] Lock acquired.\n")

        # Re-check under the lock — a caller we blocked behind may have already
        # brought up a healthy, current-code daemon while we waited. A bare 2s
        # ping can't tell a slow recall / a launchd relaunch (throttle + reload)
        # from death, so wait it out before forcing a (re)start.
        resp = _can_connect()
        if not resp.get("ok"):
            resp = _await_responsive(_GRACE_DEADLINE_S)
        if resp.get("ok") and not _needs_restart(resp):
            sys.stderr.write("[brain-daemon] Daemon healthy (handled by another caller / recovered).\n")
            return True

        # A plain-process daemon on a tree this checkout superseded: a reload
        # would exec THAT tree's launcher again, so shut it down and let the
        # no-launchd spawn below start this tree. Stopped means its singleton
        # flock is free — released last in its _cleanup, after the socket and
        # PID file, so the spawn path finds nothing to mistake for a zombie.
        if resp.get("ok") and _superseded(resp):
            sys.stderr.write("[brain-daemon] daemon runs superseded tree %s — "
                             "replacing it with this one\n"
                             % resp["result"]["source_dir"])
            send_command("shutdown", timeout=5.0)
            if not _singleton_free_within(_GRACE_DEADLINE_S):
                sys.stderr.write("[brain-daemon] superseded daemon did not stop "
                                 "within %.0fs — not spawning beside it\n" % _GRACE_DEADLINE_S)
                return _can_connect().get("ok", False)
            resp = {}

        # Healthy but on stale CODE → ask it to reload in place (the daemon
        # execs hooks/scripts/brain-daemon, same PID — see
        # BrainDaemon._exec_reload). Reload is the owner of "healthy + stale";
        # kickstart below narrows to corpses and db-dir divergence (a data
        # move is a deliberate operator action — the spawn ladder + plist
        # re-materialization is its reviewed path). A reload that doesn't
        # come back healthy falls through to kickstart: at that point the
        # daemon IS the corpse case.
        if resp.get("ok") and _code_changed(resp) and not _db_dir_changed(resp, db_path):
            sys.stderr.write("[brain-daemon] stale code on a healthy daemon — "
                             "reloading in place\n")
            try:
                r = send_command("restart", timeout=5.0)
            except Exception as e:
                r = {}
                sys.stderr.write("[brain-daemon] reload request failed: %s\n" % e)
            # The daemon is syncing its runtime to uv.lock first, still
            # serving, and swaps itself when the sync ends. Waiting out the
            # grace budget here and then kickstarting would kill it mid-sync
            # (BrainDaemon._step_runtime_sync).
            if r.get("ok") and r.get("result", {}).get("status") == "syncing":
                sys.stderr.write("[brain-daemon] daemon is syncing its runtime "
                                 "before reloading — serving meanwhile\n")
                return True
            # Poll for responsive AND current-code — a bare responsive check
            # can catch the OLD daemon in its last pre-teardown tick (stale
            # fingerprint) and misread the reload as failed.
            deadline = time.monotonic() + _GRACE_DEADLINE_S
            while time.monotonic() < deadline:
                time.sleep(0.5)
                resp = _can_connect()
                if resp.get("ok") and not _needs_restart(resp):
                    sys.stderr.write("[brain-daemon] Daemon reloaded in place.\n")
                    return True
            sys.stderr.write("[brain-daemon] reload did not converge within "
                             "%.0fs — falling back to kickstart\n" % _GRACE_DEADLINE_S)
            # A reload that degraded into exit + KeepAlive respawn (backstop
            # fired, exec failed) can still be mid-boot when the poll expires
            # — ThrottleInterval (10s) + boot can exceed the grace budget.
            # Same corpse test recover_daemon applies: a freshly-bound or
            # binding daemon is booting, not a corpse — kickstarting it now
            # would SIGKILL it mid-boot and reset the throttle.
            age = pid_file_age_s()
            if (age is not None and age < _GRACE_DEADLINE_S) or \
                    (age is None and port_is_occupied()):
                sys.stderr.write("[brain-daemon] successor is booting — "
                                 "deferring, not kickstarting\n")
                return False

        # Route the (re)start through launchd. `kickstart -k` kills any running
        # instance (covers healthy-but-stale AND hung-corpse) and respawns it in
        # one launchd-serialized call — no competing-spawn race with KeepAlive.
        evidence = _restart_evidence()
        if kickstart():
            record_forced_restart(
                "ensure_daemon: kickstart -k, daemon %s" % (
                    "answered but needed a restart" if resp.get("ok")
                    else "unresponsive (%s)" % (resp.get("transport") or "no reply")),
                evidence, db_path)
            # Wait past ThrottleInterval (10s) + embedder reload — a tighter
            # window gives up while the daemon is still coming back and would
            # re-kickstart it, resetting the throttle (the self-sustaining storm).
            t0 = time.monotonic()
            resp = _await_responsive(_KICKSTART_DEADLINE_S)
            if resp.get("ok") and not _needs_restart(resp):
                sys.stderr.write("[brain-daemon] Daemon ready via launchd (took %.1fs)\n"
                                 % (time.monotonic() - t0))
                return True
            sys.stderr.write("[brain-daemon] Daemon not ready within %.0fs after kickstart\n"
                             % _KICKSTART_DEADLINE_S)
            return False

        # kickstart did not succeed. Decide carefully whether a direct spawn is
        # safe — spawning a rival when one already exists IS the Errno-48 orphan
        # storm (2026-06-05).
        #
        # (a) A daemon is still responsive RIGHT NOW. Re-ping here — the `resp`
        #     from the pre-kickstart re-check is stale: `kickstart -k` SIGKILLs
        #     the incumbent before respawning, so a kickstart that failed/timed
        #     out AFTER the kill leaves the port free. Only defer to an incumbent
        #     that is actually still up (deferring on the stale snapshot would
        #     report "ready" with nothing serving). If it IS still up, kickstart
        #     failing means THIS context couldn't reach launchd, not that the
        #     daemon is unmanaged — defer, never spawn a competitor.
        resp = _can_connect()
        if resp.get("ok"):
            sys.stderr.write(
                "[brain-daemon] launchd kickstart unavailable but a daemon is "
                "already responsive — deferring (NOT spawning a competitor; it "
                "may run stale code until launchd cycles it).\n")
            return True

        # (b) Nothing is serving. Spawn directly ONLY if launchd genuinely is
        #     not managing the daemon (fresh install). If launchd DOES manage it
        #     but kickstart failed transiently, let KeepAlive bring it up rather
        #     than racing a manual spawn. (Role note: _relaunch_daemon KILLS in
        #     this same managed+kickstart-failed state — correct there, where the
        #     target is a live corpse KeepAlive can't respawn past. Here "down"
        #     means the process exited, so KeepAlive owns the respawn. Don't
        #     unify the two arms.)
        if manages():
            sys.stderr.write(
                "[brain-daemon] launchd manages the daemon but kickstart failed "
                "and nothing is serving — leaving it to KeepAlive.\n")
            return False

        # No launchd managing the daemon (fresh install / not bootstrapped).
        # No KeepAlive to race here, so spawn directly — still under the lock.
        sys.stderr.write("[brain-daemon] launchd not managing daemon — spawning directly\n")
        # (a)'s re-ping says whether anything listens. Refused: nothing does —
        # a stopped daemon's TIME_WAIT connections refuse too. A timeout or a
        # broken reply: something accepted and did not answer, a zombie on
        # whatever address it bound.
        if resp.get("transport") in ("timeout", "reset", "empty", "protocol"):
            sys.stderr.write("[brain-daemon] Port occupied but unresponsive — killing zombie\n")
            record_forced_restart("ensure_daemon: killing a zombie before spawning (%s)"
                                  % resp["transport"], db_path=db_path)
            kill_daemon()
            time.sleep(1)
        # Two-writer guard: wait (bounded) for the daemon's SINGLETON flock to
        # free before spawning, so a spawn past this point can never open
        # brain.db alongside a live writer. (The old code got this implicitly
        # by serializing on that same flock; the dedicated startup lock above
        # no longer does.)
        if not _singleton_free_within(_GRACE_DEADLINE_S):
            sys.stderr.write("[brain-daemon] singleton flock still held after "
                             "%.0fs — not spawning a second writer\n" % _GRACE_DEADLINE_S)
            return False
        spawn_detached_daemon(db_path)
        for i in range(20):  # 10 seconds max
            time.sleep(0.5)
            if _can_connect().get("ok"):
                sys.stderr.write("[brain-daemon] Daemon ready (took %.1fs)\n" % ((i + 1) * 0.5))
                return True
        sys.stderr.write("[brain-daemon] Daemon failed to start within 10s\n")
        return False

    except Exception as e:
        sys.stderr.write("[brain-daemon] ensure_daemon error: %s\n" % e)
        return False
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def stop_daemon():
    """Gracefully stop the daemon."""
    resp = send_command("shutdown", timeout=5.0)
    if not resp.get("ok"):
        kill_daemon()
        return
    for _ in range(20):
        if not is_daemon_running():
            return
        time.sleep(0.1)
    kill_daemon()


# ─── Hung-daemon recovery ───
#
# A daemon that hangs after host sleep ("corpse") keeps its port bound but
# services nothing, so launchd's crash-respawn never fires and nothing
# inside the frozen process can rescue it. Recovery must come from OUTSIDE
# the frozen process: force it to die and let launchd respawn it. Every recovery caller — the MCP health
# monitor and the recall hook — routes through recover_daemon(); shared
# cooldown + circuit-breaker state (one /tmp file) keeps them from fighting.

_RECOVERY_COOLDOWN_S = 30.0      # don't issue a 2nd restart while one is in flight
_RECOVERY_MAX_ATTEMPTS = 5       # circuit breaker: stop after this many…
_RECOVERY_WINDOW_S = 600.0       # …within this sliding window, then surface loudly


def _read_recovery_state() -> dict:
    try:
        with open(get_recovery_state_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"last_attempt": 0.0, "attempts": 0}


def _write_recovery_state(last_attempt: float, attempts: int):
    try:
        with open(get_recovery_state_path(), "w") as f:
            json.dump({"last_attempt": last_attempt, "attempts": attempts}, f)
    except OSError:
        pass


def _relaunch_daemon(db_path: Optional[str], why: str = "recover_daemon"):
    """Bring a fresh daemon up. launchd is the canonical owner (kickstart -k);
    if launchd isn't managing it (kickstart fails), fall back to kill + Popen
    spawn via ensure_daemon.

    Role note — this kill fires even when launchd MANAGES the daemon, and that
    is deliberate (do not "unify" it with ensure_daemon's defer-to-KeepAlive
    arm): recovery's target is the hung CORPSE — a live process holding the
    port and serving nothing. KeepAlive only respawns on process EXIT, so it
    can never get past a live corpse; when kickstart -k (launchd's own killer)
    is unreachable, the manual SIGKILL is what produces the exit KeepAlive
    needs. ensure_daemon's managed arm defers instead because its "down" means
    the process already exited — there, KeepAlive genuinely owns the respawn.

    Each act that stops the daemon leaves a record_forced_restart row, `why`
    naming the decision; the defer paths stop nothing and leave none. The
    evidence is read before the kickstart, which changes it."""
    evidence = _restart_evidence()
    if kickstart():
        record_forced_restart("%s: kickstart -k" % why, evidence, db_path)
        return
    # kickstart failed. Re-ping before ANY kill decision: recover_daemon's
    # single 2s ping can't tell slow/busy from dead (a recall runs 10-17s), and
    # a kickstart can fail transiently while a healthy daemon serves. A daemon
    # that answers NOW must be deferred to, never SIGKILLed mid-request —
    # ensure_daemon's ladder has this rung; it was missing here.
    resp = _can_connect()
    if resp.get("ok"):
        sys.stderr.write("[brain-daemon] kickstart unavailable but the daemon answers "
                         "pings again — deferring (recovery not needed).\n")
        return
    # A manual kill+respawn only converges from the daemon's SOURCE checkout:
    # a linked worktree / 2nd clone would SIGKILL the shared daemon and then
    # refuse to spawn (pure client, can't converge its own code), leaving
    # nothing serving. Defer loudly instead.
    if not _is_daemon_source(resp):
        sys.stderr.write("[brain-daemon] kickstart unavailable and this checkout is not the "
                         "daemon's source — deferring kill+respawn to launchd/source "
                         "(NOT killing the shared daemon).\n")
        return
    # No launchd (or kickstart failed) and we ARE the source — own the kill + respawn.
    record_forced_restart("%s: kill + respawn" % why, evidence, db_path)
    kill_daemon()
    if not db_path:
        db_path = os.path.join(resolve_db_dir(), "brain.db")
    ensure_daemon(db_path)


def recover_daemon(db_path: Optional[str] = None) -> bool:
    """Recover a hung/unreachable daemon. Returns True iff it's healthy now.

    The single recovery path for every caller. Idempotent and safe to call
    repeatedly — it no-ops when:
      - the daemon already answers a ping (and clears any failure streak),
      - maintenance mode is on (intentional shutdown),
      - a restart is still in flight (cooldown), or
      - the circuit breaker has tripped (too many failed restarts).
    Otherwise it forces a restart and returns False; the next ping confirms.
    """
    if is_maintenance_mode():
        return False

    # Fast single-ping confirm — recover_daemon runs on synchronous hook paths
    # (recall/Stop/Edit, host timeouts 5-21s) and the MCP monitor, so it must NOT
    # pay the multi-second grace ensure_daemon uses; the cooldown + circuit
    # breaker below are what prevent restart storms. A transient blip is absorbed
    # by the cooldown; a genuine corpse is force-restarted (and re-confirmed by
    # the next caller's ping, not by blocking this one).
    if is_daemon_responsive():
        if _read_recovery_state().get("attempts"):
            _write_recovery_state(0.0, 0)  # healthy again — reset the streak
        return True

    # Corpse test, not just liveness test. Unresponsive with a fresh PID file
    # (claimed at the lock or rewritten at bind) is a daemon still starting
    # up, and unresponsive-with-the-port-held-mid-teardown is a restart in
    # flight — killing either was the double-kill of 2026-08-17/18 (a hook's
    # recovery kickstarting into a deploy restart's boot window, SIGKILLing
    # the healthy fresh daemon). A genuine corpse is unresponsive AND long
    # past its last claim or bind (stale PID file). No attempt is
    # stamped here — declining to kill a booting daemon is not a recovery
    # attempt, and the next caller re-evaluates with a grown age.
    age = pid_file_age_s()
    if age is not None and age < _GRACE_DEADLINE_S:
        return False  # claimed or bound seconds ago — give it its window
    if age is None and port_is_occupied():
        return False  # teardown/boot in motion — the successor is binding

    now = time.time()
    state = _read_recovery_state()
    last_attempt = state.get("last_attempt", 0.0)
    attempts = state.get("attempts", 0)

    # Sliding window: an old failure streak ages out, so the breaker can't
    # lock recovery out forever after a long-past incident.
    if now - last_attempt > _RECOVERY_WINDOW_S:
        last_attempt, attempts = 0.0, 0

    if now - last_attempt < _RECOVERY_COOLDOWN_S:
        return False  # a restart is already in flight — give it time to boot

    if attempts >= _RECOVERY_MAX_ATTEMPTS:
        sys.stderr.write(
            "[brain-daemon] recovery circuit OPEN — %d restarts in %ds did not "
            "revive the daemon; not restarting again. Investigate manually.\n"
            % (attempts, int(_RECOVERY_WINDOW_S)))
        return False

    attempts += 1
    _write_recovery_state(now, attempts)
    sys.stderr.write("[brain-daemon] daemon unresponsive — forcing restart (attempt %d)\n" % attempts)
    _relaunch_daemon(db_path, "recover_daemon: unresponsive, restart attempt %d" % attempts)
    return False  # not yet verified; the next ping confirms recovery
