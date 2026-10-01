"""
brain — Daemon Configuration

Constants, paths, and fingerprinting for the brain daemon.
Pure config — no classes, no side effects beyond CPU-only env setup.
"""

import os
import sys
import time
import hashlib

from servers.brain_constants import user_config_dir

# ─── Force CPU-only BEFORE any downstream import ───
# On macOS Apple Silicon, CoreML/Metal XPC connections cause SIGABRT in
# background/daemon processes that lack GPU context. DAEMON_CPU_ENV is the SINGLE
# source of the CPU-only invariant: the launchd plist, every spawn site's env=,
# and this import-time application all draw from it, so a daemon started fresh,
# restarted, or launchd-spawned runs with the IDENTICAL accelerator env (a prior
# drift left the restart path without VECLIB_MAXIMUM_THREADS / MPS_FALLBACK).
DAEMON_CPU_ENV = {
    "ORT_DISABLE_ALL_ACCELERATORS": "1",
    "ONNX_PROVIDERS": "CPUExecutionProvider",
    "PYTORCH_MPS_DISABLE": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
}
# ORT accelerator-disable is the load-bearing SIGABRT guard — force it even over
# an ambient value (matches pre-consolidation behavior); setdefault the rest so
# an explicit plist/operator value wins.
os.environ["ORT_DISABLE_ALL_ACCELERATORS"] = DAEMON_CPU_ENV["ORT_DISABLE_ALL_ACCELERATORS"]
for _cpu_k, _cpu_v in DAEMON_CPU_ENV.items():
    os.environ.setdefault(_cpu_k, _cpu_v)

# ─── Constants ───

# S2 scheduling moved to Brain.run_maintenance_if_due (see brain.py).
# Brain owns MAINTENANCE_IDLE_THRESHOLD_SECONDS and MAINTENANCE_MIN_INTERVAL_SECONDS
# as class constants; daemon just polls. Old S2_IDLE_THRESHOLD / S2_MIN_INTERVAL
# deleted along with the scheduling logic in daemon_server._serve.
AUTOSAVE_INTERVAL_SECONDS = 60  # Save every 60 seconds if dirty
# Cached SessionContexts flush on their own, faster cadence: a crash loses at
# most this much session state (a graceful shutdown loses none — Brain.close
# flushes). Only changed sessions are written.
SESSION_FLUSH_INTERVAL_SECONDS = 10
SOCKET_BACKLOG = 5
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB max message
THREAD_POOL_SIZE = max(4, (os.cpu_count() or 4) // 2)
# Scales hook/request concurrency with the host. SQLite WAL allows concurrent
# readers, and daemon_server serializes writers — no deadlock risk. Embedder
# (fastembed + ORT 1.24.4) is thread-safe for concurrent InferenceSession.run().
POOL_THREAD_PREFIX = "pool-worker"
# Pool worker names: "pool-worker_N" idle, "pool-worker:<cmd>" while handling a
# request (daemon_server._handle_client). The THREADS alarm leaves both out —
# the pool is bounded by THREAD_POOL_SIZE, so it cannot run away.
THREAD_ALARM_MAX_OTHERS = 15  # threads outside the pool before THREADS fires
# daemon.log history: the oldest line kept is at most this many days old.
# daemon_server._rotate_log_if_due rotates every keep_days / 2 into one backup
# (daemon.log.1), so 30-60 days are always on disk. Override per brain with
# brain_meta `daemon_log.keep_days`.
DAEMON_LOG_KEEP_DAYS = 60
# Loopback only. The daemon has no access control of its own — the hook HMAC
# (dispatch_common.verify_caller_session) is attribution checked in the MCP
# proxy, and `eval` runs code — so nothing beyond this host may reach the port.
# A literal IPv4 address, never "localhost": the server is AF_INET, and macOS
# resolves that name to ::1 first. The server binds and every client connects
# through get_daemon_addr(); the two clients that copy the host instead of
# importing it are pinned to this value (dashboard: test_dashboard_disconnection,
# post_tool_trace: test_daemon_instance).
DAEMON_HOST = "127.0.0.1"
# DAEMON_PORT is defined below _read_env_file_key — it reads the user env file.

# launchd's plist ThrottleInterval — launchd won't relaunch the daemon faster
# than this after an exit. The recovery deadlines in daemon_client DERIVE from it
# (they must exceed throttle + embedder reload, or a recovering daemon is
# mistaken for a corpse and re-kickstarted — the self-sustaining storm). Keep in
# sync with the plist's <key>ThrottleInterval</key> (Step 7 generates the plist
# from this + a contract test). SIGKILL grace is launchd's SIGTERM→SIGKILL window
# (~macOS default); SHUTDOWN_BACKSTOP_S must stay under it so we exit on our own
# terms.
LAUNCHD_THROTTLE_INTERVAL_S = 10
LAUNCHD_SIGKILL_GRACE_S = 20


# ─── Identity binding ───
# Concrete names for the human operator and the agent. Stamped onto S0
# trace_events metadata at write time so each event independently records who
# said what. Source: ~/.config/brain/env (sourced by boot-brain.sh, inherited
# into daemon environment).
#
# The agent name always has a value — an entity with no name at all forces
# every render into awkward name-free prose. The operator name does not: we
# cannot invent a human's name, so empty still means unset there and the DAL
# skips stamping it.
#
# Whether the agent name was CHOSEN is a separate fact from what it IS, and
# both are needed. Emptiness used to stand in for "unchosen", which is why a
# default silently disarms everything keyed on it — the identity warning, and
# the Nursery's naming invitation. `agent_name_is_default()` carries that fact
# explicitly instead, so an entity can genuinely be Jade while the system still
# knows nobody picked it.

DEFAULT_AGENT_NAME = 'Jade'


def get_operator_name() -> str:
    """Canonical name of the current human partner. Empty if unset."""
    return os.environ.get('BRAIN_OPERATOR_NAME', '').strip()


def get_agent_name() -> str:
    """Canonical name of the agent (the brain's self-token).

    Never empty — falls back to DEFAULT_AGENT_NAME. Ask
    `agent_name_is_default()` if you need to know whether the operator
    actually chose it.
    """
    return os.environ.get('BRAIN_AGENT_NAME', '').strip() or DEFAULT_AGENT_NAME


def agent_name_is_default() -> bool:
    """True while the agent is running under the name it shipped with.

    The name is issued, not chosen — the distinction the entity's own pitch
    rests on, and what the boot's naming invitation and the trace-identity
    warning both gate on.
    """
    return not os.environ.get('BRAIN_AGENT_NAME', '').strip()


# ─── Code Fingerprinting ───

def _fingerprint_dir(path: str) -> str:
    """Recursive content hash (md5) of every *.py under `path`, keyed by
    RELATIVE path + NUL-separated bytes. Recursive so changes in subpackages
    (scales/, db_backends/, ...) are detected; content-based + path-relative so
    identical code matches across checkouts; pure/path-parameterized for tests."""
    h = hashlib.md5()
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__')
        for f in sorted(g for g in files if g.endswith('.py')):
            rel = os.path.relpath(os.path.join(root, f), path).encode()
            with open(os.path.join(root, f), 'rb') as fh:
                h.update(rel + b'\0' + fh.read() + b'\0')
    return h.hexdigest()[:16]


def _code_fingerprint() -> str:
    """Fingerprint of server code by file CONTENT not mtime — mtime hashing made
    every worktree look "stale" with identical code → restart churn (2026-06-05)."""
    try:
        return _fingerprint_dir(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        return "unknown"


# Captured at import time — represents the code version this process loaded
_CODE_FINGERPRINT = _code_fingerprint()

# Wall-clock at import — when the daemon's CODE IMAGE loaded (this module is
# imported while daemon_server loads, before the brain does). Not from
# BrainDaemon._run_started_at, which resets on every supervisor retry within
# one image. Note: an in-place reload (BrainDaemon._exec_reload) re-imports
# this module in the same PID, so "uptime" measures the age of the loaded
# code, not of the OS process — the more useful reading now that the PID
# survives deploys.
_PROCESS_STARTED_AT = time.time()


def _is_worktree_checkout(repo_root: str) -> bool:
    """True if `repo_root` is a *linked* git worktree — its `.git` is a FILE
    (a `gitdir:` pointer) rather than a DIRECTORY (primary checkout) or absent
    (tarball install → False). Pure/path-parameterized for tests. Gates daemon
    staleness restarts; the why lives in `daemon_client._code_changed`."""
    return os.path.isfile(os.path.join(repo_root, ".git"))


# Absolute repo root of THIS checkout (servers/'s parent). Canonical home: the
# daemon ping reports it as `source_dir`; _code_changed compares against it.
# Bootstrap sites (brain_mcp, daemon_server.start) recompute it locally
# because they set sys.path BEFORE they can import this module.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Linked-worktree heuristic — FALLBACK used by _code_changed only when the daemon
# predates `source_dir` reporting; the exact signal is the source_dir comparison.
_IS_WORKTREE = _is_worktree_checkout(REPO_ROOT)


# ─── Path Helpers ───

def brain_tmp_dir() -> str:
    """Root dir for brain's ephemeral cross-process files — the current-stop
    marker hook_recall writes for the PostToolUse hook. Honors BRAIN_TMP_DIR,
    defaults to /tmp.

    Production leaves BRAIN_TMP_DIR unset → /tmp. The seam is wired on BOTH
    sides: the daemon WRITES via this helper, and the out-of-process reader
    (hooks/scripts/post_tool_trace.py) honors the same env protocol. It is
    deliberately servers-decoupled, so it reads
    os.environ.get('BRAIN_TMP_DIR', '/tmp') directly rather than import this —
    keep it in sync if the default ever changes.

    BrainTestBase + IsolatedBrain set it to a per-test/run temp dir so two
    concurrent test processes don't collide on fixed filenames (hardcoded
    session_ids) and the files are cleaned up with the temp dir instead of
    leaking into /tmp.

    Eval harnesses set it the same way, to each item's brain dir alongside
    BRAIN_DB_DIR, so concurrent runs never share a current-stop marker (grep
    'BRAIN_TMP_DIR' across eval/ for the setters).

    NOT for the daemon DISCOVERY paths (pid/lock/status/maintenance) —
    those are the well-known uid-keyed rendezvous hooks and the statusline must
    find, and stay at /tmp regardless. NOT for user-facing post-mortem artifacts
    (get_mcp_crash_path) — those want a findable, uid-keyed location too.
    """
    return os.environ.get('BRAIN_TMP_DIR', '/tmp')


def _read_env_file_key(path: str, key: str):
    """Read one KEY=value from a config file that shell consumers SOURCE —
    so tolerate the shell grammar the same file is written in: optional
    `export `, single/double quotes, inline comments on unquoted values,
    $VAR expansion. resolved.env writes values single-quoted. Returns None
    if absent/unreadable — the two readers of one file must not disagree
    on its grammar (that's the split-brain class D-13 kills)."""
    try:
        with open(path, errors='replace') as f:
            for raw in f:
                line = raw.strip()
                if line.startswith('export '):
                    line = line[len('export '):].lstrip()
                if not line.startswith(key + '='):
                    continue
                v = line.split('=', 1)[1].strip()
                if v[:1] in ('"', "'"):
                    end = v.find(v[0], 1)
                    v = v[1:end] if end > 0 else v[1:]
                else:
                    v = v.split(' #', 1)[0].rstrip()
                if v:
                    return os.path.expanduser(os.path.expandvars(v))
    except OSError:
        pass
    return None


def _resolve_daemon_port() -> int:
    """Per-user daemon port. Env-first: brain-env.sh documents
    BRAIN_DAEMON_PORT as the override and every shell/hook client honors it —
    the daemon's own bind must read the same source or an override splits the
    system (clients on one port, daemon on another, kickstart storms against a
    healthy daemon). Falls back to the user env file for Python processes NOT
    launched through brain-env.sh (the MCP server — CC spawns it with a bare
    env), then to the uid formula. A malformed value warns and uses the
    formula instead of crash-looping the daemon under KeepAlive."""
    xdg = user_config_dir()
    raw = (os.environ.get("BRAIN_DAEMON_PORT")
           or _read_env_file_key(os.path.join(xdg, 'brain', 'env'),
                                 'BRAIN_DAEMON_PORT'))
    if raw:
        try:
            return int(raw)
        except ValueError:
            sys.stderr.write(
                "[daemon-config] WARN: BRAIN_DAEMON_PORT=%r is not an integer "
                "— using the uid formula\n" % (raw,))
    return 47200 + (os.getuid() % 100)


DAEMON_PORT = _resolve_daemon_port()


def _instance_suffix() -> str:
    """Per-instance key for every rendezvous path below.

    Empty for the production daemon (BRAIN_INSTANCE unset — every path is
    byte-identical to the pre-instance form). An eval entity sets
    BRAIN_INSTANCE so its PID/lock/status/recovery files never collide with
    production's: an entity acquiring production's daemon lock exits as a
    duplicate, and an entity touching production's maintenance lock silently
    disables production's startup and auto-recovery.
    """
    inst = os.environ.get("BRAIN_INSTANCE", "").strip()
    return "-" + inst if inst else ""


def _validate_instance_env() -> None:
    """Import-time, loud, once. A bad instance name would otherwise surface
    deep in runtime at the first path call; an instance inheriting
    production's port or DB would collide with the live daemon — those two
    env vars must be EXPLICIT (the user env file and the uid formula both
    resolve to production's values, so falling back to them is the failure)."""
    inst = os.environ.get("BRAIN_INSTANCE", "").strip()
    if not inst:
        return
    if not all(c.isalnum() or c in "_-" for c in inst) or len(inst) > 32:
        raise SystemExit(
            "[daemon-config] BRAIN_INSTANCE=%r must be 1-32 chars of "
            "[A-Za-z0-9_-] — it names /tmp rendezvous files and the "
            "launchd label." % inst)
    if not (os.environ.get("BRAIN_DAEMON_PORT")
            and os.environ.get("BRAIN_DB_DIR")):
        raise SystemExit(
            "[daemon-config] BRAIN_INSTANCE=%r requires explicit "
            "BRAIN_DAEMON_PORT and BRAIN_DB_DIR in the environment — an "
            "instance inheriting production's port or DB dir would collide "
            "with the live daemon." % inst)
    # Presence is not enough: brain-env.sh unconditionally exports the uid
    # formula when the var is unset, so every shell-launched process arrives
    # here with production's port already "set". Refuse production's VALUE —
    # the formula and the user env file are the two sources production reads.
    xdg = user_config_dir()
    prod_port = (_read_env_file_key(os.path.join(xdg, 'brain', 'env'),
                                    'BRAIN_DAEMON_PORT')
                 or str(47200 + (os.getuid() % 100)))
    if os.environ["BRAIN_DAEMON_PORT"].strip() == prod_port.strip():
        raise SystemExit(
            "[daemon-config] BRAIN_INSTANCE=%r has BRAIN_DAEMON_PORT=%s — "
            "production's port (the shell fills it in when unset). Pick a "
            "distinct port for the instance." % (inst, prod_port.strip()))
    # The DB-dir half of this collision is enforced where it can't be spoofed:
    # the daemon's bind-time DB lock (get_db_lock_path) makes a second writer
    # on any one brain.db exit as a duplicate regardless of env.


_validate_instance_env()


def resolve_db_dir(trust_env: bool = True) -> str:
    """Where the brain's data lives — the Python half of the resolution
    contract (D-13: one configurable location, every runtime resolves
    through the same chain).

    `trust_env=False` drops the `$BRAIN_DB_DIR` rung and lets the rest of the
    ladder answer. Only for callers asking the OTHER question — *where is
    there an existing brain* rather than *where should the brain be*. The
    default rung is deliberately unconditional (the hook wrappers validate the
    variable, and a dir with no brain.db yet is a legitimate birthplace), so a
    caller that needs a brain to actually be there must opt out of it rather
    than reinterpret the answer.

    Order — mirrors the shell ladder's semantics exactly (a knob dir WITHOUT
    brain.db is an explicit birthplace choice, not a verdict; it must not
    beat a rung that finds a real brain, or shell and Python split-brain):
    BRAIN_DB_DIR env (hook-exported or shell; trusted — the hook wrappers
    validate it) → the user config file (~/.config/brain/env — the one knob
    users edit) if brain.db is there → resolved.env (the record
    resolve-brain-db.sh persists after running its full ladder; adopted only
    if brain.db is actually there, the shell's 4b guard) → the XDG service
    dir (${XDG_DATA_HOME:-~/.local/share}/brain — where new brains are born,
    D-13) if brain.db is there → the legacy dir if brain.db is there → the
    knob dir as the chosen birthplace (absolute paths only — a relative one
    would resolve against each consumer's cwd) → the XDG dir as the final
    default. Python never re-runs the shell ladder's inference (Cowork scan,
    plugin-data adoption, the adoption net); it reads the knob and the
    persisted record — the launchd daemon launch path goes through the shell
    resolver, which enforces the net (the direct-spawn fallback does not).
    """
    d = os.environ.get('BRAIN_DB_DIR')
    if d and trust_env:
        return d
    xdg = user_config_dir()
    cfg = _read_env_file_key(os.path.join(xdg, 'brain', 'env'), 'BRAIN_DB_DIR')
    if cfg and os.path.isfile(os.path.join(cfg, 'brain.db')):
        return cfg
    rec = _read_env_file_key(
        os.path.join(xdg, 'brain', 'resolved.env'), 'BRAIN_DB_DIR')
    if rec and os.path.isfile(os.path.join(rec, 'brain.db')):
        return rec
    native = os.path.join(
        os.environ.get('XDG_DATA_HOME')
        or os.path.join(os.path.expanduser('~'), '.local', 'share'), 'brain')
    if os.path.isfile(os.path.join(native, 'brain.db')):
        return native
    legacy = os.path.join(os.path.expanduser('~'), 'AgentsContext', 'brain')
    if os.path.isfile(os.path.join(legacy, 'brain.db')):
        return legacy
    if cfg and os.path.isabs(cfg):
        return cfg
    return native


def get_daemon_addr():
    """(host, port) the daemon binds and every client connects to."""
    return (DAEMON_HOST, DAEMON_PORT)


def get_pid_path() -> str:
    """Get the daemon PID file path."""
    return os.path.join("/tmp", "brain-daemon-{}{}.pid".format(os.getuid(), _instance_suffix()))


def get_lock_path() -> str:
    """Get the daemon lock file path for startup serialization."""
    return os.path.join("/tmp", "brain-daemon-{}{}.lock".format(os.getuid(), _instance_suffix()))


def get_db_lock_path(db_path: str) -> str:
    """Writer-identity lock keyed on the BRAIN, not the instance label: one
    daemon per brain.db, machine-wide, regardless of BRAIN_INSTANCE. This is
    the backstop the instance-keyed singleton lock cannot provide — a
    mis-enved entity whose launch ladder resolves onto production's DB exits
    as a duplicate instead of becoming a silent second writer (the
    index-corruption class). Keyed on the realpath'd DB directory so symlinked
    and relative spellings of one brain share one inode."""
    real = os.path.realpath(os.path.dirname(os.path.abspath(db_path)))
    return os.path.join("/tmp", "brain-db-{}-{}.lock".format(
        os.getuid(), hashlib.sha1(real.encode()).hexdigest()[:12]))


def get_startup_lock_path() -> str:
    """The RESTARTER coordination lock — serializes concurrent ensure_daemon
    callers so N session boots (re)start at most once.

    Deliberately a DIFFERENT file from get_lock_path(): that flock is the
    daemon's singleton identity, held for its entire serving life. When
    ensure_daemon used the same inode (pre-2026-08-19), acquiring it while a
    healthy daemon served blocked forever (hanging the boot hook — the
    stale-code branch was dead code in production), and holding it through a
    (re)start made every respawn exit as a duplicate. One inode cannot answer
    both "who is the one daemon" and "who is the one restarter"."""
    return os.path.join("/tmp", "brain-startup-{}{}.lock".format(os.getuid(), _instance_suffix()))


def get_status_path() -> str:
    """Get the daemon status file path (read by statusline script)."""
    return os.path.join("/tmp", "brain-status-{}{}.json".format(os.getuid(), _instance_suffix()))


def get_mcp_crash_path() -> str:
    """The report the MCP server leaves when its tool schemas fail to build;
    boot-brain.sh prints it at the next SessionStart and builds the same path
    in shell. Keyed like the rendezvous files, so a test entity never
    overwrites production's report and one user's file never stands in for
    another's."""
    return os.path.join("/tmp", "brain-mcp-crash-{}{}.txt".format(os.getuid(), _instance_suffix()))


def get_daemon_log_path(db_path: str) -> str:
    """Path to the daemon log — where the daemon writes and where every spawn
    site redirects the successor's stdout/stderr. Honors BRAIN_DB_DIR, else the
    DB's own directory. Single source so a restart-spawned and a boot-spawned
    daemon can't log to different files under an env override."""
    db_dir = os.environ.get("BRAIN_DB_DIR") or os.path.dirname(db_path)
    return os.path.join(db_dir, "daemon.log")


def get_maintenance_path() -> str:
    """Get the maintenance mode lock file path."""
    return os.path.join("/tmp", "brain-maintenance-{}{}.lock".format(os.getuid(), _instance_suffix()))


def is_maintenance_mode() -> bool:
    """Check if brain is in maintenance mode (DB operations, no daemon)."""
    return os.path.exists(get_maintenance_path())


def get_recovery_state_path() -> str:
    """Path to the cross-process daemon-recovery state file.

    Holds the cooldown timestamp + circuit-breaker failure count shared by
    every recovery caller (hooks, MCP health monitor) so concurrent callers
    don't double-restart a daemon mid-respawn or loop forever on one that
    can't come back."""
    return os.path.join("/tmp", "brain-recovery-{}{}.json".format(os.getuid(), _instance_suffix()))
