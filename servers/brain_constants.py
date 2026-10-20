"""
brain — Shared Constants

Constants used across multiple brain mixin modules.
Extracted to avoid circular imports (mixins can't import from brain.py).
This module imports nothing from the brain, so per-tool-call hooks can
reach it without paying for the daemon's config import.
"""
import os
import re


def user_config_dir() -> str:
    """`${XDG_CONFIG_HOME:-~/.config}` — the root of the brain's user-owned
    files (`brain/env`, `brain/resolved.env`, `brain/hook-secret`). The Python
    half of the shell resolver's spelling (api-key-env.sh); one accessor so the
    readers of one directory cannot split."""
    return os.environ.get('XDG_CONFIG_HOME') or os.path.join(os.path.expanduser('~'), '.config')


def user_env_file() -> str:
    """`~/.config/brain/env` — the one user-editable config file (API key,
    names, BRAIN_DB_DIR, the gate switches). The Python spelling of
    api-key-env.sh's brain_user_env_file."""
    return os.path.join(user_config_dir(), 'brain', 'env')


def _env_assignment(line: str, key: str):
    """The raw right-hand side when `line` assigns `key` in shell grammar
    (optional `export `), else None. The ONE place the file's line shape is
    known — the reader and the writer below both go through it."""
    s = line.strip()
    if s.startswith('export '):
        s = s[len('export '):].lstrip()
    if not s.startswith(key + '='):
        return None
    return s.split('=', 1)[1].strip()


def read_env_file_key(path: str, key: str):
    """Read one KEY=value from a config file that shell consumers SOURCE —
    so tolerate the shell grammar the same file is written in: optional
    `export `, single/double quotes, inline comments on unquoted values,
    $VAR expansion. resolved.env writes values single-quoted. Returns None
    if absent/unreadable — the readers of one file must not disagree on its
    grammar (that's the split-brain class D-13 kills). Lives in this leaf so
    the per-hook gate (servers/brain_gate.py) can read the file without
    importing daemon_config."""
    try:
        with open(path, errors='replace') as f:
            for raw in f:
                v = _env_assignment(raw, key)
                if v is None:
                    continue
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


def write_env_file_key(path: str, key: str, value: str) -> None:
    """Replace (or, with value '', remove) the `KEY=` line of a config file
    that shell consumers source. Other lines are kept byte-for-byte. Atomic
    replace, mode 600 — the file holds the API key. The writer half of
    read_env_file_key's grammar, in the same leaf for the same reason."""
    lines = []
    try:
        with open(path, errors='replace') as f:
            lines = [l.rstrip('\n') for l in f if _env_assignment(l, key) is None]
    except OSError:
        pass
    if value:
        lines.append('%s=%s' % (key, value))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp.%d' % os.getpid()
    try:
        os.unlink(tmp)   # a leftover from a crashed writer; O_EXCL below refuses anything else
    except OSError:
        pass
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def brain_tmp_dir() -> str:
    """Root for the brain's per-session marker files (session_marker_path).
    Honors BRAIN_TMP_DIR — the test suite pins one per process so two pytest
    runs never collide on fixed filenames; production leaves it unset →
    /tmp."""
    return os.environ.get('BRAIN_TMP_DIR', '/tmp')


def session_marker_path(name, session_id) -> str:
    """This session's marker file for `name` under brain_tmp_dir, or '' with
    no session id (callers then decide without caching).

    A marker holds what a hook must know BEFORE it talks to the daemon, or
    without talking to it at all — the gate's per-session project, the
    prompt hook's first-call budget. State the daemon can read from traces
    does not belong here. Keyed by the whole id: Codex ids are UUIDv7, whose
    leading characters are a timestamp every chat opened in the same minute
    shares."""
    if not session_id:
        return ''
    return os.path.join(brain_tmp_dir(), 'brain-%s-%d-%s' % (
        name, os.getuid(), re.sub(r'[^\w-]', '_', session_id)))


# ═══════════════════════════════════════════════════════════════
# CONFIDENCE: Type defaults and dynamics
# ═══════════════════════════════════════════════════════════════

# Default confidence by node type — how reliable each type tends to be
TYPE_CONFIDENCE = {
    'rule': 0.85, 'decision': 0.80, 'lesson': 0.85, 'correction': 0.95,
    'constraint': 0.85, 'convention': 0.80, 'procedure': 0.85,
    'purpose': 0.80, 'mechanism': 0.75, 'impact': 0.75,
    'mental_model': 0.65, 'hypothesis': 0.50, 'uncertainty': 0.40,
    'concept': 0.70, 'context': 0.60, 'task': 0.70,
    'intuition': 0.40, 'thought': 0.35,
    'pattern': 0.60, 'tension': 0.55, 'aspiration': 0.50,
    'person': 0.85, 'project': 0.80, 'file': 0.75,
    'vocabulary': 0.90, 'validation': 0.90,
    # Legacy code cognition types
    'fn_reasoning': 0.75, 'param_influence': 0.70, 'code_concept': 0.70,
    'arch_constraint': 0.85, 'causal_chain': 0.70, 'bug_lesson': 0.85,
    'comment_anchor': 0.80,
    'capability': 0.70, 'reasoning_trace': 0.65,
}

# Page size cap
MAX_PAGE_SIZE = 100

# Zero-Memory boot block (the Nursery's spoken half, rendered by brain_voice).
# Fires while the brain is young; retires only when BOTH thresholds are
# exceeded — at least this old AND at least this many lived (non-seed)
# memories.
ZERO_MEMORY_MIN_DAYS = 10
ZERO_MEMORY_MIN_LIVED = 100

# Terms dropped from FTS5 queries (Fts5DAL._sanitize_query).
STOP_WORDS = {
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of',
    'that', 'this', 'it', 'be', 'as', 'by', 'from', 'has', 'was', 'are', 'were', 'been', 'have', 'had',
    'not', 'they', 'their', 'will', 'would', 'can', 'could', 'do', 'does', 'did', 'should', 'may',
    'might', 'shall', 'than', 'into', 'about', 'also', 'its', 'just', 'more', 'other', 'some', 'such',
    'then', 'there', 'these', 'what', 'when', 'where', 'who', 'how', 'all', 'each', 'both', 'few',
    'most', 'any', 'own', 'same', 'over', 'only', 'very', 'after', 'before', 'between', 'under',
    'above', 'out', 'up', 'down', 'use', 'used', 'using', 'new', 'like', 'get', 'set', 'one', 'two',
}

# Relevance floor — each result must clear it on its own (v8.7: was
# all-or-nothing, gated by the top result). The Haiku distiller is the primary
# quality gate; this floor catches obvious noise.
RELEVANCE_FLOOR_PRIMARY = 0.25

# Title-match boost ceiling — read only by the eval/oracle_audit title-boost probes.
TITLE_MATCH_BOOST = 0.3

# Stopwords for the idf lane's title tokens (recall_laf). df-over-titles misprices
# conversational words (titles are noun-phrases: 'does' df=58 scores as rare as
# 'fatigue' df=41) — these never earn a title boost regardless of idf.
_TITLE_BOOST_STOPWORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'of', 'to', 'in', 'on', 'at',
    'for', 'with', 'from', 'by', 'as', 'over', 'under', 'about', 'into',
    'out', 'up', 'down', 'off', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'am', 'do', 'does', 'did', 'done', 'have', 'has', 'had',
    'will', 'would', 'can', 'could', 'should', 'shall', 'may', 'might',
    'how', 'what', 'whats', 'when', 'where', 'which', 'who', 'whom', 'why',
    'that', 'this', 'these', 'those', 'there', 'here', 'i', 'me', 'my',
    'we', 'our', 'us', 'you', 'your', 'yours', 'it', 'its', 'they', 'them',
    'their', 'he', 'she', 'his', 'her', 'again', 'also', 'just', 'so',
    'too', 'not', 'no', 'yes', 'if', 'then', 'than', 'any', 'all', 'some',
    'more', 'most', 'other', 'very',
})

# Noise floor — minimum blended score for a candidate.
# v9: Raised from 0.05 to cut pure embedding noise before scoring.
NOISE_FLOOR_THRESHOLD = 0.15


# recall_episodes — episodic pull over the traces layer (TraceDAL.filter_events +
# optional semantic re-rank over the existing trace_embeddings). trace→trace, returns
# full episode records.
EPISODE_DEFAULT_LIMIT = 10           # episodes returned by default
EPISODE_MAX_LIMIT = 500              # hard cap on a single pull
EPISODE_DEFAULT_WINDOW_DAYS = 7      # default created_at lower bound when no session/time scope
                                     # (perf: bounds the contains LIKE scan over a big append-only table)
EPISODE_SEMANTIC_CANDIDATE_CAP = 500  # max candidates fetched before semantic (query) re-rank

# Retrieval stats — surfacer guidance thresholds
RETRIEVAL_LOW_CONFIDENCE = 0.35   # Top score below this → surfacer gets low-confidence warning


# ═══════════════════════════════════════════════════════════════
# S2 MAINTENANCE SCHEDULING — brain.run_maintenance_if_due()
# ═══════════════════════════════════════════════════════════════
# When does the brain consider running S2? Two conditions:
#
# 1. No USER PROMPT for MAINTENANCE_IDLE_THRESHOLD_SECONDS — the clock is
#    brain.activity.last_user_activity, reset only by hook_recall. Tool-use
#    hooks, internal IPC and pings deliberately do NOT count: the signal is
#    "the operator isn't typing", not "nothing is talking to the daemon".
#    Safer to spend API budget then than mid-turn.
# 2. MAINTENANCE_MIN_INTERVAL_SECONDS have passed since the previous run,
#    tracked in brain_meta (s2_last_run_ts). Persists across daemon
#    restarts so reboots don't re-trigger maintenance immediately.
#
# In steady state — no new encoding → no new merge/place/heal candidates —
# a fire is cheap: decoders scan, find 0 work, encoders skip. So tune these
# for "how often to catch backlog during active use," not "how cheap is
# a no-op fire."
MAINTENANCE_IDLE_THRESHOLD_SECONDS = 3 * 60   # 3 min idle → more chances
                                              # to fire during active work
MAINTENANCE_MIN_INTERVAL_SECONDS = 60 * 60    # 1 hour between runs — throttled
                                              # to reduce S2-vs-S1 contention
                                              # (S2 cycles run minutes and share
                                              # the org rate limit with surface)
# In addition to the time gate, require at least this many S1 Encoder (Scribe)
# runs since the last S2 run. S2's material is encoded nodes — a recall (S1
# Surface) reads the graph and creates nothing to consolidate, and surfaces
# fire across many parallel sessions. So cadence ties to ENCODING, not recalls:
# no new encoded nodes → nothing for S2 to do → don't fire. The Scribe runs
# every 5th conversational turn, so 2 runs ≈ 10 turns of new material. The
# FORCE_FIRE valve below still overrides this for multi-day stalls.
MAINTENANCE_MIN_ENCODE_RUNS = 2
# Safety valve: if maintenance hasn't fired in this long, force-fire on the
# next poll regardless of idle. Catches multi-day stalls (observed 2.5d gaps
# between 04-28 and 04-30 when the idle gate never opened).
MAINTENANCE_FORCE_FIRE_SECONDS = 24 * 60 * 60

# Boot grace period (added 2026-05-08): never fire S2 maintenance during the
# first N seconds after daemon start, regardless of idle/min-interval state.
# Why: maintenance with `last_activity_ts == 0.0` was treated as infinitely
# idle and fired on the very first daemon poll, which made first-recall-
# after-restart consistently time out (consolidation encoder held the write
# lock for tens of seconds while making LLM calls). Boot should not be
# overloaded with many things — the user gets a working brain immediately;
# maintenance can wait its turn after the first interaction.
MAINTENANCE_BOOT_GRACE_SECONDS = 90

# Eval brain marker: a file of this name beside brain.db marks a brain made for
# an eval (a seed, a premade eval brain, or a copy of the operator's). Its
# modification time is the moment the eval took the brain over. On a marked
# brain nothing spends tokens on its own: S2 never schedules itself, and the S1
# Scribe ignores every session whose last turn predates the mark and encodes an
# idle tail only when scribe_flush asks. s2_force and scribe_flush still run on
# demand, and the Scribe's 5-turn encode keeps production's rhythm on the
# eval's own conversation. Written by Brain.mark_eval_brain() before any daemon
# opens the brain; read once when the brain opens (Brain.eval_brain_since()).
# The mark separates sessions, not turns: a session that continues across it
# brings its earlier unencoded turns along, so a premade eval brain is saved
# fully encoded (scribe_flush first) or the eval uses fresh session ids.
EVAL_BRAIN_MARKER = 'eval-brain'

# Log retention + orphan sweep cadence, run as a DBMaintenance task. Housekeeping
# with no deadline — hourly reclaims space long before it matters. It lived in the
# Claude Code idle hook until that event stopped firing and it went six weeks
# without running; the scheduler thread has no such dependency.
LOGS_MAINTENANCE_INTERVAL_S = 60 * 60

# Session live time: a session whose saved state has not changed for this many
# days is dead — the log retention pass prunes its session_state row, and a
# later resume starts it fresh. Override per brain with brain_meta
# `session.live_days`.
SESSION_LIVE_DAYS = 30

# Weekly durable size snapshot (file + per-table via dbstat) for both DBs,
# written to debug_log so growth is a query, not archaeology. Trend data —
# weekly is plenty; the daemon log already carries per-tick file stats.
DB_SIZE_TELEMETRY_INTERVAL_S = 7 * 24 * 60 * 60

# ── LLM rejection backoff ──
# How long LLM features stay paused after the provider REFUSES a call (a dead
# key, an exhausted quota). A refusal costs no tokens, so this ladder trades
# resume latency against noise, not money: an operator who re-enables the same
# key, or raises its usage limit, resumes within the hour with no restart, while
# a key that stays dead stops being hammered after the first few minutes.
# Escalates per consecutive rejection; the last value is the ceiling.
LLM_REJECT_BACKOFF_MINUTES = (5, 15, 30, 60)

# Strikes age out after a quiet stretch longer than the ladder's ceiling, so a
# rejection weeks from now starts at 5 minutes instead of inheriting an hour.
LLM_REJECT_STRIKE_RESET_SECONDS = 2 * 60 * 60

# ── LLM transport ──
# Hard upper bound on any single Anthropic SDK call (S1 surface, S1 encode, S2
# encoders). The SDK default is roughly 600s but is measured against
# time.monotonic(), which does NOT advance while the process is suspended
# (macOS sleep). A call started right before sleep can therefore hang
# indefinitely after wake. A post-sleep hang is recovered reactively
# (ensure_daemon at session start / the MCP health monitor during a session,
# both force-restarting via launchctl kickstart -k); this constant bounds
# normal-mode hangs (slow API, throttled response, etc.) so a stuck call
# doesn't tie up a worker forever. Community encoder round 2 on cold-cache
# batches can legitimately take ~218s; 600s leaves headroom without inviting
# silence.
#
# It lives here rather than in scales/runner.py because it is daemon-level
# transport policy, not encoder-lane detail: brain.py builds the shared client
# with it too, and was reaching down into the encoder lane for it.
ANTHROPIC_CLIENT_TIMEOUT = 600.0

# Connect gets its OWN bound. ANTHROPIC_CLIENT_TIMEOUT's 600s is read-side
# reasoning (long encoder generations); passed as a bare float it applies to
# connect too, granting "can't reach the server" the same patience as "slow
# generation" — a flapping network held an encode permit ~53 min that way
# (e2dc24d3, 2026-08-18). Construction sites combine the two via
# httpx.Timeout(ANTHROPIC_CLIENT_TIMEOUT, connect=ANTHROPIC_CONNECT_TIMEOUT):
# TCP+TLS to the API lands in ~1s; 10s absorbs SYN retransmits on a congested
# link without letting a dead network eat the read budget.
ANTHROPIC_CONNECT_TIMEOUT = 10.0

# ── Dashboard (keyless-onboarding notices) ──
# The dashboard's own default lives in dashboard/server.py (deliberately
# servers-decoupled); this is the DAEMON-side single source for building the
# /setup URL in keyless notices — daemon_hooks (per-turn PAUSED block) and
# brain_voice (boot banner) both read it here instead of re-inlining the
# literal port (code review 2026-07-17, contract-first).
DASHBOARD_DEFAULT_PORT = 47303


def dashboard_url(path: str = '') -> str:
    """The dashboard's URL, `path` appended. DASHBOARD_PORT from the process
    environment, else from the user env file (the boot gate asks before the
    shell has sourced it; the daemon's load_env may not have run yet either),
    else the fixed default — the same ladder the dashboard service itself
    resolves."""
    import os
    port = (os.environ.get('DASHBOARD_PORT')
            or read_env_file_key(user_env_file(), 'DASHBOARD_PORT')
            or DASHBOARD_DEFAULT_PORT)
    return "http://localhost:%s%s" % (port, path)


def dashboard_setup_url() -> str:
    """The /setup URL keyless notices point users at."""
    return dashboard_url('/setup')
