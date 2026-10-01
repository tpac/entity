"""DB path resolution + read-only SQLite connect helper.

The dashboard reads two databases:
  brain.db       — nodes, edges, embeddings
  brain_logs.db  — traces, hook errors, telemetry

All connections open in read-only mode (`mode=ro`) — the dashboard never
writes to brain. Writers go through the daemon.

Payload files: the dashboard is a sanctioned direct reader of
{BRAIN_DB_DIR}/payloads/ (docs/TRACE-MODES-DESIGN.md) — its charter is
reading the substrate when the daemon is down, so payload reads never route
through daemon TCP. `read_payload_pointer` resolves a db_dir-relative
pointer from a trace row; `chain_payload_files` finds a chain's payload
files by layout (payloads/{date}/{chain}/NNN-{kind}.{ext}) for readers
whose trace rows carry no pointer (judge, consolidation prompts).
"""

import os
import re
import sqlite3
from contextlib import contextmanager

from .log import warn


def _read_env_file_key(path: str, key: str):
    """Shell-grammar-tolerant KEY=value read (export prefix, quotes, inline
    comments, $VAR) — mirror of servers.daemon_config._read_env_file_key;
    kept in sync by tests/test_db_resolution.py."""
    try:
        with open(path, errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if line.startswith("export "):
                    line = line[len("export "):].lstrip()
                if not line.startswith(key + "="):
                    continue
                v = line.split("=", 1)[1].strip()
                if v[:1] in ('"', "'"):
                    end = v.find(v[0], 1)
                    v = v[1:end] if end > 0 else v[1:]
                else:
                    v = v.split(" #", 1)[0].rstrip()
                if v:
                    return os.path.expanduser(os.path.expandvars(v))
    except OSError:
        pass
    return None


def _brain_dir() -> str:
    """Resolve the brain data dir — the dashboard's sanctioned mirror of
    daemon_config.resolve_db_dir (the disconnection contract forbids importing
    servers.*; both read the same D-13 chain): BRAIN_DB_DIR env → the user
    config file (~/.config/brain/env) if brain.db is there → resolved.env
    (the shell resolver's record, only if brain.db is actually there) → the
    XDG service dir (~/.local/share/brain, where new brains are born) if
    brain.db is there → legacy if brain.db is there → the knob dir as the
    chosen birthplace (absolute only) → the XDG dir as the final default. A
    knob dir without brain.db must not beat a rung that finds a real brain —
    the shell demotes it the same way."""
    d = os.environ.get("BRAIN_DB_DIR")
    if d:
        return d
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config")
    cfg = _read_env_file_key(os.path.join(xdg, "brain", "env"), "BRAIN_DB_DIR")
    if cfg and os.path.isfile(os.path.join(cfg, "brain.db")):
        return cfg
    rec = _read_env_file_key(
        os.path.join(xdg, "brain", "resolved.env"), "BRAIN_DB_DIR")
    if rec and os.path.isfile(os.path.join(rec, "brain.db")):
        return rec
    native = os.path.join(
        os.environ.get("XDG_DATA_HOME")
        or os.path.join(os.path.expanduser("~"), ".local", "share"), "brain")
    if os.path.isfile(os.path.join(native, "brain.db")):
        return native
    legacy = os.path.join(os.path.expanduser("~"), "AgentsContext", "brain")
    if os.path.isfile(os.path.join(legacy, "brain.db")):
        return legacy
    if cfg and os.path.isabs(cfg):
        return cfg
    return native


def brain_db_path() -> str:
    return os.path.join(_brain_dir(), "brain.db")


def logs_db_path() -> str:
    return os.path.join(_brain_dir(), "brain_logs.db")


def read_payload_pointer(pointer: str):
    """Read a payload file by its db_dir-relative pointer (from a trace
    row's ref_id/metadata). Returns str or None (pruned / missing / bad
    pointer). Pointers are data, not a path authority — absolute paths and
    traversal are rejected, mirroring brain.read_payload's guard."""
    if not pointer or not isinstance(pointer, str):
        return None
    norm = os.path.normpath(pointer)
    if (os.path.isabs(norm) or norm.startswith("..")
            or not norm.startswith("payloads" + os.sep)):
        return None
    try:
        with open(os.path.join(_brain_dir(), norm),
                  encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def payload_sort_key(path: str):
    """(seq, attempt) ordering for recorder filenames — NNN-{kind}.{ext}
    with collision ordinals as NNN-{kind}.{A}.{ext}. A plain basename sort
    is WRONG: '000-judge.2.json' < '000-judge.json' lexically ('2' < 'j'),
    so sorted(...)[-1] would return the OLDEST attempt. The base file is
    attempt 1; ordinals start at 2."""
    name = os.path.basename(path)
    m_seq = re.match(r"(\d+)-", name)
    m_att = re.search(r"\.(\d+)\.[^.]+$", name)
    return (int(m_seq.group(1)) if m_seq else -1,
            int(m_att.group(1)) if m_att else 1)


def chain_payload_files(chain_id: str, kind: str = ""):
    """List a chain's payload files, sorted (seq, attempt) — [-1] is the
    newest write. Returns absolute paths. `kind` filters to one payload
    kind (the filename embeds the kind verbatim: NNN-{kind}.{ext}). Empty
    when the chain never recorded or its files were pruned. chain_id lands
    in a path segment — reject separators/traversal outright."""
    if (not chain_id or "/" in chain_id or os.sep in chain_id
            or ".." in chain_id):
        return []
    root = os.path.join(_brain_dir(), "payloads")
    if not os.path.isdir(root):
        return []
    out = []
    for day in os.listdir(root):
        chain_dir = os.path.join(root, day, chain_id)
        if not os.path.isdir(chain_dir):
            continue
        for name in os.listdir(chain_dir):
            if kind and ("-%s." % kind) not in name:
                continue
            out.append(os.path.join(chain_dir, name))
    return sorted(out, key=payload_sort_key)


@contextmanager
def ro_connect(path: str, timeout: float = 3):
    """Open a read-only connection. Yields None if file is missing; the caller
    decides how to degrade (usually: return an empty list)."""
    if not os.path.exists(path):
        yield None
        return
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
    try:
        yield conn
    finally:
        conn.close()


def fetch_by_id(conn, table: str, columns: str, ids) -> dict:
    """Fetch whole rows by PRIMARY id, keyed by id: {id: row}.

    Centralizes the one idiom three queries/ modules shared — placeholder +
    single-table `WHERE id IN (...)` + `{r[0]: r}`. Scope is deliberately
    narrow: JOIN / multi-predicate / dual-`IN` / non-`id`-key lookups (the edge
    and metadata queries elsewhere in queries/) don't fit this shape and stay
    inline. The first selected column MUST be the id (it becomes the dict key).
    `table` and `columns` are code-controlled constants, never user input —
    there is no injection surface; `ids` are bound parameters. Falsy ids are
    dropped; an empty set short-circuits to {} (no invalid `IN ()`).

    Liveness-NEUTRAL by design: a point-fetch-by-id returns archived rows too,
    matching the brain's fetch-by-id contract (liveness is enforced at call
    sites that need it, e.g. _enrich_consolidation uses the `archived` flag as
    its own tiebreaker). This helper must never silently filter what a caller
    asked for by id.
    """
    ids = [i for i in ids if i]
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        "SELECT %s FROM %s WHERE id IN (%s)" % (columns, table, placeholders),
        ids,
    ).fetchall()
    return {r[0]: r for r in rows}


def direct_query(sql: str, args=(), db_path: str = None):
    """Read-only query against brain.db (or any other path).

    Returns [] on missing file or any sqlite error — silent degradation is
    correct here because the dashboard must keep painting even when the
    underlying DB is being recreated.
    """
    path = db_path or brain_db_path()
    if not os.path.exists(path):
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=3)
        rows = conn.execute(sql, args).fetchall()
        conn.close()
        return rows
    except Exception as e:
        warn('db', 'direct_query against %s failed' % path, exc=e)
        return []
