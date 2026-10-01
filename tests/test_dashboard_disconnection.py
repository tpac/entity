"""Contract: the dashboard stays disconnected from the brain.

Two invariants this test locks in:

1. **No imports from servers.* or hooks.* in dashboard/.**
   The dashboard is a passive observer over the brain's databases plus a
   TCP socket to the daemon. The moment it imports a Brain mixin, a DAL
   class, or a hook helper, "passive observer" stops being true — code in
   the dashboard can now reach into the brain process state, mutate it,
   or accidentally depend on internal contracts that are free to change
   under it. The standing principle: "Dashboard must reflect reality — not
   be its own separate data funnel"; reflection is one-way.

2. **No non-read-only SQLite connections in dashboard/.**
   Every connection opens with ``mode=ro``. Writes go through the daemon
   (which the dashboard speaks to over TCP, never via in-process imports).
   This keeps the dashboard from ever holding an exclusive writer lock on
   ``brain.db`` while the daemon needs one — an old class of bug that
   produced index corruption (see feedback_no_sqlite3_cli_against_live_brain
   in the user's auto-memory).

If you must talk to the daemon from the dashboard, use
``dashboard.daemon_client.daemon_send`` — that's the ONE sanctioned bridge.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / 'dashboard'

# Modules that, if imported from dashboard, would breach the boundary.
# `dashboard.daemon_client` is the SOLE allowed bridge to the daemon.
FORBIDDEN_IMPORT_PREFIXES = ('servers', 'hooks')


def _python_files():
    return [p for p in DASHBOARD.rglob('*.py') if p.is_file()]


def _imports_in(path: Path):
    """Yield (module, lineno) for every `import X` / `from X import …` in path."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as e:
        raise AssertionError('dashboard file %s is not parseable: %s' % (path, e))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield (alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            # Skip relative imports — those are intra-package.
            if node.level and node.level > 0:
                continue
            if node.module:
                yield (node.module, node.lineno)


def test_no_imports_from_servers_or_hooks():
    """dashboard/*.py must not import from `servers.*` or `hooks.*`.

    If this fails, refactor the dashboard to read what it needs from the
    DB (mode=ro) or to ask the daemon via TCP. Don't import.
    """
    violations = []
    for p in _python_files():
        rel = p.relative_to(ROOT).as_posix()
        for module, lineno in _imports_in(p):
            top = module.split('.', 1)[0]
            if top in FORBIDDEN_IMPORT_PREFIXES:
                violations.append('%s:%d  import %s' % (rel, lineno, module))
    assert not violations, (
        'Dashboard breached the disconnection boundary:\n  '
        + '\n  '.join(violations)
        + '\n\nUse dashboard.daemon_client for daemon talk, or open the DB '
        'in mode=ro via dashboard.db.ro_connect.'
    )


# Open-mode patterns to flag. We match the literal strings the dashboard
# uses to open SQLite. Any non-ro mode (rw, rwc, memory) is a writer; that's
# a daemon-side concern, not the dashboard's.
_NON_RO_PATTERNS = [
    re.compile(r"sqlite3\.connect\s*\(\s*['\"][^'\"]+['\"]\s*\)"),  # bare path → writer
    re.compile(r"mode=(rw|rwc|memory)"),
]
_RO_PATTERN = re.compile(r"mode=ro")


def _strip_comments_and_strings(text: str) -> str:
    """Return `text` with comment lines and triple-quoted blocks removed.

    Used by the SQLite-connect scanner so docstrings and comments that
    mention `sqlite3.connect(...)` (e.g. to explain WHY a file no longer
    does that) don't trip the contract test. Heuristic — handles the
    docstring conventions actually used in this repo (triple-double-quote
    blocks on their own line, single-line comments) and nothing fancier.
    """
    out_lines = []
    in_triple = False  # currently inside a triple-quoted block
    for line in text.splitlines():
        stripped = line.lstrip()
        # Toggle on lines that contain a triple-quote. Handles single-line
        # docstrings ("""x""") and multi-line block starts/ends. Doesn't
        # handle triple-quote inside a regular string literal — we don't
        # use that pattern in this codebase.
        triple_count = stripped.count('"""') + stripped.count("'''")
        if triple_count and not in_triple:
            in_triple = (triple_count % 2 == 1)
            continue  # drop the opening line entirely
        if in_triple:
            if triple_count:
                in_triple = (triple_count % 2 == 0)
            continue  # drop docstring body
        if stripped.startswith('#'):
            continue
        out_lines.append(line)
    return '\n'.join(out_lines)


def test_all_sqlite_connects_are_read_only():
    """Every sqlite3.connect call in dashboard/ must use mode=ro.

    The test inspects the source text rather than runtime behavior, since
    runtime-only verification would miss code paths the test never hits.
    Any connect that opens a non-ro URI or a bare file path → writer
    capability → violation.

    Comments and docstrings are stripped before scanning — that lets a
    module's docstring legitimately mention `sqlite3.connect(...)` to
    explain WHY the module no longer calls it.
    """
    violations = []
    for p in _python_files():
        rel = p.relative_to(ROOT).as_posix()
        text = _strip_comments_and_strings(p.read_text())
        for lineno, line in enumerate(text.splitlines(), start=1):
            if 'sqlite3.connect' not in line:
                continue
            if _RO_PATTERN.search(line):
                continue
            # No mode=ro on this line — verify the connect is non-writer.
            # The dashboard's connect helper always passes mode=ro; bare
            # connects elsewhere are the violation.
            for pat in _NON_RO_PATTERNS:
                if pat.search(line):
                    violations.append('%s:%d  %s' % (rel, lineno, line.strip()[:120]))
                    break
            else:
                # `sqlite3.connect` present without mode= and without one of
                # the writer patterns either — could be the helper itself
                # building a URI dynamically. Flag for review.
                violations.append('%s:%d  %s' % (rel, lineno, line.strip()[:120]))
    assert not violations, (
        'Dashboard opened a non-read-only SQLite connection:\n  '
        + '\n  '.join(violations)
        + '\n\nUse dashboard.db.ro_connect() — it pins mode=ro.'
    )


# ── 3. Mirrored constants must match their servers-side source ──
# The disconnection contract forbids importing servers.*, so any constant the
# dashboard needs is REPLICATED by hand. That trades one failure mode for
# another: the copy silently drifts. Mirror-and-pin — the mirror is the only
# legal mechanism, and this is the pin that makes drift loud.
#
# Importing servers.trace_contract HERE is fine and is not a contract breach:
# the rule binds dashboard/ code, not the test suite that guards it.

def test_mirrored_ref_type_constants_match_trace_contract():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from servers.trace_contract import (
        EMITTER_REF_TYPES, OPERATOR_DIALOGUE_REF_TYPES, RESIDUE_REF_TYPES,
    )
    from dashboard.queries.s2_runs import (
        _EMITTER_REF_TYPES, _NON_RUN_REF_TYPES, _RESIDUE_REF_TYPES,
    )
    from dashboard.queries.traces import _SAID

    assert tuple(_RESIDUE_REF_TYPES) == tuple(RESIDUE_REF_TYPES), (
        'dashboard._RESIDUE_REF_TYPES drifted from trace_contract.RESIDUE_REF_TYPES '
        '— the run-card queries would count encoder notes as runs'
    )
    assert tuple(_EMITTER_REF_TYPES) == tuple(EMITTER_REF_TYPES), (
        'dashboard._EMITTER_REF_TYPES drifted from trace_contract.EMITTER_REF_TYPES '
        '— per-write mutation rows would render as phantom S2 run cards'
    )
    # The derived set the queries actually filter on must cover both families,
    # or one of them leaks back into the run cards.
    assert set(_NON_RUN_REF_TYPES) == set(RESIDUE_REF_TYPES) | set(EMITTER_REF_TYPES)
    # Bound as SQL parameters, so the literal scan below can't see it.
    assert tuple(_SAID) == tuple(OPERATOR_DIALOGUE_REF_TYPES), (
        'dashboard traces._SAID drifted from OPERATOR_DIALOGUE_REF_TYPES — the '
        'Streams identity cue would read a ref_type nothing writes')


def test_mirrored_batch_unwrap_matches_dispatch():
    # s2_runs reads recorded brain_batch operations the way dispatch ran
    # them; a drift shows links for ops that never executed (or hides ones
    # that did).
    import json
    import sys
    sys.path.insert(0, str(ROOT))
    from servers.contract import unwrap_operations
    from servers.dispatch_write import _element_as_dict
    from dashboard.queries.s2_runs import _batch_operations

    op = {"op": "connect", "source_id": "a", "target_id": "b", "relation": "r"}
    cases = [
        [op, op],
        json.dumps([op, op]),
        [op, json.dumps(op)],
        json.dumps([op, json.dumps(op)]),
        [op, "not json", 7, json.dumps([1])],
        "not json",
        json.dumps({"op": "connect"}),
    ]
    for ops in cases:
        listed = unwrap_operations(ops) if isinstance(ops, str) else ops
        want = ([] if not isinstance(listed, list)
                else [_element_as_dict(e)[0] or {} for e in listed])
        got = _batch_operations({"input": {"operations": ops}})
        assert got == want, (ops, got, want)


# A positive filter (`ref_type = 'x'`) survives new types but not a rename:
# after one, the query silently reads rows nothing writes any more. Every
# ref_type literal a query names must be one the write path accepts
# (validate_trace_event checks writes against REF_TYPES).
_REF_TYPE_COMPARE = re.compile(r"""ref_type['"\]]*\s*(?:==|!=|=)\s*'([a-z0-9_]+)'""")
_REF_TYPE_IN = re.compile(r"ref_type\s+(?:NOT\s+)?IN\s*\(([^)]*)\)", re.IGNORECASE)


def _ref_type_literals(text):
    """Yield every ref_type literal compared or IN-listed in `text`."""
    yield from _REF_TYPE_COMPARE.findall(text)
    for group in _REF_TYPE_IN.findall(text):
        yield from re.findall(r"'([a-z0-9_]+)'", group)


def test_queried_ref_types_are_writable():
    import sys
    sys.path.insert(0, str(ROOT))
    from servers.trace_contract import REF_TYPES

    writable = {rt for types in REF_TYPES.values() for rt in types}
    unknown = sorted(
        '%s: %s' % (p.relative_to(ROOT), rt)
        for p in (DASHBOARD / 'queries').glob('*.py')
        for rt in _ref_type_literals(p.read_text())
        if rt not in writable)
    assert not unknown, (
        'Dashboard queries name ref_types no writer can produce:\n  '
        + '\n  '.join(unknown))


def test_mirrored_daemon_host_matches_daemon_config():
    import sys
    sys.path.insert(0, str(ROOT))
    from servers.daemon_config import DAEMON_HOST
    from dashboard import daemon_client as dcl

    assert dcl.DAEMON_HOST == DAEMON_HOST, (
        'dashboard DAEMON_HOST drifted from daemon_config.DAEMON_HOST — the '
        'dashboard would dial an address the loopback-only daemon never binds'
    )


def test_mirrored_primary_aspects_match_the_registry(tmp_path, monkeypatch):
    # The graph colours each node type and relation by its primary aspect; a
    # drifted rule colours by an aspect the brain doesn't read it as.
    import json
    import sys
    sys.path.insert(0, str(ROOT))
    from servers.aspect_store import SEED_ASPECTS_JSON_PATH
    from servers.aspects import AspectRegistry
    from dashboard.queries.aspects import primary_aspects

    seed = json.loads(Path(SEED_ASPECTS_JSON_PATH).read_text())
    synthetic = {
        '_schema': {'node_types': ['doc_only']},
        'first': {'node_types': ['lesson', {'type': 'rule'}, {'bad': 1}, 7],
                  'edge_relations': ['refines', {'verb': 'supersedes'},
                                     'co_anchored']},
        'second': {'node_types': ['lesson', 'moment'],
                   'edge_relations': ['supersedes', 'grounds'],
                   },
        'broken': {'node_types': 'not-a-list', 'edge_relations': None},
        'not_an_aspect': 'a string entry',
        'blank': {'node_types': ['', {'type': 5}], 'edge_relations': ['']},
        'noise': {'node_types': ['moment'],
                  'edge_relations': ['co_anchored', 'community_member']},
    }
    for data in (seed, synthetic):
        path = tmp_path / 'aspects_v1.json'
        path.write_text(json.dumps(data))
        monkeypatch.setenv('ASPECTS_JSON_PATH', str(path))
        got = primary_aspects()
        registry = AspectRegistry.from_dict(None, data)
        assert got['edge_relations'] == registry.primary_edge_map()
        types = {t for a in registry.all().values() for t in a.node_types}
        assert got['node_types'] == {
            t: registry.by_node_type(t).name for t in types}


def test_mirrored_user_message_fields():
    # The encode chip skips wake envelopes by the marker; the Surface card skips
    # turns the hook registered without surface by the stamp. A drifted copy
    # counts the wrong turns.
    import sys
    sys.path.insert(0, str(ROOT))
    from servers.trace_contract import REGISTER_ONLY_FIELD, WAKE_ENVELOPE_MARKER
    from dashboard.queries import _meta

    assert _meta.WAKE_ENVELOPE_MARKER == WAKE_ENVELOPE_MARKER
    assert _meta.REGISTER_ONLY_FIELD == REGISTER_ONLY_FIELD


# ── The one sanctioned wire copy actually works ────────────────────────────
# dashboard/daemon_client.daemon_send is a DELIBERATE duplicate of
# servers.daemon_client.send_command (this file's first test is why: the
# dashboard must run with servers/ absent or broken). Step 6d rewrote its
# framing to match the owner's — read until the newline the daemon terminates
# every reply with, instead of re-parsing the whole buffer as JSON after each
# chunk — and that rewrite had no behavioral coverage at all.

import json as _json
import socket as _socket
import threading as _threading


def _fake_daemon(handler):
    """One-shot TCP server. Returns (host, port)."""
    srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()

    def run():
        try:
            conn, _ = srv.accept()
            try:
                conn.recv(65536)
                handler(conn)
            finally:
                conn.close()
        except OSError:
            pass
        finally:
            srv.close()

    _threading.Thread(target=run, daemon=True).start()
    return host, port


def _send(monkeypatch, handler, cmd="ping", timeout=3):
    from dashboard import daemon_client as dcl
    host, port = _fake_daemon(handler)
    monkeypatch.setattr(dcl, "DAEMON_HOST", host)
    monkeypatch.setattr(dcl, "DAEMON_PORT", port)
    return dcl.daemon_send(cmd, timeout=timeout)


def test_daemon_send_returns_the_result_payload(monkeypatch):
    payload = {"ok": True, "result": {"pid": 42, "uptime_seconds": 7}}
    got = _send(monkeypatch, lambda c: c.sendall(_json.dumps(payload).encode() + b"\n"))
    assert got == {"pid": 42, "uptime_seconds": 7}


def test_daemon_send_reassembles_a_reply_split_across_chunks(monkeypatch):
    # The framing that matters: a large reply arrives in several segments and
    # is only complete at the newline.
    big = {"ok": True, "result": {"blob": "x" * 300000}}
    raw = _json.dumps(big).encode() + b"\n"

    def handler(conn):
        for i in range(0, len(raw), 8192):
            conn.sendall(raw[i:i + 8192])

    got = _send(monkeypatch, handler)
    assert got["blob"] == "x" * 300000


def test_daemon_send_returns_none_on_daemon_level_error(monkeypatch):
    got = _send(monkeypatch, lambda c: c.sendall(b'{"ok": false, "error": "nope"}\n'))
    assert got is None


def test_daemon_send_returns_none_when_the_daemon_is_down(monkeypatch):
    from dashboard import daemon_client as dcl
    srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    host, port = srv.getsockname()
    srv.close()
    monkeypatch.setattr(dcl, "DAEMON_HOST", host)
    monkeypatch.setattr(dcl, "DAEMON_PORT", port)
    assert dcl.daemon_send("ping", timeout=1) is None
    assert dcl.daemon_alive() is False
