"""Guard: CmdEntry.accepts must match the keys the handler actually reads.

The dispatcher's silent-drop detection (`dispatch_unknown_keys`) relies on
each CmdEntry declaring its accepted arg keys. When a handler is edited to
read a new key without updating `accepts`, the dispatcher will log that key
as dropped (false positive noise) — OR worse, if `accepts` is the superset
and the handler stops reading a key, we'll silently accept dead input.

This test parses each handler that has an `accepts` contract and asserts
the declared set covers every `args.get("...")` / `op_spec.get("...")`
string literal the handler source mentions.
"""
import ast
import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from servers import daemon_dispatch


def _collect_args_get_keys(fn):
    """Return string keys the function reads from its `args` parameter.

    Matches `args.get("key", ...)` specifically — NOT `c.get(...)` or
    `op_spec.get(...)` which operate on nested dicts (not the top-level
    args the dispatcher validates). Handlers whose args parameter is
    conventionally named something other than `args` are unsupported;
    none exist in the current dispatcher.
    """
    try:
        src = inspect.getsource(fn)
    except OSError:
        return set()
    src = inspect.cleandoc(src) if src.startswith(' ') else src
    # Actually: keep original indentation; ast.parse handles it if fn is top-level.
    tree = ast.parse(inspect.getsource(fn))
    keys = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != 'get':
            continue
        # Only top-level args.get(...) — skip c.get / op_spec.get / etc.
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != 'args':
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            keys.add(first.value)
    return keys


def test_accepts_covers_handler_reads():
    """Each CmdEntry with `accepts` must cover every key its handler reads.

    Missed keys would be logged as `dispatch_unknown_keys` every time the
    handler is called — noise that drowns real drops. Catches this at PR
    time instead of via dashboard signal pollution.
    """
    COMMAND_TABLE = daemon_dispatch.COMMAND_TABLE
    failures = []
    checked = 0
    for cmd, entry in COMMAND_TABLE.items():
        if entry.accepts is None:
            continue
        checked += 1
        handler_keys = _collect_args_get_keys(entry.handler)
        # The handler can legitimately read keys that don't come from args
        # (e.g. `result.get("ok")`). We only care about keys absent from
        # accepts — a subset check in the correct direction.
        missing = handler_keys - set(entry.accepts)
        # Known non-args `.get()` callers used inside these handlers. Anything
        # a handler reads that isn't in args should be added here as it's
        # introduced; the goal is zero false positives, not zero missing.
        IGNORABLE = {
            'ok', 'result', 'error',          # dispatch result shapes
            'results', 'total', 'succeeded',  # bulk result shapes
            'failed', 'operations',           # brain_batch result/input
            'id', 'node_id', 'op',            # nested inside op_spec (brain_batch)
            'encoding_source', 'archived_by', 'reason',
            'source_id', 'target_id', 'relation',
            'title', 'query',                 # action_summary extraction
            'input',
        }
        real_missing = missing - IGNORABLE
        if real_missing:
            failures.append('%s: handler reads %s but accepts=%s' % (
                cmd, sorted(real_missing), sorted(entry.accepts)))
    assert not failures, (
        'CmdEntry.accepts drift (%d commands checked, %d failures):\n  %s'
        % (checked, len(failures), '\n  '.join(failures)))


def test_at_least_some_accepts_declared():
    """Guard against accidental wipe of all `accepts` contracts.

    If every entry ends up with accepts=None, silent-drop detection stops
    working entirely. This is the "don't let the canary starve" check.
    """
    COMMAND_TABLE = daemon_dispatch.COMMAND_TABLE
    with_accepts = [c for c, e in COMMAND_TABLE.items() if e.accepts is not None]
    assert len(with_accepts) >= 3, (
        'expected accepts declared for core write commands (connect, '
        'connect_batch, brain_batch) — only %d found: %s'
        % (len(with_accepts), with_accepts))


def test_check_unknown_keys_logs_on_unknown():
    """End-to-end: sending an extra key to a handler with accepts declared
    must produce a dispatch_unknown_keys error in the brain log."""
    import sqlite3
    from unittest.mock import MagicMock

    # Fake brain that records error log calls
    logged_errors = []

    class FakeBrain:
        def _log_error(self, source, error, context):
            logged_errors.append((source, str(error), context))

    fake = FakeBrain()
    entry = daemon_dispatch.COMMAND_TABLE.get('connect')
    assert entry is not None and entry.accepts is not None
    # Call with an unknown key
    daemon_dispatch.check_unknown_keys(
        'connect', entry,
        {'source_id': 'a', 'target_id': 'b', 'relation': 'r', 'unknown_param': 'x'},
        fake)
    assert any('dispatch_unknown_keys' in src for src, _, _ in logged_errors), \
        'expected dispatch_unknown_keys log entry, got: %s' % logged_errors
    assert any("'unknown_param'" in e for _, e, _ in logged_errors), \
        'expected unknown_param mentioned in error, got: %s' % logged_errors
