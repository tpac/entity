"""
brain — Shared Test Base with Brain-Aware Logging

Every test result is logged to a separate test logs DB (debug_log table) in
the system temp dir, accumulating across runs as raw material for trend
analysis until the OS clears it.

Usage:
    from tests.brain_test_base import BrainTestBase

All test classes should inherit from BrainTestBase instead of defining
their own.
"""

import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest

# Ensure parent is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.isolated_brain import brain_env


# ── Shared logs DB for cross-test-run persistence ────────────────────
# Test results accumulate in this DB across runs. Nothing reads them today;
# they are kept as raw material, queried ad hoc.

def _get_test_logs_db_path():
    """Resolve the test logs DB path.

    ALWAYS uses a separate test logs DB — never writes to production
    brain_logs.db. Test results polluting production logs makes debug_log
    useless for real hook telemetry (lesson from 4,885 test rows vs 79 real
    rows in debug_log).

    It lives in the temp dir because conftest fails the run on any database
    opened outside one (see tests/test_isolated_brain.py).
    """
    return os.path.join(tempfile.gettempdir(), 'brain_test_logs.db')


def _get_test_logs_conn():
    """Get a connection to the shared test logs DB."""
    db_path = _get_test_logs_db_path()
    # timeout= is sqlite's busy-timeout: two concurrent `pytest tests/` runs
    # share this one file, so a writer waits (up to 10s) for the other's commit
    # instead of erroring immediately with 'database is locked'.
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    # Ensure the debug_log table exists
    conn.execute("""CREATE TABLE IF NOT EXISTS debug_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        event_type TEXT NOT NULL,
        source TEXT,
        metadata TEXT,
        created_at TEXT
    )""")
    conn.commit()
    return conn


# Module-level connection — shared across all tests in a run
_logs_conn = None
_run_id = None
_run_start = None


def _ensure_logs():
    global _logs_conn, _run_id, _run_start
    if _logs_conn is None:
        _logs_conn = _get_test_logs_conn()
        # PID in the run id: two streams launching `pytest` in the same second
        # would otherwise share a run_id and interleave their rows under it.
        _run_id = 'test_run_%d_%d' % (int(time.time()), os.getpid())
        _run_start = time.time()
    return _logs_conn


def log_test_result(test_class, test_method, status, duration_ms,
                    error_msg=None, error_type=None):
    """Log a single test result to the shared debug_log."""
    metadata = {
        'test_class': test_class,
        'test_method': test_method,
        'status': status,
        'duration_ms': round(duration_ms, 1),
    }
    if error_msg:
        metadata['error'] = error_msg[:500]
    if error_type:
        metadata['error_type'] = error_type

    # Best-effort telemetry: writing the result row must NEVER fail the test
    # under teardown. The shared test logs DB can be contended by a second
    # concurrent `pytest tests/` run; if the 10s busy-timeout is still exceeded
    # (or any other DB error), skip the row rather than erroring the test.
    try:
        conn = _ensure_logs()
        conn.execute(
            "INSERT INTO debug_log (session_id, event_type, source, metadata, created_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            (_run_id, 'test_result', f'test:{test_class}.{test_method}',
             json.dumps(metadata), ))
        conn.commit()
    except Exception as _e:
        print('[brain_test_base] test-log write skipped (%s): %s'
              % (type(_e).__name__, _e), file=sys.stderr)


# ── Conversation turns from message rows ───────────────────────────

def turns_of(messages):
    """Role-tagged message rows → turns, as brain.conversation_turns hands
    them to the Surface and the encoder (trace_contract.group_turns). A row
    without a ref_type reads as the operator's or the assistant's by its
    role; a row without a chain joins nothing."""
    from servers.trace_contract import group_turns
    return group_turns([dict(m, ref_type=m.get('ref_type') or (
        'assistant_message' if m.get('role') == 'assistant' else 'user_message'))
        for m in messages or []])


# ── BrainTestBase — for unit tests ──────────────────────────────────

class BrainTestBase(unittest.TestCase):
    """Base class that creates a fresh brain per test and logs results.

    Every test result (pass/fail/error + duration) goes to the separate
    test logs DB that _get_test_logs_db_path owns — never to the production
    brain_logs.db.

    The embedder (nomic-embed-text-v1.5-Q, the default in servers/embedder.py)
    is loaded once per process via that module's singleton, so the first test
    to need it pays the load and every later one reuses it at zero cost.
    """

    # Set needs_embedder = False in subclasses that don't need semantic search.
    # This skips model loading entirely, saving ~1GB memory and ~1.5s startup.
    #
    # It also decides whether the brain is SEEDED, because servers/brain.py
    # welds the two: `if not skip_embedder: seed_baby_brain(self)`. So False
    # means an empty graph and True means the 26-node nursery pack — which is
    # what each class has always got, just not by anyone's intent. A False
    # brain writes no vectors and its recall() refuses rather than returning a
    # misleading empty; open the class True if it asserts on recall hits.
    needs_embedder = True

    def setUp(self):
        self._test_start = time.time()
        self.brain = None  # so tearDown's `is not None` guard works if it raises
        self._env = None

        # ONE provisioning door. Everything this method used to hand-roll —
        # the temp dir, the ASPECTS_JSON_PATH pin that keeps a test from
        # healing the operator's live taxonomy, the BRAIN_TMP_DIR pin that
        # stops two concurrent pytest processes colliding on fixed filenames,
        # the dead-embedder refusal, the drain wrappers that make embedding
        # look synchronous — belongs to brain_env and its context manager, and
        # was a second copy here that had already drifted from the first.
        #
        # `seeded` tracks `needs_embedder` because servers/brain.py welds
        # them: seeding runs under `if not skip_embedder`. So this preserves
        # exactly what each class gets today — the seed pack for a class that
        # loads the model, an empty graph for one that does not. Unwelding
        # them is a behaviour change with its own step; it is not this one.
        #
        # load_env=False matches what this base has always done: it never read
        # .env, and picking up API keys here would change whether an
        # LLM-calling test finds a credential.
        env = brain_env(needs_embedder=self.needs_embedder,
                        seeded=self.needs_embedder, load_env=False)
        env.__enter__()
        self._env = env
        self.brain = env.brain
        # Kept as attributes because ~170 assertions across 14 files reach for
        # them; they name the same paths the door already owns.
        self.tmp = env.db_dir
        self.db_path = env.brain_db

    def tearDown(self):
        # The telemetry runs inside try/finally so that cleanup happens even
        # if it raises. The version this replaced restored the two environment
        # pins FIRST for the same reason — "so neither a log_test_result nor a
        # brain.close() failure can leak the per-test path into every
        # subsequent test in the process". log_test_result guards its own DB
        # write, but the outcome determination below does not, and a leaked
        # $ASPECTS_JSON_PATH pointing at a deleted temp dir gives every later
        # test in the process an empty registry. `finally` keeps that
        # guarantee without depending on statement order.
        try:
            duration_ms = (time.time() - self._test_start) * 1000

            # Determine test outcome
            result = self._outcome.result if hasattr(self, '_outcome') else None
            status = 'pass'
            error_msg = None
            error_type = None

            if result and hasattr(result, 'failures'):
                # unittest TestResult API
                test_id = str(self)
                for failed_test, traceback in result.failures:
                    if str(failed_test) == test_id:
                        status = 'fail'
                        error_msg = traceback
                        error_type = 'AssertionError'
                        break
                for errored_test, traceback in result.errors:
                    if str(errored_test) == test_id:
                        status = 'error'
                        error_msg = traceback
                        # Extract error type from traceback
                        if traceback:
                            lines = traceback.strip().split('\n')
                            if lines:
                                error_type = lines[-1].split(':')[0]
                        break

            log_test_result(
                test_class=self.__class__.__name__,
                test_method=self._testMethodName,
                status=status,
                duration_ms=duration_ms,
                error_msg=error_msg,
                error_type=error_type,
            )
        finally:
            # Teardown belongs to the door: __exit__ closes the brain, restores
            # every environment variable __enter__ pinned, and removes the temp
            # dir — in that order, so a restored override can never point a
            # later test at a directory that has just been deleted.
            #
            # Guarded because setUp can fail before the context exists (a dead
            # embedder refuses at __enter__, having already cleaned up after
            # itself); unittest still calls tearDown, and exiting a context
            # that was never entered would raise over the real failure.
            if self._env is not None:
                self._env.__exit__(None, None, None)
                self._env = None
