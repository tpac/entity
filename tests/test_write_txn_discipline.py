"""Contract: write-path transaction discipline is STRUCTURAL, not by-convention.

brain_batch (and the bg-writer drain) wrap many sub-ops in one BEGIN IMMEDIATE
/ COMMIT envelope. For that to be atomic, the DAL writers running inside must
NOT self-commit. Pre-2026-05-30 this was enforced by a fragile convention:
every writer took a `commit` kwarg and every batch-context caller had to
remember `commit=not _batch_mode`. We shipped exactly the bug that convention
invites — 3 callers forgot the kwarg; only a code review caught it.

The structural fix (docs/WRITE-TXN-ISOLATION-ROOTFIX.md, Option A): batch state
lives on the CONNECTION (BatchAwareConnection.in_batch); the envelope owner
flips it; writers consult it via commit_unless_batched(). There is no kwarg to
forget. These tests lock that so a future writer can't reintroduce the gap:

1. commit_unless_batched honors conn.in_batch (behavioral).
2. No DAL writer self-commits via a bare `self.conn.commit()` — all route
   through the helper (source contract — fails on a planted regression).
3. The GraphDAL batch-reachable writers expose no `commit` parameter (the old
   forgettable knob is gone — signature contract).
4. The Brain's three SQLite connections are BatchAwareConnection, so in_batch
   is always present in production (wiring contract).
"""

import os
import re
import sys
import glob
import inspect
import sqlite3
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.db_backends.sqlite import BatchAwareConnection, commit_unless_batched
from servers.dal import SourceRefDAL
from servers.dal_graph import GraphDAL

_SERVERS_DIR = os.path.join(os.path.dirname(__file__), '..', 'servers')


class TestCommitHelperBehavior(unittest.TestCase):
    """commit_unless_batched is the single gate — it must read conn.in_batch."""

    def _fresh_conn(self):
        conn = sqlite3.connect(':memory:', factory=BatchAwareConnection)
        conn.execute('CREATE TABLE t (x INTEGER)')
        conn.commit()
        return conn

    def test_commits_when_not_batched(self):
        conn = self._fresh_conn()
        conn.execute('INSERT INTO t (x) VALUES (1)')
        self.assertTrue(conn.in_transaction, "DML should open a deferred txn")
        commit_unless_batched(conn)  # in_batch defaults False
        self.assertFalse(conn.in_transaction,
                         "standalone write must commit (in_batch=False)")

    def test_defers_when_batched(self):
        conn = self._fresh_conn()
        conn.in_batch = True
        conn.execute('INSERT INTO t (x) VALUES (1)')
        commit_unless_batched(conn)
        self.assertTrue(conn.in_transaction,
                        "must NOT commit inside a batch (in_batch=True)")
        conn.rollback()  # the envelope owner would commit/rollback; prove it CAN

    def test_safe_on_plain_connection(self):
        """A non-BatchAware connection has no in_batch attr — getattr default
        False → treated as standalone → commits. (Maintenance/test paths.)"""
        plain = sqlite3.connect(':memory:')
        plain.execute('CREATE TABLE t (x INTEGER)')
        plain.execute('INSERT INTO t (x) VALUES (1)')
        commit_unless_batched(plain)  # must not raise
        self.assertFalse(plain.in_transaction,
                         "plain conn should commit (safe default)")


class TestNoSelfCommitInDAL(unittest.TestCase):
    """Source contract: no DAL writer may self-commit via `self.conn.commit()`.
    Every commit routes through commit_unless_batched(self.conn) so batch
    atomicity can't be broken by a writer that bypasses the gate."""

    def test_no_bare_self_conn_commit(self):
        offenders = []
        for path in glob.glob(os.path.join(_SERVERS_DIR, 'dal*.py')):
            with open(path, encoding='utf-8') as f:
                for lineno, line in enumerate(f, 1):
                    if 'self.conn.commit()' in line:
                        offenders.append('%s:%d' % (os.path.basename(path), lineno))
        self.assertEqual(
            offenders, [],
            "DAL writers must commit via commit_unless_batched(self.conn), not a "
            "bare self.conn.commit() — that bypasses the batch gate and breaks "
            "brain_batch atomicity. Offending lines: %s" % offenders)


class TestBrainSelfCommitsAreMarked(unittest.TestCase):
    """Source contract for the brain mixins (brain.py + brain_*.py): a
    `self.conn.commit()` statement is allowed ONLY at an explicit durability
    point tagged `# commit-ok: <reason>` (save/autosave, shutdown, a backfill
    that holds write_lock). Every OTHER self.conn write must route through
    self._maybe_commit() / commit_unless_batched so it respects conn.in_batch.

    Closes the blind spot the F3 code review found: the dal*.py-only scan above
    did not cover brain.py, where a Brain method once self-committed and
    bypassed the gate. Markers make the few legitimate exceptions explicit and
    auditable instead of invisible.

    Detection uses `tokenize`, not a line prefix, so it catches the shapes the
    old `startswith('self.conn.commit()')` missed — ALIASED commits
    (`instance.conn.commit()`) and COMPOUND statements (`x; self.conn.commit()`)
    — while string/comment prose mentioning the call is excluded for free
    (tokenize categorizes those separately). `logs_conn.commit()` is excluded by
    the leading-dot requirement (it's brain_logs.db, a separate connection)."""

    @staticmethod
    def _conn_commit_lines(path):
        """Line numbers of real `<expr>.conn.commit(` calls, skipping strings
        and comments. Matches the token run  `.` `conn` `.` `commit` `(`."""
        import tokenize
        SKIP = {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                tokenize.COMMENT, tokenize.ENCODING}
        with open(path, 'rb') as f:
            toks = [t for t in tokenize.tokenize(f.readline) if t.type not in SKIP]
        lines = []
        for i in range(len(toks) - 4):
            if [t.string for t in toks[i:i + 5]] == ['.', 'conn', '.', 'commit', '(']:
                lines.append(toks[i + 1].start[0])
        return lines

    def test_brain_self_conn_commits_are_marked(self):
        offenders = []
        paths = (glob.glob(os.path.join(_SERVERS_DIR, 'brain.py')) +
                 glob.glob(os.path.join(_SERVERS_DIR, 'brain_*.py')))
        for path in paths:
            src_lines = open(path, encoding='utf-8').read().splitlines()
            for ln in self._conn_commit_lines(path):
                if 'commit-ok:' not in src_lines[ln - 1]:
                    offenders.append('%s:%d' % (os.path.basename(path), ln))
        self.assertEqual(
            offenders, [],
            "A *.conn.commit() in a brain mixin must either route through "
            "self._maybe_commit() (gate on conn.in_batch) or be tagged "
            "`# commit-ok: <reason>` if it's a deliberate explicit-durability "
            "point. Untagged offenders: %s" % offenders)

    def test_detector_catches_aliased_and_compound_skips_prose(self):
        """Teeth: the tokenize detector must catch the shapes the old
        startswith() missed (aliased + compound) and ignore string/comment prose
        and logs_conn — otherwise the upgrade is theater."""
        import tempfile
        src = (
            "x = 1\n"                                  # 1
            "instance.conn.commit()\n"                 # 2  aliased — must catch
            "foo(); self.conn.commit()\n"              # 3  compound — must catch
            "self.logs_conn.commit()\n"                # 4  separate DB — must skip
            "# a comment about self.conn.commit() here\n"  # 5  comment — skip
            "s = 'self.conn.commit() in a string'\n"   # 6  string — skip
        )
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(src)
            path = f.name
        try:
            self.assertEqual(self._conn_commit_lines(path), [2, 3])
        finally:
            os.unlink(path)


class TestSetConfigHoldsWriteLock(unittest.TestCase):
    """set_config is a foreground self.conn write that S2 maintenance calls
    lock-free on a pool thread (gating timestamps, failure counters, journals).
    It MUST hold write_lock so it can't interleave with a concurrent client
    brain_batch on the shared connection — without it, set_config's INSERT
    joins the batch's open transaction and is lost if the batch rolls back.
    (Source contract — the behavioral race is timing-dependent to reproduce
    deterministically; this locks the guard that prevents it.)"""

    def test_set_config_acquires_write_lock(self):
        from servers.brain import Brain
        src = inspect.getsource(Brain.set_config)
        self.assertIn(
            'with self.write_lock', src,
            "Brain.set_config must acquire write_lock — S2 maintenance calls it "
            "lock-free on a pool thread, and without the lock its INSERT "
            "interleaves with a concurrent brain_batch on the shared connection "
            "and is lost on rollback.")


class TestSaveHoldsWriteLock(unittest.TestCase):
    """Same class of bug as set_config above. Brain.save() commits the
    foreground self.conn at an explicit durability point. The daemon's S2
    idle-maintenance path (_run_idle_maintenance) calls save() lock-free on a
    pool thread — the SAME pool that handles client commands. If a client
    brain_batch is mid-flight under write_lock (BEGIN IMMEDIATE + many writes
    on the shared self.conn), a lock-free save().commit() commits the batch's
    PARTIAL transaction, breaking its all-or-nothing atomicity. save() MUST
    hold write_lock around the commit so it serializes against brain_batch;
    write_lock is an RLock, so the primary autosave path (daemon_server, which
    already holds write_lock before calling save) re-acquires safely.
    (Source contract — the behavioral race is timing-dependent to reproduce
    deterministically; this locks the guard that prevents it.)"""

    def test_save_acquires_write_lock(self):
        from servers.brain import Brain
        src = inspect.getsource(Brain.save)
        self.assertIn(
            'with self.write_lock', src,
            "Brain.save must acquire write_lock around its self.conn.commit() — "
            "the daemon's S2 maintenance path calls save() lock-free on a pool "
            "thread, and without the lock its commit lands mid-flight on a "
            "concurrent brain_batch and commits a partial transaction.")


class TestGraphWritersHaveNoCommitKwarg(unittest.TestCase):
    """Signature contract: the batch-reachable GraphDAL writers expose no
    `commit` parameter. The old forgettable knob is gone — the writer reads
    conn.in_batch itself, so a caller can't pass the wrong value."""

    BATCH_REACHABLE_WRITERS = [
        (GraphDAL, 'add_relation'), (GraphDAL, 'remove_relation'),
        (GraphDAL, 'delete_node_edges'),
        # source_refs extracted to SourceRefDAL (Phase 5) — still batch-reachable
        (SourceRefDAL, 'add_source_refs'), (SourceRefDAL, 'replace_source_refs'),
    ]

    def test_writers_have_no_commit_param(self):
        for cls, name in self.BATCH_REACHABLE_WRITERS:
            method = getattr(cls, name)
            params = inspect.signature(method).parameters
            self.assertNotIn(
                'commit', params,
                "%s.%s still takes a `commit` kwarg — the structural fix "
                "removes it; the writer must gate on conn.in_batch instead."
                % (cls.__name__, name))


class TestBrainConnectionsAreBatchAware(BrainTestBase):
    """Wiring contract: the Brain's SQLite connections carry in_batch so the
    gate is always live in production (not just when a stray getattr defaults)."""
    needs_embedder = False

    def test_all_three_connections_batch_aware(self):
        for attr in ('conn', 'conn_bg_writer', 'logs_conn', 'logs_conn_w'):
            conn = getattr(self.brain, attr)
            self.assertIsInstance(
                conn, BatchAwareConnection,
                "brain.%s must be a BatchAwareConnection so conn.in_batch is "
                "always present" % attr)
            self.assertFalse(conn.in_batch,
                             "brain.%s.in_batch must default False" % attr)


if __name__ == '__main__':
    unittest.main()
