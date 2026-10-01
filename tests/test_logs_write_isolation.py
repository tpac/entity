"""Write-boundary guardrails for brain_logs.db (the SQLITE_BUSY_SNAPSHOT fix).

The bug: brain_logs.db writes ran on the same shared
connection that serves concurrent reads. An open read cursor pins a WAL
snapshot on the connection; once ANY other connection commits (a hook
process, the MCP monitor), a write on the snapshot-holding connection is a
read->write upgrade from a stale snapshot — SQLite fails it INSTANTLY with
'database is locked', never invoking the busy handler. busy_timeout offers
zero protection. Every such failure silently dropped a trace event.

The fix: each logs DAL routes writes to a dedicated write connection used
only under the write lock, so no read cursor can ever exist on it at write
time. These tests pin both halves: the mechanism (single shared connection
still fails — if SQLite/Python semantics ever change, we learn) and the fix
(split-connection DAL survives the exact production interleave).

Both halves turn on telling two 'database is locked' failures apart, so this
file times every failure: a snapshot upgrade never reaches the busy handler,
while a writer merely queueing for the WAL writer slot fails only once
busy_timeout has expired.

Run: ./dev pytest tests/test_logs_write_isolation.py -v
"""
import os
import sqlite3
import tempfile
import threading
import time
import unittest

from servers.dal_logs import LogsDAL, TraceDAL, SessionStateDAL
from servers.db_backends import sqlite as sqlite_backend
from servers.schema import ensure_logs_schema
from tests.brain_test_base import BrainTestBase


# A failure that did NOT burn at least this share of busy_timeout never waited
# on the busy handler, so it is the snapshot upgrade. Attributing on the
# fraction rather than a small fixed bound keeps the alarming class the
# default: contention has to prove itself by actually spending the timeout.
CONTENTION_MIN_FRACTION = 0.9


def _connect(path):
    """A connection wired like the daemon's own — production's pragma set.

    WHICH pragmas is load-bearing, and it is not only busy_timeout. This file
    used to hand-roll WAL + busy_timeout=200 and so inherited SQLite's default
    synchronous=FULL: every commit from the tight-loop external committer in
    the stress test below fsynced the WAL and held the writer slot for
    milliseconds, starving writers past the 200ms cap. The expiry was then
    recorded as a dropped write — 4-8 reds per 40 runs on an idle machine
    (measured 2026-09-20; every failure 219-302ms, none near 0ms).

    Production is neither of those: synchronous=NORMAL, busy_timeout=30s.
    Either one alone clears the flake (0/40 with only the cap raised, 0/40
    with only NORMAL), so the old settings were the single corner where it
    appeared. Taking the daemon's own pragma set fixes the cause and stops
    this file drifting from the topology it exists to model — and it leaves
    the snapshot path, which ignores busy_timeout entirely, as the only way
    a write here can still be dropped.
    """
    conn = sqlite3.connect(path, check_same_thread=False)
    sqlite_backend.apply_pragmas(conn)
    return conn


def _describe(drops):
    return ', '.join('%.0fms %r' % (ms, exc) for ms, exc in drops)


class LogsWriteIsolationBase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, 'brain_logs.db')
        self.read_conn = _connect(self.db_path)
        ensure_logs_schema(self.read_conn, db_path=self.db_path)
        self.write_conn = _connect(self.db_path)
        self.external = _connect(self.db_path)  # models a hook process
        self.lock = threading.RLock()
        self.busy_ms = self.write_conn.execute(
            'PRAGMA busy_timeout').fetchone()[0]

    def tearDown(self):
        for conn in (self.read_conn, self.write_conn, self.external):
            try:
                conn.close()
            except Exception:
                pass
        self.tmpdir.cleanup()

    def _seed_traces(self, dal, n=300):
        dal.append_batch([{
            'chain_id': 'seed', 'scale': 's0', 'event_type': 'O',
            'ref_type': 'user_message', 'ref_id': '', 'summary': 'row %d' % i,
            'metadata': None, 'session_id': 'seed'} for i in range(n)])

    def _poison(self):
        """Reproduce the production interleave: pin a read snapshot with a
        live, partially-fetched cursor, then advance the WAL from an external
        connection. Returns the open cursor (kept alive by the caller)."""
        cur = self.read_conn.execute(
            'SELECT id, summary FROM trace_events')
        cur.fetchone()  # partially fetched -> statement stays open
        self.external.execute(
            "INSERT INTO debug_log (event_type, source, created_at) "
            "VALUES ('e', 'external', '2026-01-01T00:00:00+00:00')")
        self.external.commit()
        return cur


class TestMechanismStillExists(LogsWriteIsolationBase):
    """Pin the failure mode itself on a single shared connection. If this
    ever fails, SQLite/Python cursor semantics changed and the split (plus
    this file) should be re-evaluated."""

    def test_single_connection_write_fails_instantly(self):
        dal = TraceDAL(self.read_conn)  # pre-fix wiring: one shared conn
        self._seed_traces(dal)
        cur = self._poison()
        t0 = time.perf_counter()
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            dal.append(chain_id='c1', scale='s0', event_type='O',
                       ref_type='user_message', session_id='s')
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertIn('locked', str(ctx.exception))
        # The bypass IS the mechanism: nothing else can fail a write in less
        # than busy_timeout. If this ever waits the timeout out instead, the
        # failure became ordinary writer-slot contention and the test below
        # is no longer pinning what it claims to pin.
        self.assertLess(
            elapsed_ms, self.busy_ms * CONTENTION_MIN_FRACTION,
            'expected an instant snapshot-upgrade failure, but the write '
            'took %.0fms of a %dms busy_timeout — it waited on the busy '
            'handler, so this is not the SQLITE_BUSY_SNAPSHOT path'
            % (elapsed_ms, self.busy_ms))
        cur.close()


class TestSplitConnectionSurvives(LogsWriteIsolationBase):
    """The fix: the exact same interleave, with the DAL wired the way
    Brain.__init__ wires it (read conn + write conn + shared lock)."""

    def _dal(self, cls):
        return cls(self.read_conn, write_conn=self.write_conn,
                   write_lock=self.lock)

    def test_trace_append_survives_poisoned_read_snapshot(self):
        dal = self._dal(TraceDAL)
        self._seed_traces(dal)
        cur = self._poison()
        event_id = dal.append(chain_id='c1', scale='s0', event_type='O',
                              ref_type='user_message', session_id='s')
        self.assertTrue(event_id)
        row = self.external.execute(
            'SELECT chain_id FROM trace_events WHERE id = ?',
            (event_id,)).fetchone()
        self.assertEqual(row[0], 'c1')
        cur.close()

    def test_store_embeddings_survives_poisoned_read_snapshot(self):
        dal = self._dal(TraceDAL)
        self._seed_traces(dal, n=50)
        tid = dal.append(chain_id='c2', scale='s0', event_type='O',
                         ref_type='user_message', session_id='s')
        cur = self._poison()
        n = dal.store_embeddings([(tid, b'\x00\x01', 'text')], model='m')
        self.assertEqual(n, 1)
        cur.close()

    def test_debug_log_and_session_state_survive(self):
        logs = self._dal(LogsDAL)
        state = self._dal(SessionStateDAL)
        trace = self._dal(TraceDAL)
        self._seed_traces(trace, n=50)
        cur = self._poison()
        logs.write_event('error', 'test', {'k': 'v'}, session_id='s')
        state.set('sess', 'key', 'value')
        cur.close()
        self.assertEqual(self.external.execute(
            "SELECT COUNT(*) FROM debug_log WHERE source='test'"
        ).fetchone()[0], 1)
        self.assertEqual(self.external.execute(
            "SELECT value FROM session_state WHERE session_id='sess'"
        ).fetchone()[0], 'value')

    def test_concurrent_readers_writers_external_commits(self):
        """Stress the production topology: reader threads iterating live
        cursors on the read conn, writer threads appending through the DAL,
        an external committer advancing the WAL. Zero drops expected."""
        dal = self._dal(TraceDAL)
        self._seed_traces(dal)
        errors = []
        # A side thread that dies takes its half of the interleave with it,
        # and a dead thread's traceback goes to stderr, which pytest hides on
        # a passing test — so the run would go green having exercised nothing.
        # Record instead, and fail on it below.
        side_errors = []
        stop = threading.Event()

        def reader():
            try:
                while not stop.is_set():
                    cur = self.read_conn.execute(
                        'SELECT id, summary FROM trace_events')
                    cur.fetchone()          # hold a snapshot briefly
                    cur.fetchall()
            except Exception as e:
                if not stop.is_set():       # not just teardown racing us
                    side_errors.append(('reader', e))

        def external_committer():
            conn = _connect(self.db_path)
            try:
                while not stop.is_set():
                    conn.execute(
                        "INSERT INTO debug_log (event_type, source, "
                        "created_at) "
                        "VALUES ('e', 'ext', '2026-01-01T00:00:00+00:00')")
                    conn.commit()
            except Exception as e:
                if not stop.is_set():
                    side_errors.append(('committer', e))
            conn.close()

        def writer(k):
            for i in range(25):
                t0 = time.perf_counter()
                try:
                    dal.append(chain_id='w%d' % k, scale='s0',
                               event_type='O', ref_type='user_message',
                               session_id='s%d' % k)
                except Exception as e:
                    errors.append(((time.perf_counter() - t0) * 1000.0, e))
                    return

        side = [threading.Thread(target=reader, daemon=True),
                threading.Thread(target=external_committer, daemon=True)]
        writers = [threading.Thread(target=writer, args=(k,))
                   for k in range(4)]
        for t in side + writers:
            t.start()
        for t in writers:
            t.join(timeout=60)
        stop.set()
        for t in side:
            t.join(timeout=10)

        self.assertEqual(
            side_errors, [],
            'a side thread died mid-run, so the interleave stopped before the '
            'writers did — the drop counts below mean nothing: %r'
            % (side_errors,))
        # Split the drops by what they waited on — same message, two causes.
        floor_ms = self.busy_ms * CONTENTION_MIN_FRACTION
        snapshot = [d for d in errors if d[0] < floor_ms]
        contention = [d for d in errors if d[0] >= floor_ms]
        self.assertEqual(
            snapshot, [],
            'SNAPSHOT-UPGRADE DROPS (%d): a write failed without waiting out '
            'busy_timeout=%dms. That is the 371895a8 signature — the write '
            'connection held a read snapshot. The connection split regressed; '
            'this is a production bug, not a flaky test. %s'
            % (len(snapshot), self.busy_ms, _describe(snapshot)))
        self.assertEqual(
            contention, [],
            'WRITER-SLOT STARVATION (%d): a write waited out the full '
            'busy_timeout=%dms for the WAL writer slot. Not the snapshot bug '
            '(that one never waits) — this machine starved a writer past a '
            'timeout production also runs with. %s'
            % (len(contention), self.busy_ms, _describe(contention)))
        n = self.external.execute(
            "SELECT COUNT(*) FROM trace_events WHERE chain_id LIKE 'w%'"
        ).fetchone()[0]
        self.assertEqual(n, 100)
        # Guard against a vacuous green: if the committer thread had died on
        # its first statement, the WAL would never advance and the interleave
        # under test would not have happened at all.
        self.assertGreater(
            self.external.execute(
                "SELECT COUNT(*) FROM debug_log WHERE source='ext'"
            ).fetchone()[0], 0,
            'external committer never committed — the WAL never advanced, so '
            'this run did not exercise the interleave')


class TestBrainWiresTheSplit(BrainTestBase):
    """Pin the Brain.__init__ wiring itself. The DAL tests above construct
    split-connection DALs by hand — a revert of the brain.py kwargs would
    leave them green while production silently fell back to single-connection
    semantics (the whole bug, undetected). This closes that gap."""
    needs_embedder = False

    def test_every_logs_dal_gets_write_conn_and_logs_lock(self):
        b = self.brain
        for name in ('_logs_dal', '_trace_dal', '_interaction_dal',
                     '_session_state'):
            dal = getattr(b, name)
            self.assertIs(dal.wconn, b.logs_conn_w,
                          '%s.wconn is not brain.logs_conn_w' % name)
            self.assertIsNot(dal.wconn, b.logs_conn,
                             '%s writes on the shared read connection' % name)
            self.assertIs(dal._wlock, b.logs_write_lock,
                          '%s does not serialize on logs_write_lock' % name)


if __name__ == '__main__':
    unittest.main()
