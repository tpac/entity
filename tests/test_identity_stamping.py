"""Identity stamping at trace write (v27 substrate).

When set_identity() is called on TraceDAL, every subsequent append()
and append_batch() injects human_identity/agent_identity into the
event's metadata via setdefault (caller can override per-event).

Pure-DAL tests — no live Brain.
"""

import json
import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.dal_logs import TraceDAL
from servers.schema import ensure_logs_schema


def _open_logs():
    conn = sqlite3.connect(':memory:')
    ensure_logs_schema(conn)
    return conn


def _read_metadata(conn, event_id):
    row = conn.execute(
        'SELECT metadata FROM trace_events WHERE id = ?',
        (event_id,)).fetchone()
    return json.loads(row[0]) if row and row[0] else None


class StampingDisabledTest(unittest.TestCase):
    """Without set_identity(), TraceDAL behaves as before — null metadata
    stays null, dict metadata stays untouched."""

    def setUp(self):
        self.conn = _open_logs()
        self.dal = TraceDAL(self.conn)

    def tearDown(self):
        self.conn.close()

    def test_no_stamp_when_unconfigured_null_metadata(self):
        eid = self.dal.append(
            chain_id='c1', scale='s0', event_type='K',
            ref_type='user_message', metadata=None)
        self.assertIsNone(_read_metadata(self.conn, eid))

    def test_no_stamp_when_unconfigured_dict_metadata(self):
        eid = self.dal.append(
            chain_id='c1', scale='s0', event_type='K',
            ref_type='user_message', metadata={'content': 'hi'})
        meta = _read_metadata(self.conn, eid)
        self.assertEqual(meta, {'content': 'hi'})

    def test_empty_string_identity_disabled(self):
        self.dal.set_identity('', '')
        eid = self.dal.append(
            chain_id='c1', scale='s0', event_type='K',
            ref_type='user_message', metadata={'content': 'hi'})
        meta = _read_metadata(self.conn, eid)
        self.assertEqual(meta, {'content': 'hi'})


class StampingEnabledTest(unittest.TestCase):
    """With set_identity(), every trace event metadata carries
    human_identity + agent_identity. Caller overrides win."""

    def setUp(self):
        self.conn = _open_logs()
        self.dal = TraceDAL(self.conn)
        self.dal.set_identity('Ada', 'Anchor')

    def tearDown(self):
        self.conn.close()

    def test_stamp_when_metadata_present(self):
        eid = self.dal.append(
            chain_id='c1', scale='s0', event_type='K',
            ref_type='user_message', metadata={'content': 'hi'})
        meta = _read_metadata(self.conn, eid)
        self.assertEqual(meta['content'], 'hi')
        self.assertEqual(meta['human_identity'], 'Ada')
        self.assertEqual(meta['agent_identity'], 'Anchor')

    def test_stamp_when_metadata_none(self):
        eid = self.dal.append(
            chain_id='c1', scale='s0', event_type='K',
            ref_type='user_message', metadata=None)
        meta = _read_metadata(self.conn, eid)
        self.assertEqual(meta, {'human_identity': 'Ada',
                                'agent_identity': 'Anchor'})

    def test_caller_override_preserved(self):
        # An explicit per-event identity (e.g., a different speaker in a
        # multi-party scenario) wins over the daemon-default stamp.
        eid = self.dal.append(
            chain_id='c1', scale='s0', event_type='K',
            ref_type='user_message',
            metadata={'content': 'hi', 'human_identity': 'Alice'})
        meta = _read_metadata(self.conn, eid)
        self.assertEqual(meta['human_identity'], 'Alice')
        self.assertEqual(meta['agent_identity'], 'Anchor')

    def test_partial_identity_human_only(self):
        self.dal.set_identity('Ada', '')
        eid = self.dal.append(
            chain_id='c1', scale='s0', event_type='K',
            ref_type='user_message', metadata={})
        meta = _read_metadata(self.conn, eid)
        self.assertEqual(meta, {'human_identity': 'Ada'})

    def test_set_identity_strips_whitespace(self):
        self.dal.set_identity('  Ada  ', '  Anchor  ')
        self.assertEqual(self.dal._human_identity, 'Ada')
        self.assertEqual(self.dal._agent_identity, 'Anchor')

    def test_stamp_in_append_batch(self):
        events = [
            {'chain_id': 'c1', 'scale': 's0', 'event_type': 'K',
             'ref_type': 'user_message', 'metadata': {'content': 'A'}},
            {'chain_id': 'c1', 'scale': 's0', 'event_type': 'delta',
             'ref_type': 'assistant_message', 'metadata': {'content': 'B'}},
        ]
        ids = self.dal.append_batch(events)
        for eid in ids:
            meta = _read_metadata(self.conn, eid)
            self.assertEqual(meta['human_identity'], 'Ada')
            self.assertEqual(meta['agent_identity'], 'Anchor')

    def test_stamp_skips_non_dict_metadata(self):
        """Defensive: if a caller passes a non-dict (e.g. a JSON string
        that wasn't decoded by the dispatch layer), don't crash — return
        the value unchanged. Dispatch is responsible for normalizing
        shapes before reaching the DAL."""
        # Direct helper call — append() requires a valid contract triple
        # and would mask the defensive behavior behind validation.
        result = self.dal._stamp_identity('{"tool": "Bash"}')
        self.assertEqual(result, '{"tool": "Bash"}')
        result = self.dal._stamp_identity(42)
        self.assertEqual(result, 42)

    def test_stamp_on_all_scales(self):
        """Identity stamps every trace, not just s0 — DAL is policy-neutral.
        Higher-scale traces (S1, S2) still benefit from knowing which
        operator was present when they ran."""
        cases = [
            ('s0', 'K', 'user_message'),
            ('s1', 'delta', 'additionalContext'),
            ('s2', 'delta', 'consolidated'),
        ]
        for scale, event_type, ref_type in cases:
            eid = self.dal.append(
                chain_id='c1', scale=scale, event_type=event_type,
                ref_type=ref_type, metadata=None)
            meta = _read_metadata(self.conn, eid)
            self.assertEqual(meta['human_identity'], 'Ada', scale)
            self.assertEqual(meta['agent_identity'], 'Anchor', scale)


class WritePathWarningTest(unittest.TestCase):
    """The 'identity unconfigured on scale write' loud signal fires at the
    write site (not at boot) — one-shot per TraceDAL lifetime so the
    operator sees the gap without log spam.

    It gates on identity being UNCHOSEN, not merely absent: the agent name
    now falls back to a shipped default, so a non-empty name no longer proves
    anyone picked it, and keying on emptiness would silence this forever."""

    def setUp(self):
        self.conn = _open_logs()
        self.dal = TraceDAL(self.conn)

    def tearDown(self):
        self.conn.close()

    def _capture_stderr(self, fn):
        import io
        import sys
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            fn()
        finally:
            sys.stderr = old
        return buf.getvalue()

    def test_fires_once_on_first_write_when_unset(self):
        def do_write():
            self.dal.append(chain_id='c', scale='s0', event_type='K',
                            ref_type='user_message', metadata=None)
        out = self._capture_stderr(do_write)
        self.assertIn('identity unconfigured on first scale-s0 write', out)
        self.assertIn('BRAIN_OPERATOR_NAME', out)
        self.assertIn('BRAIN_AGENT_NAME', out)

    def test_does_not_fire_again_on_subsequent_writes(self):
        def do_writes():
            for _ in range(3):
                self.dal.append(chain_id='c', scale='s0', event_type='K',
                                ref_type='user_message', metadata=None)
        out = self._capture_stderr(do_writes)
        # Only one occurrence regardless of how many writes happened
        self.assertEqual(out.count('identity unconfigured'), 1)

    def test_silent_when_identity_configured(self):
        self.dal.set_identity('Ada', 'Anchor')

        def do_write():
            self.dal.append(chain_id='c', scale='s0', event_type='K',
                            ref_type='user_message', metadata=None)
        out = self._capture_stderr(do_write)
        self.assertEqual(out, '')

    def test_fires_on_append_batch_too(self):
        def do_batch():
            self.dal.append_batch([
                {'chain_id': 'c', 'scale': 's0', 'event_type': 'K',
                 'ref_type': 'user_message', 'metadata': None},
            ])
        out = self._capture_stderr(do_batch)
        self.assertIn('identity unconfigured', out)

    def test_default_agent_name_still_fires(self):
        """A shipped default is a name, not a choice.

        This is the case a non-empty default would otherwise silence forever:
        both identity slots are populated, so an emptiness check would go
        quiet, but nobody has actually named this entity. The message must
        also stop claiming nothing is recorded — the default IS stamped.
        """
        from servers.daemon_config import DEFAULT_AGENT_NAME
        self.dal.set_identity('Ada', DEFAULT_AGENT_NAME, agent_is_default=True)

        def do_write():
            self.dal.append(chain_id='c', scale='s0', event_type='K',
                            ref_type='user_message', metadata=None)
        out = self._capture_stderr(do_write)
        self.assertIn('identity unconfigured', out)
        self.assertIn('BRAIN_AGENT_NAME', out)
        self.assertNotIn('BRAIN_OPERATOR_NAME', out)
        self.assertIn('stamped with the default agent name', out)

    def test_partial_identity_still_fires(self):
        # Only one side set — still missing the other
        self.dal.set_identity('Ada', '')

        def do_write():
            self.dal.append(chain_id='c', scale='s0', event_type='K',
                            ref_type='user_message', metadata=None)
        out = self._capture_stderr(do_write)
        self.assertIn('identity unconfigured', out)
        self.assertNotIn('BRAIN_OPERATOR_NAME', out)
        self.assertIn('BRAIN_AGENT_NAME', out)


class ConfigReaderTest(unittest.TestCase):
    """daemon_config helpers read env vars and strip whitespace."""

    def test_reads_env_vars(self):
        # Save/restore env to avoid test pollution
        prior_op = os.environ.pop('BRAIN_OPERATOR_NAME', None)
        prior_ag = os.environ.pop('BRAIN_AGENT_NAME', None)
        try:
            os.environ['BRAIN_OPERATOR_NAME'] = '  EnvTom  '
            os.environ['BRAIN_AGENT_NAME'] = 'EnvAnchor'
            from servers.daemon_config import get_operator_name, get_agent_name
            self.assertEqual(get_operator_name(), 'EnvTom')
            self.assertEqual(get_agent_name(), 'EnvAnchor')
        finally:
            if prior_op is not None:
                os.environ['BRAIN_OPERATOR_NAME'] = prior_op
            else:
                os.environ.pop('BRAIN_OPERATOR_NAME', None)
            if prior_ag is not None:
                os.environ['BRAIN_AGENT_NAME'] = prior_ag
            else:
                os.environ.pop('BRAIN_AGENT_NAME', None)

    def test_unset_agent_falls_back_to_default_operator_stays_empty(self):
        """The two names have deliberately different unset semantics.

        An entity with no name at all forces every render into awkward
        name-free prose, so the agent name falls back to a shipped default.
        A human's name cannot be invented, so the operator's stays empty and
        the DAL skips stamping it.
        """
        prior_op = os.environ.pop('BRAIN_OPERATOR_NAME', None)
        prior_ag = os.environ.pop('BRAIN_AGENT_NAME', None)
        try:
            from servers.daemon_config import (
                get_operator_name, get_agent_name, agent_name_is_default,
                DEFAULT_AGENT_NAME)
            self.assertEqual(get_operator_name(), '')
            self.assertEqual(get_agent_name(), DEFAULT_AGENT_NAME)
            self.assertTrue(agent_name_is_default())
            # Whitespace is not a choice either.
            os.environ['BRAIN_AGENT_NAME'] = '   '
            self.assertEqual(get_agent_name(), DEFAULT_AGENT_NAME)
            self.assertTrue(agent_name_is_default())
            # A real name is a choice — and the pairing is the whole
            # mechanism: what the name IS vs whether anyone picked it.
            os.environ['BRAIN_AGENT_NAME'] = 'Anchor'
            self.assertEqual(get_agent_name(), 'Anchor')
            self.assertFalse(agent_name_is_default())
        finally:
            if prior_op is not None:
                os.environ['BRAIN_OPERATOR_NAME'] = prior_op
            else:
                os.environ.pop('BRAIN_OPERATOR_NAME', None)
            if prior_ag is not None:
                os.environ['BRAIN_AGENT_NAME'] = prior_ag
            else:
                os.environ.pop('BRAIN_AGENT_NAME', None)


if __name__ == '__main__':
    unittest.main()
