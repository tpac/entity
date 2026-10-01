"""Fts5DAL.search excludes archived nodes at READ time.

FTS5 (nodes_fts) is a separate virtual table with no `archived` column — the
one recall candidate lane that historically didn't filter liveness. A lingering
nodes_fts row for an archived node (cleanup failed/raced, or pre-fix data)
would surface a dead node in recall. The JOIN-on-nodes + `archived = 0` filter
makes the flag the single source of truth at read time, so FTS cleanup on
archive is hygiene, not correctness.

These tests simulate the LINGERING-ENTRY leak directly: flip archived=1 WITHOUT
scrubbing the FTS row (archive_node would delete it, masking the read filter),
then assert search no longer returns it.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestFtsArchivedFilter(BrainTestBase):
    needs_embedder = False

    def _node(self, title, content='body text'):
        return self.brain.remember(type='fact', title=title, content=content,
                                   encoding_source='anchor')['id']

    def _archive_leaving_fts(self, node_id):
        # Flip archived without scrubbing nodes_fts — the lingering-entry leak.
        self.brain.conn.execute(
            'UPDATE nodes SET archived = 1 WHERE id = ?', (node_id,))
        self.brain.conn.commit()

    def test_search_excludes_archived_by_default(self):
        live = self._node('zephyr quantum widget alpha')
        dead = self._node('zephyr quantum widget beta')
        self._archive_leaving_fts(dead)
        hits = self.brain._fts.search('zephyr quantum widget')
        self.assertIn(live, hits)
        self.assertNotIn(dead, hits)  # archived dropped at read despite live FTS row

    def test_include_archived_returns_archived(self):
        dead = self._node('xylophone marmalade vortex')
        self._archive_leaving_fts(dead)
        self.assertNotIn(dead, self.brain._fts.search('xylophone marmalade'))
        self.assertIn(
            dead,
            self.brain._fts.search('xylophone marmalade', include_archived=True))

    def test_live_search_unaffected(self):
        a = self._node('borealis cascade lumen one')
        b = self._node('borealis cascade lumen two')
        hits = self.brain._fts.search('borealis cascade lumen')
        self.assertIn(a, hits)
        self.assertIn(b, hits)

    def test_limit_yields_live_hits_not_dead_slots(self):
        """With archived rows present, LIMIT should return up to N LIVE hits —
        dead rows no longer consume slots (the recall-quality upside)."""
        live_ids = [self._node('kraken nebula token %d' % i) for i in range(3)]
        dead_ids = [self._node('kraken nebula token dead %d' % i) for i in range(3)]
        for d in dead_ids:
            self._archive_leaving_fts(d)
        hits = self.brain._fts.search('kraken nebula token', limit=3)
        self.assertEqual(len(hits), 3)
        self.assertTrue(set(hits).issubset(set(live_ids)))
        self.assertFalse(set(hits) & set(dead_ids))


if __name__ == '__main__':
    unittest.main()
