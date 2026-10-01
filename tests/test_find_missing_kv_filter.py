"""find_missing's source_kv_keys filter — without this, the field-cohort
backfill stalls before reaching all nodes that have the source field
populated (top-N by last_accessed gets full of nodes that LACK the field
and never drains since backfill correctly skips them).
"""

import unittest
from tests.brain_test_base import BrainTestBase
from servers.dal import VectorDAL


class TestFindMissingKvFilter(BrainTestBase):
    needs_embedder = False

    def _make_node(self, title, **fields):
        """Create a node with metadata_kv fields. Strip the auto-vector
        creation by NOT seeding any node_enrichments rows after creation."""
        result = self.brain.remember(type='fact', title=title, content='c', **fields)
        # Wipe any vectors auto-created so find_missing returns this node
        self.brain.conn.execute(
            'DELETE FROM node_enrichments WHERE node_id = ?',
            (result['id'],))
        self.brain.conn.commit()
        return result['id']

    def test_filter_excludes_nodes_lacking_source_field(self):
        """find_missing with source_kv_keys=['their_raw_quote'] returns only
        nodes that have their_raw_quote populated — even if many other nodes
        also lack the their_raw_quote vector."""
        # 3 nodes WITH their_raw_quote
        with_quote_ids = [
            self._make_node(f'WithQuote {i}', their_raw_quote=f'quote {i}')
            for i in range(3)
        ]
        # 5 nodes WITHOUT their_raw_quote
        without_ids = [
            self._make_node(f'NoQuote {i}')
            for i in range(5)
        ]

        vdal = VectorDAL(self.brain.conn)

        # Without the filter: returns nodes regardless of whether they have
        # their_raw_quote. Up to 8 active nodes from this test.
        unfiltered = vdal.find_missing('their_raw_quote', limit=20)
        unfiltered_ids = {r['id'] for r in unfiltered}
        for nid in with_quote_ids + without_ids:
            self.assertIn(nid, unfiltered_ids,
                          'unfiltered must return all missing-vector nodes')

        # With the filter: only the 3 with their_raw_quote
        filtered = vdal.find_missing(
            'their_raw_quote', limit=20,
            source_kv_keys=['their_raw_quote'])
        filtered_ids = {r['id'] for r in filtered}
        for nid in with_quote_ids:
            self.assertIn(nid, filtered_ids,
                          'filtered must return nodes WITH their_raw_quote')
        for nid in without_ids:
            self.assertNotIn(nid, filtered_ids,
                             'filtered must EXCLUDE nodes without the kv key')

    def test_filter_or_semantics_across_multiple_keys(self):
        """source_kv_keys is OR-style: a node qualifies if ANY of the keys
        is populated. Used by groups like high_meta that blend multiple kv
        fields (situation OR their_raw_quote OR my_raw_quote)."""
        only_situation = self._make_node('OnlySituation', situation='When debugging')
        only_quote = self._make_node('OnlyQuote', their_raw_quote='said this')
        both = self._make_node('Both', situation='When', their_raw_quote='said')
        neither = self._make_node('Neither')

        vdal = VectorDAL(self.brain.conn)

        result = vdal.find_missing(
            'high_meta', limit=20,
            source_kv_keys=['situation', 'their_raw_quote', 'my_raw_quote'])
        ids = {r['id'] for r in result}
        self.assertIn(only_situation, ids)
        self.assertIn(only_quote, ids)
        self.assertIn(both, ids)
        self.assertNotIn(neither, ids,
                         'node with NONE of the kv keys must be excluded')

    def test_filter_skipped_when_keys_none(self):
        """source_kv_keys=None preserves original behavior — no filter."""
        node_ids = [self._make_node(f'Plain {i}') for i in range(3)]
        vdal = VectorDAL(self.brain.conn)
        result = vdal.find_missing('high_meta', limit=20, source_kv_keys=None)
        ids = {r['id'] for r in result}
        for nid in node_ids:
            self.assertIn(nid, ids)


    def test_whitespace_only_value_is_not_eligible(self):
        """A whitespace-only kv value yields no embed text (the text-builder
        uses `val.strip()`), so find_missing must exclude it via trim() — else
        it clogs the batch AND false-trips the dead-handler alarm. Locks the
        filter<->builder alignment."""
        ws = self._make_node('Whitespace')
        self.brain.conn.execute(
            "INSERT OR REPLACE INTO node_metadata_kv (node_id, key, value) "
            "VALUES (?, ?, ?)", (ws, 'situation', '   '))
        real = self._make_node('RealSituation', situation='When debugging the daemon')
        self.brain.conn.commit()

        vdal = VectorDAL(self.brain.conn)
        ids = {r['id'] for r in vdal.find_missing(
            'high_meta', limit=20, source_kv_keys=['situation'])}
        self.assertIn(real, ids, 'a real situation value must be eligible')
        self.assertNotIn(ws, ids,
                         'whitespace-only kv value must be excluded (trim semantics)')


if __name__ == '__main__':
    unittest.main()
