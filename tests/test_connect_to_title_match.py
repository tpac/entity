"""connect_to catalog-title resolution — the deterministic ladder's Pass 2
(`_match_catalog_title`, decision 2026-07-30: vectors deleted from the write
path; token-exact or bounded near-title only).

Contract under test:
  • distance 0 (normalized token equality) resolves at any length
  • distance 1..NEAR_TITLE_MAX_OPS resolves ONLY behind three gates:
    ≥ NEAR_TITLE_MIN_TOKENS distinct query tokens, unique best candidate,
    runner-up ≥ NEAR_TITLE_MARGIN ops further out
  • ambiguity/photo-finish REFUSES (loud) rather than picks
  • acceptance at distance ≥ 1 logs connect_to_near_title (tolerance is
    visible, never silent)
  • the whole pass works with no embedder — needs_embedder=False here IS the
    determinism proof (the old fuzzy pass required vectors)

The class exercises the real write path via remember(connect_to=...), not the
helper in isolation, so the ladder order (id → sibling → catalog) stays pinned.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.brain_remember import _title_tokens, _token_edit_distance


class TestTitleTokens(unittest.TestCase):
    def test_punctuation_variance_vanishes(self):
        # Hyphen/em-dash/percent/colon variance normalizes away; numbers stay.
        self.assertEqual(_title_tokens('cap-of-4 change: 19% — done'),
                         ['cap', 'of', '4', 'change', '19', 'done'])
        self.assertEqual(_title_tokens('Cap of 4 change 19 done'),
                         ['cap', 'of', '4', 'change', '19', 'done'])

    def test_edit_distance_order_sensitive(self):
        a = ['recall', 'surface', 'design']
        b = ['surface', 'recall', 'design']
        self.assertEqual(_token_edit_distance(a, b, 4), 2)

    def test_edit_distance_caps_out(self):
        self.assertEqual(
            _token_edit_distance(['a'] * 10, ['b'] * 10, 2), 3)


class TestConnectToCatalogTitleMatch(BrainTestBase):
    needs_embedder = False   # determinism proof: no vectors anywhere

    def _mk(self, title):
        return self.brain.remember(type='fact', title=title, content='c',
                                   encoding_source='anchor')['id']

    def _connect(self, title_ref):
        src = self._mk('source node for edge %s' % title_ref[:12])
        r = self.brain._apply_connect_to(
            src, [{'title': title_ref, 'relation': 'extends',
                   'why': 'test edge with enough length to pass'}],
            encoding_source='anchor')
        return r

    def test_exact_normalized_title_resolves_any_length(self):
        nid = self._mk('Daemon restart protocol')   # 3 tokens < MIN_TOKENS
        r = self._connect('daemon RESTART — protocol')  # dist 0 after normalize
        self.assertEqual(len(r['created']), 1)
        self.assertEqual(r['created'][0]['target_id'], nid)

    def test_one_op_paraphrase_resolves_with_loud_log(self):
        # The Graph→Enrichment replica: one substituted token among many.
        nid = self._mk('Enrichment lane ceiling conversion: 19% of 52-gold '
                       'ceiling at current scoring')
        r = self._connect('Graph lane ceiling conversion: 19% of 52-gold '
                          'ceiling at current scoring')
        self.assertEqual(len(r['created']), 1)
        self.assertEqual(r['created'][0]['target_id'], nid)
        warned = self.brain._logs_dal.conn.execute(
            "SELECT COUNT(*) FROM debug_log "
            "WHERE source='connect_to_near_title'").fetchone()[0]
        self.assertGreaterEqual(warned, 1)

    def test_two_op_paraphrase_resolves(self):
        nid = self._mk('Enrichment lane ceiling conversion nineteen of '
                       'fifty-two gold ceiling at current scoring')
        r = self._connect('Graph lane ceiling conversion nineteen of '
                          'fifty-two gold FLOOR at current scoring')
        self.assertEqual(len(r['created']), 1)
        self.assertEqual(r['created'][0]['target_id'], nid)

    def test_three_op_paraphrase_drops(self):
        self._mk('Enrichment lane ceiling conversion nineteen of '
                 'fifty-two gold ceiling at current scoring')
        r = self._connect('Graph lane FLOOR conversion nineteen of '
                          'fifty-two gold BASEMENT at current scoring')
        self.assertEqual(r['created'], [])
        self.assertEqual(len(r['failed']), 1)

    def test_short_query_near_miss_drops(self):
        # 4 distinct tokens < NEAR_TITLE_MIN_TOKENS: near-title gate closed.
        self._mk('daemon restart protocol steps')
        r = self._connect('daemon reboot protocol steps')   # 1 op, too short
        self.assertEqual(r['created'], [])

    def test_ambiguous_tie_refuses(self):
        self._mk('recall pipeline stage one design for surface selection')
        self._mk('recall pipeline stage two design for surface selection')
        r = self._connect('recall pipeline stage six design for surface '
                          'selection')   # 1 op from BOTH
        self.assertEqual(r['created'], [])
        self.assertIn('ambiguous', r['failed'][0]['reason'])

    def test_photo_finish_inside_margin_refuses(self):
        # best=1, runner=2 → margin 1 < NEAR_TITLE_MARGIN → refuse.
        self._mk('recall pipeline stage one design for surface selection')
        self._mk('recall pipeline stage one design for surface projection')
        r = self._connect('recall pipeline stage six design for surface '
                          'selection')
        self.assertEqual(r['created'], [])
        self.assertIn('photo-finish', r['failed'][0]['reason'])

    def test_exact_match_ignores_near_twin(self):
        # A verbatim copy is unambiguous intent even when a 1-op twin exists.
        nid = self._mk('recall pipeline stage one design for surface selection')
        self._mk('recall pipeline stage two design for surface selection')
        r = self._connect('recall pipeline stage one design for surface '
                          'selection')
        self.assertEqual(len(r['created']), 1)
        self.assertEqual(r['created'][0]['target_id'], nid)

    def test_duplicate_exact_titles_refuse(self):
        self._mk('duplicated exact title for ambiguity test case')
        self._mk('duplicated exact title for ambiguity test case')
        r = self._connect('duplicated exact title for ambiguity test case')
        self.assertEqual(r['created'], [])
        self.assertIn('ambiguous', r['failed'][0]['reason'])

    def test_id_pass_still_wins(self):
        nid = self._mk('a node referenced by id not title')
        r = self._connect(nid)
        self.assertEqual(len(r['created']), 1)
        self.assertEqual(r['created'][0]['target_id'], nid)

    # ── recall-guarantee regressions (bug 69c2cbab) ──

    def test_probe_count_covers_dp_cap(self):
        """69c2cbab #2: the margin gate reasons about runners out to `cap`
        (=3), so probes must cover cap, not just the acceptance bar. A d=3
        runner sharing ONLY the 4th-longest token was invisible with 3
        probes — margin passed and a photo-finish was ACCEPTED. With cap+1
        probes it is seen, and the match refuses."""
        # 4 distinct-length long tokens pin probe selection deterministically.
        self._mk('aaaaaaaaaa bbbbbbbbb cccccccc ddddddd one two three four five')  # d=0 base
        # Best candidate: d=2 from the query (two short-token subs).
        # (base above IS the best at d=2; runner below at d=3.)
        r = self._connect('aaaaaaaaaa bbbbbbbbb cccccccc ddddddd uno dos three four five')
        # sanity: with no runner, the 2-op match accepts
        self.assertEqual(len(r['created']), 1)
        # Runner at d=3 from the SAME query, sharing only 'ddddddd' (the 4th
        # probe) among the long tokens.
        self._mk('xxxxxxxxxx yyyyyyyyy zzzzzzzz ddddddd uno dos three four five')
        r2 = self._connect('aaaaaaaaaa bbbbbbbbb cccccccc ddddddd uno dos three four five')
        self.assertEqual(r2['created'], [])
        self.assertIn('photo-finish', r2['failed'][0]['reason'])

    def test_content_mentions_cannot_crowd_pool(self):
        """69c2cbab #1: the candidate probe is title-scoped — a node that
        merely MENTIONS probe tokens in content must not enter the pool."""
        target = self._mk('zebrafish study alpha protocol design notes')
        self.brain.remember(type='fact', title='completely unrelated title',
                            content='zebrafish zebrafish zebrafish study '
                                    'alpha protocol design notes',
                            encoding_source='anchor')
        rows, saturated = self.brain._title_candidate_rows(
            ['zebrafish', 'protocol', 'study', 'alpha'])
        ids = {r[0] for r in rows}
        self.assertIn(target, ids)
        self.assertFalse(saturated)
        self.assertEqual(len(ids), 1, 'content-only mention entered the pool')

    def test_saturated_pool_refuses(self):
        """69c2cbab #1: when the FTS pool hits its limit, recall can no
        longer be assumed — the write path must refuse, not match."""
        target_title = 'saturation probe target with many distinct tokens'
        self._mk(target_title)
        real_door = self.brain._title_candidate_rows
        self.brain._title_candidate_rows = (
            lambda tokens, limit=None: (real_door(tokens, limit)[0], True))
        try:
            r = self._connect(target_title)
        finally:
            del self.brain._title_candidate_rows
        self.assertEqual(r['created'], [])
        self.assertIn('saturated', r['failed'][0]['reason'])

    def test_per_node_connect_to_excludes_batch_siblings(self):
        """69c2cbab #3: a just-created sibling 1 op from a real catalog
        target must not tie it into an ambiguity refusal — per-node
        connect_to excludes the batch's own creations from catalog
        candidacy (sibling resolution stays exact-only Pass 1)."""
        catalog = self._mk('recall pipeline stage one design for surface selection')
        result = self.brain.remember_batch(nodes=[
            {'type': 'fact', 'title': 'source node in batch', 'content': 'c',
             'encoding_source': 'anchor',
             'connect_to': [{'title': 'recall pipeline stage six design for '
                                      'surface selection',
                             'relation': 'extends',
                             'why': 'near-title ref to catalog node only'}]},
            {'type': 'fact',
             'title': 'recall pipeline stage two design for surface selection',
             'content': 'sibling 1 op from the entry', 'encoding_source': 'anchor'},
        ])
        made = result.get('connect_to_made') or []
        self.assertEqual(len(made), 1, result.get('connect_to_failed'))
        self.assertEqual(made[0]['target_id'], catalog)

    def test_tokenizer_correspondence_no_dead_probes(self):
        """69c2cbab #4: _title_tokens and FTS5 (porter unicode61,
        remove_diacritics default) must tokenize compatibly — every token
        the matcher emits for a realistic-nasty title must find that title
        through the FTS door, alone and as a probe set."""
        titles = [
            'cap-of-4 change: 19% — done (4549bfd)',
            'LAF_v1 scoring über-lane at 52-gold ceiling',
            'Runner seam Phase 1 live (945feba, 2026-07-28): encoder-lane SDK',
            'consolidation scoring ceilings converge across scored runs',
        ]
        for title in titles:
            nid = self._mk(title)
            toks = _title_tokens(title)
            self.assertTrue(toks, title)
            for tok in toks:
                rows, _ = self.brain._title_candidate_rows([tok])
                self.assertIn(nid, {r[0] for r in rows},
                              'dead probe %r for title %r' % (tok, title))

    def test_archived_nodes_invisible(self):
        nid = self._mk('archived target title that must never resolve here')
        self.brain.archive_node(nid, archived_by='test')
        r = self._connect('archived target title that must never resolve here')
        self.assertEqual(r['created'], [])


if __name__ == '__main__':
    unittest.main()
