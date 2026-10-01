"""The inject render policy and the shown/selected trace contract behind it.

Seeds first: every Haiku pick renders, spread neighbors are one title line
each. Whole or nothing: content is shown whole, cut on a sentence boundary
with the remainder named, or left out with a pointer — never a stub. Field
set: counterpart's quote in, situation / reasoning / my quote / bookkeeping
out, sparse fields when present, edges as titles with ids, corrections
capped. The K trace's `selected` is what the stream SAW; the picks and every
reason a pick did not render ride alongside.
"""
import json
import unittest

from servers.contract import CONTENT_CUT_MARKER, render_corrections
from servers.loud_truncation import cap_text_at_boundary
from servers.scales.s1.surface_contract import (
    SURFACE, SURFACE_ALSO_LIT_LIMIT, SURFACE_CONTENT_MIN_CHARS,
    SURFACE_CORRECTION_LIMIT, SURFACE_INJECT_BUDGET, _MAX_INJECT_CHARS,
    _build_user_content_xml, _render_seed, render_surface_inject,
)
from tests.brain_test_base import BrainTestBase


_SENTENCE = ('This is sentence number %d of the memory body, carrying one '
             'concrete claim about the design. ')


def _body(chars, tag=''):
    """Whole sentences up to `chars` (never a mid-word end), opened by `tag`
    so every synthetic node's content is distinct."""
    out = ('Node %s opens. ' % tag) if tag else ''
    for k in range(1, 400):
        piece = _SENTENCE % k
        if len(out) + len(piece) > chars:
            break
        out += piece
    return out.rstrip()


def _node(i, content_chars, *, typ='decision', quote=True, corrections=0,
          edges=3, event=False, refs=None, status=None, my_quote=True):
    nid = '%08x' % (0x10000000 + i)
    meta = {'situation': 'When doing X and Y happens',
            'reasoning': 'Because it was measured',
            'question': 'What about X?', 'project': 'brain',
            'source_context': 'session abc', 'community_size': '12'}
    if quote:
        meta['their_raw_quote'] = 'traces are the source of truth'
    if my_quote:
        meta['my_raw_quote'] = 'I retract the delta proposal'
    if event:
        meta['event_time'] = '2026-09-12'
    if status:
        meta['evolution_status'] = status
    if i % 2:
        meta['correction_pattern'] = 'treating merge and deploy as one gate'
    return {
        'id': nid, 'type': typ,
        'title': 'Seed %d about the inject render policy' % i,
        'content': _body(content_chars, tag=nid),
        'created_at': '2026-09-01T00:00:00+00:00',
        '_metadata': meta,
        'connections': [
            {'id': '%08x' % (0x20000000 + i * 10 + e),
             'title': 'Neighbor %d-%d' % (i, e), 'type': 'finding',
             'relation': 'grounds', 'direction': 'outgoing',
             'description': 'a long edge description ' * 8,
             'created_at': '2026-08-01T00:00:00+00:00'}
            for e in range(edges)],
        '_corrections': [
            {'id': '%08x' % (0x30000000 + i * 10 + c),
             'title': 'Correction %d-%d' % (i, c), 'type': 'correction',
             'direction': 'corrector', 'relation': 'supersedes',
             'edge_description': 'why ' * 20, 'content': 'body ' * 20}
            for c in range(corrections)],
        'source_refs': list(refs or []),
    }


SCOPE = {'project': 'brain', 'counterpart': 'Sam'}


class TestSentenceCut(unittest.TestCase):

    def _cut(self, s, limit):
        return cap_text_at_boundary(s, limit, marker=CONTENT_CUT_MARKER)

    def test_fits_unchanged(self):
        s = _body(300)
        self.assertEqual(self._cut(s, 400), s)

    def test_cuts_on_sentence_and_names_remainder(self):
        s = _body(3000)
        out = self._cut(s, 1000)
        kept, _, marker = out.partition(' … ')
        self.assertTrue(kept.endswith('.'), kept[-40:])
        self.assertLessEqual(len(kept), 1000)
        self.assertEqual(marker, '(+%d chars: get_nodes)' % (len(s) - len(kept)))

    def test_no_sentence_falls_back_to_word_break(self):
        s = ' '.join(['word'] * 400)
        out = self._cut(s, 500)
        kept = out.split(' … ')[0]
        self.assertTrue(kept.endswith('word'))
        self.assertLessEqual(len(kept), 500)


class TestCorrectionCap(unittest.TestCase):

    def _corrs(self, n):
        return [{'id': '%08x' % (0x40000000 + c), 'title': 'C%d' % c,
                 'direction': 'corrector', 'relation': 'supersedes'}
                for c in range(n)]

    def test_limit_renders_first_n_and_counts_the_rest(self):
        lines = render_corrections(self._corrs(5), mode='lean', limit=2)
        self.assertEqual(len(lines), 3)
        self.assertIn('C0', lines[0])
        self.assertIn('C1', lines[1])
        self.assertEqual(lines[2].strip(), '+3 more corrections')

    def test_no_limit_renders_all(self):
        self.assertEqual(len(render_corrections(self._corrs(5), mode='lean')), 5)

    def test_singular_count(self):
        lines = render_corrections(self._corrs(3), mode='lean', limit=2)
        self.assertEqual(lines[-1].strip(), '+1 more correction')

    def test_header_title_is_whole_to_the_cap_then_loud(self):
        """The ⚠ header is a WARNING, not a lure: it must land before it is
        cut. This brain puts the verdict at the END of a title and the old
        60-char cut removed it from 92% of them. Whole to TITLE_WARN_LIMIT,
        and loud past it so a copied title is still diagnosable."""
        from servers.contract import TITLE_WARN_LIMIT
        from servers.loud_truncation import looks_truncated
        fits = 'A' * (TITLE_WARN_LIMIT - 20) + ' — the verdict'
        line = render_corrections([dict(self._corrs(1)[0], title=fits)],
                                  mode='lean')[0]
        self.assertIn(fits, line)
        over = 'B' * (TITLE_WARN_LIMIT + 50)
        cut = render_corrections([dict(self._corrs(1)[0], title=over)],
                                 mode='lean')[0]
        self.assertTrue(looks_truncated(cut.split('" (id:')[0]), cut)

    def test_edge_description_renders_whole(self):
        """A reader may copy the why verbatim as a swap's `old` — the same
        promise render_edge_lines makes — so the correction block must not
        cut it; a cut copy fails the exactly-once match every time."""
        why = ('the earlier review carried this as named-not-done; ' * 8).strip()
        self.assertGreater(len(why), 200)
        corr = dict(self._corrs(1)[0], edge_description=why)
        lines = render_corrections([corr], mode='balanced')
        # its own line, under the ⚠ line — the Edges list's `description:`
        self.assertEqual(lines[1], '     description: %s' % why)
        self.assertNotIn('…', lines[1])


class TestSeedFieldPolicy(unittest.TestCase):
    """What one rendered pick shows and hides."""

    def _render(self, node, mode='arc', cap=None, scope=SCOPE):
        return _render_seed(node, mode, cap, scope=scope)[0]

    def test_ruled_fields_out(self):
        text = self._render(_node(1, 500, event=True, status='disproven'))
        for label in ('Situation:', 'Reasoning:', 'Question:', 'My Raw Quote',
                      'Source Context', 'Community Size', 'Event Time:',
                      'Evolution Status', 'Brain activated'):
            self.assertNotIn(label, text, label)

    def test_counterpart_quote_labeled_with_name(self):
        text = self._render(_node(1, 500))
        self.assertIn('  Sam said: traces are the source of truth', text)
        self.assertNotIn('Their Raw Quote', text)

    def test_quote_label_without_counterpart(self):
        text = self._render(_node(1, 500), scope=None)
        self.assertIn('  They said: traces are the source of truth', text)

    def test_event_date_once_under_header(self):
        text = self._render(_node(1, 500, event=True))
        lines = text.split('\n')
        self.assertEqual(lines[1], '  Event date: 2026-09-12')
        self.assertEqual(text.count('2026-09-12'), 1)

    def test_status_marks_header(self):
        text = self._render(_node(1, 500, status='disproven'))
        self.assertIn('⚠ DISPROVEN', text.split('\n')[0])
        text = self._render(_node(1, 500, status='validated'))
        self.assertNotIn('⚠', text.split('\n')[0])

    def test_sparse_fields_when_present(self):
        text = self._render(_node(1, 500))
        self.assertIn('  Correction Pattern: treating merge and deploy as one gate', text)
        self.assertNotIn('Correction Pattern', self._render(_node(2, 500)))

    def test_conversation_line_when_source_refs(self):
        text = self._render(_node(1, 500, refs=['1f8ef04c', 'bb247556']))
        self.assertIn('  Conversation: get_traces(["1f8ef04c", "bb247556"])', text)
        self.assertNotIn('Conversation:', self._render(_node(1, 500)))

    def test_corrections_capped(self):
        text = self._render(_node(1, 500, corrections=SURFACE_CORRECTION_LIMIT + 3))
        self.assertEqual(text.count('⚠ Updated by "'), SURFACE_CORRECTION_LIMIT)
        self.assertIn('+3 more corrections', text)

    def test_edges_are_titles_with_ids_no_descriptions(self):
        text = self._render(_node(1, 500))
        self.assertIn('    this grounds "Neighbor 1-0" (id:2000000a)', text)
        self.assertNotIn('a long edge description', text)

    def test_background_is_header_only(self):
        node = _node(1, 500, event=True)
        node['situation'] = 'top-level, as the canonical pull promotes it'
        text = self._render(node, mode='background')
        lines = text.split('\n')
        self.assertEqual(len(lines), 2, text)
        self.assertTrue(lines[0].startswith('[decision] "Seed 1'))
        self.assertEqual(lines[1], '  Event date: 2026-09-12')

    def test_top_level_situation_never_renders(self):
        node = _node(1, 500)
        node['situation'] = 'top-level, as the canonical pull promotes it'
        for mode in ('arc', 'fact'):
            self.assertNotIn('Situation', self._render(node, mode=mode))

    def test_fact_content_whole_over_cap(self):
        node = _node(1, 3000)
        text = self._render(node, mode='fact', cap=500)
        self.assertIn(node['content'], text)

    def test_arc_whole_when_it_fits(self):
        node = _node(1, 800)
        self.assertIn(node['content'], self._render(node, cap=2000))

    def test_arc_cut_on_sentence_with_pointer(self):
        node = _node(1, 3000)
        text = self._render(node, cap=1000)
        content_line = [l for l in text.split('\n') if l.startswith('  Content:')][0]
        self.assertRegex(content_line, r'\. … \(\+\d+ chars: get_nodes\)$')
        self.assertLess(len(content_line), 1100)

    def test_arc_omitted_below_minimum_with_pointer(self):
        node = _node(1, 3000)
        text = self._render(node, cap=SURFACE_CONTENT_MIN_CHARS - 50)
        self.assertNotIn('  Content:', text)
        self.assertIn('  (content: %d chars — get_nodes)' % len(node['content']), text)
        self.assertIn('Sam said:', text)


class TestInjectBudget(unittest.TestCase):
    """Seeds first, neighbors as one line, leftover flows back, hard cap."""

    def _world(self, seed_specs, neighbors=0, neighbor_chars=500):
        rich, selected, acts = {}, {}, {}
        for i, chars, mode in seed_specs:
            n = _node(i, chars)
            rich[n['id']] = n
            selected[n['id']] = mode
            acts[n['id']] = 1.0 - i * 0.05
        for j in range(neighbors):
            n = _node(100 + j, neighbor_chars, typ='finding', quote=False, edges=1)
            rich[n['id']] = n
            acts[n['id']] = 0.9 - j * 0.005
        return rich, selected, acts

    def test_every_pick_renders_with_many_lit_neighbors(self):
        rich, selected, acts = self._world(
            [(i, 1500, 'arc') for i in range(1, 6)], neighbors=60)
        out = render_surface_inject(acts, {}, rich, selected_mode=selected, scope=SCOPE)
        self.assertEqual(set(out['shown']), set(selected))
        self.assertEqual(out['not_shown'], [])
        self.assertNotIn('Brain activated', out['text'])
        for nid in selected:
            self.assertIn('(id:%s' % nid[:8], out['text'])

    def test_neighbors_are_title_lines_capped(self):
        rich, selected, acts = self._world([(1, 800, 'arc')], neighbors=60)
        out = render_surface_inject(acts, {}, rich, selected_mode=selected, scope=SCOPE)
        self.assertEqual(len(out['also_lit']), SURFACE_ALSO_LIT_LIMIT)
        block = out['text'].split('Also lit (%d):\n' % SURFACE_ALSO_LIT_LIMIT)[1]
        lines = block.split('\n')
        self.assertEqual(len(lines), SURFACE_ALSO_LIT_LIMIT)
        for line in lines:
            self.assertRegex(line, r'^  \[finding\] "Seed \d+ about the inject render policy" \(id:[0-9a-f]{8}\)$')
        # A neighbor's body never renders — only its title line.
        self.assertNotIn(rich[out['also_lit'][0]]['content'][:80], out['text'])

    def test_neighbors_exclude_seen_and_seeds(self):
        rich, selected, acts = self._world([(1, 800, 'arc')], neighbors=12)
        seen = ['%08x' % (0x10000000 + 100), '%08x' % (0x10000000 + 101)]
        out = render_surface_inject(acts, {}, rich, selected_mode=selected,
                                    scope=SCOPE, seen_ids=seen)
        self.assertFalse(set(seen) & set(out['also_lit']))
        self.assertFalse(set(selected) & set(out['also_lit']))
        self.assertEqual(len(out['also_lit']), SURFACE_ALSO_LIT_LIMIT)

    def test_seed_order_follows_activation(self):
        rich, selected, acts = self._world([(3, 500, 'arc'), (1, 500, 'arc'), (2, 500, 'arc')])
        out = render_surface_inject(acts, {}, rich, selected_mode=selected, scope=SCOPE)
        self.assertEqual([s[-1] for s in out['shown']], ['1', '2', '3'])

    def test_leftover_flows_back_to_cut_seed(self):
        # One short seed, one long seed: the long one gets more than half.
        rich, selected, acts = self._world([(1, 200, 'arc'), (2, 6000, 'arc')])
        out = render_surface_inject(acts, {}, rich, selected_mode=selected,
                                    scope=SCOPE, total_budget=4000)
        long_block = out['text'].split('\n\n')[1]
        content_line = [l for l in long_block.split('\n') if l.startswith('  Content:')][0]
        self.assertGreater(len(content_line), 2600)
        self.assertLessEqual(len(out['text']), 4000 + 200)

    def test_budget_does_not_scale_with_lit_count(self):
        few = self._world([(i, 2000, 'arc') for i in range(1, 4)], neighbors=2)
        many = self._world([(i, 2000, 'arc') for i in range(1, 4)], neighbors=70)
        out_few = render_surface_inject(few[2], {}, few[0], selected_mode=few[1], scope=SCOPE)
        out_many = render_surface_inject(many[2], {}, many[0], selected_mode=many[1], scope=SCOPE)

        def content_chars(out):
            return sum(len(l) for l in out['text'].split('\n') if l.startswith('  Content:'))
        # The neighbor list costs ~10 lines; the seeds' content moves by no
        # more than that reservation, not by an order of magnitude.
        self.assertGreater(content_chars(out_many), content_chars(out_few) * 0.8)

    def test_long_fact_pick_does_not_starve_the_others(self):
        # One verbatim pick eating most of the soft target plus four arc
        # picks: the arc picks still show real content, grown into the room
        # below the hard cap, and nothing renders as a bare pointer.
        rich, selected, acts = self._world(
            [(1, 5000, 'fact')] + [(i, 1500, 'arc') for i in range(2, 6)])
        out = render_surface_inject(acts, {}, rich, selected_mode=selected, scope=SCOPE)
        self.assertEqual(len(out['shown']), 5)
        self.assertNotIn('(content:', out['text'])
        self.assertLessEqual(len(out['text']), _MAX_INJECT_CHARS)
        content_lines = [l for l in out['text'].split('\n') if l.startswith('  Content:')]
        self.assertEqual(len(content_lines), 5)
        for line in content_lines[1:]:
            self.assertGreater(len(line), SURFACE_CONTENT_MIN_CHARS)

    def test_second_pass_growth_stays_within_its_grant(self):
        # Omitted picks grow from zero, not from a cap they never spent, so
        # the second pass cannot push the inject over the hard cap and cost
        # the neighbor list to the shrink ladder.
        rich, selected, acts = self._world(
            [(1, 3000, 'fact')] + [(i, 2500, 'arc') for i in range(2, 6)], neighbors=10)
        out = render_surface_inject(acts, {}, rich, selected_mode=selected, scope=SCOPE)
        self.assertLessEqual(len(out['text']), _MAX_INJECT_CHARS)
        self.assertEqual(len(out['also_lit']), SURFACE_ALSO_LIT_LIMIT)
        self.assertEqual(len(out['shown']), 5)

    def test_neighbors_named_by_an_edge_line_are_not_repeated(self):
        rich, selected, acts = self._world([(1, 500, 'arc')], neighbors=12)
        seed = next(iter(selected))
        # Make the seed's first edge point at the strongest neighbor.
        top = max((n for n in acts if n != seed), key=acts.get)
        rich[seed]['connections'][0]['id'] = top
        out = render_surface_inject(acts, {}, rich, selected_mode=selected, scope=SCOPE)
        self.assertNotIn(top, out['also_lit'])
        self.assertIn('(id:%s)' % top[:8], out['text'])   # named once, by the edge

    def test_missing_rich_node_is_not_shown_with_reason(self):
        rich, selected, acts = self._world([(1, 500, 'arc'), (2, 500, 'arc')])
        ghost = '%08x' % 0x0badf00d
        selected[ghost] = 'arc'
        out = render_surface_inject(acts, {}, rich, selected_mode=selected, scope=SCOPE)
        self.assertEqual(set(out['shown']), set(selected) - {ghost})
        self.assertEqual(out['not_shown'], [{'id': ghost[:8], 'reason': 'no_node'}])

    def test_hard_cap_shrinks_from_the_least_valuable_end(self):
        # Three verbatim picks that cannot all fit whole: the neighbor list
        # goes first, then edges, then content to the minimum — every pick
        # still renders and nothing is cut mid-line.
        rich, selected, acts = self._world([(i, 5000, 'fact') for i in range(1, 4)], neighbors=8)
        out = render_surface_inject(acts, {}, rich, selected_mode=selected, scope=SCOPE)
        self.assertLessEqual(len(out['text']), _MAX_INJECT_CHARS)
        self.assertEqual(len(out['shown']), 3)
        self.assertEqual(out['not_shown'], [])
        self.assertEqual(out['also_lit'], [])
        self.assertNotIn('Also lit', out['text'])
        # Whole content ends on its own last sentence; a cut names the rest.
        for line in out['text'].split('\n'):
            if line.startswith('  Content:'):
                self.assertRegex(line, r'(\.|get_nodes\))$')
        # At least one of the three had to be cut to fit.
        self.assertIn('chars: get_nodes)', out['text'])

    def test_empty_selection_renders_nothing(self):
        rich, _, acts = self._world([(1, 500, 'arc')])
        out = render_surface_inject(acts, {}, rich, selected_mode={}, scope=SCOPE)
        self.assertEqual(out, {'text': '', 'shown': [], 'not_shown': [], 'also_lit': []})


class TestSelectorConversationCaps(unittest.TestCase):
    """The current message and the assistant's last message reach the picker
    at 8,000; older turns at their tighter caps."""

    def _xml(self, recent, current):
        built = _build_user_content_xml([], current, recent, None, '', SURFACE)
        return built[0] if isinstance(built, tuple) else built

    def test_caps_by_position(self):
        recent = [
            {'role': 'user', 'content': 'U1 ' * 700},        # 2,100 chars
            {'role': 'assistant', 'content': 'A1 ' * 2000},  # 6,000 chars
            {'role': 'user', 'content': 'U2 ' * 700},
            {'role': 'assistant', 'content': 'A2 ' * 2000},  # the last one
        ]
        current = 'C ' * 3500                                  # 7,000 chars
        xml = self._xml(recent, current)
        turns = xml.split('<turn ')[1:]
        self.assertEqual(len(turns), 3)
        u1 = turns[0].split('<user>')[1].split('</user>')[0]
        a1 = turns[0].split('<assistant>')[1].split('</assistant>')[0]
        a2 = turns[1].split('<assistant>')[1].split('</assistant>')[0]
        cur = turns[2].split('<user>')[1].split('</user>')[0]
        self.assertEqual(len(u1), SURFACE['user_message_limit'])
        self.assertEqual(len(a1), SURFACE['anchor_message_limit'])
        self.assertEqual(len(a2), 6000)          # whole: under last_anchor cap
        self.assertEqual(len(cur), 7000)         # whole: under current cap
        self.assertEqual(SURFACE['current_message_limit'], 8000)
        self.assertEqual(SURFACE['last_anchor_message_limit'], 8000)

    def test_last_assistant_cap_survives_a_trailing_lone_user(self):
        # An interrupted turn leaves a lone user message last; the assistant
        # message before it is still the most recent reply and keeps its cap.
        recent = [{'role': 'user', 'content': 'u1'},
                  {'role': 'assistant', 'content': 'A' * 6000},
                  {'role': 'user', 'content': 'wait, stop'}]
        xml = self._xml(recent, 'current')
        a = xml.split('<assistant>')[1].split('</assistant>')[0]
        self.assertEqual(len(a), 6000)

    def test_caps_cut_at_the_limit(self):
        recent = [{'role': 'user', 'content': 'u'},
                  {'role': 'assistant', 'content': 'x' * 9000}]
        xml = self._xml(recent, 'y' * 9000)
        a = xml.split('<assistant>')[1].split('</assistant>')[0]
        cur = xml.split('current_msg="true">')[1].split('<user>')[1].split('</user>')[0]
        self.assertEqual(len(a), SURFACE['last_anchor_message_limit'])
        self.assertEqual(len(cur), SURFACE['current_message_limit'])


class TestShownIsSelected(BrainTestBase):
    """End to end through run_surface: the K trace's `selected` is what the
    stream saw, the picker's decision rides alongside, the gates record why."""

    needs_embedder = False

    def _run(self, session_id, candidates, picks, recent_messages=None):
        from servers.scales.s1 import surface as surface_mod

        def _fake_call_surface(brain, cands, user_message, recent, sid, result,
                               frame='', scope=None):
            return ({'selected': [{'id': p[:8], 'mode': 'arc'} for p in picks]},
                    'prompt', 100, brain.get_interaction_stamp('surface'),
                    {'input_tokens': 50, 'output_tokens': 10, 'cache_read_tokens': 0,
                     'cache_creation_tokens': 0, 'elapsed_ms': 5, 'rounds': 1,
                     'truncated': 0})

        ctx = self.brain.get_or_create_session(session_id)
        orig = surface_mod._call_surface
        surface_mod._call_surface = _fake_call_surface
        try:
            text = surface_mod.run_surface(
                self.brain, ctx, candidates, 'user msg', recent_messages or [], {},
                'enriched query', [], 'test-recall-ref', session_id, None,
                query_vec=None)
        finally:
            surface_mod._call_surface = orig
        evts = self.brain.query_traces(
            scale='s1', ref_type='surface_selected',
            session_id=session_id, hours=None).get('events') or []
        self.assertTrue(evts, 'no surface_selected K trace written')
        return text, evts[0]

    def _remember(self, title, content='c'):
        return self.brain.remember(type='test', title=title, content=content,
                                   encoding_source='anchor:test')

    def _cand(self, node):
        return {'id': node['id'], 'title': node['title'], 'type': 'test', 'score': 0.9}

    def test_selected_is_shown_and_picked_is_logged(self):
        a = self._remember('shown_a', 'Alpha content for the inject.')
        b = self._remember('shown_b', 'Beta content for the inject.')
        text, k = self._run('t-shown-eq', [self._cand(a), self._cand(b)],
                            [a['id'], b['id']])
        shown = set(json.loads(k['ref_id']))
        self.assertEqual(shown, {a['id'][:8], b['id'][:8]})
        meta = k.get('metadata') or {}
        self.assertEqual(meta.get('picked'), sorted([a['id'][:8], b['id'][:8]]))
        self.assertEqual(meta.get('not_shown'), [])
        self.assertTrue(k['summary'].startswith('2 surfaced'), k['summary'])
        self.assertIn('Alpha content', text)
        self.assertIn('Beta content', text)
        self.assertNotIn('Brain activated', text)

    def test_already_shown_pick_is_gated_and_recorded(self):
        a = self._remember('seen_a', 'Already in context.')
        b = self._remember('seen_b', 'New to this turn.')
        recent = [{'role': 'user', 'content': 'earlier turn',
                   'surfaced': [{'id': a['id'][:8], 'title': 'seen_a'}]}]
        text, k = self._run('t-already-shown', [self._cand(a), self._cand(b)],
                            [a['id'], b['id']], recent_messages=recent)
        self.assertEqual(set(json.loads(k['ref_id'])), {b['id'][:8]})
        meta = k.get('metadata') or {}
        self.assertIn({'id': a['id'][:8], 'reason': 'already_shown'}, meta.get('not_shown'))
        self.assertIn(a['id'][:8], meta.get('picked'))
        self.assertNotIn('Already in context', text)
        # The Δ trace keeps the PICKER's verdict: a gated pick is not a
        # supervision negative.
        deltas = self.brain.query_traces(
            scale='s1', ref_type='additionalContext',
            session_id='t-already-shown', hours=None).get('events') or []
        self.assertTrue(deltas)
        outcomes = (deltas[0].get('metadata') or {}).get('outcomes_per_candidate') or {}
        self.assertEqual(outcomes.get(a['id'][:8]), 'selected')
        self.assertEqual(outcomes.get(b['id'][:8]), 'selected')

    def test_real_node_situation_never_renders(self):
        # get_node promotes situation to the top-level node key; the inject
        # must skip it there, not only in the KV loop.
        node = self.brain.remember(type='test', title='with_situation', content='Body.',
                                   situation='When the situation must not render',
                                   encoding_source='anchor:test')
        text, _ = self._run('t-situation', [self._cand(node)], [node['id']])
        self.assertIn('Body.', text)
        self.assertNotIn('Situation:', text)
        self.assertNotIn('must not render', text)

    def test_archived_pick_with_survivor_renders_the_survivor(self):
        live = self._remember('survivor_live', 'The knowledge lives on here.')
        dead = self._remember('absorbed_dead', 'Old form of the claim.')
        self.assertTrue(self.brain.archive_node(
            dead['id'], archived_by='anchor:test', reason='absorbed',
            survivor_id=live['id']).get('ok'))
        text, k = self._run('t-redirect', [self._cand(dead)], [dead['id']])
        self.assertEqual(set(json.loads(k['ref_id'])), {live['id'][:8]})
        meta = k.get('metadata') or {}
        self.assertEqual(meta.get('redirected'), {dead['id'][:8]: live['id'][:8]})
        self.assertEqual(meta.get('picked'), [dead['id'][:8]])
        self.assertIn('lives on here', text)
        self.assertIn('%s ↦ %s' % (dead['id'][:8], live['id'][:8]), text)

    def test_two_picks_absorbed_into_one_survivor_both_marked(self):
        live = self._remember('survivor_two', 'One body carries both.')
        d1 = self._remember('absorbed_one', 'Old form one.')
        d2 = self._remember('absorbed_two', 'Old form two.')
        for d in (d1, d2):
            self.assertTrue(self.brain.archive_node(
                d['id'], archived_by='anchor:test', reason='absorbed',
                survivor_id=live['id']).get('ok'))
        text, k = self._run('t-redirect-two', [self._cand(d1), self._cand(d2)],
                            [d1['id'], d2['id']])
        self.assertEqual(set(json.loads(k['ref_id'])), {live['id'][:8]})
        self.assertEqual(text.count('lives on here') + text.count('carries both'), 1)
        for d in (d1, d2):
            self.assertIn('%s ↦ %s' % (d['id'][:8], live['id'][:8]), text)

    def test_archived_pick_without_survivor_is_dropped_with_reason(self):
        live = self._remember('gate_live', 'Live content.')
        dead = self._remember('gate_dead', 'Dead content.')
        self.assertTrue(self.brain.archive_node(
            dead['id'], archived_by='anchor:test', reason='retired').get('ok'))
        text, k = self._run('t-no-survivor', [self._cand(live), self._cand(dead)],
                            [live['id'], dead['id']])
        self.assertEqual(set(json.loads(k['ref_id'])), {live['id'][:8]})
        meta = k.get('metadata') or {}
        self.assertIn({'id': dead['id'][:8], 'reason': 'archived_no_survivor'},
                      meta.get('not_shown'))
        self.assertNotIn('Dead content', text)

    def test_source_refs_reach_the_inject(self):
        node = self.brain.remember(type='test', title='with_refs', content='Anchored.',
                                   encoding_source='anchor:test',
                                   source_refs=['1f8ef04c', 'bb247556'])
        text, _ = self._run('t-refs', [self._cand(node)], [node['id']])
        self.assertIn('Conversation: get_traces(["1f8ef04c", "bb247556"])', text)


if __name__ == '__main__':
    unittest.main()
