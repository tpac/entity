"""Every LLM-facing edge render goes through contract.render_edge_lines — one
grammar for the encoder catalog, Anchor's get_node, the recall surface and
the S2 units, so they cannot drift and none of them truncates a description
(a reader may copy it verbatim as a swap's `old`).

render_rich_node is pinned in test_format_node.py and the consolidation
encoder in test_s2_consolidation_supersession.py; this file holds the healer,
which builds its own framing and used to cut descriptions to 60 chars.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase  # noqa: E402


class TestHealerEdgeRender(BrainTestBase):
    needs_embedder = False

    def test_healer_connections_render_whole_in_the_one_grammar(self):
        from servers.scales.s2.healer_encoder import HealerEncoder
        from servers.contract import render_edge_lines
        a = self.brain.remember(type='fact', title='Half-formed node', content='c')['id']
        b = self.brain.remember(type='decision', title='A neighbor whose title runs well past fifty characters, which the healer used to cut', content='c')['id']
        long_desc = 'w' * 100 + ' — the part past sixty characters the healer never saw'
        self.brain.connect_typed(a, b, relation='grounds', weight=0.6,
                                 description=long_desc, encoding_source='test')
        rich = self.brain.get_node(a)
        text = HealerEncoder(self.brain)._format_batch(
            [{'node_id': a, 'rich_node': rich, 'needs_question': True}])
        self.assertIn('CONNECTIONS (1):', text)
        self.assertIn(' — ' + long_desc, text)
        self.assertIn(rich['connections'][0]['title'][:100], text)
        # byte-identical to the shared renderer's line, at the healer's indent
        (expected,) = render_edge_lines(rich['connections'][0],
                                        {'time_format': 'relative'}, indent='  ')
        self.assertIn(expected, text)

    def test_healer_shows_a_corrector_relation_once(self):
        """CORRECTIONS renders the corrector's verb and description whole, so
        CONNECTIONS leaves that relation out and keeps the pair's others."""
        from servers.scales.s2.healer_encoder import HealerEncoder
        a = self.brain.remember(type='fact', title='Half-formed node', content='c')['id']
        b = self.brain.remember(type='decision', title='The newer word', content='c')['id']
        self.brain.connect_typed(b, a, relation='supersedes', weight=0.6,
                                 description='the newer word replaced it',
                                 encoding_source='test')
        self.brain.connect_typed(b, a, relation='extends', weight=0.5,
                                 description='and builds on it', encoding_source='test')
        text = HealerEncoder(self.brain)._format_batch(
            [{'node_id': a, 'rich_node': self.brain.get_node(a), 'needs_question': True}])
        self.assertIn('⚠ Replaced by "The newer word"', text)
        self.assertEqual(text.count('the newer word replaced it'), 1)
        self.assertIn('"The newer word" extends this — and builds on it', text)
        # the node's own content first, then what corrected it, then edges
        self.assertLess(text.index('CONTENT:'), text.index('CORRECTIONS:'))
        self.assertLess(text.index('CORRECTIONS:'), text.index('CONNECTIONS'))


if __name__ == '__main__':
    import unittest
    unittest.main()
