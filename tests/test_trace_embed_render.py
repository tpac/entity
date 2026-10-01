"""Trace embedding render templates (v27 episodic-references substrate).

The worker calls _render_trace_for_embedding(trace_row) → str before
handing the text to the embedder. This is the §5.3 template applied
to live trace_event rows from find_unembedded().

Pure-function tests — no Brain, no embedder, no DB.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.embed_queue import _render_trace_for_embedding


def _row(ref_type, summary='', content=None, tool=None,
         human=None, agent=None):
    meta = {}
    if content is not None:
        meta['content'] = content
    if tool is not None:
        meta['tool'] = tool
    if human is not None:
        meta['human_identity'] = human
    if agent is not None:
        meta['agent_identity'] = agent
    return {'ref_type': ref_type, 'summary': summary, 'metadata': meta}


class IdentityPresentTest(unittest.TestCase):
    def test_user_message_uses_human_identity_and_content(self):
        row = _row('user_message', content='Hello there',
                   human='Ada', agent='Anchor')
        self.assertEqual(_render_trace_for_embedding(row), 'Ada: Hello there')

    def test_assistant_message_uses_agent_identity_and_content(self):
        row = _row('assistant_message', content='Hello back',
                   human='Ada', agent='Anchor')
        self.assertEqual(_render_trace_for_embedding(row),
                         'Anchor: Hello back')

    def test_tool_result_uses_agent_identity_and_tool(self):
        row = _row('tool_result', summary='Bash: ls -la',
                   tool='Bash', human='Ada', agent='Anchor')
        self.assertEqual(_render_trace_for_embedding(row),
                         'Anchor via Bash: Bash: ls -la')


class IdentityMissingFallbackTest(unittest.TestCase):
    """When identity isn't stamped (fresh install, env vars unset, or
    historical traces from before stamping was active), render with
    OPERATOR/ANCHOR sentinels so the embedding pipeline keeps producing
    usable vectors."""

    def test_user_message_falls_back_to_OPERATOR(self):
        row = _row('user_message', content='Hello')
        self.assertEqual(_render_trace_for_embedding(row),
                         'OPERATOR: Hello')

    def test_assistant_message_falls_back_to_ANCHOR(self):
        row = _row('assistant_message', content='Reply')
        self.assertEqual(_render_trace_for_embedding(row),
                         'ANCHOR: Reply')

    def test_tool_result_falls_back_to_ANCHOR_with_unknown_tool(self):
        row = _row('tool_result', summary='something')
        self.assertEqual(_render_trace_for_embedding(row),
                         'ANCHOR via tool: something')


class ContentVsSummaryPreferenceTest(unittest.TestCase):
    """§5.3: prefer metadata.content (full text) over summary (200-char cap)."""

    def test_content_preferred_when_both_present(self):
        row = _row('user_message',
                   summary='short summary',
                   content='full longer content',
                   human='Ada', agent='Anchor')
        self.assertEqual(_render_trace_for_embedding(row),
                         'Ada: full longer content')

    def test_summary_used_when_content_missing(self):
        row = _row('user_message', summary='only summary',
                   human='Ada', agent='Anchor')
        self.assertEqual(_render_trace_for_embedding(row),
                         'Ada: only summary')

    def test_empty_content_falls_back_to_summary(self):
        row = _row('user_message', summary='fallback', content='',
                   human='Ada', agent='Anchor')
        self.assertEqual(_render_trace_for_embedding(row),
                         'Ada: fallback')


class EdgeCasesTest(unittest.TestCase):
    def test_unknown_ref_type_best_effort(self):
        # Future S1 ref_types not yet templated — should still emit
        # something the embedder can use, not crash.
        row = _row('something_new', summary='whatever')
        self.assertEqual(_render_trace_for_embedding(row),
                         'something_new: whatever')

    def test_missing_metadata_handles_gracefully(self):
        row = {'ref_type': 'user_message', 'summary': 'bare row'}
        # No metadata key — should not crash
        self.assertEqual(_render_trace_for_embedding(row),
                         'OPERATOR: bare row')

    def test_null_metadata_handles_gracefully(self):
        row = {'ref_type': 'user_message', 'summary': 'null meta',
               'metadata': None}
        self.assertEqual(_render_trace_for_embedding(row),
                         'OPERATOR: null meta')


if __name__ == '__main__':
    unittest.main()
