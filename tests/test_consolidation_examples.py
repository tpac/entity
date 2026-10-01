"""Authoring gate for the consolidation prompt's worked examples.

Examples ARE the encoder's training signal (brain nodes f3086e73, 928f5694):
Sonnet imitates examples more than abstract rules. So the examples must be
flawless — this mechanical gate (no LLM) enforces:

  1. No truncation — full text, never '...' / '…' inside example code.
  2. Every `absorb` op rewrites BOTH title and content (the mandatory-rewrite
     contract; a content-less or title-stale absorb is the failure mode that
     drove losslessness to 68%).
  3. Every caller capability the encoder is expected to use is *exemplified*,
     not just described: absorb(title+content), KV revise, drop_fields,
     prune_edges, connect(similar_to), connect(correction), revise, disconnect.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.eval_optional import require_eval  # noqa: E402
require_eval()  # D-8: eval/ is absent from the public tree

from eval.agent_introspect.consolidation_contract_eval import validate_example_authoring

CANDIDATE = os.path.join(
    os.path.dirname(__file__), '..', 'eval', 'candidate_prompts',
    's2_consolidation_absorb.md')


def _report():
    with open(CANDIDATE) as f:
        return validate_example_authoring(f.read())


def test_examples_pass_authoring_gate():
    r = _report()
    assert r['ok'], 'authoring failures:\n  ' + '\n  '.join(r['failures'])


def test_every_caller_capability_exemplified():
    r = _report()
    missing = [k for k, v in r['coverage'].items() if not v]
    assert not missing, 'capabilities not exemplified: %s' % missing


def test_every_absorb_rewrites_title_and_content():
    r = _report()
    assert r['n_absorb'] >= 4, 'expected several absorb examples, got %d' % r['n_absorb']
    bad = [f for f in r['failures'] if f.startswith('absorb op missing')]
    assert not bad, 'absorbs missing title/content:\n  ' + '\n  '.join(bad)


def test_no_truncated_example_content():
    r = _report()
    trunc = [f for f in r['failures'] if 'truncation' in f]
    assert not trunc, 'truncated example content:\n  ' + '\n  '.join(trunc)
