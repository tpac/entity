"""S1 Scribe model resolution: the encoder model comes from the s1e
interaction config, not a source literal.

`run_encoding_agent` is monolithic (catalog, traces, loop), so a
behavioural test would need a full brain + LLM double. These pins hold the
two lines that make the model table-driven, the same way the query-expansion
gate is pinned: as live statements, not substrings that could survive in a
comment. Reverting the `run_llm_loop` call to a model literal fails the
second test.
"""

import unittest
from pathlib import Path

from servers.scales.s1 import encode


class ScribeModelResolutionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        src = Path(encode.__file__).read_text()
        cls.src = src
        cls.stripped = [ln.strip() for ln in src.splitlines()]

    def test_model_is_read_from_the_s1e_interaction_config(self):
        self.assertIn(
            "enc_model = enc['config']['model']",
            self.stripped,
            'the Scribe model must resolve from the s1e interaction config — '
            'a subscript, no caller-side fallback: the resolver returns a '
            'total config (code default overlaid with any DB override)')
        self.assertIn(
            "enc_effort = enc['config']['effort']",
            self.stripped,
            "effort must subscript the resolved config too — the old "
            "`.get('effort') or None` silently turned the 'medium' code "
            "default into API-default high")

    def test_the_llm_loop_receives_the_resolved_model(self):
        self.assertIn(
            'model=enc_model,', self.stripped,
            'run_llm_loop must receive the config-resolved model')
        self.assertNotIn(
            'model="claude-sonnet-4-6"', self.src,
            'a model literal in the call would make the config key dead — '
            'the a6dfcfe3 failure shape: config carries a value nothing reads')


if __name__ == '__main__':
    unittest.main()
