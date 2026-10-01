"""Contract tests for the eval measurement-validity layer.

Three contracts that must not drift:
  1. suspect_reasons is the ONE vocabulary for per-rep suspect marking —
     table-tested so a renamed label breaks here, not silently in a report.
  2. The analyzer's gold-scan basis agrees with the classifier's own
     scannability gates (term extractor fallback + shared phrase constant) —
     the two modules once disagreed on golds like "220" ("encoder gap" in
     one, found in the other).
  3. A dead embedder cannot produce a green corpus: the build-error gate
     books every RECALL_FAILURE_MODES source RED, and the eval brain door
     refuses a dead embedder (checked against a mocked readiness seam).
No brain needed — pure functions plus one mocked module seam.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.eval_optional import require_eval  # noqa: E402
require_eval()  # D-8: eval/ is absent from the public tree

from eval.longmem.validity import suspect_reasons, RECALL_FAILURE_MODES
from eval.longmem.analyzer import (_gold_scan_basis, _gold_scan_terms,
                                   _find_gold_bearing_nodes)
from eval.longmem.classifier import _extract_key_terms, PHRASE_SCAN_MIN_CHARS


class TestSuspectReasons(unittest.TestCase):
    def test_clean_rep_is_valid(self):
        self.assertEqual(suspect_reasons(), [])

    def test_each_signal_maps_to_its_label(self):
        cases = [
            (dict(recall_mode='embed_failed'),
             ['recall_degraded:embed_failed']),
            (dict(recall_mode='embedder_unavailable'),
             ['recall_degraded:embedder_unavailable']),
            (dict(recall_mode='laf_v1'), []),  # healthy mode, not suspect
            (dict(new_errors=[{'type': 'x_failed'}]), ['brain_error:x_failed']),
            (dict(new_errors=[{'source': 's1e'}]), ['brain_error:s1e']),
            (dict(answerer_error='529'), ['answerer_error']),
            (dict(judge_parse_failed=True), ['judge_parse_failed']),
            (dict(harness_error='boom'), ['harness_error']),
        ]
        for kwargs, want in cases:
            self.assertEqual(suspect_reasons(**kwargs), want, kwargs)

    def test_signals_compose(self):
        got = suspect_reasons(recall_mode='embed_failed', harness_error='x',
                              answerer_error='y', judge_parse_failed=True,
                              new_errors=[{'type': 't'}])
        self.assertEqual(len(got), 5)

    def test_failure_modes_are_the_owners(self):
        # The tuple must come from servers.brain_recall — a local copy is
        # exactly the drift this module exists to prevent.
        from servers.brain_recall import RECALL_FAILURE_MODES as OWNER
        self.assertIs(RECALL_FAILURE_MODES, OWNER)


class TestGoldScanParity(unittest.TestCase):
    """The analyzer must be able to scan every gold the classifier can scan
    (and refuse the ones it refuses)."""

    def _classifier_scannable(self, gold: str) -> bool:
        terms = _extract_key_terms(gold)
        if len(terms) == 1 and len(terms[0]) < 2:
            terms = []
        phrase = len(gold) >= PHRASE_SCAN_MIN_CHARS
        return bool(terms or phrase)

    def test_parity_on_representative_golds(self):
        golds = ['220', '22', '1.5', 'AI', 'USD', '6 PM', '3', 'no',
                 'a walking tour of Kyoto temples', '3200 yen', 'x']
        for g in golds:
            analyzer_scannable = _gold_scan_basis(g) != 'unscannable'
            self.assertEqual(analyzer_scannable, self._classifier_scannable(g),
                             'scannability drift on gold %r' % g)

    def test_digit_gold_scans_by_terms(self):
        # "220" was the headline false "encoder gap": zero >=4-char terms.
        self.assertEqual(_gold_scan_basis('220'), 'terms')
        self.assertEqual(_gold_scan_terms('220'), ['220'])
        nodes = [{'id': 'n1' * 4, 'title': 'pace', 'content': 'reads 220 pages',
                  'kv': {}}]
        self.assertTrue(_find_gold_bearing_nodes(nodes, '220'))

    def test_short_phrase_never_substring_matches(self):
        # "AI" must not hit 'explain'; unscannable, not a verdict.
        self.assertEqual(_gold_scan_basis(''), 'unscannable')
        nodes = [{'id': 'n1' * 4, 'title': 'explain the plan', 'content': '',
                  'kv': {}}]
        self.assertEqual(_find_gold_bearing_nodes(nodes, ''), [])

    def test_voice_alias_fields_searched(self):
        nodes = [{'id': 'n1' * 4, 'title': 't', 'content': '',
                  'kv': {'anchor_raw_quote': 'the fare was 3200 yen'}}]
        self.assertTrue(_find_gold_bearing_nodes(nodes, '3200 yen fare'))


class TestDeadEmbedderBuildGates(unittest.TestCase):
    """A brain without vectors builds a corpus that is invalid but looks
    green (2026-09-18: dangling blob symlink in a seeded fastembed cache —
    recall_embedder_unavailable in brain.log, build completed). Two gates:
    the build-error classifier books every recall-failure source RED, and
    create_fresh_eval_brain refuses to hand out a brain whose embedder died."""

    def test_red_sources_decided_for_every_recall_failure_mode(self):
        from eval.longmem.build_corpus import (_RECALL_FAILURE_SOURCES,
                                               _RED_OVERRIDE_SOURCES,
                                               _is_benign_build_error)
        # Every owner mode has an explicit entry (a new mode must be DECIDED
        # here, not silently stay benign); None = deliberately not overridden.
        self.assertEqual(set(_RECALL_FAILURE_SOURCES), set(RECALL_FAILURE_MODES),
                         'a recall failure mode has no RED-source decision — '
                         'a build could go vector-blind and still book green')
        for source in _RED_OVERRIDE_SOURCES:
            # Each mapped class has a CONSTRUCTED log site (exc=None, no
            # stack frames) the frame rule alone would book benign — the
            # override must book that shape RED (caught-exception sites are
            # red either way):
            self.assertFalse(_is_benign_build_error(source, 'x', ''))

    def test_embedder_dead_classes_are_red(self):
        from eval.longmem.build_corpus import _is_benign_build_error
        self.assertFalse(_is_benign_build_error(
            'recall_embedder_unavailable', 'embedder not ready', ''))
        self.assertFalse(_is_benign_build_error(
            'recall_embed_failed', 'embed_query returned empty', ''))
        # Control: an unlisted constructed loud-log stays benign.
        self.assertTrue(_is_benign_build_error('some_designed_skip', 'x', ''))

    def test_laf_shared_sources_stay_benign_when_constructed(self):
        # 'recall_laf' constructed = the empty-scores exit, shared with a
        # benign state (fresh brain before its first embed drain has an empty
        # LAF universe on every healthy build's first turns); real LAF
        # failures RAISE and carry frames, red via the frame rule. And the
        # override must be an EXACT source match — 'recall_laf_config' is a
        # designed z_norm degradation a prefix would wrongly capture.
        from eval.longmem.build_corpus import _is_benign_build_error
        self.assertTrue(_is_benign_build_error(
            'recall_laf', 'laf returned no field scores (empty universe or '
            'scorer failure)', ''))
        self.assertTrue(_is_benign_build_error(
            'recall_laf_config', 'unknown z_norm in K-store config', ''))
        # A real scorer failure carries stack frames → red via the frame rule.
        self.assertFalse(_is_benign_build_error(
            'recall_laf', 'laf scoring failed',
            'Traceback...\n  File "recall_laf.py", line 1, in scores\n'))

    def test_require_embedder_refuses_dead_embedder(self):
        from unittest import mock
        from eval.longmem.fresh_brain import _require_embedder
        from servers import embedder
        with mock.patch.object(embedder, 'is_ready', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'embedder is DEAD'):
                _require_embedder()
        with mock.patch.object(embedder, 'is_ready', return_value=True):
            _require_embedder()  # ready → no raise


if __name__ == '__main__':
    unittest.main()
