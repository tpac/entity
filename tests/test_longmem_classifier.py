"""Tests for eval/longmem/classifier.py — failure bucket selection.

Covers:
- _extract_key_terms: tokenization discipline (digits, significant words, stopwords)
- _scan_brain_for_gold: ground-truth scan against a real BrainTestBase brain
- _bucket: all branches with synthetic scan + trace dicts (no LLM calls)
- Backward-compat: classify_failure integration smoke test

The _context_has_gold Haiku call is stubbed for bucket-logic tests — we only
want to exercise the deterministic decision tree, not hit the network.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from tests.eval_optional import require_eval  # noqa: E402
require_eval()  # D-8: eval/ is absent from the public tree

from eval.longmem import classifier


class TestExtractKeyTerms(unittest.TestCase):
    """Term extraction from gold answers."""

    def test_empty(self):
        self.assertEqual(classifier._extract_key_terms(""), [])
        self.assertEqual(classifier._extract_key_terms(None), [])

    def test_digits_extracted(self):
        terms = classifier._extract_key_terms("320 pages")
        self.assertIn("320", terms)
        self.assertIn("pages", terms)

    def test_stopwords_dropped(self):
        terms = classifier._extract_key_terms("The answer is that the book has 320 pages")
        # "the", "that", "has" should be dropped
        self.assertNotIn("the", terms)
        self.assertNotIn("that", terms)
        self.assertNotIn("has", terms)
        self.assertIn("320", terms)
        self.assertIn("answer", terms)
        self.assertIn("book", terms)
        self.assertIn("pages", terms)

    def test_short_words_filtered_unless_allcaps(self):
        terms = classifier._extract_key_terms("meet at 6 PM")
        # "at" too short and stop
        # "PM" is a 2-char uppercase — allowed
        self.assertIn("6", terms)
        self.assertIn("pm", terms)
        self.assertNotIn("at", terms)

    def test_proper_nouns_lowercased(self):
        terms = classifier._extract_key_terms("Hawaii accommodation was $400 per night")
        self.assertIn("hawaii", terms)
        self.assertIn("accommodation", terms)
        self.assertIn("night", terms)
        self.assertIn("400", terms)

    def test_dedup_preserves_order(self):
        terms = classifier._extract_key_terms("Memrise app uses memrise features")
        # 'memrise' seen twice, should appear once
        self.assertEqual(terms.count("memrise"), 1)

    def test_limit(self):
        text = " ".join([f"word{i}" for i in range(20)])
        terms = classifier._extract_key_terms(text, limit=5)
        self.assertLessEqual(len(terms), 5)

    def test_handles_non_ascii(self):
        # unicode names should still tokenize on latin words
        terms = classifier._extract_key_terms("I met Clara at the café")
        self.assertIn("clara", terms)
        # "café" has non-ascii — our regex is [A-Za-z'-]{3,} so it may miss it;
        # that's acceptable — test just verifies no crash and latin words still work.


class TestScanBrainForGold(BrainTestBase):
    """Ground-truth scan against a real brain with controlled content."""

    needs_embedder = False

    def _add(self, **kwargs):
        defaults = dict(type="fact", title="test")
        defaults.update(kwargs)
        return self.brain.remember(**defaults)["id"]

    def test_empty_brain_not_found(self):
        # Seed brain has nodes, but none about "vermilion"
        scan = classifier._scan_brain_for_gold(self.brain, "vermilion")
        self.assertFalse(scan["found"])

    def test_found_in_title(self):
        self._add(title="The Nightingale has 320 pages",
                  content="Book by Kristin Hannah.")
        scan = classifier._scan_brain_for_gold(self.brain, "320 pages")
        self.assertTrue(scan["found"])
        self.assertTrue(any("320" in str(m) or "pages" in str(m).lower()
                            for m in scan["matches"]))

    def test_found_in_content(self):
        self._add(title="Hawaii trip notes",
                  content="Accommodation cost $400 per night at the hotel.")
        scan = classifier._scan_brain_for_gold(self.brain, "$400 per night")
        self.assertTrue(scan["found"])

    def test_found_via_phrase_short(self):
        # Short answer — phrase pass should catch it even if term extraction is sparse.
        self._add(title="Gym routine", content="Ada goes to the gym at 6 PM daily")
        scan = classifier._scan_brain_for_gold(self.brain, "6 PM")
        self.assertTrue(scan["found"])

    def test_metadata_kv_scan_catches(self):
        # Node carries the fact only in metadata, not title/content.
        self._add(title="Unrelated title",
                  content="Short content with no match.",
                  situation="When debugging the dessy123 subsystem")
        scan = classifier._scan_brain_for_gold(self.brain, "dessy123")
        self.assertTrue(scan["found"])
        # matches should include the metadata source
        self.assertTrue(any("meta" in m["match_source"] for m in scan["matches"]))

    def test_and_across_terms(self):
        # Two nodes: each carries ONE of two gold terms but not both.
        # Scan with both terms should NOT match either (AND rule).
        self._add(title="The Nightingale was great", content="A wonderful story.")
        self._add(title="Tokyo hotel was expensive",
                  content="$400 per night accommodation.")
        # Gold "Nightingale 320" — 'nightingale' in one node, '320' in NEITHER.
        # Expected: not found.
        scan = classifier._scan_brain_for_gold(self.brain, "Nightingale 320")
        self.assertFalse(scan["found"])

    def test_archived_excluded(self):
        nid = self._add(title="Archived fact: $400 Hawaii",
                        content="Should not be found once archived.")
        # Archive the node
        self.brain.conn.execute("UPDATE nodes SET archived=1 WHERE id=?", (nid,))
        self.brain.conn.commit()
        scan = classifier._scan_brain_for_gold(self.brain, "Hawaii $400")
        # Even if term "hawaii" alone matches, AND with "400" requires content
        # AND archived=0 — so the archived node is excluded.
        self.assertFalse(scan["found"])


class TestBucketLogic(unittest.TestCase):
    """Pure decision-tree tests — no brain, no LLM, synthetic inputs."""

    def _trace(self, n_cand=5, n_sel=3, ctx_chars=500):
        return {
            "query": "test",
            "candidates": [{"id": f"n{i}", "title": "", "score": 0.5, "type": ""}
                           for i in range(n_cand)],
            "selected": [{"id": f"n{i}"} for i in range(n_sel)],
            "dropped": [],
            "context": "x" * ctx_chars,
        }

    # scan.found = False → ENCODE_MISS always
    def test_encode_miss_when_scan_empty_regardless_of_trace(self):
        scan = {"found": False, "matches": [], "terms_used": ["a"], "phrase_used": None}
        # even with a full trace, scan says "not in brain"
        b = classifier._bucket(scan, self._trace(), True, False)
        self.assertEqual(b, "ENCODE_MISS")

    def test_encode_miss_when_scan_empty_no_trace(self):
        scan = {"found": False, "matches": [], "terms_used": [], "phrase_used": None}
        self.assertEqual(classifier._bucket(scan, None, False, True), "ENCODE_MISS")

    # scan.found = True → branch on trace state
    def test_recall_miss_when_fact_in_brain_no_trace(self):
        scan = {"found": True, "matches": [{"node_id": "abc"}],
                "terms_used": ["a"], "phrase_used": None}
        self.assertEqual(classifier._bucket(scan, None, False, True), "RECALL_MISS")

    def test_recall_miss_when_zero_candidates(self):
        scan = {"found": True, "matches": [{"node_id": "abc"}],
                "terms_used": ["a"], "phrase_used": None}
        self.assertEqual(
            classifier._bucket(scan, self._trace(n_cand=0, n_sel=0, ctx_chars=0),
                               False, False),
            "RECALL_MISS")

    def test_surface_miss_when_candidates_but_no_selection(self):
        scan = {"found": True, "matches": [{"node_id": "abc"}],
                "terms_used": ["a"], "phrase_used": None}
        self.assertEqual(
            classifier._bucket(scan, self._trace(n_cand=5, n_sel=0, ctx_chars=0),
                               False, False),
            "SURFACE_MISS")

    def test_surface_miss_when_empty_context(self):
        scan = {"found": True, "matches": [{"node_id": "abc"}],
                "terms_used": ["a"], "phrase_used": None}
        self.assertEqual(
            classifier._bucket(scan, self._trace(n_cand=5, n_sel=3, ctx_chars=0),
                               False, False),
            "SURFACE_MISS")

    def test_answer_miss_when_context_has_gold(self):
        scan = {"found": True, "matches": [{"node_id": "abc"}],
                "terms_used": ["a"], "phrase_used": None}
        with patch.object(classifier, "_context_has_gold", return_value=True):
            b = classifier._bucket(scan, self._trace(), True, False,
                                   question="q", gold="g")
        self.assertEqual(b, "ANSWER_MISS")

    def test_recall_miss_when_context_missing_gold(self):
        # Fact IS in brain, context was delivered, but gold fact not in context.
        scan = {"found": True, "matches": [{"node_id": "abc"}],
                "terms_used": ["a"], "phrase_used": None}
        with patch.object(classifier, "_context_has_gold", return_value=False):
            b = classifier._bucket(scan, self._trace(), True, False,
                                   question="q", gold="g")
        self.assertEqual(b, "RECALL_MISS")


class TestClassifyFailureIntegration(BrainTestBase):
    """Integration smoke test: full classify_failure() with a real brain.

    Stubs out _context_has_gold and _reason to avoid LLM calls.
    """

    needs_embedder = False

    def test_encode_miss_end_to_end(self):
        # Brain has seeds but nothing about a made-up fact.
        with patch.object(classifier, "_context_has_gold", return_value=False), \
             patch.object(classifier, "_reason", return_value="stub"):
            result = classifier.classify_failure(
                brain=self.brain,
                question="What is the zithromax dosage?",
                gold="500mg zithromax daily",
                hypothesis="I don't know",
                query_session_id="nonexistent-session",
                has_context=False,
                abstained=True,
            )
        self.assertEqual(result["failure_bucket"], "ENCODE_MISS")
        self.assertIn("gold_in_brain", result["failure_evidence"])
        self.assertFalse(result["failure_evidence"]["gold_in_brain"]["found"])

    def test_recall_miss_end_to_end(self):
        # Seed a node carrying the gold fact.
        self.brain.remember(
            type="fact", title="Zithromax dose",
            content="500mg zithromax daily for 5 days was prescribed.")
        with patch.object(classifier, "_context_has_gold", return_value=False), \
             patch.object(classifier, "_reason", return_value="stub"):
            result = classifier.classify_failure(
                brain=self.brain,
                question="What dose?",
                gold="500mg zithromax",
                hypothesis="I don't know",
                query_session_id="nonexistent-session",
                has_context=False,
                abstained=True,
            )
        # Fact is in brain; no trace → RECALL_MISS
        self.assertEqual(result["failure_bucket"], "RECALL_MISS")
        self.assertTrue(result["failure_evidence"]["gold_in_brain"]["found"])


if __name__ == "__main__":
    unittest.main()
