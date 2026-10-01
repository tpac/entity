"""Contract tests for the Frozen Corpus helpers (eval/longmem/corpus.py).

Pure functions — no Brain, no LLM. Locks the content-addressing and S2-delta
aggregation that the two-stage eval (build_corpus → sweep) depends on.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.eval_optional import require_eval  # noqa: E402
require_eval()  # D-8: eval/ is absent from the public tree

from eval.longmem.corpus import (
    corpus_config_hash, source_token, summarize_s2_deltas, merge_s2_totals,
    counterpart_name, counterpart_token, frame_user_msg,
    COUNTERPART_NAMES, RESERVED_PERSONA_NAMES,
)


class TestCorpusConfigHash(unittest.TestCase):
    def test_deterministic(self):
        cfg = {"s1e": "active", "ingest_surface": "active", "s2_every_n": 2,
               "oracle": "longmemeval_oracle.json", "qids": ["a", "b"]}
        self.assertEqual(corpus_config_hash(cfg), corpus_config_hash(dict(cfg)))

    def test_key_order_independent(self):
        a = {"s1e": "active", "s2_every_n": 2, "qids": ["a"]}
        b = {"qids": ["a"], "s2_every_n": 2, "s1e": "active"}
        self.assertEqual(corpus_config_hash(a), corpus_config_hash(b),
                         "dict key order must not change the hash (sort_keys)")

    def test_sensitive_to_inputs(self):
        base = {"s1e": "active", "s2_every_n": 2, "qids": ["a", "b"]}
        self.assertNotEqual(corpus_config_hash(base),
                            corpus_config_hash({**base, "s1e": "file:deadbeef"}))
        self.assertNotEqual(corpus_config_hash(base),
                            corpus_config_hash({**base, "s2_every_n": 3}))
        self.assertNotEqual(corpus_config_hash(base),
                            corpus_config_hash({**base, "qids": ["a", "c"]}))


class TestVariantAddressing(unittest.TestCase):
    """Locks the frozen variant pins: the retired surface/spread variants
    leave one pipeline, so the pins ignore the shell, keep every corpus hash,
    and the sweep leg still refuses a corpus stamped with other pins."""

    def test_pins_ignore_the_shell(self):
        from eval.longmem.corpus import (require_variant_pins,
                                         _ADDRESS_BASELINE_VARIANTS)
        saved = {k: os.environ.pop(k, None) for k in
                 ("BRAIN_SURFACE_VARIANT", "BRAIN_RECALL_VARIANT")}
        try:
            self.assertEqual(require_variant_pins(), _ADDRESS_BASELINE_VARIANTS)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_baseline_pins_leave_hash_stable(self):
        from eval.longmem.corpus import require_variant_pins, address_variants
        cfg = {"s1e": "active", "qids": ["a"]}
        before = corpus_config_hash(cfg)
        address_variants(cfg, require_variant_pins())
        self.assertEqual(before, corpus_config_hash(cfg),
                         "baseline variants must not invalidate existing corpora")

    def test_sweep_refuses_mismatched_stamp(self):
        from eval.longmem.corpus import require_variant_pins, check_variant_pins
        manifest = {"variant_pins": {"surface_variant": "v5_agentic",
                                     "recall_variant": "cluster"}}
        with self.assertRaises(SystemExit):
            check_variant_pins(manifest, require_variant_pins(), "sweep")

    def test_prestamp_manifest_passes(self):
        from eval.longmem.corpus import require_variant_pins, check_variant_pins
        check_variant_pins({}, require_variant_pins(), "sweep")


class TestCounterpartIdentity(unittest.TestCase):
    """Locks the named-counterpart fidelity fix (2026-09-19): a per-item
    deterministic user name, distinct from the operator/seed/prompt-exemplar
    vocabulary, joining the corpus content address so post-fix builds never
    cache-hit a pre-fix anonymous corpus."""

    def test_deterministic_and_from_list(self):
        name = counterpart_name("2698e78f_abs")
        self.assertEqual(name, counterpart_name("2698e78f_abs"))
        self.assertIn(name, COUNTERPART_NAMES)

    def test_names_vary_across_qids(self):
        names = {counterpart_name("qid-%d" % i) for i in range(50)}
        self.assertGreater(len(names), 1, "one name for every qid means the "
                           "hash pick is broken")

    def test_reserved_vocabulary_excluded(self):
        # Persona exclusions are a frozen property of the name list
        # (RESERVED_PERSONA_NAMES beside it); agent identity comes from
        # daemon_config; the operator's name comes from the shell env
        # captured BEFORE any eval code could pin a synthetic name
        # (fresh_brain._SHELL_OPERATOR) — never as a literal here: tests/
        # ships in the public tree and the export's personal-information
        # scrub (gate B) rejects the operator's real name. Where the shell
        # carries no operator (CI, public tree) that half is untestable by
        # construction; personas and agent names are always checked.
        from servers.daemon_config import get_agent_name, DEFAULT_AGENT_NAME
        from eval.longmem.fresh_brain import _SHELL_OPERATOR
        reserved = set(RESERVED_PERSONA_NAMES) | {get_agent_name(),
                                                  DEFAULT_AGENT_NAME}
        if _SHELL_OPERATOR:
            reserved.add(_SHELL_OPERATOR.strip())
        self.assertTrue(reserved.isdisjoint(COUNTERPART_NAMES),
                        "counterpart names collide with reserved identity "
                        "vocabulary: %s" % (reserved & set(COUNTERPART_NAMES)))

    def test_token_addresses_the_name_list(self):
        tok = counterpart_token()
        self.assertEqual(tok, counterpart_token())
        self.assertTrue(tok.startswith("names_sha1:"))
        cfg = {"s1e": "active", "qids": ["a"]}
        self.assertNotEqual(
            corpus_config_hash(cfg),
            corpus_config_hash({**cfg, "counterpart_identity": tok}),
            "the counterpart key must move the corpus hash — a post-fix "
            "build must never cache-hit a pre-fix anonymous corpus")


class TestFrameUserMsg(unittest.TestCase):
    """Locks the replay's timeline framing prefix — including that
    counterpart=None reproduces the pre-fix framing byte for byte."""

    def test_prefix_with_name_and_date(self):
        framed = frame_user_msg("hi", "2023/05/20 (Sat) 19:23", "Ingrid")
        self.assertEqual(
            framed,
            "[Current date: 2023/05/20 (Sat) 19:23 | Operator: Ingrid]\n\nhi")
        # Both prefix parsers must survive the tail — the replay clock reads
        # the date through _REPLAY_DATE_RE, and the eval report lanes strip
        # the WHOLE bracket with the shared '^\[Current date:[^\]]*\]' shape.
        import re
        from servers.clock import _REPLAY_DATE_RE
        m = _REPLAY_DATE_RE.search(framed)
        self.assertIsNotNone(m, "clock regex must still parse the framed date")
        self.assertEqual(m.group(1), "2023/05/20")
        stripped = re.sub(r'^\[Current date:[^\]]*\]\s*', '', framed)
        self.assertEqual(stripped, "hi",
                         "the report-lane stripper must swallow the operator "
                         "tail with the same bracket")

    def test_name_only(self):
        self.assertEqual(frame_user_msg("hi", None, "Ingrid"),
                         "[Operator: Ingrid]\n\nhi")

    def test_none_counterpart_is_prefix_exact(self):
        self.assertEqual(
            frame_user_msg("hi", "2023/05/20 (Sat) 19:23", None),
            "[Current date: 2023/05/20 (Sat) 19:23]\n\nhi")
        self.assertEqual(frame_user_msg("hi", None, None), "hi")


class TestSourceToken(unittest.TestCase):
    def test_active(self):
        self.assertEqual(source_token("active"), "active")
        self.assertEqual(source_token(None), "active")

    def test_missing_path(self):
        tok = source_token("/no/such/prompt_file.txt")
        self.assertTrue(tok.startswith("missing:"))

    def test_file_content_hash(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("prompt body v1")
            p = f.name
        try:
            tok = source_token(p)
            self.assertTrue(tok.startswith("file:"))
            # Same content → same token; different content → different token.
            self.assertEqual(tok, source_token(p))
            with open(p, "w") as f:
                f.write("prompt body v2")
            self.assertNotEqual(tok, source_token(p))
        finally:
            os.unlink(p)


class TestSummarizeS2Deltas(unittest.TestCase):
    def test_aggregates_fires_work_actions(self):
        # Deltas are run_s2() returns: units nested, elapsed_ms alongside.
        deltas = [
            {"units": {"community_detection": {"actions": 2},
                       "healer": {"actions": 0}}, "elapsed_ms": 10},
            {"units": {"community_detection": {"actions": 1},
                       "healer": {"skipped": "no gaps"}}, "elapsed_ms": 5},
        ]
        s = summarize_s2_deltas(deltas)
        self.assertEqual(s["community_detection"]["fires"], 2)
        self.assertEqual(s["community_detection"]["did_work"], 2)
        self.assertEqual(s["community_detection"]["actions"], 3)
        self.assertEqual(s["healer"]["fires"], 2)
        self.assertEqual(s["healer"]["did_work"], 0)
        self.assertEqual(s["healer"]["skipped"], 1)

    def test_counts_errors_and_samples(self):
        deltas = [{"units": {"consolidation": {"error": "cannot start a transaction"}}}]
        s = summarize_s2_deltas(deltas)
        self.assertEqual(s["consolidation"]["errors"], 1)
        self.assertEqual(len(s["consolidation"]["sample_errors"]), 1)
        # An errored fire is neither did_work nor a successful action.
        self.assertEqual(s["consolidation"]["did_work"], 0)
        self.assertEqual(s["consolidation"]["actions"], 0)

    def test_ignores_elapsed_and_nondict(self):
        # elapsed_ms and a skipped cycle carry no units; non-dict unit results drop.
        s = summarize_s2_deltas([
            {"units": {}, "skipped": "already running"},
            {"units": {"x": "notadict"}, "elapsed_ms": 99},
        ])
        self.assertEqual(s, {})

    def test_empty(self):
        self.assertEqual(summarize_s2_deltas([]), {})
        self.assertEqual(summarize_s2_deltas(None), {})


class TestMergeS2Totals(unittest.TestCase):
    def test_sums_across_items(self):
        items = [
            {"s2_delta": {"healer": {"fires": 2, "did_work": 1, "actions": 1,
                                     "errors": 0, "skipped": 1}}},
            {"s2_delta": {"healer": {"fires": 3, "did_work": 2, "actions": 4,
                                     "errors": 1, "skipped": 0}}},
        ]
        t = merge_s2_totals(items)
        self.assertEqual(t["healer"]["fires"], 5)
        self.assertEqual(t["healer"]["actions"], 5)
        self.assertEqual(t["healer"]["errors"], 1)


if __name__ == "__main__":
    unittest.main()
