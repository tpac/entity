"""Grep-pin: brain.record_payload is the ONLY capture writer.

docs/TRACE-MODES-DESIGN.md rollout step 2 retired the ad-hoc capture
mechanisms — the /tmp prompt/judge dumps and the BRAIN_PROMPT_CAPTURE_DIR
env machinery. This pin keeps them retired: any new code writing (or
reading) capture through those seams fails here and must route through
record_payload / read_payload (in-process) or the sanctioned direct
payload readers (dashboard/db.py, eval fresh_brain.capture_files_for).

Operational STATE files are not capture and are deliberately out of pin
scope: `brain-{session}-current-stop.txt` survives.

ALLOWLIST: eval/longmem/connect_ab.py READS a frozen pre-migration
capture corpus (pooled brain-encoding-prompt-*.json) as replay input —
the artifacts predate record_payload and can't be rerouted; delete when
that corpus is regenerated in payload form.
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (pattern, why it's forbidden)
FORBIDDEN = [
    (re.compile(r"environ(\.get)?\s*[\(\[]\s*['\"]BRAIN_PROMPT_CAPTURE"),
     "BRAIN_PROMPT_CAPTURE env machinery was retired — capture rides "
     "record_round_fn → brain.round_recorder"),
    (re.compile(r"brain-encoding-prompt-"),
     "encoder prompt tmp dumps were retired — brain.record_payload"
     "(chain, 'prompt') + the O trace's pointer ref_id"),
    (re.compile(r"brain-consolidation-prompt-"),
     "consolidation prompt tmp dumps were retired — "
     "brain.record_payload(chain, 'prompt', seq=batch)"),
    (re.compile(r"brain-judge-result-"),
     "judge tmp dumps were retired — brain.record_payload(chain, 'judge')"),
]

# {(relpath, pattern.pattern)} — see module docstring.
ALLOWLIST = {
    ("eval/longmem/connect_ab.py", r"brain-encoding-prompt-"),
}

SCAN_DIRS = ("servers", "dashboard", "hooks", "eval", "scripts")
SCAN_EXTS = (".py", ".sh", ".js")
SKIP_DIRS = ("__pycache__", "node_modules", ".venv", "venv", "data")


def _scan_files():
    for base in SCAN_DIRS:
        root = os.path.join(REPO, base)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if fn.endswith(SCAN_EXTS):
                    yield os.path.join(dirpath, fn)


class TestCaptureGrepPin(unittest.TestCase):

    def test_no_capture_writer_outside_record_payload(self):
        offenders = []
        for path in _scan_files():
            rel = os.path.relpath(path, REPO)
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            for pattern, why in FORBIDDEN:
                if (rel, pattern.pattern) in ALLOWLIST:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if pattern.search(line):
                        offenders.append("%s:%d [%s] %s\n    → %s"
                                         % (rel, i, pattern.pattern,
                                            line.strip()[:120], why))
        self.assertFalse(
            offenders,
            "retired capture seams reintroduced:\n" + "\n".join(offenders))

    def test_allowlist_entries_still_exist(self):
        """A stale allowlist entry means the legacy branch died — delete the
        entry (and this reminder keeps the allowlist honest).

        An entry whose whole top-level directory is absent is not stale: D-8
        keeps `eval/` out of the public tree, so there the eval entry names a
        branch that still exists upstream. Only check what this tree carries.
        """
        for rel, pat in ALLOWLIST:
            path = os.path.join(REPO, rel)
            if not os.path.isdir(os.path.join(REPO, rel.split("/")[0])):
                continue
            self.assertTrue(os.path.exists(path),
                            "allowlisted file gone: %s — drop the entry" % rel)
            with open(path, encoding="utf-8", errors="replace") as f:
                self.assertIn(pat.replace("\\", ""), f.read(),
                              "allowlisted pattern no longer in %s — drop "
                              "the entry" % rel)


if __name__ == "__main__":
    unittest.main()
