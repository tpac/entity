"""Guardrail: no NEW host-shaped literal outside servers/host_contract.py.

The host-contract design's whole point is "a host's tool name or envelope tag
appearing anywhere but the contract is the violation" (docs/HOST-CONTRACT-DESIGN.md).
This test RATCHETS that the way test_raw_sql_guardrail ratchets raw DML: it
freezes today's per-file count of QUOTED host literals across servers/, hooks/
and dashboard/, and fails when a file grows one — forcing the author to route
the decision through the contract (a kind, a policy, a stamped status) or to
consciously bump the baseline with a why.

It ratchets BOTH ways: when a design step retires a site, the count drops and
the test fails too, demanding the baseline be lowered — so an allowance can't
silently hide a later re-introduction.

Detection is deliberately simple: a tool name or host key inside quotes
('Bash', "Edit", 'codex'), or a quoted string containing an envelope tag. Prose
mentions are not counted; a quoted mention inside a comment IS (retirement
bookkeeping — the baseline row says so). The tool-name and host-key sets are
DERIVED from the contract — adding a tool or a host widens the scan and the
baseline must be reconciled, which is the point.

KNOWN BLIND SPOTS, by construction: a name the contract does NOT declare scores
zero here (the detector for a genuinely new host name is the write door
stamping kind_status 'unknown' into the errors table — design step 1), and a
name embedded in a longer string ('Bash: %s') is not an exact quoted name (the
summary heads are held to the contract by test_host_contract.TestHookMirrors).

Run: ./dev pytest tests/test_host_shape_guardrail.py -v
"""
import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from servers.host_contract import HOST_CONTRACT  # noqa: E402
from servers.trace_contract import WAKE_ENVELOPE_MARKER  # noqa: E402

# hooks/adapters/ is NOT scanned: those modules are a host's own setup and
# onboarding BEHAVIOUR, keyed by host by design (brain node 368b15af) — the
# adapter is the host's code, not host shape leaking into shared code.
SCAN_ROOTS = ('servers', 'hooks/scripts', 'dashboard')
# Matched against the repo-relative path: retired code and vendored runtimes.
EXCLUDE_PARTS = ('archive/', 'venv/')
# The owner.
EXCLUDE_FILES = {'servers/host_contract.py'}

TOOL_NAMES = sorted({n for e in HOST_CONTRACT.values() for n in e['tools']}
                    | {a for e in HOST_CONTRACT.values() for a in e['matcher_aliases']})
HOST_KEYS = sorted(HOST_CONTRACT)
# Envelope tags the hosts inject into prompts. Hardcoded here until the
# contract's `envelopes` tables are populated (design step 3), when this
# derives from them the way TOOL_NAMES derives from `tools`.
ENVELOPE_TAGS = (
    WAKE_ENVELOPE_MARKER, '<system-reminder>', '<send_user_message_question_reply>',
    '# Files mentioned by the user:', '# Files pasted by the user:', '## My request',
    '# Response annotations:', '# Diff comments:', '# Browser comments:',
    '# Selected text:', '# Failing PR checks:',
)

TOOL_RE = re.compile(r"""(['"])(%s)\1""" % '|'.join(map(re.escape, TOOL_NAMES)))
KEY_RE = re.compile(r"""(['"])(%s)\1""" % '|'.join(map(re.escape, HOST_KEYS)))
TAG_RE = re.compile(r"""(['"])[^'"\n]*(%s)[^'"\n]*\1""" % '|'.join(map(re.escape, ENVELOPE_TAGS)))

# Frozen baseline (2026-09-08): quoted host-literal sites per repo-relative
# path, each with the design step that retires it. Lower a number when a step
# lands; bump (with a why) only for a genuine new exception.
ALLOWED = {
    'servers/scales/s1/encoder_actions.py': 4,  # frozen legacy summary-name map (D10); normalized predicates have no host names
    'servers/trace_contract.py': 1,             # step 3: WAKE_ENVELOPE_MARKER moves into the contract's envelopes
    'hooks/scripts/post_tool_trace.py': 11,     # summary's 10 + step 1's raw patch capture (1); both need tool_input's per-tool fields until canonical arguments (deferred), mirrored by TestHookMirrors
    'hooks/scripts/pre_response_recall.py': 1,  # step 3: wake routing reads the contract's envelope table
    'dashboard/queries/_meta.py': 1,            # step 3: the dashboard's one mirror of WAKE_ENVELOPE_MARKER (it may not import servers/), read by stats.py; test_dashboard_disconnection pins it
}


def _scan():
    counts = {}
    for root in SCAN_ROOTS:
        for p in sorted((ROOT / root).rglob('*.py')):
            rel = str(p.relative_to(ROOT))
            if rel in EXCLUDE_FILES or any(part in rel for part in EXCLUDE_PARTS):
                continue
            text = p.read_text(errors='replace')
            n = len(TOOL_RE.findall(text)) + len(KEY_RE.findall(text)) + len(TAG_RE.findall(text))
            if n:
                counts[rel] = n
    return counts


class TestHostShapeGuardrail(unittest.TestCase):
    def test_no_new_host_literals_outside_the_contract(self):
        counts = _scan()
        grown = {f: (n, ALLOWED.get(f, 0)) for f, n in counts.items() if n > ALLOWED.get(f, 0)}
        self.assertFalse(
            grown,
            'host tool names / envelope tags appeared outside servers/host_contract.py '
            '(file: found vs allowed): %s — route the decision through the contract '
            '(classify_tool / envelope_policy / a stamped kind), or bump ALLOWED with a why'
            % grown)

    def test_baseline_is_lowered_when_sites_retire(self):
        counts = _scan()
        shrunk = {f: (counts.get(f, 0), n) for f, n in ALLOWED.items() if counts.get(f, 0) < n}
        self.assertFalse(
            shrunk,
            'fewer host literals than ALLOWED (file: found vs allowed): %s — lower the '
            'baseline so the allowance cannot hide a later re-introduction' % shrunk)

    def test_owner_is_excluded_and_present(self):
        self.assertTrue((ROOT / 'servers' / 'host_contract.py').exists())
        self.assertNotIn('servers/host_contract.py', _scan())

    def test_detector_has_teeth(self):
        text = ("x = 'Bash'\n"                       # quoted tool name: counted
                "# Bash is the shell tool\n"          # prose: not counted
                'y = "<task-notification>%"\n'        # quoted tag inside a longer string: counted
                "if host == 'codex':\n"               # quoted host key: counted
                "z = 'shell'\n")                      # a KIND, not a tool name: not counted
        self.assertEqual(len(TOOL_RE.findall(text)), 1)
        self.assertEqual(len(TAG_RE.findall(text)), 1)
        self.assertEqual(len(KEY_RE.findall(text)), 1)


if __name__ == '__main__':
    unittest.main()
