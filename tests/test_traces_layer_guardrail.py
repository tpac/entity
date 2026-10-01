"""Traces-layer guardrail: no trace READS on TraceDAL outside brain_traces.py.

The traces consolidation (docs/TRACES-LAYER-DESIGN.md) set one
routing rule: reading traces through the API is a `brain.` method living in
brain_traces.py — only that file touches TraceDAL's read surface. Four operator
corrections and an audit's worth of bypasses came from this rule being enforced
socially; this test makes it structural.

WRITES are a different contract: every scale records its own O/K/Δ via
TraceDAL.append/append_batch — the validated write chokepoint — so writer
sites are allowed everywhere. Reads are what consolidate.

Sanctioned read exceptions (each named, each with a why):
- recall_laf.py: event_vector_rows — the LAF engine's vector-substrate
  pull (scoring internals, not an event read).
- embed_queue.py: find_unembedded — the embedding-reconciliation worker's
  substrate maintenance scan.
(daemon_hooks.py get_session_turns was the audit-#4 KNOWN DEBT entry —
retired 2026-08-07 by widening brain.get_conversation with
with_surfaced/exclude_trace_id/older_than. Do not add siblings.)

Detection is deliberately simple: `_trace_dal.<method>` tokens. Unknown method
names fail CLOSED (treated as reads) so a new TraceDAL method can't slip past
the gate unclassified.

Run: BRAIN_ALLOW_ANY_PYTHON=1 python3 -m pytest tests/test_traces_layer_guardrail.py -v
"""
import re
import pathlib
import unittest

SERVERS = pathlib.Path(__file__).resolve().parent.parent / 'servers'
# The layer itself + the DAL's own file are where the calls are supposed to live.
EXCLUDE = {'brain_traces.py', 'dal_logs.py'}

CALL_RE = re.compile(r'_trace_dal\s*\.\s*(\w+)')

# The write/lifecycle surface — allowed from any scale (each unit records its
# own traces through the validated append chokepoint).
WRITE_OK = {'append', 'append_batch', 'store_embeddings', 'set_identity'}

# (servers-relative path, method) → why. Exact allowlist; anything else fails.
READ_EXCEPTIONS = {
    # (present_streams/session_activity moved INTO brain_traces.py 2026-08-07
    # — the presence-door exception for brain.py is retired, not relocated.)
    ('recall_laf.py', 'event_vector_rows'): 'LAF episodic matrix vector substrate',
    ('embed_queue.py', 'find_unembedded'): 'embedding reconciliation scan',
}


def _violations():
    out = []
    for p in sorted(SERVERS.rglob('*.py')):
        if p.name in EXCLUDE:
            continue
        rel = str(p.relative_to(SERVERS))
        for i, line in enumerate(p.read_text().splitlines(), 1):
            for method in CALL_RE.findall(line):
                if method in WRITE_OK:
                    continue
                if (rel, method) in READ_EXCEPTIONS:
                    continue
                out.append('%s:%d — _trace_dal.%s' % (rel, i, method))
    return out


class TestTracesLayerGuardrail(unittest.TestCase):

    def test_no_trace_reads_outside_brain_traces(self):
        v = _violations()
        self.assertEqual(v, [], (
            '\nTraceDAL read(s) outside brain_traces.py:\n  %s\n'
            'Route the read through a brain.<method> in servers/brain_traces.py '
            '(the ONE traces functional layer — docs/TRACES-LAYER-DESIGN.md). '
            'If this is genuinely a new sanctioned exception, add it to '
            'READ_EXCEPTIONS with a why.' % '\n  '.join(v)))

    def test_exceptions_still_exist(self):
        # The allowlist ratchets both ways: an entry whose call site is gone
        # must be removed, so a stale allowance can't hide a reintroduction.
        live = set()
        for p in sorted(SERVERS.rglob('*.py')):
            if p.name in EXCLUDE:
                continue
            rel = str(p.relative_to(SERVERS))
            for method in CALL_RE.findall(p.read_text()):
                live.add((rel, method))
        stale = [k for k in READ_EXCEPTIONS if k not in live]
        self.assertEqual(stale, [], (
            'READ_EXCEPTIONS entries with no live call site — remove them: %r'
            % stale))


if __name__ == '__main__':
    unittest.main()
