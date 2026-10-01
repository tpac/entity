"""Guardrail: no NEW wall-clock ELAPSED-INTERVAL gates in `servers/`.

The rule this enforces is stated in `servers/clock.py` ("Which clock for an
ELAPSED INTERVAL — the answer is three-valued"). Short version:

  A. semantic / persisted time              -> wall clock
  B. elapsed real time that counts suspend  -> wall clock
  C. elapsed PROCESS time (ticks, cooldowns,
     rate limits, deadlines, TTLs)          -> time.monotonic()

A category-C gate left on the wall clock fails SILENTLY and invisibly: a
backward step (NTP correction, manual set) makes `now - stamp` negative, so the
gate never fires again until the clock catches up, while everything looks
healthy. That is not hypothetical — commit e03e26b fixed three such sites in
the daemon poll loop, and the author of that commit, who had just named the bug
class, still missed two more (`daemon_server.py` uptime reset,
`db_maintenance._tick`). A stated rule alone did not hold; this ratchet is what
holds it.

WHY A SEPARATE FILE from `test_clock_contract_sync.py`: that test's
PROTECTED_DIRS is `servers/scales` and is ASSERTED EQUAL to
`test_time_window_contract.CTX_PROTECTED_DIRS`. Widening it to cover `servers/`
would silently widen the semantic-time rule too and break that equality. Two
different rules, two files.

CALIBRATION — this class is real but is NOT currently firing. Two independent
measurements on 2026-09-14 (daemon.log timestamps, ~71 day-boundaries;
trace_events by rowid, 279,298 rows over 5 months) found ZERO backward
wall-clock steps; macOS normally slews rather than steps. So the baseline below
is deliberately NOT a to-do list to burn down tonight — converting a working
site carries more risk than the unfired bug. The point of this test is that the
count does not GROW. See brain node 816a5627.

Detection is deliberately simple, in the `test_raw_sql_guardrail.py` spirit: a
comparison whose left side subtracts from a `time.time()`-derived value, on one
line. Known blind spots (acceptable, not bugs in the test):
  - two-line forms (`idle = time.time() - x` then `if idle > y:` later) — e.g.
    the daemon's idle timeout, which is category B and correct anyway;
  - values passed as arguments rather than compared (`_keepalive_due(now - x,
    ...)`) — the keepalive's deferred site;
  - a name assigned from `time.time()` in one function and compared in another
    that does not stamp it itself — name matching is scoped per `def` (plus
    module-level stamps), because file-wide matching made every monotonic gate
    in a file that stamps `now = time.time()` anywhere read as wall-clock debt.

Escape hatch: put `# wall-clock-ok: <why>` on the line for a genuine A or B
site. Use it to STATE A CATEGORY, never to silence a category-C gate.

Run: BRAIN_ALLOW_ANY_PYTHON=1 python3 -m pytest tests/test_interval_clock_guardrail.py -v
"""
import re
import pathlib
import unittest

SERVERS = pathlib.Path(__file__).resolve().parent.parent / 'servers'

_ASSIGN = re.compile(r'(?:self\.)?(\w+)\s*=\s*_?time\.time\(\)')
_CMP = r'(?:[<>]=?)'
_INLINE = re.compile(r'time\.time\(\)\s*-\s*[^\n]*?' + _CMP)
_OK = re.compile(r'#\s*wall-clock-ok:')
_DEF = re.compile(r'^\s*(?:async\s+)?def\s+\w+')

# Frozen baseline (2026-09-14): wall-clock elapsed-interval gates per
# servers-relative path. Every entry below is category C (should be monotonic)
# unless its note says otherwise — they are KNOWN, MEASURED NOT-FIRING debt, not
# an emergency. Lower a number when a site is converted; raising one requires a
# genuine category-A/B justification here AND a `# wall-clock-ok:` at the site.
ALLOWED = {
    # brain.py: removed 2026-09-21. Its two entries ("error-dedup window;
    # db-size-check throttle") were never wall-clock — both read
    # time.monotonic() and always had. They were an artifact of matching the
    # name `now` file-wide; see _blocks.
    'brain_recall.py': 1,     # C: recall cache TTL
    'brain_remember.py': 1,   # C: lock-confirm TTL
    'daemon_client.py': 2,    # C: recovery window + cooldown
    'daemon_server.py': 1,    # C: HEALTHY_UPTIME_RESET_S streak reset (arguably B —
                              #    should a suspend count as healthy serving? decide
                              #    before converting; see DAEMON-TIMING-ARCH-PLAN Step 4)
    'db_maintenance.py': 4,   # C: checkpoint / optimize / quick_check / registered tasks.
                              #    Highest-leverage of the set — it gates BACKUPS, and
                              #    its `last_*_at = 0.0` "due immediately" seed makes the
                              #    conversion non-mechanical (ARCH-PLAN Step 4).
    'embed_queue.py': 2,      # C: coverage-sweep throttle + max-staleness
    'scales/runner.py': 1,    # C: LLM agentic-loop deadline
}


def _blocks(lines):
    """(start_index, [lines]) per `def`, plus the module-level remainder.

    A name is only wall-clock-derived where it was ASSIGNED from time.time().
    Scanning the whole file for the name instead flags every same-named local
    in the file — and `now` is the most reused local there is, so one
    `now = time.time()` anywhere made every `now = time.monotonic()` gate in
    that file read as debt. brain.py's two frozen entries were both that: the
    error-dedup window and the db-size throttle have been on time.monotonic()
    since they were written.
    """
    out, cur, start = [], [], 1
    for i, line in enumerate(lines, 1):
        if _DEF.match(line):
            out.append((start, cur))
            cur, start = [], i
        cur.append(line)
    out.append((start, cur))
    return out


def _gates_in(text):
    """Line numbers of wall-clock elapsed-interval comparisons."""
    lines = text.splitlines()
    blocks = _blocks(lines)
    # Module-level stamps are visible inside every function; per-function
    # stamps are not visible outside their own.
    module_names = set(_ASSIGN.findall('\n'.join(
        ln for ln in blocks[0][1]))) if blocks else set()
    found = []
    for start, block in blocks:
        names = set(_ASSIGN.findall('\n'.join(block))) | module_names
        by_name = [re.compile(r'\b' + re.escape(n) + r'\s*-\s*[\w\.\[\]\'"()]+\s*' + _CMP)
                   for n in names]
        for offset, line in enumerate(block):
            if _OK.search(line):
                continue
            if _INLINE.search(line) or any(r.search(line) for r in by_name):
                found.append(start + offset)
    return sorted(found)


def _scan():
    counts = {}
    for p in sorted(SERVERS.rglob('*.py')):
        hits = _gates_in(p.read_text())
        if hits:
            counts[str(p.relative_to(SERVERS))] = len(hits)
    return counts


class TestIntervalClockGuardrail(unittest.TestCase):

    def test_no_new_wall_clock_interval_gates(self):
        counts = _scan()
        grown = {f: (n, ALLOWED.get(f, 0))
                 for f, n in counts.items() if n > ALLOWED.get(f, 0)}
        self.assertFalse(grown, (
            "New wall-clock elapsed-interval gate(s): %s (file: found vs allowed).\n"
            "An elapsed interval that must be immune to clock steps (a tick, "
            "cooldown, rate limit, TTL or deadline) belongs on time.monotonic() "
            "— see the three-way rule in servers/clock.py. If this site is "
            "genuinely category A (semantic/persisted time) or B (elapsed real "
            "time that must count suspend, like the idle timeout), mark the line "
            "`# wall-clock-ok: <why>`. Do NOT raise the baseline to silence a "
            "category-C gate." % grown))

    def test_baseline_is_not_stale(self):
        """Ratchets downward too — a converted site must lower the baseline, so
        the allowance can't silently hide a later re-introduction."""
        counts = _scan()
        stale = {f: (counts.get(f, 0), n)
                 for f, n in ALLOWED.items() if counts.get(f, 0) < n}
        self.assertFalse(stale, (
            "Baseline too high for %s (file: found vs allowed) — a site was "
            "converted or removed. Lower the number so the allowance can't hide "
            "a re-introduction." % stale))

    def test_rule_is_documented_where_authors_look(self):
        """The ratchet points at a rule; the rule has to exist. If clock.py's
        three-way section is renamed or dropped, this fails rather than leaving
        the error message pointing at nothing."""
        clock = (SERVERS / 'clock.py').read_text()
        self.assertIn('three-valued', clock,
                      "servers/clock.py lost the three-way elapsed-interval rule "
                      "this guardrail cites in its failure message.")
        for cat in ('A. SEMANTIC', 'B. ELAPSED REAL TIME', 'C. ELAPSED PROCESS TIME'):
            self.assertIn(cat, clock, "clock.py lost category %r" % cat)

    # ── the detector itself ────────────────────────────────────────────────
    # Without these, a change that quietly stops detecting makes every file
    # read as clean and the ratchet reports success while enforcing nothing.

    def test_detector_catches_wall_clock_gates(self):
        caught = """
import time
class A:
    def tick(self):
        now = time.time()
        if now - self.last > 60:
            pass
    def inline(self):
        if time.time() - self.last > 60:
            pass
"""
        self.assertEqual(len(_gates_in(caught)), 2,
                         "detector stopped catching wall-clock gates")

    def test_detector_sees_module_level_stamps(self):
        modlevel = """
import time
START = time.time()
def due():
    if START - other > 5:
        pass
"""
        self.assertEqual(len(_gates_in(modlevel)), 1,
                         "module-level time.time() stamp no longer reaches functions")

    def test_detector_ignores_monotonic_in_another_function(self):
        """The brain.py false positive: one wall-clock `now` must not convict
        a monotonic `now` in a different function."""
        mixed = """
import time
class A:
    def wall(self):
        now = time.time()
        return now
    def mono(self):
        now = time.monotonic()
        if now - self.last < 60:
            pass
"""
        self.assertEqual(_gates_in(mixed), [],
                         "monotonic gate convicted by an unrelated wall-clock name")

    def test_detector_honours_the_escape_hatch(self):
        annotated = """
import time
def age(self):
    now = time.time()
    if now - self.created > 86400:  # wall-clock-ok: semantic age of a stored row
        pass
"""
        self.assertEqual(_gates_in(annotated), [])


if __name__ == '__main__':
    unittest.main()
