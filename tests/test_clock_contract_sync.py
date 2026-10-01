"""Clock contract: no direct datetime.now() / date.today() on the grain axis.

Why this file exists
--------------------
A grain-axis caller that silently falls back to datetime.now() when no
conversation date is passed resolves "today/yesterday" in eval replays of
historical conversations against the real wall-clock instead of the
conversation date — corrupting every date the encoder derives.

servers/clock.py is the single source of truth for "now". This test keeps
it that way by failing if anyone adds a direct wall-clock call anywhere
under servers/scales (s1, s2, and any future grain).

Allowed exemptions:
  - servers/clock.py itself (the implementation)
  - Trace timestamps, perf timers, "created_at" stamps where wall-clock
    IS the right answer (these patterns are excluded by suffix matching
    on the surrounding line — e.g. lines doing `created_at = ...isoformat()`)
  - DAL code (system bookkeeping is wall-clock by design)

The intent: anything that needs to know "what date is THIS CONVERSATION
happening" must route through brain_now / conversation_now.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent

# Directories where wall-clock calls are forbidden (semantic-time code).
# The whole grain axis, as a PREFIX, so a future grain is covered without
# anyone remembering to list it. What belongs on that axis, and why a
# real-elapsed package must not join it, is stated once in
# servers/scales/__init__.py — don't restate it here.
#
# Must equal test_time_window_contract.CTX_PROTECTED_DIRS; the two are halves
# of one rule. test_prefix_zones_agree below enforces that rather than trusting
# this comment.
PROTECTED_DIRS = [
    'servers/scales',
]

# Files within protected dirs where wall-clock is legitimate (logs/traces).
# Keep this list TIGHT — every entry is a foothold for the bug to come back.
EXEMPT_FILES = {
    # None at the moment — all grain-axis semantic-time goes through clock.
}

# Patterns that flag a direct wall-clock call.
FORBIDDEN_PATTERNS = [
    re.compile(r'\bdatetime\.now\s*\('),
    re.compile(r'\b_dt\.datetime\.now\s*\('),
    re.compile(r'\bdate\.today\s*\('),
    re.compile(r'\b_dt\.date\.today\s*\('),
]

# Lines that ARE allowed even in protected dirs — bookkeeping wall-clock
# uses where conversation_now() would be semantically wrong.
LEGITIMATE_USES = [
    # Trace event created_at — these are system clocks, not conversation time
    re.compile(r'created_at\s*[:=].*now'),
    re.compile(r'set_at\s*[:=].*now'),
    re.compile(r'updated_at\s*[:=].*now'),
    re.compile(r'timestamp\s*[:=].*now'),
    # Latency / perf
    re.compile(r't0\s*=.*now'),
    re.compile(r'started_at\s*[:=].*now'),
    # Inside isoformat() chains for trace metadata are OK if marked
    re.compile(r'#\s*clock-ok'),
]


def _is_legit(line: str) -> bool:
    return any(p.search(line) for p in LEGITIMATE_USES)


def _scan_file(p: Path):
    """Yield (line_no, line) tuples that violate the contract."""
    violations = []
    try:
        text = p.read_text()
    except Exception:
        return violations
    for i, line in enumerate(text.splitlines(), start=1):
        if any(p.search(line) for p in FORBIDDEN_PATTERNS):
            if _is_legit(line):
                continue
            violations.append((i, line.strip()))
    return violations


def test_no_direct_wallclock_on_grain_axis():
    """Scan the grain axis for direct datetime.now() / date.today() calls.

    All semantic-time calls should go through servers/clock.py
    (brain_now or conversation_now). Bookkeeping uses are tagged as
    LEGITIMATE_USES above.
    """
    failures = []
    for prot in PROTECTED_DIRS:
        for root, dirs, files in os.walk(ROOT / prot):
            # Skip __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('__')]
            for f in files:
                if not f.endswith('.py'):
                    continue
                rel = Path(root, f).relative_to(ROOT)
                if str(rel) in EXEMPT_FILES:
                    continue
                violations = _scan_file(Path(root, f))
                for lineno, line in violations:
                    failures.append(f'{rel}:{lineno}: {line}')
    assert not failures, (
        'Clock contract violation: direct wall-clock call under '
        'servers/scales (the grain axis).\n'
        'Route through servers/clock.py (brain_now or conversation_now).\n'
        'If genuinely bookkeeping (trace timestamp etc), tag the line with '
        '"# clock-ok".\n'
        'If your whole package is real-elapsed by design — it addresses a '
        'live stream rather than running an integrate loop — it belongs in '
        'servers/channels/, the way self_channel and thalamus do. Move it; '
        'do NOT grow LEGITIMATE_USES, which loosens the rule for all of '
        'servers/scales.\n\n'
        + '\n'.join(failures)
    )


def test_clock_functions_return_tz_aware():
    """brain_now and conversation_now must always return tz-aware datetimes."""
    from servers.clock import brain_now, conversation_now
    assert brain_now().tzinfo is not None
    assert conversation_now().tzinfo is not None
    assert conversation_now(messages=[]).tzinfo is not None


def test_conversation_now_parses_replay_prefix():
    """The eval replay's '[Current date: YYYY/MM/DD ...]' prefix must be
    recognized — this is the load-bearing case for the eval path."""
    from servers.clock import conversation_now
    import datetime as dt
    msgs = [
        {'role': 'user',
         'content': '[Current date: 2023/03/19 (Sun) 07:16]\n\nI did a 5K today'},
    ]
    assert conversation_now(messages=msgs).date() == dt.date(2023, 3, 19)


def test_conversation_now_parses_replay_prefix_with_operator_tail():
    """The replay's named-counterpart framing shares the date bracket
    ('[Current date: … | Operator: Ingrid]', eval/longmem/corpus.frame_user_msg)
    — the clock must still resolve the date through the tail, and the time
    must survive too."""
    from servers.clock import conversation_now
    import datetime as dt
    msgs = [
        {'role': 'user',
         'content': '[Current date: 2023/03/19 (Sun) 07:16 | '
                    'Operator: Ingrid]\n\nI did a 5K today'},
    ]
    now = conversation_now(messages=msgs)
    assert now.date() == dt.date(2023, 3, 19)
    assert (now.hour, now.minute) == (7, 16)


def test_conversation_now_uses_latest_user_message():
    """When multiple user messages have prefixes, the most RECENT one wins."""
    from servers.clock import conversation_now
    import datetime as dt
    msgs = [
        {'role': 'user', 'content': '[Current date: 2023/03/19]\n\nfirst'},
        {'role': 'assistant', 'content': 'ok'},
        {'role': 'user', 'content': '[Current date: 2023/03/26]\n\nsecond'},
    ]
    assert conversation_now(messages=msgs).date() == dt.date(2023, 3, 26)


def test_conversation_now_falls_back_to_brain_now():
    """Without a replay prefix and without session_started_at, falls back to
    operator wall-clock (matches brain_now)."""
    from servers.clock import conversation_now, brain_now
    c = conversation_now(messages=[{'role': 'user', 'content': 'hello'}])
    n = brain_now()
    # Within a few seconds of each other — exact equality is impossible
    delta_s = abs((c - n).total_seconds())
    assert delta_s < 5.0, f'fallback should match brain_now ± 5s; delta {delta_s}s'


# ── The relative-time grammar (resolve_offset) ──


def test_resolve_offset_directions_are_opposite():
    """One grammar, two directions: the same shorthand resolves backwards for
    a lookback bound and forwards for a deadline."""
    from servers.clock import resolve_offset, iso_now, PAST, FUTURE
    now = iso_now()
    assert resolve_offset('2h', direction=PAST) < now
    assert resolve_offset('2h', direction=FUTURE) > now
    # Case-insensitive, and 'w' is 7 days.
    assert resolve_offset('1W', direction=FUTURE) > resolve_offset(
        '6d', direction=FUTURE)


def test_resolve_offset_normalizes_a_tz_offset_literal_to_utc():
    """An offset-bearing literal is CONVERTED, never passed through wearing
    its own offset. As text '…12:00:00+03:00' sorts AFTER '…09:30:00+00:00'
    while naming an EARLIER instant, so a bound that keeps its offset shifts
    every window it filters. Absolute literals ignore direction."""
    from servers.clock import resolve_offset, PAST, FUTURE
    for direction in (PAST, FUTURE):
        assert resolve_offset('2026-06-14T12:00:00+03:00',
                              direction=direction) == \
            '2026-06-14T09:00:00+00:00'


def test_resolve_offset_reports_an_unrepresentable_offset_as_valueerror():
    """An offset large enough to leave datetime's range must surface as
    ValueError, not OverflowError: both doors guard on ValueError alone, so
    an OverflowError sails past them and out as an opaque failure instead of
    the loud, actionable rejection the write boundary promises."""
    from servers.clock import resolve_offset, PAST, FUTURE
    for value in ('99999999h', '500000w'):
        for direction in (PAST, FUTURE):
            with pytest.raises(ValueError):
                resolve_offset(value, direction=direction)


def test_both_doors_reject_an_unrepresentable_offset_loudly():
    """The doors' own guards, end to end — a fat-fingered deadline comes back
    as guidance, never as a crash. And the guidance must be TRUE: each door
    supplies the subject, the grammar supplies the reason, so a
    valid-but-out-of-range shorthand is never described as 'not shorthand'."""
    from servers.brain_traces import _resolve_time_bound
    from servers.channels.thalamus.thalamus_contract import resolve_when
    for door, subject in ((_resolve_time_bound, 'time bound'),
                          (resolve_when, 'when=')):
        with pytest.raises(ValueError) as caught:
            door('99999999h')
        msg = str(caught.value)
        assert subject in msg, f'{subject!r} must name the offending param'
        assert 'range' in msg, f'range refusal lost its reason: {msg}'
        assert 'neither' not in msg, (
            f'valid shorthand wrongly reported as unparseable: {msg}')
        # ...while a genuinely unparseable value still says so.
        with pytest.raises(ValueError) as caught:
            door('next full moon')
        assert 'neither' in str(caught.value)


def test_resolve_offset_rejects_an_unknown_direction():
    """direction is closed vocabulary. A typo must not fall through to the
    else-branch and silently turn a lookback bound into a deadline."""
    from servers.clock import resolve_offset
    with pytest.raises(ValueError):
        resolve_offset('2h', direction='pst')


def test_one_grammar_across_both_doors():
    """The grammar is a PUBLISHED contract — quoted verbatim in the
    recall_episodes / remind / thalamus_resolve tool descriptions — so both
    doors must resolve an absolute literal to the same instant. They had
    already drifted: the trace bound preserved a literal's own UTC offset
    while the Thalamus door converted it, so one string named two different
    moments depending on which door received it."""
    from servers.brain_traces import _resolve_time_bound
    from servers.channels.thalamus.thalamus_contract import resolve_when
    literal = '2026-06-14T12:00:00+03:00'
    assert _resolve_time_bound(literal) == resolve_when(literal)
    assert _resolve_time_bound(literal) == '2026-06-14T09:00:00+00:00'
    # Each door still owns its own empty-value convention: a queue has a
    # "next opportunity", a lookback bound does not.
    assert _resolve_time_bound('') == ''
    assert resolve_when('now') is None


def test_prefix_zones_agree():
    """The two clock contracts must scan the SAME tree.

    They enforce halves of one rule — this file bans bare datetime.now() /
    date.today(); test_time_window_contract bans iso_now() / iso_cutoff()
    without at=. Widening one alone is worse than leaving both narrow,
    because the widened one's comment then promises a coverage the other
    does not deliver, and the gap is silent: a bare iso_cutoff() in a newly
    added grain would ship uncaught and anchor an eval replay to wall-clock.

    That is not hypothetical — it is exactly what the 2026-09-01 channels/
    move did, and only an audit caught it. This assertion is the guard that
    would have caught it instead.
    """
    from tests.test_time_window_contract import CTX_PROTECTED_DIRS
    assert sorted(PROTECTED_DIRS) == sorted(CTX_PROTECTED_DIRS), (
        'The two clock contracts scan different trees:\n'
        '  test_clock_contract_sync.PROTECTED_DIRS      = %r\n'
        '  test_time_window_contract.CTX_PROTECTED_DIRS = %r\n'
        'Widen or narrow BOTH, or the narrower one becomes a silent hole '
        'the wider one claims to cover.' % (PROTECTED_DIRS, CTX_PROTECTED_DIRS)
    )
