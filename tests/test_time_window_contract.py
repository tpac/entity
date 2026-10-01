"""Time-window contract: ban SQL datetime('now',...) and require explicit
``at=`` on grain-axis cutoffs.

Why this file exists
--------------------
Two related drift surfaces, one test file:

1. **SQL ``datetime('now', '-N units')`` against TEXT timestamp columns.**
   SQLite returns a space-separated format with no microseconds and no
   timezone. Brain stores ISO with 'T'. Lex comparison breaks
   (``'T' (0x54) > ' ' (0x20)``); same-day rows incorrectly pass ``>``.
   Discovered 2026-05-24 — 12 broken sites. Fix is to bind
   ``servers.clock.iso_cutoff(...)``. This scan keeps the bug from
   coming back.

2. **Grain-axis ``iso_cutoff()`` / ``iso_now()`` calls without explicit
   ``at=``.** Grain-axis code runs in eval replays where wall-clock is the
   wrong anchor — see servers/clock.py:conversation_now (a grain-axis
   caller defaulting to wall-clock now() mis-anchors every relative date
   in a replay). Forcing ``at=`` at the call site means authors think about
   anchoring instead of silently defaulting to wall-clock.

Mirrors the pattern of ``tests/test_clock_contract_sync.py``.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


# ─── Scan 1: SQL datetime('now',...) codebase-wide ──────────────────────

# Match SQLite's STRING-returning date functions called with a literal
# 'now'. These return formatted strings that can't be lex-compared
# against ISO-T stored timestamps.
#
# Deliberately EXCLUDES julianday() — it returns a numeric Julian day
# count, format-agnostic, and is the safe way to do "hours between
# 'now' and a stored ISO-T timestamp" arithmetic in SQL (see
# servers/dal.py edge-decay query).
SQL_DATETIME_RE = re.compile(
    r"""\b(?:datetime|strftime|date|time)\s*\(\s*['"]now['"]""",
    re.IGNORECASE,
)

# Directories scanned for the SQL ban. The full Python runtime that
# touches brain SQLite databases.
SQL_SCAN_DIRS = ['servers', 'dashboard', 'hooks']

# Paths that are exempt — the helper itself plus test/docstring files that
# deliberately contain the banned pattern to demonstrate or explain it.
# Keep this list TIGHT. Every entry is a foothold for the bug.
SQL_EXEMPT_PATHS = {
    # Documentation references in the helper itself.
    'servers/clock.py',
    # Test file describing the bug — it exists to demonstrate the
    # broken pattern still fails.
    'tests/test_iso_cutoff_sql.py',
    # Self.
    'tests/test_time_window_contract.py',
    # Dashboard implements its own correct workaround via utc_cutoff();
    # the docstring mentions datetime('now') to explain why.
    'dashboard/brain_dashboard_standalone.py',
}

# Lines containing this marker are exempt — the escape hatch for an
# intentional datetime('now') against a SQLite-native timestamp column.
SQL_EXEMPT_MARKER = '# sql-datetime-ok'


def _scan_sql(p: Path):
    violations = []
    try:
        text = p.read_text()
    except (OSError, UnicodeDecodeError):
        return violations
    for lineno, line in enumerate(text.splitlines(), start=1):
        if SQL_EXEMPT_MARKER in line:
            continue
        # Skip comment lines — the bug pattern in a comment is
        # documentation, not a real call.
        stripped = line.lstrip()
        if stripped.startswith('#') or stripped.startswith('"""') \
                or stripped.startswith("'''"):
            continue
        if SQL_DATETIME_RE.search(line):
            violations.append((lineno, line.strip()))
    return violations


def test_no_sqlite_datetime_now_in_sql_strings():
    """No new SQL ``datetime('now', ...)`` against ISO-T columns.

    If this test fails, the offending caller is using SQLite's
    space-separated datetime() output against a 'T'-separated stored
    timestamp. Replace with ``from servers.clock import iso_cutoff``
    and bind the cutoff as a parameter.

    If the call is intentional (e.g., comparing against a SQLite-native
    TIMESTAMP column that uses the space-separated format), add a
    trailing ``# sql-datetime-ok`` comment on the same line.
    """
    all_violations = {}
    for d in SQL_SCAN_DIRS:
        for p in (ROOT / d).rglob('*.py'):
            rel = p.relative_to(ROOT).as_posix()
            if rel in SQL_EXEMPT_PATHS:
                continue
            v = _scan_sql(p)
            if v:
                all_violations[rel] = v

    # No grandfathered line-number sites remain: the brain_dashboard.db
    # mid-deprecation INSERT was retired (DAEMON_DOWN now writes ISO-T to
    # brain_logs.db.hook_errors). The inline # sql-datetime-ok marker is the
    # only exemption path now — anything the scan still surfaces is a real bug.
    unexpected = all_violations

    if unexpected:
        msg = ['Found SQL datetime("now",...) calls outside the exempt list:']
        for rel, hits in unexpected.items():
            for ln, src in hits:
                msg.append(f'  {rel}:{ln}  {src}')
        msg.append('')
        msg.append('Fix: import iso_cutoff and bind:')
        msg.append('  from .clock import iso_cutoff')
        msg.append('  WHERE created_at > ?   --  bind iso_cutoff(hours=N)')
        msg.append('')
        msg.append('Or, if intentional (e.g., a column that stores SQLite-native '
                   'space-separated timestamps), suppress with a trailing '
                   '# sql-datetime-ok comment.')
        pytest.fail('\n'.join(msg))


# ─── Scan 2: grain-axis calls without explicit at= ──────────────────────

# Conversation-time directories. Grain-axis code anchors to the conversation,
# not the host wall-clock — passing at=conversation_now(...) keeps eval
# replays honest. Must equal test_clock_contract_sync.PROTECTED_DIRS; the two
# are halves of one rule, and test_prefix_zones_agree enforces the equality so
# neither can be widened alone. What belongs on the axis: servers/scales/__init__.py.
CTX_PROTECTED_DIRS = [
    'servers/scales',
]

# Match iso_now( / iso_cutoff( and look at the args.
CTX_CALL_RE = re.compile(r'\b(iso_now|iso_cutoff)\s*\(([^)]*)\)')

CTX_EXEMPT_MARKER = '# clock-ok'

CTX_EXEMPT_PATHS = {
    # None right now.
}


def _scan_ctx(p: Path):
    """Yield (line_no, line, fname) for iso_now/iso_cutoff calls in
    grain-axis code without an at= kwarg. clock-ok lines are skipped."""
    violations = []
    try:
        text = p.read_text()
    except (OSError, UnicodeDecodeError):
        return violations
    for lineno, line in enumerate(text.splitlines(), start=1):
        if CTX_EXEMPT_MARKER in line:
            continue
        for m in CTX_CALL_RE.finditer(line):
            fname, args = m.group(1), m.group(2)
            # Bare wall-clock — fine for system bookkeeping but suspect
            # on the grain axis. Require explicit `at=` so the author confirms
            # they want wall-clock here (and not conversation_now()).
            if 'at=' not in args:
                violations.append((lineno, line.strip(), fname))
    return violations


def test_grain_axis_cutoffs_require_explicit_at():
    """On the grain axis, every iso_now()/iso_cutoff() call must pass at=.

    Grain-axis code reads and writes data tied to the conversation. Eval replays
    inject historical `[Current date: ...]` prefixes — wall-clock is
    the wrong anchor there. Forcing explicit ``at=`` makes the author
    decide which 'now' applies:

        ts = iso_now(at=conversation_now(messages=msgs))   # eval-aware
        ts = iso_now(at=brain_now(self.brain))             # operator TZ
        ts = iso_now(at=None)                              # explicit wall-clock

    If a call truly wants wall-clock and doesn't care about eval, mark
    the line with ``# clock-ok`` and the test will skip it.
    """
    all_violations = {}
    for d in CTX_PROTECTED_DIRS:
        for p in (ROOT / d).rglob('*.py'):
            rel = p.relative_to(ROOT).as_posix()
            if rel in CTX_EXEMPT_PATHS:
                continue
            v = _scan_ctx(p)
            if v:
                all_violations[rel] = v

    if all_violations:
        msg = ['Found bare iso_now()/iso_cutoff() calls under servers/scales (the grain axis):']
        for rel, hits in all_violations.items():
            for ln, src, fname in hits:
                msg.append(f'  {rel}:{ln}  {src}')
        msg.append('')
        msg.append('Pass at= explicitly so eval replays inherit conversation time:')
        msg.append('  from servers.clock import iso_now, conversation_now')
        msg.append('  ts = iso_now(at=conversation_now(messages=msgs))')
        msg.append('')
        msg.append('Or mark the line # clock-ok if wall-clock is genuinely correct.')
        pytest.fail('\n'.join(msg))
