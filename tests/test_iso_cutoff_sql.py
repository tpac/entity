"""SQL time-window regression test for servers.clock.iso_cutoff.

Why this file exists
--------------------
Discovered 2026-05-24: many SQL `WHERE col > datetime('now', '-N hours')`
queries were silently wrong against ISO-T-formatted timestamp columns.
SQLite's ``datetime('now', '-N hours')`` returns ``'2026-05-24 17:07:13'``
(space-separated, no microseconds, no TZ). Stored values use ``'T'``
separators (with either ``Z`` or ``+00:00`` suffix). Lexicographic
comparison breaks because ``'T' (0x54) > ' ' (0x20)``: a row stored at
``'2026-05-24T01:00:00...'`` (early in the day) lexicographically beats
the cutoff ``'2026-05-24 17:07:13'`` and passes a ``>`` filter it should
have failed.

Fix: ``iso_cutoff()`` returns a matching-format string bound as a SQL
parameter. This test locks the behavior in.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from servers.clock import iso_cutoff, iso_now


ISO_PLUS_RE = re.compile(
    r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$')


# ─── 1. Shape of the cutoff string ──────────────────────────────────────


def test_iso_cutoff_format():
    """Cutoff string must be full-ISO with microseconds and +00:00 suffix.

    Anything shorter (e.g. without microseconds, or with 'Z') would lex-
    compare against stored timestamps in ways that re-introduce the bug.
    """
    s = iso_cutoff(hours=1)
    assert ISO_PLUS_RE.match(s), f'unexpected cutoff shape: {s!r}'


def test_iso_cutoff_args_subtract():
    """Each kwarg shifts the cutoff into the past."""
    base = datetime.now(timezone.utc)
    one_hour = datetime.fromisoformat(iso_cutoff(hours=1))
    one_day = datetime.fromisoformat(iso_cutoff(days=1))
    one_min = datetime.fromisoformat(iso_cutoff(minutes=1))

    assert timedelta(minutes=59) < (base - one_hour) < timedelta(minutes=61)
    assert timedelta(hours=23) < (base - one_day) < timedelta(hours=25)
    assert timedelta(seconds=50) < (base - one_min) < timedelta(seconds=70)


# ─── Conversation-time anchoring (at= kwarg) ────────────────────────────


def test_iso_now_default_is_wallclock():
    """iso_now() without args returns wall-clock UTC."""
    s = iso_now()
    parsed = datetime.fromisoformat(s)
    drift = abs(datetime.now(timezone.utc) - parsed)
    assert drift < timedelta(seconds=2), f'iso_now drifted: {drift}'


def test_iso_now_honors_explicit_anchor():
    """iso_now(at=X) returns X normalized to UTC ISO.

    Lets eval-time callers stamp conversation_now() into created_at,
    so historical replays anchor to the conversation, not wall-clock.
    """
    anchor = datetime(2023, 3, 19, 14, 16, 0, tzinfo=timezone.utc)
    s = iso_now(at=anchor)
    assert s == '2023-03-19T14:16:00+00:00', f'expected anchored iso, got {s!r}'


def test_iso_cutoff_honors_explicit_anchor():
    """iso_cutoff(hours=24, at=conversation_now()) windows against the
    anchor, not wall-clock.

    Without this, an eval replay of a 2023 conversation would compute
    'the last 24 hours' against 2026 and silently match nothing.
    """
    anchor = datetime(2023, 3, 19, 14, 16, 0, tzinfo=timezone.utc)
    s = iso_cutoff(hours=24, at=anchor)
    expected = datetime(2023, 3, 18, 14, 16, 0, tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(s)
    assert parsed == expected, f'cutoff = {parsed!r}, expected {expected!r}'


def test_iso_cutoff_at_is_keyword_only():
    """at= must be passed by keyword — positional would silently mean
    something else if signature changes. Lock it in."""
    with pytest.raises(TypeError):
        iso_cutoff(1, 0, 0, datetime.now(timezone.utc))  # type: ignore[misc]


# ─── 2. SQL windowing against both stored formats ───────────────────────


def _build_db_with_rows(rows):
    """Build an in-memory SQLite with a created_at TEXT column."""
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE t (label TEXT, created_at TEXT)')
    conn.executemany('INSERT INTO t (label, created_at) VALUES (?, ?)', rows)
    conn.commit()
    return conn


def _iso_plus(dt: datetime) -> str:
    """TraceDAL-style storage: ISO with +00:00 suffix."""
    return dt.astimezone(timezone.utc).isoformat()


def _iso_z(dt: datetime) -> str:
    """Brain.now()-style storage: ISO with 'Z' suffix."""
    return dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


# ─── 2a. Same-day-same-hour boundary (the trip hazard) ──────────────────


@pytest.mark.parametrize('formatter,label', [
    (_iso_plus, '+00:00'),
    (_iso_z, 'Z'),
])
def test_same_day_same_hour_boundary(formatter, label):
    """Most-likely-to-fool-future-engineers case.

    All three rows fall on the same calendar day AND the same hour.
    A `created_at > iso_cutoff(minutes=30)` filter must pick out exactly
    the row that's < 30 min old. The old datetime('now', ...) path would
    have admitted all three because ``'T' > ' '`` at position 10.
    """
    now = datetime.now(timezone.utc)
    rows = [
        ('fresh_10min', formatter(now - timedelta(minutes=10))),  # in window
        ('older_45min', formatter(now - timedelta(minutes=45))),  # out
        ('older_55min', formatter(now - timedelta(minutes=55))),  # out
    ]
    conn = _build_db_with_rows(rows)

    cutoff = iso_cutoff(minutes=30)
    found = sorted(r[0] for r in conn.execute(
        'SELECT label FROM t WHERE created_at > ? ORDER BY label', (cutoff,)
    ).fetchall())

    assert found == ['fresh_10min'], (
        f'{label} format: expected only fresh_10min, got {found!r}')


# ─── 2b. Cross-day boundary ──────────────────────────────────────────────


@pytest.mark.parametrize('formatter,label', [
    (_iso_plus, '+00:00'),
    (_iso_z, 'Z'),
])
def test_cross_day_boundary(formatter, label):
    """Day-crossing window picks the correct subset."""
    now = datetime.now(timezone.utc)
    rows = [
        ('today_now', formatter(now - timedelta(minutes=5))),    # in
        ('today_recent', formatter(now - timedelta(hours=2))),   # in
        ('yesterday_close', formatter(now - timedelta(hours=25))),  # out
        ('week_ago', formatter(now - timedelta(days=7))),        # out
    ]
    conn = _build_db_with_rows(rows)

    cutoff = iso_cutoff(hours=24)
    found = sorted(r[0] for r in conn.execute(
        'SELECT label FROM t WHERE created_at > ? ORDER BY label', (cutoff,)
    ).fetchall())

    assert found == sorted(['today_now', 'today_recent']), (
        f'{label} format: window picked {found!r}')


# ─── 2c. Backwards-window DELETE (cleanup query shape) ──────────────────


@pytest.mark.parametrize('formatter,label', [
    (_iso_plus, '+00:00'),
    (_iso_z, 'Z'),
])
def test_delete_older_than_cutoff(formatter, label):
    """The `<` window used by cleanup queries (e.g. ``run_maintenance``)."""
    now = datetime.now(timezone.utc)
    rows = [
        ('fresh', formatter(now - timedelta(hours=1))),
        ('old_31d', formatter(now - timedelta(days=31))),
        ('old_60d', formatter(now - timedelta(days=60))),
    ]
    conn = _build_db_with_rows(rows)

    cutoff = iso_cutoff(days=30)
    cur = conn.execute('DELETE FROM t WHERE created_at < ?', (cutoff,))
    deleted = cur.rowcount
    remaining = sorted(r[0] for r in conn.execute(
        'SELECT label FROM t ORDER BY label').fetchall())

    assert deleted == 2, f'{label} format: deleted {deleted}, expected 2'
    assert remaining == ['fresh']


# ─── 3. The old broken pattern still fails (proof of regression) ────────


@pytest.mark.parametrize('formatter,label', [
    (_iso_plus, '+00:00'),
    (_iso_z, 'Z'),
])
def test_old_datetime_now_pattern_is_still_broken(formatter, label):
    """Regression guardrail: confirm SQLite's ``datetime('now', ...)`` would
    still mis-window the same data — so we know the helper is doing real
    work, not coincidentally agreeing with the broken path.

    If this ever starts passing, SQLite's datetime() behavior changed
    (unlikely) — and the helper may no longer be necessary.

    Anchored at a fixed mid-day instant on purpose — do NOT revert to
    ``datetime.now()``. The lex bug only fires when the stored rows and
    SQLite's space-separated cutoff share a calendar DAY: the date part has
    to be equal for the ``'T' (0x54) > ' ' (0x20)`` trap at position 10 to
    decide the ``>`` comparison. With wall-clock ``now()`` the 55-min-old row
    landed on the *previous* UTC day for runs in the [00:30, 00:55) UTC band
    (cutoff today, older row yesterday → date parts genuinely differ → bug
    does NOT fire), and this guardrail flipped to a false failure. A frozen
    anchor keeps every timestamp same-day so the bug fires deterministically.
    """
    base = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    rows = [
        ('fresh_10min', formatter(base - timedelta(minutes=10))),  # should be in
        ('older_55min_same_hour',
         formatter(base - timedelta(minutes=55))),  # should be OUT but bug admits it
    ]
    conn = _build_db_with_rows(rows)

    # The old broken pattern. SQLite still computes the cutoff with its own
    # datetime() — space-separated, microseconds + TZ stripped — which is what
    # re-creates the bug; we just feed it the same fixed anchor instead of
    # wall-clock 'now' so the result is deterministic. Derived from `base` so
    # the rows and the cutoff can never drift apart.
    anchor_sql = base.strftime('%Y-%m-%dT%H:%M:%S')  # '2026-06-15T12:00:00'
    broken = sorted(r[0] for r in conn.execute(
        "SELECT label FROM t WHERE created_at > datetime(?, '-30 minutes') "
        'ORDER BY label', (anchor_sql,)
    ).fetchall())

    # Bug signature: when same-day-same-hour ISO-T strings hit a space-
    # separated cutoff, the older row leaks through.
    assert 'older_55min_same_hour' in broken, (
        f'{label}: expected the bug to still fire (older row should leak), '
        f'got {broken!r}')
