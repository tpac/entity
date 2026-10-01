"""Single source of truth for "now" across S1/S2/encoder.

Why this file exists
--------------------
Grain-axis code runs in two clocks: production (the conversation is happening
now) and eval replay (the encoder runs over a historical conversation dated,
say, 2023-03-19). A caller that defaults to the wall-clock resolves
"today/yesterday/last Tuesday" against the replay date instead of the
conversation date, and every relative phrase mis-anchors to NOW. Routing every
"now" through this module is what keeps replays honest.

The architectural fix: ONE function for "now". Callers state which "now" they
want — wall-clock (operator's TZ) or conversation-anchored. Eval inherits the
fix because every code path uses the same helper.

See brain memory dcb5b951 for the architecture decision and 6d5b789e for the
bug being fixed.

Resolution priority (brain_now)
-------------------------------
  1. Explicit `tz` parameter (callers that know better)
  2. Brain config 'operator_tz' (when wired through userConfig)
  3. Host wall-clock (the daemon runs on the operator's machine)
  4. UTC fallback — logged loudly so we know if we got here

UTC-internal is the better long-term architecture (storage UTC, render in
operator TZ). Deferred for now — see BACKLOG.md "UTC-internal clock refactor".

Contract
--------
DO NOT call `datetime.now()`, `date.today()`, or `time.time()` directly in
S1/S2 code. Use brain_now() / conversation_now(). A contract test
(tests/test_clock_contract_sync.py) scans for direct calls and fails.
Exempt: telemetry/perf timers (use time.monotonic explicitly), schema files.

Which clock for an ELAPSED INTERVAL — the answer is three-valued
---------------------------------------------------------------
The exemption above ("use time.monotonic") is not the whole rule, and reading
it as "elapsed means monotonic" is wrong. `time.monotonic()` does NOT advance
while the process is suspended (macOS sleep) — see the note on
ANTHROPIC_CLIENT_TIMEOUT in brain_constants.py. So:

  A. SEMANTIC / PERSISTED time -> WALL CLOCK.
     Anything compared against a stored timestamp, or that means a date.
     e.g. brain.scribe_due (its docstring says so), the maintenance brain_meta
     stamp, a backup seeded from a filesystem mtime, trace created_at.

  B. ELAPSED REAL TIME THAT MUST COUNT SUSPEND -> WALL CLOCK.
     "Has four hours of world-time passed?" A laptop asleep for four hours HAS
     been idle for four hours. This is why the daemon's idle timeout fires on
     lid-open: correct, not a bug.

  C. ELAPSED PROCESS TIME, IMMUNE TO CLOCK STEPS -> time.monotonic().
     Poll tick gates, retry cooldowns, rate limits, stall checks, deadlines.
     On wall clock a backward step (NTP correction, manual set) makes the
     delta NEGATIVE and the gate never fires again until the clock catches up
     — silently, while everything looks healthy.

The failure mode runs both ways, which is why there is no blanket answer and
no helper that decides for you: converting a category-B gate to monotonic
disables it across sleep, and leaving a category-C gate on wall clock freezes
it on a backward step. State the category at the site.

`tests/test_interval_clock_guardrail.py` ratchets category C: a new wall-clock
interval gate fails the build unless it declares itself A or B with
`# wall-clock-ok: <why>`. (Deliberately a separate file from
test_clock_contract_sync.py — that one's PROTECTED_DIRS is asserted equal to
test_time_window_contract's, so widening it would silently widen the
semantic-time rule too.)
"""
from __future__ import annotations

import datetime as _dt
import re as _re
import sys as _sys
from typing import Any, Iterable, Optional


# ─── Public API ─────────────────────────────────────────────────────────


def brain_now(brain=None, tz: Optional[str] = None) -> _dt.datetime:
    """Return the current 'now' in the operator's frame as a timezone-aware
    datetime.

    Resolution priority:
      1. `tz` parameter (e.g. "America/New_York")
      2. brain.config.get('operator_tz') if brain is provided
      3. Host system TZ (daemon runs on operator's machine, so this IS
         operator TZ in practice)
      4. UTC fallback with stderr warning

    Callers should pass `brain` when they have one — lets brain config drive
    TZ when the userConfig wires that in later.
    """
    target_tz = _resolve_tz(brain=brain, tz=tz)
    if target_tz is None:
        # Final UTC fallback — log so we know.
        print('[clock] WARN brain_now falling back to UTC (no host TZ, no '
              'operator_tz, no explicit tz)', file=_sys.stderr)
        return _dt.datetime.now(_dt.timezone.utc)
    return _dt.datetime.now(target_tz)


def brain_today(brain=None, tz: Optional[str] = None) -> _dt.date:
    """Return today's date in the operator's frame. Convenience wrapper."""
    return brain_now(brain=brain, tz=tz).date()


def iso_now(at: Optional[_dt.datetime] = None) -> str:
    """Return an ISO-format UTC timestamp suitable for stored ``created_at`` /
    ``updated_at`` / cutoff comparisons.

    This is the single source of truth for "now" as a timestamp STRING.
    ``Brain.now()`` and the TraceDAL inserts route through this helper so
    all rows written by brain code converge on one format.

    Format: ``'YYYY-MM-DDTHH:MM:SS.ffffff+00:00'``.

    Args:
        at: pass a datetime to override wall-clock. Use this in S1/S2 code
            to honor ``conversation_now()`` during eval replays — the
            stamped timestamp anchors to the conversation's notional now
            instead of the host wall-clock.

            Examples:
                ts = iso_now()                            # wall-clock (default)
                ts = iso_now(at=conversation_now(msgs))   # eval-anchored

    See ``iso_cutoff`` for the WHERE-clause companion.
    """
    base = at if at is not None else _dt.datetime.now(_dt.timezone.utc)
    return base.astimezone(_dt.timezone.utc).isoformat()


def iso_cutoff(hours: float = 0, minutes: float = 0,
               days: float = 0, *,
               at: Optional[_dt.datetime] = None) -> str:
    """Return an ISO-format UTC cutoff timestamp suitable for SQL WHERE clauses.

    Use bound as a parameter against ISO-T-formatted timestamp columns:

        cur.execute(
            "SELECT ... WHERE created_at > ?",
            (iso_cutoff(hours=24),))

    Why this exists
    ---------------
    SQLite's ``datetime('now', '-N hours')`` returns a space-separated format
    without microseconds or timezone (e.g. ``'2026-05-24 17:07:13'``). Brain
    stores ISO with a 'T' separator (e.g. ``'2026-05-24T17:07:13.123456+00:00'``
    or, in historical rows, with a 'Z' suffix). Lexicographic comparison
    breaks because ``'T' (0x54) > ' ' (0x20)`` — same-day-earlier rows
    incorrectly look "greater than" a same-day cutoff. Replace any
    ``WHERE col > datetime('now', '-N units')`` in SQL with a bound
    ``iso_cutoff(...)`` to compare apples to apples.

    Format coexistence (historical, latent-safe)
    --------------------------------------------
    Pre-2026-05-24 ``Brain.now()`` emitted ``'…T…ffffffZ'`` while TraceDAL
    inserts emitted ``'…T…ffffff+00:00'``. New writes converge on
    ``+00:00`` via ``iso_now()`` — but stored rows from before that point
    keep their original suffix. ``'Z' (0x5A) > '+' (0x2B)`` so at an
    exact-microsecond match a Z-stored row compares as later than a
    +00:00 cutoff. Statistically irrelevant for time windows of hours/
    days, but worth knowing if you ever do exact-instant comparisons
    across mixed-format columns.

    **Invariant: queries stay within a single column.** Every WHERE-clause
    timestamp comparison in the codebase reads from ONE column. Don't
    compare timestamps lexicographically across tables with different
    historical suffix conventions — normalize first or use parsed
    datetimes.

    Conversation-time anchoring
    ---------------------------
    By default, the cutoff is computed against wall-clock UTC — correct
    for system bookkeeping (log cleanup, integrity audits, dashboard
    counts) where the question is "what happened in the real world in
    the last N hours."

    In S1/S2 code that filters CONVERSATION data (recall windows,
    encoding-time neighborhoods), pass ``at=conversation_now(...)`` so
    eval replays of historical conversations anchor the cutoff to the
    conversation's notional time, not the host wall-clock. Without
    this, eval replays of 2023 conversations would compute "the last
    24 hours" against 2026 and silently return empty / wrong windows.

    See servers/clock.py:conversation_now for resolution priority.

    Args:
        hours, minutes, days: subtracted from the anchor time.
        at: optional anchor (datetime). Defaults to wall-clock UTC.

    Returns:
        ISO-8601 string in UTC, ``'YYYY-MM-DDTHH:MM:SS.ffffff+00:00'``.
    """
    base = at if at is not None else _dt.datetime.now(_dt.timezone.utc)
    dt = base - _dt.timedelta(hours=hours, minutes=minutes, days=days)
    return dt.astimezone(_dt.timezone.utc).isoformat()


def iso_after(hours: float = 0, minutes: float = 0,
              days: float = 0, *,
              at: Optional[_dt.datetime] = None) -> str:
    """Forward-offset sibling of ``iso_cutoff`` — an ISO UTC timestamp ``N``
    units AFTER the anchor, for a future deadline written to a column and later
    compared with a plain ``WHERE col > ?`` bound to ``iso_now()``::

        expires_at = iso_after(hours=1)        # 1h from now, UTC ISO

    Same format contract as ``iso_now``/``iso_cutoff`` — ``'…T…+00:00'``,
    lex-comparable, never SQLite's space-separated ``datetime('now', '+N
    hours')``. Wall-clock by default (real-elapsed deadlines / system
    bookkeeping). Pass ``at=conversation_now(...)`` only if a deadline must
    anchor to conversation time during S1/S2 replay — rarely.

    Args:
        hours, minutes, days: added to the anchor time.
        at: optional anchor (datetime). Defaults to wall-clock UTC.
    Returns:
        ISO-8601 string in UTC, ``'YYYY-MM-DDTHH:MM:SS.ffffff+00:00'``.
    """
    base = at if at is not None else _dt.datetime.now(_dt.timezone.utc)
    dt = base + _dt.timedelta(hours=hours, minutes=minutes, days=days)
    return dt.astimezone(_dt.timezone.utc).isoformat()


# The relative-time grammar a producer or caller WRITES, resolving into the
# iso_cutoff/iso_after pair above. It is a published contract — quoted
# verbatim in the recall_episodes / remind / thalamus_resolve tool
# descriptions — so it lives in one place or those descriptions become true
# for one caller and false for the other.
PAST = 'past'      # shorthand runs backwards → iso_cutoff (a lookback bound)
FUTURE = 'future'  # shorthand runs forwards  → iso_after (a deadline)

_OFFSET_RE = _re.compile(r'(\d+)\s*([mhdw])', _re.IGNORECASE)


def resolve_offset(value: Any, *, direction: str) -> str:
    """Resolve a relative-time expression to an ISO UTC timestamp.

    Accepts the shorthand ('30m', '2h', '3d', '1w', case-insensitive) or an
    ISO timestamp literal. ``direction`` picks which way the shorthand runs;
    an ISO literal is absolute and ignores it.

    Always returns the ``'…T…+00:00'`` UTC form the rest of this file emits,
    so the result stays lex-comparable against stored timestamps. An
    offset-bearing literal is CONVERTED to UTC rather than passed through
    wearing its own offset — ``'2026-09-01T12:00:00+03:00'`` sorts as later
    than ``'2026-09-01T09:30:00+00:00'`` as text while naming an earlier
    instant, which silently shifts any window it bounds. The literal goes
    through ``fromisoformat`` rather than a date-prefix check so that junk
    and impossible dates are REJECTED and the space-separated form is
    re-emitted as ISO-T (see iso_cutoff on the ``'T' > ' '`` lex hazard).

    Raises ValueError on anything else — including an offset too large to
    land inside datetime's range. A malformed or unrepresentable deadline
    must fail loud, never become an immediate delivery or a disabled filter,
    and every caller here guards on ValueError alone.

    The message is a bare REASON ("is neither…", "resolves outside…") with no
    subject: callers own their empty-value convention and supply their own
    subject and framing, so the composed sentence names the offending
    parameter AND why it was refused. A caller that swallows the reason tells
    a producer its valid-but-out-of-range shorthand "is not shorthand".
    """
    if direction not in (PAST, FUTURE):
        raise ValueError('resolve_offset: direction=%r is not %r or %r'
                         % (direction, PAST, FUTURE))
    s = str(value).strip()
    m = _OFFSET_RE.fullmatch(s)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        kw = {'m': {'minutes': n}, 'h': {'hours': n},
              'd': {'days': n}, 'w': {'days': 7 * n}}[unit]
        try:
            return iso_cutoff(**kw) if direction == PAST else iso_after(**kw)
        except OverflowError:
            # datetime's range, surfaced as the door's own vocabulary: the
            # callers catch ValueError, so an OverflowError here would sail
            # past their guards and out as an opaque failure.
            raise ValueError(
                'resolves outside the representable date range — use a '
                'smaller offset')
    try:
        parsed = _dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
    except ValueError:
        raise ValueError(
            "is neither relative shorthand ('30m','2h','3d','1w') nor an "
            "ISO timestamp ('2026-06-14T12:00:00')")
    if parsed.tzinfo is None:      # storage is tz-aware — match it
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.astimezone(_dt.timezone.utc).isoformat()


def conversation_now(messages: Optional[Iterable[Any]] = None,
                      session_started_at: Optional[Any] = None,
                      brain=None,
                      tz: Optional[str] = None) -> _dt.datetime:
    """Return the 'now' this CONVERSATION thinks it's happening.

    Used by the encoder for resolving relative dates ("today", "yesterday",
    "last Tuesday") in the conversation under encoding. Always returns a
    timezone-aware datetime.

    Resolution priority:
      1. `[Current date: ...]` prefix in the most recent user message
         (eval replay injects this — see eval/longmem/replay.py)
      2. `session_started_at` — the conversation's own date, for a replay
         of old conversations whose messages carry no prefix (a haystack's
         session date). Never a live session's boot time: a live
         conversation is happening now, which is step 3.
      3. brain_now() — operator wall-clock (the production path; the
         conversation is happening NOW)

    Args:
        messages: iterable of message dicts {role, content, ...} or strings;
                  the function looks for the eval's date-prefix marker.
        session_started_at: ISO timestamp string OR datetime; the replayed
                  conversation's date. No production caller passes it.
        brain: for TZ resolution (see brain_now).
        tz: explicit TZ override.

    Never raises — always returns a valid datetime.
    """
    # 1. Eval replay prefix on last user message
    prefix_date = _extract_replay_date_prefix(messages)
    if prefix_date is not None:
        return prefix_date

    # 2. Session start
    if session_started_at is not None:
        parsed = _parse_to_datetime(session_started_at, tz=tz, brain=brain)
        if parsed is not None:
            return parsed

    # 3. Wall-clock fallback (production path)
    return brain_now(brain=brain, tz=tz)


# ─── Internals ──────────────────────────────────────────────────────────


# Eval replay prepends "[Current date: 2023/03/19 (Sun) 07:16]" or
# "[Current date: 2023-03-19]" to user messages — optionally with further
# "|"-joined framing in the same bracket ("… | Operator: Ingrid]", the named
# counterpart the replay presents). Match any of these shapes.
_REPLAY_DATE_RE = _re.compile(
    r'\[Current date:\s*([0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2})'
    r'(?:\s+\([^)]*\))?'              # optional "(Sun)" weekday
    r'(?:\s+(\d{1,2}:\d{2}(?::\d{2})?))?'  # optional HH:MM[:SS]
    r'(?:\s*\|[^\]\n]*)?'             # optional "| …" framing tail (one line —
                                      # a multi-line bracket is never the
                                      # replay's, don't let a paste hijack it)
    r'\s*\]',
    _re.IGNORECASE,
)


def _extract_replay_date_prefix(messages) -> Optional[_dt.datetime]:
    """If the most recent user message starts with '[Current date: ...]',
    parse and return that datetime (UTC-aware). Else None.

    Eval replay injects this prefix — the format's owner is
    eval/longmem/corpus.frame_user_msg (its docstring carries the contract
    this regex must track).
    """
    if not messages:
        return None
    # Find last user message; if `messages` is non-list, snapshot to list.
    try:
        msg_list = list(messages)
    except TypeError:
        return None

    for m in reversed(msg_list):
        if isinstance(m, dict):
            if m.get('role') != 'user':
                continue
            content = m.get('content') or m.get('text') or ''
        elif isinstance(m, str):
            content = m
        else:
            continue
        if not content:
            continue
        match = _REPLAY_DATE_RE.search(content)
        if not match:
            # Only inspect the LAST user message — eval prepends per turn
            # but we want the most recent one, which should be sufficient.
            return None
        date_str = match.group(1).replace('/', '-')
        time_str = match.group(2) or '00:00:00'
        try:
            # Pad single-digit components for fromisoformat
            y, mo, d = date_str.split('-')
            iso = '%s-%s-%sT%s' % (y, mo.zfill(2), d.zfill(2),
                                     time_str if len(time_str.split(':')) >= 2 else (time_str + ':00:00'))
            # Ensure HH:MM:SS
            parts = time_str.split(':')
            if len(parts) == 2:
                time_str = time_str + ':00'
            iso = '%s-%s-%sT%s' % (y, mo.zfill(2), d.zfill(2), time_str)
            dt = _dt.datetime.fromisoformat(iso)
            # Assume operator TZ for the conversation timestamp
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_resolve_tz() or _dt.timezone.utc)
            return dt
        except (ValueError, IndexError):
            return None
    return None


def _parse_to_datetime(val: Any, tz: Optional[str] = None,
                        brain=None) -> Optional[_dt.datetime]:
    """Parse a value (str / datetime / date / int epoch) to a tz-aware
    datetime. Returns None on failure."""
    if val is None:
        return None
    if isinstance(val, _dt.datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=_resolve_tz(brain=brain, tz=tz)
                                or _dt.timezone.utc)
        return val
    if isinstance(val, _dt.date):
        target = _resolve_tz(brain=brain, tz=tz) or _dt.timezone.utc
        return _dt.datetime.combine(val, _dt.time(0, 0), tzinfo=target)
    if isinstance(val, (int, float)):
        return _dt.datetime.fromtimestamp(val, tz=_dt.timezone.utc)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        # Try ISO first
        try:
            dt = _dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_resolve_tz(brain=brain, tz=tz)
                                  or _dt.timezone.utc)
            return dt
        except ValueError:
            pass
        # Fall back to dateutil if available
        try:
            from dateutil import parser as _dp
            dt = _dp.parse(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_resolve_tz(brain=brain, tz=tz)
                                  or _dt.timezone.utc)
            return dt
        except Exception:
            return None
    return None


def _resolve_tz(brain=None, tz: Optional[str] = None) -> Optional[_dt.tzinfo]:
    """Resolve a tzinfo per the priority chain:
       1. Explicit tz string  2. brain.config['operator_tz']
       3. Host system TZ      4. None (caller falls back to UTC)
    """
    candidates = [tz]
    if brain is not None:
        try:
            cfg = getattr(brain, 'config', None)
            if cfg is not None:
                if hasattr(cfg, 'get'):
                    candidates.append(cfg.get('operator_tz'))
                else:
                    candidates.append(getattr(cfg, 'operator_tz', None))
        except Exception:
            pass

    for c in candidates:
        if not c:
            continue
        try:
            from zoneinfo import ZoneInfo  # 3.9+
            return ZoneInfo(c)
        except Exception:
            continue

    # Host TZ — astimezone(None) attaches the local one.
    try:
        local = _dt.datetime.now().astimezone().tzinfo
        if local is not None:
            return local
    except Exception:
        pass
    return None


__all__ = ['brain_now', 'brain_today', 'conversation_now']
