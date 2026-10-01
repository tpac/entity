"""Date-interval parsing from text: explicit-year date references to
(start_ts, end_ts) intervals at the precision the text gives.

The first step of recall_by_time's window parser (fetch_tools._parse_when_to_ts):
a start_when / end_when such as 'March 2023' or 'Q1 2024' resolves here, and
relative phrases ('last week', '3 weeks ago') fall through to _parse_date_expr.

Interval convention (inclusive Unix seconds):
  - "Q1 2024"       -> (quarter-start, quarter-end)
  - "May 2023"      -> (month-start, month-end)
  - "May 22, 2023"  -> (day-start, day-end)
  - "2023-05"       -> (month-start, month-end)
  - "2023-05-22"    -> (day-start, day-end)

Design: regex-first, year-required. `dateparser.search` is deliberately not
used: it fills a missing year with the current one.
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import datetime, timezone
from typing import List, Optional, Tuple


# ── Regex patterns ──────────────────────────────────────────────────────
#
# Date references are matched directly (no `dateparser.search`) so we can
# guarantee precision detection and avoid dateparser's "fill missing year
# with current year" behavior. All patterns require explicit year context.

_MONTH_NAME = (
    r'(?:january|february|march|april|may|june|july|august|september|'
    r'october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|'
    r'oct|nov|dec)'
)

_PATTERN_QUARTER = re.compile(
    r'\b(q[1-4])\s+(\d{4})\b', re.IGNORECASE
)
_PATTERN_ISO_DAY = re.compile(
    r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b'
)
_PATTERN_ISO_MONTH = re.compile(
    r'\b(\d{4})[-/](\d{1,2})\b(?![-/]\d)'   # avoid matching YYYY-MM-DD
)
_PATTERN_MONTH_DAY_YEAR = re.compile(
    rf'\b({_MONTH_NAME})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b',
    re.IGNORECASE
)
_PATTERN_MONTH_YEAR = re.compile(
    rf'\b({_MONTH_NAME})\s+(\d{{4}})\b',
    re.IGNORECASE
)
# Range: "between Feb and April 2023" / "from Jan to March 2023" /
# "January-March 2023" / "Jan-March 2023" / "January through March 2023".
# Same-year only; both months share the trailing year.
# Separator spacing: words (to/and/through) require whitespace on both
# sides; dashes (-, –, —) allow either spacing — matches compact forms
# like "Jan-March 2023" as well as "Jan - March 2023".
_PATTERN_MONTH_RANGE = re.compile(
    rf'\b(?:between\s+|from\s+)?({_MONTH_NAME})'
    rf'(?:\s*[-–—]\s*|\s+(?:to|and|through)\s+)'
    rf'({_MONTH_NAME})\s+(\d{{4}})\b',
    re.IGNORECASE
)

_MONTH_NAME_TO_NUM = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
    'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7,
    'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12,
    'december': 12,
}


def _interval_for_quarter(year: int, q: int) -> Tuple[int, int]:
    start_month = (q - 1) * 3 + 1
    end_month = q * 3
    s = datetime(year, start_month, 1, tzinfo=timezone.utc)
    last_day = monthrange(year, end_month)[1]
    e = datetime(year, end_month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    return int(s.timestamp()), int(e.timestamp())


def _interval_for_month(year: int, month: int) -> Tuple[int, int]:
    last_day = monthrange(year, month)[1]
    s = datetime(year, month, 1, tzinfo=timezone.utc)
    e = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    return int(s.timestamp()), int(e.timestamp())


def _interval_for_day(year: int, month: int, day: int) -> Tuple[int, int]:
    s = datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc)
    e = datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)
    return int(s.timestamp()), int(e.timestamp())


def extract_intervals_from_text(text: Optional[str]) -> List[Tuple[int, int, str]]:
    """Find date references in `text` and return interval tuples.

    Returns: [(start_ts, end_ts, raw_text), ...]. Empty list if no dates
    or if `text` is empty.

    Requires explicit year context — dateless month names ("May") and
    bare day numbers are skipped to avoid current-year fallback errors.

    Deduplicates by (start_ts, end_ts) — multiple raw spellings of the
    same interval collapse to the first match.

    Detection patterns (in priority order):
      1. Q[1-4] YYYY                  -> quarter
      2. YYYY-MM-DD or YYYY/MM/DD     -> day
      3. YYYY-MM (no day)             -> month
      3b. MonthName .. MonthName YYYY -> range (same-year, month-precision span)
      4. MonthName DD, YYYY           -> day
      5. MonthName YYYY               -> month

    Standalone-year matching (e.g., bare "2024") is intentionally NOT
    implemented in v1. Bare 4-digit numbers in technical text (`2000 chars`,
    `4096 tokens`, `2048 ms`) generate too many false positives. Every
    year-bearing interval still flows through patterns 1-5 above, all of
    which require explicit month or quarter context.
    """
    if not text:
        return []
    out: List[Tuple[int, int, str]] = []
    seen: set = set()
    consumed_spans: List[Tuple[int, int]] = []  # char ranges already matched

    def _spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    def _is_consumed(span: Tuple[int, int]) -> bool:
        return any(_spans_overlap(span, c) for c in consumed_spans)

    def _add(span: Tuple[int, int], start_ts: int, end_ts: int, raw: str) -> None:
        key = (start_ts, end_ts)
        if key in seen:
            return
        seen.add(key)
        consumed_spans.append(span)
        out.append((start_ts, end_ts, raw.strip()))

    # 1. Quarter
    for m in _PATTERN_QUARTER.finditer(text):
        if _is_consumed(m.span()):
            continue
        try:
            q = int(m.group(1)[1])
            year = int(m.group(2))
            s, e = _interval_for_quarter(year, q)
            _add(m.span(), s, e, m.group(0))
        except Exception:
            continue

    # 2. ISO day (must come before ISO month — overlap guard handles it)
    for m in _PATTERN_ISO_DAY.finditer(text):
        if _is_consumed(m.span()):
            continue
        try:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            s, e = _interval_for_day(year, month, day)
            _add(m.span(), s, e, m.group(0))
        except Exception:
            continue

    # 3. ISO month (YYYY-MM without trailing -DD)
    for m in _PATTERN_ISO_MONTH.finditer(text):
        if _is_consumed(m.span()):
            continue
        try:
            year, month = int(m.group(1)), int(m.group(2))
            if not (1 <= month <= 12):
                continue
            s, e = _interval_for_month(year, month)
            _add(m.span(), s, e, m.group(0))
        except Exception:
            continue

    # 3b. MonthName .. MonthName YYYY  (range, same year)
    for m in _PATTERN_MONTH_RANGE.finditer(text):
        if _is_consumed(m.span()):
            continue
        try:
            mname1 = m.group(1).lower()
            mname2 = m.group(2).lower()
            month1 = _MONTH_NAME_TO_NUM.get(mname1)
            month2 = _MONTH_NAME_TO_NUM.get(mname2)
            year = int(m.group(3))
            if not (month1 and month2):
                continue
            # Require forward ordering (skip cross-year guesses for now).
            if month1 > month2:
                continue
            s_start = datetime(year, month1, 1, tzinfo=timezone.utc)
            last_day = monthrange(year, month2)[1]
            s_end = datetime(year, month2, last_day, 23, 59, 59,
                              tzinfo=timezone.utc)
            _add(m.span(), int(s_start.timestamp()),
                  int(s_end.timestamp()), m.group(0))
        except Exception:
            continue

    # 4. MonthName DD, YYYY
    for m in _PATTERN_MONTH_DAY_YEAR.finditer(text):
        if _is_consumed(m.span()):
            continue
        try:
            mname = m.group(1).lower()
            month = _MONTH_NAME_TO_NUM.get(mname)
            if not month:
                continue
            day, year = int(m.group(2)), int(m.group(3))
            s, e = _interval_for_day(year, month, day)
            _add(m.span(), s, e, m.group(0))
        except Exception:
            continue

    # 5. MonthName YYYY
    for m in _PATTERN_MONTH_YEAR.finditer(text):
        if _is_consumed(m.span()):
            continue
        try:
            mname = m.group(1).lower()
            month = _MONTH_NAME_TO_NUM.get(mname)
            if not month:
                continue
            year = int(m.group(2))
            s, e = _interval_for_month(year, month)
            _add(m.span(), s, e, m.group(0))
        except Exception:
            continue

    return out
