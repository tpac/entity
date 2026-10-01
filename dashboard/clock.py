"""ISO-format timestamp helpers for the dashboard.

Mirrors servers/clock.py semantics: brain stores ISO-T with the
trailing +00:00 zone marker, NOT SQLite's native space-separated
wall-clock string. Time-window queries must use this format or lex
comparison silently breaks at midnight.
"""

from datetime import datetime, timezone, timedelta


def utc_cutoff(hours: float = 0, minutes: float = 0, days: float = 0) -> str:
    """Return an ISO cutoff timestamp compatible with stored created_at values."""
    dt = datetime.now(timezone.utc) - timedelta(hours=hours, minutes=minutes, days=days)
    return dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')

