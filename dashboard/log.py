"""Single stderr logger + in-memory error ring for the dashboard.

The dashboard intentionally degrades gracefully when something goes wrong —
half the panels would crash the page if any one query raised, so each query
catches its own exceptions and returns an empty result. Swallowing exceptions
silently would hide every failure; this module keeps degradation loud.

This module does two things:

1. **Loud stderr logging** — `warn(component, message, exc=)` prints to
   stderr in a uniform format. The preview console + a shell-launched
   dashboard both see it; failures stop being silent.

2. **In-memory ring buffer** — every warn() also appends to a bounded deque
   the `/api/dashboard-errors` route exposes. This makes the dashboard
   self-monitoring: when something goes wrong, you can see it in the
   Logs tab (Dashboard sub-feed) without tailing terminal output.

Loud-by-default principle: a silent failure is worse than a noisy one.
The ring buffer makes "noisy" cheap — it doesn't fill disk, it doesn't
spam every console; it just sits there waiting to be looked at when a
panel shows blank.
"""

import sys
import time
from collections import deque
from threading import Lock

# Bounded — drops oldest on overflow. 200 entries × a few hundred bytes each
# = ~50KB max footprint. Plenty to catch a regression cluster, small enough
# to ignore.
_RING: deque = deque(maxlen=200)
# Every warn() since start — keeps counting after the ring drops its oldest,
# so the Logs badge can show errors the ring no longer holds.
_TOTAL = 0
# When this process started: a new value tells the badge the count restarted.
STARTED = time.time()
_LOCK = Lock()  # threads (HTTP handler) write; readers iterate snapshots


def warn(component: str, message: str, exc: BaseException = None) -> None:
    """Emit a single line to stderr + append to the ring. Never raises."""
    entry = {
        'ts': time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
        'component': component,
        'message': message,
        'exc_type': type(exc).__name__ if exc is not None else None,
        'exc_text': str(exc) if exc is not None else None,
    }
    # stderr first — even if the ring append throws (it shouldn't), the
    # operator still gets the log line.
    try:
        if exc is None:
            line = '[dashboard.%s] %s' % (component, message)
        else:
            line = '[dashboard.%s] %s: %s: %s' % (
                component, message, entry['exc_type'], exc,
            )
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass
    global _TOTAL
    try:
        with _LOCK:
            _RING.append(entry)
            _TOTAL += 1
    except Exception:
        pass


def recent(limit: int = 100) -> list:
    """Return the most recent `limit` warn entries, newest first.

    Snapshot under the lock so a concurrent warn() can't mutate the list
    mid-iteration. Returned list is a freshly-allocated copy — callers can
    sort / filter / slice freely.
    """
    with _LOCK:
        items = list(_RING)
    items.reverse()  # newest first
    return items[:limit]


def total() -> int:
    """How many warn() calls this process has made."""
    return _TOTAL


def clear() -> None:
    """Drop all entries and the count — a reset for tests; the Logs badge
    tracks what the operator has seen client-side."""
    global _TOTAL
    with _LOCK:
        _RING.clear()
        _TOTAL = 0
