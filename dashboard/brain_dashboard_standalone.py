#!/usr/bin/env python3
"""Standalone Brain Dashboard — entry point.

Serves the dashboard HTML on port 47303 (override via $DASHBOARD_PORT) and
proxies live data via the brain daemon TCP socket. Falls back to direct
read-only SQLite when the daemon is unreachable — the dashboard never
crashes because the brain is down.

Start: python3 dashboard/brain_dashboard_standalone.py

All real code lives in the `dashboard` package — this file exists to keep
the launch path (.claude/launch.json, test_time_window_contract.py, the
eval longmem fresh_brain runner) stable while the implementation is split
across `server.py`, `daemon_client.py`, `db.py`, `clock.py`, `queries/*`,
and `static/*`.
"""

if __name__ == "__main__":
    # Support both `python dashboard/brain_dashboard_standalone.py` (script)
    # and `python -m dashboard.brain_dashboard_standalone` (module). The first
    # invocation runs without `dashboard` on sys.path, so add it explicitly.
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    from dashboard.server import run
    run()
