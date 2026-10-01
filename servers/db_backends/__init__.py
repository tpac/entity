"""Database backend plug-points.

The brain currently uses SQLite. The db_maintenance scheduler talks to
whatever backend implements the BackendOps protocol (declared in
servers/db_maintenance.py). To swap SQLite for another store later,
implement a new module in this package and re-export it as `current`.

Beyond that protocol, a backend must also expose `connect_maintenance(db_path)`
— a private connection with a short busy_timeout, for running maintenance
against a LIVE database. It is not on BackendOps because the scheduler never
calls it; the callers are the brain's own sweeps. Never hand such a sweep the
foreground connection: its commits would land on whatever transaction that
connection is holding.
"""

from . import sqlite as current  # noqa: F401 — re-exported

__all__ = ['current']
