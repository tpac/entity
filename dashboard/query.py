"""@safe_query — one owner for the shape every dashboard query shares.

Every query shares one shape (ro_connect + None guard + try/except + warn());
the decorator owns it:

    @safe_query('queries.X', logs_db_path, default=[])
    def query_X(conn, ...):
        rows = conn.execute(SQL, params).fetchall()
        return [shape(r) for r in rows]

The decorator:
  - Opens the read-only SQLite connection
  - If the DB file is missing OR the open failed → returns `default` quietly
  - If the wrapped function raises → logs via warn() AND returns `default`
  - Otherwise returns the function's value

This intentionally preserves graceful degradation (the dashboard never
crashes a panel because one query failed), but makes every failure visible
in stderr — loud-by-default discipline applied at the substrate.

Why this isn't a context manager:
  - Most query functions take args beyond `conn`; a decorator gives clean
    `query_X(arg1, arg2)` signatures while still owning lifecycle.
  - One function = one boundary of concern; this is what `@` is for.

The `db_path_fn` argument is a CALLABLE returning the path, not the path
itself, so $BRAIN_DB_DIR changes between import time and call time are
honored (eval brain harnesses repoint the dir per test).
"""

from functools import wraps
from typing import Any, Callable

from .db import ro_connect
from .log import warn


def safe_query(component: str, db_path_fn: Callable[[], str], default: Any = None):
    """Decorate a query function so it:
      - Opens `ro_connect(db_path_fn())` and passes the conn as the first arg.
      - Catches every exception, logs it via `warn(component, ...)`, and
        returns `default` (defaults to `[]` if you pass `None`).
      - Returns `default` when the DB file is missing.

    The decorated function MUST accept `conn` as its first positional argument.
    """
    if default is None:
        default = []

    def decorator(fn: Callable):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                with ro_connect(db_path_fn()) as conn:
                    if conn is None:
                        return default
                    return fn(conn, *args, **kwargs)
            except Exception as e:
                # Caught OUTSIDE the with-block on purpose: if ro_connect
                # itself raises (rare, but e.g. permission denied), we still
                # want the loud-stderr line + graceful degradation.
                warn(component, '%s failed' % fn.__name__, exc=e)
                return default
        return wrapped
    return decorator
