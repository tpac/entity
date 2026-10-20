"""Thalamus view — the brain's standing-intent queue, read-only.

The sibling of queries/self_channel: that module is sessions speaking to each
other (ephemeral, consume-once, TTL); this is the brain speaking to its
sessions (durable, windowed, ledgered). One read surface over brain_logs.db:

  - query_items: every item with its DELIVERY LEDGER folded in — who saw it,
    when, and at which moment (prompt/stop). The daemon's own `thalamus_list`
    returns delivery COUNTS; the dashboard's whole job here is the fan-out
    (an item delivered to four sessions and still unanswered is the picture
    the counts flatten), so it reads the ledger directly, the same shape
    self_channel folds self_delivered into the courier log — and it keeps
    working while the daemon is down.

Open items are never windowed out: an ask filed three weeks ago that nobody
answered is exactly what the operator needs to see. `hours` bounds the CLOSED
tail only.

Read-only, like every queries.* module. RESOLVING is a write — it goes through
the daemon (thalamus_resolve), never from here.
"""

import json

from ..clock import utc_cutoff
from ..db import logs_db_path
from ..query import safe_query

_ITEM_COLS = (
    "id, source, body, refs, audience, target_session, needs_answer, "
    "dedup_key, deliver_at, expires_at, state, answer, answered_at, "
    "created_at, updated_at, armed_epoch"
)

_OPEN = 'open'
# servers/channels/thalamus/thalamus_contract.LIST_ORDER_SQL, copied — the
# disconnection contract forbids importing it; tests/test_thalamus.py pins the
# copy. Open first; then what is due, most recently due first; then what is
# scheduled, soonest first. Both `?` bind now.
_LIST_ORDER_SQL = ("(state <> 'open'), (COALESCE(deliver_at, created_at) > ?), "
                   "CASE WHEN COALESCE(deliver_at, created_at) > ? "
                   "THEN COALESCE(deliver_at, created_at) END, "
                   "COALESCE(deliver_at, created_at) DESC, id")


@safe_query('queries.thalamus', logs_db_path)
def query_items(conn, hours: int = 168, include_closed: bool = True,
                limit: int = 100):
    """The queue: items + their per-session delivery ledger.

    Open items always appear regardless of `hours`; closed ones (answered /
    dismissed / withdrawn / expired / sent) only inside the window. In
    thalamus_list's order (_LIST_ORDER_SQL): open first, then what is due,
    then what is scheduled — the operator's priority order."""
    cutoff = utc_cutoff(hours=hours)
    if include_closed:
        where, params = "WHERE state = ? OR created_at > ?", [_OPEN, cutoff]
    else:
        where, params = "WHERE state = ?", [_OPEN]
    now = utc_cutoff()
    rows = conn.execute(
        "SELECT %s FROM thalamus_items %s ORDER BY %s LIMIT ?"
        % (_ITEM_COLS, where, _LIST_ORDER_SQL),
        params + [now, now, int(limit)]).fetchall()

    # Delivery fan-out per item (one row per session per armed epoch).
    delivered = {}
    for iid, sid, at, via, epoch in conn.execute(
            "SELECT item_id, session_id, delivered_at, via, armed_epoch "
            "FROM thalamus_deliveries ORDER BY delivered_at").fetchall():
        delivered.setdefault(iid, []).append({
            "session_id": sid or '',
            "delivered_at": at,
            "via": via or '',
            "armed_epoch": epoch or 0,
        })

    out = []
    for r in rows:
        item = dict(zip([c.strip() for c in _ITEM_COLS.split(',')], r))
        try:
            item["refs"] = json.loads(item["refs"]) if item["refs"] else []
        except (ValueError, TypeError):
            item["refs"] = []
        item["needs_answer"] = bool(item["needs_answer"])
        item["deliveries"] = delivered.get(item["id"], [])
        out.append(item)
    return out
