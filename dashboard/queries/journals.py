"""Encoder journals — the one reader for `journal_note` traces, all scales.

Every journaling encoder (S1 Scribe + the four S2 units) writes the SAME shape
through `brain.write_journal_notes`: one trace row per note, `event_type='delta'`,
`ref_type='journal_note'`, subject in `ref_id`, `{note, tag, undelivered}` in
metadata (`undelivered` = the Thalamus door's reason when an addressed tell/ask
line was refused and kept as residue). This module is the single dashboard-side
reader of that shape.

Two doors, one row formatter:

  * `notes_by_chain(conn, chain_ids)` — takes an ALREADY-OPEN logs_db conn, for
    callers enriching run cards (queries.encoding, queries.s2_runs) that are
    mid-query and must not open a second connection.
  * `query_journal_notes(...)` — owns its own connection, for the standalone
    Journals view and the per-card peek.

Adding an encoder needs no new reader.
"""

import json
import re

from ..clock import utc_cutoff
from ..db import logs_db_path
from ..query import safe_query

# Note prose is capped at JOURNAL_NOTE_LIMIT (600) server-side; no second cap
# here — a note is already terse by contract, and re-truncating for display
# would hide the tail of exactly the notes worth reading.

# ── Tag normalization (the read-time tolerance the write side assumes) ──
# `tag` is the grouping key, contracted as one word. But rows written while the
# read side rendered a persisting item's head inline (`open ×3 since 08-17 ·
# subject · note`) carry that head echoed back in the tag — trace_contract
# deliberately tolerates the echo rather than spending prompt budget teaching
# encoders not to ("a tolerant reader costs nothing"). `resolve_target`
# supplies that tolerance for the subject field; nothing did for the tag, so
# live data carries 25 variants of `open` and the grouping key groups nothing.
#
# Splitting it here is both the fix and a free signal: ×N and the since-date
# are the escalation data (JOURNAL_OPEN_NUDGE_RUNS = 5), so the view can rank
# by how long something has been open without a second source.
_TAG_LIFECYCLE = re.compile(
    r'^(?P<tag>\S+)'
    r'(?:\s*[×x]\s*(?P<runs>\d+))?'
    r'(?:\s+since\s+(?P<since>[0-9./-]+))?'
    r'\s*$'
)


def split_tag(raw: str):
    """`'open ×3 since 08-17'` → `('open', 3, '08-17')`.

    Returns (tag, open_runs, since) with open_runs=0 / since='' when the tag is
    a plain word. An unparseable tag returns as-is with no lifecycle — never
    dropped, so a novel tag shape stays visible instead of vanishing into a
    normalization hole.
    """
    t = (raw or '').strip()
    if not t:
        return '', 0, ''
    m = _TAG_LIFECYCLE.match(t)
    if not m:
        return t, 0, ''
    return (m.group('tag'),
            int(m.group('runs')) if m.group('runs') else 0,
            m.group('since') or '')


def journal_unit(chain_id: str) -> str:
    """Which encoder wrote this note, from its chain_id.

    Chain formats are contract (CLAUDE.md → SessionContext): S1 encode is
    `s1e-{session_short}-{stop}`, S2 is `s2-{YYYYMMDDHHMMSS}-{unit}`. So the
    unit is the leading token at S1 and the trailing one at S2. Unknown shapes
    return '' rather than guessing — the UI groups those under "other".
    """
    if not chain_id:
        return ''
    if chain_id.startswith('s1e-'):
        return 's1e'
    if chain_id.startswith('s2-'):
        parts = chain_id.split('-')
        return parts[-1] if len(parts) >= 3 else ''
    return ''


def _note_row(row) -> dict:
    """Format one trace row into a note dict.

    Row order is fixed by the two SELECTs below:
      (id, chain_id, scale, session_id, ref_id, metadata, created_at)
    A row whose metadata is missing/corrupt or whose note is empty returns
    None — the same emptiness gate `build_journal_note_metadata` applies on
    the write side, so reader and writer agree on what counts as a note.
    """
    try:
        meta = json.loads(row[5]) if row[5] else {}
    except (ValueError, TypeError):
        return None
    if not isinstance(meta, dict):
        return None
    note = (meta.get('note') or '').strip()
    if not note:
        return None
    raw_tag = (meta.get('tag') or '').strip()
    tag, open_runs, since = split_tag(raw_tag)
    return {
        'id': row[0],
        'chain_id': row[1] or '',
        'scale': row[2] or '',
        'unit': journal_unit(row[1] or ''),
        'session_id': row[3] or '',
        'subject': row[4] or '',
        'tag': tag,
        'tag_raw': raw_tag,
        'open_runs': open_runs,
        'since': since,
        'note': note,
        'undelivered': (meta.get('undelivered') or '').strip(),
        'created_at': row[6] or '',
    }


_SELECT = ('SELECT id, chain_id, scale, session_id, ref_id, metadata, created_at '
           "FROM trace_events WHERE ref_type = 'journal_note'")


def notes_by_chain(conn, chain_ids) -> dict:
    """{chain_id: [note, ...]} for the given chains, oldest note first.

    `conn` is a caller-owned read-only logs_db connection. Chains with no
    notes are absent from the result (callers use `.get(chain, [])`), so a
    clean run renders no journal section rather than an empty one.
    """
    ids = [c for c in (chain_ids or []) if c]
    if not ids:
        return {}
    out: dict = {}
    # Chunked IN (...) — SQLITE_MAX_VARIABLE_NUMBER is 999 on older builds and
    # the encoding feed pulls up to 200 chains per poll.
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        rows = conn.execute(
            '%s AND chain_id IN (%s) ORDER BY id' % (_SELECT, ','.join('?' * len(chunk))),
            chunk,
        ).fetchall()
        for r in rows:
            note = _note_row(r)
            if note:
                out.setdefault(note['chain_id'], []).append(note)
    return out


@safe_query('queries.journals', logs_db_path)
def query_journal_notes(conn, hours: int = 48, unit: str = '', tag: str = '',
                        limit: int = 300):
    """The journal feed — newest first, across every encoder.

    Filters compose, and ALL of them run in SQL — before the LIMIT: a tag
    filter applied after the LIMIT would see only the newest N rows and
    disagree with the summary's window-wide counts.

    `unit` filters on the encoder (s1e / consolidation / community_detection /
    healer / aspect_integration) via a chain_id LIKE, since the unit lives in
    the chain_id rather than its own column.

    `tag` matches the NORMALIZED tag: the stored value may carry a lifecycle
    suffix (`open ×3 since 08-17`, see split_tag), so an exact comparison
    would miss most `open` rows and a bare LIKE prefix would fold `still-open`
    into `open`. Matching `= tag` OR `LIKE tag || ' %'` accepts the plain verb
    and any lifecycle variant of it while keeping distinct verbs distinct.
    """
    conditions = ['created_at > ?']
    params = [utc_cutoff(hours=hours)]
    if unit:
        # s1e sits at the FRONT of its chain_id, S2 units at the end.
        conditions.append('chain_id LIKE ?')
        params.append((unit + '-%') if unit == 's1e' else ('%-' + unit))
    if tag:
        want = tag.strip()
        conditions.append(
            "(json_extract(metadata, '$.tag') = ? "
            " OR json_extract(metadata, '$.tag') LIKE ? || ' %')")
        params.extend([want, want])
    sql = '%s AND %s ORDER BY created_at DESC, id DESC LIMIT ?' % (
        _SELECT, ' AND '.join(conditions))
    rows = conn.execute(sql, params + [limit]).fetchall()
    return [n for n in (_note_row(r) for r in rows) if n]


@safe_query('queries.journals', logs_db_path, default={})
def query_journal_summary(conn, hours: int = 48):
    """Counts for the Journals view's filter chrome: notes per unit, per tag.

    Cheap aggregate over the same rows the feed reads, so the operator sees
    "which encoder is talking, and about what kind of thing" before filtering.
    Tag counting happens in Python because the tag lives inside the metadata
    JSON — the row volume here is small (hundreds), so this is not worth an
    expression index.
    """
    rows = conn.execute(
        '%s AND created_at > ?' % _SELECT, (utc_cutoff(hours=hours),)
    ).fetchall()
    units: dict = {}
    tags: dict = {}
    total = 0
    # Long-lived open items — the escalation signal. An item at or past
    # JOURNAL_OPEN_NUDGE_RUNS runs is one the encoder has been re-flagging
    # without resolution; those are the ones that need an operator, and the
    # closed-loop failure this view exists to break.
    longest: dict = {}
    for r in rows:
        note = _note_row(r)
        if not note:
            continue
        total += 1
        units[note['unit'] or 'other'] = units.get(note['unit'] or 'other', 0) + 1
        if note['tag']:
            tags[note['tag']] = tags.get(note['tag'], 0) + 1
        runs = note['open_runs']
        if runs:
            prior = longest.get(note['subject'])
            if not prior or runs > prior['open_runs']:
                longest[note['subject']] = note
    standing = sorted(longest.values(), key=lambda n: -n['open_runs'])[:12]
    return {'total': total, 'units': units, 'tags': tags, 'standing': standing}
