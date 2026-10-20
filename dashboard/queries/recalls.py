"""S1 recall events — the live decoding feed.

Reads `trace_events`: joins O / K / delta rows by chain_id to produce one
summary per recall.

Cursor is an ISO timestamp (`since_ts`): `trace_events.id` is unordered 8-char
hex; `created_at` is monotonic per writer.
"""

import json

from ..db import chain_payload_files, logs_db_path, read_payload_pointer
from ..query import safe_query
from ._meta import extract_meta_fields, S0_SESSION_STAMP_FIELDS


_EMPTY_STAMP = tuple('' for _ in S0_SESSION_STAMP_FIELDS)


def _turn_stamps(conn, chain_ids):
    """The S0 session stamp (model, host) of the TURN each S1 recall chain
    belongs to, as {s1r_chain_id: (model, host)}. The stamp lives on the S0
    rows, not on S1 — the recall chain `s1r-<short>-<stop>` and its turn's
    `s0-<short>-<stop>` share the suffix, so the join is a chain-id rewrite,
    done for the whole page in ONE query (this feeds a 2s-polled feed). The
    assistant_message row wins over user_message: at Stop the transcript names
    THIS turn's model, while the prompt-time row can only carry the previous
    turn's. Chosen in Python, not SQL — a text predicate on `metadata` would
    also match the turn's own content. Empty strings for a chain from before
    the stamp existed."""
    s0_by_s1 = {c: 's0-' + c[len('s1r-'):] for c in chain_ids
                if c and c.startswith('s1r-')}
    out = {c: _EMPTY_STAMP for c in chain_ids}
    if not s0_by_s1:
        return out
    s0_ids = list(s0_by_s1.values())
    best = {}   # s0 chain → (rank, created_at, stamp)
    for chain, ref_type, created_at, meta in conn.execute(
        "SELECT chain_id, ref_type, created_at, metadata FROM trace_events "
        "WHERE scale = 's0' AND ref_type IN ('assistant_message', 'user_message') "
        "AND chain_id IN (%s)" % ','.join('?' * len(s0_ids)), s0_ids,
    ).fetchall():
        stamp = extract_meta_fields(meta, *S0_SESSION_STAMP_FIELDS)
        if not any(stamp):
            continue
        key = (ref_type == 'assistant_message', created_at or '')
        if chain not in best or key > best[chain][0]:
            best[chain] = (key, stamp)
    for s1, s0 in s0_by_s1.items():
        if s0 in best:
            out[s1] = best[s0][1]
    return out


def read_judge_payload(chain_id: str = "", pointer: str = "",
                       scan_chain: bool = True):
    """Read one recall's judge data (surface prompt + output).

    Primary: `pointer` — the db_dir-relative payload pointer the surface
    threads into its K trace (payload_pointer); one O(1) file read,
    daemon-down safe. Fallback for pointerless rows on card expand: scan
    the chain's payload dir ([-1] = newest attempt via payload_sort_key).
    The POLLED feed passes scan_chain=False — the pointer exists so the feed
    never scans the payloads tree per row every 2s.
    Returns (prompt, output) — (None, None) when nothing exists."""
    raw = read_payload_pointer(pointer) if pointer else None
    if raw is None and scan_chain and chain_id:
        files = chain_payload_files(chain_id, kind='judge')
        if files:
            try:
                with open(files[-1], encoding='utf-8', errors='replace') as f:
                    raw = f.read()
            except OSError:
                raw = None
    if raw is None:
        return None, None
    try:
        data = json.loads(raw)
        return (data.get("surface_prompt"), data.get("surface_output"))
    except Exception:
        return None, None


def query_recall_prompt(chain_id: str = "", pointer: str = ""):
    """Lazy-load one recall's full surface prompt on card expand.

    Split out of query_recall_log so the polled decoding feed stays small: the
    ~35KB prompt is fetched only when the operator expands a card's "Show
    Prompt". Keyed by the row's payload pointer (chain scan as fallback —
    fine for a one-shot expand).
    Returns {"judge_prompt": str} or {"error": str} for the UI fallback."""
    prompt, _out = read_judge_payload(chain_id, pointer=pointer)
    if not prompt:
        return {"error": "no judge payload for %s (pruned, or recording "
                         "gated off)" % (pointer or chain_id)}
    return {"judge_prompt": prompt}


@safe_query('queries.recalls', logs_db_path)
def query_recall_log(conn, since_ts: str = '', limit: int = 50):
    """Read recall events from S1 traces, joined into one row per chain."""
    conditions = ["scale = 's1'", "event_type = 'O'", "ref_type = 'recall'"]
    params = []
    if since_ts:
        conditions.append("created_at > ?")
        params.append(since_ts)
    where = ' AND '.join(conditions)
    rows = conn.execute(
        "SELECT id, chain_id, summary, metadata, session_id, created_at "
        "FROM trace_events WHERE %s ORDER BY created_at DESC LIMIT ?" % where,
        params + [limit],
    ).fetchall()

    stamps = _turn_stamps(conn, [r[1] for r in rows])
    results = []
    for r in rows:
        trace_id = r[0]
        chain_id = r[1]
        summary = r[2] or ''
        session_id_row = r[4] or ''
        timestamp = r[5] or ''

        candidates = []
        query = ''
        candidate_count = 0
        try:
            if summary and 'candidates for:' in summary:
                candidate_count = int(summary.split(' candidates')[0])
                query = summary.split('for: ', 1)[1] if 'for: ' in summary else ''
        except (ValueError, IndexError):
            pass
        try:
            meta = json.loads(r[3]) if r[3] else {}
            if not query:
                query = meta.get('query', '')
            for cand_str in meta.get('candidates', []):
                parts = cand_str.split('|')
                if len(parts) >= 4:
                    candidates.append({
                        'id': parts[0],
                        'title': '|'.join(parts[1:-2]),
                        'score': parts[-2],
                        'type': parts[-1],
                    })
        except Exception:
            pass

        # K event (judge-selected + activation expansion) in the same chain.
        # ref_id is the JSON-encoded list of short ids the Haiku judge picked;
        # metadata.activations is the spread-activation expansion (post-
        # selection neighbors that lit up enough to enter additionalContext).
        # Both are short (8-char) ids — graph node ids share that format, so
        # they reconcile directly with no remapping.
        selected_ids = []
        activation_ids = []
        # surface_selected only: the chain also carries the tool moment's
        # tool_shown K rows (memories shown at a file's first edit).
        k_row = conn.execute(
            "SELECT ref_id, summary, metadata FROM trace_events "
            "WHERE chain_id = ? AND event_type = 'K' "
            "AND ref_type = 'surface_selected'",
            (chain_id,),
        ).fetchone()
        judge_pointer = ''
        if k_row:
            try:
                selected_ids = json.loads(k_row[0]) if k_row[0] else []
            except Exception:
                pass
            try:
                k_meta = json.loads(k_row[2]) if k_row[2] else {}
                activation_ids = [
                    entry.get('id') for entry in (k_meta.get('activations') or [])
                    if entry.get('id')
                ]
                # db_dir-relative pointer to this recall's judge payload —
                # the O(1) read key for the polled feed and card expand.
                judge_pointer = k_meta.get('payload_pointer') or ''
            except Exception:
                pass

        # Δ event (additionalContext) in same chain
        judge_output = None
        d_row = conn.execute(
            "SELECT metadata FROM trace_events "
            "WHERE chain_id = ? AND event_type = 'delta'",
            (chain_id,),
        ).fetchone()
        if d_row:
            try:
                d_meta = json.loads(d_row[0]) if d_row[0] else {}
                judge_output = d_meta.get('content', '')
            except Exception:
                pass

        # scan_chain=False: this is the 2s-polled feed — pointer only, never a
        # per-row payloads-tree scan.
        j_prompt, j_output_file = read_judge_payload(
            chain_id, pointer=judge_pointer, scan_chain=False)
        titles = {c['id']: c['title'] for c in candidates}
        model, host = stamps.get(chain_id, _EMPTY_STAMP)

        results.append({
            "id": trace_id,
            "chain_id": chain_id,
            "payload_pointer": judge_pointer,
            "session_id": session_id_row,
            "query": query,
            "returned_ids": [c['id'] for c in candidates],
            "returned_count": candidate_count or len(candidates),
            "titles": titles,
            "snippets": {},
            "timestamp": timestamp,
            "source": "hook",
            "used_ids": selected_ids,
            "used_count": len(selected_ids),
            "activation_ids": activation_ids,
            # judge_prompt (the full surface prompt, ~35KB each) is NOT shipped
            # in the list — it was 75% of a 2.3MB polled payload, hidden behind
            # "Show Prompt" anyway. The frontend lazy-loads it per-card on expand
            # via query_recall_prompt(chain_id, pointer). has_prompt tells the UI
            # whether to render the button without shipping the bytes.
            "has_prompt": bool(j_prompt),
            # judge_output (the additionalContext) STAYS inline — it's the card's
            # displayed content, and it's an order of magnitude smaller (~8KB).
            # Prefer the judge payload over trace metadata — the trace
            # truncates at 4000 chars, but the payload holds the full context.
            "judge_output": j_output_file or judge_output,
            # The turn's model/host — the Live card's label for which model
            # this turn ran on (joined from the S0 row, see _turn_stamps).
            "model": model,
            "host": host,
        })
    return results
