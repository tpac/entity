"""Encoding runs.

`query_encoding_runs`: one card per S1E run, with the
nodes and edges the run recorded in its delta trace; `query_encoding_prompt`
lazy-loads the run's prompt from its payload pointer.
"""

import json

from ..clock import utc_cutoff
from ..db import (brain_db_path, fetch_by_id, logs_db_path,
                  read_payload_pointer, ro_connect)
from ..log import warn
from ..query import safe_query
from .journals import notes_by_chain


@safe_query('queries.encoding', logs_db_path)
def _query_encoding_chains(conn, limit: int, hours: int):
    """Phase 1 of query_encoding_runs: pull S1E chain skeletons from logs_db.

    Split into its own decorated function so the logs-db connection is
    managed cleanly by @safe_query. Brain-db enrichment happens in a second
    pass (manual ro_connect) below, because @safe_query is single-DB.
    """
    conditions = [
        "scale = 's1'",
        "event_type = 'O'",
        "ref_type = 'encoding_prompt'",
        "created_at > ?",
    ]
    params = [utc_cutoff(hours=hours)]
    where = ' AND '.join(conditions)
    chain_rows = conn.execute(
        "SELECT chain_id, ref_id, summary, metadata, session_id, created_at "
        "FROM trace_events WHERE %s ORDER BY created_at DESC LIMIT ?" % where,
        params + [limit],
    ).fetchall()

    runs = []
    for row in chain_rows:
        chain_id = row[0]
        prompt_info = row[2] or ''
        session = row[4] or ''
        timestamp = row[5] or ''
        # Chain ID format: s1e-{session_short}-{counter}. Trailing token is
        # the counter. chain_ids are formatted strings (unlike the hex
        # trace_events.id), so a string-split is the parse.
        counter = chain_id.split('-')[-1] if chain_id else ''

        k_row = conn.execute(
            "SELECT summary, ref_id FROM trace_events "
            "WHERE chain_id = ? AND event_type = 'K'",
            (chain_id,),
        ).fetchone()
        catalog_info = k_row[0] if k_row else ''

        # The authoritative run summary is the `encoding_run` delta — it carries
        # the created/revised rollup + the "N actions" line. S1E also writes
        # many *other* delta rows per chain (one edge_relation_revised per edge,
        # node_revised per node, journal_note per residue item), so an
        # unfiltered `event_type='delta'` + fetchone() would grab an edge
        # revision with no created/revised and show "0 actions" for a run that
        # wrote. Filter to encoding_run.
        d_row = conn.execute(
            "SELECT summary, created_at, metadata FROM trace_events "
            "WHERE chain_id = ? AND event_type = 'delta' "
            "AND ref_type = 'encoding_run'",
            (chain_id,),
        ).fetchone()
        summary = d_row[0] if d_row else '(encoding in progress or no actions)'
        delta_ts = d_row[1] if d_row else ''

        # The run records the exact node ids it created/revised in its delta
        # metadata (build_delta_metadata, populated for both remember_batch and
        # brain_batch). Read that authoritative list rather than reconstructing
        # it from encoding_source + a time window — the reconstruction drifts the
        # moment the tag or the run duration changes; the trace never does.
        created_ids, revised_ids = [], []
        if d_row and d_row[2]:
            try:
                dmeta = json.loads(d_row[2])
                created_ids = [i for i in (dmeta.get('created') or []) if i]
                revised_ids = [i for i in (dmeta.get('revised') or []) if i]
            except (ValueError, TypeError):
                pass

        # NOTE: the full encoder prompt (user_content) is deliberately NOT read
        # here. A long session's prompt is hundreds of KB; shipping it inline for
        # every run in the polled list bloated the response into the multi-MB
        # range and broke the browser transfer (BrokenPipe → empty S1 panel).
        # The frontend lazy-loads it per-run on card expand via
        # query_encoding_prompt(chain_id), which re-derives the file from the
        # chain — so nothing prompt-related rides along in this list row.
        runs.append({
            "chain_id": chain_id,
            "counter": counter,
            "start_ts": timestamp,
            "delta_ts": delta_ts,
            "session_id": session,
            "summary": summary[:500],
            "prompt_info": prompt_info,
            "catalog_info": catalog_info,
            "created_ids": created_ids,
            "revised_ids": revised_ids,
            "nodes": [],
            "edges": [],
        })

    # Journal notes — the encoder's residue (why/doubt/next). Read through the
    # one journal reader (queries.journals) on the connection we already hold,
    # so S1E and every S2 unit surface the same shape from the same SQL. The
    # card renders them FIRST in the details, above the nodes/edges, so the
    # operator reads the encoder's mind before its hands.
    # Isolated: this whole function is @safe_query-wrapped, so an exception
    # here would return [] for the entire encoding feed. Residue is worth
    # losing on its own; the runs are not.
    try:
        by_chain = notes_by_chain(conn, [r['chain_id'] for r in runs])
    except Exception as e:
        warn('queries.encoding', 'journal notes unavailable', exc=e)
        by_chain = {}
    for run in runs:
        run['journal_notes'] = by_chain.get(run['chain_id'], [])
    return runs


@safe_query('queries.encoding', logs_db_path, default={})
def query_encoding_prompt(conn, chain_id: str = ""):
    """Lazy-load one run's full encoder prompt on card expand.

    Split out of query_encoding_runs so the polled list stays small: the prompt
    body is fetched only when the operator expands a specific run's card. Keyed
    by chain_id → the run's encoding_prompt O trace → its ref_id, a db_dir-
    relative payload pointer (brain.record_payload) read directly from
    {BRAIN_DB_DIR}/payloads/ — daemon-down safe.
    Returns {"user_content": str} or {"error": str} for the UI fallback."""
    if not chain_id:
        return {"error": "chain_id required"}
    # ORDER BY created_at DESC: a retried run reuses the chain and writes a
    # second O row with its own per-attempt pointer — the card must show the
    # newest attempt's prompt.
    row = conn.execute(
        "SELECT ref_id FROM trace_events "
        "WHERE chain_id = ? AND scale = 's1' AND event_type = 'O' "
        "AND ref_type = 'encoding_prompt' "
        "ORDER BY created_at DESC LIMIT 1",
        (chain_id,),
    ).fetchone()
    ref = row[0] if row else ''
    if not ref:
        return {"error": "no prompt payload for chain %s (recording gated "
                         "off, or trace missing)" % chain_id}
    content = read_payload_pointer(ref)
    if content is None:
        return {"error": "payload pruned (or deleted from payloads/)"}
    return {"user_content": content}


def query_encoding_runs(limit: int = 10, hours: int = 24):
    """S1E runs — one chain per run, enriched with the exact nodes the run
    recorded, plus the edges touching them.

    Two-pass: chain skeletons from logs_db (via @safe_query), then per-run
    enrichment from brain.db (manual second connection, @safe_query is single-DB).

    The nodes shown are the precise ids the run recorded in its delta trace
    (`created`/`revised`), fetched by id — NOT reconstructed from
    encoding_source + a time window. The trace is the run's own statement of what
    it did, so the view can't drift when tagging changes (e.g. a run writing via
    brain_batch instead of remember_batch) or when a run outlasts any fixed
    window. A run that recorded no node writes shows empty — truthfully.
    """
    runs = _query_encoding_chains(limit, hours)
    if not runs:
        return runs
    # Strip the internal id scaffolding up front so it never leaks into the
    # response on the bconn-None / exception paths (the per-run loop below
    # consumes from this map instead of from the run dict).
    ids_by_chain = {
        r['chain_id']: ((r.pop('created_ids', []) or []), (r.pop('revised_ids', []) or []))
        for r in runs
    }
    try:
        with ro_connect(brain_db_path()) as bconn:
            if bconn is None:
                return runs
            for run in runs:
                created_ids, revised_ids = ids_by_chain.get(run['chain_id'], ([], []))
                node_ids = created_ids + revised_ids
                if not node_ids:
                    continue  # nothing recorded for this run — leave it empty

                by_id = fetch_by_id(
                    bconn, 'nodes',
                    'id, type, title, substr(content,1,200), created_at, encoding_source',
                    node_ids)

                run['nodes'] = []
                seen = set()
                for nid in created_ids:
                    r = by_id.get(nid)
                    if r and nid not in seen:
                        seen.add(nid)
                        run['nodes'].append({
                            "id": r[0], "type": r[1], "title": r[2],
                            "content": r[3], "timestamp": r[4], "encoding_source": r[5],
                        })
                for nid in revised_ids:
                    r = by_id.get(nid)
                    if r and nid not in seen:
                        seen.add(nid)
                        run['nodes'].append({
                            "id": r[0], "type": r[1], "title": r[2],
                            "content": r[3], "timestamp": r[4],
                            "encoding_source": r[5], "kind": "revised",
                        })

                # Edges the run formed: those touching its own node set, bounded
                # by the run's real span (start trace → delta trace). Derived from
                # the authoritative node set, so no fixed-window or tag guessing.
                start_ts = run.get('start_ts', '')
                end_ts = run.get('delta_ts') or start_ts
                if start_ts and end_ts:
                    nid_ph = ','.join('?' * len(node_ids))
                    edges = bconn.execute(
                        "SELECT er.claim_source_id, er.claim_target_id, er.relation, e.weight, e.created_at, "
                        "n1.title, n2.title "
                        "FROM edges e "
                        "JOIN edge_relations er ON er.edge_id = e.edge_id "
                        "LEFT JOIN nodes n1 ON n1.id = er.claim_source_id "
                        "LEFT JOIN nodes n2 ON n2.id = er.claim_target_id "
                        "WHERE e.created_at BETWEEN ? AND ? "
                        "AND er.archived = 0 "
                        
                        "AND (e.source_id IN (%s) OR e.target_id IN (%s)) "
                        "ORDER BY e.created_at" % (nid_ph, nid_ph),
                        [start_ts, end_ts] + node_ids + node_ids,
                    ).fetchall()
                    # Ends ride along by id: the graph lights the memories a
                    # finished run wove its new ones into (its own new nodes
                    # aren't in the graph it loaded before they existed).
                    run['edges'] = [
                        {"source_id": e[0], "target_id": e[1],
                         "relation": e[2], "weight": e[3],
                         "source_title": e[5] or e[0][:12],
                         "target_title": e[6] or e[1][:12],
                         "timestamp": e[4]}
                        for e in edges
                    ]
    except Exception as e:
        warn('queries.encoding', 'enriching runs with nodes/edges failed', exc=e)
    return runs
