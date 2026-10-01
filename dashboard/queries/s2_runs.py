"""S2 unit runs — consolidation, community detection, aspect integration
and healer.

Chain ids look like `s2-{YYYYMMDDHHMMSS}-{unit}`; runs are selected by the unit
substring, since no scale + ref_type combination identifies them.

`_query_s2_unit_runs` owns the shared O/K/delta pull and brain-side
enrichment; each unit declares its enricher. Healer stays separate — it's
logs-only with no brain-side enrichment, a different shape.
"""

import json
from typing import Callable, List

from ..clock import utc_cutoff
from ..db import brain_db_path, fetch_by_id, logs_db_path, ro_connect
from ..log import warn
from .journals import notes_by_chain


# Mirror of servers/trace_contract.RESIDUE_REF_TYPES + EMITTER_REF_TYPES. The
# dashboard disconnection contract forbids importing servers.* (see
# queries/aspects.py), so we replicate. A consistency test pins these against the
# server-side constants — mirror-and-pin, never import.
#
# Both families are event_type='delta' on the unit's chain without being the
# unit's per-RUN integration delta, so an unfiltered pull renders them as phantom
# run cards: residue is what an encoder leaves beside its run (journal notes,
# invocations, checkpoints, thalamus filings); emitter rows are per-WRITE
# mutations (one per node/edge touched).
_RESIDUE_REF_TYPES = ('journal_note', 'journal_invocation',
                      'journal_checkpoint', 'thalamus_filed')
_EMITTER_REF_TYPES = (
    'node_created', 'node_archived',
    'node_revised', 'edge_relation_revised',
    'node_lock_changed',
)
# Everything the run-card queries must not mistake for a run.
_NON_RUN_REF_TYPES = _RESIDUE_REF_TYPES + _EMITTER_REF_TYPES


def _fetch_ok_deltas(conn, unit_keyword: str, hours: int, delta_columns: str,
                     ok_extra_columns: str = ''):
    """Pull (delta_rows, ok_by_chain) for a given S2 unit from logs_db.

    `delta_columns` is the column list selected for delta rows — callers
    differ on whether they need `ref_type`. `ok_extra_columns` lets a
    caller pull extra fields off the O/K rows (e.g. metadata).

    Both queries share the same `chain_id LIKE '%X%'` substring filter
    and the same time-window cutoff. Brittle but works while chain_ids
    are string-keyed.
    """
    since = utc_cutoff(hours=hours)
    excl = ','.join(['?'] * len(_NON_RUN_REF_TYPES))
    delta_rows = conn.execute(
        "SELECT %s FROM trace_events WHERE chain_id LIKE ? "
        "AND event_type = 'delta' AND (ref_type IS NULL OR ref_type NOT IN (%s)) "
        "AND created_at > ? ORDER BY created_at DESC"
        % (delta_columns, excl),
        ('%' + unit_keyword + '%', *_NON_RUN_REF_TYPES, since),
    ).fetchall()
    ok_select = "chain_id, event_type, summary, created_at"
    if ok_extra_columns:
        ok_select += ', ' + ok_extra_columns
    ok_rows = conn.execute(
        "SELECT %s FROM trace_events WHERE chain_id LIKE ? "
        "AND event_type IN ('O', 'K') AND created_at > ? ORDER BY created_at DESC"
        % ok_select,
        ('%' + unit_keyword + '%', since),
    ).fetchall()
    return delta_rows, ok_rows


def _query_s2_unit_runs(unit_keyword: str, hours: int, delta_columns: str,
                        enricher: Callable, ok_extra_columns: str = '',
                        component: str = 'queries.s2_runs') -> List[dict]:
    """Shared skeleton for consolidation + community runs.

    Steps:
      1. Open logs_db, pull O/K and delta rows for `unit_keyword`.
      2. Build a per-chain, time-ordered O/K index + a `nearest_ok(chain,
         delta_ts)` closure that snaps a delta to the O/K immediately
         preceding it.
      3. Open brain_db. For each delta, call `enricher(bconn, delta_row,
         nearest_ok) -> dict` and append to the result list.

    Either DB unreachable → return []. The enricher's job is to produce the
    per-run dict (chain_id, timestamp, summary, etc + unit-specific fields);
    it calls `nearest_ok(chain_id, delta_created_at)` to resolve its run's
    O/K summaries.
    Loud-by-default: any failure logs via warn() before returning [].
    """
    runs: List[dict] = []

    try:
        with ro_connect(logs_db_path()) as conn:
            if conn is None:
                return []
            delta_rows, ok_rows = _fetch_ok_deltas(
                conn, unit_keyword, hours, delta_columns, ok_extra_columns)
            # Journal notes for every chain in one pass, on the connection we
            # already hold. Attached below the enricher loop so no unit's
            # enricher has to know journals exist — every S2 unit gets the
            # same residue shape S1E gets, from the one journal reader.
            #
            # Isolated in its own try: runs are the payload, residue is the
            # garnish. Sharing the outer except meant one bad journal read
            # returned [] for the WHOLE feed — the operator lost every run
            # card because a sub-read failed.
            try:
                journal_by_chain = notes_by_chain(
                    conn, [r[0] for r in delta_rows if r])
            except Exception as e:
                warn(component, '%s journal notes unavailable' % unit_keyword, exc=e)
                journal_by_chain = {}
    except Exception as e:
        warn(component, '%s logs_db pull failed' % unit_keyword, exc=e)
        return []

    # One chain can hold several O/K/delta triples, and keying O/K by chain_id
    # alone would collapse them onto a single (oldest) pair. Build a
    # per-chain, time-ordered O/K list and snap each delta to the O and K that
    # immediately precede it.
    ok_events: dict = {}
    for r in ok_rows:
        chain, et, summary, ts = r[0], r[1], r[2] or '', r[3]
        meta = r[4] if len(r) > 4 else None
        ok_events.setdefault(chain, []).append(
            {'et': et, 'ts': ts, 'summary': summary, 'metadata': meta})
    for evs in ok_events.values():
        evs.sort(key=lambda e: e['ts'])  # ascending; ISO-T strings compare lexically

    def nearest_ok(chain_id: str, delta_ts) -> dict:
        """Most-recent O and K in `chain_id` at or before `delta_ts`.

        Returns {event_type: {'summary', 'metadata'}} — the same shape the
        enrichers already consume via .get('O'/'K', {}).get('summary')."""
        payload: dict = {}
        for e in ok_events.get(chain_id, []):
            if e['ts'] <= delta_ts:
                payload[e['et']] = {'summary': e['summary'], 'metadata': e['metadata']}
            else:
                break  # list is ascending — nothing after this can precede the delta
        return payload

    try:
        with ro_connect(brain_db_path()) as bconn:
            if bconn is None:
                return []
            for delta_row in delta_rows:
                try:
                    run = enricher(bconn, delta_row, nearest_ok)
                    run['journal_notes'] = journal_by_chain.get(
                        run.get('chain_id', ''), [])
                    runs.append(run)
                except Exception as e:
                    # One delta failing shouldn't drop the whole feed.
                    warn(component, '%s enricher failed for chain %s' % (
                        unit_keyword, delta_row[0] if delta_row else '?'), exc=e)
    except Exception as e:
        warn(component, '%s brain_db enrichment failed' % unit_keyword, exc=e)
        return []

    return runs


# ── Consolidation enricher ──────────────────────────────────────────────────

def _short(nid) -> str:
    """8-char id label for a link endpoint we couldn't resolve to a title.
    str() guards against a malformed delta carrying a non-string id."""
    return str(nid)[:8]


def _batch_operations(action: dict) -> list:
    """A brain_batch action's operations as dispatch executed them: a
    JSON-encoded array, or a JSON-encoded element, is decoded (mirrors
    servers/contract.unwrap_operations and dispatch_write._element_as_dict;
    test_dashboard_disconnection pins the two together). An element that
    isn't an op dict stays in place as {} so positions keep matching
    failed_ops indexes."""
    ops = (action.get('input') or {}).get('operations') or []
    if isinstance(ops, str):
        try:
            ops = json.loads(ops)
        except ValueError:
            ops = None
    if not isinstance(ops, list):
        warn('queries.s2_runs', 'brain_batch operations not a list: %.80r'
             % ((action.get('input') or {}).get('operations'),))
        return []
    out = []
    for op in ops:
        if isinstance(op, str):
            try:
                op = json.loads(op)
            except ValueError:
                op = None
        out.append(op if isinstance(op, dict) else {})
    return out


def _deconstruct_consolidation_ops(meta: dict):
    """Read what a consolidation run produced from its OWN delta record.

    Consolidation folds clusters into existing survivors rather than creating
    nodes (consolidation_enrichment_prompt.py: "Consolidation does not create
    new nodes. It strengthens existing ones."), so a run's product is enriched
    survivors, folded-in (archived) originals, and the links it draws between
    nodes it keeps. Only what the write path accepted counts:

      • survivors + archived: the ok-gated top-level `revised`/`archived`
        buckets build_delta_metadata aggregates from each write's `affected`
        (absorb: survivor → revised, absorbed → archived). A refused op lands
        in neither.
      • links: every accepted `connect` op — KEEP/SKIP (`similar_to`),
        SUPERSESSION (`supersedes`), CONTRADICTION (`corrects`), partition
        (`depends_on`), … . Connects aren't bucketed as node pairs, so they
        come from the recorded op input minus the ops the action's
        `failed_ops` reports refused (index None = the whole call).

    Returns (synth_ids, archived_ids, links); links = list of
    {source_id, target_id, relation, description}.
    """
    links = []
    for ad in (meta.get('action_details') or []):
        # A call that answered ok=False wrote nothing, whether or not dispatch
        # listed its refused ops (a refusal before dispatch_command lists none).
        if not isinstance(ad, dict) or ad.get('error'):
            continue
        # index None = the whole call; an entry naming no index refuses nothing.
        refused = {f.get('index', -1) for f in (ad.get('failed_ops') or [])
                   if isinstance(f, dict)}
        if None in refused:
            continue
        for i, op in enumerate(_batch_operations(ad)):
            if op.get('op') != 'connect' or i in refused:
                continue
            rel = op.get('relation')
            src, tgt = op.get('source_id'), op.get('target_id')
            if rel and src and tgt:
                links.append({'source_id': src, 'target_id': tgt,
                              'relation': rel,
                              'description': op.get('description', '') or ''})
    synth_ids = [i for i in (meta.get('revised') or []) if i]
    archived_ids = [i for i in (meta.get('archived') or []) if i]
    return synth_ids, archived_ids, links


def _enrich_consolidation(bconn, delta_row, nearest_ok) -> dict:
    chain_id, summary, meta_raw, created_at = delta_row
    ok_payload = nearest_ok(chain_id, created_at)

    # `final_text` (the whole agent transcript) is deliberately NOT read here:
    # the run's residue now comes from its parsed journal_note rows, attached
    # centrally in _query_s2_unit_runs — same shape S1E shows.
    meta = {}
    if meta_raw:
        try:
            meta = json.loads(meta_raw)
        except (ValueError, TypeError) as e:
            # Loud-by-default: surface a corrupt delta payload (stderr + Logs
            # tab); the card still renders, empty, rather than dropping the run.
            warn('queries.s2_runs',
                 'consolidation delta %s has unparseable metadata'
                 % (chain_id or '?'), exc=e)

    # Trace-authoritative: fetch by id exactly the nodes the run recorded
    # touching — mirroring query_encoding_runs in encoding.py — never a
    # reconstruction from encoding_source + a time window, which credits
    # other writers' edges in the same hour to the run.
    synth_ids, archived_ids, link_specs = _deconstruct_consolidation_ops(meta)

    synthesized, archived_nodes, links = [], [], []
    node_ids = list({*synth_ids, *archived_ids,
                     *(s for k in link_specs
                       for s in (k['source_id'], k['target_id']))})

    if node_ids:
        by_id = fetch_by_id(
            bconn, 'nodes',
            'id, type, title, substr(content,1,500), confidence, archived',
            node_ids)

        # Liveness is the DB `archived` flag, not op ordering. A node that was a
        # survivor in one op and absorbed in a later op (chain merge), or a
        # survivor archived by a later run, is shown as archived — never as a
        # live survivor. The shared `seen` set dedups one node to one section.
        seen = set()
        for nid in synth_ids:
            if nid in seen:
                continue
            r = by_id.get(nid)
            if not r:
                continue
            seen.add(nid)
            node = {"id": r[0], "type": r[1], "title": r[2], "content": r[3]}
            if r[5]:  # archived flag → it didn't survive
                archived_nodes.append(node)
            else:
                node["confidence"] = r[4]
                synthesized.append(node)
        for nid in archived_ids:
            if nid in seen:
                continue
            r = by_id.get(nid)
            if not r:
                continue
            seen.add(nid)
            archived_nodes.append({"id": r[0], "type": r[1], "title": r[2],
                                   "content": r[3]})
        for k in link_specs:
            rs, rt = by_id.get(k['source_id']), by_id.get(k['target_id'])
            links.append({"source": rs[2] if rs else _short(k['source_id']),
                          "target": rt[2] if rt else _short(k['target_id']),
                          "relation": k['relation'],
                          "description": k['description']})

    return {
        "chain_id": chain_id,
        "timestamp": created_at,
        "summary": summary or '',
        "o_summary": ok_payload.get('O', {}).get('summary', ''),
        "k_summary": ok_payload.get('K', {}).get('summary', ''),
        "synthesized": synthesized,
        "archived": archived_nodes,
        "links": links,
    }


def query_consolidation_runs(hours: int = 24):
    """S2 consolidation runs — survivors, folded-in originals and links per run."""
    return _query_s2_unit_runs(
        'consolidation', hours,
        delta_columns='chain_id, summary, metadata, created_at',
        enricher=_enrich_consolidation,
        ok_extra_columns='metadata',
    )


# ── The O/K/Δ base run ──────────────────────────────────────────────────────
# Every S2 unit's card is an O/K/Δ triple at minimum; only what the run
# PRODUCED differs. This is that minimum, and it's the whole card for units
# whose product is already in the summaries (aspect integration). Unit-specific
# enrichers build on it rather than restating the five shared fields.

def _base_run(delta_row, nearest_ok) -> dict:
    """{chain_id, timestamp, summary, ref_type, o_summary, k_summary} from a
    delta row whose first four columns are (chain_id, summary, metadata,
    created_at) and whose optional fifth is ref_type."""
    chain_id, summary, _meta_raw, created_at = delta_row[:4]
    ok_payload = nearest_ok(chain_id, created_at)
    return {
        "chain_id": chain_id,
        "timestamp": created_at,
        "summary": summary or '',
        "ref_type": (delta_row[4] or '') if len(delta_row) > 4 else '',
        "o_summary": ok_payload.get('O', {}).get('summary', ''),
        "k_summary": ok_payload.get('K', {}).get('summary', ''),
    }


def _enrich_ok_delta(bconn, delta_row, nearest_ok) -> dict:
    """Enricher for units with no brain-side product to fetch."""
    return _base_run(delta_row, nearest_ok)


# ── Community enricher ──────────────────────────────────────────────────────
# Community is a bit different: the community-node list is fetched ONCE
# (not per-delta), so the enricher reads it from a closure rather than
# requerying. We bind that via a factory.

def _make_community_enricher(community_list):
    def _enrich(bconn, delta_row, nearest_ok):
        run = _base_run(delta_row, nearest_ok)
        run["communities"] = community_list[:15]
        return run
    return _enrich


def _fetch_community_nodes(bconn) -> list:
    """Pull the 30 most-recent community nodes with their KV metadata."""
    communities = bconn.execute(
        "SELECT id, title, substr(content,1,400), confidence, created_at "
        "FROM nodes WHERE type = 'community' AND archived = 0 "
        "AND encoding_source = 's2:community_detection' "
        "ORDER BY created_at DESC LIMIT 30"
    ).fetchall()
    out = []
    for c in communities:
        cid, title, content, conf, created = c
        meta = dict(bconn.execute(
            "SELECT key, value FROM node_metadata_kv WHERE node_id = ?",
            (cid,),
        ).fetchall())
        member_count = bconn.execute(
            "SELECT COUNT(*) FROM edges e "
            "JOIN edge_relations er ON er.edge_id = e.edge_id "
            "WHERE (e.source_id = ? OR e.target_id = ?) "
            "AND er.relation = 'community_member' "
            "AND er.archived = 0",
            (cid, cid),
        ).fetchone()[0]
        out.append({
            "id": cid, "title": title, "content": content,
            "confidence": conf, "created_at": created,
            "members": member_count,
            "maturity": meta.get('community_maturity', '?'),
        })
    return out


def query_community_runs(hours: int = 24):
    """S2 community detection runs — one entry per delta, sharing a snapshot
    of the 30 most-recent community nodes."""
    # We need to fetch the community node list BEFORE _query_s2_unit_runs
    # opens its brain.db connection, so do it in our own ro_connect first.
    try:
        with ro_connect(brain_db_path()) as bconn:
            if bconn is None:
                return []
            community_list = _fetch_community_nodes(bconn)
    except Exception as e:
        warn('queries.s2_runs', 'community node pre-fetch failed', exc=e)
        return []

    # Dedupe across deltas: render each chain once. This mirrors the prior
    # behavior (seen_chains set) and prevents the same community list from
    # repeating once per delta event in the same chain.
    seen = set()
    runs = _query_s2_unit_runs(
        'community_detection', hours,
        delta_columns='chain_id, summary, metadata, created_at, ref_type',
        enricher=_make_community_enricher(community_list),
    )
    deduped = []
    for r in runs:
        if r['chain_id'] in seen:
            continue
        seen.add(r['chain_id'])
        deduped.append(r)
    return deduped


def query_aspect_runs(hours: int = 24):
    """S2 aspect-integration runs.

    Its product (candidate type/relation strings merged into aspects_v1.json)
    is already stated in its O/K/Δ summaries, so the base O/K/Δ run IS the
    whole card; journal notes attach centrally like every other unit's. The
    taxonomy itself is on the Overview tab (queries.aspects); this is the
    per-run activity view.
    """
    return _query_s2_unit_runs(
        'aspect_integration', hours,
        delta_columns='chain_id, summary, metadata, created_at, ref_type',
        enricher=_enrich_ok_delta,
    )


# ── Healer (different shape: logs-only, single forward pass) ────────────────

def query_healer_runs(hours: int = 24, limit: int = 30):
    """S2 Healer passes — one card per `healer_generated` delta event.

    Pair each delta with its nearest preceding O/K events in the same chain
    via a single forward pass — different enough from consolidation/community
    that sharing the helper would distort it.
    """
    rows = []
    journal_by_chain: dict = {}
    try:
        with ro_connect(logs_db_path()) as conn:
            if conn is None:
                return []
            rows = conn.execute(
                "SELECT chain_id, event_type, ref_type, summary, metadata, created_at "
                "FROM trace_events WHERE chain_id LIKE '%healer%' "
                "AND created_at > ? ORDER BY created_at ASC",
                (utc_cutoff(hours=hours),),
            ).fetchall()
            # Isolated like the shared skeleton's — a journal failure must not
            # cost the operator the healer feed.
            try:
                journal_by_chain = notes_by_chain(conn, {r[0] for r in rows})
            except Exception as e:
                warn('queries.s2_runs', 'healer journal notes unavailable', exc=e)
                journal_by_chain = {}
    except Exception as e:
        warn('queries.s2_runs', 'healer pull failed', exc=e)
        return []

    # Walk forward; remember most-recent O and K per chain, snap each delta
    # to whatever O/K preceded it. Linear time, no resort needed.
    last_o: dict = {}
    last_k: dict = {}
    runs = []
    for chain_id, event_type, ref_type, summary, _meta, created_at in rows:
        if event_type == 'O':
            last_o[chain_id] = (ref_type or '', summary or '')
        elif event_type == 'K':
            last_k[chain_id] = (ref_type or '', summary or '')
        elif event_type == 'delta':
            # Only the unit's per-RUN delta makes a card. Unlike the other run
            # queries this one filters in Python (it needs the O/K rows in the
            # same forward pass, so the SQL can't pre-filter by ref_type). The
            # Healer's field-fills go through dispatch and emit node_revised on
            # this chain; skipping them keeps one card per healer_generated delta.
            if (ref_type or '') in _NON_RUN_REF_TYPES:
                continue
            o = last_o.get(chain_id, ('', ''))
            k = last_k.get(chain_id, ('', ''))
            runs.append({
                'chain_id': chain_id,
                'timestamp': created_at,
                'summary': summary or '',
                'ref_type': ref_type or '',
                'o_summary': o[1],
                'o_ref_type': o[0],
                'k_summary': k[1],
                'k_ref_type': k[0],
                'journal_notes': journal_by_chain.get(chain_id, []),
            })

    # Newest first — matches the consolidation/community convention.
    runs.sort(key=lambda r: r['timestamp'], reverse=True)
    return runs[:limit]
