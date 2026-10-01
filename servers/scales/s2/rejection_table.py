"""S2 Rejection Table — fingerprint-based suppression for integration units.

Solves the re-proposal bug: when an encoder rejects a proposal, nothing in
the graph changes, so the decoder keeps producing the same proposal every
run. Without a rejection memory, the encoder judges the same thing forever.

Design:
- Proposal fingerprint = hash of parameters the encoder actually judges
  (not implementation artifacts). When graph state changes in a way that
  would alter the proposal's inputs, the fingerprint changes and the old
  rejection no longer applies.
- Table `s2_rejections` in brain.db holds (fingerprint, integration_unit,
  proposal_type, proposed_ids, created_at). Proposed IDs enable S3 pattern
  analysis.
- Table schema lives in servers/schema.py (single source of truth).
  ensure_schema() creates it at Brain startup — do not CREATE TABLE here.

Used by S2 Community Detection and Consolidation. Pattern for any S2 unit:
- Informational outcomes (CONSOLIDATE/EVOLVE/KEEP/community placement)
  write semantic edges — state is discoverable from the graph via JOIN.
- Marker-only outcomes (SKIP, "looked at, didn't act") record a fingerprint
  here so the decoder filters the proposal on subsequent runs.

Decoder-side: call filter_rejected(brain, proposals) before sending to encoder.
Encoder-side: call record_rejections(brain, skipped, integration_unit=...)
after the encoder decides a proposal isn't worth a semantic edge.
"""
import hashlib
import json
from datetime import datetime, timezone

from ...contract import VALID_BATCH_OPS, unwrap_operations


# ═══════════════════════════════════════════════════════════════
# FINGERPRINT COMPUTATION
# ═══════════════════════════════════════════════════════════════
#
# Each fingerprint captures what the encoder actually judges on.
# Implementation artifacts (internal fraction, shared_count, exact
# affinity) are excluded so proposals aren't falsely invalidated by
# cosmetic changes. When real inputs change (community grew toward
# the node, cluster composition changed, drift thresholds raised),
# the fingerprint naturally changes.

def compute_fingerprint(proposal):
    """Stable fingerprint for a proposal.

    - add_to_existing: node_id + top community id + affinity_tier, read
      from communities[0] (the decoder sorts candidates by affinity desc,
      so [0] is the strongest — the dominant input to the place/skip
      judgment). Tier captures the regime where encoder judgment would
      change: borderline <0.40, moderate <0.65, strong >=0.65.
    - new_community: top 40% of representative IDs (structural hubs
      define cluster identity; peripheral member changes don't
      invalidate the rejection)
    - drift: node_id + foreign_community_id
    - health_update: community_id + signal type (dead/degrading/maturing)
    - merge_communities: larger_id + smaller_id
    """
    ptype = proposal.get('type', '')

    if ptype == 'add_to_existing':
        # The decoder carries candidate communities in `communities`
        # (sorted by affinity desc) — NOT top-level community_id/affinity.
        # Reading those absent keys collapsed every fingerprint to
        # 'add:<node>::borderline', so one rejection suppressed the node
        # into ALL communities at ALL tiers forever. Fingerprint the
        # strongest candidate instead.
        top = (proposal.get('communities') or [{}])[0]
        aff = top.get('affinity', 0)
        if aff >= 0.65:
            tier = 'strong'
        elif aff >= 0.40:
            tier = 'moderate'
        else:
            tier = 'borderline'
        raw = 'add:%s:%s:%s' % (
            proposal.get('node_id', ''),
            top.get('id', ''),
            tier)

    elif ptype == 'new_community':
        # Structural hubs (top 40% by internal edges) define cluster identity.
        # Falls back to sorted member IDs if representatives not populated.
        reps = proposal.get('representatives', [])
        if reps:
            rep_ids = sorted(r.get('id', '') for r in reps if r.get('id'))
            n_keep = max(2, int(len(rep_ids) * 0.4 + 0.5))
            core = rep_ids[:n_keep]
        else:
            members = sorted(proposal.get('members', []))
            n_keep = max(2, int(len(members) * 0.4 + 0.5))
            core = members[:n_keep]
        raw = 'new:' + ':'.join(core)

    elif ptype == 'drift':
        foreign = proposal.get('foreign', [{}])
        foreign_id = foreign[0].get('id', '') if foreign else ''
        raw = 'drift:%s:%s' % (proposal.get('node_id', ''), foreign_id)

    elif ptype == 'health_update':
        raw = 'health:%s:%s' % (
            proposal.get('community_id', ''),
            proposal.get('signal', ''))

    elif ptype == 'merge_communities':
        raw = 'merge:%s:%s' % (
            proposal.get('larger_id', ''),
            proposal.get('smaller_id', ''))

    elif ptype == 'consolidation_cluster':
        # Cluster identity = sorted (member_id, member_updated_at) pairs.
        # Members alone aren't enough: a small content edit can leave
        # similarity above threshold so the cluster re-surfaces with the
        # same IDs, and the old rejection would block re-evaluation even
        # though the content the encoder would judge has changed.
        # Including updated_at makes any revise()/remember() on a member
        # invalidate the rejection automatically — no bookkeeping needed.
        # updated_at, not revised_at, on purpose: re-proposal is gated
        # upstream by the revised_at change set, so a metadata-only revise
        # invalidating a fingerprint is inert — see consolidation.py.
        # Required: a producer that omitted it would mint fingerprints blind
        # to member edits, and a rejection would then outlive the content it
        # judged.
        ts_by_id = proposal['member_updated_at']
        members = sorted(proposal.get('members', []))
        raw = 'consol:' + ':'.join(f'{m}|{ts_by_id.get(m, "")}' for m in members)

    elif ptype == 'unplaceable':
        # A node the decoder examined and couldn't place into any community.
        # The fingerprint is the node's 1-hop neighborhood — each neighbor +
        # that neighbor's community (or '' if unplaced) — the only inputs that
        # decide placeability (cluster membership and add-to-existing affinity
        # are both edge-local). When the node gains/loses an edge OR a neighbor
        # joins a community, this string changes, the fingerprint changes, and
        # the node falls out of suppression to be re-examined. Likelihood-based
        # re-examination, not time-based.
        raw = 'unplaceable:%s:%s' % (
            proposal.get('node_id', ''),
            proposal.get('neighborhood', ''))

    else:
        raw = '%s:%s' % (ptype, proposal.get('node_id', ''))

    return hashlib.md5(raw.encode()).hexdigest()[:16]


def get_proposed_ids(proposal):
    """Extract all node IDs involved in a proposal (for S3 analysis)."""
    ids = []
    if proposal.get('node_id'):
        ids.append(proposal['node_id'])
    if proposal.get('members'):
        ids.extend(proposal['members'])
    if proposal.get('community_id'):
        ids.append(proposal['community_id'])
    # add_to_existing carries its target communities under `communities`,
    # not community_id — capture them for S3 analysis too.
    for c in proposal.get('communities', []):
        if isinstance(c, dict) and c.get('id'):
            ids.append(c['id'])
    if proposal.get('larger_id'):
        ids.append(proposal['larger_id'])
    if proposal.get('smaller_id'):
        ids.append(proposal['smaller_id'])
    for f in proposal.get('foreign', []):
        if isinstance(f, dict) and f.get('id'):
            ids.append(f['id'])
    return ids


# Consecutive thwarted runs a unit shields (retry, no fingerprints) before
# giving up and stamping anyway. A run is thwarted when its encoder tried to
# act and dispatch dropped the attempt — an invalid concept-verb op, or a
# whole brain_batch call rejected (empty/malformed operations). Bounds both
# retry loops: a persistently thwarted encoder otherwise pins the unit
# forever — no fingerprint, no baseline advance, a full re-encode every
# cycle with zero progress — and the coordinator's consecutive_failures
# never fires because these runs return success. A stamped give-up is
# recoverable: the fingerprint invalidates on any member edit.
THWARTED_RETRY_LIMIT = 3


def thwart_shield(brain, streak_key):
    """Read a unit's thwarted-run streak → (streak, shield_active).

    While shield_active, the run may retry thwarted work instead of stamping
    fingerprints; past THWARTED_RETRY_LIMIT the caller stamps anyway (the
    give-up). Pair with settle_thwart_streak once the run's partition is
    known — both units share this state machine so the limit means the same
    thing everywhere.
    """
    streak = int(brain.get_config(streak_key) or 0)
    return streak, streak < THWARTED_RETRY_LIMIT


def settle_thwart_streak(brain, streak_key, streak, shielded):
    """Advance the streak when this run actually shielded thwarted work,
    reset otherwise — a clean run, a run that pinned nothing, or a give-up
    (which stamps) all break the streak. Counts shielded runs since the last
    clean stamp, not wall-clock cycles: interleaved encode-failure or
    early-return cycles pin the unit too and simply don't advance it.
    """
    if shielded:
        brain.set_config(streak_key, str(streak + 1))
    elif streak:
        brain.set_config(streak_key, '0')


def loud_warning(brain, source, prefix, msg):
    """_log_warning + console print, one message — the S2 units' paired
    emission for shield events."""
    brain._log_warning(source, msg)
    print('[%s] %s' % (prefix, msg), flush=True)


# Every key a brain_batch op can carry a node id under, across both
# polarities — invalid ops are made-up verbs, so nothing constrains which
# keys they carry; attribution reads them all.
_OP_NODE_ID_KEYS = ('node_id', 'source_id', 'target_id',
                    'survivor_id', 'absorbed_id', 'larger_id', 'smaller_id')


def _batch_ops(action_details, skip_errored=False):
    """Yield op dicts from the brain_batch actions in action_details.

    Tolerates malformed entries (non-dict actions, input=None) and unwraps a
    stringified operations array the way dispatch does before executing it.
    skip_errored drops calls dispatch rejected whole — they wrote nothing.
    """
    for action in action_details or []:
        if not isinstance(action, dict) or action.get('tool') != 'brain_batch':
            continue
        if skip_errored and action.get('error'):
            continue
        ops = (action.get('input') or {}).get('operations')
        if isinstance(ops, str):
            ops = unwrap_operations(ops)
        if not isinstance(ops, list):
            continue
        for op_spec in ops:
            if isinstance(op_spec, dict):
                yield op_spec


def _op_node_ids(op_spec):
    ids = {op_spec[k] for k in _OP_NODE_ID_KEYS if op_spec.get(k)}
    ids.update(m for m in op_spec.get('members') or [] if m)
    return ids


def had_rejected_batch_call(action_details):
    """True when the run contains a brain_batch call dispatch rejected whole.

    A call-level rejection (empty `operations`, a non-list, a string that
    doesn't parse) writes NOTHING and references no node ids, so op-level
    attribution can't shield anything from the skip→rejection path. The same
    cluster can yield real ops on a retry, so a rejected call is a thwarted
    run, not a clean SKIP. Detection is the dispatch verdict: the runner
    records `error` on every action whose handler returned ok=False.
    """
    return any(
        isinstance(a, dict) and a.get('tool') == 'brain_batch' and a.get('error')
        for a in action_details or [])


def node_ids_touched_by_valid_ops(action_details):
    """Node IDs targeted by valid-named ops in brain_batch calls dispatch
    accepted whole.

    The attribution counterpart of had_rejected_batch_call: a cluster whose
    members a real op targeted was PROCESSED this run — the rejected-call
    shield must not pull it into retry. A targeted op may still have failed
    per-op (dispatch returns ok=True with per-op errors); those ids count
    deliberately — per-op failures feed back to the encoder in-loop (its
    retry channel), and stamping such clusters matches the standing
    fingerprint-every-processed-cluster policy.
    """
    return {i for op in _batch_ops(action_details, skip_errored=True)
            if op.get('op', '') in VALID_BATCH_OPS
            for i in _op_node_ids(op)}


def node_ids_touched_by_invalid_ops(action_details):
    """Node IDs the encoder targeted with an op OUTSIDE the closed vocabulary.

    A concept-verb op name (`consolidate`, `reject`, `keep`, `skip`) is a
    decision leaked into the op field; dispatch drops it and logs
    `brain_batch_invalid_op`. The encoder *tried* to act and was thwarted —
    a failure to retry, not a clean SKIP. Intersect with a proposal's
    `get_proposed_ids()` to pull thwarted work out of the skip→rejection
    path (stamping it would abandon the intended action AND suppress the
    proposal until a member's updated_at changes).
    """
    return {i for op in _batch_ops(action_details)
            if op.get('op', '') not in VALID_BATCH_OPS
            for i in _op_node_ids(op)}


# ═══════════════════════════════════════════════════════════════
# FILTERING AND RECORDING
# ═══════════════════════════════════════════════════════════════

def filter_rejected(brain, proposals):
    """Remove proposals that match a previous rejection fingerprint.

    Returns (surviving_proposals, suppressed_count).
    """
    if not proposals:
        return proposals, 0

    fps_with_proposal = [(p, compute_fingerprint(p)) for p in proposals]
    all_fp = [fp for _, fp in fps_with_proposal]

    rejected_set = set()
    for chunk_start in range(0, len(all_fp), 900):
        chunk = all_fp[chunk_start:chunk_start + 900]
        if not chunk:
            continue
        placeholders = ','.join('?' * len(chunk))
        rows = brain.conn.execute(
            "SELECT fingerprint FROM s2_rejections WHERE fingerprint IN (%s)" % placeholders,
            chunk).fetchall()
        for row in rows:
            rejected_set.add(row[0])

    surviving = []
    suppressed = 0
    for p, fp in fps_with_proposal:
        if fp in rejected_set:
            suppressed += 1
        else:
            surviving.append(p)
    return surviving, suppressed


def record_rejections(brain, proposals, integration_unit='s2:community_detection'):
    """Write rejected proposal fingerprints to s2_rejections.

    Uses INSERT OR IGNORE — duplicate fingerprints from repeated rejection
    are silently deduplicated.
    """
    if not proposals:
        return 0
    ts = datetime.now(timezone.utc).isoformat()  # clock-ok — bookkeeping timestamp for rejection record
    count = 0
    for p in proposals:
        fp = compute_fingerprint(p)
        ids_json = json.dumps(get_proposed_ids(p))
        brain.conn.execute(
            "INSERT OR IGNORE INTO s2_rejections "
            "(fingerprint, integration_unit, proposal_type, proposed_ids, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (fp, integration_unit, p.get('type', ''), ids_json, ts))
        count += 1
    brain.conn.commit()
    return count


def clear_unplaceable_rejections(brain, proposals):
    """Drop each proposal's current `unplaceable` rejection row, batched
    (one DELETE scan per chunk, not one per node).

    Paired with record_rejections to keep s2_rejections at one unplaceable row
    per node. The fingerprint is neighborhood-derived, so a re-mark writes a
    NEW fingerprint — the stale row can't be REPLACEd on the fingerprint PK; it
    must be deleted by (proposal_type, proposed_ids). proposed_ids matches what
    record_rejections wrote for an unplaceable proposal: json.dumps([node_id]).

    Does NOT commit: the caller holds write_lock and record_rejections' commit
    closes the DELETE+INSERT as one transaction.
    """
    if not proposals:
        return
    keys = [json.dumps([p['node_id']]) for p in proposals]
    for i in range(0, len(keys), 400):
        chunk = keys[i:i + 400]
        ph = ','.join('?' * len(chunk))
        brain.conn.execute(
            "DELETE FROM s2_rejections "
            "WHERE proposal_type = 'unplaceable' AND proposed_ids IN (%s)" % ph,
            chunk)


# ═══════════════════════════════════════════════════════════════
# MATCHER — proposal → encoder action
# ═══════════════════════════════════════════════════════════════

def match_proposals_to_actions(sent_proposals, action_details):
    """Walk brain_batch operations to determine which proposals were acted on.

    Returns (acted_on, skipped) lists. A proposal is "acted on" if the
    encoder made ANY operation targeting its constituent nodes (success
    or failure — the encoder still saw and judged it).

    Matching rules per proposal type:
    - new_community: remember op (type=community) whose connect_to IDs
      overlap >= 50% with the proposal's member set
    - add_to_existing: connect op (community_member) with matching (source, target)
    - drift (accept): connect op to foreign community
    - drift (reject): no op at all — declining is silence, and the encoder's
      algorithmic Δ raises _sys_drift_threshold for every skipped drift
    - health_update: archive or revise (community_maturity) on the community
    - merge_communities: absorb of the pair, archive on smaller_id, or revise on larger_id
    """
    acted_idx = set()

    for op_spec in _batch_ops(action_details):
        op = op_spec.get('op', '')

        if op == 'remember':
            # Read the way the create door reads it (community_contract owns
            # that); each target is an existing member's id.
            conn_targets = community_member_targets(op_spec)
            if not conn_targets:
                continue
            for i, p in enumerate(sent_proposals):
                if p.get('type') != 'new_community':
                    continue
                members = set(p.get('members', []))
                if not members:
                    continue
                overlap = len(conn_targets & members) / len(members)
                if overlap >= 0.5:
                    acted_idx.add(i)

        elif op == 'connect' and op_spec.get('relation') == 'community_member':
            src = op_spec.get('source_id')
            tgt = op_spec.get('target_id')
            if not (src and tgt):
                continue
            for i, p in enumerate(sent_proposals):
                if p.get('type') == 'add_to_existing':
                    # Candidate communities live in `communities` (the
                    # proposal carries no top-level community_id). The
                    # encoder connects the node to one of them — match on
                    # node_id + any candidate, else a real placement is
                    # mis-stamped as a rejection and its acted-on count lost.
                    cand_ids = {c.get('id')
                                for c in p.get('communities', [])
                                if isinstance(c, dict)}
                    if p.get('node_id') == tgt and src in cand_ids:
                        acted_idx.add(i)
                elif p.get('type') == 'drift':
                    if p.get('node_id') == tgt:
                        for f in p.get('foreign', []):
                            if isinstance(f, dict) and f.get('id') == src:
                                acted_idx.add(i)
                                break

        elif op == 'revise':
            nid = op_spec.get('node_id')
            if not nid:
                continue
            if 'community_maturity' in op_spec:
                for i, p in enumerate(sent_proposals):
                    if (p.get('type') == 'health_update'
                            and p.get('community_id') == nid):
                        acted_idx.add(i)
            for i, p in enumerate(sent_proposals):
                if (p.get('type') == 'merge_communities'
                        and p.get('larger_id') == nid):
                    acted_idx.add(i)

        elif op == 'absorb':
            for i, p in enumerate(sent_proposals):
                if (p.get('type') == 'merge_communities'
                        and p.get('larger_id') == op_spec.get('survivor_id')
                        and p.get('smaller_id') == op_spec.get('absorbed_id')):
                    acted_idx.add(i)

        elif op == 'archive':
            nid = op_spec.get('node_id')
            if not nid:
                continue
            for i, p in enumerate(sent_proposals):
                if p.get('type') == 'health_update' and p.get('community_id') == nid:
                    acted_idx.add(i)
                elif (p.get('type') == 'merge_communities'
                        and p.get('smaller_id') == nid):
                    acted_idx.add(i)

    acted_on = [sent_proposals[i] for i in sorted(acted_idx)]
    skipped = [p for i, p in enumerate(sent_proposals) if i not in acted_idx]
    return acted_on, skipped


# ═══════════════════════════════════════════════════════════════
# PRIORITY ORDERING
# ═══════════════════════════════════════════════════════════════
#
# TYPE_PRIORITY lives in community_contract.py (source of truth).
# Community-specific: the proposal types belong to the community unit.

from .community_contract import (  # noqa: E402
    TYPE_PRIORITY, community_member_targets)


def sort_proposals_by_priority(proposals):
    """Sort proposals by type priority, then by confidence within type (descending)."""
    def _sort_key(p):
        type_rank = TYPE_PRIORITY.get(p.get('type', ''), 99)
        ptype = p.get('type', '')
        if ptype == 'new_community':
            confidence = p.get('internal_fraction', 0)
        elif ptype == 'add_to_existing':
            # Affinity lives in communities[0] (sorted desc), not top-level.
            confidence = (p.get('communities') or [{}])[0].get('affinity', 0)
        elif ptype == 'merge_communities':
            confidence = p.get('overlap_pct', 0)
        elif ptype == 'health_update':
            signal = p.get('signal', '')
            confidence = {'dead': 1.0, 'degrading': 0.5, 'corridor_maturing': 0.3}.get(signal, 0)
        elif ptype == 'drift':
            foreign = p.get('foreign', [{}])
            confidence = foreign[0].get('affinity', 0) if foreign else 0
        else:
            confidence = 0
        return (type_rank, -confidence)

    return sorted(proposals, key=_sort_key)
