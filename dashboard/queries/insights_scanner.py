"""Runtime-signal insights for the Live tab's top panel.

Distinct from queries/stats.py:query_insights() (which surfaces
DATA-quality issues for the Overview tab — orphan-locked, thin nodes,
zero quotes). This module flags BEHAVIORAL anomalies the operator
should see in real time:

  - S2 units that haven't fired in a while
  - Surface selecting nothing too often (recall pipeline drift)
  - Error count spiking over the prior-hour baseline

Pure derived analysis — no new tables, no daemon round-trips. Reads
trace_events + error_logs via @safe_query. Returns a list of
{severity, icon, title, detail, evidence} dicts ready for the
frontend to render as cards.

Severity ladder: 'low' / 'medium' / 'high'. The frontend maps these
to color + icon weight; rules below pick severity based on how far
the signal has drifted from baseline.
"""

import json

from ..clock import utc_cutoff
from ..db import logs_db_path
from ..query import safe_query
from ._meta import EMPTY_SELECTION_REF_ID


# ── Thresholds ────────────────────────────────────────────────────────
# Centralized so the rule logic stays readable. Tune-by-edit, not by
# config — this module IS the policy.

S2_STALE_WARN_HOURS = 6        # S2 unit silent > 6h → medium
S2_STALE_HIGH_HOURS = 24       # S2 unit silent > 24h → high
S2_LOOKBACK_DAYS    = 7        # Don't flag units that haven't run in a week
                               # — operator probably knows they were retired

EMPTY_SELECTION_MIN_SAMPLES = 10    # Need at least N K events to compute rate
EMPTY_SELECTION_WARN_RATE   = 0.40  # > 40% empty → medium
EMPTY_SELECTION_HIGH_RATE   = 0.70  # > 70% empty → high

ERROR_SPIKE_MIN_CURRENT     = 5     # Need ≥5 errors in current hour to flag
ERROR_SPIKE_RATIO           = 3.0   # current ≥ prior × this → flag
ERROR_SPIKE_HIGH_RATIO      = 10.0  # ratio ≥ this → high

SURFACE_ID_DRIFT_MIN        = 3     # combined recover+lost events in 24h → flag
SURFACE_ID_DRIFT_HIGH       = 5     # lost (unresolvable) picks ≥ this → high

# Mirrors JOURNAL_OPEN_NUDGE_RUNS in servers/trace_contract.py — the run count
# at which the encoder's own prompt starts telling it to escalate. Past that,
# the encoder has done everything it can on its own.
JOURNAL_STANDING_RUNS       = 5
JOURNAL_STANDING_HIGH_RUNS  = 10    # re-raised this many times → high

# Known S2 unit identifiers. Chain ids look like `s2-{YYYYMMDDHHMMSS}-{unit}`
# (older rows: `s2-{YYYYMMDD}-{unit}`); we split on '-' and key by the unit
# slug. The slug is `split('-', 2)[2]`, so it's format-agnostic — the longer
# timestamp segment doesn't shift it. Adding a new unit means extending this
# map (the dashboard SHOULD flag drift the moment a unit stops firing; if a
# new unit is added but not listed here, it just stays unflagged — fail safe).
# Source: chain_id substrings observed in trace_events. `revise` is excluded —
# it's a sub-operation (S2 units write `s2-{ts}-revise` chains when they
# revise nodes/edges), not a top-level unit.
KNOWN_S2_UNITS = {
    'consolidation':           'S2 Consolidation',
    'community_detection':     'S2 Community Detection',
    'healer':                  'S2 Healer',
    'aspect_integration':      'S2 Aspect Integration',
}


def _unit_slug_from_chain(chain_id):
    """`s2-20260525-healer` → 'healer'. Tolerates extra dashes in the
    unit name (`s2-20260525-community_detection` works). Returns
    empty string for malformed ids."""
    if not chain_id or not chain_id.startswith('s2-'):
        return ''
    parts = chain_id.split('-', 2)   # ['s2', '20260525', 'unit_with_underscores']
    return parts[2] if len(parts) >= 3 else ''


def _hours_ago(iso_ts):
    """Return age in hours for an ISO-T timestamp string. Uses
    julianday so the dashboard's clock-handling rules stay honored
    (TEXT timestamps + Python datetime parsing has subtle pitfalls
    around 'T' vs ' ' separators — let SQLite handle it)."""
    # We don't have SQL here; reach for Python parsing. The timestamp
    # is always ISO-T with timezone per the brain's clock contract.
    from datetime import datetime, timezone
    try:
        # fromisoformat handles +00:00 cleanly in 3.11+; older Pythons
        # would need a fallback, but ./dev pins 3.11.11.
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    except Exception:
        return None


# ── Rules ─────────────────────────────────────────────────────────────

@safe_query('queries.insights_scanner', logs_db_path, default=[])
def _scan_s2_silence(conn):
    """Flag S2 units that ran recently-ish but have gone quiet.

    Strategy: for each known unit, find the latest trace_events row in
    the last S2_LOOKBACK_DAYS. If it's older than warn/high thresholds,
    flag. Units that have been silent the entire lookback window get
    nothing — they're probably disabled, not stuck."""
    since = utc_cutoff(days=S2_LOOKBACK_DAYS)
    rows = conn.execute(
        "SELECT chain_id, MAX(created_at) FROM trace_events "
        "WHERE scale = 's2' AND created_at > ? "
        "GROUP BY chain_id ORDER BY MAX(created_at) DESC",
        (since,),
    ).fetchall()

    # Roll up to one row per unit (latest across all chain_ids for that unit).
    latest_per_unit = {}     # slug → (last_iso, last_chain_id)
    for chain_id, last_seen in rows:
        slug = _unit_slug_from_chain(chain_id)
        if not slug:
            continue
        if slug not in latest_per_unit or last_seen > latest_per_unit[slug][0]:
            latest_per_unit[slug] = (last_seen, chain_id)

    insights = []
    for slug, label in KNOWN_S2_UNITS.items():
        if slug not in latest_per_unit:
            continue   # never ran in the window → don't flag
        last_iso, last_chain_id = latest_per_unit[slug]
        age_h = _hours_ago(last_iso)
        if age_h is None or age_h < S2_STALE_WARN_HOURS:
            continue
        severity = 'high' if age_h >= S2_STALE_HIGH_HOURS else 'medium'
        insights.append({
            'severity': severity,
            'icon': '\U0001f6e0️',  # 🛠️
            'title': '%s hasn\'t run in %dh' % (label, int(age_h)),
            'detail': ('Background unit is idle. Last activity at %s. '
                       'If this exceeds the unit\'s natural cadence, '
                       'check daemon logs for a stuck cycle.'
                       % last_iso[:19].replace('T', ' ')),
            'evidence': {'chain_id': last_chain_id, 'last_seen': last_iso},
        })
    return insights


@safe_query('queries.insights_scanner', logs_db_path, default=[])
def _scan_empty_selections(conn):
    """Flag when the surface is selecting nothing too often.

    K events with an empty `ref_id` selection mean Haiku saw the candidates
    and picked none. Some empties are normal (low-quality recalls happen);
    a sustained high rate means something drifted — surface prompt,
    embedder, candidate pool quality, or daemon health."""
    since = utc_cutoff(hours=1)
    row = conn.execute(
        "SELECT "
        "  SUM(CASE WHEN ref_id = ? THEN 1 ELSE 0 END) AS empty_count, "
        "  COUNT(*) AS total_count "
        "FROM trace_events "
        "WHERE scale = 's1' AND event_type = 'K' "
        "AND ref_type = 'surface_selected' "
        "AND created_at > ?",
        (EMPTY_SELECTION_REF_ID, since),
    ).fetchone()
    if not row:
        return []
    empty_count, total_count = row[0] or 0, row[1] or 0
    if total_count < EMPTY_SELECTION_MIN_SAMPLES:
        return []
    rate = empty_count / total_count
    if rate < EMPTY_SELECTION_WARN_RATE:
        return []
    severity = 'high' if rate >= EMPTY_SELECTION_HIGH_RATE else 'medium'
    return [{
        'severity': severity,
        'icon': '\U0001f3af',  # 🎯
        'title': 'Surface selected nothing %d/%d times this hour (%d%%)'
                  % (empty_count, total_count, int(rate * 100)),
        'detail': ('Surface is rejecting all candidates at an '
                   'elevated rate. Suspect: surface prompt drift, '
                   'embedder degradation, or candidate-pool quality.'),
        'evidence': {'empty_count': empty_count, 'total_count': total_count,
                     'rate': round(rate, 2)},
    }]


@safe_query('queries.insights_scanner', logs_db_path, default=[])
def _scan_error_spike(conn):
    """Compare error count in the current hour vs the prior hour.

    Queries `debug_log WHERE event_type='error'` — the brain-side error
    stream. Hook errors (hook_errors table) are a different concern
    (hook-pipeline integrity, not brain runtime); they get their own
    rule later if needed."""
    now_cutoff   = utc_cutoff(hours=1)
    prior_cutoff = utc_cutoff(hours=2)
    cur = conn.execute(
        "SELECT COUNT(*) FROM debug_log "
        "WHERE event_type = 'error' AND created_at > ?",
        (now_cutoff,),
    ).fetchone()
    prior = conn.execute(
        "SELECT COUNT(*) FROM debug_log "
        "WHERE event_type = 'error' AND created_at > ? AND created_at <= ?",
        (prior_cutoff, now_cutoff),
    ).fetchone()
    current_count = (cur[0] if cur else 0) or 0
    prior_count   = (prior[0] if prior else 0) or 0
    if current_count < ERROR_SPIKE_MIN_CURRENT:
        return []
    # Avoid div-by-zero by using max(prior, 1) — a spike from zero to 5+
    # is meaningful and shouldn't be silenced by a zero-divide guard.
    ratio = current_count / max(prior_count, 1)
    if ratio < ERROR_SPIKE_RATIO:
        return []
    severity = 'high' if ratio >= ERROR_SPIKE_HIGH_RATIO else 'medium'
    return [{
        'severity': severity,
        'icon': '⚠️',  # ⚠️
        'title': 'Error spike: %d this hour vs %d prior (%.1fx)'
                  % (current_count, prior_count, ratio),
        'detail': ('Error log rate has jumped over the prior-hour baseline. '
                   'Open the Logs tab to see the new entries.'),
        'evidence': {'current': current_count, 'prior': prior_count,
                     'ratio': round(ratio, 2)},
    }]


@safe_query('queries.insights_scanner', logs_db_path, default=[])
def _scan_surface_id_drift(conn):
    """Flag Haiku emitting selection ids that had to be recovered or were lost.

    surface.py's id-resolution writes two drift warnings:
      surface_id_fuzzy_recovered  — a whitespace/fragment id was recovered
                                     against the candidate pool (pick survived,
                                     but the emission is degrading)
      surface_unknown_selected_id — an id matched no candidate and resolved to
                                     nothing (pick LOST — context came out
                                     thinner than Haiku intended, silently)

    The SURFACE_SELECTED_ID_PATTERN schema constraint should hold both at ~0.
    A nonzero rate means the pattern isn't holding on some path (round-1 text,
    an older prompt variant) — this is the exact silent-context-loss class
    that produced the v12.1 wrong-abstention miss, so it must alert, not just
    sit in the Logs tab. Empty-selection scanning can't catch it: recovered
    picks make ref_id non-empty, so the pipeline looks healthy."""
    since = utc_cutoff(hours=24)
    rows = conn.execute(
        "SELECT source, COUNT(*) FROM debug_log "
        "WHERE event_type = 'warning' AND created_at > ? "
        "AND source IN ('surface_unknown_selected_id', "
        "               'surface_id_fuzzy_recovered') "
        "GROUP BY source",
        (since,),
    ).fetchall()
    counts = {s: n for s, n in rows}
    lost      = counts.get('surface_unknown_selected_id', 0)
    recovered = counts.get('surface_id_fuzzy_recovered', 0)
    if lost + recovered < SURFACE_ID_DRIFT_MIN:
        return []
    severity = 'high' if lost >= SURFACE_ID_DRIFT_HIGH else 'medium'
    return [{
        'severity': severity,
        'icon': '\U0001f524',  # 🔤 — the id charset (lowercase hex)
        'title': 'Surface id corruption: %d lost, %d recovered (24h)'
                  % (lost, recovered),
        'detail': ('Haiku emitted selection ids that failed to match a '
                   'candidate. Lost picks dropped context silently; recovered '
                   'picks survived but the emission is degrading. The '
                   'SURFACE_SELECTED_ID_PATTERN schema constraint should keep '
                   'this at zero — a nonzero rate means it is not holding on '
                   'some path. Open the Logs tab for the per-id detail.'),
        'evidence': {'lost': lost, 'recovered': recovered},
    }]


@safe_query('queries.insights_scanner', logs_db_path)
def _scan_standing_journal_items(conn):
    """An encoder has been asking the same question, run after run, into nothing.

    Journal notes tagged `open` carry a rendered lifecycle head — `open ×12
    since 08-17` — counting the runs the subject has stayed unresolved. Past
    JOURNAL_STANDING_RUNS the encoder's own prompt tells it to escalate, but
    the only escalation route it has is... another journal note. That is the
    closed loop: the S2 consolidation encoder wrote
    "operator confirmation still needed" three times running with no path to an
    operator.

    This rule is the path. It is the one insight whose subject is a question
    for a PERSON, not a system anomaly — so it earns the panel even on a
    perfectly healthy brain.
    """
    # A week, not a day: persistence is the whole signal, and the slower
    # encoders (consolidation runs on idle) need days to accumulate the run
    # count that makes an item standing in the first place. A 72h window
    # measured nothing but the fast units.
    since = utc_cutoff(hours=168)
    rows = conn.execute(
        "SELECT chain_id, ref_id, metadata FROM trace_events "
        "WHERE ref_type = 'journal_note' AND created_at > ?",
        (since,),
    ).fetchall()
    # Reuse the journals reader's tag parser — the ×N count lives inside a
    # free-text tag, and re-implementing that split here is how the two
    # would drift.
    from .journals import journal_unit, split_tag
    worst = {}
    for chain_id, ref_id, meta_raw in rows:
        try:
            meta = json.loads(meta_raw) if meta_raw else {}
        except (ValueError, TypeError):
            continue
        if not isinstance(meta, dict):
            continue
        _tag, runs, since_str = split_tag(meta.get('tag') or '')
        if runs < JOURNAL_STANDING_RUNS:
            continue
        subject = ref_id or ''
        prior = worst.get(subject)
        if not prior or runs > prior['runs']:
            worst[subject] = {'runs': runs, 'since': since_str,
                              'subject': subject or '(no subject)',
                              'unit': journal_unit(chain_id),
                              'note': (meta.get('note') or '')[:220]}
    if not worst:
        return []
    items = sorted(worst.values(), key=lambda i: -i['runs'])
    top = items[0]
    severity = 'high' if top['runs'] >= JOURNAL_STANDING_HIGH_RUNS else 'medium'
    detail = ('%s has re-raised "%s" across %d runs%s and cannot resolve it '
              'alone: %s' % (top['unit'] or 'an encoder', top['subject'],
                             top['runs'],
                             (' since %s' % top['since']) if top['since'] else '',
                             top['note']))
    if len(items) > 1:
        detail += ' — and %d other standing item%s. Open the Journals tab.' % (
            len(items) - 1, '' if len(items) == 2 else 's')
    else:
        detail += ' Open the Journals tab.'
    return [{
        'severity': severity,
        'icon': '\U0001f4d3',  # 📓
        'title': '%d journal item%s waiting on you' % (
            len(items), '' if len(items) == 1 else 's'),
        'detail': detail,
        'evidence': {'items': len(items), 'max_runs': top['runs']},
    }]


# ── Aggregator ────────────────────────────────────────────────────────

def scan_all():
    """Run every rule and concatenate results. Each rule is independent
    + decorated with @safe_query, so one failing rule doesn't take down
    the whole panel (the dashboard's loud-by-default substrate logs the
    exception; the rule returns []; the panel renders the other rules).
    """
    return (
        _scan_s2_silence()
      + _scan_empty_selections()
      + _scan_error_spike()
      + _scan_surface_id_drift()
      + _scan_standing_journal_items()
    )
