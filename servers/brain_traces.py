"""The traces functional layer — every trace read (and API-level write) on Brain.

One rule, no judgment: reading or writing traces through the API? It's a
`brain.` method, and it lives HERE. Only this file touches TraceDAL; the
vocabulary (ref types, turn classification, journal parse/render) stays in
trace_contract.py. Sanctioned direct-DAL exceptions: the LAF scorer's
vector-substrate pull (`event_vector_rows` — recall_laf's episodic
matrix), embed_queue's vector
substrate maintenance (`find_unembedded`/`store_embeddings`), and the
read-only dashboard.

Sections:
- Generic door   — query_traces, get_trace, get_traces, count_traces
- Journal + arc  — journal_notes, write_journal_notes, write_session_arc
- Episodic       — recall_episodes (decode-over-traces sibling of recall)
- Conversation   — get_conversation, turns_since_last_encode,
                   get_conversation_around (context grouped by session)

Traces are the universal record of the whole fractal — S0 exchanges, S1 runs,
S2 runs — so these are brain-level capabilities: they span every scale, owned
by none. Scale packages (scales/s0|s1|s2) host integration units, never data
access; scale is a tag in the substrate, not a boundary in the read path.

Design: docs/TRACES-LAYER-DESIGN.md
"""
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from . import embedder
from .clock import iso_cutoff, resolve_offset, PAST
from .contract import flag_truncation as _flag_truncation
from .brain_constants import (
    EPISODE_DEFAULT_LIMIT, EPISODE_DEFAULT_WINDOW_DAYS,
    EPISODE_SEMANTIC_CANDIDATE_CAP)


def _resolve_time_bound(value):
    """Resolve a recall_episodes time bound to an ISO created_at string.

    Accepts relative shorthand ('30m', '2h', '3d', '1w', case-insensitive) →
    that-much-ago, or an ISO timestamp literal. Returns '' for falsy input.
    Anything else raises ValueError — a malformed bound must fail loud, not
    silently bind a non-timestamp into a lexical comparison (empty result, or
    a no-op that disables the filter). Parsing is a presentation concern,
    kept out of the DAL, which only ever sees resolved ISO bounds.

    The grammar itself lives in clock.py, shared with the Thalamus door: one
    published contract, one implementation, one direction argument. This
    keeps only what is genuinely local — the ''-for-empty convention and the
    bound-flavoured error message.
    """
    if not value:
        return ''
    try:
        return resolve_offset(value, direction=PAST)
    except ValueError as e:
        raise ValueError('time bound %r %s' % (value, e))


def stamp_s0_session(metadata, env):
    """Merge the session stamp (trace_contract.S0_SESSION_STAMP_FIELDS — model,
    host) into an S0 row's metadata. `env` is a mapping carrying those fields
    (vars(ctx) on the hook path, session_env_for() on the dispatched path).
    setdefault semantics — an explicit per-event value wins; an unknown (empty)
    field is left out rather than stamped blank, so a row from before the value
    was learned carries no key at all. Returns the metadata unchanged when
    nothing is known, and unchanged when it is not a dict (None stays None; a
    non-dict wire payload is the DAL's to warn about, never a raise here — the
    same posture as TraceDAL._stamp_identity)."""
    from .trace_contract import S0_SESSION_STAMP_FIELDS
    stamp = {k: env.get(k, '') for k in S0_SESSION_STAMP_FIELDS if env.get(k, '')}
    if not stamp or (metadata is not None and not isinstance(metadata, dict)):
        return metadata
    metadata = dict(metadata or {})
    for k, v in stamp.items():
        metadata.setdefault(k, v)
    return metadata


def stamp_tool_result(metadata, env):
    """Build the tool stamp at the S0 write door, then merge session fields.

    Missing tells means an old client: use session identity, marked legacy.
    Present-but-empty/ambiguous tells are event-local uncertainty, never a
    reason to inherit the session's host. No session state is mutated here.
    Unknown wire extras survive; malformed payloads remain visible as raw.
    """
    from .host_contract import (
        resolve_host, classify_tool, VOCAB_VERSION, contract_fingerprint)
    from .trace_contract import build_tool_result_metadata

    raw = dict(metadata) if isinstance(metadata, dict) else {'raw': metadata}
    if 'tells' in raw:
        tells = raw['tells']
        present = {t: True for t in tells if isinstance(t, str)} \
            if isinstance(tells, (list, tuple)) else {}
        host, host_status, _ = resolve_host(present)
    else:
        host, host_status, tells = env.get('host', ''), 'legacy', []
    tool = raw.get('tool', '')
    kind, kind_status = classify_tool(host, tool)
    stamp = build_tool_result_metadata(
        tool=tool, kind=kind, kind_status=kind_status, host_status=host_status,
        tells=tells, vocab_version=VOCAB_VERSION,
        impl_identity=contract_fingerprint(),
        tool_use_id=raw.get('tool_use_id') or '',
        turn_id=raw.get('turn_id') or '',
        payload_keys=raw.get('payload_keys') or [])
    # The raw name and opaque extras belong to the client. Only the daemon's
    # normalization fields are authoritative (including against spoofed stamps).
    raw.update(stamp)
    if isinstance(metadata, dict) and 'tool' in metadata:
        raw['tool'] = metadata['tool']
    raw['host'] = host
    return stamp_s0_session(raw, env)


def _s0_trace(brain, ctx, event_type, ref_type, summary, metadata=None,
              content=None, ref_id=''):
    """Append one S0 turn-trace, binding the per-turn invariants in ONE place:
    chain (ctx.s0_chain()), scale ('s0'), the session (ctx.session_id), the
    session stamp (model/host from ctx, via stamp_s0_session) — and the
    stored-content cap: pass the turn's full text as `content` and it
    lands in metadata['content'] capped LOUDLY at the pipeline store limit
    (a marker names the dropped count — never a silent slice, per the
    standing truncation rule), so the timeline's sides can neither drift
    apart nor get cut invisibly. The S0 turn events differ only in
    event_type / ref_type / summary / metadata / ref_id; everything else is
    turn-fixed. Routing them all through here keeps session_id from being
    dropped — the self_message append once omitted it, leaving cross-stream
    deliveries unattributable to the recipient session.

    Callers: the hooks (daemon_hooks) and the delivery leg (channels/
    delivery.py). Call it by BARE NAME — test_trace_contract_sync resolves
    `_s0_trace(...)` as a scale-binding helper; an attribute call would be
    extractor-blind.

    Returns the appended trace_event id (hook_recall passes the current
    prompt's id to get_session_turns as exclude_trace_id)."""
    if content:
        from .pipeline_contract import PIPELINE as _PL
        from .loud_truncation import cap_text_loud
        metadata = dict(metadata or {})
        metadata['content'] = cap_text_loud(
            content, _PL['assistant_response_store'],
            marker='…[+%d chars truncated at trace store]')
    metadata = stamp_s0_session(metadata, vars(ctx))
    return brain._trace_dal.append(
        chain_id=ctx.s0_chain(), scale='s0', session_id=ctx.session_id,
        event_type=event_type, ref_type=ref_type, summary=summary,
        metadata=metadata, ref_id=ref_id)


class BrainTracesMixin:
    """Brain-level trace capabilities: the generic query door, journal/arc
    residue, episodic recall, and conversation reads. Composed onto Brain."""

    # ── Generic door ──

    def get_trace(self, trace_id):
        """Single trace_event point lookup. Returns the full row dict
        (id/chain_id/scale/event_type/ref_type/ref_id/summary/metadata/
        session_id/created_at) or None if not found.

        Convention mirrors brain.get_node — same single-input shape so
        callers don't have to think about batch vs point at the API
        edge. For batch lookups use brain.get_traces.
        """
        if trace_id is None:
            return None
        rows = self._trace_dal.get_by_ids([trace_id])
        return rows[0] if rows else None

    def get_traces(self, trace_ids):
        """Batch trace_event lookup. Returns a list of full row dicts
        in ascending-id order; missing ids are silently skipped.

        Use this when a caller has a list of trace_ids (e.g., expanding
        node.source_refs at render time, or get_traces tool exposure
        for the encoder). Single point lookup → use brain.get_trace.
        """
        if not trace_ids:
            return []
        return self._trace_dal.get_by_ids(list(trace_ids))

    def query_traces(self, scale: str = '', hours: int = 24,
                     event_type: str = '', chain_id: str = '',
                     session_id: str = '', session_ids=None,
                     ref_type: str = '', ref_id: str = '', chain_suffix: str = '',
                     exclude_ref_types=None,
                     grouped: bool = False, limit: int = 100,
                     older_than: str = ''):
        """Query trace events — the fractal learning loop data.

        Modes:
        - chain_id set: return single chain with all events
        - ref_type set: filter events by ref_type (+ optional ref_id to scope to
          one subject, chain_suffix to scope to one S2 unit's chains, session_id
          to scope to one session, hours=None to disable the time window)
        - grouped=True + session_id: return chains grouped with nested events
          (NOTE: unlike the flat single-session pull, the grouped path IS
          hours-bound — pass a wide hours for historical sessions)
        - session_ids (list) set: cross-session pull; hours ignored
        - session_id (str) set: single-session pull; hours ignored
        - default: return flat recent events (hours-bound; + optional chain_suffix
          to scope to one S2 unit, exclude_ref_types to drop residue like
          journal_note, hours=None to disable the window)

        older_than (ISO, strict `created_at <`) positions the newest-first
        LIMIT window at a historical instant — the replay as-of bound,
        pushed into SQL so the limit clips the right end of the ordering.
        Applies to the flat modes (ref_type / default); chain_id and grouped
        pulls are whole-chain reads and don't take it.
        """
        if chain_id:
            return {'chain': self._trace_dal.get_chain(chain_id)}
        # Every bounded branch fetches limit+1: an extra row is PROOF the
        # window has more data than the limit returned. Silent saturation is
        # the dangerous case — a limit-clipped result covers a fraction of the
        # requested window while looking complete (a 168h cost tally that
        # actually spanned 2 days, 2026-08-06). Loud-by-default: the caller
        # gets a 'truncated' payload, never a plausible-looking partial.
        if ref_type:
            rows = self._trace_dal.get_by_ref_type(
                ref_type=ref_type, scale=scale, hours=hours, limit=limit + 1,
                session_id=session_id, ref_id=ref_id, chain_suffix=chain_suffix,
                older_than=older_than)
            out = _flag_truncation({'events': rows[:limit]}, rows, limit,
                                   key='events')
            # The WINDOW clips as silently as the limit did. This branch
            # outranks the session modes, so a session-scoped ref_type pull
            # stays hours-bound (deliberate — get_by_ref_type composes
            # predicates, no authority rule; surface passes hours=None when it
            # wants the whole session). An ad-hoc caller reading the session
            # bullets instead of this one gets a 24h slice that looks like the
            # session: 0 rows for one session, 386 of 1,163 for another, in a
            # 2026-08-16 cost audit. Naming the applied window costs one key
            # and makes the clip visible without changing what anyone gets.
            if hours is not None and (session_id or ref_id):
                out['window_hours'] = hours
            return out
        if grouped and session_id:
            chains = self._trace_dal.get_chains(
                session_id=session_id, scale=scale, hours=hours,
                limit=limit + 1)
            return _flag_truncation({'chains': chains[:limit]}, chains, limit,
                                    key='chains')
        # Single or multi session pulls — both authoritative, both ignore hours.
        # get_recent raises ValueError if both are set; we don't second-guess.
        rows = self._trace_dal.get_recent(
            scale=scale, hours=hours, event_type=event_type,
            session_id=session_id, session_ids=session_ids, limit=limit + 1,
            chain_suffix=chain_suffix, exclude_ref_types=exclude_ref_types,
            older_than=older_than)
        return _flag_truncation({'events': rows[:limit]}, rows, limit,
                                key='events')

    def count_traces(self, field: str, scale: str = '', hours: int = 24):
        """Count trace events grouped by a field."""
        return self._trace_dal.count_by(field=field, scale=scale, hours=hours)

    # ── Journal + arc (encoder residue) ──

    def journal_notes(self, *, subject: str = '', scale: str = '',
                      session_id: str = '', unit: str = '',
                      k=None, limit: int = 200):
        """Read encoder journal notes — through the trace API, never raw SQL.

        Composes query_traces(ref_type='journal_note', ...) — the public door;
        TraceDAL stays underneath it. Two modes:
        • subject set → every note ABOUT that subject (ref_id), newest first
          (the hotspot view: N notes on one subject).
        • else → continuity: notes from the last K note-bearing RUNS of an
          encoder, scoped by scale + (session_id for S1 | unit for S2). K
          defaults to JOURNAL_CONTINUITY_RUNS[encoder] (s1e / unit) → DEFAULT.

        Runs group by chain_id (per-run-unique at both scales). Returns note
        dicts {tag, note, subject, chain_id, created_at}, newest first.
        Subject→title resolution is left to the render layer — it avoids a
        heavy get_node per note here, and the consumer already holds the node.
        """
        events = self.query_traces(
            ref_type='journal_note', scale=scale, ref_id=subject,
            session_id=session_id, chain_suffix=unit, hours=None, limit=limit,
        ).get('events', [])
        return self._journal_notes_from_events(
            events, subject=subject, scale=scale, session_id=session_id,
            unit=unit, k=k)

    def _journal_notes_from_events(self, events, *, subject='', scale='',
                                   session_id='', unit='', k=None,
                                   compact=False):
        """One lifecycle reducer for history and working continuity.

        Input is newest-first. Compaction affects only the working read;
        explicit subject history and journal_notes() retain every row.
        """
        from .trace_contract import (JOURNAL_CONTINUITY_RUNS,
                                      JOURNAL_CONTINUITY_RUNS_DEFAULT,
                                      JOURNAL_RESOLVE_TAGS, JOURNAL_OPEN_TAGS,
                                      JOURNAL_OPEN_PIN_CAP, resolve_target,
                                      journal_key)
        open_meta = {}   # id(event) → {'open_runs': N, 'first_seen': iso}
        if not subject:  # continuity: resolve-filter + K runs + open pins
            if k is None:
                key = 's1e' if scale == 's1' else unit
                k = JOURNAL_CONTINUITY_RUNS.get(key, JOURNAL_CONTINUITY_RUNS_DEFAULT)

            def _tag(e):
                return journal_key((e.get('metadata') or {}).get('tag'))

            def _subj(e):
                return journal_key(e.get('ref_id'))

            def _note(e):
                return (e.get('metadata') or {}).get('note') or ''

            # Every subject actually written in this fetch — the guard that
            # keeps resolve_target from inventing a retire target.
            known_subjects = {_subj(e) for e in events if _subj(e)}

            # Pass 1 — resolve-filtering, newest→oldest across ALL fetched
            # events: a `resolved`/`retire` note retires every strictly-older
            # note with the same (normalized) subject. Read-time only; the
            # trace rows are untouched. The resolve note itself stays until it
            # ages out — it documents the resolution.
            resolved_seen, alive = set(), []
            for e in events:                       # created_at DESC
                s = _subj(e)
                if s and s in resolved_seen:
                    continue
                if s and _tag(e) in JOURNAL_RESOLVE_TAGS:
                    # The slot the encoder filled may be the referenced note's
                    # TAG rather than its subject; recover the real target.
                    resolved_seen.add(
                        resolve_target(s, _note(e), known_subjects))
                alive.append(e)

            # Pass 2 — the K-run window over surviving notes.
            seen, window = [], []
            for e in alive:
                ch = e.get('chain_id') or ''
                if ch not in seen:
                    if len(seen) >= k:
                        break
                    seen.append(ch)
                window.append(e)

            # Pass 3 — open pins: the newest surviving `open`-tagged note per
            # subject stays visible beyond the window until resolved (capped).
            # ×N persistence = distinct runs mentioning the subject, computed
            # over the post-resolve SURVIVORS — a resolve closes the epoch, so
            # a re-opened subject starts at ×1 with a fresh first_seen. The
            # reader does the bumping, never the encoder.
            # Horizon note: everything here operates within the `limit` newest
            # fetched events (~dozens of runs). That bound is the backstop of
            # last resort — the ×N nudge fires at JOURNAL_OPEN_NUDGE_RUNS,
            # long before any pin could age past the horizon.
            runs_by_subj, first_seen = {}, {}
            for e in alive:
                s = _subj(e)
                if not s:
                    continue
                if _tag(e) in JOURNAL_RESOLVE_TAGS:
                    continue  # a resolve closes an epoch; it doesn't open one
                ch = e.get('chain_id')
                if ch:
                    runs_by_subj.setdefault(s, set()).add(ch)
                c = e.get('created_at') or ''
                if c and (s not in first_seen or c < first_seen[s]):
                    first_seen[s] = c

            in_window = {id(e) for e in window}
            pinned, pinned_subjects, pins_dropped = [], set(), 0
            for e in alive:                        # newest first
                if _tag(e) not in JOURNAL_OPEN_TAGS:
                    continue
                s = _subj(e)
                if not s or s in pinned_subjects:
                    continue
                pinned_subjects.add(s)
                open_meta[id(e)] = {
                    'open_runs': len(runs_by_subj.get(s, ())) or 1,
                    'first_seen': first_seen.get(s, ''),
                }
                if id(e) not in in_window:
                    if len(pinned) < JOURNAL_OPEN_PIN_CAP:
                        pinned.append(e)
                    else:
                        pins_dropped += 1
            if pins_dropped:
                # Loud by default: a dropped pin is an unresolved open item
                # silently leaving the encoder's sight.
                self._log_warning(
                    'journal_open_pin_overflow',
                    '%s/%s: %d open item(s) beyond the %d-pin cap dropped from '
                    'continuity — resolve or promote some' % (
                        scale, unit or session_id, pins_dropped,
                        JOURNAL_OPEN_PIN_CAP))

            events = window + pinned
        if compact:
            # One current lifecycle per subject, including INSIDE K. Keep
            # distinct ordinary observations; a subject is not one thought.
            lifecycle_seen, observations, compacted = set(), set(), []
            for e in events:
                tag = journal_key((e.get('metadata') or {}).get('tag'))
                subj = journal_key(e.get('ref_id'))
                if tag in JOURNAL_RESOLVE_TAGS + JOURNAL_OPEN_TAGS:
                    target = (resolve_target(subj, _note(e), known_subjects)
                              if tag in JOURNAL_RESOLVE_TAGS else subj)
                    if target in lifecycle_seen:
                        continue
                    lifecycle_seen.add(target)
                    if target != subj:
                        e = dict(e, ref_id=target)
                else:
                    key = (tag, subj, _note(e),
                           (e.get('metadata') or {}).get('undelivered', ''))
                    if key in observations:
                        continue
                    observations.add(key)
                compacted.append(e)
            events = compacted
        return [{
            'tag': (e.get('metadata') or {}).get('tag', ''),
            'note': (e.get('metadata') or {}).get('note', ''),
            'undelivered': (e.get('metadata') or {}).get('undelivered', ''),
            'subject': e.get('ref_id', ''),
            'chain_id': e.get('chain_id', ''),
            'created_at': e.get('created_at', ''),
            **({'event_id': e['id']} if compact else {}),
            **open_meta.get(id(e), {}),
        } for e in events]

    def journal_view(self, *, scale, unit='', session_id='', previous=None):
        """Current scoped continuity, with frozen private selection in a run.

        The binding owns invocation lifetime; this door owns event reads and
        lifecycle reduction. `previous` is a read receipt, never a second
        writable journal. Only committed lifecycle events affecting its
        selected subjects enter later requests. New observations wait for a
        fresh invocation. History remains available through journal_notes.
        """
        from .trace_contract import (
            JOURNAL_VIEW_HISTORY_LIMIT, JOURNAL_VIEW_PAGE_SIZE,
            JOURNAL_VIEW_MAX_PAGES, JOURNAL_LIFECYCLE_TAGS,
            JOURNAL_RESOLVE_TAGS, journal_key, resolve_target)
        if (scale == 's1' and (not session_id or unit)
                or scale == 's2' and (not unit or session_id)
                or scale not in ('s1', 's2')):
            raise ValueError('journal view requires an S1 session or S2 unit')
        scope = (scale, unit, session_id)
        if previous is not None and previous['scope'] != scope:
            raise ValueError('journal view receipt belongs to another scope')
        if previous is None:
            page = self._trace_dal.journal_page(
                scale=scale, unit=unit, session_id=session_id,
                limit=JOURNAL_VIEW_HISTORY_LIMIT)
            events = page['events']
            notes = self._journal_notes_from_events(
                events, scale=scale, unit=unit, session_id=session_id,
                compact=True)
            return {'scope': scope, 'events': events, 'notes': notes,
                    'cursor': page['cursor'],
                    'admitted': {n['event_id'] for n in notes},
                    'subjects': {journal_key(n['subject']) for n in notes},
                    'history_truncated': page['truncated'],
                    'changes_pending': False}

        # Build a new receipt; a read failure leaves the binding's last
        # committed view and cursor intact, including partially read pages.
        events = list(previous['events'])
        admitted = set(previous['admitted'])
        cursor = previous['cursor']
        subjects = previous['subjects']
        for _ in range(JOURNAL_VIEW_MAX_PAGES):
            page = self._trace_dal.journal_page(
                scale=scale, unit=unit, session_id=session_id,
                after=cursor, limit=JOURNAL_VIEW_PAGE_SIZE)
            for e in page['events']:
                meta = e.get('metadata') or {}
                tag = journal_key(meta.get('tag'))
                subj = journal_key(e.get('ref_id'))
                target = (resolve_target(subj, meta.get('note'), subjects)
                          if tag in JOURNAL_RESOLVE_TAGS else subj)
                if tag in JOURNAL_LIFECYCLE_TAGS and target in subjects:
                    events.append(e)
                    admitted.add(e['id'])
            cursor = page['cursor']
            if not page['truncated']:
                break
        events.sort(key=lambda e: e['journal_cursor'], reverse=True)
        notes = self._journal_notes_from_events(
            events, scale=scale, unit=unit, session_id=session_id,
            k=len(events), compact=True)
        # K chose the private notes at invocation start, not anew per batch.
        # Only admitted post-cursor updates can extend that selection;
        # supporting history must not become newly eligible context.
        notes = [n for n in notes if n['event_id'] in admitted]
        return dict(previous, events=events, notes=notes, cursor=cursor,
                    admitted=admitted,
                    changes_pending=page['truncated'])

    def write_journal_notes(self, *, final_text, chain_id, scale, session_id=''):
        """Write door — the mirror of journal_notes (read). Extract the
        encoder's `## Review` fenced block, parse it, and write each note as its
        own journal_note trace row (event_type='delta', ref_id=subject), all
        sharing the run's chain_id.

        The JOURNAL_ADDRESSED_TAGS notes (tell/ask) are NOT written — they are
        messages to the people working, not residue for the next run — and
        come back under 'addressed' for the caller to route; this run's
        resolve-verb lines come back under 'resolved' ({subject, note}) so the
        caller can close what they name. The traces layer only partitions; it
        never imports a channel and holds no routing policy.

        Returns a structured result so the caller (and the trace) can see what
        happened: `{'written': int, 'malformed': int, 'status': str,
        'addressed': [...], 'resolved': [...]}` where status is one of:
          • 'ok'                 — a non-empty review processed (counts tell the rest)
          • 'salvaged'           — no `## Review` heading, but a heading-less fence
                                   of valid notes was harvested (drift, logged loud)
          • 'empty_review'       — a fenced review that was empty (a legit clean run)
          • 'no_review_section'  — the encoder emitted no `## Review` at all
          • 'no_review_extracted'— `## Review` present but no parseable fence (drift)
          • 'error'              — an unexpected failure (isolated; see below)

        LOUD BY DEFAULT — nothing is dropped silently. The encoder is expected
        to ALWAYS emit a `## Review` section (empty fence on a clean run), so a
        missing section or a broken fence is real drift and gets a warning. A
        malformed line, a subject-less note, or unbuildable metadata each logs
        loud and is skipped — one bad note never sinks the rest. If notes parsed
        but none survived, that's an accidental full drop → loud. The whole body
        is failure-isolated: any unexpected error is logged loud and swallowed —
        a journal write must never break or roll back the encoder's actual run.
        """
        from .trace_contract import (extract_review_block, parse_journal_notes,
                                      salvage_review_fence,
                                      JOURNAL_REVIEW_MARKER, is_addressed,
                                      JOURNAL_RESOLVE_TAGS, journal_key)

        def _result(written, malformed, status, addressed=(), resolved=()):
            return {'written': written, 'malformed': malformed,
                    'status': status, 'addressed': list(addressed),
                    'resolved': list(resolved)}

        try:
            salvaged = False
            block = extract_review_block(final_text)
            if block is None:
                # Drift salvage: the encoder sometimes writes a valid notes
                # fence but drops the heading — harvest it (strict gate in
                # salvage_review_fence) rather than lose the batch's residue.
                # Still a warning: drift stays visible, just no longer lossy.
                block = salvage_review_fence(final_text)
                if block is not None:
                    salvaged = True
                    self._log_warning(
                        'journal_note_review_salvaged',
                        'chain=%s: no %r heading, but a valid heading-less notes '
                        'fence was found — salvaged (format drift)'
                        % (chain_id, JOURNAL_REVIEW_MARKER))
                elif JOURNAL_REVIEW_MARKER in (final_text or ''):
                    self._log_warning(
                        'journal_note_no_review_extracted',
                        'chain=%s: %r present but no parseable fenced block'
                        % (chain_id, JOURNAL_REVIEW_MARKER))
                    return _result(0, 0, 'no_review_extracted')
                else:
                    self._log_warning(
                        'journal_note_no_review_section',
                        'chain=%s: encoder final_text (%d chars) has no %r section'
                        % (chain_id, len(final_text or ''), JOURNAL_REVIEW_MARKER))
                    return _result(0, 0, 'no_review_section')
            if block == '':
                # Fenced review present but empty — the legit "clean run, nothing
                # to note" case. Visible (debug), not an alarm.
                self.log_debug('journal_note_empty_review', 'write_journal_notes',
                               chain_id=chain_id)
                return _result(0, 0, 'empty_review')

            notes, malformed = parse_journal_notes(block)
            for raw in malformed:
                self._log_warning('journal_note_malformed',
                                  'chain=%s: %s' % (chain_id, raw[:200]))

            # One pass partitions the parsed notes (parse_journal_notes
            # already strips every field and rejects empty subjects):
            # addressed lines go back to the caller, the rest are residue,
            # and resolve lines are also named so the caller can close items.
            residue, addressed, resolved = [], [], []
            for n in notes:
                if is_addressed(n.get('tag')):
                    addressed.append(n)
                    continue
                residue.append(n)
                if journal_key(n.get('tag')) in JOURNAL_RESOLVE_TAGS:
                    resolved.append(n)
        except Exception as e:
            self._log_error('journal_note_write_failed', e, 'chain=%s' % chain_id)
            return _result(0, 0, 'error')
        # The row write is isolated on its own: a failed batch must not take
        # the addressed lines down with it — those are messages to a person,
        # and the caller still routes them (and can keep them as residue).
        try:
            written = self.write_journal_note_rows(
                residue, chain_id=chain_id, scale=scale, session_id=session_id)
        except Exception as e:
            self._log_error('journal_note_write_failed', e,
                            'chain=%s: %d residue row(s) lost; addressed lines '
                            'handed back' % (chain_id, len(residue)))
            return _result(0, len(malformed), 'error', addressed, resolved)
        if residue and not written:
            self._log_warning(
                'journal_note_all_dropped',
                'chain=%s: parsed %d notes but wrote 0 (all failed subject/build)'
                % (chain_id, len(residue)))
        return _result(written, len(malformed),
                       'salvaged' if salvaged else 'ok', addressed, resolved)

    def write_journal_note_rows(self, notes, *, chain_id, scale, session_id=''):
        """Write notes ({tag, subject, note[, undelivered]} dicts) as
        journal_note rows on `chain_id` — the row-writing half of
        write_journal_notes, also the door a binding uses to keep an
        addressed line as residue when the Thalamus rejected it
        (`undelivered` = the door's reason). Notes here may be hand-built, so
        the per-note guards stay: a subject-less or unbuildable note is
        skipped and warned, never sinks the batch. Returns the number of rows
        written."""
        from .trace_contract import build_journal_note_metadata
        events = []
        for n in notes:
            subject = (n.get('subject') or '').strip()
            if not subject:
                self._log_warning('journal_note_no_subject',
                                  'chain=%s: %s' % (chain_id, str(n)[:200]))
                continue
            try:
                meta = build_journal_note_metadata(
                    note=n.get('note') or '', tag=n.get('tag', ''),
                    undelivered=n.get('undelivered', ''))
            except (ValueError, KeyError) as e:
                self._log_warning('journal_note_build_failed',
                                  'chain=%s: %s | %s' % (chain_id, e, str(n)[:160]))
                continue
            events.append({
                'chain_id': chain_id, 'scale': scale, 'event_type': 'delta',
                'ref_type': 'journal_note', 'ref_id': subject,
                'summary': meta['note'][:80], 'metadata': meta,
                'session_id': session_id,
            })
        if events:
            self._trace_dal.append_batch(events)
        return len(events)

    def write_thalamus_filed(self, *, chain_id, session_id, item_id, source,
                             body, target_session='', needs_answer=False,
                             dedup_key='', route='queue', filing='new'):
        """Write door for a Thalamus filing made from a producer's RUN — one
        `thalamus_filed` delta row on the run's chain (ref_id = item id), the
        sibling of write_journal_notes' rows. The scale is the chain's
        (trace_contract.scale_for_chain); the payload shape is the contract's
        (build_thalamus_filed_metadata, validated at the write boundary).

        Failure-isolated: the item is already committed when this runs, and a
        trace failure must never undo or mask a filing — logged loud, the
        caller's result stands. A chain whose scale the contract does not
        register for this ref_type (an s0 chain — a filing is a run's act,
        not a turn's) is exactly such a failure: LOUD in the errors log, no
        row."""
        try:
            from .trace_contract import (REF_THALAMUS_FILED, scale_for_chain,
                                          build_thalamus_filed_metadata)
            meta = build_thalamus_filed_metadata(
                source=source, body=body, target_session=target_session,
                needs_answer=needs_answer, dedup_key=dedup_key, route=route,
                filing=filing)
            return self._trace_dal.append(
                chain_id=chain_id, scale=scale_for_chain(chain_id),
                session_id=session_id, event_type='delta',
                ref_type=REF_THALAMUS_FILED, ref_id=item_id,
                summary=meta['body'][:80], metadata=meta)
        except Exception as e:
            self._log_error('thalamus_filed_trace_failed', e,
                            'chain=%s item=%s — filing stands, trace lost'
                            % (chain_id, item_id))
            return None

    def write_session_arc(self, *, final_text, session_id, limit=800):
        """Write door for the session arc — the journal mechanism's second
        component (design §7.2: Encode → Arc → Review). Extract the encoder's
        `## Arc` fenced one-liner and ACCUMULATE it onto the per-session
        digest at `session_context_{session_id}` (read back by
        session_context_for: the Frame's 'Current focus', the next encoding
        run's context block, and recall ranking).

        `limit` bounds the accumulated digest (chars); truncation drops the
        OLDEST lines from the front — the digest is a rolling journey, newest
        movement last. Caller passes its contract value (S1E:
        ENCODING_AGENT['session_context_limit']).

        Returns {'written': bool, 'status': str} where status is one of:
          • 'ok'              — a non-empty arc line appended
          • 'empty_arc'       — fenced arc present but empty (legit: nothing
                                progressed this run)
          • 'no_arc_section'  — no `## Arc` at all (drift — the block was
                                injected, the encoder ignored it)
          • 'no_arc_extracted'— `## Arc` present but no parseable fence (drift)
          • 'error'           — unexpected failure (isolated; never breaks
                                the encoder's run)

        LOUD BY DEFAULT — an opted-in encoder is expected to ALWAYS emit
        `## Arc` (empty fence when nothing moved), so a missing section or a
        broken fence logs a warning. An empty fence is visible (debug), not
        an alarm.
        """
        from .trace_contract import extract_arc_block, JOURNAL_ARC_MARKER
        try:
            if not session_id:
                return {'written': False, 'status': 'error'}
            block = extract_arc_block(final_text)
            if block is None:
                if JOURNAL_ARC_MARKER in (final_text or ''):
                    self._log_warning(
                        'session_arc_no_arc_extracted',
                        'session=%s: %r present but no parseable fenced block'
                        % (session_id[:8], JOURNAL_ARC_MARKER))
                    return {'written': False, 'status': 'no_arc_extracted'}
                self._log_warning(
                    'session_arc_no_arc_section',
                    'session=%s: encoder final_text (%d chars) has no %r section'
                    % (session_id[:8], len(final_text or ''), JOURNAL_ARC_MARKER))
                return {'written': False, 'status': 'no_arc_section'}
            if block == '':
                self.log_debug('session_arc_empty', 'write_session_arc',
                               session_id=session_id)
                return {'written': False, 'status': 'empty_arc'}
            # Accumulate: newline-joined journey, oldest lines truncated from
            # the front at a line boundary.
            new_line = block.splitlines()[0].strip()  # ONE line by contract
            existing = self.session_context_for(session_id)
            combined = (existing + '\n' + new_line) if existing else new_line
            if len(combined) > limit:
                truncated = combined[len(combined) - limit:]
                nl_idx = truncated.find('\n')
                if 0 <= nl_idx < 60:
                    truncated = truncated[nl_idx + 1:]
                combined = truncated
            self.set_config('session_context_' + session_id, combined)
            return {'written': True, 'status': 'ok'}
        except Exception as e:
            self._log_error('session_arc_write_failed', e,
                            'session=%s' % (session_id or '')[:8])
            return {'written': False, 'status': 'error'}

    # ── Episodic recall ──

    def recall_episodes(self, query: str = None, contains: str = None,
                        session_id: str = None, session_ids=None,
                        scale: str = 's0', event_type: str = None,
                        ref_type=None, older_than: str = None,
                        younger_than: str = None, sort_order: str = 'desc',
                        limit: Optional[int] = EPISODE_DEFAULT_LIMIT
                        ) -> Dict[str, Any]:
        """Search/filter trace_events and return full episode records — the
        decode-over-traces sibling of recall.

        Needles (composable): query (semantic — ranks vs trace_embeddings;
        s0-only today) and/or contains (substring over summary + metadata).
        Frame: session_id | session_ids, scale (default 's0'; '' = all),
        event_type, ref_type, time bounds (older_than / younger_than accept ISO
        or relative shorthand). With no session scope and no time bound at all,
        a default EPISODE_DEFAULT_WINDOW_DAYS lower bound is applied so an
        unbounded scan can't walk the whole append-only table; an explicit
        older_than is left as-is (no forced floor).

        ref_type: a str (one type) or a list (several). UNSET → the conversation
        default, PINNED to operator dialogue (OPERATOR_DIALOGUE_REF_TYPES —
        deliberately not the dial: flipped correspondents like self_message /
        thalamus_delivery stay opt-in via explicit ref_type, like tool_result)
        at s0 — so the default tracks the contract and can't drift, and
        non-conversational s0 traffic (tool_result, heartbeat, structural
        deltas) stays out of the common query. Pass ref_type='tool_result' for
        the "what I did with files/commands" lens, or SAID_AND_DID_REF_TYPES for
        the interleaved said+did timeline. Non-s0 scales have no conversational
        notion, so unset means all ref_types at that scale.

        Ordering: with query, ranked by relevance (each episode carries _score);
        otherwise by created_at (sort_order 'desc' default = latest first).

        Returns {'episodes': [<full trace records>], 'ranked_by':
                 'relevance'|'time', 'truncated': ...} — two shapes by path,
                 matching the truncation contract (contract.py): the TIME path
                 (window-coverage claim) returns False, or the full payload
                 dict {limit, coverage_start, coverage_end, note} when the +1
                 probe proves the window held more; the SEMANTIC path (ranked
                 top-k — truncation is its contract) keeps a bare bool: True
                 when more matched than were ranked (hit limit or the
                 candidate cap). Truthiness works for both; only the dict
                 form carries coverage details (and triggers the MCP banner).
        """
        from .trace_contract import OPERATOR_DIALOGUE_REF_TYPES
        # Honest limit (mirrors filter_nodes): the signature default is a
        # bounded page (EPISODE_DEFAULT_LIMIT), and EXPLICIT limit=None is the
        # opt-in for unbounded (all episodes in the window — internal window
        # pulls). A number is an honest page with no silent ceiling; the
        # agent-facing default + cap live at the dispatch door
        # (_handle_recall_episodes: absent → EPISODE_DEFAULT_LIMIT, clamp to
        # EPISODE_MAX_LIMIT). The time path's +1 probe rides limit+1 directly.
        if limit is not None:
            limit = max(int(limit), 1)
        younger_iso = _resolve_time_bound(younger_than)
        older_iso = _resolve_time_bound(older_than)
        if (not younger_iso and not older_iso
                and not session_id and not session_ids):
            younger_iso = iso_cutoff(days=EPISODE_DEFAULT_WINDOW_DAYS)

        # ref_type whitelist: explicit (str→[str] or list) wins; else the s0
        # conversation default comes from the contract dial (no hardcoded list
        # to drift); other scales have no conversational notion → no filter.
        if ref_type:
            ref_types = [ref_type] if isinstance(ref_type, str) else list(ref_type)
        elif scale == 's0':
            ref_types = list(OPERATOR_DIALOGUE_REF_TYPES)
        else:
            ref_types = None
        common = dict(
            contains=contains or '', scale=scale or '',
            event_type=event_type or '', ref_types=ref_types,
            session_id=session_id or '', session_ids=session_ids,
            younger_than=younger_iso, older_than=older_iso)

        # Semantic path: rank a lean candidate scan (id+vector via one JOIN, no
        # metadata decode) by cosine, then batch-hydrate full records for ONLY
        # the top-k. Degrades to the time path on any non-ValueError error
        # ValueError (e.g. both session
        # forms) surfaces.
        if query:
            try:
                qvec = embedder.embed_query(query)
                if qvec:
                    cands = self._trace_dal.filter_event_vectors(
                        limit=EPISODE_SEMANTIC_CANDIDATE_CAP, **common)
                    if cands:
                        scored = sorted(
                            ((embedder.cosine_similarity(qvec, vec), tid)
                             for tid, vec in cands),
                            key=lambda x: -x[0])[:limit]
                        recs = {r['id']: r for r in self._trace_dal.get_by_ids(
                            [tid for _, tid in scored])}
                        episodes = [dict(recs[tid], _score=round(score, 4))
                                    for score, tid in scored if tid in recs]
                        return {'episodes': episodes,
                                # Honest even at limit=None: the candidate scan
                                # caps at EPISODE_SEMANTIC_CANDIDATE_CAP, so a
                                # saturated scan means matches were dropped —
                                # flag it (the docstring's "hit limit or the
                                # candidate cap") rather than claim completeness.
                                'truncated': (
                                    len(cands) >= EPISODE_SEMANTIC_CANDIDATE_CAP
                                    or (limit is not None and len(cands) > limit)),
                                'ranked_by': 'relevance'}
                else:
                    # embed_query returned no vector → embedder unavailable.
                    # A no-MATCH (cands empty) is legit and stays quiet; an
                    # un-embeddable semantic request is a real degrade — surface
                    # it instead of silently answering by recency.
                    self._log_error(
                        'recall_episodes_embed_unavailable',
                        RuntimeError('embed_query returned no vector'),
                        'semantic query requested but not embeddable (embedder '
                        'unavailable?); degraded to time path')
            except ValueError:
                raise
            except Exception as e:
                self._log_error('recall_episodes_semantic', e,
                                'semantic rerank failed; degraded to time path')
            # qvec unavailable / nothing embedded / degraded → time path.

        # Time path: indexed WHERE + ORDER BY created_at + LIMIT early-exits, so
        # only limit+1 rows are fetched and decoded. This path claims WINDOW
        # coverage (time-ordered scan), so saturation gets the full unified
        # payload — exact via the +1 probe (the old `len >= limit` heuristic
        # false-flagged exact fits). The semantic path above keeps its bare
        # bool: ranked top-k, where truncation is the contract, not a lie.
        order = sort_order if sort_order in ('asc', 'desc') else 'desc'
        if limit is None:
            # Unbounded: fetch every matching event. You asked for all and got
            # all, so there is nothing to probe or flag.
            fetched = self._trace_dal.filter_events(
                sort_order=order, limit=None, **common)
            return {'episodes': fetched, 'truncated': False, 'ranked_by': 'time'}
        fetched = self._trace_dal.filter_events(
            sort_order=order, limit=limit + 1, **common)
        return _flag_truncation(
            {'episodes': fetched[:limit], 'truncated': False,
             'ranked_by': 'time'},
            fetched, limit, key='episodes')

    # ── Conversation ──

    def get_conversation(self, session_id: str, limit: int = 20,
                         with_judge_output: bool = True,
                         with_surfaced: bool = False,
                         exclude_trace_id: str = None,
                         older_than: str = None, *,
                         around_timestamp: str = None,
                         before: int = 10, after: int = 5,
                         include_timestamp_ties: bool = False) -> List[Dict]:
        """Get conversation turns within one required session.

        The simple path — S1E, scribe_due, the surface window, the LAF
        moment stack: anything that knows its session_id and wants the last
        N turns. A centered window uses the same reader and row contract;
        get_conversation_around resolves a memory's anchors before calling it.
        A timestamp positions the window; it never selects a session.

        Returns: [{role, ref_type, content, timestamp, trace_id, judge_output}]
            ref_type: the CORRESPONDENT axis (user_message = operator,
                      self_message = a stream, thalamus_delivery = the brain)
                      passed through from get_session_turns — the encoder's
                      render keys speaker elements on it.
            trace_id: 8-char hex id from trace_events (v29) — rendered as the
                      `trace="…"` attribute on the S1 encoder's timeline turns,
                      which it copies into source_refs.
            judge_output: surface selection from S1R for the user turn (if any).
                      Filling it costs an extra query over the window's recall
                      chains — callers that only read role/content (the
                      scribe_due poll) pass with_judge_output=False.
            with_surfaced=True adds `surfaced` per user turn — the memories
                      the surface selected ([{id, title}]); the v13 XML
                      surface layout's <shown> source.
            exclude_trace_id drops one row in SQL — mid-turn readers that
                      want PREVIOUS turns only pass the current prompt's
                      trace id (see get_session_turns for the interrupt
                      subtlety).
            older_than: ISO strict `created_at <` bound, applied in SQL —
                      the replay as-of cut: "the last N turns as of that
                      instant", not "the last N turns now, minus the future".
            around_timestamp: ISO center for a historic window, using
                      before/after instead of the recent-turn limit.
            include_timestamp_ties: keep whole timestamp groups at historic
                      window boundaries, even if this exceeds before/after.
        """
        try:
            if not session_id:
                raise ValueError('conversation requires a session_id')
            turns = self._trace_dal.get_session_turns(
                session_id, limit=limit, with_judge_output=with_judge_output,
                with_surfaced=with_surfaced, exclude_trace_id=exclude_trace_id,
                older_than=older_than, around_timestamp=around_timestamp,
                before=before, after=after,
                include_timestamp_ties=include_timestamp_ties)
            out = []
            for t in turns:
                row = {'role': t['role'],
                       'ref_type': t.get('ref_type', ''),
                       'trace_id': t.get('trace_id'),
                       'content': t.get('content', ''),
                       'timestamp': t.get('timestamp', ''),
                       'judge_output': t.get('judge_output', '')}
                if with_surfaced:
                    row['surfaced'] = t.get('surfaced', [])
                out.append(row)
            return out
        except Exception as e:
            # Empty-list degrade keeps every consumer alive (Scribe cadence,
            # surface window, moment stack) — but never silently.
            self._log_error('get_conversation', e,
                            'session=%s' % (session_id or '')[:8])
            return []

    def turns_since_last_encode(self, session_id: str) -> int:
        """Conversational turns this session has had since its last S1 encode.

        The S1 Scribe's cadence signal, read LIVE from traces instead of a
        maintained counter. The old `conversational_count` desynced across
        resume/restart (boot reset it while the traces stayed truthful), which
        starved the Scribe — so the gate now derives the count from the event
        log that never lies.

        Anchors on the most recent SUCCESSFUL run — the latest `encoding_run`
        delta — not the latest attempt. Anchoring on `encoding_prompt` (the
        old form) meant a run that failed AFTER writing its prompt trace
        silently reset the cadence: its turns were skipped, not retried, and
        a failed tail encode was never retried at all (found live 2026-07-28,
        fb78aab9 #38). The count anchors at that successful run's START (its
        chain's encoding_prompt timestamp), so turns that arrived while it ran
        still count as unencoded. Failed attempts stay "due"; the daemon's
        retry cooldown paces the re-fire and scribe_repeated_failure escalates
        a wedged session. No prior successful encode → counts all turns, so a
        fresh session fires at the threshold. Same turn definition the encoder
        reads (trace_contract).
        """
        if not session_id:
            return 0
        last = self._trace_dal.get_by_ref_type(
            'encoding_run', scale='s1', session_id=session_id,
            hours=None, limit=1)
        since = ''
        if last:
            # The run's start = the LAST encoding_prompt in its chain (a
            # failed attempt retried at the same stop shares the chain and
            # writes an earlier prompt; ASC + [-1] anchors on the successful
            # attempt's own start). ref_type-bounded chain pull: 1-2 rows
            # decoded on the scribe poll, vs the old newest-50 scan whose
            # Python chain match silently fell back to the run's END time
            # past 50 retries (2026-08-07 review, findings 5+7).
            chain = last[0].get('chain_id') or ''
            prompts = (self._trace_dal.get_chain(
                chain, ref_type='encoding_prompt') if chain else [])
            since = (prompts[-1]['created_at'] if prompts
                     else last[0]['created_at'])
        return self._trace_dal.conversational_turns_since(session_id, since)

    # ── Presence (self-channel liveness reads over S0 traces) ──

    def present_streams(self, exclude_session: str = '',
                        window_min: float = 30, limit: int = 5,
                        sort_by: str = 'recency') -> list:
        """Streams of thought awake RIGHT NOW — the self-channel presence roster.

        Distinct from `live_sessions()`: that one is "recent meaningful work"
        (≥min_messages, survives week-long gaps, for the Frame's cross-session
        slots). `present_streams` is WALL-CLOCK "who is awake this moment" —
        sessions whose session_state row updated within the last `window_min`
        minutes, newest first, excluding the caller.

        Wall-clock is correct here: presence is real-time, not conversation-time,
        so it's exempt from the conversation_now() rule like other bookkeeping
        reads. See docs/BOOT-REIGNITION.md (presence at scale).

        Liveness is sourced from real-turn S0 traces (TraceDAL), NOT
        session_state.updated_at — the latter moves on any session-state change
        (a boot, a SessionEnd save, a flush of a changed ctx), none of which is
        a turn. Traces only record actual turns, so the signal is honest.

        Returns [{'session_id': str, 'live_recency': iso, 'focus': str}], newest
        first. `live_recency` is the PRESENCE_LIVE_REF_TYPES clock — heartbeats
        INCLUDED, so it answers "can this stream be reached", never "has it gone
        quiet". The name is deliberate on both halves: not `updated_at` (a
        session_state column this value is not), and not `last_turn` (the contract
        says a wakeup re-arm is never a turn, yet this counts one). Its sibling
        `conv_recency` — computed by the same query, not projected here — is the
        clock for "has work happened". `focus` is that
        session's latest conversational turn — user_message OR assistant_message
        per trace_contract.OPERATOR_DIALOGUE_REF_TYPES, excluding the wake-envelope
        marker (raw — render layer trims it).
        """
        from .clock import iso_cutoff
        try:
            rows = self._trace_dal.active_sessions_by_turn(
                iso_cutoff(minutes=window_min),
                exclude_session=exclude_session, limit=limit, sort_by=sort_by)
            return [{'session_id': r['session_id'], 'live_recency': r['live_recency'],
                     'focus': r['focus'], 'turn_count': r.get('turn_count', 0)}
                    for r in rows]
        except Exception as e:
            try:
                self._log_error('present_streams_query', e,
                                'window_min=%s limit=%d' % (window_min, limit))
            except Exception:
                pass
            return []

    def session_activity(self, session_id: str, msg_limit: int = 2) -> dict:
        """Per-session activity snapshot for self_peek — first/last turn and the
        last conversational messages, from real S0 traces (TraceDAL). Mirrors
        present_streams (wall-clock, presence-adjacent, read-only). Returns {} on
        error so a peek degrades gracefully rather than raising."""
        if not session_id:
            return {}
        try:
            return self._trace_dal.session_activity(
                session_id, msg_limit=msg_limit)
        except Exception as e:
            try:
                self._log_error('session_activity_query', e,
                                'session=%s' % (session_id or '')[:8])
            except Exception:
                pass
            return {}

    def get_conversation_around(self, node_id: str = None,
                                session_id: str = None,
                                timestamp: str = None,
                                before: int = 10, after: int = 5) -> Dict:
        """Conversation context grouped by recorded session and excerpt.

        Node lookups expand ALL get_source_refs through get_traces. Ref order
        never chooses a session. With no refs, exact creation evidence anchors
        the window. An explicit session remains authoritative, with timestamp
        (or the node's created_at) positioning its window. Session-known callers
        needing only turns can use get_conversation directly.

        Returns {basis, conversations, missing_trace_ids}. Each conversation is
        {session_id, windows: [{anchor_trace_ids, turns}]}. Overlapping excerpts
        merge by shared trace ids within a session; gaps stay separate. before
        and after count exchanges per anchor, as in get_conversation. Boundary
        timestamp ties stay together, so a cited row cannot be clipped by a tie
        and the nominal window size may expand.

        missing_trace_ids names anchors whose trace, session stamp or conversation
        could not be read. Valid excerpts survive missing citations. A failed
        source lookup never substitutes the node's creation conversation.
        """
        result = {'basis': None, 'conversations': [], 'missing_trace_ids': []}
        try:
            if before < 0 or after < 0:
                raise ValueError('conversation window sizes must be nonnegative')
            if session_id:
                result['basis'] = 'explicit_session'
                center = timestamp or self._resolve_node_timestamp(node_id)
                anchors = [{'id': None, 'session_id': session_id, 'created_at': center}]
            elif node_id:
                refs = sorted(set(self.get_source_refs(node_id)))
                if refs:
                    result['basis'] = 'source_refs'
                    # Preserve the unavailable list if the batch read fails.
                    result['missing_trace_ids'] = refs
                    by_id = {row['id']: row for row in self.get_traces(refs)}
                    result['missing_trace_ids'] = [ref for ref in refs if ref not in by_id]
                    anchors = [by_id[ref] for ref in refs if ref in by_id]
                else:
                    result['basis'] = 'creation_trace'
                    origin = self._node_creation_anchor(node_id)
                    anchors = [origin] if origin else []
            else:
                raise ValueError('conversation requires a recorded session and timestamp')

            if not anchors:
                raise ValueError('no recorded conversation anchors available')

            by_session = {}
            for anchor in sorted(anchors, key=lambda row: (
                    row.get('session_id') or '', row.get('created_at') or '', row.get('id') or '')):
                sid = anchor.get('session_id')
                center = timestamp or anchor.get('created_at')
                trace_id = anchor.get('id')
                if not sid or not center:
                    if trace_id:
                        result['missing_trace_ids'].append(trace_id)
                    self._log_error('get_conversation_around',
                                    ValueError('anchor lacks a recorded session or timestamp'),
                                    'trace=%s' % trace_id)
                    continue
                turns = self.get_conversation(
                    sid, around_timestamp=center, before=before, after=after,
                    with_judge_output=False, include_timestamp_ties=True)
                if not turns:
                    if trace_id:
                        result['missing_trace_ids'].append(trace_id)
                    continue
                for turn in turns:
                    turn.pop('judge_output', None)
                windows = by_session.setdefault(sid, [])
                anchor_ids = [trace_id] if trace_id else []
                have = {turn['trace_id'] for turn in windows[-1]['turns']} if windows else set()
                if have.intersection(turn['trace_id'] for turn in turns):
                    # Centers are ordered within each session, with identical
                    # window sizes. An overlapping excerpt extends the tail;
                    # preserving reader order also preserves timestamp ties.
                    windows[-1]['turns'].extend(turn for turn in turns if turn['trace_id'] not in have)
                    windows[-1]['anchor_trace_ids'].extend(anchor_ids)
                else:
                    windows.append({'anchor_trace_ids': anchor_ids, 'turns': turns})
            result['conversations'] = [
                {'session_id': sid, 'windows': windows} for sid, windows in by_session.items()]
        except Exception as e:
            self._log_error(
                'get_conversation_around', e,
                'node=%s session=%s' % (node_id or '', session_id or ''))
        result['missing_trace_ids'] = sorted(set(result['missing_trace_ids']))
        return result

    def _resolve_node_timestamp(self, node_id):
        """Get a node's created_at timestamp — exact id match via NodeDAL."""
        node = self._nodes.get_naked_node(node_id) if node_id else None
        return node['created_at'] if node else None

    def _node_creation_anchor(self, node_id):
        """Exact creation evidence for nodes without source refs.

        Prefer the encoding run's center to its per-write event. Conflicting
        recorded sessions are an attribution error, never a tie to guess at.
        """
        origins = [row for row in self._trace_dal.get_node_creation_traces(node_id)
                   if row['session_id']]
        if len({row['session_id'] for row in origins}) > 1:
            raise ValueError('conflicting sessions in node creation traces')
        if origins:
            hit = next((row for row in origins
                        if row['ref_type'] == 'encoding_run'), origins[0])
            return hit
        return None

    # ═══════════════════════════════════════════════════════════
    # Payload recorder (docs/TRACE-MODES-DESIGN.md)
    # ═══════════════════════════════════════════════════════════
    # Fat payloads (full prompts, per-round payloads, failed-run conversations)
    # live in FILES under {db_dir}/payloads/ — never in trace rows. The trace
    # row stays the authoritative index (bounded forensics + this pointer);
    # a deleted file degrades to a clean "pruned" read, so the user deleting
    # payloads from outside is always safe.

    def _payload_root(self):
        """{db_dir}/payloads — derived from THIS brain instance's db_path,
        never a global env var (a88343d6: env-resolved writer/reader path
        seams split-brain silently). IsolatedBrain and eval's fresh brains
        get their own root by construction."""
        return os.path.join(os.path.dirname(os.path.abspath(self.db_path)),
                            'payloads')

    def _payload_chain_dir(self, chain_id):
        """Date-first, chain-second. The date dir is the chain's FIRST
        payload's day: a chain dir already existing under yesterday is reused
        so a run straddling midnight stays one `ls`."""
        from .clock import iso_cutoff, iso_now
        today = iso_now()[:10]
        yesterday = iso_cutoff(days=1)[:10]
        for day in (today, yesterday):
            d = os.path.join(self._payload_root(), day, chain_id)
            if os.path.isdir(d):
                return d, day
        d = os.path.join(self._payload_root(), today, chain_id)
        os.makedirs(d, exist_ok=True)
        return d, today

    # Gate-config TTL (seconds). Per-round/per-prompt kinds put the gate
    # lookup on every agent round and every user prompt (judge), so the
    # config read is TTL-cached (the LAFEngine.config pattern — performance
    # charter, docs/TRACE-MODES-DESIGN.md). Brain.invalidate_interaction_caches
    # invalidates on a `trace_recording` flip or clear, so entering AND
    # leaving debug bite on the very next write, not a TTL later.
    TRACE_RECORDING_CFG_TTL_S = 60.0

    def _trace_recording_config(self):
        """Active `trace_recording` config dict, TTL-cached."""
        now = time.monotonic()
        if (getattr(self, '_trace_rec_cfg', None) is not None
                and now - getattr(self, '_trace_rec_cfg_ts', 0.0)
                < self.TRACE_RECORDING_CFG_TTL_S):
            return self._trace_rec_cfg
        self._trace_rec_cfg = self.get_interaction_config('trace_recording')
        self._trace_rec_cfg_ts = now
        return self._trace_rec_cfg

    def invalidate_trace_recording_cache(self):
        """Drop the cached gate config — called by
        Brain.invalidate_interaction_caches when the `trace_recording`
        pointer flips or clears."""
        self._trace_rec_cfg = None

    def _payload_kind_enabled(self, kind, chain_id):
        """Gate resolution for one payload kind. Effective policy = the
        contract's NORMAL defaults (complete by construction over
        PAYLOAD_KIND_EXT) overlaid with the active `trace_recording` config —
        so a kind added to the contract AFTER a brain's config was seeded
        still resolves (to its contract default) instead of silently never
        recording; the config-missing gap is loud-logged (rate-limited by
        _log_error's fingerprint dedup, so a wired call site reminds
        periodically rather than spamming or going quiet forever)."""
        from .trace_contract import TRACE_RECORDING_NORMAL
        cfg = self._trace_recording_config()
        cfg_kinds = cfg.get('kinds')
        effective = dict(TRACE_RECORDING_NORMAL['kinds'])
        if isinstance(cfg_kinds, dict):
            if kind not in cfg_kinds:
                self._log_error(
                    'record_payload_config_missing_kind',
                    ValueError('kind %r absent from active trace_recording '
                               'config — using contract default' % (kind,)),
                    context='chain=%s' % chain_id)
            effective.update(cfg_kinds)
        elif cfg:
            # Config exists but `kinds` is malformed — degrading to the
            # contract defaults must be loud, not the silent branch in an
            # otherwise loud function ("the interesting run is always the
            # one you weren't capturing").
            self._log_error(
                'record_payload_malformed_config',
                ValueError('trace_recording config has non-dict kinds: %r'
                           % (cfg_kinds,)),
                context='chain=%s' % chain_id)
        return bool(effective.get(kind))

    def record_payload(self, chain_id, kind, content, *, seq=None):
        """Write one payload file for a chain; return its pointer (path
        RELATIVE to db_dir) or None (gated off / unknown kind / empty /
        write failed — failures loud-log, never raise into the caller).

        The ONE capture writer — call sites hold zero knowledge of gates,
        paths, or formats. Gating: the `trace_recording` K-store interaction
        (per-kind on/off; modes are named config versions — see
        TRACE_RECORDING_NORMAL/DEBUG in trace_contract.py). Performance
        charter: compact JSON (no indent — the C json encoder holds the GIL;
        `jq .` the file instead), and this is never called on the recall hot
        path.

        Chain dirs are APPEND-ONLY: files open with O_EXCL and collisions get
        an attempt ordinal (`000-prompt.2.md`) — a Scribe idle-tail retry
        reuses the same chain_id, and overwriting would destroy exactly the
        failed attempt's forensics. The retention pass is the only deleter.
        chain_id is sanitized into a path segment: the writer must be as
        traversal-proof as read_payload's guard, or a chain containing '/'
        creates dirs outside payloads/ that the pruner never sees.
        """
        from .trace_contract import PAYLOAD_KIND_EXT
        pointer = None
        with self.loud('record_payload',
                       'chain=%s kind=%s' % (chain_id, kind)):
            if not chain_id or content in (None, '', {}, []):
                return None
            ext = PAYLOAD_KIND_EXT.get(kind)
            if ext is None:
                # A typo'd kind at a wired call site must be visible —
                # _log_error's fingerprint rate-limit dedups repeats.
                # That dedup RELIES on the fingerprint being
                # source:type:message[:100] and EXCLUDING context (which
                # carries the per-chain id) — folding context in would
                # defeat it and log one row per payload write.
                self._log_error('record_payload_unknown_kind',
                                ValueError('unknown payload kind %r'
                                           % (kind,)),
                                context='chain=%s' % chain_id)
                return None
            if not self._payload_kind_enabled(kind, chain_id):
                return None
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False,
                                     default=str)
            safe_chain = re.sub(r'[^A-Za-z0-9._-]', '_', str(chain_id))
            if not re.search(r'[A-Za-z0-9]', safe_chain):
                # Dot/punctuation-only names ('.', '..') are path syntax,
                # not names — they'd land files outside the date layout
                # where the retention pass never looks.
                safe_chain = 'chain-' + (safe_chain.replace('.', '_')
                                         or 'empty')
            chain_dir, day = self._payload_chain_dir(safe_chain)
            base = '%03d-%s' % (int(seq or 0), kind)
            for attempt in range(1, 100):
                name = ('%s.%s' % (base, ext) if attempt == 1
                        else '%s.%d.%s' % (base, attempt, ext))
                path = os.path.join(chain_dir, name)
                try:
                    fd = os.open(path,
                                 os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                except FileExistsError:
                    continue
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(content)
                pointer = os.path.join('payloads', day, safe_chain, name)
                return pointer
            raise RuntimeError('100 attempt ordinals exhausted for %s' % base)
        return pointer  # loud() swallowed a failure → None

    def round_recorder(self, chain_id, seq_base=0):
        """Build the per-round capture closure for run_llm_loop's
        `record_round_fn` — the runner has no brain by design, so each
        caller hands it this closure over (brain, chain_id). The layer owns
        the payload shape (build_round_payload) and the gate; with the
        `round_payload` kind off (the normal config) the per-round cost is
        one TTL-cached gate lookup. Never raises into the caller's loop.

        `seq_base` — file-seq offset for multi-batch encoders that run
        several run_llm_loop calls on ONE run chain (consolidation,
        community): without it every batch's round 0 collides on
        000-round_payload.json and batch identity dies in an attempt
        ordinal. Pass batch_num*100 (rounds are single digits); the
        payload's `round` field stays the in-batch index."""
        from .trace_contract import build_round_payload

        def _record(round_idx, parts):
            try:
                self.record_payload(
                    chain_id, 'round_payload',
                    build_round_payload(label=chain_id, round_idx=round_idx,
                                        seq=seq_base + round_idx, **parts),
                    seq=seq_base + round_idx)
            except Exception as e:
                self._log_error('round_recorder', e,
                                context='chain=%s round=%s'
                                        % (chain_id, round_idx))
        return _record

    def record_failed_run(self, chain_id, error):
        """Record the `failed_run` payload for a dead agent run — the full
        conversation at failure time. The LAYER owns the payload shape
        ({'error', 'messages'}) and its cap; consumers hand over the raw
        exception (RunLoopError carries `.msgs`, already bounded by
        tool_result_cap). Round-0 / unwrapped failures have no msgs and
        record nothing — the prompt kind is that half of the story.
        Returns the pointer or None."""
        from .trace_contract import FAILED_RUN_ERROR_CAP
        msgs = getattr(error, 'msgs', None)
        if not msgs:
            return None
        return self.record_payload(
            chain_id, 'failed_run',
            {'error': str(error)[:FAILED_RUN_ERROR_CAP], 'messages': msgs})

    def prune_payloads_if_due(self, now=None):
        """Age-prune {db_dir}/payloads/ date dirs older than retention_days
        (from the trace_recording config). Self-gated — deliberately runs
        AHEAD of the S2 fire conditions (a keyless brain still prunes): an
        in-memory hourly throttle keeps the per-poll cost at ~zero and the
        `s2_payload_prune_last_ts` brain_meta stamp enforces once-per-day.
        The stamp is written only after a FULLY successful prune — both
        exceptions and partially-failed removals retry within the hour,
        loudly, instead of silently waiting a day. Wall-clock deliberately
        (system bookkeeping, exempt from conversation-time). Never raises;
        returns date-dirs actually removed — failed removals loud-log and
        don't count."""
        import shutil
        import time as _time
        from datetime import datetime, timezone
        from .clock import iso_cutoff
        from .trace_contract import TRACE_RECORDING_NORMAL
        try:
            now = now if now is not None else _time.time()
            if now - getattr(self, '_payload_prune_checked', 0) < 3_600:
                return 0
            self._payload_prune_checked = now
            last = float(self.get_config('s2_payload_prune_last_ts') or 0)
            if now - last < 86_400:
                return 0
            days = self.get_interaction_config(
                'trace_recording').get('retention_days')
            if days is None:
                days = TRACE_RECORDING_NORMAL['retention_days']
            days = int(days)
            if days < 0:
                # A negative value would compute a FUTURE cutoff and delete
                # today's dirs out from under live runs — refuse loudly.
                self._log_error('payload_prune_bad_retention',
                                ValueError('retention_days=%d — using '
                                           'default' % days))
                days = TRACE_RECORDING_NORMAL['retention_days']
            # days=0 legitimately means "keep only today": cutoff is today,
            # and the strict `<` below never touches today's dir.
            cutoff = iso_cutoff(
                days=days,
                at=datetime.fromtimestamp(now, tz=timezone.utc))[:10]
            root = self._payload_root()
            removed, failed = 0, []
            if os.path.isdir(root):
                for name in os.listdir(root):
                    if not (re.fullmatch(r'\d{4}-\d{2}-\d{2}', name)
                            and name < cutoff):
                        continue
                    target = os.path.join(root, name)
                    shutil.rmtree(target, ignore_errors=True)
                    if os.path.exists(target):
                        failed.append(name)   # counter must not lie
                    else:
                        removed += 1
            if failed:
                # No stamp on partial failure: the undeletable dirs retry
                # within the hour (loudly, fingerprint-capped) instead of
                # silently waiting a day.
                self._log_error(
                    'payload_prune_failed_dirs',
                    RuntimeError('could not remove: %s' % ', '.join(failed)))
            else:
                self.set_config('s2_payload_prune_last_ts', str(now))
            return removed
        except Exception as e:
            self._log_error('payload_prune', e, context='')
            return 0

    def read_payload(self, pointer):
        """Read a payload by its relative pointer → str, or None (pruned /
        missing / never recorded). The pointer must stay inside db_dir —
        absolute paths and traversal are rejected (pointers come from trace
        metadata, which is data, not a path authority). Only a MISSING file
        is the silent 'pruned' answer; I/O failures (permissions, EIO) loud-
        log — 'pruned' must never mask an outage sitting on intact files.
        Bytes that don't decode (a crash mid-write splitting a multibyte
        char) come back with replacement chars rather than raising —
        degraded forensics beat none."""
        if not pointer or not isinstance(pointer, str):
            return None
        norm = os.path.normpath(pointer)
        if (os.path.isabs(norm) or norm.startswith('..')
                or not norm.startswith('payloads' + os.sep)):
            return None
        path = os.path.join(os.path.dirname(os.path.abspath(self.db_path)),
                            norm)
        try:
            with open(path, encoding='utf-8', errors='replace') as f:
                return f.read()
        except FileNotFoundError:
            return None
        except OSError as e:
            self._log_error('read_payload', e, context=norm)
            return None
