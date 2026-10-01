"""Presence — the self-channel awareness layer (PULL, not push).

Who else is awake right now (roster) and where they are (peek). Composition
only — no raw SQL here: the session_state reads live on the brain
(`brain.present_streams`), a stream's focus is `brain.session_context_for`, and
the line is `self_contract.render_presence`.

This is the PULL primitive: I look at my other streams of thought when interest
calls. The AUTOMATIC surfacing of the presence line at boot / hook fire is a
separate, eval-gated step (docs/BOOT-REIGNITION.md) — build it on top of
build_presence().
"""

from datetime import datetime, timezone

from servers.channels.self_channel import self_contract, signal


def _age_min(iso_ts):
    """Minutes since `iso_ts` — wall-clock (presence is real-time, exempt from the
    conversation_now rule, like present_streams). Empty/unparseable → a large age
    (treated as lost)."""
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    except Exception:
        return 1e9


def _first_line(focus, max_chars=100):
    """One-line roster focus — first line of the focus string present_streams
    hands us (the stream's latest user_message summary). Pure render, no read."""
    s = (focus or '').strip()
    if not s:
        return ''
    return s.splitlines()[0].strip()[:max_chars].rstrip()


def build_presence(brain, my_session_id='', limit=None, rich=False,
                   active_streams=True, sort_by='recency'):
    """Roster of streams of thought awake now — who they are, not just that
    they exist.

    Bounded by `limit` (default: PRESENCE_MAX_STREAMS) and the wall-clock window.
    Params:
      rich           — when True, each entry also carries peek-level detail
                       (arc, recent_msgs, session_started_at, pending_inbox_count)
                       so you can tell WHO a stream is in one call, no N peeks.
                       `focus` (last conversational msg) and `arc` (encoder's
                       session arc) are DISTINCT fields — same source confusion
                       that bit discovery, kept separate here.
      active_streams — True (default): only active (reachable, not-lost) streams;
                       False also surfaces recently-lost ones in a grace window.
      sort_by        — 'recency' (default, most recent conversational turn) or
                       'length' (number of turns). Boot heartbeats no longer
                       dominate: a 0-turn fresh stream sorts last under recency.

    Returns {'streams': [...], 'lost': [...], 'line': <rendered presence line>}.
    `line` is the one-line ambient form; the structured list is what `peek`
    drills into (and what `rich` pre-expands).
    """
    cap = limit or self_contract.PRESENCE_MAX_STREAMS
    # Fetch out to the lost-grace window so a recently-gone stream is visible, not
    # silently dropped at the live edge; headroom on the limit so lost entries
    # don't crowd out the live cap.
    window = self_contract.ROSTER_LIVE_WINDOW_MIN + self_contract.ROSTER_LOST_GRACE_MIN
    raw = brain.present_streams(
        exclude_session=my_session_id or '',
        window_min=window,
        limit=cap + 5,
        sort_by=sort_by)
    live, lost = [], []
    for r in raw:
        sid = r.get('session_id', '')
        state = self_contract.classify_liveness(_age_min(r.get('live_recency', '')))
        entry = {
            'session_id': sid,
            'short': sid[:8],
            'focus': _first_line(r.get('focus', '')),
            'updated_at': r.get('live_recency', ''),
            'turn_count': r.get('turn_count', 0),
            'state': state,
        }
        # 'active'/'dormant'/'lost' is classify_liveness's existing definition —
        # active_streams=True keeps the live set (active+dormant), excluding lost;
        # active_streams=False also surfaces lost in the grace window.
        if state == 'lost':
            if not active_streams and len(lost) < cap:
                lost.append(entry)
            continue
        if len(live) < cap:
            if rich:
                pk = peek(brain, sid)
                entry['arc'] = pk.get('focus', '')
                entry['recent_msgs'] = pk.get('recent_msgs', [])
                entry['cwd'] = pk.get('cwd', '')
                entry['branch'] = pk.get('branch', '')
                entry['worktree'] = pk.get('worktree', '')
                entry['model'] = pk.get('model', '')
                entry['host'] = pk.get('host', '')
                entry['session_started_at'] = pk.get('session_started_at', '')
                entry['pending_inbox_count'] = pk.get('pending_inbox_count', 0)
            live.append(entry)
    line = self_contract.render_presence(
        [(s['short'], s['focus'], s['state']) for s in live],
        lost=[(s['short'], s['focus']) for s in lost])
    return {'streams': live, 'lost': lost, 'line': line}


def _empty_peek(session_id=''):
    # MUST carry every key the full peek() return does (degrades, never
    # half-shaped) — including turn_count/cwd/branch/worktree — so bracket-access
    # consumers never KeyError. test_peek_empty_path_has_all_keys locks this.
    return {'session_id': session_id, 'short': session_id[:8] if session_id else '',
            'focus': '', 'recent_msgs': [], 'turn_count': 0,
            'cwd': '', 'branch': '', 'worktree': '', 'project': '',
            'model': '', 'host': '',
            'session_started_at': '',
            'last_active_at': '', 'liveness': 'lost', 'pending_inbox_count': 0,
            'found': False}


def peek(brain, session_id, msg_limit=2):
    """Look into one stream of thought — where it is right now, without
    interrupting it. The interest-driven pull; read-only.

    Returns its arc (focus), the last `msg_limit` conversational messages
    (each capped at PEEK_MSG_MAX), when it started, when it was last active +
    its liveness state, and how many messages wait in its OWN inbox. `found` is
    true once it has an arc OR any real turn — so a fresh stream with a single
    message still peeks usefully (the arc lags; turns don't). On the empty/
    error path every key is still present (degrades, never half-shaped)."""
    if not session_id:
        return _empty_peek()
    arc = (brain.session_context_for(session_id) or '').strip()
    act = brain.session_activity(session_id, msg_limit=msg_limit) or {}
    env = brain.session_env_for(session_id)
    recent = [{'ts': m.get('ts', ''), 'role': m.get('ref_type', ''),
               'text': (m.get('text', '') or '')[:self_contract.PEEK_MSG_MAX]}
              for m in act.get('recent_msgs', [])]
    last_active = act.get('last_active_at', '') or ''
    liveness = (self_contract.classify_liveness(_age_min(last_active))
                if last_active else 'lost')
    return {
        'session_id': session_id, 'short': session_id[:8],
        'focus': arc,
        'recent_msgs': recent,
        'turn_count': act.get('turn_count', 0),
        'cwd': env.get('cwd', ''),
        'branch': env.get('branch', ''),
        'worktree': env.get('worktree', ''),
        'project': env.get('project', ''),
        # What the stream rides on right now — the session row mirrors the
        # latest per-turn value (trace_contract.S0_SESSION_STAMP_FIELDS).
        'model': env.get('model', ''),
        'host': env.get('host', ''),
        'session_started_at': act.get('started_at', '') or '',
        'last_active_at': last_active,
        'liveness': liveness,
        'pending_inbox_count': signal.pending_count(brain, session_id),
        'found': bool(arc or recent),
    }
