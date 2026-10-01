"""Frame Constructor — Anchor's structured awareness object.

Produces a markdown Frame, deterministically composed from per-session state
(no LLM call, no queried nodes, no new SQL). Sections:

    Session       — deterministic situational header: project, counterpart,
                    clock, worktree. Each line exists to prevent a named
                    failure (see _render_session_header).
    Current focus — the encoder's rolling per-session arc.

Same Frame renders at boot (SessionStart hook) and during conversation
(hook_recall → run_surface).

Queried-node sections are deliberately absent: recency-by-type pulls fill
with engineering nodes and pollute Haiku's per-turn prior, and any selection
here must be measured first ("organized priming" is the recorded
return-intent, behind the identity-prior redesign). Seed-brain
operator/identity scaffolding lives in the conditional Zero-Memory boot
block; everything else is recoverable via recall() on demand.
"""

import datetime as _dt

from servers.clock import conversation_now


def _render_session_header(brain, session_id: str, at=None) -> str:
    """Session — the deterministic situational anchor.

    Every line names the failure it prevents:
      Project     — cross-project contamination: without a current-project
                    anchor Haiku can't discount foreign-project candidates
                    (the People Inc pick, s1r-42ff289f-22). '(unscoped)' is
                    itself signal: no project pressure applies.
      Counterpart — speaker/attribution bleed once multiple counterparts
                    exist (brain.counterpart_for — install default today).
      Now         — temporal misjudgment: candidates render relative times,
                    so consumers half-know 'now'; the explicit clock makes
                    'yesterday'/'the Jul 22 deadline' resolvable. Production
                    is wall-clock; REPLAYS MUST PASS `at` (get_frame/
                    build_frame thread it) — conversation_now(brain=) alone
                    cannot see a replay's injected date.
      Worktree    — parallel-stream confusion: which checkout this stream
                    is acting on (only rendered when in one).

    Declaration and marks cannot drift: the Project/Counterpart lines render
    FROM brain.session_scope — the same dict the candidate mismatch marks
    (contract.scope_marks) compare against.
    """
    scope = brain.session_scope(session_id) or {}
    now = at or conversation_now(brain=brain)
    if now.tzinfo is None:
        # A naive `at` is taken as UTC — astimezone() on a naive datetime
        # would reinterpret it as machine-local (wrong day on non-UTC boxes).
        now = now.replace(tzinfo=_dt.timezone.utc)
    lines = ['## Session']
    lines.append('- Project: %s' % (scope.get('project') or '(unscoped)'))
    if scope.get('counterpart'):
        lines.append('- Counterpart: %s' % scope['counterpart'])
    lines.append('- Now: %s (%s)' % (
        now.astimezone(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        now.astimezone(_dt.timezone.utc).strftime('%A')))
    worktree = (brain.session_env_for(session_id) or {}).get('worktree')
    if worktree:
        lines.append('- Worktree: %s' % worktree)
    return '\n'.join(lines) + '\n'


def _render_current_focus(brain, session_id: str) -> str:
    """Encoder's rolling session arc — per-session key (no parallel-session leak).

    Reads `brain.session_context_for(session_id)`, so parallel sessions never
    see each other's arc. Cross-session continuity is a deliberate query at
    boot, not a side-effect of shared state.
    """
    ctx = (brain.session_context_for(session_id) or '').strip()
    if not ctx:
        return "## Current focus\n(fresh session)\n"
    return "## Current focus\n%s\n" % ctx


def build_frame(brain, session_id: str, at=None) -> str:
    """Construct the Frame as markdown text — deterministic, no LLM call,
    no queried nodes (see the module docstring for why).

    Args:
        brain: Brain instance
        session_id: current session ID (for header + focus)
        at: conversation-time datetime for the header clock; defaults to
            conversation_now(brain=brain). Replays pass their injected time.
    Returns:
        Markdown string.
    """
    sections = [
        _render_session_header(brain, session_id, at=at),
        _render_current_focus(brain, session_id),
    ]
    return "\n".join(s for s in sections if s).strip() + "\n"


# ── Standing items — boot-only, operator-extensible type injection ──
# NOT a Frame section: the Frame renders on every recall turn; standing items
# are a wake-up ritual (see them once, act, archive). Injected by
# render_boot after the Frame.
#
# BRAIN_BOOT_INJECT_TYPES (~/.config/brain/env, comma-separated) picks which
# node types surface at boot. Ships EMPTY — an encoder's long-lived item is
# handed up as a Thalamus `ask`, not printed as a node type at every boot.
# The mechanism stays for operators who want a type of their own here.

BOOT_INJECT_TYPES_DEFAULT = ''
BOOT_INJECT_CAP = 10


def render_standing_items(brain, session_id: str = '') -> str:
    """Every live node of the configured types, newest first, capped.

    Empty string when nothing qualifies — a clean boot adds nothing. The
    lifecycle exit is on the node: handle it, then archive (or revise) it and
    it leaves the next boot. session_id threads the booting session's scope
    veil through filter_nodes — a walled item must not print into every
    other session's boot.
    """
    import os
    raw = os.environ.get('BRAIN_BOOT_INJECT_TYPES', BOOT_INJECT_TYPES_DEFAULT)
    types = [t.strip() for t in raw.split(',') if t.strip()]
    if not types:
        return ''
    result = brain.filter_nodes(field='type', include=types,
                                limit=BOOT_INJECT_CAP + 1, rich=False,
                                session_id=session_id)
    if (result or {}).get('error'):
        # Loud by default — an error dict is not "nothing standing".
        brain._log_error('boot_standing_items_filter',
                         ValueError(str(result.get('error'))),
                         'render_standing_items: filter_nodes errored for types=%r' % types)
        return ''
    nodes = (result or {}).get('nodes') or []
    if not nodes:
        return ''
    overflow = len(nodes) > BOOT_INJECT_CAP
    nodes = nodes[:BOOT_INJECT_CAP]
    lines = ['## Standing items']
    for n in nodes:
        lines.append('- [%s] %s (id:%s, since %s)' % (
            n.get('type', '?'), n.get('title', '?'),
            (n.get('id') or '')[:8], (n.get('created_at') or '')[:10]))
    if overflow:
        lines.append("- (+more — filter_nodes(field='type', include=%r))" % types)
    return '\n'.join(lines)
