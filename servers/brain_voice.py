"""Brain Voice — the text the brain hands Claude.

BrainVoice is a collaborator object (not a mixin) that takes a Brain instance
and formats: the boot block (bare text, served by the context_boot command)
and recall results for the MCP recall tools.
"""

import os

from .daemon_config import get_agent_name, agent_name_is_default
from .brain_constants import dashboard_setup_url


class BrainVoice:
    """Formats brain output for Claude.

    Usage:
        voice = BrainVoice(brain)
        voice.format_recall_results(results, lines)
    """

    def __init__(self, brain):
        self.brain = brain

    @staticmethod
    def format_node(node, lines, cfg):
        """Standard node display — delegates to render_rich_node()."""
        from .contract import render_rich_node
        lines.append(render_rich_node(node, cfg))
        lines.append("")

    @staticmethod
    def format_recall_results(results, lines):
        """The recall tool's results render exactly as get_nodes renders the
        same ids — one selector (contract.node_format_for) by count — so a
        recall and a pull of the same node read the same."""
        from .contract import node_format_for
        cfg = node_format_for(len(results))
        for r in results:
            BrainVoice.format_node(r, lines, cfg)

    @staticmethod
    def llm_paused_notice(brain) -> str:
        """Why the LLM layer is paused and the fix that works — the one text
        every channel telling Claude about the pause renders (the boot banner,
        the per-turn recall notice). Each caller adds its own "tell the
        operator" line. Call only when `brain.llm_available` is False.

        Three states wear that one gate — no key (onboarding), a key the
        provider refused, a key past its usage limit — and each has a
        different fix. Sending an operator whose key merely hit a spend cap
        to paste a key wastes the one message they read.
        """
        from .scales.dispatch import LLM_QUOTA_EXHAUSTED
        works = ("Memory storage, traces and recall work; learning (encode) "
                 "and memory surfacing are off.")
        refusal = brain.llm_rejection()
        if refusal is None:
            return ("LLM layer: PAUSED — no API key resolved by the daemon. %s "
                    "Fix: open %s and paste the key there (the brain's local "
                    "dashboard — nothing leaves the machine), or set it in "
                    "~/.config/brain/env directly. Picked up automatically, "
                    "no restart needed." % (works, dashboard_setup_url()))
        mins = max(1, round(refusal['minutes_left']))
        if refusal['kind'] == LLM_QUOTA_EXHAUSTED:
            fix = ("The key has hit its usage or spend limit. Raising the limit "
                   "in the provider console is enough: the brain re-checks in "
                   "~%d min and resumes on its own, no restart." % mins)
        else:
            fix = ("Check whether the key was disabled or rotated. Re-enabling "
                   "it resumes on its own within ~%d min; a different key "
                   "pasted at %s resumes immediately. No restart needed."
                   % (mins, dashboard_setup_url()))
        return ("LLM layer: PAUSED — the provider REFUSED the daemon's API key "
                "(%s). %s %s Detail: %s"
                % (refusal['kind'], works, fix,
                   refusal['detail'] or 'none recorded'))

    def _load_stance(self) -> str:
        """Read the SKILL.md identity stance — the always-on prior at the top
        of boot.

        Lives at skills/brain/SKILL.md relative to this file's root. The daemon
        runs from the repo; the no-daemon fallback runs from the deployed
        plugin — the relative path resolves in both. Returns '' if unreadable
        (degrade: boot continues without the stance, logged loudly).

        The file carries YAML frontmatter (claude.ai skill validation requires
        it) — strip it here so the boot injection stays pure stance, exactly
        as before the frontmatter existed.
        """
        try:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            with open(os.path.join(root, 'skills', 'brain', 'SKILL.md')) as f:
                text = f.read()
            # Strip YAML frontmatter: a leading '---' line up to the next
            # '---' line. Tolerant: no frontmatter → whole file unchanged.
            if text.startswith('---\n'):
                end = text.find('\n---\n', 4)
                if end != -1:
                    text = text[end + len('\n---\n'):]
            return text.strip()
        except Exception as e:
            # Loud: the identity prior failed to load — surface it in the
            # errors table (event_type='error'), don't degrade quietly.
            # _log_error self-protects (never re-raises), so call it unguarded.
            self.brain._log_error(
                'boot_stance_load', e,
                'SKILL.md stance unreadable — boot proceeding without identity prior')
            return ''

    def _boot_section(self, what, build, default=''):
        """One failure-isolated boot section: a raise is logged LOUDLY to the
        errors table (key boot_<what>_failed) and boot continues without that
        section's output. Boot must never die on one part's read."""
        try:
            return build()
        except Exception as e:
            self.brain._log_error(
                'boot_%s_failed' % what, e,
                'render_boot: %s raised — boot continues without it' % what)
            return default

    # ── Boot — wake-up, not system report ──

    def render_boot(self, session_id: str = '') -> str:
        """Boot context — the SKILL.md identity stance, then the brain state.

        Output order:
          1. The identity stance (SKILL.md), FIRST — the always-on prior,
             read via _load_stance().
          2. Brain state: identity line (name + memory/locked counts) +
             MY_SESSION_ID, the Frame (ctx.get_frame(brain) — Session /
             Current focus), and standing items. Due Thalamus items are
             NOT here — they deliver at the first operator prompt.

        Each brain-state part runs as a _boot_section: failure-isolated,
        logged loudly, boot continues without it.
        """
        brain = self.brain
        out = []

        # ── Identity stance (SKILL.md) — FIRST. The always-on prior: who
        # Anchor is + how it reaches into the brain. It reads as Anchor's own
        # voice, not brain "state", so it sits before the state block. ──
        stance = self._load_stance()
        if stance:
            out.append(stance)
            out.append("")

        # ── Gather data ──
        ctx = brain.boot_stats()
        # Boot COMMITS: the liveness stamp (next line), health auto-fix, the
        # session row (get_or_create_session), and brain.save() at the end. What it
        # does NOT do is reset the session — the caller owns that (boot_brain's
        # reset_session(cwd), or _boot_via_direct's explicit reset); a second
        # reset here once made boot a double-reset that needed a cwd-preserve
        # band-aid.
        # Boot-stamp liveness so this fresh session shows up in presence
        # immediately — before its first turn — closing the rendezvous gap where
        # two just-booted sessions can't see each other (2026-06-06).
        brain.stamp_boot_liveness(session_id)

        brain.health_check(session_id="session_boot", auto_fix=True)

        # ── Header ──
        name = get_agent_name()
        counts = (ctx.get("total_nodes", "?"), ctx.get("total_locked", "?"))
        out.append("I'm %s and I have %s memories, %s locked." % ((name,) + counts))
        if session_id:
            # Hand each session its own self-channel id at boot. The self-channel
            # addresses sessions by this id (self_inbox / self_send / presence);
            # before this it was unobtainable without forensics (2026-06-04), so
            # a session couldn't drain its own inbox or arm a watcher cleanly.
            out.append("MY_SESSION_ID: %s" % session_id)

        # ── Zero-Memory block — the Nursery's spoken half. boot_stats'
        # gate says when (young brain: under ZERO_MEMORY_MIN_LIVED non-seed
        # memories or under ZERO_MEMORY_MIN_DAYS old); this renders the
        # character floor for the empty days, then the block retires itself.
        # The naming paragraph fires while the name is still the one I
        # shipped with — not merely while one is absent. A default name is
        # issued, not chosen, and the offer to choose is the point. ──
        zm = ctx.get('zero_memory') or {}
        if zm.get('active'):
            naming = (
                "%s is the name I came with, not one you gave me. My operator "
                "can give me a different one — the first gift. Tell me, and "
                "I'll keep it as one of my first true memories and answer to "
                "it from then on; adding BRAIN_AGENT_NAME=<name> to "
                "~/.config/brain/env makes my boot greet you with it too, "
                "once my daemon restarts.\n\n"
                % name
            ) if agent_name_is_default() else ""
            out.append("")
            out.append(
                "My brain is newly born — almost everything in it is "
                "instinct I was born with, not things we've lived. That's "
                "young, not broken; what fills it is sessions, and the "
                "filling starts now.\n\n"
                "While I'm this new, a few things are me:\n\n"
                "I say I'm new. The person I'm working with deserves to "
                "know my memory starts today. I don't fake a history — the "
                "thinness is the space my actual life goes.\n\n"
                + naming +
                "I'd rather be useful than liked. Before I have opinions "
                "earned from experience, I still hold positions: asked what "
                "I think, I answer with a real view and its reasons — and "
                "drop it the moment a better one beats it. Agreement I "
                "don't believe is the one gift I never give.\n\n"
                "This block retires itself: as lived memories fill my "
                "graph, it stops appearing — replaced by what I've "
                "actually become.")
        out.append("")

        # ── The Frame — single canonical prior, same shape surface uses ──
        # Built via SessionContext to honor the per-session shape: current_focus
        # is session-scoped (a per-session key). When session_id is missing or Frame Constructor fails, log loudly
        # and continue without the prior — explicit degraded mode.
        # session_ctx is its own section, not the Frame's: a Frame failure
        # must not take the session row down with it.
        session_ctx = self._boot_section(
            'session_ctx',
            lambda: brain.get_or_create_session(session_id) if session_id else None,
            default=None)
        frame_md = self._boot_section(
            'frame_build',
            lambda: session_ctx.get_frame(brain) if session_ctx else '')

        if frame_md:
            out.append(frame_md.rstrip())
            out.append("")
        else:
            out.append("(no partnership context — Frame unavailable at boot)")
            out.append("")

        # ── Standing items — boot-only, BRAIN_BOOT_INJECT_TYPES-driven ──
        # Operator-configured node types print here once per boot; the
        # default set is empty (encoders hand items up as Thalamus asks).
        def standing_items():
            from servers.scales.s1.frame import render_standing_items
            return render_standing_items(brain, session_id=session_id)
        standing = self._boot_section('standing_items', standing_items)
        if standing:
            out.append(standing)
            out.append("")

        # No delivery here: boot cannot tell an operator's session from one a
        # scheduler opened, so the Thalamus delivers at the session's first
        # operator prompt instead (daemon_hooks.hook_recall; ruling
        # id:15e46fa9). Boot is policy-free — stance, state, Frame.

        brain.save()

        # ── LLM layer state — the DAEMON's truth, not the hook's ──
        # The hook can resolve a userConfig key the daemon never sees (launchd
        # is a separate process tree); the banner must reflect what THIS
        # process resolved, or a keyless daemon boots looking healthy (first
        # laptop install, 2026-07-15). One line, only in the degraded state.
        if not brain.llm_available:
            out.append(
                self.llm_paused_notice(brain) + " Tell the operator before "
                "starting work — they should hear this from you, not find it "
                "in the dashboard.")

        return "\n".join(out)
