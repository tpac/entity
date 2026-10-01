"""S1 Scribe — the turn encoder as an in-process integration unit.

S1 Scribe is one of S1's integration units — the same O/K/Δ shape as the S2
units, run at faster cadence (every Nth conversational turn). O = the session's
turns + surface selections; K = node catalog + journal notes + the learnable
`s1e` interaction prompt; Δ = encoded nodes/edges. So it subclasses
`IntegrationUnit` and writes through the shared in-process encoder dispatch
(`_make_encoder_dispatch`), exactly like consolidation / community / healer —
which is what makes its revise/edge traces carry the run chain instead of a
date-fallback chain.

What stays S1-specific:
- **Chain format** — `s1e-{session}-{stop}` (session-scoped, turn-cadenced),
  NOT S2's time-based `s2-{ts}-{unit}`. `chain_id()` is overridden to the
  single source (SessionContext.s1e_chain).
- **The encode logic** — `encode.run_encoding(brain, dispatch, counter,
  session_id)` stays the standalone callable so the eval harnesses
  (`longmem/replay`) drive it directly with their own brain + dispatch. `S1Scribe` is the *production* wrapper: it binds the
  daemon's brain + the in-process dispatch + the run chain.

Execution: in-process on the daemon's brain — writes serialize under
`brain.write_lock`, vectors fill via the async embed_queue.
"""

from servers.scales.s2.base import IntegrationUnit


class S1Scribe(IntegrationUnit):
    """S1 turn encoder. Triggered every Nth conversational turn, or on idle for
    the tail (brain.scribe_due). Writes via the shared encoder dispatch so its
    revise / edge traces carry the s1e run chain."""

    NAME = 'scribe'
    SCALE = 's1'
    ENCODING_SOURCE = 'encoder:sonnet'

    O_SOURCES = ['s0_conversation', 'surface_selections']
    K_SOURCES = ['interaction:s1e', 'node_catalog', 'journal_notes', 'session_arc']

    def __init__(self, brain, session_id, counter, dispatch_fn=None):
        super().__init__(brain, dispatch_fn)
        self.session_id = session_id
        self.counter = counter

    def chain_id(self):
        """S1's run chain — ``s1e-{session_short}-{stop}`` — from the single
        source (SessionContext.s1e_chain), NOT S2's time-based format. Cached
        per run like the base, so every write in this run shares one chain.

        This format ends in ``-{stop}`` (a number), not ``-{NAME}`` like the base
        chain. That's fine: those base couplings — ``_last_run_timestamp``'s
        ``LIKE '%-{NAME}'`` and the dashboard slug parser — only fire for S2 units,
        which gate themselves on ``_last_run_timestamp`` / ``_has_new_traces``
        when the coordinator runs them. S1Scribe is triggered
        by ``brain.scribe_due`` (the poll), never the coordinator, so it doesn't
        flow through either consumer; its cadence is the trace-derived
        ``turns_since_last_encode``."""
        if not getattr(self, '_chain_id', None):
            from servers.session_context import SessionContext
            self._chain_id = SessionContext(
                self.session_id, stop_counter=self.counter).s1e_chain()
        return self._chain_id

    def scope_policy(self):
        """Session-scoped: the Scribe's writes carry the session's derived
        scope provenance (brain.scope_policy_for — the single builder,
        dimension-generic). '' per field when the session has none, which
        strips any LLM-supplied value instead."""
        return self.brain.scope_policy_for(self.session_id)

    def run(self):
        """Encode this window: bind the daemon's brain + the in-process encoder
        dispatch (which stamps the s1e run chain on writes) to the standalone
        run_encoding core."""
        from servers.scales.s1.encode import run_encoding
        dispatch_fn = self._make_encoder_dispatch()
        return run_encoding(self.brain, dispatch_fn, self.counter, self.session_id,
                            journal=self.journal)

    def _journal_kwargs(self):
        # Session-walled residue + the `## Arc` opt-in: the shape that makes
        # tell/ask lines file DIRECTED at this session (the unit default —
        # scoped by NAME, no session — would file them broadcast).
        from .encode import scribe_journal_kwargs
        return scribe_journal_kwargs(self.session_id)
