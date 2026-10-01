"""The journal component — one object that attaches the encoder journal
(the residue contract) to ANY agent request, loop or single-shot.

The journal's TEXT and FORMAT live in trace_contract (single source: the
review block, the closure, the notes renderer, the `tag · subject · note`
parser, the fence scanner). This component owns the ORCHESTRATION: which
blocks decorate a request, in what order, and how residue is harvested from
the response — so an encoder binds the whole mechanism in one place instead
of hand-assembling injects.

There are no per-shape dialects — one wire format, one parser, one
instruction block. Two request-derived conditionals only:
  • the closure block is appended only for multi-round (loop) requests — it
    exists to disambiguate the terminal turn, which a single-shot call
    doesn't have;
  • harvest() returns the final text with the journal sections stripped, so
    a single-shot response's JSON payload survives extract_json (whose
    rfind-based scan would be corrupted by a `]`/`}` inside a fence that
    follows the payload). Loop callers simply ignore the return value.

Placement rules the component enforces (earned decisions, see brain nodes
2a81ea82 / 0e1a7303 / 7905e385): instruction blocks go at the SYSTEM tail —
arc (when bound) before review, closure last; the continuity notes are
CONTEXT, rendered by continuity() for the caller to place in user content
(S2 units prepend it; S1E embeds it in its structured layout — placement is
the binding's business where shipped layouts differ, the text is not).

Provider separation: this module deals only in strings and note dicts —
never SDK types. The provider seam is scales/runner.py.
"""


class JournalBinding:
    """One encoder's attachment to the journal.

    Identity mirrors journal_notes() scoping: S2 units bind (scale, unit);
    the S1 Scribe binds (scale, session_id) — its residue is session-walled.
    `arc=True` opts into the second journal object (the `## Arc` fence →
    session arc accumulator); only the Scribe carries it today.

    `source` is the producer's encoding_source. A binding WITH a source is
    ADDRESSED: its review lines tagged tell/ask are messages to the people
    working, filed through the Thalamus door instead of written as residue.
    The audience falls out of the binding: a session (the Scribe) makes the
    item directed — a tell or an ask reaches that session at its next Stop;
    no session (an S2 unit) makes it the door's default audience — a tell
    reaches the first session to pull at boot or Stop, an ask reaches every
    session at boot until answered. One instruction, one door, two
    audiences. Without a source (a bare eval binding) they stay plain notes.
    """

    def __init__(self, brain, *, scale, unit='', session_id='', arc=False,
                 source=''):
        self.brain = brain
        self.scale = scale
        self.unit = unit
        self.session_id = session_id
        self.arc = arc
        self.source = source
        self.addressed = bool(source)
        self._run_chain = None
        self._view = None
        self._selection_failed = False
        self.stats = {}

    # Error-log keys preserve the pre-component vocabulary so log continuity
    # survives the refactor (s1e_* for the Scribe, s2_{unit}_* for S2 units).
    def _log_key(self, op):
        if self.scale == 's1':
            return 's1e_journal_notes_%s' % op
        return 's2_%s_journal_%s' % (self.unit, op)

    def continuity(self, *, chain_id=None):
        """Prepare current continuity before each independent request.

        A chain identifies an invocation, not a batch. Freeze its private
        selection, refresh committed lifecycle state and message outcomes.
        A new chain resets the selection; omission of chain_id is the
        single-request convenience used by standalone readers.
        """
        from servers.trace_contract import render_journal_view
        if chain_id is None or chain_id != self._run_chain:
            self._run_chain, self._view = chain_id, None
            self._selection_failed = False
        try:
            if not self._selection_failed:
                self._view = self.brain.journal_view(
                    scale=self.scale, unit=self.unit, session_id=self.session_id,
                    previous=self._view)
            view = self._view or {'selection_failed': True}
        except Exception as e:
            # No initial snapshot means there is no safe private selection
            # to recover mid-run: a later load would echo this run's notes.
            # Retry on the next invocation; live messages remain independent.
            self._selection_failed = self._view is None
            self.brain._log_error(
                self._log_key('read'), e,
                'journal continuity read failed — retaining last known state')
            view = dict(self._view or {}, stale=True,
                        selection_failed=self._selection_failed)
        try:
            text, stats = render_journal_view(view)
        except Exception as e:
            self.brain._log_error(self._log_key('render'), e,
                                  'journal rendering failed — encoding without residue')
            text, stats = render_journal_view({'render_failed': True})
        messages = self.messages()
        self.stats = dict(stats, message_chars=len(messages),
                          continuity_chars=len(text) + len(messages))
        try:
            self.brain.log_debug('journal_continuity', 'JournalBinding',
                                 chain_id=chain_id or '',
                                 cursor=view.get('cursor', 0), **self.stats)
        except Exception as e:
            self.brain._log_error(self._log_key('telemetry'), e,
                                  'journal continuity telemetry failed')
        return text + messages

    def messages(self):
        """Live outcomes, read for EACH request by continuity().

        Earlier batches can file or resolve an item. Reusing their input
        snapshot lets a later batch overwrite the newer message or miss an
        answer. Thalamus remains the owner of delivery state.
        """
        if not self.addressed:
            return ''
        try:
            return self._producer_view()
        except Exception as e:
            self.brain._log_error(
                self._log_key('view'), e,
                'producer view read failed — encoding without it')
            return ''

    def _producer_view(self):
        """The join for the producer view: this binding's own items (the
        door orders and windows them — producer_items) as rows the contract
        phrases. The tag/subject mapping is the exact inverse of the forward
        map in _route_addressed (tag → needs_answer, subject → dedup_key). A
        refused filing already sits in the residue notes with its reason."""
        from servers.channels.thalamus import thalamus
        from servers.channels.thalamus import thalamus_contract as tc
        from servers.trace_contract import (render_producer_view,
                                            JOURNAL_TELL_TAG, JOURNAL_ASK_TAG,
                                            JOURNAL_RUN_SUBJECT)
        rows, seen = [], set()
        for i in thalamus.producer_items(
                self.brain, self.source, self.session_id,
                settled_days=tc.PRODUCER_VIEW_SETTLED_DAYS):
            # One row per subject: the door orders open-first, so a key
            # re-asked after a settlement shows its live state, not both.
            if i['dedup_key'] in seen:
                continue
            seen.add(i['dedup_key'])
            rows.append({
                'tag': (JOURNAL_ASK_TAG if tc.kind_of(i) == tc.KIND_ASK
                        else JOURNAL_TELL_TAG),
                'subject': i['dedup_key'] or JOURNAL_RUN_SUBJECT,
                'note': i['body'], 'fate': tc.fate_of(i), 'answer': i['answer'],
            })
        return render_producer_view(rows)

    def decorate_system(self, system_prompt, multi_round=True):
        """The WRITE-side instructions, appended at the system tail in the
        contract order: arc (when bound) → review block → closure. The
        closure (terminal-turn definition + `## Review` placement + DONE)
        applies only to multi-round requests and must be genuinely last —
        call this AFTER all other prompt assembly (e.g. edge-aspect vocab).
        """
        from servers.trace_contract import (render_journal_arc_block,
                                            render_journal_review_block,
                                            render_prompt_closure)
        out = system_prompt
        if self.arc:
            out = out.rstrip() + '\n\n' + render_journal_arc_block()
        out = out.rstrip() + '\n\n' + render_journal_review_block()
        if multi_round:
            out = out.rstrip() + '\n\n' + render_prompt_closure()
        return out

    def harvest(self, final_text, chain_id, arc_limit=800):
        """The response side: write this run's residue notes (+ the session
        arc when bound) and return `final_text` with the journal sections
        stripped — the payload remainder a single-shot caller parses JSON
        from. Note-write failures are isolated (logged loud, run intact);
        write_session_arc is failure-isolated internally.
        """
        from servers.trace_contract import strip_journal_sections
        res = {}
        try:
            res = self.brain.write_journal_notes(
                final_text=final_text, chain_id=chain_id,
                scale=self.scale, session_id=self.session_id)
        except Exception as e:
            self.brain._log_error(
                self._log_key('write'), e,
                'residue note write failed — run otherwise intact')
        # The routing half is isolated like the write half: a Thalamus read
        # or write hiccup must never abort the encoder's run (S2's batch loop
        # relies on it) nor cost the session arc below.
        try:
            if self.addressed:
                self._route_addressed(res, chain_id)
            elif res.get('addressed'):
                # No source to file under — the lines stay residue, but a
                # message nobody will deliver is worth a warning.
                self.brain._log_warning(
                    self._log_key('addressed_unbound'),
                    'chain=%s: %d tell/ask line(s) from a binding without a '
                    'source — written as plain notes, undelivered'
                    % (chain_id, len(res['addressed'])))
                self._write_rows(res['addressed'], chain_id)
        except Exception as e:
            self.brain._log_error(
                self._log_key('route'), e,
                'addressed-line routing failed — run otherwise intact')
        if self.arc:
            self.brain.write_session_arc(
                final_text=final_text, session_id=self.session_id,
                limit=arc_limit)
        return strip_journal_sections(final_text)

    def _route_addressed(self, res, chain_id):
        """File this run's tell/ask lines as Thalamus items — the non-LLM
        entrance to the one door: `tell` is a notice, `ask` needs an answer;
        the audience is the binding's (see the class docstring). The
        normalized subject is the item's dedup_key — "one line per subject"
        holds literally: a re-assertion updates, never duplicates — and the
        line's anchor (any node id the subject embeds — journal_anchor_refs)
        is its refs. The run chain rides along so the
        filing is traced on it. A door rejection is LOUD and the line is
        kept as a plain note carrying the reason, so the residue survives
        and the encoder reads the rejection next run. A `resolved · <subject>`
        line also withdraws this binding's open item under that key — the
        encoder's existing verb closes its own item."""
        addressed = res.get('addressed') or []
        resolved = res.get('resolved') or []
        if not (addressed or resolved):
            return
        from servers.channels.thalamus import thalamus
        from servers.trace_contract import (JOURNAL_ASK_TAG, journal_key,
                                            journal_anchor_refs,
                                            resolve_target)
        kept = []
        for n in addressed:
            try:
                r = thalamus.file(
                    self.brain, self.source, n['note'],
                    needs_answer=journal_key(n.get('tag')) == JOURNAL_ASK_TAG,
                    for_whom=self.session_id or None,
                    dedup_key=journal_key(n['subject']),
                    refs=journal_anchor_refs(n['subject']) or None,
                    session_id=self.session_id, run_chain=chain_id)
            except Exception as e:
                r = {'ok': False, 'error': str(e)}
            if not r.get('ok'):
                self.brain._log_warning(
                    self._log_key('addressed_rejected'),
                    'chain=%s: %s · %s not filed — %s' % (
                        chain_id, n.get('tag'), n['subject'], r.get('error')))
                kept.append(dict(n, undelivered=r.get('error') or 'rejected'))
        if kept:
            self._write_rows(kept, chain_id)
        if resolved:
            # Close only what this binding actually has open (one read) —
            # with the read side's tolerance for an echoed
            # `tag · subject · …` head (resolve_target), so the same resolve
            # line that retires the note also closes the item.
            open_keys = {i['dedup_key'] for i in thalamus.producer_items(
                             self.brain, self.source, self.session_id)
                         if i['dedup_key']}
            for r in resolved:
                target = resolve_target(journal_key(r['subject']), r['note'],
                                        open_keys)
                if target in open_keys:
                    thalamus.withdraw(self.brain, self.source,
                                      dedup_key=target,
                                      target_session=self.session_id)

    def _write_rows(self, notes, chain_id):
        self.brain.write_journal_note_rows(
            notes, chain_id=chain_id, scale=self.scale,
            session_id=self.session_id)
