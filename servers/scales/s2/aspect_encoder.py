"""S2 Aspect Encoder — classifies candidates into the required aspects via the s2_aspects model.

Reads the taxonomy via brain.aspects (the menu + current member lists),
builds a prompt with the menu + candidate strings + example records, calls
the model once, parses the JSON response, validates each classification, and
writes through the registry's door (brain.aspects.add_members). Auto-merge —
no operator review gate in v1.

The orchestrator (aspect_integration.py) writes the per-cycle audit record
(aspects_proposed.json) from what this returns.

The unit does NOT touch the graph — taxonomy in / taxonomy out, written
through brain.aspects.add_members.
"""

from servers.aspect_store import (
    CLASSIFIER_FIELDS,
    MEMBER_FIELDS,
    field_values,
    filled_fields,
    new_member_fill,
)
from servers.aspects import rank_members
from servers.contract import TITLE_REF_LIMIT, node_ref, truncate_marked
from servers.trace_contract import build_delta_metadata

from .base import IntegrationUnit
from .aspect_contract import ASPECT


# Which aspects the classifier may route to, and which categories each takes,
# are per-aspect FACTS in aspects_v1.json (`routable`, `accepts`) — read via
# the registry, never hardcoded here. `accepts` is design intent, not
# discoverable from current member lists alone (an aspect with empty
# node_types might be edge-only by design or just not yet populated).
# survivor_lineage declares routable=false (system-generated absorbed_into
# edges, never classified). `wisdom` IS routable so it GROWS: it's a
# multi-membership view (its node types also live in lesson_insight /
# identity_bearing), and the encoder multi-homes generative types into it,
# guided by the aspect's `meaning`. The decoder proposes only unclassified
# strings, so existing types stay as seeded while NEW generative types
# auto-join wisdom alongside their primary aspect.


def _routing_map(aspects):
    """{aspect_name: accepts-set} for the routable aspects — the classifier's
    closed list, derived from the per-aspect facts. `aspects` is the
    name→Aspect dict from brain.aspects.all()."""
    return {name: set(a.accepts) for name, a in aspects.items() if a.routable}


def fill_for(proposal):
    """The fields a candidate still needs, as a list of MEMBER_FIELDS names.
    The decoder sets `fill` per candidate; a proposal without it is a new
    string that needs everything its category takes — `aspects` plus every
    enum field stored for the category."""
    fill = proposal.get('fill')
    if fill:
        return list(fill)
    return new_member_fill(proposal['category'])


def render_field_definitions():
    """The FIELD DEFINITIONS block of the user message — rendered from
    CLASSIFIER_FIELDS so the encoder is taught exactly the fields a fill line
    can name (a curated field is never taught), in the words the candidate
    blocks and the return rules use.
    Opens by tying `source` / `target` to the way examples are drawn, so a
    value name like `source_actor` points at something on the page."""
    lines = []
    lines.append('═' * 70)
    lines.append('FIELD DEFINITIONS — each candidate lists the fields you must fill')
    lines.append('═' * 70)
    lines.append('')
    lines.append('A node is named  [type] "title" (id:xxxxxxxx)  everywhere below. An edge example is drawn as')
    lines.append('  [type] "source title" (id:xxxxxxxx) --value--> [type] "target title" (id:xxxxxxxx)  — the row as stored:')
    lines.append('the left node is the source, the right node is the target; value is the candidate string.')
    lines.append('A title or snippet cut for length ends in …')
    lines.append('')
    for name, spec in CLASSIFIER_FIELDS.items():
        lines.append('%s (%s): %s' % (name, ', '.join(spec['applies_to']), spec['meaning']))
        if spec['kind'] == 'enum':
            width = max(len(v) for v in spec['values'])
            for val, meaning in spec['values'].items():
                lines.append('  %s — %s' % (val.ljust(width), meaning))
            lines.append('  Rule: %s' % spec['rule'])
        lines.append('')
    return lines


def _node_head(ntype, title, nid):
    """A node named the way every brain surface names one — the rich-node
    header grammar `[type] "title" (id:xxxxxxxx)` via contract.node_ref, title
    cut at TITLE_REF_LIMIT with the standard mark. One grammar on the page:
    the bracket is always a type, the quotes always a title."""
    return '[%s] %s' % (ntype or '?', node_ref(nid, truncate_marked(title or '', TITLE_REF_LIMIT)))


class AspectEncoder(IntegrationUnit):
    NAME = 'aspect_integration'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:aspect_integration'

    O_SOURCES = ['aspect_proposals']
    K_SOURCES = ['llm_aspect_classifier', 'aspects_v1.json', 'journal_notes']

    # Residue flows to journal_note trace rows via the journal binding on
    # _call_llm (decorate → harvest), read back via continuity(). For this
    # unit the notes are the ONLY durable record of a classification's
    # reasoning — aspects_proposed.json is overwritten every cycle, and a
    # classification is permanent (never re-examined).

    def __init__(self, brain, dispatch_fn=None, config=None):
        super().__init__(brain, dispatch_fn)
        self.config = config or ASPECT

    def run(self, proposals, usage=None):
        """Classify proposals into existing aspects and fill their fields.

        Args:
            proposals: list from decoder. Each: {category, value, count,
                fill, examples[, aspects]}
            usage: {category: {string: count}} from the decoder's scan —
                ranks each aspect's menu sample by how often the brain writes
                a member. None renders the first N in taxonomy order.

        Returns: {classified, rejected, accepted_details, rejected_details,
                  errors, journal, per_aspect}
        """
        if not proposals:
            self.trace('delta', 'aspect_classified', 'No proposals to process')
            return {'classified': 0, 'rejected': 0, 'errors': [], 'journal': ''}

        aspects = self.brain.aspects.all()   # {name: Aspect} — registry view
        # Two blocks, stable first: the menu + field definitions are
        # byte-identical across the batches of one cycle (the menu samples
        # most-used members, which a count-1 newcomer never enters), so
        # run_llm_once's breakpoint on the first block makes every call after
        # the first pay a tenth for it. Journal continuity and the candidates
        # change per call and follow.
        user_content = [
            self._format_menu(aspects, usage),
            self.journal.continuity(chain_id=self.chain_id())
            + self._format_candidates(proposals),
        ]
        result, telemetry = self._call_llm('s2_aspects', user_content,
                                           journal=True)

        if result is None:
            err = 'LLM call failed'
            self.brain._log_error(self.NAME, Exception(err), 'classifying %d proposals' % len(proposals))
            self._trace_failed_delta(err, telemetry, len(proposals))
            return {'classified': 0, 'rejected': 0, 'errors': [err], 'journal': ''}

        # The model's JSON output isn't fully predictable in shape — sometimes
        # returns the wrapped object {classifications: [...]}, sometimes a
        # bare array [...]. Accept both. runner.extract_json picks the first
        # JSON structure it sees, so a bare array comes through as a list.
        if isinstance(result, list):
            classifications = result
        elif isinstance(result, dict):
            classifications = result.get('classifications', [])
        else:
            classifications = []
        if not classifications:
            err = 'response had no classifications'
            self.brain._log_error(self.NAME, Exception(err),
                                  'response type: %s, keys: %s' % (
                                      type(result).__name__,
                                      list(result.keys()) if isinstance(result, dict) else 'n/a'))
            self._trace_failed_delta(err, telemetry, len(proposals))
            return {'classified': 0, 'rejected': 0, 'errors': [err], 'journal': ''}

        # Validate, then write through the registry's single door — it merges
        # (idempotent append), writes atomically, and re-derives its own maps.
        accepted, rejected = self._validate_classifications(classifications, proposals)
        written = self.brain.aspects.add_members(accepted, source=self.ENCODING_SOURCE)

        # Per-aspect counts for trace. Multi-membership: each classification
        # touches potentially multiple aspects — primary first, then secondaries.
        per_aspect_primary = {}
        per_aspect_any = {}
        multi_count = 0
        for c in accepted:
            asp_list = c['aspects']
            if not asp_list:
                continue          # a fill-only entry whose candidate had no homes
            primary = asp_list[0]
            per_aspect_primary[primary] = per_aspect_primary.get(primary, 0) + 1
            for a in asp_list:
                per_aspect_any[a] = per_aspect_any.get(a, 0) + 1
            if len(asp_list) > 1:
                multi_count += 1
        per_aspect = per_aspect_primary  # primary-aspect counts; aspect_integration sums these

        journal = '%d classified (%d multi-aspect), %d rejected. By primary aspect: %s' % (
            len(accepted), multi_count, len(rejected),
            ', '.join('%s=%d' % kv for kv in sorted(per_aspect_primary.items())))

        # Structured Δ for Aspect. It mutates aspects_v1.json, not the graph,
        # so created/revised/archived don't apply — the real change record is
        # WHICH string routed to WHICH aspect(s). Carried in `classifications`
        # (extras) so the per-classification decision survives, not just counts.
        classifications_made = [
            {k: v for k, v in c.items() if k != 'rationale'}
            for c in accepted
        ]
        fields_filled = sum(1 for c in accepted for k in filled_fields(c['category'])
                            if c.get(k) is not None)

        self.trace('delta', 'aspect_classified',
                   '%d classified, %d rejected, %dms, %d→%d tok' % (
                       len(accepted), len(rejected),
                       telemetry.get('elapsed_ms', 0),
                       telemetry.get('input_tokens', 0),
                       telemetry.get('output_tokens', 0)),
                   metadata=build_delta_metadata(
                       actions=len(classifications),
                       write_actions=len(accepted),
                       rounds=1,
                       inputs_processed=len(proposals),
                       outcomes={
                           'classified': len(accepted),
                           'rejected': len(rejected),
                           'fields_filled': fields_filled,
                           **{'aspect_' + k: v for k, v in per_aspect.items()},
                       },
                       journal_entry=journal,
                       errors=[r['reason'] for r in rejected[:5]],
                       classifications=classifications_made,
                       elapsed_ms=telemetry.get('elapsed_ms', 0),
                       input_tokens=telemetry.get('input_tokens', 0),
                       output_tokens=telemetry.get('output_tokens', 0),
                       cache_read_tokens=telemetry.get('cache_read_tokens', 0),
                       cache_creation_tokens=telemetry.get('cache_creation_tokens', 0),
                       model=telemetry.get('model', ''),
                       interaction=telemetry['interaction'],
                   ))

        return {
            'classified': len(accepted),
            'rejected': len(rejected),
            'accepted_details': accepted,
            'rejected_details': rejected,
            # Changes the door actually wrote (memberships + fields). An
            # accepted entry can write nothing — a fill-only entry that left
            # its field open — and the orchestrator's loop reads this, not
            # `classified`, to know whether another pass would see new work.
            'written': written,
            'errors': [],
            'journal': journal,
            'per_aspect': per_aspect,
        }

    def _trace_failed_delta(self, err, telemetry, n_proposals):
        """Delta trace for a failed run — every exit path stamps the unit's
        delta ref_type (the _last_run_timestamp contract), and with the
        journal binding a failed call may already have written residue rows
        on this chain: without a delta they'd be orphaned (unreachable from
        every run view) and the burned tokens unrecorded."""
        self.trace('delta', 'aspect_classified',
                   'run failed: %s (%d proposals, %dms, %d→%d tok)' % (
                       err, n_proposals,
                       telemetry.get('elapsed_ms', 0),
                       telemetry.get('input_tokens', 0),
                       telemetry.get('output_tokens', 0)),
                   metadata=build_delta_metadata(
                       actions=0, write_actions=0, rounds=1,
                       inputs_processed=n_proposals,
                       outcomes={'classified': 0, 'rejected': 0},
                       journal_entry='',
                       errors=[err],
                       elapsed_ms=telemetry.get('elapsed_ms', 0),
                       input_tokens=telemetry.get('input_tokens', 0),
                       output_tokens=telemetry.get('output_tokens', 0),
                       cache_read_tokens=telemetry.get('cache_read_tokens', 0),
                       cache_creation_tokens=telemetry.get('cache_creation_tokens', 0),
                       model=telemetry.get('model', ''),
                       interaction=telemetry['interaction'],
                   ))

    # ─── prompt construction ─────────────────────────────────────────

    def _format_menu(self, aspects, usage=None):
        """Block 1 of the user message: ASPECT MENU + FIELD DEFINITIONS.

        `aspects` is the {name: Aspect} view from brain.aspects.all().
        `usage` ranks each aspect's member sample (see aspects.rank_members).
        """
        lines = []
        routing = _routing_map(aspects)
        cap = self.config['menu_members_per_aspect']
        usage = usage or {}

        lines.append('═' * 70)
        lines.append('ASPECT MENU — %d aspects, closed list. A candidate whose fill names aspects '
                     'is routed to one of these.' % len(routing))
        lines.append('Each aspect lists its most-used members, not every member — a sample of its shape.')
        lines.append('═' * 70)
        lines.append('')

        # Render routable aspects in a stable order: node-only first (Frame),
        # then the semantic rest, then catch-alls (the prompt-invisible
        # structural aspects) last. Helps the encoder build a mental map.
        # Groups derive from the per-aspect facts; within a group, taxonomy
        # (JSON) order — reproduces the hand-curated order this replaced.
        def _menu_group(name):
            if routing[name] == {'node_types'}:
                return 0
            return 1 if aspects[name].prompt_visible else 2

        order = sorted(routing, key=_menu_group)
        for name in order:
            aspect = aspects[name]
            members_n = list(aspect.node_types)
            members_e = list(aspect.edge_relations)
            lines.append('── %s ──' % name)
            lines.append('  meaning: %s' % aspect.meaning)
            lines.append('  accepts: %s' % ', '.join(sorted(routing[name])))
            if members_n:
                lines.append('  current node_types (%s): %s' % rank_members(
                    members_n, usage.get('node_types') or {}, cap))
            if members_e:
                lines.append('  current edge_relations (%s): %s' % rank_members(
                    members_e, usage.get('edge_relations') or {}, cap))
            lines.append('')

        lines.extend(render_field_definitions())
        return '\n'.join(lines) + '\n'

    def _format_candidates(self, proposals):
        """Block 2 of the user message: CANDIDATES + the return line."""
        lines = []
        lines.append('═' * 70)
        lines.append('CANDIDATES — %d strings; each lists the fields to fill' % len(proposals))
        lines.append('═' * 70)
        lines.append('')

        # A candidate is rendered as the entry the encoder returns: the same
        # key names, filled keys shown, missing keys named in `fill`. No word
        # appears here that the encoder does not also write (the tool-schema
        # contract test pins it), so a stateless reader maps page to output
        # with no translation step.
        for i, p in enumerate(proposals, 1):
            lines.append('── #%d ──' % i)
            lines.append('  category: %s' % p['category'])
            lines.append('  value: "%s"' % p['value'])
            lines.append('  count: %d' % p['count'])
            if p.get('aspects'):
                lines.append('  aspects: %s' % ', '.join(p['aspects']))
            lines.append('  fill: %s' % ', '.join(fill_for(p)))
            lines.append('  examples:')
            for ex in p.get('examples', []):
                if p['category'] == 'node_types':
                    lines.append('    - %s' % _node_head(
                        ex.get('type', p['value']), ex.get('title', ''), ex.get('id')))
                    if ex.get('content_snippet'):
                        lines.append('      content: %s' % ex['content_snippet'])
                    if ex.get('situation'):
                        lines.append('      situation: %s' % ex['situation'])
                else:
                    lines.append('    - %s --%s--> %s' % (
                        _node_head(ex.get('src_type', ''), ex.get('src_title', ''), ex.get('src_id')),
                        p['value'],
                        _node_head(ex.get('tgt_type', ''), ex.get('tgt_title', ''), ex.get('tgt_id'))))
                    if ex.get('description'):
                        lines.append('      description: %s' % ex['description'])
                    if ex.get('src_content_snippet'):
                        lines.append('      source content: %s' % ex['src_content_snippet'])
                    if ex.get('tgt_content_snippet'):
                        lines.append('      target content: %s' % ex['tgt_content_snippet'])
            lines.append('')

        lines.append('═' * 70)
        lines.append('Return a JSON array with one entry per candidate, in order — '
                     'copy category and value, add every field named in fill.')

        return '\n'.join(lines)

    # ─── validation + merge ──────────────────────────────────────────

    def _validate_classifications(self, classifications, proposals):
        """Drop classifications that target invalid aspects or wrong categories.

        Multi-membership: each classification carries `aspects` (a list, primary
        first). Validation drops the whole entry if any listed aspect is invalid
        or doesn't accept the category. Returns (accepted, rejected).
        """
        candidates_by_value = {}
        for p in proposals:
            candidates_by_value[(p['category'], p['value'])] = p

        routing = _routing_map(self.brain.aspects.all())
        accepted = []
        rejected = []
        seen = set()

        for c in classifications:
            if not isinstance(c, dict):
                rejected.append({'classification': c, 'reason': 'not a dict'})
                continue

            value = c.get('value', '')
            category = c.get('category', '')
            aspects_list = c.get('aspects')

            # The reply is free JSON text: nothing upstream guarantees these are
            # strings, and an unhashable value would raise in the lookups below
            # and lose the whole reply instead of one entry.
            if not (value and category and isinstance(value, str)
                    and isinstance(category, str)):
                rejected.append({'classification': c, 'reason': 'missing or non-string field (value/category)'})
                continue
            if (category, value) not in candidates_by_value:
                rejected.append({'classification': c, 'reason': 'value+category not in candidate list'})
                continue
            proposal = candidates_by_value[(category, value)]
            fill = fill_for(proposal)

            if 'aspects' not in fill:
                # Already routed: the candidate's homes are fixed facts of
                # the taxonomy (they may include non-routable aspects, which
                # the closed-list check below would wrongly refuse). A
                # returned `aspects` is not in its fill — dropped, recorded
                # (the only signal that the encoder tries to re-route filled
                # candidates).
                homes = list(proposal.get('aspects') or [])
                if aspects_list is not None and aspects_list != homes:
                    self.brain._log_warning(
                        self.NAME,
                        'field not in fill — dropped',
                        'value="%s" category=%s aspects=%s (kept: %s)' % (
                            value, category, aspects_list, homes))
                aspects_list = homes
            else:
                if (not isinstance(aspects_list, list) or not aspects_list
                        or not all(isinstance(a, str) for a in aspects_list)):
                    rejected.append({'classification': c, 'reason': 'missing or non-string field (aspects)'})
                    continue
                # Validate every listed aspect against the closed list (the
                # routable aspects from the registry). Reject on unknown or
                # non-routable aspect names (encoder hallucination).
                invalid = [a for a in aspects_list if a not in routing]
                if invalid:
                    rejected.append({'classification': c,
                                     'reason': 'aspects not in closed list: %s' % invalid})
                    continue
                # Filter to aspects that accept this category. Edge-only aspects
                # for a node_type candidate get filtered out; if any survive,
                # accept the entry with the filtered list (first survivor = new
                # primary). If nothing survives, reject. Logged so we can see
                # how often the encoder picks the wrong category.
                valid_for_cat = [a for a in aspects_list if category in routing[a]]
                dropped = [a for a in aspects_list if a not in valid_for_cat]
                if not valid_for_cat:
                    rejected.append({'classification': c,
                                     'reason': 'no listed aspect accepts category "%s": %s' % (category, aspects_list)})
                    continue
                if dropped:
                    self.brain._log_error(
                        self.NAME,
                        Exception('aspect/category mismatch — filtered'),
                        'value="%s" category=%s dropped=%s kept=%s' % (
                            value, category, dropped, valid_for_cat))
                    aspects_list = valid_for_cat

                # Noise + semantic is a LEGAL answer, kept as returned.
                # Noise vetoes at read time, so the noise member is
                # the load-bearing half: stripping it — as this boundary used to,
                # on the reasoning that a semantic claim refutes noise — deleted
                # exactly the half that does the filtering and silently un-noised
                # machinery the classifier had correctly flagged. Keeping both
                # preserves what the string MEANS while the veto governs what it
                # DOES. Recorded, not refused: dual routing is legal now, so it
                # belongs in the debug log rather than the errors table, and it is
                # the only signal telling us whether the classifier uses this at all.
                if 'noise' in aspects_list and len(aspects_list) > 1:
                    self.brain._log_warning(
                        self.NAME,
                        'noise + semantic aspect — kept both, noise vetoes on read',
                        'value="%s" category=%s aspects=%s' % (
                            value, category, aspects_list))

            if (category, value) in seen:
                rejected.append({'classification': c, 'reason': 'duplicate classification of (%s, %s)' % (category, value)})
                continue

            # Enum fields, per the candidate's fill line. In fill: a valid value
            # is kept; a missing one stays open (warning — the decoder's
            # missing-field scan brings the candidate back next cycle); an
            # unknown value rejects the entry, same class as a hallucinated
            # aspect name. Not in fill (already stored, or not a field of this
            # category at all): dropped and recorded.
            fields = {}
            bad_value = None
            for name, spec in MEMBER_FIELDS.items():
                if spec['kind'] != 'enum':
                    continue
                val = c.get(name)
                if name in fill:
                    if val is None:
                        self.brain._log_warning(
                            self.NAME,
                            'field left open — candidate returns next cycle',
                            'value="%s" category=%s field=%s' % (value, category, name))
                    elif val not in field_values(name):
                        bad_value = (name, val)
                        break
                    else:
                        fields[name] = val
                elif val is not None:
                    self.brain._log_warning(
                        self.NAME,
                        'field not in fill — dropped',
                        'value="%s" category=%s %s=%r' % (value, category, name, val))
            if bad_value:
                rejected.append({'classification': c,
                                 'reason': '%s not in closed list: %r' % bad_value})
                continue
            seen.add((category, value))

            # Normalize to canonical shape on the way out — `aspects` list,
            # primary first, no separate `aspect` key, enum fields flat (the
            # shape add_members reads).
            normalized = {
                'category': category,
                'value': value,
                'aspects': aspects_list,
                'rationale': c.get('rationale', ''),
                **fields,
            }
            accepted.append(normalized)

        return accepted, rejected
