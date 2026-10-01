"""S2 Aspect Integration — orchestrator: decoder → encoder, batched per cycle.

Thin wrapper following the same pattern as Healer / CommunityDetection /
Consolidation. Each pass decodes (unclassified strings + unfilled members),
classifies via the s2_aspects model, and merges into aspects_v1.json through
the registry's door; the registry re-derives on every write, so the next pass
never re-proposes what the last one filled. Only the first pass of a scheduled
cycle is gated on enough pending work (the decoder's batch gate); a forced
cycle runs whatever waits, and once a cycle has called, later passes drain the
rest. Up to `max_calls_per_cycle`
passes, stopping early when nothing is left or a call fails. Calls seconds
apart share the system prompt and the menu block through the cache
breakpoints run_llm_once places. All file I/O, no brain mutations.

Writes the per-cycle audit record (aspects_proposed.json): every accepted
and rejected entry of the cycle, each with the `fill` it was asked for.
"""

from servers.aspect_store import atomic_json_write

from .aspect_contract import aspects_proposed_path
from .aspect_decoder import AspectDecoder
from .aspect_encoder import AspectEncoder


class AspectIntegration(AspectDecoder):
    """Orchestrator: decoder → encoder passes for one S2 cycle."""

    def run(self):
        accepted_all, rejected_all = [], []
        totals = {'actions': 0, 'proposals': 0, 'classified': 0, 'rejected': 0,
                  'calls': 0, 'errors': [], 'per_aspect': {}}
        stats, remaining, skipped = {}, 0, None

        for i in range(self.config['max_calls_per_cycle']):
            decode = super().run(gate=(i == 0 and not self.forced))
            stats = decode.get('stats', {})
            proposals = decode.get('proposals', [])
            if decode.get('skipped') or not proposals:
                skipped = decode.get('skipped')
                remaining = 0
                break
            fills = {(p['category'], p['value']): p['fill'] for p in proposals}

            encoder = AspectEncoder(self.brain, self.dispatch, self.config)
            enc = encoder.run(proposals, usage=decode.get('usage'))
            totals['calls'] += 1
            totals['proposals'] += len(proposals)
            if not enc:
                totals['errors'].append('aspect classification failed')
                break
            for row in enc.get('accepted_details', []):
                accepted_all.append({**row, 'fill': fills.get((row['category'], row['value']), [])})
            for row in enc.get('rejected_details', []):
                entry = row.get('classification')
                key = ((entry.get('category'), entry.get('value'))
                       if isinstance(entry, dict) else None)
                rejected_all.append({**row, 'fill': fills.get(key, [])})
            totals['classified'] += enc.get('classified', 0)
            totals['rejected'] += enc.get('rejected', 0)
            totals['errors'].extend(enc.get('errors', []))
            for k, v in (enc.get('per_aspect') or {}).items():
                totals['per_aspect'][k] = totals['per_aspect'].get(k, 0) + v
            remaining = decode.get('remaining', 0)
            if enc.get('errors') or remaining <= 0:
                break
            if not enc.get('written'):
                # Nothing reached the file (every entry rejected, or every
                # fill left open), so the next decode would propose this same
                # batch again. Stop rather than re-spend the budget on it.
                break

        if totals['calls']:
            self._write_audit_trail(accepted_all, rejected_all)
        if not totals['calls'] and skipped:
            return {'actions': 0, 'skipped': skipped, 'stats': stats}
        totals['actions'] = totals['classified']
        result = {**totals, 'remaining': remaining, 'stats': stats}
        if totals['errors']:
            # The coordinator's failure policy reads `error` (partial progress
            # is kept; the failure still counts toward the unit's consecutive
            # failures). `errors` keeps the per-call detail.
            result['error'] = '; '.join(totals['errors'])
        return result

    def _write_audit_trail(self, accepted, rejected):
        """Per-cycle audit JSON (overwrites the prior cycle's). The operator's
        review surface: one row per entry the cycle accepted or rejected, with
        the fields it was asked for, its answer and its rationale."""
        from datetime import datetime, timezone
        record = {
            'cycle_at': datetime.now(timezone.utc).isoformat(),  # clock-ok — S2 cycle telemetry
            'classifications_accepted': accepted,
            'classifications_rejected': rejected,
        }
        try:
            atomic_json_write(aspects_proposed_path(), record)
        except Exception as e:
            self.brain._log_error(
                self.NAME, e,
                'failed to write audit trail — classifications already merged into aspects_v1.json')
