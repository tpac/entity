"""S2 Healer — orchestrator for node healing.

Two responsibilities, both run on each healer pass:
  1. Algorithmic invariant restoration (no LLM): archive any active edge
     that points at an archived node. Pre-April-2026 archive_node bugs and
     mid-migration races leave dangling edges; this scrubs them every cycle.
  2. LLM-based gap filling: decoder scans for nodes missing
     question/situation/reasoning → encoder generates via Haiku.

Follows the same pattern as CommunityDetection and Consolidation.
"""

from .healer_decoder import HealerDecoder
from .healer_encoder import HealerEncoder


class Healer(HealerDecoder):
    """Orchestrator: invariant restorer → decoder → encoder."""

    def run(self):
        # Invariant restorer (cheap, no LLM): archive edges to archived
        # nodes. Runs every cycle so historical leaks self-heal over time.
        archive_error = None
        try:
            # absorbed_into (survivor_lineage aspect) is exempt — those edges
            # are SUPPOSED to span an archived endpoint (the redirect to the
            # living survivor); scrubbing them as "dangling" would sever the
            # resolve_live chain. archive_exempt_relations() is the single
            # source and logs loudly if the aspect is missing/empty (which would
            # silently disable the exemption and reap redirect edges).
            # Under write_lock: the sweep is a foreground brain.conn write and
            # now COMMITS its own flip (2026-08-06); the emit below is a logs
            # write the emitter contract requires inside the caller's lock.
            from servers.mutation_emitter import (edge_flip_rows,
                                                  emit_mutation_traces)
            from servers.clock import brain_today
            with self.brain.write_lock:
                sweep = self.brain._graph.archive_dangling_edges(
                    archived_by='s2:healer',
                    exempt_relations=self.brain.archive_exempt_relations())
                edges_archived = sweep['archived']
                if sweep['edge_relations']:
                    # One trace row per scrubbed relation (per-edge rows
                    # ruled 2026-08-03, no rollup shape), post-commit, on the
                    # explicit s2 maintenance chain. Own try: the sweep is
                    # COMMITTED by here — a row-shaping failure must degrade
                    # to missing traces (design-sanctioned), never clobber
                    # edges_archived or misattribute the error to the sweep
                    # (review 2026-08-06).
                    try:
                        emit_mutation_traces(
                            self.brain, 'healer_dangling_sweep',
                            {'edges': edge_flip_rows(
                                self.brain.conn, sweep['edge_relations'],
                                's2:healer',
                                'dangling edge — endpoint archived')},
                            chain_id='maint-%s-mutation'
                                     % brain_today(self.brain).strftime('%Y%m%d'))
                    except Exception as emit_err:
                        self.brain._log_error(
                            'healer_sweep_trace_emit', emit_err,
                            'sweep committed %d flips; trace rows lost'
                            % edges_archived)
        except Exception as e:
            edges_archived = 0
            # Capture the error for the result dict so the caller can
            # distinguish "0 to archive" from "archive blew up". Also log
            # to brain errors so it shows in the errors view. Both paths
            # run — visibility is layered.
            archive_error = '%s: %s' % (type(e).__name__, e)
            try:
                self.brain._log_error('healer_archive_dangling', e,
                                      'archive_dangling_edges failed')
            except Exception as log_err:
                # _log_error itself failing is rare; surface to stderr so
                # daemon.log captures it rather than cascade-silencing.
                import sys
                print('[healer] archive_dangling_edges failed + '
                      '_log_error failed: archive=%r log=%r' % (e, log_err),
                      file=sys.stderr, flush=True)

        def _result(base):
            """Build healer result with archive visibility baked in."""
            base['edges_archived'] = edges_archived
            if archive_error is not None:
                base['archive_error'] = archive_error
            return base

        # Decode: find gaps and build proposals
        decode_result = super().run()

        if decode_result.get('skipped'):
            return _result({'actions': edges_archived,
                            'skipped': decode_result['skipped']})

        proposals = decode_result.get('proposals', [])
        if not proposals:
            return _result({'actions': edges_archived,
                            'proposals': 0,
                            'stats': decode_result.get('stats', {})})

        # Encode: generate missing fields via Haiku
        encoder = HealerEncoder(self.brain, self.dispatch, self.config)
        encode_result = encoder.run(proposals)

        if not encode_result:
            return _result({'actions': edges_archived,
                            'error': 'healing failed'})

        return _result({
            'actions': encode_result.get('fields_written', 0) + edges_archived,
            'proposals': len(proposals),
            'nodes_healed': encode_result.get('nodes_healed', 0),
            'fields_written': encode_result.get('fields_written', 0),
            'skipped': encode_result.get('skipped', 0),
            'stats': decode_result.get('stats', {}),
            'errors': encode_result.get('errors', []),
        })
