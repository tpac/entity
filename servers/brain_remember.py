"""
brain — BrainRemember Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from . import embedder
from .brain_constants import TYPE_CONFIDENCE
from .dal_graph import absorb_migrates_relation
from .clock import iso_now
from .loud_truncation import cap_text_loud
from .contract import (ALL_FIELDS, FIELDS_KEY, FOLD_RULE, READ_ONLY_ATTACHMENT_KEYS,
                       REVISE_IMMUTABLE_FIELDS, absorb_folded, apply_swaps,
                       connect_to_target, connect_to_description,
                       connect_to_uses_old_key, is_swap,
                       is_swap_list, refusal_pointers, unpack_fields, validate_swaps)
from typing import Any, Dict, List, Optional
import json
import re
import time

# ── connect_to catalog-title matching (deterministic, no vectors) ──

_TITLE_TOKEN_RE = re.compile(r'[a-z0-9]+')


def _title_tokens(text):
    """Normalize a title to its token sequence: NFKD → strip combining marks
    → lowercase → split on every non-alphanumeric run (hyphen/underscore/
    em-dash/percent variance vanishes; numbers and hashes survive — they're
    the distinctive tokens). No stemming, no stopwords: predictability IS
    the safety property of the near-title matcher.

    The diacritic strip mirrors FTS5 unicode61's remove_diacritics default —
    the two tokenizers must agree or a probe goes dead: unicode61 indexes
    'über' as 'uber', and a bare [a-z0-9] scan would emit the probe 'ber',
    which matches nothing (bug 69c2cbab #4 — tokenizer correspondence)."""
    import unicodedata
    decomposed = unicodedata.normalize('NFKD', text or '')
    stripped = ''.join(c for c in decomposed if not unicodedata.combining(c))
    return _TITLE_TOKEN_RE.findall(stripped.lower())


def _token_edit_distance(a, b, cap):
    """Levenshtein over token SEQUENCES (order-sensitive on purpose — a
    reordered title is not 'a bit off'). Early-exits at `cap`: any distance
    beyond it returns cap+1 — callers only need 'too far', not how far."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a):
        cur = [i + 1]
        row_min = cur[0]
        for j, tb in enumerate(b):
            c = min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ta != tb))
            cur.append(c)
            row_min = min(row_min, c)
        if row_min > cap:
            return cap + 1
        prev = cur
    return min(prev[-1], cap + 1)


def _swap_intended(value):
    """A value that is TRYING to be a swap — a dict carrying `old` or `new`,
    or a list holding one — but is not well-formed. Distinguishes a typo'd
    swap (refuse loudly) from an open KV key legitimately holding a list."""
    def _looks(d):
        return isinstance(d, dict) and ('old' in d or 'new' in d)
    return _looks(value) or (isinstance(value, list) and any(_looks(x) for x in value))


def _stringified_swap(value):
    """A string that decodes to a swap or swap list, well-formed or not.

    A client flattens an object into a string when its tool schema doesn't
    declare the parameter — every open KV key on `revise` (the schema lists
    only the contract's fields). Stored, the swap's own JSON becomes the
    field's text. Any other string, JSON-looking or not, is a legitimate text
    value."""
    if not (isinstance(value, str) and value.lstrip().startswith(('{', '['))):
        return False
    try:
        return _swap_intended(json.loads(value))
    except (ValueError, RecursionError):
        return False


class BrainRememberMixin:
    """Remember methods for Brain."""

    # ═══════════════════════════════════════════════════════════════
    # Unified metadata storage — single path for remember() and revise()
    # ═══════════════════════════════════════════════════════════════

    # Keys that are never node metadata: control parameters consumed by
    # remember()/revise() logic, plus the read-only attachments a node view
    # carries (connections, communities, _metadata, ...), which remember() and
    # revise() name as dropped before they get here.
    _CONTROL_FIELDS = (frozenset({'reason', 'updates', 'connect_to'})
                       | READ_ONLY_ATTACHMENT_KEYS)

    def _store_node_metadata(self, node_id: str, fields: Dict[str, Any],
                             caller: str = 'unknown') -> int:
        """Store metadata fields for a node. Single path for all write operations.

        Routes each field to the correct storage:
          - STRUCTURAL_FIELDS → already on nodes table (skip)
          - situation → node_metadata_kv (canonical; _situation embedding
            derived later by backfill)
          - PROMOTED metadata_kv fields → node_metadata_kv
          - Emergent/unknown fields → node_metadata_kv
          - Control fields and read-only attachments → skip (_CONTROL_FIELDS)

        Warns on fields that don't match any storage path.

        Returns count of fields stored.
        """
        from .contract import STRUCTURAL_FIELDS, PROMOTED_FIELDS

        kv_fields = {}
        stored = 0

        for field, value in fields.items():
            # Control params — consumed by callers, never stored
            if field in self._CONTROL_FIELDS:
                continue

            # Structural — already on nodes table, handled by INSERT/UPDATE
            if field in STRUCTURAL_FIELDS:
                continue

            # Empty values — skip
            if value is None or (isinstance(value, str) and not value.strip()):
                continue

            # Situation lives in node_metadata_kv as of v24 — alongside
            # question, reasoning, etc. The embedding (BLOB) is generated
            # later by backfill_vectors() reading from kv.
            #
            # Pass raw values through to set_many — its _encode_value handles
            # str / list / dict / primitives consistently. Doing str(value)
            # here would str()-ify lists into Python repr (`"['a','b']"`)
            # which isn't JSON-parseable. Aspects (Step 5b) need clean lists.
            if field == 'situation':
                kv_fields['situation'] = value
                continue

            # Promoted metadata_kv field — store in KV
            if field in PROMOTED_FIELDS:
                pf = PROMOTED_FIELDS[field]
                if pf.get('store') == 'metadata_kv':
                    kv_fields[field] = value
                    continue
                # Promoted field with different store — log error so it's visible
                self._log_error('store_metadata_unhandled',
                                ValueError('field "%s" (store=%s) not handled' % (
                                    field, pf.get('store'))),
                                '%s: node %s' % (caller, node_id[:8]))
                continue

            # Emergent field — any unknown field goes to metadata_kv
            kv_fields[field] = value

        if kv_fields:
            try:
                count = self._meta_kv.set_many(node_id, kv_fields)
                stored += count
            except Exception as _e:
                self._log_error('store_metadata_kv', _e,
                                '%s: KV for %s (%d fields)' % (
                                    caller, node_id[:8], len(kv_fields)))

        return stored

    # ═══════════════════════════════════════════════════════════════
    # Unified archive — single path for all archive operations
    # ═══════════════════════════════════════════════════════════════

    def _resync_vector_cache(self, where: str) -> None:
        """Restore cache == DB after a rollback. Cache mutations inside a
        batch envelope are eager (the DAL drops cache rows when it deletes DB
        rows); a rollback restores the DB but not the cache — without this,
        a rolled-back revise/archive/absorb leaves a LIVE node invisible to
        cache-served scans until restart (the inverse of the 2026-07-17
        healer bug). reload() is one SELECT; rollback is the rare loud path,
        so the cost is fine. Plain VectorDAL has no cache — nothing to do."""
        reload_fn = getattr(self._vec_dal, 'reload', None)
        if reload_fn is None:
            return
        try:
            reload_fn()
        except Exception as _e:
            self._log_error('vector_cache_resync', _e, where)

    def _deindex_node(self, node_id: str) -> int:
        """Remove a node from the search indexes (vectors and FTS5) — the one
        de-indexing path, used by archive_node. Returns enrichment rows deleted
        (trace metadata).

        Composes with batch envelopes: each store's delete gates its commit on
        conn.in_batch. Fts5DAL.delete tolerates a missing table and logs its
        own failures (loud-by-default lives there); the sqlite_master probe
        just avoids a stderr line on test DBs that lack the virtual table."""
        vectors_deleted = self._vec_dal.delete_for_node(node_id)
        has_fts5 = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='nodes_fts'"
        ).fetchone() is not None
        if has_fts5:
            self._fts.delete(node_id)   # self-guarding: logs internally
        self._maybe_commit()            # Fts5DAL.delete doesn't self-commit
        return vectors_deleted

    def archive_exempt_relations(self) -> tuple:
        """Relations exempt from dangling-edge archival — the survivor-redirect
        links (survivor_lineage aspect: absorbed_into) that must outlive the
        nodes they leave. Single source for archive_node + the Healer; the DAL
        stays aspect-agnostic and just takes the strings.

        LOUD on empty: survivor_lineage is a REQUIRED aspect, so an empty result
        means the registry is unset (init failed — brain.py leaves self.aspects
        unset by design) or the working copy predates the aspect and hasn't
        self-healed yet. Either way the exemption is DISABLED and the
        dangling-edge reaper will scrub absorbed_into redirect edges — a silent
        correctness failure, so we log it (rate-limited) instead of letting it
        pass quietly or raising AttributeError mid-archive."""
        rels = ()
        if getattr(self, 'aspects', None) is not None:
            rels = tuple(self.aspects.relations_in(['survivor_lineage']))
        if not rels:
            self._log_error(
                'survivor_lineage_exempt_empty',
                ValueError('survivor_lineage aspect missing/empty'),
                'absorbed_into archival exemption DISABLED — redirect edges '
                'will be reaped by the dangling-edge sweep')
        return rels

    def archive_node(self, node_id: str, archived_by: str,
                     reason: str = '', survivor_id: str = None,
                     extra: Dict[str, Any] = None) -> Dict[str, Any]:
        """Archive a node. Single path for all callers.

        What it does:
          1. Guards: rejects locked nodes
          2. Sets archived=1, updated_at=now
          3. Stores audit metadata: archived_by, archived_reason, archived_at
          4. Soft-archives edge_relations (v25 — archived=1 preserves history
             for future recovery; edges aggregate row stays for edge_id stability)
          5. Deletes vectors from node_enrichments (embeddings are expensive to keep)
          6. Removes from FTS5 index

        Args:
            node_id: Node to archive.
            archived_by: Who is archiving. Convention: "s2:consolidation",
                         "s2:community_detection", "hook:integrity", "anchor", etc.
            reason: Human-readable reason for the archive.
            survivor_id: Live node this one's content survives in (absorb,
                         supersession). Writes the `_sys_archived_survivor_id`
                         pointer resolve_live walks + the `absorbed_into`
                         lineage edge. A first-class parameter, NOT an extra
                         key — a misspelled extra key silently loses lineage
                         (the `superseded_by` bug, 2026-07-30).
            extra: Optional dict of additional AUDIT metadata to store.
                   Never carries behavior; survivor-looking keys are refused
                   loudly (see tripwire below).

        Returns:
            Dict with ok=True/False and details.
        """
        ts = iso_now()

        # Fetch node
        row = self.conn.execute(
            'SELECT id, locked, title, type FROM nodes WHERE id = ?',
            (node_id,)).fetchone()
        if not row:
            return {'ok': False, 'error': 'Node not found', 'node_id': node_id}

        full_id, locked, title, node_type = row

        # Guard: never archive locked nodes
        if locked:
            self._log_error('archive_guarded',
                            ValueError('Cannot archive locked node'),
                            '%s tried to archive %s "%s"' % (
                                archived_by, node_id[:8], (title or '')[:40]))
            return {'ok': False, 'error': 'Cannot archive locked node',
                    'node_id': node_id}

        # 1. Set archived=1
        self.conn.execute(
            'UPDATE nodes SET archived = 1, updated_at = ? WHERE id = ?',
            (ts, full_id))

        # 2. Store audit metadata (_sys_ prefix = system fields, filtered from LLM rendering)
        audit = {
            '_sys_archived_by': archived_by,
            '_sys_archived_reason': reason or 'no reason provided',
            '_sys_archived_at': ts,
        }
        if survivor_id:
            audit['_sys_archived_survivor_id'] = str(survivor_id)
        if extra:
            for k, v in extra.items():
                if v is None:
                    continue
                # Tripwire: survivor semantics must go through the survivor_id
                # PARAMETER — a survivor-looking extra key is the exact
                # misspelling that silently orphaned 9 handoff nodes
                # (`superseded_by`, 2026-07-30). Refuse it loudly rather than
                # store a pointer resolve_live will never walk.
                if k in ('survivor_id', 'superseded_by', 'survivor',
                         'consolidated_into', 'absorbed_into'):
                    self._log_warning(
                        'archive_survivor_key_in_extra',
                        "extra key %r on archive of %s looks like survivor "
                        "lineage — dropped; pass survivor_id= instead"
                        % (k, node_id[:8]), 'archived_by=%s' % archived_by)
                    continue
                audit['_sys_archived_%s' % k] = str(v)
        try:
            self._meta_kv.set_many(full_id, audit)
        except Exception as _e:
            self._log_error('archive_metadata', _e,
                            'storing audit for %s' % node_id[:8])

        # 3–5 run inside ONE connection batch envelope. delete_node_edges and
        # add_relation self-commit via commit_unless_batched, so without this a
        # STANDALONE archive (consolidation orphan-heal, hook:integrity — paths
        # that don't set conn.in_batch) would commit the node+edges and then a
        # failure in the vector/FTS cleanup would leave a half-archived node:
        # archived=1 but still in FTS5 → it resurfaces in recall (the exact
        # dead-node-leak class). Composes with an outer batch (absorb): defers
        # the commit/rollback to the owner via the saved in_batch state.
        prior = self.conn.in_batch
        self.conn.in_batch = True
        try:
            # 3. Soft-archive edge_relations touching this node (v25) via the
            # DAL — single source, no inline SQL. The survivor-redirect
            # relations (survivor_lineage aspect: absorbed_into) are EXEMPT —
            # they must outlive the node so the resolve_live chain A→B→C
            # survives B's own archival. edge_relations holds the flipped
            # [edge_id, relation] pairs (observed truth, exemptions excluded)
            # for the caller's node_archived trace row.
            edge_relations = self._graph.delete_node_edges(
                [full_id], archived_by=archived_by,
                exempt_relations=self.archive_exempt_relations())

            # 3b. Survivor-redirect edge. When this archive carries a survivor
            # (the absorb op, consolidation merges, handoff supersession all
            # route here via the survivor_id parameter), record
            # absorbed→survivor as a first-class `absorbed_into` edge,
            # multi-homed in correction_improvement (correction_enrich walks it)
            # + survivor_lineage (the redirect/archival-exempt role). Written
            # AFTER the soft-archive above (which exempts it) so it lands and
            # stays archived=0. `_sys_archived_survivor_id` is still written
            # (step 2 audit) as the backfill source + resolve_live's read path.
            absorbed_into_edge = None
            if survivor_id and survivor_id != full_id:
                try:
                    _res = self._graph.add_relation(
                        full_id, survivor_id, 'absorbed_into',
                        description=reason or 'absorbed into %s' % survivor_id[:8],
                        encoding_source=archived_by)
                    absorbed_into_edge = {
                        'source_id': full_id, 'target_id': str(survivor_id),
                        'relation': 'absorbed_into',
                        'edge_id': _res.get('edge_id') or '',
                        'deltas': _res.get('deltas') or [],
                        'warnings': list(_res.get('warnings') or []),
                    }
                except Exception as _e:
                    self._log_error('archive_absorbed_into_edge', _e,
                                    'absorbed_into %s -> %s' % (
                                        full_id[:8], str(survivor_id)[:8]))

            # 4. De-index INSIDE the archive transaction — atomic with the
            # archived=1 flag: a de-index failure rolls the WHOLE archive back
            # (the caller sees a clean failure, no half-archived node, no lost
            # trace). The eager cache drop is covered by resync-on-rollback below; when
            # archive runs inside an outer envelope, the deletes join the outer
            # commit and that envelope's rollback resyncs.
            vectors_deleted = self._deindex_node(full_id)

            if not prior:
                self.conn.commit()  # commit-ok: single atomic archive
        except Exception:
            if not prior:
                self.conn.rollback()
                self._resync_vector_cache('archive_node rollback')
            raise
        finally:
            self.conn.in_batch = prior

        # No inline trace: the caller shapes this return into a node_archived
        # manifest row and the dispatch chokepoint emits it post-commit
        # (servers/mutation_emitter.py — THE one mutation-trace writer). The
        # legacy inline emit died here at plan step 8: it wrote mid-envelope
        # to a different DB, so it could orphan a trace for an archive that
        # then rolled back.
        return {
            'ok': True,
            'node_id': full_id,
            'title': (title or '')[:60],
            'type': node_type,
            'archived_by': archived_by,
            'reason': reason,
            'edges_deleted': len(edge_relations),
            'edge_relations': edge_relations,
            'absorbed_into_edge': absorbed_into_edge,
            'vectors_deleted': vectors_deleted,
        }

    # Pending lock-flip confirmations: token -> {node_id, locked, requested_at}.
    # In-memory by design — lock flips are rare and operator-present; a pending
    # confirm must not outlive a daemon restart without being re-requested.
    LOCK_CONFIRM_TTL_S = 600

    def set_node_lock(self, node_id: str, locked: bool,
                      reason: str = '', confirm_token: str = None,
                      encoding_source: str = 'anchor',
                      session_id: str = '') -> Dict[str, Any]:
        """Flip the locked flag on an existing node — the ONE door for lock
        transitions. revise() deliberately treats `locked` as immutable; this
        is the explicit operator path, not a relaxation of that contract.

        Two-phase: without confirm_token nothing executes — returns
        confirmation_required=True, a one-shot token (10-min TTL, in-memory)
        and a human-readable summary the caller must relay to the operator
        for an explicit yes. Called again with the token, it flips the flag.
        A token is bound to (node, direction, requesting session): a confirm
        from a different session is refused, and a mismatched confirm does
        NOT consume the pending token. Any phase-2 match — including one that
        lands as a no-op — consumes it.

        Anchor-only at this boundary (mirrors remember()'s creation-time lock
        rule): a non-anchor encoding_source is refused here, and the shared
        S1/S2 encoder dispatch closure refuses the command as defense-in-depth.

        `reason` is relayed in the phase-1 summary; the durable audit is the
        node_lock_changed trace the dispatch layer emits — a direct caller
        bypassing dispatch gets no persistence beyond that.

        Guards: node must exist; an archived node cannot be locked; already
        in the requested state is a no-op (no confirmation round-trip).
        """
        if not isinstance(locked, bool):
            return {'ok': False, 'node_id': node_id,
                    'error': 'locked must be a boolean, got %s (%r)'
                             % (type(locked).__name__, locked)}
        if encoding_source and not encoding_source.startswith('anchor'):
            self._log_error('lock_non_anchor',
                            ValueError('set_node_lock is anchor-only'),
                            'refused encoding_source=%s for %s'
                            % (encoding_source, node_id[:8]))
            return {'ok': False, 'node_id': node_id,
                    'error': 'set_node_lock is anchor-only '
                             '(encoding_source=%s)' % encoding_source}

        row = self.conn.execute(
            'SELECT id, locked, archived, title, type, created_at, access_count '
            'FROM nodes WHERE id = ?', (node_id,)).fetchone()
        if not row:
            return {'ok': False, 'error': 'Node not found', 'node_id': node_id}
        full_id, old_locked, archived, title, node_type, created_at, access_count = row
        old_locked = bool(old_locked)

        if locked and archived:
            # Writes never redirect — an absorbed id refuses with the
            # pointer (the shared write-door contract).
            surv = self._nodes.survivor_of(full_id)
            if surv:
                from .contract import absorbed_refusal
                msg = absorbed_refusal('lock', full_id, surv)
                self._log_error('lock_archived_node', ValueError(msg),
                                'set_node_lock(%s, locked=True)' % full_id[:8])
                return {'ok': False, 'error': msg,
                        'node_id': full_id, 'survivor_id': surv}
            self._log_error('lock_archived_node',
                            ValueError('Cannot lock an archived node'),
                            'set_node_lock(%s, locked=True)' % full_id[:8])
            return {'ok': False, 'error': 'Cannot lock an archived node',
                    'node_id': full_id}

        pending = getattr(self, '_pending_lock_confirms', None)
        if pending is None:
            pending = self._pending_lock_confirms = {}
        now = time.time()
        for tok in [t for t, p in pending.items()
                    if now - p['requested_at'] > self.LOCK_CONFIRM_TTL_S]:
            del pending[tok]

        verb = 'lock' if locked else 'unlock'
        if not confirm_token:
            if old_locked == locked:
                return {'ok': True, 'node_id': full_id, 'changed': False,
                        'note': 'node already %s' % ('locked' if locked else 'unlocked')}
            import secrets
            token = '%s-%s' % (verb, secrets.token_hex(4))
            pending[token] = {'node_id': full_id, 'locked': locked,
                              'requested_at': now, 'session_id': session_id or ''}
            operator = self.operator_name or 'the human operator'
            return {
                'ok': True, 'confirmation_required': True,
                'confirm_token': token, 'node_id': full_id,
                'summary': '%s [%s] "%s" (created %s, %d accesses) — reason: %s '
                           '— needs an explicit yes from %s before re-calling '
                           'with the token'
                           % (verb.upper(), node_type, (title or '')[:60],
                              (created_at or '?')[:10], access_count or 0,
                              reason or 'none given', operator),
            }

        # Validate BEFORE consuming: a mismatched confirm (wrong node id,
        # wrong direction, another session's token) must not burn a pending
        # confirmation the operator is about to give.
        spec = pending.get(confirm_token)
        mismatch = (spec is None
                    or spec['node_id'] != full_id
                    or spec['locked'] != locked
                    or (spec['session_id'] and spec['session_id'] != session_id))
        if mismatch:
            # Expiry/decline-then-retry is a normal operator flow — warning,
            # not error, so the error feed stays real errors only.
            self._log_warning('lock_confirm_invalid',
                              'invalid/expired/mismatched lock confirm token',
                              'set_node_lock(%s, locked=%s, token=%s)'
                              % (full_id[:8], locked, str(confirm_token)[:16]))
            return {'ok': False, 'node_id': full_id,
                    'error': 'Confirmation token invalid or expired — call again '
                             'without confirm_token to request a fresh one'}
        del pending[confirm_token]  # one-shot: consumed on every match below

        if old_locked == locked:
            return {'ok': True, 'node_id': full_id, 'changed': False,
                    'note': 'node already %s' % ('locked' if locked else 'unlocked')}

        latency = now - spec['requested_at']
        if latency < 2.0:
            # A human read-and-reply takes seconds; an instant confirm is
            # affirmative evidence the caller skipped the relay. Loud, not
            # blocking — the two-phase door is a forcing function by design.
            self._log_warning('lock_confirm_instant',
                              'confirm arrived %.1fs after request — likely '
                              'self-confirmed without a human turn' % latency,
                              'set_node_lock(%s, locked=%s)' % (full_id[:8], locked))
        self._nodes.set_locked(full_id, locked)
        return {
            'ok': True, 'node_id': full_id, 'changed': True,
            'title': (title or '')[:60], 'type': node_type,
            'locked': locked,
            'confirm_latency_s': round(latency, 1),
            'deltas': [{'field': 'locked', 'old': old_locked, 'new': locked}],
        }

    def absorb(self, survivor_id: str, absorbed_id: str,
               content: str = None, reason: str = '', updates=None,
               prune_edges=None, drop_fields=None,
               archived_by: str = 'anchor', **kwargs) -> Dict[str, Any]:
        """Merge: fold absorbed_id INTO survivor_id, then archive absorbed.

        Transfer-by-default. Everything the caller doesn't deliberately override
        moves structurally — so a merge can't silently drop information the way
        the imperative revise+connect+archive dance does (preservation audit,
        node 988de522). The caller hand-writes only the synthesis (`content`)
        and names what to drop (`prune_edges`, `drop_fields`).

        Transfers:
          - source_refs  → union onto survivor (INSERT OR IGNORE dedups)
          - edges        → re-point absorbed's external edges to survivor,
                           upsert-dedup, drop the absorbed<->survivor intra edge;
                           community membership migrates only when BOTH inputs
                           are communities (live ordinary-node members only)
          - access_count → survivor += absorbed (usage history is additive)
          - metadata KV  → fill keys survivor LACKS from absorbed; survivor wins;
                           `_sys_` keys skipped
          - ANY field    → caller overrides via `content`, an `updates` dict, OR
                           field kwargs (title=, confidence=, situation=, type=,
                           ...) — the SAME shape as revise(). Applied
                           through revise() AFTER the auto-transfers, so explicit
                           overrides win over KV-fill. Untouched fields keep the
                           survivor's value; absorbed's content persists on the
                           archived husk. Auto-transfers above cover only the
                           unambiguously-additive dimensions; everything
                           judgment-laden (confidence, title, ...) is explicit.
        Then archive_node(absorbed, survivor_id=...) for provenance.

        Guards: absorbed must be archivable (NOT locked) — the type
        constraint that makes "a locked node is always the survivor" structural.
        survivor must exist and be unarchived; survivor != absorbed.

        Atomicity: the merge is several writes, so it is all-or-nothing. In a
        batch we nest a SAVEPOINT (a mid-merge failure rolls back ONLY our writes,
        not the whole batch); standalone we flip conn.in_batch so the composed DAL
        writers don't commit mid-merge, then commit once at the end — or roll back
        on ANY failure (a raise, a failed override-revise, or a refused archive).
        No partial merge ever commits.
        """
        if survivor_id == absorbed_id:
            return {'ok': False, 'error': 'survivor and absorbed are the same node'}

        rows = self._nodes.get_bulk([survivor_id, absorbed_id])
        if survivor_id not in rows:
            return {'ok': False, 'error': 'survivor not found', 'node_id': survivor_id}
        if absorbed_id not in rows:
            return {'ok': False, 'error': 'absorbed not found', 'node_id': absorbed_id}
        if rows[survivor_id]['archived']:
            live = self._nodes.survivor_of(survivor_id)
            if live:
                from .contract import absorbed_refusal
                return {'ok': False,
                        'error': absorbed_refusal('absorb into',
                                                  survivor_id, live),
                        'node_id': survivor_id, 'survivor_id': live}
            return {'ok': False, 'error': 'survivor is archived',
                    'node_id': survivor_id}

        absorbed = rows[absorbed_id]
        a_locked = absorbed['locked']
        a_archived, a_access = absorbed['archived'], absorbed['access_count']
        if a_locked:
            self._log_warning(
                'absorb_guarded',
                'Cannot absorb locked node %s into %s — absorbed must be archivable' % (
                    absorbed_id[:8], survivor_id[:8]),
                'absorb refused: the absorbed node is locked (only the survivor may '
                'be locked)')
            return {'ok': False, 'error': 'Cannot absorb locked node',
                    'absorbed_id': absorbed_id}
        if a_archived:
            return {'ok': False, 'error': 'absorbed node is already archived',
                    'absorbed_id': absorbed_id}

        # Caller field overrides — revise() shape: content / updates / kwargs.
        field_updates = dict(updates or {})
        field_updates.update(kwargs)
        if content is not None:
            field_updates['content'] = content
        # Absorb's overrides pass no revise door, so `fields` unpacks here,
        # before the KV-fill below reads the keys.
        field_updates, err = unpack_fields(field_updates, revising=True)
        if err:
            return {'ok': False, 'error': err + '. Nothing was written.',
                    'node_id': survivor_id}

        report = {'ok': True, 'survivor_id': survivor_id,
                  'absorbed_id': absorbed_id, 'absorbed_archived': False}

        # ── Atomicity envelope (see docstring) ──
        was_batch = self.conn.in_batch
        if was_batch:
            self.conn.execute('SAVEPOINT absorb_sp')
        self.conn.in_batch = True   # composed DAL writers must NOT commit mid-merge
        success = False
        try:
            # 1. source_refs — union onto survivor
            absorbed_refs = self._source_refs.get_source_refs(absorbed_id)
            report['source_refs_added'] = (
                self._source_refs.add_source_refs(survivor_id, absorbed_refs)
                if absorbed_refs else 0)

            # 2. edges — re-point absorbed's external edges to survivor,
            # preserving each relation's weight + description.
            # The graph contract distinguishes merging stories from assigning
            # an ordinary node to the absorbed node's former communities.
            prune = set(prune_edges or [])
            survivor_type, absorbed_type = rows[survivor_id]['type'], absorbed['type']
            expected_members = None
            if ('community_member' not in prune
                    and survivor_type == absorbed_type == 'community'):
                members = self._graph.get_members_bulk([survivor_id, absorbed_id])
                expected_members = {m['id'] for ms in members.values() for m in ms}
            conns = self._graph.get_connections_bulk(
                [absorbed_id]).get(absorbed_id, [])
            # Each migration's add_relation result (edge_id + deltas) is kept —
            # the dispatch caller shapes them into the mutation manifest so the
            # emitter can trace the re-pointed edges. Discarded before step 7.
            migrated_edges = []
            for c in conns:
                neighbor = c['id']
                if neighbor == survivor_id:
                    continue  # intra-pair edge dies with the absorbed node
                outgoing = c.get('direction') == 'outgoing'
                for rel in c.get('relations', []):
                    relation = rel.get('relation')
                    if (not relation or relation in prune
                            or not absorb_migrates_relation(
                                relation, survivor_type, absorbed_type, c['type'])):
                        continue
                    src, tgt = ((survivor_id, neighbor) if outgoing
                                else (neighbor, survivor_id))
                    kw = {'encoding_source': archived_by}
                    if rel.get('description'):
                        kw['description'] = rel['description']
                    if rel.get('weight') is not None:
                        kw['weight'] = rel['weight']
                    res = self._graph.add_relation(src, tgt, relation, **kw)
                    migrated_edges.append({
                        'source_id': src, 'target_id': tgt,
                        'relation': relation,
                        'edge_id': res.get('edge_id') or '',
                        'deltas': res.get('deltas') or [],
                        'warnings': list(res.get('warnings') or []),
                    })
            report['edges_migrated'] = len(migrated_edges)
            report['migrated_edges'] = migrated_edges

            # Validate the automatic transfer while the original endpoint
            # types still apply. The explicit revise below may intentionally
            # retype the survivor; that does not undo the transferred links.
            if expected_members is not None:
                actual_members = {m['id'] for m in self._graph.get_members_bulk(
                    [survivor_id]).get(survivor_id, [])}
                missing = expected_members - actual_members
                if missing:
                    raise RuntimeError('community absorb lost member links: %s'
                                       % ', '.join(sorted(missing)))

            # 3. access_count — additive (usage history)
            if a_access:
                self.conn.execute(
                    'UPDATE nodes SET access_count = access_count + ? WHERE id = ?',
                    (a_access, survivor_id))
            report['access_count_added'] = a_access or 0

            # 4 + 5. Fill KV the survivor LACKS from absorbed, then apply caller
            # overrides — through ONE revise() call so embedding-bearing fields
            # (situation, ...) get re-embedded and there is a single field-write
            # path. Caller overrides win (field_updates last); _sys_ never moves.
            drop = set(drop_fields or [])
            kv = self._meta_kv.get_all_bulk([survivor_id, absorbed_id])
            s_kv, a_kv = kv.get(survivor_id, {}), kv.get(absorbed_id, {})
            fill = {k: v for k, v in a_kv.items()
                    if not k.startswith('_sys_') and k not in drop
                    and k not in field_updates
                    and (v or '').strip() and not (s_kv.get(k) or '').strip()}
            # Voice fields are the exception to survivor-wins-drop. A distinctive
            # operator/Anchor quote on the absorbed node is meaning paraphrase
            # can't recover; dropping it because the survivor already has a
            # (different) quote is real signal loss. When BOTH carry a quote and
            # they're genuinely distinct, merge-append instead of dropping the
            # absorbed one. (survivor-lacks is already covered by `fill`; caller
            # override via field_updates and `drop` still win below.)
            voice = {}
            for vf in ('their_raw_quote', 'my_raw_quote'):
                if vf in drop or vf in field_updates:
                    continue
                s_val = (s_kv.get(vf) or '').strip()
                a_val = (a_kv.get(vf) or '').strip()
                if (s_val and a_val and a_val != s_val
                        and a_val not in s_val and s_val not in a_val):
                    voice[vf] = '%s\n\n%s' % (s_val, a_val)
            merged = {**fill, **voice, **field_updates}
            rev_failed = False
            if merged:
                rev = self.revise(survivor_id, updates=merged,
                                  reason=reason or 'absorb %s' % absorbed_id[:8])
                report['voice_merged'] = sorted(voice.keys())
                report['fields_filled'] = sorted(fill.keys())
                report['fields_revised'] = sorted(k for k in field_updates
                                                  if k != 'connect_to')
                # Survivor-revise deltas ride the report so the dispatch caller
                # can manifest a node_revised row for the merge (emitter input);
                # discarded before step 7.
                report['deltas'] = rev.get('deltas', [])
                if rev.get('warnings'):
                    report['revise_warnings'] = rev['warnings']
                if rev.get('error'):
                    report['ok'] = False
                    report['revise_error'] = rev['error']
                    rev_failed = True

            # 6. archive the absorbed node — SKIPPED if the override-revise failed,
            # so a failed synthesis never destroys the source.
            if not rev_failed:
                arch = self.archive_node(
                    absorbed_id, archived_by=archived_by,
                    reason=reason or 'absorbed into %s' % survivor_id[:8],
                    survivor_id=survivor_id)
                report['absorbed_archived'] = arch.get('ok', False)
                if arch.get('ok'):
                    success = True
                    # The absorbed node's archive details — the dispatch
                    # caller shapes them into the node_archived manifest row
                    # + the absorbed_into edge row.
                    report['absorbed_archive'] = arch
                else:
                    report['ok'] = False
                    report['archive_error'] = arch.get('error')
        except Exception:
            self._absorb_unwind(was_batch)
            raise

        if success:
            if was_batch:
                self.conn.execute('RELEASE absorb_sp')
            self.conn.in_batch = was_batch
            self._maybe_commit()
            # Whatever form the content took (whole value, swaps, none), the
            # merge keeps what the absorbed node said only if the survivor now
            # carries it — contract.FOLD_RULE. Loud, not refused: a survivor
            # that already states the claim is a legitimate merge without the
            # citation.
            survivor = self._nodes.get_bulk([survivor_id]).get(survivor_id) or {}
            if not absorb_folded(survivor_id, absorbed_id, absorbed.get('content'),
                                 rows[survivor_id].get('content') or '',
                                 survivor.get('content') or ''):
                note = ("absorb: the survivor's content does not keep the "
                        "absorbed node (id:%s), so what it said may be lost "
                        "with the archive. %s" % (absorbed_id[:8], FOLD_RULE))
                report.setdefault('warnings', []).append(note)
                self._log_warning('absorb_content_unfolded', note,
                                  'absorb %s into %s' % (absorbed_id[:8],
                                                         survivor_id[:8]))
        else:
            # guard/override/archive failure mid-merge — undo every transfer so
            # no half-merge is ever committed.
            self._absorb_unwind(was_batch)
        return report

    def _absorb_unwind(self, was_batch: bool) -> None:
        """Roll back an in-flight absorb. In a batch: roll the SAVEPOINT back
        (the rest of the batch survives). Standalone: rollback the implicit txn.
        Always restores conn.in_batch."""
        try:
            if was_batch:
                self.conn.execute('ROLLBACK TO absorb_sp')
                self.conn.execute('RELEASE absorb_sp')
            else:
                self.conn.rollback()
        finally:
            # In finally, not the try body: absorb archives the absorbed node
            # mid-SAVEPOINT (eager cache drop), so the cache must be resynced
            # even if the ROLLBACK TO itself throws (e.g. a lost savepoint) —
            # otherwise the dropped rows persist and a live node goes
            # cache-invisible until restart.
            self._resync_vector_cache('absorb unwind')
            self.conn.in_batch = was_batch

    def remember(self, type: str, title: str, content: Optional[str] = None,
                 locked: bool = False,
                 emotion: float = 0, emotion_label: str = 'neutral',
                 # No `project` param: provenance is system-stamped at the
                 # write boundary and rides **extra_fields into
                 # node_metadata_kv (PROMOTED_FIELDS).
                 confidence: float = 1.0,
                 personal: Optional[str] = None,
                 personal_context: Optional[str] = None,
                 encoding_source: Optional[str] = None,
                 situation: Optional[str] = None,
                 question: Optional[str] = None,
                 event_time: Optional[str] = None,
                 thought: Optional[str] = None,
                 evolution_status: Optional[str] = None,
                 # Promoted metadata fields (stored in node_metadata_kv)
                 reasoning: Optional[str] = None,
                 their_raw_quote: Optional[str] = None,
                 my_raw_quote: Optional[str] = None,
                 # Corrections are correction_improvement-aspect edges
                 # (corrects/supersedes/reframes/...), not a field. See
                 # render_corrections() + correction_enrich() for the read path.
                 correction_pattern: Optional[str] = None,
                 source_context: Optional[str] = None,
                 confidence_rationale: Optional[str] = None,
                 alternatives: Optional[List[Dict[str, str]]] = None,
                 change_impacts: Optional[List[Dict[str, str]]] = None,
                 source_attribution: Optional[str] = None,
                 scope: Optional[str] = None,
                 connect_to: Optional[List[Any]] = None,
                 # v29 / Phase B: source_refs anchors the node to trace events.
                 # Sparse by design (1-3 refs typical). Each ref is an 8-char
                 # hex trace_event.id. Persisted via SourceRefDAL.add_source_refs
                 # → node_source_refs join table; the DAL refuses any ref
                 # that isn't an 8-char hex string.
                 source_refs: Optional[List[str]] = None,
                 # The caller's SessionContext, resolved at dispatch from the
                 # session id the call carries. Nothing here reads it today;
                 # it is how remember() knows which session is writing.
                 ctx=None,
                 **extra_fields) -> Dict[str, Any]:
        """
        Store a new memory node with semantic indexing and connections.

        Accepts ALL contract fields. Core fields go to the nodes table,
        promoted fields go to node_metadata_kv/node_enrichments, and any
        unknown fields are stored as emergent metadata in node_metadata_kv.

        Returns:
            Dict with id, type, title and embedding_queued, plus
            connect_to_result when connect_to was passed.
        """
        # Validate personal flag
        if personal and personal not in ('fixed', 'fluid', 'contextual'):
            personal = None

        # Constitution: only intentional (anchor) encoding can create locked nodes.
        # All automated sources (encoder, idle, hook) must earn permanence.
        # encoding_source convention: "category:process" e.g. "encoder:sonnet", "idle:redistribution"
        if encoding_source and not encoding_source.startswith('anchor') and locked:
            locked = False
            # The demotion is right; its silence wasn't — surface the drift
            # so a caller teaching itself to lock is visible, not swallowed.
            self._log_warning(
                'locked_demoted',
                "non-anchor locked:true demoted (encoding_source=%s, "
                "title=%r)" % (encoding_source, (title or '')[:80]),
                'write boundary')

        node_id = self._generate_id(type)
        ts = self.now()

        # A node view echoed back into a write carries read-only attachments
        # (connections, communities, _metadata, ...). None is written — edges
        # go through connect_to — so name each dropped one in the errors table
        # rather than dropping it silently. Truthiness, not key presence: an
        # empty attachment is not an attempt.
        _dropped = sorted(k for k in READ_ONLY_ATTACHMENT_KEYS
                          if extra_fields.get(k))
        if _dropped:
            self._log_error(
                'write_read_only_attachment',
                ValueError('read-only attachment(s) not written: %s — edges go '
                           'through connect_to' % ', '.join(_dropped)),
                'remember node %s' % node_id[:8])

        # ══════════════════════════════════════════════════════════════
        # v6: AUTO-ENRICHMENT — make every node rich by default
        # The brain's data was shallow because rich encoding required
        # extra effort. Now remember() fills in what it can automatically.
        # ══════════════════════════════════════════════════════════════

        # Auto-set confidence by type if caller left it at default
        # TYPE_CONFIDENCE from brain_constants defines how reliable each type tends to be
        if confidence == 1.0:  # default = unset by caller
            confidence = TYPE_CONFIDENCE.get(type, 0.70)

        # v4: Fixed personal nodes are always locked — their whole point is permanence
        if personal == 'fixed':
            locked = True

        # v5: Auto-generate content summary for tiered recall
        content_summary = self._generate_summary(title, content)

        # INSERT into nodes table
        self.conn.execute(
            '''INSERT INTO nodes
               (id, type, title, content, content_summary,
                locked, confidence, emotion, emotion_label,
                personal, personal_context, encoding_source,
                evolution_status, last_accessed, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (node_id, type, title, content, content_summary,
             1 if locked else 0, confidence, emotion, emotion_label,
             personal, personal_context, encoding_source or 'anchor',
             evolution_status, ts, ts, ts)
        )
        self._maybe_commit()

        # Encode-time voice fidelity validation (Phase B+ structural backup).
        # A/B testing of v20 surfaced a speaker-misattribution failure mode:
        # when the conversation has operator-asks-question + anchor-articulates-
        # principle shape, Sonnet's voice-attribution logic matches content to
        # field rather than speaker to field — it can land identical strings in
        # BOTH their_raw_quote and my_raw_quote. Prompt teaching alone didn't
        # catch this (failed across v19, v20.0, v20.1 on the same corpus).
        # Loud at write boundary so the error surfaces every time it happens.
        # Non-blocking: write proceeds, but the encoder errors table records the
        # violation for retrospective audit + future S2Healer-driven cleanup.
        if (their_raw_quote and my_raw_quote
                and their_raw_quote.strip() == my_raw_quote.strip()
                and len(their_raw_quote.strip()) > 0):
            self._log_error(
                'voice_fidelity_identical_strings',
                ValueError(
                    "their_raw_quote == my_raw_quote on node %s "
                    "(type=%s, title=%r). Voice fields are for different "
                    "speakers; identical strings indicate Sonnet matched "
                    "content-to-field rather than speaker-to-field. "
                    "Quote: %r" % (
                        node_id, type, (title or '')[:80],
                        their_raw_quote.strip()[:160])),
                'encode-time voice fidelity check')

        # Store all metadata via unified path — promoted, emergent, and extra fields.
        _meta_fields = {}
        # Promoted fields passed as explicit args
        for _name, _val in [
            ('reasoning', reasoning), ('their_raw_quote', their_raw_quote),
            ('my_raw_quote', my_raw_quote),
            ('correction_pattern', correction_pattern), ('source_context', source_context),
            ('confidence_rationale', confidence_rationale), ('scope', scope),
            ('source_attribution', source_attribution), ('situation', situation),
            ('question', question), ('event_time', event_time),
            ('thought', thought),
        ]:
            if _val is not None:
                _meta_fields[_name] = _val
        # Extra fields from callers (community metadata, any emergent fields)
        _meta_fields.update(extra_fields)
        if _meta_fields:
            self._store_node_metadata(node_id, _meta_fields, caller='remember')

        try:
            self._fts.upsert(node_id, title, content or '')
            self._maybe_commit()
        except Exception as e:
            self._log_error('fts5_sync_remember', e, 'syncing FTS5 for node %s' % node_id[:12])

        # Vector computation handled by the embed_queue worker — this node
        # is marked dirty and will be embedded within ~5s. S2 Heal catches
        # anything that slips through on crash.
        try:
            from . import embed_queue
            embed_queue.enqueue(node_id)
            embedding_queued = True
        except Exception as e:
            self._log_error('embed_enqueue_remember', e, 'enqueue %s' % node_id[:12])
            embedding_queued = False

        # v29 / Phase B: persist source_refs to node_source_refs join table.
        # SourceRefDAL.add_source_refs refuses (raises) any ref that isn't an
        # 8-char hex string, uses INSERT OR IGNORE (first-write-wins,
        # re-encode is safe), and preserves the encoder's write order via
        # `position`. Empty/None input is a no-op. A refused batch is logged
        # and the node stands without refs — it doesn't fail the write.
        # Edges co_anchored creates are surfaced so the dispatch handler can
        # pop them off the agent-facing payload. They are NOT traced —
        # co_anchored is noise-aspect (ruled 2026-08-04); soft/derived edges
        # stay out of trace coverage. The graph edge itself is still first-class.
        co_anchored_made = []
        if source_refs:
            try:
                self._source_refs.add_source_refs(node_id, source_refs)
            except Exception as e:
                self._log_error(
                    'source_refs_persist', e,
                    'persisting source_refs for node %s (%d refs)' % (
                        node_id[:12], len(source_refs)))

            # Step 7 / decision 15: co_anchored auto-edge. When this node's
            # source_refs overlap with any existing node's refs, write a
            # structural co_anchored edge to each sibling. The graph layer
            # is the signal — no score boost, no magnitude to guess.
            # Hidden from the encoder catalog and graph traversal via the
            # noise aspect. Sparse refs (1-3) × small cohort → negligible cost.
            try:
                graph_dal = self._graph
                siblings: set = set()
                for tid in source_refs:
                    if not isinstance(tid, str):
                        continue
                    for sibling_id in self._source_refs.get_nodes_referencing(tid):
                        if sibling_id != node_id:
                            siblings.add(sibling_id)
                for sibling_id in siblings:
                    _r = graph_dal.add_relation(
                        node_id, sibling_id, 'co_anchored',
                        description='shared episodic anchor',
                        encoding_source='dispatch:co_anchored',
                    )
                    co_anchored_made.append({
                        'src_id': node_id, 'target_id': sibling_id,
                        'relation': 'co_anchored',
                        'edge_id': (_r or {}).get('edge_id'),
                        'deltas': (_r or {}).get('deltas', [])})
            except Exception as e:
                self._log_error(
                    'co_anchored_autoedge', e,
                    'co_anchored auto-edge for node %s (%d refs)' % (
                        node_id[:12], len(source_refs)))

        # connect_to: title-resolved typed edges. When called standalone (not
        # from a batch), there are no siblings — only catalog fallback applies.
        # Inside remember_batch / brain_batch, connect_to is popped from the
        # spec BEFORE this call and processed AFTER all siblings are created.
        connect_to_result = None
        if connect_to:
            # Capture the outcome so the caller sees what linked and why an
            # edge didn't — connect_to is fail-soft, but no longer silent to
            # the caller (it used to surface only on the dashboard).
            connect_to_result = self._apply_connect_to(
                node_id, connect_to, sibling_map=None,
                encoding_source=encoding_source or 'anchor')

        # No auto-edges at store time: co_accessed write-adjacency and
        # emergent_bridge shared-neighbor closure were both retired — see
        # nodes 2f344177, 072e26d8.

        result = {
            'id': node_id,
            'type': type,
            'title': title,
            'embedding_queued': embedding_queued,
        }
        # connect_to_result is present ONLY when connect_to was passed — keeps
        # the response clean for the common no-edge case.
        if connect_to_result is not None:
            result['connect_to_result'] = connect_to_result
        # co_anchored edges this remember() materialized (src-tagged, with
        # edge_id+deltas) — the dispatch handler pops these off the payload;
        # noise-aspect, never traced.
        if co_anchored_made:
            result['co_anchored_made'] = co_anchored_made
        return result

    # ═══════════════════════════════════════════════════════════════
    # v8: revise() — Encoding IS updating existing knowledge
    # ═══════════════════════════════════════════════════════════════

    def invalidate_source_fields(self, node_id, fields, origin) -> bool:
        """THE vector-invalidation path: a node's source `fields` changed, so
        delete every vector whose text depends on them (pipeline_contract.
        vectors_affected_by) and mark the node for re-embed; the embed_queue
        worker's backfill recreates the rows as missing. Two callers, one
        mechanism: revise() for node kv/columns (INVALIDATED_BY_REVISE) and
        _edge_text_changed for edge descriptions (INVALIDATED_BY_EDGE_WRITE).

        WITHOUT the delete, VectorDAL.find_missing() skips the row (it
        exists) and the vector keeps encoding outdated text indefinitely.
        The delete is ONE call — typed DB delete + exactly-mirrored cache drop
        (CachedVectorDAL) or DB-only (plain VectorDAL); the old two-call shape
        (raw SQL + whole-node cache drop) was the 2026-07-17 healer-
        invisibility bug.

        Failure is a CORRECTNESS issue (recall serves stale embeddings until
        the next backfill cycle): logged loudly under `<origin>_vector_
        invalidate`, never raised — a failed invalidation must not fail the
        write that caused it. Returns True when the delete failed so revise()
        can surface partial success in its return dict.
        """
        failed = False
        try:
            from .pipeline_contract import vectors_affected_by
            invalidated = set()
            for field in fields:
                invalidated |= vectors_affected_by(field)
            if invalidated:
                self._vec_dal.delete_for_node(node_id, vector_types=invalidated)
        except Exception as e:
            failed = True
            self._log_error('%s_vector_invalidate' % origin, e,
                            'invalidating vectors for %s — STALE EMBEDDINGS '
                            'will be served by recall until next backfill '
                            'cycle catches up' % node_id[:8])
        # (Re)computation is the embed_queue worker's — the node is marked
        # dirty so the stale text→vector pair refreshes within ~5s.
        try:
            from . import embed_queue
            embed_queue.enqueue(node_id)
        except Exception as e:
            self._log_error('embed_enqueue_%s' % origin, e,
                            'enqueue %s' % node_id[:12])
        return failed

    def revise(self, node_id: str, content: str = None, reason: str = '',
               updates: Dict[str, Any] = None, encoding_source: str = None,
               **kwargs) -> Dict[str, Any]:
        """Update fields on an existing node — value or swap (contract.REVISE_RULE).

        Four ways to call (all equivalent):
          revise(node_id, content="new text", reason="why")
          revise(node_id, updates={"confidence": 0.9}, reason="why")
          revise(node_id, situation="When debugging", reason="adding situation")
          revise(node_id, fields={"situation": "When debugging"}, reason="why")
        `fields` (contract.FIELDS_KEY) is the tool-facing envelope for a key a
        flat tool schema does not list; contract.unpack_fields merges it in.

        A text field takes its NEW VALUE (whole field replaced) or a swap
        `{old, new}` / list of swaps that changes only what is stale, each
        `old` matching the stored value exactly once (contract.apply_swaps):
          revise(node_id, reason="why",
                 title={"old": "9.6.0", "new": "9.7.2"},
                 content=[{"old": "...", "new": "..."}])
        Non-text and bare_only fields take bare values; a swap on them is an
        error. An empty string is not an update, on any field.

        Edges ride as `connect_to`, exactly as on remember: an entry changes
        the edge this node already has to `target` (its `description` or `relation`,
        value or swap) or creates it (_apply_revise_connect_to). Edge results
        ride in `connect_to_result`, never in the node's deltas.

        Behavior contract:
        - Immutable fields ({id, created_at, locked}) are skipped with a
          warning. Other fields in the same call still process; the skipped
          field surfaces in the result dict's `warnings` list.
        - Specified fields are written; unspecified fields are PRESERVED.
        - Returns deltas in the result dict — caller (typically daemon_dispatch)
          emits a trace event with these deltas as the canonical revision
          history. There is no per-node history blob; query traces instead.
        - encoding_source is the caller's provenance for edges this revise
          creates or renames; it is never written onto the node.

        After any revision: re-embeds, re-indexes FTS5, updates timestamps.
        """
        # Merge updates from all sources
        all_updates = dict(updates or {})
        all_updates.update(kwargs)
        if content:
            all_updates['content'] = content

        # The doors unpack `fields` first; this covers callers that pass no
        # door (absorb's overrides, direct calls), so none stores it as a key.
        all_updates, err = unpack_fields(all_updates, revising=True)
        if err:
            return {'error': err + '. Nothing from this op was written.',
                    'node_id': node_id}

        # An empty value is not an update, on any field: the rule KV storage
        # already keeps (it skips empty values), held for the node's columns
        # too, so `title: ''` never blanks a title. Named in the warnings.
        emptied = sorted(f for f, v in all_updates.items()
                         if isinstance(v, str) and not v.strip())
        for f in emptied:
            all_updates.pop(f)

        # Edges are not node fields: popped before classification, routed
        # after the node writes. (Without the pop, _store_node_metadata's
        # control-field skip would drop it while the deltas reported it.)
        connect_to = all_updates.pop('connect_to', None)

        # Read-only attachments echoed from a node view are not fields either:
        # popped before classification so fields_updated and the deltas never
        # report a write that _store_node_metadata would drop. Named in the
        # warnings and the errors table.
        read_only = sorted(k for k in READ_ONLY_ATTACHMENT_KEYS
                           if k in all_updates)
        for k in read_only:
            all_updates.pop(k)
        if read_only:
            self._log_error(
                'write_read_only_attachment',
                ValueError('read-only attachment(s) not written: %s — edges go '
                           'through connect_to' % ', '.join(read_only)),
                'revise node %s' % str(node_id)[:8])

        # A swap that arrived as text is refused, never unwrapped: the brain
        # does not guess which strings were meant as objects. The refusal
        # names the doors that deliver an object.
        for f, v in all_updates.items():
            if _stringified_swap(v):
                return {'error': (
                    '%s arrived as a JSON string holding a swap, not as an '
                    '{old, new} object — a client sends an object as text '
                    'when the tool schema does not declare the field. Send '
                    'it inside `%s`, or in a revise_batch / brain_batch '
                    'revise op, where it arrives as an object, or send the '
                    'whole new value as a plain string. Nothing from this op '
                    'was written.' % (f, FIELDS_KEY)),
                    'node_id': node_id}

        # Swaps resolve against the stored value once the row is in hand;
        # bare values pass straight through. A swap on a field that is not
        # text is refused here, before any write.
        swaps = {}
        for f, v in list(all_updates.items()):
            spec = ALL_FIELDS.get(f)
            if is_swap(v) or is_swap_list(v):
                if spec is not None and (spec.get('type') != 'str'
                                         or spec.get('bare_only')):
                    return {'error': ('%s takes a bare value, not a swap '
                                      '{old, new} — swaps are for text fields'
                                      % f), 'node_id': node_id}
                swaps[f] = all_updates.pop(f)
            elif _swap_intended(v) or (isinstance(v, (dict, list))
                                       and spec is not None
                                       and spec.get('type') == 'str'):
                # Swap-shaped but malformed (an extra or missing key), or any
                # structured value on a known text field: name the shape,
                # never write it as data. Open KV keys keep their freedom to
                # hold lists and dicts that are not trying to be swaps.
                _ok, err = (validate_swaps(v, f) if v else (False, None))
                return {'error': err or ('%s must be a string or a non-empty '
                                         'list of {old, new} swaps' % f),
                        'node_id': node_id}

        if not all_updates and not swaps and not connect_to:
            return {'error': 'No updates provided' + (
                        ' — an empty value is not an update: %s'
                        % ', '.join(emptied) if emptied else '') + (
                        ' — read-only, not written: %s' % ', '.join(read_only)
                        if read_only else ''),
                    'node_id': node_id}

        # Capture the FULL field set NOW for vector invalidation. `all_updates`
        # gets mutated below (content is popped, etc.), so we need the
        # original set or the invalidation step misses fields.
        fields_changed_for_invalidation = set(all_updates.keys()) | set(swaps)

        # v29 / Phase B: source_refs is a join-table field, not a node column.
        # Pop it before the field classification so it doesn't land in
        # node_metadata_kv as an extra field. Persist via replace_source_refs
        # AFTER the existence check (node_id must exist for the FK).
        # Per the unified 2-class revise contract (decision 995ffeb1) and
        # EPISODIC-REFERENCES.md §6.2: when source_refs is PRESENT in the
        # revise payload, REPLACE the entire list. When ABSENT, preserve.
        # Use a sentinel to distinguish "key absent" from "explicit empty list".
        _SR_ABSENT = object()
        new_source_refs = all_updates.pop('source_refs', _SR_ABSENT)

        # Fetch existing node
        row = self.conn.execute(
            'SELECT id, type, title, content, archived FROM nodes WHERE id = ?',
            (node_id,)).fetchone()
        if not row:
            return {'error': 'Node not found', 'node_id': node_id}
        if row[4] == 1:
            # Writes never redirect (the canonical-pull contract) — an
            # absorbed node refuses WITH the pointer so the caller re-aims.
            surv = self._nodes.survivor_of(node_id)
            if surv:
                from .contract import absorbed_refusal
                return {'error': absorbed_refusal('revise', node_id, surv),
                        'node_id': node_id, 'survivor_id': surv}
            return {'error': 'Cannot revise archived node', 'node_id': node_id}

        existing_id, node_type, title, old_content, _ = row
        old_content = old_content or ''
        ts = self.now()

        # Resolve swaps against the stored values — all of them before any
        # write, so one bad `old` leaves the node untouched. title/content
        # come from the row; every other text field (promoted or open KV)
        # from node_metadata_kv.
        if swaps:
            stored = {'title': title or '', 'content': old_content}
            kv_keys = [f for f in swaps if f not in stored]
            if kv_keys:
                stored.update(self._meta_kv.get_fields(node_id, kv_keys))
            for f, sw in swaps.items():
                current = stored.get(f)
                # On a miss the refusal names where `old` does live — the
                # node's other text fields, read only when needed, once.
                read = {}

                def _elsewhere():
                    if not read:
                        read.update({k: v for k, v in self._meta_kv.get(node_id).items()
                                     if isinstance(v, str)},
                                    title=title or '', content=old_content)
                    return read
                if not isinstance(current, str) or not current:
                    # An empty field whose `old` lives in another field is the
                    # wrong field, not a missing value: apply_swaps names it.
                    olds = [e.get('old') for e in (sw if isinstance(sw, list) else [sw])]
                    if not any(isinstance(o, str) and o and o in text
                               for k, text in _elsewhere().items() if k != f
                               for o in olds):
                        return {'error': ('%s has no stored value to swap into — '
                                          'pass the field\'s new value instead'
                                          % f), 'node_id': node_id}
                    current = ''
                patched, edit_err = apply_swaps(current, sw, f, _elsewhere)
                if edit_err:
                    return {'error': edit_err, 'node_id': node_id,
                            'current': {f: current}}
                all_updates[f] = patched

        # ── Field classification ──
        # Top-level fields live on the nodes table (updatable via SQL).
        # Immutable fields are silently skipped with a warning.
        # No `project`: provenance lives in node_metadata_kv (PROMOTED_FIELDS)
        # and revise strips it at the write boundary.
        NODES_TABLE_FIELDS = {
            'title', 'type', 'confidence', 'emotion',
            'emotion_label', 'personal', 'personal_context',
            'evolution_status', 'encoding_source',
            'archived',  # allows revise(archived=True) for consolidation
        }
        IMMUTABLE = set(REVISE_IMMUTABLE_FIELDS)

        # ── Filter skipped fields (immutable, locked-archive) ──
        # Skipped fields don't write to nodes/KV/SQL and don't appear in
        # deltas. They surface in the return dict's `warnings` list so
        # callers can detect partial-success without parsing logs.
        # Any field that is neither a nodes column nor immutable falls through
        # to node_metadata_kv as an extra field (the generic extra-fields path,
        # same as remember()). A legacy/unknown field (e.g. the v28-dropped
        # `keywords`) is therefore stored as KV, not skipped — no longer a crash
        # (keywords is out of NODES_TABLE_FIELDS, so no SELECT/UPDATE of the gone
        # column) and intentionally not special-cased.
        skipped_fields = []  # list of (field, reason)
        writable = {}
        for field, value in all_updates.items():
            if field in IMMUTABLE:
                self._log_error('revise_immutable',
                                ValueError('Cannot revise immutable field: %s' % field),
                                'node %s attempted to revise %s' % (node_id[:8], field))
                skipped_fields.append((field, 'immutable'))
                continue
            if field == 'archived' and value:
                lock_row = self.conn.execute(
                    'SELECT locked FROM nodes WHERE id = ?',
                    (node_id,)).fetchone()
                if lock_row and lock_row[0]:
                    self._log_error('revise_archive_locked',
                                    ValueError('Cannot archive locked node'),
                                    'node %s' % node_id[:8])
                    skipped_fields.append((field, 'locked'))
                    continue
            writable[field] = value

        # ── Capture old values for delta computation (before any write) ──
        # Used by callers (typically dispatch) to emit trace events with
        # field-level history. Replaces the old _sys_revision_history blob.
        old_values = {}
        if 'content' in writable:
            old_values['content'] = old_content

        top_level_to_capture = [k for k in writable if k in NODES_TABLE_FIELDS]
        if top_level_to_capture:
            cols = ', '.join(top_level_to_capture)
            old_row = self.conn.execute(
                'SELECT %s FROM nodes WHERE id = ?' % cols, (node_id,)
            ).fetchone()
            if old_row:
                for i, k in enumerate(top_level_to_capture):
                    old_values[k] = old_row[i]

        kv_to_capture = [
            k for k in writable
            if k not in NODES_TABLE_FIELDS and k != 'content'
        ]
        if kv_to_capture:
            kv_old = self._meta_kv.get_fields(node_id, kv_to_capture)
            for k in kv_to_capture:
                old_values[k] = kv_old.get(k)  # None if not previously set

        # Content: replace with new value (history lives in trace deltas now,
        # not the legacy _sys_revision_history KV blob).
        new_content = old_content
        if 'content' in writable:
            new_content = writable.pop('content')

        # Build SQL UPDATE for all fields.
        # Always update: content, content_summary, updated_at.
        # revised_at bumps ONLY when the CLAIM changed — content or title,
        # exactly the fields the consolidation clustering embeddings are
        # built from. Metadata enrichment (question, situation, confidence,
        # evolution_status, ...) bumps updated_at alone: it cannot mint a
        # near-duplicate, so it must not re-enter the node into the
        # consolidation change set (B1 2026-08-11: healer question stamps
        # were resetting suppression on ~24 claim-unchanged nodes/day).
        claim_changed = bool(
            ('content' in old_values and old_values['content'] != new_content)
            or ('title' in writable
                and old_values.get('title') != writable['title']))
        set_parts = ['content = ?', 'content_summary = ?', 'updated_at = ?']
        params = [new_content, self._generate_summary(title, new_content), ts]
        if claim_changed:
            set_parts.append('revised_at = ?')
            params.append(ts)

        for field, value in writable.items():
            if field in NODES_TABLE_FIELDS:
                set_parts.append('%s = ?' % field)
                params.append(value)
                if field == 'title':
                    title = value  # track for re-embed

        params.append(node_id)
        self.conn.execute(
            'UPDATE nodes SET %s WHERE id = ?' % ', '.join(set_parts), params)
        self._maybe_commit()

        # Store metadata via unified path — handles promoted, emergent, situation.
        # Only writable (non-skipped) fields get persisted.
        if writable:
            self._store_node_metadata(node_id, writable, caller='revise')

        # Vector invalidation + re-embed mark. Title changes invalidate the
        # title slot too — collected via the SQL UPDATE above and added to the
        # field set. The failure flag is surfaced in the return dict so callers
        # can detect partial success.
        vector_invalidation_failed = self.invalidate_source_fields(
            node_id, fields_changed_for_invalidation, origin='revise')

        try:
            self._fts.upsert(node_id, title, new_content)
            self._maybe_commit()
        except Exception as e:
            self._log_error("fts5_sync_revise", e, "syncing FTS5 for %s" % node_id[:8])

        # ── VERIFICATION: read-back to confirm writes landed ──
        verification_failures = []

        # Verify nodes table fields
        readback = self._nodes.get_naked_node(node_id)
        if readback:
            for field in list(writable.keys()):
                if field in readback:
                    expected = writable[field]
                    actual = readback.get(field)
                    if actual != expected and str(actual) != str(expected):
                        verification_failures.append(field)
            # Content was popped from `writable` earlier; verify separately
            # (REPLACE semantic — readback must equal new_content exactly).
            # Use old_values to detect that content was actually a write target —
            # `content` named-arg path AND `updates={'content': ...}` path both
            # populate old_values['content'], so this catches either.
            if 'content' in old_values:
                actual_content = readback.get('content') or ''
                if actual_content != new_content:
                    verification_failures.append('content')

        # Situation embedding deferred to backfill — no inline verification needed

        # Vector invalidation failure surfaces here too — same severity as a
        # missed field write, since recall correctness depends on it.
        if vector_invalidation_failed:
            verification_failures.append('vector_invalidation')

        verified = len(verification_failures) == 0

        # ── Build deltas for trace event emission ──
        # Caller (typically daemon_dispatch) uses these to write a single
        # node_revised trace event capturing what changed in this call.
        deltas = []
        for field, new_val in writable.items():
            old_val = old_values.get(field)
            if old_val != new_val:
                deltas.append({
                    'field': field,
                    'old': old_val,
                    'new': new_val,
                })
        # Content was popped from `writable` earlier; check separately.
        if 'content' in old_values and old_values['content'] != new_content:
            deltas.append({
                'field': 'content',
                'old': old_values['content'],
                'new': new_content,
            })

        # Warnings surface skipped fields without requiring log parsing.
        warnings = []
        for field, _reason in skipped_fields:
            if _reason == 'immutable':
                warnings.append('immutable field skipped: %s' % field)
            elif _reason == 'locked':
                warnings.append('archive blocked (locked): %s' % field)
        for field in emptied:
            warnings.append('%s: an empty value is not an update — left as it '
                            'was' % field)
        for field in read_only:
            warnings.append('%s: a read-only attachment of the node view — not '
                            'written (edges go through connect_to)' % field)

        # fields_updated: what was actually written. Excludes skipped fields.
        # Includes 'content' if it was passed (popped from writable earlier).
        fields_updated = list(writable.keys())
        if 'content' in old_values:
            fields_updated.append('content')

        # v29 / Phase B: persist source_refs with field-level REPLACE semantics
        # (decision 995ffeb1 — unified revise contract; §6.2). When the key was
        # absent in the payload, preserve existing refs (no-op). When present —
        # even as an empty list — replace the entire list.
        source_refs_replaced = None  # None=untouched, int=count after replace
        if new_source_refs is not _SR_ABSENT:
            try:
                source_refs_replaced = self._source_refs.replace_source_refs(
                    node_id, new_source_refs or [])
                fields_updated.append('source_refs')
                deltas.append({
                    'field': 'source_refs',
                    'op': 'replace',
                    'count_after': source_refs_replaced,
                })
            except Exception as e:
                self._log_error(
                    'source_refs_persist_revise', e,
                    'replacing source_refs on revise %s' % node_id[:12])

            # Step 7: refresh co_anchored cohort after REPLACE. New refs may
            # bring new siblings; old siblings whose shared refs are now gone
            # become stale edges that S2Healer archives (§10.6 Responsibility 2).
            # add_relation is idempotent — existing edges are field-preserving
            # no-ops.
            try:
                graph_dal = self._graph
                siblings: set = set()
                for tid in (new_source_refs or []):
                    if not isinstance(tid, str):
                        continue
                    for sibling_id in self._source_refs.get_nodes_referencing(tid):
                        if sibling_id != node_id:
                            siblings.add(sibling_id)
                for sibling_id in siblings:
                    graph_dal.add_relation(
                        node_id, sibling_id, 'co_anchored',
                        description='shared episodic anchor',
                        encoding_source='dispatch:co_anchored',
                    )
            except Exception as e:
                self._log_error(
                    'co_anchored_autoedge_revise', e,
                    'co_anchored auto-edge on revise %s' % node_id[:12])

        # Edges last: node writes are committed truth by now, and connect_to is
        # fail-soft per entry (a bad edge never undoes a good field write).
        # Edge warnings stay on the edge side (connect_to_result and the edge
        # manifest rows) — a node warning would mint a node_revised trace for
        # a revise that changed no node field.
        connect_to_result = None
        if connect_to:
            connect_to_result = self._apply_revise_connect_to(
                node_id, connect_to, encoding_source=encoding_source)

        result = {
            'id': node_id,
            'type': writable.get('type', node_type),
            'title': title,
            # None = metadata-only revise; the stored revised_at was not touched.
            'revised_at': ts if claim_changed else None,
            'content_length': len(new_content),
            'fields_updated': fields_updated,
            'deltas': deltas,
            'warnings': warnings,
            'verified': verified,
            'verification_failures': verification_failures if not verified else [],
            'pending_resolved': 0,
            'source_refs_replaced': source_refs_replaced,
        }
        # Present ONLY when connect_to was passed — same contract as remember.
        if connect_to_result is not None:
            result['connect_to_result'] = connect_to_result
        return result

    # ═══════════════════════════════════════════════════════════════
    # connect_to resolution — sibling-aware, sequencing-agnostic
    # ═══════════════════════════════════════════════════════════════

    def _resolve_connect_to_entry(self, entry, sibling_map=None,
                                  exclude_self=None, exclude_ids=None):
        """Resolve a connect_to entry to (target_id, relation_pairs, reason).

        sibling_map: {lowercased_title: node_id} for nodes created in the
                     same batch. Sibling matching is CASE-INSENSITIVE —
                     keys are lowered when the map is built (see
                     remember_batch sibling_map construction) AND lowered
                     again at lookup. Sibling exact-match (case-insensitive)
                     wins over the catalog-title pass — NEW wins on title
                     collision. If you mean an existing catalog node, use
                     revise on its id, not duplicate-title remember.
                     (Catalog matching is _match_catalog_title — token-exact
                     or bounded near-title, deterministic, no vectors.)
        exclude_self: source node_id; resolution to this id is treated as a
                      self-reference and rejected.

        Returns (target_id, [(relation, description), ...], None) on success,
        or (None, [], reason) on any failure — `reason` is a short caller-facing
        string so the failure can surface in the write response, not just the
        dashboard. All failures also log loudly via _log_error — no silent skips.
        """
        # Parse entry shape (string or dict)
        if isinstance(entry, str):
            title_query = entry
            relation_pairs = [('related', '')]
        elif isinstance(entry, dict):
            title_query = connect_to_target(entry)
            if connect_to_uses_old_key(entry):
                self._log_warning(
                    'connect_to_old_key',
                    'connect_to entry names its target by the old key '
                    '`title`: %r' % str(title_query)[:90])
            if not title_query:
                reason = "connect_to entry missing 'target' field"
                self._log_error(
                    'connect_to_invalid', ValueError(reason),
                    'entry=%s' % str(entry)[:200])
                return None, [], reason
            if not isinstance(title_query, str):
                # A non-empty non-string title (int, list, ...) clears the
                # falsy guard above but would crash the .strip()/.lower()/regex
                # calls downstream — that AttributeError escapes _apply_connect_to
                # and rolls back the whole batch. Reject loudly instead.
                reason = ("connect_to entry 'target' must be a string, got %s"
                          % type(title_query).__name__)
                self._log_error(
                    'connect_to_invalid', TypeError(reason),
                    'entry=%s' % str(entry)[:200])
                return None, [], reason
            if isinstance(entry.get('relations'), list):
                relation_pairs = []
                for r in entry['relations']:
                    if not isinstance(r, dict):
                        continue
                    rel = r.get('relation', 'related')
                    desc = connect_to_description(r) or ''
                    relation_pairs.append((rel, desc))
                if not relation_pairs:
                    reason = "connect_to relations array is empty or malformed"
                    self._log_error(
                        'connect_to_invalid', ValueError(reason),
                        'entry=%s' % str(entry)[:200])
                    return None, [], reason
            else:
                rel = entry.get('relation', 'related')
                desc = connect_to_description(entry) or ''
                relation_pairs = [(rel, desc)]
        else:
            reason = ("connect_to entry must be str or dict, got %s"
                      % type(entry).__name__)
            self._log_error(
                'connect_to_invalid', TypeError(reason),
                'entry=%s' % str(entry)[:200])
            return None, [], reason

        target_id = None

        # Pass 0: hex-shaped target ⇒ it IS an id, never a title. Since s1e
        # v34 (Option D) ids are the expected form for catalog targets, so a
        # miss or ambiguous prefix is a miscopy — fail loudly and TERMINALLY
        # rather than fall into title matching, where a bad hex string can
        # match a node whose title merely quotes an id (silently-wrong edge;
        # a wrong edge outlives a missing one). One exception before the
        # terminal decision: a sibling created in this batch whose exact
        # title is hex-shaped still resolves as a sibling (Pass 1 semantics —
        # sibling ids don't exist at authoring time).
        if re.fullmatch(r'[0-9a-fA-F]{8,}', title_query.strip()):
            resolved, reason = self._resolve_id_prefix(title_query)
            if resolved:
                target_id = resolved
            elif sibling_map and title_query.lower() in sibling_map:
                target_id = sibling_map[title_query.lower()]
            else:
                self._log_error(
                    'connect_to_bad_id', ValueError(reason),
                    'target=%s' % title_query[:80])
                return None, [], reason

        # Pass 1: sibling map (NEW wins on title collision)
        if not target_id and sibling_map:
            target_id = sibling_map.get(title_query.lower())

        # Pass 2: catalog title — deterministic token matching only (exact =
        # distance 0 at any length; near = distance ≤ NEAR_TITLE_MAX_OPS behind
        # floor/uniqueness/margin gates). Vectors are deliberately absent from
        # the write path (decision 2026-07-30): a wrong edge outlives a missing
        # one. find_node_by_title (interactive fuzzy search) is NOT called here.
        ambiguous_reason = None
        if not target_id:
            try:
                target_id, ambiguous_reason = self._match_catalog_title(
                    title_query, exclude_ids=exclude_ids)
            except Exception as e:
                reason = "title lookup failed: %s" % str(e)[:120]
                self._log_error(
                    'connect_to_failed', e,
                    'catalog title match for %r' % title_query[:80])
                return None, [], reason

        # Self-reference guard
        if target_id and exclude_self and target_id == exclude_self:
            reason = "connect_to would create self-edge"
            self._log_error(
                'connect_to_self', ValueError(reason),
                'node=%s title=%s' % (exclude_self[:8], title_query[:80]))
            return None, [], reason

        # Unresolved — neither sibling nor catalog matched
        if not target_id:
            reason = ambiguous_reason or (
                "title %r resolved to no node (pass the node id, the "
                "sibling's exact title, or the exact catalog title)"
                % title_query[:80])
            self._log_error(
                'connect_to_unresolved',
                ValueError("connect_to title resolved to nothing"),
                'title=%s%s' % (title_query[:80],
                                ' | %s' % ambiguous_reason if ambiguous_reason
                                else ''))
            return None, [], reason

        return target_id, relation_pairs, None

    def _title_candidate_rows(self, tokens, limit=None):
        """The ONE lexical candidate door for title→node matching: probe the
        FTS5 index TITLE-SCOPED (`title:"tok"*` per probe, OR'd, prefix-
        tolerant) and hydrate (id, title) for the live matches. Shared by
        _match_catalog_title (write path) and find_node_by_title's lexical
        pass (interactive) — neither scans the nodes table.

        The column scope is load-bearing for the pigeonhole recall guarantee:
        an unscoped MATCH ranks title-hits against every node that merely
        MENTIONS a probe token in content, and bm25 ORDER BY + LIMIT then
        makes the pool a relevance cut that can rank a qualifying title out
        (bug 69c2cbab #1). Title-only matching makes the pool exactly "titles
        carrying a probe token" — but recall is still only guaranteed when
        that pool fits `limit`, so saturation is SURFACED, never assumed away.

        Returns (rows, saturated): rows are live (id, title) pairs;
        saturated=True means the FTS pool hit `limit` and recall can no
        longer be assumed — the write path must refuse rather than match.
        Empty rows when FTS has nothing (or is unavailable — Fts5DAL.search
        already logs that loud).

        limit=None (the default) uses contract.TITLE_CANDIDATE_POOL_LIMIT —
        the bound that decides how often a legitimate connect_to is refused
        as saturated. Callers pass an explicit limit only to test the
        saturation branch."""
        from .contract import TITLE_CANDIDATE_POOL_LIMIT
        if limit is None:
            limit = TITLE_CANDIDATE_POOL_LIMIT
        ids = self._fts.search(' '.join(tokens), limit=limit, prefix=True,
                               column='title', min_token_len=1)
        if not ids:
            return [], False
        saturated = len(ids) >= limit
        ph = ','.join('?' * len(ids))
        rows = self.conn.execute(
            'SELECT id, title FROM nodes WHERE archived = 0 '
            'AND id IN (%s)' % ph, ids).fetchall()
        return rows, saturated

    def _match_catalog_title(self, title_query, exclude_ids=None):
        """Deterministic catalog-title match for connect_to (write path).

        Token-sequence Levenshtein against live titles:
          • distance 0 — exact normalized match, accepted at any length
            (a verbatim copy is unambiguous intent);
          • distance 1..NEAR_TITLE_MAX_OPS — the bounded-tolerance rung:
            requires ≥ NEAR_TITLE_MIN_TOKENS distinct query tokens, a UNIQUE
            best candidate, and the runner-up ≥ NEAR_TITLE_MARGIN ops further
            out. Acceptance logs loud (connect_to_near_title) with both
            strings — tolerance is visible, never silent.
        Ambiguity (two candidates tied at best, duplicates at distance 0, or
        a photo-finish inside the margin) REFUSES rather than picks.

        Candidate generation rides the FTS5 title index (_title_candidate_rows
        — the ONE indexed door every lexical title consumer shares): a title
        within K ops of the query misses at most K query tokens, so probing
        the K+1 longest tokens (pigeonhole) keeps recall while the index keeps
        it off a table scan. FTS rows are written synchronously in remember()
        and revise(), so just-created and just-renamed nodes are immediately
        matchable — none of the async-vector blindness the old fuzzy path had.

        Returns (target_id, None) on acceptance, (None, reason) on an
        ambiguity refusal (caller surfaces the reason), (None, None) when
        nothing is within the bar.

        exclude_ids: node ids removed from candidacy entirely (batch-level
        connect_to passes its own creations — a sibling sharing the target's
        title must not tie the true catalog node into an ambiguity refusal).
        """
        from .contract import (NEAR_TITLE_MAX_OPS, NEAR_TITLE_MIN_TOKENS,
                               NEAR_TITLE_MARGIN)
        q = _title_tokens(title_query)
        if not q:
            return None, None
        # Margin needs to see runner-up distances up to best+MARGIN, so the
        # DP cap extends past MAX_OPS; beyond the cap everything is "too far".
        cap = NEAR_TITLE_MAX_OPS + NEAR_TITLE_MARGIN - 1
        # Pigeonhole must cover the CAP, not just the acceptance bar: the
        # margin gate reasons about runners out to distance `cap`, and a
        # runner missing every probe is invisible — which silently turns a
        # photo-finish refusal into an acceptance (bug 69c2cbab #2). cap+1
        # probes ⇒ any title within `cap` ops shares at least one.
        probes = sorted(set(q), key=lambda t: (-len(t), t))
        probes = probes[:cap + 1]
        rows, saturated = self._title_candidate_rows(probes)
        if saturated:
            # The FTS pool hit its limit: candidates may have been cut, so
            # neither uniqueness nor margin can be trusted. Refuse loudly —
            # same posture as ambiguity (never guess at a write boundary).
            reason = ("title candidate pool saturated — recall guarantee "
                      "cannot hold; pass the node id")
            self._log_warning(
                'connect_to_pool_saturated',
                'probe pool hit limit for query %r — refusing'
                % title_query[:90])
            return None, reason
        scored = []
        skip = exclude_ids or ()
        for nid, title in rows:
            if nid in skip:
                continue
            d = _token_edit_distance(q, _title_tokens(title), cap)
            if d <= cap:
                scored.append((d, nid, title))
        within = [s for s in scored if s[0] <= NEAR_TITLE_MAX_OPS]
        if not within:
            return None, None
        within.sort(key=lambda s: s[0])
        best_d, best_id, best_title = within[0]
        tied = [s for s in within if s[0] == best_d]
        if len(tied) > 1:
            reason = ("title matches %d nodes at distance %d (%s) — refusing "
                      "ambiguous match; pass the node id"
                      % (len(tied), best_d,
                         ', '.join(s[1][:8] for s in tied[:4])))
            return None, reason
        if best_d == 0:
            return best_id, None
        # Near acceptance gates: length floor + margin to the runner-up.
        if len(set(q)) < NEAR_TITLE_MIN_TOKENS:
            return None, None
        runner_d = min((s[0] for s in scored if s[1] != best_id),
                       default=cap + 1)
        if runner_d - best_d < NEAR_TITLE_MARGIN:
            reason = ("near-title photo-finish (best=%d, runner-up=%d) — "
                      "refusing; pass the node id" % (best_d, runner_d))
            return None, reason
        self._log_warning(
            'connect_to_near_title',
            'near-title accepted at distance %d: query=%r matched=%r (%s)'
            % (best_d, title_query[:90], best_title[:90], best_id[:8]))
        return best_id, None

    def _resolve_id_prefix(self, prefix):
        """An 8+-hex id prefix → the one node id it names, or (None, reason).
        Shared by connect_to resolution on remember (Pass 0) and revise."""
        prefix = prefix.strip().lower()
        try:
            ids = self._nodes.resolve_id_prefix(prefix)
        except Exception as e:
            self._log_error('connect_to_id_lookup_failed', e,
                            'id-prefix lookup for %r' % prefix[:80])
            return None, "id-prefix lookup failed: %s" % str(e)[:120]
        if len(ids) == 1:
            return ids[0], None
        if len(ids) > 1:
            return None, ("id prefix %r is ambiguous (matches multiple nodes) "
                          "— pass more characters of the id" % prefix[:16])
        return None, "id %r matches no node — check the copied id" % prefix[:16]

    def _apply_revise_connect_to(self, node_id, connect_to_spec,
                                 encoding_source=None):
        """connect_to on REVISE (contract.REVISE_RULE): each entry addresses
        the edge between this node and `target` — an existing node id, never a
        sibling title. The edge is identified by (this node, target, relation),
        either direction: a bare `relation` names the row (optional when the
        pair carries exactly one), a swap renames it in place (revise_edge),
        `description` as a value replaces it and as a swap patches it.
        No such relation → created from this node to `target` with a bare
        `description` (>=30 chars); the claim keeps that order whatever the pair's
        first claim was, so either voice reads right.

        Unlike remember's connect_to there is no default relation (absent
        means "the one relation the pair has") and no title matching. `why` is
        accepted as the legacy alias of `description`, as on remember.

        Fail-soft per entry like remember's connect_to, never silent: every
        failure carries a caller-facing reason and logs the real exception.
        Returns {'created': [...], 'revised': [...], 'failed': [{target,
        reason, current?}], 'warnings': [...]}, `current` only on a refused
        description swap (contract.REFUSAL_POINTER_KEYS); created/revised
        records are {src_id, target_id, relation, edge_id, deltas, warnings}
        in the claim's own order — the shape dispatch's edge manifest reads.
        """
        out = {'created': [], 'revised': [], 'failed': [], 'warnings': []}
        if isinstance(connect_to_spec, str):
            try:
                parsed = json.loads(connect_to_spec)
            except ValueError:
                parsed = None
            connect_to_spec = parsed if isinstance(parsed, list) else connect_to_spec
        if not isinstance(connect_to_spec, list):
            reason = ("connect_to must be a list, got %s"
                      % type(connect_to_spec).__name__)
            self._log_error('connect_to_invalid', TypeError(reason),
                            'revise src=%s' % node_id[:8])
            out['failed'].append({'target': str(connect_to_spec)[:80],
                                  'reason': reason})
            return out

        def fail(label, reason, exc=None):
            # `reason` is the prose, or a refusal dict carrying pointers
            # beside it (a refused description swap's `current`).
            refusal = reason if isinstance(reason, dict) else {'error': reason}
            self._log_error('revise_connect_to_failed',
                            exc or ValueError(refusal['error']),
                            'src=%s target=%s' % (node_id[:8], label))
            out['failed'].append({'target': label, 'reason': refusal['error'],
                                  **refusal_pointers(refusal)})

        gdal = self._graph
        for entry in connect_to_spec:
            target_q = connect_to_target(entry)
            if connect_to_uses_old_key(entry):
                self._log_warning(
                    'connect_to_old_key',
                    'revise connect_to entry names its target by the old '
                    'key `title`: %r' % str(target_q)[:90])
            label = str(target_q or entry)[:80]
            if not isinstance(entry, dict):
                fail(label, 'connect_to entry must be an object {target, relation, description}')
                continue
            if (not isinstance(target_q, str)
                    or not re.fullmatch(r'[0-9a-fA-F]{8,}', target_q.strip())):
                fail(label, "on revise `target` must be an existing node's id "
                            "(8+ hex chars) — sibling titles only exist on remember")
                continue
            target_id, reason = self._resolve_id_prefix(target_q)
            if not target_id:
                fail(label, reason)
                continue
            if target_id == node_id:
                fail(label, 'a node cannot connect_to itself')
                continue

            if isinstance(entry.get('relations'), list):
                pairs = [(r.get('relation'), connect_to_description(r))
                         for r in entry['relations'] if isinstance(r, dict)]
                if not pairs:
                    fail(label, 'connect_to relations array is empty or malformed')
                    continue
            else:
                pairs = [(entry.get('relation'), connect_to_description(entry))]

            edge_id = gdal.get_edge_id(node_id, target_id)
            for rel, why in pairs:
                # Re-read per pair: a rename in this loop changes the row set.
                active = ({r['relation']: r for r in gdal.get_relations(edge_id)}
                          if edge_id else {})
                kind, res, exc = self._revise_one_edge_relation(
                    node_id, target_id, edge_id, active, rel, why, encoding_source)
                if kind == 'failed':
                    fail(label, res, exc)
                    continue
                out[kind].append(res)
                out['warnings'].extend(res.get('warnings') or [])
                edge_id = edge_id or res.get('edge_id')
        return out

    def _revise_one_edge_relation(self, node_id, target_id, edge_id, active,
                                  rel, why, encoding_source):
        """One (relation, why) pair of a revise connect_to entry. Returns
        ('created' | 'revised', record, None) or ('failed', reason, exc|None);
        a refused description swap's reason is the refusal dict {error,
        current}. The record's src/target are the edge's STORED endpoints."""
        # A swap that arrived as text is refused, as on the node's own fields
        # (revise): stored, its JSON would become the description or relation.
        for name, value in (('relation', rel), ('description', why)):
            if _stringified_swap(value):
                return 'failed', ('%s arrived as a JSON string holding a swap, '
                                  'not as an {old, new} object — send it as an '
                                  'object, or the whole new value as a plain '
                                  'string. This edge was left unchanged.'
                                  % name), None
        # relation: absent | bare name | swap (rename)
        rel_swap = None
        if is_swap_list(rel):
            if len(rel) != 1:
                return 'failed', 'a relation takes one swap {old, new}, not a list', None
            rel_swap = rel[0]
        elif is_swap(rel):
            rel_swap = rel
        elif rel is not None and not isinstance(rel, str):
            return 'failed', 'relation must be a string or a swap {old, new}', None

        new_relation = None
        if rel_swap:
            if not isinstance(rel_swap.get('old'), str) or not isinstance(rel_swap.get('new'), str):
                return 'failed', 'relation swap needs string old and new', None
            row = rel_swap['old']
            new_relation = rel_swap['new']
            if row not in active:
                return 'failed', ("edge to %s has no active relation %r to rename "
                                  "(has: %s)" % (target_id[:8], row, sorted(active))), None
        elif rel:
            row = rel if rel in active else None
        elif len(active) == 1:
            row = next(iter(active))
        elif not active:
            return 'failed', ("no edge between %s and %s — pass `relation` (and "
                              "a bare `description` of 30+ chars) to create one"
                              % (node_id[:8], target_id[:8])), None
        else:
            return 'failed', ("edge to %s carries %d relations (%s) — pass "
                              "`relation` to say which"
                              % (target_id[:8], len(active), ', '.join(sorted(active)))), None

        # A bare description is a whole one: the same >=30 floor whether it
        # creates the edge or replaces a description (a swap patches a span
        # and is judged by its match, not its length).
        if isinstance(why, str) and len(why.strip()) < 30:
            return 'failed', ("a bare `description` on the %r edge to %s needs 30+ chars — "
                              "what the edge MEANS; a swap {old, new} patches a "
                              "span" % (rel or row, target_id[:8])), None

        if row is None:
            # Create, in this node's order: the claim keeps it whichever way
            # the pair's first claim was written.
            if not isinstance(why, str):
                return 'failed', ("creating the %r edge to %s needs a bare `description` "
                                  "of 30+ chars — what the edge MEANS"
                                  % (rel, target_id[:8])), None
            try:
                edge_res = self.connect_typed(node_id, target_id, relation=rel,
                                              weight=0.6, description=why,
                                              encoding_source=encoding_source)
            except Exception as e:
                return 'failed', 'connect_typed failed: %s' % str(e)[:160], e
            record = {'src_id': node_id, 'target_id': target_id, 'relation': rel,
                      'edge_id': (edge_res or {}).get('edge_id'),
                      'deltas': (edge_res or {}).get('deltas', []),
                      'warnings': []}
            return 'created', record, None

        # Revise the existing row: description value-or-swap, optional rename.
        description = None
        if isinstance(why, str):
            description = why
        elif is_swap(why) or is_swap_list(why):
            stored = active[row].get('description') or ''
            description, err = apply_swaps(stored, why, 'description', unit='edge')
            if err:
                return 'failed', {'error': err,
                                  'current': {'description': stored}}, None
        elif why is not None:
            return 'failed', 'description must be a string or a swap {old, new}', None
        if description is None and new_relation is None:
            return 'failed', ("the %r edge to %s changes nothing — pass a new "
                              "`description`, a swap into it, or a relation swap"
                              % (row, target_id[:8])), None
        res = self.revise_edge(node_id, target_id, row, new_relation=new_relation,
                               description=description,
                               encoding_source=encoding_source)
        if not res.get('ok'):
            return 'failed', res.get('error', 'revise_edge failed'), None
        return 'revised', {'src_id': res['source_id'], 'target_id': res['target_id'],
                           'relation': res['relation'], 'edge_id': res['edge_id'],
                           'deltas': res.get('deltas', []), 'warnings': []}, None

    def _apply_connect_to(self, src_id, connect_to_spec, sibling_map=None,
                          encoding_source=None, exclude_ids=None):
        """Resolve and create edges for each connect_to entry from src_id.

        Each entry is independent — failures on one don't affect others.
        All failures log loudly; the function never raises and never blocks
        the surrounding write path.

        exclude_ids: batch-created node ids, removed from CATALOG candidacy
        in the title match (sibling resolution stays Pass 1, exact-only, by
        design). Without this, a just-created sibling sitting 1-2 ops from a
        real catalog target ties it into an ambiguity refusal — or wins the
        edge outright (bug 69c2cbab #3). Batch callers pass their created
        set; the single-node remember() path has no siblings to exclude.

        encoding_source: provenance for the edges minted here — the creating
        node's resolved source (e.g. 'anchor' for a direct-MCP remember,
        'encoder:sonnet' for the Scribe). These edges hang off a node created
        in the same write, so they share its provenance. Passed through to
        connect_typed; None means preserve-existing (the legacy behavior, which
        on a fresh edge fell to the '' default — the bug this closes). The MCP
        proxy can't reach these edges (they're minted inside remember()), so the
        source is threaded here from the caller rather than stamped at the proxy.

        Returns a dict:
            {'created': [{'src_id', 'target_id', 'relation', 'edge_id', 'deltas'}, ...],
             'failed':  [{'title', 'reason'}, ...]}
        Each `created` entry is self-contained (src_id tagged here, so callers
        don't re-derive it) and carries the edge_id + add_relation deltas so the
        caller can emit a directional `edge_relation_revised` trace (connect_to
        edges used to be the one edge path that emitted nothing). Every failure
        carries a caller-facing `reason` so the write response can show WHY an
        edge didn't form — not just a count, and not only on the dashboard.
        Callers derive counts via len(). The encoder still gets its
        "tried N, failed M" signal from these lengths.

        Input flexibility: a JSON-string that parses to a list is accepted
        (callers sometimes stringify the array) — the coercion is visible via
        a `failed` entry only when the string does NOT parse to a list.
        """
        created = []
        failed = []
        if not connect_to_spec:
            return {'created': created, 'failed': failed}

        from .contract import normalize_connect_to
        entries = normalize_connect_to(connect_to_spec)
        if entries is None:
            if isinstance(connect_to_spec, str):
                reason = ("connect_to must be a list (or a JSON string that "
                          "parses to one); got an unparseable str")
            else:
                reason = ("connect_to must be a list, got %s"
                          % type(connect_to_spec).__name__)
            self._log_error('connect_to_invalid', TypeError(reason),
                            'src=%s' % src_id[:8])
            failed.append({'title': str(connect_to_spec)[:80], 'reason': reason})
            return {'created': created, 'failed': failed}

        for entry in entries:
            title_query = connect_to_target(entry) or entry
            target_id, relation_pairs, reason = self._resolve_connect_to_entry(
                entry, sibling_map=sibling_map, exclude_self=src_id,
                exclude_ids=exclude_ids)
            if target_id is None:
                # Resolution failed — already logged via _log_error. Surface
                # the reason so the caller can self-correct in the moment.
                failed.append({'title': str(title_query)[:80], 'reason': reason})
                continue
            for rel, desc in relation_pairs:
                try:
                    edge_res = self.connect_typed(src_id, target_id, relation=rel,
                                       weight=0.6, description=desc,
                                       encoding_source=encoding_source)
                    created.append({'src_id': src_id, 'target_id': target_id,
                                    'relation': rel,
                                    'edge_id': (edge_res or {}).get('edge_id'),
                                    'deltas': (edge_res or {}).get('deltas', [])})
                except Exception as e:
                    reason = "connect_typed failed: %s" % str(e)[:120]
                    failed.append({'title': str(title_query)[:80], 'reason': reason})
                    self._log_error(
                        'connect_to_failed', e,
                        'src=%s target=%s rel=%s' % (
                            src_id[:8], target_id[:8], rel))
        return {'created': created, 'failed': failed}

    def remember_batch(self, nodes: List[Dict],
                        connect_to: Optional[List[str]] = None,
                        ctx=None) -> Dict[str, Any]:
        """Create multiple nodes in one call. Each node uses the same contract as remember().

        Args:
            nodes: List of dicts, each with the same fields remember() accepts
                   (type, title, content, situation, reasoning, etc.)
            connect_to: List of catalog targets (node id or exact/near title —
                        deterministic ladder, no vectors) to connect all new
                        nodes to

        Returns:
            {nodes_created, results: [{id, type, title, embedding_queued}], connections_created}

        Note: pairwise auto-connect of new nodes in the batch was removed
        (2026-05-24). It used to write `related_to` edges with empty
        descriptions between every pair of created nodes — semantic noise
        that confused recall and accumulated as `related_to`-with-no-why
        pollution. If callers want intra-batch edges, pass them
        explicitly via per-node `connect_to`. Episodically-bonded nodes
        (same encoding event) now share that lineage via source_refs +
        the future `co_anchored` edge (decision 15 in
        EPISODIC-REFERENCES.md), not a synthetic relation.
        """
        results = []
        created_ids = []
        connections_created = 0

        # Pass 1: create all nodes. Pop per-node connect_to BEFORE calling
        # remember() so it doesn't fire there with an empty sibling_map —
        # we'll process them all together once siblings exist (Pass 2).
        # Build sibling_map: lowercased title → node_id for sequencing-
        # agnostic resolution (B can connect_to A even if declared first).
        sibling_map = {}
        deferred_connects = []  # [(node_id, ct_spec)]
        # node_id → the node's resolved encoding_source, so connect_to edges
        # (resolved in pass 2, below) inherit the provenance of the node they
        # hang off — 'anchor' for direct-MCP, 'encoder:sonnet' for the Scribe.
        node_sources = {}
        for spec in nodes:
            if isinstance(spec, dict):
                ct_spec = spec.pop('connect_to', None)
            else:
                ct_spec = None
            result = self.remember(**spec, ctx=ctx)
            results.append(result)
            if result.get('id'):
                created_ids.append(result['id'])
                node_sources[result['id']] = spec.get('encoding_source') or 'anchor'
                title = (spec.get('title') or '').lower()
                if title:
                    sibling_map[title] = result['id']
                if ct_spec:
                    deferred_connects.append((result['id'], ct_spec))

        # Pass 2: resolve per-node connect_to with full sibling_map populated.
        # Collect each made edge (with its src_id) so the dispatch handler can
        # emit a directional edge_relation_revised trace per edge.
        connect_to_failed = []  # [{title, reason}] across all nodes
        connect_to_made = []    # [{src_id, target_id, relation, edge_id, deltas}]
        created_set = set(created_ids)
        for src_id, ct_spec in deferred_connects:
            r = self._apply_connect_to(
                src_id, ct_spec, sibling_map=sibling_map,
                encoding_source=node_sources.get(src_id, 'anchor'),
                exclude_ids=created_set)
            connections_created += len(r['created'])
            connect_to_made.extend(r['created'])  # entries are src-tagged by _apply_connect_to
            connect_to_failed.extend(r['failed'])

        # Batch-level connect_to: same edge from EVERY created node to one
        # catalog target. Resolution rides the SAME deterministic ladder as
        # per-node entries (_resolve_connect_to_entry: id → catalog exact/near;
        # sibling_map deliberately absent — siblings are excluded here by
        # contract). Failures are loud and reported, never silently skipped
        # (the pre-2026-07-30 fuzzy version continued without a word).
        if connect_to:
            for entry in connect_to:
                target_id, relation_pairs, fail_reason = (
                    self._resolve_connect_to_entry(entry, sibling_map=None,
                                                   exclude_ids=created_set))
                if not target_id:
                    connect_to_failed.append(
                        {'title': str(connect_to_target(entry))[:80],
                         'reason': fail_reason})
                    continue
                if target_id in created_set:
                    continue  # sibling — batch-level targets catalog only

                for node_id in created_ids:
                    for rel, desc in relation_pairs:
                        try:
                            edge_res = self.connect_typed(node_id, target_id, relation=rel,
                                              weight=0.6, description=desc,
                                              encoding_source=node_sources.get(node_id, 'anchor'))
                            connections_created += 1
                            connect_to_made.append({
                                'src_id': node_id, 'target_id': target_id,
                                'relation': rel,
                                'edge_id': (edge_res or {}).get('edge_id'),
                                'deltas': (edge_res or {}).get('deltas', [])})
                        except Exception as _e:
                            self._log_error('batch_connect_to', _e, 'connecting %s → %s' % (node_id[:8], target_id[:8]))

        return {
            'nodes_created': len(created_ids),
            'results': results,
            'connections_created': connections_created,
            'connect_to_made': connect_to_made,
            'connect_to_failures': len(connect_to_failed),
            'connect_to_failed': connect_to_failed,
        }

    def revise_batch(self, revisions: List[Dict]) -> Dict[str, Any]:
        """Revise multiple nodes in one call. Each revision uses the same
        contract as revise() — per-field replace, immutable fields skipped
        with warning, deltas captured for trace history.

        Args:
            revisions: List of dicts, each with:
                - node_id (required): ID of node to revise
                - reason (required): why this revision
                - content, situation, reasoning, etc.: any revisable field

        Example:
            revise_batch(revisions=[
                {"node_id": "abc123", "content": "Judge now runs in daemon", "reason": "architecture changed"},
                {"node_id": "def456", "situation": "When debugging daemon connectivity", "reason": "adding situation"},
                {"node_id": "ghi789", "reasoning": "updated — encoder uses node catalog", "reason": "encoder v3.2"},
            ])

        Returns:
            {revised: count,
             results: [{node_id, status, error?, deltas?, warnings?}]}

            Per-result `deltas` and `warnings` mirror what revise() returns —
            callers (typically dispatch) use them to emit one trace event per
            revised node.
        """
        results = []
        revised_count = 0

        for _idx, spec in enumerate(revisions):
            node_id = spec.get('node_id')
            if not node_id:
                results.append({'op': 'revise', 'index': _idx, 'ok': False,
                                'error': 'missing node_id',
                                'status': 'skipped'})
                continue

            reason = spec.get('reason', '')
            content = spec.get('content')

            # Extract field updates. Excludes node_id/reason/content (handled
            # explicitly) AND the dispatch control keys — mirroring standalone
            # revise()'s DISPATCH_KEYS reservation. `encoding_source` in
            # particular is the immutable CREATOR mark, not a revisable field:
            # the dispatch layer injects it for trace attribution, but a revise
            # must never write it onto the node, or a bulk-edit would silently
            # relabel who created the memory. (chain_id/session_id/_caller_session
            # are trace/identity control, never node fields.)
            _REVISE_CONTROL = ('node_id', 'reason', 'content', 'encoding_source',
                               'chain_id', 'session_id', '_caller_session')
            updates = {k: v for k, v in spec.items()
                       if k not in _REVISE_CONTROL and v is not None}

            try:
                result = self.revise(node_id=node_id, content=content,
                                     reason=reason, updates=updates,
                                     encoding_source=spec.get('encoding_source'))
                # `ok` rides alongside `status` so log_failed_batch_ops (which
                # scans per-op rows for ok=False on every dispatched command)
                # sees per-row failures — without it a revise_batch whose every
                # row failed (e.g. a swap whose old was not found) logged nothing.
                if result.get('error'):
                    results.append({'op': 'revise', 'index': _idx,
                                    'node_id': node_id, 'ok': False,
                                    'status': 'error', 'error': result['error'],
                                    **refusal_pointers(result)})
                else:
                    row = {
                        'op': 'revise',
                        'index': _idx,
                        'node_id': node_id,
                        'ok': True,
                        'status': 'revised',
                        'deltas': result.get('deltas', []),
                        'warnings': result.get('warnings', []),
                    }
                    # Edge outcomes ride with the row — every failed entry's
                    # reason is visible, and dispatch builds edge traces from it.
                    if 'connect_to_result' in result:
                        row['connect_to_result'] = result['connect_to_result']
                    results.append(row)
                    revised_count += 1
            except Exception as e:
                self._log_error('revise_batch', e, 'revising %s' % node_id[:8])
                results.append({'op': 'revise', 'index': _idx,
                                'node_id': node_id, 'ok': False,
                                'status': 'error', 'error': str(e)})

        return {
            'revised': revised_count,
            'results': results,
        }

    def _generate_summary(self, title: str, content: Optional[str] = None) -> Optional[str]:
        """Generate a content_summary (~200-char head) for tiered recall.

        Returns first sentence of content, or a loudly-capped 200-char head if
        no sentence boundary. Returns None if content is empty or very short
        (title suffices). The cap goes through the shared loud primitive: this
        is a stored field a reader can see, and an ASCII '...' is a cut no
        marker detector recognizes (loud_truncation.looks_truncated).
        """
        if not content or len(content) < 30:
            return None
        # First sentence
        period_idx = content.find('. ')
        if 0 < period_idx < 200:
            return content[:period_idx + 1]
        return cap_text_loud(content, 200)

    def find_node_by_title(self, title_query: str, threshold: float = 0.75,
                           top_k: int = 1,
                           session_id: str = '') -> Optional[Dict[str, Any]]:
        """Find a node by fuzzy title matching using embedding similarity.

        Embeds the query, scans all node embeddings, returns the best match(es)
        above threshold with a content snippet so the caller can verify
        correctness.

        Args:
            title_query: Title to search for (fuzzy)
            threshold: Minimum similarity (0.0-1.0). Default 0.75 is conservative
                       to prevent false matches. Lower to 0.6 for broader search.
            top_k: Return top K matches (default 1 = best match only)

        Returns: {id, title, type, similarity, content_snippet} or None.
                 If top_k > 1, returns list of matches.

        """
        scored = {}  # id → result dict, dedup by node

        # Path 1: lexical — FTS-indexed candidates (_title_candidate_rows,
        # the same door the connect_to write path uses; no table scan),
        # verified by in-order token containment: every query token appears
        # in the title, in order, as a substring — the semantics the old
        # `LIKE %w%w%` full scan implemented, minus its cross-word false
        # positives and minus the scan.
        q_tokens = _title_tokens(title_query)
        text_ids = []
        if q_tokens:
            # Interactive lookup is best-effort: saturation just bounds the
            # lexical pool (Path 2 embeddings backstop); only the WRITE path
            # must refuse on it.
            candidate_rows, _saturated = self._title_candidate_rows(q_tokens)
            for nid, title in candidate_rows:
                t_lower = title.lower()
                pos = 0
                for tok in q_tokens:
                    pos = t_lower.find(tok, pos)
                    if pos == -1:
                        break
                    pos += len(tok)
                else:
                    text_ids.append(nid)
        if text_ids:
            ph = ','.join('?' * len(text_ids))
            for nid, title, ntype, snippet in self.conn.execute(
                    "SELECT id, title, type, SUBSTR(content, 1, 200) "
                    "FROM nodes WHERE id IN (%s)" % ph, text_ids).fetchall():
                scored[nid] = {
                    "id": nid, "title": title, "type": ntype,
                    "similarity": 0.95,  # text match = high confidence
                    "content_snippet": snippet or "",
                }

        # Path 2: Embedding similarity — semantic fallback
        if embedder.is_ready() and len(scored) < top_k:
            query_vec = embedder.embed_query(title_query)
            if query_vec:
                rows = self.conn.execute(
                    "SELECT ne.node_id, ne.embedding, n.title, n.type, "
                    "SUBSTR(n.content, 1, 200) as snippet "
                    "FROM node_enrichments ne JOIN nodes n ON ne.node_id = n.id "
                    "WHERE ne.vector_type = '_primary' AND n.archived = 0"
                ).fetchall()
                for node_id, emb_blob, title, ntype, snippet in rows:
                    if not emb_blob or node_id in scored:
                        continue
                    sim = embedder.cosine_similarity(query_vec, emb_blob)
                    if sim >= threshold:
                        scored[node_id] = {
                            "id": node_id, "title": title, "type": ntype,
                            "similarity": round(sim, 3),
                            "content_snippet": snippet or "",
                        }

        # Scope veil: fuzzy title search is ambient DISCOVERY (a probe
        # phrase enumerates the corpus with content snippets), not the
        # sanctioned reach-for-a-known-id — walled nodes never match.
        # Sessionless callers get the outward-only veil.
        _veil = self.scope_veil(session_id or '')
        if _veil:
            scored = {nid: v for nid, v in scored.items() if nid not in _veil}

        results = sorted(scored.values(), key=lambda x: x["similarity"], reverse=True)

        if top_k == 1:
            return results[0] if results else None
        return results[:top_k]
