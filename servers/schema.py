"""
brain — Canonical Database Schema

SINGLE SOURCE OF TRUTH for every table, column, index, and constraint.

HOW MIGRATION WORKS:
  1. On startup, Brain calls ensure_schema(conn)
  2. ensure_schema creates any missing tables from TABLES
  3. For each existing table, it diffs current columns against TABLES
     and ALTERs in any missing columns
  4. Creates all indexes from INDEXES

HOW TO ADD A NEW COLUMN (brain.db only):
  Add it to the relevant table in TABLES below. That's it.
  ensure_schema will ALTER TABLE ADD it on next startup.

  This does NOT hold for LOG_TABLES. ensure_logs_schema runs
  CREATE TABLE IF NOT EXISTS and no column diff, so a column added to a
  LOG_TABLES entry silently never appears on an existing brain_logs.db.
  Add it with an explicit _add_column_if_missing call.

NODE TYPES:
  NODE_TYPES below is documentation and preferred-type guidance only.
  There is no CHECK constraint to update — NODE_TYPE_CHECK has been empty
  since v8.3 so agents can use any type string. Editing NODE_TYPES changes
  no table and requires no migration.

WHAT NOT TO DO:
  Do NOT add migration code in brain.py.
  Do NOT create nodes_vN rebuild tables in brain.py.
  All schema changes go HERE, in this file.
"""

import re
import sqlite3
from datetime import datetime, timezone

BRAIN_VERSION = 39  # v39: community_key_decisions moves into the standard `thought` field ("Defining moments: <title> (id:xxxxxxxx); ..."), then every row of a retired community key goes (COMMUNITY_RETIRED_KEYS: members, key_decisions, size, is_corridor, dominant_type, narrative, open_questions, latest_development) — nothing writes them, and a stored copy would read as current in every view — and community_internal_fraction takes the stamp's 2dp form. _migrate_v39_community_metadata; fresh brains never hold them. See v38 note below for prior version.
# v38: drop entity_dates, the index of dates mined from node and edge text — recall_by_time's `event` anchor reads the node's event_time field instead (ENTITY_DATES_TABLE, _migrate_v38_drop_entity_dates); fresh brains never create it. See v37 note below for prior version.
# v37: drop nodes.critical and idx_nodes_critical — no node carries the flag, nothing can set it and nothing reads it (CRITICAL_COLUMN, _migrate_v37_drop_critical); fresh brains never create it. See v36 note below for prior version.
# v36: drop the tables, columns and indexes no code reads (DEAD_MAIN_TABLES, DEAD_COLUMNS, DEAD_INDEXES); scope/source_attribution values move to node_metadata_kv first (COLUMNS_TO_KV). _migrate_v36_drop_dead_state; fresh brains never create them. See v35 note below for prior version.
# v35: edge_relations carries each claim's own ends (claim_source_id/claim_target_id) — the order its writer gave. _migrate_v35_claim_ends copies every row's ends from its pair, so every claim reads exactly as before; from here add_relation stores the caller's order and reads take direction from the claim, not the pair. See v34 note below for prior version.
# v34: drop the TF-IDF index (node_vectors, doc_freq). Its only reader was recall's keyword step, retired with it; the tables were written on every remember/revise and read by nothing. See v33 note below for prior version.
# v33: delete every edge_context vector — re-queued at boot, rebuilt by the embed worker. The text definition changed (noise excluded, top-15) and nothing invalidated edge_context on edge writes before this version (GraphDAL.on_edge_text_changed). One-shot rebuild; the write-side hook stops the leak. See v32 note below for prior version.
# v32: drop dead edges columns — relation/edge_type/description/stability/decay_rate (v22 leftovers: constant or NULL on every row, zero readers; live relation data incl. decay_rate is edge_relations) plus index idx_edges_type. _migrate_v32_drop_dead_edge_columns converges drifted installs to the declared 7-column edges shape. See v31 note below for prior version.
# v31: voice-quote fields renamed — user_raw_quote → their_raw_quote, anchor_raw_quote → my_raw_quote. _migrate_v31_voice_fields relabels both node_metadata_kv.key and node_enrichments.vector_type (the per-field embedding lane); no re-embed, field names never enter the embedded text.
# v30: drop nodes.project column — project is now system-stamped kv provenance (node_metadata_kv['project']), not a nodes column. _migrate_v30_project_to_kv moves values (slug map: every legacy value → brain) then DROP COLUMN. See v29 note below for prior version.
BRAIN_VERSION_KEY = 'brain_schema_version'

# brain_logs.db structural version. v1 is the baseline stamp: the table shapes
# that existed when versioning was introduced. Structural changes from here get
# a numbered step in LOGS_MIGRATIONS and bump this.
# v5: drop the log tables and columns no code reads or writes
# (DEAD_LOG_TABLES, DEAD_LOG_COLUMNS). See v4 note below for prior version.
# v4: delete every tool_result trace vector — the embed set no longer writes
# them (embed_queue.EAGER_TRACE_REF_TYPES). See v3 note below for prior version.
# v3: thalamus audience values renamed to name the recipient set they select
# ('once' → 'first_session', 'all' → 'every_session').
LOGS_VERSION = 5
LOGS_VERSION_KEY = 'logs_schema_version'

# ─── Allowed node types ───
NODE_TYPES = [
    'person', 'project', 'decision', 'rule', 'concept',
    'task', 'file', 'context', 'intuition', 'procedure',
    'thought', 'object',
    # v4 Code Cognition types
    'fn_reasoning',      # Intent and reasoning behind a function
    'param_influence',   # Parameter with systemic effects across codebase
    'code_concept',      # Semantic unit spanning multiple files/functions (blast radius)
    'arch_constraint',   # What limits what and why
    'causal_chain',      # Regression path: trigger → propagation → failure → root cause
    'bug_lesson',        # General principle extracted from a specific bug
    'comment_anchor',    # Load-bearing comment in code that transfers knowledge
    # v4 Evolution types — forward-facing, describe what is BECOMING
    'tension',           # Contradiction between two existing nodes → must resolve
    'hypothesis',        # Untested belief with confidence score → validate or disprove
    'pattern',           # Meta-observation about recurring behavior → confirm or dismiss
    'catalyst',          # Emotional inflection point that changed direction → permanent
    'aspiration',        # Directional goal without finish line → compass for decisions
    # v4 Self-reflection types — brain looking inward
    'performance',       # Brain's own quality metrics over time (trending, not snapshot)
    'failure_mode',      # Named class of recurring failures with prevention strategy
    'capability',        # What the brain can/cannot do — self-inventory
    'interaction',       # Observed dynamics of human-Claude working relationship
    'meta_learning',     # How the brain learned something — reusable methods
    # v5 Cognitive layer — Claude's own thoughts as first-class data
    'correction',        # Self-correction trace: Claude assumed X, reality was Y, pattern Z
    'validation',        # Positive signal: this approach worked, user confirmed
    'mental_model',      # Claude's understanding of how systems/processes work
    'reasoning_trace',   # Reusable logic chain (not tied to a single decision)
    'uncertainty',       # Where Claude knows it doesn't understand something
    # v5 Engineering memory — kinds of understanding, not code elements
    'purpose',           # What something is and why it exists (system/module/file/function scope)
    'mechanism',         # How something works: flows, algorithms, interactions
    'impact',            # What changes ripple where ("update X → check Y")
    'constraint',        # What must or must not be done (replaces arch_constraint)
    'convention',        # Patterns, utilities, coding style for a codebase
    'lesson',            # What went wrong, root cause, preventive principle
    'vocabulary',        # How operator refers to things → code mapping (shared brain)
]

# v8.3: CHECK constraint removed — agents can use any type string.
# NODE_TYPES list kept for documentation and preferred-type guidance.
NODE_TYPE_CHECK = ""  # was: CHECK(type IN (...)) — removed to allow expressive typing

# ─── Canonical table definitions ───
# Each entry: { 'create': SQL, 'columns': { col: default_for_alter } }
# columns dict is used for diff-based migration (add missing columns via ALTER TABLE)
TABLES = {
    'nodes': {
        'create': f"""CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL {NODE_TYPE_CHECK},
            title TEXT NOT NULL,
            content TEXT,
            access_count INTEGER DEFAULT 1,
            locked INTEGER DEFAULT 0,
            archived INTEGER DEFAULT 0,
            emotion REAL DEFAULT 0,
            emotion_label TEXT DEFAULT 'neutral',
            confidence REAL DEFAULT NULL,
            personal TEXT DEFAULT NULL,
            personal_context TEXT DEFAULT NULL,
            evolution_status TEXT DEFAULT NULL,
            content_summary TEXT DEFAULT NULL,
            encoding_source TEXT DEFAULT NULL,
            revised_at TEXT DEFAULT NULL,
            last_accessed TEXT,
            created_at TEXT,
            updated_at TEXT
        )""",
        'columns': {
            'id': None, 'type': None, 'title': None, 'content': None,
            'access_count': '1',
            'locked': '0', 'archived': '0',
            'emotion': '0', 'emotion_label': "'neutral'",
            'confidence': 'NULL',
            'personal': 'NULL',              # v4: null | 'fixed' | 'fluid' | 'contextual'
            'personal_context': 'NULL',      # v4: qualifier for contextual personal nodes
            'evolution_status': 'NULL',      # v4: 'active' | 'resolved' | 'validated' | 'disproven' | 'confirmed' | 'dismissed'
            'content_summary': 'NULL',       # v5: max 200 chars, auto-generated for tiered recall
            'encoding_source': 'NULL',       # v7: who created this node. Convention: "category:process". anchor = direct MCP, encoder:sonnet = encoding agent, idle:redistribution/consolidation/etc, hook:boot/compaction. Only 'anchor' can lock.
            'revised_at': 'NULL',            # v8: when this node was last revised via revise()
            'last_accessed': 'NULL',
            'created_at': 'NULL', 'updated_at': 'NULL',
        }
    },

    # v22: Physical edge — one row per pair, direction in source/target ordering.
    # source = "actor" node, target = "acted upon" node.
    # Single-direction: no mirror rows. Query both directions with OR.
    'edges': {
        'create': """CREATE TABLE IF NOT EXISTS edges (
            edge_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            weight REAL DEFAULT 0.5,
            last_strengthened TEXT,
            created_at TEXT,
            UNIQUE(source_id, target_id)
        )""",
        'columns': {
            'edge_id': None, 'source_id': None, 'target_id': None,
            'weight': '0.5',
            'last_strengthened': 'NULL', 'created_at': 'NULL',
        }
    },

    # v22: Semantic edge layer — multiple relations per edge.
    # References edges via edge_id. Open text relation types.
    'edge_relations': {
        'create': """CREATE TABLE IF NOT EXISTS edge_relations (
            edge_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            description TEXT DEFAULT '',
            weight REAL DEFAULT 0.5,
            encoding_source TEXT DEFAULT '',
            created_at TEXT,
            updated_at TEXT DEFAULT NULL,
            archived INTEGER DEFAULT 0,
            archived_at TEXT DEFAULT NULL,
            archived_by TEXT DEFAULT NULL,
            embedding BLOB DEFAULT NULL,
            embedding_model TEXT DEFAULT NULL,
            claim_source_id TEXT DEFAULT NULL,
            claim_target_id TEXT DEFAULT NULL,
            PRIMARY KEY (edge_id, relation)
        )""",
        'columns': {
            'edge_id': None, 'relation': None,
            'description': "''", 'weight': '0.5',
            'encoding_source': "''",
            'created_at': 'NULL',
            # When the CLAIM on this line last changed — description, weight
            # or the relation verb (add_relation Branch 2, rename_relation).
            # NULL until the first repair: the edge line renders
            # `updated_at or created_at`, so an untouched row keeps showing
            # its birth date honestly. A no-op re-connect never stamps it.
            'updated_at': 'NULL',
            # v25: soft-archive columns. PK stays (edge_id, relation);
            # add_relation handles re-archive as un-archive (UPDATE
            # archived=0) rather than requiring a composite PK.
            'archived': '0',
            'archived_at': 'NULL',
            'archived_by': 'NULL',
            # v26: stored edge embedding for surface_spread. Computed at
            # add_relation time from `_compose_enriched_edge_text`
            # (intrinsic — relation + description + family meaning, no
            # partner title). NULL until backfilled by
            # scripts/backfill_edge_embeddings.py or a future write.
            # embedding_model carries the model name so cross-model
            # staleness is detectable on swap.
            'embedding': 'NULL',
            'embedding_model': 'NULL',
            # v35: the claim's own ends — the order its writer gave. The
            # `edges` row is the pair (weight, strengthening); its source/
            # target only record how the pair's FIRST claim was written, so
            # direction is read from here, never from the pair.
            'claim_source_id': 'NULL',
            'claim_target_id': 'NULL',
        }
    },

    # v27: episodic references — multi-ref pointer from a node to the
    # trace events that anchor it. Sparse by design (typically 1-3 refs
    # per node). `position` preserves the encoder's write order so
    # render can expand primary refs first under budget. `trace_id`
    # is a cross-DB reference to brain_logs.trace_events.id — no
    # SQLite FK (cross-DB FKs aren't enforced); invalid refs degrade
    # gracefully at recall and are cleaned by S2Healer.
    # v29: trace_id is TEXT (8-char hex), matching node id shape.
    'node_source_refs': {
        'create': """CREATE TABLE IF NOT EXISTS node_source_refs (
            node_id    TEXT  NOT NULL,
            trace_id   TEXT  NOT NULL,
            position   INTEGER  NOT NULL DEFAULT 1,
            created_at TEXT,
            PRIMARY KEY (node_id, trace_id),
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        )""",
        'columns': {
            'node_id': None, 'trace_id': None,
            'position': '1', 'created_at': 'NULL',
        }
    },

    'brain_meta': {
        'create': """CREATE TABLE IF NOT EXISTS brain_meta (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )""",
        'columns': {'key': None, 'value': 'NULL', 'updated_at': 'NULL'}
    },

    # v6→v23: Node vectors — ALL embeddings live here.
    # v6: enrichment vectors (question, anchor, bridge, keywords)
    # v23: also primary (_primary) and situation (_situation) vectors
    # generated at encode time by an LLM. These are searched alongside the primary embedding.
    # See PLAN.md "Embedding Migration to LLM" for design rationale and benchmark results.
    'node_enrichments': {
        'create': """CREATE TABLE IF NOT EXISTS node_enrichments (
            id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL,
            vector_type TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB,
            model TEXT DEFAULT 'nomic-ai/nomic-embed-text-v1.5-Q',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        )""",
        'columns': {
            'id': None, 'node_id': None, 'vector_type': None, 'text': None,
            'embedding': 'NULL', 'model': "'nomic-ai/nomic-embed-text-v1.5-Q'",
            'created_at': 'CURRENT_TIMESTAMP',
        }
    },

    # v21: Key-value metadata — extensible without schema changes
    'node_metadata_kv': {
        'create': """CREATE TABLE IF NOT EXISTS node_metadata_kv (
            node_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            PRIMARY KEY (node_id, key),
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        )""",
        'columns': {
            'node_id': None, 'key': None, 'value': 'NULL',
        }
    },

    's2_rejections': {
        # Rejection memory for S2 integration units. When an encoder rejects
        # a proposal, its fingerprint is stored here so the decoder can
        # suppress re-proposals of identical input on subsequent runs.
        # Fingerprint captures what the encoder actually judges on (not
        # implementation artifacts) — changes in graph state that would
        # alter the proposal's inputs naturally produce a new fingerprint,
        # letting legitimate re-proposals through.
        # See servers/scales/s2/rejection_table.py for fingerprint logic.
        'create': """CREATE TABLE IF NOT EXISTS s2_rejections (
            fingerprint TEXT PRIMARY KEY,
            integration_unit TEXT NOT NULL,
            proposal_type TEXT NOT NULL,
            proposed_ids TEXT,
            created_at TEXT NOT NULL
        )""",
        'columns': {
            'fingerprint': 'TEXT PRIMARY KEY',
            'integration_unit': 'TEXT NOT NULL',
            'proposal_type': 'TEXT NOT NULL',
            'proposed_ids': 'TEXT',
            'created_at': 'TEXT NOT NULL',
        }
    },
}

# ─── Canonical indexes ───
INDEXES = [
    # nodes
    'CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_archived ON nodes(archived)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_emotion ON nodes(emotion)',
    'CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at)',
    # edges (v22: edge_id PK, source/target for lookups)
    'CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)',
    'CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)',
    'CREATE INDEX IF NOT EXISTS idx_edges_weight ON edges(weight)',
    'CREATE INDEX IF NOT EXISTS idx_edges_pair ON edges(source_id, target_id)',
    # edge_relations (v22: edge_id FK, relation for type queries)
    'CREATE INDEX IF NOT EXISTS idx_edge_relations_edge ON edge_relations(edge_id)',
    'CREATE INDEX IF NOT EXISTS idx_edge_relations_relation ON edge_relations(relation)',
    # v27: node_source_refs reverse-lookup (engram cohort detection — get all nodes anchored to a given trace)
    'CREATE INDEX IF NOT EXISTS idx_nsr_trace ON node_source_refs(trace_id)',
    # v6 (LLM migration): node_enrichments
    'CREATE INDEX IF NOT EXISTS idx_enrichments_node ON node_enrichments(node_id)',
    'CREATE INDEX IF NOT EXISTS idx_enrichments_type ON node_enrichments(vector_type)',
    # v23: composite index for vector lookups (backfill_vectors, recall scan)
    'CREATE INDEX IF NOT EXISTS idx_enrichments_node_type ON node_enrichments(node_id, vector_type)',
    # s2_rejections — per-unit queries for cleanup/analysis
    'CREATE INDEX IF NOT EXISTS idx_s2_rejections_unit ON s2_rejections(integration_unit)',
    'CREATE INDEX IF NOT EXISTS idx_s2_rejections_created ON s2_rejections(created_at)',
]


def _now():
    """UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════
# Versioned migration runner — the ONE mechanism for "the code changed,
# migrate this install's data forward".
#
# There is a fleet now (installs other than the author's), so a change cannot
# be applied by hand on one machine. Every versioned stream below self-applies
# at open, exactly once, in order.
#
# Two streams use this, each with its own counter so they move independently:
#
#   brain.db      brain_meta.brain_schema_version   BRAIN_VERSION
#   brain_logs.db logs_meta.logs_schema_version     LOGS_VERSION
#
# (logs_meta.seed_prompts_version was a third stream — the shipped-prompt
# reconcile, deleted with the distribution machinery. Its stamp rows on
# existing installs are inert history.)
#
# Forward-only and idempotent by version guard: a step runs iff the DB's stored
# version is below it. Re-opening a current DB does no writes.
#
# THE STAMP BELONGS TO THE RUNNER. A caller that stamps its target version
# before calling here kills every step: the runner re-reads the version, sees
# itself already current, and returns. That inversion shipped once (dfc74ee,
# reverted) and made MAIN_MIGRATIONS dead by construction.
# ═══════════════════════════════════════════════════════════════

def read_schema_version(conn, meta_table: str, version_key: str) -> int:
    """Stored version for a stream, or 0 when the row is absent.

    0 covers both a fresh DB and a pre-versioning install. Those two need
    OPPOSITE handling (baseline vs. run everything), so the caller decides
    between them with a structural test — never from this number alone.

    Deliberately does not swallow operational errors. A locked DB or a corrupt
    value read as 0 would let `fresh` baseline-stamp a fully-populated brain and
    skip every pending step, silently and permanently. Missing-table is the one
    benign case, and callers create the meta table before reaching here.
    """
    cur = conn.execute(
        "SELECT value FROM %s WHERE key = ?" % meta_table, (version_key,))
    row = cur.fetchone()
    return int(row[0]) if row else 0


def read_meta_value(conn, meta_table: str, key: str):
    """Raw stored value for `key`, or None when the row is absent.

    The un-coerced counterpart to `read_schema_version`, for meta rows that
    carry payload rather than a counter (the override-collapse audit record).
    """
    cur = conn.execute(
        "SELECT value FROM %s WHERE key = ?" % meta_table, (key,))
    row = cur.fetchone()
    return row[0] if row else None


def write_meta_value(conn, meta_table: str, key: str, value: str) -> None:
    """Record `value` under `key` in a meta k/v table. Not committed here —
    the caller owns the transaction boundary."""
    conn.execute(
        "INSERT OR REPLACE INTO %s (key, value, updated_at) "
        "VALUES (?, ?, ?)" % meta_table,
        (key, value, _now()))


def stamp_schema_version(conn, meta_table: str, version_key: str,
                         version: int) -> None:
    """Record a stream as migrated up to `version`."""
    write_meta_value(conn, meta_table, version_key, str(version))


def run_versioned_migrations(conn, meta_table: str, version_key: str,
                             target_version: int, steps, label: str = '',
                             db_path=None, fresh: bool = False) -> int:
    """Run every step the stored version hasn't seen, then stamp. Returns the
    version found on entry (0 = fresh or pre-versioning).

    `steps` is an ordered [(version, callable(conn))]. A step runs iff
    stored_version < its version.

    `fresh` must come from a STRUCTURAL test (no tables yet), not from the
    counter — a pre-versioning install also reads 0 and needs every step.
    A fresh DB is born at the current shape: stamp, run nothing.

    `db_path` enables the pre-migration backup required for any DB that is
    about to be rewritten. Backup fires for a non-fresh DB with pending
    steps INCLUDING one at version 0: pre-versioning installs are the oldest
    DBs in the fleet and the ones a first migration is most likely to damage.

    A failing step is NOT swallowed: the connection is rolled back and the
    exception propagates with the stamp unwritten, so the migration retries on
    the next open instead of marking a half-applied DB current.
    """
    versions = [v for v, _ in steps]
    if versions != sorted(set(versions)):
        raise ValueError('%s: migration steps must be sorted and unique: %s'
                         % (label or meta_table, versions))
    if versions and versions[-1] > target_version:
        raise ValueError(
            '%s: step v%d is above target v%d — it would run early and '
            'stamp low, then run again after the next bump'
            % (label or meta_table, versions[-1], target_version))

    current = read_schema_version(conn, meta_table, version_key)
    if current >= target_version:
        return current

    pending = [(v, fn) for v, fn in steps if current < v]
    if pending and not fresh and db_path:
        # Version-tagged, raw (compress=False): this runs at daemon boot
        # before the port answers pings, and the health monitor force-
        # restarts an unresponsive daemon — the backup must finish in
        # seconds, not stall behind gzip. Idempotent per version, so a
        # retried migration keeps the pre-first-attempt state.
        from .db_backup import backup_before_destructive
        backup_before_destructive(db_path, 'v%d' % current, compress=False)

    ran = 0
    if not (fresh and current == 0):
        for step_version, step_fn in pending:
            try:
                step_fn(conn)
            except Exception:
                conn.rollback()
                raise
            ran += 1
            if label:
                print('[brain] %s: applied step v%d' % (label, step_version),
                      flush=True)

    stamp_schema_version(conn, meta_table, version_key, target_version)
    # Summary only when steps actually ran: a stream whose ladder is empty
    # advances its counter without doing anything, and announcing that as a
    # migration is how a no-op gets mistaken for work.
    if label and ran:
        print('[brain] %s: v%d -> v%d (%d step%s)'
              % (label, current, target_version, ran, '' if ran == 1 else 's'),
              flush=True)
    return current


def _rebuild_nodes(conn, spec):
    """Rebuild nodes table when CHECK constraint changes (new node types)."""
    conn.execute('PRAGMA foreign_keys=OFF')

    # Get current columns
    cur = conn.execute('PRAGMA table_info(nodes)')
    current_cols = [row[1] for row in cur.fetchall()]

    canonical_cols = list(spec['columns'].keys())
    shared_cols = [c for c in current_cols if c in canonical_cols]

    # Build insert columns — use defaults for missing ones
    insert_parts = []
    for c in canonical_cols:
        if c in shared_cols:
            insert_parts.append(c)
        else:
            default = spec['columns'][c]
            insert_parts.append(f"{default} AS {c}" if default is not None else f"NULL AS {c}")

    try:
        conn.execute(spec['create'].replace('nodes', 'nodes_canonical'))
        conn.execute(f"""INSERT OR IGNORE INTO nodes_canonical ({','.join(canonical_cols)})
                        SELECT {','.join(insert_parts)} FROM nodes""")
        conn.execute('DROP TABLE nodes')
        conn.execute('ALTER TABLE nodes_canonical RENAME TO nodes')
    except Exception as e:
        print(f"[brain] nodes rebuild note: {e}")
        try:
            conn.execute('DROP TABLE IF EXISTS nodes_canonical')
        except Exception:
            pass

    conn.execute('PRAGMA foreign_keys=ON')


def _backfill_data(conn, from_version):
    """One-time data backfills for existing brains."""
    if from_version < 8:
        try:
            conn.execute('UPDATE nodes SET confidence = 1.0 WHERE locked = 1 AND confidence IS NULL')
            conn.execute('UPDATE nodes SET confidence = 0.5 WHERE locked = 0 AND confidence IS NULL')
        except Exception:
            pass

    if from_version < 15:
        # v15: Generate content_summary for existing nodes that have content
        try:
            cur = conn.execute(
                "SELECT id, title, content FROM nodes WHERE content IS NOT NULL AND content != '' AND content_summary IS NULL"
            )
            for row in cur.fetchall():
                node_id, title, content = row
                # First sentence or first 200 chars
                summary = content
                period_idx = content.find('. ')
                if 0 < period_idx < 200:
                    summary = content[:period_idx + 1]
                elif len(content) > 200:
                    summary = content[:197] + '...'
                conn.execute(
                    "UPDATE nodes SET content_summary = ? WHERE id = ?",
                    (summary, node_id)
                )
            print(f"[brain] v15 backfill: generated content_summary for existing nodes")
        except Exception as e:
            print(f"[brain] v15 backfill note: {e}")

    if from_version < 22:
        # v22: Rebuild edges + edge_relations with edge_id, single-direction,
        # clean columns (no deprecated relation/edge_type/description/stability).
        _migrate_edges_v22(conn)

    # v23 / v24 have no migration step: no database outside the original
    # development install ever existed at those versions, so no upgrade path
    # was needed. New DBs are created directly at BRAIN_VERSION. v23 consolidated
    # node_embeddings into node_enrichments; v24 promoted situation text
    # from node_enrichments.text to node_metadata_kv. Both are the current
    # state of any fresh brain.

    if from_version < 25:
        # v25: edge_relations soft-archive — matches the node soft-archive
        # pattern. Prevents the asymmetry where archive_node destroyed edge
        # history forever. Three columns added to edge_relations: archived,
        # archived_at, archived_by. archive_node and GraphDAL.remove_relation
        # flip archived=1 instead of DELETEing.
        _migrate_edge_soft_archive_v25(conn)

    if from_version < 26:
        # v26: edge_relations stored embedding — closes the asymmetry
        # where nodes had stored embeddings (one-time write at create,
        # free reads forever) but edges did not (recomputed via fastembed
        # on every spread call, dominating surface_spread phase at ~24s
        # cold). The embedding column holds the vector for
        # `_compose_enriched_edge_text` output, computed at add_relation
        # time. Backfill of existing rows is done by
        # scripts/backfill_edge_embeddings.py (one-shot, idempotent).
        _migrate_edge_embedding_v26(conn)

    if from_version < 28:
        # v28: drop nodes.keywords column + rebuild nodes_fts without
        # the keywords column. The auto-extractor produced near-duplicate
        # tokenizer dumps (idiotic./idiotic, r1r10/r1-r10, skill.md/skillmd)
        # that actively hurt FTS5 precision and TF-IDF scoring. Porter
        # stemming on title+content provides the same lexical signal
        # without the noise. See servers/brain_remember.py — the
        # _extract_keywords and enrich_keywords functions were removed
        # in the same change. Note: skipping from_version < 27 because
        # v27 was schema-additive (new tables only) and doesn't gate
        # anything in this migration.
        _migrate_v28_drop_keywords(conn)

    if from_version < 29:
        # v29: trace_events.id INTEGER → TEXT (8-char hex), trickling through
        # trace_embeddings.trace_id and node_source_refs.trace_id. Brain-wide
        # ID consistency — every entity (node, edge, trace) now shares the
        # 8-char hex shape. Removes the example-authoring sentinel-range hack.
        # node_source_refs in brain.db is the only v29-affected table here;
        # the logs DB migration runs from ensure_logs_schema.
        _migrate_v29_trace_id_main(conn)

    if from_version < 30:
        # v30: drop nodes.project. Project became system-stamped kv
        # provenance (node_metadata_kv['project']) on 2026-07-03 — read by the
        # LAF proj lane + dict filters, never an agent-authored column. Moves
        # legacy values to kv (slug map) then DROP COLUMN, mirroring v28.
        _migrate_v30_project_to_kv(conn)


def _trace_id_column_is_integer(conn, table: str, column: str) -> bool:
    """Returns True only if the table exists AND the column type is INTEGER
    (i.e., legacy pre-v29 state needing migration). Returns False on fresh
    brains (table missing) and on already-migrated brains (column TEXT).
    Self-detecting idempotency for the v29 trace_id migration."""
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        for row in cur.fetchall():
            if row[1] == column:
                return (row[2] or '').upper() == 'INTEGER'
    except Exception:
        pass
    return False


def _migrate_v29_trace_id_main(conn):
    """v29 (brain.db side): node_source_refs.trace_id INTEGER → TEXT.

    The table is empty in production (Phase B WRITE path hasn't shipped),
    so this is a simple recreate. If somehow rows exist, they're migrated
    via deterministic hex: printf('%08x', old_int).
    """
    if not _trace_id_column_is_integer(conn, 'node_source_refs', 'trace_id'):
        # Either fresh brain (no table yet — CREATE TABLE will use TEXT) or
        # already migrated. Skip in both cases.
        return

    # Python's sqlite3 module wraps DML in implicit transactions; DDL after
    # an INSERT in the same conn can be swallowed (no commit between them).
    # We force autocommit for the duration of the rebuild so each statement
    # is durable as it runs, then restore the prior isolation_level.
    prior_isolation = conn.isolation_level
    try:
        n_rows = conn.execute("SELECT COUNT(*) FROM node_source_refs").fetchone()[0]
        print(f"[brain] v29: migrating node_source_refs.trace_id INTEGER → TEXT ({n_rows} rows)")
        conn.commit()  # close any pending tx before switching mode
        conn.isolation_level = None  # autocommit
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("""
            CREATE TABLE node_source_refs_v29 (
                node_id    TEXT  NOT NULL,
                trace_id   TEXT  NOT NULL,
                position   INTEGER  NOT NULL DEFAULT 1,
                created_at TEXT,
                PRIMARY KEY (node_id, trace_id),
                FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            INSERT INTO node_source_refs_v29 (node_id, trace_id, position, created_at)
            SELECT node_id, printf('%08x', trace_id), position, created_at
            FROM node_source_refs
        """)
        n_after = conn.execute("SELECT COUNT(*) FROM node_source_refs_v29").fetchone()[0]
        if n_after != n_rows:
            raise RuntimeError(f"v29: node_source_refs row count drift {n_rows} → {n_after}")
        conn.execute("DROP TABLE node_source_refs")
        conn.execute("ALTER TABLE node_source_refs_v29 RENAME TO node_source_refs")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nsr_trace ON node_source_refs(trace_id)")
        conn.execute("PRAGMA foreign_keys=ON")
        print(f"[brain] v29: node_source_refs migrated cleanly ({n_after} rows)")
    except Exception as e:
        print(f"[brain] v29: node_source_refs migration FAILED: {e}")
        try:
            conn.execute("DROP TABLE IF EXISTS node_source_refs_v29")
            conn.execute("PRAGMA foreign_keys=ON")
        except Exception:
            pass
        raise
    finally:
        conn.isolation_level = prior_isolation


def _migrate_v29_trace_id_logs(conn):
    """v29 (brain_logs.db side): trace_events.id INTEGER → TEXT and
    trace_embeddings.trace_id INTEGER → TEXT.

    Production data: ~60K trace_events rows, ~13K trace_embeddings rows.
    Deterministic hex via printf('%08x', old_int) — preserves lexicographic
    ordering and lets any external integer reference (from logs, debug
    output) be resolved by formatting.

    Self-detecting via PRAGMA table_info — runs only if column is still
    INTEGER. Called from ensure_logs_schema (logs DB has no brain_meta
    version anchor, so column-type probe is the idempotency signal).
    """
    needs_trace_events = _trace_id_column_is_integer(conn, 'trace_events', 'id')
    needs_trace_embeddings = _trace_id_column_is_integer(conn, 'trace_embeddings', 'trace_id')

    if not needs_trace_events and not needs_trace_embeddings:
        return  # fresh brain (table missing — CREATE will use TEXT) or already migrated

    print(f"[brain] v29: logs DB migration starting "
          f"(trace_events={needs_trace_events}, trace_embeddings={needs_trace_embeddings})")

    # Force autocommit so each DDL statement is durable as it runs (Python's
    # sqlite3 swallows DDL after DML in implicit transaction mode).
    prior_isolation = conn.isolation_level
    conn.commit()
    conn.isolation_level = None

    if needs_trace_events:
        try:
            n_rows = conn.execute("SELECT COUNT(*) FROM trace_events").fetchone()[0]
            print(f"[brain] v29: migrating trace_events.id INTEGER → TEXT ({n_rows} rows)")
            conn.execute("""
                CREATE TABLE trace_events_v29 (
                    id TEXT PRIMARY KEY,
                    chain_id TEXT NOT NULL,
                    scale TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    ref_type TEXT,
                    ref_id TEXT,
                    summary TEXT,
                    metadata TEXT,
                    session_id TEXT,
                    interaction_id INTEGER,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                INSERT INTO trace_events_v29
                    (id, chain_id, scale, event_type, ref_type, ref_id,
                     summary, metadata, session_id, interaction_id, created_at)
                SELECT printf('%08x', id), chain_id, scale, event_type,
                       ref_type, ref_id, summary, metadata, session_id,
                       interaction_id, created_at
                FROM trace_events
            """)
            n_after = conn.execute("SELECT COUNT(*) FROM trace_events_v29").fetchone()[0]
            if n_after != n_rows:
                raise RuntimeError(f"v29: trace_events row count drift {n_rows} → {n_after}")
            conn.execute("DROP TABLE trace_events")
            conn.execute("ALTER TABLE trace_events_v29 RENAME TO trace_events")
            # Recreate indexes (will be applied again by ensure_logs_schema
            # post-migration, but recreate here for safety during rollback).
            for idx in [
                "CREATE INDEX IF NOT EXISTS idx_trace_chain ON trace_events(chain_id)",
                "CREATE INDEX IF NOT EXISTS idx_trace_scale ON trace_events(scale)",
                "CREATE INDEX IF NOT EXISTS idx_trace_created ON trace_events(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_trace_session ON trace_events(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_trace_scope_created ON trace_events(scale, ref_type, created_at)",
            ]:
                conn.execute(idx)
            print(f"[brain] v29: trace_events migrated cleanly ({n_after} rows)")
        except Exception as e:
            print(f"[brain] v29: trace_events migration FAILED: {e}")
            try:
                conn.execute("DROP TABLE IF EXISTS trace_events_v29")
            except Exception:
                pass
            raise

    if needs_trace_embeddings:
        try:
            n_rows = conn.execute("SELECT COUNT(*) FROM trace_embeddings").fetchone()[0]
            print(f"[brain] v29: migrating trace_embeddings.trace_id INTEGER → TEXT ({n_rows} rows)")
            conn.execute("""
                CREATE TABLE trace_embeddings_v29 (
                    trace_id    TEXT    PRIMARY KEY,
                    vector      BLOB    NOT NULL,
                    text        TEXT,
                    model       TEXT,
                    created_at  TEXT
                )
            """)
            conn.execute("""
                INSERT INTO trace_embeddings_v29 (trace_id, vector, text, model, created_at)
                SELECT printf('%08x', trace_id), vector, text, model, created_at
                FROM trace_embeddings
            """)
            n_after = conn.execute("SELECT COUNT(*) FROM trace_embeddings_v29").fetchone()[0]
            if n_after != n_rows:
                raise RuntimeError(f"v29: trace_embeddings row count drift {n_rows} → {n_after}")
            conn.execute("DROP TABLE trace_embeddings")
            conn.execute("ALTER TABLE trace_embeddings_v29 RENAME TO trace_embeddings")
            print(f"[brain] v29: trace_embeddings migrated cleanly ({n_after} rows)")
        except Exception as e:
            print(f"[brain] v29: trace_embeddings migration FAILED: {e}")
            try:
                conn.execute("DROP TABLE IF EXISTS trace_embeddings_v29")
            except Exception:
                pass
            conn.isolation_level = prior_isolation
            raise

    conn.isolation_level = prior_isolation


def _migrate_v30_project_to_kv(conn):
    """v30: move nodes.project → node_metadata_kv['project'], then drop the
    column. Project became system-stamped kv provenance (2026-07-03); the
    nodes.project column is legacy. Mirrors _migrate_v28_drop_keywords —
    index before column. Idempotent: re-run finds the column already gone.

    Slug map: every legacy value → 'brain'. The old column carried topical
    costume names ('S1Scribe', 'aspects_refactor', 'dashboard', ...) for what
    was all one repo's work, so the slug is the repo, not the costume.
    """
    # Column already gone (fresh v30 brain, or a re-run) → nothing to do.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(nodes)").fetchall()}
    if 'project' not in cols:
        print("[brain] v30: nodes.project already absent — skipped")
        return

    # 1. Move values → kv. INSERT OR REPLACE on PK (node_id, key) is idempotent.
    #    Total mapping (no unmapped value possible): every legacy value → brain.
    rows = conn.execute(
        "SELECT id FROM nodes "
        "WHERE project IS NOT NULL AND project != ''").fetchall()
    for (nid,) in rows:
        conn.execute(
            "INSERT OR REPLACE INTO node_metadata_kv (node_id, key, value) "
            "VALUES (?, 'project', 'brain')", (nid,))
    print(f"[brain] v30: moved {len(rows)} nodes.project values → kv")

    # 2. Drop the index before the column (SQLite refuses to drop an
    #    indexed column). No-op if already gone.
    try:
        conn.execute("DROP INDEX IF EXISTS idx_nodes_project")
    except Exception as e:
        print(f"[brain] v30: drop idx_nodes_project warning: {e}")

    # 3. Drop the column (SQLite 3.35+). Fail loud — a silent half-migration
    #    (values moved but column kept) is the worst case.
    try:
        conn.execute("ALTER TABLE nodes DROP COLUMN project")
        print("[brain] v30: nodes.project column dropped")
    except Exception as e:
        print(f"[brain] v30: DROP COLUMN project failed: {e}")
        raise


def _migrate_v28_drop_keywords(conn):
    """v28: drop nodes.keywords + rebuild nodes_fts without it.

    Order matters: drop the index BEFORE dropping the column (SQLite
    refuses to drop a column that's referenced by an index). Rebuild
    nodes_fts as a separate step because CREATE VIRTUAL TABLE IF NOT
    EXISTS doesn't reshape the existing virtual table.
    """
    # 1. Drop the legacy keywords index (no-op if already gone).
    try:
        conn.execute("DROP INDEX IF EXISTS idx_nodes_keywords")
    except Exception as e:
        print(f"[brain] v28: drop idx_nodes_keywords warning: {e}")

    # 2. Rebuild nodes_fts: virtual tables can't be ALTERed, so drop +
    #    recreate + repopulate. The cascade drops the auto-managed
    #    nodes_fts_data / _idx / _content / _docsize / _config tables.
    try:
        conn.execute("DROP TABLE IF EXISTS nodes_fts")
        conn.execute("""CREATE VIRTUAL TABLE nodes_fts USING fts5(
            node_id UNINDEXED,
            title,
            content,
            tokenize='porter unicode61'
        )""")
        conn.execute("""
            INSERT INTO nodes_fts (node_id, title, content)
            SELECT id, title, COALESCE(content, '')
            FROM nodes WHERE archived = 0
        """)
        repop_count = conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0]
        print(f"[brain] v28: nodes_fts rebuilt without keywords column "
              f"({repop_count} rows)")
    except Exception as e:
        print(f"[brain] v28: nodes_fts rebuild error: {e}")
        raise  # FTS5 broken means recall lexical channel broken — fail loud

    # 3. Drop the keywords column from nodes. SQLite 3.35+ supports
    #    ALTER TABLE ... DROP COLUMN. The column may already be gone
    #    on fresh brains created at v28; treat that as success.
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(nodes)").fetchall()}
        if 'keywords' in cols:
            conn.execute("ALTER TABLE nodes DROP COLUMN keywords")
            print("[brain] v28: nodes.keywords column dropped")
        else:
            print("[brain] v28: nodes.keywords already absent — skipped")
    except Exception as e:
        # If anything still references the column (views, triggers), this
        # fails. Surface and re-raise — silent half-migration is the worst case.
        print(f"[brain] v28: DROP COLUMN keywords failed: {e}")
        raise


def _migrate_edge_embedding_v26(conn):
    """v26: add embedding + embedding_model columns to edge_relations.

    Existing rows get NULL for both. Backfill is offline via
    scripts/backfill_edge_embeddings.py — kept out of the migration
    path because embedding 21K rows takes minutes of fastembed work
    and shouldn't block daemon boot. Spread reads tolerate NULL by
    falling through to the on-demand embed path (same as before).
    """
    existing = {row[1] for row in conn.execute(
        "PRAGMA table_info(edge_relations)").fetchall()}
    if 'embedding' not in existing:
        conn.execute("ALTER TABLE edge_relations ADD COLUMN embedding BLOB")
    if 'embedding_model' not in existing:
        conn.execute(
            "ALTER TABLE edge_relations ADD COLUMN embedding_model TEXT")
    conn.commit()
    import sys
    print("[schema] v26 migration: edge_relations.embedding column added "
          "(NULL until backfill — run scripts/backfill_edge_embeddings.py)",
          file=sys.stderr, flush=True)


def _migrate_edge_soft_archive_v25(conn):
    """v25: add soft-archive columns to edge_relations.

    Existing rows default to archived=0 (active). All future archive_node
    and remove_relation calls set archived=1 instead of deleting. The
    edges aggregate table is untouched — all reads filter via edge_relations
    joins, so archived edges simply stop joining in.
    """
    existing = {row[1] for row in conn.execute(
        "PRAGMA table_info(edge_relations)").fetchall()}
    if 'archived' not in existing:
        conn.execute("ALTER TABLE edge_relations ADD COLUMN archived INTEGER DEFAULT 0")
    if 'archived_at' not in existing:
        conn.execute("ALTER TABLE edge_relations ADD COLUMN archived_at TEXT")
    if 'archived_by' not in existing:
        conn.execute("ALTER TABLE edge_relations ADD COLUMN archived_by TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_edge_relations_active "
        "ON edge_relations(edge_id, archived)")
    conn.commit()
    import sys
    print("[schema] v25 migration: edge_relations now supports soft-archive",
          file=sys.stderr)


def _migrate_edges_v22(conn):
    """Rebuild edges + edge_relations for v22: edge_id, single-direction, clean columns.

    Handles upgrading from:
    - v20: old edges (bidirectional, deprecated columns), no edge_relations
    - v21: old edges + old edge_relations (source_id/target_id PK)

    The migration:
    1. Reads all edges, deduplicates mirrors (keeps one direction per pair)
    2. Assigns edge_id to each unique pair
    3. Migrates relation data to edge_relations with edge_id reference
    4. Rebuilds both tables with clean schemas
    """
    import hashlib

    def _edge_id(src, tgt):
        """Deterministic edge ID from source+target."""
        h = hashlib.md5((src + ':' + tgt).encode()).hexdigest()[:8]
        return 'edg_' + h

    try:
        # Check if already migrated (edges table has edge_id as PK, not just as column)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(edges)").fetchall()}
        if 'edge_id' in cols:
            # Check if edge_id is populated (v22 complete) or NULL (v21 partial)
            sample = conn.execute("SELECT edge_id FROM edges LIMIT 1").fetchone()
            if sample and sample[0] is not None:
                print("[brain] v22 migration: edges already migrated, skipping")
                return
            # edge_id column exists but values are NULL — need full migration

        # Check if old edge_relations exists (v21) or not (v20)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        has_old_edge_relations = 'edge_relations' in tables

        # --- Step 1: Read all old edges, deduplicate mirrors ---
        old_edges = conn.execute("""
            SELECT source_id, target_id, weight, co_access_count,
                   last_strengthened, created_at,
                   relation, description, decay_rate
            FROM edges
            WHERE source_id < target_id
        """).fetchall()

        # Also get node created_at for direction detection
        node_dates = {}
        for row in conn.execute("SELECT id, created_at FROM nodes").fetchall():
            node_dates[row[0]] = row[1] or ''

        print(f"[brain] v22 migration: {len(old_edges)} unique edge pairs to migrate")

        # --- Step 2: Build new edge rows with direction ---
        new_edges = []  # (edge_id, source, target, weight, co_access, last_str, created)
        edge_id_map = {}  # (canonical_src, canonical_tgt) -> edge_id

        for src, tgt, weight, co_access, last_str, created, rel, desc, decay in old_edges:
            # Direction: newer node is source (encoder's intent)
            src_date = node_dates.get(src, '')
            tgt_date = node_dates.get(tgt, '')

            if src_date > tgt_date:
                final_src, final_tgt = src, tgt
            elif tgt_date > src_date:
                final_src, final_tgt = tgt, src
            else:
                # Same date or missing — lexicographic
                final_src, final_tgt = (src, tgt) if src < tgt else (tgt, src)

            eid = _edge_id(final_src, final_tgt)
            edge_id_map[(src, tgt)] = (eid, final_src, final_tgt)
            # Also map the reverse for lookups
            edge_id_map[(tgt, src)] = (eid, final_src, final_tgt)

            new_edges.append((eid, final_src, final_tgt, weight,
                              co_access or 0, last_str, created))

        # --- Step 3: Build new edge_relations rows ---
        new_relations = []  # (edge_id, relation, description, weight, encoding_source, decay_rate, created)

        if has_old_edge_relations:
            # v21→v22: read from old edge_relations (already has multi-relation data)
            old_rels = conn.execute("""
                SELECT source_id, target_id, relation, description, weight, decay_rate, created_at
                FROM edge_relations
                WHERE source_id < target_id
            """).fetchall()

            for src, tgt, rel, desc, w, decay, created in old_rels:
                mapping = edge_id_map.get((src, tgt))
                if mapping:
                    eid = mapping[0]
                    new_relations.append((eid, rel or 'related', desc or '',
                                          w, 'migration:v22', decay, created))

            print(f"[brain] v22 migration: {len(old_rels)} edge_relations from v21")
        else:
            # v20→v22: no edge_relations, read from old edges columns
            for src, tgt, weight, co_access, last_str, created, rel, desc, decay in old_edges:
                mapping = edge_id_map.get((src, tgt))
                if mapping:
                    eid = mapping[0]
                    new_relations.append((eid, rel or 'related', desc or '',
                                          weight, 'migration:v22', decay, created))

            print(f"[brain] v22 migration: {len(new_relations)} relations from old edges")

        # --- Step 4: Create new tables ---
        conn.execute("DROP TABLE IF EXISTS edges_v22")
        conn.execute("""CREATE TABLE edges_v22 (
            edge_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            weight REAL DEFAULT 0.5,
            co_access_count INTEGER DEFAULT 0,
            last_strengthened TEXT,
            created_at TEXT,
            UNIQUE(source_id, target_id)
        )""")

        conn.execute("DROP TABLE IF EXISTS edge_relations_v22")
        conn.execute("""CREATE TABLE edge_relations_v22 (
            edge_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            description TEXT DEFAULT '',
            weight REAL DEFAULT 0.5,
            encoding_source TEXT DEFAULT '',
            decay_rate REAL DEFAULT NULL,
            created_at TEXT,
            PRIMARY KEY (edge_id, relation)
        )""")

        # --- Step 5: Insert data ---
        conn.executemany(
            "INSERT OR IGNORE INTO edges_v22 VALUES (?, ?, ?, ?, ?, ?, ?)",
            new_edges)
        conn.executemany(
            "INSERT OR IGNORE INTO edge_relations_v22 VALUES (?, ?, ?, ?, ?, ?, ?)",
            new_relations)

        # --- Step 6: Swap tables ---
        conn.execute("DROP TABLE IF EXISTS edges")
        conn.execute("ALTER TABLE edges_v22 RENAME TO edges")

        if has_old_edge_relations:
            conn.execute("DROP TABLE IF EXISTS edge_relations")
        conn.execute("ALTER TABLE edge_relations_v22 RENAME TO edge_relations")

        conn.commit()

        # Verify
        edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        rel_count = conn.execute("SELECT COUNT(*) FROM edge_relations").fetchone()[0]
        print(f"[brain] v22 migration complete: {edge_count} edges, {rel_count} relations")

    except Exception as e:
        print(f"[brain] v22 migration ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise  # Don't silently continue with broken data


def rename_kv_field(conn, old_key: str, new_key: str) -> dict:
    """Rename a promoted metadata field everywhere its NAME is stored.

    A field named in `contract.PROMOTED_FIELDS` with store=metadata_kv lives
    under its own name in two places, and both must move together:

      node_metadata_kv.key      — the value itself
      node_enrichments.vector_type — the per-field embedding lane, when the
                                     field has one in EMBEDDING_GROUPS

    Renaming only the kv key orphans the vectors: the activation kernel looks
    up the new name, misses, and falls back to a blend while a lazy backfill
    re-embeds rows that were never stale. Field names never enter the embedded
    TEXT (the builders join values only), so the stored blobs stay valid — this
    is a relabel, not a re-embed.

    SQL-native and sub-second by design: this runs at daemon boot before the
    port answers pings, and a slow step invites the health monitor to kickstart
    the daemon mid-migration. Returns the per-table row counts for the log line.
    """
    kv = conn.execute(
        "UPDATE node_metadata_kv SET key = ? WHERE key = ?",
        (new_key, old_key)).rowcount
    vec = conn.execute(
        "UPDATE node_enrichments SET vector_type = ? WHERE vector_type = ?",
        (new_key, old_key)).rowcount
    return {'node_metadata_kv': kv, 'node_enrichments': vec}


def _migrate_v31_voice_fields(conn):
    """v31: voice-quote fields renamed to symmetric, frame-neutral names.

    `user_raw_quote` / `anchor_raw_quote` encoded an assistant-serves-user
    frame — a role paired with a name, marking one voice as default and the
    other as bolt-on — and 'anchor' is jargon to the stateless model reading
    the field list. The pair is now two positions: theirs and mine.

    Idempotent by construction: the UPDATEs match the old names, so a re-run
    finds nothing.
    """
    for old, new in (('user_raw_quote', 'their_raw_quote'),
                     ('anchor_raw_quote', 'my_raw_quote')):
        counts = rename_kv_field(conn, old, new)
        print('[brain] v31: %s -> %s (%d kv, %d vectors)'
              % (old, new, counts['node_metadata_kv'],
                 counts['node_enrichments']), flush=True)


def _migrate_v32_drop_dead_edge_columns(conn):
    """v32: drop the five dead edges columns and their index.

    `relation` / `edge_type` / `description` / `stability` / `decay_rate`
    are v22 leftovers — constant or NULL across every row, zero production
    readers; the live relation data (including its decay_rate) lives on
    edge_relations. The declared edges schema above has been the clean
    7-column shape since v22, so this converges a drifted install to its
    own declaration, mirroring v28/v30 (index before column).
    Idempotent: PRAGMA-detected, a re-run finds the columns already gone.
    """
    cols = {row[1] for row in
            conn.execute("PRAGMA table_info(edges)").fetchall()}
    dead = [c for c in ('relation', 'edge_type', 'description',
                        'stability', 'decay_rate') if c in cols]
    if not dead:
        print("[brain] v32: edges dead columns already absent — skipped")
        return

    try:
        conn.execute("DROP INDEX IF EXISTS idx_edges_type")
    except Exception as e:
        print(f"[brain] v32: drop idx_edges_type warning: {e}")

    # Fail loud — a silent half-migration (some columns dropped, others
    # kept) would leave PRAGMA-based detection lying to the next run.
    for col in dead:
        try:
            conn.execute("ALTER TABLE edges DROP COLUMN %s" % col)
            print("[brain] v32: edges.%s dropped" % col)
        except Exception as e:
            print(f"[brain] v32: DROP COLUMN {col} failed: {e}")
            raise


def _migrate_v33_edge_context_rebuild(conn):
    """v33: delete every edge_context vector so the embed worker rebuilds it.

    Two reasons every row is stale at once. The text definition changed:
    noise-aspect relations are excluded and the top-15 (was top-5, with only
    community_member excluded) descriptions are embedded, so no existing row
    matches its node's text. And until v33 no edge write invalidated
    edge_context (GraphDAL.on_edge_text_changed), so ~44% of rows embedded an
    outdated graph snapshot anyway. Deleting makes them missing; the daemon
    re-queues them at boot (_enqueue_vector_backfill_gaps) and the embed
    worker rebuilds in its batches. Idempotent.
    """
    cur = conn.execute(
        "DELETE FROM node_enrichments WHERE vector_type = 'edge_context'")
    print("[brain] v33: %d edge_context vectors deleted — re-queued at daemon "
          "boot, rebuilt by the embed worker" % cur.rowcount)


def _migrate_v34_drop_tfidf_index(conn):
    """v34: drop the TF-IDF index — node_vectors and doc_freq.

    Its only reader was recall's keyword step, retired because every live node
    is embedded and a keyword-only score could not clear recall's floors. The
    tables were written on every remember/revise and read by nothing. Their
    indexes (idx_vectors_term, idx_vectors_node) drop with them. Idempotent.
    """
    for table in ('node_vectors', 'doc_freq'):
        conn.execute("DROP TABLE IF EXISTS %s" % table)
    print("[brain] v34: TF-IDF index dropped (node_vectors, doc_freq)")


def _migrate_v35_claim_ends(conn):
    """v35: give every relation row its own ends, copied from its pair.

    Until v35 a claim had no order of its own: it read the order of the
    `edges` row, which is whatever the pair's first claim was written as.
    Copying those ends keeps every existing claim reading exactly as it did;
    claims stored against their writer's order stay so until repaired.
    Idempotent: only rows without ends are filled. A relation row whose pair
    is gone (an orphan the maintenance sweep removes) has no ends to copy
    and is left as it is.
    """
    cur = conn.execute(
        "UPDATE edge_relations SET "
        "claim_source_id = (SELECT e.source_id FROM edges e "
        "                   WHERE e.edge_id = edge_relations.edge_id), "
        "claim_target_id = (SELECT e.target_id FROM edges e "
        "                   WHERE e.edge_id = edge_relations.edge_id) "
        "WHERE claim_source_id IS NULL AND EXISTS "
        "(SELECT 1 FROM edges e WHERE e.edge_id = edge_relations.edge_id)")
    print("[brain] v35: %d relation rows given their pair's ends" % cur.rowcount)


# The v36 step's inputs. Frozen: they are what the step removes, not a view of
# the current schema, so a name added after a brain passed v36 would never
# reach that brain — later removals get their own step.
DEAD_MAIN_TABLES = (
    'reasoning_steps',   # before reasoning_chains: its chain_id FK points there
    'reasoning_chains',
    'bridge_proposals',
    'correction_traces',
    'embedding_fidelity',
    'emotion_calibration',
    'node_communities',
    'node_embeddings',
    'node_metadata',
    'project_maps',
    'projects',
    'prune_archive',
    'schema_migrations',
    'session_activity',
    'session_syntheses',
    'summaries',
    'version_history',
)

# Declared columns nothing reads. activation / stability / recency_score /
# emotion_source / encoding_version / co_access_count were still written (a
# constant, or a counter nothing consults); the rest were no longer written.
DEAD_COLUMNS = (
    ('nodes', ('activation', 'stability', 'recency_score', 'emotion_source',
               'encoding_version', 'source_turn_id', 'resolved_at',
               'resolved_by', 'due_date', 'scope', 'source_attribution')),
    ('edges', ('co_access_count',)),
    ('edge_relations', ('decay_rate',)),
)

# nodes columns whose field moved to node_metadata_kv without its old values.
# v36 copies each value to the KV key of the same name before the column goes;
# an existing KV row wins, since kv is canonical.
COLUMNS_TO_KV = ('scope', 'source_attribution')

# Indexes on live columns that no schema declares. The v25 step created
# idx_edge_relations_active; fresh brains never had it, and
# idx_edge_relations_edge and the primary key serve the same edge_id lookups.
DEAD_INDEXES = ('idx_edge_relations_active',)


def _drop_columns(conn, table, columns):
    """Drop whichever of `columns` `table` still has, after dropping every
    index that names one of them (SQLite refuses to drop a column an index
    uses, as a key, in an expression or in a partial index's WHERE). Indexes
    are found by their SQL, not by name, so one this code never declared
    cannot block the drop. Constraint indexes carry no SQL and are left to
    fail the drop loudly. Fails loud."""
    present = {row[1] for row in
               conn.execute("PRAGMA table_info(%s)" % table).fetchall()}
    doomed = sorted(c for c in columns if c in present)
    if not doomed:
        return
    names = re.compile(r'\b(%s)\b' % '|'.join(map(re.escape, doomed)),
                       re.IGNORECASE)
    for index, sql in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index' "
            "AND tbl_name = ? AND sql IS NOT NULL", (table,)).fetchall():
        if names.search(re.split(r'\sON\s', sql, maxsplit=1,
                                 flags=re.IGNORECASE)[-1]):
            conn.execute('DROP INDEX IF EXISTS "%s"' % index)
    for column in doomed:
        conn.execute("ALTER TABLE %s DROP COLUMN %s" % (table, column))


def _migrate_v36_drop_dead_state(conn):
    """v36: drop the tables, columns and indexes nothing reads.

    No code reads any DEAD_MAIN_TABLES entry or DEAD_COLUMNS column, and fresh
    brains create none of them. Most of the tables are empty; the rest hold
    history no reader wants (embedding_fidelity's vectors from the retired
    redistribution, the old ladders' version_history / schema_migrations, two
    session_syntheses rows). The DEAD_LOG_TABLES go too: a brain.db from before
    the logs split can still hold them, and the split only moves the tables
    LOG_TABLES declares. COLUMNS_TO_KV values move to node_metadata_kv before
    their columns drop. Idempotent: IF EXISTS drops, INSERT OR IGNORE, and
    PRAGMA-probed columns.
    """
    for table in DEAD_MAIN_TABLES + DEAD_LOG_TABLES:
        conn.execute("DROP TABLE IF EXISTS %s" % table)
    for index in DEAD_INDEXES:
        conn.execute("DROP INDEX IF EXISTS %s" % index)
    node_columns = {row[1] for row in
                    conn.execute("PRAGMA table_info(nodes)").fetchall()}
    for column in COLUMNS_TO_KV:
        if column in node_columns:
            conn.execute(
                "INSERT OR IGNORE INTO node_metadata_kv (node_id, key, value) "
                "SELECT id, ?, {0} FROM nodes "
                "WHERE {0} IS NOT NULL AND {0} != ''".format(column),
                (column,))
    for table, columns in DEAD_COLUMNS:
        _drop_columns(conn, table, columns)
    print("[brain] v36: dead tables, columns and indexes dropped", flush=True)


# The v37 step's input, frozen for the reason DEAD_MAIN_TABLES is.
CRITICAL_COLUMN = ('nodes', ('critical',))


def _migrate_v37_drop_critical(conn):
    """v37: drop nodes.critical and its index idx_nodes_critical.

    No node carries the flag, nothing can set it and nothing reads it. Every
    value is 0, so re-adding the column with DEFAULT 0 restores the same
    state. Idempotent: _drop_columns probes the column.
    """
    _drop_columns(conn, *CRITICAL_COLUMN)
    print("[brain] v37: nodes.critical dropped", flush=True)


# The v38 step's input, frozen for the reason DEAD_MAIN_TABLES is.
ENTITY_DATES_TABLE = 'entity_dates'


def _migrate_v38_drop_entity_dates(conn):
    """v38: drop entity_dates, the index of dates mined from node and edge text.

    recall_by_time's `event` anchor reads each node's event_time field instead
    (MetadataDAL.node_ids_with_time_in); nothing else read the table and
    nothing writes it any more. Its indexes drop with it. Idempotent: IF EXISTS.
    """
    conn.execute("DROP TABLE IF EXISTS %s" % ENTITY_DATES_TABLE)
    print("[brain] v38: entity_dates dropped", flush=True)


def _key_decisions_as_thought(value):
    """The `thought` a community's key-decisions string becomes: the
    encoder's comma-joined "id: title" pairs as "Defining moments: title
    (id:xxxxxxxx); ...", or the value verbatim after the prefix when it does
    not parse as those pairs."""
    parts = re.split(r',\s*(?=[0-9a-f]{8}\s*:)', value.strip())
    pairs = [re.fullmatch(r'([0-9a-f]{8})\s*:\s*(.+)', p.strip(), re.S)
             for p in parts]
    body = ('; '.join('%s (id:%s)' % (m.group(2).strip(), m.group(1))
                      for m in pairs)
            if all(pairs) else value.strip())
    return 'Defining moments: ' + body


def _migrate_v39_community_metadata(conn):
    """v39: a community's key decisions move into its `thought`, every
    retired community key's rows go, and internal_fraction takes the stamp's
    two-decimal form.

    `community_key_decisions` was the encoder's pick of a community's
    defining members, written at creation and read by nothing since. The
    same judgment now lives in the standard `thought` field, which the
    community encoder maintains on every visit. Each row becomes the node's
    thought (appended below one that is already there). A row whose node no
    longer exists (production carried 20) has no node to hold a thought —
    writing one would fail the foreign key and, with it, the boot — so it is
    only deleted.

    Then the rows of every key in COMMUNITY_RETIRED_KEYS are deleted, orphans
    included. Nothing writes those keys, and every view renders each stored
    key, so a kept row would show a frozen value as current.

    Last, each stored `community_internal_fraction` is rewritten as
    str(round(value, 2)) — the form the structural stamp writes and compares
    against. Older code stored three decimals, so without this the first
    decode cycle would revise nearly every community (a trace row and a
    write-lock hold each) for a value that did not move. Same number, fewer
    digits; a value that does not parse is left for the stamp to replace.
    Idempotent: a re-run finds nothing to move, delete or round.
    """
    from servers.scales.s2.community_contract import COMMUNITY_RETIRED_KEYS
    rows = conn.execute(
        "SELECT kd.node_id, kd.value, t.value FROM node_metadata_kv kd "
        "JOIN nodes n ON n.id = kd.node_id "
        "LEFT JOIN node_metadata_kv t "
        "ON t.node_id = kd.node_id AND t.key = 'thought' "
        "WHERE kd.key = 'community_key_decisions'").fetchall()
    moved = 0
    for node_id, decisions, thought in rows:
        if not (decisions or '').strip():
            continue
        text = _key_decisions_as_thought(decisions)
        conn.execute(
            "INSERT OR REPLACE INTO node_metadata_kv (node_id, key, value) "
            "VALUES (?, 'thought', ?)",
            (node_id, '%s\n\n%s' % (thought, text) if thought else text))
        moved += 1
    retired = sorted(COMMUNITY_RETIRED_KEYS)
    deleted = conn.execute(
        "DELETE FROM node_metadata_kv WHERE key IN (%s)"
        % ','.join('?' * len(retired)), retired).rowcount
    rounded = 0
    for node_id, value in conn.execute(
            "SELECT node_id, value FROM node_metadata_kv "
            "WHERE key = 'community_internal_fraction'").fetchall():
        try:
            two = str(round(float(value), 2))
        except (TypeError, ValueError):
            continue
        if two != value:
            conn.execute(
                "UPDATE node_metadata_kv SET value = ? WHERE node_id = ? "
                "AND key = 'community_internal_fraction'", (two, node_id))
            rounded += 1
    print('[brain] v39: %d community key_decisions moved into thought, '
          '%d retired community-key rows deleted, %d internal_fraction '
          'values rounded to 2dp' % (moved, deleted, rounded), flush=True)


# Numbered structural migrations for brain.db, for the runner to apply.
# The declarative TABLES diff and the _backfill_data ladder both stay; this is
# for changes neither can express. A v33+ change adds (33, _migrate_v33) here
# and bumps BRAIN_VERSION — nothing else.
MAIN_MIGRATIONS = [
    (31, _migrate_v31_voice_fields),
    (32, _migrate_v32_drop_dead_edge_columns),
    (33, _migrate_v33_edge_context_rebuild),
    (34, _migrate_v34_drop_tfidf_index),
    (35, _migrate_v35_claim_ends),
    (36, _migrate_v36_drop_dead_state),
    (37, _migrate_v37_drop_critical),
    (38, _migrate_v38_drop_entity_dates),
    (39, _migrate_v39_community_metadata),
]


def ensure_schema(conn, db_path=None):
    """
    The ONLY function that touches table structure.
    Mirrors schema.js ensureSchema() exactly.

    Args:
        conn: SQLite connection
        db_path: Optional path to the DB file (enables pre-migration backup)
    """
    # 1. Create brain_meta first
    conn.execute(TABLES['brain_meta']['create'])

    # 2. Check current schema version. Read into a LOCAL and pass it down:
    #    steps 2b/8 must see the pre-migration value, and the stamp does not
    #    land until the runner at step 9.
    current_version = read_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY)

    # 2a. A brain newer than this code (a hook fallback running an older
    #     install, a downgrade) is left exactly as it is. Every step below
    #     converges the DB toward THIS code's declaration: run from older
    #     code it would re-add the columns and tables a later version dropped,
    #     and that version's step never runs again to remove them.
    if current_version > BRAIN_VERSION:
        print('[brain] schema: brain.db is v%d, this code declares v%d — '
              'leaving its structure untouched' % (current_version, BRAIN_VERSION),
              flush=True)
        conn.commit()
        return

    # 2b. Backup before migration if version is changing. Version 0 is
    #     excluded here — on this ladder it means "brand new", where a
    #     backup would copy an empty file; a populated pre-versioning brain
    #     is backed up by the runner at step 9 instead. Raw (compress=False):
    #     boot path, see run_versioned_migrations.
    backup_path = None
    if current_version > 0 and current_version < BRAIN_VERSION and db_path:
        from .db_backup import backup_before_destructive
        backup_path = backup_before_destructive(
            db_path, 'v%d' % current_version, compress=False)

    # 3. Get list of existing tables
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cur.fetchall()}

    # 4. Check if nodes table needs rebuild (CHECK constraint removal)
    nodes_need_rebuild = False
    if 'nodes' in existing_tables:
        cur = conn.execute("SELECT sql FROM sqlite_master WHERE name='nodes'")
        row = cur.fetchone()
        if row:
            current_sql = row[0] or ''
            # Rebuild if the old CHECK constraint is still present
            if 'CHECK(type IN' in current_sql or "CHECK (type IN" in current_sql:
                nodes_need_rebuild = True

    # 5. For each canonical table: create or diff+alter
    for table_name, spec in TABLES.items():
        if table_name not in existing_tables:
            conn.execute(spec['create'])
            continue

        if table_name == 'nodes' and nodes_need_rebuild:
            # The rebuild DROPs the nodes table, and its trigger is the DDL
            # probe above — NOT a version change — so the version-gated
            # backup at step 2b can miss it (a current-version brain with
            # legacy CHECK DDL). Backup keyed on the actual trigger; if it
            # cannot be taken, the rebuild waits for a boot that can — the
            # probe re-fires and the legacy CHECK is tolerable meanwhile.
            if db_path:
                from .db_backup import backup_before_destructive
                if not backup_before_destructive(db_path, 'nodes-rebuild',
                                                 compress=False):
                    print('[brain] nodes rebuild SKIPPED: no backup could '
                          'be taken before DROP TABLE nodes', flush=True)
                    continue
            _rebuild_nodes(conn, spec)
            continue

        # Get current columns
        cur = conn.execute(f"PRAGMA table_info({table_name})")
        existing_cols = {row[1] for row in cur.fetchall()}

        # Add missing columns
        for col_name, default_val in spec['columns'].items():
            if col_name not in existing_cols:
                def_clause = f' DEFAULT {default_val}' if default_val is not None else ''
                try:
                    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name}{def_clause}")
                except Exception:
                    pass

    # 6. Create all indexes
    for idx in INDEXES:
        try:
            conn.execute(idx)
        except Exception:
            pass

    # 6b. FTS5 full-text search virtual table
    # FTS5 tables use CREATE VIRTUAL TABLE — no ALTER TABLE support, separate from TABLES dict.
    # Porter stemming: "recommending" matches "recommend". Unicode61 for international text.
    try:
        # v28: 3-column FTS5 schema. Pre-v28 brains had a 4th `keywords`
        # column; _migrate_v28_drop_keywords drops + recreates this table
        # on upgrade. CREATE VIRTUAL TABLE IF NOT EXISTS doesn't rebuild,
        # so existing brains rely on that migration to flip the schema.
        conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
            node_id UNINDEXED,
            title,
            content,
            tokenize='porter unicode61'
        )""")
        # Auto-populate on first run: FTS5 empty but nodes exist
        _fts_count = conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0]
        _node_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE archived = 0").fetchone()[0]
        if _fts_count == 0 and _node_count > 0:
            conn.execute("""
                INSERT INTO nodes_fts (node_id, title, content)
                SELECT id, title, COALESCE(content, '')
                FROM nodes WHERE archived = 0
            """)
            conn.commit()
            print(f"[brain] FTS5 index populated: {_node_count} nodes")
    except Exception as e:
        print(f"[brain] FTS5 setup warning: {e}")

    # 7. One-time data backfills
    if current_version > 0 and current_version < BRAIN_VERSION:
        _backfill_data(conn, current_version)

    # 8. Numbered migrations + the version stamp, both owned by the runner.
    #    Nothing above this line may stamp BRAIN_VERSION: the runner re-reads
    #    the version, so an earlier stamp makes it early-return and silently
    #    skips every step. Running last also means a crash anywhere above
    #    leaves the DB unstamped and the whole sequence retries next open.
    #    `fresh` is structural — a brand-new DB has no `nodes` yet, while a
    #    pre-versioning brain reads version 0 WITH tables and needs the ladder.
    #    db_path is passed even though step 2b already backs up: 2b skips
    #    version 0, so a populated pre-versioning brain.db would otherwise get
    #    no backup from either path once MAIN_MIGRATIONS has a real step. The
    #    two calls cannot double-copy — backup_before_destructive returns an
    #    existing backup untouched. Empty ladders hiding a gap is how attempt 1
    #    shipped a dead runner; this one is closed before its first customer.
    run_versioned_migrations(conn, 'brain_meta', BRAIN_VERSION_KEY,
                             BRAIN_VERSION, MAIN_MIGRATIONS,
                             label='brain schema', db_path=db_path,
                             fresh=('nodes' not in existing_tables))

    # Announced only after the runner returns, because the runner can raise and
    # leave the DB unstamped. Printed before it, this line claims an upgrade
    # that did not happen and repeats the claim on every failed boot.
    if current_version < BRAIN_VERSION:
        print(f"[brain] Schema ensured: v{current_version} -> v{BRAIN_VERSION}"
              + (f" (backup: {backup_path})" if backup_path else ""))

    conn.commit()


# ═══════════════════════════════════════════════════════════════
# LOGS DATABASE — separate from brain.db for isolation
# ═══════════════════════════════════════════════════════════════

# Tables that live in brain_logs.db instead of brain.db.
# These are operational/telemetry tables that grow unbounded and
# don't need referential integrity with the knowledge graph.
LOG_TABLES = {
    # logs_meta — brain_logs.db's version counter, the counterpart to
    # brain_meta in brain.db. The two are deliberately separate so each DB
    # carries its own version and migrates independently. Created explicitly
    # ahead of the table loop in ensure_logs_schema, so the runner can read a
    # version without depending on dict ordering.
    'logs_meta': {
        'create': """CREATE TABLE IF NOT EXISTS logs_meta (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )""",
        'columns': {'key': None, 'value': 'NULL', 'updated_at': 'NULL'}
    },
    'debug_log': {
        'create': """CREATE TABLE IF NOT EXISTS debug_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            event_type TEXT NOT NULL,
            source TEXT,
            metadata TEXT,
            created_at TEXT
        )""",
    },

    'hook_errors': {
        'create': """CREATE TABLE IF NOT EXISTS hook_errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            hook_name TEXT NOT NULL,
            level TEXT NOT NULL DEFAULT 'error',
            error TEXT NOT NULL,
            context TEXT DEFAULT '',
            traceback TEXT DEFAULT '',
            surfaced INTEGER DEFAULT 0
        )""",
        # Every write prunes to the newest keep_rows. One outage writes a
        # DAEMON_DOWN row per live session, so this must outlast several
        # bursts. hook_common.log_hook_error repeats it as a literal (that
        # path must work when servers/ cannot import); test_dal_logs pins both.
        'keep_rows': 400,
    },

    # v9.2: Session state — first-class session-scoped data.
    # Replaces scattered in-memory dicts and brain_meta config keys.
    # Key types: 'fatigue' (node_id=node, value=count), 'journal' (node_id='', value=text),
    # 'context' (node_id='', value=journey text). Session-keyed, auto-cleanup by age.
    'session_state': {
        'create': """CREATE TABLE IF NOT EXISTS session_state (
            session_id TEXT NOT NULL,
            key TEXT NOT NULL,
            node_id TEXT NOT NULL DEFAULT '',
            value TEXT,
            updated_at TEXT,
            PRIMARY KEY (session_id, key, node_id)
        )""",
    },

    'interactions': {
        'create': """CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            template TEXT NOT NULL,
            parameters TEXT,
            created_at TEXT NOT NULL,
            created_by TEXT DEFAULT 'anchor',
            parent_version INTEGER,
            UNIQUE(name, version)
        )""",
    },

    # interaction_active — pointer per name to the currently-active version.
    # Separates "register a new version" from "make it live." Without this,
    # any register_interaction call silently changed Anchor's runtime behavior.
    # Now: register inserts a row; set_active flips the pointer; runtime reads
    # the active pointer. Schema added 2026-05-10.
    'interaction_active': {
        'create': """CREATE TABLE IF NOT EXISTS interaction_active (
            name TEXT PRIMARY KEY,
            version INTEGER NOT NULL,
            set_at TEXT NOT NULL,
            set_by TEXT NOT NULL
        )""",
    },
    # v29: trace_events.id is TEXT (8-char hex), matching node/edge id shape.
    # Generation: TraceDAL.append() calls secrets.token_hex(4) with collision
    # retry. Historical rows migrate via printf('%08x', old_int) — deterministic
    # so any pre-migration integer reference can still be resolved.
    'trace_events': {
        'create': """CREATE TABLE IF NOT EXISTS trace_events (
            id TEXT PRIMARY KEY,
            chain_id TEXT NOT NULL,
            scale TEXT NOT NULL,
            event_type TEXT NOT NULL,
            ref_type TEXT,
            ref_id TEXT,
            summary TEXT,
            metadata TEXT,
            session_id TEXT,
            interaction_id INTEGER,
            created_at TEXT NOT NULL
        )""",
    },

    # v27: per-trace embeddings for episodic references. One row per
    # unique trace_id (PK); shared by N nodes referencing the same
    # trace. The `text` column stores the rendered string that was
    # fed to the embedder — matches node_enrichments convention;
    # enables diagnostics and detects drift if the trace rendering
    # changes over time.
    # v29: trace_id is TEXT (matches trace_events.id).
    'trace_embeddings': {
        'create': """CREATE TABLE IF NOT EXISTS trace_embeddings (
            trace_id    TEXT    PRIMARY KEY,
            vector      BLOB    NOT NULL,
            text        TEXT,
            model       TEXT,
            created_at  TEXT
        )""",
    },

    # Self channel — directed-signal courier (Phase 2a). One session sends a
    # message addressed to another live session (self:<sid>) or self:broadcast;
    # the recipient consumes it once via self_delivered. Pull-based in 2a.
    # Per-message expires_at (resolved by address at send) enforces TTL — readers
    # filter expires_at > now; reap deletes expires_at <= now OR IS NULL. Nullable
    # so the ALTER ADD on the one existing courier succeeds; send() always sets
    # it, so a NULL expiry only ever means a pre-column legacy row — reaped as dead.
    'self_inflight': {
        'create': """CREATE TABLE IF NOT EXISTS self_inflight (
            id TEXT PRIMARY KEY,
            from_session TEXT NOT NULL,
            address TEXT NOT NULL,
            body TEXT NOT NULL,
            refs TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            expires_at TEXT
        )""",
    },
    # One row per (message, recipient) — broadcast fans out, each recipient
    # consumes exactly once (PK guards double-delivery).
    'self_delivered': {
        'create': """CREATE TABLE IF NOT EXISTS self_delivered (
            message_id TEXT NOT NULL,
            to_session TEXT NOT NULL,
            delivered_at TEXT NOT NULL,
            PRIMARY KEY (message_id, to_session)
        )""",
    },

    # Thalamus — the brain speaking to its sessions (servers/channels/thalamus/).
    # One item is a STANDING INTENT that can deliver to N sessions over its
    # window; the courier above stays one-shot. `deliver_at` NULL = next
    # opportunity; `dedup_key` is producer-owned or '' (never derived from
    # text); states in thalamus_contract. Delivery is pull-only: sessions
    # self-serve at Stop/boot and record their delivery in the ledger below.
    'thalamus_items': {
        'create': """CREATE TABLE IF NOT EXISTS thalamus_items (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            body TEXT NOT NULL,
            refs TEXT DEFAULT '',
            audience TEXT DEFAULT 'first_session',
            target_session TEXT DEFAULT '',
            needs_answer INTEGER DEFAULT 0,
            dedup_key TEXT DEFAULT '',
            deliver_at TEXT,
            expires_at TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'open',
            answer TEXT DEFAULT '',
            answered_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            armed_epoch INTEGER DEFAULT 0
        )""",
    },
    # One row per (item, session, epoch) delivery — the durable ledger the
    # courier's expiring receipts can't be (annotate-at-render). APPEND-ONLY:
    # a defer re-arms the item by incrementing its armed_epoch (a new
    # generation), never by deleting ledger rows — "never delivered" and
    # "delivered, then deferred" must stay distinguishable (Phase 3 retry
    # gates on unacked). PK guards re-render idempotence per epoch.
    'thalamus_deliveries': {
        'create': """CREATE TABLE IF NOT EXISTS thalamus_deliveries (
            item_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            delivered_at TEXT NOT NULL,
            via TEXT NOT NULL,
            armed_epoch INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (item_id, session_id, armed_epoch)
        )""",
    },

    # Boot observability — the exact `for_claude` text the daemon rendered and
    # served to a session at SessionStart. context_boot is read-only re: the
    # knowledge graph, but it logs what it served here (the same pattern as
    # recall writing a trace from a read path). One row per real boot; the
    # dashboard's Sessions→Boot view reads the latest per session. "What
    # actually got to boot." Written by Brain.record_boot_render via
    # _handle_context_boot; never gated on a BRAIN_VERSION bump (new table).
    'boot_renders': {
        'create': """CREATE TABLE IF NOT EXISTS boot_renders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user TEXT DEFAULT '',
            project TEXT DEFAULT '',
            char_count INTEGER DEFAULT 0,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""",
    },

}

LOG_INDEXES = [
    'CREATE INDEX IF NOT EXISTS idx_debug_session ON debug_log(session_id)',
    'CREATE INDEX IF NOT EXISTS idx_debug_type ON debug_log(event_type)',
    'CREATE INDEX IF NOT EXISTS idx_debug_created ON debug_log(created_at)',
    # hook_errors
    'CREATE INDEX IF NOT EXISTS idx_hook_errors_hook ON hook_errors(hook_name)',
    'CREATE INDEX IF NOT EXISTS idx_hook_errors_created ON hook_errors(created_at)',
    # interactions
    'CREATE INDEX IF NOT EXISTS idx_interactions_name ON interactions(name)',
    # trace_events
    'CREATE INDEX IF NOT EXISTS idx_trace_chain ON trace_events(chain_id)',
    'CREATE INDEX IF NOT EXISTS idx_trace_scale ON trace_events(scale)',
    'CREATE INDEX IF NOT EXISTS idx_trace_created ON trace_events(created_at)',
    'CREATE INDEX IF NOT EXISTS idx_trace_session ON trace_events(session_id)',
    # v27: composite covers the embed-worker hot query (find_unembedded):
    # filter by scale + ref_type, sort by created_at DESC, stop early at
    # LIMIT. Without this SQLite materializes all matches into a temp
    # B-tree just to sort — every drain tick. With it the planner walks
    # the index in DESC order and short-circuits after LIMIT matches.
    'CREATE INDEX IF NOT EXISTS idx_trace_scope_created ON trace_events(scale, ref_type, created_at)',
    # journal-note subject pulls: WHERE ref_type='journal_note' AND ref_id=?
    # ORDER BY created_at — the hotspot/subject query (N notes on one node).
    'CREATE INDEX IF NOT EXISTS idx_trace_ref_subject ON trace_events(ref_type, ref_id, created_at)',
    # get_session_turns hot path (hook_recall, once per user prompt): filter by
    # session + scale, walk created_at DESC, stop at LIMIT — same short-circuit
    # pattern as idx_trace_scope_created above. Without it the planner loads
    # every row of the session before sorting.
    'CREATE INDEX IF NOT EXISTS idx_trace_session_created ON trace_events(session_id, scale, created_at)',
    # v9.2: session_state
    'CREATE INDEX IF NOT EXISTS idx_session_state_session ON session_state(session_id)',
    # self channel — drain/peek filter by address + expires_at; reap by expires_at;
    # outbox orders by created_at
    'CREATE INDEX IF NOT EXISTS idx_self_inflight_address ON self_inflight(address)',
    'CREATE INDEX IF NOT EXISTS idx_self_inflight_created ON self_inflight(created_at)',
    'CREATE INDEX IF NOT EXISTS idx_self_inflight_expires ON self_inflight(expires_at)',
    'CREATE INDEX IF NOT EXISTS idx_self_delivered_to ON self_delivered(to_session)',
    'CREATE INDEX IF NOT EXISTS idx_thalamus_items_state ON thalamus_items(state)',
    'CREATE INDEX IF NOT EXISTS idx_thalamus_items_source ON thalamus_items(source)',
    # boot_renders — dashboard reads latest-per-session and newest-first
    'CREATE INDEX IF NOT EXISTS idx_boot_renders_session ON boot_renders(session_id)',
    'CREATE INDEX IF NOT EXISTS idx_boot_renders_created ON boot_renders(created_at)',
]


# Numbered structural migrations for brain_logs.db, for the runner to apply.
# Empty at v1: the baseline is "whatever shape this DB already had". The
# operations in ensure_logs_schema below are self-detecting and idempotent and
# have run unversioned for a long time; rewriting proven code to be
# version-gated would risk the working path for no gain. New structural changes
# ride these rails instead of adding another unversioned ALTER.
def _migrate_logs_v2_thalamus_armed_epoch(conn):
    """Thalamus ledger goes append-only: armed_epoch on both tables; the
    ledger PK widens to (item_id, session_id, armed_epoch). SQLite can't
    alter a PK, so the ledger is recreated. Column-probe idempotent.

    Both halves use raw conn.execute — NEVER _add_column_if_missing inside a
    versioned step: that helper swallows non-duplicate OperationalErrors, and
    a swallowed failure would let the runner stamp the version with the
    column missing, permanently (a step must raise so the stamp stays
    unwritten and the migration retries on the next open).

    The step is ONE transaction (BEGIN IMMEDIATE): in legacy isolation mode
    only DML opens an implicit transaction, so without it the RENAME/CREATE
    autocommit — a failure in the second half would then roll back only the
    copy, stranding the ledger rows in thalamus_deliveries_old while the
    retry's column probe skips the first half: permanent ledger loss.
    Inside one transaction the runner's rollback restores everything
    (SQLite DDL is transactional) and the retry reruns the step whole.
    IMMEDIATE also fails fast on a locked DB, before any DDL runs."""
    if not conn.in_transaction:
        conn.execute('BEGIN IMMEDIATE')
    cols = [r[1] for r in conn.execute(
        'PRAGMA table_info(thalamus_deliveries)').fetchall()]
    if 'armed_epoch' not in cols:
        conn.execute('ALTER TABLE thalamus_deliveries '
                     'RENAME TO thalamus_deliveries_old')
        conn.execute(LOG_TABLES['thalamus_deliveries']['create'])
        conn.execute(
            'INSERT INTO thalamus_deliveries '
            '(item_id, session_id, delivered_at, via, armed_epoch) '
            'SELECT item_id, session_id, delivered_at, via, 0 '
            'FROM thalamus_deliveries_old')
        conn.execute('DROP TABLE thalamus_deliveries_old')
    cols = [r[1] for r in conn.execute(
        'PRAGMA table_info(thalamus_items)').fetchall()]
    if 'armed_epoch' not in cols:
        conn.execute('ALTER TABLE thalamus_items '
                     'ADD COLUMN armed_epoch INTEGER DEFAULT 0')


def _migrate_logs_v3_audience_recipient_set(conn):
    """v3: thalamus audience values renamed to name the RECIPIENT SET they
    select — 'once' → 'first_session', 'all' → 'every_session'. 'once' read
    as a frequency, which was never the enum's axis: delivery cardinality is
    the ledger PK's job (once per session per armed_epoch), the enum only
    picks who is eligible. Data-only UPDATEs, idempotent (renamed rows match
    nothing on retry). BEGIN IMMEDIATE for the same atomicity/fast-fail
    contract as the v2 step."""
    if not conn.in_transaction:
        conn.execute('BEGIN IMMEDIATE')
    conn.execute("UPDATE thalamus_items SET audience = 'first_session' "
                 "WHERE audience = 'once'")
    conn.execute("UPDATE thalamus_items SET audience = 'every_session' "
                 "WHERE audience = 'all'")


def _migrate_logs_v4_drop_tool_result_vectors(conn):
    """v4: delete every tool_result trace vector.

    The embed set is conversation (embed_queue.EAGER_TRACE_REF_TYPES), so no
    tool row gets a vector any more. These are the ones written before that:
    ~137k rows, 86% of trace_embeddings, and nothing ranks them — recall_laf's
    matrix is pinned to OPERATOR_DIALOGUE_REF_TYPES and recall_episodes'
    semantic scan defaults to it. The one path that could reach them, the
    opt-in `recall_episodes(query=..., ref_type='tool_result')` lens, had zero
    invocations in the 99 days it existed; every recorded use of that lens
    passed `contains=`, which reads the row.

    Only the vector goes. The trace_events rows stay, so the encoder timeline,
    the `contains=` lens and the ten nodes whose source_refs point at a tool
    row are all untouched.

    Deleting leaves ~560MB on the freelist, which does not shrink the file:
    reclaiming it is a VACUUM, held manually under the maintenance-lock file
    (db_maintenance's docstring owns that call and names this pairing). Doing
    it here would VACUUM a 1.2GB DB at daemon boot, before the port answers
    pings — the health monitor force-restarts an unresponsive daemon, and a
    kill mid-VACUUM would retry forever with the stamp unwritten.

    Idempotent: the retry deletes nothing. BEGIN IMMEDIATE for the same
    atomicity/fast-fail contract as the v2 and v3 steps.
    """
    if not conn.in_transaction:
        conn.execute('BEGIN IMMEDIATE')
    cur = conn.execute(
        "DELETE FROM trace_embeddings WHERE trace_id IN "
        "(SELECT id FROM trace_events WHERE ref_type = 'tool_result')")
    print("[brain] logs v4: %d tool_result vectors deleted — run VACUUM under "
          "the maintenance lock to reclaim the space" % cur.rowcount,
          flush=True)


# The logs v5 step's inputs (v36 also drops DEAD_LOG_TABLES from a pre-split
# brain.db). Frozen, like DEAD_MAIN_TABLES.
DEAD_LOG_TABLES = (
    'access_log', 'brain_telemetry', 'conflict_log', 'curiosity_log',
    'dream_log', 'eval_snapshots', 'health_log', 'message_stream',
    'miss_log', 'pending_consolidation', 'recall_gaps', 'recall_log',
    'schema_migrations', 'staged_learnings', 'suggest_log', 'tuning_log',
)

# Log columns nothing writes or reads. Every debug_log writer fills only
# session_id / event_type / source / metadata / created_at (log_debug folds
# its keyword arguments into metadata); self_inflight.intent left the self
# channel's contract when messages lost their intent axis.
DEAD_LOG_COLUMNS = (
    ('debug_log', ('file_target', 'suggestions_served', 'procedures_served',
                   'node_ids_served', 'latency_ms', 'brain_reachable')),
    ('self_inflight', ('intent',)),
)


def _migrate_logs_v5_drop_dead_state(conn):
    """v5: drop the log tables and columns nothing reads or writes.

    No code reads or writes any DEAD_LOG_TABLES entry and fresh DBs create
    none of them; dream_log was the last declared, its writer gone with the
    dream system. Older couriers still carry self_inflight.intent (DEFAULT
    'signal'); fresh ones never had it. Idempotent: IF EXISTS drops and
    PRAGMA-probed columns. BEGIN IMMEDIATE for the same atomicity/fast-fail
    contract as the v2-v4 steps.
    """
    if not conn.in_transaction:
        conn.execute('BEGIN IMMEDIATE')
    for table in DEAD_LOG_TABLES:
        conn.execute("DROP TABLE IF EXISTS %s" % table)
    for table, columns in DEAD_LOG_COLUMNS:
        _drop_columns(conn, table, columns)
    print("[brain] logs v5: dead log tables and columns dropped", flush=True)

LOGS_MIGRATIONS = [
    (2, _migrate_logs_v2_thalamus_armed_epoch),
    (3, _migrate_logs_v3_audience_recipient_set),
    (4, _migrate_logs_v4_drop_tool_result_vectors),
    (5, _migrate_logs_v5_drop_dead_state),
]


def ensure_logs_schema(conn, db_path=None):
    """Create all log tables in the logs database (brain_logs.db).

    Also handles column migrations for existing tables via ALTER TABLE, and
    runs the versioned structural migrations (LOGS_MIGRATIONS) that every
    install applies to itself at open.

    `db_path` enables the pre-migration backup for a DB that a numbered step is
    about to rewrite. It is optional only because test callers open bare
    connections; the daemon path always passes it.
    """
    # Fresh-DB test FIRST, before anything creates a table. A brand-new
    # brain_logs.db is born at the current shape and needs no steps; a
    # pre-versioning one reads version 0 with tables present and needs them.
    # `sqlite3.connect` on a missing file leaves sqlite_master empty, and
    # neither the WAL pragma nor the v29 probe creates anything.
    was_fresh = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
    ).fetchone()[0] == 0

    conn.execute('PRAGMA journal_mode=WAL')

    # logs_meta before the runner can read a version. Cheap and idempotent, so
    # it runs ahead of the main table loop rather than relying on dict order.
    conn.execute(LOG_TABLES['logs_meta']['create'])

    # A logs DB newer than this code is left as it is, for the reason in
    # ensure_schema step 2a: the CREATE loop below would bring back the
    # tables a later version dropped.
    stored = read_schema_version(conn, 'logs_meta', LOGS_VERSION_KEY)
    if stored > LOGS_VERSION:
        print('[brain] logs schema: brain_logs.db is v%d, this code declares '
              'v%d — leaving its structure untouched' % (stored, LOGS_VERSION),
              flush=True)
        conn.commit()
        return

    # v29: trace_events.id and trace_embeddings.trace_id must migrate from
    # INTEGER to TEXT BEFORE the CREATE TABLE IF NOT EXISTS runs — otherwise
    # SQLite skips the new TEXT definition because the table already exists.
    # The migration helper is self-detecting (column-type probe) so it's
    # safe on fresh brains and idempotent on already-migrated ones.
    _migrate_v29_trace_id_logs(conn)

    for table_name, spec in LOG_TABLES.items():
        conn.execute(spec['create'])

    _add_column_if_missing(conn, 'trace_events', 'interaction_id', 'INTEGER')

    # Per-message self-channel TTL: add expires_at to the courier. send() stamps
    # it on every new message; any legacy NULL row is swept by reap as dead.
    # Predates versioning and is idempotent, so it stays unconditional — but
    # there IS a fleet now, so anything new goes through LOGS_MIGRATIONS rather
    # than an unversioned ALTER.
    _add_column_if_missing(conn, 'self_inflight', 'expires_at', 'TEXT')

    for idx in LOG_INDEXES:
        try:
            conn.execute(idx)
        except Exception:
            pass

    # Hand the runner a clean transaction boundary: schema work above can
    # leave a write transaction open, and inside one a failing step's
    # rollback would discard this whole schema bootstrap rather than the
    # step alone — and a step that flips PRAGMAs (foreign_keys is a silent
    # no-op inside a transaction) or VACUUMs would misbehave.
    conn.commit()

    # Versioned structural migrations + the version stamp, owned by the runner.
    # Runs after the tables exist so a step can assume current shapes.
    run_versioned_migrations(conn, 'logs_meta', LOGS_VERSION_KEY,
                             LOGS_VERSION, LOGS_MIGRATIONS,
                             label='logs schema', db_path=db_path,
                             fresh=was_fresh)
    conn.commit()


def _add_column_if_missing(conn, table: str, column: str, col_type: str):
    """Add a column to an existing table if it doesn't exist.

    SQLite has no IF NOT EXISTS for ALTER TABLE, so the duplicate-column error is
    the expected idempotent-rerun signal — swallowed silently (it isn't a
    failure). Any OTHER OperationalError (locked DB, disk full) is a real
    migration failure and is logged LOUDLY (stderr→daemon.log) rather than
    hidden, so a column that silently failed to add can't break a feature at
    runtime with no trace. Boot continues regardless — matching every migration
    block in this file. Non-OperationalError exceptions (e.g. a bad col_type)
    propagate: those are dev-time bugs that should fail loud."""
    try:
        conn.execute('ALTER TABLE %s ADD COLUMN %s %s' % (table, column, col_type))
    except sqlite3.OperationalError as e:
        if 'duplicate column' not in str(e).lower():
            print('[brain] schema: ALTER TABLE %s ADD COLUMN %s %s failed: %s'
                  % (table, column, col_type, e))


def migrate_logs_to_separate_db(main_conn, logs_conn, main_db_path=None):
    """One-time migration: copy log tables from brain.db to brain_logs.db.

    Idempotent — skips tables that already have data in logs_conn.
    After copying, drops the table from main_conn to reclaim space.

    Runs unconditionally at every Brain() open, so it decides its own backup:
    the version-gated pre-migration backup doesn't cover it (a brain already
    stamped current can still carry legacy log tables). When any log table
    actually exists in the main DB — i.e. something is about to be DROPped —
    brain.db is backed up first via `main_db_path`. The common boot (no legacy
    tables) exits before touching anything.
    """
    present = [t for t in LOG_TABLES if main_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (t,)).fetchone()]
    if not present:
        return []
    if main_db_path:
        from .db_backup import backup_before_destructive
        if not backup_before_destructive(main_db_path, 'pre-logs-split',
                                         compress=False):
            print('[brain] logs split SKIPPED: no backup could be taken '
                  'before dropping legacy log tables', flush=True)
            return []

    migrated = []
    for table_name in present:
        # Check if logs DB already has data for this table
        try:
            cur = logs_conn.execute('SELECT COUNT(*) FROM %s' % table_name)
            if cur.fetchone()[0] > 0:
                # Already migrated — just drop from main
                try:
                    main_conn.execute('DROP TABLE IF EXISTS %s' % table_name)
                except Exception:
                    pass
                continue
        except Exception:
            pass

        # Copy data
        try:
            cur = main_conn.execute('SELECT * FROM %s' % table_name)
            rows = cur.fetchall()
            if rows:
                # Get column names from main DB
                col_cur = main_conn.execute('PRAGMA table_info(%s)' % table_name)
                col_names = [r[1] for r in col_cur.fetchall()]
                placeholders = ','.join(['?'] * len(col_names))
                col_list = ','.join(col_names)
                logs_conn.executemany(
                    'INSERT OR IGNORE INTO %s (%s) VALUES (%s)' % (table_name, col_list, placeholders),
                    rows
                )
            # Drop from main DB
            main_conn.execute('DROP TABLE IF EXISTS %s' % table_name)
            migrated.append(table_name)
        except Exception as e:
            print('[brain] Log migration note for %s: %s' % (table_name, e))

    if migrated:
        # Rows imported from the legacy DB arrive AFTER ensure_logs_schema
        # stamped this DB as current — the stamp covered the table shapes it
        # created, not values that showed up behind it. A LOGS migration is by
        # design a data rewrite, so leaving the stamp would exempt exactly
        # these rows from it forever. Reset the counter so the ladder re-runs
        # against the imported data on the next open; the steps are idempotent
        # and the shapes are already current, so a replay is cheap.
        try:
            logs_conn.execute('DELETE FROM logs_meta WHERE key = ?',
                              (LOGS_VERSION_KEY,))
            print('[brain] logs schema: version reset — %d imported table(s) '
                  'must face the migration ladder' % len(migrated), flush=True)
        except Exception as e:
            print('[brain] logs schema: version reset failed: %s' % e,
                  flush=True)
        logs_conn.commit()
        main_conn.commit()
        print('[brain] Migrated %d log table(s) to brain_logs.db: %s' % (len(migrated), ', '.join(migrated)))

    return migrated
