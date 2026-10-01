"""
brain — Daemon Command Dispatch

Table-driven command routing. Each command maps to a handler function with
read/write classification for lock management. Handlers live in the
dispatch_* domain modules; this file is the single registry (COMMAND_TABLE)
that daemon_server.py looks up.

Commands are plain functions: handler(brain, args, graph_changes) -> dict
"""

from typing import Dict

from .dispatch_common import (
    CmdEntry, check_unknown_keys, failed_write_ops, log_failed_batch_ops,
    op_targets, caller_session,
)
from .mutation_emitter import emit_mutation_traces
from .dispatch_write import (
    _handle_remember, _handle_remember_command, _handle_remember_batch,
    _handle_revise, _handle_revise_batch,
    _handle_brain_batch, _handle_connect, _handle_connect_batch, _handle_revise_edge,
    _handle_set_node_lock,
)
from .dispatch_read import (
    _handle_recall, _handle_recall_batch, _handle_get_node,
    _handle_get_nodes, _handle_find_node_by_title, _handle_filter_nodes,
    _handle_context_boot,
)
from .dispatch_observability import (
    _handle_trace_append, _handle_get_trace, _handle_get_traces, _handle_query_traces,
    _handle_recall_episodes,
    _handle_count_traces, _handle_query_logs,
    _handle_clear_errors, _handle_list_interactions,
    _handle_get_interaction, _handle_get_interaction_effective,
    _handle_set_interaction_active, _handle_register_interaction,
    _handle_clear_interaction_override,
)
from .dispatch_ops import (
    _handle_ping, _handle_get_config, _handle_pre_edit,
    _handle_save, _handle_reset_session, _handle_set_config,
    _handle_eval,
    _handle_drop_sys_revision_history,
    _handle_s2_force, _handle_drain_embeddings,
)
from .dispatch_self import (
    _handle_self_presence, _handle_self_peek, _handle_self_send, _handle_self_inbox,
    _handle_self_inbox_peek, _handle_self_outbox)
from .dispatch_thalamus import (
    _handle_remind, _handle_thalamus_list, _handle_thalamus_resolve)


COMMAND_TABLE: Dict[str, CmdEntry] = {
    # ── Reads (no lock needed) ──
    "ping":                     CmdEntry(_handle_ping,                 is_write=False),
    "context_boot":             CmdEntry(_handle_context_boot,         is_write=False),
    "recall":                   CmdEntry(_handle_recall,               is_write=False),
    "get_config":               CmdEntry(_handle_get_config,           is_write=False),
    "pre_edit":                 CmdEntry(_handle_pre_edit,             is_write=False),

    # ── Self channel — presence (read-only) + signal (writes brain_logs.db,
    #    lock-guarded inside signal.py, so is_write=False at the daemon layer) ──
    "self_presence":            CmdEntry(_handle_self_presence,        is_write=False),
    "self_peek":                CmdEntry(_handle_self_peek,            is_write=False),
    "self_send":                CmdEntry(_handle_self_send,            is_write=False),
    "self_inbox":               CmdEntry(_handle_self_inbox,           is_write=False),
    "self_inbox_peek":          CmdEntry(_handle_self_inbox_peek,      is_write=False),
    "self_outbox":              CmdEntry(_handle_self_outbox,          is_write=False),

    # ── Thalamus — the brain speaking to its streams (writes brain_logs.db,
    #    lock-guarded inside thalamus.py, so is_write=False at the daemon layer) ──
    "remind":                   CmdEntry(_handle_remind,               is_write=False),
    "thalamus_list":            CmdEntry(_handle_thalamus_list,        is_write=False),
    "thalamus_resolve":         CmdEntry(_handle_thalamus_resolve,     is_write=False),

    # ── Deterministic background-loop triggers (eval entities) — is_write=False
    #    because both own their serialization internally (run_s2's single-flight
    #    lock; drain batches take brain.write_lock per batch), exactly like the
    #    daemon poll that runs them outside dispatch. Holding the dispatch
    #    exclusive lock for a multi-minute S2 cycle would also deadlock S2's own
    #    encoder dispatch. ──
    "s2_force":                 CmdEntry(_handle_s2_force,             is_write=False),
    "drain_embeddings":         CmdEntry(_handle_drain_embeddings,     is_write=False),

    # ── Writes (exclusive lock) ──
    "save":                CmdEntry(_handle_save,               is_write=True, marks_dirty=False),
    "reset_session":       CmdEntry(_handle_reset_session,      is_write=True, marks_dirty=True),
    "set_config":          CmdEntry(_handle_set_config,         is_write=True, marks_dirty=True),
    "remember":              CmdEntry(_handle_remember_command,     is_write=True, marks_dirty=True),
    "remember_batch":        CmdEntry(_handle_remember_batch,      is_write=True, marks_dirty=True),
    "revise":                CmdEntry(_handle_revise,               is_write=True, marks_dirty=True),
    "revise_batch":          CmdEntry(_handle_revise_batch,         is_write=True, marks_dirty=True),
    "find_node_by_title":    CmdEntry(_handle_find_node_by_title,  is_write=False, marks_dirty=False),
    "filter_nodes":          CmdEntry(_handle_filter_nodes,        is_write=False, marks_dirty=False),
    "query_logs":            CmdEntry(_handle_query_logs,          is_write=False, marks_dirty=False),
    "clear_errors":          CmdEntry(_handle_clear_errors,        is_write=True,  marks_dirty=False),
    "query_traces":          CmdEntry(_handle_query_traces,        is_write=False, marks_dirty=False),
    "recall_episodes":       CmdEntry(_handle_recall_episodes,     is_write=False, marks_dirty=False),
    "count_traces":          CmdEntry(_handle_count_traces,        is_write=False, marks_dirty=False),
    "list_interactions":     CmdEntry(_handle_list_interactions,   is_write=False, marks_dirty=False),
    "get_interaction":       CmdEntry(_handle_get_interaction,     is_write=False, marks_dirty=False),
    "get_interaction_effective": CmdEntry(_handle_get_interaction_effective, is_write=False, marks_dirty=False),
    "register_interaction":  CmdEntry(_handle_register_interaction,is_write=True,  marks_dirty=False),
    "set_interaction_active": CmdEntry(_handle_set_interaction_active, is_write=True,  marks_dirty=False),
    "clear_interaction_override": CmdEntry(_handle_clear_interaction_override, is_write=True, marks_dirty=False),
    "trace_append":          CmdEntry(_handle_trace_append,        is_write=True,  marks_dirty=False),
    "get_node":              CmdEntry(_handle_get_node,             is_write=False, marks_dirty=False),
    "get_nodes":             CmdEntry(_handle_get_nodes,            is_write=False, marks_dirty=False),
    "get_trace":             CmdEntry(_handle_get_trace,            is_write=False, marks_dirty=False),
    "get_traces":            CmdEntry(_handle_get_traces,           is_write=False, marks_dirty=False),
    "recall_batch":          CmdEntry(_handle_recall_batch,         is_write=False, marks_dirty=False),
    "connect":               CmdEntry(_handle_connect,             is_write=True, marks_dirty=True,
                                      accepts=frozenset({"source_id", "target_id", "relation",
                                                         "weight", "description", "encoding_source",
                                                         "chain_id", "session_id", "reason"})),
    "connect_batch":         CmdEntry(_handle_connect_batch,       is_write=True, marks_dirty=True,
                                      accepts=frozenset({"connections", "encoding_source",
                                                         "chain_id", "session_id", "reason"})),
    "revise_edge":           CmdEntry(_handle_revise_edge,         is_write=True, marks_dirty=True,
                                      accepts=frozenset({"source_id", "target_id", "relation",
                                                         "new_relation", "description", "weight",
                                                         "encoding_source", "chain_id", "session_id",
                                                         "reason"})),
    "brain_batch":           CmdEntry(_handle_brain_batch,         is_write=True, marks_dirty=True,
                                      accepts=frozenset({"operations", "encoding_source",
                                                         "chain_id", "session_id", "reason"})),
    # Operator-only (scales/dispatch.py OPERATOR_ONLY_COMMANDS): the encoder
    # dispatch closure refuses it by membership, and the owner method refuses
    # non-anchor encoding_source at the write boundary. The one door for lock
    # flips. (WRITE_COMMANDS is attribution classification, not a gate.)
    "set_node_lock":         CmdEntry(_handle_set_node_lock,       is_write=True, marks_dirty=True,
                                      accepts=frozenset({"node_id", "locked", "reason",
                                                         "confirm_token", "encoding_source",
                                                         "chain_id", "session_id"})),
    "eval":                  CmdEntry(_handle_eval,                is_write=True, marks_dirty=True),
    "drop_sys_revision_history": CmdEntry(_handle_drop_sys_revision_history, is_write=True, marks_dirty=True),
}

# "shutdown" is handled directly by daemon_server (needs to set self.running)
# "hook_*" commands are dispatched via HOOK_TABLE in daemon_server


def dispatch_command(brain, cmd, args, graph_changes):
    """Execute a registry command. THE one execution path for every caller.

    resolve entry -> validate the arg contract -> run the handler -> post-handler
    loudness checks. Every dispatch caller routes through here: the daemon's TCP
    loop (daemon_server._dispatch), the shared S1+S2 encoder closure
    (scales/s2/base._make_encoder_dispatch), and IsolatedBrain.dispatch — the
    surface the eval scripts run on. Handlers are never called directly.

    THE CALLER OWNS THE WRITE LOCK. For write commands this must be called INSIDE
    the caller's `brain.write_lock` envelope (daemon: `_locked_exec`; encoder:
    `with brain.write_lock`) — brain.db writes share one connection, and an
    unlocked commit there lands another thread's partial batch. (brain_logs.db
    writes serialize themselves inside the DAL write boundary — logs_write_lock
    + logs_conn_w — and no longer depend on this envelope.) This function
    deliberately does not acquire the lock itself: lock policy stays with the
    caller that knows whether it already holds it.

    Registry lookup for POLICY (is_write, marks_dirty) stays with the caller;
    this function owns EXECUTION.
    """
    entry = COMMAND_TABLE.get(cmd)
    if entry is None:
        return {"ok": False, "error": "Unknown command: {}".format(cmd)}

    check_unknown_keys(cmd, entry, args, brain)

    # Captured BEFORE the handler runs: handlers mutate `args` in place (see
    # dispatch_common._pop_session_ctx), so anything read after the call is gone.
    # Capturing identity here — not per handler — is the dispatch-level
    # normalization the identity≠filter refactor left open: every mutation
    # trace inherits the calling session without each handler re-resolving it.
    source = (args or {}).get('encoding_source') or 'anchor'
    session_id = caller_session(args or {})
    chain_id = (args or {}).get('chain_id') or ''
    targets = op_targets(args) if entry.is_write else None

    result = entry.handler(brain, args, graph_changes)

    # Mutation manifest: a converted handler returns what it changed under a
    # `mutations` key; the emitter turns it into trace rows. Popped so the
    # manifest never rides into the caller's tool result. Handlers not yet
    # converted return no manifest and keep their inline _emit_* — the
    # no-double-write property is structural, not remembered.
    if isinstance(result, dict):
        manifest = result.pop('mutations', None)
        if manifest and result.get('ok'):
            emit_mutation_traces(brain, cmd, manifest, session_id=session_id,
                                 chain_id=chain_id, encoding_source=source)

    # Refused write ops ride the reply top-level, like `affected`: the runner
    # copies them onto its action record for the run trace's refused_ops.
    failed = failed_write_ops(cmd, result, targets) if targets is not None else []
    if failed:
        result['failed_ops'] = failed
    log_failed_batch_ops(brain, source, cmd, result, failed)
    return result
