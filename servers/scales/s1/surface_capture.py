"""Production capture at the Haiku boundary — the corpus source for the
surface A/B replay bench (eval/surface_capture_ab.py).

Every production recall writes one self-contained JSON file: the structured
inputs to build_surface_prompt, the rendered prompt actually sent (byte
truth), the full agentic-loop messages (round-2 tool calls + rendered tool
results — the "behind the scenes" the operator can't eyeball from a prompt
diff), the selection output, and provenance stamps (git sha, interaction
version, layout, schema hash). Replay re-renders from the structured inputs
and byte-compares against `rendered.user_content`; a mismatch means the
capture missed a field or the renderer drifted — the harness must refuse to
score that item.

REPLAY TIME-PINNING CONTRACT: the rendered candidates carry relative age
strings ("3d ago") computed from wall-clock brain_now() at capture time.
A replay that re-renders later MUST pin servers.clock.brain_now (as
imported by surface_contract's _format_age) to this capture's `ts` before
the byte-compare, or ages drift and every item spuriously fails fidelity.

Off unless the brain opts in with brain_meta `surface_capture.enabled`: the
corpus is a builder's instrument, so an install never carries it. A brain
feeding the replay bench keeps it on continuously — an opt-in window would
bias the corpus toward whatever we happened to be doing that week.
BRAIN_SURFACE_CAPTURE=off forces it off on any brain (the replay bench sets
it, so a replay cannot write itself into the corpus it reads).
Location: $BRAIN_SURFACE_CAPTURE_DIR, else $BRAIN_DB_DIR/surface_captures/
(durable, lives with the brain data — /tmp dies on reboot). Layout:
<dir>/<YYYY-MM-DD>/<HHMMSS>-<recall_ref>.json, atomic tmp+rename.

Capture must NEVER break a recall: every entry point catches everything and
logs to the brain errors table (loud, non-blocking). Pruning is manual —
~30 recalls/day at ~100-200KB each is a few MB/day.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess

from servers.clock import iso_now

CAPTURE_VERSION = 1
_MAX_PAYLOAD_BYTES = 2_000_000  # a recall payload past this is malformed

_git_sha_cache = None


def _git_sha():
    """Repo sha at capture time, cached per process (the daemon restarts on
    deploy, so per-process == per-code-version)."""
    global _git_sha_cache
    if _git_sha_cache is None:
        try:
            root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))
            _git_sha_cache = subprocess.run(
                ['git', '-C', root, 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip() or 'unknown'
        except Exception:
            _git_sha_cache = 'unknown'
    return _git_sha_cache


def _schema_sha():
    from .surface_contract import SURFACE_SELECTION_SCHEMA
    return hashlib.sha1(json.dumps(
        SURFACE_SELECTION_SCHEMA, sort_keys=True).encode()).hexdigest()[:12]


def capture_dir(brain=None):
    """Resolve the capture directory, or None when capture is disabled."""
    if os.environ.get('BRAIN_SURFACE_CAPTURE', '').strip().lower() == 'off':
        return None
    if brain is None or not brain.get_config('surface_capture.enabled', False):
        return None
    explicit = os.environ.get('BRAIN_SURFACE_CAPTURE_DIR', '').strip()
    if explicit:
        return explicit
    base = os.environ.get('BRAIN_DB_DIR', '').strip()
    if not base and brain is not None:
        db_path = getattr(brain, 'db_path', '') or ''
        base = os.path.dirname(db_path) if db_path else ''
    if not base:
        return None
    return os.path.join(base, 'surface_captures')


def _slim_turns(turns):
    """The window as the render reads it: each row's speaker key, full text
    (flat `content`, which trace_contract.row_text reads first) and
    surfaced ids, plus its id, chain and time for reading a capture back —
    not the whole trace record, whose metadata repeats the text."""
    from servers.trace_contract import row_text

    def slim(row):
        if row is None:
            return None
        out = {k: row[k] for k in ('id', 'ref_type', 'chain_id', 'created_at',
                                   'surfaced') if k in row}
        out['content'] = row_text(row)
        return out
    return [{'user': slim(t['user']), 'with': [slim(r) for r in t['with']],
             'assistant': slim(t['assistant']),
             'actions': [slim(r) for r in t['actions']]} for t in turns or []]


def begin(brain, *, candidates_data, user_message, recent_turns,
          retrieval_stats, frame, layout,
          surface_instructions, interaction_stamp,
          user_content, max_tokens, variant, model, effort, session_id,
          shuffle_seed=None, scope=None):
    """Snapshot everything known at the Haiku call boundary.

    Called from _call_surface BEFORE the agentic loop runs — candidates_data
    is deep-copied here because the loop appends tool-fetched candidates in
    place, and replay needs production's ROUND-1 pool. Returns the payload
    dict (a sink the loop appends rounds into), or None when disabled.
    """
    try:
        if capture_dir(brain) is None:
            return None
        return {
            'v': CAPTURE_VERSION,
            # Wall-clock is correct: ts records when the capture file was
            # physically written, and replay pins brain_now to it for age
            # rendering. Captures are off during eval replays, so no
            # conversation-time is ever violated here.
            'ts': iso_now(),  # clock-ok
            'session_id': session_id,
            'stamps': {
                'git_sha': _git_sha(),
                'interaction': 'surface',
                # Flattened from the K-provenance stamp dict — key names kept
                # stable for existing capture readers; fingerprint/source ride
                # alongside so captures stay comparable across installs.
                'interaction_version': (interaction_stamp or {}).get('version'),
                'interaction_id': (interaction_stamp or {}).get('id'),
                'interaction_fingerprint': (interaction_stamp or {}).get('fingerprint', ''),
                'interaction_source': (interaction_stamp or {}).get('source', ''),
                'layout': layout,
                'variant': variant,
                'model': model,
                'effort': effort,
                'schema_sha': _schema_sha(),
            },
            'inputs': {
                'user_message': user_message,
                'recent_turns': _slim_turns(recent_turns),
                'retrieval_stats': retrieval_stats,
                'frame': frame,
                'candidates_pre_tools': copy.deepcopy(candidates_data),
                # Presentation-shuffle seed (§20.12 A2): re-rendering must
                # pass this back to build_surface_prompt or the candidate
                # order won't byte-match. None on pre-shuffle captures.
                'shuffle_seed': shuffle_seed,
                # Session scope (differential exposure): re-rendering must
                # pass this back or foreign-project marks / suppressed KV
                # lines won't byte-match production. None on unscoped
                # sessions and pre-scope captures.
                'scope': scope,
            },
            'rendered': {
                'system': surface_instructions,
                'user_content': user_content,
                'max_tokens': max_tokens,
            },
            # Filled by the agentic loop: the literal message history —
            # round-1 user content, assistant tool_use blocks, full rendered
            # tool results — plus the raw final text.
            'rounds': {},
            # Filled by finish().
            'output': {},
        }
    except Exception as e:
        try:
            brain._log_error('surface_capture', e, 'begin() failed')
        except Exception:
            pass
        return None


def record_rounds(capture, *, messages, raw_final, tool_trace):
    """Attach the agentic loop's literal history to the capture. No-op on
    None (capture disabled). `messages` is the loop's real list — serialized
    at write time, after the loop is done mutating it."""
    if capture is None:
        return
    capture['rounds'] = {
        'messages': messages,
        'raw_final': raw_final,
        'tool_trace': tool_trace,
    }


def finish(brain, capture, *, recall_ref, surfaced, resolved_mode,
           selection_reason, telemetry):
    """Add the selection outcome and write the capture file. Never raises.

    Merges into `output` — _call_surface already put the raw Haiku text
    there before stashing the capture on the brain."""
    if capture is None:
        return None
    try:
        capture['recall_ref'] = str(recall_ref or '')
        capture.setdefault('output', {})
        capture['output'].update({
            'surfaced': surfaced,
            'resolved_mode': resolved_mode,
            'selection_reason': selection_reason,
        })
        capture['telemetry'] = telemetry
        return _write(brain, capture)
    except Exception as e:
        try:
            brain._log_error('surface_capture', e, 'finish() failed')
        except Exception:
            pass
        return None


def _write(brain, capture):
    base = capture_dir(brain)
    if base is None:
        return None
    ts = capture['ts']                      # '2026-07-12T00:31:05.123+00:00'
    day, clock = ts[:10], ts[11:19].replace(':', '')
    out_dir = os.path.join(base, day)
    os.makedirs(out_dir, exist_ok=True)
    ref = (capture.get('recall_ref') or 'noref').replace(os.sep, '_')[:40]
    path = os.path.join(out_dir, '%s-%s.json' % (clock, ref))
    # default=str: candidate dicts can carry numpy scalars / bytes fields —
    # stringify rather than fail the whole capture.
    blob = json.dumps(capture, default=str)
    if len(blob) > _MAX_PAYLOAD_BYTES:
        brain._log_error(
            'surface_capture',
            ValueError('payload %d bytes > %d cap — skipped'
                       % (len(blob), _MAX_PAYLOAD_BYTES)),
            'recall_ref=%s' % capture.get('recall_ref'))
        return None
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(blob)
    os.replace(tmp, path)
    return path
