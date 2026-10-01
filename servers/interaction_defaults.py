"""Interaction defaults — the name→(template, config) index for every
interaction whose default lives in code.

One concern: this is the complete answer to "what does interaction X run on
when no DB override exists." Imports only — no content of its own. Each
config default lives in its CONSUMER's contract file (one default home per
interaction); each template lives in its prompt `.py` file. The override
resolver (the get_interaction_* accessors) resolves against this index.

Deliberately wider than the old shipped-prompt roster (7 fleet-advanced
names), which died with the distribution machinery: this index covers every
name with a reader.

Names deliberately absent:
- `encoding_agent`, `s2_edge_families`, `s2_node_families` — dead legacy;
  existing DB rows are inert history.
- `boot`, `voice_surface`, `pre_edit`, `signal_assembler` — retired: every
  config key grepped reader-less (pre_edit's plausible keys read from
  brain_meta, a different store).
- Aspects (`aspects_v1.json`) — already the after-model; no override layer.

ADDING A NEW LEARNABLE BOUNDARY — the complete recipe (a step skipped here
ships a boundary that LOOKS learnable but resolves to a raising KeyError,
discovered in production via the errors table):
1. Config dict in the consumer's contract file (`*_INTERACTION_DEFAULT`
   naming); template in a `*_prompt.py` exporting SYSTEM_PROMPT (LLM
   boundaries only — config-only names use template '').
2. Import both here; add the INTERACTION_DEFAULTS entry.
3. Validator in INTERACTION_VALIDATORS if the config has invariants a typo'd
   override must not violate (scopes is the model).
4. Prompt-backed name → add to PROMPT_BACKED in
   tests/test_interaction_defaults.py (its complement test fails loudly if
   you forget).
5. Verdict row in the collapse module's COLLAPSE_POLICY
   (its test file holds registry↔table parity).
Read it ONLY through the Brain's accessors: get_interaction_effective for an
agent run (template, config and the stamp its delta records, from one
resolution — S2's `_call_llm(name)` does this for the single-shot units),
get_interaction_config or _prompt for a plain read — a direct import of the
default trips the bypass guard.
"""
import hashlib
import json

from .pipeline_contract import (EDGE_CONTEXT_INTERACTION_DEFAULT,
                                validate_edge_context_config)
from .recall_laf import DEFAULT_CONFIG as _RECALL_LAF_DEFAULT
from .scales.s1.encode_contract import (
    S1E_GIST_INTERACTION_DEFAULT, S1E_INTERACTION_DEFAULT,
    validate_s1e_gist_config)
from .scales.s1.encoding_gist_prompt import SYSTEM_PROMPT as _S1E_GIST_PROMPT
from .scales.s1.encoding_prompt import SYSTEM_PROMPT as _S1E_PROMPT
from .scales.s1.surface_contract import SURFACE_INTERACTION_DEFAULT
from .scales.s1.surface_prompt import SYSTEM_PROMPT as _SURFACE_PROMPT
from .scales.s2.aspect_contract import ASPECT_INTERACTION_DEFAULT
from .scales.s2.aspect_prompt import SYSTEM_PROMPT as _ASPECT_PROMPT
from .scales.s2.community_contract import COMMUNITY_DETECTION, COMMUNITY_ENRICHMENT
from .scales.s2.community_enrichment_prompt import SYSTEM_PROMPT as _COMMUNITY_PROMPT
from .scales.s2.consolidation_contract import CONSOLIDATION_ENRICHMENT
from .scales.s2.consolidation_enrichment_prompt import SYSTEM_PROMPT as _CONSOLIDATION_PROMPT
from .scales.s2.healer_contract import HEALER_INTERACTION_DEFAULT
from .scales.s2.healer_prompt import SYSTEM_PROMPT as _HEALER_PROMPT
from .scopes import SCOPES_CONFIG_V1, validate_scopes_config
from .trace_contract import TRACE_RECORDING_NORMAL

# name → (template, config). Config-only interactions carry '' templates.
INTERACTION_DEFAULTS = {
    's1e':                   (_S1E_PROMPT, S1E_INTERACTION_DEFAULT),
    's1e_gist':              (_S1E_GIST_PROMPT, S1E_GIST_INTERACTION_DEFAULT),
    'surface':               (_SURFACE_PROMPT, SURFACE_INTERACTION_DEFAULT),
    's2_community_enrichment':     (_COMMUNITY_PROMPT, COMMUNITY_ENRICHMENT),
    's2_consolidation_enrichment': (_CONSOLIDATION_PROMPT, CONSOLIDATION_ENRICHMENT),
    's2_healer':             (_HEALER_PROMPT, HEALER_INTERACTION_DEFAULT),
    's2_aspects':            (_ASPECT_PROMPT, ASPECT_INTERACTION_DEFAULT),
    'recall_laf':            ('', _RECALL_LAF_DEFAULT),
    # edge_context producer policy (top_k); backfill_vectors reads it per pass.
    'edge_context':          ('', EDGE_CONTEXT_INTERACTION_DEFAULT),
    'trace_recording':       ('', TRACE_RECORDING_NORMAL),
    'scopes':                ('', SCOPES_CONFIG_V1),
    # Decoder parameters (not an LLM template); CommunityDecoder/Encoder
    # read this through the resolver at construction.
    's2_community':          ('', COMMUNITY_DETECTION),
}


# name → callable(config: dict) → [violation strings]; empty list = valid.
# Consulted at BOTH doors of the override model: register_interaction REFUSES
# a version whose config has violations (write door), and the resolver LOGS
# violations and runs on the code default (read seam) — an invalid override
# must never become the running K, and must never crash a read path.
INTERACTION_VALIDATORS = {
    'scopes': validate_scopes_config,
    's1e_gist': validate_s1e_gist_config,
    'edge_context': validate_edge_context_config,
}


def interaction_fingerprint(name: str, template: str, config: dict) -> str:
    """12-hex sha256 over name + template + canonical(config).

    Content-address of the EFFECTIVE prompt+config a run used. The row
    pointer (interaction_id = a rowid, interaction_version = a per-install
    counter) is install-local and can dangle; the fingerprint identifies the
    K by content, so it is stable across installs and unchanged when an
    override row is collapsed into a byte-identical code default. 48 bits is
    ample for the tens of distinct Ks a brain ever runs.
    """
    h = hashlib.sha256()
    h.update((name or '').encode())
    h.update((template or '').encode())
    h.update(json.dumps(config or {}, sort_keys=True).encode())
    return h.hexdigest()[:12]
