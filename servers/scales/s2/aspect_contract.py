"""S2 Aspect Integration — Contract and Configuration.

Classifies distinct node types and edge relations into the required
aspects defined in `aspects_v1.json`. Closed-list classification — encoder
can only route to existing aspects (or `noise`/`generic_relation` as
catch-alls), never proposes new ones. New aspects are added by humans
editing `aspects_v1.json`.

The unit reads the taxonomy via brain.aspects and writes through its door
(AspectRegistry.add_members); it never mutates brain state. No suppression
machinery — closed list means every string gets a home, so SKIP isn't a
valid encoder action.

Path resolution and the seed-reconcile live in servers/aspect_store.py and
servers/aspects.py (re-exported here for existing importers).
"""

from servers.aspect_store import (  # noqa: F401 — canonical home; re-exported
    MEMBER_FIELDS,
    REQUIRED_ASPECTS,
    SEED_ASPECTS_JSON_PATH,
    aspects_json_path,
    aspects_proposed_path,
)


ASPECT = {
    # LLM config
    # Opus 5.5 reads the act a value names where Sonnet 4.6 falls back to the
    # preposition (act + object + `from` values, `operates_within`), on the
    # same prompt and batches. Effort is set, not left to the API default.
    'model': 'claude-opus-5-5',
    'effort': 'medium',
    # ~100 output tokens per entry plus thinking, which counts against the
    # cap; 32k clears a 50-fill + 30-route batch with headroom. A max_tokens
    # stop is logged by _call_llm as truncation.
    'max_tokens': 32000,

    # Batch sizing — chosen for clean per-item attention without losing
    # the cross-string visibility that helps consistency (e.g., `corrects`
    # and `corrected_by` classified together).
    'max_candidates_per_call': 30,

    # Batch gate — the decoder calls the encoder only when enough work is
    # waiting: at least `min_pending_to_call` candidates (unclassified strings
    # plus unfilled members), or one of them already written
    # `call_now_at_count` times. Every call re-sends the instructions and the
    # menu (~15k tokens), and calls hours apart pay that write in full each
    # time, so a one-verb call costs nearly what a thirty-verb call does.
    # Once a cycle has called, its later passes drain what is left: the
    # prefix is cached for those seconds-apart calls.
    'min_pending_to_call': 20,
    'call_now_at_count': 5,

    # Calls per S2 cycle. The orchestrator re-decodes between calls (the
    # registry re-derives on every write, so a batch never re-proposes what
    # the previous one filled) and stops early when nothing is left or a
    # call fails. Calls seconds apart share the system prompt and the menu
    # block through the five-minute cache breakpoints run_llm_once places.
    'max_calls_per_cycle': 6,

    # Menu width: each aspect shows its most-used members, ranked by how
    # often this brain writes them. The menu is for routing consistency
    # (see the aspect's shape), not a membership check — the decoder never
    # proposes a string that is already filed — so a sample serves, keeps
    # the prompt independent of brain size (1,600+ verbs in a grown brain),
    # and stays byte-identical across batches for the cache.
    'menu_members_per_aspect': 20,

    # Decoder filters
    # 1 = classify EVERY string, singletons included. An unaspected string is
    # invisible to every aspect-driven consumer (correction_enrich, the wisdom
    # Frame pull, noise filtering); a one-off typo filed under `noise` is the
    # cheaper failure. Most strings arrive exactly once, so this is the floor,
    # not a tuning knob.
    'min_count_threshold': 1,
    'examples_per_candidate': 3,     # nodes/edges shown per candidate string

    # Fill lane — classified members missing a MEMBER_FIELDS value
    # (`orientation` today). They ride behind the unclassified strings, up
    # to this many per call, so a large backlog (every verb on the day a
    # field is declared) never starves new vocabulary of its aspect home.
    # Fill candidates are cheaper to judge than routing (homes fixed, one
    # field), so the cap is wider — but the backlog is count-sorted, so every
    # later batch judges from thinner evidence (often one example); 50 keeps
    # per-item attention on that tail. Set to 1–2 for a supervised trial: one
    # audit row per call to read.
    'max_fill_per_call': 50,
}

# Interaction config default for the `s2_aspects` K — sliced from ASPECT so
# model/effort/max_tokens have exactly one home in this file.
ASPECT_INTERACTION_DEFAULT = {
    'model': ASPECT['model'],
    'effort': ASPECT['effort'],
    'max_tokens': ASPECT['max_tokens'],
}

