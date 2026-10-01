"""Aspect store — byte-level core + structural contract for aspects_v1.json.

Path resolution, the required-name contract, the single atomic JSON writer,
and the structural validator every write is gated on. The semantic layer
(AspectRegistry in servers/aspects.py) is the ONE public door for reading
and writing taxonomy content — everything here is its plumbing.

This module imports nothing from servers/ so the dependency is strictly
one-directional (aspects.py → here, scales/s2 → here). The import cycle the
old placement forced (aspects.py ↔ aspect_contract.py, both via
function-local imports) is dissolved by this file existing.
"""

import json
import os
import tempfile


# ─────────────────────────────────────────────────────────────────
# Required aspects — code routes on these by string. Must always exist
# in the brain. Test asserts equivalence with aspects_v1.json keys.
# ─────────────────────────────────────────────────────────────────
REQUIRED_ASPECTS: tuple = (
    # Node-facing (used by Frame)
    'identity_bearing',     # principle / identity / vision / rule / operator types
    'episodic_anchor',      # moment / anchor_quote / user_quote / quote types
    'active_thread',        # open / tension / hypothesis / aspiration types
    'lesson_insight',       # lesson / insight / validation / reflection types
    'wisdom',               # generative subset (insight/lesson/principle/vision/reflection/meta_learning/philosophy)

    # Edge-facing (used by S2 community / consolidation / healer)
    'generic_relation',         # related, related_to (skip set in community/consolidation)
    'noise',                    # co_anchored, community_member (skip set + structural)
    'correction_improvement',   # corrects, supersedes
    'extension_refinement',     # extends, refines, elaborates
    'explanation_causation',    # explains, causes
    'dependency_flow',          # depends_on, enables
    'contradiction_conflict',   # contradicts, challenges
    'validation_evidence',      # validates, demonstrates
    'hierarchical_structure',   # part_of, supersedes_structurally
    'temporal_sequence',        # follows_from, leads_to
    'survivor_lineage',         # absorbed_into — archived→living-descendant redirect
    'settlement',               # similar_to, resolves, supersedes, … — pair already
                                # settled; consolidation suppression set (closed list)
)


# Runtime aspect state lives next to brain.db (per-operator, not in repo).
# Repo-bundled seed (SEED_ASPECTS_JSON_PATH) is the first-boot baseline;
# the registry copies seed → user dir on first load when missing. After
# that, all writes stay in the user dir — the repo seed is read-only and
# never touched by runtime.
#
# Paths are resolved at CALL time, not import time. BRAIN_DB_DIR (and the
# explicit overrides) are read on every call so a later `os.environ` change
# takes effect — IsolatedBrain sets BRAIN_DB_DIR in __enter__, AFTER this
# module is imported, and a module-level constant would freeze the live
# user-dir path and leak heals into it (observed 2026-06-16). Pinned by
# tests/test_aspects_path_isolation.py. Resolution routes through
# daemon_config.resolve_db_dir (D-13 single-location contract), which reads
# the env var first — preserving the call-time override behavior.

def aspects_json_path() -> str:
    """Path to the per-operator working aspects file (call-time resolved)."""
    explicit = os.environ.get('ASPECTS_JSON_PATH')
    if explicit:
        return explicit
    from servers.daemon_config import resolve_db_dir
    return os.path.join(resolve_db_dir(), 'aspects_v1.json')


def aspects_proposed_path() -> str:
    """Path to the per-cycle audit artifact (call-time resolved)."""
    explicit = os.environ.get('ASPECTS_PROPOSED_PATH')
    if explicit:
        return explicit
    from servers.daemon_config import resolve_db_dir
    return os.path.join(resolve_db_dir(), 'aspects_proposed.json')


# Repo seed — frozen baseline shipped with the plugin. Never written.
SEED_ASPECTS_JSON_PATH = os.path.join(
    os.path.dirname(__file__), 'scales', 's2', 'aspects_v1.json')


# ─────────────────────────────────────────────────────────────────
# Member fields — the ONE declaration per fillable field.
#
# A member of an aspect's node_types / edge_relations list is either a bare
# string or an object carrying the string under MEMBER_KEY[category] plus
# the fields declared here. Everything that knows about a field derives
# from this dict: the structural validator below, the registry's parsed
# facts (AspectRegistry._adopt / fact_of), the write door's merge
# (add_members), the seed heal (reconcile job 5), the classifier's per-field
# validation, and — through CLASSIFIER_FIELDS, which leaves out the curated
# fields — its FIELD DEFINITIONS prompt block and the decoder's
# missing-field scan. Adding a dimension is one more entry here.
#
# `aspects` is itself an entry (kind='menu'): routing a new string and
# filling a dimension on a classified one are the same mechanism to the
# encoder — a candidate lists the fields it still needs and this dict says
# what each means. It is stored as MEMBERSHIP, not as a key on the member
# object; only kind='enum' entries are stored on the object.
#
# The prose here is rendered verbatim into the classifier's user message,
# so it is written for a stateless reader who sees each example drawn as
#   [type] "source title" --value--> [type] "target title"
# and nothing else. `value` is the candidate string; source/target are the
# left/right node of that drawing. Keep every term one the encoder writes.
# ─────────────────────────────────────────────────────────────────
MEMBER_KEY = {'node_types': 'type', 'edge_relations': 'verb'}

MEMBER_FIELDS = {
    'aspects': {
        'applies_to': ('node_types', 'edge_relations'),
        'kind': 'menu',
        'meaning': 'the aspect names this value belongs to, primary first, '
                   'chosen from the ASPECT MENU above.',
    },
    'orientation': {
        'applies_to': ('edge_relations',),
        'kind': 'enum',
        'meaning': 'which node does the work the value names.',
        'values': {
            'source_actor':
                'the source acts on the target.   A --corrects--> B: A corrects B.   '
                'Also `enables`, `explains`, `supersedes`, `depends_on` (A does the depending).',
            'target_actor':
                'the target acts on the source; the value is in inverse voice.   '
                'A --corrected_by--> B: B corrected A.   '
                'Also `caused_by`, `evolved_from`, `absorbed_into`, `rejected_for` '
                '(A was set aside and B kept: B is the one that acts).',
            'symmetric':
                'swapping source and target keeps the claim.   A --similar_to--> B.   '
                'Also `contradicts`, `co_occurred_with`, `same_root_cause`.',
        },
        'rule': 'Decide from the value\'s plain meaning first, then read each example '
                'as a sentence, source first, and ask who DOES the work the value '
                'names — not who is the grammatical subject. The drawing is the stored '
                'row, and a row can be written backwards, so the examples confirm a '
                'reading; they do not decide it. A `_by` suffix is a strong hint for '
                'target_actor, not the rule: `preferred_over` has no suffix and the '
                'source is the one kept; `rejected_for` has no suffix and the target is '
                'the one kept. For choice-and-replacement values the actor is the node '
                'that is kept. An example that disagrees with the value\'s meaning is a '
                'backwards row, not a symmetric value — say so in the rationale.',
    },
    # `within` stores the field only on the members of these aspects: a
    # multi-homed verb carries it in those homes and nowhere else. `curated`
    # keeps it out of the classifier loop — no fill-lane scan, no FIELD
    # DEFINITIONS entry; its values are set by hand in the repo seed and reach
    # existing brains through the seed heal. The values are listed strongest
    # first: the correction render ranks by this order.
    'flavor': {
        'applies_to': ('edge_relations',),
        'within': ('correction_improvement',),
        'curated': True,
        'kind': 'enum',
        'meaning': 'what the value did to the claim of the node it acts on (the '
                   'node orientation does not name as the actor), and so what a '
                   'reader of that node does next.',
        'values': {
            'replaced':
                'the acted-on claim no longer stands; the actor holds what replaced it.   '
                'A --supersedes--> B: B is replaced.   Also `absorbed_into`, '
                '`consolidated_into`, `overrides`, `rejected_for`.',
            'corrected':
                'the acted-on claim was wrong in whole or in part; the actor says what '
                'was wrong.   A --corrects--> B.   Also `fixes`, `falsifies`, `retracts`.',
            'resolved':
                'the acted-on question or problem is closed; the actor holds the answer '
                'or the fix.   A --resolves--> B.   Also `resolved_by`, `settles`, '
                '`partially_resolves` (in part).',
            'reframed':
                'the acted-on claim still stands, read through the actor\'s frame or '
                'scope.   A --reframes--> B.   Also `revises`, `narrows`, `redefines`.',
            'unchanged':
                'nothing in the acted-on claim has changed yet: the actor works on it, '
                'proposes a fix or flags it, and the pair reads as an ordinary edge.   '
                'A --addresses--> B.   Also `proposes_fix_for`, `may_address`, '
                '`should_have_read`.',
        },
        'rule': 'Judge from the reader of the acted-on node: what must they do '
                'differently now? A verb with no flavor yet renders as the generic '
                'warning.',
    },
}


def stored_fields(category: str, aspect: str = None) -> tuple:
    """The MEMBER_FIELDS names stored ON a member object of this category —
    the enum fields that apply to it, and with `aspect` given only those its
    `within` admits there. `aspects` (kind='menu') is membership, never an
    object key."""
    return tuple(name for name, spec in MEMBER_FIELDS.items()
                 if spec['kind'] == 'enum' and category in spec['applies_to']
                 and (aspect is None or aspect in spec.get('within', (aspect,))))


# The fields the classifier loop owns: everything MEMBER_FIELDS declares
# except the curated ones. Its FIELD DEFINITIONS block and the decoder's fill
# lane read this view; its per-field validation reads MEMBER_FIELDS whole, so
# a curated value the model writes anyway is dropped with a warning.
CLASSIFIER_FIELDS = {name: spec for name, spec in MEMBER_FIELDS.items()
                     if not spec.get('curated')}


def filled_fields(category: str) -> tuple:
    """The stored fields of this category the classifier fills — what the
    decoder's fill lane asks for when one is missing."""
    return tuple(name for name in stored_fields(category)
                 if name in CLASSIFIER_FIELDS)


def new_member_fill(category: str) -> list:
    """The fields a string with no aspect home yet must fill: `aspects` plus
    every enum field the classifier fills for its category. The decoder
    writes this onto the candidate and the encoder validates against it —
    one definition."""
    return ['aspects'] + list(filled_fields(category))


def field_values(field: str) -> tuple:
    """Closed value list of an enum field; () for a menu field."""
    return tuple((MEMBER_FIELDS.get(field) or {}).get('values') or ())


def member_name(member, category: str):
    """The string a member carries — itself for a bare string, its
    MEMBER_KEY value for an object. None when malformed (the validator
    reports it; readers skip it)."""
    if isinstance(member, str):
        return member or None
    if isinstance(member, dict):
        name = member.get(MEMBER_KEY[category])
        return name if isinstance(name, str) and name else None
    return None


def member_facts(member, category: str) -> dict:
    """{field: value} stored on a member object; {} for a bare string.
    Restricted to stored_fields(category) — an undeclared key is not a
    fact, it is a violation the validator names."""
    if not isinstance(member, dict):
        return {}
    return {k: member[k] for k in stored_fields(category)
            if member.get(k) is not None}


def validate_taxonomy(data) -> list:
    """Structural invariants any taxonomy copy must satisfy. Returns a list of
    violation strings — empty means valid.

    This is the STRUCTURAL set, shared by the seed and the working copy: it
    is what a write must never produce, so the registry's door refuses on it
    and the boot load reports it. Seed-only CURATION standards (locked flags,
    meaning length, display labels, no-extra-aspects) live in
    tests/test_aspects_contract.py — the working copy legitimately grows
    emergent unlocked aspects, so the gate must not enforce those.

    Invariants:
      1. every entry is an object
      2. node_types / edge_relations, when present, are lists of members —
         a non-empty string, or an object carrying the string under
         MEMBER_KEY[category] plus only the enum fields MEMBER_FIELDS
         declares for that category, each value from its closed list.
         Uniqueness is keyed on the member's string, whatever its shape.
         A member listed in several aspects carries ONE value per field
         across all of them (multi-homed agreement).
      3. every REQUIRED_ASPECTS name is present
      4. per-aspect fact fields, when present, are well-shaped: `accepts`
         is a non-empty list drawn from {node_types, edge_relations};
         `routable` / `prompt_visible` are booleans.
         Presence itself is a seed-curation standard (tests), not a gate —
         emergent aspects and pre-heal working copies legitimately lack them.

    Noise-exclusivity is deliberately NOT an invariant.
    Noise vetoes instead: a string in noise is machinery whatever else claims
    it, enforced in the registry rather than forbidden here. Requiring
    disjointness made the additive heal refuse whole reconciles — a seed
    member colliding with a classifier-grown working copy left every divergent
    install stuck on its old taxonomy, silently, with no migration lane to
    reach it. It was also justified by an exclusion filter reading "not in
    noise" as "is real knowledge", which never shipped — the static verb list
    it would have replaced was deleted unread instead.

    Keys starting with '_' are reserved for in-file documentation
    (`_schema`) — JSON has no comments. Skipped here and by every reader
    (registry _adopt, dashboard); refused as write targets by add_members.
    """
    violations = []
    if not isinstance(data, dict):
        return ['taxonomy root is %s, expected object' % type(data).__name__]
    facts_seen = {}   # (category, member, field) → (value, first aspect)

    for name, entry in data.items():
        if name.startswith('_'):
            continue
        if not isinstance(entry, dict):
            violations.append("aspect '%s': entry is %s, expected object"
                              % (name, type(entry).__name__))
            continue
        for category in ('node_types', 'edge_relations'):
            members = entry.get(category)
            if members is None:
                continue
            if not isinstance(members, list):
                violations.append("aspect '%s'.%s: %s, expected list"
                                  % (name, category, type(members).__name__))
                continue
            seen = set()
            for m in members:
                mname = member_name(m, category)
                if mname is None:
                    violations.append("aspect '%s'.%s: malformed member %r"
                                      % (name, category, m))
                    continue
                if mname in seen:
                    violations.append("aspect '%s'.%s: duplicate member '%s'"
                                      % (name, category, mname))
                seen.add(mname)
                if not isinstance(m, dict):
                    continue
                allowed = (MEMBER_KEY[category],) + stored_fields(category, name)
                extra = sorted(k for k in m if k not in allowed)
                if extra:
                    violations.append(
                        "aspect '%s'.%s: member '%s' carries undeclared keys %s "
                        "(MEMBER_FIELDS declares %s)"
                        % (name, category, mname, extra, list(allowed[1:])))
                for fld, val in member_facts(m, category).items():
                    if val not in field_values(fld):
                        violations.append(
                            "aspect '%s'.%s: member '%s'.%s = %r, expected one of %s"
                            % (name, category, mname, fld, val,
                               list(field_values(fld))))
                    else:
                        prior = facts_seen.setdefault((category, mname, fld),
                                                      (val, name))
                        if prior[0] != val:
                            violations.append(
                                "member '%s'.%s disagrees across homes: %r in '%s', "
                                "%r in '%s' — one value per member"
                                % (mname, fld, prior[0], prior[1], val, name))
        accepts = entry.get('accepts')
        if accepts is not None:
            if (not isinstance(accepts, list) or not accepts
                    or any(a not in ('node_types', 'edge_relations')
                           for a in accepts)
                    or len(set(accepts)) != len(accepts)):
                violations.append(
                    "aspect '%s'.accepts: %r — expected a non-empty unique "
                    "subset of ['node_types', 'edge_relations']" % (name, accepts))
        for flag in ('routable', 'prompt_visible'):
            val = entry.get(flag)
            if val is not None and not isinstance(val, bool):
                violations.append("aspect '%s'.%s: %r, expected boolean"
                                  % (name, flag, val))

    missing = [n for n in REQUIRED_ASPECTS if n not in data]
    if missing:
        violations.append('required aspects missing: %s' % missing)

    return violations


def atomic_json_write(path: str, data) -> None:
    """The ONE atomic JSON writer for aspect files: temp file + os.replace.

    A crash mid-write can't leave a truncated/0-byte file — a corrupt working
    copy loads as an empty registry → relations_in(['survivor_lineage'])
    returns () → the absorbed_into exemption silently disables and the reaper
    scrubs redirect edges.

    One canonical dump shape (indent=2, ensure_ascii=False, trailing newline)
    so the file stops interleaving two writers' escaping styles.
    """
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + '_',
                               suffix='.tmp', dir=d)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write('\n')
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
