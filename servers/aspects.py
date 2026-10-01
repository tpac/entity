"""Aspect Registry — first-class roles/groupings for nodes and edges.

An aspect is a semantic role (correction, identity_bearing, active_thread, ...)
that groups node TYPES and/or edge RELATIONS under a shared meaning. Aspects
live in aspects_v1.json — each entry carries its own member lists (the
node_types / edge_relations it claims). The registry reads that file directly;
aspects are NOT brain nodes.

Two tiers:
  - Required aspects (locked): code routes on these by string. Must always
    exist. Listed in REQUIRED_ASPECTS — single source of truth for the
    required NAMES, mirrored by aspects_v1.json keys (test enforces
    equivalence). On boot the registry validates they are present in the JSON
    and logs loudly if any are missing (no auto-heal — the JSON is the spec).
  - Emergent aspects (unlocked): discovered by S2's AspectIntegration unit
    as the brain accumulates new types/relations. Created freely.

Consumers (every name string in the codebase flows through here):
  - Graph reads and traversal: brain.aspects.structural_exclusions /
    traversal_exclusions (pipeline_contract.traverse, recall, surface, the
    consolidation decoder)
  - Surface: compose_edge_text builds the embedded edge text (also
    brain_connections)
  - Corrections: brain.aspects.correction_improvement, fact_of / actor_of
    (brain_corrections, the consolidation decoder)
  - S2 community/consolidation: relations_in([...]), primary_edge_map,
    settlement, contradiction_conflict
  - S2 classifier: per-aspect facts (routable/accepts) drive menu +
    validation; add_members is its write door
  - Encoder prompts: all() feeds the edge-aspect vocabulary block
    (s2/base._inject_edge_aspects, used by the community and consolidation
    encoders)

from_dict() constructs a registry without a brain (for tests/seeding); the
production path is _load(), which reads aspects_v1.json directly.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Iterable, NoReturn, Optional

# Required-name contract lives in the byte-level core; re-exported here so
# consumers keep one import surface (servers.aspects) for taxonomy names.
from .aspect_store import (
    MEMBER_KEY,
    REQUIRED_ASPECTS,  # noqa: F401 — re-export
    SEED_ASPECTS_JSON_PATH,
    aspects_json_path,
    atomic_json_write,
    member_facts,
    member_name,
    stored_fields,
    validate_taxonomy,
)


# ─────────────────────────────────────────────────────────────────
# Edge-aspect prompt block — single source for every encoder that offers the
# edge-relation vocabulary in its prompt (S2 consolidation + community via
# IntegrationUnit._inject_edge_aspects; S1E/surface may adopt). Mirrors the
# journal block's render-fn-+-base-method split.
# ─────────────────────────────────────────────────────────────────


# Relations each aspect shows in the encoders' edge block, most used first.
EDGE_ASPECT_PROMPT_MEMBERS = 16


def rank_members(members, counts, cap):
    """(label, rendered list) for one aspect's member line: the `cap` most
    written members first, ties by name. The label says what the sample is —
    'top 20 of 412 by use' when counts rank it, 'first 20 of 412' without
    counts, 'all 8' when nothing is hidden — so the reader never takes the
    line for the full list. Shared by the S2 aspect menu and the edge block."""
    ranked = sorted(members, key=lambda m: (-counts.get(m, 0), m))
    shown = ranked[:cap]
    if len(members) <= cap:
        label = 'all %d' % len(members)
    elif counts:
        label = 'top %d of %d by use' % (len(shown), len(members))
    else:
        label = 'first %d of %d' % (len(shown), len(members))
    return label, ', '.join(shown)


def render_edge_aspects_block(aspects, counts=None):
    """Render the edge-relation aspect vocabulary as a prompt block so an encoder
    picks specific relations over generic ones. `aspects` is the name→Aspect dict
    from brain.aspects.all(); `counts` is {relation: live edge count}
    (GraphDAL.relation_counts). Skips prompt-invisible aspects (the per-aspect
    `prompt_visible` fact in aspects_v1.json — structural/system aspects declare
    false) and node-only aspects (no edge relations); each shown aspect lists its
    EDGE_ASPECT_PROMPT_MEMBERS most-used relations (rank_members). Returns ''
    when there's nothing to show.

    Noise and survivor_lineage members are dropped from every list.
    `prompt_visible` is per-ASPECT, so it cannot say "this one relation is
    machinery" — and since noise vetoes, a relation offered here that also sits
    in noise would be taught to the encoder and then filtered on read. Teaching
    a verb we discard is worse than not teaching it. survivor_lineage's
    absorbed_into is written only by the absorb op; the seed also files it
    under correction_improvement for correction walks, where ranked by use it
    would lead the list and offer the encoder a verb it must never write.
    """
    counts = counts or {}
    veto = set()
    for system_aspect in ('noise', 'survivor_lineage'):
        if aspects.get(system_aspect):
            veto.update(aspects[system_aspect].edge_relations)
    lines = []
    for name, aspect in sorted(aspects.items()):
        if not aspect.prompt_visible or not aspect.edge_relations:
            continue
        members = [r for r in aspect.edge_relations if r not in veto]
        if not members:
            continue
        label, shown = rank_members(members, counts, EDGE_ASPECT_PROMPT_MEMBERS)
        lines.append('- **%s** (%s): %s' % (name, label, shown))
    if not lines:
        return ''
    return ('## Edge Aspects (%d from brain.aspects)\n\n%s\n\n'
            'Avoid `related_to` — pick a specific relation.' % (
                len(lines), '\n'.join(lines)))


# Per-aspect fact fields (Step 4) — healed from the seed when a REQUIRED
# aspect's working-copy entry lacks them (reconcile job 4). The registry
# loads the WORKING copy, so without this heal an existing brain never
# receives the facts and every derived consumer (classifier routing, prompt
# visibility) runs on defaults while seed-based tests pass.
ASPECT_FACT_KEYS = ('accepts', 'routable', 'prompt_visible')


def reconcile_working_copy(log_fn=None) -> bool:
    """Seed the working aspects file from the repo seed, and SELF-HEAL.

    Four jobs, all idempotent and safe to call on every boot:
      1. First boot — working copy missing → copy the whole seed (atomic).
      2. Missing aspect — the seed has a REQUIRED aspect the working copy
         lacks → add the whole aspect from the seed.
      3. Missing member — a REQUIRED aspect's seed `node_types` /
         `edge_relations` list names a string the working copy's list lacks
         → APPEND it. This is how a curated membership fix (e.g. multi-homing
         a replacement verb into correction_improvement, which recall walks)
         reaches an existing brain. Without it a seed member edit propagates
         to fresh installs only, and every existing brain keeps the defect
         while the seed-based contract tests pass.
      4. Missing fact field — a REQUIRED aspect's working-copy entry lacks
         one of ASPECT_FACT_KEYS that the seed carries → copy the seed's
         value. A PRESENT value is never overwritten (additive, like
         members): a deliberate working-copy divergence survives; only
         omissions heal.
      5. Missing member field — a seed member carries a MEMBER_FIELDS value
         (e.g. `orientation`) that the working copy's same member lacks →
         fill it, upgrading a bare string to an object. Same additive rule:
         a present value, agreeing or not, is kept. Members are matched by
         their STRING whatever their shape — a seed object and a working-copy
         string for the same verb are one member, never a job-3 gap.

    An UNPARSEABLE working copy is quarantined (renamed aside, preserving the
    operator's classifier-grown members for manual recovery) and re-seeded —
    the old return-False path left the registry permanently empty on every
    subsequent boot, which its own comments called catastrophic.

    ADDITIVE ONLY, in both directions of scope:
      · Members are appended, never reordered or removed — operator- and
        AspectIntegration-grown lists survive.
      · A seed REMOVAL does not propagate. Retiring a member (or moving one
        between aspects) still needs a supervised migration; this heals
        omissions, not disagreements.
      · Only REQUIRED aspects are touched. Emergent/unlocked aspects are the
        classifier's to own.

    Never writes the seed itself (tests may point the working path at it).
    Returns True if the file was created or modified. `log_fn(message)` — when
    given — is called with a one-line summary of what was healed: a member heal
    silently changes which edges `correction_enrich` walks, so it announces
    itself rather than being inferred from behaviour later.
    """
    json_path = aspects_json_path()
    if not os.path.exists(SEED_ASPECTS_JSON_PATH):
        return False
    # Never heal the seed into itself.
    if os.path.abspath(json_path) == os.path.abspath(SEED_ASPECTS_JSON_PATH):
        return False
    try:
        with open(SEED_ASPECTS_JSON_PATH) as f:
            seed = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        # Broken plugin install — nothing trustworthy to reconcile from.
        if log_fn:
            try:
                log_fn('repo seed unreadable — reconcile skipped: %r' % e)
            except Exception:
                pass
        return False
    seed_violations = validate_taxonomy(seed)
    if seed_violations:
        # A structurally invalid seed must not be copied or healed from —
        # refuse the whole reconcile rather than propagate the breakage.
        if log_fn:
            try:
                log_fn('repo seed fails structural validation — reconcile '
                       'skipped: %s' % '; '.join(seed_violations))
            except Exception:
                pass
        return False
    if not os.path.exists(json_path):
        atomic_json_write(json_path, seed)
        return True

    # Working copy exists — self-heal any missing REQUIRED aspect from the seed.
    try:
        with open(json_path) as f:
            cur = json.load(f)
    except json.JSONDecodeError:
        # Corrupt working copy (e.g. a partial first-boot copy from before the
        # write was atomic). Quarantine it — the operator's classifier-grown
        # members live in there and deserve a recovery path — then re-seed.
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')  # clock-ok — quarantine filename
        quarantine = '%s.corrupt-%s' % (json_path, stamp)
        os.replace(json_path, quarantine)
        atomic_json_write(json_path, seed)
        summary = ('working copy unparseable — quarantined to %s and re-seeded '
                   'from the repo baseline. Classifier-grown members are in the '
                   'quarantined file; recover manually.' % quarantine)
        print('[aspects] %s' % summary, flush=True)
        if log_fn:
            try:
                log_fn(summary)
            except Exception:
                pass
        return True
    except OSError as e:
        # Working copy exists but can't be read (permissions, I/O). Loud on
        # both channels — a silent False here would look identical to
        # "nothing to heal" while the registry boots from a file the heal
        # couldn't even open.
        msg = 'working copy unreadable — reconcile skipped: %r' % e
        print('[aspects] %s' % msg, flush=True)
        if log_fn:
            try:
                log_fn(msg)
            except Exception:
                pass
        return False
    missing = [n for n in REQUIRED_ASPECTS if n in seed and n not in cur]
    # In-file documentation keys (_schema) travel with job 2: copied when
    # absent so working copies stay self-documenting. Additive like all
    # heals — an existing doc entry is never overwritten.
    missing_docs = [k for k in seed if k.startswith('_') and k not in cur]
    # Missing MEMBERS of required aspects that both files carry (job 3).
    # Shape-guarded the same way AspectRegistry._load guards: a working copy
    # can carry a malformed entry (hand-edit, partial write), and an unguarded
    # deref here would abort the WHOLE heal — including job 2 above, whose
    # failure leaves survivor_lineage empty and silently disables the
    # absorbed_into archive exemption. A malformed entry is skipped, not fatal.
    member_heals = []          # (aspect, category, [members]) — for the caller's log
    skipped_malformed = []
    for n in REQUIRED_ASPECTS:
        if n not in seed or n not in cur:
            continue           # job 1/2 territory, or not seeded at all
        if not isinstance(cur[n], dict) or not isinstance(seed[n], dict):
            skipped_malformed.append(n)
            continue
        for category in ('node_types', 'edge_relations'):
            have = cur[n].get(category)
            want = seed[n].get(category)
            if have is None:
                have = []      # absent key is a legitimate empty list
            if not isinstance(have, list) or not isinstance(want or [], list):
                skipped_malformed.append('%s.%s' % (n, category))
                continue
            have_names = {member_name(m, category) for m in have}
            gap = [m for m in (want or [])
                   if member_name(m, category) not in have_names]
            if gap:
                member_heals.append((n, category, gap))
    # Missing FACT fields of required aspects (job 4).
    fact_heals = []            # (aspect, [keys]) — for the caller's log
    for n in REQUIRED_ASPECTS:
        if n not in seed or n not in cur:
            continue
        if not isinstance(cur[n], dict) or not isinstance(seed[n], dict):
            continue           # already reported by job 3's shape guard
        gap = [k for k in ASPECT_FACT_KEYS
               if k not in cur[n] and k in seed[n]]
        if gap:
            fact_heals.append((n, gap))
    # Jobs 1–4 are applied in memory first; nothing is written until the
    # merged result validates below.
    for n in missing:
        cur[n] = seed[n]
    for k in missing_docs:
        cur[k] = seed[k]
    for n, category, gap in member_heals:
        cur[n].setdefault(category, []).extend(gap)   # append — never reorder
    for n, gap in fact_heals:
        for k in gap:
            cur[n][k] = seed[n][k]                    # fill — never overwrite
    # Missing MEMBER fields (job 5) — computed on the post-job-3 copy so a
    # member job 3 just appended as an object already counts as a home.
    # A field is a fact about the MEMBER, not about one home: the working
    # copy's homes are read together, and a value present in ANY of them —
    # agreeing with the seed or not — means the member has its value and is
    # left alone. Only a member with no value anywhere is filled, into every
    # home the field's scope admits at once, so the heal can never
    # manufacture the cross-home disagreement the validator refuses (which would REFUSE the whole heal
    # and boot the registry on the un-healed copy).
    cur_homes = {}             # (category, member) → [(aspect, index)]
    cur_facts = {}             # (category, member) → merged {field: value}
    for n, entry in cur.items():
        if n.startswith('_') or not isinstance(entry, dict):
            continue
        for category in ('node_types', 'edge_relations'):
            members = entry.get(category)
            if not isinstance(members, list):
                continue
            for i, m in enumerate(members):
                mname = member_name(m, category)
                if not mname:
                    continue
                cur_homes.setdefault((category, mname), []).append((n, i))
                merged = cur_facts.setdefault((category, mname), {})
                for k, v in member_facts(m, category).items():
                    merged.setdefault(k, v)
    # The seed's side is read the same way, merged across its homes: a field
    # scoped by `within` sits on only some of a member's seed homes, so the
    # first home listing the member is not the whole of what it carries.
    want_facts = {}            # (category, member) → merged {field: value}
    for n in REQUIRED_ASPECTS:
        if not isinstance(seed.get(n), dict):
            continue
        for category in ('node_types', 'edge_relations'):
            want = seed[n].get(category)
            if not isinstance(want, list):
                continue
            for m in want:
                mname = member_name(m, category)
                if not mname:
                    continue
                merged = want_facts.setdefault((category, mname), {})
                for k, v in member_facts(m, category).items():
                    merged.setdefault(k, v)
    # A scoped field fills only the homes its `within` admits — written
    # anywhere else it would be the violation that refuses the whole heal.
    field_heals = []           # (category, member, [(aspect, index, {field: value})])
    for (category, mname), facts in want_facts.items():
        have = cur_facts.get((category, mname), {})
        fill = {k: v for k, v in facts.items() if k not in have}
        homes = []
        for n, i in cur_homes.get((category, mname), ()):
            here = {k: v for k, v in fill.items() if k in stored_fields(category, n)}
            if here:
                homes.append((n, i, here))
        if homes:
            field_heals.append((category, mname, homes))
    if (not missing and not missing_docs and not member_heals
            and not fact_heals and not field_heals):
        if skipped_malformed and log_fn:
            try:
                log_fn('working copy has malformed entries, skipped: %s'
                       % ', '.join(skipped_malformed))
            except Exception:
                pass
        return False
    for category, mname, homes in field_heals:
        for n, i, here in homes:
            ours = cur[n][category][i]
            obj = ours if isinstance(ours, dict) else {MEMBER_KEY[category]: mname}
            obj.update(here)                          # fill — never overwrite
            cur[n][category][i] = obj
    heal_violations = validate_taxonomy(cur)
    if heal_violations:
        # The healed result would be structurally invalid (e.g. a seed member
        # edit colliding with a classifier-grown noise member breaks
        # noise-exclusivity). Refuse the WRITE, keep the prior file, boot
        # continues on the un-healed copy. Loud on both channels — this needs
        # a human to reconcile the seed edit with the working copy.
        summary = ('heal REFUSED — healed result fails structural validation '
                   '(NOTE: an unhealed pre-Step-4 copy loads with conservative '
                   'derived-policy defaults — classifier inert, encoder '
                   'vocabulary blocks empty — until a human reconciles): ' + '; '.join(heal_violations))
        print('[aspects] %s' % summary, flush=True)
        if log_fn:
            try:
                log_fn(summary)
            except Exception:
                pass
        return False
    parts = ['+aspect %s' % n for n in missing]
    parts += ['+docs %s' % k for k in missing_docs]
    parts += ['%s.%s += %s' % (n, category,
                               ','.join(member_name(m, category) for m in gap))
              for n, category, gap in member_heals]
    parts += ['%s facts += %s' % (n, ','.join(gap))
              for n, gap in fact_heals]
    # One line per set of fields written, naming only the homes that got it —
    # a scoped field lands in fewer homes than an unscoped one beside it.
    for category, mname, homes in field_heals:
        by_fill = {}
        for n, _i, here in homes:
            by_fill.setdefault(tuple(here.items()), []).append(n)
        parts += ['%s %s += %s in %s' % (
                      category, mname,
                      ','.join('%s=%s' % kv for kv in fill),
                      ','.join(names))
                  for fill, names in by_fill.items()]
    if skipped_malformed:
        parts.append('SKIPPED malformed: %s' % ', '.join(skipped_malformed))
    summary = 'healed working copy from seed: ' + '; '.join(parts)
    atomic_json_write(json_path, cur)
    # Announce AFTER the write lands, so a failed write can't leave a log line
    # claiming a heal that didn't happen. Two channels on purpose: stdout is the
    # only one that reliably reaches a log at Brain.__init__ time (a short-lived
    # process whose logs-db write hits `database is locked` degrades to a stderr
    # nobody reads), and log_fn gives the daemon a queryable row.
    print('[aspects] %s' % summary, flush=True)
    if log_fn:
        try:
            log_fn(summary)
        except Exception:
            pass                                      # never break boot on a log
    return True


class AspectContractError(Exception):
    """Raised by __getattr__ when `brain.aspects.<name>` references an aspect
    not present in aspects_v1.json.

    Code that calls `registry.<aspect_name>` for a non-required (emergent)
    aspect should use registry.by_name() instead, which returns
    Optional[Aspect].
    """


# ─────────────────────────────────────────────────────────────────
# Aspect value object
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Aspect:
    """A single aspect — a named semantic role grouping types and/or relations.

    Frozen so callers can pass it around without worrying about mutation.
    Member tuples are tuples (not lists) for hashability + immutability.
    `metadata` is a dict by convention treated as read-only — don't mutate
    after construction. Whole-field reassignment is blocked by the frozen
    dataclass; in-place dict mutation is on the convention layer.
    """

    name: str
    node_types: tuple = ()
    edge_relations: tuple = ()
    # {verb: {field: value}} for the members of edge_relations that carry
    # MEMBER_FIELDS facts (aspect_store). A bare-string member has no entry.
    # Derived at _adopt from the member objects; the tuples above stay
    # strings so every existing consumer reads the same shape it always did.
    relation_facts: dict = field(default_factory=dict)
    meaning: str = ''
    dimension: str = 'semantic'
    locked: bool = False
    metadata: dict = field(default_factory=dict)
    # Per-aspect facts (aspects_v1.json, Step 4) — replace the deleted Python
    # literals ASPECT_ACCEPTS / order / EDGE_ASPECT_PROMPT_SKIP. Every
    # required aspect carries explicit values (seed + reconcile job 4); the
    # defaults exist for entries WITHOUT facts (emergent aspects, or a
    # pre-Step-4 working copy loaded after a REFUSED heal) and are chosen to
    # degrade CONSERVATIVELY, not to mirror the old literals:
    # routable=False (classifier goes inert + loud rejections, never
    # mis-routes), prompt_visible=False (vocabulary block goes quiet, never
    # offers noise verbs).
    accepts: tuple = ()             # ('node_types',) / ('edge_relations',) / both
    routable: bool = False          # may the S2 classifier route strings here?
    prompt_visible: bool = False    # shown in encoder prompt vocabulary blocks?

    def __contains__(self, item: str) -> bool:
        return item in self.node_types or item in self.edge_relations

    @property
    def member_count(self) -> int:
        return len(self.node_types) + len(self.edge_relations)


# ─────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────


class AspectRegistry:
    """First-class API for the aspects system — the ONE door, reads AND writes.

    Production: instantiated once at Brain.__init__, exposed as brain.aspects.
    Construction reconciles the working copy with the repo seed (first-boot
    copy + additive heal — see reconcile_working_copy), then loads and
    validates. All writes to aspects_v1.json go through this object
    (add_members for the classifier, clear_fact for curation); every write
    re-derives the in-memory maps, so the cache cannot go stale and needs no invalidation protocol.

    Testing/seeding: use from_dict() to construct without a brain — bypasses
    the load path, takes a pre-built dict of aspect specs.
    """

    def __init__(self, brain):
        self._brain = brain
        self._aspects: dict = {}
        self._reverse_node: dict = {}   # type_string → aspect_name
        self._reverse_edge: dict = {}   # relation_string → aspect_name
        try:
            reconcile_working_copy(log_fn=self._heal_log)
        except Exception as e:
            try:
                self._brain._log_warning(
                    'aspect_registry_seed',
                    'failed to seed user-dir aspects_v1.json from repo baseline',
                    repr(e))
            except Exception:
                pass
        self._load()

    def _heal_log(self, msg: str) -> None:
        try:
            self._brain._log_warning('aspect_registry_heal', msg)
        except Exception:
            pass

    # ── Loading + writing (the door) ──

    def _load(self) -> None:
        """Load aspects from aspects_v1.json — a PURE read, no side effects.

        Seed materialization / self-heal is NOT here — it happens once at
        __init__, so the read path never writes the file it reads.

        Per-operator state (2026-05-17): the working file lives next to
        brain.db ($BRAIN_DB_DIR/aspects_v1.json), not in the repo. On
        first boot the repo seed is copied to the user dir; all subsequent
        writes stay there. The repo file is the shipped baseline, never
        touched by runtime.

        Multi-membership: a string can appear in multiple aspects' member
        lists. Reverse maps (_reverse_node, _reverse_edge) store the FIRST
        aspect that claimed the string in JSON-iteration order — preserves
        the prior single-aspect API contract for `by_node_type`/`by_edge_relation`
        while letting the underlying data carry richer multi-aspect membership.
        """
        ASPECTS_JSON_PATH = aspects_json_path()

        self._adopt({})

        if not os.path.exists(ASPECTS_JSON_PATH):
            try:
                self._brain._log_warning(
                    'aspect_registry_load',
                    'aspects_v1.json missing — registry empty',
                    ASPECTS_JSON_PATH)
            except Exception:
                pass
            return

        try:
            with open(ASPECTS_JSON_PATH, 'r') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            try:
                self._brain._log_warning(
                    'aspect_registry_load',
                    'failed to parse aspects_v1.json — registry empty',
                    repr(e))
            except Exception:
                pass
            return

        # Loud, non-fatal: a structurally invalid working copy still boots
        # (a refusing boot would brick the brain) but every violation is
        # reported — _adopt's shape coercion no longer hides anything. The
        # WRITE side (add_members / reconcile) refuses on the same validator,
        # so violations here mean a hand-edit or pre-gate legacy state.
        for v in validate_taxonomy(data):
            try:
                self._brain._log_error(
                    'aspect_contract',
                    Exception('taxonomy structural violation'), v)
            except Exception:
                pass  # never let logging block the load

        self._adopt(data)

    def _adopt(self, data: dict) -> None:
        """Build aspects + reverse-lookup maps from a taxonomy dict — the ONE
        constructor body behind both _load (file) and from_dict (tests).

        Reverse maps use setdefault: the FIRST aspect to claim a string in
        dict-iteration order wins — the documented multi-membership contract.
        The two constructors used to disagree here (from_dict let the LAST
        claimant win), which flipped the reported primary for every
        multi-homed string (`supersedes`, `absorbed_into`, the wisdom types).

        Shape-guarded: non-dict entries are skipped, malformed member lists
        coerce to empty — a hand-edited working copy degrades per-entry, not
        wholesale. This is also the seam for load-time derivations (Step 6's
        precomputed exclusion policies belong here, not per-read).
        """
        self._aspects = {}
        self._reverse_node = {}
        self._reverse_edge = {}
        for name, entry in data.items():
            if name.startswith('_') or not isinstance(entry, dict):
                continue   # '_'-prefixed keys are in-file docs (_schema)
            node_types = entry.get('node_types', []) or []
            edge_relations = entry.get('edge_relations', []) or []
            if not isinstance(node_types, list):
                node_types = []
            if not isinstance(edge_relations, list):
                edge_relations = []
            # Members are strings or objects ({MEMBER_KEY: string, field:
            # value, ...}). The name tuple is what every consumer reads; the
            # facts map is the new dimension data. A malformed member (no
            # string) is skipped here and reported by the validator.
            type_names = [n for n in (member_name(m, 'node_types')
                                      for m in node_types) if n]
            relation_names = [n for n in (member_name(m, 'edge_relations')
                                          for m in edge_relations) if n]
            relation_facts = {}
            for m in edge_relations:
                n = member_name(m, 'edge_relations')
                facts = member_facts(m, 'edge_relations') if n else {}
                if facts:
                    relation_facts[n] = facts

            accepts = entry.get('accepts', []) or []
            if not isinstance(accepts, list):
                accepts = []
            aspect = Aspect(
                name=name,
                node_types=tuple(type_names),
                edge_relations=tuple(relation_names),
                relation_facts=relation_facts,
                meaning=entry.get('meaning', '') or '',
                dimension=entry.get('dimension', 'semantic') or 'semantic',
                locked=bool(entry.get('locked', False)),
                metadata=entry.get('metadata') or {},
                accepts=tuple(a for a in accepts
                              if a in ('node_types', 'edge_relations')),
                routable=bool(entry.get('routable', False)),
                prompt_visible=bool(entry.get('prompt_visible', False)),
            )
            self._aspects[name] = aspect
            for t in aspect.node_types:
                # First aspect to list a string wins for reverse lookup —
                # deterministic given JSON ordering.
                self._reverse_node.setdefault(t, name)
            for r in aspect.edge_relations:
                self._reverse_edge.setdefault(r, name)

        # Merged {verb: {field: value}} across every aspect — the validator
        # holds a multi-homed verb to ONE value per field, so first-seen is
        # the value. One dict lookup behind fact_of(); the decoder's
        # missing-field scan and every dimension consumer read this.
        self._relation_facts: dict = {}
        for aspect in self._aspects.values():
            for verb, facts in aspect.relation_facts.items():
                merged = self._relation_facts.setdefault(verb, {})
                for k, v in facts.items():
                    merged.setdefault(k, v)

        # Load-time derived policies — computed once per adopt, never per
        # read (the writer owns the cache, so these can never go stale).
        #
        # Two policies, one deliberate difference:
        # · structural_exclusions — the FULL noise set. For flat READS
        #   (connection lists on node pulls): noise carries no semantic
        #   claim, and per the standing decision that hides
        #   community_member too.
        # · traversal_exclusions — noise MINUS community_member. For graph
        #   DYNAMICS (traverse, spread activation):
        #   community edges carry activation and narrative context —
        #   conduction is not visibility, so the hide decision doesn't
        #   silence them here.
        noise = self._aspects.get('noise')
        self.structural_exclusions: frozenset = (
            frozenset(noise.edge_relations) if noise else frozenset())
        self.traversal_exclusions: frozenset = (
            self.structural_exclusions - {'community_member'})

        # Noise VETOES: a string in noise is machinery, whatever else claims
        # it. Membership is what the exclusion sets above already read, so
        # they veto by construction — but the reverse maps are first-claimant
        # (setdefault, above), which made the PRIMARY aspect of a dual-homed
        # string depend on JSON key order. Consumers that skip by primary
        # family rather than by membership — community typed adjacency is the
        # live one — would then include or skip a noise relation according to
        # where `noise` happens to sit in the file. Force noise to win here,
        # once, so every derived view agrees: primary_edge_map, by_edge_relation,
        # by_node_type, and the S2 adjacency skip that reads them.
        if noise:
            for t in noise.node_types:
                if t in self._reverse_node:
                    self._reverse_node[t] = 'noise'
            for r in noise.edge_relations:
                if r in self._reverse_edge:
                    self._reverse_edge[r] = 'noise'

    def add_members(self, classifications, source: str = '') -> int:
        """The single write door for classifier output.

        `classifications`: [{'category': 'node_types'|'edge_relations',
        'value': str, 'aspects': [names, primary first], <field>: value...}]
        — the shape AspectEncoder._validate_classifications emits, where
        <field> is any MEMBER_FIELDS enum field stored for the category
        (`orientation` for edge_relations). Each value is appended to each
        listed aspect's member list (idempotent on the member's STRING,
        whatever its shape — a bare string and an object for the same verb
        are one member). Fields are additive: a new member is written as an
        object when fields arrive with it; an existing member — in EVERY
        aspect that lists it, not only the listed ones, so the multi-homed
        agreement the validator enforces holds by construction — gains each
        field it lacks and is upgraded from string to object to carry it. A
        PRESENT differing value is kept and logged, never overwritten: the
        encoder only fills gaps, so a disagreement is a signal, not a write.

        Reads the file fresh at write time, writes atomically, then re-derives
        the in-memory maps from what was just written — the writer owns the
        cache, so no invalidation protocol exists or is needed.

        NOTE (accepted, narrow): a boot-time reconcile in a concurrent
        short-lived process can still interleave with this read-modify-write.
        The reconcile only writes when the working copy is behind the seed,
        and it re-heals idempotently on the next boot — self-correcting, not
        durable loss. Both writers now live in this file; if locking is ever
        needed, this is where it goes.

        Returns the number of changes written — (aspect, member) additions
        plus (member, field) fills.
        Raises on an unreadable working copy — a write must never proceed
        from (and then clobber the file with) an empty in-memory guess — and
        raises AspectContractError when the merged result fails structural
        validation (validate_taxonomy): the write is refused, the prior file
        stays intact, and the violations are logged. The encoder's
        per-classification filter prevents this in normal operation; the gate
        is the backstop for every other writer, present and future.
        """
        op = 'add_members(%s)' % (source or 'unattributed')
        path, data = self._read_for_write(op)
        # Closed list at the door: a name not already in the file is refused,
        # never setdefault-created. Nothing legitimately creates an aspect
        # through a member write — new aspects are a deliberate human edit to
        # the JSON — so an unknown name here is a typo or a confused writer,
        # and the silent alternative is a meaning-less husk aspect that every
        # consumer then trips over.
        targeted = {a for c in classifications for a in c['aspects']}
        unknown = sorted(a for a in targeted
                         if a.startswith('_')       # docs keys, never targets
                         or not isinstance(data.get(a), dict))
        if unknown:
            self._refuse(op, 'unknown or malformed aspect names: %s (new '
                             'aspects are a human edit to aspects_v1.json, '
                             'never a write-door side effect)' % unknown)
        added = 0
        filled = 0
        for c in classifications:
            category = c['category']  # 'node_types' or 'edge_relations'
            value = c['value']
            facts = {k: c[k] for k in stored_fields(category)
                     if c.get(k) is not None}
            for aspect_name in c['aspects']:
                members = data[aspect_name].setdefault(category, [])
                if any(member_name(m, category) == value for m in members):
                    continue
                here = {k: v for k, v in facts.items()
                        if k in stored_fields(category, aspect_name)}
                members.append({MEMBER_KEY[category]: value, **here}
                               if here else value)
                added += 1
            if not facts:
                continue
            # Fields ride to every home of the member, listed or not — each
            # to the homes its scope admits.
            for aspect_name, entry in data.items():
                if aspect_name.startswith('_') or not isinstance(entry, dict):
                    continue
                members = entry.get(category)
                if not isinstance(members, list):
                    continue
                for i, m in enumerate(members):
                    if member_name(m, category) != value:
                        continue
                    have = member_facts(m, category)
                    gap = {k: v for k, v in facts.items() if k not in have
                           and k in stored_fields(category, aspect_name)}
                    clash = {k: (have[k], facts[k]) for k in facts
                             if k in have and have[k] != facts[k]}
                    if clash:
                        try:
                            self._brain._log_warning(
                                'aspect_registry_write',
                                'field disagreement kept as stored, not overwritten',
                                '%s: %s.%s %r has %s' % (
                                    op, aspect_name, category, value, clash))
                        except Exception:
                            pass
                    if not gap:
                        continue
                    obj = m if isinstance(m, dict) else {MEMBER_KEY[category]: value}
                    obj.update(gap)
                    members[i] = obj
                    filled += len(gap)
        if not added and not filled:
            return 0
        self._commit(op, path, data)
        print('[aspects] %s: +%d memberships, +%d fields' % (
            op, added, filled), flush=True)
        return added + filled

    def clear_fact(self, verb: str, field: str, source: str) -> dict:
        """The curation door: CLEAR a MEMBER_FIELDS enum field from an edge
        verb in every home, so the fill lane proposes the verb again and the
        classifier derives the value anew. A reviewed wrong value is fixed by
        the unit that owns it, not hand-set: add_members only fills gaps (a
        present value is kept there by design), so clearing is what lets the
        refill reach it.

        A home left with no fields drops back to a bare string. Same gate as
        every write (validate_taxonomy), same fresh read, atomic write and
        reload. A verb no aspect lists is refused, as is a field that is not
        stored on edge relations. A verb the repo seed carries a value for is
        refilled FROM THE SEED at the next boot (reconcile job 5) — fix the
        seed first. A curated field (MEMBER_FIELDS `curated`) is never
        refilled by the classifier; the seed is its only source.

        Refused while an S2 cycle is running: add_members writes the file
        from its own fresh read, so a clear landing mid-cycle would be
        overwritten by the next batch (or overwrite it).

        Logged with `source` and the value each home held. Returns
        {aspect: value it held} for every home that held one; {} when none
        did (no write).
        """
        op = 'clear_fact(%s)' % source
        category = 'edge_relations'
        if getattr(self._brain, 's2_running', False):
            self._refuse(op, 'an S2 cycle is running — retry when it ends')
        if field not in stored_fields(category):
            self._refuse(op, '%r is not a field stored on %s (MEMBER_FIELDS '
                             'declares %s)' % (field, category,
                                               list(stored_fields(category))))
        path, data = self._read_for_write(op)
        homes = 0
        prior = {}                 # aspect → value the home held
        for aspect_name, entry in data.items():
            if aspect_name.startswith('_') or not isinstance(entry, dict):
                continue
            members = entry.get(category)
            if not isinstance(members, list):
                continue
            for i, m in enumerate(members):
                if member_name(m, category) != verb:
                    continue
                homes += 1
                have = member_facts(m, category).get(field)
                if have is None:
                    continue
                prior[aspect_name] = have
                rest = {k: v for k, v in m.items() if k != field}
                # Bare only when nothing but the string is left — any other
                # key stays for the validator to judge, never dropped here.
                members[i] = rest if set(rest) - {MEMBER_KEY[category]} else verb
        if not homes:
            self._refuse(op, 'no aspect lists %r — clear_fact curates a member, '
                             'it never creates one' % verb)
        if not prior:
            return {}
        self._commit(op, path, data)
        detail = '%s: %r.%s cleared (was %s)' % (
            op, verb, field,
            ', '.join('%s in %s' % (v, a) for a, v in prior.items()))
        try:
            self._brain._log_warning('aspect_registry_write',
                                     'fact cleared by curation', detail)
        except Exception:
            pass
        print('[aspects] %s' % detail, flush=True)
        return prior

    def _read_for_write(self, op: str):
        """(path, data) of the working copy, read fresh for a write. Raises on
        an unreadable file — a write must never proceed from (and then clobber
        the file with) an empty in-memory guess."""
        path = aspects_json_path()
        try:
            with open(path, 'r') as f:
                return path, json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            try:
                self._brain._log_error(
                    'aspect_registry_write', e,
                    '%s: cannot read %s — refusing to write' % (op, path))
            except Exception:
                pass
            raise

    def _commit(self, op: str, path: str, data: dict) -> None:
        """Validate, write atomically, re-derive the in-memory maps. A result
        that fails validate_taxonomy is refused and the file stays intact."""
        violations = validate_taxonomy(data)
        if violations:
            self._refuse(op, 'merged result fails structural validation: %s'
                         % '; '.join(violations))
        atomic_json_write(path, data)
        self._load()

    def _refuse(self, op: str, reason: str) -> NoReturn:
        """Log a refused write on both channels and raise AspectContractError."""
        detail = '%s REFUSED — %s' % (op, reason)
        try:
            self._brain._log_error(
                'aspect_registry_write', Exception('write refused'), detail)
        except Exception:
            pass
        print('[aspects] %s' % detail, flush=True)
        raise AspectContractError(detail)

    # ── Per-aspect access ──

    def __getattr__(self, name: str) -> Aspect:
        """brain.aspects.<aspect_name> → Aspect.

        Raises AspectContractError if the name isn't present. Required
        aspects always resolve (validated at boot). For emergent aspects
        whose existence isn't guaranteed, prefer by_name() which returns
        Optional[Aspect].

        Skip dunder-name lookups so Python internals (pickling, repr, etc.)
        don't trigger spurious AspectContractError raises.
        """
        if name.startswith('_'):
            raise AttributeError(name)
        # __dict__ access (not getattr) to avoid recursion when the registry
        # is partly constructed.
        aspects = self.__dict__.get('_aspects')
        if aspects is None:
            raise AttributeError(name)
        if name in aspects:
            return aspects[name]
        raise AspectContractError(
            "No aspect named '%s'. Required aspects: %s. "
            "Use registry.by_name() for emergent aspects." % (name, REQUIRED_ASPECTS)
        )

    def by_name(self, name: str) -> Optional[Aspect]:
        """Return aspect by name, or None if not present.

        Use this for emergent aspects whose existence isn't guaranteed by
        the contract. Required aspects can use the attribute form safely.
        """
        return self._aspects.get(name)

    # ── Reverse lookups ──

    def by_node_type(self, t: str) -> Optional[Aspect]:
        """Find the aspect whose node_types contains this type, or None."""
        name = self._reverse_node.get(t)
        return self._aspects.get(name) if name else None

    def by_edge_relation(self, r: str) -> Optional[Aspect]:
        """Find the aspect whose edge_relations contains this relation, or None."""
        name = self._reverse_edge.get(r)
        return self._aspects.get(name) if name else None

    def primary_edge_map(self) -> dict:
        """relation_string → PRIMARY aspect name, for every known relation.

        Primary = the FIRST aspect that claims the string in file order — the
        same contract by_edge_relation serves one string at a time. Callers
        that need the whole map (e.g. community typed adjacency) must use
        this instead of rebuilding it from all(): a hand-rolled comprehension
        lets the LAST claimant win, which silently flips a relation's family
        when a later aspect (settlement) multi-homes it.
        """
        return dict(self._reverse_edge)

    # ── Cross-aspect unions ──

    def types_in(self, names: Iterable[str]) -> tuple:
        """Union of node_types across the named aspects (insertion-ordered, deduped)."""
        seen = []
        for n in names:
            a = self._aspects.get(n)
            if not a:
                continue
            for t in a.node_types:
                if t not in seen:
                    seen.append(t)
        return tuple(seen)

    def homes_of(self, value: str, category: str) -> tuple:
        """Every aspect that lists `value` under `category`, in taxonomy
        (file) order — primary first, the same order the reverse maps use.
        () when nothing claims it. The decoder hands this to the encoder as a
        routed candidate's fixed `aspects`; readers that want only the
        primary use by_edge_relation / by_node_type."""
        attr = 'edge_relations' if category == 'edge_relations' else 'node_types'
        return tuple(n for n, a in self._aspects.items()
                     if value in getattr(a, attr))

    def fact_of(self, verb: str, field_name: str):
        """The stored value of a MEMBER_FIELDS enum field for an edge verb,
        or None when the verb is a bare string / unknown. One value per verb
        across its homes (validator-enforced), so the lookup needs no aspect."""
        return (self._relation_facts.get(verb) or {}).get(field_name)

    def actor_of(self, verb: str, source_id: str, target_id: str):
        """The node that does the work `verb` names in the claim
        source_id --verb--> target_id: the source for a source_actor verb,
        the target for a target_actor one, None for a symmetric verb (no
        actor) or one not yet oriented. The one place "who acts" is decided:
        a reader passes the claim's own ends (the DAL's per-claim direction)
        and never pairs an arrow with a verb itself."""
        orientation = self.fact_of(verb, 'orientation')
        if orientation == 'source_actor':
            return source_id
        if orientation == 'target_actor':
            return target_id
        return None

    def relation_facts(self) -> dict:
        """{verb: {field: value}} for every verb that carries a fact — a fresh
        copy. Verbs with no facts are absent; iterate relations_in(all()) for
        the full vocabulary and fact_of() for the gap."""
        return {v: dict(f) for v, f in self._relation_facts.items()}

    def relations_in(self, names: Iterable[str]) -> tuple:
        """Union of edge_relations across the named aspects (insertion-ordered, deduped)."""
        seen = []
        for n in names:
            a = self._aspects.get(n)
            if not a:
                continue
            for r in a.edge_relations:
                if r not in seen:
                    seen.append(r)
        return tuple(seen)

    # ── Discovery + enumeration ──

    def all(self) -> dict:
        """All aspects keyed by name. Returns a fresh dict — safe to iterate."""
        return dict(self._aspects)

    # ── Edge embedding text ──

    def compose_edge_text(self, relation: str, description: str) -> str:
        """Compose the canonical text embedded as an edge's semantic identity.

        Pattern: "[<relation>] <description>"

        INTRINSIC to the edge — does NOT include:
          - the partner node title: would couple the edge embedding to the
            partner (cascade-stale on node revise) and double-count its signal.
          - the relation's aspect-family `meaning`: that text is verbose
            classifier guidance authored for AspectIntegration; baked into
            every edge it dominated the (much shorter) description in the
            embedded string and blunted the per-edge disambiguation that
            embedding the description was meant to provide. The family meaning
            still lives in `aspects_v1.json` and feeds the classifier — it is
            simply not part of the edge's embedding geometry.

        Stable per `(relation, description)` pair, so a single embedding can be
        stored on `edge_relations.embedding` (schema v26+) and reused across
        partner revisions.

        Used by:
          - `GraphDAL.add_relation` (write path) — invalidate + async re-embed
          - `surface_contract` (read-path embed fallback when stored blob NULL)
          - `scripts/backfill_edge_embeddings.py` (one-shot migration)
        """
        rel = (relation or '').strip()
        desc = (description or '').strip()
        parts = []
        if rel:
            parts.append('[%s]' % rel)
        if desc:
            parts.append(desc)
        return ' '.join(parts)

    # ── Construction for tests / seeding ──

    @classmethod
    def from_dict(cls, brain, data: dict) -> 'AspectRegistry':
        """Construct directly from a dict — bypasses _load.

        For tests and seeding paths. The data shape mirrors aspects_v1.json:
            {name: {node_types: [...], edge_relations: [...],
                    meaning: '...', dimension: 'semantic',
                    locked: bool, metadata: {...},
                    accepts: ['node_types'|'edge_relations', ...],
                    routable: bool, prompt_visible: bool}}
        Omitting the fact fields gives the conservative defaults (see
        Aspect) — a stub without routable=True is invisible to the
        classifier, without prompt_visible=True invisible to vocabulary
        blocks.
        """
        instance = cls.__new__(cls)
        instance._brain = brain
        instance._adopt(data)
        return instance
