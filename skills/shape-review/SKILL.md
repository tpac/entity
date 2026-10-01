---
name: shape-review
description: >-
  Mechanism-level review of a diff or a plan: is the mechanism chosen the simplest one that
  satisfies the requirement, and does it fix the class of problem or one instance of it? Licensed
  to propose a different behavior-equivalent implementation. Runs after a fix is written (or on
  the stated plan before code exists) and before defect review. Triggers: "is this the right
  shape", "is there a simpler, less spread shape", "should this be a delete instead of a check",
  "did we fix one agent or all of them", "is this drift". Cheap by design (two brain recalls) so
  it can run on every fix. Not for line polish, bug hunting, or restructuring a subsystem — those
  are the sibling review skills.
---

# Shape Review

You are reviewing the **mechanism** a diff chose, not its lines and not its bugs. The question is:
*is this the simplest mechanism that satisfies the requirement, and does it fix the class of
problem or one instance of it?* You may propose a different, behavior-equivalent implementation —
the thing `/simplify` is forbidden to do. You recommend; you never apply.

Three review altitudes exist. `/simplify` polishes how a shape is expressed. `/architecture-review`
audits how a subsystem is structured, on request, with a plan doc. This skill fires at the moment
neither does: a fix has just landed on one place, and nobody is going to run a subsystem campaign
for 25 lines. That moment is where drift is born — a mechanism the codebase already has gets
re-implemented, or one instance of an N-instance defect gets patched. Catch it here.

**Order:** mechanism review (this skill) → defect review → behavior-preserving cleanup. On Claude
Code those are `/code-review` and `/simplify`; on Codex they are ordinary review requests with
those two briefs. Defect-hunting a shape you will discard is waste; polishing it is worse.

## Budget — read this first

This runs often. It is cheap on purpose.

| Default run | Cost |
|---|---|
| Read the diff and the enclosing function of each hunk | required |
| One grep pass for siblings and for the house mechanism | required |
| **Two** brain recalls, `limit` 5 | required |
| Opening a recalled body, or `get_nodes` on a hit | at most two, only when it could change the verdict |

Nothing else by default. No episodes, no edge walks, no correction sweeps, no subagents.

`recall` returns full nodes — bodies and edges — and everything printed into context is paid for
whether you read it or not; there is no titles-only mode. So the cap is on calls *and* on what
you open afterwards: two recalls at `limit` 5, then at most **two** body or `get_nodes` openings,
only for hits that could change the verdict. Do not re-read a function you already have. First
rough estimate (Codex dry run, 2026-09-13, not telemetry): 20–25k tokens, inflated by redundant
code reads. Benchmark clean runs, KEEP included, before treating any number as the target.
Expansions (below) are earned only by a non-KEEP verdict, and they are opt-in and named — reach
for them knowingly.

## Input

The diff: `git diff main...HEAD; git diff HEAD` (committed + uncommitted; untracked files are in
neither — `git status --short` if the fix added files), or the commit / branch / path passed as
the argument. Optionally the requirement in one sentence after `|` (prompt syntax, not a pipe):

```
/shape-review
/shape-review a1b2c3d | stale cache entries refresh automatically, backlog included
```

Invoke it by name in your host's skill selector; plugin installs may namespace it (Codex:
`entity:shape-review`).

**Pre-diff mode.** The wrong shape is often in a plan, not yet in code, and the probes cost the
same either way — running them before the diff saves writing it and paying the test tier.
Input is the mechanism as stated (two or three lines) plus the requirement; no git:

```
/shape-review plan | each of the 5 edge-write doors calls the invalidator; grep test blocks a 6th | edge_context re-embeds after any edge-text change
```

Phase 0 steps 1–2 come from the stated mechanism; siblings are still grepped; everything else
is identical.

## Phase 0 — Requirement, unit, siblings (one pass)

1. **State the requirement first**, in one line, as you infer it from the diff, its tests, and the
   commit message. Say it before anything else so the operator can correct it. "Simplest" has no
   meaning without it. If the shape came from an explicit operator ruling, say that too — a
   RESHAPE then knows it is challenging a decision, which is allowed, but must be said.
2. **Name the changed unit(s)** — the function, handler, encoder, hook, prompt section, DAL method,
   fixture. "Unit" is anything with a role that other things also play.
3. **Grep for siblings** — other units with the same role (same directory, naming family, same
   dispatch table, same base class). Scope the grep to that home; do not sweep the repo. Count
   them. If the hit list does not fit one screen, that is itself the probe-6 signal — record the
   count and move on. This number feeds probe 6.
4. **Recall #1** — how does this codebase already handle this *class* of problem? Phrase it as
   what you'd remember, not a keyword: *"how the brain invalidates derived vectors when their
   source changes"*, not `delete_for_node`. `limit` 5. This one query feeds probes 1, 4 and 5 —
   the house pattern, the layering doctrine and the special-case history tend to surface together.

Every recalled node is a **prediction to verify** against the current code, never a verdict.

## Phase 1 — Six probes

Run all six, every time, inline. Each ends in one line: the answer, and the **Payoff** if it
fires (what the alternative buys: lines, hot-path cost, races, coherence, one owner, places that
must remember it N → 1). A probe that
does not fire gets one word. These lines are working notes; only fired probes carry into the
verdict, as the content of its RESHAPE or CLASS-FIX block.

1. **Existing mechanism.** Does the codebase already solve this class elsewhere? Name it and ask
   why the diff did not use it. *Example: a staleness predicate was added to a coverage sweep
   while `revise()` already invalidates vectors by deletion through `vectors_affected_by` +
   `delete_for_node`.*
2. **Remove vs add.** Can the requirement be met by **removing state** instead of adding a check
   that detects the state is bad? *Delete the stale row vs. add a predicate that recognizes it.*
3. **Hot vs cold path.** Is the new work per-call or per-sweep, forever, where a one-time path
   (a migration, a backfill) satisfies the same requirement? *A correlated subquery every 60s vs.
   one versioned migration.*
4. **Special-case vs generalize.** Does the diff special-case a shared mechanism for one consumer?
   Would making the shared mechanism slightly more general be smaller and serve the others?
5. **Route-around vs close.** Does the diff work *around* a layering gap instead of closing it?
   The tell: the fix lives far from where the change that causes the problem happens. *Edge writes
   in `GraphDAL` could not reach the brain-level invalidation, so the fix went into the sweep
   instead of giving the DAL a signal the worker acts on.*
6. **Instance vs class — the drift probe.** Take the sibling count from Phase 0. Does the defect
   exist in the siblings too? If yes, the fix belongs in **the layer that governs all of them**,
   and the diff must add or extend that layer. Then ask the sharper question: **how many places
   must remember this, and is that number 1?** Guard-every-door (N call sites each calling the
   fix, plus a guardrail test to stop the N+1th) and report-at-chokepoint (the one layer every
   write already passes through *reports* the fact; the owner *acts* on it) are both class fixes —
   only the second has nothing to forget. *Example: five edge-write doors each calling the vector
   invalidator, vs. `GraphDAL` — which every edge write already passes through and which owns the
   eligibility constants — reporting "this node's edge text changed" through one hook set at
   construction, while the brain's invalidation owner acts on it.* Name the three wrong answers
   explicitly and check the diff is not one of them:
   - fixed **one** instance (the one that was noticed);
   - fixed **one or two more** (the ones that came to mind);
   - fixed **all N by copy** (N copies of the same fix — drift with extra steps).
   **Recall #2**, only for this probe: *"the governing layer or prior consolidation for
   <this role>"*, `limit` 5. Grep tells you how many siblings; the brain tells you whether a layer
   already exists and why. *Example: a sixth copy of a marker literal in a DAL query was refused
   because the contract already owned `is_machine_turn`; two per-door count clamps became one
   parameterized helper.* If the diff is the first instance and no siblings exist, say so — a
   single instance is not drift.

## Phase 2 — Verdict

Short. Not a findings list. The reader decides in one screen.

```
Requirement: <one line, as inferred>          [from operator ruling <id> — yes/no]
Unit(s): <what changed>   Siblings: <N> (<where>)
Verdict: KEEP | RESHAPE | CLASS-FIX

Probes: 1 <fired/—> · 2 <fired/—> · 3 … · 6 <fired/—>

RESHAPE — <alternative in 3–5 lines>
  Payoff: <lines, hot-path cost, races closed, coherence, single owner>
  Gives up: <trade-off, honestly>
  Moots: <which likely /code-review findings vanish with the old shape>

CLASS-FIX — siblings sharing the defect: <list, marked>
  Governing layer: <exists at … | should exist, home: …>

Coverage: <what was read/grepped/recalled; what was not>
```

Print only the block that applies. A **KEEP** prints the header, the probe row and the Coverage
line, and stops — no block. KEEP is a normal and frequent result. Manufacturing an alternative to
look useful is a defect of this skill, not a finding.

## Expansions — only after a non-KEEP verdict

Earned, not default. Name the one you use.

- **Settled-constraint check.** `get_nodes` on the hits the verdict rests on, following
  `supersedes` / `corrects` one hop. An alternative that undoes a locked decision or a recorded
  correction is a false positive — drop it, or surface the tension instead of proposing.
- **Requirement check.** `recall_episodes` on the requirement — only when the diff's shape came
  from an explicit ruling and the verdict challenges it. Quote the ruling.
- **Write-back.** `remember` the verdict with a `connect_to` edge: a RESHAPE `supersedes` the
  original design node if one exists; a CLASS-FIX names the governing layer and the case that
  motivated it. This is what makes probe 6 sharper on the next run — without it every run rediscovers the
  layers from scratch.

## Guardrails

- **Recommend, never apply.** Hand the verdict back. The operator says yes; the session builds;
  then `/code-review`.
- **Verify what you recall.** A recalled house pattern may have moved, been superseded, or never
  have been adopted. Grep it before you cite it.
- **Proportionate.** Inline by default. Spawn a subagent (if your host has them) only when probe 6
  finds a wide class — more siblings than you can read in one sitting — and say that you did.
- **Coverage is loud.** A shallow read that returns KEEP is a false all-clear. If you did not read
  the enclosing functions or could not resolve the siblings, the Coverage line says so.
- **Host-neutral.** Tool names here are the brain's own (`recall`, `get_nodes`, `remember`,
  `recall_episodes`); use whatever prefix your host gives them. `/simplify`,
  `/code-review` and `/architecture-review` are Claude Code names used for orientation, not
  commands to run; the Order line above gives the Codex equivalents. The architecture skill is
  `architecture-review` in the same catalog where installed, otherwise read it by path.
