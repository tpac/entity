---
name: shape-review
description: >-
  Mechanism-level review of a diff or a plan: is the mechanism chosen the simplest one that
  satisfies the requirement, and does it fix the class of problem or one instance of it? Licensed
  to propose a different behavior-equivalent implementation. Runs after a fix is written (or on
  the stated plan before code exists) and before defect review. Triggers: "is this the right
  shape", "is there a simpler, less spread shape", "should this be a delete instead of a check",
  "did we fix one agent or all of them", "is this drift", "we deleted X — did the mechanism go or
  just the symbol", "is this the last instance of the class or are there more". Cheap by design
  (two brain recalls) so it can run on every fix. Not for line polish, bug hunting, or restructuring a subsystem — those
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
only for hits that could change the verdict. Do not re-read a function you already have. Measured
2026-10-05 on Opus 5.5 subagents, 15 runs, KEEP and non-KEEP alike: 105–157k tokens per run, of
which ~75k is the agent's fixed overhead before any skill work; the skill itself costs 40–85k.
Inline in a live session that overhead is already paid, so a run costs the skill's share only.
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

**Deletions are diffs too.** A removal gets the same six probes: did the mechanism go or only its
symbol, and do siblings of the deleted thing still stand (probe 6 on what remains).

## Phase 0 — Requirement, unit, siblings (one pass)

1. **State the requirement first**, in one line, as you infer it from the diff, its tests, and the
   commit message. Say it before anything else so the operator can correct it. "Simplest" has no
   meaning without it. If the shape came from an explicit operator ruling, say that too — a
   RESHAPE then knows it is challenging a decision, which is allowed, but must be said.
2. **Name the changed unit(s) and their role** — the function, handler, encoder, hook, prompt
   section, DAL method, fixture. Then state the role as the question the unit answers (*"keep a
   log sink from eating disk"*, not `_check_rate_limit`) and its effect: the table, file, env var
   or state it writes, deletes or reads. "Unit" is anything with a role that other things also play.
3. **Recall #1** — how does this codebase already handle this *class* of problem? Phrase it as
   what you'd remember, not a keyword: *"how the brain invalidates derived vectors when their
   source changes"*, not `delete_for_node`. `limit` 5. This one query feeds probes 1, 4 and 5 —
   the house pattern, the layering doctrine and the special-case history tend to surface together.
4. **Recall #2** — *"what else in this codebase answers <role question>; the governing layer or
   prior consolidation for this role"*, `limit` 5. Siblings are defined by role, and the brain
   holds role-named classes whose members share no name in code. Every unit the recall names is a
   sibling candidate to verify by grep in the next step.
5. **Grep for siblings by effect, then by name** — first the effect from step 2 (the table or
   file written, the state deleted, the env var read), then the names the recalls surfaced, then
   the unit's own naming family, dispatch table or base class. Scope each grep to its home; do
   not sweep the repo. Drop hits whose role differs and say so. Count. If the hit list does not
   fit one screen, that is itself the probe-6 signal — record the count and move on. This number
   feeds probe 6.

Every recalled node is a **prediction to verify** against the current code, never a verdict.

## Phase 1 — Six probes

Run all six, every time, inline. Each ends in one line: a **strength** on the scale below, the
answer, and the **Payoff** when strength is 2 or more (what the alternative buys: lines, hot-path
cost, races, coherence, one owner, places that must remember it N → 1). A probe at 0 gets the
number alone. These lines are working notes; probes at 2 or 3 carry into the verdict, as the
content of its RESHAPE or CLASS-FIX block.

Strength scale, used for every probe and for the verdict's confidence:
- **0** — nothing there; the question was asked and the code answers it cleanly.
- **1** — a faint alternative or a possible sibling; not clearly better, or not verified.
- **2** — a clear alternative that wins on at least one payoff axis, or a sibling verified in code.
- **3** — the alternative dominates, or the class is confirmed; keeping the shape is itself a defect.
Strength is evidence, not conviction: 3 means read and verified, 1 means suspected from a recall
or a name. The reader turns the numbers into a picture; the verdict does not round them.

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
   Recall #2 (Phase 0) already said whether a layer exists and why; verify it against the code
   here. *Example: a sixth copy of a marker literal in a DAL query was refused
   because the contract already owned `is_machine_turn`; two per-door count clamps became one
   parameterized helper.* If the diff is the first instance and no siblings exist, say so — a
   single instance is not drift.

## Phase 2 — Verdict

Short. Not a findings list. The reader decides in one screen.

```
Requirement: <one line, as inferred>          [from operator ruling <id> — yes/no]
Unit(s): <what changed>   Role: <the question the unit answers>
Siblings: <N> (<where>; enumerated by: name | role | effect-verified)
Verdict: KEEP | RESHAPE | CLASS-FIX   Confidence: <0–3> — <what would move it>

Probes: 1:<0–3> · 2:<0–3> · 3:<0–3> · 4:<0–3> · 5:<0–3> · 6:<0–3>

RESHAPE — <alternative in 3–5 lines>
  Payoff: <lines, hot-path cost, races closed, coherence, single owner>
  Gives up: <trade-off, honestly>
  Moots: <which likely /code-review findings vanish with the old shape>

CLASS-FIX — siblings sharing the defect: <list, marked>
  Governing layer: <exists at … | should exist, home: …>

Coverage: <what was read/grepped/recalled; the grep patterns behind any "no caller" claim or sibling count; what was not>
```

Print only the block that applies. A **KEEP** prints the header, the probe row, Confidence and the
Coverage line, and stops — no block. A probe at 1 under a KEEP is not hidden: it stays in the row,
so the reader sees where the shape is merely adequate. KEEP is a normal and frequent result. Manufacturing an alternative to
look useful is a defect of this skill, not a finding.

**Write-back is part of a non-KEEP verdict, not an expansion.** Before handing back a RESHAPE or
CLASS-FIX, `remember` it with a `connect_to` edge: a RESHAPE `supersedes` the original design node
if one exists; a CLASS-FIX names the governing layer and the case that motivated it. If recall #1
or #2 already surfaced a node holding this verdict, `revise` or `connect` that node instead of
writing a second. Write back the verdict only: a recalled node's claims about main, deploys or
merges are not checked against a branch or an older checkout, so do not "correct" them from
here. This is what makes probe 6 sharper on the next run — without it every run rediscovers the
layers from scratch.

## Expansions — only after a non-KEEP verdict

Earned, not default. Name the one you use.

- **Settled-constraint check.** `get_nodes` on the hits the verdict rests on, following
  `supersedes` / `corrects` one hop. An alternative that undoes a locked decision or a recorded
  correction is a false positive — drop it, or surface the tension instead of proposing.
- **Requirement check.** `recall_episodes` on the requirement — when the diff's shape came from an
  explicit ruling the verdict challenges, and on any RESHAPE whose Gives-up names a property the
  operator may already have accepted (a known caveat, a declined guard, a tolerated race). Two
  recalls at limit 5 can miss a ruling, and a RESHAPE that undoes one is a false positive. Quote
  the ruling when found and lower Confidence to the number it leaves.

## Guardrails

- **Recommend, never apply.** Hand the verdict back. The operator says yes; the session builds;
  then `/code-review`.
- **Verify what you recall.** A recalled house pattern may have moved, been superseded, or never
  have been adopted. Grep it before you cite it.
- **A "no caller" claim is a grep claim.** Grep the method name, the owning type (callers hold the
  instance in a local variable the method-name grep never sees) and attribute access on it, and
  put the patterns in Coverage. *A full-text search mode was called callerless after a method-name
  grep; the verbatim-phrase tool called it through a local.*
- **Proportionate.** Inline by default. Spawn a subagent (if your host has them) only when probe 6
  finds a wide class — more siblings than you can read in one sitting — and say that you did.
- **Coverage is loud.** A shallow read that returns KEEP is a false all-clear. If you did not read
  the enclosing functions or could not resolve the siblings, the Coverage line says so.
- **Host-neutral.** Tool names here are the brain's own (`recall`, `get_nodes`, `remember`,
  `recall_episodes`); use whatever prefix your host gives them. `/simplify`,
  `/code-review` and `/architecture-review` are Claude Code names used for orientation, not
  commands to run; the Order line above gives the Codex equivalents. The architecture skill is
  `architecture-review` in the same catalog where installed, otherwise read it by path.
