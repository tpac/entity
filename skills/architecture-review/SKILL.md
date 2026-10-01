---
name: architecture-review
description: >-
  Review the architecture of a subsystem or cross-file concern in a codebase.
  Use for questions about ownership, placement, coupling, duplicated responsibilities,
  consolidation, or a subsystem refactor plan. Trace callers and contracts, verify prior
  brain decisions, and recommend keeping the structure, a focused change, or a
  dependency-ordered plan. For the mechanism chosen by one diff or plan use shape-review;
  for line cleanup or correctness bugs use the host's ordinary review workflow.
---

# Architecture Review

Determine whether a subsystem's structure serves its requirement, and what change, if any,
would improve it enough to justify migration. Review the call graph and ownership boundaries,
not just the files named in the request. Recommend; do not implement the recommendations.

## Input and host adaptation

Accept a concern, desired outcome, and optional file leads. For example:

```text
architecture-review: can cache invalidation have one owner? |
src/cache.py src/pipeline.py
```

Use the skill name shown by the host's selector. Plugin installations may namespace it:
with the Entity Codex naming pattern, expect `entity:architecture-review` (a skill mention
is `$entity:architecture-review`); a standalone Codex copy uses `$architecture-review`.
Claude Code uses its registered slash-command name. These are invocation examples,
not commands to execute through a shell.

Brain tools below use bare names; resolve the host's actual MCP prefixes and argument
schemas. Search/read with the available tools (`rg` or equivalent). When delegating,
use host subagents, not new user-visible tasks. Claude's Agent/Explore names are not
portable requirements; Codex exposes collaboration tools in hosts that support them.

If leads are absent, find the likely owner. State the inferred outcome and boundary
before expanding; ask only when ambiguity would materially change the review.
An operator asking to "unify X" supplies a goal to examine, not proof that unification
is the right answer. Honor explicit constraints and explain any tension with the goal.

## Phase 0 — Requirement and bounded discovery

State the requirement, the observed structural cost, and the initial boundary in a few
lines. If the cost is only suspected, label it as a hypothesis. Existing behavior is
evidence, not automatically the intended contract. Distinguish required behavior,
suspected accidental differences, and differences whose purpose is unresolved.
Architectural discovery may reveal a needed behavior change even when the operator
did not request one initially. Explain the insight and consequences for discussion;
do not treat a proposed new contract as an agreed requirement.

### Code

Read the owning implementation, contracts, relevant callers, and tests. Search for
siblings with the same responsibility. Follow unresolved dispatch or dependencies only
when they could change the recommendation. File size and similar syntax alone do not
establish a structural defect; trace which invariant and lifecycle each unit owns.

Expand beyond the leads when needed, but stop when the material ownership questions
are answered. Record unresolved callers and coverage limits instead of silently turning
a focused review into a repository audit. Do not re-read code already in context.

### History

Start with up to two focused `recall` queries, `limit: 5`: prior decisions about this
concern, and the existing owner or consolidation history. Reuse relevant history already
in context. Open at most two load-bearing hits initially; follow corrections or use
`recall_episodes` only to resolve a specific uncertainty that could change the verdict.
Name the uncertainty before expanding the budget.

`recall` can return bodies and edges, not just titles. If the host supports result
projection, retain the response and expose title/ID summaries first, then selected bodies.
Otherwise count the full response as context cost; ignoring a body does not save tokens.
Share selected evidence with reviewers rather than having each repeat discovery/recall.

Separate:

- **Current operator constraints:** requirements and explicit rulings applicable here.
- **Historical decisions:** their rationale, scope, and evidence; verify current relevance.
- **Completed or in-flight work:** verify adoption in code and identify ownership conflicts.

A zero-caller method may be a planned migration destination. Conversely, an old "keep
this for migration" decision may already be resolved. A recalled node is a prediction
to verify, not an automatic veto or permission. Surface a conflict with an applicable
locked decision or correction; do not silently discard it or recommend reversing it as
settled. A newer explicit operator ruling takes precedence.

If the brain is unavailable, report it and continue the code review. Mark recommendations
that depend on unverified intent as provisional; unavailable history is not empty history.
Do not retry indefinitely or claim a complete review while material evidence is missing.

End discovery with coverage: owners/callers/tests read, history checked, unresolved
boundaries, and whether those gaps could change the decision.

## Phase 1 — Five angles, proportionate execution

Always cover the five angles below. Delegate when an independent answer to an unresolved
question could materially change the decision; name that question. A large straightforward
boundary may need no extra reviewer, while a small disputed invariant may benefit from one.
Give reviewers the shared discovery evidence, constraints, permitted scope, and a bounded output.
Do not tell an independent reviewer the desired verdict.

Respect the host's available concurrent slots, including the parent. Batch or reuse
workers when necessary, giving each angle an explicit brief. If agents are unavailable,
cover all angles inline and disclose any material loss of independent scrutiny. Report
missing coverage or unresolved disagreement; do not call reused workers independent reviewers.

1. **Placement:** Who owns the invariant? Does logic live with the audience, lifecycle,
   and responsibility it serves? Name the existing owner before proposing a new module.
2. **Unification across callers:** Do repeated sequences enforce the same invariant and
   change for the same reason? If so, identify the shared owner. Determine whether differing
   policies or lifecycles are required or accidental before preserving or consolidating them.
   Two components interpreting the same node differently may expose a missing common contract.
   Compare concrete inputs/outputs, intended meaning, and caller dependencies; propose the
   correct shared interpretation if supported, including any behavior that must change.
3. **Cohesion:** Which responsibilities change independently? A split needs a meaningful
   seam and a concrete benefit; moving lines into more files is not itself a benefit.
4. **Coupling:** Do callers use the owner's contract, or bypass it and duplicate its
   knowledge? Verify the repo's current layering rules and contract definitions rather
   than imposing a generic architecture. Name the interface that should carry the work.
5. **Altitude:** Is the proposal repairing an instance or the governing mechanism?
   Count the places that must remember an invariant. A shared helper still drifts if
   every new caller must remember to invoke it; look for an existing common boundary.
   Generalization must earn its complexity and preserve legitimate consumer differences.

For each angle, return either no supported change or a candidate with: code evidence,
mechanism, concrete cost, proposed owner/target, trade-off, and evidence gaps. Keep findings
bounded; no quota. These may be working notes: share evidence once and summarize angles
with no supported change together in the final response. Wait for every assigned review
before synthesizing, or disclose which results are missing and what conclusions remain provisional.

## Phase 2 — Reverse pass and synthesis

First reverse-check the overall decision, including KEEP: what minimum structure and
semantics does the outcome require, and what evidence supports retaining or changing
the current design? Which unexamined path or behavioral difference could disprove the
decision? Absence of a proposed change is not evidence that the requirement is met.

Then deduplicate candidates by underlying mechanism. For each surviving recommendation,
work backward from the intended outcome, marking any proposed change to that outcome:

- Can the current structure already meet it? What observed cost justifies changing it?
- Would deleting an obsolete path or extending an existing owner be sufficient?
- What invariant makes the proposed consolidation valid? What distinct policy would be
  lost, and who gains responsibility or coupling?
- Does apparent structural duplication reveal conflicting interpretations of the same
  concept? State the proposed semantics, affected callers, compatibility consequences,
  and any decision still needed. Neither preserve drift nor normalize differences without evidence.
- Does the proposal reduce places that must remember the rule, or merely move repetition?
- What does it give up: migration risk, runtime cost, flexibility, or operational complexity?
- What evidence would refute it? Verify the structural benefit separately from behavior:
  show the intended ownership/routing property, preserve required behavior, and demonstrate
  any deliberate behavior correction against the proposed contract. Existing tests may
  encode accidental behavior; explain that conflict rather than silently treating either
  the test or the proposal as authoritative.

Reject unsupported changes and changes that do not earn their migration cost. Surface
constraint conflicts and unresolved alternatives instead of laundering them into a plan.
Do not treat agent agreement as evidence beyond the code and history they cite.

## Phase 3 — Decision and appropriately sized output

Lead with the requirement, reviewed boundary, coverage limits, and one outcome:

- **KEEP:** no supported structural change within the reviewed boundary. Explain why;
  include material unknowns. Do not generate implementation steps.
- **FOCUSED CHANGE:** a local ownership/interface correction is enough. Give the problem,
  target, evidence, trade-off, and verification without inflating it into a campaign.
- **PLAN:** multiple dependent changes are warranted. Give a scope statement and dependency
  summary, then self-contained steps using the template below.

When evidence could change the outcome, mark it provisional and name the missing check.
Do not describe a provisional KEEP as a clean bill of health.

For PLAN, use the repo's plan convention, normally `docs/<TOPIC>-ARCH-PLAN.md`, when file
output is authorized. In read-only mode return the plan in the response. A review authorizes
recommendations, not source edits, commits, deployment, or spawning implementation tasks.

```markdown
## Step N — <imperative title>

**Problem and evidence.** Structural cost, owning invariant, and verified code references.
**Target state.** Owner/interface and intended semantics; required behavior preserved, proposed corrections, and decisions still needed.
**Trade-off.** Why this earns migration cost; alternatives rejected and what it gives up.
**Files and call sites.** Exact scope and any unresolved dynamic callers.
**Verification.** Evidence of structural benefit and checks of intended behavior, including deliberate changes; name existing tests accurately and identify missing checks.
**Blast radius.** Consumers, migration/recovery concerns, and likely extent of the change.
**Depends on.** Earlier steps, or none. Identify changes that must land together.
**Constraints.** Applicable operator rulings/history, verification status, and open tensions.
```

Make each step understandable without this conversation. Separate sessions are an option,
not a requirement: preserve continuity when it helps, and keep atomic changes together.
List checks actually performed separately from verification proposed for implementation.
