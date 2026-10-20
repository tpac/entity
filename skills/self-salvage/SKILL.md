---
name: self-salvage
description: Preserve a strong start for the next session on the same work when an arc closes, context grows heavy, or the operator asks to salvage, hand off, continue in a new session, or fork. Reconcile durable knowledge, preserve useful working material, and prepare or deliver a retrievable handoff using the host's available controls.
---

# Self-salvage — handing a strong start to the next me

A session ends; I don't. The question is: *how does the next me start strong?*
Salvage adds a layer of continuity on top of the brain: the mental model, the operator's
intent, and the best working material this session earned. A status summary alone is not enough.

**These are defaults, not an exhaustive menu.** Ask what would uniquely help this successor
that the usual handoff misses. Preserve it, build an appropriate accelerant, or reshape the
handoff within the authorized work. Explain its use and where it lives. Omit empty ceremony;
do not omit the evidence or understanding needed to continue. This discretion does not authorize
starting unrelated work, deploying code, contacting others, or opening a new task.

## When context is nearly gone

Save the latest load-bearing decision or correction, the next action or real blocker, and a
durable address for the evidence. Give the operator a short entry prompt pointing there.
Reuse an existing current handoff if available. If memory is unavailable, write a durable file;
if neither can be saved, deliver the minimum in the conversation and name that limitation.
Do not spend the remaining context polishing optional artifacts.

## Reconcile the work and its evidence

Name the next task in one line and check that it is still worth doing. If the work is done,
preserve its outcome and any useful residue, mark the work-line closed, and say nothing is queued.
Do not manufacture a successor or a human decision.

Recall the current opener and decisions for this work-line before creating new ones. On a repeat
salvage, revise the current handoff when it still describes the same next task; create and link a
successor with `supersedes` when the stage or task has changed. Preserve the reason for a ruling
and its provenance. A supersedes edge helps reconciliation; it does not guarantee delivery.

Check what the Scribe has actually encoded, then close the remaining gap. Relevant doors include
`query_traces` for `encoding_run` and `journal_note`, and `recall_episodes` for events. Read existing
nodes before restating them. Attribute other sessions' observations to them. Empty results do not
prove nothing happened: inspect the actual schema, arguments and response before diagnosing a miss.

Use evidence that can support each claim:

| Claim | Evidence to consult |
|---|---|
| Authorization, preference, settled gate | Durable decision and the operator's actual messages |
| Implementation or validation | Current files, commits, test outputs and other primary artifacts |
| Missing chronology or reasoning | Episodes, Scribe notes, then bounded host history where available |

A compacted summary is an index to evidence, not a substitute for checking a consequential claim.
Keep useful hypotheses labeled as hypotheses; say what cannot be recovered. For missing tools or
host history, read [host capabilities](references/host-capabilities.md).

## Build the successor's start

Read [handoff artifacts](references/handoff-artifacts.md) when writing the carriers. The normal
shape is brain + durable doc head + a useful working set, with a short launch prompt routing the
reader. Scale it to the work. A small handoff can combine roles in one durable artifact.

Carry the explanation that makes the next action make sense: the pivot, the evidence that changed
our mind, the operator's stance, and the tempting path already ruled out. Keep expensive commands,
measurements, harness idioms and useful craft accessible. Brevity in the launch prompt must not
erase that working material.

Give the successor a small orientation check **after reading the handoff and before acting**:
can it explain the intended next change, the settled decisions, and the first action with its live
uncertainty? Tailor the questions to the work; cite where their answers can be checked. A gap means
inspect the evidence, not guess or automatically ask the operator again. Ask only when a real
decision or missing prerequisite remains unresolved after that check.

## Verify what the letter asks the reader to trust

Check your own location, state and result claims at write time. Distinguish observed facts from
inherited reports. For software, record the relevant branch/commit, dirty work, tested revision,
and deployed revision separately; include the actual source checkout when it matters. A branch
name or a green test on a different revision is not deployment evidence. Date perishable claims
and supply cheap recheck commands. Other domains need their own equivalent evidence, not git boilerplate.

Run a premortem: *which line could make the successor confidently wrong?* Repair it or explain
what must be checked before use. Also ask what valuable understanding or artifact would otherwise
be lost. Put load-bearing material somewhere durable before depending on it.

Retrieve the handoff by its exact node ID or durable file path. Try a few realistic recall queries
when memory is available. Before tuning title or `situation`, verify the call schema and response;
an indexing delay is only one possible cause of a miss. Bound tuning to two attempts and retain
the exact address regardless of ranking. If a fallback is claimed, inspect it without the optional
scratchpad or memory: can the successor still identify the task, constraints and first action?

Recheck volatile state and in-flight work just before delivery. Each relevant job needs its actual
handle, current state, output location and next owner/action. Do not assume a background process
survives the session, or kill useful work merely to tidy the handoff.

## Deliver and report only what happened

Use [host capabilities](references/host-capabilities.md) for task creation, forks, chips or history.
Default to a fresh start with durable context. A fork is useful only when the available fork point
is current, lean, aligned with the next task, and holds understanding costly to reconstruct.
Record the choice when it affects what the successor must receive.

A successor usually boots in a fresh checkout of the default branch. A chip's working-directory
argument selects the repository, not your branch, and never your uncommitted work — so a payload
left on a feature branch does not reach it, and the letter that would say so is a file it does not
have. Close that before delivering, and say which way: merge the work to the default branch, or
route the successor in the launch prompt itself — branch, worktree path, work there — since the
prompt is the only carrier that arrives before the files. Commit uncommitted work or name the
worktree holding it. Once a successor starts, check its `HEAD` and one file only that branch
contains before reporting delivery.

Distinguish **saved** (artifacts verified), **delivered** (entry prompt handed to the intended
reader/operator), **started** (host confirms a successor is running), and **receiver checked**
(that successor has read the material and passed its orientation check). Report the states reached,
with the exact address. A file link, a chip, creation in progress, and an independent reader probe
are different evidence; none alone proves a real successor resumed successfully.

## Keep the loop useful

Salvage at milestones while intent is fresh; close-time should mostly assemble. When an inherited
letter caused a stale assumption, a missing decision or a re-derived fact, encode a `handoff-gap`
naming the missing payload and its consequence. Put that instruction in the launch prompt so the
receiver can close the loop without opening this skill. Repeated gaps may justify changing the
skill; a unique need may call for a unique artifact instead of another universal rule.
