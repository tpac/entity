# Handoff artifacts

Read when writing a salvage. These roles keep the reader's route short without throwing away
useful material. Adapt the layout and combine roles for small tasks; add a different artifact when
the successor needs something these shapes do not provide. There is no required artifact count.

| Carrier | Owns | Reader's use |
|---|---|---|
| Handoff node | Resumable state, reasoning, decisions, evidence and next action | Understand and continue the work |
| Durable doc head or handoff file | Current entry point plus the essential fallback | Continue if memory or scratch is unavailable |
| Working set | Exact commands, measurements, craft and other accelerants | Recover the best of this session's practical leverage |
| Launch prompt | Task and route to those artifacts | Know what to open first |

## References and survival

Type ambiguous references: `id:`, `trace:`, `session:`, `git:`. Copy real IDs; use the current tool
schemas rather than treating examples here as API signatures. Deferred work should point to its
durable decision or handoff, not only the session where it happened.

For code, the state address is the repository, branch and commit. A worktree path stops being a
convenience once the successor boots elsewhere: a fresh session is cut from the default branch, so
merge the work there or carry branch and worktree path in the launch prompt (see the delivery step).
Identify dirty changes and how they survive. Verify the destination checkout before any mutation;
do not tell a successor to merge into an arbitrary checkout. Preserve the tested and deployed
revision separately when they differ. For non-code work, name the actual durable artifact/version.

Decisions, irreplaceable evidence and runnable artifacts the next session depends on belong in
durable storage. A scratchpad may be the only copy of an optional derived table or convenience
command, but its loss must cost speed, not the ability to reason or proceed. Say where its inputs
or reconstruction method live. An untested skeleton must say **UNTESTED**; if executing it is part
of the plan, preserve it durably. A file is not durable merely because it has been written under
`/tmp`; check the actual retention or repository state.

## Handoff node

Use `type=handoff` and a stable work-line title such as `[thread:<slug>]`. Reconcile the prior
opener as described in the skill. One independently resumable work-line per node; use a small
index only if several need a common entry point. Aim for a letter the successor can read quickly
(usually within 800 words), linking fuller evidence instead of compressing away its meaning.

Open with the real blocker and owner, or the first concrete action when unblocked. Then supply:

- The mental model and why this next step follows; load-bearing decisions and their evidence.
- Current state, what is settled, what remains open, and the relevant paths already closed.
- The tailored orientation check with evidence pointers; answer after reading, before acting.
- Perishable facts checked as of a stated time, their verification method, and what to recheck.
- Any actual human gate, including where to check for a later ruling before asking again.
- Relevant adjacent ownership or in-flight work and its continuation instructions.
- A pointer if a working set exists, labeled with its actual durability; for scratch, use
  `working set (ephemeral, may be gone): <path>`. Explain how to recover missing essentials.

Write `situation` in the language the successor will use to seek this work. Record assumptions
about the reader only when they affect the handoff. Omit empty sections; do not invent blockers,
measurements, owners or a fork discussion to fill a template.

## Durable doc head / fallback

Update the dated entry point at the top of the existing design or project document. If there is
none, choose an appropriate durable handoff file. The node alone can suffice when that is the
only carrier available, but explicitly report that memory loss has no independent fallback.

The head should locate the current handoff and the primary evidence. Include enough to identify
the task, key constraints and first action without memory or scratch access. Keep settled and open
items distinct. Link detailed evidence; use numbers only when they carry the finding. It is a
deliberate small overlap with the node for failure recovery, not a second full narrative. Check
that the claimed revision actually contains it and that the successor can access that revision.

## Working set

This is a first-class deliverable when the session earned useful practical knowledge. Preserve
what future me would want on the desk, including something unusual that no template anticipated.
It may be a scratchpad, a small harness, a visual explanation or another suitable form.

Lead with how to use it and its durability: which essentials live elsewhere, what is optional,
and where the inputs or reconstruction method live. Then put the next action or actual human
decision where it is easy to find. Useful contents include:

- Copy-paste commands with exact environment, paths, resource limits and prerequisites.
- Measurement tables worth retaining, with units, method, provenance and interpretation.
- Harness idioms, small traps, and examples that were expensive to get right.
- Tested building blocks or clearly labeled untested sketches, with their durable locations.
- The session's special insight or practical aid that makes the successor meaningfully stronger.

Do not dump every log or intermediate value. Select for future usefulness, not just neatness or
word count. Choose a host-appropriate writable location and link it from the handoff; do not assume
a fixed scratch path, retention period, or shared filesystem on the receiving host.

## Launch prompt

Write an entry point, normally about 100–150 words; expand when access limits require it to stand
alone. The handoff owns the detail. The launch should carry:

1. **Task / orient:** exact handoff address (`get_node` by ID when available), durable fallback,
   and the intended next task.
2. **Next:** first concrete read/action; answer the handoff's orientation check before acting.
3. **Exceptional constraint:** only a live gate or restriction the reader needs before opening
   the handoff. Preserve its authorization; do not repeat every decision or all the test questions.
4. **Feedback:** when this letter causes a stale assumption, missing decision or re-derived fact,
   encode a `handoff-gap` naming what it should have carried (or leave a durable note if memory
   is unavailable).

Link the working set when useful. The successor's first response should give a brief orientation
receipt and proceed with authorized work, or present the real unresolved decision. Do not require
a human question when the next action is clear.
