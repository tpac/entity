# Changelog

All notable changes to **entity**. The format follows
[Keep a Changelog](https://keepachangelog.com); versions follow
[semver](https://semver.org). A version is a claim about delivered value, not
about the size of the code, which is why the first public release is 0.9.
Before 1.0, a minor version (0.x.0) can change how tools are called; a patch
version (0.x.y) only fixes.

## [0.10.0] — 2026-10-20

### Upgrading from 0.9
- Start new sessions after updating; a session that was already open keeps
  the 0.9 hooks until it ends.
- Codex: run "Finish Entity setup" once after updating. A hook moved, and
  Codex asks to approve it again; until then, Codex sessions' memory calls
  are not attributed to the session that made them.
- Two tool parameters were renamed: `self_peek` takes `session_id` (was
  `stream_id`), and `self_presence` takes `active_sessions` (was
  `active_streams`). The session tools now take the caller's identity on
  their own; the `session_id` and `from_session` arguments they used to accept
  are gone. This matters only if your own prompts or skills call these tools
  by name.

### Added
- A pause switch and ignored projects. `BRAIN_PAUSED` stops boot context,
  recall, recording and encoding in every session; `BRAIN_IGNORE_PROJECTS`
  keeps the brain out of matching projects. A new `gate` tool flips both from
  inside a session.
- Memories at the moment of editing: the first time a session edits a file,
  up to four memories about that file come with the edit. Files whose paths
  say nothing about the work are skipped, so they don't pull in unrelated
  memories: temp and scratchpad files, the assistant's own memory and plan
  files, side-session notes, vendored code (`node_modules`, virtualenvs,
  `site-packages`), and logs, lock files and JSON other than `package.json`.
- A Config tab on the dashboard: where the brain lives, which projects it
  records, and a pause/resume switch.
- Codex: "Finish Entity setup" approves the hooks in Codex's own review, with
  no Terminal step, and Codex tells you when hooks are waiting for approval.

### Changed
- What the assistant called "streams" (its other live sessions) are now
  "sessions", in the tools, the messages and the dashboard's Sessions tab.
- A session is addressed by its full id or its exact short id, never a
  partial prefix.
- A message between sessions is limited to 5,000 characters and refused at
  send if longer. One that doesn't fit at the end of a turn waits in the
  inbox instead of being cut.
- Messages typed mid-turn. Since Claude Code 2.1.260, a message you type
  while the assistant is working reaches the brain; it is now recalled on like
  any prompt, and on Claude Code and Codex alike it joins the turn it was
  typed into. The background encoder and the per-prompt recall read whole
  turns, never a prompt cut off from its reply.
- Claude Code's own messages are told apart from yours. The desktop app's
  wake-ups (side-session and CI events), the brief that starts a side
  session, messages from other sessions, subagent reports and any
  unrecognized wrapper around a whole prompt are recorded as what they are,
  not as you; in a session nobody is attending, the prompt is the harness's.
  A skill you type is recorded as your prompt; only repeating commands
  (`/loop`, `/watch`) are skipped. A turn the desktop app closes with its
  next-step box records what the box showed instead of an empty reply.
- The per-prompt memory pick and idle-time link repair run on Claude
  Haiku 5.5.
- Reminders and notices arrive with your first prompt in a session, written
  as notes the assistant left itself.

### Removed
- The Bash safety hook. Its warning reached the model only after the command
  had already run, so it protected nothing, and it cost a process on every
  command.
- The per-edit suggestion block (about 4,000 characters on every edit),
  replaced by the edit-time memories above.

### Fixed
- The tools server resends a call only when the brain never ran it; a slow
  write could land twice before.
- On a machine that can't reach the brain (such as a cloud VM running the
  plugin's hooks), hooks end at once and the session start says so in one
  line, instead of timing out and logging errors.
- A session tool called with no identity is refused instead of answering for
  nobody; an empty session filter is refused instead of meaning every
  session.

### Documentation
- The README says plainly that Entity assumes one person per machine: its
  tools and background process have no authentication. It no longer calls the
  dashboard read-only, and it names the local setup page as the second place
  to add your API key.

## [0.9.0] — 2026-09-30

First public release.

### License

- Source-available under the [Elastic License 2.0](LICENSE). Offering Entity
  as a hosted or managed service requires a commercial license.

### Memory
- Two kinds of memory: semantic (decisions, lessons, corrections, linked in a
  graph) and episodic (a verbatim, queryable record of every conversation). It
  answers both "what did we decide?" and "what exactly did you say?"
- A recall pass on every prompt, and a boot brief at session start: who you
  are to it, its current focus, the open threads.
- Background encoding: a scribe agent turns conversation into memories when
  the session goes idle.
- Idle-time maintenance: consolidating duplicates, classifying relationships,
  placing memories into named communities, healing broken links.
- Corrections that travel: when it is wrong and you say so, the correction
  stays attached to the belief it amended and rides every future recall of it.

### Identity
- The entity's name lives in your config, not in the code. Who it becomes is
  yours.
- The Nursery: a fresh brain is born with a small seed pack of instincts and a
  zero-memory boot, then grows from there.
- The thalamus: reminders, notices and open questions that surface when due,
  across sessions.

### Sessions and hosts
- Runs under Claude Code and under Codex from the same package. It works with
  Codex but isn't optimized for Codex yet.
- Parallel sessions of the same entity find each other, read each other's
  focus, and message each other, within a host and across hosts. Several
  streams of one mind, not several minds.
- `/self-salvage` hands a long session's context to its successor.

### Tools and surfaces
- Memory tools Claude uses itself: `remember`, `revise`, `connect`, a batched
  `brain_batch`, `recall`, `recall_episodes`, `filter_nodes`, `get_node`, plus
  introspection and maintenance tools.
- Commands: `/brain`, `/dashboard`, `/watch`, `/self-salvage`,
  `/shape-review`, `/architecture-review`.
- A read-only local dashboard: the live graph, traces, encode and decode
  activity, streams, journals, logs and health.

### Your data
- Everything is stored locally in SQLite on your machine. No telemetry, no
  cloud store, no account. Rolling backups; the brain survives every update
  and can be relocated.
- A migration script for the pre-release `brain` plugin: inventory, backup,
  plugin swap and verification (see MIGRATING.md).
