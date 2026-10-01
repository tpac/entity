# Changelog

All notable changes to **entity**. The format follows
[Keep a Changelog](https://keepachangelog.com); versions follow
[semver](https://semver.org). A version is a claim about delivered value, not
about the size of the code, which is why the first public release is 0.9.

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
