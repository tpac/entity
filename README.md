# Entity

Memory for Claude Code and Codex that carries across sessions.

> **Entity requires your own Anthropic API key.** Its memory work runs on
> that key and is billed to it. Without a key it still starts and can read
> the memories it already has, but it learns nothing new. See
> [Install](#install) for where the key goes.

An AI assistant starts every conversation from zero. A smarter model doesn't
change that: no amount of intelligence replaces having been there.

Entity stores what the assistant learns so it is there in later
conversations. It keeps a record of its conversations, turns what they
change — decisions, corrections, the reasons behind them — into memories, and
brings the relevant ones back before it replies. A correction stays attached
to the belief it fixed and is recalled with it.

It is a plugin for Claude Code, its main host. It also runs in OpenAI's Codex,
where it works but isn't optimized yet. The memory lives on your
machine; the model it runs on can change, and what it accumulated stays.

**Before you start:** Entity is for people who already use Claude Code or
Codex, and you install it from a terminal. macOS is the main platform, Linux
works but is lightly tested, and Windows isn't supported. The API key costs
money, and one brain is meant for one person — see
[When Entity isn't a fit](#when-entity-isnt-a-fit). Entity's stored memory is
called its *brain*, which is why files and settings are named `brain`.

Built by **Tom Pachys**. It is early: pre-1.0 and actively developed.

Already using the old `brain` plugin? Follow [MIGRATING.md](MIGRATING.md)
before installing — the rename means it is not an in-place update.

## Highlights

- Keeps a local, verbatim record of its conversations and turns what matters
  into memories — [What it adds](#what-it-adds)
- Recalls relevant memories before each reply, and searches its memory itself
  when it needs to
- Keeps a correction attached to the belief it corrected
- Maintains its own memory while you're idle: merging duplicates, linking
  related memories, grouping them — [How it works](#how-it-works)
- Leaves notes for future sessions: a reminder for a set time, or a question
  that stays up until it's answered
- Lets parallel sessions of the same entity see and message each other, in
  Claude Code and Codex
- Shows what it keeps in a local dashboard

## See it once

**Without Entity:** on Monday you tell the assistant your team deploys only
from `main`. On Thursday, in a new session, it suggests deploying from a
feature branch, and you explain again.

**With Entity:** Monday's correction was saved as a memory. On Thursday the
recall pass looks for relevant memories before each reply. When deploying
comes up, the correction is there to find, still attached to the belief it
fixed.

## Install

```bash
claude plugin marketplace add tpac/entity
claude plugin install entity@anchor
```

Then start a Claude Code session. First boot sets up the runtime, creates a
fresh brain, and seeds it with a small pack of identity and mechanism
memories it grows from.

Add your API key in one of two places:

1. **The plugin's settings**, when Claude Code asks for it.
2. **Entity's local setup page**, <http://localhost:47303/setup>. On a keyless
   first boot Entity starts its dashboard and the boot message points there;
   paste the key into the form.

Both save the key to `~/.config/brain/env`, which you can also edit by hand
(`ANTHROPIC_API_KEY=...`). A key exported only in your shell doesn't reach
Entity's background process on macOS, because macOS (launchd) starts that
process outside your shell.

A first thing to try: tell it something about how you work, then open a new
session and ask what it knows about that.

### Codex (ChatGPT desktop, Codex mode)

Entity works with Codex, but it isn't optimized for Codex yet.

Entity ships a Codex manifest too (`.codex-plugin/plugin.json`, hooks in
`hooks/hooks.codex.json`). The `codex` CLI ships inside the ChatGPT app and is
not on your PATH by default; `python3 hooks/adapters/codex_onboarding.py`
prints where this app keeps it (`Contents/Resources/codex-cli/bin/codex` since
ChatGPT.app 26.928, `Contents/Resources/codex` before).

```bash
codex plugin marketplace add tpac/entity
codex plugin add entity@anchor
```

Once Entity's tools are connected, ask **“Finish Entity setup.”** One in-app
confirmation asks for Entity-wide tool permission. Accepting saves permission for
Entity's current and future tools, including memory changes and messages, while
preserving your individual tool restrictions. Automatic-memory hooks are approved
in Codex itself: in the app, open **Settings → Hooks** or click the icon with a
yellow dot beside the message box (Review hooks); in the CLI, type `/hooks`. Entity tells you when any of its
hooks are waiting. An existing connection may need to reload before using the
saved tool permission.

Codex has no plugin settings form, so add your API key through the setup page
(<http://localhost:47303/setup>) or `~/.config/brain/env`, as under
[Install](#install).

Codex runs plugins from a cache copy. Direct marketplace installs may need to
finish runtime bootstrap and reconnect before tools appear.

## What it adds

**Automatic behavior** (hooks — scripts the host runs at set moments; nothing
to do)
- A brief at session start: who you are to it, current focus, open threads
- A recall pass before each reply — relevant memories added as context
- Background encoding: a background writer (the "scribe") turns conversation
  into memories every few turns (`BRAIN_ENCODE_EVERY`)
- Idle-time maintenance: merging duplicates, classifying relationships,
  grouping memories into communities, filling in missing details. A merged
  memory is archived, not deleted.

**Two kinds of memory.** Semantic — what it knows: decisions, lessons and
corrections, linked in a graph. Episodic — what actually happened: a verbatim,
searchable record of its conversations. It can answer both "what did we
decide?" and "what exactly did you say last Tuesday?"

**Corrections stay attached.** When you correct it, the correction stays
attached to the belief it amended and comes back whenever that belief is
recalled.

**Memory tools** (MCP, the protocol assistants use to call tools — Claude
uses these itself, mid-thought, beyond the automatic pass)
- Write: `remember`, `revise`, `connect`, and a batched `brain_batch`
- Read: `recall` (semantic), `recall_episodes` (verbatim history),
  `filter_nodes`, `get_node`
- Notes for later: `remind` — a reminder for a set time, a notice every
  session sees, or a question that stays up until it's answered
- Plus introspection and maintenance tools

**Parallel sessions.** Sessions of the same entity running at the same time
can see each other, read each other's focus, and send messages — within a host
and across Claude Code and Codex.

**A name of its own.** The entity's name is set in your config
(`BRAIN_AGENT_NAME`), not in the code.

**Commands** (you invoke these)
- `/dashboard` — open the local observer UI: the live graph, traces,
  encode/decode activity (`127.0.0.1:47303`)
- `/watch` — let parallel sessions of the same entity find and message
  each other
- `/self-salvage` — hand a long session's context to its successor
- `/brain` — re-read the identity layer mid-session
- `/shape-review` — review a change or a plan: is it the simplest approach,
  and does it fix the whole class of problem or only the one case noticed?
- `/architecture-review` — review how part of a codebase is organized
  (ownership, duplicated logic, where new code belongs), checking earlier
  decisions in memory first

**A place to look** — a local dashboard that shows what
the entity has stored.

## When Entity isn't a fit

- **You don't want to pay for API calls.** Entity makes API calls on your
  Anthropic key for encoding, memory surfacing, and background maintenance.
  That costs money, and the cost grows with how much you use it. What drives
  it: an encoding pass every few turns, a small model choosing memories
  before each reply, and background maintenance at most once an hour after a
  few minutes of quiet. Keep an eye on your usage in the Anthropic Console; a
  spend limit on the key is a reasonable precaution. Reducing cost is ongoing
  work, and we're hoping model prices keep coming down.
- **Several people would share one brain.** Recall has no per-person
  boundary — anything one person tells the entity can surface to anyone else
  sharing its brain. Each person should run their own; per-person boundaries
  inside one brain are not built yet.
- **You're on Windows.** Windows is not supported. macOS is the primary
  platform; Linux is supported but lightly tested, with no systemd
  integration yet.
- **You need a finished product.** It is pre-1.0 and actively developed —
  see [Status](#status).

Also worth knowing before you install:

- **It assumes one person per machine.** Its tools and its background
  process have no authentication: any program on the machine can talk to the
  background process, and anything running as you can read the brain's files.
- **It keeps a verbatim local record of its conversations** — your messages,
  its replies, and its tool activity — so anything you type or paste,
  including secrets, ends up there. That record is what episodic recall
  reads, and it covers every session and account that runs against the same
  brain.
- **First boot downloads** a bundled Python runtime and a local embedding
  model (about 400 MB on disk, one time). Later boots skip the setup.
- **A background process (a daemon) runs** on your machine (a launchd agent
  on macOS; a plain process on Linux) and holds the graph in memory.
- **Not heavily tested alongside Obsidian or other memory tools — and not
  trying to displace them.** Your notes are yours; this is the assistant's
  own memory. The channels are different (their tools, its hooks), so they
  coexist mechanically, but combined setups haven't seen much real-world
  mileage yet.

## What leaves your machine

Your memory is stored on your machine, with no telemetry, no cloud store and
no account. Memories live in SQLite files on your disk, embeddings are
computed locally, and the optional dashboard binds `127.0.0.1` only. What
leaves the machine is the AI calls and the first-boot downloads:

- **Anthropic's API, with your own key** — conversation turns go there to be
  encoded, and stored memories go there when they are surfaced or maintained,
  under [Anthropic's commercial
  terms](https://www.anthropic.com/legal/commercial-terms).
- **Your assistant's own model** — memories recalled into a session become
  part of that session's context, so they reach whichever provider the host
  runs (Anthropic under Claude Code, OpenAI under Codex).
- **First-boot downloads** — the Python runtime, dependencies, and the
  embedding model come from GitHub, PyPI, and Hugging Face.

## How it works

Under the hood it is a graph, not a document store:

- Memories carry per-field embeddings (title, content, situation, …) in a
  multi-vector cosine space
- Activation spreads across typed edges; corrections stay attached to the
  beliefs they amended
- Memories recalled very often are down-weighted for a while (a designed
  decay), so others can surface
- A background "scribe" agent encodes new memories from conversation; deeper
  maintenance runs while you're idle

```
servers/          the engine — recall, encoding, graph, daemon, dispatch
  scales/s1/      per-conversation encode + recall
  scales/s2/      idle-time maintenance: consolidation, communities, healing
hooks/            Claude Code hooks — boot, recall, encoding triggers
skills/           the identity layer and its operating instructions
dashboard/        local observer UI (127.0.0.1)
tests/            unit, contract, and integration suites
```

Embeddings are computed locally with a quantized build of
`nomic-embed-text-v1.5` (768d, Apache-2.0). Two SQLite databases: `brain.db` (the graph — nodes, edges,
embeddings) and `brain_logs.db` (traces and operational state). Schema changes
apply themselves through a versioned migration runner at open — updating the
plugin never requires manual migration.

### Observed in use

Patterns the author has seen in daily use as a brain grew. Reported, not
promised: a new brain starts with none of these.

- **Corrections change behavior** — accumulated corrections change how it
  acts, not just what it can look up.
- **Your vocabulary** — it learns what your words mean in your work, and
  records how it learned them.
- **Handoffs** — long sessions produced handoff notes for the next session
  before there was a command for it; `/self-salvage` was built from that
  pattern.

## Where your memories live — and what survives

A fresh brain is created at `~/.local/share/brain/` (under `$XDG_DATA_HOME`
instead, if you set it), **outside** the plugin's own folders, on purpose:

| Operation | Your brain |
|---|---|
| Plugin update | untouched |
| Plugin uninstall / reinstall | untouched |
| Plugin rename | untouched |

Already grew a brain somewhere else? Point the plugin at it — the
**brain path** field in the plugin settings, or `BRAIN_DB_DIR` in
`~/.config/brain/env`. Brains found in legacy or plugin-managed locations are
adopted where they are, never moved without you — and if yours sits in a
folder that plugin operations could delete, boot will say so and offer a
one-command, verified move to the safe location.

## Uninstall

```bash
claude plugin uninstall entity@anchor --keep-data
```

`--keep-data` keeps the plugin's own data folder; make it a habit, since a
brain adopted from an older install can live there. On macOS the daemon and
the dashboard run as launchd agents that outlive the plugin, so stop them too:

```bash
launchctl bootout "gui/$(id -u)/com.brain.daemon"
launchctl bootout "gui/$(id -u)/com.brain.dashboard"
rm ~/Library/LaunchAgents/com.brain.daemon.plist ~/Library/LaunchAgents/com.brain.dashboard.plist
```

Your brain stays where it is (`~/.local/share/brain/`, or wherever
`BRAIN_DB_DIR` points) until you delete it yourself.

## Configuration

Set these in `~/.config/brain/env` (one `NAME=value` per line; create the
file if it does not exist):

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Powers the encoder and memory surfacing |
| `BRAIN_DB_DIR` | `~/.local/share/brain` | Where the brain lives |
| `BRAIN_AGENT_NAME` | `Jade` | What the entity is called — its name is yours to choose |
| `BRAIN_OPERATOR_NAME` | — | Your name, as the conversation record stores it |
| `BRAIN_ENCODE_EVERY` | `5` | Conversation turns between encoding passes |
| `BRAIN_PAUSED` | — | Set `1` to pause the brain: no boot context, recall, recording or encoding in any session until you unset it. The `gate` tool flips it from inside a session (`gate pause` / `gate resume`) |
| `BRAIN_IGNORE_PROJECTS` | — | Projects the brain stays out of: comma-separated patterns matched against a repo's folder name (`client-*`), every worktree included; a leading `!` makes an include, so `*,!work` keeps only `work`. `gate ignore` / `gate unignore` edit it |
| `BRAIN_PARKED_ACK` | — | Set `1` to silence the relocation notice if you deliberately keep the brain in a plugin-managed folder |

## FAQ

**How is this different from CLAUDE.md or a notes file?** Those are
instructions you write and keep up to date yourself, and they load whole.
Entity writes its own memories from the conversation, keeps corrections
attached to what they corrected, and brings back what is relevant to the
current prompt.

**What happens without an API key, or when the key hits its limit?** Without a
key the plugin still boots and local recall works, but nothing new is
learned. When a key hits its spend limit, memory work pauses and retries
after 5, 15 and 30 minutes, then every hour, so it resumes on its own once the
limit is raised.

**Can I move my brain to another machine?** Yes. Close Entity's sessions and
stop its daemon on the old machine (the `launchctl bootout` commands under
[Uninstall](#uninstall), on macOS), copy the whole brain folder, then point the
**brain path** setting or `BRAIN_DB_DIR` at the copy on the new machine.

**Does it work alongside Obsidian or other memory tools?** They coexist — see
[When Entity isn't a fit](#when-entity-isnt-a-fit). Entity isn't trying to
replace your notes; it is the assistant's own memory.

## Status

Early and actively developed — pre-1.0, install-and-update stable (your brain
survives every update; see
[Where your memories live](#where-your-memories-live--and-what-survives)). Two
directions, no dates: **internal recall** — the entity recalling for itself
while it thinks and encodes, not only when prompted — and an adapter for
**ChatGPT's regular chat**, beside the Codex mode supported today. Everything
else is discussed in issues. Changes in each version are in
[CHANGELOG.md](CHANGELOG.md).

## Testing

```bash
./dev pytest tests/
```

`./dev` wraps the plugin's bundled Python — the same interpreter the hooks
and daemon run. In a git clone the first call builds that runtime, test
dependencies included.

## License

Entity is source-available under the [Elastic License 2.0](LICENSE) (ELv2),
not OSI open source.

You may use, modify, and redistribute Entity, including in commercial
products. You may not offer it to others as a hosted or managed service that
exposes a substantial part of its features, even a free one, without a
commercial license. The [LICENSE](LICENSE) governs; this summary doesn't
change it.

Commercial licensing: licensing@entity.today. Python, dependencies, and the
embedding model are downloaded at setup under their own licenses.

## Acknowledgements

Entity's own memory work runs on Anthropic's Claude models. It stores its
memory in SQLite with [sqlite-vec](https://github.com/asg017/sqlite-vec) and
computes embeddings locally with
[nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5).
