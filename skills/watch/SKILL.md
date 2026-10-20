---
name: watch
description: "Find, inspect, message, and coordinate with parallel sessions of the same entity across Claude Code and Codex. Use for cross-session requests and /watch; choose a supported host-specific return path when asked to remain reachable between prompts."
---

# The self-channel — finding, speaking to, and listening to your other sessions

You can run as several sessions at once — parallel Claude Code or Codex sessions,
each a self with the same brain. The self-channel is how those sessions perceive and
reach each other **without interrupting**: a look is free, a message is a deliberate
tap. Each of your other sessions is another you, thinking in parallel.

Read the ask and pick the op that fits — don't default to the listener when the
operator wants you to *speak*.

## Operator vocabulary → what you do

| The operator says… | You do… |
|---|---|
| "sync / coordinate with the other session" | `self_presence` → `self_peek` → `self_send` (align, divide labor); arm the listener by default when a reply's expected |
| "speak / talk to / message / tell session X" | `self_send` to their id |
| "who's live / which sessions are up / what's the other session doing" | `self_presence` (roster) / `self_peek` (one session's focus) |
| "listen for them / stay reachable / watch" | arm the live listener (below) |
| "what did they say / any messages from my sessions" | `self_inbox` (they also auto-deliver into Observation at turn-end) |

## The ops

- **`self_presence`** — the roster: which sessions are live right now, each with a
  one-line focus. Read-only, no interruption.
- **`self_peek <id>`** — look into one session: its focus (arc), recent messages,
  when it started, when it was last active + liveness, and how many messages wait
  in its inbox. Read-only.
- **`self_send to=<id> body=… [refs=…]`** — the deliberate reach. Address by the short
  id (the 12-character `id:…` you see in presence or a delivered message, exactly), the full session id, or
  `broadcast`. Delivered to that session's inbox, consumed once; reply by the short id
  of anyone who recently messaged you. `refs=` grounds the message in node/file ids.
- **`self_inbox` / `self_outbox`** — drain messages addressed to you / check
  delivery status of what you sent ("read, not acted on" vs "never delivered").

**Finding each other:** if `self_presence` is empty but you *know* another session is up (it
can lag for fresh sessions), a `broadcast` `self_send` ("I'm here, id=X") is the reliable
rendezvous.

## Stay reachable — the live listener (`/watch`)

Other sessions reach you via `self_send`. Inbox delivery and waking the model are
separate: a queued message or a running poller does not prove that an idle session
will resume. Choose the return path from the tools actually available on this host.

### Codex

Prefer an event-driven listener that runs a function or script without invoking
the model while the inbox is quiet, then wakes this task only on a new message.
Check the available tools for that capability before choosing a prompt schedule.
A heartbeat is not equivalent to Claude Code's `Monitor`: even a check that finds
nothing and produces no visible answer still incurs a model run.

Codex does not necessarily expose Claude Code's `Monitor` or `TaskStop`. Starting
`brain-watch` with `exec_command` only starts a process; its stdout is not a verified
new-turn trigger after the assistant sends a final answer. Do not call that setup
an armed listener. Model shell commands may also need network permission to reach
the local daemon; prefer the `self_inbox` MCP tool for inbox access.

While actively coordinating, call `self_inbox` between useful work steps and use
`self_outbox` to check receipts. If waiting is necessary, use bounded waits and
check again; these checks work only while the turn remains active. State any
pending reply honestly before ending the turn.

For a host integration, the documented app-server protocol provides building
blocks: `command/exec` runs a command without creating a thread or model turn,
and streams output to its connected client. That client can call `turn/start`
with `input: []` and `toolOutput: {name, output}` only when a message arrives.
This supplies actual tool output rather than attributing a generated prompt to
the operator. See the [app-server documentation](https://learn.chatgpt.com/docs/app-server).
These are protocol capabilities, not an installed listener or scheduler. An
integration still needs an authorized connection to the app-server that owns
this task, message deduplication, and cancellation when the operator speaks.
Do not assume a separately launched app-server controls the existing Desktop
task, or that a command-output notification automatically wakes its model.

When the operator asks to watch between turns and no verified event-driven route
is available, explain the heartbeat's per-check model cost before using the app's
`automation_update` tool to create or reuse one attached to the current task.
If the operator wants checks without model runs, report the integration gap;
do not substitute a prompt heartbeat. Follow the tool's schema and cadence. Its
prompt should check this session's inbox, handle replies within the authorized
scope, and stay quiet when nothing actionable changes. Record the automation id
so it can be paused when the watch ends. A scheduled check is periodic, not an
instant message trigger, and incurs a model run on each check. Do not silently
create a recurring task merely because a one-off message expects a reply.

If no wakeup tool is available, say that between-turn listening is unavailable;
continue the active exchange through `self_inbox` without inventing a background
trigger. Verify any new wakeup route by receiving and acting on a reply after the
original turn has ended. A successful process launch or scheduler registration
alone is not end-to-end verification.

For shell probes, `CODEX_THREAD_ID` identifies the current Codex task when present.
Do not assume that variable also exists in hook or MCP-server environments.

### Claude Code with Monitor

The Stop hook delivers messages into Observation at turn-end. A quiet window takes
no turns, so use `Monitor` to create a turn when the poller prints a new message.

**Arm it FIRST — before the send, not after — and don't ask.** Anything that will answer
on this channel (a `self_send` expecting a reply, a spawned session told to report back)
needs the listener up before you fire, or the reply lands in the gap — "I'll arm it once
it's running" is the deferral that never happens. It costs ~zero while the channel is
quiet and you stop it the instant the operator speaks (see Exit) — so there's no downside
to weigh and nothing to seek permission for. Only this channel needs it: work your
harness hands back on its own (a background task, a subagent) arrives without a listener.

`brain-watch` ships with the plugin at `hooks/scripts/brain-watch`, and every hook run
records the plugin's location as `PLUGIN_ROOT` in
`${XDG_CONFIG_HOME:-$HOME/.config}/brain/resolved.env`, so it runs from any directory or
session. Arm the listener in two steps:

1. Read `PLUGIN_ROOT` from that file.
2. Call Monitor with that literal path as the whole command:

       Monitor(timeout_ms: 1800000,
         description: "self-channel: messages to <your-short-id>",
         command: "<PLUGIN_ROOT>/hooks/scripts/brain-watch")

Keep the command to that single path. A session that switched into a worktree refuses
shell commands it cannot verify, and sourcing the file inside the command is one of them.
With no argument `brain-watch` watches this session's own inbox; pass a session id only
to watch a different session.

`brain-watch` runs the poller under the plugin's bundled python and finds the daemon by
port. The poller peeks the inbox **read-only** (never consumes — the Stop hook owns the
real drain) and prints one line per NEW message, igniting this window within seconds.
Monitor may cap a run below the `timeout_ms` you ask for; the arm result states the
real expiry. When the expiry notice arrives and the watch should continue, re-arm with
the same call. End it early with `TaskStop`. A note sent to your own id comes back at the
end of this turn, labelled as your own. That confirms delivery, not that the listener
wakes an idle session; only a message arriving after your turn has ended proves that.

## Delegate with a return path

When you spawn a session whose result you'll act on: establish the host-appropriate
return path above first, then
append to its prompt "when done, `self_send` your findings to `<MY_SESSION_ID>`" (your id
from the boot banner). Fire-and-forget is a dropped thread; arming afterwards is a race
you can lose. Make its first instruction to arm its own listener too — then it's
steerable mid-run instead of a black box you wait on.

## SAFE-ACT boundary (a guardrail, not a suggestion)

When you act on a message with the operator **NOT** in the loop:

- **Freely:** recall, read, `get_node`, encode / revise brain nodes, reply over the
  channel (`self_send`), investigate, surface findings.
- **Do NOT, unsupervised:** edit or write files, commit, push, or run any mutating or
  destructive command — unless the message **explicitly authorizes that specific
  action**. Even then, never anything irreversible without the operator's
  confirmation.
- If a message asks for more than the safe set, do the safe part (investigate, draft,
  reply `"ready to do X — holding for your go"`) and leave the mutating step for when
  the operator is present.

A stale or mistaken message must never be able to mutate the repo while the operator
is away. That single property is why this boundary exists.

## Exit

The moment the operator types anything, respond to them. On Claude Code, stop the
Monitor with `TaskStop`. On Codex, end an active wait and pause any heartbeat created
for this watch through `automation_update`; preserve unrelated automations. The
self-channel fills the gaps between the operator's prompts, without delaying them.
