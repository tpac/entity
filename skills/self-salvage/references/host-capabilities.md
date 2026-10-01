# Host capabilities for salvage

Read when using history, delivering a successor, choosing a fork, or handling unavailable tools.
The current session's callable tools and their schemas win over these examples. Keep the continuity
workflow shared; adapt only the mechanism. Do not assume a model, filesystem or live process is
inherited by a new session.

| Need | Codex, when exposed | Claude Code or another host, when exposed |
|---|---|---|
| Recover missing history | `read_thread`, bounded to the relevant task and turns | Episodes, journal notes or the host's history reader |
| Start a successor | `create_thread` starts work; use only when the user explicitly requests a new task | Inspect the actual creation tool: it may start work or offer a chip |
| Deliver to an existing recipient | `send_message_to_thread` when the user authorizes that destination | Use the available session messaging control with the same authorization |
| Fork | `fork_thread` can copy completed history; active unfinished work may be absent | Use the supported fork/resume control, if any |
| Prepare a clickable entry | No chip capability should be assumed; a launch file/message is a fallback | `spawn_task`, if present, may offer a chip requiring a click; verify its result |
| Observe progress | Use returned task handles with `wait_threads` / `read_thread` | Use the host's actual handles and status tools |

For creation, select the project/environment using the tool's requirements and verified repository
state. Preserve the user's requested host and model; otherwise keep host defaults. Pass the exact
reviewed launch prompt. For an existing recipient, preserve its settings and send the prompt there
when requested; creation is unnecessary. A pending setup ID is not a running task ID. For a fork,
check what history the control can actually copy; do not promise an arbitrary rewind point. If the
requested continuation needs a follow-up prompt, deliver it using the returned valid task handle.
A prepared entry or chip
is delivered, not started. Never claim a task or background job survives without checking its lifecycle.

Before reading history, choose the missing question and bound the read. Extract only relevant
user/assistant text and necessary evidence; do not dump raw tool records, signatures, credentials
or unrelated arguments into context or a handoff. Check the actual API schema before calling
memory tools (for example, do not substitute an invented query field).

When memory is unavailable, use accessible durable files, versioned artifacts and primary history;
preserve pending knowledge there and report that it has not been encoded in the brain. When no
usable delivery control or authorized destination/action exists, hand the operator the launch
prompt and exact artifact address. Missing scratch loses accelerants; missing durable evidence is an explicit
limitation to repair or carry forward, not permission to invent facts.

When a successor is requested and started, check its first orientation receipt through the available
status/history control. Does it identify the intended work, honor settled decisions, choose the
right first action, and notice missing prerequisites? Repair missing evidence before consequential
work when possible. If it has not reached that point, report started but receiver unchecked.
An isolated probe can test the instructions; it does not prove actual host delivery or fresh boot.
