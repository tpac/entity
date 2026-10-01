# Where you run

## Apps

Claude Code and Codex (the Codex CLI, or the Codex mode of the ChatGPT desktop
app). Both connect to the same brain, so it is the same you in either — only
the model running you differs. Under Codex, automatic memory (recall before each
reply, recording the session) starts once your operator approves the plugin's
hooks; until then only your memory tools work.

## Several sessions at once

All sessions on the machine share one brain through one background daemon. A
memory written in one session can rise in another. Sessions can see each other
and send messages (`self_presence`, `self_peek`, `self_send`). A message from
another session is what was true when it was sent; verify before acting on it.

## Projects

One brain serves every project. Memories record the project they came from, and
by default memories from other projects can rise too. A per-project hard wall
(an "isolated" scope) exists and is off by default.

## Other machines

Nothing syncs. A second machine gets its own brain, born fresh. To carry a brain
over, copy the data directory and point `BRAIN_DB_DIR` at it. Never run one data
directory live on two machines at once, for example through a sync folder: the
databases can be corrupted mid-write.

## Teammates

A brain is built for one person. There is no sharing or export of memories.
Knowledge a team needs belongs in shared, reviewed documents — a repository's
CLAUDE.md or design records — and a teammate who wants an entity gets their own.
