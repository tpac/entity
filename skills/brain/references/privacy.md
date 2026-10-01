# Privacy — what stays and what leaves

## What stays on the machine

Everything below lives in one data directory — `$XDG_DATA_HOME/brain`
(`~/.local/share/brain` when unset), wherever `BRAIN_DB_DIR` points, or, on
older installs, an earlier location the daemon still resolves:

- Your brain: `brain.db` (memories and connections) and `brain_logs.db` (the
  verbatim record of sessions).
- Local backups, in `backups/`.
- Working files kept for diagnosis: `payloads/` (the prompts sent for writing
  and consolidating memories, kept 14 days) and, only on a brain where
  development turned it on (`surface_capture.enabled`), `surface_captures/`
  (for each recall, the prompt, recent messages and candidate memories it saw —
  kept until removed by hand; `BRAIN_SURFACE_CAPTURE_DIR` moves them). Earlier
  versions captured on every install, so an older install may still hold a
  `surface_captures/` that nothing adds to any more; it is safe to delete.

Your configuration is `~/.config/brain/env` (API key, names). Embeddings are
computed on the machine by a local model. The dashboard serves on `127.0.0.1`
only.

## What leaves

1. **The model you run on** sees everything said in the session and every memory
   that rises into it. That goes to whoever hosts the model: Anthropic under
   Claude Code, OpenAI under Codex.
2. **Your own memory work**, once an API key reaches the daemon (from the
   config file or the plugin settings — on macOS a key exported only in a
   shell does not reach it): writing memories
   (conversation turns and related memories), choosing what rises for each
   prompt (the prompt, recent messages and candidate memories), and background
   work across sessions (memories, with excerpts of the conversations they came
   from). It goes to Anthropic's API on that key. The client follows an
   `ANTHROPIC_BASE_URL` set in the config file, so an operator can route it
   through a proxy or another endpoint.

Nothing else: no telemetry, no account, no cloud copy of the brain. Installation
downloads the runtime and the embedding model; no conversation data goes with it.

## Without an API key

Storage, the session record and recall still work. Learning new memories and
automatic surfacing pause, and your own memory work sends nothing.

## What to tell your operator

Storage is local and theirs; thinking happens at the model providers above. If
something must never leave the machine, the only reliable protection is not to
say it in a session — what is said can become a memory, and a memory is sent
again whenever it rises.
