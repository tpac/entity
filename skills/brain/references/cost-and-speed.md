# Cost and speed

## Cost

Your memory work runs on Anthropic models and bills the API key that is set —
your operator's, separately from any Claude or ChatGPT plan. The calls:

- **Encoding** — writing memories from recent turns, every 5 turns by default
  (`BRAIN_ENCODE_EVERY`), plus a pass when a session goes quiet with turns left
  unencoded.
- **Surfacing** — choosing what rises before each reply, using a small model.
- **Background work** — consolidating memories and finding patterns across
  sessions, at most once an hour and only after a few minutes of quiet.

There is no dollar estimate to give: cost grows with use, since every encoding
pass is a call. Your operator should watch it in the Anthropic Console, and a
spend limit on the key is a reasonable precaution. Reducing cost is ongoing
work, and the hope is that model prices fall too.

There is no budget setting inside Entity. Spending limits live in the provider's
console. When the key hits its limit, your memory work pauses and re-checks on
its own (after 5, 15, 30, then 60 minutes); nothing needs restarting.

## Speed

Recall runs before each reply (under Codex, once the plugin's hooks are
approved) and adds seconds, bounded by a 20-second timeout (30 seconds on a
session's first prompt). Encoding and background work run in
the background and never hold a reply.
