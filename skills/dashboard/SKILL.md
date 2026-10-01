---
name: dashboard
description: Open the brain dashboard — the read-only observer UI (graph, traces, live decode/encode, streams, logs) served locally at http://localhost:47303. Invoke `/dashboard` (`/entity:dashboard`) to make sure it's running and open it in the browser. Use whenever the operator says "open / show / launch the dashboard", "let me see the brain dashboard", "bring up the dashboard", or similar. The dashboard is an always-on singleton on a fixed port — you don't launch one per chat, you ensure the one is up and open it.
---

# Open the brain dashboard

The dashboard is the brain's **read-only observer UI** — graph, traces, live
decode/encode, streams, logs — served on a fixed local port (47303 by default,
`$DASHBOARD_PORT` to override). It is a **singleton**: one process, shared by every
session (like the daemon). You never launch one per chat — you ensure the single
one is up, then open it. The first `/dashboard` **installs** the singleton; from then on launchd keeps it
alive and `/dashboard` just opens it.

When invoked:

1. **Ensure the dashboard is running** — run the ensure script (idempotent; it
   handles port resolution, first-run install, restart, and waiting).
   The Bash shell has NO `CLAUDE_PLUGIN_*` vars (those exist only in hook
   executions) — locate the plugin via the state file the brain persists on
   every boot:
   ```bash
   . "${XDG_CONFIG_HOME:-$HOME/.config}/brain/resolved.env" 2>/dev/null
   if [ -n "$PLUGIN_ROOT" ]; then "$PLUGIN_ROOT/hooks/scripts/ensure-dashboard.sh"
   else echo "no resolved.env — brain has not booted on this machine yet"; fi
   # from the brain repo root, simply:  hooks/scripts/ensure-dashboard.sh
   ```
   If it reports no `resolved.env`, the brain hasn't booted on this machine
   since install — tell the operator to send one message (any session) so boot
   persists it, or find the plugin dir under `~/.claude/plugins/` and run the
   script from there.
   - already up → no-op;
   - down + first run (macOS) → **installs the launchd singleton** (materializes
     the plist for this machine + loads it); launchd's RunAtLoad + KeepAlive keep
     it up across reboots/crashes thereafter;
   - down + already installed → kickstarts it;
   - non-macOS → detached fallback (no boot persistence yet).

   It exits 0 once the dashboard answers (waits up to ~15s; if it blocks the tool,
   run it in the background and poll). On non-zero exit, surface
   `$BRAIN_DB_DIR/dashboard.log` rather than failing silently.

2. **Open it** — two ways, both pointing at the SAME running singleton (never a
   second server), port 47303 by default:
   - **In a pane** (when the operator wants it inside their chat app): if your
     host gives you a browser tool — a built-in browser pane or a browser
     extension — navigate it straight to `http://127.0.0.1:47303/`. Shows the
     live singleton directly — no spawn, no launch config. If there is no such
     tool or the call errors, say so and fall back to the browser below.
   - **In the system browser** (default / fallback): macOS `open
     "http://127.0.0.1:47303/"`, Linux `xdg-open "http://127.0.0.1:47303/"`.
   - **Always also print the URL** as a clickable link — the operator's way in if
     neither opener is available.

3. **Report** — confirm it's open at `http://localhost:47303`, and whether it was
   already up, **just installed** (first run), or restarted.
