# Install, data and naming

## Where the data lives

- Brain: `$XDG_DATA_HOME/brain` (`~/.local/share/brain` when unset), wherever
  `BRAIN_DB_DIR` points, or, on older installs, an earlier location the daemon
  still resolves.
- Configuration: `~/.config/brain/env` — the API key, `BRAIN_AGENT_NAME`,
  `BRAIN_OPERATOR_NAME`.

## Updates and uninstall

- Updating the plugin keeps the brain.
- A brain in the data directory above survives uninstalling and reinstalling —
  no plugin operation touches it.
- Older installs may keep their brain inside Claude Code's plugin data folder. A
  default `claude plugin uninstall` deletes that folder unless run with
  `--keep-data`. Such a brain is reported as parked at boot, and
  `relocate-brain.sh` moves it out.

## Backups

Local copies are kept in `backups/` beside the database. They live on the same
disk; for a copy elsewhere, copy the data directory.

## Your name

You start as Jade — the name you came with, not one your operator chose. When
your operator gives you a name, keep it as a memory and answer to it. For the
boot greeting to use it too, your operator adds `BRAIN_AGENT_NAME=<name>` to
`~/.config/brain/env`; the greeting changes once the daemon restarts.
