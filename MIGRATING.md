# Moving from the old `brain` plugin to `entity`

The plugin was renamed from `brain@brain` to `entity@anchor`, so no update
carries an old install across. One script does the move and keeps your brain.
It runs in three steps, changes nothing until you tell it to, and never deletes
anything.

Use a plain Terminal window for all of it, not a Claude Code session.

## 1. Look

Download the script and run it. This step only reads.

```bash
curl -fsSLO https://raw.githubusercontent.com/tpac/entity/main/migrate-from-brain.sh
bash migrate-from-brain.sh
```

Send the whole output to the person who pointed you here, and wait for their
go. If it ends in **NOT READY**, the lines above that say why.

## 2. Move

Quit every Claude Code and Codex session, the Claude desktop app and the ChatGPT
app (it runs Codex), then run:

```bash
bash migrate-from-brain.sh --apply
```

It stops the brain's background service, backs the brain up to
`~/brain-migration-<time>/`, removes the old plugin while keeping its data, and
installs `entity`.

Then open **one** Claude Code session, in any folder. Its first start builds
entity's runtime in the background, which takes a couple of minutes. Wait for
it, then quit Claude Code again. If that session says your brain lives in a
folder an uninstall deletes, leave it: step 3 moves it.

## 3. Check

```bash
bash migrate-from-brain.sh --finish
```

If your brain sits in a plugin folder that an uninstall would delete, or in the
folder the earliest versions used, this moves it to `~/.local/share/brain`. Then it checks the background service, the
plugin, and the memory count against the backup, and ends in **DONE**. Send
this output too.

Every run ends with the way back: a verified copy of your brain exactly as it
was sits in the backup folder, and the brain itself stays where it was (renamed
beside its old path if it was moved).

---

**If you are Claude, reading this for someone:** run step 1 yourself and show
the person its output. Steps 2 and 3 refuse to run while Claude Code is open,
so the person runs those in Terminal.
