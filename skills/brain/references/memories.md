# Seeing, correcting and forgetting memories

## Seeing

- The dashboard, `http://localhost:47303` (`DASHBOARD_PORT` moves it), shows the
  graph, session records and live encoding. It reads the brain; its only writes
  are a message sent to a session from it and an API key pasted into its setup
  page.
- Your memory tools (`recall`, `get_nodes`, `recall_episodes`) show the same
  memories in the conversation.

## Correcting

Your operator tells you what is wrong; you revise the memory at once, with the
reason. When a new understanding replaces an old one, write the new memory and
link it to the old one with a correction edge — the correction then travels with
the memory it corrects. A correction is cheapest while the memory is young:
later memories get built on top of a wrong one.

Confidence on a memory records how sure you were when it was written. It moves
only when a revision sets it — yours or your background encoder's; nothing
recalibrates it on its own.

## Forgetting

- **Archiving** takes a memory out of recall. It is a soft delete: the content is
  kept, and the memory can still be read by id.
- There is no user-facing hard delete of a single memory.
- Archiving a memory does not remove its text from the verbatim session record,
  the diagnostic captures (`payloads/`, and `surface_captures/` where enabled
  or left by an earlier version) or the backups.
- Removing everything means deleting the data directory (see
  [install-and-data.md](install-and-data.md)). Outside it, check for a spare
  copy a relocation leaves beside the old location, captures moved elsewhere by
  `BRAIN_SURFACE_CAPTURE_DIR`, and the host's own session transcripts.
