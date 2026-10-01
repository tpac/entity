#!/bin/bash
# brain — PreToolUse(Edit|Write) hook: surfaces brain rules before edits.
# Output: hookSpecificOutput.additionalContext, or nothing (hook_common.emit_hook_output).
source "$(dirname "$0")/resolve-brain-db.sh"
# No brain yet (fresh install, adoption pending): nothing to say. Exit 0 with no
# stdout — both hosts read silence as "nothing to report"; a printed decision
# would be rejected by Codex as invalid hook output.
[ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ] && exit 0
export HOOK_INPUT=$(cat)
exec "${BRAIN_PYTHON_HOOK:-python3}" "$(dirname "$0")/pre_edit_suggest.py"
