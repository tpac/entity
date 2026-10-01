#!/bin/bash
# brain — PreToolUse(mcp__brain__*) hook: signs the calling session into the
# brain tool call, for hosts whose MCP proxy gets no session identity (Codex).
# Output: hookSpecificOutput permissionDecision allow + updatedInput, or nothing
# (hook_common.emit_updated_input).
source "$(dirname "$0")/resolve-brain-db.sh"
# No brain yet (fresh install, adoption pending): nothing to attribute to. Exit 0
# with no stdout — both hosts read silence as "nothing to report".
[ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ] && exit 0
export HOOK_INPUT=$(cat)
exec "${BRAIN_PYTHON_HOOK:-python3}" "$(dirname "$0")/stamp_caller_session.py"
