"""Session-local setup interactions, independent of the brain daemon.

The stdin owner routes responses here; waiting for consent never blocks that
reader. Matching consent can save tool policy; hook trust stays Codex's, and
setup says where in Codex to grant it.
"""
import json
import re
import sys
import threading
import uuid

# codex_onboarding.py owns host discovery, status, and the consented tool-policy write.
import codex_onboarding as codex

SETUP_TOOL = {
    'name': 'setup',
    'title': 'Set up Entity memory and permissions',
    'description': ('Check Entity setup with action=status. action=review asks once to '
                    'allow Entity tools without individual prompts, and says where in '
                    'Codex to approve automatic-memory hooks. Tool approval includes '
                    'reading, changing and deleting memories and messaging other Entity '
                    'sessions. Hook trust is decided in Codex. Check status afterwards; '
                    'saved policy may need a connection reload.'),
    'inputSchema': {'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': ['status', 'review'], 'default': 'status'}},
        },
    'annotations': {'destructiveHint': False, 'openWorldHint': False},
}

IDENTITY_NOTICE = (
    'Entity could not verify session identity for this call. Automatic memory '
    'capture may be unavailable: hooks may need approval, be disabled, or be '
    'failing. In Codex call setup(action="status") to diagnose. Tell the user '
    'what remains unavailable and offer to finish Entity setup, unless you '
    'already told the user the same thing in this chat. '
    'Do not reopen a declined confirmation unless the user asks again. Do not claim '
    'automatic memory is active from an MCP connection alone.'
)

# What stops while each Entity hook awaits Codex approval, keyed by the script
# its command runs; the PostToolUse trace says which tools by its matcher.
# tests/test_codex_onboarding.py holds every handler in hooks.codex.json to an
# entry here.
HOOK_EFFECTS = {
    'boot-brain.sh': "memory isn't loaded when a chat starts",
    'pre-response-recall.sh': "memories aren't recalled for your messages",
    'pre-edit-suggest.sh': "related memories aren't suggested before edits",
    'stamp-caller-session.sh': "Entity's tool calls aren't linked to this chat",
    'post-response-track.sh': "replies aren't recorded",
    'session-end.sh': "the end of a chat isn't recorded",
}
TRACED_TOOLS = {'Bash': 'shell commands', 'Edit|Write|Agent': 'file edits',
                'mcp__.*': 'tool calls'}

# Where the user grants hook trust: Codex's own review. Settings > Hooks was
# seen on screen (2026-10-11); the composer's icon-only Review hooks button,
# marked by a yellow dot while hooks wait, is read from ChatGPT.app 26.928
# (brain node b72a3bd2); the CLI has /hooks. No deep link reaches either, so
# setup says where instead of opening anything.
HOOK_WHERE = ('in the Codex app, open Settings → Hooks or click the icon with a yellow dot '
              'beside the message box (Review hooks), and trust the Entity hooks; in the '
              'Codex CLI, type /hooks.')
ENABLE_WHERE = 'turn them on under Settings → Hooks in the Codex app, or with /hooks in the Codex CLI.'
HOOK_REVIEW = 'To approve: ' + HOOK_WHERE
DISABLED_NOTE = 'Some Entity hooks are disabled; ' + ENABLE_WHERE

# The setup line reaches the user through the model: the prompt hook carries
# it on the first prompt of every chat while hooks await approval (brain node
# 941811c5). When the prompt hook is itself awaiting approval it cannot run,
# and the MCP instructions carry the line instead. Every unprompted carrier
# (these two, IDENTITY_NOTICE, the instructions) asks the model not to repeat
# what it already told the user in the chat; a new finding is still said, and
# the answer to a status the user asked for carries no such clause. That is
# guidance to the model, not a guarantee.
PROMPT_NOTICE = ('[brain] Entity setup is incomplete. Unless you already told the user the '
                 'same thing in this chat, tell them in one line in your reply: "%s"')
CONNECT_NOTICE = ('Entity setup is incomplete. In your first reply in this chat, unless you '
                  'already told the user the same thing, tell them in one line: "%s"')
# Per app-server call. The whole check (three calls) measured 193 ms median,
# 327 ms cold on a real Codex home (brain node b72a3bd2). Connect is tight: the proxy answers initialize after
# it, and Codex waits 1 s for optional servers. The prompt hook runs the check
# beside its daemon call, which hides it.
CONNECT_CHECK_TIMEOUT_S = 0.5
PROMPT_CHECK_TIMEOUT_S = 2.0


def hook_effect(hook):
    script = re.search(r'([\w.-]+)"?\s*$', hook.get('command') or '')
    script = script.group(1) if script else ''
    if script == 'post_tool_trace.py':
        return "%s aren't recorded" % TRACED_TOOLS.get(hook.get('matcher'), 'tool calls')
    return HOOK_EFFECTS.get(script, 'part of automatic memory is off')


def approval_line(status, then='Say “Finish Entity setup” to fix it.'):
    """The line a user reads while enabled Entity hooks await approval, or ''.
    `then` says what to do: the notices send the user to setup, setup's own
    answer says where in Codex (HOOK_REVIEW)."""
    waiting = codex.awaiting_review(status['hooks'])
    if not waiting:
        return ''
    effects = list(dict.fromkeys(map(hook_effect, waiting)))
    effect = ("automatic memory is off" if len(waiting) == len(status['hooks']) else
              ' and '.join(effects) if len(effects) <= 2 else 'parts of automatic memory are off')
    count = 'One Entity hook needs' if len(waiting) == 1 else '%d Entity hooks need' % len(waiting)
    return '%s approval in Codex, so %s. %s' % (count, effect, then)


def after_form_note(status):
    """Where hooks are approved or turned on, for any answer that follows the
    tool form. The form says where too, so the user may act while it is open:
    this claims no current state, which a snapshot taken before the form would."""
    if codex.awaiting_review(status['hooks']):
        return 'If Entity hooks still need approval: ' + HOOK_WHERE
    if status['state'] == 'disabled':
        return 'If any Entity hooks are still disabled, ' + ENABLE_WHERE
    return ''


def prompt_notice(plugin_root):
    """The prompt hook's context for the first prompt of a chat, or ''.
    Asks as setup(action='status') does, so the line and the status agree.
    Raises ReviewError when Codex cannot be asked."""
    line = approval_line(codex.setup_status(plugin_root=plugin_root, timeout=PROMPT_CHECK_TIMEOUT_S))
    return PROMPT_NOTICE % line if line else ''


class SetupSession:
    tools = [SETUP_TOOL]
    instructions = (
        "If automatic memory context is absent or a tool reports missing session "
        "identity, call setup(action='status'). Tell the user what remains "
        "unavailable and offer to finish Entity setup, unless you already told "
        "the user the same thing in this chat. Only call "
        "setup(action='review') while the user is asking to complete setup. "
        "When it shows a confirmation, explain that it asks for Entity-wide tool "
        "permission; hooks are approved in Codex itself. Report these permissions separately: missing "
        "Entity-wide tool approval alone does not mean automatic memory is off. "
        "After a declined, dismissed or unchecked confirmation, explain that "
        "permissions were unchanged and the user can say 'Finish Entity setup' "
        "anytime. Do not reopen it unless the user asks again. After expiry, "
        "explain that nothing changed and offer a fresh attempt; do not treat "
        "expiry as consent or repeatedly reopen forms. Check status afterwards; "
        "a saved tool policy does not prove the current "
        "connection reloaded it or that automatic memory works."
    )
    notice = IDENTITY_NOTICE
    # Read before the shared instructions; set at initialize.
    preface = ''

    def handles(self, name):
        return name == 'setup'

    def __init__(self, send, plugin_root, timeout=120, log=None):
        self.send, self.plugin_root, self.timeout, self.log = send, plugin_root, timeout, log
        self.form_supported = False
        self.current = None
        self.lock = threading.RLock()

    def initialize(self, params):
        self.close()
        # The proxy has already filtered capabilities by the negotiated version.
        capability = params.get('capabilities', {}).get('elicitation')
        self.form_supported = (isinstance(capability, dict) and
                               (not capability or 'form' in capability))
        self.preface = self._connect_notice()

    def _connect_notice(self):
        """The setup line for the instructions, only while the prompt hook that
        would carry it is itself awaiting approval."""
        try:
            status = codex.setup_status(plugin_root=self.plugin_root, timeout=CONNECT_CHECK_TIMEOUT_S)
        except Exception as error:
            if self.log:
                self.log('mcp_setup', 'connect check failed: %s' % error, 'host setup', level='warning')
            return ''
        if not any(h['eventName'] == 'userPromptSubmit' for h in codex.awaiting_review(status['hooks'])):
            return ''
        return CONNECT_NOTICE % approval_line(status)

    def _reply(self, request_id, payload, error=False):
        response = {'content': [{'type': 'text', 'text': json.dumps(payload)}]}
        if error:
            response['isError'] = True
            message = payload.get('message', payload.get('state', 'setup failed'))
            sys.stderr.write('[entity-setup] %s\n' % message)
            if self.log:
                self.log('mcp_setup', message, 'host setup', level='warning')
        self.send({'jsonrpc': '2.0', 'id': request_id, 'result': response})

    def result(self, op, payload, error=False):
        with self.lock:
            if self.current is not op:
                return
            self.close()
            self._reply(op['id'], payload, error)

    def _dispatch(self, job):
        worker = threading.Thread(target=job, daemon=True)
        worker.start()
        return worker

    def start(self, request_id, arguments, identity_verified=False):
        with self.lock:
            if self.current is not None:
                self._reply(request_id, {'state': 'busy', 'message': 'Entity setup is already checking or waiting for confirmation.'})
                return
            op = self.current = {'id': request_id, 'eid': None, 'timer': None}
        def inspect():
            try:
                self._start(op, arguments, identity_verified)
            except Exception as error:
                self.result(op, {'state': 'unknown', 'message': str(error)}, True)
        self._dispatch(inspect)

    def _start(self, op, arguments, identity_verified=False):
        action = arguments.get('action', 'status')
        if action not in ('status', 'review'):
            self.result(op, {'state': 'error', 'message': 'Use action=status or action=review.'}, True)
            return
        status = codex.setup_status(plugin_root=self.plugin_root)
        status['caller_identity_verified'] = identity_verified
        policy = status['tool_approval']
        needs_tools = policy is not None and policy['mode'] != 'approve'
        # Hook trust is granted in Codex; setup's part is saying where.
        hooks_note = (approval_line(status, HOOK_REVIEW) or
                      (DISABLED_NOTE if status['state'] == 'disabled' else ''))
        if action == 'status' or not policy or not policy['server_enabled'] or not needs_tools:
            status['message'] = {
                'not_found': 'Codex did not find this Entity installation’s hooks. Check that the plugin is installed and enabled.',
                'definitions_trusted': 'Codex reports these hook definitions trusted and enabled. Verify automatic recall and capture in the app before declaring memory ready.',
                'disabled': hooks_note,
                'review_required': hooks_note,
            }[status['state']]
            if needs_tools:
                status['message'] += ' Entity-wide tool approval is not saved; existing tool approval settings still apply. This does not disable approved memory hooks. Say “Finish Entity setup” to continue.'
            if policy and not policy['server_enabled']:
                status['message'] += ' Entity tools are disabled in Codex; enable the server there first.'
            self.result(op, status)
            return
        if not self.form_supported:
            self.result(op, {'state': 'confirmation_unavailable', 'trust_granted': False,
                        'message': ' '.join(filter(None, (
                            'This connection cannot show the setup confirmation, so no permissions '
                            'were changed; Entity tools keep their current approval settings.', hooks_note)))})
            return
        form_note = after_form_note(status)
        eid = 'entity-setup-' + uuid.uuid4().hex
        timer = threading.Timer(self.timeout, self._expire, args=(op,))
        timer.daemon = True
        request = {'jsonrpc': '2.0', 'id': eid, 'method': 'elicitation/create', 'params': {
            'mode': 'form',
            'message': ('Allow Entity to use its memory tools without asking you each time.\n\n'
                        'This includes reading, changing and deleting saved memories, and '
                        'messaging your other Entity sessions. '
                        'This applies to current and future tools from Entity’s brain server; '
                        'your individual tool restrictions stay in place.\n\n'
                        'Automatic-memory hooks are approved separately, in Codex itself.' +
                        ('\n\n' + form_note if form_note else '')),
            'requestedSchema': {'type': 'object', 'properties': {
                'enable_entity': {'type': 'boolean', 'title': 'Allow Entity tools and continue setup', 'default': False}},
                'required': ['enable_entity']}}}
        with self.lock:
            if self.current is not op:
                return
            op.update(eid=eid, timer=timer, plugin_id=policy['plugin_id'], form_note=form_note)
            self.send(request)
            timer.start()

    def _expire(self, op):
        with self.lock:
            if self.current is op and op['eid']:
                self.result(op, {'state': 'confirmation_expired', 'trust_granted': False,
                                 'message': ' '.join(filter(None, (
                                     'The confirmation expired. No permissions were changed.', op['form_note'],
                                     'Say “Finish Entity setup” to try the tool approval again.')))})

    def receive(self, message):
        with self.lock:
            op = self.current
            if op is None or not op['eid'] or message.get('id') != op['eid']:
                return False
            op['eid'] = None  # Consume before dispatch: replay and expiry cannot save again.
            op['timer'].cancel()
        reply = message.get('result') or {}
        content = reply.get('content') or {} if isinstance(reply, dict) else {}
        accepted = (isinstance(reply, dict) and reply.get('action') == 'accept' and
                    isinstance(content, dict) and content.get('enable_entity') is True and
                    'error' not in message)
        if not accepted:
            action = reply.get('action') if isinstance(reply, dict) else None
            reason = ('invalid_response' if 'error' in message else
                      'declined' if action == 'decline' else
                      'dismissed' if action == 'cancel' else
                      'unchecked' if action == 'accept' and isinstance(content, dict)
                      and content.get('enable_entity') is False else 'invalid_response')
            # Hooks never needed this consent: where to approve them comes either way.
            self.result(op, {'state': 'review_cancelled', 'reason': reason, 'trust_granted': False,
                             'message': ' '.join(filter(None, (
                                 'No permissions were changed: Entity tools and approved hooks keep their '
                                 'current settings.', op['form_note'],
                                 'Say “Finish Entity setup” anytime to allow Entity tools.')))})
            return True
        def save_tools():
            with self.lock:
                if self.current is not op:
                    return
            result = {'state': 'setup_updated', 'hook_trust_granted': False,
                      'tool_permission': None, 'runtime_verified': False}
            try:
                result['tool_permission'] = codex.approve_tools(
                    plugin_root=self.plugin_root, plugin_id=op['plugin_id'])
                result['message'] = ' '.join(filter(None, (op['form_note'],
                    'Check setup again afterwards; an existing Codex connection may need to reload saved tool permissions.')))
                self.result(op, result)
            except Exception as error:
                result.update(state='setup_incomplete', message=str(error))
                self.result(op, result, True)
        self._dispatch(save_tools)
        return True

    def cancel(self, request_id):
        with self.lock:
            if self.current is not None and self.current['id'] == request_id:
                self.close()

    def close(self):
        with self.lock:
            if self.current is not None and self.current['timer']:
                self.current['timer'].cancel()
            self.current = None


def create_extension(send, plugin_root, log=None):
    return SetupSession(send, plugin_root, log=log)
