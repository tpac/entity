"""Session-local setup interactions, independent of the brain daemon.

The stdin owner routes responses here; waiting for consent never blocks that
reader. Matching consent can save tool policy and open native hook review.
"""
import json
import sys
import threading
import uuid

# codex_onboarding.py owns host discovery, status, and native review launch.
import codex_onboarding as codex

SETUP_TOOL = {
    'name': 'setup',
    'title': 'Set up Entity memory and permissions',
    'description': ('Check Entity setup with action=status. action=review presents one '
                    'clear confirmation to allow Entity tools without individual prompts '
                    'and, when needed, open Codex hook review for automatic memory. '
                    'Tool approval includes reading, changing and deleting memories and '
                    'messaging other Entity sessions. Hook trust is decided in Codex. '
                    'Check status afterwards; saved policy may need a connection reload.'),
    'inputSchema': {'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': ['status', 'review'], 'default': 'status'}},
        },
    'annotations': {'destructiveHint': False, 'openWorldHint': False},
}

IDENTITY_NOTICE = (
    'Entity could not verify session identity for this call. Automatic memory '
    'capture may be unavailable: hooks may need approval, be disabled, or be '
    'failing. Tell the user; in Codex call setup(action="status") to diagnose '
    'and explain what remains unavailable. Offer to finish Entity setup; '
    'do not reopen a declined confirmation unless the user asks again. Do not claim '
    'automatic memory is active from an MCP connection alone.'
)


class SetupSession:
    tools = [SETUP_TOOL]
    instructions = (
        "If automatic memory context is absent or a tool reports missing session "
        "identity, tell the user and call setup(action='status'). Offer "
        "to finish Entity setup when setup is incomplete. Only call "
        "setup(action='review') while the user is asking to complete setup. "
        "Explain it requests Entity-wide tool permission and opens Codex's "
        "separate hook review. Report these permissions separately: missing "
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
        needs_hooks = status['state'] == 'review_required'
        if action == 'status' or not policy or not policy['server_enabled'] or not (needs_tools or needs_hooks):
            status['message'] = {
                'not_found': 'Codex did not find this Entity installation’s hooks. Check that the plugin is installed and enabled.',
                'definitions_trusted': 'Codex reports these hook definitions trusted and enabled. Verify automatic recall and capture in the app before declaring memory ready.',
                'disabled': 'Some Entity hooks are disabled. Open Codex CLI and use /hooks to review their settings. The automatic startup review only covers new or changed enabled hooks.',
                'review_required': 'Some Entity automatic-memory hooks still need Codex approval. Those hooks will not run until approved. Say “Finish Entity setup” to continue.',
            }[status['state']]
            if needs_tools:
                status['message'] += ' Entity-wide tool approval is not saved; existing tool approval settings still apply. This does not disable approved memory hooks. Say “Finish Entity setup” to continue.'
            if policy and not policy['server_enabled']:
                status['message'] += ' Entity tools are disabled in Codex; enable the server there first.'
            self.result(op, status)
            return
        if not self.form_supported:
            self.result(op, {'state': 'confirmation_unavailable', 'trust_granted': False,
                        'message': 'This connection cannot show the setup confirmation. Review Entity tool permissions in Codex and hook trust under /hooks; this tool changed no permissions.'})
            return
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
                        'If automatic-memory hooks need approval, Codex will open their review '
                        'in Terminal. You make the hook-trust decision there. No commands need '
                        'to be typed. Return here afterwards to check setup.'),
            'requestedSchema': {'type': 'object', 'properties': {
                'enable_entity': {'type': 'boolean', 'title': 'Allow Entity tools and continue setup', 'default': False}},
                'required': ['enable_entity']}}}
        with self.lock:
            if self.current is not op:
                return
            op.update(eid=eid, timer=timer, plugin_id=policy['plugin_id'],
                      needs_tools=needs_tools, needs_hooks=needs_hooks)
            self.send(request)
            timer.start()

    def _expire(self, op):
        with self.lock:
            if self.current is op and op['eid']:
                self.result(op, {'state': 'confirmation_expired', 'trust_granted': False,
                                 'message': 'The confirmation expired. No permissions were changed. Say “Finish Entity setup” to try again.'})

    def receive(self, message):
        with self.lock:
            op = self.current
            if op is None or not op['eid'] or message.get('id') != op['eid']:
                return False
            op['eid'] = None  # Consume before dispatch: replay and expiry cannot launch again.
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
            self.result(op, {'state': 'review_cancelled', 'reason': reason, 'trust_granted': False,
                             'message': 'No permissions were changed. Existing tool permissions and approved hooks were left as they were. If automatic-memory hooks still need approval, those hooks remain unavailable. Say “Finish Entity setup” anytime to continue.'})
            return True
        def launch():
            with self.lock:
                if self.current is not op:
                    return
            result = {'state': 'setup_updated', 'hook_trust_granted': False,
                      'tool_permission': None, 'hook_review': None, 'runtime_verified': False}
            try:
                if op['needs_tools']:
                    result['tool_permission'] = codex.approve_tools(
                        plugin_root=self.plugin_root, plugin_id=op['plugin_id'])
                if op['needs_hooks']:
                    with self.lock:
                        if self.current is not op:
                            return
                    result['hook_review'] = codex.open_review()
                result['message'] = 'Check setup again after hook review. An existing Codex connection may need to reload saved tool permissions.'
                self.result(op, result)
            except Exception as error:
                result.update(state='setup_incomplete', message=str(error))
                self.result(op, result, True)
        self._dispatch(launch)
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
