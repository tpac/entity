"""Hook review and consent contract. No daemon, real hook grants, or brain DB."""
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hooks' / 'adapters'))
import codex_onboarding as codex
from codex_setup import SetupSession


def setup_status(hooks=None):
    status = codex.summarize_hooks({'data': [{'hooks': hooks if hooks is not None else [hook()]}]})
    status['tool_approval'] = {'plugin_id': 'entity@local', 'mode': 'auto', 'server_enabled': True, 'restricted_tools': []}
    return status


def hook(**changes):
    return {'key': 'entity@local:hooks/hooks.codex.json:stop:0:0',
            'pluginId': 'entity@local', 'source': 'plugin',
            'sourcePath': '/installed/entity/hooks/hooks.codex.json',
            'trustStatus': 'untrusted', 'enabled': True, 'currentHash': 'sha256:a',
            'eventName': 'stop', **changes}


class TestStatus(unittest.TestCase):
    def summarize(self, hooks, **changes):
        return codex.summarize_hooks({'data': [{'hooks': hooks, **changes}]}, '/installed/entity')

    def test_foreign_plugin_and_other_install_are_excluded(self):
        result = self.summarize([hook(), hook(pluginId='other@local'),
                                 hook(sourcePath='/other/entity/hooks.json'), hook(source='user')])
        self.assertEqual(result['review_count'], 1)
        self.assertEqual(len(result['hooks']), 1)
        self.assertFalse(result['trust_complete'])

    def test_trust_does_not_claim_runtime_success(self):
        result = self.summarize([hook(trustStatus='trusted')])
        self.assertTrue(result['trust_complete'])
        self.assertFalse(result['runtime_verified'])

    def test_modified_disabled_missing_and_errors_cannot_be_ready(self):
        for hooks, state in [([hook(trustStatus='modified')], 'review_required'),
                             ([hook(trustStatus='trusted', enabled=False)], 'disabled'),
                             ([hook(enabled=False)], 'disabled'),
                             ([], 'not_found')]:
            result = self.summarize(hooks)
            self.assertEqual(result['state'], state)
            self.assertFalse(result['trust_complete'])
        with self.assertRaises(codex.ReviewError):
            self.summarize([hook(trustStatus='trusted')], errors=[{'message': 'bad manifest'}])


class TestConsent(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.session = SetupSession(self.sent.append, '/installed/entity')
        # Deterministic consent-state tests; worker behavior is exercised below
        # using the real dispatch method and explicit synchronization events.
        self.session._dispatch = lambda job: job()
        self.addCleanup(self.session.close)
        self.status = patch.object(codex, 'setup_status', return_value=setup_status())
        self.status_mock = self.status.start()
        self.addCleanup(self.status.stop)
        self.initialize({'protocolVersion': '2025-11-25', 'capabilities': {'elicitation': {'form': {}}}})
        self.approval = patch.object(codex, 'approve_tools', return_value={'saved': True, 'mode': 'approve'})
        self.approve = self.approval.start()
        self.addCleanup(self.approval.stop)

    def initialize(self, params):
        from servers import brain_mcp as proxy
        proxy.handle_initialize(0, params, self.session)

    def request(self, rid=1):
        self.session.start(rid, {'action': 'review'})
        return self.sent[-1]['id']

    def reply(self, eid, action='accept', value=True):
        self.session.receive({'id': eid, 'result': {'action': action, 'content': {'enable_entity': value}}})

    def payload(self):
        return json.loads(self.sent[-1]['result']['content'][0]['text'])

    def test_accept_saves_tools_once_and_says_where_to_approve_hooks(self):
        import codex_setup
        form_note = codex_setup.after_form_note(self.status_mock.return_value)
        eid = self.request()
        self.assertEqual(self.sent[-1]['method'], 'elicitation/create')
        self.assertIn(form_note, self.sent[-1]['params']['message'], 'the form says where hooks go')
        self.reply(eid)
        self.reply(eid)
        self.approve.assert_called_once_with(plugin_root='/installed/entity', plugin_id='entity@local')
        self.assertEqual(self.sent[-1]['id'], 1)
        self.assertFalse(self.payload()['hook_trust_granted'])
        # The user may have approved hooks while the form was open: no snapshot count.
        self.assertIn(form_note, self.payload()['message'])
        self.assertNotIn('need approval in Codex', self.payload()['message'])

    def test_decline_cancel_false_and_malformed_accept_save_nothing_but_say_where_hooks_go(self):
        import codex_setup
        for action, value, reason in [('decline', True, 'declined'), ('cancel', True, 'dismissed'),
                                      ('accept', False, 'unchecked'), ('accept', 'true', 'invalid_response')]:
            self.reply(self.request(), action, value)
            self.assertEqual(self.payload()['state'], 'review_cancelled')
            self.assertEqual(self.payload()['reason'], reason)
            # Hooks never needed this consent: refusing tools must not hide them.
            self.assertIn(codex_setup.after_form_note(self.status_mock.return_value), self.payload()['message'])
        self.approve.assert_not_called()
        self.status_mock.return_value = setup_status([hook(trustStatus='trusted')])
        self.reply(self.request(), 'decline')
        self.assertNotIn(codex_setup.HOOK_WHERE, self.payload()['message'], 'no hook advice when none wait')

    def test_declined_form_cannot_grant_on_replay_but_fresh_request_can(self):
        declined = self.request()
        self.reply(declined, action='decline')
        self.assertIsNone(self.session.current)
        self.assertEqual(len(self.sent), 2, 'A declined form must not reopen itself')
        self.reply(declined)
        self.approve.assert_not_called()
        fresh = self.request(2)
        self.assertNotEqual(fresh, declined)
        self.reply(fresh)
        self.approve.assert_called_once()

    def test_cancellation_expiry_and_unknown_id_save_nothing(self):
        eid = self.request()
        self.reply('wrong-id')
        self.session.cancel(1)
        self.reply(eid)
        eid = self.request(2)
        self.session._expire(self.session.current)
        self.assertEqual(self.payload()['state'], 'confirmation_expired')
        import codex_setup
        self.assertIn(codex_setup.after_form_note(self.status_mock.return_value), self.payload()['message'])
        self.reply(eid)
        self.approve.assert_not_called()

    def test_no_form_capability_or_old_protocol_does_not_prompt(self):
        for caps, version in [({}, '2025-11-25'), ({'elicitation': {'url': {}}}, '2025-11-25'),
                              ({'elicitation': {}}, '2024-11-05')]:
            self.initialize({'protocolVersion': version, 'capabilities': caps})
            self.session.start(1, {'action': 'review'})
            self.assertEqual(self.payload()['state'], 'confirmation_unavailable')
        import codex_setup
        self.assertIn(codex_setup.HOOK_REVIEW, self.payload()['message'])
        self.approve.assert_not_called()

    def test_status_never_prompts_and_preserves_identity_evidence(self):
        self.session.start(9, {'action': 'status'}, identity_verified=True)
        self.assertTrue(self.payload()['caller_identity_verified'])
        self.assertEqual(self.payload()['state'], 'review_required')
        self.approve.assert_not_called()

    def test_status_names_waiting_hooks_and_where_to_approve_them(self):
        import codex_setup
        status = self.status_mock.return_value
        self.session.start(9, {'action': 'status'})
        message = self.payload()['message']
        self.assertTrue(message.startswith(codex_setup.approval_line(status, codex_setup.HOOK_REVIEW)), message)
        self.assertEqual(message.count('Finish Entity setup'), 1, 'one call to action: the tool approval')

    def test_second_request_does_not_replace_first_confirmation(self):
        eid = self.request()
        self.request(2)
        self.assertEqual(self.payload()['state'], 'busy')
        self.reply(eid)
        self.assertEqual(self.sent[-1]['id'], 1)
        self.approve.assert_called_once()

    def test_inspection_does_not_block_reader_and_cancel_suppresses_its_result(self):
        session = SetupSession(self.sent.append, '/installed/entity')
        self.addCleanup(session.close)
        entered, release = threading.Event(), threading.Event()
        workers = []
        dispatch = session._dispatch
        session._dispatch = lambda job: workers.append(dispatch(job))
        status = setup_status()
        def slow_status(**kwargs):
            entered.set()
            release.wait(3)
            return status
        with patch.object(codex, 'setup_status', side_effect=slow_status):
            try:
                session.start(1, {'action': 'status'})
                self.assertTrue(entered.wait(1), 'inspection worker did not start')
                from servers import brain_mcp as proxy
                self.assertEqual(proxy.handle_ping(2)['result'], {})
                session.start(2, {'action': 'review'})
                self.assertEqual(self.sent[-1]['id'], 2)
                self.assertEqual(self.payload()['state'], 'busy')
                session.cancel(1)
            finally:
                release.set()
                for worker in workers:
                    worker.join(3)
        self.assertEqual(len(self.sent), 1, 'cancelled inspection must not send a stale result')

    def test_permission_write_runs_on_worker_and_reader_can_cancel_its_result(self):
        eid = self.request()
        entered, release = threading.Event(), threading.Event()
        workers = []
        dispatch = SetupSession._dispatch.__get__(self.session)
        self.session._dispatch = lambda job: workers.append(dispatch(job))
        def slow_write(**kwargs):
            entered.set()
            release.wait(3)
            return {'saved': True, 'mode': 'approve'}
        self.approve.side_effect = slow_write
        try:
            self.reply(eid)
            self.assertTrue(entered.wait(1), 'permission worker did not start')
            self.session.cancel(1)
        finally:
            release.set()
            for worker in workers:
                worker.join(3)
        self.assertEqual(len(self.sent), 1, 'a cancelled write must not send a stale tool result')

    def test_reinitialize_invalidates_old_inspection_with_reused_request_id(self):
        jobs = []
        self.session._dispatch = jobs.append
        self.session.start(1, {'action': 'status'})
        self.initialize({})
        self.session.start(1, {'action': 'status'})
        jobs[0]()
        self.assertEqual(self.sent, [])
        jobs[1]()
        self.assertEqual(len(self.sent), 1)

    def test_setup_failure_uses_injected_proxy_logger(self):
        with patch.object(codex, 'setup_status', side_effect=codex.ReviewError('status unavailable')), \
             patch.object(self.session, 'log') as log:
            self.session.start(1, {'action': 'status'})
            log.assert_called_once_with('mcp_setup', 'status unavailable', 'host setup', level='warning')

    def test_tool_permission_is_offered_even_when_hooks_are_trusted(self):
        import codex_setup
        self.status_mock.return_value = setup_status([hook(trustStatus='trusted')])
        self.reply(self.request())
        self.approve.assert_called_once()
        self.assertTrue(self.payload()['tool_permission']['saved'])
        self.assertNotIn(codex_setup.HOOK_WHERE, self.payload()['message'])

    def test_waiting_hooks_alone_get_where_to_approve_without_a_form(self):
        import codex_setup
        status = setup_status()
        status['tool_approval']['mode'] = 'approve'
        self.status_mock.return_value = status
        self.session.start(1, {'action': 'review'})
        self.assertNotIn('method', self.sent[-1], 'nothing to consent to: no form')
        self.assertEqual(self.payload()['message'], codex_setup.approval_line(status, codex_setup.HOOK_REVIEW))
        self.approve.assert_not_called()

    def test_cancel_before_acceptance_worker_starts_saves_no_permission(self):
        eid = self.request()
        jobs = []
        self.session._dispatch = jobs.append
        self.reply(eid)
        self.session.cancel(1)
        jobs[0]()
        self.approve.assert_not_called()

    def test_model_arguments_cannot_replace_user_confirmation(self):
        self.session.start(1, {'action': 'review', 'enable_entity': True})
        self.assertEqual(self.sent[-1]['method'], 'elicitation/create')
        self.approve.assert_not_called()

    def test_failed_permission_write_is_not_success(self):
        self.approve.side_effect = codex.ReviewError('configuration changed; check again')
        self.reply(self.request())
        self.assertEqual(self.payload()['state'], 'setup_incomplete')
        self.assertIsNone(self.payload()['tool_permission'])
        self.assertTrue(self.sent[-1]['result']['isError'])

    def test_disabled_hooks_point_to_codex_settings_in_status_and_after_the_form(self):
        import codex_setup
        self.status_mock.return_value = setup_status([hook(trustStatus='trusted', enabled=False)])
        self.session.start(1, {'action': 'status'})
        self.assertTrue(self.payload()['message'].startswith(codex_setup.DISABLED_NOTE))
        self.reply(self.request(2))
        self.assertIn(codex_setup.ENABLE_WHERE, self.payload()['message'])

    def test_review_with_tools_disabled_explains_instead_of_asking(self):
        status = setup_status()
        status['tool_approval']['server_enabled'] = False
        self.status_mock.return_value = status
        self.session.start(1, {'action': 'review'})
        self.assertNotIn('method', self.sent[-1], 'consent to a disabled server cannot apply: no form')
        self.assertIn('enable the server there first', self.payload()['message'])
        self.approve.assert_not_called()

    def test_review_without_a_tool_policy_reports_the_install_state(self):
        import codex_setup
        status = setup_status()
        status['tool_approval'] = None
        self.status_mock.return_value = status
        self.session.start(1, {'action': 'review'})
        self.assertNotIn('method', self.sent[-1])
        self.assertEqual(self.payload()['state'], 'review_required')
        self.assertEqual(self.payload()['message'], codex_setup.approval_line(status, codex_setup.HOOK_REVIEW))
        self.approve.assert_not_called()


class TestProxy(unittest.TestCase):
    def setUp(self):
        self.extension = SetupSession(lambda message: None, '/installed/entity')
        self.addCleanup(self.extension.close)
        status = patch.object(codex, 'setup_status', return_value=setup_status([hook(trustStatus='trusted')]))
        status.start()
        self.addCleanup(status.stop)

    def test_notice_repeats_on_each_unattributed_call_even_with_success(self):
        from servers import brain_mcp as proxy
        def send(cmd, args, note):
            note('test identity gap')
            return {'ok': True, 'result': {'ok': True}}
        with patch.object(proxy, 'daemon_send', side_effect=send), patch.object(proxy, '_note_identity_gap'):
            for rid in [1, 2]:
                response = proxy.handle_tools_call(rid, {'name': 'ping'}, self.extension)
                self.assertIn('could not verify session identity', response['result']['content'][1]['text'])

    def test_setup_is_served_locally_without_daemon_dispatch(self):
        from servers import brain_mcp as proxy
        with patch.object(self.extension, 'start') as start, patch.object(proxy, 'daemon_send') as send:
            self.assertIsNone(proxy.handle_tools_call(1, {'name': 'setup', 'arguments': {'action': 'status'}}, self.extension))
            self.assertEqual(start.call_args.args[:2], (1, {'action': 'status'}))
            send.assert_not_called()

    def test_adapter_receives_no_reserved_identity_fields(self):
        from servers import brain_mcp as proxy
        with patch.object(proxy, '_stamp_caller_session', return_value={'action': 'status', proxy.CALLER_SESSION_KEY: 'verified'}), \
             patch.object(self.extension, 'start') as start:
            proxy.handle_tools_call(1, {'name': 'setup', 'arguments': {}}, self.extension)
        start.assert_called_once_with(1, {'action': 'status'}, identity_verified=True)

    def test_argument_shape_is_a_protocol_error(self):
        from servers import brain_mcp as proxy
        response = proxy.handle_tools_call(1, {'name': 'recall', 'arguments': []})
        self.assertEqual(response['error']['code'], -32602)

    def test_setup_is_advertised_and_protocol_is_negotiated(self):
        from servers import brain_mcp as proxy
        response = proxy.handle_initialize(1, {'protocolVersion': '2025-11-25', 'capabilities': {'elicitation': {'form': {}}}}, self.extension)
        self.assertEqual(response['result']['protocolVersion'], '2025-11-25')
        self.assertIn('setup', [t['name'] for t in proxy.handle_tools_list(2, self.extension)['result']['tools']])
        self.extension.close()

    def test_supported_protocols_and_fallback_share_one_contract(self):
        from servers import brain_mcp as proxy
        for version in proxy.SUPPORTED_PROTOCOL_VERSIONS:
            response = proxy.handle_initialize(1, {'protocolVersion': version})
            self.assertEqual(response['result']['protocolVersion'], version)
        self.assertEqual(proxy.PROTOCOL_VERSION, proxy.SUPPORTED_PROTOCOL_VERSIONS[-1],
                         'a version we do not speak gets our latest')
        for requested in ('unknown', None):
            response = proxy.handle_initialize(1, {'protocolVersion': requested})
            self.assertEqual(response['result']['protocolVersion'], proxy.PROTOCOL_VERSION)

    def test_server_version_is_this_copys_code_fingerprint(self):
        from servers import brain_mcp as proxy
        from servers.daemon_config import _CODE_FINGERPRINT
        info = proxy.handle_initialize(1, {'protocolVersion': '2025-11-25'})['result']['serverInfo']
        self.assertEqual(info['version'], _CODE_FINGERPRINT)

    def test_adapter_receives_negotiated_version_and_compatible_capabilities(self):
        from servers import brain_mcp as proxy
        capabilities = {'elicitation': {'form': {}}, 'roots': {}}
        with patch.object(self.extension, 'initialize') as initialize:
            proxy.handle_initialize(1, {'protocolVersion': '2024-11-05', 'capabilities': capabilities}, self.extension)
            initialize.assert_called_with({'protocolVersion': '2024-11-05', 'capabilities': {'roots': {}}})
            proxy.handle_initialize(1, {'protocolVersion': 'unknown', 'capabilities': capabilities}, self.extension)
            initialize.assert_called_with({'protocolVersion': proxy.PROTOCOL_VERSION, 'capabilities': capabilities})
        self.assertIn('elicitation', capabilities, 'negotiation must not mutate the client request')
        proxy.handle_initialize(1, {'protocolVersion': '2024-11-05', 'capabilities': capabilities}, self.extension)
        self.assertFalse(self.extension.form_supported)
        proxy.handle_initialize(1, {'protocolVersion': 'unknown', 'capabilities': capabilities}, self.extension)
        self.assertTrue(self.extension.form_supported)

    def test_shared_entrypoint_does_not_expose_adapter_settings(self):
        from servers import brain_mcp as proxy
        self.assertNotIn('setup', [t['name'] for t in proxy.handle_tools_list(1)['result']['tools']])
        response = proxy.handle_initialize(2)
        self.assertEqual(response['result']['instructions'], proxy.SERVER_INSTRUCTIONS)

    def test_legacy_client_reader_roundtrip_keeps_its_protocol_and_core_tools(self):
        from servers import brain_mcp as proxy
        messages = [
            {'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2024-11-05', 'capabilities': {}}},
            {'method': 'notifications/initialized'},
            {'id': 2, 'method': 'tools/list'},
        ]
        output = io.StringIO()
        with patch.object(proxy, 'ensure_daemon_running', return_value=True), \
             patch.object(proxy, '_health_monitor'), \
             patch('sys.stdin', io.StringIO('\n'.join(json.dumps(m) for m in messages))), \
             patch('sys.stdout', output):
            proxy.main()
        replies = {message['id']: message['result'] for message in
                   map(json.loads, output.getvalue().splitlines())}
        self.assertEqual(replies[1]['protocolVersion'], '2024-11-05')
        names = [tool['name'] for tool in replies[2]['tools']]
        self.assertIn('recall', names)
        self.assertNotIn('setup', names)

    def test_extension_load_is_confined_and_failure_keeps_core_tools(self):
        from servers import brain_mcp as proxy
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / 'hooks/adapters'
            scripts.mkdir(parents=True)
            outside = root / 'outside.py'
            outside.write_text('raise AssertionError("must never execute outside package")')
            (scripts / 'escape.py').symlink_to(outside)
            (scripts / 'broken.py').write_text('raise RuntimeError("broken adapter")')
            (scripts / 'conflict.py').write_text('from types import SimpleNamespace\ndef create_extension(*args, **kwargs):\n    return SimpleNamespace(tools=[{"name": "recall"}])\n')
            with patch.object(proxy, '_parent', str(root)), patch.object(proxy, '_log_proxy_error') as log:
                for name in ['../outside.py', str(outside), 'escape.py', 'missing.py', 'broken.py', 'conflict.py']:
                    self.assertIsNone(proxy.load_extension(name))
                self.assertEqual(log.call_count, 6)
                self.assertIn('recall', [t['name'] for t in proxy.handle_tools_list(1)['result']['tools']])

    def test_real_adapter_loads_through_generic_entrypoint(self):
        from servers import brain_mcp as proxy
        extension = proxy.load_extension('codex_setup.py')
        self.assertIsNotNone(extension)
        self.addCleanup(extension.close)
        self.assertEqual([tool['name'] for tool in extension.tools], ['setup'])

    def test_reader_handles_ping_and_cancellation_while_confirmation_is_pending(self):
        from servers import brain_mcp as proxy
        import codex_setup
        messages = [
            {'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-11-25', 'capabilities': {'elicitation': {'form': {}}}}},
            {'id': 2, 'method': 'tools/call', 'params': {'name': 'setup', 'arguments': {'action': 'review'}}},
            {'id': 3, 'method': 'ping'},
            {'method': 'notifications/cancelled', 'params': {'requestId': 2}},
            {'id': 'entity-setup-fixed', 'result': {'action': 'accept', 'content': {'enable_entity': True}}},
        ]
        output = io.StringIO()
        extension = SetupSession(proxy.send, '/installed/entity')
        extension._dispatch = lambda job: job()
        self.addCleanup(extension.close)
        status = setup_status()
        with patch.object(proxy, 'ensure_daemon_running', return_value=True), \
             patch.object(proxy, '_health_monitor'), \
             patch.object(codex, 'setup_status', return_value=status), \
             patch.object(codex, 'approve_tools') as write, \
             patch.object(codex_setup.uuid, 'uuid4') as uid, \
             patch('sys.stdin', io.StringIO('\n'.join(json.dumps(m) for m in messages))), \
             patch('sys.stdout', output):
            uid.return_value.hex = 'fixed'
            proxy.main(extension=extension)
        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertIn({'jsonrpc': '2.0', 'id': 3, 'result': {}}, replies)
        self.assertEqual(len([r for r in replies if r.get('method') == 'elicitation/create']), 1)
        self.assertIsNone(extension.current)
        write.assert_not_called()


ROOT = Path(__file__).resolve().parents[1]


def manifest_hooks(**trust):
    """hooks.codex.json as hooks/list reports it, untrusted unless `trust`
    names an event's status."""
    hooks = []
    for event, groups in json.loads((ROOT / 'hooks/hooks.codex.json').read_text())['hooks'].items():
        name = event[0].lower() + event[1:]
        for group in groups:
            for handler in group['hooks']:
                hooks.append(hook(eventName=name, command=handler['command'],
                                  matcher=group.get('matcher', ''), trustStatus=trust.get(name, 'untrusted')))
    return hooks


class TestApprovalLine(unittest.TestCase):
    """The words a user reads while Entity hooks await Codex approval."""

    def line(self, hooks):
        import codex_setup
        return codex_setup.approval_line(codex.summarize_hooks({'data': [{'hooks': hooks}]}))

    def trusted_except(self, *waiting):
        return [h if (h['eventName'], h['matcher']) in waiting else dict(h, trustStatus='trusted')
                for h in manifest_hooks()]

    def test_one_waiting_hook_names_what_stops(self):
        self.assertEqual(self.line(self.trusted_except(('preToolUse', 'mcp__brain__.*'))),
                         "One Entity hook needs approval in Codex, so Entity's tool calls aren't "
                         "linked to this chat. Say “Finish Entity setup” to fix it.")
        self.assertIn("so shell commands aren't recorded.",
                      self.line(self.trusted_except(('postToolUse', 'Bash'))))

    def test_two_effects_join_and_more_collapse(self):
        two = self.line(self.trusted_except(('postToolUse', 'Bash'), ('stop', '')))
        self.assertTrue(two.startswith('2 Entity hooks need approval in Codex, so '), two)
        self.assertIn(two.split(', so ')[1].split('. ')[0],
                      ("shell commands aren't recorded and replies aren't recorded",
                       "replies aren't recorded and shell commands aren't recorded"))
        three = self.line(self.trusted_except(('postToolUse', 'Bash'), ('stop', ''), ('sessionEnd', '')))
        self.assertIn('so parts of automatic memory are off.', three)

    def test_every_hook_waiting_means_memory_is_off(self):
        line = self.line(manifest_hooks())
        self.assertIn('Entity hooks need approval in Codex, so automatic memory is off.', line)
        self.assertTrue(line.endswith('Say “Finish Entity setup” to fix it.'))

    def test_trusted_managed_and_disabled_hooks_say_nothing(self):
        hooks = [dict(h, trustStatus='trusted') for h in manifest_hooks()]
        hooks[0]['trustStatus'] = 'managed'
        hooks[1].update(trustStatus='untrusted', enabled=False)
        self.assertEqual(self.line(hooks), '')

    def test_every_codex_manifest_handler_has_a_named_effect(self):
        import codex_setup
        for item in manifest_hooks():
            script = item['command'].rstrip('"').rsplit('/', 1)[-1]
            with self.subTest(script=script, matcher=item['matcher']):
                if script == 'post_tool_trace.py':
                    self.assertIn(item['matcher'], codex_setup.TRACED_TOOLS, 'name the traced tools')
                else:
                    self.assertIn(script, codex_setup.HOOK_EFFECTS, 'name what stops')

    def test_every_unprompted_carrier_asks_not_to_repeat(self):
        import codex_setup
        for name in ('PROMPT_NOTICE', 'CONNECT_NOTICE', 'IDENTITY_NOTICE'):
            self.assertIn('already told the user', getattr(codex_setup, name), name)
        self.assertIn('already told the user', SetupSession.instructions)

    def test_prompt_notice_wraps_the_line_and_asks_codex_once(self):
        import codex_setup
        waiting = setup_status(self.trusted_except(('stop', '')))
        with patch.object(codex, 'setup_status', return_value=waiting) as status:
            notice = codex_setup.prompt_notice('/installed/entity')
        status.assert_called_once_with(plugin_root='/installed/entity',
                                       timeout=codex_setup.PROMPT_CHECK_TIMEOUT_S)
        self.assertEqual(notice, codex_setup.PROMPT_NOTICE % codex_setup.approval_line(waiting))
        with patch.object(codex, 'setup_status', return_value=setup_status([hook(trustStatus='trusted')])):
            self.assertEqual(codex_setup.prompt_notice('/installed/entity'), '')


class TestConnectNotice(unittest.TestCase):
    """The MCP instructions carry the line only when the prompt hook cannot."""

    def instructions(self, status=None, error=None):
        from servers import brain_mcp as proxy
        logged = []
        session = SetupSession(lambda message: None, '/installed/entity', log=lambda *a, **k: logged.append(a))
        self.addCleanup(session.close)
        with patch.object(codex, 'setup_status', return_value=status, side_effect=error) as check:
            text = proxy.handle_initialize(1, {'protocolVersion': '2025-11-25'}, session)['result']['instructions']
        check.assert_called_once_with(plugin_root='/installed/entity', timeout=0.5)
        return text, logged

    def test_waiting_prompt_hook_puts_the_line_first(self):
        import codex_setup
        from servers import brain_mcp as proxy
        status = setup_status(manifest_hooks())
        text, _ = self.instructions(status)
        lead = codex_setup.CONNECT_NOTICE % codex_setup.approval_line(status)
        self.assertTrue(text.startswith(lead + ' ' + proxy.SERVER_INSTRUCTIONS), text[:200])
        self.assertTrue(text.endswith(SetupSession.instructions))

    def test_trusted_prompt_hook_leaves_the_line_to_the_hook(self):
        from servers import brain_mcp as proxy
        status = setup_status([dict(h, trustStatus='trusted') if h['eventName'] == 'userPromptSubmit' else h
                               for h in manifest_hooks()])
        text, _ = self.instructions(status)
        self.assertEqual(text, proxy.SERVER_INSTRUCTIONS + ' ' + SetupSession.instructions)

    def test_failed_check_is_logged_and_adds_nothing(self):
        from servers import brain_mcp as proxy
        text, logged = self.instructions(error=codex.ReviewError('Codex setup request timed out.'))
        self.assertEqual(text, proxy.SERVER_INSTRUCTIONS + ' ' + SetupSession.instructions)
        self.assertIn('timed out', logged[0][1])


class TestPromptHookCarriesNotice(unittest.TestCase):
    """The trusted prompt hook speaks for the hooks awaiting approval."""

    def setUp(self):
        self.adapter = type(sys)('codex_setup')
        self.adapter.prompt_notice = self.notice = Mock(return_value='[brain] SETUP LINE')
        self.tmp = tempfile.mkdtemp(prefix='entity-hook-notice-')
        self.addCleanup(__import__('shutil').rmtree, self.tmp, True)

    def run_hook(self, prompt, tells, session='chat-one', join_s=None):
        import runpy
        scripts = ROOT / 'hooks' / 'scripts'
        sys.path.insert(0, str(scripts))
        self.addCleanup(sys.path.remove, str(scripts))
        import hook_common
        data = {'prompt': prompt, 'session_id': session, 'model': 'm', 'hook_event_name': 'UserPromptSubmit'}
        reply = {'ok': True, 'result': {'json': {'additionalContext': 'recalled'}}}
        with patch.dict(os.environ, {**tells, 'BRAIN_TMP_DIR': self.tmp}, clear=True), \
             patch.dict(sys.modules, {'codex_setup': self.adapter}), \
             patch.object(hook_common, 'get_hook_input', return_value=data), \
             patch.object(hook_common, 'daemon_available', return_value=True), \
             patch.object(hook_common, 'daemon_call_raw', return_value=reply) as self.daemon, \
             patch.object(hook_common, 'brain_debug'), \
             patch.object(hook_common, 'emit_hook_output') as emit, \
             patch('os._exit'), \
             patch.object(hook_common, 'run_hook', side_effect=lambda name, fn, on_error=None, gated=True: fn()), \
             patch.object(hook_common, 'log_hook_error') as self.logged, \
             patch.object(threading.Thread, 'join', self.short_join(join_s)):
            runpy.run_path(str(scripts / 'pre_response_recall.py'))
        return emit.call_args.args[1]['additionalContext']

    @staticmethod
    def short_join(join_s, _join=threading.Thread.join):
        return lambda thread, timeout=None: _join(thread, timeout if join_s is None else join_s)

    def test_codex_first_prompt_gets_the_notice_after_recall_and_later_prompts_do_not(self):
        codex_tells = {'PLUGIN_DATA': 'x'}
        self.assertEqual(self.run_hook('what did we decide', codex_tells), 'recalled\n\n[brain] SETUP LINE')
        self.notice.assert_called_once_with(str(ROOT))
        self.assertEqual(self.run_hook('and then what', codex_tells), 'recalled')
        self.assertEqual(self.run_hook('a new chat', codex_tells, session='chat-two'),
                         'recalled\n\n[brain] SETUP LINE')

    def test_chats_opened_the_same_minute_each_get_the_notice(self):
        # Codex ids are UUIDv7: chats opened together share their first 8 characters.
        codex_tells = {'PLUGIN_DATA': 'x'}
        for session in ('01a122b6-9059-7ca1-bb1b-a5e6ff4ce0cf', '01a122b6-1111-7ca1-bb1b-a5e6ff4ce0cf'):
            self.assertEqual(self.run_hook('what did we decide', codex_tells, session=session),
                             'recalled\n\n[brain] SETUP LINE')

    def test_a_check_still_running_at_reply_time_asks_again_next_prompt(self):
        codex_tells = {'PLUGIN_DATA': 'x'}
        release = threading.Event()
        self.notice.side_effect = lambda root: release.wait(5) and '[brain] SETUP LINE'
        self.assertEqual(self.run_hook('what did we decide', codex_tells, join_s=0.05), 'recalled')
        release.set()
        self.notice.side_effect = None
        self.assertEqual(self.run_hook('and then', codex_tells), 'recalled\n\n[brain] SETUP LINE')

    def test_a_failed_check_is_logged_once_per_chat(self):
        codex_tells = {'PLUGIN_DATA': 'x'}
        self.notice.side_effect = codex.ReviewError('Codex was not found.')
        self.assertEqual(self.run_hook('what did we decide', codex_tells), 'recalled')
        self.assertIn('setup check', self.logged.call_args.args[2])
        self.assertEqual(self.run_hook('and then', codex_tells), 'recalled')
        self.assertEqual(self.notice.call_count, 1)

    def test_a_host_without_an_adapter_never_asks(self):
        self.assertEqual(self.run_hook('what did we decide', {'CLAUDE_CODE_SESSION_ID': 's'}), 'recalled')
        self.notice.assert_not_called()

    def test_a_machine_envelope_defers_to_the_next_operator_prompt(self):
        codex_tells = {'PLUGIN_DATA': 'x'}
        self.run_hook('<task-notification>\ndone</task-notification>', codex_tells)
        self.notice.assert_not_called()
        self.assertEqual(self.run_hook('what happened', codex_tells), 'recalled\n\n[brain] SETUP LINE')

    def test_the_hook_sends_the_contracts_reading_of_the_prompt(self):
        cc = {'CLAUDE_CODE_SESSION_ID': 's'}
        self.run_hook('<agent-message from="x">\n[Subagent hand-back] notice\n  the report', cc)
        sent = self.daemon.call_args.args[1]
        self.assertEqual(sent['pieces'], [['subagent', 'the report', '<agent-message']])
        self.assertEqual(sent['recall_text'], 'the report')
        self.assertFalse(sent['register_only'])
        self.assertTrue(sent['impl_identity'])
        self.run_hook('<system-reminder>\nworktree notice\n</system-reminder>', cc)
        sent = self.daemon.call_args.args[1]
        self.assertEqual(sent['pieces'], [['harness', 'worktree notice', '<system-reminder>']])
        self.assertTrue(sent['register_only'])

    def test_a_slash_command_never_reaches_the_daemon(self):
        with self.assertRaises(SystemExit):
            self.run_hook('/watch', {'CLAUDE_CODE_SESSION_ID': 's'})
        self.daemon.assert_not_called()


class TestFindCodex(unittest.TestCase):
    """One owner of the codex lookup: the desktop bundle's current layout
    (codex-cli/ + codex-package.json entrypoint), the flat layout before it,
    then PATH. scripts/codex-install.sh asks this module instead of spelling
    a path."""

    def _bundle(self, tmp, *, packaged, flat):
        resources = Path(tmp)
        if packaged:
            (resources / 'codex-cli/bin').mkdir(parents=True)
            (resources / 'codex-cli/codex-package.json').write_text(json.dumps(
                {'layoutVersion': 1, 'entrypoint': 'bin/codex'}))
            (resources / 'codex-cli/bin/codex').write_text('#!/bin/sh\n')
            (resources / 'codex-cli/bin/codex').chmod(0o755)
        if flat:
            (resources / 'codex').write_text('#!/bin/sh\n')
            (resources / 'codex').chmod(0o755)
        return resources

    def test_candidates_read_the_package_entrypoint_then_the_flat_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources = Path(tmp)
            (resources / 'codex-cli').mkdir()
            (resources / 'codex-cli/codex-package.json').write_text(json.dumps({'entrypoint': 'other/codex'}))
            self.assertEqual(list(codex.bundled_codex_candidates([tmp])),
                             [str(resources / 'codex-cli/other/codex'), str(resources / 'codex')])

    def test_candidates_without_a_package_file_assume_bin_codex(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list(codex.bundled_codex_candidates([tmp])),
                             [f'{tmp}/codex-cli/bin/codex', f'{tmp}/codex'])

    def test_find_codex_prefers_the_packaged_layout_over_the_flat_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources = self._bundle(tmp, packaged=True, flat=True)
            with patch.object(codex, 'APP_RESOURCES', (tmp,)), patch.object(codex.shutil, 'which', return_value=None):
                self.assertEqual(codex.find_codex(), str(resources / 'codex-cli/bin/codex'))

    def test_find_codex_falls_back_to_the_flat_layout_then_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources = self._bundle(tmp, packaged=False, flat=True)
            with patch.object(codex, 'APP_RESOURCES', (tmp,)), patch.object(codex.shutil, 'which', return_value='/usr/bin/true'):
                self.assertEqual(codex.find_codex(), str(resources / 'codex'))
            with patch.object(codex, 'APP_RESOURCES', (tmp + '/none',)), patch.object(codex.shutil, 'which', return_value='/usr/bin/true'):
                self.assertEqual(codex.find_codex(), '/usr/bin/true')
            with patch.object(codex, 'APP_RESOURCES', (tmp + '/none',)), patch.object(codex.shutil, 'which', return_value=None):
                with self.assertRaisesRegex(codex.ReviewError, 'not found'):
                    codex.find_codex()

    def test_cli_entry_prints_the_lookup_and_honours_codex_override(self):
        module = str(Path(codex.__file__))
        ok = subprocess.run([sys.executable, module, '--codex', '/usr/bin/true'], capture_output=True, text=True)
        self.assertEqual((ok.returncode, ok.stdout.strip()), (0, '/usr/bin/true'))
        bad = subprocess.run([sys.executable, module, '--codex', '/nonexistent/codex'], capture_output=True, text=True)
        self.assertEqual((bad.returncode, bad.stdout), (1, ''))
        self.assertIn('not found', bad.stderr)


def _desktop_codex():
    """The bundled CLI through the adapter's own lookup, or None."""
    try:
        return codex.find_codex()
    except codex.ReviewError:
        return None


@unittest.skipUnless(_desktop_codex(), 'requires the desktop bundled Codex for native protocol checks')
@pytest.mark.slow
class TestNativeStatus(unittest.TestCase):
    def test_real_codex_detects_untrusted_trusted_and_modified_definitions(self):
        with tempfile.TemporaryDirectory(prefix='entity-native-status-') as tmp:
            home = Path(tmp)
            plugin = home / 'plugins/cache/test/entity/0.9.0'
            (plugin / '.codex-plugin').mkdir(parents=True)
            (plugin / 'hooks').mkdir()
            (plugin / '.codex-plugin/plugin.json').write_text(json.dumps({
                'name': 'entity', 'version': '0.9.0', 'hooks': './hooks/hooks.json'}))
            definition = plugin / 'hooks/hooks.json'
            definition.write_text(json.dumps({'hooks': {'Stop': [
                {'hooks': [{'type': 'command', 'command': '/usr/bin/true'}]}]}}))
            config = home / 'config.toml'
            original = ('# preserve this comment\n[features]\nhooks=true\nplugin_hooks=true\n[plugins."entity@test"]\nenabled=true\n'
                        '[plugins."entity@test".mcp_servers.brain.tools.recall]\napproval_mode="prompt"\n'
                        '[plugins."other@test".mcp_servers.brain]\ndefault_tools_approval_mode="prompt"\n')
            config.write_text(original)
            def inspect():
                return codex.setup_status(codex_home=home, cwd=home, plugin_root=plugin)
            untrusted = inspect()
            self.assertEqual(untrusted['state'], 'review_required')
            self.assertEqual((untrusted['hooks'][0]['command'], untrusted['hooks'][0]['matcher']),
                             ('/usr/bin/true', None))
            import codex_setup
            self.assertIn('so automatic memory is off', codex_setup.approval_line(untrusted))
            self.assertEqual(config.read_text(), original, 'Inspection must not persist hook approval')
            wrong = 'entity@different'
            with self.assertRaisesRegex(codex.ReviewError, 'installation changed'):
                codex.approve_tools(plugin_root=plugin, plugin_id=wrong, codex_home=home, cwd=home)
            self.assertEqual(config.read_text(), original)
            permission = codex.approve_tools(plugin_root=plugin, plugin_id='entity@test', codex_home=home, cwd=home)
            self.assertTrue(permission['saved'])
            approved_config = config.read_text()
            self.assertNotIn('trusted_hash', approved_config, 'Tool permission must never grant hook trust')
            self.assertEqual(inspect()['tool_approval']['mode'], 'approve')
            self.assertEqual(inspect()['tool_approval']['restricted_tools'], ['recall'])
            import tomllib
            saved = tomllib.loads(approved_config)
            self.assertEqual(saved['plugins']['other@test']['mcp_servers']['brain']['default_tools_approval_mode'], 'prompt')
            self.assertIn('# preserve this comment', approved_config)
            call = codex._Client.call
            before_race = approved_config.replace('"approve"', '"auto"')
            self.assertNotEqual(before_race, approved_config)
            config.write_text(before_race)
            concurrent = before_race.replace('"prompt"', '"auto"')
            self.assertNotEqual(concurrent, before_race)
            def changed_after_read(client, method, params):
                result = call(client, method, params)
                if method == 'config/read' and params.get('includeLayers'):
                    config.write_text(concurrent)
                return result
            with patch.object(codex._Client, 'call', new=changed_after_read):
                with self.assertRaises(codex.ReviewError):
                    codex.approve_tools(plugin_root=plugin, plugin_id='entity@test', codex_home=home, cwd=home)
            self.assertEqual(config.read_text(), concurrent, 'A stale write must preserve the concurrent edit')
            item = untrusted['hooks'][0]
            # Test fixture only: simulate Codex's saved user decision in this
            # throwaway home. Production adapter has no configuration writer.
            config.write_text(approved_config + '\n[hooks.state.' + json.dumps(item['key']) + ']\ntrusted_hash=' + json.dumps(item['currentHash']) + '\n')
            trusted = inspect()
            self.assertTrue(trusted['trust_complete'])
            self.assertFalse(trusted['runtime_verified'])
            definition.write_text(definition.read_text().replace('/usr/bin/true', '/usr/bin/false'))
            changed = inspect()
            self.assertEqual(changed['state'], 'review_required')
            self.assertFalse(changed['trust_complete'])
            self.assertNotEqual(changed['hooks'][0]['currentHash'], item['currentHash'])
