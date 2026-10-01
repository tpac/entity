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
from unittest.mock import patch
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

    def test_launcher_preserves_paths_and_home_without_shell_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "fake codex ' binary"
            output = root / 'captured.json'
            marker = root / 'injected'
            binary.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                              'open(os.environ["CAPTURE"],"w").write(json.dumps([sys.argv[1:],os.environ["CODEX_HOME"]]))\n')
            binary.chmod(0o700)
            home = "odd ' home; touch " + str(marker) + '; $(false)'
            cwd = "space ' project"
            script = root / 'review.command'
            script.write_text(codex.review_script(binary, home, cwd))
            subprocess.run(['/bin/sh', str(script)], env={**os.environ, 'CAPTURE': str(output)},
                           check=True, capture_output=True, timeout=5)
            self.assertEqual(json.loads(output.read_text()), [['--no-alt-screen', '--cd', cwd], home])
            self.assertFalse(marker.exists())

    def test_unsupported_hosts_return_manual_guidance_without_launch(self):
        for platform in ('linux', 'win32'):
            with patch.object(codex.sys, 'platform', platform), \
                 patch.object(codex.subprocess, 'run') as launch:
                with self.assertRaisesRegex(codex.ReviewError, 'Open Codex CLI'):
                    codex.open_review()
            launch.assert_not_called()


class TestConsent(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.session = SetupSession(self.sent.append, '/installed/entity')
        # Deterministic consent-state tests; worker behavior is exercised below
        # using the real dispatch method and explicit synchronization events.
        self.session._dispatch = lambda job: job()
        self.addCleanup(self.session.close)
        self.initialize({'protocolVersion': '2025-11-25', 'capabilities': {'elicitation': {'form': {}}}})
        self.status = patch.object(codex, 'setup_status', return_value=setup_status())
        self.status_mock = self.status.start()
        self.addCleanup(self.status.stop)
        self.approval = patch.object(codex, 'approve_tools', return_value={'saved': True, 'mode': 'approve'})
        self.approve = self.approval.start()
        self.addCleanup(self.approval.stop)
        self.launch = patch.object(codex, 'open_review', return_value={'state': 'review_launched', 'trust_granted': False})
        self.open = self.launch.start()
        self.addCleanup(self.launch.stop)

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

    def test_accept_opens_once_and_does_not_grant_trust(self):
        eid = self.request()
        self.assertEqual(self.sent[-1]['method'], 'elicitation/create')
        self.reply(eid)
        self.reply(eid)
        self.open.assert_called_once_with()
        self.approve.assert_called_once_with(plugin_root='/installed/entity', plugin_id='entity@local')
        self.assertEqual(self.sent[-1]['id'], 1)
        self.assertFalse(self.payload()['hook_trust_granted'])

    def test_decline_cancel_false_and_malformed_accept_never_launch(self):
        for action, value, reason in [('decline', True, 'declined'), ('cancel', True, 'dismissed'),
                                      ('accept', False, 'unchecked'), ('accept', 'true', 'invalid_response')]:
            self.reply(self.request(), action, value)
            self.assertEqual(self.payload()['state'], 'review_cancelled')
            self.assertEqual(self.payload()['reason'], reason)
        self.open.assert_not_called()
        self.approve.assert_not_called()

    def test_declined_form_cannot_grant_on_replay_but_fresh_request_can(self):
        declined = self.request()
        self.reply(declined, action='decline')
        self.assertIsNone(self.session.current)
        self.assertEqual(len(self.sent), 2, 'A declined form must not reopen itself')
        self.reply(declined)
        self.approve.assert_not_called()
        self.open.assert_not_called()
        fresh = self.request(2)
        self.assertNotEqual(fresh, declined)
        self.reply(fresh)
        self.approve.assert_called_once()
        self.open.assert_called_once()

    def test_cancellation_expiry_and_unknown_id_cannot_launch(self):
        eid = self.request()
        self.reply('wrong-id')
        self.session.cancel(1)
        self.reply(eid)
        eid = self.request(2)
        self.session._expire(self.session.current)
        self.assertEqual(self.payload()['state'], 'confirmation_expired')
        self.reply(eid)
        self.open.assert_not_called()
        self.approve.assert_not_called()

    def test_no_form_capability_or_old_protocol_does_not_prompt(self):
        for caps, version in [({}, '2025-11-25'), ({'elicitation': {'url': {}}}, '2025-11-25'),
                              ({'elicitation': {}}, '2024-11-05')]:
            self.initialize({'protocolVersion': version, 'capabilities': caps})
            self.session.start(1, {'action': 'review'})
            self.assertEqual(self.payload()['state'], 'confirmation_unavailable')
        self.open.assert_not_called()
        self.approve.assert_not_called()

    def test_status_never_prompts_and_preserves_identity_evidence(self):
        self.session.start(9, {'action': 'status'}, identity_verified=True)
        self.assertTrue(self.payload()['caller_identity_verified'])
        self.assertEqual(self.payload()['state'], 'review_required')
        self.open.assert_not_called()
        self.approve.assert_not_called()

    def test_second_request_does_not_replace_first_confirmation(self):
        eid = self.request()
        self.request(2)
        self.assertEqual(self.payload()['state'], 'busy')
        self.reply(eid)
        self.assertEqual(self.sent[-1]['id'], 1)
        self.open.assert_called_once()

    def test_launch_error_is_not_success(self):
        self.open.side_effect = codex.ReviewError('launch unavailable')
        self.reply(self.request())
        self.assertEqual(self.payload()['state'], 'setup_incomplete')
        self.assertTrue(self.payload()['tool_permission']['saved'])
        self.assertTrue(self.sent[-1]['result']['isError'])

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

    def test_launch_runs_on_worker_and_reader_can_cancel_its_result(self):
        eid = self.request()
        entered, release = threading.Event(), threading.Event()
        workers = []
        dispatch = SetupSession._dispatch.__get__(self.session)
        self.session._dispatch = lambda job: workers.append(dispatch(job))
        def slow_launch():
            entered.set()
            release.wait(3)
            return {'state': 'review_launched', 'trust_granted': False}
        self.open.side_effect = slow_launch
        try:
            self.reply(eid)
            self.assertTrue(entered.wait(1), 'launch worker did not start')
            self.session.cancel(1)
        finally:
            release.set()
            for worker in workers:
                worker.join(3)
        self.assertEqual(len(self.sent), 1, 'cancelled launch must not send a stale tool result')

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
        self.status_mock.return_value = setup_status([hook(trustStatus='trusted')])
        self.reply(self.request())
        self.approve.assert_called_once()
        self.open.assert_not_called()
        self.assertTrue(self.payload()['tool_permission']['saved'])

    def test_existing_tool_approval_only_opens_hook_review(self):
        status = setup_status()
        status['tool_approval']['mode'] = 'approve'
        self.status_mock.return_value = status
        self.reply(self.request())
        self.approve.assert_not_called()
        self.open.assert_called_once()

    def test_cancel_before_acceptance_worker_starts_saves_no_permission(self):
        eid = self.request()
        jobs = []
        self.session._dispatch = jobs.append
        self.reply(eid)
        self.session.cancel(1)
        jobs[0]()
        self.approve.assert_not_called()
        self.open.assert_not_called()

    def test_model_arguments_cannot_replace_user_confirmation(self):
        self.session.start(1, {'action': 'review', 'enable_entity': True})
        self.assertEqual(self.sent[-1]['method'], 'elicitation/create')
        self.approve.assert_not_called()
        self.open.assert_not_called()

    def test_failed_permission_write_does_not_launch_hook_review(self):
        self.approve.side_effect = codex.ReviewError('configuration changed; check again')
        self.reply(self.request())
        self.open.assert_not_called()
        self.assertEqual(self.payload()['state'], 'setup_incomplete')
        self.assertIsNone(self.payload()['tool_permission'])


class TestProxy(unittest.TestCase):
    def setUp(self):
        self.extension = SetupSession(lambda message: None, '/installed/entity')
        self.addCleanup(self.extension.close)

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
        response = proxy.handle_initialize(1, {'protocolVersion': 'unknown'})
        self.assertEqual(response['result']['protocolVersion'], proxy.PROTOCOL_VERSION)

    def test_adapter_receives_negotiated_version_and_compatible_capabilities(self):
        from servers import brain_mcp as proxy
        capabilities = {'elicitation': {'form': {}}, 'roots': {}}
        with patch.object(self.extension, 'initialize') as initialize:
            for version in (proxy.PROTOCOL_VERSION, 'unknown'):
                proxy.handle_initialize(1, {'protocolVersion': version, 'capabilities': capabilities}, self.extension)
                initialize.assert_called_with({'protocolVersion': proxy.PROTOCOL_VERSION, 'capabilities': {'roots': {}}})
        self.assertIn('elicitation', capabilities, 'negotiation must not mutate the client request')
        proxy.handle_initialize(1, {'protocolVersion': 'unknown', 'capabilities': capabilities}, self.extension)
        self.assertFalse(self.extension.form_supported)

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
             patch.object(proxy, 'check_daemon_fingerprint'), \
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
             patch.object(proxy, 'check_daemon_fingerprint'), \
             patch.object(proxy, '_health_monitor'), \
             patch.object(codex, 'setup_status', return_value=status), \
             patch.object(codex, 'open_review') as launch, \
             patch.object(codex_setup.uuid, 'uuid4') as uid, \
             patch('sys.stdin', io.StringIO('\n'.join(json.dumps(m) for m in messages))), \
             patch('sys.stdout', output):
            uid.return_value.hex = 'fixed'
            proxy.main(extension=extension)
        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertIn({'jsonrpc': '2.0', 'id': 3, 'result': {}}, replies)
        self.assertEqual(len([r for r in replies if r.get('method') == 'elicitation/create']), 1)
        self.assertIsNone(extension.current)
        launch.assert_not_called()


@unittest.skipUnless(os.path.isfile('/Applications/ChatGPT.app/Contents/Resources/codex'),
                     'requires the desktop bundled Codex for native protocol checks')
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
