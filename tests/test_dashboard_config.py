"""Dashboard Config tab backend — one composed read (daemon ping + gate
status + the dashboard's own paths) and two daemon-mediated writes (pause /
resume, the API key the /setup page collects).

The dashboard writes no file itself: both POSTs proxy to daemon commands,
the server accepts only pause / resume from a page, and the loopback guard
covers the new write route like the others. The HTTP tests run the real
server on an ephemeral port with the daemon client monkeypatched.
"""
import http.client
import json
import pathlib
import threading
import time

import pytest

from dashboard import server as dash_server
from dashboard.queries import config, system

PING = {'pid': 7, 'db_dir': '/brains/a', 'source_dir': '/src',
        'code_fingerprint': 'abcdef0123456789', 'uptime_seconds': 5}
GATE = {'paused': False, 'ignore_projects': ['*', '!brain'], 'env_file': '/cfg/brain/env',
        'verdicts': {'': {'run': False, 'reason': 'ignored_project:'},
                     'brain': {'run': True, 'reason': ''}}}


def _fake_send(monkeypatch, answers):
    calls = []

    def send(cmd, args=None, timeout=10):
        calls.append((cmd, args or {}))
        return answers.get(cmd)
    # The ping goes through system.daemon_record (the one ping→view mapping);
    # the gate call is config's own.
    monkeypatch.setattr(system, 'daemon_send', send)
    monkeypatch.setattr(config, 'daemon_send', send)
    return calls


def _fake_call(monkeypatch, reply):
    calls = []

    def call(cmd, args=None, timeout=10):
        calls.append((cmd, args or {}))
        return reply(cmd, args or {}) if callable(reply) else reply
    monkeypatch.setattr(config, 'daemon_call', call)
    return calls


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv('BRAIN_DB_DIR', '/brains/a')
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    return tmp_path


def test_query_config_composes_daemon_and_dashboard_views(monkeypatch, isolated):
    calls = _fake_send(monkeypatch, {'ping': PING, 'gate': GATE})
    out = config.query_config(['brain', 'brain'])
    assert set(out['daemon']) == {'alive', 'pid', 'uptime', 'code_fingerprint', 'db_dir', 'source_dir', 'encode_every'}
    assert out['daemon']['alive'] and out['daemon']['db_dir'] == '/brains/a'
    assert out['daemon']['code_fingerprint'] == 'abcdef012345'   # the record's truncation
    assert out['daemon']['source_dir'] == '/src' and out['daemon']['uptime'] == 5
    assert out['dashboard']['db_dir'] == '/brains/a'
    assert out['dashboard']['env_file'] == str(isolated / 'brain' / 'env')
    assert out['db_dir_matches'] is True
    assert out['gate'] == GATE
    assert out['api_key_present'] is False
    # One gate call: the unscoped project first, then each named one, deduped.
    assert calls == [('ping', {}), ('gate', {'action': 'status', 'projects': ['', 'brain']})]


def test_query_config_flags_a_db_dir_mismatch(monkeypatch, isolated):
    monkeypatch.setenv('BRAIN_DB_DIR', '/brains/b')
    _fake_send(monkeypatch, {'ping': PING, 'gate': GATE})
    out = config.query_config([])
    assert out['daemon']['db_dir'] == '/brains/a'
    assert out['dashboard']['db_dir'] == '/brains/b'
    assert out['db_dir_matches'] is False


def test_query_config_when_the_daemon_is_down(monkeypatch, isolated):
    (isolated / 'brain').mkdir()
    # Shell spelling — the file is sourced by brain-env.sh, so `export` is a
    # natural way to write it; presence must read it like BRAIN_DB_DIR is read.
    (isolated / 'brain' / 'env').write_text('export ANTHROPIC_API_KEY="sk-ant-x"\n')
    calls = _fake_send(monkeypatch, {})
    out = config.query_config(['brain'])
    assert out['daemon'] == {'alive': False, 'error': 'no response'}
    assert out['gate'] is None and out['db_dir_matches'] is None
    assert out['api_key_present'] is True          # a presence read, no daemon needed
    assert calls == [('ping', {})]                 # the gate is not asked


def test_set_api_key_relays_the_daemon_verdict_without_the_key(monkeypatch):
    calls = _fake_call(monkeypatch, {'ok': True, 'result': {'key_present': True}})
    ok, msg = config.set_api_key('sk-ant-test-abc123')
    assert ok and 'sk-ant' not in msg
    assert calls == [('set_api_key', {'api_key': 'sk-ant-test-abc123'})]
    _fake_call(monkeypatch, {'ok': False, 'error': "that doesn't look like an Anthropic API key"})
    ok, msg = config.set_api_key('changeme')
    assert not ok and 'changeme' not in msg
    _fake_call(monkeypatch, {'ok': False, 'error': 'daemon unreachable'})
    ok, msg = config.set_api_key('sk-ant-test-abc123')
    assert not ok and 'daemon' in msg and 'sk-ant' not in msg


# ── The routes, through the real server ───────────────────────────────────

@pytest.fixture
def served():
    srv = dash_server.ThreadedHTTPServer(('127.0.0.1', 0), dash_server.DashboardHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


def _post(port, path, body, host='127.0.0.1', ctype='application/json'):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    raw = json.dumps(body)
    conn.request('POST', path, raw, headers={'Host': host, 'Content-Type': ctype,
                                             'Content-Length': str(len(raw))})
    r = conn.getresponse()
    out = (r.status, json.loads(r.read().decode()))
    conn.close()
    return out


def test_gate_route_proxies_pause_and_resume_only(monkeypatch, served):
    calls = _fake_call(monkeypatch, lambda cmd, args: {'ok': True, 'result': {'paused': args['action'] == 'pause'}})
    status, out = _post(served, '/api/gate', {'action': 'pause'})
    assert (status, out) == (200, {'ok': True, 'result': {'paused': True}})
    status, out = _post(served, '/api/gate', {'action': 'resume'})
    assert (status, out['result']) == (200, {'paused': False})
    assert calls == [('gate', {'action': 'pause'}), ('gate', {'action': 'resume'})]
    # The ignore list is the MCP tool's to edit, not a page's.
    status, out = _post(served, '/api/gate', {'action': 'ignore', 'pattern': '*'})
    assert status == 400 and not out['ok']
    status, out = _post(served, '/api/gate', {})
    assert status == 400
    assert len(calls) == 2


def test_gate_route_reports_a_daemon_refusal(monkeypatch, served):
    _fake_call(monkeypatch, {'ok': False, 'error': 'daemon unreachable'})
    status, out = _post(served, '/api/gate', {'action': 'pause'})
    assert status == 502 and out == {'ok': False, 'error': 'daemon unreachable'}


def test_gate_route_is_behind_the_loopback_guard(monkeypatch, served):
    calls = _fake_call(monkeypatch, {'ok': True, 'result': {}})
    status, _ = _post(served, '/api/gate', {'action': 'pause'}, host='attacker.example')
    assert status == 403
    status, _ = _post(served, '/api/gate', {'action': 'pause'}, ctype='text/plain')
    assert status == 400
    assert calls == []


def test_setup_key_route_proxies_to_the_daemon(monkeypatch, served):
    calls = _fake_call(monkeypatch, {'ok': True, 'result': {'key_present': True}})
    status, out = _post(served, '/api/setup-key', {'api_key': ' sk-ant-test-abc123 '})
    assert status == 200 and out['ok'] and 'sk-ant' not in out['message']
    assert calls == [('set_api_key', {'api_key': 'sk-ant-test-abc123'})]
    _fake_call(monkeypatch, {'ok': False, 'error': 'the key contains whitespace'})
    status, out = _post(served, '/api/setup-key', {'api_key': 'sk-ant bad'})
    assert status == 400 and not out['ok'] and 'whitespace' in out['message']


def test_config_route_passes_the_projects(monkeypatch, served, isolated):
    calls = _fake_send(monkeypatch, {'ping': PING, 'gate': GATE})
    conn = http.client.HTTPConnection('127.0.0.1', served, timeout=5)
    conn.request('GET', '/api/config?projects=brain,,client-x')
    out = json.loads(conn.getresponse().read().decode())
    conn.close()
    assert out['gate'] == GATE and out['db_dir_matches'] is True
    assert calls[1] == ('gate', {'action': 'status', 'projects': ['', 'brain', 'client-x']})


# ── The real wire: dashboard → daemon process → env file → hook gate ──────
# Everything above monkeypatches the daemon. These spawn one — keyless, on an
# empty temp brain, under the suite's own instance and port pins — and prove
# the one thing the in-process tests cannot: a POST on the dashboard changes
# what the hooks read. Each process's own behavior is covered elsewhere
# (test_brain_gate for the gate, the handler and the boot script; the tests
# above for the routes), so this stays at the seams: the file the daemon
# wrote is the file the gate reads, and the verdicts the dashboard renders are
# the daemon's own. One daemon per module (~4 s to a live ping on an empty
# brain); no key in its config dir, so it makes no network call.

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def live_daemon(tmp_path_factory):
    from servers import daemon_client
    from servers.daemon_launch import spawn_detached_daemon

    root = tmp_path_factory.mktemp('gate-wire')
    cfg_dir = root / 'xdg' / 'brain'
    cfg_dir.mkdir(parents=True)
    db_dir = root / 'brain'
    db_dir.mkdir()
    mp = pytest.MonkeyPatch()
    # The config dir holds NO key: the daemon runs keyless and warms no API
    # connection. The gate's env file is this dir's — the daemon writes it,
    # brain_gate in this process reads it, the dashboard reads it for presence.
    mp.setenv('XDG_CONFIG_HOME', str(root / 'xdg'))
    mp.delenv('ANTHROPIC_API_KEY', raising=False)
    mp.setenv('BRAIN_DB_DIR', str(db_dir))
    spawn_detached_daemon(str(db_dir / 'brain.db'))
    deadline = time.time() + 90
    while time.time() < deadline:
        if daemon_client.send_command('ping', timeout=2).get('ok'):
            break
        time.sleep(0.5)
    else:
        daemon_client.stop_daemon()
        mp.undo()
        pytest.fail('the isolated daemon did not answer ping within 90 s — see %s'
                    % (db_dir / 'daemon.log'))
    try:
        yield {'env_file': cfg_dir / 'env', 'db_dir': db_dir}
    finally:
        daemon_client.stop_daemon()
        mp.undo()


def _get(port, path):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    conn.request('GET', path)
    out = json.loads(conn.getresponse().read().decode())
    conn.close()
    return out


def test_wire_dashboard_pause_is_what_the_hooks_read(live_daemon, served):
    from servers import brain_gate

    out = _get(served, '/api/config?projects=brain')
    assert out['daemon']['alive'] is True
    assert out['db_dir_matches'] is True, (out['daemon'].get('db_dir'), out['dashboard'])
    assert out['api_key_present'] is False
    assert out['gate']['paused'] is False
    assert out['gate']['verdicts']['brain'] == {'run': True, 'reason': ''}
    assert out['gate']['env_file'] == str(live_daemon['env_file'])

    status, body = _post(served, '/api/gate', {'action': 'pause'})
    assert (status, body['result']['paused']) == (200, True)
    # The daemon wrote the switch into the file the hook-side gate reads …
    assert brain_gate.read_switch(brain_gate.PAUSED_KEY) == '1'
    assert brain_gate.hook_allows(str(ROOT), 'wire-sess-1') == brain_gate.Decision(False, brain_gate.REASON_PAUSED)
    # … and the verdict the dashboard renders is the daemon's own.
    assert _get(served, '/api/config?projects=brain')['gate']['verdicts']['brain'] == {
        'run': False, 'reason': brain_gate.REASON_PAUSED}

    status, body = _post(served, '/api/gate', {'action': 'resume'})
    assert (status, body['result']['paused']) == (200, False)
    assert brain_gate.read_switch(brain_gate.PAUSED_KEY) == ''
    assert brain_gate.hook_allows(str(ROOT), 'wire-sess-2').run is True
    assert live_daemon['env_file'].read_text().count('BRAIN_PAUSED') == 0


def test_wire_ignore_pattern_on_the_daemon_shows_in_the_verdicts(live_daemon, served):
    from servers import brain_gate, daemon_client

    assert daemon_client.send_command('gate', {'action': 'ignore', 'pattern': 'client-*'})['ok']
    try:
        out = _get(served, '/api/config?projects=client-x,brain')
        assert out['gate']['ignore_projects'] == ['client-*']
        assert out['gate']['verdicts']['client-x'] == {'run': False, 'reason': 'ignored_project:client-x'}
        assert out['gate']['verdicts']['brain']['run'] is True
        # Same file, same answer on the hook side.
        assert brain_gate.decide('client-x').run is False
        assert brain_gate.decide('brain').run is True
    finally:
        daemon_client.send_command('gate', {'action': 'unignore', 'pattern': 'client-*'})
    assert _get(served, '/api/config')['gate']['ignore_projects'] == []
