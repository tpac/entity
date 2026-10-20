"""Dashboard /api/self-send — the operator's door into the self-channel courier.

The route proxies to the daemon's self_send and hands the daemon's refusal
(an over-limit body, an unknown target) back to the page as its own error.
The HTTP tests run the real server on an ephemeral port with the daemon
client monkeypatched.
"""
import http.client
import json
import threading

import pytest

from dashboard import server as dash_server


@pytest.fixture
def served():
    srv = dash_server.ThreadedHTTPServer(('127.0.0.1', 0), dash_server.DashboardHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


def _post(port, body):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    raw = json.dumps(body)
    conn.request('POST', '/api/self-send', raw,
                 headers={'Host': '127.0.0.1', 'Content-Type': 'application/json',
                          'Content-Length': str(len(raw))})
    r = conn.getresponse()
    out = (r.status, json.loads(r.read().decode()))
    conn.close()
    return out


def _daemon(monkeypatch, reply):
    calls = []

    def call(cmd, args=None, timeout=10):
        calls.append((cmd, args))
        return reply
    monkeypatch.setattr(dash_server, 'daemon_alive', lambda: True)
    monkeypatch.setattr(dash_server, 'daemon_call', call)
    return calls


def test_send_relays_the_daemon_result(monkeypatch, served):
    calls = _daemon(monkeypatch, {'ok': True, 'result': {'id': 'abc'}})
    status, out = _post(served, {'to': 'broadcast', 'body': 'hello'})
    assert (status, out) == (200, {'ok': True, 'result': {'id': 'abc'}})
    assert calls[0][0] == 'self_send'
    assert calls[0][1]['from_session'] == 'operator-dashboard'


def test_send_relays_the_daemon_refusal(monkeypatch, served):
    refusal = 'self.signal.send: the message is 5,001 characters and the limit is 5,000.'
    _daemon(monkeypatch, {'ok': False, 'error': refusal})
    status, out = _post(served, {'to': 'broadcast', 'body': 'x' * 5001})
    assert (status, out) == (502, {'ok': False, 'error': refusal})
