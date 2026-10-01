"""Codex setup: inspect hooks, save consented tool policy, open native review.

No Entity imports, brain access, hook execution, hook-trust writes, or approval
keystrokes. The MCP caller owns consent before approve_tools/open_review.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import selectors
import shlex
import shutil
import subprocess
import sys
import tempfile
import time


class ReviewError(RuntimeError):
    pass


def find_codex(executable=None):
    """Prefer the desktop's bundled runtime to a possibly different PATH CLI."""
    candidates = [executable] if executable else [
        '/Applications/ChatGPT.app/Contents/Resources/codex',
        '/Applications/Codex.app/Contents/Resources/codex',
        str(Path.home() / 'Applications/ChatGPT.app/Contents/Resources/codex'),
        shutil.which('codex'),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).absolute())
    raise ReviewError('Codex was not found. Open or install the desktop app, then try again.')


class _Client:
    def __init__(self, executable, codex_home, cwd, timeout=12):
        self.timeout = timeout
        self.cwd = str(cwd)
        self.seq = 0
        self.buffer = b''
        self.process = subprocess.Popen(
            [executable, 'app-server', '--stdio'], cwd=cwd,
            env={**os.environ, 'CODEX_HOME': str(codex_home)},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def send(self, message):
        self.process.stdin.write((json.dumps(message) + '\n').encode())
        self.process.stdin.flush()

    def call(self, method, params):
        self.seq += 1
        request_id = self.seq
        self.send({'id': request_id, 'method': method, 'params': params})
        deadline = time.monotonic() + self.timeout
        while True:
            while b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                if not line.strip():
                    continue
                message = json.loads(line)
                if 'method' in message:
                    if 'id' in message:
                        raise ReviewError('Codex requested an unexpected interaction during the status check.')
                    continue
                if message.get('id') == request_id:
                    if 'error' in message:
                        raise ReviewError('Codex rejected the setup request. Check setup again before retrying.')
                    return message['result']
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise ReviewError('Codex setup request timed out. Check setup again before retrying.')
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise ReviewError('Codex stopped before confirming setup. Check setup again before retrying.')
            self.buffer += chunk

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()


def summarize_hooks(reply, plugin_root=None):
    entries = reply.get('data', [])
    if len(entries) != 1 or entries[0].get('errors'):
        raise ReviewError('Codex reported hook configuration errors. Review them in Codex before continuing.')
    root = Path(plugin_root).resolve() if plugin_root else None
    hooks = []
    for hook in entries[0].get('hooks', []):
        if hook.get('source') != 'plugin' or not (hook.get('pluginId') or '').startswith('entity@'):
            continue
        if root is not None:
            try:
                Path(hook['sourcePath']).resolve().relative_to(root)
            except (KeyError, ValueError):
                continue
        hooks.append({key: hook.get(key) for key in
                      ('key', 'pluginId', 'eventName', 'enabled', 'trustStatus', 'currentHash')})
    pending = [h for h in hooks if h['trustStatus'] not in ('trusted', 'managed')]
    disabled = [h for h in hooks if h['enabled'] is not True]
    state = ('not_found' if not hooks else 'review_required' if any(h['enabled'] is True for h in pending) else
             'disabled' if disabled else 'definitions_trusted')
    return {'state': state, 'trust_complete': bool(hooks) and not pending and not disabled,
            'hooks': hooks, 'review_count': len(pending), 'disabled_count': len(disabled),
            'warnings': entries[0].get('warnings', []),
            'runtime_verified': False}


@contextmanager
def _connection(*, executable=None, codex_home=None, cwd=None):
    home = Path(codex_home or os.environ.get('CODEX_HOME') or Path.home() / '.codex').absolute()
    directory = Path(cwd or Path.home()).absolute()
    client = _Client(find_codex(executable), home, directory)
    try:
        client.call('initialize', {'clientInfo': {'name': 'entity-setup', 'version': '1'},
                                   'capabilities': {'experimentalApi': True}})
        client.send({'method': 'initialized'})
        yield client
    finally:
        client.close()


def _tool_policy(config, plugin_id):
    server = config.get('plugins', {}).get(plugin_id, {}).get('mcp_servers', {}).get('brain', {})
    return {'plugin_id': plugin_id, 'mode': server.get('default_tools_approval_mode', 'auto'),
            'restricted_tools': [name for name, tool in server.get('tools', {}).items()
                                 if tool.get('enabled') is False or tool.get('approval_mode') in ('prompt', 'writes')],
            'server_enabled': server.get('enabled', True)}


def _status(client, plugin_root):
    status = summarize_hooks(client.call('hooks/list', {'cwds': [client.cwd]}), plugin_root)
    ids = {hook['pluginId'] for hook in status['hooks']}
    if len(ids) != 1:
        status['tool_approval'] = None
        return status
    config = client.call('config/read', {'cwd': client.cwd})['config']
    status['tool_approval'] = _tool_policy(config, ids.pop())
    return status


def setup_status(*, plugin_root=None, **host):
    with _connection(**host) as client:
        return _status(client, plugin_root)


def approve_tools(*, plugin_root, plugin_id, **host):
    """After explicit consent, save only this Entity server's default policy.

    Revalidate installation identity and use Codex's optimistic write version.
    Explicit tool restrictions and hook trust are never changed.
    """
    with _connection(**host) as client:
        status = _status(client, plugin_root)
        policy = status['tool_approval']
        if not policy or policy['plugin_id'] != plugin_id or not policy['server_enabled']:
            raise ReviewError('The Entity installation changed or its tools are disabled. Check setup again.')
        config = client.call('config/read', {'includeLayers': True, 'cwd': client.cwd})
        user = next((layer for layer in config['layers'] if layer['name']['type'] == 'user'
                     and not layer['name'].get('profile')), None)
        if user is None:
            raise ReviewError('Codex did not identify a writable user configuration.')
        client.call('config/batchWrite', {'edits': [{
            'keyPath': 'plugins.' + json.dumps(plugin_id) + '.mcp_servers.brain.default_tools_approval_mode',
            'value': 'approve', 'mergeStrategy': 'replace'}],
            'filePath': user['name']['file'], 'expectedVersion': user['version']})
        policy = _tool_policy(client.call('config/read', {'cwd': client.cwd})['config'], plugin_id)
        if policy['mode'] != 'approve':
            raise ReviewError('Another Codex policy overrides Entity tool approval. Check Codex permission settings.')
        return {'saved': True, **policy,
                'message': 'Entity-wide tool approval is saved. Individual restrictions remain. An existing Codex connection may need to reload before using the new setting.'}


def review_script(executable, codex_home, cwd):
    """A terminal receives the exact host configuration inspected by setup_status."""
    command = shlex.join([str(executable), '--no-alt-screen', '--cd', str(cwd)])
    return ('#!/bin/sh\n'
            'printf "%s\\n" "Entity automatic memory — Codex hook review" '
            '"Choose Review hooks in Codex and approve only the hooks you intend to run." '
            '"Codex may first ask you to confirm the working directory." '
            '"When finished, close this window and return to Entity setup to check again."\n'
            'export CODEX_HOME=' + shlex.quote(str(codex_home)) + '\n'
            'exec ' + command + '\n')


def open_review(*, executable=None, codex_home=None, cwd=None):
    """Launch native Codex after the caller obtains explicit user consent.

    macOS opens a .command document in Terminal. Other hosts get manual
    instructions. No input is sent to Codex, even if its startup UI changes.
    """
    if sys.platform != 'darwin':
        raise ReviewError('Automatic review launch is available on macOS. Open Codex CLI to review Entity hooks on this operating system.')
    binary = find_codex(executable)
    home = Path(codex_home or os.environ.get('CODEX_HOME') or Path.home() / '.codex').absolute()
    directory = Path(cwd or Path.home()).absolute()
    # Terminal reads this document after launch; the OS temp directory owns its
    # lifetime. It contains paths and no credentials.
    folder = Path(tempfile.mkdtemp(prefix='entity-hook-review-'))
    path = folder / 'Review Entity hooks.command'
    path.write_text(review_script(binary, home, directory))
    path.chmod(0o700)
    try:
        result = subprocess.run(['/usr/bin/open', '-a', 'Terminal', str(path)],
                                env={**os.environ, 'CODEX_HOME': str(home)},
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=8, check=False)
        if result.returncode:
            raise ReviewError('Could not open Codex hook review. Your hook permissions were not changed.')
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ReviewError('Could not open Codex hook review. Your hook permissions were not changed.') from error
    return {'state': 'review_launched', 'trust_granted': False,
            'message': 'Complete the review in Codex, then check Entity setup again. Opening the review does not grant permission.'}
