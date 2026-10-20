"""Codex setup: inspect hooks, save consented tool policy.

No Entity imports, brain access, hook execution, or hook-trust writes. The MCP
caller owns consent before approve_tools.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time


class ReviewError(RuntimeError):
    pass


# Trust states under which Codex runs a hook; every other state awaits review.
TRUSTED = ('trusted', 'managed')


def awaiting_review(hooks):
    """Enabled hooks Codex will not run until the user approves them."""
    return [h for h in hooks if h['enabled'] is True and h['trustStatus'] not in TRUSTED]


# Desktop bundles that ship a codex CLI, in preference order.
APP_RESOURCES = (
    '/Applications/ChatGPT.app/Contents/Resources',
    '/Applications/Codex.app/Contents/Resources',
    str(Path.home() / 'Applications/ChatGPT.app/Contents/Resources'),
)


def bundled_codex_candidates(resources_dirs=None):
    """Where each desktop bundle keeps its codex CLI. ChatGPT.app 26.928 moved
    it under codex-cli/, whose codex-package.json names the entrypoint
    (bin/codex); earlier apps shipped it flat as Resources/codex."""
    for resources in resources_dirs or APP_RESOURCES:
        resources = Path(resources)
        try:
            entrypoint = json.loads((resources / 'codex-cli/codex-package.json').read_text())['entrypoint']
        except (OSError, ValueError, KeyError, TypeError):
            entrypoint = 'bin/codex'
        yield str(resources / 'codex-cli' / entrypoint)
        yield str(resources / 'codex')


def find_codex(executable=None):
    """Prefer the desktop's bundled runtime to a possibly different PATH CLI."""
    candidates = [executable] if executable else [*bundled_codex_candidates(), shutil.which('codex')]
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
                      ('key', 'pluginId', 'eventName', 'command', 'matcher', 'enabled',
                       'trustStatus', 'currentHash')})
    pending = [h for h in hooks if h['trustStatus'] not in TRUSTED]
    disabled = [h for h in hooks if h['enabled'] is not True]
    state = ('not_found' if not hooks else 'review_required' if awaiting_review(hooks) else
             'disabled' if disabled else 'definitions_trusted')
    return {'state': state, 'trust_complete': bool(hooks) and not pending and not disabled,
            'hooks': hooks, 'review_count': len(pending), 'disabled_count': len(disabled),
            'warnings': entries[0].get('warnings', []),
            'runtime_verified': False}


@contextmanager
def _connection(*, executable=None, codex_home=None, cwd=None, timeout=12):
    home = Path(codex_home or os.environ.get('CODEX_HOME') or Path.home() / '.codex').absolute()
    directory = Path(cwd or Path.home()).absolute()
    client = _Client(find_codex(executable), home, directory, timeout)
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


if __name__ == '__main__':
    # The one owner of the codex lookup: scripts/codex-install.sh asks here.
    # Usage: codex_onboarding.py [--codex /path]  → prints the executable.
    try:
        print(find_codex(sys.argv[2] if sys.argv[1:2] == ['--codex'] else None))
    except ReviewError as error:
        sys.exit(str(error))
