"""Run the real launcher against a disposable runtime and a simulated clock.

The host's bootstrap budget must keep one MCP connection alive while a cold
runtime becomes ready, without extending Claude's default or leaking protocol
output. Downloads and brain startup are replaced only at their boundaries.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch(tmp_path):
    scripts = tmp_path / 'hooks/scripts'
    scripts.mkdir(parents=True)
    commands = tmp_path / 'commands'
    commands.mkdir()
    for name in ('mcp-launch.sh', 'runtime-state.sh'):
        shutil.copy2(ROOT / 'hooks/scripts' / name, scripts / name)

    def script(path, content):
        path.write_text('#!/bin/bash\n' + content)
        path.chmod(0o700)

    (tmp_path / 'clock').write_text('0')
    script(commands / 'date', '/bin/cat "$FIXTURE_ROOT/clock"\n')
    script(commands / 'sleep', '''n=$(( $(/bin/cat "$FIXTURE_ROOT/clock") + 1 ))
printf '%s' "$n" > "$FIXTURE_ROOT/clock"
if [ "$n" -ge "$READY_AT" ]; then
    . "$FIXTURE_ROOT/hooks/scripts/runtime-state.sh"
    brain_runtime_fingerprint "$FIXTURE_ROOT" > "$FIXTURE_ROOT/.runtime-ready"
fi
''')
    script(scripts / 'ensure-runtime.sh', 'exit 0\n')
    script(scripts / 'brain-env.sh', '''PLUGIN_DIR="$FIXTURE_ROOT"
brain_python_as() { printf '%s' "$FIXTURE_ROOT/proxy"; }
''')
    script(tmp_path / 'proxy', '''printf '%s\\n' '{"initialized":true}'
''')
    (tmp_path / 'uv.lock').write_text('# fixture lock\n')
    (tmp_path / 'venv/bin').mkdir(parents=True)
    script(tmp_path / 'venv/bin/python', 'exit 0\n')
    env = {**os.environ, 'FIXTURE_ROOT': str(tmp_path), 'READY_AT': '30',
           'PATH': str(commands) + ':/usr/bin:/bin'}
    env.pop('BRAIN_MCP_BOOTSTRAP_WAIT_S', None)

    def run(budget=None, ready_at=30, warm=False):
        settings = {**env, 'READY_AT': str(ready_at)}
        if budget is not None:
            settings['BRAIN_MCP_BOOTSTRAP_WAIT_S'] = budget
        if warm:
            subprocess.run(['/bin/bash', '-c', '. "$1"; brain_runtime_fingerprint "$2" > "$2/.runtime-ready"',
                            '_', str(scripts / 'runtime-state.sh'), str(tmp_path)], check=True)
        result = subprocess.run(['/bin/bash', str(scripts / 'mcp-launch.sh')],
                                env=settings, capture_output=True, text=True, timeout=10)
        return result, int((tmp_path / 'clock').read_text())
    return run


def test_codex_first_connection_survives_bootstrap_longer_than_claude_budget(launch):
    manifest = json.loads((ROOT / '.codex-plugin/plugin.json').read_text())
    server = manifest['mcpServers']['brain']
    budget = server['env']['BRAIN_MCP_BOOTSTRAP_WAIT_S']
    assert server['startup_timeout_sec'] >= int(budget) + 30
    result, elapsed = launch(budget)
    assert result.returncode == 0, result.stderr
    assert elapsed == 30
    assert json.loads(result.stdout) == {'initialized': True}
    assert 'runtime ready' in result.stderr


def test_claude_default_still_exits_at_25_seconds(launch):
    claude = json.loads((ROOT / '.mcp.json').read_text())
    assert 'BRAIN_MCP_BOOTSTRAP_WAIT_S' not in claude['brain'].get('env', {})
    result, elapsed = launch()
    assert result.returncode == 1
    assert elapsed == 25
    assert result.stdout == ''
    assert '.bootstrap.log' in result.stderr


def test_warm_start_skips_bootstrap_wait(launch):
    result, elapsed = launch('300', warm=True)
    assert result.returncode == 0, result.stderr
    assert elapsed == 0
    assert json.loads(result.stdout) == {'initialized': True}
    assert result.stderr == ''


def test_configured_deadline_is_bounded_and_keeps_stdout_clean(launch):
    result, elapsed = launch('3', ready_at=30)
    assert result.returncode == 1
    assert elapsed == 3
    assert result.stdout == ''
    assert '.bootstrap.log' in result.stderr


@pytest.mark.parametrize('budget', ['0', '-1', '08', '1+1', '10000', '$(false)'])
def test_invalid_budget_is_rejected_before_bootstrap(launch, budget):
    result, elapsed = launch(budget)
    assert result.returncode == 1
    assert elapsed == 0
    assert result.stdout == ''
    assert 'positive integer' in result.stderr
