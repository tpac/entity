"""The runtime's external versions live in one place, and they hold.

pyproject.toml declares the dependencies and the embedder pin; uv.lock records
the exact set every install syncs to (hooks/scripts/ensure-runtime.sh). These
pin:
  - the lock is current with pyproject. Installs sync with --frozen, so a
    dependency edit without `uv lock` would ship the old set silently;
  - the embedder pin has one home, with no second copy to fall out of step;
  - the locked stack embeds the reference texts exactly as the stack that wrote
    a live brain's stored vectors did. A library bump that moves vectors
    (onnxruntime 1.30 did: cosine 0.963 worst) splits new queries from every
    stored vector, so it has to arrive as a deliberate re-embed, not a lock bump;
  - the readiness states in hooks/scripts/runtime-state.sh, and that a failed
    re-sync keeps the runtime that already served (ensure-runtime.sh).
"""
import base64
import json
import os
import platform
import subprocess
import sys
import time
import tomllib

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from servers import embedder
from servers.brain import Brain
from tests.isolated_brain import _SHARED_MODEL_CACHE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(ROOT, 'tests', 'embedding_reference.json')


def _toml(name):
    with open(os.path.join(ROOT, name), 'rb') as f:
        return tomllib.load(f)


def _as_set(reqs):
    return {(r.name.lower(), SpecifierSet(str(r.specifier)))
            for r in (Requirement(s) for s in reqs)}


def _locked_set(entries):
    return {(e['name'].lower(), SpecifierSet(e.get('specifier', '')))
            for e in entries}


class TestLockIsCurrent:
    def test_lock_records_what_pyproject_declares(self):
        project = _toml('pyproject.toml')
        lock = _toml('uv.lock')
        root = next(p for p in lock['package']
                    if p['name'] == project['project']['name'])
        meta = root['metadata']
        assert _locked_set(meta['requires-dist']) == \
            _as_set(project['project']['dependencies']), \
            'pyproject.toml changed without `uv lock`'
        assert _locked_set(meta.get('requires-dev', {}).get('dev', [])) == \
            _as_set(project['dependency-groups']['dev']), \
            'the dev group changed without `uv lock`'

    def test_runtime_python_satisfies_requires_python(self):
        with open(os.path.join(ROOT, '.python-version')) as f:
            runtime = f.read().strip()
        requires = SpecifierSet(_toml('pyproject.toml')['project']['requires-python'])
        assert requires.contains(runtime + '.0')


class TestEmbedderPinHasOneHome:
    # node_enrichments.model's SQL column default names the model: DDL cannot
    # import the pin, and nothing relies on it — VectorDAL.store requires the
    # model from every caller.
    MODEL_NAME_ALSO_IN = {os.path.join('servers', 'schema.py')}
    SCANNED = ('servers', 'hooks', 'dashboard', 'skills',
               '.claude-plugin', '.codex-plugin')

    def _runtime_files(self):
        for top in self.SCANNED:
            for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, top)):
                dirnames[:] = [d for d in dirnames if d != '__pycache__']
                for name in filenames:
                    if name.endswith(('.py', '.sh', '.json', '.md')) or '.' not in name:
                        path = os.path.join(dirpath, name)
                        yield os.path.relpath(path, ROOT), path

    def test_brain_defaults_derive_from_the_pin(self):
        for key, value in embedder.EMBEDDER_PIN.items():
            assert Brain._EMBEDDER_DEFAULTS[key] == value

    def test_no_second_copy_in_runtime_files(self):
        pin = embedder.EMBEDDER_PIN
        strays = []
        for rel, path in self._runtime_files():
            with open(path, encoding='utf-8', errors='replace') as f:
                text = f.read()
            for key in ('pinned_revision', 'pinned_onnx_sha256'):
                if pin[key] in text:
                    strays.append('%s carries %s' % (rel, key))
            if pin['model_name'] in text and rel not in self.MODEL_NAME_ALSO_IN:
                strays.append('%s carries model_name' % rel)
        assert not strays, ('the embedder pin lives in pyproject.toml '
                            '[tool.brain.embedder] only: %s' % strays)


class TestLockedStackEmbedsLikeTheReference:
    """The live brain's stored vectors came from the reference stack; the
    locked stack must reproduce them. Every allowed version measured
    bit-identical (1.000000), so the bar sits just under exact."""

    MIN_COSINE = 0.9999

    def test_reference_texts_embed_unchanged(self):
        if not os.path.exists(REFERENCE):
            pytest.skip('reference vectors are repo-only: data under tests/ does '
                        'not ship in the public tree (D-8)')
        with open(REFERENCE) as f:
            ref = json.load(f)
        if list(ref['platform']) != [platform.system(), platform.machine()]:
            pytest.skip("reference captured on %s; ONNX Runtime's int8 kernels "
                        'differ by CPU architecture, so a difference here would '
                        'not be a lock signal' % ' '.join(ref['platform']))
        assert ref['model_name'] == embedder.EMBEDDER_PIN['model_name']
        assert ref['pinned_revision'] == embedder.EMBEDDER_PIN['pinned_revision']
        embedder.load_model({**Brain._EMBEDDER_DEFAULTS,
                             'cache_dir': _SHARED_MODEL_CACHE})
        assert embedder.is_ready(), embedder.stats.get('load_error')

        worst = (float('inf'), '')
        for kind in ('document', 'query'):
            fresh = embedder.embed_batch(ref['texts'], kind)
            for i, (want, got) in enumerate(zip(ref[kind], fresh)):
                c = embedder.cosine_similarity(base64.b64decode(want), got)
                worst = min(worst, (c, '%s #%d' % (kind, i)))
        assert worst[0] >= self.MIN_COSINE, (
            'the locked stack moved the vectors (worst cosine %.6f at %s vs the '
            'reference stack %s) — stored vectors would split from new queries; '
            'hold the library back or migrate deliberately (re-embed)'
            % (worst[0], worst[1], ref['stack']))


# ─── The runtime's readiness states and re-sync fallback ─────────────────

SCRIPTS = os.path.join(ROOT, 'hooks', 'scripts')


def _state(tree, fn):
    return subprocess.run(
        ['/bin/bash', '-c', '. "$1"; %s "$2"' % fn, '_',
         os.path.join(SCRIPTS, 'runtime-state.sh'), str(tree)]).returncode == 0


def _fingerprint(tree):
    return subprocess.run(
        ['/bin/bash', '-c', '. "$1"; brain_runtime_fingerprint "$2"', '_',
         os.path.join(SCRIPTS, 'runtime-state.sh'), str(tree)],
        capture_output=True, text=True, check=True).stdout


def _synced_tree(tmp_path):
    (tmp_path / 'uv.lock').write_text('lock v1\n')
    (tmp_path / '.python-version').write_text('3.11\n')
    (tmp_path / 'venv' / 'bin').mkdir(parents=True)
    python = tmp_path / 'venv' / 'bin' / 'python'
    python.write_text('#!/bin/sh\n')
    python.chmod(0o755)
    (tmp_path / '.runtime-ready').write_text(_fingerprint(tmp_path))
    return tmp_path


def _rewritten_after_sync(tree, path):
    """Order the mtimes as a git checkout after the last sync leaves them:
    sentinel first, `path` rewritten later, both in the past."""
    t = time.time()
    os.utime(tree / '.runtime-ready', (t - 10, t - 10))
    os.utime(path, (t - 5, t - 5))


class TestRuntimeReadiness:
    def test_synced_runtime_is_ready_and_installed(self, tmp_path):
        tree = _synced_tree(tmp_path)
        assert _state(tree, 'brain_runtime_ready')
        assert _state(tree, 'brain_runtime_installed')

    def test_changed_lock_is_installed_but_not_ready(self, tmp_path):
        tree = _synced_tree(tmp_path)
        (tree / 'uv.lock').write_text('lock v2\n')
        _rewritten_after_sync(tree, tree / 'uv.lock')
        assert not _state(tree, 'brain_runtime_ready')
        assert _state(tree, 'brain_runtime_installed')

    def test_changed_python_version_is_not_ready(self, tmp_path):
        tree = _synced_tree(tmp_path)
        (tree / '.python-version').write_text('3.12\n')
        _rewritten_after_sync(tree, tree / '.python-version')
        assert not _state(tree, 'brain_runtime_ready')

    def test_rewritten_identical_lock_stays_ready_and_refreshes(self, tmp_path):
        tree = _synced_tree(tmp_path)
        _rewritten_after_sync(tree, tree / 'uv.lock')  # same bytes, newer mtime
        assert _state(tree, 'brain_runtime_ready')
        assert not (tree / 'uv.lock').stat().st_mtime > \
            (tree / '.runtime-ready').stat().st_mtime  # sentinel touched

    def test_pre_lock_empty_sentinel_is_installed_but_not_ready(self, tmp_path):
        tree = _synced_tree(tmp_path)
        (tree / '.runtime-ready').write_text('')
        assert not _state(tree, 'brain_runtime_ready')
        assert _state(tree, 'brain_runtime_installed')

    def test_missing_lock_is_not_ready(self, tmp_path):
        tree = _synced_tree(tmp_path)
        (tree / 'uv.lock').unlink()
        assert not _state(tree, 'brain_runtime_ready')

    def test_mid_first_bootstrap_is_not_installed(self, tmp_path):
        tree = _synced_tree(tmp_path)
        (tree / '.runtime-ready').unlink()             # venv made, never synced
        assert not _state(tree, 'brain_runtime_installed')


class TestResyncFailureKeepsRuntime:
    """A re-sync that fails (offline, index outage) must not disable a runtime
    that already served; a first install that fails still fails."""

    def _plugin(self, tmp_path, installed):
        scripts = tmp_path / 'hooks' / 'scripts'
        scripts.mkdir(parents=True)
        for name in ('ensure-runtime.sh', 'runtime-state.sh'):
            (scripts / name).write_text(open(os.path.join(SCRIPTS, name)).read())
            (scripts / name).chmod(0o755)
        (tmp_path / 'uv.lock').write_text('lock v2\n')
        (tmp_path / '.python-version').write_text('3.11\n')
        (tmp_path / 'py').mkdir()
        (tmp_path / 'py' / 'marker').write_text('')
        (tmp_path / 'venv' / 'bin').mkdir(parents=True)
        python = tmp_path / 'venv' / 'bin' / 'python'
        python.write_text('#!/bin/sh\n')
        python.chmod(0o755)
        if installed:
            (tmp_path / '.runtime-ready').write_text('synced to lock v1\n')
            _rewritten_after_sync(tmp_path, tmp_path / 'uv.lock')  # lock v2 arrived later
        # A uv at the pinned version whose sync fails, as offline would.
        version = subprocess.run(
            ['/bin/bash', '-c', 'grep -o "BRAIN_UV_VERSION:-[0-9.]*" "$1" | cut -d- -f2',
             '_', str(scripts / 'ensure-runtime.sh')],
            capture_output=True, text=True, check=True).stdout.strip()
        (tmp_path / 'bin').mkdir()
        uv = tmp_path / 'bin' / 'uv'
        uv.write_text('#!/bin/sh\n[ "$1" = --version ] && echo "uv %s" && exit 0\n'
                      'echo "network unreachable" >&2; exit 2\n' % version)
        uv.chmod(0o755)
        return scripts / 'ensure-runtime.sh'

    def test_failed_resync_serves_the_previous_sync(self, tmp_path):
        script = self._plugin(tmp_path, installed=True)
        r = subprocess.run(['/bin/bash', str(script)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert 'keeping the previously synced libraries' in r.stderr
        assert (tmp_path / '.runtime-ready').read_text() == 'synced to lock v1\n'
        assert (tmp_path / '.bootstrap-failed').exists()
        again = subprocess.run(['/bin/bash', str(script)], capture_output=True, text=True)
        assert again.returncode == 0, again.stderr     # cooldown: serve, no retry

    def test_failed_first_install_still_fails(self, tmp_path):
        script = self._plugin(tmp_path, installed=False)
        r = subprocess.run(['/bin/bash', str(script)], capture_output=True, text=True)
        assert r.returncode != 0
        assert 'dep sync failed' in r.stderr


def _plugin_with_fake_uv(tmp_path, sync_ok=True):
    """A plugin tree whose uv is a logging fake at the pinned version: `sync`
    creates $UV_PROJECT_ENVIRONMENT with a python that really runs the swap
    (os.replace) and fakes everything else (model pre-fetch, version print)."""
    scripts = tmp_path / 'hooks' / 'scripts'
    scripts.mkdir(parents=True)
    for name in ('ensure-runtime.sh', 'runtime-state.sh'):
        (scripts / name).write_text(open(os.path.join(SCRIPTS, name)).read())
        (scripts / name).chmod(0o755)
    (tmp_path / 'uv.lock').write_text('lock v2\n')
    (tmp_path / '.python-version').write_text('3.11.16\n')
    version = subprocess.run(
        ['/bin/bash', '-c', 'grep -o "BRAIN_UV_VERSION:-[0-9.]*" "$1" | cut -d- -f2',
         '_', str(scripts / 'ensure-runtime.sh')],
        capture_output=True, text=True, check=True).stdout.strip()
    python = ('#!/bin/sh\ncase "$2" in *os.replace*) exec "%s" "$@" ;; esac\n'
              'echo 3.11.16\n' % sys.executable)
    (tmp_path / 'bin').mkdir()
    uv = tmp_path / 'bin' / 'uv'
    uv.write_text(
        '#!/bin/sh\n'
        '[ "$1" = --version ] && echo "uv %s" && exit 0\n'
        'echo "${UV_PYTHON_PREFERENCE:-unset} ${UV_PYTHON_DOWNLOADS:-unset}'
        ' ${UV_PYTHON_INSTALL_DIR:-unset} $*" >> "%s"\n'
        '%s'
        'mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"\n'
        "cat > \"$UV_PROJECT_ENVIRONMENT/bin/python\" <<'PY'\n%sPY\n"
        'chmod +x "$UV_PROJECT_ENVIRONMENT/bin/python"\n'
        'exit 0\n' % (version, tmp_path / 'uv-calls.log',
                       '' if sync_ok else 'echo "network unreachable" >&2; exit 2\n',
                       python))
    uv.chmod(0o755)
    return scripts / 'ensure-runtime.sh'


def _installed_build(tree, name, lock='lock v1\n'):
    """A venv build that served `lock`: venvs/<name> with a python, venv -> it,
    and a sentinel older than the current uv.lock."""
    (tree / 'venvs' / name / 'bin').mkdir(parents=True)
    (tree / 'venvs' / name / 'bin' / 'python').write_text('#!/bin/sh\n')
    (tree / 'venvs' / name / 'bin' / 'python').chmod(0o755)
    if not (tree / 'venv').is_symlink():
        os.symlink('venvs/' + name, tree / 'venv')
    (tree / '.runtime-ready').write_text('synced to %s' % lock)
    _rewritten_after_sync(tree, tree / 'uv.lock')


class TestInterpreterComesFromThePlugin:
    """Every install runs one exact interpreter, and only from its own py/.

    '3.11' let each fresh install take whatever patch was newest that day
    (Codex got 3.11.16 while the daemon ran 3.11.11). `uv sync` downloads the
    pinned patch into py/ itself when it builds a venv, but only when told
    only-managed (else it takes any python3.x on PATH) and downloads-automatic
    (else a user's uv.toml can forbid it). No `uv python install`: since uv 0.8
    it also links a global python3.x into ~/.local/bin."""

    def test_python_pin_is_an_exact_patch(self):
        with open(os.path.join(ROOT, '.python-version')) as f:
            pin = f.read().strip()
        parts = pin.split('.')
        assert len(parts) == 3 and all(p.isdigit() for p in parts), pin

    def test_first_install_lets_uv_fetch_python_only_into_py(self, tmp_path):
        script = _plugin_with_fake_uv(tmp_path)
        r = subprocess.run(['/bin/bash', str(script)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        calls = [c.split() for c in (tmp_path / 'uv-calls.log').read_text().splitlines()]
        assert [c[3] for c in calls] == ['sync'], calls   # no `uv python install`
        assert calls[0][:3] == ['only-managed', 'automatic', str(tmp_path / 'py')], calls


class TestVenvBuildsBesideAndSwaps:
    """A plugin update's shape for the venv: each sync builds a complete venv
    in venvs/ and `venv` is repointed in one rename when it is done. An in-place
    sync across a Python change deleted the whole venv first; a session booting
    in that window read 'not installed' (upgrade smoke, 2026-09-23)."""

    def _build(self, tree):
        return (tree / 'venv').resolve().name

    def test_first_install_points_venv_at_a_complete_build(self, tmp_path):
        script = _plugin_with_fake_uv(tmp_path)
        r = subprocess.run(['/bin/bash', str(script)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert (tmp_path / 'venv').is_symlink()
        assert os.readlink(tmp_path / 'venv') == 'venvs/' + self._build(tmp_path)
        assert _state(tmp_path, 'brain_runtime_ready')
        assert not list(tmp_path.glob('.venv-swap.*'))

    def test_a_pre_swap_venv_directory_moves_aside_and_stays_as_previous(self, tmp_path):
        script = _plugin_with_fake_uv(tmp_path)
        (tmp_path / 'venv' / 'bin').mkdir(parents=True)          # the old layout
        (tmp_path / 'venv' / 'bin' / 'python').write_text('#!/bin/sh\n')
        (tmp_path / 'venv' / 'bin' / 'python').chmod(0o755)
        (tmp_path / '.runtime-ready').write_text('synced to lock v1\n')
        _rewritten_after_sync(tmp_path, tmp_path / 'uv.lock')
        r = subprocess.run(['/bin/bash', str(script)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert (tmp_path / 'venv').is_symlink()
        kept = sorted(p.name for p in (tmp_path / 'venvs').iterdir())
        assert len(kept) == 2 and any(k.startswith('legacy-') for k in kept), kept
        assert _state(tmp_path, 'brain_runtime_ready')

    def test_failed_build_leaves_the_serving_venv_untouched(self, tmp_path):
        script = _plugin_with_fake_uv(tmp_path, sync_ok=False)
        _installed_build(tmp_path, 'old')
        r = subprocess.run(['/bin/bash', str(script)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert 'keeping the previously synced libraries' in r.stderr
        assert os.readlink(tmp_path / 'venv') == 'venvs/old'
        assert [p.name for p in (tmp_path / 'venvs').iterdir()] == ['old'], \
            'the failed build must not be left behind'

    def test_success_keeps_the_previous_build_and_prunes_older(self, tmp_path):
        script = _plugin_with_fake_uv(tmp_path)
        _installed_build(tmp_path, 'older')
        _installed_build(tmp_path, 'previous')
        os.remove(tmp_path / 'venv')
        os.symlink('venvs/previous', tmp_path / 'venv')
        r = subprocess.run(['/bin/bash', str(script)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        kept = {p.name for p in (tmp_path / 'venvs').iterdir()}
        assert kept == {'previous', self._build(tmp_path)}, kept
