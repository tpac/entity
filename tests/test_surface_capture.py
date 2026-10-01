"""Tests for the surface capture harness (surface_capture.py) — the
production corpus source for the prompt A/B replay bench.

The load-bearing test is the byte-truth fidelity proof: re-rendering from
ONLY what the capture stores must byte-equal the rendered prompt production
sent. Every input field is non-default here, so if begin() ever misses a
field that build_surface_prompt reads, this test fails instead of the
corpus silently diverging from production.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales.s1 import surface_capture
from servers.scales.s1.surface_contract import (
    build_surface_prompt, SURFACE_LAYOUT)


class FakeBrain:
    def __init__(self, capture_enabled=True):
        self.errors = []
        self._config = {'surface_capture.enabled': capture_enabled}

    def get_config(self, key, default_val=None):
        return self._config.get(key, default_val)

    def _log_error(self, source, exc, context=''):
        self.errors.append((source, str(exc), context))


# created_at present so the render includes relative age strings — the
# time-dependent part of the prompt the fidelity contract cares about.
CANDS = [
    {'id': 'a' * 32, 'title': 'Alpha decision', 'score': 0.91,
     'type': 'decision', 'content': 'We chose alpha over beta.',
     'situation': 'When picking greek letters',
     'created_at': '2026-07-01T10:00:00+00:00'},
    {'id': 'b' * 32, 'title': 'Beta lesson', 'score': 0.55,
     'type': 'lesson', 'content': 'Beta needs a second reviewer.',
     'situation': 'When reviewing beta',
     'created_at': '2026-06-15T10:00:00+00:00'},
]

INPUT_KW = dict(
    user_message='what did we decide about alpha?',
    recent_messages=[{'role': 'user', 'content': 'earlier turn'},
                     {'role': 'assistant', 'content': 'earlier reply'}],
    retrieval_stats={'n_results': 2},
    frame='## Current focus\nGreek letters',
)


def _begin(brain, user_content='RENDERED', layout='xml_v13', **over):
    kw = dict(
        candidates_data=CANDS, layout=layout,
        surface_instructions='SYSTEM TEXT',
        interaction_stamp={'fingerprint': 'abcdef123456',
                           'source': 'override', 'version': 14, 'id': 124},
        user_content=user_content, max_tokens=600,
        variant='v5_agentic', model='claude-haiku-4-5',
        session_id='sess-cap', **INPUT_KW)
    kw.update(over)
    return surface_capture.begin(brain, **kw)


class TestCaptureDir:
    def test_off_unless_the_brain_opts_in(self, monkeypatch, tmp_path):
        monkeypatch.setenv('BRAIN_SURFACE_CAPTURE_DIR', str(tmp_path))
        monkeypatch.delenv('BRAIN_SURFACE_CAPTURE', raising=False)
        assert surface_capture.capture_dir(FakeBrain(capture_enabled=False)) is None
        assert surface_capture.capture_dir() is None

    def test_kill_switch_beats_the_opt_in(self, monkeypatch, tmp_path):
        monkeypatch.setenv('BRAIN_SURFACE_CAPTURE_DIR', str(tmp_path))
        monkeypatch.setenv('BRAIN_SURFACE_CAPTURE', 'off')
        assert surface_capture.capture_dir(FakeBrain()) is None

    def test_explicit_dir_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv('BRAIN_SURFACE_CAPTURE_DIR', str(tmp_path))
        monkeypatch.delenv('BRAIN_SURFACE_CAPTURE', raising=False)
        assert surface_capture.capture_dir(FakeBrain()) == str(tmp_path)

    def test_db_dir_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv('BRAIN_SURFACE_CAPTURE_DIR', raising=False)
        monkeypatch.delenv('BRAIN_SURFACE_CAPTURE', raising=False)
        monkeypatch.setenv('BRAIN_DB_DIR', str(tmp_path))
        assert surface_capture.capture_dir(FakeBrain()) == \
            os.path.join(str(tmp_path), 'surface_captures')

    def test_disabled_when_no_location(self, monkeypatch):
        monkeypatch.delenv('BRAIN_SURFACE_CAPTURE_DIR', raising=False)
        monkeypatch.delenv('BRAIN_SURFACE_CAPTURE', raising=False)
        monkeypatch.delenv('BRAIN_DB_DIR', raising=False)
        assert surface_capture.capture_dir(FakeBrain()) is None


class TestCaptureLifecycle:
    def _enable(self, monkeypatch, tmp_path):
        monkeypatch.setenv('BRAIN_SURFACE_CAPTURE_DIR', str(tmp_path))
        monkeypatch.delenv('BRAIN_SURFACE_CAPTURE', raising=False)

    def test_full_lifecycle_writes_self_contained_file(
            self, monkeypatch, tmp_path):
        self._enable(monkeypatch, tmp_path)
        brain = FakeBrain()
        cap = _begin(brain)
        assert cap is not None
        surface_capture.record_rounds(
            cap, messages=[{'role': 'user', 'content': 'RENDERED'}],
            raw_final='{"selected":[]}',
            tool_trace=[{'round': 0, 'stop_reason': 'end_turn'}])
        cap['output'] = {'raw': '{"selected":[]}'}
        path = surface_capture.finish(
            brain, cap, recall_ref='r-123', surfaced={'selected': []},
            resolved_mode={'a' * 32: 'arc'}, selection_reason='',
            telemetry={'rounds': 1, 'output_tokens': 90})
        assert path and os.path.exists(path)
        data = json.loads(open(path).read())
        assert data['v'] == surface_capture.CAPTURE_VERSION
        assert data['recall_ref'] == 'r-123'
        assert data['stamps']['interaction_version'] == 14
        assert data['stamps']['layout'] == 'xml_v13'
        assert data['stamps']['schema_sha']
        assert data['inputs']['candidates_pre_tools'][0]['id'] == 'a' * 32
        assert data['rendered']['user_content'] == 'RENDERED'
        assert data['rendered']['system'] == 'SYSTEM TEXT'
        assert data['rounds']['tool_trace'][0]['round'] == 0
        assert data['output']['raw'] == '{"selected":[]}'
        assert data['output']['resolved_mode'] == {'a' * 32: 'arc'}
        assert brain.errors == []

    def test_candidates_snapshot_is_a_copy(self, monkeypatch, tmp_path):
        # The agentic loop appends tool-fetched candidates in place AFTER
        # begin(); the capture must hold the round-1 pool.
        self._enable(monkeypatch, tmp_path)
        cands = [dict(c) for c in CANDS]
        cap = _begin(FakeBrain(), candidates_data=cands)
        cands.append({'id': 'd' * 32, 'title': 'tool-fetched'})
        cands[0]['title'] = 'mutated'
        snap = cap['inputs']['candidates_pre_tools']
        assert len(snap) == 2
        assert snap[0]['title'] == 'Alpha decision'

    def test_begin_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.setenv('BRAIN_SURFACE_CAPTURE', 'off')
        assert _begin(FakeBrain()) is None
        # And the None flows through the pipeline as a no-op.
        surface_capture.record_rounds(None, messages=[], raw_final='',
                                      tool_trace=[])
        assert surface_capture.finish(
            FakeBrain(), None, recall_ref='r', surfaced={},
            resolved_mode={}, selection_reason='', telemetry={}) is None

    def test_finish_never_raises_on_unwritable_dir(
            self, monkeypatch, tmp_path):
        blocked = tmp_path / 'blocked'
        blocked.write_text('a file, not a dir')  # makedirs will fail
        monkeypatch.setenv('BRAIN_SURFACE_CAPTURE_DIR',
                           str(blocked / 'sub'))
        monkeypatch.delenv('BRAIN_SURFACE_CAPTURE', raising=False)
        brain = FakeBrain()
        cap = _begin(brain)
        result = surface_capture.finish(
            brain, cap, recall_ref='r', surfaced={}, resolved_mode={},
            selection_reason='', telemetry={})
        assert result is None
        assert any(src == 'surface_capture' for src, _, _ in brain.errors)

    def test_oversized_payload_skipped_loudly(self, monkeypatch, tmp_path):
        self._enable(monkeypatch, tmp_path)
        brain = FakeBrain()
        cap = _begin(brain)
        cap['rounds'] = {'messages': ['x' * 2_100_000]}
        result = surface_capture.finish(
            brain, cap, recall_ref='r', surfaced={}, resolved_mode={},
            selection_reason='', telemetry={})
        assert result is None
        assert any(src == 'surface_capture' for src, _, _ in brain.errors)


class TestByteTruthFidelity:
    """Proof #1: re-rendering from ONLY the capture's structured inputs
    must byte-equal the rendered prompt production sent. If begin() ever
    misses a field build_surface_prompt reads, this fails."""

    def test_capture_inputs_rerender_byte_identical(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv('BRAIN_SURFACE_CAPTURE_DIR', str(tmp_path))
        monkeypatch.delenv('BRAIN_SURFACE_CAPTURE', raising=False)

        # Production-side render, every field non-default — including the
        # presentation-shuffle seed (§20.12 A2) and the session scope
        # (differential exposure), which production always passes and the
        # capture must round-trip for byte fidelity.
        seed = 0x5eed
        scope = {'project': 'brain', 'counterpart': 'Ada'}
        layout = SURFACE_LAYOUT
        user_content, max_tokens = build_surface_prompt(
            CANDS, INPUT_KW['user_message'],
            recent_messages=INPUT_KW['recent_messages'],
            retrieval_stats=INPUT_KW['retrieval_stats'],
            frame=INPUT_KW['frame'],
            shuffle_seed=seed, scope=scope)

        cap = _begin(FakeBrain(), user_content=user_content, layout=layout,
                     shuffle_seed=seed, scope=scope)

        # Replay-side re-render, from the capture alone.
        i = cap['inputs']
        rerendered, _ = build_surface_prompt(
            i['candidates_pre_tools'], i['user_message'],
            recent_messages=i['recent_messages'],
            retrieval_stats=i['retrieval_stats'],
            frame=i['frame'],
            shuffle_seed=i['shuffle_seed'],
            scope=i['scope'])

        assert rerendered == cap['rendered']['user_content']

    def test_pre_shuffle_capture_rerenders_without_seed(self):
        """Old captures have no shuffle_seed key — replay reads it as None
        and the re-render must equal the unshuffled production render."""
        user_content, _ = build_surface_prompt(
            CANDS, INPUT_KW['user_message'])
        rerendered, _ = build_surface_prompt(
            CANDS, INPUT_KW['user_message'], shuffle_seed=None)
        assert rerendered == user_content

    def test_render_is_time_dependent_so_replay_must_pin_clock(
            self, monkeypatch):
        """Executable form of the time-pinning contract (module docstring):
        candidate age strings come from wall-clock brain_now(), so the same
        capture re-rendered at a different time produces different bytes.
        The replay bench MUST pin brain_now to the capture's ts before the
        byte-compare. If this test ever fails, rendering became
        time-independent — delete this test AND the pinning requirement."""
        import servers.clock as clock_mod
        from datetime import datetime, timezone as tzmod

        r1, _ = build_surface_prompt(
            CANDS, INPUT_KW['user_message'])
        monkeypatch.setattr(
            clock_mod, 'brain_now',
            lambda brain=None, tz=None: datetime(
                2027, 3, 1, tzinfo=tzmod.utc))
        r2, _ = build_surface_prompt(
            CANDS, INPUT_KW['user_message'])
        assert r1 != r2
