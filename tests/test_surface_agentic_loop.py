"""Unit tests for the v5_agentic surface loop (_call_surface_agentic).

Covers the loop contract:
  - Prompt-cache breakpoints: BP1 on the system block (tools+system,
    1h — survives the final round's tool_choice flip), BP2 on the
    round-1 user content; tools byte-identical on EVERY round.
  - Final-round discipline (2026-07-11): the last round is sent with
    tool_choice='none', so max_rounds is the hard API-call cap — the
    forced-finalize third call is gone.
  - Admission-floor tripwire + result_ids/dropped_ids trace attribution.

All tests run against fakes — no brain, no network.
"""
import json
from types import SimpleNamespace

import pytest

from servers.scales.s1 import surface as surface_mod
from servers.scales.s1 import fetch_tools as fetch_tools_mod


SELECTION_JSON = '{"selected":[{"id":"aaaa1111","mode":"arc"}]}'


class FakeUsage:
    def __init__(self, input_tokens=10000, output_tokens=100,
                 cache_read=0, cache_creation=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_creation


def text_response(text=SELECTION_JSON, **usage_kw):
    return SimpleNamespace(
        stop_reason='end_turn',
        content=[SimpleNamespace(type='text', text=text)],
        usage=FakeUsage(**usage_kw))


def tool_use_response(tool='recall_topical', tool_input=None, **usage_kw):
    return SimpleNamespace(
        stop_reason='tool_use',
        content=[SimpleNamespace(
            type='tool_use', id='tu_1', name=tool,
            input=tool_input or {'query': 'x'})],
        usage=FakeUsage(**usage_kw))


class FakeClient:
    """Scripted client — pops responses in order, records every kwargs."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeBrain:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def _log_error(self, source, exc, context=''):
        self.errors.append((source, str(exc), context))

    def _log_warning(self, source, message, context=''):
        self.warnings.append((source, message, context))


def run_loop(client, brain, candidates, max_rounds=2):
    return surface_mod._call_surface_agentic(
        client, brain, candidates, 'SYSTEM', 'USER CONTENT', 600,
        'sess-test', 'claude-haiku-4-5', max_rounds=max_rounds)


class TestCachePrefix:
    def test_first_message_carries_cache_control(self):
        client = FakeClient([text_response()])
        raw, trace, tel = run_loop(client, FakeBrain(), [])
        first_msg = client.calls[0]['messages'][0]
        block = first_msg['content'][0]
        assert block['cache_control'] == {'type': 'ephemeral'}
        assert block['text'] == 'USER CONTENT'
        assert raw == SELECTION_JSON

    def test_system_carries_bp1_cache_control(self):
        # BP1 (tools+system tier, 1h): byte-identical across recalls AND
        # unaffected by the final round's tool_choice flip.
        client = FakeClient([text_response()])
        run_loop(client, FakeBrain(), [])
        system = client.calls[0]['system']
        assert system[0]['text'] == 'SYSTEM'
        assert system[0]['cache_control'] == {'type': 'ephemeral',
                                              'ttl': '1h'}

    def test_tools_present_on_every_round(self, monkeypatch):
        monkeypatch.setattr(
            fetch_tools_mod, 'execute_tool',
            lambda *a, **kw: {'results': [], 'latency_ms': 1})
        client = FakeClient([tool_use_response(), text_response()])
        run_loop(client, FakeBrain(), [])
        assert len(client.calls) == 2
        for call in client.calls:
            assert call['tools'] == fetch_tools_mod.TOOL_DEFINITIONS

    def test_cache_miss_on_round2_warns(self, monkeypatch):
        monkeypatch.setattr(
            fetch_tools_mod, 'execute_tool',
            lambda *a, **kw: {'results': [], 'latency_ms': 1})
        brain = FakeBrain()
        client = FakeClient([
            tool_use_response(input_tokens=20000),
            # round 2: big prompt, zero cache read → should warn
            text_response(input_tokens=22000, cache_read=0),
        ])
        run_loop(client, brain, [])
        assert any(w[0] == 'surface_cache_miss' for w in brain.warnings)

    def test_cache_hit_on_round2_quiet(self, monkeypatch):
        monkeypatch.setattr(
            fetch_tools_mod, 'execute_tool',
            lambda *a, **kw: {'results': [], 'latency_ms': 1})
        brain = FakeBrain()
        client = FakeClient([
            tool_use_response(input_tokens=20000, cache_creation=19000),
            text_response(input_tokens=2000, cache_read=19000),
        ])
        raw, trace, tel = run_loop(client, brain, [])
        assert not any(w[0] == 'surface_cache_miss' for w in brain.warnings)
        assert tel['cache_read_tokens'] == 19000


class TestFinalRoundDiscipline:
    """The final round is sent with tool_choice='none' (2026-07-11) —
    max_rounds is the hard API-call cap. Replaces the forced-finalize
    fallback (a third tools-stripped call that cost ~5.7s and breached
    the 20s hook budget on 2-tool-round recalls)."""

    def test_final_round_sends_tool_choice_none(self, monkeypatch):
        monkeypatch.setattr(
            fetch_tools_mod, 'execute_tool',
            lambda *a, **kw: {'results': [], 'latency_ms': 1})
        client = FakeClient([tool_use_response(), text_response()])
        raw, trace, tel = run_loop(client, FakeBrain(), [])
        assert 'tool_choice' not in client.calls[0]   # round 0: auto
        assert client.calls[1]['tool_choice'] == {'type': 'none'}
        assert raw == SELECTION_JSON
        assert tel['rounds'] == 2

    def test_max_rounds_is_the_api_call_cap(self):
        # Even if the model tool-uses on the final round (impossible via
        # the real API under tool_choice='none' — only a fake can), the
        # loop must NOT spend an extra call; it logs loudly and stops.
        brain = FakeBrain()
        client = FakeClient([tool_use_response(), text_response()])
        raw, trace, tel = run_loop(client, brain, [], max_rounds=1)
        assert len(client.calls) == 1
        assert client.calls[0]['tool_choice'] == {'type': 'none'}
        assert tel['rounds'] == 1
        assert any(w[0] == 'surface_final_round_tool_use'
                   for w in brain.warnings)

    def test_empty_tool_use_round_retries_without_history_append(self):
        # stop_reason='tool_use' with ZERO tool_use blocks (the May-2026
        # Haiku mode) must not append empty messages (API 400 next round).
        # The loop retries with untouched history; the retry is the final
        # round, so tool_choice='none' guarantees it finalizes.
        brain = FakeBrain()
        empty_tool_use = SimpleNamespace(
            stop_reason='tool_use', content=[], usage=FakeUsage())
        client = FakeClient([empty_tool_use, text_response()])
        raw, trace, tel = run_loop(client, brain, [])
        assert raw == SELECTION_JSON
        assert len(client.calls) == 2
        assert len(client.calls[1]['messages']) == 1  # history untouched
        assert client.calls[1]['tool_choice'] == {'type': 'none'}
        assert any(w[0] == 'surface_empty_tool_use' for w in brain.warnings)


class TestEffortAndThinking:
    """effort/thinking reach every round when set; unset, the request
    carries neither (Haiku 4.5 rejects effort)."""

    def test_unset_sends_neither(self, monkeypatch):
        monkeypatch.setattr(
            fetch_tools_mod, 'execute_tool',
            lambda *a, **kw: {'results': [], 'latency_ms': 1})
        client = FakeClient([tool_use_response(), text_response()])
        run_loop(client, FakeBrain(), [])
        for call in client.calls:
            assert 'thinking' not in call
            assert 'effort' not in call['output_config']

    def test_set_rides_every_round(self, monkeypatch):
        monkeypatch.setattr(
            fetch_tools_mod, 'execute_tool',
            lambda *a, **kw: {'results': [], 'latency_ms': 1})
        client = FakeClient([tool_use_response(), text_response()])
        surface_mod._call_surface_agentic(
            client, FakeBrain(), [], 'SYSTEM', 'USER CONTENT', 4096,
            'sess-test', 'claude-haiku-5-5',
            effort='low', thinking={'type': 'disabled'})
        assert len(client.calls) == 2
        for call in client.calls:
            assert call['output_config']['effort'] == 'low'
            assert call['output_config']['format']['type'] == 'json_schema'
            assert call['thinking'] == {'type': 'disabled'}

    def test_telemetry_records_the_effort_sent(self):
        _, _, tel = surface_mod._call_surface_agentic(
            FakeClient([text_response()]), FakeBrain(), [], 'SYSTEM',
            'USER CONTENT', 4096, 'sess-test', 'claude-haiku-5-5',
            effort='medium')
        assert tel['effort'] == 'medium'
        assert tel['model'] == 'claude-haiku-5-5'
        _, _, tel = run_loop(FakeClient([text_response()]), FakeBrain(), [])
        assert tel['effort'] == ''


class TestStopsWithoutSelection:
    """A max_tokens or refusal stop leaves no selection; each is logged by
    name instead of surfacing nothing silently."""

    def _stopped(self, stop_reason):
        return SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type='thinking', thinking='')],
            usage=FakeUsage(output_tokens=600))

    def test_truncation_is_logged(self):
        brain = FakeBrain()
        raw, trace, tel = run_loop(
            FakeClient([self._stopped('max_tokens')]), brain, [])
        assert raw == ''
        assert tel['truncated'] == 1
        assert [e[0] for e in brain.errors] == ['surface_truncation']

    def test_refusal_is_logged(self):
        brain = FakeBrain()
        raw, trace, tel = run_loop(
            FakeClient([self._stopped('refusal')]), brain, [])
        assert raw == ''
        assert [e[0] for e in brain.errors] == ['surface_refusal']

    def test_normal_stop_logs_nothing(self):
        brain = FakeBrain()
        run_loop(FakeClient([text_response()]), brain, [])
        assert brain.errors == []


class TestPerRoundTelemetry:
    """Each tool_trace record carries its API call's cost (2026-07-11):
    total_ms + the four USAGE_FIELDS, mirroring run_llm_loop's
    per_round_stats — so a slow/retried call is distinguishable from a
    verbose one per round, not just in the summed telemetry."""

    def test_single_round_records_cost(self):
        client = FakeClient([text_response(input_tokens=1500,
                                           output_tokens=250)])
        raw, trace, tel = run_loop(client, FakeBrain(), [])
        rec = trace[0]
        assert isinstance(rec['total_ms'], int) and rec['total_ms'] >= 0
        assert rec['input_tokens'] == 1500
        assert rec['output_tokens'] == 250
        assert rec['cache_read_tokens'] == 0
        assert rec['cache_creation_tokens'] == 0

    def test_each_round_carries_own_usage(self, monkeypatch):
        monkeypatch.setattr(
            fetch_tools_mod, 'execute_tool',
            lambda *a, **kw: {'results': [], 'latency_ms': 1})
        client = FakeClient([
            tool_use_response(output_tokens=150, cache_creation=12000),
            text_response(output_tokens=300, cache_read=12000),
        ])
        raw, trace, tel = run_loop(client, FakeBrain(), [])
        assert trace[0]['output_tokens'] == 150
        assert trace[0]['cache_creation_tokens'] == 12000
        assert trace[1]['output_tokens'] == 300
        assert trace[1]['cache_read_tokens'] == 12000


class TestFloorAndAttribution:
    def _run_with_fetched(self, monkeypatch, pool, fetched):
        monkeypatch.setattr(
            fetch_tools_mod, 'execute_tool',
            lambda *a, **kw: {'results': list(fetched), 'latency_ms': 1})
        # Fakes carry only id/score — skip the rich candidate renderer.
        # Signature mirrors production (the scope kwarg the loop threads).
        monkeypatch.setattr(
            fetch_tools_mod, 'format_tool_result_for_haiku',
            lambda result, scope=None:
                '%d results' % len(result.get('results') or []))
        brain = FakeBrain()
        client = FakeClient([tool_use_response(), text_response()])
        raw, trace, tel = run_loop(client, brain, pool)
        calls = [c for r in trace for c in r.get('tool_calls', [])]
        return brain, pool, calls

    def test_floor_drops_all_trips_warning_and_records_ids(self, monkeypatch):
        pool = [{'id': 'p%d' % i, 'score': s}
                for i, s in enumerate([0.5, 0.6, 0.7])]
        fetched = [{'id': 'f1f1f1f1', 'score': 0.1},
                   {'id': 'f2f2f2f2', 'score': 0.2}]
        brain, pool_after, calls = self._run_with_fetched(
            monkeypatch, pool, fetched)
        assert calls[0]['result_count'] == 0
        assert calls[0]['result_ids'] == []
        assert calls[0]['dropped_below_floor'] == 2
        assert calls[0]['dropped_ids'] == ['f1f1f1f1', 'f2f2f2f2']
        assert any(w[0] == 'surface_floor_dropped_all'
                   for w in brain.warnings)
        # Dropped candidates never join the selection pool.
        assert len(pool_after) == 3

    def test_admitted_results_join_pool_with_ids(self, monkeypatch):
        pool = [{'id': 'p%d' % i, 'score': s}
                for i, s in enumerate([0.5, 0.6, 0.7])]
        fetched = [{'id': 'f1f1f1f1', 'score': 0.9},
                   {'id': 'f2f2f2f2', 'score': 0.1}]
        brain, pool_after, calls = self._run_with_fetched(
            monkeypatch, pool, fetched)
        assert calls[0]['result_count'] == 1
        assert calls[0]['result_ids'] == ['f1f1f1f1']
        assert calls[0]['dropped_ids'] == ['f2f2f2f2']
        # Partial drop is normal — no tripwire.
        assert not any(w[0] == 'surface_floor_dropped_all'
                       for w in brain.warnings)
        assert len(pool_after) == 4


class TestUnparseableVsTruncated:
    """_call_surface logs an unparseable reply once, by its real cause: a
    reply cut off mid-JSON is the loop's surface_truncation, not also
    surface_haiku_unparseable."""

    class _Brain(FakeBrain):
        def get_interaction_effective(self, name):
            return {'template': 'SYSTEM', 'config': {'layout': 'xml_v13'},
                    'stamp': {}}

        def _ensure_anthropic_client(self):
            return None

        def get_config(self, key, default=None):
            return default

    def _run(self, monkeypatch, raw, truncated):
        monkeypatch.setattr(
            surface_mod, '_call_surface_agentic',
            lambda *a, **kw: (raw, [], {'truncated': truncated}))
        brain = self._Brain()
        surfaced, *_ = surface_mod._call_surface(
            brain, [], 'msg', [], 'sess-t', {})
        return surfaced, [e[0] for e in brain.errors]

    def test_cut_off_reply_is_not_also_unparseable(self, monkeypatch):
        surfaced, errors = self._run(monkeypatch, '{"selected":[{"id":', 1)
        assert surfaced == {'selected': []}
        assert 'surface_haiku_unparseable' not in errors

    def test_whole_garbled_reply_is_unparseable(self, monkeypatch):
        surfaced, errors = self._run(monkeypatch, 'no json here', 0)
        assert surfaced == {'selected': []}
        assert errors == ['surface_haiku_unparseable']
