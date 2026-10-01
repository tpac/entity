"""Locks _parse_surfacer_json — the Haiku selection-JSON extractor.

Regression coverage for the three shapes Haiku returns (bare, fenced,
JSON+prose) PLUS the trailing-comma repair added 2026-07-18 (pool60 build:
'{"selected":[...], }' — complete selections were being discarded over a
lone trailing comma, one carrying a real pick). The repair is a LAST resort
(fires only after both standard parse paths fail) and must never regress the
clean paths.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales.s1.surface import _parse_surfacer_json as p


def test_bare_json():
    assert p('{"selected": [{"id": "ab"}]}')['selected'][0]['id'] == 'ab'


def test_fenced_json():
    assert p('```json\n{"selected": []}\n```') == {'selected': []}


def test_json_with_trailing_prose():
    # raw_decode consumes the first object, ignores the tail
    assert p('{"selected": []}\n\nHere is why I picked...') == {'selected': []}


def test_trailing_comma_empty_selection():
    # the exact pool60 payloads
    assert p('{"selected":[], }') == {'selected': []}


def test_trailing_comma_with_real_pick():
    r = p('{"selected":[{"id":"473f0e25", "mode":"background"}], }')
    assert r == {'selected': [{'id': '473f0e25', 'mode': 'background'}]}


def test_trailing_comma_nested_array_and_object():
    r = p('{"selected": [{"id": "x"},], "reason": "done",}')
    assert r['selected'] == [{'id': 'x'}]
    assert r['reason'] == 'done'


def test_no_json_returns_none():
    assert p('I think the answer is 42.') is None


def test_truncated_open_brace_returns_none():
    # constrained-decode whitespace spiral — a lone '{'
    assert p('{') is None


def test_empty_returns_none():
    assert p('') is None
    assert p(None) is None


def test_non_dict_top_level_returns_none():
    # a top-level array is not the selection contract
    assert p('["a", "b"]') is None
