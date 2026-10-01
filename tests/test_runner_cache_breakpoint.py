"""The rolling cache breakpoint's migration contract.

The loop appends assistant turns and tool results after the assembled prompt,
outside all three static breakpoints — so every later round re-sent them at full
input price (657K tokens billed 4× on an external install, 2026-08-14). A fourth
breakpoint rides the newest block and must MIGRATE: the API caps breakpoints at
4, and the static three are permanent, so an accumulating marker would 400 the
run on round 2. These pin migration, placement and the static three's survival.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales.runner import _roll_cache_breakpoint


def _fresh_messages():
    """The shape run_llm_loop builds: [0] carries the static user breakpoints."""
    return [{"role": "user", "content": [
        {"type": "text", "text": "preamble",
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {"type": "text", "text": "body",
         "cache_control": {"type": "ephemeral", "ttl": "5m"}},
    ]}]


def _marked(messages):
    return [(i, j) for i, m in enumerate(messages)
            for j, b in enumerate(m['content']) if 'cache_control' in b]


def test_marks_last_block_of_newest_message():
    msgs = _fresh_messages()
    msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": "a"}]})
    msgs.append({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "a", "content": "one"},
        {"type": "tool_result", "tool_use_id": "b", "content": "two"},
    ]})
    _roll_cache_breakpoint(msgs)

    assert msgs[-1]['content'][-1]['cache_control'] == {
        "type": "ephemeral", "ttl": "5m"}
    assert 'cache_control' not in msgs[-1]['content'][0]


def test_migrates_never_accumulates():
    """Two rounds must leave exactly ONE rolling marker — 4 breakpoints is a
    hard API ceiling and the static three already spend three of them."""
    msgs = _fresh_messages()
    for round_n in range(2):
        msgs.append({"role": "assistant",
                     "content": [{"type": "tool_use", "id": str(round_n)}]})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": str(round_n),
             "content": "result"}]})
        _roll_cache_breakpoint(msgs)

    rolling = [pos for pos in _marked(msgs) if pos[0] != 0]
    assert len(rolling) == 1
    assert rolling[0][0] == len(msgs) - 1


def test_static_breakpoints_survive():
    msgs = _fresh_messages()
    msgs.append({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "a", "content": "r"}]})
    _roll_cache_breakpoint(msgs)

    assert msgs[0]['content'][0]['cache_control']['ttl'] == "1h"
    assert msgs[0]['content'][1]['cache_control']['ttl'] == "5m"


def test_empty_tail_is_a_noop():
    """A round that dispatched no tool calls appends an empty content list —
    marking nothing is correct, crashing is not."""
    msgs = _fresh_messages()
    msgs.append({"role": "user", "content": []})
    _roll_cache_breakpoint(msgs)

    assert _marked(msgs) == [(0, 0), (0, 1)]


# ── run_llm_once: single-shot breakpoints ──
# A list of strings becomes consecutive text blocks; the first carries its own
# breakpoint (a byte-stable menu is paid once per five minutes), the last the
# whole-prompt one. A plain string is one block with one breakpoint.

def _once_client():
    seen = {}

    class _Resp:
        content = []
        stop_reason = 'end_turn'
        usage = None

    class _Messages:
        def create(self, **kw):
            seen.update(kw)
            return _Resp()

    class _Client:
        messages = _Messages()
    return _Client(), seen


def test_run_llm_once_string_is_one_block():
    from servers.scales.runner import run_llm_once
    client, seen = _once_client()
    run_llm_once(client, 'm', 10, 'SYS', 'BODY')
    content = seen['messages'][0]['content']
    assert [b['text'] for b in content] == ['BODY']
    assert 'cache_control' in content[0]


def test_run_llm_once_list_is_blocks_with_a_prefix_breakpoint():
    from servers.scales.runner import run_llm_once
    client, seen = _once_client()
    run_llm_once(client, 'm', 10, 'SYS', ['MENU', 'BODY'])
    content = seen['messages'][0]['content']
    assert [b['text'] for b in content] == ['MENU', 'BODY']
    assert 'cache_control' in content[0]
    assert 'cache_control' in content[1]
    assert 'cache_control' in seen['system'][0]
