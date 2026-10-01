"""Guardrail: every S2 encoder LLM call carries the journal.

The healer and aspect units ran mute for months — no journal binding, so no
residue, and nothing failed (the silence was only visible in a corpus-level
audit, finding 78677e17). This holds the line structurally: any `._call_llm(`
call site in an S2 unit module must opt into the journal (`journal=True` as a
top-level argument), and the loop-encoder path must decorate + harvest.
A deliberate journal-exempt call would edit this test — which is the point:
exemption becomes a reviewed decision, not a default.
"""
import inspect
import ast
import os
import unittest

S2_DIR = os.path.join(os.path.dirname(__file__), '..',
                      'servers', 'scales', 's2')

CALL_TOKEN = '._call_llm('   # matches self._call_llm( and super()._call_llm(


def _s2_sources():
    for fname in sorted(os.listdir(S2_DIR)):
        if not fname.endswith('.py'):
            continue
        path = os.path.join(S2_DIR, fname)
        with open(path) as f:
            yield fname, f.read()


def _code_lines(src):
    """Source lines with comments stripped — so prose mentioning a symbol
    can't classify a file."""
    return [line.split('#', 1)[0] for line in src.splitlines()]


def _call_args(src, start):
    """The argument text of a call, parens balanced (start = index just
    after the opening paren). Handles nested calls, which the previous
    regex ([^)]*) could not — it truncated at the first ')' and both
    false-failed and false-passed idiomatic call shapes."""
    depth, i = 1, start
    while i < len(src) and depth:
        if src[i] == '(':
            depth += 1
        elif src[i] == ')':
            depth -= 1
        i += 1
    return src[start:i - 1]


def _top_level_args(args):
    """Split an argument string on depth-0 commas."""
    parts, depth, cur = [], 0, []
    for ch in args:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append(''.join(cur).strip())
    return parts


class TestJournalBindingGuardrail(unittest.TestCase):

    def test_every_encoder_prepares_scoped_continuity(self):
        """Refresh policy belongs to the binding, never a caller's frozen text.

        Discover the real S2 LLM callers; a new encoder cannot quietly bypass
        the preparation contract. Scribe has its standalone encode entry.
        """
        sources = [(name, src) for name, src in _s2_sources()
                   if name != 'base.py' and any(
                       'run_llm_loop(' in line or CALL_TOKEN in line
                       for line in _code_lines(src))]
        from servers.scales.s1 import encode
        sources.append(('s1/encode.py', inspect.getsource(encode)))
        for name, src in sources:
            with self.subTest(encoder=name):
                tree = ast.parse(src)
                calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute)]
                preparations = [n for n in calls if n.func.attr == 'continuity']
                self.assertTrue(preparations, name + ' omits journal continuity')
                for call in preparations:
                    self.assertIn('chain_id', [kw.arg for kw in call.keywords])
                for call in calls:
                    if isinstance(call.func.value, ast.Attribute):
                        self.assertFalse(call.func.value.attr == 'journal'
                                         and call.func.attr in ('residue', 'messages'),
                                         name + ' bypasses shared preparation')
                # Multiple independent batches require preparation INSIDE
                # their loop. Aspect and Scribe have one request per run.
                if name in ('community_encoder.py', 'consolidation_encoder.py',
                            'healer_encoder.py'):
                    loops = [n for n in ast.walk(tree)
                             if isinstance(n, (ast.For, ast.While))]
                    self.assertTrue(any(call in list(ast.walk(loop))
                                        for call in preparations for loop in loops),
                                    name + ' freezes continuity outside the batch loop')

    def test_every_call_llm_site_binds_journal(self):
        """A `._call_llm(...)` in an S2 module without a top-level
        journal=True is a new mute unit in the making."""
        offenders = []
        for fname, src in _s2_sources():
            code = '\n'.join(_code_lines(src))
            pos = 0
            while True:
                idx = code.find(CALL_TOKEN, pos)
                if idx == -1:
                    break
                arg_start = idx + len(CALL_TOKEN)
                args = _call_args(code, arg_start)
                pos = arg_start + len(args)
                bound = any(a.replace(' ', '').startswith('journal=True')
                            for a in _top_level_args(args))
                if not bound:
                    line = code[:idx].count('\n') + 1
                    offenders.append('%s:%d' % (fname, line))
        self.assertEqual(offenders, [], (
            '._call_llm without top-level journal=True: %s — single-shot S2 '
            'runs must carry the journal binding (or edit this test with the '
            'reason for the exemption)' % offenders))

    def test_every_run_llm_loop_encoder_decorates_and_harvests(self):
        """Loop encoders bind via decorate_system + journal.harvest (the
        harvest call may live in the shared _fold_batch_result). Classified
        by CODE calls, not by prose mentions in comments/docstrings."""
        for fname, src in _s2_sources():
            if fname == 'base.py':
                continue
            code = _code_lines(src)
            if not any('run_llm_loop(' in line for line in code):
                continue
            self.assertTrue(
                any('.journal.decorate_system(' in line for line in code),
                '%s runs the LLM loop without decorating the system prompt '
                'with the journal blocks' % fname)
            self.assertTrue(
                any('.journal.harvest(' in line
                    or '_fold_batch_result(' in line for line in code),
                '%s runs the LLM loop without harvesting residue' % fname)

    def test_fold_batch_result_still_harvests(self):
        """The loop encoders satisfy the harvest check via
        _fold_batch_result — so the shared body must actually harvest, or
        both loop encoders go mute while this guardrail stays green."""
        from servers.scales.s2.base import IntegrationUnit
        src = inspect.getsource(IntegrationUnit._fold_batch_result)
        self.assertIn('.journal.harvest(', src)

    def test_call_llm_journal_path_still_harvests(self):
        """journal=True must keep meaning decorate + harvest inside
        _call_llm — the single-shot units' whole binding rides on it."""
        from servers.scales.s2.base import IntegrationUnit
        src = inspect.getsource(IntegrationUnit._call_llm)
        self.assertIn('.decorate_system(', src)
        self.assertIn('.journal.harvest(', src)


if __name__ == '__main__':
    unittest.main()
