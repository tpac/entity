"""Guardrail: dev tooling must still import.

`scripts/` and `eval/` ship to nobody, and `tests/bench_*` / `tests/benchmark_*`
are not collected by pytest (they are `bench_`-prefixed, not `test_`). Nothing
holds their imports to reality, so when a `servers.` symbol is deleted the
harness that named it just rots in place — patched by later refactor sweeps
that update its import lines without ever running it.

That is not hypothetical. September 2026: seven files were found dying on
ImportError, each orphaned months earlier by a deletion elsewhere.
`tests/bench_precision_corpus.py` — whose own docstring called it a "Sacred
system benchmark — run BEFORE and AFTER any precision pipeline change" — had
been un-runnable for five months, and `tests/bench_precision_lifecycle.py` was
orphaned by a commit whose message reads "+ cleanup stale references".
Retired in 77ccb7b and 9fa94b8; this test is what makes the next one loud.

Two properties make the rot invisible, and this test is shaped around both:

  • The dead import is usually FUNCTION-LOCAL. A module-level smoke import
    sails past it; only running the harness reaches it. So every `ImportFrom`
    node is checked, at any nesting depth.
  • Nobody runs the harness. So the check must not require running it — the
    target module's AST is read instead of importing it. The green path costs
    no imports at all and loads no models.

A name that is not found statically is CONFIRMED with a real import before
this fails, so a dynamically-bound name (module `__getattr__`, `globals()[...]`)
cannot produce a false red.

Scope note: `archive/` directories are excluded. They are a declared graveyard
— `eval/archive/` deliberately holds files pinned to `servers.recall_scoring`,
deleted long ago, and `scripts/export-public-tree.sh` denylists `tests/archive`
as dev residue for the same reason. The subject here is LIVE dev tooling.

Run: ./dev pytest tests/test_dev_tooling_guardrail.py -v
"""
import ast
import importlib
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEARCH_DIRS = ('scripts', 'eval', 'tests')
# `archive` — declared graveyard (see module docstring). The rest never hold
# .py source; skipping them keeps the walk cheap.
SKIP_DIRS = {'__pycache__', 'archive', 'venv', 'node_modules', '.git'}


def _dev_tooling_files():
    for d in SEARCH_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for path in base.rglob('*.py'):
            if SKIP_DIRS & set(path.relative_to(ROOT).parts):
                continue
            yield path


def _module_path(dotted):
    """servers.scales.s1.frame -> the .py file backing it, or None."""
    stem = ROOT.joinpath(*dotted.split('.'))
    for cand in (stem.with_suffix('.py'), stem / '__init__.py'):
        if cand.is_file():
            return cand
    return None


def _module_scope_names(tree):
    """Names bound at MODULE scope.

    Descends through `if` / `try` / `with` / `for` bodies (still module scope)
    but never into a function or class body — a local variable there must not
    mask a missing top-level symbol.
    """
    names = set()

    def visit(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                names.add(node.name)          # the def itself binds; its body does not
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    names.update(n.id for n in ast.walk(t)
                                 if isinstance(n, ast.Name))
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    names.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    names.add(a.asname or a.name.split('.')[0])
            else:
                for field in ('body', 'orelse', 'finalbody'):
                    visit(getattr(node, field, []) or [])
                for handler in getattr(node, 'handlers', []) or []:
                    visit(handler.body)

    visit(tree.body)
    return names


def _resolves(module, name):
    """Is `name` importable from `module`? Static first, import to confirm."""
    mod_file = _module_path(module)
    if mod_file is None:
        return False, f'no module {module}'
    try:
        names = _module_scope_names(ast.parse(mod_file.read_text('utf-8')))
    except SyntaxError:
        return True, ''                        # that module's own problem, not ours
    if name in names:
        return True, ''
    if _module_path(f'{module}.{name}') is not None:
        return True, ''                        # `from servers import dal_graph`
    # Not found statically — confirm with a real import before failing, so a
    # dynamically-bound name can't produce a false red.
    try:
        if hasattr(importlib.import_module(module), name):
            return True, ''
    except Exception:
        pass
    return False, f'{module} has no {name}'


class TestDevToolingImports(unittest.TestCase):

    def test_every_dev_tooling_file_parses(self):
        broken = []
        for path in _dev_tooling_files():
            try:
                ast.parse(path.read_text('utf-8'))
            except SyntaxError as e:
                broken.append(f'{path.relative_to(ROOT)}:{e.lineno}: {e.msg}')
        self.assertEqual(broken, [], 'dev tooling that no longer parses')

    def test_no_dev_tooling_imports_a_deleted_symbol(self):
        """Every `from servers... import X` in scripts/, eval/, tests/ names
        something that exists — at any nesting depth."""
        dead = []
        for path in _dev_tooling_files():
            rel = path.relative_to(ROOT)
            try:
                tree = ast.parse(path.read_text('utf-8'))
            except SyntaxError:
                continue                       # reported by the parse test above
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and not node.level:
                    module = node.module or ''
                    if not module.startswith('servers'):
                        continue
                    for a in node.names:
                        if a.name == '*':
                            continue
                        ok, why = _resolves(module, a.name)
                        if not ok:
                            dead.append(f'{rel}:{node.lineno}  '
                                        f'from {module} import {a.name}  ({why})')
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        if (a.name.startswith('servers')
                                and _module_path(a.name) is None):
                            dead.append(f'{rel}:{node.lineno}  '
                                        f'import {a.name}  (no such module)')
        self.assertEqual(
            sorted(dead), [],
            'dev tooling importing symbols that no longer exist — it cannot '
            'run. Fix the import if the harness still answers a live question; '
            'delete the file if it does not. Do not patch the import line '
            'without running it: that is exactly how these rot.')


if __name__ == '__main__':
    unittest.main()
