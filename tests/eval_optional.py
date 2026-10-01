"""Graceful skip for tests that need the `eval/` harness.

D-8 keeps `eval/` out of the public tree — a published harness is a claim,
and the corpora are personal. The tests that consume it must therefore skip
in an exported tree rather than abort it: a module-level ImportError takes
the whole run down (`Interrupted: N errors during collection`), which turns
"six files can't run here" into "the suite refuses to start".

Two coupling shapes, one door:
  - modules that import eval at module level — call `require_eval()` above
    the import;
  - modules that reach into eval inside a test body — call it at module
    level anyway; the whole module needs the harness either way.
"""

import importlib.util

import pytest


def _have_eval():
    try:
        return importlib.util.find_spec('eval') is not None
    except (ImportError, ValueError):
        return False


HAVE_EVAL = _have_eval()

_REASON = ('eval/ harness is not in this tree — excluded from the public '
           'export by D-8')


def require_eval():
    """Skip the calling module unless the eval/ harness is importable."""
    if not HAVE_EVAL:
        pytest.skip(_REASON, allow_module_level=True)
