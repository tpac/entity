"""Bypass guard: the override model has exactly one resolution seam.

Two dangerous shapes, each of which silently removes a learnable boundary:

1. A runtime import of a default around the resolver. `from ...surface_prompt
   import SYSTEM_PROMPT` (or a `*_CONFIG_V1` import) inside `servers/` skips
   the override check: the boundary keeps working and never again reads a
   deployed override — no error, no signal. Only
   `servers/interaction_defaults.py` (the registry the resolver serves) may
   import those constants.

2. A reach past the Brain's accessor doors to `_interaction_dal`. The
   register/activate/clear doors carry the validators and the cache
   invalidation; the DAL alone does not — bypassing them is how an eval
   override used to read stale for a whole TTL window. Only the Brain's own
   modules may touch the attribute. `tests/` is deliberately out of scope:
   constructing door-refused states (unparseable JSON, reserved provenance)
   is legitimate fixture work there, and `tests/interaction_override.py` is
   itself the one sanctioned door for eval overrides.

Run: ./dev pytest tests/test_interaction_bypass_guard.py -v
"""
import pathlib
import re
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent

# ── Shape 1: direct default imports in runtime code ─────────────────────────
# Each regex has two alternates: the one-line form and the parenthesised
# (possibly multiline) form — a linter reflowing a long import must not
# carry a violation out of the detector's sight.
PROMPT_IMPORT_RE = re.compile(
    r'from\s+\S*_prompt\s+import\s+'
    r'(?:[^\n(]*\bSYSTEM_PROMPT\b|\([^)]*\bSYSTEM_PROMPT\b)')
# Config defaults wear three naming conventions: the legacy *_CONFIG_V1
# (scopes), the contract-file *_INTERACTION_DEFAULT (everything since
# Step 2), and a handful of plain names that predate both. A guard that
# only knew the first covered one default out of fourteen.
CONFIG_V1_IMPORT_RE = re.compile(
    r'from\s+\S+\s+import\s+'
    r'(?:[^\n(]*\b\w+(?:_CONFIG_V1|_INTERACTION_DEFAULT)\b'
    r'|\([^)]*\b\w+(?:_CONFIG_V1|_INTERACTION_DEFAULT)\b)')
# Plain-named config defaults, scanned as an explicit list (too generic to
# suffix-match). DEFAULT_CONFIG (recall_laf) is deliberately absent: the
# name is used by unrelated modules and it is only ever self-referenced.
PLAIN_DEFAULT_SYMBOLS = ('COMMUNITY_DETECTION', 'COMMUNITY_ENRICHMENT',
                         'CONSOLIDATION_ENRICHMENT', 'TRACE_RECORDING_NORMAL')
PLAIN_DEFAULT_RE = re.compile(
    r'from\s+\S+\s+import\s+(?:[^\n(]*|\([^)]*)\b(%s)\b'
    % '|'.join(PLAIN_DEFAULT_SYMBOLS))
# brain_traces legitimately imports TRACE_RECORDING_NORMAL as the overlay
# BASE its readers merge the resolved config onto — it is not a bypass;
# the effective value still comes from the resolver.
PLAIN_DEFAULT_ALLOWED = {
    'servers/interaction_defaults.py',
    'servers/brain_traces.py',
}
PROMPT_IMPORT_ALLOWED = {'servers/interaction_defaults.py'}

# ── Shape 2: reaching past the accessor doors ────────────────────────────────
DAL_REACH_RE = re.compile(r'\b_interaction_dal\b')
# brain.py owns the attribute; brain_recall.py is a Brain mixin (self-access).
DAL_ALLOWED = {'servers/brain.py', 'servers/brain_recall.py'}
DAL_SCAN_DIRS = ('servers', 'eval', 'scripts')


def _py_files(*dirs):
    for d in dirs:
        root = REPO / d
        if root.exists():
            yield from sorted(root.rglob('*.py'))


class TestNoDirectDefaultImports(unittest.TestCase):
    def test_runtime_never_imports_a_default_around_the_resolver(self):
        offenders = []
        for path in _py_files('servers'):
            rel = str(path.relative_to(REPO))
            if rel in PROMPT_IMPORT_ALLOWED:
                continue
            text = path.read_text()
            for regex in (PROMPT_IMPORT_RE, CONFIG_V1_IMPORT_RE):
                for m in regex.finditer(text):
                    offenders.append('%s: %s' % (rel, m.group(0).strip()))
            if rel not in PLAIN_DEFAULT_ALLOWED:
                for m in PLAIN_DEFAULT_RE.finditer(text):
                    offenders.append('%s: %s' % (rel, m.group(0).strip()))
        self.assertEqual(offenders, [], (
            'runtime code imports a prompt/config default directly, skipping '
            'the override resolver — read it via get_interaction_prompt/'
            '_config instead:\n' + '\n'.join(offenders)))


class TestNoInteractionDalReachArounds(unittest.TestCase):
    def test_only_the_brain_touches_the_interaction_dal(self):
        offenders = []
        for path in _py_files(*DAL_SCAN_DIRS):
            rel = str(path.relative_to(REPO))
            if rel in DAL_ALLOWED:
                continue
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if DAL_REACH_RE.search(line):
                    offenders.append('%s:%d: %s' % (rel, i, line.strip()))
        self.assertEqual(offenders, [], (
            '_interaction_dal reached outside its owner — route writes '
            'through brain.register_interaction / set_interaction_active / '
            'clear_interaction_override (evals: '
            'tests/interaction_override.override_interaction), reads through '
            'the resolver accessors:\n' + '\n'.join(offenders)))


class TestDetectorsHaveTeeth(unittest.TestCase):
    """A guard is only as good as its regexes — assert they catch the shapes
    they must and ignore the legitimate neighbours."""

    def test_prompt_import_detector(self):
        catches = [
            'from .scales.s1.surface_prompt import SYSTEM_PROMPT',
            'from servers.scales.s2.healer_prompt import SYSTEM_PROMPT as H',
            'from ..encoding_prompt import FOO, SYSTEM_PROMPT',
            'from .surface_prompt import (\n    FOO,\n    SYSTEM_PROMPT,\n)',
        ]
        for s in catches:
            self.assertTrue(PROMPT_IMPORT_RE.search(s), s)
        ignores = [
            'from .surface_contract import build_surface_prompt',
            'prompt = brain.get_interaction_prompt("surface")',
            'from .trace_contract import SYSTEM_PROMPT_KINDS',
        ]
        for s in ignores:
            self.assertFalse(PROMPT_IMPORT_RE.search(s), s)

    def test_config_v1_import_detector(self):
        self.assertTrue(CONFIG_V1_IMPORT_RE.search(
            'from .scopes import SCOPES_CONFIG_V1'))
        self.assertTrue(CONFIG_V1_IMPORT_RE.search(
            'from .scopes import (\n    SCOPES_CONFIG_V1,\n)'))
        self.assertTrue(CONFIG_V1_IMPORT_RE.search(
            'from .healer_contract import HEALER_INTERACTION_DEFAULT'))
        self.assertFalse(CONFIG_V1_IMPORT_RE.search(
            'HEALER_INTERACTION_DEFAULT = {'))   # the defining assignment
        self.assertTrue(PLAIN_DEFAULT_RE.search(
            'from .community_contract import COMMUNITY_DETECTION'))
        self.assertFalse(PLAIN_DEFAULT_RE.search(
            'COMMUNITY_DETECTION = {'))          # the defining assignment
        self.assertFalse(CONFIG_V1_IMPORT_RE.search(
            'SCOPES_CONFIG_V1 = {'))          # the defining assignment
        self.assertFalse(CONFIG_V1_IMPORT_RE.search(
            'from .scopes import validate_scopes_config'))

    def test_dal_reach_detector(self):
        self.assertTrue(DAL_REACH_RE.search(
            "brain._interaction_dal.set_active('s1e', 2)"))
        self.assertFalse(DAL_REACH_RE.search(
            'dal = InteractionDAL(conn)'))    # the class itself, in its owner


if __name__ == '__main__':
    unittest.main()
