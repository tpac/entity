"""The eval-support override door: does it reach the resolver, and does it
always take itself back?

These pin the two properties eval arms rely on and could not previously get:
an override that ACTIVATES (six hand-rolled copies gated activation on
`version > 1`, an auto-activate assumption that died with Step 6, so a v1
override on a freshly wiped corpus brain never took effect and both arms ran
the same prompt), and an override that REVERTS (without which a forgotten
pointer opts that name out of code defaults for the life of the brain).

Run: ./dev pytest tests/test_interaction_override.py -v
"""
import json
import pytest

from tests.isolated_brain import brain_env
from tests.interaction_override import (
    override_interaction, interaction_override, _as_dict)


class TestOverrideDoor:
    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            # Start from no pointers, so a measured "default" is the code
            # default and not an inherited override — the trap a baseline
            # eval arm falls into.
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.execute('DELETE FROM interaction_active')
            env.brain.logs_conn.commit()
            yield

    def _pointer(self, name):
        row = self.brain.logs_conn.execute(
            'SELECT version FROM interaction_active WHERE name = ?',
            (name,)).fetchone()
        return row[0] if row else None

    # ─── reaching the resolver ──────────────────────────────────────────

    def test_v1_override_on_a_wiped_brain_activates(self):
        """The bug in every hand-rolled copy: `if version > 1: set_active`.

        On a wiped corpus brain the override IS v1, so the pointer was never
        flipped and the arm silently ran the code default.
        """
        version = override_interaction(self.brain, 's1e', template='ARM-A')

        assert version == 1
        assert self._pointer('s1e') == 1
        assert self.brain.get_interaction_prompt('s1e') == 'ARM-A'
        assert self.brain.get_interaction_stamp('s1e')['source'] == 'override'

    def test_template_none_preserves_the_code_default_not_an_empty_row(self):
        """`_interaction_dal.get_active` returns None on a pointer-less brain,
        so "preserve what's active" used to write an empty template."""
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        default_template = INTERACTION_DEFAULTS['s1e'][0]
        assert default_template, 's1e default template is the fixture here'

        override_interaction(self.brain, 's1e',
                             parameters={'effort': 'sentinel'})

        assert self.brain.get_interaction_prompt('s1e') == default_template
        assert (self.brain.get_interaction_config('s1e')['effort']
                == 'sentinel')

    def test_merge_overlays_onto_the_effective_config(self):
        """recall_laf's config carries fitted gains — a wholesale replace
        resets a corpus brain's gains to module defaults and lets a
        base-parity check pass against a config that brain never ran."""
        override_interaction(self.brain, 'recall_laf', template='',
                             parameters={'z_norm': 'support', 'gain_x': 0.5})

        override_interaction(self.brain, 'recall_laf', template='',
                             parameters={'z_norm': 'global'}, merge=True)
        merged = self.brain.get_interaction_config('recall_laf')
        assert merged['z_norm'] == 'global'
        assert merged['gain_x'] == 0.5, 'merge must not drop existing keys'

        override_interaction(self.brain, 'recall_laf', template='',
                             parameters={'z_norm': 'support'}, merge=False)
        replaced = self.brain.get_interaction_config('recall_laf')
        assert replaced['z_norm'] == 'support'
        assert 'gain_x' not in replaced, 'merge=False stores what was passed'

    def test_byte_identical_override_warns_instead_of_passing_silently(self, capsys):
        """Fingerprint that does not move = both arms measure the same K, and
        the A/B reports a difference of zero as if it were a result."""
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        template, config = INTERACTION_DEFAULTS['s1e']

        before = self.brain.get_interaction_stamp('s1e')['fingerprint']
        override_interaction(self.brain, 's1e', template=template,
                             parameters=dict(config))
        after = self.brain.get_interaction_stamp('s1e')['fingerprint']

        assert after == before
        assert 'byte-identical' in capsys.readouterr().err

    def test_config_only_override_of_a_name_that_has_a_default_template(self):
        """`template=''` is the config-only idiom, NOT a template override —
        the resolver keeps serving the code default because it takes a row's
        template only when non-empty. A verify that asserted "effective
        template == what I set" would raise on every config-only override of a
        name with a real default template."""
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        default_template = INTERACTION_DEFAULTS['s2_healer'][0]
        assert default_template, 'fixture needs a non-empty default template'

        override_interaction(self.brain, 's2_healer', template='',
                             parameters={'limit': 3}, merge=True)

        assert (self.brain.get_interaction_prompt('s2_healer')
                == default_template)
        assert (self.brain.get_interaction_config('s2_healer')['limit'] == 3)

    def test_parameters_accepts_the_json_string_the_dal_stores(self):
        override_interaction(self.brain, 'recall_laf', template='',
                             parameters=json.dumps({'z_norm': 'support'}))
        assert (self.brain.get_interaction_config('recall_laf')['z_norm']
                == 'support')
        assert _as_dict('') == {}
        assert _as_dict(None) is None
        # A JSON array would make the resolver log-and-default at READ time —
        # silently, from the arm's side. Refuse it where the mistake is.
        with pytest.raises(TypeError):
            _as_dict('[1, 2]')

    # ─── taking it back ─────────────────────────────────────────────────

    def test_context_manager_clears_on_normal_exit(self):
        from servers.interaction_defaults import INTERACTION_DEFAULTS

        with interaction_override(self.brain, 's1e', template='ARM-B'):
            assert self.brain.get_interaction_prompt('s1e') == 'ARM-B'

        assert self._pointer('s1e') is None
        assert (self.brain.get_interaction_prompt('s1e')
                == INTERACTION_DEFAULTS['s1e'][0])
        assert self.brain.get_interaction_stamp('s1e')['source'] == 'default'

    def test_context_manager_clears_when_the_body_raises(self):
        """The named verification: enter, raise inside, assert no pointer
        remains on exit — and that the body's exception is the one that
        propagates, not something the cleanup raised over it."""
        sentinel = RuntimeError('arm blew up mid-encode')

        with pytest.raises(RuntimeError) as caught:
            with interaction_override(self.brain, 's1e', template='ARM-C'):
                assert self._pointer('s1e') == 1
                raise sentinel

        assert caught.value is sentinel
        assert self._pointer('s1e') is None
        assert self.brain.get_interaction_stamp('s1e')['source'] == 'default'

    def test_exit_clear_survives_a_body_that_already_cleared(self):
        """A double clear must not raise — it is what makes the unguarded
        clear in __exit__ safe when the body cleared or re-pointed."""
        with interaction_override(self.brain, 's1e', template='ARM-D'):
            self.brain.clear_interaction_override('s1e')

        assert self._pointer('s1e') is None

    def test_unregistered_name_is_refused_before_anything_is_written(self):
        """The invariant the unguarded exit clear rests on.

        `clear_interaction_override` raises KeyError for a name that deleted
        nothing and has no code default. If such a name could get INSIDE a
        context manager, the exit clear would raise from __exit__ and replace
        a propagating body exception — a leak-proof block turned
        error-masking. It cannot: entry resolves the name first and the
        resolver refuses unknown names, leaving no row and no pointer behind.
        """
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        assert 'not_a_real_interaction' not in INTERACTION_DEFAULTS

        with pytest.raises(KeyError):
            override_interaction(self.brain, 'not_a_real_interaction',
                                 template='X')

        assert self._pointer('not_a_real_interaction') is None
        assert self.brain.logs_conn.execute(
            'SELECT COUNT(*) FROM interactions WHERE name = ?',
            ('not_a_real_interaction',)).fetchone()[0] == 0
