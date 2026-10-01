"""The scale write-command classification — derivation invariants.

dispatch.py no longer holds a dispatch factory (S1 Scribe + the S2 units share
the in-process _make_encoder_dispatch). What survives is the canonical
write-command classification the attribution chokepoint reads. These pin the
derivation that keeps a new attributed write tagged automatically — the gap that
once mislabelled brain_batch encodes as 'anchor' and hid them from the encoder
view. (Attribution BEHAVIOR is tested in test_s1_scribe.py via
apply_encoder_attribution; the daemon-subset contract in test_contract_sync.py
Layer 5.)
"""

from servers.scales.dispatch import (
    WRITE_COMMANDS, NON_ATTRIBUTED_WRITES, ATTRIBUTED_WRITE_COMMANDS,
    OPERATOR_ONLY_COMMANDS)


def test_attributed_is_derived_from_write_commands():
    # Derived, not hand-maintained: a new attributed write added to
    # WRITE_COMMANDS becomes attributed automatically and can't slip through
    # untagged.
    assert ATTRIBUTED_WRITE_COMMANDS == WRITE_COMMANDS - NON_ATTRIBUTED_WRITES
    assert ATTRIBUTED_WRITE_COMMANDS <= WRITE_COMMANDS


def test_non_attributed_are_the_structural_writes():
    # trace_append / set_config mint no attributable node/edge, so they
    # are explicitly excluded from attribution.
    assert NON_ATTRIBUTED_WRITES == {'trace_append', 'set_config'}
    assert not (NON_ATTRIBUTED_WRITES & ATTRIBUTED_WRITE_COMMANDS)


def test_operator_only_commands_are_registered_and_disjoint():
    # Operator-only commands must exist in the daemon registry (or the closure
    # refusal guards a phantom) and must never appear in the scale write
    # classification — a command in both would be attributed as an encoder
    # write while the closure refuses it.
    from servers.daemon_dispatch import COMMAND_TABLE
    assert OPERATOR_ONLY_COMMANDS <= set(COMMAND_TABLE)
    assert not (OPERATOR_ONLY_COMMANDS & WRITE_COMMANDS)
