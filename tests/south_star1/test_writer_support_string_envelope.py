"""Durable writer support-string envelope tests."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
import unittest

from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot_envelope import (
    verify_writer_snapshot_advance_envelope,
)
from grimace._south_star1.writer_snapshot_envelope import (
    writer_snapshot_advance_envelope_for_emitted_text,
)
from grimace._south_star1.writer_snapshot_prefix_envelope import (
    writer_snapshot_prefix_read_envelope_for_emitted_texts,
)
from grimace._south_star1.writer_support_string_envelope import (
    verify_writer_support_string_envelope,
)
from grimace._south_star1.writer_support_string_envelope import (
    writer_support_string_envelope_for_prefix_read,
)
from grimace._south_star1.writer_support_string_envelope import (
    writer_support_string_envelope_for_string,
)
from grimace._south_star1.writer_envelope_terms import _canonical_json
from grimace._south_star1.writer_support_string_envelope import (
    _support_string_manifest,
)
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import tetrahedral_facts
from tests.south_star1.helpers import two_atom_facts


class WriterSupportStringEnvelopeTest(unittest.TestCase):
    def test_one_token_support_string_json_round_trips(self) -> None:
        prepared = _prepare(two_atom_facts())
        source = _advanced_snapshot_after_prefix(
            prepared,
            _initial_snapshot(prepared),
            ("C",),
        )
        envelope = _json_round_trip(
            writer_support_string_envelope_for_string(
                prepared=prepared,
                snapshot=source,
                string="C",
            )
        )

        verification = verify_writer_support_string_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.string, "C")

    def test_multi_token_support_string_json_round_trips(self) -> None:
        prepared = _prepare(two_atom_facts())
        envelope = _json_round_trip(
            writer_support_string_envelope_for_string(
                prepared=prepared,
                snapshot=_initial_snapshot(prepared),
                string="CC",
            )
        )

        verification = verify_writer_support_string_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(envelope["emitted_texts"], ["C", "C"])
        self.assertEqual(len(envelope["text_projection_chain"]), 2)

    def test_empty_terminal_support_string_json_round_trips(self) -> None:
        prepared = _prepare(two_atom_facts())
        source = _advanced_snapshot_after_prefix(
            prepared,
            _initial_snapshot(prepared),
            ("C", "C"),
        )
        envelope = _json_round_trip(
            writer_support_string_envelope_for_string(
                prepared=prepared,
                snapshot=source,
                string="",
            )
        )

        verification = verify_writer_support_string_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(envelope["emitted_texts"], [])
        self.assertEqual(envelope["text_projection_chain"], [])

    def test_prefix_read_source_support_string_json_round_trips(self) -> None:
        prepared = _prepare(two_atom_facts())
        snapshot = _initial_snapshot(prepared)
        prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=snapshot,
            emitted_texts=("C",),
        )
        envelope = _json_round_trip(
            writer_support_string_envelope_for_prefix_read(
                prepared=prepared,
                prefix_read_envelope=prefix,
                string="C",
            )
        )

        verification = verify_writer_support_string_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.source_kind, "prefix_read")

    def test_branching_frontier_support_string_verifies(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        envelope = writer_support_string_envelope_for_string(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared),
            string="C1CC1",
        )

        self.assertTrue(
            verify_writer_support_string_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_terminal_support_identity_is_present(self) -> None:
        envelope = _terminal_envelope()

        self.assertTrue(envelope["terminal_support_identities"])
        self.assertEqual(
            envelope["terminal_support_identities"],
            envelope["terminal_projection"]["terminal_support_identities"],
        )
        self.assertTrue(_verify(envelope).accepted)

    def test_unknown_schema_name_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["schema_name"] = "other"

        self.assertFalse(_verify(envelope).accepted)

    def test_unknown_schema_version_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["schema_version"] = 999

        self.assertFalse(_verify(envelope).accepted)

    def test_extra_top_level_field_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["extra"] = {}

        self.assertFalse(_verify(envelope).accepted)

    def test_missing_required_field_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        del envelope["terminal_projection"]

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_prepared_identity_is_rejected(self) -> None:
        envelope = _snapshot_envelope()

        self.assertFalse(
            verify_writer_support_string_envelope(
                prepared=_prepare(tetrahedral_facts()),
                envelope=envelope,
            ).accepted
        )

    def test_wrong_source_kind_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["source_kind"] = "choice_snapshot"

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_source_snapshot_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["source_snapshot"]["cursor"]["digest"] = "0" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_prefix_read_envelope_is_rejected(self) -> None:
        prepared = _prepare(two_atom_facts())
        envelope = _prefix_envelope(prepared)
        envelope["prefix_read_envelope"]["schema_name"] = "bad"

        self.assertFalse(
            verify_writer_support_string_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_wrong_count_envelope_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["count_envelope"]["schema_name"] = "bad"

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_string_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["string"] = "bad"

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_emitted_texts_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["emitted_texts"] = ["C"]

        self.assertFalse(_verify(envelope).accepted)

    def test_string_emitted_texts_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["string"] = "C"

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_replay_envelope_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["replay_envelope"]["emitted_texts"] = ["C"]

        self.assertFalse(_verify(envelope).accepted)

    def test_non_advanced_replay_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["replay_envelope"]["outcome_kind"] = "blocked"

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_final_snapshot_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["final_snapshot"] = envelope["source_snapshot"]

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_terminal_frontier_product_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["terminal_frontier_product"]["digest"] = "1" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_support_string_manifest_digest_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["digest"] = "4" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_support_string_manifest_is_smaller_than_full_envelope(self) -> None:
        envelope = _snapshot_envelope()

        self.assertLess(
            len(_canonical_json(_support_string_manifest(envelope))),
            len(_canonical_json(envelope)),
        )

    def test_changed_terminal_projection_digest_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["terminal_projection"]["digest"] = "2" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_terminal_support_identity_is_rejected(self) -> None:
        envelope = _terminal_envelope()
        envelope["terminal_support_identities"][0]["source_state_digest"] = (
            "3" * 64
        )

        self.assertFalse(_verify_terminal(envelope).accepted)

    def test_changed_terminal_ordinal_is_rejected(self) -> None:
        envelope = _terminal_envelope()
        envelope["terminal_support_identities"][0]["terminal_ordinal"] += 1

        self.assertFalse(_verify_terminal(envelope).accepted)

    def test_changed_terminal_parent_weight_is_rejected(self) -> None:
        envelope = _terminal_envelope()
        envelope["terminal_support_identities"][0]["parent_weight"] += 1

        self.assertFalse(_verify_terminal(envelope).accepted)

    def test_changed_text_projection_chain_step_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["text_projection_chain"][0]["emitted_text"] = "bad"

        self.assertFalse(_verify(envelope).accepted)

    def test_reordered_text_projection_chain_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        chain = envelope["text_projection_chain"]
        chain[0], chain[1] = chain[1], chain[0]

        self.assertFalse(_verify(envelope).accepted)

    def test_removed_text_projection_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["text_projection_chain"] = envelope["text_projection_chain"][1:]

        self.assertFalse(_verify(envelope).accepted)

    def test_extra_text_projection_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["text_projection_chain"].append(
            deepcopy(envelope["text_projection_chain"][0])
        )

        self.assertFalse(_verify(envelope).accepted)

    def test_verifier_does_not_call_certified_support_iteration(self) -> None:
        source = inspect.getsource(verify_writer_support_string_envelope)
        self.assertNotIn("_iter_writer_snapshot_certified_support_strings", source)


def _snapshot_envelope():
    prepared = _prepare(two_atom_facts())
    return deepcopy(
        writer_support_string_envelope_for_string(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared),
            string="CC",
        )
    )


def _prefix_envelope(prepared):
    snapshot = _initial_snapshot(prepared)
    prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
        prepared=prepared,
        snapshot=snapshot,
        emitted_texts=("C",),
    )
    return deepcopy(
        writer_support_string_envelope_for_prefix_read(
            prepared=prepared,
            prefix_read_envelope=prefix,
            string="C",
        )
    )


def _terminal_envelope():
    prepared = _prepare(two_atom_facts())
    source = _advanced_snapshot_after_prefix(
        prepared,
        _initial_snapshot(prepared),
        ("C", "C"),
    )
    return deepcopy(
        writer_support_string_envelope_for_string(
            prepared=prepared,
            snapshot=source,
            string="",
        )
    )


def _verify(envelope):
    return verify_writer_support_string_envelope(
        prepared=_prepare(two_atom_facts()),
        envelope=envelope,
    )


def _verify_terminal(envelope):
    return verify_writer_support_string_envelope(
        prepared=_prepare(two_atom_facts()),
        envelope=envelope,
    )


def _advanced_snapshot_after_prefix(prepared, snapshot, emitted_texts):
    current = snapshot
    for emitted_text in emitted_texts:
        advance = writer_snapshot_advance_envelope_for_emitted_text(
            prepared=prepared,
            snapshot=current,
            emitted_text=emitted_text,
        )
        verification = verify_writer_snapshot_advance_envelope(
            prepared=prepared,
            envelope=advance,
        )
        if not verification.accepted or verification.advanced_snapshot is None:
            raise AssertionError("prefix helper failed to advance")
        current = verification.advanced_snapshot
    return current


def _json_round_trip(envelope):
    return json.loads(json.dumps(envelope, sort_keys=True))


def _initial_snapshot(prepared):
    options = _writer_options()
    return capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=initial_writer_frontier_cursor(prepared, options),
    )


def _writer_options():
    return SouthStarRuntimeOptions(
        rooted_at_atom=-1,
        canonical=False,
        do_random=True,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


if __name__ == "__main__":
    unittest.main()
