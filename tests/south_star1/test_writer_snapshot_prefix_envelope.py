"""Durable writer snapshot prefix-read envelope tests."""

from __future__ import annotations

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import grimace._south_star1.writer_frontier as writer_frontier_module
import grimace._south_star1.writer_snapshot_prefix_envelope as prefix_module
import grimace._south_star1.writer_snapshot_replay_envelope as replay_module
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_frontier import writer_frontier_choices
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot_envelope import (
    verify_writer_snapshot_advance_envelope,
)
from grimace._south_star1.writer_snapshot_envelope import (
    writer_snapshot_advance_envelope_for_emitted_text,
)
from grimace._south_star1.writer_snapshot_prefix_envelope import (
    verify_writer_snapshot_prefix_read_envelope,
)
from grimace._south_star1.writer_snapshot_prefix_envelope import (
    writer_snapshot_prefix_read_envelope_for_emitted_texts,
)
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import tetrahedral_facts


class WriterSnapshotPrefixEnvelopeTest(unittest.TestCase):
    def test_empty_prefix_readable_envelope_json_round_trips(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = _json_round_trip(
            writer_snapshot_prefix_read_envelope_for_emitted_texts(
                prepared=prepared,
                snapshot=_initial_snapshot(prepared),
                emitted_texts=(),
            )
        )

        verification = verify_writer_snapshot_prefix_read_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.read_kind, "readable")
        self.assertIsNotNone(verification.support_count)
        self.assertIsNotNone(verification.completion_count)

    def test_multi_step_prefix_readable_envelope_json_round_trips(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared)
        emitted_texts = _legal_prefix(prepared, snapshot, length=2)
        envelope = _json_round_trip(
            writer_snapshot_prefix_read_envelope_for_emitted_texts(
                prepared=prepared,
                snapshot=snapshot,
                emitted_texts=emitted_texts,
            )
        )

        verification = verify_writer_snapshot_prefix_read_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(envelope["read_kind"], "readable")
        self.assertEqual(envelope["emitted_texts"], list(emitted_texts))
        self.assertIsNotNone(envelope["prefix_read_certificate"])

    def test_readable_without_counts_json_round_trips(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = _json_round_trip(
            writer_snapshot_prefix_read_envelope_for_emitted_texts(
                prepared=prepared,
                snapshot=_initial_snapshot(prepared),
                emitted_texts=(),
                include_counts=False,
            )
        )

        verification = verify_writer_snapshot_prefix_read_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertFalse(envelope["include_counts"])
        self.assertIsNone(envelope["support_count"])
        self.assertIsNone(envelope["completion_count"])

    def test_invalid_text_at_first_step_verifies(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = _json_round_trip(
            writer_snapshot_prefix_read_envelope_for_emitted_texts(
                prepared=prepared,
                snapshot=_initial_snapshot(prepared),
                emitted_texts=("not-a-choice",),
            )
        )

        verification = verify_writer_snapshot_prefix_read_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.read_kind, "invalid_emitted_text")
        self.assertIsNotNone(envelope["failure"]["failed_advance_envelope"])

    def test_invalid_text_after_successful_step_verifies(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared)
        emitted_texts = (*_legal_prefix(prepared, snapshot, length=1), "bad")
        envelope = _json_round_trip(
            writer_snapshot_prefix_read_envelope_for_emitted_texts(
                prepared=prepared,
                snapshot=snapshot,
                emitted_texts=emitted_texts,
            )
        )

        verification = verify_writer_snapshot_prefix_read_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.read_kind, "invalid_emitted_text")

    def test_blocked_replay_verifies(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        snapshot = _initial_snapshot(prepared)

        with _blocked_capability_patch(prepared, snapshot.cursor):
            envelope = _json_round_trip(
                writer_snapshot_prefix_read_envelope_for_emitted_texts(
                    prepared=prepared,
                    snapshot=snapshot,
                    emitted_texts=("C",),
                )
            )
            verification = verify_writer_snapshot_prefix_read_envelope(
                prepared=prepared,
                envelope=envelope,
            )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.read_kind, "replay_blocked")

    def test_final_frontier_blocked_verifies(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        snapshot = _initial_snapshot(prepared)
        emitted_texts = _legal_prefix(prepared, snapshot, length=1)

        with _final_frontier_blocked_patch(prepared, snapshot, emitted_texts):
            envelope = _json_round_trip(
                writer_snapshot_prefix_read_envelope_for_emitted_texts(
                    prepared=prepared,
                    snapshot=snapshot,
                    emitted_texts=emitted_texts,
                )
            )
            verification = verify_writer_snapshot_prefix_read_envelope(
                prepared=prepared,
                envelope=envelope,
            )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.read_kind, "final_frontier_blocked")
        self.assertEqual(envelope["final_frontier_product_kind"], "blocked")

    def test_unknown_schema_name_is_rejected(self) -> None:
        envelope = _readable_envelope()
        envelope["schema_name"] = "other"

        self.assertFalse(_verify(envelope).accepted)

    def test_unknown_schema_version_is_rejected(self) -> None:
        envelope = _readable_envelope()
        envelope["schema_version"] = 999

        self.assertFalse(_verify(envelope).accepted)

    def test_extra_top_level_field_is_rejected(self) -> None:
        envelope = _readable_envelope()
        envelope["extra"] = {}

        self.assertFalse(_verify(envelope).accepted)

    def test_missing_required_field_is_rejected(self) -> None:
        envelope = _readable_envelope()
        del envelope["public_frontier"]

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_prepared_identity_is_rejected(self) -> None:
        envelope = _readable_envelope()

        self.assertFalse(
            verify_writer_snapshot_prefix_read_envelope(
                prepared=_prepare(tetrahedral_facts()),
                envelope=envelope,
            ).accepted
        )

    def test_wrong_emitted_texts_are_rejected(self) -> None:
        envelope = _readable_envelope()
        envelope["emitted_texts"] = ["not-a-choice"]

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_include_counts_is_rejected(self) -> None:
        envelope = _readable_envelope()
        envelope["include_counts"] = False

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_replay_envelope_is_rejected(self) -> None:
        envelope = _readable_envelope()
        envelope["replay_envelope"]["outcome_kind"] = "blocked"

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_final_snapshot_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared)
        envelope = _readable_envelope(
            prepared=prepared,
            snapshot=snapshot,
            emitted_texts=_legal_prefix(prepared, snapshot, length=1),
        )
        envelope["final_snapshot"] = envelope["source_snapshot"]

        self.assertFalse(
            verify_writer_snapshot_prefix_read_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_readable_with_blocked_product_kind_is_rejected(self) -> None:
        envelope = _readable_envelope()
        envelope["final_frontier_product_kind"] = "blocked"

        self.assertFalse(_verify(envelope).accepted)

    def test_stale_prefix_certificate_is_rejected(self) -> None:
        envelope = _readable_envelope()
        envelope["prefix_read_certificate"]["digest"] = "0" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_support_count_is_rejected(self) -> None:
        envelope = _readable_envelope()
        envelope["support_count"] += 1

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_completion_count_is_rejected(self) -> None:
        envelope = _readable_envelope()
        envelope["completion_count"] += 1

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_public_choice_text_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared)
        emitted_texts = _legal_prefix(prepared, snapshot, length=1)
        envelope = _readable_envelope(
            prepared=prepared,
            snapshot=snapshot,
            emitted_texts=emitted_texts,
        )
        if not envelope["public_frontier"]["choices"]:
            self.skipTest("fixture has no text choices")
        envelope["public_frontier"]["choices"][0]["emitted_text"] = "bad"

        self.assertFalse(
            verify_writer_snapshot_prefix_read_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_changed_successor_cursor_digest_is_rejected(self) -> None:
        envelope = _readable_envelope()
        if not envelope["public_frontier"]["choices"]:
            self.skipTest("fixture has no text choices")
        envelope["public_frontier"]["choices"][0]["successor_cursor"][
            "digest"
        ] = "1" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_terminal_support_identity_is_exposed_in_readable_envelope(
        self,
    ) -> None:
        prepared, envelope = _terminal_readable_envelope()

        terminal = envelope["public_frontier"]["terminal"]
        self.assertIsNotNone(terminal)
        self.assertTrue(terminal["terminal_support_identities"])
        self.assertEqual(
            terminal["terminal_support_identities"],
            envelope["final_frontier_product"]["terminal_support_identities"],
        )
        verification = verify_writer_snapshot_prefix_read_envelope(
            prepared=prepared,
            envelope=envelope,
        )
        self.assertTrue(verification.accepted)

    def test_changed_terminal_support_ordinal_is_rejected(self) -> None:
        prepared, envelope = _terminal_readable_envelope()
        identity = envelope["final_frontier_product"][
            "terminal_support_identities"
        ][0]
        identity["terminal_ordinal"] += 1

        self.assertFalse(
            verify_writer_snapshot_prefix_read_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_changed_terminal_support_source_digest_is_rejected(self) -> None:
        prepared, envelope = _terminal_readable_envelope()
        identity = envelope["final_frontier_product"][
            "terminal_support_identities"
        ][0]
        identity["source_state_digest"] = "0" * 64

        self.assertFalse(
            verify_writer_snapshot_prefix_read_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_changed_terminal_support_finalized_digest_is_rejected(
        self,
    ) -> None:
        prepared, envelope = _terminal_readable_envelope()
        identity = envelope["final_frontier_product"][
            "terminal_support_identities"
        ][0]
        identity["finalized_state_digest"] = "1" * 64

        self.assertFalse(
            verify_writer_snapshot_prefix_read_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_changed_terminal_parent_weight_is_rejected(self) -> None:
        prepared, envelope = _terminal_readable_envelope()
        identity = envelope["final_frontier_product"][
            "terminal_support_identities"
        ][0]
        identity["parent_weight"] += 1

        self.assertFalse(
            verify_writer_snapshot_prefix_read_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_changed_terminal_projection_digest_is_rejected(self) -> None:
        prepared, envelope = _terminal_readable_envelope()
        envelope["final_frontier_product"]["terminal_projection_certificate"][
            "digest"
        ] = "2" * 64

        self.assertFalse(
            verify_writer_snapshot_prefix_read_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_changed_final_frontier_product_digest_is_rejected(self) -> None:
        envelope = _readable_envelope()
        envelope["final_frontier_product"]["digest"] = "3" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_checked_frontier_digest_is_rejected(self) -> None:
        envelope = _readable_envelope()
        envelope["final_frontier_product"]["checked_frontier_certificate"][
            "digest"
        ] = "4" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_support_count_certificate_digest_is_rejected(
        self,
    ) -> None:
        envelope = _readable_envelope()
        envelope["final_frontier_product"]["support_count_certificate"][
            "digest"
        ] = "5" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_completion_count_certificate_digest_is_rejected(
        self,
    ) -> None:
        envelope = _readable_envelope()
        envelope["final_frontier_product"]["completion_count_certificate"][
            "digest"
        ] = "6" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_invalid_replay_with_prefix_certificate_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        invalid = writer_snapshot_prefix_read_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared),
            emitted_texts=("not-a-choice",),
        )
        invalid["prefix_read_certificate"] = _readable_envelope()[
            "prefix_read_certificate"
        ]

        self.assertFalse(
            verify_writer_snapshot_prefix_read_envelope(
                prepared=prepared,
                envelope=invalid,
            ).accepted
        )

    def test_blocked_replay_with_final_product_is_rejected(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        snapshot = _initial_snapshot(prepared)
        with _blocked_capability_patch(prepared, snapshot.cursor):
            envelope = writer_snapshot_prefix_read_envelope_for_emitted_texts(
                prepared=prepared,
                snapshot=snapshot,
                emitted_texts=("C",),
            )
        envelope["final_frontier_product_kind"] = "blocked"

        with _blocked_capability_patch(prepared, snapshot.cursor):
            self.assertFalse(
                verify_writer_snapshot_prefix_read_envelope(
                    prepared=prepared,
                    envelope=envelope,
                ).accepted
            )

    def test_final_frontier_blocked_stale_diagnostic_digest_is_rejected(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared)
        emitted_texts = _legal_prefix(prepared, snapshot, length=1)
        context = _final_frontier_blocked_patch(prepared, snapshot, emitted_texts)
        with context:
            envelope = writer_snapshot_prefix_read_envelope_for_emitted_texts(
                prepared=prepared,
                snapshot=snapshot,
                emitted_texts=emitted_texts,
            )
            envelope["final_frontier_product"][
                "diagnostic_certificate_digest"
            ] = "7" * 64
            self.assertFalse(
                verify_writer_snapshot_prefix_read_envelope(
                    prepared=prepared,
                    envelope=envelope,
                ).accepted
            )

    def test_final_frontier_blocked_with_legal_product_kind_is_rejected(
        self,
    ) -> None:
        prepared = _prepare(tetrahedral_facts())
        snapshot = _initial_snapshot(prepared)
        emitted_texts = _legal_prefix(prepared, snapshot, length=1)

        with _final_frontier_blocked_patch(prepared, snapshot, emitted_texts):
            envelope = writer_snapshot_prefix_read_envelope_for_emitted_texts(
                prepared=prepared,
                snapshot=snapshot,
                emitted_texts=emitted_texts,
            )
            envelope["final_frontier_product_kind"] = "legal"
            self.assertFalse(
                verify_writer_snapshot_prefix_read_envelope(
                    prepared=prepared,
                    envelope=envelope,
                ).accepted
            )

    def test_prefix_verifier_calls_replay_source_lookup_once(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = _readable_envelope(prepared=prepared)
        calls = 0
        original = replay_module._source_snapshot_from_envelope

        def counted_source(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        with patch.object(
            replay_module,
            "_source_snapshot_from_envelope",
            counted_source,
        ):
            verification = verify_writer_snapshot_prefix_read_envelope(
                prepared=prepared,
                envelope=envelope,
            )

        self.assertTrue(verification.accepted)
        self.assertEqual(calls, 1)


def _readable_envelope(prepared=None, snapshot=None, emitted_texts=None):
    prepared = _prepare(cco_facts()) if prepared is None else prepared
    snapshot = _initial_snapshot(prepared) if snapshot is None else snapshot
    emitted_texts = () if emitted_texts is None else emitted_texts
    return deepcopy(
        writer_snapshot_prefix_read_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=snapshot,
            emitted_texts=emitted_texts,
        )
    )


def _verify(envelope):
    return verify_writer_snapshot_prefix_read_envelope(
        prepared=_prepare(cco_facts()),
        envelope=envelope,
    )


def _legal_prefix(prepared, snapshot, *, length: int) -> tuple[str, ...]:
    emitted: list[str] = []
    current = snapshot
    for _ in range(length):
        choice = writer_frontier_choices(prepared, current.cursor).choices[0]
        emitted.append(choice.emitted_text)
        advance = writer_snapshot_advance_envelope_for_emitted_text(
            prepared=prepared,
            snapshot=current,
            emitted_text=choice.emitted_text,
        )
        verification = verify_writer_snapshot_advance_envelope(
            prepared=prepared,
            envelope=advance,
        )
        if not verification.accepted or verification.advanced_snapshot is None:
            raise AssertionError("legal prefix helper failed to advance")
        current = verification.advanced_snapshot
    return tuple(emitted)


def _terminal_readable_envelope():
    from tests.south_star1.test_writer_snapshot import two_atom_facts

    prepared = _prepare(two_atom_facts())
    snapshot = _initial_snapshot(prepared)
    emitted_texts = _legal_prefix(prepared, snapshot, length=2)
    envelope = _readable_envelope(
        prepared=prepared,
        snapshot=snapshot,
        emitted_texts=emitted_texts,
    )
    if envelope["public_frontier"]["terminal"] is None:
        raise AssertionError("expected a terminal-capable prefix")
    return prepared, envelope


def _blocked_capability_patch(prepared, cursor):
    product = writer_frontier_module._snapshot_advance_writer_frontier_product(
        prepared,
        cursor,
    )
    capability = next(
        capability
        for support in product.branch_supports
        for capability in support.execution_capabilities
    )

    def unsupported(capabilities):
        return frozenset(item for item in capabilities if item is capability)

    return patch.object(
        writer_frontier_module,
        "_unsupported_public_writer_execution_capabilities",
        unsupported,
    )


def _final_frontier_blocked_patch(prepared, snapshot, emitted_texts):
    current = snapshot
    for emitted_text in emitted_texts:
        verification = verify_writer_snapshot_advance_envelope(
            prepared=prepared,
            envelope=writer_snapshot_advance_envelope_for_emitted_text(
                prepared=prepared,
                snapshot=current,
                emitted_text=emitted_text,
            ),
        )
        current = verification.advanced_snapshot
    with _blocked_capability_patch(prepared, current.cursor):
        blocked_product = (
            writer_frontier_module._snapshot_advance_writer_frontier_product(
                prepared,
                current.cursor,
            )
        )
    original = prefix_module._snapshot_advance_writer_frontier_product

    def patched_product(product_prepared, cursor):
        if product_prepared is prepared and cursor == current.cursor:
            return blocked_product
        return original(product_prepared, cursor)

    return patch.object(
        prefix_module,
        "_snapshot_advance_writer_frontier_product",
        patched_product,
    )


def _json_round_trip(envelope):
    return json.loads(json.dumps(envelope, sort_keys=True))


def _initial_snapshot(prepared):
    options = _writer_options()
    return capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=initial_writer_frontier_cursor(prepared, options),
    )


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _writer_options() -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


if __name__ == "__main__":
    unittest.main()
