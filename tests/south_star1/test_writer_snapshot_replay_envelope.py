"""Durable writer snapshot replay envelope tests."""

from __future__ import annotations

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import grimace._south_star1.writer_frontier as writer_frontier_module
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
from grimace._south_star1.writer_snapshot_replay_envelope import (
    verify_writer_snapshot_replay_envelope,
)
from grimace._south_star1.writer_snapshot_replay_envelope import (
    writer_snapshot_replay_envelope_for_emitted_texts,
)
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import tetrahedral_facts


class WriterSnapshotReplayEnvelopeTest(unittest.TestCase):
    def test_empty_replay_envelope_json_round_trips(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = _json_round_trip(
            writer_snapshot_replay_envelope_for_emitted_texts(
                prepared=prepared,
                snapshot=_initial_snapshot(prepared),
                emitted_texts=(),
            )
        )

        verification = verify_writer_snapshot_replay_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.outcome_kind, "advanced")
        self.assertEqual(envelope["step_advance_envelopes"], [])
        self.assertIsNotNone(envelope["replay_certificate"])

    def test_multi_step_replay_envelope_json_round_trips(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared)
        emitted_texts = _legal_prefix(prepared, snapshot, length=2)
        envelope = _json_round_trip(
            writer_snapshot_replay_envelope_for_emitted_texts(
                prepared=prepared,
                snapshot=snapshot,
                emitted_texts=emitted_texts,
            )
        )

        verification = verify_writer_snapshot_replay_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.outcome_kind, "advanced")
        self.assertEqual(envelope["consumed_emitted_texts"], list(emitted_texts))
        self.assertIsNotNone(verification.current_snapshot)

    def test_invalid_text_at_first_step_verifies(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = _json_round_trip(
            writer_snapshot_replay_envelope_for_emitted_texts(
                prepared=prepared,
                snapshot=_initial_snapshot(prepared),
                emitted_texts=("not-a-choice",),
            )
        )

        verification = verify_writer_snapshot_replay_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.outcome_kind, "invalid_emitted_text")
        self.assertEqual(verification.failed_step_index, 0)
        self.assertIsNotNone(envelope["failed_advance_envelope"])

    def test_invalid_text_after_advanced_step_verifies(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared)
        first = _legal_prefix(prepared, snapshot, length=1)
        envelope = _json_round_trip(
            writer_snapshot_replay_envelope_for_emitted_texts(
                prepared=prepared,
                snapshot=snapshot,
                emitted_texts=(*first, "not-a-choice"),
            )
        )

        verification = verify_writer_snapshot_replay_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.outcome_kind, "invalid_emitted_text")
        self.assertEqual(verification.failed_step_index, 1)

    def test_blocked_replay_envelope_json_round_trips(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        snapshot = _initial_snapshot(prepared)

        with _blocked_capability_patch(prepared):
            envelope = _json_round_trip(
                writer_snapshot_replay_envelope_for_emitted_texts(
                    prepared=prepared,
                    snapshot=snapshot,
                    emitted_texts=("C",),
                )
            )
            verification = verify_writer_snapshot_replay_envelope(
                prepared=prepared,
                envelope=envelope,
            )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.outcome_kind, "blocked")
        self.assertEqual(verification.failed_step_index, 0)

    def test_unknown_schema_name_is_rejected(self) -> None:
        envelope = _advanced_envelope()
        envelope["schema_name"] = "other"

        self.assertFalse(
            verify_writer_snapshot_replay_envelope(
                prepared=_prepare(cco_facts()),
                envelope=envelope,
            ).accepted
        )

    def test_extra_top_level_field_is_rejected(self) -> None:
        envelope = _advanced_envelope()
        envelope["extra"] = {}

        self.assertFalse(
            verify_writer_snapshot_replay_envelope(
                prepared=_prepare(cco_facts()),
                envelope=envelope,
            ).accepted
        )

    def test_wrong_source_decoder_boundary_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = _advanced_envelope(prepared=prepared)
        envelope["source_snapshot"]["decoder_boundary"][
            "consumed_token_count"
        ] += 1

        self.assertFalse(
            verify_writer_snapshot_replay_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_wrong_emitted_texts_are_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = _advanced_envelope(prepared=prepared)
        envelope["emitted_texts"] = ["not-a-choice"]

        self.assertFalse(
            verify_writer_snapshot_replay_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_step_envelope_order_swap_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared)
        emitted_texts = _legal_prefix(prepared, snapshot, length=2)
        envelope = _advanced_envelope(
            prepared=prepared,
            snapshot=snapshot,
            emitted_texts=emitted_texts,
        )
        envelope["step_advance_envelopes"] = [
            envelope["step_advance_envelopes"][1],
            envelope["step_advance_envelopes"][0],
        ]

        self.assertFalse(
            verify_writer_snapshot_replay_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_step_source_chain_mismatch_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared)
        emitted_texts = _legal_prefix(prepared, snapshot, length=2)
        envelope = _advanced_envelope(
            prepared=prepared,
            snapshot=snapshot,
            emitted_texts=emitted_texts,
        )
        envelope["step_advance_envelopes"][1]["source_snapshot"] = (
            envelope["source_snapshot"]
        )

        self.assertFalse(
            verify_writer_snapshot_replay_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_replay_certificate_step_count_mismatch_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = _advanced_envelope(prepared=prepared)
        envelope["replay_certificate"]["step_certificate_digests"] = []

        self.assertFalse(
            verify_writer_snapshot_replay_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_advanced_replay_with_failed_step_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared)
        envelope = _advanced_envelope(prepared=prepared, snapshot=snapshot)
        envelope["failed_advance_envelope"] = (
            writer_snapshot_advance_envelope_for_emitted_text(
                prepared=prepared,
                snapshot=snapshot,
                emitted_text="not-a-choice",
            )
        )

        self.assertFalse(
            verify_writer_snapshot_replay_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_invalid_replay_with_additional_step_after_failure_is_rejected(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared)
        invalid = writer_snapshot_replay_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=snapshot,
            emitted_texts=("not-a-choice",),
        )
        invalid["step_advance_envelopes"] = [
            writer_snapshot_advance_envelope_for_emitted_text(
                prepared=prepared,
                snapshot=snapshot,
                emitted_text=_legal_prefix(prepared, snapshot, length=1)[0],
            )
        ]

        self.assertFalse(
            verify_writer_snapshot_replay_envelope(
                prepared=prepared,
                envelope=invalid,
            ).accepted
        )

    def test_blocked_replay_with_legal_failed_shape_is_rejected(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        snapshot = _initial_snapshot(prepared)
        with _blocked_capability_patch(prepared):
            blocked = writer_snapshot_replay_envelope_for_emitted_texts(
                prepared=prepared,
                snapshot=snapshot,
                emitted_texts=("C",),
            )
        legal_prepared = _prepare(cco_facts())
        legal_snapshot = _initial_snapshot(legal_prepared)
        blocked["failed_advance_envelope"] = (
            writer_snapshot_advance_envelope_for_emitted_text(
                prepared=legal_prepared,
                snapshot=legal_snapshot,
                emitted_text=_legal_prefix(
                    legal_prepared,
                    legal_snapshot,
                    length=1,
                )[0],
            )
        )

        with _blocked_capability_patch(prepared):
            self.assertFalse(
                verify_writer_snapshot_replay_envelope(
                    prepared=prepared,
                    envelope=blocked,
                ).accepted
            )

    def test_consumed_remaining_partition_mismatch_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = writer_snapshot_replay_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared),
            emitted_texts=("not-a-choice",),
        )
        envelope["consumed_emitted_texts"] = ["not-a-choice"]

        self.assertFalse(
            verify_writer_snapshot_replay_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_empty_replay_with_non_source_current_snapshot_is_rejected(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared)
        emitted_text = _legal_prefix(prepared, snapshot, length=1)[0]
        advanced = writer_snapshot_advance_envelope_for_emitted_text(
            prepared=prepared,
            snapshot=snapshot,
            emitted_text=emitted_text,
        )
        envelope = writer_snapshot_replay_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=snapshot,
            emitted_texts=(),
        )
        envelope["current_snapshot"] = advanced["advance_certificate"][
            "advanced_snapshot"
        ]

        self.assertFalse(
            verify_writer_snapshot_replay_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )


def _advanced_envelope(prepared=None, snapshot=None, emitted_texts=None):
    prepared = _prepare(cco_facts()) if prepared is None else prepared
    snapshot = _initial_snapshot(prepared) if snapshot is None else snapshot
    emitted_texts = (
        _legal_prefix(prepared, snapshot, length=1)
        if emitted_texts is None
        else emitted_texts
    )
    return deepcopy(
        writer_snapshot_replay_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=snapshot,
            emitted_texts=emitted_texts,
        )
    )


def _legal_prefix(prepared, snapshot, *, length: int) -> tuple[str, ...]:
    emitted: list[str] = []
    current = snapshot
    for _ in range(length):
        choice = writer_frontier_choices(
            prepared,
            current.cursor,
        ).choices[0]
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


def _blocked_capability_patch(prepared):
    snapshot = _initial_snapshot(prepared)
    product = writer_frontier_module._snapshot_advance_writer_frontier_product(
        prepared,
        snapshot.cursor,
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
