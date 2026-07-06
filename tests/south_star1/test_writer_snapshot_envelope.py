"""Durable writer snapshot advance envelope tests."""

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
from grimace._south_star1.writer_snapshot import (
    _checked_writer_snapshot_text_projection_lookup,
)
from grimace._south_star1.writer_snapshot_envelope import (
    verify_writer_snapshot_advance_envelope,
)
from grimace._south_star1.writer_snapshot_envelope import (
    writer_snapshot_advance_envelope_for_emitted_text,
)
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import tetrahedral_facts


class WriterSnapshotEnvelopeTest(unittest.TestCase):
    def test_legal_advance_envelope_json_round_trips(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared)
        emitted_text = writer_frontier_choices(
            prepared,
            snapshot.cursor,
        ).choices[0].emitted_text

        envelope = _json_round_trip(
            writer_snapshot_advance_envelope_for_emitted_text(
                prepared=prepared,
                snapshot=snapshot,
                emitted_text=emitted_text,
            )
        )
        verification = verify_writer_snapshot_advance_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.outcome_kind, "advanced")
        self.assertIsNotNone(verification.advanced_snapshot)
        self.assertEqual(
            envelope["advance_certificate"]["selected_text_projection"][
                "emitted_text"
            ],
            emitted_text,
        )

    def test_invalid_text_envelope_json_round_trips(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared)

        envelope = _json_round_trip(
            writer_snapshot_advance_envelope_for_emitted_text(
                prepared=prepared,
                snapshot=snapshot,
                emitted_text="not-a-choice",
            )
        )
        verification = verify_writer_snapshot_advance_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.outcome_kind, "invalid_emitted_text")
        self.assertNotIn(
            "not-a-choice",
            envelope["advance_certificate"]["projected_emitted_texts"],
        )

    def test_blocked_envelope_json_round_trips(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        snapshot = _initial_snapshot(prepared)
        product = _checked_writer_snapshot_text_projection_lookup(
            snapshot,
            prepared=prepared,
            emitted_text=writer_frontier_choices(
                prepared,
                snapshot.cursor,
            ).choices[0].emitted_text,
        ).product
        capability = next(
            capability
            for support in product.branch_supports
            for capability in support.execution_capabilities
        )

        def unsupported(capabilities):
            return frozenset(
                item for item in capabilities if item is capability
            )

        with patch.object(
            writer_frontier_module,
            "_unsupported_public_writer_execution_capabilities",
            unsupported,
        ):
            envelope = _json_round_trip(
                writer_snapshot_advance_envelope_for_emitted_text(
                    prepared=prepared,
                    snapshot=snapshot,
                    emitted_text="C",
                )
            )
            verification = verify_writer_snapshot_advance_envelope(
                prepared=prepared,
                envelope=envelope,
            )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.outcome_kind, "blocked")
        self.assertEqual(envelope["frontier_product_kind"], "blocked")
        self.assertTrue(
            envelope["advance_certificate"]["diagnostic_certificate"][
                "unsupported_execution_capabilities"
            ]
        )

    def test_unknown_schema_name_is_rejected(self) -> None:
        envelope = _legal_envelope()
        envelope["schema_name"] = "other"

        self.assertFalse(
            verify_writer_snapshot_advance_envelope(
                prepared=_prepare(cco_facts()),
                envelope=envelope,
            ).accepted
        )

    def test_unknown_schema_version_is_rejected(self) -> None:
        envelope = _legal_envelope()
        envelope["schema_version"] = 999

        self.assertFalse(
            verify_writer_snapshot_advance_envelope(
                prepared=_prepare(cco_facts()),
                envelope=envelope,
            ).accepted
        )

    def test_extra_top_level_field_is_rejected(self) -> None:
        envelope = _legal_envelope()
        envelope["extra"] = {}

        self.assertFalse(
            verify_writer_snapshot_advance_envelope(
                prepared=_prepare(cco_facts()),
                envelope=envelope,
            ).accepted
        )

    def test_missing_required_field_is_rejected(self) -> None:
        envelope = _legal_envelope()
        del envelope["advance_certificate"]

        self.assertFalse(
            verify_writer_snapshot_advance_envelope(
                prepared=_prepare(cco_facts()),
                envelope=envelope,
            ).accepted
        )

    def test_wrong_prepared_identity_is_rejected(self) -> None:
        envelope = _legal_envelope()

        self.assertFalse(
            verify_writer_snapshot_advance_envelope(
                prepared=_prepare(tetrahedral_facts()),
                envelope=envelope,
            ).accepted
        )

    def test_wrong_source_cursor_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = _legal_envelope(prepared=prepared)
        envelope["source_snapshot"]["cursor"]["digest"] = "0" * 64

        self.assertFalse(
            verify_writer_snapshot_advance_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_wrong_emitted_text_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = _legal_envelope(prepared=prepared)
        envelope["emitted_text"] = "not-a-choice"

        self.assertFalse(
            verify_writer_snapshot_advance_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_wrong_outcome_kind_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = _legal_envelope(prepared=prepared)
        envelope["outcome_kind"] = "blocked"

        self.assertFalse(
            verify_writer_snapshot_advance_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_advanced_changed_successor_cursor_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = _legal_envelope(prepared=prepared)
        step = envelope["advance_certificate"]["step_certificate"]
        step["successor_cursor"]["digest"] = "1" * 64

        self.assertFalse(
            verify_writer_snapshot_advance_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_advanced_stale_selected_projection_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = _legal_envelope(prepared=prepared)
        selected = envelope["advance_certificate"]["selected_text_projection"]
        selected["digest"] = "2" * 64

        self.assertFalse(
            verify_writer_snapshot_advance_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_invalid_text_changed_to_legal_projection_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared)
        legal_text = writer_frontier_choices(
            prepared,
            snapshot.cursor,
        ).choices[0].emitted_text
        envelope = writer_snapshot_advance_envelope_for_emitted_text(
            prepared=prepared,
            snapshot=snapshot,
            emitted_text="not-a-choice",
        )
        envelope["emitted_text"] = legal_text

        self.assertFalse(
            verify_writer_snapshot_advance_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_blocked_legal_product_shape_is_rejected(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        envelope = _blocked_envelope(prepared)
        envelope["frontier_product_kind"] = "legal"

        with _blocked_capability_patch(prepared):
            self.assertFalse(
                verify_writer_snapshot_advance_envelope(
                    prepared=prepared,
                    envelope=envelope,
                ).accepted
            )

    def test_blocked_changed_cursor_identity_is_rejected(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        envelope = _blocked_envelope(prepared)
        blocked = envelope["advance_certificate"][
            "blocked_frontier_certificate"
        ]
        blocked["cursor"]["digest"] = "3" * 64

        with _blocked_capability_patch(prepared):
            self.assertFalse(
                verify_writer_snapshot_advance_envelope(
                    prepared=prepared,
                    envelope=envelope,
                ).accepted
            )

    def test_envelope_schema_does_not_mention_choice_snapshot(self) -> None:
        envelope = _legal_envelope()
        self.assertNotIn("choice_snapshot", json.dumps(envelope))


def _legal_envelope(prepared=None):
    prepared = _prepare(cco_facts()) if prepared is None else prepared
    snapshot = _initial_snapshot(prepared)
    emitted_text = writer_frontier_choices(
        prepared,
        snapshot.cursor,
    ).choices[0].emitted_text
    return deepcopy(
        writer_snapshot_advance_envelope_for_emitted_text(
            prepared=prepared,
            snapshot=snapshot,
            emitted_text=emitted_text,
        )
    )


def _blocked_envelope(prepared):
    snapshot = _initial_snapshot(prepared)
    with _blocked_capability_patch(prepared):
        return deepcopy(
            writer_snapshot_advance_envelope_for_emitted_text(
                prepared=prepared,
                snapshot=snapshot,
                emitted_text="C",
            )
        )


def _blocked_capability_patch(prepared):
    snapshot = _initial_snapshot(prepared)
    product = _checked_writer_snapshot_text_projection_lookup(
        snapshot,
        prepared=prepared,
        emitted_text=writer_frontier_choices(
            prepared,
            snapshot.cursor,
        ).choices[0].emitted_text,
    ).product
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
