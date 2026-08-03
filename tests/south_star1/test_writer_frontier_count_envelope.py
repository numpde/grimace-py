"""Durable writer frontier count envelope tests."""

from __future__ import annotations

from tests.south_star1.writer_test_context import initial_writer_snapshot
from tests.south_star1.writer_test_context import prepare_writer_facts
from tests.south_star1.writer_test_context import writer_runtime_options

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_frontier_count_envelope import (
    verify_writer_frontier_count_envelope,
)
from grimace._south_star1.writer_frontier_count_envelope import (
    writer_frontier_count_envelope_for_prefix_read,
)
from grimace._south_star1.writer_frontier_count_envelope import (
    writer_frontier_count_envelope_for_snapshot,
)
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
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import tetrahedral_facts
from tests.south_star1.helpers import two_atom_facts


class WriterFrontierCountEnvelopeTest(unittest.TestCase):
    def test_verification_does_not_rebuild_count_dag(self) -> None:
        prepared = prepare_writer_facts(cco_facts())
        snapshot = initial_writer_snapshot(prepared, writer_runtime_options())
        envelope = writer_frontier_count_envelope_for_snapshot(
            prepared=prepared, snapshot=snapshot
        )
        with patch(
            "grimace._south_star1.writer_frontier_count_envelope.writer_count_certificate_dag_envelope_for_product",
            side_effect=AssertionError("count DAG regenerated"),
        ):
            verification = verify_writer_frontier_count_envelope(
                prepared=prepared, envelope=envelope
            )
        self.assertTrue(verification.accepted, verification.reason)

    def test_initial_snapshot_count_envelope_json_round_trips(self) -> None:
        prepared = prepare_writer_facts(cco_facts())
        envelope = _json_round_trip(
            writer_frontier_count_envelope_for_snapshot(
                prepared=prepared,
                snapshot=initial_writer_snapshot(prepared, writer_runtime_options()),
            )
        )

        verification = verify_writer_frontier_count_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.source_kind, "snapshot")
        self.assertEqual(verification.support_count, envelope["support_count"])
        self.assertEqual(
            verification.completion_count,
            envelope["completion_count"],
        )

    def test_prefix_read_count_envelope_json_round_trips(self) -> None:
        prepared = prepare_writer_facts(cco_facts())
        snapshot = initial_writer_snapshot(prepared, writer_runtime_options())
        prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=snapshot,
            emitted_texts=_legal_prefix(prepared, snapshot, length=1),
        )
        envelope = _json_round_trip(
            writer_frontier_count_envelope_for_prefix_read(
                prepared=prepared,
                prefix_read_envelope=prefix,
            )
        )

        verification = verify_writer_frontier_count_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.source_kind, "prefix_read")

    def test_terminal_frontier_count_envelope_json_round_trips(self) -> None:
        prepared, prefix = _terminal_prefix_read_envelope()
        envelope = _json_round_trip(
            writer_frontier_count_envelope_for_prefix_read(
                prepared=prepared,
                prefix_read_envelope=prefix,
            )
        )

        verification = verify_writer_frontier_count_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertTrue(envelope["coverage"]["terminal_covered"])
        self.assertIsNotNone(envelope["coverage"]["terminal_choice_coverage"])

    def test_branching_frontier_count_envelope_verifies(self) -> None:
        prepared = prepare_writer_facts(cyclopropane_facts())
        envelope = writer_frontier_count_envelope_for_snapshot(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, writer_runtime_options()),
        )

        self.assertTrue(
            verify_writer_frontier_count_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )
        self.assertGreater(len(envelope["coverage"]["branch_terms_covered"]), 0)

    def test_no_terminal_frontier_count_envelope_verifies(self) -> None:
        prepared = prepare_writer_facts(cco_facts())
        envelope = writer_frontier_count_envelope_for_snapshot(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, writer_runtime_options()),
        )

        self.assertTrue(
            verify_writer_frontier_count_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )
        self.assertFalse(envelope["coverage"]["terminal_covered"])
        self.assertIsNone(envelope["coverage"]["terminal_choice_coverage"])

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
        del envelope["coverage"]

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_prepared_identity_is_rejected(self) -> None:
        envelope = _snapshot_envelope()

        self.assertFalse(
            verify_writer_frontier_count_envelope(
                prepared=prepare_writer_facts(tetrahedral_facts()),
                envelope=envelope,
            ).accepted
        )

    def test_wrong_source_kind_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["source_kind"] = "choice_snapshot"

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_snapshot_cursor_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["source_snapshot"]["cursor"]["digest"] = "0" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_prefix_read_envelope_is_rejected(self) -> None:
        prepared = prepare_writer_facts(cco_facts())
        envelope = _prefix_envelope(prepared=prepared)
        envelope["prefix_read_envelope"]["schema_name"] = "bad"

        self.assertFalse(
            verify_writer_frontier_count_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_non_readable_prefix_read_source_is_rejected(self) -> None:
        prepared = prepare_writer_facts(cco_facts())
        snapshot = initial_writer_snapshot(prepared, writer_runtime_options())
        prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=snapshot,
            emitted_texts=("not-a-choice",),
        )

        with self.assertRaisesRegex(Exception, "prefix_read_envelope_not_readable"):
            writer_frontier_count_envelope_for_prefix_read(
                prepared=prepared,
                prefix_read_envelope=prefix,
            )

    def test_changed_frontier_product_digest_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["frontier_product"]["digest"] = "1" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_terminal_support_identity_is_rejected(self) -> None:
        prepared, prefix = _terminal_prefix_read_envelope()
        envelope = writer_frontier_count_envelope_for_prefix_read(
            prepared=prepared,
            prefix_read_envelope=prefix,
        )
        identity = envelope["coverage"]["terminal_choice_coverage"][
            "terminal_support_identities"
        ][0]
        identity["terminal_ordinal"] += 1

        self.assertFalse(
            verify_writer_frontier_count_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_changed_branch_support_identity_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        branch = envelope["coverage"]["branch_terms_covered"][0][
            "branch_support_identity"
        ]
        branch["source_state_digest"] = "2" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_support_count_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["support_count"] += 1

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_completion_count_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["completion_count"] += 1

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_support_count_certificate_digest_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["support_count_certificate"]["digest"] = "3" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_completion_count_certificate_digest_is_rejected(
        self,
    ) -> None:
        envelope = _snapshot_envelope()
        envelope["completion_count_certificate"]["digest"] = "4" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_removed_choice_coverage_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["coverage"]["text_choices_covered"] = (
            envelope["coverage"]["text_choices_covered"][1:]
        )

        self.assertFalse(_verify(envelope).accepted)

    def test_added_extra_choice_coverage_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["coverage"]["text_choices_covered"].append(
            deepcopy(envelope["coverage"]["text_choices_covered"][0])
        )

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_per_choice_successor_count_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["coverage"]["text_choices_covered"][0][
            "successor_support_count_digest"
        ] = "5" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_terminal_coverage_present_without_terminal_is_rejected(
        self,
    ) -> None:
        envelope = _snapshot_envelope()
        envelope["coverage"]["terminal_choice_coverage"] = {
            "support_count": 1,
            "completion_count": 1,
        }

        self.assertFalse(_verify(envelope).accepted)

    def test_terminal_coverage_missing_with_terminal_is_rejected(self) -> None:
        prepared, prefix = _terminal_prefix_read_envelope()
        envelope = writer_frontier_count_envelope_for_prefix_read(
            prepared=prepared,
            prefix_read_envelope=prefix,
        )
        envelope["coverage"]["terminal_choice_coverage"] = None

        self.assertFalse(
            verify_writer_frontier_count_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )


def _snapshot_envelope():
    prepared = prepare_writer_facts(cco_facts())
    return deepcopy(
        writer_frontier_count_envelope_for_snapshot(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, writer_runtime_options()),
        )
    )


def _prefix_envelope(*, prepared):
    snapshot = initial_writer_snapshot(prepared, writer_runtime_options())
    prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
        prepared=prepared,
        snapshot=snapshot,
        emitted_texts=_legal_prefix(prepared, snapshot, length=1),
    )
    return deepcopy(
        writer_frontier_count_envelope_for_prefix_read(
            prepared=prepared,
            prefix_read_envelope=prefix,
        )
    )


def _terminal_prefix_read_envelope():
    prepared = prepare_writer_facts(two_atom_facts())
    snapshot = initial_writer_snapshot(prepared, writer_runtime_options())
    prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
        prepared=prepared,
        snapshot=snapshot,
        emitted_texts=_legal_prefix(prepared, snapshot, length=2),
    )
    return prepared, prefix


def _verify(envelope):
    return verify_writer_frontier_count_envelope(
        prepared=prepare_writer_facts(cco_facts()),
        envelope=envelope,
    )


def _legal_prefix(prepared, snapshot, *, length: int) -> tuple[str, ...]:
    emitted: list[str] = []
    current = snapshot
    for _ in range(length):
        envelope = writer_snapshot_advance_envelope_for_emitted_text(
            prepared=prepared,
            snapshot=current,
            emitted_text=_first_choice_text(prepared, current),
        )
        verification = verify_writer_snapshot_advance_envelope(
            prepared=prepared,
            envelope=envelope,
        )
        if not verification.accepted or verification.advanced_snapshot is None:
            raise AssertionError("legal prefix helper failed to advance")
        emitted.append(envelope["emitted_text"])
        current = verification.advanced_snapshot
    return tuple(emitted)


def _first_choice_text(prepared, snapshot) -> str:
    from grimace._south_star1.writer_frontier import writer_frontier_choices

    return writer_frontier_choices(prepared, snapshot.cursor).choices[0].emitted_text


def _json_round_trip(envelope):
    return json.loads(json.dumps(envelope, sort_keys=True))



if __name__ == "__main__":
    unittest.main()
