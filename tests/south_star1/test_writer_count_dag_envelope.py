"""Bounded writer count-certificate DAG envelope tests."""

from __future__ import annotations

from copy import deepcopy
import json
import unittest

from grimace._south_star1.writer_count_dag_envelope import (
    WriterEnvelopeWorkBudget,
)
from grimace._south_star1.writer_count_dag_envelope import (
    WriterCountDagBuildDiagnostics,
)
from grimace._south_star1.writer_count_dag_envelope import (
    WriterEnvelopeWorkExceeded,
)
from grimace._south_star1.writer_count_dag_envelope import count_dag_node_by_id
from grimace._south_star1.writer_count_dag_envelope import (
    validate_writer_count_certificate_dag_envelope,
)
from grimace._south_star1.writer_count_dag_envelope import (
    writer_count_certificate_dag_envelope_for_product,
)
from grimace._south_star1.writer_envelope_consistency import (
    verify_writer_support_image_envelope_consistency,
)
from grimace._south_star1.writer_frontier import _checked_writer_frontier_product
from grimace._south_star1.writer_frontier_count_envelope import (
    verify_writer_frontier_count_envelope,
)
from grimace._south_star1.writer_frontier_count_envelope import (
    writer_frontier_count_envelope_for_prefix_read,
)
from grimace._south_star1.writer_frontier_count_envelope import (
    writer_frontier_count_envelope_for_snapshot,
)
from grimace._south_star1.writer_support_image_envelope import (
    verify_writer_support_image_envelope,
)
from grimace._south_star1.writer_support_image_envelope import (
    writer_support_image_envelope_for_snapshot,
)
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.test_writer_frontier_count_envelope import (
    _initial_snapshot,
)
from tests.south_star1.test_writer_frontier_count_envelope import _legal_prefix
from tests.south_star1.test_writer_frontier_count_envelope import _prepare
from tests.south_star1.test_writer_frontier_count_envelope import (
    _terminal_prefix_read_envelope,
)
from grimace._south_star1.writer_snapshot_prefix_envelope import (
    writer_snapshot_prefix_read_envelope_for_emitted_texts,
)


class WriterCountDagEnvelopeTest(unittest.TestCase):
    def test_count_dag_envelope_validates_for_initial_snapshot(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = writer_frontier_count_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared),
        )
        dag = envelope["count_dag"]

        validate_writer_count_certificate_dag_envelope(dag)
        nodes = count_dag_node_by_id(dag)

        self.assertEqual(
            envelope["support_count_certificate"],
            nodes[dag["roots"]["support_count_root"]],
        )
        self.assertEqual(
            envelope["completion_count_certificate"],
            nodes[dag["roots"]["completion_count_root"]],
        )
        self.assertGreater(dag["metrics"]["node_count"], 0)
        self.assertGreater(dag["metrics"]["edge_count"], 0)

    def test_count_dag_envelope_validates_for_prefix_read_source(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared)
        prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=snapshot,
            emitted_texts=_legal_prefix(prepared, snapshot, length=1),
        )
        envelope = writer_frontier_count_envelope_for_prefix_read(
            prepared=prepared,
            prefix_read_envelope=prefix,
        )

        validate_writer_count_certificate_dag_envelope(envelope["count_dag"])
        self.assertTrue(
            verify_writer_frontier_count_envelope(
                prepared=prepared,
                envelope=json.loads(json.dumps(envelope, sort_keys=True)),
            ).accepted
        )

    def test_terminal_frontier_has_terminal_choice_count_node(self) -> None:
        prepared, prefix = _terminal_prefix_read_envelope()
        envelope = writer_frontier_count_envelope_for_prefix_read(
            prepared=prepared,
            prefix_read_envelope=prefix,
        )
        dag = envelope["count_dag"]
        terminal_root = dag["roots"]["terminal_choice_count_root"]

        self.assertIsNotNone(terminal_root)
        self.assertEqual(
            envelope["terminal_choice_count_certificate"],
            count_dag_node_by_id(dag)[terminal_root],
        )
        self.assertEqual(
            envelope["coverage"]["terminal_choice_coverage"][
                "terminal_choice_count_node_id"
            ],
            terminal_root,
        )

    def test_branching_frontier_has_branch_completion_nodes(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        envelope = writer_frontier_count_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared),
        )
        branch_nodes = [
            node
            for node in envelope["count_dag"]["nodes"]
            if node["kind"] == "writer_branch_completion_term"
        ]

        self.assertGreater(len(branch_nodes), 0)
        self.assertGreater(
            len(envelope["coverage"]["branch_terms_covered"]),
            0,
        )

    def test_count_dag_diagnostics_report_shared_subproof_hits(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        diagnostics = WriterCountDagBuildDiagnostics()
        envelope = writer_frontier_count_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared),
            count_dag_diagnostics=diagnostics,
        )

        self.assertGreater(
            diagnostics.attempted_node_emissions,
            envelope["count_dag"]["metrics"]["node_count"],
        )
        self.assertGreater(diagnostics.dedup_hits, 0)

    def test_support_image_and_consistency_verifiers_accept_dag_count_envelope(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        image = writer_support_image_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared),
        )

        self.assertTrue(
            verify_writer_support_image_envelope(
                prepared=prepared,
                envelope=image,
            ).accepted
        )
        self.assertTrue(
            verify_writer_support_image_envelope_consistency(image).accepted
        )

    def test_unknown_count_dag_schema_is_rejected(self) -> None:
        envelope = _count_envelope()
        envelope["count_dag"]["schema_name"] = "other"

        self.assertFalse(_verify(envelope).accepted)

    def test_extra_count_dag_field_is_rejected(self) -> None:
        envelope = _count_envelope()
        envelope["count_dag"]["extra"] = {}

        self.assertFalse(_verify(envelope).accepted)

    def test_missing_root_is_rejected(self) -> None:
        envelope = _count_envelope()
        envelope["count_dag"]["roots"]["support_count_root"] = "missing"

        self.assertFalse(_verify(envelope).accepted)

    def test_missing_child_node_is_rejected(self) -> None:
        envelope = _count_envelope()
        child_id = _first_node_with_children(envelope["count_dag"])["children"][0]
        envelope["count_dag"]["nodes"] = [
            node
            for node in envelope["count_dag"]["nodes"]
            if node["node_id"] != child_id
        ]

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_node_kind_is_rejected(self) -> None:
        envelope = _count_envelope()
        envelope["count_dag"]["nodes"][0]["kind"] = "other"

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_node_digest_is_rejected(self) -> None:
        envelope = _count_envelope()
        envelope["count_dag"]["nodes"][0]["digest"] = "0" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_choice_coverage_missing_node_is_rejected(self) -> None:
        envelope = _count_envelope()
        envelope["coverage"]["text_choices_covered"][0][
            "completion_count_node_id"
        ] = "missing"

        self.assertFalse(_verify(envelope).accepted)

    def test_terminal_coverage_missing_node_is_rejected(self) -> None:
        prepared, prefix = _terminal_prefix_read_envelope()
        envelope = writer_frontier_count_envelope_for_prefix_read(
            prepared=prepared,
            prefix_read_envelope=prefix,
        )
        envelope["coverage"]["terminal_choice_coverage"][
            "terminal_choice_count_node_id"
        ] = "missing"

        self.assertFalse(
            verify_writer_frontier_count_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_cycle_is_rejected(self) -> None:
        envelope = _count_envelope()
        node = envelope["count_dag"]["nodes"][0]
        node["children"].append(node["node_id"])

        self.assertFalse(_verify(envelope).accepted)

    def test_node_count_budget_exceeded_is_typed(self) -> None:
        product = _product()

        with self.assertRaises(WriterEnvelopeWorkExceeded) as context:
            writer_count_certificate_dag_envelope_for_product(
                product,
                budget=WriterEnvelopeWorkBudget(max_count_nodes=0),
            )

        self.assertEqual(context.exception.violation.metric, "count_node_count")

    def test_edge_count_budget_exceeded_is_typed(self) -> None:
        product = _product()

        with self.assertRaises(WriterEnvelopeWorkExceeded) as context:
            writer_count_certificate_dag_envelope_for_product(
                product,
                budget=WriterEnvelopeWorkBudget(max_count_edges=0),
            )

        self.assertEqual(context.exception.violation.metric, "count_edge_count")

    def test_depth_budget_exceeded_is_typed(self) -> None:
        product = _product()

        with self.assertRaises(WriterEnvelopeWorkExceeded) as context:
            writer_count_certificate_dag_envelope_for_product(
                product,
                budget=WriterEnvelopeWorkBudget(max_count_depth=0),
            )

        self.assertEqual(context.exception.violation.metric, "count_depth")

    def test_digest_term_byte_budget_exceeded_is_typed(self) -> None:
        product = _product()

        with self.assertRaises(WriterEnvelopeWorkExceeded) as context:
            writer_count_certificate_dag_envelope_for_product(
                product,
                budget=WriterEnvelopeWorkBudget(max_digest_term_bytes=1),
            )

        self.assertEqual(context.exception.violation.metric, "digest_term_bytes")


def _count_envelope():
    prepared = _prepare(cco_facts())
    return deepcopy(
        writer_frontier_count_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared),
        )
    )


def _verify(envelope):
    return verify_writer_frontier_count_envelope(
        prepared=_prepare(cco_facts()),
        envelope=envelope,
    )


def _product():
    prepared = _prepare(cco_facts())
    snapshot = _initial_snapshot(prepared)
    return _checked_writer_frontier_product(
        prepared,
        snapshot.cursor,
        include_counts=True,
        include_frontier_certificate=True,
        include_count_certificate=True,
    )


def _first_node_with_children(dag):
    for node in dag["nodes"]:
        if node["children"]:
            return node
    raise AssertionError("expected count DAG node with children")


if __name__ == "__main__":
    unittest.main()
