"""Bounded writer count-certificate DAG envelope tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
import time
import unittest
from unittest.mock import patch

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
from grimace._south_star1.writer_count_dag_envelope import count_dag_manifest
from grimace._south_star1.writer_count_dag_envelope import (
    validate_writer_count_certificate_dag_envelope,
)
from grimace._south_star1.writer_count_dag_envelope import (
    writer_count_certificate_dag_envelope_for_product,
)
from grimace._south_star1.writer_count_dag_envelope import _CountDagBuilder
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
from tests.south_star1.qualification_plan import (
    selected_slow_qualification_cases,
)
from tests.south_star1.slow_qualification_assets import (
    build_slow_count_envelope,
    require_slow_count_envelope,
)
from tests.south_star1.test_writer_default_parity_corpus import _facts
from tests.south_star1.test_writer_default_parity_corpus import _initial_snapshot as _default_initial_snapshot
from tests.south_star1.test_writer_default_parity_corpus import _prepare_default
from grimace._south_star1.writer_snapshot_prefix_envelope import (
    writer_snapshot_prefix_read_envelope_for_emitted_texts,
)


class WriterCountDagEnvelopeTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1",
        "set SOUTH_STAR1_RUN_SLOW=1 to run coupled cases",
    )
    def test_slow_coupled_count_dag_build(self) -> None:
        for case in selected_slow_qualification_cases():
            with self.subTest(case=case.name):
                cached = build_slow_count_envelope(case)
                self.assertTrue(cached.envelope_path.is_file())
                self.assertTrue(cached.metadata_path.is_file())

    @unittest.skipUnless(
        os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1",
        "set SOUTH_STAR1_RUN_SLOW=1 to run coupled cases",
    )
    def test_slow_coupled_count_dag_validate(self) -> None:
        for case in selected_slow_qualification_cases():
            with self.subTest(case=case.name):
                cache_started = time.monotonic()
                cached = require_slow_count_envelope(case)
                print(f"cache_read_seconds={time.monotonic() - cache_started:.3f}", flush=True)
                envelope = json.loads(cached.envelope_path.read_text())
                prepared = _prepare_default(_facts(case))
                snapshot = _default_initial_snapshot(prepared, case.rooted_at_atom)
                with (
                    patch("grimace._south_star1.writer_frontier_count_envelope.writer_frontier_count_envelope_for_snapshot", side_effect=AssertionError("count envelope built")),
                    patch("grimace._south_star1.writer_frontier_count_envelope.writer_count_certificate_dag_envelope_for_product", side_effect=AssertionError("count DAG built")),
                ):
                    default_started = time.monotonic()
                    verification = verify_writer_frontier_count_envelope(
                        prepared=prepared, envelope=envelope,
                        budget=WriterEnvelopeWorkBudget(),
                    )
                    print(f"default_validation_seconds={time.monotonic() - default_started:.3f}", flush=True)
                    exact_started = time.monotonic()
                    validate_writer_count_certificate_dag_envelope(
                        envelope["count_dag"],
                        budget=replace(WriterEnvelopeWorkBudget(), max_count_nodes=17_698),
                    )
                    print(f"exact_boundary_validation_seconds={time.monotonic() - exact_started:.3f}", flush=True)
                    rejecting_started = time.monotonic()
                    with self.assertRaises(WriterEnvelopeWorkExceeded) as raised:
                        validate_writer_count_certificate_dag_envelope(
                            envelope["count_dag"],
                            budget=replace(WriterEnvelopeWorkBudget(), max_count_nodes=17_697),
                        )
                    print(f"rejecting_boundary_validation_seconds={time.monotonic() - rejecting_started:.3f}", flush=True)
                self.assertTrue(verification.accepted, verification.reason)
                self.assertEqual(raised.exception.violation.metric, "count_node_count")

    def test_twenty_thousand_one_synthetic_nodes_remain_rejected(self) -> None:
        builder = _CountDagBuilder(budget=WriterEnvelopeWorkBudget())
        with self.assertRaises(WriterEnvelopeWorkExceeded) as raised:
            for index in range(20_001):
                builder._node("synthetic", {"index": index}, [])
        self.assertEqual(raised.exception.violation.metric, "count_node_count")
        self.assertEqual(raised.exception.violation.limit, 20_000)

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
        self.assertLess(
            dag["metrics"]["manifest_digest_input_bytes"],
            dag["metrics"]["full_node_digest_input_bytes"],
        )
        self.assertEqual(
            dag["metrics"]["digest_input_bytes"],
            dag["metrics"]["manifest_digest_input_bytes"],
        )

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

    def test_changed_node_payload_with_stale_node_digest_is_rejected(self) -> None:
        envelope = _count_envelope()
        envelope["count_dag"]["nodes"][0]["support_count"] = 999

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_child_reference_with_stale_dag_digest_is_rejected(self) -> None:
        envelope = _count_envelope()
        node = _first_node_with_multiple_children(envelope["count_dag"])
        node["children"] = list(reversed(node["children"]))

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_metrics_with_stale_dag_digest_is_rejected(self) -> None:
        envelope = _count_envelope()
        envelope["count_dag"]["metrics"]["node_count"] += 1

        self.assertFalse(_verify(envelope).accepted)

    def test_count_dag_manifest_excludes_full_node_payloads(self) -> None:
        envelope = _count_envelope()
        dag = envelope["count_dag"]
        manifest = count_dag_manifest(dag)

        self.assertEqual(
            frozenset(manifest["nodes"][0]),
            frozenset(("node_id", "kind", "digest", "children")),
        )
        self.assertLess(
            dag["metrics"]["manifest_digest_input_bytes"],
            dag["metrics"]["full_node_digest_input_bytes"],
        )

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


def _first_node_with_multiple_children(dag):
    for node in dag["nodes"]:
        if len(node["children"]) > 1:
            return node
    raise AssertionError("expected count DAG node with multiple children")


if __name__ == "__main__":
    unittest.main()
