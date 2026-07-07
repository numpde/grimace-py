"""Gated calibration probe for aspirin-sized writer envelopes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
import json
import os
import time
import unittest
from unittest.mock import patch

from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_envelope_consistency import (
    _count_json_nodes,
)
from grimace._south_star1.writer_envelope_consistency import (
    _count_nested_envelopes,
)
from grimace._south_star1.writer_envelope_consistency import (
    verify_writer_support_image_envelope_consistency,
)
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkExceeded
from grimace._south_star1.writer_execution_evidence import (
    WriterFiniteRelationWorkEnvelope,
)
from grimace._south_star1.writer_execution_evidence import (
    WriterGraphObligationWorkEnvelope,
)
from grimace._south_star1.writer_execution_evidence import (
    WriterResidualWorkEnvelope,
)
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_count_dag_envelope import (
    WriterCountDagBuildDiagnostics,
)
from grimace._south_star1.writer_frontier_count_envelope import (
    writer_frontier_count_envelope_for_snapshot,
)
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_support_image_envelope import (
    verify_writer_support_image_envelope,
)
from grimace._south_star1.writer_support_image_envelope import (
    writer_support_image_envelope_for_snapshot,
)
import grimace._south_star1.writer_execution_evidence as writer_execution_evidence
from tests.south_star1.helpers import cco_facts


ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
RUN_SLOW_ENV = "SOUTH_STAR1_RUN_SLOW"
ASPIRIN_FACTS_SOURCE = (
    "ordinary_molecule_facts_from_smiles("
    "'CC(=O)Oc1ccccc1C(=O)O')"
)
WIDENED_WRITER_WORK_BOUNDS = {
    "graph_obligation": {
        "max_component_atom_count": 13,
        "max_component_bond_count": 13,
        "max_edge_obligation_count": 13,
        "max_residual_attachment_count": 5,
        "max_residual_attachment_action_count": 5,
        "max_boundary_incidence_count": 16,
        "max_closure_candidate_count": 16,
        "max_live_branch_return_closure_candidate_count": 16,
        "max_deferred_branch_return_closure_candidate_count": 16,
        "max_deferred_control_live_closure_candidate_count": 16,
        "max_unsupported_closure_candidate_count": 16,
        "max_open_closure_count": 16,
        "max_closed_closure_count": 16,
        "max_attachment_atom_count": 13,
        "max_attachment_boundary_count": 8,
        "max_attachment_cyclic_rank": 13,
    },
    "residual": {
        "max_component_variable_count": 64,
        "max_component_factor_count": 64,
        "max_checked_candidate_rows": 4096,
        "max_largest_factor_scope": 16,
        "max_largest_candidate_row_count": 4096,
    },
    "finite_relation": {
        "max_row_count": 64,
        "max_total_candidate_count": 256,
        "max_largest_candidate_count": 64,
    },
}
ASPIRIN_PROBE_ENVELOPE_BUDGET = WriterEnvelopeWorkBudget(
    max_count_nodes=100_000,
    max_count_edges=500_000,
    max_count_depth=10_000,
)


@dataclass(frozen=True, slots=True)
class WriterAspirinEnvelopeProbeResult:
    input_smiles: str
    facts_source: str
    widened_runtime_bounds: dict[str, object]
    high_envelope_budget: dict[str, object]
    accepted: bool
    blocked_kind: str | None
    work_violation: dict[str, object] | None
    default_budget_violation: dict[str, object] | None
    high_budget_accepted: bool | None
    high_budget_count_dag_metrics: dict[str, object] | None
    count_dag_root_ids: dict[str, object] | None
    count_dag_choice_count_root_count: int | None
    count_dag_terminal_choice_count_root_present: bool | None
    count_dag_node_kind_counts: dict[str, int] | None
    count_dag_dedup_attempts: dict[str, int] | None
    count_dag_dedup_hits: dict[str, int] | None
    count_dag_dedup_ratio: float | None
    count_dag_top_largest_nodes: tuple[dict[str, object], ...] | None
    count_dag_top_high_fanout_nodes: tuple[dict[str, object], ...] | None
    count_dag_top_deepest_nodes: tuple[dict[str, object], ...] | None
    next_blocker: dict[str, object] | str | None
    support_count: int | None
    completion_count: int | None
    support_string_count: int | None
    support_image_distinct_count: int | None
    support_image_witness_count: int | None
    count_dag_node_count: int | None
    count_dag_edge_count: int | None
    count_dag_max_depth: int | None
    count_dag_digest_input_bytes: int | None
    largest_count_node_digest_input_bytes: int | None
    support_image_total_emitted_text_bytes: int | None
    coverage_bucket_count: int | None
    bucket_assignment_count: int | None
    nested_envelope_count: int | None
    consistency_node_count: int | None
    timings: tuple[tuple[str, float], ...]


class WriterAspirinEnvelopeProbeTest(unittest.TestCase):
    def test_support_image_probe_path_low_budget_fails_typed(self) -> None:
        prepared = _prepare(cco_facts())
        budget = WriterEnvelopeWorkBudget(max_support_strings=0)

        with self.assertRaises(WriterEnvelopeWorkExceeded) as raised:
            writer_support_image_envelope_for_snapshot(
                prepared=prepared,
                snapshot=_initial_snapshot(prepared),
                budget=budget,
            )

        violation = raised.exception.violation
        self.assertEqual(violation.operation, "support_image_envelope")
        self.assertEqual(violation.metric, "support_string_count")
        self.assertGreater(violation.actual, violation.limit)
        self.assertIn("WRITER_ENVELOPE_WORK_EXCEEDED", str(raised.exception))

    def test_aspirin_support_image_envelope_probe(self) -> None:
        if os.environ.get(RUN_SLOW_ENV) != "1":
            self.skipTest(f"set {RUN_SLOW_ENV}=1 to run the aspirin probe")

        result = _run_aspirin_probe()
        print(json.dumps(asdict(result), indent=2, sort_keys=True))

        if result.accepted:
            return
        self.assertIsNotNone(result.work_violation, result.blocked_kind)


def _run_aspirin_probe() -> WriterAspirinEnvelopeProbeResult:
    timings: list[tuple[str, float]] = []
    default_budget = WriterEnvelopeWorkBudget()
    high_budget = ASPIRIN_PROBE_ENVELOPE_BUDGET
    default_budget_violation = None
    count_envelope = None
    diagnostics = WriterCountDagBuildDiagnostics()

    try:
        with widened_writer_work_envelope():
            prepared = _timed(timings, "prepare", _prepare_aspirin)
            snapshot = _timed(
                timings,
                "initial_snapshot",
                lambda: _initial_snapshot(prepared),
            )
            try:
                _timed(
                    timings,
                    "default_count_envelope",
                    lambda: writer_frontier_count_envelope_for_snapshot(
                        prepared=prepared,
                        snapshot=snapshot,
                        budget=default_budget,
                    ),
                )
            except WriterEnvelopeWorkExceeded as exc:
                default_budget_violation = asdict(exc.violation)
            count_envelope = _timed(
                timings,
                "high_budget_count_envelope",
                lambda: writer_frontier_count_envelope_for_snapshot(
                    prepared=prepared,
                    snapshot=snapshot,
                    budget=high_budget,
                    count_dag_diagnostics=diagnostics,
                ),
            )
            image_envelope = _timed(
                timings,
                "high_budget_support_image_envelope",
                lambda: writer_support_image_envelope_for_snapshot(
                    prepared=prepared,
                    snapshot=snapshot,
                    budget=high_budget,
                ),
            )
            verification = _timed(
                timings,
                "high_budget_live_support_image_verify",
                lambda: verify_writer_support_image_envelope(
                    prepared=prepared,
                    envelope=image_envelope,
                    budget=high_budget,
                ),
            )
            if not verification.accepted:
                return _probe_result(
                    accepted=False,
                    blocked_kind=verification.reason,
                    work_violation=_work_violation_from_reason(
                        verification.reason,
                    ),
                    default_budget_violation=default_budget_violation,
                    high_budget_accepted=True,
                    next_blocker=verification.reason,
                    count_envelope=count_envelope,
                    image_envelope=image_envelope,
                    diagnostics=diagnostics,
                    timings=timings,
                )
            consistency = _timed(
                timings,
                "high_budget_structural_consistency_verify",
                lambda: verify_writer_support_image_envelope_consistency(
                    image_envelope,
                    budget=high_budget,
                ),
            )
            if not consistency.accepted:
                return _probe_result(
                    accepted=False,
                    blocked_kind=consistency.reason,
                    work_violation=_work_violation_from_reason(
                        consistency.reason,
                    ),
                    default_budget_violation=default_budget_violation,
                    high_budget_accepted=True,
                    next_blocker=consistency.reason,
                    count_envelope=count_envelope,
                    image_envelope=image_envelope,
                    diagnostics=diagnostics,
                    timings=timings,
                )
            return _probe_result(
                accepted=True,
                blocked_kind=None,
                work_violation=None,
                default_budget_violation=default_budget_violation,
                high_budget_accepted=True,
                next_blocker=None,
                count_envelope=count_envelope,
                image_envelope=image_envelope,
                diagnostics=diagnostics,
                timings=timings,
            )
    except WriterEnvelopeWorkExceeded as exc:
        return _probe_result(
            accepted=False,
            blocked_kind="envelope_work_exceeded",
            work_violation=asdict(exc.violation),
            default_budget_violation=default_budget_violation,
            high_budget_accepted=False,
            next_blocker=asdict(exc.violation),
            count_envelope=count_envelope,
            image_envelope=None,
            diagnostics=diagnostics,
            timings=timings,
        )


@contextmanager
def widened_writer_work_envelope():
    graph = WriterGraphObligationWorkEnvelope(
        **WIDENED_WRITER_WORK_BOUNDS["graph_obligation"]
    )
    residual = WriterResidualWorkEnvelope(
        **WIDENED_WRITER_WORK_BOUNDS["residual"]
    )
    finite = WriterFiniteRelationWorkEnvelope(
        **WIDENED_WRITER_WORK_BOUNDS["finite_relation"]
    )

    with (
        patch.object(
            writer_execution_evidence,
            "_PUBLIC_WRITER_GRAPH_OBLIGATION_WORK_ENVELOPE",
            graph,
        ),
        patch.object(
            writer_execution_evidence,
            "_PUBLIC_WRITER_RESIDUAL_WORK_ENVELOPE",
            residual,
        ),
        patch.object(
            writer_execution_evidence,
            "_PUBLIC_WRITER_FINITE_RELATION_WORK_ENVELOPE",
            finite,
        ),
    ):
        yield


def _probe_result(
    *,
    accepted: bool,
    blocked_kind: str | None,
    work_violation: dict[str, object] | None,
    default_budget_violation: dict[str, object] | None,
    high_budget_accepted: bool | None,
    next_blocker: dict[str, object] | str | None,
    count_envelope,
    image_envelope,
    diagnostics: WriterCountDagBuildDiagnostics,
    timings: list[tuple[str, float]],
) -> WriterAspirinEnvelopeProbeResult:
    count_metrics = _count_metrics(count_envelope, diagnostics)
    image_metrics = _image_metrics(image_envelope)
    dag_profile = _count_dag_profile(
        None if count_envelope is None else count_envelope["count_dag"],
        diagnostics,
    )
    return WriterAspirinEnvelopeProbeResult(
        input_smiles=ASPIRIN_SMILES,
        facts_source=ASPIRIN_FACTS_SOURCE,
        widened_runtime_bounds=WIDENED_WRITER_WORK_BOUNDS,
        high_envelope_budget=asdict(ASPIRIN_PROBE_ENVELOPE_BUDGET),
        accepted=accepted,
        blocked_kind=blocked_kind,
        work_violation=work_violation,
        default_budget_violation=default_budget_violation,
        high_budget_accepted=high_budget_accepted,
        high_budget_count_dag_metrics=count_metrics,
        count_dag_root_ids=dag_profile["root_ids"],
        count_dag_choice_count_root_count=dag_profile[
            "choice_count_root_count"
        ],
        count_dag_terminal_choice_count_root_present=dag_profile[
            "terminal_choice_count_root_present"
        ],
        count_dag_node_kind_counts=dag_profile["node_kind_counts"],
        count_dag_dedup_attempts=dag_profile["dedup_attempts"],
        count_dag_dedup_hits=dag_profile["dedup_hits"],
        count_dag_dedup_ratio=dag_profile["dedup_ratio"],
        count_dag_top_largest_nodes=dag_profile["top_largest_nodes"],
        count_dag_top_high_fanout_nodes=dag_profile["top_high_fanout_nodes"],
        count_dag_top_deepest_nodes=dag_profile["top_deepest_nodes"],
        next_blocker=next_blocker,
        support_count=_field(count_envelope, "support_count"),
        completion_count=_field(count_envelope, "completion_count"),
        support_string_count=image_metrics["support_string_count"],
        support_image_distinct_count=_field(image_envelope, "distinct_count"),
        support_image_witness_count=_field(image_envelope, "witness_count"),
        count_dag_node_count=count_metrics["node_count"],
        count_dag_edge_count=count_metrics["edge_count"],
        count_dag_max_depth=count_metrics["max_depth"],
        count_dag_digest_input_bytes=count_metrics["digest_input_bytes"],
        largest_count_node_digest_input_bytes=count_metrics[
            "largest_node_digest_input_bytes"
        ],
        support_image_total_emitted_text_bytes=image_metrics[
            "total_emitted_text_bytes"
        ],
        coverage_bucket_count=image_metrics["coverage_bucket_count"],
        bucket_assignment_count=image_metrics["bucket_assignment_count"],
        nested_envelope_count=image_metrics["nested_envelope_count"],
        consistency_node_count=image_metrics["consistency_node_count"],
        timings=tuple(timings),
    )


def _count_metrics(
    envelope,
    diagnostics: WriterCountDagBuildDiagnostics,
) -> dict[str, int | None]:
    if envelope is None:
        if diagnostics.pre_digest_metrics is not None:
            metrics = diagnostics.pre_digest_metrics
            return {
                "node_count": metrics["node_count"],
                "edge_count": metrics["edge_count"],
                "max_depth": metrics["max_depth"],
                "digest_input_bytes": metrics["digest_input_bytes"],
                "largest_node_digest_input_bytes": max(
                    (
                        node["digest_input_bytes"]
                        for node in diagnostics.pre_digest_nodes
                    ),
                    default=0,
                ),
            }
        return {
            "node_count": None,
            "edge_count": None,
            "max_depth": None,
            "digest_input_bytes": None,
            "largest_node_digest_input_bytes": None,
        }
    dag = envelope["count_dag"]
    metrics = dag["metrics"]
    return {
        "node_count": metrics["node_count"],
        "edge_count": metrics["edge_count"],
        "max_depth": metrics["max_depth"],
        "digest_input_bytes": metrics["digest_input_bytes"],
        "largest_node_digest_input_bytes": max(
            (node["digest_input_bytes"] for node in dag["nodes"]),
            default=0,
        ),
    }


def _image_metrics(envelope) -> dict[str, int | None]:
    if envelope is None:
        return {
            "support_string_count": None,
            "total_emitted_text_bytes": None,
            "coverage_bucket_count": None,
            "bucket_assignment_count": None,
            "nested_envelope_count": None,
            "consistency_node_count": None,
        }
    coverage = envelope["enumeration_coverage"]
    terminal = coverage["terminal_bucket"]
    return {
        "support_string_count": len(envelope["support_string_envelopes"]),
        "total_emitted_text_bytes": sum(
            len(text.encode("utf-8"))
            for string_envelope in envelope["support_string_envelopes"]
            for text in string_envelope["emitted_texts"]
        ),
        "coverage_bucket_count": len(coverage["text_buckets"])
        + (0 if terminal is None else 1),
        "bucket_assignment_count": sum(
            len(bucket["string_indices"])
            for bucket in coverage["text_buckets"]
        )
        + (1 if terminal is not None and terminal["string_index"] is not None else 0),
        "nested_envelope_count": _count_nested_envelopes(envelope),
        "consistency_node_count": _count_json_nodes(envelope),
    }


def _count_dag_profile(
    dag,
    diagnostics: WriterCountDagBuildDiagnostics,
) -> dict[str, object]:
    if dag is None and diagnostics.pre_digest_nodes:
        nodes = diagnostics.pre_digest_nodes
        depths = diagnostics.pre_digest_depths
        root_ids = diagnostics.pre_digest_roots
    elif dag is not None:
        nodes = dag["nodes"]
        depths = _count_dag_depths(dag)
        root_ids = dag["roots"]
    else:
        return {
            "root_ids": None,
            "choice_count_root_count": None,
            "terminal_choice_count_root_present": None,
            "node_kind_counts": None,
            "dedup_attempts": dict(diagnostics.attempted_node_emissions_by_kind),
            "dedup_hits": dict(diagnostics.dedup_hits_by_kind),
            "dedup_ratio": None,
            "top_largest_nodes": None,
            "top_high_fanout_nodes": None,
            "top_deepest_nodes": None,
        }
    assert root_ids is not None
    attempts = dict(diagnostics.attempted_node_emissions_by_kind)
    hits = dict(diagnostics.dedup_hits_by_kind)
    attempted_total = sum(attempts.values())
    return {
        "root_ids": root_ids,
        "choice_count_root_count": len(root_ids["choice_count_roots"]),
        "terminal_choice_count_root_present": (
            root_ids["terminal_choice_count_root"] is not None
        ),
        "node_kind_counts": _node_kind_counts(nodes),
        "dedup_attempts": attempts,
        "dedup_hits": hits,
        "dedup_ratio": (
            None
            if attempted_total == 0
            else diagnostics.dedup_hits / attempted_total
        ),
        "top_largest_nodes": tuple(
            _node_summary(node, depths)
            for node in sorted(
                nodes,
                key=lambda item: item["digest_input_bytes"],
                reverse=True,
            )[:10]
        ),
        "top_high_fanout_nodes": tuple(
            _node_summary(node, depths)
            for node in sorted(
                nodes,
                key=lambda item: len(item["children"]),
                reverse=True,
            )[:10]
        ),
        "top_deepest_nodes": tuple(
            _node_summary(node, depths)
            for node in sorted(
                nodes,
                key=lambda item: depths[item["node_id"]],
                reverse=True,
            )[:10]
        ),
    }


def _node_kind_counts(nodes) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        kind = node["kind"]
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _count_dag_depths(dag) -> dict[str, int]:
    node_by_id = {node["node_id"]: node for node in dag["nodes"]}
    depths: dict[str, int] = {}

    def depth(node_id: str) -> int:
        if node_id in depths:
            return depths[node_id]
        node = node_by_id[node_id]
        value = 1 + max((depth(child) for child in node["children"]), default=0)
        depths[node_id] = value
        return value

    for node_id in node_by_id:
        depth(node_id)
    return depths


def _node_summary(node, depths: dict[str, int]) -> dict[str, object]:
    return {
        "node_id": node["node_id"],
        "kind": node["kind"],
        "digest_input_bytes": node["digest_input_bytes"],
        "child_count": len(node["children"]),
        "depth": depths[node["node_id"]],
        "identity_digest": _node_identity_digest(node),
    }


def _node_identity_digest(node) -> str | None:
    for key in (
        "cursor",
        "terminal_projection",
        "text_projection",
        "branch_certificate",
    ):
        value = node.get(key)
        if isinstance(value, dict) and isinstance(value.get("digest"), str):
            return value["digest"]
    for key in ("state_key_digest", "terminal_projection_digest"):
        value = node.get(key)
        if isinstance(value, str):
            return value
    return None


def _work_violation_from_reason(reason: str | None) -> dict[str, object] | None:
    if reason is None:
        return None
    prefix = "WRITER_ENVELOPE_WORK_EXCEEDED: "
    if not reason.startswith(prefix):
        return None
    fields: dict[str, object] = {}
    for part in reason[len(prefix) :].split("; "):
        key, value = part.split("=", 1)
        if key in {"actual", "limit"}:
            fields[key] = int(value)
        else:
            fields[key] = value.strip("'")
    return fields


def _field(envelope, key: str):
    if envelope is None:
        return None
    return envelope[key]


def _timed(timings: list[tuple[str, float]], label: str, fn):
    start = time.perf_counter()
    try:
        return fn()
    finally:
        timings.append((label, time.perf_counter() - start))


def _prepare_aspirin():
    facts = ordinary_molecule_facts_from_smiles(ASPIRIN_SMILES)
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
        policy=ordinary_policy_for_facts(facts),
    )


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


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


if __name__ == "__main__":
    unittest.main()
