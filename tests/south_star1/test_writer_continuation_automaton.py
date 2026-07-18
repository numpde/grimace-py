"""Exact weighted writer continuation automaton regressions."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import os
import unittest
from unittest.mock import patch

from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.ids import BondId
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.writer_continuation_automaton import advance_writer_continuation
from grimace._south_star1.writer_continuation_automaton import compile_writer_continuation_automaton
from grimace._south_star1.writer_continuation_automaton import verify_writer_continuation_automaton_consistency
from grimace._south_star1.writer_continuation_automaton import WriterContinuationChoice
from grimace._south_star1.writer_continuation_automaton import WriterContinuationCursor
from grimace._south_star1.writer_continuation_automaton import writer_continuation_choices
from grimace._south_star1.writer_continuation_automaton import writer_continuation_completion_count
from grimace._south_star1.writer_continuation_automaton import writer_continuation_is_terminal
from grimace._south_star1.writer_continuation_automaton import writer_continuation_probabilities
from grimace._south_star1.writer_continuation_automaton import writer_continuation_provenance_edge
from grimace._south_star1.writer_continuation_automaton import writer_continuation_support_count
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_frontier import iter_writer_frontier_support
from grimace._south_star1.writer_events import WriterRingEndpointEmitted
from grimace._south_star1.writer_events import WriterRingEndpointPaired
from grimace._south_star1.writer_snapshot import advance_writer_frontier_snapshot
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import shared_acyclic_directional_facts
from tests.south_star1.helpers import tetrahedral_facts
from tests.south_star1.test_writer_state_kernel import chain_facts
from tests.south_star1.test_writer_state_kernel import duplicate_single_atom_policy
from tests.south_star1.test_writer_stereo_residual import _directional_non_single_ring_carrier_facts
from tests.south_star1.test_writer_stereo_residual import _directional_ring_carrier_facts
from tests.south_star1.test_writer_stereo_residual import _shared_directional_ring_carrier_facts
from tests.south_star1.test_writer_support_artifact_fact_verifier import _initial_snapshot
from tests.south_star1.test_writer_support_artifact_fact_verifier import _prepare
from tests.south_star1.test_writer_support_artifact_fact_verifier import _writer_options


class WriterContinuationAutomatonTest(unittest.TestCase):
    def test_small_fixture_support_images_and_root_counts_agree(self) -> None:
        cases = (
            ("default", cco_facts(), _writer_options()),
            ("tetra", tetrahedral_facts(), _writer_options()),
            ("directional", directional_facts(), _writer_options(rooted_at_atom=2)),
            (
                "shared_acyclic",
                shared_acyclic_directional_facts(),
                _writer_options(rooted_at_atom=0),
            ),
            (
                "simple_ring",
                _directional_ring_carrier_facts(),
                _writer_options(rooted_at_atom=0),
            ),
            (
                "non_single_ring",
                _directional_non_single_ring_carrier_facts(),
                _writer_options(rooted_at_atom=0),
            ),
        )
        for name, facts, options in cases:
            with self.subTest(name=name):
                prepared = _prepare(facts)
                snapshot = _initial_snapshot(prepared, options)
                automaton = compile_writer_continuation_automaton(
                    prepared=prepared,
                    snapshot=snapshot,
                )
                counted = _checked_writer_frontier_branch_supports(
                    prepared,
                    snapshot.cursor,
                    include_counts=True,
                    include_frontier_certificate=True,
                    include_count_certificate=True,
                )
                self.assertEqual(
                    writer_continuation_support_count(automaton),
                    counted.support_count_certificate.support_count,
                )
                self.assertEqual(
                    writer_continuation_completion_count(automaton),
                    counted.count_certificate.completion_count,
                )
                self.assertEqual(
                    _automaton_strings(automaton),
                    tuple(iter_writer_frontier_support(prepared, snapshot.cursor)),
                )
                self.assertTrue(
                    verify_writer_continuation_automaton_consistency(
                        automaton=automaton
                    ).accepted
                )

    def test_live_frontier_agrees_at_every_cco_cursor(self) -> None:
        prepared, snapshot, automaton = _cco_automaton()
        pending = [(snapshot.cursor, automaton.root)]
        seen = set()
        while pending:
            live_cursor, compiled_cursor = pending.pop()
            if live_cursor in seen:
                continue
            seen.add(live_cursor)
            batch = _checked_writer_frontier_branch_supports(
                prepared,
                live_cursor,
                include_counts=True,
                include_frontier_certificate=True,
                include_count_certificate=True,
            )
            compiled_choices = writer_continuation_choices(
                automaton, compiled_cursor
            )
            self.assertEqual(
                tuple(choice.emitted_text for choice in compiled_choices),
                tuple(choice.emitted_text for choice in batch.choices.choices),
            )
            self.assertEqual(
                writer_continuation_support_count(automaton, compiled_cursor),
                batch.support_count_certificate.support_count,
            )
            self.assertEqual(
                writer_continuation_completion_count(automaton, compiled_cursor),
                batch.count_certificate.completion_count,
            )
            terminal = batch.choices.terminal
            self.assertEqual(
                writer_continuation_is_terminal(automaton, compiled_cursor),
                terminal is not None,
            )
            probabilities = writer_continuation_probabilities(
                automaton, compiled_cursor
            )
            self.assertEqual(
                sum(item.numerator for item in probabilities),
                probabilities[0].denominator,
            )
            for live_choice, compiled_choice in zip(
                batch.choices.choices, compiled_choices
            ):
                self.assertEqual(
                    compiled_choice.immediate_multiplicity,
                    live_choice.immediate_multiplicity,
                )
                self.assertEqual(
                    compiled_choice.support_count,
                    live_choice.support_count,
                )
                self.assertEqual(
                    compiled_choice.completion_count,
                    live_choice.completion_count,
                )
                advanced = advance_writer_continuation(
                    automaton,
                    compiled_cursor,
                    live_choice.emitted_text,
                )
                provenance = _cursor_provenance(
                    automaton, live_choice.successor
                )
                self.assertEqual(advanced.node_id, provenance.compiled_node_id)
                self.assertEqual(
                    advanced.completion_scale,
                    provenance.normalization_scale,
                )
                pending.append((live_choice.successor, advanced))

    def test_weight_normalization_scales_completion_not_support(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            chain_facts(("C",)),
            writer_surface=SouthStarWriterSurface(),
            policy=duplicate_single_atom_policy(),
        )
        options = _writer_options()
        snapshot = _initial_snapshot(prepared, options)
        automaton = compile_writer_continuation_automaton(
            prepared=prepared,
            snapshot=snapshot,
        )
        choice = writer_continuation_choices(automaton)[0]
        self.assertEqual(choice.immediate_multiplicity, 2)
        self.assertEqual(choice.support_count, 1)
        self.assertEqual(choice.completion_count, 2)
        self.assertEqual(choice.successor_scale, 2)
        terminal_cursor = advance_writer_continuation(
            automaton, automaton.root, choice.emitted_text
        )
        self.assertTrue(writer_continuation_is_terminal(automaton, terminal_cursor))
        self.assertEqual(
            writer_continuation_support_count(automaton, terminal_cursor), 1
        )
        self.assertEqual(
            writer_continuation_completion_count(automaton, terminal_cursor), 2
        )
        self.assertTrue(
            any(
                item.normalization_scale == 2
                for item in automaton.provenance.cursors
            )
        )

    def test_compile_does_not_invoke_legacy_count_or_support_paths(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared, _writer_options())
        patches = (
            patch(
                "grimace._south_star1.writer_frontier_count_envelope."
                "writer_frontier_count_envelope_for_snapshot",
                side_effect=AssertionError("count envelope path"),
            ),
            patch(
                "grimace._south_star1.writer_count_dag_envelope."
                "writer_count_certificate_dag_envelope_for_product",
                side_effect=AssertionError("count DAG path"),
            ),
            patch(
                "grimace._south_star1.writer_support_string_envelope."
                "_iter_writer_snapshot_certified_support_strings",
                side_effect=AssertionError("support materialization path"),
            ),
        )
        with patches[0], patches[1], patches[2]:
            automaton = compile_writer_continuation_automaton(
                prepared=prepared, snapshot=snapshot
            )
        self.assertGreater(automaton.metrics.semantic_node_count, 0)

    def test_compiled_cursor_snapshot_resume_tracks_live_snapshot(self) -> None:
        prepared, snapshot, automaton = _cco_automaton()
        choice = writer_continuation_choices(automaton)[0]
        live = advance_writer_frontier_snapshot(
            snapshot,
            prepared=prepared,
            emitted_text=choice.emitted_text,
        )
        compiled = advance_writer_continuation(
            automaton,
            automaton.root,
            choice.emitted_text,
        )
        provenance_edge = writer_continuation_provenance_edge(
            automaton,
            source_raw_cursor_digest=(
                automaton.provenance.root_raw_cursor_digest
            ),
            emitted_text=choice.emitted_text,
        )
        self.assertEqual(
            provenance_edge.successor_raw_cursor_digest,
            _identity_digest(live.cursor),
        )
        resumed = WriterContinuationCursor(
            node_id=compiled.node_id,
            completion_scale=compiled.completion_scale,
        )
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            live.cursor,
            include_counts=True,
            include_frontier_certificate=True,
            include_count_certificate=True,
        )
        self.assertEqual(
            tuple(
                item.emitted_text
                for item in writer_continuation_choices(automaton, resumed)
            ),
            tuple(item.emitted_text for item in batch.choices.choices),
        )
        self.assertEqual(
            writer_continuation_completion_count(automaton, resumed),
            batch.count_certificate.completion_count,
        )

    def test_consistency_rejects_core_and_provenance_mutations(self) -> None:
        prepared, snapshot, original = _cco_automaton()
        root_id = original.root.node_id
        terminal_id = next(
            node.node_id for node in original.nodes if node.terminal_available
        )
        cases = (
            ("terminal_weight", lambda item: _replace_node(item, terminal_id, terminal_completion_count=2)),
            ("terminal_availability", lambda item: _replace_node(item, terminal_id, terminal_available=False)),
            ("duplicate_text", lambda item: _replace_node(item, root_id, choices=item.nodes[root_id].choices + (item.nodes[root_id].choices[0],))),
            ("missing_text", lambda item: _replace_node(item, root_id, choices=item.nodes[root_id].choices[1:])),
            ("successor", lambda item: _replace_choice(item, root_id, 0, successor_node_id=root_id)),
            ("successor_scale", lambda item: _replace_choice(item, root_id, 0, successor_scale=2)),
            ("multiplicity", lambda item: _replace_choice(item, root_id, 0, immediate_multiplicity=99)),
            ("choice_support", lambda item: _replace_choice(item, root_id, 0, support_count=99)),
            ("choice_completion", lambda item: _replace_choice(item, root_id, 0, completion_count=99)),
            ("node_support", lambda item: _replace_node(item, root_id, support_count=99)),
            ("node_completion", lambda item: _replace_node(item, root_id, completion_count=99)),
            ("root_scale", lambda item: replace(item, root=WriterContinuationCursor(root_id, 2))),
            ("signature", lambda item: _replace_node(item, root_id, signature_digest="0" * 64)),
            ("cycle", lambda item: _replace_choice(item, root_id, 0, successor_node_id=root_id)),
            ("unreachable", _append_unreachable_node),
            ("unequal_merge", _merge_unequal_provenance),
            ("equal_split", _split_equal_signature),
            ("edge_source", _detach_edge_provenance),
            ("missing_branch", _remove_branch_provenance),
            ("terminal_source", _detach_terminal_provenance),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                forged = mutate(original)
                checked = verify_writer_continuation_automaton_consistency(
                    automaton=forged,
                    prepared=prepared,
                    snapshot=snapshot,
                )
                self.assertFalse(checked.accepted)
                self.assertIn("continuation", checked.reason)

    @unittest.skipUnless(
        os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1",
        "full shared-ring continuation integration is slow-gated",
    )
    def test_full_shared_ring_root_metrics_and_internal_consistency(self) -> None:
        prepared = _prepare(_shared_directional_ring_carrier_facts())
        snapshot = _initial_snapshot(
            prepared, _writer_options(rooted_at_atom=1)
        )
        automaton = compile_writer_continuation_automaton(
            prepared=prepared, snapshot=snapshot
        )
        self.assertTrue(
            verify_writer_continuation_automaton_consistency(
                automaton=automaton
            ).accepted
        )
        self.assertEqual(writer_continuation_support_count(automaton), 3_744)
        self.assertEqual(writer_continuation_completion_count(automaton), 3_744)
        self.assertEqual(automaton.metrics.raw_cursor_count, 19_595)
        self.assertEqual(automaton.metrics.primitive_cursor_count, 19_595)
        self.assertEqual(automaton.metrics.semantic_node_count, 2_101)
        self.assertEqual(automaton.metrics.semantic_edge_count, 2_843)
        self.assertEqual(automaton.metrics.maximum_depth, 27)
        self.assertEqual(automaton.metrics.maximum_out_degree, 5)
        self.assertEqual(
            automaton.metrics.largest_equivalence_class_membership,
            3_744,
        )
        self.assertEqual(automaton.metrics.canonical_core_bytes, 1_457_372)
        _assert_shared_ring_paths(
            prepared=prepared,
            snapshot=snapshot,
            automaton=automaton,
        )


@lru_cache(maxsize=1)
def _cco_automaton():
    prepared = _prepare(cco_facts())
    snapshot = _initial_snapshot(prepared, _writer_options())
    return prepared, snapshot, compile_writer_continuation_automaton(
        prepared=prepared, snapshot=snapshot
    )


def _automaton_strings(automaton):
    memo = {}

    def visit(node_id):
        if node_id in memo:
            return memo[node_id]
        node = automaton.nodes[node_id]
        values = [""] if node.terminal_available else []
        for choice in node.choices:
            values.extend(
                choice.emitted_text + suffix
                for suffix in visit(choice.successor_node_id)
            )
        memo[node_id] = tuple(sorted(values))
        return memo[node_id]

    return visit(automaton.root.node_id)


def _cursor_provenance(automaton, cursor):
    digest = _identity_digest(cursor)
    return next(
        item
        for item in automaton.provenance.cursors
        if item.raw_cursor_digest == digest
    )


def _replace_node(automaton, node_id, **changes):
    nodes = list(automaton.nodes)
    nodes[node_id] = replace(nodes[node_id], **changes)
    return replace(automaton, nodes=tuple(nodes))


def _replace_choice(automaton, node_id, choice_index, **changes):
    node = automaton.nodes[node_id]
    choices = list(node.choices)
    choices[choice_index] = replace(choices[choice_index], **changes)
    return _replace_node(automaton, node_id, choices=tuple(choices))


def _append_unreachable_node(automaton):
    duplicate = replace(
        automaton.nodes[0], node_id=len(automaton.nodes)
    )
    return replace(automaton, nodes=automaton.nodes + (duplicate,))


def _split_equal_signature(automaton):
    duplicate_id = len(automaton.nodes)
    duplicate = replace(automaton.nodes[0], node_id=duplicate_id)
    root = automaton.nodes[automaton.root.node_id]
    extra = WriterContinuationChoice(
        emitted_text="invented",
        immediate_multiplicity=1,
        successor_node_id=duplicate_id,
        successor_scale=1,
        support_count=duplicate.support_count,
        completion_count=duplicate.completion_count,
    )
    forged = replace(automaton, nodes=automaton.nodes + (duplicate,))
    return _replace_node(
        forged,
        root.node_id,
        choices=tuple(sorted(root.choices + (extra,), key=lambda item: item.emitted_text)),
    )


def _merge_unequal_provenance(automaton):
    cursors = list(automaton.provenance.cursors)
    source = next(
        item
        for item in cursors
        if item.compiled_node_id != automaton.root.node_id
    )
    index = cursors.index(source)
    cursors[index] = replace(
        source, compiled_node_id=automaton.root.node_id
    )
    return replace(
        automaton,
        provenance=replace(automaton.provenance, cursors=tuple(cursors)),
    )


def _detach_edge_provenance(automaton):
    edges = list(automaton.provenance.edges)
    edges[0] = replace(edges[0], source_raw_cursor_digest="0" * 64)
    return replace(
        automaton,
        provenance=replace(automaton.provenance, edges=tuple(edges)),
    )


def _remove_branch_provenance(automaton):
    edges = list(automaton.provenance.edges)
    edges[0] = replace(edges[0], branch_certificate_digests=())
    return replace(
        automaton,
        provenance=replace(automaton.provenance, edges=tuple(edges)),
    )


def _detach_terminal_provenance(automaton):
    terminals = list(automaton.provenance.terminals)
    terminals[0] = replace(
        terminals[0],
        source_node_id=automaton.root.node_id,
        source_raw_cursor_digest=automaton.provenance.cursors[-1].raw_cursor_digest,
    )
    return replace(
        automaton,
        provenance=replace(automaton.provenance, terminals=tuple(terminals)),
    )


def _assert_shared_ring_paths(*, prepared, snapshot, automaton):
    pending = [(snapshot.cursor, automaton.root)]
    seen = set()
    found = set()
    while pending and len(found) < 6:
        live_cursor, compiled_cursor = pending.pop()
        if live_cursor in seen:
            continue
        seen.add(live_cursor)
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            live_cursor,
            include_counts=False,
            include_frontier_certificate=True,
            include_count_certificate=False,
        )
        choices = {
            item.emitted_text: item for item in batch.choices.choices
        }
        for support in batch.supports:
            for event in support.events:
                if (
                    isinstance(event, WriterRingEndpointEmitted)
                    and event.bond == BondId(1)
                ):
                    found.add(("opening", event.direction_mark))
                if (
                    isinstance(event, WriterRingEndpointPaired)
                    and event.bond == BondId(1)
                ):
                    found.add(
                        ("pair", event.first_endpoint_direction_mark)
                    )
        for emitted_text, live_choice in choices.items():
            advanced = advance_writer_continuation(
                automaton, compiled_cursor, emitted_text
            )
            provenance = _cursor_provenance(
                automaton, live_choice.successor
            )
            if (
                advanced.node_id != provenance.compiled_node_id
                or advanced.completion_scale
                != provenance.normalization_scale
            ):
                raise AssertionError("shared-ring continuation advance drift")
            pending.append((live_choice.successor, advanced))
    expected = {
        (phase, mark)
        for phase in ("opening", "pair")
        for mark in (
            DirectionMark.ABSENT,
            DirectionMark.FWD,
            DirectionMark.REV,
        )
    }
    if found != expected:
        raise AssertionError(f"missing shared-ring paths: {expected - found}")


if __name__ == "__main__":
    unittest.main()
