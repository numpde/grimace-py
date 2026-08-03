"""Focused tests for bounded directional non-neighbor writer support."""

from __future__ import annotations

import unittest
from dataclasses import replace

import grimace._south_star1.writer_capabilities as writer_capabilities
import grimace._south_star1.writer_frontier as writer_frontier_module
import grimace._south_star1.writer_snapshot as writer_snapshot
from grimace._south_star1 import writer_stereo
from grimace._south_star1.facts import (
    BondOrder,
    ComponentFacts,
    DirectionalSiteFacts,
    DirectionalValue,
    LigandKind,
    LigandOccurrence,
    MoleculeFacts,
    SiteStatus,
    StereoFacts,
)
from grimace._south_star1.ids import AtomId, BondId, ComponentId, OccurrenceId, SiteId
from grimace._south_star1.prepared_runtime import (
    SouthStarRuntimeOptions,
)
from grimace._south_star1.residual_constraints import ResidualPropagationKind
from grimace._south_star1.writer_frontier import (
    count_writer_cursor_completions,
    count_writer_frontier_support,
    initial_writer_frontier_cursor,
    iter_writer_frontier_support,
)
from tests.south_star1.helpers import atom, bond, single_bond
from tests.south_star1.writer_test_context import prepare_writer_facts
from tests.south_star1.writer_test_context import writer_runtime_options


def _facts():
    site = SiteId(0)
    facts = MoleculeFacts(
        atoms=(
            replace(atom(0, "C"), implicit_h_count=1),
            atom(1, "C"),
            atom(2, "F"),
            atom(3, "Cl"),
        ),
        bonds=(
            bond(0, 0, 1, BondOrder.DOUBLE),
            single_bond(1, 0, 2),
            single_bond(2, 1, 3),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
        ),
        stereo=StereoFacts(
            directional=(
                DirectionalSiteFacts(
                    id=site,
                    center_bond=BondId(0),
                    left_endpoint=AtomId(0),
                    right_endpoint=AtomId(1),
                    status=SiteStatus.SPECIFIED,
                    target=DirectionalValue.OPPOSITE,
                    left_ligands=(OccurrenceId(0), OccurrenceId(1)),
                    right_ligands=(OccurrenceId(2),),
                    reference_pair=(OccurrenceId(0), OccurrenceId(2)),
                ),
            )
        ),
        ligand_occurrences=(
            LigandOccurrence(
                id=OccurrenceId(0),
                site=site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(2),
                bond=BondId(1),
            ),
            LigandOccurrence(
                id=OccurrenceId(1),
                site=site,
                kind=LigandKind.IMPLICIT_H,
                atom=AtomId(0),
                bond=None,
            ),
            LigandOccurrence(
                id=OccurrenceId(2),
                site=site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(3),
                bond=BondId(2),
            ),
        ),
    )
    return facts


def _first_choice_with_residual_work_evidence(prepared, cursor):
    pending = [cursor]
    seen = set()

    while pending:
        current = pending.pop(0)
        if current in seen:
            continue
        seen.add(current)

        snapshot = writer_frontier_module._writer_frontier_choice_snapshot(
            prepared,
            current,
            include_counts=False,
        )
        for choice in snapshot.choices:
            if choice.residual_work_evidence:
                return choice
            pending.append(choice.successor)

    raise AssertionError("no choice with residual work evidence")


class BoundedDirectionalNonNeighborWriterTest(unittest.TestCase):
    def test_implicit_h_is_not_a_residual_carrier_variable(self) -> None:
        prepared = prepare_writer_facts(_facts())
        cursor = initial_writer_frontier_cursor(prepared, writer_runtime_options())
        residual = cursor.weighted_states[0][0].stereo_state.residual_snapshot

        self.assertEqual(
            {var.kind for var, _domain in residual.domains},
            {"directional_site_carrier"},
        )
        self.assertNotIn(
            "directional_non_neighbor_ligand",
            {var.kind for var, _domain in residual.domains},
        )

    def test_out_of_scope_non_neighbor_shapes_remain_typed_blockers(self) -> None:
        base = _facts()
        site = base.stereo.directional[0]
        pseudo = replace(
            base.ligand_occurrences[1],
            kind=LigandKind.PSEUDO,
            atom=None,
        )
        pseudo_facts = replace(
            base,
            atoms=(replace(base.atoms[0], implicit_h_count=0), *base.atoms[1:]),
            ligand_occurrences=(
                base.ligand_occurrences[0],
                pseudo,
                base.ligand_occurrences[2],
            ),
        )
        second_h = LigandOccurrence(
            id=OccurrenceId(3),
            site=site.id,
            kind=LigandKind.IMPLICIT_H,
            atom=site.left_endpoint,
            bond=None,
        )
        two_h_facts = replace(
            base,
            atoms=(replace(base.atoms[0], implicit_h_count=2), *base.atoms[1:]),
            stereo=StereoFacts(
                directional=(
                    replace(
                        site,
                        left_ligands=(*site.left_ligands, second_h.id),
                    ),
                ),
            ),
            ligand_occurrences=(*base.ligand_occurrences, second_h),
        )
        carrierless_site = replace(
            site,
            left_ligands=(base.ligand_occurrences[1].id,),
            reference_pair=(
                base.ligand_occurrences[1].id,
                base.ligand_occurrences[2].id,
            ),
        )
        carrierless_facts = replace(
            base,
            stereo=StereoFacts(directional=(carrierless_site,)),
            ligand_occurrences=base.ligand_occurrences[1:],
        )
        for facts, carrier_bond in (
            (pseudo_facts, BondId(1)),
            (two_h_facts, BondId(1)),
            (carrierless_facts, BondId(2)),
        ):
            facts.validate()
            prepared = prepare_writer_facts(facts)
            blocker = writer_stereo._unsupported_directional_non_neighbor_ligand_blocker_for_bond(
                prepared,
                carrier_bond,
                operation="directional carrier-mark restriction",
            )
            self.assertIsNotNone(blocker)
            self.assertEqual(
                blocker.kind,
                "unsupported_directional_non_neighbor_ligand",
            )

    def test_bounded_non_neighbor_directional_site_has_writer_support(self) -> None:
        prepared = prepare_writer_facts(_facts())
        cursor = initial_writer_frontier_cursor(prepared, writer_runtime_options())
        strings = tuple(iter_writer_frontier_support(prepared, cursor))
        self.assertTrue(strings)
        self.assertEqual(
            len(strings),
            count_writer_frontier_support(prepared, cursor.support_state),
        )
        self.assertGreater(count_writer_cursor_completions(prepared, cursor), 0)

    def test_bounded_non_neighbor_directional_site_records_live_work(
        self,
    ) -> None:
        prepared = prepare_writer_facts(_facts())
        cursor = initial_writer_frontier_cursor(prepared, writer_runtime_options())

        choice = _first_choice_with_residual_work_evidence(prepared, cursor)

        self.assertIn(
            (
                writer_capabilities._WriterExecutionCapabilityKind
                .DIRECTIONAL_CARRIER_RESTRICTION
            ),
            choice.execution_capabilities,
        )
        self.assertIn(
            (
                writer_capabilities._WriterExecutionCapabilityKind
                .DIRECTIONAL_SITE_COMPATIBILITY
            ),
            choice.execution_capabilities,
        )
        self.assertIn(
            (
                writer_capabilities._WriterExecutionCapabilityKind
                .RESIDUAL_PROPAGATION
            ),
            choice.execution_capabilities,
        )
        self.assertIn(
            "directional carrier-mark restriction",
            {item.operation for item in choice.residual_work_evidence},
        )
        for item in choice.residual_work_evidence:
            self.assertIs(
                item.result_kind,
                ResidualPropagationKind.CERTIFIED_CONSISTENT,
            )
            self.assertGreaterEqual(item.component_variable_count, 1)
            self.assertGreaterEqual(item.component_factor_count, 1)

    def test_bounded_non_neighbor_directional_site_snapshots_resume(
        self,
    ) -> None:
        prepared = prepare_writer_facts(_facts())
        options = writer_runtime_options()
        snapshot = writer_snapshot.capture_initial_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
        )

        choices = writer_snapshot.resume_writer_frontier_choices_from_snapshot(
            snapshot,
            prepared=prepared,
        )
        self.assertTrue(choices.choices)

        advanced = writer_snapshot.advance_writer_frontier_snapshot(
            snapshot,
            prepared=prepared,
            emitted_text=choices.choices[0].emitted_text,
        )
        self.assertTrue(advanced.cursor.weighted_states)


if __name__ == "__main__":
    unittest.main()
