"""Focused tests for bounded directional non-neighbor writer support."""

from __future__ import annotations

import unittest
from dataclasses import replace

import grimace._south_star1.writer_capabilities as writer_capabilities
import grimace._south_star1.writer_frontier as writer_frontier_module
import grimace._south_star1.writer_snapshot as writer_snapshot
from grimace._south_star1.facts import BondOrder, ComponentFacts, DirectionalSiteFacts, DirectionalValue, LigandKind, LigandOccurrence, MoleculeFacts, SiteStatus, StereoFacts
from grimace._south_star1.ids import AtomId, BondId, ComponentId, OccurrenceId, SiteId
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions, SouthStarWriterSurface, prepare_south_star_mol_from_facts
from grimace._south_star1.residual_constraints import ResidualPropagationKind
from grimace._south_star1.writer_frontier import count_writer_cursor_completions, count_writer_frontier_support, initial_writer_frontier_cursor, iter_writer_frontier_support
from tests.south_star1.helpers import atom, bond, single_bond


def _options() -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(serialization_language=SerializationLanguageMode.WRITER_SHAPED)


def _prepared():
    site = SiteId(0)
    facts = MoleculeFacts(
        atoms=(replace(atom(0, "C"), implicit_h_count=1), atom(1, "C"), atom(2, "F"), atom(3, "Cl")),
        bonds=(bond(0, 0, 1, BondOrder.DOUBLE), single_bond(1, 0, 2), single_bond(2, 1, 3)),
        components=(ComponentFacts(id=ComponentId(0), atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)), bonds=(BondId(0), BondId(1), BondId(2))),),
        stereo=StereoFacts(directional=(DirectionalSiteFacts(id=site, center_bond=BondId(0), left_endpoint=AtomId(0), right_endpoint=AtomId(1), status=SiteStatus.SPECIFIED, target=DirectionalValue.OPPOSITE, left_ligands=(OccurrenceId(0), OccurrenceId(1)), right_ligands=(OccurrenceId(2),), reference_pair=(OccurrenceId(0), OccurrenceId(2))),)),
        ligand_occurrences=(
            LigandOccurrence(id=OccurrenceId(0), site=site, kind=LigandKind.NEIGHBOR_ATOM, atom=AtomId(2), bond=BondId(1)),
            LigandOccurrence(id=OccurrenceId(1), site=site, kind=LigandKind.IMPLICIT_H, atom=AtomId(0), bond=None),
            LigandOccurrence(id=OccurrenceId(2), site=site, kind=LigandKind.NEIGHBOR_ATOM, atom=AtomId(3), bond=BondId(2)),
        ),
    )
    return prepare_south_star_mol_from_facts(facts, writer_surface=SouthStarWriterSurface())


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
    def test_bounded_non_neighbor_directional_site_has_writer_support(self) -> None:
        prepared = _prepared()
        cursor = initial_writer_frontier_cursor(prepared, _options())
        strings = tuple(iter_writer_frontier_support(prepared, cursor))
        self.assertTrue(strings)
        self.assertEqual(len(strings), count_writer_frontier_support(prepared, cursor.support_state))
        self.assertGreater(count_writer_cursor_completions(prepared, cursor), 0)

    def test_bounded_non_neighbor_directional_site_records_live_work(
        self,
    ) -> None:
        prepared = _prepared()
        cursor = initial_writer_frontier_cursor(prepared, _options())

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
        prepared = _prepared()
        options = _options()
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
