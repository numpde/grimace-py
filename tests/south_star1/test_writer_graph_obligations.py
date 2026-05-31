"""Writer residual graph-obligation classifier tests."""

from __future__ import annotations

from dataclasses import replace
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_frontier import writer_frontier_choices
from grimace._south_star1.writer_graph_obligations import WriterBoundaryOwnerKind
from grimace._south_star1.writer_graph_obligations import build_writer_block_cut_metadata
from grimace._south_star1.writer_graph_obligations import classify_writer_residual_attachments
from grimace._south_star1.writer_graph_obligations import writer_boundary_incidence_sort_tuple
from grimace._south_star1.writer_graph_obligations import writer_residual_attachment_sort_tuple
from grimace._south_star1.writer_state import ComponentCursor
from grimace._south_star1.writer_state import ObligationState
from grimace._south_star1.writer_state import WriterAtomFrame
from grimace._south_star1.writer_state import WriterPolicyState
from grimace._south_star1.writer_state import WriterRingState
from grimace._south_star1.writer_state import WriterState
from grimace._south_star1.writer_state import writer_state_from_key
from grimace._south_star1.writer_state import writer_state_key
from grimace._south_star1.writer_stereo import empty_writer_stereo_state
from grimace._south_star1.writer_transitions import legal_writer_transitions
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import single_bond


class WriterGraphObligationsTest(unittest.TestCase):
    def test_cco_prefix_classifies_active_residual_attachment(self) -> None:
        prepared = _prepare(cco_facts())
        key = _cco_after_second_atom_key(prepared, _writer_options(rooted_at_atom=0))

        summary = _summary(prepared, key)

        self.assertFalse(summary.has_cyclic_attachment)
        self.assertEqual(len(summary.attachments.attachments), 1)
        attachment = summary.attachments.attachments[0]
        self.assertEqual(attachment.atoms, frozenset((AtomId(2),)))
        self.assertEqual(attachment.latent_bonds, frozenset())
        self.assertEqual(attachment.cyclic_rank, 0)
        self.assertEqual(len(attachment.boundary), 1)
        incidence = attachment.boundary[0]
        self.assertEqual((incidence.bond, incidence.written_atom, incidence.residual_atom), (BondId(1), AtomId(1), AtomId(2)))
        self.assertIs(incidence.owner_kind, WriterBoundaryOwnerKind.ACTIVE_ATOM)
        self.assertEqual(summary.boundary_by_owner_atom, ((AtomId(1), (0,)),))

    def test_branch_prefix_classifies_sibling_attachment_as_branch_owned(self) -> None:
        prepared = _prepare(cco_facts())
        key = _cco_branch_child_key(prepared, _writer_options(rooted_at_atom=1))

        summary = _summary(prepared, key)

        self.assertFalse(summary.has_cyclic_attachment)
        self.assertEqual(len(summary.attachments.attachments), 1)
        incidence = summary.attachments.attachments[0].boundary[0]
        self.assertEqual((incidence.bond, incidence.written_atom, incidence.residual_atom), (BondId(1), AtomId(1), AtomId(2)))
        self.assertIs(incidence.owner_kind, WriterBoundaryOwnerKind.BRANCH_RETURN)
        self.assertEqual(summary.boundary_by_owner_atom, ((AtomId(1), (0,)),))

    def test_ring_entry_classifies_one_cyclic_attachment_not_two_children(self) -> None:
        prepared = _prepare(six_ring_facts())
        key = _emitted_root_key(prepared, root=AtomId(0))

        summary = _summary(prepared, key)

        self.assertTrue(summary.has_cyclic_attachment)
        self.assertEqual(len(summary.attachments.attachments), 1)
        attachment = summary.attachments.attachments[0]
        self.assertEqual(attachment.atoms, frozenset(AtomId(index) for index in range(1, 6)))
        self.assertEqual(len(attachment.boundary), 2)
        self.assertEqual(
            tuple(
                (item.bond, item.written_atom, item.residual_atom, item.owner_kind)
                for item in attachment.boundary
            ),
            (
                (BondId(0), AtomId(0), AtomId(1), WriterBoundaryOwnerKind.ACTIVE_ATOM),
                (BondId(5), AtomId(0), AtomId(5), WriterBoundaryOwnerKind.ACTIVE_ATOM),
            ),
        )
        with self.assertRaises(SouthStarError):
            legal_writer_transitions(prepared, writer_state_from_key(key))

    def test_cycle_plus_isolate_classifier_exposes_non_tree_shape(self) -> None:
        prepared = _prepare(cycle_plus_isolate_component_facts())
        key = _emitted_root_key(prepared, root=AtomId(0))

        summary = _summary(prepared, key)

        self.assertTrue(summary.has_cyclic_attachment)
        self.assertEqual(len(summary.attachments.attachments), 2)
        self.assertEqual(
            tuple(
                sorted(
                    len(attachment.boundary)
                    for attachment in summary.attachments.attachments
                )
            ),
            (0, 2),
        )

    def test_boundary_incidences_to_same_written_atom_remain_distinct(self) -> None:
        prepared = _prepare(triangle_facts())
        key = _emitted_root_key(prepared, root=AtomId(0))

        summary = _summary(prepared, key)

        self.assertTrue(summary.has_cyclic_attachment)
        self.assertEqual(len(summary.attachments.attachments), 1)
        boundary = summary.attachments.attachments[0].boundary
        self.assertEqual(len(boundary), 2)
        self.assertEqual({item.bond for item in boundary}, {BondId(0), BondId(2)})
        self.assertEqual({item.written_atom for item in boundary}, {AtomId(0)})
        self.assertEqual(
            tuple(sorted(boundary, key=writer_boundary_incidence_sort_tuple)),
            boundary,
        )

    def test_attachment_sort_tuple_is_canonical(self) -> None:
        prepared = _prepare(six_ring_facts())
        key = _emitted_root_key(prepared, root=AtomId(0))
        attachment = _summary(prepared, key).attachments.attachments[0]

        self.assertEqual(
            writer_residual_attachment_sort_tuple(attachment),
            (
                (1, 2, 3, 4, 5),
                (1, 2, 3, 4),
                tuple(writer_boundary_incidence_sort_tuple(item) for item in attachment.boundary),
                0,
                (0,),
            ),
        )


def _summary(prepared, key):
    return classify_writer_residual_attachments(
        prepared,
        key,
        build_writer_block_cut_metadata(prepared),
    )


def _prepare(facts: MoleculeFacts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _writer_options(*, rooted_at_atom: int = -1) -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=rooted_at_atom,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def _cco_after_second_atom_key(prepared, options):
    cursor = initial_writer_frontier_cursor(prepared, options)
    after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
    after_second = writer_frontier_choices(prepared, after_root).choices[0].successor
    return after_second.weighted_states[0][0]


def _cco_branch_child_key(prepared, options):
    cursor = initial_writer_frontier_cursor(prepared, options)
    after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
    after_branch_open = writer_frontier_choices(prepared, after_root).choices[0].successor
    after_branch_child = writer_frontier_choices(
        prepared,
        after_branch_open,
    ).choices[0].successor
    return after_branch_child.weighted_states[0][0]


def _emitted_root_key(prepared, *, root: AtomId):
    return writer_state_key(
        WriterState(
            component_cursor=ComponentCursor(
                component_index=0,
                component_roots=(root,),
            ),
            active=WriterAtomFrame(
                atom=root,
                parent=None,
                incoming_bond=None,
                atom_emitted=True,
            ),
            branch_stack=(),
            visited_atoms=frozenset((root,)),
            written_bonds=frozenset(),
            obligations=ObligationState(),
            ring_state=WriterRingState(),
            stereo_state=empty_writer_stereo_state(),
            policy_state=WriterPolicyState(),
        )
    )


def six_ring_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=tuple(atom(index, "C") for index in range(6)),
        bonds=tuple(
            single_bond(index, index, (index + 1) % 6)
            for index in range(6)
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(6)),
                bonds=tuple(BondId(index) for index in range(6)),
            ),
        ),
    )


def triangle_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=tuple(atom(index, "C") for index in range(3)),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
            single_bond(2, 2, 0),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
        ),
    )


def cycle_plus_isolate_component_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=tuple(atom(index, "C") for index in range(4)),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
            single_bond(2, 2, 0),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
