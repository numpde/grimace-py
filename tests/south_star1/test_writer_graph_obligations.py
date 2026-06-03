"""Writer residual graph-obligation classifier tests."""

from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

import grimace._south_star1.writer_snapshot as writer_snapshot
import grimace._south_star1.writer_transitions as writer_transitions
from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.facts import BondOrder
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
from grimace._south_star1.writer_graph_obligations import WriterBoundaryIncidence
from grimace._south_star1.writer_graph_obligations import WriterBoundaryOwnerKind
from grimace._south_star1.writer_graph_obligations import WriterEdgeObligationKind
from grimace._south_star1.writer_graph_obligations import WriterGraphObligationSummary
from grimace._south_star1.writer_graph_obligations import WriterResidualAttachment
from grimace._south_star1.writer_graph_obligations import WriterResidualAttachmentAction
from grimace._south_star1.writer_graph_obligations import WriterResidualAttachmentActionKind
from grimace._south_star1.writer_graph_obligations import WriterResidualAttachmentState
from grimace._south_star1.writer_graph_obligations import build_writer_graph_obligation_context
from grimace._south_star1.writer_graph_obligations import build_writer_block_cut_metadata
from grimace._south_star1.writer_graph_obligations import classify_writer_edge_obligations
from grimace._south_star1.writer_graph_obligations import classify_writer_residual_attachments
from grimace._south_star1.writer_graph_obligations import validate_writer_initial_support_graph_surface
from grimace._south_star1.writer_graph_obligations import validate_writer_edge_obligation_partition
from grimace._south_star1.writer_graph_obligations import writer_boundary_incidence_sort_tuple
from grimace._south_star1.writer_graph_obligations import writer_edge_obligation_partition_sort_tuple
from grimace._south_star1.writer_graph_obligations import writer_graph_completion_status
from grimace._south_star1.writer_graph_obligations import writer_residual_attachment_action_incidences
from grimace._south_star1.writer_graph_obligations import writer_residual_attachment_action_incidences_for_atom
from grimace._south_star1.writer_graph_obligations import writer_residual_attachment_sort_tuple
from grimace._south_star1.writer_state import ComponentCursor
from grimace._south_star1.writer_state import ObligationState
from grimace._south_star1.writer_state import WriterAtomFrame
from grimace._south_star1.writer_state import WriterClosedClosure
from grimace._south_star1.writer_state import WriterClosureLabel
from grimace._south_star1.writer_state import WriterOpenClosureEndpoint
from grimace._south_star1.writer_state import WriterPolicyState
from grimace._south_star1.writer_state import WriterRingLabelState
from grimace._south_star1.writer_state import WriterRingState
from grimace._south_star1.writer_state import WriterRingStateKey
from grimace._south_star1.writer_state import WriterState
from grimace._south_star1.writer_state import writer_state_from_key
from grimace._south_star1.writer_state import writer_state_key
from grimace._south_star1.writer_state import writer_state_key_sort_tuple
from grimace._south_star1.writer_stereo import empty_writer_stereo_state
from grimace._south_star1.writer_transitions import legal_writer_transitions
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import bond
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import single_bond


def _synthetic_action_summary(
    *,
    attachments: tuple[WriterResidualAttachment, ...],
    actions: tuple[WriterResidualAttachmentAction, ...],
) -> WriterGraphObligationSummary:
    return WriterGraphObligationSummary(
        attachments=WriterResidualAttachmentState(attachments=attachments),
        attachment_actions=actions,
        boundary_by_owner_atom=(),
        boundary_by_pending_parent=(),
        has_cyclic_attachment=False,
        has_unsupported_attachment=False,
    )


class WriterGraphObligationsTest(unittest.TestCase):
    def test_prepared_metadata_caches_block_cut_and_component_surfaces(self) -> None:
        prepared = _prepare(cco_facts())

        self.assertEqual(
            prepared.writer_graph_metadata.block_cut,
            build_writer_block_cut_metadata(prepared),
        )
        self.assertEqual(len(prepared.writer_graph_metadata.component_surfaces), 1)
        surface = prepared.writer_graph_metadata.component_surfaces[0]
        self.assertEqual(surface.component_index, 0)
        self.assertEqual(surface.atoms, frozenset((AtomId(0), AtomId(1), AtomId(2))))
        self.assertEqual(surface.bonds, frozenset((BondId(0), BondId(1))))
        self.assertTrue(surface.connected)
        self.assertTrue(surface.tree)
        self.assertEqual(surface.cyclic_rank, 0)
        self.assertEqual(surface.cyclic_block_ids, frozenset())
        self.assertIsNone(surface.unsupported_reason)

    def test_cached_block_cut_matches_fresh_metadata_for_cyclic_shapes(self) -> None:
        for facts in (triangle_facts(), six_ring_facts()):
            prepared = _prepare(facts)
            self.assertEqual(
                prepared.writer_graph_metadata.block_cut,
                build_writer_block_cut_metadata(prepared),
            )

    def test_component_surface_marks_cyclic_and_malformed_components_unsupported(self) -> None:
        triangle = _prepare(triangle_facts()).writer_graph_metadata.component_surfaces[0]
        malformed = _prepare(
            cycle_plus_isolate_component_facts()
        ).writer_graph_metadata.component_surfaces[0]

        self.assertTrue(triangle.connected)
        self.assertFalse(triangle.tree)
        self.assertEqual(triangle.cyclic_rank, 1)
        self.assertIsNotNone(triangle.unsupported_reason)
        self.assertFalse(malformed.connected)
        self.assertFalse(malformed.tree)
        self.assertEqual(malformed.cyclic_rank, 1)
        self.assertIsNotNone(malformed.unsupported_reason)

    def test_context_builder_returns_partition_and_residual_summary(self) -> None:
        prepared = _prepare(cco_facts())
        key = _cco_after_second_atom_key(prepared, _writer_options(rooted_at_atom=0))

        context = build_writer_graph_obligation_context(prepared, key)

        self.assertIs(context.prepared_metadata, prepared.writer_graph_metadata)
        self.assertEqual(
            tuple((item.bond, item.kind) for item in context.edge_partition.obligations),
            (
                (BondId(0), WriterEdgeObligationKind.TREE_ENTRY),
                (BondId(1), WriterEdgeObligationKind.BOUNDARY_INCIDENCE),
            ),
        )
        self.assertFalse(context.residual_summary.has_cyclic_attachment)
        self.assertEqual(len(context.residual_summary.attachments.attachments), 1)
        self.assertEqual(
            tuple(action.kind for action in context.residual_summary.attachment_actions),
            (WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,),
        )

    def test_context_builder_exposes_cyclic_summary_without_closure_candidate(self) -> None:
        prepared = _prepare(triangle_facts())
        key = _emitted_root_key(prepared, root=AtomId(0))

        context = build_writer_graph_obligation_context(prepared, key)

        self.assertNotIn(
            WriterEdgeObligationKind.CLOSURE_CANDIDATE,
            {item.kind for item in context.edge_partition.obligations},
        )
        self.assertTrue(context.residual_summary.has_cyclic_attachment)
        self.assertEqual(
            tuple(action.kind for action in context.residual_summary.attachment_actions),
            (WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY,),
        )

    def test_residual_attachment_action_incidences_expand_boundaries(self) -> None:
        attachment = WriterResidualAttachment(
            attachment_id=7,
            atoms=frozenset((AtomId(2), AtomId(3))),
            latent_bonds=frozenset((BondId(2),)),
            boundary=(
                WriterBoundaryIncidence(
                    bond=BondId(0),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(2),
                    owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
                ),
                WriterBoundaryIncidence(
                    bond=BondId(1),
                    written_atom=AtomId(1),
                    residual_atom=AtomId(3),
                    owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
                ),
            ),
            cyclic_rank=1,
            block_ids=frozenset((4,)),
        )
        action = WriterResidualAttachmentAction(
            attachment_id=7,
            kind=WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY,
            owner_atoms=(AtomId(0), AtomId(1)),
            boundary_bonds=(BondId(0), BondId(1)),
        )
        summary = _synthetic_action_summary(
            attachments=(attachment,),
            actions=(action,),
        )

        incidences = writer_residual_attachment_action_incidences(summary)

        self.assertEqual(
            tuple(item.incidence.bond for item in incidences),
            (BondId(0), BondId(1)),
        )
        self.assertEqual(tuple(item.action for item in incidences), (action, action))
        self.assertEqual(
            tuple(item.attachment for item in incidences),
            (attachment, attachment),
        )
        active_only = writer_residual_attachment_action_incidences_for_atom(
            summary,
            AtomId(0),
        )
        self.assertEqual(len(active_only), 1)
        self.assertEqual(active_only[0].incidence.bond, BondId(0))

    def test_residual_attachment_action_incidences_reject_unknown_attachment(self) -> None:
        action = WriterResidualAttachmentAction(
            attachment_id=99,
            kind=WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY,
            owner_atoms=(AtomId(0),),
            boundary_bonds=(BondId(0),),
        )
        summary = _synthetic_action_summary(
            attachments=(),
            actions=(action,),
        )

        with self.assertRaises(ValueError):
            writer_residual_attachment_action_incidences(summary)

    def test_residual_attachment_action_incidences_reject_duplicate_attachment_ids(self) -> None:
        first = WriterResidualAttachment(
            attachment_id=7,
            atoms=frozenset((AtomId(2),)),
            latent_bonds=frozenset((BondId(2),)),
            boundary=(),
            cyclic_rank=0,
            block_ids=frozenset(),
        )
        second = WriterResidualAttachment(
            attachment_id=7,
            atoms=frozenset((AtomId(3),)),
            latent_bonds=frozenset((BondId(3),)),
            boundary=(),
            cyclic_rank=0,
            block_ids=frozenset(),
        )
        summary = _synthetic_action_summary(
            attachments=(first, second),
            actions=(),
        )

        with self.assertRaises(ValueError):
            writer_residual_attachment_action_incidences(summary)

    def test_context_builder_exposes_closure_candidate_partition(self) -> None:
        prepared = _prepare(triangle_facts())
        key = _triangle_all_visited_two_written_key()

        context = build_writer_graph_obligation_context(prepared, key)

        self.assertIn(
            WriterEdgeObligationKind.CLOSURE_CANDIDATE,
            {item.kind for item in context.edge_partition.obligations},
        )
        self.assertTrue(context.residual_summary.has_cyclic_attachment)
        self.assertEqual(context.residual_summary.attachment_actions, ())

    def test_supported_graph_surface_accepts_all_acyclic_components(self) -> None:
        validate_writer_initial_support_graph_surface(_prepare(chain_plus_singleton_facts()))

    def test_supported_graph_surface_rejects_cyclic_and_malformed_components(self) -> None:
        for facts in (
            triangle_facts(),
            singleton_plus_triangle_facts(),
            triangle_plus_singleton_facts(),
            cycle_plus_isolate_component_facts(),
        ):
            with self.assertRaises(SouthStarError):
                validate_writer_initial_support_graph_surface(_prepare(facts))

    def test_production_paths_use_cached_writer_graph_context(self) -> None:
        child_source = inspect.getsource(
            writer_transitions._child_obligations_from_context
        )
        legal_source = inspect.getsource(writer_transitions.legal_writer_transitions)
        cursor_source = inspect.getsource(
            writer_snapshot.validate_writer_cursor_against_prepared
        )

        self.assertIn("build_writer_transition_expansion_context", legal_source)
        self.assertNotIn("build_writer_graph_obligation_context", child_source)
        self.assertNotIn("build_writer_block_cut_metadata", child_source)
        self.assertEqual(cursor_source.count("build_writer_graph_obligation_context"), 1)
        self.assertNotIn("build_writer_block_cut_metadata", cursor_source)
        self.assertNotIn("validate_writer_edge_obligation_partition", cursor_source)

    def test_cco_prefix_edge_partition_tracks_tree_and_boundary(self) -> None:
        prepared = _prepare(cco_facts())
        key = _cco_after_second_atom_key(prepared, _writer_options(rooted_at_atom=0))

        partition = classify_writer_edge_obligations(prepared, key)

        validate_writer_edge_obligation_partition(prepared, key, partition)
        self.assertEqual(
            tuple((item.bond, item.kind) for item in partition.obligations),
            (
                (BondId(0), WriterEdgeObligationKind.TREE_ENTRY),
                (BondId(1), WriterEdgeObligationKind.BOUNDARY_INCIDENCE),
            ),
        )
        self.assertEqual(
            writer_edge_obligation_partition_sort_tuple(partition),
            tuple(
                (int(item.bond), item.kind.value, int(item.a), int(item.b))
                for item in partition.obligations
            ),
        )

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
        self.assertEqual(
            tuple(action.kind for action in summary.attachment_actions),
            (WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,),
        )

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
        self.assertEqual(
            tuple(action.kind for action in summary.attachment_actions),
            (WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,),
        )

    def test_post_bond_pending_state_partitions_pending_entry(self) -> None:
        prepared = _prepare(carbonyl_facts())
        cursor = initial_writer_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )
        after_c = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_bond = writer_frontier_choices(prepared, after_c).choices[0].successor
        key = after_bond.weighted_states[0][0]

        partition = classify_writer_edge_obligations(prepared, key)

        validate_writer_edge_obligation_partition(prepared, key, partition)
        self.assertEqual(
            tuple((item.bond, item.kind) for item in partition.obligations),
            ((BondId(0), WriterEdgeObligationKind.PENDING_ENTRY),),
        )

    def test_ring_entry_classifies_one_cyclic_attachment_not_two_children(self) -> None:
        prepared = _prepare(six_ring_facts())
        key = _emitted_root_key(prepared, root=AtomId(0))

        partition = classify_writer_edge_obligations(prepared, key)
        summary = _summary(prepared, key)

        self.assertNotIn(
            WriterEdgeObligationKind.CLOSURE_CANDIDATE,
            {item.kind for item in partition.obligations},
        )
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
        self.assertEqual(
            tuple(action.kind for action in summary.attachment_actions),
            (WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY,),
        )
        transitions = legal_writer_transitions(prepared, writer_state_from_key(key))

        self.assertEqual(
            {transition.kind for transition in transitions},
            {writer_transitions.WriterTransitionKind.OPEN_CLOSURE_ENDPOINT},
        )
        self.assertEqual({transition.emitted_text for transition in transitions}, {"1"})

    def test_single_boundary_cyclic_residual_is_cyclic_tree_entry(self) -> None:
        prepared = _prepare(triangle_tail_facts())
        key = _emitted_root_key(prepared, root=AtomId(0))

        summary = _summary(prepared, key)

        self.assertEqual(len(summary.attachments.attachments), 1)
        self.assertEqual(
            tuple(action.kind for action in summary.attachment_actions),
            (WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY,),
        )
        attachment = summary.attachments.attachments[0]
        self.assertEqual(len(attachment.boundary), 1)
        self.assertEqual(attachment.boundary[0].bond, BondId(0))

        transitions = legal_writer_transitions(prepared, writer_state_from_key(key))

        self.assertNotIn(
            writer_transitions.WriterTransitionKind.OPEN_CLOSURE_ENDPOINT,
            {transition.kind for transition in transitions},
        )
        self.assertEqual(
            {transition.kind for transition in transitions},
            {writer_transitions.WriterTransitionKind.ENTER_INLINE_CHILD},
        )

    def test_triangle_partial_state_with_mixed_boundary_ownership_is_blocked(self) -> None:
        prepared = _prepare(triangle_facts())
        key = _triangle_two_visited_key()

        partition = classify_writer_edge_obligations(prepared, key)
        summary = _summary(prepared, key)

        validate_writer_edge_obligation_partition(prepared, key, partition)
        self.assertEqual(
            tuple((item.bond, item.kind) for item in partition.obligations),
            (
                (BondId(0), WriterEdgeObligationKind.TREE_ENTRY),
                (BondId(1), WriterEdgeObligationKind.BOUNDARY_INCIDENCE),
                (BondId(2), WriterEdgeObligationKind.BOUNDARY_INCIDENCE),
            ),
        )
        self.assertTrue(summary.has_cyclic_attachment)
        self.assertEqual(len(summary.attachments.attachments), 1)
        self.assertEqual(len(summary.attachments.attachments[0].boundary), 2)
        self.assertEqual(
            tuple(action.kind for action in summary.attachment_actions),
            (WriterResidualAttachmentActionKind.BLOCKED_UNOWNED,),
        )
        transitions = legal_writer_transitions(prepared, writer_state_from_key(key))

        self.assertNotIn(
            writer_transitions.WriterTransitionKind.OPEN_CLOSURE_ENDPOINT,
            {transition.kind for transition in transitions},
        )

    def test_frozen_single_boundary_residual_is_blocked_unowned(self) -> None:
        prepared = _prepare(cco_facts())
        key = _cco_frozen_single_boundary_key()

        summary = _summary(prepared, key)

        self.assertEqual(len(summary.attachments.attachments), 1)
        self.assertEqual(len(summary.attachments.attachments[0].boundary), 1)
        self.assertEqual(
            tuple(action.kind for action in summary.attachment_actions),
            (WriterResidualAttachmentActionKind.BLOCKED_UNOWNED,),
        )
        transitions = legal_writer_transitions(prepared, writer_state_from_key(key))

        self.assertNotIn(
            writer_transitions.WriterTransitionKind.ENTER_INLINE_CHILD,
            {transition.kind for transition in transitions},
        )

    def test_triangle_closure_candidate_is_explicit_and_fails_closed(self) -> None:
        prepared = _prepare(triangle_facts())
        key = _triangle_all_visited_two_written_key()

        partition = classify_writer_edge_obligations(prepared, key)
        summary = _summary(prepared, key)

        validate_writer_edge_obligation_partition(prepared, key, partition)
        self.assertEqual(
            tuple((item.bond, item.kind) for item in partition.obligations),
            (
                (BondId(0), WriterEdgeObligationKind.TREE_ENTRY),
                (BondId(1), WriterEdgeObligationKind.TREE_ENTRY),
                (BondId(2), WriterEdgeObligationKind.CLOSURE_CANDIDATE),
            ),
        )
        self.assertTrue(summary.has_cyclic_attachment)
        with self.assertRaises(SouthStarError):
            legal_writer_transitions(prepared, writer_state_from_key(key))

    def test_open_closure_endpoint_is_partitioned_and_cuts_residual_attachment(self) -> None:
        prepared = _prepare(triangle_facts())
        key = _triangle_root_with_open_closure_key()

        partition = classify_writer_edge_obligations(prepared, key)
        summary = _summary(prepared, key)

        validate_writer_edge_obligation_partition(prepared, key, partition)
        self.assertEqual(
            tuple((item.bond, item.kind) for item in partition.obligations),
            (
                (BondId(0), WriterEdgeObligationKind.BOUNDARY_INCIDENCE),
                (BondId(1), WriterEdgeObligationKind.LATENT_RESIDUAL),
                (BondId(2), WriterEdgeObligationKind.OPEN_CLOSURE_ENDPOINT),
            ),
        )
        self.assertFalse(summary.has_cyclic_attachment)
        self.assertEqual(len(summary.attachments.attachments), 1)
        self.assertEqual(
            summary.attachments.attachments[0].boundary[0].bond,
            BondId(0),
        )

    def test_closed_closure_endpoint_is_partitioned_and_excluded_from_residuals(self) -> None:
        prepared = _prepare(triangle_facts())
        key = _triangle_closed_closure_key()

        partition = classify_writer_edge_obligations(prepared, key)
        summary = _summary(prepared, key)

        validate_writer_edge_obligation_partition(prepared, key, partition)
        self.assertEqual(
            tuple((item.bond, item.kind) for item in partition.obligations),
            (
                (BondId(0), WriterEdgeObligationKind.TREE_ENTRY),
                (BondId(1), WriterEdgeObligationKind.TREE_ENTRY),
                (BondId(2), WriterEdgeObligationKind.CLOSED_CLOSURE),
            ),
        )
        self.assertFalse(summary.has_cyclic_attachment)
        self.assertEqual(summary.attachments.attachments, ())

    def test_graph_completion_accepts_acyclic_terminal_state(self) -> None:
        prepared = _prepare(cco_facts())
        key = _cco_terminal_key()
        context = build_writer_graph_obligation_context(prepared, key)

        status = writer_graph_completion_status(prepared, key, context)

        self.assertTrue(status.complete)
        self.assertEqual(status.unresolved_kinds, ())
        self.assertEqual(status.unresolved_bonds, ())

    def test_graph_completion_reports_boundary_prefix(self) -> None:
        prepared = _prepare(cco_facts())
        key = _cco_after_second_atom_key(prepared, _writer_options(rooted_at_atom=0))
        context = build_writer_graph_obligation_context(prepared, key)

        status = writer_graph_completion_status(prepared, key, context)

        self.assertFalse(status.complete)
        self.assertEqual(
            status.unresolved_kinds,
            (WriterEdgeObligationKind.BOUNDARY_INCIDENCE,),
        )
        self.assertEqual(status.unresolved_bonds, (BondId(1),))

    def test_graph_completion_accepts_closed_closure_terminal(self) -> None:
        prepared = _prepare(triangle_facts())
        key = _triangle_closed_closure_key()
        context = build_writer_graph_obligation_context(prepared, key)

        status = writer_graph_completion_status(prepared, key, context)

        self.assertTrue(status.complete)

    def test_graph_completion_reports_open_closure_endpoint(self) -> None:
        prepared = _prepare(triangle_facts())
        key = _triangle_root_with_open_closure_key()
        context = build_writer_graph_obligation_context(prepared, key)

        status = writer_graph_completion_status(prepared, key, context)

        self.assertFalse(status.complete)
        self.assertIn(
            WriterEdgeObligationKind.OPEN_CLOSURE_ENDPOINT,
            status.unresolved_kinds,
        )

    def test_graph_completion_reports_closure_candidate(self) -> None:
        prepared = _prepare(triangle_facts())
        key = _triangle_all_visited_two_written_key()
        context = build_writer_graph_obligation_context(prepared, key)

        status = writer_graph_completion_status(prepared, key, context)

        self.assertFalse(status.complete)
        self.assertEqual(
            status.unresolved_kinds,
            (WriterEdgeObligationKind.CLOSURE_CANDIDATE,),
        )
        self.assertEqual(status.unresolved_bonds, (BondId(2),))

    def test_open_closure_bond_cannot_also_be_tree_entry(self) -> None:
        prepared = _prepare(triangle_facts())
        key = replace(
            _triangle_root_with_open_closure_key(),
            written_bonds=frozenset((BondId(2),)),
        )
        partition = classify_writer_edge_obligations(prepared, key)

        with self.assertRaises(SouthStarError):
            validate_writer_edge_obligation_partition(prepared, key, partition)

    def test_duplicate_open_closure_bond_records_reject(self) -> None:
        prepared = _prepare(triangle_facts())
        first = _closure_label()
        second = WriterClosureLabel(value=2, text="2")
        key = replace(
            _triangle_root_with_open_closure_key(),
            ring_state=WriterRingStateKey(
                open_endpoints=(
                    WriterOpenClosureEndpoint(
                        bond=BondId(2),
                        first_atom=AtomId(0),
                        second_atom=AtomId(2),
                        label=first,
                        first_endpoint_text="1",
                        first_endpoint_bond_text="",
                    ),
                    WriterOpenClosureEndpoint(
                        bond=BondId(2),
                        first_atom=AtomId(0),
                        second_atom=AtomId(2),
                        label=second,
                        first_endpoint_text="2",
                        first_endpoint_bond_text="",
                    ),
                ),
                label_state=WriterRingLabelState(allocated=(first, second)),
            ),
        )
        partition = classify_writer_edge_obligations(prepared, key)

        with self.assertRaises(SouthStarError):
            validate_writer_edge_obligation_partition(prepared, key, partition)

    def test_duplicate_closed_closure_bond_records_reject(self) -> None:
        prepared = _prepare(triangle_facts())
        first = _closure_label()
        second = WriterClosureLabel(value=2, text="2")
        key = replace(
            _triangle_closed_closure_key(),
            ring_state=WriterRingStateKey(
                closed_closures=(
                    WriterClosedClosure(
                        bond=BondId(2),
                        first_atom=AtomId(0),
                        second_atom=AtomId(2),
                        label=first,
                        first_endpoint_text="1",
                        second_endpoint_text="1",
                        first_endpoint_bond_text="",
                        second_endpoint_bond_text="",
                    ),
                    WriterClosedClosure(
                        bond=BondId(2),
                        first_atom=AtomId(0),
                        second_atom=AtomId(2),
                        label=second,
                        first_endpoint_text="2",
                        second_endpoint_text="2",
                        first_endpoint_bond_text="",
                        second_endpoint_bond_text="",
                    ),
                ),
                label_state=WriterRingLabelState(reusable=(first, second)),
            ),
        )
        partition = classify_writer_edge_obligations(prepared, key)

        with self.assertRaises(SouthStarError):
            validate_writer_edge_obligation_partition(prepared, key, partition)

    def test_open_and_closed_closure_same_bond_rejects(self) -> None:
        prepared = _prepare(triangle_facts())
        label = _closure_label()
        key = replace(
            _triangle_closed_closure_key(),
            ring_state=WriterRingStateKey(
                open_endpoints=(
                    WriterOpenClosureEndpoint(
                        bond=BondId(2),
                        first_atom=AtomId(0),
                        second_atom=AtomId(2),
                        label=label,
                        first_endpoint_text="1",
                        first_endpoint_bond_text="",
                    ),
                ),
                closed_closures=(
                    WriterClosedClosure(
                        bond=BondId(2),
                        first_atom=AtomId(0),
                        second_atom=AtomId(2),
                        label=label,
                        first_endpoint_text="1",
                        second_endpoint_text="1",
                        first_endpoint_bond_text="",
                        second_endpoint_bond_text="",
                    ),
                ),
                label_state=WriterRingLabelState(allocated=(label,)),
            ),
        )
        partition = classify_writer_edge_obligations(prepared, key)

        with self.assertRaises(SouthStarError):
            validate_writer_edge_obligation_partition(prepared, key, partition)

    def test_nonempty_ring_state_key_round_trips_and_sorts_structurally(self) -> None:
        key = _triangle_closed_closure_key()

        round_tripped = writer_state_key(writer_state_from_key(key))

        self.assertEqual(round_tripped, key)
        self.assertEqual(
            writer_state_key_sort_tuple(key)[7],
            (
                (),
                ((2, 0, 2, (1, "1"), "1", "1", "", ""),),
                ((), ((1, "1"),)),
            ),
        )

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
        self.assertEqual(
            sorted(
                (action.kind for action in summary.attachment_actions),
                key=lambda kind: kind.value,
            ),
            [
                WriterResidualAttachmentActionKind.BLOCKED_ORPHAN,
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY,
            ],
        )

    def test_orphan_residual_bond_is_latent_with_empty_boundary_attachment(self) -> None:
        prepared = _prepare(chain_plus_orphan_chain_same_component_facts())
        key = _emitted_root_key(prepared, root=AtomId(0))

        partition = classify_writer_edge_obligations(prepared, key)
        summary = _summary(prepared, key)

        validate_writer_edge_obligation_partition(prepared, key, partition)
        self.assertEqual(
            tuple((item.bond, item.kind) for item in partition.obligations),
            (
                (BondId(0), WriterEdgeObligationKind.BOUNDARY_INCIDENCE),
                (BondId(1), WriterEdgeObligationKind.LATENT_RESIDUAL),
            ),
        )
        self.assertEqual(len(summary.attachments.attachments), 2)
        self.assertEqual(
            tuple(
                sorted(
                    len(attachment.boundary)
                    for attachment in summary.attachments.attachments
                )
            ),
            (0, 1),
        )
        self.assertIn(
            WriterResidualAttachmentActionKind.BLOCKED_ORPHAN,
            {action.kind for action in summary.attachment_actions},
        )

    def test_multi_boundary_residual_without_open_owner_is_blocked_unowned(self) -> None:
        prepared = _prepare(triangle_with_frozen_tail_facts())
        key = _triangle_with_frozen_tail_key()

        summary = _summary(prepared, key)

        self.assertEqual(len(summary.attachments.attachments), 1)
        self.assertEqual(
            tuple(action.kind for action in summary.attachment_actions),
            (WriterResidualAttachmentActionKind.BLOCKED_UNOWNED,),
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
        prepared.writer_graph_metadata.block_cut,
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


def _triangle_two_visited_key():
    return writer_state_key(
        WriterState(
            component_cursor=ComponentCursor(
                component_index=0,
                component_roots=(AtomId(0),),
            ),
            active=WriterAtomFrame(
                atom=AtomId(1),
                parent=AtomId(0),
                incoming_bond=BondId(0),
                atom_emitted=True,
            ),
            branch_stack=(),
            visited_atoms=frozenset((AtomId(0), AtomId(1))),
            written_bonds=frozenset((BondId(0),)),
            obligations=ObligationState(),
            ring_state=WriterRingState(),
            stereo_state=empty_writer_stereo_state(),
            policy_state=WriterPolicyState(),
        )
    )


def _cco_terminal_key():
    return writer_state_key(
        WriterState(
            component_cursor=ComponentCursor(
                component_index=0,
                component_roots=(AtomId(0),),
            ),
            active=WriterAtomFrame(
                atom=AtomId(2),
                parent=AtomId(1),
                incoming_bond=BondId(1),
                atom_emitted=True,
            ),
            branch_stack=(),
            visited_atoms=frozenset((AtomId(0), AtomId(1), AtomId(2))),
            written_bonds=frozenset((BondId(0), BondId(1))),
            obligations=ObligationState(),
            ring_state=WriterRingState(),
            stereo_state=empty_writer_stereo_state(),
            policy_state=WriterPolicyState(),
        )
    )


def _cco_frozen_single_boundary_key():
    return writer_state_key(
        WriterState(
            component_cursor=ComponentCursor(
                component_index=0,
                component_roots=(AtomId(0),),
            ),
            active=WriterAtomFrame(
                atom=AtomId(0),
                parent=None,
                incoming_bond=None,
                atom_emitted=True,
            ),
            branch_stack=(),
            visited_atoms=frozenset((AtomId(0), AtomId(1))),
            written_bonds=frozenset((BondId(0),)),
            obligations=ObligationState(),
            ring_state=WriterRingState(),
            stereo_state=empty_writer_stereo_state(),
            policy_state=WriterPolicyState(),
        )
    )


def _triangle_all_visited_two_written_key():
    return writer_state_key(
        WriterState(
            component_cursor=ComponentCursor(
                component_index=0,
                component_roots=(AtomId(0),),
            ),
            active=WriterAtomFrame(
                atom=AtomId(2),
                parent=AtomId(1),
                incoming_bond=BondId(1),
                atom_emitted=True,
            ),
            branch_stack=(),
            visited_atoms=frozenset((AtomId(0), AtomId(1), AtomId(2))),
            written_bonds=frozenset((BondId(0), BondId(1))),
            obligations=ObligationState(),
            ring_state=WriterRingState(),
            stereo_state=empty_writer_stereo_state(),
            policy_state=WriterPolicyState(),
        )
    )


def _triangle_with_frozen_tail_key():
    return writer_state_key(
        WriterState(
            component_cursor=ComponentCursor(
                component_index=0,
                component_roots=(AtomId(0),),
            ),
            active=WriterAtomFrame(
                atom=AtomId(3),
                parent=AtomId(1),
                incoming_bond=BondId(3),
                atom_emitted=True,
            ),
            branch_stack=(),
            visited_atoms=frozenset((AtomId(0), AtomId(1), AtomId(3))),
            written_bonds=frozenset((BondId(0), BondId(3))),
            obligations=ObligationState(),
            ring_state=WriterRingState(),
            stereo_state=empty_writer_stereo_state(),
            policy_state=WriterPolicyState(),
        )
    )


def _closure_label() -> WriterClosureLabel:
    return WriterClosureLabel(value=1, text="1")


def _triangle_root_with_open_closure_key():
    label = _closure_label()
    return replace(
        _emitted_root_key(_prepare(triangle_facts()), root=AtomId(0)),
        ring_state=WriterRingStateKey(
            open_endpoints=(
                WriterOpenClosureEndpoint(
                    bond=BondId(2),
                    first_atom=AtomId(0),
                    second_atom=AtomId(2),
                    label=label,
                    first_endpoint_text="1",
                    first_endpoint_bond_text="",
                ),
            ),
            label_state=WriterRingLabelState(allocated=(label,)),
        ),
    )


def _triangle_closed_closure_key():
    label = _closure_label()
    return replace(
        _triangle_all_visited_two_written_key(),
        ring_state=WriterRingStateKey(
            closed_closures=(
                WriterClosedClosure(
                    bond=BondId(2),
                    first_atom=AtomId(0),
                    second_atom=AtomId(2),
                    label=label,
                    first_endpoint_text="1",
                    second_endpoint_text="1",
                    first_endpoint_bond_text="",
                    second_endpoint_bond_text="",
                ),
            ),
            label_state=WriterRingLabelState(reusable=(label,)),
        ),
    )


def carbonyl_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "O")),
        bonds=(bond(0, 0, 1, BondOrder.DOUBLE),),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1)),
                bonds=(BondId(0),),
            ),
        ),
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


def triangle_tail_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=tuple(atom(index, "C") for index in range(4)),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
            single_bond(2, 2, 3),
            single_bond(3, 3, 1),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2), BondId(3)),
            ),
        ),
    )


def triangle_with_frozen_tail_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=tuple(atom(index, "C") for index in range(4)),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
            single_bond(2, 2, 0),
            single_bond(3, 1, 3),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2), BondId(3)),
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


def chain_plus_singleton_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "O")),
        bonds=(single_bond(0, 0, 1),),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1)),
                bonds=(BondId(0),),
            ),
            ComponentFacts(
                id=ComponentId(1),
                atoms=(AtomId(2),),
                bonds=(),
            ),
        ),
    )


def singleton_plus_triangle_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "O"), atom(1, "C"), atom(2, "C"), atom(3, "C")),
        bonds=(
            single_bond(0, 1, 2),
            single_bond(1, 2, 3),
            single_bond(2, 3, 1),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0),),
                bonds=(),
            ),
            ComponentFacts(
                id=ComponentId(1),
                atoms=(AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
        ),
    )


def triangle_plus_singleton_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C"), atom(3, "O")),
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
            ComponentFacts(
                id=ComponentId(1),
                atoms=(AtomId(3),),
                bonds=(),
            ),
        ),
    )


def chain_plus_orphan_chain_same_component_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=tuple(atom(index, "C") for index in range(4)),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 2, 3),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1)),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
