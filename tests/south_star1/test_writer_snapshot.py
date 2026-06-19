"""Writer frontier snapshot tests."""

from __future__ import annotations

from dataclasses import replace
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import BondFacts
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import BondTextChoice
from grimace._south_star1.policy import BondTextDomain
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.ordinary_policy import OrdinaryPolicyOptions
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.residual_constraints import DirectionalCarrierResidual
from grimace._south_star1.residual_constraints import DirectionalNormalizedSign
from grimace._south_star1.residual_constraints import DirectionalResidualFactor
from grimace._south_star1.residual_constraints import ResidualPropagationKind
from grimace._south_star1.residual_constraints import ResidualStore
from grimace._south_star1.residual_constraints import ResidualStoreValueSnapshot
from grimace._south_star1.residual_constraints import add_factor_and_propagate
from grimace._south_star1.residual_constraints import direction_var
from grimace._south_star1.residual_constraints import tetra_var
from grimace._south_star1.writer_frontier import count_writer_cursor_completions
from grimace._south_star1.writer_frontier import count_writer_frontier_support
from grimace._south_star1.writer_frontier import WriterFrontierCursor
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_frontier import _initial_writer_transition_frontier_cursor as initial_writer_transition_frontier_cursor
from grimace._south_star1.writer_frontier import writer_frontier_choices
from grimace._south_star1.writer_snapshot import WriterDecoderBoundary
from grimace._south_star1.writer_snapshot import WriterFrontierFrame
from grimace._south_star1.writer_snapshot import WriterSearchSnapshot
from grimace._south_star1.writer_snapshot import _prepared_identity
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot import resume_writer_frontier_choices_from_snapshot
from grimace._south_star1.writer_snapshot import validate_writer_cursor_against_prepared
from grimace._south_star1.writer_snapshot import validate_writer_search_snapshot
from grimace._south_star1.writer_snapshot import writer_frontier_cursor_from_snapshot
from grimace._south_star1.writer_state import ComponentCursor
from grimace._south_star1.writer_state import ObligationState
from grimace._south_star1.writer_state import ObligationStateKey
from grimace._south_star1.writer_state import PendingEntryPhase
from grimace._south_star1.writer_state import PendingWriterEntry
from grimace._south_star1.writer_state import WriterAtomFrame
from grimace._south_star1.writer_state import WriterBranchFrame
from grimace._south_star1.writer_state import WriterClosedClosure
from grimace._south_star1.writer_state import WriterClosureLabel
from grimace._south_star1.writer_state import WriterOpenClosureEndpoint
from grimace._south_star1.writer_state import WriterPolicyState
from grimace._south_star1.writer_state import WriterRingLabelState
from grimace._south_star1.writer_state import WriterRingState
from grimace._south_star1.writer_state import WriterRingStateKey
from grimace._south_star1.writer_state import WriterState
from grimace._south_star1.writer_state import writer_state_key
from grimace._south_star1.writer_stereo import empty_writer_stereo_state
from grimace._south_star1.writer_stereo import reconstruct_writer_local_order_records
from grimace._south_star1.writer_stereo import _writer_stereo_relation_definitions
from grimace._south_star1.writer_stereo import reconstruct_writer_stereo_residual_snapshot
from grimace._south_star1.writer_stereo import WriterAtomOccurrenceRecord
from grimace._south_star1.writer_stereo import WriterBondOccurrenceRecord
from grimace._south_star1.writer_stereo import WriterLocalOrderRecord
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


class WriterSnapshotTest(unittest.TestCase):
    def test_weighted_cursor_snapshot_round_trips_choices_and_counts(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
            decoder_boundary=WriterDecoderBoundary(consumed_token_count=0),
        )

        self.assertEqual(
            writer_frontier_cursor_from_snapshot(snapshot, prepared=prepared),
            cursor,
        )
        self.assertEqual(
            resume_writer_frontier_choices_from_snapshot(snapshot, prepared=prepared),
            writer_frontier_choices(prepared, cursor),
        )
        self.assertEqual(
            count_writer_frontier_support(prepared, snapshot.cursor.support_state),
            count_writer_frontier_support(prepared, cursor.support_state),
        )
        self.assertEqual(
            count_writer_cursor_completions(prepared, snapshot.cursor),
            count_writer_cursor_completions(prepared, cursor),
        )

    def test_internal_cyclic_root_snapshot_round_trips_choices_and_counts(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_transition_frontier_cursor(prepared, options)
        after_root = _only_choice(prepared, cursor, "C").successor

        _assert_snapshot_round_trips_cursor(self, prepared, options, after_root)

    def test_internal_cyclic_open_closure_snapshot_round_trips_choices_and_counts(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_transition_frontier_cursor(prepared, options)
        after_root = _only_choice(prepared, cursor, "C").successor
        opened = _only_choice(prepared, after_root, "1").successor

        _assert_snapshot_round_trips_cursor(self, prepared, options, opened)

    def test_internal_cyclic_closed_closure_snapshot_round_trips_choices_and_counts(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_transition_frontier_cursor(prepared, options)
        after_root = _only_choice(prepared, cursor, "C").successor
        opened = _only_choice(prepared, after_root, "1").successor
        after_first_child = _only_choice(prepared, opened, "C").successor
        after_second_child = _only_choice(prepared, after_first_child, "C").successor
        closed = _only_choice(prepared, after_second_child, "1").successor

        _assert_snapshot_round_trips_cursor(self, prepared, options, closed)

    def test_stereo_residual_snapshot_round_trips(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
        key = after_center.weighted_states[0][0]
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=after_center,
        )

        validate_writer_search_snapshot(snapshot, prepared=prepared)
        self.assertNotEqual(key.stereo_state.residual_snapshot.domains, ())
        self.assertEqual(
            writer_frontier_cursor_from_snapshot(snapshot, prepared=prepared),
            after_center,
        )

    def test_stereo_residual_reconstruction_matches_representative_states(self) -> None:
        tetra_prepared = _prepare(tetrahedral_facts())
        tetra_options = _writer_options(rooted_at_atom=1)
        tetra_cursor = initial_writer_frontier_cursor(tetra_prepared, tetra_options)
        tetra_initial = tetra_cursor.weighted_states[0][0]
        after_f = writer_frontier_choices(tetra_prepared, tetra_cursor).choices[0].successor
        after_center = writer_frontier_choices(tetra_prepared, after_f).choices[0].successor
        tetra_partial = after_center.weighted_states[0][0]
        _terminal_prepared, _terminal_options, tetra_terminal = _terminal_tetra_key()

        directional_prepared = _prepare(directional_facts())
        directional_options = _writer_options(rooted_at_atom=2)
        directional_cursor = initial_writer_frontier_cursor(
            directional_prepared,
            directional_options,
        )
        directional_initial = directional_cursor.weighted_states[0][0]
        after_f = writer_frontier_choices(
            directional_prepared,
            directional_cursor,
        ).choices[0].successor
        directional_partial = after_f.weighted_states[0][0]
        directional_terminal = _first_terminal_key(
            directional_prepared,
            directional_options,
        )

        from tests.south_star1.test_writer_stereo_residual import _two_independent_tetra_facts

        mixed_prepared = _prepare(_two_independent_tetra_facts())
        mixed_options = _writer_options(rooted_at_atom=0)
        mixed_initial = initial_writer_frontier_cursor(
            mixed_prepared,
            mixed_options,
        ).weighted_states[0][0]

        cases = (
            (tetra_prepared, tetra_initial),
            (tetra_prepared, tetra_partial),
            (_terminal_prepared, tetra_terminal),
            (directional_prepared, directional_initial),
            (directional_prepared, directional_partial),
            (directional_prepared, directional_terminal),
            (mixed_prepared, mixed_initial),
        )

        for prepared, key in cases:
            with self.subTest(key=key.active.atom):
                _assert_residual_reconstructs(self, prepared, key)

    def test_tampered_mode_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        tampered = replace(
            snapshot,
            serialization_language=SerializationLanguageMode.EXHAUSTIVE,
        )

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(tampered, prepared=prepared)

    def test_structural_prepared_identity_mismatch_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        changed_facts = replace(
            cco_facts(),
            atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C")),
        )
        changed_prepared = _prepare(changed_facts)

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(snapshot, prepared=changed_prepared)

    def test_unknown_frame_payload_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        tampered = replace(snapshot, frame_stack=snapshot.frame_stack + (object(),))

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(tampered, prepared=prepared)

    def test_dormant_snapshot_frame_types_are_not_exported(self) -> None:
        import grimace._south_star1.writer_snapshot as writer_snapshot

        self.assertFalse(hasattr(writer_snapshot, "WriterTransitionFrame"))
        self.assertFalse(hasattr(writer_snapshot, "WriterStereoResidualFrame"))
        self.assertFalse(hasattr(writer_snapshot, "WriterDelayedFactorFrame"))

    def test_extra_context_frame_payload_is_rejected_until_stack_resume_exists(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=after_center,
        )
        tampered = replace(
            snapshot,
            frame_stack=snapshot.frame_stack + (object(),),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(tampered, prepared=prepared)

    def test_cursor_audit_rejects_unknown_active_atom(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        key = cursor.weighted_states[0][0]
        tampered_key = replace(key, active=replace(key.active, atom=AtomId(99)))

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=_writer_options(),
            )

    def test_cursor_audit_rejects_missing_active_frame(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        key = cursor.weighted_states[0][0]
        tampered_key = replace(key, active=None)

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _unchecked_cursor_with_key(tampered_key),
                runtime_options=_writer_options(),
            )

    def test_snapshot_rejects_missing_active_frame_before_resume(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        key = cursor.weighted_states[0][0]
        tampered_cursor = _unchecked_cursor_with_key(replace(key, active=None))
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        tampered_snapshot = replace(
            snapshot,
            cursor=tampered_cursor,
            frame_stack=(WriterFrontierFrame(tampered_cursor),),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(tampered_snapshot, prepared=prepared)

    def test_cursor_audit_rejects_root_frame_mismatch(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_frontier_cursor(prepared, options)
        key = cursor.weighted_states[0][0]
        tampered_key = replace(key, active=replace(key.active, atom=AtomId(1)))

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_snapshot_rejects_negative_component_index_without_index_error(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        key = cursor.weighted_states[0][0]
        tampered_key = replace(
            key,
            component_cursor=replace(key.component_cursor, component_index=-1),
        )
        tampered_cursor = _cursor_with_key(tampered_key)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        tampered_snapshot = replace(
            snapshot,
            cursor=tampered_cursor,
            frame_stack=(WriterFrontierFrame(tampered_cursor),),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(tampered_snapshot, prepared=prepared)

    def test_snapshot_rejects_out_of_range_component_index_without_index_error(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        key = cursor.weighted_states[0][0]
        tampered_key = replace(
            key,
            component_cursor=replace(
                key.component_cursor,
                component_index=len(prepared.facts.components),
            ),
        )
        tampered_cursor = _cursor_with_key(tampered_key)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        tampered_snapshot = replace(
            snapshot,
            cursor=tampered_cursor,
            frame_stack=(WriterFrontierFrame(tampered_cursor),),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(tampered_snapshot, prepared=prepared)

    def test_cursor_audit_rejects_invalid_pending_graph_triple(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=1),
        )
        after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
        key = after_root.weighted_states[0][0]
        tampered_key = replace(
            key,
            obligations=ObligationStateKey(
                pending_entry=PendingWriterEntry(
                    parent=AtomId(1),
                    child=AtomId(2),
                    bond=BondId(0),
                    branch=False,
                )
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=_writer_options(rooted_at_atom=1),
            )

    def test_cursor_audit_rejects_post_bond_pending_without_bond_record(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=1)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
        pending_cursor = writer_frontier_choices(prepared, after_root).choices[0].successor
        key = pending_cursor.weighted_states[0][0]
        assert key.obligations.pending_entry is not None
        tampered_key = replace(
            key,
            obligations=ObligationStateKey(
                pending_entry=replace(
                    key.obligations.pending_entry,
                    phase=PendingEntryPhase.NEEDS_ATOM_AFTER_BOND,
                )
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_pre_bond_pending_with_bond_record(self) -> None:
        prepared = _prepare(directional_facts())
        options = _writer_options(rooted_at_atom=2)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        post_bond = writer_frontier_choices(prepared, after_f).choices[0].successor
        key = post_bond.weighted_states[0][0]
        assert key.obligations.pending_entry is not None
        tampered_key = replace(
            key,
            obligations=ObligationStateKey(
                pending_entry=replace(
                    key.obligations.pending_entry,
                    phase=PendingEntryPhase.NEEDS_BOND_OR_ATOM,
                )
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_accepts_post_bond_pending_with_matching_bond_record(self) -> None:
        prepared = _prepare(directional_facts())
        options = _writer_options(rooted_at_atom=2)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        post_bond = writer_frontier_choices(prepared, after_f).choices[0].successor

        validate_writer_cursor_against_prepared(
            prepared,
            post_bond,
            runtime_options=options,
        )

    def test_cursor_audit_rejects_branch_pending_for_unique_child(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
        key = after_root.weighted_states[0][0]
        tampered_key = replace(
            key,
            obligations=ObligationStateKey(
                pending_entry=PendingWriterEntry(
                    parent=AtomId(0),
                    child=AtomId(1),
                    bond=BondId(0),
                    branch=True,
                )
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_inline_pending_with_unresolved_sibling(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=1)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
        key = after_root.weighted_states[0][0]
        tampered_key = replace(
            key,
            obligations=ObligationStateKey(
                pending_entry=PendingWriterEntry(
                    parent=AtomId(1),
                    child=AtomId(0),
                    bond=BondId(0),
                    branch=False,
                )
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_accepts_branch_post_bond_pending_with_sibling(self) -> None:
        prepared = _prepare(directional_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _directional_double_branch_post_bond_key(prepared, options)

        validate_writer_cursor_against_prepared(
            prepared,
            _cursor_with_key(key),
            runtime_options=options,
        )

    def test_cursor_audit_rejects_incoherent_open_closure_state(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        key = cursor.weighted_states[0][0]
        label = _closure_label()
        tampered_key = replace(
            key,
            ring_state=WriterRingStateKey(
                open_endpoints=(
                    WriterOpenClosureEndpoint(
                        bond=BondId(0),
                        first_atom=AtomId(1),
                        second_atom=AtomId(0),
                        label=label,
                        first_endpoint_text="1",
                        first_endpoint_bond_text="",
                    ),
                ),
                label_state=WriterRingLabelState(allocated=(label,)),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=_writer_options(),
            )

    def test_cursor_audit_accepts_coherent_open_closure_state(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_root_with_open_closure_key()

        validate_writer_cursor_against_prepared(
            prepared,
            _cursor_with_key(key),
            runtime_options=options,
        )

    def test_cursor_audit_rejects_open_closure_label_text_mismatch(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_root_with_open_closure_key()
        endpoint = replace(key.ring_state.open_endpoints[0], first_endpoint_text="9")
        tampered_key = replace(
            key,
            ring_state=WriterRingStateKey(
                open_endpoints=(endpoint,),
                label_state=key.ring_state.label_state,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_open_closure_label_value_text_mismatch(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_root_with_open_closure_key()
        label = WriterClosureLabel(value=1, text="7")
        endpoint = replace(
            key.ring_state.open_endpoints[0],
            label=label,
            first_endpoint_text="7",
        )
        tampered_key = replace(
            key,
            ring_state=WriterRingStateKey(
                open_endpoints=(endpoint,),
                label_state=WriterRingLabelState(allocated=(label,)),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_open_closure_label_outside_policy(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_root_with_open_closure_key()
        label = WriterClosureLabel(value=10, text="%10")
        endpoint = replace(
            key.ring_state.open_endpoints[0],
            label=label,
            first_endpoint_text="%10",
        )
        tampered_key = replace(
            key,
            ring_state=WriterRingStateKey(
                open_endpoints=(endpoint,),
                label_state=WriterRingLabelState(allocated=(label,)),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_accepts_policy_domain_open_label_without_allocator_history(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_root_with_open_closure_key()
        label = WriterClosureLabel(value=2, text="2")
        endpoint = replace(
            key.ring_state.open_endpoints[0],
            label=label,
            first_endpoint_text="2",
        )
        tampered_key = replace(
            key,
            ring_state=WriterRingStateKey(
                open_endpoints=(endpoint,),
                label_state=WriterRingLabelState(allocated=(label,)),
            ),
        )

        validate_writer_cursor_against_prepared(
            prepared,
            _cursor_with_key(tampered_key),
            runtime_options=options,
        )

    def test_cursor_audit_rejects_open_closure_bond_text_outside_policy(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_root_with_open_closure_key()
        endpoint = replace(key.ring_state.open_endpoints[0], first_endpoint_bond_text="~")
        tampered_key = replace(
            key,
            ring_state=WriterRingStateKey(
                open_endpoints=(endpoint,),
                label_state=key.ring_state.label_state,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_directional_open_closure_bond_text(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_root_with_open_closure_key()
        endpoint = replace(key.ring_state.open_endpoints[0], first_endpoint_bond_text="/")
        tampered_key = replace(
            key,
            ring_state=WriterRingStateKey(
                open_endpoints=(endpoint,),
                label_state=key.ring_state.label_state,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_open_closure_bond_text_without_compatible_partner(
        self,
    ) -> None:
        prepared = _prepare_with_non_single_closure_ring_endpoint_choices(
            BondOrder.DOUBLE,
            (
                BondTextChoice("absent", "", False),
                BondTextChoice("order", "=", False),
                BondTextChoice("partnerless", "~", False),
            ),
        )
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_root_with_open_closure_key()
        endpoint = replace(
            key.ring_state.open_endpoints[0],
            first_endpoint_bond_text="~",
        )
        tampered_key = replace(
            key,
            ring_state=WriterRingStateKey(
                open_endpoints=(endpoint,),
                label_state=key.ring_state.label_state,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_accepts_open_closure_partner_at_active_atom(self) -> None:
        prepared = _prepare(triangle_tail_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_tail_open_to_active_key()

        validate_writer_cursor_against_prepared(
            prepared,
            _cursor_with_key(key),
            runtime_options=options,
        )

    def test_cursor_audit_rejects_open_closure_partner_at_frozen_atom(self) -> None:
        prepared = _prepare(triangle_tail_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_tail_open_to_active_key()
        endpoint = key.ring_state.open_endpoints[0]
        tampered_key = replace(
            key,
            ring_state=WriterRingStateKey(
                open_endpoints=(
                    replace(
                        endpoint,
                        first_atom=endpoint.second_atom,
                        second_atom=endpoint.first_atom,
                    ),
                ),
                label_state=key.ring_state.label_state,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_open_closure_unreachable_unvisited_partner(self) -> None:
        prepared = _prepare(two_atom_facts())
        options = _writer_options(rooted_at_atom=0)
        label = _closure_label()
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(0),
            first_atom=AtomId(0),
            second_atom=AtomId(1),
            label=label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        key = replace(
            _manual_emitted_root_key(AtomId(0)),
            ring_state=WriterRingStateKey(
                open_endpoints=(endpoint,),
                label_state=WriterRingLabelState(allocated=(label,)),
            ),
            stereo_state=replace(
                empty_writer_stereo_state(),
                atom_occurrences=(
                    WriterAtomOccurrenceRecord(AtomId(0), TetraToken.NONE),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(key),
                runtime_options=options,
            )

    def test_cursor_audit_accepts_coherent_closed_closure_state(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_closed_closure_key()

        validate_writer_cursor_against_prepared(
            prepared,
            _cursor_with_key(key),
            runtime_options=options,
        )

    def test_cursor_audit_rejects_closed_closure_label_text_mismatch(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_closed_closure_key()
        closure = replace(key.ring_state.closed_closures[0], second_endpoint_text="9")
        tampered_key = replace(
            key,
            ring_state=WriterRingStateKey(
                closed_closures=(closure,),
                label_state=key.ring_state.label_state,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_closed_closure_label_value_text_mismatch(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_closed_closure_key()
        label = WriterClosureLabel(value=1, text="7")
        closure = replace(
            key.ring_state.closed_closures[0],
            label=label,
            first_endpoint_text="7",
            second_endpoint_text="7",
        )
        tampered_key = replace(
            key,
            ring_state=WriterRingStateKey(
                closed_closures=(closure,),
                label_state=WriterRingLabelState(reusable=(label,)),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_closed_closure_label_outside_policy(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_closed_closure_key()
        label = WriterClosureLabel(value=10, text="%10")
        closure = replace(
            key.ring_state.closed_closures[0],
            label=label,
            first_endpoint_text="%10",
            second_endpoint_text="%10",
        )
        tampered_key = replace(
            key,
            ring_state=WriterRingStateKey(
                closed_closures=(closure,),
                label_state=WriterRingLabelState(reusable=(label,)),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_closed_closure_bond_text_outside_policy(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_closed_closure_key()
        closure = replace(key.ring_state.closed_closures[0], second_endpoint_bond_text="~")
        tampered_key = replace(
            key,
            ring_state=WriterRingStateKey(
                closed_closures=(closure,),
                label_state=key.ring_state.label_state,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_directional_closed_closure_bond_text(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_closed_closure_key()
        closure = replace(key.ring_state.closed_closures[0], first_endpoint_bond_text="\\")
        tampered_key = replace(
            key,
            ring_state=WriterRingStateKey(
                closed_closures=(closure,),
                label_state=key.ring_state.label_state,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_invalid_non_single_closed_closure_bond_text_pairs(
        self,
    ) -> None:
        rows = (
            (BondOrder.DOUBLE, "", ""),
            (BondOrder.DOUBLE, "=", "="),
            (BondOrder.TRIPLE, "", ""),
            (BondOrder.TRIPLE, "#", "#"),
        )

        for order, first_text, second_text in rows:
            with self.subTest(order=order, first=first_text, second=second_text):
                prepared = _prepare_with_joint_non_single_ring_closures(
                    non_single_closure_triangle_facts(order),
                )
                options = _writer_options(rooted_at_atom=0)
                key = _triangle_closed_closure_key()
                closure = replace(
                    key.ring_state.closed_closures[0],
                    first_endpoint_bond_text=first_text,
                    second_endpoint_bond_text=second_text,
                )
                tampered_key = replace(
                    key,
                    ring_state=WriterRingStateKey(
                        closed_closures=(closure,),
                        label_state=key.ring_state.label_state,
                    ),
                )

                with self.assertRaises(SouthStarError):
                    validate_writer_cursor_against_prepared(
                        prepared,
                        _cursor_with_key(tampered_key),
                        runtime_options=options,
                    )

    def test_cursor_audit_rejects_orphan_allocated_ring_label(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        label = _closure_label()
        key = replace(
            _triangle_root_with_open_closure_key(),
            ring_state=WriterRingStateKey(
                open_endpoints=(),
                label_state=WriterRingLabelState(allocated=(label,)),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_orphan_reusable_ring_label(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        label = _closure_label()
        key = replace(
            _triangle_root_with_open_closure_key(),
            ring_state=WriterRingStateKey(
                open_endpoints=(),
                label_state=WriterRingLabelState(reusable=(label,)),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_duplicate_open_closure_labels(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        label = _closure_label()
        key = replace(
            _triangle_root_with_open_closure_key(),
            ring_state=WriterRingStateKey(
                open_endpoints=(
                    WriterOpenClosureEndpoint(
                        bond=BondId(0),
                        first_atom=AtomId(0),
                        second_atom=AtomId(1),
                        label=label,
                        first_endpoint_text="1",
                        first_endpoint_bond_text="",
                    ),
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

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_incomplete_completed_component(self) -> None:
        prepared = _prepare(chain_plus_singleton_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_c = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_second_c = writer_frontier_choices(prepared, after_c).choices[0].successor
        after_dot = writer_frontier_choices(prepared, after_second_c).choices[0].successor
        key = after_dot.weighted_states[0][0]
        tampered_key = replace(key, visited_atoms=frozenset((AtomId(0),)))

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_unreachable_current_component_atom(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=1)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
        key = after_root.weighted_states[0][0]
        tampered_key = replace(
            key,
            visited_atoms=frozenset((*key.visited_atoms, AtomId(2))),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_orphan_residual_attachment(self) -> None:
        prepared = _prepare(chain_plus_isolate_same_component_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _manual_emitted_root_key(AtomId(0))

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_terminal_looking_latent_residual_bond(self) -> None:
        prepared = _prepare(chain_plus_orphan_chain_same_component_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _terminal_looking_orphan_chain_key()

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_closure_candidate_edge_obligation(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_closure_candidate_key()

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(key),
                runtime_options=options,
            )

    def test_cursor_audit_accepts_closure_open_ready_cyclic_residual(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _manual_emitted_root_key(AtomId(0))

        validate_writer_cursor_against_prepared(
            prepared,
            _cursor_with_key(key),
            runtime_options=options,
        )

    def test_cursor_audit_accepts_single_boundary_cyclic_tree_entry(self) -> None:
        prepared = _prepare(triangle_tail_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _manual_emitted_root_key(AtomId(0))

        validate_writer_cursor_against_prepared(
            prepared,
            _cursor_with_key(key),
            runtime_options=options,
        )

    def test_cursor_audit_rejects_unowned_multi_boundary_residual_attachment(self) -> None:
        prepared = _prepare(triangle_with_frozen_tail_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_with_frozen_tail_key()

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_mixed_owned_unowned_boundary_attachment(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_two_visited_key()

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_frozen_single_boundary_attachment(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _cco_frozen_single_boundary_key()

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(key),
                runtime_options=options,
            )

    def test_snapshot_rejects_completed_cyclic_component_outside_current_component(self) -> None:
        prepared = _prepare(triangle_plus_singleton_facts())
        options = _writer_options(rooted_at_atom=3)
        key = _manual_emitted_root_key(
            AtomId(3),
            component_index=1,
            component_roots=(AtomId(0), AtomId(3)),
            visited_atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
            written_bonds=(BondId(0), BondId(1), BondId(2)),
        )
        cursor = _cursor_with_key(key)
        snapshot = _snapshot_for_cursor(prepared, options, cursor)

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(snapshot, prepared=prepared)

    def test_cursor_audit_rejects_future_cyclic_component_outside_current_component(self) -> None:
        prepared = _prepare(singleton_plus_triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _manual_emitted_root_key(
            AtomId(0),
            component_roots=(AtomId(0), AtomId(1)),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(key),
                runtime_options=options,
            )

    def test_cursor_audit_accepts_all_acyclic_multi_component_surface(self) -> None:
        prepared = _prepare(chain_plus_singleton_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_c = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_second_c = writer_frontier_choices(prepared, after_c).choices[0].successor
        after_dot = writer_frontier_choices(prepared, after_second_c).choices[0].successor

        validate_writer_cursor_against_prepared(
            prepared,
            after_dot,
            runtime_options=options,
        )

    def test_cursor_audit_accepts_acyclic_residual_attachment(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _cco_after_second_atom_key(prepared, options)

        validate_writer_cursor_against_prepared(
            prepared,
            _cursor_with_key(key),
            runtime_options=options,
        )

    def test_cursor_audit_rejects_stranded_unvisited_child_obligation(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _cco_after_second_atom_key(prepared, options)
        tampered_key = replace(
            key,
            active=replace(
                key.active,
                atom=AtomId(0),
                parent=None,
                incoming_bond=None,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_accepts_branch_stack_owned_sibling_obligation(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=1)
        branch_key = _cco_branch_child_key(prepared, options)

        validate_writer_cursor_against_prepared(
            prepared,
            _cursor_with_key(branch_key),
            runtime_options=options,
        )

    def test_cursor_audit_rejects_branch_stack_without_sibling_obligation(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=1)
        branch_key = _cco_branch_child_key(prepared, options)
        tampered_key = replace(
            branch_key,
            visited_atoms=frozenset((*branch_key.visited_atoms, AtomId(2))),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_branch_state_missing_return_owner(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=1)
        branch_key = _cco_branch_child_key(prepared, options)
        tampered_key = replace(branch_key, branch_stack=())

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_terminal_state_with_stale_branch_stack(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _cco_after_third_atom_key(prepared, options)
        root_frame = replace(
            key.active,
            atom=AtomId(0),
            parent=None,
            incoming_bond=None,
        )
        tampered_key = replace(
            key,
            branch_stack=(WriterBranchFrame(return_atom=root_frame),),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_linear_prefix_with_stale_branch_stack(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _cco_after_second_atom_key(prepared, options)
        root_frame = replace(
            key.active,
            atom=AtomId(0),
            parent=None,
            incoming_bond=None,
        )
        tampered_key = replace(
            key,
            branch_stack=(WriterBranchFrame(return_atom=root_frame),),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_branch_return_not_active_ancestor(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _cco_after_third_atom_key(prepared, options)
        tampered_key = replace(
            key,
            branch_stack=(WriterBranchFrame(return_atom=key.active),),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_active_frame_tree_orientation_mismatch(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _cco_after_third_atom_key(prepared, options)
        root_frame = replace(
            key.active,
            atom=AtomId(0),
            parent=None,
            incoming_bond=None,
        )
        tampered_key = replace(
            key,
            active=replace(
                key.active,
                atom=AtomId(1),
                parent=AtomId(2),
                incoming_bond=BondId(1),
            ),
            branch_stack=(WriterBranchFrame(return_atom=root_frame),),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_completed_component_with_nonterminal_active(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _cco_after_third_atom_key(prepared, options)
        tampered_key = replace(
            key,
            active=replace(
                key.active,
                atom=AtomId(0),
                parent=None,
                incoming_bond=None,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_future_unemitted_bond_occurrence(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
        key = after_root.weighted_states[0][0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                bond_occurrences=key.stereo_state.bond_occurrences
                + (
                    WriterBondOccurrenceRecord(
                        bond=BondId(1),
                        parent=AtomId(1),
                        child=AtomId(2),
                        mark=DirectionMark.ABSENT,
                    ),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_future_unvisited_atom_occurrence(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
        key = after_root.weighted_states[0][0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                atom_occurrences=key.stereo_state.atom_occurrences
                + (
                    WriterAtomOccurrenceRecord(
                        atom=AtomId(2),
                        token=TetraToken.NONE,
                    ),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_future_unvisited_local_order(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
        key = after_root.weighted_states[0][0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=key.stereo_state.local_orders
                + (
                    WriterLocalOrderRecord(
                        atom=AtomId(2),
                        order=(),
                    ),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_missing_visited_atom_occurrence(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _cco_after_second_atom_key(prepared, options)
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                atom_occurrences=key.stereo_state.atom_occurrences[:-1],
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_missing_written_bond_occurrence(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _cco_after_second_atom_key(prepared, options)
        tampered_key = replace(
            key,
            stereo_state=replace(key.stereo_state, bond_occurrences=()),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_reversed_written_bond_occurrence(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _cco_after_second_atom_key(prepared, options)
        record = key.stereo_state.bond_occurrences[0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                bond_occurrences=(
                    replace(record, parent=record.child, child=record.parent),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_invalid_local_order_occurrence(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
        key = after_center.weighted_states[0][0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=(
                    WriterLocalOrderRecord(
                        atom=AtomId(1),
                        order=(OccurrenceId(999),),
                    ),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_wrong_site_local_order_occurrence(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
        key = after_center.weighted_states[0][0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=(
                    WriterLocalOrderRecord(
                        atom=AtomId(1),
                        order=(OccurrenceId(0),),
                    ),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_premature_tetra_local_order_closure(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_center_key(prepared, options)
        record = _local_order_for_atom(key, AtomId(0))
        forged_order = record.order + (
            OccurrenceId(2),
            OccurrenceId(1),
            OccurrenceId(3),
        )
        tampered_key = _key_with_reconstructed_residual(
            prepared,
            replace(
                key,
                stereo_state=replace(
                    key.stereo_state,
                    local_orders=_replace_local_order_record(
                        key.stereo_state.local_orders,
                        replace(record, order=forged_order, closed=True),
                    ),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_future_occurrence_in_open_local_order(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_center_key(prepared, options)
        record = _local_order_for_atom(key, AtomId(0))
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=_replace_local_order_record(
                    key.stereo_state.local_orders,
                    replace(record, order=record.order + (OccurrenceId(1),)),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_missing_written_local_order_occurrence(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_center_key(prepared, options)
        record = _local_order_for_atom(key, AtomId(0))
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=_replace_local_order_record(
                    key.stereo_state.local_orders,
                    replace(record, order=()),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_implicit_h_before_local_order_closure(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_center_key(prepared, options)
        record = _local_order_for_atom(key, AtomId(0))
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=_replace_local_order_record(
                    key.stereo_state.local_orders,
                    replace(record, order=record.order + (OccurrenceId(3),)),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_closed_branch_return_local_order(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_branch_child_key(prepared, options)
        record = _local_order_for_atom(key, AtomId(0))
        forged_order = record.order + (
            OccurrenceId(1),
            OccurrenceId(3),
        )
        tampered_key = _key_with_reconstructed_residual(
            prepared,
            replace(
                key,
                stereo_state=replace(
                    key.stereo_state,
                    local_orders=_replace_local_order_record(
                        key.stereo_state.local_orders,
                        replace(record, order=forged_order, closed=True),
                    ),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_completed_branch_child_left_open(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_after_branch_return_key(prepared, options)
        record = _local_order_for_atom(key, AtomId(3))
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=_replace_local_order_record(
                    key.stereo_state.local_orders,
                    replace(record, closed=False),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_accepts_reachable_tetra_traversal_states(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        pending = [initial_writer_transition_frontier_cursor(prepared, options)]
        seen = set()
        visited = 0

        while pending:
            cursor = pending.pop(0)
            if cursor in seen:
                continue
            seen.add(cursor)
            visited += 1

            validate_writer_cursor_against_prepared(
                prepared,
                cursor,
                runtime_options=options,
            )

            choices = writer_frontier_choices(prepared, cursor)
            if choices.terminal is not None:
                validate_writer_cursor_against_prepared(
                    prepared,
                    choices.terminal.finalized_cursor,
                    runtime_options=options,
                )
            pending.extend(choice.successor for choice in choices.choices)

        self.assertGreaterEqual(visited, 7)

    def test_cursor_audit_rejects_missing_tetra_ring_endpoint_occurrence(self) -> None:
        from tests.south_star1.test_writer_stereo_residual import ring_core_tetra_facts

        prepared = _prepare(ring_core_tetra_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _first_key_with_ring_core_tetra_open_endpoint(prepared, options)
        record = _local_order_for_atom(key, AtomId(0))
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=_replace_local_order_record(
                    key.stereo_state.local_orders,
                    replace(record, order=()),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_duplicated_tetra_ring_endpoint_occurrence(self) -> None:
        from tests.south_star1.test_writer_stereo_residual import ring_core_tetra_facts

        prepared = _prepare(ring_core_tetra_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _first_key_with_ring_core_tetra_open_endpoint(prepared, options)
        record = _local_order_for_atom(key, AtomId(0))
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=_replace_local_order_record(
                    key.stereo_state.local_orders,
                    replace(record, order=record.order + record.order),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_local_order_reconstruction_rejects_multiple_tetra_ring_incidences(
        self,
    ) -> None:
        from tests.south_star1.test_writer_stereo_residual import ring_core_tetra_facts

        prepared = _prepare(ring_core_tetra_facts())
        rows = (
            ((BondId(2), AtomId(2)), (BondId(0), AtomId(1))),
            ((BondId(0), AtomId(1)), (BondId(2), AtomId(2))),
        )

        for incidences in rows:
            with self.subTest(incidences=incidences):
                with self.assertRaises(SouthStarError) as caught:
                    reconstruct_writer_local_order_records(
                        prepared,
                        atom_occurrences=(
                            WriterAtomOccurrenceRecord(
                                AtomId(0),
                                TetraToken.AT,
                            ),
                        ),
                        parent_by_child={},
                        closed_atoms=frozenset(),
                        ring_incidences_by_atom={AtomId(0): incidences},
                    )

                self.assertIs(
                    caught.exception.kind,
                    SouthStarErrorKind.UNSUPPORTED_STEREO,
                )

    def test_cursor_audit_accepts_reachable_ring_core_tetra_traversal_states(self) -> None:
        from tests.south_star1.test_writer_stereo_residual import ring_core_tetra_facts

        prepared = _prepare(ring_core_tetra_facts())
        options = _writer_options(rooted_at_atom=0)
        pending = [initial_writer_transition_frontier_cursor(prepared, options)]
        seen = set()
        visited = 0

        while pending:
            cursor = pending.pop(0)
            if cursor in seen:
                continue
            seen.add(cursor)
            visited += 1

            validate_writer_cursor_against_prepared(
                prepared,
                cursor,
                runtime_options=options,
            )

            choices = writer_frontier_choices(prepared, cursor)
            if choices.terminal is not None:
                validate_writer_cursor_against_prepared(
                    prepared,
                    choices.terminal.finalized_cursor,
                    runtime_options=options,
                )
            pending.extend(choice.successor for choice in choices.choices)

        self.assertGreater(visited, 1)

    def test_cursor_audit_rejects_missing_non_stereo_history_mid_traversal(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _cco_after_branch_return_key(prepared, options)
        self.assertTrue(key.stereo_state.local_orders)
        self.assertTrue(
            any(record.closed for record in key.stereo_state.local_orders)
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=(),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_missing_non_stereo_history_at_terminal(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _first_terminal_key(prepared, options)
        self.assertTrue(key.stereo_state.local_orders)
        self.assertTrue(
            any(record.closed for record in key.stereo_state.local_orders)
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=(),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_accepts_legitimately_empty_non_stereo_history(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=1)
        initial = initial_writer_frontier_cursor(prepared, options)
        root_emitted = writer_frontier_choices(
            prepared,
            initial,
        ).choices[0].successor

        for cursor in (initial, root_emitted):
            with self.subTest(cursor=cursor):
                for key, _weight in cursor.weighted_states:
                    self.assertFalse(key.stereo_state.local_orders)
                validate_writer_cursor_against_prepared(
                    prepared,
                    cursor,
                    runtime_options=options,
                )

    def test_cursor_audit_accepts_reachable_non_stereo_traversal_states(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=1)
        pending = [initial_writer_frontier_cursor(prepared, options)]
        seen = set()
        visited = 0

        while pending:
            cursor = pending.pop(0)
            if cursor in seen:
                continue
            seen.add(cursor)
            visited += 1

            validate_writer_cursor_against_prepared(
                prepared,
                cursor,
                runtime_options=options,
            )

            choices = writer_frontier_choices(prepared, cursor)
            if choices.terminal is not None:
                validate_writer_cursor_against_prepared(
                    prepared,
                    choices.terminal.finalized_cursor,
                    runtime_options=options,
                )
            pending.extend(choice.successor for choice in choices.choices)

        self.assertGreaterEqual(visited, 5)

    def test_cursor_audit_rejects_child_atom_occurrence_before_tree_parent(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _first_terminal_key(prepared, options)
        tampered_key = _key_with_rebuilt_stereo_history(
            prepared,
            key,
            atom_occurrences=(
                key.stereo_state.atom_occurrences[1],
                key.stereo_state.atom_occurrences[0],
                key.stereo_state.atom_occurrences[2],
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_non_depth_first_subtree_interleaving(self) -> None:
        prepared = _prepare(depth_first_interleaving_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _manual_depth_first_interleaving_key()
        tampered_key = _key_with_rebuilt_stereo_history(
            prepared,
            key,
            atom_occurrences=key.stereo_state.atom_occurrences,
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_swapped_child_order_with_original_active_child(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _first_terminal_key(prepared, options)
        tampered_key = _key_with_rebuilt_stereo_history(
            prepared,
            key,
            atom_occurrences=(
                key.stereo_state.atom_occurrences[0],
                key.stereo_state.atom_occurrences[2],
                key.stereo_state.atom_occurrences[1],
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_reordered_disconnected_component_roots(self) -> None:
        prepared = _prepare(two_singletons_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _manual_two_singletons_key()
        tampered_key = _key_with_rebuilt_stereo_history(
            prepared,
            key,
            atom_occurrences=(
                key.stereo_state.atom_occurrences[1],
                key.stereo_state.atom_occurrences[0],
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_tetra_sibling_swap_recomputed_from_forgery(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _first_terminal_key(prepared, options)
        center = key.stereo_state.atom_occurrences[1]
        flipped_center = replace(
            center,
            token=(
                TetraToken.AT
                if center.token is TetraToken.ATAT
                else TetraToken.ATAT
            ),
        )
        tampered_key = _key_with_rebuilt_stereo_history(
            prepared,
            key,
            atom_occurrences=(
                key.stereo_state.atom_occurrences[0],
                flipped_center,
                key.stereo_state.atom_occurrences[3],
                key.stereo_state.atom_occurrences[2],
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_duplicate_atom_occurrence(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_center_key(prepared, options)
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                atom_occurrences=key.stereo_state.atom_occurrences
                + (key.stereo_state.atom_occurrences[-1],),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_duplicate_bond_occurrence(self) -> None:
        prepared = _prepare(directional_facts())
        options = _writer_options(rooted_at_atom=2)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_slash = writer_frontier_choices(prepared, after_f).choices[0].successor
        key = after_slash.weighted_states[0][0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                bond_occurrences=key.stereo_state.bond_occurrences
                + (key.stereo_state.bond_occurrences[-1],),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_duplicate_local_order(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_center_key(prepared, options)
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=key.stereo_state.local_orders
                + (key.stereo_state.local_orders[-1],),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_terminal_snapshot_retains_active_final_atom(self) -> None:
        prepared, options, key = _terminal_tetra_key()
        cursor = _cursor_with_key(key)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        retained = snapshot.cursor.weighted_states[0][0]

        self.assertIsNotNone(retained.active)
        self.assertEqual(retained.active.atom, key.active.atom)

    def test_residual_factor_addition_rejects_zero_support_snapshot_source(self) -> None:
        left = direction_var(("left", 0))
        right = direction_var(("right", 0))
        store = ResidualStore()
        store.add_var(left, (DirectionMark.FWD,))
        store.add_var(right, (DirectionMark.ABSENT,))
        factor = DirectionalResidualFactor(
            scope=(left, right),
            status=SiteStatus.SPECIFIED,
            target=DirectionalValue.OPPOSITE,
            carrier_models={
                left: DirectionalCarrierResidual(left, "left", 1, 1),
                right: DirectionalCarrierResidual(right, "right", 1, 1),
            },
        )
        before = store.value_snapshot()

        self.assertIs(
            add_factor_and_propagate(store, factor).kind,
            ResidualPropagationKind.CONTRADICTION,
        )
        self.assertEqual(store.value_snapshot(), before)


    def test_cursor_audit_wraps_invalid_residual_snapshot_round_trip(self) -> None:
        var = tetra_var(("test", 0))
        residual_snapshot = ResidualStore().value_snapshot()
        residual_snapshot = replace(
            residual_snapshot,
            domains=((var, (TetraToken.AT, TetraToken.ATAT)),),
            assignments=((var, TetraToken.AT),),
            factors=(),
        )
        prepared = _prepare(cco_facts())
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        key = cursor.weighted_states[0][0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=residual_snapshot,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_missing_initial_tetra_factor(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = initial_writer_frontier_cursor(prepared, options).weighted_states[0][0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=empty_writer_stereo_state().residual_snapshot,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_missing_directional_site_factor(self) -> None:
        prepared = _prepare(directional_facts())
        options = _writer_options(rooted_at_atom=2)
        key = initial_writer_frontier_cursor(prepared, options).weighted_states[0][0]
        snapshot = key.stereo_state.residual_snapshot
        tampered_snapshot = replace(
            snapshot,
            factors=tuple(
                factor
                for factor in snapshot.factors
                if factor.key.kind != "directional_site"
            ),
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=tampered_snapshot,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_missing_directional_bond_factor(self) -> None:
        prepared = _prepare(directional_facts())
        options = _writer_options(rooted_at_atom=2)
        key = initial_writer_frontier_cursor(prepared, options).weighted_states[0][0]
        snapshot = key.stereo_state.residual_snapshot
        tampered_snapshot = replace(
            snapshot,
            factors=tuple(
                factor
                for factor in snapshot.factors
                if factor.key.kind != "directional_bond_emission"
            ),
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=tampered_snapshot,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_altered_live_factor_definition(self) -> None:
        prepared = _prepare(directional_facts())
        options = _writer_options(rooted_at_atom=2)
        key = initial_writer_frontier_cursor(prepared, options).weighted_states[0][0]
        snapshot = key.stereo_state.residual_snapshot
        tampered_factors = tuple(
            replace(factor, allowed_marks=(DirectionMark.ABSENT,))
            if factor.key.kind == "directional_bond_emission"
            else factor
            for factor in snapshot.factors
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=replace(snapshot, factors=tampered_factors),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_extra_tetra_factor_after_closure(self) -> None:
        prepared, options, key = _terminal_tetra_key()
        domains, factors = _writer_stereo_relation_definitions(prepared)
        factor = next(
            item
            for item in factors
            if item.key.kind == "tetra_site"
        )
        factor_vars = frozenset(factor.scope)
        tampered_snapshot = ResidualStoreValueSnapshot(
            domains=tuple(
                (var, domain)
                for var, domain in domains
                if var in factor_vars
            ),
            assignments=(),
            factors=(factor.value_snapshot(),),
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=tampered_snapshot,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_initial_tetra_domain_narrowed_without_atom_event(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = initial_writer_frontier_cursor(prepared, options).weighted_states[0][0]
        token_var = next(
            var
            for var, _domain in key.stereo_state.residual_snapshot.domains
            if var.kind == "tetra_token"
        )
        tampered_snapshot = replace(
            key.stereo_state.residual_snapshot,
            domains=tuple(
                (var, (TetraToken.AT,))
                if var == token_var
                else (var, domain)
                for var, domain in key.stereo_state.residual_snapshot.domains
            ),
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=tampered_snapshot,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_missing_tetra_assignment_after_atom_event(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_center_key(prepared, options)
        tampered_snapshot = replace(
            key.stereo_state.residual_snapshot,
            assignments=(),
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=tampered_snapshot,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_tetra_assignment_disagreeing_with_atom_event(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_center_key(prepared, options)
        var, value = key.stereo_state.residual_snapshot.assignments[0]
        other = TetraToken.ATAT if value is TetraToken.AT else TetraToken.AT
        tampered_snapshot = replace(
            key.stereo_state.residual_snapshot,
            domains=tuple(
                (domain_var, (other,))
                if domain_var == var
                else (domain_var, domain)
                for domain_var, domain in key.stereo_state.residual_snapshot.domains
            ),
            assignments=((var, other),),
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=tampered_snapshot,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_missing_directional_assignment_after_bond_event(self) -> None:
        prepared = _prepare(directional_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _directional_live_after_bond_key(prepared, options)
        tampered_snapshot = replace(
            key.stereo_state.residual_snapshot,
            assignments=(),
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=tampered_snapshot,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_directional_assignment_disagreeing_with_bond_event(self) -> None:
        prepared = _prepare(directional_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _directional_live_after_bond_key(prepared, options)
        var, value = key.stereo_state.residual_snapshot.assignments[0]
        other = (
            DirectionalNormalizedSign.NEGATIVE
            if value is DirectionalNormalizedSign.POSITIVE
            else DirectionalNormalizedSign.POSITIVE
        )
        tampered_snapshot = replace(
            key.stereo_state.residual_snapshot,
            domains=tuple(
                (domain_var, (other,))
                if domain_var == var
                else (domain_var, domain)
                for domain_var, domain in key.stereo_state.residual_snapshot.domains
            ),
            assignments=tuple(
                (assigned_var, other)
                if assigned_var == var
                else (assigned_var, assigned_value)
                for assigned_var, assigned_value in key.stereo_state.residual_snapshot.assignments
            ),
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=tampered_snapshot,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_tampered_tetra_token_after_factor_discharge(self) -> None:
        prepared, options, key = _terminal_tetra_key()
        record = key.stereo_state.atom_occurrences[-1]
        other = TetraToken.ATAT if record.token is TetraToken.AT else TetraToken.AT
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                atom_occurrences=key.stereo_state.atom_occurrences[:-1]
                + (replace(record, token=other),),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_tampered_closed_local_order_parity(self) -> None:
        prepared, options, key = _terminal_tetra_key()
        record = key.stereo_state.local_orders[-1]
        tampered_order = (
            record.order[1],
            record.order[0],
            *record.order[2:],
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=key.stereo_state.local_orders[:-1]
                + (replace(record, order=tampered_order),),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_tampered_directional_mark_after_factor_discharge(self) -> None:
        prepared = _prepare(directional_facts())
        options = _writer_options(rooted_at_atom=2)
        key = _first_terminal_key(prepared, options)
        record = key.stereo_state.bond_occurrences[-1]
        other = DirectionMark.REV if record.mark is DirectionMark.FWD else DirectionMark.FWD
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                bond_occurrences=key.stereo_state.bond_occurrences[:-1]
                + (replace(record, mark=other),),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_residual_assignment_without_occurrence(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_center_key(prepared, options)
        extra_var = tetra_var(("writer", 999))
        tampered_snapshot = replace(
            key.stereo_state.residual_snapshot,
            domains=key.stereo_state.residual_snapshot.domains
            + ((extra_var, (TetraToken.AT, TetraToken.ATAT)),),
            assignments=key.stereo_state.residual_snapshot.assignments
            + ((extra_var, TetraToken.AT),),
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=tampered_snapshot,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_frontier_frame_cursor_must_match_snapshot_cursor(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        choices = writer_frontier_choices(prepared, cursor)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        tampered = replace(
            snapshot,
            frame_stack=(WriterFrontierFrame(choices.choices[0].successor),),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(tampered, prepared=prepared)


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _prepare_with_joint_non_single_ring_closures(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
        policy=ordinary_policy_for_facts(
            facts,
            options=OrdinaryPolicyOptions(non_single_ring_closures="joint"),
        ),
    )


def _prepare_with_non_single_closure_ring_endpoint_choices(
    order: BondOrder,
    choices: tuple[BondTextChoice, ...],
):
    facts = non_single_closure_triangle_facts(order)
    policy = ordinary_policy_for_facts(
        facts,
        options=OrdinaryPolicyOptions(non_single_ring_closures="joint"),
    )
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
        policy=replace(
            policy,
            bond_text_domains=tuple(
                (
                    BondTextDomain(
                        bond=domain.bond,
                        slot_kind=domain.slot_kind,
                        choices=choices,
                    )
                    if (
                        domain.bond == BondId(2)
                        and domain.slot_kind == "ring_endpoint"
                    )
                    else domain
                )
                for domain in policy.bond_text_domains
            ),
        ),
    )


def _writer_options(*, rooted_at_atom: int = -1) -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=rooted_at_atom,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def _cursor_with_key(key) -> WriterFrontierCursor:
    return WriterFrontierCursor(weighted_states=((key, 1),))


def _only_choice(prepared, cursor, emitted_text: str):
    matches = tuple(
        choice
        for choice in writer_frontier_choices(prepared, cursor).choices
        if choice.emitted_text == emitted_text
    )
    assert len(matches) == 1
    return matches[0]


def _assert_snapshot_round_trips_cursor(
    testcase,
    prepared,
    options: SouthStarRuntimeOptions,
    cursor: WriterFrontierCursor,
) -> None:
    snapshot = capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=cursor,
    )
    testcase.assertEqual(
        resume_writer_frontier_choices_from_snapshot(snapshot, prepared=prepared),
        writer_frontier_choices(prepared, cursor),
    )
    testcase.assertEqual(
        count_writer_frontier_support(prepared, snapshot.cursor.support_state),
        count_writer_frontier_support(prepared, cursor.support_state),
    )
    testcase.assertEqual(
        count_writer_cursor_completions(prepared, snapshot.cursor),
        count_writer_cursor_completions(prepared, cursor),
    )


def _unchecked_cursor_with_key(key) -> WriterFrontierCursor:
    cursor = object.__new__(WriterFrontierCursor)
    object.__setattr__(cursor, "weighted_states", ((key, 1),))
    return cursor


def _snapshot_for_cursor(
    prepared,
    options: SouthStarRuntimeOptions,
    cursor: WriterFrontierCursor,
) -> WriterSearchSnapshot:
    return WriterSearchSnapshot(
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
        prepared_identity=_prepared_identity(prepared, options),
        runtime_options=options,
        cursor=cursor,
        decoder_boundary=WriterDecoderBoundary(),
        frame_stack=(WriterFrontierFrame(cursor),),
    )


def _manual_emitted_root_key(
    root: AtomId,
    *,
    component_index: int = 0,
    component_roots: tuple[AtomId, ...] | None = None,
    visited_atoms: tuple[AtomId, ...] | None = None,
    written_bonds: tuple[BondId, ...] = (),
):
    if component_roots is None:
        component_roots = (root,)
    if visited_atoms is None:
        visited_atoms = (root,)
    return writer_state_key(
        WriterState(
            component_cursor=ComponentCursor(
                component_index=component_index,
                component_roots=component_roots,
            ),
            active=WriterAtomFrame(
                atom=root,
                parent=None,
                incoming_bond=None,
                atom_emitted=True,
            ),
            branch_stack=(),
            visited_atoms=frozenset(visited_atoms),
            written_bonds=frozenset(written_bonds),
            obligations=ObligationState(),
            ring_state=WriterRingState(),
            stereo_state=replace(
                empty_writer_stereo_state(),
                atom_occurrences=tuple(
                    WriterAtomOccurrenceRecord(
                        atom=atom_id,
                        token=TetraToken.NONE,
                    )
                    for atom_id in visited_atoms
                ),
            ),
            policy_state=WriterPolicyState(),
        )
    )


def _triangle_closure_candidate_key():
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
            stereo_state=replace(
                empty_writer_stereo_state(),
                atom_occurrences=(
                    WriterAtomOccurrenceRecord(AtomId(0), TetraToken.NONE),
                    WriterAtomOccurrenceRecord(AtomId(1), TetraToken.NONE),
                ),
                bond_occurrences=(
                    WriterBondOccurrenceRecord(
                        BondId(0),
                        AtomId(0),
                        AtomId(1),
                        DirectionMark.ABSENT,
                    ),
                ),
            ),
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
            stereo_state=replace(
                empty_writer_stereo_state(),
                atom_occurrences=(
                    WriterAtomOccurrenceRecord(AtomId(0), TetraToken.NONE),
                    WriterAtomOccurrenceRecord(AtomId(1), TetraToken.NONE),
                ),
                bond_occurrences=(
                    WriterBondOccurrenceRecord(
                        BondId(0),
                        AtomId(0),
                        AtomId(1),
                        DirectionMark.ABSENT,
                    ),
                ),
            ),
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
            stereo_state=replace(
                empty_writer_stereo_state(),
                atom_occurrences=(
                    WriterAtomOccurrenceRecord(AtomId(0), TetraToken.NONE),
                    WriterAtomOccurrenceRecord(AtomId(1), TetraToken.NONE),
                    WriterAtomOccurrenceRecord(AtomId(3), TetraToken.NONE),
                ),
                bond_occurrences=(
                    WriterBondOccurrenceRecord(
                        BondId(0),
                        AtomId(0),
                        AtomId(1),
                        DirectionMark.ABSENT,
                    ),
                    WriterBondOccurrenceRecord(
                        BondId(3),
                        AtomId(1),
                        AtomId(3),
                        DirectionMark.ABSENT,
                    ),
                ),
            ),
            policy_state=WriterPolicyState(),
        )
    )


def _closure_label() -> WriterClosureLabel:
    return WriterClosureLabel(value=1, text="1")


def _triangle_root_with_open_closure_key():
    label = _closure_label()
    endpoint = WriterOpenClosureEndpoint(
        bond=BondId(2),
        first_atom=AtomId(0),
        second_atom=AtomId(2),
        label=label,
        first_endpoint_text="1",
        first_endpoint_bond_text="",
    )
    return replace(
        _manual_emitted_root_key(AtomId(0)),
        ring_state=WriterRingStateKey(
            open_endpoints=(endpoint,),
            label_state=WriterRingLabelState(allocated=(label,)),
        ),
        stereo_state=replace(
            empty_writer_stereo_state(),
            atom_occurrences=(
                WriterAtomOccurrenceRecord(AtomId(0), TetraToken.NONE),
            ),
        ),
    )


def _triangle_closed_closure_key():
    prepared = _prepare(triangle_facts())
    label = _closure_label()
    closure = WriterClosedClosure(
        bond=BondId(2),
        first_atom=AtomId(0),
        second_atom=AtomId(2),
        label=label,
        first_endpoint_text="1",
        second_endpoint_text="1",
        first_endpoint_bond_text="",
        second_endpoint_bond_text="",
    )
    key = replace(
        _triangle_closure_candidate_key(),
        ring_state=WriterRingStateKey(
            closed_closures=(closure,),
            label_state=WriterRingLabelState(reusable=(label,)),
        ),
        stereo_state=replace(
            empty_writer_stereo_state(),
            atom_occurrences=(
                WriterAtomOccurrenceRecord(AtomId(0), TetraToken.NONE),
                WriterAtomOccurrenceRecord(AtomId(1), TetraToken.NONE),
                WriterAtomOccurrenceRecord(AtomId(2), TetraToken.NONE),
            ),
            bond_occurrences=(
                WriterBondOccurrenceRecord(
                    BondId(0),
                    AtomId(0),
                    AtomId(1),
                    DirectionMark.ABSENT,
                ),
                WriterBondOccurrenceRecord(
                    BondId(1),
                    AtomId(1),
                    AtomId(2),
                    DirectionMark.ABSENT,
                ),
            ),
        ),
    )
    return _key_with_reconstructed_local_orders(prepared, key)


def _triangle_terminal_open_closure_key():
    label = _closure_label()
    endpoint = WriterOpenClosureEndpoint(
        bond=BondId(2),
        first_atom=AtomId(0),
        second_atom=AtomId(2),
        label=label,
        first_endpoint_text="1",
        first_endpoint_bond_text="",
    )
    return replace(
        _triangle_closed_closure_key(),
        ring_state=WriterRingStateKey(
            open_endpoints=(endpoint,),
            label_state=WriterRingLabelState(allocated=(label,)),
        ),
    )


def _triangle_tail_open_to_active_key():
    prepared = _prepare(triangle_tail_facts())
    label = _closure_label()
    endpoint = WriterOpenClosureEndpoint(
        bond=BondId(2),
        first_atom=AtomId(0),
        second_atom=AtomId(2),
        label=label,
        first_endpoint_text="1",
        first_endpoint_bond_text="",
    )
    key = writer_state_key(
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
            ring_state=WriterRingState(
                open_endpoints=(endpoint,),
                label_state=WriterRingLabelState(allocated=(label,)),
            ),
            stereo_state=replace(
                empty_writer_stereo_state(),
                atom_occurrences=(
                    WriterAtomOccurrenceRecord(AtomId(0), TetraToken.NONE),
                    WriterAtomOccurrenceRecord(AtomId(1), TetraToken.NONE),
                    WriterAtomOccurrenceRecord(AtomId(2), TetraToken.NONE),
                ),
                bond_occurrences=(
                    WriterBondOccurrenceRecord(
                        BondId(0),
                        AtomId(0),
                        AtomId(1),
                        DirectionMark.ABSENT,
                    ),
                    WriterBondOccurrenceRecord(
                        BondId(1),
                        AtomId(1),
                        AtomId(2),
                        DirectionMark.ABSENT,
                    ),
                ),
            ),
            policy_state=WriterPolicyState(),
        )
    )
    return _key_with_reconstructed_local_orders(prepared, key)


def _tetra_center_key(prepared, options):
    cursor = initial_writer_frontier_cursor(prepared, options)
    after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
    after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
    return after_center.weighted_states[0][0]


def _tetra_branch_child_key(prepared, options):
    cursor = initial_writer_frontier_cursor(prepared, options)
    after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
    after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
    after_branch_open = writer_frontier_choices(
        prepared,
        after_center,
    ).choices[0].successor
    after_branch_child = writer_frontier_choices(
        prepared,
        after_branch_open,
    ).choices[0].successor
    return after_branch_child.weighted_states[0][0]


def _tetra_after_branch_return_key(prepared, options):
    cursor = initial_writer_frontier_cursor(prepared, options)
    after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
    after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
    after_branch_open = writer_frontier_choices(
        prepared,
        after_center,
    ).choices[0].successor
    after_branch_child = writer_frontier_choices(
        prepared,
        after_branch_open,
    ).choices[0].successor
    after_branch_close = writer_frontier_choices(
        prepared,
        after_branch_child,
    ).choices[0].successor
    return after_branch_close.weighted_states[0][0]


def _local_order_for_atom(key, atom: AtomId) -> WriterLocalOrderRecord:
    for record in key.stereo_state.local_orders:
        if record.atom == atom:
            return record
    raise AssertionError(f"missing local-order record for atom {atom!r}")


def _replace_local_order_record(
    records: tuple[WriterLocalOrderRecord, ...],
    replacement: WriterLocalOrderRecord,
) -> tuple[WriterLocalOrderRecord, ...]:
    return tuple(
        replacement if record.atom == replacement.atom else record
        for record in records
    )


def _key_with_reconstructed_residual(prepared, key):
    stereo_state = key.stereo_state
    return replace(
        key,
        stereo_state=replace(
            stereo_state,
            residual_snapshot=reconstruct_writer_stereo_residual_snapshot(
                prepared,
                stereo_state,
            ),
        ),
    )


def _key_with_rebuilt_stereo_history(
    prepared,
    key,
    *,
    atom_occurrences: tuple[WriterAtomOccurrenceRecord, ...],
):
    rebuilt = replace(
        key,
        stereo_state=replace(
            key.stereo_state,
            atom_occurrences=atom_occurrences,
        ),
    )
    rebuilt = _key_with_reconstructed_local_orders(prepared, rebuilt)
    return _key_with_reconstructed_residual(prepared, rebuilt)


def _key_with_reconstructed_local_orders(prepared, key):
    parent_by_child = {
        child: parent
        for child, (parent, _bond) in _snapshot_parent_links(prepared, key).items()
    }
    open_frame_atoms = {
        frame.return_atom.atom
        for frame in key.branch_stack
    }
    if key.active.atom_emitted:
        open_frame_atoms.add(key.active.atom)
    closed_atoms = frozenset(set(key.visited_atoms) - open_frame_atoms)
    return replace(
        key,
        stereo_state=replace(
            key.stereo_state,
            local_orders=reconstruct_writer_local_order_records(
                prepared,
                atom_occurrences=key.stereo_state.atom_occurrences,
                parent_by_child=parent_by_child,
                closed_atoms=closed_atoms,
            ),
        ),
    )


def _snapshot_parent_links(prepared, key):
    parent_by_child = {}
    for index in range(key.component_cursor.component_index + 1):
        component = prepared.facts.components[index]
        component_bonds = frozenset(component.bonds)
        written = frozenset(
            bond for bond in key.written_bonds if bond in component_bonds
        )
        root = key.component_cursor.component_roots[index]
        adjacency = {}
        for bond in written:
            fact = prepared.graph_index.bond_by_id[bond]
            adjacency.setdefault(fact.a, []).append((fact.b, bond))
            adjacency.setdefault(fact.b, []).append((fact.a, bond))
        seen = {root}
        stack = [root]
        while stack:
            parent = stack.pop()
            for child, bond in adjacency.get(parent, ()):
                if child in seen:
                    continue
                seen.add(child)
                parent_by_child[child] = (parent, bond)
                stack.append(child)
    return parent_by_child


def _cco_after_second_atom_key(prepared, options):
    cursor = initial_writer_frontier_cursor(prepared, options)
    after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
    after_second = writer_frontier_choices(prepared, after_root).choices[0].successor
    return after_second.weighted_states[0][0]


def _cco_after_third_atom_key(prepared, options):
    cursor = initial_writer_frontier_cursor(prepared, options)
    after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
    after_second = writer_frontier_choices(prepared, after_root).choices[0].successor
    after_third = writer_frontier_choices(prepared, after_second).choices[0].successor
    return after_third.weighted_states[0][0]


def _cco_branch_child_key(prepared, options):
    cursor = initial_writer_frontier_cursor(prepared, options)
    after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
    after_branch_open = writer_frontier_choices(prepared, after_root).choices[0].successor
    after_branch_child = writer_frontier_choices(
        prepared,
        after_branch_open,
    ).choices[0].successor
    return after_branch_child.weighted_states[0][0]


def _cco_after_branch_return_key(prepared, options):
    cursor = initial_writer_frontier_cursor(prepared, options)
    after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
    after_branch_open = writer_frontier_choices(prepared, after_root).choices[0].successor
    after_branch_child = writer_frontier_choices(
        prepared,
        after_branch_open,
    ).choices[0].successor
    after_branch_close = writer_frontier_choices(
        prepared,
        after_branch_child,
    ).choices[0].successor
    return after_branch_close.weighted_states[0][0]


def _directional_double_branch_post_bond_key(prepared, options):
    cursor = initial_writer_frontier_cursor(prepared, options)
    after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
    branch_open = writer_frontier_choices(prepared, after_root).choices[0].successor
    double_branch_key = next(
        key
        for key, _ in branch_open.weighted_states
        if key.obligations.pending_entry is not None
        and key.obligations.pending_entry.bond == BondId(0)
    )
    post_bond = writer_frontier_choices(
        prepared,
        _cursor_with_key(double_branch_key),
    ).choices[0].successor
    return post_bond.weighted_states[0][0]


def _directional_live_after_bond_key(prepared, options):
    pending = [initial_writer_frontier_cursor(prepared, options)]
    seen = set()

    while pending:
        cursor = pending.pop(0)
        if cursor in seen:
            continue
        seen.add(cursor)

        for key, _ in cursor.weighted_states:
            if (
                key.stereo_state.bond_occurrences
                and key.stereo_state.residual_snapshot.assignments
            ):
                return key

        choices = writer_frontier_choices(prepared, cursor)
        pending.extend(choice.successor for choice in choices.choices)

    raise AssertionError("no live directional state after bond emission")


def _first_terminal_key(prepared, options):
    cursor = initial_writer_frontier_cursor(prepared, options)
    seen = set()
    while True:
        if cursor in seen:
            raise AssertionError("frontier cursor cycle while seeking terminal")
        seen.add(cursor)
        choices = writer_frontier_choices(prepared, cursor)
        if choices.terminal is not None:
            return choices.terminal.finalized_cursor.weighted_states[0][0]
        if not choices.choices:
            raise AssertionError("frontier cursor has no terminal path")
        cursor = choices.choices[0].successor


def _first_key_with_ring_core_tetra_open_endpoint(prepared, options):
    pending = [initial_writer_transition_frontier_cursor(prepared, options)]
    seen = set()
    while pending:
        cursor = pending.pop(0)
        if cursor in seen:
            continue
        seen.add(cursor)
        for key, _weight in cursor.weighted_states:
            if key.ring_state.open_endpoints and key.stereo_state.local_orders:
                return key
        choices = writer_frontier_choices(prepared, cursor)
        pending.extend(choice.successor for choice in choices.choices)
    raise AssertionError("no ring-core tetra open endpoint state")


def _manual_depth_first_interleaving_key():
    key = writer_state_key(
        WriterState(
            component_cursor=ComponentCursor(
                component_index=0,
                component_roots=(AtomId(0),),
            ),
            active=WriterAtomFrame(
                atom=AtomId(3),
                parent=AtomId(1),
                incoming_bond=BondId(2),
                atom_emitted=True,
            ),
            branch_stack=(),
            visited_atoms=frozenset((AtomId(0), AtomId(1), AtomId(2), AtomId(3))),
            written_bonds=frozenset((BondId(0), BondId(1), BondId(2))),
            obligations=ObligationState(),
            ring_state=WriterRingState(),
            stereo_state=replace(
                empty_writer_stereo_state(),
                atom_occurrences=(
                    WriterAtomOccurrenceRecord(AtomId(0), TetraToken.NONE),
                    WriterAtomOccurrenceRecord(AtomId(1), TetraToken.NONE),
                    WriterAtomOccurrenceRecord(AtomId(2), TetraToken.NONE),
                    WriterAtomOccurrenceRecord(AtomId(3), TetraToken.NONE),
                ),
                bond_occurrences=(
                    WriterBondOccurrenceRecord(
                        BondId(0),
                        AtomId(0),
                        AtomId(1),
                        DirectionMark.ABSENT,
                    ),
                    WriterBondOccurrenceRecord(
                        BondId(1),
                        AtomId(0),
                        AtomId(2),
                        DirectionMark.ABSENT,
                    ),
                    WriterBondOccurrenceRecord(
                        BondId(2),
                        AtomId(1),
                        AtomId(3),
                        DirectionMark.ABSENT,
                    ),
                ),
            ),
            policy_state=WriterPolicyState(),
        )
    )
    return key


def _manual_two_singletons_key():
    key = writer_state_key(
        WriterState(
            component_cursor=ComponentCursor(
                component_index=1,
                component_roots=(AtomId(0), AtomId(1)),
            ),
            active=WriterAtomFrame(
                atom=AtomId(1),
                parent=None,
                incoming_bond=None,
                atom_emitted=True,
            ),
            branch_stack=(),
            visited_atoms=frozenset((AtomId(0), AtomId(1))),
            written_bonds=frozenset(),
            obligations=ObligationState(),
            ring_state=WriterRingState(),
            stereo_state=replace(
                empty_writer_stereo_state(),
                atom_occurrences=(
                    WriterAtomOccurrenceRecord(AtomId(0), TetraToken.NONE),
                    WriterAtomOccurrenceRecord(AtomId(1), TetraToken.NONE),
                ),
            ),
            policy_state=WriterPolicyState(),
        )
    )
    return key


def _assert_residual_reconstructs(self, prepared, key) -> None:
    self.assertEqual(
        key.stereo_state.residual_snapshot,
        reconstruct_writer_stereo_residual_snapshot(
            prepared,
            key.stereo_state,
        ),
    )


def _terminal_tetra_key():
    from tests.south_star1.test_writer_stereo_residual import terminal_tetra_center_facts
    from tests.south_star1.test_writer_stereo_residual import terminal_tetra_center_policy

    prepared = prepare_south_star_mol_from_facts(
        terminal_tetra_center_facts(),
        writer_surface=SouthStarWriterSurface(),
        policy=terminal_tetra_center_policy(),
    )
    options = _writer_options(rooted_at_atom=0)
    cursor = initial_writer_frontier_cursor(prepared, options)
    after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
    after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
    terminal = writer_frontier_choices(prepared, after_center).terminal
    assert terminal is not None
    return prepared, options, terminal.finalized_cursor.weighted_states[0][0]


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


def chain_plus_isolate_same_component_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "O")),
        bonds=(single_bond(0, 0, 1),),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2)),
                bonds=(BondId(0),),
            ),
        ),
    )


def chain_plus_orphan_chain_same_component_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "O"), atom(3, "F")),
        bonds=(single_bond(0, 0, 1), single_bond(1, 2, 3)),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1)),
            ),
        ),
    )


def _terminal_looking_orphan_chain_key():
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
            stereo_state=replace(
                empty_writer_stereo_state(),
                atom_occurrences=(
                    WriterAtomOccurrenceRecord(AtomId(0), TetraToken.NONE),
                    WriterAtomOccurrenceRecord(AtomId(1), TetraToken.NONE),
                ),
                bond_occurrences=(
                    WriterBondOccurrenceRecord(
                        BondId(0),
                        AtomId(0),
                        AtomId(1),
                        DirectionMark.ABSENT,
                    ),
                ),
            ),
            policy_state=WriterPolicyState(),
        )
    )


def triangle_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C")),
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


def non_single_closure_triangle_facts(order: BondOrder) -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C")),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
            BondFacts(
                id=BondId(2),
                a=AtomId(2),
                b=AtomId(0),
                order=order,
                is_aromatic=False,
                is_conjugated=False,
            ),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2)),
                bonds=(BondId(0), BondId(1), BondId(2)),
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


def two_atom_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C")),
        bonds=(single_bond(0, 0, 1),),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1)),
                bonds=(BondId(0),),
            ),
        ),
    )


def triangle_tail_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C"), atom(3, "O")),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
            single_bond(2, 2, 0),
            single_bond(3, 2, 3),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2), BondId(3)),
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


def depth_first_interleaving_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "O"), atom(3, "Br")),
        bonds=(
            single_bond(0, 0, 1),
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
    )


def two_singletons_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "O")),
        bonds=(),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0),),
                bonds=(),
            ),
            ComponentFacts(
                id=ComponentId(1),
                atoms=(AtomId(1),),
                bonds=(),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
