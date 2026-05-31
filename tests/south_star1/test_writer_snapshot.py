"""Writer frontier snapshot tests."""

from __future__ import annotations

from dataclasses import replace
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.residual_constraints import ResidualStore
from grimace._south_star1.residual_constraints import tetra_var
from grimace._south_star1.writer_frontier import count_writer_cursor_completions
from grimace._south_star1.writer_frontier import count_writer_frontier_support
from grimace._south_star1.writer_frontier import WriterFrontierCursor
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
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
from grimace._south_star1.writer_stereo import WriterAtomOccurrenceRecord
from grimace._south_star1.writer_stereo import WriterBondOccurrenceRecord
from grimace._south_star1.writer_stereo import WriterDelayedStereoFactor
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
            stereo_state=replace(
                key.stereo_state,
                delayed_factors=(_pending_ring_pair_factor(endpoint),),
            ),
        )

        with self.assertRaises(SouthStarError):
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
            stereo_state=replace(
                key.stereo_state,
                delayed_factors=(_pending_ring_pair_factor(endpoint),),
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
            stereo_state=replace(
                key.stereo_state,
                delayed_factors=(
                    _pending_ring_pair_factor(
                        replace(
                            endpoint,
                            first_atom=endpoint.second_atom,
                            second_atom=endpoint.first_atom,
                        )
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
                    WriterAtomOccurrenceRecord(AtomId(0), TetraToken.NONE, None),
                ),
                delayed_factors=(_pending_ring_pair_factor(endpoint),),
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
            stereo_state=replace(
                key.stereo_state,
                delayed_factors=(_closed_ring_pair_factor(closure),),
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
            stereo_state=replace(
                key.stereo_state,
                delayed_factors=(_closed_ring_pair_factor(closure),),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_terminal_open_closure_state(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_terminal_open_closure_key()

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_open_endpoint_without_ring_pair_factor(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = replace(
            _triangle_root_with_open_closure_key(),
            stereo_state=replace(
                _triangle_root_with_open_closure_key().stereo_state,
                delayed_factors=(),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_wrong_pending_ring_pair_evidence(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_root_with_open_closure_key()
        factor = key.stereo_state.delayed_factors[0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                delayed_factors=(replace(factor, evidence=(("ring_endpoint", 2, "open", 0, 2, 9, "9", "1", ""),)),),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_closed_closure_without_ring_pair_factor(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = replace(
            _triangle_closed_closure_key(),
            stereo_state=replace(
                _triangle_closed_closure_key().stereo_state,
                delayed_factors=(),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_wrong_closed_ring_pair_evidence(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _triangle_closed_closure_key()
        factor = key.stereo_state.delayed_factors[0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                delayed_factors=(replace(factor, evidence=(("ring_pair", 2, 0, 2, 9, "9", "1", "1", "", ""),)),),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_ring_pair_factor_without_closure_state(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_frontier_cursor(prepared, options)
        key = cursor.weighted_states[0][0]
        label = _closure_label()
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(0),
            first_atom=AtomId(0),
            second_atom=AtomId(1),
            label=label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                delayed_factors=(_pending_ring_pair_factor(endpoint),),
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

    def test_cursor_audit_rejects_cyclic_residual_attachment_without_closure_candidate(self) -> None:
        prepared = _prepare(triangle_facts())
        options = _writer_options(rooted_at_atom=0)
        key = _manual_emitted_root_key(AtomId(0))

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
                        var=None,
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
                        var=None,
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

    def test_cursor_audit_rejects_atom_occurrence_assignment_mismatch(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
        key = after_center.weighted_states[0][0]
        records = tuple(
            replace(record, token=TetraToken.NONE)
            if record.var is not None
            else record
            for record in key.stereo_state.atom_occurrences
        )
        tampered_key = replace(
            key,
            stereo_state=replace(key.stereo_state, atom_occurrences=records),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_bond_occurrence_assignment_mismatch(self) -> None:
        prepared = _prepare(directional_facts())
        options = _writer_options(rooted_at_atom=2)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_slash = writer_frontier_choices(prepared, after_f).choices[0].successor
        key = after_slash.weighted_states[0][0]
        records = tuple(
            replace(record, mark=DirectionMark.ABSENT)
            if record.var is not None
            else record
            for record in key.stereo_state.bond_occurrences
        )
        tampered_key = replace(
            key,
            stereo_state=replace(key.stereo_state, bond_occurrences=records),
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

    def test_cursor_audit_rejects_duplicate_delayed_factor(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_center_key(prepared, options)
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                delayed_factors=key.stereo_state.delayed_factors
                + (key.stereo_state.delayed_factors[-1],),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_snapshot_rejects_pending_and_closed_delayed_factor_for_same_site(self) -> None:
        prepared, options, key = _terminal_tetra_key()
        closed_factor = key.stereo_state.delayed_factors[0]
        pending_factor = WriterDelayedStereoFactor(
            kind=closed_factor.kind,
            site=closed_factor.site,
            scope=closed_factor.scope,
            evidence=closed_factor.evidence,
            closed=False,
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                delayed_factors=key.stereo_state.delayed_factors + (pending_factor,),
            ),
        )
        valid_cursor = _cursor_with_key(key)
        tampered_cursor = _cursor_with_key(tampered_key)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=valid_cursor,
        )
        tampered_snapshot = replace(
            snapshot,
            cursor=tampered_cursor,
            frame_stack=(WriterFrontierFrame(tampered_cursor),),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(tampered_snapshot, prepared=prepared)

    def test_snapshot_rejects_pending_tetra_factor_with_closed_local_order(self) -> None:
        prepared, options, key = _terminal_tetra_key()
        closed_factor = key.stereo_state.delayed_factors[0]
        pending_factor = WriterDelayedStereoFactor(
            kind=closed_factor.kind,
            site=closed_factor.site,
            scope=closed_factor.scope,
            evidence=closed_factor.evidence,
            closed=False,
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                delayed_factors=(pending_factor,),
                residual_snapshot=replace(
                    key.stereo_state.residual_snapshot,
                    factors=(),
                ),
            ),
        )
        valid_cursor = _cursor_with_key(key)
        tampered_cursor = _cursor_with_key(tampered_key)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=valid_cursor,
        )
        tampered_snapshot = replace(
            snapshot,
            cursor=tampered_cursor,
            frame_stack=(WriterFrontierFrame(tampered_cursor),),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(tampered_snapshot, prepared=prepared)

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

    def test_cursor_audit_rejects_closed_delay_without_residual_factor(self) -> None:
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
        key = terminal.finalized_cursor.weighted_states[0][0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=ResidualStore().value_snapshot(),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_duplicate_residual_factor_snapshot(self) -> None:
        prepared, options, key = _terminal_tetra_key()
        factor = key.stereo_state.residual_snapshot.factors[0]
        tampered_snapshot = replace(
            key.stereo_state.residual_snapshot,
            factors=key.stereo_state.residual_snapshot.factors + (factor,),
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

    def test_cursor_audit_rejects_closed_factor_semantic_mismatch(self) -> None:
        prepared, options, key = _terminal_tetra_key()
        factor = key.stereo_state.residual_snapshot.factors[0]
        tampered_snapshot = replace(
            key.stereo_state.residual_snapshot,
            factors=(replace(factor, local_order=tuple(reversed(factor.local_order))),),
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

    def test_cursor_audit_rejects_occurrence_without_delayed_factor(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_center_key(prepared, options)
        tampered_key = replace(
            key,
            stereo_state=replace(key.stereo_state, delayed_factors=()),
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


def _writer_options(*, rooted_at_atom: int = -1) -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=rooted_at_atom,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def _cursor_with_key(key) -> WriterFrontierCursor:
    return WriterFrontierCursor(weighted_states=((key, 1),))


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
                        var=None,
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
                WriterAtomOccurrenceRecord(AtomId(0), TetraToken.NONE, None),
            ),
            delayed_factors=(_pending_ring_pair_factor(endpoint),),
        ),
    )


def _triangle_closed_closure_key():
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
    return replace(
        _triangle_closure_candidate_key(),
        ring_state=WriterRingStateKey(
            closed_closures=(closure,),
            label_state=WriterRingLabelState(reusable=(label,)),
        ),
        stereo_state=replace(
            empty_writer_stereo_state(),
            atom_occurrences=(
                WriterAtomOccurrenceRecord(AtomId(0), TetraToken.NONE, None),
                WriterAtomOccurrenceRecord(AtomId(1), TetraToken.NONE, None),
                WriterAtomOccurrenceRecord(AtomId(2), TetraToken.NONE, None),
            ),
            bond_occurrences=(
                WriterBondOccurrenceRecord(
                    BondId(0),
                    AtomId(0),
                    AtomId(1),
                    DirectionMark.ABSENT,
                    None,
                ),
                WriterBondOccurrenceRecord(
                    BondId(1),
                    AtomId(1),
                    AtomId(2),
                    DirectionMark.ABSENT,
                    None,
                ),
            ),
            delayed_factors=(_closed_ring_pair_factor(closure),),
        ),
    )


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
        stereo_state=replace(
            _triangle_closed_closure_key().stereo_state,
            delayed_factors=(_pending_ring_pair_factor(endpoint),),
        ),
    )


def _triangle_tail_open_to_active_key():
    label = _closure_label()
    endpoint = WriterOpenClosureEndpoint(
        bond=BondId(2),
        first_atom=AtomId(0),
        second_atom=AtomId(2),
        label=label,
        first_endpoint_text="1",
        first_endpoint_bond_text="",
    )
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
            ring_state=WriterRingState(
                open_endpoints=(endpoint,),
                label_state=WriterRingLabelState(allocated=(label,)),
            ),
            stereo_state=replace(
                empty_writer_stereo_state(),
                atom_occurrences=(
                    WriterAtomOccurrenceRecord(AtomId(0), TetraToken.NONE, None),
                    WriterAtomOccurrenceRecord(AtomId(1), TetraToken.NONE, None),
                    WriterAtomOccurrenceRecord(AtomId(2), TetraToken.NONE, None),
                ),
                bond_occurrences=(
                    WriterBondOccurrenceRecord(
                        BondId(0),
                        AtomId(0),
                        AtomId(1),
                        DirectionMark.ABSENT,
                        None,
                    ),
                    WriterBondOccurrenceRecord(
                        BondId(1),
                        AtomId(1),
                        AtomId(2),
                        DirectionMark.ABSENT,
                        None,
                    ),
                ),
                delayed_factors=(_pending_ring_pair_factor(endpoint),),
            ),
            policy_state=WriterPolicyState(),
        )
    )


def _pending_ring_pair_factor(endpoint: WriterOpenClosureEndpoint) -> WriterDelayedStereoFactor:
    return WriterDelayedStereoFactor(
        kind="ring_pair",
        site=SiteId(int(endpoint.bond)),
        evidence=(
            (
                "ring_endpoint",
                int(endpoint.bond),
                "open",
                int(endpoint.first_atom),
                int(endpoint.second_atom),
                endpoint.label.value,
                endpoint.label.text,
                endpoint.first_endpoint_text,
                endpoint.first_endpoint_bond_text,
            ),
        ),
        closed=False,
    )


def _closed_ring_pair_factor(closure: WriterClosedClosure) -> WriterDelayedStereoFactor:
    return WriterDelayedStereoFactor(
        kind="ring_pair",
        site=SiteId(int(closure.bond)),
        evidence=(
            (
                "ring_pair",
                int(closure.bond),
                int(closure.first_atom),
                int(closure.second_atom),
                closure.label.value,
                closure.label.text,
                closure.first_endpoint_text,
                closure.second_endpoint_text,
                closure.first_endpoint_bond_text,
                closure.second_endpoint_bond_text,
            ),
        ),
        closed=True,
    )


def _tetra_center_key(prepared, options):
    cursor = initial_writer_frontier_cursor(prepared, options)
    after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
    after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
    return after_center.weighted_states[0][0]


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


if __name__ == "__main__":
    unittest.main()
