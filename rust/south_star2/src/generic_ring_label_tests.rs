use std::collections::BTreeSet;

use crate::native::NativeSolverState;
use crate::solver::Consistency;
use crate::{BondId, PreparedGraphBuilder, PreparedMolecule};

use super::{
    ring_label_number_text, ring_label_spellings, NonStereoWriterState, PreparedAtomToken,
    PreparedNonStereo, RingLabelSlot, WriterRegime,
};

fn state(regime: WriterRegime) -> NonStereoWriterState<NativeSolverState> {
    let mut graph = PreparedGraphBuilder::new();
    graph.add_atom().unwrap();
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        vec![PreparedAtomToken::Fixed("C".to_owned())],
        Vec::new(),
    )
    .unwrap();
    match NonStereoWriterState::initial_with_regime(&surface, regime).unwrap() {
        Consistency::Consistent(state) => state,
        Consistency::Contradiction => panic!("single carbon fixture must be consistent"),
    }
}

#[test]
fn generic_ring_domain_contains_110_forms_for_100_numeric_ids() {
    let state = state(WriterRegime::Generic {
        component_order: super::ComponentOrder::AnyOrder,
    });
    let options = state.available_ring_label_options();
    assert_eq!(options.len(), 110);
    let ids = options.iter().map(|(slot, _)| slot.index()).collect::<BTreeSet<_>>();
    assert_eq!(ids, (0..=99).collect());

    assert_eq!(ring_label_spellings(0), ["0".to_owned(), "%00".to_owned()]);
    assert_eq!(ring_label_spellings(1), ["1".to_owned(), "%01".to_owned()]);
    assert_eq!(ring_label_spellings(9), ["9".to_owned(), "%09".to_owned()]);
    assert_eq!(ring_label_spellings(10), ["%10".to_owned()]);
    assert_eq!(ring_label_spellings(99), ["%99".to_owned()]);
    assert_eq!(state.label_text_options(RingLabelSlot(1)), ["1".to_owned(), "%01".to_owned()]);
    assert_eq!(state.label_text_options(RingLabelSlot(0)), ["0".to_owned(), "%00".to_owned()]);
    assert_eq!(state.label_text_options(RingLabelSlot(99)), ["%99".to_owned()]);
}

#[test]
fn generic_ring_ids_exclude_open_labels_and_reuse_closed_ids() {
    let mut state = state(WriterRegime::Generic {
        component_order: super::ComponentOrder::AnyOrder,
    });
    let first = BondId::new(7);
    let last = BondId::new(8);
    state.labels.allocate_at(RingLabelSlot(0), first);
    state.labels.allocate_at(RingLabelSlot(99), last);

    let available = state.available_ring_label_options();
    assert!(!available.iter().any(|(slot, _)| slot.index() == 0));
    assert!(!available.iter().any(|(slot, _)| slot.index() == 99));
    assert_eq!(available.len(), 107);
    assert_eq!(ring_label_spellings(0), ["0".to_owned(), "%00".to_owned()]);
    assert_eq!(ring_label_spellings(99), ["%99".to_owned()]);

    state.labels.release(RingLabelSlot(0), first);
    let available = state.available_ring_label_options();
    assert_eq!(available.iter().filter(|(slot, _)| slot.index() == 0).count(), 2);
    assert_eq!(available.iter().filter(|(slot, _)| slot.index() == 99).count(), 0);
    state.labels.release(RingLabelSlot(99), last);
    let available = state.available_ring_label_options();
    assert_eq!(available.iter().filter(|(slot, _)| slot.index() == 99).count(), 1);
}

#[test]
fn rdkit_ring_ids_keep_the_lowest_free_positive_label_and_one_spelling() {
    let mut state = state(WriterRegime::RdkitRandomNoncanonical);
    assert_eq!(state.available_ring_label_options(), [(RingLabelSlot(1), "1".to_owned())]);
    assert_eq!(state.label_text_options(RingLabelSlot(1)), ["1".to_owned()]);
    state.labels.allocate_at(RingLabelSlot(1), BondId::new(7));
    assert_eq!(state.available_ring_label_options(), [(RingLabelSlot(2), "2".to_owned())]);
}

#[test]
fn generic_ring_number_rendering_is_zero_based() {
    assert_eq!(ring_label_number_text(0), "0");
    assert_eq!(ring_label_number_text(1), "1");
    assert_eq!(ring_label_number_text(9), "9");
    assert_eq!(ring_label_number_text(10), "%10");
    assert_eq!(ring_label_number_text(99), "%99");
}
