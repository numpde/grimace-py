//! Test-only derivation of the first attached-forest completion boundary.
//!
//! This reads actual walker progress and existing solver domains. It does not
//! choose roles, roots, visit ranks, or an event schedule.

use std::collections::{BTreeSet, VecDeque};

use super::*;
use crate::native::NativeSolverState;
use crate::model::FactorActivation;
use crate::prepared::PreparedGraphBuilder;
use crate::traversal::ObservedBondProgress;
use crate::traversal::{LocalBondOrder, LocalLayoutContext};
use crate::solver::Consistency;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DerivedEntry {
    atom: AtomId,
    from: AtomId,
    bond: BondId,
}

fn derived_forest_entries(state: &NonStereoWriterState<NativeSolverState>) -> Option<Vec<DerivedEntry>> {
    if !state.labels.is_clean() { return None; }
    let graph = state.surface.molecule().graph();
    let observed = state.structural.observe_raw().traversal;
    let visited = observed.visited_atoms.iter().copied().collect::<BTreeSet<_>>();
    let frames = observed.active_frame.iter().chain(observed.branch_returns.iter())
        .map(|frame| frame.atom).collect::<BTreeSet<_>>();
    let mut residual = vec![Vec::<(BondId, AtomId)>::new(); graph.atom_count()];
    let mut boundary = vec![Vec::<(BondId, AtomId)>::new(); graph.atom_count()];
    for bond in graph.bond_ids() {
        if observed.bond_progress[bond.index()] != ObservedBondProgress::Unrepresented { continue; }
        let partition = state.surface.molecule().bond_role_partition(bond).unwrap();
        if !state.structural.bond_decision_domain(bond).is_subset_of(partition.traversal_values()) {
            return None;
        }
        let edge = graph.bond(bond).unwrap();
        let (a, b) = (edge.a(), edge.b());
        match (visited.contains(&a), visited.contains(&b)) {
            (false, false) => {
                residual[a.index()].push((bond, b));
                residual[b.index()].push((bond, a));
            }
            (true, false) if frames.contains(&a) => boundary[b.index()].push((bond, a)),
            (false, true) if frames.contains(&b) => boundary[a.index()].push((bond, b)),
            _ => return None,
        }
    }
    let mut seen = BTreeSet::new();
    let mut entries = Vec::new();
    for atom in graph.atom_ids().filter(|atom| !visited.contains(atom)) {
        if !seen.insert(atom) { continue; }
        let mut component = vec![atom];
        let mut queue = VecDeque::from([atom]);
        while let Some(current) = queue.pop_front() {
            for &(_, next) in &residual[current.index()] {
                if seen.insert(next) {
                    component.push(next);
                    queue.push_back(next);
                }
            }
        }
        let edge_count = component.iter().map(|atom| residual[atom.index()].len()).sum::<usize>() / 2;
        if edge_count + 1 != component.len() { return None; }
        let attachments = component.iter().flat_map(|atom|
            boundary[atom.index()].iter().map(move |&(bond, from)| (*atom, bond, from)))
            .collect::<Vec<_>>();
        let &[(root, bond, from)] = attachments.as_slice() else { return None; };
        let mut oriented = BTreeSet::from([root]);
        let mut queue = VecDeque::from([(root, from, bond)]);
        while let Some((current, source, incoming)) = queue.pop_front() {
            entries.push(DerivedEntry { atom: current, from: source, bond: incoming });
            for &(next_bond, next) in &residual[current.index()] {
                if oriented.insert(next) { queue.push_back((next, current, next_bond)); }
            }
        }
        assert_eq!(oriented.len(), component.len());
    }
    Some(entries)
}

fn shared_directional_forest_fixture() -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut builder = PreparedGraphBuilder::new();
    let atoms = (0..6).map(|_| builder.add_atom().unwrap()).collect::<Vec<_>>();
    let bonds = (0..5).map(|index| builder.add_bond(atoms[index], atoms[index + 1]).unwrap())
        .collect::<Vec<_>>();
    let relation = |double_bond, left, left_carrier, right, right_carrier, parity| {
        PreparedDirectionalRelation {
            double_bond,
            left_endpoint: atoms[left],
            left_carriers: vec![PreparedDirectionalCarrier::unflipped(left_carrier)].into_boxed_slice(),
            right_endpoint: atoms[right],
            right_carriers: vec![PreparedDirectionalCarrier::unflipped(right_carrier)].into_boxed_slice(),
            side_phase_xor: parity,
        }
    };
    let surface = PreparedNonStereo::with_atom_tokens_and_directional_for_writer(
        PreparedMolecule::new(builder.build()),
        ["A", "B", "C", "D", "E", "F"].into_iter()
            .map(|label| PreparedAtomToken::Fixed(label.to_owned())).collect(),
        [NonStereoBondToken::Elided, NonStereoBondToken::Double,
            NonStereoBondToken::Elided, NonStereoBondToken::Double,
            NonStereoBondToken::Elided].to_vec(),
        vec![relation(bonds[1], 1, bonds[0], 2, bonds[2], false),
            relation(bonds[3], 3, bonds[2], 4, bonds[4], true)],
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true,
    ).unwrap();
    (surface, atoms, bonds)
}

fn force_fixture_traversal_roles(
    state: &mut NonStereoWriterState<NativeSolverState>, bonds: &[BondId],
) {
    let restrictions = bonds.iter().map(|&bond| {
        let molecule = state.surface.molecule();
        (molecule.bond_decision_variable(bond).unwrap(), molecule.bond_role_partition(bond).unwrap().traversal_values())
    }).collect::<Vec<_>>();
    state.structural = state.structural.transitioned_semantics(&restrictions, &[])
        .unwrap().unwrap_consistent();
}

#[derive(Clone, Debug)]
struct LocalRealizationRow {
    entry: DerivedEntry,
    bond_text: &'static str,
    atom_text: String,
    restrictions: Vec<(VariableId, Domain)>,
    activate: Vec<FactorId>,
}

fn one_finite_atom_spelling(start: AtomSpellingState) -> Option<String> {
    let mut queue = VecDeque::from([(start.clone(), String::new())]);
    let mut seen = vec![start];
    while let Some((state, text)) = queue.pop_front() {
        if state.is_complete() { return Some(text); }
        for choice in state.choices() {
            if seen.contains(&choice.successor) { continue; }
            let mut next_text = text.clone();
            next_text.push(choice.text);
            seen.push(choice.successor.clone());
            queue.push_back((choice.successor, next_text));
        }
    }
    None
}

fn local_realization_rows(
    state: &NonStereoWriterState<NativeSolverState>,
    entries: &[DerivedEntry],
    entry: DerivedEntry,
) -> Vec<LocalRealizationRow> {
    let context = LocalLayoutContext {
        order: LocalBondOrder {
            atom: entry.atom,
            entry_bond: Some(entry.bond),
            emitted_bonds: Vec::new(),
            ring_occurrence_count: 0,
        },
        waiting_ring_bonds: Vec::new(),
        residual_attachment_bonds: entries.iter().filter(|child| child.from == entry.atom)
            .map(|child| vec![child.bond]).collect(),
    };
    let bonds = state.traversal_emission_specs(entry.bond, entry.from)
        .expect("the fixture has only prepared physical carrier choices");
    let mut rows = Vec::new();
    for (spelling, atom_restrictions, activate) in
        state.atom_token_specs(entry.atom, Some(entry.bond), &context)
    {
        assert!(atom_restrictions.is_empty() && activate.is_empty(),
            "fixed nonstereo fixture atoms have one semantic spelling class");
        let AtomSpellingStart::Grammar { state: atom_grammar, .. } = spelling;
        let Some(atom_text) = one_finite_atom_spelling(atom_grammar) else { continue; };
        for bond in &bonds {
            let mut restrictions = atom_restrictions.clone();
            restrictions.extend(bond.semantic_restrictions.iter().copied());
            let partition = state.surface.molecule().bond_role_partition(entry.bond).unwrap();
            restrictions.push((state.surface.molecule().bond_decision_variable(entry.bond).unwrap(),
                partition.traversal_values()));
            if let Some(root) = state.surface.molecule().root_variable(entry.atom) {
                restrictions.push((root, Domain::singleton(0).unwrap()));
            }
            rows.push(LocalRealizationRow {
                entry, bond_text: bond.text, atom_text: atom_text.clone(),
                restrictions, activate: activate.clone(),
            });
        }
    }
    rows
}

fn joint_rows_are_consistent(
    state: &NonStereoWriterState<NativeSolverState>, rows: &[&LocalRealizationRow],
) -> bool {
    let restrictions = rows.iter().flat_map(|row| row.restrictions.iter().copied())
        .collect::<Vec<_>>();
    assert!(state.surface.molecule().graph().atom_ids()
        .all(|atom| state.surface.tetrahedral_center(atom).is_none()));
    let activate = rows.iter().flat_map(|row| row.activate.iter().copied())
        .collect::<Vec<_>>();
    matches!(state.structural.transitioned_semantics(&restrictions, &activate).unwrap(),
        Consistency::Consistent(_))
}

fn assert_fixture_factor_inventory(surface: &PreparedNonStereo) {
    assert!(surface.event_schedule_factor.is_none(), "this proof gate excludes future event facts");
    assert!(surface.ordinary_order_pairs.iter().all(|pairs| pairs.is_empty()),
        "this fixture must not inherit speculative ordinary pair order");
    let model = surface.molecule().constraint_model();
    for index in 0..model.factor_count() {
        let factor = model.factor(FactorId::new(index as u32)).unwrap();
        let activation = model.factor_activation(FactorId::new(index as u32)).unwrap();
        match factor {
            FactorDefinition::TraversalTopology(_) | FactorDefinition::BinaryRelation(_) =>
                assert_eq!(activation, FactorActivation::Always),
            FactorDefinition::DirectionalRingPlacement(_) | FactorDefinition::RingCountAtMost(_) =>
                assert_eq!(activation, FactorActivation::Latent),
            _ => panic!("fixture has an unexamined factor family: {factor:?}"),
        }
    }
}

// This chain has one incoming incidence at each unvisited atom. Every atom is
// fixed, non-aromatic, and non-tetrahedral, so its lexical paths carry the same
// empty atom restriction set. Only the three physical carrier marks vary.
fn fixture_rows(
    rooted: &NonStereoWriterState<NativeSolverState>, entries: &[DerivedEntry],
    bonds: &[BondId], marks: [CarrierMark; 3],
) -> Option<Vec<LocalRealizationRow>> {
    entries.iter().copied().map(|entry| {
        let rows = local_realization_rows(rooted, entries, entry);
        assert!(!rows.is_empty());
        let site = [bonds[0], bonds[2], bonds[4]].iter().position(|&bond| bond == entry.bond);
        assert_eq!(rows.len(), if site.is_some() { 2 } else { 1 });
        assert!(rows.iter().all(|row| row.activate.is_empty()));
        assert!(rows.iter().all(|row| row.restrictions == rows[0].restrictions
            || site.is_some()), "ordinary entry has another semantic class");
        let selected = rows.into_iter().find(|row| match site {
            Some(index) => {
                let PreparedDirectionalBondStatus::CompiledSite(carrier) =
                    rooted.surface.directional_status(entry.bond) else { unreachable!() };
                row.restrictions.iter().any(|&(variable, domain)| variable == carrier.mark_variable
                    && domain == marks[index].singleton_domain())
            }
            None => true,
        });
        selected
    }).collect()
}

fn independent_joint_carrier_parity(marks: [CarrierMark; 3]) -> bool {
    let sign = marks.map(|mark| match mark {
        CarrierMark::SlashAtFixedA => Some(false),
        CarrierMark::BackslashAtFixedA => Some(true),
        CarrierMark::Plain => None,
    });
    match sign {
        [Some(left), Some(shared), Some(right)] => left != shared && shared == right,
        _ => false,
    }
}

fn assert_guided_inline_trace(
    rooted: &NonStereoWriterState<NativeSolverState>,
    rows: &[LocalRealizationRow], marks: [CarrierMark; 3], bonds: &[BondId],
) {
    let restrictions = rows.iter().flat_map(|row| row.restrictions.iter().copied())
        .collect::<Vec<_>>();
    let mut state = rooted.clone();
    state.structural = state.structural.transitioned_semantics(&restrictions, &[])
        .unwrap().unwrap_consistent();
    let mut visited = state.structural.observe_raw().traversal.visited_atoms.len();
    for row in rows {
        assert_eq!(state.active_atom(), Some(row.entry.from));
        assert_eq!(state.pending, None);
        let entry = row.entry;
        let candidates = state.choices().unwrap();
        let entered = candidates.into_iter().find(|choice| {
            choice.successor().active_atom() == Some(entry.atom)
                && matches!(&choice.successor().pending,
                    Some(PendingEmission::InlineAtom(incident))
                    if incident.bond() == entry.bond && incident.atom() == entry.atom)
        }).expect("induced inline child must be a labelled event");
        assert_eq!(entered.text(), row.bond_text);
        state = entered.into_successor();
        visited += 1;
        let observed = state.structural.observe_raw().traversal;
        assert_eq!(observed.visited_atoms.len(), visited);
        assert_eq!(observed.bond_progress[entry.bond.index()],
            ObservedBondProgress::Traversed { from: entry.from, to: entry.atom });
        assert_eq!(observed.active_frame.as_ref().unwrap().entry_bond, Some(entry.bond));
        assert!(observed.branch_returns.is_empty());
        assert!(matches!(&state.pending, Some(PendingEmission::InlineAtom(committed))
            if committed.incident == state.surface.molecule().graph().neighbors(entry.from).unwrap()
                .iter().copied().find(|incident| incident.bond() == entry.bond).unwrap()));
        let mut grammar_state = match &state.pending {
            Some(PendingEmission::InlineAtom(committed)) => committed.grammar.clone(),
            _ => panic!("guided inline trace requires its committed child grammar"),
        };
        for character in row.atom_text.chars() {
            let grammar_choice = grammar_state.choices().into_iter()
                .find(|step| step.text == character).unwrap();
            let choice = state.choices().unwrap().into_iter().find(|choice| {
                choice.text() == character.to_string()
                    && match &choice.successor().atom_spelling {
                        Some(progress) => progress.state == grammar_choice.successor,
                        None => grammar_choice.successor.is_complete(),
                    }
            }).expect("row grammar character and state must be offered");
            grammar_state = grammar_choice.successor;
            state = choice.into_successor();
            if let Some(progress) = &state.atom_spelling {
                assert_eq!(progress.state, grammar_state);
            } else {
                assert!(grammar_state.is_complete());
            }
        }
        assert!(grammar_state.is_complete());
        assert!(state.atom_spelling.is_none());
    }
    assert!(state.is_accepted());
    assert!(state.choices().unwrap().is_empty());
    for (index, &bond) in [bonds[0], bonds[2], bonds[4]].iter().enumerate() {
        let PreparedDirectionalBondStatus::CompiledSite(site) = state.surface.directional_status(bond)
            else { unreachable!() };
        assert_eq!(state.structural.semantic_domain(site.mark_variable), marks[index].singleton_domain());
    }
}

#[test]
fn shared_carrier_forest_has_complete_joint_rows_and_guided_acceptance_from_both_sources() {
    let (surface, atoms, bonds) = shared_directional_forest_fixture();
    assert_fixture_factor_inventory(&surface);
    for &root in &[atoms[0], atoms[5]] {
        let mut state = NonStereoWriterState::<NativeSolverState>::initial(&surface)
            .unwrap().unwrap_consistent();
        force_fixture_traversal_roles(&mut state, &bonds);
        let (_, rooted) = super::tests::only_root_atom(&state, root);
        let entries = derived_forest_entries(&rooted).unwrap();
        assert_eq!(entries.len(), 5);
        assert!(entries.iter().all(|entry| entry.from != entry.atom));
        let mut compatible = 0;
        for encoded in 0..27 {
            let mut remaining = encoded;
            let marks = std::array::from_fn(|_| {
                let mark = CarrierMark::from_value((remaining % 3) as u8).unwrap();
                remaining /= 3;
                mark
            });
            let Some(rows) = fixture_rows(&rooted, &entries, &bonds, marks) else {
                assert!(!independent_joint_carrier_parity(marks),
                    "a parity-compatible mark must have a lexical row");
                continue;
            };
            let references = rows.iter().collect::<Vec<_>>();
            let consistent = joint_rows_are_consistent(&rooted, &references);
            assert_eq!(consistent, independent_joint_carrier_parity(marks),
                "joint parity/mark disagreement for {marks:?}");
            if !consistent { continue; }
            compatible += 1;
            for row in &rows {
                if let Some(index) = [bonds[0], bonds[2], bonds[4]].iter()
                    .position(|&bond| bond == row.entry.bond) {
                    let graph_bond = surface.molecule().graph().bond(row.entry.bond).unwrap();
                    let from_a = graph_bond.a() == row.entry.from;
                    let expected = match (marks[index], from_a) {
                        (CarrierMark::SlashAtFixedA, true)
                        | (CarrierMark::BackslashAtFixedA, false) => "/",
                        (CarrierMark::BackslashAtFixedA, true)
                        | (CarrierMark::SlashAtFixedA, false) => "\\",
                        _ => unreachable!(),
                    };
                    assert_eq!(row.bond_text, expected);
                }
            }
            assert_guided_inline_trace(&rooted, &rows, marks, &bonds);
        }
        assert_eq!(compatible, 2);
    }
}

