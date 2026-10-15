use super::*;
use crate::native::NativeSolverState;
use crate::prepared::PreparedGraphBuilder;

type State = NonStereoWriterState<NativeSolverState>;

fn ring(elements: &[(&str, u8, u32)]) -> PreparedNonStereo {
    let mut graph = PreparedGraphBuilder::new();
    let ids = elements.iter().map(|_| graph.add_atom().unwrap()).collect::<Vec<_>>();
    for index in 0..ids.len() { graph.add_bond(ids[index], ids[(index + 1) % ids.len()]).unwrap(); }
    let atoms = elements.iter().map(|(element, h, valence)| PreparedAtomToken::Facts {
        facts: AtomFacts::new(ElementSymbol::new(element).unwrap(), true, None, 0, *h, 0, Some(*valence)).unwrap(),
        source_context: AtomSourceContext::default(), tetrahedral_reference_order: None, at_for_even_parity: true,
    }).collect();
    PreparedNonStereo::with_atom_tokens(PreparedMolecule::new(graph.build()), atoms, vec![NonStereoBondToken::Aromatic; ids.len()]).unwrap()
}

fn segmented_ring(elements: &[(&str, u8, u32)]) -> PreparedNonStereo {
    let mut graph = PreparedGraphBuilder::new();
    let ids = elements.iter().map(|_| graph.add_atom().unwrap()).collect::<Vec<_>>();
    for index in 0..ids.len() { graph.add_bond(ids[index], ids[(index + 1) % ids.len()]).unwrap(); }
    let atoms = elements.iter().map(|(element, h, valence)| facts(element, true, 0, *h, Some(*valence))).collect();
    PreparedNonStereo::with_atom_tokens_and_directional_for_writer(
        PreparedMolecule::new(graph.build()), atoms,
        vec![NonStereoBondToken::Aromatic; ids.len()], Vec::new(), WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true,
    ).unwrap()
}

fn consume(surface: &PreparedNonStereo, text: &str) -> Vec<State> {
    let mut states = vec![State::initial(surface).unwrap().unwrap_consistent()];
    for expected in text.chars() {
        states = states.into_iter().flat_map(|state| state.choices().unwrap().into_iter()
            .filter(|choice| choice.text() == expected.to_string()).map(|choice| choice.into_successor())).collect();
        if states.is_empty() { break; }
    }
    states
}

fn supports(surface: &PreparedNonStereo, text: &str) -> bool {
    consume(surface, text).iter().any(State::is_accepted)
}

#[test]
fn benzene_matching_assignments_and_ring_end_placement_are_correlated() {
    let surface = ring(&[("C", 1, 4); 6]);
    for witness in ["C1=CC=CC=C1", "C=1C=CC=CC1", "C1C=CC=CC=1", "C=1C=CC=CC=1", "c1ccccc1", "c:1:c:c:c:c:c1"] {
        assert!(supports(&surface, witness), "missing {witness}");
    }
    assert!(!consume(&surface, "C1=C").is_empty());
    assert!(consume(&surface, "C1=C=").is_empty(), "adjacent doubles must contradict the carbon demand before EOS");
    for mixed in ["C1=CC=CC=c1", "c1=ccccc1"] { assert!(supports(&surface, mixed)); }
    for impossible in ["C1C=CC=CC1", "C1=C=CC=CC1"] {
        assert!(!supports(&surface, impossible), "accepted {impossible}");
    }
}

#[test]
fn aromatic_matching_survives_forest_segment_ring_obligations() {
    for (elements, witnesses) in [
        (vec![("C", 1, 4); 6], ["c1cc2.c2cc1", "C1=CC=2.C2C=C1"]),
        (vec![("N", 1, 3), ("C", 1, 4), ("C", 1, 4), ("C", 1, 4), ("C", 1, 4)],
         ["[nH]1cc2.c2c1", "N1C=C2.C2=C1"]),
    ] {
        let surface = segmented_ring(&elements);
        assert_eq!(surface.molecule().topology(), crate::model::TraversalTopologyRequirement::Forest);
        for witness in witnesses { assert!(supports(&surface, witness), "missing {witness}"); }
        if elements.len() == 5 {
            assert!(!consume(&surface, "N1C=C2.C").is_empty());
            assert!(!supports(&surface, "N1C=C2.C=2C1"), "two double bonds exceed one carbon demand");
        }
    }
}

#[test]
fn shared_bracket_prefix_preserves_both_component_modes() {
    let surface = ring(&[("C", 1, 4); 6]);
    let states = consume(&surface, "[");
    let modes = states.iter().map(|state| {
        let atom = state.structural.active_atom().unwrap();
        state.structural.semantic_domain(surface.atom_aromatic_modes[atom.index()].unwrap())
    }).collect::<Vec<_>>();
    assert!(modes.contains(&Domain::singleton(0).unwrap()));
    assert!(modes.contains(&Domain::singleton(1).unwrap()));
    assert!(supports(&surface, "[CH]1=CC=CC=C1"));
    assert!(supports(&surface, "[cH]1ccccc1"));
}

#[test]
fn general_writer_admits_mixed_bracket_cases_with_one_matching() {
    let surface = ring(&[("C", 1, 4); 6]);
    for witness in [
        "[cH]1:[cH]-[CH]=[CH]-[CH]=[CH]-1",
        "[cH]1-[CH]=[CH]-[CH]=[CH]-[CH]=1",
        "c1=cc=cc=c1",
        "c1=c-c=c-c=c-1",
    ] {
        assert!(supports(&surface, witness), "missing {witness}");
    }
    for impossible in [
        "[cH]1:[CH]-[CH]=[CH]-[CH]=[CH]-1",
        "[cH]1:[cH]-[CH]=[CH]=[CH]-[CH]-1",
        "c1=cc=cc-c1",
    ] {
        assert!(!supports(&surface, impossible), "accepted {impossible}");
    }
}

#[test]
fn bare_aromatic_carbon_accounts_for_graph_hydrogen_and_exocyclic_double() {
    let mut graph_h_atoms = vec![facts("H", false, 0, 0, None), facts("C", true, 0, 0, Some(4))];
    graph_h_atoms.extend((0..5).map(|_| facts("C", true, 0, 1, Some(4))));
    let mut graph_h_edges = vec![(0, 1, NonStereoBondToken::Single)];
    graph_h_edges.extend((0..6).map(|i| (1 + i, 1 + (i + 1) % 6, NonStereoBondToken::Aromatic)));
    let graph_h = graph_surface(graph_h_atoms, &graph_h_edges).unwrap();
    let carbon = graph_h.molecule().graph().atom_ids().nth(1).unwrap();
    let context = graph_h.atom_spelling_context(carbon, None);
    assert_eq!(context.bonds.single_bonds, 1);
    assert_eq!(context.bonds.aromatic_bonds, 2);
    assert!(AtomSpellingState::initial(*graph_h.atom_facts(carbon), context).choices().iter()
        .any(|choice| choice.field == AtomSpellingField::BareElement && choice.text == 'c'));
    let wrong_h = AtomFacts::new(ElementSymbol::new("C").unwrap(), true, None, 0, 1, 0, Some(4)).unwrap();
    assert!(!AtomSpellingState::initial(wrong_h, context).choices().iter()
        .any(|choice| choice.field == AtomSpellingField::BareElement && choice.text == 'c'));

    let mut carbonyl_atoms = vec![facts("C", true, 0, 0, Some(4))];
    carbonyl_atoms.extend((0..4).map(|_| facts("C", true, 0, 1, Some(4))));
    carbonyl_atoms.push(facts("N", true, 0, 1, Some(3)));
    carbonyl_atoms.push(facts("O", false, 0, 0, None));
    let mut carbonyl_edges = (0..6).map(|i| (i, (i + 1) % 6, NonStereoBondToken::Aromatic)).collect::<Vec<_>>();
    carbonyl_edges.push((0, 6, NonStereoBondToken::Double));
    let carbonyl = graph_surface(carbonyl_atoms, &carbonyl_edges).unwrap();
    assert!(supports(&carbonyl, "O=c1cccc[nH]1"));
    assert!(!supports(&carbonyl, "O=[cH]1cccc[nH]1"));
}

#[test]
fn mixed_brackets_preserve_heteroatom_and_charge_demands() {
    let pyrrole = ring(&[("N", 1, 3), ("C", 1, 4), ("C", 1, 4), ("C", 1, 4), ("C", 1, 4)]);
    assert!(supports(&pyrrole, "[nH]1:[cH]=[CH]-[CH]=[cH]:1"));
    assert!(!supports(&pyrrole, "[nH]1:[cH]=[CH]=[CH]-[cH]:1"));

    let mut charged_atoms = vec![facts("N", true, 1, 1, Some(4))];
    charged_atoms.extend((0..5).map(|_| facts("C", true, 0, 1, Some(4))));
    let edges = (0..6).map(|i| (i, (i + 1) % 6, NonStereoBondToken::Aromatic)).collect::<Vec<_>>();
    let charged = graph_surface(charged_atoms, &edges).unwrap();
    assert!(supports(&charged, "[nH+]1:[cH]-[CH]=[CH]-[CH]=[CH]-1"));
    assert!(!supports(&charged, "[nH+]1:[CH]-[CH]=[CH]-[CH]=[CH]-1"));
}

#[test]
fn heteroatom_zero_and_one_double_demands_use_same_native_matching() {
    for (elements, witness) in [
        (vec![("N", 0, 3), ("C", 1, 4), ("C", 1, 4), ("C", 1, 4), ("C", 1, 4), ("C", 1, 4)], "N1=CC=CC=C1"),
        (vec![("N", 1, 3), ("C", 1, 4), ("C", 1, 4), ("C", 1, 4), ("C", 1, 4)], "N1C=CC=C1"),
        (vec![("O", 0, 2), ("C", 1, 4), ("C", 1, 4), ("C", 1, 4), ("C", 1, 4)], "O1C=CC=C1"),
        (vec![("S", 0, 2), ("C", 1, 4), ("C", 1, 4), ("C", 1, 4), ("C", 1, 4)], "S1C=CC=C1"),
        (vec![("P", 1, 3), ("C", 1, 4), ("C", 1, 4), ("C", 1, 4), ("C", 1, 4)], "P1C=CC=C1"),
    ] {
        assert!(supports(&ring(&elements), witness), "missing {witness}");
    }
}

fn facts(element: &str, aromatic: bool, charge: i8, h: u8, valence: Option<u32>) -> PreparedAtomToken {
    PreparedAtomToken::Facts {
        facts: AtomFacts::new(ElementSymbol::new(element).unwrap(), aromatic, None, charge, h, 0, valence).unwrap(),
        source_context: AtomSourceContext::default(), tetrahedral_reference_order: None, at_for_even_parity: true,
    }
}

fn graph_surface(atoms: Vec<PreparedAtomToken>, edges: &[(usize, usize, NonStereoBondToken)]) -> Result<PreparedNonStereo, PreparedNonStereoError> {
    let mut graph = PreparedGraphBuilder::new();
    let ids = atoms.iter().map(|_| graph.add_atom().unwrap()).collect::<Vec<_>>();
    for &(a, b, _) in edges { graph.add_bond(ids[a], ids[b]).unwrap(); }
    PreparedNonStereo::with_atom_tokens(PreparedMolecule::new(graph.build()), atoms, edges.iter().map(|edge| edge.2).collect())
}

#[test]
fn maximal_ordinary_single_bridge_respects_both_emitted_aromatic_modes() {
    let atoms = (0..12).map(|index| facts("C", true, 0, if index == 0 || index == 6 { 0 } else { 1 }, Some(4))).collect();
    let mut edges = (0..6).map(|index| (index, (index + 1) % 6, NonStereoBondToken::Aromatic)).collect::<Vec<_>>();
    edges.extend((6..12).map(|index| (index, 6 + (index + 1 - 6) % 6, NonStereoBondToken::Aromatic)));
    edges.push((0, 6, NonStereoBondToken::Elided));
    let bridge = BondId::new(u32::try_from(edges.len() - 1).unwrap());
    let surface = graph_surface(atoms, &edges).unwrap();
    let source = AtomId::new(0);
    let other = AtomId::new(6);
    for (source_text, source_mode) in [("c", 0_u8), ("C", 1)] {
        let roots = consume(&surface, source_text).into_iter().filter(|state| {
            state.active_atom() == Some(source)
                && state.atom_spelling.is_none()
                && state.structural.semantic_domain(surface.atom_aromatic_modes[source.index()].unwrap())
                    == Domain::singleton(source_mode).unwrap()
        }).collect::<Vec<_>>();
        assert!(!roots.is_empty(), "prepared source mode must reach a completed atom token");
        for other_mode in [0_u8, 1] {
            let mut checked = 0;
            for root in &roots {
                let mut state = root.clone();
                state.structural = state.structural.transitioned_semantics(
                    &[(surface.atom_aromatic_modes[other.index()].unwrap(), Domain::singleton(other_mode).unwrap())], &[])
                    .unwrap().unwrap_consistent();
                let texts = state.chemical_bond_specs(bridge, source).into_iter()
                    .map(|(_, text, _, _)| text).collect::<std::collections::BTreeSet<_>>();
                let expected = if source_mode == 0 && other_mode == 0 {
                    std::collections::BTreeSet::from(["-"])
                } else {
                    std::collections::BTreeSet::from(["", "-"])
                };
                assert_eq!(texts, expected, "source={source_text}, other_mode={other_mode}");
                checked += 1;
            }
            assert!(checked > 0);
        }
    }
}

#[test]
fn maximal_ordinary_single_ring_plan_requires_a_visible_dash_for_two_aromatic_modes() {
    let atoms = (0..12).map(|index| facts("C", true, 0,
        if [0, 1, 6, 7].contains(&index) { 0 } else { 1 }, Some(4))).collect();
    let mut edges = (0..6).map(|index| (index, (index + 1) % 6, NonStereoBondToken::Aromatic)).collect::<Vec<_>>();
    edges.extend((6..12).map(|index| (index, 6 + (index + 1 - 6) % 6, NonStereoBondToken::Aromatic)));
    edges.extend([(0, 6, NonStereoBondToken::Elided), (1, 7, NonStereoBondToken::Elided)]);
    let candidate_ring = BondId::new(12);
    let surface = graph_surface(atoms, &edges).unwrap();
    let source = AtomId::new(0);
    let other = AtomId::new(6);
    let all_ring_plans = BondRepresentation::role_partition().ring_values();
    for (source_text, source_mode) in [("c", 0_u8), ("C", 1)] {
        let roots = consume(&surface, source_text).into_iter().filter(|state| {
            state.active_atom() == Some(source) && state.atom_spelling.is_none()
                && state.structural.semantic_domain(surface.atom_aromatic_modes[source.index()].unwrap())
                    == Domain::singleton(source_mode).unwrap()
        }).collect::<Vec<_>>();
        assert!(!roots.is_empty());
        for other_mode in [0_u8, 1] {
            for root in &roots {
                let mut state = root.clone();
                state.structural = state.structural.transitioned_semantics(
                    &[(surface.atom_aromatic_modes[other.index()].unwrap(), Domain::singleton(other_mode).unwrap())], &[])
                    .unwrap().unwrap_consistent();
                let plans = state.ring_emission_specs(candidate_ring, source).into_iter()
                    .fold(Domain::empty(), |plans, (_, _, restrictions, allowed)| {
                        assert!(restrictions.iter().all(|(variable, domain)|
                            state.structural.semantic_domain(*variable).intersect(*domain) == *domain));
                        plans.union(allowed)
                    });
                let expected = if source_mode == 0 && other_mode == 0 {
                    all_ring_plans.intersect(BondRepresentation::explicit_domain())
                } else { all_ring_plans };
                assert_eq!(plans, expected, "source={source_text}, other_mode={other_mode}");
            }
        }
    }
}

#[test]
fn maximal_aromatic_bridge_branch_publishes_dash_before_child_atom() {
    let atoms = (0..12).map(|index| facts("C", true, 0,
        if index == 0 || index == 6 { 0 } else { 1 }, Some(4))).collect();
    let mut edges = (0..6).map(|index| (index, (index + 1) % 6, NonStereoBondToken::Aromatic)).collect::<Vec<_>>();
    edges.extend((6..12).map(|index| (index, 6 + (index + 1 - 6) % 6, NonStereoBondToken::Aromatic)));
    edges.push((0, 6, NonStereoBondToken::Elided));
    let surface = graph_surface(atoms, &edges).unwrap();
    let source = AtomId::new(0);
    let other = AtomId::new(6);
    let bridge = BondId::new(12);
    let roots = consume(&surface, "c").into_iter().filter(|state|
        state.active_atom() == Some(source) && state.atom_spelling.is_none()).collect::<Vec<_>>();
    assert!(!roots.is_empty());
    let mut branch_count = 0;
    for mut root in roots {
        root.structural = root.structural.transitioned_semantics(
            &[(surface.atom_aromatic_modes[other.index()].unwrap(), Domain::singleton(0).unwrap())], &[])
            .unwrap().unwrap_consistent();
        for opening in root.choices().unwrap() {
            let opened = opening.into_successor();
            if !matches!(opened.structural.observe_raw().traversal.bond_progress[0],
                crate::traversal::ObservedBondProgress::RingOpen { first_endpoint } if first_endpoint == source) {
                continue;
            }
            for branch in opened.choices().unwrap() {
                let pending = branch.into_successor();
                if !matches!(&pending.pending, Some(PendingEmission::BranchTraversalEmission(incident))
                    if incident.bond() == bridge) { continue; }
                branch_count += 1;
                let texts = pending.choices().unwrap().into_iter().map(|choice| choice.text().to_owned())
                    .collect::<std::collections::BTreeSet<_>>();
                assert_eq!(texts, std::collections::BTreeSet::from(["-".to_owned()]));
            }
        }
    }
    assert!(branch_count > 0, "the bridge branch must be reachable after the selected ring opening");
}

#[test]
fn fused_naphthalene_accepts_three_explicit_matching_witnesses() {
    let atoms = (0..10).map(|i| facts("C", true, 0, if i == 3 || i == 8 { 0 } else { 1 }, Some(4))).collect();
    let mut edges = (0..9).map(|i| (i, i + 1, NonStereoBondToken::Aromatic)).collect::<Vec<_>>();
    edges.extend([(0, 9, NonStereoBondToken::Aromatic), (3, 8, NonStereoBondToken::Aromatic)]);
    let surface = graph_surface(atoms, &edges).unwrap();
    // On traversal 0..9, these spell the matchings {01,23,45,67,89},
    // {12,34,56,78,90}, and {12,38,45,67,90}; the last uses the fusion edge.
    for witness in ["C1=CC=C2C=CC=CC2=C1", "C1C=CC2=CC=CC=C2C=1", "C1C=CC=2C=CC=CC2C=1", "c1ccc2ccccc2c1"] {
        assert!(supports(&surface, witness), "missing {witness}");
    }
    assert!(!supports(&surface, "C1CCC2CCCCC2C1"));
}

#[test]
fn disconnected_aromatic_components_choose_modes_independently() {
    let atoms = (0..12).map(|_| facts("C", true, 0, 1, Some(4))).collect();
    let edges = [0, 6].into_iter().flat_map(|offset| (0..6).map(move |i| (offset + i, offset + (i + 1) % 6, NonStereoBondToken::Aromatic))).collect::<Vec<_>>();
    let surface = graph_surface(atoms, &edges).unwrap();
    for first in ["c1ccccc1", "C1=CC=CC=C1"] {
        for second in ["c1ccccc1", "C1=CC=CC=C1"] {
            assert!(supports(&surface, &format!("{first}.{second}")), "missing component modes {first}.{second}");
        }
    }
    assert!(supports(&surface, "c1ccccc1.C1=CC=CC=c1"));
}

#[test]
fn charged_atoms_keep_source_hydrogens_in_both_modes() {
    for (element, charge, count, valence, witnesses) in [
        ("N", 1, 6, 4, ["[NH+]1=CC=CC=C1", "[nH+]1ccccc1"]),
        ("C", -1, 5, 3, ["[CH-]1C=CC=C1", "[cH-]1cccc1"]),
    ] {
        let mut atoms = vec![facts(element, true, charge, 1, Some(valence))];
        atoms.extend((1..count).map(|_| facts("C", true, 0, 1, Some(4))));
        let edges = (0..count).map(|i| (i, (i + 1) % count, NonStereoBondToken::Aromatic)).collect::<Vec<_>>();
        let surface = graph_surface(atoms, &edges).unwrap();
        for witness in witnesses { assert!(supports(&surface, witness), "missing {witness}"); }
        assert!(!supports(&surface, &witnesses[0].replace("H", "H2")), "changed source H count");
    }
}

#[test]
fn exocyclic_double_consumes_valence_without_aromatic_double_demand() {
    let mut atoms = vec![facts("C", true, 0, 0, Some(4)), facts("N", true, 0, 1, Some(3))];
    atoms.extend((2..6).map(|_| facts("C", true, 0, 1, Some(4))));
    atoms.push(facts("O", false, 0, 0, None));
    let mut edges = (0..6).map(|i| (i, (i + 1) % 6, NonStereoBondToken::Aromatic)).collect::<Vec<_>>();
    edges.push((0, 6, NonStereoBondToken::Double));
    let surface = graph_surface(atoms, &edges).unwrap();
    for witness in ["O=C1NC=CC=C1", "O=c1[nH]cccc1"] {
        assert!(supports(&surface, witness), "missing {witness}");
    }
    assert!(consume(&surface, "O=C1=").is_empty(), "exocyclic double already fills carbon valence");
    assert!(!supports(&surface, "OC1NC=CC=C1"), "lost fixed exocyclic double");
}

#[test]
fn aromatic_source_bonds_and_unsupported_demands_fail_preparation() {
    let edge = [(0, 1, NonStereoBondToken::Aromatic)];
    let mismatch = graph_surface(vec![facts("C", true, 0, 1, Some(4)), facts("C", false, 0, 3, None)], &edge);
    assert!(matches!(mismatch, Err(PreparedNonStereoError::SourceBondMismatch { bond }) if bond == BondId::new(0)));
    for valence in [1, 4] {
        let demand = graph_surface(vec![facts("C", true, 0, 1, Some(valence)), facts("C", true, 0, 1, Some(3))], &edge);
        assert!(matches!(demand, Err(PreparedNonStereoError::AromaticDemand { atom, .. }) if atom == AtomId::new(0)), "unsupported demand from total valence {valence}");
    }
}

#[test]
fn source_aromatic_component_requires_a_double_bond_assignment() {
    let atoms = (0..5).map(|_| facts("C", true, 0, 1, Some(4))).collect();
    let edges = (0..5).map(|i| (i, (i + 1) % 5, NonStereoBondToken::Aromatic)).collect::<Vec<_>>();
    let odd_carbon_ring = graph_surface(atoms, &edges);
    assert!(odd_carbon_ring.as_ref().unwrap_err().to_string().contains("required double bonds cannot be assigned"));
    assert!(matches!(odd_carbon_ring, Err(PreparedNonStereoError::InfeasibleAromaticComponent { atom }) if atom == AtomId::new(0)));

    let zero_demand_atoms = (0..5).map(|_| facts("O", true, 0, 0, Some(2))).collect();
    assert!(graph_surface(zero_demand_atoms, &edges).is_ok(), "zero double demands need no matching edges");

    // Abstract source facts: four demands have even parity, but three leaves
    // all need the same center, so no assignment can satisfy the component.
    let star_atoms = vec![
        facts("C", true, 0, 0, Some(4)),
        facts("C", true, 0, 2, Some(4)),
        facts("C", true, 0, 2, Some(4)),
        facts("C", true, 0, 2, Some(4)),
    ];
    let star_edges = (1..4).map(|leaf| (0, leaf, NonStereoBondToken::Aromatic)).collect::<Vec<_>>();
    assert!(matches!(graph_surface(star_atoms, &star_edges), Err(PreparedNonStereoError::InfeasibleAromaticComponent { atom }) if atom == AtomId::new(0)));
}

#[test]
fn chemical_aromatic_directional_carriers_fail_preparation_explicitly() {
    let mut graph = PreparedGraphBuilder::new();
    let ids = (0..8).map(|_| graph.add_atom().unwrap()).collect::<Vec<_>>();
    for i in 0..6 { graph.add_bond(ids[i], ids[(i + 1) % 6]).unwrap(); }
    let double_bond = graph.add_bond(ids[0], ids[6]).unwrap();
    let right_carrier = graph.add_bond(ids[6], ids[7]).unwrap();
    let mut atoms = vec![facts("C", true, 0, 0, Some(4))];
    atoms.extend((1..6).map(|index| if index == 3 { facts("N", true, 0, 1, Some(3)) } else { facts("C", true, 0, 1, Some(4)) }));
    atoms.extend([facts("C", false, 0, 1, None), facts("C", false, 0, 3, None)]);
    let mut tokens = vec![NonStereoBondToken::Aromatic; 6];
    tokens.extend([NonStereoBondToken::Double, NonStereoBondToken::Single]);
    let result = PreparedNonStereo::with_atom_tokens_and_directional(
        PreparedMolecule::new(graph.build()), atoms, tokens,
        vec![PreparedDirectionalRelation {
            double_bond, left_endpoint: ids[0],
            left_carriers: vec![PreparedDirectionalCarrier::unflipped(BondId::new(0))].into_boxed_slice(),
            right_endpoint: ids[6],
            right_carriers: vec![PreparedDirectionalCarrier::unflipped(right_carrier)].into_boxed_slice(),
            side_phase_xor: false,
        }],
    );
    assert!(matches!(result, Err(PreparedNonStereoError::UnsupportedAromaticDirectionalCarrier { bond }) if bond == BondId::new(0)));
}

#[test]
fn bare_boron_phosphorus_and_wildcard_keep_both_notation_languages() {
    for (element, source, kekule) in [
        ("B", "b1ccccc1", "B1=CC=CC=C1"),
        ("P", "p1ccccc1", "P1=CC=CC=C1"),
        ("*", "*1ccccc1", "*1=CC=CC=C1"),
    ] {
        let mut elements = vec![(element, 0, 3)];
        elements.extend([("C", 1, 4); 5]);
        let surface = ring(&elements);
        assert!(supports(&surface, source), "missing {source}");
        assert!(supports(&surface, kekule), "missing {kekule}");
        if element == "*" {
            assert!(supports(&surface, "*:1:c:c:c:c:c:1"));
            let states = consume(&surface, "*");
            let modes = states.iter().map(|state| {
                let atom = state.structural.active_atom().unwrap();
                state.structural.semantic_domain(surface.atom_aromatic_modes[atom.index()].unwrap())
            }).collect::<Vec<_>>();
            assert!(modes.contains(&Domain::from_bits(3)));
            assert!(!consume(&surface, "*1c").is_empty());
            assert!(!consume(&surface, "*1C").is_empty());
        }
    }
}
