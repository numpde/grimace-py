use std::collections::BTreeSet;

use super::*;
use crate::native::NativeSolverState;
use crate::prepared::PreparedGraphBuilder;

type State = ConnectedNonStereoWriterState<NativeSolverState>;

fn fixture(
    atom_text: &[&str],
    edges: &[(usize, usize, NonStereoBondToken)],
) -> (PreparedConnectedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = atom_text
        .iter()
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let mut bonds = Vec::with_capacity(edges.len());
    let mut bond_tokens = Vec::with_capacity(edges.len());
    for &(a, b, token) in edges {
        bonds.push(graph.add_bond(atoms[a], atoms[b]).unwrap());
        bond_tokens.push(token);
    }
    let surface = PreparedConnectedNonStereo::new(
        PreparedMolecule::new(graph.build()),
        atom_text.iter().map(|text| (*text).to_owned()).collect(),
        bond_tokens,
    )
    .unwrap();
    (surface, atoms, bonds)
}

fn incident(surface: &PreparedConnectedNonStereo, atom: AtomId, bond: BondId) -> AdjacentBond {
    surface
        .molecule()
        .graph()
        .neighbors(atom)
        .expect("fixture atom must exist")
        .iter()
        .copied()
        .find(|candidate| candidate.bond() == bond)
        .expect("fixture bond must be incident to the atom")
}

fn only_choice(state: &State, expected: NonStereoChoice, text: &str) {
    let choices = state.choices().unwrap();
    assert_eq!(choices.len(), 1);
    assert_eq!(choices[0].choice(), expected);
    assert_eq!(choices[0].text(), text);
}

fn advance(state: &State, choice: NonStereoChoice) -> (String, State) {
    state.advance(choice).unwrap()
}

#[test]
fn surface_rejects_invalid_bindings() {
    let empty = PreparedMolecule::new(PreparedGraphBuilder::new().build());
    assert!(matches!(
        PreparedConnectedNonStereo::new(empty, Vec::new(), Vec::new()),
        Err(PreparedConnectedNonStereoError::EmptyMolecule)
    ));

    let mut graph = PreparedGraphBuilder::new();
    graph.add_atom().unwrap();
    graph.add_atom().unwrap();
    let disconnected = PreparedMolecule::new(graph.build());
    assert!(matches!(
        PreparedConnectedNonStereo::new(
            disconnected,
            vec!["C".to_owned(), "O".to_owned()],
            Vec::new(),
        ),
        Err(PreparedConnectedNonStereoError::DisconnectedMolecule)
    ));

    let mut graph = PreparedGraphBuilder::new();
    graph.add_atom().unwrap();
    let single = PreparedMolecule::new(graph.build());
    assert!(matches!(
        PreparedConnectedNonStereo::new(single.clone(), Vec::new(), Vec::new()),
        Err(PreparedConnectedNonStereoError::AtomTextCountMismatch { .. })
    ));
    assert!(matches!(
        PreparedConnectedNonStereo::new(single, vec![String::new()], Vec::new()),
        Err(PreparedConnectedNonStereoError::EmptyAtomText(atom))
            if atom == AtomId::new(0)
    ));

    let mut graph = PreparedGraphBuilder::new();
    let atoms: [AtomId; 2] = std::array::from_fn(|_| graph.add_atom().unwrap());
    graph.add_bond(atoms[0], atoms[1]).unwrap();
    let bonded = PreparedMolecule::new(graph.build());
    assert!(matches!(
        PreparedConnectedNonStereo::new(bonded, vec!["C".to_owned(), "O".to_owned()], Vec::new(),),
        Err(PreparedConnectedNonStereoError::BondTokenCountMismatch { .. })
    ));
}

#[test]
fn closed_labels_are_immediately_reusable() {
    let first = BondId::new(0);
    let second = BondId::new(1);
    let third = BondId::new(2);
    let mut labels = RingLabels::default();

    let zero = labels.allocate(first).unwrap();
    let one = labels.allocate(second).unwrap();
    assert_eq!(zero, RingLabelSlot(0));
    assert_eq!(one, RingLabelSlot(1));
    labels.release(zero, first);
    assert_eq!(labels.allocate(third).unwrap(), RingLabelSlot(0));
}

#[test]
fn ring_label_spelling_fails_at_the_typed_dialect_boundary() {
    assert_eq!(ring_label_number_text(1).unwrap(), "1");
    assert_eq!(ring_label_number_text(9).unwrap(), "9");
    assert_eq!(ring_label_number_text(10).unwrap(), "%10");
    assert_eq!(ring_label_number_text(99).unwrap(), "%99");
    assert_eq!(
        ring_label_number_text(100),
        Err(SpellingError::RingLabelOutOfRange { label: 100 })
    );

    let mut labels = RingLabels::default();
    for index in 0..99_u32 {
        labels.allocate(BondId::new(index)).unwrap();
    }
    assert_eq!(
        labels.next_available(),
        Err(SpellingError::RingLabelOutOfRange { label: 100 })
    );
}

#[test]
fn unavailable_visible_choice_is_typed() {
    let surface = fixture(&["C"], &[]).0;
    let initial = State::initial(&surface).unwrap();

    assert!(matches!(
        initial.advance(NonStereoChoice::BranchClose),
        Err(NonStereoAdvanceError::ChoiceUnavailable(
            NonStereoChoice::BranchClose
        ))
    ));
}

#[test]
fn elided_triangle_emits_a_complete_ring() {
    let (surface, atoms, bonds) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = State::initial(&surface).unwrap();
    let left = incident(&surface, atoms[0], bonds[0]);
    let right = incident(&surface, atoms[0], bonds[1]);
    let between = incident(&surface, atoms[2], bonds[2]);
    let closing = incident(&surface, atoms[1], bonds[0]);

    let (root, rooted) = advance(&initial, NonStereoChoice::Root(atoms[0]));
    assert_eq!(
        rooted.choices().unwrap(),
        vec![
            VisibleChoice {
                choice: NonStereoChoice::RingOpen(left),
                text: "1".to_owned(),
            },
            VisibleChoice {
                choice: NonStereoChoice::RingOpen(right),
                text: "1".to_owned(),
            },
        ]
    );
    let (open, opened) = advance(&rooted, NonStereoChoice::RingOpen(left));
    only_choice(&opened, NonStereoChoice::InlineChild(right), "C");
    let (first_child, walked) = advance(&opened, NonStereoChoice::InlineChild(right));
    let (second_child, walked) = advance(&walked, NonStereoChoice::InlineChild(between));
    only_choice(&walked, NonStereoChoice::RingClose(closing), "1");
    let (close, accepted) = advance(&walked, NonStereoChoice::RingClose(closing));

    assert_eq!(
        [root, open, first_child, second_child, close].concat(),
        "C1CC1"
    );
    assert!(accepted.is_accepted());
}

#[test]
fn explicit_ring_bond_is_emitted_at_closure_before_its_label() {
    let (surface, atoms, bonds) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = State::initial(&surface).unwrap();
    let ring = incident(&surface, atoms[0], bonds[0]);
    let entry = incident(&surface, atoms[0], bonds[1]);
    let between = incident(&surface, atoms[2], bonds[2]);
    let closing = incident(&surface, atoms[1], bonds[0]);

    let (root, rooted) = advance(&initial, NonStereoChoice::Root(atoms[0]));
    let (open, opened) = advance(&rooted, NonStereoChoice::RingOpen(ring));
    let (first_child, walked) = advance(&opened, NonStereoChoice::InlineChild(entry));
    let (second_child, walked) = advance(&walked, NonStereoChoice::InlineChild(between));
    only_choice(&walked, NonStereoChoice::RingClose(closing), "=");

    let (bond, pending_label) = advance(&walked, NonStereoChoice::RingClose(closing));
    assert_eq!(bond, "=");
    assert_eq!(pending_label.active_atom(), Some(atoms[1]));
    assert!(!pending_label.graph_is_complete());
    only_choice(&pending_label, NonStereoChoice::Pending, "1");

    let (label, accepted) = advance(&pending_label, NonStereoChoice::Pending);
    assert_eq!(
        [root, open, first_child, second_child, bond, label].concat(),
        "C1CC=1"
    );
    assert!(accepted.is_accepted());
}

#[test]
fn directed_ring_closure_uses_the_first_endpoint_orientation() {
    let (surface, atoms, bonds) = fixture(
        &["N", "B", "C"],
        &[
            (0, 1, NonStereoBondToken::DativeAToB),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = State::initial(&surface).unwrap();
    let ring = incident(&surface, atoms[0], bonds[0]);
    let entry = incident(&surface, atoms[0], bonds[1]);
    let between = incident(&surface, atoms[2], bonds[2]);
    let closing = incident(&surface, atoms[1], bonds[0]);

    let (root, rooted) = advance(&initial, NonStereoChoice::Root(atoms[0]));
    let (open, opened) = advance(&rooted, NonStereoChoice::RingOpen(ring));
    let (first_child, walked) = advance(&opened, NonStereoChoice::InlineChild(entry));
    let (second_child, walked) = advance(&walked, NonStereoChoice::InlineChild(between));
    only_choice(&walked, NonStereoChoice::RingClose(closing), "->");
    let (bond, pending_label) = advance(&walked, NonStereoChoice::RingClose(closing));
    let (label, accepted) = advance(&pending_label, NonStereoChoice::Pending);

    assert_eq!(
        [root, open, first_child, second_child, bond, label].concat(),
        "N1CB->1"
    );
    assert!(accepted.is_accepted());
}

#[test]
fn explicit_inline_bond_commits_before_child_entry() {
    let (surface, atoms, bonds) = fixture(&["C", "O"], &[(0, 1, NonStereoBondToken::Double)]);
    let initial = State::initial(&surface).unwrap();
    let edge = incident(&surface, atoms[0], bonds[0]);
    let (_, rooted) = advance(&initial, NonStereoChoice::Root(atoms[0]));
    let (bond, pending) = advance(&rooted, NonStereoChoice::InlineChild(edge));

    assert_eq!(bond, "=");
    assert_eq!(rooted.active_atom(), Some(atoms[0]));
    assert_eq!(pending.active_atom(), Some(atoms[0]));
    only_choice(&pending, NonStereoChoice::Pending, "O");
    let (atom, accepted) = advance(&pending, NonStereoChoice::Pending);
    assert_eq!(atom, "O");
    assert!(accepted.is_accepted());
}

#[test]
fn explicit_branch_commits_at_open_parenthesis() {
    let (surface, atoms, bonds) = fixture(
        &["C", "O", "N"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = State::initial(&surface).unwrap();
    let oxygen = incident(&surface, atoms[0], bonds[0]);
    let nitrogen = incident(&surface, atoms[0], bonds[1]);
    let (root, rooted) = advance(&initial, NonStereoChoice::Root(atoms[0]));
    let (open, pending_branch) = advance(&rooted, NonStereoChoice::BranchOpen(oxygen));
    let (bond, pending_atom) = advance(&pending_branch, NonStereoChoice::Pending);
    let (atom, branch) = advance(&pending_atom, NonStereoChoice::Pending);
    let (close, restored) = advance(&branch, NonStereoChoice::BranchClose);
    let (inline, accepted) = advance(&restored, NonStereoChoice::InlineChild(nitrogen));

    assert_eq!([root, open, bond, atom, close, inline].concat(), "C(=O)N");
    assert!(accepted.is_accepted());
}

fn reachable_strings(surface: &PreparedConnectedNonStereo) -> BTreeSet<String> {
    let initial = State::initial(surface).unwrap();
    let mut pending = vec![(initial, String::new())];
    let mut complete = BTreeSet::new();
    let mut explored = 0_usize;

    while let Some((state, prefix)) = pending.pop() {
        explored += 1;
        assert!(
            explored <= 200_000,
            "writer test exceeded its exploration bound"
        );
        if state.is_accepted() {
            complete.insert(prefix);
            continue;
        }

        let choices = state.choices().unwrap();
        assert!(
            !choices.is_empty(),
            "writer must not dead-end before acceptance"
        );
        for visible in choices {
            let expected = visible.text().to_owned();
            let (token, successor) = state.advance(visible.choice()).unwrap();
            assert_eq!(token, expected);
            pending.push((successor, format!("{prefix}{token}")));
        }
    }
    complete
}

fn permutations<T: Copy>(items: &[T]) -> Vec<Vec<T>> {
    fn recurse<T: Copy>(
        items: &[T],
        used: &mut [bool],
        current: &mut Vec<T>,
        output: &mut Vec<Vec<T>>,
    ) {
        if current.len() == items.len() {
            output.push(current.clone());
            return;
        }
        for index in 0..items.len() {
            if used[index] {
                continue;
            }
            used[index] = true;
            current.push(items[index]);
            recurse(items, used, current, output);
            current.pop();
            used[index] = false;
        }
    }

    if items.is_empty() {
        return vec![Vec::new()];
    }
    let mut output = Vec::new();
    recurse(
        items,
        &mut vec![false; items.len()],
        &mut Vec::with_capacity(items.len()),
        &mut output,
    );
    output
}

fn reference_tree_subtrees(
    surface: &PreparedConnectedNonStereo,
    atom: AtomId,
    parent: Option<AtomId>,
) -> BTreeSet<String> {
    let children = surface
        .molecule()
        .graph()
        .neighbors(atom)
        .expect("reference atom must exist")
        .iter()
        .copied()
        .filter(|incident| Some(incident.atom()) != parent)
        .collect::<Vec<_>>();
    let mut support = BTreeSet::new();

    for order in permutations(&children) {
        let mut partial = vec![surface.atom_text(atom).to_owned()];
        for (index, incident) in order.iter().copied().enumerate() {
            let child_support = reference_tree_subtrees(surface, incident.atom(), Some(atom));
            let bond = surface.bond_text(incident.bond(), atom);
            let inline = index + 1 == order.len();
            let mut next = Vec::new();
            for prefix in &partial {
                for child in &child_support {
                    if inline {
                        next.push(format!("{prefix}{bond}{child}"));
                    } else {
                        next.push(format!("{prefix}({bond}{child})"));
                    }
                }
            }
            partial = next;
        }
        support.extend(partial);
    }
    support
}

fn reference_tree_strings(surface: &PreparedConnectedNonStereo) -> BTreeSet<String> {
    surface
        .molecule()
        .graph()
        .atom_ids()
        .flat_map(|root| reference_tree_subtrees(surface, root, None))
        .collect()
}

#[test]
fn connected_tree_support_remains_exact() {
    let fixtures = [
        fixture(&["C"], &[]).0,
        fixture(
            &["C", "N", "O"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (1, 2, NonStereoBondToken::Elided),
            ],
        )
        .0,
        fixture(
            &["C", "N", "O", "F"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (0, 2, NonStereoBondToken::Elided),
                (0, 3, NonStereoBondToken::Elided),
            ],
        )
        .0,
        fixture(
            &["C", "N", "O", "F", "S"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (0, 2, NonStereoBondToken::Elided),
                (1, 3, NonStereoBondToken::Elided),
                (1, 4, NonStereoBondToken::Double),
            ],
        )
        .0,
    ];
    for surface in fixtures {
        assert_eq!(
            reachable_strings(&surface),
            reference_tree_strings(&surface)
        );
    }
}

#[test]
fn connected_triangle_support_is_writer_shaped() {
    let surface = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    )
    .0;

    assert_eq!(
        reachable_strings(&surface),
        BTreeSet::from(["C1CC1".to_owned()])
    );
}

#[test]
fn fused_and_bridged_cycles_have_complete_online_walks() {
    let fixtures = [
        fixture(
            &["A", "B", "C", "D"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (1, 2, NonStereoBondToken::Elided),
                (2, 0, NonStereoBondToken::Elided),
                (1, 3, NonStereoBondToken::Elided),
                (2, 3, NonStereoBondToken::Elided),
            ],
        )
        .0,
        fixture(
            &["A", "B", "C", "D", "E"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (1, 3, NonStereoBondToken::Elided),
                (0, 2, NonStereoBondToken::Elided),
                (2, 3, NonStereoBondToken::Elided),
                (0, 4, NonStereoBondToken::Elided),
                (4, 3, NonStereoBondToken::Elided),
            ],
        )
        .0,
    ];

    for surface in fixtures {
        assert!(!reachable_strings(&surface).is_empty());
    }
}
