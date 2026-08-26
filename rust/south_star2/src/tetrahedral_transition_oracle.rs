//! Independent one-step qualification for activated ring-coupled tetrahedral events.
//!
//! This oracle deliberately rebuilds its tiny semantic model from fixture facts. It does not
//! consult the production frontier, residual partition, constraint model, or solver state when
//! deriving expected choices. Production-only identifiers are used after derivation to translate
//! an observed active layout factor into the fixture's root/entry context.

use std::collections::BTreeSet;

use super::*;
use crate::native::NativeSolverState;
use crate::prepared::PreparedGraphBuilder;
use crate::tetrahedral::TetrahedralLigand;
use crate::traversal::ObservedBondProgress;

type State = NonStereoWriterState<NativeSolverState>;

const CENTER: usize = 0;
const A: usize = 1;
const B: usize = 2;
const D: usize = 3;
const CENTER_A: usize = 0;
const CENTER_B: usize = 1;
const A_B: usize = 2;
const CENTER_D: usize = 3;

const TRAVERSAL: u8 = 0;
const RING_00: u8 = 1;
const RING_10: u8 = 2;
const RING_01: u8 = 3;
const RING_11: u8 = 4;

const EDGES: [(usize, usize); 4] = [(CENTER, A), (CENTER, B), (A, B), (CENTER, D)];

#[derive(Clone)]
struct Fixture {
    surface: PreparedNonStereo,
    atoms: [AtomId; 4],
}

fn fixture() -> Fixture {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = std::array::from_fn(|_| graph.add_atom().unwrap());
    let bonds = [
        graph.add_bond(atoms[CENTER], atoms[A]).unwrap(),
        graph.add_bond(atoms[CENTER], atoms[B]).unwrap(),
        graph.add_bond(atoms[A], atoms[B]).unwrap(),
        graph.add_bond(atoms[CENTER], atoms[D]).unwrap(),
    ];
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        vec![
            PreparedAtomToken::Tetrahedral {
                reference_order: [
                    TetrahedralLigand::Bond(bonds[CENTER_A]),
                    TetrahedralLigand::Bond(bonds[CENTER_B]),
                    TetrahedralLigand::Bond(bonds[CENTER_D]),
                    TetrahedralLigand::VirtualHydrogen,
                ],
                text_by_parity: ["[C@H]".to_owned(), "[C@@H]".to_owned()],
            },
            PreparedAtomToken::Fixed("A".to_owned()),
            PreparedAtomToken::Fixed("B".to_owned()),
            PreparedAtomToken::Fixed("D".to_owned()),
        ],
        vec![
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
        ],
    )
    .unwrap();
    Fixture { surface, atoms }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Ligand {
    Bond(usize),
    Hydrogen,
}

const REFERENCE: [Ligand; 4] = [
    Ligand::Bond(CENTER_A),
    Ligand::Bond(CENTER_B),
    Ligand::Bond(CENTER_D),
    Ligand::Hydrogen,
];

#[derive(Clone, Debug, PartialEq, Eq)]
struct Assignment {
    plans: [u8; 4],
    order: u8,
    pattern: u8,
}

#[derive(Clone, Debug)]
struct ActivationFacts {
    context: Context,
    prefix: Vec<Ligand>,
    entry: Option<usize>,
    waiting_rings: Vec<usize>,
    attachment_groups: Vec<Vec<usize>>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Context {
    Root,
    Entry(usize),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Progress {
    Unrepresented,
    Traversed { from: usize, to: usize },
    RingOpen { first: usize },
    RingClosed { first: usize, second: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Pending {
    OpeningLabel {
        bond: usize,
        endpoint: usize,
        label: usize,
    },
    ClosureLabel {
        bond: usize,
        endpoint: usize,
        label: usize,
    },
}

#[derive(Clone, Debug)]
struct OracleState {
    assignments: Vec<Assignment>,
    active_atom: usize,
    entry_bond: Option<usize>,
    visited: Vec<usize>,
    emitted_bonds: Vec<usize>,
    ring_occurrence_count: usize,
    progress: Vec<Progress>,
    context: Context,
    labels: Vec<(usize, usize)>,
    pending: Option<Pending>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SemanticState {
    active_atom: Option<usize>,
    entry_bond: Option<usize>,
    visited: Vec<usize>,
    emitted_bonds: Vec<usize>,
    ring_occurrence_count: usize,
    progress: Vec<Progress>,
    plan_domains: Vec<Vec<u8>>,
    order_domain: Vec<u8>,
    pattern_domain: Vec<u8>,
    context: Context,
    labels: Vec<(usize, usize)>,
    pending: Option<Pending>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VisibleSuccessor {
    text: String,
    successor: SemanticState,
}

fn initial_assignments() -> Vec<Assignment> {
    let plan_domains = [
        vec![TRAVERSAL, RING_10, RING_01, RING_11],
        vec![TRAVERSAL, RING_00],
        vec![TRAVERSAL, RING_00],
        vec![TRAVERSAL, RING_00],
    ];
    let mut assignments = Vec::new();
    for &p0 in &plan_domains[0] {
        for &p1 in &plan_domains[1] {
            for &p2 in &plan_domains[2] {
                for &p3 in &plan_domains[3] {
                    let plans = [p0, p1, p2, p3];
                    if !is_spanning_tree(&plans) {
                        continue;
                    }
                    let pattern = role_pattern(&plans);
                    for order in 0..24 {
                        assignments.push(Assignment {
                            plans,
                            order,
                            pattern,
                        });
                    }
                }
            }
        }
    }
    assignments
}

fn is_spanning_tree(plans: &[u8; 4]) -> bool {
    let traversal_edges = (0..4)
        .filter(|bond| plans[*bond] == TRAVERSAL)
        .collect::<Vec<_>>();
    if traversal_edges.len() != 3 {
        return false;
    }
    let mut reached = [false; 4];
    reached[0] = true;
    loop {
        let mut changed = false;
        for &bond in &traversal_edges {
            let (left, right) = EDGES[bond];
            if reached[left] && !reached[right] {
                reached[right] = true;
                changed = true;
            }
            if reached[right] && !reached[left] {
                reached[left] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    reached.into_iter().all(|value| value)
}

fn role_pattern(plans: &[u8; 4]) -> u8 {
    [CENTER_A, CENTER_B, CENTER_D]
        .into_iter()
        .enumerate()
        .fold(0, |pattern, (bit, bond)| {
            pattern | (u8::from(plans[bond] != TRAVERSAL) << bit)
        })
}

fn permutation(value: u8) -> [usize; 4] {
    let mut permutations = Vec::new();
    for first in 0..4 {
        for second in 0..4 {
            if second == first {
                continue;
            }
            for third in 0..4 {
                if third == first || third == second {
                    continue;
                }
                let fourth = (0..4)
                    .find(|candidate| {
                        *candidate != first && *candidate != second && *candidate != third
                    })
                    .unwrap();
                permutations.push([first, second, third, fourth]);
            }
        }
    }
    permutations[value as usize]
}

fn order(value: u8) -> [Ligand; 4] {
    permutation(value).map(|index| REFERENCE[index])
}

fn even(value: u8) -> bool {
    let value = permutation(value);
    let inversions = (0..4)
        .flat_map(|left| ((left + 1)..4).map(move |right| (left, right)))
        .filter(|(left, right)| value[*left] > value[*right])
        .count();
    inversions % 2 == 0
}

fn role_is_ring(assignment: &Assignment, bond: usize) -> bool {
    assignment.plans[bond] != TRAVERSAL
}

fn activation_accepts(assignment: &Assignment, facts: &ActivationFacts) -> bool {
    if facts
        .entry
        .is_some_and(|bond| role_is_ring(assignment, bond))
    {
        return false;
    }
    if !facts
        .waiting_rings
        .iter()
        .all(|bond| role_is_ring(assignment, *bond))
    {
        return false;
    }
    if !facts.attachment_groups.iter().all(|group| {
        group
            .iter()
            .filter(|bond| !role_is_ring(assignment, **bond))
            .count()
            == 1
    }) {
        return false;
    }
    if assignment.pattern != role_pattern(&assignment.plans) {
        return false;
    }
    layout_accepts(assignment, &facts.prefix)
}

fn layout_accepts(assignment: &Assignment, prefix: &[Ligand]) -> bool {
    let order = order(assignment.order);
    if !order.starts_with(prefix) {
        return false;
    }
    if prefix
        .iter()
        .any(|ligand| matches!(ligand, Ligand::Bond(bond) if role_is_ring(assignment, *bond)))
    {
        return false;
    }
    let mut saw_traversal = false;
    for ligand in &order[prefix.len()..] {
        let Ligand::Bond(bond) = ligand else {
            return false;
        };
        let ring = role_is_ring(assignment, *bond);
        if ring && saw_traversal {
            return false;
        }
        saw_traversal |= !ring;
    }
    true
}

fn activate(
    assignments: &[Assignment],
    facts: &ActivationFacts,
    parity_even: bool,
) -> Vec<Assignment> {
    assignments
        .iter()
        .filter(|assignment| {
            activation_accepts(assignment, facts) && even(assignment.order) == parity_even
        })
        .cloned()
        .collect()
}

fn root_choices() -> Vec<VisibleSuccessor> {
    let facts = ActivationFacts {
        context: Context::Root,
        prefix: vec![Ligand::Hydrogen],
        entry: None,
        waiting_rings: Vec::new(),
        attachment_groups: vec![vec![CENTER_A, CENTER_B], vec![CENTER_D]],
    };
    [true, false]
        .into_iter()
        .filter_map(|parity_even| {
            let assignments = activate(&initial_assignments(), &facts, parity_even);
            (!assignments.is_empty()).then(|| VisibleSuccessor {
                text: if parity_even { "[C@H]" } else { "[C@@H]" }.to_owned(),
                successor: semantic(&OracleState {
                    assignments,
                    active_atom: CENTER,
                    entry_bond: None,
                    visited: vec![CENTER],
                    emitted_bonds: Vec::new(),
                    ring_occurrence_count: 0,
                    progress: vec![Progress::Unrepresented; 4],
                    context: facts.context,
                    labels: Vec::new(),
                    pending: None,
                }),
            })
        })
        .collect()
}

fn root_oracle_state(text: &str) -> OracleState {
    let facts = ActivationFacts {
        context: Context::Root,
        prefix: vec![Ligand::Hydrogen],
        entry: None,
        waiting_rings: Vec::new(),
        attachment_groups: vec![vec![CENTER_A, CENTER_B], vec![CENTER_D]],
    };
    let parity_even = match text {
        "[C@H]" => true,
        "[C@@H]" => false,
        _ => panic!("unexpected center token {text}"),
    };
    OracleState {
        assignments: activate(&initial_assignments(), &facts, parity_even),
        active_atom: CENTER,
        entry_bond: None,
        visited: vec![CENTER],
        emitted_bonds: Vec::new(),
        ring_occurrence_count: 0,
        progress: vec![Progress::Unrepresented; 4],
        context: facts.context,
        labels: Vec::new(),
        pending: None,
    }
}

#[derive(Copy, Clone)]
enum Endpoint {
    First,
}

#[derive(Copy, Clone)]
enum Spelling {
    Omit,
    Emit,
}

fn endpoint_support(plan: u8, endpoint: Endpoint, spelling: Spelling) -> bool {
    match (endpoint, spelling) {
        (Endpoint::First, Spelling::Omit) => matches!(plan, RING_00 | RING_01),
        (Endpoint::First, Spelling::Emit) => matches!(plan, RING_10 | RING_11),
    }
}

fn opening_choices(source: &OracleState) -> Vec<VisibleSuccessor> {
    let mut choices = Vec::new();
    for bond in [CENTER_A, CENTER_B] {
        for spelling in [Spelling::Omit, Spelling::Emit] {
            if bond == CENTER_B && matches!(spelling, Spelling::Emit) {
                continue;
            }
            let assignments = source
                .assignments
                .iter()
                .filter(|assignment| {
                    role_is_ring(assignment, bond)
                        && endpoint_support(assignment.plans[bond], Endpoint::First, spelling)
                        && order(assignment.order)
                            .starts_with(&[Ligand::Hydrogen, Ligand::Bond(bond)])
                })
                .cloned()
                .collect::<Vec<_>>();
            if assignments.is_empty() {
                continue;
            }
            let mut state = source.clone();
            state.assignments = assignments;
            state.emitted_bonds.push(bond);
            state.ring_occurrence_count += 1;
            state.progress[bond] = Progress::RingOpen { first: CENTER };
            state.labels.push((bond, 0));
            state.pending = matches!(spelling, Spelling::Emit).then_some(Pending::OpeningLabel {
                bond,
                endpoint: CENTER,
                label: 0,
            });
            choices.push(VisibleSuccessor {
                text: match spelling {
                    Spelling::Omit => "1".to_owned(),
                    Spelling::Emit => "=".to_owned(),
                },
                successor: semantic(&state),
            });
        }
    }
    choices
}

fn entered_center_choices() -> Vec<VisibleSuccessor> {
    let facts = ActivationFacts {
        context: Context::Entry(CENTER_B),
        prefix: vec![Ligand::Bond(CENTER_B), Ligand::Hydrogen],
        entry: Some(CENTER_B),
        waiting_rings: vec![CENTER_A],
        attachment_groups: vec![vec![CENTER_D]],
    };
    let base = initial_assignments()
        .into_iter()
        .filter(|assignment| {
            assignment.plans[CENTER_A] == RING_10
                && assignment.plans[CENTER_B] == TRAVERSAL
                && assignment.plans[A_B] == TRAVERSAL
                && assignment.plans[CENTER_D] == TRAVERSAL
        })
        .collect::<Vec<_>>();
    [true, false]
        .into_iter()
        .filter_map(|parity_even| {
            let assignments = activate(&base, &facts, parity_even);
            (!assignments.is_empty()).then(|| VisibleSuccessor {
                text: if parity_even { "[C@H]" } else { "[C@@H]" }.to_owned(),
                successor: semantic(&OracleState {
                    assignments,
                    active_atom: CENTER,
                    entry_bond: Some(CENTER_B),
                    visited: vec![CENTER, A, B],
                    emitted_bonds: Vec::new(),
                    ring_occurrence_count: 0,
                    progress: vec![
                        Progress::RingOpen { first: A },
                        Progress::Traversed {
                            from: B,
                            to: CENTER,
                        },
                        Progress::Traversed { from: A, to: B },
                        Progress::Unrepresented,
                    ],
                    context: facts.context,
                    labels: vec![(CENTER_A, 0)],
                    pending: None,
                }),
            })
        })
        .collect()
}

fn entered_oracle_state(text: &str) -> OracleState {
    entered_center_choices()
        .into_iter()
        .find(|choice| choice.text == text)
        .map(|_| {
            let facts = ActivationFacts {
                context: Context::Entry(CENTER_B),
                prefix: vec![Ligand::Bond(CENTER_B), Ligand::Hydrogen],
                entry: Some(CENTER_B),
                waiting_rings: vec![CENTER_A],
                attachment_groups: vec![vec![CENTER_D]],
            };
            let base = initial_assignments()
                .into_iter()
                .filter(|assignment| assignment.plans[CENTER_A] == RING_10)
                .collect::<Vec<_>>();
            let assignments = activate(&base, &facts, text == "[C@H]");
            OracleState {
                assignments,
                active_atom: CENTER,
                entry_bond: Some(CENTER_B),
                visited: vec![CENTER, A, B],
                emitted_bonds: Vec::new(),
                ring_occurrence_count: 0,
                progress: vec![
                    Progress::RingOpen { first: A },
                    Progress::Traversed {
                        from: B,
                        to: CENTER,
                    },
                    Progress::Traversed { from: A, to: B },
                    Progress::Unrepresented,
                ],
                context: facts.context,
                labels: vec![(CENTER_A, 0)],
                pending: None,
            }
        })
        .expect("the entered center has one parity-compatible atom token")
}

fn closure_choices(source: &OracleState) -> Vec<VisibleSuccessor> {
    let assignments = source
        .assignments
        .iter()
        .filter(|assignment| {
            endpoint_support(assignment.plans[CENTER_A], Endpoint::First, Spelling::Emit)
                && order(assignment.order).starts_with(&[
                    Ligand::Bond(CENTER_B),
                    Ligand::Hydrogen,
                    Ligand::Bond(CENTER_A),
                ])
        })
        .cloned()
        .collect::<Vec<_>>();
    assert!(!assignments.is_empty());
    let mut state = source.clone();
    state.assignments = assignments;
    state.emitted_bonds.push(CENTER_A);
    state.ring_occurrence_count += 1;
    state.progress[CENTER_A] = Progress::RingClosed {
        first: A,
        second: CENTER,
    };
    state.pending = Some(Pending::ClosureLabel {
        bond: CENTER_A,
        endpoint: CENTER,
        label: 0,
    });
    vec![VisibleSuccessor {
        text: "=".to_owned(),
        successor: semantic(&state),
    }]
}

fn semantic(state: &OracleState) -> SemanticState {
    assert!(!state.assignments.is_empty());
    SemanticState {
        active_atom: Some(state.active_atom),
        entry_bond: state.entry_bond,
        visited: state.visited.clone(),
        emitted_bonds: state.emitted_bonds.clone(),
        ring_occurrence_count: state.ring_occurrence_count,
        progress: state.progress.clone(),
        plan_domains: (0..4)
            .map(|bond| projected(&state.assignments, |assignment| assignment.plans[bond]))
            .collect(),
        order_domain: projected(&state.assignments, |assignment| assignment.order),
        pattern_domain: projected(&state.assignments, |assignment| assignment.pattern),
        context: state.context,
        labels: state.labels.clone(),
        pending: state.pending.clone(),
    }
}

fn projected(assignments: &[Assignment], value: impl Fn(&Assignment) -> u8) -> Vec<u8> {
    assignments
        .iter()
        .map(value)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn observe(fixture: &Fixture, state: &State) -> SemanticState {
    let observed = state.observe_raw();
    let frame = observed
        .structural
        .traversal
        .active_frame
        .as_ref()
        .expect("qualified successor must retain the active center frame");
    assert_eq!(frame.atom, fixture.atoms[CENTER]);
    let center = fixture
        .surface
        .tetrahedral_center(fixture.atoms[CENTER])
        .unwrap();
    let active_contexts = std::iter::once((center.root_layout_factor, Context::Root))
        .chain(
            center
                .entry_layout_factors
                .iter()
                .map(|(bond, factor)| (*factor, Context::Entry(bond.index()))),
        )
        .filter_map(|(factor, context)| {
            observed
                .structural
                .active_factors
                .contains(&factor)
                .then_some(context)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        active_contexts.len(),
        1,
        "exactly one center layout context must be active"
    );
    let domain_for_atom = |domains: &[(AtomId, Domain)]| {
        domains
            .iter()
            .find_map(|(atom, domain)| (*atom == fixture.atoms[CENTER]).then_some(*domain))
            .unwrap()
            .iter()
            .collect::<Vec<_>>()
    };
    SemanticState {
        active_atom: Some(frame.atom.index()),
        entry_bond: frame.entry_bond.map(|bond| bond.index()),
        visited: observed
            .structural
            .traversal
            .visited_atoms
            .iter()
            .map(|atom| atom.index())
            .collect(),
        emitted_bonds: frame
            .emitted_bonds
            .iter()
            .map(|bond| bond.index())
            .collect(),
        ring_occurrence_count: frame.ring_occurrence_count,
        progress: observed
            .structural
            .traversal
            .bond_progress
            .iter()
            .map(|progress| match progress {
                ObservedBondProgress::Unrepresented => Progress::Unrepresented,
                ObservedBondProgress::Traversed { from, to } => Progress::Traversed {
                    from: from.index(),
                    to: to.index(),
                },
                ObservedBondProgress::RingOpen { first_endpoint } => Progress::RingOpen {
                    first: first_endpoint.index(),
                },
                ObservedBondProgress::RingClosed {
                    first_endpoint,
                    second_endpoint,
                } => Progress::RingClosed {
                    first: first_endpoint.index(),
                    second: second_endpoint.index(),
                },
            })
            .collect(),
        plan_domains: observed
            .structural
            .bond_plan_domains
            .iter()
            .map(|domain| domain.iter().collect())
            .collect(),
        order_domain: domain_for_atom(&observed.tetrahedral_order_domains),
        pattern_domain: domain_for_atom(&observed.tetrahedral_role_pattern_domains),
        context: active_contexts[0],
        labels: observed
            .labels_by_bond
            .iter()
            .map(|(bond, label)| (bond.index(), *label))
            .collect(),
        pending: observed.pending.map(|pending| match pending {
            ObservedPending::RingOpeningLabel {
                bond,
                endpoint,
                label,
            } => Pending::OpeningLabel {
                bond: bond.index(),
                endpoint: endpoint.index(),
                label,
            },
            ObservedPending::RingClosureLabel {
                bond,
                endpoint,
                label,
            } => Pending::ClosureLabel {
                bond: bond.index(),
                endpoint: endpoint.index(),
                label,
            },
            other => panic!("unexpected pending syntax in tetrahedral oracle: {other:?}"),
        }),
    }
}

fn visible(fixture: &Fixture, choices: &[Choice<State>]) -> Vec<VisibleSuccessor> {
    let mut values = choices
        .iter()
        .map(|choice| VisibleSuccessor {
            text: choice.text().to_owned(),
            successor: observe(fixture, choice.successor()),
        })
        .collect::<Vec<_>>();
    values.sort();
    values
}

fn sorted(mut values: Vec<VisibleSuccessor>) -> Vec<VisibleSuccessor> {
    values.sort();
    values
}

fn initial(fixture: &Fixture) -> State {
    State::initial(&fixture.surface)
        .unwrap()
        .unwrap_consistent()
}

fn is_opening(fixture: &Fixture, source: &State, choice: &Choice<State>, endpoint: usize) -> bool {
    let source = source.observe_raw();
    let successor = choice.successor().observe_raw();
    (0..4).any(|bond| {
        matches!(
            (
                &source.structural.traversal.bond_progress[bond],
                &successor.structural.traversal.bond_progress[bond],
            ),
            (
                ObservedBondProgress::Unrepresented,
                ObservedBondProgress::RingOpen { first_endpoint },
            ) if *first_endpoint == fixture.atoms[endpoint]
        )
    })
}

#[test]
fn root_activation_and_ring_openings_match_primitive_one_step_oracle() {
    let fixture = fixture();
    let state = initial(&fixture);
    let source = state.observe_raw();
    let production_root_choices = state
        .choices()
        .unwrap()
        .into_iter()
        .filter(|choice| choice.successor().active_atom() == Some(fixture.atoms[CENTER]))
        .collect::<Vec<_>>();

    assert_eq!(
        visible(&fixture, &production_root_choices),
        sorted(root_choices())
    );
    assert_eq!(
        state.observe_raw(),
        source,
        "root choice derivation mutated its source"
    );

    for root in production_root_choices {
        let oracle_source = root_oracle_state(root.text());
        assert_eq!(
            observe(&fixture, root.successor()),
            semantic(&oracle_source)
        );
        let production_source = root.successor().clone();
        let source_observed = production_source.observe_raw();
        let openings = production_source
            .choices()
            .unwrap()
            .into_iter()
            .filter(|choice| is_opening(&fixture, &production_source, choice, CENTER))
            .collect::<Vec<_>>();

        assert_eq!(
            visible(&fixture, &openings),
            sorted(opening_choices(&oracle_source)),
            "root token {} has a different ring-opening relation",
            root.text()
        );
        assert_eq!(
            production_source.observe_raw(),
            source_observed,
            "ring-opening choice derivation mutated its source"
        );
    }
}

fn select_choice(
    state: State,
    predicate: impl Fn(&State, &Choice<State>) -> bool,
    description: &str,
) -> State {
    let source = state.observe_raw();
    let mut matches = state
        .choices()
        .unwrap()
        .into_iter()
        .filter(|choice| predicate(&state, choice))
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "expected one {description}");
    assert_eq!(
        state.observe_raw(),
        source,
        "selecting {description} mutated its source"
    );
    matches.remove(0).into_successor()
}

#[test]
fn entered_activation_and_explicit_closure_match_primitive_one_step_oracle() {
    let fixture = fixture();
    let rooted_at_a = select_choice(
        initial(&fixture),
        |_, choice| {
            choice.text() == "A" && choice.successor().active_atom() == Some(fixture.atoms[A])
        },
        "root at A",
    );
    let opened = select_choice(
        rooted_at_a,
        |source, choice| {
            choice.text() == "1"
                && matches!(
                    (
                        &source.observe_raw().structural.traversal.bond_progress[CENTER_A],
                        &choice.successor().observe_raw().structural.traversal.bond_progress
                            [CENTER_A],
                    ),
                    (
                        ObservedBondProgress::Unrepresented,
                        ObservedBondProgress::RingOpen { first_endpoint },
                    ) if *first_endpoint == fixture.atoms[A]
                )
        },
        "omitted opening of the explicit center-A bond",
    );
    let entered_b = select_choice(
        opened,
        |_, choice| {
            choice.text() == "B"
                && choice.successor().active_atom() == Some(fixture.atoms[B])
                && matches!(
                    choice.successor().observe_raw().structural.traversal.bond_progress[A_B],
                    ObservedBondProgress::Traversed { from, to }
                        if from == fixture.atoms[A] && to == fixture.atoms[B]
                )
        },
        "inline traversal from A to B",
    );
    let source = entered_b.observe_raw();
    let atom_choices = entered_b
        .choices()
        .unwrap()
        .into_iter()
        .filter(|choice| {
            choice.successor().active_atom() == Some(fixture.atoms[CENTER])
                && matches!(
                    choice.successor().observe_raw().structural.traversal.bond_progress
                        [CENTER_B],
                    ObservedBondProgress::Traversed { from, to }
                        if from == fixture.atoms[B] && to == fixture.atoms[CENTER]
                )
        })
        .collect::<Vec<_>>();

    assert_eq!(
        visible(&fixture, &atom_choices),
        sorted(entered_center_choices())
    );
    assert_eq!(
        entered_b.observe_raw(),
        source,
        "atom choices mutated their source"
    );

    for atom_choice in atom_choices {
        let oracle_source = entered_oracle_state(atom_choice.text());
        assert_eq!(
            observe(&fixture, atom_choice.successor()),
            semantic(&oracle_source)
        );
        let production_source = atom_choice.successor().clone();
        let source_observed = production_source.observe_raw();
        let closures = production_source
            .choices()
            .unwrap()
            .into_iter()
            .filter(|choice| {
                matches!(
                    (
                        &production_source.observe_raw().structural.traversal.bond_progress
                            [CENTER_A],
                        &choice.successor().observe_raw().structural.traversal.bond_progress
                            [CENTER_A],
                    ),
                    (
                        ObservedBondProgress::RingOpen { first_endpoint },
                        ObservedBondProgress::RingClosed {
                            first_endpoint: closed_first,
                            second_endpoint,
                        },
                    ) if *first_endpoint == fixture.atoms[A]
                        && *closed_first == fixture.atoms[A]
                        && *second_endpoint == fixture.atoms[CENTER]
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(
            visible(&fixture, &closures),
            sorted(closure_choices(&oracle_source)),
            "entered token {} has a different explicit closure relation",
            atom_choice.text()
        );
        assert_eq!(
            production_source.observe_raw(),
            source_observed,
            "ring-closure choice derivation mutated its source"
        );
    }
}
