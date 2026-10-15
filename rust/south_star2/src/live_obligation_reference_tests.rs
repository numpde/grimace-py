//! Finite raw event grammar for testing productive labelled transitions.
//!
//! This deliberately does not call the writer or EventSchedule. It searches
//! complete futures only in the test, after constructing locally legal states.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Role { Tree, Ring }

#[derive(Clone, Debug)]
struct Graph {
    atoms: &'static [&'static str],
    edges: Vec<(usize, usize, Role)>,
    root: usize,
    labels: usize,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum BondState { Unwritten, Tree, Open { at: usize, label: usize }, Closed }

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct State {
    visited: u16,
    active: usize,
    returns: Vec<usize>,
    bonds: Vec<BondState>,
    started_child: u16,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Event { Open { bond: usize, label: usize }, Close { bond: usize, label: usize },
    Branch { bond: usize }, Inline { bond: usize }, DotBranch { atom: usize },
    DotInline { atom: usize }, Return }

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Step { event: Event, text: String, next: State }

impl Graph {
    fn initial(&self) -> State {
        State { visited: 1 << self.root, active: self.root, returns: Vec::new(),
            bonds: vec![BondState::Unwritten; self.edges.len()], started_child: 0 }
    }

    fn accepted(&self, state: &State) -> bool {
        state.visited == (1 << self.atoms.len()) - 1 && state.returns.is_empty()
            && state.bonds.iter().all(|bond| matches!(bond, BondState::Tree | BondState::Closed))
    }

    fn raw_steps(&self, state: &State) -> Vec<Step> {
        self.raw_steps_with_dots(state, false)
    }

    fn raw_steps_with_dots(&self, state: &State, dots: bool) -> Vec<Step> {
        if self.accepted(state) { return Vec::new(); }
        let mut steps = Vec::new();
        let mut pending_ring = false;
        let used = state.bonds.iter().filter_map(|bond| match bond {
            BondState::Open { label, .. } => Some(*label), _ => None,
        }).collect::<BTreeSet<_>>();
        if state.started_child & (1 << state.active) == 0 {
            for (id, &(a, b, role)) in self.edges.iter().enumerate() {
                if role != Role::Ring || a != state.active && b != state.active { continue; }
                let other = if a == state.active { b } else { a };
                match state.bonds[id] {
                    BondState::Unwritten => {
                        pending_ring = true;
                        if state.visited & (1 << other) != 0 { continue; }
                        for label in 1..=self.labels {
                            if used.contains(&label) { continue; }
                            // Labels are anonymous resources for this structural
                            // experiment. The least free slot represents every
                            // capacity-equivalent spelling of this opening.
                            if dots && (1..label).any(|slot| !used.contains(&slot)) { continue; }
                            let mut next = state.clone();
                            next.bonds[id] = BondState::Open { at: state.active, label };
                            for text in [format!("{}", label - 1), format!("%{:02}", label - 1),
                                format!("-{}", label - 1), format!("-%{:02}", label - 1)] {
                                steps.push(Step { event: Event::Open { bond: id, label }, text, next: next.clone() });
                            }
                        }
                    }
                    BondState::Open { at, label } if at == other => {
                        pending_ring = true;
                        let mut next = state.clone();
                        next.bonds[id] = BondState::Closed;
                        for text in [format!("{}", label - 1), format!("%{:02}", label - 1),
                            format!("-{}", label - 1), format!("-%{:02}", label - 1)] {
                            steps.push(Step { event: Event::Close { bond: id, label }, text, next: next.clone() });
                        }
                    }
                    BondState::Open { at, .. } if at == state.active => {}
                    BondState::Closed => {}
                    _ => unreachable!(),
                }
            }
        }
        // The older diagnostic mode fixes every locally pending ring before a
        // child. The dot experiment also retains grammar-legal dead prefixes
        // that leave such an incidence unwritten.
        if pending_ring && !dots { return steps; }
        let children = self.edges.iter().enumerate().filter_map(|(id, &(a, b, role))| {
            if role != Role::Tree || state.bonds[id] != BondState::Unwritten { return None; }
            let other = if a == state.active { b } else if b == state.active { a } else { return None; };
            (state.visited & (1 << other) == 0).then_some((id, other))
        }).collect::<Vec<_>>();
        for (id, child) in children.iter().copied() {
            let mut next = state.clone();
            next.bonds[id] = BondState::Tree;
            next.visited |= 1 << child;
            next.started_child |= 1 << state.active;
            next.active = child;
            if dots {
                let mut branch = next.clone();
                branch.returns.push(state.active);
                steps.push(Step { event: Event::Branch { bond: id },
                    text: format!("({}", self.atoms[child]), next: branch });
                steps.push(Step { event: Event::Inline { bond: id },
                    text: self.atoms[child].to_owned(), next });
                continue;
            }
            let (event, text) = if children.len() > 1 {
                next.returns.push(state.active);
                (Event::Branch { bond: id }, format!("({}", self.atoms[child]))
            } else { (Event::Inline { bond: id }, self.atoms[child].to_owned()) };
            steps.push(Step { event, text, next });
        }
        if dots {
            // OpenSMILES `chain dot branched_atom` and `branch ::= '(' dot chain ')'`.
            for atom in 0..self.atoms.len() {
                if state.visited & (1 << atom) != 0 { continue; }
                let mut next = state.clone();
                next.visited |= 1 << atom;
                next.started_child |= 1 << state.active;
                next.active = atom;
                steps.push(Step { event: Event::DotInline { atom },
                    text: format!(".{}", self.atoms[atom]), next: next.clone() });
                next.returns.push(state.active);
                steps.push(Step { event: Event::DotBranch { atom },
                    text: format!("(.{}", self.atoms[atom]), next });
            }
        }
        if (children.is_empty() || dots) && !state.returns.is_empty() {
            let mut next = state.clone();
            next.active = next.returns.pop().unwrap();
            steps.push(Step { event: Event::Return, text: ")".to_owned(), next });
        }
        steps
    }
}

struct Reference {
    states: Vec<State>,
    transitions: Vec<Vec<(Event, String, usize)>>,
    productive: BTreeSet<usize>,
}

impl Reference {
    fn build(graph: &Graph) -> Self {
        Self::build_with_dots(graph, false)
    }

    fn build_with_dots(graph: &Graph, dots: bool) -> Self {
        Self::build_from_state(graph, dots, graph.initial())
    }

    fn build_from_state(graph: &Graph, dots: bool, initial: State) -> Self {
        assert!(graph.atoms.len() <= 8 && graph.edges.len() <= 12
            && graph.labels <= if dots { 6 } else { 2 });
        let mut states = vec![initial];
        let mut ids = BTreeMap::from([(states[0].clone(), 0)]);
        let mut transitions = Vec::new();
        let mut queue = VecDeque::from([0]);
        while let Some(id) = queue.pop_front() {
            let mut outgoing = Vec::new();
            for step in graph.raw_steps_with_dots(&states[id], dots) {
                let next = if let Some(&known) = ids.get(&step.next) { known } else {
                    if !dots {
                        assert!(states.len() < 20_000,
                            "raw grammar state budget exhausted: graph={graph:?}");
                    }
                    let fresh = states.len();
                    ids.insert(step.next.clone(), fresh);
                    states.push(step.next);
                    queue.push_back(fresh);
                    fresh
                };
                outgoing.push((step.event, step.text, next));
            }
            assert_eq!(id, transitions.len());
            transitions.push(outgoing);
        }
        let mut productive = states.iter().enumerate().filter_map(|(id, state)|
            graph.accepted(state).then_some(id)).collect::<BTreeSet<_>>();
        loop {
            let before = productive.len();
            for (id, edges) in transitions.iter().enumerate() {
                if edges.iter().any(|(_, _, next)| productive.contains(next)) {
                    productive.insert(id);
                }
            }
            if productive.len() == before { break; }
        }
        Self { states, transitions, productive }
    }

    fn choice_set(&self, id: usize) -> BTreeSet<(String, State)> {
        self.transitions[id].iter().filter(|(_, _, next)| self.productive.contains(next))
            .map(|(_, text, next)| (text.clone(), self.states[*next].clone())).collect()
    }

    fn raw_set(&self, id: usize) -> BTreeSet<(String, State)> {
        self.transitions[id].iter().map(|(_, text, next)|
            (text.clone(), self.states[*next].clone())).collect()
    }
}

// The reference keeps every physical return for grammar. Only the feasibility
// key drops a saved frame whose incidences are already represented. The held
// ring bond and its physical label remain in `bonds` after its owner is dropped.
fn feasibility_projection(graph: &Graph, state: &State) -> State {
    let mut projection = state.clone();
    projection.returns.retain(|&atom| graph.edges.iter().enumerate().any(
        |(id, &(a, b, _))| (a == atom || b == atom)
            && (state.bonds[id] == BondState::Unwritten
                || matches!(state.bonds[id], BondState::Open { at, .. } if at != atom))));
    projection
}

fn shortest_completion(reference: &Reference, from: usize) -> Option<String> {
    let mut queue = VecDeque::from([(from, String::new())]);
    let mut seen = BTreeSet::from([from]);
    while let Some((id, text)) = queue.pop_front() {
        if reference.transitions[id].is_empty() && reference.productive.contains(&id) {
            return Some(text);
        }
        for (_, spelling, next) in &reference.transitions[id] {
            if seen.insert(*next) {
                queue.push_back((*next, format!("{text}{spelling}")));
            }
        }
    }
    None
}

fn raw_prefix(graph: &Graph, events: &[Event]) -> State {
    let mut state = graph.initial();
    for event in events {
        state = graph.raw_steps_with_dots(&state, true).into_iter()
            .find(|step| &step.event == event)
            .unwrap_or_else(|| panic!("raw prefix event {event:?} unavailable after {state:?}"))
            .next;
    }
    state
}

#[test]
fn unrepresented_traversal_owner_defeats_unconditional_frame_forgetting() {
    let graph = Graph { atoms: &["C", "C", "C"],
        edges: vec![(0, 1, Role::Tree), (0, 2, Role::Tree), (1, 2, Role::Ring)],
        root: 0, labels: 1 };
    let saved = raw_prefix(&graph, &[Event::Branch { bond: 0 },
        Event::Open { bond: 2, label: 1 }]);
    let lost = raw_prefix(&graph, &[Event::Inline { bond: 0 },
        Event::Open { bond: 2, label: 1 }]);
    let saved_reference = Reference::build_from_state(&graph, true, saved.clone());
    let lost_reference = Reference::build_from_state(&graph, true, lost.clone());
    assert!(saved_reference.productive.contains(&0));
    assert!(!lost_reference.productive.contains(&0));
    assert_ne!(feasibility_projection(&graph, &saved),
        feasibility_projection(&graph, &lost));
    let mut unsafe_saved = saved;
    unsafe_saved.returns.clear();
    assert_eq!(unsafe_saved, lost,
        "dropping an owner with an unwritten Traversal incidence would merge them");
}

#[test]
fn ordered_saved_owners_distinguish_two_label_completions() {
    let edges = vec![(0, 3, Role::Tree), (1, 4, Role::Tree),
        (1, 3, Role::Ring), (2, 3, Role::Ring), (3, 4, Role::Ring)];
    let first_graph = Graph { atoms: &["C", "C", "C", "C", "C"],
        edges: edges.clone(), root: 1, labels: 2 };
    let second_graph = Graph { root: 0, ..first_graph.clone() };
    let t_first = raw_prefix(&first_graph, &[
        Event::Open { bond: 2, label: 1 }, Event::DotBranch { atom: 0 },
        Event::DotBranch { atom: 2 }, Event::Open { bond: 3, label: 2 }]);
    let u_first = raw_prefix(&second_graph, &[
        Event::DotBranch { atom: 1 }, Event::Open { bond: 2, label: 1 },
        Event::DotBranch { atom: 2 }, Event::Open { bond: 3, label: 2 }]);
    let t_reference = Reference::build_from_state(&first_graph, true, t_first.clone());
    let u_reference = Reference::build_from_state(&second_graph, true, u_first.clone());
    assert!(t_reference.productive.contains(&0),
        "T-first order should release a label before opening T-U");
    assert!(!u_reference.productive.contains(&0),
        "U-first order needs a third simultaneous ring label");
    let ordered_t = feasibility_projection(&first_graph, &t_first);
    let ordered_u = feasibility_projection(&second_graph, &u_first);
    assert_ne!(ordered_t, ordered_u);
    let mut unordered_t = ordered_t;
    let mut unordered_u = ordered_u;
    unordered_t.returns.sort_unstable();
    unordered_u.returns.sort_unstable();
    assert_eq!(unordered_t, unordered_u,
        "an owner set would merge prefixes with different feasibility");
}

fn check_exhausted_saved_frames_through(max_atoms: usize) {
    // Every labelled simple graph through max_atoms: each possible bond is
    // absent, Tree, or Ring; root 0; capacities 0..=number of Ring bonds.
    // This is a finite falsification experiment, not a stereo or general proof.
    let mut cases = 0;
    let mut deleted = 0;
    let mut compared = 0;
    let mut single_deletion_pairs = 0;
    for n in 1..=max_atoms {
        let pairs = (0..n).flat_map(|a| ((a + 1)..n).map(move |b| (a, b)))
            .collect::<Vec<_>>();
        let assignments = 3usize.pow(pairs.len() as u32);
        for assignment in 0..assignments {
            let mut code = assignment;
            let mut edges = Vec::new();
            for &(a, b) in &pairs {
                match code % 3 {
                    1 => edges.push((a, b, Role::Tree)),
                    2 => edges.push((a, b, Role::Ring)),
                    _ => {}
                }
                code /= 3;
            }
            let ring_bonds = edges.iter().filter(|edge| edge.2 == Role::Ring).count();
            for labels in 0..=ring_bonds {
                let graph = Graph { atoms: &["C", "C", "C", "C"][..n],
                    edges: edges.clone(), root: 0, labels };
                let reference = Reference::build_with_dots(&graph, true);
                let mut groups = BTreeMap::<State, Vec<usize>>::new();
                for (id, state) in reference.states.iter().enumerate() {
                    let key = feasibility_projection(&graph, state);
                    deleted += state.returns.len() - key.returns.len();
                    groups.entry(key).or_default().push(id);
                }
                for (key, ids) in groups {
                    if ids.len() < 2 { continue; }
                    compared += 1;
                    let first = ids[0];
                    for &other in &ids[1..] {
                        let left = &reference.states[first].returns;
                        let right = &reference.states[other].returns;
                        let (long, short) = if left.len() >= right.len() {
                            (left, right)
                        } else { (right, left) };
                        if long.len() == short.len() + 1 && (0..long.len()).any(|i| {
                            let mut without = long.clone();
                            without.remove(i);
                            &without == short
                        }) {
                            single_deletion_pairs += 1;
                        }
                        assert_eq!(reference.productive.contains(&first),
                            reference.productive.contains(&other),
                            "frame-forget counterexample: n={n} edges={edges:?} labels={labels} \
                             projection={key:?} left={:?} right={:?} shortest-left={:?} shortest-right={:?}",
                            reference.states[first], reference.states[other],
                            shortest_completion(&reference, first),
                            shortest_completion(&reference, other));
                    }
                }
                cases += 1;
            }
        }
    }
    assert!(deleted > 0 && compared > 0 && single_deletion_pairs > 0,
        "projection must forget and compare one-frame deletions: \
         {cases} cases, {deleted} dropped frames, {compared} groups, \
         {single_deletion_pairs} direct pairs");
}

#[test]
fn exhausted_saved_frames_preserve_bounded_dot_grammar_feasibility() {
    // 59 graph-capacity cases: all fixed-role simple graphs on 1..=3 atoms.
    check_exhausted_saved_frames_through(3);
}

#[test]
#[ignore = "2,246-case matrix exceeded the 120-second bounded diagnostic budget"]
fn exhausted_saved_frames_four_atom_diagnostic() {
    check_exhausted_saved_frames_through(4);
}

#[test]
fn raw_grammar_exposes_productive_paths_and_legal_dead_ends() {
    let graphs = [
        Graph { atoms: &["C", "N", "O"], edges: vec![(0, 1, Role::Tree), (1, 2, Role::Tree)], root: 0, labels: 1 },
        Graph { atoms: &["C", "N", "O"], edges: vec![(0, 1, Role::Tree), (1, 2, Role::Tree), (0, 2, Role::Ring)], root: 0, labels: 1 },
        Graph { atoms: &["C", "N", "O", "F", "Cl"], edges: vec![
            (0, 1, Role::Tree), (1, 2, Role::Tree), (1, 3, Role::Tree),
            (0, 4, Role::Tree)], root: 0, labels: 1 },
        // A-X-B is a long tree arm. Both B-C and B-D need a ring label.
        Graph { atoms: &["C", "N", "O", "F", "Cl"], edges: vec![
            (0, 1, Role::Tree), (1, 2, Role::Tree), (0, 3, Role::Tree),
            (0, 4, Role::Tree), (2, 3, Role::Ring), (2, 4, Role::Ring)], root: 0, labels: 1 },
        Graph { atoms: &["C", "N", "O", "F", "Cl"], edges: vec![
            (0, 1, Role::Tree), (1, 2, Role::Tree), (0, 3, Role::Tree),
            (0, 4, Role::Tree), (2, 3, Role::Ring), (2, 4, Role::Ring)], root: 0, labels: 2 },
        Graph { atoms: &["C", "N", "O", "F", "Cl"], edges: vec![
            (0, 1, Role::Tree), (1, 2, Role::Ring), (0, 3, Role::Tree),
            (0, 4, Role::Tree), (2, 3, Role::Tree), (2, 4, Role::Ring)], root: 0, labels: 1 },
    ];
    for graph in &graphs {
        let reference = Reference::build(graph);
        assert!(reference.productive.contains(&0));
        if graph.edges.len() == 4 {
            assert!(reference.productive.iter().any(|id|
                reference.states[*id].returns.len() == 2),
                "a productive path must preserve two ordered branch returns");
        }
        if graph.edges.len() == 6 && graph.labels == 1 && graph.edges[1].2 == Role::Tree {
            let first = reference.choice_set(0);
            let branch_to = |bond: usize, atom: usize| {
                let mut next = graph.initial();
                next.visited |= 1 << atom;
                next.active = atom;
                next.returns.push(0);
                next.started_child |= 1;
                next.bonds[bond] = BondState::Tree;
                (format!("({}", graph.atoms[atom]), next)
            };
            assert_eq!(first, BTreeSet::from([branch_to(2, 3), branch_to(3, 4)]));
            assert!(reference.raw_set(0).iter().any(|(text, _)| text == "(N"));
            let x = reference.transitions[0].iter().find_map(|(event, _, next)|
                (*event == Event::Branch { bond: 0 }).then_some(*next)).unwrap();
            let mut frontier = vec![x];
            let mut seen = BTreeSet::new();
            let mut dead_end = false;
            while let Some(id) = frontier.pop() {
                if !seen.insert(id) { continue; }
                if reference.transitions[id].is_empty() && !graph.accepted(&reference.states[id]) {
                    dead_end = true;
                }
                frontier.extend(reference.transitions[id].iter().map(|(_, _, next)| *next));
            }
            assert!(dead_end, "locally legal X-first must reach a raw incomplete dead end");
            let mut paths = vec![(0, 0_u8)];
            let mut seen_paths = BTreeSet::new();
            let mut reused = false;
            while let Some((id, opens)) = paths.pop() {
                if !seen_paths.insert((id, opens)) { continue; }
                if graph.accepted(&reference.states[id]) && opens == 2 { reused = true; }
                for (event, _, next) in &reference.transitions[id] {
                    if reference.productive.contains(next) {
                        paths.push((*next, opens + u8::from(matches!(event, Event::Open { label: 1, .. }))));
                    }
                }
            }
            assert!(reused, "a productive one-label path must release and reuse label 1");
        } else if graph.edges.len() == 6 {
            assert!(reference.choice_set(0).iter().any(|(text, _)| text == "(N"));
        }
    }
}

#[test]
fn held_target_in_the_dead_branch_component_still_requires_its_saved_owner() {
    // P,A,H,B,C,D: A owns the sole unrepresented entry to held H. H is
    // connected to the B-C-D cycle through D-H, but that path cannot release
    // P-H before the cycle needs a label.
    let graph = Graph { atoms: &["P", "A", "H", "B", "C", "D"], edges: vec![
        (0, 1, Role::Tree), (0, 2, Role::Ring), (1, 2, Role::Tree),
        (1, 3, Role::Tree), (3, 4, Role::Tree), (4, 5, Role::Ring),
        (5, 3, Role::Tree), (5, 2, Role::Ring)], root: 0, labels: 1 };
    let reference = Reference::build(&graph);
    let opened = reference.transitions[0].iter().find_map(|(event, _, next)|
        (*event == Event::Open { bond: 1, label: 1 }).then_some(*next)).unwrap();
    let at_a = reference.transitions[opened].iter().find_map(|(event, _, next)|
        (*event == Event::Inline { bond: 0 }).then_some(*next)).unwrap();
    let raw = reference.raw_set(at_a);
    assert!(raw.iter().any(|(text, _)| text == "(B"));
    assert_eq!(reference.choice_set(at_a).iter().map(|(text, _)| text.as_str())
        .collect::<Vec<_>>(), vec!["(H"]);
    let h = reference.transitions[at_a].iter().find_map(|(event, _, next)|
        (*event == Event::Branch { bond: 2 }).then_some(*next)).unwrap();
    let closed = reference.transitions[h].iter().find_map(|(event, _, next)|
        (*event == Event::Close { bond: 1, label: 1 }).then_some(*next)).unwrap();
    let opened_hd = reference.transitions[closed].iter().find_map(|(event, _, next)|
        (*event == Event::Open { bond: 7, label: 1 }).then_some(*next)).unwrap();
    assert!(reference.productive.contains(&opened_hd));
}

#[test]
fn committed_triangle_traversal_keeps_its_saved_parent_available() {
    use super::*;
    use crate::native::NativeSolverState;
    use crate::prepared::PreparedGraphBuilder;
    use crate::traversal::ObservedBondProgress;

    // A-B and A-T are committed traversals; B-T is a ring. Dot syntax may
    // introduce T, but it cannot make a second occurrence of A represent A-T.
    let graph = Graph { atoms: &["C", "C", "C"], edges: vec![
        (0, 1, Role::Tree), (1, 2, Role::Ring), (0, 2, Role::Tree),
    ], root: 0, labels: 1 };
    let mut builder = PreparedGraphBuilder::new();
    let atoms = (0..3).map(|_| builder.add_atom().unwrap()).collect::<Vec<_>>();
    let bonds = graph.edges.iter().map(|&(a, b, _)|
        builder.add_bond(atoms[a], atoms[b]).unwrap()).collect::<Vec<_>>();
    let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(builder.build()),
        vec![PreparedAtomToken::Fixed("C".to_owned()); 3],
        vec![NonStereoBondToken::Elided; 3], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, 0,
    ).unwrap();
    let mut state = NonStereoWriterState::<NativeSolverState>::initial(&surface)
        .unwrap().unwrap_consistent();
    let restrictions = bonds.iter().zip(&graph.edges).map(|(bond, &(_, _, role))| {
        let partition = surface.molecule().bond_role_partition(*bond).unwrap();
        (surface.molecule().bond_decision_variable(*bond).unwrap(),
            match role { Role::Tree => partition.traversal_values(),
                Role::Ring => partition.ring_values() })
    }).collect::<Vec<_>>();
    state.structural = state.structural.transitioned_semantics(&restrictions, &[])
        .unwrap().unwrap_consistent();
    let (root_text, a) = super::tests::only_root_atom(&state, atoms[0]);
    let source = a.observe_raw();
    assert_eq!(source.structural.traversal.visited_atoms, vec![atoms[0]]);
    assert_eq!(source.structural.traversal.bond_progress,
        vec![ObservedBondProgress::Unrepresented; 3]);
    let choices = a.choices().unwrap();
    assert!(!choices.iter().any(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::InlineAtom(incident)) if incident.bond() == bonds[0])),
        "inline B would discard A, the only visited endpoint of committed A-T");
    let branch_choice = choices.into_iter().find(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::BranchTraversalEmission(incident)) if incident.bond() == bonds[0]))
        .expect("A can save a return before entering B");
    assert_eq!(branch_choice.text(), "(");
    let branch_text = branch_choice.text().to_owned();
    let branch = branch_choice.into_successor();
    let b_states = super::tests::completed_atom_witnesses(&branch, atoms[1]);
    assert_eq!(b_states.len(), 1);
    let (b_text, b) = b_states.into_iter().next().unwrap();
    assert_eq!(b_text, "C");
    let observed_b = b.observe_raw();
    assert!(std::ptr::eq(a.surface.molecule().graph(), b.surface.molecule().graph()));
    assert_eq!(source.structural.bond_plan_domains, observed_b.structural.bond_plan_domains);
    assert_eq!(observed_b.structural.traversal.visited_atoms, vec![atoms[0], atoms[1]]);
    assert_eq!(observed_b.structural.traversal.bond_progress[0],
        ObservedBondProgress::Traversed { from: atoms[0], to: atoms[1] });
    assert_eq!(observed_b.structural.traversal.branch_returns.len(), 1);
    assert_eq!(observed_b.structural.traversal.branch_returns[0].atom, atoms[0]);
    let opened_choice = b.choices().unwrap().into_iter().find(|choice|
        choice.text() == "0" && choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[1]))
        .expect("B opens the committed B-T ring");
    let opened_text = opened_choice.text().to_owned();
    let opened = opened_choice.into_successor();
    let observed_open = opened.observe_raw();
    assert!(std::ptr::eq(b.surface.molecule().graph(), opened.surface.molecule().graph()));
    assert_eq!(observed_open.structural.traversal.bond_progress[1],
        ObservedBondProgress::RingOpen { first_endpoint: atoms[1] });
    assert_eq!(observed_open.structural.traversal.bond_progress[2],
        ObservedBondProgress::Unrepresented);
    let returned_choice = opened.choices().unwrap().into_iter().find(|choice| choice.text() == ")")
        .expect("saved A remains available");
    let returned_text = returned_choice.text().to_owned();
    let returned = returned_choice.into_successor();
    let inline_t_choice = returned.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::InlineAtom(incident)) if incident.bond() == bonds[2]))
        .expect("A enters T through its committed traversal");
    assert_eq!(inline_t_choice.text(), "-");
    let inline_t_text = inline_t_choice.text().to_owned();
    let inline_t = inline_t_choice.into_successor();
    let t_states = super::tests::completed_atom_witnesses(&inline_t, atoms[2]);
    assert_eq!(t_states.len(), 1);
    let (t_text, t) = t_states.into_iter().next().unwrap();
    assert_eq!(t_text, "C");
    let closed_choice = t.choices().unwrap().into_iter().find(|choice|
        choice.text() == "0" && choice.successor().labels.is_clean())
        .expect("T closes its ring to B");
    assert_eq!(format!("{root_text}{branch_text}{b_text}{opened_text}{returned_text}{inline_t_text}{t_text}{}",
        closed_choice.text()), "C(C0)-C0");
    let closed = closed_choice.into_successor();
    assert!(closed.is_accepted(), "the selected explicit-bond triangle path must finish");

    // CC0 is a legal raw prefix with B active and no saved A. Its only
    // unvisited atom is T; a dot-started T cannot traverse the committed A-T
    // bond because A is neither active nor saved, and A cannot be reintroduced.
    let mut inline = graph.initial();
    inline.visited |= 1 << 1;
    inline.active = 1;
    inline.started_child |= 1;
    inline.bonds[0] = BondState::Tree;
    let raw_open = graph.raw_steps(&inline).into_iter().find(|step|
        step.event == Event::Open { bond: 1, label: 1 })
        .expect("CC0 is grammar-legal");
    assert_eq!(raw_open.next.returns, Vec::<usize>::new());
    assert_eq!(raw_open.next.bonds[2], BondState::Unwritten);
    assert_eq!(raw_open.next.visited, 0b011);
}

#[test]
fn one_label_diamond_production_first_branches_match_reference_successors() {
    use super::*;
    use crate::native::NativeSolverState;
    use crate::prepared::PreparedGraphBuilder;
    use crate::traversal::ObservedBondProgress;

    let graph = Graph { atoms: &["C", "N", "O", "F", "Cl"], edges: vec![
        (0, 1, Role::Tree), (1, 2, Role::Tree), (0, 3, Role::Tree),
        (0, 4, Role::Tree), (2, 3, Role::Ring), (2, 4, Role::Ring)], root: 0, labels: 1 };
    let reference = Reference::build(&graph);
    assert!(reference.raw_set(0).iter().any(|(text, _)| text == "(N"));
    let mut builder = PreparedGraphBuilder::new();
    let atoms = (0..graph.atoms.len()).map(|_| builder.add_atom().unwrap()).collect::<Vec<_>>();
    let bonds = graph.edges.iter().map(|&(a, b, _)| builder.add_bond(atoms[a], atoms[b]).unwrap()).collect::<Vec<_>>();
    let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(builder.build()),
        graph.atoms.iter().map(|text| PreparedAtomToken::Fixed((*text).to_owned())).collect(),
        vec![NonStereoBondToken::Elided; bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, 0,
    ).unwrap();
    let mut state = NonStereoWriterState::<NativeSolverState>::initial(&surface)
        .unwrap().unwrap_consistent();
    let restrictions = bonds.iter().zip(&graph.edges).map(|(bond, &(_, _, role))| {
        let partition = surface.molecule().bond_role_partition(*bond).unwrap();
        let domain = match role { Role::Tree => partition.traversal_values(),
            Role::Ring => partition.ring_values() };
        (surface.molecule().bond_decision_variable(*bond).unwrap(), domain)
    }).collect::<Vec<_>>();
    state.structural = state.structural.transitioned_semantics(&restrictions, &[])
        .unwrap().unwrap_consistent();
    let (_, rooted) = super::tests::only_root_atom(&state, atoms[0]);
    assert_eq!(rooted.active_atom(), Some(atoms[0]));
    let mut actual = BTreeSet::new();
    for branch in rooted.choices().unwrap() {
        let Some(PendingEmission::BranchTraversalEmission(incident)) = &branch.successor().pending else { continue; };
        let child = incident.atom().index();
        let branch_text = branch.text().to_owned();
        let opened = branch.into_successor();
        for (atom_text, emitted) in super::tests::completed_atom_witnesses(&opened, atoms[child]) {
            let text = format!("{branch_text}{atom_text}");
            assert_eq!(text, format!("({}", graph.atoms[child]));
            let observed = emitted.observe_raw();
            assert_eq!(observed.pending, None);
            assert!(!emitted.is_spelling_atom());
            assert_eq!(observed.structural.traversal.visited_atoms.len(), 2);
            let active = observed.structural.traversal.active_frame.as_ref().unwrap();
            let mut projected = graph.initial();
            projected.visited = observed.structural.traversal.visited_atoms.iter()
                .fold(0_u16, |bits, atom| bits | (1 << atom.index()));
            projected.active = active.atom.index();
            projected.returns = observed.structural.traversal.branch_returns.iter().rev()
                .map(|frame| frame.atom.index()).collect();
            projected.started_child = observed.structural.traversal.branch_returns.iter()
                .chain([active].into_iter())
                .filter(|frame| frame.branch_started)
                .fold(0_u16, |bits, frame| bits | (1 << frame.atom.index()));
            for (id, bond) in observed.structural.traversal.bond_progress.iter().enumerate() {
                projected.bonds[id] = match bond {
                    ObservedBondProgress::Unrepresented => BondState::Unwritten,
                    ObservedBondProgress::Traversed { .. } => BondState::Tree,
                    ObservedBondProgress::RingOpen { first_endpoint } => BondState::Open {
                        at: first_endpoint.index(),
                        label: observed.labels_by_bond.iter().find(|(candidate, _)| *candidate == bonds[id])
                            .expect("open ring owns a label").1 + 1,
                    },
                    ObservedBondProgress::RingClosed { .. } => BondState::Closed,
                };
            }
            assert_eq!(observed.labels_by_bond.len(), 0);
            assert_eq!(observed.structural.traversal.branch_returns.len(), 1);
            assert_eq!(active.atom.index(), child);
            assert!(actual.insert((text, projected)),
                "distinct production branches must retain distinct structural successors");
        }
    }
    assert_eq!(actual, reference.choice_set(0));
}

#[test]
fn unseeded_x_branch_preserves_ring_role_and_rejects_traversal_role() {
    use super::*;
    use crate::native::NativeSolverState;
    use crate::prepared::PreparedGraphBuilder;

    let edges = [(0, 1), (1, 2), (0, 3), (0, 4), (2, 3), (2, 4)];
    let mut builder = PreparedGraphBuilder::new();
    let atoms = (0..5).map(|_| builder.add_atom().unwrap()).collect::<Vec<_>>();
    let bonds = edges.map(|(a, b)| builder.add_bond(atoms[a], atoms[b]).unwrap());
    let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(builder.build()),
        ["C", "N", "O", "F", "Cl"].into_iter()
            .map(|text| PreparedAtomToken::Fixed(text.to_owned())).collect(),
        vec![NonStereoBondToken::Elided; bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, 0,
    ).unwrap();
    let state = NonStereoWriterState::<NativeSolverState>::initial(&surface)
        .unwrap().unwrap_consistent();
    let (_, a) = super::tests::only_root_atom(&state, atoms[0]);
    let branch = a.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident ==
            surface.molecule().graph().neighbors(atoms[0]).unwrap().iter()
                .find(|incident| incident.bond() == bonds[0]).copied().unwrap()
        )).expect("A-X remains a legal first branch").into_successor();
    let x_states = super::tests::completed_atom_witnesses(&branch, atoms[1]);
    assert_eq!(x_states.len(), 1);
    let (_, x) = x_states.into_iter().next().unwrap();
    assert_eq!(x.active_atom(), Some(atoms[1]));
    let choices = x.choices().unwrap();
    assert!(!choices.iter().any(|choice|
        matches!(&choice.successor().pending, Some(PendingEmission::InlineAtom(entry)) if entry.incident ==
            surface.molecule().graph().neighbors(atoms[1]).unwrap().iter()
                .find(|incident| incident.bond() == bonds[1]).copied().unwrap()
        )), "X-B traversal strands both later ring endpoints at B");
    assert!(choices.iter().any(|choice|
        choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[1])),
        "X-B can open as a ring and later release its label");
    let mut expected = BTreeSet::new();
    for b_entry in [1, 4, 5] {
        let graph = Graph { atoms: &["C", "N", "O", "F", "Cl"],
            edges: edges.iter().enumerate().map(|(id, &(a, b))|
                (a, b, if [0, 2, 3, b_entry].contains(&id) { Role::Tree } else { Role::Ring }))
                .collect(), root: 0, labels: 1 };
        let reference = Reference::build(&graph);
        let x_id = reference.transitions[0].iter().find_map(|(event, _, next)|
            (*event == Event::Branch { bond: 0 }).then_some(*next)).unwrap();
        expected.extend(reference.choice_set(x_id));
    }
    let mut actual = BTreeMap::<(String, State), Vec<Vec<crate::domain::Domain>>>::new();
    for choice in &choices {
        let emitted = if matches!(&choice.successor().pending,
            Some(PendingEmission::RingOpeningLabel { .. })) {
            choice.successor().choices().unwrap().into_iter()
                .map(|tail| (format!("{}{}", choice.text(), tail.text()), tail.into_successor()))
                .collect::<Vec<_>>()
        } else { vec![(choice.text().to_owned(), choice.successor().clone())] };
        for (text, successor) in emitted {
        let observed = successor.observe_raw();
        assert_eq!(observed.pending, None);
        let traversal = &observed.structural.traversal;
        let mut next = State { visited: traversal.visited_atoms.iter().fold(0_u16,
                |bits, atom| bits | (1 << atom.index())),
            active: traversal.active_frame.as_ref().unwrap().atom.index(),
            returns: traversal.branch_returns.iter().rev().map(|frame| frame.atom.index()).collect(),
            bonds: vec![BondState::Unwritten; bonds.len()],
            started_child: traversal.branch_returns.iter()
                .chain(traversal.active_frame.iter())
                .filter(|frame| frame.branch_started)
                .fold(0_u16, |bits, frame| bits | (1 << frame.atom.index())) };
        for (id, progress) in traversal.bond_progress.iter().enumerate() {
            next.bonds[id] = match progress {
                crate::traversal::ObservedBondProgress::Unrepresented => BondState::Unwritten,
                crate::traversal::ObservedBondProgress::Traversed { .. } => BondState::Tree,
                crate::traversal::ObservedBondProgress::RingOpen { first_endpoint } => BondState::Open {
                    at: first_endpoint.index(),
                    label: observed.labels_by_bond.iter().find(|(bond, _)| *bond == bonds[id])
                        .expect("held ring has a slot").1 + 1,
                },
                crate::traversal::ObservedBondProgress::RingClosed { .. } => BondState::Closed,
            };
        }
        let versions = actual.entry((text, next)).or_default();
        assert!(!versions.contains(&observed.structural.bond_plan_domains),
            "duplicate full semantic successor at X");
        versions.push(observed.structural.bond_plan_domains);
        }
    }
    assert_eq!(actual.keys().cloned().collect::<BTreeSet<_>>(), expected,
        "unseeded X structural projections match compatible raw roles");
    assert!(actual.iter().all(|((text, _), versions)|
        versions.len() == if text.starts_with('-') { 1 } else { 2 }),
        "omitted first-end spelling keeps two separate far-end plans");
    let xb_open = choices.into_iter().find(|choice|
        choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[1]))
        .unwrap().into_successor();
    let returned = xb_open.choices().unwrap().into_iter()
        .find(|choice| choice.text() == ")").unwrap().into_successor();
    let branch_c = returned.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::BranchTraversalEmission(incident)) if incident.bond() == bonds[2]))
        .expect("A-C remains a branch after X-B opens").into_successor();
    let (_, c) = super::tests::completed_atom_witnesses(&branch_c, atoms[3])
        .into_iter().next().unwrap();
    let inline_b = c.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::InlineAtom(incident)) if incident.bond() == bonds[4]))
        .expect("C enters B through traversal").into_successor();
    let (_, b) = super::tests::completed_atom_witnesses(&inline_b, atoms[2])
        .into_iter().next().unwrap();
    let xb_closed = b.choices().unwrap().into_iter().find(|choice|
        !choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[1]))
        .expect("B closes X-B before its next ring event").into_successor();
    assert!(xb_closed.labels.is_clean(), "closing X-B frees the sole slot");
    let bd_open = xb_closed.choices().unwrap().into_iter().find(|choice|
        choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[5]))
        .expect("B reuses the released slot for B-D").into_successor();
    let returned = bd_open.choices().unwrap().into_iter()
        .find(|choice| choice.text() == ")").unwrap().into_successor();
    let inline_d = returned.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::InlineAtom(incident)) if incident.bond() == bonds[3]))
        .expect("A enters D after the C-B branch").into_successor();
    let (_, d) = super::tests::completed_atom_witnesses(&inline_d, atoms[4])
        .into_iter().next().unwrap();
    let bd_closed = d.choices().unwrap().into_iter().find(|choice|
        choice.successor().labels.is_clean())
        .expect("D closes B-D after the label was reused").into_successor();
    assert!(bd_closed.is_accepted());
}

#[test]
fn held_label_makes_b_triangle_first_branch_unproductive() {
    use super::*;
    use crate::native::NativeSolverState;
    use crate::prepared::PreparedGraphBuilder;

    // Two triangles share A-B. P-H is held until H is entered through A-H;
    // B-C-D cannot discharge its ring before that release.
    let edges = [(0, 2), (0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (4, 5)];
    let mut builder = PreparedGraphBuilder::new();
    let atoms = (0..6).map(|_| builder.add_atom().unwrap()).collect::<Vec<_>>();
    let bonds = edges.map(|(a, b)| builder.add_bond(atoms[a], atoms[b]).unwrap());
    let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(builder.build()),
        ["C", "N", "O", "F", "Cl", "Br"].into_iter()
            .map(|text| PreparedAtomToken::Fixed(text.to_owned())).collect(),
        vec![NonStereoBondToken::Elided; bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, 0,
    ).unwrap();
    let state = NonStereoWriterState::<NativeSolverState>::initial(&surface)
        .unwrap().unwrap_consistent();
    let (_, p) = super::tests::only_root_atom(&state, atoms[0]);
    let ph_open = p.choices().unwrap().into_iter().find(|choice|
        choice.text() == "0" && choice.successor().labels.bonds_by_slot.values()
            .any(|bond| *bond == bonds[0])).expect("P opens P-H").into_successor();
    let pa = ph_open.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::InlineAtom(incident)) if incident.bond() == bonds[1]))
        .expect("P enters A while P-H remains held").into_successor();
    let (_, a) = super::tests::completed_atom_witnesses(&pa, atoms[1])
        .into_iter().next().unwrap();
    assert_eq!(a.active_atom(), Some(atoms[1]));
    assert!(a.labels.bonds_by_slot.values().any(|bond| *bond == bonds[0]));
    let first = a.choices().unwrap();
    assert!(!first.iter().any(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::BranchTraversalEmission(incident)) if incident.bond() == bonds[3])),
        "A-B first cannot finish the B-C-D triangle before P-H releases its slot");
    let ah = first.into_iter().find(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::BranchTraversalEmission(incident)) if incident.bond() == bonds[2]))
        .expect("A-H first retains a completion").into_successor();
    let (_, h) = super::tests::completed_atom_witnesses(&ah, atoms[2])
        .into_iter().next().unwrap();
    let ph_closed = h.choices().unwrap().into_iter().find(|choice|
        choice.successor().labels.is_clean())
        .expect("H closes P-H and releases its slot").into_successor();
    let returned = ph_closed.choices().unwrap().into_iter().find(|choice|
        choice.text() == ")").expect("return from A-H").into_successor();
    let ab = returned.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::InlineAtom(incident)) if incident.bond() == bonds[3]))
        .expect("A enters B after H closes the held ring").into_successor();
    let (_, b) = super::tests::completed_atom_witnesses(&ab, atoms[3])
        .into_iter().next().unwrap();
    let bc = b.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::BranchTraversalEmission(incident)) if incident.bond() == bonds[4]))
        .expect("B enters C as a branch").into_successor();
    let (_, c) = super::tests::completed_atom_witnesses(&bc, atoms[4])
        .into_iter().next().unwrap();
    let cd_open = c.choices().unwrap().into_iter().find(|choice|
        choice.text() == "0" && choice.successor().labels.bonds_by_slot.values()
            .any(|bond| *bond == bonds[6]))
        .expect("C reuses the slot for C-D").into_successor();
    let returned = cd_open.choices().unwrap().into_iter().find(|choice|
        choice.text() == ")").expect("return from B-C").into_successor();
    let bd = returned.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::InlineAtom(incident)) if incident.bond() == bonds[5]))
        .expect("B enters D inline").into_successor();
    let (_, d) = super::tests::completed_atom_witnesses(&bd, atoms[5])
        .into_iter().next().unwrap();
    let cd_closed = d.choices().unwrap().into_iter().find(|choice|
        choice.successor().labels.is_clean()).expect("D closes C-D").into_successor();
    assert!(cd_closed.is_accepted());

    // With only the future Event factor disabled, the live child-entry rule
    // still rejects A-B before publishing a branch.
    let mut local_surface = surface.clone();
    local_surface.event_schedule_factor = None;
    let local_state = NonStereoWriterState::<NativeSolverState>::initial(&local_surface)
        .unwrap().unwrap_consistent();
    let (_, local_p) = super::tests::only_root_atom(&local_state, atoms[0]);
    let local_ph = local_p.choices().unwrap().into_iter().find(|choice|
        choice.text() == "0" && choice.successor().labels.bonds_by_slot.values()
            .any(|bond| *bond == bonds[0])).unwrap().into_successor();
    let local_pa = local_ph.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::InlineAtom(incident)) if incident.bond() == bonds[1]))
        .unwrap().into_successor();
    let (_, local_a) = super::tests::completed_atom_witnesses(&local_pa, atoms[1])
        .into_iter().next().unwrap();
    let local_choices = local_a.choices().unwrap();
    assert!(!local_choices.iter().any(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::BranchTraversalEmission(incident)) if incident.bond() == bonds[3])));
    assert!(local_choices.iter().any(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::BranchTraversalEmission(incident)) if incident.bond() == bonds[2])));

    let mut builder = PreparedGraphBuilder::new();
    let spare_atoms = (0..6).map(|_| builder.add_atom().unwrap()).collect::<Vec<_>>();
    let spare_bonds = edges.map(|(a, b)| builder.add_bond(spare_atoms[a], spare_atoms[b]).unwrap());
    let mut spare_surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(builder.build()),
        ["C", "N", "O", "F", "Cl", "Br"].into_iter()
            .map(|text| PreparedAtomToken::Fixed(text.to_owned())).collect(),
        vec![NonStereoBondToken::Elided; spare_bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, 1,
    ).unwrap();
    spare_surface.event_schedule_factor = None;
    let spare_state = NonStereoWriterState::<NativeSolverState>::initial(&spare_surface)
        .unwrap().unwrap_consistent();
    let (_, spare_p) = super::tests::only_root_atom(&spare_state, spare_atoms[0]);
    let spare_ph = spare_p.choices().unwrap().into_iter().find(|choice|
        choice.text() == "0" && choice.successor().labels.bonds_by_slot.values()
            .any(|bond| *bond == spare_bonds[0])).unwrap().into_successor();
    let spare_pa = spare_ph.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::InlineAtom(incident)) if incident.bond() == spare_bonds[1]))
        .unwrap().into_successor();
    let (_, spare_a) = super::tests::completed_atom_witnesses(&spare_pa, spare_atoms[1])
        .into_iter().next().unwrap();
    assert!(spare_a.choices().unwrap().iter().any(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::BranchTraversalEmission(incident)) if incident.bond() == spare_bonds[3])));
}

#[test]
fn held_target_in_same_residual_component_rejects_b_branch() {
    use super::*;
    use crate::native::NativeSolverState;
    use crate::prepared::PreparedGraphBuilder;

    let edges = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 4), (4, 5), (5, 3), (5, 2)];
    let mut builder = PreparedGraphBuilder::new();
    let atoms = (0..6).map(|_| builder.add_atom().unwrap()).collect::<Vec<_>>();
    let bonds = edges.map(|(a, b)| builder.add_bond(atoms[a], atoms[b]).unwrap());
    let mut surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(builder.build()),
        ["P", "A", "H", "B", "C", "D"].into_iter()
            .map(|text| PreparedAtomToken::Fixed(text.to_owned())).collect(),
        vec![NonStereoBondToken::Elided; bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, 0,
    ).unwrap();
    surface.event_schedule_factor = None;
    let state = NonStereoWriterState::<NativeSolverState>::initial(&surface)
        .unwrap().unwrap_consistent();
    let (_, p) = super::tests::only_root_atom(&state, atoms[0]);
    let ph = p.choices().unwrap().into_iter().find(|choice|
        choice.text() == "0" && choice.successor().labels.bonds_by_slot.values()
            .any(|bond| *bond == bonds[1])).unwrap().into_successor();
    let pa = ph.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::InlineAtom(incident)) if incident.bond() == bonds[0]))
        .unwrap().into_successor();
    let (_, a) = super::tests::completed_atom_witnesses(&pa, atoms[1])
        .into_iter().next().unwrap();
    assert!(a.structural.derive_candidates().candidates().iter().any(|candidate|
        matches!(candidate, StructuralCandidate::BranchChild { incident }
            if incident.bond() == bonds[3])));
    let choices = a.choices().unwrap();
    assert!(!choices.iter().any(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::BranchTraversalEmission(incident)) if incident.bond() == bonds[3])));
    assert!(choices.iter().any(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::BranchTraversalEmission(incident)) if incident.bond() == bonds[2])));
}

#[test]
fn tree_inline_child_can_drain_to_saved_ring_target_owner() {
    use super::*;
    use crate::native::NativeSolverState;
    use crate::prepared::PreparedGraphBuilder;

    // P holds P-H; X owns the sole X-H traversal entry. X-A branches into
    // a tree, where A-B can drain inline before returning to X.
    let edges = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 4)];
    let mut builder = PreparedGraphBuilder::new();
    let atoms = (0..5).map(|_| builder.add_atom().unwrap()).collect::<Vec<_>>();
    let bonds = edges.map(|(a, b)| builder.add_bond(atoms[a], atoms[b]).unwrap());
    let mut surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(builder.build()),
        ["P", "X", "H", "A", "B"].into_iter()
            .map(|text| PreparedAtomToken::Fixed(text.to_owned())).collect(),
        vec![NonStereoBondToken::Elided; bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, 0,
    ).unwrap();
    surface.event_schedule_factor = None;
    let state = NonStereoWriterState::<NativeSolverState>::initial(&surface)
        .unwrap().unwrap_consistent();
    let (_, p) = super::tests::only_root_atom(&state, atoms[0]);
    let ph = p.choices().unwrap().into_iter().find(|choice|
        choice.text() == "0" && choice.successor().labels.bonds_by_slot.values()
            .any(|bond| *bond == bonds[1])).unwrap().into_successor();
    let px = ph.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::InlineAtom(incident)) if incident.bond() == bonds[0]))
        .unwrap().into_successor();
    let (_, x) = super::tests::completed_atom_witnesses(&px, atoms[1])
        .into_iter().next().unwrap();
    let xa = x.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::BranchTraversalEmission(incident)) if incident.bond() == bonds[3]))
        .expect("the tree branch drains to saved X").into_successor();
    let (_, a) = super::tests::completed_atom_witnesses(&xa, atoms[3])
        .into_iter().next().unwrap();
    assert!(a.structural.derive_candidates().candidates().iter().any(|candidate|
        matches!(candidate, StructuralCandidate::InlineChild { incident }
            if incident.bond() == bonds[4])));
    assert!(a.choices().unwrap().iter().any(|choice|
        matches!(&choice.successor().pending,
            Some(PendingEmission::InlineAtom(incident)) if incident.bond() == bonds[4])));
}
