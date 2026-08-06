//! Graph-general control state for the South Star 2 writer.
//!
//! This module owns only evolving topology facts: represented atoms and bonds,
//! the active textual path, and branch returns. Ring endpoints, emitted text,
//! and constraint state remain separate facts and must be composed before a
//! walker interface is exposed.

use crate::ids::{AtomId, BondId};
use crate::prepared::{AdjacentBond, PreparedGraph};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DenseSet {
    universe_len: usize,
    marked_count: usize,
    words: Box<[u64]>,
}

impl DenseSet {
    fn new(universe_len: usize) -> Self {
        let word_count = universe_len / u64::BITS as usize
            + usize::from(universe_len % u64::BITS as usize != 0);
        Self {
            universe_len,
            marked_count: 0,
            words: vec![0; word_count].into_boxed_slice(),
        }
    }

    fn contains(&self, index: usize) -> bool {
        let (word, mask) = self.location(index);
        self.words[word] & mask != 0
    }

    fn insert_new(&mut self, index: usize) {
        let (word, mask) = self.location(index);
        assert!(
            self.words[word] & mask == 0,
            "graph progress fact must be recorded exactly once"
        );
        self.words[word] |= mask;
        self.marked_count += 1;
    }

    const fn is_complete(&self) -> bool {
        self.marked_count == self.universe_len
    }

    fn location(&self, index: usize) -> (usize, u64) {
        assert!(
            index < self.universe_len,
            "prepared identifier must fit the traversal universe"
        );
        let word = index / u64::BITS as usize;
        let bit = index % u64::BITS as usize;
        (word, 1_u64 << bit)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GraphProgress {
    visited_atoms: DenseSet,
    written_bonds: DenseSet,
}

impl GraphProgress {
    fn new(graph: &PreparedGraph) -> Self {
        Self {
            visited_atoms: DenseSet::new(graph.atom_count()),
            written_bonds: DenseSet::new(graph.bond_count()),
        }
    }

    fn atom_is_visited(&self, atom: AtomId) -> bool {
        self.visited_atoms.contains(atom.index())
    }

    fn visit_atom(&mut self, atom: AtomId) {
        self.visited_atoms.insert_new(atom.index());
    }

    fn write_bond(&mut self, bond: BondId) {
        self.written_bonds.insert_new(bond.index());
    }

    const fn is_complete(&self) -> bool {
        self.visited_atoms.is_complete() && self.written_bonds.is_complete()
    }

    fn classify_incident(&self, incident: AdjacentBond) -> IncidentBondState {
        if self.written_bonds.contains(incident.bond().index()) {
            IncidentBondState::Written
        } else if self.atom_is_visited(incident.atom()) {
            IncidentBondState::UnwrittenToVisitedAtom
        } else {
            IncidentBondState::UnwrittenToUnvisitedAtom
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum IncidentBondState {
    Written,
    UnwrittenToUnvisitedAtom,
    UnwrittenToVisitedAtom,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TraversalState {
    progress: GraphProgress,
    active: Option<AtomId>,
    branch_returns: Vec<AtomId>,
}

impl TraversalState {
    pub(crate) fn new(graph: &PreparedGraph) -> Self {
        Self {
            progress: GraphProgress::new(graph),
            active: None,
            branch_returns: Vec::new(),
        }
    }

    pub(crate) const fn active_atom(&self) -> Option<AtomId> {
        self.active
    }

    pub(crate) const fn is_between_components(&self) -> bool {
        self.active.is_none()
    }

    pub(crate) const fn graph_is_complete(&self) -> bool {
        self.progress.is_complete()
    }

    pub(crate) fn unvisited_atoms<'a>(
        &'a self,
        graph: &'a PreparedGraph,
    ) -> impl Iterator<Item = AtomId> + 'a {
        graph
            .atom_ids()
            .filter(|atom| !self.progress.atom_is_visited(*atom))
    }

    pub(crate) fn begin_component(&mut self, root: AtomId) {
        assert!(
            self.active.is_none() && self.branch_returns.is_empty(),
            "a component can begin only between active paths"
        );
        self.progress.visit_atom(root);
        self.active = Some(root);
    }

    pub(crate) fn enter_inline_child(
        &mut self,
        graph: &PreparedGraph,
        incident: AdjacentBond,
    ) {
        self.enter_child(graph, incident, ChildPlacement::Inline);
    }

    pub(crate) fn enter_branch_child(
        &mut self,
        graph: &PreparedGraph,
        incident: AdjacentBond,
    ) {
        self.enter_child(graph, incident, ChildPlacement::Branch);
    }

    /// Complete the current textual path and restore its enclosing branch.
    ///
    /// The transition kernel is responsible for establishing that the active
    /// atom has no pending graph or ring work before calling this operation.
    pub(crate) fn complete_path(&mut self) -> Option<AtomId> {
        assert!(self.active.is_some(), "no active path to complete");
        self.active = self.branch_returns.pop();
        self.active
    }

    pub(crate) fn classify_incident(&self, incident: AdjacentBond) -> IncidentBondState {
        self.progress.classify_incident(incident)
    }

    fn enter_child(
        &mut self,
        graph: &PreparedGraph,
        incident: AdjacentBond,
        placement: ChildPlacement,
    ) {
        let parent = self.active.expect("a child requires an active atom");
        let bond = graph
            .bond(incident.bond())
            .expect("incident bond must belong to the prepared graph");
        assert_eq!(
            bond.other(parent),
            Some(incident.atom()),
            "incident bond must connect the active atom to the child"
        );
        assert_eq!(
            self.progress.classify_incident(incident),
            IncidentBondState::UnwrittenToUnvisitedAtom,
            "a child edge must be unwritten and lead to an unvisited atom"
        );

        if placement == ChildPlacement::Branch {
            self.branch_returns.push(parent);
        }
        self.progress.write_bond(incident.bond());
        self.progress.visit_atom(incident.atom());
        self.active = Some(incident.atom());
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ChildPlacement {
    Inline,
    Branch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prepared::PreparedGraphBuilder;

    fn incident(graph: &PreparedGraph, atom: AtomId, bond: BondId) -> AdjacentBond {
        graph
            .neighbors(atom)
            .expect("fixture atom must exist")
            .iter()
            .copied()
            .find(|candidate| candidate.bond() == bond)
            .expect("fixture bond must be incident to the atom")
    }

    #[test]
    fn empty_graph_starts_between_components_and_complete() {
        let graph = PreparedGraphBuilder::new().build();
        let state = TraversalState::new(&graph);

        assert!(state.is_between_components());
        assert!(state.graph_is_complete());
        assert_eq!(state.unvisited_atoms(&graph).count(), 0);
    }

    #[test]
    fn disconnected_component_roots_remain_ordinary_choices() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 2] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        assert_eq!(state.unvisited_atoms(&graph).collect::<Vec<_>>(), atoms.to_vec());
        state.begin_component(atoms[1]);
        assert_eq!(state.active_atom(), Some(atoms[1]));
        assert_eq!(state.unvisited_atoms(&graph).collect::<Vec<_>>(), vec![atoms[0]]);
        assert_eq!(state.complete_path(), None);

        state.begin_component(atoms[0]);
        assert_eq!(state.complete_path(), None);
        assert!(state.graph_is_complete());
    }

    #[test]
    fn inline_children_advance_without_branch_returns() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let bonds = [
            builder.add_bond(atoms[0], atoms[1]).unwrap(),
            builder.add_bond(atoms[1], atoms[2]).unwrap(),
        ];
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(atoms[0]);
        state.enter_inline_child(&graph, incident(&graph, atoms[0], bonds[0]));
        state.enter_inline_child(&graph, incident(&graph, atoms[1], bonds[1]));

        assert_eq!(state.active_atom(), Some(atoms[2]));
        assert!(state.branch_returns.is_empty());
        assert!(state.graph_is_complete());
        assert_eq!(state.complete_path(), None);
    }

    #[test]
    fn branch_children_restore_the_parent() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let branch = builder.add_bond(atoms[0], atoms[1]).unwrap();
        let inline = builder.add_bond(atoms[0], atoms[2]).unwrap();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(atoms[0]);
        state.enter_branch_child(&graph, incident(&graph, atoms[0], branch));
        assert_eq!(state.branch_returns, vec![atoms[0]]);
        assert_eq!(state.complete_path(), Some(atoms[0]));

        state.enter_inline_child(&graph, incident(&graph, atoms[0], inline));
        assert!(state.branch_returns.is_empty());
        assert!(state.graph_is_complete());
        assert_eq!(state.complete_path(), None);
    }

    #[test]
    fn inline_descent_inside_a_branch_returns_to_the_branch_parent() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let branch = builder.add_bond(atoms[0], atoms[1]).unwrap();
        let inline = builder.add_bond(atoms[1], atoms[2]).unwrap();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(atoms[0]);
        state.enter_branch_child(&graph, incident(&graph, atoms[0], branch));
        state.enter_inline_child(&graph, incident(&graph, atoms[1], inline));

        assert_eq!(state.active_atom(), Some(atoms[2]));
        assert_eq!(state.complete_path(), Some(atoms[0]));
    }

    #[test]
    fn cyclic_topology_exposes_an_unwritten_edge_to_a_visited_atom() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let first = builder.add_bond(atoms[0], atoms[1]).unwrap();
        let second = builder.add_bond(atoms[1], atoms[2]).unwrap();
        let closing = builder.add_bond(atoms[2], atoms[0]).unwrap();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(atoms[0]);
        state.enter_inline_child(&graph, incident(&graph, atoms[0], first));
        state.enter_inline_child(&graph, incident(&graph, atoms[1], second));

        assert_eq!(
            state.classify_incident(incident(&graph, atoms[2], closing)),
            IncidentBondState::UnwrittenToVisitedAtom
        );
        assert!(!state.graph_is_complete());
    }

    #[test]
    fn cloned_traversal_has_independent_live_state() {
        let mut builder = PreparedGraphBuilder::new();
        let atom = builder.add_atom().unwrap();
        let graph = builder.build();
        let source = TraversalState::new(&graph);
        let mut successor = source.clone();

        successor.begin_component(atom);
        assert_eq!(source.active_atom(), None);
        assert_eq!(successor.active_atom(), Some(atom));
        assert_eq!(source.unvisited_atoms(&graph).collect::<Vec<_>>(), vec![atom]);
        assert_eq!(successor.unvisited_atoms(&graph).count(), 0);
    }

    #[test]
    #[should_panic(expected = "recorded exactly once")]
    fn revisiting_an_atom_fails_fast() {
        let mut builder = PreparedGraphBuilder::new();
        let atom = builder.add_atom().unwrap();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(atom);
        state.complete_path();
        state.begin_component(atom);
    }

    #[test]
    #[should_panic(expected = "connect the active atom")]
    fn child_incidents_must_belong_to_the_active_atom() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let unrelated = builder.add_bond(atoms[1], atoms[2]).unwrap();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(atoms[0]);
        state.enter_inline_child(&graph, incident(&graph, atoms[1], unrelated));
    }

    #[test]
    #[should_panic(expected = "prepared identifier must fit")]
    fn stale_prepared_ids_fail_fast() {
        let graph = PreparedGraphBuilder::new().build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(AtomId::new(0));
    }
}
