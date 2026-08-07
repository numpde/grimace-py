//! Graph-general control state for the South Star 2 writer.
//!
//! This module owns evolving graph-representation facts, the active textual
//! path, branch returns, and live ring-label bindings. Label rendering and
//! constraint state remain separate concerns.

use std::collections::BTreeMap;

use crate::ids::{AtomId, BondId};
use crate::prepared::{AdjacentBond, PreparedGraph};

#[derive(Clone, Debug, PartialEq, Eq)]
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

/// A zero-based live ring-label resource.
///
/// Rendering policy may map a slot to any valid SMILES label spelling. The
/// traversal state tracks only equality, ownership, and reuse.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct RingLabelSlot(usize);

impl RingLabelSlot {
    pub(crate) const fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ActiveRingLabels {
    bonds_by_slot: BTreeMap<RingLabelSlot, BondId>,
}

impl ActiveRingLabels {
    fn next_available(&self) -> RingLabelSlot {
        let mut candidate = 0;
        for slot in self.bonds_by_slot.keys() {
            if slot.index() != candidate {
                break;
            }
            candidate += 1;
        }
        RingLabelSlot(candidate)
    }

    fn allocate(&mut self, bond: BondId) -> RingLabelSlot {
        let slot = self.next_available();
        assert_eq!(
            self.bonds_by_slot.insert(slot, bond),
            None,
            "a newly allocated ring-label slot must be free"
        );
        slot
    }

    fn require_owner(&self, slot: RingLabelSlot, bond: BondId) {
        assert_eq!(
            self.bonds_by_slot.get(&slot),
            Some(&bond),
            "an open ring label must belong to its bond"
        );
    }

    fn release(&mut self, slot: RingLabelSlot, bond: BondId) {
        self.require_owner(slot, bond);
        assert_eq!(self.bonds_by_slot.remove(&slot), Some(bond));
    }

    fn is_empty(&self) -> bool {
        self.bonds_by_slot.is_empty()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum BondProgress {
    Unrepresented,
    Traversed {
        from: AtomId,
        to: AtomId,
    },
    RingOpen {
        first_endpoint: AtomId,
        label_slot: RingLabelSlot,
    },
    RingClosed {
        first_endpoint: AtomId,
        second_endpoint: AtomId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GraphProgress {
    visited_atoms: DenseSet,
    bonds: Box<[BondProgress]>,
    represented_bond_count: usize,
}

impl GraphProgress {
    fn new(graph: &PreparedGraph) -> Self {
        Self {
            visited_atoms: DenseSet::new(graph.atom_count()),
            bonds: vec![BondProgress::Unrepresented; graph.bond_count()].into_boxed_slice(),
            represented_bond_count: 0,
        }
    }

    fn atom_is_visited(&self, atom: AtomId) -> bool {
        self.visited_atoms.contains(atom.index())
    }

    fn visit_atom(&mut self, atom: AtomId) {
        self.visited_atoms.insert_new(atom.index());
    }

    fn traverse_bond(&mut self, bond: BondId, from: AtomId, to: AtomId) {
        let progress = self.bond_progress_mut(bond);
        assert_eq!(
            *progress,
            BondProgress::Unrepresented,
            "a traversed bond must not already have a representation"
        );
        *progress = BondProgress::Traversed { from, to };
        self.represented_bond_count += 1;
    }

    fn open_ring(
        &mut self,
        bond: BondId,
        first_endpoint: AtomId,
        label_slot: RingLabelSlot,
    ) {
        let progress = self.bond_progress_mut(bond);
        assert_eq!(
            *progress,
            BondProgress::Unrepresented,
            "a ring bond must be unrepresented when its first endpoint is written"
        );
        *progress = BondProgress::RingOpen {
            first_endpoint,
            label_slot,
        };
    }

    fn close_ring(
        &mut self,
        bond: BondId,
        second_endpoint: AtomId,
    ) -> RingLabelSlot {
        let progress = self.bond_progress_mut(bond);
        let BondProgress::RingOpen {
            first_endpoint,
            label_slot,
        } = *progress
        else {
            panic!("a ring bond must be open before its second endpoint is written");
        };
        assert_ne!(
            first_endpoint, second_endpoint,
            "ring endpoints must belong to distinct atoms"
        );
        *progress = BondProgress::RingClosed {
            first_endpoint,
            second_endpoint,
        };
        self.represented_bond_count += 1;
        label_slot
    }

    const fn is_complete(&self) -> bool {
        self.visited_atoms.is_complete() && self.represented_bond_count == self.bonds.len()
    }

    fn classify_incident(
        &self,
        graph: &PreparedGraph,
        at: AtomId,
        incident: AdjacentBond,
    ) -> IncidentBondState {
        let bond = graph
            .bond(incident.bond())
            .expect("incident bond must belong to the prepared graph");
        assert_eq!(
            bond.other(at),
            Some(incident.atom()),
            "incident bond must connect the active atom to its neighbour"
        );

        match self.bond_progress(incident.bond()) {
            BondProgress::Unrepresented => {
                if self.atom_is_visited(incident.atom()) {
                    IncidentBondState::UnrepresentedToVisitedAtom
                } else {
                    IncidentBondState::UnrepresentedToUnvisitedAtom
                }
            }
            BondProgress::Traversed { .. } | BondProgress::RingClosed { .. } => {
                IncidentBondState::Represented
            }
            BondProgress::RingOpen { first_endpoint, .. } if *first_endpoint == at => {
                IncidentBondState::RingOpenAtCurrentAtom
            }
            BondProgress::RingOpen { first_endpoint, .. } => {
                assert_eq!(
                    *first_endpoint,
                    incident.atom(),
                    "open ring endpoint must belong to one endpoint of its bond"
                );
                IncidentBondState::RingOpenAtOtherAtom
            }
        }
    }

    fn ring_label_slot(&self, bond: BondId) -> Option<RingLabelSlot> {
        match self.bond_progress(bond) {
            BondProgress::RingOpen { label_slot, .. } => Some(*label_slot),
            BondProgress::Unrepresented
            | BondProgress::Traversed { .. }
            | BondProgress::RingClosed { .. } => None,
        }
    }

    fn bond_progress(&self, bond: BondId) -> &BondProgress {
        self.bonds
            .get(bond.index())
            .expect("prepared bond identifier must fit the traversal universe")
    }

    fn bond_progress_mut(&mut self, bond: BondId) -> &mut BondProgress {
        self.bonds
            .get_mut(bond.index())
            .expect("prepared bond identifier must fit the traversal universe")
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum IncidentBondState {
    Represented,
    UnrepresentedToUnvisitedAtom,
    UnrepresentedToVisitedAtom,
    RingOpenAtCurrentAtom,
    RingOpenAtOtherAtom,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TraversalState {
    progress: GraphProgress,
    active: Option<AtomId>,
    branch_returns: Vec<AtomId>,
    ring_labels: ActiveRingLabels,
}

impl TraversalState {
    pub(crate) fn new(graph: &PreparedGraph) -> Self {
        Self {
            progress: GraphProgress::new(graph),
            active: None,
            branch_returns: Vec::new(),
            ring_labels: ActiveRingLabels::default(),
        }
    }

    pub(crate) const fn active_atom(&self) -> Option<AtomId> {
        self.active
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

    pub(crate) fn next_ring_label_slot(&self) -> RingLabelSlot {
        self.ring_labels.next_available()
    }

    pub(crate) fn begin_component(&mut self, root: AtomId) {
        assert!(
            self.active.is_none()
                && self.branch_returns.is_empty()
                && self.ring_labels.is_empty(),
            "a component can begin only after the previous component is closed"
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

    pub(crate) fn open_ring_endpoint(
        &mut self,
        graph: &PreparedGraph,
        incident: AdjacentBond,
    ) -> RingLabelSlot {
        let active = self.active.expect("a ring endpoint requires an active atom");
        assert!(
            matches!(
                self.progress.classify_incident(graph, active, incident),
                IncidentBondState::UnrepresentedToUnvisitedAtom
                    | IncidentBondState::UnrepresentedToVisitedAtom
            ),
            "a first ring endpoint requires an unrepresented incident bond"
        );

        let label_slot = self.ring_labels.allocate(incident.bond());
        self.progress
            .open_ring(incident.bond(), active, label_slot);
        label_slot
    }

    pub(crate) fn close_ring_endpoint(
        &mut self,
        graph: &PreparedGraph,
        incident: AdjacentBond,
    ) -> RingLabelSlot {
        let active = self.active.expect("a ring endpoint requires an active atom");
        assert_eq!(
            self.progress.classify_incident(graph, active, incident),
            IncidentBondState::RingOpenAtOtherAtom,
            "a second ring endpoint must pair a bond opened at its other atom"
        );

        let label_slot = self
            .progress
            .ring_label_slot(incident.bond())
            .expect("an open ring bond must own a label slot");
        self.ring_labels
            .require_owner(label_slot, incident.bond());
        let closed_slot = self.progress.close_ring(incident.bond(), active);
        assert_eq!(closed_slot, label_slot);
        self.ring_labels.release(label_slot, incident.bond());
        label_slot
    }

    pub(crate) fn ring_label_slot_for_active_incident(
        &self,
        graph: &PreparedGraph,
        incident: AdjacentBond,
    ) -> Option<RingLabelSlot> {
        let active = self.active.expect("ring-label lookup requires an active atom");
        self.progress.classify_incident(graph, active, incident);
        self.progress.ring_label_slot(incident.bond())
    }

    /// Complete the current textual path and restore its enclosing branch.
    ///
    /// The transition kernel is responsible for establishing that the active
    /// atom has no pending graph, label-rendering, or emission work before
    /// calling this operation.
    pub(crate) fn complete_path(&mut self) -> Option<AtomId> {
        assert!(self.active.is_some(), "no active path to complete");
        let restored = self.branch_returns.pop();
        if restored.is_none() {
            assert!(
                self.ring_labels.is_empty(),
                "a component cannot end with an unpaired ring endpoint"
            );
        }
        self.active = restored;
        self.active
    }

    pub(crate) fn classify_active_incident(
        &self,
        graph: &PreparedGraph,
        incident: AdjacentBond,
    ) -> IncidentBondState {
        let active = self.active.expect("incident classification requires an active atom");
        self.progress.classify_incident(graph, active, incident)
    }

    fn enter_child(
        &mut self,
        graph: &PreparedGraph,
        incident: AdjacentBond,
        placement: ChildPlacement,
    ) {
        let parent = self.active.expect("a child requires an active atom");
        assert_eq!(
            self.progress.classify_incident(graph, parent, incident),
            IncidentBondState::UnrepresentedToUnvisitedAtom,
            "a child edge must be unrepresented and lead to an unvisited atom"
        );

        if placement == ChildPlacement::Branch {
            self.branch_returns.push(parent);
        }
        self.progress
            .traverse_bond(incident.bond(), parent, incident.atom());
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
    fn empty_graph_starts_complete() {
        let graph = PreparedGraphBuilder::new().build();
        let state = TraversalState::new(&graph);

        assert_eq!(state.active_atom(), None);
        assert!(state.graph_is_complete());
        assert_eq!(state.unvisited_atoms(&graph).count(), 0);
        assert_eq!(state.next_ring_label_slot().index(), 0);
    }

    #[test]
    fn disconnected_component_roots_remain_ordinary_choices() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 2] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        assert_eq!(state.unvisited_atoms(&graph).collect::<Vec<_>>(), atoms.to_vec());
        state.begin_component(atoms[1]);
        assert_eq!(state.unvisited_atoms(&graph).collect::<Vec<_>>(), vec![atoms[0]]);
        assert_eq!(state.complete_path(), None);
        state.begin_component(atoms[0]);
        assert_eq!(state.complete_path(), None);
        assert!(state.graph_is_complete());
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
        assert_eq!(state.complete_path(), Some(atoms[0]));
        state.enter_inline_child(&graph, incident(&graph, atoms[0], inline));
        assert!(state.graph_is_complete());
        assert_eq!(state.complete_path(), None);
    }

    #[test]
    fn ring_endpoint_lifecycle_preserves_its_abstract_label() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let first = builder.add_bond(atoms[0], atoms[1]).unwrap();
        let second = builder.add_bond(atoms[1], atoms[2]).unwrap();
        let ring = builder.add_bond(atoms[2], atoms[0]).unwrap();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(atoms[0]);
        let opening = incident(&graph, atoms[0], ring);
        let label_slot = state.open_ring_endpoint(&graph, opening);
        assert_eq!(label_slot.index(), 0);
        assert_eq!(
            state.ring_label_slot_for_active_incident(&graph, opening),
            Some(label_slot)
        );

        state.enter_inline_child(&graph, incident(&graph, atoms[0], first));
        state.enter_inline_child(&graph, incident(&graph, atoms[1], second));
        let closing = incident(&graph, atoms[2], ring);
        assert_eq!(
            state.ring_label_slot_for_active_incident(&graph, closing),
            Some(label_slot)
        );
        assert_eq!(state.close_ring_endpoint(&graph, closing), label_slot);

        assert_eq!(
            state.progress.bond_progress(ring),
            &BondProgress::RingClosed {
                first_endpoint: atoms[0],
                second_endpoint: atoms[2],
            }
        );
        assert_eq!(state.next_ring_label_slot().index(), 0);
        assert!(state.graph_is_complete());
        assert_eq!(state.complete_path(), None);
    }

    #[test]
    fn ring_label_slots_reuse_the_least_free_resource() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 5] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let path = [
            builder.add_bond(atoms[0], atoms[1]).unwrap(),
            builder.add_bond(atoms[1], atoms[2]).unwrap(),
            builder.add_bond(atoms[2], atoms[3]).unwrap(),
            builder.add_bond(atoms[3], atoms[4]).unwrap(),
        ];
        let first_ring = builder.add_bond(atoms[0], atoms[2]).unwrap();
        let long_ring = builder.add_bond(atoms[0], atoms[4]).unwrap();
        let reused_ring = builder.add_bond(atoms[2], atoms[4]).unwrap();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(atoms[0]);
        assert_eq!(
            state
                .open_ring_endpoint(&graph, incident(&graph, atoms[0], first_ring))
                .index(),
            0
        );
        assert_eq!(
            state
                .open_ring_endpoint(&graph, incident(&graph, atoms[0], long_ring))
                .index(),
            1
        );

        state.enter_inline_child(&graph, incident(&graph, atoms[0], path[0]));
        state.enter_inline_child(&graph, incident(&graph, atoms[1], path[1]));
        assert_eq!(
            state
                .close_ring_endpoint(&graph, incident(&graph, atoms[2], first_ring))
                .index(),
            0
        );
        assert_eq!(
            state
                .open_ring_endpoint(&graph, incident(&graph, atoms[2], reused_ring))
                .index(),
            0
        );

        state.enter_inline_child(&graph, incident(&graph, atoms[2], path[2]));
        state.enter_inline_child(&graph, incident(&graph, atoms[3], path[3]));
        assert_eq!(
            state
                .close_ring_endpoint(&graph, incident(&graph, atoms[4], long_ring))
                .index(),
            1
        );
        assert_eq!(
            state
                .close_ring_endpoint(&graph, incident(&graph, atoms[4], reused_ring))
                .index(),
            0
        );

        assert!(state.graph_is_complete());
        assert_eq!(state.next_ring_label_slot().index(), 0);
        assert_eq!(state.complete_path(), None);
    }

    #[test]
    #[should_panic(expected = "child edge must be unrepresented")]
    fn open_ring_bonds_cannot_be_traversed_as_children() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 2] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let bond = builder.add_bond(atoms[0], atoms[1]).unwrap();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);
        let edge = incident(&graph, atoms[0], bond);

        state.begin_component(atoms[0]);
        state.open_ring_endpoint(&graph, edge);
        state.enter_inline_child(&graph, edge);
    }

    #[test]
    #[should_panic(expected = "cannot end with an unpaired ring endpoint")]
    fn components_cannot_finish_with_open_ring_endpoints() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 2] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let bond = builder.add_bond(atoms[0], atoms[1]).unwrap();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(atoms[0]);
        state.open_ring_endpoint(&graph, incident(&graph, atoms[0], bond));
        state.complete_path();
    }

    #[test]
    fn cloned_traversal_has_independent_ring_labels() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 2] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let bond = builder.add_bond(atoms[0], atoms[1]).unwrap();
        let graph = builder.build();
        let mut source = TraversalState::new(&graph);
        source.begin_component(atoms[0]);
        let mut successor = source.clone();

        let slot = successor.open_ring_endpoint(&graph, incident(&graph, atoms[0], bond));
        assert_eq!(source.next_ring_label_slot().index(), 0);
        assert_eq!(successor.next_ring_label_slot().index(), 1);
        assert_eq!(
            successor.ring_label_slot_for_active_incident(
                &graph,
                incident(&graph, atoms[0], bond)
            ),
            Some(slot)
        );
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
    #[should_panic(expected = "prepared identifier must fit")]
    fn stale_prepared_ids_fail_fast() {
        let graph = PreparedGraphBuilder::new().build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(AtomId::new(0));
    }
}
