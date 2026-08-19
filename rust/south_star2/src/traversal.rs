//! Graph-general control state for the South Star 2 writer.
//!
//! This module owns evolving graph-representation facts, active and suspended
//! writer frames, and live ring-label bindings. Each frame retains the residual
//! attachment groups computed when its atom was entered.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

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
        let word_count = universe_len.div_ceil(u64::BITS as usize);
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
    retired_at_current_atom: BTreeSet<RingLabelSlot>,
}

impl ActiveRingLabels {
    fn next_available(&self) -> RingLabelSlot {
        let mut candidate = RingLabelSlot(0);
        while self.bonds_by_slot.contains_key(&candidate)
            || self.retired_at_current_atom.contains(&candidate)
        {
            candidate = RingLabelSlot(
                candidate
                    .index()
                    .checked_add(1)
                    .expect("ring-label slot space must not overflow"),
            );
        }
        candidate
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

    fn retire_at_current_atom(&mut self, slot: RingLabelSlot, bond: BondId) {
        self.require_owner(slot, bond);
        assert_eq!(self.bonds_by_slot.remove(&slot), Some(bond));
        assert!(
            self.retired_at_current_atom.insert(slot),
            "a ring-label slot may retire only once at one atom"
        );
    }

    fn finish_current_atom(&mut self) {
        self.retired_at_current_atom.clear();
    }

    fn is_empty(&self) -> bool {
        self.bonds_by_slot.is_empty()
    }

    fn is_clean(&self) -> bool {
        self.bonds_by_slot.is_empty() && self.retired_at_current_atom.is_empty()
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

    fn open_ring(&mut self, bond: BondId, first_endpoint: AtomId, label_slot: RingLabelSlot) {
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

    fn close_ring(&mut self, bond: BondId, second_endpoint: AtomId) -> RingLabelSlot {
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

    fn ring_open_facts(&self, bond: BondId) -> Option<(AtomId, RingLabelSlot)> {
        match self.bond_progress(bond) {
            BondProgress::RingOpen {
                first_endpoint,
                label_slot,
            } => Some((*first_endpoint, *label_slot)),
            BondProgress::Unrepresented
            | BondProgress::Traversed { .. }
            | BondProgress::RingClosed { .. } => None,
        }
    }

    fn ring_label_slot(&self, bond: BondId) -> Option<RingLabelSlot> {
        self.ring_open_facts(bond)
            .map(|(_first_endpoint, label_slot)| label_slot)
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

/// Incidences from one writer frame into one connected component of the
/// currently unvisited induced graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResidualAttachment {
    incidences: Vec<AdjacentBond>,
}

impl ResidualAttachment {
    pub(crate) fn incidences(&self) -> &[AdjacentBond] {
        &self.incidences
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WriterFrame {
    atom: AtomId,
    attachments: Vec<ResidualAttachment>,
}

impl WriterFrame {
    fn new(graph: &PreparedGraph, progress: &GraphProgress, atom: AtomId) -> Self {
        Self {
            atom,
            attachments: residual_attachments(graph, progress, atom),
        }
    }

    fn remove_ring_incidence(&mut self, incident: AdjacentBond) {
        let attachment = self
            .attachments
            .iter_mut()
            .find(|attachment| attachment.incidences.contains(&incident))
            .expect("a ring opening to an unvisited atom must belong to an active attachment");
        assert!(
            attachment.incidences.len() > 1,
            "one traversal entry must remain for every residual attachment"
        );
        let offset = attachment
            .incidences
            .iter()
            .position(|candidate| *candidate == incident)
            .expect("active attachment must contain its ring incidence");
        attachment.incidences.remove(offset);
    }

    fn consume_child_attachment(&mut self, incident: AdjacentBond) {
        let offset = self
            .attachments
            .iter()
            .position(|attachment| {
                attachment.incidences.len() == 1 && attachment.incidences[0] == incident
            })
            .expect("a child must be the sole remaining incidence of its attachment");
        self.attachments.remove(offset);
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
    active: Option<WriterFrame>,
    branch_returns: Vec<WriterFrame>,
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

    pub(crate) fn active_atom(&self) -> Option<AtomId> {
        self.active.as_ref().map(|frame| frame.atom)
    }

    pub(crate) const fn graph_is_complete(&self) -> bool {
        self.progress.is_complete()
    }

    pub(crate) fn can_complete_path(&self) -> bool {
        self.active.is_some() && (!self.branch_returns.is_empty() || self.ring_labels.is_empty())
    }

    pub(crate) fn active_attachments(&self) -> &[ResidualAttachment] {
        self.active
            .as_ref()
            .map(|frame| frame.attachments.as_slice())
            .expect("attachment lookup requires an active writer frame")
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

    pub(crate) fn begin_component(&mut self, graph: &PreparedGraph, root: AtomId) {
        assert!(
            self.active.is_none()
                && self.branch_returns.is_empty()
                && self.ring_labels.is_clean(),
            "a component can begin only after the previous component is closed"
        );
        self.progress.visit_atom(root);
        self.active = Some(WriterFrame::new(graph, &self.progress, root));
    }

    pub(crate) fn enter_inline_child(&mut self, graph: &PreparedGraph, incident: AdjacentBond) {
        self.enter_child(graph, incident, ChildPlacement::Inline);
    }

    pub(crate) fn enter_branch_child(&mut self, graph: &PreparedGraph, incident: AdjacentBond) {
        self.enter_child(graph, incident, ChildPlacement::Branch);
    }

    pub(crate) fn open_ring_endpoint(
        &mut self,
        graph: &PreparedGraph,
        incident: AdjacentBond,
    ) -> RingLabelSlot {
        let active_atom = self
            .active_atom()
            .expect("a ring endpoint requires an active atom");
        let incident_state = self
            .progress
            .classify_incident(graph, active_atom, incident);
        assert_eq!(
            incident_state,
            IncidentBondState::UnrepresentedToUnvisitedAtom,
            "a first ring endpoint must be written before its other atom is visited"
        );
        self.active
            .as_mut()
            .expect("ring opening requires an active frame")
            .remove_ring_incidence(incident);

        let label_slot = self.ring_labels.allocate(incident.bond());
        self.progress
            .open_ring(incident.bond(), active_atom, label_slot);
        label_slot
    }

    pub(crate) fn close_ring_endpoint(
        &mut self,
        graph: &PreparedGraph,
        incident: AdjacentBond,
    ) -> RingLabelSlot {
        let active_atom = self
            .active_atom()
            .expect("a ring endpoint requires an active atom");
        assert_eq!(
            self.progress
                .classify_incident(graph, active_atom, incident),
            IncidentBondState::RingOpenAtOtherAtom,
            "a second ring endpoint must pair a bond opened at its other atom"
        );

        let label_slot = self
            .progress
            .ring_label_slot(incident.bond())
            .expect("an open ring bond must own a label slot");
        self.ring_labels.require_owner(label_slot, incident.bond());
        let closed_slot = self.progress.close_ring(incident.bond(), active_atom);
        assert_eq!(closed_slot, label_slot);
        self.ring_labels
            .retire_at_current_atom(label_slot, incident.bond());
        label_slot
    }

    pub(crate) fn ring_open_facts_for_active_incident(
        &self,
        graph: &PreparedGraph,
        incident: AdjacentBond,
    ) -> Option<(AtomId, RingLabelSlot)> {
        let active_atom = self
            .active_atom()
            .expect("ring lookup requires an active atom");
        self.progress
            .classify_incident(graph, active_atom, incident);
        self.progress.ring_open_facts(incident.bond())
    }

    pub(crate) fn ring_label_slot_for_active_incident(
        &self,
        graph: &PreparedGraph,
        incident: AdjacentBond,
    ) -> Option<RingLabelSlot> {
        self.ring_open_facts_for_active_incident(graph, incident)
            .map(|(_first_endpoint, label_slot)| label_slot)
    }

    pub(crate) fn complete_path(&mut self) -> Option<AtomId> {
        let active = self.active.take().expect("no active path to complete");
        assert!(
            active.attachments.is_empty(),
            "a path cannot complete with unresolved residual attachments"
        );
        self.ring_labels.finish_current_atom();
        let restored = self.branch_returns.pop();
        if restored.is_none() {
            assert!(
                self.ring_labels.is_empty(),
                "a component cannot end with an unpaired ring endpoint"
            );
        }
        self.active = restored;
        self.active_atom()
    }

    pub(crate) fn classify_active_incident(
        &self,
        graph: &PreparedGraph,
        incident: AdjacentBond,
    ) -> IncidentBondState {
        let active_atom = self
            .active_atom()
            .expect("incident classification requires an active atom");
        self.progress
            .classify_incident(graph, active_atom, incident)
    }

    fn enter_child(
        &mut self,
        graph: &PreparedGraph,
        incident: AdjacentBond,
        placement: ChildPlacement,
    ) {
        let mut parent = self
            .active
            .take()
            .expect("a child requires an active frame");
        assert_eq!(
            self.progress
                .classify_incident(graph, parent.atom, incident),
            IncidentBondState::UnrepresentedToUnvisitedAtom,
            "a child edge must be unrepresented and lead to an unvisited atom"
        );
        parent.consume_child_attachment(incident);
        let parent_atom = parent.atom;

        match placement {
            ChildPlacement::Branch => {
                assert!(
                    !parent.attachments.is_empty(),
                    "a branch requires another residual attachment for the main continuation"
                );
            }
            ChildPlacement::Inline => {
                assert!(
                    parent.attachments.is_empty(),
                    "the inline child must consume the final residual attachment"
                );
            }
        }

        self.ring_labels.finish_current_atom();
        if placement == ChildPlacement::Branch {
            self.branch_returns.push(parent);
        }

        let child = incident.atom();
        self.progress
            .traverse_bond(incident.bond(), parent_atom, child);
        self.progress.visit_atom(child);
        self.active = Some(WriterFrame::new(graph, &self.progress, child));
    }
}

fn residual_attachments(
    graph: &PreparedGraph,
    progress: &GraphProgress,
    active: AtomId,
) -> Vec<ResidualAttachment> {
    let mut component_by_atom = vec![usize::MAX; graph.atom_count()];
    let mut component_mins = Vec::new();
    let mut component_count = 0_usize;

    for root in graph.atom_ids() {
        if progress.atom_is_visited(root) || component_by_atom[root.index()] != usize::MAX {
            continue;
        }

        component_by_atom[root.index()] = component_count;
        let mut minimum = root;
        let mut pending = VecDeque::from([root]);
        while let Some(atom) = pending.pop_front() {
            minimum = minimum.min(atom);
            for incident in graph
                .neighbors(atom)
                .expect("prepared atom must have an adjacency row")
            {
                let neighbour = incident.atom();
                if progress.atom_is_visited(neighbour)
                    || component_by_atom[neighbour.index()] != usize::MAX
                {
                    continue;
                }
                component_by_atom[neighbour.index()] = component_count;
                pending.push_back(neighbour);
            }
        }
        component_mins.push(minimum);
        component_count += 1;
    }

    let mut groups = vec![Vec::new(); component_count];
    for incident in graph
        .neighbors(active)
        .expect("active atom must have an adjacency row")
        .iter()
        .copied()
    {
        if progress.classify_incident(graph, active, incident)
            != IncidentBondState::UnrepresentedToUnvisitedAtom
        {
            continue;
        }
        let component = component_by_atom[incident.atom().index()];
        assert_ne!(
            component,
            usize::MAX,
            "an unvisited incident endpoint must belong to a residual component"
        );
        groups[component].push(incident);
    }

    let mut attachments = groups
        .into_iter()
        .enumerate()
        .filter(|(_, incidences)| !incidences.is_empty())
        .map(|(component, incidences)| {
            (component_mins[component], ResidualAttachment { incidences })
        })
        .collect::<Vec<_>>();
    attachments.sort_by_key(|(minimum, _)| *minimum);
    attachments
        .into_iter()
        .map(|(_, attachment)| attachment)
        .collect()
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
    fn triangle_root_owns_one_residual_attachment() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let left = builder.add_bond(atoms[0], atoms[1]).unwrap();
        let right = builder.add_bond(atoms[0], atoms[2]).unwrap();
        builder.add_bond(atoms[1], atoms[2]).unwrap();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(&graph, atoms[0]);

        assert_eq!(state.active_attachments().len(), 1);
        assert_eq!(
            state.active_attachments()[0].incidences(),
            &[
                incident(&graph, atoms[0], left),
                incident(&graph, atoms[0], right),
            ]
        );
    }

    #[test]
    fn star_root_owns_one_attachment_per_leaf() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 4] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let bonds = [
            builder.add_bond(atoms[0], atoms[1]).unwrap(),
            builder.add_bond(atoms[0], atoms[2]).unwrap(),
            builder.add_bond(atoms[0], atoms[3]).unwrap(),
        ];
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(&graph, atoms[0]);

        assert_eq!(state.active_attachments().len(), 3);
        for (attachment, bond) in state.active_attachments().iter().zip(bonds) {
            assert_eq!(attachment.incidences(), &[incident(&graph, atoms[0], bond)]);
        }
    }

    #[test]
    fn branch_return_restores_retained_parent_attachments() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let branch = builder.add_bond(atoms[0], atoms[1]).unwrap();
        let inline = builder.add_bond(atoms[0], atoms[2]).unwrap();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(&graph, atoms[0]);
        state.enter_branch_child(&graph, incident(&graph, atoms[0], branch));
        assert_eq!(state.complete_path(), Some(atoms[0]));
        assert_eq!(state.active_attachments().len(), 1);
        assert_eq!(
            state.active_attachments()[0].incidences(),
            &[incident(&graph, atoms[0], inline)]
        );
    }

    #[test]
    fn ring_opening_reduces_but_does_not_remove_an_attachment() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let left = builder.add_bond(atoms[0], atoms[1]).unwrap();
        let right = builder.add_bond(atoms[0], atoms[2]).unwrap();
        builder.add_bond(atoms[1], atoms[2]).unwrap();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(&graph, atoms[0]);
        state.open_ring_endpoint(&graph, incident(&graph, atoms[0], left));

        assert_eq!(state.active_attachments().len(), 1);
        assert_eq!(
            state.active_attachments()[0].incidences(),
            &[incident(&graph, atoms[0], right)]
        );
    }

    #[test]
    fn ring_endpoint_lifecycle_preserves_its_label_slot() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let first = builder.add_bond(atoms[0], atoms[1]).unwrap();
        let second = builder.add_bond(atoms[1], atoms[2]).unwrap();
        let ring = builder.add_bond(atoms[2], atoms[0]).unwrap();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(&graph, atoms[0]);
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
        assert!(state.graph_is_complete());
        assert_eq!(state.complete_path(), None);
    }

    #[test]
    fn ring_label_slots_do_not_reuse_until_the_atom_is_left() {
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
        let later_ring = builder.add_bond(atoms[2], atoms[4]).unwrap();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(&graph, atoms[0]);
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
        assert_eq!(state.next_ring_label_slot().index(), 2);
        assert_eq!(
            state
                .open_ring_endpoint(&graph, incident(&graph, atoms[2], later_ring))
                .index(),
            2
        );
        state.enter_inline_child(&graph, incident(&graph, atoms[2], path[2]));
        assert_eq!(state.next_ring_label_slot().index(), 0);
        state.enter_inline_child(&graph, incident(&graph, atoms[3], path[3]));

        assert_eq!(
            state
                .close_ring_endpoint(&graph, incident(&graph, atoms[4], long_ring))
                .index(),
            1
        );
        assert_eq!(
            state
                .close_ring_endpoint(&graph, incident(&graph, atoms[4], later_ring))
                .index(),
            2
        );
        assert!(state.graph_is_complete());
        assert_eq!(state.next_ring_label_slot().index(), 0);
        assert_eq!(state.complete_path(), None);
    }
}
