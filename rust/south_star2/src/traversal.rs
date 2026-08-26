//! Graph-general control state for the South Star 2 writer.
//!
//! This module owns evolving graph-representation facts plus active and
//! suspended writer frames. Concrete SMILES spelling, including ring-label
//! assignment, remains outside structural traversal state.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use crate::ids::{AtomId, BondId};
use crate::persistent::PagedStore;
use crate::prepared::{AdjacentBond, PreparedGraph};

#[derive(Clone, Debug, PartialEq, Eq)]
struct DenseSet {
    universe_len: usize,
    marked_count: usize,
    words: PagedStore<u64>,
}

impl DenseSet {
    fn new(universe_len: usize) -> Self {
        let word_count = universe_len.div_ceil(u64::BITS as usize);
        Self {
            universe_len,
            marked_count: 0,
            words: PagedStore::filled(word_count, 0),
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum BondProgress {
    Unrepresented,
    Traversed {
        from: AtomId,
        to: AtomId,
    },
    RingOpen {
        first_endpoint: AtomId,
    },
    RingClosed {
        first_endpoint: AtomId,
        second_endpoint: AtomId,
    },
}

#[cfg(test)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ObservedBondProgress {
    Unrepresented,
    Traversed {
        from: AtomId,
        to: AtomId,
    },
    RingOpen {
        first_endpoint: AtomId,
    },
    RingClosed {
        first_endpoint: AtomId,
        second_endpoint: AtomId,
    },
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ObservedFrame {
    pub(crate) atom: AtomId,
    pub(crate) entry_bond: Option<BondId>,
    pub(crate) emitted_bonds: Vec<BondId>,
    pub(crate) ring_occurrence_count: usize,
    pub(crate) attachment_groups: Vec<Vec<AdjacentBond>>,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ObservedTraversalState {
    pub(crate) visited_atoms: Vec<AtomId>,
    pub(crate) bond_progress: Vec<ObservedBondProgress>,
    pub(crate) active_frame: Option<ObservedFrame>,
    pub(crate) branch_returns: Vec<ObservedFrame>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GraphProgress {
    visited_atoms: DenseSet,
    bonds: PagedStore<BondProgress>,
    represented_bond_count: usize,
    open_ring_count: usize,
}

impl GraphProgress {
    fn new(graph: &PreparedGraph) -> Self {
        Self {
            visited_atoms: DenseSet::new(graph.atom_count()),
            bonds: PagedStore::filled(graph.bond_count(), BondProgress::Unrepresented),
            represented_bond_count: 0,
            open_ring_count: 0,
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

    fn open_ring(&mut self, bond: BondId, first_endpoint: AtomId) {
        let progress = self.bond_progress_mut(bond);
        assert_eq!(
            *progress,
            BondProgress::Unrepresented,
            "a ring bond must be unrepresented when its first endpoint is written"
        );
        *progress = BondProgress::RingOpen { first_endpoint };
        self.open_ring_count += 1;
    }

    fn close_ring(&mut self, bond: BondId, second_endpoint: AtomId) {
        let progress = self.bond_progress_mut(bond);
        let BondProgress::RingOpen { first_endpoint } = *progress else {
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
        self.open_ring_count -= 1;
        self.represented_bond_count += 1;
    }

    const fn is_complete(&self) -> bool {
        self.visited_atoms.is_complete() && self.represented_bond_count == self.bonds.len()
    }

    const fn has_visited_atoms(&self) -> bool {
        self.visited_atoms.marked_count != 0
    }

    const fn has_open_rings(&self) -> bool {
        self.open_ring_count != 0
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
            BondProgress::RingOpen { first_endpoint } if *first_endpoint == at => {
                IncidentBondState::RingOpenAtCurrentAtom
            }
            BondProgress::RingOpen { first_endpoint } => {
                assert_eq!(
                    *first_endpoint,
                    incident.atom(),
                    "open ring endpoint must belong to one endpoint of its bond"
                );
                IncidentBondState::RingOpenAtOtherAtom
            }
        }
    }

    fn ring_first_endpoint(&self, bond: BondId) -> Option<AtomId> {
        match self.bond_progress(bond) {
            BondProgress::RingOpen { first_endpoint } => Some(*first_endpoint),
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

/// Incidences from one writer frame into one connected component of the
/// currently unvisited induced graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResidualAttachment {
    incidences: Vec<AdjacentBond>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResidualComponent {
    ordered_members: Arc<[AtomId]>,
    minimum_offset: usize,
    atom_count: usize,
}

impl ResidualComponent {
    fn minimum(&self) -> AtomId {
        self.ordered_members[self.minimum_offset]
    }
}

#[derive(Clone, Debug)]
struct ResidualPartition {
    component_by_atom: PagedStore<Option<usize>>,
    components: PagedStore<Option<Arc<ResidualComponent>>>,
    #[cfg(test)]
    atoms_scanned_while_splitting: usize,
    #[cfg(test)]
    minimum_members_skipped: usize,
}

impl PartialEq for ResidualPartition {
    fn eq(&self, other: &Self) -> bool {
        self.component_by_atom == other.component_by_atom && self.components == other.components
    }
}

impl Eq for ResidualPartition {}

impl ResidualPartition {
    fn new(graph: &PreparedGraph) -> Self {
        let mut component_by_atom = vec![None; graph.atom_count()];
        let mut components = vec![None; graph.atom_count()];

        for prepared in graph.components() {
            let component = prepared
                .atoms()
                .first()
                .expect("a prepared component must contain an atom")
                .index();
            for &atom in prepared.atoms() {
                component_by_atom[atom.index()] = Some(component);
            }
            components[component] = Some(Arc::new(ResidualComponent {
                ordered_members: Arc::from(prepared.atoms()),
                minimum_offset: 0,
                atom_count: prepared.atoms().len(),
            }));
        }

        Self {
            component_by_atom: PagedStore::from_values(component_by_atom),
            components: PagedStore::from_values(components),
            #[cfg(test)]
            atoms_scanned_while_splitting: 0,
            #[cfg(test)]
            minimum_members_skipped: 0,
        }
    }

    fn atom_is_unvisited(&self, atom: AtomId) -> bool {
        self.component_by_atom
            .get(atom.index())
            .copied()
            .flatten()
            .is_some()
    }

    fn remove_atom(&mut self, graph: &PreparedGraph, atom: AtomId) {
        let component_id = self.component_by_atom[atom.index()]
            .take()
            .expect("an entered atom must belong to one live residual component");
        let mut component = self.components[component_id]
            .take()
            .expect("a live atom component identifier must resolve");

        let residual_neighbours = graph
            .neighbors(atom)
            .expect("entered atom must have an adjacency row")
            .iter()
            .filter(|incident| {
                self.component_by_atom[incident.atom().index()] == Some(component_id)
            })
            .map(|incident| incident.atom())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        match residual_neighbours.len() {
            0 => assert!(
                component.atom_count == 1,
                "a connected residual component cannot retain atoms beyond an isolated deletion"
            ),
            1 => {
                let component_state = Arc::make_mut(&mut component);
                component_state.atom_count -= 1;
                while self.component_by_atom[component_state.minimum().index()]
                    != Some(component_id)
                {
                    component_state.minimum_offset += 1;
                    #[cfg(test)]
                    {
                        self.minimum_members_skipped += 1;
                    }
                }
                self.components[component_id] = Some(component);
            }
            _ => self.split_component(graph, component_id, component, residual_neighbours),
        }

        #[cfg(test)]
        self.assert_consistent(graph);
    }

    fn split_component(
        &mut self,
        graph: &PreparedGraph,
        original_id: usize,
        component: Arc<ResidualComponent>,
        residual_neighbours: Vec<AtomId>,
    ) {
        #[cfg(test)]
        {
            self.atoms_scanned_while_splitting += component.atom_count - 1;
        }

        let mut discovered = BTreeSet::new();
        let mut groups = Vec::new();
        for root in residual_neighbours {
            if !discovered.insert(root) {
                continue;
            }
            let mut atoms = BTreeSet::from([root]);
            let mut pending = VecDeque::from([root]);
            while let Some(current) = pending.pop_front() {
                for incident in graph
                    .neighbors(current)
                    .expect("residual atom must have an adjacency row")
                {
                    let neighbour = incident.atom();
                    if self.component_by_atom[neighbour.index()] != Some(original_id)
                        || !discovered.insert(neighbour)
                    {
                        continue;
                    }
                    atoms.insert(neighbour);
                    pending.push_back(neighbour);
                }
            }
            groups.push(atoms);
        }
        assert_eq!(
            discovered.len(),
            component.atom_count - 1,
            "residual neighbours must reach the entire affected component after deletion"
        );

        for (offset, atoms) in groups.into_iter().enumerate() {
            let component_id = atoms
                .first()
                .expect("a split residual component must contain an atom")
                .index();
            if offset == 0 {
                debug_assert!(self.components[original_id].is_none());
            }
            for &member in &atoms {
                self.component_by_atom[member.index()] = Some(component_id);
            }
            assert!(self.components[component_id].is_none());
            let atom_count = atoms.len();
            self.components[component_id] = Some(Arc::new(ResidualComponent {
                ordered_members: atoms.into_iter().collect::<Vec<_>>().into(),
                minimum_offset: 0,
                atom_count,
            }));
        }
    }

    fn attachments(
        &self,
        graph: &PreparedGraph,
        progress: &GraphProgress,
        active: AtomId,
    ) -> Vec<ResidualAttachment> {
        let mut groups = BTreeMap::<(AtomId, usize), Vec<AdjacentBond>>::new();
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
            let component_id = self.component_by_atom[incident.atom().index()]
                .expect("an unvisited endpoint must belong to a live residual component");
            let component = self
                .components
                .get(component_id)
                .and_then(Option::as_ref)
                .expect("a live residual component identifier must resolve");
            groups
                .entry((component.minimum(), component_id))
                .or_default()
                .push(incident);
        }
        groups
            .into_values()
            .map(|incidences| ResidualAttachment { incidences })
            .collect()
    }

    #[cfg(test)]
    fn assert_consistent(&self, graph: &PreparedGraph) {
        for atom in graph.atom_ids() {
            let component = self.component_by_atom[atom.index()];
            if let Some(component) = component {
                assert!(self.components[component]
                    .as_ref()
                    .expect("live component identifier must resolve")
                    .ordered_members
                    .contains(&atom));
            }
        }
        for component in 0..self.components.len() {
            let Some(members) = self.components[component].as_ref() else {
                continue;
            };
            let live_members = graph
                .atom_ids()
                .filter(|atom| self.component_by_atom[atom.index()] == Some(component))
                .collect::<Vec<_>>();
            assert_eq!(live_members.len(), members.atom_count);
            assert_eq!(live_members.first().copied(), Some(members.minimum()));
            assert!(live_members
                .iter()
                .all(|atom| members.ordered_members.contains(atom)));
        }
    }
}

impl ResidualAttachment {
    pub(crate) fn incidences(&self) -> &[AdjacentBond] {
        &self.incidences
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WriterFrame {
    atom: AtomId,
    entry_bond: Option<BondId>,
    emitted_bonds: Vec<BondId>,
    ring_occurrence_count: usize,
    attachments: Vec<ResidualAttachment>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LocalBondOrder {
    pub(crate) atom: AtomId,
    pub(crate) entry_bond: Option<BondId>,
    pub(crate) emitted_bonds: Vec<BondId>,
    pub(crate) ring_occurrence_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FrameNode {
    frame: Arc<WriterFrame>,
    parent: Option<Arc<FrameNode>>,
}

impl WriterFrame {
    fn new(
        graph: &PreparedGraph,
        progress: &GraphProgress,
        residual: &ResidualPartition,
        atom: AtomId,
        entry_bond: Option<BondId>,
    ) -> Self {
        let attachments = residual.attachments(graph, progress, atom);
        #[cfg(test)]
        assert_eq!(
            attachments,
            residual_attachments_full(graph, progress, atom),
            "incremental residual partition must match full recomputation"
        );
        Self {
            atom,
            entry_bond,
            emitted_bonds: Vec::new(),
            ring_occurrence_count: 0,
            attachments,
        }
    }

    fn commit_child(&mut self, incident: AdjacentBond) {
        assert!(
            self.attachments.iter().any(|attachment| {
                attachment.incidences.len() == 1 && attachment.incidences[0] == incident
            }),
            "a committed child must be the sole incidence of one active attachment"
        );
        assert!(
            self.entry_bond != Some(incident.bond())
                && !self.emitted_bonds.contains(&incident.bond()),
            "one local bond occurrence may be committed only once"
        );
        self.emitted_bonds.push(incident.bond());
    }

    fn commit_ring(&mut self, incident: AdjacentBond) {
        assert_eq!(
            self.ring_occurrence_count,
            self.emitted_bonds.len(),
            "ring occurrences must precede traversal-child occurrences"
        );
        assert!(
            self.entry_bond != Some(incident.bond())
                && !self.emitted_bonds.contains(&incident.bond()),
            "one local bond occurrence may be committed only once"
        );
        self.emitted_bonds.push(incident.bond());
        self.ring_occurrence_count += 1;
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PathCompletion {
    CloseBranch,
    FinishComponent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TraversalState {
    progress: GraphProgress,
    residual: ResidualPartition,
    active: Option<Arc<WriterFrame>>,
    branch_returns: Option<Arc<FrameNode>>,
}

impl TraversalState {
    pub(crate) fn new(graph: &PreparedGraph) -> Self {
        Self {
            progress: GraphProgress::new(graph),
            residual: ResidualPartition::new(graph),
            active: None,
            branch_returns: None,
        }
    }

    pub(crate) fn active_atom(&self) -> Option<AtomId> {
        self.active.as_ref().map(|frame| frame.atom)
    }

    pub(crate) const fn graph_is_complete(&self) -> bool {
        self.progress.is_complete()
    }

    pub(crate) const fn has_visited_atoms(&self) -> bool {
        self.progress.has_visited_atoms()
    }

    #[cfg(test)]
    pub(crate) fn atom_is_visited(&self, atom: AtomId) -> bool {
        self.progress.atom_is_visited(atom)
    }

    #[cfg(test)]
    pub(crate) fn observe_raw(&self) -> ObservedTraversalState {
        let visited_atoms = (0..self.progress.visited_atoms.universe_len)
            .filter(|index| self.progress.visited_atoms.contains(*index))
            .map(|index| AtomId::new(u32::try_from(index).unwrap()))
            .collect();
        let bond_progress = (0..self.progress.bonds.len())
            .map(|index| match self.progress.bonds[index] {
                BondProgress::Unrepresented => ObservedBondProgress::Unrepresented,
                BondProgress::Traversed { from, to } => {
                    ObservedBondProgress::Traversed { from, to }
                }
                BondProgress::RingOpen { first_endpoint } => {
                    ObservedBondProgress::RingOpen { first_endpoint }
                }
                BondProgress::RingClosed {
                    first_endpoint,
                    second_endpoint,
                } => ObservedBondProgress::RingClosed {
                    first_endpoint,
                    second_endpoint,
                },
            })
            .collect();
        let observe_frame = |frame: &WriterFrame| ObservedFrame {
            atom: frame.atom,
            entry_bond: frame.entry_bond,
            emitted_bonds: frame.emitted_bonds.clone(),
            ring_occurrence_count: frame.ring_occurrence_count,
            attachment_groups: frame
                .attachments
                .iter()
                .map(|attachment| attachment.incidences.clone())
                .collect(),
        };
        let active_frame = self.active.as_deref().map(&observe_frame);
        let mut branch_returns = Vec::new();
        let mut cursor = self.branch_returns.as_deref();
        while let Some(node) = cursor {
            branch_returns.push(observe_frame(node.frame.as_ref()));
            cursor = node.parent.as_deref();
        }
        ObservedTraversalState {
            visited_atoms,
            bond_progress,
            active_frame,
            branch_returns,
        }
    }

    pub(crate) fn path_completion(&self) -> Option<PathCompletion> {
        self.active.as_ref()?;
        if self.branch_returns.is_some() {
            Some(PathCompletion::CloseBranch)
        } else if self.progress.has_open_rings() {
            None
        } else {
            Some(PathCompletion::FinishComponent)
        }
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
            .filter(|atom| self.residual.atom_is_unvisited(*atom))
    }

    pub(crate) fn begin_component(&mut self, graph: &PreparedGraph, root: AtomId) {
        assert!(
            self.active.is_none()
                && self.branch_returns.is_none()
                && !self.progress.has_open_rings(),
            "a component can begin only after the previous component is closed"
        );
        self.progress.visit_atom(root);
        self.residual.remove_atom(graph, root);
        self.active = Some(Arc::new(WriterFrame::new(
            graph,
            &self.progress,
            &self.residual,
            root,
            None,
        )));
    }

    pub(crate) fn active_local_bond_order(&self) -> LocalBondOrder {
        let frame = self
            .active
            .as_ref()
            .expect("local bond order requires an active writer frame");
        LocalBondOrder {
            atom: frame.atom,
            entry_bond: frame.entry_bond,
            emitted_bonds: frame.emitted_bonds.clone(),
            ring_occurrence_count: frame.ring_occurrence_count,
        }
    }

    pub(crate) fn commit_active_child(&mut self, incident: AdjacentBond) {
        Arc::make_mut(
            self.active
                .as_mut()
                .expect("child commitment requires an active writer frame"),
        )
        .commit_child(incident);
    }

    pub(crate) fn enter_inline_child(&mut self, graph: &PreparedGraph, incident: AdjacentBond) {
        self.enter_child(graph, incident, ChildPlacement::Inline);
    }

    pub(crate) fn enter_branch_child(&mut self, graph: &PreparedGraph, incident: AdjacentBond) {
        self.enter_child(graph, incident, ChildPlacement::Branch);
    }

    pub(crate) fn open_ring_endpoint(&mut self, graph: &PreparedGraph, incident: AdjacentBond) {
        let active_atom = self
            .active_atom()
            .expect("a ring endpoint requires an active atom");
        assert_eq!(
            self.progress
                .classify_incident(graph, active_atom, incident),
            IncidentBondState::UnrepresentedToUnvisitedAtom,
            "a first ring endpoint must be written before its other atom is visited"
        );
        Arc::make_mut(
            self.active
                .as_mut()
                .expect("ring opening requires an active frame"),
        )
        .commit_ring(incident);
        Arc::make_mut(
            self.active
                .as_mut()
                .expect("ring opening requires an active frame"),
        )
        .remove_ring_incidence(incident);
        self.progress.open_ring(incident.bond(), active_atom);
    }

    pub(crate) fn close_ring_endpoint(&mut self, graph: &PreparedGraph, incident: AdjacentBond) {
        let active_atom = self
            .active_atom()
            .expect("a ring endpoint requires an active atom");
        assert_eq!(
            self.progress
                .classify_incident(graph, active_atom, incident),
            IncidentBondState::RingOpenAtOtherAtom,
            "a second ring endpoint must pair a bond opened at its other atom"
        );
        Arc::make_mut(
            self.active
                .as_mut()
                .expect("ring closure requires an active frame"),
        )
        .commit_ring(incident);
        self.progress.close_ring(incident.bond(), active_atom);
    }

    pub(crate) fn ring_first_endpoint_for_active_incident(
        &self,
        graph: &PreparedGraph,
        incident: AdjacentBond,
    ) -> Option<AtomId> {
        let active_atom = self
            .active_atom()
            .expect("ring-endpoint lookup requires an active atom");
        self.progress
            .classify_incident(graph, active_atom, incident);
        self.progress.ring_first_endpoint(incident.bond())
    }

    pub(crate) fn complete_path(&mut self, _graph: &PreparedGraph) -> Option<AtomId> {
        let active = self.active.take().expect("no active path to complete");
        assert!(
            active.attachments.is_empty(),
            "a path cannot complete with unresolved residual attachments"
        );
        let restored = self.branch_returns.take().map(|node| {
            self.branch_returns = node.parent.clone();
            Arc::clone(&node.frame)
        });
        if restored.is_none() {
            assert!(
                !self.progress.has_open_rings(),
                "a component cannot end with an unpaired ring endpoint"
            );
        }
        self.active = restored;
        #[cfg(test)]
        if let Some(active) = &self.active {
            assert_eq!(
                active.attachments,
                self.residual
                    .attachments(_graph, &self.progress, active.atom),
                "a restored parent frame must match the live residual partition"
            );
            assert_eq!(
                active.attachments,
                residual_attachments_full(_graph, &self.progress, active.atom),
                "a restored parent frame must match full residual recomputation"
            );
        }
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
        Arc::make_mut(&mut parent).consume_child_attachment(incident);
        assert_eq!(
            parent.emitted_bonds.last(),
            Some(&incident.bond()),
            "a child must be committed in local order before its atom is entered"
        );
        let parent_atom = parent.atom;

        match placement {
            ChildPlacement::Branch => {
                assert!(
                    !parent.attachments.is_empty(),
                    "a branch requires another residual attachment for the main continuation"
                );
                self.branch_returns = Some(Arc::new(FrameNode {
                    frame: parent,
                    parent: self.branch_returns.take(),
                }));
            }
            ChildPlacement::Inline => {
                assert!(
                    parent.attachments.is_empty(),
                    "the inline child must consume the final residual attachment"
                );
            }
        }

        let child = incident.atom();
        self.progress
            .traverse_bond(incident.bond(), parent_atom, child);
        self.progress.visit_atom(child);
        self.residual.remove_atom(graph, child);
        self.active = Some(Arc::new(WriterFrame::new(
            graph,
            &self.progress,
            &self.residual,
            child,
            Some(incident.bond()),
        )));
    }

    #[cfg(test)]
    fn residual_split_scan_count(&self) -> usize {
        self.residual.atoms_scanned_while_splitting
    }

    #[cfg(test)]
    fn residual_minimum_skip_count(&self) -> usize {
        self.residual.minimum_members_skipped
    }
}

#[cfg(test)]
fn residual_attachments_full(
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

    fn graph_fixture(atom_count: usize, edges: &[(usize, usize)]) -> PreparedGraph {
        let mut builder = PreparedGraphBuilder::new();
        let atoms = (0..atom_count)
            .map(|_| builder.add_atom().unwrap())
            .collect::<Vec<_>>();
        for &(a, b) in edges {
            builder.add_bond(atoms[a], atoms[b]).unwrap();
        }
        builder.build()
    }

    fn permutations(values: &[AtomId]) -> Vec<Vec<AtomId>> {
        fn extend(
            prefix: &mut Vec<AtomId>,
            remaining: &mut Vec<AtomId>,
            out: &mut Vec<Vec<AtomId>>,
        ) {
            if remaining.is_empty() {
                out.push(prefix.clone());
                return;
            }
            for index in 0..remaining.len() {
                let value = remaining.remove(index);
                prefix.push(value);
                extend(prefix, remaining, out);
                prefix.pop();
                remaining.insert(index, value);
            }
        }

        let mut out = Vec::new();
        extend(&mut Vec::new(), &mut values.to_vec(), &mut out);
        out
    }

    fn full_residual_components(
        graph: &PreparedGraph,
        deleted: &BTreeSet<AtomId>,
    ) -> Vec<Vec<AtomId>> {
        let mut unseen = graph
            .atom_ids()
            .filter(|atom| !deleted.contains(atom))
            .collect::<BTreeSet<_>>();
        let mut components = Vec::new();

        while let Some(&root) = unseen.first() {
            unseen.remove(&root);
            let mut component = vec![root];
            let mut pending = VecDeque::from([root]);
            while let Some(atom) = pending.pop_front() {
                for incident in graph.neighbors(atom).unwrap() {
                    if unseen.remove(&incident.atom()) {
                        component.push(incident.atom());
                        pending.push_back(incident.atom());
                    }
                }
            }
            component.sort_unstable();
            components.push(component);
        }
        components.sort();
        components
    }

    fn incremental_components(residual: &ResidualPartition) -> Vec<Vec<AtomId>> {
        let mut components = (0..residual.components.len())
            .filter(|index| residual.components[*index].is_some())
            .map(|component| {
                (0..residual.component_by_atom.len())
                    .filter(|atom| residual.component_by_atom[*atom] == Some(component))
                    .map(|atom| AtomId::new(u32::try_from(atom).unwrap()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        components.sort();
        components
    }

    fn enter_inline(state: &mut TraversalState, graph: &PreparedGraph, incident: AdjacentBond) {
        state.commit_active_child(incident);
        state.enter_inline_child(graph, incident);
    }

    fn enter_branch(state: &mut TraversalState, graph: &PreparedGraph, incident: AdjacentBond) {
        state.commit_active_child(incident);
        state.enter_branch_child(graph, incident);
    }

    #[test]
    fn residual_partition_matches_full_recomputation_for_bounded_deletions() {
        let fixtures = [
            ("path", graph_fixture(5, &[(0, 1), (1, 2), (2, 3), (3, 4)])),
            ("star", graph_fixture(5, &[(0, 1), (0, 2), (0, 3), (0, 4)])),
            (
                "cycle",
                graph_fixture(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
            ),
            (
                "articulation",
                graph_fixture(5, &[(0, 1), (1, 2), (2, 0), (0, 3), (3, 4)]),
            ),
            (
                "fused cycles",
                graph_fixture(5, &[(0, 1), (1, 2), (2, 0), (1, 3), (2, 3), (3, 4)]),
            ),
            (
                "disconnected mixture",
                graph_fixture(5, &[(0, 1), (1, 2), (2, 0), (3, 4)]),
            ),
        ];

        for (name, graph) in fixtures {
            let atoms = graph.atom_ids().collect::<Vec<_>>();
            for order in permutations(&atoms) {
                let mut residual = ResidualPartition::new(&graph);
                let mut deleted = BTreeSet::new();
                for atom in order {
                    residual.remove_atom(&graph, atom);
                    assert!(deleted.insert(atom));
                    assert_eq!(
                        incremental_components(&residual),
                        full_residual_components(&graph, &deleted),
                        "{name} after deleting {atom:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn endpoint_path_walk_never_scans_a_residual_component() {
        const ATOM_COUNT: usize = 64;

        let mut builder = PreparedGraphBuilder::new();
        let atoms = (0..ATOM_COUNT)
            .map(|_| builder.add_atom().unwrap())
            .collect::<Vec<_>>();
        let bonds = (0..ATOM_COUNT - 1)
            .map(|index| builder.add_bond(atoms[index], atoms[index + 1]).unwrap())
            .collect::<Vec<_>>();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);
        state.begin_component(&graph, atoms[0]);
        for index in 0..ATOM_COUNT - 1 {
            enter_inline(
                &mut state,
                &graph,
                incident(&graph, atoms[index], bonds[index]),
            );
        }

        assert_eq!(state.residual_split_scan_count(), 0);
    }

    #[test]
    fn adversarial_path_minimum_work_is_counted_and_amortized() {
        const ATOM_COUNT: usize = 66;

        let mut builder = PreparedGraphBuilder::new();
        let atoms = (0..ATOM_COUNT)
            .map(|_| builder.add_atom().unwrap())
            .collect::<Vec<_>>();
        let order = (1..ATOM_COUNT - 1)
            .chain([0, ATOM_COUNT - 1])
            .collect::<Vec<_>>();
        let bonds = order
            .windows(2)
            .map(|pair| builder.add_bond(atoms[pair[0]], atoms[pair[1]]).unwrap())
            .collect::<Vec<_>>();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);
        state.begin_component(&graph, atoms[order[0]]);
        for (bond, pair) in bonds.into_iter().zip(order.windows(2)) {
            enter_inline(&mut state, &graph, incident(&graph, atoms[pair[0]], bond));
        }

        assert_eq!(state.residual_split_scan_count(), 0);
        assert_eq!(state.residual_minimum_skip_count(), ATOM_COUNT - 1);
    }

    #[test]
    fn one_path_step_copies_only_touched_graph_pages() {
        const ATOM_COUNT: usize = 130;

        let mut builder = PreparedGraphBuilder::new();
        let atoms = (0..ATOM_COUNT)
            .map(|_| builder.add_atom().unwrap())
            .collect::<Vec<_>>();
        let bonds = (0..ATOM_COUNT - 1)
            .map(|index| builder.add_bond(atoms[index], atoms[index + 1]).unwrap())
            .collect::<Vec<_>>();
        let graph = builder.build();
        let mut source = TraversalState::new(&graph);
        source.begin_component(&graph, atoms[0]);
        let mut successor = source.clone();

        source.progress.visited_atoms.words.reset_copy_counts();
        source.progress.bonds.reset_copy_counts();
        source.residual.component_by_atom.reset_copy_counts();
        source.residual.components.reset_copy_counts();
        enter_inline(&mut successor, &graph, incident(&graph, atoms[0], bonds[0]));

        assert_eq!(source.progress.visited_atoms.words.copy_counts(), (0, 1));
        assert_eq!(source.progress.bonds.copy_counts(), (1, 1));
        assert_eq!(source.residual.component_by_atom.copy_counts(), (1, 1));
        assert_eq!(source.residual.components.copy_counts(), (1, 1));
        assert!(source
            .progress
            .bonds
            .shares_value_page_with(&successor.progress.bonds, bonds[128].index()));
        assert!(source
            .residual
            .component_by_atom
            .shares_value_page_with(&successor.residual.component_by_atom, atoms[129].index()));
        let source_component = source.residual.components[0].as_ref().unwrap();
        let successor_component = successor.residual.components[0].as_ref().unwrap();
        assert!(Arc::ptr_eq(
            &source_component.ordered_members,
            &successor_component.ordered_members
        ));
        assert_eq!(source.active_atom(), Some(atoms[0]));
        assert_eq!(successor.active_atom(), Some(atoms[1]));
    }

    #[test]
    fn branch_forks_and_returns_share_suspended_ancestors() {
        let graph = graph_fixture(5, &[(0, 1), (0, 2), (1, 3), (1, 4)]);
        let atoms = graph.atom_ids().collect::<Vec<_>>();
        let mut state = TraversalState::new(&graph);
        state.begin_component(&graph, atoms[0]);
        let root_branch = graph.neighbors(atoms[0]).unwrap()[0];
        enter_branch(&mut state, &graph, root_branch);
        let nested_branch = graph
            .neighbors(atoms[1])
            .unwrap()
            .iter()
            .copied()
            .find(|incident| incident.atom() == atoms[3])
            .unwrap();
        enter_branch(&mut state, &graph, nested_branch);

        let fork = state.clone();
        let state_top = state.branch_returns.as_ref().unwrap();
        let fork_top = fork.branch_returns.as_ref().unwrap();
        assert!(Arc::ptr_eq(state_top, fork_top));
        assert!(Arc::ptr_eq(&state_top.frame, &fork_top.frame));
        assert!(Arc::ptr_eq(
            state_top.parent.as_ref().unwrap(),
            fork_top.parent.as_ref().unwrap()
        ));

        assert_eq!(state.complete_path(&graph), Some(atoms[1]));
        assert!(Arc::ptr_eq(
            state.branch_returns.as_ref().unwrap(),
            fork_top.parent.as_ref().unwrap()
        ));
        assert!(Arc::ptr_eq(state.active.as_ref().unwrap(), &fork_top.frame));
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
        enter_branch(&mut state, &graph, incident(&graph, atoms[0], branch));
        assert_eq!(state.complete_path(&graph), Some(atoms[0]));
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
    fn ring_endpoint_lifecycle_retains_only_structural_facts() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let first = builder.add_bond(atoms[0], atoms[1]).unwrap();
        let second = builder.add_bond(atoms[1], atoms[2]).unwrap();
        let ring = builder.add_bond(atoms[2], atoms[0]).unwrap();
        let graph = builder.build();
        let mut state = TraversalState::new(&graph);

        state.begin_component(&graph, atoms[0]);
        let opening = incident(&graph, atoms[0], ring);
        state.open_ring_endpoint(&graph, opening);
        assert_eq!(
            state.active_local_bond_order(),
            LocalBondOrder {
                atom: atoms[0],
                entry_bond: None,
                emitted_bonds: vec![ring],
                ring_occurrence_count: 1,
            }
        );
        assert_eq!(
            state.ring_first_endpoint_for_active_incident(&graph, opening),
            Some(atoms[0])
        );

        enter_inline(&mut state, &graph, incident(&graph, atoms[0], first));
        enter_inline(&mut state, &graph, incident(&graph, atoms[1], second));
        let closing = incident(&graph, atoms[2], ring);
        assert_eq!(
            state.ring_first_endpoint_for_active_incident(&graph, closing),
            Some(atoms[0])
        );
        state.close_ring_endpoint(&graph, closing);
        assert_eq!(
            state.active_local_bond_order(),
            LocalBondOrder {
                atom: atoms[2],
                entry_bond: Some(second),
                emitted_bonds: vec![ring],
                ring_occurrence_count: 1,
            }
        );
        assert!(state.graph_is_complete());
        assert_eq!(state.complete_path(&graph), None);
    }
}
