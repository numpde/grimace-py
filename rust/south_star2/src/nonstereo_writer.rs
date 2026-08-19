//! Connected non-stereo visible-token state.
//!
//! Graph and constraint semantics live in `WriterState`. This module owns only
//! concrete non-stereo spelling facts, live ring-label assignments, and the
//! small lexical commitments forced by multi-token SMILES constructs.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use crate::ids::{AtomId, BondId};
use crate::prepared::{AdjacentBond, PreparedBond, PreparedGraph, PreparedMolecule};
use crate::solver::ConstraintSolver;
use crate::writer_state::{StructuralFrontier, WriterState};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum NonStereoBondToken {
    Elided,
    Aromatic,
    Single,
    Double,
    Triple,
    DativeAToB,
    DativeBToA,
}

impl NonStereoBondToken {
    fn text_from(self, bond: PreparedBond, from: AtomId) -> &'static str {
        let from_a = if bond.a() == from {
            true
        } else if bond.b() == from {
            false
        } else {
            panic!("bond text requires one endpoint of the prepared bond");
        };

        match self {
            Self::Elided => "",
            Self::Aromatic => ":",
            Self::Single => "-",
            Self::Double => "=",
            Self::Triple => "#",
            Self::DativeAToB if from_a => "->",
            Self::DativeAToB => "<-",
            Self::DativeBToA if from_a => "<-",
            Self::DativeBToA => "->",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedConnectedNonStereo {
    molecule: PreparedMolecule,
    atom_text: Arc<[Box<str>]>,
    bond_tokens: Arc<[NonStereoBondToken]>,
}

impl PreparedConnectedNonStereo {
    pub(crate) fn new(
        molecule: PreparedMolecule,
        atom_text: Vec<String>,
        bond_tokens: Vec<NonStereoBondToken>,
    ) -> Result<Self, PreparedConnectedNonStereoError> {
        let graph = molecule.graph();
        if graph.atom_count() == 0 {
            return Err(PreparedConnectedNonStereoError::EmptyMolecule);
        }
        if !graph_is_connected(graph) {
            return Err(PreparedConnectedNonStereoError::DisconnectedMolecule);
        }
        if atom_text.len() != graph.atom_count() {
            return Err(PreparedConnectedNonStereoError::AtomTextCountMismatch {
                expected: graph.atom_count(),
                actual: atom_text.len(),
            });
        }
        if bond_tokens.len() != graph.bond_count() {
            return Err(PreparedConnectedNonStereoError::BondTokenCountMismatch {
                expected: graph.bond_count(),
                actual: bond_tokens.len(),
            });
        }
        for (atom, text) in graph.atom_ids().zip(&atom_text) {
            if text.is_empty() {
                return Err(PreparedConnectedNonStereoError::EmptyAtomText(atom));
            }
        }

        Ok(Self {
            molecule,
            atom_text: Arc::from(
                atom_text
                    .into_iter()
                    .map(String::into_boxed_str)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ),
            bond_tokens: Arc::from(bond_tokens.into_boxed_slice()),
        })
    }

    fn molecule(&self) -> &PreparedMolecule {
        &self.molecule
    }

    fn atom_text(&self, atom: AtomId) -> &str {
        self.atom_text
            .get(atom.index())
            .map(AsRef::as_ref)
            .expect("prepared atom text must match the bound molecule")
    }

    fn bond_text(&self, bond: BondId, from: AtomId) -> &'static str {
        let topology = *self
            .molecule
            .graph()
            .bond(bond)
            .expect("prepared bond token must match the bound molecule");
        self.bond_tokens
            .get(bond.index())
            .copied()
            .expect("prepared bond token must match the bound molecule")
            .text_from(topology, from)
    }

    fn child_prefix(&self, parent: AtomId, incident: AdjacentBond) -> &str {
        let bond_text = self.bond_text(incident.bond(), parent);
        if bond_text.is_empty() {
            self.atom_text(incident.atom())
        } else {
            bond_text
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PreparedConnectedNonStereoError {
    EmptyMolecule,
    DisconnectedMolecule,
    AtomTextCountMismatch { expected: usize, actual: usize },
    BondTokenCountMismatch { expected: usize, actual: usize },
    EmptyAtomText(AtomId),
}

impl fmt::Display for PreparedConnectedNonStereoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMolecule => {
                formatter.write_str("a connected non-stereo surface requires at least one atom")
            }
            Self::DisconnectedMolecule => {
                formatter.write_str("a connected non-stereo surface requires one graph component")
            }
            Self::AtomTextCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} prepared atom texts, received {actual}"
            ),
            Self::BondTokenCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} prepared bond tokens, received {actual}"
            ),
            Self::EmptyAtomText(atom) => write!(
                formatter,
                "prepared atom text for {atom:?} must not be empty"
            ),
        }
    }
}

impl std::error::Error for PreparedConnectedNonStereoError {}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RingLabelSlot(usize);

impl RingLabelSlot {
    const fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RingLabels {
    bonds_by_slot: BTreeMap<RingLabelSlot, BondId>,
}

impl RingLabels {
    fn next_available(&self) -> RingLabelSlot {
        let mut candidate = 0_usize;
        for slot in self.bonds_by_slot.keys() {
            if slot.index() != candidate {
                break;
            }
            candidate += 1;
        }
        RingLabelSlot(candidate)
    }

    fn allocate(&mut self, bond: BondId) -> RingLabelSlot {
        assert!(
            self.bonds_by_slot.values().all(|owner| *owner != bond),
            "one ring bond may own only one visible label"
        );
        let slot = self.next_available();
        assert_eq!(
            self.bonds_by_slot.insert(slot, bond),
            None,
            "a newly allocated visible ring label must be free"
        );
        slot
    }

    fn slot_for_bond(&self, bond: BondId) -> RingLabelSlot {
        self.bonds_by_slot
            .iter()
            .find_map(|(slot, owner)| (*owner == bond).then_some(*slot))
            .expect("an open structural ring must own a visible label")
    }

    fn release(&mut self, slot: RingLabelSlot, bond: BondId) {
        assert_eq!(
            self.bonds_by_slot.remove(&slot),
            Some(bond),
            "a closing ring must release its own visible label"
        );
    }

    fn is_empty(&self) -> bool {
        self.bonds_by_slot.is_empty()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PendingEmission {
    InlineAtom(AdjacentBond),
    BranchBondOrAtom(AdjacentBond),
    BranchAtom(AdjacentBond),
    RingClosureLabel {
        incident: AdjacentBond,
        label_slot: RingLabelSlot,
    },
}

impl PendingEmission {
    const fn incident(self) -> AdjacentBond {
        match self {
            Self::InlineAtom(incident)
            | Self::BranchBondOrAtom(incident)
            | Self::BranchAtom(incident)
            | Self::RingClosureLabel { incident, .. } => incident,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum NonStereoChoice {
    Root(AtomId),
    RingOpen(AdjacentBond),
    RingClose(AdjacentBond),
    BranchOpen(AdjacentBond),
    InlineChild(AdjacentBond),
    Pending,
    BranchClose,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VisibleChoice {
    choice: NonStereoChoice,
    text: String,
}

impl VisibleChoice {
    pub(crate) const fn choice(&self) -> NonStereoChoice {
        self.choice
    }

    pub(crate) fn text(&self) -> &str {
        &self.text
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ConnectedNonStereoWriterState<S> {
    surface: PreparedConnectedNonStereo,
    structural: WriterState<S>,
    labels: RingLabels,
    pending: Option<PendingEmission>,
}

impl<S: ConstraintSolver> ConnectedNonStereoWriterState<S> {
    pub(crate) fn initial(surface: &PreparedConnectedNonStereo) -> Result<Self, S::Error> {
        Ok(Self {
            surface: surface.clone(),
            structural: WriterState::initial(surface.molecule())?,
            labels: RingLabels::default(),
            pending: None,
        })
    }

    pub(crate) fn active_atom(&self) -> Option<AtomId> {
        self.structural.active_atom()
    }

    pub(crate) const fn graph_is_complete(&self) -> bool {
        self.structural.graph_is_complete()
    }

    pub(crate) fn is_accepted(&self) -> bool {
        self.pending.is_none()
            && self.labels.is_empty()
            && self.structural.active_atom().is_none()
            && self.structural.graph_is_complete()
    }

    pub(crate) fn choices(&self) -> Vec<VisibleChoice> {
        if self.is_accepted() {
            return Vec::new();
        }
        if self.pending.is_some() {
            return vec![self.visible_choice(NonStereoChoice::Pending)];
        }

        let frontier = self.frontier();
        if !frontier.component_roots().is_empty() {
            return frontier
                .component_roots()
                .iter()
                .copied()
                .map(NonStereoChoice::Root)
                .map(|choice| self.visible_choice(choice))
                .collect();
        }

        let mut choices = Vec::new();
        choices.extend(
            frontier
                .ring_closures()
                .iter()
                .copied()
                .map(NonStereoChoice::RingClose)
                .map(|choice| self.visible_choice(choice)),
        );
        choices.extend(
            frontier
                .ring_openings()
                .iter()
                .copied()
                .map(NonStereoChoice::RingOpen)
                .map(|choice| self.visible_choice(choice)),
        );
        if !choices.is_empty() {
            return choices;
        }

        choices.extend(
            frontier
                .branch_children()
                .iter()
                .copied()
                .map(NonStereoChoice::BranchOpen)
                .map(|choice| self.visible_choice(choice)),
        );
        choices.extend(
            frontier
                .inline_children()
                .iter()
                .copied()
                .map(NonStereoChoice::InlineChild)
                .map(|choice| self.visible_choice(choice)),
        );
        if frontier.can_complete_path() && !self.structural.graph_is_complete() {
            choices.push(self.visible_choice(NonStereoChoice::BranchClose));
        }
        choices
    }

    pub(crate) fn advance(
        &self,
        choice: NonStereoChoice,
    ) -> Result<(String, Self), S::Error> {
        let visible = self
            .choices()
            .into_iter()
            .find(|candidate| candidate.choice == choice)
            .expect("advance requires an advertised non-stereo choice");
        let token = visible.text;

        let successor = match choice {
            NonStereoChoice::Root(root) => Self {
                surface: self.surface.clone(),
                structural: self.structural.begin_component(root),
                labels: self.labels.clone(),
                pending: None,
            }
            .normalize_component_completion(),
            NonStereoChoice::RingOpen(incident) => {
                let structural = self.structural.open_ring_endpoint(incident)?;
                let mut labels = self.labels.clone();
                let slot = labels.allocate(incident.bond());
                assert_eq!(
                    token,
                    ring_label_text(slot),
                    "advertised ring label must match the allocated label"
                );
                Self {
                    surface: self.surface.clone(),
                    structural,
                    labels,
                    pending: None,
                }
            }
            NonStereoChoice::RingClose(incident) => {
                let first_endpoint = self.structural.ring_closure_first_endpoint(incident);
                let label_slot = self.labels.slot_for_bond(incident.bond());
                let bond_text = self.surface.bond_text(incident.bond(), first_endpoint);
                if bond_text.is_empty() {
                    let structural = self.structural.close_ring_endpoint(incident);
                    let mut labels = self.labels.clone();
                    labels.release(label_slot, incident.bond());
                    Self {
                        surface: self.surface.clone(),
                        structural,
                        labels,
                        pending: None,
                    }
                    .normalize_component_completion()
                } else {
                    Self {
                        surface: self.surface.clone(),
                        structural: self.structural.clone(),
                        labels: self.labels.clone(),
                        pending: Some(PendingEmission::RingClosureLabel {
                            incident,
                            label_slot,
                        }),
                    }
                }
            }
            NonStereoChoice::BranchOpen(incident) => Self {
                surface: self.surface.clone(),
                structural: self.structural.commit_traversal_edge(incident)?,
                labels: self.labels.clone(),
                pending: Some(PendingEmission::BranchBondOrAtom(incident)),
            },
            NonStereoChoice::InlineChild(incident) => {
                let parent = self
                    .structural
                    .active_atom()
                    .expect("inline emission requires an active atom");
                let bond_text = self.surface.bond_text(incident.bond(), parent);
                let structural = self.structural.commit_traversal_edge(incident)?;
                if bond_text.is_empty() {
                    Self {
                        surface: self.surface.clone(),
                        structural: structural.enter_inline_child(incident),
                        labels: self.labels.clone(),
                        pending: None,
                    }
                    .normalize_component_completion()
                } else {
                    Self {
                        surface: self.surface.clone(),
                        structural,
                        labels: self.labels.clone(),
                        pending: Some(PendingEmission::InlineAtom(incident)),
                    }
                }
            }
            NonStereoChoice::Pending => self.advance_pending(),
            NonStereoChoice::BranchClose => Self {
                surface: self.surface.clone(),
                structural: self.structural.complete_path(),
                labels: self.labels.clone(),
                pending: None,
            },
        };

        Ok((token, successor))
    }

    fn advance_pending(&self) -> Self {
        let pending = self.pending.expect("no committed text is pending");
        let incident = pending.incident();

        match pending {
            PendingEmission::InlineAtom(_) => Self {
                surface: self.surface.clone(),
                structural: self.structural.enter_inline_child(incident),
                labels: self.labels.clone(),
                pending: None,
            }
            .normalize_component_completion(),
            PendingEmission::BranchBondOrAtom(_) => {
                let parent = self
                    .structural
                    .active_atom()
                    .expect("a committed branch child requires its active parent");
                if self.surface.bond_text(incident.bond(), parent).is_empty() {
                    Self {
                        surface: self.surface.clone(),
                        structural: self.structural.enter_branch_child(incident),
                        labels: self.labels.clone(),
                        pending: None,
                    }
                    .normalize_component_completion()
                } else {
                    Self {
                        surface: self.surface.clone(),
                        structural: self.structural.clone(),
                        labels: self.labels.clone(),
                        pending: Some(PendingEmission::BranchAtom(incident)),
                    }
                }
            }
            PendingEmission::BranchAtom(_) => Self {
                surface: self.surface.clone(),
                structural: self.structural.enter_branch_child(incident),
                labels: self.labels.clone(),
                pending: None,
            }
            .normalize_component_completion(),
            PendingEmission::RingClosureLabel { label_slot, .. } => {
                assert_eq!(
                    self.labels.slot_for_bond(incident.bond()),
                    label_slot,
                    "a pending ring label must retain its assignment"
                );
                let structural = self.structural.close_ring_endpoint(incident);
                let mut labels = self.labels.clone();
                labels.release(label_slot, incident.bond());
                Self {
                    surface: self.surface.clone(),
                    structural,
                    labels,
                    pending: None,
                }
                .normalize_component_completion()
            }
        }
    }

    fn visible_choice(&self, choice: NonStereoChoice) -> VisibleChoice {
        VisibleChoice {
            choice,
            text: self.choice_text(choice),
        }
    }

    fn choice_text(&self, choice: NonStereoChoice) -> String {
        match choice {
            NonStereoChoice::Root(root) => self.surface.atom_text(root).to_owned(),
            NonStereoChoice::RingOpen(_) => ring_label_text(self.labels.next_available()),
            NonStereoChoice::RingClose(incident) => {
                let first_endpoint = self.structural.ring_closure_first_endpoint(incident);
                let bond_text = self.surface.bond_text(incident.bond(), first_endpoint);
                if bond_text.is_empty() {
                    ring_label_text(self.labels.slot_for_bond(incident.bond()))
                } else {
                    bond_text.to_owned()
                }
            }
            NonStereoChoice::BranchOpen(_) => "(".to_owned(),
            NonStereoChoice::InlineChild(incident) => {
                let parent = self
                    .structural
                    .active_atom()
                    .expect("inline choice requires an active atom");
                self.surface.child_prefix(parent, incident).to_owned()
            }
            NonStereoChoice::Pending => {
                let pending = self.pending.expect("no committed text is pending");
                let incident = pending.incident();
                match pending {
                    PendingEmission::InlineAtom(_) | PendingEmission::BranchAtom(_) => {
                        self.surface.atom_text(incident.atom()).to_owned()
                    }
                    PendingEmission::BranchBondOrAtom(_) => {
                        let parent = self
                            .structural
                            .active_atom()
                            .expect("a committed branch child requires its active parent");
                        self.surface.child_prefix(parent, incident).to_owned()
                    }
                    PendingEmission::RingClosureLabel { label_slot, .. } => {
                        ring_label_text(label_slot)
                    }
                }
            }
            NonStereoChoice::BranchClose => ")".to_owned(),
        }
    }

    fn frontier(&self) -> StructuralFrontier {
        let frontier = self.structural.structural_frontier();
        assert!(
            !frontier.is_contradiction(),
            "connected non-stereo writer reached a structural contradiction"
        );
        frontier
    }

    fn normalize_component_completion(mut self) -> Self {
        if self.pending.is_some() || !self.structural.graph_is_complete() {
            return self;
        }
        assert!(
            self.labels.is_empty(),
            "a complete structural graph must not retain visible ring labels"
        );
        assert!(
            self.frontier().can_complete_path(),
            "a complete connected graph must have a completable top-level path"
        );
        let completed = self.structural.complete_path();
        assert_eq!(
            completed.active_atom(),
            None,
            "connected graph completion must not restore a branch parent"
        );
        self.structural = completed;
        self
    }
}

fn ring_label_text(label_slot: RingLabelSlot) -> String {
    let label = label_slot
        .index()
        .checked_add(1)
        .expect("visible ring-label number must not overflow");
    ring_label_number_text(label)
}

fn ring_label_number_text(label: usize) -> String {
    assert!(label > 0, "visible ring labels are one-based");
    assert!(
        label <= 99,
        "ring labels above 99 require an explicit dialect policy"
    );
    if label < 10 {
        label.to_string()
    } else {
        format!("%{label}")
    }
}

fn graph_is_connected(graph: &PreparedGraph) -> bool {
    if graph.atom_count() == 0 {
        return false;
    }
    let root = AtomId::new(0);
    let mut visited = vec![false; graph.atom_count()];
    visited[root.index()] = true;
    let mut pending = vec![root];
    let mut visited_count = 0_usize;

    while let Some(atom) = pending.pop() {
        visited_count += 1;
        for incident in graph
            .neighbors(atom)
            .expect("prepared atom must have an adjacency row")
        {
            let neighbour = incident.atom();
            if !visited[neighbour.index()] {
                visited[neighbour.index()] = true;
                pending.push(neighbour);
            }
        }
    }
    visited_count == graph.atom_count()
}

#[cfg(test)]
mod tests {
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
        let choices = state.choices();
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
            PreparedConnectedNonStereo::new(
                bonded,
                vec!["C".to_owned(), "O".to_owned()],
                Vec::new(),
            ),
            Err(PreparedConnectedNonStereoError::BondTokenCountMismatch { .. })
        ));
    }

    #[test]
    fn closed_labels_are_immediately_reusable_spelling_resources() {
        let first = BondId::new(0);
        let second = BondId::new(1);
        let third = BondId::new(2);
        let mut labels = RingLabels::default();

        let zero = labels.allocate(first);
        let one = labels.allocate(second);
        assert_eq!(zero, RingLabelSlot(0));
        assert_eq!(one, RingLabelSlot(1));
        labels.release(zero, first);
        assert_eq!(labels.allocate(third), RingLabelSlot(0));
    }

    #[test]
    fn ring_label_spelling_matches_the_selected_smiles_dialect() {
        assert_eq!(ring_label_number_text(1), "1");
        assert_eq!(ring_label_number_text(9), "9");
        assert_eq!(ring_label_number_text(10), "%10");
        assert_eq!(ring_label_number_text(99), "%99");
    }

    #[test]
    #[should_panic(expected = "above 99 require an explicit dialect policy")]
    fn unselected_large_ring_label_dialect_fails_at_rendering() {
        let _ = ring_label_number_text(100);
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
            rooted.choices(),
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

        assert_eq!([root, open, first_child, second_child, close].concat(), "C1CC1");
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
    fn dative_bond_text_follows_prepared_orientation() {
        let (surface, atoms, bonds) =
            fixture(&["N", "B"], &[(0, 1, NonStereoBondToken::DativeAToB)]);
        let initial = State::initial(&surface).unwrap();
        let edge_from_n = incident(&surface, atoms[0], bonds[0]);
        let edge_from_b = incident(&surface, atoms[1], bonds[0]);

        let (_, rooted_at_n) = advance(&initial, NonStereoChoice::Root(atoms[0]));
        only_choice(
            &rooted_at_n,
            NonStereoChoice::InlineChild(edge_from_n),
            "->",
        );

        let (_, rooted_at_b) = advance(&initial, NonStereoChoice::Root(atoms[1]));
        only_choice(
            &rooted_at_b,
            NonStereoChoice::InlineChild(edge_from_b),
            "<-",
        );
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

    fn reachable_strings(surface: &PreparedConnectedNonStereo) -> BTreeMap<String, ()> {
        let initial = State::initial(surface).unwrap();
        let mut pending = vec![(initial, String::new())];
        let mut complete = BTreeMap::new();
        let mut explored = 0_usize;

        while let Some((state, prefix)) = pending.pop() {
            explored += 1;
            assert!(explored <= 100_000, "writer test exceeded its exploration bound");
            if state.is_accepted() {
                complete.insert(prefix, ());
                continue;
            }

            let choices = state.choices();
            assert!(!choices.is_empty(), "writer must not dead-end before acceptance");
            for visible in choices {
                let (token, successor) = state.advance(visible.choice()).unwrap();
                assert_eq!(token, visible.text());
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

    fn reference_subtree_strings(
        surface: &PreparedConnectedNonStereo,
        atom: AtomId,
        parent: Option<AtomId>,
    ) -> BTreeMap<String, ()> {
        let children = surface
            .molecule()
            .graph()
            .neighbors(atom)
            .expect("reference atom must exist")
            .iter()
            .copied()
            .filter(|incident| Some(incident.atom()) != parent)
            .collect::<Vec<_>>();
        let mut support = BTreeMap::new();

        for order in permutations(&children) {
            let mut partial = vec![surface.atom_text(atom).to_owned()];
            for (index, incident) in order.iter().copied().enumerate() {
                let child_support = reference_subtree_strings(surface, incident.atom(), Some(atom));
                let bond = surface.bond_text(incident.bond(), atom);
                let inline = index + 1 == order.len();
                let mut next = Vec::new();
                for prefix in &partial {
                    for child in child_support.keys() {
                        if inline {
                            next.push(format!("{prefix}{bond}{child}"));
                        } else {
                            next.push(format!("{prefix}({bond}{child})"));
                        }
                    }
                }
                partial = next;
            }
            support.extend(partial.into_iter().map(|text| (text, ())));
        }
        support
    }

    fn reference_tree_strings(surface: &PreparedConnectedNonStereo) -> BTreeMap<String, ()> {
        surface
            .molecule()
            .graph()
            .atom_ids()
            .flat_map(|root| reference_subtree_strings(surface, root, None))
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
            BTreeMap::from([("C1CC1".to_owned(), ())])
        );
    }
}
