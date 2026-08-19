//! Connected non-stereo visible-token state.
//!
//! Roots, ring endpoints, inline children, and branch syntax advance through the
//! same writer state. An explicit child bond leaves traversal at the parent until
//! its atom is emitted. An explicit ring-closing bond likewise leaves the ring
//! open until its label is emitted.

use std::fmt;
use std::sync::Arc;

use crate::ids::{AtomId, BondId};
use crate::prepared::{AdjacentBond, PreparedBond, PreparedGraph, PreparedMolecule};
use crate::solver::ConstraintSolver;
use crate::traversal::RingLabelSlot;
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
            Self::EmptyAtomText(atom) => {
                write!(
                    formatter,
                    "prepared atom text for {atom:?} must not be empty"
                )
            }
        }
    }
}

impl std::error::Error for PreparedConnectedNonStereoError {}

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

#[derive(Clone, Debug)]
pub(crate) struct ConnectedNonStereoWriterState<S> {
    surface: PreparedConnectedNonStereo,
    structural: WriterState<S>,
    pending: Option<PendingEmission>,
}

impl<S: ConstraintSolver> ConnectedNonStereoWriterState<S> {
    pub(crate) fn initial(surface: &PreparedConnectedNonStereo) -> Result<Self, S::Error> {
        Ok(Self {
            surface: surface.clone(),
            structural: WriterState::initial(surface.molecule())?,
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
            && self.structural.active_atom().is_none()
            && self.structural.graph_is_complete()
    }

    pub(crate) fn root_choices(&self) -> Vec<(AtomId, &str)> {
        if self.pending.is_some() || self.is_accepted() {
            return Vec::new();
        }
        self.frontier()
            .component_roots()
            .iter()
            .copied()
            .map(|root| (root, self.surface.atom_text(root)))
            .collect()
    }

    pub(crate) fn ring_open_choices(&self) -> Vec<(AdjacentBond, String)> {
        if self.pending.is_some() {
            return Vec::new();
        }
        let frontier = self.frontier();
        if frontier.ring_openings().is_empty() {
            return Vec::new();
        }
        let label_text = ring_label_text(self.structural.next_ring_label_slot());
        frontier
            .ring_openings()
            .iter()
            .copied()
            .map(|incident| (incident, label_text.clone()))
            .collect()
    }

    pub(crate) fn ring_close_choices(&self) -> Vec<(AdjacentBond, String)> {
        if self.pending.is_some() {
            return Vec::new();
        }
        self.frontier()
            .ring_closures()
            .iter()
            .copied()
            .map(|incident| {
                let (first_endpoint, label_slot) = self.structural.ring_closure_facts(incident);
                let bond_text = self.surface.bond_text(incident.bond(), first_endpoint);
                let token = if bond_text.is_empty() {
                    ring_label_text(label_slot)
                } else {
                    bond_text.to_owned()
                };
                (incident, token)
            })
            .collect()
    }

    pub(crate) fn branch_choices(&self) -> Vec<(AdjacentBond, &'static str)> {
        if self.pending.is_some() {
            return Vec::new();
        }
        self.frontier()
            .branch_children()
            .iter()
            .copied()
            .map(|incident| (incident, "("))
            .collect()
    }

    pub(crate) fn inline_choices(&self) -> Vec<(AdjacentBond, &str)> {
        if self.pending.is_some() {
            return Vec::new();
        }
        let Some(parent) = self.structural.active_atom() else {
            return Vec::new();
        };
        self.frontier()
            .inline_children()
            .iter()
            .copied()
            .map(|incident| (incident, self.surface.child_prefix(parent, incident)))
            .collect()
    }

    pub(crate) fn pending_choice(&self) -> Option<(AdjacentBond, String)> {
        let pending = self.pending?;
        let incident = pending.incident();
        let text = match pending {
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
        };
        Some((incident, text))
    }

    pub(crate) fn branch_close_choice(&self) -> Option<&'static str> {
        if self.pending.is_some() {
            return None;
        }
        let frontier = self.frontier();
        (frontier.can_complete_path() && !self.structural.graph_is_complete()).then_some(")")
    }

    pub(crate) fn emit_root(&self, root: AtomId) -> (String, Self) {
        assert!(
            self.pending.is_none(),
            "pending text must be emitted before a root"
        );
        assert!(
            self.frontier().component_roots().contains(&root),
            "root emission requires an advertised component root"
        );
        let token = self.surface.atom_text(root).to_owned();
        let successor = Self {
            surface: self.surface.clone(),
            structural: self.structural.begin_component(root),
            pending: None,
        }
        .normalize_component_completion();
        (token, successor)
    }

    pub(crate) fn emit_ring_open(
        &self,
        incident: AdjacentBond,
    ) -> Result<(String, Self), S::Error> {
        assert!(
            self.pending.is_none(),
            "pending text must be emitted before a ring opening"
        );
        assert!(
            self.frontier().ring_openings().contains(&incident),
            "ring opening requires an advertised residual-attachment incidence"
        );

        let expected_slot = self.structural.next_ring_label_slot();
        let token = ring_label_text(expected_slot);
        let (structural, actual_slot) = self.structural.open_ring_endpoint(incident)?;
        assert_eq!(
            actual_slot, expected_slot,
            "observed next ring label must match the allocated slot"
        );
        Ok((
            token,
            Self {
                surface: self.surface.clone(),
                structural,
                pending: None,
            },
        ))
    }

    pub(crate) fn emit_ring_close(&self, incident: AdjacentBond) -> (String, Self) {
        assert!(
            self.pending.is_none(),
            "pending text must be emitted before a ring closure"
        );
        assert!(
            self.frontier().ring_closures().contains(&incident),
            "ring closure requires an advertised open endpoint"
        );

        let (first_endpoint, label_slot) = self.structural.ring_closure_facts(incident);
        let bond_text = self.surface.bond_text(incident.bond(), first_endpoint);
        if bond_text.is_empty() {
            let token = ring_label_text(label_slot);
            let (structural, actual_slot) = self.structural.close_ring_endpoint(incident);
            assert_eq!(actual_slot, label_slot);
            let successor = Self {
                surface: self.surface.clone(),
                structural,
                pending: None,
            }
            .normalize_component_completion();
            return (token, successor);
        }

        (
            bond_text.to_owned(),
            Self {
                surface: self.surface.clone(),
                structural: self.structural.clone(),
                pending: Some(PendingEmission::RingClosureLabel {
                    incident,
                    label_slot,
                }),
            },
        )
    }

    pub(crate) fn emit_branch_child(
        &self,
        incident: AdjacentBond,
    ) -> Result<(String, Self), S::Error> {
        assert!(
            self.pending.is_none(),
            "pending text must be emitted before a branch"
        );
        assert!(
            self.frontier().branch_children().contains(&incident),
            "branch emission requires an advertised branch child"
        );
        let structural = self.structural.commit_traversal_edge(incident)?;
        Ok((
            "(".to_owned(),
            Self {
                surface: self.surface.clone(),
                structural,
                pending: Some(PendingEmission::BranchBondOrAtom(incident)),
            },
        ))
    }

    pub(crate) fn emit_inline_child(
        &self,
        incident: AdjacentBond,
    ) -> Result<(String, Self), S::Error> {
        assert!(
            self.pending.is_none(),
            "pending text must be emitted before a child"
        );
        assert_eq!(
            self.frontier().inline_children(),
            &[incident],
            "inline emission requires the sole advertised inline child"
        );
        let parent = self
            .structural
            .active_atom()
            .expect("inline emission requires an active atom");
        let bond_text = self.surface.bond_text(incident.bond(), parent);
        let structural = self.structural.commit_traversal_edge(incident)?;

        if bond_text.is_empty() {
            let token = self.surface.atom_text(incident.atom()).to_owned();
            let successor = Self {
                surface: self.surface.clone(),
                structural: structural.enter_inline_child(incident),
                pending: None,
            }
            .normalize_component_completion();
            return Ok((token, successor));
        }

        Ok((
            bond_text.to_owned(),
            Self {
                surface: self.surface.clone(),
                structural,
                pending: Some(PendingEmission::InlineAtom(incident)),
            },
        ))
    }

    pub(crate) fn emit_pending(&self) -> (String, Self) {
        let pending = self.pending.expect("no committed text is pending");
        let incident = pending.incident();

        match pending {
            PendingEmission::InlineAtom(_) => {
                let token = self.surface.atom_text(incident.atom()).to_owned();
                let successor = Self {
                    surface: self.surface.clone(),
                    structural: self.structural.enter_inline_child(incident),
                    pending: None,
                }
                .normalize_component_completion();
                (token, successor)
            }
            PendingEmission::BranchBondOrAtom(_) => {
                let parent = self
                    .structural
                    .active_atom()
                    .expect("a committed branch child requires its active parent");
                let bond_text = self.surface.bond_text(incident.bond(), parent);
                if bond_text.is_empty() {
                    let token = self.surface.atom_text(incident.atom()).to_owned();
                    let successor = Self {
                        surface: self.surface.clone(),
                        structural: self.structural.enter_branch_child(incident),
                        pending: None,
                    }
                    .normalize_component_completion();
                    return (token, successor);
                }
                (
                    bond_text.to_owned(),
                    Self {
                        surface: self.surface.clone(),
                        structural: self.structural.clone(),
                        pending: Some(PendingEmission::BranchAtom(incident)),
                    },
                )
            }
            PendingEmission::BranchAtom(_) => {
                let token = self.surface.atom_text(incident.atom()).to_owned();
                let successor = Self {
                    surface: self.surface.clone(),
                    structural: self.structural.enter_branch_child(incident),
                    pending: None,
                }
                .normalize_component_completion();
                (token, successor)
            }
            PendingEmission::RingClosureLabel { label_slot, .. } => {
                assert_eq!(
                    self.structural.ring_closure_facts(incident).1,
                    label_slot,
                    "a pending ring label must retain its open slot"
                );
                let token = ring_label_text(label_slot);
                let (structural, actual_slot) = self.structural.close_ring_endpoint(incident);
                assert_eq!(actual_slot, label_slot);
                let successor = Self {
                    surface: self.surface.clone(),
                    structural,
                    pending: None,
                }
                .normalize_component_completion();
                (token, successor)
            }
        }
    }

    pub(crate) fn emit_branch_close(&self) -> (String, Self) {
        assert_eq!(
            self.branch_close_choice(),
            Some(")"),
            "branch close requires a completed branch path"
        );
        let structural = self.structural.complete_path();
        assert!(
            structural.active_atom().is_some(),
            "a branch close must restore its parent"
        );
        (
            ")".to_owned(),
            Self {
                surface: self.surface.clone(),
                structural,
                pending: None,
            },
        )
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
    if label < 10 {
        label.to_string()
    } else if label < 100 {
        format!("%{label}")
    } else {
        format!("%({label})")
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
    fn ring_label_spelling_matches_smiles_syntax() {
        assert_eq!(ring_label_number_text(1), "1");
        assert_eq!(ring_label_number_text(9), "9");
        assert_eq!(ring_label_number_text(10), "%10");
        assert_eq!(ring_label_number_text(99), "%99");
        assert_eq!(ring_label_number_text(100), "%(100)");
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

        let (root, rooted) = initial.emit_root(atoms[0]);
        assert_eq!(
            rooted.ring_open_choices(),
            vec![(left, "1".to_owned()), (right, "1".to_owned())]
        );
        let (open, opened) = rooted.emit_ring_open(left).unwrap();
        assert_eq!(rooted.active_atom(), Some(atoms[0]));
        assert_eq!(opened.inline_choices(), vec![(right, "C")]);
        let (first_child, walked) = opened.emit_inline_child(right).unwrap();
        let (second_child, walked) = walked.emit_inline_child(between).unwrap();
        assert_eq!(
            walked.ring_close_choices(),
            vec![(closing, "1".to_owned())]
        );
        let (close, accepted) = walked.emit_ring_close(closing);

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

        let (root, rooted) = initial.emit_root(atoms[0]);
        let (open, opened) = rooted.emit_ring_open(ring).unwrap();
        assert_eq!(open, "1");
        let (first_child, walked) = opened.emit_inline_child(entry).unwrap();
        let (second_child, walked) = walked.emit_inline_child(between).unwrap();
        assert_eq!(
            walked.ring_close_choices(),
            vec![(closing, "=".to_owned())]
        );

        let (bond, pending_label) = walked.emit_ring_close(closing);
        assert_eq!(bond, "=");
        assert_eq!(pending_label.active_atom(), Some(atoms[1]));
        assert!(!pending_label.graph_is_complete());
        assert_eq!(
            pending_label.pending_choice(),
            Some((closing, "1".to_owned()))
        );
        assert!(pending_label.ring_open_choices().is_empty());
        assert!(pending_label.ring_close_choices().is_empty());
        assert!(pending_label.inline_choices().is_empty());

        let (label, accepted) = pending_label.emit_pending();
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
        let (_, rooted) = initial.emit_root(atoms[0]);
        let edge = incident(&surface, atoms[0], bonds[0]);
        let (bond, pending) = rooted.emit_inline_child(edge).unwrap();

        assert_eq!(bond, "=");
        assert_eq!(rooted.active_atom(), Some(atoms[0]));
        assert_eq!(pending.active_atom(), Some(atoms[0]));
        assert_eq!(pending.pending_choice(), Some((edge, "O".to_owned())));
        let (atom, accepted) = pending.emit_pending();
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

        let (_, rooted_at_n) = initial.emit_root(atoms[0]);
        assert_eq!(rooted_at_n.inline_choices(), vec![(edge_from_n, "->")]);

        let (_, rooted_at_b) = initial.emit_root(atoms[1]);
        assert_eq!(rooted_at_b.inline_choices(), vec![(edge_from_b, "<-")]);
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
        let (root, rooted) = initial.emit_root(atoms[0]);
        let (open, pending_branch) = rooted.emit_branch_child(oxygen).unwrap();
        let (bond, pending_atom) = pending_branch.emit_pending();
        let (atom, branch) = pending_atom.emit_pending();
        let (close, restored) = branch.emit_branch_close();
        let (inline, accepted) = restored.emit_inline_child(nitrogen).unwrap();

        assert_eq!([root, open, bond, atom, close, inline].concat(), "C(=O)N");
        assert!(accepted.is_accepted());
    }

    fn reachable_strings(surface: &PreparedConnectedNonStereo) -> BTreeSet<String> {
        let initial = State::initial(surface).unwrap();
        let mut pending = vec![(initial, String::new())];
        let mut complete = BTreeSet::new();

        while let Some((state, prefix)) = pending.pop() {
            if state.is_accepted() {
                complete.insert(prefix);
                continue;
            }
            let mut successor_count = 0_usize;
            if state.pending_choice().is_some() {
                let (token, successor) = state.emit_pending();
                pending.push((successor, format!("{prefix}{token}")));
                successor_count += 1;
            } else {
                for (root, _) in state.root_choices() {
                    let (token, successor) = state.emit_root(root);
                    pending.push((successor, format!("{prefix}{token}")));
                    successor_count += 1;
                }
                for (incident, _) in state.ring_open_choices() {
                    let (token, successor) = state.emit_ring_open(incident).unwrap();
                    pending.push((successor, format!("{prefix}{token}")));
                    successor_count += 1;
                }
                for (incident, _) in state.ring_close_choices() {
                    let (token, successor) = state.emit_ring_close(incident);
                    pending.push((successor, format!("{prefix}{token}")));
                    successor_count += 1;
                }
                for (incident, _) in state.branch_choices() {
                    let (token, successor) = state.emit_branch_child(incident).unwrap();
                    pending.push((successor, format!("{prefix}{token}")));
                    successor_count += 1;
                }
                for (incident, _) in state.inline_choices() {
                    let (token, successor) = state.emit_inline_child(incident).unwrap();
                    pending.push((successor, format!("{prefix}{token}")));
                    successor_count += 1;
                }
                if state.branch_close_choice().is_some() {
                    let (token, successor) = state.emit_branch_close();
                    pending.push((successor, format!("{prefix}{token}")));
                    successor_count += 1;
                }
            }
            assert!(successor_count > 0, "connected emission must not dead-end");
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
                let child_support = reference_subtree_strings(surface, incident.atom(), Some(atom));
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
    fn simple_cycle_support_is_online_and_complete() {
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
}
