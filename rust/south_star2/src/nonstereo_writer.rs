//! Connected non-stereo visible-token state.
//!
//! The implemented transitions currently cover roots, inline children, and
//! branch syntax. Ring endpoint emission and disconnected-component separators
//! are unfinished syntax in the same writer, not separate preparation modes.

use std::fmt;
use std::sync::Arc;

use crate::ids::{AtomId, BondId};
use crate::prepared::{AdjacentBond, PreparedBond, PreparedGraph, PreparedMolecule};
use crate::solver::ConstraintSolver;
use crate::writer_state::WriterState;

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
            Self::EmptyMolecule => formatter
                .write_str("a connected non-stereo surface requires at least one atom"),
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
                write!(formatter, "prepared atom text for {atom:?} must not be empty")
            }
        }
    }
}

impl std::error::Error for PreparedConnectedNonStereoError {}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PendingChild {
    InlineAtom(AdjacentBond),
    BranchBondOrAtom(AdjacentBond),
    BranchAtom(AdjacentBond),
}

impl PendingChild {
    const fn incident(self) -> AdjacentBond {
        match self {
            Self::InlineAtom(incident)
            | Self::BranchBondOrAtom(incident)
            | Self::BranchAtom(incident) => incident,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ConnectedNonStereoWriterState<S> {
    surface: PreparedConnectedNonStereo,
    structural: WriterState<S>,
    pending_child: Option<PendingChild>,
}

impl<S: ConstraintSolver> ConnectedNonStereoWriterState<S> {
    pub(crate) fn initial(
        surface: &PreparedConnectedNonStereo,
    ) -> Result<Self, S::Error> {
        Ok(Self {
            surface: surface.clone(),
            structural: WriterState::initial(surface.molecule())?,
            pending_child: None,
        })
    }

    pub(crate) const fn active_atom(&self) -> Option<AtomId> {
        self.structural.active_atom()
    }

    pub(crate) const fn graph_is_complete(&self) -> bool {
        self.structural.graph_is_complete()
    }

    pub(crate) fn is_accepted(&self) -> bool {
        self.pending_child.is_none()
            && self.structural.active_atom().is_none()
            && self.structural.graph_is_complete()
    }

    pub(crate) fn root_choices(&self) -> Vec<(AtomId, &str)> {
        if self.pending_child.is_some() || self.is_accepted() {
            return Vec::new();
        }

        self.structural
            .structural_frontier()
            .component_roots()
            .iter()
            .copied()
            .map(|root| (root, self.surface.atom_text(root)))
            .collect()
    }

    pub(crate) fn branch_choices(&self) -> Vec<(AdjacentBond, &'static str)> {
        if self.pending_child.is_some() {
            return Vec::new();
        }

        self.structural
            .structural_frontier()
            .branch_children()
            .iter()
            .copied()
            .map(|incident| (incident, "("))
            .collect()
    }

    pub(crate) fn inline_choices(&self) -> Vec<(AdjacentBond, &str)> {
        if self.pending_child.is_some() {
            return Vec::new();
        }
        let Some(active) = self.structural.active_atom() else {
            return Vec::new();
        };

        self.structural
            .structural_frontier()
            .inline_children()
            .iter()
            .copied()
            .map(|incident| (incident, self.surface.child_prefix(active, incident)))
            .collect()
    }

    pub(crate) fn pending_choice(&self) -> Option<(AdjacentBond, &str)> {
        let pending = self.pending_child?;
        let incident = pending.incident();
        let text = match pending {
            PendingChild::InlineAtom(_) | PendingChild::BranchAtom(_) => {
                self.surface.atom_text(incident.atom())
            }
            PendingChild::BranchBondOrAtom(_) => {
                let parent = self
                    .structural
                    .active_atom()
                    .expect("a committed branch child requires its active parent");
                self.surface.child_prefix(parent, incident)
            }
        };
        Some((incident, text))
    }

    pub(crate) fn branch_close_choice(&self) -> Option<&'static str> {
        if self.pending_child.is_some() {
            return None;
        }
        let frontier = self.structural.structural_frontier();
        (frontier.can_complete_path() && !self.structural.graph_is_complete()).then_some(")")
    }

    pub(crate) fn emit_root(&self, root: AtomId) -> (String, Self) {
        assert!(
            self.pending_child.is_none(),
            "pending text must be emitted before a root"
        );
        assert!(
            self.structural
                .structural_frontier()
                .component_roots()
                .contains(&root),
            "root emission requires an advertised component root"
        );

        let token = self.surface.atom_text(root).to_owned();
        let successor = Self {
            surface: self.surface.clone(),
            structural: self.structural.begin_component(root),
            pending_child: None,
        }
        .normalize_component_completion();
        (token, successor)
    }

    pub(crate) fn emit_branch_child(&self, incident: AdjacentBond) -> (String, Self) {
        assert!(
            self.pending_child.is_none(),
            "pending text must be emitted before a branch"
        );
        assert!(
            self.structural
                .structural_frontier()
                .branch_children()
                .contains(&incident),
            "branch emission requires an advertised branch child"
        );

        (
            "(".to_owned(),
            Self {
                surface: self.surface.clone(),
                structural: self.structural.clone(),
                pending_child: Some(PendingChild::BranchBondOrAtom(incident)),
            },
        )
    }

    pub(crate) fn emit_inline_child(&self, incident: AdjacentBond) -> (String, Self) {
        assert!(
            self.pending_child.is_none(),
            "pending text must be emitted before a child"
        );
        assert_eq!(
            self.structural.structural_frontier().inline_children(),
            &[incident],
            "inline emission requires the sole advertised inline child"
        );

        let parent = self
            .structural
            .active_atom()
            .expect("inline emission requires an active atom");
        let bond_text = self.surface.bond_text(incident.bond(), parent);

        if bond_text.is_empty() {
            let token = self.surface.atom_text(incident.atom()).to_owned();
            let successor = Self {
                surface: self.surface.clone(),
                structural: self.structural.enter_inline_child(incident),
                pending_child: None,
            }
            .normalize_component_completion();
            return (token, successor);
        }

        (
            bond_text.to_owned(),
            Self {
                surface: self.surface.clone(),
                structural: self.structural.clone(),
                pending_child: Some(PendingChild::InlineAtom(incident)),
            },
        )
    }

    pub(crate) fn emit_pending(&self) -> (String, Self) {
        let pending = self.pending_child.expect("no committed text is pending");
        let incident = pending.incident();

        match pending {
            PendingChild::InlineAtom(_) => {
                let token = self.surface.atom_text(incident.atom()).to_owned();
                let successor = Self {
                    surface: self.surface.clone(),
                    structural: self.structural.enter_inline_child(incident),
                    pending_child: None,
                }
                .normalize_component_completion();
                (token, successor)
            }
            PendingChild::BranchBondOrAtom(_) => {
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
                        pending_child: None,
                    }
                    .normalize_component_completion();
                    return (token, successor);
                }

                (
                    bond_text.to_owned(),
                    Self {
                        surface: self.surface.clone(),
                        structural: self.structural.clone(),
                        pending_child: Some(PendingChild::BranchAtom(incident)),
                    },
                )
            }
            PendingChild::BranchAtom(_) => {
                let token = self.surface.atom_text(incident.atom()).to_owned();
                let successor = Self {
                    surface: self.surface.clone(),
                    structural: self.structural.enter_branch_child(incident),
                    pending_child: None,
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
        assert!(
            !structural.graph_is_complete(),
            "closing a branch must leave its main continuation unfinished"
        );

        (
            ")".to_owned(),
            Self {
                surface: self.surface.clone(),
                structural,
                pending_child: None,
            },
        )
    }

    fn normalize_component_completion(mut self) -> Self {
        if self.pending_child.is_some() || !self.structural.graph_is_complete() {
            return self;
        }
        assert!(
            self.structural
                .structural_frontier()
                .can_complete_path(),
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

    fn incident(
        surface: &PreparedConnectedNonStereo,
        atom: AtomId,
        bond: BondId,
    ) -> AdjacentBond {
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
            PreparedConnectedNonStereo::new(single, vec![String::new()], Vec::new()),
            Err(PreparedConnectedNonStereoError::EmptyAtomText(AtomId::new(0)))
        ));
    }

    #[test]
    fn surface_does_not_case_split_cyclic_topology() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        graph.add_bond(atoms[0], atoms[1]).unwrap();
        graph.add_bond(atoms[1], atoms[2]).unwrap();
        graph.add_bond(atoms[2], atoms[0]).unwrap();

        let surface = PreparedConnectedNonStereo::new(
            PreparedMolecule::new(graph.build()),
            vec!["C".to_owned(), "C".to_owned(), "C".to_owned()],
            vec![NonStereoBondToken::Elided; 3],
        )
        .unwrap();
        let initial = State::initial(&surface).unwrap();

        assert_eq!(initial.root_choices().len(), 3);
    }

    #[test]
    fn root_atom_is_the_first_visible_transition_and_can_accept() {
        let (surface, atoms, _) = fixture(&["C"], &[]);
        let initial = State::initial(&surface).unwrap();

        assert_eq!(initial.root_choices(), vec![(atoms[0], "C")]);
        assert!(!initial.is_accepted());
        assert_eq!(initial.active_atom(), None);

        let (token, accepted) = initial.emit_root(atoms[0]);
        assert_eq!(token, "C");
        assert_eq!(initial.active_atom(), None);
        assert_eq!(accepted.active_atom(), None);
        assert!(accepted.graph_is_complete());
        assert!(accepted.is_accepted());
        assert!(accepted.root_choices().is_empty());
    }

    #[test]
    fn elided_inline_bond_emits_and_enters_the_child_atom_together() {
        let (surface, atoms, bonds) = fixture(
            &["C", "O"],
            &[(0, 1, NonStereoBondToken::Elided)],
        );
        let initial = State::initial(&surface).unwrap();
        let (_, rooted) = initial.emit_root(atoms[0]);
        let edge = incident(&surface, atoms[0], bonds[0]);

        assert_eq!(rooted.inline_choices(), vec![(edge, "O")]);
        let (token, accepted) = rooted.emit_inline_child(edge);

        assert_eq!(token, "O");
        assert_eq!(rooted.active_atom(), Some(atoms[0]));
        assert_eq!(accepted.active_atom(), None);
        assert!(accepted.graph_is_complete());
        assert!(accepted.is_accepted());
        assert_eq!(accepted.pending_choice(), None);
    }

    #[test]
    fn explicit_inline_bond_leaves_the_parent_active_until_atom_emission() {
        let (surface, atoms, bonds) = fixture(
            &["C", "O"],
            &[(0, 1, NonStereoBondToken::Double)],
        );
        let initial = State::initial(&surface).unwrap();
        let (_, rooted) = initial.emit_root(atoms[0]);
        let edge = incident(&surface, atoms[0], bonds[0]);

        assert_eq!(rooted.inline_choices(), vec![(edge, "=")]);
        let (bond_token, pending) = rooted.emit_inline_child(edge);

        assert_eq!(bond_token, "=");
        assert_eq!(rooted.active_atom(), Some(atoms[0]));
        assert_eq!(pending.active_atom(), Some(atoms[0]));
        assert!(!pending.graph_is_complete());
        assert!(pending.root_choices().is_empty());
        assert!(pending.branch_choices().is_empty());
        assert!(pending.inline_choices().is_empty());
        assert_eq!(pending.pending_choice(), Some((edge, "O")));
        assert_eq!(pending.branch_close_choice(), None);

        let (atom_token, accepted) = pending.emit_pending();
        assert_eq!(atom_token, "O");
        assert_eq!(pending.active_atom(), Some(atoms[0]));
        assert_eq!(accepted.active_atom(), None);
        assert!(accepted.graph_is_complete());
        assert!(accepted.is_accepted());
        assert_eq!(accepted.pending_choice(), None);
    }

    #[test]
    fn dative_bond_text_depends_on_topology_and_traversal_direction() {
        let (surface, atoms, bonds) = fixture(
            &["N", "B"],
            &[(0, 1, NonStereoBondToken::DativeAToB)],
        );
        let edge_from_n = incident(&surface, atoms[0], bonds[0]);
        let edge_from_b = incident(&surface, atoms[1], bonds[0]);
        let initial = State::initial(&surface).unwrap();

        let (_, rooted_at_n) = initial.emit_root(atoms[0]);
        assert_eq!(rooted_at_n.inline_choices(), vec![(edge_from_n, "->")]);
        assert_eq!(rooted_at_n.emit_inline_child(edge_from_n).0, "->");

        let (_, rooted_at_b) = initial.emit_root(atoms[1]);
        assert_eq!(rooted_at_b.inline_choices(), vec![(edge_from_b, "<-")]);
        assert_eq!(rooted_at_b.emit_inline_child(edge_from_b).0, "<-");
    }

    #[test]
    fn explicit_branch_commits_child_until_atom_then_emits_close() {
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
        let mut tokens = Vec::new();

        let (token, rooted) = initial.emit_root(atoms[0]);
        tokens.push(token);
        assert_eq!(
            rooted.branch_choices(),
            vec![(oxygen, "("), (nitrogen, "(")]
        );

        let (token, branch_open) = rooted.emit_branch_child(oxygen);
        tokens.push(token);
        assert_eq!(branch_open.active_atom(), Some(atoms[0]));
        assert_eq!(branch_open.pending_choice(), Some((oxygen, "=")));
        assert!(branch_open.branch_choices().is_empty());
        assert!(branch_open.inline_choices().is_empty());

        let (token, branch_bond) = branch_open.emit_pending();
        tokens.push(token);
        assert_eq!(branch_bond.active_atom(), Some(atoms[0]));
        assert_eq!(branch_bond.pending_choice(), Some((oxygen, "O")));

        let (token, branch_atom) = branch_bond.emit_pending();
        tokens.push(token);
        assert_eq!(branch_atom.active_atom(), Some(atoms[1]));
        assert_eq!(branch_atom.branch_close_choice(), Some(")"));
        assert_eq!(branch_atom.active_atom(), Some(atoms[1]));

        let (token, restored) = branch_atom.emit_branch_close();
        tokens.push(token);
        assert_eq!(restored.active_atom(), Some(atoms[0]));
        assert_eq!(restored.inline_choices(), vec![(nitrogen, "N")]);

        let (token, accepted) = restored.emit_inline_child(nitrogen);
        tokens.push(token);
        assert!(accepted.is_accepted());

        assert_eq!(
            tokens.iter().map(String::as_str).collect::<Vec<_>>(),
            vec!["C", "(", "=", "O", ")", "N"]
        );
        assert_eq!(tokens.concat(), "C(=O)N");
    }

    fn reachable_strings(surface: &PreparedConnectedNonStereo) -> BTreeSet<String> {
        let initial = State::initial(surface).unwrap();
        let mut pending = vec![(initial, String::new())];
        let mut complete = BTreeSet::new();
        let mut explored = 0_usize;

        while let Some((state, prefix)) = pending.pop() {
            explored += 1;
            assert!(
                explored <= 100_000,
                "connected-tree writer test exceeded its exploration bound"
            );

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
                for (incident, _) in state.branch_choices() {
                    let (token, successor) = state.emit_branch_child(incident);
                    pending.push((successor, format!("{prefix}{token}")));
                    successor_count += 1;
                }
                for (incident, _) in state.inline_choices() {
                    let (token, successor) = state.emit_inline_child(incident);
                    pending.push((successor, format!("{prefix}{token}")));
                    successor_count += 1;
                }
                if state.branch_close_choice().is_some() {
                    let (token, successor) = state.emit_branch_close();
                    pending.push((successor, format!("{prefix}{token}")));
                    successor_count += 1;
                }
            }

            assert!(
                successor_count > 0,
                "connected-tree emission must not dead-end before acceptance"
            );
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
                let child_support =
                    reference_subtree_strings(surface, incident.atom(), Some(atom));
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
    fn complete_connected_tree_support_matches_an_independent_reference() {
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
            fixture(
                &["C", "O", "N"],
                &[
                    (0, 1, NonStereoBondToken::Double),
                    (0, 2, NonStereoBondToken::Elided),
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
}
