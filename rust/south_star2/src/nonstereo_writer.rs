//! First visible-token boundary for one connected non-stereo writer surface.
//!
//! This layer deliberately supports only component-root and inline-child
//! emission. An explicit bond commits one inline child while leaving traversal
//! at the parent until the child atom token is emitted.

use std::fmt;
use std::sync::Arc;

use crate::ids::{AtomId, BondId};
use crate::prepared::{AdjacentBond, PreparedGraph, PreparedMolecule};
use crate::solver::ConstraintSolver;
use crate::writer_state::WriterState;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DirectedBondText {
    a_to_b: Box<str>,
    b_to_a: Box<str>,
}

impl DirectedBondText {
    pub(crate) fn new(a_to_b: impl Into<String>, b_to_a: impl Into<String>) -> Self {
        Self {
            a_to_b: a_to_b.into().into_boxed_str(),
            b_to_a: b_to_a.into().into_boxed_str(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedConnectedNonStereo {
    molecule: PreparedMolecule,
    atom_text: Arc<[Box<str>]>,
    directed_bond_text: Arc<[DirectedBondText]>,
}

impl PreparedConnectedNonStereo {
    pub(crate) fn new(
        molecule: PreparedMolecule,
        atom_text: Vec<String>,
        directed_bond_text: Vec<DirectedBondText>,
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
        if directed_bond_text.len() != graph.bond_count() {
            return Err(PreparedConnectedNonStereoError::BondTextCountMismatch {
                expected: graph.bond_count(),
                actual: directed_bond_text.len(),
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
            directed_bond_text: Arc::from(directed_bond_text.into_boxed_slice()),
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

    fn directed_bond_text(&self, bond: BondId, from: AtomId) -> &str {
        let topology = self
            .molecule
            .graph()
            .bond(bond)
            .expect("prepared bond text must match the bound molecule");
        let text = self
            .directed_bond_text
            .get(bond.index())
            .expect("prepared bond text must match the bound molecule");

        if topology.a() == from {
            &text.a_to_b
        } else if topology.b() == from {
            &text.b_to_a
        } else {
            panic!("directed bond text requires one endpoint of the prepared bond");
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PreparedConnectedNonStereoError {
    EmptyMolecule,
    DisconnectedMolecule,
    AtomTextCountMismatch { expected: usize, actual: usize },
    BondTextCountMismatch { expected: usize, actual: usize },
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
            Self::BondTextCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} prepared directed bond texts, received {actual}"
            ),
            Self::EmptyAtomText(atom) => {
                write!(formatter, "prepared atom text for {atom:?} must not be empty")
            }
        }
    }
}

impl std::error::Error for PreparedConnectedNonStereoError {}

#[derive(Clone, Debug)]
pub(crate) struct ConnectedNonStereoWriterState<S> {
    surface: PreparedConnectedNonStereo,
    structural: WriterState<S>,
    pending_inline_atom: Option<AdjacentBond>,
}

impl<S: ConstraintSolver> ConnectedNonStereoWriterState<S> {
    pub(crate) fn initial(
        surface: &PreparedConnectedNonStereo,
    ) -> Result<Self, S::Error> {
        Ok(Self {
            surface: surface.clone(),
            structural: WriterState::initial(surface.molecule())?,
            pending_inline_atom: None,
        })
    }

    pub(crate) const fn active_atom(&self) -> Option<AtomId> {
        self.structural.active_atom()
    }

    pub(crate) const fn graph_is_complete(&self) -> bool {
        self.structural.graph_is_complete()
    }

    pub(crate) fn root_choices(&self) -> Vec<(AtomId, &str)> {
        if self.pending_inline_atom.is_some() {
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

    pub(crate) fn inline_choices(&self) -> Vec<(AdjacentBond, &str)> {
        if self.pending_inline_atom.is_some() {
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
            .map(|incident| {
                let bond_text =
                    self.surface
                        .directed_bond_text(incident.bond(), active);
                let text = if bond_text.is_empty() {
                    self.surface.atom_text(incident.atom())
                } else {
                    bond_text
                };
                (incident, text)
            })
            .collect()
    }

    pub(crate) fn pending_inline_choice(&self) -> Option<(AdjacentBond, &str)> {
        self.pending_inline_atom
            .map(|incident| (incident, self.surface.atom_text(incident.atom())))
    }

    pub(crate) fn emit_root(&self, root: AtomId) -> (String, Self) {
        assert!(
            self.pending_inline_atom.is_none(),
            "a pending inline atom must be emitted before another root"
        );
        assert!(
            self.structural
                .structural_frontier()
                .component_roots()
                .contains(&root),
            "root emission requires an advertised component root"
        );

        let token = self.surface.atom_text(root).to_owned();
        let structural = self.structural.begin_component(root);
        (
            token,
            Self {
                surface: self.surface.clone(),
                structural,
                pending_inline_atom: None,
            },
        )
    }

    pub(crate) fn emit_inline_child(&self, incident: AdjacentBond) -> (String, Self) {
        assert!(
            self.pending_inline_atom.is_none(),
            "a pending inline atom must be emitted before another child"
        );
        assert_eq!(
            self.structural.structural_frontier().inline_children(),
            &[incident],
            "inline emission requires the sole advertised inline child"
        );

        let active = self
            .structural
            .active_atom()
            .expect("inline emission requires an active atom");
        let bond_text = self
            .surface
            .directed_bond_text(incident.bond(), active);

        if bond_text.is_empty() {
            let token = self.surface.atom_text(incident.atom()).to_owned();
            let structural = self.structural.enter_inline_child(incident);
            return (
                token,
                Self {
                    surface: self.surface.clone(),
                    structural,
                    pending_inline_atom: None,
                },
            );
        }

        (
            bond_text.to_owned(),
            Self {
                surface: self.surface.clone(),
                structural: self.structural.clone(),
                pending_inline_atom: Some(incident),
            },
        )
    }

    pub(crate) fn emit_pending_inline_atom(&self) -> (String, Self) {
        let incident = self
            .pending_inline_atom
            .expect("no committed inline atom is pending");
        let token = self.surface.atom_text(incident.atom()).to_owned();
        let structural = self.structural.enter_inline_child(incident);

        (
            token,
            Self {
                surface: self.surface.clone(),
                structural,
                pending_inline_atom: None,
            },
        )
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
    use super::*;
    use crate::native::NativeSolverState;
    use crate::prepared::PreparedGraphBuilder;

    fn fixture(
        atom_text: &[&str],
        edges: &[(usize, usize, &str, &str)],
    ) -> (PreparedConnectedNonStereo, Vec<AtomId>, Vec<BondId>) {
        let mut graph = PreparedGraphBuilder::new();
        let atoms = atom_text
            .iter()
            .map(|_| graph.add_atom().unwrap())
            .collect::<Vec<_>>();
        let mut bonds = Vec::with_capacity(edges.len());
        let mut bond_text = Vec::with_capacity(edges.len());

        for &(a, b, a_to_b, b_to_a) in edges {
            bonds.push(graph.add_bond(atoms[a], atoms[b]).unwrap());
            bond_text.push(DirectedBondText::new(a_to_b, b_to_a));
        }

        let surface = PreparedConnectedNonStereo::new(
            PreparedMolecule::new(graph.build()),
            atom_text.iter().map(|text| (*text).to_owned()).collect(),
            bond_text,
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
    fn surface_rejects_missing_or_disconnected_text_bindings() {
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
    fn root_atom_is_the_first_visible_transition() {
        let (surface, atoms, _) = fixture(&["C"], &[]);
        let initial =
            ConnectedNonStereoWriterState::<NativeSolverState>::initial(&surface).unwrap();

        assert_eq!(initial.root_choices(), vec![(atoms[0], "C")]);
        assert!(initial.inline_choices().is_empty());
        assert_eq!(initial.active_atom(), None);

        let (token, rooted) = initial.emit_root(atoms[0]);
        assert_eq!(token, "C");
        assert_eq!(initial.active_atom(), None);
        assert_eq!(rooted.active_atom(), Some(atoms[0]));
        assert!(rooted.graph_is_complete());
        assert!(rooted.root_choices().is_empty());
    }

    #[test]
    fn elided_inline_bond_emits_and_enters_the_child_atom_together() {
        let (surface, atoms, bonds) = fixture(&["C", "O"], &[(0, 1, "", "")]);
        let initial =
            ConnectedNonStereoWriterState::<NativeSolverState>::initial(&surface).unwrap();
        let (_, rooted) = initial.emit_root(atoms[0]);
        let edge = incident(&surface, atoms[0], bonds[0]);

        assert_eq!(rooted.inline_choices(), vec![(edge, "O")]);
        let (token, walked) = rooted.emit_inline_child(edge);

        assert_eq!(token, "O");
        assert_eq!(rooted.active_atom(), Some(atoms[0]));
        assert_eq!(walked.active_atom(), Some(atoms[1]));
        assert!(walked.graph_is_complete());
        assert_eq!(walked.pending_inline_choice(), None);
    }

    #[test]
    fn explicit_inline_bond_leaves_the_parent_active_until_atom_emission() {
        let (surface, atoms, bonds) = fixture(&["C", "O"], &[(0, 1, "=", "=")]);
        let initial =
            ConnectedNonStereoWriterState::<NativeSolverState>::initial(&surface).unwrap();
        let (_, rooted) = initial.emit_root(atoms[0]);
        let edge = incident(&surface, atoms[0], bonds[0]);

        assert_eq!(rooted.inline_choices(), vec![(edge, "=")]);
        let (bond_token, pending) = rooted.emit_inline_child(edge);

        assert_eq!(bond_token, "=");
        assert_eq!(rooted.active_atom(), Some(atoms[0]));
        assert_eq!(pending.active_atom(), Some(atoms[0]));
        assert!(!pending.graph_is_complete());
        assert!(pending.root_choices().is_empty());
        assert!(pending.inline_choices().is_empty());
        assert_eq!(pending.pending_inline_choice(), Some((edge, "O")));

        let (atom_token, walked) = pending.emit_pending_inline_atom();
        assert_eq!(atom_token, "O");
        assert_eq!(pending.active_atom(), Some(atoms[0]));
        assert_eq!(walked.active_atom(), Some(atoms[1]));
        assert!(walked.graph_is_complete());
        assert_eq!(walked.pending_inline_choice(), None);
    }

    #[test]
    fn directed_bond_text_depends_on_the_traversal_endpoint() {
        let (surface, atoms, bonds) = fixture(&["N", "B"], &[(0, 1, "->", "<-")]);
        let edge_from_n = incident(&surface, atoms[0], bonds[0]);
        let edge_from_b = incident(&surface, atoms[1], bonds[0]);
        let initial =
            ConnectedNonStereoWriterState::<NativeSolverState>::initial(&surface).unwrap();

        let (_, rooted_at_n) = initial.emit_root(atoms[0]);
        assert_eq!(rooted_at_n.inline_choices(), vec![(edge_from_n, "->")]);
        assert_eq!(rooted_at_n.emit_inline_child(edge_from_n).0, "->");

        let (_, rooted_at_b) = initial.emit_root(atoms[1]);
        assert_eq!(rooted_at_b.inline_choices(), vec![(edge_from_b, "<-")]);
        assert_eq!(rooted_at_b.emit_inline_child(edge_from_b).0, "<-");
    }
}
