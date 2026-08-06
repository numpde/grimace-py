//! Rust-owned molecular topology for the South Star 2 walker.
//!
//! Topology is deliberately independent of SMILES rendering. Atom and bond
//! text, including context-dependent alternatives, belongs to a writer model
//! rather than the molecular graph.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use crate::ids::{AtomId, BondId};
use crate::model::ConstraintModel;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PreparedBond {
    a: AtomId,
    b: AtomId,
}

impl PreparedBond {
    pub const fn a(self) -> AtomId {
        self.a
    }

    pub const fn b(self) -> AtomId {
        self.b
    }

    pub fn other(self, atom: AtomId) -> Option<AtomId> {
        if atom == self.a {
            Some(self.b)
        } else if atom == self.b {
            Some(self.a)
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AdjacentBond {
    atom: AtomId,
    bond: BondId,
}

impl AdjacentBond {
    pub const fn atom(self) -> AtomId {
        self.atom
    }

    pub const fn bond(self) -> BondId {
        self.bond
    }
}

#[derive(Debug)]
pub struct PreparedGraph {
    atom_count: usize,
    bonds: Box<[PreparedBond]>,
    adjacency: Box<[Box<[AdjacentBond]>]>,
}

impl PreparedGraph {
    pub const fn atom_count(&self) -> usize {
        self.atom_count
    }

    pub fn bond_count(&self) -> usize {
        self.bonds.len()
    }

    pub fn atom_ids(&self) -> impl ExactSizeIterator<Item = AtomId> + DoubleEndedIterator + '_ {
        (0..self.atom_count).map(atom_id_from_index)
    }

    pub fn bond_ids(&self) -> impl ExactSizeIterator<Item = BondId> + DoubleEndedIterator + '_ {
        (0..self.bonds.len()).map(bond_id_from_index)
    }

    pub fn bond(&self, bond: BondId) -> Option<&PreparedBond> {
        self.bonds.get(bond.index())
    }

    pub fn neighbors(&self, atom: AtomId) -> Option<&[AdjacentBond]> {
        self.adjacency.get(atom.index()).map(AsRef::as_ref)
    }
}

#[derive(Clone, Debug)]
pub struct PreparedMolecule {
    graph: Arc<PreparedGraph>,
    constraints: Arc<ConstraintModel>,
}

impl PreparedMolecule {
    pub fn new(graph: PreparedGraph, constraints: ConstraintModel) -> Self {
        Self {
            graph: Arc::new(graph),
            constraints: Arc::new(constraints),
        }
    }

    pub fn graph(&self) -> &PreparedGraph {
        self.graph.as_ref()
    }

    pub fn constraint_model(&self) -> &ConstraintModel {
        self.constraints.as_ref()
    }
}

#[derive(Debug, Default)]
pub struct PreparedGraphBuilder {
    atom_count: usize,
    bonds: Vec<PreparedBond>,
    bond_pairs: BTreeSet<(AtomId, AtomId)>,
}

impl PreparedGraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_atom(&mut self) -> Result<AtomId, PreparedGraphError> {
        let atom = AtomId::new(
            u32::try_from(self.atom_count)
                .map_err(|_| PreparedGraphError::AtomCapacityExceeded)?,
        );
        self.atom_count += 1;
        Ok(atom)
    }

    pub fn add_bond(
        &mut self,
        a: AtomId,
        b: AtomId,
    ) -> Result<BondId, PreparedGraphError> {
        self.require_atom(a)?;
        self.require_atom(b)?;
        if a == b {
            return Err(PreparedGraphError::SelfBond(a));
        }

        let pair = ordered_pair(a, b);
        if self.bond_pairs.contains(&pair) {
            return Err(PreparedGraphError::DuplicateBond {
                a: pair.0,
                b: pair.1,
            });
        }

        let bond = BondId::new(
            u32::try_from(self.bonds.len())
                .map_err(|_| PreparedGraphError::BondCapacityExceeded)?,
        );
        self.bonds.push(PreparedBond { a, b });
        self.bond_pairs.insert(pair);
        Ok(bond)
    }

    pub fn build(self) -> PreparedGraph {
        let mut adjacency = vec![Vec::new(); self.atom_count];
        for (index, bond) in self.bonds.iter().copied().enumerate() {
            let bond_id = bond_id_from_index(index);
            adjacency[bond.a.index()].push(AdjacentBond {
                atom: bond.b,
                bond: bond_id,
            });
            adjacency[bond.b.index()].push(AdjacentBond {
                atom: bond.a,
                bond: bond_id,
            });
        }
        for row in &mut adjacency {
            row.sort_unstable();
        }

        PreparedGraph {
            atom_count: self.atom_count,
            bonds: self.bonds.into_boxed_slice(),
            adjacency: adjacency
                .into_iter()
                .map(Vec::into_boxed_slice)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn require_atom(&self, atom: AtomId) -> Result<(), PreparedGraphError> {
        if atom.index() < self.atom_count {
            Ok(())
        } else {
            Err(PreparedGraphError::UnknownAtom(atom))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreparedGraphError {
    AtomCapacityExceeded,
    BondCapacityExceeded,
    UnknownAtom(AtomId),
    SelfBond(AtomId),
    DuplicateBond { a: AtomId, b: AtomId },
}

impl fmt::Display for PreparedGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AtomCapacityExceeded => {
                formatter.write_str("prepared atom identifier capacity exceeded")
            }
            Self::BondCapacityExceeded => {
                formatter.write_str("prepared bond identifier capacity exceeded")
            }
            Self::UnknownAtom(atom) => write!(formatter, "unknown prepared atom {atom:?}"),
            Self::SelfBond(atom) => write!(formatter, "self-bond at {atom:?}"),
            Self::DuplicateBond { a, b } => write!(formatter, "duplicate bond {a:?}-{b:?}"),
        }
    }
}

impl std::error::Error for PreparedGraphError {}

fn ordered_pair(a: AtomId, b: AtomId) -> (AtomId, AtomId) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn atom_id_from_index(index: usize) -> AtomId {
    AtomId::new(
        u32::try_from(index)
            .expect("prepared graph builder validated the atom identifier capacity"),
    )
}

fn bond_id_from_index(index: usize) -> BondId {
    BondId::new(
        u32::try_from(index)
            .expect("prepared graph builder validated the bond identifier capacity"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConstraintModelBuilder, Domain};

    #[test]
    fn empty_graph_is_valid_topology() {
        let graph = PreparedGraphBuilder::new().build();

        assert_eq!(graph.atom_count(), 0);
        assert_eq!(graph.bond_count(), 0);
        assert_eq!(graph.atom_ids().collect::<Vec<_>>(), Vec::<AtomId>::new());
        assert_eq!(graph.bond_ids().collect::<Vec<_>>(), Vec::<BondId>::new());
        assert_eq!(graph.neighbors(AtomId::new(0)), None);
    }

    #[test]
    fn atom_and_bond_ids_are_stable() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let first = builder.add_bond(atoms[0], atoms[1]).unwrap();
        let second = builder.add_bond(atoms[1], atoms[2]).unwrap();
        let graph = builder.build();

        assert_eq!(atoms, [AtomId::new(0), AtomId::new(1), AtomId::new(2)]);
        assert_eq!(first, BondId::new(0));
        assert_eq!(second, BondId::new(1));
        assert_eq!(graph.atom_ids().collect::<Vec<_>>(), atoms.to_vec());
        assert_eq!(graph.bond_ids().collect::<Vec<_>>(), vec![first, second]);
    }

    #[test]
    fn adjacency_is_symmetric_and_sorted() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| builder.add_atom().unwrap());
        let right = builder.add_bond(atoms[0], atoms[2]).unwrap();
        let left = builder.add_bond(atoms[0], atoms[1]).unwrap();
        let graph = builder.build();

        assert_eq!(
            graph.neighbors(atoms[0]).unwrap(),
            &[
                AdjacentBond {
                    atom: atoms[1],
                    bond: left,
                },
                AdjacentBond {
                    atom: atoms[2],
                    bond: right,
                },
            ]
        );
        assert_eq!(graph.bond(left).copied().unwrap().other(atoms[0]), Some(atoms[1]));
        assert_eq!(
            graph.neighbors(atoms[1]).unwrap(),
            &[AdjacentBond {
                atom: atoms[0],
                bond: left,
            }]
        );
    }

    #[test]
    fn invalid_bonds_do_not_consume_an_identifier() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 2] = std::array::from_fn(|_| builder.add_atom().unwrap());

        assert_eq!(
            builder.add_bond(atoms[0], AtomId::new(99)),
            Err(PreparedGraphError::UnknownAtom(AtomId::new(99)))
        );
        assert_eq!(
            builder.add_bond(atoms[0], atoms[0]),
            Err(PreparedGraphError::SelfBond(atoms[0]))
        );
        assert_eq!(builder.add_bond(atoms[0], atoms[1]).unwrap(), BondId::new(0));
        assert_eq!(
            builder.add_bond(atoms[1], atoms[0]),
            Err(PreparedGraphError::DuplicateBond {
                a: atoms[0],
                b: atoms[1],
            })
        );
    }

    #[test]
    fn disconnected_and_cyclic_topologies_are_preserved() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms: [AtomId; 5] = std::array::from_fn(|_| builder.add_atom().unwrap());
        builder.add_bond(atoms[0], atoms[1]).unwrap();
        builder.add_bond(atoms[1], atoms[2]).unwrap();
        builder.add_bond(atoms[2], atoms[0]).unwrap();
        let graph = builder.build();

        assert_eq!(graph.atom_count(), 5);
        assert_eq!(graph.bond_count(), 3);
        for atom in &atoms[..3] {
            assert_eq!(graph.neighbors(*atom).unwrap().len(), 2);
        }
        for atom in &atoms[3..] {
            assert!(graph.neighbors(*atom).unwrap().is_empty());
        }
    }

    #[test]
    fn large_graph_preserves_all_atoms_and_bonds() {
        let mut builder = PreparedGraphBuilder::new();
        let atoms = (0..100)
            .map(|_| builder.add_atom().unwrap())
            .collect::<Vec<_>>();
        for pair in atoms.windows(2) {
            builder.add_bond(pair[0], pair[1]).unwrap();
        }
        let graph = builder.build();

        assert_eq!(graph.atom_count(), 100);
        assert_eq!(graph.bond_count(), 99);
        assert_eq!(graph.neighbors(atoms[0]).unwrap().len(), 1);
        assert_eq!(graph.neighbors(atoms[50]).unwrap().len(), 2);
    }

    #[test]
    fn prepared_molecule_clone_shares_immutable_data() {
        let mut graph = PreparedGraphBuilder::new();
        graph.add_atom().unwrap();

        let mut constraints = ConstraintModelBuilder::new();
        let variable = constraints
            .add_variable(Domain::from_indices([0, 1]).unwrap())
            .unwrap();
        let prepared = PreparedMolecule::new(graph.build(), constraints.build());
        let cloned = prepared.clone();

        assert_eq!(
            prepared
                .constraint_model()
                .variable(variable)
                .unwrap()
                .initial_domain(),
            Domain::from_indices([0, 1]).unwrap()
        );
        assert!(std::ptr::eq(prepared.graph(), cloned.graph()));
        assert!(std::ptr::eq(
            prepared.constraint_model(),
            cloned.constraint_model()
        ));
    }
}
