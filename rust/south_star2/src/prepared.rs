//! Rust-owned molecular graph input for the South Star 2 walker.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use crate::ids::{AtomId, BondId, TokenId};
use crate::model::ConstraintModel;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PreparedAtom {
    token: TokenId,
}

impl PreparedAtom {
    pub const fn token(self) -> TokenId {
        self.token
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PreparedBond {
    a: AtomId,
    b: AtomId,
    token: Option<TokenId>,
}

impl PreparedBond {
    pub const fn a(self) -> AtomId {
        self.a
    }

    pub const fn b(self) -> AtomId {
        self.b
    }

    pub const fn token(self) -> Option<TokenId> {
        self.token
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
    tokens: Box<[Box<str>]>,
    atoms: Box<[PreparedAtom]>,
    bonds: Box<[PreparedBond]>,
    adjacency: Box<[Box<[AdjacentBond]>]>,
}

impl PreparedGraph {
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    pub fn bond_count(&self) -> usize {
        self.bonds.len()
    }

    pub fn token_text(&self, token: TokenId) -> Option<&str> {
        self.tokens.get(token.index()).map(|text| text.as_ref())
    }

    pub fn atom(&self, atom: AtomId) -> Option<&PreparedAtom> {
        self.atoms.get(atom.index())
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
    tokens: Vec<Box<str>>,
    atoms: Vec<PreparedAtom>,
    bonds: Vec<PreparedBond>,
    bond_pairs: BTreeSet<(AtomId, AtomId)>,
}

impl PreparedGraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern_token(&mut self, text: impl AsRef<str>) -> Result<TokenId, PreparedGraphError> {
        let text = text.as_ref();
        if text.is_empty() {
            return Err(PreparedGraphError::EmptyToken);
        }
        if let Some(index) = self
            .tokens
            .iter()
            .position(|candidate| candidate.as_ref() == text)
        {
            return Ok(token_id_from_index(index));
        }

        let token = TokenId::new(
            u16::try_from(self.tokens.len())
                .map_err(|_| PreparedGraphError::TokenCapacityExceeded)?,
        );
        self.tokens.push(text.into());
        Ok(token)
    }

    pub fn add_atom(&mut self, token: TokenId) -> Result<AtomId, PreparedGraphError> {
        self.require_token(token)?;
        let atom = AtomId::new(
            u32::try_from(self.atoms.len())
                .map_err(|_| PreparedGraphError::AtomCapacityExceeded)?,
        );
        self.atoms.push(PreparedAtom { token });
        Ok(atom)
    }

    pub fn add_bond(
        &mut self,
        a: AtomId,
        b: AtomId,
        token: Option<TokenId>,
    ) -> Result<BondId, PreparedGraphError> {
        self.require_atom(a)?;
        self.require_atom(b)?;
        if let Some(token) = token {
            self.require_token(token)?;
        }
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
        self.bonds.push(PreparedBond { a, b, token });
        self.bond_pairs.insert(pair);
        Ok(bond)
    }

    pub fn build(self) -> PreparedGraph {
        let mut adjacency = vec![Vec::new(); self.atoms.len()];
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
            tokens: self.tokens.into_boxed_slice(),
            atoms: self.atoms.into_boxed_slice(),
            bonds: self.bonds.into_boxed_slice(),
            adjacency: adjacency
                .into_iter()
                .map(Vec::into_boxed_slice)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn require_token(&self, token: TokenId) -> Result<(), PreparedGraphError> {
        self.tokens
            .get(token.index())
            .map(|_| ())
            .ok_or(PreparedGraphError::UnknownToken(token))
    }

    fn require_atom(&self, atom: AtomId) -> Result<(), PreparedGraphError> {
        self.atoms
            .get(atom.index())
            .map(|_| ())
            .ok_or(PreparedGraphError::UnknownAtom(atom))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreparedGraphError {
    EmptyToken,
    TokenCapacityExceeded,
    UnknownToken(TokenId),
    AtomCapacityExceeded,
    BondCapacityExceeded,
    UnknownAtom(AtomId),
    SelfBond(AtomId),
    DuplicateBond { a: AtomId, b: AtomId },
}

impl fmt::Display for PreparedGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyToken => formatter.write_str("prepared tokens must be nonempty"),
            Self::TokenCapacityExceeded => {
                formatter.write_str("prepared token identifier capacity exceeded")
            }
            Self::UnknownToken(token) => write!(formatter, "unknown prepared token {token:?}"),
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

fn token_id_from_index(index: usize) -> TokenId {
    TokenId::new(
        u16::try_from(index)
            .expect("prepared token builder validated the identifier capacity"),
    )
}

fn bond_id_from_index(index: usize) -> BondId {
    BondId::new(
        u32::try_from(index)
            .expect("prepared graph builder validated the identifier capacity"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConstraintModelBuilder, Domain};

    #[test]
    fn token_interning_is_stable_and_nonempty() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();

        assert_eq!(builder.intern_token("C").unwrap(), carbon);
        assert_eq!(builder.intern_token("N").unwrap(), TokenId::new(1));
        assert_eq!(builder.intern_token(""), Err(PreparedGraphError::EmptyToken));
    }

    #[test]
    fn empty_graph_is_valid_prepared_input() {
        let graph = PreparedGraphBuilder::new().build();

        assert_eq!(graph.token_count(), 0);
        assert_eq!(graph.atom_count(), 0);
        assert_eq!(graph.bond_count(), 0);
        assert_eq!(graph.atom(AtomId::new(0)), None);
    }

    #[test]
    fn single_atom_graph_has_expected_tokens_and_adjacency() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();
        let atom = builder.add_atom(carbon).unwrap();
        let graph = builder.build();

        assert_eq!(graph.atom_count(), 1);
        assert_eq!(graph.bond_count(), 0);
        assert_eq!(graph.token_text(carbon), Some("C"));
        assert_eq!(graph.atom(atom).copied().unwrap().token(), carbon);
        assert!(graph.neighbors(atom).unwrap().is_empty());
    }

    #[test]
    fn adjacency_is_symmetric_and_sorted() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();
        let double = builder.intern_token("=").unwrap();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| builder.add_atom(carbon).unwrap());
        let right = builder
            .add_bond(atoms[0], atoms[2], Some(double))
            .unwrap();
        let left = builder.add_bond(atoms[0], atoms[1], None).unwrap();
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
        assert_eq!(graph.bond(right).copied().unwrap().token(), Some(double));
        assert_eq!(
            graph.neighbors(atoms[1]).unwrap(),
            &[AdjacentBond {
                atom: atoms[0],
                bond: left,
            }]
        );
    }

    #[test]
    fn invalid_additions_do_not_consume_identifiers() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();
        assert_eq!(
            builder.add_atom(TokenId::new(99)),
            Err(PreparedGraphError::UnknownToken(TokenId::new(99)))
        );
        let atoms: [AtomId; 2] = std::array::from_fn(|_| builder.add_atom(carbon).unwrap());
        assert_eq!(atoms, [AtomId::new(0), AtomId::new(1)]);

        assert_eq!(
            builder.add_bond(atoms[0], AtomId::new(99), None),
            Err(PreparedGraphError::UnknownAtom(AtomId::new(99)))
        );
        assert_eq!(
            builder.add_bond(atoms[0], atoms[0], None),
            Err(PreparedGraphError::SelfBond(atoms[0]))
        );
        assert_eq!(
            builder.add_bond(atoms[0], atoms[1], None).unwrap(),
            BondId::new(0)
        );
        assert_eq!(
            builder.add_bond(atoms[1], atoms[0], None),
            Err(PreparedGraphError::DuplicateBond {
                a: atoms[0],
                b: atoms[1],
            })
        );
    }

    #[test]
    fn disconnected_graph_is_valid_prepared_input() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();
        let first = builder.add_atom(carbon).unwrap();
        let second = builder.add_atom(carbon).unwrap();
        let graph = builder.build();

        assert_eq!(graph.atom_count(), 2);
        assert_eq!(graph.bond_count(), 0);
        assert!(graph.neighbors(first).unwrap().is_empty());
        assert!(graph.neighbors(second).unwrap().is_empty());
    }

    #[test]
    fn cyclic_graph_is_valid_prepared_input() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| builder.add_atom(carbon).unwrap());
        builder.add_bond(atoms[0], atoms[1], None).unwrap();
        builder.add_bond(atoms[1], atoms[2], None).unwrap();
        builder.add_bond(atoms[2], atoms[0], None).unwrap();
        let graph = builder.build();

        assert_eq!(graph.atom_count(), 3);
        assert_eq!(graph.bond_count(), 3);
        for atom in atoms {
            assert_eq!(graph.neighbors(atom).unwrap().len(), 2);
        }
    }

    #[test]
    fn large_graph_preserves_all_atoms_and_bonds() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();
        let atoms = (0..100)
            .map(|_| builder.add_atom(carbon).unwrap())
            .collect::<Vec<_>>();
        for pair in atoms.windows(2) {
            builder.add_bond(pair[0], pair[1], None).unwrap();
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
        let carbon = graph.intern_token("C").unwrap();
        graph.add_atom(carbon).unwrap();

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
