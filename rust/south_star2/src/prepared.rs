//! Validated Rust-owned molecule input for the South Star 2 walker.

use std::collections::{BTreeSet, VecDeque};
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

    pub const fn other(self, atom: AtomId) -> Option<AtomId> {
        if atom.get() == self.a.get() {
            Some(self.b)
        } else if atom.get() == self.b.get() {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedGraph {
    tokens: Box<[Box<str>]>,
    atoms: Box<[PreparedAtom]>,
    bonds: Box<[PreparedBond]>,
    adjacency: Box<[Box<[AdjacentBond]>]>,
    cycle_rank: usize,
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

    pub const fn cycle_rank(&self) -> usize {
        self.cycle_rank
    }

    pub const fn is_acyclic(&self) -> bool {
        self.cycle_rank == 0
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

#[derive(Clone, Debug, PartialEq, Eq)]
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

    pub fn constraint_model(&self) -> Arc<ConstraintModel> {
        Arc::clone(&self.constraints)
    }
}

#[derive(Clone, Debug, Default)]
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

    pub fn intern_token(
        &mut self,
        text: impl AsRef<str>,
    ) -> Result<TokenId, PreparedGraphError> {
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

    pub fn add_atom(
        &mut self,
        token: TokenId,
    ) -> Result<AtomId, PreparedGraphError> {
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

    pub fn build(self) -> Result<PreparedGraph, PreparedGraphError> {
        if self.atoms.is_empty() {
            return Err(PreparedGraphError::EmptyGraph);
        }

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

        let reachable = reachable_atom_count(&adjacency);
        if reachable != self.atoms.len() {
            return Err(PreparedGraphError::DisconnectedGraph {
                reachable,
                total: self.atoms.len(),
            });
        }

        let cycle_rank = self.bonds.len() - (self.atoms.len() - 1);
        Ok(PreparedGraph {
            tokens: self.tokens.into_boxed_slice(),
            atoms: self.atoms.into_boxed_slice(),
            bonds: self.bonds.into_boxed_slice(),
            adjacency: adjacency
                .into_iter()
                .map(Vec::into_boxed_slice)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            cycle_rank,
        })
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
    EmptyGraph,
    DisconnectedGraph { reachable: usize, total: usize },
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
            Self::EmptyGraph => formatter.write_str("prepared graph requires an atom"),
            Self::DisconnectedGraph { reachable, total } => write!(
                formatter,
                "prepared graph must be connected: reachable={reachable}, total={total}"
            ),
        }
    }
}

impl std::error::Error for PreparedGraphError {}

fn reachable_atom_count(adjacency: &[Vec<AdjacentBond>]) -> usize {
    let mut seen = vec![false; adjacency.len()];
    let mut pending = VecDeque::from([AtomId::new(0)]);
    let mut count = 0;

    while let Some(atom) = pending.pop_front() {
        if seen[atom.index()] {
            continue;
        }
        seen[atom.index()] = true;
        count += 1;
        pending.extend(adjacency[atom.index()].iter().map(|entry| entry.atom));
    }
    count
}

const fn ordered_pair(a: AtomId, b: AtomId) -> (AtomId, AtomId) {
    if a.get() < b.get() {
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
    use crate::{ConstraintModelBuilder, Domain, NativeSolverState};

    #[test]
    fn token_interning_is_stable_and_nonempty() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();

        assert_eq!(builder.intern_token("C").unwrap(), carbon);
        assert_eq!(builder.intern_token("N").unwrap(), TokenId::new(1));
        assert_eq!(builder.intern_token(""), Err(PreparedGraphError::EmptyToken));
        assert_eq!(
            PreparedGraphBuilder::new().build(),
            Err(PreparedGraphError::EmptyGraph)
        );
    }

    #[test]
    fn single_atom_graph_is_connected_and_acyclic() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();
        let atom = builder.add_atom(carbon).unwrap();
        let graph = builder.build().unwrap();

        assert_eq!(graph.atom_count(), 1);
        assert_eq!(graph.bond_count(), 0);
        assert_eq!(graph.cycle_rank(), 0);
        assert!(graph.is_acyclic());
        assert_eq!(graph.token_text(carbon), Some("C"));
        assert_eq!(graph.atom(atom).copied().unwrap().token(), carbon);
        assert_eq!(graph.neighbors(atom), Some(&[][..]));
    }

    #[test]
    fn connected_tree_has_symmetric_sorted_adjacency() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();
        let double = builder.intern_token("=").unwrap();
        let atoms: [AtomId; 3] =
            std::array::from_fn(|_| builder.add_atom(carbon).unwrap());
        let right = builder
            .add_bond(atoms[0], atoms[2], Some(double))
            .unwrap();
        let left = builder.add_bond(atoms[0], atoms[1], None).unwrap();
        let graph = builder.build().unwrap();

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
    fn invalid_bonds_do_not_consume_an_identifier() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();
        assert_eq!(
            builder.add_atom(TokenId::new(99)),
            Err(PreparedGraphError::UnknownToken(TokenId::new(99)))
        );
        let atoms: [AtomId; 2] =
            std::array::from_fn(|_| builder.add_atom(carbon).unwrap());
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
    fn disconnected_graph_is_rejected() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();
        builder.add_atom(carbon).unwrap();
        builder.add_atom(carbon).unwrap();

        assert_eq!(
            builder.build(),
            Err(PreparedGraphError::DisconnectedGraph {
                reachable: 1,
                total: 2,
            })
        );
    }

    #[test]
    fn cycles_are_recorded_but_not_rejected() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();
        let atoms: [AtomId; 3] =
            std::array::from_fn(|_| builder.add_atom(carbon).unwrap());
        builder.add_bond(atoms[0], atoms[1], None).unwrap();
        builder.add_bond(atoms[1], atoms[2], None).unwrap();
        builder.add_bond(atoms[2], atoms[0], None).unwrap();

        let graph = builder.build().unwrap();

        assert_eq!(graph.cycle_rank(), 1);
        assert!(!graph.is_acyclic());
    }

    #[test]
    fn preparation_has_no_small_molecule_envelope() {
        let mut builder = PreparedGraphBuilder::new();
        let carbon = builder.intern_token("C").unwrap();
        let atoms = (0..100)
            .map(|_| builder.add_atom(carbon).unwrap())
            .collect::<Vec<_>>();
        for pair in atoms.windows(2) {
            builder.add_bond(pair[0], pair[1], None).unwrap();
        }

        let graph = builder.build().unwrap();

        assert_eq!(graph.atom_count(), 100);
        assert_eq!(graph.bond_count(), 99);
        assert!(graph.is_acyclic());
    }

    #[test]
    fn prepared_molecule_shares_its_constraint_model() {
        let mut graph = PreparedGraphBuilder::new();
        let carbon = graph.intern_token("C").unwrap();
        graph.add_atom(carbon).unwrap();

        let mut constraints = ConstraintModelBuilder::new();
        let variable = constraints
            .add_variable(Domain::from_indices([0, 1]).unwrap())
            .unwrap();
        let prepared = PreparedMolecule::new(graph.build().unwrap(), constraints.build());
        let model = prepared.constraint_model();
        let state = NativeSolverState::initial(Arc::clone(&model)).unwrap();

        assert_eq!(state.domain(variable), Some(Domain::from_indices([0, 1]).unwrap()));
        let second_handle = prepared.constraint_model();
        assert!(Arc::ptr_eq(&model, &second_handle));
    }
}
