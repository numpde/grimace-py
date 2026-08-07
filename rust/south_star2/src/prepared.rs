//! Rust-owned molecular topology for the South Star 2 walker.
//!
//! Topology is deliberately independent of SMILES rendering. Atom and bond
//! text, including context-dependent alternatives, belongs to a writer model
//! rather than the molecular graph.

use std::collections::{BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;

use crate::ids::{AtomId, BondId, VariableId};
use crate::model::{
    BondRole, ConstraintModel, ConstraintModelBuilder, ConstraintModelError, SpanningTreeEdge,
};

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
    bond_role_variables: Arc<[VariableId]>,
}

impl PreparedMolecule {
    pub fn new(graph: PreparedGraph) -> Result<Self, ConstraintModelError> {
        let (constraints, bond_role_variables) = compile_graph_constraints(&graph)?;
        Ok(Self {
            graph: Arc::new(graph),
            constraints: Arc::new(constraints),
            bond_role_variables: Arc::from(bond_role_variables),
        })
    }

    pub fn graph(&self) -> &PreparedGraph {
        self.graph.as_ref()
    }

    pub fn constraint_model(&self) -> &ConstraintModel {
        self.constraints.as_ref()
    }

    pub fn bond_role_variable(&self, bond: BondId) -> Option<VariableId> {
        self.bond_role_variables.get(bond.index()).copied()
    }
}

fn compile_graph_constraints(
    graph: &PreparedGraph,
) -> Result<(ConstraintModel, Box<[VariableId]>), ConstraintModelError> {
    let mut builder = ConstraintModelBuilder::new();
    let mut bond_role_variables = Vec::with_capacity(graph.bond_count());

    for _bond in graph.bond_ids() {
        bond_role_variables.push(builder.add_variable(BondRole::role_domain())?);
    }

    for component in graph_components(graph) {
        let edges = component.bonds.iter().map(|bond_id| {
            let bond = graph
                .bond(*bond_id)
                .expect("component bond must belong to the prepared graph");
            SpanningTreeEdge::new(
                bond_role_variables[bond_id.index()],
                bond.a(),
                bond.b(),
            )
        });
        builder.add_spanning_tree(component.atoms, edges)?;
    }

    Ok((
        builder.build(),
        bond_role_variables.into_boxed_slice(),
    ))
}

#[derive(Debug)]
struct GraphComponent {
    atoms: Vec<AtomId>,
    bonds: BTreeSet<BondId>,
}

fn graph_components(graph: &PreparedGraph) -> Vec<GraphComponent> {
    let mut visited = vec![false; graph.atom_count()];
    let mut components = Vec::new();

    for root in graph.atom_ids() {
        if visited[root.index()] {
            continue;
        }
        visited[root.index()] = true;
        let mut pending = VecDeque::from([root]);
        let mut atoms = Vec::new();
        let mut bonds = BTreeSet::new();

        while let Some(atom) = pending.pop_front() {
            atoms.push(atom);
            for incident in graph
                .neighbors(atom)
                .expect("prepared atom must have an adjacency row")
            {
                bonds.insert(incident.bond());
                let neighbour = incident.atom();
                if !visited[neighbour.index()] {
                    visited[neighbour.index()] = true;
                    pending.push_back(neighbour);
                }
            }
        }

        atoms.sort_unstable();
        components.push(GraphComponent { atoms, bonds });
    }

    components
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
    use crate::ids::FactorId;
    use crate::model::FactorDefinition;

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
    fn empty_prepared_molecule_has_no_graph_constraints() {
        let prepared = PreparedMolecule::new(PreparedGraphBuilder::new().build()).unwrap();

        assert_eq!(prepared.constraint_model().variable_count(), 0);
        assert_eq!(prepared.constraint_model().factor_count(), 0);
        assert_eq!(prepared.bond_role_variable(BondId::new(0)), None);
    }

    #[test]
    fn prepared_molecule_compiles_one_spanning_tree_factor_per_component() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 6] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let triangle = [
            graph.add_bond(atoms[0], atoms[1]).unwrap(),
            graph.add_bond(atoms[1], atoms[2]).unwrap(),
            graph.add_bond(atoms[2], atoms[0]).unwrap(),
        ];
        let bridge = graph.add_bond(atoms[3], atoms[4]).unwrap();
        let prepared = PreparedMolecule::new(graph.build()).unwrap();
        let model = prepared.constraint_model();

        assert_eq!(model.variable_count(), 4);
        assert_eq!(model.factor_count(), 3);

        let role_variables = triangle
            .iter()
            .copied()
            .chain(std::iter::once(bridge))
            .map(|bond| prepared.bond_role_variable(bond).unwrap())
            .collect::<Vec<_>>();
        for variable in &role_variables {
            assert_eq!(
                model.variable(*variable).unwrap().initial_domain(),
                BondRole::role_domain()
            );
        }

        let FactorDefinition::SpanningTree(first) =
            model.factor(FactorId::new(0)).unwrap()
        else {
            panic!("expected first component spanning-tree factor");
        };
        assert_eq!(first.atoms(), &atoms[..3]);
        assert_eq!(first.variables(), &role_variables[..3]);

        let FactorDefinition::SpanningTree(second) =
            model.factor(FactorId::new(1)).unwrap()
        else {
            panic!("expected second component spanning-tree factor");
        };
        assert_eq!(second.atoms(), &atoms[3..5]);
        assert_eq!(second.variables(), &role_variables[3..4]);

        let FactorDefinition::SpanningTree(third) =
            model.factor(FactorId::new(2)).unwrap()
        else {
            panic!("expected isolated-atom spanning-tree factor");
        };
        assert_eq!(third.atoms(), &atoms[5..6]);
        assert!(third.edges().is_empty());
        assert!(third.variables().is_empty());
    }

    #[test]
    fn prepared_molecule_clone_shares_all_immutable_data() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 2] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let bond = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let prepared = PreparedMolecule::new(graph.build()).unwrap();
        let cloned = prepared.clone();

        assert_eq!(
            prepared.bond_role_variable(bond),
            cloned.bond_role_variable(bond)
        );
        assert!(Arc::ptr_eq(&prepared.graph, &cloned.graph));
        assert!(Arc::ptr_eq(&prepared.constraints, &cloned.constraints));
        assert!(Arc::ptr_eq(
            &prepared.bond_role_variables,
            &cloned.bond_role_variables
        ));
    }
}
