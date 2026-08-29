//! Rust-owned molecular topology for the South Star 2 walker.
//!
//! Topology is deliberately independent of SMILES rendering. Atom and bond
//! text, including context-dependent alternatives, belongs to a writer model
//! rather than the molecular graph.

use std::collections::{BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::{AtomId, BondId, FactorId, VariableId};
use crate::model::{
    BondRole, ConstraintModel, ConstraintModelBuilder, EdgeRolePartition, SpanningTreeEdge,
    TetrahedralLayoutBond,
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
    components: Box<[PreparedComponent]>,
    component_by_atom: Box<[usize]>,
}

#[derive(Debug)]
pub(crate) struct PreparedComponent {
    atoms: Box<[AtomId]>,
    bonds: Box<[BondId]>,
}

impl PreparedComponent {
    pub(crate) fn atoms(&self) -> &[AtomId] {
        &self.atoms
    }

    pub(crate) fn bonds(&self) -> &[BondId] {
        &self.bonds
    }
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

    pub(crate) fn components(&self) -> &[PreparedComponent] {
        &self.components
    }

    pub(crate) fn component_of_atom(&self, atom: AtomId) -> Option<usize> {
        self.component_by_atom.get(atom.index()).copied()
    }
}

#[derive(Clone, Debug)]
pub struct PreparedMolecule {
    graph: Arc<PreparedGraph>,
    constraints: Arc<ConstraintModel>,
    bond_decision_variables: Arc<[VariableId]>,
    bond_role_partitions: Arc<[EdgeRolePartition]>,
}

impl PreparedMolecule {
    pub fn new(graph: PreparedGraph) -> Self {
        let graph = Arc::new(graph);
        let initial_domains = vec![BondRole::role_domain(); graph.bond_count()];
        let role_partitions = vec![EdgeRolePartition::bond_role(); graph.bond_count()];
        Self::from_decisions(graph, &initial_domains, &role_partitions)
    }

    pub(crate) fn with_bond_decisions(
        source: &Self,
        initial_domains: &[Domain],
        role_partitions: &[EdgeRolePartition],
    ) -> Self {
        assert_eq!(
            initial_domains.len(),
            source.graph.bond_count(),
            "prepared bond decisions must match the graph bond count"
        );
        assert_eq!(
            role_partitions.len(),
            source.graph.bond_count(),
            "prepared bond role partitions must match the graph bond count"
        );
        Self::from_decisions(Arc::clone(&source.graph), initial_domains, role_partitions)
    }

    pub(crate) fn constraint_assembly(
        source: &Self,
        initial_domains: &[Domain],
        role_partitions: &[EdgeRolePartition],
    ) -> PreparedConstraintAssembly {
        PreparedConstraintAssembly::new(Arc::clone(&source.graph), initial_domains, role_partitions)
    }

    fn from_decisions(
        graph: Arc<PreparedGraph>,
        initial_domains: &[Domain],
        role_partitions: &[EdgeRolePartition],
    ) -> Self {
        PreparedConstraintAssembly::new(graph, initial_domains, role_partitions).finish()
    }

    pub fn graph(&self) -> &PreparedGraph {
        self.graph.as_ref()
    }

    pub fn constraint_model(&self) -> &ConstraintModel {
        self.constraints.as_ref()
    }

    pub(crate) fn constraint_model_arc(&self) -> Arc<ConstraintModel> {
        Arc::clone(&self.constraints)
    }

    pub(crate) fn bond_decision_variable(&self, bond: BondId) -> Option<VariableId> {
        self.bond_decision_variables.get(bond.index()).copied()
    }

    pub(crate) fn bond_role_partition(&self, bond: BondId) -> Option<EdgeRolePartition> {
        self.bond_role_partitions.get(bond.index()).copied()
    }
}

pub(crate) struct PreparedConstraintAssembly {
    graph: Arc<PreparedGraph>,
    builder: ConstraintModelBuilder,
    bond_decision_variables: Vec<VariableId>,
    bond_role_partitions: Box<[EdgeRolePartition]>,
}

impl PreparedConstraintAssembly {
    fn new(
        graph: Arc<PreparedGraph>,
        initial_domains: &[Domain],
        role_partitions: &[EdgeRolePartition],
    ) -> Self {
        assert_eq!(initial_domains.len(), graph.bond_count());
        assert_eq!(role_partitions.len(), graph.bond_count());
        let mut builder = ConstraintModelBuilder::new();
        let mut bond_decision_variables = Vec::with_capacity(graph.bond_count());

        for bond in graph.bond_ids() {
            bond_decision_variables.push(
                builder
                    .add_variable(initial_domains[bond.index()])
                    .expect("prepared bond decisions must fit the constraint identifier space"),
            );
        }
        Self {
            graph,
            builder,
            bond_decision_variables,
            bond_role_partitions: role_partitions.into(),
        }
    }

    pub(crate) fn add_isolated_variable(&mut self, initial_domain: Domain) -> VariableId {
        self.builder
            .add_variable(initial_domain)
            .expect("prepared semantic variables must fit the constraint identifier space")
    }

    pub(crate) fn add_binary_relation(
        &mut self,
        left: VariableId,
        right: VariableId,
        allowed_pairs: impl IntoIterator<Item = (u8, u8)>,
    ) -> FactorId {
        self.builder
            .add_binary_relation(left, right, allowed_pairs)
            .expect("prepared binary relation must define a valid factor")
    }

    pub(crate) fn add_latent_directional_ring_placement(
        &mut self,
        mark_variable: VariableId,
        plan_variable: VariableId,
        allowed_pairs: impl IntoIterator<Item = (u8, u8)>,
    ) -> FactorId {
        self.builder
            .add_latent_directional_ring_placement(mark_variable, plan_variable, allowed_pairs)
            .expect("prepared directional ring placement must define a valid latent factor")
    }

    pub(crate) fn bond_decision_variable(&self, bond: BondId) -> VariableId {
        self.bond_decision_variables[bond.index()]
    }

    pub(crate) fn bond_role_partition(&self, bond: BondId) -> EdgeRolePartition {
        self.bond_role_partitions[bond.index()]
    }

    pub(crate) fn add_latent_tetrahedral_layout(
        &mut self,
        order_variable: VariableId,
        role_pattern_variable: VariableId,
        bonds: impl IntoIterator<Item = TetrahedralLayoutBond>,
        allowed_orders_by_pattern: impl IntoIterator<Item = Domain>,
    ) -> FactorId {
        self.builder
            .add_latent_tetrahedral_layout(
                order_variable,
                role_pattern_variable,
                bonds,
                allowed_orders_by_pattern,
            )
            .expect("prepared tetrahedral layout must define a valid latent factor")
    }

    pub(crate) fn finish(mut self) -> PreparedMolecule {
        for component in self.graph.components() {
            let edges = component.bonds().iter().map(|bond_id| {
                let bond = self
                    .graph
                    .bond(*bond_id)
                    .expect("component bond must belong to the prepared graph");
                SpanningTreeEdge::with_role_partition(
                    self.bond_decision_variables[bond_id.index()],
                    bond.a(),
                    bond.b(),
                    self.bond_role_partitions[bond_id.index()],
                )
            });
            self.builder
                .add_spanning_tree(component.atoms().iter().copied(), edges)
                .expect("prepared components must define valid spanning-tree factors");
        }
        PreparedMolecule {
            graph: self.graph,
            constraints: Arc::new(self.builder.build()),
            bond_decision_variables: Arc::from(self.bond_decision_variables),
            bond_role_partitions: Arc::from(self.bond_role_partitions),
        }
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
            u32::try_from(self.atom_count).map_err(|_| PreparedGraphError::AtomCapacityExceeded)?,
        );
        self.atom_count += 1;
        Ok(atom)
    }

    pub fn add_bond(&mut self, a: AtomId, b: AtomId) -> Result<BondId, PreparedGraphError> {
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

        let (components, component_by_atom) = prepared_components(self.atom_count, &adjacency);

        PreparedGraph {
            atom_count: self.atom_count,
            bonds: self.bonds.into_boxed_slice(),
            adjacency: adjacency
                .into_iter()
                .map(Vec::into_boxed_slice)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            components,
            component_by_atom,
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

fn prepared_components(
    atom_count: usize,
    adjacency: &[Vec<AdjacentBond>],
) -> (Box<[PreparedComponent]>, Box<[usize]>) {
    let mut component_by_atom = vec![usize::MAX; atom_count];
    let mut components = Vec::new();

    for root_index in 0..atom_count {
        if component_by_atom[root_index] != usize::MAX {
            continue;
        }
        let component = components.len();
        component_by_atom[root_index] = component;
        let root = atom_id_from_index(root_index);
        let mut pending = VecDeque::from([root]);
        let mut atoms = Vec::new();
        let mut bonds = BTreeSet::new();

        while let Some(atom) = pending.pop_front() {
            atoms.push(atom);
            for incident in &adjacency[atom.index()] {
                bonds.insert(incident.bond());
                let neighbour = incident.atom();
                if component_by_atom[neighbour.index()] == usize::MAX {
                    component_by_atom[neighbour.index()] = component;
                    pending.push_back(neighbour);
                }
            }
        }
        atoms.sort_unstable();
        components.push(PreparedComponent {
            atoms: atoms.into_boxed_slice(),
            bonds: bonds.into_iter().collect::<Vec<_>>().into_boxed_slice(),
        });
    }

    (
        components.into_boxed_slice(),
        component_by_atom.into_boxed_slice(),
    )
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
        assert_eq!(
            graph.bond(left).copied().unwrap().other(atoms[0]),
            Some(atoms[1])
        );
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
        assert_eq!(
            builder.add_bond(atoms[0], atoms[1]).unwrap(),
            BondId::new(0)
        );
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
        assert_eq!(graph.components().len(), 3);
        assert_eq!(graph.components()[0].atoms(), &atoms[..3]);
        assert_eq!(
            graph.components()[0].bonds(),
            &[BondId::new(0), BondId::new(1), BondId::new(2)]
        );
        assert_eq!(graph.components()[1].atoms(), &atoms[3..4]);
        assert!(graph.components()[1].bonds().is_empty());
        assert_eq!(graph.components()[2].atoms(), &atoms[4..]);
        assert_eq!(
            atoms.map(|atom| graph.component_of_atom(atom).unwrap()),
            [0, 0, 0, 1, 2]
        );
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
        let prepared = PreparedMolecule::new(PreparedGraphBuilder::new().build());

        assert_eq!(prepared.constraint_model().variable_count(), 0);
        assert_eq!(prepared.constraint_model().factor_count(), 0);
        assert_eq!(prepared.bond_decision_variable(BondId::new(0)), None);
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
        let prepared = PreparedMolecule::new(graph.build());
        let model = prepared.constraint_model();

        assert_eq!(model.variable_count(), 4);
        assert_eq!(model.factor_count(), 3);

        let role_variables = triangle
            .iter()
            .copied()
            .chain(std::iter::once(bridge))
            .map(|bond| prepared.bond_decision_variable(bond).unwrap())
            .collect::<Vec<_>>();
        for variable in &role_variables {
            assert_eq!(
                model.variable(*variable).unwrap().initial_domain(),
                BondRole::role_domain()
            );
        }

        let FactorDefinition::SpanningTree(first) = model.factor(FactorId::new(0)).unwrap() else {
            panic!("expected first component spanning-tree factor");
        };
        assert_eq!(first.atoms(), &atoms[..3]);
        assert_eq!(first.variables(), &role_variables[..3]);

        let FactorDefinition::SpanningTree(second) = model.factor(FactorId::new(1)).unwrap() else {
            panic!("expected second component spanning-tree factor");
        };
        assert_eq!(second.atoms(), &atoms[3..5]);
        assert_eq!(second.variables(), &role_variables[3..4]);

        let FactorDefinition::SpanningTree(third) = model.factor(FactorId::new(2)).unwrap() else {
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
        let prepared = PreparedMolecule::new(graph.build());
        let cloned = prepared.clone();

        assert_eq!(
            prepared.bond_decision_variable(bond),
            cloned.bond_decision_variable(bond)
        );
        assert!(Arc::ptr_eq(&prepared.graph, &cloned.graph));
        assert!(Arc::ptr_eq(&prepared.constraints, &cloned.constraints));
        assert!(Arc::ptr_eq(
            &prepared.bond_decision_variables,
            &cloned.bond_decision_variables
        ));
    }
}
