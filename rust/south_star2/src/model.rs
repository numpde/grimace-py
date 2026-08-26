//! Solver-neutral constraint definitions prepared once per molecule.

use std::collections::BTreeSet;
use std::fmt;

use crate::domain::Domain;
use crate::ids::{AtomId, FactorId, VariableId};

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BondRole {
    Traversal = 0,
    Ring = 1,
}

impl BondRole {
    pub const fn value_index(self) -> u8 {
        self as u8
    }

    pub const fn singleton_domain(self) -> Domain {
        Domain::from_bits(1_u64 << self.value_index())
    }

    pub const fn role_domain() -> Domain {
        Domain::from_bits(
            (1_u64 << Self::Traversal.value_index()) | (1_u64 << Self::Ring.value_index()),
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EdgeRolePartition {
    traversal_values: Domain,
    ring_values: Domain,
}

impl EdgeRolePartition {
    pub const fn new(traversal_values: Domain, ring_values: Domain) -> Self {
        Self {
            traversal_values,
            ring_values,
        }
    }

    pub const fn traversal_values(self) -> Domain {
        self.traversal_values
    }

    pub const fn ring_values(self) -> Domain {
        self.ring_values
    }

    pub const fn bond_role() -> Self {
        Self::new(
            BondRole::Traversal.singleton_domain(),
            BondRole::Ring.singleton_domain(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VariableDefinition {
    initial_domain: Domain,
}

impl VariableDefinition {
    pub const fn initial_domain(&self) -> Domain {
        self.initial_domain
    }
}

/// A compiled binary finite-domain relation.
///
/// Support masks are stored in both directions so a solver can revise either
/// domain without enumerating the relation's Cartesian product.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryRelationFactor {
    variables: [VariableId; 2],
    allowed_right_by_left: Box<[Domain]>,
    allowed_left_by_right: Box<[Domain]>,
}

impl BinaryRelationFactor {
    pub const fn left(&self) -> VariableId {
        self.variables[0]
    }

    pub const fn right(&self) -> VariableId {
        self.variables[1]
    }

    pub const fn variables(&self) -> &[VariableId; 2] {
        &self.variables
    }

    pub fn allowed_right(&self, left_value: u8) -> Domain {
        self.allowed_right_by_left
            .get(left_value as usize)
            .copied()
            .unwrap_or_else(Domain::empty)
    }

    pub fn allowed_left(&self, right_value: u8) -> Domain {
        self.allowed_left_by_right
            .get(right_value as usize)
            .copied()
            .unwrap_or_else(Domain::empty)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SpanningTreeEdge {
    decision_variable: VariableId,
    role_partition: EdgeRolePartition,
    a: AtomId,
    b: AtomId,
}

impl SpanningTreeEdge {
    pub const fn new(decision_variable: VariableId, a: AtomId, b: AtomId) -> Self {
        Self::with_role_partition(decision_variable, a, b, EdgeRolePartition::bond_role())
    }

    pub const fn with_role_partition(
        decision_variable: VariableId,
        a: AtomId,
        b: AtomId,
        role_partition: EdgeRolePartition,
    ) -> Self {
        Self {
            decision_variable,
            role_partition,
            a,
            b,
        }
    }

    pub const fn decision_variable(self) -> VariableId {
        self.decision_variable
    }

    pub const fn role_partition(self) -> EdgeRolePartition {
        self.role_partition
    }

    pub const fn a(self) -> AtomId {
        self.a
    }

    pub const fn b(self) -> AtomId {
        self.b
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanningTreeFactor {
    atoms: Box<[AtomId]>,
    edges: Box<[SpanningTreeEdge]>,
    variables: Box<[VariableId]>,
}

impl SpanningTreeFactor {
    pub fn atoms(&self) -> &[AtomId] {
        &self.atoms
    }

    pub fn edges(&self) -> &[SpanningTreeEdge] {
        &self.edges
    }

    pub fn variables(&self) -> &[VariableId] {
        &self.variables
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TetrahedralLayoutBond {
    decision_variable: VariableId,
    role_partition: EdgeRolePartition,
    pattern_bit: u8,
}

impl TetrahedralLayoutBond {
    pub const fn new(
        decision_variable: VariableId,
        role_partition: EdgeRolePartition,
        pattern_bit: u8,
    ) -> Self {
        Self {
            decision_variable,
            role_partition,
            pattern_bit,
        }
    }

    pub const fn decision_variable(self) -> VariableId {
        self.decision_variable
    }

    pub const fn role_partition(self) -> EdgeRolePartition {
        self.role_partition
    }

    pub const fn pattern_bit(self) -> u8 {
        self.pattern_bit
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TetrahedralLayoutFactor {
    order_variable: VariableId,
    role_pattern_variable: VariableId,
    bonds: Box<[TetrahedralLayoutBond]>,
    allowed_orders_by_pattern: Box<[Domain]>,
    variables: Box<[VariableId]>,
}

impl TetrahedralLayoutFactor {
    pub const fn order_variable(&self) -> VariableId {
        self.order_variable
    }

    pub const fn role_pattern_variable(&self) -> VariableId {
        self.role_pattern_variable
    }

    pub fn bonds(&self) -> &[TetrahedralLayoutBond] {
        &self.bonds
    }

    pub fn allowed_orders(&self, pattern: u8) -> Domain {
        self.allowed_orders_by_pattern
            .get(pattern as usize)
            .copied()
            .unwrap_or_else(Domain::empty)
    }

    pub fn variables(&self) -> &[VariableId] {
        &self.variables
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FactorDefinition {
    BinaryRelation(BinaryRelationFactor),
    SpanningTree(SpanningTreeFactor),
    TetrahedralLayout(TetrahedralLayoutFactor),
}

impl FactorDefinition {
    pub fn variables(&self) -> &[VariableId] {
        match self {
            Self::BinaryRelation(factor) => factor.variables(),
            Self::SpanningTree(factor) => factor.variables(),
            Self::TetrahedralLayout(factor) => factor.variables(),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FactorActivation {
    Always,
    Latent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintModel {
    variables: Box<[VariableDefinition]>,
    factors: Box<[FactorDefinition]>,
    factor_activation: Box<[FactorActivation]>,
    initial_factor_ids: Box<[FactorId]>,
    initial_factors_by_variable: Box<[Box<[FactorId]>]>,
    potential_factors_by_variable: Box<[Box<[FactorId]>]>,
}

impl ConstraintModel {
    pub fn empty() -> Self {
        ConstraintModelBuilder::new().build()
    }

    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    pub fn factor_count(&self) -> usize {
        self.factors.len()
    }

    pub fn variable(&self, variable: VariableId) -> Option<&VariableDefinition> {
        self.variables.get(variable.index())
    }

    pub fn factor(&self, factor: FactorId) -> Option<&FactorDefinition> {
        self.factors.get(factor.index())
    }

    pub fn factor_activation(&self, factor: FactorId) -> Option<FactorActivation> {
        self.factor_activation.get(factor.index()).copied()
    }

    pub fn initial_factor_ids(&self) -> &[FactorId] {
        &self.initial_factor_ids
    }

    pub fn initial_factors_for_variable(&self, variable: VariableId) -> Option<&[FactorId]> {
        self.initial_factors_by_variable
            .get(variable.index())
            .map(AsRef::as_ref)
    }

    pub fn potential_factors_for_variable(&self, variable: VariableId) -> Option<&[FactorId]> {
        self.potential_factors_by_variable
            .get(variable.index())
            .map(AsRef::as_ref)
    }

    pub(crate) fn initial_domains(&self) -> impl Iterator<Item = Domain> + '_ {
        self.variables
            .iter()
            .map(VariableDefinition::initial_domain)
    }
}

impl Default for ConstraintModel {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Clone, Debug, Default)]
pub struct ConstraintModelBuilder {
    variables: Vec<VariableDefinition>,
    factors: Vec<FactorDefinition>,
    factor_activation: Vec<FactorActivation>,
    factors_by_variable: Vec<Vec<FactorId>>,
}

impl ConstraintModelBuilder {
    pub const fn new() -> Self {
        Self {
            variables: Vec::new(),
            factors: Vec::new(),
            factor_activation: Vec::new(),
            factors_by_variable: Vec::new(),
        }
    }

    pub fn add_variable(
        &mut self,
        initial_domain: Domain,
    ) -> Result<VariableId, ConstraintModelError> {
        if initial_domain.is_empty() {
            return Err(ConstraintModelError::EmptyInitialDomain);
        }
        let value = u32::try_from(self.variables.len())
            .map_err(|_| ConstraintModelError::VariableCapacityExceeded)?;
        let variable = VariableId::new(value);
        self.variables.push(VariableDefinition { initial_domain });
        self.factors_by_variable.push(Vec::new());
        Ok(variable)
    }

    pub fn add_binary_relation(
        &mut self,
        left: VariableId,
        right: VariableId,
        allowed_pairs: impl IntoIterator<Item = (u8, u8)>,
    ) -> Result<FactorId, ConstraintModelError> {
        let left_domain = self
            .variables
            .get(left.index())
            .ok_or(ConstraintModelError::UnknownVariable(left))?
            .initial_domain;
        let right_domain = self
            .variables
            .get(right.index())
            .ok_or(ConstraintModelError::UnknownVariable(right))?
            .initial_domain;

        if left == right {
            return Err(ConstraintModelError::RepeatedVariableInFactor(left));
        }

        let mut allowed_right_by_left = vec![Domain::empty(); left_domain.value_span()];
        let mut allowed_left_by_right = vec![Domain::empty(); right_domain.value_span()];

        for (left_value, right_value) in allowed_pairs {
            if !left_domain.contains(left_value) {
                return Err(ConstraintModelError::RelationValueOutsideInitialDomain {
                    variable: left,
                    value_index: left_value,
                });
            }
            if !right_domain.contains(right_value) {
                return Err(ConstraintModelError::RelationValueOutsideInitialDomain {
                    variable: right,
                    value_index: right_value,
                });
            }

            allowed_right_by_left[left_value as usize] = allowed_right_by_left[left_value as usize]
                .union(Domain::from_bits(1_u64 << right_value));
            allowed_left_by_right[right_value as usize] = allowed_left_by_right
                [right_value as usize]
                .union(Domain::from_bits(1_u64 << left_value));
        }

        self.push_factor(
            FactorDefinition::BinaryRelation(BinaryRelationFactor {
                variables: [left, right],
                allowed_right_by_left: allowed_right_by_left.into_boxed_slice(),
                allowed_left_by_right: allowed_left_by_right.into_boxed_slice(),
            }),
            FactorActivation::Always,
        )
    }

    pub fn add_spanning_tree(
        &mut self,
        atoms: impl IntoIterator<Item = AtomId>,
        edges: impl IntoIterator<Item = SpanningTreeEdge>,
    ) -> Result<FactorId, ConstraintModelError> {
        let mut atoms = atoms.into_iter().collect::<Vec<_>>();
        if atoms.is_empty() {
            return Err(ConstraintModelError::EmptySpanningTreeAtomSet);
        }
        atoms.sort_unstable();
        if atoms.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ConstraintModelError::RepeatedAtomInSpanningTree);
        }

        let atom_set = atoms.iter().copied().collect::<BTreeSet<_>>();
        let edges = edges.into_iter().collect::<Vec<_>>();
        let mut variables = Vec::with_capacity(edges.len());
        let mut seen_variables = BTreeSet::new();

        for edge in &edges {
            if edge.a == edge.b {
                return Err(ConstraintModelError::SpanningTreeSelfEdge(edge.a));
            }
            for atom in [edge.a, edge.b] {
                if !atom_set.contains(&atom) {
                    return Err(ConstraintModelError::SpanningTreeEdgeOutsideAtomSet(atom));
                }
            }

            let variable = edge.decision_variable;
            let initial_domain = self
                .variables
                .get(variable.index())
                .ok_or(ConstraintModelError::UnknownVariable(variable))?
                .initial_domain;
            let traversal_values = initial_domain.intersect(edge.role_partition.traversal_values());
            let ring_values = initial_domain.intersect(edge.role_partition.ring_values());
            if !edge
                .role_partition
                .traversal_values()
                .intersect(edge.role_partition.ring_values())
                .is_empty()
                || traversal_values.union(ring_values) != initial_domain
            {
                return Err(ConstraintModelError::InvalidEdgeRolePartition(variable));
            }
            if !seen_variables.insert(variable) {
                return Err(ConstraintModelError::RepeatedVariableInFactor(variable));
            }
            if self
                .prepared_edge_role_partition(variable)
                .is_some_and(|existing| existing != edge.role_partition)
            {
                return Err(ConstraintModelError::ConflictingEdgeRolePartition(variable));
            }
            if self.factors_by_variable[variable.index()]
                .iter()
                .any(|factor| {
                    matches!(
                        self.factors
                            .get(factor.index())
                            .expect("prepared factor adjacency must resolve"),
                        FactorDefinition::SpanningTree(_)
                    )
                })
            {
                return Err(ConstraintModelError::OverlappingSpanningTreeVariable(
                    variable,
                ));
            }
            variables.push(variable);
        }

        self.push_factor(
            FactorDefinition::SpanningTree(SpanningTreeFactor {
                atoms: atoms.into_boxed_slice(),
                edges: edges.into_boxed_slice(),
                variables: variables.into_boxed_slice(),
            }),
            FactorActivation::Always,
        )
    }

    pub fn add_latent_tetrahedral_layout(
        &mut self,
        order_variable: VariableId,
        role_pattern_variable: VariableId,
        bonds: impl IntoIterator<Item = TetrahedralLayoutBond>,
        allowed_orders_by_pattern: impl IntoIterator<Item = Domain>,
    ) -> Result<FactorId, ConstraintModelError> {
        let order_domain = self
            .variables
            .get(order_variable.index())
            .ok_or(ConstraintModelError::UnknownVariable(order_variable))?
            .initial_domain;
        let pattern_domain = self
            .variables
            .get(role_pattern_variable.index())
            .ok_or(ConstraintModelError::UnknownVariable(role_pattern_variable))?
            .initial_domain;
        if order_variable == role_pattern_variable {
            return Err(ConstraintModelError::RepeatedVariableInFactor(
                order_variable,
            ));
        }

        let bonds = bonds.into_iter().collect::<Vec<_>>();
        if !(3..=4).contains(&bonds.len()) {
            return Err(ConstraintModelError::InvalidTetrahedralBondCount(
                bonds.len(),
            ));
        }
        let pattern_count = 1_usize << bonds.len();
        let expected_pattern_domain = Domain::from_bits((1_u64 << pattern_count) - 1);
        if pattern_domain != expected_pattern_domain {
            return Err(ConstraintModelError::InvalidTetrahedralPatternDomain {
                variable: role_pattern_variable,
                expected: expected_pattern_domain,
            });
        }

        let mut seen_variables = BTreeSet::from([order_variable, role_pattern_variable]);
        let mut seen_bits = BTreeSet::new();
        for bond in &bonds {
            let variable = bond.decision_variable;
            let initial_domain = self
                .variables
                .get(variable.index())
                .ok_or(ConstraintModelError::UnknownVariable(variable))?
                .initial_domain;
            if !seen_variables.insert(variable) {
                return Err(ConstraintModelError::RepeatedVariableInFactor(variable));
            }
            if usize::from(bond.pattern_bit) >= bonds.len() || !seen_bits.insert(bond.pattern_bit) {
                return Err(ConstraintModelError::InvalidTetrahedralPatternBit(
                    bond.pattern_bit,
                ));
            }
            let traversal = initial_domain.intersect(bond.role_partition.traversal_values());
            let ring = initial_domain.intersect(bond.role_partition.ring_values());
            if !bond
                .role_partition
                .traversal_values()
                .intersect(bond.role_partition.ring_values())
                .is_empty()
                || traversal.union(ring) != initial_domain
            {
                return Err(ConstraintModelError::InvalidEdgeRolePartition(variable));
            }
            if self
                .prepared_edge_role_partition(variable)
                .is_some_and(|existing| existing != bond.role_partition)
            {
                return Err(ConstraintModelError::ConflictingEdgeRolePartition(variable));
            }
        }
        if seen_bits.len() != bonds.len() {
            return Err(ConstraintModelError::IncompleteTetrahedralPatternBits);
        }

        let allowed_orders_by_pattern = allowed_orders_by_pattern.into_iter().collect::<Vec<_>>();
        if allowed_orders_by_pattern.len() != pattern_count {
            return Err(ConstraintModelError::TetrahedralLayoutRowCountMismatch {
                expected: pattern_count,
                actual: allowed_orders_by_pattern.len(),
            });
        }
        for &allowed in &allowed_orders_by_pattern {
            if !allowed.is_subset_of(order_domain) {
                return Err(ConstraintModelError::TetrahedralOrderOutsideInitialDomain(
                    order_variable,
                ));
            }
        }

        let mut variables = Vec::with_capacity(bonds.len() + 2);
        variables.extend([order_variable, role_pattern_variable]);
        variables.extend(bonds.iter().map(|bond| bond.decision_variable));
        self.push_factor(
            FactorDefinition::TetrahedralLayout(TetrahedralLayoutFactor {
                order_variable,
                role_pattern_variable,
                bonds: bonds.into_boxed_slice(),
                allowed_orders_by_pattern: allowed_orders_by_pattern.into_boxed_slice(),
                variables: variables.into_boxed_slice(),
            }),
            FactorActivation::Latent,
        )
    }

    pub fn build(self) -> ConstraintModel {
        let initial_factor_ids = self
            .factor_activation
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, activation)| {
                (activation == FactorActivation::Always).then(|| {
                    FactorId::new(
                        u32::try_from(index)
                            .expect("prepared factor count must fit its identifier space"),
                    )
                })
            })
            .collect::<Vec<_>>();
        let initial_factors_by_variable = self
            .factors_by_variable
            .iter()
            .map(|factors| {
                factors
                    .iter()
                    .copied()
                    .filter(|factor| {
                        self.factor_activation[factor.index()] == FactorActivation::Always
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        ConstraintModel {
            variables: self.variables.into_boxed_slice(),
            factors: self.factors.into_boxed_slice(),
            factor_activation: self.factor_activation.into_boxed_slice(),
            initial_factor_ids: initial_factor_ids.into_boxed_slice(),
            initial_factors_by_variable,
            potential_factors_by_variable: self
                .factors_by_variable
                .into_iter()
                .map(Vec::into_boxed_slice)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn push_factor(
        &mut self,
        definition: FactorDefinition,
        activation: FactorActivation,
    ) -> Result<FactorId, ConstraintModelError> {
        let value = u32::try_from(self.factors.len())
            .map_err(|_| ConstraintModelError::FactorCapacityExceeded)?;
        let factor = FactorId::new(value);
        for variable in definition.variables().iter().copied() {
            self.factors_by_variable[variable.index()].push(factor);
        }
        self.factors.push(definition);
        self.factor_activation.push(activation);
        Ok(factor)
    }

    fn prepared_edge_role_partition(&self, variable: VariableId) -> Option<EdgeRolePartition> {
        self.factors_by_variable[variable.index()]
            .iter()
            .find_map(|factor| match &self.factors[factor.index()] {
                FactorDefinition::BinaryRelation(_) => None,
                FactorDefinition::SpanningTree(spanning) => spanning
                    .edges()
                    .iter()
                    .find(|edge| edge.decision_variable() == variable)
                    .map(|edge| edge.role_partition()),
                FactorDefinition::TetrahedralLayout(layout) => layout
                    .bonds()
                    .iter()
                    .find(|bond| bond.decision_variable() == variable)
                    .map(|bond| bond.role_partition()),
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstraintModelError {
    EmptyInitialDomain,
    VariableCapacityExceeded,
    FactorCapacityExceeded,
    UnknownVariable(VariableId),
    RepeatedVariableInFactor(VariableId),
    RelationValueOutsideInitialDomain {
        variable: VariableId,
        value_index: u8,
    },
    EmptySpanningTreeAtomSet,
    RepeatedAtomInSpanningTree,
    SpanningTreeSelfEdge(AtomId),
    SpanningTreeEdgeOutsideAtomSet(AtomId),
    InvalidEdgeRolePartition(VariableId),
    ConflictingEdgeRolePartition(VariableId),
    OverlappingSpanningTreeVariable(VariableId),
    InvalidTetrahedralBondCount(usize),
    InvalidTetrahedralPatternDomain {
        variable: VariableId,
        expected: Domain,
    },
    InvalidTetrahedralPatternBit(u8),
    IncompleteTetrahedralPatternBits,
    TetrahedralLayoutRowCountMismatch {
        expected: usize,
        actual: usize,
    },
    TetrahedralOrderOutsideInitialDomain(VariableId),
}

impl fmt::Display for ConstraintModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInitialDomain => {
                formatter.write_str("constraint variables require a nonempty initial domain")
            }
            Self::VariableCapacityExceeded => {
                formatter.write_str("too many variables for the South Star 2 identifier space")
            }
            Self::FactorCapacityExceeded => {
                formatter.write_str("too many factors for the South Star 2 identifier space")
            }
            Self::UnknownVariable(variable) => {
                write!(
                    formatter,
                    "constraint factor references unknown variable {variable:?}"
                )
            }
            Self::RepeatedVariableInFactor(variable) => {
                write!(
                    formatter,
                    "constraint factor repeats variable {variable:?} in its scope"
                )
            }
            Self::RelationValueOutsideInitialDomain {
                variable,
                value_index,
            } => {
                write!(
                    formatter,
                    "relation value {value_index} is outside the initial domain of {variable:?}"
                )
            }
            Self::EmptySpanningTreeAtomSet => {
                formatter.write_str("a spanning-tree factor requires at least one atom")
            }
            Self::RepeatedAtomInSpanningTree => {
                formatter.write_str("a spanning-tree factor cannot repeat an atom")
            }
            Self::SpanningTreeSelfEdge(atom) => {
                write!(
                    formatter,
                    "a spanning-tree factor cannot contain a self-edge at {atom:?}"
                )
            }
            Self::SpanningTreeEdgeOutsideAtomSet(atom) => {
                write!(
                    formatter,
                    "spanning-tree edge endpoint {atom:?} is outside the factor atom set"
                )
            }
            Self::InvalidEdgeRolePartition(variable) => {
                write!(
                    formatter,
                    "edge variable {variable:?} is not partitioned into disjoint Traversal and Ring values"
                )
            }
            Self::ConflictingEdgeRolePartition(variable) => write!(
                formatter,
                "edge variable {variable:?} uses inconsistent Traversal/Ring partitions across factors"
            ),
            Self::OverlappingSpanningTreeVariable(variable) => {
                write!(
                    formatter,
                    "constraint variable {variable:?} belongs to multiple spanning-tree factors"
                )
            }
            Self::InvalidTetrahedralBondCount(count) => write!(
                formatter,
                "a tetrahedral layout factor requires three or four bond variables, got {count}"
            ),
            Self::InvalidTetrahedralPatternDomain { variable, expected } => write!(
                formatter,
                "tetrahedral role-pattern variable {variable:?} must have initial domain {expected:?}"
            ),
            Self::InvalidTetrahedralPatternBit(bit) => write!(
                formatter,
                "tetrahedral layout pattern bit {bit} is repeated or outside the bond scope"
            ),
            Self::IncompleteTetrahedralPatternBits => formatter.write_str(
                "tetrahedral layout pattern bits must cover every bond position exactly once",
            ),
            Self::TetrahedralLayoutRowCountMismatch { expected, actual } => write!(
                formatter,
                "tetrahedral layout requires {expected} pattern rows, got {actual}"
            ),
            Self::TetrahedralOrderOutsideInitialDomain(variable) => write!(
                formatter,
                "tetrahedral layout contains an order outside the initial domain of {variable:?}"
            ),
        }
    }
}

impl std::error::Error for ConstraintModelError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_value_domain() -> Domain {
        Domain::from_indices([0, 1]).unwrap()
    }

    #[test]
    fn bond_role_domains_have_stable_values() {
        assert_eq!(BondRole::Traversal.value_index(), 0);
        assert_eq!(BondRole::Ring.value_index(), 1);
        assert_eq!(BondRole::role_domain(), two_value_domain());
        assert_eq!(
            BondRole::Traversal.singleton_domain(),
            Domain::singleton(0).unwrap()
        );
        assert_eq!(
            BondRole::Ring.singleton_domain(),
            Domain::singleton(1).unwrap()
        );
    }

    #[test]
    fn empty_constraint_model_is_valid() {
        let model = ConstraintModel::empty();

        assert_eq!(model.variable_count(), 0);
        assert_eq!(model.factor_count(), 0);
        assert_eq!(model.variable(VariableId::new(0)), None);
    }

    #[test]
    fn invalid_variable_is_rejected_without_consuming_an_id() {
        let mut builder = ConstraintModelBuilder::new();

        assert_eq!(
            builder.add_variable(Domain::empty()),
            Err(ConstraintModelError::EmptyInitialDomain)
        );
        assert_eq!(
            builder.add_variable(Domain::singleton(0).unwrap()).unwrap(),
            VariableId::new(0)
        );
    }

    #[test]
    fn model_assigns_stable_ids_and_adjacency() {
        let mut builder = ConstraintModelBuilder::new();
        let left_domain = two_value_domain();
        let right_domain = Domain::singleton(7).unwrap();
        let left = builder.add_variable(left_domain).unwrap();
        let right = builder.add_variable(right_domain).unwrap();
        let factor = builder
            .add_binary_relation(left, right, [(0, 7), (1, 7)])
            .unwrap();
        let model = builder.build();

        assert_eq!(left, VariableId::new(0));
        assert_eq!(right, VariableId::new(1));
        assert_eq!(factor, FactorId::new(0));
        assert_eq!(model.variable(left).unwrap().initial_domain(), left_domain);
        assert_eq!(
            model.variable(right).unwrap().initial_domain(),
            right_domain
        );
        assert_eq!(
            model.potential_factors_for_variable(left),
            Some(&[factor][..])
        );
        assert_eq!(
            model.potential_factors_for_variable(right),
            Some(&[factor][..])
        );
    }

    #[test]
    fn binary_relation_is_compiled_in_both_directions() {
        let mut builder = ConstraintModelBuilder::new();
        let left = builder.add_variable(two_value_domain()).unwrap();
        let right = builder.add_variable(two_value_domain()).unwrap();
        let factor_id = builder
            .add_binary_relation(left, right, [(0, 1), (1, 0), (1, 0)])
            .unwrap();
        let model = builder.build();
        let FactorDefinition::BinaryRelation(factor) = model.factor(factor_id).unwrap() else {
            panic!("expected binary relation factor");
        };

        assert_eq!(factor.variables(), &[left, right]);
        assert_eq!(factor.allowed_right(0), Domain::singleton(1).unwrap());
        assert_eq!(factor.allowed_right(1), Domain::singleton(0).unwrap());
        assert_eq!(factor.allowed_left(0), Domain::singleton(1).unwrap());
        assert_eq!(factor.allowed_left(1), Domain::singleton(0).unwrap());
    }

    #[test]
    fn relation_tables_match_the_initial_value_spans() {
        let mut builder = ConstraintModelBuilder::new();
        let left = builder
            .add_variable(Domain::from_indices([0, 2]).unwrap())
            .unwrap();
        let right = builder
            .add_variable(Domain::from_indices([1, 7]).unwrap())
            .unwrap();
        let factor_id = builder
            .add_binary_relation(left, right, [(0, 1), (2, 7)])
            .unwrap();
        let model = builder.build();
        let FactorDefinition::BinaryRelation(factor) = model.factor(factor_id).unwrap() else {
            panic!("expected binary relation factor");
        };

        assert_eq!(factor.allowed_right_by_left.len(), 3);
        assert_eq!(factor.allowed_left_by_right.len(), 8);
        assert!(factor.allowed_right(63).is_empty());
    }

    #[test]
    fn invalid_binary_factor_is_rejected_without_consuming_an_id() {
        let mut builder = ConstraintModelBuilder::new();
        let left = builder.add_variable(two_value_domain()).unwrap();
        let right = builder.add_variable(two_value_domain()).unwrap();

        assert_eq!(
            builder.add_binary_relation(left, right, [(0, 2)]),
            Err(ConstraintModelError::RelationValueOutsideInitialDomain {
                variable: right,
                value_index: 2,
            })
        );
        assert_eq!(
            builder.add_binary_relation(left, VariableId::new(99), [(0, 0)]),
            Err(ConstraintModelError::UnknownVariable(VariableId::new(99)))
        );
        assert_eq!(
            builder.add_binary_relation(left, left, [(0, 0)]),
            Err(ConstraintModelError::RepeatedVariableInFactor(left))
        );
        assert_eq!(
            builder
                .add_binary_relation(left, right, [(0, 0), (1, 1)])
                .unwrap(),
            FactorId::new(0)
        );
    }

    #[test]
    fn empty_binary_relation_is_a_valid_model_definition() {
        let mut builder = ConstraintModelBuilder::new();
        let left = builder.add_variable(two_value_domain()).unwrap();
        let right = builder.add_variable(two_value_domain()).unwrap();
        let factor_id = builder
            .add_binary_relation(left, right, std::iter::empty())
            .unwrap();
        let model = builder.build();
        let FactorDefinition::BinaryRelation(factor) = model.factor(factor_id).unwrap() else {
            panic!("expected binary relation factor");
        };

        assert!(factor.allowed_right(0).is_empty());
        assert!(factor.allowed_left(0).is_empty());
    }

    #[test]
    fn spanning_tree_factor_preserves_sorted_atoms_edges_and_adjacency() {
        let mut builder = ConstraintModelBuilder::new();
        let first = builder.add_variable(BondRole::role_domain()).unwrap();
        let second = builder.add_variable(BondRole::role_domain()).unwrap();
        let atoms = [AtomId::new(2), AtomId::new(0), AtomId::new(1)];
        let factor_id = builder
            .add_spanning_tree(
                atoms,
                [
                    SpanningTreeEdge::new(first, AtomId::new(0), AtomId::new(1)),
                    SpanningTreeEdge::new(second, AtomId::new(1), AtomId::new(2)),
                ],
            )
            .unwrap();
        let model = builder.build();
        let FactorDefinition::SpanningTree(factor) = model.factor(factor_id).unwrap() else {
            panic!("expected spanning-tree factor");
        };

        assert_eq!(
            factor.atoms(),
            &[AtomId::new(0), AtomId::new(1), AtomId::new(2)]
        );
        assert_eq!(factor.variables(), &[first, second]);
        assert_eq!(
            model.potential_factors_for_variable(first),
            Some(&[factor_id][..])
        );
        assert_eq!(
            model.potential_factors_for_variable(second),
            Some(&[factor_id][..])
        );
    }

    #[test]
    fn isolated_atom_spanning_tree_has_empty_scope() {
        let mut builder = ConstraintModelBuilder::new();
        let factor_id = builder
            .add_spanning_tree([AtomId::new(7)], std::iter::empty())
            .unwrap();
        let model = builder.build();
        let FactorDefinition::SpanningTree(factor) = model.factor(factor_id).unwrap() else {
            panic!("expected spanning-tree factor");
        };

        assert_eq!(factor.atoms(), &[AtomId::new(7)]);
        assert!(factor.edges().is_empty());
        assert!(factor.variables().is_empty());
    }

    #[test]
    fn invalid_spanning_tree_definition_is_rejected_without_consuming_an_id() {
        let mut builder = ConstraintModelBuilder::new();
        let role = builder.add_variable(BondRole::role_domain()).unwrap();
        let invalid_role = builder
            .add_variable(Domain::from_indices([0, 2]).unwrap())
            .unwrap();
        let globally_overlapping_role = builder
            .add_variable(Domain::from_indices([0, 1]).unwrap())
            .unwrap();

        assert_eq!(
            builder.add_spanning_tree([], std::iter::empty()),
            Err(ConstraintModelError::EmptySpanningTreeAtomSet)
        );
        assert_eq!(
            builder.add_spanning_tree([AtomId::new(0), AtomId::new(0)], std::iter::empty()),
            Err(ConstraintModelError::RepeatedAtomInSpanningTree)
        );
        assert_eq!(
            builder.add_spanning_tree(
                [AtomId::new(0)],
                [SpanningTreeEdge::new(role, AtomId::new(0), AtomId::new(0),)],
            ),
            Err(ConstraintModelError::SpanningTreeSelfEdge(AtomId::new(0)))
        );
        assert_eq!(
            builder.add_spanning_tree(
                [AtomId::new(0), AtomId::new(1)],
                [SpanningTreeEdge::new(role, AtomId::new(0), AtomId::new(2),)],
            ),
            Err(ConstraintModelError::SpanningTreeEdgeOutsideAtomSet(
                AtomId::new(2)
            ))
        );
        assert_eq!(
            builder.add_spanning_tree(
                [AtomId::new(0), AtomId::new(1)],
                [SpanningTreeEdge::new(
                    invalid_role,
                    AtomId::new(0),
                    AtomId::new(1),
                )],
            ),
            Err(ConstraintModelError::InvalidEdgeRolePartition(invalid_role))
        );
        assert_eq!(
            builder.add_spanning_tree(
                [AtomId::new(0), AtomId::new(1)],
                [SpanningTreeEdge::with_role_partition(
                    globally_overlapping_role,
                    AtomId::new(0),
                    AtomId::new(1),
                    EdgeRolePartition::new(
                        Domain::from_indices([0, 2]).unwrap(),
                        Domain::from_indices([1, 2]).unwrap(),
                    ),
                )],
            ),
            Err(ConstraintModelError::InvalidEdgeRolePartition(
                globally_overlapping_role
            ))
        );
        assert_eq!(
            builder.add_spanning_tree(
                [AtomId::new(0), AtomId::new(1), AtomId::new(2)],
                [
                    SpanningTreeEdge::new(role, AtomId::new(0), AtomId::new(1)),
                    SpanningTreeEdge::new(role, AtomId::new(1), AtomId::new(2)),
                ],
            ),
            Err(ConstraintModelError::RepeatedVariableInFactor(role))
        );
        assert_eq!(
            builder
                .add_spanning_tree([AtomId::new(0)], std::iter::empty())
                .unwrap(),
            FactorId::new(0)
        );
    }

    #[test]
    fn spanning_tree_projectors_cannot_overlap_on_a_variable() {
        let mut builder = ConstraintModelBuilder::new();
        let role = builder.add_variable(BondRole::role_domain()).unwrap();
        let atoms = [AtomId::new(0), AtomId::new(1)];
        assert_eq!(
            builder
                .add_spanning_tree(atoms, [SpanningTreeEdge::new(role, atoms[0], atoms[1])],)
                .unwrap(),
            FactorId::new(0)
        );

        assert_eq!(
            builder.add_spanning_tree(atoms, [SpanningTreeEdge::new(role, atoms[0], atoms[1])],),
            Err(ConstraintModelError::OverlappingSpanningTreeVariable(role))
        );
        assert_eq!(
            builder
                .add_spanning_tree([AtomId::new(2)], std::iter::empty())
                .unwrap(),
            FactorId::new(1)
        );
    }

    #[test]
    fn tetrahedral_layout_retains_latent_context_and_role_mapping() {
        let mut builder = ConstraintModelBuilder::new();
        let order = builder
            .add_variable(Domain::from_indices(0_u8..4).unwrap())
            .unwrap();
        let pattern = builder
            .add_variable(Domain::from_indices(0_u8..8).unwrap())
            .unwrap();
        let role_domain = Domain::from_indices([0, 1, 2]).unwrap();
        let partition = EdgeRolePartition::new(
            Domain::singleton(0).unwrap(),
            Domain::from_indices([1, 2]).unwrap(),
        );
        let bonds = (0_u8..3)
            .map(|bit| {
                let variable = builder.add_variable(role_domain).unwrap();
                TetrahedralLayoutBond::new(variable, partition, bit)
            })
            .collect::<Vec<_>>();
        let rows = (0_u8..8)
            .map(|pattern| Domain::singleton(pattern % 4).unwrap())
            .collect::<Vec<_>>();

        let factor_id = builder
            .add_latent_tetrahedral_layout(order, pattern, bonds.clone(), rows.clone())
            .unwrap();
        let model = builder.build();
        let FactorDefinition::TetrahedralLayout(factor) = model.factor(factor_id).unwrap() else {
            panic!("expected tetrahedral layout factor");
        };

        assert_eq!(
            model.factor_activation(factor_id),
            Some(FactorActivation::Latent)
        );
        assert!(model.initial_factor_ids().is_empty());
        assert_eq!(model.initial_factors_for_variable(order), Some(&[][..]));
        assert_eq!(
            model.potential_factors_for_variable(order),
            Some(&[factor_id][..])
        );
        assert_eq!(factor.order_variable(), order);
        assert_eq!(factor.role_pattern_variable(), pattern);
        assert_eq!(factor.bonds(), bonds);
        assert_eq!(
            factor.variables(),
            &[
                order,
                pattern,
                bonds[0].decision_variable(),
                bonds[1].decision_variable(),
                bonds[2].decision_variable()
            ]
        );
        for (pattern, expected) in rows.into_iter().enumerate() {
            assert_eq!(factor.allowed_orders(pattern as u8), expected);
        }
    }

    #[test]
    fn edge_role_partition_is_one_model_wide_structural_fact() {
        let domain = Domain::from_indices([0, 1, 2]).unwrap();
        let first = EdgeRolePartition::new(
            Domain::singleton(0).unwrap(),
            Domain::from_indices([1, 2]).unwrap(),
        );
        let second = EdgeRolePartition::new(
            Domain::from_indices([0, 1]).unwrap(),
            Domain::singleton(2).unwrap(),
        );
        let rows = || [Domain::singleton(0).unwrap(); 8];

        let mut layout_first = ConstraintModelBuilder::new();
        let order = layout_first
            .add_variable(Domain::singleton(0).unwrap())
            .unwrap();
        let pattern = layout_first
            .add_variable(Domain::from_indices(0_u8..8).unwrap())
            .unwrap();
        let roles: [VariableId; 3] =
            std::array::from_fn(|_| layout_first.add_variable(domain).unwrap());
        layout_first
            .add_latent_tetrahedral_layout(
                order,
                pattern,
                roles
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(bit, variable)| TetrahedralLayoutBond::new(variable, first, bit as u8)),
                rows(),
            )
            .unwrap();
        assert_eq!(
            layout_first.add_spanning_tree(
                [AtomId::new(0), AtomId::new(1)],
                [SpanningTreeEdge::with_role_partition(
                    roles[0],
                    AtomId::new(0),
                    AtomId::new(1),
                    second,
                )],
            ),
            Err(ConstraintModelError::ConflictingEdgeRolePartition(roles[0]))
        );

        let mut spanning_first = ConstraintModelBuilder::new();
        let order = spanning_first
            .add_variable(Domain::singleton(0).unwrap())
            .unwrap();
        let pattern = spanning_first
            .add_variable(Domain::from_indices(0_u8..8).unwrap())
            .unwrap();
        let roles: [VariableId; 3] =
            std::array::from_fn(|_| spanning_first.add_variable(domain).unwrap());
        spanning_first
            .add_spanning_tree(
                [AtomId::new(0), AtomId::new(1)],
                [SpanningTreeEdge::with_role_partition(
                    roles[0],
                    AtomId::new(0),
                    AtomId::new(1),
                    first,
                )],
            )
            .unwrap();
        assert_eq!(
            spanning_first.add_latent_tetrahedral_layout(
                order,
                pattern,
                roles.iter().copied().enumerate().map(|(bit, variable)| {
                    TetrahedralLayoutBond::new(variable, second, bit as u8)
                }),
                rows(),
            ),
            Err(ConstraintModelError::ConflictingEdgeRolePartition(roles[0]))
        );
    }

    #[test]
    fn tetrahedral_layout_rejects_invalid_pattern_shape() {
        let mut builder = ConstraintModelBuilder::new();
        let order = builder
            .add_variable(Domain::from_indices(0_u8..4).unwrap())
            .unwrap();
        let malformed_pattern = builder
            .add_variable(Domain::from_indices(0_u8..4).unwrap())
            .unwrap();
        let roles = (0..3)
            .map(|_| builder.add_variable(BondRole::role_domain()).unwrap())
            .collect::<Vec<_>>();
        let bonds = roles
            .iter()
            .copied()
            .enumerate()
            .map(|(bit, variable)| {
                TetrahedralLayoutBond::new(variable, EdgeRolePartition::bond_role(), bit as u8)
            })
            .collect::<Vec<_>>();

        assert_eq!(
            builder.add_latent_tetrahedral_layout(
                order,
                malformed_pattern,
                bonds,
                vec![Domain::singleton(0).unwrap(); 8],
            ),
            Err(ConstraintModelError::InvalidTetrahedralPatternDomain {
                variable: malformed_pattern,
                expected: Domain::from_indices(0_u8..8).unwrap(),
            })
        );
    }
}
