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
    role_variable: VariableId,
    a: AtomId,
    b: AtomId,
}

impl SpanningTreeEdge {
    pub const fn new(role_variable: VariableId, a: AtomId, b: AtomId) -> Self {
        Self {
            role_variable,
            a,
            b,
        }
    }

    pub const fn role_variable(self) -> VariableId {
        self.role_variable
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FactorDefinition {
    BinaryRelation(BinaryRelationFactor),
    SpanningTree(SpanningTreeFactor),
}

impl FactorDefinition {
    pub fn variables(&self) -> &[VariableId] {
        match self {
            Self::BinaryRelation(factor) => factor.variables(),
            Self::SpanningTree(factor) => factor.variables(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintModel {
    variables: Box<[VariableDefinition]>,
    factors: Box<[FactorDefinition]>,
    factors_by_variable: Box<[Box<[FactorId]>]>,
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

    pub fn factors_for_variable(&self, variable: VariableId) -> Option<&[FactorId]> {
        self.factors_by_variable
            .get(variable.index())
            .map(AsRef::as_ref)
    }

    pub(crate) fn initial_domains(&self) -> impl Iterator<Item = Domain> + '_ {
        self.variables.iter().map(VariableDefinition::initial_domain)
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
    factors_by_variable: Vec<Vec<FactorId>>,
}

impl ConstraintModelBuilder {
    pub const fn new() -> Self {
        Self {
            variables: Vec::new(),
            factors: Vec::new(),
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
                return Err(
                    ConstraintModelError::RelationValueOutsideInitialDomain {
                        variable: left,
                        value_index: left_value,
                    },
                );
            }
            if !right_domain.contains(right_value) {
                return Err(
                    ConstraintModelError::RelationValueOutsideInitialDomain {
                        variable: right,
                        value_index: right_value,
                    },
                );
            }

            allowed_right_by_left[left_value as usize] =
                allowed_right_by_left[left_value as usize]
                    .union(Domain::from_bits(1_u64 << right_value));
            allowed_left_by_right[right_value as usize] =
                allowed_left_by_right[right_value as usize]
                    .union(Domain::from_bits(1_u64 << left_value));
        }

        self.push_factor(FactorDefinition::BinaryRelation(BinaryRelationFactor {
            variables: [left, right],
            allowed_right_by_left: allowed_right_by_left.into_boxed_slice(),
            allowed_left_by_right: allowed_left_by_right.into_boxed_slice(),
        }))
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

            let variable = edge.role_variable;
            let initial_domain = self
                .variables
                .get(variable.index())
                .ok_or(ConstraintModelError::UnknownVariable(variable))?
                .initial_domain;
            if !initial_domain.is_subset_of(BondRole::role_domain()) {
                return Err(ConstraintModelError::InvalidBondRoleDomain(variable));
            }
            if !seen_variables.insert(variable) {
                return Err(ConstraintModelError::RepeatedVariableInFactor(variable));
            }
            variables.push(variable);
        }

        self.push_factor(FactorDefinition::SpanningTree(SpanningTreeFactor {
            atoms: atoms.into_boxed_slice(),
            edges: edges.into_boxed_slice(),
            variables: variables.into_boxed_slice(),
        }))
    }

    pub fn build(self) -> ConstraintModel {
        ConstraintModel {
            variables: self.variables.into_boxed_slice(),
            factors: self.factors.into_boxed_slice(),
            factors_by_variable: self
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
    ) -> Result<FactorId, ConstraintModelError> {
        let value = u32::try_from(self.factors.len())
            .map_err(|_| ConstraintModelError::FactorCapacityExceeded)?;
        let factor = FactorId::new(value);
        for variable in definition.variables().iter().copied() {
            self.factors_by_variable[variable.index()].push(factor);
        }
        self.factors.push(definition);
        Ok(factor)
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
    InvalidBondRoleDomain(VariableId),
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
                write!(formatter, "constraint factor references unknown variable {variable:?}")
            }
            Self::RepeatedVariableInFactor(variable) => {
                write!(formatter, "constraint factor repeats variable {variable:?} in its scope")
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
                write!(formatter, "a spanning-tree factor cannot contain a self-edge at {atom:?}")
            }
            Self::SpanningTreeEdgeOutsideAtomSet(atom) => {
                write!(
                    formatter,
                    "spanning-tree edge endpoint {atom:?} is outside the factor atom set"
                )
            }
            Self::InvalidBondRoleDomain(variable) => {
                write!(
                    formatter,
                    "spanning-tree variable {variable:?} contains a value outside the bond-role domain"
                )
            }
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
            builder
                .add_variable(Domain::singleton(0).unwrap())
                .unwrap(),
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
        assert_eq!(model.variable(right).unwrap().initial_domain(), right_domain);
        assert_eq!(model.factors_for_variable(left), Some(&[factor][..]));
        assert_eq!(model.factors_for_variable(right), Some(&[factor][..]));
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
        assert_eq!(model.factors_for_variable(first), Some(&[factor_id][..]));
        assert_eq!(model.factors_for_variable(second), Some(&[factor_id][..]));
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

        assert_eq!(
            builder.add_spanning_tree([], std::iter::empty()),
            Err(ConstraintModelError::EmptySpanningTreeAtomSet)
        );
        assert_eq!(
            builder.add_spanning_tree(
                [AtomId::new(0), AtomId::new(0)],
                std::iter::empty()
            ),
            Err(ConstraintModelError::RepeatedAtomInSpanningTree)
        );
        assert_eq!(
            builder.add_spanning_tree(
                [AtomId::new(0)],
                [SpanningTreeEdge::new(
                    role,
                    AtomId::new(0),
                    AtomId::new(0),
                )],
            ),
            Err(ConstraintModelError::SpanningTreeSelfEdge(AtomId::new(0)))
        );
        assert_eq!(
            builder.add_spanning_tree(
                [AtomId::new(0), AtomId::new(1)],
                [SpanningTreeEdge::new(
                    role,
                    AtomId::new(0),
                    AtomId::new(2),
                )],
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
            Err(ConstraintModelError::InvalidBondRoleDomain(invalid_role))
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
}
