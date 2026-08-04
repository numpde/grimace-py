//! Solver-neutral constraint definitions prepared once per molecule.

use std::fmt;

use crate::domain::Domain;
use crate::ids::{FactorId, VariableId};

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
    left: VariableId,
    right: VariableId,
    allowed_right_by_left: Box<[Domain]>,
    allowed_left_by_right: Box<[Domain]>,
}

impl BinaryRelationFactor {
    pub const fn left(&self) -> VariableId {
        self.left
    }

    pub const fn right(&self) -> VariableId {
        self.right
    }

    pub const fn variables(&self) -> [VariableId; 2] {
        [self.left, self.right]
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FactorDefinition {
    BinaryRelation(BinaryRelationFactor),
}

impl FactorDefinition {
    pub const fn variables(&self) -> [VariableId; 2] {
        match self {
            Self::BinaryRelation(factor) => factor.variables(),
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

        let value = u32::try_from(self.factors.len())
            .map_err(|_| ConstraintModelError::FactorCapacityExceeded)?;
        let factor = FactorId::new(value);
        self.factors
            .push(FactorDefinition::BinaryRelation(BinaryRelationFactor {
                left,
                right,
                allowed_right_by_left: allowed_right_by_left.into_boxed_slice(),
                allowed_left_by_right: allowed_left_by_right.into_boxed_slice(),
            }));
        self.factors_by_variable[left.index()].push(factor);
        self.factors_by_variable[right.index()].push(factor);
        Ok(factor)
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
                write!(
                    formatter,
                    "binary relation repeats variable {variable:?} in its scope"
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
        let FactorDefinition::BinaryRelation(factor) = model.factor(factor_id).unwrap();

        assert_eq!(factor.variables(), [left, right]);
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
        let FactorDefinition::BinaryRelation(factor) = model.factor(factor_id).unwrap();

        assert_eq!(factor.allowed_right_by_left.len(), 3);
        assert_eq!(factor.allowed_left_by_right.len(), 8);
        assert!(factor.allowed_right(63).is_empty());
    }

    #[test]
    fn invalid_factor_is_rejected_without_consuming_an_id() {
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
        let FactorDefinition::BinaryRelation(factor) = model.factor(factor_id).unwrap();

        assert!(factor.allowed_right(0).is_empty());
        assert!(factor.allowed_left(0).is_empty());
    }
}
