#![forbid(unsafe_code)]

//! South Star 2: a pure-Rust incremental constraint-driven SMILES walker.
//!
//! This crate intentionally starts below the Python and RDKit boundaries.

use std::fmt;

macro_rules! define_id {
    ($name:ident, $repr:ty) => {
        #[repr(transparent)]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name($repr);

        impl $name {
            pub const fn new(value: $repr) -> Self {
                Self(value)
            }

            pub const fn get(self) -> $repr {
                self.0
            }

            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

define_id!(AtomId, u32);
define_id!(BondId, u32);
define_id!(FactorId, u32);
define_id!(VariableId, u32);
define_id!(TokenId, u16);

pub const DOMAIN_VALUE_CAPACITY: u8 = u64::BITS as u8;

/// A compact finite domain with at most 64 values.
#[must_use]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Domain(u64);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DomainError {
    value_index: u8,
}

impl DomainError {
    pub const fn value_index(self) -> u8 {
        self.value_index
    }
}

impl fmt::Display for DomainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "finite-domain value index {} is outside the supported range 0..{}",
            self.value_index, DOMAIN_VALUE_CAPACITY
        )
    }
}

impl std::error::Error for DomainError {}

impl Domain {
    pub const fn empty() -> Self {
        Self(0)
    }

    pub fn singleton(value_index: u8) -> Result<Self, DomainError> {
        Self::validate_value_index(value_index)?;
        Ok(Self(1_u64 << value_index))
    }

    pub fn from_indices(
        values: impl IntoIterator<Item = u8>,
    ) -> Result<Self, DomainError> {
        let mut bits = 0_u64;
        for value_index in values {
            Self::validate_value_index(value_index)?;
            bits |= 1_u64 << value_index;
        }
        Ok(Self(bits))
    }

    pub const fn contains(self, value_index: u8) -> bool {
        value_index < DOMAIN_VALUE_CAPACITY && self.0 & (1_u64 << value_index) != 0
    }

    pub const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn is_subset_of(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn is_singleton(self) -> bool {
        self.0.count_ones() == 1
    }

    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    pub fn iter(self) -> impl Iterator<Item = u8> {
        let mut bits = self.0;
        std::iter::from_fn(move || {
            if bits == 0 {
                return None;
            }
            let value_index = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            Some(value_index)
        })
    }

    const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    const fn validate_value_index(value_index: u8) -> Result<(), DomainError> {
        if value_index < DOMAIN_VALUE_CAPACITY {
            Ok(())
        } else {
            Err(DomainError { value_index })
        }
    }
}

/// Immutable definition of one solver-neutral finite-domain variable.
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
/// Support masks are stored in both directions so the native solver can revise
/// either domain without enumerating the relation's Cartesian product.
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
    pub fn variables(&self) -> [VariableId; 2] {
        match self {
            Self::BinaryRelation(factor) => factor.variables(),
        }
    }
}

/// Immutable molecule-local constraint definitions.
///
/// This slice defines variables, factors, and variable-to-factor adjacency only.
/// Solver state, factor activation, and propagation remain separate concerns.
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
        if left == right {
            return Err(ConstraintModelError::RepeatedVariableInFactor(left));
        }

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

        let mut allowed_right_by_left =
            vec![Domain::empty(); DOMAIN_VALUE_CAPACITY as usize];
        let mut allowed_left_by_right =
            vec![Domain::empty(); DOMAIN_VALUE_CAPACITY as usize];

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
    fn ids_are_compact_ordered_values() {
        let first = AtomId::new(2);
        let second = AtomId::new(7);

        assert!(first < second);
        assert_eq!(first.get(), 2);
        assert_eq!(second.index(), 7);
        assert_eq!(TokenId::new(3).index(), 3);
    }

    #[test]
    fn domains_build_intersect_and_union() {
        let left = Domain::from_indices([0, 2, 4]).unwrap();
        let right = Domain::from_indices([1, 2, 4]).unwrap();
        let intersection = left.intersect(right);
        let union = left.union(right);

        assert_eq!(intersection.iter().collect::<Vec<_>>(), vec![2, 4]);
        assert_eq!(intersection.len(), 2);
        assert_eq!(union.iter().collect::<Vec<_>>(), vec![0, 1, 2, 4]);
        assert!(intersection.is_subset_of(union));
        assert!(!intersection.is_singleton());
        assert!(!intersection.is_empty());
    }

    #[test]
    fn singleton_and_empty_domains_are_distinct() {
        let singleton = Domain::singleton(63).unwrap();

        assert!(singleton.contains(63));
        assert!(singleton.is_singleton());
        assert!(Domain::empty().is_empty());
    }

    #[test]
    fn out_of_range_domain_values_are_rejected() {
        let error = Domain::singleton(64).unwrap_err();

        assert_eq!(error.value_index(), 64);
    }

    #[test]
    fn empty_constraint_model_is_valid() {
        let model = ConstraintModel::empty();

        assert_eq!(model.variable_count(), 0);
        assert_eq!(model.factor_count(), 0);
        assert_eq!(model.variable(VariableId::new(0)), None);
    }

    #[test]
    fn constraint_model_assigns_stable_variable_and_factor_ids() {
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
        assert_eq!(model.variable_count(), 2);
        assert_eq!(model.factor_count(), 1);
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
        let FactorDefinition::BinaryRelation(factor) =
            model.factor(factor_id).unwrap();

        assert_eq!(factor.variables(), [left, right]);
        assert_eq!(
            factor.allowed_right(0),
            Domain::singleton(1).unwrap()
        );
        assert_eq!(
            factor.allowed_right(1),
            Domain::singleton(0).unwrap()
        );
        assert_eq!(
            factor.allowed_left(0),
            Domain::singleton(1).unwrap()
        );
        assert_eq!(
            factor.allowed_left(1),
            Domain::singleton(0).unwrap()
        );
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
            builder.add_binary_relation(left, left, [(0, 0)]),
            Err(ConstraintModelError::RepeatedVariableInFactor(left))
        );

        let factor = builder
            .add_binary_relation(left, right, [(0, 0), (1, 1)])
            .unwrap();
        assert_eq!(factor, FactorId::new(0));
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
        let FactorDefinition::BinaryRelation(factor) =
            model.factor(factor_id).unwrap();

        assert!(factor.allowed_right(0).is_empty());
        assert!(factor.allowed_left(0).is_empty());
    }
}
