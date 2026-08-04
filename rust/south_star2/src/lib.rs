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

/// Immutable molecule-local constraint definitions.
///
/// Factor definitions and adjacency are deliberately deferred until the first
/// factor type is implemented. Stable variable identity and nonempty initial
/// domains are established now without committing to a backend representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintModel {
    variables: Box<[VariableDefinition]>,
}

impl ConstraintModel {
    pub fn empty() -> Self {
        ConstraintModelBuilder::new().build()
    }

    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    pub fn variable(&self, variable: VariableId) -> Option<&VariableDefinition> {
        self.variables.get(variable.index())
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
}

impl ConstraintModelBuilder {
    pub const fn new() -> Self {
        Self {
            variables: Vec::new(),
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
        Ok(variable)
    }

    pub fn build(self) -> ConstraintModel {
        ConstraintModel {
            variables: self.variables.into_boxed_slice(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstraintModelError {
    EmptyInitialDomain,
    VariableCapacityExceeded,
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
        }
    }
}

impl std::error::Error for ConstraintModelError {}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(model.variable(VariableId::new(0)), None);
    }

    #[test]
    fn constraint_model_assigns_stable_variable_ids() {
        let mut builder = ConstraintModelBuilder::new();
        let first_domain = Domain::from_indices([0, 1]).unwrap();
        let second_domain = Domain::singleton(7).unwrap();
        let first = builder.add_variable(first_domain).unwrap();
        let second = builder.add_variable(second_domain).unwrap();
        let model = builder.build();

        assert_eq!(first, VariableId::new(0));
        assert_eq!(second, VariableId::new(1));
        assert_eq!(model.variable_count(), 2);
        assert_eq!(model.variable(first).unwrap().initial_domain(), first_domain);
        assert_eq!(model.variable(second).unwrap().initial_domain(), second_domain);
    }

    #[test]
    fn constraint_model_rejects_empty_variable_domains_atomically() {
        let mut builder = ConstraintModelBuilder::new();

        assert_eq!(
            builder.add_variable(Domain::empty()),
            Err(ConstraintModelError::EmptyInitialDomain)
        );

        let variable = builder
            .add_variable(Domain::singleton(0).unwrap())
            .unwrap();
        let model = builder.build();
        assert_eq!(variable, VariableId::new(0));
        assert_eq!(model.variable_count(), 1);
    }
}
