//! South Star 2: a pure-Rust incremental constraint-driven SMILES walker.
//!
//! This crate intentionally starts below the Python and RDKit boundaries.

use std::fmt;

macro_rules! define_id {
    ($name:ident, $repr:ty) => {
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

const DOMAIN_CAPACITY: u8 = u64::BITS as u8;

/// A compact finite domain with at most 64 values.
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
            "finite-domain value index {} exceeds capacity {}",
            self.value_index, DOMAIN_CAPACITY
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
        value_index < DOMAIN_CAPACITY && self.0 & (1_u64 << value_index) != 0
    }

    pub const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn is_singleton(self) -> bool {
        self.0.count_ones() == 1
    }

    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    pub fn iter(self) -> impl Iterator<Item = u8> {
        (0..DOMAIN_CAPACITY).filter(move |&value_index| self.contains(value_index))
    }

    const fn validate_value_index(value_index: u8) -> Result<(), DomainError> {
        if value_index < DOMAIN_CAPACITY {
            Ok(())
        } else {
            Err(DomainError { value_index })
        }
    }
}

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
    fn domains_build_and_intersect() {
        let left = Domain::from_indices([0, 2, 4]).unwrap();
        let right = Domain::from_indices([1, 2, 4]).unwrap();
        let intersection = left.intersect(right);

        assert_eq!(intersection.iter().collect::<Vec<_>>(), vec![2, 4]);
        assert_eq!(intersection.len(), 2);
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
}
