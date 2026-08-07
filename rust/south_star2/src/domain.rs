//! Compact finite domains for the incremental constraint engine.

use std::fmt;

pub const DOMAIN_VALUE_CAPACITY: u8 = u64::BITS as u8;

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

    pub fn from_indices(values: impl IntoIterator<Item = u8>) -> Result<Self, DomainError> {
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

    pub(crate) const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    pub(crate) const fn value_span(self) -> usize {
        if self.0 == 0 {
            0
        } else {
            (u64::BITS - self.0.leading_zeros()) as usize
        }
    }

    const fn validate_value_index(value_index: u8) -> Result<(), DomainError> {
        if value_index < DOMAIN_VALUE_CAPACITY {
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
    fn value_span_tracks_only_the_highest_represented_value() {
        assert_eq!(Domain::from_indices([0, 3]).unwrap().value_span(), 4);
        assert_eq!(Domain::singleton(63).unwrap().value_span(), 64);
        assert_eq!(Domain::empty().value_span(), 0);
    }
}
