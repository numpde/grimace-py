//! Compact stable identifiers used by the South Star 2 kernel.

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
        assert_eq!(BondId::new(3).index(), 3);
    }
}
