//! Prepared four-ligand tetrahedral order domains.

use crate::domain::Domain;
use crate::ids::BondId;

pub(crate) const TETRAHEDRAL_ORDER_COUNT: usize = 24;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum TetrahedralLigand {
    Bond(BondId),
    VirtualHydrogen,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum TetrahedralParity {
    Even,
    Odd,
}

impl TetrahedralParity {
    pub(crate) const ALL: [Self; 2] = [Self::Even, Self::Odd];

    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Even => 0,
            Self::Odd => 1,
        }
    }
}

pub(crate) fn full_order_domain() -> Domain {
    Domain::from_indices(0..u8::try_from(TETRAHEDRAL_ORDER_COUNT).unwrap()).unwrap()
}

pub(crate) fn parity_domain(parity: TetrahedralParity) -> Domain {
    domain_for(|permutation| permutation_parity(permutation) == parity)
}

pub(crate) fn prefix_domain(
    reference_order: &[TetrahedralLigand; 4],
    prefix: &[TetrahedralLigand],
) -> Domain {
    assert!(prefix.len() <= reference_order.len());
    domain_for(|permutation| {
        permutation
            .iter()
            .take(prefix.len())
            .enumerate()
            .all(|(offset, reference_index)| reference_order[*reference_index] == prefix[offset])
    })
}

pub(crate) fn singleton_order(
    reference_order: &[TetrahedralLigand; 4],
    order: &[TetrahedralLigand],
) -> Domain {
    assert_eq!(order.len(), reference_order.len());
    prefix_domain(reference_order, order)
}

pub(crate) fn order_for_value(
    reference_order: &[TetrahedralLigand; 4],
    value: u8,
) -> Option<[TetrahedralLigand; 4]> {
    permutations()
        .nth(value as usize)
        .map(|permutation| permutation.map(|reference_index| reference_order[reference_index]))
}

fn domain_for(mut predicate: impl FnMut(&[usize; 4]) -> bool) -> Domain {
    Domain::from_indices(
        permutations()
            .enumerate()
            .filter_map(|(value, permutation)| {
                predicate(&permutation).then_some(u8::try_from(value).unwrap())
            }),
    )
    .unwrap()
}

fn permutations() -> impl Iterator<Item = [usize; 4]> {
    (0..4).flat_map(|a| {
        (0..4).filter(move |b| *b != a).flat_map(move |b| {
            (0..4).filter(move |c| *c != a && *c != b).map(move |c| {
                let d = (0..4).find(|d| *d != a && *d != b && *d != c).unwrap();
                [a, b, c, d]
            })
        })
    })
}

fn permutation_parity(permutation: &[usize; 4]) -> TetrahedralParity {
    let inversions = (0..4)
        .flat_map(|left| ((left + 1)..4).map(move |right| (left, right)))
        .filter(|(left, right)| permutation[*left] > permutation[*right])
        .count();
    if inversions % 2 == 0 {
        TetrahedralParity::Even
    } else {
        TetrahedralParity::Odd
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn reference() -> [TetrahedralLigand; 4] {
        [
            TetrahedralLigand::Bond(BondId::new(0)),
            TetrahedralLigand::Bond(BondId::new(1)),
            TetrahedralLigand::Bond(BondId::new(2)),
            TetrahedralLigand::VirtualHydrogen,
        ]
    }

    #[test]
    fn all_orders_are_unique_and_split_evenly_by_parity() {
        let reference = reference();
        let orders = (0..TETRAHEDRAL_ORDER_COUNT as u8)
            .map(|value| order_for_value(&reference, value).unwrap())
            .collect::<BTreeSet<_>>();

        assert_eq!(orders.len(), TETRAHEDRAL_ORDER_COUNT);
        assert_eq!(parity_domain(TetrahedralParity::Even).len(), 12);
        assert_eq!(parity_domain(TetrahedralParity::Odd).len(), 12);
        assert!(parity_domain(TetrahedralParity::Even)
            .intersect(parity_domain(TetrahedralParity::Odd))
            .is_empty());
        assert_eq!(
            parity_domain(TetrahedralParity::Even).union(parity_domain(TetrahedralParity::Odd)),
            full_order_domain()
        );
    }

    #[test]
    fn every_prefix_mask_matches_direct_order_comparison() {
        let reference = reference();
        for value in 0..TETRAHEDRAL_ORDER_COUNT as u8 {
            let order = order_for_value(&reference, value).unwrap();
            for prefix_len in 0..=4 {
                let mask = prefix_domain(&reference, &order[..prefix_len]);
                for candidate in 0..TETRAHEDRAL_ORDER_COUNT as u8 {
                    let candidate_order = order_for_value(&reference, candidate).unwrap();
                    assert_eq!(
                        mask.contains(candidate),
                        candidate_order[..prefix_len] == order[..prefix_len]
                    );
                }
            }
            assert_eq!(
                singleton_order(&reference, &order),
                Domain::singleton(value).unwrap()
            );
        }
    }
}
