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

pub(crate) fn full_role_pattern_domain(bond_count: usize) -> Domain {
    assert!((3..=4).contains(&bond_count));
    Domain::from_indices(0..(1_u8 << bond_count)).unwrap()
}

pub(crate) fn layout_order_rows(
    reference_order: &[TetrahedralLigand; 4],
    context_prefix: &[TetrahedralLigand],
    bond_bits: &[(BondId, u8)],
) -> Vec<Domain> {
    assert!((3..=4).contains(&bond_bits.len()));
    let bit_for_bond = |bond: BondId| {
        bond_bits
            .iter()
            .find_map(|(candidate, bit)| (*candidate == bond).then_some(*bit))
            .expect("every prepared bond ligand must own one role-pattern bit")
    };
    (0..(1_u8 << bond_bits.len()))
        .map(|pattern| {
            Domain::from_indices((0..TETRAHEDRAL_ORDER_COUNT as u8).filter(|value| {
                let order = order_for_value(reference_order, *value)
                    .expect("tetrahedral order value must be prepared");
                if !order.starts_with(context_prefix) {
                    return false;
                }
                if context_prefix.iter().any(|ligand| {
                    matches!(ligand, TetrahedralLigand::Bond(bond)
                        if pattern & (1_u8 << bit_for_bond(*bond)) != 0)
                }) {
                    return false;
                }
                let mut saw_traversal = false;
                for ligand in &order[context_prefix.len()..] {
                    let TetrahedralLigand::Bond(bond) = ligand else {
                        return false;
                    };
                    let is_ring = pattern & (1_u8 << bit_for_bond(*bond)) != 0;
                    if is_ring && saw_traversal {
                        return false;
                    }
                    saw_traversal |= !is_ring;
                }
                true
            }))
            .unwrap()
        })
        .collect()
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

    #[test]
    fn layout_rows_are_context_then_rings_then_traversal_children() {
        let reference = reference();
        let bond_bits = [
            (BondId::new(0), 0),
            (BondId::new(1), 1),
            (BondId::new(2), 2),
        ];
        let root_rows = layout_order_rows(
            &reference,
            &[TetrahedralLigand::VirtualHydrogen],
            &bond_bits,
        );
        let entered_rows = layout_order_rows(
            &reference,
            &[
                TetrahedralLigand::Bond(BondId::new(1)),
                TetrahedralLigand::VirtualHydrogen,
            ],
            &bond_bits,
        );

        for (pattern, rows) in [(0_u8, &root_rows), (3, &root_rows), (5, &root_rows)] {
            for value in rows[pattern as usize].iter() {
                let order = order_for_value(&reference, value).unwrap();
                assert_eq!(order[0], TetrahedralLigand::VirtualHydrogen);
                let suffix = &order[1..];
                let first_traversal = suffix.iter().position(|ligand| {
                    let TetrahedralLigand::Bond(bond) = ligand else {
                        unreachable!()
                    };
                    let bit = bond_bits
                        .iter()
                        .find_map(|(candidate, bit)| (*candidate == *bond).then_some(*bit))
                        .unwrap();
                    pattern & (1_u8 << bit) == 0
                });
                if let Some(first_traversal) = first_traversal {
                    assert!(suffix[first_traversal..].iter().all(|ligand| {
                        let TetrahedralLigand::Bond(bond) = ligand else {
                            unreachable!()
                        };
                        let bit = bond_bits
                            .iter()
                            .find_map(|(candidate, bit)| (*candidate == *bond).then_some(*bit))
                            .unwrap();
                        pattern & (1_u8 << bit) == 0
                    }));
                }
            }
        }
        assert!(entered_rows[2].is_empty());
        assert!(!entered_rows[0].is_empty());
        assert!(!entered_rows[1].is_empty());
    }
}
