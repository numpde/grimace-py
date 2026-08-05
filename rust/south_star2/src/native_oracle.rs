//! Exhaustive tiny-CSP oracle checks for the native binary solver.

use std::sync::Arc;

use crate::domain::Domain;
use crate::model::ConstraintModelBuilder;
use crate::native::{NativeSolverError, NativeSolverState};

fn relation_rows(mask: u16) -> Vec<(u8, u8)> {
    let mut rows = Vec::new();
    for left in 0_u8..2 {
        for right in 0_u8..2 {
            let bit = usize::from(left) * 2 + usize::from(right);
            if mask & (1_u16 << bit) != 0 {
                rows.push((left, right));
            }
        }
    }
    rows
}

fn relation_accepts(mask: u16, left: u8, right: u8) -> bool {
    let bit = usize::from(left) * 2 + usize::from(right);
    mask & (1_u16 << bit) != 0
}

fn exhaustive_triangle_supports(masks: [u16; 3]) -> Option<[Domain; 3]> {
    let mut supported = [Domain::empty(); 3];
    let mut found = false;

    for x in 0_u8..2 {
        for y in 0_u8..2 {
            for z in 0_u8..2 {
                if !relation_accepts(masks[0], x, y)
                    || !relation_accepts(masks[1], y, z)
                    || !relation_accepts(masks[2], z, x)
                {
                    continue;
                }
                found = true;
                supported[0] = supported[0].union(Domain::singleton(x).unwrap());
                supported[1] = supported[1].union(Domain::singleton(y).unwrap());
                supported[2] = supported[2].union(Domain::singleton(z).unwrap());
            }
        }
    }

    found.then_some(supported)
}

#[test]
fn every_boolean_triangle_matches_exhaustive_assignment_support() {
    let binary = Domain::from_indices([0, 1]).unwrap();

    for xy in 0_u16..16 {
        for yz in 0_u16..16 {
            for zx in 0_u16..16 {
                let masks = [xy, yz, zx];
                let expected = exhaustive_triangle_supports(masks);

                let mut builder = ConstraintModelBuilder::new();
                let x = builder.add_variable(binary).unwrap();
                let y = builder.add_variable(binary).unwrap();
                let z = builder.add_variable(binary).unwrap();
                builder
                    .add_binary_relation(x, y, relation_rows(xy))
                    .unwrap();
                builder
                    .add_binary_relation(y, z, relation_rows(yz))
                    .unwrap();
                builder
                    .add_binary_relation(z, x, relation_rows(zx))
                    .unwrap();

                match (
                    expected,
                    NativeSolverState::initial(Arc::new(builder.build())),
                ) {
                    (None, Err(NativeSolverError::Contradiction)) => {}
                    (Some(expected), Ok(state)) => {
                        assert_eq!(
                            [
                                state.domain(x).unwrap(),
                                state.domain(y).unwrap(),
                                state.domain(z).unwrap(),
                            ],
                            expected,
                            "triangle relation masks {masks:?}"
                        );
                    }
                    (None, Ok(state)) => panic!(
                        "solver accepted unsatisfiable triangle {masks:?}: {:?}",
                        state.semantic_snapshot()
                    ),
                    (Some(expected), Err(error)) => panic!(
                        "solver rejected satisfiable triangle {masks:?} with {expected:?}: {error}"
                    ),
                    (None, Err(error)) => panic!(
                        "solver returned the wrong error for unsatisfiable triangle {masks:?}: {error}"
                    ),
                }
            }
        }
    }
}
