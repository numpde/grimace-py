//! Exhaustive checks for the structural writer frontier.
//!
//! This oracle knows only graph topology plus the public-within-the-crate
//! structural frontier and transitions. It does not inspect solver domains or
//! traversal internals. Complete frontier-driven walks must realize exactly the
//! spanning-forest bond-role assignments admitted by the prepared graph.

use std::collections::BTreeSet;

use crate::ids::BondId;
use crate::native::{NativeSolverError, NativeSolverState};
use crate::prepared::{PreparedGraph, PreparedGraphBuilder, PreparedMolecule};
use crate::writer_state::WriterState;

struct PendingWalk {
    state: WriterState<NativeSolverState>,
    traversal_mask: u64,
}

fn bond_bit(bond: BondId) -> u64 {
    assert!(bond.index() < u64::BITS as usize);
    1_u64 << bond.index()
}

fn frontier_reachable_traversal_masks(prepared: &PreparedMolecule) -> BTreeSet<u64> {
    assert!(prepared.graph().bond_count() < u64::BITS as usize);

    let initial = WriterState::<NativeSolverState>::initial(prepared).unwrap();
    let mut pending = vec![PendingWalk {
        state: initial,
        traversal_mask: 0,
    }];
    let mut complete_masks = BTreeSet::new();
    let mut explored = 0_usize;

    while let Some(current) = pending.pop() {
        explored += 1;
        assert!(
            explored <= 1_000_000,
            "tiny structural oracle exceeded its exploration bound"
        );

        let frontier = current.state.structural_frontier();
        let mut successor_count = 0_usize;

        for &root in frontier.component_roots() {
            successor_count += 1;
            pending.push(PendingWalk {
                state: current.state.begin_component(root),
                traversal_mask: current.traversal_mask,
            });
        }

        for &incident in frontier.ring_openings() {
            successor_count += 1;
            let (state, _label_slot) = current
                .state
                .open_ring_endpoint(incident)
                .expect("an advertised ring opening must have a valid successor");
            pending.push(PendingWalk {
                state,
                traversal_mask: current.traversal_mask,
            });
        }

        for &incident in frontier.ring_closures() {
            successor_count += 1;
            let (state, _label_slot) = current.state.close_ring_endpoint(incident);
            pending.push(PendingWalk {
                state,
                traversal_mask: current.traversal_mask,
            });
        }

        if frontier.may_finish_ring_choices() {
            match current.state.finish_ring_choices() {
                Ok(state) => {
                    successor_count += 1;
                    pending.push(PendingWalk {
                        state,
                        traversal_mask: current.traversal_mask,
                    });
                }
                Err(NativeSolverError::Contradiction) => {}
                Err(NativeSolverError::UnknownVariable(variable)) => {
                    panic!("ring-choice commitment referenced unknown variable {variable:?}")
                }
            }
        }

        for &incident in frontier.branch_children() {
            successor_count += 1;
            pending.push(PendingWalk {
                state: current.state.enter_branch_child(incident),
                traversal_mask: current.traversal_mask | bond_bit(incident.bond()),
            });
        }

        for &incident in frontier.inline_children() {
            successor_count += 1;
            pending.push(PendingWalk {
                state: current.state.enter_inline_child(incident),
                traversal_mask: current.traversal_mask | bond_bit(incident.bond()),
            });
        }

        if frontier.can_complete_path() {
            successor_count += 1;
            pending.push(PendingWalk {
                state: current.state.complete_path(),
                traversal_mask: current.traversal_mask,
            });
        }

        if successor_count == 0 {
            assert!(
                current.state.graph_is_complete(),
                "the structural frontier must not dead-end before graph completion"
            );
            assert_eq!(current.state.active_atom(), None);
            complete_masks.insert(current.traversal_mask);
        }
    }

    complete_masks
}

fn graph_component_count(graph: &PreparedGraph) -> usize {
    let mut visited = vec![false; graph.atom_count()];
    let mut components = 0_usize;

    for root in graph.atom_ids() {
        if visited[root.index()] {
            continue;
        }
        components += 1;
        visited[root.index()] = true;
        let mut pending = vec![root];

        while let Some(atom) = pending.pop() {
            for incident in graph
                .neighbors(atom)
                .expect("prepared atom must have an adjacency row")
            {
                let neighbour = incident.atom();
                if !visited[neighbour.index()] {
                    visited[neighbour.index()] = true;
                    pending.push(neighbour);
                }
            }
        }
    }

    components
}

fn selected_edges_preserve_graph_components(graph: &PreparedGraph, mask: u64) -> bool {
    if graph.atom_count() == 0 {
        return true;
    }

    let mut original_component = vec![usize::MAX; graph.atom_count()];
    let mut component = 0_usize;
    for root in graph.atom_ids() {
        if original_component[root.index()] != usize::MAX {
            continue;
        }
        original_component[root.index()] = component;
        let mut pending = vec![root];
        while let Some(atom) = pending.pop() {
            for incident in graph
                .neighbors(atom)
                .expect("prepared atom must have an adjacency row")
            {
                let neighbour = incident.atom();
                if original_component[neighbour.index()] == usize::MAX {
                    original_component[neighbour.index()] = component;
                    pending.push(neighbour);
                }
            }
        }
        component += 1;
    }

    let mut selected_component = vec![usize::MAX; graph.atom_count()];
    let mut selected = 0_usize;
    for root in graph.atom_ids() {
        if selected_component[root.index()] != usize::MAX {
            continue;
        }
        selected_component[root.index()] = selected;
        let mut pending = vec![root];
        while let Some(atom) = pending.pop() {
            for incident in graph
                .neighbors(atom)
                .expect("prepared atom must have an adjacency row")
            {
                if mask & bond_bit(incident.bond()) == 0 {
                    continue;
                }
                let neighbour = incident.atom();
                if selected_component[neighbour.index()] == usize::MAX {
                    selected_component[neighbour.index()] = selected;
                    pending.push(neighbour);
                }
            }
        }
        selected += 1;
    }

    for left in graph.atom_ids() {
        for right in graph.atom_ids() {
            if original_component[left.index()] == original_component[right.index()]
                && selected_component[left.index()] != selected_component[right.index()]
            {
                return false;
            }
        }
    }
    true
}

fn exhaustive_spanning_forest_masks(graph: &PreparedGraph) -> BTreeSet<u64> {
    assert!(graph.bond_count() < u64::BITS as usize);
    let component_count = graph_component_count(graph);
    let required_edges = graph.atom_count().saturating_sub(component_count);
    let mut masks = BTreeSet::new();

    for mask in 0_u64..(1_u64 << graph.bond_count()) {
        if mask.count_ones() as usize != required_edges {
            continue;
        }
        if selected_edges_preserve_graph_components(graph, mask) {
            masks.insert(mask);
        }
    }

    masks
}

fn fixture(atom_count: usize, edges: &[(usize, usize)]) -> PreparedMolecule {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..atom_count)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    for &(a, b) in edges {
        graph.add_bond(atoms[a], atoms[b]).unwrap();
    }
    PreparedMolecule::new(graph.build())
}

#[test]
fn frontier_driven_walks_realize_exactly_the_spanning_forests() {
    let fixtures: [(&str, PreparedMolecule); 6] = [
        ("single atom", fixture(1, &[])),
        ("single bridge", fixture(2, &[(0, 1)])),
        ("triangle", fixture(3, &[(0, 1), (1, 2), (2, 0)])),
        ("square", fixture(4, &[(0, 1), (1, 2), (2, 3), (3, 0)])),
        (
            "square with diagonal",
            fixture(4, &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),
        ),
        (
            "two components",
            fixture(5, &[(0, 1), (1, 2), (2, 0), (3, 4)]),
        ),
    ];

    for (name, prepared) in fixtures {
        assert_eq!(
            frontier_reachable_traversal_masks(&prepared),
            exhaustive_spanning_forest_masks(prepared.graph()),
            "structural frontier mismatch for {name}"
        );
    }
}
