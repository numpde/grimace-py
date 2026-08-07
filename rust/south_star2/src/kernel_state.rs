//! Atomic composition of traversal and constraint successors.
//!
//! This is deliberately not a writer-event API. A caller derives one concrete
//! traversal mutation and its domain restrictions, then asks the kernel to
//! publish both successor parts together.

use crate::domain::Domain;
use crate::ids::VariableId;
use crate::solver::ConstraintSolver;
use crate::traversal::TraversalState;

#[derive(Clone, Debug)]
pub(crate) struct KernelState<S> {
    traversal: TraversalState,
    constraints: S,
}

impl<S> KernelState<S> {
    pub(crate) fn from_parts(traversal: TraversalState, constraints: S) -> Self {
        Self {
            traversal,
            constraints,
        }
    }

    pub(crate) fn traversal(&self) -> &TraversalState {
        &self.traversal
    }

    pub(crate) fn constraints(&self) -> &S {
        &self.constraints
    }
}

impl<S: ConstraintSolver> KernelState<S> {
    /// Return a successor whose traversal and constraint changes are atomic.
    ///
    /// Constraint solving happens first. If the restriction batch is
    /// contradictory, the traversal mutation is not executed. On success the
    /// source state remains unchanged and the mutation's local effect is
    /// returned with the successor.
    pub(crate) fn transitioned<R>(
        &self,
        restrictions: &[(VariableId, Domain)],
        mutate_traversal: impl FnOnce(&mut TraversalState) -> R,
    ) -> Result<(Self, R), S::Error> {
        let constraints = self.constraints.restricted(restrictions)?;
        let mut traversal = self.traversal.clone();
        let effect = mutate_traversal(&mut traversal);
        Ok((
            Self {
                traversal,
                constraints,
            },
            effect,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::Arc;

    use super::*;
    use crate::model::ConstraintModelBuilder;
    use crate::native::NativeSolverState;
    use crate::prepared::{AdjacentBond, PreparedGraph, PreparedGraphBuilder};

    fn incident(graph: &PreparedGraph, atom: crate::AtomId, bond: crate::BondId) -> AdjacentBond {
        graph
            .neighbors(atom)
            .expect("fixture atom must exist")
            .iter()
            .copied()
            .find(|candidate| candidate.bond() == bond)
            .expect("fixture bond must be incident to the atom")
    }

    #[test]
    fn successful_transition_publishes_both_successor_parts() {
        let mut graph_builder = PreparedGraphBuilder::new();
        let first = graph_builder.add_atom().unwrap();
        let second = graph_builder.add_atom().unwrap();
        let bond = graph_builder.add_bond(first, second).unwrap();
        let graph = graph_builder.build();

        let mut model_builder = ConstraintModelBuilder::new();
        let variable = model_builder
            .add_variable(Domain::from_indices([0, 1]).unwrap())
            .unwrap();
        let solver = NativeSolverState::initial(Arc::new(model_builder.build())).unwrap();
        let mut traversal = TraversalState::new(&graph);
        traversal.begin_component(first);
        let source = KernelState::from_parts(traversal, solver);

        let (successor, active) = source
            .transitioned(
                &[(variable, Domain::singleton(1).unwrap())],
                |traversal| {
                    traversal.enter_inline_child(&graph, incident(&graph, first, bond));
                    traversal.active_atom()
                },
            )
            .unwrap();

        assert_eq!(source.traversal().active_atom(), Some(first));
        assert_eq!(
            source.constraints().domain(variable),
            Some(Domain::from_indices([0, 1]).unwrap())
        );
        assert_eq!(active, Some(second));
        assert_eq!(successor.traversal().active_atom(), Some(second));
        assert_eq!(
            successor.constraints().domain(variable),
            Some(Domain::singleton(1).unwrap())
        );
    }

    #[test]
    fn contradiction_does_not_execute_the_traversal_mutation() {
        let mut graph_builder = PreparedGraphBuilder::new();
        let atom = graph_builder.add_atom().unwrap();
        let graph = graph_builder.build();

        let mut model_builder = ConstraintModelBuilder::new();
        let variable = model_builder
            .add_variable(Domain::from_indices([0, 1]).unwrap())
            .unwrap();
        let solver = NativeSolverState::initial(Arc::new(model_builder.build())).unwrap();
        let traversal = TraversalState::new(&graph);
        let source = KernelState::from_parts(traversal, solver);
        let mutation_ran = Cell::new(false);

        let result = source.transitioned(&[(variable, Domain::empty())], |traversal| {
            mutation_ran.set(true);
            traversal.begin_component(atom);
        });

        assert!(result.is_err());
        assert!(!mutation_ran.get());
        assert_eq!(source.traversal().active_atom(), None);
        assert_eq!(
            source.constraints().domain(variable),
            Some(Domain::from_indices([0, 1]).unwrap())
        );
    }
}
