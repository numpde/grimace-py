//! Native implementation of the solver-neutral constraint contract.

use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::VariableId;
use crate::model::ConstraintModel;
use crate::native::{NativeSolverError, NativeSolverState};
use crate::solver::ConstraintSolver;

impl ConstraintSolver for NativeSolverState {
    type Error = NativeSolverError;

    fn initial(model: Arc<ConstraintModel>) -> Result<Self, Self::Error> {
        NativeSolverState::initial(model)
    }

    fn restricted(
        &self,
        restrictions: &[(VariableId, Domain)],
    ) -> Result<Self, Self::Error> {
        NativeSolverState::with_restrictions(self, restrictions.iter().copied())
    }

    fn domain(&self, variable: VariableId) -> Option<Domain> {
        NativeSolverState::domain(self, variable)
    }
}
