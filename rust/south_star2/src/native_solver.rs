//! Native implementation of the solver-neutral constraint contract.

use std::fmt;
use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::VariableId;
use crate::model::ConstraintModel;
use crate::native::{NativeSolverError, NativeSolverState};
use crate::solver::{Consistency, ConstraintSolver};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum NativeSolverFailure {
    UnknownVariable(VariableId),
}

impl fmt::Display for NativeSolverFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownVariable(variable) => {
                write!(formatter, "unknown constraint variable {variable:?}")
            }
        }
    }
}

impl std::error::Error for NativeSolverFailure {}

fn classify(
    result: Result<NativeSolverState, NativeSolverError>,
) -> Result<Consistency<NativeSolverState>, NativeSolverFailure> {
    match result {
        Ok(state) => Ok(Consistency::Consistent(state)),
        Err(NativeSolverError::Contradiction) => Ok(Consistency::Contradiction),
        Err(NativeSolverError::UnknownVariable(variable)) => {
            Err(NativeSolverFailure::UnknownVariable(variable))
        }
    }
}

impl ConstraintSolver for NativeSolverState {
    type Failure = NativeSolverFailure;

    fn initial(model: Arc<ConstraintModel>) -> Result<Consistency<Self>, Self::Failure> {
        classify(NativeSolverState::initial(model))
    }

    fn restricted(
        &self,
        restrictions: &[(VariableId, Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        classify(NativeSolverState::with_restrictions(
            self,
            restrictions.iter().copied(),
        ))
    }

    fn domain(&self, variable: VariableId) -> Option<Domain> {
        NativeSolverState::domain(self, variable)
    }
}
