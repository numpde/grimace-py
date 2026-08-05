//! Solver-neutral state transition boundary.
//!
//! The contract contains only operations already exercised by the native
//! backend. Factor lifecycle is added when writer events require it.

use std::error::Error;
use std::hash::Hash;
use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::VariableId;
use crate::model::ConstraintModel;
use crate::native::{ConstraintStateSnapshot, NativeSolverError, NativeSolverState};

pub(crate) trait ConstraintSolver: Clone + Sized {
    type Error: Error;
    type Snapshot: Clone + Eq + Hash;

    fn initial(model: Arc<ConstraintModel>) -> Result<Self, Self::Error>;

    fn restricted(
        &self,
        restrictions: &[(VariableId, Domain)],
    ) -> Result<Self, Self::Error>;

    fn domain(&self, variable: VariableId) -> Option<Domain>;

    fn semantic_snapshot(&self) -> Self::Snapshot;
}

impl ConstraintSolver for NativeSolverState {
    type Error = NativeSolverError;
    type Snapshot = ConstraintStateSnapshot;

    fn initial(model: Arc<ConstraintModel>) -> Result<Self, Self::Error> {
        NativeSolverState::initial(model)
    }

    fn restricted(
        &self,
        restrictions: &[(VariableId, Domain)],
    ) -> Result<Self, Self::Error> {
        NativeSolverState::with_restrictions(self, restrictions.iter().copied())
            .map(|(state, _summary)| state)
    }

    fn domain(&self, variable: VariableId) -> Option<Domain> {
        NativeSolverState::domain(self, variable)
    }

    fn semantic_snapshot(&self) -> Self::Snapshot {
        NativeSolverState::semantic_snapshot(self)
    }
}
