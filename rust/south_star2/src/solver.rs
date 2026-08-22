//! Solver-neutral state transition boundary.
//!
//! The contract contains only operations already exercised by independent
//! backends. Factor lifecycle and canonical state identity are added when the
//! writer has concrete consumers for them.

use std::error::Error;
use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::VariableId;
use crate::model::ConstraintModel;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Consistency<S> {
    Consistent(S),
    Contradiction,
}

impl<S> Consistency<S> {
    pub(crate) fn map<T>(self, map: impl FnOnce(S) -> T) -> Consistency<T> {
        match self {
            Self::Consistent(value) => Consistency::Consistent(map(value)),
            Self::Contradiction => Consistency::Contradiction,
        }
    }

    #[cfg(test)]
    pub(crate) fn unwrap_consistent(self) -> S {
        match self {
            Self::Consistent(value) => value,
            Self::Contradiction => panic!("expected a consistent constraint state"),
        }
    }
}

pub(crate) trait ConstraintSolver: Clone + Sized {
    type Failure: Error;

    /// Build a solver state containing every prepared variable with a nonempty
    /// domain that refines its model domain.
    fn initial(model: Arc<ConstraintModel>) -> Result<Consistency<Self>, Self::Failure>;

    /// Apply every supplied restriction to the same model and variable set.
    /// A consistent result must represent exactly the source assignments that
    /// satisfy all restrictions; every returned domain is nonempty, refines
    /// both the source and prepared domain, and is contained in its supplied
    /// restriction when that variable was restricted.
    fn restricted(
        &self,
        restrictions: &[(VariableId, Domain)],
    ) -> Result<Consistency<Self>, Self::Failure>;

    fn domain(&self, variable: VariableId) -> Option<Domain>;
}
