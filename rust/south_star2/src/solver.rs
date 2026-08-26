//! Solver-neutral state transition boundary.
//!
//! The contract contains only operations already exercised by independent
//! backends. Factor lifecycle and canonical state identity are added when the
//! writer has concrete consumers for them.

use std::error::Error;
use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::{FactorId, VariableId};
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

    /// Return `Contradiction` exactly when the prepared model's initially active
    /// factors have no satisfying assignment. A consistent state represents
    /// exactly all assignments satisfying those factors, and every exposed
    /// domain is their exact projection. Latent factors have no semantic effect
    /// until activated by `transitioned()`.
    fn initial(model: Arc<ConstraintModel>) -> Result<Consistency<Self>, Self::Failure>;

    /// Apply every supplied restriction to the same model and variable set.
    /// Return `Contradiction` exactly when no assignment represented by the
    /// source state satisfies every restriction.
    /// A consistent result must represent exactly the source assignments that
    /// satisfy all restrictions; every returned domain is nonempty, refines
    /// both the source and prepared domain, and is contained in its supplied
    /// restriction when that variable was restricted.
    fn restricted(
        &self,
        restrictions: &[(VariableId, Domain)],
    ) -> Result<Consistency<Self>, Self::Failure>;

    /// Apply restrictions and monotonically activate prepared latent factors
    /// in one atomic transition. A consistent result represents exactly the
    /// source assignments satisfying every restriction, every previously
    /// active factor, and every newly activated factor. Return `Contradiction`
    /// exactly when no such assignment exists. Repeated activation is a no-op;
    /// an unknown factor is a backend failure.
    fn transitioned(
        &self,
        restrictions: &[(VariableId, Domain)],
        activate: &[FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        assert!(
            activate.is_empty(),
            "a solver backend must implement prepared factor activation before accepting latent factors"
        );
        self.restricted(restrictions)
    }

    /// Return the exact projected domain of a known variable in this state, or
    /// `None` when the variable does not belong to the prepared model.
    fn domain(&self, variable: VariableId) -> Option<Domain>;

    /// Return whether a known prepared factor is active in this state, or
    /// `None` when the factor does not belong to the prepared model.
    fn factor_is_active(&self, _factor: FactorId) -> Option<bool> {
        None
    }
}
