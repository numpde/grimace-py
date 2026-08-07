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

pub(crate) trait ConstraintSolver: Clone + Sized {
    type Error: Error;

    fn initial(model: Arc<ConstraintModel>) -> Result<Self, Self::Error>;

    fn restricted(&self, restrictions: &[(VariableId, Domain)]) -> Result<Self, Self::Error>;

    fn domain(&self, variable: VariableId) -> Option<Domain>;
}
