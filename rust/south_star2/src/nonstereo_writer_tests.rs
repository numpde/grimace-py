use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::*;
use crate::native::NativeSolverState;
use crate::native_oracle::ExhaustiveSolverState;
use crate::native_solver::NativeSolverFailure;
use crate::prepared::PreparedGraphBuilder;
use crate::traversal::ObservedBondProgress;

type State = NonStereoWriterState<NativeSolverState>;

#[derive(Clone, Debug, PartialEq, Eq)]
enum InjectedSolverFailure {
    Native(NativeSolverFailure),
    Restriction,
}

impl fmt::Display for InjectedSolverFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native(failure) => failure.fmt(formatter),
            Self::Restriction => formatter.write_str("injected restriction failure"),
        }
    }
}

impl Error for InjectedSolverFailure {}

#[derive(Clone, Debug)]
struct FailingRestrictionSolver(NativeSolverState);

impl ConstraintSolver for FailingRestrictionSolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .map(Self))
    }

    fn restricted(
        &self,
        _restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        Err(InjectedSolverFailure::Restriction)
    }

    fn transitioned(
        &self,
        _restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        _activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        Err(InjectedSolverFailure::Restriction)
    }

    fn admit_live_ring_cut(&self, factor: crate::ids::FactorId,
        context: &crate::live_ring_cut::LiveRingCutContext,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::admit_live_ring_cut(
            &self.0, factor, context).map_err(InjectedSolverFailure::Native)?.map(Self))
    }

    fn domain_bound(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain_bound(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

#[derive(Clone, Debug)]
struct FailSecondVariableSolver(NativeSolverState);

impl ConstraintSolver for FailSecondVariableSolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .map(Self))
    }

    fn restricted(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if restrictions
            .iter()
            .any(|(variable, _)| *variable == crate::ids::VariableId::new(1))
        {
            return Err(InjectedSolverFailure::Restriction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&self.0, restrictions)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn transitioned(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if restrictions
            .iter()
            .any(|(variable, _)| *variable == crate::ids::VariableId::new(1))
        {
            return Err(InjectedSolverFailure::Restriction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, restrictions, activate)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn admit_live_ring_cut(&self, factor: crate::ids::FactorId,
        context: &crate::live_ring_cut::LiveRingCutContext,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::admit_live_ring_cut(
            &self.0, factor, context).map_err(InjectedSolverFailure::Native)?.map(Self))
    }

    fn domain_bound(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain_bound(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

#[derive(Clone, Debug)]
struct FailFirstRingAlternativeSolver(NativeSolverState);

impl ConstraintSolver for FailFirstRingAlternativeSolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .map(Self))
    }

    fn restricted(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if matches!(
            restrictions,
            [(variable, domain)]
                if *variable == crate::ids::VariableId::new(0)
                    && *domain == BondRepresentation::Ring01.singleton_domain()
        ) {
            return Err(InjectedSolverFailure::Restriction);
        }
        assert!(
            !matches!(
                restrictions,
                [(variable, domain)]
                    if *variable == crate::ids::VariableId::new(0)
                        && *domain
                            == BondRepresentation::Ring10
                                .singleton_domain()
                                .union(BondRepresentation::Ring11.singleton_domain())
            ),
            "choice generation must stop after the first backend failure"
        );
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&self.0, restrictions)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn transitioned(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if matches!(
            restrictions,
            [(variable, domain)]
                if *variable == crate::ids::VariableId::new(0)
                    && *domain == BondRepresentation::Ring01.singleton_domain()
        ) {
            return Err(InjectedSolverFailure::Restriction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, restrictions, activate)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn admit_live_ring_cut(&self, factor: crate::ids::FactorId,
        context: &crate::live_ring_cut::LiveRingCutContext,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::admit_live_ring_cut(
            &self.0, factor, context).map_err(InjectedSolverFailure::Native)?.map(Self))
    }

    fn domain_bound(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain_bound(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

#[derive(Clone, Debug)]
struct RejectFirstVariableSolver(NativeSolverState);

impl ConstraintSolver for RejectFirstVariableSolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .map(Self))
    }

    fn restricted(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if restrictions
            .iter()
            .any(|(variable, _)| *variable == crate::ids::VariableId::new(0))
        {
            return Ok(Consistency::Contradiction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&self.0, restrictions)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn transitioned(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if restrictions
            .iter()
            .any(|(variable, _)| *variable == crate::ids::VariableId::new(0))
        {
            return Ok(Consistency::Contradiction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, restrictions, activate)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn admit_live_ring_cut(&self, factor: crate::ids::FactorId,
        context: &crate::live_ring_cut::LiveRingCutContext,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::admit_live_ring_cut(
            &self.0, factor, context).map_err(InjectedSolverFailure::Native)?.map(Self))
    }

    fn domain_bound(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain_bound(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

#[derive(Clone, Debug)]
struct RejectEveryRestrictionSolver(NativeSolverState);

impl ConstraintSolver for RejectEveryRestrictionSolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .map(Self))
    }

    fn restricted(
        &self,
        _restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(Consistency::Contradiction)
    }

    fn transitioned(
        &self,
        _restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        _activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(Consistency::Contradiction)
    }

    fn admit_exact(&self) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::admit_exact(&self.0)
            .map_err(InjectedSolverFailure::Native)?.map(Self))
    }

    fn admit_live_ring_cut(&self, factor: crate::ids::FactorId,
        context: &crate::live_ring_cut::LiveRingCutContext,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::admit_live_ring_cut(
            &self.0, factor, context).map_err(InjectedSolverFailure::Native)?.map(Self))
    }

    fn domain_bound(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain_bound(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

// These deliberate contract violations exercise writer failure classification,
// not solver-backend conformance.
#[derive(Clone, Debug)]
struct NonconformingRejectTetrahedralParitySolver(NativeSolverState);

impl ConstraintSolver for NonconformingRejectTetrahedralParitySolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .map(Self))
    }

    fn restricted(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if restrictions
            .iter()
            .any(|(variable, _)| self.0.domain_bound(*variable) == Some(full_order_domain()))
        {
            return Ok(Consistency::Contradiction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&self.0, restrictions)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn transitioned(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if restrictions
            .iter()
            .any(|(variable, _)| self.0.domain_bound(*variable) == Some(full_order_domain()))
        {
            return Ok(Consistency::Contradiction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, restrictions, activate)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn admit_live_ring_cut(&self, factor: crate::ids::FactorId,
        context: &crate::live_ring_cut::LiveRingCutContext,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::admit_live_ring_cut(
            &self.0, factor, context).map_err(InjectedSolverFailure::Native)?.map(Self))
    }

    fn domain_bound(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain_bound(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

#[derive(Clone, Debug)]
struct NonconformingRejectEvenTetrahedralParitySolver(NativeSolverState);

impl ConstraintSolver for NonconformingRejectEvenTetrahedralParitySolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .map(Self))
    }

    fn restricted(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if restrictions.iter().any(|(variable, domain)| {
            self.0.domain_bound(*variable) == Some(full_order_domain())
                && domain.is_subset_of(parity_domain(TetrahedralParity::Even))
        }) {
            return Ok(Consistency::Contradiction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&self.0, restrictions)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn transitioned(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if restrictions.iter().any(|(variable, domain)| {
            self.0.domain_bound(*variable) == Some(full_order_domain())
                && domain.is_subset_of(parity_domain(TetrahedralParity::Even))
        }) {
            return Ok(Consistency::Contradiction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, restrictions, activate)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn admit_live_ring_cut(&self, factor: crate::ids::FactorId,
        context: &crate::live_ring_cut::LiveRingCutContext,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::admit_live_ring_cut(
            &self.0, factor, context).map_err(InjectedSolverFailure::Native)?.map(Self))
    }

    fn domain_bound(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain_bound(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

#[derive(Clone, Debug)]
struct FailTetrahedralRestrictionSolver(NativeSolverState);

static TETRAHEDRAL_BACKEND_ATTEMPTS: AtomicUsize = AtomicUsize::new(0);

impl ConstraintSolver for FailTetrahedralRestrictionSolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .map(Self))
    }

    fn restricted(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if restrictions
            .iter()
            .any(|(variable, _)| self.0.domain_bound(*variable) == Some(full_order_domain()))
        {
            assert_eq!(
                TETRAHEDRAL_BACKEND_ATTEMPTS.fetch_add(1, Ordering::Relaxed),
                0,
                "atom-token generation must stop at the first backend failure"
            );
            return Err(InjectedSolverFailure::Restriction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&self.0, restrictions)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn transitioned(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if restrictions
            .iter()
            .any(|(variable, _)| self.0.domain_bound(*variable) == Some(full_order_domain()))
        {
            assert_eq!(
                TETRAHEDRAL_BACKEND_ATTEMPTS.fetch_add(1, Ordering::Relaxed),
                0,
                "atom-token generation must stop at the first backend failure"
            );
            return Err(InjectedSolverFailure::Restriction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, restrictions, activate)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn admit_live_ring_cut(&self, factor: crate::ids::FactorId,
        context: &crate::live_ring_cut::LiveRingCutContext,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::admit_live_ring_cut(
            &self.0, factor, context).map_err(InjectedSolverFailure::Native)?.map(Self))
    }

    fn domain_bound(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain_bound(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

#[derive(Clone, Debug)]
struct NonconformingNonProjectingTetrahedralSolver(NativeSolverState);

impl ConstraintSolver for NonconformingNonProjectingTetrahedralSolver {
    type Failure = NativeSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)?.map(Self))
    }

    fn restricted(
        &self,
        _restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(Consistency::Consistent(self.clone()))
    }

    fn transitioned(
        &self,
        _restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, &[], activate)?
                .map(Self),
        )
    }

    fn admit_live_ring_cut(&self, factor: crate::ids::FactorId,
        context: &crate::live_ring_cut::LiveRingCutContext,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::admit_live_ring_cut(
            &self.0, factor, context)?.map(Self))
    }

    fn domain_bound(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain_bound(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

#[derive(Clone, Debug)]
struct WriterPolicyContradictionSolver(NativeSolverState);

impl ConstraintSolver for WriterPolicyContradictionSolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .map(Self))
    }

    fn restricted(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        let role_partition = BondRepresentation::role_partition();

        let requested_first_ring = matches!(
            restrictions,
            [(variable, domain)]
                if *variable == crate::ids::VariableId::new(0)
                    && !domain.is_empty()
                    && domain.is_subset_of(role_partition.ring_values())
        );
        let effective = if requested_first_ring {
            let first_ring = restrictions[0];
            vec![
                first_ring,
                (
                    crate::ids::VariableId::new(1),
                    role_partition.traversal_values(),
                ),
                (
                    crate::ids::VariableId::new(2),
                    role_partition.traversal_values(),
                ),
                (
                    crate::ids::VariableId::new(3),
                    role_partition.traversal_values(),
                ),
                (crate::ids::VariableId::new(4), role_partition.ring_values()),
            ]
        } else {
            restrictions.to_vec()
        };
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&self.0, &effective)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn transitioned(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        let role_partition = BondRepresentation::role_partition();
        let requested_first_ring = matches!(
            restrictions,
            [(variable, domain)]
                if *variable == crate::ids::VariableId::new(0)
                    && !domain.is_empty()
                    && domain.is_subset_of(role_partition.ring_values())
        );
        let effective = if requested_first_ring {
            let first_ring = restrictions[0];
            vec![
                first_ring,
                (
                    crate::ids::VariableId::new(1),
                    role_partition.traversal_values(),
                ),
                (
                    crate::ids::VariableId::new(2),
                    role_partition.traversal_values(),
                ),
                (
                    crate::ids::VariableId::new(3),
                    role_partition.traversal_values(),
                ),
                (crate::ids::VariableId::new(4), role_partition.ring_values()),
            ]
        } else {
            restrictions.to_vec()
        };
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, &effective, activate)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn admit_live_ring_cut(&self, factor: crate::ids::FactorId,
        context: &crate::live_ring_cut::LiveRingCutContext,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::admit_live_ring_cut(
            &self.0, factor, context).map_err(InjectedSolverFailure::Native)?.map(Self))
    }

    fn domain_bound(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain_bound(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

#[derive(Clone, Debug)]
struct PendingContradictionSolver(NativeSolverState);

impl ConstraintSolver for PendingContradictionSolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        use crate::model::BondRole;

        let native = <NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .unwrap_consistent();
        let restrictions = [
            (
                crate::ids::VariableId::new(0),
                BondRole::Traversal.singleton_domain(),
            ),
            (
                crate::ids::VariableId::new(1),
                BondRole::Traversal.singleton_domain(),
            ),
            (
                crate::ids::VariableId::new(2),
                BondRole::Traversal.singleton_domain(),
            ),
            (
                crate::ids::VariableId::new(3),
                BondRole::Ring.singleton_domain(),
            ),
        ];
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&native, &restrictions)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn restricted(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(
            <NativeSolverState as ConstraintSolver>::restricted(&self.0, restrictions)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn transitioned(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, restrictions, activate)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn admit_live_ring_cut(&self, factor: crate::ids::FactorId,
        context: &crate::live_ring_cut::LiveRingCutContext,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::admit_live_ring_cut(
            &self.0, factor, context).map_err(InjectedSolverFailure::Native)?.map(Self))
    }

    fn domain_bound(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain_bound(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

#[derive(Clone, Debug)]
struct FailDirectionalMarkSolver(NativeSolverState);

impl ConstraintSolver for FailDirectionalMarkSolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .map(Self))
    }

    fn restricted(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        self.transitioned(restrictions, &[])
    }

    fn transitioned(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        if restrictions.iter().any(|(_variable, domain)| {
            *domain == CarrierMark::BackslashAtFixedA.singleton_domain()
        }) {
            return Err(InjectedSolverFailure::Restriction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, restrictions, activate)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn admit_live_ring_cut(&self, factor: crate::ids::FactorId,
        context: &crate::live_ring_cut::LiveRingCutContext,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::admit_live_ring_cut(
            &self.0, factor, context).map_err(InjectedSolverFailure::Native)?.map(Self))
    }

    fn domain_bound(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain_bound(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

static REJECT_DIRECTIONAL_MARK_VARIABLE: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Nonconforming fault injector used only to verify candidate-local mark rejection.
#[derive(Clone, Debug)]
struct RejectPlainDirectionalMarkSolver(NativeSolverState);

impl ConstraintSolver for RejectPlainDirectionalMarkSolver {
    type Failure = InjectedSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)
            .map_err(InjectedSolverFailure::Native)?
            .map(Self))
    }

    fn restricted(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        self.transitioned(restrictions, &[])
    }

    fn transitioned(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        let rejected = REJECT_DIRECTIONAL_MARK_VARIABLE.load(Ordering::Relaxed);
        if restrictions.iter().any(|(variable, domain)| {
            variable.index() == rejected && *domain == CarrierMark::Plain.singleton_domain()
        }) {
            return Ok(Consistency::Contradiction);
        }
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, restrictions, activate)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self),
        )
    }

    fn admit_live_ring_cut(&self, factor: crate::ids::FactorId,
        context: &crate::live_ring_cut::LiveRingCutContext,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::admit_live_ring_cut(
            &self.0, factor, context).map_err(InjectedSolverFailure::Native)?.map(Self))
    }

    fn domain_bound(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain_bound(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

static DIRECTIONAL_RESTRICTION_CALLS: AtomicUsize = AtomicUsize::new(0);
static DIRECTIONAL_RESTRICTION_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Clone, Debug)]
struct CountDirectionalRestrictionsSolver(NativeSolverState);

impl ConstraintSolver for CountDirectionalRestrictionsSolver {
    type Failure = NativeSolverFailure;

    fn initial(
        model: Arc<crate::model::ConstraintModel>,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::initial(model)?.map(Self))
    }

    fn restricted(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        self.transitioned(restrictions, &[])
    }

    fn transitioned(
        &self,
        restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        activate: &[crate::ids::FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        DIRECTIONAL_RESTRICTION_CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(
            <NativeSolverState as ConstraintSolver>::transitioned(&self.0, restrictions, activate)?
                .map(Self),
        )
    }

    fn admit_live_ring_cut(&self, factor: crate::ids::FactorId,
        context: &crate::live_ring_cut::LiveRingCutContext,
    ) -> Result<Consistency<Self>, Self::Failure> {
        Ok(<NativeSolverState as ConstraintSolver>::admit_live_ring_cut(
            &self.0, factor, context)?.map(Self))
    }

    fn domain_bound(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
        self.0.domain_bound(variable)
    }

    fn factor_is_active(&self, factor: crate::ids::FactorId) -> Option<bool> {
        self.0.factor_is_active(factor)
    }
}

fn fixture(
    atom_text: &[&str],
    edges: &[(usize, usize, NonStereoBondToken)],
) -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = atom_text
        .iter()
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let mut bonds = Vec::with_capacity(edges.len());
    let mut bond_tokens = Vec::with_capacity(edges.len());
    for &(a, b, token) in edges {
        bonds.push(graph.add_bond(atoms[a], atoms[b]).unwrap());
        bond_tokens.push(token);
    }
    let surface = PreparedNonStereo::new(
        PreparedMolecule::new(graph.build()),
        atom_text.iter().map(|text| (*text).to_owned()).collect(),
        bond_tokens,
    )
    .unwrap();
    (surface, atoms, bonds)
}

fn tetrahedral_star_fixture(leaf_count: usize) -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    tetrahedral_star_fixture_with_tokens(leaf_count, vec![NonStereoBondToken::Elided; leaf_count])
}

fn tetrahedral_star_fixture_with_tokens(
    leaf_count: usize,
    bond_tokens: Vec<NonStereoBondToken>,
) -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    assert!(matches!(leaf_count, 3 | 4));
    assert_eq!(bond_tokens.len(), leaf_count);
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..=leaf_count)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = (0..leaf_count)
        .map(|index| graph.add_bond(atoms[0], atoms[index + 1]).unwrap())
        .collect::<Vec<_>>();
    let mut reference_order = bonds
        .iter()
        .copied()
        .map(TetrahedralLigand::Bond)
        .collect::<Vec<_>>();
    if leaf_count == 3 {
        reference_order.push(TetrahedralLigand::VirtualHydrogen);
    }
    let mut atom_tokens = vec![PreparedAtomToken::Tetrahedral {
        reference_order: reference_order.try_into().unwrap(),
        text_by_parity: if leaf_count == 3 {
            ["[C@H]".to_owned(), "[C@@H]".to_owned()]
        } else {
            ["[C@]".to_owned(), "[C@@]".to_owned()]
        },
    }];
    atom_tokens.extend(
        (0..leaf_count)
            .map(|index| PreparedAtomToken::Fixed(char::from(b'A' + index as u8).to_string())),
    );
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        atom_tokens,
        bond_tokens,
    )
    .unwrap();
    (surface, atoms, bonds)
}

fn entered_tetrahedral_fixture(
    entry_token: NonStereoBondToken,
    parent_has_second_child: bool,
) -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atom_count = if parent_has_second_child { 6 } else { 5 };
    let atoms = (0..atom_count)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let mut bonds = vec![graph.add_bond(atoms[0], atoms[1]).unwrap()];
    bonds.extend((2..5).map(|child| graph.add_bond(atoms[1], atoms[child]).unwrap()));
    if parent_has_second_child {
        bonds.push(graph.add_bond(atoms[0], atoms[5]).unwrap());
    }
    let mut atom_tokens = vec![
        PreparedAtomToken::Fixed("P".to_owned()),
        PreparedAtomToken::Tetrahedral {
            reference_order: [bonds[0], bonds[1], bonds[2], bonds[3]].map(TetrahedralLigand::Bond),
            text_by_parity: ["[C@]".to_owned(), "[C@@]".to_owned()],
        },
    ];
    atom_tokens.extend(
        (2..atom_count)
            .map(|index| PreparedAtomToken::Fixed(char::from(b'A' + index as u8).to_string())),
    );
    let mut bond_tokens = vec![entry_token];
    bond_tokens.extend(vec![NonStereoBondToken::Elided; bonds.len() - 1]);
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        atom_tokens,
        bond_tokens,
    )
    .unwrap();
    (surface, atoms, bonds)
}

fn ring_coupled_tetrahedral_fixture(
    first_token: NonStereoBondToken,
) -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..4)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = vec![
        graph.add_bond(atoms[0], atoms[1]).unwrap(),
        graph.add_bond(atoms[0], atoms[2]).unwrap(),
        graph.add_bond(atoms[1], atoms[2]).unwrap(),
        graph.add_bond(atoms[0], atoms[3]).unwrap(),
    ];
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        vec![
            PreparedAtomToken::Tetrahedral {
                reference_order: [
                    TetrahedralLigand::Bond(bonds[0]),
                    TetrahedralLigand::Bond(bonds[1]),
                    TetrahedralLigand::Bond(bonds[3]),
                    TetrahedralLigand::VirtualHydrogen,
                ],
                text_by_parity: ["[C@H]".to_owned(), "[C@@H]".to_owned()],
            },
            PreparedAtomToken::Fixed("A".to_owned()),
            PreparedAtomToken::Fixed("B".to_owned()),
            PreparedAtomToken::Fixed("D".to_owned()),
        ],
        vec![
            first_token,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
        ],
    )
    .unwrap();
    (surface, atoms, bonds)
}

fn adjacent_ring_coupled_tetrahedral_fixture() -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..6)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = vec![
        graph.add_bond(atoms[0], atoms[1]).unwrap(),
        graph.add_bond(atoms[1], atoms[2]).unwrap(),
        graph.add_bond(atoms[2], atoms[3]).unwrap(),
        graph.add_bond(atoms[3], atoms[0]).unwrap(),
        graph.add_bond(atoms[0], atoms[4]).unwrap(),
        graph.add_bond(atoms[1], atoms[5]).unwrap(),
    ];
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        vec![
            PreparedAtomToken::Tetrahedral {
                reference_order: [
                    TetrahedralLigand::Bond(bonds[0]),
                    TetrahedralLigand::Bond(bonds[3]),
                    TetrahedralLigand::Bond(bonds[4]),
                    TetrahedralLigand::VirtualHydrogen,
                ],
                text_by_parity: ["[C@H]".to_owned(), "[C@@H]".to_owned()],
            },
            PreparedAtomToken::Tetrahedral {
                reference_order: [
                    TetrahedralLigand::Bond(bonds[0]),
                    TetrahedralLigand::Bond(bonds[1]),
                    TetrahedralLigand::Bond(bonds[5]),
                    TetrahedralLigand::VirtualHydrogen,
                ],
                text_by_parity: ["[C@H]".to_owned(), "[C@@H]".to_owned()],
            },
            PreparedAtomToken::Fixed("A".to_owned()),
            PreparedAtomToken::Fixed("B".to_owned()),
            PreparedAtomToken::Fixed("X".to_owned()),
            PreparedAtomToken::Fixed("Q".to_owned()),
        ],
        vec![NonStereoBondToken::Elided; bonds.len()],
    )
    .unwrap();
    (surface, atoms, bonds)
}

fn disconnected_ring_coupled_tetrahedral_fixture() -> (PreparedNonStereo, Vec<AtomId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..8)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let mut bonds = Vec::new();
    for offset in [0, 4] {
        bonds.extend([
            graph.add_bond(atoms[offset], atoms[offset + 1]).unwrap(),
            graph.add_bond(atoms[offset], atoms[offset + 2]).unwrap(),
            graph
                .add_bond(atoms[offset + 1], atoms[offset + 2])
                .unwrap(),
            graph.add_bond(atoms[offset], atoms[offset + 3]).unwrap(),
        ]);
    }
    let mut atom_tokens = Vec::new();
    for component in 0..2 {
        let bond_offset = component * 4;
        atom_tokens.extend([
            PreparedAtomToken::Tetrahedral {
                reference_order: [
                    TetrahedralLigand::Bond(bonds[bond_offset]),
                    TetrahedralLigand::Bond(bonds[bond_offset + 1]),
                    TetrahedralLigand::Bond(bonds[bond_offset + 3]),
                    TetrahedralLigand::VirtualHydrogen,
                ],
                text_by_parity: ["[C@H]".to_owned(), "[C@@H]".to_owned()],
            },
            PreparedAtomToken::Fixed(format!("A{component}")),
            PreparedAtomToken::Fixed(format!("B{component}")),
            PreparedAtomToken::Fixed(format!("D{component}")),
        ]);
    }
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        atom_tokens,
        vec![
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
        ],
    )
    .unwrap();
    (surface, atoms)
}

fn independent_local_layout_groups(
    atom_count: usize,
    endpoints: &[(usize, usize)],
    observed: &ObservedNonStereoState,
    active: usize,
) -> (Vec<BondId>, Vec<Vec<BondId>>) {
    let visited = observed
        .structural
        .traversal
        .visited_atoms
        .iter()
        .map(|atom| atom.index())
        .collect::<BTreeSet<_>>();
    let mut component = vec![None; atom_count];
    let mut next_component = 0;
    for root in 0..atom_count {
        if visited.contains(&root) || component[root].is_some() {
            continue;
        }
        let mut stack = vec![root];
        component[root] = Some(next_component);
        while let Some(atom) = stack.pop() {
            for &(a, b) in endpoints {
                let other = if a == atom {
                    Some(b)
                } else if b == atom {
                    Some(a)
                } else {
                    None
                };
                if let Some(other) = other {
                    if !visited.contains(&other) && component[other].is_none() {
                        component[other] = Some(next_component);
                        stack.push(other);
                    }
                }
            }
        }
        next_component += 1;
    }

    let mut waiting = Vec::new();
    let mut groups = BTreeMap::<usize, Vec<BondId>>::new();
    for (index, &(a, b)) in endpoints.iter().enumerate() {
        let other = if a == active {
            Some(b)
        } else if b == active {
            Some(a)
        } else {
            None
        };
        let Some(other) = other else { continue };
        let bond = BondId::new(u32::try_from(index).unwrap());
        match observed.structural.traversal.bond_progress[index] {
            ObservedBondProgress::RingOpen { first_endpoint }
                if first_endpoint.index() == other =>
            {
                waiting.push(bond);
            }
            ObservedBondProgress::Unrepresented if !visited.contains(&other) => {
                groups
                    .entry(component[other].expect("unvisited atom must own a component"))
                    .or_default()
                    .push(bond);
            }
            _ => {}
        }
    }
    waiting.sort_unstable();
    let mut groups = groups.into_values().collect::<Vec<_>>();
    for group in &mut groups {
        group.sort_unstable();
    }
    groups.sort_unstable();
    (waiting, groups)
}

fn set_partitions<T: Copy>(values: &[T]) -> Vec<Vec<Vec<T>>> {
    if values.is_empty() {
        return vec![Vec::new()];
    }
    let first = values[0];
    let mut partitions = Vec::new();
    for suffix in set_partitions(&values[1..]) {
        let mut own_group = vec![vec![first]];
        own_group.extend(suffix.clone());
        partitions.push(own_group);
        for group in 0..suffix.len() {
            let mut inserted = suffix.clone();
            inserted[group].push(first);
            partitions.push(inserted);
        }
    }
    partitions
}

fn incident(surface: &PreparedNonStereo, atom: AtomId, bond: BondId) -> AdjacentBond {
    surface
        .molecule()
        .graph()
        .neighbors(atom)
        .expect("fixture atom must exist")
        .iter()
        .copied()
        .find(|candidate| candidate.bond() == bond)
        .expect("fixture bond must be incident to the atom")
}

fn only_choice<S: ConstraintSolver>(
    state: &NonStereoWriterState<S>,
    text: &str,
) -> (String, NonStereoWriterState<S>) {
    let branch_atom_is_ready = match &state.pending {
        Some(PendingEmission::BranchTraversalEmission(incident)) => {
            let atom = incident.atom();
            let first_chars = declared_atom_witnesses(&state.surface, atom)
                .into_iter()
                .filter_map(|target| target.chars().next())
                .collect::<BTreeSet<_>>();
            ring_witness_choices(&state)
                .iter()
                .any(|choice| first_chars.contains(&choice.text().chars().next().unwrap()))
        }
        _ => false,
    };
    if state.is_spelling_atom()
        || branch_atom_is_ready
        || matches!(&state.pending,
            Some(
                PendingEmission::SegmentRootAtom(_)
                    | PendingEmission::InlineAtom(_)
                    | PendingEmission::BranchAtom(_)
            )
        )
    {
        let atom = pending_atom(state)
            .or_else(|| state.active_atom())
            .expect("pending atom spelling identifies its AtomId");
        let target = declared_atom_witnesses(&state.surface, atom)
            .into_iter()
            .find(|target| {
                if text.contains("@@") {
                    target.contains("@@")
                } else if text.contains('@') {
                    target.contains('@') && !target.contains("@@")
                } else {
                    true
                }
            })
            .expect("test atom has a selected production spelling");
        let successors = consume_atom_spelling_states(state.clone(), &target, 0);
        assert_eq!(
            successors.len(),
            1,
            "one declared atom witness must identify one test path"
        );
        let successor = successors.into_iter().next().unwrap();
        let emitted_text = if state.surface.tetrahedral_center(atom).is_some() {
            target
        } else {
            state.surface.atom_text(atom).to_owned()
        };
        return (emitted_text, successor);
    }
    let choices = ring_witness_choices(&state)
        .into_iter()
        .filter(|choice| choice.text() == text)
        .collect::<Vec<_>>();
    assert_eq!(
        choices.len(),
        1,
        "expected one production transition for {text:?}"
    );
    let choice = choices.into_iter().next().unwrap();
    (choice.text, choice.successor)
}

/// Finite, declared atom spellings used by structural writer tests.
/// This is a witness set, not an enumeration of the atom grammar.
fn declared_atom_witnesses(surface: &PreparedNonStereo, atom: AtomId) -> Vec<String> {
    let facts = surface.atom_facts(atom);
    if surface.tetrahedral_center(atom).is_some() {
        let hydrogen = if facts.non_graph_hydrogens == 1 {
            "H"
        } else {
            ""
        };
        return vec![format!("[C@{hydrogen}]"), format!("[C@@{hydrogen}]")];
    }
    if facts.element.as_str() == "*" {
        vec![format!("[*:{}]", facts.atom_class)]
    } else if facts.element.as_str() == "F" && facts.non_graph_hydrogens == 1 {
        // Isolated fluorine has both bare and bracketed declared witnesses;
        // an aromatic carrier fixture has only the bracketed spelling.
        if surface
            .molecule()
            .graph()
            .neighbors(atom)
            .expect("test atom exists")
            .is_empty()
        {
            vec!["F".to_owned(), "[FH]".to_owned()]
        } else {
            vec!["[FH]".to_owned()]
        }
    } else {
        vec![facts.element.as_str().to_owned()]
    }
}

fn pending_atom<S>(state: &NonStereoWriterState<S>) -> Option<AtomId> {
    state
        .atom_spelling
        .as_ref()
        .map(|progress| progress.atom)
        .or_else(|| match &state.pending {
            Some(PendingEmission::SegmentRootAtom(atom)) => Some(*atom),
            Some(PendingEmission::InlineAtom(edge) | PendingEmission::BranchAtom(edge)) => {
                Some(edge.atom())
            }
            Some(PendingEmission::BranchTraversalEmission(edge)) => Some(edge.atom()),
            _ => None,
        })
}

fn branch_event_atom<S>(state: &NonStereoWriterState<S>) -> Option<AtomId> {
    match &state.pending {
        Some(PendingEmission::BranchTraversalEmission(incident)) => Some(incident.atom()),
        _ => pending_atom(state),
    }
}

fn atom_was_emitted<S: ConstraintSolver>(state: &NonStereoWriterState<S>, atom: AtomId) -> bool {
    state.active_atom() == Some(atom) || state.structural.atom_is_visited(atom)
}

pub(super) fn completed_atom_witnesses<S: ConstraintSolver>(
    state: &NonStereoWriterState<S>,
    atom: AtomId,
) -> Vec<(String, NonStereoWriterState<S>)> {
    declared_atom_witnesses(&state.surface, atom)
        .into_iter()
        .flat_map(|target| {
            consume_atom_spelling_states(state.clone(), &target, 0)
                .into_iter()
                .filter(|successor| atom_was_emitted(successor, atom))
                .map(move |successor| (target.clone(), successor))
        })
        .collect()
}

fn atom_witness_completions_after_choice<S: ConstraintSolver>(
    choice: &Choice<NonStereoWriterState<S>>,
    atom: AtomId,
) -> Vec<(String, NonStereoWriterState<S>)> {
    let prefix = choice.text();
    declared_atom_witnesses(&choice.successor().surface, atom)
        .into_iter()
        .filter(|target| target.starts_with(prefix))
        .flat_map(|target| {
            consume_atom_spelling_states(choice.successor().clone(), &target, prefix.len())
                .into_iter()
                .filter(|successor| atom_was_emitted(successor, atom))
                .map(move |successor| (target.clone(), successor))
        })
        .collect()
}

fn pending_atom_entries<S: ConstraintSolver>(
    state: NonStereoWriterState<S>,
    atom: AtomId,
) -> Vec<(NonStereoWriterState<S>, String)> {
    let mut pending = vec![(state, 0_usize)];
    let mut entries = Vec::new();
    let mut explored = 0_usize;
    while let Some((state, atom_offset)) = pending.pop() {
        explored += 1;
        assert!(
            explored <= 100_000,
            "test exceeded its pending-atom path bound"
        );
        for choice in state.choices().expect("production choices are available") {
            if !selected_atom_spelling_transition(&state, &choice, atom_offset) {
                continue;
            }
            let text = choice.text().to_owned();
            let successor = choice.into_successor();
            let starts_target_spelling = successor
                .atom_spelling
                .as_ref()
                .is_some_and(|progress| progress.atom == atom)
                || (successor.active_atom() == Some(atom) && state.active_atom() != Some(atom));
            if starts_target_spelling {
                entries.push((successor, text));
                continue;
            }
            let next_offset = next_atom_spelling_offset(&state, &successor, atom_offset, &text);
            pending.push((successor, next_offset));
        }
    }
    entries
}

fn complete_atom_witness_from_prefix<S: ConstraintSolver>(
    state: NonStereoWriterState<S>,
    atom: AtomId,
    prefix: &str,
) -> Vec<(String, NonStereoWriterState<S>)> {
    declared_atom_witnesses(&state.surface, atom)
        .into_iter()
        .filter(|target| target.starts_with(prefix))
        .flat_map(|target| {
            if target.len() == prefix.len() {
                return atom_was_emitted(&state, atom)
                    .then_some((target, state.clone()))
                    .into_iter()
                    .collect::<Vec<_>>();
            }
            consume_atom_spelling_states(state.clone(), &target, prefix.len())
                .into_iter()
                .filter(|successor| atom_was_emitted(successor, atom))
                .map(|successor| (target.clone(), successor))
                .collect()
        })
        .collect()
}

pub(super) fn only_root_atom<S: ConstraintSolver>(
    state: &NonStereoWriterState<S>,
    atom: AtomId,
) -> (String, NonStereoWriterState<S>) {
    assert_eq!(state.active_atom(), None);
    assert_eq!(state.pending, None);
    for witness in declared_atom_witnesses(&state.surface, atom) {
        let matching = completed_atom_witnesses(state, atom)
            .into_iter()
            .filter(|(candidate, _)| candidate == &witness)
            .collect::<Vec<_>>();
        if let [(_, rooted)] = matching.as_slice() {
            let emitted_text = if state.surface.tetrahedral_center(atom).is_some() {
                witness
            } else {
                state.surface.atom_text(atom).to_owned()
            };
            return (emitted_text, rooted.clone());
        }
        assert!(
            matching.is_empty(),
            "one declared root spelling must identify one test path"
        );
    }
    panic!("no declared root spelling is accepted for AtomId {atom:?}");
}

fn only_component_separator<S: ConstraintSolver>(
    state: &NonStereoWriterState<S>,
    root: AtomId,
) -> (String, NonStereoWriterState<S>) {
    let choices = ring_witness_choices(&state)
        .into_iter()
        .filter(|choice| {
            choice.text() == "."
                && choice.successor().pending == Some(PendingEmission::SegmentRootAtom(root))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        choices.len(),
        1,
        "the selected root identifies one dot path"
    );
    let choice = choices.into_iter().next().unwrap();
    (choice.text, choice.successor)
}

fn choices_matching_declared_witnesses<S: ConstraintSolver>(
    state: &NonStereoWriterState<S>,
    atom_offset: usize,
) -> Vec<Choice<NonStereoWriterState<S>>> {
    ring_witness_choices(&state)
        .into_iter()
        .filter(|choice| selected_atom_spelling_transition(state, choice, atom_offset))
        .collect()
}

fn next_atom_spelling_offset<S: ConstraintSolver>(
    state: &NonStereoWriterState<S>,
    successor: &NonStereoWriterState<S>,
    atom_offset: usize,
    text: &str,
) -> usize {
    match (&state.atom_spelling, &successor.atom_spelling) {
        (Some(current), Some(next)) if current.atom == next.atom => atom_offset + text.len(),
        (_, Some(_)) => text.len(),
        (_, None) => 0,
    }
}

fn declared_successors<S: ConstraintSolver>(
    state: &NonStereoWriterState<S>,
    atom_offset: usize,
) -> Vec<(NonStereoWriterState<S>, usize)> {
    state
        .choices()
        .expect("production choices are available")
        .into_iter()
        .filter_map(|choice| {
            if !selected_atom_spelling_transition(state, &choice, atom_offset)
                || !selected_ring_label_witness(state, &choice)
            {
                return None;
            }
            let text = choice.text().to_owned();
            let successor = choice.into_successor();
            let next_offset = next_atom_spelling_offset(state, &successor, atom_offset, &text);
            Some((successor, next_offset))
        })
        .collect()
}

pub(super) fn selected_ring_label_witness<S: ConstraintSolver>(
    state: &NonStereoWriterState<S>,
    choice: &Choice<NonStereoWriterState<S>>,
) -> bool {
    let successor = choice.successor();
    let assigned = successor.labels.bonds_by_slot.keys()
        .find(|slot| !state.labels.bonds_by_slot.contains_key(slot)).copied();
    let released = state.labels.bonds_by_slot.keys()
        .find(|slot| !successor.labels.bonds_by_slot.contains_key(slot)).copied();
    let pending_label = match &state.pending {
        Some(PendingEmission::RingOpeningLabel { label_slot, .. })
        | Some(PendingEmission::RingClosureLabel { label_slot, .. }) => Some(label_slot),
        _ => None,
    };
    if let Some(slot) = assigned {
        if slot != preferred_test_ring_label(&state.labels) {
            return false;
        }
        // Explicit operators allocate the ID before its visible label.
        return choice.text() == ring_label_number_text(slot.index())
            || matches!(&successor.pending, Some(PendingEmission::RingOpeningLabel { .. }));
    }
    if let Some(slot) = pending_label.copied().or(released) {
        return choice.text() == ring_label_number_text(slot.index())
            || matches!(&successor.pending, Some(PendingEmission::RingClosureLabel { .. }));
    }
    true
}

pub(super) fn ring_witness_choices<S: ConstraintSolver>(
    state: &NonStereoWriterState<S>,
) -> Vec<Choice<NonStereoWriterState<S>>> {
    state
        .choices()
        .expect("production ring choices are available")
        .into_iter()
        .filter(|choice| selected_ring_label_witness(state, choice))
        .collect()
}

fn preferred_test_ring_label(labels: &RingLabels) -> RingLabelSlot {
    (1..=99)
        .chain(0..=0)
        .map(RingLabelSlot)
        .find(|slot| !labels.bonds_by_slot.contains_key(slot))
        .expect("fixture witness family has a free ring label")
}

fn consume_atom_spelling_states<S: ConstraintSolver>(
    state: NonStereoWriterState<S>,
    target: &str,
    offset: usize,
) -> Vec<NonStereoWriterState<S>> {
    let Some(suffix) = target.get(offset..) else {
        return Vec::new();
    };
    let mut states = vec![state];
    for expected in suffix.chars() {
        let mut successors = Vec::new();
        for state in states {
            for choice in state
                .choices()
                .expect("production spelling choices are available")
            {
                if choice.text() == expected.to_string() {
                    successors.push(choice.into_successor());
                }
            }
        }
        states = successors;
        if states.is_empty() {
            break;
        }
    }
    states
        .into_iter()
        .filter(|state| !state.is_spelling_atom())
        .collect()
}

fn choice_at<S: ConstraintSolver>(
    state: &NonStereoWriterState<S>,
    index: usize,
) -> (String, NonStereoWriterState<S>) {
    let choice = ring_witness_choices(&state).into_iter().nth(index).unwrap();
    (choice.text, choice.successor)
}

fn initial(surface: &PreparedNonStereo) -> State {
    State::initial(surface).unwrap().unwrap_consistent()
}

#[test]
fn compact_connected_rings_reuse_label_after_closure() {
    let (surface, _, _) = fixture(
        &["C"; 6],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
            (2, 0, NonStereoBondToken::Elided),
            (2, 3, NonStereoBondToken::Elided),
            (3, 4, NonStereoBondToken::Elided),
            (4, 5, NonStereoBondToken::Elided),
            (5, 3, NonStereoBondToken::Elided),
        ],
    );
    let state = State::initial_with_regime(&surface, WriterRegime::GenericCompact)
        .unwrap()
        .unwrap_consistent();
    let witness = "C0CC0C0CC0";
    let mut pending = vec![(state, 0_usize)];
    let mut released = false;
    let mut reopened = false;
    let mut accepted = false;
    let mut explored = 0_usize;
    while let Some((state, offset)) = pending.pop() {
        explored += 1;
        assert!(explored <= 500, "compact ring witness search exceeded its bound");
        if offset == 5 && state.labels.is_clean() {
            released = true;
        }
        if offset == 7 && state.labels.bonds_by_slot.contains_key(&RingLabelSlot(0)) {
            reopened = true;
        }
        if offset == witness.len() {
            accepted |= state.is_accepted();
            continue;
        }
        for choice in state.choices().expect("compact ring choices are available") {
            let text = choice.text();
            if !text.is_empty() && witness[offset..].starts_with(text) {
                let next_offset = offset + text.len();
                pending.push((choice.into_successor(), next_offset));
            }
        }
    }
    assert!(released, "first ring must release label zero");
    assert!(reopened, "second ring must reuse label zero");
    assert!(accepted, "connected ring witness must complete");
}

/// Walk structural paths through the finite atom spellings declared by tests.
/// This bounds test traversal; it does not count the complete atom language.
fn reachable_terminal_states<S: ConstraintSolver>(
    state: NonStereoWriterState<S>,
) -> Vec<NonStereoWriterState<S>> {
    reachable_terminal_states_with_bond_spelling(state, true)
}

fn reachable_terminal_states_with_bond_spelling<S: ConstraintSolver>(
    state: NonStereoWriterState<S>,
    include_explicit_single: bool,
) -> Vec<NonStereoWriterState<S>> {
    let mut pending = vec![(state, 0_usize)];
    let mut complete = Vec::new();
    let mut explored = 0_usize;
    while let Some((state, atom_offset)) = pending.pop() {
        explored += 1;
        assert!(explored <= 100_000, "writer test exceeded its state bound");
        if state.is_accepted() {
            complete.push(state);
            continue;
        }
        let choices = ring_witness_choices(&state);
        assert!(
            !choices.is_empty(),
            "nonaccepted writer state has no continuation"
        );
        for choice in choices {
            if (!include_explicit_single && choice.text() == "-")
                || !selected_atom_spelling_transition(&state, &choice, atom_offset)
                || !selected_ring_label_witness(&state, &choice)
            {
                continue;
            }
            let token = choice.text().to_owned();
            let successor = choice.into_successor();
            let next_offset = next_atom_spelling_offset(&state, &successor, atom_offset, &token);
            pending.push((successor, next_offset));
        }
    }
    complete
}

#[test]
fn surface_accepts_general_graphs_and_rejects_invalid_bindings() {
    let empty = PreparedMolecule::new(PreparedGraphBuilder::new().build());
    let empty = PreparedNonStereo::new(empty, Vec::new(), Vec::new()).unwrap();
    assert_eq!(empty.molecule().graph().atom_count(), 0);

    let mut graph = PreparedGraphBuilder::new();
    graph.add_atom().unwrap();
    graph.add_atom().unwrap();
    let disconnected = PreparedMolecule::new(graph.build());
    let disconnected = PreparedNonStereo::new(
        disconnected,
        vec!["C".to_owned(), "O".to_owned()],
        Vec::new(),
    )
    .unwrap();
    assert_eq!(disconnected.molecule().graph().atom_count(), 2);

    let mut graph = PreparedGraphBuilder::new();
    graph.add_atom().unwrap();
    let single = PreparedMolecule::new(graph.build());
    assert!(matches!(
        PreparedNonStereo::new(single.clone(), Vec::new(), Vec::new()),
        Err(PreparedNonStereoError::AtomTextCountMismatch { .. })
    ));
    assert!(matches!(
        PreparedNonStereo::new(single, vec![String::new()], Vec::new()),
        Err(PreparedNonStereoError::EmptyAtomText(atom))
            if atom == AtomId::new(0)
    ));

    let mut graph = PreparedGraphBuilder::new();
    let atoms: [AtomId; 2] = std::array::from_fn(|_| graph.add_atom().unwrap());
    graph.add_bond(atoms[0], atoms[1]).unwrap();
    let bonded = PreparedMolecule::new(graph.build());
    assert!(matches!(
        PreparedNonStereo::new(bonded, vec!["C".to_owned(), "O".to_owned()], Vec::new(),),
        Err(PreparedNonStereoError::BondTokenCountMismatch { .. })
    ));
}

#[test]
fn surface_prepares_local_bond_representation_domains_without_binary_factors() {
    let (surface, _, bonds) = fixture(
        &["A", "B", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Double),
        ],
    );
    let molecule = surface.molecule();
    let model = molecule.constraint_model();
    let initial_domain = |bond| {
        let variable = molecule.bond_decision_variable(bond).unwrap();
        model.variable(variable).unwrap().initial_domain()
    };

    assert_eq!(
        initial_domain(bonds[0]),
        BondRepresentation::elided_domain().union(BondRepresentation::explicit_domain())
    );
    assert_eq!(
        initial_domain(bonds[1]),
        BondRepresentation::explicit_domain()
    );
    assert_eq!(
        molecule.bond_role_partition(bonds[0]),
        Some(BondRepresentation::role_partition())
    );
    assert_eq!(model.factor_count(), 1 + surface.ring_count_factors.iter().flatten().count());
    assert!(matches!(
        model.factor(crate::ids::FactorId::new((model.factor_count() - 1) as u32)),
        Some(crate::model::FactorDefinition::TraversalTopology(_))
    ));
}

#[test]
fn surface_prepares_one_latent_layout_context_per_tetrahedral_entry() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms: [AtomId; 5] = std::array::from_fn(|_| graph.add_atom().unwrap());
    let bonds: [BondId; 4] =
        std::array::from_fn(|index| graph.add_bond(atoms[0], atoms[index + 1]).unwrap());
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        vec![
            PreparedAtomToken::Tetrahedral {
                reference_order: bonds.map(TetrahedralLigand::Bond),
                text_by_parity: ["[C@]".to_owned(), "[C@@]".to_owned()],
            },
            PreparedAtomToken::Fixed("A".to_owned()),
            PreparedAtomToken::Fixed("B".to_owned()),
            PreparedAtomToken::Fixed("C".to_owned()),
            PreparedAtomToken::Fixed("D".to_owned()),
        ],
        vec![NonStereoBondToken::Elided; bonds.len()],
    )
    .unwrap();
    let model = surface.molecule().constraint_model();
    let center = surface.tetrahedral_center(atoms[0]).unwrap();
    let (resource, mask, free_bits) = surface.ordered_ring_factors[atoms[0].index()].as_ref().unwrap();
    assert!(surface.ring_count_factors[atoms[0].index()].is_none());
    let Some(FactorDefinition::OrderedRingPhase(phase)) = model.factor(*resource) else {
        panic!("tetrahedral center must own one ordered ring resource");
    };
    assert_eq!(phase.order_variable(), center.order_variable);
    assert_eq!(phase.waiting_mask_variable(), *mask);
    assert_eq!(phase.free_bits(), free_bits.as_ref());

    assert_eq!(model.variable_count(), bonds.len() + 2
        + surface.ring_count_factors.iter().flatten().map(|(_, bits)| bits.len()).sum::<usize>()
        + 1 + free_bits.len());
    assert_eq!(model.factor_count(), 1 + bonds.len() + 1
        + surface.ring_count_factors.iter().flatten().count() + 1);
    assert_eq!(
        model.potential_factors_for_variable(center.order_variable),
        Some(
            &[
                center.root_layout_factor,
                center.entry_layout_factors[0].1,
                center.entry_layout_factors[1].1,
                center.entry_layout_factors[2].1,
                center.entry_layout_factors[3].1,
                *resource,
            ][..]
        )
    );
    assert_eq!(
        model.initial_factors_for_variable(center.order_variable),
        Some(&[][..])
    );
    assert_eq!(
        model
            .variable(center.order_variable)
            .unwrap()
            .initial_domain(),
        full_order_domain()
    );
    assert_eq!(
        model
            .variable(center.role_pattern_variable)
            .unwrap()
            .initial_domain(),
        full_role_pattern_domain(bonds.len())
    );
    for factor in std::iter::once(center.root_layout_factor).chain(
        center
            .entry_layout_factors
            .iter()
            .map(|(_, factor)| *factor),
    ) {
        assert_eq!(
            model.factor_activation(factor),
            Some(crate::model::FactorActivation::Latent)
        );
    }
}

#[test]
fn prepared_tetrahedral_layouts_preserve_every_fixed_local_role_assignment() {
    let mut checked_context_patterns = 0;
    let mut legal_context_patterns = 0;
    let mut illegal_entry_ring_patterns = 0;
    let mut checked_parity_transitions = 0;
    for bond_count in [3_usize, 4] {
        for reverse_markers in [false, true] {
            let mut graph = PreparedGraphBuilder::new();
            let atoms = (0..=bond_count).map(|_| graph.add_atom().unwrap()).collect::<Vec<_>>();
            let bonds = (0..bond_count)
                .map(|index| graph.add_bond(atoms[0], atoms[index + 1]).unwrap())
                .collect::<Vec<_>>();
            let mut reference = bonds.iter().copied().map(TetrahedralLigand::Bond).collect::<Vec<_>>();
            if bond_count == 3 {
                reference.push(TetrahedralLigand::VirtualHydrogen);
            }
            let mut markers = ["[C@]".to_owned(), "[C@@]".to_owned()];
            if reverse_markers {
                markers.reverse();
            }
            let mut tokens = vec![PreparedAtomToken::Tetrahedral {
                reference_order: reference.try_into().unwrap(),
                text_by_parity: markers,
            }];
            tokens.extend((0..bond_count).map(|_| PreparedAtomToken::Fixed("C".to_owned())));
            let surface = PreparedNonStereo::with_atom_tokens_and_directional_for_writer(
                PreparedMolecule::new(graph.build()),
                tokens,
                vec![NonStereoBondToken::Elided; bond_count],
                Vec::new(),
                WriterRegime::Generic { component_order: ComponentOrder::AnyOrder },
                true,
            ).unwrap();
            assert_eq!(surface.molecule().topology(), crate::model::TraversalTopologyRequirement::Forest);
            let center = surface.tetrahedral_center(atoms[0]).unwrap();
            assert_eq!(center.at_for_even_parity, !reverse_markers);
            let model = surface.molecule().constraint_model();
            let initial_solver = <NativeSolverState as ConstraintSolver>::initial(
                surface.molecule().constraint_model_arc(),
            ).unwrap().unwrap_consistent();
            let topology_factors = model.initial_factor_ids().iter().copied().filter(|id| {
                matches!(model.factor(*id), Some(crate::model::FactorDefinition::TraversalTopology(factor))
                    if factor.requirement() == crate::model::TraversalTopologyRequirement::Forest)
            }).collect::<Vec<_>>();
            assert_eq!(topology_factors.len(), 1);
            assert_eq!(initial_solver.factor_is_active(topology_factors[0]), Some(true));
            for (entry, factor_id) in std::iter::once((None, center.root_layout_factor))
                .chain(center.entry_layout_factors.iter().map(|(bond, factor)| (Some(*bond), *factor))) {
                assert_eq!(initial_solver.factor_is_active(factor_id), Some(false));
                let Some(crate::model::FactorDefinition::TetrahedralLayout(factor)) = model.factor(factor_id) else {
                    panic!("prepared context must have a tetrahedral layout factor");
                };
                let mut prefix = entry.map(TetrahedralLigand::Bond).into_iter().collect::<Vec<_>>();
                if bond_count == 3 {
                    prefix.push(TetrahedralLigand::VirtualHydrogen);
                }
                for pattern in 0..(1_u8 << bond_count) {
                    checked_context_patterns += 1;
                    let mut ring = Vec::new();
                    let mut traversal = Vec::new();
                    for (bit, bond) in bonds.iter().copied().enumerate() {
                        if Some(bond) == entry { continue; }
                        let group = if pattern & (1 << bit) == 0 { &mut traversal } else { &mut ring };
                        group.push(TetrahedralLigand::Bond(bond));
                    }
                    let entry_is_ring = entry.is_some_and(|bond| {
                        let bit = bonds.iter().position(|candidate| *candidate == bond).unwrap();
                        pattern & (1 << bit) != 0
                    });
                    if entry_is_ring {
                        illegal_entry_ring_patterns += 1;
                    } else {
                        legal_context_patterns += 1;
                    }
                    let mut expected = BTreeSet::new();
                    if !entry_is_ring {
                        for ring_order in permutations(&ring) {
                            for traversal_order in permutations(&traversal) {
                                expected.insert(prefix.iter().chain(&ring_order).chain(&traversal_order)
                                    .copied().collect::<Vec<_>>());
                            }
                        }
                    }
                    let actual = factor.allowed_orders(pattern).iter()
                        .map(|value| independent_tetrahedral_order(&center.reference_order, value).to_vec())
                        .collect::<BTreeSet<_>>();
                    assert_eq!(actual, expected, "prepared bond_count={bond_count}, reverse={reverse_markers}, entry={entry:?}, pattern={pattern}");
                    for parity in TetrahedralParity::ALL {
                        checked_parity_transitions += 1;
                        let expected_marker = if (parity == TetrahedralParity::Even) != reverse_markers {
                            TetrahedralMarker::At
                        } else {
                            TetrahedralMarker::AtAt
                        };
                        assert_eq!(surface.atom_spelling_context(atoms[0], Some(parity)).tetrahedral_marker,
                            Some(expected_marker));
                        let expected_count = expected.iter().filter(|order| {
                            let positions = order.iter().map(|ligand| center.reference_order.iter()
                                .position(|candidate| candidate == ligand).unwrap()).collect::<Vec<_>>();
                            let inversions = (0..4).flat_map(|left| ((left + 1)..4).map(move |right| (left, right)))
                                .filter(|(left, right)| positions[*left] > positions[*right]).count();
                            (inversions % 2 == 0) == (parity == TetrahedralParity::Even)
                        }).count();
                        assert_eq!(factor.allowed_orders(pattern).intersect(parity_domain(parity)).len(), expected_count);
                        let expected_values = Domain::from_indices((0..24).filter(|value| {
                            expected.contains(&independent_tetrahedral_order(&center.reference_order, *value).to_vec())
                                && (independent_tetrahedral_order_is_even(&center.reference_order, *value)
                                    == (parity == TetrahedralParity::Even))
                        })).unwrap();
                        let mut restrictions = vec![
                            (center.role_pattern_variable, Domain::singleton(pattern).unwrap()),
                            (center.order_variable, parity_domain(parity)),
                        ];
                        for (bit, bond) in bonds.iter().copied().enumerate() {
                            let partition = surface.molecule().bond_role_partition(bond).unwrap();
                            restrictions.push((
                                surface.molecule().bond_decision_variable(bond).unwrap(),
                                if pattern & (1 << bit) == 0 {
                                    partition.traversal_values()
                                } else {
                                    partition.ring_values()
                                },
                            ));
                        }
                        let result = initial_solver.transitioned(&restrictions, &[factor_id]).unwrap();
                        if expected_count == 0 {
                            assert!(matches!(result, Consistency::Contradiction),
                                "illegal or wrong-parity assignment survived: bond_count={bond_count}, entry={entry:?}, pattern={pattern}, parity={parity:?}");
                        } else {
                            let solved = result.unwrap_consistent();
                            assert_eq!(solved.factor_is_active(topology_factors[0]), Some(true));
                            assert_eq!(solved.factor_is_active(factor_id), Some(true));
                            assert_eq!(solved.domain_bound(center.order_variable),
                                Some(expected_values));
                            assert_eq!(solved.domain_bound(center.role_pattern_variable),
                                Some(Domain::singleton(pattern).unwrap()));
                            for (bit, bond) in bonds.iter().copied().enumerate() {
                                let partition = surface.molecule().bond_role_partition(bond).unwrap();
                                let role = if pattern & (1 << bit) == 0 {
                                    partition.traversal_values()
                                } else {
                                    partition.ring_values()
                                };
                                assert_eq!(solved.domain_bound(surface.molecule().bond_decision_variable(bond).unwrap())
                                    .unwrap().intersect(role), solved.domain_bound(surface.molecule().bond_decision_variable(bond).unwrap()).unwrap());
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!((checked_context_patterns, legal_context_patterns,
        illegal_entry_ring_patterns, checked_parity_transitions), (224, 136, 88, 448));
}

#[test]
fn fixed_tetrahedral_roles_reach_the_writer_atom_entry_seam() {
    let mut checked = 0;
    let mut legal = 0;
    let mut illegal_entries = 0;
    let mut parity_probes = 0;
    for bond_count in [3_usize, 4] {
        for reverse_markers in [false, true] {
            let mut graph = PreparedGraphBuilder::new();
            let atoms = (0..=bond_count).map(|_| graph.add_atom().unwrap()).collect::<Vec<_>>();
            let bonds = (0..bond_count)
                .map(|index| graph.add_bond(atoms[0], atoms[index + 1]).unwrap())
                .collect::<Vec<_>>();
            let mut reference = bonds.iter().copied().map(TetrahedralLigand::Bond).collect::<Vec<_>>();
            if bond_count == 3 {
                reference.push(TetrahedralLigand::VirtualHydrogen);
            }
            let reference: [TetrahedralLigand; 4] = reference.try_into().unwrap();
            let marker = |parity: TetrahedralParity| {
                let at = (parity == TetrahedralParity::Even) != reverse_markers;
                match (bond_count, at) {
                    (3, true) => "[C@H]",
                    (3, false) => "[C@@H]",
                    (4, true) => "[C@]",
                    (4, false) => "[C@@]",
                    _ => unreachable!(),
                }
            };
            let mut tokens = vec![PreparedAtomToken::Tetrahedral {
                reference_order: reference,
                text_by_parity: [marker(TetrahedralParity::Even).to_owned(), marker(TetrahedralParity::Odd).to_owned()],
            }];
            tokens.extend((0..bond_count).map(|index| PreparedAtomToken::Fixed(char::from(b'A' + index as u8).to_string())));
            let surface = PreparedNonStereo::with_atom_tokens_and_directional_for_writer(
                PreparedMolecule::new(graph.build()), tokens,
                vec![NonStereoBondToken::Elided; bond_count], Vec::new(),
                WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true,
            ).unwrap();
            let center = surface.tetrahedral_center(atoms[0]).unwrap();
            for entry in std::iter::once(None).chain(bonds.iter().copied().map(Some)) {
                let factor = center.layout_factor(entry);
                for pattern in 0..(1_u8 << bond_count) {
                    checked += 1;
                    let entry_is_ring = entry.is_some_and(|bond| {
                        let bit = bonds.iter().position(|candidate| *candidate == bond).unwrap();
                        pattern & (1 << bit) != 0
                    });
                    let mut ring = Vec::new();
                    let mut traversal = Vec::new();
                    let mut restrictions = Vec::new();
                    for (bit, bond) in bonds.iter().copied().enumerate() {
                        let partition = surface.molecule().bond_role_partition(bond).unwrap();
                        let ring_role = pattern & (1 << bit) != 0;
                        restrictions.push((surface.molecule().bond_decision_variable(bond).unwrap(),
                            if ring_role { partition.ring_values() } else { partition.traversal_values() }));
                        if Some(bond) != entry {
                            (if ring_role { &mut ring } else { &mut traversal }).push(TetrahedralLigand::Bond(bond));
                        }
                    }
                    let mut seeded = initial(&surface);
                    seeded.structural = seeded.structural.transitioned_semantics(&restrictions, &[])
                        .unwrap().unwrap_consistent();
                    let mut actual = Vec::new();
                    if let Some(entry_bond) = entry {
                        let index = bonds.iter().position(|bond| *bond == entry_bond).unwrap();
                        for (_, leaf) in completed_atom_witnesses(&seeded, atoms[index + 1]) {
                            for choice in ring_witness_choices(&leaf) {
                                let pending = choice.successor();
                                if pending_atom(pending) != Some(atoms[0])
                                    && pending.active_atom() != Some(atoms[0]) {
                                    continue;
                                }
                                actual.extend(complete_atom_witness_from_prefix(pending.clone(), atoms[0], choice.text())
                                    .into_iter().filter(|(_, state)| {
                                        state.structural.active_local_bond_order().entry_bond == Some(entry_bond)
                                    }));
                            }
                        }
                    } else {
                        actual = completed_atom_witnesses(&seeded, atoms[0]);
                    }
                    if entry_is_ring {
                        illegal_entries += 1;
                        assert!(actual.is_empty(), "ring bond cannot be traversal entry: {bond_count} {entry:?} {pattern}");
                        parity_probes += 2;
                        continue;
                    }
                    legal += 1;
                    let mut prefix = entry.map(TetrahedralLigand::Bond).into_iter().collect::<Vec<_>>();
                    if bond_count == 3 { prefix.push(TetrahedralLigand::VirtualHydrogen); }
                    let expected_orders = permutations(&ring).into_iter().flat_map(|ring_order| {
                        permutations(&traversal).into_iter().map({
                            let prefix = prefix.clone();
                            move |traversal_order| prefix.iter().chain(&ring_order).chain(&traversal_order)
                                .copied().collect::<Vec<_>>()
                        })
                    }).collect::<BTreeSet<_>>();
                    for parity in TetrahedralParity::ALL {
                        parity_probes += 1;
                        let expected_domain = Domain::from_indices((0..24).filter(|value| {
                            expected_orders.contains(&independent_tetrahedral_order(&reference, *value).to_vec())
                                && independent_tetrahedral_order_is_even(&reference, *value)
                                    == (parity == TetrahedralParity::Even)
                        })).unwrap();
                        let matching = actual.iter().filter(|(text, _)| text == marker(parity)).collect::<Vec<_>>();
                        assert_eq!(!matching.is_empty(), !expected_domain.is_empty(),
                            "writer entry parity: {bond_count} {entry:?} {pattern} {parity:?}");
                        for (_, state) in matching {
                            assert_eq!(state.structural.semantic_domain(center.order_variable), expected_domain,
                                "writer order: {bond_count} {entry:?} {pattern} {parity:?}");
                            assert_eq!(state.structural.semantic_domain(center.role_pattern_variable), Domain::singleton(pattern).unwrap());
                            assert_eq!(state.structural.semantic_domain(surface.molecule().root_variable(atoms[0]).unwrap()),
                                Domain::singleton(u8::from(entry.is_none())).unwrap());
                            assert!(state.structural.atom_is_visited(atoms[0]));
                            assert!(state.structural.factor_is_active(factor));
                            assert!(std::iter::once(center.root_layout_factor)
                                .chain(center.entry_layout_factors.iter().map(|(_, id)| *id))
                                .filter(|id| *id != factor)
                                .all(|id| !state.structural.factor_is_active(id)));
                            for (variable, role) in &restrictions {
                                assert_eq!(state.structural.semantic_domain(*variable).intersect(*role),
                                    state.structural.semantic_domain(*variable));
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!((checked, legal, illegal_entries, parity_probes), (224, 136, 88, 448));
}

#[test]
fn branch_and_explicit_inline_tetrahedral_entries_activate_the_selected_layout() {
    for (token, has_sibling) in [
        (NonStereoBondToken::Elided, true),
        (NonStereoBondToken::Double, true),
        (NonStereoBondToken::Double, false),
    ] {
        let (surface, atoms, bonds) = entered_tetrahedral_fixture(token, has_sibling);
        let center = surface.tetrahedral_center(atoms[1]).unwrap();
        let (_, parent) = only_root_atom(&initial(&surface), atoms[0]);
        let pending = if has_sibling {
            let branch = ring_witness_choices(&parent)
                .into_iter()
                .find(|choice| matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == incident(&surface, atoms[0], bonds[0])))
                .unwrap().into_successor();
            if token == NonStereoBondToken::Double {
                only_choice(&branch, "=").1
            } else {
                branch
            }
        } else {
            only_choice(&parent, "=").1
        };
        let entries = completed_atom_witnesses(&pending, atoms[1]);
        assert_eq!(entries.len(), 2);
        for (text, state) in entries {
            let parity = if text == "[C@]" { TetrahedralParity::Even } else {
                assert_eq!(text, "[C@@]");
                TetrahedralParity::Odd
            };
            assert_eq!(state.structural.active_local_bond_order().entry_bond, Some(bonds[0]));
            assert!(state.structural.atom_is_visited(atoms[1]));
            assert!(state.structural.factor_is_active(center.layout_factor(Some(bonds[0]))));
            assert!(!state.structural.factor_is_active(center.root_layout_factor));
            assert_eq!(state.structural.semantic_domain(center.order_variable),
                center.token_domain(Some(bonds[0]), parity));
        }
    }
}

#[test]
fn adjacent_tetrahedral_child_entry_retains_the_parent_layout_and_fixed_roles() {
    let (surface, atoms, bonds) = adjacent_ring_coupled_tetrahedral_fixture();
    let centers = [atoms[0], atoms[1]].map(|atom| surface.tetrahedral_center(atom).unwrap());
    let roles = [false, false, false, true, false, false];
    let restrictions = bonds.iter().copied().zip(roles).map(|(bond, ring)| {
        let partition = surface.molecule().bond_role_partition(bond).unwrap();
        (surface.molecule().bond_decision_variable(bond).unwrap(),
            if ring { partition.ring_values() } else { partition.traversal_values() })
    }).collect::<Vec<_>>();
    let mut seeded = initial(&surface);
    seeded.structural = seeded.structural.transitioned_semantics(&restrictions, &[])
        .unwrap().unwrap_consistent();
    let roots = completed_atom_witnesses(&seeded, atoms[0]);
    assert_eq!(roots.len(), 2);
    let mut child_entries = 0;
    for (parent_text, parent) in roots {
        assert!(parent.structural.factor_is_active(centers[0].root_layout_factor));
        let parent_domain = parent.structural.semantic_domain(centers[0].order_variable);
        let parent_orders = [
            [TetrahedralLigand::VirtualHydrogen, TetrahedralLigand::Bond(bonds[3]),
                TetrahedralLigand::Bond(bonds[0]), TetrahedralLigand::Bond(bonds[4])],
            [TetrahedralLigand::VirtualHydrogen, TetrahedralLigand::Bond(bonds[3]),
                TetrahedralLigand::Bond(bonds[4]), TetrahedralLigand::Bond(bonds[0])],
        ];
        let parent_even = parent_text == "[C@H]";
        let expected_parent = Domain::from_indices((0..24).filter(|value| {
            parent_orders.contains(&independent_tetrahedral_order(&centers[0].reference_order, *value))
                && independent_tetrahedral_order_is_even(&centers[0].reference_order, *value) == parent_even
        })).unwrap();
        assert_eq!(parent_domain, expected_parent);
        for (pending, prefix) in pending_atom_entries(parent, atoms[1]) {
            for (child_text, child) in complete_atom_witness_from_prefix(pending, atoms[1], &prefix) {
                if child.structural.active_local_bond_order().entry_bond != Some(bonds[0]) {
                    continue;
                }
                child_entries += 1;
                assert_eq!(child.structural.active_local_bond_order().entry_bond, Some(bonds[0]));
                assert!(child.structural.factor_is_active(centers[0].root_layout_factor));
                assert!(child.structural.factor_is_active(centers[1].layout_factor(Some(bonds[0]))));
                assert_eq!(child.structural.semantic_domain(centers[0].order_variable), expected_parent);
                let child_orders = [
                    [TetrahedralLigand::Bond(bonds[0]), TetrahedralLigand::VirtualHydrogen,
                        TetrahedralLigand::Bond(bonds[1]), TetrahedralLigand::Bond(bonds[5])],
                    [TetrahedralLigand::Bond(bonds[0]), TetrahedralLigand::VirtualHydrogen,
                        TetrahedralLigand::Bond(bonds[5]), TetrahedralLigand::Bond(bonds[1])],
                ];
                let child_even = child_text == "[C@H]";
                let expected_child = Domain::from_indices((0..24).filter(|value| {
                    child_orders.contains(&independent_tetrahedral_order(&centers[1].reference_order, *value))
                        && independent_tetrahedral_order_is_even(&centers[1].reference_order, *value) == child_even
                })).unwrap();
                assert_eq!(child.structural.semantic_domain(centers[1].order_variable), expected_child);
                for (variable, role) in &restrictions {
                    assert_eq!(child.structural.semantic_domain(*variable).intersect(*role),
                        child.structural.semantic_domain(*variable));
                }
            }
        }
    }
    assert!(child_entries > 0);
}

#[test]
fn surface_rejects_invalid_tetrahedral_bindings() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms: [AtomId; 4] = std::array::from_fn(|_| graph.add_atom().unwrap());
    let bonds: [BondId; 3] =
        std::array::from_fn(|index| graph.add_bond(atoms[0], atoms[index + 1]).unwrap());
    let molecule = PreparedMolecule::new(graph.build());
    let fixed = || {
        vec![
            PreparedAtomToken::Fixed("A".to_owned()),
            PreparedAtomToken::Fixed("B".to_owned()),
            PreparedAtomToken::Fixed("C".to_owned()),
        ]
    };
    let prepare = |reference_order, text_by_parity| {
        let mut atoms = vec![PreparedAtomToken::Tetrahedral {
            reference_order,
            text_by_parity,
        }];
        atoms.extend(fixed());
        PreparedNonStereo::with_atom_tokens(
            molecule.clone(),
            atoms,
            vec![NonStereoBondToken::Elided; bonds.len()],
        )
    };

    assert!(matches!(
        prepare(
            [
                TetrahedralLigand::Bond(bonds[0]),
                TetrahedralLigand::Bond(bonds[1]),
                TetrahedralLigand::Bond(bonds[2]),
                TetrahedralLigand::VirtualHydrogen,
            ],
            [String::new(), "[C@@H]".to_owned()],
        ),
        Err(PreparedNonStereoError::EmptyTetrahedralAtomText(atom)) if atom == atoms[0]
    ));
    assert!(matches!(
        prepare(
            [
                TetrahedralLigand::Bond(bonds[0]),
                TetrahedralLigand::Bond(bonds[1]),
                TetrahedralLigand::Bond(bonds[2]),
                TetrahedralLigand::VirtualHydrogen,
            ],
            ["[C@H]".to_owned(), "[C@H]".to_owned()],
        ),
        Err(PreparedNonStereoError::RepeatedTetrahedralAtomText(atom)) if atom == atoms[0]
    ));
    assert!(matches!(
        prepare(
            [
                TetrahedralLigand::Bond(bonds[0]),
                TetrahedralLigand::Bond(bonds[1]),
                TetrahedralLigand::VirtualHydrogen,
                TetrahedralLigand::VirtualHydrogen,
            ],
            ["[C@H]".to_owned(), "[C@@H]".to_owned()],
        ),
        Err(PreparedNonStereoError::MultipleVirtualHydrogens(atom)) if atom == atoms[0]
    ));
    assert!(matches!(
        prepare(
            [
                TetrahedralLigand::Bond(bonds[0]),
                TetrahedralLigand::Bond(bonds[1]),
                TetrahedralLigand::Bond(bonds[1]),
                TetrahedralLigand::VirtualHydrogen,
            ],
            ["[C@H]".to_owned(), "[C@@H]".to_owned()],
        ),
        Err(PreparedNonStereoError::RepeatedTetrahedralLigand(atom)) if atom == atoms[0]
    ));
    assert!(matches!(
        prepare(
            [
                TetrahedralLigand::Bond(bonds[0]),
                TetrahedralLigand::Bond(bonds[1]),
                TetrahedralLigand::Bond(BondId::new(99)),
                TetrahedralLigand::VirtualHydrogen,
            ],
            ["[C@H]".to_owned(), "[C@@H]".to_owned()],
        ),
        Err(PreparedNonStereoError::TetrahedralLigandsDoNotMatchGraph(atom)) if atom == atoms[0]
    ));
}

fn resolved_tetrahedral_order(state: &State, surface: &PreparedNonStereo, atom: AtomId) -> u8 {
    let center = surface.tetrahedral_center(atom).unwrap();
    let domain = state.structural.semantic_domain(center.order_variable);
    assert!(domain.is_singleton());
    domain.iter().next().unwrap()
}

fn independent_tetrahedral_order(
    reference: &[TetrahedralLigand; 4],
    value: u8,
) -> [TetrahedralLigand; 4] {
    permutations(reference)[value as usize]
        .clone()
        .try_into()
        .unwrap()
}

fn independent_tetrahedral_order_is_even(reference: &[TetrahedralLigand; 4], value: u8) -> bool {
    let order = independent_tetrahedral_order(reference, value);
    let positions = order.map(|ligand| {
        reference
            .iter()
            .position(|candidate| *candidate == ligand)
            .unwrap()
    });
    (0..4)
        .flat_map(|left| ((left + 1)..4).map(move |right| (left, right)))
        .filter(|(left, right)| positions[*left] > positions[*right])
        .count()
        % 2
        == 0
}

#[test]
fn tetrahedral_domains_match_independent_permutation_and_parity_enumeration() {
    let reference = [
        TetrahedralLigand::Bond(BondId::new(3)),
        TetrahedralLigand::Bond(BondId::new(7)),
        TetrahedralLigand::VirtualHydrogen,
        TetrahedralLigand::Bond(BondId::new(11)),
    ];
    let orders = (0..24)
        .map(|value| independent_tetrahedral_order(&reference, value))
        .collect::<BTreeSet<_>>();
    assert_eq!(orders.len(), 24);
    assert_eq!(
        (0..24)
            .filter(|value| independent_tetrahedral_order_is_even(&reference, *value))
            .count(),
        12
    );
    for value in 0..24 {
        assert_eq!(
            parity_domain(TetrahedralParity::Even).contains(value),
            independent_tetrahedral_order_is_even(&reference, value)
        );
        let order = independent_tetrahedral_order(&reference, value);
        for prefix_len in 0..=4 {
            let domain = prefix_domain(&reference, &order[..prefix_len]);
            for candidate in 0..24 {
                assert_eq!(
                    domain.contains(candidate),
                    independent_tetrahedral_order(&reference, candidate)[..prefix_len]
                        == order[..prefix_len]
                );
            }
        }
    }
}

#[test]
fn four_arm_root_reaches_every_local_order_under_exactly_one_parity_token() {
    let (surface, atoms, _) = tetrahedral_star_fixture(4);
    let initial = initial(&surface);
    let root_choices = completed_atom_witnesses(&initial, atoms[0]);

    assert_eq!(
        root_choices
            .iter()
            .map(|(text, _)| text.as_str())
            .collect::<Vec<_>>(),
        vec!["[C@]", "[C@@]"]
    );
    let mut orders_by_token = BTreeMap::<String, BTreeSet<u8>>::new();
    for (token, root) in root_choices {
        for terminal in reachable_terminal_states(root) {
            orders_by_token
                .entry(token.clone())
                .or_default()
                .insert(resolved_tetrahedral_order(&terminal, &surface, atoms[0]));
        }
    }

    let reference = &surface
        .tetrahedral_center(atoms[0])
        .unwrap()
        .reference_order;
    let expected_even = (0..24)
        .filter(|value| independent_tetrahedral_order_is_even(reference, *value))
        .collect();
    let expected_odd = (0..24)
        .filter(|value| !independent_tetrahedral_order_is_even(reference, *value))
        .collect();
    assert_eq!(orders_by_token["[C@]"], expected_even);
    assert_eq!(orders_by_token["[C@@]"], expected_odd);
    assert_eq!(orders_by_token["[C@]"].len(), 12);
    assert_eq!(orders_by_token["[C@@]"].len(), 12);
}

#[test]
fn entered_four_arm_center_reaches_all_six_remaining_orders() {
    let (surface, atoms, bonds) = tetrahedral_star_fixture(4);
    let initial = initial(&surface);
    let (_, rooted_leaf) = only_root_atom(&initial, atoms[1]);
    let atom_choices = pending_atom_entries(rooted_leaf, atoms[0])
        .into_iter()
        .flat_map(|(entry, prefix)| complete_atom_witness_from_prefix(entry, atoms[0], &prefix))
        .collect::<Vec<_>>();

    assert_eq!(
        atom_choices
            .iter()
            .map(|(text, _)| text.as_str())
            .collect::<Vec<_>>(),
        vec!["[C@]", "[C@@]", "[C@]", "[C@@]"]
    );
    let mut orders = BTreeSet::new();
    for (_, atom_state) in atom_choices {
        for terminal in reachable_terminal_states(atom_state) {
            let value = resolved_tetrahedral_order(&terminal, &surface, atoms[0]);
            let order = independent_tetrahedral_order(
                &surface
                    .tetrahedral_center(atoms[0])
                    .unwrap()
                    .reference_order,
                value,
            );
            assert_eq!(order[0], TetrahedralLigand::Bond(bonds[0]));
            orders.insert(value);
        }
    }
    assert_eq!(orders.len(), 6);
}

#[test]
fn virtual_hydrogen_has_the_context_position_for_root_and_entry() {
    let (surface, atoms, bonds) = tetrahedral_star_fixture(3);
    let center = surface.tetrahedral_center(atoms[0]).unwrap();
    let initial = initial(&surface);

    let rooted_orders = completed_atom_witnesses(&initial, atoms[0])
        .into_iter()
        .flat_map(|(_, root)| reachable_terminal_states(root))
        .map(|terminal| {
            independent_tetrahedral_order(
                &center.reference_order,
                resolved_tetrahedral_order(&terminal, &surface, atoms[0]),
            )
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(rooted_orders.len(), 6);
    assert!(rooted_orders
        .iter()
        .all(|order| order[0] == TetrahedralLigand::VirtualHydrogen));

    let (_, rooted_leaf) = only_root_atom(&initial, atoms[1]);
    let entered_orders = pending_atom_entries(rooted_leaf, atoms[0])
        .into_iter()
        .flat_map(|(entry, prefix)| complete_atom_witness_from_prefix(entry, atoms[0], &prefix))
        .flat_map(|(_, atom_state)| reachable_terminal_states(atom_state))
        .map(|terminal| {
            independent_tetrahedral_order(
                &center.reference_order,
                resolved_tetrahedral_order(&terminal, &surface, atoms[0]),
            )
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(entered_orders.len(), 2);
    assert!(entered_orders.iter().all(|order| {
        order[..2]
            == [
                TetrahedralLigand::Bond(bonds[0]),
                TetrahedralLigand::VirtualHydrogen,
            ]
    }));
}

#[test]
fn pending_atom_stages_preserve_tetrahedral_parity_branches() {
    let (elided_branch_surface, atoms, bonds) =
        entered_tetrahedral_fixture(NonStereoBondToken::Elided, true);
    let (_, parent) = only_root_atom(&initial(&elided_branch_surface), atoms[0]);
    let parent_choices = ring_witness_choices(&parent);
    let pending_branch = parent_choices
        .into_iter()
        .find(|choice| {
            matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == incident(
                    &elided_branch_surface,
                    atoms[0],
                    bonds[0],
                ))
        })
        .unwrap()
        .into_successor();
    assert_eq!(
        completed_atom_witnesses(&pending_branch, atoms[1])
            .iter()
            .map(|(text, _)| text.as_str())
            .collect::<Vec<_>>(),
        vec!["[C@]", "[C@@]"]
    );

    let (explicit_branch_surface, atoms, bonds) =
        entered_tetrahedral_fixture(NonStereoBondToken::Double, true);
    let (_, parent) = only_root_atom(&initial(&explicit_branch_surface), atoms[0]);
    let pending_bond = ring_witness_choices(&parent)
        .into_iter()
        .find(|choice| {
            matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == incident(
                    &explicit_branch_surface,
                    atoms[0],
                    bonds[0],
                ))
        })
        .unwrap()
        .into_successor();
    let (_, pending_atom) = only_choice(&pending_bond, "=");
    assert_eq!(
        completed_atom_witnesses(&pending_atom, atoms[1])
            .iter()
            .map(|(text, _)| text.as_str())
            .collect::<Vec<_>>(),
        vec!["[C@]", "[C@@]"]
    );

    let (inline_surface, atoms, _) = entered_tetrahedral_fixture(NonStereoBondToken::Double, false);
    let (_, parent) = only_root_atom(&initial(&inline_surface), atoms[0]);
    let (_, pending_atom) = only_choice(&parent, "=");
    assert_eq!(
        completed_atom_witnesses(&pending_atom, atoms[1])
            .iter()
            .map(|(text, _)| text.as_str())
            .collect::<Vec<_>>(),
        vec!["[C@]", "[C@@]"]
    );
}

#[test]
fn component_separator_defers_tetrahedral_parity_until_the_atom_token() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..6)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = (2..6)
        .map(|child| graph.add_bond(atoms[1], atoms[child]).unwrap())
        .collect::<Vec<_>>();
    let mut atom_tokens = vec![
        PreparedAtomToken::Fixed("X".to_owned()),
        PreparedAtomToken::Tetrahedral {
            reference_order: [bonds[0], bonds[1], bonds[2], bonds[3]].map(TetrahedralLigand::Bond),
            text_by_parity: ["[C@]".to_owned(), "[C@@]".to_owned()],
        },
    ];
    atom_tokens.extend(
        ["A", "B", "C", "D"]
            .into_iter()
            .map(|text| PreparedAtomToken::Fixed(text.to_owned())),
    );
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        atom_tokens,
        vec![NonStereoBondToken::Elided; bonds.len()],
    )
    .unwrap();
    let (_, after_first) = only_root_atom(&initial(&surface), atoms[0]);
    let root_choices = ring_witness_choices(&after_first);
    let dot = root_choices
        .into_iter()
        .find(|choice| !completed_atom_witnesses(choice.successor(), atoms[1]).is_empty())
        .unwrap();

    assert_eq!(dot.text(), ".");
    assert_eq!(
        completed_atom_witnesses(dot.successor(), atoms[1])
            .iter()
            .map(|(text, _)| text.as_str())
            .collect::<Vec<_>>(),
        vec!["[C@]", "[C@@]"]
    );
}

#[test]
fn suspended_tetrahedral_parent_retains_prefix_and_filters_child_order() {
    let (surface, atoms, bonds) = tetrahedral_star_fixture(4);
    let center = surface.tetrahedral_center(atoms[0]).unwrap();
    let mut parent = completed_atom_witnesses(&initial(&surface), atoms[0])
        .into_iter()
        .find(|(text, _)| text.contains('@') && !text.contains("@@"))
        .map(|(_, state)| state)
        .unwrap();

    for &bond in &bonds[..2] {
        let selected = ring_witness_choices(&parent)
            .into_iter()
            .find(|choice| {
                matches!(&choice.successor().pending,
                    Some(PendingEmission::BranchTraversalEmission(incident))
                        if incident.bond() == bond
                )
            })
            .unwrap()
            .into_successor();
        let child_atom = branch_event_atom(&selected).expect("branch event names its child AtomId");
        let (_, child) = only_choice(&selected, surface.atom_text(child_atom));
        let (_, restored) = only_choice(&child, ")");
        assert_eq!(restored.active_atom(), Some(atoms[0]));
        parent = restored;
    }

    let local = parent.structural.active_local_bond_order();
    assert_eq!(local.emitted_bonds, bonds[..2]);
    assert_eq!(
        parent.structural.semantic_domain(center.order_variable),
        center
            .prefix_domain_with_bond_order(None, &bonds[..2])
            .intersect(parity_domain(TetrahedralParity::Even))
    );
    let next = ring_witness_choices(&parent);
    assert_eq!(next.len(), 2, "the admitted child has two spelling classes");
    for choice in next {
        assert_eq!(choice.text(), "(");
        let entered = choice.successor().structural.observe_raw().traversal;
        assert_ne!(entered.active_frame.as_ref().unwrap().atom, atoms[0]);
        assert_eq!(entered.branch_returns[0].atom, atoms[0]);
        assert_eq!(entered.branch_returns[0].emitted_bonds.len(), 3);
    }
}

#[test]
fn explicit_inline_bond_completes_tetrahedral_parent_before_child_atom() {
    let (surface, atoms, bonds) = tetrahedral_star_fixture_with_tokens(
        4,
        vec![
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
        ],
    );
    let center = surface.tetrahedral_center(atoms[0]).unwrap();
    let mut parent = completed_atom_witnesses(&initial(&surface), atoms[0])
        .into_iter()
        .find(|(text, _)| text.contains('@') && !text.contains("@@"))
        .map(|(_, state)| state)
        .unwrap();
    for &bond in &bonds[..3] {
        let pending = ring_witness_choices(&parent)
            .into_iter()
            .find(|choice| {
                matches!(&choice.successor().pending,
                    Some(PendingEmission::BranchTraversalEmission(incident))
                        if incident.bond() == bond
                )
            })
            .unwrap()
            .into_successor();
        let child_atom = branch_event_atom(&pending).expect("branch event names its child AtomId");
        let (_, child) = only_choice(&pending, surface.atom_text(child_atom));
        parent = only_choice(&child, ")").1;
    }

    let (_, pending_atom) = only_choice(&parent, "=");
    assert_eq!(pending_atom.active_atom(), Some(atoms[4]));
    assert!(pending_atom.structural.atom_is_visited(atoms[4]));
    assert_eq!(
        parent.structural.active_local_bond_order().emitted_bonds,
        bonds[..3]
    );
    assert_eq!(
        pending_atom
            .structural
            .semantic_domain(center.order_variable),
        center.completed_order_domain(None, &bonds)
    );
    let (_, accepted) = only_choice(&pending_atom, "D");
    assert!(accepted.is_accepted());
}

#[test]
fn unresolved_tetrahedral_completion_is_typed_and_leaves_source_unchanged() {
    let (surface, atoms, bonds) = tetrahedral_star_fixture(4);
    let initial =
        NonStereoWriterState::<NonconformingNonProjectingTetrahedralSolver>::initial(&surface)
            .unwrap()
            .unwrap_consistent();
    let mut parent = completed_atom_witnesses(&initial, atoms[0])
        .into_iter()
        .find(|(text, _)| text.contains('@') && !text.contains("@@"))
        .map(|(_, state)| state)
        .unwrap();
    for &bond in &bonds[..3] {
        let pending = ring_witness_choices(&parent)
            .into_iter()
            .find(|choice| {
                matches!(&choice.successor().pending,
                    Some(PendingEmission::BranchTraversalEmission(incident))
                        if incident.bond() == bond
                )
            })
            .unwrap()
            .into_successor();
        let child_atom = branch_event_atom(&pending).expect("branch event identifies its child");
        let (_, child) = only_choice(&pending, surface.atom_text(child_atom));
        parent = only_choice(&child, ")").1;
    }
    let source = parent.observe_raw();

    assert!(matches!(
        parent.choices(),
        Err(ChoiceFailure::Invariant(
            WriterInvariantFailure::UnresolvedTetrahedralFrame { atom }
        )) if atom == atoms[0]
    ));
    assert_eq!(parent.observe_raw(), source);
}

#[test]
fn adjacent_tetrahedral_centers_resolve_independent_local_orders() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..8)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let shared = graph.add_bond(atoms[0], atoms[1]).unwrap();
    let left = (2..5)
        .map(|child| graph.add_bond(atoms[0], atoms[child]).unwrap())
        .collect::<Vec<_>>();
    let right = (5..8)
        .map(|child| graph.add_bond(atoms[1], atoms[child]).unwrap())
        .collect::<Vec<_>>();
    let mut atom_tokens = vec![
        PreparedAtomToken::Tetrahedral {
            reference_order: [shared, left[0], left[1], left[2]].map(TetrahedralLigand::Bond),
            text_by_parity: ["[C@]".to_owned(), "[C@@]".to_owned()],
        },
        PreparedAtomToken::Tetrahedral {
            reference_order: [shared, right[0], right[1], right[2]].map(TetrahedralLigand::Bond),
            text_by_parity: ["[C@]".to_owned(), "[C@@]".to_owned()],
        },
    ];
    atom_tokens.extend((2..8).map(|index| PreparedAtomToken::Fixed(format!("A{index}"))));
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        atom_tokens,
        vec![NonStereoBondToken::Elided; 7],
    )
    .unwrap();
    let left_center = surface.tetrahedral_center(atoms[0]).unwrap();
    let right_center = surface.tetrahedral_center(atoms[1]).unwrap();
    let root = completed_atom_witnesses(&initial(&surface), atoms[0])
        .into_iter()
        .find(|(text, _)| text.contains('@') && !text.contains("@@"))
        .map(|(_, state)| state)
        .unwrap();
    let terminals = reachable_terminal_states_with_bond_spelling(root, false);

    assert!(!terminals.is_empty());
    assert!(terminals.iter().all(|terminal| {
        terminal
            .structural
            .semantic_domain(left_center.order_variable)
            .is_singleton()
            && terminal
                .structural
                .semantic_domain(right_center.order_variable)
                .is_singleton()
    }));
    let right_orders = terminals
        .iter()
        .map(|terminal| resolved_tetrahedral_order(terminal, &surface, atoms[1]))
        .collect::<BTreeSet<_>>();
    assert!(right_orders
        .iter()
        .any(|value| independent_tetrahedral_order_is_even(&right_center.reference_order, *value)));
    assert!(right_orders.iter().any(|value| {
        !independent_tetrahedral_order_is_even(&right_center.reference_order, *value)
    }));
    for terminal in terminals {
        let order = independent_tetrahedral_order(
            &right_center.reference_order,
            resolved_tetrahedral_order(&terminal, &surface, atoms[1]),
        );
        assert_eq!(order[0], TetrahedralLigand::Bond(shared));
    }
}

#[test]
fn ring_capable_tetrahedral_center_activates_layout_at_its_atom_event() {
    let (surface, atoms, _) = ring_coupled_tetrahedral_fixture(NonStereoBondToken::Elided);
    let state = State::initial(&surface).unwrap().unwrap_consistent();
    let source = state.observe_raw();

    let choices = completed_atom_witnesses(&state, atoms[0]);
    assert_eq!(choices.len(), 2);
    let center = surface.tetrahedral_center(atoms[0]).unwrap();
    for (_, successor) in choices {
        let active_factors = &successor.observe_raw().structural.active_factors;
        assert!(successor
            .structural
            .factor_is_active(center.root_layout_factor));
        assert!(active_factors.contains(&center.root_layout_factor));
        assert_eq!(
            active_factors.len(),
            source.structural.active_factors.len() + 1,
            "the chosen layout is the only newly active factor"
        );
        assert!(center
            .entry_layout_factors
            .iter()
            .all(|(_, factor)| !successor.structural.factor_is_active(*factor)));
        assert!(!successor
            .structural
            .semantic_domain(center.role_pattern_variable)
            .is_empty());
    }
    assert_eq!(state.observe_raw(), source);
}

#[test]
fn prospective_frame_context_derives_exact_local_role_patterns() {
    let (surface, atoms, bonds) = tetrahedral_star_fixture(3);
    let state = initial(&surface);
    let center = surface.tetrahedral_center(atoms[0]).unwrap();
    let root_context = LocalLayoutContext {
        order: crate::traversal::LocalBondOrder {
            atom: atoms[0],
            entry_bond: None,
            emitted_bonds: Vec::new(),
            ring_occurrence_count: 0,
        },
        waiting_ring_bonds: Vec::new(),
        residual_attachment_bonds: vec![vec![bonds[0], bonds[1]], vec![bonds[2]]],
    };
    let entered_context = LocalLayoutContext {
        order: crate::traversal::LocalBondOrder {
            atom: atoms[0],
            entry_bond: Some(bonds[0]),
            emitted_bonds: Vec::new(),
            ring_occurrence_count: 0,
        },
        waiting_ring_bonds: vec![bonds[1]],
        residual_attachment_bonds: vec![vec![bonds[2]]],
    };

    assert_eq!(
        state.local_role_pattern_domain(center, &root_context),
        Domain::from_indices([1, 2]).unwrap()
    );
    assert_eq!(
        state.local_role_pattern_domain(center, &entered_context),
        Domain::singleton(2).unwrap()
    );
}

#[test]
fn every_small_event_context_matches_the_declarative_role_pattern_law() {
    for bond_count in [3, 4] {
        let (surface, atoms, bonds) = tetrahedral_star_fixture(bond_count);
        let state = initial(&surface);
        let center = surface.tetrahedral_center(atoms[0]).unwrap();
        let entry_options = std::iter::once(None).chain(bonds.iter().copied().map(Some));
        for entry in entry_options {
            let candidates = bonds
                .iter()
                .copied()
                .filter(|bond| Some(*bond) != entry)
                .collect::<Vec<_>>();
            for waiting_bits in 0..(1_usize << candidates.len()) {
                let waiting = candidates
                    .iter()
                    .enumerate()
                    .filter_map(|(index, bond)| (waiting_bits & (1 << index) != 0).then_some(*bond))
                    .collect::<Vec<_>>();
                let residual = candidates
                    .iter()
                    .copied()
                    .filter(|bond| !waiting.contains(bond))
                    .collect::<Vec<_>>();
                for attachments in set_partitions(&residual) {
                    let context = LocalLayoutContext {
                        order: crate::traversal::LocalBondOrder {
                            atom: atoms[0],
                            entry_bond: entry,
                            emitted_bonds: Vec::new(),
                            ring_occurrence_count: 0,
                        },
                        waiting_ring_bonds: waiting.clone(),
                        residual_attachment_bonds: attachments.clone(),
                    };
                    let expected =
                        Domain::from_indices((0..(1_u8 << bond_count)).filter(|pattern| {
                            let bit = |bond: BondId| center.pattern_bit(bond);
                            entry.is_none_or(|bond| pattern & (1 << bit(bond)) == 0)
                                && waiting.iter().all(|bond| pattern & (1 << bit(*bond)) != 0)
                                && attachments.iter().all(|group| {
                                    group
                                        .iter()
                                        .filter(|bond| pattern & (1 << bit(**bond)) == 0)
                                        .count()
                                        == 1
                                })
                        }))
                        .unwrap();
                    assert_eq!(
                        state.local_role_pattern_domain(center, &context),
                        expected,
                        "entry {entry:?}, waiting {waiting:?}, attachments {attachments:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn live_tetrahedral_context_matches_declarative_graph_recomputation() {
    let (surface, atoms, bonds) = ring_coupled_tetrahedral_fixture(NonStereoBondToken::Elided);
    let endpoints = [(0, 1), (0, 2), (1, 2), (0, 3)];
    let initial = initial(&surface);
    let prospective = initial.structural.prospective_root_layout_context(atoms[0]);
    assert_eq!(prospective.order.entry_bond, None);
    assert!(prospective.waiting_ring_bonds.is_empty());
    assert_eq!(
        prospective.residual_attachment_bonds,
        vec![vec![bonds[0], bonds[1]], vec![bonds[3]]]
    );

    let mut pending = vec![(initial, 0_usize)];
    let mut saw_root = false;
    let mut saw_entry = false;
    let mut saw_waiting_closure = false;
    let mut explored = 0;
    while let Some((state, atom_offset)) = pending.pop() {
        explored += 1;
        assert!(explored < 20_000);
        if state.pending.is_none() && state.active_atom() == Some(atoms[0]) {
            let observed = state.observe_raw();
            let context = state.structural.active_local_layout_context();
            let (waiting, mut groups) =
                independent_local_layout_groups(4, &endpoints, &observed, 0);
            let mut actual_groups = context.residual_attachment_bonds.clone();
            for group in &mut actual_groups {
                group.sort_unstable();
            }
            actual_groups.sort_unstable();
            groups.sort_unstable();
            assert_eq!(context.waiting_ring_bonds, waiting);
            assert_eq!(actual_groups, groups);
            saw_root |= context.order.entry_bond.is_none();
            saw_entry |= context.order.entry_bond.is_some();
            saw_waiting_closure |= !waiting.is_empty();
        }
        if saw_root && saw_entry && saw_waiting_closure {
            break;
        }
        pending.extend(declared_successors(&state, atom_offset));
    }
    assert!(saw_root && saw_entry && saw_waiting_closure);
}

#[test]
fn explicit_ring_endpoint_commits_local_order_before_pending_label() {
    let (surface, atoms, bonds) = ring_coupled_tetrahedral_fixture(NonStereoBondToken::Double);
    let source = initial(&surface);
    let pending = completed_atom_witnesses(&source, atoms[0])
        .into_iter()
        .flat_map(|(_, root)| ring_witness_choices(&root))
        .find(|choice| {
            choice.text() == "="
                && matches!(&choice.successor().pending,
                    Some(PendingEmission::RingOpeningLabel { incident, .. })
                        if incident.bond() == bonds[0]
                )
        })
        .unwrap()
        .into_successor();
    let before_label = pending.structural.active_local_bond_order();

    assert_eq!(before_label.emitted_bonds, vec![bonds[0]]);
    assert_eq!(before_label.ring_occurrence_count, 1);
    let after_label = ring_witness_choices(&pending).remove(0).into_successor();
    assert_eq!(
        after_label.structural.active_local_bond_order(),
        before_label
    );
}

#[test]
fn selected_tetrahedral_ring_order_controls_open_and_close_choices() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms: [AtomId; 4] = std::array::from_fn(|_| graph.add_atom().unwrap());
    let bonds = [
        graph.add_bond(atoms[0], atoms[1]).unwrap(),
        graph.add_bond(atoms[1], atoms[2]).unwrap(),
        graph.add_bond(atoms[0], atoms[2]).unwrap(),
        graph.add_bond(atoms[2], atoms[3]).unwrap(),
    ];
    let reference = [
        TetrahedralLigand::Bond(bonds[1]),
        TetrahedralLigand::Bond(bonds[2]),
        TetrahedralLigand::Bond(bonds[3]),
        TetrahedralLigand::VirtualHydrogen,
    ];
    let surface = PreparedNonStereo::with_atom_tokens_and_directional_for_writer(
        PreparedMolecule::new(graph.build()),
        vec![
            PreparedAtomToken::Fixed("A".to_owned()),
            PreparedAtomToken::Fixed("B".to_owned()),
            PreparedAtomToken::Tetrahedral {
                reference_order: reference,
                text_by_parity: ["[C@H]".to_owned(), "[C@@H]".to_owned()],
            },
            PreparedAtomToken::Fixed("D".to_owned()),
        ],
        vec![NonStereoBondToken::Elided; bonds.len()],
        Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder },
        true,
    ).unwrap();
    let molecule = surface.molecule();
    let roles = [false, false, true, true];
    let restrictions = bonds.iter().copied().zip(roles).map(|(bond, ring)| {
        let partition = molecule.bond_role_partition(bond).unwrap();
        (molecule.bond_decision_variable(bond).unwrap(),
            if ring { partition.ring_values() } else { partition.traversal_values() })
    }).collect::<Vec<_>>();
    let mut initial = initial(&surface);
    initial.structural = initial.structural.transitioned_semantics(&restrictions, &[])
        .unwrap().unwrap_consistent();
    let (_, rooted) = only_root_atom(&initial, atoms[0]);
    let openings = ring_witness_choices(&rooted).into_iter().filter(|choice| {
        choice.text() == "1" && matches!(choice.successor().observe_raw().structural.traversal.bond_progress[bonds[2].index()],
            ObservedBondProgress::RingOpen { first_endpoint } if first_endpoint == atoms[0])
    }).collect::<Vec<_>>();
    assert_eq!(openings.len(), 2, "both omitted-opening representations retain their writer histories");
    let center = surface.tetrahedral_center(atoms[2]).unwrap();
    let mut saw_first = BTreeSet::new();
    let mut completed_local_orders = 0;
    for opening in openings {
        let opened = opening.into_successor();
        let (_, at_b) = only_choice(&opened, "B");
        let entered = completed_atom_witnesses(&at_b, atoms[2]);
        assert_eq!(entered.len(), 2);
        for (marker, state) in entered {
        assert_eq!(state.structural.active_local_bond_order().entry_bond, Some(bonds[1]));
        assert_eq!(state.structural.active_local_bond_order().emitted_bonds, Vec::<BondId>::new());
        let context = state.structural.active_local_layout_context();
        assert_eq!(context.waiting_ring_bonds, vec![bonds[2]]);
        let close_first = [
            TetrahedralLigand::Bond(bonds[1]),
            TetrahedralLigand::VirtualHydrogen,
            TetrahedralLigand::Bond(bonds[2]),
            TetrahedralLigand::Bond(bonds[3]),
        ];
        let open_first = [close_first[0], close_first[1], close_first[3], close_first[2]];
        let order = if independent_tetrahedral_order_is_even(&reference,
            (0..24).find(|value| independent_tetrahedral_order(&reference, *value) == close_first).unwrap())
            == (marker == "[C@H]") { close_first } else { open_first };
        let expected = Domain::from_indices((0..24).filter(|value| {
            independent_tetrahedral_order(&reference, *value) == order
                && independent_tetrahedral_order_is_even(&reference, *value) == (marker == "[C@H]")
        })).unwrap();
        assert!(!expected.is_empty(), "each marker must select one ring order");
        assert_eq!(state.structural.semantic_domain(center.order_variable), expected);
        let first_bond = match order[2] { TetrahedralLigand::Bond(bond) => bond, _ => unreachable!() };
        let second_bond = match order[3] { TetrahedralLigand::Bond(bond) => bond, _ => unreachable!() };
        saw_first.insert(first_bond);
        let first_choices = ring_witness_choices(&state);
        assert!(!first_choices.is_empty(), "the selected first ring event remains available");
        for first_choice in first_choices {
            let mut first = first_choice.into_successor();
            assert_eq!(first.structural.active_local_bond_order().emitted_bonds, vec![first_bond]);
            assert_eq!(first.structural.semantic_domain(center.order_variable), expected);
            if first.pending.is_some() {
                let labels = ring_witness_choices(&first);
                assert_eq!(labels.len(), 1);
                first = labels.into_iter().next().unwrap().into_successor();
            }
            let second_choices = ring_witness_choices(&first);
            assert!(!second_choices.is_empty(), "the other ring event remains available");
            for choice in second_choices {
            let mut second = choice.into_successor();
            assert_eq!(second.structural.active_local_bond_order().emitted_bonds, vec![first_bond, second_bond]);
            assert_eq!(second.structural.semantic_domain(center.order_variable), expected);
            if second.pending.is_some() {
                let labels = ring_witness_choices(&second);
                assert_eq!(labels.len(), 1);
                second = labels.into_iter().next().unwrap().into_successor();
            }
            assert_eq!(second.structural.active_local_bond_order().emitted_bonds, vec![first_bond, second_bond]);
            assert_eq!(second.structural.semantic_domain(center.order_variable), expected);
            assert_eq!(second.pending, None);
            assert_eq!(second.structural.active_local_bond_order().ring_occurrence_count, 2);
            let pattern = (1_u8 << center.pattern_bit(bonds[2])) | (1_u8 << center.pattern_bit(bonds[3]));
            assert_eq!(second.structural.semantic_domain(center.role_pattern_variable),
                Domain::singleton(pattern).unwrap());
            assert!(second.structural.factor_is_active(center.layout_factor(Some(bonds[1]))));
            for (bond, ring) in bonds.iter().copied().zip(roles) {
                if bond == bonds[0] { continue; }
                let partition = molecule.bond_role_partition(bond).unwrap();
                let expected_role = if ring { partition.ring_values() } else { partition.traversal_values() };
                assert!(second.structural.semantic_domain(molecule.bond_decision_variable(bond).unwrap())
                    .is_subset_of(expected_role));
            }
            let departures = ring_witness_choices(&second);
            assert!(!departures.is_empty(), "completed center can advance through its actual choice path");
            assert!(departures.iter().all(|choice| choice.successor().active_atom() != Some(atoms[2])));
            completed_local_orders += 1;
            }
        }
        }
    }
    assert_eq!(saw_first, BTreeSet::from([bonds[2], bonds[3]]));
    assert!(completed_local_orders >= 2);
}

fn independent_ring_phase_feasible(order: &[usize], ring_mask: u8, waiting_mask: u8, free: usize) -> bool {
    let mut available = free;
    for &incidence in order {
        if ring_mask & (1 << incidence) == 0 { continue; }
        if waiting_mask & (1 << incidence) != 0 {
            available += 1;
        } else if available == 0 {
            return false;
        } else {
            available -= 1;
        }
    }
    true
}

#[test]
fn finite_ring_phase_reference_matrix_distinguishes_order_from_count() {
    let mut checked = 0;
    let mut count_only_false_positives = 0;
    for degree in [3_usize, 4] {
        let permutations = (0..24).map(|value| {
            let reference = [0_usize, 1, 2, 3];
            independent_tetrahedral_order(&reference.map(|index| TetrahedralLigand::Bond(BondId::new(index as u32))), value)
                .map(|ligand| match ligand { TetrahedralLigand::Bond(bond) => bond.index(), _ => unreachable!() })
        }).map(|order| order.into_iter().filter(|incidence| *incidence < degree).collect::<Vec<_>>())
            .collect::<BTreeSet<_>>();
        for order in permutations {
            for ring_mask in 0..(1_u8 << degree) {
                for waiting_mask in 0..(1_u8 << degree) {
                    if waiting_mask & !ring_mask != 0 { continue; }
                    for free in 0..=degree {
                        let feasible = independent_ring_phase_feasible(&order, ring_mask, waiting_mask, free);
                        let count_bound = free + 2 * waiting_mask.count_ones() as usize;
                        let count_allows = ring_mask.count_ones() as usize <= count_bound;
                        assert!(!feasible || count_allows, "count remains necessary");
                        count_only_false_positives += usize::from(count_allows && !feasible);
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 1000);
    assert!(count_only_false_positives > 0);
    assert!(!independent_ring_phase_feasible(&[2, 3, 0, 1], 0b1111, 0b0011, 1));
    assert!(independent_ring_phase_feasible(&[0, 2, 3, 1], 0b1111, 0b0011, 1));
    assert!(4 <= 1 + 2 * 2, "degree-four count bound is vacuous in this case");
}

#[test]
fn ordered_ring_phase_native_projection_matches_independent_finite_relation() {
    use crate::model::{ConstraintModelBuilder, TetrahedralLayoutBond};
    for degree in [3_usize, 4] {
        let mut builder = ConstraintModelBuilder::new();
        let bond_ids = (0..degree).map(|index| BondId::new(index as u32)).collect::<Vec<_>>();
        let roles = (0..degree).map(|_| builder.add_variable(BondRole::role_domain()).unwrap())
            .collect::<Vec<_>>();
        let mut reference = bond_ids.iter().copied().map(TetrahedralLigand::Bond).collect::<Vec<_>>();
        if degree == 3 { reference.push(TetrahedralLigand::VirtualHydrogen); }
        let reference: [TetrahedralLigand; 4] = reference.try_into().unwrap();
        let order = builder.add_variable(full_order_domain()).unwrap();
        let pattern = builder.add_variable(full_role_pattern_domain(degree)).unwrap();
        let waiting = builder.add_variable(Domain::from_indices(0..(1_u8 << degree)).unwrap()).unwrap();
        let free_bits = (0..3).map(|_| builder.add_variable(Domain::from_bits(3)).unwrap())
            .collect::<Vec<_>>();
        let bond_bits = bond_ids.iter().copied().enumerate().map(|(bit, bond)| (bond, bit as u8))
            .collect::<Vec<_>>();
        let rows = layout_order_rows(&reference, None, &bond_bits);
        let layout = builder.add_latent_tetrahedral_layout(order, pattern,
            roles.iter().copied().enumerate().map(|(bit, variable)|
                TetrahedralLayoutBond::new(variable, EdgeRolePartition::bond_role(), bit as u8)),
            rows.iter().copied()).unwrap();
        let resource = builder.add_latent_ordered_ring_phase(
            bond_ids.iter().copied().zip(roles.iter().copied()).map(|(bond, variable)|
                (bond, SpanningTreeEdge::new(variable, AtomId::new(0), AtomId::new(bond.index() as u32 + 1)))),
            reference, order, waiting, free_bits.iter().copied());
        let solver = <NativeSolverState as ConstraintSolver>::initial(Arc::new(builder.build()))
            .unwrap().unwrap_consistent();
        for role_pattern in 0..(1_u8 << degree) {
            for waiting_mask in 0..(1_u8 << degree) {
                if waiting_mask & !role_pattern != 0 { continue; }
                for free in 0..=degree {
                    for parity in TetrahedralParity::ALL {
                        let mut restrictions = vec![
                            (pattern, Domain::singleton(role_pattern).unwrap()),
                            (waiting, Domain::singleton(waiting_mask).unwrap()),
                            (order, parity_domain(parity)),
                        ];
                        restrictions.extend(roles.iter().copied().enumerate().map(|(bit, variable)|
                            (variable, if role_pattern & (1 << bit) != 0 {
                                BondRole::Ring.singleton_domain()
                            } else { BondRole::Traversal.singleton_domain() })));
                        restrictions.extend(free_bits.iter().copied().enumerate().map(|(bit, variable)|
                            (variable, Domain::singleton(((free >> bit) & 1) as u8).unwrap())));
                        let expected = Domain::from_indices(rows[role_pattern as usize]
                            .intersect(parity_domain(parity)).iter().filter(|value| {
                                let ligand_order = independent_tetrahedral_order(&reference, *value);
                                let incidence_order = ligand_order.iter().filter_map(|ligand| match ligand {
                                    TetrahedralLigand::Bond(bond) => Some(bond.index()),
                                    TetrahedralLigand::VirtualHydrogen => None,
                                }).collect::<Vec<_>>();
                                independent_ring_phase_feasible(&incidence_order, role_pattern, waiting_mask, free)
                            })).unwrap();
                        let actual = solver.transitioned(&restrictions, &[layout, resource]).unwrap();
                        if expected.is_empty() {
                            assert!(matches!(actual, Consistency::Contradiction),
                                "degree={degree} roles={role_pattern} waiting={waiting_mask} free={free} parity={parity:?}");
                        } else {
                            assert_eq!(actual.unwrap_consistent().domain_bound(order), Some(expected),
                                "degree={degree} roles={role_pattern} waiting={waiting_mask} free={free} parity={parity:?}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn ordered_ring_phase_joint_layout_projection_matches_exhaustive_and_literal_balance() {
    use crate::model::{ConstraintModelBuilder, TetrahedralLayoutBond};
    for (degree, waiting_mask, free, parity) in [
        (3_usize, 0b010_u8, 0_usize, TetrahedralParity::Even),
        (3, 0b010, 0, TetrahedralParity::Odd),
        (4, 0b0011, 1, TetrahedralParity::Even),
        (4, 0b0011, 1, TetrahedralParity::Odd),
    ] {
        let mut builder = ConstraintModelBuilder::new();
        let ids = (0..degree).map(|index| BondId::new(index as u32)).collect::<Vec<_>>();
        let roles = (0..degree).map(|_| builder.add_variable(BondRole::role_domain()).unwrap())
            .collect::<Vec<_>>();
        let mut reference = ids.iter().copied().map(TetrahedralLigand::Bond).collect::<Vec<_>>();
        if degree == 3 { reference.push(TetrahedralLigand::VirtualHydrogen); }
        let reference: [TetrahedralLigand; 4] = reference.try_into().unwrap();
        let order = builder.add_variable(full_order_domain()).unwrap();
        let pattern = builder.add_variable(full_role_pattern_domain(degree)).unwrap();
        let waiting = builder.add_variable(Domain::singleton(waiting_mask).unwrap()).unwrap();
        let free_bits = (0..3).map(|bit| builder.add_variable(
            Domain::singleton(((free >> bit) & 1) as u8).unwrap()).unwrap()).collect::<Vec<_>>();
        let bond_bits = ids.iter().copied().enumerate().map(|(bit, bond)| (bond, bit as u8))
            .collect::<Vec<_>>();
        let rows = layout_order_rows(&reference, None, &bond_bits);
        let layout = builder.add_latent_tetrahedral_layout(order, pattern,
            roles.iter().copied().enumerate().map(|(bit, variable)|
                TetrahedralLayoutBond::new(variable, EdgeRolePartition::bond_role(), bit as u8)),
            rows.iter().copied()).unwrap();
        let resource = builder.add_latent_ordered_ring_phase(
            ids.iter().copied().zip(roles.iter().copied()).map(|(bond, variable)|
                (bond, SpanningTreeEdge::new(variable, AtomId::new(0), AtomId::new(bond.index() as u32 + 1)))),
            reference, order, waiting, free_bits);
        if degree == 3 {
            let cross = builder.add_variable(BondRole::role_domain()).unwrap();
            builder.add_spanning_tree((0..=degree).map(|index| AtomId::new(index as u32)),
                roles.iter().copied().enumerate().map(|(index, variable)|
                    SpanningTreeEdge::new(variable, AtomId::new(0), AtomId::new(index as u32 + 1)))
                    .chain([SpanningTreeEdge::new(cross, AtomId::new(1), AtomId::new(2))])).unwrap();
        }
        let model = Arc::new(builder.build());
        let native = <NativeSolverState as ConstraintSolver>::initial(model.clone()).unwrap().unwrap_consistent();
        let oracle = <ExhaustiveSolverState as ConstraintSolver>::initial(model).unwrap().unwrap_consistent();
        let restrictions = [(order, parity_domain(parity))];
        let native = native.transitioned(&restrictions, &[layout, resource]).unwrap();
        let oracle = oracle.transitioned(&restrictions, &[layout, resource]).unwrap();
        let mut expected_pairs = BTreeSet::new();
        for role_pattern in 0..(1_u8 << degree) {
            if waiting_mask & !role_pattern != 0 { continue; }
            // The degree-three graph is a center star plus the 1-2 cross edge.
            // With bond 1 waiting, its tree has exactly that center bond as Ring.
            if degree == 3 && role_pattern != 0b010 { continue; }
            for value in rows[role_pattern as usize].intersect(parity_domain(parity)).iter() {
                let ligand_order = independent_tetrahedral_order(&reference, value);
                let incidence_order = ligand_order.iter().filter_map(|ligand| match ligand {
                    TetrahedralLigand::Bond(bond) => Some(bond.index()),
                    TetrahedralLigand::VirtualHydrogen => None,
                }).collect::<Vec<_>>();
                if independent_ring_phase_feasible(&incidence_order, role_pattern, waiting_mask, free) {
                    expected_pairs.insert((value, role_pattern));
                }
            }
        }
        assert!(!expected_pairs.is_empty());
        let native = native.unwrap_consistent();
        let oracle = oracle.unwrap_consistent();
        let expected_orders = Domain::from_indices(expected_pairs.iter().map(|(value, _)| *value)).unwrap();
        let expected_patterns = Domain::from_indices(expected_pairs.iter().map(|(_, value)| *value)).unwrap();
        let native_domain = |variable| native.domain_bound(variable);
        let oracle_domain = |variable| oracle.domain_bound(variable);
        for domain in [&native_domain as &dyn Fn(VariableId) -> Option<Domain>, &oracle_domain] {
            assert_eq!(domain(order), Some(expected_orders));
            assert_eq!(domain(pattern), Some(expected_patterns));
            for (bit, variable) in roles.iter().copied().enumerate() {
                let expected = Domain::from_indices(expected_pairs.iter().map(|(_, pattern)|
                    u8::from(pattern & (1 << bit) != 0))).unwrap();
                assert_eq!(domain(variable), Some(expected));
            }
        }
        for value in 0..24 {
            for role_pattern in 0..(1_u8 << degree) {
                let pair = [(order, Domain::singleton(value).unwrap()),
                    (pattern, Domain::singleton(role_pattern).unwrap())];
                let actual = native.restricted(&pair).unwrap();
                assert_eq!(matches!(actual, Consistency::Consistent(_)),
                    expected_pairs.contains(&(value, role_pattern)),
                    "degree={degree} waiting={waiting_mask} free={free} parity={parity:?} order={value} pattern={role_pattern}");
            }
        }
    }
}

#[test]
fn ordered_ring_phase_projects_unfixed_waiting_and_free_domains() {
    use crate::model::{ConstraintModelBuilder, TetrahedralLayoutBond};
    let mut builder = ConstraintModelBuilder::new();
    let ids = [BondId::new(0), BondId::new(1), BondId::new(2)];
    let roles = std::array::from_fn::<_, 3, _>(|_| builder.add_variable(BondRole::role_domain()).unwrap());
    let reference = [TetrahedralLigand::Bond(ids[0]), TetrahedralLigand::Bond(ids[1]),
        TetrahedralLigand::Bond(ids[2]), TetrahedralLigand::VirtualHydrogen];
    let order = builder.add_variable(full_order_domain()).unwrap();
    let pattern = builder.add_variable(full_role_pattern_domain(3)).unwrap();
    let waiting = builder.add_variable(Domain::from_indices([0, 1]).unwrap()).unwrap();
    let free = builder.add_variable(Domain::from_bits(3)).unwrap();
    let bits = [(ids[0], 0_u8), (ids[1], 1), (ids[2], 2)];
    let rows = layout_order_rows(&reference, None, &bits);
    let layout = builder.add_latent_tetrahedral_layout(order, pattern,
        roles.iter().copied().enumerate().map(|(bit, variable)|
            TetrahedralLayoutBond::new(variable, EdgeRolePartition::bond_role(), bit as u8)),
        rows.iter().copied()).unwrap();
    let resource = builder.add_latent_ordered_ring_phase(
        ids.into_iter().zip(roles).map(|(bond, variable)|
            (bond, SpanningTreeEdge::new(variable, AtomId::new(0), AtomId::new(bond.index() as u32 + 1)))),
        reference, order, waiting, [free]);
    let model = Arc::new(builder.build());
    let native = <NativeSolverState as ConstraintSolver>::initial(model.clone()).unwrap().unwrap_consistent();
    let oracle = <ExhaustiveSolverState as ConstraintSolver>::initial(model).unwrap().unwrap_consistent();
    let mut restrictions = vec![(pattern, Domain::singleton(0b111).unwrap())];
    restrictions.extend(roles.map(|variable| (variable, BondRole::Ring.singleton_domain())));
    let native = native.transitioned(&restrictions, &[layout, resource]).unwrap().unwrap_consistent();
    let oracle = oracle.transitioned(&restrictions, &[layout, resource]).unwrap().unwrap_consistent();
    let expected = (0..24).flat_map(|value| (0..=1).flat_map(move |mask| (0..=1).map(move |budget|
        (value, mask, budget as u8)))).filter(|(value, mask, budget)| {
        let ligand_order = independent_tetrahedral_order(&reference, *value);
        let indices = ligand_order.iter().filter_map(|ligand| match ligand {
            TetrahedralLigand::Bond(bond) => Some(bond.index()),
            TetrahedralLigand::VirtualHydrogen => None,
        }).collect::<Vec<_>>();
        rows[7].contains(*value) && independent_ring_phase_feasible(&indices, 0b111, *mask, usize::from(*budget))
    }).collect::<BTreeSet<_>>();
    assert!(!expected.is_empty());
    let expected_order = Domain::from_indices(expected.iter().map(|(value, _, _)| *value)).unwrap();
    let expected_waiting = Domain::from_indices(expected.iter().map(|(_, mask, _)| *mask)).unwrap();
    let expected_free = Domain::from_indices(expected.iter().map(|(_, _, free)| *free)).unwrap();
    for (variable, projected) in [(order, expected_order), (waiting, expected_waiting), (free, expected_free)] {
        assert_eq!(native.domain_bound(variable), Some(projected));
        assert_eq!(oracle.domain_bound(variable), Some(projected));
    }
    for value in 0..24 {
        for mask in 0..=1 {
            for budget in 0..=1 {
                let tuple = [(order, Domain::singleton(value).unwrap()),
                    (waiting, Domain::singleton(mask).unwrap()),
                    (free, Domain::singleton(budget).unwrap())];
                let permitted = expected.contains(&(value, mask, budget));
                assert_eq!(matches!(native.restricted(&tuple).unwrap(), Consistency::Consistent(_)), permitted);
                assert_eq!(matches!(oracle.restricted(&tuple).unwrap(), Consistency::Consistent(_)), permitted);
            }
        }
    }
}

#[test]
fn zero_free_label_prunes_tetra_marker_that_opens_before_waiting_closure() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms: [AtomId; 4] = std::array::from_fn(|_| graph.add_atom().unwrap());
    let bonds = [
        graph.add_bond(atoms[0], atoms[1]).unwrap(),
        graph.add_bond(atoms[1], atoms[2]).unwrap(),
        graph.add_bond(atoms[0], atoms[2]).unwrap(),
        graph.add_bond(atoms[2], atoms[3]).unwrap(),
    ];
    let reference = [
        TetrahedralLigand::Bond(bonds[1]),
        TetrahedralLigand::Bond(bonds[2]),
        TetrahedralLigand::Bond(bonds[3]),
        TetrahedralLigand::VirtualHydrogen,
    ];
    let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(graph.build()),
        vec![
            PreparedAtomToken::Fixed("O".to_owned()),
            PreparedAtomToken::Fixed("C".to_owned()),
            PreparedAtomToken::Tetrahedral {
                reference_order: reference,
                text_by_parity: ["[C@H]".to_owned(), "[C@@H]".to_owned()],
            },
            PreparedAtomToken::Fixed("F".to_owned()),
        ],
        vec![NonStereoBondToken::Elided; bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, 0,
    ).unwrap();
    let start = initial(&surface);
    let (_, rooted) = only_root_atom(&start, atoms[0]);
    let opened = rooted.choices().unwrap().into_iter().find(|choice| {
        choice.text() == "0" && matches!(choice.successor().observe_raw().structural.traversal.bond_progress[bonds[2].index()],
            ObservedBondProgress::RingOpen { first_endpoint } if first_endpoint == atoms[0])
    }).expect("A-C can open the only label").into_successor();
    let (_, at_b) = only_choice(&opened, "C");
    let entered = completed_atom_witnesses(&at_b, atoms[2]);
    assert_eq!(opened.labels.bonds_by_slot.len(), 1);
    assert_eq!(opened.labels.bonds_by_slot.values().copied().collect::<Vec<_>>(), vec![bonds[2]]);
    assert!(entered.iter().all(|(_, state)| state.structural.active_local_layout_context().waiting_ring_bonds == vec![bonds[2]]));
    let close_first = [reference[0], reference[3], reference[1], reference[2]];
    let expected_marker = if independent_tetrahedral_order_is_even(&reference,
        (0..24).find(|value| independent_tetrahedral_order(&reference, *value) == close_first).unwrap()) {
        "[C@H]"
    } else {
        "[C@@H]"
    };
    let close_first_state = entered.iter().find(|(marker, _)| marker == expected_marker)
        .expect("close-first marker must be available before pruning").1.clone();
    let center = surface.tetrahedral_center(atoms[2]).unwrap();
    let selected = Domain::from_indices((0..24).filter(|value|
        independent_tetrahedral_order(&reference, *value) == close_first)).unwrap();
    assert_eq!(close_first_state.structural.semantic_domain(center.order_variable), selected);
    assert!(close_first_state.choices().unwrap().iter().any(|choice| {
        matches!(choice.successor().observe_raw().structural.traversal.bond_progress[bonds[2].index()],
            ObservedBondProgress::RingClosed { .. })
    }), "the surviving marker can close its waiting ring first");
    let mut frontier = vec![close_first_state];
    let mut visited = 0;
    let mut accepted = false;
    while let Some(state) = frontier.pop() {
        visited += 1;
        assert!(visited <= 5000, "close-first completion exceeded finite test budget");
        if state.is_accepted() { accepted = true; break; }
        if let Ok(choices) = state.choices() {
            frontier.extend(choices.into_iter().map(Choice::into_successor));
        }
    }
    assert!(accepted, "close-first marker must have an accepted continuation");
    assert_eq!(entered.iter().map(|(marker, _)| marker.as_str()).collect::<BTreeSet<_>>(),
        BTreeSet::from([expected_marker]),
        "with no free label, closing A-C must precede opening C-F at atom entry");
}

#[test]
fn every_ring_endpoint_path_commits_one_local_occurrence() {
    let (surface, atoms, bonds) = ring_coupled_tetrahedral_fixture(NonStereoBondToken::Double);
    let mut pending = vec![(initial(&surface), 0_usize)];
    let mut saw = [false; 4];
    let mut explored = 0;

    while let Some((state, atom_offset)) = pending.pop() {
        explored += 1;
        assert!(explored < 20_000);
        let source = state.observe_raw();
        let Some(source_frame) = source.structural.traversal.active_frame.as_ref() else {
            for choice in ring_witness_choices(&state) {
                if !selected_atom_spelling_transition(&state, &choice, atom_offset) {
                    continue;
                }
                let token = choice.text().to_owned();
                let successor = choice.into_successor();
                let next_offset =
                    next_atom_spelling_offset(&state, &successor, atom_offset, &token);
                pending.push((successor, next_offset));
            }
            continue;
        };
        for choice in ring_witness_choices(&state) {
            if !selected_atom_spelling_transition(&state, &choice, atom_offset) {
                continue;
            }
            let successor = choice.successor().observe_raw();
            let source_progress = &source.structural.traversal.bond_progress[bonds[0].index()];
            let successor_progress =
                &successor.structural.traversal.bond_progress[bonds[0].index()];
            let event = match (source_progress, successor_progress) {
                (
                    ObservedBondProgress::Unrepresented,
                    ObservedBondProgress::RingOpen { first_endpoint },
                ) if *first_endpoint == atoms[0] && source_frame.atom == atoms[0] => {
                    Some((0, choice.text() == "="))
                }
                (
                    ObservedBondProgress::RingOpen { first_endpoint },
                    ObservedBondProgress::RingClosed {
                        first_endpoint: closed_first,
                        second_endpoint,
                    },
                ) if *first_endpoint == *closed_first
                    && *second_endpoint == atoms[0]
                    && source_frame.atom == atoms[0] =>
                {
                    Some((2, choice.text() == "="))
                }
                _ => None,
            };
            if let Some((base, explicit)) = event {
                let successor_frame = successor
                    .structural
                    .traversal
                    .active_frame
                    .as_ref()
                    .expect("ring endpoint must retain the active tetrahedral frame");
                assert_eq!(
                    successor_frame.emitted_bonds.len(),
                    source_frame.emitted_bonds.len() + 1
                );
                assert_eq!(successor_frame.emitted_bonds.last(), Some(&bonds[0]));
                assert_eq!(
                    successor_frame.ring_occurrence_count,
                    source_frame.ring_occurrence_count + 1
                );
                saw[base + usize::from(explicit)] = true;

                if explicit {
                    let pending_state = choice.successor().clone();
                    let after_label = ring_witness_choices(&pending_state)
                        .into_iter()
                        .next()
                        .unwrap()
                        .into_successor()
                        .observe_raw();
                    assert_eq!(
                        after_label.structural.traversal.active_frame,
                        successor.structural.traversal.active_frame,
                        "a pending label must not recommit the ring occurrence"
                    );
                }
            }
            let token = choice.text().to_owned();
            let successor = choice.into_successor();
            let next_offset = next_atom_spelling_offset(&state, &successor, atom_offset, &token);
            pending.push((successor, next_offset));
        }
        if saw.iter().all(|value| *value) {
            break;
        }
    }

    assert_eq!(saw, [true; 4], "opening/closure emit/omit paths must occur");
}

#[test]
fn ring_coupled_tetrahedral_center_has_complete_online_walks() {
    let (surface, atoms, _) = ring_coupled_tetrahedral_fixture(NonStereoBondToken::Elided);

    let center = surface.tetrahedral_center(atoms[0]).unwrap();
    let mut pending = vec![(initial(&surface), String::new(), 0_usize)];
    let mut paths = Vec::new();
    let mut saw_entered_context = false;
    let mut saw_waiting_closure = false;
    let mut explored = 0;
    while let Some((state, prefix, atom_offset)) = pending.pop() {
        explored += 1;
        assert!(explored < 20_000);
        if state.is_accepted() {
            paths.push(surface.normalize_test_atom_classes(&prefix));
            continue;
        }
        if state.pending.is_none() && state.active_atom() == Some(atoms[0]) {
            let context = state.structural.active_local_layout_context();
            if let Some(entry) = context.order.entry_bond {
                saw_entered_context = true;
                assert!(state
                    .structural
                    .factor_is_active(center.layout_factor(Some(entry))));
            }
            saw_waiting_closure |= !context.waiting_ring_bonds.is_empty();
        }
        let choices = ring_witness_choices(&state);
        assert!(
            !choices.is_empty(),
            "nonaccepted ring-coupled tetrahedral state has no continuation"
        );
        for choice in choices {
            if !selected_atom_spelling_transition(&state, &choice, atom_offset) {
                continue;
            }
            let token = choice.text().to_owned();
            let successor = choice.into_successor();
            let next_offset = next_atom_spelling_offset(&state, &successor, atom_offset, &token);
            pending.push((successor, format!("{prefix}{token}"), next_offset));
        }
    }

    assert!(!paths.is_empty());
    assert!(paths.iter().any(|text| text.contains('1')));
    assert!(saw_entered_context);
    assert!(saw_waiting_closure);
}

#[test]
fn adjacent_ring_coupled_centers_activate_and_resolve_in_one_component() {
    let (surface, atoms, _) = adjacent_ring_coupled_tetrahedral_fixture();
    let centers = [
        surface.tetrahedral_center(atoms[0]).unwrap(),
        surface.tetrahedral_center(atoms[1]).unwrap(),
    ];
    let factor_ids = centers.map(|center| {
        std::iter::once(center.root_layout_factor)
            .chain(
                center
                    .entry_layout_factors
                    .iter()
                    .map(|(_, factor)| *factor),
            )
            .collect::<Vec<_>>()
    });
    let mut pending = vec![(initial(&surface), 0_usize)];
    let mut saw_both_active = false;
    let mut saw_one_suspended_while_the_other_was_active = false;
    let mut accepted = 0;
    let mut explored = 0;

    while let Some((state, atom_offset)) = pending.pop() {
        explored += 1;
        assert!(explored < 100_000);
        let observed = state.observe_raw();
        let center_is_active = factor_ids.each_ref().map(|ids| {
            ids.iter()
                .any(|factor| observed.structural.active_factors.contains(factor))
        });
        saw_both_active |= center_is_active == [true, true];
        if let Some(active) = observed
            .structural
            .traversal
            .active_frame
            .as_ref()
            .map(|frame| frame.atom)
        {
            let suspended = observed
                .structural
                .traversal
                .branch_returns
                .iter()
                .map(|frame| frame.atom)
                .collect::<BTreeSet<_>>();
            saw_one_suspended_while_the_other_was_active |= (active == atoms[0]
                && suspended.contains(&atoms[1]))
                || (active == atoms[1] && suspended.contains(&atoms[0]));
        }
        if state.is_accepted() {
            accepted += 1;
            for center in centers {
                assert!(state
                    .structural
                    .semantic_domain(center.order_variable)
                    .is_singleton());
                assert!(state
                    .structural
                    .semantic_domain(center.role_pattern_variable)
                    .is_singleton());
            }
            if saw_both_active && saw_one_suspended_while_the_other_was_active {
                break;
            }
            continue;
        }
        pending.extend(declared_successors(&state, atom_offset));
    }

    assert!(accepted > 0);
    assert!(saw_both_active);
    assert!(saw_one_suspended_while_the_other_was_active);
}

#[test]
fn disconnected_ring_coupled_centers_keep_factors_local_and_reuse_labels() {
    let (surface, atoms) = disconnected_ring_coupled_tetrahedral_fixture();
    let centers = [
        surface.tetrahedral_center(atoms[0]).unwrap(),
        surface.tetrahedral_center(atoms[4]).unwrap(),
    ];
    let mut pending = vec![(initial(&surface), String::new(), 0_usize)];
    let mut accepted = None;
    let mut explored = 0;
    while let Some((state, text, atom_offset)) = pending.pop() {
        explored += 1;
        assert!(explored < 100_000);
        if state.is_accepted() {
            accepted = Some((state, surface.normalize_test_atom_classes(&text)));
            break;
        }
        for choice in ring_witness_choices(&state) {
            if !selected_atom_spelling_transition(&state, &choice, atom_offset) {
                continue;
            }
            let token = choice.text().to_owned();
            let successor = choice.into_successor();
            let next_offset = next_atom_spelling_offset(&state, &successor, atom_offset, &token);
            pending.push((successor, format!("{text}{token}"), next_offset));
        }
    }
    let (accepted, text) = accepted.expect("disconnected cyclic fixture must have support");
    let components = text.split('.').collect::<Vec<_>>();
    assert_eq!(components.len(), 2);
    assert!(components.iter().all(|component| component.contains('1')));
    assert!(accepted.labels.is_clean());
    for center in centers {
        assert!(accepted
            .structural
            .semantic_domain(center.order_variable)
            .is_singleton());
        assert!(accepted
            .structural
            .semantic_domain(center.role_pattern_variable)
            .is_singleton());
        assert!(std::iter::once(center.root_layout_factor)
            .chain(
                center
                    .entry_layout_factors
                    .iter()
                    .map(|(_, factor)| *factor)
            )
            .any(|factor| accepted.structural.factor_is_active(factor)));
    }
}

#[test]
fn disconnected_ring_coupled_suffix_resolves_after_first_component() {
    let (surface, atoms) = disconnected_ring_coupled_tetrahedral_fixture();
    let second_center = surface.tetrahedral_center(atoms[4]).unwrap();
    let mut pending = vec![(initial(&surface), 0_usize)];
    let mut boundary = None;
    let mut explored = 0;
    while let Some((state, atom_offset)) = pending.pop() {
        explored += 1;
        assert!(explored < 100_000);
        let visited = state
            .observe_raw()
            .structural
            .traversal
            .visited_atoms
            .into_iter()
            .collect::<BTreeSet<_>>();
        if visited == BTreeSet::from([atoms[0], atoms[1], atoms[2], atoms[3]])
            && state.active_atom().is_none()
            && state.pending.is_none()
        {
            boundary = Some(state);
            break;
        }
        if !state.is_accepted() {
            let choices = ring_witness_choices(&state);
            assert!(
                !choices.is_empty(),
                "nonaccepted disconnected ring-coupled state has no continuation"
            );
            for choice in choices {
                if choice.text() == "-" {
                    continue;
                }
                if !selected_atom_spelling_transition(&state, &choice, atom_offset) {
                    continue;
                }
                let token = choice.text().to_owned();
                let successor = choice.into_successor();
                let next_offset =
                    next_atom_spelling_offset(&state, &successor, atom_offset, &token);
                pending.push((successor, next_offset));
            }
        }
    }

    let boundary = boundary.expect("first component must complete before the second root");
    assert!(boundary.labels.is_clean());
    let terminals = reachable_terminal_states_with_bond_spelling(boundary, false);
    assert!(!terminals.is_empty());
    for terminal in terminals {
        assert!(terminal.is_accepted());
        assert!(terminal.labels.is_clean());
        assert!(terminal
            .structural
            .semantic_domain(second_center.order_variable)
            .is_singleton());
        assert!(terminal
            .structural
            .semantic_domain(second_center.role_pattern_variable)
            .is_singleton());
    }
}

#[test]
fn contradictory_tetrahedral_token_is_filtered_without_suppressing_its_sibling() {
    let (surface, atoms, _) = tetrahedral_star_fixture(4);
    let state =
        NonStereoWriterState::<NonconformingRejectEvenTetrahedralParitySolver>::initial(&surface)
            .unwrap()
            .unwrap_consistent();
    let source = state.observe_raw();
    let center_choices = completed_atom_witnesses(&state, atoms[0]);

    assert_eq!(center_choices.len(), 1);
    assert_eq!(center_choices[0].0, "[C@@]");
    assert_eq!(state.observe_raw(), source);
}

#[test]
fn backend_failure_aborts_the_tetrahedral_atom_token_batch() {
    let (surface, _, _) = tetrahedral_star_fixture(4);
    TETRAHEDRAL_BACKEND_ATTEMPTS.store(0, Ordering::Relaxed);
    let state = NonStereoWriterState::<FailTetrahedralRestrictionSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();

    assert!(matches!(
        state.choices(),
        Err(ChoiceFailure::Backend(InjectedSolverFailure::Restriction))
    ));
}

#[test]
fn explicit_bond_is_hidden_when_joint_child_entry_contradicts() {
    let (surface, atoms, _) = entered_tetrahedral_fixture(NonStereoBondToken::Double, false);
    let initial =
        NonStereoWriterState::<NonconformingRejectTetrahedralParitySolver>::initial(&surface)
            .unwrap()
            .unwrap_consistent();
    let (_, parent) = only_root_atom(&initial, atoms[0]);

    assert!(matches!(
        parent.choices(),
        Err(ChoiceFailure::Invariant(
            WriterInvariantFailure::AllCandidatesSemanticallyRejected { candidate_count: 2 }
        ))
    ));
}

#[test]
fn parenthesis_only_publishes_jointly_admitted_child_spellings() {
    let (surface, atoms, bonds) = entered_tetrahedral_fixture(NonStereoBondToken::Elided, true);
    let initial =
        NonStereoWriterState::<NonconformingRejectTetrahedralParitySolver>::initial(&surface)
            .unwrap()
            .unwrap_consistent();
    let (_, parent) = only_root_atom(&initial, atoms[0]);
    let choices = ring_witness_choices(&parent);

    assert_eq!(choices.len(), 2);
    let decision = surface.molecule().bond_decision_variable(bonds[4]).unwrap();
    let mut texts = BTreeSet::new();
    for choice in choices {
        assert_eq!(choice.text(), "(");
        let Some(PendingEmission::BranchTraversalEmission(entry)) = &choice.successor().pending
        else { panic!("published parenthesis must carry its child entry") };
        assert_eq!(entry.bond(), bonds[4]);
        let domain = choice.successor().structural.semantic_domain(decision);
        match entry.bond_text {
            "" => assert!(domain.is_subset_of(BondRepresentation::elided_domain())),
            "-" => assert!(domain.is_subset_of(BondRepresentation::explicit_domain())),
            other => panic!("unexpected branch bond text: {other}"),
        }
        texts.insert(entry.bond_text);
    }
    assert_eq!(texts, BTreeSet::from(["", "-"]));
}

#[test]
fn parenthesis_is_hidden_when_every_joint_child_entry_contradicts() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..9).map(|_| graph.add_atom().unwrap()).collect::<Vec<_>>();
    let edges = [(0, 1), (0, 2), (1, 3), (1, 4), (1, 5),
        (2, 6), (2, 7), (2, 8)];
    let bonds = edges.map(|(a, b)| graph.add_bond(atoms[a], atoms[b]).unwrap());
    let tetrahedral = |entry, children: [usize; 3]| PreparedAtomToken::Tetrahedral {
        reference_order: [entry, bonds[children[0]], bonds[children[1]], bonds[children[2]]]
            .map(TetrahedralLigand::Bond),
        text_by_parity: ["[C@]".to_owned(), "[C@@]".to_owned()],
    };
    let mut tokens = vec![
        PreparedAtomToken::Fixed("P".to_owned()),
        tetrahedral(bonds[0], [2, 3, 4]),
        tetrahedral(bonds[1], [5, 6, 7]),
    ];
    tokens.extend((3..9).map(|_| PreparedAtomToken::Fixed("F".to_owned())));
    let surface = PreparedNonStereo::with_atom_tokens(
        PreparedMolecule::new(graph.build()),
        tokens,
        vec![NonStereoBondToken::Elided; bonds.len()],
    ).unwrap();
    let state = NonStereoWriterState::<NonconformingRejectTetrahedralParitySolver>::initial(&surface)
        .unwrap().unwrap_consistent();
    let (_, parent) = only_root_atom(&state, atoms[0]);
    assert!(matches!(parent.choices(),
        Err(ChoiceFailure::Invariant(
            WriterInvariantFailure::AllCandidatesSemanticallyRejected { candidate_count }
        )) if candidate_count > 0));
}

#[test]
fn tetrahedral_atom_spellings_partition_the_order_domain() {
    let (surface, atoms, _) = tetrahedral_star_fixture(4);
    let center = surface.tetrahedral_center(atoms[0]).unwrap();
    let initial = initial(&surface);
    let witnesses = completed_atom_witnesses(&initial, atoms[0]);
    assert_eq!(
        witnesses
            .iter()
            .map(|(text, _)| text.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["[C@]", "[C@@]"])
    );
    let first = witnesses[0]
        .1
        .structural
        .semantic_domain(center.order_variable);
    let second = witnesses[1]
        .1
        .structural
        .semantic_domain(center.order_variable);
    assert!(first.intersect(second).is_empty());
    assert_eq!(first.union(second), full_order_domain());
}

#[test]
fn ring_and_child_atom_frontiers_preserve_ring_and_parity_choices() {
    type WriterState = NonStereoWriterState<NativeSolverState>;

    let (surface, atoms, bonds) = ring_coupled_tetrahedral_fixture(NonStereoBondToken::Double);
    let initial = WriterState::initial(&surface).unwrap().unwrap_consistent();
    let (_, rooted) = only_root_atom(&initial, atoms[0]);
    let mut pending = vec![(rooted, 0_usize)];
    let mut saw_opening = false;
    let mut saw_closure = false;
    let mut explored = 0;
    while let Some((state, atom_offset)) = pending.pop() {
        explored += 1;
        assert!(explored < 20_000);
        let source = state.observe_raw();
        let choices = ring_witness_choices(&state);
        for choice in &choices {
            let successor = choice.successor().observe_raw();
            match (
                &source.structural.traversal.bond_progress[bonds[0].index()],
                &successor.structural.traversal.bond_progress[bonds[0].index()],
            ) {
                (ObservedBondProgress::Unrepresented, ObservedBondProgress::RingOpen { .. }) => {
                    saw_opening = true;
                }
                (
                    ObservedBondProgress::RingOpen { .. },
                    ObservedBondProgress::RingClosed { .. },
                ) => {
                    saw_closure = true;
                }
                _ => {}
            }
        }
        for choice in choices {
            if !selected_atom_spelling_transition(&state, &choice, atom_offset) {
                continue;
            }
            let token = choice.text().to_owned();
            let successor = choice.into_successor();
            let next_offset = next_atom_spelling_offset(&state, &successor, atom_offset, &token);
            pending.push((successor, next_offset));
        }
        if saw_opening && saw_closure {
            break;
        }
    }
    assert!(saw_opening && saw_closure);

    for token in [NonStereoBondToken::Elided, NonStereoBondToken::Double] {
        let (surface, atoms, _) = entered_tetrahedral_fixture(token, false);
        let initial = WriterState::initial(&surface).unwrap().unwrap_consistent();
        let (_, parent) = only_root_atom(&initial, atoms[0]);
        let atom_frontiers = if token == NonStereoBondToken::Double {
            ring_witness_choices(&parent)
                .into_iter()
                .filter(|choice| choice.text() == "=")
                .map(Choice::into_successor)
                .collect::<Vec<_>>()
        } else {
            vec![parent]
        };
        let atom_choices = atom_frontiers.iter()
            .flat_map(|state| completed_atom_witnesses(state, atoms[1]))
            .collect::<Vec<_>>();
        assert_eq!(
            atom_choices.len(),
            2,
            "both tetrahedral marker witnesses survive"
        );
        let order_domains = atom_choices
            .iter()
            .map(|(_, successor)| {
                successor
                    .structural
                    .semantic_domain(surface.tetrahedral_center(atoms[1]).unwrap().order_variable)
            })
            .collect::<Vec<_>>();
        assert!(order_domains[0].intersect(order_domains[1]).is_empty());
    }
}

#[test]
fn isolated_tetrahedral_order_domains_do_not_create_exact_search_work() {
    let (surface, _, _) = tetrahedral_star_fixture(4);
    let state = initial(&surface);

    assert_eq!(
        state
            .structural
            .constraints_for_test()
            .tetrahedral_factor_revision_count(),
        0,
        "latent tetrahedral factors do no initial work"
    );
    assert_eq!(
        state.structural.constraints_for_test().exact_run_counts(),
        (0, 0)
    );
    state
        .structural
        .constraints_for_test()
        .reset_tetrahedral_factor_revision_count();
    let _choices = ring_witness_choices(&state);
    assert!(
        state
            .structural
            .constraints_for_test()
            .tetrahedral_factor_revision_count()
            > 0,
        "the tetrahedral root candidates revise their newly active factors"
    );
    assert_eq!(
        state.structural.constraints_for_test().exact_run_counts(),
        (0, 0)
    );
}

#[test]
fn pending_prevalidation_work_is_measured_at_the_lexical_frontier() {
    let (surface, atoms, _) = entered_tetrahedral_fixture(NonStereoBondToken::Double, false);
    let (_, parent) = only_root_atom(&initial(&surface), atoms[0]);

    parent.reset_writer_work_counts();
    let pending_atom = ring_witness_choices(&parent)
        .into_iter()
        .find(|choice| choice.text() == "=")
        .unwrap()
        .into_successor();
    assert_eq!(
        parent.writer_work_counts(),
        (1, 2),
        "the explicit bond validates one two-choice atom frontier"
    );

    pending_atom.reset_writer_work_counts();
    assert_eq!(ring_witness_choices(&pending_atom).len(), 2);
    assert_eq!(
        pending_atom.writer_work_counts(),
        (1, 0),
        "publishing the pending atom frontier does not discard its successors"
    );
}

#[test]
fn top_level_component_completion_is_silent_normalization() {
    let (surface, atoms, _) = fixture(&["A", "B"], &[]);
    let initial = initial(&surface);
    let (_, first) = only_root_atom(&initial, atoms[0]);

    assert_eq!(initial.active_atom(), None, "the source remains unchanged");
    assert_eq!(first.active_atom(), None);
    assert!(!first.graph_is_complete());
    assert_eq!(first.pending, None);
    assert!(first.labels.is_clean());
    assert_eq!(
        first.structural.derive_candidates().candidates(),
        &[StructuralCandidate::Root { atom: atoms[1] }]
    );
}

#[test]
fn equal_dot_choices_commit_distinct_pending_component_roots() {
    let (surface, atoms, _) = fixture(&["A", "B", "C"], &[]);
    let initial = initial(&surface);
    let (_, after_first) = only_root_atom(&initial, atoms[0]);

    let choices = ring_witness_choices(&after_first);
    assert_eq!(choices.len(), 2);
    assert!(choices.iter().all(|choice| choice.text() == "."));
    assert_eq!(
        after_first.structural.derive_candidates().candidates(),
        &[
            StructuralCandidate::Root { atom: atoms[1] },
            StructuralCandidate::Root { atom: atoms[2] },
        ]
    );
    assert_eq!(
        after_first.active_atom(),
        None,
        "the source remains unchanged"
    );
    for (choice, root) in choices.iter().zip(&atoms[1..]) {
        let witness = declared_atom_witnesses(&surface, *root)
            .into_iter()
            .next()
            .unwrap();
        let atom_choices = choices_matching_declared_witnesses(choice.successor(), 0);
        assert!(!atom_choices.is_empty());
        assert!(atom_choices
            .iter()
            .any(|atom_choice| witness.starts_with(atom_choice.text())));
        let completed = consume_atom_spelling_states(choice.successor().clone(), &witness, 0);
        assert!(completed.iter().any(|state| atom_was_emitted(state, *root)));
    }
}

#[test]
fn equal_text_choices_retain_distinct_successors() {
    let (surface, atoms, _) = fixture(&["C", "C"], &[(0, 1, NonStereoBondToken::Elided)]);
    let initial = initial(&surface);

    let choices = choices_matching_declared_witnesses(&initial, 0);
    assert!(choices.len() >= 2);
    assert!(choices
        .iter()
        .all(|choice| choice.text() == choices[0].text()));
    assert_eq!(initial.active_atom(), None);

    let reached_by_choice = choices
        .iter()
        .map(|choice| {
            atoms
                .iter()
                .copied()
                .filter(|atom| {
                    let witness = declared_atom_witnesses(&surface, *atom)[0].clone();
                    witness.starts_with(choice.text())
                        && consume_atom_spelling_states(
                            choice.successor().clone(),
                            &witness,
                            choice.text().len(),
                        )
                        .iter()
                        .any(|state| atom_was_emitted(state, *atom))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert!(reached_by_choice
        .iter()
        .any(|left| { reached_by_choice.iter().any(|right| left != right) }));
    assert_eq!(
        reached_by_choice
            .iter()
            .flatten()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        atoms.iter().copied().collect()
    );
}

#[test]
fn equal_parentheses_enter_distinct_children_and_restrict_parent_order() {
    let (surface, atoms, bonds) = tetrahedral_star_fixture(4);
    let center = surface.tetrahedral_center(atoms[0]).unwrap();
    let rooted = completed_atom_witnesses(&initial(&surface), atoms[0])
        .into_iter()
        .find(|(text, _)| text.contains('@') && !text.contains("@@"))
        .unwrap().1;
    let before = rooted.observe_raw();
    let branches = ring_witness_choices(&rooted)
        .into_iter()
        .filter(|choice| choice.text() == "(")
        .collect::<Vec<_>>();
    assert!(branches.len() >= 2);
    for (child, bond) in [(atoms[1], bonds[0]), (atoms[2], bonds[1])] {
        let successor = branches.iter().find(|choice| {
            matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == incident(&surface, atoms[0], bond))
        }).unwrap().successor();
        assert_eq!(successor.active_atom(), Some(child));
        assert_eq!(successor.structural.observe_raw().traversal.visited_atoms, vec![atoms[0], child]);
        assert!(successor.structural.bond_decision_domain(bond).is_subset_of(
            surface.molecule().bond_role_partition(bond).unwrap().traversal_values()
        ));
    }
    let chosen = |bond| branches.iter().find(|choice| {
        matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == incident(&surface, atoms[0], bond))
    }).unwrap().successor();
    assert_ne!(
        chosen(bonds[0]).structural.semantic_domain(center.order_variable),
        chosen(bonds[1]).structural.semantic_domain(center.order_variable),
    );
    assert_eq!(rooted.observe_raw(), before);
}

#[test]
fn accepted_state_returns_an_ordinary_empty_choice_result() {
    let (surface, atoms, _) = fixture(&["C"], &[]);
    let initial = initial(&surface);
    let (_, accepted) = only_root_atom(&initial, atoms[0]);

    assert!(accepted.is_accepted());
    assert!(ring_witness_choices(&accepted).is_empty());
}

#[test]
fn choices_derives_the_source_frontier_once() {
    let surface = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    )
    .0;
    let initial = initial(&surface);
    let (_, rooted) = only_root_atom(&initial, AtomId::new(0));
    let before = rooted.structural.candidate_batch_derivation_count();

    let choices = rooted.choices().unwrap();

    assert_eq!(choices.len(), 640);
    assert_eq!(
        rooted.structural.candidate_batch_derivation_count(),
        before + 1
    );
}

#[test]
fn backend_failure_aborts_the_candidate_batch() {
    let surface = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    )
    .0;
    let initial = NonStereoWriterState::<FailingRestrictionSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let (_, mut rooted) = only_root_atom(&initial, AtomId::new(0));
    rooted.labels.maximum_spelling_label = Some(0);
    // ID zero is legal; reserve it to exhaust this test-only label domain.
    rooted.labels.allocate_at(RingLabelSlot(0), BondId::new(999));

    assert!(matches!(
        rooted.choices(),
        Err(ChoiceFailure::Backend(InjectedSolverFailure::Restriction))
    ));
}

#[test]
fn late_backend_failure_discards_an_earlier_accepted_choice() {
    let surface = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    )
    .0;
    let initial = NonStereoWriterState::<FailSecondVariableSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let (_, rooted) = only_root_atom(&initial, AtomId::new(0));

    assert!(matches!(
        rooted.choices(),
        Err(ChoiceFailure::Backend(InjectedSolverFailure::Restriction))
    ));
}

#[test]
fn ring_alternative_generation_stops_at_the_first_backend_failure() {
    let (surface, atoms, bonds) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = NonStereoWriterState::<FailFirstRingAlternativeSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let (_, rooted) = only_root_atom(&initial, atoms[0]);
    let candidate = rooted
        .structural
        .derive_candidates()
        .candidates()
        .iter()
        .copied()
        .find(|candidate| {
            matches!(
                candidate,
                StructuralCandidate::RingOpen { incident } if incident.bond() == bonds[0]
            )
        })
        .unwrap();
    let StructuralCandidate::RingOpen { incident } = candidate else {
        unreachable!()
    };

    assert!(matches!(
        rooted.attempt_ring_openings(candidate, incident, &[]),
        Err(InjectedSolverFailure::Restriction)
    ));
}

#[test]
fn contradictory_candidate_is_filtered_without_suppressing_its_sibling() {
    let (surface, atoms, bonds) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    assert_eq!(
        surface.molecule().bond_decision_variable(bonds[0]),
        Some(crate::ids::VariableId::new(0))
    );
    let initial = NonStereoWriterState::<RejectFirstVariableSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let (_, rooted) = only_root_atom(&initial, atoms[0]);

    let choices = rooted.choices().unwrap();

    assert_eq!(choices.len(), 320);
    assert!(choices.iter().all(|choice| {
        choice
            .successor()
            .labels
            .bonds_by_slot
            .values()
            .copied()
            .collect::<Vec<_>>()
            == vec![bonds[1]]
    }));
    assert!(rooted.labels.is_clean());
}

#[test]
fn writer_policy_contradiction_is_candidate_local() {
    let (surface, atoms, bonds) = fixture(
        &["R", "A", "B", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (0, 3, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
            (2, 3, NonStereoBondToken::Elided),
        ],
    );
    let initial = NonStereoWriterState::<WriterPolicyContradictionSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let (_, rooted) = only_root_atom(&initial, atoms[0]);

    let choices = ring_witness_choices(&rooted);

    assert!(!choices.is_empty());
    assert!(choices.iter().all(|choice| {
        !choice
            .successor()
            .labels
            .bonds_by_slot
            .values()
            .any(|bond| *bond == bonds[0])
    }));
    assert!(choices.iter().any(|choice| {
        choice
            .successor()
            .labels
            .bonds_by_slot
            .values()
            .any(|bond| *bond == bonds[1])
    }));
}

#[test]
fn pending_atom_completion_rejects_a_contradictory_structural_frontier() {
    let (surface, atoms, _) = fixture(
        &["R", "A", "Q", "X"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (1, 2, NonStereoBondToken::Elided),
            (1, 3, NonStereoBondToken::Elided),
            (2, 3, NonStereoBondToken::Elided),
        ],
    );
    let initial = NonStereoWriterState::<PendingContradictionSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let (_, rooted) = only_root_atom(&initial, atoms[0]);

    let (_, after_double_bond) = only_choice(&rooted, "=");
    let mut after_child = vec![after_double_bond];
    let mut witness_prefix = String::new();
    for expected in "[*:2".chars() {
        let mut successors = Vec::new();
        for state in after_child {
            let choices = state.choices().unwrap_or_else(|failure| {
                panic!(
                    "child witness failed after {witness_prefix:?} before {expected:?}: {failure:?}; pending={:?}, spelling={:?}",
                    state.pending,
                    state.atom_spelling
                )
            });
            successors.extend(
                choices
                    .into_iter()
                    .filter(|choice| choice.text() == expected.to_string())
                    .map(Choice::into_successor),
            );
        }
        witness_prefix.push(expected);
        after_child = successors;
    }
    assert!(
        !after_child.is_empty(),
        "the declared child witness remains reachable"
    );
    let failures = after_child
        .iter()
        .map(NonStereoWriterState::choices)
        .collect::<Vec<_>>();
    assert!(
        failures.iter().all(|choices| matches!(
            choices,
            Err(ChoiceFailure::Invariant(
                WriterInvariantFailure::PendingEmissionRejected
            ))
        )),
        "unexpected candidate outcomes at the final child-atom field: {failures:?}"
    );
}

#[test]
fn all_candidate_spelling_exhaustion_is_typed() {
    let (surface, atoms, _) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let (_, mut rooted) = only_root_atom(&initial, atoms[0]);
    rooted.labels.maximum_spelling_label = Some(0);
    // ID zero is legal; reserve it to exhaust this test-only label domain.
    rooted.labels.allocate_at(RingLabelSlot(0), BondId::new(999));

    assert!(matches!(
        rooted.choices(),
        Err(ChoiceFailure::Spelling(
            SpellingFailure::RingLabelExhausted {
                next_label: 1,
                maximum_label: 0,
                blocked_candidate_count: 6
            }
        ))
    ));
}

#[test]
fn explicit_pending_labels_retain_spelling_exhaustion_classification() {
    let (surface, atoms, _) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Double),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let (_, mut rooted) = only_root_atom(&initial, atoms[0]);
    rooted.labels.maximum_spelling_label = Some(0);
    // ID zero is legal; reserve it to exhaust this test-only label domain.
    rooted.labels.allocate_at(RingLabelSlot(0), BondId::new(999));

    assert!(matches!(
        rooted.choices(),
        Err(ChoiceFailure::Spelling(
            SpellingFailure::RingLabelExhausted {
                next_label: 1,
                maximum_label: 0,
                blocked_candidate_count: 4
            }
        ))
    ));
}

#[test]
fn explicit_closure_pending_label_retains_spelling_exhaustion_classification() {
    let (surface, atoms, bonds) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let (_, rooted) = only_root_atom(&initial, atoms[0]);
    let opening = ring_witness_choices(&rooted)
        .into_iter()
        .find(|choice| {
            choice.text() == "="
                && choice
                    .successor()
                    .labels
                    .bonds_by_slot
                    .values()
                    .any(|bond| *bond == bonds[0])
        })
        .unwrap()
        .into_successor();
    let (_, opened) = only_choice(&opening, "1");
    let (_, walked) = only_choice(&opened, "C");
    let (_, mut walked) = only_choice(&walked, "C");
    walked.labels.maximum_spelling_label = Some(0);

    assert!(matches!(
        walked.choices(),
        Err(ChoiceFailure::Spelling(
            SpellingFailure::RingLabelExhausted {
                next_label: 1,
                maximum_label: 0,
                blocked_candidate_count: 2
            }
        ))
    ));
}

#[test]
fn viable_unspellable_candidate_outweighs_a_contradictory_sibling() {
    let (surface, atoms, _) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = NonStereoWriterState::<RejectFirstVariableSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let (_, mut rooted) = only_root_atom(&initial, atoms[0]);
    rooted.labels.maximum_spelling_label = Some(0);
    // ID zero is legal; reserve it to exhaust this test-only label domain.
    rooted.labels.allocate_at(RingLabelSlot(0), BondId::new(999));

    assert!(matches!(
        rooted.choices(),
        Err(ChoiceFailure::Spelling(
            SpellingFailure::RingLabelExhausted {
                next_label: 1,
                maximum_label: 0,
                blocked_candidate_count: 3
            }
        ))
    ));
}

#[test]
fn contradiction_precedes_spelling_for_each_candidate() {
    let (surface, atoms, _) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = NonStereoWriterState::<RejectEveryRestrictionSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let (_, mut rooted) = only_root_atom(&initial, atoms[0]);
    rooted.labels.maximum_spelling_label = Some(0);
    // ID zero is legal; reserve it to exhaust this test-only label domain.
    rooted.labels.allocate_at(RingLabelSlot(0), BondId::new(999));

    assert!(matches!(
        rooted.choices(),
        Err(ChoiceFailure::Invariant(
            WriterInvariantFailure::AllCandidatesSemanticallyRejected { candidate_count: 6 }
        ))
    ));
}

#[test]
fn explicit_endpoint_contradiction_precedes_pending_label_spelling() {
    let (surface, atoms, bonds) = fixture(
        &["R", "A", "B", "C"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Elided),
            (0, 3, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
            (2, 3, NonStereoBondToken::Elided),
        ],
    );
    let initial = NonStereoWriterState::<RejectEveryRestrictionSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let (_, mut rooted) = only_root_atom(&initial, atoms[0]);
    rooted.labels.maximum_spelling_label = Some(0);
    // ID zero is legal; reserve it to exhaust this test-only label domain.
    rooted.labels.allocate_at(RingLabelSlot(0), BondId::new(999));
    let candidate = rooted
        .structural
        .derive_candidates()
        .candidates()
        .iter()
        .copied()
        .find(|candidate| {
            matches!(
                candidate,
                StructuralCandidate::RingOpen { incident } if incident.bond() == bonds[0]
            )
        })
        .unwrap();
    let StructuralCandidate::RingOpen { incident } = candidate else {
        unreachable!()
    };

    let attempts = rooted
        .attempt_ring_openings(candidate, incident, &[])
        .unwrap();

    assert_eq!(attempts.len(), 2);
    assert!(attempts.into_iter().all(|attempt| matches!(
        attempt,
        CandidateAttempt::Rejected {
            reason: CandidateRejection::Contradiction
        }
    )));
}

#[test]
fn unspellable_openings_do_not_suppress_a_valid_closure() {
    let (surface, atoms, bonds) = fixture(
        &["A", "B", "C", "D", "E"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
            (1, 3, NonStereoBondToken::Elided),
            (1, 4, NonStereoBondToken::Elided),
            (3, 4, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let (_, rooted) = only_root_atom(&initial, atoms[0]);
    let opened = ring_witness_choices(&rooted)
        .into_iter()
        .find(|choice| {
            choice
                .successor()
                .labels
                .bonds_by_slot
                .values()
                .any(|bond| *bond == bonds[0])
        })
        .unwrap()
        .into_successor();
    let (_, walked) = only_choice(&opened, "C");
    let (_, mut walked) = only_choice(&walked, "B");
    assert_eq!(walked.active_atom(), Some(atoms[1]));
    walked.labels.maximum_spelling_label = Some(1);
    walked.labels.allocate_at(RingLabelSlot(0), BondId::new(999));
    assert_eq!(walked.labels.next_available(), RingLabelSlot(2));

    let choices = ring_witness_choices(&walked);

    assert_eq!(choices.len(), 1);
    assert_eq!(choices[0].text(), "1");
    assert!(!choices[0]
        .successor()
        .labels
        .bonds_by_slot
        .values()
        .any(|bond| *bond == bonds[0]));
}

#[test]
fn closed_labels_are_immediately_reusable() {
    let first = BondId::new(0);
    let second = BondId::new(1);
    let third = BondId::new(2);
    let mut labels = RingLabels::default();

    let zero = labels.allocate(first);
    let one = labels.allocate(second);
    assert_eq!(zero, RingLabelSlot(0));
    assert_eq!(one, RingLabelSlot(1));
    labels.release(zero, first);
    assert_eq!(labels.allocate(third), RingLabelSlot(0));
}

#[test]
fn ring_label_fullness_uses_only_regime_spelling_slots() {
    let regimes = [
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder },
        WriterRegime::GenericCompact,
    ];
    for regime in regimes {
        let mut labels = RingLabels::default();
        labels.maximum_spelling_label = Some(0);
        assert!(!labels.is_full_for(regime));
        labels.allocate_at(RingLabelSlot(0), BondId::new(0));
        assert!(labels.is_full_for(regime));
        labels.maximum_spelling_label = Some(1);
        assert!(!labels.is_full_for(regime));
        labels.allocate_at(RingLabelSlot(1), BondId::new(1));
        assert!(labels.is_full_for(regime));
    }

    let mut rdkit = RingLabels::default();
    rdkit.maximum_spelling_label = Some(0);
    assert!(rdkit.is_full_for(WriterRegime::RdkitRandomNoncanonical));
    rdkit.maximum_spelling_label = Some(1);
    assert!(!rdkit.is_full_for(WriterRegime::RdkitRandomNoncanonical));
    rdkit.allocate_at(RingLabelSlot(0), BondId::new(0));
    assert!(!rdkit.is_full_for(WriterRegime::RdkitRandomNoncanonical),
        "slot zero is outside RDKit's spelling range");
    rdkit.allocate_at(RingLabelSlot(1), BondId::new(1));
    assert!(rdkit.is_full_for(WriterRegime::RdkitRandomNoncanonical));
}

#[test]
fn ring_label_spelling_matches_the_selected_smiles_dialect() {
    assert_eq!(ring_label_number_text(1), "1");
    assert_eq!(ring_label_number_text(9), "9");
    assert_eq!(ring_label_number_text(10), "%10");
    assert_eq!(ring_label_number_text(99), "%99");
}

#[test]
#[should_panic(expected = "above 99 require an explicit dialect policy")]
fn unselected_large_ring_label_dialect_fails_at_rendering() {
    let _ = ring_label_number_text(100);
}

#[test]
fn elided_triangle_emits_a_complete_ring() {
    let (surface, atoms, bonds) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let left = incident(&surface, atoms[0], bonds[0]);
    let right = incident(&surface, atoms[0], bonds[1]);
    let between = incident(&surface, atoms[2], bonds[2]);
    let closing = incident(&surface, atoms[1], bonds[0]);

    let (root, rooted) = only_root_atom(&initial, atoms[0]);
    let rooted_choices = ring_witness_choices(&rooted);
    assert_eq!(
        rooted_choices.iter().map(Choice::text).collect::<Vec<_>>(),
        vec!["1", "1", "-", "1", "1", "-"]
    );
    assert!(rooted_choices[0]
        .successor()
        .labels
        .bonds_by_slot
        .values()
        .any(|bond| *bond == left.bond()));
    assert!(rooted_choices[3]
        .successor()
        .labels
        .bonds_by_slot
        .values()
        .any(|bond| *bond == right.bond()));
    let first_opening = rooted_choices.into_iter().next().unwrap();
    let open = first_opening.text;
    let opened = first_opening.successor;
    let (first_child, walked) = only_choice(&opened, "C");
    let (second_child, walked) = only_choice(&walked, "C");
    assert_eq!(walked.structural.active_atom(), Some(atoms[1]));
    assert_eq!(between.bond(), bonds[2]);
    assert_eq!(closing.bond(), bonds[0]);
    let closing_choices = walked.choices().unwrap().into_iter()
        .filter(|choice| choice.text() == "%01").collect::<Vec<_>>();
    assert_eq!(closing_choices.len(), 1);
    let closing_choice = closing_choices.into_iter().next().unwrap();
    let close = closing_choice.text().to_owned();
    let accepted = closing_choice.into_successor();

    assert_eq!(
        [root, open, first_child, second_child, close].concat(),
        "C1CC%01"
    );
    assert!(accepted.is_accepted());
}

#[test]
fn explicit_ring_bond_is_emitted_at_closure_before_its_label() {
    let (surface, atoms, bonds) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let ring = incident(&surface, atoms[0], bonds[0]);
    let entry = incident(&surface, atoms[0], bonds[1]);
    let between = incident(&surface, atoms[2], bonds[2]);
    let closing = incident(&surface, atoms[1], bonds[0]);

    let (root, rooted) = only_root_atom(&initial, atoms[0]);
    let (open, opened) = choice_at(&rooted, 0);
    assert!(opened
        .labels
        .bonds_by_slot
        .values()
        .any(|bond| *bond == ring.bond()));
    let (first_child, walked) = only_choice(&opened, "C");
    assert_eq!(entry.bond(), bonds[1]);
    let (second_child, walked) = only_choice(&walked, "C");
    assert_eq!(between.bond(), bonds[2]);
    assert_eq!(closing.bond(), bonds[0]);

    let (bond, pending_label) = only_choice(&walked, "=");
    assert_eq!(bond, "=");
    assert_eq!(pending_label.active_atom(), Some(atoms[1]));
    assert!(pending_label.graph_is_complete());

    let (label, accepted) = only_choice(&pending_label, "1");
    assert_eq!(
        [root, open, first_child, second_child, bond, label].concat(),
        "C1CC=1"
    );
    assert!(accepted.is_accepted());
}

#[test]
fn explicit_ring_opening_choices_refine_the_fixed_endpoint_plan() {
    let (surface, atoms, bonds) = fixture(
        &["A", "B", "C"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let initial_plan = BondRepresentation::explicit_domain();

    for (root, omitted_plan, emitted_plan) in [
        (
            atoms[0],
            BondRepresentation::Ring01.singleton_domain(),
            BondRepresentation::Ring10
                .singleton_domain()
                .union(BondRepresentation::Ring11.singleton_domain()),
        ),
        (
            atoms[1],
            BondRepresentation::Ring10.singleton_domain(),
            BondRepresentation::Ring01
                .singleton_domain()
                .union(BondRepresentation::Ring11.singleton_domain()),
        ),
    ] {
        let (_, rooted) = only_root_atom(&initial, root);
        let choices = ring_witness_choices(&rooted)
            .into_iter()
            .filter(|choice| {
                choice
                    .successor()
                    .labels
                    .bonds_by_slot
                    .values()
                    .any(|bond| *bond == bonds[0])
            })
            .collect::<Vec<_>>();

        assert_eq!(choices.len(), 2);
        let omitted = choices.iter().find(|choice| choice.text() == "1").unwrap();
        let emitted = choices.iter().find(|choice| choice.text() == "=").unwrap();
        assert_eq!(
            omitted
                .successor()
                .structural
                .bond_decision_domain(bonds[0]),
            omitted_plan
        );
        assert_eq!(
            emitted
                .successor()
                .structural
                .bond_decision_domain(bonds[0]),
            emitted_plan
        );
        assert!(matches!(&emitted.successor().pending,
            Some(PendingEmission::RingOpeningLabel { incident, .. })
                if incident.bond() == bonds[0]
        ));
        assert_eq!(emitted.successor().choices().unwrap()[0].text(), "1");
        assert_eq!(
            rooted.structural.bond_decision_domain(bonds[0]),
            initial_plan
        );
    }
}

#[test]
fn explicit_ring_closure_resolves_opening_only_closure_only_and_both_plans() {
    let (surface, atoms, bonds) = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let (root_text, rooted) = only_root_atom(&initial, atoms[0]);
    let opening_choices = ring_witness_choices(&rooted)
        .into_iter()
        .filter(|choice| {
            choice
                .successor()
                .labels
                .bonds_by_slot
                .values()
                .any(|bond| *bond == bonds[0])
        })
        .collect::<Vec<_>>();
    let omitted_opening = opening_choices
        .iter()
        .find(|choice| choice.text() == "1")
        .unwrap()
        .successor()
        .clone();
    let emitted_opening = opening_choices
        .into_iter()
        .find(|choice| choice.text() == "=")
        .unwrap()
        .into_successor();

    let (_, walked) = only_choice(&omitted_opening, "C");
    let (_, walked) = only_choice(&walked, "C");
    let (closure_text, pending_label) = only_choice(&walked, "=");
    assert_eq!(
        pending_label.structural.bond_decision_domain(bonds[0]),
        BondRepresentation::Ring01.singleton_domain()
    );
    let (closure_label, closure_only) = only_choice(&pending_label, "1");
    assert_eq!(
        [root_text.as_str(), "1CC", &closure_text, &closure_label].concat(),
        "C1CC=1"
    );
    assert!(closure_only.is_accepted());

    let (opening_label, emitted_opening) = only_choice(&emitted_opening, "1");
    let (_, walked) = only_choice(&emitted_opening, "C");
    let (_, walked) = only_choice(&walked, "C");
    assert_eq!(
        walked.structural.bond_decision_domain(bonds[0]),
        BondRepresentation::Ring10
            .singleton_domain()
            .union(BondRepresentation::Ring11.singleton_domain())
    );
    let closure_choices = ring_witness_choices(&walked);
    assert_eq!(closure_choices.len(), 2);

    let omitted_closure = closure_choices
        .iter()
        .find(|choice| choice.text() == "1")
        .unwrap();
    assert_eq!(
        omitted_closure
            .successor()
            .structural
            .bond_decision_domain(bonds[0]),
        BondRepresentation::Ring10.singleton_domain()
    );
    assert_eq!(
        [root_text.as_str(), "=", &opening_label, "CC", "1"].concat(),
        "C=1CC1"
    );
    assert!(omitted_closure.successor().is_accepted());

    let emitted_closure = closure_choices
        .into_iter()
        .find(|choice| choice.text() == "=")
        .unwrap()
        .into_successor();
    assert_eq!(
        emitted_closure.structural.bond_decision_domain(bonds[0]),
        BondRepresentation::Ring11.singleton_domain()
    );
    let (label, both) = only_choice(&emitted_closure, "1");
    assert_eq!(
        [root_text.as_str(), "=", &opening_label, "CC=", &label,].concat(),
        "C=1CC=1"
    );
    assert!(both.is_accepted());
}

fn independent_ring_endpoint_projection(
    domain: Domain,
    endpoint: FixedBondEndpoint,
    spelling: RingEndpointSpelling,
) -> Domain {
    Domain::from_indices(domain.iter().filter(|value| {
        let placement = match *value {
            value if value == BondRepresentation::Traversal.value_index() => return false,
            value if value == BondRepresentation::Ring00.value_index() => (false, false),
            value if value == BondRepresentation::Ring10.value_index() => (true, false),
            value if value == BondRepresentation::Ring01.value_index() => (false, true),
            value if value == BondRepresentation::Ring11.value_index() => (true, true),
            _ => panic!("ring-plan oracle received an unknown representation value"),
        };
        let emitted = match endpoint {
            FixedBondEndpoint::A => placement.0,
            FixedBondEndpoint::B => placement.1,
        };
        emitted == (spelling == RingEndpointSpelling::Emit)
    }))
    .unwrap()
}

#[test]
fn endpoint_projection_masks_match_an_independent_ring_plan_oracle() {
    for bits in 0_u64..(1_u64 << 5) {
        let domain = Domain::from_bits(bits);
        for endpoint in [FixedBondEndpoint::A, FixedBondEndpoint::B] {
            for spelling in [RingEndpointSpelling::Omit, RingEndpointSpelling::Emit] {
                assert_eq!(
                    domain.intersect(BondRepresentation::endpoint_domain(endpoint, spelling)),
                    independent_ring_endpoint_projection(domain, endpoint, spelling),
                    "domain {domain:?}, endpoint {endpoint:?}, spelling {spelling:?}"
                );
            }
        }
    }
}

fn fixed_ring_placement_results_from(
    token: NonStereoBondToken,
    root_index: usize,
) -> Vec<(String, Domain)> {
    let (surface, atoms, bonds) = fixture(
        &["A", "R", "Q"],
        &[
            (0, 1, token),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let (root_text, root_state) = only_root_atom(&initial, atoms[root_index]);
    let mut pending = vec![(root_state, root_text, 0_usize)];
    let mut complete = Vec::new();
    let mut visited_state_count = 0_usize;
    while let Some((state, prefix, atom_offset)) = pending.pop() {
        visited_state_count += 1;
        assert!(
            visited_state_count <= 100_000,
            "explicit ring exploration exceeded its bounded test envelope"
        );
        if state.is_accepted() {
            let plan = state.structural.bond_decision_domain(bonds[0]);
            assert!(plan.is_singleton());
            if plan != BondRepresentation::Traversal.singleton_domain() {
                complete.push((surface.normalize_test_atom_classes(&prefix), plan));
            }
            continue;
        }
        for choice in ring_witness_choices(&state) {
            if !selected_atom_spelling_transition(&state, &choice, atom_offset) {
                continue;
            }
            let text = choice.text().to_owned();
            let successor = choice.into_successor();
            let next_offset = next_atom_spelling_offset(&state, &successor, atom_offset, &text);
            pending.push((successor, format!("{prefix}{text}"), next_offset));
        }
    }
    complete.sort_by(|left, right| left.0.cmp(&right.0));
    complete
}

fn fixed_ring_placement_results(token: NonStereoBondToken) -> Vec<(String, Domain)> {
    fixed_ring_placement_results_from(token, 0)
}

fn assert_ring_placement_mapping(
    token: NonStereoBondToken,
    root_index: usize,
    expected: &[(&str, BondRepresentation)],
) {
    let actual = fixed_ring_placement_results_from(token, root_index);
    let expected = expected.iter().flat_map(|&(text, plan)| {
        let atom_positions = text.match_indices(['A', 'Q', 'R'])
            .map(|(position, _)| position).collect::<Vec<_>>();
        assert_eq!(atom_positions.len(), 3);
        (0..4).map(move |spelling| {
            let mut text = text.to_owned();
            for edge in (0..2).rev() {
                if spelling & (1 << edge) != 0 {
                    text.insert(atom_positions[edge + 1], '-');
                }
            }
            (text, plan)
        })
    }).collect::<Vec<_>>();
    assert_eq!(
        actual.len(),
        expected.len(),
        "actual ring witnesses: {actual:?}"
    );
    for (text, plan) in expected {
        assert!(
            actual.iter().any(|(actual_text, actual_plan)| {
                actual_text == &text && *actual_plan == plan.singleton_domain()
            }),
            "missing ring placement {text:?} -> {plan:?} in {actual:?}"
        );
    }
}

#[test]
fn every_prepared_ring_token_uses_endpoint_relative_placement() {
    let standard = [
        (NonStereoBondToken::Aromatic, ":"),
        (NonStereoBondToken::Single, "-"),
        (NonStereoBondToken::Double, "="),
        (NonStereoBondToken::Triple, "#"),
    ];
    for (token, text) in standard {
        assert_ring_placement_mapping(
            token,
            0,
            &[
                (&format!("A{text}1QR1"), BondRepresentation::Ring10),
                (&format!("A1QR{text}1"), BondRepresentation::Ring01),
                (&format!("A{text}1QR{text}1"), BondRepresentation::Ring11),
            ],
        );
    }

    assert_ring_placement_mapping(
        NonStereoBondToken::DativeAToB,
        0,
        &[
            ("A->1QR1", BondRepresentation::Ring10),
            ("A1QR<-1", BondRepresentation::Ring01),
            ("A->1QR<-1", BondRepresentation::Ring11),
        ],
    );
    assert_ring_placement_mapping(
        NonStereoBondToken::DativeBToA,
        0,
        &[
            ("A<-1QR1", BondRepresentation::Ring10),
            ("A1QR->1", BondRepresentation::Ring01),
            ("A<-1QR->1", BondRepresentation::Ring11),
        ],
    );
    assert_ring_placement_mapping(
        NonStereoBondToken::DativeAToB,
        1,
        &[
            ("R<-1QA1", BondRepresentation::Ring01),
            ("R1QA->1", BondRepresentation::Ring10),
            ("R<-1QA->1", BondRepresentation::Ring11),
        ],
    );
    assert_ring_placement_mapping(
        NonStereoBondToken::DativeBToA,
        1,
        &[
            ("R->1QA1", BondRepresentation::Ring01),
            ("R1QA<-1", BondRepresentation::Ring10),
            ("R->1QA<-1", BondRepresentation::Ring11),
        ],
    );
    assert_ring_placement_mapping(
        NonStereoBondToken::Elided,
        0,
        &[
            ("A1QR1", BondRepresentation::Ring00),
            ("A-1QR1", BondRepresentation::Ring10),
            ("A1QR-1", BondRepresentation::Ring01),
            ("A-1QR-1", BondRepresentation::Ring11),
        ],
    );
}

#[test]
fn directed_ring_closure_uses_the_emitting_fixed_endpoint_orientation() {
    let (surface, atoms, bonds) = fixture(
        &["N", "B", "C"],
        &[
            (0, 1, NonStereoBondToken::DativeAToB),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let ring = incident(&surface, atoms[0], bonds[0]);
    let entry = incident(&surface, atoms[0], bonds[1]);
    let between = incident(&surface, atoms[2], bonds[2]);
    let closing = incident(&surface, atoms[1], bonds[0]);

    let (root, rooted) = only_root_atom(&initial, atoms[0]);
    let opening = ring_witness_choices(&rooted)
        .into_iter()
        .find(|choice| {
            choice
                .successor()
                .labels
                .bonds_by_slot
                .values()
                .any(|bond| *bond == ring.bond())
        })
        .unwrap();
    let open = opening.text;
    let opened = opening.successor;
    let (first_child, walked) = only_choice(&opened, "C");
    assert_eq!(entry.bond(), bonds[1]);
    let (second_child, walked) = only_choice(&walked, "B");
    assert_eq!(between.bond(), bonds[2]);
    assert_eq!(closing.bond(), bonds[0]);

    let (bond, pending_label) = only_choice(&walked, "<-");
    let (label, accepted) = only_choice(&pending_label, "1");

    assert_eq!(
        [root, open, first_child, second_child, bond, label].concat(),
        "N1CB<-1"
    );
    assert!(accepted.is_accepted());
}

#[test]
fn explicit_inline_bond_enters_child_before_atom_spelling() {
    let (surface, atoms, bonds) = fixture(&["C", "O"], &[(0, 1, NonStereoBondToken::Double)]);
    let initial = initial(&surface);
    let edge = incident(&surface, atoms[0], bonds[0]);
    let (_, rooted) = only_root_atom(&initial, atoms[0]);
    let (bond, pending) = only_choice(&rooted, "=");

    assert_eq!(bond, "=");
    assert_eq!(rooted.active_atom(), Some(atoms[0]));
    assert_eq!(pending.active_atom(), Some(atoms[1]));
    assert!(pending.structural.atom_is_visited(atoms[1]));
    assert!(matches!(&pending.pending, Some(PendingEmission::InlineAtom(entry)) if entry.incident == edge));
    let (atom, accepted) = only_choice(&pending, "O");
    assert_eq!(atom, "O");
    assert!(accepted.is_accepted());
}

#[test]
fn dative_bond_text_follows_prepared_orientation() {
    let (surface, atoms, bonds) = fixture(&["N", "B"], &[(0, 1, NonStereoBondToken::DativeAToB)]);
    let initial = initial(&surface);
    let edge_from_n = incident(&surface, atoms[0], bonds[0]);
    let edge_from_b = incident(&surface, atoms[1], bonds[0]);

    let (_, rooted_at_n) = only_root_atom(&initial, atoms[0]);
    assert_eq!(rooted_at_n.pending, None);
    assert_eq!(edge_from_n.bond(), bonds[0]);
    assert_eq!(ring_witness_choices(&rooted_at_n)[0].text(), "->");

    let (_, rooted_at_b) = only_root_atom(&initial, atoms[1]);
    assert_eq!(edge_from_b.bond(), bonds[0]);
    assert_eq!(ring_witness_choices(&rooted_at_b)[0].text(), "<-");
}

#[test]
fn explicit_branch_commits_at_open_parenthesis() {
    let (surface, atoms, bonds) = fixture(
        &["C", "O", "N"],
        &[
            (0, 1, NonStereoBondToken::Double),
            (0, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let oxygen = incident(&surface, atoms[0], bonds[0]);
    let nitrogen = incident(&surface, atoms[0], bonds[1]);
    let (root, rooted) = only_root_atom(&initial, atoms[0]);
    let branch_choice = ring_witness_choices(&rooted)
        .into_iter()
        .find(|choice| {
            matches!(&choice.successor.pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == oxygen)
        })
        .unwrap();
    let open = branch_choice.text;
    let pending_branch = branch_choice.successor;
    assert_eq!(rooted.active_atom(), Some(atoms[0]));
    assert_eq!(pending_branch.active_atom(), Some(atoms[1]));
    assert!(pending_branch.structural.atom_is_visited(atoms[1]));
    let (bond, pending_atom) = only_choice(&pending_branch, "=");
    let (atom, branch) = only_choice(&pending_atom, "O");
    let (close, restored) = only_choice(&branch, ")");
    assert_eq!(nitrogen.bond(), bonds[1]);
    let (inline, accepted) = only_choice(&restored, "N");

    assert_eq!([root, open, bond, atom, close, inline].concat(), "C(=O)N");
    assert!(accepted.is_accepted());
}

/// Explore writer paths constrained to the declared witnesses for each atom.
fn reachable_declared_witness_paths(surface: &PreparedNonStereo) -> Vec<String> {
    let initial = initial(surface);
    let mut pending = vec![(initial, String::new(), 0_usize)];
    let mut complete = Vec::new();
    let mut explored = 0_usize;

    while let Some((state, prefix, atom_offset)) = pending.pop() {
        explored += 1;
        assert!(
            explored <= 100_000,
            "writer test exceeded its exploration bound"
        );
        if state.is_accepted() {
            complete.push(normalize_test_atom_spellings(surface, &prefix));
            continue;
        }

        let choices = ring_witness_choices(&state);
        assert!(
            !choices.is_empty(),
            "writer must not dead-end before acceptance"
        );
        for choice in choices {
            let token = choice.text().to_owned();
            if !selected_atom_spelling_transition(&state, &choice, atom_offset) {
                continue;
            }
            let successor = choice.into_successor();
            let next_offset = next_atom_spelling_offset(&state, &successor, atom_offset, &token);
            pending.push((successor, format!("{prefix}{token}"), next_offset));
        }
    }
    complete
}

fn normalize_test_atom_spellings(surface: &PreparedNonStereo, text: &str) -> String {
    let mut normalized = surface.normalize_test_atom_classes(text);
    for atom in surface.molecule().graph().atom_ids() {
        if surface.atom_facts(atom).element.as_str() == "F"
            && surface.atom_facts(atom).non_graph_hydrogens == 1
        {
            normalized = normalized.replace("[FH]", surface.atom_text(atom));
        }
    }
    normalized
}

fn selected_atom_spelling_transition<S: ConstraintSolver>(
    state: &NonStereoWriterState<S>,
    choice: &Choice<NonStereoWriterState<S>>,
    atom_offset: usize,
) -> bool {
    let successor = choice.successor();
    let finished_source = state.atom_spelling.as_ref().is_some_and(|progress| {
        progress.state.is_complete()
            && declared_atom_witnesses(&state.surface, progress.atom)
                .iter()
                .any(|target| target.len() == atom_offset)
            && successor.atom_spelling.as_ref().map(|next| next.atom) != Some(progress.atom)
    });
    let atom = state
        .atom_spelling
        .as_ref()
        .filter(|_| !finished_source)
        .map(|progress| progress.atom)
        .or_else(|| {
            successor
                .atom_spelling
                .as_ref()
                .map(|progress| progress.atom)
        })
        .or_else(|| {
            let atom = successor.active_atom()?;
            (Some(atom) != state.active_atom()
                && declared_atom_witnesses(&state.surface, atom)
                    .iter()
                    .any(|target| target == &choice.text()))
            .then_some(atom)
        });
    if atom.is_none() {
        return true;
    }
    let offset = if state.is_spelling_atom() && !finished_source {
        atom_offset
    } else {
        0
    };
    let targets = match atom {
        Some(atom) => declared_atom_witnesses(&state.surface, atom),
        None => state
            .surface
            .molecule()
            .graph()
            .atom_ids()
            .flat_map(|atom| declared_atom_witnesses(&state.surface, atom))
            .collect(),
    };
    targets.into_iter().any(|target| {
        target
            .get(offset..)
            .and_then(|suffix| suffix.chars().next())
            .is_some_and(|character| choice.text() == character.to_string())
    })
}

fn reachable_strings(surface: &PreparedNonStereo) -> BTreeSet<String> {
    reachable_declared_witness_paths(surface)
        .into_iter()
        .collect()
}

fn permutations<T: Copy>(items: &[T]) -> Vec<Vec<T>> {
    fn recurse<T: Copy>(
        items: &[T],
        used: &mut [bool],
        current: &mut Vec<T>,
        output: &mut Vec<Vec<T>>,
    ) {
        if current.len() == items.len() {
            output.push(current.clone());
            return;
        }
        for index in 0..items.len() {
            if used[index] {
                continue;
            }
            used[index] = true;
            current.push(items[index]);
            recurse(items, used, current, output);
            current.pop();
            used[index] = false;
        }
    }

    if items.is_empty() {
        return vec![Vec::new()];
    }
    let mut output = Vec::new();
    recurse(
        items,
        &mut vec![false; items.len()],
        &mut Vec::with_capacity(items.len()),
        &mut output,
    );
    output
}

fn reference_tree_subtrees(
    surface: &PreparedNonStereo,
    atom: AtomId,
    parent: Option<AtomId>,
) -> BTreeSet<String> {
    let children = surface
        .molecule()
        .graph()
        .neighbors(atom)
        .expect("reference atom must exist")
        .iter()
        .copied()
        .filter(|incident| Some(incident.atom()) != parent)
        .collect::<Vec<_>>();
    let mut support = BTreeSet::new();

    for order in permutations(&children) {
        let mut partial = vec![surface.atom_text(atom).to_owned()];
        for (index, incident) in order.iter().copied().enumerate() {
            let child_support = reference_tree_subtrees(surface, incident.atom(), Some(atom));
            let bond = surface.bond_text(incident.bond(), atom);
            let bond_spellings: &[&str] = if surface.bond_tokens[incident.bond().index()]
                == NonStereoBondToken::Elided
            {
                &["", "-"]
            } else {
                &[bond]
            };
            let inline = index + 1 == order.len();
            let mut next = Vec::new();
            for prefix in &partial {
                for child in &child_support {
                    for spelling in bond_spellings {
                        if inline {
                            next.push(format!("{prefix}{spelling}{child}"));
                        } else {
                            next.push(format!("{prefix}({spelling}{child})"));
                        }
                    }
                }
            }
            partial = next;
        }
        support.extend(partial);
    }
    support
}

fn reference_tree_strings(surface: &PreparedNonStereo) -> BTreeSet<String> {
    surface
        .molecule()
        .graph()
        .atom_ids()
        .flat_map(|root| reference_tree_subtrees(surface, root, None))
        .collect()
}

fn reference_component_strings(
    surface: &PreparedNonStereo,
    component_atoms: &[AtomId],
) -> BTreeSet<String> {
    component_atoms
        .iter()
        .copied()
        .flat_map(|root| reference_tree_subtrees(surface, root, None))
        .collect()
}

fn reference_component_composition(
    surface: &PreparedNonStereo,
    components: &[Vec<AtomId>],
) -> BTreeSet<String> {
    let component_support = components
        .iter()
        .map(|atoms| reference_component_strings(surface, atoms))
        .collect::<Vec<_>>();
    let component_ids = (0..components.len()).collect::<Vec<_>>();
    let mut support = BTreeSet::new();

    for order in permutations(&component_ids) {
        let mut partial = BTreeSet::from([String::new()]);
        for component in order {
            let mut next = BTreeSet::new();
            for prefix in &partial {
                for component_text in &component_support[component] {
                    let separator = if prefix.is_empty() { "" } else { "." };
                    next.insert(format!("{prefix}{separator}{component_text}"));
                }
            }
            partial = next;
        }
        support.extend(partial);
    }
    support
}

#[test]
fn empty_and_isolated_component_semantics_are_exact() {
    let empty = fixture(&[], &[]).0;
    let empty_initial = initial(&empty);
    assert!(empty_initial.is_accepted());
    assert!(ring_witness_choices(&empty_initial).is_empty());
    assert_eq!(reachable_strings(&empty), BTreeSet::from([String::new()]));

    let two = fixture(&["A", "B"], &[]).0;
    assert_eq!(
        reachable_strings(&two),
        BTreeSet::from(["A.B".to_owned(), "B.A".to_owned()])
    );

    let distinct = fixture(&["A", "B", "C"], &[]).0;
    let distinct_paths = reachable_declared_witness_paths(&distinct);
    assert_eq!(distinct_paths.len(), 6);
    assert_eq!(distinct_paths.iter().collect::<BTreeSet<_>>().len(), 6);

    let identical = fixture(&["C", "C", "C"], &[]).0;
    let identical_paths = reachable_declared_witness_paths(&identical);
    assert_eq!(identical_paths.len(), 6);
    assert_eq!(
        identical_paths.into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from(["C.C.C".to_owned()])
    );
}

#[test]
fn branch_closure_and_component_separator_have_distinct_tokens() {
    let (surface, atoms, bonds) = fixture(
        &["A", "B", "C", "D"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let branch_incident = incident(&surface, atoms[0], bonds[0]);
    let (root, rooted) = only_root_atom(&initial, atoms[0]);
    let branch_choice = ring_witness_choices(&rooted)
        .into_iter()
        .find(|choice| {
            matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == branch_incident)
        })
        .unwrap();
    let open = branch_choice.text;
    let (branch_atom, branch) = only_choice(&branch_choice.successor, "B");
    let (close, restored) = only_choice(&branch, ")");
    let (inline_atom, boundary) = only_choice(&restored, "C");

    assert_eq!(boundary.active_atom(), None);
    assert!(boundary.labels.is_clean());
    let (separator, pending_root) = only_component_separator(&boundary, atoms[3]);
    assert_eq!(
        pending_root.pending,
        Some(PendingEmission::SegmentRootAtom(atoms[3]))
    );
    let (last_atom, accepted) = only_choice(&pending_root, "D");

    assert_eq!(
        [
            root,
            open,
            branch_atom,
            close,
            inline_atom,
            separator,
            last_atom,
        ]
        .concat(),
        "A(B)C.D"
    );
    assert!(accepted.is_accepted());
}

#[test]
fn disconnected_tree_support_matches_component_product() {
    let (surface, atoms, _) = fixture(
        &["A", "B", "C", "D", "E", "F"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (2, 3, NonStereoBondToken::Elided),
            (2, 4, NonStereoBondToken::Double),
        ],
    );
    let components = vec![
        vec![atoms[0], atoms[1]],
        vec![atoms[2], atoms[3], atoms[4]],
        vec![atoms[5]],
    ];

    assert_eq!(
        reachable_strings(&surface),
        reference_component_composition(&surface, &components)
    );
}

#[test]
fn cyclic_components_close_and_reuse_ring_label_one() {
    let (surface, atoms, bonds) = fixture(
        &["C", "C", "C", "C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
            (3, 4, NonStereoBondToken::Elided),
            (3, 5, NonStereoBondToken::Elided),
            (4, 5, NonStereoBondToken::Elided),
        ],
    );
    let initial = initial(&surface);
    let (first_root, rooted) = only_root_atom(&initial, atoms[0]);
    let first_opening = ring_witness_choices(&rooted).into_iter().next().unwrap();
    assert_eq!(first_opening.text(), "1");
    let first_open = first_opening.text;
    let (_, walked) = only_choice(&first_opening.successor, "C");
    let (_, walked) = only_choice(&walked, "C");
    let (first_close, boundary) = only_choice(&walked, "1");

    assert_eq!(boundary.active_atom(), None);
    assert!(boundary.labels.is_clean());
    assert!(!boundary.graph_is_complete());
    assert!(bonds[..3].iter().all(|bond| boundary
        .structural
        .bond_decision_domain(*bond)
        .is_singleton()));
    let (separator, pending_root) = only_component_separator(&boundary, atoms[3]);
    let (second_root, rooted) = only_choice(&pending_root, "C");
    assert!(rooted.labels.is_clean());
    let second_opening = ring_witness_choices(&rooted).into_iter().next().unwrap();
    assert_eq!(second_opening.text(), "1");
    let second_open = second_opening.text;
    let (_, walked) = only_choice(&second_opening.successor, "C");
    let (_, walked) = only_choice(&walked, "C");
    let (second_close, accepted) = only_choice(&walked, "1");

    assert_eq!(bonds.len(), 6);
    assert_eq!(
        [
            first_root,
            first_open,
            "CC".to_owned(),
            first_close,
            separator,
            second_root,
            second_open,
            "CC".to_owned(),
            second_close,
        ]
        .concat(),
        "C1CC1.C1CC1"
    );
    assert!(accepted.labels.is_clean());
    assert!(accepted.is_accepted());
    assert!(bonds.iter().all(|bond| accepted
        .structural
        .bond_decision_domain(*bond)
        .is_singleton()));

    let terminal_paths = reachable_declared_witness_paths(&surface);
    assert_eq!(terminal_paths.len(), 72 * (1 << 8));
    let triangle_spellings = ["", "-"].into_iter().flat_map(|opening| {
        ["", "-"].into_iter().flat_map(move |first| {
            ["", "-"].into_iter().flat_map(move |second| {
                ["", "-"].into_iter().map(move |closing| {
                    format!("C{opening}1{first}C{second}C{closing}1")
                })
            })
        })
    }).collect::<BTreeSet<_>>();
    let expected = triangle_spellings.iter().flat_map(|first| {
        triangle_spellings.iter().map(move |second| format!("{first}.{second}"))
    }).collect::<BTreeSet<_>>();
    assert_eq!(expected.len(), 1 << 8);
    assert_eq!(
        terminal_paths.into_iter().collect::<BTreeSet<_>>(),
        expected
    );
}

#[test]
fn connected_tree_support_remains_exact() {
    let fixtures = [
        fixture(&["C"], &[]).0,
        fixture(
            &["C", "N", "O"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (1, 2, NonStereoBondToken::Elided),
            ],
        )
        .0,
        fixture(
            &["C", "N", "O", "F"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (0, 2, NonStereoBondToken::Elided),
                (0, 3, NonStereoBondToken::Elided),
            ],
        )
        .0,
        fixture(
            &["C", "A", "O", "F", "S"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (0, 2, NonStereoBondToken::Elided),
                (1, 3, NonStereoBondToken::Elided),
                (1, 4, NonStereoBondToken::Double),
            ],
        )
        .0,
    ];
    for surface in fixtures {
        assert_eq!(
            reachable_strings(&surface),
            reference_tree_strings(&surface)
        );
    }
}

#[test]
fn connected_triangle_support_is_writer_shaped() {
    let surface = fixture(
        &["C", "C", "C"],
        &[
            (0, 1, NonStereoBondToken::Elided),
            (0, 2, NonStereoBondToken::Elided),
            (1, 2, NonStereoBondToken::Elided),
        ],
    )
    .0;

    assert_eq!(
        reachable_strings(&surface),
        ["", "-"].into_iter().flat_map(|opening| {
            ["", "-"].into_iter().flat_map(move |first| {
                ["", "-"].into_iter().flat_map(move |second| {
                    ["", "-"].into_iter().map(move |closing| {
                        format!("C{opening}1{first}C{second}C{closing}1")
                    })
                })
            })
        }).collect::<BTreeSet<_>>()
    );
}

#[test]
fn fused_and_bridged_cycles_have_complete_online_walks() {
    let fixtures = [
        fixture(
            &["A", "B", "C", "D"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (1, 2, NonStereoBondToken::Elided),
                (2, 0, NonStereoBondToken::Elided),
                (1, 3, NonStereoBondToken::Elided),
                (2, 3, NonStereoBondToken::Elided),
            ],
        )
        .0,
        fixture(
            &["A", "B", "C", "D", "E"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (1, 3, NonStereoBondToken::Elided),
                (0, 2, NonStereoBondToken::Elided),
                (2, 3, NonStereoBondToken::Elided),
                (0, 4, NonStereoBondToken::Elided),
                (4, 3, NonStereoBondToken::Elided),
            ],
        )
        .0,
    ];

    for surface in fixtures {
        assert!(!reachable_strings(&surface).is_empty());
    }
}

fn directional_chain_fixture() -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    directional_chain_fixture_with_carrier_tokens(
        NonStereoBondToken::Elided,
        NonStereoBondToken::Elided,
    )
}

fn directional_chain_fixture_with_carrier_tokens(
    left_token: NonStereoBondToken,
    right_token: NonStereoBondToken,
) -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    directional_chain_fixture_with_atoms(["F", "L", "R", "Cl"], left_token, right_token)
}

fn directional_chain_fixture_with_atoms(
    atom_texts: [&str; 4],
    left_token: NonStereoBondToken,
    right_token: NonStereoBondToken,
) -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..4)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = [
        graph.add_bond(atoms[0], atoms[1]).unwrap(),
        graph.add_bond(atoms[1], atoms[2]).unwrap(),
        graph.add_bond(atoms[2], atoms[3]).unwrap(),
    ];
    let molecule = PreparedMolecule::new(graph.build());
    let surface = PreparedNonStereo::with_atom_tokens_and_directional(
        molecule,
        atom_texts
            .map(|text| PreparedAtomToken::Fixed(text.to_owned()))
            .into(),
        vec![left_token, NonStereoBondToken::Double, right_token],
        vec![PreparedDirectionalRelation {
            double_bond: bonds[1],
            left_endpoint: atoms[1],
            left_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[0])].into_boxed_slice(),
            right_endpoint: atoms[2],
            right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[2])]
                .into_boxed_slice(),
            side_phase_xor: false,
        }],
    )
    .unwrap();
    (surface, atoms, bonds.into())
}

fn directional_tetrahedral_child_fixture() -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..7)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = [
        graph.add_bond(atoms[0], atoms[2]).unwrap(),
        graph.add_bond(atoms[0], atoms[1]).unwrap(),
        graph.add_bond(atoms[1], atoms[3]).unwrap(),
        graph.add_bond(atoms[2], atoms[4]).unwrap(),
        graph.add_bond(atoms[2], atoms[5]).unwrap(),
        graph.add_bond(atoms[2], atoms[6]).unwrap(),
    ];
    let surface = PreparedNonStereo::with_atom_tokens_and_directional(
        PreparedMolecule::new(graph.build()),
        vec![
            // The source is a symbolic role, not sulfur with its valence-dependent spelling.
            PreparedAtomToken::Fixed("Q".to_owned()),
            PreparedAtomToken::Fixed("D".to_owned()),
            PreparedAtomToken::Tetrahedral {
                reference_order: [
                    TetrahedralLigand::Bond(bonds[0]),
                    TetrahedralLigand::Bond(bonds[3]),
                    TetrahedralLigand::Bond(bonds[4]),
                    TetrahedralLigand::Bond(bonds[5]),
                ],
                text_by_parity: ["[C@]".to_owned(), "[C@@]".to_owned()],
            },
            PreparedAtomToken::Fixed("R".to_owned()),
            PreparedAtomToken::Fixed("A".to_owned()),
            PreparedAtomToken::Fixed("B".to_owned()),
            PreparedAtomToken::Fixed("C".to_owned()),
        ],
        vec![
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
        ],
        vec![PreparedDirectionalRelation {
            double_bond: bonds[1],
            left_endpoint: atoms[0],
            left_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[0])].into_boxed_slice(),
            right_endpoint: atoms[1],
            right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[2])]
                .into_boxed_slice(),
            side_phase_xor: false,
        }],
    )
    .unwrap();
    (surface, atoms, bonds.into())
}

fn directional_tetrahedral_parent_fixture() -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..6)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = [
        graph.add_bond(atoms[0], atoms[2]).unwrap(),
        graph.add_bond(atoms[0], atoms[1]).unwrap(),
        graph.add_bond(atoms[1], atoms[3]).unwrap(),
        graph.add_bond(atoms[0], atoms[4]).unwrap(),
        graph.add_bond(atoms[0], atoms[5]).unwrap(),
    ];
    let surface = PreparedNonStereo::with_atom_tokens_and_directional(
        PreparedMolecule::new(graph.build()),
        vec![
            PreparedAtomToken::Tetrahedral {
                reference_order: [
                    TetrahedralLigand::Bond(bonds[0]),
                    TetrahedralLigand::Bond(bonds[1]),
                    TetrahedralLigand::Bond(bonds[3]),
                    TetrahedralLigand::Bond(bonds[4]),
                ],
                text_by_parity: ["[C@]".to_owned(), "[C@@]".to_owned()],
            },
            PreparedAtomToken::Fixed("D".to_owned()),
            PreparedAtomToken::Fixed("T".to_owned()),
            PreparedAtomToken::Fixed("R".to_owned()),
            PreparedAtomToken::Fixed("A".to_owned()),
            PreparedAtomToken::Fixed("B".to_owned()),
        ],
        vec![
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
        ],
        vec![PreparedDirectionalRelation {
            double_bond: bonds[1],
            left_endpoint: atoms[0],
            left_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[0])].into_boxed_slice(),
            right_endpoint: atoms[1],
            right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[2])]
                .into_boxed_slice(),
            side_phase_xor: false,
        }],
    )
    .unwrap();
    (surface, atoms, bonds.into())
}

#[test]
fn directional_marks_render_from_canonical_fixed_endpoints() {
    assert_eq!(CarrierMark::SlashAtFixedA.directional_text(true), Some("/"));
    assert_eq!(
        CarrierMark::SlashAtFixedA.directional_text(false),
        Some("\\")
    );
    assert_eq!(
        CarrierMark::BackslashAtFixedA.directional_text(true),
        Some("\\")
    );
    assert_eq!(
        CarrierMark::BackslashAtFixedA.directional_text(false),
        Some("/")
    );
    assert_eq!(CarrierMark::Plain.directional_text(true), None);

    let (surface, atoms, bonds) = directional_chain_fixture();
    assert!(endpoint_flip(
        surface.molecule().graph(),
        bonds[0],
        atoms[1]
    ));
    assert!(!endpoint_flip(
        surface.molecule().graph(),
        bonds[2],
        atoms[2]
    ));
}

fn directional_hexagon_surface(
    side_phase_xors: [bool; 3],
) -> Result<PreparedNonStereo, PreparedNonStereoError> {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..6)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = (0..6)
        .map(|index| {
            graph
                .add_bond(atoms[index], atoms[(index + 1) % 6])
                .unwrap()
        })
        .collect::<Vec<_>>();
    PreparedNonStereo::with_atom_tokens_and_directional(
        PreparedMolecule::new(graph.build()),
        (0..6)
            .map(|index| PreparedAtomToken::Fixed(format!("A{index}")))
            .collect(),
        vec![
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
        ],
        vec![
            PreparedDirectionalRelation {
                double_bond: bonds[1],
                left_endpoint: atoms[1],
                left_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[0])]
                    .into_boxed_slice(),
                right_endpoint: atoms[2],
                right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[2])]
                    .into_boxed_slice(),
                side_phase_xor: side_phase_xors[0],
            },
            PreparedDirectionalRelation {
                double_bond: bonds[3],
                left_endpoint: atoms[3],
                left_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[2])]
                    .into_boxed_slice(),
                right_endpoint: atoms[4],
                right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[4])]
                    .into_boxed_slice(),
                side_phase_xor: side_phase_xors[1],
            },
            PreparedDirectionalRelation {
                double_bond: bonds[5],
                left_endpoint: atoms[5],
                left_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[4])]
                    .into_boxed_slice(),
                right_endpoint: atoms[0],
                right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[0])]
                    .into_boxed_slice(),
                side_phase_xor: side_phase_xors[2],
            },
        ],
    )
}

#[test]
fn preparation_rejects_contradictory_directional_cycles_and_flattens_consistent_ones() {
    assert!(matches!(
        directional_hexagon_surface([false, false, false]),
        Err(PreparedNonStereoError::ContradictoryDirectionalParity { .. })
    ));
    let consistent = directional_hexagon_surface([true, false, false]).unwrap();
    assert_eq!(
        consistent.molecule().constraint_model().factor_count(),
        13 + consistent.ring_count_factors.iter().flatten().count(),
        "three pattern-to-left, pattern-to-right, and cross-side relations plus one spanning factor"
    );
    let model = consistent.molecule().constraint_model_arc();
    let native = <NativeSolverState as ConstraintSolver>::initial(Arc::clone(&model))
        .unwrap()
        .unwrap_consistent();
    let exhaustive = <ExhaustiveSolverState as ConstraintSolver>::initial(Arc::clone(&model))
        .unwrap()
        .unwrap_consistent();
    assert_eq!(native.exact_run_counts(), (1, 0));
    for index in 0..model.variable_count() {
        let variable = crate::ids::VariableId::new(index.try_into().unwrap());
        assert_eq!(native.domain_bound(variable), exhaustive.domain_bound(variable));
    }
}

#[test]
fn directional_side_patterns_propagate_marks_without_touching_roles_or_exact_search() {
    let (surface, _, bonds) = directional_chain_fixture();
    let PreparedDirectionalBondStatus::CompiledSite(left) = surface.directional_status(bonds[0])
    else {
        panic!("left carrier must be compiled");
    };
    let PreparedDirectionalBondStatus::CompiledSite(right) = surface.directional_status(bonds[2])
    else {
        panic!("right carrier must be compiled");
    };
    assert_ne!(left.mark_variable, right.mark_variable);

    let initial = NativeSolverState::initial(surface.molecule().constraint_model_arc()).unwrap();
    assert_eq!(initial.exact_run_counts(), (0, 0));
    let source_bond_domains = bonds
        .iter()
        .map(|bond| {
            initial
                .domain_bound(surface.molecule().bond_decision_variable(*bond).unwrap())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let successor = NativeSolverState::with_restrictions(
        &initial,
        [(
            left.mark_variable,
            CarrierMark::SlashAtFixedA.singleton_domain(),
        )],
    )
    .unwrap();

    assert_eq!(
        successor.domain_bound(left.mark_variable),
        Some(CarrierMark::SlashAtFixedA.singleton_domain())
    );
    assert_eq!(
        successor.domain_bound(right.mark_variable),
        Some(CarrierMark::BackslashAtFixedA.singleton_domain())
    );
    assert_eq!(successor.exact_run_counts(), (0, 0));
    assert_eq!(
        bonds
            .iter()
            .map(|bond| successor
                .domain_bound(surface.molecule().bond_decision_variable(*bond).unwrap())
                .unwrap())
            .collect::<Vec<_>>(),
        source_bond_domains
    );
}

#[test]
fn shared_directional_carrier_reuses_one_sign_variable() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..6)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = (0..5)
        .map(|index| graph.add_bond(atoms[index], atoms[index + 1]).unwrap())
        .collect::<Vec<_>>();
    let surface = PreparedNonStereo::with_atom_tokens_and_directional(
        PreparedMolecule::new(graph.build()),
        (0..6)
            .map(|index| PreparedAtomToken::Fixed(format!("A{index}")))
            .collect(),
        vec![
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
        ],
        vec![
            PreparedDirectionalRelation {
                double_bond: bonds[1],
                left_endpoint: atoms[1],
                left_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[0])]
                    .into_boxed_slice(),
                right_endpoint: atoms[2],
                right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[2])]
                    .into_boxed_slice(),
                side_phase_xor: false,
            },
            PreparedDirectionalRelation {
                double_bond: bonds[3],
                left_endpoint: atoms[3],
                left_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[2])]
                    .into_boxed_slice(),
                right_endpoint: atoms[4],
                right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[4])]
                    .into_boxed_slice(),
                side_phase_xor: true,
            },
        ],
    )
    .unwrap();

    let variables = [bonds[0], bonds[2], bonds[4]].map(|bond| {
        let PreparedDirectionalBondStatus::CompiledSite(site) = surface.directional_status(bond)
        else {
            panic!("shared relation carriers must be compiled");
        };
        site.mark_variable
    });
    assert_eq!(variables.into_iter().collect::<BTreeSet<_>>().len(), 3);
    assert_eq!(
        surface.molecule().constraint_model().factor_count(),
        10 + surface.ring_count_factors.iter().flatten().count(),
        "two side-pattern triples plus one molecular spanning factor"
    );
}

fn shared_directional_chain_fixture() -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..6)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = (0..5)
        .map(|index| graph.add_bond(atoms[index], atoms[index + 1]).unwrap())
        .collect::<Vec<_>>();
    let surface = PreparedNonStereo::with_atom_tokens_and_directional(
        PreparedMolecule::new(graph.build()),
        ["A", "B", "C", "D", "E", "F"]
            .map(|text| PreparedAtomToken::Fixed(text.to_owned()))
            .into(),
        vec![
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
        ],
        vec![
            PreparedDirectionalRelation {
                double_bond: bonds[1],
                left_endpoint: atoms[1],
                left_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[0])]
                    .into_boxed_slice(),
                right_endpoint: atoms[2],
                right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[2])]
                    .into_boxed_slice(),
                side_phase_xor: false,
            },
            PreparedDirectionalRelation {
                double_bond: bonds[3],
                left_endpoint: atoms[3],
                left_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[2])]
                    .into_boxed_slice(),
                right_endpoint: atoms[4],
                right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[4])]
                    .into_boxed_slice(),
                side_phase_xor: true,
            },
        ],
    )
    .unwrap();
    (surface, atoms, bonds)
}

fn disconnected_directional_fixture() -> (PreparedNonStereo, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..8)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let mut bonds = Vec::new();
    for offset in [0, 4] {
        bonds.push(graph.add_bond(atoms[offset], atoms[offset + 1]).unwrap());
        bonds.push(
            graph
                .add_bond(atoms[offset + 1], atoms[offset + 2])
                .unwrap(),
        );
        bonds.push(
            graph
                .add_bond(atoms[offset + 2], atoms[offset + 3])
                .unwrap(),
        );
    }
    let relations = [0, 1]
        .map(|component| {
            let atom_offset = component * 4;
            let bond_offset = component * 3;
            PreparedDirectionalRelation {
                double_bond: bonds[bond_offset + 1],
                left_endpoint: atoms[atom_offset + 1],
                left_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[bond_offset])]
                    .into_boxed_slice(),
                right_endpoint: atoms[atom_offset + 2],
                right_carriers: vec![PreparedDirectionalCarrier::unflipped(
                    bonds[bond_offset + 2],
                )]
                .into_boxed_slice(),
                side_phase_xor: component == 1,
            }
        })
        .into();
    let surface = PreparedNonStereo::with_atom_tokens_and_directional(
        PreparedMolecule::new(graph.build()),
        (0..8)
            .map(|index| PreparedAtomToken::Fixed(format!("A{index}")))
            .collect(),
        vec![
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
        ],
        relations,
    )
    .unwrap();
    (surface, bonds)
}

#[test]
fn one_visible_sign_choice_propagates_the_complete_shared_carrier_component() {
    let (surface, atoms, bonds) = shared_directional_chain_fixture();
    let (_, rooted) = only_root_atom(&initial(&surface), atoms[0]);
    let slash = ring_witness_choices(&rooted)
        .into_iter()
        .find(|choice| choice.text() == "/")
        .unwrap()
        .into_successor();

    let domains = [bonds[0], bonds[2], bonds[4]].map(|bond| {
        let PreparedDirectionalBondStatus::CompiledSite(site) = surface.directional_status(bond)
        else {
            panic!("shared carrier must be compiled");
        };
        slash.structural.semantic_domain(site.mark_variable)
    });
    assert!(domains.into_iter().all(Domain::is_singleton));
}

#[test]
fn shared_site_walk_exposes_only_the_propagated_future_signs() {
    let (surface, atoms, _) = shared_directional_chain_fixture();
    let (_, rooted) = only_root_atom(&initial(&surface), atoms[0]);
    let mut state = ring_witness_choices(&rooted)
        .into_iter()
        .find(|choice| choice.text() == "/")
        .unwrap()
        .into_successor();
    (_, state) = only_choice(&state, "B");
    (_, state) = only_choice(&state, "=");
    (_, state) = only_choice(&state, "C");

    let middle_signs = ring_witness_choices(&state);
    assert_eq!(middle_signs.len(), 1);
    assert!(matches!(middle_signs[0].text(), "/" | "\\"));
    state = middle_signs.into_iter().next().unwrap().into_successor();
    (_, state) = only_choice(&state, "D");
    (_, state) = only_choice(&state, "=");
    (_, state) = only_choice(&state, "E");

    let outer_signs = ring_witness_choices(&state);
    assert_eq!(outer_signs.len(), 1);
    assert!(matches!(outer_signs[0].text(), "/" | "\\"));
    state = outer_signs.into_iter().next().unwrap().into_successor();
    (_, state) = only_choice(&state, "F");
    assert!(state.is_accepted());
}

#[test]
fn one_candidate_side_patterns_match_exhaustive_projection_for_every_unary_restriction() {
    let (surface, _, bonds) = shared_directional_chain_fixture();
    let variables = [bonds[0], bonds[2], bonds[4]].map(|bond| {
        let PreparedDirectionalBondStatus::CompiledSite(site) = surface.directional_status(bond)
        else {
            panic!("shared carrier must be compiled");
        };
        site.mark_variable
    });
    let model = surface.molecule().constraint_model_arc();

    for code in 0..27_u32 {
        let mut remaining = code;
        let mut restrictions = Vec::new();
        for variable in variables {
            match remaining % 3 {
                0 => {}
                1 => restrictions.push((variable, CarrierMark::SlashAtFixedA.singleton_domain())),
                2 => {
                    restrictions.push((variable, CarrierMark::BackslashAtFixedA.singleton_domain()))
                }
                _ => unreachable!(),
            }
            remaining /= 3;
        }
        let native = <NativeSolverState as ConstraintSolver>::initial(Arc::clone(&model))
            .unwrap()
            .unwrap_consistent()
            .restricted(&restrictions)
            .unwrap();
        let exhaustive = <ExhaustiveSolverState as ConstraintSolver>::initial(Arc::clone(&model))
            .unwrap()
            .unwrap_consistent()
            .restricted(&restrictions)
            .unwrap();
        match (native, exhaustive) {
            (Consistency::Contradiction, Consistency::Contradiction) => {}
            (Consistency::Consistent(native), Consistency::Consistent(exhaustive)) => {
                for variable in variables {
                    assert_eq!(native.domain_bound(variable), exhaustive.domain_bound(variable));
                }
                assert_eq!(native.exact_run_counts(), (0, 0));
            }
            _ => panic!("native and exhaustive directional projections disagree for {code}"),
        }
    }
}

#[test]
fn selectable_side_patterns_match_exhaustive_projection_for_every_mark_domain() {
    let (surface, _, bonds) =
        selectable_directional_fixture(NonStereoBondToken::Elided, NonStereoBondToken::Single);
    let variables = [bonds[0], bonds[3], bonds[2]].map(|bond| {
        let PreparedDirectionalBondStatus::CompiledSite(site) = surface.directional_status(bond)
        else {
            panic!("selectable carrier must be compiled");
        };
        site.mark_variable
    });
    let model = surface.molecule().constraint_model_arc();

    for code in 0..343_u32 {
        let mut remaining = code;
        let restrictions = variables.map(|variable| {
            let domain = Domain::from_bits(1_u64 + u64::from(remaining % 7));
            remaining /= 7;
            (variable, domain)
        });
        let native = <NativeSolverState as ConstraintSolver>::initial(Arc::clone(&model))
            .unwrap()
            .unwrap_consistent()
            .restricted(&restrictions)
            .unwrap();
        let exhaustive = <ExhaustiveSolverState as ConstraintSolver>::initial(Arc::clone(&model))
            .unwrap()
            .unwrap_consistent()
            .restricted(&restrictions)
            .unwrap();
        match (native, exhaustive) {
            (Consistency::Contradiction, Consistency::Contradiction) => {}
            (Consistency::Consistent(native), Consistency::Consistent(exhaustive)) => {
                for index in 0..model.variable_count() {
                    let variable = crate::ids::VariableId::new(index.try_into().unwrap());
                    assert_eq!(native.domain_bound(variable), exhaustive.domain_bound(variable));
                }
                assert_eq!(native.exact_run_counts(), (0, 0));
            }
            _ => panic!("native and exhaustive selectable projections disagree for {code}"),
        }
    }
}

#[test]
fn disconnected_directional_components_remain_constraint_and_writer_local() {
    let (surface, bonds) = disconnected_directional_fixture();
    let sites = [bonds[0], bonds[2], bonds[3], bonds[5]].map(|bond| {
        let PreparedDirectionalBondStatus::CompiledSite(site) = surface.directional_status(bond)
        else {
            panic!("disconnected carrier must be compiled");
        };
        site
    });
    let native_initial =
        NativeSolverState::initial(surface.molecule().constraint_model_arc()).unwrap();
    let restricted = NativeSolverState::with_restrictions(
        &native_initial,
        [(
            sites[0].mark_variable,
            CarrierMark::SlashAtFixedA.singleton_domain(),
        )],
    )
    .unwrap();
    assert!(restricted
        .domain_bound(sites[1].mark_variable)
        .unwrap()
        .is_singleton());
    assert_eq!(
        restricted.domain_bound(sites[2].mark_variable),
        Some(CarrierMark::marked_domain())
    );
    assert_eq!(
        restricted.domain_bound(sites[3].mark_variable),
        Some(CarrierMark::marked_domain())
    );

    let terminals = reachable_terminal_states(initial(&surface));
    assert!(!terminals.is_empty());
    assert!(terminals.iter().all(State::is_accepted));
}

#[test]
fn multi_carrier_region_is_admitted_but_marked_incomplete() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..6)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = [
        graph.add_bond(atoms[0], atoms[1]).unwrap(),
        graph.add_bond(atoms[1], atoms[2]).unwrap(),
        graph.add_bond(atoms[2], atoms[3]).unwrap(),
        graph.add_bond(atoms[1], atoms[4]).unwrap(),
        graph.add_bond(atoms[1], atoms[5]).unwrap(),
    ];
    let surface = PreparedNonStereo::with_atom_tokens_and_directional(
        PreparedMolecule::new(graph.build()),
        ["F", "C", "C", "F", "Cl", "Br"]
            .map(|text| PreparedAtomToken::Fixed(text.to_owned()))
            .into(),
        vec![
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Single,
            NonStereoBondToken::Elided,
        ],
        vec![PreparedDirectionalRelation {
            double_bond: bonds[1],
            left_endpoint: atoms[1],
            left_carriers: vec![
                PreparedDirectionalCarrier::unflipped(bonds[0]),
                PreparedDirectionalCarrier::unflipped(bonds[3]),
                PreparedDirectionalCarrier::unflipped(bonds[4]),
            ]
            .into_boxed_slice(),
            right_endpoint: atoms[2],
            right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[2])]
                .into_boxed_slice(),
            side_phase_xor: false,
        }],
    )
    .unwrap();
    assert!(matches!(
        surface.directional_status(bonds[0]),
        PreparedDirectionalBondStatus::IncompleteSelection(_)
    ));
    assert!(matches!(
        surface.directional_status(bonds[3]),
        PreparedDirectionalBondStatus::IncompleteSelection(_)
    ));
    assert!(matches!(
        surface.directional_status(bonds[2]),
        PreparedDirectionalBondStatus::IncompleteSelection(_)
    ));
    let (_, rooted) = only_root_atom(&initial(&surface), atoms[0]);
    let before = rooted.observe_raw();
    assert!(matches!(
        rooted.choices(),
        Err(ChoiceFailure::Incomplete(
            WriterIncompleteness::DirectionalCarrierSelection {
                double_bond,
                endpoint,
            }
        )) if double_bond == bonds[1] && endpoint == atoms[1]
    ));
    assert_eq!(rooted.observe_raw(), before);
}

#[test]
fn two_candidate_side_prepares_six_live_patterns_without_mixed_search() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..5)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = [
        graph.add_bond(atoms[0], atoms[1]).unwrap(),
        graph.add_bond(atoms[1], atoms[2]).unwrap(),
        graph.add_bond(atoms[2], atoms[3]).unwrap(),
        graph.add_bond(atoms[1], atoms[4]).unwrap(),
    ];
    let surface = PreparedNonStereo::with_atom_tokens_and_directional(
        PreparedMolecule::new(graph.build()),
        ["F", "L", "R", "Cl", "Br"]
            .map(|text| PreparedAtomToken::Fixed(text.to_owned()))
            .into(),
        vec![
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Single,
        ],
        vec![PreparedDirectionalRelation {
            double_bond: bonds[1],
            left_endpoint: atoms[1],
            left_carriers: vec![
                PreparedDirectionalCarrier::unflipped(bonds[0]),
                PreparedDirectionalCarrier {
                    bond: bonds[3],
                    side_flip: true,
                },
            ]
            .into_boxed_slice(),
            right_endpoint: atoms[2],
            right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[2])]
                .into_boxed_slice(),
            side_phase_xor: false,
        }],
    )
    .unwrap();

    let left = surface
        .directional_sides
        .iter()
        .find(|side| side.endpoint == atoms[1])
        .unwrap();
    let right = surface
        .directional_sides
        .iter()
        .find(|side| side.endpoint == atoms[2])
        .unwrap();
    assert_eq!(left.double_bond, bonds[1]);
    assert_eq!(left.patterns.len(), 6);
    assert_eq!(right.patterns.len(), 2);
    assert!(left.patterns.iter().all(|pattern| {
        pattern.marks.iter().any(|mark| *mark != CarrierMark::Plain)
            && pattern
                .marks
                .iter()
                .zip(&left.carriers)
                .filter_map(|(mark, carrier)| {
                    mark.canonical_sign().map(|sign| {
                        sign ^ endpoint_flip(
                            surface.molecule().graph(),
                            carrier.bond,
                            left.endpoint,
                        ) ^ carrier.side_flip
                    })
                })
                .all(|phase| phase == pattern.phase)
    }));

    let native = NativeSolverState::initial(surface.molecule().constraint_model_arc()).unwrap();
    assert_eq!(native.exact_run_counts(), (0, 0));
    for bond in [bonds[0], bonds[3]] {
        let PreparedDirectionalBondStatus::CompiledSite(site) = surface.directional_status(bond)
        else {
            panic!("two-candidate sides must compile physical mark variables")
        };
        assert_eq!(
            native.domain_bound(site.mark_variable),
            Some(CarrierMark::domain())
        );
    }
}

fn selectable_directional_fixture(
    first_token: NonStereoBondToken,
    second_token: NonStereoBondToken,
) -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..5)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = [
        graph.add_bond(atoms[0], atoms[1]).unwrap(),
        graph.add_bond(atoms[1], atoms[2]).unwrap(),
        graph.add_bond(atoms[2], atoms[3]).unwrap(),
        graph.add_bond(atoms[1], atoms[4]).unwrap(),
    ];
    let surface = PreparedNonStereo::with_atom_tokens_and_directional(
        PreparedMolecule::new(graph.build()),
        ["F", "L", "R", "Cl", "Br"]
            .map(|text| PreparedAtomToken::Fixed(text.to_owned()))
            .into(),
        vec![
            first_token,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            second_token,
        ],
        vec![PreparedDirectionalRelation {
            double_bond: bonds[1],
            left_endpoint: atoms[1],
            left_carriers: vec![
                PreparedDirectionalCarrier::unflipped(bonds[0]),
                PreparedDirectionalCarrier {
                    bond: bonds[3],
                    side_flip: true,
                },
            ]
            .into_boxed_slice(),
            right_endpoint: atoms[2],
            right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[2])]
                .into_boxed_slice(),
            side_phase_xor: false,
        }],
    )
    .unwrap();
    (surface, atoms, bonds.into())
}

fn selectable_tetrahedral_child_fixture() -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..5)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = [
        graph.add_bond(atoms[0], atoms[1]).unwrap(),
        graph.add_bond(atoms[1], atoms[2]).unwrap(),
        graph.add_bond(atoms[2], atoms[3]).unwrap(),
        graph.add_bond(atoms[1], atoms[4]).unwrap(),
    ];
    let surface = PreparedNonStereo::with_atom_tokens_and_directional(
        PreparedMolecule::new(graph.build()),
        vec![
            PreparedAtomToken::Fixed("F".to_owned()),
            PreparedAtomToken::Tetrahedral {
                reference_order: [
                    TetrahedralLigand::Bond(bonds[0]),
                    TetrahedralLigand::VirtualHydrogen,
                    TetrahedralLigand::Bond(bonds[1]),
                    TetrahedralLigand::Bond(bonds[3]),
                ],
                text_by_parity: ["[C@H]".to_owned(), "[C@@H]".to_owned()],
            },
            PreparedAtomToken::Fixed("R".to_owned()),
            PreparedAtomToken::Fixed("Cl".to_owned()),
            PreparedAtomToken::Fixed("Br".to_owned()),
        ],
        vec![
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Single,
        ],
        vec![PreparedDirectionalRelation {
            double_bond: bonds[1],
            left_endpoint: atoms[1],
            left_carriers: vec![
                PreparedDirectionalCarrier::unflipped(bonds[0]),
                PreparedDirectionalCarrier {
                    bond: bonds[3],
                    side_flip: true,
                },
            ]
            .into_boxed_slice(),
            right_endpoint: atoms[2],
            right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[2])]
                .into_boxed_slice(),
            side_phase_xor: false,
        }],
    )
    .unwrap();
    (surface, atoms, bonds.into())
}

fn selectable_directional_ring_fixture() -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    selectable_directional_ring_fixture_with_token(NonStereoBondToken::Elided)
}

fn selectable_directional_ring_fixture_with_token(
    carrier_token: NonStereoBondToken,
) -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..5)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = [
        graph.add_bond(atoms[0], atoms[1]).unwrap(),
        graph.add_bond(atoms[1], atoms[2]).unwrap(),
        graph.add_bond(atoms[2], atoms[3]).unwrap(),
        graph.add_bond(atoms[1], atoms[4]).unwrap(),
        graph.add_bond(atoms[0], atoms[4]).unwrap(),
    ];
    let surface = PreparedNonStereo::with_atom_tokens_and_directional(
        PreparedMolecule::new(graph.build()),
        ["A", "L", "R", "X", "B"]
            .map(|text| PreparedAtomToken::Fixed(text.to_owned()))
            .into(),
        vec![
            carrier_token,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
        ],
        vec![PreparedDirectionalRelation {
            double_bond: bonds[1],
            left_endpoint: atoms[1],
            left_carriers: vec![
                PreparedDirectionalCarrier::unflipped(bonds[0]),
                PreparedDirectionalCarrier {
                    bond: bonds[3],
                    side_flip: true,
                },
            ]
            .into_boxed_slice(),
            right_endpoint: atoms[2],
            right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[2])]
                .into_boxed_slice(),
            side_phase_xor: false,
        }],
    )
    .unwrap();
    (surface, atoms, bonds.into())
}

#[test]
fn explicit_directional_ring_carriers_offer_plain_and_marked_endpoint_tokens() {
    for (token, plain_text) in [
        (NonStereoBondToken::Single, "-"),
        (NonStereoBondToken::Aromatic, ":"),
    ] {
        let (surface, atoms, bonds) = selectable_directional_ring_fixture_with_token(token);
        for (root, endpoint_is_a) in [(atoms[0], true), (atoms[1], false)] {
            let (_, rooted) = only_root_atom(&initial(&surface), root);
            let choices = ring_witness_choices(&rooted)
                .into_iter()
                .filter(|choice| {
                    matches!(
                        choice.successor().observe_raw().structural.traversal.bond_progress[bonds[0].index()],
                        ObservedBondProgress::RingOpen { first_endpoint } if first_endpoint == root
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(
                choices.iter().map(Choice::text).collect::<BTreeSet<_>>(),
                BTreeSet::from(["1", plain_text, "/", "\\"])
            );
            let PreparedDirectionalBondStatus::CompiledSite(site) =
                surface.directional_status(bonds[0])
            else {
                panic!("directional ring carrier must compile");
            };
            for choice in choices
                .iter()
                .filter(|choice| matches!(choice.text(), "/" | "\\"))
            {
                let expected_mark = if (choice.text() == "/") == endpoint_is_a {
                    CarrierMark::SlashAtFixedA
                } else {
                    CarrierMark::BackslashAtFixedA
                };
                assert_eq!(
                    choice
                        .successor()
                        .structural
                        .semantic_domain(site.mark_variable),
                    expected_mark.singleton_domain()
                );
            }
        }
    }
}

#[test]
fn explicit_plain_directional_ring_plans_render_at_either_or_both_endpoints() {
    for (token, glyph) in [
        (NonStereoBondToken::Single, '-'),
        (NonStereoBondToken::Aromatic, ':'),
    ] {
        let (surface, _, bonds) = selectable_directional_ring_fixture_with_token(token);
        let PreparedDirectionalBondStatus::CompiledSite(site) =
            surface.directional_status(bonds[0])
        else {
            panic!("directional ring carrier must compile");
        };
        let mut pending = vec![(String::new(), initial(&surface), 0_usize)];
        let mut observed = BTreeSet::new();
        let mut explored = 0_usize;
        while let Some((text, state, atom_offset)) = pending.pop() {
            explored += 1;
            assert!(
                explored <= 100_000,
                "explicit ring fixture exceeded its bound"
            );
            if state.is_accepted() {
                let plan = state.structural.semantic_domain(site.ring_plan_variable);
                for candidate in [
                    DirectionalRingPlan::PlainAtA,
                    DirectionalRingPlan::PlainAtB,
                    DirectionalRingPlan::PlainAtBoth,
                ] {
                    if plan == candidate.singleton_domain() {
                        let expected_count =
                            usize::from(candidate == DirectionalRingPlan::PlainAtBoth) + 1;
                        let ring_glyph_count = text
                            .match_indices(&format!("{glyph}1"))
                            .filter(|(index, _)| text[index + 2..].chars().next() != Some(']'))
                            .count();
                        assert_eq!(ring_glyph_count, expected_count, "ring output: {text:?}");
                        observed.insert(candidate.value_index());
                    }
                }
                continue;
            }
            for choice in ring_witness_choices(&state) {
                if !selected_atom_spelling_transition(&state, &choice, atom_offset) {
                    continue;
                }
                let token = choice.text().to_owned();
                let successor = choice.into_successor();
                let next_offset =
                    next_atom_spelling_offset(&state, &successor, atom_offset, &token);
                pending.push((format!("{text}{token}"), successor, next_offset));
            }
        }
        assert_eq!(
            observed,
            BTreeSet::from([
                DirectionalRingPlan::PlainAtA.value_index(),
                DirectionalRingPlan::PlainAtB.value_index(),
                DirectionalRingPlan::PlainAtBoth.value_index(),
            ])
        );
    }
}

#[test]
fn selectable_inline_elision_exposes_plain_atom_and_directional_tokens() {
    let (surface, atoms, bonds) =
        selectable_directional_fixture(NonStereoBondToken::Elided, NonStereoBondToken::Single);
    let (_, rooted) = only_root_atom(&initial(&surface), atoms[0]);

    let choices = ring_witness_choices(&rooted);
    let directional = choices
        .iter()
        .filter(|choice| matches!(choice.text(), "/" | "\\"))
        .collect::<Vec<_>>();
    assert_eq!(
        directional
            .iter()
            .map(|choice| choice.text())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["/", "\\"])
    );
    let site = match surface.directional_status(bonds[0]) {
        PreparedDirectionalBondStatus::CompiledSite(site) => site,
        _ => panic!("selectable carrier must compile"),
    };
    let plain = choices
        .iter()
        .flat_map(|choice| atom_witness_completions_after_choice(choice, atoms[1]))
        .collect::<Vec<_>>();
    assert!(!plain.is_empty(), "the child AtomId has a declared witness");
    assert!(plain.iter().all(|(_, state)| {
        state.structural.semantic_domain(site.mark_variable)
            == CarrierMark::Plain.singleton_domain()
    }));
    for choice in directional {
        assert!(matches!(&choice.successor().pending,
            Some(PendingEmission::InlineAtom(entry))
                if entry.incident == incident(&surface, atoms[0], bonds[0])));
    }
}

#[test]
fn selectable_explicit_plain_and_directional_tokens_share_one_inline_frontier() {
    let (surface, atoms, bonds) =
        selectable_directional_fixture(NonStereoBondToken::Elided, NonStereoBondToken::Single);
    let (_, rooted) = only_root_atom(&initial(&surface), atoms[4]);

    let choices = ring_witness_choices(&rooted);
    assert_eq!(
        choices.iter().map(Choice::text).collect::<BTreeSet<_>>(),
        BTreeSet::from(["-", "/", "\\"])
    );
    assert!(choices.iter().all(|choice| {
        matches!(&choice.successor().pending, Some(PendingEmission::InlineAtom(entry)) if entry.incident == incident(
                &surface, atoms[4], bonds[3],
            ))
    }));
}

#[test]
fn contradictory_explicit_plain_mark_does_not_suppress_directional_siblings() {
    let (surface, atoms, bonds) =
        selectable_directional_fixture(NonStereoBondToken::Elided, NonStereoBondToken::Single);
    let PreparedDirectionalBondStatus::CompiledSite(site) = surface.directional_status(bonds[3])
    else {
        panic!("selectable carrier must compile");
    };
    REJECT_DIRECTIONAL_MARK_VARIABLE.store(site.mark_variable.index(), Ordering::Relaxed);
    let initial = NonStereoWriterState::<RejectPlainDirectionalMarkSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let (_, rooted) = only_root_atom(&initial, atoms[4]);
    let before = rooted.observe_raw();
    assert_eq!(
        ring_witness_choices(&rooted)
            .iter()
            .map(Choice::text)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["/", "\\"])
    );
    assert_eq!(rooted.observe_raw(), before);
}

#[test]
fn selectable_directional_ring_walks_have_bounded_complete_suffixes() {
    let (surface, _, bonds) = selectable_directional_ring_fixture();
    let terminals = reachable_terminal_states(initial(&surface));
    assert!(!terminals.is_empty());

    for terminal in terminals {
        assert!(terminal.is_accepted());
        assert!(terminal.labels.is_clean());
        for bond in [bonds[0], bonds[2]] {
            let PreparedDirectionalBondStatus::CompiledSite(site) =
                surface.directional_status(bond)
            else {
                panic!("selectable ring carrier must remain compiled");
            };
            assert!(terminal
                .structural
                .semantic_domain(site.mark_variable)
                .is_singleton());
            let role = terminal.structural.bond_decision_domain(bond);
            if role == BondRole::Ring.singleton_domain() {
                assert!(terminal.structural.factor_is_active(site.ring_plan_factor));
                assert!(terminal
                    .structural
                    .semantic_domain(site.ring_plan_variable)
                    .is_singleton());
            }
        }
        assert!(surface.directional_sides.iter().all(|side| terminal
            .structural
            .semantic_domain(side.pattern_variable)
            .is_singleton()));
    }
}

#[test]
fn selectable_branch_parenthesis_commits_plain_or_directional_mark() {
    let (surface, atoms, bonds) =
        selectable_directional_fixture(NonStereoBondToken::Elided, NonStereoBondToken::Single);
    let (_, rooted) = only_root_atom(&initial(&surface), atoms[1]);
    let branches = ring_witness_choices(&rooted)
        .into_iter()
        .filter(|choice| {
            choice.text() == "("
                && matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == incident(
                        &surface, atoms[1], bonds[0],
                    ))
        })
        .map(Choice::into_successor)
        .collect::<Vec<_>>();
    let site = match surface.directional_status(bonds[0]) {
        PreparedDirectionalBondStatus::CompiledSite(site) => site,
        _ => panic!("selectable carrier must compile"),
    };
    assert_eq!(branches.len(), 3);
    let mut marks = Domain::empty();
    let mut directional = BTreeSet::new();
    for branch in branches {
        let mark = branch.structural.semantic_domain(site.mark_variable);
        assert!(mark.is_singleton());
        marks = marks.union(mark);
        if mark == CarrierMark::Plain.singleton_domain() {
            let plain = completed_atom_witnesses(&branch, atoms[0]);
            assert_eq!(plain.len(), 1);
            assert!(plain[0].1.pending.is_none());
        } else {
            let choices = ring_witness_choices(&branch);
            assert_eq!(choices.len(), 1);
            directional.insert(choices[0].text().to_owned());
        }
    }
    assert_eq!(marks, CarrierMark::domain());
    assert_eq!(directional, BTreeSet::from(["/".to_owned(), "\\".to_owned()]));
}

#[test]
fn selectable_directional_token_preserves_tetrahedral_child_parity_frontier() {
    let (surface, atoms, bonds) = selectable_tetrahedral_child_fixture();
    let (_, rooted) = only_root_atom(&initial(&surface), atoms[0]);
    let plain_children = completed_atom_witnesses(&rooted, atoms[1]);
    assert_eq!(
        plain_children
            .iter()
            .map(|(text, _)| text.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["[C@H]", "[C@@H]"])
    );
    for token in ["/", "\\"] {
        let pending = ring_witness_choices(&rooted)
            .into_iter()
            .filter(|choice| choice.text() == token)
            .map(Choice::into_successor)
            .collect::<Vec<_>>();
        let choices = pending.iter()
            .flat_map(|state| completed_atom_witnesses(state, atoms[1]))
            .collect::<Vec<_>>();
        assert_eq!(
            choices
                .iter()
                .map(|(text, _)| text.as_str())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["[C@H]", "[C@@H]"])
        );

        let center = surface.tetrahedral_center(atoms[1]).unwrap();
        let reference = &center.reference_order;
        let PreparedDirectionalBondStatus::CompiledSite(site) =
            surface.directional_status(bonds[0])
        else {
            panic!("selected child carrier must compile");
        };
        let expected_mark = match token {
            "/" => CarrierMark::SlashAtFixedA,
            "\\" => CarrierMark::BackslashAtFixedA,
            _ => unreachable!(),
        };
        for (text, successor) in choices {
            let parity_index = usize::from(text == "[C@@H]");
            let expected_orders = (0..24_u8)
                .filter(|value| {
                    let order = independent_tetrahedral_order(reference, *value);
                    order[..2]
                        == [
                            TetrahedralLigand::Bond(bonds[0]),
                            TetrahedralLigand::VirtualHydrogen,
                        ]
                        && independent_tetrahedral_order_is_even(reference, *value)
                            == (parity_index == 0)
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(successor.pending, None);
            assert_eq!(successor.active_atom(), Some(atoms[1]));
            let observed_orders = successor.structural.semantic_domain(center.order_variable);
            assert_eq!(
                (0..24_u8)
                    .filter(|value| observed_orders.contains(*value))
                    .collect::<BTreeSet<_>>(),
                expected_orders,
                "{} after {token} must identify its exact ligand-order successor",
                text
            );
            assert_eq!(
                successor.structural.semantic_domain(site.mark_variable),
                expected_mark.singleton_domain(),
                "{token} must identify its exact carrier-mark successor"
            );
        }
    }
}

#[test]
fn selectable_tetrahedral_child_has_bounded_complete_mixed_walks() {
    let (surface, atoms, bonds) = selectable_tetrahedral_child_fixture();
    let terminals = reachable_terminal_states(initial(&surface));
    assert!(!terminals.is_empty());
    let center = surface.tetrahedral_center(atoms[1]).unwrap();
    let PreparedDirectionalBondStatus::CompiledSite(site) = surface.directional_status(bonds[0])
    else {
        panic!("selectable child fixture carrier must compile");
    };

    for terminal in terminals {
        assert!(terminal.is_accepted());
        assert!(terminal
            .structural
            .semantic_domain(center.order_variable)
            .is_singleton());
        assert!(terminal
            .structural
            .semantic_domain(center.role_pattern_variable)
            .is_singleton());
        assert!(terminal
            .structural
            .semantic_domain(site.mark_variable)
            .is_singleton());
        assert!(surface.directional_sides.iter().all(|side| terminal
            .structural
            .semantic_domain(side.pattern_variable)
            .is_singleton()));
    }
}

#[test]
fn selectable_ring_opening_preserves_unresolved_future_mark_spelling() {
    let (surface, atoms, bonds) = selectable_directional_ring_fixture();
    let (_, rooted) = only_root_atom(&initial(&surface), atoms[1]);
    let opening_incident = incident(&surface, atoms[1], bonds[0]);
    let candidate = rooted
        .structural
        .derive_candidates()
        .candidates()
        .iter()
        .copied()
        .find(|candidate| {
            matches!(candidate, StructuralCandidate::RingOpen { incident: found } if *found == opening_incident)
        })
        .unwrap();
    let PreparedDirectionalBondStatus::CompiledSite(site) = surface.directional_status(bonds[0])
    else {
        panic!("selectable ring carrier must compile");
    };

    let unrestricted = rooted.attempt_directional_ring_candidate(candidate, opening_incident);
    assert!(unrestricted
        .iter()
        .all(|attempt| !matches!(attempt, CandidateAttempt::Incomplete(_))));
    assert_eq!(
        unrestricted
            .iter()
            .filter_map(|attempt| match attempt {
                CandidateAttempt::Accepted { text, successor }
                    if text == "1" || (matches!(text.as_str(), "/" | "\\")
                        && successor.labels.slot_for_bond(bonds[0]) == RingLabelSlot(1)) => Some(text.as_str()),
                _ => None,
            })
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["1", "/", "\\"])
    );
    let label_only = unrestricted
        .iter()
        .find_map(|attempt| match attempt {
            CandidateAttempt::Accepted { text, successor } if text == "1" => Some(successor),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        label_only.structural.semantic_domain(site.mark_variable),
        CarrierMark::domain()
    );
    assert_eq!(
        label_only
            .structural
            .semantic_domain(site.ring_plan_variable),
        DirectionalRingPlan::PlainNone
            .singleton_domain()
            .union(DirectionalRingPlan::SlashAtA.singleton_domain())
            .union(DirectionalRingPlan::BackslashAtA.singleton_domain())
    );
    assert!(label_only
        .structural
        .factor_is_active(site.ring_plan_factor));

    let mut plain_source = rooted.clone();
    plain_source.structural = plain_source
        .structural
        .transitioned_semantics(
            &[(site.mark_variable, CarrierMark::Plain.singleton_domain())],
            &[],
        )
        .unwrap()
        .unwrap_consistent();
    let plain = plain_source.attempt_directional_ring_candidate(candidate, opening_incident);
    assert!(plain
        .iter()
        .all(|attempt| !matches!(attempt, CandidateAttempt::Incomplete(_))));
    assert!(plain.iter().any(|attempt| matches!(
        attempt,
        CandidateAttempt::Accepted { successor, .. }
            if successor.structural.semantic_domain(site.mark_variable)
                == CarrierMark::Plain.singleton_domain()
    )));
}

#[test]
fn label_only_directional_ring_opening_resolves_mark_at_the_closure_token() {
    let (surface, atoms, bonds) = selectable_directional_ring_fixture();
    let (_, rooted) = only_root_atom(&initial(&surface), atoms[1]);
    let mut pending = vec![(rooted, 0_usize)];
    let opened = loop {
        let (state, atom_offset) = pending.pop().expect("fixture reaches a ring opening");
        let mut found = None;
        for (successor, next_offset) in declared_successors(&state, atom_offset) {
            if matches!(
                successor.observe_raw().structural.traversal.bond_progress[bonds[0].index()],
                ObservedBondProgress::RingOpen { first_endpoint } if first_endpoint == atoms[1]
            ) {
                found = Some(successor);
                break;
            }
            pending.push((successor, next_offset));
        }
        if let Some(opened) = found {
            break opened;
        }
    };
    let PreparedDirectionalBondStatus::CompiledSite(site) = surface.directional_status(bonds[0])
    else {
        panic!("selectable ring carrier must compile");
    };

    let mut stack = vec![(opened, 0_usize)];
    let closure_source = loop {
        let (state, atom_offset) = stack
            .pop()
            .expect("opened ring must reach its other endpoint");
        let observed = state.observe_raw();
        if !state.is_spelling_atom()
            && observed
                .structural
                .traversal
                .active_frame
                .as_ref()
                .is_some_and(|frame| frame.atom == atoms[0])
            && matches!(
                observed.structural.traversal.bond_progress[bonds[0].index()],
                ObservedBondProgress::RingOpen { first_endpoint } if first_endpoint == atoms[1]
            )
        {
            break state;
        }
        stack.extend(declared_successors(&state, atom_offset));
    };

    let before = closure_source.observe_raw();
    let choices = ring_witness_choices(&closure_source);
    assert_eq!(closure_source.observe_raw(), before);
    let closure_choices = choices
        .iter()
        .filter(|choice| {
            matches!(
                choice
                    .successor()
                    .observe_raw()
                    .structural
                    .traversal
                    .bond_progress[bonds[0].index()],
                ObservedBondProgress::RingClosed { .. }
            )
        })
        .collect::<Vec<_>>();
    let source_marks = closure_source
        .structural
        .semantic_domain(site.mark_variable);
    let mut expected_texts = BTreeSet::from(["1"]);
    for mark in [CarrierMark::SlashAtFixedA, CarrierMark::BackslashAtFixedA] {
        if source_marks.contains(mark.value_index()) {
            expected_texts.insert(match mark {
                CarrierMark::SlashAtFixedA => "/",
                CarrierMark::BackslashAtFixedA => "\\",
                CarrierMark::Plain => unreachable!(),
            });
        }
    }
    assert_eq!(
        closure_choices
            .iter()
            .map(|choice| choice.text())
            .collect::<BTreeSet<_>>(),
        expected_texts
    );
    for choice in closure_choices {
        let successor = choice.successor();
        let mark = successor.structural.semantic_domain(site.mark_variable);
        let plan = successor
            .structural
            .semantic_domain(site.ring_plan_variable);
        assert!(mark.is_singleton());
        assert!(plan.is_singleton());
        let plan = DirectionalRingPlan::ALL
            .into_iter()
            .find(|candidate| plan.contains(candidate.value_index()))
            .unwrap();
        assert_eq!(mark, plan.mark().singleton_domain());
        assert!(successor.structural.factor_is_active(site.ring_plan_factor));
    }
}

#[test]
fn exhausted_marked_ring_labels_are_spelling_rejections_before_incompleteness() {
    let (surface, atoms, bonds) = selectable_directional_ring_fixture();
    let (_, mut rooted) = only_root_atom(&initial(&surface), atoms[1]);
    rooted.labels.maximum_spelling_label = Some(0);
    // ID zero is legal; reserve it to exhaust this test-only label domain.
    rooted.labels.allocate_at(RingLabelSlot(0), BondId::new(999));
    let opening_incident = incident(&surface, atoms[1], bonds[0]);
    let candidate = rooted
        .structural
        .derive_candidates()
        .candidates()
        .iter()
        .copied()
        .find(|candidate| {
            matches!(candidate, StructuralCandidate::RingOpen { incident: found } if *found == opening_incident)
        })
        .unwrap();
    let before = rooted.observe_raw();
    let attempts = rooted.attempt_directional_ring_candidate(candidate, opening_incident);
    assert_eq!(rooted.observe_raw(), before);
    assert!(attempts.iter().all(|attempt| matches!(
        attempt,
        CandidateAttempt::Rejected {
            reason: CandidateRejection::RingLabelUnavailable { .. }
        }
    )));
}

#[test]
fn every_complete_selectable_walk_resolves_physical_marks_and_side_patterns() {
    let (surface, _, bonds) =
        selectable_directional_fixture(NonStereoBondToken::Elided, NonStereoBondToken::Single);
    let terminals = reachable_terminal_states(initial(&surface));
    assert!(!terminals.is_empty());
    for terminal in terminals {
        assert!(terminal.is_accepted());
        for bond in [bonds[0], bonds[2], bonds[3]] {
            let PreparedDirectionalBondStatus::CompiledSite(site) =
                surface.directional_status(bond)
            else {
                panic!("selectable carrier must remain compiled");
            };
            assert!(terminal
                .structural
                .semantic_domain(site.mark_variable)
                .is_singleton());
            let role = terminal.structural.bond_decision_domain(bond);
            if role == BondRole::Traversal.singleton_domain() {
                assert!(!terminal.structural.factor_is_active(site.ring_plan_factor));
            } else {
                assert_eq!(role, BondRole::Ring.singleton_domain());
                assert!(terminal.structural.factor_is_active(site.ring_plan_factor));
                let plan = terminal.structural.semantic_domain(site.ring_plan_variable);
                assert!(plan.is_singleton());
                let plan = DirectionalRingPlan::ALL
                    .into_iter()
                    .find(|candidate| plan.contains(candidate.value_index()))
                    .unwrap();
                assert_eq!(
                    terminal.structural.semantic_domain(site.mark_variable),
                    plan.mark().singleton_domain()
                );
            }
        }
        assert!(surface.directional_sides.iter().all(|side| terminal
            .structural
            .semantic_domain(side.pattern_variable)
            .is_singleton()));
    }
}

#[test]
fn repeated_directional_relation_for_one_double_bond_is_rejected() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..6)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = [
        graph.add_bond(atoms[0], atoms[2]).unwrap(),
        graph.add_bond(atoms[1], atoms[2]).unwrap(),
        graph.add_bond(atoms[2], atoms[3]).unwrap(),
        graph.add_bond(atoms[3], atoms[4]).unwrap(),
        graph.add_bond(atoms[3], atoms[5]).unwrap(),
    ];
    let relation = |left, right| PreparedDirectionalRelation {
        double_bond: bonds[2],
        left_endpoint: atoms[2],
        left_carriers: vec![left].into_boxed_slice(),
        right_endpoint: atoms[3],
        right_carriers: vec![right].into_boxed_slice(),
        side_phase_xor: false,
    };

    assert_eq!(
        PreparedNonStereo::with_atom_tokens_and_directional(
            PreparedMolecule::new(graph.build()),
            ["A", "B", "L", "R", "C", "D"]
                .map(|text| PreparedAtomToken::Fixed(text.to_owned()))
                .into(),
            vec![
                NonStereoBondToken::Elided,
                NonStereoBondToken::Elided,
                NonStereoBondToken::Double,
                NonStereoBondToken::Elided,
                NonStereoBondToken::Elided,
            ],
            vec![
                relation(
                    PreparedDirectionalCarrier::unflipped(bonds[0]),
                    PreparedDirectionalCarrier::unflipped(bonds[3]),
                ),
                relation(
                    PreparedDirectionalCarrier::unflipped(bonds[1]),
                    PreparedDirectionalCarrier::unflipped(bonds[4]),
                ),
            ],
        )
        .unwrap_err(),
        PreparedNonStereoError::RepeatedDirectionalRelation(bonds[2])
    );
}

#[test]
fn inline_directional_carrier_emits_sign_before_the_child_atom() {
    let (surface, atoms, bonds) = directional_chain_fixture();
    let (_, root) = only_root_atom(&initial(&surface), atoms[0]);
    assert_eq!(root.active_atom(), Some(atoms[0]));

    let choices = ring_witness_choices(&root);
    assert_eq!(
        choices.iter().map(Choice::text).collect::<BTreeSet<_>>(),
        BTreeSet::from(["/", "\\"])
    );
    assert!(choices.iter().all(|choice| {
        matches!(&choice.successor().pending, Some(PendingEmission::InlineAtom(entry)) if entry.incident == incident(
                &surface, atoms[0], bonds[0],
            ))
    }));
    let slash = choices
        .into_iter()
        .find(|choice| choice.text() == "/")
        .unwrap()
        .into_successor();
    let PreparedDirectionalBondStatus::CompiledSite(left) = surface.directional_status(bonds[0])
    else {
        panic!("left carrier must be compiled");
    };
    let PreparedDirectionalBondStatus::CompiledSite(right) = surface.directional_status(bonds[2])
    else {
        panic!("right carrier must be compiled");
    };
    assert_eq!(
        slash.structural.semantic_domain(left.mark_variable),
        CarrierMark::SlashAtFixedA.singleton_domain()
    );
    assert_eq!(
        slash.structural.semantic_domain(right.mark_variable),
        CarrierMark::BackslashAtFixedA.singleton_domain()
    );
    only_choice(&slash, "L");
}

#[test]
fn rdkit_anchor_restricts_an_incoming_mark_before_publishing_its_token() {
    let (surface, atoms, bonds) = directional_chain_fixture_with_atoms(
        ["C", "C", "C", "C"],
        NonStereoBondToken::Elided,
        NonStereoBondToken::Elided,
    );
    let root = State::initial_with_regime(&surface, crate::WriterRegime::RdkitRandomNoncanonical)
        .unwrap()
        .unwrap_consistent();
    let (_, root) = only_root_atom(&root, atoms[0]);
    let choices = ring_witness_choices(&root);

    assert_eq!(
        choices.iter().map(Choice::text).collect::<BTreeSet<_>>(),
        BTreeSet::from(["/"])
    );
    let slash = choices.into_iter().next().unwrap().into_successor();
    let PreparedDirectionalBondStatus::CompiledSite(left) = surface.directional_status(bonds[0])
    else {
        panic!("left carrier must be compiled");
    };
    let PreparedDirectionalBondStatus::CompiledSite(right) = surface.directional_status(bonds[2])
    else {
        panic!("right carrier must be compiled");
    };
    assert_eq!(
        slash.structural.semantic_domain(left.mark_variable),
        CarrierMark::SlashAtFixedA.singleton_domain()
    );
    assert_eq!(
        slash.structural.semantic_domain(right.mark_variable),
        CarrierMark::BackslashAtFixedA.singleton_domain()
    );
    assert!(matches!(&slash.pending, Some(PendingEmission::InlineAtom(entry))
        if entry.incident == incident(&surface, atoms[0], bonds[0])));
}

#[test]
fn final_inline_directional_child_waits_for_atom_spelling_after_graph_completion() {
    let (surface, atoms, bonds) = directional_chain_fixture();
    let (_, root) = only_root_atom(&initial(&surface), atoms[0]);
    let first_sign = ring_witness_choices(&root)
        .into_iter()
        .find(|choice| matches!(choice.text(), "/" | "\\"))
        .unwrap()
        .into_successor();
    let (_, left) = only_choice(&first_sign, surface.atom_text(atoms[1]));
    let (_, double) = only_choice(&left, "=");
    let (_, right) = only_choice(&double, surface.atom_text(atoms[2]));
    let final_sign = ring_witness_choices(&right)
        .into_iter()
        .find(|choice| matches!(choice.text(), "/" | "\\"))
        .unwrap()
        .into_successor();
    assert!(matches!(&final_sign.pending, Some(PendingEmission::InlineAtom(entry))
        if entry.incident == incident(&surface, atoms[2], bonds[2])));
    assert_eq!(final_sign.active_atom(), Some(atoms[3]));
    assert!(final_sign.graph_is_complete());
    assert!(!final_sign.is_accepted());
    let (_, complete) = only_choice(&final_sign, surface.atom_text(atoms[3]));
    assert!(complete.is_accepted());
}

#[test]
fn directional_glyph_replaces_explicit_single_and_aromatic_base_tokens() {
    for token in [NonStereoBondToken::Single, NonStereoBondToken::Aromatic] {
        let (surface, atoms, _) =
            directional_chain_fixture_with_carrier_tokens(token, NonStereoBondToken::Elided);
        let (_, rooted) = only_root_atom(&initial(&surface), atoms[0]);
        assert_eq!(
            ring_witness_choices(&rooted)
                .iter()
                .map(Choice::text)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["/", "\\"])
        );
    }
}

#[test]
fn branch_directional_mark_is_committed_at_the_parenthesis() {
    let (surface, atoms, bonds) = directional_chain_fixture();
    let (_, rooted) = only_root_atom(&initial(&surface), atoms[1]);
    let branches = ring_witness_choices(&rooted)
        .into_iter()
        .filter(|choice| {
            choice.text() == "("
                && matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == incident(
                        &surface, atoms[1], bonds[0],
                    ))
        })
        .map(Choice::into_successor)
        .collect::<Vec<_>>();
    let PreparedDirectionalBondStatus::CompiledSite(left) = surface.directional_status(bonds[0])
    else {
        panic!("left carrier must be compiled");
    };
    assert_eq!(branches.len(), 2);
    let mut signs = BTreeSet::new();
    let mut mark_union = Domain::empty();
    for branch in branches {
        let mark = branch.structural.semantic_domain(left.mark_variable);
        assert!(mark.is_singleton());
        mark_union = mark_union.union(mark);
        let choices = ring_witness_choices(&branch);
        assert_eq!(choices.len(), 1, "the branch carries its selected bond text");
        let sign = choices[0].text();
        signs.insert(sign.to_owned());
        assert!(matches!(&choices[0].successor().pending,
            Some(PendingEmission::BranchAtom(entry))
                if entry.incident == incident(&surface, atoms[1], bonds[0])));
        if sign == "/" {
            assert_eq!(mark, CarrierMark::BackslashAtFixedA.singleton_domain());
        }
    }
    assert_eq!(mark_union, CarrierMark::marked_domain());
    assert_eq!(signs, BTreeSet::from(["/".to_owned(), "\\".to_owned()]));
}

#[test]
fn backend_failure_on_directional_mark_aborts_the_complete_frontier() {
    let (surface, atoms, _) = directional_chain_fixture();
    let initial = NonStereoWriterState::<FailDirectionalMarkSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let (_, rooted) = only_root_atom(&initial, atoms[0]);
    let before = rooted.observe_raw();
    assert!(matches!(
        rooted.choices(),
        Err(ChoiceFailure::Backend(InjectedSolverFailure::Restriction))
    ));
    assert_eq!(rooted.observe_raw(), before);
}

#[test]
fn each_directional_mark_choice_uses_one_solver_restriction_batch() {
    let _counter_guard = DIRECTIONAL_RESTRICTION_TEST_LOCK.lock().unwrap();
    let (surface, atoms, _) = directional_chain_fixture();
    let initial = NonStereoWriterState::<CountDirectionalRestrictionsSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let (_, rooted) = only_root_atom(&initial, atoms[0]);
    DIRECTIONAL_RESTRICTION_CALLS.store(0, Ordering::Relaxed);
    let choices = ring_witness_choices(&rooted);
    assert_eq!(choices.len(), 2);
    assert_eq!(
        DIRECTIONAL_RESTRICTION_CALLS.load(Ordering::Relaxed),
        2,
        "each slash/backslash successor uses one atomic solver transition"
    );
}

#[test]
fn directional_ring_endpoints_use_one_solver_transition_per_semantic_plan() {
    let _counter_guard = DIRECTIONAL_RESTRICTION_TEST_LOCK.lock().unwrap();
    type CountingState = NonStereoWriterState<CountDirectionalRestrictionsSolver>;

    let (surface, atoms, bonds) = selectable_directional_ring_fixture();
    let initial = CountingState::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let (_, rooted) = only_root_atom(&initial, atoms[1]);
    let opening_incident = incident(&surface, atoms[1], bonds[0]);
    let opening = rooted
        .structural
        .derive_candidates()
        .candidates()
        .iter()
        .copied()
        .find(|candidate| {
            matches!(candidate, StructuralCandidate::RingOpen { incident: found } if *found == opening_incident)
        })
        .unwrap();
    DIRECTIONAL_RESTRICTION_CALLS.store(0, Ordering::Relaxed);
    let opening_attempts = rooted.attempt_directional_ring_candidate(opening, opening_incident);
    assert_eq!(
        DIRECTIONAL_RESTRICTION_CALLS.load(Ordering::Relaxed),
        3
    );
    let opened = opening_attempts
        .into_iter()
        .find_map(|attempt| match attempt {
            CandidateAttempt::Accepted { text, successor } if text == "1" => Some(successor),
            _ => None,
        })
        .unwrap();

    let mut stack = vec![(opened, 0_usize)];
    let closure_source = loop {
        let (state, atom_offset) = stack
            .pop()
            .expect("opened ring must reach its closure endpoint");
        let observed = state.observe_raw();
        if observed
            .structural
            .traversal
            .active_frame
            .as_ref()
            .is_some_and(|frame| frame.atom == atoms[0])
            && matches!(
                observed.structural.traversal.bond_progress[bonds[0].index()],
                ObservedBondProgress::RingOpen { .. }
            )
        {
            break state;
        }
        stack.extend(declared_successors(&state, atom_offset));
    };
    let incident = incident(&surface, atoms[0], bonds[0]);
    let closure = closure_source
        .structural
        .derive_candidates()
        .candidates()
        .iter()
        .copied()
        .find(|candidate| {
            matches!(candidate, StructuralCandidate::RingClose { incident: found, .. } if *found == incident)
        })
        .unwrap();
    DIRECTIONAL_RESTRICTION_CALLS.store(0, Ordering::Relaxed);
    let closure_attempts = closure_source.attempt_directional_ring_candidate(closure, incident);
    // The walked prefix resolves one sign, leaving omission and one marked endpoint plan.
    assert!(!closure_attempts.is_empty());
    assert_eq!(
        DIRECTIONAL_RESTRICTION_CALLS.load(Ordering::Relaxed),
        2
    );
}

#[test]
fn directional_branch_commit_narrows_a_tetrahedral_parent_before_sign_selection() {
    let (surface, atoms, bonds) = directional_tetrahedral_parent_fixture();
    let rooted = completed_atom_witnesses(&initial(&surface), atoms[0])
        .into_iter()
        .find(|(text, _)| text.contains('@') && !text.contains("@@"))
        .map(|(_, state)| state)
        .unwrap();
    let center = surface.tetrahedral_center(atoms[0]).unwrap();
    let before = rooted.structural.semantic_domain(center.order_variable);
    let choices = ring_witness_choices(&rooted);
    let branch_choices = choices
        .into_iter()
        .filter(|choice| {
            choice.text() == "("
                && matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == incident(
                        &surface, atoms[0], bonds[0],
                    ))
        })
        .collect::<Vec<_>>();
    assert_eq!(branch_choices.len(), 1);
    let branch = branch_choices.into_iter().next().unwrap().into_successor();
    let after = branch.structural.semantic_domain(center.order_variable);
    let expected_orders = (0..24_u8)
        .filter(|value| {
            let order = independent_tetrahedral_order(&center.reference_order, *value);
            order[0] == TetrahedralLigand::Bond(bonds[0])
                && independent_tetrahedral_order_is_even(&center.reference_order, *value)
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        (0..24_u8)
            .filter(|value| after.contains(*value))
            .collect::<BTreeSet<_>>(),
        expected_orders
    );
    assert!(after.is_subset_of(before));
    let PreparedDirectionalBondStatus::CompiledSite(site) = surface.directional_status(bonds[0])
    else {
        panic!("tetrahedral parent carrier must be compiled");
    };
    assert_eq!(
        branch.structural.semantic_domain(site.mark_variable),
        CarrierMark::marked_domain()
    );
}

#[test]
fn directional_token_leading_to_tetrahedral_child_preserves_the_atom_parity_frontier() {
    let (surface, atoms, bonds) = directional_tetrahedral_child_fixture();
    let mut stack = vec![(String::new(), initial(&surface), 0_usize)];
    let mut directional_children = BTreeMap::<String, Vec<State>>::new();
    let mut explored = 0_usize;
    while let Some((prefix, state, atom_offset)) = stack.pop() {
        explored += 1;
        assert!(explored <= 100_000, "incoming-carrier witness exceeded its path bound");
        for choice in ring_witness_choices(&state) {
            if !selected_atom_spelling_transition(&state, &choice, atom_offset) {
                continue;
            }
            let token = choice.text().to_owned();
            let next_prefix = format!("{prefix}{token}");
            let successor = choice.successor();
            let next_offset = next_atom_spelling_offset(&state, successor, atom_offset, &token);
            if matches!(choice.text(), "/" | "\\")
                && state.active_atom() == Some(atoms[0])
                && matches!(&successor.pending, Some(PendingEmission::InlineAtom(entry)) if entry.incident == incident(
                        &surface, atoms[0], bonds[0],
                    ))
            {
                assert_eq!(successor.active_atom(), Some(atoms[2]));
                assert!(successor.atom_spelling.is_none());
                directional_children
                    .entry(next_prefix)
                    .or_default()
                    .push(choice.into_successor());
            } else {
                stack.push((next_prefix, choice.into_successor(), next_offset));
            }
        }
    }
    assert!(
        !directional_children.is_empty(),
        "fixture must reach the carrier inline"
    );
    let expected = declared_atom_witnesses(&surface, atoms[2])
        .into_iter()
        .collect::<BTreeSet<_>>();
    for (prefix, states) in &directional_children {
        // A decoder prefix may retain several hidden states that partition the spellings.
        let witnesses = states
            .iter()
            .flat_map(|state| completed_atom_witnesses(state, atoms[2]))
            .map(|(text, _)| text)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            witnesses, expected,
            "incoming carrier prefix {prefix:?} must preserve both tetrahedral spellings"
        );
    }
}

#[test]
fn every_complete_directional_walk_resolves_each_carrier_sign() {
    let (surface, _, bonds) = directional_chain_fixture();
    let terminals = reachable_terminal_states(initial(&surface));
    assert!(!terminals.is_empty());
    for terminal in terminals {
        assert!(terminal.is_accepted());
        for bond in [bonds[0], bonds[2]] {
            let PreparedDirectionalBondStatus::CompiledSite(site) =
                surface.directional_status(bond)
            else {
                panic!("directional carrier must remain compiled");
            };
            assert!(terminal
                .structural
                .semantic_domain(site.mark_variable)
                .is_singleton());
        }
    }
}

#[test]
fn compiled_directional_ring_candidate_publishes_complete_endpoint_choices() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..3)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = [
        graph.add_bond(atoms[0], atoms[1]).unwrap(),
        graph.add_bond(atoms[1], atoms[2]).unwrap(),
        graph.add_bond(atoms[0], atoms[2]).unwrap(),
    ];
    let surface = PreparedNonStereo::with_atom_tokens_and_directional(
        PreparedMolecule::new(graph.build()),
        ["A", "B", "C"]
            .map(|text| PreparedAtomToken::Fixed(text.to_owned()))
            .into(),
        vec![
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
        ],
        vec![PreparedDirectionalRelation {
            double_bond: bonds[0],
            left_endpoint: atoms[0],
            left_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[2])].into_boxed_slice(),
            right_endpoint: atoms[1],
            right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[1])]
                .into_boxed_slice(),
            side_phase_xor: false,
        }],
    )
    .unwrap();
    let (_, rooted) = only_root_atom(&initial(&surface), atoms[0]);
    let before = rooted.observe_raw();
    let choices = ring_witness_choices(&rooted);
    assert!(choices
        .iter()
        .any(|choice| matches!(choice.text(), "1" | "/" | "\\")));
    assert_eq!(rooted.observe_raw(), before);
}

fn directional_ring_with_ordinary_siblings_fixture() -> (PreparedNonStereo, Vec<AtomId>, Vec<BondId>)
{
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..6)
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let bonds = [
        graph.add_bond(atoms[0], atoms[1]).unwrap(),
        graph.add_bond(atoms[0], atoms[2]).unwrap(),
        graph.add_bond(atoms[0], atoms[3]).unwrap(),
        graph.add_bond(atoms[1], atoms[2]).unwrap(),
        graph.add_bond(atoms[2], atoms[3]).unwrap(),
        graph.add_bond(atoms[0], atoms[4]).unwrap(),
        graph.add_bond(atoms[4], atoms[5]).unwrap(),
    ];
    let surface = PreparedNonStereo::with_atom_tokens_and_directional(
        PreparedMolecule::new(graph.build()),
        ["A", "B", "C", "D", "E", "F"]
            .map(|text| PreparedAtomToken::Fixed(text.to_owned()))
            .into(),
        vec![
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Elided,
            NonStereoBondToken::Double,
            NonStereoBondToken::Elided,
        ],
        vec![PreparedDirectionalRelation {
            double_bond: bonds[5],
            left_endpoint: atoms[0],
            left_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[0])].into_boxed_slice(),
            right_endpoint: atoms[4],
            right_carriers: vec![PreparedDirectionalCarrier::unflipped(bonds[6])]
                .into_boxed_slice(),
            side_phase_xor: false,
        }],
    )
    .unwrap();
    (surface, atoms, bonds.into())
}

#[test]
fn contradictory_incomplete_ring_candidate_does_not_suppress_valid_siblings() {
    let (surface, atoms, _) = directional_ring_with_ordinary_siblings_fixture();
    let initial = NonStereoWriterState::<RejectFirstVariableSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let (_, rooted) = only_root_atom(&initial, atoms[0]);

    let choices = ring_witness_choices(&rooted);

    assert!(!choices.is_empty());
    assert!(choices.iter().all(|choice| matches!(choice.text(), "1" | "-")));
}

#[test]
fn backend_failure_while_filtering_incomplete_ring_candidate_aborts_the_batch() {
    let (surface, atoms, _) = directional_ring_with_ordinary_siblings_fixture();
    let initial = NonStereoWriterState::<FailingRestrictionSolver>::initial(&surface)
        .unwrap()
        .unwrap_consistent();
    let (_, rooted) = only_root_atom(&initial, atoms[0]);

    assert!(matches!(
        rooted.choices(),
        Err(ChoiceFailure::Backend(InjectedSolverFailure::Restriction))
    ));
}

#[test]
#[ignore = "opt-in native work-count probe; run with --ignored --nocapture"]
fn native_work_count_probe_fixed_interior_prefix() {
    fn json_quote(text: &str) -> String {
        format!("\"{}\"", text.replace('\\', "\\\\").replace('\"', "\\\""))
    }

    fn prepared_chain(
        component_atoms: usize,
        spectator_count: usize,
        tetra_mode: Option<bool>,
    ) -> PreparedNonStereo {
        let tetra_atoms = if tetra_mode.is_some() { 4 } else { 0 };
        let total = component_atoms + spectator_count * 2 + tetra_atoms;
        let mut graph = PreparedGraphBuilder::new();
        let atoms = (0..total)
            .map(|_| graph.add_atom().unwrap())
            .collect::<Vec<_>>();
        let mut bonds = Vec::new();
        for index in 0..component_atoms - 1 {
            bonds.push(graph.add_bond(atoms[index], atoms[index + 1]).unwrap());
        }
        for component in 0..spectator_count {
            let first = component_atoms + component * 2;
            bonds.push(graph.add_bond(atoms[first], atoms[first + 1]).unwrap());
        }
        let mut atom_tokens = vec![PreparedAtomToken::Fixed("C".to_owned()); total];
        if let Some(connected) = tetra_mode {
            let center = component_atoms + spectator_count * 2;
            let leaves = [center + 1, center + 2, center + 3];
            let center_bonds =
                leaves.map(|leaf| graph.add_bond(atoms[center], atoms[leaf]).unwrap());
            bonds.extend(center_bonds);
            if connected {
                bonds.push(
                    graph
                        .add_bond(atoms[component_atoms - 1], atoms[leaves[0]])
                        .unwrap(),
                );
            }
            atom_tokens[center] = PreparedAtomToken::Tetrahedral {
                reference_order: [
                    TetrahedralLigand::Bond(center_bonds[0]),
                    TetrahedralLigand::Bond(center_bonds[1]),
                    TetrahedralLigand::Bond(center_bonds[2]),
                    TetrahedralLigand::VirtualHydrogen,
                ],
                text_by_parity: ["[C@H]".to_owned(), "[C@@H]".to_owned()],
            };
        }
        PreparedNonStereo::with_atom_tokens(
            PreparedMolecule::new(graph.build()),
            atom_tokens,
            vec![NonStereoBondToken::Elided; bonds.len()],
        )
        .unwrap()
    }

    fn probe(name: &str, surface: PreparedNonStereo, inactive_center: Option<usize>) {
        let initial = initial(&surface);
        let (_, root) = only_root_atom(&initial, AtomId::new(0));
        assert_eq!(root.active_atom(), Some(AtomId::new(0)));
        if let Some(index) = inactive_center {
            let center = surface
                .tetrahedral_center(AtomId::new(u32::try_from(index).unwrap()))
                .expect("inactive-stereo case must prepare its remote tetrahedral center");
            for factor in std::iter::once(center.root_layout_factor).chain(
                center
                    .entry_layout_factors
                    .iter()
                    .map(|(_, factor)| *factor),
            ) {
                assert_eq!(
                    root.structural
                        .constraints_for_test()
                        .factor_is_active(factor),
                    Some(false)
                );
            }
        }
        let native = root.structural.constraints_for_test();
        native.reset_projector_work_counts();
        native.reset_exact_run_counts();
        native.reset_mixed_search_branch_count();
        native.reset_tetrahedral_factor_revision_count();
        native.reset_directional_ring_factor_revision_count();
        native.reset_solver_copy_counts();
        root.structural.reset_traversal_copy_counts();
        root.reset_writer_work_counts();

        let choices = ring_witness_choices(&root);
        let interior = choices
            .iter()
            .filter(|choice| {
                pending_atom(choice.successor()) == Some(AtomId::new(1))
                    || choice.successor().active_atom() == Some(AtomId::new(1))
            })
            .count();
        assert!(interior > 0, "{name}: fixed 0-to-1 transition is required");
        assert!(choices.iter().all(|choice| {
            choice.successor().active_atom().is_some()
                || choice.successor().is_spelling_atom()
                || choice.successor().is_accepted()
                || choice.successor().pending.is_some()
        }));
        let projector = native.projector_work_counts();
        let exact = native.exact_run_counts();
        let branches = native.mixed_search_branch_count();
        let tetra = native.tetrahedral_factor_revision_count();
        let directional = native.directional_ring_factor_revision_count();
        let solver_copies = native.solver_copy_counts();
        let traversal_copies = root.structural.traversal_copy_counts();
        let writer = root.writer_work_counts();
        let successor_observation = choices[0]
            .successor()
            .structural
            .constraints_for_test()
            .projector_work_counts();
        assert_eq!(
            successor_observation, projector,
            "source and successor read one shared lineage total"
        );
        assert_eq!(
            choices[0].successor().writer_work_counts(),
            writer,
            "discarded prevalidation remains visible on the source lineage"
        );
        println!(
            "{{\"schema\":\"south_star2_native_work_count_v1\",\"case\":\"{name}\",\"graph_atoms\":{},\"prefix\":\"root_atom_0_active_atom_0\",\"event\":\"choices_to_active_atom_1\",\"choices\":{},\"fixed_transition_successors\":{},\"projector\":{{\"first_edge_visits\":{},\"atom_visits\":{},\"second_edge_visits\":{},\"potential_layout_checks\":{}}},\"exact_runs\":{{\"binary\":{},\"mixed\":{}}},\"mixed_branch_points\":{},\"tetra_factor_revisions\":{},\"directional_ring_revisions\":{},\"copy_pages\":{{\"solver_domains\":{{\"index_nodes\":{},\"value_pages\":{}}},\"solver_active_factors\":{{\"index_nodes\":{},\"value_pages\":{}}},\"solver_activation_adjacency\":{{\"index_nodes\":{},\"value_pages\":{}}},\"traversal_aggregate\":{{\"index_nodes\":{},\"value_pages\":{}}}}},\"writer_frontier\":{{\"pending_atom_evaluations\":{},\"discarded_prevalidated_successors\":{}}},\"successor_snapshot_equals_source_shared_total\":true}}",
            surface.molecule().graph().atom_count(), choices.len(), interior,
            projector.0, projector.1, projector.2, projector.3,
            exact.0, exact.1, branches, tetra, directional,
            solver_copies.0.0, solver_copies.0.1,
            solver_copies.1.0, solver_copies.1.1,
            solver_copies.2.0, solver_copies.2.1,
            traversal_copies.0, traversal_copies.1, writer.0, writer.1,
        );
    }

    for (name, atoms, spectators, tetra) in [
        ("fixed_ccc", 3, 0, None),
        ("fixed_ccc_plus_2_disconnected_ethanes", 3, 2, None),
        (
            "fixed_ccc_plus_disconnected_inactive_tetra",
            3,
            0,
            Some(false),
        ),
        ("connected_extension_ccccccc", 7, 0, None),
        (
            "connected_extension_plus_distant_inactive_tetra",
            7,
            0,
            Some(true),
        ),
    ] {
        let inactive_center = tetra.map(|_| atoms + spectators * 2);
        probe(
            name,
            prepared_chain(atoms, spectators, tetra),
            inactive_center,
        );
    }

    fn triangle_case(
        tail_atoms: usize,
        plain_spectators: usize,
        disconnected_tetra: bool,
        distant_connected_tetra: bool,
    ) -> (PreparedNonStereo, [BondId; 3], usize) {
        let tetra_atoms = usize::from(disconnected_tetra || distant_connected_tetra) * 4;
        let tail_extra = tail_atoms;
        let total = 3 + tail_extra + plain_spectators * 2 + tetra_atoms;
        let mut graph = PreparedGraphBuilder::new();
        let atoms = (0..total)
            .map(|_| graph.add_atom().unwrap())
            .collect::<Vec<_>>();
        let triangle = [
            graph.add_bond(atoms[0], atoms[1]).unwrap(),
            graph.add_bond(atoms[1], atoms[2]).unwrap(),
            graph.add_bond(atoms[2], atoms[0]).unwrap(),
        ];
        let mut bonds = triangle.to_vec();
        let mut next = 3;
        if tail_atoms > 0 {
            bonds.push(graph.add_bond(atoms[2], atoms[next]).unwrap());
            for index in 0..tail_atoms - 1 {
                bonds.push(
                    graph
                        .add_bond(atoms[next + index], atoms[next + index + 1])
                        .unwrap(),
                );
            }
            next += tail_atoms;
        }
        for _ in 0..plain_spectators {
            bonds.push(graph.add_bond(atoms[next], atoms[next + 1]).unwrap());
            next += 2;
        }
        let mut atom_tokens = vec![PreparedAtomToken::Fixed("C".to_owned()); total];
        let mut inactive_center = usize::MAX;
        if disconnected_tetra || distant_connected_tetra {
            let center = next;
            let leaves = [center + 1, center + 2, center + 3];
            let center_bonds =
                leaves.map(|leaf| graph.add_bond(atoms[center], atoms[leaf]).unwrap());
            bonds.extend(center_bonds);
            if distant_connected_tetra {
                let tail_end = if tail_atoms == 0 { 2 } else { 2 + tail_atoms };
                bonds.push(graph.add_bond(atoms[tail_end], atoms[leaves[0]]).unwrap());
            }
            atom_tokens[center] = PreparedAtomToken::Tetrahedral {
                reference_order: [
                    TetrahedralLigand::Bond(center_bonds[0]),
                    TetrahedralLigand::Bond(center_bonds[1]),
                    TetrahedralLigand::Bond(center_bonds[2]),
                    TetrahedralLigand::VirtualHydrogen,
                ],
                text_by_parity: ["[C@H]".to_owned(), "[C@@H]".to_owned()],
            };
            inactive_center = center;
        }
        let surface = PreparedNonStereo::with_atom_tokens(
            PreparedMolecule::new(graph.build()),
            atom_tokens,
            vec![NonStereoBondToken::Elided; bonds.len()],
        )
        .unwrap();
        (surface, triangle, inactive_center)
    }

    fn probe_triangle(
        name: &str,
        tail_atoms: usize,
        plain_spectators: usize,
        disconnected_tetra: bool,
        distant_connected_tetra: bool,
    ) -> (Vec<String>, (usize, usize, usize, usize), (usize, usize)) {
        let (surface, triangle, inactive_center) = triangle_case(
            tail_atoms,
            plain_spectators,
            disconnected_tetra,
            distant_connected_tetra,
        );
        let (_, root) = only_root_atom(&initial(&surface), AtomId::new(0));
        assert_eq!(root.active_atom(), Some(AtomId::new(0)));
        let unresolved_seed_domains =
            triangle.map(|bond| root.structural.bond_role_domain_for_test(bond).len());
        assert!(
            unresolved_seed_domains.iter().any(|size| *size > 1),
            "{name}: triangle must retain an unresolved role decision"
        );
        if inactive_center != usize::MAX {
            let center = surface
                .tetrahedral_center(AtomId::new(inactive_center as u32))
                .unwrap();
            for factor in std::iter::once(center.root_layout_factor).chain(
                center
                    .entry_layout_factors
                    .iter()
                    .map(|(_, factor)| *factor),
            ) {
                assert_eq!(
                    root.structural
                        .constraints_for_test()
                        .factor_is_active(factor),
                    Some(false)
                );
            }
        }
        let native = root.structural.constraints_for_test();
        native.reset_projector_work_counts();
        native.reset_exact_run_counts();
        native.reset_mixed_search_branch_count();
        native.reset_tetrahedral_factor_revision_count();
        native.reset_directional_ring_factor_revision_count();
        native.reset_solver_copy_counts();
        root.structural.reset_traversal_copy_counts();
        root.reset_writer_work_counts();
        let choices = ring_witness_choices(&root);
        assert!(!choices.is_empty());
        let strings = choices
            .iter()
            .map(|choice| choice.text().to_owned())
            .collect::<Vec<_>>();
        let projector = native.projector_work_counts();
        assert!(
            projector.0 + projector.1 + projector.2 > 0,
            "{name}: unresolved triangle choices must execute spanning projector loops"
        );
        let copies = native.solver_copy_counts();
        let traversal_copies = root.structural.traversal_copy_counts();
        let writer = root.writer_work_counts();
        let exact = native.exact_run_counts();
        assert_eq!(
            choices[0]
                .successor()
                .structural
                .constraints_for_test()
                .projector_work_counts(),
            projector
        );
        assert_eq!(choices[0].successor().writer_work_counts(), writer);
        println!(
            "{{\"schema\":\"south_star2_native_work_count_v1\",\"case\":\"{name}\",\"graph_atoms\":{},\"triangle_seed_role_domain_sizes\":[{},{},{}],\"prefix\":\"root_atom_0_active_atom_0\",\"event\":\"complete_choices_batch_for_unresolved_triangle_roles\",\"remote_tetra_factor_family_inactive_at_seed\":{},\"choices\":{},\"projector\":{{\"first_edge_visits\":{},\"atom_visits\":{},\"second_edge_visits\":{},\"potential_layout_checks\":{}}},\"exact_runs\":{{\"binary\":{},\"mixed\":{}}},\"mixed_branch_points\":{},\"tetra_factor_revisions\":{},\"directional_ring_revisions\":{},\"copy_pages\":{{\"solver_domains\":{{\"index_nodes\":{},\"value_pages\":{}}},\"solver_active_factors\":{{\"index_nodes\":{},\"value_pages\":{}}},\"solver_activation_adjacency\":{{\"index_nodes\":{},\"value_pages\":{}}},\"traversal_aggregate\":{{\"index_nodes\":{},\"value_pages\":{}}}}},\"writer_frontier\":{{\"pending_atom_evaluations\":{},\"discarded_prevalidated_successors\":{}}},\"successor_snapshot_equals_source_shared_total\":true}}",
            surface.molecule().graph().atom_count(), unresolved_seed_domains[0], unresolved_seed_domains[1], unresolved_seed_domains[2],
            inactive_center != usize::MAX,
            choices.len(), projector.0, projector.1, projector.2, projector.3,
            exact.0, exact.1, native.mixed_search_branch_count(), native.tetrahedral_factor_revision_count(), native.directional_ring_factor_revision_count(),
            copies.0.0, copies.0.1, copies.1.0, copies.1.1, copies.2.0, copies.2.1, traversal_copies.0, traversal_copies.1, writer.0, writer.1,
        );
        (strings, projector, (exact.0, exact.1))
    }

    let triangle_base = probe_triangle("triangle_unresolved_roles", 0, 0, false, false);
    let triangle_plain = probe_triangle("triangle_plus_disconnected_plain", 0, 2, false, false);
    let triangle_stereo = probe_triangle(
        "triangle_plus_disconnected_inactive_tetra",
        0,
        0,
        true,
        false,
    );
    let triangle_tail = probe_triangle("triangle_plus_connected_tail", 6, 0, false, false);
    let triangle_tail_stereo = probe_triangle(
        "triangle_tail_plus_distant_inactive_tetra",
        6,
        0,
        false,
        true,
    );
    assert_eq!(
        triangle_base.0, triangle_plain.0,
        "plain disconnected spectators preserve triangle alternatives"
    );
    assert_eq!(
        triangle_base.0, triangle_stereo.0,
        "inactive disconnected stereo preserves triangle alternatives"
    );
    assert_eq!(
        triangle_base.1, triangle_plain.1,
        "plain disconnected spectators must not add active projector work"
    );
    assert_eq!(
        triangle_base.1, triangle_stereo.1,
        "inactive disconnected stereo must not add active projector work"
    );
    assert_eq!(
        triangle_base.2, triangle_plain.2,
        "plain disconnected spectators must not add exact-search work"
    );
    assert_eq!(
        triangle_base.2, triangle_stereo.2,
        "inactive disconnected stereo must not add exact-search work"
    );
    assert_eq!(
        triangle_base.0, triangle_tail.0,
        "connected tail growth preserves root-local alternatives"
    );
    assert_eq!(
        triangle_base.0, triangle_tail_stereo.0,
        "distant inactive tetra preserves root-local alternatives"
    );
    assert!(
        triangle_tail.1 .0 + triangle_tail.1 .1 + triangle_tail.1 .2
            > triangle_base.1 .0 + triangle_base.1 .1 + triangle_base.1 .2,
        "connected tail growth must be attributed to more projector visits"
    );
    assert!(
        triangle_tail_stereo.1 .0 + triangle_tail_stereo.1 .1 + triangle_tail_stereo.1 .2
            > triangle_base.1 .0 + triangle_base.1 .1 + triangle_base.1 .2,
        "connected tail plus distant tetra is attributed to connected projector visits"
    );

    let (surface, atoms, bonds) = adjacent_ring_coupled_tetrahedral_fixture();
    let centers = [atoms[0], atoms[1]].map(|atom| surface.tetrahedral_center(atom).unwrap());
    let center_factors = centers.map(|center| {
        std::iter::once(center.root_layout_factor)
            .chain(
                center
                    .entry_layout_factors
                    .iter()
                    .map(|(_, factor)| *factor),
            )
            .collect::<Vec<_>>()
    });
    let center_domains = |state: &State| {
        centers
            .iter()
            .map(|center| {
                std::iter::once(
                    state
                        .structural
                        .semantic_domain(center.order_variable)
                        .len(),
                )
                .chain(std::iter::once(
                    state
                        .structural
                        .semantic_domain(center.role_pattern_variable)
                        .len(),
                ))
                .chain(
                    center
                        .bond_pattern_bits
                        .iter()
                        .map(|(bond, _)| state.structural.bond_role_domain_for_test(*bond).len()),
                )
                .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };

    let family_activity = |state: &State| -> [bool; 2] {
        let active = state.observe_raw().structural.active_factors;
        std::array::from_fn(|index| center_factors[index].iter().any(|id| active.contains(id)))
    };
    let shared_bond = bonds[0];
    let mut pending = vec![(initial(&surface), 0_usize)];
    let mut selected = None;
    let mut explored = 0;
    while let Some((state, atom_offset)) = pending.pop() {
        explored += 1;
        assert!(
            explored < 100_000,
            "coupled-center witness search must stay bounded"
        );
        if state.is_accepted() {
            continue;
        }
        let observed = state.observe_raw();
        let active_before = family_activity(&state);
        let before = center_domains(&state);
        let unresolved = before.iter().flatten().any(|size| *size > 1);
        let active_frame = observed.structural.traversal.active_frame.as_ref();
        let adjacent_center_attachment = active_frame.is_some_and(|frame| {
            frame.atom == atoms[1]
                && frame
                    .attachment_groups
                    .iter()
                    .flatten()
                    .any(|attachment| attachment.bond() == shared_bond)
        });
        if active_before.iter().any(|active| *active) && unresolved && adjacent_center_attachment {
            let shared_bond_before = state
                .structural
                .bond_role_domain_for_test(shared_bond)
                .len();
            let native = state.structural.constraints_for_test();
            native.reset_projector_work_counts();
            native.reset_exact_run_counts();
            native.reset_mixed_search_branch_count();
            native.reset_tetrahedral_factor_revision_count();
            native.reset_directional_ring_factor_revision_count();
            native.reset_solver_copy_counts();
            state.structural.reset_traversal_copy_counts();
            state.reset_writer_work_counts();
            let choices = ring_witness_choices(&state);
            let revisions = native.tetrahedral_factor_revision_count();
            let narrowed = choices.iter().find_map(|choice| {
                if !selected_atom_spelling_transition(&state, choice, atom_offset) {
                    return None;
                }
                let after = center_domains(choice.successor());
                let active_after = family_activity(choice.successor());
                let shared_bond_after = choice
                    .successor()
                    .structural
                    .bond_role_domain_for_test(shared_bond)
                    .len();
                let center_zero_narrows = after[0]
                    .iter()
                    .zip(&before[0])
                    .any(|(after, before)| after < before);
                (active_before == [false, true]
                    && active_after == [true, true]
                    && shared_bond_before == 1
                    && shared_bond_after == shared_bond_before
                    && center_zero_narrows)
                    .then(|| {
                        (
                            after,
                            active_after,
                            shared_bond_after,
                            choice.text().to_owned(),
                        )
                    })
            });
            if revisions > 0 {
                if let Some((after, active_after, shared_bond_after, text)) = narrowed {
                    selected = Some((
                        state,
                        choices,
                        before,
                        after,
                        revisions,
                        active_before,
                        active_after,
                        shared_bond_before,
                        shared_bond_after,
                        text,
                    ));
                    break;
                }
            }
            for choice in choices {
                if !selected_atom_spelling_transition(&state, &choice, atom_offset) {
                    continue;
                }
                let token = choice.text().to_owned();
                let successor = choice.into_successor();
                let next_offset =
                    next_atom_spelling_offset(&state, &successor, atom_offset, &token);
                pending.push((successor, next_offset));
            }
        } else {
            pending.extend(declared_successors(&state, atom_offset));
        }
    }

    let (
        state,
        choices,
        before,
        after,
        revisions,
        active_before,
        active_after,
        shared_bond_before,
        shared_bond_after,
        narrowing_text,
    ) = selected.expect(
        "adjacent-center residual shared-bond attachment must have a bounded successor that narrows a center domain and revises a tetra factor",
    );
    assert_eq!(state.active_atom(), Some(atoms[1]));
    let source_observation = state.observe_raw();
    let source_frame = source_observation
        .structural
        .traversal
        .active_frame
        .as_ref()
        .unwrap();
    assert_eq!(source_frame.entry_bond, Some(bonds[5]));
    assert!(source_frame
        .attachment_groups
        .iter()
        .flatten()
        .any(|attachment| attachment.bond() == shared_bond));
    assert_eq!(
        shared_bond_before, 1,
        "shared center bond is already role-fixed"
    );
    assert_eq!(
        active_before,
        [false, true],
        "center 1 is active before the adjacent center is entered"
    );
    assert_eq!(
        active_after,
        [true, true],
        "center 0 activates while center 1 remains active"
    );
    assert_eq!(shared_bond_before, 1);
    assert_eq!(
        shared_bond_after, 1,
        "shared bond remains role-fixed across the local propagation"
    );
    assert!(!state.pending.is_some());
    assert_eq!(narrowing_text, "[C@H]");
    assert_eq!(revisions, 2);
    assert!(revisions > 0);
    assert!(after.iter().enumerate().any(|(center, values)| {
        values
            .iter()
            .zip(&before[center])
            .any(|(after, before)| after < before)
    }));
    let native = state.structural.constraints_for_test();
    let projector = native.projector_work_counts();
    let exact = native.exact_run_counts();
    let copies = native.solver_copy_counts();
    let traversal_copies = state.structural.traversal_copy_counts();
    let writer = state.writer_work_counts();
    let successor = choices
        .iter()
        .find(|choice| {
            center_domains(choice.successor()) == after
                && family_activity(choice.successor()) == active_after
                && choice
                    .successor()
                    .structural
                    .bond_role_domain_for_test(shared_bond)
                    .len()
                    == shared_bond_after
                && choice.text() == narrowing_text
        })
        .expect("selected narrowing successor must be retained");
    assert_eq!(
        successor
            .successor()
            .structural
            .constraints_for_test()
            .projector_work_counts(),
        projector
    );
    assert_eq!(successor.successor().writer_work_counts(), writer);
    println!(
        "{{\"schema\":\"south_star2_native_work_count_v1\",\"case\":\"adjacent_center_activation_positive_revision\",\"source_prefix\":{{\"active_atom\":{},\"entry_bond\":{},\"pending_present\":{},\"shared_bond_attached_as_residual\":true,\"active_center_families\":[{},{}],\"shared_bond_role_domain\":{}}},\"successor\":{{\"choice_text\":{},\"active_atom\":{},\"entry_bond\":{},\"pending_present\":{},\"active_center_families\":[{},{}],\"shared_bond_role_domain\":{}}},\"before_domains\":{:?},\"after_domains\":{:?},\"event\":\"adjacent_center_activation_local_propagation_not_exact_search\",\"choices\":{},\"projector\":{{\"first_edge_visits\":{},\"atom_visits\":{},\"second_edge_visits\":{},\"potential_layout_checks\":{}}},\"exact_runs\":{{\"binary\":{},\"mixed\":{}}},\"mixed_branch_points\":{},\"tetra_factor_revisions\":{},\"directional_ring_revisions\":{},\"copy_pages\":{{\"solver_domains\":{{\"index_nodes\":{},\"value_pages\":{}}},\"solver_active_factors\":{{\"index_nodes\":{},\"value_pages\":{}}},\"solver_activation_adjacency\":{{\"index_nodes\":{},\"value_pages\":{}}},\"traversal_aggregate\":{{\"index_nodes\":{},\"value_pages\":{}}}}},\"writer_frontier\":{{\"pending_atom_evaluations\":{},\"discarded_prevalidated_successors\":{}}},\"successor_snapshot_equals_source_shared_total\":true}}",
        state.active_atom().unwrap().index(),
        state.observe_raw().structural.traversal.active_frame.as_ref().and_then(|frame| frame.entry_bond).map(|bond| bond.index()).unwrap_or(usize::MAX),
        state.pending.is_some(), active_before[0], active_before[1], shared_bond_before,
        json_quote(&narrowing_text),
        successor.successor().active_atom().map(|atom| atom.index()).unwrap_or(usize::MAX),
        successor.successor().observe_raw().structural.traversal.active_frame.as_ref().and_then(|frame| frame.entry_bond).map(|bond| bond.index()).unwrap_or(usize::MAX),
        successor.successor().pending.is_some(), active_after[0], active_after[1], shared_bond_after, before, after, choices.len(),
        projector.0, projector.1, projector.2, projector.3, exact.0, exact.1,
        native.mixed_search_branch_count(), revisions, native.directional_ring_factor_revision_count(),
        copies.0.0, copies.0.1, copies.1.0, copies.1.1, copies.2.0, copies.2.1,
        traversal_copies.0, traversal_copies.1, writer.0, writer.1,
    );

    let mut model = crate::model::ConstraintModelBuilder::new();
    let variables = (0..3)
        .map(|_| {
            model
                .add_variable(crate::model::BondRole::role_domain())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let triangle_atoms = [
        crate::ids::AtomId::new(0),
        crate::ids::AtomId::new(1),
        crate::ids::AtomId::new(2),
    ];
    let edges = [(0, 1), (1, 2), (2, 0)].map(|(a, b)| {
        crate::model::SpanningTreeEdge::new(variables[a], triangle_atoms[a], triangle_atoms[b])
    });
    model.add_spanning_tree(triangle_atoms, edges).unwrap();
    let source = NativeSolverState::initial(Arc::new(model.build())).unwrap();
    source.reset_projector_work_counts();
    let rejected = source.with_restrictions(variables.iter().copied().map(|variable| {
        (
            variable,
            crate::model::BondRole::Traversal.singleton_domain(),
        )
    }));
    assert!(matches!(
        rejected,
        Err(crate::native::NativeSolverError::Contradiction)
    ));
    let rejected_work = source.projector_work_counts();
    assert!(
        rejected_work.0 > 0,
        "contradictory candidate must visit edges before rejection"
    );
    println!(
        "{{\"schema\":\"south_star2_native_work_count_v1\",\"case\":\"rejected_candidate_source_lineage_observer\",\"event\":\"all_triangle_edges_forced_traversal_then_cycle_rejected\",\"surviving_successors\":0,\"source_projector_after_rejection\":{{\"first_edge_visits\":{},\"atom_visits\":{},\"second_edge_visits\":{},\"potential_layout_checks\":{}}},\"source_snapshot_retains_rejected_work\":true}}",
        rejected_work.0, rejected_work.1, rejected_work.2, rejected_work.3,
    );
}
#[test]
fn exhausted_ring_labels_reject_a_forced_new_ring_before_branch_entry() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..5).map(|_| graph.add_atom().unwrap()).collect::<Vec<_>>();
    let bonds = [(0, 1), (2, 3), (2, 4), (3, 4)]
        .into_iter().map(|(a,b)|graph.add_bond(atoms[a], atoms[b]).unwrap()).collect::<Vec<_>>();
    let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(graph.build()),
        (0..atoms.len()).map(|_|PreparedAtomToken::Fixed("C".to_owned())).collect(),
        vec![NonStereoBondToken::Elided; bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, 0,
    ).unwrap();
    let state = initial(&surface);
    let (_, rooted) = only_root_atom(&state, atoms[0]);
    let opening = rooted.choices().unwrap().into_iter().find(|choice|
        choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[0])
    ).expect("first ethane ring opens");
    let held = opening.into_successor();
    let (_, pending_triangle) = only_component_separator(&held, atoms[2]);
    let (_, triangle) = only_choice(&pending_triangle, "C");
    let branch_incident = incident(&surface, atoms[2], bonds[1]);
    let rescue = triangle.choices().unwrap().into_iter().find(|choice|
        choice.successor().pending == Some(PendingEmission::SegmentRootAtom(atoms[1]))
    ).expect("dot path reaches the held ethane endpoint").into_successor();
    let (_, rescue_atom) = only_choice(&rescue, "C");
    let released = rescue_atom.choices().unwrap().into_iter().find(|choice|
        choice.successor().labels.is_clean()
    ).expect("closing the held ethane ring frees its visible label").into_successor();
    let (_, restored) = only_choice(&released, ")");
    assert!(restored.choices().unwrap().iter().any(|choice|
        matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == branch_incident)
    ), "releasing a label restores the triangle branch");
    let branch = restored.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == branch_incident)
    ).unwrap().into_successor();
    let (_, child) = only_choice(&branch, "C");
    let ring = child.choices().unwrap().into_iter().find(|choice|
        choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[3])
            && choice.successor().pending.is_none()
    ).expect("rescued branch opens its ring").into_successor();
    let (_, parent) = only_choice(&ring, ")");
    let (_, pending_last) = only_choice(&parent, "-");
    let (_, last) = only_choice(&pending_last, "C");
    assert!(last.choices().unwrap().iter().any(|choice| choice.successor().is_accepted()));
    assert!(!triangle.choices().unwrap().iter().any(|choice|
        matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == branch_incident)
    ), "a branch child that can only demand an unavailable ring label must not be offered");
}

#[test]
fn opening_the_last_ring_label_keeps_a_traversal_continuation() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..3).map(|_| graph.add_atom().unwrap()).collect::<Vec<_>>();
    let bonds = [(0, 1), (0, 2), (1, 2)]
        .into_iter().map(|(a, b)| graph.add_bond(atoms[a], atoms[b]).unwrap())
        .collect::<Vec<_>>();
    let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(graph.build()),
        (0..3).map(|_| PreparedAtomToken::Fixed("C".to_owned())).collect(),
        vec![NonStereoBondToken::Elided; bonds.len()],
        Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder },
        true, 0,
    ).unwrap();
    let state = initial(&surface);
    let (_, rooted) = only_root_atom(&state, atoms[0]);
    let opening = rooted.choices().unwrap().into_iter().find(|choice|
        choice.successor().pending.is_none()
            && choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[0])
    ).expect("one ring label is available").into_successor();
    let partition = surface.molecule().bond_role_partition(bonds[1]).unwrap();
    assert!(opening.structural.bond_decision_domain(bonds[1])
        .is_subset_of(partition.traversal_values()));
    assert!(opening.choices().unwrap().iter().any(|choice|
        choice.successor().active_atom() == Some(atoms[2])
            || matches!(&choice.successor().pending, Some(PendingEmission::InlineAtom(entry)) if entry.incident ==
                incident(&surface, atoms[0], bonds[1])
            )
    ));
}

#[test]
#[ignore = "public 100-label capacity diagnostic; run with bounded single-test command"]
fn public_ring_label_capacity_rejects_a_forced_101st_opening() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..203).map(|_| graph.add_atom().unwrap()).collect::<Vec<_>>();
    let mut bonds = (0..100).map(|i|graph.add_bond(atoms[2*i], atoms[2*i+1]).unwrap()).collect::<Vec<_>>();
    let triangle = [
        graph.add_bond(atoms[200], atoms[201]).unwrap(),
        graph.add_bond(atoms[200], atoms[202]).unwrap(),
        graph.add_bond(atoms[201], atoms[202]).unwrap(),
    ];
    bonds.extend(triangle);
    let surface = PreparedNonStereo::with_atom_tokens_and_directional_for_writer(
        PreparedMolecule::new(graph.build()),
        (0..atoms.len()).map(|_|PreparedAtomToken::Fixed("C".to_owned())).collect(),
        vec![NonStereoBondToken::Elided; bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true,
    ).unwrap();
    let mut state = initial(&surface);
    for i in 0..100 {
        let rooted = if i == 0 { only_root_atom(&state, atoms[0]).1 } else {
            let (_, pending) = only_component_separator(&state, atoms[2*i]);
            only_choice(&pending, "C").1
        };
        let opening = rooted.choices().unwrap().into_iter().find(|choice|
            choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[i])
        ).expect("ethane ring opens");
        state = opening.into_successor();
        assert_eq!(state.labels.bonds_by_slot.len(), i+1);
    }
    let (_, pending_triangle) = only_component_separator(&state, atoms[200]);
    let (_, triangle_root) = only_choice(&pending_triangle, "C");
    let branch_incident = incident(&surface, atoms[200], bonds[100]);
    let choices = triangle_root.choices().unwrap();
    let rescue = choices.iter().find(|choice|
        choice.successor().pending == Some(PendingEmission::SegmentRootAtom(atoms[1]))
    ).expect("dot path reaches the first held ethane endpoint").successor().clone();
    let (_, rescue_atom) = only_choice(&rescue, "C");
    assert!(rescue_atom.choices().unwrap().iter().any(|choice|
        choice.successor().labels.bonds_by_slot.len() == 99
            && !choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[0])
    ), "closing a held ethane ring frees one visible label");
    assert!(!triangle_root.choices().unwrap().iter().any(|choice|
        matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == branch_incident)
    ), "a branch child that can only demand an unavailable ring label must not be offered");
}

#[test]
fn positive_ring_capacity_rejects_diamond_branch_requiring_two_openings() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..4).map(|_| graph.add_atom().unwrap()).collect::<Vec<_>>();
    let bonds = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
        .into_iter().map(|(a, b)| graph.add_bond(atoms[a], atoms[b]).unwrap())
        .collect::<Vec<_>>();
    let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(graph.build()),
        (0..4).map(|_| PreparedAtomToken::Fixed("C".to_owned())).collect(),
        vec![NonStereoBondToken::Elided; bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, 0,
    ).unwrap();
    let state = initial(&surface);
    assert!(surface.event_schedule_factor.is_some(),
        "ordinary-only cycle graphs prepare the event schedule in the test namespace");
    let (_, root) = only_root_atom(&state, atoms[0]);
    let source = root.observe_raw();
    assert!(!root.choices().unwrap().iter().any(|choice|
        matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident ==
            incident(&surface, atoms[0], bonds[0])
        )
    ), "diamond branch needs two concurrent ring labels before leaving its local ring phase");
    assert_eq!(root.observe_raw(), source, "rejected branches must leave the source untouched");
}

fn live_ring_cut_writer_prefix(maximum_label: usize) ->
    (PreparedNonStereo, State, [AtomId; 8], [BondId; 9]) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms: [AtomId; 8] = std::array::from_fn(|_| graph.add_atom().unwrap());
    let bonds = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5),
        (5, 6), (5, 7), (0, 4), (6, 7)]
        .map(|(a, b)| graph.add_bond(atoms[a], atoms[b]).unwrap());
    let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(graph.build()),
        (0..8).map(|_| PreparedAtomToken::Fixed("C".to_owned())).collect(),
        vec![NonStereoBondToken::Elided; bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, maximum_label,
    ).unwrap();
    assert!(surface.live_ring_cut_factor.is_some());
    let mut state = initial(&surface);
    let fixed_roles = bonds.iter().copied().enumerate().map(|(index, bond)| {
        (surface.molecule().bond_decision_variable(bond).unwrap(),
            if index < 7 { BondRole::Traversal.singleton_domain() }
            else { BondRole::Ring.singleton_domain() })
    }).collect::<Vec<_>>();
    state.structural = state.structural.transitioned_semantics(&fixed_roles, &[])
        .unwrap().unwrap_consistent();
    let (_, p) = only_root_atom(&state, atoms[0]);
    let opened = p.choices().unwrap().into_iter().find(|choice|
        choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[7]))
        .expect("P opens P-H").into_successor();
    let child = |state: State, from: usize, bond: BondId| {
        let pending = state.choices().unwrap().into_iter().find(|choice| {
            matches!(&choice.successor().pending,
                Some(PendingEmission::InlineAtom(entry) | PendingEmission::BranchTraversalEmission(entry))
                    if entry.incident == incident(&surface, atoms[from], bond))
        }).expect("fixed traversal child is offered").into_successor();
        only_choice(&pending, "C").1
    };
    let a = child(opened, 0, bonds[0]);
    let x = child(a, 1, bonds[1]);
    assert_eq!(x.active_atom(), Some(atoms[2]));
    assert_eq!(x.labels.bonds_by_slot.len(), 1);
    assert!(x.structural.factor_is_active(surface.event_schedule_factor.unwrap()));
    (surface, x, atoms, bonds)
}

#[test]
fn live_ring_cut_writer_prefix_keeps_h_first_while_event_rejects_c_first() {
    let (surface, x, atoms, bonds) = live_ring_cut_writer_prefix(0);
    let b_order = surface.ordinary_order_pairs[atoms[3].index()].iter().find(|pair|
        [pair.first, pair.second] == [bonds[3], bonds[4]])
        .expect("B has H/C local order");
    let c_first = Domain::singleton(u8::from(b_order.first == bonds[3])).unwrap();
    let h_first = Domain::singleton(u8::from(b_order.first == bonds[4])).unwrap();
    let c = x.structural.restricted_semantic_domains_for_test(&[(b_order.variable, c_first)])
        .unwrap();
    let c_rejected = match c {
        Consistency::Contradiction => true, // EventSchedule can reject during restriction.
        Consistency::Consistent(c) => {
            let mut c_state = x.clone();
            c_state.structural = c;
            matches!(c_state.admit_exact(),
                SuccessorAttempt::Rejected(CandidateRejection::Contradiction))
        }
    };
    assert!(c_rejected, "EventSchedule must reject C-before-H before publication");
    let h = x.structural.restricted_semantic_domains_for_test(&[(b_order.variable, h_first)])
        .unwrap().unwrap_consistent();
    let mut h_state = x.clone();
    h_state.structural = h;
    assert!(matches!(h_state.admit_exact(), SuccessorAttempt::Accepted(_)));

    let unresolved = x.structural.propagated_semantic_domains_for_test(&[
        (b_order.variable, h_first)]).unwrap().unwrap_consistent();
    let probe = unresolved.solver_for_test().clone();
    probe.reset_live_cut_phase_counts();
    assert!(matches!(unresolved.admit_live_ring_cut(surface.live_ring_cut_factor.unwrap(),
        &[bonds[7]], 1).unwrap(), Consistency::Consistent(_)));
    let (_, _, deferred, temporal) = probe.live_cut_phase_counts();
    assert!(deferred > 0 && temporal > 0,
        "unresolved physical scope follows ordinary Event branches without cut-driven assignment");

    let c_order = surface.ordinary_order_pairs[atoms[5].index()].iter().find(|pair|
        [bonds[5], bonds[6]].contains(&pair.first)
            && [bonds[5], bonds[6]].contains(&pair.second)).unwrap();
    let mut fixed_scope = atoms.iter().copied().enumerate().map(|(index, atom)|
        (surface.molecule().root_variable(atom).unwrap(),
            Domain::singleton(u8::from(index == 0)).unwrap())).collect::<Vec<_>>();
    fixed_scope.extend(surface.ordinary_order_pairs.iter().flat_map(|pairs| pairs.iter())
        .filter(|pair| pair.variable != b_order.variable && pair.variable != c_order.variable)
        .map(|pair| (pair.variable, Domain::singleton(u8::from(
            ([bonds[0], bonds[7]].contains(&pair.first)
                && [bonds[0], bonds[7]].contains(&pair.second)
                && pair.first == bonds[0])
                || ([bonds[3], bonds[7]].contains(&pair.first)
                    && [bonds[3], bonds[7]].contains(&pair.second)
                    && pair.first == bonds[7]))).unwrap())));
    let fixed_case = |b: Domain, c: u8| {
        let mut restrictions = fixed_scope.clone();
        restrictions.push((b_order.variable, b));
        restrictions.push((c_order.variable, Domain::singleton(c).unwrap()));
        x.structural.propagated_semantic_domains_for_test(&restrictions)
            .unwrap().unwrap_consistent()
    };
    let obstructed = fixed_case(c_first, 0);
    assert!(surface.event_schedule_visits.iter().flat_map(|visit| visit.rank_bits.iter())
        .any(|rank| obstructed.solver_for_test().domain_bound(*rank).unwrap().len() > 1));
    assert!(matches!(obstructed.clone().admit_exact().unwrap(), Consistency::Contradiction));
    let probe = obstructed.solver_for_test().clone();
    probe.reset_live_cut_phase_counts();
    assert!(matches!(obstructed.admit_live_ring_cut(surface.live_ring_cut_factor.unwrap(),
        &[bonds[7]], 1).unwrap(), Consistency::Contradiction));
    let (checks, scans, deferred, temporal) = probe.live_cut_phase_counts();
    assert_eq!((checks, scans, deferred, temporal), (1, 1, 0, 0),
        "fixed obstruction must be rejected before any Event branch");
    for c in [0, 1] {
        let surviving = fixed_case(h_first, c);
        assert!(matches!(surviving.clone().admit_exact().unwrap(), Consistency::Consistent(_)),
            "Event supports C child order {c} after H closes P-H");
        let probe = surviving.solver_for_test().clone();
        probe.reset_live_cut_phase_counts();
        assert!(matches!(surviving.admit_live_ring_cut(surface.live_ring_cut_factor.unwrap(),
            &[bonds[7]], 1).unwrap(), Consistency::Consistent(_)));
        let (checks, _, _, temporal) = probe.live_cut_phase_counts();
        assert_eq!(checks, 1, "fixed physical assignment is checked once");
        assert!(temporal > 0, "Event still completes the surviving physical assignment");
    }
}

#[test]
fn live_ring_cut_two_slots_preserve_both_b_and_c_child_orders() {
    let (surface, x, atoms, bonds) = live_ring_cut_writer_prefix(1);
    let b_order = surface.ordinary_order_pairs[atoms[3].index()].iter().find(|pair|
        [bonds[3], bonds[4]].contains(&pair.first)
            && [bonds[3], bonds[4]].contains(&pair.second)).unwrap();
    let c_order = surface.ordinary_order_pairs[atoms[5].index()].iter().find(|pair|
        [bonds[5], bonds[6]].contains(&pair.first)
            && [bonds[5], bonds[6]].contains(&pair.second)).unwrap();
    for b in [0, 1] {
        for c in [0, 1] {
            let restricted = x.structural.propagated_semantic_domains_for_test(&[
                (b_order.variable, Domain::singleton(b).unwrap()),
                (c_order.variable, Domain::singleton(c).unwrap()),
            ]).unwrap().unwrap_consistent();
            assert!(matches!(restricted.clone().admit_exact().unwrap(), Consistency::Consistent(_)),
                "two-slot Event reference supports B={b}, C={c}");
            let probe = restricted.solver_for_test().clone();
            probe.reset_live_cut_phase_counts();
            assert!(matches!(restricted.admit_live_ring_cut(surface.live_ring_cut_factor.unwrap(),
                &[bonds[7]], 2).unwrap(), Consistency::Consistent(_)),
                "two-slot cut admission preserves B={b}, C={c}");
            let (checks, scans, deferred, _) = probe.live_cut_phase_counts();
            assert_eq!((checks, scans, deferred), (0, 0, 0),
                "one held ring below two slots skips the cut phase");
        }
    }
}

#[test]
fn fixed_ring_path_dot_order_changes_peak_label_demand() {
    fn accepted_suffix(start: State) -> Option<State> {
        let mut pending = vec![start];
        let mut explored = 0;
        while let Some(state) = pending.pop() {
            explored += 1;
            assert!(explored < 10_000, "three-atom path suffix must stay finite");
            if state.is_accepted() { return Some(state); }
            pending.extend(state.choices().unwrap().into_iter().map(Choice::into_successor));
        }
        None
    }

    for maximum_label in [0, 1] {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let ab = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let bc = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
            PreparedMolecule::new(graph.build()),
            (0..3).map(|_| PreparedAtomToken::Fixed("C".to_owned())).collect(),
            vec![NonStereoBondToken::Elided; 2], Vec::new(),
            WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true,
            0,
        ).unwrap();
        let mut start = initial(&surface);
        start.labels.maximum_spelling_label = Some(maximum_label);
        let (_, rooted) = only_root_atom(&start, atoms[0]);
        let mut rooted = rooted;
        let order = &surface.ordinary_order_pairs[atoms[1].index()];
        assert_eq!(order.len(), 1);
        assert_eq!((order[0].first, order[0].second), (ab, bc));
        let mut physical = atoms.into_iter().map(|atom|
            (surface.molecule().root_variable(atom).unwrap(), Domain::singleton(1).unwrap())
        ).collect::<Vec<_>>();
        physical.push((order[0].variable, Domain::singleton(0).unwrap()));
        rooted.structural = rooted.structural.restricted_semantic_domains_for_test(&physical)
            .unwrap().unwrap_consistent();
        rooted.structural = rooted.structural.restricted_bond_domains_for_test(&[
            (ab, BondRole::Ring.singleton_domain()),
            (bc, BondRole::Ring.singleton_domain()),
        ]).unwrap().unwrap_consistent();
        let opened = rooted.choices().unwrap().into_iter().filter(|choice|
            choice.successor().pending.is_none()
                && choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == ab)
        ).map(Choice::into_successor).collect::<Vec<_>>();
        assert!(!opened.is_empty(), "A must be able to open fixed ring AB");
        for state in opened {
            assert_eq!(state.labels.bonds_by_slot.len(), 1);
            let offered = state.choices().unwrap().into_iter().filter_map(|choice|
                match choice.successor().pending {
                    Some(PendingEmission::SegmentRootAtom(atom)) if atom == atoms[1] || atom == atoms[2] =>
                        Some((atom, choice.text().to_owned(), choice.into_successor())),
                    _ => None,
                }
            ).collect::<Vec<_>>();
            assert!(offered.iter().any(|(atom, _, _)| *atom == atoms[1]),
                "B-first must be offered at either label capacity");
            assert_eq!(offered.iter().any(|(atom, _, _)| *atom == atoms[2]), maximum_label == 1,
                "C-first needs two simultaneously open ring labels");
            for (atom, text, pending_root) in offered {
                let terminal = accepted_suffix(pending_root).unwrap_or_else(||
                    panic!("offered dot root {atom:?} via {text:?} needs an accepted suffix"));
                assert_eq!(terminal.structural.bond_decision_domain(ab), BondRole::Ring.singleton_domain());
                assert_eq!(terminal.structural.bond_decision_domain(bc), BondRole::Ring.singleton_domain());
                for atom in atoms {
                    assert_eq!(terminal.structural.semantic_domain(surface.molecule().root_variable(atom).unwrap()),
                        Domain::singleton(1).unwrap());
                }
                assert_eq!(terminal.structural.semantic_domain(order[0].variable),
                    Domain::singleton(0).unwrap(), "B emits AB before BC");
            }
        }
    }
}

// This finite reference is deliberately outside the writer's Event relation.
fn ring_boundary_size(mask: usize, edges: &[(usize, usize)]) -> usize {
    edges.iter().filter(|&&(a, b)| ((mask >> a) & 1) != ((mask >> b) & 1)).count()
}

fn ring_order_peak(order: &[usize], edges: &[(usize, usize)]) -> usize {
    let mut visited = 0;
    let mut peak = 0;
    for &atom in order {
        peak = peak.max(ring_boundary_size(visited, edges));
        visited |= 1 << atom;
        peak = peak.max(ring_boundary_size(visited, edges));
    }
    peak
}

fn ring_suffix_fits(prefix: &[usize], next: usize, atom_count: usize,
    edges: &[(usize, usize)], capacity: usize) -> bool {
    fn search(order: &mut Vec<usize>, atom_count: usize, edges: &[(usize, usize)],
        capacity: usize) -> bool {
        if ring_order_peak(order, edges) > capacity { return false; }
        if order.len() == atom_count { return true; }
        for atom in 0..atom_count {
            if order.contains(&atom) { continue; }
            order.push(atom);
            let fits = search(order, atom_count, edges, capacity);
            order.pop();
            if fits { return true; }
        }
        false
    }
    let mut order = prefix.to_vec();
    order.push(next);
    search(&mut order, atom_count, edges, capacity)
}

fn ring_vertex_orders(atom_count: usize) -> Vec<Vec<usize>> {
    fn extend(order: &mut Vec<usize>, atom_count: usize, result: &mut Vec<Vec<usize>>) {
        if order.len() == atom_count {
            result.push(order.clone());
            return;
        }
        for atom in 0..atom_count {
            if order.contains(&atom) { continue; }
            order.push(atom);
            extend(order, atom_count, result);
            order.pop();
        }
    }
    let mut result = Vec::new();
    extend(&mut Vec::new(), atom_count, &mut result);
    result
}

fn ring_layout_action(states: Vec<State>, bond: BondId) -> Vec<State> {
    let mut next = Vec::new();
    for state in states {
        let before_count = next.len();
        let before = state.labels.bonds_by_slot.iter().find_map(|(&slot, &held)|
            (held == bond).then_some(slot));
        next.extend(state.choices().unwrap().into_iter().filter(|choice| {
            let after = choice.successor().labels.bonds_by_slot.iter()
                .find_map(|(&slot, &held)| (held == bond).then_some(slot));
            choice.successor().pending.is_none()
                && if before.is_some() { after.is_none() }
                    else { after == Some(state.labels.next_available()) }
        }).map(Choice::into_successor));
        assert!(next.len() > before_count, "each concrete prefix must retain its ring action");
        assert!(next.len() <= 4096, "ring layout witness expansion bound");
    }
    next
}

fn ring_layout_atom_spelling_completes(state: &State, atom: AtomId, prefix: &str) -> bool {
    let hydrogens = state.surface.atom_facts(atom).non_graph_hydrogens;
    let bracket = match hydrogens {
        0 => "[C]".to_owned(),
        1 => "[CH]".to_owned(),
        count => format!("[CH{count}]"),
    };
    ["C".to_owned(), bracket].into_iter().filter(|target| target.starts_with(prefix))
        .any(|target| consume_atom_spelling_states(state.clone(), &target, prefix.len())
            .into_iter().any(|completed| completed.structural.atom_is_visited(atom)))
}

#[test]
fn all_up_to_three_atom_fixed_ring_layouts_match_boundary_continuations() {
    let mut graphs = 0;
    let mut orders = 0;
    let mut prefixes = 0;
    let mut distinct_orders = 0;
    let mut equal_text_distinct_atom_entries = false;
    for atom_count in 1..=3 {
        let possible_edges = (0..atom_count).flat_map(|a|
            (a+1..atom_count).map(move |b| (a, b))).collect::<Vec<_>>();
        for edge_mask in 0..(1 << possible_edges.len()) {
        let edges = possible_edges.iter().copied().enumerate()
            .filter_map(|(bit, edge)| ((edge_mask >> bit) & 1 == 1).then_some(edge))
            .collect::<Vec<_>>();
        let mut graph = PreparedGraphBuilder::new();
        let atoms = (0..atom_count).map(|_| graph.add_atom().unwrap()).collect::<Vec<_>>();
        let bonds = edges.iter().map(|&(a, b)| graph.add_bond(atoms[a], atoms[b]).unwrap())
            .collect::<Vec<_>>();
        let molecule = PreparedMolecule::new(graph.build());
        for capacity in 1..=3 {
            let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
                molecule.clone(),
                vec![PreparedAtomToken::Fixed("C".to_owned()); atom_count],
                vec![NonStereoBondToken::Elided; edges.len()], Vec::new(),
                WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true,
                capacity - 1,
            ).unwrap();
            let mut start = initial(&surface);
            let restrictions = atoms.iter().map(|&atom|
                (surface.molecule().root_variable(atom).unwrap(), Domain::singleton(1).unwrap()))
                .collect::<Vec<_>>();
            // Local incidence order stays free: the witness chooses closures first.
            start.structural = start.structural.restricted_semantic_domains_for_test(&restrictions)
                .unwrap().unwrap_consistent();
            start.structural = start.structural.restricted_bond_domains_for_test(&bonds.iter()
                .copied().map(|bond| (bond, BondRole::Ring.singleton_domain()))
                .collect::<Vec<_>>()).unwrap().unwrap_consistent();
            graphs += 1;
            let mut seen_orders = BTreeSet::new();

            fn visit(state: State, prefix: &mut Vec<usize>, atoms: &[AtomId],
                bonds: &[BondId], edges: &[(usize, usize)], capacity: usize,
                orders: &mut usize, prefixes: &mut usize,
                seen_orders: &mut BTreeSet<Vec<usize>>) {
                let visited = prefix.iter().fold(0, |mask, &atom| mask | (1 << atom));
                let held = bonds.iter().enumerate().filter_map(|(i, &bond)|
                    state.labels.bonds_by_slot.values().any(|&held| held == bond).then_some(i))
                    .collect::<Vec<_>>();
                let expected_held = edges.iter().enumerate().filter_map(|(i, &(a, b))|
                    (((visited >> a) & 1) != ((visited >> b) & 1)).then_some(i))
                    .collect::<Vec<_>>();
                assert_eq!(held, expected_held, "held physical identities at {prefix:?}");
                assert_eq!(held.len(), ring_boundary_size(visited, edges));
                assert!(state.pending.is_none() && state.atom_spelling.is_none());
                if (edges == [(0, 1), (1, 2)] && capacity == 1)
                    || (edges == [(0, 1), (0, 2), (1, 2)] && capacity == 2) {
                    let actual_completed = atoms.iter().map(|&atom|
                        state.structural.atom_is_visited(atom)).collect::<Vec<_>>();
                    assert_eq!(actual_completed, (0..atoms.len()).map(|atom|
                        (visited >> atom) & 1 == 1).collect::<Vec<_>>());
                    crate::boundary_cut_plan::check_actual_completed_ring_prefix(
                        atoms.len(), edges, capacity, prefix);
                }
                *prefixes += 1;
                assert!(*prefixes <= 50_000, "ring layout concrete prefix bound");
                if prefix.len() == atoms.len() {
                    let mut terminal = vec![state];
                    let mut accepted = false;
                    for _ in 0..8 {
                        let Some(current) = terminal.pop() else { break; };
                        if current.is_accepted() { accepted = true; break; }
                        terminal.extend(current.choices().unwrap().into_iter().map(Choice::into_successor));
                    }
                    assert!(accepted, "finished atom order {prefix:?} must reach acceptance");
                    *orders += 1;
                    seen_orders.insert(prefix.clone());
                    return;
                }
                let choices = state.choices().unwrap();
                let offered = choices.iter().filter_map(|choice| match choice.successor().pending {
                    Some(PendingEmission::SegmentRootAtom(atom)) =>
                        atoms.iter().position(|&candidate| candidate == atom),
                    _ => None,
                }).collect::<BTreeSet<_>>();
                let expected = (0..atoms.len()).filter(|&atom| (visited >> atom) & 1 == 0
                    && ring_suffix_fits(prefix, atom, atoms.len(), edges, capacity))
                    .collect::<BTreeSet<_>>();
                assert_eq!(offered, expected, "productive next labelled atom at {prefix:?}, capacity {capacity}");
                for choice in choices.into_iter().filter(|choice| matches!(choice.successor().pending,
                    Some(PendingEmission::SegmentRootAtom(atom)) if offered.contains(&atoms.iter().position(|&a| a == atom).unwrap()))) {
                    let atom = match choice.successor().pending { Some(PendingEmission::SegmentRootAtom(atom)) => atom, _ => unreachable!() };
                    let index = atoms.iter().position(|&a| a == atom).unwrap();
                    assert!(ring_layout_atom_spelling_completes(choice.successor(), atom, ""),
                        "each offered dot-root choice must have an actual atom spelling");
                    let completions = completed_atom_witnesses(choice.successor(), atom);
                    assert!(!completions.is_empty(),
                        "each offered dot-root choice must complete atom {atom:?} at {prefix:?}");
                    for (_, current) in completions {
                        // Follow every labelled successor with the chosen atom, then close
                        // all rings to finished atoms before opening rings to future atoms.
                        let mut wanted = edges.iter().enumerate().filter_map(|(i, &(a, b))|
                            ((a == index || b == index) && ((visited >> (a ^ b ^ index)) & 1 == 1))
                            .then_some(i)).chain(edges.iter().enumerate().filter_map(|(i, &(a, b))|
                            ((a == index || b == index) && ((visited >> (a ^ b ^ index)) & 1 == 0))
                            .then_some(i))).collect::<Vec<_>>();
                        let mut states = vec![current];
                        for edge in wanted.drain(..) {
                            states = ring_layout_action(states, bonds[edge]);
                        }
                        prefix.push(index);
                        for current in states {
                            visit(current, prefix, atoms, bonds, edges, capacity, orders, prefixes,
                                seen_orders);
                        }
                        prefix.pop();
                    }
                }
            }
            // The first root is a direct atom entry. Compare those labelled
            // entries before following each one through its remaining spelling.
            let expected_roots = (0..atom_count).filter(|&atom|
                ring_suffix_fits(&[], atom, atom_count, &edges, capacity))
                .collect::<BTreeSet<_>>();
            let root_choices = match start.choices() {
                Ok(choices) => choices,
                Err(ChoiceFailure::Invariant(WriterInvariantFailure::AllCandidatesSemanticallyRejected { .. }))
                    if expected_roots.is_empty() => Vec::new(),
                Err(failure) => panic!("initial ring-layout choice failure: {failure:?}"),
            };
            let mut root_text_atoms = BTreeMap::<String, BTreeSet<AtomId>>::new();
            let root_atom = |choice: &Choice<State>| {
                let visited = atoms.iter().copied().filter(|&atom|
                    choice.successor().structural.atom_is_visited(atom)).collect::<Vec<_>>();
                let [atom] = visited.as_slice() else {
                    panic!("initial labelled root must visit exactly one atom: {visited:?}");
                };
                *atom
            };
            let offered_roots = root_choices.iter().map(|choice| {
                let atom = root_atom(choice);
                root_text_atoms.entry(choice.text().to_owned()).or_default().insert(atom);
                atoms.iter().position(|&candidate| candidate == atom)
                    .expect("initial labelled choice names a graph atom")
            }).collect::<BTreeSet<_>>();
            assert_eq!(offered_roots, expected_roots,
                "productive initial labelled root at graph {edge_mask:03b}, capacity {capacity}");
            equal_text_distinct_atom_entries |= root_text_atoms.values().any(|atoms| atoms.len() >= 2);
            for choice in &root_choices {
                let atom_id = root_atom(&choice);
                assert!(ring_layout_atom_spelling_completes(choice.successor(), atom_id, choice.text()),
                    "each offered initial root choice must have an actual atom spelling: {:?}",
                    choice.text());
                if choice.text() == "C" {
                    assert!(!atom_witness_completions_after_choice(choice, atom_id).is_empty(),
                        "each bare-C initial root choice must retain its own C witness");
                }
            }
            for atom in offered_roots {
                let completions = completed_atom_witnesses(&start, atoms[atom]);
                assert!(!completions.is_empty(),
                    "each offered initial root atom must retain its declared C witness");
                for (_, state) in completions {
                    let mut states = vec![state];
                    for (edge, &(a, b)) in edges.iter().enumerate() {
                        if a != atom && b != atom { continue; }
                        states = ring_layout_action(states, bonds[edge]);
                    }
                    for state in states {
                        visit(state, &mut vec![atom], &atoms, &bonds, &edges, capacity,
                            &mut orders, &mut prefixes, &mut seen_orders);
                    }
                }
            }
            let expected_orders = ring_vertex_orders(atom_count).into_iter()
                .filter(|order| ring_order_peak(order, &edges) <= capacity)
                .collect::<BTreeSet<_>>();
            assert_eq!(seen_orders, expected_orders,
                "every feasible labelled vertex order in graph {edge_mask:03b}, capacity {capacity}");
            distinct_orders += seen_orders.len();
        }
        }
    }
    assert_eq!(graphs, 33);
    assert!(equal_text_distinct_atom_entries,
        "one checkpoint must offer equal text with distinct labelled successor atoms");
    eprintln!("ring layout contract: {graphs} graph/capacity cases, {distinct_orders} distinct orders, {orders} labelled walks, {prefixes} concrete prefixes");
}

#[test]
fn ring_layout_path_cycle_and_k4_sentinels() {
    let path = [(0, 1), (1, 2)];
    assert_eq!(ring_order_peak(&[0, 1, 2], &path), 1);
    assert_eq!(ring_order_peak(&[0, 2, 1], &path), 2);
    let cycle = [(0, 1), (1, 2), (0, 2)];
    assert_eq!(ring_order_peak(&[0, 1, 2], &cycle), 2);
    let k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    assert_eq!(ring_order_peak(&[0, 1, 2, 3], &k4), 4);
    assert!(!ring_suffix_fits(&[0], 1, 4, &k4, 3));
    assert!(ring_suffix_fits(&[0], 1, 4, &k4, 4));
}

fn diamond_schedule_fits(
    bonds: &[BondId; 5], parent_reference: &[TetrahedralLigand; 4],
    child_reference: &[TetrahedralLigand; 4], parent_value: u8, child_value: u8, first: BondId,
) -> bool {
    let parent_order = independent_tetrahedral_order(parent_reference, parent_value);
    let child_order = independent_tetrahedral_order(child_reference, child_value);
    if parent_order[0] != TetrahedralLigand::VirtualHydrogen
        || child_order[..2] != [TetrahedralLigand::Bond(bonds[0]),
            TetrahedralLigand::VirtualHydrogen]
        || parent_order[1] != TetrahedralLigand::Bond(first) {
        return false;
    }
    let mut open = [false; 2];
    for ligand in &parent_order[1..] {
        let ring_indices: &[usize] = match ligand {
            TetrahedralLigand::Bond(bond) if *bond == bonds[0] => {
                // At B, the entry and virtual H precede its two ring incidences.
                for ring in &child_order[2..] {
                    let index = if *ring == TetrahedralLigand::Bond(bonds[3]) { 0 }
                        else if *ring == TetrahedralLigand::Bond(bonds[4]) { 1 }
                        else { return false };
                    open[index] = !open[index];
                    if open.iter().filter(|value| **value).count() > 1 { return false; }
                }
                &[]
            }
            TetrahedralLigand::Bond(bond) if *bond == bonds[1] => &[0],
            TetrahedralLigand::Bond(bond) if *bond == bonds[2] => &[1],
            _ => return false,
        };
        for index in ring_indices {
            open[*index] = !open[*index];
            if open.iter().filter(|value| **value).count() > 1 { return false; }
        }
    }
    open == [false; 2]
}

#[test]
fn finite_two_center_diamond_schedule_requires_close_before_next_open() {
    let bonds: [BondId; 5] = std::array::from_fn(|index| BondId::new(index as u32));
    let parent_reference = [TetrahedralLigand::Bond(bonds[0]),
        TetrahedralLigand::Bond(bonds[1]), TetrahedralLigand::Bond(bonds[2]),
        TetrahedralLigand::VirtualHydrogen];
    let child_reference = [TetrahedralLigand::Bond(bonds[0]),
        TetrahedralLigand::Bond(bonds[3]), TetrahedralLigand::Bond(bonds[4]),
        TetrahedralLigand::VirtualHydrogen];
    let surviving = (0..24).flat_map(|parent_value| (0..24).flat_map(move |child_value|
        [bonds[1], bonds[2]].into_iter().filter_map(move |first|
            diamond_schedule_fits(&bonds, &parent_reference, &child_reference,
                parent_value, child_value, first).then_some((parent_value, child_value, first)))))
        .collect::<Vec<_>>();
    assert_eq!(surviving.len(), 2);
    let projected = surviving.iter().map(|(parent, _, first)|
        (independent_tetrahedral_order_is_even(&parent_reference, *parent), *first))
        .collect::<BTreeSet<_>>();
    assert_eq!(projected, BTreeSet::from([(true, bonds[1]), (false, bonds[2])]));
    for (_, child, first) in surviving {
        let child_order = independent_tetrahedral_order(&child_reference, child);
        assert_eq!(child_order[2], TetrahedralLigand::Bond(
            if first == bonds[1] { bonds[3] } else { bonds[4] }));
    }
}

fn tetrahedral_diamond_surface() -> (PreparedNonStereo, [AtomId; 4], [BondId; 5]) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms: [AtomId; 4] = std::array::from_fn(|_| graph.add_atom().unwrap());
    let bonds = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
        .map(|(a, b)| graph.add_bond(atoms[a], atoms[b]).unwrap());
    let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(graph.build()),
        vec![
            PreparedAtomToken::Tetrahedral {
                reference_order: [TetrahedralLigand::Bond(bonds[0]),
                    TetrahedralLigand::Bond(bonds[1]), TetrahedralLigand::Bond(bonds[2]),
                    TetrahedralLigand::VirtualHydrogen],
                text_by_parity: ["[C@H]".to_owned(), "[C@@H]".to_owned()],
            },
            PreparedAtomToken::Tetrahedral {
                reference_order: [TetrahedralLigand::Bond(bonds[0]),
                    TetrahedralLigand::Bond(bonds[3]), TetrahedralLigand::Bond(bonds[4]),
                    TetrahedralLigand::VirtualHydrogen],
                text_by_parity: ["[C@H]".to_owned(), "[C@@H]".to_owned()],
            },
            PreparedAtomToken::Fixed("C".to_owned()),
            PreparedAtomToken::Fixed("O".to_owned()),
        ],
        vec![NonStereoBondToken::Elided; bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, 0,
    ).unwrap();
    (surface, atoms, bonds)
}

#[test]
fn actual_root_event_activates_schedule_in_one_solver_transition() {
    let _counter_guard = DIRECTIONAL_RESTRICTION_TEST_LOCK.lock().unwrap();
    let (surface, atoms, _) = tetrahedral_diamond_surface();
    let source = NonStereoWriterState::<CountDirectionalRestrictionsSolver>::initial(&surface)
        .unwrap().unwrap_consistent();
    let factor = surface.event_schedule_factor.unwrap();
    assert!(!source.structural.factor_is_active(factor));
    let before = source.observe_raw();

    DIRECTIONAL_RESTRICTION_CALLS.store(0, Ordering::Relaxed);
    let successor = source.attempt_candidate_with_transition(
        StructuralCandidate::Root { atom: atoms[0] }, None, &[], &[],
    ).unwrap().unwrap_consistent();

    assert_eq!(DIRECTIONAL_RESTRICTION_CALLS.load(Ordering::Relaxed), 1);
    assert!(successor.factor_is_active(factor));
    assert_eq!(source.observe_raw(), before);
    let (resource_restrictions, resource_factors) = source.ring_resource_entry_constraint(
        StructuralCandidate::Root { atom: atoms[0] },
        None,
    ).unwrap();
    let old_first = source.structural.attempt_candidate_with_semantic_transition(
        StructuralCandidate::Root { atom: atoms[0] }, None,
        &resource_restrictions, &resource_factors,
    ).unwrap().unwrap_consistent();
    let (schedule_restrictions, schedule_factors) = source.event_schedule_transition(
        StructuralCandidate::Root { atom: atoms[0] },
    ).unwrap();
    let old_successor = old_first.transitioned_semantics(
        &schedule_restrictions, &schedule_factors,
    ).unwrap().unwrap_consistent();
    assert_eq!(successor.observe_raw(), old_successor.observe_raw());
    for index in 0..surface.molecule().constraint_model().variable_count() {
        let variable = VariableId::new(index as u32);
        assert_eq!(successor.semantic_domain(variable), old_successor.semantic_domain(variable),
            "candidate and legacy transition disagree at {variable:?}");
    }
}

#[test]
fn event_schedule_factor_matches_independent_diamond_event_reference() {
    let (surface, atoms, bonds) = tetrahedral_diamond_surface();
    let factor_id = surface.event_schedule_factor.unwrap();
    let FactorDefinition::EventSchedule(schedule) = surface.molecule().constraint_model()
        .factor(factor_id).unwrap() else { panic!("prepared event schedule factor expected") };
    let parent = surface.tetrahedral_center(atoms[0]).unwrap();
    let child = surface.tetrahedral_center(atoms[1]).unwrap();
    let model = surface.molecule().constraint_model();
    let FactorDefinition::TetrahedralLayout(parent_layout) = model.factor(parent.layout_factor(None)).unwrap()
        else { panic!("parent layout expected") };
    let FactorDefinition::TetrahedralLayout(child_layout) = model.factor(child.layout_factor(Some(bonds[0]))).unwrap()
        else { panic!("child layout expected") };
    let child_pattern = (1_u8 << child.pattern_bit(bonds[3])) | (1_u8 << child.pattern_bit(bonds[4]));
    let mut surviving = Vec::new();
    for parent_value in 0..24 {
        for child_value in 0..24 {
            let reference = [bonds[1], bonds[2]].into_iter().any(|first|
                diamond_schedule_fits(&bonds, &parent.reference_order, &child.reference_order,
                    parent_value, child_value, first));
            let relation = parent_layout.allowed_orders(0).contains(parent_value)
                && child_layout.allowed_orders(child_pattern).contains(child_value)
                && [bonds[1], bonds[2]].into_iter().any(|first| {
                    let last = if first == bonds[1] { bonds[2] } else { bonds[1] };
                    let first_atom = if first == bonds[1] { atoms[2] } else { atoms[3] };
                    let last_atom = if last == bonds[1] { atoms[2] } else { atoms[3] };
                    let ranked = [atoms[0], first_atom, atoms[1], last_atom];
                    let mut assigned = BTreeMap::<VariableId, u8>::new();
                    for (rank, atom) in ranked.into_iter().enumerate() {
                        let visit = &schedule.visits[atom.index()];
                        for (bit, variable) in visit.rank_bits.iter().enumerate() {
                            assigned.insert(*variable, ((rank >> bit) & 1) as u8);
                        }
                        assigned.insert(visit.kind, [EntryKind::Root, EntryKind::BondBranch,
                            EntryKind::BondBranch, EntryKind::BondInline][rank] as u8);
                        for (bit, variable) in visit.end_bits.iter().enumerate() {
                            assigned.insert(*variable, u8::from((rank == 1 || rank == 2) && bit <= rank));
                        }
                    }
                    assigned.insert(parent.order_variable, parent_value);
                    assigned.insert(child.order_variable, child_value);
                    assigned.insert(schedule.centers.iter().find(|center| center.atom == atoms[0])
                        .unwrap().role_pattern_variable, 0);
                    assigned.insert(schedule.centers.iter().find(|center| center.atom == atoms[1])
                        .unwrap().role_pattern_variable, child_pattern);
                    for (bit, variable) in schedule.capacity_bits.iter().enumerate() {
                        assigned.insert(*variable, u8::from(bit == 0));
                    }
                    for (index, variable) in schedule.forest_roots.iter().enumerate() {
                        assigned.insert(*variable, u8::from(index == atoms[0].index()));
                    }
                    for center in schedule.ordinary_centers.iter() {
                        let entry = if center.atom == atoms[2] { bonds[1] } else { bonds[2] };
                        for pair in center.pairs.iter() {
                            assigned.insert(pair.variable, u8::from(pair.second == entry));
                        }
                    }
                    for edge in schedule.edges.iter() {
                        let role = if edge.bond.index() < 3 {
                            edge.role_partition.traversal_values()
                        } else { edge.role_partition.ring_values() };
                        assigned.insert(edge.decision_variable, role.iter().next().unwrap());
                    }
                    schedule.permits(|variable| assigned[&variable])
                });
            assert_eq!(relation, reference,
                "event schedule disagrees with independent event cuts at orders {parent_value}, {child_value}");
            if relation { surviving.push((parent_value, child_value)); }
        }
    }
    assert_eq!(surviving.len(), 2);
}

#[test]
fn event_preparation_uses_the_actual_writer_label_namespace() {
    let prepare = |edge_count: usize, maximum: usize| {
        let mut graph = PreparedGraphBuilder::new();
        let atoms = (0..4).map(|_| graph.add_atom().unwrap()).collect::<Vec<_>>();
        for index in 0..edge_count {
            graph.add_bond(atoms[index], atoms[index + 1]).unwrap();
        }
        let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
            PreparedMolecule::new(graph.build()),
            (0..4).map(|_| PreparedAtomToken::Fixed("C".to_owned())).collect(),
            vec![NonStereoBondToken::Elided; edge_count], Vec::new(),
            WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, maximum,
        ).unwrap();
        assert_eq!(initial(&surface).labels.maximum_spelling_label(), maximum);
        surface.event_schedule_factor.is_some()
    };
    assert!(!prepare(1, 0), "one forest edge fits one visible label");
    assert!(prepare(2, 0), "two forest edges can exceed one label even with an isolated atom");
    assert!(!prepare(2, MAX_RING_LABEL_INDEX), "ordinary capacity does not need the event factor");
    assert!(ring_count_exceeds_schedule_capacity(100, WriterRegime::RdkitRandomNoncanonical, 99));
}

/// Count a complete native frontier with a fixed leaf degree and remote cycles.
#[test]
#[ignore = "opt-in bounded work diagnostic; run one selected family size"]
fn work_diagnostic_remote_triangle_chain() {
    use std::io::Write;
    let sizes = std::env::var("GRIMACE_WORK_SIZES").unwrap_or_else(|_| "1,2".to_owned());
    let chain = std::env::var("GRIMACE_WORK_MODE").is_ok_and(|mode| mode == "chain_prep");
    for gadgets in sizes.split(',').map(|part| part.parse::<usize>().unwrap()) {
        let mut graph = PreparedGraphBuilder::new();
        let root = graph.add_atom().unwrap();
        let mut previous = root;
        let mut edges = 0;
        let atoms = if chain {
            for _ in 1..gadgets {
                let atom = graph.add_atom().unwrap();
                graph.add_bond(previous, atom).unwrap();
                previous = atom;
                edges += 1;
            }
            gadgets
        } else {
            for _ in 0..gadgets {
                let a = graph.add_atom().unwrap();
                let b = graph.add_atom().unwrap();
                let c = graph.add_atom().unwrap();
                for (left, right) in [(previous, a), (a, b), (b, c), (c, a)] {
                    graph.add_bond(left, right).unwrap();
                    edges += 1;
                }
                previous = c;
            }
            1 + 3 * gadgets
        };
        let maximum = std::env::var("GRIMACE_WORK_MAX_LABEL")
            .map(|value| value.parse::<usize>().unwrap()).unwrap_or(0);
        println!("STAGE family={} size={gadgets} atoms={atoms} edges={edges} max_label={maximum} phase=prepare_start",
            if chain { "acyclic_chain" } else { "remote_triangles" });
        std::io::stdout().flush().unwrap();
        crate::work_diagnostic::begin();
        let prep_started = std::time::Instant::now();
        let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
            PreparedMolecule::new(graph.build()),
            (0..atoms).map(|_| PreparedAtomToken::Fixed("C".to_owned())).collect(),
            vec![NonStereoBondToken::Elided; edges], Vec::new(),
            WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, maximum,
        ).unwrap();
        let prep_elapsed = prep_started.elapsed();
        let prep_work = crate::work_diagnostic::finish();
        let model = surface.molecule().constraint_model();
        let factor_refs = (0..model.factor_count()).map(|index| {
            model.factor(FactorId::new(index as u32)).unwrap().variables().len()
        }).sum::<usize>();
        let schedule = surface.event_schedule_factor.and_then(|id| model.factor(id));
        let (rank_bits, kind_vars, end_bits, schedule_refs) = match schedule {
            Some(crate::model::FactorDefinition::EventSchedule(schedule)) => (
                schedule.visits.iter().map(|visit| visit.rank_bits.len()).sum::<usize>(),
                schedule.visits.len(),
                schedule.visits.iter().map(|visit| visit.end_bits.len()).sum::<usize>(),
                schedule.variables().len(),
            ),
            _ => (0, 0, 0, 0),
        };
        let domain_size = std::mem::size_of::<crate::domain::Domain>();
        let model_payload_lower_bytes = model.variable_count() * domain_size
            + factor_refs * std::mem::size_of::<crate::ids::VariableId>();
        println!("STAGE family={} size={gadgets} phase=prepare_done elapsed_ms={} counters={prep_work:?} vars={} factors={} refs={factor_refs} rank_bits={rank_bits} kind_vars={kind_vars} end_bits={end_bits} schedule_refs={schedule_refs} model_payload_lower_bytes={model_payload_lower_bytes}",
            if chain { "acyclic_chain" } else { "remote_triangles" }, prep_elapsed.as_millis(),
            model.variable_count(), model.factor_count());
        std::io::stdout().flush().unwrap();
        if chain { continue; }
        println!("STAGE family=remote_triangles size={gadgets} phase=root_start");
        std::io::stdout().flush().unwrap();
        crate::work_diagnostic::begin();
        let root_started = std::time::Instant::now();
        let state = initial(&surface);
        let (_, rooted) = only_root_atom(&state, root);
        let root_elapsed = root_started.elapsed();
        let root_work = crate::work_diagnostic::finish();
        println!("STAGE family=remote_triangles size={gadgets} phase=root_done elapsed_ms={} counters={root_work:?}", root_elapsed.as_millis());
        std::io::stdout().flush().unwrap();
        rooted.structural.constraints_for_test().reset_solver_copy_counts();
        rooted.structural.reset_traversal_copy_counts();
        println!("STAGE family=remote_triangles size={gadgets} phase=choices_start");
        std::io::stdout().flush().unwrap();
        crate::work_diagnostic::begin();
        let choices_started = std::time::Instant::now();
        let choices = rooted.choices().unwrap();
        let choices_elapsed = choices_started.elapsed();
        let work = crate::work_diagnostic::finish();
        let ((domain_nodes, domain_pages), (active_nodes, active_pages),
            (adj_nodes, adj_pages)) = rooted.structural.constraints_for_test().solver_copy_counts();
        let (traversal_nodes, traversal_pages) = rooted.structural.traversal_copy_counts();
        let texts = choices.iter().map(|choice| choice.text().to_owned()).collect::<BTreeSet<_>>();
        let observed = choices.iter().map(|choice|
            (choice.text(), choice.successor().observe_raw())).collect::<Vec<_>>();
        let duplicate_observed = observed.iter().enumerate().filter(|(index, row)|
            observed[..*index].contains(row)).count();
        println!("WORK gadgets={gadgets} atoms={atoms} edges={edges} max_label={maximum} choices_elapsed_ms={} vars={} factors={} refs={factor_refs} rank_bits={rank_bits} kind_vars={kind_vars} end_bits={end_bits} schedule_refs={schedule_refs} model_payload_lower_bytes={model_payload_lower_bytes} attempts={} rejected={} outputs={} texts={} duplicate_observed={} paged_reads={} factor_revisions={} search_nodes={} grouped_branches={} scalar_branches={} sibling_offered={} sibling_proposals={} sibling_propagated={} children_explored={} copied_domain_nodes={domain_nodes} copied_domain_pages={domain_pages} copied_active_nodes={active_nodes} copied_active_pages={active_pages} copied_adj_nodes={adj_nodes} copied_adj_pages={adj_pages} copied_traversal_nodes={traversal_nodes} copied_traversal_pages={traversal_pages}",
            choices_elapsed.as_millis(),
            model.variable_count(), model.factor_count(), work.candidate_attempts,
            work.candidate_rejections, choices.len(), texts.len(), duplicate_observed, work.paged_reads,
            work.factor_revisions, work.search_nodes, work.grouped_branches,
            work.scalar_branches, work.sibling_offered, work.sibling_proposals, work.sibling_propagated,
            work.children_explored);
        std::io::stdout().flush().unwrap();
    }
}

#[test]
fn tetrahedral_diamond_first_child_respects_future_ring_schedule() {
    let (surface, atoms, bonds) = tetrahedral_diamond_surface();
    let mut state = initial(&surface);
    let restrictions = bonds.iter().copied().enumerate().map(|(index, bond)| {
        let partition = surface.molecule().bond_role_partition(bond).unwrap();
        (surface.molecule().bond_decision_variable(bond).unwrap(),
            if index < 3 { partition.traversal_values() } else { partition.ring_values() })
    }).collect::<Vec<_>>();
    state.structural = state.structural.transitioned_semantics(&restrictions, &[])
        .unwrap().unwrap_consistent();
    let roots = completed_atom_witnesses(&state, atoms[0]);
    assert_eq!(roots.len(), 2);
    let parent_reference = surface.tetrahedral_center(atoms[0]).unwrap().reference_order;
    let child_reference = surface.tetrahedral_center(atoms[1]).unwrap().reference_order;
    for (text, root) in roots {
        let parity_even = text == "[C@H]";
        for first in [bonds[1], bonds[2]] {
            let viable = (0..24).any(|parent_value|
                independent_tetrahedral_order_is_even(&parent_reference, parent_value) == parity_even
                    && (0..24).any(|child_value| diamond_schedule_fits(&bonds, &parent_reference, &child_reference, parent_value, child_value, first)));
            let offered = root.choices().unwrap().iter().any(|choice|
                matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident ==
                    incident(&surface, atoms[0], first)));
            assert_eq!(offered, viable,
                "root marker {text}, first bond {first:?}: branch must preserve one complete two-center schedule");
        }
    }
}

#[test]
fn unseeded_diamond_branch_seal_checks_future_ring_schedule() {
    let (surface, atoms, bonds) = tetrahedral_diamond_surface();
    let state = initial(&surface);
    let roots = completed_atom_witnesses(&state, atoms[0]);
    assert_eq!(roots.len(), 2);
    let parent_reference = surface.tetrahedral_center(atoms[0]).unwrap().reference_order;
    let child_reference = surface.tetrahedral_center(atoms[1]).unwrap().reference_order;
    for (text, root) in roots {
        let factor = surface.event_schedule_factor.unwrap();
        assert!(root.structural.factor_is_active(factor));
        let parity_even = text == "[C@H]";
        for first in [bonds[1], bonds[2]] {
            let viable = (0..24).any(|parent_value|
                independent_tetrahedral_order_is_even(&parent_reference, parent_value) == parity_even
                    && (0..24).any(|child_value| diamond_schedule_fits(&bonds, &parent_reference, &child_reference, parent_value, child_value, first)));
            let offered = root.choices().unwrap().iter().any(|choice|
                matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident ==
                    incident(&surface, atoms[0], first)));
            assert_eq!(offered, viable, "unseeded root marker {text}, first bond {first:?}");
            if offered {
                let branch = root.choices().unwrap().into_iter().find(|choice|
                    matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident ==
                        incident(&surface, atoms[0], first))).unwrap();
                assert!(branch.successor().structural.factor_is_active(factor));
                assert_eq!(branch.successor().structural.fixed_bond_role(bonds[3]), Some(BondRole::Ring));
                assert_eq!(branch.successor().structural.fixed_bond_role(bonds[4]), Some(BondRole::Ring));
            }
        }
    }
}

#[test]
fn ordinary_root_subdivided_diamond_first_branch_respects_future_ring_schedule() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms: [AtomId; 5] = std::array::from_fn(|_| graph.add_atom().unwrap());
    // A-X-B is the long traversal arm; C and D are distinct short arms.
    let bonds = [(0, 1), (1, 2), (0, 3), (0, 4), (2, 3), (2, 4)]
        .map(|(a, b)| graph.add_bond(atoms[a], atoms[b]).unwrap());
    let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(graph.build()),
        vec![
            PreparedAtomToken::Fixed("C".to_owned()),
            PreparedAtomToken::Fixed("N".to_owned()),
            PreparedAtomToken::Tetrahedral {
                reference_order: [TetrahedralLigand::Bond(bonds[1]),
                    TetrahedralLigand::Bond(bonds[4]), TetrahedralLigand::Bond(bonds[5]),
                    TetrahedralLigand::VirtualHydrogen],
                text_by_parity: ["[C@H]".to_owned(), "[C@@H]".to_owned()],
            },
            PreparedAtomToken::Fixed("C".to_owned()),
            PreparedAtomToken::Fixed("O".to_owned()),
        ],
        vec![NonStereoBondToken::Elided; bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, 0,
    ).unwrap();
    let mut state = initial(&surface);
    let restrictions = bonds.iter().copied().enumerate().map(|(index, bond)| {
        let partition = surface.molecule().bond_role_partition(bond).unwrap();
        (surface.molecule().bond_decision_variable(bond).unwrap(),
            if index < 4 { partition.traversal_values() } else { partition.ring_values() })
    }).collect::<Vec<_>>();
    state.structural = state.structural.transitioned_semantics(&restrictions, &[])
        .unwrap().unwrap_consistent();
    let (_, root) = only_root_atom(&state, atoms[0]);
    let mut viable_first = BTreeSet::new();
    for children in permutations(&[bonds[0], bonds[2], bonds[3]]) {
        for b_events in permutations(&[Some(bonds[4]), Some(bonds[5]), None]) {
            let mut live = BTreeSet::new();
            let mut fits = true;
            for child in &children {
                let events: Vec<BondId> = if *child == bonds[0] {
                    b_events.iter().flatten().copied().collect()
                } else if *child == bonds[2] { vec![bonds[4]] }
                else { vec![bonds[5]] };
                for ring in events {
                    if !live.remove(&ring) { live.insert(ring); }
                    fits &= live.len() <= 1;
                }
            }
            if fits && live.is_empty() { viable_first.insert(children[0]); }
        }
    }
    assert_eq!(viable_first, BTreeSet::from([bonds[2], bonds[3]]));
    let choices = root.choices().unwrap();
    for first in [bonds[0], bonds[2], bonds[3]] {
        let offered = choices.iter().any(|choice|
            matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident ==
                incident(&surface, atoms[0], first)));
        assert_eq!(offered, viable_first.contains(&first),
            "first ordinary child {first:?} must preserve a complete one-label schedule");
    }
}

#[test]
fn disconnected_spectator_cannot_rescue_subdivided_diamond_first_branch() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms: [AtomId; 6] = std::array::from_fn(|_| graph.add_atom().unwrap());
    // A-X-B is the long arm; the isolated O has no ring event or label release.
    let bonds = [(0, 1), (1, 2), (0, 3), (0, 4), (2, 3), (2, 4)]
        .map(|(a, b)| graph.add_bond(atoms[a], atoms[b]).unwrap());
    let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(graph.build()),
        vec![
            PreparedAtomToken::Fixed("C".to_owned()),
            PreparedAtomToken::Fixed("N".to_owned()),
            PreparedAtomToken::Tetrahedral {
                reference_order: [TetrahedralLigand::Bond(bonds[1]),
                    TetrahedralLigand::Bond(bonds[4]), TetrahedralLigand::Bond(bonds[5]),
                    TetrahedralLigand::VirtualHydrogen],
                text_by_parity: ["[C@H]".to_owned(), "[C@@H]".to_owned()],
            },
            PreparedAtomToken::Fixed("C".to_owned()),
            PreparedAtomToken::Fixed("O".to_owned()),
            PreparedAtomToken::Fixed("O".to_owned()),
        ],
        vec![NonStereoBondToken::Elided; bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, 0,
    ).unwrap();
    assert!(surface.event_schedule_factor.is_some());
    let mut state = initial(&surface);
    let restrictions = bonds.iter().copied().enumerate().map(|(index, bond)| {
        let partition = surface.molecule().bond_role_partition(bond).unwrap();
        (surface.molecule().bond_decision_variable(bond).unwrap(),
            if index < 4 { partition.traversal_values() } else { partition.ring_values() })
    }).collect::<Vec<_>>();
    state.structural = state.structural.transitioned_semantics(&restrictions, &[])
        .unwrap().unwrap_consistent();
    let (_, root) = only_root_atom(&state, atoms[0]);

    let mut viable_first = BTreeSet::new();
    for children in permutations(&[bonds[0], bonds[2], bonds[3]]) {
        for b_events in permutations(&[Some(bonds[4]), Some(bonds[5]), None]) {
            for spectator_position in 0..=children.len() {
                let mut live = BTreeSet::new();
                let mut fits = true;
                let mut child_events = Vec::new();
                for child in &children {
                    let events: Vec<BondId> = if *child == bonds[0] {
                        b_events.iter().flatten().copied().collect()
                    } else if *child == bonds[2] { vec![bonds[4]] }
                    else { vec![bonds[5]] };
                    child_events.push(events);
                }
                child_events.insert(spectator_position, Vec::new());
                for events in child_events {
                    for ring in events {
                        if !live.remove(&ring) { live.insert(ring); }
                        fits &= live.len() <= 1;
                    }
                }
                if fits && live.is_empty() { viable_first.insert(children[0]); }
            }
        }
    }
    assert_eq!(viable_first, BTreeSet::from([bonds[2], bonds[3]]));
    let choices = root.choices().unwrap();
    let offered = |first| choices.iter().any(|choice|
        matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident ==
            incident(&surface, atoms[0], first)));
    assert!(offered(bonds[2]), "the C-first completion must remain available");
    assert!(offered(bonds[3]), "the D-first completion must remain available");
    assert!(!offered(bonds[0]), "isolated spectator cannot rescue A-X first");
}

#[test]
fn unseeded_ordinary_root_keeps_order_facts_across_first_branch() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms: [AtomId; 4] = std::array::from_fn(|_| graph.add_atom().unwrap());
    let bonds = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
        .map(|(a, b)| graph.add_bond(atoms[a], atoms[b]).unwrap());
    let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(graph.build()),
        vec![PreparedAtomToken::Fixed("C".to_owned()),
            PreparedAtomToken::Tetrahedral {
                reference_order: [TetrahedralLigand::Bond(bonds[0]),
                    TetrahedralLigand::Bond(bonds[3]), TetrahedralLigand::Bond(bonds[4]),
                    TetrahedralLigand::VirtualHydrogen],
                text_by_parity: ["[C@H]".to_owned(), "[C@@H]".to_owned()],
            }, PreparedAtomToken::Fixed("C".to_owned()), PreparedAtomToken::Fixed("O".to_owned())],
        vec![NonStereoBondToken::Elided; bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, 0,
    ).unwrap();
    let state = initial(&surface);
    let (_, root) = only_root_atom(&state, atoms[0]);
    let factor = surface.event_schedule_factor.unwrap();
    assert!(root.structural.factor_is_active(factor));
    let first = root.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident ==
            incident(&surface, atoms[0], bonds[1]))).expect("A-C is a feasible first branch");
    let successor = first.successor();
    assert!(successor.structural.factor_is_active(factor));
    assert_eq!(successor.structural.fixed_bond_role(bonds[3]), Some(BondRole::Ring));
    assert_eq!(successor.structural.fixed_bond_role(bonds[4]), Some(BondRole::Ring));
    let pair = surface.ordinary_order_pairs[atoms[0].index()].iter().find(|pair|
        [pair.first, pair.second] == [bonds[0], bonds[2]]
            || [pair.first, pair.second] == [bonds[2], bonds[0]])
        .expect("root B/D precedence variable");
    let b_before_d = u8::from(pair.first == bonds[2]);
    assert!(successor.structural.semantic_domain(pair.variable).contains(b_before_d));
    let (_, c) = only_choice(successor, "C");
    let c_entry = surface.ordinary_order_pairs[atoms[2].index()].iter().find(|pair|
        [pair.first, pair.second] == [bonds[1], bonds[3]]
            || [pair.first, pair.second] == [bonds[3], bonds[1]])
        .expect("C entry and ring incidence share one order fact");
    let entry_first = u8::from(c_entry.second == bonds[1]);
    assert_eq!(c.structural.semantic_domain(c_entry.variable),
        Domain::singleton(entry_first).unwrap());
    let opened = c.choices().unwrap().into_iter().find(|choice|
        choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[3]))
        .expect("C opens BC before leaving its branch").into_successor();
    let (_, restored) = only_choice(&opened, ")");
    assert!(restored.structural.factor_is_active(factor));
    let c_visit = &surface.event_schedule_visits[atoms[2].index()];
    assert!(c_visit.end_bits.iter().enumerate().all(|(rank, bit)|
        restored.structural.semantic_domain(*bit)
            == Domain::singleton(u8::from(rank <= 1)).unwrap()));
    assert!(restored.structural.semantic_domain(pair.variable).contains(b_before_d));
    assert_eq!(restored.structural.semantic_domain(c_entry.variable),
        Domain::singleton(entry_first).unwrap());
    let c_before_b = surface.ordinary_order_pairs[atoms[0].index()].iter().find(|pair|
        [pair.first, pair.second] == [bonds[0], bonds[1]]
            || [pair.first, pair.second] == [bonds[1], bonds[0]])
        .unwrap();
    assert_eq!(restored.structural.semantic_domain(c_before_b.variable),
        Domain::singleton(u8::from(c_before_b.first == bonds[0])).unwrap());
}

#[test]
fn releasing_a_real_held_label_restores_diamond_branch() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..6).map(|_| graph.add_atom().unwrap()).collect::<Vec<_>>();
    let bonds = [(0, 1), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5)]
        .into_iter().map(|(a, b)| graph.add_bond(atoms[a], atoms[b]).unwrap())
        .collect::<Vec<_>>();
    let surface = PreparedNonStereo::with_ring_label_maximum_for_test(
        PreparedMolecule::new(graph.build()),
        (0..6).map(|_| PreparedAtomToken::Fixed("C".to_owned())).collect(),
        vec![NonStereoBondToken::Elided; bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true, 1,
    ).unwrap();
    assert!(surface.event_schedule_factor.is_some(),
        "a disconnected dot-branch schedule participates in the event relation");
    let state = initial(&surface);
    let (_, ethane) = only_root_atom(&state, atoms[0]);
    let held = ethane.choices().unwrap().into_iter().find(|choice|
        choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[0])
    ).expect("ethane ring opens").into_successor();
    let (_, pending_diamond) = only_component_separator(&held, atoms[2]);
    let (_, diamond) = only_choice(&pending_diamond, "C");
    let branch_incident = incident(&surface, atoms[2], bonds[1]);
    assert!(!diamond.choices().unwrap().iter().any(|choice|
        matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == branch_incident)
    ));
    let rescue = diamond.choices().unwrap().into_iter().find(|choice|
        choice.successor().pending == Some(PendingEmission::SegmentRootAtom(atoms[1]))
    ).expect("dot path reaches the held ethane endpoint").into_successor();
    let (_, rescue_atom) = only_choice(&rescue, "C");
    let released = rescue_atom.choices().unwrap().into_iter().find(|choice|
        choice.successor().labels.is_clean()
    ).expect("closing held ethane ring frees its label").into_successor();
    let (_, restored) = only_choice(&released, ")");
    assert!(restored.choices().unwrap().iter().any(|choice|
        matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == branch_incident)
    ));
    let branch = restored.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == branch_incident)
    ).unwrap().into_successor();
    let (_, child) = only_choice(&branch, "C");
    let first_open = child.choices().unwrap().into_iter().find(|choice|
        choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[4])
    ).expect("B opens BC").into_successor();
    let second_open = first_open.choices().unwrap().into_iter().find(|choice|
        choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[5])
    ).expect("B opens BD").into_successor();
    let (_, parent) = only_choice(&second_open, ")");
    let c_incident = incident(&surface, atoms[2], bonds[2]);
    let c_branch = parent.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == c_incident)
    ).expect("A enters C").into_successor();
    let (_, c) = only_choice(&c_branch, "C");
    let c_closed = c.choices().unwrap().into_iter().find(|choice|
        !choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[4])
    ).expect("C closes BC").into_successor();
    let (_, parent) = only_choice(&c_closed, ")");
    let d_incident = incident(&surface, atoms[2], bonds[3]);
    let d_pending = parent.choices().unwrap().into_iter().find(|choice|
        matches!(&choice.successor().pending, Some(PendingEmission::InlineAtom(entry)) if entry.incident == d_incident)
    ).expect("A enters D").into_successor();
    let (_, d) = only_choice(&d_pending, "C");
    let d_closed = d.choices().unwrap().into_iter().find(|choice|
        choice.successor().labels.is_clean()
    ).expect("D closes BD").into_successor();
    assert!(d_closed.is_accepted());
}

#[test]
#[ignore = "100-label positive-capacity diagnostic; run with bounded single-test command"]
fn public_ring_label_capacity_rejects_two_forced_openings_with_one_free_slot() {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = (0..202).map(|_| graph.add_atom().unwrap()).collect::<Vec<_>>();
    let mut bonds = (0..99).map(|i| graph.add_bond(atoms[2*i], atoms[2*i+1]).unwrap())
        .collect::<Vec<_>>();
    let diamond = [(198, 199), (198, 200), (198, 201), (199, 200), (199, 201)]
        .into_iter().map(|(a, b)| graph.add_bond(atoms[a], atoms[b]).unwrap())
        .collect::<Vec<_>>();
    bonds.extend(diamond);
    let surface = PreparedNonStereo::with_atom_tokens_and_directional_for_writer(
        PreparedMolecule::new(graph.build()),
        (0..atoms.len()).map(|_| PreparedAtomToken::Fixed("C".to_owned())).collect(),
        vec![NonStereoBondToken::Elided; bonds.len()], Vec::new(),
        WriterRegime::Generic { component_order: ComponentOrder::AnyOrder }, true,
    ).unwrap();
    let mut state = initial(&surface);
    for i in 0..99 {
        let rooted = if i == 0 { only_root_atom(&state, atoms[0]).1 } else {
            let (_, pending) = only_component_separator(&state, atoms[2*i]);
            only_choice(&pending, "C").1
        };
        state = rooted.choices().unwrap().into_iter().find(|choice|
            choice.successor().labels.bonds_by_slot.values().any(|bond| *bond == bonds[i])
        ).expect("each ethane ring holds one real label").into_successor();
    }
    assert_eq!(state.labels.bonds_by_slot.len(), 99);
    let (_, pending_diamond) = only_component_separator(&state, atoms[198]);
    let (_, root) = only_choice(&pending_diamond, "C");
    let branch = incident(&surface, atoms[198], bonds[99]);
    assert!(!root.choices().unwrap().iter().any(|choice|
        matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == branch)
    ));
    let rescue = root.choices().unwrap().into_iter().find(|choice|
        choice.successor().pending == Some(PendingEmission::SegmentRootAtom(atoms[1]))
    ).expect("dot path reaches a held ethane endpoint").into_successor();
    let (_, rescue_atom) = only_choice(&rescue, "C");
    let released = rescue_atom.choices().unwrap().into_iter().find(|choice|
        choice.successor().labels.bonds_by_slot.len() == 98
    ).expect("real closure frees one label").into_successor();
    let (_, restored) = only_choice(&released, ")");
    assert!(restored.choices().unwrap().iter().any(|choice|
        matches!(&choice.successor().pending, Some(PendingEmission::BranchTraversalEmission(entry)) if entry.incident == branch)
    ));
}
