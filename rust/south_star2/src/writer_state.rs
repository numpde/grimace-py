//! Concrete composition of residual writer frames and constraint state.
//!
//! The graphic-matroid factor guards global Traversal/Ring feasibility. Writer
//! policy is stricter: each residual attachment receives exactly one traversal
//! entry, while its other active incidences become ring endpoints.

use crate::domain::Domain;
use crate::ids::{AtomId, BondId, FactorId, VariableId};
use crate::model::BondRole;
use crate::prepared::{AdjacentBond, PreparedMolecule};
use crate::solver::{Consistency, ConstraintSolver};
#[cfg(test)]
use crate::traversal::ObservedTraversalState;
use crate::traversal::{
    IncidentBondState, LocalBondOrder, LocalLayoutContext, PathCompletion, TraversalState,
};

use std::collections::BTreeMap;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(test)]
use std::sync::Arc;

#[derive(Debug, Default)]
pub(crate) struct StructuralFrontier {
    component_roots: Vec<AtomId>,
    branch_children: Vec<AdjacentBond>,
    inline_children: Vec<AdjacentBond>,
    ring_openings: Vec<AdjacentBond>,
    ring_closures: Vec<AdjacentBond>,
    path_completion: Option<PathCompletion>,
    contradiction: bool,
}

impl StructuralFrontier {
    pub(crate) fn component_roots(&self) -> &[AtomId] {
        &self.component_roots
    }

    pub(crate) fn branch_children(&self) -> &[AdjacentBond] {
        &self.branch_children
    }

    pub(crate) fn inline_children(&self) -> &[AdjacentBond] {
        &self.inline_children
    }

    pub(crate) fn ring_openings(&self) -> &[AdjacentBond] {
        &self.ring_openings
    }

    pub(crate) fn ring_closures(&self) -> &[AdjacentBond] {
        &self.ring_closures
    }

    pub(crate) const fn path_completion(&self) -> Option<PathCompletion> {
        self.path_completion
    }

    pub(crate) const fn is_contradiction(&self) -> bool {
        self.contradiction
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum StructuralCandidate {
    Root {
        atom: AtomId,
    },
    RingOpen {
        incident: AdjacentBond,
    },
    RingClose {
        incident: AdjacentBond,
        first_endpoint: AtomId,
    },
    BranchChild {
        incident: AdjacentBond,
    },
    InlineChild {
        incident: AdjacentBond,
    },
    CloseBranch,
    FinishComponent,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct StructuralCandidateBatch {
    candidates: Vec<StructuralCandidate>,
    contradiction: bool,
}

impl StructuralCandidateBatch {
    pub(crate) fn candidates(&self) -> &[StructuralCandidate] {
        &self.candidates
    }

    pub(crate) const fn is_contradiction(&self) -> bool {
        self.contradiction
    }
}

#[derive(Clone, Debug)]
pub(crate) struct WriterState<S> {
    prepared: PreparedMolecule,
    traversal: TraversalState,
    constraints: S,
    #[cfg(test)]
    candidate_batch_derivations: Arc<AtomicUsize>,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ObservedWriterState {
    pub(crate) traversal: ObservedTraversalState,
    pub(crate) bond_plan_domains: Vec<Domain>,
    pub(crate) active_factors: Vec<FactorId>,
}

impl<S: ConstraintSolver> WriterState<S> {
    pub(crate) fn initial(prepared: &PreparedMolecule) -> Result<Consistency<Self>, S::Failure> {
        Ok(
            S::initial(prepared.constraint_model_arc())?.map(|constraints| {
                assert_initial_solver_shape(prepared, &constraints);
                Self {
                    prepared: prepared.clone(),
                    traversal: TraversalState::new(prepared.graph()),
                    constraints,
                    #[cfg(test)]
                    candidate_batch_derivations: Arc::new(AtomicUsize::new(0)),
                }
            }),
        )
    }

    pub(crate) fn active_atom(&self) -> Option<AtomId> {
        self.traversal.active_atom()
    }

    pub(crate) const fn graph_is_complete(&self) -> bool {
        self.traversal.graph_is_complete()
    }

    pub(crate) const fn has_visited_atoms(&self) -> bool {
        self.traversal.has_visited_atoms()
    }

    pub(crate) fn ring_is_open(&self, bond: BondId) -> bool {
        self.traversal.ring_is_open(bond)
    }

    #[cfg(test)]
    pub(crate) fn atom_is_visited(&self, atom: AtomId) -> bool {
        self.traversal.atom_is_visited(atom)
    }

    fn bond_role_domain(&self, bond: BondId) -> Domain {
        let decision_domain = self.bond_decision_domain(bond);
        let partition = self
            .prepared
            .bond_role_partition(bond)
            .expect("prepared bond must have a role partition");
        let mut roles = Domain::empty();
        if !decision_domain
            .intersect(partition.traversal_values())
            .is_empty()
        {
            roles = roles.union(BondRole::Traversal.singleton_domain());
        }
        if !decision_domain
            .intersect(partition.ring_values())
            .is_empty()
        {
            roles = roles.union(BondRole::Ring.singleton_domain());
        }
        roles
    }

    pub(crate) fn bond_decision_domain(&self, bond: BondId) -> Domain {
        self.constraints
            .domain(decision_variable(&self.prepared, bond))
            .expect("prepared bond decision must belong to the writer constraint model")
    }

    pub(crate) fn semantic_domain(&self, variable: VariableId) -> Domain {
        self.constraints
            .domain(variable)
            .expect("prepared semantic variable must belong to the writer constraint model")
    }

    pub(crate) fn factor_is_active(&self, factor: FactorId) -> bool {
        self.constraints
            .factor_is_active(factor)
            .expect("prepared semantic factor must belong to the solver state")
    }

    pub(crate) fn structural_frontier(&self) -> StructuralFrontier {
        #[cfg(test)]
        self.candidate_batch_derivations
            .fetch_add(1, Ordering::Relaxed);

        let graph = self.prepared.graph();
        let Some(active) = self.traversal.active_atom() else {
            return StructuralFrontier {
                component_roots: self.traversal.unvisited_atoms(graph).collect(),
                ..StructuralFrontier::default()
            };
        };

        let mut frontier = StructuralFrontier::default();
        let mut ring_phase = false;
        let mut children = Vec::new();

        for incident in graph
            .neighbors(active)
            .expect("active atom must belong to the prepared graph")
            .iter()
            .copied()
        {
            match self.traversal.classify_active_incident(graph, incident) {
                IncidentBondState::RingOpenAtOtherAtom => {
                    ring_phase = true;
                    frontier.ring_closures.push(incident);
                }
                IncidentBondState::UnrepresentedToVisitedAtom => {
                    panic!(
                        "a visited endpoint must already be represented or own an open ring endpoint"
                    );
                }
                IncidentBondState::Represented
                | IncidentBondState::RingOpenAtCurrentAtom
                | IncidentBondState::UnrepresentedToUnvisitedAtom => {}
            }
        }

        for attachment in self.traversal.active_attachments() {
            let incidences = attachment.incidences();
            assert!(
                !incidences.is_empty(),
                "residual attachments must not be empty"
            );

            let role_domains = incidences
                .iter()
                .copied()
                .map(|incident| (incident, self.bond_role_domain(incident.bond())))
                .collect::<Vec<_>>();
            let traversal_capable_count = role_domains
                .iter()
                .filter(|(_, domain)| domain.contains(BondRole::Traversal.value_index()))
                .count();

            if traversal_capable_count == 0 {
                frontier.contradiction = true;
                continue;
            }

            if incidences.len() == 1 {
                children.push(incidences[0]);
                continue;
            }

            ring_phase = true;
            let ring_opening_count = frontier.ring_openings.len();
            for (candidate, domain) in role_domains {
                if !domain.contains(BondRole::Ring.value_index()) {
                    continue;
                }
                let candidate_can_traverse = domain.contains(BondRole::Traversal.value_index());
                if traversal_capable_count > usize::from(candidate_can_traverse) {
                    frontier.ring_openings.push(candidate);
                }
            }
            if frontier.ring_openings.len() == ring_opening_count {
                frontier.contradiction = true;
            }
        }

        if frontier.contradiction {
            frontier.branch_children.clear();
            frontier.inline_children.clear();
            frontier.ring_openings.clear();
            frontier.ring_closures.clear();
            return frontier;
        }
        if ring_phase {
            return frontier;
        }

        match children.len() {
            0 => frontier.path_completion = self.traversal.path_completion(),
            1 => frontier.inline_children = children,
            _ => frontier.branch_children = children,
        }
        frontier
    }

    pub(crate) fn derive_candidates(&self) -> StructuralCandidateBatch {
        let frontier = self.structural_frontier();
        if frontier.is_contradiction() {
            return StructuralCandidateBatch {
                candidates: Vec::new(),
                contradiction: true,
            };
        }

        let mut candidates = Vec::new();
        candidates.extend(
            frontier
                .component_roots()
                .iter()
                .copied()
                .map(|atom| StructuralCandidate::Root { atom }),
        );
        candidates.extend(frontier.ring_closures().iter().copied().map(|incident| {
            let first_endpoint = self
                .traversal
                .ring_first_endpoint_for_active_incident(self.prepared.graph(), incident)
                .expect("an advertised ring closure must retain its first endpoint");
            StructuralCandidate::RingClose {
                incident,
                first_endpoint,
            }
        }));
        candidates.extend(
            frontier
                .ring_openings()
                .iter()
                .copied()
                .map(|incident| StructuralCandidate::RingOpen { incident }),
        );
        candidates.extend(
            frontier
                .branch_children()
                .iter()
                .copied()
                .map(|incident| StructuralCandidate::BranchChild { incident }),
        );
        candidates.extend(
            frontier
                .inline_children()
                .iter()
                .copied()
                .map(|incident| StructuralCandidate::InlineChild { incident }),
        );
        match frontier.path_completion() {
            Some(PathCompletion::CloseBranch) => {
                candidates.push(StructuralCandidate::CloseBranch);
            }
            Some(PathCompletion::FinishComponent) => {
                candidates.push(StructuralCandidate::FinishComponent);
            }
            None => {}
        }

        StructuralCandidateBatch {
            candidates,
            contradiction: false,
        }
    }

    pub(crate) fn attempt_candidate(
        &self,
        candidate: StructuralCandidate,
    ) -> Result<Consistency<Self>, S::Failure> {
        self.attempt_candidate_with_refinement(candidate, None, &[], &[])
    }

    pub(crate) fn attempt_candidate_with_bond_refinement(
        &self,
        candidate: StructuralCandidate,
        allowed_representations: Domain,
    ) -> Result<Consistency<Self>, S::Failure> {
        assert!(
            !allowed_representations.is_empty(),
            "a visible bond refinement must retain at least one representation"
        );
        assert!(
            matches!(
                candidate,
                StructuralCandidate::RingOpen { .. } | StructuralCandidate::RingClose { .. }
            ),
            "only a ring endpoint candidate accepts a visible bond refinement"
        );
        self.attempt_candidate_with_refinement(candidate, Some(allowed_representations), &[], &[])
    }

    pub(crate) fn attempt_candidate_with_semantic_transition(
        &self,
        candidate: StructuralCandidate,
        allowed_representations: Option<Domain>,
        restrictions: &[(VariableId, Domain)],
        activate: &[FactorId],
    ) -> Result<Consistency<Self>, S::Failure> {
        assert!(
            !restrictions.is_empty() || !activate.is_empty(),
            "a semantic candidate transition must restrict or activate prepared state"
        );
        self.attempt_candidate_with_refinement(
            candidate,
            allowed_representations,
            restrictions,
            activate,
        )
    }

    pub(crate) fn transitioned_semantics(
        &self,
        restrictions: &[(VariableId, Domain)],
        activate: &[FactorId],
    ) -> Result<Consistency<Self>, S::Failure> {
        assert!(
            !restrictions.is_empty() || !activate.is_empty(),
            "a semantic transition must restrict or activate prepared state"
        );
        Ok(self
            .restricted_constraints(None, None, restrictions, activate)?
            .map(|constraints| Self {
                prepared: self.prepared.clone(),
                traversal: self.traversal.clone(),
                constraints,
                #[cfg(test)]
                candidate_batch_derivations: self.candidate_batch_derivations.clone(),
            }))
    }

    fn attempt_candidate_with_refinement(
        &self,
        candidate: StructuralCandidate,
        allowed_representations: Option<Domain>,
        semantic_restrictions: &[(VariableId, Domain)],
        activate: &[FactorId],
    ) -> Result<Consistency<Self>, S::Failure> {
        let bond = match candidate {
            StructuralCandidate::RingOpen { incident }
            | StructuralCandidate::RingClose { incident, .. }
            | StructuralCandidate::BranchChild { incident }
            | StructuralCandidate::InlineChild { incident } => Some(incident.bond()),
            StructuralCandidate::Root { .. }
            | StructuralCandidate::CloseBranch
            | StructuralCandidate::FinishComponent => None,
        };
        let role = match candidate {
            StructuralCandidate::RingOpen { .. } => Some(BondRole::Ring),
            StructuralCandidate::BranchChild { .. } | StructuralCandidate::InlineChild { .. } => {
                Some(BondRole::Traversal)
            }
            StructuralCandidate::Root { .. }
            | StructuralCandidate::RingClose { .. }
            | StructuralCandidate::CloseBranch
            | StructuralCandidate::FinishComponent => None,
        };
        if allowed_representations.is_some() {
            assert!(matches!(
                candidate,
                StructuralCandidate::RingOpen { .. } | StructuralCandidate::RingClose { .. }
            ));
        }
        let bond_domain = allowed_representations
            .or_else(|| role.map(|role| self.role_domain(bond.unwrap(), role)));
        let restriction_bond = bond_domain.map(|_| bond.unwrap());
        let attempted = self
            .restricted_constraints(
                restriction_bond,
                bond_domain,
                semantic_restrictions,
                activate,
            )?
            .map(|constraints| {
                let mut traversal = self.traversal.clone();
                match candidate {
                    StructuralCandidate::Root { atom } => {
                        traversal.begin_component(self.prepared.graph(), atom);
                    }
                    StructuralCandidate::RingOpen { incident } => {
                        traversal.open_ring_endpoint(self.prepared.graph(), incident);
                    }
                    StructuralCandidate::RingClose {
                        incident,
                        first_endpoint,
                    } => {
                        assert_eq!(
                            traversal.ring_first_endpoint_for_active_incident(
                                self.prepared.graph(),
                                incident,
                            ),
                            Some(first_endpoint),
                            "a ring-closure candidate must retain its source-local first endpoint"
                        );
                        traversal.close_ring_endpoint(self.prepared.graph(), incident);
                    }
                    StructuralCandidate::BranchChild { incident }
                    | StructuralCandidate::InlineChild { incident } => {
                        traversal.commit_active_child(incident);
                    }
                    StructuralCandidate::CloseBranch => {
                        assert!(
                            traversal.complete_path(self.prepared.graph()).is_some(),
                            "closing a branch must restore its suspended parent"
                        );
                    }
                    StructuralCandidate::FinishComponent => {
                        assert_eq!(
                            traversal.complete_path(self.prepared.graph()),
                            None,
                            "finishing a component must not restore a branch parent"
                        );
                    }
                }
                Self {
                    prepared: self.prepared.clone(),
                    traversal,
                    constraints,
                    #[cfg(test)]
                    candidate_batch_derivations: self.candidate_batch_derivations.clone(),
                }
            });
        #[cfg(test)]
        {
            Ok(attempted.map(|mut successor| {
                successor.candidate_batch_derivations = Arc::new(AtomicUsize::new(0));
                successor
            }))
        }
        #[cfg(not(test))]
        {
            Ok(attempted)
        }
    }

    fn role_domain(&self, bond: BondId, role: BondRole) -> Domain {
        let partition = self
            .prepared
            .bond_role_partition(bond)
            .expect("prepared bond must have a role partition");
        match role {
            BondRole::Traversal => partition.traversal_values(),
            BondRole::Ring => partition.ring_values(),
        }
    }

    fn restricted_constraints(
        &self,
        bond: Option<BondId>,
        bond_domain: Option<Domain>,
        semantic_restrictions: &[(VariableId, Domain)],
        activate: &[FactorId],
    ) -> Result<Consistency<S>, S::Failure> {
        assert_eq!(bond.is_some(), bond_domain.is_some());
        let mut merged = BTreeMap::<VariableId, Domain>::new();
        if let (Some(bond), Some(domain)) = (bond, bond_domain) {
            merged.insert(decision_variable(&self.prepared, bond), domain);
        }
        for &(variable, domain) in semantic_restrictions {
            merged
                .entry(variable)
                .and_modify(|current| *current = current.intersect(domain))
                .or_insert(domain);
        }
        if merged.values().any(|domain| domain.is_empty()) {
            return Ok(Consistency::Contradiction);
        }
        if activate.is_empty() && merged.is_empty()
            || (semantic_restrictions.is_empty()
                && activate.is_empty()
                && merged.iter().all(|(variable, allowed)| {
                    self.constraints
                        .domain(*variable)
                        .expect("prepared restriction variable must exist")
                        .is_subset_of(*allowed)
                }))
        {
            return Ok(Consistency::Consistent(self.constraints.clone()));
        }
        self.constraints
            .transitioned(&merged.into_iter().collect::<Vec<_>>(), activate)
    }

    #[cfg(test)]
    pub(crate) fn candidate_batch_derivation_count(&self) -> usize {
        self.candidate_batch_derivations.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn constraints_for_test(&self) -> &S {
        &self.constraints
    }

    #[cfg(test)]
    pub(crate) fn observe_raw(&self) -> ObservedWriterState {
        let bond_plan_domains = self
            .prepared
            .graph()
            .bond_ids()
            .map(|bond| {
                self.constraints
                    .domain(decision_variable(&self.prepared, bond))
                    .expect("observed bond decision must belong to the prepared solver state")
            })
            .collect();
        ObservedWriterState {
            traversal: self.traversal.observe_raw(),
            bond_plan_domains,
            active_factors: (0..self.prepared.constraint_model().factor_count())
                .map(|index| FactorId::new(u32::try_from(index).unwrap()))
                .filter(|factor| self.factor_is_active(*factor))
                .collect(),
        }
    }

    pub(crate) fn enter_committed_inline_child(&self, incident: AdjacentBond) -> Self {
        assert_eq!(
            self.bond_role_domain(incident.bond()),
            BondRole::Traversal.singleton_domain(),
            "an inline child must already be committed to Traversal"
        );
        let mut successor = self.clone();
        successor
            .traversal
            .enter_inline_child(self.prepared.graph(), incident);
        successor
    }

    pub(crate) fn active_local_bond_order(&self) -> LocalBondOrder {
        self.traversal.active_local_bond_order()
    }

    pub(crate) fn active_local_layout_context(&self) -> LocalLayoutContext {
        self.traversal
            .active_local_layout_context(self.prepared.graph())
    }

    pub(crate) fn prospective_root_layout_context(&self, atom: AtomId) -> LocalLayoutContext {
        let mut traversal = self.traversal.clone();
        traversal.begin_component(self.prepared.graph(), atom);
        traversal.active_local_layout_context(self.prepared.graph())
    }

    pub(crate) fn prospective_inline_child_layout_context(
        &self,
        incident: AdjacentBond,
    ) -> LocalLayoutContext {
        let mut traversal = self.traversal.clone();
        traversal.commit_active_child(incident);
        traversal.enter_inline_child(self.prepared.graph(), incident);
        traversal.active_local_layout_context(self.prepared.graph())
    }

    pub(crate) fn prospective_committed_inline_child_layout_context(
        &self,
        incident: AdjacentBond,
    ) -> LocalLayoutContext {
        let mut traversal = self.traversal.clone();
        traversal.enter_inline_child(self.prepared.graph(), incident);
        traversal.active_local_layout_context(self.prepared.graph())
    }

    pub(crate) fn prospective_committed_branch_child_layout_context(
        &self,
        incident: AdjacentBond,
    ) -> LocalLayoutContext {
        let mut traversal = self.traversal.clone();
        traversal.enter_branch_child(self.prepared.graph(), incident);
        traversal.active_local_layout_context(self.prepared.graph())
    }

    pub(crate) fn enter_committed_branch_child(&self, incident: AdjacentBond) -> Self {
        assert_eq!(
            self.bond_role_domain(incident.bond()),
            BondRole::Traversal.singleton_domain(),
            "a branch child must already be committed to Traversal"
        );
        let mut successor = self.clone();
        successor
            .traversal
            .enter_branch_child(self.prepared.graph(), incident);
        successor
    }

    #[cfg(test)]
    fn restricted_role(&self, bond: BondId, role: BondRole) -> Result<Consistency<S>, S::Failure> {
        let allowed = self.role_domain(bond, role);
        let variable = decision_variable(&self.prepared, bond);
        let current = self
            .constraints
            .domain(variable)
            .expect("prepared bond decision must belong to the writer constraint model");
        if current.is_subset_of(allowed) {
            return Ok(Consistency::Consistent(self.constraints.clone()));
        }
        self.constraints.restricted(&[(variable, allowed)])
    }
}

fn assert_initial_solver_shape<S: ConstraintSolver>(prepared: &PreparedMolecule, constraints: &S) {
    for index in 0..prepared.constraint_model().variable_count() {
        let variable = VariableId::new(
            u32::try_from(index).expect("prepared variable count must fit the identifier space"),
        );
        let initial = prepared
            .constraint_model()
            .variable(variable)
            .expect("prepared variable index must resolve")
            .initial_domain();
        let current = constraints
            .domain(variable)
            .expect("a solver state must retain every prepared variable");
        assert!(
            !current.is_empty() && current.is_subset_of(initial),
            "a solver state domain must be nonempty and refine its prepared domain"
        );
    }
}

fn decision_variable(prepared: &PreparedMolecule, bond: BondId) -> VariableId {
    prepared
        .bond_decision_variable(bond)
        .expect("prepared bond must have a decision variable")
}

#[cfg(test)]
fn role_restriction(
    prepared: &PreparedMolecule,
    bond: BondId,
    role: BondRole,
) -> (VariableId, Domain) {
    let partition = prepared
        .bond_role_partition(bond)
        .expect("prepared bond must have a role partition");
    let domain = match role {
        BondRole::Traversal => partition.traversal_values(),
        BondRole::Ring => partition.ring_values(),
    };
    (decision_variable(prepared, bond), domain)
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;
    use crate::model::EdgeRolePartition;
    use crate::native::NativeSolverState;
    use crate::prepared::PreparedGraphBuilder;

    #[derive(Clone)]
    struct MissingDomainSolver;

    impl ConstraintSolver for MissingDomainSolver {
        type Failure = Infallible;

        fn initial(
            _model: Arc<crate::model::ConstraintModel>,
        ) -> Result<Consistency<Self>, Self::Failure> {
            Ok(Consistency::Consistent(Self))
        }

        fn restricted(
            &self,
            _restrictions: &[(VariableId, Domain)],
        ) -> Result<Consistency<Self>, Self::Failure> {
            Ok(Consistency::Consistent(self.clone()))
        }

        fn transitioned(
            &self,
            _restrictions: &[(VariableId, Domain)],
            _activate: &[FactorId],
        ) -> Result<Consistency<Self>, Self::Failure> {
            Ok(Consistency::Consistent(self.clone()))
        }

        fn domain(&self, _variable: VariableId) -> Option<Domain> {
            None
        }

        fn factor_is_active(&self, _factor: FactorId) -> Option<bool> {
            None
        }
    }

    #[derive(Clone)]
    struct DomainCountingSolver {
        native: NativeSolverState,
        domain_reads: Arc<AtomicUsize>,
        restriction_calls: Arc<AtomicUsize>,
    }

    impl ConstraintSolver for DomainCountingSolver {
        type Failure = crate::native_solver::NativeSolverFailure;

        fn initial(
            model: Arc<crate::model::ConstraintModel>,
        ) -> Result<Consistency<Self>, Self::Failure> {
            Ok(
                <NativeSolverState as ConstraintSolver>::initial(model)?.map(|native| Self {
                    native,
                    domain_reads: Arc::new(AtomicUsize::new(0)),
                    restriction_calls: Arc::new(AtomicUsize::new(0)),
                }),
            )
        }

        fn restricted(
            &self,
            restrictions: &[(VariableId, Domain)],
        ) -> Result<Consistency<Self>, Self::Failure> {
            self.restriction_calls.fetch_add(1, Ordering::Relaxed);
            Ok(
                <NativeSolverState as ConstraintSolver>::restricted(&self.native, restrictions)?
                    .map(|native| Self {
                        native,
                        domain_reads: Arc::clone(&self.domain_reads),
                        restriction_calls: Arc::clone(&self.restriction_calls),
                    }),
            )
        }

        fn transitioned(
            &self,
            restrictions: &[(VariableId, Domain)],
            activate: &[FactorId],
        ) -> Result<Consistency<Self>, Self::Failure> {
            self.restriction_calls.fetch_add(1, Ordering::Relaxed);
            Ok(<NativeSolverState as ConstraintSolver>::transitioned(
                &self.native,
                restrictions,
                activate,
            )?
            .map(|native| Self {
                native,
                domain_reads: Arc::clone(&self.domain_reads),
                restriction_calls: Arc::clone(&self.restriction_calls),
            }))
        }

        fn domain(&self, variable: VariableId) -> Option<Domain> {
            self.domain_reads.fetch_add(1, Ordering::Relaxed);
            self.native.domain(variable)
        }

        fn factor_is_active(&self, factor: FactorId) -> Option<bool> {
            self.native.factor_is_active(factor)
        }
    }

    fn incident(prepared: &PreparedMolecule, atom: AtomId, bond: BondId) -> AdjacentBond {
        prepared
            .graph()
            .neighbors(atom)
            .expect("fixture atom must exist")
            .iter()
            .copied()
            .find(|candidate| candidate.bond() == bond)
            .expect("fixture bond must be incident to the atom")
    }

    fn attempt(
        state: &WriterState<NativeSolverState>,
        candidate: StructuralCandidate,
    ) -> WriterState<NativeSolverState> {
        state
            .attempt_candidate(candidate)
            .unwrap()
            .unwrap_consistent()
    }

    fn rooted(prepared: &PreparedMolecule, atom: AtomId) -> WriterState<NativeSolverState> {
        let initial = WriterState::<NativeSolverState>::initial(prepared)
            .unwrap()
            .unwrap_consistent();
        assert!(initial
            .derive_candidates()
            .candidates()
            .contains(&StructuralCandidate::Root { atom }));
        attempt(&initial, StructuralCandidate::Root { atom })
    }

    #[test]
    #[should_panic(expected = "a solver state must retain every prepared variable")]
    fn initial_state_checks_the_solver_domain_shape_at_the_boundary() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 2] = std::array::from_fn(|_| graph.add_atom().unwrap());
        graph.add_bond(atoms[0], atoms[1]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());

        let _ = WriterState::<MissingDomainSolver>::initial(&prepared);
    }

    #[test]
    fn role_restriction_does_not_rescan_the_solver_domain_shape() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let left = graph.add_bond(atoms[0], atoms[1]).unwrap();
        graph.add_bond(atoms[0], atoms[2]).unwrap();
        graph.add_bond(atoms[1], atoms[2]).unwrap();
        for _ in 0..128 {
            let a = graph.add_atom().unwrap();
            let b = graph.add_atom().unwrap();
            graph.add_bond(a, b).unwrap();
        }
        let prepared = PreparedMolecule::new(graph.build());
        assert_eq!(prepared.constraint_model().variable_count(), 131);
        let state = WriterState::<DomainCountingSolver>::initial(&prepared)
            .unwrap()
            .unwrap_consistent();
        state.constraints.domain_reads.store(0, Ordering::Relaxed);

        let _ = state
            .restricted_role(left, BondRole::Ring)
            .unwrap()
            .unwrap_consistent();

        assert_eq!(state.constraints.domain_reads.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn ring_endpoint_refinement_is_one_atomic_solver_restriction() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let left = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let right = graph.add_bond(atoms[0], atoms[2]).unwrap();
        let between = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let binary = PreparedMolecule::new(graph.build());
        let traversal = Domain::from_bits(1);
        let ring = Domain::from_bits(0b1_1110);
        let plan = traversal.union(ring);
        let prepared = PreparedMolecule::with_bond_decisions(
            &binary,
            &[plan; 3],
            &[EdgeRolePartition::new(traversal, ring); 3],
        );
        let initial = WriterState::<DomainCountingSolver>::initial(&prepared)
            .unwrap()
            .unwrap_consistent();
        let rooted = initial
            .attempt_candidate(StructuralCandidate::Root { atom: atoms[0] })
            .unwrap()
            .unwrap_consistent();
        let opening_incident = incident(&prepared, atoms[0], left);
        rooted
            .constraints
            .restriction_calls
            .store(0, Ordering::Relaxed);

        let selected = Domain::from_bits(0b1_0100);
        let opened = rooted
            .attempt_candidate_with_bond_refinement(
                StructuralCandidate::RingOpen {
                    incident: opening_incident,
                },
                selected,
            )
            .unwrap()
            .unwrap_consistent();

        assert_eq!(
            rooted.constraints.restriction_calls.load(Ordering::Relaxed),
            1
        );
        assert_eq!(rooted.bond_decision_domain(left), plan);
        assert_eq!(opened.bond_decision_domain(left), selected);
        assert_eq!(
            opened.bond_role_domain(left),
            BondRole::Ring.singleton_domain()
        );

        let right_incident = incident(&prepared, atoms[0], right);
        let walked = opened
            .attempt_candidate(StructuralCandidate::InlineChild {
                incident: right_incident,
            })
            .unwrap()
            .unwrap_consistent()
            .enter_committed_inline_child(right_incident);
        let between_incident = incident(&prepared, atoms[2], between);
        let walked = walked
            .attempt_candidate(StructuralCandidate::InlineChild {
                incident: between_incident,
            })
            .unwrap()
            .unwrap_consistent()
            .enter_committed_inline_child(between_incident);
        let closing_incident = incident(&prepared, atoms[1], left);
        walked
            .constraints
            .restriction_calls
            .store(0, Ordering::Relaxed);
        let resolved = Domain::from_bits(0b1_0000);
        let closed = walked
            .attempt_candidate_with_bond_refinement(
                StructuralCandidate::RingClose {
                    incident: closing_incident,
                    first_endpoint: atoms[0],
                },
                resolved,
            )
            .unwrap()
            .unwrap_consistent();

        assert_eq!(
            walked.constraints.restriction_calls.load(Ordering::Relaxed),
            1
        );
        assert_eq!(walked.bond_decision_domain(left), selected);
        assert_eq!(closed.bond_decision_domain(left), resolved);
    }

    #[test]
    fn completion_distinguishes_branch_return_from_component_finish() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 2] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = rooted(&prepared, atoms[0]);

        assert!(!rooted.graph_is_complete());
        assert_eq!(
            rooted.derive_candidates().candidates(),
            &[StructuralCandidate::FinishComponent]
        );

        let finished = attempt(&rooted, StructuralCandidate::FinishComponent);
        assert_eq!(finished.active_atom(), None);
        assert_eq!(
            finished.derive_candidates().candidates(),
            &[StructuralCandidate::Root { atom: atoms[1] }]
        );
    }

    #[test]
    fn triangle_requires_one_ring_before_its_single_attachment_entry() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let left = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let right = graph.add_bond(atoms[0], atoms[2]).unwrap();
        let between = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = rooted(&prepared, atoms[0]);
        let left_incident = incident(&prepared, atoms[0], left);
        let right_incident = incident(&prepared, atoms[0], right);

        let frontier = rooted.structural_frontier();
        assert_eq!(frontier.ring_openings(), &[left_incident, right_incident]);
        assert!(frontier.branch_children().is_empty());
        assert!(frontier.inline_children().is_empty());

        let opened = attempt(
            &rooted,
            StructuralCandidate::RingOpen {
                incident: left_incident,
            },
        );
        assert_eq!(
            rooted.bond_role_domain(left),
            BondRole::role_domain(),
            "the source state must remain unchanged"
        );
        let opened_frontier = opened.structural_frontier();
        assert!(opened_frontier.ring_openings().is_empty());
        assert_eq!(opened_frontier.inline_children(), &[right_incident]);
        assert_eq!(
            opened.bond_role_domain(left),
            BondRole::Ring.singleton_domain()
        );
        assert_eq!(
            opened.bond_role_domain(right),
            BondRole::Traversal.singleton_domain()
        );
        assert_eq!(
            opened.bond_role_domain(between),
            BondRole::Traversal.singleton_domain()
        );

        let committed = attempt(
            &opened,
            StructuralCandidate::InlineChild {
                incident: right_incident,
            },
        );
        let walked = committed.enter_committed_inline_child(right_incident);
        let between_incident = incident(&prepared, atoms[2], between);
        let committed = attempt(
            &walked,
            StructuralCandidate::InlineChild {
                incident: between_incident,
            },
        );
        let walked = committed.enter_committed_inline_child(between_incident);
        let closing = incident(&prepared, atoms[1], left);
        assert_eq!(walked.structural_frontier().ring_closures(), &[closing]);
        let closing_candidate = StructuralCandidate::RingClose {
            incident: closing,
            first_endpoint: atoms[0],
        };
        assert!(walked
            .derive_candidates()
            .candidates()
            .contains(&closing_candidate));

        let closed = attempt(&walked, closing_candidate);
        assert!(closed.graph_is_complete());
        assert_eq!(
            closed.structural_frontier().path_completion(),
            Some(PathCompletion::FinishComponent)
        );
        let finished = attempt(&closed, StructuralCandidate::FinishComponent);
        assert_eq!(finished.active_atom(), None);
    }

    #[test]
    fn a_matroid_basis_can_still_contradict_writer_policy() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let left = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let right = graph.add_bond(atoms[0], atoms[2]).unwrap();
        let between = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = rooted(&prepared, atoms[0]);
        let constraints = rooted
            .constraints
            .restricted(&[
                role_restriction(&prepared, left, BondRole::Traversal),
                role_restriction(&prepared, right, BondRole::Traversal),
                role_restriction(&prepared, between, BondRole::Ring),
            ])
            .unwrap()
            .unwrap_consistent();
        let basis = WriterState {
            prepared: rooted.prepared.clone(),
            traversal: rooted.traversal.clone(),
            constraints,
            candidate_batch_derivations: rooted.candidate_batch_derivations.clone(),
        };

        let frontier = basis.structural_frontier();
        assert!(frontier.is_contradiction());
        assert!(frontier.ring_openings().is_empty());
        assert!(frontier.branch_children().is_empty());
        assert!(frontier.inline_children().is_empty());
    }

    #[test]
    fn star_attachments_are_branches_then_one_inline_continuation() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 4] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let bonds = [
            graph.add_bond(atoms[0], atoms[1]).unwrap(),
            graph.add_bond(atoms[0], atoms[2]).unwrap(),
            graph.add_bond(atoms[0], atoms[3]).unwrap(),
        ];
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = rooted(&prepared, atoms[0]);
        let children = bonds.map(|bond| incident(&prepared, atoms[0], bond));

        assert_eq!(rooted.structural_frontier().branch_children(), &children);
        let committed = attempt(
            &rooted,
            StructuralCandidate::BranchChild {
                incident: children[0],
            },
        );
        let branch = committed.enter_committed_branch_child(children[0]);
        assert_eq!(
            branch.structural_frontier().path_completion(),
            Some(PathCompletion::CloseBranch)
        );
        let restored = attempt(&branch, StructuralCandidate::CloseBranch);

        assert_eq!(
            restored.structural_frontier().branch_children(),
            &children[1..]
        );
        let committed = attempt(
            &restored,
            StructuralCandidate::BranchChild {
                incident: children[1],
            },
        );
        let branch = committed.enter_committed_branch_child(children[1]);
        let restored = attempt(&branch, StructuralCandidate::CloseBranch);
        assert_eq!(
            restored.structural_frontier().inline_children(),
            &[children[2]]
        );
    }

    #[test]
    fn ring_neighbours_share_an_attachment_but_a_substituent_does_not() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 4] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let left = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let right = graph.add_bond(atoms[0], atoms[2]).unwrap();
        graph.add_bond(atoms[1], atoms[2]).unwrap();
        let substituent = graph.add_bond(atoms[0], atoms[3]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = rooted(&prepared, atoms[0]);

        let frontier = rooted.structural_frontier();
        assert_eq!(
            frontier.ring_openings(),
            &[
                incident(&prepared, atoms[0], left),
                incident(&prepared, atoms[0], right),
            ]
        );
        assert!(frontier.branch_children().is_empty());

        let opened = attempt(
            &rooted,
            StructuralCandidate::RingOpen {
                incident: incident(&prepared, atoms[0], left),
            },
        );
        assert_eq!(
            opened.structural_frontier().branch_children(),
            &[
                incident(&prepared, atoms[0], right),
                incident(&prepared, atoms[0], substituent),
            ]
        );
    }

    #[test]
    fn fused_residual_system_remains_one_attachment() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 5] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let root_edges = [
            graph.add_bond(atoms[0], atoms[1]).unwrap(),
            graph.add_bond(atoms[0], atoms[2]).unwrap(),
            graph.add_bond(atoms[0], atoms[3]).unwrap(),
        ];
        graph.add_bond(atoms[1], atoms[4]).unwrap();
        graph.add_bond(atoms[2], atoms[4]).unwrap();
        graph.add_bond(atoms[3], atoms[4]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = rooted(&prepared, atoms[0]);
        let openings = root_edges.map(|bond| incident(&prepared, atoms[0], bond));

        assert_eq!(rooted.structural_frontier().ring_openings(), &openings);
        assert!(rooted.structural_frontier().branch_children().is_empty());
    }
}
