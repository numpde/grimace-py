//! Concrete composition of residual writer frames and constraint state.
//!
//! The graphic-matroid factor guards global Traversal/Ring feasibility. Writer
//! Tree policy gives each residual attachment one traversal entry. Forest
//! traversal may return to a parent after visiting one incidence of a chemical
//! residual component, then enter another incidence of that same component.

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
use crate::walker::ComponentOrder;

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
    dot_branch_roots: Vec<AtomId>,
    dot_segment_roots: Vec<AtomId>,
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

    pub(crate) fn dot_branch_roots(&self) -> &[AtomId] {
        &self.dot_branch_roots
    }

    pub(crate) fn dot_segment_roots(&self) -> &[AtomId] {
        &self.dot_segment_roots
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
    DotBranchRoot { atom: AtomId },
    DotSegmentRoot { atom: AtomId },
    CloseBranch,
    FinishSegment,
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
    component_order: ComponentOrder,
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
    #[cfg(test)]
    pub(crate) fn restricted_bond_domains_for_test(
        &self,
        restrictions: &[(BondId, Domain)],
    ) -> Result<Consistency<Self>, S::Failure> {
        let restrictions = restrictions.iter().map(|&(bond, domain)| {
            (decision_variable(&self.prepared, bond), domain)
        }).collect::<Vec<_>>();
        Ok(self.constraints.restricted(&restrictions)?.map(|constraints| {
            let mut next = self.clone();
            next.constraints = constraints;
            next
        }))
    }

    #[cfg(test)]
    pub(crate) fn initial(prepared: &PreparedMolecule) -> Result<Consistency<Self>, S::Failure> {
        Self::initial_with_component_order(prepared, ComponentOrder::AnyOrder)
    }

    pub(crate) fn initial_with_component_order(
        prepared: &PreparedMolecule,
        component_order: ComponentOrder,
    ) -> Result<Consistency<Self>, S::Failure> {
        Ok(
            S::initial(prepared.constraint_model_arc())?.map(|constraints| {
                assert_initial_solver_shape(prepared, &constraints);
                Self {
                    prepared: prepared.clone(),
                    component_order,
                    traversal: TraversalState::new_with_segments(
                        prepared.graph(),
                        prepared.topology() == crate::model::TraversalTopologyRequirement::Forest,
                    ),
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

    pub(crate) fn scope_is_complete(&self) -> bool {
        self.traversal.scope_is_complete()
    }

    pub(crate) fn prepared_topology(&self) -> crate::model::TraversalTopologyRequirement {
        self.prepared.topology()
    }

    pub(crate) fn visited_atom_count(&self) -> usize {
        self.traversal.visited_atom_count()
    }

    pub(crate) fn open_branch_roots(&self) -> Vec<AtomId> {
        self.traversal.open_branch_roots()
    }

    pub(crate) fn closing_branch_root(&self) -> Option<AtomId> {
        self.traversal.closing_branch_root()
    }

    pub(crate) fn fixed_bond_role(&self, bond: BondId) -> Option<BondRole> {
        let roles = self.bond_role_domain(bond);
        if roles == BondRole::Traversal.singleton_domain() { Some(BondRole::Traversal) }
        else if roles == BondRole::Ring.singleton_domain() { Some(BondRole::Ring) }
        else { None }
    }

    pub(crate) const fn graph_is_complete(&self) -> bool {
        self.traversal.graph_is_complete()
    }

    pub(crate) fn chemical_component_is_complete(&self, component: usize) -> bool {
        self.traversal.chemical_component_is_complete(self.prepared.graph(), component)
    }

    pub(crate) const fn has_visited_atoms(&self) -> bool {
        self.traversal.has_visited_atoms()
    }

    pub(crate) fn ring_is_open(&self, bond: BondId) -> bool {
        self.traversal.ring_is_open(bond)
    }

    pub(crate) fn atom_is_visited(&self, atom: AtomId) -> bool {
        self.traversal.atom_is_visited(atom)
    }

    #[cfg(test)]
    pub(crate) fn bond_role_domain_for_test(&self, bond: BondId) -> Domain {
        self.bond_role_domain(bond)
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
            .domain_bound(decision_variable(&self.prepared, bond))
            .expect("prepared bond decision must belong to the writer constraint model")
    }

    pub(crate) fn semantic_domain(&self, variable: VariableId) -> Domain {
        self.constraints
            .domain_bound(variable)
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
            let mut component_roots = self.traversal.unvisited_atoms(graph).collect::<Vec<_>>();
            if self.prepared.topology() == crate::model::TraversalTopologyRequirement::Forest {
                component_roots.retain(|atom| self.traversal.can_enter_atom(graph, *atom, None));
            }
            if self.component_order == ComponentOrder::InputOrder {
                if let Some(first_unvisited) = component_roots.first().copied() {
                    let component = graph
                        .component_of_atom(first_unvisited)
                        .expect("unvisited atom must belong to a prepared component");
                    component_roots
                        .retain(|atom| graph.component_of_atom(*atom) == Some(component));
                }
            }
            return StructuralFrontier {
                component_roots,
                ..StructuralFrontier::default()
            };
        };

        let mut frontier = StructuralFrontier::default();
        let mut ring_phase = false;
        let mut children = Vec::new();
        let can_emit_ring = self.traversal.active_can_emit_ring();
        let segments = self.prepared.topology()
            == crate::model::TraversalTopologyRequirement::Forest;
        let unresolved_incidence_count = self.traversal.active_attachments()
            .iter()
            .map(|attachment| attachment.incidences().len())
            .sum::<usize>();

        for incident in graph
            .neighbors(active)
            .expect("active atom must belong to the prepared graph")
            .iter()
            .copied()
        {
            match self.traversal.classify_active_incident(graph, incident) {
                IncidentBondState::RingOpenAtOtherAtom => {
                    if !can_emit_ring {
                        frontier.contradiction = true;
                        continue;
                    }
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
                .filter(|(incident, domain)| {
                    domain.contains(BondRole::Traversal.value_index())
                        && self.traversal.can_enter_atom(graph, incident.atom(), Some(incident.bond()))
                })
                .count();

            if segments {
                let ring_opening_count = frontier.ring_openings.len();
                for (incident, domain) in role_domains {
                    if domain.contains(BondRole::Traversal.value_index())
                        && self.traversal.can_enter_atom(graph, incident.atom(), Some(incident.bond()))
                    {
                        children.push(incident);
                    }
                    if can_emit_ring && domain.contains(BondRole::Ring.value_index()) {
                        frontier.ring_openings.push(incident);
                    }
                }
                if traversal_capable_count == 0
                    && frontier.ring_openings.len() == ring_opening_count
                {
                    frontier.contradiction = true;
                }
                continue;
            }
            if traversal_capable_count == 0 {
                frontier.contradiction = true;
                continue;
            }

            if incidences.len() == 1 {
                if traversal_capable_count == 1 {
                    children.push(incidences[0]);
                }
                continue;
            }

            ring_phase = true;
            let ring_opening_count = frontier.ring_openings.len();
            for (candidate, domain) in role_domains {
                if !can_emit_ring || !domain.contains(BondRole::Ring.value_index()) {
                    continue;
                }
                let candidate_can_traverse = domain.contains(BondRole::Traversal.value_index())
                    && self.traversal.can_enter_atom(graph, candidate.atom(), Some(candidate.bond()));
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

        if self.prepared.topology() == crate::model::TraversalTopologyRequirement::Forest {
            let roots = self.traversal.unvisited_atoms(graph)
                .filter(|atom| self.traversal.can_enter_atom(graph, *atom, None))
                .collect::<Vec<_>>();
            if self.can_close_parent_ring_phase(None) {
                frontier.dot_branch_roots = roots.clone();
            }
            if self.traversal.can_finish_active_segment() {
                frontier.dot_segment_roots = roots;
            }
        }

        match children.len() {
            0 if self.traversal.can_finish_active_segment() => {
                frontier.path_completion = self.traversal.path_completion();
            }
            0 => {}
            1 if !segments => frontier.inline_children = children,
            _ if !children.is_empty() => {
                frontier.branch_children = if segments {
                    children.into_iter()
                        .filter(|incident| self.can_close_parent_ring_phase(Some(incident.bond())))
                        .collect()
                } else {
                    children
                };
            }
            _ => {}
        }
        if segments && unresolved_incidence_count == 1 {
            frontier.inline_children = frontier.branch_children.clone();
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
        candidates.extend(frontier.dot_branch_roots().iter().copied().map(|atom| StructuralCandidate::DotBranchRoot { atom }));
        candidates.extend(frontier.dot_segment_roots().iter().copied().map(|atom| StructuralCandidate::DotSegmentRoot { atom }));
        match frontier.path_completion() {
            Some(PathCompletion::CloseBranch) => {
                candidates.push(StructuralCandidate::CloseBranch);
            }
            Some(PathCompletion::FinishSegment) if frontier.dot_segment_roots().is_empty() => {
                candidates.push(StructuralCandidate::FinishSegment);
            }
            Some(PathCompletion::FinishSegment) => {}
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
                component_order: self.component_order,
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
        #[cfg(test)]
        crate::work_diagnostic::record(|counts| counts.candidate_attempts += 1);
        match candidate {
            StructuralCandidate::BranchChild { incident }
                if self.prepared.topology() == crate::model::TraversalTopologyRequirement::Forest
                    && !self.can_close_parent_ring_phase(Some(incident.bond())) => {
                #[cfg(test)]
                crate::work_diagnostic::record(|counts| counts.candidate_rejections += 1);
                return Ok(Consistency::Contradiction);
            }
            StructuralCandidate::DotBranchRoot { .. }
                if !self.can_close_parent_ring_phase(None) => {
                #[cfg(test)]
                crate::work_diagnostic::record(|counts| counts.candidate_rejections += 1);
                return Ok(Consistency::Contradiction);
            }
            _ => {}
        }
        let bond = match candidate {
            StructuralCandidate::RingOpen { incident }
            | StructuralCandidate::RingClose { incident, .. }
            | StructuralCandidate::BranchChild { incident }
            | StructuralCandidate::InlineChild { incident } => Some(incident.bond()),
            StructuralCandidate::Root { .. }
            | StructuralCandidate::DotBranchRoot { .. }
            | StructuralCandidate::DotSegmentRoot { .. }
            | StructuralCandidate::CloseBranch
            | StructuralCandidate::FinishSegment => None,
        };
        let role = match candidate {
            StructuralCandidate::RingOpen { .. } => Some(BondRole::Ring),
            StructuralCandidate::BranchChild { .. } | StructuralCandidate::InlineChild { .. } => {
                Some(BondRole::Traversal)
            }
            StructuralCandidate::Root { .. }
            | StructuralCandidate::DotBranchRoot { .. }
            | StructuralCandidate::DotSegmentRoot { .. }
            | StructuralCandidate::RingClose { .. }
            | StructuralCandidate::CloseBranch
            | StructuralCandidate::FinishSegment => None,
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
        let mut restrictions = semantic_restrictions.to_vec();
        let entering = match candidate {
            StructuralCandidate::Root { atom }
            | StructuralCandidate::DotBranchRoot { atom }
            | StructuralCandidate::DotSegmentRoot { atom } => Some((atom, 1)),
            StructuralCandidate::BranchChild { incident }
            | StructuralCandidate::InlineChild { incident } => Some((incident.atom(), 0)),
            _ => None,
        };
        if let Some((atom, role)) = entering {
            if let Some(variable) = self.prepared.root_variable(atom) {
                restrictions.push((variable, Domain::singleton(role).unwrap()));
            }
        }
        if matches!(candidate, StructuralCandidate::BranchChild { .. } | StructuralCandidate::DotBranchRoot { .. }) {
            for attachment in self.traversal.active_attachments() {
                for incident in attachment.incidences() {
                    if Some(incident.bond()) != bond {
                        restrictions.push((
                            decision_variable(&self.prepared, incident.bond()),
                            self.role_domain(incident.bond(), BondRole::Traversal),
                        ));
                    }
                }
            }
        }
        let attempted = self
            .restricted_constraints(
                restriction_bond,
                bond_domain,
                &restrictions,
                activate,
            )?
            .map(|constraints| {
                let mut traversal = self.traversal.clone();
                match candidate {
                    StructuralCandidate::Root { atom } => {
                        traversal.begin_segment(self.prepared.graph(), atom);
                    }
                    StructuralCandidate::DotBranchRoot { atom } => {
                        traversal.begin_dot_branch(self.prepared.graph(), atom);
                    }
                    StructuralCandidate::DotSegmentRoot { atom } => {
                        traversal.begin_dot_segment(self.prepared.graph(), atom);
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
                    StructuralCandidate::FinishSegment => {
                        assert_eq!(
                            traversal.complete_path(self.prepared.graph()),
                            None,
                            "finishing a component must not restore a branch parent"
                        );
                    }
                }
                Self {
                    prepared: self.prepared.clone(),
                    component_order: self.component_order,
                    traversal,
                    constraints,
                    #[cfg(test)]
                    candidate_batch_derivations: self.candidate_batch_derivations.clone(),
                }
            });
        #[cfg(test)]
        if matches!(&attempted, Consistency::Contradiction) {
            crate::work_diagnostic::record(|counts| counts.candidate_rejections += 1);
        }
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

    fn can_close_parent_ring_phase(&self, selected_bond: Option<BondId>) -> bool {
        let graph = self.prepared.graph();
        for attachment in self.traversal.active_attachments() {
            for incident in attachment.incidences() {
                if Some(incident.bond()) == selected_bond {
                    continue;
                }
                if !self.bond_role_domain(incident.bond()).contains(BondRole::Traversal.value_index()) {
                    return false;
                }
                if graph.neighbors(incident.atom()).unwrap().iter().any(|other| {
                    other.bond() != incident.bond()
                        && self.traversal.atom_is_visited(other.atom())
                        && self.bond_role_domain(other.bond())
                            .is_subset_of(BondRole::Traversal.singleton_domain())
                }) {
                    return false;
                }
            }
        }
        true
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
                        .domain_bound(*variable)
                        .expect("prepared restriction variable must exist")
                        .is_subset_of(*allowed)
                }))
        {
            return Ok(Consistency::Consistent(self.constraints.clone()));
        }
        self.constraints
            .transitioned_feasible(&merged.into_iter().collect::<Vec<_>>(), activate)
    }

    #[cfg(test)]
    pub(crate) fn candidate_batch_derivation_count(&self) -> usize {
        self.candidate_batch_derivations.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn reset_traversal_copy_counts(&self) {
        self.traversal.reset_copy_counts();
    }

    #[cfg(test)]
    pub(crate) fn traversal_copy_counts(&self) -> (usize, usize) {
        self.traversal.copy_counts()
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
                    .domain_bound(decision_variable(&self.prepared, bond))
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
        traversal.begin_segment(self.prepared.graph(), atom);
        traversal.active_local_layout_context(self.prepared.graph())
    }

    pub(crate) fn prospective_dot_branch_layout_context(&self, atom: AtomId) -> LocalLayoutContext {
        let mut traversal = self.traversal.clone();
        traversal.begin_dot_branch(self.prepared.graph(), atom);
        traversal.active_local_layout_context(self.prepared.graph())
    }

    pub(crate) fn prospective_dot_segment_layout_context(&self, atom: AtomId) -> LocalLayoutContext {
        let mut traversal = self.traversal.clone();
        traversal.begin_dot_segment(self.prepared.graph(), atom);
        traversal.active_local_layout_context(self.prepared.graph())
    }

    pub(crate) fn prospective_inline_child_layout_context(
        &self,
        incident: AdjacentBond,
    ) -> LocalLayoutContext {
        let traversal = self.prospective_child_traversal(StructuralCandidate::InlineChild { incident });
        traversal.active_local_layout_context(self.prepared.graph())
    }

    pub(crate) fn prospective_branch_child_layout_context(
        &self,
        incident: AdjacentBond,
    ) -> LocalLayoutContext {
        let traversal = self.prospective_child_traversal(StructuralCandidate::BranchChild { incident });
        traversal.active_local_layout_context(self.prepared.graph())
    }

    fn prospective_child_traversal(&self, candidate: StructuralCandidate) -> TraversalState {
        let mut traversal = self.traversal.clone();
        match candidate {
            StructuralCandidate::InlineChild { incident } => {
                traversal.commit_active_child(incident);
                traversal.enter_inline_child(self.prepared.graph(), incident);
            }
            StructuralCandidate::BranchChild { incident } => {
                traversal.commit_active_child(incident);
                traversal.enter_branch_child(self.prepared.graph(), incident);
            }
            _ => panic!("child preview requires a child candidate"),
        }
        traversal
    }

    pub(crate) fn may_release_held_ring_after_child_entry(
        &self,
        candidate: StructuralCandidate,
        held_bonds: &[BondId],
    ) -> bool {
        self.prospective_child_traversal(candidate)
            .may_release_held_ring_before_opening(self.prepared.graph(), held_bonds)
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
            .domain_bound(variable)
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
            .domain_bound(variable)
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
    use crate::traversal::{ObservedBondProgress, ObservedTraversalState};

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

        fn domain_bound(&self, _variable: VariableId) -> Option<Domain> {
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

        fn domain_bound(&self, variable: VariableId) -> Option<Domain> {
            self.domain_reads.fetch_add(1, Ordering::Relaxed);
            self.native.domain_bound(variable)
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

    fn offered_step(
        state: &WriterState<NativeSolverState>,
        candidate: StructuralCandidate,
    ) -> WriterState<NativeSolverState> {
        assert!(
            state.derive_candidates().candidates().contains(&candidate),
            "a prescribed structural event must be offered before it is attempted: {candidate:?}"
        );
        attempt(state, candidate)
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

    fn forest(graph: crate::prepared::PreparedGraph) -> PreparedMolecule {
        let source = PreparedMolecule::new(graph);
        let domains = vec![BondRole::role_domain(); source.graph().bond_count()];
        let partitions = vec![EdgeRolePartition::bond_role(); source.graph().bond_count()];
        PreparedMolecule::constraint_assembly(&source, &domains, &partitions)
            .finish_with_topology(crate::model::TraversalTopologyRequirement::Forest)
    }

    // A ring opened at a visited atom must still have a place to write its
    // second occurrence once the other endpoint has been visited.
    fn check_named_ring_endpoints(
        graph: &crate::prepared::PreparedGraph,
        raw: &ObservedTraversalState,
    ) -> Result<(), &'static str> {
        for bond in graph.bond_ids() {
            let edge = graph.bond(bond).unwrap();
            let a_visited = raw.visited_atoms.contains(&edge.a());
            let b_visited = raw.visited_atoms.contains(&edge.b());
            match raw.bond_progress[bond.index()] {
                ObservedBondProgress::Unrepresented if a_visited && b_visited => {
                    return Err("unrepresented bond has two visited endpoints");
                }
                ObservedBondProgress::RingOpen { first_endpoint } => {
                    let other = edge.other(first_endpoint).expect("named ring endpoint belongs to bond");
                    if !raw.visited_atoms.contains(&first_endpoint) {
                        return Err("ring first endpoint is unvisited");
                    }
                    if raw.visited_atoms.contains(&other) {
                        let Some(active) = &raw.active_frame else {
                            return Err("visited ring second endpoint has no active frame");
                        };
                        if active.atom != other || active.branch_started
                            || active.ring_occurrence_count != active.emitted_bonds.len()
                        {
                            return Err("visited ring second endpoint cannot emit its occurrence");
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    #[test]
    fn finite_rooted_forest_prefixes_have_at_most_one_unvisited_tree_attachment() {
        let mut checked = 0_usize;
        let mut attached = 0_usize;
        let mut unattached = 0_usize;
        for atom_count in 2..=4_usize {
            let pairs = (0..atom_count)
                .flat_map(|a| (a + 1..atom_count).map(move |b| (a, b)))
                .collect::<Vec<_>>();
            for graph_mask in 0..(1_usize << pairs.len()) {
                for traversal_mask in 0..(1_usize << pairs.len()) {
                    if traversal_mask & !graph_mask != 0 {
                        continue;
                    }
                    let traversal_edges = pairs.iter().enumerate()
                        .filter(|(index, _)| traversal_mask & (1 << index) != 0)
                        .map(|(_, &pair)| pair)
                        .collect::<Vec<_>>();
                    let mut full_component = (0..atom_count).collect::<Vec<_>>();
                    let mut cyclic = false;
                    for &(a, b) in &traversal_edges {
                        let first = full_component[a];
                        let second = full_component[b];
                        if first == second {
                            cyclic = true;
                            break;
                        }
                        for component in &mut full_component {
                            if *component == second {
                                *component = first;
                            }
                        }
                    }
                    if cyclic {
                        continue;
                    }
                    for visited in 1..(1_usize << atom_count) - 1 {
                        let mut visited_component = (0..atom_count).collect::<Vec<_>>();
                        let mut unvisited_component = (0..atom_count).collect::<Vec<_>>();
                        for &(a, b) in &traversal_edges {
                            if (visited >> a) & 1 != (visited >> b) & 1 {
                                continue;
                            }
                            let components = if visited & (1 << a) != 0 {
                                &mut visited_component
                            } else {
                                &mut unvisited_component
                            };
                            let first = components[a];
                            let second = components[b];
                            for component in components {
                                if *component == second {
                                    *component = first;
                                }
                            }
                        }
                        for roots in 1..(1_usize << atom_count) {
                            if roots & !visited != 0 {
                                continue;
                            }
                            let valid_visited_trees = (0..atom_count)
                                .filter(|&atom| visited & (1 << atom) != 0)
                                .all(|atom| (0..atom_count)
                                    .filter(|&other| visited & (1 << other) != 0
                                        && visited_component[other] == visited_component[atom]
                                        && roots & (1 << other) != 0)
                                    .count() == 1);
                            if !valid_visited_trees {
                                continue;
                            }
                            let rooted_forest = (0..atom_count)
                                .filter(|&atom| roots & (1 << atom) != 0)
                                .all(|atom| (0..atom)
                                    .filter(|&other| roots & (1 << other) != 0
                                        && full_component[other] == full_component[atom])
                                    .count() == 0);
                            if !rooted_forest {
                                continue;
                            }
                            checked += 1;
                            let mut seen_unvisited_components = Vec::new();
                            for atom in 0..atom_count {
                                if visited & (1 << atom) != 0
                                    || seen_unvisited_components.contains(&unvisited_component[atom])
                                {
                                    continue;
                                }
                                seen_unvisited_components.push(unvisited_component[atom]);
                                let attachment_count = traversal_edges.iter()
                                    .filter(|&&(a, b)| {
                                        (visited & (1 << a) != 0
                                            && visited & (1 << b) == 0
                                            && unvisited_component[b] == unvisited_component[atom])
                                            || (visited & (1 << b) != 0
                                                && visited & (1 << a) == 0
                                                && unvisited_component[a] == unvisited_component[atom])
                                    })
                                    .count();
                                assert!(attachment_count <= 1,
                                    "n={atom_count} graph={graph_mask:b} traversal={traversal_mask:b} visited={visited:b} roots={roots:b}");
                                if attachment_count == 0 { unattached += 1; } else { attached += 1; }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!((checked, attached, unattached), (10_900, 8_450, 8_456));
    }

    #[test]
    fn named_ring_endpoint_remains_writable_across_stable_frame_transitions() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let cross = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let child = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let prepared = forest(graph.build());
        let graph = prepared.graph();
        let state = rooted(&prepared, atoms[0]);
        check_named_ring_endpoints(graph, &state.observe_raw().traversal).unwrap();
        let mut prematurely_visited = state.observe_raw().traversal;
        prematurely_visited.visited_atoms.push(atoms[1]);
        assert_eq!(
            check_named_ring_endpoints(graph, &prematurely_visited),
            Err("unrepresented bond has two visited endpoints")
        );
        let state = offered_step(&state, StructuralCandidate::RingOpen {
            incident: incident(&prepared, atoms[0], cross),
        });
        check_named_ring_endpoints(graph, &state.observe_raw().traversal).unwrap();
        let state = offered_step(&state, StructuralCandidate::DotSegmentRoot { atom: atoms[2] });
        let raw = state.observe_raw().traversal;
        check_named_ring_endpoints(graph, &raw).unwrap();

        let entry = incident(&prepared, atoms[2], child);
        let state = offered_step(&state, StructuralCandidate::InlineChild { incident: entry })
            .enter_committed_inline_child(entry);
        let ready_to_close = state.observe_raw().traversal;
        check_named_ring_endpoints(graph, &ready_to_close).unwrap();

        // Both bonds already have valid progress. Each mutation below makes
        // only the named second ring occurrence unavailable.
        let mut no_active = ready_to_close.clone();
        no_active.active_frame = None;
        assert_eq!(
            check_named_ring_endpoints(graph, &no_active),
            Err("visited ring second endpoint has no active frame")
        );
        let mut sealed = ready_to_close.clone();
        sealed.active_frame.as_mut().unwrap().branch_started = true;
        assert_eq!(
            check_named_ring_endpoints(graph, &sealed),
            Err("visited ring second endpoint cannot emit its occurrence")
        );
        let mut past_ring_phase = ready_to_close.clone();
        past_ring_phase.active_frame.as_mut().unwrap().ring_occurrence_count = 1;
        assert_eq!(
            check_named_ring_endpoints(graph, &past_ring_phase),
            Err("visited ring second endpoint cannot emit its occurrence")
        );
        let mut unvisited_first = ready_to_close.clone();
        unvisited_first.visited_atoms.retain(|atom| *atom != atoms[0]);
        assert_eq!(
            check_named_ring_endpoints(graph, &unvisited_first),
            Err("ring first endpoint is unvisited")
        );

        let state = offered_step(&state, StructuralCandidate::RingClose {
            incident: incident(&prepared, atoms[1], cross),
            first_endpoint: atoms[0],
        });
        check_named_ring_endpoints(graph, &state.observe_raw().traversal).unwrap();
    }

    #[test]
    fn forest_parent_retains_a_sibling_after_branching_into_one_chemical_attachment() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 6] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let bonds = [
            graph.add_bond(atoms[0], atoms[1]).unwrap(),
            graph.add_bond(atoms[1], atoms[2]).unwrap(),
            graph.add_bond(atoms[2], atoms[5]).unwrap(),
            graph.add_bond(atoms[5], atoms[4]).unwrap(),
            graph.add_bond(atoms[4], atoms[3]).unwrap(),
            graph.add_bond(atoms[3], atoms[0]).unwrap(),
        ];
        let prepared = forest(graph.build());
        let rooted = rooted(&prepared, atoms[0]);
        let branch = incident(&prepared, atoms[0], bonds[0]);
        let sibling = incident(&prepared, atoms[0], bonds[5]);
        let frontier = rooted.structural_frontier();
        assert_eq!(frontier.branch_children(), &[branch, sibling]);
        assert!(frontier.inline_children().is_empty());
        assert_eq!(frontier.ring_openings(), &[branch, sibling]);

        let committed = offered_step(&rooted, StructuralCandidate::BranchChild { incident: branch });
        assert_eq!(committed.bond_role_domain(bonds[5]), BondRole::Traversal.singleton_domain());
        let entered = committed.enter_committed_branch_child(branch);
        let next = incident(&prepared, atoms[1], bonds[1]);
        let committed = offered_step(&entered, StructuralCandidate::InlineChild { incident: next });
        let entered = committed.enter_committed_inline_child(next);
        let opened = offered_step(&entered, StructuralCandidate::RingOpen {
            incident: incident(&prepared, atoms[2], bonds[2]),
        });
        let observed = opened.observe_raw().traversal;
        assert_eq!(observed.visited_atoms, atoms[..3]);
        assert_eq!(observed.bond_progress[bonds[2].index()], ObservedBondProgress::RingOpen {
            first_endpoint: atoms[2],
        });
        assert_eq!(observed.branch_returns.len(), 1);
        let restored = offered_step(&opened, StructuralCandidate::CloseBranch);
        assert_eq!(restored.active_atom(), Some(atoms[0]));
        assert_eq!(restored.structural_frontier().inline_children(), &[sibling]);
        assert_eq!(restored.traversal.active_attachments().len(), 1);
        assert_eq!(restored.traversal.active_attachments()[0].incidences(), &[sibling]);
        let observed = restored.observe_raw().traversal;
        assert!(observed.branch_returns.is_empty());
        assert_eq!(observed.active_frame.unwrap().atom, atoms[0]);
        assert_eq!(observed.bond_progress[bonds[2].index()], ObservedBondProgress::RingOpen {
            first_endpoint: atoms[2],
        });
        let committed = offered_step(&restored, StructuralCandidate::InlineChild { incident: sibling });
        let entered = committed.enter_committed_inline_child(sibling);
        let next = incident(&prepared, atoms[3], bonds[4]);
        let committed = offered_step(&entered, StructuralCandidate::InlineChild { incident: next });
        let entered = committed.enter_committed_inline_child(next);
        let next = incident(&prepared, atoms[4], bonds[3]);
        let committed = offered_step(&entered, StructuralCandidate::InlineChild { incident: next });
        let entered = committed.enter_committed_inline_child(next);
        let closing = incident(&prepared, atoms[5], bonds[2]);
        let closed = offered_step(&entered, StructuralCandidate::RingClose {
            incident: closing,
            first_endpoint: atoms[2],
        });
        let observed = closed.observe_raw().traversal;
        assert_eq!(observed.visited_atoms, atoms);
        for (bond, from, to) in [
            (bonds[0], atoms[0], atoms[1]),
            (bonds[1], atoms[1], atoms[2]),
            (bonds[3], atoms[4], atoms[5]),
            (bonds[4], atoms[3], atoms[4]),
            (bonds[5], atoms[0], atoms[3]),
        ] {
            assert_eq!(observed.bond_progress[bond.index()], ObservedBondProgress::Traversed {
                from,
                to,
            });
        }
        assert_eq!(observed.bond_progress[bonds[2].index()], ObservedBondProgress::RingClosed {
            first_endpoint: atoms[2],
            second_endpoint: atoms[5],
        });
        assert!(offered_step(&closed, StructuralCandidate::FinishSegment).scope_is_complete());
    }

    #[test]
    fn dot_branch_ring_closes_after_return_at_a_later_atom() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let parent = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let cross = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let prepared = forest(graph.build());
        let state = rooted(&prepared, atoms[0]);
        check_named_ring_endpoints(prepared.graph(), &state.observe_raw().traversal).unwrap();
        let state = offered_step(&state, StructuralCandidate::DotBranchRoot { atom: atoms[2] });
        check_named_ring_endpoints(prepared.graph(), &state.observe_raw().traversal).unwrap();
        let opening = incident(&prepared, atoms[2], cross);
        let state = offered_step(&state, StructuralCandidate::RingOpen { incident: opening });
        check_named_ring_endpoints(prepared.graph(), &state.observe_raw().traversal).unwrap();
        let raw = state.observe_raw().traversal;
        assert_eq!(raw.visited_atoms, vec![atoms[0], atoms[2]]);
        assert_eq!(raw.branch_returns.len(), 1);
        assert_eq!(raw.bond_progress[cross.index()], ObservedBondProgress::RingOpen {
            first_endpoint: atoms[2],
        });
        assert!(state.derive_candidates().candidates().contains(&StructuralCandidate::CloseBranch));
        assert!(!state.derive_candidates().candidates().contains(&StructuralCandidate::FinishSegment));
        let state = offered_step(&state, StructuralCandidate::CloseBranch);
        check_named_ring_endpoints(prepared.graph(), &state.observe_raw().traversal).unwrap();
        let raw = state.observe_raw().traversal;
        assert_eq!(raw.active_frame.unwrap().atom, atoms[0]);
        assert!(raw.branch_returns.is_empty());
        assert!(state.traversal.can_enter_atom(prepared.graph(), atoms[1], Some(parent)));
        assert!(state.structural_frontier().inline_children().contains(&incident(&prepared, atoms[0], parent)));
        assert!(!state.derive_candidates().candidates().iter().any(|candidate| matches!(
            candidate, StructuralCandidate::RingOpen { .. } | StructuralCandidate::RingClose { .. }
        )));
        let entry = incident(&prepared, atoms[0], parent);
        let state = offered_step(&state, StructuralCandidate::InlineChild { incident: entry })
            .enter_committed_inline_child(entry);
        check_named_ring_endpoints(prepared.graph(), &state.observe_raw().traversal).unwrap();
        let closing = incident(&prepared, atoms[1], cross);
        let state = offered_step(&state, StructuralCandidate::RingClose {
            incident: closing,
            first_endpoint: atoms[2],
        });
        check_named_ring_endpoints(prepared.graph(), &state.observe_raw().traversal).unwrap();
        let raw = state.observe_raw().traversal;
        assert_eq!(raw.visited_atoms, atoms);
        assert_eq!(raw.bond_progress[parent.index()], ObservedBondProgress::Traversed {
            from: atoms[0],
            to: atoms[1],
        });
        assert_eq!(raw.bond_progress[cross.index()], ObservedBondProgress::RingClosed {
            first_endpoint: atoms[2],
            second_endpoint: atoms[1],
        });
        assert!(offered_step(&state, StructuralCandidate::FinishSegment).scope_is_complete());
    }

    #[test]
    fn completed_segment_starts_a_new_root_with_a_cross_segment_ring() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let cross = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let child = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let prepared = forest(graph.build());
        let state = rooted(&prepared, atoms[0]);
        let opening = incident(&prepared, atoms[0], cross);
        let state = offered_step(&state, StructuralCandidate::RingOpen { incident: opening });
        assert!(state.derive_candidates().candidates().contains(
            &StructuralCandidate::DotSegmentRoot { atom: atoms[2] }
        ));
        assert!(!state.derive_candidates().candidates().contains(&StructuralCandidate::FinishSegment));
        let raw = state.observe_raw().traversal;
        assert_eq!(raw.visited_atoms, vec![atoms[0]]);
        assert_eq!(raw.bond_progress[cross.index()], ObservedBondProgress::RingOpen {
            first_endpoint: atoms[0],
        });
        let state = offered_step(&state, StructuralCandidate::DotSegmentRoot { atom: atoms[2] });
        let raw = state.observe_raw().traversal;
        assert_eq!(raw.visited_atoms, vec![atoms[0], atoms[2]]);
        let frame = raw.active_frame.unwrap();
        assert_eq!(frame.atom, atoms[2]);
        assert_eq!(frame.entry_bond, None);
        assert!(raw.branch_returns.is_empty());
        for incident in prepared.graph().neighbors(atoms[0]).unwrap() {
            if !raw.visited_atoms.contains(&incident.atom()) {
                assert_ne!(raw.bond_progress[incident.bond().index()], ObservedBondProgress::Unrepresented);
            }
        }
        assert!(state.traversal.can_enter_atom(prepared.graph(), atoms[1], Some(child)));
        let entry = incident(&prepared, atoms[2], child);
        let state = offered_step(&state, StructuralCandidate::InlineChild { incident: entry })
            .enter_committed_inline_child(entry);
        let closing = incident(&prepared, atoms[1], cross);
        let state = offered_step(&state, StructuralCandidate::RingClose {
            incident: closing,
            first_endpoint: atoms[0],
        });
        let raw = state.observe_raw().traversal;
        assert_eq!(raw.visited_atoms, atoms);
        assert_eq!(raw.bond_progress[child.index()], ObservedBondProgress::Traversed {
            from: atoms[2],
            to: atoms[1],
        });
        assert_eq!(raw.bond_progress[cross.index()], ObservedBondProgress::RingClosed {
            first_endpoint: atoms[0],
            second_endpoint: atoms[1],
        });
        assert!(state.derive_candidates().candidates().contains(&StructuralCandidate::FinishSegment));
        assert!(offered_step(&state, StructuralCandidate::FinishSegment).scope_is_complete());
    }

    #[test]
    fn forest_branch_start_rejects_a_remaining_parent_ring_role() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let left = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let right = graph.add_bond(atoms[0], atoms[2]).unwrap();
        graph.add_bond(atoms[1], atoms[2]).unwrap();
        let prepared = forest(graph.build());
        let rooted = rooted(&prepared, atoms[0]);
        let constraints = rooted.constraints
            .restricted(&[role_restriction(&prepared, right, BondRole::Ring)])
            .unwrap().unwrap_consistent();
        let committed_ring = WriterState { constraints, ..rooted };
        assert!(matches!(
            committed_ring.attempt_candidate(StructuralCandidate::BranchChild {
                incident: incident(&prepared, atoms[0], left),
            }).unwrap(),
            Consistency::Contradiction
        ));
    }

    #[test]
    fn forest_nested_dot_branch_cannot_reserve_a_second_entry_to_one_atom() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 4] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let root_edge = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let leaf_edge = graph.add_bond(atoms[1], atoms[2]).unwrap();
        graph.add_bond(atoms[1], atoms[3]).unwrap();
        let prepared = forest(graph.build());
        let rooted = rooted(&prepared, atoms[0]);
        let first = attempt(&rooted, StructuralCandidate::DotBranchRoot { atom: atoms[2] });
        assert_eq!(first.bond_role_domain(root_edge), BondRole::Traversal.singleton_domain());
        assert!(first.structural_frontier().ring_openings().contains(
            &incident(&prepared, atoms[2], leaf_edge)
        ));
        assert!(!first.structural_frontier().dot_branch_roots().contains(&atoms[3]));
        assert!(matches!(
            first.attempt_candidate(StructuralCandidate::DotBranchRoot { atom: atoms[3] }).unwrap(),
            Consistency::Contradiction
        ));
    }

    #[test]
    fn forest_rejects_two_emitted_roots_in_one_forced_traversal_component() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 4] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let bonds = [
            graph.add_bond(atoms[0], atoms[1]).unwrap(),
            graph.add_bond(atoms[1], atoms[2]).unwrap(),
            graph.add_bond(atoms[2], atoms[3]).unwrap(),
        ];
        let prepared = forest(graph.build());
        let first = rooted(&prepared, atoms[0]);
        let second = attempt(&first, StructuralCandidate::DotBranchRoot { atom: atoms[3] });
        let restrictions = bonds.map(|bond| (bond, BondRole::Traversal.singleton_domain()));
        assert!(matches!(
            second.restricted_bond_domains_for_test(&restrictions).unwrap(),
            Consistency::Contradiction
        ));
    }

    #[test]
    fn forest_root_selection_and_traversal_edges_prune_each_other() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let left = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let right = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let prepared = forest(graph.build());
        let rooted = rooted(&prepared, atoms[0]);
        assert!(rooted.semantic_domain(prepared.root_variable(atoms[2]).unwrap()).contains(1));
        let left_traversal = rooted.restricted_bond_domains_for_test(&[
            (left, BondRole::Traversal.singleton_domain()),
        ]).unwrap().unwrap_consistent();
        let two_roots = attempt(&left_traversal, StructuralCandidate::DotBranchRoot { atom: atoms[2] });
        assert_eq!(two_roots.bond_role_domain(right), BondRole::Ring.singleton_domain());

        let traversal = rooted.restricted_bond_domains_for_test(&[
            (left, BondRole::Traversal.singleton_domain()),
            (right, BondRole::Traversal.singleton_domain()),
        ]).unwrap().unwrap_consistent();
        assert_eq!(
            traversal.semantic_domain(prepared.root_variable(atoms[2]).unwrap()),
            Domain::singleton(0).unwrap(),
        );
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
            &[StructuralCandidate::FinishSegment]
        );

        let finished = attempt(&rooted, StructuralCandidate::FinishSegment);
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
            Some(PathCompletion::FinishSegment)
        );
        let finished = attempt(&closed, StructuralCandidate::FinishSegment);
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
            component_order: ComponentOrder::AnyOrder,
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
        let between = graph.add_bond(atoms[1], atoms[2]).unwrap();
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
        let branch_incident = incident(&prepared, atoms[0], substituent);
        let branch = attempt(
            &opened,
            StructuralCandidate::BranchChild {
                incident: branch_incident,
            },
        )
        .enter_committed_branch_child(branch_incident);
        assert_eq!(
            branch.structural_frontier().path_completion(),
            Some(PathCompletion::CloseBranch)
        );
        let restored = attempt(&branch, StructuralCandidate::CloseBranch);
        let remaining = incident(&prepared, atoms[0], right);
        assert!(restored.structural_frontier().ring_openings().is_empty());
        assert!(restored.structural_frontier().ring_closures().is_empty());
        assert_eq!(
            restored.structural_frontier().inline_children(),
            &[remaining]
        );

        let walked = attempt(
            &restored,
            StructuralCandidate::InlineChild {
                incident: remaining,
            },
        )
        .enter_committed_inline_child(remaining);
        let next = incident(&prepared, atoms[2], between);
        let walked = attempt(&walked, StructuralCandidate::InlineChild { incident: next })
            .enter_committed_inline_child(next);
        let closure = incident(&prepared, atoms[1], left);
        assert_eq!(walked.structural_frontier().ring_closures(), &[closure]);
        let closed = attempt(
            &walked,
            StructuralCandidate::RingClose {
                incident: closure,
                first_endpoint: atoms[0],
            },
        );
        assert!(closed.graph_is_complete());
        assert_eq!(
            closed.structural_frontier().path_completion(),
            Some(PathCompletion::FinishSegment)
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
