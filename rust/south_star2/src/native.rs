//! Native finite-domain solving for South Star 2.
//!
//! Factor propagation is seeded only from changed variables. Binary relation
//! components retain exact finite-domain filtering; the traversal-topology factor has
//! its own exact graphic-matroid projection and never enumerates spanning trees.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::domain::Domain;
use crate::ids::{FactorId, VariableId};
use crate::event_schedule::EventScheduleFactor;
use crate::model::{
    BinaryRelationFactor, ConstraintModel, DirectionalRingPlacementFactor, EdgeRolePartition,
    FactorActivation, FactorDefinition, TetrahedralLayoutFactor, TraversalTopologyFactor,
};
use crate::persistent::PagedStore;

#[derive(Clone, Debug)]
pub(crate) struct NativeSolverState {
    model: Arc<ConstraintModel>,
    exact_plan: Arc<NativeExactPlan>,
    domains: PagedStore<Domain>,
    active_factors: PagedStore<bool>,
    activated_factors_by_variable: PagedStore<Arc<[FactorId]>>,
    /// Feasible writer transitions may retain unsupported future values.
    projected_exact: bool,
    #[cfg(test)]
    mixed_search_branches: Arc<AtomicUsize>,
    #[cfg(test)]
    binary_exact_runs: Arc<AtomicUsize>,
    #[cfg(test)]
    mixed_exact_runs: Arc<AtomicUsize>,
    #[cfg(test)]
    tetrahedral_factor_revisions: Arc<AtomicUsize>,
    #[cfg(test)]
    directional_ring_factor_revisions: Arc<AtomicUsize>,
    #[cfg(test)]
    spanning_first_edge_visits: Arc<AtomicUsize>,
    #[cfg(test)]
    spanning_atom_visits: Arc<AtomicUsize>,
    #[cfg(test)]
    spanning_second_edge_visits: Arc<AtomicUsize>,
    #[cfg(test)]
    potential_layout_checks: Arc<AtomicUsize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum NativeSolverError {
    UnknownVariable(VariableId),
    UnknownFactor(FactorId),
    Contradiction,
}

impl fmt::Display for NativeSolverError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownVariable(variable) => {
                write!(formatter, "unknown constraint variable {variable:?}")
            }
            Self::UnknownFactor(factor) => {
                write!(formatter, "unknown constraint factor {factor:?}")
            }
            Self::Contradiction => formatter.write_str("constraint state is contradictory"),
        }
    }
}

impl std::error::Error for NativeSolverError {}

impl NativeSolverState {
    pub(crate) fn initial(model: Arc<ConstraintModel>) -> Result<Self, NativeSolverError> {
        let domains = PagedStore::from_values(model.initial_domains());
        let exact_plan = Arc::new(NativeExactPlan::compile(&model));
        let variable_count = model.variable_count();
        let initial_factor_ids = model.initial_factor_ids().to_vec();
        let active_factors = PagedStore::from_values((0..model.factor_count()).map(|index| {
            model.factor_activation(factor_id_from_index(index)) == Some(FactorActivation::Always)
        }));
        let mut state = Self {
            model,
            exact_plan,
            domains,
            active_factors,
            activated_factors_by_variable: PagedStore::filled(
                variable_count,
                Arc::<[FactorId]>::from([]),
            ),
            projected_exact: true,
            #[cfg(test)]
            mixed_search_branches: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            binary_exact_runs: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            mixed_exact_runs: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            tetrahedral_factor_revisions: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            directional_ring_factor_revisions: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            spanning_first_edge_visits: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            spanning_atom_visits: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            spanning_second_edge_visits: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            potential_layout_checks: Arc::new(AtomicUsize::new(0)),
        };
        state.enforce_consistency(
            initial_factor_ids,
            (0..variable_count).map(variable_id_from_index),
        )?;
        Ok(state)
    }

    pub(crate) fn domain_bound(&self, variable: VariableId) -> Option<Domain> {
        self.domains.get(variable.index()).copied()
    }

    pub(crate) fn factor_is_active(&self, factor: FactorId) -> Option<bool> {
        self.active_factors.get(factor.index()).copied()
    }

    #[cfg(test)]
    pub(crate) fn reset_mixed_search_branch_count(&self) {
        self.mixed_search_branches.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn mixed_search_branch_count(&self) -> usize {
        self.mixed_search_branches.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn reset_exact_run_counts(&self) {
        self.binary_exact_runs.store(0, Ordering::Relaxed);
        self.mixed_exact_runs.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn exact_run_counts(&self) -> (usize, usize) {
        (
            self.binary_exact_runs.load(Ordering::Relaxed),
            self.mixed_exact_runs.load(Ordering::Relaxed),
        )
    }

    #[cfg(test)]
    pub(crate) fn reset_tetrahedral_factor_revision_count(&self) {
        self.tetrahedral_factor_revisions
            .store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn tetrahedral_factor_revision_count(&self) -> usize {
        self.tetrahedral_factor_revisions.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn reset_directional_ring_factor_revision_count(&self) {
        self.directional_ring_factor_revisions
            .store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn directional_ring_factor_revision_count(&self) -> usize {
        self.directional_ring_factor_revisions
            .load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn reset_projector_work_counts(&self) {
        self.spanning_first_edge_visits.store(0, Ordering::Relaxed);
        self.spanning_atom_visits.store(0, Ordering::Relaxed);
        self.spanning_second_edge_visits.store(0, Ordering::Relaxed);
        self.potential_layout_checks.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn projector_work_counts(&self) -> (usize, usize, usize, usize) {
        (
            self.spanning_first_edge_visits.load(Ordering::Relaxed),
            self.spanning_atom_visits.load(Ordering::Relaxed),
            self.spanning_second_edge_visits.load(Ordering::Relaxed),
            self.potential_layout_checks.load(Ordering::Relaxed),
        )
    }

    #[cfg(test)]
    pub(crate) fn solver_copy_counts(&self) -> ((usize, usize), (usize, usize), (usize, usize)) {
        (
            self.domains.copy_counts(),
            self.active_factors.copy_counts(),
            self.activated_factors_by_variable.copy_counts(),
        )
    }

    #[cfg(test)]
    pub(crate) fn reset_solver_copy_counts(&self) {
        self.domains.reset_copy_counts();
        self.active_factors.reset_copy_counts();
        self.activated_factors_by_variable.reset_copy_counts();
    }

    /// Return one atomically restricted and propagated successor.
    ///
    /// Repeated restrictions for the same variable are intersected before any
    /// candidate state is created. The source state is never mutated.
    pub(crate) fn with_restrictions(
        &self,
        restrictions: impl IntoIterator<Item = (VariableId, Domain)>,
    ) -> Result<Self, NativeSolverError> {
        self.with_transition(restrictions, std::iter::empty())
    }

    pub(crate) fn with_transition(
        &self,
        restrictions: impl IntoIterator<Item = (VariableId, Domain)>,
        activate: impl IntoIterator<Item = FactorId>,
    ) -> Result<Self, NativeSolverError> {
        self.with_transition_mode(restrictions, activate, false)
    }

    pub(crate) fn with_feasible_transition(
        &self,
        restrictions: impl IntoIterator<Item = (VariableId, Domain)>,
        activate: impl IntoIterator<Item = FactorId>,
    ) -> Result<Self, NativeSolverError> {
        self.with_transition_mode(restrictions, activate, true)
    }

    fn with_transition_mode(
        &self,
        restrictions: impl IntoIterator<Item = (VariableId, Domain)>,
        activate: impl IntoIterator<Item = FactorId>,
        feasibility_only: bool,
    ) -> Result<Self, NativeSolverError> {
        let mut domains_by_variable = BTreeMap::new();
        let mut contradictory = false;
        for (variable, allowed) in restrictions {
            let current = self
                .domain_bound(variable)
                .ok_or(NativeSolverError::UnknownVariable(variable))?;
            let entry = domains_by_variable.entry(variable).or_insert(current);
            *entry = (*entry).intersect(allowed);
            contradictory |= entry.is_empty();
        }
        let mut newly_activated = BTreeSet::new();
        for factor in activate {
            let active = self
                .factor_is_active(factor)
                .ok_or(NativeSolverError::UnknownFactor(factor))?;
            if !active {
                newly_activated.insert(factor);
            }
        }
        if contradictory {
            return Err(NativeSolverError::Contradiction);
        }

        let mut successor = self.clone();
        let mut changed_variables = Vec::new();
        let mut seed_factors = BTreeSet::new();

        for (variable, restricted) in domains_by_variable {
            let current = successor.domains[variable.index()];
            if restricted == current {
                continue;
            }
            successor.domains[variable.index()] = restricted;
            changed_variables.push(variable);
            seed_factors.extend(successor.active_factors_for_variable(variable));
        }

        for factor in newly_activated {
            successor.active_factors[factor.index()] = true;
            let variables = successor
                .model
                .factor(factor)
                .expect("validated activation factor must exist")
                .variables()
                .to_vec();
            for variable in variables.iter().copied() {
                let mut factors =
                    successor.activated_factors_by_variable[variable.index()].to_vec();
                factors.push(factor);
                successor.activated_factors_by_variable[variable.index()] = Arc::from(factors);
            }
            seed_factors.insert(factor);
            changed_variables.extend(variables);
        }

        if !feasibility_only && !successor.projected_exact {
            changed_variables.extend(successor.exact_plan.components.iter()
                .flat_map(|component| component.projected_variables().iter().copied()));
        }
        if changed_variables.is_empty() && seed_factors.is_empty() {
            return Ok(successor);
        }

        if feasibility_only && (0..successor.model.factor_count()).any(|index|
            successor.active_factors[index] && matches!(successor.model.factor(factor_id_from_index(index)),
                Some(FactorDefinition::EventSchedule(_)))) {
            successor.enforce_feasibility(seed_factors, changed_variables)?;
            successor.projected_exact = false;
        } else {
            successor.enforce_consistency(seed_factors, changed_variables)?;
            successor.projected_exact = true;
        }
        Ok(successor)
    }

    fn active_factors_for_variable(&self, variable: VariableId) -> Vec<FactorId> {
        let mut factors = self
            .model
            .initial_factors_for_variable(variable)
            .expect("known variable must have an initial adjacency row")
            .to_vec();
        factors.extend(
            self.activated_factors_by_variable[variable.index()]
                .iter()
                .copied(),
        );
        factors
    }

    fn enforce_consistency(
        &mut self,
        factor_seeds: impl IntoIterator<Item = FactorId>,
        binary_seeds: impl IntoIterator<Item = VariableId>,
    ) -> Result<(), NativeSolverError> {
        let mut exact_seeds = binary_seeds.into_iter().collect::<BTreeSet<_>>();
        exact_seeds.extend(self.propagate(factor_seeds)?);

        while !exact_seeds.is_empty() {
            let seeds = exact_seeds.iter().copied().collect::<Vec<_>>();
            let exact_reductions = self.complete_filter_exact_components(seeds)?;
            if exact_reductions.is_empty() {
                break;
            }

            let mut seed_factors = BTreeSet::new();
            for variable in &exact_reductions {
                seed_factors.extend(self.active_factors_for_variable(*variable));
            }

            let propagated_reductions = self.propagate(seed_factors)?;
            exact_seeds = exact_reductions
                .into_iter()
                .chain(propagated_reductions)
                .collect();
        }

        Ok(())
    }

    fn enforce_feasibility(
        &mut self,
        factor_seeds: impl IntoIterator<Item = FactorId>,
        component_seeds: impl IntoIterator<Item = VariableId>,
    ) -> Result<(), NativeSolverError> {
        let mut seeds = component_seeds.into_iter().collect::<BTreeSet<_>>();
        seeds.extend(self.propagate(factor_seeds)?);
        self.check_exact_components_feasible(seeds)
    }

    fn propagate(
        &mut self,
        seeds: impl IntoIterator<Item = FactorId>,
    ) -> Result<BTreeSet<VariableId>, NativeSolverError> {
        let mut queue = VecDeque::new();
        let mut queued = BTreeSet::new();
        let mut all_reductions = BTreeSet::new();

        for factor in seeds {
            assert_eq!(
                self.factor_is_active(factor),
                Some(true),
                "only active factors may enter the propagation queue"
            );
            enqueue_factor(factor, &mut queue, &mut queued);
        }

        while let Some(factor_id) = queue.pop_front() {
            #[cfg(test)]
            crate::work_diagnostic::record(|counts| counts.factor_revisions += 1);
            queued.remove(&factor_id);

            let factor = self
                .model
                .factor(factor_id)
                .expect("factor adjacency must reference a prepared factor");
            let reductions = match factor {
                FactorDefinition::AromaticMatching(matching) => {
                    crate::aromatic_matching::revise_aromatic_matching(matching, &mut self.domains)?
                }
                FactorDefinition::BinaryRelation(relation) => {
                    revise_binary_relation(relation, &mut self.domains)?
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>()
                }
                FactorDefinition::DirectionalRingPlacement(placement) => {
                    #[cfg(test)]
                    self.directional_ring_factor_revisions
                        .fetch_add(1, Ordering::Relaxed);
                    revise_directional_ring_placement(placement, &mut self.domains)?
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>()
                }
                FactorDefinition::TraversalTopology(spanning_tree) => {
                    #[cfg(test)]
                    let work = (
                        self.spanning_first_edge_visits.as_ref(),
                        self.spanning_atom_visits.as_ref(),
                        self.spanning_second_edge_visits.as_ref(),
                    );
                    #[cfg(not(test))]
                    let work = ();
                    revise_traversal_topology(spanning_tree, &mut self.domains, work)?
                }
                FactorDefinition::TetrahedralLayout(layout) => {
                    #[cfg(test)]
                    self.tetrahedral_factor_revisions
                        .fetch_add(1, Ordering::Relaxed);
                    revise_tetrahedral_layout(layout, &mut self.domains)?
                }
                FactorDefinition::RingCountAtMost(count) => {
                    revise_ring_count_at_most(count, &mut self.domains)?
                }
                FactorDefinition::OrderedRingPhase(phase) => {
                    revise_ordered_ring_phase(phase, &mut self.domains)?
                }
                FactorDefinition::OrdinaryOrder(order) => {
                    revise_ordinary_order(order, &mut self.domains)?
                }
                FactorDefinition::EventSchedule(schedule) => {
                    revise_event_schedule(schedule, &mut self.domains)?
                }
            };

            for variable in reductions {
                all_reductions.insert(variable);
                for neighbour in self.active_factors_for_variable(variable) {
                    enqueue_factor(neighbour, &mut queue, &mut queued);
                }
            }
        }

        Ok(all_reductions)
    }

    fn complete_filter_exact_components(
        &mut self,
        seeds: impl IntoIterator<Item = VariableId>,
    ) -> Result<BTreeSet<VariableId>, NativeSolverError> {
        let component_ids = seeds
            .into_iter()
            .filter_map(|variable| self.exact_plan.component_for(variable))
            .collect::<BTreeSet<_>>();
        let plan = Arc::clone(&self.exact_plan);
        let mut reductions = BTreeSet::new();

        for component_id in component_ids {
            match &plan.components[component_id] {
                ExactComponent::Binary(component) => {
                    #[cfg(test)]
                    self.binary_exact_runs.fetch_add(1, Ordering::Relaxed);
                    reductions.extend(self.complete_filter_binary_component(component)?);
                }
                ExactComponent::Mixed(component) => {
                    reductions.extend(self.complete_filter_mixed_component(component, false)?);
                }
            }
        }

        Ok(reductions)
    }

    fn check_exact_components_feasible(
        &mut self,
        seeds: impl IntoIterator<Item = VariableId>,
    ) -> Result<(), NativeSolverError> {
        let component_ids = seeds.into_iter()
            .filter_map(|variable| self.exact_plan.component_for(variable))
            .collect::<BTreeSet<_>>();
        let plan = Arc::clone(&self.exact_plan);
        for component_id in component_ids {
            match &plan.components[component_id] {
                ExactComponent::Mixed(component) => {
                    self.complete_filter_mixed_component(component, true)?;
                }
                ExactComponent::Binary(component) => {
                    self.complete_filter_binary_component(component)?;
                }
            }
        }
        Ok(())
    }

    fn complete_filter_binary_component(
        &mut self,
        component: &BinaryExactComponent,
    ) -> Result<Vec<VariableId>, NativeSolverError> {
        if !component
            .variables
            .iter()
            .any(|variable| self.domains[variable.index()].len() > 1)
        {
            return Ok(Vec::new());
        }

        let domains = component
            .variables
            .iter()
            .map(|variable| self.domains[variable.index()])
            .collect::<Vec<_>>();
        let supported = component
            .exact_supported_domains(&self.model, domains)
            .ok_or(NativeSolverError::Contradiction)?;

        let mut reductions = Vec::new();
        for (variable, supported_domain) in component.variables.iter().copied().zip(supported) {
            let current = self.domains[variable.index()];
            debug_assert!(supported_domain.is_subset_of(current));
            if supported_domain != current {
                self.domains[variable.index()] = supported_domain;
                reductions.push(variable);
            }
        }
        Ok(reductions)
    }

    fn complete_filter_mixed_component(
        &mut self,
        component: &MixedExactComponent,
        first_solution_only: bool,
    ) -> Result<Vec<VariableId>, NativeSolverError> {
        let active_layouts = component
            .potential_layout_factors
            .iter()
            .copied()
            .filter(|factor| {
                #[cfg(test)]
                self.potential_layout_checks.fetch_add(1, Ordering::Relaxed);
                self.factor_is_active(*factor) == Some(true)
            })
            .collect::<Vec<_>>();
        let active_resources = component.potential_resource_factors.iter().copied()
            .filter(|factor| self.factor_is_active(*factor) == Some(true))
            .collect::<Vec<_>>();
        if component.always_core_variables.is_empty() && active_layouts.is_empty()
            && active_resources.is_empty() {
            return Ok(Vec::new());
        }
        let mut core_variables = component.always_core_variables.to_vec();
        let mut active_layout_core_variables = BTreeSet::new();
        let mut active_layout_cores_overlap = false;
        for factor in &active_layouts {
            let FactorDefinition::TetrahedralLayout(layout) = self
                .model
                .factor(*factor)
                .expect("active layout factor must exist")
            else {
                unreachable!("potential layout IDs must retain their factor kind");
            };
            for variable in [layout.order_variable(), layout.role_pattern_variable()] {
                active_layout_cores_overlap |= !active_layout_core_variables.insert(variable);
                core_variables.push(variable);
            }
        }
        for factor in &active_resources {
            match self.model.factor(*factor).expect("active resource factor must exist") {
                FactorDefinition::RingCountAtMost(count) => {
                    core_variables.extend(count.budget_bits().iter().copied());
                    core_variables.extend(count.bonds().iter().map(|bond| bond.decision_variable()));
                }
                FactorDefinition::OrderedRingPhase(phase) => {
                    core_variables.extend(phase.free_bits().iter().copied());
                    core_variables.extend([phase.waiting_mask_variable(), phase.order_variable()]);
                    core_variables.extend(phase.bonds().iter().map(|(_, edge)| edge.decision_variable()));
                }
                FactorDefinition::EventSchedule(schedule) => {
                    core_variables.extend(schedule.variables().iter().copied());
                }
                FactorDefinition::OrdinaryOrder(order) => {
                    core_variables.extend(order.variables().iter().copied());
                }
                _ => unreachable!("potential resource IDs must retain their factor kind"),
            }
        }
        core_variables.sort_unstable();
        core_variables.dedup();
        let structural_role_is_ambiguous = component.structural_variables.iter().any(|entry| {
            role_membership(self.domains[entry.variable.index()], entry.role_partition)
                == (true, true)
        });
        if component.always_core_variables.is_empty()
            && !active_layout_cores_overlap
            && !structural_role_is_ambiguous
            && active_resources.is_empty()
        {
            return Ok(Vec::new());
        }

        #[cfg(test)]
        self.mixed_exact_runs.fetch_add(1, Ordering::Relaxed);

        if !core_variables
            .iter()
            .any(|variable| self.domains[variable.index()].len() > 1)
        {
            return Ok(Vec::new());
        }

        let mut projected_variables = component
            .always_core_variables
            .iter()
            .copied()
            .chain(
                component
                    .structural_variables
                    .iter()
                    .map(|entry| entry.variable),
            )
            .collect::<Vec<_>>();
        for factor in active_layouts {
            let FactorDefinition::TetrahedralLayout(layout) = self
                .model
                .factor(factor)
                .expect("active layout factor must exist")
            else {
                unreachable!("potential layout IDs must retain their factor kind");
            };
            projected_variables.extend([layout.order_variable(), layout.role_pattern_variable()]);
        }
        for factor in &active_resources {
            match self.model.factor(*factor).expect("active resource factor must exist") {
                FactorDefinition::RingCountAtMost(count) => projected_variables.extend(count.budget_bits().iter().copied()),
                FactorDefinition::OrderedRingPhase(phase) => {
                    projected_variables.extend(phase.free_bits().iter().copied());
                    projected_variables.extend([phase.waiting_mask_variable(), phase.order_variable()]);
                }
                FactorDefinition::EventSchedule(schedule) => {
                    if !first_solution_only {
                        projected_variables.extend(schedule.variables().iter().copied());
                    }
                }
                FactorDefinition::OrdinaryOrder(order) => {
                    projected_variables.extend(order.variables().iter().copied());
                }
                _ => unreachable!("potential resource IDs must retain their factor kind"),
            }
        }
        projected_variables.sort_unstable();
        projected_variables.dedup();

        let target = projected_variables
            .iter()
            .map(|variable| self.domains[variable.index()])
            .collect::<Vec<_>>();
        let mut supported = vec![Domain::empty(); projected_variables.len()];
        enum SearchEntry {
            Candidate(NativeSolverState),
            Pending {
                parent: Arc<NativeSolverState>,
                restrictions: Vec<(VariableId, Domain)>,
            },
        }
        let mut stack = vec![SearchEntry::Candidate(self.clone())];
        let mut found_solution = false;
        let event_schedule = active_resources.iter().find_map(|factor|
            match self.model.factor(*factor) {
                Some(FactorDefinition::EventSchedule(schedule)) => Some(schedule),
                _ => None,
            });

        #[cfg(test)]
        let mut first_node = true;
        while let Some(entry) = stack.pop() {
            let candidate = match entry {
                SearchEntry::Candidate(candidate) => candidate,
                SearchEntry::Pending { parent, restrictions } => {
                    #[cfg(test)]
                    crate::work_diagnostic::record(|counts| counts.sibling_proposals += 1);
                    match parent.restricted_many_and_propagated(&restrictions) {
                        Ok(child) => {
                            #[cfg(test)]
                            crate::work_diagnostic::record(|counts| counts.sibling_propagated += 1);
                            child
                        }
                        Err(NativeSolverError::Contradiction) => continue,
                        Err(failure) => return Err(failure),
                    }
                }
            };
            #[cfg(test)]
            crate::work_diagnostic::record(|counts| {
                counts.search_nodes += 1;
                if !first_node { counts.children_explored += 1; }
            });
            #[cfg(test)]
            { first_node = false; }
            if event_schedule.is_some() {
                if let Some(edge) = component.structural_variables.iter().find(|edge| {
                    let domain = candidate.domains[edge.variable.index()];
                    !domain.intersect(edge.role_partition.traversal_values()).is_empty()
                        && !domain.intersect(edge.role_partition.ring_values()).is_empty()
                }) {
                    #[cfg(test)]
                    candidate.mixed_search_branches.fetch_add(1, Ordering::Relaxed);
                    #[cfg(test)]
                    crate::work_diagnostic::record(|counts| counts.grouped_branches += 1);
                    let parent = Arc::new(candidate);
                    for role in [edge.role_partition.ring_values(), edge.role_partition.traversal_values()] {
                        #[cfg(test)]
                        crate::work_diagnostic::record(|counts| counts.sibling_offered += 1);
                        stack.push(SearchEntry::Pending {
                            parent: Arc::clone(&parent),
                            restrictions: vec![(edge.variable, role)],
                        });
                    }
                    continue;
                }
            }
            if let Some(branches) = event_schedule.and_then(|schedule|
                event_rank_branches(schedule, &candidate.domains)) {
                #[cfg(test)]
                candidate.mixed_search_branches.fetch_add(1, Ordering::Relaxed);
                #[cfg(test)]
                crate::work_diagnostic::record(|counts| counts.grouped_branches += 1);
                let parent = Arc::new(candidate);
                for restrictions in branches.into_iter().rev() {
                    #[cfg(test)]
                    crate::work_diagnostic::record(|counts| counts.sibling_offered += 1);
                    debug_assert!(restrictions.iter().any(|(variable, allowed)|
                        parent.domains[variable.index()].intersect(*allowed)
                            != parent.domains[variable.index()]),
                        "event rank branch must narrow its parent");
                    stack.push(SearchEntry::Pending {
                        parent: Arc::clone(&parent),
                        restrictions,
                    });
                }
                continue;
            }
            let Some(variable) = core_variables
                .iter()
                .copied()
                .filter(|variable| candidate.domains[variable.index()].len() > 1)
                .min_by_key(|variable| candidate.domains[variable.index()].len())
            else {
                found_solution = true;
                if first_solution_only { break; }
                for (accumulator, variable) in supported
                    .iter_mut()
                    .zip(projected_variables.iter().copied())
                {
                    *accumulator = accumulator.union(candidate.domains[variable.index()]);
                }
                if supported == target {
                    break;
                }
                continue;
            };

            #[cfg(test)]
            candidate.mixed_search_branches.fetch_add(1, Ordering::Relaxed);
            #[cfg(test)]
            crate::work_diagnostic::record(|counts| counts.scalar_branches += 1);

            let mut values = candidate.domains[variable.index()]
                .iter()
                .collect::<Vec<_>>();
            values.reverse();
            let parent = Arc::new(candidate);
            for value in values {
                #[cfg(test)]
                crate::work_diagnostic::record(|counts| counts.sibling_offered += 1);
                stack.push(SearchEntry::Pending {
                    parent: Arc::clone(&parent),
                    restrictions: vec![(variable, Domain::from_bits(1_u64 << value))],
                });
            }
        }

        if !found_solution {
            return Err(NativeSolverError::Contradiction);
        }

        if first_solution_only { return Ok(Vec::new()); }

        let mut reductions = Vec::new();
        for (variable, supported_domain) in projected_variables.iter().copied().zip(supported) {
            let current = self.domains[variable.index()];
            debug_assert!(supported_domain.is_subset_of(current));
            if supported_domain != current {
                self.domains[variable.index()] = supported_domain;
                reductions.push(variable);
            }
        }
        Ok(reductions)
    }

    fn restricted_many_and_propagated(
        &self,
        restrictions: &[(VariableId, Domain)],
    ) -> Result<Self, NativeSolverError> {
        let mut successor = self.clone();
        let mut seed_factors = BTreeSet::new();
        for &(variable, allowed) in restrictions {
            let narrowed = successor.domains[variable.index()].intersect(allowed);
            if narrowed.is_empty() { return Err(NativeSolverError::Contradiction); }
            if narrowed != successor.domains[variable.index()] {
                successor.domains[variable.index()] = narrowed;
                seed_factors.extend(successor.active_factors_for_variable(variable));
            }
        }
        successor.propagate(seed_factors)?;
        Ok(successor)
    }
}

fn event_rank_branches(
    schedule: &EventScheduleFactor,
    domains: &PagedStore<Domain>,
) -> Option<Vec<Vec<(VariableId, Domain)>>> {
    let count = schedule.visits.len();
    for rank in 0..count {
        let mut candidate_visits = schedule.visits.iter().filter(|visit|
            visit.rank_bits.iter().enumerate().all(|(bit, variable)|
                domains[variable.index()].contains(((rank >> bit) & 1) as u8)))
            .collect::<Vec<_>>();
        if let Some(fixed) = candidate_visits.iter().find(|visit|
            visit.rank_bits.iter().all(|variable| domains[variable.index()].is_singleton())) {
            candidate_visits = vec![*fixed];
        }
        if candidate_visits.len() == 1 {
            let visit = candidate_visits[0];
            if visit.rank_bits.iter().all(|variable| domains[variable.index()].is_singleton())
                && domains[visit.kind.index()].is_singleton()
                && visit.end_bits.iter().all(|variable| domains[variable.index()].is_singleton()) {
                continue;
            }
        }
        let mut branches = Vec::new();
        for visit in candidate_visits {
            for kind in domains[visit.kind.index()].iter() {
                let branch = kind == 1 || kind == 3;
                let ends = if branch { rank..count } else { 0..1 };
                for end in ends {
                    if !visit.end_bits.iter().enumerate().all(|(bit, variable)|
                        domains[variable.index()].contains(u8::from(branch && bit <= end))) {
                        continue;
                    }
                    let restrictions = visit.rank_bits.iter().enumerate().map(|(bit, variable)|
                        (*variable, Domain::singleton(((rank >> bit) & 1) as u8).unwrap()))
                        .chain(std::iter::once((visit.kind, Domain::singleton(kind).unwrap())))
                        .chain(visit.end_bits.iter().enumerate().map(|(bit, variable)|
                            (*variable, Domain::singleton(u8::from(branch && bit <= end)).unwrap())))
                        .collect::<Vec<_>>();
                    branches.push(restrictions);
                }
            }
        }
        return Some(branches);
    }
    None
}

#[derive(Debug)]
struct NativeExactPlan {
    components: Box<[ExactComponent]>,
    component_by_variable: Box<[Option<usize>]>,
}

impl NativeExactPlan {
    fn compile(model: &ConstraintModel) -> Self {
        let variable_count = model.variable_count();
        let factor_count = model.factor_count();
        let mut visited_semantic = vec![false; factor_count];
        let mut components = Vec::new();
        let mut component_by_variable = vec![None; variable_count];

        for seed_index in 0..factor_count {
            let seed = factor_id_from_index(seed_index);
            if visited_semantic[seed_index]
                || matches!(
                    model.factor(seed).expect("prepared factor must exist"),
                    FactorDefinition::TraversalTopology(_)
                        | FactorDefinition::AromaticMatching(_)
                        | FactorDefinition::DirectionalRingPlacement(_)
                )
            {
                continue;
            }

            let mut binary_factors = BTreeSet::new();
            let mut potential_layout_factors = BTreeSet::new();
            let mut potential_resource_factors = BTreeSet::new();
            let mut always_core_variables = BTreeSet::new();
            let mut potential_core_variables = BTreeSet::new();
            let mut structural_factors = BTreeSet::new();
            let mut structural_partitions = BTreeMap::new();
            let mut pending = VecDeque::from([seed]);

            while let Some(factor_id) = pending.pop_front() {
                if visited_semantic[factor_id.index()] {
                    continue;
                }
                let factor = model
                    .factor(factor_id)
                    .expect("pending semantic factor must exist");
                let semantic_variables = match factor {
                    FactorDefinition::BinaryRelation(relation) => {
                        visited_semantic[factor_id.index()] = true;
                        binary_factors.insert(factor_id);
                        always_core_variables.extend(relation.variables().iter().copied());
                        potential_core_variables.extend(relation.variables().iter().copied());
                        relation.variables().as_slice()
                    }
                    FactorDefinition::TetrahedralLayout(layout) => {
                        visited_semantic[factor_id.index()] = true;
                        potential_layout_factors.insert(factor_id);
                        potential_core_variables
                            .extend([layout.order_variable(), layout.role_pattern_variable()]);
                        for bond in layout.bonds() {
                            structural_partitions
                                .entry(bond.decision_variable())
                                .and_modify(|existing| {
                                    assert_eq!(
                                        *existing,
                                        bond.role_partition(),
                                        "one structural variable must retain one role partition"
                                    );
                                })
                                .or_insert(bond.role_partition());
                        }
                        layout.variables()
                    }
                    FactorDefinition::RingCountAtMost(count) => {
                        visited_semantic[factor_id.index()] = true;
                        potential_resource_factors.insert(factor_id);
                        potential_core_variables.extend(count.budget_bits().iter().copied());
                        for bond in count.bonds() {
                            structural_partitions
                                .entry(bond.decision_variable())
                                .and_modify(|existing| assert_eq!(*existing, bond.role_partition()))
                                .or_insert(bond.role_partition());
                        }
                        count.variables()
                    }
                    FactorDefinition::OrderedRingPhase(phase) => {
                        visited_semantic[factor_id.index()] = true;
                        potential_resource_factors.insert(factor_id);
                        potential_core_variables.extend(phase.free_bits().iter().copied());
                        potential_core_variables.extend([phase.order_variable(), phase.waiting_mask_variable()]);
                        for (_, bond) in phase.bonds() {
                            structural_partitions
                                .entry(bond.decision_variable())
                                .and_modify(|existing| assert_eq!(*existing, bond.role_partition()))
                                .or_insert(bond.role_partition());
                        }
                        phase.variables()
                    }
                    FactorDefinition::EventSchedule(schedule) => {
                        visited_semantic[factor_id.index()] = true;
                        potential_resource_factors.insert(factor_id);
                        potential_core_variables.extend(schedule.variables().iter().copied());
                        for edge in schedule.edges.iter() {
                            structural_partitions.entry(edge.decision_variable)
                                .and_modify(|existing| assert_eq!(*existing, edge.role_partition))
                                .or_insert(edge.role_partition);
                        }
                        schedule.variables()
                    }
                    FactorDefinition::OrdinaryOrder(order) => {
                        visited_semantic[factor_id.index()] = true;
                        potential_resource_factors.insert(factor_id);
                        potential_core_variables.extend(order.variables().iter().copied());
                        order.variables()
                    }
                    FactorDefinition::AromaticMatching(_) => {
                        unreachable!(
                            "exclusive aromatic matching factors are not exact-search seeds"
                        )
                    }
                    FactorDefinition::DirectionalRingPlacement(_) => {
                        unreachable!("local directional ring factors are not exact-search seeds")
                    }
                    FactorDefinition::TraversalTopology(_) => {
                        unreachable!("semantic traversal cannot queue a spanning factor")
                    }
                };

                for &variable in semantic_variables {
                    for &neighbour in model
                        .potential_factors_for_variable(variable)
                        .expect("semantic scope variable must have potential adjacency")
                    {
                        match model.factor(neighbour).expect("adjacent factor must exist") {
                            FactorDefinition::BinaryRelation(_)
                            | FactorDefinition::TetrahedralLayout(_)
                            | FactorDefinition::RingCountAtMost(_)
                            | FactorDefinition::OrderedRingPhase(_)
                            | FactorDefinition::OrdinaryOrder(_)
                            | FactorDefinition::EventSchedule(_) => {
                                if !visited_semantic[neighbour.index()] {
                                    pending.push_back(neighbour);
                                }
                            }
                            FactorDefinition::AromaticMatching(_)
                            | FactorDefinition::DirectionalRingPlacement(_) => {}
                            FactorDefinition::TraversalTopology(spanning_tree) => {
                                if !structural_factors.insert(neighbour) {
                                    continue;
                                }
                                for edge in spanning_tree.edges() {
                                    structural_partitions
                                        .entry(edge.decision_variable())
                                        .and_modify(|existing| {
                                            assert_eq!(
                                                *existing,
                                                edge.role_partition(),
                                                "one structural variable must retain one role partition"
                                            );
                                        })
                                        .or_insert(edge.role_partition());
                                }
                                for &structural_variable in spanning_tree.variables() {
                                    for &candidate in model
                                        .potential_factors_for_variable(structural_variable)
                                        .expect("structural variable must have potential adjacency")
                                    {
                                        if matches!(
                                            model.factor(candidate).expect("factor must exist"),
                                            FactorDefinition::BinaryRelation(_)
                                                | FactorDefinition::TetrahedralLayout(_)
                                                | FactorDefinition::RingCountAtMost(_)
                                                | FactorDefinition::OrderedRingPhase(_)
                                                | FactorDefinition::OrdinaryOrder(_)
                                                | FactorDefinition::EventSchedule(_)
                                        ) && !visited_semantic[candidate.index()]
                                        {
                                            pending.push_back(candidate);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let always_core_variables = always_core_variables.into_iter().collect::<Vec<_>>();
            let potential_layout_factors = potential_layout_factors.into_iter().collect::<Vec<_>>();
            let component =
                if structural_partitions.is_empty() && potential_layout_factors.is_empty()
                    && potential_resource_factors.is_empty() {
                    // Arc consistency is exact on an acyclic binary incidence
                    // graph, so only cyclic pure-binary components need search.
                    (binary_factors.len() >= always_core_variables.len()).then(|| {
                        ExactComponent::Binary(BinaryExactComponent::compile(
                            model,
                            always_core_variables,
                            binary_factors.into_iter().collect(),
                        ))
                    })
                } else {
                    let structural_variables = structural_partitions
                        .into_iter()
                        .map(|(variable, role_partition)| StructuralProjection {
                            variable,
                            role_partition,
                        })
                        .collect::<Vec<_>>();
                    let projected_variables = potential_core_variables
                        .into_iter()
                        .chain(structural_variables.iter().map(|entry| entry.variable))
                        .collect::<BTreeSet<_>>();
                    Some(ExactComponent::Mixed(MixedExactComponent {
                        always_core_variables: always_core_variables.into_boxed_slice(),
                        potential_layout_factors: potential_layout_factors.into_boxed_slice(),
                        potential_resource_factors: potential_resource_factors.into_iter().collect::<Vec<_>>().into_boxed_slice(),
                        structural_variables: structural_variables.into_boxed_slice(),
                        projected_variables: projected_variables
                            .into_iter()
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                    }))
                };

            let Some(component) = component else {
                continue;
            };
            let component_id = components.len();
            for variable in component.projected_variables() {
                assert_eq!(
                    component_by_variable[variable.index()],
                    None,
                    "native exact components must be disjoint"
                );
                component_by_variable[variable.index()] = Some(component_id);
            }
            components.push(component);
        }

        Self {
            components: components.into_boxed_slice(),
            component_by_variable: component_by_variable.into_boxed_slice(),
        }
    }

    fn component_for(&self, variable: VariableId) -> Option<usize> {
        self.component_by_variable
            .get(variable.index())
            .copied()
            .flatten()
    }
}

#[derive(Debug)]
enum ExactComponent {
    Binary(BinaryExactComponent),
    Mixed(MixedExactComponent),
}

impl ExactComponent {
    fn projected_variables(&self) -> &[VariableId] {
        match self {
            Self::Binary(component) => &component.variables,
            Self::Mixed(component) => &component.projected_variables,
        }
    }
}

#[derive(Debug)]
struct MixedExactComponent {
    always_core_variables: Box<[VariableId]>,
    potential_layout_factors: Box<[FactorId]>,
    potential_resource_factors: Box<[FactorId]>,
    structural_variables: Box<[StructuralProjection]>,
    projected_variables: Box<[VariableId]>,
}

#[derive(Copy, Clone, Debug)]
struct StructuralProjection {
    variable: VariableId,
    role_partition: EdgeRolePartition,
}

#[derive(Copy, Clone, Debug)]
struct LocalBinaryFactor {
    factor: FactorId,
    left: usize,
    right: usize,
}

#[derive(Debug)]
struct BinaryExactComponent {
    variables: Box<[VariableId]>,
    factors: Box<[LocalBinaryFactor]>,
    factors_by_variable: Vec<Vec<usize>>,
}

impl BinaryExactComponent {
    fn compile(
        model: &ConstraintModel,
        variables: Vec<VariableId>,
        factor_ids: Vec<FactorId>,
    ) -> Self {
        let mut factors = Vec::with_capacity(factor_ids.len());
        let mut factors_by_variable = vec![Vec::new(); variables.len()];

        for factor_id in factor_ids {
            let FactorDefinition::BinaryRelation(relation) = model
                .factor(factor_id)
                .expect("component factor must exist")
            else {
                unreachable!("binary component must contain only binary relations");
            };
            let left = variables
                .binary_search(&relation.left())
                .expect("component must contain the factor's left variable");
            let right = variables
                .binary_search(&relation.right())
                .expect("component must contain the factor's right variable");
            let local_factor = factors.len();
            factors.push(LocalBinaryFactor {
                factor: factor_id,
                left,
                right,
            });
            factors_by_variable[left].push(local_factor);
            factors_by_variable[right].push(local_factor);
        }

        Self {
            variables: variables.into_boxed_slice(),
            factors: factors.into_boxed_slice(),
            factors_by_variable,
        }
    }

    fn exact_supported_domains(
        &self,
        model: &ConstraintModel,
        mut domains: Vec<Domain>,
    ) -> Option<Vec<Domain>> {
        let all_factors = (0..self.factors.len()).collect::<Vec<_>>();
        self.propagate(model, &mut domains, all_factors).ok()?;

        let target_domains = domains.clone();
        let mut supported = vec![Domain::empty(); domains.len()];
        let mut stack = vec![SearchNode {
            domains,
            seeds: Vec::new(),
        }];
        let mut found_solution = false;

        while let Some(node) = stack.pop() {
            let mut candidate = node.domains;
            if self.propagate(model, &mut candidate, node.seeds).is_err() {
                continue;
            }

            let Some(variable) = branch_variable(&candidate) else {
                found_solution = true;
                for (accumulator, value) in supported.iter_mut().zip(candidate) {
                    *accumulator = accumulator.union(value);
                }
                if supported == target_domains {
                    break;
                }
                continue;
            };

            let mut values = candidate[variable].iter().collect::<Vec<_>>();
            values.reverse();
            for value in values {
                let mut child = candidate.clone();
                child[variable] = Domain::from_bits(1_u64 << value);
                stack.push(SearchNode {
                    domains: child,
                    seeds: self.factors_by_variable[variable].clone(),
                });
            }
        }

        found_solution.then_some(supported)
    }

    fn propagate(
        &self,
        model: &ConstraintModel,
        domains: &mut [Domain],
        seeds: impl IntoIterator<Item = usize>,
    ) -> Result<(), NativeSolverError> {
        let mut queue = VecDeque::new();
        let mut queued = vec![false; self.factors.len()];

        for factor in seeds {
            if !queued[factor] {
                queued[factor] = true;
                queue.push_back(factor);
            }
        }

        while let Some(factor_index) = queue.pop_front() {
            queued[factor_index] = false;
            let factor = &self.factors[factor_index];
            let FactorDefinition::BinaryRelation(relation) = model
                .factor(factor.factor)
                .expect("compiled binary factor must resolve")
            else {
                unreachable!("compiled binary factor ID must retain its kind");
            };
            let old_left = domains[factor.left];
            let old_right = domains[factor.right];
            let (new_left, new_right) = revised_binary_domains(relation, old_left, old_right)?;

            if new_left != old_left {
                domains[factor.left] = new_left;
                enqueue_local_factors(
                    factor.left,
                    &self.factors_by_variable,
                    &mut queue,
                    &mut queued,
                );
            }
            if new_right != old_right {
                domains[factor.right] = new_right;
                enqueue_local_factors(
                    factor.right,
                    &self.factors_by_variable,
                    &mut queue,
                    &mut queued,
                );
            }
        }

        Ok(())
    }
}

struct SearchNode {
    domains: Vec<Domain>,
    seeds: Vec<usize>,
}

fn revise_binary_relation(
    factor: &BinaryRelationFactor,
    domains: &mut PagedStore<Domain>,
) -> Result<[Option<VariableId>; 2], NativeSolverError> {
    let left = factor.left();
    let right = factor.right();
    let old_left = domains[left.index()];
    let old_right = domains[right.index()];
    let (new_left, new_right) = revised_binary_domains(factor, old_left, old_right)?;

    let left_reduced = (new_left != old_left).then_some(left);
    let right_reduced = (new_right != old_right).then_some(right);
    if left_reduced.is_some() {
        domains[left.index()] = new_left;
    }
    if right_reduced.is_some() {
        domains[right.index()] = new_right;
    }
    Ok([left_reduced, right_reduced])
}

fn revised_binary_domains(
    factor: &BinaryRelationFactor,
    old_left: Domain,
    old_right: Domain,
) -> Result<(Domain, Domain), NativeSolverError> {
    let new_left = values_with_support(old_left, old_right, |value| factor.allowed_right(value));
    if new_left.is_empty() {
        return Err(NativeSolverError::Contradiction);
    }

    let new_right = values_with_support(old_right, new_left, |value| factor.allowed_left(value));
    if new_right.is_empty() {
        return Err(NativeSolverError::Contradiction);
    }

    Ok((new_left, new_right))
}

fn revise_directional_ring_placement(
    factor: &DirectionalRingPlacementFactor,
    domains: &mut PagedStore<Domain>,
) -> Result<[Option<VariableId>; 2], NativeSolverError> {
    let mark = factor.mark_variable();
    let plan = factor.plan_variable();
    let old_mark = domains[mark.index()];
    let old_plan = domains[plan.index()];
    let new_mark = values_with_support(old_mark, old_plan, |value| factor.allowed_plans(value));
    if new_mark.is_empty() {
        return Err(NativeSolverError::Contradiction);
    }
    let new_plan = values_with_support(old_plan, new_mark, |value| factor.allowed_marks(value));
    if new_plan.is_empty() {
        return Err(NativeSolverError::Contradiction);
    }
    let mark_reduced = (new_mark != old_mark).then_some(mark);
    let plan_reduced = (new_plan != old_plan).then_some(plan);
    if mark_reduced.is_some() {
        domains[mark.index()] = new_mark;
    }
    if plan_reduced.is_some() {
        domains[plan.index()] = new_plan;
    }
    Ok([mark_reduced, plan_reduced])
}

fn values_with_support(
    domain: Domain,
    other_domain: Domain,
    support: impl Fn(u8) -> Domain,
) -> Domain {
    let mut retained = Domain::empty();
    for value in domain.iter() {
        if !support(value).intersect(other_domain).is_empty() {
            retained = retained.union(Domain::from_bits(1_u64 << value));
        }
    }
    retained
}

fn revise_ring_count_at_most(
    factor: &crate::model::RingCountAtMostFactor,
    domains: &mut PagedStore<Domain>,
) -> Result<Vec<VariableId>, NativeSolverError> {
    let mut budget = 0_usize;
    for (bit, &variable) in factor.budget_bits().iter().enumerate() {
        let domain = domains[variable.index()];
        if !domain.is_singleton() { return Ok(Vec::new()); }
        budget |= usize::from(domain.iter().next().unwrap()) << bit;
    }
    let mut required = 0_usize;
    let mut ambiguous = Vec::new();
    for bond in factor.bonds() {
        let variable = bond.decision_variable();
        let domain = domains[variable.index()];
        let traversal = !domain.intersect(bond.role_partition().traversal_values()).is_empty();
        let ring = !domain.intersect(bond.role_partition().ring_values()).is_empty();
        match (traversal, ring) {
            (false, true) => required += 1,
            (true, true) => ambiguous.push((variable, bond.role_partition().traversal_values())),
            (true, false) => {},
            (false, false) => return Err(NativeSolverError::Contradiction),
        }
    }
    if required > budget { return Err(NativeSolverError::Contradiction); }
    if required < budget { return Ok(Vec::new()); }
    let mut reductions = Vec::new();
    for (variable, traversal) in ambiguous {
        domains[variable.index()] = domains[variable.index()].intersect(traversal);
        reductions.push(variable);
    }
    Ok(reductions)
}

fn revise_ordered_ring_phase(
    factor: &crate::model::OrderedRingPhaseFactor,
    domains: &mut PagedStore<Domain>,
) -> Result<Vec<VariableId>, NativeSolverError> {
    let mut free = 0_usize;
    for (bit, &variable) in factor.free_bits().iter().enumerate() {
        let domain = domains[variable.index()];
        if !domain.is_singleton() { return Ok(Vec::new()); }
        free |= usize::from(domain.iter().next().unwrap()) << bit;
    }
    let orders = domains[factor.order_variable().index()];
    let masks = domains[factor.waiting_mask_variable().index()];
    let mut supported_orders = Domain::empty();
    let mut supported_masks = Domain::empty();
    let mut supported_roles = vec![Domain::empty(); factor.bonds().len()];
    for mask in masks.iter() {
        for order in orders.iter() {
            for pattern in 0..(1_u8 << factor.bonds().len()) {
                let roles_compatible = factor.bonds().iter().enumerate().all(|(position, (_, edge))| {
                    let role = if pattern & (1 << position) != 0 {
                        edge.role_partition().ring_values()
                    } else {
                        edge.role_partition().traversal_values()
                    };
                    !domains[edge.decision_variable().index()].intersect(role).is_empty()
                });
                if !roles_compatible || !factor.permits(order, mask, free,
                    |position| pattern & (1 << position) != 0) { continue; }
                supported_orders = supported_orders.union(Domain::singleton(order).unwrap());
                supported_masks = supported_masks.union(Domain::singleton(mask).unwrap());
                for (position, (_, edge)) in factor.bonds().iter().enumerate() {
                    let role = if pattern & (1 << position) != 0 {
                        edge.role_partition().ring_values()
                    } else {
                        edge.role_partition().traversal_values()
                    };
                    supported_roles[position] = supported_roles[position].union(role);
                }
            }
        }
    }
    if supported_orders.is_empty() { return Err(NativeSolverError::Contradiction); }
    let mut reductions = Vec::new();
    for (variable, supported) in [(factor.order_variable(), supported_orders),
        (factor.waiting_mask_variable(), supported_masks)] {
        let narrowed = domains[variable.index()].intersect(supported);
        if narrowed != domains[variable.index()] {
            domains[variable.index()] = narrowed;
            reductions.push(variable);
        }
    }
    for (position, (_, edge)) in factor.bonds().iter().enumerate() {
        let variable = edge.decision_variable();
        let narrowed = domains[variable.index()].intersect(supported_roles[position]);
        if narrowed != domains[variable.index()] {
            domains[variable.index()] = narrowed;
            reductions.push(variable);
        }
    }
    Ok(reductions)
}

fn revise_tetrahedral_layout(
    factor: &TetrahedralLayoutFactor,
    domains: &mut PagedStore<Domain>,
) -> Result<Vec<VariableId>, NativeSolverError> {
    let order_variable = factor.order_variable();
    let pattern_variable = factor.role_pattern_variable();
    let current_order = domains[order_variable.index()];
    let current_patterns = domains[pattern_variable.index()];
    let mut supported_order = Domain::empty();
    let mut supported_patterns = Domain::empty();
    let mut supported_bonds = vec![Domain::empty(); factor.bonds().len()];

    for pattern in current_patterns.iter() {
        let order_support = current_order.intersect(factor.allowed_orders(pattern));
        if order_support.is_empty() {
            continue;
        }
        let mut bond_support = Vec::with_capacity(factor.bonds().len());
        let mut viable = true;
        for bond in factor.bonds() {
            let current = domains[bond.decision_variable().index()];
            let role_values = if pattern & (1_u8 << bond.pattern_bit()) == 0 {
                bond.role_partition().traversal_values()
            } else {
                bond.role_partition().ring_values()
            };
            let support = current.intersect(role_values);
            if support.is_empty() {
                viable = false;
                break;
            }
            bond_support.push(support);
        }
        if !viable {
            continue;
        }
        supported_order = supported_order.union(order_support);
        supported_patterns = supported_patterns.union(Domain::from_bits(1_u64 << pattern));
        for (accumulator, support) in supported_bonds.iter_mut().zip(bond_support) {
            *accumulator = accumulator.union(support);
        }
    }

    if supported_patterns.is_empty() {
        return Err(NativeSolverError::Contradiction);
    }

    let mut reductions = Vec::new();
    for (variable, supported) in std::iter::once((order_variable, supported_order))
        .chain(std::iter::once((pattern_variable, supported_patterns)))
        .chain(
            factor
                .bonds()
                .iter()
                .map(|bond| bond.decision_variable())
                .zip(supported_bonds),
        )
    {
        let current = domains[variable.index()];
        debug_assert!(supported.is_subset_of(current));
        if supported != current {
            domains[variable.index()] = supported;
            reductions.push(variable);
        }
    }
    Ok(reductions)
}

fn event_partial_local_order_feasible(
    factor: &EventScheduleFactor,
    domains: &PagedStore<Domain>,
    atom: crate::ids::AtomId,
    incoming: Option<crate::ids::BondId>,
    children: &[(usize, crate::ids::BondId, u8)],
) -> bool {
    let incidences = factor.edges.iter().filter(|edge| edge.a == atom || edge.b == atom)
        .map(|edge| edge.bond).collect::<Vec<_>>();
    let ring = factor.edges.iter().filter(|edge| (edge.a == atom || edge.b == atom)
        && domains[edge.decision_variable.index()].is_subset_of(edge.role_partition.ring_values()))
        .map(|edge| edge.bond).collect::<Vec<_>>();
    let mut before = Vec::new();
    if let Some(entry) = incoming {
        before.extend(incidences.iter().copied().filter(|bond| *bond != entry)
            .map(|bond| (entry, bond)));
    }
    for ring_bond in ring {
        before.extend(children.iter().map(|(_, child, _)| (ring_bond, *child)));
    }
    before.extend(children.windows(2).map(|pair| (pair[0].1, pair[1].1)));
    if children.windows(2).any(|pair| pair[0].2 == 2) { return false; }

    if let Some(center) = factor.centers.iter().find(|center| center.atom == atom) {
        let rows = match incoming {
            Some(entry) => center.entry_layout_rows.iter()
                .find_map(|(bond, rows)| (*bond == entry).then_some(rows)),
            None => Some(&center.root_layout_rows),
        };
        let Some(rows) = rows else { return false; };
        return domains[center.order_variable.index()].iter().any(|value| {
            let Some(order) = crate::tetrahedral::order_for_value(&center.reference_order, value) else {
                return false;
            };
            let position = |bond| order.iter().position(|ligand|
                *ligand == crate::tetrahedral::TetrahedralLigand::Bond(bond));
            domains[center.role_pattern_variable.index()].iter().any(|pattern| {
                rows[pattern as usize].contains(value)
                    && center.bond_pattern_bits.iter().all(|(bond, bit)| {
                        let edge = factor.edges.iter().find(|edge| edge.bond == *bond)
                            .expect("prepared center bond must have a role");
                        let role = if pattern & (1 << bit) != 0 {
                            edge.role_partition.ring_values()
                        } else { edge.role_partition.traversal_values() };
                        !domains[edge.decision_variable.index()].intersect(role).is_empty()
                    })
                    && before.iter().all(|(first, second)|
                        position(*first).zip(position(*second)).is_some_and(|(a, b)| a < b))
            })
        });
    }
    let Some(center) = factor.ordinary_centers.iter().find(|center| center.atom == atom) else {
        return false;
    };
    before.into_iter().all(|(first, second)| {
        let Some(pair) = center.pairs.iter().find(|pair|
            (pair.first == first && pair.second == second)
                || (pair.first == second && pair.second == first)) else { return false; };
        domains[pair.variable.index()].contains(u8::from(pair.first == second))
    })
}

fn revise_event_schedule(
    factor: &EventScheduleFactor,
    domains: &mut PagedStore<Domain>,
) -> Result<Vec<VariableId>, NativeSolverError> {
    let count = factor.visits.len();
    let possible = factor.visits.iter().map(|visit| {
        (0..count).filter(|rank| visit.rank_bits.iter().enumerate().all(|(bit, variable)|
            domains[variable.index()].contains(((rank >> bit) & 1) as u8)))
            .collect::<Vec<_>>()
    }).collect::<Vec<_>>();
    if possible.iter().any(Vec::is_empty) { return Err(NativeSolverError::Contradiction); }
    let fixed = possible.iter().filter(|ranks| ranks.len() == 1)
        .map(|ranks| ranks[0]).collect::<Vec<_>>();
    if fixed.iter().copied().collect::<BTreeSet<_>>().len() != fixed.len() {
        return Err(NativeSolverError::Contradiction);
    }
    let mut reductions = Vec::new();
    for (visit, ranks) in factor.visits.iter().zip(&possible) {
        let supported = ranks.iter().copied().filter(|rank|
            ranks.len() == 1 || !fixed.contains(rank)).collect::<Vec<_>>();
        if supported.is_empty() { return Err(NativeSolverError::Contradiction); }
        if supported.len() == 1 {
            let rank = supported[0];
            let allowed_kind = if rank == 0 { Domain::singleton(0).unwrap() }
                else if factor.segments { Domain::from_indices(1..=4).unwrap() }
                else { crate::event_schedule::EntryKind::domain() };
            let narrowed = domains[visit.kind.index()].intersect(allowed_kind);
            if narrowed.is_empty() { return Err(NativeSolverError::Contradiction); }
            if narrowed != domains[visit.kind.index()] {
                domains[visit.kind.index()] = narrowed;
                reductions.push(visit.kind);
            }
        }
        if let Some(root) = factor.forest_roots.get(visit.atom.index()) {
            let bonded = Domain::from_indices([1, 2]).unwrap();
            let nonbonded = Domain::from_indices([0, 3, 4]).unwrap();
            let kind_domain = domains[visit.kind.index()];
            let root_domain = domains[root.index()];
            let supported_roots = (if kind_domain.intersect(bonded).is_empty() {
                Domain::empty()
            } else { Domain::singleton(0).unwrap() }).union(
                if kind_domain.intersect(nonbonded).is_empty() {
                    Domain::empty()
                } else { Domain::singleton(1).unwrap() });
            let narrowed_root = root_domain.intersect(supported_roots);
            if narrowed_root.is_empty() { return Err(NativeSolverError::Contradiction); }
            if narrowed_root != root_domain {
                domains[root.index()] = narrowed_root;
                reductions.push(*root);
            }
            let supported_kinds = (if narrowed_root.contains(0) { bonded } else { Domain::empty() })
                .union(if narrowed_root.contains(1) { nonbonded } else { Domain::empty() });
            let narrowed_kind = kind_domain.intersect(supported_kinds);
            if narrowed_kind.is_empty() { return Err(NativeSolverError::Contradiction); }
            if narrowed_kind != kind_domain {
                domains[visit.kind.index()] = narrowed_kind;
                reductions.push(visit.kind);
            }
        }
        let kinds = domains[visit.kind.index()];
        let branch_kinds = Domain::from_indices([1, 3]).unwrap();
        if kinds.intersect(branch_kinds).is_empty() {
            for variable in visit.end_bits.iter().copied() {
                let narrowed = domains[variable.index()].intersect(Domain::singleton(0).unwrap());
                if narrowed.is_empty() { return Err(NativeSolverError::Contradiction); }
                if narrowed != domains[variable.index()] {
                    domains[variable.index()] = narrowed;
                    reductions.push(variable);
                }
            }
        } else if kinds.is_subset_of(branch_kinds) && supported.len() == 1 {
            for variable in visit.end_bits.iter().take(supported[0] + 1).copied() {
                let narrowed = domains[variable.index()].intersect(Domain::singleton(1).unwrap());
                if narrowed.is_empty() { return Err(NativeSolverError::Contradiction); }
                if narrowed != domains[variable.index()] {
                    domains[variable.index()] = narrowed;
                    reductions.push(variable);
                }
            }
        }
        let mut forced_one = false;
        for variable in visit.end_bits.iter().rev().copied() {
            forced_one |= domains[variable.index()] == Domain::singleton(1).unwrap();
            if forced_one && domains[variable.index()] != Domain::singleton(1).unwrap() {
                let narrowed = domains[variable.index()].intersect(Domain::singleton(1).unwrap());
                if narrowed.is_empty() { return Err(NativeSolverError::Contradiction); }
                domains[variable.index()] = narrowed;
                reductions.push(variable);
            }
        }
        let mut forced_zero = false;
        for variable in visit.end_bits.iter().copied() {
            forced_zero |= domains[variable.index()] == Domain::singleton(0).unwrap();
            if forced_zero && domains[variable.index()] != Domain::singleton(0).unwrap() {
                let narrowed = domains[variable.index()].intersect(Domain::singleton(0).unwrap());
                if narrowed.is_empty() { return Err(NativeSolverError::Contradiction); }
                domains[variable.index()] = narrowed;
                reductions.push(variable);
            }
        }
        for (bit, variable) in visit.rank_bits.iter().copied().enumerate() {
            let bit_domain = Domain::from_indices(supported.iter()
                .map(|rank| ((rank >> bit) & 1) as u8)).unwrap();
            let narrowed = domains[variable.index()].intersect(bit_domain);
            if narrowed != domains[variable.index()] {
                domains[variable.index()] = narrowed;
                reductions.push(variable);
            }
        }
    }
    let capacity_upper = factor.capacity_bits.iter().enumerate().fold(0_usize, |upper, (bit, variable)|
        upper | (usize::from(domains[variable.index()].contains(1)) << bit));
    let mut prefix = BTreeSet::new();
    let mut source = None;
    let mut returns = Vec::<(crate::ids::AtomId, usize)>::new();
    let mut incoming = vec![None; count];
    let mut children = vec![Vec::<(usize, crate::ids::BondId, u8)>::new(); count];
    for rank in 0..count {
        let Some(visit) = factor.visits.iter().find(|visit|
            possible[visit.atom.index()].len() == 1 && possible[visit.atom.index()][0] == rank)
        else { break; };
        let kind_domain = domains[visit.kind.index()];
        if !kind_domain.is_singleton() { break; }
        let kind = kind_domain.iter().next().unwrap();
        let branch = kind == 1 || kind == 3;
        let end = if branch {
            if visit.end_bits.iter().any(|variable| !domains[variable.index()].is_singleton()) {
                break;
            }
            visit.end_bits.iter().take_while(|variable| domains[variable.index()].contains(1))
                .count().checked_sub(1).ok_or(NativeSolverError::Contradiction)?
        } else { 0 };
        while returns.last().is_some_and(|(_, end)| *end < rank) {
            source = Some(returns.pop().unwrap().0);
        }
        if kind == 0 {
            if rank != 0 && (factor.segments || !returns.is_empty()) {
                return Err(NativeSolverError::Contradiction);
            }
        } else {
            let parent = source.ok_or(NativeSolverError::Contradiction)?;
            if kind == 1 || kind == 2 {
                let edge = factor.edges.iter().find(|edge|
                    (edge.a == parent && edge.b == visit.atom)
                        || (edge.b == parent && edge.a == visit.atom))
                    .ok_or(NativeSolverError::Contradiction)?;
                if domains[edge.decision_variable.index()]
                    .intersect(edge.role_partition.traversal_values()).is_empty() {
                    return Err(NativeSolverError::Contradiction);
                }
                incoming[visit.atom.index()] = Some(edge.bond);
                children[parent.index()].push((rank, edge.bond, kind));
                if !event_partial_local_order_feasible(factor, domains, parent,
                    incoming[parent.index()], &children[parent.index()]) {
                    return Err(NativeSolverError::Contradiction);
                }
                if let Some(root) = factor.forest_roots.get(visit.atom.index()) {
                    if !domains[root.index()].contains(0) { return Err(NativeSolverError::Contradiction); }
                }
            } else {
                if !factor.segments { return Err(NativeSolverError::Contradiction); }
                if let Some(root) = factor.forest_roots.get(visit.atom.index()) {
                    if !domains[root.index()].contains(1) { return Err(NativeSolverError::Contradiction); }
                }
            }
            if branch {
                if end < rank || returns.last().is_some_and(|(_, parent_end)| end > *parent_end) {
                    return Err(NativeSolverError::Contradiction);
                }
                returns.push((parent, end));
            }
        }
        source = Some(visit.atom);
        prefix.insert(visit.atom);
        if !event_partial_local_order_feasible(factor, domains, visit.atom,
            incoming[visit.atom.index()], &children[visit.atom.index()]) {
            return Err(NativeSolverError::Contradiction);
        }
        let mut next_source = source;
        let mut pending_returns = returns.clone();
        while pending_returns.last().is_some_and(|(_, end)| *end <= rank) {
            next_source = Some(pending_returns.pop().unwrap().0);
        }
        let accessible = pending_returns.iter().map(|(atom, _)| *atom)
            .chain(next_source).collect::<BTreeSet<_>>();
        if factor.edges.iter().any(|edge| {
            let role = domains[edge.decision_variable.index()];
            if !role.is_subset_of(edge.role_partition.traversal_values()) { return false; }
            let visited_a = prefix.contains(&edge.a);
            let visited_b = prefix.contains(&edge.b);
            visited_a != visited_b
                && !accessible.contains(&if visited_a { edge.a } else { edge.b })
        }) { return Err(NativeSolverError::Contradiction); }
        let forced_live = factor.edges.iter().filter(|edge| {
            let domain = domains[edge.decision_variable.index()];
            domain.is_subset_of(edge.role_partition.ring_values())
                && prefix.contains(&edge.a) != prefix.contains(&edge.b)
        }).count();
        if forced_live > capacity_upper { return Err(NativeSolverError::Contradiction); }
    }
    if factor.variables().iter().all(|variable| domains[variable.index()].is_singleton()) {
        let assignment = |variable: VariableId| domains[variable.index()].iter().next().unwrap();
        if !factor.permits(assignment) { return Err(NativeSolverError::Contradiction); }
    }
    Ok(reductions)
}

fn revise_ordinary_order(
    factor: &crate::model::OrdinaryOrderFactor,
    domains: &mut PagedStore<Domain>,
) -> Result<Vec<VariableId>, NativeSolverError> {
    let mut bonds = BTreeSet::new();
    let mut fixed = BTreeMap::<crate::ids::BondId, Vec<crate::ids::BondId>>::new();
    let mut indegree = BTreeMap::<crate::ids::BondId, usize>::new();
    for pair in factor.pairs() {
        bonds.insert(pair.first);
        bonds.insert(pair.second);
        indegree.entry(pair.first).or_default();
        indegree.entry(pair.second).or_default();
        let domain = domains[pair.variable.index()];
        if domain.is_singleton() {
            let (earlier, later) = if domain.contains(0) {
                (pair.first, pair.second)
            } else { (pair.second, pair.first) };
            fixed.entry(earlier).or_default().push(later);
            *indegree.get_mut(&later).expect("pair endpoint must exist") += 1;
        }
    }
    let mut pending = indegree.iter().filter_map(|(bond, count)| (*count == 0).then_some(*bond))
        .collect::<Vec<_>>();
    let mut removed = 0;
    while let Some(bond) = pending.pop() {
        removed += 1;
        for next in fixed.get(&bond).into_iter().flatten() {
            let count = indegree.get_mut(next).expect("pair endpoint must exist");
            *count -= 1;
            if *count == 0 { pending.push(*next); }
        }
    }
    if removed != bonds.len() { return Err(NativeSolverError::Contradiction); }
    Ok(Vec::new())
}

fn revise_traversal_topology(
    factor: &TraversalTopologyFactor,
    domains: &mut PagedStore<Domain>,
    #[cfg(test)] work: (&AtomicUsize, &AtomicUsize, &AtomicUsize),
    #[cfg(not(test))] _work: (),
) -> Result<Vec<VariableId>, NativeSolverError> {
    let rooted = !factor.root_variables().is_empty();
    let super_root = factor.atoms().len();
    let mut components = DisjointSet::new(super_root + usize::from(rooted));

    if rooted {
        for (atom, variable) in factor.root_variables().iter().copied().enumerate() {
            let domain = domains[variable.index()];
            if domain == Domain::singleton(1).unwrap() && !components.union(atom, super_root) {
                return Err(NativeSolverError::Contradiction);
            }
        }
    }

    // Forced traversal edges are the independent set that the remaining
    // graphic-matroid basis must extend. A cycle makes extension impossible.
    for edge in factor.edges() {
        #[cfg(test)]
        work.0.fetch_add(1, Ordering::Relaxed);
        let domain = domains[edge.decision_variable().index()];
        match role_membership(domain, edge.role_partition()) {
            (true, false) => {
                let a = factor_atom_index(factor, edge.a());
                let b = factor_atom_index(factor, edge.b());
                if !components.union(a, b) {
                    return Err(NativeSolverError::Contradiction);
                }
            }
            (false, true) | (true, true) => {}
            (false, false) => return Err(NativeSolverError::Contradiction),
        }
    }

    if factor.requirement() == crate::model::TraversalTopologyRequirement::Forest {
        let mut reductions = Vec::new();
        if rooted {
            for (atom, variable) in factor.root_variables().iter().copied().enumerate() {
                let domain = domains[variable.index()];
                if components.find(atom) == components.find(super_root) && domain.contains(1) && domain.contains(0) {
                    domains[variable.index()] = Domain::singleton(0).unwrap();
                    reductions.push(variable);
                }
            }
        }
        for edge in factor.edges() {
            let variable = edge.decision_variable();
            let domain = domains[variable.index()];
            let (can_traverse, can_ring) = role_membership(domain, edge.role_partition());
            if can_traverse && can_ring &&
                components.find(factor_atom_index(factor, edge.a())) ==
                components.find(factor_atom_index(factor, edge.b())) {
                domains[variable.index()] = domain.intersect(edge.role_partition().ring_values());
                reductions.push(variable);
            }
        }
        return Ok(reductions);
    }

    let mut quotient_by_root = BTreeMap::new();
    for atom in 0..factor.atoms().len() {
        #[cfg(test)]
        work.1.fetch_add(1, Ordering::Relaxed);
        let root = components.find(atom);
        if !quotient_by_root.contains_key(&root) {
            let quotient = quotient_by_root.len();
            quotient_by_root.insert(root, quotient);
        }
    }

    let mut reductions = Vec::new();
    let mut quotient_edges = Vec::new();

    for edge in factor.edges() {
        #[cfg(test)]
        work.2.fetch_add(1, Ordering::Relaxed);
        let variable = edge.decision_variable();
        let domain = domains[variable.index()];
        let membership = role_membership(domain, edge.role_partition());
        if membership == (true, false) || membership == (false, true) {
            continue;
        }
        if membership != (true, true) {
            return Err(NativeSolverError::Contradiction);
        }

        let a_root = components.find(factor_atom_index(factor, edge.a()));
        let b_root = components.find(factor_atom_index(factor, edge.b()));
        if a_root == b_root {
            let restricted = domain.intersect(edge.role_partition().ring_values());
            if restricted != domain {
                domains[variable.index()] = restricted;
                reductions.push(variable);
            }
            continue;
        }

        quotient_edges.push(QuotientEdge {
            variable,
            traversal_values: edge.role_partition().traversal_values(),
            a: quotient_by_root[&a_root],
            b: quotient_by_root[&b_root],
        });
    }

    let quotient_node_count = quotient_by_root.len();
    let bridges = quotient_bridges(quotient_node_count, &quotient_edges)?;
    for (edge, is_bridge) in quotient_edges.iter().zip(bridges) {
        if is_bridge {
            let domain = domains[edge.variable.index()];
            let restricted = domain.intersect(edge.traversal_values);
            if restricted != domain {
                domains[edge.variable.index()] = restricted;
                reductions.push(edge.variable);
            }
        }
    }

    Ok(reductions)
}

fn role_membership(domain: Domain, partition: crate::model::EdgeRolePartition) -> (bool, bool) {
    (
        !domain.intersect(partition.traversal_values()).is_empty(),
        !domain.intersect(partition.ring_values()).is_empty(),
    )
}

fn factor_atom_index(factor: &TraversalTopologyFactor, atom: crate::AtomId) -> usize {
    factor
        .atoms()
        .binary_search(&atom)
        .expect("traversal-topology edge endpoints must belong to the factor atom set")
}

#[derive(Copy, Clone, Debug)]
struct QuotientEdge {
    variable: VariableId,
    traversal_values: Domain,
    a: usize,
    b: usize,
}

fn quotient_bridges(
    node_count: usize,
    edges: &[QuotientEdge],
) -> Result<Vec<bool>, NativeSolverError> {
    if node_count <= 1 {
        return Ok(vec![false; edges.len()]);
    }

    let mut adjacency = vec![Vec::new(); node_count];
    for (edge_index, edge) in edges.iter().enumerate() {
        adjacency[edge.a].push(edge_index);
        adjacency[edge.b].push(edge_index);
    }

    let mut discovery = vec![usize::MAX; node_count];
    let mut low = vec![0; node_count];
    let mut bridges = vec![false; edges.len()];
    let mut next_time = 0;
    mark_quotient_bridges(
        0,
        None,
        edges,
        &adjacency,
        &mut discovery,
        &mut low,
        &mut bridges,
        &mut next_time,
    );

    if discovery.iter().any(|time| *time == usize::MAX) {
        return Err(NativeSolverError::Contradiction);
    }
    Ok(bridges)
}

#[allow(clippy::too_many_arguments)]
fn mark_quotient_bridges(
    node: usize,
    parent_edge: Option<usize>,
    edges: &[QuotientEdge],
    adjacency: &[Vec<usize>],
    discovery: &mut [usize],
    low: &mut [usize],
    bridges: &mut [bool],
    next_time: &mut usize,
) {
    discovery[node] = *next_time;
    low[node] = *next_time;
    *next_time += 1;

    for &edge_index in &adjacency[node] {
        if Some(edge_index) == parent_edge {
            continue;
        }
        let edge = edges[edge_index];
        let other = if edge.a == node { edge.b } else { edge.a };

        if discovery[other] == usize::MAX {
            mark_quotient_bridges(
                other,
                Some(edge_index),
                edges,
                adjacency,
                discovery,
                low,
                bridges,
                next_time,
            );
            low[node] = low[node].min(low[other]);
            if low[other] > discovery[node] {
                bridges[edge_index] = true;
            }
        } else {
            low[node] = low[node].min(discovery[other]);
        }
    }
}

#[derive(Debug)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        let parent = self.parent[value];
        if parent != value {
            self.parent[value] = self.find(parent);
        }
        self.parent[value]
    }

    /// Merge two sets and return whether they were previously distinct.
    fn union(&mut self, left: usize, right: usize) -> bool {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return false;
        }

        if self.rank[left_root] < self.rank[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        if self.rank[left_root] == self.rank[right_root] {
            self.rank[left_root] += 1;
        }
        true
    }
}

fn branch_variable(domains: &[Domain]) -> Option<usize> {
    domains
        .iter()
        .enumerate()
        .filter(|(_, domain)| domain.len() > 1)
        .min_by_key(|(index, domain)| (domain.len(), *index))
        .map(|(index, _)| index)
}

fn enqueue_factor(
    factor: FactorId,
    queue: &mut VecDeque<FactorId>,
    queued: &mut BTreeSet<FactorId>,
) {
    if queued.insert(factor) {
        queue.push_back(factor);
    }
}

fn enqueue_local_factors(
    variable: usize,
    factors_by_variable: &[Vec<usize>],
    queue: &mut VecDeque<usize>,
    queued: &mut [bool],
) {
    for &factor in &factors_by_variable[variable] {
        if !queued[factor] {
            queued[factor] = true;
            queue.push_back(factor);
        }
    }
}

fn factor_id_from_index(index: usize) -> FactorId {
    FactorId::new(
        u32::try_from(index).expect("constraint model validated the factor identifier capacity"),
    )
}

fn variable_id_from_index(index: usize) -> VariableId {
    VariableId::new(
        u32::try_from(index).expect("constraint model validated the variable identifier capacity"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::AtomId;
    use crate::solver::ConstraintSolver;
    use crate::model::{
        BondRole, ConstraintModelBuilder, EdgeRolePartition, SpanningTreeEdge,
        TetrahedralLayoutBond,
    };

    #[test]
    #[ignore = "opt-in work count for lazy mixed-search siblings"]
    fn mixed_first_witness_does_not_propagate_queued_sibling() {
        let mut builder = ConstraintModelBuilder::new();
        let variable = builder.add_variable(two_values()).unwrap();
        let mut state = NativeSolverState::initial(Arc::new(builder.build())).unwrap();
        let component = MixedExactComponent {
            always_core_variables: vec![variable].into(),
            potential_layout_factors: Box::new([]),
            potential_resource_factors: Box::new([]),
            structural_variables: Box::new([]),
            projected_variables: vec![variable].into(),
        };

        crate::work_diagnostic::begin();
        assert_eq!(state.complete_filter_mixed_component(&component, true), Ok(Vec::new()));
        let work = crate::work_diagnostic::finish();
        assert_eq!(work.sibling_offered, 2);
        assert_eq!(work.sibling_proposals, 1);
        assert_eq!(work.sibling_propagated, 1);
        assert_eq!(work.children_explored, 1);
    }

    #[test]
    fn mixed_search_skips_contradictory_first_child_and_finds_later_witness() {
        let mut builder = ConstraintModelBuilder::new();
        let x = builder.add_variable(two_values()).unwrap();
        let y = builder.add_variable(two_values()).unwrap();
        let z = builder.add_variable(two_values()).unwrap();
        builder.add_binary_relation(x, y, [(0, 0), (1, 0), (1, 1)]).unwrap();
        builder.add_binary_relation(x, z, [(0, 1), (1, 0), (1, 1)]).unwrap();
        builder.add_binary_relation(y, z, [(0, 0), (1, 1)]).unwrap();
        let model = Arc::new(builder.build());
        let mut state = NativeSolverState::initial(Arc::clone(&model)).unwrap();
        let component = MixedExactComponent {
            always_core_variables: vec![x, y, z].into(),
            potential_layout_factors: Box::new([]),
            potential_resource_factors: Box::new([]),
            structural_variables: Box::new([]),
            projected_variables: vec![x, y, z].into(),
        };
        // Model the locally consistent, pre-exact state that an initial or transition pass
        // can receive: exact filtering already removed x=0 from this prepared state.
        state.domains[x.index()] = two_values();
        assert!(matches!(state.restricted_many_and_propagated(&[(x, Domain::singleton(0).unwrap())]),
            Err(NativeSolverError::Contradiction)));
        assert_eq!(state.complete_filter_mixed_component(&component, true), Ok(Vec::new()));

        let mut unsatisfiable = state.clone();
        unsatisfiable.domains[x.index()] = Domain::singleton(0).unwrap();
        assert_eq!(unsatisfiable.complete_filter_mixed_component(&component, true),
            Err(NativeSolverError::Contradiction));

        assert_eq!(state.complete_filter_mixed_component(&component, false), Ok(vec![x]));
        let oracle = <crate::native_oracle::ExhaustiveSolverState as ConstraintSolver>::initial(model)
            .unwrap().unwrap_consistent();
        for variable in [x, y, z] {
            assert_eq!(state.domain_bound(variable), oracle.domain_bound(variable));
        }
    }

    #[test]
    fn active_ring_count_exact_projection_unions_both_role_assignments() {
        let mut builder = ConstraintModelBuilder::new();
        let roles = (0..2).map(|_| builder.add_variable(BondRole::role_domain()).unwrap())
            .collect::<Vec<_>>();
        let budget = builder.add_variable(two_values()).unwrap();
        builder.add_binary_relation(roles[0], roles[1], [(0, 1), (1, 0)]).unwrap();
        let bonds = roles.iter().enumerate().map(|(index, &variable)|
            SpanningTreeEdge::with_role_partition(variable, AtomId::new(0),
                AtomId::new((index + 1) as u32), EdgeRolePartition::bond_role()));
        let factor = builder.add_latent_ring_count_at_most(bonds, [budget]).unwrap();
        let model = Arc::new(builder.build());
        let native = NativeSolverState::with_transition(
            &NativeSolverState::initial(Arc::clone(&model)).unwrap(), [], [factor]).unwrap();
        let oracle = <crate::native_oracle::ExhaustiveSolverState as ConstraintSolver>::initial(model)
            .unwrap().unwrap_consistent().transitioned(&[], &[factor])
            .unwrap().unwrap_consistent();
        for variable in roles.iter().copied().chain([budget]) {
            assert_eq!(native.domain_bound(variable), oracle.domain_bound(variable));
        }
        assert_eq!(native.domain_bound(roles[0]), Some(BondRole::role_domain()));
        assert_eq!(native.domain_bound(roles[1]), Some(BondRole::role_domain()));
        assert_eq!(native.domain_bound(budget), Some(Domain::singleton(1).unwrap()));
    }

    #[test]
    fn active_ring_count_joins_correlated_roles_in_exact_projection() {
        let mut builder = ConstraintModelBuilder::new();
        let roles = (0..3).map(|_| builder.add_variable(BondRole::role_domain()).unwrap())
            .collect::<Vec<_>>();
        let budget = builder.add_variable(Domain::from_bits(3)).unwrap();
        builder.add_binary_relation(roles[0], roles[1], [(0, 0), (1, 1)]).unwrap();
        builder.add_binary_relation(roles[1], roles[2], [(0, 1), (1, 0)]).unwrap();
        let bonds = roles.iter().enumerate().map(|(i, &variable)| {
            SpanningTreeEdge::with_role_partition(variable, AtomId::new(0),
                AtomId::new((i + 1) as u32), EdgeRolePartition::bond_role())
        });
        let factor = builder.add_latent_ring_count_at_most(bonds, [budget]).unwrap();
        let model = Arc::new(builder.build());
        let initial = NativeSolverState::initial(Arc::clone(&model)).unwrap();
        assert!(roles.iter().all(|&role| initial.domain_bound(role) == Some(BondRole::role_domain())));
        let native = NativeSolverState::with_transition(&initial,
            [(budget, Domain::singleton(1).unwrap())], [factor]).unwrap();
        let oracle = <crate::native_oracle::ExhaustiveSolverState as ConstraintSolver>::initial(model)
            .unwrap().unwrap_consistent().transitioned(
                &[(budget, Domain::singleton(1).unwrap())], &[factor]
            ).unwrap().unwrap_consistent();
        for (variable, expected) in roles.into_iter().zip([0, 0, 1]) {
            assert_eq!(native.domain_bound(variable), Some(Domain::singleton(expected).unwrap()));
            assert_eq!(native.domain_bound(variable), oracle.domain_bound(variable));
        }
    }

    #[test]
    fn ring_count_budget_spans_more_than_sixty_four_incident_bonds() {
        let mut builder = ConstraintModelBuilder::new();
        let bonds = (0..70).map(|i| {
            let domain = if i < 2 { BondRole::Ring.singleton_domain() }
                else { BondRole::Traversal.singleton_domain() };
            let variable = builder.add_variable(domain).unwrap();
            SpanningTreeEdge::with_role_partition(variable, AtomId::new(0),
                AtomId::new((i + 1) as u32), EdgeRolePartition::bond_role())
        }).collect::<Vec<_>>();
        let bits = (0..7).map(|_| builder.add_variable(Domain::from_bits(3)).unwrap())
            .collect::<Vec<_>>();
        let factor = builder.add_latent_ring_count_at_most(bonds, bits.iter().copied()).unwrap();
        let initial = NativeSolverState::initial(Arc::new(builder.build())).unwrap();
        assert_eq!(initial.factor_is_active(factor), Some(false));
        let restrictions = bits.iter().enumerate().map(|(i, &variable)|
            (variable, Domain::singleton(u8::from(i == 0)).unwrap()));
        assert!(matches!(NativeSolverState::with_transition(&initial, restrictions, [factor]),
            Err(NativeSolverError::Contradiction)));
    }

    #[test]
    fn active_ring_count_projects_an_unfixed_budget_bit() {
        let mut builder = ConstraintModelBuilder::new();
        let role = builder.add_variable(BondRole::Ring.singleton_domain()).unwrap();
        let budget = builder.add_variable(Domain::from_bits(3)).unwrap();
        let factor = builder.add_latent_ring_count_at_most([
            SpanningTreeEdge::with_role_partition(role, AtomId::new(0), AtomId::new(1),
                EdgeRolePartition::bond_role())
        ], [budget]).unwrap();
        let model = Arc::new(builder.build());
        let native = NativeSolverState::with_transition(
            &NativeSolverState::initial(Arc::clone(&model)).unwrap(), [], [factor]
        ).unwrap();
        let oracle = <crate::native_oracle::ExhaustiveSolverState as ConstraintSolver>::initial(model)
            .unwrap().unwrap_consistent().transitioned(&[], &[factor])
            .unwrap().unwrap_consistent();
        assert_eq!(native.domain_bound(budget), Some(Domain::singleton(1).unwrap()));
        assert_eq!(native.domain_bound(budget), oracle.domain_bound(budget));
    }

    fn two_values() -> Domain {
        Domain::from_indices([0, 1]).unwrap()
    }

    fn equality_chain() -> (Arc<ConstraintModel>, [VariableId; 5]) {
        let mut builder = ConstraintModelBuilder::new();
        let variables: [VariableId; 5] =
            std::array::from_fn(|_| builder.add_variable(two_values()).unwrap());
        builder
            .add_binary_relation(variables[0], variables[1], [(0, 0), (1, 1)])
            .unwrap();
        builder
            .add_binary_relation(variables[1], variables[2], [(0, 0), (1, 1)])
            .unwrap();
        builder
            .add_binary_relation(variables[3], variables[4], [(0, 0), (1, 1)])
            .unwrap();
        (Arc::new(builder.build()), variables)
    }

    fn latent_layout_model() -> (
        Arc<ConstraintModel>,
        VariableId,
        VariableId,
        [VariableId; 3],
        FactorId,
    ) {
        let mut builder = ConstraintModelBuilder::new();
        let order = builder.add_variable(two_values()).unwrap();
        let pattern = builder
            .add_variable(Domain::from_indices(0_u8..8).unwrap())
            .unwrap();
        let bond_domain = Domain::from_indices([0, 1, 2]).unwrap();
        let partition = EdgeRolePartition::new(
            Domain::singleton(0).unwrap(),
            Domain::from_indices([1, 2]).unwrap(),
        );
        let bonds: [VariableId; 3] =
            std::array::from_fn(|_| builder.add_variable(bond_domain).unwrap());
        let rows = [
            Domain::singleton(0).unwrap(),
            Domain::singleton(1).unwrap(),
            Domain::empty(),
            Domain::empty(),
            Domain::empty(),
            Domain::empty(),
            Domain::empty(),
            Domain::empty(),
        ];
        let factor = builder
            .add_latent_tetrahedral_layout(
                order,
                pattern,
                bonds.iter().copied().enumerate().map(|(bit, variable)| {
                    TetrahedralLayoutBond::new(variable, partition, bit as u8)
                }),
                rows,
            )
            .unwrap();
        (Arc::new(builder.build()), order, pattern, bonds, factor)
    }

    fn domains_for(state: &NativeSolverState, variables: &[VariableId]) -> Vec<Domain> {
        variables
            .iter()
            .map(|variable| state.domain_bound(*variable).unwrap())
            .collect()
    }

    #[test]
    fn latent_layout_has_no_effect_until_atomic_activation() {
        let (model, order, pattern, bonds, factor) = latent_layout_model();
        let source = NativeSolverState::initial(model).unwrap();

        assert_eq!(source.factor_is_active(factor), Some(false));
        assert_eq!(source.domain_bound(pattern).unwrap().len(), 8);
        assert_eq!(source.domain_bound(bonds[1]).unwrap().len(), 3);

        let successor = source
            .with_transition([(order, Domain::singleton(0).unwrap())], [factor])
            .unwrap();
        assert_eq!(successor.factor_is_active(factor), Some(true));
        assert_eq!(successor.domain_bound(order), Some(Domain::singleton(0).unwrap()));
        assert_eq!(
            successor.domain_bound(pattern),
            Some(Domain::singleton(0).unwrap())
        );
        for bond in bonds {
            assert_eq!(successor.domain_bound(bond), Some(Domain::singleton(0).unwrap()));
        }

        assert_eq!(source.factor_is_active(factor), Some(false));
        assert_eq!(source.domain_bound(order), Some(two_values()));
    }

    #[test]
    fn inactive_layout_component_performs_no_exact_work() {
        let (model, _, _, bonds, _) = latent_layout_model();
        let source = NativeSolverState::initial(model).unwrap();
        source.reset_exact_run_counts();
        source.reset_mixed_search_branch_count();

        let successor = source
            .with_restrictions([(bonds[0], Domain::singleton(1).unwrap())])
            .unwrap();

        assert_eq!(
            successor.domain_bound(bonds[0]),
            Some(Domain::singleton(1).unwrap())
        );
        assert_eq!(successor.exact_run_counts(), (0, 0));
        assert_eq!(successor.mixed_search_branch_count(), 0);
    }

    #[test]
    fn factor_activation_is_idempotent_and_unknown_ids_fail() {
        let (model, _, _, _, factor) = latent_layout_model();
        let source = NativeSolverState::initial(model).unwrap();
        let active = source
            .with_transition(std::iter::empty::<(VariableId, Domain)>(), [factor, factor])
            .unwrap();
        let repeated = active
            .with_transition(std::iter::empty::<(VariableId, Domain)>(), [factor])
            .unwrap();

        assert_eq!(repeated.factor_is_active(factor), Some(true));
        assert!(matches!(
            repeated.with_transition(
                std::iter::empty::<(VariableId, Domain)>(),
                [FactorId::new(99)]
            ),
            Err(NativeSolverError::UnknownFactor(factor)) if factor == FactorId::new(99)
        ));
    }

    #[test]
    fn traversal_resolved_layout_avoids_mixed_search() {
        let mut builder = ConstraintModelBuilder::new();
        let order = builder.add_variable(two_values()).unwrap();
        let pattern = builder
            .add_variable(Domain::from_indices(0_u8..8).unwrap())
            .unwrap();
        let bonds: [VariableId; 3] =
            std::array::from_fn(|_| builder.add_variable(BondRole::role_domain()).unwrap());
        let factor = builder
            .add_latent_tetrahedral_layout(
                order,
                pattern,
                bonds.iter().copied().enumerate().map(|(bit, variable)| {
                    TetrahedralLayoutBond::new(variable, EdgeRolePartition::bond_role(), bit as u8)
                }),
                [
                    two_values(),
                    Domain::empty(),
                    Domain::empty(),
                    Domain::empty(),
                    Domain::empty(),
                    Domain::empty(),
                    Domain::empty(),
                    Domain::empty(),
                ],
            )
            .unwrap();
        let atoms: [crate::AtomId; 4] =
            std::array::from_fn(|index| crate::AtomId::new(index as u32));
        builder
            .add_spanning_tree(
                atoms,
                bonds.iter().copied().enumerate().map(|(index, variable)| {
                    SpanningTreeEdge::new(variable, atoms[0], atoms[index + 1])
                }),
            )
            .unwrap();
        let source = NativeSolverState::initial(Arc::new(builder.build())).unwrap();
        source.reset_exact_run_counts();
        source.reset_mixed_search_branch_count();

        let successor = source
            .with_transition(std::iter::empty::<(VariableId, Domain)>(), [factor])
            .unwrap();

        assert_eq!(successor.domain_bound(order), Some(two_values()));
        assert_eq!(
            successor.domain_bound(pattern),
            Some(Domain::singleton(0).unwrap())
        );
        assert_eq!(successor.exact_run_counts(), (0, 0));
        assert_eq!(successor.mixed_search_branch_count(), 0);
    }

    #[test]
    fn disjoint_traversal_resolved_layouts_avoid_mixed_search_together() {
        let mut builder = ConstraintModelBuilder::new();
        let orders: [VariableId; 2] =
            std::array::from_fn(|_| builder.add_variable(two_values()).unwrap());
        let patterns: [VariableId; 2] = std::array::from_fn(|_| {
            builder
                .add_variable(Domain::from_indices(0_u8..8).unwrap())
                .unwrap()
        });
        let bonds: [VariableId; 5] =
            std::array::from_fn(|_| builder.add_variable(BondRole::role_domain()).unwrap());
        let layout_bonds = [
            [bonds[0], bonds[1], bonds[2]],
            [bonds[0], bonds[3], bonds[4]],
        ];
        let factors: [FactorId; 2] = std::array::from_fn(|center| {
            builder
                .add_latent_tetrahedral_layout(
                    orders[center],
                    patterns[center],
                    layout_bonds[center]
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(bit, variable)| {
                            TetrahedralLayoutBond::new(
                                variable,
                                EdgeRolePartition::bond_role(),
                                bit as u8,
                            )
                        }),
                    [
                        two_values(),
                        Domain::empty(),
                        Domain::empty(),
                        Domain::empty(),
                        Domain::empty(),
                        Domain::empty(),
                        Domain::empty(),
                        Domain::empty(),
                    ],
                )
                .unwrap()
        });
        let atoms: [crate::AtomId; 6] =
            std::array::from_fn(|index| crate::AtomId::new(index as u32));
        let edges = [
            (bonds[0], atoms[0], atoms[1]),
            (bonds[1], atoms[0], atoms[2]),
            (bonds[2], atoms[0], atoms[3]),
            (bonds[3], atoms[1], atoms[4]),
            (bonds[4], atoms[1], atoms[5]),
        ];
        builder
            .add_spanning_tree(
                atoms,
                edges.map(|(variable, a, b)| SpanningTreeEdge::new(variable, a, b)),
            )
            .unwrap();
        let source = NativeSolverState::initial(Arc::new(builder.build())).unwrap();
        source.reset_exact_run_counts();
        source.reset_mixed_search_branch_count();

        let successor = source
            .with_transition(std::iter::empty::<(VariableId, Domain)>(), factors)
            .unwrap();

        for order in orders {
            assert_eq!(successor.domain_bound(order), Some(two_values()));
        }
        for pattern in patterns {
            assert_eq!(
                successor.domain_bound(pattern),
                Some(Domain::singleton(0).unwrap())
            );
        }
        assert_eq!(successor.exact_run_counts(), (0, 0));
        assert_eq!(successor.mixed_search_branch_count(), 0);
    }

    #[test]
    fn initial_state_propagates_all_prepared_relations() {
        let mut builder = ConstraintModelBuilder::new();
        let left = builder.add_variable(two_values()).unwrap();
        let right = builder.add_variable(two_values()).unwrap();
        builder.add_binary_relation(left, right, [(0, 1)]).unwrap();

        let state = NativeSolverState::initial(Arc::new(builder.build())).unwrap();

        assert_eq!(state.domain_bound(left), Some(Domain::singleton(0).unwrap()));
        assert_eq!(state.domain_bound(right), Some(Domain::singleton(1).unwrap()));
    }

    #[test]
    fn empty_prepared_relation_is_rejected_as_a_contradiction() {
        let mut builder = ConstraintModelBuilder::new();
        let left = builder.add_variable(two_values()).unwrap();
        let right = builder.add_variable(two_values()).unwrap();
        builder
            .add_binary_relation(left, right, std::iter::empty())
            .unwrap();

        assert!(matches!(
            NativeSolverState::initial(Arc::new(builder.build())),
            Err(NativeSolverError::Contradiction)
        ));
    }

    #[test]
    fn restriction_changes_only_the_affected_binary_component() {
        let (model, variables) = equality_chain();
        let source = NativeSolverState::initial(model).unwrap();

        let successor = source
            .with_restrictions([(variables[0], Domain::singleton(0).unwrap())])
            .unwrap();

        for variable in &variables[..3] {
            assert_eq!(
                successor.domain_bound(*variable),
                Some(Domain::singleton(0).unwrap())
            );
        }
        for variable in &variables[3..] {
            assert_eq!(successor.domain_bound(*variable), Some(two_values()));
        }
    }

    #[test]
    fn restriction_batch_updates_independent_components() {
        let (model, variables) = equality_chain();
        let source = NativeSolverState::initial(model).unwrap();

        let successor = source
            .with_restrictions([
                (variables[0], Domain::singleton(0).unwrap()),
                (variables[3], Domain::singleton(1).unwrap()),
            ])
            .unwrap();

        for variable in &variables[..3] {
            assert_eq!(
                successor.domain_bound(*variable),
                Some(Domain::singleton(0).unwrap())
            );
        }
        for variable in &variables[3..] {
            assert_eq!(
                successor.domain_bound(*variable),
                Some(Domain::singleton(1).unwrap())
            );
        }
    }

    #[test]
    fn repeated_restrictions_are_intersected_before_propagation() {
        let mut builder = ConstraintModelBuilder::new();
        let variable = builder
            .add_variable(Domain::from_indices([0, 1, 2]).unwrap())
            .unwrap();
        let source = NativeSolverState::initial(Arc::new(builder.build())).unwrap();

        let successor = source
            .with_restrictions([
                (variable, Domain::from_indices([0, 1]).unwrap()),
                (variable, Domain::from_indices([1, 2]).unwrap()),
            ])
            .unwrap();

        assert_eq!(
            successor.domain_bound(variable),
            Some(Domain::singleton(1).unwrap())
        );
    }

    #[test]
    fn conflicting_restrictions_leave_the_source_unchanged() {
        let mut builder = ConstraintModelBuilder::new();
        let variable = builder.add_variable(two_values()).unwrap();
        let source = NativeSolverState::initial(Arc::new(builder.build())).unwrap();
        let before = source.domain_bound(variable);

        assert!(matches!(
            source.with_restrictions([
                (variable, Domain::singleton(0).unwrap()),
                (variable, Domain::singleton(1).unwrap()),
            ]),
            Err(NativeSolverError::Contradiction)
        ));
        assert_eq!(source.domain_bound(variable), before);
    }

    #[test]
    fn empty_and_noop_restrictions_preserve_domains() {
        let mut builder = ConstraintModelBuilder::new();
        let variable = builder.add_variable(two_values()).unwrap();
        let source = NativeSolverState::initial(Arc::new(builder.build())).unwrap();

        let empty_successor = source
            .with_restrictions(std::iter::empty::<(VariableId, Domain)>())
            .unwrap();
        let noop_successor = source
            .with_restrictions([(variable, two_values())])
            .unwrap();

        assert_eq!(empty_successor.domain_bound(variable), source.domain_bound(variable));
        assert_eq!(noop_successor.domain_bound(variable), source.domain_bound(variable));
    }

    #[test]
    fn successor_shares_the_model_but_not_mutable_domains() {
        let (model, variables) = equality_chain();
        let source = NativeSolverState::initial(model).unwrap();
        let source_domains = domains_for(&source, &variables);

        let successor = source
            .with_restrictions([(variables[0], Domain::singleton(0).unwrap())])
            .unwrap();

        assert!(Arc::ptr_eq(&source.model, &successor.model));
        assert!(Arc::ptr_eq(&source.exact_plan, &successor.exact_plan));
        assert_eq!(domains_for(&source, &variables), source_domains);
        assert_ne!(domains_for(&successor, &variables), source_domains);
    }

    #[test]
    fn one_restriction_copies_only_its_domain_page() {
        let mut builder = ConstraintModelBuilder::new();
        let variables = (0..130)
            .map(|_| builder.add_variable(two_values()).unwrap())
            .collect::<Vec<_>>();
        let source = NativeSolverState::initial(Arc::new(builder.build())).unwrap();
        source.domains.reset_copy_counts();

        let successor = source
            .with_restrictions([(variables[0], Domain::singleton(0).unwrap())])
            .unwrap();

        assert_eq!(source.domains.copy_counts(), (1, 1));
        assert!(source
            .domains
            .shares_value_page_with(&successor.domains, variables[129].index()));
        assert_eq!(source.domain_bound(variables[0]), Some(two_values()));
        assert_eq!(
            successor.domain_bound(variables[0]),
            Some(Domain::singleton(0).unwrap())
        );
    }

    #[test]
    fn successful_restriction_preserves_the_complete_solver_contract() {
        let mut builder = ConstraintModelBuilder::new();
        let variables = (0..130)
            .map(|_| builder.add_variable(two_values()).unwrap())
            .collect::<Vec<_>>();
        builder
            .add_binary_relation(variables[0], variables[129], [(0, 0), (1, 1)])
            .unwrap();
        let model = Arc::new(builder.build());
        let source = NativeSolverState::initial(Arc::clone(&model)).unwrap();
        let allowed = Domain::singleton(0).unwrap();

        let successor = source.with_restrictions([(variables[0], allowed)]).unwrap();

        assert!(Arc::ptr_eq(&source.model, &successor.model));
        for &variable in &variables {
            let prepared = model.variable(variable).unwrap().initial_domain();
            let source_domain = source.domain_bound(variable).unwrap();
            let successor_domain = successor.domain_bound(variable).unwrap();
            assert!(!successor_domain.is_empty());
            assert!(successor_domain.is_subset_of(prepared));
            assert!(successor_domain.is_subset_of(source_domain));
        }
        assert!(successor
            .domain_bound(variables[0])
            .unwrap()
            .is_subset_of(allowed));
    }

    #[test]
    fn rejected_restriction_does_not_copy_a_dense_domain_successor() {
        let mut builder = ConstraintModelBuilder::new();
        let variables = (0..130)
            .map(|_| builder.add_variable(two_values()).unwrap())
            .collect::<Vec<_>>();
        builder
            .add_binary_relation(variables[0], variables[129], [(0, 0), (1, 1)])
            .unwrap();
        let source = NativeSolverState::initial(Arc::new(builder.build())).unwrap();
        source.domains.reset_copy_counts();

        assert!(matches!(
            source.with_restrictions([
                (variables[0], Domain::singleton(0).unwrap()),
                (variables[129], Domain::singleton(1).unwrap()),
            ]),
            Err(NativeSolverError::Contradiction)
        ));

        assert_eq!(source.domains.copy_counts(), (1, 2));
        assert_eq!(source.domain_bound(variables[0]), Some(two_values()));
        assert_eq!(source.domain_bound(variables[64]), Some(two_values()));
        assert_eq!(source.domain_bound(variables[129]), Some(two_values()));
    }

    #[test]
    fn failed_restriction_leaves_the_source_unchanged() {
        let (model, variables) = equality_chain();
        let source = NativeSolverState::initial(model).unwrap();
        let fixed = source
            .with_restrictions([(variables[0], Domain::singleton(0).unwrap())])
            .unwrap();
        let before = domains_for(&fixed, &variables);

        assert!(matches!(
            fixed.with_restrictions([(variables[1], Domain::singleton(1).unwrap())]),
            Err(NativeSolverError::Contradiction)
        ));
        assert_eq!(domains_for(&fixed, &variables), before);
    }

    #[test]
    fn domains_are_independent_of_restriction_order() {
        let (model, variables) = equality_chain();
        let source = NativeSolverState::initial(model).unwrap();

        let left_first = source
            .with_restrictions([(variables[0], Domain::singleton(0).unwrap())])
            .unwrap();
        let left_then_right = left_first
            .with_restrictions([(variables[2], Domain::singleton(0).unwrap())])
            .unwrap();
        let right_first = source
            .with_restrictions([(variables[2], Domain::singleton(0).unwrap())])
            .unwrap();
        let right_then_left = right_first
            .with_restrictions([(variables[0], Domain::singleton(0).unwrap())])
            .unwrap();

        assert_eq!(
            domains_for(&left_then_right, &variables),
            domains_for(&right_then_left, &variables)
        );
    }

    #[test]
    fn cyclic_arc_consistent_contradiction_is_rejected() {
        let mut builder = ConstraintModelBuilder::new();
        let variables: [VariableId; 3] =
            std::array::from_fn(|_| builder.add_variable(two_values()).unwrap());
        builder
            .add_binary_relation(variables[0], variables[1], [(0, 0), (1, 1)])
            .unwrap();
        builder
            .add_binary_relation(variables[1], variables[2], [(0, 0), (1, 1)])
            .unwrap();
        builder
            .add_binary_relation(variables[2], variables[0], [(0, 1), (1, 0)])
            .unwrap();

        assert!(matches!(
            NativeSolverState::initial(Arc::new(builder.build())),
            Err(NativeSolverError::Contradiction)
        ));
    }

    #[test]
    fn equality_chain_cannot_cross_a_triangle_spanning_tree() {
        let mut builder = ConstraintModelBuilder::new();
        let variables: [VariableId; 3] =
            std::array::from_fn(|_| builder.add_variable(BondRole::role_domain()).unwrap());
        builder
            .add_binary_relation(variables[0], variables[1], [(0, 0), (1, 1)])
            .unwrap();
        builder
            .add_binary_relation(variables[1], variables[2], [(0, 0), (1, 1)])
            .unwrap();
        let atoms = [
            crate::AtomId::new(0),
            crate::AtomId::new(1),
            crate::AtomId::new(2),
        ];
        builder
            .add_spanning_tree(
                atoms,
                [
                    SpanningTreeEdge::new(variables[0], atoms[0], atoms[1]),
                    SpanningTreeEdge::new(variables[1], atoms[1], atoms[2]),
                    SpanningTreeEdge::new(variables[2], atoms[2], atoms[0]),
                ],
            )
            .unwrap();

        assert!(matches!(
            NativeSolverState::initial(Arc::new(builder.build())),
            Err(NativeSolverError::Contradiction)
        ));
    }

    #[test]
    fn mixed_search_branches_only_on_the_semantic_core() {
        const EDGE_COUNT: usize = 100;

        let mut builder = ConstraintModelBuilder::new();
        let variables = (0..EDGE_COUNT)
            .map(|_| builder.add_variable(BondRole::role_domain()).unwrap())
            .collect::<Vec<_>>();
        builder
            .add_binary_relation(variables[0], variables[50], [(0, 0), (1, 1)])
            .unwrap();
        let atoms = (0..EDGE_COUNT)
            .map(|index| crate::AtomId::new(u32::try_from(index).unwrap()))
            .collect::<Vec<_>>();
        let edges = (0..EDGE_COUNT)
            .map(|index| {
                SpanningTreeEdge::new(
                    variables[index],
                    atoms[index],
                    atoms[(index + 1) % EDGE_COUNT],
                )
            })
            .collect::<Vec<_>>();
        builder.add_spanning_tree(atoms, edges).unwrap();

        let state = NativeSolverState::initial(Arc::new(builder.build())).unwrap();

        assert_eq!(
            state.domain_bound(variables[0]),
            Some(BondRole::Traversal.singleton_domain())
        );
        assert_eq!(
            state.domain_bound(variables[50]),
            Some(BondRole::Traversal.singleton_domain())
        );
        assert_eq!(state.mixed_search_branch_count(), 1);
        state.reset_mixed_search_branch_count();
        assert_eq!(state.mixed_search_branch_count(), 0);
    }

    #[test]
    fn pure_spanning_model_has_no_semantic_exact_descriptor_or_run() {
        const EDGE_COUNT: usize = 100;

        let mut builder = ConstraintModelBuilder::new();
        let variables = (0..EDGE_COUNT)
            .map(|_| builder.add_variable(BondRole::role_domain()).unwrap())
            .collect::<Vec<_>>();
        let atoms = (0..EDGE_COUNT)
            .map(|index| crate::AtomId::new(u32::try_from(index).unwrap()))
            .collect::<Vec<_>>();
        let edges = (0..EDGE_COUNT)
            .map(|index| {
                SpanningTreeEdge::new(
                    variables[index],
                    atoms[index],
                    atoms[(index + 1) % EDGE_COUNT],
                )
            })
            .collect::<Vec<_>>();
        builder.add_spanning_tree(atoms, edges).unwrap();

        let state = NativeSolverState::initial(Arc::new(builder.build())).unwrap();

        assert!(state.exact_plan.components.is_empty());
        assert_eq!(state.exact_run_counts(), (0, 0));
    }

    #[test]
    fn acyclic_binary_components_need_no_exact_descriptor() {
        let (model, _) = equality_chain();

        let state = NativeSolverState::initial(model).unwrap();

        assert!(state.exact_plan.components.is_empty());
        assert_eq!(state.exact_run_counts(), (0, 0));
    }

    #[test]
    fn cyclic_binary_component_is_compiled_once() {
        let mut builder = ConstraintModelBuilder::new();
        let variables: [VariableId; 3] =
            std::array::from_fn(|_| builder.add_variable(two_values()).unwrap());
        for (left, right) in [(0, 1), (1, 2), (2, 0)] {
            builder
                .add_binary_relation(variables[left], variables[right], [(0, 0), (1, 1)])
                .unwrap();
        }

        let state = NativeSolverState::initial(Arc::new(builder.build())).unwrap();

        assert_eq!(state.exact_plan.components.len(), 1);
        assert!(matches!(
            state.exact_plan.components[0],
            ExactComponent::Binary(_)
        ));
        assert_eq!(state.exact_run_counts(), (1, 0));
    }

    #[test]
    fn one_spanning_projector_joins_binary_disconnected_semantic_relations() {
        let mut builder = ConstraintModelBuilder::new();
        let variables: [VariableId; 4] =
            std::array::from_fn(|_| builder.add_variable(BondRole::role_domain()).unwrap());
        builder
            .add_binary_relation(variables[0], variables[1], [(0, 0), (1, 1)])
            .unwrap();
        builder
            .add_binary_relation(variables[2], variables[3], [(0, 1), (1, 0)])
            .unwrap();
        let atoms: [crate::AtomId; 4] =
            std::array::from_fn(|index| crate::AtomId::new(u32::try_from(index).unwrap()));
        builder
            .add_spanning_tree(
                atoms,
                (0..4).map(|index| {
                    SpanningTreeEdge::new(variables[index], atoms[index], atoms[(index + 1) % 4])
                }),
            )
            .unwrap();

        let state = NativeSolverState::initial(Arc::new(builder.build())).unwrap();

        assert_eq!(state.exact_plan.components.len(), 1);
        assert!(matches!(
            state.exact_plan.components[0],
            ExactComponent::Mixed(_)
        ));
        let component = state.exact_plan.component_for(variables[0]).unwrap();
        assert!(variables
            .iter()
            .all(|variable| state.exact_plan.component_for(*variable) == Some(component)));
    }

    #[test]
    fn rooted_forest_root_variables_participate_in_mixed_projection() {
        let mut builder = ConstraintModelBuilder::new();
        let atoms = [crate::AtomId::new(0), crate::AtomId::new(1)];
        let bond = builder.add_variable(BondRole::role_domain()).unwrap();
        let roots = atoms.map(|_| builder.add_variable(Domain::from_indices([0, 1]).unwrap()).unwrap());
        let signal = builder.add_variable(Domain::from_indices([0, 1]).unwrap()).unwrap();
        for root in roots {
            builder.add_binary_relation(root, signal, [(0, 0), (1, 1)]).unwrap();
        }
        builder.add_rooted_spanning_forest(
            atoms,
            [SpanningTreeEdge::new(bond, atoms[0], atoms[1])],
            atoms.into_iter().zip(roots),
        ).unwrap();
        let state = NativeSolverState::initial(Arc::new(builder.build())).unwrap();
        let component = state.exact_plan.component_for(bond).unwrap();
        assert!(matches!(state.exact_plan.components[component], ExactComponent::Mixed(_)));
        assert!(roots.iter().chain([signal].iter()).all(|&variable|
            state.exact_plan.component_for(variable) == Some(component)));

        let forced_traversal = crate::solver::ConstraintSolver::restricted(&state, &[(bond, BondRole::Traversal.singleton_domain())])
            .unwrap().unwrap_consistent();
        assert_eq!(forced_traversal.domain_bound(signal), Some(Domain::singleton(0).unwrap()));
        let selected = crate::solver::ConstraintSolver::restricted(&state, &[(signal, Domain::singleton(1).unwrap())])
            .unwrap().unwrap_consistent();
        assert_eq!(selected.domain_bound(bond), Some(BondRole::Ring.singleton_domain()));
    }

    #[test]
    fn binary_relation_joins_multiple_spanning_projectors() {
        let mut builder = ConstraintModelBuilder::new();
        let variables: [VariableId; 6] =
            std::array::from_fn(|_| builder.add_variable(BondRole::role_domain()).unwrap());
        let atoms: [crate::AtomId; 6] =
            std::array::from_fn(|index| crate::AtomId::new(u32::try_from(index).unwrap()));
        builder
            .add_binary_relation(variables[0], variables[3], [(0, 0), (0, 1), (1, 0), (1, 1)])
            .unwrap();
        for offset in [0, 3] {
            builder
                .add_spanning_tree(
                    atoms[offset..offset + 3].iter().copied(),
                    [
                        SpanningTreeEdge::new(variables[offset], atoms[offset], atoms[offset + 1]),
                        SpanningTreeEdge::new(
                            variables[offset + 1],
                            atoms[offset + 1],
                            atoms[offset + 2],
                        ),
                        SpanningTreeEdge::new(
                            variables[offset + 2],
                            atoms[offset + 2],
                            atoms[offset],
                        ),
                    ],
                )
                .unwrap();
        }

        let state = NativeSolverState::initial(Arc::new(builder.build())).unwrap();

        assert_eq!(state.exact_plan.components.len(), 1);
        assert!(matches!(
            state.exact_plan.components[0],
            ExactComponent::Mixed(_)
        ));
        let component = state.exact_plan.component_for(variables[0]).unwrap();
        assert!(variables
            .iter()
            .all(|variable| state.exact_plan.component_for(*variable) == Some(component)));
    }

    #[test]
    fn restriction_runs_only_its_compiled_mixed_component() {
        let mut builder = ConstraintModelBuilder::new();
        let variables: [VariableId; 6] =
            std::array::from_fn(|_| builder.add_variable(BondRole::role_domain()).unwrap());
        let atoms: [crate::AtomId; 6] =
            std::array::from_fn(|index| crate::AtomId::new(u32::try_from(index).unwrap()));
        for offset in [0, 3] {
            builder
                .add_binary_relation(
                    variables[offset],
                    variables[offset + 1],
                    [(0, 0), (0, 1), (1, 0), (1, 1)],
                )
                .unwrap();
            builder
                .add_spanning_tree(
                    atoms[offset..offset + 3].iter().copied(),
                    [
                        SpanningTreeEdge::new(variables[offset], atoms[offset], atoms[offset + 1]),
                        SpanningTreeEdge::new(
                            variables[offset + 1],
                            atoms[offset + 1],
                            atoms[offset + 2],
                        ),
                        SpanningTreeEdge::new(
                            variables[offset + 2],
                            atoms[offset + 2],
                            atoms[offset],
                        ),
                    ],
                )
                .unwrap();
        }
        let state = NativeSolverState::initial(Arc::new(builder.build())).unwrap();
        assert_eq!(state.exact_plan.components.len(), 2);
        assert_eq!(
            state.exact_plan.component_for(variables[0]),
            state.exact_plan.component_for(variables[2]),
            "a structural-only projected variable must route to the mixed descriptor"
        );
        assert_ne!(
            state.exact_plan.component_for(variables[0]),
            state.exact_plan.component_for(variables[3])
        );
        state.reset_exact_run_counts();

        let successor = state
            .with_restrictions([(variables[0], BondRole::Traversal.singleton_domain())])
            .unwrap();

        assert_eq!(successor.exact_run_counts(), (0, 1));
        assert_eq!(
            successor.domain_bound(variables[3]),
            Some(BondRole::role_domain())
        );

        state.reset_exact_run_counts();
        let structural_successor = state
            .with_restrictions([(variables[2], BondRole::Ring.singleton_domain())])
            .unwrap();
        assert_eq!(structural_successor.exact_run_counts(), (0, 1));
    }

    #[test]
    fn cyclic_search_removes_globally_unsupported_values() {
        let mut builder = ConstraintModelBuilder::new();
        let x = builder
            .add_variable(Domain::from_indices([0, 1, 2]).unwrap())
            .unwrap();
        let y = builder.add_variable(two_values()).unwrap();
        let z = builder.add_variable(two_values()).unwrap();
        builder
            .add_binary_relation(x, y, [(0, 0), (1, 1), (2, 0)])
            .unwrap();
        builder.add_binary_relation(y, z, [(0, 0), (1, 1)]).unwrap();
        builder
            .add_binary_relation(z, x, [(0, 0), (1, 1), (1, 2)])
            .unwrap();

        let state = NativeSolverState::initial(Arc::new(builder.build())).unwrap();

        assert_eq!(state.domain_bound(x), Some(Domain::from_indices([0, 1]).unwrap()));
        assert_eq!(state.domain_bound(y), Some(two_values()));
        assert_eq!(state.domain_bound(z), Some(two_values()));
    }

    #[test]
    fn restriction_after_exact_filtering_produces_exact_successor() {
        let mut builder = ConstraintModelBuilder::new();
        let x = builder
            .add_variable(Domain::from_indices([0, 1, 2]).unwrap())
            .unwrap();
        let y = builder.add_variable(two_values()).unwrap();
        let z = builder.add_variable(two_values()).unwrap();
        builder
            .add_binary_relation(x, y, [(0, 0), (1, 1), (2, 0)])
            .unwrap();
        builder.add_binary_relation(y, z, [(0, 0), (1, 1)]).unwrap();
        builder
            .add_binary_relation(z, x, [(0, 0), (1, 1), (1, 2)])
            .unwrap();
        let source = NativeSolverState::initial(Arc::new(builder.build())).unwrap();

        let successor = source
            .with_restrictions([(x, Domain::singleton(0).unwrap())])
            .unwrap();

        assert_eq!(successor.domain_bound(x), Some(Domain::singleton(0).unwrap()));
        assert_eq!(successor.domain_bound(y), Some(Domain::singleton(0).unwrap()));
        assert_eq!(successor.domain_bound(z), Some(Domain::singleton(0).unwrap()));
    }

    #[test]
    fn unknown_variable_is_rejected() {
        let state = NativeSolverState::initial(Arc::new(ConstraintModel::empty())).unwrap();
        let variable = VariableId::new(7);

        assert!(matches!(
            state.with_restrictions([(variable, Domain::singleton(0).unwrap())]),
            Err(NativeSolverError::UnknownVariable(found)) if found == variable
        ));
    }

    #[test]
    fn unknown_variable_precedes_batch_contradiction() {
        let mut builder = ConstraintModelBuilder::new();
        let variable = builder.add_variable(two_values()).unwrap();
        let state = NativeSolverState::initial(Arc::new(builder.build())).unwrap();
        let unknown = VariableId::new(99);

        assert!(matches!(
            state.with_restrictions([
                (variable, Domain::empty()),
                (unknown, Domain::singleton(0).unwrap()),
            ]),
            Err(NativeSolverError::UnknownVariable(found)) if found == unknown
        ));
    }
}

#[cfg(test)]
mod aromatic_matching_plan_tests {
    use super::*;
    use crate::model::{AromaticMatchingEdge, ConstraintModelBuilder};

    #[test]
    fn aromatic_matching_initial_and_transition_never_enter_exact_search_plan() {
        let mut builder = ConstraintModelBuilder::new();
        let modes = (0..6).map(|_| builder.add_variable(Domain::from_bits(3)).unwrap()).collect::<Vec<_>>();
        let edges = (0..6)
            .map(|_| builder.add_variable(Domain::from_bits(7)).unwrap())
            .collect::<Vec<_>>();
        builder
            .add_aromatic_matching(
                modes.clone(),
                true,
                [1; 6],
                edges.iter().enumerate().map(|(index, &variable)| {
                    AromaticMatchingEdge::new(variable, index, (index + 1) % 6)
                }),
            )
            .unwrap();
        let initial = NativeSolverState::initial(Arc::new(builder.build())).unwrap();
        assert!(initial.exact_plan.components.is_empty());
        assert!(initial
            .exact_plan
            .component_by_variable
            .iter()
            .all(Option::is_none));
        let next = initial
            .with_restrictions([(edges[0], Domain::from_bits(4))])
            .unwrap();
        assert_eq!(next.domain_bound(modes[0]), Some(Domain::from_bits(2)));
        assert_eq!(next.domain_bound(edges[1]), Some(Domain::from_bits(2)));
        assert_eq!(next.binary_exact_runs.load(Ordering::Relaxed), 0);
        assert_eq!(next.mixed_exact_runs.load(Ordering::Relaxed), 0);
        assert_eq!(next.mixed_search_branches.load(Ordering::Relaxed), 0);
        assert_eq!(initial.domain_bound(modes[0]), Some(Domain::from_bits(3)));
    }
}
