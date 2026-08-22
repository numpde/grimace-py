//! Native finite-domain solving for South Star 2.
//!
//! Factor propagation is seeded only from changed variables. Binary relation
//! components retain exact finite-domain filtering; the spanning-tree factor has
//! its own exact graphic-matroid projection and never enumerates spanning trees.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::domain::Domain;
use crate::ids::{FactorId, VariableId};
use crate::model::{
    BinaryRelationFactor, BondRole, ConstraintModel, FactorDefinition, SpanningTreeFactor,
};
use crate::persistent::PagedStore;

#[derive(Clone, Debug)]
pub(crate) struct NativeSolverState {
    model: Arc<ConstraintModel>,
    exact_plan: Arc<NativeExactPlan>,
    domains: PagedStore<Domain>,
    #[cfg(test)]
    mixed_search_branches: Arc<AtomicUsize>,
    #[cfg(test)]
    binary_exact_runs: Arc<AtomicUsize>,
    #[cfg(test)]
    mixed_exact_runs: Arc<AtomicUsize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum NativeSolverError {
    UnknownVariable(VariableId),
    Contradiction,
}

impl fmt::Display for NativeSolverError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownVariable(variable) => {
                write!(formatter, "unknown constraint variable {variable:?}")
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
        let factor_count = model.factor_count();
        let variable_count = model.variable_count();
        let mut state = Self {
            model,
            exact_plan,
            domains,
            #[cfg(test)]
            mixed_search_branches: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            binary_exact_runs: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            mixed_exact_runs: Arc::new(AtomicUsize::new(0)),
        };
        state.enforce_consistency(
            (0..factor_count).map(factor_id_from_index),
            (0..variable_count).map(variable_id_from_index),
        )?;
        Ok(state)
    }

    pub(crate) fn domain(&self, variable: VariableId) -> Option<Domain> {
        self.domains.get(variable.index()).copied()
    }

    #[cfg(test)]
    fn reset_mixed_search_branch_count(&self) {
        self.mixed_search_branches.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn mixed_search_branch_count(&self) -> usize {
        self.mixed_search_branches.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    fn reset_exact_run_counts(&self) {
        self.binary_exact_runs.store(0, Ordering::Relaxed);
        self.mixed_exact_runs.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn exact_run_counts(&self) -> (usize, usize) {
        (
            self.binary_exact_runs.load(Ordering::Relaxed),
            self.mixed_exact_runs.load(Ordering::Relaxed),
        )
    }

    /// Return one atomically restricted and propagated successor.
    ///
    /// Repeated restrictions for the same variable are intersected before any
    /// candidate state is created. The source state is never mutated.
    pub(crate) fn with_restrictions(
        &self,
        restrictions: impl IntoIterator<Item = (VariableId, Domain)>,
    ) -> Result<Self, NativeSolverError> {
        let mut domains_by_variable = BTreeMap::new();
        let mut contradictory = false;
        for (variable, allowed) in restrictions {
            let current = self
                .domain(variable)
                .ok_or(NativeSolverError::UnknownVariable(variable))?;
            let entry = domains_by_variable.entry(variable).or_insert(current);
            *entry = (*entry).intersect(allowed);
            contradictory |= entry.is_empty();
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
            seed_factors.extend(
                successor
                    .model
                    .factors_for_variable(variable)
                    .expect("known variable must have an adjacency row")
                    .iter()
                    .copied(),
            );
        }

        if changed_variables.is_empty() {
            return Ok(successor);
        }

        successor.enforce_consistency(seed_factors, changed_variables)?;
        Ok(successor)
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
                seed_factors.extend(
                    self.model
                        .factors_for_variable(*variable)
                        .expect("known variable must have an adjacency row")
                        .iter()
                        .copied(),
                );
            }

            let propagated_reductions = self.propagate(seed_factors)?;
            exact_seeds = exact_reductions
                .into_iter()
                .chain(propagated_reductions)
                .collect();
        }

        Ok(())
    }

    fn propagate(
        &mut self,
        seeds: impl IntoIterator<Item = FactorId>,
    ) -> Result<BTreeSet<VariableId>, NativeSolverError> {
        let mut queue = VecDeque::new();
        let mut queued = BTreeSet::new();
        let mut all_reductions = BTreeSet::new();

        for factor in seeds {
            enqueue_factor(factor, &mut queue, &mut queued);
        }

        while let Some(factor_id) = queue.pop_front() {
            queued.remove(&factor_id);

            let factor = self
                .model
                .factor(factor_id)
                .expect("factor adjacency must reference a prepared factor");
            let reductions = match factor {
                FactorDefinition::BinaryRelation(relation) => {
                    revise_binary_relation(relation, &mut self.domains)?
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>()
                }
                FactorDefinition::SpanningTree(spanning_tree) => {
                    revise_spanning_tree(spanning_tree, &mut self.domains)?
                }
            };

            for variable in reductions {
                all_reductions.insert(variable);
                for &neighbour in self
                    .model
                    .factors_for_variable(variable)
                    .expect("factor scope must reference a prepared variable")
                {
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
                    #[cfg(test)]
                    self.mixed_exact_runs.fetch_add(1, Ordering::Relaxed);
                    reductions.extend(self.complete_filter_mixed_component(component)?);
                }
            }
        }

        Ok(reductions)
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
    ) -> Result<Vec<VariableId>, NativeSolverError> {
        if !component
            .core_variables
            .iter()
            .any(|variable| self.domains[variable.index()].len() > 1)
        {
            return Ok(Vec::new());
        }

        let target = component
            .projected_variables
            .iter()
            .map(|variable| self.domains[variable.index()])
            .collect::<Vec<_>>();
        let mut supported = vec![Domain::empty(); component.projected_variables.len()];
        let mut stack = vec![self.clone()];
        let mut found_solution = false;

        while let Some(candidate) = stack.pop() {
            let Some(variable) = component
                .core_variables
                .iter()
                .copied()
                .find(|variable| candidate.domains[variable.index()].len() > 1)
            else {
                found_solution = true;
                for (accumulator, variable) in supported
                    .iter_mut()
                    .zip(component.projected_variables.iter().copied())
                {
                    *accumulator = accumulator.union(candidate.domains[variable.index()]);
                }
                if supported == target {
                    break;
                }
                continue;
            };

            #[cfg(test)]
            candidate
                .mixed_search_branches
                .fetch_add(1, Ordering::Relaxed);

            let mut values = candidate.domains[variable.index()]
                .iter()
                .collect::<Vec<_>>();
            values.reverse();
            for value in values {
                match candidate
                    .restricted_and_propagated(variable, Domain::from_bits(1_u64 << value))
                {
                    Ok(child) => stack.push(child),
                    Err(NativeSolverError::Contradiction) => {}
                    Err(failure) => return Err(failure),
                }
            }
        }

        if !found_solution {
            return Err(NativeSolverError::Contradiction);
        }

        let mut reductions = Vec::new();
        for (variable, supported_domain) in
            component.projected_variables.iter().copied().zip(supported)
        {
            let current = self.domains[variable.index()];
            debug_assert!(supported_domain.is_subset_of(current));
            if supported_domain != current {
                self.domains[variable.index()] = supported_domain;
                reductions.push(variable);
            }
        }
        Ok(reductions)
    }

    fn restricted_and_propagated(
        &self,
        variable: VariableId,
        allowed: Domain,
    ) -> Result<Self, NativeSolverError> {
        let current = self
            .domain(variable)
            .ok_or(NativeSolverError::UnknownVariable(variable))?;
        let restricted = current.intersect(allowed);
        if restricted.is_empty() {
            return Err(NativeSolverError::Contradiction);
        }
        if restricted == current {
            return Ok(self.clone());
        }

        let mut successor = self.clone();
        successor.domains[variable.index()] = restricted;
        let seed_factors = successor
            .model
            .factors_for_variable(variable)
            .expect("known variable must have an adjacency row")
            .to_vec();
        successor.propagate(seed_factors)?;
        Ok(successor)
    }
}

#[derive(Debug)]
struct NativeExactPlan {
    components: Box<[ExactComponent]>,
    component_by_variable: Box<[Option<usize>]>,
}

impl NativeExactPlan {
    fn compile(model: &ConstraintModel) -> Self {
        let variable_count = model.variable_count();
        let has_binary_factor = (0..variable_count)
            .map(|index| {
                let variable = variable_id_from_index(index);
                model
                    .factors_for_variable(variable)
                    .expect("known variable must have an adjacency row")
                    .iter()
                    .any(|factor_id| {
                        matches!(
                            model
                                .factor(*factor_id)
                                .expect("factor adjacency must reference a prepared factor"),
                            FactorDefinition::BinaryRelation(_)
                        )
                    })
            })
            .collect::<Vec<_>>();
        let mut visited_core = vec![false; variable_count];
        let mut components = Vec::new();
        let mut component_by_variable = vec![None; variable_count];

        for seed_index in 0..variable_count {
            if !has_binary_factor[seed_index] || visited_core[seed_index] {
                continue;
            }

            let mut core_variables = BTreeSet::new();
            let mut binary_factors = BTreeSet::new();
            let mut structural_factors = BTreeSet::new();
            let mut pending = VecDeque::from([variable_id_from_index(seed_index)]);

            while let Some(variable) = pending.pop_front() {
                if !core_variables.insert(variable) {
                    continue;
                }
                visited_core[variable.index()] = true;

                for &factor_id in model
                    .factors_for_variable(variable)
                    .expect("known variable must have an adjacency row")
                {
                    match model
                        .factor(factor_id)
                        .expect("factor adjacency must reference a prepared factor")
                    {
                        FactorDefinition::BinaryRelation(relation) => {
                            if binary_factors.insert(factor_id) {
                                pending.extend(relation.variables().iter().copied());
                            }
                        }
                        FactorDefinition::SpanningTree(spanning_tree) => {
                            if structural_factors.insert(factor_id) {
                                pending.extend(
                                    spanning_tree
                                        .variables()
                                        .iter()
                                        .copied()
                                        .filter(|candidate| has_binary_factor[candidate.index()]),
                                );
                            }
                        }
                    }
                }
            }

            let core_variables = core_variables.into_iter().collect::<Vec<_>>();
            let component = if structural_factors.is_empty() {
                // Arc consistency is exact on an acyclic binary incidence
                // graph, so only cyclic pure-binary components need search.
                (binary_factors.len() >= core_variables.len()).then(|| {
                    ExactComponent::Binary(BinaryExactComponent::compile(
                        model,
                        core_variables,
                        binary_factors.into_iter().collect(),
                    ))
                })
            } else {
                let mut projected_variables =
                    core_variables.iter().copied().collect::<BTreeSet<_>>();
                for factor_id in structural_factors {
                    let FactorDefinition::SpanningTree(spanning_tree) = model
                        .factor(factor_id)
                        .expect("prepared structural factor must resolve")
                    else {
                        unreachable!("mixed component structural IDs must be spanning factors");
                    };
                    projected_variables.extend(spanning_tree.variables().iter().copied());
                }
                Some(ExactComponent::Mixed(MixedExactComponent {
                    core_variables: core_variables.into_boxed_slice(),
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
    core_variables: Box<[VariableId]>,
    projected_variables: Box<[VariableId]>,
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

fn revise_spanning_tree(
    factor: &SpanningTreeFactor,
    domains: &mut PagedStore<Domain>,
) -> Result<Vec<VariableId>, NativeSolverError> {
    let traversal = BondRole::Traversal;
    let ring = BondRole::Ring;
    let mut components = DisjointSet::new(factor.atoms().len());

    // Forced traversal edges are the independent set that the remaining
    // graphic-matroid basis must extend. A cycle makes extension impossible.
    for edge in factor.edges() {
        let role = domains[edge.role_variable().index()];
        match role_membership(role) {
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

    let mut quotient_by_root = BTreeMap::new();
    for atom in 0..factor.atoms().len() {
        let root = components.find(atom);
        if !quotient_by_root.contains_key(&root) {
            let quotient = quotient_by_root.len();
            quotient_by_root.insert(root, quotient);
        }
    }

    let mut reductions = Vec::new();
    let mut quotient_edges = Vec::new();

    for edge in factor.edges() {
        let variable = edge.role_variable();
        let role = domains[variable.index()];
        let membership = role_membership(role);
        if membership == (true, false) || membership == (false, true) {
            continue;
        }
        if membership != (true, true) {
            return Err(NativeSolverError::Contradiction);
        }

        let a_root = components.find(factor_atom_index(factor, edge.a()));
        let b_root = components.find(factor_atom_index(factor, edge.b()));
        if a_root == b_root {
            domains[variable.index()] = ring.singleton_domain();
            reductions.push(variable);
            continue;
        }

        quotient_edges.push(QuotientEdge {
            role_variable: variable,
            a: quotient_by_root[&a_root],
            b: quotient_by_root[&b_root],
        });
    }

    let quotient_node_count = quotient_by_root.len();
    let bridges = quotient_bridges(quotient_node_count, &quotient_edges)?;
    for (edge, is_bridge) in quotient_edges.iter().zip(bridges) {
        if is_bridge {
            domains[edge.role_variable.index()] = traversal.singleton_domain();
            reductions.push(edge.role_variable);
        }
    }

    Ok(reductions)
}

fn role_membership(domain: Domain) -> (bool, bool) {
    (
        domain.contains(BondRole::Traversal.value_index()),
        domain.contains(BondRole::Ring.value_index()),
    )
}

fn factor_atom_index(factor: &SpanningTreeFactor, atom: crate::AtomId) -> usize {
    factor
        .atoms()
        .binary_search(&atom)
        .expect("spanning-tree edge endpoints must belong to the factor atom set")
}

#[derive(Copy, Clone, Debug)]
struct QuotientEdge {
    role_variable: VariableId,
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
    use crate::model::{ConstraintModelBuilder, SpanningTreeEdge};

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

    fn domains_for(state: &NativeSolverState, variables: &[VariableId]) -> Vec<Domain> {
        variables
            .iter()
            .map(|variable| state.domain(*variable).unwrap())
            .collect()
    }

    #[test]
    fn initial_state_propagates_all_prepared_relations() {
        let mut builder = ConstraintModelBuilder::new();
        let left = builder.add_variable(two_values()).unwrap();
        let right = builder.add_variable(two_values()).unwrap();
        builder.add_binary_relation(left, right, [(0, 1)]).unwrap();

        let state = NativeSolverState::initial(Arc::new(builder.build())).unwrap();

        assert_eq!(state.domain(left), Some(Domain::singleton(0).unwrap()));
        assert_eq!(state.domain(right), Some(Domain::singleton(1).unwrap()));
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
                successor.domain(*variable),
                Some(Domain::singleton(0).unwrap())
            );
        }
        for variable in &variables[3..] {
            assert_eq!(successor.domain(*variable), Some(two_values()));
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
                successor.domain(*variable),
                Some(Domain::singleton(0).unwrap())
            );
        }
        for variable in &variables[3..] {
            assert_eq!(
                successor.domain(*variable),
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
            successor.domain(variable),
            Some(Domain::singleton(1).unwrap())
        );
    }

    #[test]
    fn conflicting_restrictions_leave_the_source_unchanged() {
        let mut builder = ConstraintModelBuilder::new();
        let variable = builder.add_variable(two_values()).unwrap();
        let source = NativeSolverState::initial(Arc::new(builder.build())).unwrap();
        let before = source.domain(variable);

        assert!(matches!(
            source.with_restrictions([
                (variable, Domain::singleton(0).unwrap()),
                (variable, Domain::singleton(1).unwrap()),
            ]),
            Err(NativeSolverError::Contradiction)
        ));
        assert_eq!(source.domain(variable), before);
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

        assert_eq!(empty_successor.domain(variable), source.domain(variable));
        assert_eq!(noop_successor.domain(variable), source.domain(variable));
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
        assert_eq!(source.domain(variables[0]), Some(two_values()));
        assert_eq!(
            successor.domain(variables[0]),
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
            let source_domain = source.domain(variable).unwrap();
            let successor_domain = successor.domain(variable).unwrap();
            assert!(!successor_domain.is_empty());
            assert!(successor_domain.is_subset_of(prepared));
            assert!(successor_domain.is_subset_of(source_domain));
        }
        assert!(successor
            .domain(variables[0])
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
        assert_eq!(source.domain(variables[0]), Some(two_values()));
        assert_eq!(source.domain(variables[64]), Some(two_values()));
        assert_eq!(source.domain(variables[129]), Some(two_values()));
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
            state.domain(variables[0]),
            Some(BondRole::Traversal.singleton_domain())
        );
        assert_eq!(
            state.domain(variables[50]),
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
            successor.domain(variables[3]),
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

        assert_eq!(state.domain(x), Some(Domain::from_indices([0, 1]).unwrap()));
        assert_eq!(state.domain(y), Some(two_values()));
        assert_eq!(state.domain(z), Some(two_values()));
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

        assert_eq!(successor.domain(x), Some(Domain::singleton(0).unwrap()));
        assert_eq!(successor.domain(y), Some(Domain::singleton(0).unwrap()));
        assert_eq!(successor.domain(z), Some(Domain::singleton(0).unwrap()));
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
