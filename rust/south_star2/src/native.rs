//! Native finite-domain solving for South Star 2.
//!
//! All prepared factors are active in the current model. The solver maintains
//! arc consistency after local restrictions and performs exact component-local
//! search when a cyclic binary factor component requires it.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::{FactorId, VariableId};
use crate::model::{BinaryRelationFactor, ConstraintModel, FactorDefinition};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ConstraintStateSnapshot {
    domains: Box<[Domain]>,
}

#[derive(Clone, Debug)]
pub(crate) struct NativeSolverState {
    model: Arc<ConstraintModel>,
    domains: Box<[Domain]>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PropagationSummary {
    factor_revisions: usize,
    distinct_factors_visited: usize,
    domain_reductions: usize,
}

impl PropagationSummary {
    pub(crate) const fn factor_revisions(self) -> usize {
        self.factor_revisions
    }

    pub(crate) const fn distinct_factors_visited(self) -> usize {
        self.distinct_factors_visited
    }

    pub(crate) const fn domain_reductions(self) -> usize {
        self.domain_reductions
    }
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
        let domains = model.initial_domains().collect::<Vec<_>>().into_boxed_slice();
        let factor_count = model.factor_count();
        let variable_count = model.variable_count();
        let mut state = Self { model, domains };
        state.propagate((0..factor_count).map(factor_id_from_index))?;
        state.complete_filter_components((0..variable_count).map(variable_id_from_index))?;
        Ok(state)
    }

    pub(crate) fn domain(&self, variable: VariableId) -> Option<Domain> {
        self.domains.get(variable.index()).copied()
    }

    pub(crate) fn semantic_snapshot(&self) -> ConstraintStateSnapshot {
        ConstraintStateSnapshot {
            domains: self.domains.clone(),
        }
    }

    /// Return one atomically restricted and propagated successor.
    ///
    /// Repeated restrictions for the same variable are intersected before any
    /// candidate state is created. The source state is never mutated.
    pub(crate) fn with_restrictions(
        &self,
        restrictions: impl IntoIterator<Item = (VariableId, Domain)>,
    ) -> Result<(Self, PropagationSummary), NativeSolverError> {
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
            return Ok((successor, PropagationSummary::default()));
        }

        let mut summary = successor.propagate(seed_factors)?;
        let exact_reductions =
            successor.complete_filter_components(changed_variables.iter().copied())?;
        summary.domain_reductions += changed_variables.len() + exact_reductions;
        Ok((successor, summary))
    }

    fn propagate(
        &mut self,
        seeds: impl IntoIterator<Item = FactorId>,
    ) -> Result<PropagationSummary, NativeSolverError> {
        let mut queue = VecDeque::new();
        let mut queued = BTreeSet::new();
        let mut visited = BTreeSet::new();
        let mut summary = PropagationSummary::default();

        for factor in seeds {
            enqueue_factor(factor, &mut queue, &mut queued);
        }

        while let Some(factor_id) = queue.pop_front() {
            queued.remove(&factor_id);
            summary.factor_revisions += 1;
            if visited.insert(factor_id) {
                summary.distinct_factors_visited += 1;
            }

            let factor = self
                .model
                .factor(factor_id)
                .expect("factor adjacency must reference a prepared factor");
            let reductions = match factor {
                FactorDefinition::BinaryRelation(relation) => {
                    revise_binary_relation(relation, &mut self.domains)?
                }
            };

            for variable in reductions.into_iter().flatten() {
                summary.domain_reductions += 1;
                for &neighbour in self
                    .model
                    .factors_for_variable(variable)
                    .expect("factor scope must reference a prepared variable")
                {
                    enqueue_factor(neighbour, &mut queue, &mut queued);
                }
            }
        }

        Ok(summary)
    }

    fn complete_filter_components(
        &mut self,
        seeds: impl IntoIterator<Item = VariableId>,
    ) -> Result<usize, NativeSolverError> {
        let mut covered = BTreeSet::new();
        let mut reductions = 0;

        for seed in seeds {
            if covered.contains(&seed) {
                continue;
            }
            let component = self.constraint_component(seed);
            covered.extend(component.variables.iter().copied());
            reductions += self.complete_filter_component(&component)?;
        }

        Ok(reductions)
    }

    fn complete_filter_component(
        &mut self,
        component: &ConstraintComponent,
    ) -> Result<usize, NativeSolverError> {
        if !component.requires_exact_search(&self.domains) {
            return Ok(0);
        }

        let model = Arc::clone(&self.model);
        let local = LocalBinaryComponent::new(model.as_ref(), component);
        let domains = component
            .variables
            .iter()
            .map(|variable| self.domains[variable.index()])
            .collect::<Vec<_>>();
        let supported = local
            .exact_supported_domains(domains)
            .ok_or(NativeSolverError::Contradiction)?;

        let mut reductions = 0;
        for (variable, supported_domain) in component
            .variables
            .iter()
            .copied()
            .zip(supported)
        {
            let current = self.domains[variable.index()];
            debug_assert!(supported_domain.is_subset_of(current));
            if supported_domain != current {
                self.domains[variable.index()] = supported_domain;
                reductions += 1;
            }
        }
        Ok(reductions)
    }

    fn constraint_component(&self, seed: VariableId) -> ConstraintComponent {
        let mut variables = BTreeSet::new();
        let mut factors = BTreeSet::new();
        let mut pending = VecDeque::from([seed]);

        while let Some(variable) = pending.pop_front() {
            if !variables.insert(variable) {
                continue;
            }

            for &factor_id in self
                .model
                .factors_for_variable(variable)
                .expect("known variable must have an adjacency row")
            {
                if !factors.insert(factor_id) {
                    continue;
                }
                let factor = self
                    .model
                    .factor(factor_id)
                    .expect("factor adjacency must reference a prepared factor");
                pending.extend(factor.variables());
            }
        }

        ConstraintComponent {
            variables: variables.into_iter().collect(),
            factors: factors.into_iter().collect(),
        }
    }
}

#[derive(Debug)]
struct ConstraintComponent {
    variables: Vec<VariableId>,
    factors: Vec<FactorId>,
}

impl ConstraintComponent {
    fn requires_exact_search(&self, domains: &[Domain]) -> bool {
        let unresolved = self
            .variables
            .iter()
            .any(|variable| domains[variable.index()].len() > 1);

        // Every current factor is binary. For a connected component of the
        // variable/factor incidence graph, a cycle exists exactly when
        // factor_count >= variable_count.
        unresolved && self.factors.len() >= self.variables.len()
    }
}

struct LocalBinaryFactor<'a> {
    relation: &'a BinaryRelationFactor,
    left: usize,
    right: usize,
}

struct LocalBinaryComponent<'a> {
    factors: Vec<LocalBinaryFactor<'a>>,
    factors_by_variable: Vec<Vec<usize>>,
}

impl<'a> LocalBinaryComponent<'a> {
    fn new(model: &'a ConstraintModel, component: &ConstraintComponent) -> Self {
        let mut factors = Vec::with_capacity(component.factors.len());
        let mut factors_by_variable = vec![Vec::new(); component.variables.len()];

        for factor_id in &component.factors {
            let relation = match model
                .factor(*factor_id)
                .expect("component factor must exist")
            {
                FactorDefinition::BinaryRelation(relation) => relation,
            };
            let left = component
                .variables
                .binary_search(&relation.left())
                .expect("component must contain the factor's left variable");
            let right = component
                .variables
                .binary_search(&relation.right())
                .expect("component must contain the factor's right variable");
            let local_factor = factors.len();
            factors.push(LocalBinaryFactor {
                relation,
                left,
                right,
            });
            factors_by_variable[left].push(local_factor);
            factors_by_variable[right].push(local_factor);
        }

        Self {
            factors,
            factors_by_variable,
        }
    }

    fn exact_supported_domains(&self, mut domains: Vec<Domain>) -> Option<Vec<Domain>> {
        let all_factors = (0..self.factors.len()).collect::<Vec<_>>();
        self.propagate(&mut domains, all_factors).ok()?;

        let target_domains = domains.clone();
        let mut supported = vec![Domain::empty(); domains.len()];
        let mut stack = vec![SearchNode {
            domains,
            seeds: Vec::new(),
        }];
        let mut found_solution = false;

        while let Some(node) = stack.pop() {
            let mut candidate = node.domains;
            if self.propagate(&mut candidate, node.seeds).is_err() {
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
            let old_left = domains[factor.left];
            let old_right = domains[factor.right];
            let (new_left, new_right) =
                revised_binary_domains(factor.relation, old_left, old_right)?;

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
    domains: &mut [Domain],
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
    let new_left = values_with_support(old_left, old_right, |value| {
        factor.allowed_right(value)
    });
    if new_left.is_empty() {
        return Err(NativeSolverError::Contradiction);
    }

    let new_right = values_with_support(old_right, new_left, |value| {
        factor.allowed_left(value)
    });
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
    use crate::model::ConstraintModelBuilder;

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
    fn restriction_propagates_only_through_the_affected_component() {
        let (model, variables) = equality_chain();
        let source = NativeSolverState::initial(model).unwrap();

        let (successor, summary) = source
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
        assert_eq!(summary.distinct_factors_visited(), 2);
        assert_eq!(summary.domain_reductions(), 3);
    }

    #[test]
    fn restriction_batch_updates_independent_components_once() {
        let (model, variables) = equality_chain();
        let source = NativeSolverState::initial(model).unwrap();

        let (successor, summary) = source
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
        assert_eq!(summary.distinct_factors_visited(), 3);
        assert_eq!(summary.domain_reductions(), 5);
    }

    #[test]
    fn repeated_restrictions_are_intersected_before_propagation() {
        let mut builder = ConstraintModelBuilder::new();
        let variable = builder
            .add_variable(Domain::from_indices([0, 1, 2]).unwrap())
            .unwrap();
        let source = NativeSolverState::initial(Arc::new(builder.build())).unwrap();

        let (successor, summary) = source
            .with_restrictions([
                (variable, Domain::from_indices([0, 1]).unwrap()),
                (variable, Domain::from_indices([1, 2]).unwrap()),
            ])
            .unwrap();

        assert_eq!(successor.domain(variable), Some(Domain::singleton(1).unwrap()));
        assert_eq!(summary.factor_revisions(), 0);
        assert_eq!(summary.domain_reductions(), 1);
    }

    #[test]
    fn conflicting_restrictions_leave_the_source_unchanged() {
        let mut builder = ConstraintModelBuilder::new();
        let variable = builder.add_variable(two_values()).unwrap();
        let source = NativeSolverState::initial(Arc::new(builder.build())).unwrap();
        let before = source.semantic_snapshot();

        assert!(matches!(
            source.with_restrictions([
                (variable, Domain::singleton(0).unwrap()),
                (variable, Domain::singleton(1).unwrap()),
            ]),
            Err(NativeSolverError::Contradiction)
        ));
        assert_eq!(source.semantic_snapshot(), before);
    }

    #[test]
    fn empty_and_noop_restrictions_do_not_propagate() {
        let mut builder = ConstraintModelBuilder::new();
        let variable = builder.add_variable(two_values()).unwrap();
        let source = NativeSolverState::initial(Arc::new(builder.build())).unwrap();

        let (empty_successor, empty_summary) = source
            .with_restrictions(std::iter::empty::<(VariableId, Domain)>())
            .unwrap();
        let (noop_successor, noop_summary) = source
            .with_restrictions([(variable, two_values())])
            .unwrap();

        assert_eq!(empty_successor.semantic_snapshot(), source.semantic_snapshot());
        assert_eq!(noop_successor.semantic_snapshot(), source.semantic_snapshot());
        assert_eq!(empty_summary, PropagationSummary::default());
        assert_eq!(noop_summary, PropagationSummary::default());
    }

    #[test]
    fn successor_shares_the_model_but_not_mutable_domains() {
        let (model, variables) = equality_chain();
        let source = NativeSolverState::initial(model).unwrap();
        let source_snapshot = source.semantic_snapshot();

        let (successor, _) = source
            .with_restrictions([(variables[0], Domain::singleton(0).unwrap())])
            .unwrap();

        assert!(Arc::ptr_eq(&source.model, &successor.model));
        assert_eq!(source.semantic_snapshot(), source_snapshot);
        assert_ne!(successor.semantic_snapshot(), source_snapshot);
    }

    #[test]
    fn failed_restriction_leaves_the_source_unchanged() {
        let (model, variables) = equality_chain();
        let source = NativeSolverState::initial(model).unwrap();
        let (fixed, _) = source
            .with_restrictions([(variables[0], Domain::singleton(0).unwrap())])
            .unwrap();
        let before = fixed.semantic_snapshot();

        assert!(matches!(
            fixed.with_restrictions([(variables[1], Domain::singleton(1).unwrap())]),
            Err(NativeSolverError::Contradiction)
        ));
        assert_eq!(fixed.semantic_snapshot(), before);
    }

    #[test]
    fn semantic_snapshot_is_independent_of_restriction_order() {
        let (model, variables) = equality_chain();
        let source = NativeSolverState::initial(model).unwrap();

        let (left_first, _) = source
            .with_restrictions([(variables[0], Domain::singleton(0).unwrap())])
            .unwrap();
        let (left_then_right, _) = left_first
            .with_restrictions([(variables[2], Domain::singleton(0).unwrap())])
            .unwrap();
        let (right_first, _) = source
            .with_restrictions([(variables[2], Domain::singleton(0).unwrap())])
            .unwrap();
        let (right_then_left, _) = right_first
            .with_restrictions([(variables[0], Domain::singleton(0).unwrap())])
            .unwrap();

        assert_eq!(
            left_then_right.semantic_snapshot(),
            right_then_left.semantic_snapshot()
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
        builder
            .add_binary_relation(y, z, [(0, 0), (1, 1)])
            .unwrap();
        builder
            .add_binary_relation(z, x, [(0, 0), (1, 1), (1, 2)])
            .unwrap();

        let state = NativeSolverState::initial(Arc::new(builder.build())).unwrap();

        assert_eq!(
            state.domain(x),
            Some(Domain::from_indices([0, 1]).unwrap())
        );
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
        builder
            .add_binary_relation(y, z, [(0, 0), (1, 1)])
            .unwrap();
        builder
            .add_binary_relation(z, x, [(0, 0), (1, 1), (1, 2)])
            .unwrap();
        let source = NativeSolverState::initial(Arc::new(builder.build())).unwrap();

        let (successor, _) = source
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
