//! Native finite-domain propagation for South Star 2.
//!
//! This module is crate-private while complete satisfiability search is still
//! being integrated. It maintains branch-local domains and arc consistency;
//! temporary implementation stages are not represented as capability errors.

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
        let mut state = Self { model, domains };
        state.propagate((0..factor_count).map(factor_id_from_index))?;
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
        let mut explicit_reductions = 0;
        let mut seed_factors = BTreeSet::new();

        for (variable, restricted) in domains_by_variable {
            let current = successor.domains[variable.index()];
            if restricted == current {
                continue;
            }
            successor.domains[variable.index()] = restricted;
            explicit_reductions += 1;
            seed_factors.extend(
                successor
                    .model
                    .factors_for_variable(variable)
                    .expect("known variable must have an adjacency row")
                    .iter()
                    .copied(),
            );
        }

        if explicit_reductions == 0 {
            return Ok((successor, PropagationSummary::default()));
        }

        let mut summary = successor.propagate(seed_factors)?;
        summary.domain_reductions += explicit_reductions;
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
}

fn revise_binary_relation(
    factor: &BinaryRelationFactor,
    domains: &mut [Domain],
) -> Result<[Option<VariableId>; 2], NativeSolverError> {
    let left = factor.left();
    let right = factor.right();
    let old_left = domains[left.index()];
    let old_right = domains[right.index()];

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

fn enqueue_factor(
    factor: FactorId,
    queue: &mut VecDeque<FactorId>,
    queued: &mut BTreeSet<FactorId>,
) {
    if queued.insert(factor) {
        queue.push_back(factor);
    }
}

fn factor_id_from_index(index: usize) -> FactorId {
    FactorId::new(
        u32::try_from(index).expect("constraint model validated the factor identifier capacity"),
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
