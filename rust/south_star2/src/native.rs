//! First native finite-domain backend for South Star 2.
//!
//! All prepared factors are active in this initial slice. Factor lifecycle is
//! added only when the walker has an event that genuinely needs it.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::{FactorId, VariableId};
use crate::model::{BinaryRelationFactor, ConstraintModel, FactorDefinition};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConstraintStateSnapshot {
    domains: Box<[Domain]>,
}

impl ConstraintStateSnapshot {
    pub fn domains(&self) -> &[Domain] {
        &self.domains
    }
}

#[derive(Clone, Debug)]
pub struct NativeSolverState {
    model: Arc<ConstraintModel>,
    domains: Box<[Domain]>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PropagationSummary {
    factor_revisions: usize,
    distinct_factors_visited: usize,
    domain_reductions: usize,
}

impl PropagationSummary {
    pub const fn factor_revisions(self) -> usize {
        self.factor_revisions
    }

    pub const fn distinct_factors_visited(self) -> usize {
        self.distinct_factors_visited
    }

    pub const fn domain_reductions(self) -> usize {
        self.domain_reductions
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeSolverError {
    UnknownVariable(VariableId),
    Contradiction,
    UnsupportedCyclicComponent {
        variable_count: usize,
        factor_count: usize,
    },
}

impl fmt::Display for NativeSolverError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownVariable(variable) => {
                write!(formatter, "unknown constraint variable {variable:?}")
            }
            Self::Contradiction => formatter.write_str("constraint state is contradictory"),
            Self::UnsupportedCyclicComponent {
                variable_count,
                factor_count,
            } => write!(
                formatter,
                "unresolved cyclic constraint component is not yet supported: \
                 variables={variable_count}, factors={factor_count}"
            ),
        }
    }
}

impl std::error::Error for NativeSolverError {}

impl NativeSolverState {
    pub fn initial(model: Arc<ConstraintModel>) -> Result<Self, NativeSolverError> {
        let domains = model.initial_domains().collect::<Vec<_>>().into_boxed_slice();
        let factor_ids = (0..model.factor_count())
            .map(factor_id_from_index)
            .collect::<Vec<_>>();
        let mut state = Self { model, domains };
        state.propagate(factor_ids)?;
        state.require_supported_components()?;
        Ok(state)
    }

    pub fn domain(&self, variable: VariableId) -> Option<Domain> {
        self.domains.get(variable.index()).copied()
    }

    pub fn semantic_snapshot(&self) -> ConstraintStateSnapshot {
        ConstraintStateSnapshot {
            domains: self.domains.clone(),
        }
    }

    /// Return a propagated successor without mutating the source state.
    pub fn restricted(
        &self,
        variable: VariableId,
        allowed: Domain,
    ) -> Result<(Self, PropagationSummary), NativeSolverError> {
        let current = self
            .domain(variable)
            .ok_or(NativeSolverError::UnknownVariable(variable))?;
        let restricted = current.intersect(allowed);
        if restricted.is_empty() {
            return Err(NativeSolverError::Contradiction);
        }

        let mut successor = self.clone();
        if restricted == current {
            return Ok((successor, PropagationSummary::default()));
        }

        successor.domains[variable.index()] = restricted;
        let seeds = successor
            .model
            .factors_for_variable(variable)
            .expect("known variable must have an adjacency row")
            .to_vec();
        let mut summary = successor.propagate(seeds)?;
        successor.require_supported_component(variable)?;
        summary.domain_reductions += 1;
        Ok((successor, summary))
    }

    fn propagate(
        &mut self,
        seeds: impl IntoIterator<Item = FactorId>,
    ) -> Result<PropagationSummary, NativeSolverError> {
        let mut queue = VecDeque::new();
        let mut queued = vec![false; self.model.factor_count()];
        let mut visited = vec![false; self.model.factor_count()];
        let mut summary = PropagationSummary::default();

        for factor in seeds {
            enqueue_factor(factor, &mut queue, &mut queued);
        }

        while let Some(factor_id) = queue.pop_front() {
            queued[factor_id.index()] = false;
            summary.factor_revisions += 1;
            if !visited[factor_id.index()] {
                visited[factor_id.index()] = true;
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

            for variable in reductions {
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

    fn require_supported_components(&self) -> Result<(), NativeSolverError> {
        let mut covered = vec![false; self.model.variable_count()];
        for index in 0..self.model.variable_count() {
            if covered[index] {
                continue;
            }
            let seed = variable_id_from_index(index);
            let (variables, factors) = self.constraint_component(seed);
            for variable in &variables {
                covered[variable.index()] = true;
            }
            self.require_supported_component_shape(&variables, &factors)?;
        }
        Ok(())
    }

    fn require_supported_component(
        &self,
        seed: VariableId,
    ) -> Result<(), NativeSolverError> {
        let (variables, factors) = self.constraint_component(seed);
        self.require_supported_component_shape(&variables, &factors)
    }

    fn require_supported_component_shape(
        &self,
        variables: &[VariableId],
        factors: &[FactorId],
    ) -> Result<(), NativeSolverError> {
        let unresolved = variables
            .iter()
            .any(|variable| self.domains[variable.index()].len() > 1);
        if unresolved && factors.len() >= variables.len() {
            return Err(NativeSolverError::UnsupportedCyclicComponent {
                variable_count: variables.len(),
                factor_count: factors.len(),
            });
        }
        Ok(())
    }

    fn constraint_component(
        &self,
        seed: VariableId,
    ) -> (Vec<VariableId>, Vec<FactorId>) {
        let mut variables = Vec::new();
        let mut factors = Vec::new();
        let mut seen_variables = vec![false; self.model.variable_count()];
        let mut seen_factors = vec![false; self.model.factor_count()];
        let mut pending = VecDeque::from([seed]);

        while let Some(variable) = pending.pop_front() {
            if seen_variables[variable.index()] {
                continue;
            }
            seen_variables[variable.index()] = true;
            variables.push(variable);

            for &factor_id in self
                .model
                .factors_for_variable(variable)
                .expect("known variable must have an adjacency row")
            {
                if seen_factors[factor_id.index()] {
                    continue;
                }
                seen_factors[factor_id.index()] = true;
                factors.push(factor_id);
                let factor = self
                    .model
                    .factor(factor_id)
                    .expect("factor adjacency must reference a prepared factor");
                pending.extend(factor.variables());
            }
        }

        (variables, factors)
    }
}

fn revise_binary_relation(
    factor: &BinaryRelationFactor,
    domains: &mut [Domain],
) -> Result<Vec<VariableId>, NativeSolverError> {
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

    let mut reductions = Vec::with_capacity(2);
    if new_left != old_left {
        domains[left.index()] = new_left;
        reductions.push(left);
    }
    if new_right != old_right {
        domains[right.index()] = new_right;
        reductions.push(right);
    }
    Ok(reductions)
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
    queued: &mut [bool],
) {
    if !queued[factor.index()] {
        queued[factor.index()] = true;
        queue.push_back(factor);
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
            .restricted(variables[0], Domain::singleton(0).unwrap())
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
    fn successor_shares_the_model_but_not_mutable_domains() {
        let (model, variables) = equality_chain();
        let source = NativeSolverState::initial(model).unwrap();
        let source_snapshot = source.semantic_snapshot();

        let (successor, _) = source
            .restricted(variables[0], Domain::singleton(0).unwrap())
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
            .restricted(variables[0], Domain::singleton(0).unwrap())
            .unwrap();
        let before = fixed.semantic_snapshot();

        assert!(matches!(
            fixed.restricted(variables[1], Domain::singleton(1).unwrap()),
            Err(NativeSolverError::Contradiction)
        ));
        assert_eq!(fixed.semantic_snapshot(), before);
    }

    #[test]
    fn semantic_snapshot_is_independent_of_restriction_order() {
        let (model, variables) = equality_chain();
        let source = NativeSolverState::initial(model).unwrap();

        let (left_first, _) = source
            .restricted(variables[0], Domain::singleton(0).unwrap())
            .unwrap();
        let (left_then_right, _) = left_first
            .restricted(variables[2], Domain::singleton(0).unwrap())
            .unwrap();
        let (right_first, _) = source
            .restricted(variables[2], Domain::singleton(0).unwrap())
            .unwrap();
        let (right_then_left, _) = right_first
            .restricted(variables[0], Domain::singleton(0).unwrap())
            .unwrap();

        assert_eq!(
            left_then_right.semantic_snapshot(),
            right_then_left.semantic_snapshot()
        );
    }

    #[test]
    fn unresolved_cyclic_component_fails_closed() {
        let mut builder = ConstraintModelBuilder::new();
        let variables: [VariableId; 3] =
            std::array::from_fn(|_| builder.add_variable(two_values()).unwrap());
        for (left, right) in [(0, 1), (1, 2), (2, 0)] {
            builder
                .add_binary_relation(
                    variables[left],
                    variables[right],
                    [(0, 0), (1, 1)],
                )
                .unwrap();
        }

        assert!(matches!(
            NativeSolverState::initial(Arc::new(builder.build())),
            Err(NativeSolverError::UnsupportedCyclicComponent {
                variable_count: 3,
                factor_count: 3,
            })
        ));
    }

    #[test]
    fn fully_resolved_cyclic_component_is_accepted() {
        let mut builder = ConstraintModelBuilder::new();
        let variables: [VariableId; 3] = std::array::from_fn(|_| {
            builder
                .add_variable(Domain::singleton(0).unwrap())
                .unwrap()
        });
        for (left, right) in [(0, 1), (1, 2), (2, 0)] {
            builder
                .add_binary_relation(variables[left], variables[right], [(0, 0)])
                .unwrap();
        }

        assert!(NativeSolverState::initial(Arc::new(builder.build())).is_ok());
    }

    #[test]
    fn unknown_variable_is_rejected() {
        let state = NativeSolverState::initial(Arc::new(ConstraintModel::empty())).unwrap();
        let variable = VariableId::new(7);

        assert!(matches!(
            state.restricted(variable, Domain::singleton(0).unwrap()),
            Err(NativeSolverError::UnknownVariable(found)) if found == variable
        ));
    }
}
