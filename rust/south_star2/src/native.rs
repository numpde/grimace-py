//! Native finite-domain solving for South Star 2.
//!
//! Factor propagation is seeded only from changed variables. Binary relation
//! components retain exact finite-domain filtering; the spanning-tree factor has
//! its own exact graphic-matroid projection and never enumerates spanning trees.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::{FactorId, VariableId};
use crate::model::{
    BinaryRelationFactor, BondRole, ConstraintModel, FactorDefinition, SpanningTreeFactor,
};

#[derive(Clone, Debug)]
pub(crate) struct NativeSolverState {
    model: Arc<ConstraintModel>,
    domains: Box<[Domain]>,
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
        let domains = model
            .initial_domains()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let factor_count = model.factor_count();
        let variable_count = model.variable_count();
        let mut state = Self { model, domains };
        state.enforce_consistency(
            (0..factor_count).map(factor_id_from_index),
            (0..variable_count).map(variable_id_from_index),
        )?;
        Ok(state)
    }

    pub(crate) fn domain(&self, variable: VariableId) -> Option<Domain> {
        self.domains.get(variable.index()).copied()
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
            let exact_reductions =
                self.complete_filter_binary_components(exact_seeds.iter().copied())?;
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

    fn complete_filter_binary_components(
        &mut self,
        seeds: impl IntoIterator<Item = VariableId>,
    ) -> Result<BTreeSet<VariableId>, NativeSolverError> {
        let mut covered = BTreeSet::new();
        let mut reductions = BTreeSet::new();

        for seed in seeds {
            if covered.contains(&seed) {
                continue;
            }
            let component = self.binary_constraint_component(seed);
            covered.extend(component.variables.iter().copied());
            reductions.extend(self.complete_filter_binary_component(&component)?);
        }

        Ok(reductions)
    }

    fn complete_filter_binary_component(
        &mut self,
        component: &BinaryConstraintComponent,
    ) -> Result<Vec<VariableId>, NativeSolverError> {
        if !component.requires_exact_search(&self.domains) {
            return Ok(Vec::new());
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

    fn binary_constraint_component(&self, seed: VariableId) -> BinaryConstraintComponent {
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
                let factor = self
                    .model
                    .factor(factor_id)
                    .expect("factor adjacency must reference a prepared factor");
                let FactorDefinition::BinaryRelation(relation) = factor else {
                    continue;
                };
                if factors.insert(factor_id) {
                    pending.extend(relation.variables().iter().copied());
                }
            }
        }

        BinaryConstraintComponent {
            variables: variables.into_iter().collect(),
            factors: factors.into_iter().collect(),
        }
    }
}

#[derive(Debug)]
struct BinaryConstraintComponent {
    variables: Vec<VariableId>,
    factors: Vec<FactorId>,
}

impl BinaryConstraintComponent {
    fn requires_exact_search(&self, domains: &[Domain]) -> bool {
        let unresolved = self
            .variables
            .iter()
            .any(|variable| domains[variable.index()].len() > 1);

        // This component contains only binary relations. For a connected
        // variable/factor incidence graph, a cycle exists exactly when the
        // number of factors is at least the number of variables.
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
    fn new(model: &'a ConstraintModel, component: &BinaryConstraintComponent) -> Self {
        let mut factors = Vec::with_capacity(component.factors.len());
        let mut factors_by_variable = vec![Vec::new(); component.variables.len()];

        for factor_id in &component.factors {
            let FactorDefinition::BinaryRelation(relation) = model
                .factor(*factor_id)
                .expect("component factor must exist")
            else {
                unreachable!("binary component must contain only binary relations");
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
    domains: &mut [Domain],
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
        assert_eq!(domains_for(&source, &variables), source_domains);
        assert_ne!(domains_for(&successor, &variables), source_domains);
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
