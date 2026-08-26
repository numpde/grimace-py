//! Exhaustive tiny-CSP backend and differential checks for the solver contract.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::{AtomId, FactorId, VariableId};
use crate::model::{
    BondRole, ConstraintModel, ConstraintModelBuilder, EdgeRolePartition, FactorDefinition,
    SpanningTreeEdge, SpanningTreeFactor, TetrahedralLayoutBond,
};
use crate::native::NativeSolverState;
use crate::solver::{Consistency, ConstraintSolver};

#[derive(Clone, Debug)]
pub(crate) struct ExhaustiveSolverState {
    model: Arc<ConstraintModel>,
    domains: Box<[Domain]>,
    active_factors: Box<[bool]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ExhaustiveSolverFailure {
    UnknownVariable(VariableId),
    UnknownFactor(FactorId),
}

impl fmt::Display for ExhaustiveSolverFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownVariable(variable) => {
                write!(formatter, "unknown constraint variable {variable:?}")
            }
            Self::UnknownFactor(factor) => {
                write!(formatter, "unknown constraint factor {factor:?}")
            }
        }
    }
}

impl std::error::Error for ExhaustiveSolverFailure {}

impl ConstraintSolver for ExhaustiveSolverState {
    type Failure = ExhaustiveSolverFailure;

    fn initial(model: Arc<ConstraintModel>) -> Result<Consistency<Self>, Self::Failure> {
        let initial = model.initial_domains().collect::<Vec<_>>();
        let mut active_factors = vec![false; model.factor_count()];
        for factor in model.initial_factor_ids() {
            active_factors[factor.index()] = true;
        }
        Ok(
            match exhaustive_projected_domains(model.as_ref(), &initial, &active_factors) {
                Some(domains) => Consistency::Consistent(Self {
                    model,
                    domains,
                    active_factors: active_factors.into_boxed_slice(),
                }),
                None => Consistency::Contradiction,
            },
        )
    }

    fn restricted(
        &self,
        restrictions: &[(VariableId, Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
        self.transitioned(restrictions, &[])
    }

    fn transitioned(
        &self,
        restrictions: &[(VariableId, Domain)],
        activate: &[FactorId],
    ) -> Result<Consistency<Self>, Self::Failure> {
        let mut active_factors = self.active_factors.to_vec();
        for &factor in activate {
            let active = active_factors
                .get_mut(factor.index())
                .ok_or(ExhaustiveSolverFailure::UnknownFactor(factor))?;
            *active = true;
        }

        let mut requested = BTreeMap::new();
        let mut contradictory = false;
        for &(variable, allowed) in restrictions {
            let current = self
                .domains
                .get(variable.index())
                .copied()
                .ok_or(ExhaustiveSolverFailure::UnknownVariable(variable))?;
            let restricted = requested.entry(variable).or_insert(current);
            *restricted = restricted.intersect(allowed);
            contradictory |= restricted.is_empty();
        }
        if contradictory {
            return Ok(Consistency::Contradiction);
        }
        let mut candidate = self.domains.to_vec();
        for (variable, restricted) in requested {
            candidate[variable.index()] = restricted;
        }
        Ok(
            match exhaustive_projected_domains(self.model.as_ref(), &candidate, &active_factors) {
                Some(domains) => Consistency::Consistent(Self {
                    model: Arc::clone(&self.model),
                    domains,
                    active_factors: active_factors.into_boxed_slice(),
                }),
                None => Consistency::Contradiction,
            },
        )
    }

    fn domain(&self, variable: VariableId) -> Option<Domain> {
        self.domains.get(variable.index()).copied()
    }

    fn factor_is_active(&self, factor: FactorId) -> Option<bool> {
        self.active_factors.get(factor.index()).copied()
    }
}

fn exhaustive_projected_domains(
    model: &ConstraintModel,
    domains: &[Domain],
    active_factors: &[bool],
) -> Option<Box<[Domain]>> {
    assert_eq!(domains.len(), model.variable_count());
    assert_eq!(active_factors.len(), model.factor_count());

    let mut assignment = vec![0_u8; domains.len()];
    let mut relevant = vec![false; domains.len()];
    for (index, active) in active_factors.iter().copied().enumerate() {
        if !active {
            continue;
        }
        let factor = FactorId::new(
            u32::try_from(index).expect("constraint model validated factor identifiers"),
        );
        for variable in model
            .factor(factor)
            .expect("active factor must exist")
            .variables()
        {
            relevant[variable.index()] = true;
        }
    }
    let mut supported = domains
        .iter()
        .copied()
        .zip(&relevant)
        .map(
            |(domain, relevant)| {
                if *relevant {
                    Domain::empty()
                } else {
                    domain
                }
            },
        )
        .collect::<Vec<_>>();
    let mut found = false;
    enumerate_assignments(
        model,
        domains,
        active_factors,
        &relevant,
        0,
        &mut assignment,
        &mut supported,
        &mut found,
    );
    found.then(|| supported.into_boxed_slice())
}

fn enumerate_assignments(
    model: &ConstraintModel,
    domains: &[Domain],
    active_factors: &[bool],
    relevant: &[bool],
    variable: usize,
    assignment: &mut [u8],
    supported: &mut [Domain],
    found: &mut bool,
) {
    if variable == domains.len() {
        if !assignment_satisfies(model, active_factors, assignment) {
            return;
        }
        *found = true;
        for (index, (projection, value)) in supported
            .iter_mut()
            .zip(assignment.iter().copied())
            .enumerate()
        {
            if relevant[index] {
                *projection = projection.union(Domain::from_bits(1_u64 << value));
            }
        }
        return;
    }

    if !relevant[variable] {
        assignment[variable] = domains[variable]
            .iter()
            .next()
            .expect("solver domains are nonempty");
        enumerate_assignments(
            model,
            domains,
            active_factors,
            relevant,
            variable + 1,
            assignment,
            supported,
            found,
        );
        return;
    }

    for value in domains[variable].iter() {
        assignment[variable] = value;
        enumerate_assignments(
            model,
            domains,
            active_factors,
            relevant,
            variable + 1,
            assignment,
            supported,
            found,
        );
    }
}

fn assignment_satisfies(
    model: &ConstraintModel,
    active_factors: &[bool],
    assignment: &[u8],
) -> bool {
    (0..model.factor_count()).all(|index| {
        if !active_factors[index] {
            return true;
        }
        let factor_id = FactorId::new(
            u32::try_from(index).expect("constraint model validated factor identifiers"),
        );
        match model.factor(factor_id).expect("prepared factor must exist") {
            FactorDefinition::BinaryRelation(relation) => relation
                .allowed_right(assignment[relation.left().index()])
                .contains(assignment[relation.right().index()]),
            FactorDefinition::SpanningTree(spanning_tree) => {
                assignment_satisfies_spanning_tree(spanning_tree, assignment)
            }
            FactorDefinition::TetrahedralLayout(layout) => {
                let pattern = assignment[layout.role_pattern_variable().index()];
                layout
                    .allowed_orders(pattern)
                    .contains(assignment[layout.order_variable().index()])
                    && layout.bonds().iter().all(|bond| {
                        let value = assignment[bond.decision_variable().index()];
                        let role_values = if pattern & (1_u8 << bond.pattern_bit()) == 0 {
                            bond.role_partition().traversal_values()
                        } else {
                            bond.role_partition().ring_values()
                        };
                        role_values.contains(value)
                    })
            }
        }
    })
}

fn assignment_satisfies_spanning_tree(factor: &SpanningTreeFactor, assignment: &[u8]) -> bool {
    let mut traversal_edge_count = 0;

    for edge in factor.edges() {
        let value = assignment[edge.decision_variable().index()];
        let partition = edge.role_partition();
        if partition.traversal_values().contains(value) {
            traversal_edge_count += 1;
        } else if !partition.ring_values().contains(value) {
            return false;
        }
    }

    if traversal_edge_count != factor.atoms().len().saturating_sub(1) {
        return false;
    }

    let Some(&root) = factor.atoms().first() else {
        return false;
    };
    let mut visited = BTreeSet::from([root]);
    let mut pending = VecDeque::from([root]);

    while let Some(atom) = pending.pop_front() {
        for edge in factor.edges() {
            let value = assignment[edge.decision_variable().index()];
            if !edge.role_partition().traversal_values().contains(value) {
                continue;
            }
            let other = if edge.a() == atom {
                Some(edge.b())
            } else if edge.b() == atom {
                Some(edge.a())
            } else {
                None
            };
            if let Some(other) = other {
                if visited.insert(other) {
                    pending.push_back(other);
                }
            }
        }
    }

    visited.len() == factor.atoms().len()
}

fn relation_rows(mask: u16) -> Vec<(u8, u8)> {
    let mut rows = Vec::new();
    for left in 0_u8..2 {
        for right in 0_u8..2 {
            let bit = usize::from(left) * 2 + usize::from(right);
            if mask & (1_u16 << bit) != 0 {
                rows.push((left, right));
            }
        }
    }
    rows
}

fn relation_accepts(mask: u16, left: u8, right: u8) -> bool {
    let bit = usize::from(left) * 2 + usize::from(right);
    mask & (1_u16 << bit) != 0
}

fn latent_layout_fixture() -> (
    Arc<ConstraintModel>,
    VariableId,
    VariableId,
    [VariableId; 3],
    FactorId,
) {
    let mut builder = ConstraintModelBuilder::new();
    let order = builder
        .add_variable(Domain::from_indices([0, 1]).unwrap())
        .unwrap();
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
    let factor =
        builder
            .add_latent_tetrahedral_layout(
                order,
                pattern,
                bonds.iter().copied().enumerate().map(|(bit, variable)| {
                    TetrahedralLayoutBond::new(variable, partition, bit as u8)
                }),
                [
                    Domain::singleton(0).unwrap(),
                    Domain::singleton(1).unwrap(),
                    Domain::empty(),
                    Domain::empty(),
                    Domain::empty(),
                    Domain::empty(),
                    Domain::empty(),
                    Domain::empty(),
                ],
            )
            .unwrap();
    (Arc::new(builder.build()), order, pattern, bonds, factor)
}

fn projected_domains<S: ConstraintSolver>(state: &S, variables: &[VariableId]) -> Vec<Domain> {
    variables
        .iter()
        .map(|variable| {
            state
                .domain(*variable)
                .expect("qualified variable must exist")
        })
        .collect()
}

fn activation_order_projections<S: ConstraintSolver>(
    model: Arc<ConstraintModel>,
    variables: &[VariableId],
    factor: FactorId,
    restrictions: &[(VariableId, Domain)],
) -> Vec<Vec<Domain>> {
    let initial = S::initial(model)
        .unwrap_or_else(|failure| panic!("solver initialization failed: {failure}"))
        .unwrap_consistent();
    let initial_projection = projected_domains(&initial, variables);
    assert_eq!(initial.factor_is_active(factor), Some(false));

    let restricted = initial
        .restricted(restrictions)
        .unwrap_or_else(|failure| panic!("solver restriction failed: {failure}"))
        .unwrap_consistent();
    assert_eq!(restricted.factor_is_active(factor), Some(false));
    let restricted_projection = projected_domains(&restricted, variables);

    let restricted_then_activated = restricted
        .transitioned(&[], &[factor])
        .unwrap_or_else(|failure| panic!("solver activation failed: {failure}"))
        .unwrap_consistent();
    let activated = initial
        .transitioned(&[], &[factor])
        .unwrap_or_else(|failure| panic!("solver activation failed: {failure}"))
        .unwrap_consistent();
    let activated_then_restricted = activated
        .restricted(restrictions)
        .unwrap_or_else(|failure| panic!("solver restriction failed: {failure}"))
        .unwrap_consistent();
    let atomic = initial
        .transitioned(restrictions, &[factor, factor])
        .unwrap_or_else(|failure| panic!("atomic solver transition failed: {failure}"))
        .unwrap_consistent();
    let reactivated = atomic
        .transitioned(restrictions, &[factor, factor])
        .unwrap_or_else(|failure| panic!("repeated solver activation failed: {failure}"))
        .unwrap_consistent();

    assert_eq!(initial.factor_is_active(factor), Some(false));
    assert_eq!(projected_domains(&initial, variables), initial_projection);
    assert_eq!(
        projected_domains(&restricted, variables),
        restricted_projection
    );
    for successor in [
        &restricted_then_activated,
        &activated_then_restricted,
        &atomic,
        &reactivated,
    ] {
        assert_eq!(successor.factor_is_active(factor), Some(true));
    }

    [
        restricted_then_activated,
        activated_then_restricted,
        atomic,
        reactivated,
    ]
    .iter()
    .map(|state| projected_domains(state, variables))
    .collect()
}

#[test]
fn latent_layout_activation_order_and_repetition_preserve_exact_restrictions() {
    let (model, order, pattern, bonds, factor) = latent_layout_fixture();
    let variables = std::iter::once(order)
        .chain(std::iter::once(pattern))
        .chain(bonds)
        .collect::<Vec<_>>();
    let restrictions = [
        (order, Domain::singleton(1).unwrap()),
        (bonds[0], Domain::singleton(2).unwrap()),
    ];

    let native = activation_order_projections::<NativeSolverState>(
        Arc::clone(&model),
        &variables,
        factor,
        &restrictions,
    );
    let exhaustive = activation_order_projections::<ExhaustiveSolverState>(
        model,
        &variables,
        factor,
        &restrictions,
    );

    assert!(native.windows(2).all(|pair| pair[0] == pair[1]));
    assert_eq!(native, exhaustive);
}

fn contradictory_activation_results<S: ConstraintSolver>(
    model: Arc<ConstraintModel>,
    variables: &[VariableId],
    factor: FactorId,
    restrictions: &[(VariableId, Domain)],
) -> [bool; 2] {
    let initial = S::initial(model)
        .unwrap_or_else(|failure| panic!("solver initialization failed: {failure}"))
        .unwrap_consistent();
    let restricted = initial
        .restricted(restrictions)
        .unwrap_or_else(|failure| panic!("solver restriction failed: {failure}"))
        .unwrap_consistent();
    let restricted_projection = projected_domains(&restricted, variables);

    let after_restriction = matches!(
        restricted
            .transitioned(&[], &[factor])
            .unwrap_or_else(|failure| panic!("solver activation failed: {failure}")),
        Consistency::Contradiction
    );
    let atomic = matches!(
        initial
            .transitioned(restrictions, &[factor])
            .unwrap_or_else(|failure| panic!("atomic solver transition failed: {failure}")),
        Consistency::Contradiction
    );

    assert_eq!(restricted.factor_is_active(factor), Some(false));
    assert_eq!(
        projected_domains(&restricted, variables),
        restricted_projection
    );
    [after_restriction, atomic]
}

#[test]
fn contradictory_latent_layout_activation_matches_the_exhaustive_backend() {
    let (model, order, pattern, bonds, factor) = latent_layout_fixture();
    let variables = std::iter::once(order)
        .chain(std::iter::once(pattern))
        .chain(bonds)
        .collect::<Vec<_>>();
    let restrictions = [
        (order, Domain::singleton(0).unwrap()),
        (bonds[0], Domain::singleton(1).unwrap()),
    ];

    let native = contradictory_activation_results::<NativeSolverState>(
        Arc::clone(&model),
        &variables,
        factor,
        &restrictions,
    );
    let exhaustive = contradictory_activation_results::<ExhaustiveSolverState>(
        model,
        &variables,
        factor,
        &restrictions,
    );

    assert_eq!(native, [true, true]);
    assert_eq!(native, exhaustive);
}

#[test]
fn native_and_exhaustive_backends_share_atomic_activation_semantics() {
    let (model, order, pattern, bonds, factor) = latent_layout_fixture();
    let native = <NativeSolverState as ConstraintSolver>::initial(Arc::clone(&model))
        .unwrap()
        .unwrap_consistent();
    let exhaustive = ExhaustiveSolverState::initial(model)
        .unwrap()
        .unwrap_consistent();

    assert_eq!(native.factor_is_active(factor), Some(false));
    assert_eq!(exhaustive.factor_is_active(factor), Some(false));
    for variable in std::iter::once(order)
        .chain(std::iter::once(pattern))
        .chain(bonds)
    {
        assert_eq!(native.domain(variable), exhaustive.domain(variable));
    }

    for restriction in [None, Some((order, Domain::singleton(0).unwrap()))] {
        let restrictions = restriction.into_iter().collect::<Vec<_>>();
        let native_successor = native
            .transitioned(&restrictions, &[factor])
            .unwrap()
            .unwrap_consistent();
        let exhaustive_successor = exhaustive
            .transitioned(&restrictions, &[factor])
            .unwrap()
            .unwrap_consistent();
        assert_eq!(native_successor.factor_is_active(factor), Some(true));
        assert_eq!(exhaustive_successor.factor_is_active(factor), Some(true));
        for variable in std::iter::once(order)
            .chain(std::iter::once(pattern))
            .chain(bonds)
        {
            assert_eq!(
                native_successor.domain(variable),
                exhaustive_successor.domain(variable)
            );
        }
    }
}

fn mixed_layout_triangle_fixture() -> (
    Arc<ConstraintModel>,
    VariableId,
    VariableId,
    [VariableId; 3],
    FactorId,
) {
    let mut builder = ConstraintModelBuilder::new();
    let order = builder
        .add_variable(Domain::from_indices([0, 1]).unwrap())
        .unwrap();
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
                Domain::singleton(0).unwrap(),
                Domain::singleton(1).unwrap(),
                Domain::empty(),
                Domain::empty(),
                Domain::empty(),
                Domain::empty(),
                Domain::empty(),
                Domain::singleton(0).unwrap(),
            ],
        )
        .unwrap();
    let atoms = [AtomId::new(0), AtomId::new(1), AtomId::new(2)];
    builder
        .add_spanning_tree(
            atoms,
            [
                SpanningTreeEdge::new(bonds[0], atoms[0], atoms[1]),
                SpanningTreeEdge::new(bonds[1], atoms[1], atoms[2]),
                SpanningTreeEdge::new(bonds[2], atoms[2], atoms[0]),
            ],
        )
        .unwrap();
    (Arc::new(builder.build()), order, pattern, bonds, factor)
}

#[test]
fn activated_layout_searches_jointly_through_the_spanning_projector() {
    let (model, order, pattern, bonds, factor) = mixed_layout_triangle_fixture();
    let native = <NativeSolverState as ConstraintSolver>::initial(Arc::clone(&model))
        .unwrap()
        .unwrap_consistent();
    let exhaustive = ExhaustiveSolverState::initial(model)
        .unwrap()
        .unwrap_consistent();

    let native = native
        .transitioned(&[], &[factor])
        .unwrap()
        .unwrap_consistent();
    let exhaustive = exhaustive
        .transitioned(&[], &[factor])
        .unwrap()
        .unwrap_consistent();

    assert_eq!(native.domain(order), Some(Domain::singleton(1).unwrap()));
    assert_eq!(native.domain(pattern), Some(Domain::singleton(1).unwrap()));
    assert_eq!(
        native.domain(bonds[0]),
        Some(BondRole::Ring.singleton_domain())
    );
    for bond in &bonds[1..] {
        assert_eq!(
            native.domain(*bond),
            Some(BondRole::Traversal.singleton_domain())
        );
    }
    for variable in std::iter::once(order)
        .chain(std::iter::once(pattern))
        .chain(bonds)
    {
        assert_eq!(native.domain(variable), exhaustive.domain(variable));
    }
}

fn layout_rows(entries: &[(usize, Domain)]) -> [Domain; 8] {
    let mut rows = [Domain::empty(); 8];
    for &(pattern, orders) in entries {
        rows[pattern] = orders;
    }
    rows
}

struct LayoutPairFixture {
    model: Arc<ConstraintModel>,
    orders: [VariableId; 2],
    patterns: [VariableId; 2],
    bonds: Vec<VariableId>,
    factors: [FactorId; 2],
}

impl LayoutPairFixture {
    fn variables(&self) -> Vec<VariableId> {
        self.orders
            .into_iter()
            .chain(self.patterns)
            .chain(self.bonds.iter().copied())
            .collect()
    }
}

fn shared_layout_triangle_fixture() -> LayoutPairFixture {
    let mut builder = ConstraintModelBuilder::new();
    let order_domain = Domain::from_indices([0, 1]).unwrap();
    let orders = std::array::from_fn(|_| builder.add_variable(order_domain).unwrap());
    let pattern_domain = Domain::from_indices(0_u8..8).unwrap();
    let patterns = std::array::from_fn(|_| builder.add_variable(pattern_domain).unwrap());
    let bonds = (0..3)
        .map(|_| builder.add_variable(BondRole::role_domain()).unwrap())
        .collect::<Vec<_>>();
    let both_orders = Domain::from_indices([0, 1]).unwrap();

    let first = builder
        .add_latent_tetrahedral_layout(
            orders[0],
            patterns[0],
            bonds.iter().copied().enumerate().map(|(bit, variable)| {
                TetrahedralLayoutBond::new(variable, EdgeRolePartition::bond_role(), bit as u8)
            }),
            layout_rows(&[
                (1, Domain::singleton(0).unwrap()),
                (2, Domain::singleton(1).unwrap()),
                (4, both_orders),
            ]),
        )
        .unwrap();
    let second = builder
        .add_latent_tetrahedral_layout(
            orders[1],
            patterns[1],
            [2_u8, 0, 1]
                .into_iter()
                .zip(bonds.iter().copied())
                .map(|(bit, variable)| {
                    TetrahedralLayoutBond::new(variable, EdgeRolePartition::bond_role(), bit)
                }),
            layout_rows(&[
                (1, both_orders),
                (2, Domain::singleton(0).unwrap()),
                (4, Domain::singleton(1).unwrap()),
            ]),
        )
        .unwrap();

    let atoms = [AtomId::new(0), AtomId::new(1), AtomId::new(2)];
    builder
        .add_spanning_tree(
            atoms,
            [
                SpanningTreeEdge::new(bonds[0], atoms[0], atoms[1]),
                SpanningTreeEdge::new(bonds[1], atoms[1], atoms[2]),
                SpanningTreeEdge::new(bonds[2], atoms[2], atoms[0]),
            ],
        )
        .unwrap();

    LayoutPairFixture {
        model: Arc::new(builder.build()),
        orders,
        patterns,
        bonds,
        factors: [first, second],
    }
}

fn activated_domains<S: ConstraintSolver>(
    model: Arc<ConstraintModel>,
    variables: &[VariableId],
    restrictions: &[(VariableId, Domain)],
    factors: &[FactorId],
) -> Option<Vec<Domain>> {
    let initial = S::initial(model)
        .unwrap_or_else(|failure| panic!("solver initialization failed: {failure}"))
        .unwrap_consistent();
    let Consistency::Consistent(successor) = initial
        .transitioned(restrictions, factors)
        .unwrap_or_else(|failure| panic!("solver transition failed: {failure}"))
    else {
        return None;
    };
    for &factor in factors {
        assert_eq!(successor.factor_is_active(factor), Some(true));
    }
    Some(projected_domains(&successor, variables))
}

#[test]
fn two_active_layouts_share_one_spanning_component_under_partial_ambiguity() {
    let fixture = shared_layout_triangle_fixture();
    let variables = fixture.variables();
    let both_orders = Domain::from_indices([0, 1]).unwrap();
    let cases = [
        vec![],
        vec![(fixture.orders[0], Domain::singleton(0).unwrap())],
        vec![(fixture.orders[1], Domain::singleton(0).unwrap())],
        vec![
            (fixture.orders[0], Domain::singleton(0).unwrap()),
            (fixture.orders[1], Domain::singleton(0).unwrap()),
        ],
        vec![
            (fixture.orders[0], Domain::singleton(1).unwrap()),
            (fixture.orders[1], Domain::singleton(1).unwrap()),
        ],
        vec![
            (fixture.patterns[0], Domain::from_indices([1, 4]).unwrap()),
            (fixture.orders[1], both_orders),
        ],
        vec![(fixture.bonds[0], BondRole::Traversal.singleton_domain())],
        vec![(fixture.bonds[1], BondRole::Ring.singleton_domain())],
        vec![
            (fixture.orders[0], Domain::singleton(0).unwrap()),
            (fixture.bonds[1], BondRole::Traversal.singleton_domain()),
        ],
    ];

    for restrictions in cases {
        let native = activated_domains::<NativeSolverState>(
            Arc::clone(&fixture.model),
            &variables,
            &restrictions,
            &fixture.factors,
        );
        let exhaustive = activated_domains::<ExhaustiveSolverState>(
            Arc::clone(&fixture.model),
            &variables,
            &restrictions,
            &fixture.factors,
        );
        assert_eq!(native, exhaustive, "restrictions {restrictions:?}");
    }

    let projection = activated_domains::<NativeSolverState>(
        Arc::clone(&fixture.model),
        &variables,
        &[],
        &fixture.factors,
    )
    .expect("unrestricted pair must remain consistent");
    assert!(projection.iter().all(|domain| domain.len() > 1));
}

fn disconnected_layout_pair_fixture() -> LayoutPairFixture {
    let mut builder = ConstraintModelBuilder::new();
    let order_domain = Domain::from_indices([0, 1]).unwrap();
    let orders = std::array::from_fn(|_| builder.add_variable(order_domain).unwrap());
    let pattern_domain = Domain::from_indices(0_u8..8).unwrap();
    let patterns = std::array::from_fn(|_| builder.add_variable(pattern_domain).unwrap());
    let bonds = (0..6)
        .map(|_| builder.add_variable(BondRole::role_domain()).unwrap())
        .collect::<Vec<_>>();
    let rows = layout_rows(&[
        (1, Domain::singleton(0).unwrap()),
        (2, Domain::singleton(1).unwrap()),
        (4, Domain::from_indices([0, 1]).unwrap()),
    ]);
    let mut factors = Vec::new();

    for component in 0..2 {
        let component_bonds = &bonds[(component * 3)..(component * 3 + 3)];
        factors.push(
            builder
                .add_latent_tetrahedral_layout(
                    orders[component],
                    patterns[component],
                    component_bonds
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
                    rows,
                )
                .unwrap(),
        );

        let atom_offset = u32::try_from(component * 3).unwrap();
        let atoms = [
            AtomId::new(atom_offset),
            AtomId::new(atom_offset + 1),
            AtomId::new(atom_offset + 2),
        ];
        builder
            .add_spanning_tree(
                atoms,
                [
                    SpanningTreeEdge::new(component_bonds[0], atoms[0], atoms[1]),
                    SpanningTreeEdge::new(component_bonds[1], atoms[1], atoms[2]),
                    SpanningTreeEdge::new(component_bonds[2], atoms[2], atoms[0]),
                ],
            )
            .unwrap();
    }

    LayoutPairFixture {
        model: Arc::new(builder.build()),
        orders,
        patterns,
        bonds,
        factors: factors.try_into().unwrap(),
    }
}

fn disconnected_isolation_projections<S: ConstraintSolver>(
    fixture: &LayoutPairFixture,
) -> Vec<Vec<Domain>> {
    let variables = fixture.variables();
    let second_variables = std::iter::once(fixture.orders[1])
        .chain(std::iter::once(fixture.patterns[1]))
        .chain(fixture.bonds[3..].iter().copied())
        .collect::<Vec<_>>();
    let initial = S::initial(Arc::clone(&fixture.model))
        .unwrap_or_else(|failure| panic!("solver initialization failed: {failure}"))
        .unwrap_consistent();
    let initial_second = projected_domains(&initial, &second_variables);
    let first_active = initial
        .transitioned(&[], &[fixture.factors[0]])
        .unwrap_or_else(|failure| panic!("first activation failed: {failure}"))
        .unwrap_consistent();
    assert_eq!(
        first_active.factor_is_active(fixture.factors[0]),
        Some(true)
    );
    assert_eq!(
        first_active.factor_is_active(fixture.factors[1]),
        Some(false)
    );
    assert_eq!(
        projected_domains(&first_active, &second_variables),
        initial_second
    );

    let sequential = first_active
        .transitioned(&[], &[fixture.factors[1]])
        .unwrap_or_else(|failure| panic!("second activation failed: {failure}"))
        .unwrap_consistent();
    let atomic = initial
        .transitioned(&[], &fixture.factors)
        .unwrap_or_else(|failure| panic!("atomic activation failed: {failure}"))
        .unwrap_consistent();
    let atomic_projection = projected_domains(&atomic, &variables);
    assert_eq!(
        projected_domains(&sequential, &variables),
        atomic_projection
    );

    let second_before = projected_domains(&atomic, &second_variables);
    let restricted = atomic
        .restricted(&[(fixture.orders[0], Domain::singleton(0).unwrap())])
        .unwrap_or_else(|failure| panic!("isolated restriction failed: {failure}"))
        .unwrap_consistent();
    let second_after = projected_domains(&restricted, &second_variables);
    assert_eq!(second_after, second_before);

    vec![
        atomic_projection,
        projected_domains(&restricted, &variables),
        second_before,
        second_after,
    ]
}

#[test]
fn disconnected_active_layouts_preserve_factor_isolation() {
    let fixture = disconnected_layout_pair_fixture();

    let native = disconnected_isolation_projections::<NativeSolverState>(&fixture);
    let exhaustive = disconnected_isolation_projections::<ExhaustiveSolverState>(&fixture);

    assert_eq!(native, exhaustive);
}

fn exhaustive_triangle_supports(masks: [u16; 3], domains: [Domain; 3]) -> Option<[Domain; 3]> {
    let mut supported = [Domain::empty(); 3];
    let mut found = false;

    for x in domains[0].iter() {
        for y in domains[1].iter() {
            for z in domains[2].iter() {
                if !relation_accepts(masks[0], x, y)
                    || !relation_accepts(masks[1], y, z)
                    || !relation_accepts(masks[2], z, x)
                {
                    continue;
                }
                found = true;
                supported[0] = supported[0].union(Domain::from_bits(1_u64 << x));
                supported[1] = supported[1].union(Domain::from_bits(1_u64 << y));
                supported[2] = supported[2].union(Domain::from_bits(1_u64 << z));
            }
        }
    }

    found.then_some(supported)
}

fn triangle_model(masks: [u16; 3]) -> (Arc<ConstraintModel>, [VariableId; 3]) {
    let binary = Domain::from_indices([0, 1]).unwrap();
    let mut builder = ConstraintModelBuilder::new();
    let variables: [VariableId; 3] = std::array::from_fn(|_| builder.add_variable(binary).unwrap());
    builder
        .add_binary_relation(variables[0], variables[1], relation_rows(masks[0]))
        .unwrap();
    builder
        .add_binary_relation(variables[1], variables[2], relation_rows(masks[1]))
        .unwrap();
    builder
        .add_binary_relation(variables[2], variables[0], relation_rows(masks[2]))
        .unwrap();
    (Arc::new(builder.build()), variables)
}

fn solve_triangle<S: ConstraintSolver>(
    model: Arc<ConstraintModel>,
    variables: [VariableId; 3],
    restrictions: &[(VariableId, Domain)],
) -> Option<[Domain; 3]> {
    let Consistency::Consistent(state) = S::initial(model)
        .unwrap_or_else(|failure| panic!("solver initialization failed: {failure}"))
    else {
        return None;
    };
    let Consistency::Consistent(state) = state
        .restricted(restrictions)
        .unwrap_or_else(|failure| panic!("solver restriction failed: {failure}"))
    else {
        return None;
    };
    Some(std::array::from_fn(|index| {
        state
            .domain(variables[index])
            .expect("triangle variable must exist")
    }))
}

#[derive(Copy, Clone)]
struct SpanningFixture<'a> {
    atoms: &'a [u32],
    edges: &'a [(u32, u32)],
}

fn spanning_model(components: &[SpanningFixture<'_>]) -> (Arc<ConstraintModel>, Vec<VariableId>) {
    let mut builder = ConstraintModelBuilder::new();
    let mut variables = Vec::new();

    for component in components {
        let atoms = component
            .atoms
            .iter()
            .copied()
            .map(AtomId::new)
            .collect::<Vec<_>>();
        let mut edges = Vec::with_capacity(component.edges.len());
        for &(a, b) in component.edges {
            let variable = builder.add_variable(BondRole::role_domain()).unwrap();
            variables.push(variable);
            edges.push(SpanningTreeEdge::new(
                variable,
                AtomId::new(a),
                AtomId::new(b),
            ));
        }
        builder.add_spanning_tree(atoms, edges).unwrap();
    }

    (Arc::new(builder.build()), variables)
}

fn mixed_triangle_model(masks: [u16; 2]) -> (Arc<ConstraintModel>, Vec<VariableId>) {
    let mut builder = ConstraintModelBuilder::new();
    let variables = (0..3)
        .map(|_| builder.add_variable(BondRole::role_domain()).unwrap())
        .collect::<Vec<_>>();
    builder
        .add_binary_relation(variables[0], variables[1], relation_rows(masks[0]))
        .unwrap();
    builder
        .add_binary_relation(variables[1], variables[2], relation_rows(masks[1]))
        .unwrap();
    let atoms = [AtomId::new(0), AtomId::new(1), AtomId::new(2)];
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
    (Arc::new(builder.build()), variables)
}

fn split_semantic_core_model() -> (Arc<ConstraintModel>, Vec<VariableId>) {
    let mut builder = ConstraintModelBuilder::new();
    let variables = (0..6)
        .map(|_| builder.add_variable(BondRole::role_domain()).unwrap())
        .collect::<Vec<_>>();
    let equality = [(0, 0), (1, 1)];
    builder
        .add_binary_relation(variables[0], variables[1], equality)
        .unwrap();
    builder
        .add_binary_relation(variables[2], variables[3], equality)
        .unwrap();
    let atoms = [
        AtomId::new(0),
        AtomId::new(1),
        AtomId::new(2),
        AtomId::new(3),
    ];
    let endpoints = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    builder
        .add_spanning_tree(
            atoms,
            endpoints
                .into_iter()
                .enumerate()
                .map(|(index, (a, b))| SpanningTreeEdge::new(variables[index], atoms[a], atoms[b])),
        )
        .unwrap();
    (Arc::new(builder.build()), variables)
}

fn coupled_triangle_projectors_model() -> (Arc<ConstraintModel>, Vec<VariableId>) {
    let mut builder = ConstraintModelBuilder::new();
    let variables = (0..6)
        .map(|_| builder.add_variable(BondRole::role_domain()).unwrap())
        .collect::<Vec<_>>();
    builder
        .add_binary_relation(variables[0], variables[3], [(0, 0), (1, 1)])
        .unwrap();
    builder
        .add_binary_relation(variables[1], variables[4], [(0, 1), (1, 0)])
        .unwrap();

    for offset in [0, 3] {
        let atoms = [
            AtomId::new(u32::try_from(offset).unwrap()),
            AtomId::new(u32::try_from(offset + 1).unwrap()),
            AtomId::new(u32::try_from(offset + 2).unwrap()),
        ];
        builder
            .add_spanning_tree(
                atoms,
                [
                    SpanningTreeEdge::new(variables[offset], atoms[0], atoms[1]),
                    SpanningTreeEdge::new(variables[offset + 1], atoms[1], atoms[2]),
                    SpanningTreeEdge::new(variables[offset + 2], atoms[2], atoms[0]),
                ],
            )
            .unwrap();
    }

    (Arc::new(builder.build()), variables)
}

fn solve_role_domains<S: ConstraintSolver>(
    model: Arc<ConstraintModel>,
    variables: &[VariableId],
    restrictions: &[(VariableId, Domain)],
) -> Option<Vec<Domain>> {
    let Consistency::Consistent(state) = S::initial(model)
        .unwrap_or_else(|failure| panic!("solver initialization failed: {failure}"))
    else {
        return None;
    };
    let Consistency::Consistent(state) = state
        .restricted(restrictions)
        .unwrap_or_else(|failure| panic!("solver restriction failed: {failure}"))
    else {
        return None;
    };
    Some(
        variables
            .iter()
            .map(|variable| {
                state
                    .domain(*variable)
                    .expect("spanning-tree variable must exist")
            })
            .collect(),
    )
}

fn partial_role_restrictions(variables: &[VariableId]) -> Vec<Vec<(VariableId, Domain)>> {
    let case_count = 3_usize.pow(u32::try_from(variables.len()).unwrap());
    let mut cases = Vec::with_capacity(case_count);

    for mut code in 0..case_count {
        let mut restrictions = Vec::new();
        for &variable in variables {
            match code % 3 {
                0 => {}
                1 => restrictions.push((variable, BondRole::Traversal.singleton_domain())),
                2 => restrictions.push((variable, BondRole::Ring.singleton_domain())),
                _ => unreachable!(),
            }
            code /= 3;
        }
        cases.push(restrictions);
    }

    cases
}

fn multivalue_spanning_model(
    atom_count: usize,
    endpoints: &[(usize, usize)],
) -> (Arc<ConstraintModel>, Vec<VariableId>) {
    let mut builder = ConstraintModelBuilder::new();
    let plan_domain = Domain::from_bits(0b1_1111);
    let partition = EdgeRolePartition::new(Domain::from_bits(0b1), Domain::from_bits(0b1_1110));
    let variables = endpoints
        .iter()
        .map(|_| builder.add_variable(plan_domain).unwrap())
        .collect::<Vec<_>>();
    let atoms = (0..atom_count)
        .map(|index| AtomId::new(u32::try_from(index).unwrap()))
        .collect::<Vec<_>>();
    builder
        .add_spanning_tree(
            atoms.iter().copied(),
            endpoints
                .iter()
                .copied()
                .enumerate()
                .map(|(index, (a, b))| {
                    SpanningTreeEdge::with_role_partition(
                        variables[index],
                        atoms[a],
                        atoms[b],
                        partition,
                    )
                }),
        )
        .unwrap();
    (Arc::new(builder.build()), variables)
}

fn multivalue_spanning_restrictions(variables: &[VariableId]) -> Vec<Vec<(VariableId, Domain)>> {
    let mut cases = vec![Vec::new()];
    for &variable in variables {
        for bits in 1_u64..0b10_0000 {
            cases.push(vec![(variable, Domain::from_bits(bits))]);
        }
    }
    for left in 0..variables.len() {
        for right in (left + 1)..variables.len() {
            for left_value in 0_u8..5 {
                for right_value in 0_u8..5 {
                    cases.push(vec![
                        (variables[left], Domain::from_bits(1_u64 << left_value)),
                        (variables[right], Domain::from_bits(1_u64 << right_value)),
                    ]);
                }
            }
        }
    }
    cases
}

fn restricted_domains<S: ConstraintSolver>(
    state: &S,
    variables: &[VariableId],
    restrictions: &[(VariableId, Domain)],
) -> Option<Vec<Domain>> {
    let Consistency::Consistent(restricted) = state
        .restricted(restrictions)
        .unwrap_or_else(|failure| panic!("solver restriction failed: {failure}"))
    else {
        return None;
    };
    Some(
        variables
            .iter()
            .map(|variable| {
                restricted
                    .domain(*variable)
                    .expect("spanning-tree variable must exist")
            })
            .collect(),
    )
}

fn assert_spanning_fixture(name: &str, components: &[SpanningFixture<'_>]) {
    let (model, variables) = spanning_model(components);

    for restrictions in partial_role_restrictions(&variables) {
        let expected = solve_role_domains::<ExhaustiveSolverState>(
            Arc::clone(&model),
            &variables,
            &restrictions,
        );
        let actual =
            solve_role_domains::<NativeSolverState>(Arc::clone(&model), &variables, &restrictions);
        assert_eq!(
            actual, expected,
            "{name} with role restrictions {restrictions:?}"
        );
    }
}

#[test]
fn every_boolean_triangle_matches_independent_assignment_support() {
    let binary = Domain::from_indices([0, 1]).unwrap();

    for xy in 0_u16..16 {
        for yz in 0_u16..16 {
            for zx in 0_u16..16 {
                let masks = [xy, yz, zx];
                let expected = exhaustive_triangle_supports(masks, [binary; 3]);
                let (model, variables) = triangle_model(masks);

                assert_eq!(
                    solve_triangle::<NativeSolverState>(Arc::clone(&model), variables, &[]),
                    expected,
                    "native triangle relation masks {masks:?}"
                );
                assert_eq!(
                    solve_triangle::<ExhaustiveSolverState>(model, variables, &[]),
                    expected,
                    "exhaustive triangle relation masks {masks:?}"
                );
            }
        }
    }
}

#[test]
fn native_and_exhaustive_backends_share_restriction_semantics() {
    let masks = [0b1001, 0b1001, 0b1001];
    let (model, variables) = triangle_model(masks);
    let binary = Domain::from_indices([0, 1]).unwrap();
    let batches = [
        vec![],
        vec![(variables[0], Domain::singleton(0).unwrap())],
        vec![
            (variables[1], binary),
            (variables[2], Domain::singleton(1).unwrap()),
        ],
        vec![
            (variables[0], binary),
            (variables[0], Domain::singleton(1).unwrap()),
        ],
        vec![
            (variables[0], Domain::singleton(0).unwrap()),
            (variables[0], Domain::singleton(1).unwrap()),
        ],
    ];

    for restrictions in batches {
        let mut domains = [binary; 3];
        for &(variable, allowed) in &restrictions {
            let index = variables
                .iter()
                .position(|candidate| *candidate == variable)
                .expect("restriction must reference a triangle variable");
            domains[index] = domains[index].intersect(allowed);
        }
        let expected = if domains.iter().any(|domain| domain.is_empty()) {
            None
        } else {
            exhaustive_triangle_supports(masks, domains)
        };

        assert_eq!(
            solve_triangle::<NativeSolverState>(Arc::clone(&model), variables, &restrictions,),
            expected,
            "native restrictions {restrictions:?}"
        );
        assert_eq!(
            solve_triangle::<ExhaustiveSolverState>(Arc::clone(&model), variables, &restrictions,),
            expected,
            "exhaustive restrictions {restrictions:?}"
        );
    }
}

#[test]
fn mixed_triangle_relations_match_exhaustive_projection() {
    for xy in 0_u16..16 {
        for yz in 0_u16..16 {
            let masks = [xy, yz];
            let (model, variables) = mixed_triangle_model(masks);
            let expected =
                solve_role_domains::<ExhaustiveSolverState>(Arc::clone(&model), &variables, &[]);
            let actual =
                solve_role_domains::<NativeSolverState>(Arc::clone(&model), &variables, &[]);
            assert_eq!(actual, expected, "mixed triangle relation masks {masks:?}");
        }
    }

    let equality = 0b1001;
    let (model, variables) = mixed_triangle_model([equality, equality]);
    for restrictions in partial_role_restrictions(&variables) {
        assert_eq!(
            solve_role_domains::<NativeSolverState>(Arc::clone(&model), &variables, &restrictions,),
            solve_role_domains::<ExhaustiveSolverState>(
                Arc::clone(&model),
                &variables,
                &restrictions,
            ),
            "mixed equality triangle restrictions {restrictions:?}"
        );
    }
}

#[test]
fn structural_only_restrictions_join_disconnected_semantic_relations() {
    let (model, variables) = split_semantic_core_model();

    for restrictions in partial_role_restrictions(&variables[4..]) {
        assert_eq!(
            solve_role_domains::<NativeSolverState>(Arc::clone(&model), &variables, &restrictions,),
            solve_role_domains::<ExhaustiveSolverState>(
                Arc::clone(&model),
                &variables,
                &restrictions,
            ),
            "structural-only restrictions {restrictions:?}"
        );
    }
}

#[test]
fn coupled_spanning_projectors_match_exhaustive_projection() {
    let (model, variables) = coupled_triangle_projectors_model();
    let initial = solve_role_domains::<ExhaustiveSolverState>(Arc::clone(&model), &variables, &[])
        .expect("cyclically coupled projectors must be satisfiable");
    assert_eq!(initial[0], BondRole::Traversal.singleton_domain());
    assert_eq!(initial[3], BondRole::Traversal.singleton_domain());

    for restrictions in partial_role_restrictions(&variables) {
        assert_eq!(
            solve_role_domains::<NativeSolverState>(Arc::clone(&model), &variables, &restrictions,),
            solve_role_domains::<ExhaustiveSolverState>(
                Arc::clone(&model),
                &variables,
                &restrictions,
            ),
            "coupled triangle projector restrictions {restrictions:?}"
        );
    }
}

#[test]
fn spanning_tree_factor_matches_independent_exhaustive_projection() {
    let fixtures: [(&str, Vec<SpanningFixture<'_>>); 7] = [
        (
            "single atom",
            vec![SpanningFixture {
                atoms: &[0],
                edges: &[],
            }],
        ),
        (
            "single bridge",
            vec![SpanningFixture {
                atoms: &[0, 1],
                edges: &[(0, 1)],
            }],
        ),
        (
            "triangle",
            vec![SpanningFixture {
                atoms: &[0, 1, 2],
                edges: &[(0, 1), (1, 2), (2, 0)],
            }],
        ),
        (
            "square",
            vec![SpanningFixture {
                atoms: &[0, 1, 2, 3],
                edges: &[(0, 1), (1, 2), (2, 3), (3, 0)],
            }],
        ),
        (
            "square with diagonal",
            vec![SpanningFixture {
                atoms: &[0, 1, 2, 3],
                edges: &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
            }],
        ),
        (
            "two cycles sharing a vertex",
            vec![SpanningFixture {
                atoms: &[0, 1, 2, 3, 4],
                edges: &[(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)],
            }],
        ),
        (
            "two disconnected components",
            vec![
                SpanningFixture {
                    atoms: &[0, 1, 2],
                    edges: &[(0, 1), (1, 2), (2, 0)],
                },
                SpanningFixture {
                    atoms: &[3, 4, 5],
                    edges: &[(3, 4), (4, 5), (5, 3)],
                },
            ],
        ),
    ];

    for (name, components) in fixtures {
        assert_spanning_fixture(name, &components);
    }
}

#[test]
fn multivalue_spanning_tree_partitions_match_exhaustive_projection() {
    let fixtures: [(&str, usize, &[(usize, usize)]); 4] = [
        ("triangle", 3, &[(0, 1), (1, 2), (2, 0)]),
        ("square", 4, &[(0, 1), (1, 2), (2, 3), (3, 0)]),
        (
            "square with diagonal",
            4,
            &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
        ),
        (
            "fused triangles",
            4,
            &[(0, 1), (1, 2), (2, 0), (1, 3), (2, 3)],
        ),
    ];

    for (name, atom_count, endpoints) in fixtures {
        let (model, variables) = multivalue_spanning_model(atom_count, endpoints);
        let native = <NativeSolverState as ConstraintSolver>::initial(Arc::clone(&model))
            .unwrap()
            .unwrap_consistent();
        let exhaustive = ExhaustiveSolverState::initial(model)
            .unwrap()
            .unwrap_consistent();
        for restrictions in multivalue_spanning_restrictions(&variables) {
            assert_eq!(
                restricted_domains(&native, &variables, &restrictions),
                restricted_domains(&exhaustive, &variables, &restrictions),
                "{name} with placement restrictions {restrictions:?}"
            );
        }
    }
}

#[test]
fn triangle_multivalue_projection_matches_every_combined_partial_domain() {
    let (model, variables) = multivalue_spanning_model(3, &[(0, 1), (1, 2), (2, 0)]);
    let native = <NativeSolverState as ConstraintSolver>::initial(Arc::clone(&model))
        .unwrap()
        .unwrap_consistent();
    let exhaustive = ExhaustiveSolverState::initial(model)
        .unwrap()
        .unwrap_consistent();

    for first in 1_u64..0b10_0000 {
        for second in 1_u64..0b10_0000 {
            for third in 1_u64..0b10_0000 {
                let restrictions = [
                    (variables[0], Domain::from_bits(first)),
                    (variables[1], Domain::from_bits(second)),
                    (variables[2], Domain::from_bits(third)),
                ];
                assert_eq!(
                    restricted_domains(&native, &variables, &restrictions),
                    restricted_domains(&exhaustive, &variables, &restrictions),
                    "combined placement restrictions {restrictions:?}"
                );
            }
        }
    }
}

#[test]
fn contraction_preserves_parallel_quotient_edges() {
    let (model, variables) = spanning_model(&[SpanningFixture {
        atoms: &[0, 1, 2],
        edges: &[(0, 1), (1, 2), (2, 0)],
    }]);
    let state = NativeSolverState::initial(Arc::clone(&model)).unwrap();
    let successor = state
        .restricted(&[(variables[0], BondRole::Traversal.singleton_domain())])
        .unwrap()
        .unwrap_consistent();

    assert_eq!(
        successor.domain(variables[0]),
        Some(BondRole::Traversal.singleton_domain())
    );
    assert_eq!(
        successor.domain(variables[1]),
        Some(BondRole::role_domain())
    );
    assert_eq!(
        successor.domain(variables[2]),
        Some(BondRole::role_domain())
    );
}

#[test]
fn contracted_internal_edge_is_forced_to_ring() {
    let (model, variables) = spanning_model(&[SpanningFixture {
        atoms: &[0, 1, 2],
        edges: &[(0, 1), (1, 2), (2, 0)],
    }]);
    let state = NativeSolverState::initial(model).unwrap();
    let successor = state
        .restricted(&[
            (variables[0], BondRole::Traversal.singleton_domain()),
            (variables[1], BondRole::Traversal.singleton_domain()),
        ])
        .unwrap()
        .unwrap_consistent();

    assert_eq!(
        successor.domain(variables[2]),
        Some(BondRole::Ring.singleton_domain())
    );
}

#[test]
fn forced_traversal_cycle_is_a_contradiction() {
    let (model, variables) = spanning_model(&[SpanningFixture {
        atoms: &[0, 1, 2],
        edges: &[(0, 1), (1, 2), (2, 0)],
    }]);
    let state = NativeSolverState::initial(model).unwrap();

    assert!(matches!(
        state
            .restricted(
                &variables
                    .iter()
                    .copied()
                    .map(|variable| (variable, BondRole::Traversal.singleton_domain()))
                    .collect::<Vec<_>>()
            )
            .unwrap(),
        Consistency::Contradiction
    ));
}

#[test]
fn removing_all_edges_at_one_vertex_is_a_contradiction() {
    let (model, variables) = spanning_model(&[SpanningFixture {
        atoms: &[0, 1, 2, 3],
        edges: &[(0, 1), (1, 2), (2, 3), (3, 0)],
    }]);
    let state = NativeSolverState::initial(model).unwrap();

    assert!(matches!(
        state
            .restricted(&[
                (variables[0], BondRole::Ring.singleton_domain()),
                (variables[3], BondRole::Ring.singleton_domain()),
            ])
            .unwrap(),
        Consistency::Contradiction
    ));
}
