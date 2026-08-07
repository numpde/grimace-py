//! Exhaustive tiny-CSP backend and differential checks for the solver contract.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::{AtomId, FactorId, VariableId};
use crate::model::{
    BondRole, ConstraintModel, ConstraintModelBuilder, FactorDefinition, SpanningTreeEdge,
    SpanningTreeFactor,
};
use crate::native::NativeSolverState;
use crate::solver::ConstraintSolver;

#[derive(Clone, Debug)]
struct ExhaustiveSolverState {
    model: Arc<ConstraintModel>,
    domains: Box<[Domain]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ExhaustiveSolverError {
    UnknownVariable(VariableId),
    Contradiction,
}

impl fmt::Display for ExhaustiveSolverError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownVariable(variable) => {
                write!(formatter, "unknown constraint variable {variable:?}")
            }
            Self::Contradiction => formatter.write_str("constraint state is contradictory"),
        }
    }
}

impl std::error::Error for ExhaustiveSolverError {}

impl ConstraintSolver for ExhaustiveSolverState {
    type Error = ExhaustiveSolverError;

    fn initial(model: Arc<ConstraintModel>) -> Result<Self, Self::Error> {
        let initial = model.initial_domains().collect::<Vec<_>>();
        let domains = exhaustive_projected_domains(model.as_ref(), &initial)
            .ok_or(ExhaustiveSolverError::Contradiction)?;
        Ok(Self { model, domains })
    }

    fn restricted(&self, restrictions: &[(VariableId, Domain)]) -> Result<Self, Self::Error> {
        let mut requested = BTreeMap::new();
        let mut contradictory = false;

        for &(variable, allowed) in restrictions {
            let current = self
                .domains
                .get(variable.index())
                .copied()
                .ok_or(ExhaustiveSolverError::UnknownVariable(variable))?;
            let restricted = requested.entry(variable).or_insert(current);
            *restricted = restricted.intersect(allowed);
            contradictory |= restricted.is_empty();
        }
        if contradictory {
            return Err(ExhaustiveSolverError::Contradiction);
        }

        let mut candidate = self.domains.to_vec();
        for (variable, restricted) in requested {
            candidate[variable.index()] = restricted;
        }
        let domains = exhaustive_projected_domains(self.model.as_ref(), &candidate)
            .ok_or(ExhaustiveSolverError::Contradiction)?;
        Ok(Self {
            model: Arc::clone(&self.model),
            domains,
        })
    }

    fn domain(&self, variable: VariableId) -> Option<Domain> {
        self.domains.get(variable.index()).copied()
    }
}

fn exhaustive_projected_domains(
    model: &ConstraintModel,
    domains: &[Domain],
) -> Option<Box<[Domain]>> {
    assert_eq!(domains.len(), model.variable_count());

    let mut assignment = vec![0_u8; domains.len()];
    let mut supported = vec![Domain::empty(); domains.len()];
    let mut found = false;
    enumerate_assignments(
        model,
        domains,
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
    variable: usize,
    assignment: &mut [u8],
    supported: &mut [Domain],
    found: &mut bool,
) {
    if variable == domains.len() {
        if !assignment_satisfies(model, assignment) {
            return;
        }
        *found = true;
        for (projection, value) in supported.iter_mut().zip(assignment.iter().copied()) {
            *projection = projection.union(Domain::from_bits(1_u64 << value));
        }
        return;
    }

    for value in domains[variable].iter() {
        assignment[variable] = value;
        enumerate_assignments(model, domains, variable + 1, assignment, supported, found);
    }
}

fn assignment_satisfies(model: &ConstraintModel, assignment: &[u8]) -> bool {
    (0..model.factor_count()).all(|index| {
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
        }
    })
}

fn assignment_satisfies_spanning_tree(factor: &SpanningTreeFactor, assignment: &[u8]) -> bool {
    let traversal_value = BondRole::Traversal.value_index();
    let ring_value = BondRole::Ring.value_index();
    let mut traversal_edge_count = 0;

    for edge in factor.edges() {
        match assignment[edge.role_variable().index()] {
            value if value == traversal_value => traversal_edge_count += 1,
            value if value == ring_value => {}
            _ => return false,
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
            if assignment[edge.role_variable().index()] != traversal_value {
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
    let state = S::initial(model).ok()?;
    let state = state.restricted(restrictions).ok()?;
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

fn solve_role_domains<S: ConstraintSolver>(
    model: Arc<ConstraintModel>,
    variables: &[VariableId],
    restrictions: &[(VariableId, Domain)],
) -> Option<Vec<Domain>> {
    let state = S::initial(model).ok()?;
    let state = state.restricted(restrictions).ok()?;
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
fn contraction_preserves_parallel_quotient_edges() {
    let (model, variables) = spanning_model(&[SpanningFixture {
        atoms: &[0, 1, 2],
        edges: &[(0, 1), (1, 2), (2, 0)],
    }]);
    let state = NativeSolverState::initial(Arc::clone(&model)).unwrap();
    let successor = state
        .restricted(&[(variables[0], BondRole::Traversal.singleton_domain())])
        .unwrap();

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
        .unwrap();

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

    assert!(state
        .restricted(
            &variables
                .iter()
                .copied()
                .map(|variable| (variable, BondRole::Traversal.singleton_domain()))
                .collect::<Vec<_>>()
        )
        .is_err());
}

#[test]
fn removing_all_edges_at_one_vertex_is_a_contradiction() {
    let (model, variables) = spanning_model(&[SpanningFixture {
        atoms: &[0, 1, 2, 3],
        edges: &[(0, 1), (1, 2), (2, 3), (3, 0)],
    }]);
    let state = NativeSolverState::initial(model).unwrap();

    assert!(state
        .restricted(&[
            (variables[0], BondRole::Ring.singleton_domain()),
            (variables[3], BondRole::Ring.singleton_domain()),
        ])
        .is_err());
}
