//! Exhaustive tiny-CSP backend and differential checks for the solver contract.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::{AtomId, FactorId, VariableId};
use crate::model::{
    BondRole, ConstraintModel, ConstraintModelBuilder, EdgeRolePartition, FactorDefinition,
    SpanningTreeEdge, SpanningTreeFactor,
};
use crate::native::NativeSolverState;
use crate::solver::{Consistency, ConstraintSolver};

#[derive(Clone, Debug)]
struct ExhaustiveSolverState {
    model: Arc<ConstraintModel>,
    domains: Box<[Domain]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ExhaustiveSolverFailure {
    UnknownVariable(VariableId),
}

impl fmt::Display for ExhaustiveSolverFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownVariable(variable) => {
                write!(formatter, "unknown constraint variable {variable:?}")
            }
        }
    }
}

impl std::error::Error for ExhaustiveSolverFailure {}

impl ConstraintSolver for ExhaustiveSolverState {
    type Failure = ExhaustiveSolverFailure;

    fn initial(model: Arc<ConstraintModel>) -> Result<Consistency<Self>, Self::Failure> {
        let initial = model.initial_domains().collect::<Vec<_>>();
        Ok(
            match exhaustive_projected_domains(model.as_ref(), &initial) {
                Some(domains) => Consistency::Consistent(Self { model, domains }),
                None => Consistency::Contradiction,
            },
        )
    }

    fn restricted(
        &self,
        restrictions: &[(VariableId, Domain)],
    ) -> Result<Consistency<Self>, Self::Failure> {
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
            match exhaustive_projected_domains(self.model.as_ref(), &candidate) {
                Some(domains) => Consistency::Consistent(Self {
                    model: Arc::clone(&self.model),
                    domains,
                }),
                None => Consistency::Contradiction,
            },
        )
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
            FactorDefinition::TetrahedralLayout(_) => true,
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
