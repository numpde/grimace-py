//! Exact aromatic/Kekule domain projection through general graph matching.

use std::collections::VecDeque;

use crate::domain::Domain;
use crate::ids::VariableId;
use crate::model::AromaticMatchingFactor;
use crate::native::NativeSolverError;
use crate::persistent::PagedStore;

pub(crate) fn revise_aromatic_matching(
    factor: &AromaticMatchingFactor,
    domains: &mut PagedStore<Domain>,
) -> Result<Vec<VariableId>, NativeSolverError> {
    let mode = domains[factor.mode_variable().index()];
    let edge_domains = factor
        .edges()
        .iter()
        .map(|edge| domains[edge.variable().index()])
        .collect::<Vec<_>>();
    let aromatic = mode.contains(0) && edge_domains.iter().all(|domain| domain.contains(0));
    let kekule = mode.contains(1) && kekule_feasible(factor, &edge_domains, None);
    if !aromatic && !kekule {
        return Err(NativeSolverError::Contradiction);
    }
    let mut projected = Vec::with_capacity(factor.variables().len());
    projected.push(Domain::from_bits(
        u64::from(aromatic) | (u64::from(kekule) << 1),
    ));
    for (index, domain) in edge_domains.iter().copied().enumerate() {
        let mut support = Domain::from_bits(u64::from(aromatic));
        if kekule {
            for value in [1, 2] {
                if domain.contains(value)
                    && kekule_feasible(factor, &edge_domains, Some((index, value)))
                {
                    support = support.union(Domain::from_bits(1 << value));
                }
            }
        }
        projected.push(support);
    }
    let mut changed = Vec::new();
    for (&variable, support) in factor.variables().iter().zip(projected) {
        if domains[variable.index()] != support {
            domains[variable.index()] = support;
            changed.push(variable);
        }
    }
    Ok(changed)
}

fn kekule_feasible(
    factor: &AromaticMatchingFactor,
    domains: &[Domain],
    commitment: Option<(usize, u8)>,
) -> bool {
    let edge_domain = |index: usize| match commitment {
        Some((committed, value)) if committed == index => Domain::from_bits(1 << value),
        _ => domains[index],
    };
    let mut remaining = factor
        .double_demands()
        .iter()
        .map(|&demand| demand == 1)
        .collect::<Vec<_>>();
    // Mandatory doubles consume both endpoints before residual matching.
    for (index, edge) in factor.edges().iter().enumerate() {
        let domain = edge_domain(index);
        if !domain.contains(1) {
            if !domain.contains(2) {
                return false;
            }
            let (a, b) = edge.endpoints();
            if !remaining[a] || !remaining[b] {
                return false;
            }
            remaining[a] = false;
            remaining[b] = false;
        }
    }
    let mut residual_index = vec![None; remaining.len()];
    let mut count = 0;
    for (atom, &required) in remaining.iter().enumerate() {
        if required {
            residual_index[atom] = Some(count);
            count += 1;
        }
    }
    if count % 2 != 0 {
        return false;
    }
    let mut graph = vec![Vec::new(); count];
    for (index, edge) in factor.edges().iter().enumerate() {
        if edge_domain(index).contains(2) {
            let (a, b) = edge.endpoints();
            if let (Some(a), Some(b)) = (residual_index[a], residual_index[b]) {
                graph[a].push(b);
                graph[b].push(a);
            }
        }
    }
    BlossomMatching::new(&graph).has_perfect_matching()
}

/// Edmonds' alternating forest: bases contract odd cycles during one search;
/// parent links retain the original vertices needed to expand augmentation.
struct BlossomMatching<'a> {
    graph: &'a [Vec<usize>],
    partner: Vec<Option<usize>>,
    parent: Vec<Option<usize>>,
    base: Vec<usize>,
    outer: Vec<bool>,
}

impl<'a> BlossomMatching<'a> {
    fn new(graph: &'a [Vec<usize>]) -> Self {
        let n = graph.len();
        Self {
            graph,
            partner: vec![None; n],
            parent: vec![None; n],
            base: (0..n).collect(),
            outer: vec![false; n],
        }
    }

    fn has_perfect_matching(mut self) -> bool {
        for root in 0..self.graph.len() {
            if self.partner[root].is_none() {
                let Some(mut endpoint) = self.augmenting_endpoint(root) else {
                    continue;
                };
                while let Some(parent) = self.parent[endpoint] {
                    let next = self.partner[parent];
                    self.partner[endpoint] = Some(parent);
                    self.partner[parent] = Some(endpoint);
                    let Some(next) = next else {
                        break;
                    };
                    endpoint = next;
                }
            }
        }
        self.partner.iter().all(Option::is_some)
    }

    fn common_base(&self, mut a: usize, mut b: usize) -> usize {
        let mut ancestors = vec![false; self.graph.len()];
        loop {
            a = self.base[a];
            ancestors[a] = true;
            let Some(partner) = self.partner[a] else {
                break;
            };
            a = self.parent[partner].expect("matched outer vertex has a forest predecessor");
        }
        loop {
            b = self.base[b];
            if ancestors[b] {
                return b;
            }
            let partner =
                self.partner[b].expect("blossom paths meet before an unmatched second root");
            b = self.parent[partner].expect("matched outer vertex has a forest predecessor");
        }
    }

    fn mark_blossom_path(
        &mut self,
        mut vertex: usize,
        base: usize,
        mut child: usize,
        blossom: &mut [bool],
    ) {
        while self.base[vertex] != base {
            let partner = self.partner[vertex].expect("non-base blossom vertex has a partner");
            blossom[self.base[vertex]] = true;
            blossom[self.base[partner]] = true;
            self.parent[vertex] = Some(child);
            child = partner;
            vertex =
                self.parent[partner].expect("blossom path reaches its base through the forest");
        }
    }

    fn augmenting_endpoint(&mut self, root: usize) -> Option<usize> {
        self.parent.fill(None);
        self.outer.fill(false);
        for (vertex, base) in self.base.iter_mut().enumerate() {
            *base = vertex;
        }
        let mut queue = VecDeque::from([root]);
        self.outer[root] = true;
        while let Some(vertex) = queue.pop_front() {
            for neighbour_index in 0..self.graph[vertex].len() {
                let neighbour = self.graph[vertex][neighbour_index];
                if self.base[vertex] == self.base[neighbour]
                    || self.partner[vertex] == Some(neighbour)
                {
                    continue;
                }
                let closes_blossom = neighbour == root
                    || self.partner[neighbour]
                        .is_some_and(|partner| self.parent[partner].is_some());
                if closes_blossom {
                    let base = self.common_base(vertex, neighbour);
                    let mut blossom = vec![false; self.graph.len()];
                    self.mark_blossom_path(vertex, base, neighbour, &mut blossom);
                    self.mark_blossom_path(neighbour, base, vertex, &mut blossom);
                    for atom in 0..self.graph.len() {
                        if blossom[self.base[atom]] {
                            self.base[atom] = base;
                            if !self.outer[atom] {
                                self.outer[atom] = true;
                                queue.push_back(atom);
                            }
                        }
                    }
                } else if self.parent[neighbour].is_none() {
                    self.parent[neighbour] = Some(vertex);
                    let Some(partner) = self.partner[neighbour] else {
                        return Some(neighbour);
                    };
                    self.outer[partner] = true;
                    queue.push_back(partner);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        AromaticMatchingEdge, ConstraintModelBuilder, ConstraintModelError, FactorDefinition,
    };

    fn fixture(
        demands: &[u8],
        edges: &[(usize, usize)],
        mode: Domain,
        edge_domains: &[Domain],
    ) -> (AromaticMatchingFactor, PagedStore<Domain>) {
        let mut builder = ConstraintModelBuilder::new();
        let mode_variable = builder.add_variable(mode).unwrap();
        let variables = edge_domains
            .iter()
            .map(|&domain| builder.add_variable(domain).unwrap())
            .collect::<Vec<_>>();
        let id = builder
            .add_aromatic_matching(
                mode_variable,
                demands.iter().copied(),
                edges
                    .iter()
                    .zip(variables)
                    .map(|(&(a, b), variable)| AromaticMatchingEdge::new(variable, a, b)),
            )
            .unwrap();
        let model = builder.build();
        let FactorDefinition::AromaticMatching(factor) = model.factor(id).unwrap() else {
            panic!("wrong factor")
        };
        (
            factor.clone(),
            PagedStore::from_values(model.initial_domains()),
        )
    }

    // Independent test witness enumerates tiny edge assignments, never used by runtime.
    fn brute_support(
        demands: &[u8],
        edges: &[(usize, usize)],
        mode: Domain,
        domains: &[Domain],
    ) -> Vec<Domain> {
        let mut support = vec![Domain::empty(); edges.len() + 1];
        for chosen_mode in mode.iter() {
            let count = 3_usize.pow(edges.len() as u32);
            for mut code in 0..count {
                let mut values = Vec::new();
                let mut doubles = vec![0_u8; demands.len()];
                let mut valid = true;
                for (index, &(a, b)) in edges.iter().enumerate() {
                    let value = (code % 3) as u8;
                    code /= 3;
                    values.push(value);
                    valid &= domains[index].contains(value)
                        && if chosen_mode == 0 {
                            value == 0
                        } else {
                            value == 1 || value == 2
                        };
                    if value == 2 {
                        doubles[a] += 1;
                        doubles[b] += 1;
                    }
                }
                valid &= chosen_mode == 0 || doubles == demands;
                if valid {
                    support[0] = support[0].union(Domain::from_bits(1 << chosen_mode));
                    for (index, value) in values.into_iter().enumerate() {
                        support[index + 1] =
                            support[index + 1].union(Domain::from_bits(1 << value));
                    }
                }
            }
        }
        support
    }

    fn assert_projection(
        demands: &[u8],
        edges: &[(usize, usize)],
        mode: Domain,
        domains: &[Domain],
    ) {
        let expected = brute_support(demands, edges, mode, domains);
        let (factor, mut store) = fixture(demands, edges, mode, domains);
        let before = factor
            .variables()
            .iter()
            .map(|v| store[v.index()])
            .collect::<Vec<_>>();
        let actual = revise_aromatic_matching(&factor, &mut store);
        if expected[0].is_empty() {
            assert_eq!(
                actual,
                Err(NativeSolverError::Contradiction),
                "demands={demands:?} edges={edges:?} domains={domains:?}"
            );
            assert_eq!(
                before,
                factor
                    .variables()
                    .iter()
                    .map(|v| store[v.index()])
                    .collect::<Vec<_>>()
            );
        } else {
            assert_eq!(
                expected,
                factor
                    .variables()
                    .iter()
                    .map(|v| store[v.index()])
                    .collect::<Vec<_>>(),
                "demands={demands:?} edges={edges:?} domains={domains:?}"
            );
            let changed = actual.unwrap();
            assert_eq!(
                changed,
                factor
                    .variables()
                    .iter()
                    .zip(before)
                    .zip(&expected)
                    .filter_map(|((&v, old), &new)| (old != new).then_some(v))
                    .collect::<Vec<_>>()
            );
            assert!(revise_aromatic_matching(&factor, &mut store)
                .unwrap()
                .is_empty());
        }
    }

    #[test]
    fn aromatic_matching_cycles_donors_fused_and_forced_edges() {
        let all = Domain::from_bits(7);
        let kekule = Domain::from_bits(6);
        let mode = Domain::from_bits(3);
        let only_kekule = Domain::from_bits(2);
        let benzene = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)];
        assert_projection(&[1; 6], &benzene, mode, &[all; 6]);
        assert_projection(&[1; 6], &benzene, only_kekule, &[kekule; 6]);
        assert_projection(
            &[0, 1, 1, 1, 1],
            &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
            mode,
            &[all; 5],
        );
        assert_projection(
            &[1; 3],
            &[(0, 1), (1, 2), (2, 0)],
            only_kekule,
            &[kekule; 3],
        );
        let fused = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (4, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 5),
        ];
        assert_projection(&[1; 10], &fused, mode, &[all; 11]);
        let mut forced = [kekule; 6];
        forced[0] = Domain::from_bits(4);
        forced[1] = Domain::from_bits(4);
        assert_projection(&[1; 6], &benzene, only_kekule, &forced);
        let mut forbidden = [kekule; 6];
        forbidden[0] = Domain::from_bits(2);
        forbidden[5] = Domain::from_bits(2);
        assert_projection(&[1; 6], &benzene, only_kekule, &forbidden);
        let mut aromatic_only = [all; 6];
        aromatic_only[0] = Domain::from_bits(1);
        assert_projection(&[1; 6], &benzene, mode, &aromatic_only);
    }

    #[test]
    fn aromatic_matching_all_tiny_graphs_agree_with_independent_matching_witness() {
        fn brute_matching(graph: &[Vec<usize>], remaining: u64) -> bool {
            if remaining == 0 {
                return true;
            }
            let a = remaining.trailing_zeros() as usize;
            graph[a].iter().any(|&b| {
                remaining & (1 << b) != 0
                    && brute_matching(graph, remaining & !(1 << a) & !(1 << b))
            })
        }
        for n in 0..=6 {
            let pairs = (0..n)
                .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
                .collect::<Vec<_>>();
            for mask in 0..1_usize << pairs.len() {
                let mut graph = vec![Vec::new(); n];
                for (index, &(a, b)) in pairs.iter().enumerate() {
                    if mask & (1 << index) != 0 {
                        graph[a].push(b);
                        graph[b].push(a);
                    }
                }
                assert_eq!(
                    BlossomMatching::new(&graph).has_perfect_matching(),
                    brute_matching(&graph, (1 << n) - 1),
                    "graph={graph:?}"
                );
            }
        }
        // Root 2 must contract 0-1-2 to reach the free tail through vertex 1.
        let graph = vec![vec![1, 2], vec![0, 2, 3], vec![0, 1], vec![1]];
        assert!(BlossomMatching::new(&graph).has_perfect_matching());
    }

    #[test]
    fn aromatic_matching_all_domain_restrictions_and_demands_on_triangle_with_tail() {
        let edges = [(0, 1), (1, 2), (2, 0), (1, 3)];
        for demands_mask in 0..16 {
            let demands = (0..4)
                .map(|a| ((demands_mask >> a) & 1) as u8)
                .collect::<Vec<_>>();
            for mode_bits in 1..=3 {
                for mut restriction_code in 0..7_usize.pow(4) {
                    let domains = (0..4)
                        .map(|_| {
                            let bits = restriction_code % 7 + 1;
                            restriction_code /= 7;
                            Domain::from_bits(bits as u64)
                        })
                        .collect::<Vec<_>>();
                    assert_projection(&demands, &edges, Domain::from_bits(mode_bits), &domains);
                }
            }
        }
    }

    #[test]
    fn aromatic_matching_scope_is_exclusive_in_both_registration_orders() {
        let mut b = ConstraintModelBuilder::new();
        let mode = b.add_variable(Domain::from_bits(3)).unwrap();
        let edge = b.add_variable(Domain::from_bits(7)).unwrap();
        let other = b.add_variable(Domain::from_bits(7)).unwrap();
        b.add_aromatic_matching(mode, [1, 1], [AromaticMatchingEdge::new(edge, 0, 1)])
            .unwrap();
        for variable in [mode, edge] {
            assert_eq!(
                b.add_binary_relation(variable, other, [(0, 0)]),
                Err(ConstraintModelError::LocalProjectorLeafReuse(variable))
            );
            assert_eq!(
                b.add_aromatic_matching(variable, [0], []),
                Err(ConstraintModelError::LocalProjectorLeafReuse(variable))
            );
        }
        let mut b = ConstraintModelBuilder::new();
        let mode = b.add_variable(Domain::from_bits(3)).unwrap();
        let edge = b.add_variable(Domain::from_bits(7)).unwrap();
        let other = b.add_variable(Domain::from_bits(7)).unwrap();
        b.add_binary_relation(edge, other, [(0, 0)]).unwrap();
        assert_eq!(
            b.add_aromatic_matching(mode, [1, 1], [AromaticMatchingEdge::new(edge, 0, 1)]),
            Err(ConstraintModelError::LocalProjectorLeafReuse(edge))
        );
    }

    #[test]
    fn aromatic_matching_malformed_inputs_fail_at_model_boundary() {
        let mut b = ConstraintModelBuilder::new();
        let mode = b.add_variable(Domain::from_bits(3)).unwrap();
        let edge = b.add_variable(Domain::from_bits(7)).unwrap();
        let invalid = b.add_variable(Domain::from_bits(8)).unwrap();
        assert!(matches!(
            b.add_aromatic_matching(mode, [], []),
            Err(ConstraintModelError::InvalidAromaticMatchingGraph(_))
        ));
        assert!(matches!(
            b.add_aromatic_matching(mode, [2], []),
            Err(ConstraintModelError::InvalidAromaticMatchingGraph(_))
        ));
        for endpoints in [(0, 0), (0, 2)] {
            assert!(matches!(
                b.add_aromatic_matching(
                    mode,
                    [1, 1],
                    [AromaticMatchingEdge::new(edge, endpoints.0, endpoints.1)]
                ),
                Err(ConstraintModelError::InvalidAromaticMatchingGraph(_))
            ));
        }
        assert_eq!(
            b.add_aromatic_matching(invalid, [0], []),
            Err(ConstraintModelError::InvalidAromaticMatchingDomain(invalid))
        );
        assert_eq!(
            b.add_aromatic_matching(mode, [1, 1], [AromaticMatchingEdge::new(invalid, 0, 1)]),
            Err(ConstraintModelError::InvalidAromaticMatchingDomain(invalid))
        );
        assert_eq!(
            b.add_aromatic_matching(mode, [1, 1], [AromaticMatchingEdge::new(mode, 0, 1)]),
            Err(ConstraintModelError::RepeatedVariableInFactor(mode))
        );
    }
}
