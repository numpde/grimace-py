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
    let atom_domains = factor.atom_variables().iter().map(|atom| domains[atom.index()]).collect::<Vec<_>>();
    let edge_domains = factor
        .edges()
        .iter()
        .map(|edge| domains[edge.variable().index()])
        .collect::<Vec<_>>();
    if !matching_feasible(factor, &atom_domains, &edge_domains) {
        return Err(NativeSolverError::Contradiction);
    }
    let mut projected = Vec::with_capacity(factor.variables().len());
    for (index, domain) in atom_domains.iter().enumerate() {
        let mut support = Domain::empty();
        for value in domain.iter() {
            let mut trial = atom_domains.clone();
            trial[index] = Domain::singleton(value).unwrap();
            if matching_feasible(factor, &trial, &edge_domains) {
                support = support.union(Domain::singleton(value).unwrap());
            }
        }
        projected.push(support);
    }
    for (index, domain) in edge_domains.iter().copied().enumerate() {
        let mut support = Domain::empty();
        for value in domain.iter() {
            let mut trial = edge_domains.clone();
            trial[index] = Domain::singleton(value).unwrap();
            if matching_feasible(factor, &atom_domains, &trial) {
                support = support.union(Domain::singleton(value).unwrap());
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

pub(crate) fn matching_feasible(
    factor: &AromaticMatchingFactor,
    atom_domains: &[Domain],
    edge_domains: &[Domain],
) -> bool {
    if factor.uniform_case() {
        let source = atom_domains.iter().all(|domain| domain.contains(0))
            && edge_domains.iter().all(|domain| domain.contains(0))
            && has_feasible_double_assignment(factor.double_demands(), factor.edges());
        let kekule = atom_domains.iter().all(|domain| domain.contains(1))
            && matching_feasible_with_case(factor, edge_domains, |_| false);
        return source || kekule;
    }
    matching_feasible_with_case(factor, edge_domains, |atom| atom_domains[atom].contains(0))
}

fn matching_feasible_with_case(
    factor: &AromaticMatchingFactor,
    edge_domains: &[Domain],
    lower_allowed: impl Fn(usize) -> bool,
) -> bool {
    let latent = factor.edges().iter().zip(edge_domains).map(|(edge, &domain)| {
        let (a, b) = edge.endpoints();
        let aromatic = domain.contains(0) && lower_allowed(a) && lower_allowed(b);
        (domain.contains(1) || aromatic, domain.contains(2) || aromatic)
    }).collect::<Vec<_>>();
    if latent.iter().any(|&(single, double)| !single && !double) {
        return false;
    }
    let mut remaining = factor
        .double_demands()
        .iter()
        .map(|&demand| demand == 1)
        .collect::<Vec<_>>();
    // Mandatory doubles consume both endpoints before residual matching.
    for (index, edge) in factor.edges().iter().enumerate() {
        let (single, _) = latent[index];
        if !single {
            let (a, b) = edge.endpoints();
            if !remaining[a] || !remaining[b] {
                return false;
            }
            remaining[a] = false;
            remaining[b] = false;
        }
    }
    has_perfect_matching(
        &remaining,
        factor.edges().iter().enumerate().filter_map(|(index, edge)| {
            latent[index].1.then(|| edge.endpoints())
        }),
    )
}

pub(crate) fn has_feasible_double_assignment(
    double_demands: &[u8],
    edges: &[crate::model::AromaticMatchingEdge],
) -> bool {
    let remaining = double_demands.iter().map(|&demand| demand == 1).collect::<Vec<_>>();
    has_perfect_matching(&remaining, edges.iter().map(|edge| edge.endpoints()))
}

fn has_perfect_matching(
    remaining: &[bool],
    edges: impl Iterator<Item = (usize, usize)>,
) -> bool {
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
    for (a, b) in edges {
        if let (Some(a), Some(b)) = (residual_index[a], residual_index[b]) {
            graph[a].push(b);
            graph[b].push(a);
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
    use crate::model::{AromaticMatchingEdge, ConstraintModelBuilder, ConstraintModelError, FactorDefinition};

    fn check_projection(demands: &[u8], edges: &[(usize, usize)], atom_bits: &[u64], edge_bits: &[u64], uniform: bool) {
        let mut builder = ConstraintModelBuilder::new();
        let atoms = atom_bits.iter().map(|&bits| builder.add_variable(Domain::from_bits(bits)).unwrap()).collect::<Vec<_>>();
        let bonds = edge_bits.iter().map(|&bits| builder.add_variable(Domain::from_bits(bits)).unwrap()).collect::<Vec<_>>();
        let id = builder.add_aromatic_matching(atoms.clone(), uniform, demands.iter().copied(),
            edges.iter().zip(&bonds).map(|(&(a, b), &variable)| AromaticMatchingEdge::new(variable, a, b))).unwrap();
        let model = builder.build();
        let FactorDefinition::AromaticMatching(factor) = model.factor(id).unwrap() else { unreachable!() };
        let mut store = PagedStore::from_values(model.initial_domains());
        let mut expected = vec![0_u64; atoms.len() + bonds.len()];
        for mut cases in 0..(1_usize << atoms.len()) {
            let chosen = (0..atoms.len()).map(|_| { let value = (cases & 1) as u8; cases >>= 1; value }).collect::<Vec<_>>();
            if chosen.iter().enumerate().any(|(i, &case)| atom_bits[i] & (1 << case) == 0)
                || uniform && chosen.iter().any(|&case| case != chosen[0]) { continue; }
            for mut surfaces in 0..3_usize.pow(edges.len() as u32) {
                let values = (0..edges.len()).map(|_| { let value = (surfaces % 3) as u8; surfaces /= 3; value }).collect::<Vec<_>>();
                if values.iter().enumerate().any(|(i, &value)| edge_bits[i] & (1 << value) == 0
                    || value == 0 && (chosen[edges[i].0] != 0 || chosen[edges[i].1] != 0)
                    || uniform && ((chosen[0] == 0) != (value == 0))) { continue; }
                for mut latent in 0..(1_usize << edges.len()) {
                    let mut doubles = vec![0_u8; demands.len()];
                    let valid = values.iter().enumerate().all(|(i, &value)| {
                        let order = (latent & 1) as u8 + 1; latent >>= 1;
                        if value != 0 && value != order { return false; }
                        if order == 2 { doubles[edges[i].0] += 1; doubles[edges[i].1] += 1; }
                        true
                    });
                    if valid && doubles == demands {
                        for (i, &case) in chosen.iter().enumerate() { expected[i] |= 1 << case; }
                        for (i, &value) in values.iter().enumerate() { expected[chosen.len()+i] |= 1 << value; }
                    }
                }
            }
        }
        let result = revise_aromatic_matching(factor, &mut store);
        if expected.iter().all(|&bits| bits == 0) { assert_eq!(result, Err(NativeSolverError::Contradiction)); }
        else {
            result.unwrap();
            assert_eq!(factor.variables().iter().map(|v| store[v.index()]).collect::<Vec<_>>(), expected.into_iter().map(Domain::from_bits).collect::<Vec<_>>(),
                "demands={demands:?}, edges={edges:?}, atoms={atom_bits:?}, bonds={edge_bits:?}, uniform={uniform}");
        }
    }

    #[test]
    fn mixed_surface_projection_matches_finite_witness() {
        let edges = [(0, 1), (1, 2), (2, 3)];
        for mask in 0..16_u8 {
            let demands = (0..4).map(|i| (mask >> i) & 1).collect::<Vec<_>>();
            for atom_bits in [[3, 3, 3, 3], [1, 3, 3, 2], [2, 3, 3, 1]] {
                for edge_bits in [[7, 7, 7], [1, 7, 6], [6, 3, 5], [2, 4, 1]] {
                    for uniform in [false, true] {
                        check_projection(&demands, &edges, &atom_bits, &edge_bits, uniform);
                    }
                }
            }
        }
    }

    fn check_uniform_projection(demands: &[u8], edges: &[(usize, usize)], mode_bits: u64, edge_bits: &[u64]) {
        let mut builder = ConstraintModelBuilder::new();
        let atoms = (0..demands.len()).map(|_| builder.add_variable(Domain::from_bits(mode_bits)).unwrap()).collect::<Vec<_>>();
        let bonds = edge_bits.iter().map(|&bits| builder.add_variable(Domain::from_bits(bits)).unwrap()).collect::<Vec<_>>();
        let id = builder.add_aromatic_matching(atoms, true, demands.iter().copied(),
            edges.iter().zip(&bonds).map(|(&(a, b), &variable)| AromaticMatchingEdge::new(variable, a, b))).unwrap();
        let model = builder.build();
        let FactorDefinition::AromaticMatching(factor) = model.factor(id).unwrap() else { unreachable!() };
        let mut expected = vec![0_u64; demands.len() + edges.len()];
        for mode in 0..=1_u8 {
            if mode_bits & (1 << mode) == 0 { continue; }
            for doubles_mask in 0..(1_usize << edges.len()) {
                let mut doubles = vec![0_u8; demands.len()];
                let mut valid = true;
                for (index, &(a, b)) in edges.iter().enumerate() {
                    let double = doubles_mask & (1 << index) != 0;
                    let surface = if mode == 0 { 0 } else if double { 2 } else { 1 };
                    valid &= edge_bits[index] & (1 << surface) != 0;
                    if double { doubles[a] += 1; doubles[b] += 1; }
                }
                if valid && doubles == demands {
                    for support in &mut expected[..demands.len()] { *support |= 1 << mode; }
                    for (index, support) in expected[demands.len()..].iter_mut().enumerate() {
                        *support |= 1 << if mode == 0 { 0 } else if doubles_mask & (1 << index) != 0 { 2 } else { 1 };
                    }
                }
            }
        }
        let mut store = PagedStore::from_values(model.initial_domains());
        let result = revise_aromatic_matching(factor, &mut store);
        if expected.iter().all(|&bits| bits == 0) {
            assert_eq!(result, Err(NativeSolverError::Contradiction));
        } else {
            result.unwrap();
            assert_eq!(factor.variables().iter().map(|v| store[v.index()]).collect::<Vec<_>>(),
                expected.into_iter().map(Domain::from_bits).collect::<Vec<_>>(),
                "demands={demands:?}, edges={edges:?}, mode={mode_bits}, domains={edge_bits:?}");
        }
    }

    #[test]
    fn uniform_component_triangle_tail_all_domains_and_demands_match_witness() {
        let edges = [(0, 1), (1, 2), (2, 0), (1, 3)];
        for demands_mask in 0..16 {
            let demands = (0..4).map(|atom| ((demands_mask >> atom) & 1) as u8).collect::<Vec<_>>();
            for mode_bits in 1..=3 {
                for mut restriction_code in 0..7_usize.pow(4) {
                    let domains = (0..4).map(|_| {
                        let bits = restriction_code % 7 + 1;
                        restriction_code /= 7;
                        bits as u64
                    }).collect::<Vec<_>>();
                    check_uniform_projection(&demands, &edges, mode_bits, &domains);
                }
            }
        }
    }

    #[test]
    fn uniform_cycle_donor_and_fused_cases_match_witness() {
        let benzene = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)];
        check_uniform_projection(&[1; 6], &benzene, 3, &[7; 6]);
        check_uniform_projection(&[1; 6], &benzene, 2, &[4, 6, 6, 6, 6, 6]);
        check_uniform_projection(&[1; 6], &benzene, 3, &[1, 7, 7, 7, 7, 7]);
        let donor = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)];
        check_uniform_projection(&[0, 1, 1, 1, 1], &donor, 3, &[7; 5]);
        let fused = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
            (4, 6), (6, 7), (7, 8), (8, 9), (9, 5)];
        check_uniform_projection(&[1; 10], &fused, 3, &[7; 11]);
    }

    #[test]
    fn blossom_agrees_with_small_exhaustive_matching() {
        fn brute(graph: &[Vec<usize>], remaining: u64) -> bool {
            if remaining == 0 { return true; }
            let a = remaining.trailing_zeros() as usize;
            graph[a].iter().any(|&b| remaining & (1 << b) != 0
                && brute(graph, remaining & !(1 << a) & !(1 << b)))
        }
        for n in 0..=6 {
            let pairs = (0..n).flat_map(|a| (a+1..n).map(move |b| (a,b))).collect::<Vec<_>>();
            for mask in 0..1_usize << pairs.len() {
                let mut graph = vec![Vec::new(); n];
                for (i, &(a,b)) in pairs.iter().enumerate() {
                    if mask & (1 << i) != 0 { graph[a].push(b); graph[b].push(a); }
                }
                assert_eq!(BlossomMatching::new(&graph).has_perfect_matching(), brute(&graph, (1 << n)-1));
            }
        }
    }

    #[test]
    fn matching_registration_rejects_malformed_scope_and_reuse() {
        let mut builder = ConstraintModelBuilder::new();
        let a = builder.add_variable(Domain::from_bits(3)).unwrap();
        let b = builder.add_variable(Domain::from_bits(3)).unwrap();
        let edge = builder.add_variable(Domain::from_bits(7)).unwrap();
        let invalid = builder.add_variable(Domain::from_bits(8)).unwrap();
        let other = builder.add_variable(Domain::from_bits(7)).unwrap();
        assert!(matches!(builder.add_aromatic_matching([], false, [], []),
            Err(ConstraintModelError::InvalidAromaticMatchingGraph(_))));
        assert!(matches!(builder.add_aromatic_matching([a], false, [2], []),
            Err(ConstraintModelError::InvalidAromaticMatchingGraph(_))));
        assert!(matches!(builder.add_aromatic_matching([a], false, [1, 1], []),
            Err(ConstraintModelError::InvalidAromaticMatchingGraph(_))));
        for endpoints in [(0, 0), (0, 2)] {
            assert!(matches!(builder.add_aromatic_matching([a, b], false, [1, 1],
                [AromaticMatchingEdge::new(edge, endpoints.0, endpoints.1)]),
                Err(ConstraintModelError::InvalidAromaticMatchingGraph(_))));
        }
        assert_eq!(builder.add_aromatic_matching([invalid], false, [0], []),
            Err(ConstraintModelError::InvalidAromaticMatchingDomain(invalid)));
        assert_eq!(builder.add_aromatic_matching([a, b], false, [1, 1],
            [AromaticMatchingEdge::new(invalid, 0, 1)]),
            Err(ConstraintModelError::InvalidAromaticMatchingDomain(invalid)));
        assert_eq!(builder.add_aromatic_matching([a, a], false, [0, 0], []),
            Err(ConstraintModelError::RepeatedVariableInFactor(a)));
        builder.add_aromatic_matching([a, b], false, [1, 1],
            [AromaticMatchingEdge::new(edge, 0, 1)]).unwrap();
        for variable in [a, b, edge] {
            assert_eq!(builder.add_binary_relation(variable, other, [(0, 0)]),
                Err(ConstraintModelError::LocalProjectorLeafReuse(variable)));
            assert_eq!(builder.add_aromatic_matching([variable], false, [0], []),
                Err(ConstraintModelError::LocalProjectorLeafReuse(variable)));
        }
        let mut reverse = ConstraintModelBuilder::new();
        let a = reverse.add_variable(Domain::from_bits(3)).unwrap();
        let b = reverse.add_variable(Domain::from_bits(3)).unwrap();
        let edge = reverse.add_variable(Domain::from_bits(7)).unwrap();
        let other = reverse.add_variable(Domain::from_bits(7)).unwrap();
        reverse.add_binary_relation(edge, other, [(0, 0)]).unwrap();
        assert_eq!(reverse.add_aromatic_matching([a, b], false, [1, 1],
            [AromaticMatchingEdge::new(edge, 0, 1)]),
            Err(ConstraintModelError::LocalProjectorLeafReuse(edge)));
    }
}
