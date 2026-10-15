//! Test-only nice decomposition of physical cut loads. Elimination fill is
//! structural: only original edge IDs ever introduce load to the relation.

use std::collections::BTreeSet;

use crate::boundary_cut_relation::{Graph, Relation};

#[derive(Clone, Debug, Eq, PartialEq)]
enum Step {
    Leaf,
    IntroduceVertex(usize, usize),
    IntroduceEdge(usize, usize),
    Forget(usize, usize),
    Join(usize, usize),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Node {
    step: Step,
    bag: BTreeSet<usize>,
    parent: Option<usize>,
}

struct Plan {
    graph: Graph,
    anchor: Option<usize>,
    nodes: Vec<Node>,
    root: usize,
    edge_owner: Vec<usize>,
}

impl Plan {
    fn prepare(graph: Graph, anchor: Option<usize>) -> Self {
        assert!(anchor.is_none_or(|z| z < graph.vertices));
        let order = elimination_order(&graph, anchor);
        Self::with_order(graph, anchor, order)
    }

    // A test seam for proving that a different valid decomposition does not
    // constrain the writer's atom orders.
    fn with_order(graph: Graph, anchor: Option<usize>, order: Vec<usize>) -> Self {
        assert!(anchor.is_none_or(|z| z < graph.vertices));
        let expected: BTreeSet<_> = (0..graph.vertices).filter(|v| Some(*v) != anchor).collect();
        assert_eq!(order.iter().copied().collect::<BTreeSet<_>>(), expected);
        assert_eq!(order.len(), expected.len());
        let mut adjacent = filled_adjacency(&graph, anchor);
        let mut bags = vec![BTreeSet::new(); graph.vertices];
        for &v in &order {
            let neighbors = adjacent[v].clone();
            bags[v] = neighbors.union(&BTreeSet::from([v])).copied().collect();
            for &u in &neighbors {
                adjacent[u].remove(&v);
                adjacent[u].extend(neighbors.iter().copied().filter(|&w| w != u));
            }
            adjacent[v].clear();
        }
        let mut rank = vec![order.len(); graph.vertices];
        for (i, &v) in order.iter().enumerate() {
            rank[v] = i;
        }
        let mut children = vec![Vec::new(); graph.vertices + 1];
        for &v in &order {
            let parent = bags[v]
                .iter()
                .copied()
                .filter(|&u| u != v)
                .min_by_key(|&u| (rank[u], u))
                .unwrap_or(graph.vertices);
            children[parent].push(v);
        }
        let mut owned_edges = vec![Vec::new(); graph.vertices];
        for (edge, &(u, v)) in graph.edges.iter().enumerate() {
            let first = if rank[u] < rank[v] { u } else { v };
            assert!(rank[first] < order.len());
            assert!(bags[first].contains(&u) && bags[first].contains(&v));
            owned_edges[first].push(edge);
        }
        let mut plan = Self {
            edge_owner: vec![usize::MAX; graph.edges.len()],
            graph,
            anchor,
            nodes: Vec::new(),
            root: usize::MAX,
        };
        let root_bag: BTreeSet<_> = anchor.into_iter().collect();
        let mut root = None;
        for &child in &children[anchor.unwrap_or(plan.graph.vertices)] {
            let piece = plan.build(child, &root_bag, &bags, &children, &owned_edges);
            root = Some(match root {
                None => piece,
                Some(previous) => plan.push(Step::Join(previous, piece)),
            });
        }
        plan.root = root.unwrap_or_else(|| plan.push(Step::Leaf));
        plan.validate();
        plan
    }

    fn push(&mut self, step: Step) -> usize {
        let bag = match step {
            Step::Leaf => self.anchor.into_iter().collect(),
            Step::IntroduceVertex(v, child) => {
                assert!(!self.nodes[child].bag.contains(&v));
                let mut bag = self.nodes[child].bag.clone();
                bag.insert(v);
                bag
            }
            Step::IntroduceEdge(edge, child) => {
                let (u, v) = self.graph.edges[edge];
                assert!(self.nodes[child].bag.contains(&u) && self.nodes[child].bag.contains(&v));
                self.nodes[child].bag.clone()
            }
            Step::Forget(v, child) => {
                assert!(self.nodes[child].bag.contains(&v) && self.anchor != Some(v));
                let mut bag = self.nodes[child].bag.clone();
                bag.remove(&v);
                bag
            }
            Step::Join(left, right) => {
                assert_eq!(self.nodes[left].bag, self.nodes[right].bag);
                self.nodes[left].bag.clone()
            }
        };
        let id = self.nodes.len();
        match step {
            Step::Leaf => {}
            Step::IntroduceVertex(_, child)
            | Step::IntroduceEdge(_, child)
            | Step::Forget(_, child) => {
                assert!(self.nodes[child].parent.replace(id).is_none());
            }
            Step::Join(left, right) => {
                assert!(self.nodes[left].parent.replace(id).is_none());
                assert!(self.nodes[right].parent.replace(id).is_none());
            }
        }
        self.nodes.push(Node {
            step,
            bag,
            parent: None,
        });
        id
    }

    fn build(
        &mut self,
        v: usize,
        target: &BTreeSet<usize>,
        bags: &[BTreeSet<usize>],
        children: &[Vec<usize>],
        owned_edges: &[Vec<usize>],
    ) -> usize {
        let mut current = None;
        for &child in &children[v] {
            let piece = self.build(child, &bags[v], bags, children, owned_edges);
            current = Some(match current {
                None => piece,
                Some(previous) => self.push(Step::Join(previous, piece)),
            });
        }
        let mut current = current.unwrap_or_else(|| self.push(Step::Leaf));
        let missing: Vec<_> = bags[v]
            .difference(&self.nodes[current].bag)
            .copied()
            .collect();
        for vertex in missing {
            current = self.push(Step::IntroduceVertex(vertex, current));
        }
        assert_eq!(self.nodes[current].bag, bags[v]);
        for &edge in &owned_edges[v] {
            current = self.push(Step::IntroduceEdge(edge, current));
            assert_eq!(
                std::mem::replace(&mut self.edge_owner[edge], current),
                usize::MAX
            );
        }
        for &vertex in bags[v].difference(target) {
            current = self.push(Step::Forget(vertex, current));
        }
        let missing: Vec<_> = target
            .difference(&self.nodes[current].bag)
            .copied()
            .collect();
        for vertex in missing {
            current = self.push(Step::IntroduceVertex(vertex, current));
        }
        assert_eq!(&self.nodes[current].bag, target);
        current
    }

    fn evaluate(&self, capacity: usize) -> Relation<'_> {
        let mut values: Vec<Option<Relation<'_>>> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let relation = match node.step {
                Step::Leaf => match self.anchor {
                    Some(z) => Relation::anchored_leaf(&self.graph, capacity, z),
                    None => Relation::empty(&self.graph, capacity),
                },
                Step::IntroduceVertex(v, child) => {
                    values[child].take().unwrap().introduce_vertex(v)
                }
                Step::IntroduceEdge(edge, child) => {
                    values[child].take().unwrap().introduce_edge(edge)
                }
                Step::Forget(v, child) => values[child].take().unwrap().forget(v),
                Step::Join(left, right) => {
                    let left = values[left].take().unwrap();
                    left.join(values[right].take().unwrap())
                }
            };
            assert_eq!(relation.boundary, node.bag);
            values.push(Some(relation));
        }
        values[self.root].take().unwrap()
    }

    fn validate(&self) {
        assert_eq!(self.nodes[self.root].parent, None);
        assert_eq!(self.nodes[self.root].bag, self.anchor.into_iter().collect());
        assert!(self
            .nodes
            .iter()
            .enumerate()
            .all(|(i, node)| { i == self.root || node.parent.is_some_and(|parent| parent > i) }));
        assert!(self
            .edge_owner
            .iter()
            .all(|&owner| owner < self.nodes.len()));
        let mut footprints: Vec<(BTreeSet<usize>, BTreeSet<usize>)> = Vec::new();
        for (id, node) in self.nodes.iter().enumerate() {
            let (vertices, edges) = match node.step {
                Step::Leaf => (self.anchor.into_iter().collect(), BTreeSet::new()),
                Step::IntroduceVertex(v, child) => {
                    let (mut vertices, edges) = footprints[child].clone();
                    assert!(vertices.insert(v));
                    (vertices, edges)
                }
                Step::IntroduceEdge(edge, child) => {
                    assert_eq!(self.edge_owner[edge], id);
                    let (vertices, mut edges) = footprints[child].clone();
                    assert!(edges.insert(edge));
                    (vertices, edges)
                }
                Step::Forget(_, child) => footprints[child].clone(),
                Step::Join(left, right) => {
                    let (mut vertices, mut edges) = footprints[left].clone();
                    assert_eq!(
                        vertices
                            .intersection(&footprints[right].0)
                            .copied()
                            .collect::<BTreeSet<_>>(),
                        node.bag
                    );
                    assert!(edges.is_disjoint(&footprints[right].1));
                    vertices.extend(&footprints[right].0);
                    edges.extend(&footprints[right].1);
                    (vertices, edges)
                }
            };
            footprints.push((vertices, edges));
        }
        assert_eq!(footprints[self.root].0, (0..self.graph.vertices).collect());
        assert_eq!(
            footprints[self.root].1,
            (0..self.graph.edges.len()).collect()
        );
        // Every bag containing a vertex must connect through bags containing it.
        for vertex in 0..self.graph.vertices {
            let holders: Vec<_> = self
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(i, node)| node.bag.contains(&vertex).then_some(i))
                .collect();
            assert!(!holders.is_empty());
            let highest = *holders
                .iter()
                .max_by_key(|&&i| {
                    let mut depth = 0;
                    let mut at = i;
                    while let Some(parent) = self.nodes[at].parent {
                        depth += 1;
                        at = parent;
                    }
                    std::cmp::Reverse(depth)
                })
                .unwrap();
            for &holder in &holders {
                let mut at = holder;
                while at != highest {
                    assert!(self.nodes[at].bag.contains(&vertex));
                    at = self.nodes[at].parent.expect("disconnected vertex bags");
                }
            }
        }
    }
}

fn filled_adjacency(graph: &Graph, anchor: Option<usize>) -> Vec<BTreeSet<usize>> {
    let mut adjacent = vec![BTreeSet::new(); graph.vertices];
    for &(u, v) in &graph.edges {
        adjacent[u].insert(v);
        adjacent[v].insert(u);
    }
    if let Some(z) = anchor {
        for v in 0..graph.vertices {
            if v != z {
                adjacent[v].insert(z);
                adjacent[z].insert(v);
            }
        }
    }
    adjacent
}

fn elimination_order(graph: &Graph, anchor: Option<usize>) -> Vec<usize> {
    let mut adjacent = filled_adjacency(graph, anchor);
    let mut remaining: BTreeSet<_> = (0..graph.vertices).filter(|&v| Some(v) != anchor).collect();
    let mut order = Vec::with_capacity(remaining.len());
    while let Some(&v) = remaining.iter().min_by_key(|&&v| (adjacent[v].len(), v)) {
        let neighbors = adjacent[v].clone();
        for &u in &neighbors {
            adjacent[u].remove(&v);
            adjacent[u].extend(neighbors.iter().copied().filter(|&w| w != u));
        }
        adjacent[v].clear();
        remaining.remove(&v);
        order.push(v);
    }
    order
}

#[cfg(test)]
mod tests {
    use super::{Graph, Plan};
    use crate::boundary_cut_relation::Row;
    use crate::cut_load_algebra::LoadWord;
    use std::collections::BTreeSet;

    fn permutations(values: &[usize]) -> Vec<Vec<usize>> {
        if values.is_empty() {
            return vec![Vec::new()];
        }
        let mut result = Vec::new();
        for (i, &v) in values.iter().enumerate() {
            let mut rest = values.to_vec();
            rest.remove(i);
            for mut suffix in permutations(&rest) {
                suffix.insert(0, v);
                result.push(suffix);
            }
        }
        result
    }

    // This reference uses only original labelled edges and atom permutations.
    fn raw_rows(graph: &Graph, anchor: Option<usize>, capacity: usize) -> Vec<Row> {
        let vertices: Vec<_> = (0..graph.vertices).collect();
        let mut rows = Vec::new();
        for order in permutations(&vertices) {
            if anchor.is_some_and(|z| order.first() != Some(&z)) {
                continue;
            }
            let mut seen = BTreeSet::new();
            let mut loads = vec![0];
            for &v in &order {
                seen.insert(v);
                loads.push(
                    graph
                        .edges
                        .iter()
                        .filter(|&&(u, w)| seen.contains(&u) != seen.contains(&w))
                        .count(),
                );
            }
            if loads.iter().any(|&load| load > capacity) {
                continue;
            }
            let mut gaps = vec![vec![0]];
            let mut boundary_order = Vec::new();
            for (i, &v) in order.iter().enumerate() {
                if Some(v) == anchor {
                    boundary_order.push(v);
                    gaps.push(vec![loads[i + 1]]);
                } else {
                    gaps.last_mut().unwrap().push(loads[i + 1]);
                }
            }
            rows.push(Row {
                order: boundary_order,
                gaps: gaps
                    .iter()
                    .map(|word| LoadWord::new(word, capacity).unwrap())
                    .collect(),
            });
        }
        rows
    }

    fn check(graph: Graph, anchor: Option<usize>, capacities: impl Iterator<Item = usize>) {
        let plan = Plan::prepare(graph, anchor);
        let repeated = Plan::prepare(Graph::new(plan.graph.vertices, &plan.graph.edges), anchor);
        assert_eq!(plan.nodes, repeated.nodes);
        assert_eq!(plan.edge_owner, repeated.edge_owner);
        for capacity in capacities {
            let actual = plan.evaluate(capacity);
            let expected = raw_rows(&plan.graph, anchor, capacity);
            assert_eq!(
                actual.rows.is_empty(),
                expected.is_empty(),
                "anchor={anchor:?}, K={capacity}"
            );
            for row in &actual.rows {
                assert!(
                    expected.iter().any(|raw| raw.dominates(row)),
                    "invented row: graph={:?}, anchor={anchor:?}, K={capacity}, row={row:?}",
                    plan.graph.edges
                );
            }
            for row in &expected {
                assert!(
                    actual.rows.iter().any(|got| got.dominates(row)),
                    "missing row: graph={:?}, anchor={anchor:?}, K={capacity}, row={row:?}",
                    plan.graph.edges
                );
            }
        }
    }

    #[test]
    fn all_simple_graphs_through_four_vertices_match_labelled_permutations() {
        for n in 0..=4 {
            let pairs: Vec<_> = (0..n)
                .flat_map(|u| (u + 1..n).map(move |v| (u, v)))
                .collect();
            for mask in 0..1usize << pairs.len() {
                let edges: Vec<_> = pairs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &edge)| ((mask >> i) & 1 == 1).then_some(edge))
                    .collect();
                for anchor in std::iter::once(None).chain((0..n).map(Some)) {
                    check(Graph::new(n, &edges), anchor, 0..=edges.len());
                }
            }
        }
    }

    #[test]
    fn parallel_edges_isolates_and_disconnected_components_match_raw_orders() {
        let graph = Graph::new(5, &[(0, 1), (0, 1), (2, 3)]);
        for anchor in [None, Some(0), Some(4)] {
            check(Graph::new(graph.vertices, &graph.edges), anchor, 0..=3);
        }
    }

    #[test]
    fn six_leaf_star_rejects_two_and_accepts_three() {
        let graph = Graph::new(7, &[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]);
        let plan = Plan::prepare(graph, None);
        assert!(plan.evaluate(2).rows.is_empty());
        assert!(!plan.evaluate(3).rows.is_empty());
        for capacity in 2..=3 {
            assert_eq!(
                plan.evaluate(capacity).rows.is_empty(),
                raw_rows(&plan.graph, None, capacity).is_empty()
            );
        }
    }

    #[test]
    fn alternate_elimination_order_preserves_unrestricted_writer_orders() {
        let graph = Graph::new(4, &[(0, 1), (1, 2), (2, 3), (0, 3)]);
        for anchor in [None, Some(0)] {
            let default = Plan::prepare(Graph::new(graph.vertices, &graph.edges), anchor);
            let alternate = Plan::with_order(
                Graph::new(graph.vertices, &graph.edges),
                anchor,
                (0..graph.vertices)
                    .filter(|&v| Some(v) != anchor)
                    .rev()
                    .collect(),
            );
            assert_ne!(default.nodes, alternate.nodes);
            for capacity in 0..=4 {
                let raw = raw_rows(&graph, anchor, capacity);
                for plan in [&default, &alternate] {
                    let rows = &plan.evaluate(capacity).rows;
                    for row in rows {
                        assert!(raw.iter().any(|r| r.dominates(row)));
                    }
                    for row in &raw {
                        assert!(rows.iter().any(|r| r.dominates(row)));
                    }
                }
            }
        }
    }
}
