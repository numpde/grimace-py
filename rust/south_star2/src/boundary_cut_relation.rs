//! Test-only boundary relation for unrestricted orders of a supplied physical graph.
//! Rows retain whole correlated gap tuples, quotiented by mutual load domination.

use std::collections::BTreeSet;

use crate::cut_load_algebra::{merge, LoadWord};

pub(super) struct Graph {
    pub(super) vertices: usize,
    // The position is the physical edge ID, so parallel edges remain distinct.
    pub(super) edges: Vec<(usize, usize)>,
}

impl Graph {
    pub(super) fn new(vertices: usize, edges: &[(usize, usize)]) -> Self {
        assert!(edges
            .iter()
            .all(|&(u, v)| u < vertices && v < vertices && u != v));
        Self {
            vertices,
            edges: edges.to_vec(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Row {
    pub(super) order: Vec<usize>,
    pub(super) gaps: Vec<LoadWord>,
}

impl Row {
    pub(super) fn dominates(&self, other: &Self) -> bool {
        self.order == other.order
            && self
                .gaps
                .iter()
                .zip(&other.gaps)
                .all(|(a, b)| a.dominates(b))
    }
}

fn retain_row(rows: &mut Vec<Row>, candidate: Row) {
    if rows.iter().any(|row| row.dominates(&candidate)) {
        return;
    }
    rows.retain(|row| !candidate.dominates(row));
    rows.push(candidate);
}

// A cached plan cell retains only this payload; piece ownership is checked once
// by the immutable plan and is never replicated across cached cells.
pub(super) type Profile = Vec<Row>;

pub(super) fn leaf_rows(anchor: Option<usize>) -> Profile {
    match anchor {
        Some(z) => vec![Row {
            order: vec![z],
            gaps: vec![LoadWord(vec![0]), LoadWord(vec![0])],
        }],
        None => vec![Row {
            order: vec![],
            gaps: vec![LoadWord(vec![0])],
        }],
    }
}

pub(super) fn introduce_vertex_rows(
    input: &[Row], vertex: usize, capacity: usize, anchor: Option<usize>,
) -> Profile {
    let mut rows = Vec::new();
    for row in input {
        for position in 0..=row.order.len() {
            if anchor.is_some() && position == 0 { continue; }
            let gap = &row.gaps[position].0;
            for cut in 0..gap.len() {
                let mut order = row.order.clone();
                order.insert(position, vertex);
                let mut gaps = row.gaps.clone();
                gaps.splice(position..=position, [
                    LoadWord::new(&gap[..=cut], capacity).unwrap(),
                    LoadWord::new(&gap[cut..], capacity).unwrap(),
                ]);
                retain_row(&mut rows, Row { order, gaps });
            }
        }
    }
    rows
}

pub(super) fn introduce_edge_rows(
    input: &[Row], contribution: Option<(usize, usize)>, capacity: usize,
) -> Profile {
    let Some((u, v)) = contribution else { return input.to_vec(); };
    let mut rows = Vec::new();
    for original in input {
        let mut row = original.clone();
        let a = row.order.iter().position(|&x| x == u).unwrap();
        let b = row.order.iter().position(|&x| x == v).unwrap();
        let (lo, hi) = (a.min(b), a.max(b));
        let mut possible = true;
        for gap in &mut row.gaps[lo + 1..=hi] {
            let raised: Vec<_> = gap.0.iter().map(|&load| load + 1).collect();
            match LoadWord::new(&raised, capacity) {
                Some(next) => *gap = next,
                None => { possible = false; break; }
            }
        }
        if possible { retain_row(&mut rows, row); }
    }
    rows
}

pub(super) fn forget_rows(input: &[Row], vertex: usize, capacity: usize) -> Profile {
    let mut rows = Vec::new();
    for original in input {
        let mut row = original.clone();
        let position = row.order.iter().position(|&x| x == vertex).unwrap();
        row.order.remove(position);
        // The cut before the visit can differ from the cut after it.
        let mut joined = row.gaps[position].0.clone();
        joined.extend(&row.gaps[position + 1].0);
        row.gaps.splice(position..=position + 1,
            [LoadWord::new(&joined, capacity).unwrap()]);
        retain_row(&mut rows, row);
    }
    rows
}

pub(super) fn join_rows(left: &[Row], right: &[Row], capacity: usize) -> Profile {
    let mut rows = Vec::new();
    for left in left {
        for right in right {
            if left.order != right.order { continue; }
            let mut tuples = vec![Vec::new()];
            for (a, b) in left.gaps.iter().zip(&right.gaps) {
                let choices = merge(a, b, capacity);
                tuples = tuples.into_iter().flat_map(|prefix| {
                    choices.iter().map(move |word| {
                        let mut tuple = prefix.clone();
                        tuple.push(word.clone());
                        tuple
                    })
                }).collect();
            }
            for gaps in tuples {
                retain_row(&mut rows, Row { order: left.order.clone(), gaps });
            }
        }
    }
    rows
}

pub(super) struct Relation<'g> {
    graph: &'g Graph,
    capacity: usize,
    anchor: Option<usize>,
    vertices: BTreeSet<usize>,
    edges: BTreeSet<usize>,
    pub(super) boundary: BTreeSet<usize>,
    pub(super) rows: Vec<Row>,
}

impl<'g> Relation<'g> {
    pub(super) fn empty(graph: &'g Graph, capacity: usize) -> Self {
        Self {
            graph,
            capacity,
            anchor: None,
            vertices: BTreeSet::new(),
            edges: BTreeSet::new(),
            boundary: BTreeSet::new(),
            rows: leaf_rows(None),
        }
    }

    pub(super) fn anchored_leaf(graph: &'g Graph, capacity: usize, anchor: usize) -> Self {
        assert!(anchor < graph.vertices);
        Self {
            graph,
            capacity,
            anchor: Some(anchor),
            vertices: BTreeSet::from([anchor]),
            edges: BTreeSet::new(),
            boundary: BTreeSet::from([anchor]),
            rows: leaf_rows(Some(anchor)),
        }
    }

    pub(super) fn introduce_vertex(mut self, vertex: usize) -> Self {
        assert!(vertex < self.graph.vertices && !self.vertices.contains(&vertex));
        self.vertices.insert(vertex);
        self.boundary.insert(vertex);
        self.rows = introduce_vertex_rows(&self.rows, vertex, self.capacity, self.anchor);
        self
    }

    pub(super) fn introduce_edge(mut self, edge: usize) -> Self {
        assert!(edge < self.graph.edges.len() && !self.edges.contains(&edge));
        let (u, v) = self.graph.edges[edge];
        assert!(self.boundary.contains(&u) && self.boundary.contains(&v));
        self.edges.insert(edge);
        self.rows = introduce_edge_rows(&self.rows, Some((u, v)), self.capacity);
        self
    }

    pub(super) fn forget(mut self, vertex: usize) -> Self {
        assert!(self.boundary.contains(&vertex) && self.anchor != Some(vertex));
        self.boundary.remove(&vertex);
        self.rows = forget_rows(&self.rows, vertex, self.capacity);
        self
    }

    pub(super) fn join(mut self, other: Self) -> Self {
        assert!(std::ptr::eq(self.graph, other.graph));
        assert_eq!(self.capacity, other.capacity);
        assert_eq!(self.anchor, other.anchor);
        assert_eq!(self.boundary, other.boundary);
        assert!(self.edges.is_disjoint(&other.edges));
        assert_eq!(
            self.vertices
                .intersection(&other.vertices)
                .copied()
                .collect::<BTreeSet<_>>(),
            self.boundary
        );
        self.vertices.extend(other.vertices);
        self.edges.extend(other.edges);
        self.rows = join_rows(&self.rows, &other.rows, self.capacity);
        self
    }

}

#[cfg(test)]
mod tests {
    use super::{retain_row, Graph, Relation, Row};
    use crate::cut_load_algebra::LoadWord;

    fn permutations(values: &[usize]) -> Vec<Vec<usize>> {
        if values.is_empty() {
            return vec![vec![]];
        }
        let mut result = Vec::new();
        for (i, &value) in values.iter().enumerate() {
            let mut rest = values.to_vec();
            rest.remove(i);
            for mut suffix in permutations(&rest) {
                suffix.insert(0, value);
                result.push(suffix);
            }
        }
        result
    }

    // Direct labelled orders and physical edge cuts, independent of the relation operators.
    fn raw_rows(relation: &Relation<'_>) -> Vec<Row> {
        let owned: Vec<_> = relation.vertices.iter().copied().collect();
        let mut rows = Vec::new();
        for order in permutations(&owned) {
            if relation.anchor.is_some_and(|z| order.first() != Some(&z)) {
                continue;
            }
            let mut seen = std::collections::BTreeSet::new();
            let mut gaps = vec![vec![0usize]];
            let mut boundary_order = Vec::new();
            for &vertex in &order {
                seen.insert(vertex);
                let load = relation
                    .edges
                    .iter()
                    .filter(|&&id| {
                        let (u, v) = relation.graph.edges[id];
                        seen.contains(&u) != seen.contains(&v)
                    })
                    .count();
                if relation.boundary.contains(&vertex) {
                    boundary_order.push(vertex);
                    gaps.push(vec![load]);
                } else {
                    gaps.last_mut().unwrap().push(load);
                }
            }
            let Some(gaps) = gaps
                .into_iter()
                .map(|word| LoadWord::new(&word, relation.capacity))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            rows.push(Row {
                order: boundary_order,
                gaps,
            });
        }
        rows
    }

    fn assert_covered(actual: &[Row], expected: &[Row]) {
        for row in actual {
            assert!(
                expected.iter().any(|w| w.dominates(row)),
                "unsound row {row:?}"
            );
        }
        for row in expected {
            assert!(
                actual.iter().any(|w| w.dominates(row)),
                "missing row {row:?}"
            );
        }
    }

    fn checked(relation: Relation<'_>) -> Relation<'_> {
        let expected = raw_rows(&relation);
        assert_covered(&relation.rows, &expected);
        for (i, row) in relation.rows.iter().enumerate() {
            assert_eq!(row.gaps.len(), relation.boundary.len() + 1);
            assert_eq!(
                row.order
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>(),
                relation.boundary
            );
            assert!(relation
                .rows
                .iter()
                .enumerate()
                .all(|(j, other)| i == j || !other.dominates(row)));
        }
        relation
    }

    #[test]
    fn unanchored_empty_introduce_edge_and_forget_preserve_reference_coverage() {
        let graph = Graph::new(3, &[(0, 1)]);
        for k in 0..=2 {
            let r = checked(Relation::empty(&graph, k));
            let r = checked(r.introduce_vertex(0));
            let r = checked(r.introduce_vertex(1));
            let r = checked(r.introduce_edge(0));
            let r = checked(r.forget(0));
            let r = checked(r.introduce_vertex(2));
            let r = checked(r.forget(1));
            checked(r.forget(2));
        }
    }

    #[test]
    fn anchored_join_and_serial_decompositions_cover_same_parallel_edge_graph() {
        // Distinct IDs 0 and 3 are parallel physical bonds; each belongs to one child.
        let graph = Graph::new(4, &[(0, 1), (1, 2), (0, 3), (0, 1)]);
        for k in 0..=4 {
            let leaf = || checked(Relation::anchored_leaf(&graph, k, 0));
            let left = checked(leaf().introduce_vertex(1));
            let left = checked(left.introduce_edge(0));
            let left = checked(left.introduce_vertex(2));
            let left = checked(left.introduce_edge(1));
            let left = checked(left.forget(2));
            let right = checked(leaf().introduce_vertex(1));
            let right = checked(right.introduce_edge(3));
            let right = checked(right.introduce_vertex(3));
            let right = checked(right.introduce_edge(2));
            let right = checked(right.forget(3));
            let joined = checked(left.join(right));
            let joined = checked(joined.forget(1));

            let serial = checked(leaf().introduce_vertex(1));
            let serial = checked(serial.introduce_edge(0));
            let serial = checked(serial.introduce_edge(3));
            let serial = checked(serial.introduce_vertex(2));
            let serial = checked(serial.introduce_edge(1));
            let serial = checked(serial.forget(2));
            let serial = checked(serial.introduce_vertex(3));
            let serial = checked(serial.introduce_edge(2));
            let serial = checked(serial.forget(3));
            let serial = checked(serial.forget(1));
            assert_covered(&joined.rows, &serial.rows);
            assert_covered(&serial.rows, &joined.rows);
        }
    }

    #[test]
    fn forgetting_edge_endpoint_keeps_both_adjacent_cut_values() {
        let graph = Graph::new(2, &[(0, 1)]);
        let r = Relation::anchored_leaf(&graph, 1, 0)
            .introduce_vertex(1)
            .introduce_edge(0)
            .forget(1);
        assert_eq!(r.rows[0].gaps[1].0, vec![1, 0]);
        checked(r);
    }

    #[test]
    fn same_order_requires_whole_correlated_gap_tuples() {
        let word = |n| LoadWord(vec![n]);
        let a = Row {
            order: vec![0],
            gaps: vec![word(0), word(2)],
        };
        let b = Row {
            order: vec![0],
            gaps: vec![word(2), word(0)],
        };
        let invented = Row {
            order: vec![0],
            gaps: vec![word(0), word(0)],
        };
        let mut rows = Vec::new();
        retain_row(&mut rows, a.clone());
        retain_row(&mut rows, b.clone());
        assert_eq!(rows.len(), 2);
        assert!(!rows.iter().any(|row| row.dominates(&invented)));
        retain_row(&mut rows, a);
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn join_keeps_physical_gap_tradeoffs_correlated() {
        // Each three-leaf star has width two by visiting one leaf before its
        // center. Joining them makes a six-leaf star with minimum width three.
        let graph = Graph::new(7, &[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]);
        let piece = |first_edge: usize| {
            let mut r = checked(Relation::empty(&graph, 2).introduce_vertex(0));
            for edge in first_edge..first_edge + 3 {
                r = checked(r.introduce_vertex(edge + 1));
                r = checked(r.introduce_edge(edge));
                r = checked(r.forget(edge + 1));
            }
            assert!(!r.rows.is_empty());
            r
        };
        let joined = checked(piece(0).join(piece(3)));
        assert!(joined.rows.is_empty());
    }

    #[test]
    #[should_panic]
    fn join_rejects_duplicate_physical_edge_ownership() {
        let graph = Graph::new(2, &[(0, 1)]);
        let piece = || {
            Relation::anchored_leaf(&graph, 1, 0)
                .introduce_vertex(1)
                .introduce_edge(0)
        };
        piece().join(piece());
    }

    #[test]
    #[should_panic]
    fn anchored_relation_cannot_forget_anchor() {
        let graph = Graph::new(1, &[]);
        Relation::anchored_leaf(&graph, 0, 0).forget(0);
    }

    #[test]
    #[should_panic]
    fn join_rejects_private_vertex_shared_by_children() {
        let graph = Graph::new(2, &[]);
        let piece = || {
            Relation::anchored_leaf(&graph, 0, 0)
                .introduce_vertex(1)
                .forget(1)
        };
        piece().join(piece());
    }
}
