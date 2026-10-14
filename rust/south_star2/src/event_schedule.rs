//! Assigned writer-event relation over shared topology and incidence orders.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::{AtomId, BondId, VariableId};
use crate::model::{ScheduleCenter, ScheduleEdge, ScheduleOrdinaryCenter};
use crate::tetrahedral::{order_for_value, TetrahedralLigand};

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum EntryKind {
    Root = 0,
    BondBranch = 1,
    BondInline = 2,
    DotBranch = 3,
    DotContinuation = 4,
}

impl EntryKind {
    pub(crate) const fn domain() -> Domain {
        Domain::from_bits(0b1_1111)
    }

    fn from_value(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Root),
            1 => Some(Self::BondBranch),
            2 => Some(Self::BondInline),
            3 => Some(Self::DotBranch),
            4 => Some(Self::DotContinuation),
            _ => None,
        }
    }

    fn is_branch(self) -> bool {
        matches!(self, Self::BondBranch | Self::DotBranch)
    }

    fn is_bonded(self) -> bool {
        matches!(self, Self::BondBranch | Self::BondInline)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EventVisit {
    pub(crate) atom: AtomId,
    pub(crate) rank_bits: Arc<[VariableId]>,
    pub(crate) kind: VariableId,
    pub(crate) end_bits: Arc<[VariableId]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventScheduleFactor {
    pub(crate) visits: Arc<[EventVisit]>,
    pub(crate) edges: Arc<[ScheduleEdge]>,
    pub(crate) centers: Arc<[ScheduleCenter]>,
    pub(crate) ordinary_centers: Arc<[ScheduleOrdinaryCenter]>,
    pub(crate) capacity_bits: Arc<[VariableId]>,
    pub(crate) forest_roots: Arc<[VariableId]>,
    pub(crate) segments: bool,
    variables: Box<[VariableId]>,
}

fn bit_value(bits: &[VariableId], value: &mut impl FnMut(VariableId) -> u8) -> usize {
    bits.iter().enumerate().fold(0, |number, (index, bit)| {
        number | (usize::from(value(*bit)) << index)
    })
}

impl EventScheduleFactor {
    pub(crate) fn new(
        visits: Arc<[EventVisit]>,
        edges: Arc<[ScheduleEdge]>,
        centers: Arc<[ScheduleCenter]>,
        ordinary_centers: Arc<[ScheduleOrdinaryCenter]>,
        capacity_bits: Arc<[VariableId]>,
        forest_roots: Arc<[VariableId]>,
        segments: bool,
    ) -> Self {
        assert_eq!(visits.iter().map(|visit| visit.atom).collect::<BTreeSet<_>>().len(), visits.len());
        assert!(visits.iter().enumerate().all(|(index, visit)| visit.atom.index() == index));
        assert!(forest_roots.is_empty() || forest_roots.len() == visits.len());
        let variables = visits.iter().flat_map(|visit| visit.rank_bits.iter().copied()
            .chain([visit.kind]).chain(visit.end_bits.iter().copied()))
            .chain(edges.iter().map(|edge| edge.decision_variable))
            .chain(centers.iter().flat_map(|center|
                [center.order_variable, center.role_pattern_variable]))
            .chain(ordinary_centers.iter().flat_map(|center| center.pairs.iter().map(|pair| pair.variable)))
            .chain(capacity_bits.iter().copied())
            .chain(forest_roots.iter().copied())
            .collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>().into_boxed_slice();
        Self { visits, edges, centers, ordinary_centers, capacity_bits, forest_roots,
            segments, variables }
    }

    pub(crate) fn variables(&self) -> &[VariableId] { &self.variables }

    pub(crate) fn permits(&self, mut value: impl FnMut(VariableId) -> u8) -> bool {
        let count = self.visits.len();
        if count == 0 { return true; }
        let mut by_rank = vec![None; count];
        let mut kinds = vec![EntryKind::Root; count];
        let mut ends = vec![0; count];
        for visit in self.visits.iter() {
            let rank = bit_value(&visit.rank_bits, &mut value);
            let Some(kind) = EntryKind::from_value(value(visit.kind)) else { return false; };
            if visit.end_bits.len() != count { return false; }
            let mut end = None;
            let mut saw_zero = false;
            for (rank, bit) in visit.end_bits.iter().copied().enumerate() {
                match value(bit) {
                    1 if !saw_zero => end = Some(rank),
                    0 => saw_zero = true,
                    _ => return false,
                }
            }
            if rank >= count || by_rank[rank].replace(visit.atom).is_some()
                || (kind.is_branch() && end.is_none_or(|end| end < rank))
                || (!kind.is_branch() && end.is_some()) {
                return false;
            }
            kinds[visit.atom.index()] = kind;
            ends[visit.atom.index()] = end.unwrap_or(0);
        }
        let by_rank = by_rank.into_iter().collect::<Option<Vec<_>>>();
        let Some(by_rank) = by_rank else { return false; };
        if kinds[by_rank[0].index()] != EntryKind::Root { return false; }

        let capacity = bit_value(&self.capacity_bits, &mut value);
        let mut source = None::<AtomId>;
        let mut returns = Vec::<(AtomId, usize)>::new();
        let mut incoming = vec![None::<BondId>; count];
        let mut children = vec![Vec::<(usize, BondId, EntryKind)>::new(); count];
        let mut visited = BTreeSet::<AtomId>::new();
        for (rank, atom) in by_rank.iter().copied().enumerate() {
            while returns.last().is_some_and(|(_, end)| *end < rank) {
                source = Some(returns.pop().unwrap().0);
            }
            let kind = kinds[atom.index()];
            if kind == EntryKind::Root {
                if rank != 0 && (self.segments || !returns.is_empty()) { return false; }
                if rank != 0 && self.edges.iter().any(|edge|
                    edge.role_partition.ring_values().contains(value(edge.decision_variable))
                        && (visited.contains(&edge.a) != visited.contains(&edge.b))) {
                    return false;
                }
                source = Some(atom);
                visited.insert(atom);
                continue;
            }
            if !self.segments && matches!(kind, EntryKind::DotBranch | EntryKind::DotContinuation) {
                return false;
            }
            let Some(parent) = source else { return false; };
            if kind.is_bonded() {
                let Some(edge) = self.edges.iter().find(|edge|
                    (edge.a == parent && edge.b == atom) || (edge.b == parent && edge.a == atom))
                else { return false; };
                if incoming[atom.index()].replace(edge.bond).is_some() { return false; }
                children[parent.index()].push((rank, edge.bond, kind));
            }
            if kind.is_branch() {
                let end = ends[atom.index()];
                if returns.last().is_some_and(|(_, parent_end)| end > *parent_end) { return false; }
                returns.push((parent, end));
            }
            source = Some(atom);
            visited.insert(atom);
        }
        while returns.last().is_some_and(|(_, end)| *end < count) {
            returns.pop();
        }
        if !returns.is_empty() { return false; }

        for edge in self.edges.iter() {
            let traversed = incoming[edge.a.index()] == Some(edge.bond)
                || incoming[edge.b.index()] == Some(edge.bond);
            if edge.role_partition.traversal_values().contains(value(edge.decision_variable)) != traversed {
                return false;
            }
        }
        for (atom, root) in self.forest_roots.iter().copied().enumerate() {
            if (incoming[atom].is_none()) != (value(root) == 1) { return false; }
        }

        let mut open = BTreeSet::<BondId>::new();
        for atom in by_rank {
            let incidence = self.edges.iter().filter(|edge| edge.a == atom || edge.b == atom)
                .map(|edge| edge.bond).collect::<Vec<_>>();
            let mut positions = BTreeMap::<BondId, usize>::new();
            if let Some(center) = self.centers.iter().find(|center| center.atom == atom) {
                let pattern = center.bond_pattern_bits.iter().fold(0_u8, |pattern, (bond, bit)| {
                    let edge = self.edges.iter().find(|edge| edge.bond == *bond)
                        .expect("prepared center bond must have a role");
                    if edge.role_partition.ring_values().contains(value(edge.decision_variable)) {
                        pattern | (1 << bit)
                    } else { pattern }
                });
                let rows = match incoming[atom.index()] {
                    Some(entry) => center.entry_layout_rows.iter()
                        .find_map(|(bond, rows)| (*bond == entry).then_some(rows)),
                    None => Some(&center.root_layout_rows),
                };
                let Some(rows) = rows else { return false; };
                if value(center.role_pattern_variable) != pattern
                    || !rows[pattern as usize].contains(value(center.order_variable)) {
                    return false;
                }
                let Some(order) = order_for_value(&center.reference_order, value(center.order_variable)) else {
                    return false;
                };
                for (position, ligand) in order.iter().enumerate() {
                    if let TetrahedralLigand::Bond(bond) = ligand { positions.insert(*bond, position); }
                }
            } else {
                let Some(center) = self.ordinary_centers.iter().find(|center| center.atom == atom) else {
                    return false;
                };
                for bond in &incidence { positions.insert(*bond, 0); }
                for pair in center.pairs.iter() {
                    let later = if value(pair.variable) == 0 { pair.second } else { pair.first };
                    *positions.get_mut(&later).expect("prepared pair must name one incidence") += 1;
                }
                let mut ranks = positions.values().copied().collect::<Vec<_>>();
                ranks.sort_unstable();
                if !ranks.iter().copied().eq(0..ranks.len()) { return false; }
            }
            if positions.len() != incidence.len() { return false; }
            if let Some(entry) = incoming[atom.index()] {
                if incidence.iter().any(|bond| positions[bond] < positions[&entry]) { return false; }
            }
            let mut ring_events = incidence.iter().copied().filter(|bond|
                self.edges.iter().find(|edge| edge.bond == *bond).is_some_and(|edge|
                    edge.role_partition.ring_values().contains(value(edge.decision_variable))))
                .collect::<Vec<_>>();
            ring_events.sort_by_key(|bond| positions[bond]);
            if ring_events.iter().any(|ring| children[atom.index()].iter().any(|(_, child, _)|
                positions[ring] > positions[child])) { return false; }
            let mut child_positions = children[atom.index()].iter().map(|(rank, bond, kind)|
                (*rank, positions[bond], *kind)).collect::<Vec<_>>();
            child_positions.sort_by_key(|(rank, _, _)| *rank);
            if child_positions.windows(2).any(|pair| pair[0].1 > pair[1].1)
                || child_positions.iter().enumerate().any(|(index, (_, _, kind))|
                    *kind == EntryKind::BondInline && index + 1 != child_positions.len()) {
                return false;
            }
            for ring in ring_events {
                if !open.remove(&ring) {
                    open.insert(ring);
                    if open.len() > capacity { return false; }
                }
            }
        }
        open.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{BondRole, ConstraintModelBuilder, EdgeRolePartition, FactorDefinition,
        OrdinaryOrderPair};
    use crate::native::NativeSolverState;
    use crate::native_oracle::ExhaustiveSolverState;
    use crate::solver::{Consistency, ConstraintSolver};

    #[derive(Clone)]
    enum SyntaxChild {
        BondBranch(usize, Vec<SyntaxChild>),
        BondInline(usize, Vec<SyntaxChild>),
        DotBranch(usize, Vec<SyntaxChild>),
        DotContinuation(usize, Vec<SyntaxChild>),
    }

    fn syntax_visits(children: &[SyntaxChild]) -> Vec<(usize, u8, Option<usize>)> {
        fn append(child: &SyntaxChild, visits: &mut Vec<(usize, u8, Option<usize>)>) {
            let (atom, kind, descendants) = match child {
                SyntaxChild::BondBranch(atom, descendants) => (*atom, 1, descendants),
                SyntaxChild::BondInline(atom, descendants) => (*atom, 2, descendants),
                SyntaxChild::DotBranch(atom, descendants) => (*atom, 3, descendants),
                SyntaxChild::DotContinuation(atom, descendants) => (*atom, 4, descendants),
            };
            let rank = visits.len();
            visits.push((atom, kind, None));
            for descendant in descendants { append(descendant, visits); }
            if kind == 1 || kind == 3 { visits[rank].2 = Some(visits.len() - 1); }
        }
        let mut visits = vec![(0, 0, None)];
        for child in children { append(child, &mut visits); }
        visits
    }

    #[test]
    fn three_atom_dot_event_relation_equals_generated_syntax_trees() {
        let mut builder = ConstraintModelBuilder::new();
        let visits: Arc<[EventVisit]> = (0..3).map(|index| EventVisit {
            atom: AtomId::new(index),
            rank_bits: (0..2).map(|_| builder.add_variable(Domain::from_bits(3)).unwrap())
                .collect::<Vec<_>>().into(),
            kind: builder.add_variable(EntryKind::domain()).unwrap(),
            end_bits: (0..3).map(|_| builder.add_variable(Domain::from_bits(3)).unwrap())
                .collect::<Vec<_>>().into(),
        }).collect::<Vec<_>>().into();
        let roots: Arc<[VariableId]> = (0..3).map(|_|
            builder.add_variable(Domain::singleton(1).unwrap()).unwrap())
            .collect::<Vec<_>>().into();
        let capacity = builder.add_variable(Domain::singleton(0).unwrap()).unwrap();
        let ordinary: Arc<[ScheduleOrdinaryCenter]> = (0..3).map(|index|
            ScheduleOrdinaryCenter { atom: AtomId::new(index), pairs: Arc::from([]) })
            .collect::<Vec<_>>().into();
        let factor = EventScheduleFactor::new(visits.clone(), Arc::from([]), Arc::from([]),
            ordinary, vec![capacity].into(), roots.clone(), true);

        let dot = |kind: u8, atom: usize, descendants: Vec<SyntaxChild>| match kind {
            3 => SyntaxChild::DotBranch(atom, descendants),
            4 => SyntaxChild::DotContinuation(atom, descendants),
            _ => unreachable!(),
        };
        let mut syntax_encodings = BTreeSet::new();
        for (first, second) in [(1, 2), (2, 1)] {
            for outer in [3, 4] {
                for inner in [3, 4] {
                    syntax_encodings.insert(syntax_visits(&[
                        dot(outer, first, vec![dot(inner, second, vec![])]),
                    ]));
                }
            }
            for final_kind in [3, 4] {
                syntax_encodings.insert(syntax_visits(&[
                    dot(3, first, vec![]), dot(final_kind, second, vec![]),
                ]));
            }
        }

        let mut accepted = BTreeSet::new();
        for (first, second) in [(1, 2), (2, 1)] {
            for first_kind in 0..5 {
                for second_kind in 0..5 {
                    for first_end in [None, Some(0), Some(1), Some(2)] {
                        for second_end in [None, Some(0), Some(1), Some(2)] {
                            let encoding = vec![(0, 0, None), (first, first_kind, first_end),
                                (second, second_kind, second_end)];
                            let mut assignment = BTreeMap::new();
                            for (rank, (atom, kind, end)) in encoding.iter().copied().enumerate() {
                                let visit = &visits[atom];
                                for (bit, variable) in visit.rank_bits.iter().enumerate() {
                                    assignment.insert(*variable, ((rank >> bit) & 1) as u8);
                                }
                                assignment.insert(visit.kind, kind);
                                for (bit, variable) in visit.end_bits.iter().enumerate() {
                                    assignment.insert(*variable, u8::from(end.is_some_and(|end| bit <= end)));
                                }
                            }
                            assignment.insert(capacity, 0);
                            for root in roots.iter() { assignment.insert(*root, 1); }
                            if factor.permits(|variable| assignment[&variable]) {
                                accepted.insert(encoding);
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(accepted, syntax_encodings,
            "the event relation must recognize exactly the independently generated dot syntax trees");
    }

    #[test]
    fn syntax_tree_derives_nested_dot_and_tail_assignments() {
        let mut builder = ConstraintModelBuilder::new();
        let visits: Arc<[EventVisit]> = (0..4).map(|index| EventVisit {
            atom: AtomId::new(index),
            rank_bits: (0..2).map(|_| builder.add_variable(Domain::from_bits(3)).unwrap())
                .collect::<Vec<_>>().into(),
            kind: builder.add_variable(EntryKind::domain()).unwrap(),
            end_bits: (0..4).map(|_| builder.add_variable(Domain::from_bits(3)).unwrap())
                .collect::<Vec<_>>().into(),
        }).collect::<Vec<_>>().into();
        let roles = (0..2).map(|_| builder.add_variable(BondRole::role_domain()).unwrap())
            .collect::<Vec<_>>();
        let pair = builder.add_variable(Domain::from_bits(3)).unwrap();
        let roots = (0..4).map(|_| builder.add_variable(Domain::from_bits(3)).unwrap())
            .collect::<Vec<_>>();
        let edges: Arc<[ScheduleEdge]> = (0..2).map(|index| ScheduleEdge {
            bond: BondId::new(index), a: AtomId::new(0), b: AtomId::new(index + 1),
            decision_variable: roles[index as usize], role_partition: EdgeRolePartition::bond_role(),
        }).collect::<Vec<_>>().into();
        let ordinary: Arc<[ScheduleOrdinaryCenter]> = (0..4).map(|index| ScheduleOrdinaryCenter {
            atom: AtomId::new(index),
            pairs: if index == 0 { vec![OrdinaryOrderPair {
                first: BondId::new(0), second: BondId::new(1), variable: pair,
            }].into() } else { Arc::from([]) },
        }).collect::<Vec<_>>().into();
        let capacity = builder.add_variable(Domain::singleton(0).unwrap()).unwrap();
        let factor_id = builder.add_latent_event_schedule(visits.clone(), edges,
            Arc::from([]), ordinary, vec![capacity].into(), roots.clone().into(), true).unwrap();
        let model = Arc::new(builder.build());
        let FactorDefinition::EventSchedule(factor) = model.factor(factor_id).unwrap()
            else { panic!("event factor expected") };
        let assignments = |syntax: &[SyntaxChild]| {
            let mut values = BTreeMap::new();
            for (rank, (atom, kind, end)) in syntax_visits(syntax).into_iter().enumerate() {
                let visit = &visits[atom];
                for (bit, variable) in visit.rank_bits.iter().enumerate() {
                    values.insert(*variable, ((rank >> bit) & 1) as u8);
                }
                values.insert(visit.kind, kind);
                for (bit, variable) in visit.end_bits.iter().enumerate() {
                    values.insert(*variable, u8::from(end.is_some_and(|end| bit <= end)));
                }
            }
            for role in &roles { values.insert(*role, BondRole::Traversal as u8); }
            values.insert(pair, 0);
            values.insert(capacity, 0);
            for (index, root) in roots.iter().enumerate() {
                values.insert(*root, u8::from(index == 0 || index == 3));
            }
            values
        };
        let nested = [SyntaxChild::BondBranch(1, vec![SyntaxChild::DotBranch(3, vec![])]),
            SyntaxChild::BondInline(2, vec![])];
        let tail = [SyntaxChild::BondBranch(1, vec![]),
            SyntaxChild::BondInline(2, vec![SyntaxChild::DotContinuation(3, vec![])])];
        for syntax in [&nested[..], &tail[..]] {
            let values = assignments(syntax);
            assert!(factor.permits(|variable| values[&variable]),
                "an independently constructed syntax tree must admit its derived event facts");
            let restrictions = values.iter().map(|(variable, value)|
                (*variable, Domain::singleton(*value).unwrap())).collect::<Vec<_>>();
            assert!(matches!(<NativeSolverState as ConstraintSolver>::initial(model.clone())
                .unwrap().unwrap_consistent().transitioned(&restrictions, &[factor_id]).unwrap(),
                Consistency::Consistent(_)));
            let without_roles = restrictions.iter().copied()
                .filter(|(variable, _)| !roles.contains(variable)
                    && *variable != visits[3].kind
                    && !visits[3].end_bits.contains(variable)).collect::<Vec<_>>();
            let projected = <NativeSolverState as ConstraintSolver>::initial(model.clone())
                .unwrap().unwrap_consistent().transitioned(&without_roles, &[factor_id])
                .unwrap().unwrap_consistent();
            for role in &roles {
                assert_eq!(projected.domain_bound(*role), Some(BondRole::Traversal.singleton_domain()),
                    "the assigned syntax must jointly force both chemical traversal roles");
            }
            let feasible = <NativeSolverState as ConstraintSolver>::initial(model.clone())
                .unwrap().unwrap_consistent().transitioned_feasible(&without_roles, &[factor_id])
                .unwrap().unwrap_consistent();
            let exact_noop = feasible.transitioned(&[], &[]).unwrap().unwrap_consistent();
            let exact_redundant = feasible.transitioned(&[without_roles[0]], &[])
                .unwrap().unwrap_consistent();
            let oracle = <ExhaustiveSolverState as ConstraintSolver>::initial(model.clone())
                .unwrap().unwrap_consistent().transitioned(&without_roles, &[factor_id])
                .unwrap().unwrap_consistent();
            for variable in factor.variables() {
                assert_eq!(exact_noop.domain_bound(*variable), oracle.domain_bound(*variable));
                assert_eq!(exact_redundant.domain_bound(*variable), oracle.domain_bound(*variable));
            }
        }
        let mut malformed = assignments(&nested);
        let nested_dot = &visits[3];
        malformed.insert(nested_dot.end_bits[3], 1);
        assert!(!factor.permits(|variable| malformed[&variable]),
            "a dot branch cannot close after its containing bonded branch");
        let invalid = malformed.iter().map(|(variable, value)|
            (*variable, Domain::singleton(*value).unwrap())).collect::<Vec<_>>();
        assert!(matches!(<NativeSolverState as ConstraintSolver>::initial(model)
            .unwrap().unwrap_consistent().transitioned(&invalid, &[factor_id]).unwrap(),
            Consistency::Contradiction));
    }

    #[test]
    fn assigned_event_path_matches_native_and_exhaustive() {
        let mut builder = ConstraintModelBuilder::new();
        let visits: Arc<[EventVisit]> = (0..3).map(|index| EventVisit {
            atom: AtomId::new(index),
            rank_bits: (0..2).map(|_| builder.add_variable(Domain::from_bits(3)).unwrap())
                .collect::<Vec<_>>().into(),
            kind: builder.add_variable(EntryKind::domain()).unwrap(),
            end_bits: (0..3).map(|_| builder.add_variable(Domain::from_bits(3)).unwrap())
                .collect::<Vec<_>>().into(),
        }).collect::<Vec<_>>().into();
        let roles = (0..2).map(|_| builder.add_variable(BondRole::role_domain()).unwrap())
            .collect::<Vec<_>>();
        let pair = builder.add_variable(Domain::from_bits(3)).unwrap();
        let capacity = builder.add_variable(Domain::singleton(0).unwrap()).unwrap();
        let edges: Arc<[ScheduleEdge]> = (0..2).map(|index| ScheduleEdge {
            bond: BondId::new(index), a: AtomId::new(index), b: AtomId::new(index + 1),
            decision_variable: roles[index as usize], role_partition: EdgeRolePartition::bond_role(),
        }).collect::<Vec<_>>().into();
        let ordinary: Arc<[ScheduleOrdinaryCenter]> = vec![
            ScheduleOrdinaryCenter { atom: AtomId::new(0), pairs: Arc::from([]) },
            ScheduleOrdinaryCenter { atom: AtomId::new(1), pairs: vec![OrdinaryOrderPair {
                first: BondId::new(0), second: BondId::new(1), variable: pair,
            }].into() },
            ScheduleOrdinaryCenter { atom: AtomId::new(2), pairs: Arc::from([]) },
        ].into();
        let factor = builder.add_latent_event_schedule(visits.clone(), edges,
            Arc::from([]), ordinary, vec![capacity].into(), Arc::from([]), true).unwrap();
        let model = Arc::new(builder.build());
        let assignments = |ranks: [usize; 3]| {
            let mut restrictions = Vec::new();
            for (index, visit) in visits.iter().enumerate() {
                for (bit, variable) in visit.rank_bits.iter().enumerate() {
                    restrictions.push((*variable, Domain::singleton(((ranks[index] >> bit) & 1) as u8).unwrap()));
                }
                restrictions.push((visit.kind, Domain::singleton(if index == 0 { 0 } else { 2 }).unwrap()));
                for variable in visit.end_bits.iter() {
                    restrictions.push((*variable, Domain::singleton(0).unwrap()));
                }
            }
            restrictions.extend(roles.iter().map(|variable|
                (*variable, BondRole::Traversal.singleton_domain())));
            restrictions.push((pair, Domain::singleton(0).unwrap()));
            restrictions
        };
        for (ranks, expected) in [([0, 1, 2], true), ([0, 2, 1], false)] {
            let restrictions = assignments(ranks);
            let native = <NativeSolverState as ConstraintSolver>::initial(model.clone())
                .unwrap().unwrap_consistent().transitioned(&restrictions, &[factor]).unwrap();
            let oracle = <ExhaustiveSolverState as ConstraintSolver>::initial(model.clone())
                .unwrap().unwrap_consistent().transitioned(&restrictions, &[factor]).unwrap();
            assert_eq!(matches!(native, Consistency::Consistent(_)), expected);
            assert_eq!(matches!(oracle, Consistency::Consistent(_)), expected);
        }
    }

    #[test]
    fn forced_ring_cut_rejects_before_future_ranks_are_assigned() {
        let mut builder = ConstraintModelBuilder::new();
        let visits: Arc<[EventVisit]> = (0..3).map(|index| EventVisit {
            atom: AtomId::new(index),
            rank_bits: (0..2).map(|_| builder.add_variable(Domain::from_bits(3)).unwrap())
                .collect::<Vec<_>>().into(),
            kind: builder.add_variable(EntryKind::domain()).unwrap(),
            end_bits: (0..3).map(|_| builder.add_variable(Domain::from_bits(3)).unwrap())
                .collect::<Vec<_>>().into(),
        }).collect::<Vec<_>>().into();
        let roles = (0..2).map(|_| builder.add_variable(BondRole::role_domain()).unwrap())
            .collect::<Vec<_>>();
        let edges: Arc<[ScheduleEdge]> = (0..2).map(|index| ScheduleEdge {
            bond: BondId::new(index), a: AtomId::new(0), b: AtomId::new(index + 1),
            decision_variable: roles[index as usize], role_partition: EdgeRolePartition::bond_role(),
        }).collect::<Vec<_>>().into();
        let ordinary: Arc<[ScheduleOrdinaryCenter]> = (0..3).map(|index| {
            let pairs: Arc<[OrdinaryOrderPair]> = if index == 0 {
                vec![OrdinaryOrderPair { first: BondId::new(0), second: BondId::new(1),
                    variable: builder.add_variable(Domain::from_bits(3)).unwrap() }].into()
            } else { Arc::from([]) };
            ScheduleOrdinaryCenter { atom: AtomId::new(index), pairs }
        }).collect::<Vec<_>>().into();
        let capacity = builder.add_variable(Domain::singleton(0).unwrap()).unwrap();
        let factor = builder.add_latent_event_schedule(visits.clone(), edges,
            Arc::from([]), ordinary, vec![capacity].into(), Arc::from([]), true).unwrap();
        let solver = <NativeSolverState as ConstraintSolver>::initial(Arc::new(builder.build()))
            .unwrap().unwrap_consistent();
        let restrictions = visits[0].rank_bits.iter().copied().map(|bit|
            (bit, Domain::singleton(0).unwrap())).chain(roles.iter().copied().map(|role|
            (role, BondRole::Ring.singleton_domain()))).collect::<Vec<_>>();
        assert!(matches!(solver.transitioned(&restrictions, &[factor]).unwrap(),
            Consistency::Contradiction));
    }
}
