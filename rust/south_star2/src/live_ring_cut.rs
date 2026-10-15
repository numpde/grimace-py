//! Necessary ring-label pressure across rooted physical traversal subtrees.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::ids::{AtomId, BondId, VariableId};
use crate::model::{OrdinaryOrderPair, ScheduleCenter, ScheduleEdge, ScheduleOrdinaryCenter};
use crate::tetrahedral::{order_for_value, TetrahedralLigand};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LiveRingCutContext {
    pub(crate) visited: Box<[bool]>,
    pub(crate) unwritten: Box<[bool]>,
    pub(crate) held: Box<[(BondId, AtomId)]>,
    pub(crate) capacity: usize,
    pub(crate) actual_tree_root: Option<AtomId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiveRingCutFactor {
    pub(crate) edges: Arc<[ScheduleEdge]>,
    pub(crate) centers: Arc<[ScheduleCenter]>,
    pub(crate) ordinary_centers: Arc<[ScheduleOrdinaryCenter]>,
    pub(crate) roots: Arc<[VariableId]>,
    incidence: Box<[Box<[BondId]>]>,
    variables: Box<[VariableId]>,
}

impl LiveRingCutFactor {
    pub(crate) fn new(edges: Arc<[ScheduleEdge]>, centers: Arc<[ScheduleCenter]>,
        ordinary_centers: Arc<[ScheduleOrdinaryCenter]>, roots: Arc<[VariableId]>) -> Self {
        assert!(edges.iter().enumerate().all(|(index, edge)| edge.bond.index() == index),
            "physical ring-cut bonds must be indexed by their prepared IDs");
        let atom_count = centers.len() + ordinary_centers.len();
        assert!(roots.is_empty() || roots.len() == atom_count);
        let mut incidence = vec![Vec::new(); atom_count];
        for edge in edges.iter() {
            incidence[edge.a.index()].push(edge.bond);
            incidence[edge.b.index()].push(edge.bond);
        }
        let incidence = incidence.into_iter().map(Vec::into_boxed_slice)
            .collect::<Vec<_>>().into_boxed_slice();
        let variables = edges.iter().map(|edge| edge.decision_variable)
            .chain(centers.iter().map(|center| center.order_variable))
            .chain(ordinary_centers.iter().flat_map(|center| center.pairs.iter().map(|pair| pair.variable)))
            .chain(roots.iter().copied())
            .collect::<BTreeSet<_>>().into_iter().collect();
        Self { edges, centers, ordinary_centers, roots, incidence, variables }
    }

    pub(crate) fn variables(&self) -> &[VariableId] { &self.variables }

    /// An invalid physical assignment is left to its existing topology/Event factors.
    pub(crate) fn permits(&self, context: &LiveRingCutContext,
        mut value: impl FnMut(VariableId) -> u8) -> bool {
        let count = context.visited.len();
        assert!(self.roots.is_empty() || self.roots.len() == count);
        assert_eq!(self.incidence.len(), count);
        assert_eq!(context.unwritten.len(), self.edges.len());
        let mut tree = vec![Vec::<(AtomId, BondId)>::new(); count];
        let mut ring = Vec::new();
        for edge in self.edges.iter() {
            if edge.role_partition.traversal_values().contains(value(edge.decision_variable)) {
                tree[edge.a.index()].push((edge.b, edge.bond));
                tree[edge.b.index()].push((edge.a, edge.bond));
            } else {
                ring.push(edge);
            }
        }
        let endpoint = |atom: usize, bond: BondId| {
            let edge = &self.edges[bond.index()];
            assert!(edge.a.index() == atom || edge.b.index() == atom);
            usize::from(edge.a.index() != atom)
        };
        // Each physical bond has exactly two incidence ranks, one per endpoint.
        let mut positions = vec![[usize::MAX; 2]; self.edges.len()];
        for center in self.centers.iter() {
            let Some(order) = order_for_value(&center.reference_order, value(center.order_variable)) else { return true; };
            for (position, ligand) in order.iter().enumerate() {
                if let TetrahedralLigand::Bond(bond) = ligand {
                    positions[bond.index()][endpoint(center.atom.index(), *bond)] = position;
                }
            }
        }
        for center in self.ordinary_centers.iter() {
            let atom = center.atom.index();
            for &bond in &self.incidence[atom] {
                positions[bond.index()][endpoint(atom, bond)] = 0;
            }
            for OrdinaryOrderPair { first, second, variable } in center.pairs.iter() {
                let later = if value(*variable) == 0 { second } else { first };
                positions[later.index()][endpoint(atom, *later)] += 1;
            }
        }
        for (atom, children) in tree.iter_mut().enumerate() {
            children.sort_by_key(|(_, bond)| positions[bond.index()][endpoint(atom, *bond)]);
        }
        let mut order = vec![usize::MAX; count];
        let mut end = vec![usize::MAX; count];
        let mut owner = vec![usize::MAX; count];
        let mut parent = vec![None; count];
        let mut clock = 0;
        fn walk(atom: usize, from: Option<usize>, root: usize,
            tree: &[Vec<(AtomId, BondId)>], order: &mut [usize], end: &mut [usize],
            owner: &mut [usize], parent: &mut [Option<usize>], clock: &mut usize) -> bool {
            if order[atom] != usize::MAX { return false; }
            order[atom] = *clock;
            *clock += 1;
            owner[atom] = root;
            parent[atom] = from;
            for &(next, _) in &tree[atom] {
                if Some(next.index()) != from && !walk(next.index(), Some(atom), root,
                    tree, order, end, owner, parent, clock) { return false; }
            }
            end[atom] = *clock;
            true
        }
        let roots = if self.roots.is_empty() {
            context.actual_tree_root.into_iter().map(AtomId::index).collect::<Vec<_>>()
        } else {
            self.roots.iter().enumerate().filter_map(|(index, root)|
                (value(*root) == 1).then_some(index)).collect::<Vec<_>>()
        };
        if roots.is_empty() { return true; }
        for index in roots {
            if !walk(index, None, index, &tree,
                &mut order, &mut end, &mut owner, &mut parent, &mut clock) { return true; }
        }
        if order.iter().any(|rank| *rank == usize::MAX) { return true; }
        for child in 0..count {
            if parent[child].is_none() { continue; }
            let start = order[child];
            let stop = end[child];
            if (0..count).any(|atom| (start..stop).contains(&order[atom]) && context.visited[atom]) { continue; }
            let blocked = context.held.iter().filter(|(_, target)| {
                let target = target.index();
                owner[target] == owner[child] && order[target] >= stop && !context.visited[target]
            }).count();
            let internal_ring = ring.iter().any(|edge| {
                context.unwritten[edge.bond.index()]
                    && (start..stop).contains(&order[edge.a.index()])
                    && (start..stop).contains(&order[edge.b.index()])
            });
            if blocked + usize::from(internal_ring) > context.capacity { return false; }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Domain;
    use crate::model::{BondRole, ConstraintModelBuilder, EdgeRolePartition};
    use crate::native::{NativeSolverError, NativeSolverState};

    // P-A-X is visited; P-H is held. B's H/C order decides whether C-D-E's
    // internal D-E ring must open before P-H can close.
    fn seed() -> (LiveRingCutFactor, Vec<u8>, LiveRingCutContext, VariableId) {
        let pairs = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5),
            (5, 6), (5, 7), (0, 4), (6, 7)];
        let edge_count = pairs.len();
        let edges: Arc<[ScheduleEdge]> = pairs.into_iter().enumerate().map(|(index, (a, b))| {
            ScheduleEdge {
                bond: BondId::new(index as u32),
                a: AtomId::new(a), b: AtomId::new(b),
                decision_variable: VariableId::new(index as u32),
                role_partition: EdgeRolePartition::bond_role(),
            }
        }).collect::<Vec<_>>().into();
        let mut values = vec![BondRole::Traversal.singleton_domain().iter().next().unwrap(); edge_count];
        for index in [7, 8] {
            values[index] = BondRole::Ring.singleton_domain().iter().next().unwrap();
        }
        let mut centers = Vec::new();
        let mut b_order = None;
        for atom in 0..8 {
            let incidence = edges.iter().filter(|edge| edge.a.index() == atom || edge.b.index() == atom)
                .map(|edge| edge.bond).collect::<Vec<_>>();
            let mut local = Vec::new();
            for (i, &first) in incidence.iter().enumerate() {
                for &second in &incidence[i + 1..] {
                    let variable = VariableId::new(values.len() as u32);
                    if atom == 3 && first == BondId::new(3) && second == BondId::new(4) {
                        b_order = Some(variable);
                    }
                    values.push(0);
                    local.push(OrdinaryOrderPair { first, second, variable });
                }
            }
            centers.push(ScheduleOrdinaryCenter { atom: AtomId::new(atom as u32), pairs: local.into() });
        }
        let roots: Arc<[VariableId]> = (0..8).map(|atom| {
            let variable = VariableId::new(values.len() as u32);
            values.push(u8::from(atom == 0));
            variable
        }).collect::<Vec<_>>().into();
        let factor = LiveRingCutFactor::new(edges, Arc::from([]), centers.into(), roots);
        let context = LiveRingCutContext {
            visited: vec![true, true, true, false, false, false, false, false].into(),
            unwritten: vec![false, false, true, true, true, true, true, false, true].into(),
            held: vec![(BondId::new(7), AtomId::new(4))].into(),
            capacity: 1,
            actual_tree_root: Some(AtomId::new(0)),
        };
        (factor, values, context, b_order.unwrap())
    }

    #[test]
    fn physical_subtree_pressure_respects_order_capacity_and_tree_ownership() {
        let (factor, mut values, mut context, b_order) = seed();
        assert!(factor.permits(&context, |variable| values[variable.index()]));
        values[b_order.index()] = 1; // C before H.
        assert!(!factor.permits(&context, |variable| values[variable.index()]));
        context.capacity = 2;
        assert!(factor.permits(&context, |variable| values[variable.index()]));
        context.capacity = 1;
        // H belongs to a different fixed physical tree, so a dot entry may
        // release P-H before C's internal ring is opened.
        values[3] = BondRole::Ring.singleton_domain().iter().next().unwrap();
        values[factor.roots[4].index()] = 1;
        assert!(factor.permits(&context, |variable| values[variable.index()]));
        // A completed ring may still own a pending spelling slot, but it is
        // not a held obligation with an unvisited closing endpoint.
        context.held = Box::new([]);
        values[3] = BondRole::Traversal.singleton_domain().iter().next().unwrap();
        values[factor.roots[4].index()] = 0;
        assert!(factor.permits(&context, |variable| values[variable.index()]));
    }

    #[test]
    fn native_admission_checks_the_cut_in_its_joint_assignment_search() {
        let (factor, values, context, b_order) = seed();
        let mut builder = ConstraintModelBuilder::new();
        for (index, value) in values.iter().copied().enumerate() {
            let domain = if index == b_order.index() { Domain::from_bits(3) }
                else { Domain::singleton(value).unwrap() };
            assert_eq!(builder.add_variable(domain).unwrap().index(), index);
        }
        let cut_id = builder.add_latent_live_ring_cut(factor).unwrap();
        let state = NativeSolverState::initial(Arc::new(builder.build())).unwrap();
        assert!(state.admit_live_ring_cut(cut_id, &context).is_ok(),
            "one correlated H-first assignment must survive");
        let c_first = state.with_restrictions([(b_order, Domain::singleton(1).unwrap())]).unwrap();
        assert!(matches!(c_first.admit_live_ring_cut(cut_id, &context),
            Err(NativeSolverError::Contradiction)));
        let mut two_slots = context.clone();
        two_slots.capacity = 2;
        assert!(c_first.admit_live_ring_cut(cut_id, &two_slots).is_ok());
        let mut no_held = context;
        no_held.held = Box::new([]);
        state.reset_live_cut_phase_counts();
        assert!(state.admit_live_ring_cut(cut_id, &no_held).is_ok());
        let (checks, scans, deferred, _) = state.live_cut_phase_counts();
        assert_eq!((checks, scans, deferred), (0, 0, 0),
            "H below capacity makes the cut tautological without scope inspection");
    }
}
