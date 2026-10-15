//! Test-only projection of actual writer control onto a supplied physical cut.
//! The writer and its solver remain the authorities for transitions and choices.

use std::collections::BTreeSet;

use crate::ids::{AtomId, BondId};
use crate::prepared::PreparedGraph;

use super::{BondProgress, TraversalState, WriterFrame};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum IncidenceSide {
    PiecePrivate,
    Boundary,
}

/// Each atom and physical bond has one owner. The scope of each piece also
/// includes endpoints of its owned bonds; shared scope vertices are ports.
/// This is an ownership cut, not a nice-decomposition node or relation row.
pub(crate) struct PhysicalCut<'a> {
    left_atoms: &'a BTreeSet<AtomId>,
    left_bonds: &'a BTreeSet<BondId>,
    boundary: BTreeSet<AtomId>,
}

impl<'a> PhysicalCut<'a> {
    pub(crate) fn new(
        graph: &PreparedGraph,
        left_atoms: &'a BTreeSet<AtomId>,
        left_bonds: &'a BTreeSet<BondId>,
    ) -> Self {
        assert!(left_atoms
            .iter()
            .all(|atom| atom.index() < graph.atom_count()));
        assert!(left_bonds
            .iter()
            .all(|bond| bond.index() < graph.bond_count()));
        let mut left_scope = left_atoms.clone();
        let mut right_scope: BTreeSet<_> = graph
            .atom_ids()
            .filter(|atom| !left_atoms.contains(atom))
            .collect();
        for bond in graph.bond_ids() {
            let edge = graph.bond(bond).expect("prepared bond ID must resolve");
            let scope = if left_bonds.contains(&bond) {
                &mut left_scope
            } else {
                &mut right_scope
            };
            scope.insert(edge.a());
            scope.insert(edge.b());
        }
        let boundary = left_scope.intersection(&right_scope).copied().collect();
        Self {
            left_atoms,
            left_bonds,
            boundary,
        }
    }

    pub(crate) fn boundary(&self) -> &BTreeSet<AtomId> {
        &self.boundary
    }

    fn side(&self, owner: AtomId, other: AtomId, bond: BondId) -> IncidenceSide {
        let owner_left = self.left_atoms.contains(&owner);
        if owner_left == self.left_atoms.contains(&other)
            && owner_left == self.left_bonds.contains(&bond)
            && !self.boundary.contains(&owner)
            && !self.boundary.contains(&other)
        {
            IncidenceSide::PiecePrivate
        } else {
            IncidenceSide::Boundary
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct UnresolvedIncidence {
    pub(crate) owner: AtomId,
    pub(crate) bond: BondId,
    pub(crate) side: IncidenceSide,
}

/// One projection per actual state, never one stack copy per relation row.
/// `assignment` borrows the same joint solver state as the writer.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct ActivePhase<'a> {
    frame: &'a WriterFrame,
}

pub(crate) struct LiveCutCursor<'a, S> {
    pub(crate) assignment: &'a S,
    pub(crate) active: Option<AtomId>,
    pub(crate) active_phase: Option<ActivePhase<'a>>,
    pub(crate) saved: Vec<AtomId>, // nearest return first
    pub(crate) held: Vec<(BondId, AtomId)>,
    pub(crate) unresolved: Vec<UnresolvedIncidence>,
}

impl<'a, S> LiveCutCursor<'a, S> {
    pub(crate) fn project(
        traversal: &'a TraversalState,
        assignment: &'a S,
        graph: &PreparedGraph,
        cut: &PhysicalCut<'_>,
    ) -> Self {
        let active = traversal.active_atom();
        let mut saved = Vec::new();
        let mut node = traversal.branch_returns.as_deref();
        while let Some(current) = node {
            if !owner_finished(traversal, graph, current.frame.atom) {
                saved.push(current.frame.atom);
            }
            node = current.parent.as_deref();
        }
        let held = graph
            .bond_ids()
            .filter_map(|bond| match traversal.progress.bond_progress(bond) {
                BondProgress::RingOpen { first_endpoint } => Some((bond, *first_endpoint)),
                _ => None,
            })
            .collect();
        let unresolved = unresolved(traversal, graph, cut, active, &saved);
        let active_phase = traversal
            .active
            .as_deref()
            .map(|frame| ActivePhase { frame });
        Self {
            assignment,
            active,
            active_phase,
            saved,
            held,
            unresolved,
        }
    }

    /// Advance only the projected return chain. The actual writer has already
    /// admitted and performed `)`; this method does not decide its legality.
    pub(crate) fn after_actual_return(
        &self,
        before: &TraversalState,
        after: &'a TraversalState,
        assignment: &'a S,
        graph: &PreparedGraph,
        cut: &PhysicalCut<'_>,
    ) -> Self {
        assert_eq!(self.active, before.active_atom());
        let top = before
            .branch_returns
            .as_ref()
            .expect("return requires a saved frame");
        assert_eq!(
            before.progress, after.progress,
            "return cannot change physical progress"
        );
        assert_eq!(
            top.parent, after.branch_returns,
            "return must pop exactly one actual frame"
        );
        let restored = top.frame.atom;
        assert_eq!(after.active_atom(), Some(restored));
        let mut saved = self.saved.clone();
        if saved.first() == Some(&restored) {
            saved.remove(0);
        } else {
            assert!(
                owner_finished(before, graph, restored),
                "only an exhausted saved owner may be erased"
            );
        }
        let unresolved = unresolved(after, graph, cut, Some(restored), &saved);
        let active_phase = after.active.as_deref().map(|frame| ActivePhase { frame });
        Self {
            assignment,
            active: Some(restored),
            active_phase,
            saved,
            held: self.held.clone(),
            unresolved,
        }
    }
}

fn incidence_finished(traversal: &TraversalState, bond: BondId, owner: AtomId) -> bool {
    match traversal.progress.bond_progress(bond) {
        BondProgress::Traversed { .. } | BondProgress::RingClosed { .. } => true,
        BondProgress::RingOpen { first_endpoint } => *first_endpoint == owner,
        BondProgress::Unrepresented => false,
    }
}

fn owner_finished(traversal: &TraversalState, graph: &PreparedGraph, owner: AtomId) -> bool {
    graph
        .neighbors(owner)
        .expect("saved owner belongs to prepared graph")
        .iter()
        .all(|incident| incidence_finished(traversal, incident.bond(), owner))
}

fn unresolved(
    traversal: &TraversalState,
    graph: &PreparedGraph,
    cut: &PhysicalCut<'_>,
    active: Option<AtomId>,
    saved: &[AtomId],
) -> Vec<UnresolvedIncidence> {
    std::iter::once(active)
        .flatten()
        .chain(saved.iter().copied())
        .flat_map(|owner| {
            graph
                .neighbors(owner)
                .expect("cursor owner belongs to prepared graph")
                .iter()
                .filter_map(move |incident| {
                    if incidence_finished(traversal, incident.bond(), owner) {
                        return None;
                    }
                    Some(UnresolvedIncidence {
                        owner,
                        bond: incident.bond(),
                        side: cut.side(owner, incident.atom(), incident.bond()),
                    })
                })
        })
        .collect()
}
