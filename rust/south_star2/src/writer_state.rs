//! Concrete composition of traversal and constraint state.
//!
//! This is intentionally not a generic transition wrapper. Each method owns one
//! graph-walk operation and applies exactly the constraint restriction implied by
//! that operation before publishing its successor.

use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::{AtomId, BondId, VariableId};
use crate::model::BondRole;
use crate::prepared::{AdjacentBond, PreparedMolecule};
use crate::solver::ConstraintSolver;
use crate::traversal::{IncidentBondState, RingLabelSlot, TraversalState};

#[derive(Clone, Debug)]
pub(crate) struct WriterState<S> {
    traversal: TraversalState,
    constraints: S,
}

impl<S: ConstraintSolver> WriterState<S> {
    pub(crate) fn initial(prepared: &PreparedMolecule) -> Result<Self, S::Error> {
        let constraints = S::initial(Arc::new(prepared.constraint_model().clone()))?;
        Ok(Self {
            traversal: TraversalState::new(prepared.graph()),
            constraints,
        })
    }

    pub(crate) const fn active_atom(&self) -> Option<AtomId> {
        self.traversal.active_atom()
    }

    pub(crate) const fn graph_is_complete(&self) -> bool {
        self.traversal.graph_is_complete()
    }

    pub(crate) fn bond_role_domain(
        &self,
        prepared: &PreparedMolecule,
        bond: BondId,
    ) -> Option<Domain> {
        let variable = prepared.bond_role_variable(bond)?;
        self.constraints.domain(variable)
    }

    pub(crate) fn begin_component(&self, prepared: &PreparedMolecule, root: AtomId) -> Self {
        assert!(
            prepared.graph().neighbors(root).is_some(),
            "component root must belong to the prepared graph"
        );
        let mut successor = self.clone();
        successor.traversal.begin_component(root);
        successor
    }

    pub(crate) fn enter_inline_child(
        &self,
        prepared: &PreparedMolecule,
        incident: AdjacentBond,
    ) -> Result<Self, S::Error> {
        require_child_candidate(&self.traversal, prepared, incident);
        require_departure_ready(&self.traversal, prepared, Some(incident.bond()));
        let constraints = self.constraints.restricted(&[role_restriction(
            prepared,
            incident.bond(),
            BondRole::Traversal,
        )])?;
        let mut traversal = self.traversal.clone();
        traversal.enter_inline_child(prepared.graph(), incident);
        Ok(Self {
            traversal,
            constraints,
        })
    }

    pub(crate) fn enter_branch_child(
        &self,
        prepared: &PreparedMolecule,
        incident: AdjacentBond,
    ) -> Result<Self, S::Error> {
        require_child_candidate(&self.traversal, prepared, incident);
        let constraints = self.constraints.restricted(&[role_restriction(
            prepared,
            incident.bond(),
            BondRole::Traversal,
        )])?;
        let mut traversal = self.traversal.clone();
        traversal.enter_branch_child(prepared.graph(), incident);
        Ok(Self {
            traversal,
            constraints,
        })
    }

    pub(crate) fn open_ring_endpoint(
        &self,
        prepared: &PreparedMolecule,
        incident: AdjacentBond,
    ) -> Result<(Self, RingLabelSlot), S::Error> {
        require_ring_open_candidate(&self.traversal, prepared, incident);
        let constraints = self.constraints.restricted(&[role_restriction(
            prepared,
            incident.bond(),
            BondRole::Ring,
        )])?;
        let mut traversal = self.traversal.clone();
        let label_slot = traversal.open_ring_endpoint(prepared.graph(), incident);
        Ok((
            Self {
                traversal,
                constraints,
            },
            label_slot,
        ))
    }

    pub(crate) fn close_ring_endpoint(
        &self,
        prepared: &PreparedMolecule,
        incident: AdjacentBond,
    ) -> (Self, RingLabelSlot) {
        let role_variable = role_variable(prepared, incident.bond());
        assert_eq!(
            self.constraints.domain(role_variable),
            Some(BondRole::Ring.singleton_domain()),
            "a closing ring bond must already be constrained to the Ring role"
        );

        let mut successor = self.clone();
        let label_slot = successor
            .traversal
            .close_ring_endpoint(prepared.graph(), incident);
        (successor, label_slot)
    }

    pub(crate) fn complete_path(&self, prepared: &PreparedMolecule) -> (Self, Option<AtomId>) {
        require_departure_ready(&self.traversal, prepared, None);
        let mut successor = self.clone();
        let restored = successor.traversal.complete_path();
        (successor, restored)
    }
}

fn require_child_candidate(
    traversal: &TraversalState,
    prepared: &PreparedMolecule,
    incident: AdjacentBond,
) {
    assert_eq!(
        traversal.classify_active_incident(prepared.graph(), incident),
        IncidentBondState::UnrepresentedToUnvisitedAtom,
        "a child edge must be unrepresented and lead to an unvisited atom"
    );
}

fn require_ring_open_candidate(
    traversal: &TraversalState,
    prepared: &PreparedMolecule,
    incident: AdjacentBond,
) {
    assert!(
        matches!(
            traversal.classify_active_incident(prepared.graph(), incident),
            IncidentBondState::UnrepresentedToUnvisitedAtom
                | IncidentBondState::UnrepresentedToVisitedAtom
        ),
        "a first ring endpoint requires an unrepresented incident bond"
    );
}

/// Require that leaving the active atom cannot strand graph representation work.
///
/// A branch child is exempt because the active atom is restored afterwards.
/// Inline traversal permanently leaves the parent, so every other incident bond
/// must already be represented or have its first ring endpoint written here.
fn require_departure_ready(
    traversal: &TraversalState,
    prepared: &PreparedMolecule,
    departing_via: Option<BondId>,
) {
    let active = traversal
        .active_atom()
        .expect("departure requires an active atom");
    let neighbours = prepared
        .graph()
        .neighbors(active)
        .expect("active atom must belong to the prepared graph");

    for incident in neighbours {
        if departing_via == Some(incident.bond()) {
            continue;
        }
        assert!(
            matches!(
                traversal.classify_active_incident(prepared.graph(), *incident),
                IncidentBondState::Represented | IncidentBondState::RingOpenAtCurrentAtom
            ),
            "leaving an atom must not strand unresolved incident graph work"
        );
    }
}

fn role_variable(prepared: &PreparedMolecule, bond: BondId) -> VariableId {
    prepared
        .bond_role_variable(bond)
        .expect("prepared bond must have a role variable")
}

fn role_restriction(
    prepared: &PreparedMolecule,
    bond: BondId,
    role: BondRole,
) -> (VariableId, Domain) {
    (role_variable(prepared, bond), role.singleton_domain())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::{NativeSolverError, NativeSolverState};
    use crate::prepared::PreparedGraphBuilder;

    fn incident(prepared: &PreparedMolecule, atom: AtomId, bond: BondId) -> AdjacentBond {
        prepared
            .graph()
            .neighbors(atom)
            .expect("fixture atom must exist")
            .iter()
            .copied()
            .find(|candidate| candidate.bond() == bond)
            .expect("fixture bond must be incident to the atom")
    }

    #[test]
    fn ring_choice_propagates_and_completes_a_triangle_walk() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let first = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let second = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let ring = graph.add_bond(atoms[2], atoms[0]).unwrap();
        let prepared = PreparedMolecule::new(graph.build()).unwrap();
        let initial = WriterState::<NativeSolverState>::initial(&prepared).unwrap();
        let rooted = initial.begin_component(&prepared, atoms[0]);

        assert_eq!(
            rooted.bond_role_domain(&prepared, ring),
            Some(BondRole::role_domain())
        );

        let (opened, label_slot) = rooted
            .open_ring_endpoint(&prepared, incident(&prepared, atoms[0], ring))
            .unwrap();

        assert_eq!(
            rooted.bond_role_domain(&prepared, ring),
            Some(BondRole::role_domain())
        );
        assert_eq!(
            opened.bond_role_domain(&prepared, ring),
            Some(BondRole::Ring.singleton_domain())
        );
        assert_eq!(
            opened.bond_role_domain(&prepared, first),
            Some(BondRole::Traversal.singleton_domain())
        );
        assert_eq!(
            opened.bond_role_domain(&prepared, second),
            Some(BondRole::Traversal.singleton_domain())
        );

        let walked = opened
            .enter_inline_child(&prepared, incident(&prepared, atoms[0], first))
            .unwrap()
            .enter_inline_child(&prepared, incident(&prepared, atoms[1], second))
            .unwrap();
        assert_eq!(walked.active_atom(), Some(atoms[2]));

        let (closed, closed_slot) =
            walked.close_ring_endpoint(&prepared, incident(&prepared, atoms[2], ring));
        assert_eq!(closed_slot, label_slot);
        assert!(closed.graph_is_complete());

        let (finished, restored) = closed.complete_path(&prepared);
        assert_eq!(restored, None);
        assert_eq!(finished.active_atom(), None);
        assert!(finished.graph_is_complete());
    }

    #[test]
    fn ring_choice_on_a_bridge_is_a_contradiction_without_changing_the_source() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 2] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let bridge = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let prepared = PreparedMolecule::new(graph.build()).unwrap();
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(&prepared, atoms[0]);

        assert_eq!(
            rooted.bond_role_domain(&prepared, bridge),
            Some(BondRole::Traversal.singleton_domain())
        );
        assert!(matches!(
            rooted.open_ring_endpoint(&prepared, incident(&prepared, atoms[0], bridge)),
            Err(NativeSolverError::Contradiction)
        ));
        assert_eq!(rooted.active_atom(), Some(atoms[0]));
        assert!(!rooted.graph_is_complete());

        let traversed = rooted
            .enter_inline_child(&prepared, incident(&prepared, atoms[0], bridge))
            .unwrap();
        assert_eq!(traversed.active_atom(), Some(atoms[1]));
        assert!(traversed.graph_is_complete());
    }

    #[test]
    fn branch_child_is_a_concrete_traversal_role_transition() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let branch = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let inline = graph.add_bond(atoms[0], atoms[2]).unwrap();
        let prepared = PreparedMolecule::new(graph.build()).unwrap();
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(&prepared, atoms[0]);

        let branched = rooted
            .enter_branch_child(&prepared, incident(&prepared, atoms[0], branch))
            .unwrap();
        assert_eq!(branched.active_atom(), Some(atoms[1]));
        assert_eq!(
            branched.bond_role_domain(&prepared, branch),
            Some(BondRole::Traversal.singleton_domain())
        );

        let (restored, parent) = branched.complete_path(&prepared);
        assert_eq!(parent, Some(atoms[0]));
        let finished = restored
            .enter_inline_child(&prepared, incident(&prepared, atoms[0], inline))
            .unwrap();
        assert_eq!(finished.active_atom(), Some(atoms[2]));
        assert!(finished.graph_is_complete());
    }

    #[test]
    #[should_panic(expected = "must not strand unresolved incident graph work")]
    fn inline_child_cannot_abandon_another_unrepresented_bond() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let first = graph.add_bond(atoms[0], atoms[1]).unwrap();
        graph.add_bond(atoms[0], atoms[2]).unwrap();
        let prepared = PreparedMolecule::new(graph.build()).unwrap();
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(&prepared, atoms[0]);

        let _ = rooted.enter_inline_child(&prepared, incident(&prepared, atoms[0], first));
    }

    #[test]
    #[should_panic(expected = "must not strand unresolved incident graph work")]
    fn path_cannot_finish_before_a_remote_ring_endpoint_is_closed() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let first = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let second = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let ring = graph.add_bond(atoms[2], atoms[0]).unwrap();
        let prepared = PreparedMolecule::new(graph.build()).unwrap();
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(&prepared, atoms[0]);
        let (opened, _) = rooted
            .open_ring_endpoint(&prepared, incident(&prepared, atoms[0], ring))
            .unwrap();
        let walked = opened
            .enter_inline_child(&prepared, incident(&prepared, atoms[0], first))
            .unwrap()
            .enter_inline_child(&prepared, incident(&prepared, atoms[1], second))
            .unwrap();

        let _ = walked.complete_path(&prepared);
    }
}
