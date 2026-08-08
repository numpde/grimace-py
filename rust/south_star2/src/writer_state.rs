//! Concrete composition of traversal and constraint state.
//!
//! A writer state is bound to one prepared molecule. Each transition is a
//! concrete graph-walk operation; operations that decide a bond role also
//! restrict the corresponding constraint variable before publishing a successor.

use crate::domain::Domain;
use crate::ids::{AtomId, BondId, VariableId};
use crate::model::BondRole;
use crate::prepared::{AdjacentBond, PreparedMolecule};
use crate::solver::ConstraintSolver;
use crate::traversal::{IncidentBondState, RingLabelSlot, TraversalState};

#[derive(Debug, Default)]
pub(crate) struct StructuralFrontier {
    component_roots: Vec<AtomId>,
    branch_children: Vec<AdjacentBond>,
    inline_children: Vec<AdjacentBond>,
    ring_openings: Vec<AdjacentBond>,
    ring_closures: Vec<AdjacentBond>,
    can_complete_path: bool,
}

impl StructuralFrontier {
    pub(crate) fn component_roots(&self) -> &[AtomId] {
        &self.component_roots
    }

    pub(crate) fn branch_children(&self) -> &[AdjacentBond] {
        &self.branch_children
    }

    pub(crate) fn inline_children(&self) -> &[AdjacentBond] {
        &self.inline_children
    }

    pub(crate) fn ring_openings(&self) -> &[AdjacentBond] {
        &self.ring_openings
    }

    pub(crate) fn ring_closures(&self) -> &[AdjacentBond] {
        &self.ring_closures
    }

    pub(crate) const fn can_complete_path(&self) -> bool {
        self.can_complete_path
    }
}

#[derive(Clone, Debug)]
pub(crate) struct WriterState<S> {
    prepared: PreparedMolecule,
    traversal: TraversalState,
    constraints: S,
}

impl<S: ConstraintSolver> WriterState<S> {
    pub(crate) fn initial(prepared: &PreparedMolecule) -> Result<Self, S::Error> {
        let constraints = S::initial(prepared.constraint_model_arc())?;
        Ok(Self {
            prepared: prepared.clone(),
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

    fn bond_role_domain(&self, bond: BondId) -> Domain {
        let variable = role_variable(&self.prepared, bond);
        self.constraints
            .domain(variable)
            .expect("prepared bond role must belong to the writer constraint model")
    }

    pub(crate) fn structural_frontier(&self) -> StructuralFrontier {
        let graph = self.prepared.graph();
        let Some(active) = self.traversal.active_atom() else {
            return StructuralFrontier {
                component_roots: self.traversal.unvisited_atoms(graph).collect(),
                ..StructuralFrontier::default()
            };
        };

        let incidents = graph
            .neighbors(active)
            .expect("active atom must belong to the prepared graph")
            .iter()
            .copied()
            .map(|incident| {
                let state = self.traversal.classify_active_incident(graph, incident);
                (incident, state)
            })
            .collect::<Vec<_>>();
        let departure_blockers = incidents
            .iter()
            .filter(|(_, state)| incident_blocks_departure(*state))
            .count();
        let mut frontier = StructuralFrontier::default();

        for (incident, state) in incidents {
            match state {
                IncidentBondState::UnrepresentedToUnvisitedAtom => {
                    let role_domain = self.bond_role_domain(incident.bond());
                    if role_domain.contains(BondRole::Traversal.value_index()) {
                        frontier.branch_children.push(incident);
                        if departure_blockers == 1 {
                            frontier.inline_children.push(incident);
                        }
                    }
                    if role_domain.contains(BondRole::Ring.value_index()) {
                        frontier.ring_openings.push(incident);
                    }
                }
                IncidentBondState::UnrepresentedToVisitedAtom => {
                    if self
                        .bond_role_domain(incident.bond())
                        .contains(BondRole::Ring.value_index())
                    {
                        frontier.ring_openings.push(incident);
                    }
                }
                IncidentBondState::RingOpenAtOtherAtom => {
                    frontier.ring_closures.push(incident);
                }
                IncidentBondState::Represented | IncidentBondState::RingOpenAtCurrentAtom => {}
            }
        }

        frontier.can_complete_path = departure_blockers == 0 && self.traversal.can_complete_path();
        frontier
    }

    pub(crate) fn begin_component(&self, root: AtomId) -> Self {
        let mut successor = self.clone();
        successor.traversal.begin_component(root);
        successor
    }

    pub(crate) fn enter_inline_child(&self, incident: AdjacentBond) -> Result<Self, S::Error> {
        require_departure_ready(&self.traversal, &self.prepared, Some(incident.bond()));
        let mut traversal = self.traversal.clone();
        traversal.enter_inline_child(self.prepared.graph(), incident);
        let constraints = self.constraints.restricted(&[role_restriction(
            &self.prepared,
            incident.bond(),
            BondRole::Traversal,
        )])?;
        Ok(Self {
            prepared: self.prepared.clone(),
            traversal,
            constraints,
        })
    }

    pub(crate) fn enter_branch_child(&self, incident: AdjacentBond) -> Result<Self, S::Error> {
        let mut traversal = self.traversal.clone();
        traversal.enter_branch_child(self.prepared.graph(), incident);
        let constraints = self.constraints.restricted(&[role_restriction(
            &self.prepared,
            incident.bond(),
            BondRole::Traversal,
        )])?;
        Ok(Self {
            prepared: self.prepared.clone(),
            traversal,
            constraints,
        })
    }

    pub(crate) fn open_ring_endpoint(
        &self,
        incident: AdjacentBond,
    ) -> Result<(Self, RingLabelSlot), S::Error> {
        let mut traversal = self.traversal.clone();
        let label_slot = traversal.open_ring_endpoint(self.prepared.graph(), incident);
        let constraints = self.constraints.restricted(&[role_restriction(
            &self.prepared,
            incident.bond(),
            BondRole::Ring,
        )])?;
        Ok((
            Self {
                prepared: self.prepared.clone(),
                traversal,
                constraints,
            },
            label_slot,
        ))
    }

    pub(crate) fn close_ring_endpoint(&self, incident: AdjacentBond) -> (Self, RingLabelSlot) {
        let mut successor = self.clone();
        let label_slot = successor
            .traversal
            .close_ring_endpoint(self.prepared.graph(), incident);
        (successor, label_slot)
    }

    pub(crate) fn complete_path(&self) -> Self {
        require_departure_ready(&self.traversal, &self.prepared, None);
        let mut successor = self.clone();
        successor.traversal.complete_path();
        successor
    }
}

fn incident_blocks_departure(state: IncidentBondState) -> bool {
    !matches!(
        state,
        IncidentBondState::Represented | IncidentBondState::RingOpenAtCurrentAtom
    )
}

fn departure_blocker_count(
    traversal: &TraversalState,
    prepared: &PreparedMolecule,
    ignored_bond: Option<BondId>,
) -> usize {
    let active = traversal
        .active_atom()
        .expect("departure requires an active atom");
    prepared
        .graph()
        .neighbors(active)
        .expect("active atom must belong to the prepared graph")
        .iter()
        .filter(|incident| ignored_bond != Some(incident.bond()))
        .filter(|incident| {
            incident_blocks_departure(
                traversal.classify_active_incident(prepared.graph(), **incident),
            )
        })
        .count()
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
    assert_eq!(
        departure_blocker_count(traversal, prepared, departing_via),
        0,
        "leaving an atom must not strand unresolved incident graph work"
    );
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

    fn assert_no_active_incident_choices(frontier: &StructuralFrontier) {
        assert!(frontier.branch_children().is_empty());
        assert!(frontier.inline_children().is_empty());
        assert!(frontier.ring_openings().is_empty());
        assert!(frontier.ring_closures().is_empty());
    }

    #[test]
    fn initial_frontier_contains_only_component_roots() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        graph.add_bond(atoms[0], atoms[1]).unwrap();
        graph.add_bond(atoms[1], atoms[2]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let state = WriterState::<NativeSolverState>::initial(&prepared).unwrap();

        let frontier = state.structural_frontier();

        assert_eq!(frontier.component_roots(), &atoms);
        assert_no_active_incident_choices(&frontier);
        assert!(!frontier.can_complete_path());
    }

    #[test]
    fn triangle_frontier_and_transitions_follow_role_propagation() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let first = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let second = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let ring = graph.add_bond(atoms[2], atoms[0]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(atoms[0]);
        let first_incident = incident(&prepared, atoms[0], first);
        let ring_opening = incident(&prepared, atoms[0], ring);

        let rooted_frontier = rooted.structural_frontier();
        assert!(rooted_frontier.component_roots().is_empty());
        assert_eq!(
            rooted_frontier.branch_children(),
            &[first_incident, ring_opening]
        );
        assert!(rooted_frontier.inline_children().is_empty());
        assert_eq!(
            rooted_frontier.ring_openings(),
            &[first_incident, ring_opening]
        );
        assert!(rooted_frontier.ring_closures().is_empty());
        assert!(!rooted_frontier.can_complete_path());
        assert_eq!(rooted.bond_role_domain(ring), BondRole::role_domain());

        let (opened, label_slot) = rooted.open_ring_endpoint(ring_opening).unwrap();
        assert_eq!(
            opened.bond_role_domain(ring),
            BondRole::Ring.singleton_domain()
        );
        assert_eq!(
            opened.bond_role_domain(first),
            BondRole::Traversal.singleton_domain()
        );
        assert_eq!(
            opened.bond_role_domain(second),
            BondRole::Traversal.singleton_domain()
        );
        let opened_frontier = opened.structural_frontier();
        assert_eq!(opened_frontier.branch_children(), &[first_incident]);
        assert_eq!(opened_frontier.inline_children(), &[first_incident]);
        assert!(opened_frontier.ring_openings().is_empty());
        assert!(opened_frontier.ring_closures().is_empty());
        assert!(!opened_frontier.can_complete_path());

        let walked = opened
            .enter_inline_child(first_incident)
            .unwrap()
            .enter_inline_child(incident(&prepared, atoms[1], second))
            .unwrap();
        assert_eq!(walked.active_atom(), Some(atoms[2]));
        let closing = incident(&prepared, atoms[2], ring);
        let walked_frontier = walked.structural_frontier();
        assert!(walked_frontier.branch_children().is_empty());
        assert!(walked_frontier.inline_children().is_empty());
        assert!(walked_frontier.ring_openings().is_empty());
        assert_eq!(walked_frontier.ring_closures(), &[closing]);
        assert!(!walked_frontier.can_complete_path());

        let (closed, closed_slot) = walked.close_ring_endpoint(closing);
        assert_eq!(closed_slot, label_slot);
        assert!(closed.graph_is_complete());
        let closed_frontier = closed.structural_frontier();
        assert_no_active_incident_choices(&closed_frontier);
        assert!(closed_frontier.can_complete_path());

        let finished = closed.complete_path();
        assert_eq!(finished.active_atom(), None);
        assert!(finished.graph_is_complete());
    }

    #[test]
    fn bridge_frontier_filters_ring_choice_and_direct_attempt_contradicts() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 2] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let bridge = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(atoms[0]);
        let edge = incident(&prepared, atoms[0], bridge);

        let frontier = rooted.structural_frontier();
        assert_eq!(frontier.branch_children(), &[edge]);
        assert_eq!(frontier.inline_children(), &[edge]);
        assert!(frontier.ring_openings().is_empty());
        assert!(frontier.ring_closures().is_empty());
        assert!(!frontier.can_complete_path());
        assert_eq!(
            rooted.bond_role_domain(bridge),
            BondRole::Traversal.singleton_domain()
        );

        assert!(matches!(
            rooted.open_ring_endpoint(edge),
            Err(NativeSolverError::Contradiction)
        ));
        assert_eq!(rooted.active_atom(), Some(atoms[0]));

        let traversed = rooted.enter_inline_child(edge).unwrap();
        assert_eq!(traversed.active_atom(), Some(atoms[1]));
        assert!(traversed.graph_is_complete());
    }

    #[test]
    fn completed_component_frontier_returns_to_remaining_roots() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 2] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let prepared = PreparedMolecule::new(graph.build());
        let initial = WriterState::<NativeSolverState>::initial(&prepared).unwrap();
        let rooted = initial.begin_component(atoms[1]);

        let rooted_frontier = rooted.structural_frontier();
        assert!(rooted_frontier.component_roots().is_empty());
        assert_no_active_incident_choices(&rooted_frontier);
        assert!(rooted_frontier.can_complete_path());

        let completed = rooted.complete_path();
        let completed_frontier = completed.structural_frontier();
        assert_eq!(completed_frontier.component_roots(), &[atoms[0]]);
        assert_no_active_incident_choices(&completed_frontier);
        assert!(!completed_frontier.can_complete_path());
    }

    #[test]
    fn branch_child_restores_parent_before_inline_departure() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let branch = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let inline = graph.add_bond(atoms[0], atoms[2]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(atoms[0]);

        let branched = rooted
            .enter_branch_child(incident(&prepared, atoms[0], branch))
            .unwrap();
        assert_eq!(branched.active_atom(), Some(atoms[1]));
        assert!(branched.structural_frontier().can_complete_path());

        let restored = branched.complete_path();
        assert_eq!(restored.active_atom(), Some(atoms[0]));
        let finished = restored
            .enter_inline_child(incident(&prepared, atoms[0], inline))
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
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(atoms[0]);

        let _ = rooted.enter_inline_child(incident(&prepared, atoms[0], first));
    }

    #[test]
    #[should_panic(expected = "must not strand unresolved incident graph work")]
    fn path_cannot_finish_before_a_remote_ring_endpoint_is_closed() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let first = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let second = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let ring = graph.add_bond(atoms[2], atoms[0]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(atoms[0]);
        let (opened, _) = rooted
            .open_ring_endpoint(incident(&prepared, atoms[0], ring))
            .unwrap();
        let walked = opened
            .enter_inline_child(incident(&prepared, atoms[0], first))
            .unwrap()
            .enter_inline_child(incident(&prepared, atoms[1], second))
            .unwrap();

        let _ = walked.complete_path();
    }
}
