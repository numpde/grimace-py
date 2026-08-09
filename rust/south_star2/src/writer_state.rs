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
    may_finish_ring_choices: bool,
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

    pub(crate) const fn may_finish_ring_choices(&self) -> bool {
        self.may_finish_ring_choices
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

        let mut frontier = StructuralFrontier::default();
        let mut traversal_children = Vec::new();
        let mut ring_phase = false;
        let mut finish_ring_choices_possible = true;
        let mut has_ring_capable_child = false;

        for incident in graph
            .neighbors(active)
            .expect("active atom must belong to the prepared graph")
            .iter()
            .copied()
        {
            let state = self.traversal.classify_active_incident(graph, incident);
            match state {
                IncidentBondState::UnrepresentedToUnvisitedAtom => {
                    let role_domain = self.bond_role_domain(incident.bond());
                    let can_traverse =
                        role_domain.contains(BondRole::Traversal.value_index());
                    let can_ring = role_domain.contains(BondRole::Ring.value_index());
                    assert!(
                        can_traverse || can_ring,
                        "a live bond-role domain must contain Traversal or Ring"
                    );

                    if can_ring {
                        ring_phase = true;
                        has_ring_capable_child = true;
                        frontier.ring_openings.push(incident);
                    } else {
                        traversal_children.push(incident);
                    }
                    if !can_traverse {
                        finish_ring_choices_possible = false;
                    }
                }
                IncidentBondState::UnrepresentedToVisitedAtom => {
                    assert!(
                        self.bond_role_domain(incident.bond())
                            .contains(BondRole::Ring.value_index()),
                        "an unrepresented bond to a visited atom must remain Ring-capable"
                    );
                    ring_phase = true;
                    finish_ring_choices_possible = false;
                    frontier.ring_openings.push(incident);
                }
                IncidentBondState::RingOpenAtOtherAtom => {
                    assert_eq!(
                        self.bond_role_domain(incident.bond()),
                        BondRole::Ring.singleton_domain(),
                        "an open ring bond must be constrained to Ring"
                    );
                    ring_phase = true;
                    finish_ring_choices_possible = false;
                    frontier.ring_closures.push(incident);
                }
                IncidentBondState::RingOpenAtCurrentAtom => {
                    assert_eq!(
                        self.bond_role_domain(incident.bond()),
                        BondRole::Ring.singleton_domain(),
                        "an open ring bond must be constrained to Ring"
                    );
                }
                IncidentBondState::Represented => {}
            }
        }

        if ring_phase {
            frontier.may_finish_ring_choices =
                finish_ring_choices_possible && has_ring_capable_child;
            return frontier;
        }

        match traversal_children.len() {
            0 => {
                frontier.can_complete_path = self.traversal.can_complete_path();
            }
            1 => {
                frontier.inline_children = traversal_children;
            }
            _ => {
                frontier.branch_children = traversal_children;
            }
        }
        frontier
    }

    pub(crate) fn begin_component(&self, root: AtomId) -> Self {
        assert!(
            self.structural_frontier().component_roots().contains(&root),
            "component root must be advertised by the structural frontier"
        );
        let mut successor = self.clone();
        successor.traversal.begin_component(root);
        successor
    }

    pub(crate) fn finish_ring_choices(&self) -> Result<Self, S::Error> {
        assert!(
            self.structural_frontier().may_finish_ring_choices(),
            "ring choices can finish only while undecided incident bonds can all become Traversal"
        );

        let graph = self.prepared.graph();
        let active = self
            .traversal
            .active_atom()
            .expect("finishing ring choices requires an active atom");
        let restrictions = graph
            .neighbors(active)
            .expect("active atom must belong to the prepared graph")
            .iter()
            .copied()
            .filter(|incident| {
                self.traversal.classify_active_incident(graph, *incident)
                    == IncidentBondState::UnrepresentedToUnvisitedAtom
            })
            .map(|incident| {
                role_restriction(
                    &self.prepared,
                    incident.bond(),
                    BondRole::Traversal,
                )
            })
            .collect::<Vec<_>>();
        debug_assert!(!restrictions.is_empty());

        let constraints = self.constraints.restricted(&restrictions)?;
        Ok(Self {
            prepared: self.prepared.clone(),
            traversal: self.traversal.clone(),
            constraints,
        })
    }

    pub(crate) fn enter_inline_child(&self, incident: AdjacentBond) -> Self {
        assert_eq!(
            self.structural_frontier().inline_children(),
            &[incident],
            "the sole remaining Traversal child is the inline continuation"
        );
        let mut successor = self.clone();
        successor
            .traversal
            .enter_inline_child(self.prepared.graph(), incident);
        successor
    }

    pub(crate) fn enter_branch_child(&self, incident: AdjacentBond) -> Self {
        assert!(
            self.structural_frontier()
                .branch_children()
                .contains(&incident),
            "a branch child requires more than one remaining Traversal child"
        );

        let mut successor = self.clone();
        successor
            .traversal
            .enter_branch_child(self.prepared.graph(), incident);
        successor
    }

    pub(crate) fn open_ring_endpoint(
        &self,
        incident: AdjacentBond,
    ) -> Result<(Self, RingLabelSlot), S::Error> {
        assert!(
            self.structural_frontier().ring_openings().contains(&incident),
            "a ring opening must be advertised by the structural frontier"
        );

        let constraints = self.constraints.restricted(&[role_restriction(
            &self.prepared,
            incident.bond(),
            BondRole::Ring,
        )])?;
        let mut traversal = self.traversal.clone();
        let label_slot = traversal.open_ring_endpoint(self.prepared.graph(), incident);
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
        assert!(
            self.structural_frontier().ring_closures().contains(&incident),
            "a ring closure must be advertised by the structural frontier"
        );

        let mut successor = self.clone();
        let label_slot = successor
            .traversal
            .close_ring_endpoint(self.prepared.graph(), incident);
        (successor, label_slot)
    }

    pub(crate) fn complete_path(&self) -> Self {
        assert!(
            self.structural_frontier().can_complete_path(),
            "the active path can complete only after all local ring and child work"
        );
        let mut successor = self.clone();
        successor.traversal.complete_path();
        successor
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

    fn assert_no_active_incident_choices(frontier: &StructuralFrontier) {
        assert!(frontier.branch_children().is_empty());
        assert!(frontier.inline_children().is_empty());
        assert!(frontier.ring_openings().is_empty());
        assert!(frontier.ring_closures().is_empty());
        assert!(!frontier.may_finish_ring_choices());
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
    fn sole_traversal_child_is_inline_only() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 2] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let bond = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(atoms[0]);
        let edge = incident(&prepared, atoms[0], bond);

        let frontier = rooted.structural_frontier();

        assert!(frontier.branch_children().is_empty());
        assert_eq!(frontier.inline_children(), &[edge]);
        assert!(frontier.ring_openings().is_empty());
        assert!(frontier.ring_closures().is_empty());
        assert!(!frontier.may_finish_ring_choices());
        assert!(!frontier.can_complete_path());

        let traversed = rooted.enter_inline_child(edge);
        assert_eq!(traversed.active_atom(), Some(atoms[1]));
        assert!(traversed.graph_is_complete());
    }

    #[test]
    fn ring_choice_forces_a_linear_triangle_continuation() {
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
        assert!(rooted_frontier.branch_children().is_empty());
        assert!(rooted_frontier.inline_children().is_empty());
        assert_eq!(
            rooted_frontier.ring_openings(),
            &[first_incident, ring_opening]
        );
        assert!(rooted_frontier.ring_closures().is_empty());
        assert!(rooted_frontier.may_finish_ring_choices());
        assert!(!rooted_frontier.can_complete_path());

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
        assert!(opened_frontier.branch_children().is_empty());
        assert_eq!(opened_frontier.inline_children(), &[first_incident]);
        assert!(opened_frontier.ring_openings().is_empty());
        assert!(opened_frontier.ring_closures().is_empty());
        assert!(!opened_frontier.may_finish_ring_choices());

        let walked = opened
            .enter_inline_child(first_incident)
            .enter_inline_child(incident(&prepared, atoms[1], second));
        let closing = incident(&prepared, atoms[2], ring);
        let walked_frontier = walked.structural_frontier();
        assert!(walked_frontier.branch_children().is_empty());
        assert!(walked_frontier.inline_children().is_empty());
        assert!(walked_frontier.ring_openings().is_empty());
        assert_eq!(walked_frontier.ring_closures(), &[closing]);
        assert!(!walked_frontier.may_finish_ring_choices());
        assert!(!walked_frontier.can_complete_path());

        let (closed, closed_slot) = walked.close_ring_endpoint(closing);
        assert_eq!(closed_slot, label_slot);
        assert!(closed.graph_is_complete());
        assert!(closed.structural_frontier().can_complete_path());
    }

    #[test]
    fn finished_ring_choices_require_branches_before_one_inline_child() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let first = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let between = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let last = graph.add_bond(atoms[2], atoms[0]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(atoms[0]);
        let first_incident = incident(&prepared, atoms[0], first);
        let last_incident = incident(&prepared, atoms[0], last);

        let sealed = rooted.finish_ring_choices().unwrap();
        assert_eq!(
            sealed.bond_role_domain(first),
            BondRole::Traversal.singleton_domain()
        );
        assert_eq!(
            sealed.bond_role_domain(last),
            BondRole::Traversal.singleton_domain()
        );
        assert_eq!(
            sealed.bond_role_domain(between),
            BondRole::Ring.singleton_domain()
        );

        let sealed_frontier = sealed.structural_frontier();
        assert_eq!(
            sealed_frontier.branch_children(),
            &[first_incident, last_incident]
        );
        assert!(sealed_frontier.inline_children().is_empty());
        assert!(sealed_frontier.ring_openings().is_empty());
        assert!(!sealed_frontier.may_finish_ring_choices());

        let branched = sealed.enter_branch_child(first_incident);
        let branch_ring = incident(&prepared, atoms[1], between);
        let branch_frontier = branched.structural_frontier();
        assert!(branch_frontier.branch_children().is_empty());
        assert!(branch_frontier.inline_children().is_empty());
        assert_eq!(branch_frontier.ring_openings(), &[branch_ring]);
        assert!(!branch_frontier.may_finish_ring_choices());
        assert!(!branch_frontier.can_complete_path());

        let (opened, label_slot) = branched.open_ring_endpoint(branch_ring).unwrap();
        assert!(opened.structural_frontier().can_complete_path());
        let restored = opened.complete_path();
        assert_eq!(restored.active_atom(), Some(atoms[0]));

        let restored_frontier = restored.structural_frontier();
        assert!(restored_frontier.branch_children().is_empty());
        assert_eq!(restored_frontier.inline_children(), &[last_incident]);
        assert!(restored_frontier.ring_openings().is_empty());
        assert!(!restored_frontier.may_finish_ring_choices());

        let walked = restored.enter_inline_child(last_incident);
        let closing = incident(&prepared, atoms[2], between);
        assert_eq!(walked.structural_frontier().ring_closures(), &[closing]);
        let (closed, closed_slot) = walked.close_ring_endpoint(closing);
        assert_eq!(closed_slot, label_slot);
        assert!(closed.graph_is_complete());
    }

    #[test]
    fn finishing_ring_choices_can_reveal_a_joint_contradiction() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let left = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let between = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let right = graph.add_bond(atoms[2], atoms[0]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(atoms[0]);
        let constraints = rooted
            .constraints
            .restricted(&[role_restriction(
                &prepared,
                between,
                BondRole::Traversal,
            )])
            .unwrap();
        let constrained = WriterState {
            prepared: rooted.prepared.clone(),
            traversal: rooted.traversal.clone(),
            constraints,
        };

        let frontier = constrained.structural_frontier();
        assert_eq!(
            frontier.ring_openings(),
            &[
                incident(&prepared, atoms[0], left),
                incident(&prepared, atoms[0], right),
            ]
        );
        assert!(frontier.may_finish_ring_choices());
        assert!(matches!(
            constrained.finish_ring_choices(),
            Err(NativeSolverError::Contradiction)
        ));
        assert_eq!(constrained.active_atom(), Some(atoms[0]));
        assert_eq!(constrained.bond_role_domain(left), BondRole::role_domain());
        assert_eq!(
            constrained.bond_role_domain(right),
            BondRole::role_domain()
        );

        let (opened, _) = constrained
            .open_ring_endpoint(incident(&prepared, atoms[0], left))
            .unwrap();
        assert_eq!(
            opened.bond_role_domain(right),
            BondRole::Traversal.singleton_domain()
        );
    }

    #[test]
    fn two_children_require_a_branch_then_one_inline_continuation() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let first = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let second = graph.add_bond(atoms[0], atoms[2]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(atoms[0]);
        let first_incident = incident(&prepared, atoms[0], first);
        let second_incident = incident(&prepared, atoms[0], second);

        let rooted_frontier = rooted.structural_frontier();
        assert_eq!(
            rooted_frontier.branch_children(),
            &[first_incident, second_incident]
        );
        assert!(rooted_frontier.inline_children().is_empty());
        assert!(!rooted_frontier.may_finish_ring_choices());

        let branch = rooted.enter_branch_child(first_incident);
        assert!(branch.structural_frontier().can_complete_path());
        let restored = branch.complete_path();
        assert_eq!(restored.active_atom(), Some(atoms[0]));

        let restored_frontier = restored.structural_frontier();
        assert!(restored_frontier.branch_children().is_empty());
        assert_eq!(restored_frontier.inline_children(), &[second_incident]);
        assert!(!restored_frontier.can_complete_path());

        let finished = restored.enter_inline_child(second_incident);
        assert_eq!(finished.active_atom(), Some(atoms[2]));
        assert!(finished.graph_is_complete());
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
    #[should_panic(expected = "the active path can complete only after")]
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
            .enter_inline_child(incident(&prepared, atoms[1], second));

        let _ = walked.complete_path();
    }
}
