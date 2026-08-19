//! Concrete composition of residual writer frames and constraint state.
//!
//! The graphic-matroid factor guards global Traversal/Ring feasibility. Writer
//! policy is stricter: each residual attachment receives exactly one traversal
//! entry, while its other active incidences become ring endpoints.

use crate::domain::Domain;
use crate::ids::{AtomId, BondId, VariableId};
use crate::model::BondRole;
use crate::prepared::{AdjacentBond, PreparedMolecule};
use crate::solver::ConstraintSolver;
use crate::traversal::{IncidentBondState, TraversalState};

#[derive(Debug, Default)]
pub(crate) struct StructuralFrontier {
    component_roots: Vec<AtomId>,
    branch_children: Vec<AdjacentBond>,
    inline_children: Vec<AdjacentBond>,
    ring_openings: Vec<AdjacentBond>,
    ring_closures: Vec<AdjacentBond>,
    can_complete_path: bool,
    contradiction: bool,
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

    pub(crate) const fn is_contradiction(&self) -> bool {
        self.contradiction
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

    pub(crate) fn active_atom(&self) -> Option<AtomId> {
        self.traversal.active_atom()
    }

    pub(crate) const fn graph_is_complete(&self) -> bool {
        self.traversal.graph_is_complete()
    }

    pub(crate) fn ring_closure_first_endpoint(&self, incident: AdjacentBond) -> AtomId {
        assert_eq!(
            self.traversal
                .classify_active_incident(self.prepared.graph(), incident),
            IncidentBondState::RingOpenAtOtherAtom,
            "ring-closure facts require an endpoint opened at the other atom"
        );
        self.traversal
            .ring_first_endpoint_for_active_incident(self.prepared.graph(), incident)
            .expect("an advertised ring closure must retain its first endpoint")
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
        let mut ring_phase = false;
        let mut children = Vec::new();

        for incident in graph
            .neighbors(active)
            .expect("active atom must belong to the prepared graph")
            .iter()
            .copied()
        {
            match self.traversal.classify_active_incident(graph, incident) {
                IncidentBondState::RingOpenAtOtherAtom => {
                    ring_phase = true;
                    frontier.ring_closures.push(incident);
                }
                IncidentBondState::UnrepresentedToVisitedAtom => {
                    panic!(
                        "a visited endpoint must already be represented or own an open ring endpoint"
                    );
                }
                IncidentBondState::Represented
                | IncidentBondState::RingOpenAtCurrentAtom
                | IncidentBondState::UnrepresentedToUnvisitedAtom => {}
            }
        }

        for attachment in self.traversal.active_attachments() {
            let incidences = attachment.incidences();
            assert!(
                !incidences.is_empty(),
                "residual attachments must not be empty"
            );

            let role_domains = incidences
                .iter()
                .copied()
                .map(|incident| (incident, self.bond_role_domain(incident.bond())))
                .collect::<Vec<_>>();
            let traversal_capable_count = role_domains
                .iter()
                .filter(|(_, domain)| domain.contains(BondRole::Traversal.value_index()))
                .count();

            if traversal_capable_count == 0 {
                frontier.contradiction = true;
                continue;
            }

            if incidences.len() == 1 {
                children.push(incidences[0]);
                continue;
            }

            ring_phase = true;
            let ring_opening_count = frontier.ring_openings.len();
            for (candidate, domain) in role_domains {
                if !domain.contains(BondRole::Ring.value_index()) {
                    continue;
                }
                let candidate_can_traverse = domain.contains(BondRole::Traversal.value_index());
                if traversal_capable_count > usize::from(candidate_can_traverse) {
                    frontier.ring_openings.push(candidate);
                }
            }
            if frontier.ring_openings.len() == ring_opening_count {
                frontier.contradiction = true;
            }
        }

        if frontier.contradiction {
            frontier.branch_children.clear();
            frontier.inline_children.clear();
            frontier.ring_openings.clear();
            frontier.ring_closures.clear();
            return frontier;
        }
        if ring_phase {
            return frontier;
        }

        match children.len() {
            0 => frontier.can_complete_path = self.traversal.can_complete_path(),
            1 => frontier.inline_children = children,
            _ => frontier.branch_children = children,
        }
        frontier
    }

    pub(crate) fn begin_component(&self, root: AtomId) -> Self {
        let frontier = self.structural_frontier();
        assert!(!frontier.is_contradiction());
        assert!(
            frontier.component_roots().contains(&root),
            "component root must be advertised by the structural frontier"
        );
        let mut successor = self.clone();
        successor
            .traversal
            .begin_component(self.prepared.graph(), root);
        successor
    }

    /// Commit one advertised child incidence as the traversal entry of its
    /// residual attachment without entering the child atom yet.
    pub(crate) fn commit_traversal_edge(&self, incident: AdjacentBond) -> Result<Self, S::Error> {
        let frontier = self.structural_frontier();
        assert!(!frontier.is_contradiction());
        assert!(
            frontier.branch_children().contains(&incident)
                || frontier.inline_children().contains(&incident),
            "a traversal commitment requires an advertised child"
        );

        let constraints = self.restricted_role(incident.bond(), BondRole::Traversal)?;
        Ok(Self {
            prepared: self.prepared.clone(),
            traversal: self.traversal.clone(),
            constraints,
        })
    }

    pub(crate) fn enter_inline_child(&self, incident: AdjacentBond) -> Self {
        assert_eq!(
            self.bond_role_domain(incident.bond()),
            BondRole::Traversal.singleton_domain(),
            "an inline child must already be committed to Traversal"
        );
        assert_eq!(
            self.structural_frontier().inline_children(),
            &[incident],
            "the sole remaining residual attachment is the inline continuation"
        );
        let mut successor = self.clone();
        successor
            .traversal
            .enter_inline_child(self.prepared.graph(), incident);
        successor
    }

    pub(crate) fn enter_branch_child(&self, incident: AdjacentBond) -> Self {
        assert_eq!(
            self.bond_role_domain(incident.bond()),
            BondRole::Traversal.singleton_domain(),
            "a branch child must already be committed to Traversal"
        );
        assert!(
            self.structural_frontier()
                .branch_children()
                .contains(&incident),
            "a branch child requires another residual attachment"
        );
        let mut successor = self.clone();
        successor
            .traversal
            .enter_branch_child(self.prepared.graph(), incident);
        successor
    }

    pub(crate) fn open_ring_endpoint(&self, incident: AdjacentBond) -> Result<Self, S::Error> {
        let frontier = self.structural_frontier();
        assert!(!frontier.is_contradiction());
        assert!(
            frontier.ring_openings().contains(&incident),
            "a ring opening must be advertised by the residual-attachment frontier"
        );

        let constraints = self.restricted_role(incident.bond(), BondRole::Ring)?;
        let mut traversal = self.traversal.clone();
        traversal.open_ring_endpoint(self.prepared.graph(), incident);
        Ok(Self {
            prepared: self.prepared.clone(),
            traversal,
            constraints,
        })
    }

    pub(crate) fn close_ring_endpoint(&self, incident: AdjacentBond) -> Self {
        let frontier = self.structural_frontier();
        assert!(!frontier.is_contradiction());
        assert!(
            frontier.ring_closures().contains(&incident),
            "a ring closure must be advertised by the structural frontier"
        );
        let mut successor = self.clone();
        successor
            .traversal
            .close_ring_endpoint(self.prepared.graph(), incident);
        successor
    }

    pub(crate) fn complete_path(&self) -> Self {
        let frontier = self.structural_frontier();
        assert!(!frontier.is_contradiction());
        assert!(
            frontier.can_complete_path(),
            "the active path can complete only after all residual attachments and ring work"
        );
        let mut successor = self.clone();
        successor.traversal.complete_path();
        successor
    }

    fn restricted_role(&self, bond: BondId, role: BondRole) -> Result<S, S::Error> {
        let domain = role.singleton_domain();
        if self.bond_role_domain(bond) == domain {
            return Ok(self.constraints.clone());
        }
        self.constraints
            .restricted(&[(role_variable(&self.prepared, bond), domain)])
    }
}

fn role_variable(prepared: &PreparedMolecule, bond: BondId) -> VariableId {
    prepared
        .bond_role_variable(bond)
        .expect("prepared bond must have a role variable")
}

#[cfg(test)]
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
    use crate::native::NativeSolverState;
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
    fn triangle_requires_one_ring_before_its_single_attachment_entry() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let left = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let right = graph.add_bond(atoms[0], atoms[2]).unwrap();
        let between = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(atoms[0]);
        let left_incident = incident(&prepared, atoms[0], left);
        let right_incident = incident(&prepared, atoms[0], right);

        let frontier = rooted.structural_frontier();
        assert_eq!(frontier.ring_openings(), &[left_incident, right_incident]);
        assert!(frontier.branch_children().is_empty());
        assert!(frontier.inline_children().is_empty());

        let opened = rooted.open_ring_endpoint(left_incident).unwrap();
        assert_eq!(
            rooted.bond_role_domain(left),
            BondRole::role_domain(),
            "the source state must remain unchanged"
        );
        let opened_frontier = opened.structural_frontier();
        assert!(opened_frontier.ring_openings().is_empty());
        assert_eq!(opened_frontier.inline_children(), &[right_incident]);
        assert_eq!(
            opened.bond_role_domain(left),
            BondRole::Ring.singleton_domain()
        );
        assert_eq!(
            opened.bond_role_domain(right),
            BondRole::Traversal.singleton_domain()
        );
        assert_eq!(
            opened.bond_role_domain(between),
            BondRole::Traversal.singleton_domain()
        );

        let committed = opened.commit_traversal_edge(right_incident).unwrap();
        let walked = committed.enter_inline_child(right_incident);
        let between_incident = incident(&prepared, atoms[2], between);
        let committed = walked.commit_traversal_edge(between_incident).unwrap();
        let walked = committed.enter_inline_child(between_incident);
        let closing = incident(&prepared, atoms[1], left);
        assert_eq!(walked.structural_frontier().ring_closures(), &[closing]);
        assert_eq!(walked.ring_closure_first_endpoint(closing), atoms[0]);

        let closed = walked.close_ring_endpoint(closing);
        assert!(closed.graph_is_complete());
        let finished = closed.complete_path();
        assert_eq!(finished.active_atom(), None);
    }

    #[test]
    fn a_matroid_basis_can_still_contradict_writer_policy() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let left = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let right = graph.add_bond(atoms[0], atoms[2]).unwrap();
        let between = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(atoms[0]);
        let constraints = rooted
            .constraints
            .restricted(&[
                role_restriction(&prepared, left, BondRole::Traversal),
                role_restriction(&prepared, right, BondRole::Traversal),
                role_restriction(&prepared, between, BondRole::Ring),
            ])
            .unwrap();
        let basis = WriterState {
            prepared: rooted.prepared.clone(),
            traversal: rooted.traversal.clone(),
            constraints,
        };

        let frontier = basis.structural_frontier();
        assert!(frontier.is_contradiction());
        assert!(frontier.ring_openings().is_empty());
        assert!(frontier.branch_children().is_empty());
        assert!(frontier.inline_children().is_empty());
    }

    #[test]
    fn star_attachments_are_branches_then_one_inline_continuation() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 4] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let bonds = [
            graph.add_bond(atoms[0], atoms[1]).unwrap(),
            graph.add_bond(atoms[0], atoms[2]).unwrap(),
            graph.add_bond(atoms[0], atoms[3]).unwrap(),
        ];
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(atoms[0]);
        let children = bonds.map(|bond| incident(&prepared, atoms[0], bond));

        assert_eq!(rooted.structural_frontier().branch_children(), &children);
        let committed = rooted.commit_traversal_edge(children[0]).unwrap();
        let branch = committed.enter_branch_child(children[0]);
        let restored = branch.complete_path();

        assert_eq!(
            restored.structural_frontier().branch_children(),
            &children[1..]
        );
        let committed = restored.commit_traversal_edge(children[1]).unwrap();
        let branch = committed.enter_branch_child(children[1]);
        let restored = branch.complete_path();
        assert_eq!(
            restored.structural_frontier().inline_children(),
            &[children[2]]
        );
    }

    #[test]
    fn ring_neighbours_share_an_attachment_but_a_substituent_does_not() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 4] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let left = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let right = graph.add_bond(atoms[0], atoms[2]).unwrap();
        graph.add_bond(atoms[1], atoms[2]).unwrap();
        let substituent = graph.add_bond(atoms[0], atoms[3]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(atoms[0]);

        let frontier = rooted.structural_frontier();
        assert_eq!(
            frontier.ring_openings(),
            &[
                incident(&prepared, atoms[0], left),
                incident(&prepared, atoms[0], right),
            ]
        );
        assert!(frontier.branch_children().is_empty());

        let opened = rooted
            .open_ring_endpoint(incident(&prepared, atoms[0], left))
            .unwrap();
        assert_eq!(
            opened.structural_frontier().branch_children(),
            &[
                incident(&prepared, atoms[0], right),
                incident(&prepared, atoms[0], substituent),
            ]
        );
    }

    #[test]
    fn fused_residual_system_remains_one_attachment() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 5] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let root_edges = [
            graph.add_bond(atoms[0], atoms[1]).unwrap(),
            graph.add_bond(atoms[0], atoms[2]).unwrap(),
            graph.add_bond(atoms[0], atoms[3]).unwrap(),
        ];
        graph.add_bond(atoms[1], atoms[4]).unwrap();
        graph.add_bond(atoms[2], atoms[4]).unwrap();
        graph.add_bond(atoms[3], atoms[4]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(atoms[0]);
        let openings = root_edges.map(|bond| incident(&prepared, atoms[0], bond));

        assert_eq!(rooted.structural_frontier().ring_openings(), &openings);
        assert!(rooted.structural_frontier().branch_children().is_empty());
    }
}
