//! Concrete composition of residual writer frames and constraint state.
//!
//! The graphic-matroid factor guards global Traversal/Ring feasibility. Writer
//! policy is stricter: each residual attachment receives exactly one traversal
//! entry, while its other active incidences become ring endpoints.

use crate::domain::Domain;
use crate::ids::{AtomId, BondId, VariableId};
use crate::model::BondRole;
use crate::prepared::{AdjacentBond, PreparedMolecule};
use crate::solver::{Consistency, ConstraintSolver};
use crate::traversal::{IncidentBondState, TraversalState};

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(test)]
use std::sync::Arc;

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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum StructuralCandidate {
    Root {
        atom: AtomId,
    },
    RingOpen {
        incident: AdjacentBond,
    },
    RingClose {
        incident: AdjacentBond,
        first_endpoint: AtomId,
    },
    BranchChild {
        incident: AdjacentBond,
    },
    InlineChild {
        incident: AdjacentBond,
    },
    CompletePath,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct StructuralCandidateBatch {
    candidates: Vec<StructuralCandidate>,
    contradiction: bool,
}

impl StructuralCandidateBatch {
    pub(crate) fn candidates(&self) -> &[StructuralCandidate] {
        &self.candidates
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
    #[cfg(test)]
    candidate_batch_derivations: Arc<AtomicUsize>,
}

impl<S: ConstraintSolver> WriterState<S> {
    pub(crate) fn initial(prepared: &PreparedMolecule) -> Result<Consistency<Self>, S::Failure> {
        Ok(
            S::initial(prepared.constraint_model_arc())?.map(|constraints| {
                assert_initial_solver_shape(prepared, &constraints);
                Self {
                    prepared: prepared.clone(),
                    traversal: TraversalState::new(prepared.graph()),
                    constraints,
                    #[cfg(test)]
                    candidate_batch_derivations: Arc::new(AtomicUsize::new(0)),
                }
            }),
        )
    }

    pub(crate) fn active_atom(&self) -> Option<AtomId> {
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
        #[cfg(test)]
        self.candidate_batch_derivations
            .fetch_add(1, Ordering::Relaxed);

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

    pub(crate) fn derive_candidates(&self) -> StructuralCandidateBatch {
        let frontier = self.structural_frontier();
        if frontier.is_contradiction() {
            return StructuralCandidateBatch {
                candidates: Vec::new(),
                contradiction: true,
            };
        }

        let mut candidates = Vec::new();
        candidates.extend(
            frontier
                .component_roots()
                .iter()
                .copied()
                .map(|atom| StructuralCandidate::Root { atom }),
        );
        candidates.extend(frontier.ring_closures().iter().copied().map(|incident| {
            let first_endpoint = self
                .traversal
                .ring_first_endpoint_for_active_incident(self.prepared.graph(), incident)
                .expect("an advertised ring closure must retain its first endpoint");
            StructuralCandidate::RingClose {
                incident,
                first_endpoint,
            }
        }));
        candidates.extend(
            frontier
                .ring_openings()
                .iter()
                .copied()
                .map(|incident| StructuralCandidate::RingOpen { incident }),
        );
        candidates.extend(
            frontier
                .branch_children()
                .iter()
                .copied()
                .map(|incident| StructuralCandidate::BranchChild { incident }),
        );
        candidates.extend(
            frontier
                .inline_children()
                .iter()
                .copied()
                .map(|incident| StructuralCandidate::InlineChild { incident }),
        );
        if frontier.can_complete_path() {
            candidates.push(StructuralCandidate::CompletePath);
        }

        StructuralCandidateBatch {
            candidates,
            contradiction: false,
        }
    }

    pub(crate) fn attempt_candidate(
        &self,
        candidate: StructuralCandidate,
    ) -> Result<Consistency<Self>, S::Failure> {
        let attempted = match candidate {
            StructuralCandidate::Root { atom } => {
                let mut successor = self.clone();
                successor
                    .traversal
                    .begin_component(self.prepared.graph(), atom);
                Ok(Consistency::Consistent(successor))
            }
            StructuralCandidate::RingOpen { incident } => Ok(self
                .restricted_role(incident.bond(), BondRole::Ring)?
                .map(|constraints| {
                    let mut traversal = self.traversal.clone();
                    traversal.open_ring_endpoint(self.prepared.graph(), incident);
                    Self {
                        prepared: self.prepared.clone(),
                        traversal,
                        constraints,
                        #[cfg(test)]
                        candidate_batch_derivations: self.candidate_batch_derivations.clone(),
                    }
                })),
            StructuralCandidate::RingClose {
                incident,
                first_endpoint,
            } => {
                assert_eq!(
                    self.traversal
                        .ring_first_endpoint_for_active_incident(self.prepared.graph(), incident),
                    Some(first_endpoint),
                    "a ring-closure candidate must retain its source-local first endpoint"
                );
                let mut successor = self.clone();
                successor
                    .traversal
                    .close_ring_endpoint(self.prepared.graph(), incident);
                Ok(Consistency::Consistent(successor))
            }
            StructuralCandidate::BranchChild { incident }
            | StructuralCandidate::InlineChild { incident } => Ok(self
                .restricted_role(incident.bond(), BondRole::Traversal)?
                .map(|constraints| Self {
                    prepared: self.prepared.clone(),
                    traversal: self.traversal.clone(),
                    constraints,
                    #[cfg(test)]
                    candidate_batch_derivations: self.candidate_batch_derivations.clone(),
                })),
            StructuralCandidate::CompletePath => {
                let mut successor = self.clone();
                successor.traversal.complete_path(self.prepared.graph());
                Ok(Consistency::Consistent(successor))
            }
        };
        #[cfg(test)]
        {
            attempted.map(|consistency| {
                consistency.map(|mut successor| {
                    successor.candidate_batch_derivations = Arc::new(AtomicUsize::new(0));
                    successor
                })
            })
        }
        #[cfg(not(test))]
        {
            attempted
        }
    }

    #[cfg(test)]
    pub(crate) fn candidate_batch_derivation_count(&self) -> usize {
        self.candidate_batch_derivations.load(Ordering::Relaxed)
    }

    pub(crate) fn enter_committed_inline_child(&self, incident: AdjacentBond) -> Self {
        assert_eq!(
            self.bond_role_domain(incident.bond()),
            BondRole::Traversal.singleton_domain(),
            "an inline child must already be committed to Traversal"
        );
        let mut successor = self.clone();
        successor
            .traversal
            .enter_inline_child(self.prepared.graph(), incident);
        successor
    }

    pub(crate) fn enter_committed_branch_child(&self, incident: AdjacentBond) -> Self {
        assert_eq!(
            self.bond_role_domain(incident.bond()),
            BondRole::Traversal.singleton_domain(),
            "a branch child must already be committed to Traversal"
        );
        let mut successor = self.clone();
        successor
            .traversal
            .enter_branch_child(self.prepared.graph(), incident);
        successor
    }

    fn restricted_role(&self, bond: BondId, role: BondRole) -> Result<Consistency<S>, S::Failure> {
        let domain = role.singleton_domain();
        if self.bond_role_domain(bond) == domain {
            return Ok(Consistency::Consistent(self.constraints.clone()));
        }
        self.constraints
            .restricted(&[(role_variable(&self.prepared, bond), domain)])
    }
}

fn assert_initial_solver_shape<S: ConstraintSolver>(prepared: &PreparedMolecule, constraints: &S) {
    for index in 0..prepared.constraint_model().variable_count() {
        let variable = VariableId::new(
            u32::try_from(index).expect("prepared variable count must fit the identifier space"),
        );
        let initial = prepared
            .constraint_model()
            .variable(variable)
            .expect("prepared variable index must resolve")
            .initial_domain();
        let current = constraints
            .domain(variable)
            .expect("a solver state must retain every prepared variable");
        assert!(
            !current.is_empty() && current.is_subset_of(initial),
            "a solver state domain must be nonempty and refine its prepared domain"
        );
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
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;
    use crate::native::NativeSolverState;
    use crate::prepared::PreparedGraphBuilder;

    #[derive(Clone)]
    struct MissingDomainSolver;

    impl ConstraintSolver for MissingDomainSolver {
        type Failure = Infallible;

        fn initial(
            _model: Arc<crate::model::ConstraintModel>,
        ) -> Result<Consistency<Self>, Self::Failure> {
            Ok(Consistency::Consistent(Self))
        }

        fn restricted(
            &self,
            _restrictions: &[(VariableId, Domain)],
        ) -> Result<Consistency<Self>, Self::Failure> {
            Ok(Consistency::Consistent(self.clone()))
        }

        fn domain(&self, _variable: VariableId) -> Option<Domain> {
            None
        }
    }

    #[derive(Clone)]
    struct DomainCountingSolver {
        native: NativeSolverState,
        domain_reads: Arc<AtomicUsize>,
    }

    impl ConstraintSolver for DomainCountingSolver {
        type Failure = crate::native_solver::NativeSolverFailure;

        fn initial(
            model: Arc<crate::model::ConstraintModel>,
        ) -> Result<Consistency<Self>, Self::Failure> {
            Ok(
                <NativeSolverState as ConstraintSolver>::initial(model)?.map(|native| Self {
                    native,
                    domain_reads: Arc::new(AtomicUsize::new(0)),
                }),
            )
        }

        fn restricted(
            &self,
            restrictions: &[(VariableId, Domain)],
        ) -> Result<Consistency<Self>, Self::Failure> {
            Ok(
                <NativeSolverState as ConstraintSolver>::restricted(&self.native, restrictions)?
                    .map(|native| Self {
                        native,
                        domain_reads: Arc::clone(&self.domain_reads),
                    }),
            )
        }

        fn domain(&self, variable: VariableId) -> Option<Domain> {
            self.domain_reads.fetch_add(1, Ordering::Relaxed);
            self.native.domain(variable)
        }
    }

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

    fn attempt(
        state: &WriterState<NativeSolverState>,
        candidate: StructuralCandidate,
    ) -> WriterState<NativeSolverState> {
        state
            .attempt_candidate(candidate)
            .unwrap()
            .unwrap_consistent()
    }

    fn rooted(prepared: &PreparedMolecule, atom: AtomId) -> WriterState<NativeSolverState> {
        let initial = WriterState::<NativeSolverState>::initial(prepared)
            .unwrap()
            .unwrap_consistent();
        assert!(initial
            .derive_candidates()
            .candidates()
            .contains(&StructuralCandidate::Root { atom }));
        attempt(&initial, StructuralCandidate::Root { atom })
    }

    #[test]
    #[should_panic(expected = "a solver state must retain every prepared variable")]
    fn initial_state_checks_the_solver_domain_shape_at_the_boundary() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 2] = std::array::from_fn(|_| graph.add_atom().unwrap());
        graph.add_bond(atoms[0], atoms[1]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());

        let _ = WriterState::<MissingDomainSolver>::initial(&prepared);
    }

    #[test]
    fn role_restriction_does_not_rescan_the_solver_domain_shape() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let left = graph.add_bond(atoms[0], atoms[1]).unwrap();
        graph.add_bond(atoms[0], atoms[2]).unwrap();
        graph.add_bond(atoms[1], atoms[2]).unwrap();
        for _ in 0..128 {
            let a = graph.add_atom().unwrap();
            let b = graph.add_atom().unwrap();
            graph.add_bond(a, b).unwrap();
        }
        let prepared = PreparedMolecule::new(graph.build());
        assert_eq!(prepared.constraint_model().variable_count(), 131);
        let state = WriterState::<DomainCountingSolver>::initial(&prepared)
            .unwrap()
            .unwrap_consistent();
        state.constraints.domain_reads.store(0, Ordering::Relaxed);

        let _ = state
            .restricted_role(left, BondRole::Ring)
            .unwrap()
            .unwrap_consistent();

        assert_eq!(state.constraints.domain_reads.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn triangle_requires_one_ring_before_its_single_attachment_entry() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let left = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let right = graph.add_bond(atoms[0], atoms[2]).unwrap();
        let between = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = rooted(&prepared, atoms[0]);
        let left_incident = incident(&prepared, atoms[0], left);
        let right_incident = incident(&prepared, atoms[0], right);

        let frontier = rooted.structural_frontier();
        assert_eq!(frontier.ring_openings(), &[left_incident, right_incident]);
        assert!(frontier.branch_children().is_empty());
        assert!(frontier.inline_children().is_empty());

        let opened = attempt(
            &rooted,
            StructuralCandidate::RingOpen {
                incident: left_incident,
            },
        );
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

        let committed = attempt(
            &opened,
            StructuralCandidate::InlineChild {
                incident: right_incident,
            },
        );
        let walked = committed.enter_committed_inline_child(right_incident);
        let between_incident = incident(&prepared, atoms[2], between);
        let committed = attempt(
            &walked,
            StructuralCandidate::InlineChild {
                incident: between_incident,
            },
        );
        let walked = committed.enter_committed_inline_child(between_incident);
        let closing = incident(&prepared, atoms[1], left);
        assert_eq!(walked.structural_frontier().ring_closures(), &[closing]);
        let closing_candidate = StructuralCandidate::RingClose {
            incident: closing,
            first_endpoint: atoms[0],
        };
        assert!(walked
            .derive_candidates()
            .candidates()
            .contains(&closing_candidate));

        let closed = attempt(&walked, closing_candidate);
        assert!(closed.graph_is_complete());
        let finished = attempt(&closed, StructuralCandidate::CompletePath);
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
        let rooted = rooted(&prepared, atoms[0]);
        let constraints = rooted
            .constraints
            .restricted(&[
                role_restriction(&prepared, left, BondRole::Traversal),
                role_restriction(&prepared, right, BondRole::Traversal),
                role_restriction(&prepared, between, BondRole::Ring),
            ])
            .unwrap()
            .unwrap_consistent();
        let basis = WriterState {
            prepared: rooted.prepared.clone(),
            traversal: rooted.traversal.clone(),
            constraints,
            candidate_batch_derivations: rooted.candidate_batch_derivations.clone(),
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
        let rooted = rooted(&prepared, atoms[0]);
        let children = bonds.map(|bond| incident(&prepared, atoms[0], bond));

        assert_eq!(rooted.structural_frontier().branch_children(), &children);
        let committed = attempt(
            &rooted,
            StructuralCandidate::BranchChild {
                incident: children[0],
            },
        );
        let branch = committed.enter_committed_branch_child(children[0]);
        let restored = attempt(&branch, StructuralCandidate::CompletePath);

        assert_eq!(
            restored.structural_frontier().branch_children(),
            &children[1..]
        );
        let committed = attempt(
            &restored,
            StructuralCandidate::BranchChild {
                incident: children[1],
            },
        );
        let branch = committed.enter_committed_branch_child(children[1]);
        let restored = attempt(&branch, StructuralCandidate::CompletePath);
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
        let rooted = rooted(&prepared, atoms[0]);

        let frontier = rooted.structural_frontier();
        assert_eq!(
            frontier.ring_openings(),
            &[
                incident(&prepared, atoms[0], left),
                incident(&prepared, atoms[0], right),
            ]
        );
        assert!(frontier.branch_children().is_empty());

        let opened = attempt(
            &rooted,
            StructuralCandidate::RingOpen {
                incident: incident(&prepared, atoms[0], left),
            },
        );
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
        let rooted = rooted(&prepared, atoms[0]);
        let openings = root_edges.map(|bond| incident(&prepared, atoms[0], bond));

        assert_eq!(rooted.structural_frontier().ring_openings(), &openings);
        assert!(rooted.structural_frontier().branch_children().is_empty());
    }
}
