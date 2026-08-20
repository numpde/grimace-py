//! Concrete composition of residual writer frames and constraint state.
//!
//! The graphic-matroid factor guards global Traversal/Ring feasibility. Writer
//! policy is stricter: each residual attachment receives exactly one traversal
//! entry, while its other active incidences become ring endpoints.

use std::error::Error;
use std::fmt;

use crate::domain::Domain;
use crate::ids::{AtomId, BondId, VariableId};
use crate::model::BondRole;
use crate::prepared::{AdjacentBond, PreparedMolecule};
use crate::solver::ConstraintSolver;
use crate::traversal::{IncidentBondState, TraversalState};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum WriterContradiction {
    UnrepresentedVisitedEndpoint {
        active: AtomId,
        incident: AdjacentBond,
    },
    ResidualAttachmentHasNoTraversalEntry {
        active: AtomId,
        incidences: Box<[AdjacentBond]>,
    },
    ResidualAttachmentCannotChooseRing {
        active: AtomId,
        incidences: Box<[AdjacentBond]>,
    },
    ActivePathCannotComplete {
        active: AtomId,
    },
}

impl fmt::Display for WriterContradiction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnrepresentedVisitedEndpoint { active, incident } => write!(
                formatter,
                "active atom {active:?} has an unrepresented bond to visited atom {:?}",
                incident.atom()
            ),
            Self::ResidualAttachmentHasNoTraversalEntry {
                active,
                incidences,
            } => write!(
                formatter,
                "active atom {active:?} has a residual attachment without a Traversal-capable entry: {incidences:?}"
            ),
            Self::ResidualAttachmentCannotChooseRing {
                active,
                incidences,
            } => write!(
                formatter,
                "active atom {active:?} cannot reduce a residual attachment to one entry: {incidences:?}"
            ),
            Self::ActivePathCannotComplete { active } => write!(
                formatter,
                "active atom {active:?} has no writer action and cannot complete its path"
            ),
        }
    }
}

impl Error for WriterContradiction {}

#[derive(Debug)]
pub(crate) enum TransitionError<E> {
    Constraint(E),
    Writer(WriterContradiction),
}

impl<E: fmt::Display> fmt::Display for TransitionError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constraint(error) => write!(formatter, "constraint transition failed: {error}"),
            Self::Writer(error) => write!(formatter, "writer transition contradicted: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for TransitionError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Constraint(error) => Some(error),
            Self::Writer(error) => Some(error),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum StructuralFrontier {
    ComponentRoots(Box<[AtomId]>),
    RingSuffix {
        openings: Box<[AdjacentBond]>,
        closures: Box<[AdjacentBond]>,
    },
    BranchChildren(Box<[AdjacentBond]>),
    InlineChild(AdjacentBond),
    CompletePath,
    Terminal,
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

    pub(crate) fn ring_closure_first_endpoint(
        &self,
        incident: AdjacentBond,
    ) -> Result<AtomId, WriterContradiction> {
        match self.structural_frontier()? {
            StructuralFrontier::RingSuffix { closures, .. }
                if closures.contains(&incident) => {}
            _ => panic!("ring-closure facts require an advertised closure"),
        }
        Ok(self
            .traversal
            .ring_first_endpoint_for_active_incident(self.prepared.graph(), incident)
            .expect("an advertised ring closure must retain its first endpoint"))
    }

    fn bond_role_domain(&self, bond: BondId) -> Domain {
        let variable = role_variable(&self.prepared, bond);
        self.constraints
            .domain(variable)
            .expect("prepared bond role must belong to the writer constraint model")
    }

    pub(crate) fn structural_frontier(
        &self,
    ) -> Result<StructuralFrontier, WriterContradiction> {
        let graph = self.prepared.graph();
        let Some(active) = self.traversal.active_atom() else {
            let roots = self.traversal.unvisited_atoms(graph).collect::<Vec<_>>();
            return if roots.is_empty() {
                Ok(StructuralFrontier::Terminal)
            } else {
                Ok(StructuralFrontier::ComponentRoots(
                    roots.into_boxed_slice(),
                ))
            };
        };

        let mut ring_phase = false;
        let mut openings = Vec::new();
        let mut closures = Vec::new();
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
                    closures.push(incident);
                }
                IncidentBondState::UnrepresentedToVisitedAtom => {
                    return Err(WriterContradiction::UnrepresentedVisitedEndpoint {
                        active,
                        incident,
                    });
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
                return Err(WriterContradiction::ResidualAttachmentHasNoTraversalEntry {
                    active,
                    incidences: incidences.to_vec().into_boxed_slice(),
                });
            }

            if incidences.len() == 1 {
                children.push(incidences[0]);
                continue;
            }

            ring_phase = true;
            let opening_count = openings.len();
            for (candidate, domain) in role_domains {
                if !domain.contains(BondRole::Ring.value_index()) {
                    continue;
                }
                let candidate_can_traverse = domain.contains(BondRole::Traversal.value_index());
                if traversal_capable_count > usize::from(candidate_can_traverse) {
                    openings.push(candidate);
                }
            }
            if openings.len() == opening_count {
                return Err(WriterContradiction::ResidualAttachmentCannotChooseRing {
                    active,
                    incidences: incidences.to_vec().into_boxed_slice(),
                });
            }
        }

        if ring_phase {
            return Ok(StructuralFrontier::RingSuffix {
                openings: openings.into_boxed_slice(),
                closures: closures.into_boxed_slice(),
            });
        }

        match children.len() {
            0 if self.traversal.can_complete_path() => Ok(StructuralFrontier::CompletePath),
            0 => Err(WriterContradiction::ActivePathCannotComplete { active }),
            1 => Ok(StructuralFrontier::InlineChild(children[0])),
            _ => Ok(StructuralFrontier::BranchChildren(
                children.into_boxed_slice(),
            )),
        }
    }

    pub(crate) fn begin_component(
        &self,
        root: AtomId,
    ) -> Result<Self, WriterContradiction> {
        match self.structural_frontier()? {
            StructuralFrontier::ComponentRoots(roots) if roots.contains(&root) => {}
            _ => panic!("component root must be advertised by the structural frontier"),
        }
        let mut successor = self.clone();
        successor
            .traversal
            .begin_component(self.prepared.graph(), root);
        successor.checked()
    }

    /// Commit one advertised child incidence as the traversal entry of its
    /// residual attachment without entering the child atom yet.
    pub(crate) fn commit_traversal_edge(
        &self,
        incident: AdjacentBond,
    ) -> Result<Self, TransitionError<S::Error>> {
        match self
            .structural_frontier()
            .map_err(TransitionError::Writer)?
        {
            StructuralFrontier::BranchChildren(children) if children.contains(&incident) => {}
            StructuralFrontier::InlineChild(child) if child == incident => {}
            _ => panic!("a traversal commitment requires an advertised child"),
        }

        let constraints = self
            .restricted_role(incident.bond(), BondRole::Traversal)
            .map_err(TransitionError::Constraint)?;
        Self {
            prepared: self.prepared.clone(),
            traversal: self.traversal.clone(),
            constraints,
        }
        .checked()
        .map_err(TransitionError::Writer)
    }

    pub(crate) fn enter_inline_child(
        &self,
        incident: AdjacentBond,
    ) -> Result<Self, WriterContradiction> {
        match self.structural_frontier()? {
            StructuralFrontier::InlineChild(child) if child == incident => {}
            _ => panic!("the inline child must be the sole advertised child"),
        }
        assert_eq!(
            self.bond_role_domain(incident.bond()),
            BondRole::Traversal.singleton_domain(),
            "an inline child must already be committed to Traversal"
        );

        let mut successor = self.clone();
        successor
            .traversal
            .enter_inline_child(self.prepared.graph(), incident);
        successor.checked()
    }

    pub(crate) fn enter_branch_child(
        &self,
        incident: AdjacentBond,
    ) -> Result<Self, WriterContradiction> {
        match self.structural_frontier()? {
            StructuralFrontier::BranchChildren(children) if children.contains(&incident) => {}
            _ => panic!("a branch child requires another residual attachment"),
        }
        assert_eq!(
            self.bond_role_domain(incident.bond()),
            BondRole::Traversal.singleton_domain(),
            "a branch child must already be committed to Traversal"
        );

        let mut successor = self.clone();
        successor
            .traversal
            .enter_branch_child(self.prepared.graph(), incident);
        successor.checked()
    }

    pub(crate) fn open_ring_endpoint(
        &self,
        incident: AdjacentBond,
    ) -> Result<Self, TransitionError<S::Error>> {
        match self
            .structural_frontier()
            .map_err(TransitionError::Writer)?
        {
            StructuralFrontier::RingSuffix { openings, .. }
                if openings.contains(&incident) => {}
            _ => panic!("a ring opening must be advertised by the structural frontier"),
        }

        let constraints = self
            .restricted_role(incident.bond(), BondRole::Ring)
            .map_err(TransitionError::Constraint)?;
        let mut traversal = self.traversal.clone();
        traversal.open_ring_endpoint(self.prepared.graph(), incident);
        Self {
            prepared: self.prepared.clone(),
            traversal,
            constraints,
        }
        .checked()
        .map_err(TransitionError::Writer)
    }

    pub(crate) fn close_ring_endpoint(
        &self,
        incident: AdjacentBond,
    ) -> Result<Self, WriterContradiction> {
        match self.structural_frontier()? {
            StructuralFrontier::RingSuffix { closures, .. }
                if closures.contains(&incident) => {}
            _ => panic!("a ring closure must be advertised by the structural frontier"),
        }
        let mut successor = self.clone();
        successor
            .traversal
            .close_ring_endpoint(self.prepared.graph(), incident);
        successor.checked()
    }

    pub(crate) fn complete_path(&self) -> Result<Self, WriterContradiction> {
        match self.structural_frontier()? {
            StructuralFrontier::CompletePath => {}
            _ => panic!("the active path is not ready to complete"),
        }
        let mut successor = self.clone();
        successor.traversal.complete_path();
        successor.checked()
    }

    /// Validate only the immediately constructed writer state. This is not a
    /// support-enumeration proof that every visible prefix has a complete walk.
    fn checked(self) -> Result<Self, WriterContradiction> {
        self.structural_frontier()?;
        Ok(self)
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

    fn frontier(state: &WriterState<NativeSolverState>) -> StructuralFrontier {
        state.structural_frontier().unwrap()
    }

    #[test]
    fn triangle_requires_one_ring_before_its_attachment_entry() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let left = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let right = graph.add_bond(atoms[0], atoms[2]).unwrap();
        let between = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let rooted = WriterState::<NativeSolverState>::initial(&prepared)
            .unwrap()
            .begin_component(atoms[0])
            .unwrap();
        let left_incident = incident(&prepared, atoms[0], left);
        let right_incident = incident(&prepared, atoms[0], right);

        assert_eq!(
            frontier(&rooted),
            StructuralFrontier::RingSuffix {
                openings: vec![left_incident, right_incident].into_boxed_slice(),
                closures: Vec::new().into_boxed_slice(),
            }
        );

        let opened = rooted.open_ring_endpoint(left_incident).unwrap();
        assert_eq!(
            rooted.bond_role_domain(left),
            BondRole::role_domain(),
            "the source state must remain unchanged"
        );
        assert_eq!(frontier(&opened), StructuralFrontier::InlineChild(right_incident));
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
        let walked = committed.enter_inline_child(right_incident).unwrap();
        let between_incident = incident(&prepared, atoms[2], between);
        let committed = walked.commit_traversal_edge(between_incident).unwrap();
        let walked = committed.enter_inline_child(between_incident).unwrap();
        let closing = incident(&prepared, atoms[1], left);
        assert_eq!(
            frontier(&walked),
            StructuralFrontier::RingSuffix {
                openings: Vec::new().into_boxed_slice(),
                closures: vec![closing].into_boxed_slice(),
            }
        );
        assert_eq!(
            walked.ring_closure_first_endpoint(closing).unwrap(),
            atoms[0]
        );

        let closed = walked.close_ring_endpoint(closing).unwrap();
        assert!(closed.graph_is_complete());
        assert_eq!(frontier(&closed), StructuralFrontier::CompletePath);
        let finished = closed.complete_path().unwrap();
        assert_eq!(finished.active_atom(), None);
        assert_eq!(frontier(&finished), StructuralFrontier::Terminal);
    }

    #[test]
    fn a_matroid_basis_can_contradict_writer_policy() {
        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 3] = std::array::from_fn(|_| graph.add_atom().unwrap());
        let left = graph.add_bond(atoms[0], atoms[1]).unwrap();
        let right = graph.add_bond(atoms[0], atoms[2]).unwrap();
        let between = graph.add_bond(atoms[1], atoms[2]).unwrap();
        let prepared = PreparedMolecule::new(graph.build());
        let initial = WriterState::<NativeSolverState>::initial(&prepared).unwrap();
        let rooted = initial.begin_component(atoms[0]).unwrap();
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
            constraints: constraints.clone(),
        };

        assert!(matches!(
            basis.structural_frontier(),
            Err(WriterContradiction::ResidualAttachmentCannotChooseRing { .. })
        ));

        let inactive_basis = WriterState {
            prepared: initial.prepared.clone(),
            traversal: initial.traversal.clone(),
            constraints,
        };
        assert!(matches!(
            inactive_basis.begin_component(atoms[0]),
            Err(WriterContradiction::ResidualAttachmentCannotChooseRing { .. })
        ));
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
            .begin_component(atoms[0])
            .unwrap();
        let children = bonds.map(|bond| incident(&prepared, atoms[0], bond));

        assert_eq!(
            frontier(&rooted),
            StructuralFrontier::BranchChildren(children.to_vec().into_boxed_slice())
        );
        let committed = rooted.commit_traversal_edge(children[0]).unwrap();
        let branch = committed.enter_branch_child(children[0]).unwrap();
        let restored = branch.complete_path().unwrap();
        assert_eq!(
            frontier(&restored),
            StructuralFrontier::BranchChildren(children[1..].to_vec().into_boxed_slice())
        );

        let committed = restored.commit_traversal_edge(children[1]).unwrap();
        let branch = committed.enter_branch_child(children[1]).unwrap();
        let restored = branch.complete_path().unwrap();
        assert_eq!(
            frontier(&restored),
            StructuralFrontier::InlineChild(children[2])
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
            .begin_component(atoms[0])
            .unwrap();

        let opened = rooted
            .open_ring_endpoint(incident(&prepared, atoms[0], left))
            .unwrap();
        assert_eq!(
            frontier(&opened),
            StructuralFrontier::BranchChildren(
                vec![
                    incident(&prepared, atoms[0], right),
                    incident(&prepared, atoms[0], substituent),
                ]
                .into_boxed_slice()
            )
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
            .begin_component(atoms[0])
            .unwrap();
        let openings = root_edges.map(|bond| incident(&prepared, atoms[0], bond));

        assert_eq!(
            frontier(&rooted),
            StructuralFrontier::RingSuffix {
                openings: openings.to_vec().into_boxed_slice(),
                closures: Vec::new().into_boxed_slice(),
            }
        );
    }
}
