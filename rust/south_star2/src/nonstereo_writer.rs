//! Connected non-stereo visible-token state.
//!
//! Graph and constraint semantics live in `WriterState`. This module owns only
//! concrete non-stereo spelling facts, live ring-label assignments, and the
//! small lexical commitments forced by multi-token SMILES constructs.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use crate::ids::{AtomId, BondId};
use crate::prepared::{AdjacentBond, PreparedBond, PreparedGraph, PreparedMolecule};
use crate::solver::{Consistency, ConstraintSolver};
use crate::writer_state::{StructuralCandidate, WriterState};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum NonStereoBondToken {
    Elided,
    Aromatic,
    Single,
    Double,
    Triple,
    DativeAToB,
    DativeBToA,
}

impl NonStereoBondToken {
    fn text_from(self, bond: PreparedBond, from: AtomId) -> &'static str {
        let from_a = if bond.a() == from {
            true
        } else if bond.b() == from {
            false
        } else {
            panic!("bond text requires one endpoint of the prepared bond");
        };

        match self {
            Self::Elided => "",
            Self::Aromatic => ":",
            Self::Single => "-",
            Self::Double => "=",
            Self::Triple => "#",
            Self::DativeAToB if from_a => "->",
            Self::DativeAToB => "<-",
            Self::DativeBToA if from_a => "<-",
            Self::DativeBToA => "->",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedConnectedNonStereo {
    molecule: PreparedMolecule,
    atom_text: Arc<[Box<str>]>,
    bond_tokens: Arc<[NonStereoBondToken]>,
}

impl PreparedConnectedNonStereo {
    pub(crate) fn new(
        molecule: PreparedMolecule,
        atom_text: Vec<String>,
        bond_tokens: Vec<NonStereoBondToken>,
    ) -> Result<Self, PreparedConnectedNonStereoError> {
        let graph = molecule.graph();
        if graph.atom_count() == 0 {
            return Err(PreparedConnectedNonStereoError::EmptyMolecule);
        }
        if !graph_is_connected(graph) {
            return Err(PreparedConnectedNonStereoError::DisconnectedMolecule);
        }
        if atom_text.len() != graph.atom_count() {
            return Err(PreparedConnectedNonStereoError::AtomTextCountMismatch {
                expected: graph.atom_count(),
                actual: atom_text.len(),
            });
        }
        if bond_tokens.len() != graph.bond_count() {
            return Err(PreparedConnectedNonStereoError::BondTokenCountMismatch {
                expected: graph.bond_count(),
                actual: bond_tokens.len(),
            });
        }
        for (atom, text) in graph.atom_ids().zip(&atom_text) {
            if text.is_empty() {
                return Err(PreparedConnectedNonStereoError::EmptyAtomText(atom));
            }
        }

        Ok(Self {
            molecule,
            atom_text: Arc::from(
                atom_text
                    .into_iter()
                    .map(String::into_boxed_str)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ),
            bond_tokens: Arc::from(bond_tokens.into_boxed_slice()),
        })
    }

    fn molecule(&self) -> &PreparedMolecule {
        &self.molecule
    }

    fn atom_text(&self, atom: AtomId) -> &str {
        self.atom_text
            .get(atom.index())
            .map(AsRef::as_ref)
            .expect("prepared atom text must match the bound molecule")
    }

    fn bond_text(&self, bond: BondId, from: AtomId) -> &'static str {
        let topology = *self
            .molecule
            .graph()
            .bond(bond)
            .expect("prepared bond token must match the bound molecule");
        self.bond_tokens
            .get(bond.index())
            .copied()
            .expect("prepared bond token must match the bound molecule")
            .text_from(topology, from)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PreparedConnectedNonStereoError {
    EmptyMolecule,
    DisconnectedMolecule,
    AtomTextCountMismatch { expected: usize, actual: usize },
    BondTokenCountMismatch { expected: usize, actual: usize },
    EmptyAtomText(AtomId),
}

impl fmt::Display for PreparedConnectedNonStereoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMolecule => {
                formatter.write_str("a connected non-stereo surface requires at least one atom")
            }
            Self::DisconnectedMolecule => {
                formatter.write_str("a connected non-stereo surface requires one graph component")
            }
            Self::AtomTextCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} prepared atom texts, received {actual}"
            ),
            Self::BondTokenCountMismatch { expected, actual } => write!(
                formatter,
                "expected {expected} prepared bond tokens, received {actual}"
            ),
            Self::EmptyAtomText(atom) => write!(
                formatter,
                "prepared atom text for {atom:?} must not be empty"
            ),
        }
    }
}

impl std::error::Error for PreparedConnectedNonStereoError {}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RingLabelSlot(usize);

impl RingLabelSlot {
    const fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RingLabels {
    bonds_by_slot: BTreeMap<RingLabelSlot, BondId>,
    released_in_current_suffix: BTreeSet<RingLabelSlot>,
    #[cfg(test)]
    maximum_spelling_label: Option<usize>,
}

impl RingLabels {
    fn next_available(&self) -> RingLabelSlot {
        let mut candidate = RingLabelSlot(0);
        while self.bonds_by_slot.contains_key(&candidate)
            || self.released_in_current_suffix.contains(&candidate)
        {
            candidate = RingLabelSlot(
                candidate
                    .index()
                    .checked_add(1)
                    .expect("visible ring-label space must not overflow"),
            );
        }
        candidate
    }

    fn next_label_text(&self, slot: RingLabelSlot) -> Option<String> {
        #[cfg(test)]
        let maximum = self.maximum_spelling_label.unwrap_or(99);
        #[cfg(not(test))]
        let maximum = 99;
        try_ring_label_text_with_maximum(slot, maximum)
    }

    fn allocate(&mut self, bond: BondId) -> RingLabelSlot {
        assert!(
            self.bonds_by_slot.values().all(|owner| *owner != bond),
            "one ring bond may own only one visible label"
        );
        let slot = self.next_available();
        assert_eq!(
            self.bonds_by_slot.insert(slot, bond),
            None,
            "a newly allocated visible ring label must be free"
        );
        slot
    }

    fn slot_for_bond(&self, bond: BondId) -> RingLabelSlot {
        self.bonds_by_slot
            .iter()
            .find_map(|(slot, owner)| (*owner == bond).then_some(*slot))
            .expect("an open structural ring must own a visible label")
    }

    fn release(&mut self, slot: RingLabelSlot, bond: BondId) {
        assert_eq!(
            self.bonds_by_slot.remove(&slot),
            Some(bond),
            "a closing ring must release its own visible label"
        );
        assert!(
            self.released_in_current_suffix.insert(slot),
            "one visible ring label may be released once per atom suffix"
        );
    }

    fn finish_atom_suffix(&mut self) {
        self.released_in_current_suffix.clear();
    }

    fn has_open_labels(&self) -> bool {
        !self.bonds_by_slot.is_empty()
    }

    fn is_clean(&self) -> bool {
        self.bonds_by_slot.is_empty() && self.released_in_current_suffix.is_empty()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PendingEmission {
    InlineAtom(AdjacentBond),
    BranchBondOrAtom(AdjacentBond),
    BranchAtom(AdjacentBond),
    RingClosureLabel {
        incident: AdjacentBond,
        first_endpoint: AtomId,
        label_slot: RingLabelSlot,
    },
}

impl PendingEmission {
    const fn incident(self) -> AdjacentBond {
        match self {
            Self::InlineAtom(incident)
            | Self::BranchBondOrAtom(incident)
            | Self::BranchAtom(incident)
            | Self::RingClosureLabel { incident, .. } => incident,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Choice<S> {
    text: String,
    successor: S,
}

impl<S> Choice<S> {
    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    pub(crate) fn successor(&self) -> &S {
        &self.successor
    }

    pub(crate) fn into_successor(self) -> S {
        self.successor
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CandidateRejection {
    Contradiction,
    RingLabelUnavailable,
}

enum CandidateAttempt<S, E> {
    Accepted { text: String, successor: S },
    Rejected { reason: CandidateRejection },
    Failed(E),
}

#[derive(Clone, Debug)]
pub(crate) struct ConnectedNonStereoWriterState<S> {
    surface: PreparedConnectedNonStereo,
    structural: WriterState<S>,
    labels: RingLabels,
    pending: Option<PendingEmission>,
}

impl<S: ConstraintSolver> ConnectedNonStereoWriterState<S> {
    pub(crate) fn initial(
        surface: &PreparedConnectedNonStereo,
    ) -> Result<Consistency<Self>, S::Failure> {
        Ok(
            WriterState::initial(surface.molecule())?.map(|structural| Self {
                surface: surface.clone(),
                structural,
                labels: RingLabels::default(),
                pending: None,
            }),
        )
    }

    pub(crate) fn active_atom(&self) -> Option<AtomId> {
        self.structural.active_atom()
    }

    pub(crate) const fn graph_is_complete(&self) -> bool {
        self.structural.graph_is_complete()
    }

    pub(crate) fn is_accepted(&self) -> bool {
        self.pending.is_none()
            && self.labels.is_clean()
            && self.structural.active_atom().is_none()
            && self.structural.graph_is_complete()
    }

    pub(crate) fn choices(&self) -> Result<Vec<Choice<Self>>, S::Failure> {
        if self.is_accepted() {
            return Ok(Vec::new());
        }
        if let Some(pending) = self.pending {
            return match self.attempt_pending(pending) {
                CandidateAttempt::Accepted { text, successor } => {
                    Ok(vec![Choice { text, successor }])
                }
                CandidateAttempt::Rejected { reason } => {
                    panic!("stored pending emission was rejected: {reason:?}")
                }
                CandidateAttempt::Failed(failure) => Err(failure),
            };
        }

        let batch = self.structural.derive_candidates();
        assert!(
            !batch.is_contradiction(),
            "connected non-stereo writer reached a structural contradiction"
        );
        let mut choices = Vec::new();
        for &candidate in batch.candidates() {
            if candidate == StructuralCandidate::CompletePath && self.structural.graph_is_complete()
            {
                panic!("a complete connected writer state must already be normalized");
            }
            match self.attempt_structural(candidate) {
                CandidateAttempt::Accepted { text, successor } => {
                    choices.push(Choice { text, successor });
                }
                CandidateAttempt::Rejected { reason: _ } => {}
                CandidateAttempt::Failed(failure) => return Err(failure),
            }
        }
        Ok(choices)
    }

    fn attempt_structural(
        &self,
        candidate: StructuralCandidate,
    ) -> CandidateAttempt<Self, S::Failure> {
        match candidate {
            StructuralCandidate::Root { atom } => {
                assert!(
                    self.labels.is_clean(),
                    "a connected component must start with clean ring-label spelling state"
                );
                let structural = match self.structural.attempt_candidate(candidate) {
                    Ok(Consistency::Consistent(structural)) => structural,
                    Ok(Consistency::Contradiction) => {
                        return CandidateAttempt::Rejected {
                            reason: CandidateRejection::Contradiction,
                        };
                    }
                    Err(failure) => return CandidateAttempt::Failed(failure),
                };
                self.finish_attempt(
                    self.surface.atom_text(atom).to_owned(),
                    Self {
                        surface: self.surface.clone(),
                        structural,
                        labels: self.labels.clone(),
                        pending: None,
                    },
                )
            }
            StructuralCandidate::RingOpen { incident } => {
                let label_slot = self.labels.next_available();
                let Some(text) = self.labels.next_label_text(label_slot) else {
                    return CandidateAttempt::Rejected {
                        reason: CandidateRejection::RingLabelUnavailable,
                    };
                };
                let structural = match self.structural.attempt_candidate(candidate) {
                    Ok(Consistency::Consistent(structural)) => structural,
                    Ok(Consistency::Contradiction) => {
                        return CandidateAttempt::Rejected {
                            reason: CandidateRejection::Contradiction,
                        };
                    }
                    Err(failure) => return CandidateAttempt::Failed(failure),
                };
                let mut labels = self.labels.clone();
                let slot = labels.allocate(incident.bond());
                assert_eq!(
                    slot, label_slot,
                    "advertised ring label must match the allocated label"
                );
                self.finish_attempt(
                    text,
                    Self {
                        surface: self.surface.clone(),
                        structural,
                        labels,
                        pending: None,
                    },
                )
            }
            StructuralCandidate::RingClose {
                incident,
                first_endpoint,
            } => {
                let label_slot = self.labels.slot_for_bond(incident.bond());
                let bond_text = self.surface.bond_text(incident.bond(), first_endpoint);
                if bond_text.is_empty() {
                    let structural = match self.structural.attempt_candidate(candidate) {
                        Ok(Consistency::Consistent(structural)) => structural,
                        Ok(Consistency::Contradiction) => {
                            return CandidateAttempt::Rejected {
                                reason: CandidateRejection::Contradiction,
                            };
                        }
                        Err(failure) => return CandidateAttempt::Failed(failure),
                    };
                    let mut labels = self.labels.clone();
                    labels.release(label_slot, incident.bond());
                    self.finish_attempt(
                        ring_label_text(label_slot),
                        Self {
                            surface: self.surface.clone(),
                            structural,
                            labels,
                            pending: None,
                        },
                    )
                } else {
                    self.finish_attempt(
                        bond_text.to_owned(),
                        Self {
                            surface: self.surface.clone(),
                            structural: self.structural.clone(),
                            labels: self.labels.clone(),
                            pending: Some(PendingEmission::RingClosureLabel {
                                incident,
                                first_endpoint,
                                label_slot,
                            }),
                        },
                    )
                }
            }
            StructuralCandidate::BranchChild { incident } => {
                let structural = match self.structural.attempt_candidate(candidate) {
                    Ok(Consistency::Consistent(structural)) => structural,
                    Ok(Consistency::Contradiction) => {
                        return CandidateAttempt::Rejected {
                            reason: CandidateRejection::Contradiction,
                        };
                    }
                    Err(failure) => return CandidateAttempt::Failed(failure),
                };
                let mut labels = self.labels.clone();
                labels.finish_atom_suffix();
                self.finish_attempt(
                    "(".to_owned(),
                    Self {
                        surface: self.surface.clone(),
                        structural,
                        labels,
                        pending: Some(PendingEmission::BranchBondOrAtom(incident)),
                    },
                )
            }
            StructuralCandidate::InlineChild { incident } => {
                let parent = self
                    .structural
                    .active_atom()
                    .expect("inline emission requires an active atom");
                let bond_text = self.surface.bond_text(incident.bond(), parent);
                let structural = match self.structural.attempt_candidate(candidate) {
                    Ok(Consistency::Consistent(structural)) => structural,
                    Ok(Consistency::Contradiction) => {
                        return CandidateAttempt::Rejected {
                            reason: CandidateRejection::Contradiction,
                        };
                    }
                    Err(failure) => return CandidateAttempt::Failed(failure),
                };
                let mut labels = self.labels.clone();
                labels.finish_atom_suffix();
                if bond_text.is_empty() {
                    self.finish_attempt(
                        self.surface.atom_text(incident.atom()).to_owned(),
                        Self {
                            surface: self.surface.clone(),
                            structural: structural.enter_committed_inline_child(incident),
                            labels,
                            pending: None,
                        },
                    )
                } else {
                    self.finish_attempt(
                        bond_text.to_owned(),
                        Self {
                            surface: self.surface.clone(),
                            structural,
                            labels,
                            pending: Some(PendingEmission::InlineAtom(incident)),
                        },
                    )
                }
            }
            StructuralCandidate::CompletePath => {
                assert!(
                    !self.structural.graph_is_complete(),
                    "top-level completion is normalized without a visible token"
                );
                let structural = match self.structural.attempt_candidate(candidate) {
                    Ok(Consistency::Consistent(structural)) => structural,
                    Ok(Consistency::Contradiction) => {
                        return CandidateAttempt::Rejected {
                            reason: CandidateRejection::Contradiction,
                        };
                    }
                    Err(failure) => return CandidateAttempt::Failed(failure),
                };
                let mut labels = self.labels.clone();
                labels.finish_atom_suffix();
                self.finish_attempt(
                    ")".to_owned(),
                    Self {
                        surface: self.surface.clone(),
                        structural,
                        labels,
                        pending: None,
                    },
                )
            }
        }
    }

    fn attempt_pending(&self, pending: PendingEmission) -> CandidateAttempt<Self, S::Failure> {
        let incident = pending.incident();

        match pending {
            PendingEmission::InlineAtom(_) => self.finish_attempt(
                self.surface.atom_text(incident.atom()).to_owned(),
                Self {
                    surface: self.surface.clone(),
                    structural: self.structural.enter_committed_inline_child(incident),
                    labels: self.labels.clone(),
                    pending: None,
                },
            ),
            PendingEmission::BranchBondOrAtom(_) => {
                let parent = self
                    .structural
                    .active_atom()
                    .expect("a committed branch child requires its active parent");
                if self.surface.bond_text(incident.bond(), parent).is_empty() {
                    self.finish_attempt(
                        self.surface.atom_text(incident.atom()).to_owned(),
                        Self {
                            surface: self.surface.clone(),
                            structural: self.structural.enter_committed_branch_child(incident),
                            labels: self.labels.clone(),
                            pending: None,
                        },
                    )
                } else {
                    self.finish_attempt(
                        self.surface.bond_text(incident.bond(), parent).to_owned(),
                        Self {
                            surface: self.surface.clone(),
                            structural: self.structural.clone(),
                            labels: self.labels.clone(),
                            pending: Some(PendingEmission::BranchAtom(incident)),
                        },
                    )
                }
            }
            PendingEmission::BranchAtom(_) => self.finish_attempt(
                self.surface.atom_text(incident.atom()).to_owned(),
                Self {
                    surface: self.surface.clone(),
                    structural: self.structural.enter_committed_branch_child(incident),
                    labels: self.labels.clone(),
                    pending: None,
                },
            ),
            PendingEmission::RingClosureLabel {
                first_endpoint,
                label_slot,
                ..
            } => {
                assert_eq!(
                    self.labels.slot_for_bond(incident.bond()),
                    label_slot,
                    "a pending ring label must retain its assignment"
                );
                let structural =
                    match self
                        .structural
                        .attempt_candidate(StructuralCandidate::RingClose {
                            incident,
                            first_endpoint,
                        }) {
                        Ok(Consistency::Consistent(structural)) => structural,
                        Ok(Consistency::Contradiction) => {
                            return CandidateAttempt::Rejected {
                                reason: CandidateRejection::Contradiction,
                            };
                        }
                        Err(failure) => return CandidateAttempt::Failed(failure),
                    };
                let mut labels = self.labels.clone();
                labels.release(label_slot, incident.bond());
                self.finish_attempt(
                    ring_label_text(label_slot),
                    Self {
                        surface: self.surface.clone(),
                        structural,
                        labels,
                        pending: None,
                    },
                )
            }
        }
    }

    fn finish_attempt(&self, text: String, successor: Self) -> CandidateAttempt<Self, S::Failure> {
        match successor.normalize_and_check() {
            Ok(Consistency::Consistent(successor)) => {
                CandidateAttempt::Accepted { text, successor }
            }
            Ok(Consistency::Contradiction) => CandidateAttempt::Rejected {
                reason: CandidateRejection::Contradiction,
            },
            Err(failure) => CandidateAttempt::Failed(failure),
        }
    }

    fn normalize_and_check(mut self) -> Result<Consistency<Self>, S::Failure> {
        if let Some(pending) = self.pending {
            assert!(
                self.structural.active_atom().is_some(),
                "pending text requires an active structural path"
            );
            return match self.attempt_pending(pending) {
                CandidateAttempt::Accepted { .. } => Ok(Consistency::Consistent(self)),
                CandidateAttempt::Rejected { .. } => Ok(Consistency::Contradiction),
                CandidateAttempt::Failed(failure) => Err(failure),
            };
        }
        if !self.structural.graph_is_complete() {
            let batch = self.structural.derive_candidates();
            if batch.is_contradiction() || batch.candidates().is_empty() {
                return Ok(Consistency::Contradiction);
            }
            return Ok(Consistency::Consistent(self));
        }
        assert!(
            !self.labels.has_open_labels(),
            "a complete structural graph must not retain open visible ring labels"
        );
        self.labels.finish_atom_suffix();
        let batch = self.structural.derive_candidates();
        assert_eq!(
            batch.candidates(),
            &[StructuralCandidate::CompletePath],
            "a complete connected graph must have one silent top-level completion"
        );
        let completed = match self
            .structural
            .attempt_candidate(StructuralCandidate::CompletePath)?
        {
            Consistency::Consistent(completed) => completed,
            Consistency::Contradiction => {
                panic!("top-level structural completion cannot contradict the CSP")
            }
        };
        assert_eq!(
            completed.active_atom(),
            None,
            "connected graph completion must not restore a branch parent"
        );
        self.structural = completed;
        Ok(Consistency::Consistent(self))
    }
}

fn ring_label_text(label_slot: RingLabelSlot) -> String {
    try_ring_label_text(label_slot)
        .expect("ring labels above 99 require an explicit dialect policy")
}

fn try_ring_label_text(label_slot: RingLabelSlot) -> Option<String> {
    try_ring_label_text_with_maximum(label_slot, 99)
}

fn try_ring_label_text_with_maximum(label_slot: RingLabelSlot, maximum: usize) -> Option<String> {
    let label = label_slot
        .index()
        .checked_add(1)
        .expect("visible ring-label number must not overflow");
    (label <= maximum).then(|| ring_label_number_text(label))
}

fn ring_label_number_text(label: usize) -> String {
    assert!(label > 0, "visible ring labels are one-based");
    assert!(
        label <= 99,
        "ring labels above 99 require an explicit dialect policy"
    );
    if label < 10 {
        label.to_string()
    } else {
        format!("%{label}")
    }
}

fn graph_is_connected(graph: &PreparedGraph) -> bool {
    if graph.atom_count() == 0 {
        return false;
    }
    let root = AtomId::new(0);
    let mut visited = vec![false; graph.atom_count()];
    visited[root.index()] = true;
    let mut pending = vec![root];
    let mut visited_count = 0_usize;

    while let Some(atom) = pending.pop() {
        visited_count += 1;
        for incident in graph
            .neighbors(atom)
            .expect("prepared atom must have an adjacency row")
        {
            let neighbour = incident.atom();
            if !visited[neighbour.index()] {
                visited[neighbour.index()] = true;
                pending.push(neighbour);
            }
        }
    }
    visited_count == graph.atom_count()
}

#[cfg(test)]
#[path = "nonstereo_reference.rs"]
mod reference;

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::error::Error;
    use std::sync::Arc;

    use super::*;
    use crate::native::NativeSolverState;
    use crate::native_solver::NativeSolverFailure;
    use crate::prepared::PreparedGraphBuilder;

    type State = ConnectedNonStereoWriterState<NativeSolverState>;

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum InjectedSolverFailure {
        Native(NativeSolverFailure),
        Restriction,
    }

    impl fmt::Display for InjectedSolverFailure {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Native(failure) => failure.fmt(formatter),
                Self::Restriction => formatter.write_str("injected restriction failure"),
            }
        }
    }

    impl Error for InjectedSolverFailure {}

    #[derive(Clone, Debug)]
    struct FailingRestrictionSolver(NativeSolverState);

    impl ConstraintSolver for FailingRestrictionSolver {
        type Failure = InjectedSolverFailure;

        fn initial(
            model: Arc<crate::model::ConstraintModel>,
        ) -> Result<Consistency<Self>, Self::Failure> {
            Ok(<NativeSolverState as ConstraintSolver>::initial(model)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self))
        }

        fn restricted(
            &self,
            _restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        ) -> Result<Consistency<Self>, Self::Failure> {
            Err(InjectedSolverFailure::Restriction)
        }

        fn domain(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
            self.0.domain(variable)
        }
    }

    #[derive(Clone, Debug)]
    struct RejectFirstVariableSolver(NativeSolverState);

    impl ConstraintSolver for RejectFirstVariableSolver {
        type Failure = InjectedSolverFailure;

        fn initial(
            model: Arc<crate::model::ConstraintModel>,
        ) -> Result<Consistency<Self>, Self::Failure> {
            Ok(<NativeSolverState as ConstraintSolver>::initial(model)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self))
        }

        fn restricted(
            &self,
            restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        ) -> Result<Consistency<Self>, Self::Failure> {
            if restrictions
                .iter()
                .any(|(variable, _)| *variable == crate::ids::VariableId::new(0))
            {
                return Ok(Consistency::Contradiction);
            }
            Ok(
                <NativeSolverState as ConstraintSolver>::restricted(&self.0, restrictions)
                    .map_err(InjectedSolverFailure::Native)?
                    .map(Self),
            )
        }

        fn domain(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
            self.0.domain(variable)
        }
    }

    #[derive(Clone, Debug)]
    struct WriterPolicyContradictionSolver(NativeSolverState);

    impl ConstraintSolver for WriterPolicyContradictionSolver {
        type Failure = InjectedSolverFailure;

        fn initial(
            model: Arc<crate::model::ConstraintModel>,
        ) -> Result<Consistency<Self>, Self::Failure> {
            Ok(<NativeSolverState as ConstraintSolver>::initial(model)
                .map_err(InjectedSolverFailure::Native)?
                .map(Self))
        }

        fn restricted(
            &self,
            restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        ) -> Result<Consistency<Self>, Self::Failure> {
            use crate::model::BondRole;

            let first_ring = (
                crate::ids::VariableId::new(0),
                BondRole::Ring.singleton_domain(),
            );
            let requested_first_ring = restrictions == [first_ring];
            let effective = if requested_first_ring {
                vec![
                    first_ring,
                    (
                        crate::ids::VariableId::new(1),
                        BondRole::Traversal.singleton_domain(),
                    ),
                    (
                        crate::ids::VariableId::new(2),
                        BondRole::Traversal.singleton_domain(),
                    ),
                    (
                        crate::ids::VariableId::new(3),
                        BondRole::Traversal.singleton_domain(),
                    ),
                    (
                        crate::ids::VariableId::new(4),
                        BondRole::Ring.singleton_domain(),
                    ),
                ]
            } else {
                restrictions.to_vec()
            };
            Ok(
                <NativeSolverState as ConstraintSolver>::restricted(&self.0, &effective)
                    .map_err(InjectedSolverFailure::Native)?
                    .map(Self),
            )
        }

        fn domain(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
            self.0.domain(variable)
        }
    }

    #[derive(Clone, Debug)]
    struct PendingContradictionSolver(NativeSolverState);

    impl ConstraintSolver for PendingContradictionSolver {
        type Failure = InjectedSolverFailure;

        fn initial(
            model: Arc<crate::model::ConstraintModel>,
        ) -> Result<Consistency<Self>, Self::Failure> {
            use crate::model::BondRole;

            let native = <NativeSolverState as ConstraintSolver>::initial(model)
                .map_err(InjectedSolverFailure::Native)?
                .unwrap_consistent();
            let restrictions = [
                (
                    crate::ids::VariableId::new(0),
                    BondRole::Traversal.singleton_domain(),
                ),
                (
                    crate::ids::VariableId::new(1),
                    BondRole::Traversal.singleton_domain(),
                ),
                (
                    crate::ids::VariableId::new(2),
                    BondRole::Traversal.singleton_domain(),
                ),
                (
                    crate::ids::VariableId::new(3),
                    BondRole::Ring.singleton_domain(),
                ),
            ];
            Ok(
                <NativeSolverState as ConstraintSolver>::restricted(&native, &restrictions)
                    .map_err(InjectedSolverFailure::Native)?
                    .map(Self),
            )
        }

        fn restricted(
            &self,
            restrictions: &[(crate::ids::VariableId, crate::domain::Domain)],
        ) -> Result<Consistency<Self>, Self::Failure> {
            Ok(
                <NativeSolverState as ConstraintSolver>::restricted(&self.0, restrictions)
                    .map_err(InjectedSolverFailure::Native)?
                    .map(Self),
            )
        }

        fn domain(&self, variable: crate::ids::VariableId) -> Option<crate::domain::Domain> {
            self.0.domain(variable)
        }
    }

    fn fixture(
        atom_text: &[&str],
        edges: &[(usize, usize, NonStereoBondToken)],
    ) -> (PreparedConnectedNonStereo, Vec<AtomId>, Vec<BondId>) {
        let mut graph = PreparedGraphBuilder::new();
        let atoms = atom_text
            .iter()
            .map(|_| graph.add_atom().unwrap())
            .collect::<Vec<_>>();
        let mut bonds = Vec::with_capacity(edges.len());
        let mut bond_tokens = Vec::with_capacity(edges.len());
        for &(a, b, token) in edges {
            bonds.push(graph.add_bond(atoms[a], atoms[b]).unwrap());
            bond_tokens.push(token);
        }
        let surface = PreparedConnectedNonStereo::new(
            PreparedMolecule::new(graph.build()),
            atom_text.iter().map(|text| (*text).to_owned()).collect(),
            bond_tokens,
        )
        .unwrap();
        (surface, atoms, bonds)
    }

    fn incident(surface: &PreparedConnectedNonStereo, atom: AtomId, bond: BondId) -> AdjacentBond {
        surface
            .molecule()
            .graph()
            .neighbors(atom)
            .expect("fixture atom must exist")
            .iter()
            .copied()
            .find(|candidate| candidate.bond() == bond)
            .expect("fixture bond must be incident to the atom")
    }

    fn only_choice(state: &State, text: &str) -> (String, State) {
        let choices = state.choices().unwrap();
        assert_eq!(choices.len(), 1);
        assert_eq!(choices[0].text(), text);
        let choice = choices.into_iter().next().unwrap();
        (choice.text, choice.successor)
    }

    fn choice_at(state: &State, index: usize) -> (String, State) {
        let choice = state.choices().unwrap().into_iter().nth(index).unwrap();
        (choice.text, choice.successor)
    }

    fn initial(surface: &PreparedConnectedNonStereo) -> State {
        State::initial(surface).unwrap().unwrap_consistent()
    }

    #[test]
    fn surface_rejects_invalid_bindings() {
        let empty = PreparedMolecule::new(PreparedGraphBuilder::new().build());
        assert!(matches!(
            PreparedConnectedNonStereo::new(empty, Vec::new(), Vec::new()),
            Err(PreparedConnectedNonStereoError::EmptyMolecule)
        ));

        let mut graph = PreparedGraphBuilder::new();
        graph.add_atom().unwrap();
        graph.add_atom().unwrap();
        let disconnected = PreparedMolecule::new(graph.build());
        assert!(matches!(
            PreparedConnectedNonStereo::new(
                disconnected,
                vec!["C".to_owned(), "O".to_owned()],
                Vec::new(),
            ),
            Err(PreparedConnectedNonStereoError::DisconnectedMolecule)
        ));

        let mut graph = PreparedGraphBuilder::new();
        graph.add_atom().unwrap();
        let single = PreparedMolecule::new(graph.build());
        assert!(matches!(
            PreparedConnectedNonStereo::new(single.clone(), Vec::new(), Vec::new()),
            Err(PreparedConnectedNonStereoError::AtomTextCountMismatch { .. })
        ));
        assert!(matches!(
            PreparedConnectedNonStereo::new(single, vec![String::new()], Vec::new()),
            Err(PreparedConnectedNonStereoError::EmptyAtomText(atom))
                if atom == AtomId::new(0)
        ));

        let mut graph = PreparedGraphBuilder::new();
        let atoms: [AtomId; 2] = std::array::from_fn(|_| graph.add_atom().unwrap());
        graph.add_bond(atoms[0], atoms[1]).unwrap();
        let bonded = PreparedMolecule::new(graph.build());
        assert!(matches!(
            PreparedConnectedNonStereo::new(
                bonded,
                vec!["C".to_owned(), "O".to_owned()],
                Vec::new(),
            ),
            Err(PreparedConnectedNonStereoError::BondTokenCountMismatch { .. })
        ));
    }

    #[test]
    fn equal_text_choices_retain_distinct_successors() {
        let (surface, atoms, _) = fixture(&["C", "C"], &[(0, 1, NonStereoBondToken::Elided)]);
        let initial = initial(&surface);

        let choices = initial.choices().unwrap();
        assert_eq!(choices.len(), 2);
        assert_eq!(choices[0].text(), choices[1].text());
        assert_eq!(choices[0].text(), "C");
        assert_eq!(choices[0].successor().active_atom(), Some(atoms[0]));
        assert_eq!(choices[1].successor().active_atom(), Some(atoms[1]));
        assert_eq!(initial.active_atom(), None);
    }

    #[test]
    fn choices_derives_the_source_frontier_once() {
        let surface = fixture(
            &["C", "C", "C"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (0, 2, NonStereoBondToken::Elided),
                (1, 2, NonStereoBondToken::Elided),
            ],
        )
        .0;
        let initial = initial(&surface);
        let rooted = initial
            .choices()
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .into_successor();
        let before = rooted.structural.candidate_batch_derivation_count();

        let choices = rooted.choices().unwrap();

        assert_eq!(choices.len(), 2);
        assert_eq!(
            rooted.structural.candidate_batch_derivation_count(),
            before + 1
        );
    }

    #[test]
    fn backend_failure_aborts_the_candidate_batch() {
        let surface = fixture(
            &["C", "C", "C"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (0, 2, NonStereoBondToken::Elided),
                (1, 2, NonStereoBondToken::Elided),
            ],
        )
        .0;
        let initial = ConnectedNonStereoWriterState::<FailingRestrictionSolver>::initial(&surface)
            .unwrap()
            .unwrap_consistent();
        let rooted = initial
            .choices()
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .into_successor();

        assert!(matches!(
            rooted.choices(),
            Err(InjectedSolverFailure::Restriction)
        ));
    }

    #[test]
    fn contradictory_candidate_is_filtered_without_suppressing_its_sibling() {
        let (surface, atoms, bonds) = fixture(
            &["C", "C", "C"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (0, 2, NonStereoBondToken::Elided),
                (1, 2, NonStereoBondToken::Elided),
            ],
        );
        assert_eq!(
            surface.molecule().bond_role_variable(bonds[0]),
            Some(crate::ids::VariableId::new(0))
        );
        let initial = ConnectedNonStereoWriterState::<RejectFirstVariableSolver>::initial(&surface)
            .unwrap()
            .unwrap_consistent();
        let rooted = initial
            .choices()
            .unwrap()
            .into_iter()
            .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
            .unwrap()
            .into_successor();

        let choices = rooted.choices().unwrap();

        assert_eq!(choices.len(), 1);
        assert_eq!(choices[0].text(), "1");
        assert_eq!(
            choices[0]
                .successor()
                .labels
                .bonds_by_slot
                .values()
                .copied()
                .collect::<Vec<_>>(),
            vec![bonds[1]]
        );
        assert!(rooted.labels.is_clean());
    }

    #[test]
    fn writer_policy_contradiction_is_candidate_local() {
        let (surface, atoms, bonds) = fixture(
            &["R", "A", "B", "C"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (0, 2, NonStereoBondToken::Elided),
                (0, 3, NonStereoBondToken::Elided),
                (1, 2, NonStereoBondToken::Elided),
                (2, 3, NonStereoBondToken::Elided),
            ],
        );
        let initial =
            ConnectedNonStereoWriterState::<WriterPolicyContradictionSolver>::initial(&surface)
                .unwrap()
                .unwrap_consistent();
        let rooted = initial
            .choices()
            .unwrap()
            .into_iter()
            .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
            .unwrap()
            .into_successor();

        let choices = rooted.choices().unwrap();

        assert!(!choices.is_empty());
        assert!(choices.iter().all(|choice| {
            !choice
                .successor()
                .labels
                .bonds_by_slot
                .values()
                .any(|bond| *bond == bonds[0])
        }));
        assert!(choices.iter().any(|choice| {
            choice
                .successor()
                .labels
                .bonds_by_slot
                .values()
                .any(|bond| *bond == bonds[1])
        }));
    }

    #[test]
    fn pending_atom_is_not_advertised_before_its_successor_is_valid() {
        let (surface, atoms, _) = fixture(
            &["R", "A", "B", "C"],
            &[
                (0, 1, NonStereoBondToken::Double),
                (1, 2, NonStereoBondToken::Elided),
                (1, 3, NonStereoBondToken::Elided),
                (2, 3, NonStereoBondToken::Elided),
            ],
        );
        let initial =
            ConnectedNonStereoWriterState::<PendingContradictionSolver>::initial(&surface)
                .unwrap()
                .unwrap_consistent();
        let rooted = initial
            .choices()
            .unwrap()
            .into_iter()
            .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
            .unwrap()
            .into_successor();

        assert!(rooted.choices().unwrap().is_empty());
    }

    #[test]
    fn unspellable_openings_do_not_suppress_a_valid_closure() {
        let (surface, atoms, bonds) = fixture(
            &["A", "B", "C", "D", "E"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (0, 2, NonStereoBondToken::Elided),
                (1, 2, NonStereoBondToken::Elided),
                (1, 3, NonStereoBondToken::Elided),
                (1, 4, NonStereoBondToken::Elided),
                (3, 4, NonStereoBondToken::Elided),
            ],
        );
        let initial = initial(&surface);
        let rooted = initial
            .choices()
            .unwrap()
            .into_iter()
            .find(|choice| choice.successor().active_atom() == Some(atoms[0]))
            .unwrap()
            .into_successor();
        let opened = rooted
            .choices()
            .unwrap()
            .into_iter()
            .find(|choice| {
                choice
                    .successor()
                    .labels
                    .bonds_by_slot
                    .values()
                    .any(|bond| *bond == bonds[0])
            })
            .unwrap()
            .into_successor();
        let (_, walked) = only_choice(&opened, "C");
        let (_, mut walked) = only_choice(&walked, "B");
        assert_eq!(walked.active_atom(), Some(atoms[1]));
        walked.labels.maximum_spelling_label = Some(1);
        assert_eq!(walked.labels.next_available(), RingLabelSlot(1));

        let choices = walked.choices().unwrap();

        assert_eq!(choices.len(), 1);
        assert_eq!(choices[0].text(), "1");
        assert!(!choices[0]
            .successor()
            .labels
            .bonds_by_slot
            .values()
            .any(|bond| *bond == bonds[0]));
    }

    #[test]
    fn closed_labels_become_reusable_after_the_atom_suffix() {
        let first = BondId::new(0);
        let second = BondId::new(1);
        let third = BondId::new(2);
        let mut labels = RingLabels::default();

        let zero = labels.allocate(first);
        let one = labels.allocate(second);
        assert_eq!(zero, RingLabelSlot(0));
        assert_eq!(one, RingLabelSlot(1));
        labels.release(zero, first);
        assert_eq!(labels.next_available(), RingLabelSlot(2));
        labels.finish_atom_suffix();
        assert_eq!(labels.allocate(third), RingLabelSlot(0));
    }

    #[test]
    fn ring_label_spelling_matches_the_selected_smiles_dialect() {
        assert_eq!(ring_label_number_text(1), "1");
        assert_eq!(ring_label_number_text(9), "9");
        assert_eq!(ring_label_number_text(10), "%10");
        assert_eq!(ring_label_number_text(99), "%99");
    }

    #[test]
    #[should_panic(expected = "above 99 require an explicit dialect policy")]
    fn unselected_large_ring_label_dialect_fails_at_rendering() {
        let _ = ring_label_number_text(100);
    }

    #[test]
    fn elided_triangle_emits_a_complete_ring() {
        let (surface, atoms, bonds) = fixture(
            &["C", "C", "C"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (0, 2, NonStereoBondToken::Elided),
                (1, 2, NonStereoBondToken::Elided),
            ],
        );
        let initial = initial(&surface);
        let left = incident(&surface, atoms[0], bonds[0]);
        let right = incident(&surface, atoms[0], bonds[1]);
        let between = incident(&surface, atoms[2], bonds[2]);
        let closing = incident(&surface, atoms[1], bonds[0]);

        let (root, rooted) = choice_at(&initial, atoms[0].index());
        let rooted_choices = rooted.choices().unwrap();
        assert_eq!(
            rooted_choices.iter().map(Choice::text).collect::<Vec<_>>(),
            vec!["1", "1"]
        );
        assert!(rooted_choices[0]
            .successor()
            .labels
            .bonds_by_slot
            .values()
            .any(|bond| *bond == left.bond()));
        assert!(rooted_choices[1]
            .successor()
            .labels
            .bonds_by_slot
            .values()
            .any(|bond| *bond == right.bond()));
        let first_opening = rooted_choices.into_iter().next().unwrap();
        let open = first_opening.text;
        let opened = first_opening.successor;
        let (first_child, walked) = only_choice(&opened, "C");
        let (second_child, walked) = only_choice(&walked, "C");
        assert_eq!(walked.structural.active_atom(), Some(atoms[1]));
        assert_eq!(between.bond(), bonds[2]);
        assert_eq!(closing.bond(), bonds[0]);
        let (close, accepted) = only_choice(&walked, "1");

        assert_eq!(
            [root, open, first_child, second_child, close].concat(),
            "C1CC1"
        );
        assert!(accepted.is_accepted());
    }

    #[test]
    fn explicit_ring_bond_is_emitted_at_closure_before_its_label() {
        let (surface, atoms, bonds) = fixture(
            &["C", "C", "C"],
            &[
                (0, 1, NonStereoBondToken::Double),
                (0, 2, NonStereoBondToken::Elided),
                (1, 2, NonStereoBondToken::Elided),
            ],
        );
        let initial = initial(&surface);
        let ring = incident(&surface, atoms[0], bonds[0]);
        let entry = incident(&surface, atoms[0], bonds[1]);
        let between = incident(&surface, atoms[2], bonds[2]);
        let closing = incident(&surface, atoms[1], bonds[0]);

        let (root, rooted) = choice_at(&initial, atoms[0].index());
        let (open, opened) = choice_at(&rooted, 0);
        assert!(opened
            .labels
            .bonds_by_slot
            .values()
            .any(|bond| *bond == ring.bond()));
        let (first_child, walked) = only_choice(&opened, "C");
        assert_eq!(entry.bond(), bonds[1]);
        let (second_child, walked) = only_choice(&walked, "C");
        assert_eq!(between.bond(), bonds[2]);
        assert_eq!(closing.bond(), bonds[0]);

        let (bond, pending_label) = only_choice(&walked, "=");
        assert_eq!(bond, "=");
        assert_eq!(pending_label.active_atom(), Some(atoms[1]));
        assert!(!pending_label.graph_is_complete());

        let (label, accepted) = only_choice(&pending_label, "1");
        assert_eq!(
            [root, open, first_child, second_child, bond, label].concat(),
            "C1CC=1"
        );
        assert!(accepted.is_accepted());
    }

    #[test]
    fn explicit_inline_bond_commits_before_child_entry() {
        let (surface, atoms, bonds) = fixture(&["C", "O"], &[(0, 1, NonStereoBondToken::Double)]);
        let initial = initial(&surface);
        let edge = incident(&surface, atoms[0], bonds[0]);
        let (_, rooted) = choice_at(&initial, atoms[0].index());
        let (bond, pending) = only_choice(&rooted, "=");

        assert_eq!(bond, "=");
        assert_eq!(rooted.active_atom(), Some(atoms[0]));
        assert_eq!(pending.active_atom(), Some(atoms[0]));
        assert_eq!(pending.pending, Some(PendingEmission::InlineAtom(edge)));
        let (atom, accepted) = only_choice(&pending, "O");
        assert_eq!(atom, "O");
        assert!(accepted.is_accepted());
    }

    #[test]
    fn dative_bond_text_follows_prepared_orientation() {
        let (surface, atoms, bonds) =
            fixture(&["N", "B"], &[(0, 1, NonStereoBondToken::DativeAToB)]);
        let initial = initial(&surface);
        let edge_from_n = incident(&surface, atoms[0], bonds[0]);
        let edge_from_b = incident(&surface, atoms[1], bonds[0]);

        let (_, rooted_at_n) = choice_at(&initial, atoms[0].index());
        assert_eq!(rooted_at_n.pending, None);
        assert_eq!(edge_from_n.bond(), bonds[0]);
        assert_eq!(rooted_at_n.choices().unwrap()[0].text(), "->");

        let (_, rooted_at_b) = choice_at(&initial, atoms[1].index());
        assert_eq!(edge_from_b.bond(), bonds[0]);
        assert_eq!(rooted_at_b.choices().unwrap()[0].text(), "<-");
    }

    #[test]
    fn explicit_branch_commits_at_open_parenthesis() {
        let (surface, atoms, bonds) = fixture(
            &["C", "O", "N"],
            &[
                (0, 1, NonStereoBondToken::Double),
                (0, 2, NonStereoBondToken::Elided),
            ],
        );
        let initial = initial(&surface);
        let oxygen = incident(&surface, atoms[0], bonds[0]);
        let nitrogen = incident(&surface, atoms[0], bonds[1]);
        let (root, rooted) = choice_at(&initial, atoms[0].index());
        let branch_choice = rooted
            .choices()
            .unwrap()
            .into_iter()
            .find(|choice| {
                choice.successor.pending == Some(PendingEmission::BranchBondOrAtom(oxygen))
            })
            .unwrap();
        let open = branch_choice.text;
        let pending_branch = branch_choice.successor;
        let (bond, pending_atom) = only_choice(&pending_branch, "=");
        let (atom, branch) = only_choice(&pending_atom, "O");
        let (close, restored) = only_choice(&branch, ")");
        assert_eq!(nitrogen.bond(), bonds[1]);
        let (inline, accepted) = only_choice(&restored, "N");

        assert_eq!([root, open, bond, atom, close, inline].concat(), "C(=O)N");
        assert!(accepted.is_accepted());
    }

    fn reachable_strings(surface: &PreparedConnectedNonStereo) -> BTreeSet<String> {
        let initial = initial(surface);
        let mut pending = vec![(initial, String::new())];
        let mut complete = BTreeSet::new();
        let mut explored = 0_usize;

        while let Some((state, prefix)) = pending.pop() {
            explored += 1;
            assert!(
                explored <= 100_000,
                "writer test exceeded its exploration bound"
            );
            if state.is_accepted() {
                complete.insert(prefix);
                continue;
            }

            let choices = state.choices().unwrap();
            assert!(
                !choices.is_empty(),
                "writer must not dead-end before acceptance"
            );
            for choice in choices {
                let token = choice.text().to_owned();
                pending.push((choice.into_successor(), format!("{prefix}{token}")));
            }
        }
        complete
    }

    #[test]
    fn connected_tree_support_remains_exact() {
        let fixtures = [
            fixture(&["C"], &[]).0,
            fixture(
                &["C", "N", "O"],
                &[
                    (0, 1, NonStereoBondToken::Elided),
                    (1, 2, NonStereoBondToken::Elided),
                ],
            )
            .0,
            fixture(
                &["C", "N", "O", "F"],
                &[
                    (0, 1, NonStereoBondToken::Elided),
                    (0, 2, NonStereoBondToken::Elided),
                    (0, 3, NonStereoBondToken::Elided),
                ],
            )
            .0,
            fixture(
                &["C", "N", "O", "F", "S"],
                &[
                    (0, 1, NonStereoBondToken::Elided),
                    (0, 2, NonStereoBondToken::Elided),
                    (1, 3, NonStereoBondToken::Elided),
                    (1, 4, NonStereoBondToken::Double),
                ],
            )
            .0,
        ];
        for surface in fixtures {
            assert_eq!(reachable_strings(&surface), reference::support(&surface));
        }
    }

    #[test]
    fn connected_triangle_support_is_writer_shaped() {
        let surface = fixture(
            &["C", "C", "C"],
            &[
                (0, 1, NonStereoBondToken::Elided),
                (0, 2, NonStereoBondToken::Elided),
                (1, 2, NonStereoBondToken::Elided),
            ],
        )
        .0;

        assert_eq!(
            reachable_strings(&surface),
            BTreeSet::from(["C1CC1".to_owned()])
        );
    }

    #[test]
    fn adversarial_connected_support_matches_the_main_reference() {
        let fixtures = [
            (
                "two cycles sharing a vertex with an explicit ring bond",
                fixture(
                    &["A", "B", "C", "D", "E"],
                    &[
                        (0, 1, NonStereoBondToken::Double),
                        (1, 2, NonStereoBondToken::Elided),
                        (2, 0, NonStereoBondToken::Elided),
                        (0, 3, NonStereoBondToken::Elided),
                        (3, 4, NonStereoBondToken::Elided),
                        (4, 0, NonStereoBondToken::Elided),
                    ],
                )
                .0,
                48_usize,
            ),
            (
                "fused triangles",
                fixture(
                    &["A", "B", "C", "D"],
                    &[
                        (0, 1, NonStereoBondToken::Elided),
                        (1, 2, NonStereoBondToken::Elided),
                        (2, 0, NonStereoBondToken::Elided),
                        (1, 3, NonStereoBondToken::Elided),
                        (2, 3, NonStereoBondToken::Elided),
                    ],
                )
                .0,
                28,
            ),
            (
                "three-path bridged system",
                fixture(
                    &["A", "B", "C", "D", "E"],
                    &[
                        (0, 1, NonStereoBondToken::Elided),
                        (1, 3, NonStereoBondToken::Elided),
                        (0, 2, NonStereoBondToken::Elided),
                        (2, 3, NonStereoBondToken::Elided),
                        (0, 4, NonStereoBondToken::Elided),
                        (4, 3, NonStereoBondToken::Elided),
                    ],
                )
                .0,
                36,
            ),
            (
                "ring with a branched substituent",
                fixture(
                    &["A", "B", "C", "D", "E"],
                    &[
                        (0, 1, NonStereoBondToken::Elided),
                        (1, 2, NonStereoBondToken::Elided),
                        (2, 0, NonStereoBondToken::Elided),
                        (0, 3, NonStereoBondToken::Elided),
                        (3, 4, NonStereoBondToken::Elided),
                    ],
                )
                .0,
                16,
            ),
            (
                "directed ring bond",
                fixture(
                    &["N", "B", "C"],
                    &[
                        (0, 1, NonStereoBondToken::DativeAToB),
                        (0, 2, NonStereoBondToken::Elided),
                        (1, 2, NonStereoBondToken::Elided),
                    ],
                )
                .0,
                6,
            ),
        ];

        for (name, surface, expected_count) in fixtures {
            let expected = reference::support(&surface);
            assert_eq!(
                expected.len(),
                expected_count,
                "reference fixture drift: {name}"
            );
            assert_eq!(
                reachable_strings(&surface),
                expected,
                "connected non-stereo support mismatch: {name}"
            );
        }
    }
}
