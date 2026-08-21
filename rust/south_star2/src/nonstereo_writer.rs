//! Connected non-stereo visible-token state.
//!
//! Graph and constraint semantics live in `WriterState`. This module owns only
//! concrete non-stereo spelling facts, live ring-label assignments, and the
//! small lexical commitments forced by multi-token SMILES constructs.

use std::collections::BTreeMap;
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
    #[cfg(test)]
    maximum_spelling_label: Option<usize>,
}

impl RingLabels {
    fn next_available(&self) -> RingLabelSlot {
        let mut candidate = RingLabelSlot(0);
        while self.bonds_by_slot.contains_key(&candidate) {
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
    }

    fn has_open_labels(&self) -> bool {
        !self.bonds_by_slot.is_empty()
    }

    fn is_clean(&self) -> bool {
        self.bonds_by_slot.is_empty()
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
                let labels = self.labels.clone();
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
                let labels = self.labels.clone();
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
                let labels = self.labels.clone();
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
#[path = "nonstereo_writer_tests.rs"]
mod tests;
