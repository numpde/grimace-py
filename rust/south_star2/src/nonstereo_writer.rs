//! Graph-general non-stereo visible-token state.
//!
//! Graph and constraint semantics live in `WriterState`. This module owns only
//! concrete non-stereo spelling facts, live ring-label assignments, and the
//! small lexical commitments forced by multi-token SMILES constructs.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use crate::ids::{AtomId, BondId};
use crate::prepared::{AdjacentBond, PreparedBond, PreparedMolecule};
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
pub(crate) struct PreparedNonStereo {
    molecule: PreparedMolecule,
    atom_text: Arc<[Box<str>]>,
    bond_tokens: Arc<[NonStereoBondToken]>,
}

impl PreparedNonStereo {
    pub(crate) fn new(
        molecule: PreparedMolecule,
        atom_text: Vec<String>,
        bond_tokens: Vec<NonStereoBondToken>,
    ) -> Result<Self, PreparedNonStereoError> {
        let graph = molecule.graph();
        if atom_text.len() != graph.atom_count() {
            return Err(PreparedNonStereoError::AtomTextCountMismatch {
                expected: graph.atom_count(),
                actual: atom_text.len(),
            });
        }
        if bond_tokens.len() != graph.bond_count() {
            return Err(PreparedNonStereoError::BondTokenCountMismatch {
                expected: graph.bond_count(),
                actual: bond_tokens.len(),
            });
        }
        for (atom, text) in graph.atom_ids().zip(&atom_text) {
            if text.is_empty() {
                return Err(PreparedNonStereoError::EmptyAtomText(atom));
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
pub(crate) enum PreparedNonStereoError {
    AtomTextCountMismatch { expected: usize, actual: usize },
    BondTokenCountMismatch { expected: usize, actual: usize },
    EmptyAtomText(AtomId),
}

impl fmt::Display for PreparedNonStereoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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

impl std::error::Error for PreparedNonStereoError {}

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
        try_ring_label_text_with_maximum(slot, self.maximum_spelling_label())
    }

    fn maximum_spelling_label(&self) -> usize {
        #[cfg(test)]
        {
            self.maximum_spelling_label.unwrap_or(99)
        }
        #[cfg(not(test))]
        {
            99
        }
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
    ComponentRootAtom(AtomId),
    InlineAtom(AdjacentBond),
    BranchBondOrAtom(AdjacentBond),
    BranchAtom(AdjacentBond),
    RingClosureLabel {
        incident: AdjacentBond,
        first_endpoint: AtomId,
        label_slot: RingLabelSlot,
    },
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
    RingLabelUnavailable {
        next_label: usize,
        maximum_label: usize,
    },
}

enum CandidateAttempt<S, E> {
    Accepted { text: String, successor: S },
    Rejected { reason: CandidateRejection },
    Failed(E),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum SpellingFailure {
    RingLabelExhausted {
        next_label: usize,
        maximum_label: usize,
        blocked_candidate_count: usize,
    },
}

impl fmt::Display for SpellingFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RingLabelExhausted {
                next_label,
                maximum_label,
                blocked_candidate_count,
            } => write!(
                formatter,
                "ring label {next_label} exceeds the selected dialect maximum {maximum_label} for {blocked_candidate_count} candidate(s)"
            ),
        }
    }
}

impl std::error::Error for SpellingFailure {}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum WriterInvariantFailure {
    StructuralContradiction,
    NoStructuralCandidates,
    PendingEmissionRejected,
    AllCandidatesSemanticallyRejected { candidate_count: usize },
}

impl fmt::Display for WriterInvariantFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StructuralContradiction => {
                formatter.write_str("a live writer state has a contradictory structural frontier")
            }
            Self::NoStructuralCandidates => {
                formatter.write_str("a live writer state has no structural candidates")
            }
            Self::PendingEmissionRejected => {
                formatter.write_str("a stored pending emission no longer has a valid successor")
            }
            Self::AllCandidatesSemanticallyRejected { candidate_count } => write!(
                formatter,
                "all {candidate_count} structural candidate(s) contradicted immediate writer consistency"
            ),
        }
    }
}

impl std::error::Error for WriterInvariantFailure {}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ChoiceFailure<E> {
    Backend(E),
    Spelling(SpellingFailure),
    Invariant(WriterInvariantFailure),
}

impl<E: fmt::Display> fmt::Display for ChoiceFailure<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(failure) => write!(formatter, "constraint backend failure: {failure}"),
            Self::Spelling(failure) => failure.fmt(formatter),
            Self::Invariant(failure) => write!(formatter, "writer invariant failure: {failure}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ChoiceFailure<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend(failure) => Some(failure),
            Self::Spelling(failure) => Some(failure),
            Self::Invariant(failure) => Some(failure),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NonStereoWriterState<S> {
    surface: PreparedNonStereo,
    structural: WriterState<S>,
    labels: RingLabels,
    pending: Option<PendingEmission>,
}

impl<S: ConstraintSolver> NonStereoWriterState<S> {
    pub(crate) fn initial(surface: &PreparedNonStereo) -> Result<Consistency<Self>, S::Failure> {
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

    pub(crate) fn choices(&self) -> Result<Vec<Choice<Self>>, ChoiceFailure<S::Failure>> {
        if self.is_accepted() {
            return Ok(Vec::new());
        }
        if let Some(pending) = self.pending {
            return match self.attempt_pending(pending) {
                CandidateAttempt::Accepted { text, successor } => {
                    Ok(vec![Choice { text, successor }])
                }
                CandidateAttempt::Rejected { reason } => match reason {
                    CandidateRejection::Contradiction => Err(ChoiceFailure::Invariant(
                        WriterInvariantFailure::PendingEmissionRejected,
                    )),
                    CandidateRejection::RingLabelUnavailable {
                        next_label,
                        maximum_label,
                    } => Err(ChoiceFailure::Spelling(
                        SpellingFailure::RingLabelExhausted {
                            next_label,
                            maximum_label,
                            blocked_candidate_count: 1,
                        },
                    )),
                },
                CandidateAttempt::Failed(failure) => Err(ChoiceFailure::Backend(failure)),
            };
        }

        let batch = self.structural.derive_candidates();
        if batch.is_contradiction() {
            return Err(ChoiceFailure::Invariant(
                WriterInvariantFailure::StructuralContradiction,
            ));
        }
        if batch.candidates().is_empty() {
            return Err(ChoiceFailure::Invariant(
                WriterInvariantFailure::NoStructuralCandidates,
            ));
        }
        let mut choices = Vec::new();
        let mut first_unavailable_label = None;
        let mut spelling_rejection_count = 0;
        let mut semantic_rejection_count = 0;
        for &candidate in batch.candidates() {
            if candidate == StructuralCandidate::FinishComponent {
                panic!("component completion must already be normalized");
            }
            match self.attempt_structural(candidate) {
                CandidateAttempt::Accepted { text, successor } => {
                    choices.push(Choice { text, successor });
                }
                CandidateAttempt::Rejected { reason } => match reason {
                    CandidateRejection::Contradiction => {
                        semantic_rejection_count += 1;
                    }
                    CandidateRejection::RingLabelUnavailable {
                        next_label,
                        maximum_label,
                    } => {
                        first_unavailable_label.get_or_insert((next_label, maximum_label));
                        spelling_rejection_count += 1;
                    }
                },
                CandidateAttempt::Failed(failure) => {
                    return Err(ChoiceFailure::Backend(failure));
                }
            }
        }
        if !choices.is_empty() {
            return Ok(choices);
        }
        match first_unavailable_label {
            Some((next_label, maximum_label)) => Err(ChoiceFailure::Spelling(
                SpellingFailure::RingLabelExhausted {
                    next_label,
                    maximum_label,
                    blocked_candidate_count: spelling_rejection_count,
                },
            )),
            None => {
                assert_eq!(
                    semantic_rejection_count,
                    batch.candidates().len(),
                    "every unaccepted candidate must have a classified rejection"
                );
                Err(ChoiceFailure::Invariant(
                    WriterInvariantFailure::AllCandidatesSemanticallyRejected {
                        candidate_count: semantic_rejection_count,
                    },
                ))
            }
        }
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
                if self.structural.has_visited_atoms() {
                    self.finish_attempt(
                        ".".to_owned(),
                        Self {
                            surface: self.surface.clone(),
                            structural,
                            labels: self.labels.clone(),
                            pending: Some(PendingEmission::ComponentRootAtom(atom)),
                        },
                    )
                } else {
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
            }
            StructuralCandidate::RingOpen { incident } => {
                let label_slot = self.labels.next_available();
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
                let successor = Self {
                    surface: self.surface.clone(),
                    structural,
                    labels,
                    pending: None,
                };
                match successor.normalize_and_check() {
                    Ok(Consistency::Consistent(successor)) => {
                        let Some(text) = self.labels.next_label_text(label_slot) else {
                            return CandidateAttempt::Rejected {
                                reason: CandidateRejection::RingLabelUnavailable {
                                    next_label: label_slot.index() + 1,
                                    maximum_label: self.labels.maximum_spelling_label(),
                                },
                            };
                        };
                        CandidateAttempt::Accepted { text, successor }
                    }
                    Ok(Consistency::Contradiction) => CandidateAttempt::Rejected {
                        reason: CandidateRejection::Contradiction,
                    },
                    Err(failure) => CandidateAttempt::Failed(failure),
                }
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
            StructuralCandidate::CloseBranch => {
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
            StructuralCandidate::FinishComponent => {
                panic!("top-level completion is normalized without a visible token")
            }
        }
    }

    fn attempt_pending(&self, pending: PendingEmission) -> CandidateAttempt<Self, S::Failure> {
        match pending {
            PendingEmission::ComponentRootAtom(atom) => self.finish_attempt(
                self.surface.atom_text(atom).to_owned(),
                Self {
                    surface: self.surface.clone(),
                    structural: self.structural.clone(),
                    labels: self.labels.clone(),
                    pending: None,
                },
            ),
            PendingEmission::InlineAtom(incident) => self.finish_attempt(
                self.surface.atom_text(incident.atom()).to_owned(),
                Self {
                    surface: self.surface.clone(),
                    structural: self.structural.enter_committed_inline_child(incident),
                    labels: self.labels.clone(),
                    pending: None,
                },
            ),
            PendingEmission::BranchBondOrAtom(incident) => {
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
            PendingEmission::BranchAtom(incident) => self.finish_attempt(
                self.surface.atom_text(incident.atom()).to_owned(),
                Self {
                    surface: self.surface.clone(),
                    structural: self.structural.enter_committed_branch_child(incident),
                    labels: self.labels.clone(),
                    pending: None,
                },
            ),
            PendingEmission::RingClosureLabel {
                incident,
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
        loop {
            if self.is_accepted() {
                return Ok(Consistency::Consistent(self));
            }
            let batch = self.structural.derive_candidates();
            if batch.is_contradiction() || batch.candidates().is_empty() {
                return Ok(Consistency::Contradiction);
            }
            if batch.candidates() != [StructuralCandidate::FinishComponent] {
                return Ok(Consistency::Consistent(self));
            }
            assert!(
                !self.labels.has_open_labels(),
                "a component cannot finish with open visible ring labels"
            );
            self.structural = match self
                .structural
                .attempt_candidate(StructuralCandidate::FinishComponent)?
            {
                Consistency::Consistent(completed) => completed,
                Consistency::Contradiction => {
                    panic!("top-level structural completion cannot contradict the CSP")
                }
            };
            assert_eq!(
                self.structural.active_atom(),
                None,
                "component completion must not restore a branch parent"
            );
        }
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

#[cfg(test)]
#[path = "nonstereo_writer_tests.rs"]
mod tests;
