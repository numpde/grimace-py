//! Graph-general non-stereo visible-token state.
//!
//! Graph and constraint semantics live in `WriterState`. This module owns only
//! concrete non-stereo spelling facts, live ring-label assignments, and the
//! small lexical commitments forced by multi-token SMILES constructs.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::{AtomId, BondId};
use crate::model::EdgeRolePartition;
use crate::prepared::{AdjacentBond, PreparedBond, PreparedMolecule};
use crate::solver::{Consistency, ConstraintSolver};
#[cfg(test)]
use crate::writer_state::ObservedWriterState;
use crate::writer_state::{StructuralCandidate, WriterState};

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum BondRepresentation {
    Traversal = 0,
    Ring00 = 1,
    Ring10 = 2,
    Ring01 = 3,
    Ring11 = 4,
}

impl BondRepresentation {
    const fn value_index(self) -> u8 {
        self as u8
    }

    const fn singleton_domain(self) -> Domain {
        Domain::from_bits(1_u64 << self.value_index())
    }

    const fn role_partition() -> EdgeRolePartition {
        EdgeRolePartition::new(
            Self::Traversal.singleton_domain(),
            Domain::from_bits(
                (1_u64 << Self::Ring00.value_index())
                    | (1_u64 << Self::Ring10.value_index())
                    | (1_u64 << Self::Ring01.value_index())
                    | (1_u64 << Self::Ring11.value_index()),
            ),
        )
    }

    const fn elided_domain() -> Domain {
        Self::Traversal
            .singleton_domain()
            .union(Self::Ring00.singleton_domain())
    }

    const fn explicit_domain() -> Domain {
        Self::Traversal
            .singleton_domain()
            .union(Self::Ring10.singleton_domain())
            .union(Self::Ring01.singleton_domain())
            .union(Self::Ring11.singleton_domain())
    }

    const fn endpoint_domain(
        endpoint: FixedBondEndpoint,
        spelling: RingEndpointSpelling,
    ) -> Domain {
        match (endpoint, spelling) {
            (FixedBondEndpoint::A, RingEndpointSpelling::Omit) => Self::Ring00
                .singleton_domain()
                .union(Self::Ring01.singleton_domain()),
            (FixedBondEndpoint::A, RingEndpointSpelling::Emit) => Self::Ring10
                .singleton_domain()
                .union(Self::Ring11.singleton_domain()),
            (FixedBondEndpoint::B, RingEndpointSpelling::Omit) => Self::Ring00
                .singleton_domain()
                .union(Self::Ring10.singleton_domain()),
            (FixedBondEndpoint::B, RingEndpointSpelling::Emit) => Self::Ring01
                .singleton_domain()
                .union(Self::Ring11.singleton_domain()),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum FixedBondEndpoint {
    A,
    B,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum RingEndpointSpelling {
    Omit,
    Emit,
}

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
    const fn representation_domain(self) -> Domain {
        match self {
            Self::Elided => BondRepresentation::elided_domain(),
            Self::Aromatic
            | Self::Single
            | Self::Double
            | Self::Triple
            | Self::DativeAToB
            | Self::DativeBToA => BondRepresentation::explicit_domain(),
        }
    }

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
        let decision_domains = bond_tokens
            .iter()
            .copied()
            .map(NonStereoBondToken::representation_domain)
            .collect::<Vec<_>>();
        let role_partitions = vec![BondRepresentation::role_partition(); graph.bond_count()];
        let molecule =
            PreparedMolecule::with_bond_decisions(&molecule, &decision_domains, &role_partitions);

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

    fn fixed_endpoint(&self, bond: BondId, atom: AtomId) -> FixedBondEndpoint {
        let topology = self
            .molecule
            .graph()
            .bond(bond)
            .expect("prepared bond token must match the bound molecule");
        if topology.a() == atom {
            FixedBondEndpoint::A
        } else if topology.b() == atom {
            FixedBondEndpoint::B
        } else {
            panic!("ring spelling requires one fixed endpoint of the prepared bond")
        }
    }

    fn ring_endpoint_domain(
        &self,
        bond: BondId,
        atom: AtomId,
        spelling: RingEndpointSpelling,
    ) -> Domain {
        BondRepresentation::endpoint_domain(self.fixed_endpoint(bond, atom), spelling)
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
    RingOpeningLabel {
        incident: AdjacentBond,
        label_slot: RingLabelSlot,
    },
    RingClosureLabel {
        incident: AdjacentBond,
        label_slot: RingLabelSlot,
    },
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ObservedPending {
    ComponentAtom {
        root: AtomId,
    },
    BranchBondOrAtom {
        parent: AtomId,
        child: AtomId,
        bond: BondId,
    },
    BranchAtom {
        parent: AtomId,
        child: AtomId,
        bond: BondId,
    },
    InlineAtom {
        parent: AtomId,
        child: AtomId,
        bond: BondId,
    },
    RingOpeningLabel {
        bond: BondId,
        endpoint: AtomId,
        label: usize,
    },
    RingClosureLabel {
        bond: BondId,
        endpoint: AtomId,
        label: usize,
    },
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ObservedNonStereoState {
    pub(crate) structural: ObservedWriterState,
    pub(crate) labels_by_bond: Vec<(BondId, usize)>,
    pub(crate) pending: Option<ObservedPending>,
    pub(crate) maximum_spelling_label: usize,
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

enum SuccessorAttempt<S, E> {
    Accepted(S),
    Rejected(CandidateRejection),
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

    #[cfg(test)]
    pub(crate) fn observe_raw(&self) -> ObservedNonStereoState {
        let structural = self.structural.observe_raw();
        let active = structural
            .traversal
            .active_frame
            .as_ref()
            .map(|frame| frame.atom);
        let active_endpoint = || active.expect("observed pending syntax requires an active atom");
        let pending = self.pending.map(|pending| match pending {
            PendingEmission::ComponentRootAtom(root) => ObservedPending::ComponentAtom { root },
            PendingEmission::InlineAtom(incident) => ObservedPending::InlineAtom {
                parent: active_endpoint(),
                child: incident.atom(),
                bond: incident.bond(),
            },
            PendingEmission::BranchBondOrAtom(incident) => ObservedPending::BranchBondOrAtom {
                parent: active_endpoint(),
                child: incident.atom(),
                bond: incident.bond(),
            },
            PendingEmission::BranchAtom(incident) => ObservedPending::BranchAtom {
                parent: active_endpoint(),
                child: incident.atom(),
                bond: incident.bond(),
            },
            PendingEmission::RingOpeningLabel {
                incident,
                label_slot,
            } => ObservedPending::RingOpeningLabel {
                bond: incident.bond(),
                endpoint: active_endpoint(),
                label: label_slot.index(),
            },
            PendingEmission::RingClosureLabel {
                incident,
                label_slot,
            } => ObservedPending::RingClosureLabel {
                bond: incident.bond(),
                endpoint: active_endpoint(),
                label: label_slot.index(),
            },
        });
        let labels_by_bond = self
            .labels
            .bonds_by_slot
            .iter()
            .map(|(slot, bond)| (*bond, slot.index()))
            .collect();
        ObservedNonStereoState {
            structural,
            labels_by_bond,
            pending,
            maximum_spelling_label: self.labels.maximum_spelling_label(),
        }
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
        let mut attempted_choice_count = 0;
        for &candidate in batch.candidates() {
            if candidate == StructuralCandidate::FinishComponent {
                panic!("component completion must already be normalized");
            }
            let attempts = match candidate {
                StructuralCandidate::RingOpen { incident } => self
                    .attempt_ring_openings(candidate, incident)
                    .map_err(ChoiceFailure::Backend)?,
                StructuralCandidate::RingClose { incident, .. } => self
                    .attempt_ring_closures(candidate, incident)
                    .map_err(ChoiceFailure::Backend)?,
                _ => vec![self.attempt_structural(candidate)],
            };
            for attempt in attempts {
                attempted_choice_count += 1;
                match attempt {
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
                    semantic_rejection_count, attempted_choice_count,
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

    fn attempt_ring_openings(
        &self,
        candidate: StructuralCandidate,
        incident: AdjacentBond,
    ) -> Result<Vec<CandidateAttempt<Self, S::Failure>>, S::Failure> {
        let active = self
            .structural
            .active_atom()
            .expect("ring opening spelling requires an active endpoint");
        let current = self.structural.bond_decision_domain(incident.bond());
        let mut attempts = Vec::new();
        for spelling in [RingEndpointSpelling::Omit, RingEndpointSpelling::Emit] {
            let allowed = current.intersect(self.surface.ring_endpoint_domain(
                incident.bond(),
                active,
                spelling,
            ));
            if allowed.is_empty() {
                continue;
            }
            let structural = match self
                .structural
                .attempt_candidate_with_bond_refinement(candidate, allowed)?
            {
                Consistency::Consistent(structural) => structural,
                Consistency::Contradiction => {
                    attempts.push(CandidateAttempt::Rejected {
                        reason: CandidateRejection::Contradiction,
                    });
                    continue;
                }
            };
            let label_slot = self.labels.next_available();
            let mut labels = self.labels.clone();
            let allocated = labels.allocate(incident.bond());
            assert_eq!(
                allocated, label_slot,
                "advertised ring label must match the allocated label"
            );
            let pending = match spelling {
                RingEndpointSpelling::Omit => None,
                RingEndpointSpelling::Emit => Some(PendingEmission::RingOpeningLabel {
                    incident,
                    label_slot,
                }),
            };
            let successor = Self {
                surface: self.surface.clone(),
                structural,
                labels,
                pending,
            };
            let attempt = match spelling {
                RingEndpointSpelling::Omit => match successor.normalize_and_check() {
                    SuccessorAttempt::Accepted(successor) => {
                        let Some(text) = self.labels.next_label_text(label_slot) else {
                            attempts.push(CandidateAttempt::Rejected {
                                reason: CandidateRejection::RingLabelUnavailable {
                                    next_label: label_slot.index() + 1,
                                    maximum_label: self.labels.maximum_spelling_label(),
                                },
                            });
                            continue;
                        };
                        CandidateAttempt::Accepted { text, successor }
                    }
                    SuccessorAttempt::Rejected(reason) => CandidateAttempt::Rejected { reason },
                    SuccessorAttempt::Failed(failure) => return Err(failure),
                },
                RingEndpointSpelling::Emit => {
                    let text = self.surface.bond_text(incident.bond(), active);
                    assert!(
                        !text.is_empty(),
                        "an emitted ring endpoint must have prepared bond text"
                    );
                    self.finish_attempt(text.to_owned(), successor)
                }
            };
            match attempt {
                CandidateAttempt::Failed(failure) => return Err(failure),
                attempt => attempts.push(attempt),
            }
        }
        Ok(attempts)
    }

    fn attempt_ring_closures(
        &self,
        candidate: StructuralCandidate,
        incident: AdjacentBond,
    ) -> Result<Vec<CandidateAttempt<Self, S::Failure>>, S::Failure> {
        let active = self
            .structural
            .active_atom()
            .expect("ring closure spelling requires an active endpoint");
        let current = self.structural.bond_decision_domain(incident.bond());
        let label_slot = self.labels.slot_for_bond(incident.bond());
        let mut attempts = Vec::new();
        for spelling in [RingEndpointSpelling::Omit, RingEndpointSpelling::Emit] {
            let allowed = current.intersect(self.surface.ring_endpoint_domain(
                incident.bond(),
                active,
                spelling,
            ));
            if allowed.is_empty() {
                continue;
            }
            assert!(
                allowed.is_singleton(),
                "opening and closure projections must resolve one representation plan"
            );
            let structural = match self
                .structural
                .attempt_candidate_with_bond_refinement(candidate, allowed)?
            {
                Consistency::Consistent(structural) => structural,
                Consistency::Contradiction => {
                    attempts.push(CandidateAttempt::Rejected {
                        reason: CandidateRejection::Contradiction,
                    });
                    continue;
                }
            };
            assert_eq!(
                structural.bond_decision_domain(incident.bond()),
                allowed,
                "a closed ring must retain one resolved representation plan"
            );
            let attempt = match spelling {
                RingEndpointSpelling::Omit => {
                    let mut labels = self.labels.clone();
                    labels.release(label_slot, incident.bond());
                    let successor = Self {
                        surface: self.surface.clone(),
                        structural,
                        labels,
                        pending: None,
                    };
                    match successor.normalize_and_check() {
                        SuccessorAttempt::Accepted(successor) => {
                            let Some(text) = self.labels.next_label_text(label_slot) else {
                                attempts.push(CandidateAttempt::Rejected {
                                    reason: CandidateRejection::RingLabelUnavailable {
                                        next_label: label_slot.index() + 1,
                                        maximum_label: self.labels.maximum_spelling_label(),
                                    },
                                });
                                continue;
                            };
                            CandidateAttempt::Accepted { text, successor }
                        }
                        SuccessorAttempt::Rejected(reason) => CandidateAttempt::Rejected { reason },
                        SuccessorAttempt::Failed(failure) => {
                            return Err(failure);
                        }
                    }
                }
                RingEndpointSpelling::Emit => {
                    let text = self.surface.bond_text(incident.bond(), active);
                    assert!(
                        !text.is_empty(),
                        "an emitted ring endpoint must have prepared bond text"
                    );
                    self.finish_attempt(
                        text.to_owned(),
                        Self {
                            surface: self.surface.clone(),
                            structural,
                            labels: self.labels.clone(),
                            pending: Some(PendingEmission::RingClosureLabel {
                                incident,
                                label_slot,
                            }),
                        },
                    )
                }
            };
            match attempt {
                CandidateAttempt::Failed(failure) => return Err(failure),
                attempt => attempts.push(attempt),
            }
        }
        Ok(attempts)
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
                panic!("ring openings expand into endpoint-spelling candidates: {incident:?}")
            }
            StructuralCandidate::RingClose { incident, .. } => {
                panic!("ring closures expand into endpoint-spelling candidates: {incident:?}")
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
            PendingEmission::RingOpeningLabel {
                incident,
                label_slot,
            } => {
                assert_eq!(
                    self.labels.slot_for_bond(incident.bond()),
                    label_slot,
                    "a pending opening label must retain its assignment"
                );
                self.finish_ring_label_attempt(
                    label_slot,
                    Self {
                        surface: self.surface.clone(),
                        structural: self.structural.clone(),
                        labels: self.labels.clone(),
                        pending: None,
                    },
                )
            }
            PendingEmission::RingClosureLabel {
                incident,
                label_slot,
            } => {
                assert_eq!(
                    self.labels.slot_for_bond(incident.bond()),
                    label_slot,
                    "a pending ring label must retain its assignment"
                );
                let mut labels = self.labels.clone();
                labels.release(label_slot, incident.bond());
                self.finish_ring_label_attempt(
                    label_slot,
                    Self {
                        surface: self.surface.clone(),
                        structural: self.structural.clone(),
                        labels,
                        pending: None,
                    },
                )
            }
        }
    }

    fn finish_ring_label_attempt(
        &self,
        label_slot: RingLabelSlot,
        successor: Self,
    ) -> CandidateAttempt<Self, S::Failure> {
        match successor.normalize_and_check() {
            SuccessorAttempt::Accepted(successor) => {
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
            SuccessorAttempt::Rejected(reason) => CandidateAttempt::Rejected { reason },
            SuccessorAttempt::Failed(failure) => CandidateAttempt::Failed(failure),
        }
    }

    fn finish_attempt(&self, text: String, successor: Self) -> CandidateAttempt<Self, S::Failure> {
        match successor.normalize_and_check() {
            SuccessorAttempt::Accepted(successor) => CandidateAttempt::Accepted { text, successor },
            SuccessorAttempt::Rejected(reason) => CandidateAttempt::Rejected { reason },
            SuccessorAttempt::Failed(failure) => CandidateAttempt::Failed(failure),
        }
    }

    fn normalize_and_check(mut self) -> SuccessorAttempt<Self, S::Failure> {
        if let Some(pending) = self.pending {
            assert!(
                self.structural.active_atom().is_some(),
                "pending text requires an active structural path"
            );
            return match self.attempt_pending(pending) {
                CandidateAttempt::Accepted { .. } => SuccessorAttempt::Accepted(self),
                CandidateAttempt::Rejected { reason } => SuccessorAttempt::Rejected(reason),
                CandidateAttempt::Failed(failure) => SuccessorAttempt::Failed(failure),
            };
        }
        loop {
            if self.is_accepted() {
                return SuccessorAttempt::Accepted(self);
            }
            let batch = self.structural.derive_candidates();
            if batch.is_contradiction() || batch.candidates().is_empty() {
                return SuccessorAttempt::Rejected(CandidateRejection::Contradiction);
            }
            if batch.candidates() != [StructuralCandidate::FinishComponent] {
                return SuccessorAttempt::Accepted(self);
            }
            assert!(
                !self.labels.has_open_labels(),
                "a component cannot finish with open visible ring labels"
            );
            self.structural = match self
                .structural
                .attempt_candidate(StructuralCandidate::FinishComponent)
            {
                Ok(Consistency::Consistent(completed)) => completed,
                Ok(Consistency::Contradiction) => {
                    panic!("top-level structural completion cannot contradict the CSP")
                }
                Err(failure) => return SuccessorAttempt::Failed(failure),
            };
            assert_eq!(
                self.structural.active_atom(),
                None,
                "component completion must not restore a branch parent"
            );
        }
    }
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
