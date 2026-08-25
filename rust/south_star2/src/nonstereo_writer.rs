//! Graph-general non-stereo visible-token state.
//!
//! Graph and constraint semantics live in `WriterState`. This module owns only
//! concrete non-stereo spelling facts, live ring-label assignments, and the
//! small lexical commitments forced by multi-token SMILES constructs.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::{AtomId, BondId, VariableId};
use crate::model::EdgeRolePartition;
use crate::prepared::{AdjacentBond, PreparedBond, PreparedMolecule};
use crate::solver::{Consistency, ConstraintSolver};
use crate::tetrahedral::{
    full_order_domain, parity_domain, prefix_domain, singleton_order, TetrahedralLigand,
    TetrahedralParity,
};
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
    atoms: Arc<[PreparedAtom]>,
    bond_tokens: Arc<[NonStereoBondToken]>,
}

#[derive(Clone, Debug)]
pub(crate) enum PreparedAtomToken {
    Fixed(String),
    Tetrahedral {
        reference_order: [TetrahedralLigand; 4],
        text_by_parity: [String; 2],
    },
}

#[derive(Clone, Debug)]
enum PreparedAtom {
    Fixed(Box<str>),
    Tetrahedral(PreparedTetrahedralCenter),
}

#[derive(Clone, Debug)]
struct PreparedTetrahedralCenter {
    reference_order: [TetrahedralLigand; 4],
    text_by_parity: [Box<str>; 2],
    order_variable: VariableId,
}

impl PreparedTetrahedralCenter {
    fn context_prefix(&self, entry_bond: Option<BondId>) -> Vec<TetrahedralLigand> {
        let mut prefix = Vec::with_capacity(2);
        if let Some(bond) = entry_bond {
            prefix.push(TetrahedralLigand::Bond(bond));
        }
        if self
            .reference_order
            .contains(&TetrahedralLigand::VirtualHydrogen)
        {
            prefix.push(TetrahedralLigand::VirtualHydrogen);
        }
        prefix
    }

    fn token_domain(&self, entry_bond: Option<BondId>, parity: TetrahedralParity) -> Domain {
        prefix_domain(&self.reference_order, &self.context_prefix(entry_bond))
            .intersect(parity_domain(parity))
    }

    fn prefix_domain_with_bond_order(
        &self,
        entry_bond: Option<BondId>,
        committed_bonds: &[BondId],
    ) -> Domain {
        let mut prefix = self.context_prefix(entry_bond);
        prefix.extend(committed_bonds.iter().copied().map(TetrahedralLigand::Bond));
        prefix_domain(&self.reference_order, &prefix)
    }

    fn completed_order_domain(
        &self,
        entry_bond: Option<BondId>,
        committed_bonds: &[BondId],
    ) -> Domain {
        let mut order = self.context_prefix(entry_bond);
        order.extend(committed_bonds.iter().copied().map(TetrahedralLigand::Bond));
        singleton_order(&self.reference_order, &order)
    }
}

impl PreparedNonStereo {
    pub(crate) fn new(
        molecule: PreparedMolecule,
        atom_text: Vec<String>,
        bond_tokens: Vec<NonStereoBondToken>,
    ) -> Result<Self, PreparedNonStereoError> {
        Self::with_atom_tokens(
            molecule,
            atom_text
                .into_iter()
                .map(PreparedAtomToken::Fixed)
                .collect(),
            bond_tokens,
        )
    }

    pub(crate) fn with_atom_tokens(
        molecule: PreparedMolecule,
        atoms: Vec<PreparedAtomToken>,
        bond_tokens: Vec<NonStereoBondToken>,
    ) -> Result<Self, PreparedNonStereoError> {
        let graph = molecule.graph();
        if atoms.len() != graph.atom_count() {
            return Err(PreparedNonStereoError::AtomTextCountMismatch {
                expected: graph.atom_count(),
                actual: atoms.len(),
            });
        }
        if bond_tokens.len() != graph.bond_count() {
            return Err(PreparedNonStereoError::BondTokenCountMismatch {
                expected: graph.bond_count(),
                actual: bond_tokens.len(),
            });
        }
        for (atom, prepared) in graph.atom_ids().zip(&atoms) {
            validate_prepared_atom(graph, atom, prepared)?;
        }
        let decision_domains = bond_tokens
            .iter()
            .copied()
            .map(NonStereoBondToken::representation_domain)
            .collect::<Vec<_>>();
        let role_partitions = vec![BondRepresentation::role_partition(); graph.bond_count()];
        let mut assembly =
            PreparedMolecule::constraint_assembly(&molecule, &decision_domains, &role_partitions);
        let atoms = atoms
            .into_iter()
            .map(|prepared| match prepared {
                PreparedAtomToken::Fixed(text) => PreparedAtom::Fixed(text.into_boxed_str()),
                PreparedAtomToken::Tetrahedral {
                    reference_order,
                    text_by_parity,
                } => PreparedAtom::Tetrahedral(PreparedTetrahedralCenter {
                    reference_order,
                    text_by_parity: text_by_parity.map(String::into_boxed_str),
                    order_variable: assembly.add_isolated_variable(full_order_domain()),
                }),
            })
            .collect::<Vec<_>>();
        let molecule = assembly.finish();

        Ok(Self {
            molecule,
            atoms: Arc::from(atoms.into_boxed_slice()),
            bond_tokens: Arc::from(bond_tokens.into_boxed_slice()),
        })
    }

    fn molecule(&self) -> &PreparedMolecule {
        &self.molecule
    }

    fn atom_text(&self, atom: AtomId) -> &str {
        let prepared = self
            .atoms
            .get(atom.index())
            .expect("prepared atom text must match the bound molecule");
        match prepared {
            PreparedAtom::Fixed(text) => text,
            PreparedAtom::Tetrahedral(_) => {
                panic!("tetrahedral atom text requires a parity choice")
            }
        }
    }

    fn tetrahedral_center(&self, atom: AtomId) -> Option<&PreparedTetrahedralCenter> {
        match self.atoms.get(atom.index())? {
            PreparedAtom::Fixed(_) => None,
            PreparedAtom::Tetrahedral(center) => Some(center),
        }
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

fn validate_prepared_atom(
    graph: &crate::prepared::PreparedGraph,
    atom: AtomId,
    prepared: &PreparedAtomToken,
) -> Result<(), PreparedNonStereoError> {
    match prepared {
        PreparedAtomToken::Fixed(text) => {
            if text.is_empty() {
                return Err(PreparedNonStereoError::EmptyAtomText(atom));
            }
        }
        PreparedAtomToken::Tetrahedral {
            reference_order,
            text_by_parity,
        } => {
            if text_by_parity.iter().any(String::is_empty) {
                return Err(PreparedNonStereoError::EmptyTetrahedralAtomText(atom));
            }
            if text_by_parity[0] == text_by_parity[1] {
                return Err(PreparedNonStereoError::RepeatedTetrahedralAtomText(atom));
            }
            let hydrogen_count = reference_order
                .iter()
                .filter(|ligand| **ligand == TetrahedralLigand::VirtualHydrogen)
                .count();
            if hydrogen_count > 1 {
                return Err(PreparedNonStereoError::MultipleVirtualHydrogens(atom));
            }
            let ligand_set = reference_order.iter().copied().collect::<BTreeSet<_>>();
            if ligand_set.len() != reference_order.len() {
                return Err(PreparedNonStereoError::RepeatedTetrahedralLigand(atom));
            }
            let prepared_bonds = reference_order
                .iter()
                .filter_map(|ligand| match ligand {
                    TetrahedralLigand::Bond(bond) => Some(*bond),
                    TetrahedralLigand::VirtualHydrogen => None,
                })
                .collect::<BTreeSet<_>>();
            let incident_bonds = graph
                .neighbors(atom)
                .expect("prepared atom must belong to its graph")
                .iter()
                .map(|incident| incident.bond())
                .collect::<BTreeSet<_>>();
            if prepared_bonds != incident_bonds {
                return Err(PreparedNonStereoError::TetrahedralLigandsDoNotMatchGraph(
                    atom,
                ));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PreparedNonStereoError {
    AtomTextCountMismatch { expected: usize, actual: usize },
    BondTokenCountMismatch { expected: usize, actual: usize },
    EmptyAtomText(AtomId),
    EmptyTetrahedralAtomText(AtomId),
    RepeatedTetrahedralAtomText(AtomId),
    RepeatedTetrahedralLigand(AtomId),
    MultipleVirtualHydrogens(AtomId),
    TetrahedralLigandsDoNotMatchGraph(AtomId),
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
            Self::EmptyTetrahedralAtomText(atom) => write!(
                formatter,
                "prepared tetrahedral atom texts for {atom:?} must not be empty"
            ),
            Self::RepeatedTetrahedralAtomText(atom) => write!(
                formatter,
                "prepared tetrahedral atom texts for {atom:?} must be distinct"
            ),
            Self::RepeatedTetrahedralLigand(atom) => {
                write!(
                    formatter,
                    "prepared tetrahedral ligands for {atom:?} repeat"
                )
            }
            Self::MultipleVirtualHydrogens(atom) => write!(
                formatter,
                "prepared tetrahedral center {atom:?} has multiple virtual hydrogens"
            ),
            Self::TetrahedralLigandsDoNotMatchGraph(atom) => write!(
                formatter,
                "prepared tetrahedral ligands for {atom:?} do not match its graph incidences"
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PendingAtomEntry {
    AlreadyEntered,
    Inline(AdjacentBond),
    Branch(AdjacentBond),
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
    pub(crate) tetrahedral_order_domains: Vec<(AtomId, Domain)>,
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
    Incomplete(WriterInvariantFailure),
    Failed(E),
}

enum SuccessorAttempt<S, E> {
    Accepted(S),
    Rejected(CandidateRejection),
    Incomplete(WriterInvariantFailure),
    Failed(E),
}

fn collect_attempts_fail_fast<S, E>(
    attempts: impl IntoIterator<Item = CandidateAttempt<S, E>>,
) -> Vec<CandidateAttempt<S, E>> {
    let mut collected = Vec::new();
    for attempt in attempts {
        let stop = matches!(
            attempt,
            CandidateAttempt::Incomplete(_) | CandidateAttempt::Failed(_)
        );
        collected.push(attempt);
        if stop {
            break;
        }
    }
    collected
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
    TetrahedralRingCouplingUnimplemented { atom: AtomId },
    UnresolvedTetrahedralFrame { atom: AtomId },
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
            Self::TetrahedralRingCouplingUnimplemented { atom } => write!(
                formatter,
                "tetrahedral atom {atom:?} still has a ring-capable incident bond"
            ),
            Self::UnresolvedTetrahedralFrame { atom } => write!(
                formatter,
                "tetrahedral atom {atom:?} completed without one exact procedural ligand order"
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
        let tetrahedral_order_domains = self
            .surface
            .molecule()
            .graph()
            .atom_ids()
            .filter_map(|atom| {
                self.surface
                    .tetrahedral_center(atom)
                    .map(|center| (atom, self.structural.semantic_domain(center.order_variable)))
            })
            .collect();
        ObservedNonStereoState {
            structural,
            tetrahedral_order_domains,
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
            let attempts = self.pending_attempts(pending);
            let mut choices = Vec::new();
            let mut unavailable_label = None;
            let mut spelling_rejection_count = 0;
            for attempt in attempts {
                match attempt {
                    CandidateAttempt::Accepted { text, successor } => {
                        choices.push(Choice { text, successor });
                    }
                    CandidateAttempt::Rejected { reason } => match reason {
                        CandidateRejection::Contradiction => {}
                        CandidateRejection::RingLabelUnavailable {
                            next_label,
                            maximum_label,
                        } => {
                            unavailable_label.get_or_insert((next_label, maximum_label));
                            spelling_rejection_count += 1;
                        }
                    },
                    CandidateAttempt::Incomplete(failure) => {
                        return Err(ChoiceFailure::Invariant(failure));
                    }
                    CandidateAttempt::Failed(failure) => {
                        return Err(ChoiceFailure::Backend(failure));
                    }
                }
            }
            if !choices.is_empty() {
                return Ok(choices);
            }
            return match unavailable_label {
                Some((next_label, maximum_label)) => Err(ChoiceFailure::Spelling(
                    SpellingFailure::RingLabelExhausted {
                        next_label,
                        maximum_label,
                        blocked_candidate_count: spelling_rejection_count,
                    },
                )),
                None => Err(ChoiceFailure::Invariant(
                    WriterInvariantFailure::PendingEmissionRejected,
                )),
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
                _ => self.attempt_structural(candidate),
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
                    CandidateAttempt::Incomplete(failure) => {
                        return Err(ChoiceFailure::Invariant(failure));
                    }
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
                    SuccessorAttempt::Incomplete(failure) => CandidateAttempt::Incomplete(failure),
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
                CandidateAttempt::Incomplete(_) => {
                    attempts.push(attempt);
                    break;
                }
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
                        SuccessorAttempt::Incomplete(failure) => {
                            CandidateAttempt::Incomplete(failure)
                        }
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
                CandidateAttempt::Incomplete(_) => {
                    attempts.push(attempt);
                    break;
                }
                attempt => attempts.push(attempt),
            }
        }
        Ok(attempts)
    }

    fn atom_token_specs(
        &self,
        atom: AtomId,
        entry_bond: Option<BondId>,
    ) -> Vec<(String, Vec<(VariableId, Domain)>)> {
        let Some(center) = self.surface.tetrahedral_center(atom) else {
            return vec![(self.surface.atom_text(atom).to_owned(), Vec::new())];
        };
        TetrahedralParity::ALL
            .into_iter()
            .map(|parity| {
                (
                    center.text_by_parity[parity.index()].to_string(),
                    vec![(
                        center.order_variable,
                        center.token_domain(entry_bond, parity),
                    )],
                )
            })
            .collect()
    }

    fn parent_prefix_restriction(&self, incident: AdjacentBond) -> Vec<(VariableId, Domain)> {
        let local = self.structural.active_local_bond_order();
        let Some(center) = self.surface.tetrahedral_center(local.atom) else {
            return Vec::new();
        };
        let mut committed_bonds = local.committed_bonds;
        committed_bonds.push(incident.bond());
        vec![(
            center.order_variable,
            center.prefix_domain_with_bond_order(local.entry_bond, &committed_bonds),
        )]
    }

    fn attempt_candidate_with_restrictions(
        &self,
        candidate: StructuralCandidate,
        restrictions: &[(VariableId, Domain)],
    ) -> Result<Consistency<WriterState<S>>, S::Failure> {
        if restrictions.is_empty() {
            self.structural.attempt_candidate(candidate)
        } else {
            self.structural
                .attempt_candidate_with_semantic_restrictions(candidate, restrictions)
        }
    }

    fn restrict_semantics(
        &self,
        restrictions: &[(VariableId, Domain)],
    ) -> Result<Consistency<WriterState<S>>, S::Failure> {
        if restrictions.is_empty() {
            Ok(Consistency::Consistent(self.structural.clone()))
        } else {
            self.structural.restricted_semantics(restrictions)
        }
    }

    fn validate_tetrahedral_traversal_scope(
        &self,
        structural: &WriterState<S>,
        atom: AtomId,
    ) -> Result<(), WriterInvariantFailure> {
        if self.surface.tetrahedral_center(atom).is_none() {
            return Ok(());
        }
        for incident in self
            .surface
            .molecule()
            .graph()
            .neighbors(atom)
            .expect("prepared tetrahedral atom must belong to its graph")
        {
            let partition = self
                .surface
                .molecule()
                .bond_role_partition(incident.bond())
                .expect("prepared tetrahedral incidence must have a role partition");
            let current = structural.bond_decision_domain(incident.bond());
            if !current.is_subset_of(partition.traversal_values()) {
                return Err(WriterInvariantFailure::TetrahedralRingCouplingUnimplemented { atom });
            }
        }
        Ok(())
    }

    fn validate_active_tetrahedral_completion(
        &self,
        structural: &WriterState<S>,
    ) -> Result<(), WriterInvariantFailure> {
        let local = structural.active_local_bond_order();
        let Some(center) = self.surface.tetrahedral_center(local.atom) else {
            return Ok(());
        };
        let expected = center.completed_order_domain(local.entry_bond, &local.committed_bonds);
        if expected.is_empty() || structural.semantic_domain(center.order_variable) != expected {
            return Err(WriterInvariantFailure::UnresolvedTetrahedralFrame { atom: local.atom });
        }
        Ok(())
    }

    fn attempt_structural(
        &self,
        candidate: StructuralCandidate,
    ) -> Vec<CandidateAttempt<Self, S::Failure>> {
        match candidate {
            StructuralCandidate::Root { atom } => {
                assert!(
                    self.labels.is_clean(),
                    "a connected component must start with clean ring-label spelling state"
                );
                if self.structural.has_visited_atoms() {
                    let structural = match self.structural.attempt_candidate(candidate) {
                        Ok(Consistency::Consistent(structural)) => structural,
                        Ok(Consistency::Contradiction) => {
                            return vec![CandidateAttempt::Rejected {
                                reason: CandidateRejection::Contradiction,
                            }];
                        }
                        Err(failure) => return vec![CandidateAttempt::Failed(failure)],
                    };
                    vec![self.finish_attempt(
                        ".".to_owned(),
                        Self {
                            surface: self.surface.clone(),
                            structural,
                            labels: self.labels.clone(),
                            pending: Some(PendingEmission::ComponentRootAtom(atom)),
                        },
                    )]
                } else {
                    collect_attempts_fail_fast(self.atom_token_specs(atom, None).into_iter().map(
                        |(text, restriction)| {
                            let structural = match self.attempt_candidate_with_restrictions(
                                candidate,
                                restriction.as_slice(),
                            ) {
                                Ok(Consistency::Consistent(structural)) => structural,
                                Ok(Consistency::Contradiction) => {
                                    return CandidateAttempt::Rejected {
                                        reason: CandidateRejection::Contradiction,
                                    };
                                }
                                Err(failure) => return CandidateAttempt::Failed(failure),
                            };
                            if let Err(failure) =
                                self.validate_tetrahedral_traversal_scope(&structural, atom)
                            {
                                return CandidateAttempt::Incomplete(failure);
                            }
                            self.finish_attempt(
                                text,
                                Self {
                                    surface: self.surface.clone(),
                                    structural,
                                    labels: self.labels.clone(),
                                    pending: None,
                                },
                            )
                        },
                    ))
                }
            }
            StructuralCandidate::RingOpen { incident } => {
                panic!("ring openings expand into endpoint-spelling candidates: {incident:?}")
            }
            StructuralCandidate::RingClose { incident, .. } => {
                panic!("ring closures expand into endpoint-spelling candidates: {incident:?}")
            }
            StructuralCandidate::BranchChild { incident } => {
                let restrictions = self.parent_prefix_restriction(incident);
                let structural = match self
                    .attempt_candidate_with_restrictions(candidate, restrictions.as_slice())
                {
                    Ok(Consistency::Consistent(structural)) => structural,
                    Ok(Consistency::Contradiction) => {
                        return vec![CandidateAttempt::Rejected {
                            reason: CandidateRejection::Contradiction,
                        }];
                    }
                    Err(failure) => return vec![CandidateAttempt::Failed(failure)],
                };
                let labels = self.labels.clone();
                vec![self.finish_attempt(
                    "(".to_owned(),
                    Self {
                        surface: self.surface.clone(),
                        structural,
                        labels,
                        pending: Some(PendingEmission::BranchBondOrAtom(incident)),
                    },
                )]
            }
            StructuralCandidate::InlineChild { incident } => {
                let parent = self
                    .structural
                    .active_atom()
                    .expect("inline emission requires an active atom");
                let bond_text = self.surface.bond_text(incident.bond(), parent);
                let parent_restriction = self.parent_prefix_restriction(incident);
                if bond_text.is_empty() {
                    collect_attempts_fail_fast(
                        self.atom_token_specs(incident.atom(), Some(incident.bond()))
                            .into_iter()
                            .map(|(text, mut child_restriction)| {
                                let mut restrictions = parent_restriction.clone();
                                restrictions.append(&mut child_restriction);
                                let structural = match self.attempt_candidate_with_restrictions(
                                    candidate,
                                    restrictions.as_slice(),
                                ) {
                                    Ok(Consistency::Consistent(structural)) => structural,
                                    Ok(Consistency::Contradiction) => {
                                        return CandidateAttempt::Rejected {
                                            reason: CandidateRejection::Contradiction,
                                        };
                                    }
                                    Err(failure) => return CandidateAttempt::Failed(failure),
                                };
                                if let Err(failure) = self.validate_tetrahedral_traversal_scope(
                                    &structural,
                                    incident.atom(),
                                ) {
                                    return CandidateAttempt::Incomplete(failure);
                                }
                                if let Err(failure) =
                                    self.validate_active_tetrahedral_completion(&structural)
                                {
                                    return CandidateAttempt::Incomplete(failure);
                                }
                                self.finish_attempt(
                                    text,
                                    Self {
                                        surface: self.surface.clone(),
                                        structural: structural
                                            .enter_committed_inline_child(incident),
                                        labels: self.labels.clone(),
                                        pending: None,
                                    },
                                )
                            }),
                    )
                } else {
                    let structural = match self.attempt_candidate_with_restrictions(
                        candidate,
                        parent_restriction.as_slice(),
                    ) {
                        Ok(Consistency::Consistent(structural)) => structural,
                        Ok(Consistency::Contradiction) => {
                            return vec![CandidateAttempt::Rejected {
                                reason: CandidateRejection::Contradiction,
                            }];
                        }
                        Err(failure) => return vec![CandidateAttempt::Failed(failure)],
                    };
                    if let Err(failure) = self.validate_active_tetrahedral_completion(&structural) {
                        return vec![CandidateAttempt::Incomplete(failure)];
                    }
                    vec![self.finish_attempt(
                        bond_text.to_owned(),
                        Self {
                            surface: self.surface.clone(),
                            structural,
                            labels: self.labels.clone(),
                            pending: Some(PendingEmission::InlineAtom(incident)),
                        },
                    )]
                }
            }
            StructuralCandidate::CloseBranch => {
                if let Err(failure) = self.validate_active_tetrahedral_completion(&self.structural)
                {
                    return vec![CandidateAttempt::Incomplete(failure)];
                }
                let structural = match self.structural.attempt_candidate(candidate) {
                    Ok(Consistency::Consistent(structural)) => structural,
                    Ok(Consistency::Contradiction) => {
                        return vec![CandidateAttempt::Rejected {
                            reason: CandidateRejection::Contradiction,
                        }];
                    }
                    Err(failure) => return vec![CandidateAttempt::Failed(failure)],
                };
                let labels = self.labels.clone();
                vec![self.finish_attempt(
                    ")".to_owned(),
                    Self {
                        surface: self.surface.clone(),
                        structural,
                        labels,
                        pending: None,
                    },
                )]
            }
            StructuralCandidate::FinishComponent => {
                panic!("top-level completion is normalized without a visible token")
            }
        }
    }

    fn pending_attempts(
        &self,
        pending: PendingEmission,
    ) -> Vec<CandidateAttempt<Self, S::Failure>> {
        match pending {
            PendingEmission::ComponentRootAtom(atom) => {
                self.pending_atom_attempts(atom, None, PendingAtomEntry::AlreadyEntered)
            }
            PendingEmission::InlineAtom(incident) => self.pending_atom_attempts(
                incident.atom(),
                Some(incident.bond()),
                PendingAtomEntry::Inline(incident),
            ),
            PendingEmission::BranchBondOrAtom(incident) => {
                let parent = self
                    .structural
                    .active_atom()
                    .expect("a committed branch child requires its active parent");
                if self.surface.bond_text(incident.bond(), parent).is_empty() {
                    self.pending_atom_attempts(
                        incident.atom(),
                        Some(incident.bond()),
                        PendingAtomEntry::Branch(incident),
                    )
                } else {
                    vec![self.finish_attempt(
                        self.surface.bond_text(incident.bond(), parent).to_owned(),
                        Self {
                            surface: self.surface.clone(),
                            structural: self.structural.clone(),
                            labels: self.labels.clone(),
                            pending: Some(PendingEmission::BranchAtom(incident)),
                        },
                    )]
                }
            }
            PendingEmission::BranchAtom(incident) => self.pending_atom_attempts(
                incident.atom(),
                Some(incident.bond()),
                PendingAtomEntry::Branch(incident),
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
                vec![self.finish_ring_label_attempt(
                    label_slot,
                    Self {
                        surface: self.surface.clone(),
                        structural: self.structural.clone(),
                        labels: self.labels.clone(),
                        pending: None,
                    },
                )]
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
                vec![self.finish_ring_label_attempt(
                    label_slot,
                    Self {
                        surface: self.surface.clone(),
                        structural: self.structural.clone(),
                        labels,
                        pending: None,
                    },
                )]
            }
        }
    }

    fn pending_atom_attempts(
        &self,
        atom: AtomId,
        entry_bond: Option<BondId>,
        entry: PendingAtomEntry,
    ) -> Vec<CandidateAttempt<Self, S::Failure>> {
        collect_attempts_fail_fast(self.atom_token_specs(atom, entry_bond).into_iter().map(
            |(text, restrictions)| {
                let structural = match self.restrict_semantics(&restrictions) {
                    Ok(Consistency::Consistent(structural)) => structural,
                    Ok(Consistency::Contradiction) => {
                        return CandidateAttempt::Rejected {
                            reason: CandidateRejection::Contradiction,
                        };
                    }
                    Err(failure) => return CandidateAttempt::Failed(failure),
                };
                if let Err(failure) = self.validate_tetrahedral_traversal_scope(&structural, atom) {
                    return CandidateAttempt::Incomplete(failure);
                }
                let structural = match entry {
                    PendingAtomEntry::AlreadyEntered => structural,
                    PendingAtomEntry::Inline(incident) => {
                        structural.enter_committed_inline_child(incident)
                    }
                    PendingAtomEntry::Branch(incident) => {
                        structural.enter_committed_branch_child(incident)
                    }
                };
                self.finish_attempt(
                    text,
                    Self {
                        surface: self.surface.clone(),
                        structural,
                        labels: self.labels.clone(),
                        pending: None,
                    },
                )
            },
        ))
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
            SuccessorAttempt::Incomplete(failure) => CandidateAttempt::Incomplete(failure),
            SuccessorAttempt::Failed(failure) => CandidateAttempt::Failed(failure),
        }
    }

    fn finish_attempt(&self, text: String, successor: Self) -> CandidateAttempt<Self, S::Failure> {
        match successor.normalize_and_check() {
            SuccessorAttempt::Accepted(successor) => CandidateAttempt::Accepted { text, successor },
            SuccessorAttempt::Rejected(reason) => CandidateAttempt::Rejected { reason },
            SuccessorAttempt::Incomplete(failure) => CandidateAttempt::Incomplete(failure),
            SuccessorAttempt::Failed(failure) => CandidateAttempt::Failed(failure),
        }
    }

    fn normalize_and_check(mut self) -> SuccessorAttempt<Self, S::Failure> {
        if let Some(pending) = self.pending {
            assert!(
                self.structural.active_atom().is_some(),
                "pending text requires an active structural path"
            );
            let mut viable = false;
            let mut semantic_rejection = false;
            let mut spelling_rejection = None;
            for attempt in self.pending_attempts(pending) {
                match attempt {
                    CandidateAttempt::Accepted { .. } => viable = true,
                    CandidateAttempt::Rejected { reason } => match reason {
                        CandidateRejection::Contradiction => semantic_rejection = true,
                        unavailable @ CandidateRejection::RingLabelUnavailable { .. } => {
                            spelling_rejection.get_or_insert(unavailable);
                        }
                    },
                    CandidateAttempt::Incomplete(failure) => {
                        return SuccessorAttempt::Incomplete(failure);
                    }
                    CandidateAttempt::Failed(failure) => {
                        return SuccessorAttempt::Failed(failure);
                    }
                }
            }
            if viable {
                return SuccessorAttempt::Accepted(self);
            }
            return match spelling_rejection {
                Some(reason) => SuccessorAttempt::Rejected(reason),
                None => {
                    assert!(
                        semantic_rejection,
                        "pending frontier must classify rejection"
                    );
                    SuccessorAttempt::Rejected(CandidateRejection::Contradiction)
                }
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
            if let Err(failure) = self.validate_active_tetrahedral_completion(&self.structural) {
                return SuccessorAttempt::Incomplete(failure);
            }
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

#[cfg(test)]
#[path = "nonstereo_transition_oracle.rs"]
mod transition_oracle;
