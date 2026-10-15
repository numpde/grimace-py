//! Graph-general non-stereo visible-token state.
//!
//! Graph and constraint semantics live in `WriterState`. This module owns only
//! concrete non-stereo spelling facts, live ring-label assignments, and the
//! small lexical commitments forced by multi-token SMILES constructs.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::domain::Domain;
use crate::atom_spelling::{AromaticSpelling, AtomBondContext, AtomFacts, AtomSpellingContext, AtomSpellingField, AtomSpellingPolicy, AtomSpellingState, ElementSymbol, TetrahedralMarker};
use crate::ids::{AtomId, BondId, FactorId, VariableId};
use crate::event_schedule::{EntryKind, EventVisit};
use crate::model::{AromaticMatchingEdge, BondRole, EdgeRolePartition, FactorDefinition, ScheduleCenter, ScheduleEdge, ScheduleOrdinaryCenter, OrdinaryOrderPair, SpanningTreeEdge, TetrahedralLayoutBond};
use crate::prepared::{AdjacentBond, PreparedGraph, PreparedBond, PreparedConstraintAssembly, PreparedMolecule};
use crate::solver::{Consistency, ConstraintSolver};
use crate::tetrahedral::{
    full_order_domain, full_role_pattern_domain, layout_order_rows, TetrahedralIncidence,
    TetrahedralLigand, TetrahedralParity,
};
#[cfg(test)]
use crate::tetrahedral::{parity_domain, prefix_domain};
use crate::traversal::LocalLayoutContext;
use crate::walker::{ComponentOrder, WriterRegime};
#[cfg(test)]
use crate::writer_state::ObservedWriterState;
use crate::writer_state::{PreparedChildEntry, StructuralCandidate, WriterState};

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

fn compact_explicit_domain() -> Domain {
    BondRepresentation::Traversal.singleton_domain()
        .union(BondRepresentation::Ring10.singleton_domain())
        .union(BondRepresentation::Ring01.singleton_domain())
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

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DirectionalRingPlan {
    PlainNone = 0,
    PlainAtA = 1,
    PlainAtB = 2,
    PlainAtBoth = 3,
    SlashAtA = 4,
    SlashAtB = 5,
    BackslashAtA = 6,
    BackslashAtB = 7,
}

impl DirectionalRingPlan {
    const ALL: [Self; 8] = [
        Self::PlainNone,
        Self::PlainAtA,
        Self::PlainAtB,
        Self::PlainAtBoth,
        Self::SlashAtA,
        Self::SlashAtB,
        Self::BackslashAtA,
        Self::BackslashAtB,
    ];

    const fn value_index(self) -> u8 {
        self as u8
    }

    const fn singleton_domain(self) -> Domain {
        Domain::from_bits(1_u64 << self.value_index())
    }

    const fn mark(self) -> CarrierMark {
        match self {
            Self::PlainNone | Self::PlainAtA | Self::PlainAtB | Self::PlainAtBoth => {
                CarrierMark::Plain
            }
            Self::SlashAtA | Self::SlashAtB => CarrierMark::SlashAtFixedA,
            Self::BackslashAtA | Self::BackslashAtB => CarrierMark::BackslashAtFixedA,
        }
    }

    const fn emits_at(self, endpoint: FixedBondEndpoint) -> bool {
        match (self, endpoint) {
            (Self::PlainNone, _) => false,
            (Self::PlainAtA | Self::SlashAtA | Self::BackslashAtA, FixedBondEndpoint::A)
            | (Self::PlainAtB | Self::SlashAtB | Self::BackslashAtB, FixedBondEndpoint::B)
            | (Self::PlainAtBoth, _) => true,
            _ => false,
        }
    }

    fn domain_for(token: NonStereoBondToken) -> Domain {
        Self::ALL.into_iter().fold(Domain::empty(), |domain, plan| {
            let admitted = match plan {
                Self::PlainNone => token == NonStereoBondToken::Elided,
                Self::PlainAtA | Self::PlainAtB | Self::PlainAtBoth => {
                    token != NonStereoBondToken::Elided
                }
                Self::SlashAtA | Self::SlashAtB | Self::BackslashAtA | Self::BackslashAtB => true,
            };
            if admitted {
                domain.union(plan.singleton_domain())
            } else {
                domain
            }
        })
    }

    fn endpoint_domain(
        current: Domain,
        endpoint: FixedBondEndpoint,
        mark: Option<CarrierMark>,
        emit: bool,
    ) -> Domain {
        Self::ALL.into_iter().fold(Domain::empty(), |domain, plan| {
            if current.contains(plan.value_index())
                && plan.emits_at(endpoint) == emit
                && mark.is_none_or(|mark| plan.mark() == mark)
            {
                domain.union(plan.singleton_domain())
            } else {
                domain
            }
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NonStereoBondToken {
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

    const fn is_directional_carrier_base(self) -> bool {
        matches!(self, Self::Elided | Self::Aromatic | Self::Single)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedDirectionalCarrier {
    pub bond: BondId,
    pub side_flip: bool,
}

impl PreparedDirectionalCarrier {
    const fn unflipped(bond: BondId) -> Self {
        Self {
            bond,
            side_flip: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PreparedDirectionalRelation {
    pub double_bond: BondId,
    pub left_endpoint: AtomId,
    pub left_carriers: Box<[PreparedDirectionalCarrier]>,
    pub right_endpoint: AtomId,
    pub right_carriers: Box<[PreparedDirectionalCarrier]>,
    pub side_phase_xor: bool,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CarrierMark {
    Plain = 0,
    SlashAtFixedA = 1,
    BackslashAtFixedA = 2,
}

impl CarrierMark {
    const fn value_index(self) -> u8 {
        self as u8
    }

    const fn domain() -> Domain {
        Domain::from_bits((1_u64 << 3) - 1)
    }

    const fn marked_domain() -> Domain {
        Domain::from_bits(
            (1_u64 << Self::SlashAtFixedA.value_index())
                | (1_u64 << Self::BackslashAtFixedA.value_index()),
        )
    }

    const fn singleton_domain(self) -> Domain {
        Domain::from_bits(1_u64 << self.value_index())
    }

    const fn from_value(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Plain),
            1 => Some(Self::SlashAtFixedA),
            2 => Some(Self::BackslashAtFixedA),
            _ => None,
        }
    }

    const fn directional_text(self, from_fixed_a: bool) -> Option<&'static str> {
        match (self, from_fixed_a) {
            (Self::Plain, _) => None,
            (Self::SlashAtFixedA, true) | (Self::BackslashAtFixedA, false) => Some("/"),
            (Self::BackslashAtFixedA, true) | (Self::SlashAtFixedA, false) => Some("\\"),
        }
    }

    const fn canonical_sign(self) -> Option<bool> {
        match self {
            Self::Plain => None,
            Self::SlashAtFixedA => Some(false),
            Self::BackslashAtFixedA => Some(true),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PreparedDirectionalSite {
    mark_variable: VariableId,
    ring_plan_variable: VariableId,
    ring_plan_factor: FactorId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DirectionalSidePattern {
    marks: Box<[CarrierMark]>,
    phase: bool,
}

#[derive(Clone, Debug)]
struct PreparedDirectionalSide {
    double_bond: BondId,
    endpoint: AtomId,
    pattern_variable: VariableId,
    carriers: Box<[PreparedDirectionalCarrier]>,
    patterns: Box<[DirectionalSidePattern]>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct IncompleteDirectionalSelection {
    double_bond: BondId,
    endpoint: AtomId,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PreparedDirectionalBondStatus {
    Ordinary,
    CompiledSite(PreparedDirectionalSite),
    IncompleteSelection(IncompleteDirectionalSelection),
}

#[derive(Clone, Debug)]
struct TraversalEmissionSpec {
    text: &'static str,
    semantic_restrictions: Vec<(VariableId, Domain)>,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedNonStereo {
    molecule: PreparedMolecule,
    atoms: Arc<[PreparedAtom]>,
    bond_tokens: Arc<[NonStereoBondToken]>,
    atom_aromatic_modes: Arc<[Option<VariableId>]>,
    bond_chemical_orders: Arc<[Option<VariableId>]>,
    directional_bonds: Arc<[PreparedDirectionalBondStatus]>,
    directional_sides: Arc<[PreparedDirectionalSide]>,
    ring_count_factors: Arc<[Option<(FactorId, Box<[VariableId]>)>]>,
    ordered_ring_factors: Arc<[Option<(FactorId, VariableId, Box<[VariableId]>)>]>,
    event_schedule_factor: Option<FactorId>,
    event_schedule_visits: Arc<[EventVisit]>,
    event_schedule_capacity_bits: Arc<[VariableId]>,
    ordinary_order_pairs: Arc<[Arc<[OrdinaryOrderPair]>]>,
    #[cfg(test)]
    test_atom_labels: Arc<[String]>,
    #[cfg(test)]
    test_maximum_spelling_label: usize,
    #[cfg(test)]
    work_counters: Arc<WriterWorkCounters>,
}

const MAX_RING_LABEL_INDEX: usize = 99;

fn schedule_namespace_capacity(regime: WriterRegime, maximum: usize) -> usize {
    let first = usize::from(matches!(regime, WriterRegime::RdkitRandomNoncanonical));
    maximum.saturating_add(1).saturating_sub(first)
}

fn ring_count_exceeds_schedule_capacity(ring_count: usize, regime: WriterRegime, maximum: usize) -> bool {
    ring_count > schedule_namespace_capacity(regime, maximum)
}

#[cfg(test)]
#[derive(Debug, Default)]
struct WriterWorkCounters {
    pending_atom_frontier_evaluations: AtomicUsize,
    discarded_prevalidated_successors: AtomicUsize,
}

#[derive(Clone, Debug)]
pub enum PreparedAtomToken {
    Facts {
        facts: AtomFacts,
        source_context: AtomSourceContext,
        tetrahedral_reference_order: Option<[TetrahedralLigand; 4]>,
        at_for_even_parity: bool,
    },
    #[cfg(test)]
    Fixed(String),
    #[cfg(test)]
    Tetrahedral {
        reference_order: [TetrahedralLigand; 4],
        text_by_parity: [String; 2],
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AtomSourceContext {
    pub atom_class_property_present: bool,
}

#[derive(Clone, Debug)]
enum PreparedAtom {
    Facts {
        facts: AtomFacts,
        source_context: AtomSourceContext,
        tetrahedral: Option<PreparedTetrahedralCenter>,
    },
}

#[derive(Clone, Debug)]
enum AtomSpellingStart {
    Grammar { atom: AtomId, state: AtomSpellingState },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AtomSpellingProgress {
    atom: AtomId,
    state: AtomSpellingState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ChildEntrySpelling {
    incident: AdjacentBond,
    bond_text: &'static str,
    grammar: AtomSpellingState,
}

impl ChildEntrySpelling {
    fn atom(&self) -> AtomId {
        self.incident.atom()
    }

    fn bond(&self) -> BondId {
        self.incident.bond()
    }
}

#[derive(Clone, Debug)]
struct PreparedTetrahedralCenter {
    reference_order: [TetrahedralLigand; 4],
    at_for_even_parity: bool,
    order_variable: VariableId,
    role_pattern_variable: VariableId,
    bond_pattern_bits: Box<[(BondId, u8)]>,
    root_layout_factor: FactorId,
    entry_layout_factors: Box<[(BondId, FactorId)]>,
}

impl PreparedTetrahedralCenter {
    fn incidence(&self) -> TetrahedralIncidence<'_> {
        TetrahedralIncidence::new(&self.reference_order, &self.bond_pattern_bits)
    }

    fn token_domain(&self, entry_bond: Option<BondId>, parity: TetrahedralParity) -> Domain {
        self.incidence().prefix_orders(entry_bond, &[], Some(parity))
    }

    fn prefix_domain_with_bond_order(
        &self,
        entry_bond: Option<BondId>,
        emitted_bonds: &[BondId],
    ) -> Domain {
        self.incidence().prefix_orders(entry_bond, emitted_bonds, None)
    }

    fn completed_order_domain(
        &self,
        entry_bond: Option<BondId>,
        emitted_bonds: &[BondId],
    ) -> Domain {
        assert_eq!(emitted_bonds.len() + usize::from(entry_bond.is_some()), self.bond_pattern_bits.len());
        self.incidence().prefix_orders(entry_bond, emitted_bonds, None)
    }

    fn layout_factor(&self, entry_bond: Option<BondId>) -> FactorId {
        let Some(entry_bond) = entry_bond else {
            return self.root_layout_factor;
        };
        self.entry_layout_factors
            .iter()
            .find_map(|(bond, factor)| (*bond == entry_bond).then_some(*factor))
            .expect("every prepared entry bond must own one latent layout factor")
    }

    fn pattern_bit(&self, bond: BondId) -> u8 {
        self.bond_pattern_bits
            .iter()
            .find_map(|(candidate, bit)| (*candidate == bond).then_some(*bit))
            .expect("every prepared tetrahedral bond must own one role-pattern bit")
    }
}

impl PreparedNonStereo {
    #[cfg(test)]
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
        Self::with_atom_tokens_and_directional(molecule, atoms, bond_tokens, Vec::new())
    }

    pub(crate) fn with_atom_tokens_and_directional(
        molecule: PreparedMolecule,
        atoms: Vec<PreparedAtomToken>,
        bond_tokens: Vec<NonStereoBondToken>,
        directional_relations: Vec<PreparedDirectionalRelation>,
    ) -> Result<Self, PreparedNonStereoError> {
        Self::with_atom_tokens_and_directional_for_regime(molecule, atoms, bond_tokens, directional_relations, false)
    }

    pub(crate) fn with_atom_tokens_and_directional_for_regime(
        molecule: PreparedMolecule,
        atoms: Vec<PreparedAtomToken>,
        bond_tokens: Vec<NonStereoBondToken>,
        directional_relations: Vec<PreparedDirectionalRelation>,
        compact: bool,
    ) -> Result<Self, PreparedNonStereoError> {
        Self::with_atom_tokens_and_directional_for_writer(
            molecule, atoms, bond_tokens, directional_relations,
            if compact { WriterRegime::GenericCompact } else { WriterRegime::Generic { component_order: ComponentOrder::InputOrder } }, false,
        )
    }

    pub(crate) fn with_atom_tokens_and_directional_for_writer(
        molecule: PreparedMolecule,
        atoms: Vec<PreparedAtomToken>,
        bond_tokens: Vec<NonStereoBondToken>,
        directional_relations: Vec<PreparedDirectionalRelation>,
        regime: WriterRegime,
        allow_segments: bool,
    ) -> Result<Self, PreparedNonStereoError> {
        Self::prepare_for_writer(molecule, atoms, bond_tokens, directional_relations,
            regime, allow_segments, MAX_RING_LABEL_INDEX)
    }

    #[cfg(test)]
    pub(crate) fn with_ring_label_maximum_for_test(
        molecule: PreparedMolecule,
        atoms: Vec<PreparedAtomToken>,
        bond_tokens: Vec<NonStereoBondToken>,
        directional_relations: Vec<PreparedDirectionalRelation>,
        regime: WriterRegime,
        allow_segments: bool,
        maximum_spelling_label: usize,
    ) -> Result<Self, PreparedNonStereoError> {
        Self::prepare_for_writer(molecule, atoms, bond_tokens, directional_relations,
            regime, allow_segments, maximum_spelling_label)
    }

    fn prepare_for_writer(
        molecule: PreparedMolecule,
        atoms: Vec<PreparedAtomToken>,
        bond_tokens: Vec<NonStereoBondToken>,
        directional_relations: Vec<PreparedDirectionalRelation>,
        regime: WriterRegime,
        allow_segments: bool,
        maximum_spelling_label: usize,
    ) -> Result<Self, PreparedNonStereoError> {
        let compact = regime == WriterRegime::GenericCompact;
        #[cfg(test)]
        let test_atom_labels = atoms
            .iter()
            .map(|atom| match atom {
                PreparedAtomToken::Facts { facts, .. } => facts.element.as_str().to_owned(),
                PreparedAtomToken::Fixed(text) => text.clone(),
                PreparedAtomToken::Tetrahedral { text_by_parity, .. } => text_by_parity[0].clone(),
            })
            .collect::<Vec<_>>();

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
        let chemical_aromatic_bonds = graph.bond_ids().filter(|bond| {
            let edge = graph.bond(*bond).unwrap();
            matches!(&atoms[edge.a().index()], PreparedAtomToken::Facts { .. })
                || matches!(&atoms[edge.b().index()], PreparedAtomToken::Facts { .. })
        }).collect::<BTreeSet<_>>();
        for (atom, prepared) in graph.atom_ids().zip(&atoms) {
            validate_prepared_atom(graph, atom, prepared)?;
        }
        let compiled_directional_bonds =
            provisionally_compiled_directional_bonds(&directional_relations);
        let decision_domains = graph
            .bond_ids()
            .map(|bond| {
                if compiled_directional_bonds.contains(&bond) {
                    BondRole::role_domain()
                } else {
                    if bond_tokens[bond.index()] == NonStereoBondToken::Aromatic && chemical_aromatic_bonds.contains(&bond) {
                        BondRepresentation::elided_domain().union(BondRepresentation::explicit_domain())
                    } else if compact && matches!(bond_tokens[bond.index()], NonStereoBondToken::Single | NonStereoBondToken::Elided) {
                        BondRepresentation::elided_domain().union(BondRepresentation::explicit_domain())
                    } else if matches!(regime, WriterRegime::Generic { .. })
                        && bond_tokens[bond.index()] == NonStereoBondToken::Elided {
                        BondRepresentation::elided_domain().union(BondRepresentation::explicit_domain())
                    } else {
                        bond_tokens[bond.index()].representation_domain()
                    }
                }
            })
            .collect::<Vec<_>>();
        let role_partitions = graph
            .bond_ids()
            .map(|bond| {
                if compiled_directional_bonds.contains(&bond) {
                    EdgeRolePartition::bond_role()
                } else {
                    BondRepresentation::role_partition()
                }
            })
            .collect::<Vec<_>>();
        let mut assembly =
            PreparedMolecule::constraint_assembly(&molecule, &decision_domains, &role_partitions);
        let ring_count_factors = graph.atom_ids().zip(&atoms).map(|(atom, prepared)| {
            let is_tetrahedral = match prepared {
                PreparedAtomToken::Facts { tetrahedral_reference_order, .. } => tetrahedral_reference_order.is_some(),
                #[cfg(test)]
                PreparedAtomToken::Tetrahedral { .. } => true,
                #[cfg(test)]
                PreparedAtomToken::Fixed(_) => false,
            };
            if is_tetrahedral {
                return None;
            }
            let adjacent = graph.neighbors(atom).expect("prepared atom must have adjacency");
            if adjacent.is_empty() { return None; }
            let max_bound = adjacent.len().min(2 * (MAX_RING_LABEL_INDEX + 1));
            let width = (usize::BITS - max_bound.leading_zeros()) as usize;
            let budget_bits = (0..width).map(|_| assembly.add_isolated_variable(Domain::from_bits(3)))
                .collect::<Vec<_>>();
            let bonds = adjacent.iter().map(|incident| {
                let bond = graph.bond(incident.bond()).expect("adjacent bond must exist");
                SpanningTreeEdge::with_role_partition(
                    assembly.bond_decision_variable(incident.bond()), bond.a(), bond.b(),
                    assembly.bond_role_partition(incident.bond()),
                )
            }).collect::<Vec<_>>();
            let factor = assembly.add_latent_ring_count_at_most(bonds, budget_bits.iter().copied());
            Some((factor, budget_bits.into_boxed_slice()))
        }).collect::<Vec<_>>();
        let (atom_aromatic_modes, bond_chemical_orders) =
            prepare_aromatic_matching(&mut assembly, graph, &atoms, &bond_tokens, false)?;
        for relation in &directional_relations {
            for carrier in relation.left_carriers.iter().chain(relation.right_carriers.iter()) {
                if bond_chemical_orders.get(carrier.bond.index()).is_some_and(Option::is_some) {
                    return Err(PreparedNonStereoError::UnsupportedAromaticDirectionalCarrier { bond: carrier.bond });
                }
            }
        }
        let atoms = atoms
            .into_iter()
            .enumerate()
            .map(|(index, prepared)| match prepared {
                PreparedAtomToken::Facts {
                    facts,
                    source_context,
                    tetrahedral_reference_order,
                    at_for_even_parity,
                } => {
                    let tetrahedral = tetrahedral_reference_order.map(|reference_order| {
                        prepare_tetrahedral_center(
                            &mut assembly,
                            reference_order,
                            at_for_even_parity,
                        )
                    });
                    PreparedAtom::Facts { facts, source_context, tetrahedral }
                }
                #[cfg(test)]
                #[cfg(test)]
                PreparedAtomToken::Fixed(text) => PreparedAtom::Facts {
                    facts: test_atom_facts(&molecule.graph(), &bond_tokens, AtomId::new(u32::try_from(index).unwrap()), &text, None),
                    source_context: AtomSourceContext::default(),
                    tetrahedral: None,
                },
                #[cfg(test)]
                PreparedAtomToken::Tetrahedral { reference_order, text_by_parity } => {
                    let facts = test_tetrahedral_facts(
                        reference_order.contains(&TetrahedralLigand::VirtualHydrogen),
                    );
                    PreparedAtom::Facts {
                        facts,
                        source_context: AtomSourceContext::default(),
                        tetrahedral: Some(prepare_tetrahedral_center(
                            &mut assembly,
                            reference_order,
                            text_by_parity[0].contains('@') && !text_by_parity[0].contains("@@"),
                        )),
                    }
                }
            })
            .collect::<Vec<_>>();
        let ordered_ring_factors = graph.atom_ids().map(|atom| {
            let center = match &atoms[atom.index()] {
                PreparedAtom::Facts { tetrahedral: Some(center), .. } => center,
                _ => return None,
            };
            let adjacent = graph.neighbors(atom).expect("prepared atom must have adjacency");
            let bonds = adjacent.iter().map(|incident| {
                let id = incident.bond();
                let bond = graph.bond(id).expect("adjacent bond must exist");
                (id, SpanningTreeEdge::with_role_partition(
                    assembly.bond_decision_variable(id), bond.a(), bond.b(),
                    assembly.bond_role_partition(id),
                ))
            }).collect::<Vec<_>>();
            let mask = assembly.add_isolated_variable(
                Domain::from_indices(0..(1_u8 << adjacent.len())).unwrap(),
            );
            let max_free = adjacent.len().min(MAX_RING_LABEL_INDEX + 1);
            let width = (usize::BITS - max_free.leading_zeros()) as usize;
            let bits = (0..width).map(|_| assembly.add_isolated_variable(Domain::from_bits(3)))
                .collect::<Vec<_>>();
            let factor = assembly.add_latent_ordered_ring_phase(
                bonds, center.reference_order, center.order_variable, mask, bits.iter().copied(),
            );
            Some((factor, mask, bits.into_boxed_slice()))
        }).collect::<Vec<_>>();
        let (directional_bonds, directional_sides) =
            prepare_directional_bonds(&mut assembly, graph, &bond_tokens, &directional_relations)?;
        let cycle_count = graph.bond_count().saturating_add(graph.components().len())
            .saturating_sub(graph.atom_count());
        let possible_ring_count = if allow_segments { graph.bond_count() } else { cycle_count };
        let prepare_schedule = graph.bond_count() > 0 &&
            ring_count_exceeds_schedule_capacity(possible_ring_count, regime, maximum_spelling_label);
        let (event_schedule_factor, event_schedule_visits, event_schedule_capacity_bits,
            ordinary_order_pairs) = if prepare_schedule {
            let edges: Arc<[ScheduleEdge]> = graph.bond_ids().map(|bond| {
                let edge = graph.bond(bond).expect("prepared schedule bond must exist");
                ScheduleEdge { bond, a: edge.a(), b: edge.b(),
                    decision_variable: assembly.bond_decision_variable(bond),
                    role_partition: assembly.bond_role_partition(bond) }
            }).collect::<Vec<_>>().into();
            let centers: Arc<[ScheduleCenter]> = graph.atom_ids().filter_map(|atom| {
                match &atoms[atom.index()] {
                    PreparedAtom::Facts { tetrahedral: Some(center), .. } => Some(ScheduleCenter {
                        atom, order_variable: center.order_variable,
                        role_pattern_variable: center.role_pattern_variable,
                        reference_order: center.reference_order,
                        bond_pattern_bits: center.bond_pattern_bits.iter().copied().collect::<Vec<_>>().into(),
                    }),
                    _ => None,
                }
            }).collect::<Vec<_>>().into();
            let mut ordinary_pairs = vec![Arc::<[OrdinaryOrderPair]>::from(Vec::new()); graph.atom_count()];
            let mut ordinary_centers = Vec::new();
            for atom in graph.atom_ids() {
                if matches!(atoms[atom.index()], PreparedAtom::Facts { tetrahedral: Some(_), .. }) {
                    continue;
                }
                let incidence = graph.neighbors(atom).expect("prepared atom must have adjacency")
                    .iter().map(|adjacent| adjacent.bond()).collect::<Vec<_>>();
                let mut pairs = Vec::new();
                for i in 0..incidence.len() {
                    for j in i + 1..incidence.len() {
                        pairs.push(OrdinaryOrderPair { first: incidence[i], second: incidence[j],
                            variable: assembly.add_isolated_variable(Domain::from_bits(3)) });
                    }
                }
                let pairs: Arc<[OrdinaryOrderPair]> = pairs.into();
                ordinary_pairs[atom.index()] = Arc::clone(&pairs);
                ordinary_centers.push(ScheduleOrdinaryCenter { atom, pairs });
            }
            let capacity_bits: Arc<[VariableId]> = (0..7).map(|_| {
                assembly.add_isolated_variable(Domain::from_bits(3))
            }).collect::<Vec<_>>().into();
            // todo: This allocates future visit ranks, entry kinds, and branch ends as
            // CSP facts for every atom. Move traversal bookkeeping into the live walker
            // once compact feasibility facts preserve every productive labelled choice.
            let rank_width = (usize::BITS - (graph.atom_count() - 1).leading_zeros()) as usize;
            let visits: Arc<[EventVisit]> = (0..graph.atom_count()).map(|index| EventVisit {
                atom: AtomId::new(u32::try_from(index).unwrap()),
                rank_bits: (0..rank_width).map(|_|
                    assembly.add_isolated_variable(Domain::from_bits(3)))
                    .collect::<Vec<_>>().into(),
                kind: assembly.add_isolated_variable(EntryKind::domain()),
                end_bits: (0..graph.atom_count()).map(|_|
                    assembly.add_isolated_variable(Domain::from_bits(3)))
                    .collect::<Vec<_>>().into(),
            }).collect::<Vec<_>>().into();
            let forest_roots = if allow_segments {
                assembly.prepare_forest_root_variables()
            } else { Arc::from(Vec::<VariableId>::new()) };
            let factor = assembly.add_latent_event_schedule(visits.clone(), edges, centers,
                ordinary_centers.into(), Arc::clone(&capacity_bits), forest_roots, allow_segments);
            (Some(factor), visits, capacity_bits, ordinary_pairs.into())
        } else { (None, Arc::from(Vec::<EventVisit>::new()), Arc::from(Vec::<VariableId>::new()),
            vec![Arc::<[OrdinaryOrderPair]>::from(Vec::new()); graph.atom_count()].into()) };
        let segment_topology = allow_segments;
        let molecule = if segment_topology {
            assembly.finish_with_topology(crate::model::TraversalTopologyRequirement::Forest)
        } else {
            assembly.finish()
        };

        Ok(Self {
            molecule,
            atoms: Arc::from(atoms.into_boxed_slice()),
            bond_tokens: Arc::from(bond_tokens.into_boxed_slice()),
            atom_aromatic_modes: Arc::from(atom_aromatic_modes.into_boxed_slice()),
            bond_chemical_orders: Arc::from(bond_chemical_orders.into_boxed_slice()),
            directional_bonds: Arc::from(directional_bonds.into_boxed_slice()),
            directional_sides: Arc::from(directional_sides.into_boxed_slice()),
            ring_count_factors: Arc::from(ring_count_factors.into_boxed_slice()),
            ordered_ring_factors: Arc::from(ordered_ring_factors.into_boxed_slice()),
            event_schedule_factor,
            event_schedule_visits,
            event_schedule_capacity_bits,
            ordinary_order_pairs,
            #[cfg(test)]
            test_maximum_spelling_label: maximum_spelling_label,
            #[cfg(test)]
            test_atom_labels: Arc::from(test_atom_labels.into_boxed_slice()),
            #[cfg(test)]
            work_counters: Arc::new(WriterWorkCounters::default()),
        })
    }

    pub(crate) fn molecule(&self) -> &PreparedMolecule {
        &self.molecule
    }

    fn atom_facts(&self, atom: AtomId) -> &AtomFacts {
        let prepared = self.atoms.get(atom.index()).expect("prepared atom facts must match the bound molecule");
        match prepared {
            PreparedAtom::Facts { facts, .. } => facts,
        }
    }

    fn atom_source_context(&self, atom: AtomId) -> AtomSourceContext {
        match self.atoms.get(atom.index()).expect("prepared atom facts must match the bound molecule") {
            PreparedAtom::Facts { source_context, .. } => *source_context,
        }
    }

    #[cfg(test)]
    pub(crate) fn atom_text(&self, atom: AtomId) -> &str {
        &self.test_atom_labels[atom.index()]
    }

    #[cfg(test)]
    pub(crate) fn normalize_test_atom_classes(&self, text: &str) -> String {
        let mut normalized = text.to_owned();
        for (index, atom) in self.atoms.iter().enumerate() {
            let PreparedAtom::Facts { facts, .. } = atom;
            if facts.atom_class != 0 {
                normalized = normalized.replace(
                    &format!("[*:{}]", facts.atom_class),
                    &self.test_atom_labels[index],
                );
            }
        }
        normalized
    }

    fn tetrahedral_center(&self, atom: AtomId) -> Option<&PreparedTetrahedralCenter> {
        match self.atoms.get(atom.index())? {
            PreparedAtom::Facts { tetrahedral, .. } => tetrahedral.as_ref(),
        }
    }

    fn atom_spelling_context(
        &self,
        atom: AtomId,
        parity: Option<TetrahedralParity>,
    ) -> AtomSpellingContext {
        let mut bonds = AtomBondContext::default();
        for incident in self.molecule.graph().neighbors(atom).expect("prepared atom belongs to graph") {
            match self.bond_tokens[incident.bond().index()] {
                NonStereoBondToken::Elided | NonStereoBondToken::Single => bonds.single_bonds += 1,
                NonStereoBondToken::Double => bonds.double_bonds += 1,
                NonStereoBondToken::Triple => bonds.triple_bonds += 1,
                NonStereoBondToken::Aromatic => bonds.aromatic_bonds += 1,
                NonStereoBondToken::DativeAToB | NonStereoBondToken::DativeBToA => {}
            }
        }
        let tetrahedral_marker = match (self.tetrahedral_center(atom), parity) {
            (Some(center), Some(parity)) => {
                let is_at = center.at_for_even_parity == (parity == TetrahedralParity::Even);
                Some(if is_at { TetrahedralMarker::At } else { TetrahedralMarker::AtAt })
            }
            (None, None) => None,
            _ => panic!("tetrahedral spelling parity must match prepared center"),
        };
        AtomSpellingContext { tetrahedral_marker, bonds, aromatic_spelling: AromaticSpelling::Source }
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

    fn directional_status(&self, bond: BondId) -> PreparedDirectionalBondStatus {
        *self
            .directional_bonds
            .get(bond.index())
            .expect("prepared directional status must match the bound molecule")
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

fn prepare_aromatic_matching(
    assembly: &mut PreparedConstraintAssembly,
    graph: &PreparedGraph,
    atoms: &[PreparedAtomToken],
    tokens: &[NonStereoBondToken],
    uniform_case: bool,
) -> Result<(Vec<Option<VariableId>>, Vec<Option<VariableId>>), PreparedNonStereoError> {
    let mut modes = vec![None; graph.atom_count()];
    let mut orders = vec![None; graph.bond_count()];
    let mut aromatic_edges = BTreeSet::new();
    for bond in graph.bond_ids().filter(|bond| tokens[bond.index()] == NonStereoBondToken::Aromatic) {
        let edge = graph.bond(bond).unwrap();
        match (&atoms[edge.a().index()], &atoms[edge.b().index()]) {
            (PreparedAtomToken::Facts { facts: a, .. }, PreparedAtomToken::Facts { facts: b, .. })
                if a.aromatic && b.aromatic && a.aromatic_total_valence.is_some() && b.aromatic_total_valence.is_some() => { aromatic_edges.insert(bond); }
            // Fixed adapters model lexical tokens, not chemical aromatic atoms.
            #[cfg(test)]
            (PreparedAtomToken::Fixed(_), PreparedAtomToken::Fixed(_)) => {}
            _ => return Err(PreparedNonStereoError::SourceBondMismatch { bond }),
        }
    }
    for root in graph.atom_ids() {
        if modes[root.index()].is_some() || !graph.neighbors(root).unwrap().iter().any(|edge| aromatic_edges.contains(&edge.bond())) { continue; }
        let mut members = BTreeSet::new();
        let mut queue = vec![root];
        while let Some(atom) = queue.pop() {
            if !members.insert(atom) { continue; }
            for edge in graph.neighbors(atom).unwrap().iter().filter(|edge| aromatic_edges.contains(&edge.bond())) { queue.push(edge.atom()); }
        }
        let case_variables = members.iter().map(|_| assembly.add_isolated_variable(Domain::from_bits(3))).collect::<Vec<_>>();
        let mut demands = Vec::new();
        let vertices = members.iter().enumerate().map(|(index, atom)| (*atom, index)).collect::<BTreeMap<_, _>>();
        for atom in &members {
            let PreparedAtomToken::Facts { facts, .. } = &atoms[atom.index()] else { unreachable!() };
            let mut fixed_bond_order_sum = 0;
            let mut aromatic_degree = 0;
            for edge in graph.neighbors(*atom).unwrap() {
                match tokens[edge.bond().index()] {
                    NonStereoBondToken::Aromatic => aromatic_degree += 1,
                    NonStereoBondToken::Elided | NonStereoBondToken::Single => fixed_bond_order_sum += 1,
                    NonStereoBondToken::Double => fixed_bond_order_sum += 2,
                    NonStereoBondToken::Triple => fixed_bond_order_sum += 3,
                    NonStereoBondToken::DativeAToB | NonStereoBondToken::DativeBToA => return Err(PreparedNonStereoError::AromaticDativeIncidence { atom: *atom, bond: edge.bond() }),
                }
            }
            let total_valence = facts.aromatic_total_valence.unwrap();
            let base = u32::from(facts.non_graph_hydrogens) + fixed_bond_order_sum + aromatic_degree;
            let demand = total_valence.checked_sub(base).filter(|d| *d <= 1)
                .ok_or(PreparedNonStereoError::AromaticDemand {
                    atom: *atom, total_valence, non_graph_hydrogens: facts.non_graph_hydrogens,
                    fixed_bond_order_sum, aromatic_degree,
                })?;
            demands.push(demand as u8);
            modes[atom.index()] = Some(case_variables[vertices[atom]]);
        }
        let mut edges = Vec::new();
        for bond in &aromatic_edges {
            let edge = graph.bond(*bond).unwrap();
            if !members.contains(&edge.a()) { continue; }
            let order = assembly.add_isolated_variable(Domain::from_bits(7));
            orders[bond.index()] = Some(order);
            edges.push(AromaticMatchingEdge::new(order, vertices[&edge.a()], vertices[&edge.b()]));
        }
        if !crate::aromatic_matching::has_feasible_double_assignment(&demands, &edges) {
            return Err(PreparedNonStereoError::InfeasibleAromaticComponent { atom: root });
        }
        assembly.add_aromatic_matching(case_variables, uniform_case, demands, edges);
    }
    for atom in graph.atom_ids() {
        if matches!(&atoms[atom.index()], PreparedAtomToken::Facts { facts, .. } if facts.aromatic) && modes[atom.index()].is_none() {
            return Err(PreparedNonStereoError::MissingAromaticComponent { atom });
        }
    }
    Ok((modes, orders))
}

fn provisionally_compiled_directional_bonds(
    relations: &[PreparedDirectionalRelation],
) -> BTreeSet<BondId> {
    let mut relations_by_carrier = BTreeMap::<BondId, Vec<usize>>::new();
    for (index, relation) in relations.iter().enumerate() {
        for carrier in relation
            .left_carriers
            .iter()
            .chain(relation.right_carriers.iter())
        {
            relations_by_carrier
                .entry(carrier.bond)
                .or_default()
                .push(index);
        }
    }
    let mut compiled = BTreeSet::new();
    let mut unseen = (0..relations.len()).collect::<BTreeSet<_>>();
    while let Some(first) = unseen.pop_first() {
        let mut component = vec![first];
        let mut queue = VecDeque::from([first]);
        while let Some(index) = queue.pop_front() {
            for carrier in relations[index]
                .left_carriers
                .iter()
                .chain(relations[index].right_carriers.iter())
            {
                for adjacent in &relations_by_carrier[&carrier.bond] {
                    if unseen.remove(adjacent) {
                        component.push(*adjacent);
                        queue.push_back(*adjacent);
                    }
                }
            }
        }
        if component.iter().any(|index| {
            relations[*index].left_carriers.len() > 2 || relations[*index].right_carriers.len() > 2
        }) {
            continue;
        }
        compiled.extend(component.into_iter().flat_map(|index| {
            relations[index]
                .left_carriers
                .iter()
                .chain(relations[index].right_carriers.iter())
                .map(|carrier| carrier.bond)
        }));
    }
    compiled
}

fn prepare_tetrahedral_center(
    assembly: &mut PreparedConstraintAssembly,
    reference_order: [TetrahedralLigand; 4],
    at_for_even_parity: bool,
) -> PreparedTetrahedralCenter {
    let bond_pattern_bits = reference_order
        .iter()
        .filter_map(|ligand| match ligand {
            TetrahedralLigand::Bond(bond) => Some(*bond),
            TetrahedralLigand::VirtualHydrogen => None,
        })
        .enumerate()
        .map(|(bit, bond)| (bond, u8::try_from(bit).unwrap()))
        .collect::<Vec<_>>();
    let order_variable = assembly.add_isolated_variable(full_order_domain());
    let role_pattern_variable =
        assembly.add_isolated_variable(full_role_pattern_domain(bond_pattern_bits.len()));
    let layout_bonds = bond_pattern_bits
        .iter()
        .map(|(bond, bit)| {
            TetrahedralLayoutBond::new(
                assembly.bond_decision_variable(*bond),
                assembly.bond_role_partition(*bond),
                *bit,
            )
        })
        .collect::<Vec<_>>();
    let root_layout_rows: Arc<[Domain]> = layout_order_rows(&reference_order,
        None, &bond_pattern_bits).into();
    let root_layout_factor = assembly.add_latent_tetrahedral_layout_rows(order_variable,
        role_pattern_variable, layout_bonds.iter().copied(), Arc::clone(&root_layout_rows));
    let entry_layouts = bond_pattern_bits
        .iter()
        .map(|(bond, _)| {
            let rows: Arc<[Domain]> = layout_order_rows(&reference_order,
                Some(*bond), &bond_pattern_bits).into();
            let factor = assembly.add_latent_tetrahedral_layout_rows(order_variable,
                role_pattern_variable, layout_bonds.iter().copied(), Arc::clone(&rows));
            (*bond, factor)
        })
        .collect::<Vec<_>>();

    PreparedTetrahedralCenter {
        reference_order,
        at_for_even_parity,
        order_variable,
        role_pattern_variable,
        bond_pattern_bits: bond_pattern_bits.into_boxed_slice(),
        root_layout_factor,
        entry_layout_factors: entry_layouts.into_boxed_slice(),
    }
}

fn prepare_directional_bonds(
    assembly: &mut PreparedConstraintAssembly,
    graph: &crate::prepared::PreparedGraph,
    bond_tokens: &[NonStereoBondToken],
    relations: &[PreparedDirectionalRelation],
) -> Result<
    (
        Vec<PreparedDirectionalBondStatus>,
        Vec<PreparedDirectionalSide>,
    ),
    PreparedNonStereoError,
> {
    let mut statuses = vec![PreparedDirectionalBondStatus::Ordinary; graph.bond_count()];
    let mut directional_sides = Vec::with_capacity(relations.len() * 2);
    let mut relations_by_carrier = BTreeMap::<BondId, Vec<usize>>::new();
    let mut configured_double_bonds = BTreeSet::new();

    for (index, relation) in relations.iter().enumerate() {
        if !configured_double_bonds.insert(relation.double_bond) {
            return Err(PreparedNonStereoError::RepeatedDirectionalRelation(
                relation.double_bond,
            ));
        }
        let double_bond = graph.bond(relation.double_bond).ok_or(
            PreparedNonStereoError::UnknownDirectionalBond(relation.double_bond),
        )?;
        if !((double_bond.a() == relation.left_endpoint
            && double_bond.b() == relation.right_endpoint)
            || (double_bond.b() == relation.left_endpoint
                && double_bond.a() == relation.right_endpoint))
        {
            return Err(PreparedNonStereoError::DirectionalDoubleBondEndpoints {
                double_bond: relation.double_bond,
                left: relation.left_endpoint,
                right: relation.right_endpoint,
            });
        }
        if bond_tokens[relation.double_bond.index()] != NonStereoBondToken::Double {
            return Err(PreparedNonStereoError::DirectionalDoubleBondToken(
                relation.double_bond,
            ));
        }
        if relation.left_carriers.is_empty() {
            return Err(PreparedNonStereoError::EmptyDirectionalCarrierSide {
                double_bond: relation.double_bond,
                endpoint: relation.left_endpoint,
            });
        }
        if relation.right_carriers.is_empty() {
            return Err(PreparedNonStereoError::EmptyDirectionalCarrierSide {
                double_bond: relation.double_bond,
                endpoint: relation.right_endpoint,
            });
        }
        for (endpoint, carriers) in [
            (relation.left_endpoint, relation.left_carriers.as_ref()),
            (relation.right_endpoint, relation.right_carriers.as_ref()),
        ] {
            let mut seen = BTreeSet::new();
            for carrier in carriers {
                if !seen.insert(carrier.bond) {
                    return Err(PreparedNonStereoError::RepeatedDirectionalCarrier {
                        double_bond: relation.double_bond,
                        endpoint,
                        carrier: carrier.bond,
                    });
                }
                if carrier.bond == relation.double_bond {
                    return Err(PreparedNonStereoError::DirectionalCarrierIsDoubleBond(
                        carrier.bond,
                    ));
                }
                let topology = graph
                    .bond(carrier.bond)
                    .ok_or(PreparedNonStereoError::UnknownDirectionalBond(carrier.bond))?;
                if topology.other(endpoint).is_none() {
                    return Err(PreparedNonStereoError::DirectionalCarrierNotIncident {
                        carrier: carrier.bond,
                        endpoint,
                    });
                }
                if !bond_tokens[carrier.bond.index()].is_directional_carrier_base() {
                    return Err(PreparedNonStereoError::InvalidDirectionalCarrierToken(
                        carrier.bond,
                    ));
                }
                relations_by_carrier
                    .entry(carrier.bond)
                    .or_default()
                    .push(index);
            }
        }
    }

    let mut unseen = (0..relations.len()).collect::<BTreeSet<_>>();
    while let Some(first) = unseen.pop_first() {
        let mut component = vec![first];
        let mut queue = VecDeque::from([first]);
        while let Some(relation_index) = queue.pop_front() {
            let relation = &relations[relation_index];
            for carrier in relation
                .left_carriers
                .iter()
                .chain(relation.right_carriers.iter())
            {
                for adjacent in &relations_by_carrier[&carrier.bond] {
                    if unseen.remove(adjacent) {
                        component.push(*adjacent);
                        queue.push_back(*adjacent);
                    }
                }
            }
        }

        let incomplete = component.iter().find_map(|index| {
            let relation = &relations[*index];
            if relation.left_carriers.len() > 2 {
                Some(IncompleteDirectionalSelection {
                    double_bond: relation.double_bond,
                    endpoint: relation.left_endpoint,
                })
            } else if relation.right_carriers.len() > 2 {
                Some(IncompleteDirectionalSelection {
                    double_bond: relation.double_bond,
                    endpoint: relation.right_endpoint,
                })
            } else {
                None
            }
        });
        if let Some(incomplete) = incomplete {
            for index in component {
                let relation = &relations[index];
                for carrier in relation
                    .left_carriers
                    .iter()
                    .chain(relation.right_carriers.iter())
                {
                    statuses[carrier.bond.index()] =
                        PreparedDirectionalBondStatus::IncompleteSelection(incomplete);
                }
            }
            continue;
        }

        if !directional_component_is_satisfiable(graph, relations, &component) {
            let carrier = relations[component[0]]
                .left_carriers
                .first()
                .expect("validated directional side must have a carrier")
                .bond;
            return Err(PreparedNonStereoError::ContradictoryDirectionalParity { carrier });
        }

        let carriers = component
            .iter()
            .flat_map(|index| {
                relations[*index]
                    .left_carriers
                    .iter()
                    .chain(relations[*index].right_carriers.iter())
                    .map(|carrier| carrier.bond)
            })
            .collect::<BTreeSet<_>>();
        let variables = carriers
            .iter()
            .map(|carrier| {
                (
                    *carrier,
                    assembly.add_isolated_variable(CarrierMark::domain()),
                )
            })
            .collect::<BTreeMap<_, _>>();
        for carrier in &carriers {
            let mark_variable = variables[carrier];
            let ring_plan_domain = DirectionalRingPlan::domain_for(bond_tokens[carrier.index()]);
            let ring_plan_variable = assembly.add_isolated_variable(ring_plan_domain);
            let ring_plan_factor = assembly.add_latent_directional_ring_placement(
                mark_variable,
                ring_plan_variable,
                DirectionalRingPlan::ALL.into_iter().filter_map(|plan| {
                    ring_plan_domain
                        .contains(plan.value_index())
                        .then_some((plan.mark().value_index(), plan.value_index()))
                }),
            );
            statuses[carrier.index()] =
                PreparedDirectionalBondStatus::CompiledSite(PreparedDirectionalSite {
                    mark_variable,
                    ring_plan_variable,
                    ring_plan_factor,
                });
        }

        for index in component {
            let relation = &relations[index];
            let left = prepare_directional_side(
                assembly,
                graph,
                relation.double_bond,
                relation.left_endpoint,
                &relation.left_carriers,
                &variables,
            );
            let right = prepare_directional_side(
                assembly,
                graph,
                relation.double_bond,
                relation.right_endpoint,
                &relation.right_carriers,
                &variables,
            );
            let allowed_pairs = left
                .patterns
                .iter()
                .enumerate()
                .flat_map(|(left_value, left_pattern)| {
                    right.patterns.iter().enumerate().filter_map(
                        move |(right_value, right_pattern)| {
                            (left_pattern.phase ^ right_pattern.phase == relation.side_phase_xor)
                                .then_some((left_value as u8, right_value as u8))
                        },
                    )
                })
                .collect::<Vec<_>>();
            assembly.add_binary_relation(
                left.pattern_variable,
                right.pattern_variable,
                allowed_pairs,
            );
            directional_sides.push(left);
            directional_sides.push(right);
        }
    }

    Ok((statuses, directional_sides))
}

fn prepare_directional_side(
    assembly: &mut PreparedConstraintAssembly,
    graph: &crate::prepared::PreparedGraph,
    double_bond: BondId,
    endpoint: AtomId,
    carriers: &[PreparedDirectionalCarrier],
    mark_variables: &BTreeMap<BondId, VariableId>,
) -> PreparedDirectionalSide {
    let patterns = directional_side_patterns(graph, endpoint, carriers);
    let pattern_domain = Domain::from_bits((1_u64 << patterns.len()) - 1);
    let pattern_variable = assembly.add_isolated_variable(pattern_domain);
    for (carrier_index, carrier) in carriers.iter().enumerate() {
        assembly.add_binary_relation(
            pattern_variable,
            mark_variables[&carrier.bond],
            patterns
                .iter()
                .enumerate()
                .map(|(value, pattern)| (value as u8, pattern.marks[carrier_index].value_index())),
        );
    }
    PreparedDirectionalSide {
        double_bond,
        endpoint,
        pattern_variable,
        carriers: carriers.into(),
        patterns: patterns.into_boxed_slice(),
    }
}

fn directional_side_patterns(
    graph: &crate::prepared::PreparedGraph,
    endpoint: AtomId,
    carriers: &[PreparedDirectionalCarrier],
) -> Vec<DirectionalSidePattern> {
    assert!((1..=2).contains(&carriers.len()));
    let assignment_count = 3_usize.pow(carriers.len() as u32);
    (0..assignment_count)
        .filter_map(|mut encoded| {
            let marks = (0..carriers.len())
                .map(|_| {
                    let mark = CarrierMark::from_value((encoded % 3) as u8).unwrap();
                    encoded /= 3;
                    mark
                })
                .collect::<Vec<_>>();
            let mut phase = None;
            for (carrier, mark) in carriers.iter().zip(&marks) {
                let Some(sign) = mark.canonical_sign() else {
                    continue;
                };
                let candidate_phase =
                    sign ^ endpoint_flip(graph, carrier.bond, endpoint) ^ carrier.side_flip;
                match phase {
                    Some(existing) if existing != candidate_phase => return None,
                    Some(_) => {}
                    None => phase = Some(candidate_phase),
                }
            }
            Some(DirectionalSidePattern {
                marks: marks.into_boxed_slice(),
                phase: phase?,
            })
        })
        .collect()
}

fn directional_component_is_satisfiable(
    graph: &crate::prepared::PreparedGraph,
    relations: &[PreparedDirectionalRelation],
    component: &[usize],
) -> bool {
    fn search(
        relation_pairs: &[Vec<Vec<(BondId, CarrierMark)>>],
        index: usize,
        assigned: &BTreeMap<BondId, CarrierMark>,
    ) -> bool {
        if index == relation_pairs.len() {
            return true;
        }
        relation_pairs[index].iter().any(|pair| {
            let mut next = assigned.clone();
            for (bond, mark) in pair {
                match next.get(bond) {
                    Some(existing) if existing != mark => return false,
                    Some(_) => {}
                    None => {
                        next.insert(*bond, *mark);
                    }
                }
            }
            search(relation_pairs, index + 1, &next)
        })
    }

    let relation_pairs = component
        .iter()
        .map(|index| {
            let relation = &relations[*index];
            let left =
                directional_side_patterns(graph, relation.left_endpoint, &relation.left_carriers);
            let right =
                directional_side_patterns(graph, relation.right_endpoint, &relation.right_carriers);
            left.iter()
                .flat_map(|left_pattern| {
                    right.iter().filter_map(move |right_pattern| {
                        if left_pattern.phase ^ right_pattern.phase != relation.side_phase_xor {
                            return None;
                        }
                        let assignments = relation
                            .left_carriers
                            .iter()
                            .zip(left_pattern.marks.iter())
                            .chain(
                                relation
                                    .right_carriers
                                    .iter()
                                    .zip(right_pattern.marks.iter()),
                            )
                            .map(|(carrier, mark)| (carrier.bond, *mark))
                            .collect::<Vec<_>>();
                        Some(assignments)
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    search(&relation_pairs, 0, &BTreeMap::new())
}

fn endpoint_flip(
    graph: &crate::prepared::PreparedGraph,
    carrier: BondId,
    endpoint: AtomId,
) -> bool {
    let bond = graph
        .bond(carrier)
        .expect("validated directional carrier must belong to the graph");
    if bond.a() == endpoint {
        false
    } else if bond.b() == endpoint {
        true
    } else {
        panic!("validated directional endpoint must be incident to its carrier")
    }
}

fn validate_prepared_atom(
    graph: &crate::prepared::PreparedGraph,
    atom: AtomId,
    prepared: &PreparedAtomToken,
) -> Result<(), PreparedNonStereoError> {
    match prepared {
        PreparedAtomToken::Facts { tetrahedral_reference_order: None, .. } => {}
        PreparedAtomToken::Facts { tetrahedral_reference_order: Some(reference_order), .. } => {
            validate_tetrahedral_references(graph, atom, reference_order)?;
        }
        #[cfg(test)]
        PreparedAtomToken::Fixed(text) if !text.is_empty() => {}
        #[cfg(test)]
        PreparedAtomToken::Fixed(_) => return Err(PreparedNonStereoError::EmptyAtomText(atom)),
        #[cfg(test)]
        PreparedAtomToken::Tetrahedral { reference_order, text_by_parity } => {
            if text_by_parity.iter().any(String::is_empty) {
                return Err(PreparedNonStereoError::EmptyTetrahedralAtomText(atom));
            }
            if text_by_parity[0] == text_by_parity[1] {
                return Err(PreparedNonStereoError::RepeatedTetrahedralAtomText(atom));
            }
            validate_tetrahedral_references(graph, atom, reference_order)?;
        }
    }
    Ok(())
}

fn validate_tetrahedral_references(
    graph: &crate::prepared::PreparedGraph,
    atom: AtomId,
    reference_order: &[TetrahedralLigand; 4],
) -> Result<(), PreparedNonStereoError> {
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
    Ok(())
}

#[cfg(test)]
fn test_atom_facts(
    graph: &crate::prepared::PreparedGraph,
    bond_tokens: &[NonStereoBondToken],
    atom: AtomId,
    symbol: &str,
    fixed_hydrogens: Option<u8>,
) -> AtomFacts {
    let element = ElementSymbol::new(symbol).ok();
    if let Some(element) = element {
        let valence = match symbol {
            "B" | "N" | "P" => 3_i16,
            "C" => 4,
            "O" | "S" => 2,
            "F" | "Cl" | "Br" | "I" => 1,
            _ => 0,
        };
        let bond_valence = graph
            .neighbors(atom)
            .expect("test atom exists")
            .iter()
            .map(|incident| match bond_tokens[incident.bond().index()] {
                NonStereoBondToken::Elided | NonStereoBondToken::Single => 1_i16,
                NonStereoBondToken::Double => 2,
                NonStereoBondToken::Triple => 3,
                NonStereoBondToken::Aromatic
                | NonStereoBondToken::DativeAToB
                | NonStereoBondToken::DativeBToA => 0,
            })
            .sum::<i16>();
        let hydrogens = fixed_hydrogens.unwrap_or_else(|| {
            u8::try_from((valence - bond_valence).max(0)).expect("test hydrogen count fits")
        });
        return AtomFacts::new(element, false, None, 0, hydrogens, 0, None)
            .expect("test atom facts are valid");
    }
    AtomFacts::new(
        ElementSymbol::new("*").expect("wildcard is an OpenSMILES element"),
        false,
        None,
        0,
        0,
        u64::try_from(atom.index() + 1).expect("test atom class fits"),
        None,
    )
    .expect("test atom facts are valid")
}

#[cfg(test)]
fn test_tetrahedral_facts(has_virtual_hydrogen: bool) -> AtomFacts {
    AtomFacts::new(
        ElementSymbol::new("C").expect("carbon is an OpenSMILES element"),
        false,
        None,
        0,
        u8::from(has_virtual_hydrogen),
        0,
        None,
    )
    .expect("tetrahedral test facts are valid")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PreparedNonStereoError {
    SourceBondMismatch { bond: BondId },
    AromaticDemand { atom: AtomId, total_valence: u32, non_graph_hydrogens: u8, fixed_bond_order_sum: u32, aromatic_degree: u32 },
    InfeasibleAromaticComponent { atom: AtomId },
    AromaticDativeIncidence { atom: AtomId, bond: BondId },
    MissingAromaticComponent { atom: AtomId },
    UnsupportedAromaticDirectionalCarrier { bond: BondId },
    AtomTextCountMismatch {
        expected: usize,
        actual: usize,
    },
    BondTokenCountMismatch {
        expected: usize,
        actual: usize,
    },
    EmptyAtomText(AtomId),
    EmptyTetrahedralAtomText(AtomId),
    RepeatedTetrahedralAtomText(AtomId),
    RepeatedTetrahedralLigand(AtomId),
    MultipleVirtualHydrogens(AtomId),
    TetrahedralLigandsDoNotMatchGraph(AtomId),
    UnknownDirectionalBond(BondId),
    RepeatedDirectionalRelation(BondId),
    DirectionalDoubleBondEndpoints {
        double_bond: BondId,
        left: AtomId,
        right: AtomId,
    },
    DirectionalDoubleBondToken(BondId),
    EmptyDirectionalCarrierSide {
        double_bond: BondId,
        endpoint: AtomId,
    },
    RepeatedDirectionalCarrier {
        double_bond: BondId,
        endpoint: AtomId,
        carrier: BondId,
    },
    DirectionalCarrierIsDoubleBond(BondId),
    DirectionalCarrierNotIncident {
        carrier: BondId,
        endpoint: AtomId,
    },
    InvalidDirectionalCarrierToken(BondId),
    ContradictoryDirectionalParity {
        carrier: BondId,
    },
}

impl fmt::Display for PreparedNonStereoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceBondMismatch { bond } => write!(formatter, "cannot prepare aromatic bond {bond:?}: both endpoints must supply aromatic atom facts and total valence"),
            Self::AromaticDemand { atom, total_valence, non_graph_hydrogens, fixed_bond_order_sum, aromatic_degree } => write!(formatter, "cannot prepare Kekule matching for {atom:?}: total valence {total_valence} minus non-graph hydrogens {non_graph_hydrogens}, fixed bond orders {fixed_bond_order_sum}, and aromatic degree {aromatic_degree} must require zero or one incident double bond"),
            Self::InfeasibleAromaticComponent { atom } => write!(formatter, "cannot prepare aromatic component containing {atom:?}: its required double bonds cannot be assigned to source aromatic edges without sharing an atom"),
            Self::AromaticDativeIncidence { atom, bond } => write!(formatter, "cannot prepare Kekule matching for {atom:?}: incident dative bond {bond:?} has no supported integer valence contribution"),
            Self::MissingAromaticComponent { atom } => write!(formatter, "cannot prepare aromatic atom {atom:?}: no incident source aromatic bond supplies a matching component"),
            Self::UnsupportedAromaticDirectionalCarrier { bond } => write!(formatter, "cannot prepare directional carrier {bond:?}: a chemical aromatic bond with variable Kekule order is not a supported directional role"),
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
            Self::UnknownDirectionalBond(bond) => {
                write!(formatter, "prepared directional bond {bond:?} does not exist")
            }
            Self::RepeatedDirectionalRelation(bond) => write!(
                formatter,
                "prepared directional double bond {bond:?} has more than one relation"
            ),
            Self::DirectionalDoubleBondEndpoints {
                double_bond,
                left,
                right,
            } => write!(
                formatter,
                "prepared directional double bond {double_bond:?} does not join {left:?} and {right:?}"
            ),
            Self::DirectionalDoubleBondToken(bond) => write!(
                formatter,
                "prepared directional double bond {bond:?} must use the Double base token"
            ),
            Self::EmptyDirectionalCarrierSide {
                double_bond,
                endpoint,
            } => write!(
                formatter,
                "prepared directional double bond {double_bond:?} has no carrier at {endpoint:?}"
            ),
            Self::RepeatedDirectionalCarrier {
                double_bond,
                endpoint,
                carrier,
            } => write!(
                formatter,
                "prepared directional double bond {double_bond:?} repeats carrier {carrier:?} at {endpoint:?}"
            ),
            Self::DirectionalCarrierIsDoubleBond(bond) => write!(
                formatter,
                "prepared directional carrier {bond:?} is the configured double bond"
            ),
            Self::DirectionalCarrierNotIncident { carrier, endpoint } => write!(
                formatter,
                "prepared directional carrier {carrier:?} is not incident to {endpoint:?}"
            ),
            Self::InvalidDirectionalCarrierToken(bond) => write!(
                formatter,
                "prepared directional carrier {bond:?} must use an elided, single, or aromatic base token"
            ),
            Self::ContradictoryDirectionalParity { carrier } => write!(
                formatter,
                "prepared directional parity is contradictory at carrier {carrier:?}"
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
    fn first_spelling_slot(regime: WriterRegime) -> usize {
        match regime {
            WriterRegime::Generic { .. } | WriterRegime::GenericCompact => 0,
            WriterRegime::RdkitRandomNoncanonical => 1,
        }
    }

    fn is_full_for(&self, regime: WriterRegime) -> bool {
        self.next_available_from(Self::first_spelling_slot(regime)).index()
            > self.maximum_spelling_label()
    }

    fn next_available(&self) -> RingLabelSlot {
        self.next_available_from(0)
    }

    fn next_available_from(&self, first: usize) -> RingLabelSlot {
        let mut candidate = RingLabelSlot(first);
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

    fn maximum_spelling_label(&self) -> usize {
        #[cfg(test)]
        {
            self.maximum_spelling_label.unwrap_or(MAX_RING_LABEL_INDEX)
        }
        #[cfg(not(test))]
        {
            MAX_RING_LABEL_INDEX
        }
    }

    fn allocate(&mut self, bond: BondId) -> RingLabelSlot {
        let slot = self.next_available();
        self.allocate_at(slot, bond);
        slot
    }

    fn allocate_at(&mut self, slot: RingLabelSlot, bond: BondId) {
        assert!(
            self.bonds_by_slot.values().all(|owner| *owner != bond),
            "one ring bond may own only one visible label"
        );
        assert_eq!(
            self.bonds_by_slot.insert(slot, bond),
            None,
            "a newly allocated visible ring label must be free"
        );
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

#[derive(Clone, Debug, PartialEq, Eq)]
enum PendingEmission {
    SegmentRootAtom(AtomId),
    InlineAtom(ChildEntrySpelling),
    BranchTraversalEmission(ChildEntrySpelling),
    BranchAtom(ChildEntrySpelling),
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
    BranchTraversalEmission {
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
        label: Option<usize>,
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
    pub(crate) tetrahedral_role_pattern_domains: Vec<(AtomId, Domain)>,
    pub(crate) directional_mark_domains: Vec<(BondId, Domain)>,
    pub(crate) directional_ring_plan_domains: Vec<(BondId, Domain, bool)>,
    pub(crate) directional_side_pattern_domains: Vec<(BondId, AtomId, Domain)>,
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
    Incomplete(WriterIncompleteness),
    Invariant(WriterInvariantFailure),
    Failed(E),
}

enum SuccessorAttempt<S, E> {
    Accepted(S),
    Rejected(CandidateRejection),
    Incomplete(WriterIncompleteness),
    Invariant(WriterInvariantFailure),
    Failed(E),
}

fn collect_attempts_fail_fast<S, E>(
    attempts: impl IntoIterator<Item = CandidateAttempt<S, E>>,
) -> Vec<CandidateAttempt<S, E>> {
    let mut collected = Vec::new();
    for attempt in attempts {
        let stop = matches!(
            attempt,
            CandidateAttempt::Incomplete(_)
                | CandidateAttempt::Invariant(_)
                | CandidateAttempt::Failed(_)
        );
        collected.push(attempt);
        if stop {
            break;
        }
    }
    collected
}

fn classify_missing_ring_probes<S, E>(
    probes: Vec<CandidateAttempt<S, E>>,
    incomplete: WriterIncompleteness,
) -> Vec<CandidateAttempt<S, E>> {
    let mut classified = Vec::new();
    let mut viable = false;
    for probe in probes {
        match probe {
            CandidateAttempt::Accepted { .. } => viable = true,
            CandidateAttempt::Rejected { .. }
            | CandidateAttempt::Incomplete(_)
            | CandidateAttempt::Invariant(_) => classified.push(probe),
            CandidateAttempt::Failed(failure) => {
                classified.push(CandidateAttempt::Failed(failure));
                return classified;
            }
        }
    }
    if viable {
        classified.push(CandidateAttempt::Incomplete(incomplete));
    }
    if classified.is_empty() {
        classified.push(CandidateAttempt::Rejected {
            reason: CandidateRejection::Contradiction,
        });
    }
    classified
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
    AllCandidatesSemanticallyRejected {
        candidate_count: usize,
    },
    UnresolvedTetrahedralFrame {
        atom: AtomId,
    },
    UnresolvedDirectionalCarrier {
        bond: BondId,
    },
    UnresolvedDirectionalSide {
        double_bond: BondId,
        endpoint: AtomId,
    },
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
            Self::UnresolvedTetrahedralFrame { atom } => write!(
                formatter,
                "tetrahedral atom {atom:?} completed without one exact procedural ligand order"
            ),
            Self::UnresolvedDirectionalCarrier { bond } => write!(
                formatter,
                "directional carrier {bond:?} completed without one physical mark"
            ),
            Self::UnresolvedDirectionalSide {
                double_bond,
                endpoint,
            } => write!(
                formatter,
                "directional side of {double_bond:?} at {endpoint:?} completed without one pattern"
            ),
        }
    }
}

impl std::error::Error for WriterInvariantFailure {}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum WriterIncompleteness {
    RdkitDirectionalAnchoring,
    DirectionalCarrierSelection {
        double_bond: BondId,
        endpoint: AtomId,
    },
}

impl fmt::Display for WriterIncompleteness {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RdkitDirectionalAnchoring => formatter.write_str(
                "RDKit directional anchoring currently supports one carrier per endpoint, distinct endpoints and carrier bonds across relations, and acyclic directional double/carrier bonds",
            ),
            Self::DirectionalCarrierSelection {
                double_bond,
                endpoint,
            } => write!(
                formatter,
                "directional carrier selection for double bond {double_bond:?} at {endpoint:?} is not implemented"
            ),
        }
    }
}

impl std::error::Error for WriterIncompleteness {}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ChoiceFailure<E> {
    Backend(E),
    Spelling(SpellingFailure),
    Incomplete(WriterIncompleteness),
    Invariant(WriterInvariantFailure),
}

impl<E: fmt::Display> fmt::Display for ChoiceFailure<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(failure) => write!(formatter, "constraint backend failure: {failure}"),
            Self::Spelling(failure) => failure.fmt(formatter),
            Self::Incomplete(failure) => {
                write!(formatter, "private writer incompleteness: {failure}")
            }
            Self::Invariant(failure) => write!(formatter, "writer invariant failure: {failure}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ChoiceFailure<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend(failure) => Some(failure),
            Self::Spelling(failure) => Some(failure),
            Self::Incomplete(failure) => Some(failure),
            Self::Invariant(failure) => Some(failure),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NonStereoWriterState<S> {
    surface: PreparedNonStereo,
    regime: WriterRegime,
    structural: WriterState<S>,
    labels: RingLabels,
    pending: Option<PendingEmission>,
    atom_spelling: Option<AtomSpellingProgress>,
}

impl<S: ConstraintSolver> NonStereoWriterState<S> {
    #[cfg(test)]
    pub(crate) fn initial(surface: &PreparedNonStereo) -> Result<Consistency<Self>, S::Failure> {
        Self::initial_with_component_order(surface, ComponentOrder::AnyOrder)
    }

    pub(crate) fn initial_with_component_order(
        surface: &PreparedNonStereo,
        component_order: ComponentOrder,
    ) -> Result<Consistency<Self>, S::Failure> {
        Self::initial_with_regime(surface, WriterRegime::Generic { component_order })
    }

    pub(crate) fn initial_with_regime(
        surface: &PreparedNonStereo,
        regime: WriterRegime,
    ) -> Result<Consistency<Self>, S::Failure> {
        Ok(
            WriterState::initial_with_component_order(
                surface.molecule(),
                regime.component_order(),
            )?
            .map(|structural| Self {
                surface: surface.clone(),
                regime,
                structural,
                labels: RingLabels {
                    #[cfg(test)]
                    maximum_spelling_label: Some(surface.test_maximum_spelling_label),
                    ..RingLabels::default()
                },
                pending: None,
                atom_spelling: None,
            }),
        )
    }

    pub(crate) fn active_atom(&self) -> Option<AtomId> {
        self.structural.active_atom()
    }

    #[cfg(test)]
    pub(crate) fn is_spelling_atom(&self) -> bool {
        self.atom_spelling.is_some()
    }

    pub(crate) const fn graph_is_complete(&self) -> bool {
        self.structural.graph_is_complete()
    }

    pub(crate) fn is_accepted(&self) -> bool {
        self.pending.is_none()
            && self.atom_spelling.is_none()
            && self.labels.is_clean()
            && self.structural.scope_is_complete()
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
        let pending = self.pending.as_ref().map(|pending| match pending {
            PendingEmission::SegmentRootAtom(root) => ObservedPending::ComponentAtom { root: *root },
            PendingEmission::InlineAtom(entry) => ObservedPending::InlineAtom {
                parent: self.surface.molecule().graph().bond(entry.incident.bond()).unwrap().other(entry.incident.atom()).unwrap(),
                child: entry.incident.atom(),
                bond: entry.incident.bond(),
            },
            PendingEmission::BranchTraversalEmission(entry) => {
                ObservedPending::BranchTraversalEmission {
                    parent: self.surface.molecule().graph().bond(entry.incident.bond()).unwrap().other(entry.incident.atom()).unwrap(),
                    child: entry.incident.atom(),
                    bond: entry.incident.bond(),
                }
            }
            PendingEmission::BranchAtom(entry) => ObservedPending::BranchAtom {
                parent: self.surface.molecule().graph().bond(entry.incident.bond()).unwrap().other(entry.incident.atom()).unwrap(),
                child: entry.incident.atom(),
                bond: entry.incident.bond(),
            },
            PendingEmission::RingOpeningLabel {
                incident,
                label_slot,
            } => ObservedPending::RingOpeningLabel {
                bond: incident.bond(),
                endpoint: active_endpoint(),
                label: Some(label_slot.index()),
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
        let tetrahedral_role_pattern_domains = self
            .surface
            .molecule()
            .graph()
            .atom_ids()
            .filter_map(|atom| {
                self.surface.tetrahedral_center(atom).map(|center| {
                    (
                        atom,
                        self.structural
                            .semantic_domain(center.role_pattern_variable),
                    )
                })
            })
            .collect();
        let directional_mark_domains = self
            .surface
            .molecule()
            .graph()
            .bond_ids()
            .filter_map(|bond| match self.surface.directional_status(bond) {
                PreparedDirectionalBondStatus::CompiledSite(site) => {
                    Some((bond, self.structural.semantic_domain(site.mark_variable)))
                }
                PreparedDirectionalBondStatus::Ordinary
                | PreparedDirectionalBondStatus::IncompleteSelection(_) => None,
            })
            .collect();
        let directional_ring_plan_domains = self
            .surface
            .molecule()
            .graph()
            .bond_ids()
            .filter_map(|bond| match self.surface.directional_status(bond) {
                PreparedDirectionalBondStatus::CompiledSite(site) => Some((
                    bond,
                    self.structural.semantic_domain(site.ring_plan_variable),
                    self.structural.factor_is_active(site.ring_plan_factor),
                )),
                PreparedDirectionalBondStatus::Ordinary
                | PreparedDirectionalBondStatus::IncompleteSelection(_) => None,
            })
            .collect();
        let directional_side_pattern_domains = self
            .surface
            .directional_sides
            .iter()
            .map(|side| {
                (
                    side.double_bond,
                    side.endpoint,
                    self.structural.semantic_domain(side.pattern_variable),
                )
            })
            .collect();
        ObservedNonStereoState {
            structural,
            tetrahedral_order_domains,
            tetrahedral_role_pattern_domains,
            directional_mark_domains,
            directional_ring_plan_domains,
            directional_side_pattern_domains,
            labels_by_bond,
            pending,
            maximum_spelling_label: self.labels.maximum_spelling_label(),
        }
    }

    pub(crate) fn choices(&self) -> Result<Vec<Choice<Self>>, ChoiceFailure<S::Failure>> {
        if self.is_accepted() {
            return Ok(Vec::new());
        }
        if let Some(pending) = self.pending.clone() {
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
                        return Err(ChoiceFailure::Incomplete(failure));
                    }
                    CandidateAttempt::Invariant(failure) => {
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
        if let Some(spelling) = &self.atom_spelling {
            let attempts = self.atom_spelling_attempts(spelling.clone());
            let mut choices = Vec::new();
            for attempt in attempts {
                match attempt {
                    CandidateAttempt::Accepted { text, successor } => {
                        choices.push(Choice { text, successor });
                    }
                    CandidateAttempt::Rejected { .. } => {}
                    CandidateAttempt::Incomplete(failure) => {
                        return Err(ChoiceFailure::Incomplete(failure));
                    }
                    CandidateAttempt::Invariant(failure) => {
                        return Err(ChoiceFailure::Invariant(failure));
                    }
                    CandidateAttempt::Failed(failure) => {
                        return Err(ChoiceFailure::Backend(failure));
                    }
                }
            }
            if choices.is_empty() {
                return Err(ChoiceFailure::Invariant(
                    WriterInvariantFailure::PendingEmissionRejected,
                ));
            }
            return Ok(choices);
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
            if candidate == StructuralCandidate::FinishSegment {
                panic!("component completion must already be normalized");
            }
            let attempts = match candidate {
                StructuralCandidate::RingOpen { incident }
                | StructuralCandidate::RingClose { incident, .. } => {
                    match self.surface.directional_status(incident.bond()) {
                        PreparedDirectionalBondStatus::CompiledSite(_) => {
                            self.attempt_directional_ring_candidate(candidate, incident)
                        }
                        PreparedDirectionalBondStatus::IncompleteSelection(incomplete) => self
                            .attempt_incomplete_ring_candidate(
                                candidate,
                                incident,
                                Self::incomplete_selection(incomplete),
                            ),
                        PreparedDirectionalBondStatus::Ordinary => match candidate {
                            StructuralCandidate::RingOpen { .. } => self
                                .attempt_ring_openings(candidate, incident, &[])
                                .map_err(ChoiceFailure::Backend)?,
                            StructuralCandidate::RingClose { .. } => self
                                .attempt_ring_closures(candidate, incident, &[])
                                .map_err(ChoiceFailure::Backend)?,
                            _ => unreachable!(),
                        },
                    }
                }
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
                        return Err(ChoiceFailure::Incomplete(failure));
                    }
                    CandidateAttempt::Invariant(failure) => {
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
        semantic_restrictions: &[(VariableId, Domain)],
    ) -> Result<Vec<CandidateAttempt<Self, S::Failure>>, S::Failure> {
        let active = self
            .structural
            .active_atom()
            .expect("ring opening spelling requires an active endpoint");
        let mut restrictions = self.parent_prefix_restriction(incident);
        restrictions.extend_from_slice(semantic_restrictions);
        let mut attempts = Vec::new();
        for (spelling, text, order_restriction, allowed) in self.ring_emission_specs(incident.bond(), active) {
            let mut restrictions = restrictions.clone();
            restrictions.extend(order_restriction);
            let structural = match self.attempt_candidate_with_transition(
                candidate,
                Some(allowed),
                &restrictions,
                &[],
            )? {
                Consistency::Consistent(structural) => structural,
                Consistency::Contradiction => {
                    attempts.push(CandidateAttempt::Rejected {
                        reason: CandidateRejection::Contradiction,
                    });
                    continue;
                }
            };
            let available = self.available_ring_label_options();
            if available.is_empty() {
                let first = RingLabels::first_spelling_slot(self.regime);
                attempts.push(self.ring_label_exhaustion_attempt(
                    self.labels.next_available_from(first).index(),
                ));
                continue;
            }
            match spelling {
                RingEndpointSpelling::Omit => {
                    for (label_slot, text) in available {
                        let mut labels = self.labels.clone();
                        labels.allocate_at(label_slot, incident.bond());
                        let attempt = self.finish_attempt(
                            text,
                            Self {
                                surface: self.surface.clone(),
                                regime: self.regime,
                                structural: structural.clone(),
                                labels,
                                pending: None,
                                atom_spelling: None,
                            },
                        );
                        match attempt {
                            CandidateAttempt::Failed(failure) => return Err(failure),
                            attempt => attempts.push(attempt),
                        }
                    }
                }
                RingEndpointSpelling::Emit => {
                    assert!(
                        !text.is_empty(),
                        "an emitted ring endpoint must have prepared bond text"
                    );
                    let slots = available
                        .into_iter()
                        .map(|(slot, _)| slot)
                        .fold(Vec::new(), |mut slots, slot| {
                            if slots.last() != Some(&slot) {
                                slots.push(slot);
                            }
                            slots
                        });
                    for label_slot in slots {
                        let mut labels = self.labels.clone();
                        labels.allocate_at(label_slot, incident.bond());
                        let attempt = self.finish_attempt(
                            text.to_owned(),
                            Self {
                                surface: self.surface.clone(),
                                regime: self.regime,
                                structural: structural.clone(),
                                labels,
                                pending: Some(PendingEmission::RingOpeningLabel { incident, label_slot }),
                                atom_spelling: None,
                            },
                        );
                        match attempt {
                            CandidateAttempt::Failed(failure) => return Err(failure),
                            CandidateAttempt::Invariant(_) => {
                                attempts.push(attempt);
                                break;
                            }
                            attempt => attempts.push(attempt),
                        }
                    }
                }
            }
        }
        Ok(attempts)
    }

    fn available_ring_label_options(&self) -> Vec<(RingLabelSlot, String)> {
        let maximum = self.labels.maximum_spelling_label();
        let first = RingLabels::first_spelling_slot(self.regime);
        match self.regime {
            WriterRegime::Generic { .. } => (first..=maximum)
                .map(RingLabelSlot)
                .filter(|slot| !self.labels.bonds_by_slot.contains_key(slot))
                .flat_map(|slot| {
                    ring_label_spellings(slot.index())
                        .into_iter()
                        .map(move |text| (slot, text))
                })
                .collect(),
            WriterRegime::GenericCompact | WriterRegime::RdkitRandomNoncanonical => {
                let slot = self.labels.next_available_from(first);
                if slot.index() <= maximum {
                    vec![(slot, ring_label_number_text(slot.index()))]
                } else {
                    Vec::new()
                }
            }
        }
    }

    fn label_text_options(&self, slot: RingLabelSlot) -> Vec<String> {
        if slot.index() > self.labels.maximum_spelling_label() {
            return Vec::new();
        }
        match self.regime {
            WriterRegime::Generic { .. } => ring_label_spellings(slot.index()),
            WriterRegime::GenericCompact => vec![ring_label_number_text(slot.index())],
            WriterRegime::RdkitRandomNoncanonical => {
                vec![ring_label_number_text(slot.index())]
            }
        }
    }

    fn ring_label_exhaustion_attempt(
        &self,
        next_label: usize,
    ) -> CandidateAttempt<Self, S::Failure> {
        CandidateAttempt::Rejected {
            reason: CandidateRejection::RingLabelUnavailable {
                next_label,
                maximum_label: self.labels.maximum_spelling_label(),
            },
        }
    }

    fn attempt_incomplete_ring_candidate(
        &self,
        candidate: StructuralCandidate,
        incident: AdjacentBond,
        incomplete: WriterIncompleteness,
    ) -> Vec<CandidateAttempt<Self, S::Failure>> {
        let probes = match candidate {
            StructuralCandidate::RingOpen { .. } => {
                self.attempt_ring_openings(candidate, incident, &[])
            }
            StructuralCandidate::RingClose { .. } => {
                self.attempt_ring_closures(candidate, incident, &[])
            }
            _ => unreachable!(),
        };
        match probes {
            Ok(probes) => classify_missing_ring_probes(probes, incomplete),
            Err(failure) => vec![CandidateAttempt::Failed(failure)],
        }
    }

    fn attempt_directional_ring_candidate(
        &self,
        candidate: StructuralCandidate,
        incident: AdjacentBond,
    ) -> Vec<CandidateAttempt<Self, S::Failure>> {
        let PreparedDirectionalBondStatus::CompiledSite(site) =
            self.surface.directional_status(incident.bond())
        else {
            panic!("directional ring attempt requires one compiled physical mark")
        };
        let result = match candidate {
            StructuralCandidate::RingOpen { .. } => {
                self.attempt_directional_ring_openings(candidate, incident, site)
            }
            StructuralCandidate::RingClose { .. } => {
                self.attempt_directional_ring_closures(candidate, incident, site)
            }
            _ => unreachable!(),
        };
        result.unwrap_or_else(|failure| vec![CandidateAttempt::Failed(failure)])
    }

    fn directional_ring_endpoint_alternatives(
        &self,
        incident: AdjacentBond,
        site: PreparedDirectionalSite,
    ) -> Vec<(Domain, Option<CarrierMark>, Option<&'static str>)> {
        let active = self
            .structural
            .active_atom()
            .expect("ring endpoint requires one active atom");
        let endpoint = self.surface.fixed_endpoint(incident.bond(), active);
        let current = self.structural.semantic_domain(site.ring_plan_variable);
        let mut alternatives = Vec::new();
        let omitted = DirectionalRingPlan::endpoint_domain(current, endpoint, None, false);
        if !omitted.is_empty() {
            alternatives.push((omitted, None, None));
        }
        for mark in [
            CarrierMark::Plain,
            CarrierMark::SlashAtFixedA,
            CarrierMark::BackslashAtFixedA,
        ] {
            let plans = DirectionalRingPlan::endpoint_domain(current, endpoint, Some(mark), true);
            if plans.is_empty() {
                continue;
            }
            let text = match mark {
                CarrierMark::Plain => self.surface.bond_text(incident.bond(), active),
                mark => mark
                    .directional_text(endpoint == FixedBondEndpoint::A)
                    .expect("marked ring endpoint must render one directional glyph"),
            };
            assert!(
                !text.is_empty(),
                "emitted ring endpoint requires visible text"
            );
            alternatives.push((plans, Some(mark), Some(text)));
        }
        alternatives
    }

    fn attempt_directional_ring_openings(
        &self,
        candidate: StructuralCandidate,
        incident: AdjacentBond,
        site: PreparedDirectionalSite,
    ) -> Result<Vec<CandidateAttempt<Self, S::Failure>>, S::Failure> {
        let mut attempts = Vec::new();
        for (plans, mark, text) in self.directional_ring_endpoint_alternatives(incident, site) {
            let mut restrictions = self.parent_prefix_restriction(incident);
            restrictions.push((site.ring_plan_variable, plans));
            if let Some(mark) = mark {
                restrictions.push((site.mark_variable, mark.singleton_domain()));
            }
            let structural = match self.attempt_candidate_with_transition(
                candidate,
                Some(BondRole::Ring.singleton_domain()),
                &restrictions,
                &[site.ring_plan_factor],
            )? {
                Consistency::Consistent(structural) => structural,
                Consistency::Contradiction => {
                    attempts.push(CandidateAttempt::Rejected {
                        reason: CandidateRejection::Contradiction,
                    });
                    continue;
                }
            };
            let available = self.available_ring_label_options();
            if available.is_empty() {
                let first = RingLabels::first_spelling_slot(self.regime);
                attempts.push(self.ring_label_exhaustion_attempt(
                    self.labels.next_available_from(first).index(),
                ));
                continue;
            }
            match text {
                Some(text) => {
                    let slots = available
                        .into_iter()
                        .map(|(slot, _)| slot)
                        .fold(Vec::new(), |mut slots, slot| {
                            if slots.last() != Some(&slot) {
                                slots.push(slot);
                            }
                            slots
                        });
                    for label_slot in slots {
                        let mut labels = self.labels.clone();
                        labels.allocate_at(label_slot, incident.bond());
                        let attempt = self.finish_attempt(
                            text.to_owned(),
                            Self {
                                surface: self.surface.clone(),
                                regime: self.regime,
                                structural: structural.clone(),
                                labels,
                                pending: Some(PendingEmission::RingOpeningLabel { incident, label_slot }),
                                atom_spelling: None,
                            },
                        );
                        let stop = matches!(
                            attempt,
                            CandidateAttempt::Incomplete(_)
                                | CandidateAttempt::Invariant(_)
                                | CandidateAttempt::Failed(_)
                        );
                        attempts.push(attempt);
                        if stop {
                            break;
                        }
                    }
                }
                None => {
                    for (label_slot, label_text) in available {
                        let mut labels = self.labels.clone();
                        labels.allocate_at(label_slot, incident.bond());
                        let attempt = self.finish_attempt(
                            label_text,
                            Self {
                                surface: self.surface.clone(),
                                regime: self.regime,
                                structural: structural.clone(),
                                labels,
                                pending: None,
                                atom_spelling: None,
                            },
                        );
                        match attempt {
                            CandidateAttempt::Failed(failure) => return Err(failure),
                            attempt => attempts.push(attempt),
                        }
                    }
                }
            }
        }
        Ok(attempts)
    }

    fn attempt_directional_ring_closures(
        &self,
        candidate: StructuralCandidate,
        incident: AdjacentBond,
        site: PreparedDirectionalSite,
    ) -> Result<Vec<CandidateAttempt<Self, S::Failure>>, S::Failure> {
        assert!(
            self.structural.factor_is_active(site.ring_plan_factor),
            "directional ring closure requires its opening placement factor"
        );
        let label_slot = self.labels.slot_for_bond(incident.bond());
        let mut attempts = Vec::new();
        for (plans, mark, text) in self.directional_ring_endpoint_alternatives(incident, site) {
            let mut restrictions = self.parent_prefix_restriction(incident);
            restrictions.push((site.ring_plan_variable, plans));
            if let Some(mark) = mark {
                restrictions.push((site.mark_variable, mark.singleton_domain()));
            }
            let structural = match self.attempt_candidate_with_transition(
                candidate,
                Some(BondRole::Ring.singleton_domain()),
                &restrictions,
                &[],
            )? {
                Consistency::Consistent(structural) => structural,
                Consistency::Contradiction => {
                    attempts.push(CandidateAttempt::Rejected {
                        reason: CandidateRejection::Contradiction,
                    });
                    continue;
                }
            };
            assert!(structural
                .semantic_domain(site.mark_variable)
                .is_singleton());
            assert!(structural
                .semantic_domain(site.ring_plan_variable)
                .is_singleton());
            match text {
                Some(text) => {
                    let attempt = self.finish_attempt(
                        text.to_owned(),
                        Self {
                            surface: self.surface.clone(),
                            regime: self.regime,
                            structural,
                            labels: self.labels.clone(),
                            pending: Some(PendingEmission::RingClosureLabel {
                                incident,
                                label_slot,
                            }),
                            atom_spelling: None,
                        },
                    );
                    let stop = matches!(
                        attempt,
                        CandidateAttempt::Incomplete(_)
                            | CandidateAttempt::Invariant(_)
                            | CandidateAttempt::Failed(_)
                    );
                    attempts.push(attempt);
                    if stop {
                        break;
                    }
                }
                None => {
                    for label_text in self.label_text_options(label_slot) {
                        let mut labels = self.labels.clone();
                        labels.release(label_slot, incident.bond());
                        let attempt = self.finish_attempt(
                            label_text,
                            Self {
                                surface: self.surface.clone(),
                                regime: self.regime,
                                structural: structural.clone(),
                                labels,
                                pending: None,
                                atom_spelling: None,
                            },
                        );
                        match attempt {
                            CandidateAttempt::Failed(failure) => return Err(failure),
                            attempt => attempts.push(attempt),
                        }
                    }
                }
            }
        }
        Ok(attempts)
    }

    fn attempt_ring_closures(
        &self,
        candidate: StructuralCandidate,
        incident: AdjacentBond,
        semantic_restrictions: &[(VariableId, Domain)],
    ) -> Result<Vec<CandidateAttempt<Self, S::Failure>>, S::Failure> {
        let active = self
            .structural
            .active_atom()
            .expect("ring closure spelling requires an active endpoint");
        let mut restrictions = self.parent_prefix_restriction(incident);
        restrictions.extend_from_slice(semantic_restrictions);
        let label_slot = self.labels.slot_for_bond(incident.bond());
        let mut attempts = Vec::new();
        for (spelling, text, order_restriction, allowed) in self.ring_emission_specs(incident.bond(), active) {
            let mut restrictions = restrictions.clone();
            restrictions.extend(order_restriction);
            assert!(
                allowed.is_singleton(),
                "opening and closure projections must resolve one representation plan"
            );
            let structural = match self.attempt_candidate_with_transition(
                candidate,
                Some(allowed),
                &restrictions,
                &[],
            )? {
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
                    if label_slot.index() > self.labels.maximum_spelling_label() {
                        attempts.push(self.ring_label_exhaustion_attempt(label_slot.index()));
                        continue;
                    }
                    let mut labels = self.labels.clone();
                    labels.release(label_slot, incident.bond());
                    let successor = Self {
                        surface: self.surface.clone(),
                        regime: self.regime,
                        structural,
                        labels,
                        pending: None,
                    atom_spelling: None,
};
                    match successor.normalize_and_check() {
                        SuccessorAttempt::Accepted(successor) => {
                            for text in self.label_text_options(label_slot) {
                                attempts.push(CandidateAttempt::Accepted {
                                    text,
                                    successor: successor.clone(),
                                });
                            }
                            continue;
                        }
                        SuccessorAttempt::Rejected(reason) => CandidateAttempt::Rejected { reason },
                        SuccessorAttempt::Incomplete(failure) => {
                            CandidateAttempt::Incomplete(failure)
                        }
                        SuccessorAttempt::Invariant(failure) => {
                            CandidateAttempt::Invariant(failure)
                        }
                        SuccessorAttempt::Failed(failure) => {
                            return Err(failure);
                        }
                    }
                }
                RingEndpointSpelling::Emit => {
                    assert!(
                        !text.is_empty(),
                        "an emitted ring endpoint must have prepared bond text"
                    );
                    self.finish_attempt(
                        text.to_owned(),
                        Self {
                            surface: self.surface.clone(),
                            regime: self.regime,
                            structural,
                            labels: self.labels.clone(),
                            pending: Some(PendingEmission::RingClosureLabel {
                                incident,
                                label_slot,
                            }),
                        atom_spelling: None,
},
                    )
                }
            };
            match attempt {
                CandidateAttempt::Failed(failure) => return Err(failure),
                CandidateAttempt::Invariant(_) => {
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
        context: &LocalLayoutContext,
    ) -> Vec<(AtomSpellingStart, Vec<(VariableId, Domain)>, Vec<FactorId>)> {
        let facts = *self.surface.atom_facts(atom);
        let center = self.surface.tetrahedral_center(atom);
        let parities = if center.is_some() { TetrahedralParity::ALL.into_iter().map(Some).collect::<Vec<_>>() } else { vec![None] };
        let mode_variable = self.surface.atom_aromatic_modes[atom.index()];
        let case_free_wildcard = facts.aromatic && facts.element.as_str() == "*";
        let modes = if case_free_wildcard {
            vec![0]
        } else {
            mode_variable.map(|variable| self.structural.semantic_domain(variable).iter().filter(|mode| *mode == 0 || matches!(self.regime, WriterRegime::Generic { .. } | WriterRegime::GenericCompact)).collect::<Vec<_>>()).unwrap_or_else(|| vec![0])
        };
        let mut specs = Vec::new();
        for parity in parities {
            for mode in &modes {
                let mut spelling_context = self.surface.atom_spelling_context(atom, parity);
                spelling_context.aromatic_spelling = if *mode == 1 { AromaticSpelling::Kekule } else { AromaticSpelling::Source };
                let mut restrictions = Vec::new();
                let mut activate = Vec::new();
                if let Some(variable) = mode_variable.filter(|_| !case_free_wildcard) { restrictions.push((variable, Domain::singleton(*mode).unwrap())); }
                if let Some(center) = center {
                    assert_eq!(context.order.atom, atom);
                    assert_eq!(context.order.entry_bond, entry_bond);
                    assert!(context.order.emitted_bonds.is_empty());
                    restrictions.push((center.order_variable, center.token_domain(entry_bond, parity.unwrap())));
                    restrictions.push((center.role_pattern_variable, self.local_role_pattern_domain(center, context)));
                    activate.push(center.layout_factor(entry_bond));
                } else if let Some(entry_bond) = entry_bond {
                    restrictions.extend(self.surface.ordinary_order_pairs[atom.index()].iter()
                        .filter_map(|pair| {
                            if pair.first == entry_bond {
                                Some((pair.variable, Domain::singleton(0).unwrap()))
                            } else if pair.second == entry_bond {
                                Some((pair.variable, Domain::singleton(1).unwrap()))
                            } else { None }
                        }));
                }
                let source_context = self.surface.atom_source_context(atom);
                let policy = if self.regime == WriterRegime::GenericCompact { AtomSpellingPolicy::Compact } else { AtomSpellingPolicy::Maximal };
                specs.push((AtomSpellingStart::Grammar { atom, state: AtomSpellingState::initial_with_policy(facts, spelling_context, policy, source_context.atom_class_property_present) }, restrictions, activate));
            }
        }
        specs
    }

    fn local_role_pattern_domain(
        &self,
        center: &PreparedTetrahedralCenter,
        context: &LocalLayoutContext,
    ) -> Domain {
        let bond_count = center.bond_pattern_bits.len();
        let contextual_bonds = context
            .order
            .entry_bond
            .into_iter()
            .chain(context.waiting_ring_bonds.iter().copied())
            .chain(context.residual_attachment_bonds.iter().flatten().copied())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            contextual_bonds,
            center
                .bond_pattern_bits
                .iter()
                .map(|(bond, _)| *bond)
                .collect(),
            "prospective tetrahedral context must classify every explicit ligand bond"
        );
        let domain = Domain::from_indices((0..(1_u8 << bond_count)).filter(|pattern| {
            let is_ring = |bond| pattern & (1_u8 << center.pattern_bit(bond)) != 0;
            if context.order.entry_bond.is_some_and(|bond| is_ring(bond)) {
                return false;
            }
            if !context.waiting_ring_bonds.iter().copied().all(is_ring) {
                return false;
            }
            self.structural.prepared_topology() == crate::model::TraversalTopologyRequirement::Forest
                || context.residual_attachment_bonds.iter().all(|attachment| {
                    attachment
                        .iter()
                        .copied()
                        .filter(|bond| !is_ring(*bond))
                        .count()
                        == 1
                })
        }))
        .unwrap();
        assert!(
            !domain.is_empty(),
            "a prospective writer frame must admit one local role pattern"
        );
        domain
    }

    fn parent_prefix_restriction(&self, incident: AdjacentBond) -> Vec<(VariableId, Domain)> {
        let local = self.structural.active_local_bond_order();
        if let Some(center) = self.surface.tetrahedral_center(local.atom) {
            let mut emitted_bonds = local.emitted_bonds;
            emitted_bonds.push(incident.bond());
            return vec![(
                center.order_variable,
                center.prefix_domain_with_bond_order(local.entry_bond, &emitted_bonds),
            )];
        }
        self.surface.ordinary_order_pairs[local.atom.index()].iter().filter_map(|pair| {
            let other = if pair.first == incident.bond() { pair.second }
                else if pair.second == incident.bond() { pair.first }
                else { return None };
            if local.emitted_bonds.contains(&other) || local.entry_bond == Some(other) {
                return None;
            }
            Some((pair.variable, Domain::singleton(u8::from(pair.second == incident.bond())).unwrap()))
        }).collect()
    }

    fn chemical_bond_specs(&self, bond: BondId, from: AtomId) -> Vec<(u8, &'static str, Vec<(VariableId, Domain)>, Domain)> {
        if self.regime == WriterRegime::GenericCompact {
            let graph = self.surface.molecule().graph();
            let other = graph.bond(bond).unwrap().other(from).unwrap();
            if let Some(variable) = self.surface.bond_chemical_orders[bond.index()] {
                let from_mode = self.surface.atom_aromatic_modes[from.index()];
                let other_mode = self.surface.atom_aromatic_modes[other.index()];
                let modes = |mode: Option<VariableId>| mode.map(|v| self.structural.semantic_domain(v).iter().collect::<Vec<_>>()).unwrap_or_else(|| vec![1]);
                let mut specs = Vec::new();
                for order in self.structural.semantic_domain(variable).iter() {
                    let order_restriction = vec![(variable, Domain::singleton(order).unwrap())];
                    match order {
                        0 => {
                            let implicit = self.surface.atom_facts(from).element.as_str() != "*"
                                && self.surface.atom_facts(other).element.as_str() != "*";
                            specs.push((order, if implicit { "" } else { ":" }, order_restriction,
                                if implicit { BondRepresentation::elided_domain() } else { compact_explicit_domain() }));
                        }
                        1 => {
                            if self.surface.atom_facts(from).element.as_str() == "*"
                                || self.surface.atom_facts(other).element.as_str() == "*" {
                                specs.push((order, "", order_restriction, BondRepresentation::elided_domain()));
                            } else {
                                for from_case in modes(from_mode) {
                                    let mut from_restrictions = order_restriction.clone();
                                    if let Some(v) = from_mode { from_restrictions.push((v, Domain::singleton(from_case).unwrap())); }
                                    if from_case == 1 {
                                        specs.push((order, "", from_restrictions, BondRepresentation::elided_domain()));
                                        continue;
                                    }
                                    for other_case in modes(other_mode) {
                                        let mut restrictions = from_restrictions.clone();
                                        if let Some(v) = other_mode { restrictions.push((v, Domain::singleton(other_case).unwrap())); }
                                        let both_lower = other_case == 0;
                                        specs.push((order, if both_lower { "-" } else { "" }, restrictions,
                                            if both_lower { compact_explicit_domain() } else { BondRepresentation::elided_domain() }));
                                    }
                                }
                            }
                        }
                        2 => specs.push((order, "=", order_restriction, compact_explicit_domain())),
                        _ => unreachable!(),
                    }
                }
                return specs;
            }
            let token = self.surface.bond_tokens[bond.index()];
            if matches!(token, NonStereoBondToken::Elided | NonStereoBondToken::Single) {
                let from_aromatic = self.compact_emitted_aromatic(from, None);
                let other_mode = self.surface.atom_aromatic_modes[other.index()];
                let other_modes = other_mode.map(|variable| self.structural.semantic_domain(variable).iter().collect::<Vec<_>>()).unwrap_or_else(|| vec![0]);
                return other_modes.into_iter().map(|mode| {
                    let other_aromatic = self.compact_emitted_aromatic(other, Some(mode));
                    let explicit = from_aromatic && other_aromatic;
                    (0, if explicit { "-" } else { "" }, other_mode.map(|variable| vec![(variable, Domain::singleton(mode).unwrap())]).unwrap_or_default(),
                        if explicit { compact_explicit_domain() } else { BondRepresentation::elided_domain() })
                }).collect();
            }
            return vec![(0, self.surface.bond_text(bond, from), Vec::new(), compact_explicit_domain())];
        }
        let Some(variable) = self.surface.bond_chemical_orders[bond.index()] else {
            if matches!(self.regime, WriterRegime::Generic { .. })
                && self.surface.bond_tokens[bond.index()] == NonStereoBondToken::Elided
            {
                let other = self.surface.molecule().graph().bond(bond).unwrap().other(from).unwrap();
                if self.compact_emitted_aromatic(from, None) {
                    if let Some(mode_variable) = self.surface.atom_aromatic_modes[other.index()] {
                        return self.structural.semantic_domain(mode_variable).iter().flat_map(|mode| {
                            let restriction = vec![(mode_variable, Domain::singleton(mode).unwrap())];
                            let explicit = (0, "-", restriction.clone(), BondRepresentation::explicit_domain());
                            if self.compact_emitted_aromatic(other, Some(mode)) {
                                vec![explicit]
                            } else {
                                vec![(0, "", restriction, BondRepresentation::elided_domain()), explicit]
                            }
                        }).collect();
                    }
                }
                return vec![
                    (0, "", Vec::new(), BondRepresentation::elided_domain()),
                    (0, "-", Vec::new(), BondRepresentation::explicit_domain()),
                ];
            }
            return vec![(0, self.surface.bond_text(bond, from), Vec::new(), self.surface.bond_tokens[bond.index()].representation_domain())];
        };
        let mut specs = Vec::new();
        for order in self.structural.semantic_domain(variable).iter() {
            let text = match order { 0 => ":", 1 => "-", 2 => "=", _ => unreachable!("matching order domain was prepared") };
            let restriction = (variable, Domain::singleton(order).unwrap());
            specs.push((order, text, vec![restriction], BondRepresentation::explicit_domain()));
            if let Some(mut implicit) = self.implicit_aromatic_bond_restrictions(bond, from, order) {
                implicit.insert(0, restriction);
                specs.push((order, "", implicit, BondRepresentation::elided_domain()));
            }
        }
        specs
    }

    fn implicit_aromatic_bond_restrictions(&self, bond: BondId, from: AtomId, order: u8) -> Option<Vec<(VariableId, Domain)>> {
        let other = self.surface.molecule().graph().bond(bond).unwrap().other(from).unwrap();
        let from_aromatic = self.compact_emitted_aromatic(from, None);
        let other_variable = self.surface.atom_aromatic_modes[other.index()];
        let other_can_spell_aromatic = other_variable.is_some() && self.surface.atom_facts(other).element.as_str() != "*";
        let from_wildcard = self.surface.atom_facts(from).element.as_str() == "*";
        let other_wildcard = self.surface.atom_facts(other).element.as_str() == "*";
        match order {
            // RDKit sanitization retains the admitted aromatic wildcard-ring
            // witness even when its omitted wildcard incidence parses as single.
            0 if (from_aromatic || from_wildcard)
                && (other_can_spell_aromatic || other_wildcard)
                && !(from_wildcard && other_wildcard) => Some(Vec::new()),
            1 if !from_aromatic || !other_can_spell_aromatic => Some(Vec::new()),
            1 => Some(vec![(other_variable.unwrap(), Domain::singleton(1).unwrap())]),
            _ => None,
        }
    }

    fn compact_emitted_aromatic(&self, atom: AtomId, mode: Option<u8>) -> bool {
        let facts = self.surface.atom_facts(atom);
        if !facts.aromatic || facts.element.as_str() == "*" { return false; }
        match self.surface.atom_aromatic_modes[atom.index()] {
            Some(variable) => mode.unwrap_or_else(|| {
                let domain = self.structural.semantic_domain(variable);
                assert!(domain.is_singleton(), "emitted atom mode must be selected before its bond");
                domain.iter().next().unwrap()
            }) == 0,
            None => false,
        }
    }

    fn ring_emission_specs(&self, bond: BondId, from: AtomId) -> Vec<(RingEndpointSpelling, &'static str, Vec<(VariableId, Domain)>, Domain)> {
        let current = self.structural.bond_decision_domain(bond);
        let mut specs = Vec::new();
        for (_, text, restriction, placements) in self.chemical_bond_specs(bond, from) {
            for spelling in [RingEndpointSpelling::Omit, RingEndpointSpelling::Emit] {
                let allowed = current.intersect(placements).intersect(self.surface.ring_endpoint_domain(bond, from, spelling));
                if !allowed.is_empty() { specs.push((spelling, text, restriction.clone(), allowed)); }
            }
        }
        specs
    }

    fn traversal_emission_specs(
        &self,
        bond: BondId,
        from: AtomId,
    ) -> Result<Vec<TraversalEmissionSpec>, IncompleteDirectionalSelection> {
        match self.surface.directional_status(bond) {
            PreparedDirectionalBondStatus::Ordinary => {
                Ok(self.chemical_bond_specs(bond, from).into_iter().map(|(_, text, semantic_restrictions, _)| {
                    TraversalEmissionSpec { text, semantic_restrictions }
                }).collect())
            },
            PreparedDirectionalBondStatus::IncompleteSelection(incomplete) => Err(incomplete),
            PreparedDirectionalBondStatus::CompiledSite(site) => {
                let topology = self
                    .surface
                    .molecule()
                    .graph()
                    .bond(bond)
                    .expect("prepared directional carrier must belong to the graph");
                let from_fixed_a = if topology.a() == from {
                    true
                } else if topology.b() == from {
                    false
                } else {
                    panic!("directional emission requires one carrier endpoint")
                };
                Ok(self
                    .structural
                    .semantic_domain(site.mark_variable)
                    .iter()
                    .map(|value| {
                        let mark = CarrierMark::from_value(value).expect(
                            "prepared directional mark domain has only carrier-mark values",
                        );
                        TraversalEmissionSpec {
                            text: mark
                                .directional_text(from_fixed_a)
                                .unwrap_or_else(|| self.surface.bond_text(bond, from)),
                            semantic_restrictions: vec![(site.mark_variable, mark.singleton_domain())],
                        }
                    })
                    .collect())
            }
        }
    }

    fn directional_anchor_restrictions(
        &self,
        atom: AtomId,
        from: AtomId,
        entry_bond: Option<BondId>,
    ) -> Result<Vec<(VariableId, Domain)>, WriterIncompleteness> {
        if self.regime != WriterRegime::RdkitRandomNoncanonical {
            return Ok(Vec::new());
        }
        let graph = self.surface.molecule().graph();
        let mut restrictions = Vec::new();
        for side in self
            .surface
            .directional_sides
            .iter()
            .filter(|side| side.endpoint == atom)
        {
            let double_bond = graph
                .bond(side.double_bond)
                .expect("prepared directional double bond must belong to the graph");
            let other_endpoint = double_bond
                .other(atom)
                .expect("prepared directional side endpoint must be incident");
            if self.structural.atom_is_visited(other_endpoint) {
                continue;
            }
            let [carrier] = side.carriers.as_ref() else {
                return Err(WriterIncompleteness::RdkitDirectionalAnchoring);
            };
            let shared = self
                .surface
                .directional_sides
                .iter()
                .flat_map(|side| side.carriers.iter())
                .filter(|other| other.bond == carrier.bond)
                .count()
                > 1;
            if shared {
                return Err(WriterIncompleteness::RdkitDirectionalAnchoring);
            }
            let source = if entry_bond == Some(carrier.bond) {
                from
            } else {
                atom
            };
            let carrier_topology = graph
                .bond(carrier.bond)
                .expect("prepared directional carrier must belong to the graph");
            let from_fixed_a = if carrier_topology.a() == source {
                true
            } else if carrier_topology.b() == source {
                false
            } else {
                panic!("directional anchor source must be incident to its carrier")
            };
            let mark = if from_fixed_a {
                CarrierMark::SlashAtFixedA
            } else {
                CarrierMark::BackslashAtFixedA
            };
            let PreparedDirectionalBondStatus::CompiledSite(site) =
                self.surface.directional_status(carrier.bond)
            else {
                return Err(WriterIncompleteness::RdkitDirectionalAnchoring);
            };
            restrictions.push((site.mark_variable, mark.singleton_domain()));
        }
        Ok(restrictions)
    }

    const fn incomplete_selection(
        incomplete: IncompleteDirectionalSelection,
    ) -> WriterIncompleteness {
        WriterIncompleteness::DirectionalCarrierSelection {
            double_bond: incomplete.double_bond,
            endpoint: incomplete.endpoint,
        }
    }

    fn attempt_candidate_with_transition(
        &self,
        candidate: StructuralCandidate,
        allowed_representations: Option<Domain>,
        restrictions: &[(VariableId, Domain)],
        activate: &[FactorId],
    ) -> Result<Consistency<WriterState<S>>, S::Failure> {
        let prepared = matches!(candidate, StructuralCandidate::BranchChild { .. } | StructuralCandidate::InlineChild { .. })
            .then(|| self.structural.prepare_child_entry(candidate));
        self.attempt_candidate_with_prepared_transition(candidate, prepared.as_ref(), allowed_representations, restrictions, activate)
    }

    fn attempt_candidate_with_prepared_transition(
        &self,
        candidate: StructuralCandidate,
        prepared: Option<&PreparedChildEntry>,
        allowed_representations: Option<Domain>,
        restrictions: &[(VariableId, Domain)],
        activate: &[FactorId],
    ) -> Result<Consistency<WriterState<S>>, S::Failure> {
        if prepared.is_some()
            && self.labels.has_open_labels()
            && self.labels.is_full_for(self.regime)
            && !prepared.unwrap().may_release_held_ring_before_opening(
                self.surface.molecule().graph(),
                &self.labels.bonds_by_slot.values().copied().collect::<Vec<_>>(),
            )
        {
            return Ok(Consistency::Contradiction);
        }
        let mut restrictions = restrictions.to_vec();
        let mut activate = activate.to_vec();
        if let Some((budget, factors)) = self.ring_resource_entry_constraint(candidate, prepared) {
            restrictions.extend(budget);
            activate.extend(factors);
        }
        if let Some((schedule_restrictions, schedule_factors)) = self.event_schedule_transition(candidate) {
            restrictions.extend(schedule_restrictions);
            activate.extend(schedule_factors);
        }
        if let Some(prepared) = prepared {
            assert!(allowed_representations.is_none());
            self.structural.attempt_prepared_child_with_semantic_transition(candidate, prepared, &restrictions, &activate)
        } else if allowed_representations.is_none() && restrictions.is_empty() && activate.is_empty() {
            self.structural.attempt_candidate(candidate)
        } else if restrictions.is_empty() && activate.is_empty() {
            self.structural.attempt_candidate_with_bond_refinement(
                candidate,
                allowed_representations.expect("ring transition must retain a representation"),
            )
        } else {
            self.structural.attempt_candidate_with_semantic_transition(
                candidate,
                allowed_representations,
                &restrictions,
                &activate,
            )
        }
    }

    // TODO: Legacy EventSchedule copies each observed writer event into a whole-future CSP schedule.
    // Keep its exact admission check until current walker state and affected
    // constraints can prove the same productive labelled transitions.
    fn event_schedule_transition(
        &self,
        candidate: StructuralCandidate,
    ) -> Option<(Vec<(VariableId, Domain)>, Vec<FactorId>)> {
        let factor = self.surface.event_schedule_factor?;
        let active = self.structural.factor_is_active(factor);
        let mut restrictions = Vec::new();
        let mut factors = Vec::new();
        if !active {
            let capacity = schedule_namespace_capacity(self.regime, self.labels.maximum_spelling_label());
            restrictions.extend(self.surface.event_schedule_capacity_bits.iter().enumerate().map(|(bit, variable)| {
                (*variable, Domain::singleton(((capacity >> bit) & 1) as u8).unwrap())
            }));
            factors.push(factor);
        }
        let next_rank = self.structural.visited_atom_count();
        if candidate == StructuralCandidate::CloseBranch {
            let root = self.structural.closing_branch_root().expect("closing branch must have a root");
            let visit = &self.surface.event_schedule_visits[root.index()];
            let end = next_rank.checked_sub(1).expect("closing branch must contain a visit");
            restrictions.extend(visit.end_bits.iter().enumerate().map(|(rank, bit)| {
                (*bit, Domain::singleton(u8::from(rank <= end)).unwrap())
            }));
        } else {
            let entry = match candidate {
                StructuralCandidate::Root { atom } => Some((atom, EntryKind::Root)),
                StructuralCandidate::BranchChild { incident } => Some((incident.atom(), EntryKind::BondBranch)),
                StructuralCandidate::InlineChild { incident } => Some((incident.atom(), EntryKind::BondInline)),
                StructuralCandidate::DotBranchRoot { atom } => Some((atom, EntryKind::DotBranch)),
                StructuralCandidate::DotSegmentRoot { atom } => Some((atom, EntryKind::DotContinuation)),
                StructuralCandidate::RingOpen { .. } | StructuralCandidate::RingClose { .. }
                | StructuralCandidate::FinishSegment => None,
                StructuralCandidate::CloseBranch => unreachable!(),
            };
            if let Some((atom, kind)) = entry {
                let visit = &self.surface.event_schedule_visits[atom.index()];
                restrictions.extend(visit.rank_bits.iter().enumerate().map(|(bit, variable)| {
                    (*variable, Domain::singleton(((next_rank >> bit) & 1) as u8).unwrap())
                }));
                restrictions.push((visit.kind, Domain::singleton(kind as u8).unwrap()));
                for root in self.structural.open_branch_roots() {
                    restrictions.push((self.surface.event_schedule_visits[root.index()].end_bits[next_rank],
                        Domain::singleton(1).unwrap()));
                }
                if matches!(kind, EntryKind::BondBranch | EntryKind::DotBranch) {
                    restrictions.push((visit.end_bits[next_rank], Domain::singleton(1).unwrap()));
                }
            }
        }
        if restrictions.is_empty() && factors.is_empty() { None } else { Some((restrictions, factors)) }
    }

    fn ring_resource_entry_constraint(
        &self,
        candidate: StructuralCandidate,
        prepared: Option<&PreparedChildEntry>,
    ) -> Option<(Vec<(VariableId, Domain)>, Vec<FactorId>)> {
        let target = match candidate {
            StructuralCandidate::Root { atom }
            | StructuralCandidate::DotBranchRoot { atom }
            | StructuralCandidate::DotSegmentRoot { atom } => atom,
            StructuralCandidate::BranchChild { incident }
            | StructuralCandidate::InlineChild { incident } => incident.atom(),
            StructuralCandidate::RingOpen { .. }
            | StructuralCandidate::RingClose { .. }
            | StructuralCandidate::CloseBranch
            | StructuralCandidate::FinishSegment => return None,
        };
        let degree = self.surface.molecule().graph().neighbors(target)
            .expect("prepared atom must have adjacency").len();
        let first = RingLabels::first_spelling_slot(self.regime);
        let maximum = self.labels.maximum_spelling_label();
        let capacity = maximum.saturating_add(1).saturating_sub(first);
        let occupied = self.labels.bonds_by_slot.keys().filter(|slot| {
            (first..=maximum).contains(&slot.index())
        }).count();
        let free = capacity.saturating_sub(occupied);
        if free >= degree { return None; }
        let context = match candidate {
            StructuralCandidate::Root { atom } => {
                self.structural.prospective_root_layout_context(atom)
            }
            StructuralCandidate::DotBranchRoot { atom } => {
                self.structural.prospective_dot_branch_layout_context(atom)
            }
            StructuralCandidate::DotSegmentRoot { atom } => {
                self.structural.prospective_dot_segment_layout_context(atom)
            }
            StructuralCandidate::BranchChild { .. }
            | StructuralCandidate::InlineChild { .. } => prepared
                .expect("child entry must be prepared before ring resource admission")
                .layout_context(self.surface.molecule().graph()),
            StructuralCandidate::RingOpen { .. }
            | StructuralCandidate::RingClose { .. }
            | StructuralCandidate::CloseBranch
            | StructuralCandidate::FinishSegment => unreachable!("non-entry candidate returned before context construction"),
        };
        assert_eq!(context.order.atom, target);
        if let Some((factor, mask_variable, bits)) = &self.surface.ordered_ring_factors[target.index()] {
            let model_factor = self.surface.molecule().constraint_model().factor(*factor)
                .expect("prepared ordered ring resource factor must exist");
            let FactorDefinition::OrderedRingPhase(ordered) = model_factor else {
                unreachable!("tetrahedral ring resource must use the ordered relation")
            };
            let waiting = context.waiting_ring_bonds.iter().copied().collect::<BTreeSet<_>>();
            let mask = ordered.bonds().iter().enumerate().fold(0_u8, |mask, (position, (bond, _))| {
                mask | (u8::from(waiting.contains(bond)) << position)
            });
            let mut restrictions = vec![(*mask_variable, Domain::singleton(mask).unwrap())];
            restrictions.extend(bits.iter().enumerate().map(|(bit, &variable)| {
                (variable, Domain::singleton(((free >> bit) & 1) as u8).unwrap())
            }));
            let center = self.surface.tetrahedral_center(target)
                .expect("ordered ring resource must belong to a prepared tetrahedral center");
            return Some((restrictions, vec![*factor, center.layout_factor(context.order.entry_bond)]));
        }
        let (factor, bits) = self.surface.ring_count_factors[target.index()].as_ref()?;
        let bound = free
            .saturating_add(2 * context.waiting_ring_bonds.len());
        if bound >= degree { return None; }
        let budget = bits.iter().enumerate().map(|(bit, &variable)| {
            (variable, Domain::singleton(((bound >> bit) & 1) as u8).unwrap())
        }).collect();
        Some((budget, vec![*factor]))
    }

    fn restrict_semantics(
        &self,
        restrictions: &[(VariableId, Domain)],
        activate: &[FactorId],
    ) -> Result<Consistency<WriterState<S>>, S::Failure> {
        if restrictions.is_empty() && activate.is_empty() {
            Ok(Consistency::Consistent(self.structural.clone()))
        } else {
            self.structural
                .transitioned_semantics(restrictions, activate)
        }
    }

    fn validate_active_tetrahedral_completion(
        &self,
        structural: &WriterState<S>,
    ) -> Result<(), WriterInvariantFailure> {
        let local = structural.active_local_bond_order();
        self.validate_tetrahedral_local_completion(structural, &local)
    }

    fn validate_tetrahedral_local_completion(
        &self,
        structural: &WriterState<S>,
        local: &crate::traversal::LocalBondOrder,
    ) -> Result<(), WriterInvariantFailure> {
        let Some(center) = self.surface.tetrahedral_center(local.atom) else {
            return Ok(());
        };
        let expected = center.completed_order_domain(local.entry_bond, &local.emitted_bonds);
        let represented_bond_count =
            local.emitted_bonds.len() + usize::from(local.entry_bond.is_some());
        let mut pattern = 0_u8;
        for bond in local
            .emitted_bonds
            .iter()
            .copied()
            .take(local.ring_occurrence_count)
        {
            pattern |= 1_u8 << center.pattern_bit(bond);
        }
        let expected_pattern = Domain::singleton(pattern).unwrap();
        let layout_factor = center.layout_factor(local.entry_bond);
        let roles_match = center.bond_pattern_bits.iter().all(|(bond, bit)| {
            let partition = self
                .surface
                .molecule()
                .bond_role_partition(*bond)
                .expect("prepared tetrahedral bond must retain its role partition");
            let expected_role = if pattern & (1_u8 << bit) == 0 {
                partition.traversal_values()
            } else {
                partition.ring_values()
            };
            structural
                .bond_decision_domain(*bond)
                .is_subset_of(expected_role)
        });
        if represented_bond_count != center.bond_pattern_bits.len()
            || expected.is_empty()
            || structural.semantic_domain(center.order_variable) != expected
            || structural.semantic_domain(center.role_pattern_variable) != expected_pattern
            || !structural.factor_is_active(layout_factor)
            || !roles_match
        {
            return Err(WriterInvariantFailure::UnresolvedTetrahedralFrame { atom: local.atom });
        }
        Ok(())
    }

    fn validate_active_semantic_completion(
        &self,
        structural: &WriterState<S>,
    ) -> Result<(), WriterInvariantFailure> {
        self.validate_active_tetrahedral_completion(structural)?;
        if !self.surface.directional_sides.is_empty() {
            let graph = self.surface.molecule().graph();
            let active = structural
                .active_atom()
                .expect("semantic completion requires one active atom");
            let component = graph
                .component_of_atom(active)
                .expect("active atom must belong to one prepared component");
            if structural.chemical_component_is_complete(component) {
                self.validate_directional_component_completion(structural)?;
            }
        }
        let local = structural.active_local_bond_order();
        self.validate_directional_local_completion(structural, &local)
    }

    fn validate_directional_local_completion(
        &self,
        structural: &WriterState<S>,
        local: &crate::traversal::LocalBondOrder,
    ) -> Result<(), WriterInvariantFailure> {
        for bond in local
            .entry_bond
            .into_iter()
            .chain(local.emitted_bonds.iter().copied())
        {
            if let PreparedDirectionalBondStatus::CompiledSite(site) =
                self.surface.directional_status(bond)
            {
                let role = structural.bond_decision_domain(bond);
                let mark = structural.semantic_domain(site.mark_variable);
                let plan = structural.semantic_domain(site.ring_plan_variable);
                let factor_active = structural.factor_is_active(site.ring_plan_factor);
                let valid = if role == BondRole::Traversal.singleton_domain() {
                    mark.is_singleton() && !factor_active
                } else if role == BondRole::Ring.singleton_domain() {
                    factor_active
                        && (structural.ring_is_open(bond)
                            || (plan.is_singleton()
                                && mark.is_singleton()
                                && DirectionalRingPlan::ALL
                                    .into_iter()
                                    .find(|candidate| plan.contains(candidate.value_index()))
                                    .is_some_and(|plan| mark == plan.mark().singleton_domain())))
                } else {
                    false
                };
                if !valid {
                    return Err(WriterInvariantFailure::UnresolvedDirectionalCarrier { bond });
                }
            }
        }
        Ok(())
    }

    fn validate_inline_parent_completion(
        &self,
        structural: &WriterState<S>,
        prepared: &PreparedChildEntry,
    ) -> Result<(), WriterInvariantFailure> {
        let parent = prepared.parent_order().expect("inline entry must capture its parent order");
        self.validate_tetrahedral_local_completion(structural, parent)?;
        self.validate_directional_local_completion(structural, parent)
    }

    fn validate_directional_component_completion(
        &self,
        structural: &WriterState<S>,
    ) -> Result<(), WriterInvariantFailure> {
        let active = structural
            .active_atom()
            .expect("component completion requires one active frame");
        let graph = self.surface.molecule().graph();
        let component = graph
            .component_of_atom(active)
            .expect("active atom must belong to one prepared component");
        for side in self
            .surface
            .directional_sides
            .iter()
            .filter(|side| graph.component_of_atom(side.endpoint) == Some(component))
        {
            if !structural
                .semantic_domain(side.pattern_variable)
                .is_singleton()
            {
                return Err(WriterInvariantFailure::UnresolvedDirectionalSide {
                    double_bond: side.double_bond,
                    endpoint: side.endpoint,
                });
            }
            for carrier in &side.carriers {
                let PreparedDirectionalBondStatus::CompiledSite(site) =
                    self.surface.directional_status(carrier.bond)
                else {
                    panic!("compiled directional side must retain physical mark variables")
                };
                if !structural
                    .semantic_domain(site.mark_variable)
                    .is_singleton()
                {
                    return Err(WriterInvariantFailure::UnresolvedDirectionalCarrier {
                        bond: carrier.bond,
                    });
                }
                let role = structural.bond_decision_domain(carrier.bond);
                let factor_active = structural.factor_is_active(site.ring_plan_factor);
                if role == BondRole::Traversal.singleton_domain() {
                    if factor_active {
                        return Err(WriterInvariantFailure::UnresolvedDirectionalCarrier {
                            bond: carrier.bond,
                        });
                    }
                } else if role == BondRole::Ring.singleton_domain() {
                    let plan = structural.semantic_domain(site.ring_plan_variable);
                    let mark = structural.semantic_domain(site.mark_variable);
                    let plan_mark = plan.is_singleton().then(|| {
                        let value = plan.iter().next().unwrap();
                        DirectionalRingPlan::ALL
                            .into_iter()
                            .find(|plan| plan.value_index() == value)
                            .expect("prepared directional ring plan must be known")
                            .mark()
                            .singleton_domain()
                    });
                    if !factor_active || plan_mark != Some(mark) {
                        return Err(WriterInvariantFailure::UnresolvedDirectionalCarrier {
                            bond: carrier.bond,
                        });
                    }
                } else {
                    return Err(WriterInvariantFailure::UnresolvedDirectionalCarrier {
                        bond: carrier.bond,
                    });
                }
            }
        }
        Ok(())
    }

    fn attempt_child_entry(
        &self,
        candidate: StructuralCandidate,
        incident: AdjacentBond,
    ) -> Vec<CandidateAttempt<Self, S::Failure>> {
        let prepared = self.structural.prepare_child_entry(candidate);
        let parent = self.structural.active_atom().expect("child entry requires an active parent");
        let mut common = self.parent_prefix_restriction(incident);
        let anchor = match self.directional_anchor_restrictions(
            incident.atom(), parent, Some(incident.bond()),
        ) {
            Ok(anchor) => anchor,
            Err(failure) => return vec![CandidateAttempt::Incomplete(failure)],
        };
        common.extend(anchor);
        let emissions = match self.traversal_emission_specs(incident.bond(), parent) {
            Ok(emissions) => emissions,
            Err(incomplete) => {
                let structural = match self.attempt_candidate_with_prepared_transition(
                    candidate, Some(&prepared), None, &common, &[],
                ) {
                    Ok(Consistency::Consistent(structural)) => structural,
                    Ok(Consistency::Contradiction) => return vec![CandidateAttempt::Rejected {
                        reason: CandidateRejection::Contradiction,
                    }],
                    Err(failure) => return vec![CandidateAttempt::Failed(failure)],
                };
                if matches!(candidate, StructuralCandidate::InlineChild { .. }) {
                    if let Err(failure) = self.validate_inline_parent_completion(&structural, &prepared) {
                        return vec![CandidateAttempt::Invariant(failure)];
                    }
                }
                return vec![CandidateAttempt::Incomplete(Self::incomplete_selection(incomplete))];
            }
        };
        let context = prepared.layout_context(self.surface.molecule().graph());
        let spellings = self.atom_token_specs(incident.atom(), Some(incident.bond()), &context);
        let mut attempts = Vec::new();
        for emission in emissions {
            for (start, atom_restrictions, activate) in &spellings {
                let mut restrictions = common.clone();
                restrictions.extend_from_slice(&emission.semantic_restrictions);
                restrictions.extend_from_slice(atom_restrictions);
                let structural = match self.attempt_candidate_with_prepared_transition(
                    candidate, Some(&prepared), None, &restrictions, activate,
                ) {
                    Ok(Consistency::Consistent(structural)) => structural,
                    Ok(Consistency::Contradiction) => {
                        attempts.push(CandidateAttempt::Rejected {
                            reason: CandidateRejection::Contradiction,
                        });
                        continue;
                    }
                    Err(failure) => {
                        attempts.push(CandidateAttempt::Failed(failure));
                        return attempts;
                    }
                };
                if matches!(candidate, StructuralCandidate::InlineChild { .. }) {
                    if let Err(failure) = self.validate_inline_parent_completion(&structural, &prepared) {
                        attempts.push(CandidateAttempt::Invariant(failure));
                        return attempts;
                    }
                }
                let AtomSpellingStart::Grammar { atom, state: grammar } = start;
                let spelling = AtomSpellingProgress { atom: *atom, state: grammar.clone() };
                let branch = matches!(candidate, StructuralCandidate::BranchChild { .. });
                let successor = Self {
                    surface: self.surface.clone(),
                    regime: self.regime,
                    structural,
                    labels: self.labels.clone(),
                    pending: if branch || !emission.text.is_empty() {
                        let entry = ChildEntrySpelling {
                            incident, bond_text: emission.text, grammar: grammar.clone(),
                        };
                        Some(if branch {
                            PendingEmission::BranchTraversalEmission(entry)
                        } else {
                            PendingEmission::InlineAtom(entry)
                        })
                    } else {
                        None
                    },
                    atom_spelling: None,
                };
                let new_attempts = if branch {
                    vec![self.finish_attempt("(".to_owned(), successor)]
                } else if emission.text.is_empty() {
                    successor.atom_spelling_attempts(spelling)
                } else {
                    vec![self.finish_attempt(emission.text.to_owned(), successor)]
                };
                for attempt in new_attempts {
                    let stop = matches!(attempt, CandidateAttempt::Incomplete(_)
                        | CandidateAttempt::Invariant(_) | CandidateAttempt::Failed(_));
                    attempts.push(attempt);
                    if stop {
                        return attempts;
                    }
                }
            }
        }
        attempts
    }

    fn attempt_structural(
        &self,
        candidate: StructuralCandidate,
    ) -> Vec<CandidateAttempt<Self, S::Failure>> {
        match candidate {
            StructuralCandidate::DotBranchRoot { atom }
            | StructuralCandidate::DotSegmentRoot { atom } => {
                let anchor = match self.directional_anchor_restrictions(atom, atom, None) {
                    Ok(anchor) => anchor,
                    Err(failure) => return vec![CandidateAttempt::Incomplete(failure)],
                };
                let structural = match self.attempt_candidate_with_transition(
                    candidate, None, anchor.as_slice(), &[],
                ) {
                    Ok(Consistency::Consistent(structural)) => structural,
                    Ok(Consistency::Contradiction) => return vec![CandidateAttempt::Rejected {
                        reason: CandidateRejection::Contradiction,
                    }],
                    Err(failure) => return vec![CandidateAttempt::Failed(failure)],
                };
                let text = match candidate {
                    StructuralCandidate::DotBranchRoot { .. } => "(.",
                    StructuralCandidate::DotSegmentRoot { .. } => ".",
                    _ => unreachable!(),
                };
                vec![self.finish_attempt(
                    text.to_owned(),
                    Self {
                        surface: self.surface.clone(), regime: self.regime, structural,
                        labels: self.labels.clone(),
                        pending: Some(PendingEmission::SegmentRootAtom(atom)),
                        atom_spelling: None,
                    },
                )]
            }
            StructuralCandidate::Root { atom } => {
                assert!(
                    self.structural.prepared_topology()
                        == crate::model::TraversalTopologyRequirement::Forest
                        || self.labels.is_clean(),
                    "a chemical component must start with clean ring-label spelling state"
                );
                let anchor = match self.directional_anchor_restrictions(atom, atom, None) {
                    Ok(anchor) => anchor,
                    Err(failure) => return vec![CandidateAttempt::Incomplete(failure)],
                };
                if self.structural.has_visited_atoms() {
                    let structural = match self.attempt_candidate_with_transition(
                        candidate,
                        None,
                        anchor.as_slice(),
                        &[],
                    ) {
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
                            regime: self.regime,
                            structural,
                            labels: self.labels.clone(),
                            pending: Some(PendingEmission::SegmentRootAtom(atom)),
                        atom_spelling: None,
},
                    )]
                } else {
                    let context = self.structural.prospective_root_layout_context(atom);
                    collect_attempts_fail_fast(
                        self.atom_token_specs(atom, None, &context).into_iter().flat_map(
                            |(spelling, mut restriction, activate)| {
                                restriction.extend_from_slice(&anchor);
                                let structural = match self.attempt_candidate_with_transition(
                                    candidate,
                                    None,
                                    restriction.as_slice(),
                                    activate.as_slice(),
                                ) {
                                    Ok(Consistency::Consistent(structural)) => structural,
                                    Ok(Consistency::Contradiction) => {
                                        return vec![CandidateAttempt::Rejected {
                                            reason: CandidateRejection::Contradiction,
                                        }];
                                    }
                                    Err(failure) => return vec![CandidateAttempt::Failed(failure)],
                                };
                                let state = Self {
                                        surface: self.surface.clone(),
                                        regime: self.regime,
                                        structural,
                                        labels: self.labels.clone(),
                                        pending: None,
                                        atom_spelling: None,
                                    };
                                match spelling {
                                    AtomSpellingStart::Grammar { atom, state: spelling } => {
                                        state.atom_spelling_attempts(AtomSpellingProgress { atom, state: spelling })
                                    }
                                }
                            },
                        ),
                    )
                }
            }
            StructuralCandidate::RingOpen { incident } => {
                panic!("ring openings expand into endpoint-spelling candidates: {incident:?}")
            }
            StructuralCandidate::RingClose { incident, .. } => {
                panic!("ring closures expand into endpoint-spelling candidates: {incident:?}")
            }
            StructuralCandidate::BranchChild { incident }
            | StructuralCandidate::InlineChild { incident } => {
                self.attempt_child_entry(candidate, incident)
            }
            StructuralCandidate::CloseBranch => {
                if let Err(failure) = self.validate_active_semantic_completion(&self.structural) {
                    return vec![CandidateAttempt::Invariant(failure)];
                }
                let structural = match self.attempt_candidate_with_transition(candidate, None, &[], &[]) {
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
                        regime: self.regime,
                        structural,
                        labels,
                        pending: None,
                    atom_spelling: None,
},
                )]
            }
            StructuralCandidate::FinishSegment => {
                panic!("top-level completion is normalized without a visible token")
            }
        }
    }

    fn pending_attempts(
        &self,
        pending: PendingEmission,
    ) -> Vec<CandidateAttempt<Self, S::Failure>> {
        match pending {
            PendingEmission::SegmentRootAtom(atom) => {
                self.pending_root_atom_attempts(atom)
            }
            PendingEmission::InlineAtom(entry) => {
                self.committed_child_atom_attempts(entry)
            }
            PendingEmission::BranchTraversalEmission(entry) => {
                assert_eq!(self.structural.active_atom(), Some(entry.incident.atom()));
                if entry.bond_text.is_empty() {
                    self.committed_child_atom_attempts(entry)
                } else {
                    let mut successor = self.clone();
                    successor.pending = Some(PendingEmission::BranchAtom(entry.clone()));
                    vec![self.finish_attempt(entry.bond_text.to_owned(), successor)]
                }
            }
            PendingEmission::BranchAtom(entry) => {
                self.committed_child_atom_attempts(entry)
            }
            PendingEmission::RingOpeningLabel {
                incident,
                label_slot,
            } => {
                assert_eq!(
                    self.labels.slot_for_bond(incident.bond()),
                    label_slot,
                    "a pending opening label must retain its assignment"
                );
                if label_slot.index() > self.labels.maximum_spelling_label() {
                    return vec![self.ring_label_exhaustion_attempt(label_slot.index())];
                }
                self.label_text_options(label_slot)
                    .into_iter()
                    .map(|text| {
                        self.finish_attempt(
                            text,
                            Self {
                                surface: self.surface.clone(),
                                regime: self.regime,
                                structural: self.structural.clone(),
                                labels: self.labels.clone(),
                                pending: None,
                                atom_spelling: None,
                            },
                        )
                    })
                    .collect()
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
                if label_slot.index() > self.labels.maximum_spelling_label() {
                    return vec![self.ring_label_exhaustion_attempt(label_slot.index())];
                }
                self.label_text_options(label_slot)
                    .into_iter()
                    .map(|text| {
                        let mut labels = self.labels.clone();
                        labels.release(label_slot, incident.bond());
                        self.finish_attempt(
                            text,
                            Self {
                                surface: self.surface.clone(),
                                regime: self.regime,
                                structural: self.structural.clone(),
                                labels,
                                pending: None,
                                atom_spelling: None,
                            },
                        )
                    })
                    .collect()
            }
        }
    }

    fn committed_child_atom_attempts(
        &self,
        entry: ChildEntrySpelling,
    ) -> Vec<CandidateAttempt<Self, S::Failure>> {
        assert_eq!(self.structural.active_atom(), Some(entry.incident.atom()));
        let mut successor = self.clone();
        successor.pending = None;
        successor.atom_spelling_attempts(AtomSpellingProgress {
            atom: entry.incident.atom(),
            state: entry.grammar,
        })
    }

    fn pending_root_atom_attempts(&self, atom: AtomId) -> Vec<CandidateAttempt<Self, S::Failure>> {
        #[cfg(test)]
        self.surface
            .work_counters
            .pending_atom_frontier_evaluations
            .fetch_add(1, Ordering::Relaxed);
        assert_eq!(self.structural.active_atom(), Some(atom));
        let context = self.structural.active_local_layout_context();
        collect_attempts_fail_fast(
            self.atom_token_specs(atom, None, &context)
                .into_iter()
                .flat_map(|(spelling, restrictions, activate)| {
                    let structural = match self.restrict_semantics(&restrictions, &activate) {
                        Ok(Consistency::Consistent(structural)) => structural,
                        Ok(Consistency::Contradiction) => {
                            return vec![CandidateAttempt::Rejected {
                                reason: CandidateRejection::Contradiction,
                            }];
                        }
                        Err(failure) => return vec![CandidateAttempt::Failed(failure)],
                    };
                    let state = Self {
                            surface: self.surface.clone(),
                            regime: self.regime,
                            structural,
                            labels: self.labels.clone(),
                            pending: None,
                        atom_spelling: None,
};
                    match spelling {
                        AtomSpellingStart::Grammar { atom, state: spelling } => {
                            state.atom_spelling_attempts(AtomSpellingProgress { atom, state: spelling })
                        }
                    }
                }),
        )
    }

    fn atom_spelling_attempts(
        &self,
        spelling: AtomSpellingProgress,
    ) -> Vec<CandidateAttempt<Self, S::Failure>> {
        let atom = spelling.atom;
        let facts = self.surface.atom_facts(atom);
        let source_context = self.surface.atom_source_context(atom);
        let tetrahedral = self.surface.tetrahedral_center(atom).is_some();
        spelling
            .state
            .choices()
            .into_iter()
            .filter(|choice| {
                self.regime != WriterRegime::RdkitRandomNoncanonical
                    || rdkit_atom_field_allowed(
                        facts,
                        source_context,
                        tetrahedral,
                        choice.field,
                        choice.text,
                    )
            })
            .map(|choice| {
                let structural = match self.restrict_semantics(&[], &[]) {
                    Ok(Consistency::Consistent(structural)) => structural,
                    Ok(Consistency::Contradiction) => return CandidateAttempt::Rejected { reason: CandidateRejection::Contradiction },
                    Err(failure) => return CandidateAttempt::Failed(failure),
                };
                let successor_spelling = (!choice.successor.is_complete()).then_some(
                    AtomSpellingProgress {
                        atom,
                        state: choice.successor,
                    },
                );
                let successor = Self {
                    surface: self.surface.clone(),
                    regime: self.regime,
                    structural,
                    labels: self.labels.clone(),
                    pending: None,
                    atom_spelling: successor_spelling,
                };
                self.finish_attempt(choice.text.to_string(), successor)
            })
            .collect()
    }

    fn finish_attempt(&self, text: String, successor: Self) -> CandidateAttempt<Self, S::Failure> {
        match successor.normalize_and_check() {
            SuccessorAttempt::Accepted(successor) => CandidateAttempt::Accepted { text, successor },
            SuccessorAttempt::Rejected(reason) => CandidateAttempt::Rejected { reason },
            SuccessorAttempt::Incomplete(failure) => CandidateAttempt::Incomplete(failure),
            SuccessorAttempt::Invariant(failure) => CandidateAttempt::Invariant(failure),
            SuccessorAttempt::Failed(failure) => CandidateAttempt::Failed(failure),
        }
    }

    fn normalize_and_check(mut self) -> SuccessorAttempt<Self, S::Failure> {
        if self.atom_spelling.is_some() {
            return SuccessorAttempt::Accepted(self);
        }
        if let Some(pending) = self.pending.clone() {
            assert!(
                self.structural.active_atom().is_some(),
                "pending text requires an active structural path"
            );
            let mut viable = false;
            let mut semantic_rejection = false;
            let mut spelling_rejection = None;
            for attempt in self.pending_attempts(pending) {
                match attempt {
                    CandidateAttempt::Accepted { .. } => {
                        #[cfg(test)]
                        self.surface
                            .work_counters
                            .discarded_prevalidated_successors
                            .fetch_add(1, Ordering::Relaxed);
                        viable = true;
                    }
                    CandidateAttempt::Rejected { reason } => match reason {
                        CandidateRejection::Contradiction => semantic_rejection = true,
                        unavailable @ CandidateRejection::RingLabelUnavailable { .. } => {
                            spelling_rejection.get_or_insert(unavailable);
                        }
                    },
                    CandidateAttempt::Incomplete(failure) => {
                        return SuccessorAttempt::Incomplete(failure);
                    }
                    CandidateAttempt::Invariant(failure) => {
                        return SuccessorAttempt::Invariant(failure);
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
            if batch.candidates() != [StructuralCandidate::FinishSegment] {
                return SuccessorAttempt::Accepted(self);
            }
            assert!(
                self.structural.prepared_topology()
                    == crate::model::TraversalTopologyRequirement::Forest
                    || !self.labels.has_open_labels(),
                "a chemical component cannot finish with open visible ring labels"
            );
            if let Err(failure) = self.validate_active_semantic_completion(&self.structural) {
                return SuccessorAttempt::Invariant(failure);
            }
            self.structural = match self.attempt_candidate_with_transition(
                StructuralCandidate::FinishSegment, None, &[], &[])
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

    #[cfg(test)]
    fn reset_writer_work_counts(&self) {
        self.surface
            .work_counters
            .pending_atom_frontier_evaluations
            .store(0, Ordering::Relaxed);
        self.surface
            .work_counters
            .discarded_prevalidated_successors
            .store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn writer_work_counts(&self) -> (usize, usize) {
        (
            self.surface
                .work_counters
                .pending_atom_frontier_evaluations
                .load(Ordering::Relaxed),
            self.surface
                .work_counters
                .discarded_prevalidated_successors
                .load(Ordering::Relaxed),
        )
    }
}

fn rdkit_atom_field_allowed(
    facts: &AtomFacts,
    source_context: AtomSourceContext,
    tetrahedral: bool,
    field: AtomSpellingField,
    text: char,
) -> bool {
    if tetrahedral {
        match field {
            AtomSpellingField::OpenBracket
            | AtomSpellingField::Element
            | AtomSpellingField::CloseBracket => true,
            AtomSpellingField::TetrahedralMarker => matches!(text, '@'),
            AtomSpellingField::HydrogenCount => {
                facts.non_graph_hydrogens != 1 || text == 'H'
            }
            _ => false,
        }
    } else {
        field == AtomSpellingField::BareElement
            && matches!(facts.element.as_str(), "C" | "O")
            && !facts.aromatic
            && facts.isotope.is_none()
            && facts.formal_charge == 0
            && facts.atom_class == 0
            && !source_context.atom_class_property_present
    }
}

fn ring_label_number_text(label: usize) -> String {
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

fn ring_label_spellings(label: usize) -> Vec<String> {
    let canonical = ring_label_number_text(label);
    if label < 10 {
        vec![canonical, format!("%{label:02}")]
    } else {
        vec![canonical]
    }
}

#[cfg(test)]
#[path = "nonstereo_writer_tests.rs"]
mod tests;
#[cfg(test)]
#[path = "live_obligation_reference_tests.rs"]
mod live_obligation_reference_tests;

#[cfg(test)]
#[path = "forest_realization_contract_tests.rs"]
mod forest_realization_contract_tests;

#[cfg(test)]
#[path = "generic_ring_label_tests.rs"]
mod generic_ring_label_tests;

#[cfg(test)]
#[path = "nonstereo_transition_oracle.rs"]
mod transition_oracle;

#[cfg(test)]
#[path = "tetrahedral_transition_oracle.rs"]
mod tetrahedral_transition_oracle;

#[cfg(test)]
#[path = "directional_transition_oracle.rs"]
mod directional_transition_oracle;

#[cfg(test)]
#[path = "kekule_writer_tests.rs"]
mod kekule_writer_tests;
