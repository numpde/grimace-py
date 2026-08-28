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
use crate::ids::{AtomId, BondId, FactorId, VariableId};
use crate::model::{EdgeRolePartition, TetrahedralLayoutBond};
use crate::prepared::{AdjacentBond, PreparedBond, PreparedConstraintAssembly, PreparedMolecule};
use crate::solver::{Consistency, ConstraintSolver};
use crate::tetrahedral::{
    full_order_domain, full_role_pattern_domain, layout_order_rows, parity_domain, prefix_domain,
    singleton_order, TetrahedralLigand, TetrahedralParity,
};
use crate::traversal::LocalLayoutContext;
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

    const fn is_directional_carrier_base(self) -> bool {
        matches!(self, Self::Elided | Self::Aromatic | Self::Single)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PreparedDirectionalCarrier {
    pub(crate) bond: BondId,
    pub(crate) side_flip: bool,
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
pub(crate) struct PreparedDirectionalRelation {
    pub(crate) double_bond: BondId,
    pub(crate) left_endpoint: AtomId,
    pub(crate) left_carriers: Box<[PreparedDirectionalCarrier]>,
    pub(crate) right_endpoint: AtomId,
    pub(crate) right_carriers: Box<[PreparedDirectionalCarrier]>,
    pub(crate) side_phase_xor: bool,
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

#[derive(Copy, Clone, Debug)]
struct TraversalEmissionSpec {
    text: &'static str,
    mark_restriction: Option<(VariableId, Domain)>,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedNonStereo {
    molecule: PreparedMolecule,
    atoms: Arc<[PreparedAtom]>,
    bond_tokens: Arc<[NonStereoBondToken]>,
    directional_bonds: Arc<[PreparedDirectionalBondStatus]>,
    directional_sides: Arc<[PreparedDirectionalSide]>,
    #[cfg(test)]
    work_counters: Arc<WriterWorkCounters>,
}

#[cfg(test)]
#[derive(Debug, Default)]
struct WriterWorkCounters {
    pending_atom_frontier_evaluations: AtomicUsize,
    discarded_prevalidated_successors: AtomicUsize,
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
    role_pattern_variable: VariableId,
    bond_pattern_bits: Box<[(BondId, u8)]>,
    root_layout_factor: FactorId,
    entry_layout_factors: Box<[(BondId, FactorId)]>,
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
        emitted_bonds: &[BondId],
    ) -> Domain {
        let mut prefix = self.context_prefix(entry_bond);
        prefix.extend(emitted_bonds.iter().copied().map(TetrahedralLigand::Bond));
        prefix_domain(&self.reference_order, &prefix)
    }

    fn completed_order_domain(
        &self,
        entry_bond: Option<BondId>,
        emitted_bonds: &[BondId],
    ) -> Domain {
        let mut order = self.context_prefix(entry_bond);
        order.extend(emitted_bonds.iter().copied().map(TetrahedralLigand::Bond));
        singleton_order(&self.reference_order, &order)
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
                } => PreparedAtom::Tetrahedral(prepare_tetrahedral_center(
                    &mut assembly,
                    reference_order,
                    text_by_parity,
                )),
            })
            .collect::<Vec<_>>();
        let (directional_bonds, directional_sides) =
            prepare_directional_bonds(&mut assembly, graph, &bond_tokens, &directional_relations)?;
        let molecule = assembly.finish();

        Ok(Self {
            molecule,
            atoms: Arc::from(atoms.into_boxed_slice()),
            bond_tokens: Arc::from(bond_tokens.into_boxed_slice()),
            directional_bonds: Arc::from(directional_bonds.into_boxed_slice()),
            directional_sides: Arc::from(directional_sides.into_boxed_slice()),
            #[cfg(test)]
            work_counters: Arc::new(WriterWorkCounters::default()),
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

fn prepare_tetrahedral_center(
    assembly: &mut PreparedConstraintAssembly,
    reference_order: [TetrahedralLigand; 4],
    text_by_parity: [String; 2],
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
    let add_factor = |assembly: &mut PreparedConstraintAssembly,
                      context_prefix: Vec<TetrahedralLigand>| {
        assembly.add_latent_tetrahedral_layout(
            order_variable,
            role_pattern_variable,
            layout_bonds.iter().copied(),
            layout_order_rows(&reference_order, &context_prefix, &bond_pattern_bits),
        )
    };
    let virtual_hydrogen = reference_order.contains(&TetrahedralLigand::VirtualHydrogen);
    let root_prefix = virtual_hydrogen
        .then_some(TetrahedralLigand::VirtualHydrogen)
        .into_iter()
        .collect();
    let root_layout_factor = add_factor(assembly, root_prefix);
    let entry_layout_factors = bond_pattern_bits
        .iter()
        .map(|(bond, _)| {
            let mut prefix = vec![TetrahedralLigand::Bond(*bond)];
            if virtual_hydrogen {
                prefix.push(TetrahedralLigand::VirtualHydrogen);
            }
            (*bond, add_factor(assembly, prefix))
        })
        .collect::<Vec<_>>();

    PreparedTetrahedralCenter {
        reference_order,
        text_by_parity: text_by_parity.map(String::into_boxed_str),
        order_variable,
        role_pattern_variable,
        bond_pattern_bits: bond_pattern_bits.into_boxed_slice(),
        root_layout_factor,
        entry_layout_factors: entry_layout_factors.into_boxed_slice(),
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
            let variable = variables[carrier];
            statuses[carrier.index()] =
                PreparedDirectionalBondStatus::CompiledSite(PreparedDirectionalSite {
                    mark_variable: variable,
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
    BranchTraversalEmission(AdjacentBond),
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
    pub(crate) tetrahedral_role_pattern_domains: Vec<(AtomId, Domain)>,
    pub(crate) directional_mark_domains: Vec<(BondId, Domain)>,
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
    DirectionalCarrierSelection {
        double_bond: BondId,
        endpoint: AtomId,
    },
    DirectionalRingEndpoint {
        carrier_bond: BondId,
    },
}

impl fmt::Display for WriterIncompleteness {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DirectionalCarrierSelection {
                double_bond,
                endpoint,
            } => write!(
                formatter,
                "directional carrier selection for double bond {double_bond:?} at {endpoint:?} is not implemented"
            ),
            Self::DirectionalRingEndpoint { carrier_bond } => write!(
                formatter,
                "directional ring-endpoint spelling for carrier {carrier_bond:?} is not implemented"
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
            PendingEmission::BranchTraversalEmission(incident) => {
                ObservedPending::BranchTraversalEmission {
                    parent: active_endpoint(),
                    child: incident.atom(),
                    bond: incident.bond(),
                }
            }
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
        let current = self.structural.bond_decision_domain(incident.bond());
        let mut restrictions = self.parent_prefix_restriction(incident);
        restrictions.extend_from_slice(semantic_restrictions);
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
                    SuccessorAttempt::Invariant(failure) => CandidateAttempt::Invariant(failure),
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
                CandidateAttempt::Invariant(_) => {
                    attempts.push(attempt);
                    break;
                }
                attempt => attempts.push(attempt),
            }
        }
        Ok(attempts)
    }

    fn attempt_incomplete_ring_candidate(
        &self,
        candidate: StructuralCandidate,
        incident: AdjacentBond,
        incomplete: WriterIncompleteness,
    ) -> Vec<CandidateAttempt<Self, S::Failure>> {
        let active = self
            .structural
            .active_atom()
            .expect("ring spelling requires an active endpoint");
        let current = self.structural.bond_decision_domain(incident.bond());
        let order_restriction = self.parent_prefix_restriction(incident);
        let mut attempted = 0;
        let mut viable = false;
        for spelling in [RingEndpointSpelling::Omit, RingEndpointSpelling::Emit] {
            let allowed = current.intersect(self.surface.ring_endpoint_domain(
                incident.bond(),
                active,
                spelling,
            ));
            if allowed.is_empty() {
                continue;
            }
            attempted += 1;
            match self.attempt_candidate_with_transition(
                candidate,
                Some(allowed),
                &order_restriction,
                &[],
            ) {
                Ok(Consistency::Consistent(_)) => viable = true,
                Ok(Consistency::Contradiction) => {}
                Err(failure) => return vec![CandidateAttempt::Failed(failure)],
            }
        }
        assert!(
            attempted > 0,
            "a directional ring candidate must retain one prepared endpoint placement"
        );
        if viable {
            vec![CandidateAttempt::Incomplete(incomplete)]
        } else {
            vec![CandidateAttempt::Rejected {
                reason: CandidateRejection::Contradiction,
            }]
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
        let mut attempts = Vec::new();
        let mut directional_viable = false;
        for value in self.structural.semantic_domain(site.mark_variable).iter() {
            let mark = CarrierMark::from_value(value)
                .expect("prepared directional mark domain contains only mark values");
            let mark_restriction = (site.mark_variable, mark.singleton_domain());
            if mark == CarrierMark::Plain {
                let plain = match candidate {
                    StructuralCandidate::RingOpen { .. } => {
                        self.attempt_ring_openings(candidate, incident, &[mark_restriction])
                    }
                    StructuralCandidate::RingClose { .. } => {
                        self.attempt_ring_closures(candidate, incident, &[mark_restriction])
                    }
                    _ => unreachable!(),
                };
                match plain {
                    Ok(plain) => attempts.extend(plain),
                    Err(failure) => {
                        attempts.push(CandidateAttempt::Failed(failure));
                        return attempts;
                    }
                }
                continue;
            }

            let active = self
                .structural
                .active_atom()
                .expect("ring spelling requires an active endpoint");
            let current = self.structural.bond_decision_domain(incident.bond());
            let mut restrictions = self.parent_prefix_restriction(incident);
            restrictions.push(mark_restriction);
            for spelling in [RingEndpointSpelling::Omit, RingEndpointSpelling::Emit] {
                let allowed = current.intersect(self.surface.ring_endpoint_domain(
                    incident.bond(),
                    active,
                    spelling,
                ));
                if allowed.is_empty() {
                    continue;
                }
                match self.attempt_candidate_with_transition(
                    candidate,
                    Some(allowed),
                    &restrictions,
                    &[],
                ) {
                    Ok(Consistency::Consistent(_)) => directional_viable = true,
                    Ok(Consistency::Contradiction) => {}
                    Err(failure) => {
                        attempts.push(CandidateAttempt::Failed(failure));
                        return attempts;
                    }
                }
            }
        }
        if directional_viable {
            attempts.push(CandidateAttempt::Incomplete(
                WriterIncompleteness::DirectionalRingEndpoint {
                    carrier_bond: incident.bond(),
                },
            ));
        }
        if attempts.is_empty() {
            attempts.push(CandidateAttempt::Rejected {
                reason: CandidateRejection::Contradiction,
            });
        }
        attempts
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
        let current = self.structural.bond_decision_domain(incident.bond());
        let mut restrictions = self.parent_prefix_restriction(incident);
        restrictions.extend_from_slice(semantic_restrictions);
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
                        SuccessorAttempt::Invariant(failure) => {
                            CandidateAttempt::Invariant(failure)
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
    ) -> Vec<(String, Vec<(VariableId, Domain)>, Vec<FactorId>)> {
        let Some(center) = self.surface.tetrahedral_center(atom) else {
            return vec![(
                self.surface.atom_text(atom).to_owned(),
                Vec::new(),
                Vec::new(),
            )];
        };
        assert_eq!(context.order.atom, atom);
        assert_eq!(context.order.entry_bond, entry_bond);
        assert!(
            context.order.emitted_bonds.is_empty(),
            "tetrahedral layout activates before local ring or child occurrences"
        );
        let role_patterns = self.local_role_pattern_domain(center, context);
        TetrahedralParity::ALL
            .into_iter()
            .map(|parity| {
                (
                    center.text_by_parity[parity.index()].to_string(),
                    vec![
                        (
                            center.order_variable,
                            center.token_domain(entry_bond, parity),
                        ),
                        (center.role_pattern_variable, role_patterns),
                    ],
                    vec![center.layout_factor(entry_bond)],
                )
            })
            .collect()
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
            context.residual_attachment_bonds.iter().all(|attachment| {
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
        let Some(center) = self.surface.tetrahedral_center(local.atom) else {
            return Vec::new();
        };
        let mut emitted_bonds = local.emitted_bonds;
        emitted_bonds.push(incident.bond());
        vec![(
            center.order_variable,
            center.prefix_domain_with_bond_order(local.entry_bond, &emitted_bonds),
        )]
    }

    fn traversal_emission_specs(
        &self,
        bond: BondId,
        from: AtomId,
    ) -> Result<Vec<TraversalEmissionSpec>, IncompleteDirectionalSelection> {
        match self.surface.directional_status(bond) {
            PreparedDirectionalBondStatus::Ordinary => Ok(vec![TraversalEmissionSpec {
                text: self.surface.bond_text(bond, from),
                mark_restriction: None,
            }]),
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
                            mark_restriction: Some((site.mark_variable, mark.singleton_domain())),
                        }
                    })
                    .collect())
            }
        }
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
        if allowed_representations.is_none() && restrictions.is_empty() && activate.is_empty() {
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
                restrictions,
                activate,
            )
        }
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
        let local = structural.active_local_bond_order();
        for bond in local
            .entry_bond
            .into_iter()
            .chain(local.emitted_bonds.iter().copied())
        {
            if let PreparedDirectionalBondStatus::CompiledSite(site) =
                self.surface.directional_status(bond)
            {
                if !structural
                    .semantic_domain(site.mark_variable)
                    .is_singleton()
                {
                    return Err(WriterInvariantFailure::UnresolvedDirectionalCarrier { bond });
                }
            }
        }
        Ok(())
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
            }
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
                    let context = self.structural.prospective_root_layout_context(atom);
                    collect_attempts_fail_fast(
                        self.atom_token_specs(atom, None, &context).into_iter().map(
                            |(text, restriction, activate)| {
                                let structural = match self.attempt_candidate_with_transition(
                                    candidate,
                                    None,
                                    restriction.as_slice(),
                                    activate.as_slice(),
                                ) {
                                    Ok(Consistency::Consistent(structural)) => structural,
                                    Ok(Consistency::Contradiction) => {
                                        return CandidateAttempt::Rejected {
                                            reason: CandidateRejection::Contradiction,
                                        };
                                    }
                                    Err(failure) => return CandidateAttempt::Failed(failure),
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
            StructuralCandidate::BranchChild { incident } => {
                let parent = self
                    .structural
                    .active_atom()
                    .expect("branch emission requires an active atom");
                let incomplete = self.traversal_emission_specs(incident.bond(), parent).err();
                let restrictions = self.parent_prefix_restriction(incident);
                let structural = match self.attempt_candidate_with_transition(
                    candidate,
                    None,
                    restrictions.as_slice(),
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
                if let Some(incomplete) = incomplete {
                    return vec![CandidateAttempt::Incomplete(Self::incomplete_selection(
                        incomplete,
                    ))];
                }
                let labels = self.labels.clone();
                vec![self.finish_attempt(
                    "(".to_owned(),
                    Self {
                        surface: self.surface.clone(),
                        structural,
                        labels,
                        pending: Some(PendingEmission::BranchTraversalEmission(incident)),
                    },
                )]
            }
            StructuralCandidate::InlineChild { incident } => {
                let parent = self
                    .structural
                    .active_atom()
                    .expect("inline emission requires an active atom");
                let parent_restriction = self.parent_prefix_restriction(incident);
                let emissions = match self.traversal_emission_specs(incident.bond(), parent) {
                    Ok(emissions) => emissions,
                    Err(incomplete) => {
                        let structural = match self.attempt_candidate_with_transition(
                            candidate,
                            None,
                            parent_restriction.as_slice(),
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
                        if let Err(failure) = self.validate_active_semantic_completion(&structural)
                        {
                            return vec![CandidateAttempt::Invariant(failure)];
                        }
                        return vec![CandidateAttempt::Incomplete(Self::incomplete_selection(
                            incomplete,
                        ))];
                    }
                };
                let mut attempts = Vec::new();
                for emission in emissions {
                    if emission.text.is_empty() {
                        let context = self
                            .structural
                            .prospective_inline_child_layout_context(incident);
                        for (text, mut child_restriction, activate) in
                            self.atom_token_specs(incident.atom(), Some(incident.bond()), &context)
                        {
                            let mut restrictions = parent_restriction.clone();
                            if let Some(mark) = emission.mark_restriction {
                                restrictions.push(mark);
                            }
                            restrictions.append(&mut child_restriction);
                            let structural = match self.attempt_candidate_with_transition(
                                candidate,
                                None,
                                restrictions.as_slice(),
                                activate.as_slice(),
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
                            if let Err(failure) =
                                self.validate_active_semantic_completion(&structural)
                            {
                                attempts.push(CandidateAttempt::Invariant(failure));
                                return attempts;
                            }
                            let attempt = self.finish_attempt(
                                text,
                                Self {
                                    surface: self.surface.clone(),
                                    structural: structural.enter_committed_inline_child(incident),
                                    labels: self.labels.clone(),
                                    pending: None,
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
                                return attempts;
                            }
                        }
                    } else {
                        let mut restrictions = parent_restriction.clone();
                        if let Some(mark) = emission.mark_restriction {
                            restrictions.push(mark);
                        }
                        let structural = match self.attempt_candidate_with_transition(
                            candidate,
                            None,
                            restrictions.as_slice(),
                            &[],
                        ) {
                            Ok(Consistency::Consistent(structural)) => structural,
                            Ok(Consistency::Contradiction) => {
                                return vec![CandidateAttempt::Rejected {
                                    reason: CandidateRejection::Contradiction,
                                }];
                            }
                            Err(failure) => {
                                attempts.push(CandidateAttempt::Failed(failure));
                                return attempts;
                            }
                        };
                        if let Err(failure) = self.validate_active_semantic_completion(&structural)
                        {
                            attempts.push(CandidateAttempt::Invariant(failure));
                            return attempts;
                        }
                        let attempt = self.finish_attempt(
                            emission.text.to_owned(),
                            Self {
                                surface: self.surface.clone(),
                                structural,
                                labels: self.labels.clone(),
                                pending: Some(PendingEmission::InlineAtom(incident)),
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
                            return attempts;
                        }
                    }
                }
                attempts
            }
            StructuralCandidate::CloseBranch => {
                if let Err(failure) = self.validate_active_semantic_completion(&self.structural) {
                    return vec![CandidateAttempt::Invariant(failure)];
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
                self.pending_atom_attempts(atom, None, PendingAtomEntry::AlreadyEntered, &[])
            }
            PendingEmission::InlineAtom(incident) => self.pending_atom_attempts(
                incident.atom(),
                Some(incident.bond()),
                PendingAtomEntry::Inline(incident),
                &[],
            ),
            PendingEmission::BranchTraversalEmission(incident) => {
                let parent = self
                    .structural
                    .active_atom()
                    .expect("a committed branch child requires its active parent");
                let emissions = self
                    .traversal_emission_specs(incident.bond(), parent)
                    .expect("a published branch must have a complete traversal-emission frontier");
                let mut attempts = Vec::new();
                for emission in emissions {
                    let restrictions = emission.mark_restriction.into_iter().collect::<Vec<_>>();
                    if emission.text.is_empty() {
                        let pending = self.pending_atom_attempts(
                            incident.atom(),
                            Some(incident.bond()),
                            PendingAtomEntry::Branch(incident),
                            &restrictions,
                        );
                        let stop = pending.iter().any(|attempt| {
                            matches!(
                                attempt,
                                CandidateAttempt::Incomplete(_)
                                    | CandidateAttempt::Invariant(_)
                                    | CandidateAttempt::Failed(_)
                            )
                        });
                        attempts.extend(pending);
                        if stop {
                            return attempts;
                        }
                    } else {
                        let structural = match self.restrict_semantics(&restrictions, &[]) {
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
                        let attempt = self.finish_attempt(
                            emission.text.to_owned(),
                            Self {
                                surface: self.surface.clone(),
                                structural,
                                labels: self.labels.clone(),
                                pending: Some(PendingEmission::BranchAtom(incident)),
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
                            return attempts;
                        }
                    }
                }
                attempts
            }
            PendingEmission::BranchAtom(incident) => self.pending_atom_attempts(
                incident.atom(),
                Some(incident.bond()),
                PendingAtomEntry::Branch(incident),
                &[],
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
        semantic_restrictions: &[(VariableId, Domain)],
    ) -> Vec<CandidateAttempt<Self, S::Failure>> {
        #[cfg(test)]
        self.surface
            .work_counters
            .pending_atom_frontier_evaluations
            .fetch_add(1, Ordering::Relaxed);
        let context = match entry {
            PendingAtomEntry::AlreadyEntered => self.structural.active_local_layout_context(),
            PendingAtomEntry::Inline(incident) => self
                .structural
                .prospective_committed_inline_child_layout_context(incident),
            PendingAtomEntry::Branch(incident) => self
                .structural
                .prospective_committed_branch_child_layout_context(incident),
        };
        collect_attempts_fail_fast(
            self.atom_token_specs(atom, entry_bond, &context)
                .into_iter()
                .map(|(text, mut restrictions, activate)| {
                    restrictions.extend_from_slice(semantic_restrictions);
                    let structural = match self.restrict_semantics(&restrictions, &activate) {
                        Ok(Consistency::Consistent(structural)) => structural,
                        Ok(Consistency::Contradiction) => {
                            return CandidateAttempt::Rejected {
                                reason: CandidateRejection::Contradiction,
                            };
                        }
                        Err(failure) => return CandidateAttempt::Failed(failure),
                    };
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
                }),
        )
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
            SuccessorAttempt::Invariant(failure) => CandidateAttempt::Invariant(failure),
            SuccessorAttempt::Failed(failure) => CandidateAttempt::Failed(failure),
        }
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
            if batch.candidates() != [StructuralCandidate::FinishComponent] {
                return SuccessorAttempt::Accepted(self);
            }
            assert!(
                !self.labels.has_open_labels(),
                "a component cannot finish with open visible ring labels"
            );
            if let Err(failure) = self.validate_active_semantic_completion(&self.structural) {
                return SuccessorAttempt::Invariant(failure);
            }
            if let Err(failure) = self.validate_directional_component_completion(&self.structural) {
                return SuccessorAttempt::Invariant(failure);
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

#[cfg(test)]
#[path = "tetrahedral_transition_oracle.rs"]
mod tetrahedral_transition_oracle;

#[cfg(test)]
#[path = "directional_transition_oracle.rs"]
mod directional_transition_oracle;
