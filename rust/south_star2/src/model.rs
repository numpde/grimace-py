//! Solver-neutral constraint definitions prepared once per molecule.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use crate::domain::Domain;
use crate::event_schedule::EventScheduleFactor;
use crate::ids::{AtomId, BondId, FactorId, VariableId};
use crate::tetrahedral::{order_for_value, TetrahedralLigand};

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BondRole {
    Traversal = 0,
    Ring = 1,
}

impl BondRole {
    pub const fn value_index(self) -> u8 {
        self as u8
    }

    pub const fn singleton_domain(self) -> Domain {
        Domain::from_bits(1_u64 << self.value_index())
    }

    pub const fn role_domain() -> Domain {
        Domain::from_bits(
            (1_u64 << Self::Traversal.value_index()) | (1_u64 << Self::Ring.value_index()),
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EdgeRolePartition {
    traversal_values: Domain,
    ring_values: Domain,
}

impl EdgeRolePartition {
    pub const fn new(traversal_values: Domain, ring_values: Domain) -> Self {
        Self {
            traversal_values,
            ring_values,
        }
    }

    pub const fn traversal_values(self) -> Domain {
        self.traversal_values
    }

    pub const fn ring_values(self) -> Domain {
        self.ring_values
    }

    pub const fn bond_role() -> Self {
        Self::new(
            BondRole::Traversal.singleton_domain(),
            BondRole::Ring.singleton_domain(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VariableDefinition {
    initial_domain: Domain,
}

impl VariableDefinition {
    pub const fn initial_domain(&self) -> Domain {
        self.initial_domain
    }
}

/// A compiled binary finite-domain relation.
///
/// Support masks are stored in both directions so a solver can revise either
/// domain without enumerating the relation's Cartesian product.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryRelationFactor {
    variables: [VariableId; 2],
    allowed_right_by_left: Box<[Domain]>,
    allowed_left_by_right: Box<[Domain]>,
}

impl BinaryRelationFactor {
    pub const fn left(&self) -> VariableId {
        self.variables[0]
    }

    pub const fn right(&self) -> VariableId {
        self.variables[1]
    }

    pub const fn variables(&self) -> &[VariableId; 2] {
        &self.variables
    }

    pub fn allowed_right(&self, left_value: u8) -> Domain {
        self.allowed_right_by_left
            .get(left_value as usize)
            .copied()
            .unwrap_or_else(Domain::empty)
    }

    pub fn allowed_left(&self, right_value: u8) -> Domain {
        self.allowed_left_by_right
            .get(right_value as usize)
            .copied()
            .unwrap_or_else(Domain::empty)
    }
}

/// A latent local relation between one physical directional mark and its ring plan.
///
/// This is deliberately distinct from an always-active binary relation: its plan has
/// no semantic meaning until the writer commits the physical carrier to `Ring`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectionalRingPlacementFactor {
    variables: [VariableId; 2],
    allowed_plans_by_mark: Box<[Domain]>,
    allowed_marks_by_plan: Box<[Domain]>,
}

impl DirectionalRingPlacementFactor {
    pub const fn mark_variable(&self) -> VariableId {
        self.variables[0]
    }

    pub const fn plan_variable(&self) -> VariableId {
        self.variables[1]
    }

    pub const fn variables(&self) -> &[VariableId; 2] {
        &self.variables
    }

    pub fn allowed_plans(&self, mark: u8) -> Domain {
        self.allowed_plans_by_mark
            .get(mark as usize)
            .copied()
            .unwrap_or_else(Domain::empty)
    }

    pub fn allowed_marks(&self, plan: u8) -> Domain {
        self.allowed_marks_by_plan
            .get(plan as usize)
            .copied()
            .unwrap_or_else(Domain::empty)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SpanningTreeEdge {
    decision_variable: VariableId,
    role_partition: EdgeRolePartition,
    a: AtomId,
    b: AtomId,
}

impl SpanningTreeEdge {
    pub const fn new(decision_variable: VariableId, a: AtomId, b: AtomId) -> Self {
        Self::with_role_partition(decision_variable, a, b, EdgeRolePartition::bond_role())
    }

    pub const fn with_role_partition(
        decision_variable: VariableId,
        a: AtomId,
        b: AtomId,
        role_partition: EdgeRolePartition,
    ) -> Self {
        Self {
            decision_variable,
            role_partition,
            a,
            b,
        }
    }

    pub const fn decision_variable(self) -> VariableId {
        self.decision_variable
    }

    pub const fn role_partition(self) -> EdgeRolePartition {
        self.role_partition
    }

    pub const fn a(self) -> AtomId {
        self.a
    }

    pub const fn b(self) -> AtomId {
        self.b
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TraversalTopologyRequirement {
    Tree,
    Forest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraversalTopologyFactor {
    requirement: TraversalTopologyRequirement,
    atoms: Box<[AtomId]>,
    edges: Box<[SpanningTreeEdge]>,
    root_variables: Box<[VariableId]>,
    variables: Box<[VariableId]>,
}

impl TraversalTopologyFactor {
    pub fn requirement(&self) -> TraversalTopologyRequirement {
        self.requirement
    }

    pub fn atoms(&self) -> &[AtomId] {
        &self.atoms
    }

    pub fn edges(&self) -> &[SpanningTreeEdge] {
        &self.edges
    }

    pub fn root_variables(&self) -> &[VariableId] {
        &self.root_variables
    }

    pub fn variables(&self) -> &[VariableId] {
        &self.variables
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TetrahedralLayoutBond {
    decision_variable: VariableId,
    role_partition: EdgeRolePartition,
    pattern_bit: u8,
}

impl TetrahedralLayoutBond {
    pub const fn new(
        decision_variable: VariableId,
        role_partition: EdgeRolePartition,
        pattern_bit: u8,
    ) -> Self {
        Self {
            decision_variable,
            role_partition,
            pattern_bit,
        }
    }

    pub const fn decision_variable(self) -> VariableId {
        self.decision_variable
    }

    pub const fn role_partition(self) -> EdgeRolePartition {
        self.role_partition
    }

    pub const fn pattern_bit(self) -> u8 {
        self.pattern_bit
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TetrahedralLayoutFactor {
    order_variable: VariableId,
    role_pattern_variable: VariableId,
    bonds: Box<[TetrahedralLayoutBond]>,
    allowed_orders_by_pattern: Arc<[Domain]>,
    variables: Box<[VariableId]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingCountAtMostFactor {
    bonds: Box<[SpanningTreeEdge]>,
    budget_bits: Box<[VariableId]>,
    variables: Box<[VariableId]>,
}

impl RingCountAtMostFactor {
    pub fn bonds(&self) -> &[SpanningTreeEdge] { &self.bonds }
    pub fn budget_bits(&self) -> &[VariableId] { &self.budget_bits }
    pub fn variables(&self) -> &[VariableId] { &self.variables }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderedRingPhaseFactor {
    bonds: Box<[(BondId, SpanningTreeEdge)]>,
    reference_order: [TetrahedralLigand; 4],
    order_variable: VariableId,
    waiting_mask_variable: VariableId,
    free_bits: Box<[VariableId]>,
    variables: Box<[VariableId]>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ScheduleEdge {
    pub bond: BondId,
    pub a: AtomId,
    pub b: AtomId,
    pub decision_variable: VariableId,
    pub role_partition: EdgeRolePartition,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduleCenter {
    pub atom: AtomId,
    pub order_variable: VariableId,
    pub role_pattern_variable: VariableId,
    pub reference_order: [TetrahedralLigand; 4],
    pub bond_pattern_bits: Arc<[(BondId, u8)]>,
    pub root_layout_rows: Arc<[Domain]>,
    pub entry_layout_rows: Arc<[(BondId, Arc<[Domain]>)]>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OrdinaryOrderPair {
    pub first: BondId,
    pub second: BondId,
    /// 0 emits `first` before `second`; 1 reverses that order.
    pub variable: VariableId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrdinaryOrderFactor {
    atom: AtomId,
    pairs: Arc<[OrdinaryOrderPair]>,
    variables: Box<[VariableId]>,
}

impl OrdinaryOrderFactor {
    pub fn atom(&self) -> AtomId { self.atom }
    pub fn pairs(&self) -> &[OrdinaryOrderPair] { &self.pairs }
    pub fn variables(&self) -> &[VariableId] { &self.variables }

    pub fn permits(&self, mut value: impl FnMut(VariableId) -> u8) -> bool {
        let mut predecessors = std::collections::BTreeMap::<BondId, usize>::new();
        for pair in self.pairs.iter() {
            predecessors.entry(pair.first).or_default();
            predecessors.entry(pair.second).or_default();
            let later = if value(pair.variable) == 0 { pair.second } else { pair.first };
            *predecessors.get_mut(&later).expect("pair endpoint must exist") += 1;
        }
        let mut ranks = predecessors.values().copied().collect::<Vec<_>>();
        ranks.sort_unstable();
        ranks.iter().copied().eq(0..ranks.len())
    }
}

fn assert_complete_ordinary_pairs(incidence: &[BondId], pairs: &[OrdinaryOrderPair]) {
    let expected = incidence.iter().enumerate().flat_map(|(index, first)| incidence[index + 1..]
        .iter().map(move |second| (*first, *second)))
        .map(|(a, b)| if a < b { (a, b) } else { (b, a) })
        .collect::<std::collections::BTreeSet<_>>();
    let actual = pairs.iter().map(|pair| {
        assert_ne!(pair.first, pair.second, "ordinary order pair must name two incidences");
        if pair.first < pair.second { (pair.first, pair.second) }
        else { (pair.second, pair.first) }
    }).collect::<std::collections::BTreeSet<_>>();
    assert_eq!(pairs.len(), expected.len(), "ordinary order must have one fact per incidence pair");
    assert_eq!(actual, expected, "ordinary order must cover precisely the incident bonds");
    assert_eq!(pairs.iter().map(|pair| pair.variable).collect::<std::collections::BTreeSet<_>>().len(),
        pairs.len(), "ordinary order variables must be distinct");
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduleOrdinaryCenter {
    pub atom: AtomId,
    pub pairs: Arc<[OrdinaryOrderPair]>,
}

impl OrderedRingPhaseFactor {
    pub fn bonds(&self) -> &[(BondId, SpanningTreeEdge)] { &self.bonds }
    pub fn order_variable(&self) -> VariableId { self.order_variable }
    pub fn waiting_mask_variable(&self) -> VariableId { self.waiting_mask_variable }
    pub fn free_bits(&self) -> &[VariableId] { &self.free_bits }
    pub fn variables(&self) -> &[VariableId] { &self.variables }

    pub fn permits(&self, order_value: u8, waiting_mask: u8, free: usize,
                   mut ring_for: impl FnMut(usize) -> bool) -> bool {
        let Some(order) = order_for_value(&self.reference_order, order_value) else { return false; };
        let rings = (0..self.bonds.len()).map(&mut ring_for).collect::<Vec<_>>();
        if (0..self.bonds.len()).any(|position| waiting_mask & (1 << position) != 0 && !rings[position]) {
            return false;
        }
        let mut available = free;
        for ligand in order {
            let TetrahedralLigand::Bond(bond) = ligand else { continue; };
            let position = self.bonds.iter().position(|(candidate, _)| *candidate == bond)
                .expect("prepared tetrahedral order must name a factor incidence");
            if !rings[position] { continue; }
            if waiting_mask & (1 << position) != 0 { available += 1; }
            else if available == 0 { return false; }
            else { available -= 1; }
        }
        true
    }
}

impl TetrahedralLayoutFactor {
    pub const fn order_variable(&self) -> VariableId {
        self.order_variable
    }

    pub const fn role_pattern_variable(&self) -> VariableId {
        self.role_pattern_variable
    }

    pub fn bonds(&self) -> &[TetrahedralLayoutBond] {
        &self.bonds
    }

    pub fn allowed_orders(&self, pattern: u8) -> Domain {
        self.allowed_orders_by_pattern
            .get(pattern as usize)
            .copied()
            .unwrap_or_else(Domain::empty)
    }

    pub fn variables(&self) -> &[VariableId] {
        &self.variables
    }
}

/// An aromatic component edge; endpoints index the factor's double demands.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AromaticMatchingEdge {
    variable: VariableId,
    a: usize,
    b: usize,
}

impl AromaticMatchingEdge {
    pub const fn new(variable: VariableId, a: usize, b: usize) -> Self {
        Self { variable, a, b }
    }

    pub const fn variable(self) -> VariableId {
        self.variable
    }
    pub const fn endpoints(self) -> (usize, usize) {
        (self.a, self.b)
    }
}

/// Exact local support for aromatic surface spelling and integer bond orders.
/// Atom values are lowercase 0 or uppercase 1; edge values are aromatic 0,
/// explicit single 1, or explicit double 2.
/// Every scope variable is exclusive to this projector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AromaticMatchingFactor {
    atom_variables: Box<[VariableId]>,
    uniform_case: bool,
    double_demands: Box<[u8]>,
    edges: Box<[AromaticMatchingEdge]>,
    variables: Box<[VariableId]>,
}

impl AromaticMatchingFactor {
    pub fn atom_variables(&self) -> &[VariableId] {
        &self.atom_variables
    }
    pub const fn uniform_case(&self) -> bool {
        self.uniform_case
    }
    pub fn double_demands(&self) -> &[u8] {
        &self.double_demands
    }
    pub fn edges(&self) -> &[AromaticMatchingEdge] {
        &self.edges
    }
    pub fn variables(&self) -> &[VariableId] {
        &self.variables
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FactorDefinition {
    AromaticMatching(AromaticMatchingFactor),
    BinaryRelation(BinaryRelationFactor),
    DirectionalRingPlacement(DirectionalRingPlacementFactor),
    TraversalTopology(TraversalTopologyFactor),
    TetrahedralLayout(TetrahedralLayoutFactor),
    RingCountAtMost(RingCountAtMostFactor),
    OrderedRingPhase(OrderedRingPhaseFactor),
    OrdinaryOrder(OrdinaryOrderFactor),
    EventSchedule(EventScheduleFactor),
}

impl FactorDefinition {
    pub fn variables(&self) -> &[VariableId] {
        match self {
            Self::AromaticMatching(factor) => factor.variables(),
            Self::BinaryRelation(factor) => factor.variables(),
            Self::DirectionalRingPlacement(factor) => factor.variables(),
            Self::TraversalTopology(factor) => factor.variables(),
            Self::TetrahedralLayout(factor) => factor.variables(),
            Self::RingCountAtMost(factor) => factor.variables(),
            Self::OrderedRingPhase(factor) => factor.variables(),
            Self::OrdinaryOrder(factor) => factor.variables(),
            Self::EventSchedule(factor) => factor.variables(),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FactorActivation {
    Always,
    Latent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintModel {
    variables: Box<[VariableDefinition]>,
    factors: Box<[FactorDefinition]>,
    factor_activation: Box<[FactorActivation]>,
    initial_factor_ids: Box<[FactorId]>,
    initial_factors_by_variable: Box<[Box<[FactorId]>]>,
    potential_factors_by_variable: Box<[Box<[FactorId]>]>,
}

impl ConstraintModel {
    pub fn empty() -> Self {
        ConstraintModelBuilder::new().build()
    }

    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    pub fn factor_count(&self) -> usize {
        self.factors.len()
    }

    pub fn variable(&self, variable: VariableId) -> Option<&VariableDefinition> {
        self.variables.get(variable.index())
    }

    pub fn factor(&self, factor: FactorId) -> Option<&FactorDefinition> {
        self.factors.get(factor.index())
    }

    pub fn factor_activation(&self, factor: FactorId) -> Option<FactorActivation> {
        self.factor_activation.get(factor.index()).copied()
    }

    pub fn initial_factor_ids(&self) -> &[FactorId] {
        &self.initial_factor_ids
    }

    pub fn initial_factors_for_variable(&self, variable: VariableId) -> Option<&[FactorId]> {
        self.initial_factors_by_variable
            .get(variable.index())
            .map(AsRef::as_ref)
    }

    pub fn potential_factors_for_variable(&self, variable: VariableId) -> Option<&[FactorId]> {
        self.potential_factors_by_variable
            .get(variable.index())
            .map(AsRef::as_ref)
    }

    pub(crate) fn initial_domains(&self) -> impl Iterator<Item = Domain> + '_ {
        self.variables
            .iter()
            .map(VariableDefinition::initial_domain)
    }
}

impl Default for ConstraintModel {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Clone, Debug, Default)]
pub struct ConstraintModelBuilder {
    variables: Vec<VariableDefinition>,
    factors: Vec<FactorDefinition>,
    factor_activation: Vec<FactorActivation>,
    factors_by_variable: Vec<Vec<FactorId>>,
    local_projector_leaves: BTreeSet<VariableId>,
}

impl ConstraintModelBuilder {
    pub const fn new() -> Self {
        Self {
            variables: Vec::new(),
            factors: Vec::new(),
            factor_activation: Vec::new(),
            factors_by_variable: Vec::new(),
            local_projector_leaves: BTreeSet::new(),
        }
    }

    pub fn add_variable(
        &mut self,
        initial_domain: Domain,
    ) -> Result<VariableId, ConstraintModelError> {
        if initial_domain.is_empty() {
            return Err(ConstraintModelError::EmptyInitialDomain);
        }
        let value = u32::try_from(self.variables.len())
            .map_err(|_| ConstraintModelError::VariableCapacityExceeded)?;
        let variable = VariableId::new(value);
        self.variables.push(VariableDefinition { initial_domain });
        self.factors_by_variable.push(Vec::new());
        Ok(variable)
    }

    pub fn add_latent_ring_count_at_most(
        &mut self,
        bonds: impl IntoIterator<Item = SpanningTreeEdge>,
        budget_bits: impl IntoIterator<Item = VariableId>,
    ) -> Result<FactorId, ConstraintModelError> {
        let bonds = bonds.into_iter().collect::<Vec<_>>();
        let budget_bits = budget_bits.into_iter().collect::<Vec<_>>();
        if budget_bits.len() > usize::BITS as usize {
            return Err(ConstraintModelError::InvalidRingCountBudgetWidth(budget_bits.len()));
        }
        let variables = bonds.iter().map(|bond| bond.decision_variable())
            .chain(budget_bits.iter().copied()).collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        for &variable in &variables {
            if self.variables.get(variable.index()).is_none() {
                return Err(ConstraintModelError::UnknownVariable(variable));
            }
            if !seen.insert(variable) {
                return Err(ConstraintModelError::RepeatedVariableInFactor(variable));
            }
        }
        for &variable in &budget_bits {
            if !self.variables[variable.index()].initial_domain.is_subset_of(Domain::from_bits(3)) {
                return Err(ConstraintModelError::InvalidRingCountBudgetDomain(variable));
            }
        }
        for bond in &bonds {
            let variable = bond.decision_variable();
            let domain = self.variables[variable.index()].initial_domain;
            let traversal = bond.role_partition().traversal_values();
            let ring = bond.role_partition().ring_values();
            if !traversal.intersect(ring).is_empty()
                || domain.intersect(traversal).is_empty() && domain.intersect(ring).is_empty()
                || !domain.is_subset_of(traversal.union(ring)) {
                return Err(ConstraintModelError::InvalidEdgeRolePartition(variable));
            }
        }
        self.push_factor(FactorDefinition::RingCountAtMost(RingCountAtMostFactor {
            bonds: bonds.into_boxed_slice(), budget_bits: budget_bits.into_boxed_slice(),
            variables: variables.into_boxed_slice(),
        }), FactorActivation::Latent)
    }

    pub(crate) fn add_latent_ordered_ring_phase(
        &mut self,
        bonds: impl IntoIterator<Item = (BondId, SpanningTreeEdge)>,
        reference_order: [TetrahedralLigand; 4],
        order_variable: VariableId,
        waiting_mask_variable: VariableId,
        free_bits: impl IntoIterator<Item = VariableId>,
    ) -> FactorId {
        let bonds = bonds.into_iter().collect::<Vec<_>>();
        let free_bits = free_bits.into_iter().collect::<Vec<_>>();
        assert!((3..=4).contains(&bonds.len()));
        assert!(free_bits.len() <= usize::BITS as usize);
        assert_eq!(bonds.iter().map(|(bond, _)| *bond).collect::<BTreeSet<_>>().len(), bonds.len());
        assert_eq!(reference_order.iter().filter_map(|ligand| match ligand {
            TetrahedralLigand::Bond(bond) => Some(*bond),
            TetrahedralLigand::VirtualHydrogen => None,
        }).collect::<BTreeSet<_>>().len(), bonds.len());
        assert!(reference_order.iter().all(|ligand| match ligand {
            TetrahedralLigand::Bond(bond) => bonds.iter().any(|(candidate, _)| candidate == bond),
            TetrahedralLigand::VirtualHydrogen => true,
        }));
        let mask_domain = self.variables[waiting_mask_variable.index()].initial_domain();
        assert!(mask_domain.is_subset_of(Domain::from_bits((1_u64 << (1 << bonds.len())) - 1)));
        for &variable in &free_bits {
            assert!(self.variables[variable.index()].initial_domain().is_subset_of(Domain::from_bits(3)));
        }
        let variables = bonds.iter().map(|(_, edge)| edge.decision_variable())
            .chain([order_variable, waiting_mask_variable])
            .chain(free_bits.iter().copied()).collect::<Vec<_>>();
        assert_eq!(variables.iter().copied().collect::<BTreeSet<_>>().len(), variables.len());
        self.push_factor(FactorDefinition::OrderedRingPhase(OrderedRingPhaseFactor {
            bonds: bonds.into_boxed_slice(), reference_order, order_variable,
            waiting_mask_variable, free_bits: free_bits.into_boxed_slice(),
            variables: variables.into_boxed_slice(),
        }), FactorActivation::Latent).expect("prepared ordered ring phase must have valid variables")
    }

    pub fn add_aromatic_matching(
        &mut self,
        atom_variables: impl IntoIterator<Item = VariableId>,
        uniform_case: bool,
        double_demands: impl IntoIterator<Item = u8>,
        edges: impl IntoIterator<Item = AromaticMatchingEdge>,
    ) -> Result<FactorId, ConstraintModelError> {
        let atom_variables = atom_variables.into_iter().collect::<Vec<_>>();
        let double_demands = double_demands.into_iter().collect::<Vec<_>>();
        let edges = edges.into_iter().collect::<Vec<_>>();
        if double_demands.is_empty() {
            return Err(ConstraintModelError::InvalidAromaticMatchingGraph(
                "a component requires at least one atom",
            ));
        }
        if atom_variables.len() != double_demands.len() {
            return Err(ConstraintModelError::InvalidAromaticMatchingGraph(
                "one atom representation variable is required per demand",
            ));
        }
        if double_demands.iter().any(|&demand| demand > 1) {
            return Err(ConstraintModelError::InvalidAromaticMatchingGraph(
                "per-atom double demand must be zero or one",
            ));
        }
        let mut endpoint_pairs = BTreeSet::new();
        for edge in &edges {
            if edge.a >= double_demands.len() || edge.b >= double_demands.len() {
                return Err(ConstraintModelError::InvalidAromaticMatchingGraph(
                    "edge endpoint is outside the component atom set",
                ));
            }
            if edge.a == edge.b {
                return Err(ConstraintModelError::InvalidAromaticMatchingGraph(
                    "a component cannot contain a self-edge",
                ));
            }
            if !endpoint_pairs.insert((edge.a.min(edge.b), edge.a.max(edge.b))) {
                return Err(ConstraintModelError::InvalidAromaticMatchingGraph(
                    "a component cannot contain duplicate edges",
                ));
            }
        }
        let variables = atom_variables.iter().copied()
            .chain(edges.iter().map(|edge| edge.variable))
            .collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        for (index, &variable) in variables.iter().enumerate() {
            let definition = self
                .variables
                .get(variable.index())
                .ok_or(ConstraintModelError::UnknownVariable(variable))?;
            if !seen.insert(variable) {
                return Err(ConstraintModelError::RepeatedVariableInFactor(variable));
            }
            if !self.factors_by_variable[variable.index()].is_empty()
                || self.local_projector_leaves.contains(&variable)
            {
                return Err(ConstraintModelError::LocalProjectorLeafReuse(variable));
            }
            let allowed = Domain::from_bits(if index < atom_variables.len() { 0b11 } else { 0b111 });
            if !definition.initial_domain.is_subset_of(allowed) {
                return Err(ConstraintModelError::InvalidAromaticMatchingDomain(
                    variable,
                ));
            }
        }
        let factor = self.push_factor(
            FactorDefinition::AromaticMatching(AromaticMatchingFactor {
                atom_variables: atom_variables.into_boxed_slice(),
                uniform_case,
                double_demands: double_demands.into_boxed_slice(),
                edges: edges.into_boxed_slice(),
                variables: variables.clone().into_boxed_slice(),
            }),
            FactorActivation::Always,
        )?;
        self.local_projector_leaves.extend(variables);
        Ok(factor)
    }

    pub fn add_binary_relation(
        &mut self,
        left: VariableId,
        right: VariableId,
        allowed_pairs: impl IntoIterator<Item = (u8, u8)>,
    ) -> Result<FactorId, ConstraintModelError> {
        let left_domain = self
            .variables
            .get(left.index())
            .ok_or(ConstraintModelError::UnknownVariable(left))?
            .initial_domain;
        let right_domain = self
            .variables
            .get(right.index())
            .ok_or(ConstraintModelError::UnknownVariable(right))?
            .initial_domain;

        if left == right {
            return Err(ConstraintModelError::RepeatedVariableInFactor(left));
        }

        let mut allowed_right_by_left = vec![Domain::empty(); left_domain.value_span()];
        let mut allowed_left_by_right = vec![Domain::empty(); right_domain.value_span()];

        for (left_value, right_value) in allowed_pairs {
            if !left_domain.contains(left_value) {
                return Err(ConstraintModelError::RelationValueOutsideInitialDomain {
                    variable: left,
                    value_index: left_value,
                });
            }
            if !right_domain.contains(right_value) {
                return Err(ConstraintModelError::RelationValueOutsideInitialDomain {
                    variable: right,
                    value_index: right_value,
                });
            }

            allowed_right_by_left[left_value as usize] = allowed_right_by_left[left_value as usize]
                .union(Domain::from_bits(1_u64 << right_value));
            allowed_left_by_right[right_value as usize] = allowed_left_by_right
                [right_value as usize]
                .union(Domain::from_bits(1_u64 << left_value));
        }

        self.push_factor(
            FactorDefinition::BinaryRelation(BinaryRelationFactor {
                variables: [left, right],
                allowed_right_by_left: allowed_right_by_left.into_boxed_slice(),
                allowed_left_by_right: allowed_left_by_right.into_boxed_slice(),
            }),
            FactorActivation::Always,
        )
    }

    pub fn add_latent_directional_ring_placement(
        &mut self,
        mark_variable: VariableId,
        plan_variable: VariableId,
        allowed_pairs: impl IntoIterator<Item = (u8, u8)>,
    ) -> Result<FactorId, ConstraintModelError> {
        let mark_domain = self
            .variables
            .get(mark_variable.index())
            .ok_or(ConstraintModelError::UnknownVariable(mark_variable))?
            .initial_domain;
        let plan_domain = self
            .variables
            .get(plan_variable.index())
            .ok_or(ConstraintModelError::UnknownVariable(plan_variable))?
            .initial_domain;
        if mark_variable == plan_variable {
            return Err(ConstraintModelError::RepeatedVariableInFactor(
                mark_variable,
            ));
        }
        if self.local_projector_leaves.contains(&mark_variable)
            || !self.factors_by_variable[plan_variable.index()].is_empty()
            || self.local_projector_leaves.contains(&plan_variable)
        {
            return Err(ConstraintModelError::LocalProjectorLeafReuse(
                if self.local_projector_leaves.contains(&mark_variable) {
                    mark_variable
                } else {
                    plan_variable
                },
            ));
        }
        let mut allowed_plans_by_mark = vec![Domain::empty(); mark_domain.value_span()];
        let mut allowed_marks_by_plan = vec![Domain::empty(); plan_domain.value_span()];
        for (mark, plan) in allowed_pairs {
            if !mark_domain.contains(mark) {
                return Err(ConstraintModelError::RelationValueOutsideInitialDomain {
                    variable: mark_variable,
                    value_index: mark,
                });
            }
            if !plan_domain.contains(plan) {
                return Err(ConstraintModelError::RelationValueOutsideInitialDomain {
                    variable: plan_variable,
                    value_index: plan,
                });
            }
            allowed_plans_by_mark[mark as usize] =
                allowed_plans_by_mark[mark as usize].union(Domain::from_bits(1_u64 << plan));
            allowed_marks_by_plan[plan as usize] =
                allowed_marks_by_plan[plan as usize].union(Domain::from_bits(1_u64 << mark));
        }
        let factor = self.push_factor(
            FactorDefinition::DirectionalRingPlacement(DirectionalRingPlacementFactor {
                variables: [mark_variable, plan_variable],
                allowed_plans_by_mark: allowed_plans_by_mark.into_boxed_slice(),
                allowed_marks_by_plan: allowed_marks_by_plan.into_boxed_slice(),
            }),
            FactorActivation::Latent,
        )?;
        self.local_projector_leaves.insert(plan_variable);
        Ok(factor)
    }

    pub fn add_spanning_tree(
        &mut self,
        atoms: impl IntoIterator<Item = AtomId>,
        edges: impl IntoIterator<Item = SpanningTreeEdge>,
    ) -> Result<FactorId, ConstraintModelError> {
        self.add_traversal_topology(TraversalTopologyRequirement::Tree, atoms, edges, [], false)
    }

    pub fn add_spanning_forest(
        &mut self,
        atoms: impl IntoIterator<Item = AtomId>,
        edges: impl IntoIterator<Item = SpanningTreeEdge>,
    ) -> Result<FactorId, ConstraintModelError> {
        self.add_traversal_topology(TraversalTopologyRequirement::Forest, atoms, edges, [], false)
    }

    pub fn add_rooted_spanning_forest(
        &mut self,
        atoms: impl IntoIterator<Item = AtomId>,
        edges: impl IntoIterator<Item = SpanningTreeEdge>,
        roots: impl IntoIterator<Item = (AtomId, VariableId)>,
    ) -> Result<FactorId, ConstraintModelError> {
        self.add_traversal_topology(TraversalTopologyRequirement::Forest, atoms, edges, roots, true)
    }

    fn add_traversal_topology(
        &mut self,
        requirement: TraversalTopologyRequirement,
        atoms: impl IntoIterator<Item = AtomId>,
        edges: impl IntoIterator<Item = SpanningTreeEdge>,
        roots: impl IntoIterator<Item = (AtomId, VariableId)>,
        require_roots: bool,
    ) -> Result<FactorId, ConstraintModelError> {
        let mut atoms = atoms.into_iter().collect::<Vec<_>>();
        if atoms.is_empty() {
            return Err(ConstraintModelError::EmptyTraversalTopologyAtomSet);
        }
        atoms.sort_unstable();
        if atoms.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ConstraintModelError::RepeatedAtomInTraversalTopology);
        }

        let atom_set = atoms.iter().copied().collect::<BTreeSet<_>>();
        let edges = edges.into_iter().collect::<Vec<_>>();
        let mut variables = Vec::with_capacity(edges.len());
        let mut seen_variables = BTreeSet::new();

        let root_pairs = roots.into_iter().collect::<Vec<_>>();
        let root_map = root_pairs.iter().copied().collect::<BTreeMap<_, _>>();
        if root_map.len() != root_pairs.len() ||
            (require_roots && (requirement != TraversalTopologyRequirement::Forest ||
                root_map.len() != atoms.len() ||
                root_map.keys().copied().ne(atoms.iter().copied()))) ||
            (!require_roots && !root_map.is_empty()) {
            return Err(ConstraintModelError::InvalidTraversalTopologyRootVariables);
        }
        let mut root_variables = Vec::with_capacity(root_map.len());
        for variable in root_map.values().copied() {
            let domain = self.variables.get(variable.index())
                .ok_or(ConstraintModelError::UnknownVariable(variable))?.initial_domain;
            if !domain.is_subset_of(Domain::from_indices([0, 1]).unwrap()) ||
                !seen_variables.insert(variable) {
                return Err(ConstraintModelError::InvalidTraversalTopologyRootVariables);
            }
            root_variables.push(variable);
            variables.push(variable);
        }

        for edge in &edges {
            if edge.a == edge.b {
                return Err(ConstraintModelError::TraversalTopologySelfEdge(edge.a));
            }
            for atom in [edge.a, edge.b] {
                if !atom_set.contains(&atom) {
                    return Err(ConstraintModelError::TraversalTopologyEdgeOutsideAtomSet(
                        atom,
                    ));
                }
            }

            let variable = edge.decision_variable;
            let initial_domain = self
                .variables
                .get(variable.index())
                .ok_or(ConstraintModelError::UnknownVariable(variable))?
                .initial_domain;
            let traversal_values = initial_domain.intersect(edge.role_partition.traversal_values());
            let ring_values = initial_domain.intersect(edge.role_partition.ring_values());
            if !edge
                .role_partition
                .traversal_values()
                .intersect(edge.role_partition.ring_values())
                .is_empty()
                || traversal_values.union(ring_values) != initial_domain
            {
                return Err(ConstraintModelError::InvalidEdgeRolePartition(variable));
            }
            if !seen_variables.insert(variable) {
                return Err(ConstraintModelError::RepeatedVariableInFactor(variable));
            }
            if self
                .prepared_edge_role_partition(variable)
                .is_some_and(|existing| existing != edge.role_partition)
            {
                return Err(ConstraintModelError::ConflictingEdgeRolePartition(variable));
            }
            if self.factors_by_variable[variable.index()]
                .iter()
                .any(|factor| {
                    matches!(
                        self.factors
                            .get(factor.index())
                            .expect("prepared factor adjacency must resolve"),
                        FactorDefinition::TraversalTopology(_)
                    )
                })
            {
                return Err(ConstraintModelError::OverlappingTraversalTopologyVariable(
                    variable,
                ));
            }
            variables.push(variable);
        }

        self.push_factor(
            FactorDefinition::TraversalTopology(TraversalTopologyFactor {
                requirement,
                atoms: atoms.into_boxed_slice(),
                edges: edges.into_boxed_slice(),
                root_variables: root_variables.into_boxed_slice(),
                variables: variables.into_boxed_slice(),
            }),
            FactorActivation::Always,
        )
    }

    pub fn add_latent_tetrahedral_layout(
        &mut self,
        order_variable: VariableId,
        role_pattern_variable: VariableId,
        bonds: impl IntoIterator<Item = TetrahedralLayoutBond>,
        allowed_orders_by_pattern: impl IntoIterator<Item = Domain>,
    ) -> Result<FactorId, ConstraintModelError> {
        self.add_latent_tetrahedral_layout_rows(order_variable, role_pattern_variable,
            bonds, allowed_orders_by_pattern.into_iter().collect::<Vec<_>>().into())
    }

    pub(crate) fn add_latent_tetrahedral_layout_rows(
        &mut self,
        order_variable: VariableId,
        role_pattern_variable: VariableId,
        bonds: impl IntoIterator<Item = TetrahedralLayoutBond>,
        allowed_orders_by_pattern: Arc<[Domain]>,
    ) -> Result<FactorId, ConstraintModelError> {
        let order_domain = self
            .variables
            .get(order_variable.index())
            .ok_or(ConstraintModelError::UnknownVariable(order_variable))?
            .initial_domain;
        let pattern_domain = self
            .variables
            .get(role_pattern_variable.index())
            .ok_or(ConstraintModelError::UnknownVariable(role_pattern_variable))?
            .initial_domain;
        if order_variable == role_pattern_variable {
            return Err(ConstraintModelError::RepeatedVariableInFactor(
                order_variable,
            ));
        }

        let bonds = bonds.into_iter().collect::<Vec<_>>();
        if !(3..=4).contains(&bonds.len()) {
            return Err(ConstraintModelError::InvalidTetrahedralBondCount(
                bonds.len(),
            ));
        }
        let pattern_count = 1_usize << bonds.len();
        let expected_pattern_domain = Domain::from_bits((1_u64 << pattern_count) - 1);
        if pattern_domain != expected_pattern_domain {
            return Err(ConstraintModelError::InvalidTetrahedralPatternDomain {
                variable: role_pattern_variable,
                expected: expected_pattern_domain,
            });
        }

        let mut seen_variables = BTreeSet::from([order_variable, role_pattern_variable]);
        let mut seen_bits = BTreeSet::new();
        for bond in &bonds {
            let variable = bond.decision_variable;
            let initial_domain = self
                .variables
                .get(variable.index())
                .ok_or(ConstraintModelError::UnknownVariable(variable))?
                .initial_domain;
            if !seen_variables.insert(variable) {
                return Err(ConstraintModelError::RepeatedVariableInFactor(variable));
            }
            if usize::from(bond.pattern_bit) >= bonds.len() || !seen_bits.insert(bond.pattern_bit) {
                return Err(ConstraintModelError::InvalidTetrahedralPatternBit(
                    bond.pattern_bit,
                ));
            }
            let traversal = initial_domain.intersect(bond.role_partition.traversal_values());
            let ring = initial_domain.intersect(bond.role_partition.ring_values());
            if !bond
                .role_partition
                .traversal_values()
                .intersect(bond.role_partition.ring_values())
                .is_empty()
                || traversal.union(ring) != initial_domain
            {
                return Err(ConstraintModelError::InvalidEdgeRolePartition(variable));
            }
            if self
                .prepared_edge_role_partition(variable)
                .is_some_and(|existing| existing != bond.role_partition)
            {
                return Err(ConstraintModelError::ConflictingEdgeRolePartition(variable));
            }
        }
        if seen_bits.len() != bonds.len() {
            return Err(ConstraintModelError::IncompleteTetrahedralPatternBits);
        }

        if allowed_orders_by_pattern.len() != pattern_count {
            return Err(ConstraintModelError::TetrahedralLayoutRowCountMismatch {
                expected: pattern_count,
                actual: allowed_orders_by_pattern.len(),
            });
        }
        for &allowed in allowed_orders_by_pattern.iter() {
            if !allowed.is_subset_of(order_domain) {
                return Err(ConstraintModelError::TetrahedralOrderOutsideInitialDomain(
                    order_variable,
                ));
            }
        }

        let mut variables = Vec::with_capacity(bonds.len() + 2);
        variables.extend([order_variable, role_pattern_variable]);
        variables.extend(bonds.iter().map(|bond| bond.decision_variable));
        self.push_factor(
            FactorDefinition::TetrahedralLayout(TetrahedralLayoutFactor {
                order_variable,
                role_pattern_variable,
                bonds: bonds.into_boxed_slice(),
                allowed_orders_by_pattern,
                variables: variables.into_boxed_slice(),
            }),
            FactorActivation::Latent,
        )
    }

    pub(crate) fn add_latent_event_schedule(
        &mut self,
        visits: Arc<[crate::event_schedule::EventVisit]>,
        edges: Arc<[ScheduleEdge]>,
        centers: Arc<[ScheduleCenter]>,
        ordinary_centers: Arc<[ScheduleOrdinaryCenter]>,
        capacity_bits: Arc<[VariableId]>,
        forest_roots: Arc<[VariableId]>,
        segments: bool,
    ) -> Result<FactorId, ConstraintModelError> {
        let factor = EventScheduleFactor::new(visits, edges, centers,
            ordinary_centers, capacity_bits, forest_roots, segments);
        let domain = |variable: VariableId| self.variables.get(variable.index())
            .map(|definition| definition.initial_domain)
            .ok_or(ConstraintModelError::UnknownVariable(variable));
        let binary = Domain::from_bits(3);
        let count = factor.visits.len();
        let rank_width = (usize::BITS - count.saturating_sub(1).leading_zeros()) as usize;
        if count == 0 || factor.capacity_bits.len() > usize::BITS as usize
            || factor.visits.iter().any(|visit|
            visit.rank_bits.len() != rank_width || visit.end_bits.len() != count)
            // Empty root scope permits standalone finite event-relation tests;
            // prepared writer forests supply one root fact per atom.
            || !(factor.forest_roots.is_empty() || segments && factor.forest_roots.len() == count)
            || factor.centers.len() + factor.ordinary_centers.len() != count {
            return Err(ConstraintModelError::InvalidEventSchedule("visit, root, or center shape"));
        }
        let mut edge_ids = BTreeSet::new();
        for edge in factor.edges.iter() {
            if !edge_ids.insert(edge.bond) || edge.a == edge.b
                || edge.a.index() >= count || edge.b.index() >= count {
                return Err(ConstraintModelError::InvalidEventSchedule("edge incidence"));
            }
            let initial = domain(edge.decision_variable)?;
            if !edge.role_partition.traversal_values().intersect(edge.role_partition.ring_values()).is_empty()
                || edge.role_partition.traversal_values().union(edge.role_partition.ring_values()) != initial {
                return Err(ConstraintModelError::InvalidEdgeRolePartition(edge.decision_variable));
            }
        }
        let mut covered = BTreeSet::new();
        for center in factor.centers.iter() {
            let incidence = factor.edges.iter().filter(|edge| edge.a == center.atom || edge.b == center.atom)
                .map(|edge| edge.bond).collect::<BTreeSet<_>>();
            if !(3..=4).contains(&incidence.len()) || center.bond_pattern_bits.len() != incidence.len() {
                return Err(ConstraintModelError::InvalidEventSchedule("tetrahedral center degree"));
            }
            let pattern_count = 1_usize << center.bond_pattern_bits.len();
            let pattern_bonds = center.bond_pattern_bits.iter().map(|(bond, _)| *bond).collect::<BTreeSet<_>>();
            let pattern_bits = center.bond_pattern_bits.iter().map(|(_, bit)| usize::from(*bit)).collect::<BTreeSet<_>>();
            let entry_bonds = center.entry_layout_rows.iter().map(|(bond, _)| *bond).collect::<BTreeSet<_>>();
            let reference_bonds = center.reference_order.iter().filter_map(|ligand|
                match ligand { TetrahedralLigand::Bond(bond) => Some(*bond), _ => None })
                .collect::<BTreeSet<_>>();
            let reference_bond_count = center.reference_order.iter().filter(|ligand|
                matches!(ligand, TetrahedralLigand::Bond(_))).count();
            if !covered.insert(center.atom) || center.atom.index() >= count
                || !(3..=4).contains(&incidence.len()) || pattern_bonds != incidence
                || pattern_bits != (0..incidence.len()).collect()
                || entry_bonds != incidence || center.entry_layout_rows.len() != incidence.len()
                || reference_bonds != incidence || reference_bond_count != incidence.len()
                || center.root_layout_rows.len() != pattern_count
                || center.entry_layout_rows.iter().any(|(_, rows)| rows.len() != pattern_count) {
                return Err(ConstraintModelError::InvalidEventSchedule("tetrahedral center incidence or rows"));
            }
            let order = domain(center.order_variable)?;
            let expected_pattern = Domain::from_bits((1_u64 << pattern_count) - 1);
            if domain(center.role_pattern_variable)? != expected_pattern
                || center.root_layout_rows.iter().chain(center.entry_layout_rows.iter()
                    .flat_map(|(_, rows)| rows.iter())).any(|row| !row.is_subset_of(order)) {
                return Err(ConstraintModelError::InvalidEventSchedule("tetrahedral center domain or row"));
            }
        }
        for center in factor.ordinary_centers.iter() {
            let incidence = factor.edges.iter().filter(|edge| edge.a == center.atom || edge.b == center.atom)
                .map(|edge| edge.bond).collect::<Vec<_>>();
            if !covered.insert(center.atom) || center.atom.index() >= count {
                return Err(ConstraintModelError::InvalidEventSchedule("ordinary center identity"));
            }
            assert_complete_ordinary_pairs(&incidence, &center.pairs);
        }
        if covered.len() != count {
            return Err(ConstraintModelError::InvalidEventSchedule("missing center"));
        }
        let mut event_variables = BTreeSet::new();
        for visit in factor.visits.iter() {
            for variable in visit.rank_bits.iter().chain([&visit.kind]).chain(visit.end_bits.iter()) {
                if !event_variables.insert(*variable) || !domain(*variable)?.is_subset_of(if *variable == visit.kind {
                    crate::event_schedule::EntryKind::domain()
                } else { binary }) {
                    return Err(ConstraintModelError::InvalidEventSchedule("event variable or domain"));
                }
            }
        }
        for variable in factor.capacity_bits.iter().chain(factor.forest_roots.iter())
            .chain(factor.ordinary_centers.iter().flat_map(|center| center.pairs.iter().map(|pair| &pair.variable))) {
            if !domain(*variable)?.is_subset_of(binary) {
                return Err(ConstraintModelError::InvalidEventSchedule("binary fact domain"));
            }
        }
        for &variable in factor.variables() { let _ = domain(variable)?; }
        self.push_factor(FactorDefinition::EventSchedule(factor), FactorActivation::Latent)
    }

    pub fn add_latent_ordinary_order(
        &mut self, atom: AtomId, pairs: Arc<[OrdinaryOrderPair]>,
    ) -> Result<FactorId, ConstraintModelError> {
        let incidence = pairs.iter().flat_map(|pair| [pair.first, pair.second])
            .collect::<std::collections::BTreeSet<_>>().into_iter().collect::<Vec<_>>();
        assert_complete_ordinary_pairs(&incidence, &pairs);
        let variables = pairs.iter().map(|pair| pair.variable).collect::<Vec<_>>();
        for variable in &variables {
            let domain = self.variables.get(variable.index())
                .ok_or(ConstraintModelError::UnknownVariable(*variable))?.initial_domain;
            assert!(domain.is_subset_of(Domain::from_bits(3)),
                "ordinary order direction must be binary");
        }
        self.push_factor(FactorDefinition::OrdinaryOrder(OrdinaryOrderFactor {
            atom, pairs, variables: variables.into_boxed_slice(),
        }), FactorActivation::Latent)
    }

    pub fn build(self) -> ConstraintModel {
        let initial_factor_ids = self
            .factor_activation
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, activation)| {
                (activation == FactorActivation::Always).then(|| {
                    FactorId::new(
                        u32::try_from(index)
                            .expect("prepared factor count must fit its identifier space"),
                    )
                })
            })
            .collect::<Vec<_>>();
        let initial_factors_by_variable = self
            .factors_by_variable
            .iter()
            .map(|factors| {
                factors
                    .iter()
                    .copied()
                    .filter(|factor| {
                        self.factor_activation[factor.index()] == FactorActivation::Always
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        ConstraintModel {
            variables: self.variables.into_boxed_slice(),
            factors: self.factors.into_boxed_slice(),
            factor_activation: self.factor_activation.into_boxed_slice(),
            initial_factor_ids: initial_factor_ids.into_boxed_slice(),
            initial_factors_by_variable,
            potential_factors_by_variable: self
                .factors_by_variable
                .into_iter()
                .map(Vec::into_boxed_slice)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn push_factor(
        &mut self,
        definition: FactorDefinition,
        activation: FactorActivation,
    ) -> Result<FactorId, ConstraintModelError> {
        if let Some(variable) = definition
            .variables()
            .iter()
            .copied()
            .find(|variable| self.local_projector_leaves.contains(variable))
        {
            return Err(ConstraintModelError::LocalProjectorLeafReuse(variable));
        }
        let value = u32::try_from(self.factors.len())
            .map_err(|_| ConstraintModelError::FactorCapacityExceeded)?;
        let factor = FactorId::new(value);
        for variable in definition.variables().iter().copied() {
            self.factors_by_variable[variable.index()].push(factor);
        }
        self.factors.push(definition);
        self.factor_activation.push(activation);
        Ok(factor)
    }

    fn prepared_edge_role_partition(&self, variable: VariableId) -> Option<EdgeRolePartition> {
        self.factors_by_variable[variable.index()]
            .iter()
            .find_map(|factor| match &self.factors[factor.index()] {
                FactorDefinition::AromaticMatching(_)
                | FactorDefinition::BinaryRelation(_)
                | FactorDefinition::DirectionalRingPlacement(_)
                | FactorDefinition::RingCountAtMost(_) => None,
                FactorDefinition::OrderedRingPhase(_) => None,
                FactorDefinition::OrdinaryOrder(_) => None,
                FactorDefinition::EventSchedule(schedule) => schedule.edges.iter()
                    .find(|edge| edge.decision_variable == variable)
                    .map(|edge| edge.role_partition),
                FactorDefinition::TraversalTopology(spanning) => spanning
                    .edges()
                    .iter()
                    .find(|edge| edge.decision_variable() == variable)
                    .map(|edge| edge.role_partition()),
                FactorDefinition::TetrahedralLayout(layout) => layout
                    .bonds()
                    .iter()
                    .find(|bond| bond.decision_variable() == variable)
                    .map(|bond| bond.role_partition()),
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstraintModelError {
    InvalidEventSchedule(&'static str),
    InvalidAromaticMatchingGraph(&'static str),
    InvalidAromaticMatchingDomain(VariableId),
    EmptyInitialDomain,
    VariableCapacityExceeded,
    FactorCapacityExceeded,
    UnknownVariable(VariableId),
    RepeatedVariableInFactor(VariableId),
    LocalProjectorLeafReuse(VariableId),
    RelationValueOutsideInitialDomain {
        variable: VariableId,
        value_index: u8,
    },
    EmptyTraversalTopologyAtomSet,
    RepeatedAtomInTraversalTopology,
    TraversalTopologySelfEdge(AtomId),
    TraversalTopologyEdgeOutsideAtomSet(AtomId),
    InvalidTraversalTopologyRootVariables,
    InvalidEdgeRolePartition(VariableId),
    ConflictingEdgeRolePartition(VariableId),
    OverlappingTraversalTopologyVariable(VariableId),
    InvalidTetrahedralBondCount(usize),
    InvalidRingCountBudgetWidth(usize),
    InvalidRingCountBudgetDomain(VariableId),
    InvalidTetrahedralPatternDomain {
        variable: VariableId,
        expected: Domain,
    },
    InvalidTetrahedralPatternBit(u8),
    IncompleteTetrahedralPatternBits,
    TetrahedralLayoutRowCountMismatch {
        expected: usize,
        actual: usize,
    },
    TetrahedralOrderOutsideInitialDomain(VariableId),
}

impl fmt::Display for ConstraintModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEventSchedule(reason) => write!(formatter, "cannot build event schedule: {reason}"),
            Self::InvalidAromaticMatchingGraph(reason) => write!(
                formatter, "cannot build aromatic matching component: {reason}"),
            Self::InvalidAromaticMatchingDomain(variable) => write!(
                formatter, "aromatic matching variable {variable:?} has values outside its mode (0..1) or edge chemistry (0..2) domain"),
            Self::EmptyInitialDomain => {
                formatter.write_str("constraint variables require a nonempty initial domain")
            }
            Self::VariableCapacityExceeded => {
                formatter.write_str("too many variables for the South Star 2 identifier space")
            }
            Self::FactorCapacityExceeded => {
                formatter.write_str("too many factors for the South Star 2 identifier space")
            }
            Self::UnknownVariable(variable) => {
                write!(
                    formatter,
                    "constraint factor references unknown variable {variable:?}"
                )
            }
            Self::RepeatedVariableInFactor(variable) => {
                write!(
                    formatter,
                    "constraint factor repeats variable {variable:?} in its scope"
                )
            }
            Self::LocalProjectorLeafReuse(variable) => write!(
                formatter,
                "local projector leaf variable {variable:?} cannot participate in another factor"
            ),
            Self::RelationValueOutsideInitialDomain {
                variable,
                value_index,
            } => {
                write!(
                    formatter,
                    "relation value {value_index} is outside the initial domain of {variable:?}"
                )
            }
            Self::EmptyTraversalTopologyAtomSet => {
                formatter.write_str("a traversal-topology factor requires at least one atom")
            }
            Self::RepeatedAtomInTraversalTopology => {
                formatter.write_str("a traversal-topology factor cannot repeat an atom")
            }
            Self::TraversalTopologySelfEdge(atom) => {
                write!(
                    formatter,
                    "a traversal-topology factor cannot contain a self-edge at {atom:?}"
                )
            }
            Self::TraversalTopologyEdgeOutsideAtomSet(atom) => {
                write!(
                    formatter,
                    "traversal-topology edge endpoint {atom:?} is outside the factor atom set"
                )
            }
            Self::InvalidTraversalTopologyRootVariables => write!(
                formatter,
                "a rooted forest needs one distinct binary root variable for each factor atom"
            ),
            Self::InvalidEdgeRolePartition(variable) => {
                write!(
                    formatter,
                    "edge variable {variable:?} is not partitioned into disjoint Traversal and Ring values"
                )
            }
            Self::ConflictingEdgeRolePartition(variable) => write!(
                formatter,
                "edge variable {variable:?} uses inconsistent Traversal/Ring partitions across factors"
            ),
            Self::OverlappingTraversalTopologyVariable(variable) => {
                write!(
                    formatter,
                    "constraint variable {variable:?} belongs to multiple traversal-topology factors"
                )
            }
            Self::InvalidTetrahedralBondCount(count) => write!(
                formatter,
                "a tetrahedral layout factor requires three or four bond variables, got {count}"
            ),
            Self::InvalidRingCountBudgetWidth(width) => write!(
                formatter, "cannot add a ring-count factor with {width} budget bits; the limit is {} bits", usize::BITS
            ),
            Self::InvalidRingCountBudgetDomain(variable) => write!(
                formatter, "cannot add a ring-count factor with budget bit {variable:?}; its initial domain must contain only 0 and 1"
            ),
            Self::InvalidTetrahedralPatternDomain { variable, expected } => write!(
                formatter,
                "tetrahedral role-pattern variable {variable:?} must have initial domain {expected:?}"
            ),
            Self::InvalidTetrahedralPatternBit(bit) => write!(
                formatter,
                "tetrahedral layout pattern bit {bit} is repeated or outside the bond scope"
            ),
            Self::IncompleteTetrahedralPatternBits => formatter.write_str(
                "tetrahedral layout pattern bits must cover every bond position exactly once",
            ),
            Self::TetrahedralLayoutRowCountMismatch { expected, actual } => write!(
                formatter,
                "tetrahedral layout requires {expected} pattern rows, got {actual}"
            ),
            Self::TetrahedralOrderOutsideInitialDomain(variable) => write!(
                formatter,
                "tetrahedral layout contains an order outside the initial domain of {variable:?}"
            ),
        }
    }
}

impl std::error::Error for ConstraintModelError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_count_builder_rejects_nonbinary_budget_and_oversized_width() {
        let mut builder = ConstraintModelBuilder::new();
        let nonbinary = builder.add_variable(Domain::singleton(2).unwrap()).unwrap();
        let error = builder.add_latent_ring_count_at_most([], [nonbinary]).unwrap_err();
        assert_eq!(error, ConstraintModelError::InvalidRingCountBudgetDomain(nonbinary));
        assert!(error.to_string().contains("initial domain must contain only 0 and 1"));

        let bits = (0..=usize::BITS).map(|_| builder.add_variable(Domain::from_bits(3)).unwrap())
            .collect::<Vec<_>>();
        let error = builder.add_latent_ring_count_at_most([], bits).unwrap_err();
        assert_eq!(error, ConstraintModelError::InvalidRingCountBudgetWidth(usize::BITS as usize + 1));
        assert!(error.to_string().contains("cannot add a ring-count factor"));
    }

    fn two_value_domain() -> Domain {
        Domain::from_indices([0, 1]).unwrap()
    }

    #[test]
    fn bond_role_domains_have_stable_values() {
        assert_eq!(BondRole::Traversal.value_index(), 0);
        assert_eq!(BondRole::Ring.value_index(), 1);
        assert_eq!(BondRole::role_domain(), two_value_domain());
        assert_eq!(
            BondRole::Traversal.singleton_domain(),
            Domain::singleton(0).unwrap()
        );
        assert_eq!(
            BondRole::Ring.singleton_domain(),
            Domain::singleton(1).unwrap()
        );
    }

    #[test]
    fn empty_constraint_model_is_valid() {
        let model = ConstraintModel::empty();

        assert_eq!(model.variable_count(), 0);
        assert_eq!(model.factor_count(), 0);
        assert_eq!(model.variable(VariableId::new(0)), None);
    }

    #[test]
    fn invalid_variable_is_rejected_without_consuming_an_id() {
        let mut builder = ConstraintModelBuilder::new();

        assert_eq!(
            builder.add_variable(Domain::empty()),
            Err(ConstraintModelError::EmptyInitialDomain)
        );
        assert_eq!(
            builder.add_variable(Domain::singleton(0).unwrap()).unwrap(),
            VariableId::new(0)
        );
    }

    #[test]
    fn model_assigns_stable_ids_and_adjacency() {
        let mut builder = ConstraintModelBuilder::new();
        let left_domain = two_value_domain();
        let right_domain = Domain::singleton(7).unwrap();
        let left = builder.add_variable(left_domain).unwrap();
        let right = builder.add_variable(right_domain).unwrap();
        let factor = builder
            .add_binary_relation(left, right, [(0, 7), (1, 7)])
            .unwrap();
        let model = builder.build();

        assert_eq!(left, VariableId::new(0));
        assert_eq!(right, VariableId::new(1));
        assert_eq!(factor, FactorId::new(0));
        assert_eq!(model.variable(left).unwrap().initial_domain(), left_domain);
        assert_eq!(
            model.variable(right).unwrap().initial_domain(),
            right_domain
        );
        assert_eq!(
            model.potential_factors_for_variable(left),
            Some(&[factor][..])
        );
        assert_eq!(
            model.potential_factors_for_variable(right),
            Some(&[factor][..])
        );
    }

    #[test]
    fn binary_relation_is_compiled_in_both_directions() {
        let mut builder = ConstraintModelBuilder::new();
        let left = builder.add_variable(two_value_domain()).unwrap();
        let right = builder.add_variable(two_value_domain()).unwrap();
        let factor_id = builder
            .add_binary_relation(left, right, [(0, 1), (1, 0), (1, 0)])
            .unwrap();
        let model = builder.build();
        let FactorDefinition::BinaryRelation(factor) = model.factor(factor_id).unwrap() else {
            panic!("expected binary relation factor");
        };

        assert_eq!(factor.variables(), &[left, right]);
        assert_eq!(factor.allowed_right(0), Domain::singleton(1).unwrap());
        assert_eq!(factor.allowed_right(1), Domain::singleton(0).unwrap());
        assert_eq!(factor.allowed_left(0), Domain::singleton(1).unwrap());
        assert_eq!(factor.allowed_left(1), Domain::singleton(0).unwrap());
    }

    #[test]
    fn relation_tables_match_the_initial_value_spans() {
        let mut builder = ConstraintModelBuilder::new();
        let left = builder
            .add_variable(Domain::from_indices([0, 2]).unwrap())
            .unwrap();
        let right = builder
            .add_variable(Domain::from_indices([1, 7]).unwrap())
            .unwrap();
        let factor_id = builder
            .add_binary_relation(left, right, [(0, 1), (2, 7)])
            .unwrap();
        let model = builder.build();
        let FactorDefinition::BinaryRelation(factor) = model.factor(factor_id).unwrap() else {
            panic!("expected binary relation factor");
        };

        assert_eq!(factor.allowed_right_by_left.len(), 3);
        assert_eq!(factor.allowed_left_by_right.len(), 8);
        assert!(factor.allowed_right(63).is_empty());
    }

    #[test]
    fn invalid_binary_factor_is_rejected_without_consuming_an_id() {
        let mut builder = ConstraintModelBuilder::new();
        let left = builder.add_variable(two_value_domain()).unwrap();
        let right = builder.add_variable(two_value_domain()).unwrap();

        assert_eq!(
            builder.add_binary_relation(left, right, [(0, 2)]),
            Err(ConstraintModelError::RelationValueOutsideInitialDomain {
                variable: right,
                value_index: 2,
            })
        );
        assert_eq!(
            builder.add_binary_relation(left, VariableId::new(99), [(0, 0)]),
            Err(ConstraintModelError::UnknownVariable(VariableId::new(99)))
        );
        assert_eq!(
            builder.add_binary_relation(left, left, [(0, 0)]),
            Err(ConstraintModelError::RepeatedVariableInFactor(left))
        );
        assert_eq!(
            builder
                .add_binary_relation(left, right, [(0, 0), (1, 1)])
                .unwrap(),
            FactorId::new(0)
        );
    }

    #[test]
    fn empty_binary_relation_is_a_valid_model_definition() {
        let mut builder = ConstraintModelBuilder::new();
        let left = builder.add_variable(two_value_domain()).unwrap();
        let right = builder.add_variable(two_value_domain()).unwrap();
        let factor_id = builder
            .add_binary_relation(left, right, std::iter::empty())
            .unwrap();
        let model = builder.build();
        let FactorDefinition::BinaryRelation(factor) = model.factor(factor_id).unwrap() else {
            panic!("expected binary relation factor");
        };

        assert!(factor.allowed_right(0).is_empty());
        assert!(factor.allowed_left(0).is_empty());
    }

    #[test]
    fn spanning_tree_factor_preserves_sorted_atoms_edges_and_adjacency() {
        let mut builder = ConstraintModelBuilder::new();
        let first = builder.add_variable(BondRole::role_domain()).unwrap();
        let second = builder.add_variable(BondRole::role_domain()).unwrap();
        let atoms = [AtomId::new(2), AtomId::new(0), AtomId::new(1)];
        let factor_id = builder
            .add_spanning_tree(
                atoms,
                [
                    SpanningTreeEdge::new(first, AtomId::new(0), AtomId::new(1)),
                    SpanningTreeEdge::new(second, AtomId::new(1), AtomId::new(2)),
                ],
            )
            .unwrap();
        let model = builder.build();
        let FactorDefinition::TraversalTopology(factor) = model.factor(factor_id).unwrap() else {
            panic!("expected traversal-topology factor");
        };

        assert_eq!(
            factor.atoms(),
            &[AtomId::new(0), AtomId::new(1), AtomId::new(2)]
        );
        assert_eq!(factor.variables(), &[first, second]);
        assert_eq!(
            model.potential_factors_for_variable(first),
            Some(&[factor_id][..])
        );
        assert_eq!(
            model.potential_factors_for_variable(second),
            Some(&[factor_id][..])
        );
    }

    #[test]
    fn isolated_atom_spanning_tree_has_empty_scope() {
        let mut builder = ConstraintModelBuilder::new();
        let factor_id = builder
            .add_spanning_tree([AtomId::new(7)], std::iter::empty())
            .unwrap();
        let model = builder.build();
        let FactorDefinition::TraversalTopology(factor) = model.factor(factor_id).unwrap() else {
            panic!("expected traversal-topology factor");
        };

        assert_eq!(factor.atoms(), &[AtomId::new(7)]);
        assert!(factor.edges().is_empty());
        assert!(factor.variables().is_empty());
    }

    #[test]
    fn rooted_forest_requires_one_root_variable_per_atom() {
        let mut builder = ConstraintModelBuilder::new();
        let atoms = [AtomId::new(0), AtomId::new(1)];
        assert_eq!(
            builder.add_rooted_spanning_forest(atoms, [], []),
            Err(ConstraintModelError::InvalidTraversalTopologyRootVariables),
        );
        let roots = atoms.map(|_| builder.add_variable(Domain::from_indices([0, 1]).unwrap()).unwrap());
        let factor = builder.add_rooted_spanning_forest(atoms, [], atoms.into_iter().zip(roots)).unwrap();
        let model = builder.build();
        let FactorDefinition::TraversalTopology(topology) = model.factor(factor).unwrap() else {
            panic!("expected rooted forest factor")
        };
        assert_eq!(topology.root_variables(), &roots);
        for variable in roots {
            assert_eq!(model.potential_factors_for_variable(variable), Some(&[factor][..]));
        }
    }

    #[test]
    fn invalid_spanning_tree_definition_is_rejected_without_consuming_an_id() {
        let mut builder = ConstraintModelBuilder::new();
        let role = builder.add_variable(BondRole::role_domain()).unwrap();
        let invalid_role = builder
            .add_variable(Domain::from_indices([0, 2]).unwrap())
            .unwrap();
        let globally_overlapping_role = builder
            .add_variable(Domain::from_indices([0, 1]).unwrap())
            .unwrap();

        assert_eq!(
            builder.add_spanning_tree([], std::iter::empty()),
            Err(ConstraintModelError::EmptyTraversalTopologyAtomSet)
        );
        assert_eq!(
            builder.add_spanning_tree([AtomId::new(0), AtomId::new(0)], std::iter::empty()),
            Err(ConstraintModelError::RepeatedAtomInTraversalTopology)
        );
        assert_eq!(
            builder.add_spanning_tree(
                [AtomId::new(0)],
                [SpanningTreeEdge::new(role, AtomId::new(0), AtomId::new(0),)],
            ),
            Err(ConstraintModelError::TraversalTopologySelfEdge(
                AtomId::new(0)
            ))
        );
        assert_eq!(
            builder.add_spanning_tree(
                [AtomId::new(0), AtomId::new(1)],
                [SpanningTreeEdge::new(role, AtomId::new(0), AtomId::new(2),)],
            ),
            Err(ConstraintModelError::TraversalTopologyEdgeOutsideAtomSet(
                AtomId::new(2)
            ))
        );
        assert_eq!(
            builder.add_spanning_tree(
                [AtomId::new(0), AtomId::new(1)],
                [SpanningTreeEdge::new(
                    invalid_role,
                    AtomId::new(0),
                    AtomId::new(1),
                )],
            ),
            Err(ConstraintModelError::InvalidEdgeRolePartition(invalid_role))
        );
        assert_eq!(
            builder.add_spanning_tree(
                [AtomId::new(0), AtomId::new(1)],
                [SpanningTreeEdge::with_role_partition(
                    globally_overlapping_role,
                    AtomId::new(0),
                    AtomId::new(1),
                    EdgeRolePartition::new(
                        Domain::from_indices([0, 2]).unwrap(),
                        Domain::from_indices([1, 2]).unwrap(),
                    ),
                )],
            ),
            Err(ConstraintModelError::InvalidEdgeRolePartition(
                globally_overlapping_role
            ))
        );
        assert_eq!(
            builder.add_spanning_tree(
                [AtomId::new(0), AtomId::new(1), AtomId::new(2)],
                [
                    SpanningTreeEdge::new(role, AtomId::new(0), AtomId::new(1)),
                    SpanningTreeEdge::new(role, AtomId::new(1), AtomId::new(2)),
                ],
            ),
            Err(ConstraintModelError::RepeatedVariableInFactor(role))
        );
        assert_eq!(
            builder
                .add_spanning_tree([AtomId::new(0)], std::iter::empty())
                .unwrap(),
            FactorId::new(0)
        );
    }

    #[test]
    fn spanning_tree_projectors_cannot_overlap_on_a_variable() {
        let mut builder = ConstraintModelBuilder::new();
        let role = builder.add_variable(BondRole::role_domain()).unwrap();
        let atoms = [AtomId::new(0), AtomId::new(1)];
        assert_eq!(
            builder
                .add_spanning_tree(atoms, [SpanningTreeEdge::new(role, atoms[0], atoms[1])],)
                .unwrap(),
            FactorId::new(0)
        );

        assert_eq!(
            builder.add_spanning_tree(atoms, [SpanningTreeEdge::new(role, atoms[0], atoms[1])],),
            Err(ConstraintModelError::OverlappingTraversalTopologyVariable(
                role
            ))
        );
        assert_eq!(
            builder
                .add_spanning_tree([AtomId::new(2)], std::iter::empty())
                .unwrap(),
            FactorId::new(1)
        );
    }

    #[test]
    fn tetrahedral_layout_retains_latent_context_and_role_mapping() {
        let mut builder = ConstraintModelBuilder::new();
        let order = builder
            .add_variable(Domain::from_indices(0_u8..4).unwrap())
            .unwrap();
        let pattern = builder
            .add_variable(Domain::from_indices(0_u8..8).unwrap())
            .unwrap();
        let role_domain = Domain::from_indices([0, 1, 2]).unwrap();
        let partition = EdgeRolePartition::new(
            Domain::singleton(0).unwrap(),
            Domain::from_indices([1, 2]).unwrap(),
        );
        let bonds = (0_u8..3)
            .map(|bit| {
                let variable = builder.add_variable(role_domain).unwrap();
                TetrahedralLayoutBond::new(variable, partition, bit)
            })
            .collect::<Vec<_>>();
        let rows = (0_u8..8)
            .map(|pattern| Domain::singleton(pattern % 4).unwrap())
            .collect::<Vec<_>>();

        let factor_id = builder
            .add_latent_tetrahedral_layout(order, pattern, bonds.clone(), rows.clone())
            .unwrap();
        let model = builder.build();
        let FactorDefinition::TetrahedralLayout(factor) = model.factor(factor_id).unwrap() else {
            panic!("expected tetrahedral layout factor");
        };

        assert_eq!(
            model.factor_activation(factor_id),
            Some(FactorActivation::Latent)
        );
        assert!(model.initial_factor_ids().is_empty());
        assert_eq!(model.initial_factors_for_variable(order), Some(&[][..]));
        assert_eq!(
            model.potential_factors_for_variable(order),
            Some(&[factor_id][..])
        );
        assert_eq!(factor.order_variable(), order);
        assert_eq!(factor.role_pattern_variable(), pattern);
        assert_eq!(factor.bonds(), bonds);
        assert_eq!(
            factor.variables(),
            &[
                order,
                pattern,
                bonds[0].decision_variable(),
                bonds[1].decision_variable(),
                bonds[2].decision_variable()
            ]
        );
        for (pattern, expected) in rows.into_iter().enumerate() {
            assert_eq!(factor.allowed_orders(pattern as u8), expected);
        }
    }

    #[test]
    fn edge_role_partition_is_one_model_wide_structural_fact() {
        let domain = Domain::from_indices([0, 1, 2]).unwrap();
        let first = EdgeRolePartition::new(
            Domain::singleton(0).unwrap(),
            Domain::from_indices([1, 2]).unwrap(),
        );
        let second = EdgeRolePartition::new(
            Domain::from_indices([0, 1]).unwrap(),
            Domain::singleton(2).unwrap(),
        );
        let rows = || [Domain::singleton(0).unwrap(); 8];

        let mut layout_first = ConstraintModelBuilder::new();
        let order = layout_first
            .add_variable(Domain::singleton(0).unwrap())
            .unwrap();
        let pattern = layout_first
            .add_variable(Domain::from_indices(0_u8..8).unwrap())
            .unwrap();
        let roles: [VariableId; 3] =
            std::array::from_fn(|_| layout_first.add_variable(domain).unwrap());
        layout_first
            .add_latent_tetrahedral_layout(
                order,
                pattern,
                roles
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(bit, variable)| TetrahedralLayoutBond::new(variable, first, bit as u8)),
                rows(),
            )
            .unwrap();
        assert_eq!(
            layout_first.add_spanning_tree(
                [AtomId::new(0), AtomId::new(1)],
                [SpanningTreeEdge::with_role_partition(
                    roles[0],
                    AtomId::new(0),
                    AtomId::new(1),
                    second,
                )],
            ),
            Err(ConstraintModelError::ConflictingEdgeRolePartition(roles[0]))
        );

        let mut spanning_first = ConstraintModelBuilder::new();
        let order = spanning_first
            .add_variable(Domain::singleton(0).unwrap())
            .unwrap();
        let pattern = spanning_first
            .add_variable(Domain::from_indices(0_u8..8).unwrap())
            .unwrap();
        let roles: [VariableId; 3] =
            std::array::from_fn(|_| spanning_first.add_variable(domain).unwrap());
        spanning_first
            .add_spanning_tree(
                [AtomId::new(0), AtomId::new(1)],
                [SpanningTreeEdge::with_role_partition(
                    roles[0],
                    AtomId::new(0),
                    AtomId::new(1),
                    first,
                )],
            )
            .unwrap();
        assert_eq!(
            spanning_first.add_latent_tetrahedral_layout(
                order,
                pattern,
                roles.iter().copied().enumerate().map(|(bit, variable)| {
                    TetrahedralLayoutBond::new(variable, second, bit as u8)
                }),
                rows(),
            ),
            Err(ConstraintModelError::ConflictingEdgeRolePartition(roles[0]))
        );
    }

    #[test]
    fn tetrahedral_layout_rejects_invalid_pattern_shape() {
        let mut builder = ConstraintModelBuilder::new();
        let order = builder
            .add_variable(Domain::from_indices(0_u8..4).unwrap())
            .unwrap();
        let malformed_pattern = builder
            .add_variable(Domain::from_indices(0_u8..4).unwrap())
            .unwrap();
        let roles = (0..3)
            .map(|_| builder.add_variable(BondRole::role_domain()).unwrap())
            .collect::<Vec<_>>();
        let bonds = roles
            .iter()
            .copied()
            .enumerate()
            .map(|(bit, variable)| {
                TetrahedralLayoutBond::new(variable, EdgeRolePartition::bond_role(), bit as u8)
            })
            .collect::<Vec<_>>();

        assert_eq!(
            builder.add_latent_tetrahedral_layout(
                order,
                malformed_pattern,
                bonds,
                vec![Domain::singleton(0).unwrap(); 8],
            ),
            Err(ConstraintModelError::InvalidTetrahedralPatternDomain {
                variable: malformed_pattern,
                expected: Domain::from_indices(0_u8..8).unwrap(),
            })
        );
    }

    #[test]
    fn directional_ring_plan_variable_is_an_exclusive_local_projector_leaf() {
        let domain = Domain::from_indices(0_u8..3).unwrap();

        let mut occupied_plan = ConstraintModelBuilder::new();
        let mark = occupied_plan.add_variable(domain).unwrap();
        let plan = occupied_plan.add_variable(domain).unwrap();
        let other = occupied_plan.add_variable(domain).unwrap();
        occupied_plan
            .add_binary_relation(plan, other, [(0, 0), (1, 1), (2, 2)])
            .unwrap();
        assert_eq!(
            occupied_plan.add_latent_directional_ring_placement(
                mark,
                plan,
                [(0, 0), (1, 1), (2, 2)],
            ),
            Err(ConstraintModelError::LocalProjectorLeafReuse(plan))
        );

        let mut reused_plan = ConstraintModelBuilder::new();
        let mark = reused_plan.add_variable(domain).unwrap();
        let plan = reused_plan.add_variable(domain).unwrap();
        let other = reused_plan.add_variable(domain).unwrap();
        reused_plan
            .add_latent_directional_ring_placement(mark, plan, [(0, 0), (1, 1), (2, 2)])
            .unwrap();
        assert_eq!(
            reused_plan.add_binary_relation(plan, other, [(0, 0), (1, 1), (2, 2)]),
            Err(ConstraintModelError::LocalProjectorLeafReuse(plan))
        );
        assert_eq!(
            reused_plan.add_latent_directional_ring_placement(
                plan,
                other,
                [(0, 0), (1, 1), (2, 2)],
            ),
            Err(ConstraintModelError::LocalProjectorLeafReuse(plan))
        );
    }
}
