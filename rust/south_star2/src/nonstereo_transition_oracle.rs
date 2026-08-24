//! Independent bounded conformance model for the prepared non-stereo writer.
//!
//! This module deliberately shares only primitive fixture facts with
//! production. It derives its own adjacency, components, plan domains,
//! spanning-tree projections, residual attachments, and transition relation.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::{
    BondRepresentation, NonStereoBondToken, NonStereoWriterState, ObservedNonStereoState,
    ObservedPending, PreparedNonStereo,
};
use crate::domain::Domain;
use crate::native::NativeSolverState;
use crate::prepared::{PreparedGraphBuilder, PreparedMolecule};
use crate::solver::Consistency;
use crate::traversal::{ObservedBondProgress, ObservedFrame};

type ProductionState = NonStereoWriterState<NativeSolverState>;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum FixtureBondToken {
    Elided,
    Aromatic,
    Single,
    Double,
    Triple,
    DativeAToB,
    DativeBToA,
}

impl FixtureBondToken {
    const fn initial_plan_domain(self) -> OraclePlanDomain {
        match self {
            Self::Elided => OraclePlanDomain::from_bits(
                OracleBondPlan::Traversal.bit() | OracleBondPlan::Ring00.bit(),
            ),
            Self::Aromatic
            | Self::Single
            | Self::Double
            | Self::Triple
            | Self::DativeAToB
            | Self::DativeBToA => OraclePlanDomain::from_bits(
                OracleBondPlan::Traversal.bit()
                    | OracleBondPlan::Ring10.bit()
                    | OracleBondPlan::Ring01.bit()
                    | OracleBondPlan::Ring11.bit(),
            ),
        }
    }

    const fn production(self) -> NonStereoBondToken {
        match self {
            Self::Elided => NonStereoBondToken::Elided,
            Self::Aromatic => NonStereoBondToken::Aromatic,
            Self::Single => NonStereoBondToken::Single,
            Self::Double => NonStereoBondToken::Double,
            Self::Triple => NonStereoBondToken::Triple,
            Self::DativeAToB => NonStereoBondToken::DativeAToB,
            Self::DativeBToA => NonStereoBondToken::DativeBToA,
        }
    }

    fn text_from(self, bond: OracleBond, from: usize) -> &'static str {
        let from_a = from == bond.a;
        assert!(
            from_a || from == bond.b,
            "bond text requires one fixed endpoint"
        );
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
struct FixtureBond {
    a: usize,
    b: usize,
    token: FixtureBondToken,
}

#[derive(Clone, Debug)]
struct FixtureSpec {
    atom_texts: Vec<String>,
    bonds: Vec<FixtureBond>,
}

impl FixtureSpec {
    fn new(atom_texts: &[&str], bonds: &[(usize, usize, FixtureBondToken)]) -> Self {
        Self {
            atom_texts: atom_texts.iter().map(|text| (*text).to_owned()).collect(),
            bonds: bonds
                .iter()
                .map(|&(a, b, token)| FixtureBond { a, b, token })
                .collect(),
        }
    }

    fn build_production_surface(&self) -> PreparedNonStereo {
        let mut graph = PreparedGraphBuilder::new();
        let atoms = self
            .atom_texts
            .iter()
            .map(|_| graph.add_atom().unwrap())
            .collect::<Vec<_>>();
        for bond in &self.bonds {
            graph.add_bond(atoms[bond.a], atoms[bond.b]).unwrap();
        }
        PreparedNonStereo::new(
            PreparedMolecule::new(graph.build()),
            self.atom_texts.clone(),
            self.bonds
                .iter()
                .map(|bond| bond.token.production())
                .collect(),
        )
        .unwrap()
    }

    fn build_oracle_surface(&self) -> OracleSurface {
        OracleSurface::from_fixture(self)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OracleIncidence {
    atom: usize,
    bond: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct OracleBond {
    a: usize,
    b: usize,
    token: FixtureBondToken,
}

impl OracleBond {
    fn other(self, atom: usize) -> usize {
        if atom == self.a {
            self.b
        } else {
            assert_eq!(atom, self.b, "incidence must use one endpoint of its bond");
            self.a
        }
    }

    fn fixed_endpoint(self, atom: usize) -> FixedEndpoint {
        if atom == self.a {
            FixedEndpoint::A
        } else {
            assert_eq!(atom, self.b, "ring event must use one fixed endpoint");
            FixedEndpoint::B
        }
    }
}

#[derive(Clone, Debug)]
struct OracleComponent {
    atoms: Vec<usize>,
    bonds: Vec<usize>,
}

#[derive(Clone, Debug)]
struct OracleSurface {
    atom_texts: Vec<String>,
    bonds: Vec<OracleBond>,
    adjacency: Vec<Vec<OracleIncidence>>,
    components: Vec<OracleComponent>,
}

impl OracleSurface {
    fn from_fixture(fixture: &FixtureSpec) -> Self {
        let atom_count = fixture.atom_texts.len();
        let mut adjacency = vec![Vec::new(); atom_count];
        let mut bonds = Vec::with_capacity(fixture.bonds.len());
        let mut seen = BTreeSet::new();
        for (bond_id, fixture_bond) in fixture.bonds.iter().enumerate() {
            assert!(fixture_bond.a < atom_count && fixture_bond.b < atom_count);
            assert_ne!(fixture_bond.a, fixture_bond.b);
            let endpoints = if fixture_bond.a < fixture_bond.b {
                (fixture_bond.a, fixture_bond.b)
            } else {
                (fixture_bond.b, fixture_bond.a)
            };
            assert!(
                seen.insert(endpoints),
                "primitive fixture must be a simple graph"
            );
            bonds.push(OracleBond {
                a: fixture_bond.a,
                b: fixture_bond.b,
                token: fixture_bond.token,
            });
            adjacency[fixture_bond.a].push(OracleIncidence {
                atom: fixture_bond.b,
                bond: bond_id,
            });
            adjacency[fixture_bond.b].push(OracleIncidence {
                atom: fixture_bond.a,
                bond: bond_id,
            });
        }
        for incidences in &mut adjacency {
            incidences.sort_unstable();
        }

        let mut component_by_atom = vec![None; atom_count];
        let mut components = Vec::new();
        for root in 0..atom_count {
            if component_by_atom[root].is_some() {
                continue;
            }
            let component_id = components.len();
            let mut atoms = Vec::new();
            let mut queue = VecDeque::from([root]);
            component_by_atom[root] = Some(component_id);
            while let Some(atom) = queue.pop_front() {
                atoms.push(atom);
                for incidence in &adjacency[atom] {
                    if component_by_atom[incidence.atom].is_none() {
                        component_by_atom[incidence.atom] = Some(component_id);
                        queue.push_back(incidence.atom);
                    }
                }
            }
            atoms.sort_unstable();
            components.push(OracleComponent {
                atoms,
                bonds: Vec::new(),
            });
        }
        for (bond_id, bond) in bonds.iter().enumerate() {
            let component = component_by_atom[bond.a].unwrap();
            assert_eq!(component_by_atom[bond.b], Some(component));
            components[component].bonds.push(bond_id);
        }
        Self {
            atom_texts: fixture.atom_texts.clone(),
            bonds,
            adjacency,
            components,
        }
    }

    fn initial_state(&self) -> OracleState {
        let mut state = OracleState {
            visited: vec![false; self.atom_texts.len()],
            bond_progress: vec![OracleBondProgress::Unrepresented; self.bonds.len()],
            active_atom: None,
            branch_returns: Vec::new(),
            plan_domains: self
                .bonds
                .iter()
                .map(|bond| bond.token.initial_plan_domain())
                .collect(),
            labels_by_bond: BTreeMap::new(),
            pending: None,
        };
        assert!(self.project_exact(&mut state.plan_domains));
        state
    }

    fn project_exact(&self, domains: &mut [OraclePlanDomain]) -> bool {
        self.project_exact_with_count(domains).0
    }

    fn project_exact_with_count(&self, domains: &mut [OraclePlanDomain]) -> (bool, usize) {
        assert_eq!(domains.len(), self.bonds.len());
        let mut enumerated_assignment_count = 0_usize;
        for component in &self.components {
            let mut projected = vec![OraclePlanDomain::empty(); component.bonds.len()];
            let mut assignment = vec![OracleBondPlan::Traversal; component.bonds.len()];
            let mut survivor_count = 0_usize;
            self.enumerate_component_assignments(
                component,
                domains,
                0,
                &mut assignment,
                &mut projected,
                &mut survivor_count,
                &mut enumerated_assignment_count,
            );
            if survivor_count == 0 {
                return (false, enumerated_assignment_count);
            }
            for (offset, &bond) in component.bonds.iter().enumerate() {
                domains[bond] = projected[offset];
            }
        }
        (true, enumerated_assignment_count)
    }

    fn enumerate_component_assignments(
        &self,
        component: &OracleComponent,
        domains: &[OraclePlanDomain],
        offset: usize,
        assignment: &mut [OracleBondPlan],
        projected: &mut [OraclePlanDomain],
        survivor_count: &mut usize,
        enumerated_assignment_count: &mut usize,
    ) {
        if offset == component.bonds.len() {
            *enumerated_assignment_count += 1;
            if !self.assignment_is_spanning_tree(component, assignment) {
                return;
            }
            *survivor_count += 1;
            for (projected_domain, plan) in projected.iter_mut().zip(assignment.iter().copied()) {
                *projected_domain = projected_domain.union(OraclePlanDomain::singleton(plan));
            }
            return;
        }
        let bond = component.bonds[offset];
        for plan in domains[bond].plans() {
            assignment[offset] = plan;
            self.enumerate_component_assignments(
                component,
                domains,
                offset + 1,
                assignment,
                projected,
                survivor_count,
                enumerated_assignment_count,
            );
        }
    }

    fn assignment_is_spanning_tree(
        &self,
        component: &OracleComponent,
        assignment: &[OracleBondPlan],
    ) -> bool {
        let traversal_count = assignment
            .iter()
            .filter(|plan| plan.role() == OracleBondRole::Traversal)
            .count();
        if traversal_count + 1 != component.atoms.len() {
            return false;
        }
        if component.atoms.len() == 1 {
            return true;
        }
        let mut reached = BTreeSet::from([component.atoms[0]]);
        let mut queue = VecDeque::from([component.atoms[0]]);
        while let Some(atom) = queue.pop_front() {
            for (offset, &bond_id) in component.bonds.iter().enumerate() {
                if assignment[offset].role() != OracleBondRole::Traversal {
                    continue;
                }
                let bond = self.bonds[bond_id];
                if bond.a != atom && bond.b != atom {
                    continue;
                }
                let other = bond.other(atom);
                if reached.insert(other) {
                    queue.push_back(other);
                }
            }
        }
        reached.len() == component.atoms.len()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum FixedEndpoint {
    A,
    B,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum EndpointSpelling {
    Omit,
    Emit,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum OracleBondRole {
    Traversal,
    Ring,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum OracleBondPlan {
    Traversal = 0,
    Ring00 = 1,
    Ring10 = 2,
    Ring01 = 3,
    Ring11 = 4,
}

impl OracleBondPlan {
    const ALL: [Self; 5] = [
        Self::Traversal,
        Self::Ring00,
        Self::Ring10,
        Self::Ring01,
        Self::Ring11,
    ];

    const fn bit(self) -> u8 {
        1_u8 << self as u8
    }

    const fn role(self) -> OracleBondRole {
        match self {
            Self::Traversal => OracleBondRole::Traversal,
            Self::Ring00 | Self::Ring10 | Self::Ring01 | Self::Ring11 => OracleBondRole::Ring,
        }
    }

    const fn emits_at(self, endpoint: FixedEndpoint) -> bool {
        match (self, endpoint) {
            (Self::Ring10 | Self::Ring11, FixedEndpoint::A)
            | (Self::Ring01 | Self::Ring11, FixedEndpoint::B) => true,
            (Self::Traversal | Self::Ring00 | Self::Ring01, FixedEndpoint::A)
            | (Self::Traversal | Self::Ring00 | Self::Ring10, FixedEndpoint::B) => false,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OraclePlanDomain(u8);

impl OraclePlanDomain {
    const fn empty() -> Self {
        Self(0)
    }

    const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    const fn singleton(plan: OracleBondPlan) -> Self {
        Self(plan.bit())
    }

    const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    const fn is_empty(self) -> bool {
        self.0 == 0
    }

    const fn is_singleton(self) -> bool {
        self.0 != 0 && self.0.count_ones() == 1
    }

    fn plans(self) -> impl Iterator<Item = OracleBondPlan> {
        OracleBondPlan::ALL
            .into_iter()
            .filter(move |plan| self.0 & plan.bit() != 0)
    }

    fn contains_role(self, role: OracleBondRole) -> bool {
        self.plans().any(|plan| plan.role() == role)
    }

    fn endpoint_projection(endpoint: FixedEndpoint, spelling: EndpointSpelling) -> Self {
        let emitted = spelling == EndpointSpelling::Emit;
        OracleBondPlan::ALL
            .into_iter()
            .filter(|plan| {
                plan.role() == OracleBondRole::Ring && plan.emits_at(endpoint) == emitted
            })
            .fold(Self::empty(), |domain, plan| {
                domain.union(Self::singleton(plan))
            })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum OracleBondProgress {
    Unrepresented,
    Traversed {
        from: usize,
        to: usize,
    },
    RingOpen {
        first_endpoint: usize,
    },
    RingClosed {
        first_endpoint: usize,
        second_endpoint: usize,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum OraclePending {
    ComponentAtom {
        root: usize,
    },
    BranchBondOrAtom {
        parent: usize,
        child: usize,
        bond: usize,
    },
    BranchAtom {
        parent: usize,
        child: usize,
        bond: usize,
    },
    InlineAtom {
        parent: usize,
        child: usize,
        bond: usize,
    },
    RingOpeningLabel {
        bond: usize,
        endpoint: usize,
        label: usize,
    },
    RingClosureLabel {
        bond: usize,
        endpoint: usize,
        label: usize,
    },
}

#[derive(Copy, Clone, Debug)]
enum OracleAction {
    Root(usize),
    RingOpen(OracleIncidence, EndpointSpelling),
    RingClose(OracleIncidence, EndpointSpelling),
    BranchChild(OracleIncidence),
    InlineChild(OracleIncidence),
    CloseBranch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OracleState {
    visited: Vec<bool>,
    bond_progress: Vec<OracleBondProgress>,
    active_atom: Option<usize>,
    branch_returns: Vec<usize>,
    plan_domains: Vec<OraclePlanDomain>,
    labels_by_bond: BTreeMap<usize, usize>,
    pending: Option<OraclePending>,
}

#[derive(Clone, Debug)]
struct OracleChoice {
    text: String,
    successor: OracleState,
}

enum OracleToken {
    Prepared(String),
    RingLabel(usize),
}

impl OracleToken {
    fn render(self) -> Option<String> {
        match self {
            Self::Prepared(text) => Some(text),
            Self::RingLabel(label) => ring_label_text(label),
        }
    }
}

#[derive(Default, Debug)]
struct OracleMetrics {
    exact_assignments_enumerated: usize,
    maximum_exact_assignments_per_projection: usize,
    maximum_forced_chain_depth: usize,
}

struct OracleEngine<'a> {
    surface: &'a OracleSurface,
    metrics: OracleMetrics,
}

impl<'a> OracleEngine<'a> {
    fn new(surface: &'a OracleSurface) -> Self {
        Self {
            surface,
            metrics: OracleMetrics::default(),
        }
    }

    fn choices(&mut self, state: &OracleState) -> Vec<OracleChoice> {
        self.validate_local_state(state)
            .unwrap_or_else(|failure| panic!("invalid oracle source state: {failure}\n{state:#?}"));
        if oracle_is_accepted(state) {
            return Vec::new();
        }
        if state.pending.is_some() {
            return self
                .apply_pending_once(state)
                .and_then(|(token, successor)| self.publish_token(token, successor))
                .into_iter()
                .collect();
        }
        self.action_universe(state)
            .into_iter()
            .filter_map(|action| self.attempt_action(state, action))
            .collect()
    }

    fn action_universe(&self, state: &OracleState) -> Vec<OracleAction> {
        let mut actions = (0..self.surface.atom_texts.len())
            .map(OracleAction::Root)
            .collect::<Vec<_>>();
        if let Some(active) = state.active_atom {
            for &incidence in &self.surface.adjacency[active] {
                for spelling in [EndpointSpelling::Omit, EndpointSpelling::Emit] {
                    actions.push(OracleAction::RingOpen(incidence, spelling));
                    actions.push(OracleAction::RingClose(incidence, spelling));
                }
                actions.push(OracleAction::BranchChild(incidence));
                actions.push(OracleAction::InlineChild(incidence));
            }
            actions.push(OracleAction::CloseBranch);
        }
        actions
    }

    fn attempt_action(
        &mut self,
        state: &OracleState,
        action: OracleAction,
    ) -> Option<OracleChoice> {
        if state.pending.is_some() {
            return None;
        }
        match action {
            OracleAction::Root(root) => (state.active_atom.is_none()
                && state.branch_returns.is_empty()
                && !state.visited[root])
                .then(|| self.attempt_root(state, root))
                .flatten(),
            OracleAction::RingOpen(incidence, spelling) => {
                let active = state.active_atom?;
                let attachments = self.surface.attachments(state, active);
                let closures = self.local_closures(state, active);
                let ring_phase = !closures.is_empty()
                    || attachments.iter().any(|attachment| attachment.len() > 1);
                let attachment = attachments
                    .iter()
                    .find(|attachment| attachment.contains(&incidence))?;
                (ring_phase
                    && attachment.len() > 1
                    && self.opening_is_structurally_eligible(state, attachment, incidence))
                .then(|| self.attempt_ring_opening(state, incidence, spelling))
                .flatten()
            }
            OracleAction::RingClose(incidence, spelling) => {
                let active = state.active_atom?;
                let attachments = self.surface.attachments(state, active);
                let closures = self.local_closures(state, active);
                let ring_phase = !closures.is_empty()
                    || attachments.iter().any(|attachment| attachment.len() > 1);
                (ring_phase && closures.contains(&incidence))
                    .then(|| self.attempt_ring_closure(state, incidence, spelling))
                    .flatten()
            }
            OracleAction::BranchChild(incidence) => {
                let active = state.active_atom?;
                let attachments = self.surface.attachments(state, active);
                let ring_phase = !self.local_closures(state, active).is_empty()
                    || attachments.iter().any(|attachment| attachment.len() > 1);
                (!ring_phase
                    && attachments.len() >= 2
                    && attachments
                        .iter()
                        .any(|attachment| attachment.as_slice() == [incidence]))
                .then(|| self.attempt_branch_child(state, incidence))
                .flatten()
            }
            OracleAction::InlineChild(incidence) => {
                let active = state.active_atom?;
                let attachments = self.surface.attachments(state, active);
                let ring_phase = !self.local_closures(state, active).is_empty()
                    || attachments.iter().any(|attachment| attachment.len() > 1);
                (!ring_phase && attachments.len() == 1 && attachments[0].as_slice() == [incidence])
                    .then(|| self.attempt_inline_child(state, incidence))
                    .flatten()
            }
            OracleAction::CloseBranch => {
                let active = state.active_atom?;
                let attachments = self.surface.attachments(state, active);
                let ring_phase = !self.local_closures(state, active).is_empty()
                    || attachments.iter().any(|attachment| attachment.len() > 1);
                (!ring_phase && attachments.is_empty() && !state.branch_returns.is_empty())
                    .then(|| self.attempt_close_branch(state))
                    .flatten()
            }
        }
    }

    fn attempt_root(&mut self, source: &OracleState, root: usize) -> Option<OracleChoice> {
        assert!(source.active_atom.is_none() && source.branch_returns.is_empty());
        assert!(!source.visited[root]);
        let later_component = source.visited.iter().any(|visited| *visited);
        let mut successor = source.clone();
        successor.visited[root] = true;
        successor.active_atom = Some(root);
        let text = if later_component {
            successor.pending = Some(OraclePending::ComponentAtom { root });
            ".".to_owned()
        } else {
            self.surface.atom_texts[root].clone()
        };
        self.publish(text, successor)
    }

    fn opening_is_structurally_eligible(
        &self,
        state: &OracleState,
        attachment: &[OracleIncidence],
        candidate: OracleIncidence,
    ) -> bool {
        state.plan_domains[candidate.bond].contains_role(OracleBondRole::Ring)
            && attachment.iter().copied().any(|incidence| {
                incidence != candidate
                    && state.plan_domains[incidence.bond].contains_role(OracleBondRole::Traversal)
            })
    }

    fn attempt_ring_opening(
        &mut self,
        source: &OracleState,
        incidence: OracleIncidence,
        spelling: EndpointSpelling,
    ) -> Option<OracleChoice> {
        let active = source.active_atom.unwrap();
        let bond = self.surface.bonds[incidence.bond];
        let endpoint = bond.fixed_endpoint(active);
        let allowed = OraclePlanDomain::endpoint_projection(endpoint, spelling);
        let mut successor = self.restrict_and_project(source, incidence.bond, allowed)?;
        assert_eq!(
            successor.bond_progress[incidence.bond],
            OracleBondProgress::Unrepresented
        );
        successor.bond_progress[incidence.bond] = OracleBondProgress::RingOpen {
            first_endpoint: active,
        };
        let label = smallest_free_label(&successor.labels_by_bond);
        assert_eq!(successor.labels_by_bond.insert(incidence.bond, label), None);
        match spelling {
            EndpointSpelling::Omit => self.publish_token(OracleToken::RingLabel(label), successor),
            EndpointSpelling::Emit => {
                let text = bond.token.text_from(bond, active);
                assert!(!text.is_empty());
                successor.pending = Some(OraclePending::RingOpeningLabel {
                    bond: incidence.bond,
                    endpoint: active,
                    label,
                });
                self.publish(text.to_owned(), successor)
            }
        }
    }

    fn attempt_ring_closure(
        &mut self,
        source: &OracleState,
        incidence: OracleIncidence,
        spelling: EndpointSpelling,
    ) -> Option<OracleChoice> {
        let active = source.active_atom.unwrap();
        let bond = self.surface.bonds[incidence.bond];
        let OracleBondProgress::RingOpen { first_endpoint } = source.bond_progress[incidence.bond]
        else {
            return None;
        };
        assert_eq!(incidence.atom, first_endpoint);
        let endpoint = bond.fixed_endpoint(active);
        let allowed = OraclePlanDomain::endpoint_projection(endpoint, spelling);
        let mut successor = self.restrict_and_project(source, incidence.bond, allowed)?;
        if !successor.plan_domains[incidence.bond].is_singleton() {
            return None;
        }
        successor.bond_progress[incidence.bond] = OracleBondProgress::RingClosed {
            first_endpoint,
            second_endpoint: active,
        };
        let label = *successor.labels_by_bond.get(&incidence.bond)?;
        match spelling {
            EndpointSpelling::Omit => {
                successor.labels_by_bond.remove(&incidence.bond);
                self.publish_token(OracleToken::RingLabel(label), successor)
            }
            EndpointSpelling::Emit => {
                let text = bond.token.text_from(bond, active);
                assert!(!text.is_empty());
                successor.pending = Some(OraclePending::RingClosureLabel {
                    bond: incidence.bond,
                    endpoint: active,
                    label,
                });
                self.publish(text.to_owned(), successor)
            }
        }
    }

    fn attempt_inline_child(
        &mut self,
        source: &OracleState,
        incidence: OracleIncidence,
    ) -> Option<OracleChoice> {
        let parent = source.active_atom.unwrap();
        let mut successor = self.restrict_and_project(
            source,
            incidence.bond,
            OraclePlanDomain::singleton(OracleBondPlan::Traversal),
        )?;
        let text = self.surface.bonds[incidence.bond]
            .token
            .text_from(self.surface.bonds[incidence.bond], parent);
        if text.is_empty() {
            self.enter_child(&mut successor, incidence, false);
            self.publish(self.surface.atom_texts[incidence.atom].clone(), successor)
        } else {
            successor.pending = Some(OraclePending::InlineAtom {
                parent,
                child: incidence.atom,
                bond: incidence.bond,
            });
            self.publish(text.to_owned(), successor)
        }
    }

    fn attempt_branch_child(
        &mut self,
        source: &OracleState,
        incidence: OracleIncidence,
    ) -> Option<OracleChoice> {
        let parent = source.active_atom.unwrap();
        let mut successor = self.restrict_and_project(
            source,
            incidence.bond,
            OraclePlanDomain::singleton(OracleBondPlan::Traversal),
        )?;
        successor.pending = Some(OraclePending::BranchBondOrAtom {
            parent,
            child: incidence.atom,
            bond: incidence.bond,
        });
        self.publish("(".to_owned(), successor)
    }

    fn attempt_close_branch(&mut self, source: &OracleState) -> Option<OracleChoice> {
        let mut successor = source.clone();
        successor.active_atom = successor.branch_returns.pop();
        self.publish(")".to_owned(), successor)
    }

    fn apply_pending_once(&mut self, source: &OracleState) -> Option<(OracleToken, OracleState)> {
        let pending = source.pending?;
        let mut successor = source.clone();
        successor.pending = None;
        match pending {
            OraclePending::ComponentAtom { root } => {
                assert_eq!(source.active_atom, Some(root));
                Some((
                    OracleToken::Prepared(self.surface.atom_texts[root].clone()),
                    successor,
                ))
            }
            OraclePending::InlineAtom {
                parent,
                child,
                bond,
            } => {
                assert_eq!(source.active_atom, Some(parent));
                self.enter_child(&mut successor, OracleIncidence { atom: child, bond }, false);
                Some((
                    OracleToken::Prepared(self.surface.atom_texts[child].clone()),
                    successor,
                ))
            }
            OraclePending::BranchBondOrAtom {
                parent,
                child,
                bond,
            } => {
                assert_eq!(source.active_atom, Some(parent));
                let oracle_bond = self.surface.bonds[bond];
                let text = oracle_bond.token.text_from(oracle_bond, parent);
                if text.is_empty() {
                    self.enter_child(&mut successor, OracleIncidence { atom: child, bond }, true);
                    Some((
                        OracleToken::Prepared(self.surface.atom_texts[child].clone()),
                        successor,
                    ))
                } else {
                    successor.pending = Some(OraclePending::BranchAtom {
                        parent,
                        child,
                        bond,
                    });
                    Some((OracleToken::Prepared(text.to_owned()), successor))
                }
            }
            OraclePending::BranchAtom {
                parent,
                child,
                bond,
            } => {
                assert_eq!(source.active_atom, Some(parent));
                self.enter_child(&mut successor, OracleIncidence { atom: child, bond }, true);
                Some((
                    OracleToken::Prepared(self.surface.atom_texts[child].clone()),
                    successor,
                ))
            }
            OraclePending::RingOpeningLabel {
                bond,
                endpoint,
                label,
            } => {
                assert_eq!(source.active_atom, Some(endpoint));
                assert_eq!(source.labels_by_bond.get(&bond), Some(&label));
                Some((OracleToken::RingLabel(label), successor))
            }
            OraclePending::RingClosureLabel {
                bond,
                endpoint,
                label,
            } => {
                assert_eq!(source.active_atom, Some(endpoint));
                assert_eq!(successor.labels_by_bond.remove(&bond), Some(label));
                Some((OracleToken::RingLabel(label), successor))
            }
        }
    }

    fn enter_child(&self, state: &mut OracleState, incidence: OracleIncidence, branch: bool) {
        let parent = state.active_atom.unwrap();
        assert!(!state.visited[incidence.atom]);
        assert_eq!(
            state.bond_progress[incidence.bond],
            OracleBondProgress::Unrepresented
        );
        if branch {
            state.branch_returns.push(parent);
        }
        state.bond_progress[incidence.bond] = OracleBondProgress::Traversed {
            from: parent,
            to: incidence.atom,
        };
        state.visited[incidence.atom] = true;
        state.active_atom = Some(incidence.atom);
    }

    fn restrict_and_project(
        &mut self,
        source: &OracleState,
        bond: usize,
        allowed: OraclePlanDomain,
    ) -> Option<OracleState> {
        let mut successor = source.clone();
        successor.plan_domains[bond] = successor.plan_domains[bond].intersect(allowed);
        if successor.plan_domains[bond].is_empty() {
            return None;
        }
        self.project_exact_counted(&mut successor.plan_domains)
            .then_some(successor)
    }

    fn project_exact_counted(&mut self, domains: &mut [OraclePlanDomain]) -> bool {
        let (result, assignment_count) = self.surface.project_exact_with_count(domains);
        self.metrics.exact_assignments_enumerated += assignment_count;
        self.metrics.maximum_exact_assignments_per_projection = self
            .metrics
            .maximum_exact_assignments_per_projection
            .max(assignment_count);
        result
    }

    fn publish(&mut self, text: String, mut successor: OracleState) -> Option<OracleChoice> {
        successor = self.prepare_successor(successor)?;
        Some(OracleChoice { text, successor })
    }

    fn publish_token(
        &mut self,
        token: OracleToken,
        successor: OracleState,
    ) -> Option<OracleChoice> {
        let successor = self.prepare_successor(successor)?;
        let text = token.render()?;
        Some(OracleChoice { text, successor })
    }

    fn prepare_successor(&mut self, mut successor: OracleState) -> Option<OracleState> {
        if successor.pending.is_none() {
            self.normalize_silently(&mut successor);
        }
        self.validate_local_state(&successor)
            .unwrap_or_else(|failure| {
                panic!("oracle constructed an invalid successor: {failure}\n{successor:#?}")
            });
        if successor.pending.is_some() && !self.validate_forced_pending_chain(&successor) {
            return None;
        }
        Some(successor)
    }

    fn normalize_silently(&self, state: &mut OracleState) {
        loop {
            if state.pending.is_some() {
                return;
            }
            let Some(active) = state.active_atom else {
                return;
            };
            if !self.surface.attachments(state, active).is_empty()
                || !state.branch_returns.is_empty()
                || state
                    .bond_progress
                    .iter()
                    .any(|progress| matches!(progress, OracleBondProgress::RingOpen { .. }))
            {
                return;
            }
            state.active_atom = None;
        }
    }

    fn validate_forced_pending_chain(&mut self, source: &OracleState) -> bool {
        let mut temporary = source.clone();
        let mut depth = 0_usize;
        while temporary.pending.is_some() {
            depth += 1;
            assert!(
                depth <= 2,
                "forced lexical chain exceeded the current grammar bound"
            );
            let (token, mut successor) = self
                .apply_pending_once(&temporary)
                .expect("a pending oracle state must have one forced transition");
            if successor.pending.is_none() {
                self.normalize_silently(&mut successor);
            }
            self.validate_local_state(&successor).unwrap_or_else(|failure| {
                panic!(
                    "oracle pending transition constructed an invalid successor: {failure}\n{successor:#?}"
                )
            });
            if token.render().is_none() {
                return false;
            }
            temporary = successor;
        }
        self.metrics.maximum_forced_chain_depth =
            self.metrics.maximum_forced_chain_depth.max(depth);
        true
    }

    fn local_closures(&self, state: &OracleState, active: usize) -> Vec<OracleIncidence> {
        self.surface.adjacency[active]
            .iter()
            .copied()
            .filter(|incidence| {
                matches!(
                    state.bond_progress[incidence.bond],
                    OracleBondProgress::RingOpen { first_endpoint }
                        if first_endpoint == incidence.atom
                )
            })
            .collect()
    }
}

fn smallest_free_label(labels_by_bond: &BTreeMap<usize, usize>) -> usize {
    (1..)
        .find(|label| labels_by_bond.values().all(|occupied| occupied != label))
        .unwrap()
}

fn ring_label_text(label: usize) -> Option<String> {
    match label {
        1..=9 => Some(label.to_string()),
        10..=99 => Some(format!("%{label}")),
        _ => None,
    }
}

fn oracle_is_accepted(state: &OracleState) -> bool {
    state.pending.is_none()
        && state.labels_by_bond.is_empty()
        && state.active_atom.is_none()
        && state.visited.iter().all(|visited| *visited)
        && state.bond_progress.iter().all(|progress| {
            matches!(
                progress,
                OracleBondProgress::Traversed { .. } | OracleBondProgress::RingClosed { .. }
            )
        })
}

impl OracleSurface {
    fn attachments(&self, state: &OracleState, frame_atom: usize) -> Vec<Vec<OracleIncidence>> {
        assert!(state.visited[frame_atom]);
        let mut residual_component = vec![None; self.atom_texts.len()];
        let mut next_component = 0_usize;
        for root in 0..self.atom_texts.len() {
            if state.visited[root] || residual_component[root].is_some() {
                continue;
            }
            residual_component[root] = Some(next_component);
            let mut queue = VecDeque::from([root]);
            while let Some(atom) = queue.pop_front() {
                for incidence in &self.adjacency[atom] {
                    if state.visited[incidence.atom] || residual_component[incidence.atom].is_some()
                    {
                        continue;
                    }
                    residual_component[incidence.atom] = Some(next_component);
                    queue.push_back(incidence.atom);
                }
            }
            next_component += 1;
        }

        let mut grouped = BTreeMap::<usize, Vec<OracleIncidence>>::new();
        for &incidence in &self.adjacency[frame_atom] {
            if state.visited[incidence.atom]
                || state.bond_progress[incidence.bond] != OracleBondProgress::Unrepresented
            {
                continue;
            }
            let component = residual_component[incidence.atom]
                .expect("an unvisited endpoint must belong to a residual component");
            grouped.entry(component).or_default().push(incidence);
        }
        let mut groups = grouped.into_values().collect::<Vec<_>>();
        normalize_attachment_groups(&mut groups);
        groups
    }
}

impl OracleEngine<'_> {
    fn validate_local_state(&mut self, state: &OracleState) -> Result<(), String> {
        if state.visited.len() != self.surface.atom_texts.len()
            || state.bond_progress.len() != self.surface.bonds.len()
            || state.plan_domains.len() != self.surface.bonds.len()
        {
            return Err("oracle state shape does not match its primitive surface".to_owned());
        }
        let mut exact = state.plan_domains.clone();
        if !self.project_exact_counted(&mut exact) {
            return Err("current plan domains have no spanning-tree assignment".to_owned());
        }
        if exact != state.plan_domains {
            return Err(format!(
                "plan domains are not exact projections: current={:?}, exact={exact:?}",
                state.plan_domains
            ));
        }

        let mut incoming_traversal = vec![0_usize; state.visited.len()];
        for (bond_id, progress) in state.bond_progress.iter().copied().enumerate() {
            let bond = self.surface.bonds[bond_id];
            let domain = state.plan_domains[bond_id];
            if domain.is_empty() {
                return Err(format!("bond {bond_id} has an empty plan domain"));
            }
            match progress {
                OracleBondProgress::Unrepresented => {
                    if state.visited[bond.a] && state.visited[bond.b] {
                        return Err(format!(
                            "unrepresented bond {bond_id} has two visited endpoints"
                        ));
                    }
                }
                OracleBondProgress::Traversed { from, to } => {
                    if !state.visited[from]
                        || !state.visited[to]
                        || bond.other(from) != to
                        || domain != OraclePlanDomain::singleton(OracleBondPlan::Traversal)
                    {
                        return Err(format!("invalid traversed bond {bond_id}"));
                    }
                    incoming_traversal[to] += 1;
                }
                OracleBondProgress::RingOpen { first_endpoint } => {
                    if !state.visited[first_endpoint]
                        || (first_endpoint != bond.a && first_endpoint != bond.b)
                        || !domain.contains_role(OracleBondRole::Ring)
                        || domain.contains_role(OracleBondRole::Traversal)
                    {
                        return Err(format!("invalid open ring bond {bond_id}"));
                    }
                    let other = bond.other(first_endpoint);
                    let other_endpoint = bond.fixed_endpoint(other);
                    let closure_support = domain.intersect(
                        OraclePlanDomain::endpoint_projection(
                            other_endpoint,
                            EndpointSpelling::Omit,
                        )
                        .union(OraclePlanDomain::endpoint_projection(
                            other_endpoint,
                            EndpointSpelling::Emit,
                        )),
                    );
                    if closure_support.is_empty() {
                        return Err(format!("open ring {bond_id} has no closure spelling"));
                    }
                }
                OracleBondProgress::RingClosed {
                    first_endpoint,
                    second_endpoint,
                } => {
                    if !state.visited[first_endpoint]
                        || !state.visited[second_endpoint]
                        || bond.other(first_endpoint) != second_endpoint
                        || !domain.is_singleton()
                        || !domain.contains_role(OracleBondRole::Ring)
                    {
                        return Err(format!("invalid closed ring bond {bond_id}"));
                    }
                }
            }
        }

        for component in &self.surface.components {
            let visited = component
                .atoms
                .iter()
                .copied()
                .filter(|atom| state.visited[*atom])
                .collect::<Vec<_>>();
            if visited.is_empty() {
                continue;
            }
            let roots = visited
                .iter()
                .filter(|atom| incoming_traversal[**atom] == 0)
                .count();
            if roots != 1 || visited.iter().any(|atom| incoming_traversal[*atom] > 1) {
                return Err(format!(
                    "begun component {:?} does not have one traversal root",
                    component.atoms
                ));
            }
        }

        if let Some(active) = state.active_atom {
            if !state.visited[active] {
                return Err("active atom is not visited".to_owned());
            }
        }
        if state
            .branch_returns
            .iter()
            .any(|atom| !state.visited[*atom])
        {
            return Err("suspended parent is not visited".to_owned());
        }

        let labels = state
            .labels_by_bond
            .values()
            .copied()
            .collect::<BTreeSet<_>>();
        if labels.len() != state.labels_by_bond.len() {
            return Err("ring labels are not unique".to_owned());
        }
        for (bond_id, progress) in state.bond_progress.iter().copied().enumerate() {
            let owns_label = state.labels_by_bond.contains_key(&bond_id);
            let pending_closed_label = matches!(
                state.pending,
                Some(OraclePending::RingClosureLabel { bond, .. }) if bond == bond_id
            );
            match progress {
                OracleBondProgress::RingOpen { .. } if !owns_label => {
                    return Err(format!("open ring {bond_id} lacks a label"));
                }
                OracleBondProgress::RingClosed { .. } if owns_label != pending_closed_label => {
                    return Err(format!(
                        "closed ring {bond_id} has inconsistent label ownership"
                    ));
                }
                OracleBondProgress::Unrepresented | OracleBondProgress::Traversed { .. }
                    if owns_label =>
                {
                    return Err(format!("non-ring-progress bond {bond_id} owns a label"));
                }
                _ => {}
            }
        }

        self.validate_pending(state)?;
        if state.pending.is_none() {
            self.validate_nonpending_progress(state)?;
        }
        Ok(())
    }

    fn validate_pending(&self, state: &OracleState) -> Result<(), String> {
        let Some(pending) = state.pending else {
            return Ok(());
        };
        match pending {
            OraclePending::ComponentAtom { root } => {
                if state.active_atom != Some(root) || !state.visited[root] {
                    return Err("pending component atom lacks its entered root".to_owned());
                }
            }
            OraclePending::BranchBondOrAtom {
                parent,
                child,
                bond,
            }
            | OraclePending::BranchAtom {
                parent,
                child,
                bond,
            }
            | OraclePending::InlineAtom {
                parent,
                child,
                bond,
            } => {
                let topology = self.surface.bonds[bond];
                if state.active_atom != Some(parent)
                    || state.visited[child]
                    || topology.other(parent) != child
                    || state.bond_progress[bond] != OracleBondProgress::Unrepresented
                    || state.plan_domains[bond]
                        != OraclePlanDomain::singleton(OracleBondPlan::Traversal)
                {
                    return Err("pending child commitment is inconsistent".to_owned());
                }
            }
            OraclePending::RingOpeningLabel {
                bond,
                endpoint,
                label,
            } => {
                let topology = self.surface.bonds[bond];
                if state.active_atom != Some(endpoint)
                    || state.labels_by_bond.get(&bond) != Some(&label)
                    || state.bond_progress[bond]
                        != (OracleBondProgress::RingOpen {
                            first_endpoint: endpoint,
                        })
                    || state.plan_domains[bond]
                        .plans()
                        .any(|plan| !plan.emits_at(topology.fixed_endpoint(endpoint)))
                {
                    return Err("pending opening label is inconsistent".to_owned());
                }
            }
            OraclePending::RingClosureLabel {
                bond,
                endpoint,
                label,
            } => {
                let topology = self.surface.bonds[bond];
                let OracleBondProgress::RingClosed {
                    second_endpoint, ..
                } = state.bond_progress[bond]
                else {
                    return Err("pending closure label lacks a closed ring".to_owned());
                };
                if state.active_atom != Some(endpoint)
                    || second_endpoint != endpoint
                    || state.labels_by_bond.get(&bond) != Some(&label)
                    || state.plan_domains[bond]
                        .plans()
                        .any(|plan| !plan.emits_at(topology.fixed_endpoint(endpoint)))
                {
                    return Err("pending closure label is inconsistent".to_owned());
                }
            }
        }
        Ok(())
    }

    fn validate_nonpending_progress(&self, state: &OracleState) -> Result<(), String> {
        let Some(active) = state.active_atom else {
            if !state.branch_returns.is_empty() {
                return Err("inactive state retains suspended parents".to_owned());
            }
            if state
                .bond_progress
                .iter()
                .any(|progress| matches!(progress, OracleBondProgress::RingOpen { .. }))
            {
                return Err("inactive state retains an open ring".to_owned());
            }
            return Ok(());
        };
        let attachments = self.surface.attachments(state, active);
        let closures = self.local_closures(state, active);
        let multi = attachments
            .iter()
            .filter(|group| group.len() > 1)
            .collect::<Vec<_>>();
        if !closures.is_empty() || !multi.is_empty() {
            for attachment in multi {
                if !attachment.iter().copied().any(|incidence| {
                    self.opening_is_structurally_eligible(state, attachment, incidence)
                }) {
                    return Err("multi-incidence attachment has no admissible opening".to_owned());
                }
            }
            return Ok(());
        }
        for attachment in &attachments {
            if attachment.len() != 1
                || !state.plan_domains[attachment[0].bond].contains_role(OracleBondRole::Traversal)
            {
                return Err("child attachment lacks one traversal-capable incidence".to_owned());
            }
        }
        if attachments.is_empty()
            && state.branch_returns.is_empty()
            && !state
                .bond_progress
                .iter()
                .any(|progress| matches!(progress, OracleBondProgress::RingOpen { .. }))
        {
            return Err("top-level component completion was not normalized".to_owned());
        }
        Ok(())
    }
}

fn normalize_attachment_groups(groups: &mut Vec<Vec<OracleIncidence>>) {
    for group in groups.iter_mut() {
        group.sort_unstable();
        assert!(group.windows(2).all(|pair| pair[0] != pair[1]));
    }
    groups.sort_unstable();
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct IncidenceSnapshot {
    atom: usize,
    bond: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FrameSnapshot {
    atom: usize,
    attachment_groups: Vec<Vec<IncidenceSnapshot>>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum BondProgressSnapshot {
    Unrepresented,
    Traversed {
        from: usize,
        to: usize,
    },
    RingOpen {
        first_endpoint: usize,
    },
    RingClosed {
        first_endpoint: usize,
        second_endpoint: usize,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum PendingSnapshot {
    ComponentAtom {
        root: usize,
    },
    BranchBondOrAtom {
        parent: usize,
        child: usize,
        bond: usize,
    },
    BranchAtom {
        parent: usize,
        child: usize,
        bond: usize,
    },
    InlineAtom {
        parent: usize,
        child: usize,
        bond: usize,
    },
    RingOpeningLabel {
        bond: usize,
        endpoint: usize,
        label: usize,
    },
    RingClosureLabel {
        bond: usize,
        endpoint: usize,
        label: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SemanticSnapshot {
    visited_atoms: Vec<usize>,
    bond_progress: Vec<BondProgressSnapshot>,
    active_frame: Option<FrameSnapshot>,
    branch_returns: Vec<FrameSnapshot>,
    plan_domains: Vec<u8>,
    labels_by_bond: Vec<(usize, usize)>,
    pending: Option<PendingSnapshot>,
    maximum_spelling_label: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ChoiceSnapshot {
    text: String,
    successor: SemanticSnapshot,
}

fn production_snapshot(observed: &ObservedNonStereoState) -> SemanticSnapshot {
    let traversal = &observed.structural.traversal;
    SemanticSnapshot {
        visited_atoms: traversal
            .visited_atoms
            .iter()
            .map(|atom| atom.index())
            .collect(),
        bond_progress: traversal
            .bond_progress
            .iter()
            .map(|progress| match progress {
                ObservedBondProgress::Unrepresented => BondProgressSnapshot::Unrepresented,
                ObservedBondProgress::Traversed { from, to } => BondProgressSnapshot::Traversed {
                    from: from.index(),
                    to: to.index(),
                },
                ObservedBondProgress::RingOpen { first_endpoint } => {
                    BondProgressSnapshot::RingOpen {
                        first_endpoint: first_endpoint.index(),
                    }
                }
                ObservedBondProgress::RingClosed {
                    first_endpoint,
                    second_endpoint,
                } => BondProgressSnapshot::RingClosed {
                    first_endpoint: first_endpoint.index(),
                    second_endpoint: second_endpoint.index(),
                },
            })
            .collect(),
        active_frame: traversal.active_frame.as_ref().map(observed_frame_snapshot),
        branch_returns: traversal
            .branch_returns
            .iter()
            .map(observed_frame_snapshot)
            .collect(),
        plan_domains: observed
            .structural
            .bond_plan_domains
            .iter()
            .copied()
            .map(production_domain_bits)
            .collect(),
        labels_by_bond: {
            let mut labels = observed
                .labels_by_bond
                .iter()
                .map(|(bond, slot)| (bond.index(), slot + 1))
                .collect::<Vec<_>>();
            labels.sort_unstable();
            labels
        },
        pending: observed.pending.as_ref().map(production_pending_snapshot),
        maximum_spelling_label: observed.maximum_spelling_label,
    }
}

fn observed_frame_snapshot(frame: &ObservedFrame) -> FrameSnapshot {
    let mut groups = frame
        .attachment_groups
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|incidence| IncidenceSnapshot {
                    atom: incidence.atom().index(),
                    bond: incidence.bond().index(),
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    normalize_snapshot_groups(&mut groups);
    FrameSnapshot {
        atom: frame.atom.index(),
        attachment_groups: groups,
    }
}

fn production_pending_snapshot(pending: &ObservedPending) -> PendingSnapshot {
    match pending {
        ObservedPending::ComponentAtom { root } => {
            PendingSnapshot::ComponentAtom { root: root.index() }
        }
        ObservedPending::BranchBondOrAtom {
            parent,
            child,
            bond,
        } => PendingSnapshot::BranchBondOrAtom {
            parent: parent.index(),
            child: child.index(),
            bond: bond.index(),
        },
        ObservedPending::BranchAtom {
            parent,
            child,
            bond,
        } => PendingSnapshot::BranchAtom {
            parent: parent.index(),
            child: child.index(),
            bond: bond.index(),
        },
        ObservedPending::InlineAtom {
            parent,
            child,
            bond,
        } => PendingSnapshot::InlineAtom {
            parent: parent.index(),
            child: child.index(),
            bond: bond.index(),
        },
        ObservedPending::RingOpeningLabel {
            bond,
            endpoint,
            label,
        } => PendingSnapshot::RingOpeningLabel {
            bond: bond.index(),
            endpoint: endpoint.index(),
            label: label + 1,
        },
        ObservedPending::RingClosureLabel {
            bond,
            endpoint,
            label,
        } => PendingSnapshot::RingClosureLabel {
            bond: bond.index(),
            endpoint: endpoint.index(),
            label: label + 1,
        },
    }
}

fn production_domain_bits(domain: Domain) -> u8 {
    [
        (BondRepresentation::Traversal, OracleBondPlan::Traversal),
        (BondRepresentation::Ring00, OracleBondPlan::Ring00),
        (BondRepresentation::Ring10, OracleBondPlan::Ring10),
        (BondRepresentation::Ring01, OracleBondPlan::Ring01),
        (BondRepresentation::Ring11, OracleBondPlan::Ring11),
    ]
    .into_iter()
    .filter(|(production, _)| domain.contains(production.value_index()))
    .fold(0_u8, |bits, (_, semantic)| bits | semantic.bit())
}

fn oracle_snapshot(surface: &OracleSurface, state: &OracleState) -> SemanticSnapshot {
    let frame = |atom: usize| FrameSnapshot {
        atom,
        attachment_groups: surface
            .attachments(state, atom)
            .into_iter()
            .map(|group| {
                group
                    .into_iter()
                    .map(|incidence| IncidenceSnapshot {
                        atom: incidence.atom,
                        bond: incidence.bond,
                    })
                    .collect()
            })
            .collect(),
    };
    SemanticSnapshot {
        visited_atoms: state
            .visited
            .iter()
            .enumerate()
            .filter_map(|(atom, visited)| visited.then_some(atom))
            .collect(),
        bond_progress: state
            .bond_progress
            .iter()
            .map(|progress| match progress {
                OracleBondProgress::Unrepresented => BondProgressSnapshot::Unrepresented,
                OracleBondProgress::Traversed { from, to } => BondProgressSnapshot::Traversed {
                    from: *from,
                    to: *to,
                },
                OracleBondProgress::RingOpen { first_endpoint } => BondProgressSnapshot::RingOpen {
                    first_endpoint: *first_endpoint,
                },
                OracleBondProgress::RingClosed {
                    first_endpoint,
                    second_endpoint,
                } => BondProgressSnapshot::RingClosed {
                    first_endpoint: *first_endpoint,
                    second_endpoint: *second_endpoint,
                },
            })
            .collect(),
        active_frame: state.active_atom.map(&frame),
        branch_returns: state
            .branch_returns
            .iter()
            .rev()
            .copied()
            .map(frame)
            .collect(),
        plan_domains: state.plan_domains.iter().map(|domain| domain.0).collect(),
        labels_by_bond: state
            .labels_by_bond
            .iter()
            .map(|(bond, label)| (*bond, *label))
            .collect(),
        pending: state.pending.map(oracle_pending_snapshot),
        maximum_spelling_label: 99,
    }
}

fn oracle_pending_snapshot(pending: OraclePending) -> PendingSnapshot {
    match pending {
        OraclePending::ComponentAtom { root } => PendingSnapshot::ComponentAtom { root },
        OraclePending::BranchBondOrAtom {
            parent,
            child,
            bond,
        } => PendingSnapshot::BranchBondOrAtom {
            parent,
            child,
            bond,
        },
        OraclePending::BranchAtom {
            parent,
            child,
            bond,
        } => PendingSnapshot::BranchAtom {
            parent,
            child,
            bond,
        },
        OraclePending::InlineAtom {
            parent,
            child,
            bond,
        } => PendingSnapshot::InlineAtom {
            parent,
            child,
            bond,
        },
        OraclePending::RingOpeningLabel {
            bond,
            endpoint,
            label,
        } => PendingSnapshot::RingOpeningLabel {
            bond,
            endpoint,
            label,
        },
        OraclePending::RingClosureLabel {
            bond,
            endpoint,
            label,
        } => PendingSnapshot::RingClosureLabel {
            bond,
            endpoint,
            label,
        },
    }
}

fn normalize_snapshot_groups(groups: &mut Vec<Vec<IncidenceSnapshot>>) {
    for group in groups.iter_mut() {
        group.sort_unstable();
        assert!(group.windows(2).all(|pair| pair[0] != pair[1]));
    }
    groups.sort_unstable();
}

#[derive(Clone, Debug)]
struct PairedState {
    production: ProductionState,
    oracle: OracleState,
    trace: Vec<String>,
}

#[derive(Copy, Clone, Debug)]
struct ConformanceBounds {
    state_instances: usize,
    choice_instances: usize,
    states: usize,
    transitions: usize,
    queue: usize,
    exact_assignments_per_projection: usize,
}

#[derive(Default, Clone, Debug)]
struct ConformanceMetrics {
    state_instances: usize,
    choice_instances: usize,
    unique_states: usize,
    transitions: usize,
    maximum_queue: usize,
    maximum_open_rings: usize,
    maximum_distinct_closure_bonds: usize,
    maximum_forced_chain_depth: usize,
    exact_assignments_enumerated: usize,
    maximum_exact_assignments_per_projection: usize,
    witnessed_label_owner_sets: BTreeSet<Vec<usize>>,
    witnessed_ring_opening_endpoints: BTreeSet<(usize, usize)>,
    witnessed_closed_plan_domains: BTreeSet<(usize, u8)>,
    label_one_owners: BTreeSet<usize>,
    label_one_owners_after_another_component: BTreeSet<usize>,
    witnessed_inactive_component_boundary: bool,
    witnessed_equal_text_distinct_successors: bool,
}

fn assert_fixture_conforms(
    name: &str,
    fixture: &FixtureSpec,
    bounds: ConformanceBounds,
) -> ConformanceMetrics {
    let production_surface = fixture.build_production_surface();
    let oracle_surface = fixture.build_oracle_surface();
    let production = match ProductionState::initial(&production_surface).unwrap() {
        Consistency::Consistent(state) => state,
        Consistency::Contradiction => panic!("{name}: production preparation contradicted"),
    };
    let oracle = oracle_surface.initial_state();
    let initial_production_snapshot = production_snapshot(&production.observe_raw());
    let initial_oracle_snapshot = oracle_snapshot(&oracle_surface, &oracle);
    assert_eq!(
        initial_production_snapshot, initial_oracle_snapshot,
        "{name}: independently prepared initial states differ"
    );

    let mut engine = OracleEngine::new(&oracle_surface);
    let mut queue = VecDeque::from([PairedState {
        production,
        oracle,
        trace: Vec::new(),
    }]);
    let mut cached_choices = BTreeMap::<SemanticSnapshot, Vec<ChoiceSnapshot>>::new();
    let mut metrics = ConformanceMetrics::default();

    while let Some(paired) = queue.pop_front() {
        metrics.state_instances += 1;
        assert!(
            metrics.state_instances <= bounds.state_instances,
            "{name}: exceeded {} reachable state instances",
            bounds.state_instances
        );
        metrics.maximum_queue = metrics.maximum_queue.max(queue.len() + 1);
        assert!(
            metrics.maximum_queue <= bounds.queue,
            "{name}: exceeded queue bound {} after trace {:?}",
            bounds.queue,
            paired.trace
        );
        let production_before_raw = paired.production.observe_raw();
        let production_before = production_snapshot(&production_before_raw);
        let oracle_before = oracle_snapshot(&oracle_surface, &paired.oracle);
        assert_eq!(
            production_before, oracle_before,
            "{name}: paired source mismatch after trace {:?}",
            paired.trace
        );
        assert_eq!(
            paired.production.is_accepted(),
            oracle_is_accepted(&paired.oracle),
            "{name}: acceptance mismatch after trace {:?}",
            paired.trace
        );
        assert_eq!(
            paired.production.is_accepted(),
            snapshot_is_accepted(&production_before, fixture.atom_texts.len()),
            "{name}: production acceptance disagrees with its raw observation"
        );

        let production_choices = match paired.production.choices() {
            Ok(choices) => choices,
            Err(failure) => panic!(
                "{name}: production choice failure {failure:?} after trace {:?}\nsource={production_before:#?}",
                paired.trace
            ),
        };
        assert_eq!(
            production_before_raw,
            paired.production.observe_raw(),
            "{name}: choices() mutated its source after trace {:?}",
            paired.trace
        );
        let oracle_source_before = paired.oracle.clone();
        let oracle_choices = engine.choices(&paired.oracle);
        assert_eq!(
            paired.oracle, oracle_source_before,
            "{name}: oracle choice generation mutated its source"
        );

        if production_choices.is_empty() {
            assert!(
                paired.production.is_accepted(),
                "{name}: nonaccepted production state returned Ok([])"
            );
        }
        assert_eq!(
            production_choices.is_empty(),
            oracle_choices.is_empty(),
            "{name}: terminal choice cardinality differs"
        );

        let mut production_by_choice = BTreeMap::new();
        for choice in production_choices {
            let snapshot = production_snapshot(&choice.successor().observe_raw());
            let key = ChoiceSnapshot {
                text: choice.text().to_owned(),
                successor: snapshot,
            };
            assert!(
                production_by_choice
                    .insert(key.clone(), choice.into_successor())
                    .is_none(),
                "{name}: duplicate identical production choice {key:#?}"
            );
        }
        let mut oracle_by_choice = BTreeMap::new();
        for choice in oracle_choices {
            let snapshot = oracle_snapshot(&oracle_surface, &choice.successor);
            let key = ChoiceSnapshot {
                text: choice.text,
                successor: snapshot,
            };
            assert!(
                oracle_by_choice
                    .insert(key.clone(), choice.successor)
                    .is_none(),
                "{name}: duplicate identical oracle choice {key:#?}"
            );
        }
        let production_keys = production_by_choice.keys().cloned().collect::<Vec<_>>();
        let oracle_keys = oracle_by_choice.keys().cloned().collect::<Vec<_>>();
        assert_eq!(
            production_keys, oracle_keys,
            "{name}: choice mismatch after trace {:?}\nsource={production_before:#?}",
            paired.trace
        );
        metrics.choice_instances += production_keys.len();
        assert!(
            metrics.choice_instances <= bounds.choice_instances,
            "{name}: exceeded {} checked choice instances",
            bounds.choice_instances
        );

        if let Some(previous) = cached_choices.get(&production_before) {
            assert_eq!(
                previous, &production_keys,
                "{name}: equal semantic snapshots exposed different production choices"
            );
        } else {
            cached_choices.insert(production_before.clone(), production_keys.clone());
            metrics.unique_states += 1;
            metrics.transitions += production_keys.len();
            assert!(
                metrics.unique_states <= bounds.states,
                "{name}: exceeded {} unique states after trace {:?}",
                bounds.states,
                paired.trace
            );
            assert!(
                metrics.transitions <= bounds.transitions,
                "{name}: exceeded {} unique transitions after trace {:?}",
                bounds.transitions,
                paired.trace
            );
        }
        metrics.maximum_open_rings = metrics
            .maximum_open_rings
            .max(paired.oracle.labels_by_bond.len());
        metrics.label_one_owners.extend(
            paired
                .oracle
                .labels_by_bond
                .iter()
                .filter_map(|(bond, label)| (*label == 1).then_some(*bond)),
        );
        for (&bond, &label) in &paired.oracle.labels_by_bond {
            if label != 1 {
                continue;
            }
            let bond_component = oracle_surface
                .components
                .iter()
                .position(|component| component.bonds.iter().any(|candidate| *candidate == bond));
            if oracle_surface
                .components
                .iter()
                .enumerate()
                .any(|(component_index, component)| {
                    Some(component_index) != bond_component
                        && component
                            .atoms
                            .iter()
                            .all(|atom| paired.oracle.visited[*atom])
                })
            {
                metrics
                    .label_one_owners_after_another_component
                    .insert(bond);
            }
        }
        metrics.witnessed_inactive_component_boundary |= paired.oracle.active_atom.is_none()
            && paired.oracle.pending.is_none()
            && paired.oracle.labels_by_bond.is_empty()
            && paired.oracle.visited.iter().any(|visited| *visited)
            && paired.oracle.visited.iter().any(|visited| !*visited);
        metrics
            .witnessed_label_owner_sets
            .insert(paired.oracle.labels_by_bond.keys().copied().collect());
        for (bond, progress) in paired.oracle.bond_progress.iter().enumerate() {
            match progress {
                OracleBondProgress::RingOpen { first_endpoint } => {
                    metrics
                        .witnessed_ring_opening_endpoints
                        .insert((bond, *first_endpoint));
                }
                OracleBondProgress::RingClosed { .. } => {
                    metrics
                        .witnessed_closed_plan_domains
                        .insert((bond, paired.oracle.plan_domains[bond].0));
                }
                OracleBondProgress::Unrepresented | OracleBondProgress::Traversed { .. } => {}
            }
        }
        let distinct_closures = production_keys
            .iter()
            .flat_map(|choice| {
                production_before
                    .bond_progress
                    .iter()
                    .zip(&choice.successor.bond_progress)
                    .enumerate()
                    .filter_map(|(bond, (before, after))| {
                        matches!(before, BondProgressSnapshot::RingOpen { .. })
                            .then_some(())
                            .filter(|_| matches!(after, BondProgressSnapshot::RingClosed { .. }))
                            .map(|_| bond)
                    })
            })
            .collect::<BTreeSet<_>>()
            .len();
        metrics.maximum_distinct_closure_bonds = metrics
            .maximum_distinct_closure_bonds
            .max(distinct_closures);
        metrics.witnessed_equal_text_distinct_successors |= production_keys
            .windows(2)
            .any(|pair| pair[0].text == pair[1].text && pair[0].successor != pair[1].successor);

        for key in production_keys {
            let production = production_by_choice.remove(&key).unwrap();
            let oracle = oracle_by_choice.remove(&key).unwrap();
            let mut trace = paired.trace.clone();
            trace.push(key.text.clone());
            queue.push_back(PairedState {
                production,
                oracle,
                trace,
            });
        }
    }

    metrics.maximum_forced_chain_depth = engine.metrics.maximum_forced_chain_depth;
    metrics.exact_assignments_enumerated = engine.metrics.exact_assignments_enumerated;
    metrics.maximum_exact_assignments_per_projection =
        engine.metrics.maximum_exact_assignments_per_projection;
    assert!(
        metrics.maximum_exact_assignments_per_projection <= bounds.exact_assignments_per_projection,
        "{name}: exceeded exact-projection assignment bound {}: {metrics:#?}",
        bounds.exact_assignments_per_projection
    );
    metrics
}

fn snapshot_is_accepted(snapshot: &SemanticSnapshot, atom_count: usize) -> bool {
    snapshot.pending.is_none()
        && snapshot.labels_by_bond.is_empty()
        && snapshot.active_frame.is_none()
        && snapshot.branch_returns.is_empty()
        && snapshot.visited_atoms.len() == atom_count
        && snapshot.bond_progress.iter().all(|progress| {
            matches!(
                progress,
                BondProgressSnapshot::Traversed { .. } | BondProgressSnapshot::RingClosed { .. }
            )
        })
}

fn simple_graph_fixture(atom_count: usize, mask: u64) -> FixtureSpec {
    const ATOM_TEXTS: [&str; 4] = ["A", "B", "C", "D"];
    let mut bonds = Vec::new();
    let mut bit = 0_u32;
    for a in 0..atom_count {
        for b in (a + 1)..atom_count {
            if mask & (1_u64 << bit) != 0 {
                bonds.push((a, b, FixtureBondToken::Elided));
            }
            bit += 1;
        }
    }
    FixtureSpec::new(&ATOM_TEXTS[..atom_count], &bonds)
}

fn simple_graph_count(atom_count: usize) -> u64 {
    1_u64 << (atom_count * atom_count.saturating_sub(1) / 2)
}

fn fixture_has_cycle(fixture: &FixtureSpec) -> bool {
    let mut parent = (0..fixture.atom_texts.len()).collect::<Vec<_>>();
    fn find(parent: &mut [usize], value: usize) -> usize {
        if parent[value] != value {
            parent[value] = find(parent, parent[value]);
        }
        parent[value]
    }
    for bond in &fixture.bonds {
        let left = find(&mut parent, bond.a);
        let right = find(&mut parent, bond.b);
        if left == right {
            return true;
        }
        parent[left] = right;
    }
    false
}

#[test]
fn every_labelled_simple_graph_through_three_atoms_matches_the_oracle() {
    let mut graph_count = 0_usize;
    for atom_count in 0..=3 {
        for mask in 0..simple_graph_count(atom_count) {
            let fixture = simple_graph_fixture(atom_count, mask);
            assert_fixture_conforms(
                &format!("n{atom_count}-mask-{mask:x}"),
                &fixture,
                ConformanceBounds {
                    state_instances: 128,
                    choice_instances: 128,
                    states: 128,
                    transitions: 128,
                    queue: 32,
                    exact_assignments_per_projection: 16,
                },
            );
            graph_count += 1;
        }
    }
    assert_eq!(graph_count, 12);
}

#[test]
fn every_acyclic_or_disconnected_four_atom_graph_matches_the_oracle() {
    let mut graph_count = 0_usize;
    for mask in 0..simple_graph_count(4) {
        let fixture = simple_graph_fixture(4, mask);
        if fixture_has_cycle(&fixture) {
            continue;
        }
        assert_fixture_conforms(
            &format!("n4-forest-mask-{mask:02x}"),
            &fixture,
            ConformanceBounds {
                state_instances: 256,
                choice_instances: 256,
                states: 128,
                transitions: 192,
                queue: 32,
                exact_assignments_per_projection: 16,
            },
        );
        graph_count += 1;
    }
    assert_eq!(graph_count, 38);
}

#[test]
fn every_cyclic_four_atom_graph_matches_the_oracle() {
    let mut graph_count = 0_usize;
    for mask in 0..simple_graph_count(4) {
        let fixture = simple_graph_fixture(4, mask);
        if !fixture_has_cycle(&fixture) {
            continue;
        }
        assert_fixture_conforms(
            &format!("n4-cyclic-mask-{mask:02x}"),
            &fixture,
            ConformanceBounds {
                state_instances: 1_024,
                choice_instances: 1_024,
                states: 1_024,
                transitions: 2_048,
                queue: 256,
                exact_assignments_per_projection: 64,
            },
        );
        graph_count += 1;
    }
    assert_eq!(graph_count, 26);
}

#[test]
fn every_explicit_ring_token_and_fixed_endpoint_order_matches_the_oracle() {
    for token in [
        FixtureBondToken::Aromatic,
        FixtureBondToken::Single,
        FixtureBondToken::Double,
        FixtureBondToken::Triple,
        FixtureBondToken::DativeAToB,
        FixtureBondToken::DativeBToA,
    ] {
        let fixture = FixtureSpec::new(
            &["A", "B", "C"],
            &[
                (0, 1, token),
                (0, 2, FixtureBondToken::Elided),
                (1, 2, FixtureBondToken::Elided),
            ],
        );
        let metrics = assert_fixture_conforms(
            &format!("explicit-triangle-{token:?}"),
            &fixture,
            ConformanceBounds {
                state_instances: 128,
                choice_instances: 128,
                states: 128,
                transitions: 128,
                queue: 32,
                exact_assignments_per_projection: 16,
            },
        );
        assert!(
            metrics.witnessed_ring_opening_endpoints.contains(&(0, 0))
                && metrics.witnessed_ring_opening_endpoints.contains(&(0, 1)),
            "the explicit bond was not opened from both fixed endpoints: {metrics:#?}"
        );
        for plan in [
            OracleBondPlan::Ring10,
            OracleBondPlan::Ring01,
            OracleBondPlan::Ring11,
        ] {
            assert!(
                metrics
                    .witnessed_closed_plan_domains
                    .contains(&(0, plan.bit())),
                "the explicit bond never closed with plan {plan:?}: {metrics:#?}"
            );
        }
    }
}

#[test]
fn simultaneous_explicit_ring_plans_and_labels_match_the_oracle() {
    let two_ring = FixtureSpec::new(
        &["A", "B", "C", "D"],
        &[
            (0, 1, FixtureBondToken::Elided),
            (0, 3, FixtureBondToken::Double),
            (1, 2, FixtureBondToken::Elided),
            (1, 3, FixtureBondToken::Triple),
            (2, 3, FixtureBondToken::Elided),
        ],
    );
    let metrics = assert_fixture_conforms(
        "two-simultaneous-explicit-rings",
        &two_ring,
        ConformanceBounds {
            state_instances: 2_048,
            choice_instances: 2_048,
            states: 1_024,
            transitions: 2_048,
            queue: 256,
            exact_assignments_per_projection: 128,
        },
    );
    assert!(metrics.maximum_open_rings >= 2, "metrics: {metrics:#?}");
    assert!(
        metrics.witnessed_equal_text_distinct_successors,
        "the explicit multi-ring fixture never preserved an equal-text semantic branch: {metrics:#?}"
    );
    assert!(
        metrics.maximum_distinct_closure_bonds >= 2,
        "two open explicit rings never offered competing closure bonds: {metrics:#?}"
    );
    assert!(
        metrics.witnessed_label_owner_sets.contains(&vec![1, 3]),
        "the two explicit ring bonds were never simultaneously open: {metrics:#?}"
    );

    let k4 = FixtureSpec::new(
        &["A", "B", "C", "D"],
        &[
            (0, 1, FixtureBondToken::Elided),
            (0, 2, FixtureBondToken::Double),
            (0, 3, FixtureBondToken::Triple),
            (1, 2, FixtureBondToken::Elided),
            (1, 3, FixtureBondToken::DativeAToB),
            (2, 3, FixtureBondToken::Elided),
        ],
    );
    let metrics = assert_fixture_conforms(
        "three-simultaneous-explicit-rings",
        &k4,
        ConformanceBounds {
            state_instances: 8_192,
            choice_instances: 8_192,
            states: 4_096,
            transitions: 8_192,
            queue: 1_024,
            exact_assignments_per_projection: 512,
        },
    );
    assert!(metrics.maximum_open_rings >= 3, "metrics: {metrics:#?}");
    assert!(
        metrics.witnessed_label_owner_sets.contains(&vec![1, 2, 4]),
        "the three intended explicit ring bonds were never open together: {metrics:#?}"
    );
}

#[test]
fn bridged_and_disconnected_explicit_surfaces_match_the_oracle() {
    let bridged = FixtureSpec::new(
        &["A", "B", "C", "D", "E"],
        &[
            (0, 1, FixtureBondToken::Elided),
            (1, 3, FixtureBondToken::Single),
            (0, 2, FixtureBondToken::Double),
            (2, 3, FixtureBondToken::Elided),
            (0, 4, FixtureBondToken::DativeAToB),
            (4, 3, FixtureBondToken::Triple),
        ],
    );
    let metrics = assert_fixture_conforms(
        "mixed-explicit-theta",
        &bridged,
        ConformanceBounds {
            state_instances: 4_096,
            choice_instances: 4_096,
            states: 4_096,
            transitions: 8_192,
            queue: 512,
            exact_assignments_per_projection: 1_024,
        },
    );
    assert!(
        metrics.witnessed_label_owner_sets.contains(&vec![2, 4]),
        "the double and dative theta bonds were never open together: {metrics:#?}"
    );

    let disconnected = FixtureSpec::new(
        &["A", "B", "C", "D", "E", "F"],
        &[
            (0, 1, FixtureBondToken::Elided),
            (0, 2, FixtureBondToken::Double),
            (1, 2, FixtureBondToken::Elided),
            (3, 4, FixtureBondToken::Elided),
            (3, 5, FixtureBondToken::DativeBToA),
            (4, 5, FixtureBondToken::Elided),
        ],
    );
    let metrics = assert_fixture_conforms(
        "disconnected-explicit-cycles",
        &disconnected,
        ConformanceBounds {
            state_instances: 4_096,
            choice_instances: 4_096,
            states: 4_096,
            transitions: 8_192,
            queue: 512,
            exact_assignments_per_projection: 64,
        },
    );
    assert!(metrics.maximum_open_rings >= 1);
    assert!(
        metrics.witnessed_inactive_component_boundary,
        "no silent boundary between explicit components was observed: {metrics:#?}"
    );
    assert!(
        metrics
            .label_one_owners_after_another_component
            .contains(&1)
            && metrics
                .label_one_owners_after_another_component
                .contains(&4),
        "label 1 was not reused by explicit rings after crossing both component orders: {metrics:#?}"
    );
}
