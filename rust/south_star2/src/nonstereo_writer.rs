//! Connected non-stereo visible-token state.
//!
//! Graph and constraint semantics live in `WriterState`. This module owns only
//! concrete non-stereo spelling facts, visible ring-label assignments, and the
//! small lexical commitments forced by multi-token SMILES constructs.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::ids::{AtomId, BondId};
use crate::prepared::{AdjacentBond, PreparedBond, PreparedGraph, PreparedMolecule};
use crate::solver::ConstraintSolver;
use crate::writer_state::{
    StructuralFrontier, TransitionError, WriterContradiction, WriterState,
};

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

    fn child_prefix(&self, parent: AtomId, incident: AdjacentBond) -> &str {
        let bond_text = self.bond_text(incident.bond(), parent);
        if bond_text.is_empty() {
            self.atom_text(incident.atom())
        } else {
            bond_text
        }
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

impl Error for PreparedConnectedNonStereoError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SpellingError {
    RingLabelOutOfRange { label: usize },
}

impl fmt::Display for SpellingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RingLabelOutOfRange { label } => write!(
                formatter,
                "ring label {label} is outside the selected 1..=99 spelling dialect"
            ),
        }
    }
}

impl Error for SpellingError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum NonStereoFrontierError {
    Writer(WriterContradiction),
    Spelling(SpellingError),
}

impl fmt::Display for NonStereoFrontierError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Writer(error) => write!(formatter, "writer state contradicted: {error}"),
            Self::Spelling(error) => write!(formatter, "writer spelling failed: {error}"),
        }
    }
}

impl Error for NonStereoFrontierError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Writer(error) => Some(error),
            Self::Spelling(error) => Some(error),
        }
    }
}

#[derive(Debug)]
pub(crate) enum NonStereoAdvanceError<E> {
    Constraint(E),
    Writer(WriterContradiction),
    Spelling(SpellingError),
    ChoiceUnavailable(NonStereoChoice),
}

impl<E: fmt::Display> fmt::Display for NonStereoAdvanceError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constraint(error) => write!(formatter, "constraint transition failed: {error}"),
            Self::Writer(error) => write!(formatter, "writer transition contradicted: {error}"),
            Self::Spelling(error) => write!(formatter, "writer spelling failed: {error}"),
            Self::ChoiceUnavailable(choice) => {
                write!(formatter, "non-stereo choice is not available: {choice:?}")
            }
        }
    }
}

impl<E: Error + 'static> Error for NonStereoAdvanceError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Constraint(error) => Some(error),
            Self::Writer(error) => Some(error),
            Self::Spelling(error) => Some(error),
            Self::ChoiceUnavailable(_) => None,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RingLabelSlot(usize);

impl RingLabelSlot {
    const fn index(self) -> usize {
        self.0
    }
}

/// Visible ring-label assignment. Closed labels are immediately reusable; this
/// is the minimal least-free spelling law rather than a copied batch-planner
/// schedule.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RingLabels {
    bonds_by_slot: BTreeMap<RingLabelSlot, BondId>,
}

impl RingLabels {
    fn next_available(&self) -> Result<RingLabelSlot, SpellingError> {
        let mut candidate = 0_usize;
        for slot in self.bonds_by_slot.keys() {
            if slot.index() != candidate {
                break;
            }
            candidate += 1;
        }
        let label = candidate
            .checked_add(1)
            .expect("visible ring-label number must not overflow");
        if label > 99 {
            return Err(SpellingError::RingLabelOutOfRange { label });
        }
        Ok(RingLabelSlot(candidate))
    }

    fn allocate(&mut self, bond: BondId) -> Result<RingLabelSlot, SpellingError> {
        assert!(
            self.bonds_by_slot.values().all(|owner| *owner != bond),
            "one ring bond may own only one visible label"
        );
        let slot = self.next_available()?;
        assert_eq!(
            self.bonds_by_slot.insert(slot, bond),
            None,
            "a newly allocated visible ring label must be free"
        );
        Ok(slot)
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

    fn is_empty(&self) -> bool {
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum NonStereoChoice {
    Root(AtomId),
    RingOpen(AdjacentBond),
    RingClose(AdjacentBond),
    BranchOpen(AdjacentBond),
    InlineChild(AdjacentBond),
    Pending,
    BranchClose,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VisibleChoice {
    choice: NonStereoChoice,
    text: String,
}

impl VisibleChoice {
    pub(crate) const fn choice(&self) -> NonStereoChoice {
        self.choice
    }

    pub(crate) fn text(&self) -> &str {
        &self.text
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ConnectedNonStereoWriterState<S> {
    surface: PreparedConnectedNonStereo,
    structural: WriterState<S>,
    labels: RingLabels,
    pending: Option<PendingEmission>,
}

impl<S: ConstraintSolver> ConnectedNonStereoWriterState<S> {
    pub(crate) fn initial(surface: &PreparedConnectedNonStereo) -> Result<Self, S::Error> {
        Ok(Self {
            surface: surface.clone(),
            structural: WriterState::initial(surface.molecule())?,
            labels: RingLabels::default(),
            pending: None,
        })
    }

    pub(crate) fn active_atom(&self) -> Option<AtomId> {
        self.structural.active_atom()
    }

    pub(crate) const fn graph_is_complete(&self) -> bool {
        self.structural.graph_is_complete()
    }

    pub(crate) fn is_accepted(&self) -> bool {
        self.pending.is_none()
            && self.labels.is_empty()
            && self.structural.active_atom().is_none()
            && self.structural.graph_is_complete()
    }

    pub(crate) fn choices(&self) -> Result<Vec<VisibleChoice>, NonStereoFrontierError> {
        if self.is_accepted() {
            return Ok(Vec::new());
        }
        if self.pending.is_some() {
            return Ok(vec![self.visible_choice(NonStereoChoice::Pending)?]);
        }

        match self.frontier()? {
            StructuralFrontier::ComponentRoots(roots) => {
                self.visible_choices(roots.iter().copied().map(NonStereoChoice::Root))
            }
            StructuralFrontier::RingSuffix { openings, closures } => self.visible_choices(
                closures
                    .iter()
                    .copied()
                    .map(NonStereoChoice::RingClose)
                    .chain(openings.iter().copied().map(NonStereoChoice::RingOpen)),
            ),
            StructuralFrontier::BranchChildren(children) => self.visible_choices(
                children
                    .iter()
                    .copied()
                    .map(NonStereoChoice::BranchOpen),
            ),
            StructuralFrontier::InlineChild(incident) => {
                Ok(vec![self.visible_choice(NonStereoChoice::InlineChild(incident))?])
            }
            StructuralFrontier::CompletePath => {
                assert!(
                    !self.structural.graph_is_complete(),
                    "complete connected components must normalize before frontier inspection"
                );
                Ok(vec![self.visible_choice(NonStereoChoice::BranchClose)?])
            }
            StructuralFrontier::Terminal => {
                assert!(
                    self.is_accepted(),
                    "terminal structural state must have clean spelling state"
                );
                Ok(Vec::new())
            }
        }
    }

    pub(crate) fn advance(
        &self,
        choice: NonStereoChoice,
    ) -> Result<(String, Self), NonStereoAdvanceError<S::Error>> {
        let visible = self
            .choices()
            .map_err(frontier_advance_error)?
            .into_iter()
            .find(|candidate| candidate.choice == choice)
            .ok_or(NonStereoAdvanceError::ChoiceUnavailable(choice))?;
        let token = visible.text;

        let successor = match choice {
            NonStereoChoice::Root(root) => Self {
                surface: self.surface.clone(),
                structural: self
                    .structural
                    .begin_component(root)
                    .map_err(NonStereoAdvanceError::Writer)?,
                labels: self.labels.clone(),
                pending: None,
            }
            .normalize_component_completion()
            .map_err(NonStereoAdvanceError::Writer)?,
            NonStereoChoice::RingOpen(incident) => {
                let structural = self
                    .structural
                    .open_ring_endpoint(incident)
                    .map_err(transition_advance_error)?;
                let mut labels = self.labels.clone();
                let slot = labels
                    .allocate(incident.bond())
                    .map_err(NonStereoAdvanceError::Spelling)?;
                assert_eq!(
                    token,
                    ring_label_text(slot).map_err(NonStereoAdvanceError::Spelling)?,
                    "advertised ring label must match the allocated label"
                );
                Self {
                    surface: self.surface.clone(),
                    structural,
                    labels,
                    pending: None,
                }
            }
            NonStereoChoice::RingClose(incident) => {
                let first_endpoint = self
                    .structural
                    .ring_closure_first_endpoint(incident)
                    .map_err(NonStereoAdvanceError::Writer)?;
                let label_slot = self.labels.slot_for_bond(incident.bond());
                let bond_text = self.surface.bond_text(incident.bond(), first_endpoint);
                if bond_text.is_empty() {
                    let structural = self
                        .structural
                        .close_ring_endpoint(incident)
                        .map_err(NonStereoAdvanceError::Writer)?;
                    let mut labels = self.labels.clone();
                    labels.release(label_slot, incident.bond());
                    Self {
                        surface: self.surface.clone(),
                        structural,
                        labels,
                        pending: None,
                    }
                    .normalize_component_completion()
                    .map_err(NonStereoAdvanceError::Writer)?
                } else {
                    Self {
                        surface: self.surface.clone(),
                        structural: self.structural.clone(),
                        labels: self.labels.clone(),
                        pending: Some(PendingEmission::RingClosureLabel {
                            incident,
                            label_slot,
                        }),
                    }
                }
            }
            NonStereoChoice::BranchOpen(incident) => Self {
                surface: self.surface.clone(),
                structural: self
                    .structural
                    .commit_traversal_edge(incident)
                    .map_err(transition_advance_error)?,
                labels: self.labels.clone(),
                pending: Some(PendingEmission::BranchBondOrAtom(incident)),
            },
            NonStereoChoice::InlineChild(incident) => {
                let parent = self
                    .structural
                    .active_atom()
                    .expect("inline emission requires an active atom");
                let bond_text = self.surface.bond_text(incident.bond(), parent);
                let structural = self
                    .structural
                    .commit_traversal_edge(incident)
                    .map_err(transition_advance_error)?;
                if bond_text.is_empty() {
                    Self {
                        surface: self.surface.clone(),
                        structural: structural
                            .enter_inline_child(incident)
                            .map_err(NonStereoAdvanceError::Writer)?,
                        labels: self.labels.clone(),
                        pending: None,
                    }
                    .normalize_component_completion()
                    .map_err(NonStereoAdvanceError::Writer)?
                } else {
                    Self {
                        surface: self.surface.clone(),
                        structural,
                        labels: self.labels.clone(),
                        pending: Some(PendingEmission::InlineAtom(incident)),
                    }
                }
            }
            NonStereoChoice::Pending => self
                .advance_pending()
                .map_err(NonStereoAdvanceError::Writer)?,
            NonStereoChoice::BranchClose => Self {
                surface: self.surface.clone(),
                structural: self
                    .structural
                    .complete_path()
                    .map_err(NonStereoAdvanceError::Writer)?,
                labels: self.labels.clone(),
                pending: None,
            },
        };

        Ok((token, successor))
    }

    fn advance_pending(&self) -> Result<Self, WriterContradiction> {
        let pending = self.pending.expect("no committed text is pending");
        let incident = pending.incident();

        match pending {
            PendingEmission::InlineAtom(_) => Self {
                surface: self.surface.clone(),
                structural: self.structural.enter_inline_child(incident)?,
                labels: self.labels.clone(),
                pending: None,
            }
            .normalize_component_completion(),
            PendingEmission::BranchBondOrAtom(_) => {
                let parent = self
                    .structural
                    .active_atom()
                    .expect("a committed branch child requires its active parent");
                if self.surface.bond_text(incident.bond(), parent).is_empty() {
                    Self {
                        surface: self.surface.clone(),
                        structural: self.structural.enter_branch_child(incident)?,
                        labels: self.labels.clone(),
                        pending: None,
                    }
                    .normalize_component_completion()
                } else {
                    Ok(Self {
                        surface: self.surface.clone(),
                        structural: self.structural.clone(),
                        labels: self.labels.clone(),
                        pending: Some(PendingEmission::BranchAtom(incident)),
                    })
                }
            }
            PendingEmission::BranchAtom(_) => Self {
                surface: self.surface.clone(),
                structural: self.structural.enter_branch_child(incident)?,
                labels: self.labels.clone(),
                pending: None,
            }
            .normalize_component_completion(),
            PendingEmission::RingClosureLabel { label_slot, .. } => {
                assert_eq!(
                    self.labels.slot_for_bond(incident.bond()),
                    label_slot,
                    "a pending ring label must retain its assignment"
                );
                let structural = self.structural.close_ring_endpoint(incident)?;
                let mut labels = self.labels.clone();
                labels.release(label_slot, incident.bond());
                Self {
                    surface: self.surface.clone(),
                    structural,
                    labels,
                    pending: None,
                }
                .normalize_component_completion()
            }
        }
    }

    fn visible_choices(
        &self,
        choices: impl IntoIterator<Item = NonStereoChoice>,
    ) -> Result<Vec<VisibleChoice>, NonStereoFrontierError> {
        choices
            .into_iter()
            .map(|choice| self.visible_choice(choice))
            .collect()
    }

    fn visible_choice(
        &self,
        choice: NonStereoChoice,
    ) -> Result<VisibleChoice, NonStereoFrontierError> {
        Ok(VisibleChoice {
            choice,
            text: self.choice_text(choice)?,
        })
    }

    fn choice_text(&self, choice: NonStereoChoice) -> Result<String, NonStereoFrontierError> {
        match choice {
            NonStereoChoice::Root(root) => Ok(self.surface.atom_text(root).to_owned()),
            NonStereoChoice::RingOpen(_) => ring_label_text(
                self.labels
                    .next_available()
                    .map_err(NonStereoFrontierError::Spelling)?,
            )
            .map_err(NonStereoFrontierError::Spelling),
            NonStereoChoice::RingClose(incident) => {
                let first_endpoint = self
                    .structural
                    .ring_closure_first_endpoint(incident)
                    .map_err(NonStereoFrontierError::Writer)?;
                let bond_text = self.surface.bond_text(incident.bond(), first_endpoint);
                if bond_text.is_empty() {
                    ring_label_text(self.labels.slot_for_bond(incident.bond()))
                        .map_err(NonStereoFrontierError::Spelling)
                } else {
                    Ok(bond_text.to_owned())
                }
            }
            NonStereoChoice::BranchOpen(_) => Ok("(".to_owned()),
            NonStereoChoice::InlineChild(incident) => {
                let parent = self
                    .structural
                    .active_atom()
                    .expect("inline choice requires an active atom");
                Ok(self.surface.child_prefix(parent, incident).to_owned())
            }
            NonStereoChoice::Pending => {
                let pending = self.pending.expect("no committed text is pending");
                let incident = pending.incident();
                match pending {
                    PendingEmission::InlineAtom(_) | PendingEmission::BranchAtom(_) => {
                        Ok(self.surface.atom_text(incident.atom()).to_owned())
                    }
                    PendingEmission::BranchBondOrAtom(_) => {
                        let parent = self
                            .structural
                            .active_atom()
                            .expect("a committed branch child requires its active parent");
                        Ok(self.surface.child_prefix(parent, incident).to_owned())
                    }
                    PendingEmission::RingClosureLabel { label_slot, .. } => {
                        ring_label_text(label_slot).map_err(NonStereoFrontierError::Spelling)
                    }
                }
            }
            NonStereoChoice::BranchClose => Ok(")".to_owned()),
        }
    }

    fn frontier(&self) -> Result<StructuralFrontier, NonStereoFrontierError> {
        self.structural
            .structural_frontier()
            .map_err(NonStereoFrontierError::Writer)
    }

    fn normalize_component_completion(mut self) -> Result<Self, WriterContradiction> {
        if self.pending.is_some() || !self.structural.graph_is_complete() {
            return Ok(self);
        }
        assert!(
            self.labels.is_empty(),
            "a complete structural graph must not retain visible ring labels"
        );
        assert_eq!(
            self.structural.structural_frontier()?,
            StructuralFrontier::CompletePath,
            "a complete connected graph must have one completable top-level path"
        );
        self.structural = self.structural.complete_path()?;
        assert_eq!(
            self.structural.active_atom(),
            None,
            "connected graph completion must not restore a branch parent"
        );
        Ok(self)
    }
}

fn frontier_advance_error<E>(error: NonStereoFrontierError) -> NonStereoAdvanceError<E> {
    match error {
        NonStereoFrontierError::Writer(error) => NonStereoAdvanceError::Writer(error),
        NonStereoFrontierError::Spelling(error) => NonStereoAdvanceError::Spelling(error),
    }
}

fn transition_advance_error<E>(error: TransitionError<E>) -> NonStereoAdvanceError<E> {
    match error {
        TransitionError::Constraint(error) => NonStereoAdvanceError::Constraint(error),
        TransitionError::Writer(error) => NonStereoAdvanceError::Writer(error),
    }
}

fn ring_label_text(label_slot: RingLabelSlot) -> Result<String, SpellingError> {
    let label = label_slot
        .index()
        .checked_add(1)
        .expect("visible ring-label number must not overflow");
    ring_label_number_text(label)
}

fn ring_label_number_text(label: usize) -> Result<String, SpellingError> {
    if !(1..=99).contains(&label) {
        return Err(SpellingError::RingLabelOutOfRange { label });
    }
    if label < 10 {
        Ok(label.to_string())
    } else {
        Ok(format!("%{label}"))
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
