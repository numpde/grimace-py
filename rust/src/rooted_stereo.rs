use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rustc_hash::FxHashMap;

use crate::bond_stereo_constraints::{
    ambiguous_shared_edge_groups, canonical_edge, component_sizes, flip_direction_token,
    is_stereo_double_bond, rdkit_local_writer_hazards, stereo_component_ids,
    stereo_constraint_model, stereo_side_infos, AmbiguousSharedEdgeGroup,
    RdkitTokenFlipAdjustmentObservations, StereoAssignmentState, StereoComponentConstraintModel,
    StereoComponentPhase, StereoConstraintFact, StereoConstraintLayer, StereoConstraintModel,
    StereoConstraintState, StereoDirectionToken, StereoMarkerEventFact, StereoSideInfo,
    StereoSideInfoBuild, StereoTokenBasisFact, StereoTokenFlip, StereoTokenFlipFact,
    StereoTokenObservationFact, StereoTraversalRole, CIS_STEREO_BOND_KINDS,
    TRANS_STEREO_BOND_KINDS,
};
use crate::frontier::{
    choice_texts, frontier_prefix as shared_frontier_prefix, grouped_choice_texts,
    take_choice_or_err, take_grouped_choices_or_err, take_transition_or_err, DecoderChoice,
};
use crate::prepared_graph::{PreparedSmilesGraphData, CONNECTED_STEREO_SURFACE};
use crate::smiles_shared::{add_pending, ring_label_text, take_pending_for_atom};

const HYDROGEN_NEIGHBOR: isize = -1;
const UNKNOWN_COMPONENT_PHASE: i8 = -1;
const STORED_COMPONENT_PHASE: i8 = 0;
const FLIPPED_COMPONENT_PHASE: i8 = 1;
const UNKNOWN_COMPONENT_TOKEN_FLIP: i8 = -1;
const STORED_COMPONENT_TOKEN_FLIP: i8 = 0;
const FLIPPED_COMPONENT_TOKEN_FLIP: i8 = 1;
const UNKNOWN_EDGE_ORIENTATION: i8 = -1;
const BEFORE_ATOM_EDGE_ORIENTATION: i8 = 0;
const AFTER_ATOM_EDGE_ORIENTATION: i8 = 1;

const AROMATIC_SUBSET: &[&str] = &["b", "c", "n", "o", "p", "s", "se", "as"];
const SUPPORTED_CHIRAL_TAGS: &[&str] = &[
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_TETRAHEDRAL_CW",
];

const ELEMENT_SYMBOLS: [&str; 119] = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PendingRing {
    label: usize,
    other_atom_idx: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct DeferredDirectionalComponentToken {
    component_idx: usize,
    reference_tokens: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct DeferredDirectionalToken {
    component_tokens: Arc<[DeferredDirectionalComponentToken]>,
    begin_idx: isize,
    end_idx: isize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct DirectionalMarkerTrace {
    slot: usize,
    marker: char,
    component_idx: isize,
    side_idx: isize,
    endpoint_atom_idx: isize,
    selected_neighbor_idx: isize,
    support_boundary_selected_neighbor_idx: isize,
    edge_begin_idx: isize,
    edge_end_idx: isize,
    role: StereoTraversalRole,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct MarkerEventTrace {
    slot: usize,
    marker: Option<char>,
    component_idx: isize,
    side_idx: isize,
    endpoint_atom_idx: isize,
    edge_neighbor_idx: isize,
    edge_begin_idx: isize,
    edge_end_idx: isize,
    role: StereoTraversalRole,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Part {
    Literal(String),
    Deferred(DeferredDirectionalToken),
}

struct EmittedEdgePartResult {
    part: Part,
    selected_neighbors: Vec<isize>,
    selected_orientations: Vec<i8>,
    first_emitted_candidates: Vec<isize>,
    deferred_carrier_choice_constraints: Vec<DeferredCarrierChoiceConstraint>,
    token_basis_facts: Vec<Option<StereoTokenBasisFact>>,
}

struct ProcessChildrenEdgeUpdate {
    edge_part: Part,
    selected_neighbors: Vec<isize>,
    selected_orientations: Vec<i8>,
    first_emitted_candidates: Vec<isize>,
    deferred_carrier_choice_constraints: Vec<DeferredCarrierChoiceConstraint>,
    token_basis_facts: Vec<Option<StereoTokenBasisFact>>,
    component_phases: Vec<i8>,
    component_begin_atoms: Vec<isize>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct DeferredComponentPhaseFact {
    component_idx: usize,
    begin_atom_idx: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ComponentPhaseCommitFact {
    component_idx: usize,
    phase: i8,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ComponentBeginAtomFact {
    component_idx: usize,
    begin_atom_idx: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct CommittedComponentTokenFlipFact {
    runtime_component_idx: usize,
    token_flip: StereoTokenFlip,
}

impl CommittedComponentTokenFlipFact {
    fn token_flip_fact(self) -> StereoTokenFlipFact {
        StereoTokenFlipFact {
            runtime_component_idx: self.runtime_component_idx,
            token_flip: self.token_flip,
        }
    }
}

impl DeferredComponentPhaseFact {
    fn begin_atom_fact(self) -> ComponentBeginAtomFact {
        ComponentBeginAtomFact {
            component_idx: self.component_idx,
            begin_atom_idx: self.begin_atom_idx,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct DeferredCarrierChoiceConstraint {
    side_idx: usize,
    deferred_neighbor_idx: usize,
    available_neighbors: Vec<usize>,
}

struct StereoEdgeEmissionContext<'a> {
    graph: &'a PreparedSmilesGraphData,
    constraint_model: &'a StereoConstraintModel,
    side_infos: &'a [StereoSideInfo],
    side_ids_by_component: &'a [Vec<usize>],
    edge_to_side_ids: &'a BTreeMap<(usize, usize), Vec<usize>>,
    isolated_components: &'a [bool],
}

struct StereoEdgeEmissionState<'a> {
    component_phases: &'a [i8],
    selected_neighbors: &'a [isize],
    selected_orientations: &'a [i8],
    first_emitted_candidates: &'a [isize],
    component_begin_atoms: &'a [isize],
    deferred_carrier_choice_constraints: &'a [DeferredCarrierChoiceConstraint],
    token_basis_facts: &'a [Option<StereoTokenBasisFact>],
    marker_event_traces: &'a [MarkerEventTrace],
    committed_component_token_flips: &'a [Option<StereoTokenFlip>],
}

type PendingRingBuckets = Vec<(usize, Vec<PendingRing>)>;

struct TakenPendingRings {
    pending: Arc<PendingRingBuckets>,
    rings: Vec<PendingRing>,
}

struct StereoProcessChildrenContext<'a> {
    runtime: &'a StereoWalkerRuntimeData,
    graph: &'a PreparedSmilesGraphData,
    require_completable: bool,
    completion_cache: &'a mut StereoCompletionCache,
}

struct ProcessChildrenTerminalStep {
    parent_idx: usize,
    child_idx: usize,
    edge_part: Part,
}

struct AtomStereoExpansionInput<'a> {
    graph: &'a PreparedSmilesGraphData,
    base_state: &'a RootedConnectedStereoWalkerStateData,
    atom_idx: usize,
    parent_idx: Option<usize>,
    visited_now: Arc<[bool]>,
    visited_count_now: usize,
    pending_now: Vec<(usize, Vec<PendingRing>)>,
    closures_here: Vec<PendingRing>,
    ordered_groups: Vec<Vec<usize>>,
    is_chiral_atom: bool,
}

struct ExactAtomStereoExpansionInput<'a> {
    graph: &'a PreparedSmilesGraphData,
    state: &'a RootedConnectedStereoExactStateData,
    base_action_stack: &'a [ExactWalkerAction],
    atom_idx: usize,
    parent_idx: Option<usize>,
    visited_now: Arc<[bool]>,
    visited_count_now: usize,
    pending_base: Arc<Vec<(usize, Vec<PendingRing>)>>,
    closures_here: Vec<PendingRing>,
    ordered_groups: Vec<Vec<usize>>,
    is_chiral_atom: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum RingAction {
    Close(usize),
    Open(usize),
}

fn ring_actions_for_choice(
    closures_here_len: usize,
    ordered_groups: &[Vec<usize>],
    chosen_children: &[usize],
) -> Vec<RingAction> {
    let total_group_members = ordered_groups
        .iter()
        .map(|group| group.len())
        .sum::<usize>();
    let opening_target_count = total_group_members.saturating_sub(chosen_children.len());
    let mut ring_actions = Vec::with_capacity(closures_here_len + opening_target_count);
    for closure_idx in 0..closures_here_len {
        ring_actions.push(RingAction::Close(closure_idx));
    }
    for group in ordered_groups {
        for &target_idx in group {
            if !chosen_children.contains(&target_idx) {
                ring_actions.push(RingAction::Open(target_idx));
            }
        }
    }
    ring_actions
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum WalkerAction {
    EmitLiteral(String),
    EmitRingLabel(usize),
    EmitCloseParen,
    EmitDeferred(DeferredDirectionalToken),
    EnterAtom {
        atom_idx: usize,
        parent_idx: Option<usize>,
    },
    ProcessChildren {
        parent_idx: usize,
        child_order: Arc<[usize]>,
        next_branch_index: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ExactBondToken {
    Dash,
    Eq,
    Hash,
    Dollar,
    Colon,
    Slash,
    Backslash,
    Other(String),
}

impl ExactBondToken {
    fn from_owned(token: String) -> Option<Self> {
        if token.is_empty() {
            return None;
        }
        Some(match token.as_str() {
            "-" => Self::Dash,
            "=" => Self::Eq,
            "#" => Self::Hash,
            "$" => Self::Dollar,
            ":" => Self::Colon,
            "/" => Self::Slash,
            "\\" => Self::Backslash,
            _ => Self::Other(token),
        })
    }

    fn as_str(&self) -> &str {
        match self {
            Self::Dash => "-",
            Self::Eq => "=",
            Self::Hash => "#",
            Self::Dollar => "$",
            Self::Colon => ":",
            Self::Slash => "/",
            Self::Backslash => "\\",
            Self::Other(token) => token.as_str(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ExactWalkerAction {
    EmitBondToken(ExactBondToken),
    EmitRingLabel(usize),
    EmitCloseParen,
    EnterAtom {
        atom_idx: usize,
        parent_idx: Option<usize>,
    },
    ProcessChildren {
        parent_idx: usize,
        child_order: Arc<[usize]>,
        next_branch_index: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct RootedConnectedStereoWalkerStateData {
    prefix: Arc<str>,
    visited: Arc<[bool]>,
    visited_count: usize,
    pending: Arc<Vec<(usize, Vec<PendingRing>)>>,
    free_labels: Arc<Vec<usize>>,
    next_label: usize,
    stereo_component_phases: Arc<Vec<i8>>,
    stereo_selected_neighbors: Arc<Vec<isize>>,
    stereo_selected_orientations: Arc<Vec<i8>>,
    stereo_first_emitted_candidates: Arc<Vec<isize>>,
    stereo_component_begin_atoms: Arc<Vec<isize>>,
    committed_component_token_flips: Arc<Vec<Option<StereoTokenFlip>>>,
    deferred_carrier_choice_constraints: Arc<Vec<DeferredCarrierChoiceConstraint>>,
    stereo_token_basis_facts: Arc<Vec<Option<StereoTokenBasisFact>>>,
    directional_marker_traces: Arc<Vec<DirectionalMarkerTrace>>,
    marker_event_traces: Arc<Vec<MarkerEventTrace>>,
    action_stack: Vec<WalkerAction>,
}

#[derive(Clone, Debug)]
struct RootedConnectedStereoExactDynamicData {
    visited: Arc<[bool]>,
    visited_count: usize,
    pending: Arc<Vec<(usize, Vec<PendingRing>)>>,
    free_labels: Arc<Vec<usize>>,
    next_label: usize,
}

#[derive(Clone, Debug)]
struct RootedConnectedStereoExactStateData {
    prefix: Arc<str>,
    dynamic: Arc<RootedConnectedStereoExactDynamicData>,
    action_stack: Vec<ExactWalkerAction>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StereoCompletionKey {
    visited: Arc<[bool]>,
    visited_count: usize,
    pending: Arc<Vec<(usize, Vec<PendingRing>)>>,
    free_labels: Arc<Vec<usize>>,
    next_label: usize,
    stereo_component_phases: Arc<Vec<i8>>,
    stereo_selected_neighbors: Arc<Vec<isize>>,
    stereo_selected_orientations: Arc<Vec<i8>>,
    stereo_first_emitted_candidates: Arc<Vec<isize>>,
    stereo_component_begin_atoms: Arc<Vec<isize>>,
    committed_component_token_flips: Arc<Vec<Option<StereoTokenFlip>>>,
    deferred_carrier_choice_constraints: Arc<Vec<DeferredCarrierChoiceConstraint>>,
    stereo_token_basis_facts: Arc<Vec<Option<StereoTokenBasisFact>>>,
    marker_event_traces: Arc<Vec<MarkerEventTrace>>,
    action_stack: Vec<WalkerAction>,
}

impl From<&RootedConnectedStereoWalkerStateData> for StereoCompletionKey {
    fn from(state: &RootedConnectedStereoWalkerStateData) -> Self {
        Self {
            visited: state.visited.clone(),
            visited_count: state.visited_count,
            pending: state.pending.clone(),
            free_labels: state.free_labels.clone(),
            next_label: state.next_label,
            stereo_component_phases: state.stereo_component_phases.clone(),
            stereo_selected_neighbors: state.stereo_selected_neighbors.clone(),
            stereo_selected_orientations: state.stereo_selected_orientations.clone(),
            stereo_first_emitted_candidates: state.stereo_first_emitted_candidates.clone(),
            stereo_component_begin_atoms: state.stereo_component_begin_atoms.clone(),
            committed_component_token_flips: state.committed_component_token_flips.clone(),
            deferred_carrier_choice_constraints: state.deferred_carrier_choice_constraints.clone(),
            stereo_token_basis_facts: state.stereo_token_basis_facts.clone(),
            marker_event_traces: state.marker_event_traces.clone(),
            action_stack: state.action_stack.clone(),
        }
    }
}

type StereoCompletionCache = FxHashMap<StereoCompletionKey, bool>;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum StereoNextTokenCompleteness {
    Raw,
    Completable,
}

impl StereoNextTokenCompleteness {
    fn from_require_completable(require_completable: bool) -> Self {
        if require_completable {
            Self::Completable
        } else {
            Self::Raw
        }
    }
}

#[derive(Clone, Debug)]
struct StereoWalkerRuntimeData {
    root_idx: usize,
    stereo_component_ids: Vec<isize>,
    isolated_components: Vec<bool>,
    side_infos: Vec<StereoSideInfo>,
    edge_to_side_ids: BTreeMap<(usize, usize), Vec<usize>>,
    side_ids_by_component: Vec<Vec<usize>>,
    ambiguous_shared_edge_groups: Vec<AmbiguousSharedEdgeGroup>,
    constraint_model: StereoConstraintModel,
}

#[derive(Clone)]
struct StereoDecoderBranch {
    runtime: Arc<StereoWalkerRuntimeData>,
    frontier: Vec<RootedConnectedStereoWalkerStateData>,
}

type StereoChoiceCache = Vec<DecoderChoice<RootedConnectedStereoWalkerStateData>>;

#[derive(Clone)]
enum StereoDecoderMode {
    Single {
        runtime: Arc<StereoWalkerRuntimeData>,
        frontier: Vec<RootedConnectedStereoWalkerStateData>,
        cached_choices: Option<StereoChoiceCache>,
    },
    Merged {
        branches: Vec<StereoDecoderBranch>,
    },
}

fn cached_single_stereo_choices<'a>(
    runtime: &Arc<StereoWalkerRuntimeData>,
    graph: &PreparedSmilesGraphData,
    frontier: &[RootedConnectedStereoWalkerStateData],
    cached_choices: &'a mut Option<StereoChoiceCache>,
) -> PyResult<&'a StereoChoiceCache> {
    if cached_choices.is_none() {
        *cached_choices = Some(frontier_choices_for_stereo(
            runtime.as_ref(),
            graph,
            frontier,
        )?);
    }
    match cached_choices.as_ref() {
        Some(choices) => Ok(choices),
        None => unreachable!("single decoder choice cache should be populated"),
    }
}

impl StereoDecoderMode {
    fn single(
        runtime: Arc<StereoWalkerRuntimeData>,
        frontier: Vec<RootedConnectedStereoWalkerStateData>,
    ) -> Self {
        Self::Single {
            runtime,
            frontier,
            cached_choices: None,
        }
    }

    fn merged(branches: Vec<StereoDecoderBranch>) -> Self {
        Self::Merged { branches }
    }

    fn next_token_support(
        &mut self,
        graph: &Arc<PreparedSmilesGraphData>,
    ) -> PyResult<Vec<String>> {
        match self {
            Self::Merged { branches } => {
                Ok(merged_stereo_grouped_successors(graph.clone(), branches)?
                    .into_iter()
                    .map(|(token, _)| token)
                    .collect())
            }
            Self::Single {
                runtime,
                frontier,
                cached_choices,
            } => Ok(grouped_choice_texts(cached_single_stereo_choices(
                runtime,
                graph.as_ref(),
                frontier,
                cached_choices,
            )?)),
        }
    }

    fn advance_token(
        &mut self,
        graph: &Arc<PreparedSmilesGraphData>,
        chosen_token: &str,
    ) -> PyResult<()> {
        match self {
            Self::Merged { branches } => {
                let successors = merged_stereo_grouped_successors(graph.clone(), branches)?;
                let (_, successor) = successors
                    .into_iter()
                    .find(|(token, _)| token == chosen_token)
                    .ok_or_else(|| {
                        PyKeyError::new_err(format!("Token {chosen_token:?} is not available"))
                    })?;
                *self = successor.mode;
                Ok(())
            }
            Self::Single {
                runtime,
                frontier,
                cached_choices,
            } => {
                let choices = match cached_choices.take() {
                    Some(choices) => choices,
                    None => {
                        frontier_choices_for_stereo(runtime.as_ref(), graph.as_ref(), frontier)?
                    }
                };
                *frontier = take_grouped_choices_or_err(choices, chosen_token)?;
                Ok(())
            }
        }
    }

    fn next_choice_texts(&mut self, graph: &Arc<PreparedSmilesGraphData>) -> PyResult<Vec<String>> {
        match self {
            Self::Merged { branches } => {
                Ok(merged_stereo_choice_successors(graph.clone(), branches)?
                    .into_iter()
                    .map(|(token, _)| token)
                    .collect())
            }
            Self::Single {
                runtime,
                frontier,
                cached_choices,
            } => Ok(choice_texts(cached_single_stereo_choices(
                runtime,
                graph.as_ref(),
                frontier,
                cached_choices,
            )?)),
        }
    }

    fn advance_choice(
        &mut self,
        graph: &Arc<PreparedSmilesGraphData>,
        chosen_idx: usize,
    ) -> PyResult<()> {
        match self {
            Self::Merged { branches } => {
                let mut successors = merged_stereo_choice_successors(graph.clone(), branches)?;
                if chosen_idx >= successors.len() {
                    return Err(PyKeyError::new_err(format!(
                        "Choice index {chosen_idx} is not available; choice_count={}",
                        successors.len()
                    )));
                }
                *self = successors.swap_remove(chosen_idx).1.mode;
                Ok(())
            }
            Self::Single {
                runtime,
                frontier,
                cached_choices,
            } => {
                let mut choices = match cached_choices.take() {
                    Some(choices) => choices,
                    None => {
                        frontier_choices_for_stereo(runtime.as_ref(), graph.as_ref(), frontier)?
                    }
                };
                *frontier = take_choice_or_err(&mut choices, chosen_idx)?;
                Ok(())
            }
        }
    }

    fn choice_successor_modes(
        &self,
        graph: &Arc<PreparedSmilesGraphData>,
    ) -> PyResult<Vec<(String, StereoDecoderMode)>> {
        match self {
            Self::Merged { branches } => {
                Ok(merged_stereo_choice_successors(graph.clone(), branches)?
                    .into_iter()
                    .map(|(token, successor)| (token, successor.mode))
                    .collect())
            }
            Self::Single {
                runtime, frontier, ..
            } => Ok(frontier_choice_successors_for_stereo(
                runtime.as_ref(),
                graph.as_ref(),
                frontier,
            )?
            .into_iter()
            .map(|(token, successor)| (token, Self::single(runtime.clone(), vec![successor])))
            .collect()),
        }
    }

    fn grouped_successor_modes(
        &self,
        graph: &Arc<PreparedSmilesGraphData>,
    ) -> PyResult<Vec<(String, StereoDecoderMode)>> {
        match self {
            Self::Merged { branches } => {
                Ok(merged_stereo_grouped_successors(graph.clone(), branches)?
                    .into_iter()
                    .map(|(token, successor)| (token, successor.mode))
                    .collect())
            }
            Self::Single {
                runtime, frontier, ..
            } => Ok(frontier_transitions_for_stereo_linear(
                runtime.as_ref(),
                graph.as_ref(),
                frontier,
            )?
            .into_iter()
            .map(|(token, frontier)| (token, Self::single(runtime.clone(), frontier)))
            .collect()),
        }
    }

    fn prefix(&self) -> String {
        match self {
            Self::Merged { branches } => merged_stereo_prefix(branches),
            Self::Single { frontier, .. } => stereo_frontier_prefix(frontier),
        }
    }

    fn cache_key(&self) -> String {
        match self {
            Self::Merged { branches } => merged_stereo_cache_key(branches),
            Self::Single { frontier, .. } => format!("{:?}", frontier),
        }
    }

    fn is_terminal(&self, graph: &PreparedSmilesGraphData) -> PyResult<bool> {
        match self {
            Self::Merged { branches } => merged_stereo_is_terminal(graph, branches),
            Self::Single {
                runtime,
                frontier,
                cached_choices,
            } => {
                if let Some(choices) = cached_choices {
                    Ok(choices.is_empty())
                } else {
                    stereo_frontier_is_terminal(runtime.as_ref(), graph, frontier)
                }
            }
        }
    }

    fn frontier_size(&self) -> usize {
        match self {
            Self::Single { frontier, .. } => frontier.len(),
            Self::Merged { branches } => branches.iter().map(|branch| branch.frontier.len()).sum(),
        }
    }
}

fn stereo_exact_state_from_full(
    state: RootedConnectedStereoWalkerStateData,
) -> RootedConnectedStereoExactStateData {
    let RootedConnectedStereoWalkerStateData {
        prefix,
        visited,
        visited_count,
        pending,
        free_labels,
        next_label,
        stereo_component_phases,
        stereo_selected_neighbors,
        stereo_selected_orientations,
        stereo_first_emitted_candidates,
        stereo_component_begin_atoms,
        committed_component_token_flips,
        deferred_carrier_choice_constraints,
        stereo_token_basis_facts,
        directional_marker_traces,
        marker_event_traces,
        action_stack,
    } = state;
    debug_assert!(stereo_component_phases.is_empty());
    debug_assert!(stereo_selected_neighbors.is_empty());
    debug_assert!(stereo_selected_orientations.is_empty());
    debug_assert!(stereo_first_emitted_candidates.is_empty());
    debug_assert!(stereo_component_begin_atoms.is_empty());
    debug_assert!(committed_component_token_flips.is_empty());
    debug_assert!(deferred_carrier_choice_constraints.is_empty());
    debug_assert!(stereo_token_basis_facts.is_empty());
    debug_assert!(directional_marker_traces.is_empty());
    debug_assert!(marker_event_traces.is_empty());
    RootedConnectedStereoExactStateData {
        prefix,
        dynamic: Arc::new(RootedConnectedStereoExactDynamicData {
            visited,
            visited_count,
            pending,
            free_labels,
            next_label,
        }),
        action_stack: action_stack
            .into_iter()
            .map(exact_action_from_full)
            .collect(),
    }
}

fn exact_action_from_full(action: WalkerAction) -> ExactWalkerAction {
    match action {
        WalkerAction::EmitLiteral(token) => match ExactBondToken::from_owned(token) {
            Some(token) => ExactWalkerAction::EmitBondToken(token),
            None => unreachable!("empty literal token should not reach stereo exact state"),
        },
        WalkerAction::EmitRingLabel(label) => ExactWalkerAction::EmitRingLabel(label),
        WalkerAction::EmitCloseParen => ExactWalkerAction::EmitCloseParen,
        WalkerAction::EnterAtom {
            atom_idx,
            parent_idx,
        } => ExactWalkerAction::EnterAtom {
            atom_idx,
            parent_idx,
        },
        WalkerAction::ProcessChildren {
            parent_idx,
            child_order,
            next_branch_index,
        } => ExactWalkerAction::ProcessChildren {
            parent_idx,
            child_order,
            next_branch_index,
        },
        WalkerAction::EmitDeferred(_) => {
            unreachable!("atom-stereo exact path never defers tokens")
        }
    }
}

fn push_literal_token(prefix: &mut Arc<str>, token: &str) {
    if token.is_empty() {
        return;
    }
    let mut next = String::with_capacity(prefix.len() + token.len());
    next.push_str(prefix.as_ref());
    next.push_str(token);
    *prefix = Arc::<str>::from(next);
}

fn push_char_token(prefix: &mut Arc<str>, ch: char) {
    let mut next = String::with_capacity(prefix.len() + ch.len_utf8());
    next.push_str(prefix.as_ref());
    next.push(ch);
    *prefix = Arc::<str>::from(next);
}

fn direction_erased_slot(prefix: &str) -> usize {
    prefix.chars().filter(|&ch| ch != '/' && ch != '\\').count()
}

fn stereo_traversal_role_name(role: StereoTraversalRole) -> &'static str {
    match role {
        StereoTraversalRole::TreeOrChain => "tree_or_chain",
        StereoTraversalRole::Branch => "branch",
        StereoTraversalRole::RingOpen => "ring_open",
        StereoTraversalRole::RingClose => "ring_close",
        StereoTraversalRole::Deferred => "deferred",
    }
}

fn directional_token_role(prefix: &str, next_action: Option<&WalkerAction>) -> StereoTraversalRole {
    if matches!(next_action, Some(WalkerAction::EmitRingLabel(_))) {
        StereoTraversalRole::RingClose
    } else if prefix.ends_with(|ch: char| ch.is_ascii_digit()) {
        StereoTraversalRole::RingOpen
    } else if prefix.ends_with('(') {
        StereoTraversalRole::Branch
    } else {
        StereoTraversalRole::TreeOrChain
    }
}

fn directional_marker_side_from_selected_neighbors(
    runtime: &StereoWalkerRuntimeData,
    selected_neighbors: &[isize],
    begin_idx: usize,
    end_idx: usize,
) -> Option<(usize, usize, usize)> {
    let edge = canonical_edge(begin_idx, end_idx);
    runtime
        .edge_to_side_ids
        .get(&edge)?
        .iter()
        .copied()
        .find_map(|side_idx| {
            let side_info = &runtime.side_infos[side_idx];
            let edge_neighbor_idx = if begin_idx == side_info.endpoint_atom_idx {
                end_idx
            } else if end_idx == side_info.endpoint_atom_idx {
                begin_idx
            } else {
                return None;
            };
            (selected_neighbors[side_idx] == edge_neighbor_idx as isize).then_some((
                side_info.component_idx,
                side_idx,
                edge_neighbor_idx,
            ))
        })
}

// Compatibility diagnostic for the old selected-carrier marker path. This is
// deliberately narrower than marker-placement row semantics: a visible marker
// on a complement/bridge candidate may still be valid RDKit writer output.
fn legacy_selected_carrier_directional_marker_side(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    begin_idx: usize,
    end_idx: usize,
) -> Option<(usize, usize, usize)> {
    let selected_neighbors = resolved_selected_neighbors(runtime, state);
    directional_marker_side_from_selected_neighbors(
        runtime,
        &selected_neighbors,
        begin_idx,
        end_idx,
    )
}

fn support_boundary_selected_carrier_directional_marker_side(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    begin_idx: usize,
    end_idx: usize,
) -> Option<(usize, usize, usize)> {
    let selected_neighbors =
        partial_joined_support_boundary_selected_neighbors(runtime, graph, state).ok()?;
    directional_marker_side_from_selected_neighbors(
        runtime,
        &selected_neighbors,
        begin_idx,
        end_idx,
    )
}

fn rdkit_marker_event_trace_rows_for_edge(
    runtime: &StereoWalkerRuntimeData,
    begin_idx: usize,
    end_idx: usize,
) -> Vec<(usize, usize, usize, usize)> {
    let edge = canonical_edge(begin_idx, end_idx);
    runtime
        .edge_to_side_ids
        .get(&edge)
        .into_iter()
        .flatten()
        .filter_map(|&side_idx| {
            let side_info = &runtime.side_infos[side_idx];
            let edge_neighbor_idx = if begin_idx == side_info.endpoint_atom_idx {
                end_idx
            } else if end_idx == side_info.endpoint_atom_idx {
                begin_idx
            } else {
                return None;
            };
            Some((
                side_info.component_idx,
                side_idx,
                side_info.endpoint_atom_idx,
                edge_neighbor_idx,
            ))
        })
        .collect()
}

fn append_rdkit_marker_event_traces_for_edge(
    runtime: &StereoWalkerRuntimeData,
    prefix: &str,
    marker_event_traces: &mut Vec<MarkerEventTrace>,
    begin_idx: isize,
    end_idx: isize,
    marker: Option<char>,
    role: StereoTraversalRole,
) {
    if begin_idx < 0 || end_idx < 0 {
        return;
    }
    let begin = begin_idx as usize;
    let end = end_idx as usize;
    let slot = direction_erased_slot(prefix);
    for (component_idx, side_idx, endpoint_atom_idx, edge_neighbor_idx) in
        rdkit_marker_event_trace_rows_for_edge(runtime, begin, end)
    {
        marker_event_traces.push(MarkerEventTrace {
            slot,
            marker,
            component_idx: component_idx as isize,
            side_idx: side_idx as isize,
            endpoint_atom_idx: endpoint_atom_idx as isize,
            edge_neighbor_idx: edge_neighbor_idx as isize,
            edge_begin_idx: begin_idx,
            edge_end_idx: end_idx,
            role,
        });
    }
}

fn record_rdkit_marker_event_traces_for_edge(
    runtime: &StereoWalkerRuntimeData,
    state: &mut RootedConnectedStereoWalkerStateData,
    begin_idx: isize,
    end_idx: isize,
    marker: Option<char>,
    role: StereoTraversalRole,
) {
    let prefix = state.prefix.clone();
    append_rdkit_marker_event_traces_for_edge(
        runtime,
        prefix.as_ref(),
        Arc::make_mut(&mut state.marker_event_traces),
        begin_idx,
        end_idx,
        marker,
        role,
    );
}

fn record_rdkit_directional_marker_trace(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &mut RootedConnectedStereoWalkerStateData,
    begin_idx: isize,
    end_idx: isize,
    marker_token: &str,
    role: StereoTraversalRole,
) {
    let mut chars = marker_token.chars();
    let Some(marker) = chars.next() else {
        return;
    };
    if chars.next().is_some() || (marker != '/' && marker != '\\') {
        return;
    }

    let (component_idx, side_idx, endpoint_atom_idx, selected_neighbor_idx) =
        if begin_idx >= 0 && end_idx >= 0 {
            let begin = begin_idx as usize;
            let end = end_idx as usize;
            if let Some((component_idx, side_idx, selected_neighbor_idx)) =
                legacy_selected_carrier_directional_marker_side(runtime, state, begin, end)
            {
                let endpoint_atom_idx = runtime.side_infos[side_idx].endpoint_atom_idx;
                (
                    component_idx as isize,
                    side_idx as isize,
                    endpoint_atom_idx as isize,
                    selected_neighbor_idx as isize,
                )
            } else {
                (-1, -1, -1, -1)
            }
        } else {
            (-1, -1, -1, -1)
        };
    let support_boundary_selected_neighbor_idx = if begin_idx >= 0 && end_idx >= 0 {
        let begin = begin_idx as usize;
        let end = end_idx as usize;
        support_boundary_selected_carrier_directional_marker_side(runtime, graph, state, begin, end)
            .map(|(_, _, selected_neighbor_idx)| selected_neighbor_idx as isize)
            .unwrap_or(-1)
    } else {
        -1
    };

    Arc::make_mut(&mut state.directional_marker_traces).push(DirectionalMarkerTrace {
        slot: direction_erased_slot(state.prefix.as_ref()),
        marker,
        component_idx,
        side_idx,
        endpoint_atom_idx,
        selected_neighbor_idx,
        support_boundary_selected_neighbor_idx,
        edge_begin_idx: begin_idx,
        edge_end_idx: end_idx,
        role,
    });
    record_rdkit_marker_event_traces_for_edge(
        runtime,
        state,
        begin_idx,
        end_idx,
        Some(marker),
        role,
    );
}

fn direction_marker_from_literal_token(token: &str) -> Option<char> {
    let mut chars = token.chars();
    let marker = chars.next()?;
    if chars.next().is_some() || (marker != '/' && marker != '\\') {
        return None;
    }
    Some(marker)
}

fn record_rdkit_literal_edge_marker_trace(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &mut RootedConnectedStereoWalkerStateData,
    begin_idx: isize,
    end_idx: isize,
    token: &str,
    role: StereoTraversalRole,
) {
    if direction_marker_from_literal_token(token).is_some() {
        record_rdkit_directional_marker_trace(
            runtime, graph, state, begin_idx, end_idx, token, role,
        );
    } else {
        record_rdkit_marker_event_traces_for_edge(runtime, state, begin_idx, end_idx, None, role);
    }
}

fn marker_event_marker_from_literal_part(part: &Part) -> Option<char> {
    match part {
        Part::Literal(token) => direction_marker_from_literal_token(token),
        Part::Deferred(_) => None,
    }
}

fn push_ring_label(prefix: &mut Arc<str>, label: usize) {
    push_literal_token(prefix, &ring_label_text(label));
}

fn visited_with_marked(visited: &Arc<[bool]>, atom_idx: usize) -> Arc<[bool]> {
    let mut next = visited.as_ref().to_vec();
    debug_assert!(!next[atom_idx]);
    next[atom_idx] = true;
    Arc::<[bool]>::from(next)
}

fn take_pending_for_atom_arc(
    pending: &Arc<PendingRingBuckets>,
    target_atom: usize,
) -> TakenPendingRings {
    match pending.binary_search_by_key(&target_atom, |(atom_idx, _)| *atom_idx) {
        Ok(offset) => {
            let mut next = Arc::unwrap_or_clone(pending.clone());
            let removed = next.remove(offset).1;
            TakenPendingRings {
                pending: Arc::new(next),
                rings: removed,
            }
        }
        Err(_) => TakenPendingRings {
            pending: pending.clone(),
            rings: Vec::new(),
        },
    }
}

fn push_successor_bucket(
    buckets: &mut Vec<(String, Vec<RootedConnectedStereoWalkerStateData>)>,
    token: String,
    successor: RootedConnectedStereoWalkerStateData,
) {
    if let Some((_, states)) = buckets.iter_mut().find(|(existing, _)| *existing == token) {
        if states
            .iter()
            .any(|existing| cmp_stereo_state_structure(existing, &successor) == Ordering::Equal)
        {
            return;
        }
        states.push(successor);
    } else {
        buckets.push((token, vec![successor]));
    }
}

fn cmp_stereo_state_exact(
    left: &RootedConnectedStereoWalkerStateData,
    right: &RootedConnectedStereoWalkerStateData,
) -> Ordering {
    cmp_stereo_state_structure(left, right).then(if Arc::ptr_eq(&left.prefix, &right.prefix) {
        Ordering::Equal
    } else {
        left.prefix.as_ref().cmp(right.prefix.as_ref())
    })
}

fn cmp_stereo_exact_dynamic(
    left: &RootedConnectedStereoExactDynamicData,
    right: &RootedConnectedStereoExactDynamicData,
) -> Ordering {
    left.pending
        .len()
        .cmp(&right.pending.len())
        .then(if Arc::ptr_eq(&left.pending, &right.pending) {
            Ordering::Equal
        } else {
            left.pending.cmp(&right.pending)
        })
        .then(if Arc::ptr_eq(&left.free_labels, &right.free_labels) {
            Ordering::Equal
        } else {
            left.free_labels.cmp(&right.free_labels)
        })
        .then(left.next_label.cmp(&right.next_label))
        .then(left.visited_count.cmp(&right.visited_count))
        .then(if Arc::ptr_eq(&left.visited, &right.visited) {
            Ordering::Equal
        } else {
            left.visited.cmp(&right.visited)
        })
}

fn cmp_stereo_exact_state(
    left: &RootedConnectedStereoExactStateData,
    right: &RootedConnectedStereoExactStateData,
) -> Ordering {
    left.action_stack
        .len()
        .cmp(&right.action_stack.len())
        .then(left.action_stack.cmp(&right.action_stack))
        .then(if Arc::ptr_eq(&left.dynamic, &right.dynamic) {
            Ordering::Equal
        } else {
            cmp_stereo_exact_dynamic(&left.dynamic, &right.dynamic)
        })
        .then(if Arc::ptr_eq(&left.prefix, &right.prefix) {
            Ordering::Equal
        } else {
            left.prefix.as_ref().cmp(right.prefix.as_ref())
        })
}

fn push_exact_atom_stereo_successor(
    successors: &mut Vec<RootedConnectedStereoExactStateData>,
    successor: RootedConnectedStereoExactStateData,
) {
    if successors
        .iter()
        .any(|existing| cmp_stereo_exact_state(existing, &successor) == Ordering::Equal)
    {
        return;
    }
    successors.push(successor);
}

fn cmp_stereo_state_structure(
    left: &RootedConnectedStereoWalkerStateData,
    right: &RootedConnectedStereoWalkerStateData,
) -> Ordering {
    left.action_stack
        .len()
        .cmp(&right.action_stack.len())
        .then(left.action_stack.cmp(&right.action_stack))
        .then(left.pending.len().cmp(&right.pending.len()))
        .then(if Arc::ptr_eq(&left.pending, &right.pending) {
            Ordering::Equal
        } else {
            left.pending.cmp(&right.pending)
        })
        .then(if Arc::ptr_eq(&left.free_labels, &right.free_labels) {
            Ordering::Equal
        } else {
            left.free_labels.cmp(&right.free_labels)
        })
        .then(left.next_label.cmp(&right.next_label))
        .then(
            if Arc::ptr_eq(
                &left.stereo_component_phases,
                &right.stereo_component_phases,
            ) {
                Ordering::Equal
            } else {
                left.stereo_component_phases
                    .cmp(&right.stereo_component_phases)
            },
        )
        .then(
            if Arc::ptr_eq(
                &left.stereo_selected_neighbors,
                &right.stereo_selected_neighbors,
            ) {
                Ordering::Equal
            } else {
                left.stereo_selected_neighbors
                    .cmp(&right.stereo_selected_neighbors)
            },
        )
        .then(
            if Arc::ptr_eq(
                &left.stereo_selected_orientations,
                &right.stereo_selected_orientations,
            ) {
                Ordering::Equal
            } else {
                left.stereo_selected_orientations
                    .cmp(&right.stereo_selected_orientations)
            },
        )
        .then(
            if Arc::ptr_eq(
                &left.stereo_first_emitted_candidates,
                &right.stereo_first_emitted_candidates,
            ) {
                Ordering::Equal
            } else {
                left.stereo_first_emitted_candidates
                    .cmp(&right.stereo_first_emitted_candidates)
            },
        )
        .then(
            if Arc::ptr_eq(
                &left.stereo_component_begin_atoms,
                &right.stereo_component_begin_atoms,
            ) {
                Ordering::Equal
            } else {
                left.stereo_component_begin_atoms
                    .cmp(&right.stereo_component_begin_atoms)
            },
        )
        .then(
            if Arc::ptr_eq(
                &left.committed_component_token_flips,
                &right.committed_component_token_flips,
            ) {
                Ordering::Equal
            } else {
                left.committed_component_token_flips
                    .cmp(&right.committed_component_token_flips)
            },
        )
        .then(
            if Arc::ptr_eq(
                &left.deferred_carrier_choice_constraints,
                &right.deferred_carrier_choice_constraints,
            ) {
                Ordering::Equal
            } else {
                left.deferred_carrier_choice_constraints
                    .cmp(&right.deferred_carrier_choice_constraints)
            },
        )
        .then(
            if Arc::ptr_eq(
                &left.stereo_token_basis_facts,
                &right.stereo_token_basis_facts,
            ) {
                Ordering::Equal
            } else {
                left.stereo_token_basis_facts
                    .cmp(&right.stereo_token_basis_facts)
            },
        )
        .then(left.visited_count.cmp(&right.visited_count))
        .then(if Arc::ptr_eq(&left.visited, &right.visited) {
            Ordering::Equal
        } else {
            left.visited.cmp(&right.visited)
        })
}

fn extend_linear_structural_transitions(
    transitions: &mut Vec<(String, Vec<RootedConnectedStereoWalkerStateData>)>,
    successors: BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>,
) {
    for (token, states) in successors {
        if let Some((_, existing_states)) = transitions
            .iter_mut()
            .find(|(existing_token, _)| *existing_token == token)
        {
            existing_states.extend(states);
        } else {
            transitions.push((token, states));
        }
    }
}

fn finalize_linear_structural_transitions(
    transitions: Vec<(String, Vec<RootedConnectedStereoWalkerStateData>)>,
) -> BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>> {
    let mut out = BTreeMap::new();
    for (token, mut states) in transitions {
        states.sort_by(cmp_stereo_state_structure);
        states.dedup_by(|left, right| cmp_stereo_state_structure(left, right) == Ordering::Equal);
        out.insert(token, states);
    }
    out
}

fn finalize_linear_structural_transitions_vec(
    mut transitions: Vec<(String, Vec<RootedConnectedStereoWalkerStateData>)>,
) -> Vec<(String, Vec<RootedConnectedStereoWalkerStateData>)> {
    for (_, states) in transitions.iter_mut() {
        states.sort_by(cmp_stereo_state_structure);
        states.dedup_by(|left, right| cmp_stereo_state_structure(left, right) == Ordering::Equal);
    }
    transitions.sort_by(|left, right| left.0.cmp(&right.0));
    transitions
}

fn finalize_exact_stereo_successors(
    mut successors: Vec<RootedConnectedStereoWalkerStateData>,
) -> Vec<RootedConnectedStereoWalkerStateData> {
    successors.sort_by(cmp_stereo_state_exact);
    successors.dedup_by(|left, right| cmp_stereo_state_exact(left, right) == Ordering::Equal);
    successors
}

fn finalize_exact_atom_stereo_successors(
    mut successors: Vec<RootedConnectedStereoExactStateData>,
) -> Vec<RootedConnectedStereoExactStateData> {
    successors.sort_by(cmp_stereo_exact_state);
    successors.dedup_by(|left, right| cmp_stereo_exact_state(left, right) == Ordering::Equal);
    successors
}

fn flatten_exact_stereo_successor_groups(
    successors: BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>,
) -> Vec<RootedConnectedStereoWalkerStateData> {
    let mut out = Vec::new();
    for group in successors.into_values() {
        out.extend(group);
    }
    finalize_exact_stereo_successors(out)
}

fn drain_exact_linear_stereo_actions(state: &mut RootedConnectedStereoWalkerStateData) {
    loop {
        let Some(action) = state.action_stack.last() else {
            return;
        };
        match action {
            WalkerAction::EmitLiteral(token) => {
                let token = token.clone();
                state.action_stack.pop();
                push_literal_token(&mut state.prefix, &token);
            }
            WalkerAction::EmitRingLabel(label) => {
                let label = *label;
                state.action_stack.pop();
                push_ring_label(&mut state.prefix, label);
            }
            WalkerAction::EmitCloseParen => {
                state.action_stack.pop();
                push_char_token(&mut state.prefix, ')');
            }
            _ => return,
        }
    }
}

fn drain_exact_linear_atom_stereo_actions(state: &mut RootedConnectedStereoExactStateData) {
    loop {
        let Some(action) = state.action_stack.last() else {
            return;
        };
        match action {
            ExactWalkerAction::EmitBondToken(token) => {
                let token = token.clone();
                state.action_stack.pop();
                push_literal_token(&mut state.prefix, token.as_str());
            }
            ExactWalkerAction::EmitRingLabel(label) => {
                let label = *label;
                state.action_stack.pop();
                push_ring_label(&mut state.prefix, label);
            }
            ExactWalkerAction::EmitCloseParen => {
                state.action_stack.pop();
                push_char_token(&mut state.prefix, ')');
            }
            _ => return,
        }
    }
}

fn format_hydrogen_count(count: usize) -> String {
    if count == 0 {
        String::new()
    } else if count == 1 {
        "H".to_owned()
    } else {
        format!("H{count}")
    }
}

fn format_charge(charge: i32) -> String {
    if charge == 0 {
        String::new()
    } else if charge == 1 {
        "+".to_owned()
    } else if charge == -1 {
        "-".to_owned()
    } else if charge > 0 {
        format!("+{charge}")
    } else {
        format!("-{}", charge.abs())
    }
}

fn element_symbol(atomic_num: usize) -> PyResult<&'static str> {
    ELEMENT_SYMBOLS.get(atomic_num).copied().ok_or_else(|| {
        PyValueError::new_err(format!(
            "Unsupported atomic number for prepared graph symbol lookup: {atomic_num}"
        ))
    })
}

fn prepared_atom_symbol(graph: &PreparedSmilesGraphData, atom_idx: usize) -> PyResult<String> {
    let symbol = element_symbol(graph.atom_atomic_numbers[atom_idx])?;
    if graph.atom_is_aromatic[atom_idx] && !graph.writer_kekule_smiles {
        let lowered = symbol.to_ascii_lowercase();
        if AROMATIC_SUBSET.contains(&lowered.as_str()) {
            return Ok(lowered);
        }
    }
    Ok(symbol.to_owned())
}

fn permutation_parity(reference_order: &[isize], emitted_order: &[isize]) -> PyResult<usize> {
    if reference_order.len() != emitted_order.len() {
        return Err(PyValueError::new_err(
            "Stereo neighbor order length mismatch",
        ));
    }
    let reference_set = reference_order.iter().copied().collect::<BTreeSet<_>>();
    let emitted_set = emitted_order.iter().copied().collect::<BTreeSet<_>>();
    if reference_set != emitted_set {
        return Err(PyValueError::new_err(
            "Stereo neighbor order membership mismatch",
        ));
    }

    let reference_index = reference_order
        .iter()
        .enumerate()
        .map(|(index, &neighbor)| (neighbor, index))
        .collect::<BTreeMap<_, _>>();
    let permutation = emitted_order
        .iter()
        .map(|neighbor| {
            reference_index
                .get(neighbor)
                .copied()
                .ok_or_else(|| PyValueError::new_err("Stereo neighbor order membership mismatch"))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let mut inversions = 0usize;
    for (index, &left) in permutation.iter().enumerate() {
        for &right in &permutation[index + 1..] {
            if left > right {
                inversions += 1;
            }
        }
    }
    Ok(inversions % 2)
}

fn stereo_neighbor_order(
    graph: &PreparedSmilesGraphData,
    atom_idx: usize,
    parent_idx: Option<usize>,
    ring_neighbor_order: &[usize],
    child_order: &[usize],
) -> PyResult<Vec<isize>> {
    let hydrogen_count =
        graph.atom_explicit_h_counts[atom_idx] + graph.atom_implicit_h_counts[atom_idx];
    if hydrogen_count > 1 {
        return Err(PyValueError::new_err(
            "Tetrahedral stereo currently supports at most one hydrogen ligand",
        ));
    }

    let mut emitted = Vec::new();
    if let Some(parent_idx) = parent_idx {
        emitted.push(parent_idx as isize);
    }
    if hydrogen_count == 1 {
        emitted.push(HYDROGEN_NEIGHBOR);
    }
    emitted.extend(ring_neighbor_order.iter().map(|&value| value as isize));
    emitted.extend(child_order.iter().map(|&value| value as isize));
    Ok(emitted)
}

fn stereo_atom_token(
    graph: &PreparedSmilesGraphData,
    atom_idx: usize,
    emitted_neighbor_order: &[isize],
) -> PyResult<String> {
    if !graph.writer_do_isomeric_smiles {
        return Ok(graph.atom_tokens[atom_idx].clone());
    }
    let chiral_tag = graph.atom_chiral_tags[atom_idx].as_str();
    if chiral_tag == "CHI_UNSPECIFIED" {
        return Ok(graph.atom_tokens[atom_idx].clone());
    }
    if !SUPPORTED_CHIRAL_TAGS.contains(&chiral_tag) {
        return Err(PyValueError::new_err(format!(
            "Unsupported chiral tag for rooted stereo emission: {chiral_tag}"
        )));
    }

    let hydrogen_count =
        graph.atom_explicit_h_counts[atom_idx] + graph.atom_implicit_h_counts[atom_idx];
    let mut reference_order = graph.atom_stereo_neighbor_orders[atom_idx]
        .iter()
        .map(|&neighbor| neighbor as isize)
        .collect::<Vec<_>>();
    if hydrogen_count == 1 {
        reference_order.push(HYDROGEN_NEIGHBOR);
    }

    let parity = permutation_parity(&reference_order, emitted_neighbor_order)?;
    let use_single_at = (chiral_tag == "CHI_TETRAHEDRAL_CCW" && parity == 0)
        || (chiral_tag == "CHI_TETRAHEDRAL_CW" && parity == 1);
    let stereo_mark = if use_single_at { "@" } else { "@@" };

    let mut parts = vec!["[".to_owned()];
    let isotope = graph.atom_isotopes[atom_idx];
    if isotope != 0 {
        parts.push(isotope.to_string());
    }
    parts.push(prepared_atom_symbol(graph, atom_idx)?);
    parts.push(stereo_mark.to_owned());
    parts.push(format_hydrogen_count(hydrogen_count));
    parts.push(format_charge(graph.atom_formal_charges[atom_idx]));
    let atom_map_number = if graph.writer_ignore_atom_map_numbers {
        0
    } else {
        graph.atom_map_numbers[atom_idx]
    };
    if atom_map_number != 0 {
        parts.push(format!(":{atom_map_number}"));
    }
    parts.push("]".to_owned());
    Ok(parts.concat())
}

fn validate_root_idx(graph: &PreparedSmilesGraphData, root_idx: isize) -> PyResult<usize> {
    if graph.atom_count() == 0 {
        return Ok(0);
    }
    if root_idx < 0 || root_idx as usize >= graph.atom_count() {
        return Err(PyIndexError::new_err("root_idx out of range"));
    }
    Ok(root_idx as usize)
}

fn check_supported_stereo_writer_surface(graph: &PreparedSmilesGraphData) -> PyResult<()> {
    if graph.surface_kind != CONNECTED_STEREO_SURFACE {
        return Err(PyValueError::new_err(format!(
            "Expected surface_kind={CONNECTED_STEREO_SURFACE:?}, got {:?}",
            graph.surface_kind
        )));
    }
    for (atom_idx, chiral_tag) in graph.atom_chiral_tags.iter().enumerate() {
        if !SUPPORTED_CHIRAL_TAGS.contains(&chiral_tag.as_str()) {
            return Err(PyValueError::new_err(format!(
                "Unsupported chiral tag at atom {atom_idx}: {chiral_tag}"
            )));
        }
    }
    Ok(())
}

fn ordered_neighbor_groups(
    graph: &PreparedSmilesGraphData,
    atom_idx: usize,
    visited: &[bool],
) -> Vec<Vec<usize>> {
    let mut seeds = graph
        .neighbors_of(atom_idx)
        .iter()
        .copied()
        .filter(|&neighbor_idx| !visited[neighbor_idx])
        .collect::<Vec<_>>();
    if seeds.is_empty() {
        return Vec::new();
    }
    seeds.sort_unstable();
    if seeds.len() == 1 {
        return vec![seeds];
    }

    let mut groups_with_mins = Vec::<(usize, Vec<usize>)>::new();
    let mut seen = vec![false; graph.atom_count()];
    let mut stack = Vec::new();
    for &seed in &seeds {
        if seen[seed] {
            continue;
        }
        seen[seed] = true;
        stack.push(seed);
        let mut component_min = seed;
        let mut group = vec![seed];

        while let Some(current) = stack.pop() {
            if current < component_min {
                component_min = current;
            }
            for &neighbor_idx in graph.neighbors_of(current) {
                if neighbor_idx == atom_idx || visited[neighbor_idx] || seen[neighbor_idx] {
                    continue;
                }
                seen[neighbor_idx] = true;
                if seeds.binary_search(&neighbor_idx).is_ok() {
                    group.push(neighbor_idx);
                }
                stack.push(neighbor_idx);
            }
        }

        group.sort_unstable();
        groups_with_mins.push((component_min, group));
    }

    groups_with_mins.sort_by_key(|(component_min, _)| *component_min);
    groups_with_mins
        .into_iter()
        .map(|(_component_min, group)| group)
        .collect()
}

fn for_each_cartesian_choice<F>(groups: &[Vec<usize>], f: &mut F)
where
    F: FnMut(&[usize]),
{
    fn recurse<F>(groups: &[Vec<usize>], group_idx: usize, current: &mut Vec<usize>, f: &mut F)
    where
        F: FnMut(&[usize]),
    {
        if group_idx == groups.len() {
            f(current);
            return;
        }

        for &choice in &groups[group_idx] {
            current.push(choice);
            recurse(groups, group_idx + 1, current, f);
            current.pop();
        }
    }

    let mut current = Vec::with_capacity(groups.len());
    recurse(groups, 0, &mut current, f);
    if groups.is_empty() {
        f(&[]);
    }
}

fn permutations_copy_distinct<T, F>(items: &[T], f: &mut F)
where
    T: Copy,
    F: FnMut(&[T]),
{
    fn recurse<T, F>(items: &mut [T], start: usize, f: &mut F)
    where
        T: Copy,
        F: FnMut(&[T]),
    {
        if start == items.len() {
            f(items);
            return;
        }
        for idx in start..items.len() {
            items.swap(start, idx);
            recurse(items, start + 1, f);
            items.swap(start, idx);
        }
    }

    let mut current = items.to_vec();
    recurse(&mut current, 0, f);
    if items.is_empty() {
        f(&[]);
    }
}

fn small_permutation_count(len: usize) -> usize {
    match len {
        0 | 1 => 1,
        2 => 2,
        3 => 6,
        4 => 24,
        5 => 120,
        6 => 720,
        _ => (2..=len).product(),
    }
}

fn insert_sorted(labels: &mut Vec<usize>, label: usize) {
    match labels.binary_search(&label) {
        Ok(offset) | Err(offset) => labels.insert(offset, label),
    }
}

fn allocate_label(free_labels: &mut Vec<usize>, next_label: &mut usize) -> usize {
    if !free_labels.is_empty() {
        free_labels.remove(0)
    } else {
        let label = *next_label;
        *next_label += 1;
        label
    }
}

fn with_component_phase(
    component_phases: &[i8],
    component_idx: usize,
    phase: i8,
) -> PyResult<Vec<i8>> {
    let existing = component_phases[component_idx];
    if existing == phase {
        return Ok(component_phases.to_vec());
    }
    if existing != UNKNOWN_COMPONENT_PHASE {
        return Err(PyValueError::new_err(
            "Stereo component phase was committed inconsistently",
        ));
    }
    let mut updated = component_phases.to_vec();
    updated[component_idx] = phase;
    Ok(updated)
}

fn with_component_begin_atom(
    component_begin_atoms: &[isize],
    component_idx: usize,
    atom_idx: usize,
) -> PyResult<Vec<isize>> {
    let existing = component_begin_atoms[component_idx];
    if existing == atom_idx as isize {
        return Ok(component_begin_atoms.to_vec());
    }
    if existing != -1 {
        return Err(PyValueError::new_err(
            "Stereo component begin atom was committed inconsistently",
        ));
    }
    let mut updated = component_begin_atoms.to_vec();
    updated[component_idx] = atom_idx as isize;
    Ok(updated)
}

fn apply_deferred_component_phase_fact(
    component_phases: &[i8],
    component_begin_atoms: &[isize],
    fact: DeferredComponentPhaseFact,
) -> PyResult<(Vec<i8>, Vec<isize>)> {
    let mut updated_phases = component_phases.to_vec();
    updated_phases[fact.component_idx] = UNKNOWN_COMPONENT_PHASE;
    let updated_begin_atoms =
        apply_component_begin_atom_fact(component_begin_atoms, fact.begin_atom_fact())?;
    Ok((updated_phases, updated_begin_atoms))
}

fn apply_component_begin_atom_fact(
    component_begin_atoms: &[isize],
    fact: ComponentBeginAtomFact,
) -> PyResult<Vec<isize>> {
    with_component_begin_atom(
        component_begin_atoms,
        fact.component_idx,
        fact.begin_atom_idx,
    )
}

fn apply_component_phase_commit_fact(
    component_phases: &[i8],
    fact: ComponentPhaseCommitFact,
) -> PyResult<Vec<i8>> {
    with_component_phase(component_phases, fact.component_idx, fact.phase)
}

fn component_phases_after_edge(
    graph: &PreparedSmilesGraphData,
    stereo_component_ids: &[isize],
    isolated_components: &[bool],
    component_phases: &[i8],
    component_begin_atoms: &[isize],
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<(Vec<i8>, Vec<isize>)> {
    let Some(bond_idx) = graph.bond_index(begin_idx, end_idx) else {
        return Err(PyKeyError::new_err(format!(
            "No bond between atoms {begin_idx} and {end_idx}"
        )));
    };
    let component_idx = stereo_component_ids[bond_idx];
    if component_idx < 0 || !is_stereo_double_bond(graph, bond_idx) {
        return Ok((component_phases.to_vec(), component_begin_atoms.to_vec()));
    }
    let component_idx = component_idx as usize;
    if component_phases[component_idx] != UNKNOWN_COMPONENT_PHASE {
        return Ok((component_phases.to_vec(), component_begin_atoms.to_vec()));
    }
    if !isolated_components[component_idx] && component_begin_atoms[component_idx] >= 0 {
        return Ok((component_phases.to_vec(), component_begin_atoms.to_vec()));
    }

    let stored_begin_idx = graph.bond_begin_atom_indices[bond_idx];
    let stored_end_idx = graph.bond_end_atom_indices[bond_idx];
    let stereo_kind = graph.bond_stereo_kinds[bond_idx].as_str();
    let phase = if (begin_idx, end_idx) == (stored_begin_idx, stored_end_idx)
        || CIS_STEREO_BOND_KINDS.contains(&stereo_kind)
    {
        STORED_COMPONENT_PHASE
    } else if TRANS_STEREO_BOND_KINDS.contains(&stereo_kind) {
        FLIPPED_COMPONENT_PHASE
    } else {
        return Err(PyValueError::new_err(format!(
            "Unsupported stereo bond kind: {stereo_kind}"
        )));
    };
    let updated_begin_atoms = if component_begin_atoms[component_idx] >= 0 {
        component_begin_atoms.to_vec()
    } else {
        with_component_begin_atom(component_begin_atoms, component_idx, begin_idx)?
    };
    Ok((
        with_component_phase(component_phases, component_idx, phase)?,
        updated_begin_atoms,
    ))
}

fn force_known_begin_side_selection(
    runtime: &StereoWalkerRuntimeData,
    component_phases: &[i8],
    component_begin_atoms: &[isize],
    selected_neighbors: &[isize],
    selected_orientations: &[i8],
) -> (Vec<isize>, Vec<i8>) {
    let mut updated_neighbors = selected_neighbors.to_vec();
    let mut updated_orientations = selected_orientations.to_vec();

    for component_idx in 0..runtime.isolated_components.len() {
        if !runtime.isolated_components[component_idx] {
            continue;
        }
        if component_phases[component_idx] == UNKNOWN_COMPONENT_PHASE {
            continue;
        }
        let begin_atom_idx = component_begin_atoms[component_idx];
        if begin_atom_idx < 0 {
            continue;
        }
        let Some(begin_side_idx) = runtime.side_ids_by_component[component_idx]
            .iter()
            .copied()
            .find(|&side_idx| {
                runtime.side_infos[side_idx].endpoint_atom_idx == begin_atom_idx as usize
            })
        else {
            continue;
        };
        let begin_side = &runtime.side_infos[begin_side_idx];
        if begin_side.candidate_neighbors.len() != 1 || updated_neighbors[begin_side_idx] >= 0 {
            continue;
        }
        updated_neighbors[begin_side_idx] = begin_side.candidate_neighbors[0] as isize;
        if begin_atom_idx as usize == runtime.root_idx {
            updated_orientations[begin_side_idx] = AFTER_ATOM_EDGE_ORIENTATION;
        }
    }

    (updated_neighbors, updated_orientations)
}

fn eager_component_phases_for_child_order(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    component_phases: &[i8],
    component_begin_atoms: &[isize],
    selected_neighbors: &[isize],
    parent_idx: usize,
    child_order: &[usize],
) -> PyResult<(Vec<i8>, Vec<isize>)> {
    let mut updated_phases = component_phases.to_vec();
    let mut updated_begin_atoms = component_begin_atoms.to_vec();

    for side_info in &runtime.side_infos {
        if !runtime.isolated_components[side_info.component_idx] {
            continue;
        }
        if side_info.endpoint_atom_idx != parent_idx {
            continue;
        }
        let other_endpoint_idx = side_info.other_endpoint_atom_idx;
        if !child_order.contains(&other_endpoint_idx) {
            continue;
        }
        let (next_phases, next_begin_atoms) = component_phases_after_edge(
            graph,
            &runtime.stereo_component_ids,
            &runtime.isolated_components,
            &updated_phases,
            &updated_begin_atoms,
            parent_idx,
            other_endpoint_idx,
        )?;
        let (next_phases, next_begin_atoms) = defer_component_phase_for_unresolved_begin_side(
            runtime,
            graph,
            &next_phases,
            &next_begin_atoms,
            selected_neighbors,
            parent_idx,
            other_endpoint_idx,
        )?;
        updated_phases = next_phases;
        updated_begin_atoms = next_begin_atoms;
    }

    Ok((updated_phases, updated_begin_atoms))
}

fn eager_begin_side_child_order_state(
    runtime: &StereoWalkerRuntimeData,
    component_begin_atoms: &[isize],
    selected_neighbors: &[isize],
    selected_orientations: &[i8],
    first_emitted_candidates: &[isize],
    parent_idx: usize,
    child_order: &[usize],
) -> (Vec<isize>, Vec<i8>, Vec<isize>) {
    let mut updated_neighbors = selected_neighbors.to_vec();
    let mut updated_orientations = selected_orientations.to_vec();
    let mut updated_first_candidates = first_emitted_candidates.to_vec();

    for (component_idx, &begin_atom_idx) in component_begin_atoms
        .iter()
        .enumerate()
        .take(runtime.isolated_components.len())
    {
        if !runtime.isolated_components[component_idx] {
            continue;
        }
        if begin_atom_idx != parent_idx as isize {
            continue;
        }
        let Some(begin_side_idx) = runtime.side_ids_by_component[component_idx]
            .iter()
            .copied()
            .find(|&side_idx| runtime.side_infos[side_idx].endpoint_atom_idx == parent_idx)
        else {
            continue;
        };
        let begin_side = &runtime.side_infos[begin_side_idx];
        if begin_side.candidate_neighbors.len() != 2 || updated_neighbors[begin_side_idx] >= 0 {
            continue;
        }
        let Some(first_neighbor_idx) = child_order
            .iter()
            .copied()
            .find(|neighbor_idx| begin_side.candidate_neighbors.contains(neighbor_idx))
        else {
            continue;
        };
        updated_neighbors[begin_side_idx] = first_neighbor_idx as isize;
        updated_orientations[begin_side_idx] = AFTER_ATOM_EDGE_ORIENTATION;
        if updated_first_candidates[begin_side_idx] < 0 {
            updated_first_candidates[begin_side_idx] = first_neighbor_idx as isize;
        }
    }

    (
        updated_neighbors,
        updated_orientations,
        updated_first_candidates,
    )
}

fn deferred_component_phase_fact_for_unresolved_begin_side(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    component_begin_atoms: &[isize],
    selected_neighbors: &[isize],
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<Option<DeferredComponentPhaseFact>> {
    let Some(bond_idx) = graph.bond_index(begin_idx, end_idx) else {
        return Ok(None);
    };
    let component_idx = runtime.stereo_component_ids[bond_idx];
    if component_idx < 0 || !is_stereo_double_bond(graph, bond_idx) {
        return Ok(None);
    }
    let component_idx = component_idx as usize;
    if runtime.isolated_components[component_idx] {
        return Ok(None);
    }

    let Some(begin_side_idx) = runtime.side_ids_by_component[component_idx]
        .iter()
        .copied()
        .find(|&side_idx| runtime.side_infos[side_idx].endpoint_atom_idx == begin_idx)
    else {
        return Ok(None);
    };
    let begin_side = &runtime.side_infos[begin_side_idx];
    if begin_side.candidate_neighbors.len() <= 1 || selected_neighbors[begin_side_idx] >= 0 {
        return Ok(None);
    }
    if component_begin_atoms[component_idx] >= 0
        && component_begin_atoms[component_idx] != begin_idx as isize
    {
        return Ok(None);
    }

    Ok(Some(DeferredComponentPhaseFact {
        component_idx,
        begin_atom_idx: begin_idx,
    }))
}

fn defer_component_phase_for_unresolved_begin_side(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    component_phases: &[i8],
    component_begin_atoms: &[isize],
    selected_neighbors: &[isize],
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<(Vec<i8>, Vec<isize>)> {
    let Some(fact) = deferred_component_phase_fact_for_unresolved_begin_side(
        runtime,
        graph,
        component_begin_atoms,
        selected_neighbors,
        begin_idx,
        end_idx,
    )?
    else {
        return Ok((component_phases.to_vec(), component_begin_atoms.to_vec()));
    };
    apply_deferred_component_phase_fact(component_phases, component_begin_atoms, fact)
}

fn component_phase_from_selected_begin_side_token(token: &str) -> PyResult<i8> {
    match token {
        "/" => Ok(STORED_COMPONENT_PHASE),
        "\\" => Ok(FLIPPED_COMPONENT_PHASE),
        token => Err(PyValueError::new_err(format!(
            "Unsupported selected begin-side directional token: {token:?}"
        ))),
    }
}

fn selected_begin_side_component_phase(
    runtime: &StereoWalkerRuntimeData,
    component_begin_atoms: &[isize],
    selected_neighbors: &[isize],
    component_idx: usize,
) -> PyResult<Option<i8>> {
    let begin_atom_idx = component_begin_atoms
        .get(component_idx)
        .copied()
        .unwrap_or(-1);
    if begin_atom_idx < 0 {
        return Ok(None);
    }
    let Some(begin_side_idx) = runtime.side_ids_by_component[component_idx]
        .iter()
        .copied()
        .find(|&side_idx| {
            runtime.side_infos[side_idx].endpoint_atom_idx == begin_atom_idx as usize
        })
    else {
        return Ok(None);
    };
    let selected_neighbor_idx = selected_neighbors[begin_side_idx];
    if selected_neighbor_idx < 0 {
        return Ok(None);
    }
    let selected_token = candidate_base_token(
        &runtime.side_infos[begin_side_idx],
        selected_neighbor_idx as usize,
    )?;
    Ok(Some(component_phase_from_selected_begin_side_token(
        &selected_token,
    )?))
}

fn component_phase_commit_fact_from_selected_begin_side(
    runtime: &StereoWalkerRuntimeData,
    component_begin_atoms: &[isize],
    selected_neighbors: &[isize],
    begin_idx: usize,
    component_idx: usize,
) -> PyResult<Option<ComponentPhaseCommitFact>> {
    if runtime.isolated_components[component_idx]
        || component_begin_atoms
            .get(component_idx)
            .copied()
            .unwrap_or(-1)
            != begin_idx as isize
    {
        return Ok(None);
    }
    Ok(selected_begin_side_component_phase(
        runtime,
        component_begin_atoms,
        selected_neighbors,
        component_idx,
    )?
    .map(|phase| ComponentPhaseCommitFact {
        component_idx,
        phase,
    }))
}

fn component_phase_commit_facts_from_deferred_part(
    runtime: &StereoWalkerRuntimeData,
    component_phases: &[i8],
    component_begin_atoms: &[isize],
    selected_neighbors: &[isize],
    begin_idx: usize,
    part: &Part,
) -> PyResult<Vec<ComponentPhaseCommitFact>> {
    let Part::Deferred(deferred) = part else {
        return Ok(Vec::new());
    };
    let mut facts = Vec::new();
    for component_token in deferred.component_tokens.iter() {
        let component_idx = component_token.component_idx;
        if component_phases[component_idx] != UNKNOWN_COMPONENT_PHASE {
            continue;
        }

        if let Some(fact) = component_phase_commit_fact_from_selected_begin_side(
            runtime,
            component_begin_atoms,
            selected_neighbors,
            begin_idx,
            component_idx,
        )? {
            facts.push(fact);
        }
    }
    Ok(facts)
}

fn commit_deferred_component_phase_facts_from_selected_begin_sides(
    runtime: &StereoWalkerRuntimeData,
    component_phases: &[i8],
    component_begin_atoms: &[isize],
    selected_neighbors: &[isize],
    begin_idx: usize,
    part: &Part,
) -> PyResult<Vec<i8>> {
    let mut updated_phases = component_phases.to_vec();
    for fact in component_phase_commit_facts_from_deferred_part(
        runtime,
        component_phases,
        component_begin_atoms,
        selected_neighbors,
        begin_idx,
        part,
    )? {
        updated_phases = apply_component_phase_commit_fact(&updated_phases, fact)?;
    }
    Ok(updated_phases)
}

fn candidate_base_token(side_info: &StereoSideInfo, neighbor_idx: usize) -> PyResult<String> {
    for (offset, &candidate_neighbor) in side_info.candidate_neighbors.iter().enumerate() {
        if candidate_neighbor == neighbor_idx {
            return Ok(side_info.candidate_base_tokens[offset].clone());
        }
    }
    Err(PyKeyError::new_err(format!(
        "Neighbor {neighbor_idx} is not a stereo carrier candidate for endpoint {}",
        side_info.endpoint_atom_idx
    )))
}

fn emitted_candidate_token(
    side_info: &StereoSideInfo,
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<String> {
    if begin_idx == side_info.endpoint_atom_idx {
        return candidate_base_token(side_info, end_idx);
    }
    if end_idx == side_info.endpoint_atom_idx {
        return flip_direction_token(&candidate_base_token(side_info, begin_idx)?);
    }
    Err(PyKeyError::new_err(
        "Emitted edge does not match the stereo side",
    ))
}

fn component_reference_token_for_emitted_edge(
    side_info: &StereoSideInfo,
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<String> {
    if begin_idx == side_info.endpoint_atom_idx {
        return candidate_base_token(side_info, end_idx);
    }
    if end_idx == side_info.endpoint_atom_idx {
        return candidate_base_token(side_info, begin_idx);
    }
    Err(PyKeyError::new_err(
        "Emitted edge does not match the stereo side",
    ))
}

fn component_reference_tokens_for_emitted_edge(
    side_info: &StereoSideInfo,
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<Vec<String>> {
    // A shared visible marker can be read against either the component-local
    // endpoint basis or the actual emitted character; marker rows decide which
    // interpretation remains viable for a concrete walker state.
    let mut tokens = BTreeSet::new();
    tokens.insert(component_reference_token_for_emitted_edge(
        side_info, begin_idx, end_idx,
    )?);
    tokens.insert(emitted_candidate_token(side_info, begin_idx, end_idx)?);
    Ok(tokens.into_iter().collect())
}

fn side_has_only_aromatic_carrier_edges(
    graph: &PreparedSmilesGraphData,
    side_info: &StereoSideInfo,
) -> bool {
    !side_info.candidate_neighbors.is_empty()
        && side_info.candidate_neighbors.iter().all(|&neighbor_idx| {
            graph
                .bond_index(side_info.endpoint_atom_idx, neighbor_idx)
                .map(|bond_idx| graph.bond_kinds[bond_idx] == "AROMATIC")
                .unwrap_or(false)
        })
}

fn isolated_component_token_basis_fact_from_row_state(
    context: &StereoEdgeEmissionContext<'_>,
    component_begin_atoms: &[isize],
    selected_neighbors: &[isize],
    side_info: &StereoSideInfo,
) -> PyResult<Option<StereoTokenBasisFact>> {
    if side_info.candidate_neighbors.len() != 1 {
        return Ok(None);
    }
    let component_idx = side_info.component_idx;
    let begin_atom_idx = component_begin_atoms
        .get(component_idx)
        .copied()
        .unwrap_or(-1);
    if begin_atom_idx < 0 || begin_atom_idx as usize == side_info.endpoint_atom_idx {
        return Ok(None);
    }

    let Some(begin_side_idx) = context
        .side_ids_by_component
        .get(component_idx)
        .into_iter()
        .flatten()
        .copied()
        .find(|&candidate_side_idx| {
            let candidate_side = &context.side_infos[candidate_side_idx];
            candidate_side.endpoint_atom_idx == begin_atom_idx as usize
                && candidate_side.candidate_neighbors.len() == 2
                && side_has_only_aromatic_carrier_edges(context.graph, candidate_side)
        })
    else {
        return Ok(None);
    };

    let begin_selected_neighbor = selected_neighbors[begin_side_idx];
    if begin_selected_neighbor < 0 {
        return Ok(None);
    }
    let boundary_facts =
        support_boundary_facts_from_edge_state(context, selected_neighbors, &[], &[], &[])?;
    let Some(available_begin_neighbors) = available_carrier_neighbors_from_support_boundary(
        context,
        &boundary_facts,
        begin_side_idx,
    )?
    else {
        return Ok(None);
    };
    if !available_begin_neighbors.contains(&(begin_selected_neighbor as usize)) {
        return Ok(None);
    }
    let selected_token = candidate_base_token(
        &context.side_infos[begin_side_idx],
        begin_selected_neighbor as usize,
    )?;
    Ok(Some(StereoTokenBasisFact {
        runtime_component_idx: component_idx,
        selected_begin_token: StereoDirectionToken::from_str(&selected_token)?,
    }))
}

fn updated_isolated_component_token_basis_facts_from_row_state(
    context: &StereoEdgeEmissionContext<'_>,
    component_begin_atoms: &[isize],
    selected_neighbors: &[isize],
    side_ids: &[usize],
    token_basis_facts: &[Option<StereoTokenBasisFact>],
) -> PyResult<Vec<Option<StereoTokenBasisFact>>> {
    let mut updated_token_basis_facts = token_basis_facts.to_vec();
    for &side_idx in side_ids {
        let side_info = &context.side_infos[side_idx];
        let Some(basis_fact) = isolated_component_token_basis_fact_from_row_state(
            context,
            component_begin_atoms,
            selected_neighbors,
            side_info,
        )?
        else {
            continue;
        };
        let fact_slot = &mut updated_token_basis_facts[side_info.component_idx];
        if let Some(existing_fact) = fact_slot {
            if *existing_fact != basis_fact {
                return Err(PyValueError::new_err(format!(
                    "Conflicting isolated token-basis facts for runtime component {}",
                    side_info.component_idx
                )));
            }
        } else {
            *fact_slot = Some(basis_fact);
        }
    }
    Ok(updated_token_basis_facts)
}

fn carrier_facts_by_component_from_edge_state(
    context: &StereoEdgeEmissionContext<'_>,
    selected_neighbors: &[isize],
    deferred_carrier_choice_constraints: &[DeferredCarrierChoiceConstraint],
) -> PyResult<Vec<Vec<StereoConstraintFact>>> {
    let mut facts_by_component = vec![Vec::new(); context.constraint_model.component_count()];
    for (side_idx, &neighbor_idx) in selected_neighbors.iter().enumerate() {
        if neighbor_idx < 0 {
            continue;
        }
        let Some(component_idx) = context.constraint_model.component_for_side(side_idx) else {
            continue;
        };
        facts_by_component[component_idx].push(StereoConstraintFact::CarrierSelected {
            side_idx,
            neighbor_idx: neighbor_idx as usize,
        });
    }
    for constraint in deferred_carrier_choice_constraints {
        let Some(component_idx) = context
            .constraint_model
            .component_for_side(constraint.side_idx)
        else {
            return Err(PyValueError::new_err(
                "deferred carrier-choice constraint references unknown side",
            ));
        };
        facts_by_component[component_idx].push(StereoConstraintFact::CarrierSelectionBlocked {
            side_idx: constraint.side_idx,
            neighbor_idx: constraint.deferred_neighbor_idx,
        });
    }
    Ok(facts_by_component)
}

fn available_carrier_neighbors_for_support_boundary_model(
    model: &StereoConstraintModel,
    boundary_facts: &StereoSupportBoundaryFacts,
    component_idx: usize,
    side_idx: usize,
) -> PyResult<Vec<usize>> {
    if boundary_facts
        .carrier_facts_by_component
        .get(component_idx)
        .is_none()
    {
        return Err(PyValueError::new_err(
            "support-boundary facts do not match the constraint model",
        ));
    }
    let constraint_state =
        boundary_facts.constraint_state_for_model(model, StereoConstraintLayer::Semantic)?;
    let Some(remaining_assignment_ids) = constraint_state
        .carrier_assignment_state
        .remaining_by_component
        .get(component_idx)
    else {
        return Err(PyValueError::new_err(
            "support-boundary constraint state does not match the edge-emission constraint model",
        ));
    };
    let token_phase_assignment_ids = constraint_state
        .token_phase_remaining_by_component
        .get(component_idx)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut available_assignment_ids = remaining_assignment_ids.clone();
    if let Some(marker_row_assignment_ids) =
        marker_row_neighbor_assignment_ids_from_support_boundary(
            model,
            boundary_facts,
            component_idx,
            token_phase_assignment_ids,
        )?
    {
        let marker_row_assignment_ids = marker_row_assignment_ids
            .into_iter()
            .collect::<BTreeSet<_>>();
        available_assignment_ids
            .retain(|assignment_id| marker_row_assignment_ids.contains(assignment_id));
    }
    Ok(model.available_neighbors_for_assignment_ids(
        component_idx,
        side_idx,
        &available_assignment_ids,
    ))
}

fn available_carrier_neighbors_from_support_boundary(
    context: &StereoEdgeEmissionContext<'_>,
    boundary_facts: &StereoSupportBoundaryFacts,
    side_idx: usize,
) -> PyResult<Option<Vec<usize>>> {
    let Some(component_idx) = context.constraint_model.component_for_side(side_idx) else {
        return Ok(None);
    };
    Ok(Some(
        available_carrier_neighbors_for_support_boundary_model(
            context.constraint_model,
            boundary_facts,
            component_idx,
            side_idx,
        )?,
    ))
}

fn support_boundary_carrier_obligation_neighbor(
    context: &StereoEdgeEmissionContext<'_>,
    side_idx: usize,
    available_neighbors: &[usize],
) -> Option<usize> {
    let side_info = context.side_infos.get(side_idx)?;
    let component_idx = context.constraint_model.component_for_side(side_idx)?;
    if available_neighbors.len() == 1 {
        return available_neighbors.first().copied();
    }
    if available_neighbors.len() != 2 {
        return None;
    }

    let shared_neighbors = available_neighbors
        .iter()
        .copied()
        .filter(|&neighbor_idx| {
            context
                .edge_to_side_ids
                .get(&canonical_edge(side_info.endpoint_atom_idx, neighbor_idx))
                .into_iter()
                .flatten()
                .copied()
                .any(|other_side_idx| {
                    other_side_idx != side_idx
                        && context.constraint_model.component_for_side(other_side_idx)
                            == Some(component_idx)
                })
        })
        .collect::<Vec<_>>();
    if shared_neighbors.len() == 1 {
        Some(shared_neighbors[0])
    } else {
        None
    }
}

fn deferred_carrier_choice_constraint_for_support_boundary(
    context: &StereoEdgeEmissionContext<'_>,
    state: &StereoEdgeEmissionState<'_>,
    side_idx: usize,
    neighbor_idx: usize,
    available_neighbors: &[usize],
) -> Option<DeferredCarrierChoiceConstraint> {
    let Some(side_info) = context.side_infos.get(side_idx) else {
        return None;
    };
    if state
        .component_phases
        .get(side_info.component_idx)
        .copied()
        .unwrap_or(UNKNOWN_COMPONENT_PHASE)
        != UNKNOWN_COMPONENT_PHASE
    {
        return None;
    }
    if available_neighbors.len() <= 1 || !available_neighbors.contains(&neighbor_idx) {
        return None;
    }
    let terminal_candidates = available_neighbors
        .iter()
        .copied()
        .filter(|&candidate_neighbor| context.graph.neighbors[candidate_neighbor].len() == 1)
        .collect::<Vec<_>>();
    if terminal_candidates.len() != 1 || neighbor_idx != terminal_candidates[0] {
        return None;
    }
    Some(DeferredCarrierChoiceConstraint {
        side_idx,
        deferred_neighbor_idx: neighbor_idx,
        available_neighbors: available_neighbors.to_vec(),
    })
}

fn should_defer_carrier_commit_for_constraint(
    constraint: &DeferredCarrierChoiceConstraint,
    side_idx: usize,
    neighbor_idx: usize,
) -> bool {
    constraint.side_idx == side_idx
        && constraint.deferred_neighbor_idx == neighbor_idx
        && constraint.available_neighbors.len() > 1
}

fn deferred_carrier_choice_constraints_block_neighbor(
    constraints: &[DeferredCarrierChoiceConstraint],
    side_idx: usize,
    neighbor_idx: usize,
) -> bool {
    constraints.iter().any(|constraint| {
        should_defer_carrier_commit_for_constraint(constraint, side_idx, neighbor_idx)
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CarrierCommitmentDecision {
    Commit,
    Defer(DeferredCarrierChoiceConstraint),
    Skip,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CarrierCommitmentBoundaryQuery {
    candidate_neighbor_idx: usize,
    forced_neighbor: Option<usize>,
    blocked_by_deferred_constraint: bool,
    deferrable_constraint: Option<DeferredCarrierChoiceConstraint>,
}

struct StereoSupportBoundaryFacts {
    carrier_facts_by_component: Vec<Vec<StereoConstraintFact>>,
    deferred_carrier_choice_constraints: Vec<DeferredCarrierChoiceConstraint>,
    known_token_flip_facts: Vec<StereoTokenFlipFact>,
    inferred_token_observation_facts: Vec<StereoTokenObservationFact>,
    marker_event_facts_by_component: Vec<Vec<StereoMarkerEventFact>>,
    marker_obligation_domains_by_component: Vec<Vec<RdkitWriterMarkerObligationDomain>>,
    marker_obligation_event_facts_by_component: Vec<Vec<StereoMarkerEventFact>>,
}

impl StereoSupportBoundaryFacts {
    fn constraint_state_for_model(
        &self,
        model: &StereoConstraintModel,
        layer: StereoConstraintLayer,
    ) -> PyResult<StereoConstraintState> {
        StereoConstraintState::from_facts_and_token_observations(
            model,
            layer,
            &self.carrier_facts_by_component,
            &self.known_token_flip_facts,
            &self.inferred_token_observation_facts,
        )
    }

    fn constraint_state(
        &self,
        runtime: &StereoWalkerRuntimeData,
        layer: StereoConstraintLayer,
    ) -> PyResult<StereoConstraintState> {
        self.constraint_state_for_model(&runtime.constraint_model, layer)
    }
}

fn marker_row_neighbor_assignment_ids_from_support_boundary(
    model: &StereoConstraintModel,
    boundary_facts: &StereoSupportBoundaryFacts,
    component_idx: usize,
    token_phase_assignment_ids: &[usize],
) -> PyResult<Option<Vec<usize>>> {
    let Some(marker_events) = boundary_facts
        .marker_obligation_event_facts_by_component
        .get(component_idx)
    else {
        return Err(PyValueError::new_err(
            "support-boundary marker facts do not match the constraint model",
        ));
    };
    if marker_events.is_empty() {
        return Ok(None);
    }

    let row_ids_before_marker_events = model
        .marker_placement_row_ids_for_token_phase_assignment_ids(
            component_idx,
            token_phase_assignment_ids,
        )?;
    let row_ids_after_marker_events = model
        .filter_marker_placement_row_ids_for_marker_event_facts(
            component_idx,
            &row_ids_before_marker_events,
            marker_events,
        )?;
    Ok(Some(
        model.neighbor_assignment_ids_for_marker_placement_row_ids(
            component_idx,
            &row_ids_after_marker_events,
        )?,
    ))
}

fn carrier_commitment_boundary_query_from_edge_state(
    context: &StereoEdgeEmissionContext<'_>,
    state: &StereoEdgeEmissionState<'_>,
    boundary_facts: &StereoSupportBoundaryFacts,
    side_idx: usize,
    neighbor_idx: usize,
) -> PyResult<CarrierCommitmentBoundaryQuery> {
    let blocked_by_deferred_constraint = deferred_carrier_choice_constraints_block_neighbor(
        &boundary_facts.deferred_carrier_choice_constraints,
        side_idx,
        neighbor_idx,
    );
    let available_neighbors =
        available_carrier_neighbors_from_support_boundary(context, boundary_facts, side_idx)?
            .unwrap_or_default();
    let forced_neighbor =
        support_boundary_carrier_obligation_neighbor(context, side_idx, &available_neighbors);
    let deferrable_constraint = deferred_carrier_choice_constraint_for_support_boundary(
        context,
        state,
        side_idx,
        neighbor_idx,
        &available_neighbors,
    )
    .filter(|constraint| {
        should_defer_carrier_commit_for_constraint(constraint, side_idx, neighbor_idx)
    });
    Ok(CarrierCommitmentBoundaryQuery {
        candidate_neighbor_idx: neighbor_idx,
        forced_neighbor,
        blocked_by_deferred_constraint,
        deferrable_constraint,
    })
}

fn carrier_commitment_decision_from_boundary_query(
    query: CarrierCommitmentBoundaryQuery,
) -> CarrierCommitmentDecision {
    if query.blocked_by_deferred_constraint {
        return CarrierCommitmentDecision::Skip;
    }
    if query
        .forced_neighbor
        .is_some_and(|forced_neighbor| forced_neighbor != query.candidate_neighbor_idx)
    {
        return CarrierCommitmentDecision::Skip;
    }
    if let Some(constraint) = query.deferrable_constraint {
        return CarrierCommitmentDecision::Defer(constraint);
    }
    CarrierCommitmentDecision::Commit
}

fn literal_bond_part(
    graph: &PreparedSmilesGraphData,
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<Part> {
    Ok(Part::Literal(
        graph
            .bond_token(begin_idx, end_idx)
            .ok_or_else(|| {
                PyKeyError::new_err(format!("No bond between atoms {begin_idx} and {end_idx}"))
            })?
            .to_owned(),
    ))
}

fn literal_edge_result(
    graph: &PreparedSmilesGraphData,
    begin_idx: usize,
    end_idx: usize,
    selected_neighbors: &[isize],
    selected_orientations: &[i8],
    first_emitted_candidates: &[isize],
    token_basis_facts: &[Option<StereoTokenBasisFact>],
) -> PyResult<EmittedEdgePartResult> {
    Ok(EmittedEdgePartResult {
        part: literal_bond_part(graph, begin_idx, end_idx)?,
        selected_neighbors: selected_neighbors.to_vec(),
        selected_orientations: selected_orientations.to_vec(),
        first_emitted_candidates: first_emitted_candidates.to_vec(),
        deferred_carrier_choice_constraints: Vec::new(),
        token_basis_facts: token_basis_facts.to_vec(),
    })
}

fn edge_result_with_part_and_deferred_carrier_constraints(
    part: Part,
    selected_neighbors: Vec<isize>,
    selected_orientations: Vec<i8>,
    first_emitted_candidates: Vec<isize>,
    deferred_carrier_choice_constraints: Vec<DeferredCarrierChoiceConstraint>,
    token_basis_facts: Vec<Option<StereoTokenBasisFact>>,
) -> EmittedEdgePartResult {
    EmittedEdgePartResult {
        part,
        selected_neighbors,
        selected_orientations,
        first_emitted_candidates,
        deferred_carrier_choice_constraints,
        token_basis_facts,
    }
}

fn deferred_edge_part(
    component_reference_tokens: &[(usize, Vec<String>)],
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<Part> {
    let mut component_tokens = Vec::<DeferredDirectionalComponentToken>::new();
    for (component_idx, reference_tokens) in component_reference_tokens {
        if let Some(existing) = component_tokens
            .iter_mut()
            .find(|existing| existing.component_idx == *component_idx)
        {
            existing
                .reference_tokens
                .extend(reference_tokens.iter().cloned());
            existing.reference_tokens.sort();
            existing.reference_tokens.dedup();
        } else {
            component_tokens.push(DeferredDirectionalComponentToken {
                component_idx: *component_idx,
                reference_tokens: reference_tokens.clone(),
            });
        }
    }

    Ok(Part::Deferred(DeferredDirectionalToken {
        component_tokens: Arc::from(component_tokens),
        begin_idx: begin_idx as isize,
        end_idx: end_idx as isize,
    }))
}

// Suspicious current model:
// Interleaves carrier selection, token deferral, and local ambiguity heuristics
// during edge emission. This should be split into explicit online constraints.
fn emitted_edge_part_generic(
    context: &StereoEdgeEmissionContext<'_>,
    state: &StereoEdgeEmissionState<'_>,
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<EmittedEdgePartResult> {
    let Some(side_ids) = context
        .edge_to_side_ids
        .get(&canonical_edge(begin_idx, end_idx))
    else {
        return literal_edge_result(
            context.graph,
            begin_idx,
            end_idx,
            state.selected_neighbors,
            state.selected_orientations,
            state.first_emitted_candidates,
            state.token_basis_facts,
        );
    };

    let mut updated_neighbors = state.selected_neighbors.to_vec();
    let mut updated_orientations = state.selected_orientations.to_vec();
    let mut updated_first_candidates = state.first_emitted_candidates.to_vec();
    let mut deferred_carrier_choice_constraints =
        state.deferred_carrier_choice_constraints.to_vec();
    let token_basis_facts = state.token_basis_facts.to_vec();
    let mut component_reference_tokens = Vec::<(usize, Vec<String>)>::new();

    for &side_idx in side_ids {
        let side_info = &context.side_infos[side_idx];
        let (neighbor_idx, edge_orientation) = if begin_idx == side_info.endpoint_atom_idx {
            (end_idx, AFTER_ATOM_EDGE_ORIENTATION)
        } else if end_idx == side_info.endpoint_atom_idx {
            (begin_idx, BEFORE_ATOM_EDGE_ORIENTATION)
        } else {
            continue;
        };

        if side_info.candidate_neighbors.len() == 2 && updated_first_candidates[side_idx] < 0 {
            updated_first_candidates[side_idx] = neighbor_idx as isize;
        }

        let selected_neighbor = updated_neighbors[side_idx];
        if selected_neighbor < 0 {
            let boundary_facts = support_boundary_facts_from_edge_state(
                context,
                &updated_neighbors,
                &deferred_carrier_choice_constraints,
                state.marker_event_traces,
                state.committed_component_token_flips,
            )?;
            let commitment_query = carrier_commitment_boundary_query_from_edge_state(
                context,
                state,
                &boundary_facts,
                side_idx,
                neighbor_idx,
            )?;
            match carrier_commitment_decision_from_boundary_query(commitment_query) {
                CarrierCommitmentDecision::Commit => {
                    updated_neighbors[side_idx] = neighbor_idx as isize;
                    updated_orientations[side_idx] = edge_orientation;
                }
                CarrierCommitmentDecision::Defer(constraint) => {
                    if !deferred_carrier_choice_constraints.contains(&constraint) {
                        deferred_carrier_choice_constraints.push(constraint);
                    }
                    continue;
                }
                CarrierCommitmentDecision::Skip => continue,
            }
        }
        let selected_neighbor = updated_neighbors[side_idx];
        if selected_neighbor != neighbor_idx as isize {
            continue;
        }
        component_reference_tokens.push((
            side_info.component_idx,
            component_reference_tokens_for_emitted_edge(side_info, begin_idx, end_idx)?,
        ));
    }

    if component_reference_tokens.is_empty() {
        return Ok(edge_result_with_part_and_deferred_carrier_constraints(
            literal_bond_part(context.graph, begin_idx, end_idx)?,
            updated_neighbors,
            updated_orientations,
            updated_first_candidates,
            deferred_carrier_choice_constraints,
            token_basis_facts,
        ));
    }

    Ok(edge_result_with_part_and_deferred_carrier_constraints(
        deferred_edge_part(&component_reference_tokens, begin_idx, end_idx)?,
        updated_neighbors,
        updated_orientations,
        updated_first_candidates,
        deferred_carrier_choice_constraints,
        token_basis_facts,
    ))
}

fn emitted_isolated_edge_part(
    context: &StereoEdgeEmissionContext<'_>,
    state: &StereoEdgeEmissionState<'_>,
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<EmittedEdgePartResult> {
    let Some(side_ids) = context
        .edge_to_side_ids
        .get(&canonical_edge(begin_idx, end_idx))
    else {
        return literal_edge_result(
            context.graph,
            begin_idx,
            end_idx,
            state.selected_neighbors,
            state.selected_orientations,
            state.first_emitted_candidates,
            state.token_basis_facts,
        );
    };

    let mut updated_neighbors = state.selected_neighbors.to_vec();
    let mut updated_orientations = state.selected_orientations.to_vec();
    let mut updated_first_candidates = state.first_emitted_candidates.to_vec();
    let deferred_carrier_choice_constraints = state.deferred_carrier_choice_constraints.to_vec();
    let mut component_reference_tokens = Vec::<(usize, Vec<String>)>::new();

    for &side_idx in side_ids {
        let side_info = &context.side_infos[side_idx];
        let (neighbor_idx, edge_orientation) = if begin_idx == side_info.endpoint_atom_idx {
            (end_idx, AFTER_ATOM_EDGE_ORIENTATION)
        } else if end_idx == side_info.endpoint_atom_idx {
            (begin_idx, BEFORE_ATOM_EDGE_ORIENTATION)
        } else {
            continue;
        };

        if side_info.candidate_neighbors.len() == 2 && updated_first_candidates[side_idx] < 0 {
            updated_first_candidates[side_idx] = neighbor_idx as isize;
        }

        let selected_neighbor = updated_neighbors[side_idx];
        if selected_neighbor < 0 {
            updated_neighbors[side_idx] = neighbor_idx as isize;
            updated_orientations[side_idx] = edge_orientation;
        }
        let selected_neighbor = updated_neighbors[side_idx];
        if selected_neighbor != neighbor_idx as isize {
            continue;
        }
    }

    let token_basis_facts = updated_isolated_component_token_basis_facts_from_row_state(
        context,
        state.component_begin_atoms,
        &updated_neighbors,
        side_ids,
        state.token_basis_facts,
    )?;

    for &side_idx in side_ids {
        let side_info = &context.side_infos[side_idx];
        let neighbor_idx = if begin_idx == side_info.endpoint_atom_idx {
            end_idx
        } else if end_idx == side_info.endpoint_atom_idx {
            begin_idx
        } else {
            continue;
        };
        if updated_neighbors[side_idx] != neighbor_idx as isize {
            continue;
        }
        let reference_tokens =
            component_reference_tokens_for_emitted_edge(side_info, begin_idx, end_idx)?;
        component_reference_tokens.push((side_info.component_idx, reference_tokens));
    }

    if component_reference_tokens.is_empty() {
        return Ok(edge_result_with_part_and_deferred_carrier_constraints(
            literal_bond_part(context.graph, begin_idx, end_idx)?,
            updated_neighbors,
            updated_orientations,
            updated_first_candidates,
            deferred_carrier_choice_constraints,
            token_basis_facts,
        ));
    }

    Ok(edge_result_with_part_and_deferred_carrier_constraints(
        deferred_edge_part(&component_reference_tokens, begin_idx, end_idx)?,
        updated_neighbors,
        updated_orientations,
        updated_first_candidates,
        deferred_carrier_choice_constraints,
        token_basis_facts,
    ))
}

fn emitted_edge_part(
    context: &StereoEdgeEmissionContext<'_>,
    state: &StereoEdgeEmissionState<'_>,
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<EmittedEdgePartResult> {
    let edge = canonical_edge(begin_idx, end_idx);
    let Some(side_ids) = context.edge_to_side_ids.get(&edge) else {
        return literal_edge_result(
            context.graph,
            begin_idx,
            end_idx,
            state.selected_neighbors,
            state.selected_orientations,
            state.first_emitted_candidates,
            state.token_basis_facts,
        );
    };

    let uses_isolated_component = side_ids.iter().any(|&side_idx| {
        let component_idx = context.side_infos[side_idx].component_idx;
        context
            .isolated_components
            .get(component_idx)
            .copied()
            .unwrap_or(false)
    });
    if uses_isolated_component {
        emitted_isolated_edge_part(context, state, begin_idx, end_idx)
    } else {
        emitted_edge_part_generic(context, state, begin_idx, end_idx)
    }
}

fn process_children_edge_update(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    parent_idx: usize,
    child_order: &[usize],
    child_idx: usize,
) -> PyResult<ProcessChildrenEdgeUpdate> {
    let edge_context = StereoEdgeEmissionContext {
        graph,
        constraint_model: &runtime.constraint_model,
        side_infos: &runtime.side_infos,
        side_ids_by_component: &runtime.side_ids_by_component,
        edge_to_side_ids: &runtime.edge_to_side_ids,
        isolated_components: &runtime.isolated_components,
    };
    let (current_phases, current_begin_atoms) = eager_component_phases_for_child_order(
        runtime,
        graph,
        &state.stereo_component_phases,
        &state.stereo_component_begin_atoms,
        &state.stereo_selected_neighbors,
        parent_idx,
        child_order,
    )?;
    let (current_selected_neighbors, current_selected_orientations, current_first_candidates) =
        eager_begin_side_child_order_state(
            runtime,
            &current_begin_atoms,
            &state.stereo_selected_neighbors,
            &state.stereo_selected_orientations,
            &state.stereo_first_emitted_candidates,
            parent_idx,
            child_order,
        );
    let EmittedEdgePartResult {
        part: edge_part,
        selected_neighbors: updated_neighbors,
        selected_orientations: updated_orientations,
        first_emitted_candidates: updated_first_candidates,
        deferred_carrier_choice_constraints: updated_deferred_carrier_choice_constraints,
        token_basis_facts: updated_token_basis_facts,
    } = emitted_edge_part(
        &edge_context,
        &StereoEdgeEmissionState {
            component_phases: &current_phases,
            selected_neighbors: &current_selected_neighbors,
            selected_orientations: &current_selected_orientations,
            first_emitted_candidates: &current_first_candidates,
            component_begin_atoms: &current_begin_atoms,
            deferred_carrier_choice_constraints: &state.deferred_carrier_choice_constraints,
            token_basis_facts: &state.stereo_token_basis_facts,
            marker_event_traces: &state.marker_event_traces,
            committed_component_token_flips: &state.committed_component_token_flips,
        },
        parent_idx,
        child_idx,
    )?;
    let (updated_phases, updated_begin_atoms) = component_phases_after_edge(
        graph,
        &runtime.stereo_component_ids,
        &runtime.isolated_components,
        &current_phases,
        &current_begin_atoms,
        parent_idx,
        child_idx,
    )?;
    let (updated_neighbors, updated_orientations) = force_known_begin_side_selection(
        runtime,
        &updated_phases,
        &updated_begin_atoms,
        &updated_neighbors,
        &updated_orientations,
    );
    let (updated_phases, updated_begin_atoms) = defer_component_phase_for_unresolved_begin_side(
        runtime,
        graph,
        &updated_phases,
        &updated_begin_atoms,
        &updated_neighbors,
        parent_idx,
        child_idx,
    )?;
    let updated_phases = commit_deferred_component_phase_facts_from_selected_begin_sides(
        runtime,
        &updated_phases,
        &updated_begin_atoms,
        &updated_neighbors,
        parent_idx,
        &edge_part,
    )?;
    Ok(ProcessChildrenEdgeUpdate {
        edge_part,
        selected_neighbors: updated_neighbors,
        selected_orientations: updated_orientations,
        first_emitted_candidates: updated_first_candidates,
        deferred_carrier_choice_constraints: updated_deferred_carrier_choice_constraints,
        token_basis_facts: updated_token_basis_facts,
        component_phases: updated_phases,
        component_begin_atoms: updated_begin_atoms,
    })
}

fn push_process_children_branch_actions(
    action_stack: &mut Vec<WalkerAction>,
    parent_idx: usize,
    child_order: Arc<[usize]>,
    next_branch_index: usize,
    child_idx: usize,
    edge_action: Option<WalkerAction>,
) {
    if next_branch_index + 1 < child_order.len() {
        action_stack.push(WalkerAction::ProcessChildren {
            parent_idx,
            child_order,
            next_branch_index: next_branch_index + 1,
        });
    }
    action_stack.push(WalkerAction::EmitCloseParen);
    action_stack.push(WalkerAction::EnterAtom {
        atom_idx: child_idx,
        parent_idx: Some(parent_idx),
    });
    if let Some(action) = edge_action {
        action_stack.push(action);
    }
}

fn process_children_terminal_successors(
    context: &mut StereoProcessChildrenContext<'_>,
    mut base_state: RootedConnectedStereoWalkerStateData,
    step: ProcessChildrenTerminalStep,
) -> PyResult<BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>> {
    match step.edge_part {
        Part::Literal(token) if token.is_empty() => {
            let role = directional_token_role(base_state.prefix.as_ref(), None);
            record_rdkit_marker_event_traces_for_edge(
                context.runtime,
                &mut base_state,
                step.parent_idx as isize,
                step.child_idx as isize,
                None,
                role,
            );
            base_state.action_stack.push(WalkerAction::EnterAtom {
                atom_idx: step.child_idx,
                parent_idx: Some(step.parent_idx),
            });
            stereo_next_token_successors_from_boundary(
                context.runtime,
                context.graph,
                &base_state,
                StereoNextTokenCompleteness::from_require_completable(context.require_completable),
                context.completion_cache,
            )
        }
        Part::Literal(token) => {
            let role = directional_token_role(base_state.prefix.as_ref(), None);
            record_rdkit_literal_edge_marker_trace(
                context.runtime,
                context.graph,
                &mut base_state,
                step.parent_idx as isize,
                step.child_idx as isize,
                token.as_str(),
                role,
            );
            push_literal_token(&mut base_state.prefix, &token);
            base_state.action_stack.push(WalkerAction::EnterAtom {
                atom_idx: step.child_idx,
                parent_idx: Some(step.parent_idx),
            });
            Ok(BTreeMap::from([(token, vec![base_state])]))
        }
        Part::Deferred(deferred) => {
            let mut out = BTreeMap::<String, Vec<RootedConnectedStereoWalkerStateData>>::new();
            for token in
                deferred_token_support(context.runtime, context.graph, &base_state, &deferred)?
            {
                let mut successor = base_state.clone();
                if let Err(err) = commit_deferred_token_choice(
                    context.runtime,
                    context.graph,
                    &mut successor,
                    &deferred,
                    &token,
                ) {
                    if context.require_completable {
                        return Err(err);
                    }
                    continue;
                }
                let role = directional_token_role(successor.prefix.as_ref(), None);
                record_rdkit_directional_marker_trace(
                    context.runtime,
                    context.graph,
                    &mut successor,
                    deferred.begin_idx,
                    deferred.end_idx,
                    &token,
                    role,
                );
                push_literal_token(&mut successor.prefix, &token);
                successor.action_stack.push(WalkerAction::EnterAtom {
                    atom_idx: step.child_idx,
                    parent_idx: Some(step.parent_idx),
                });
                if !context.require_completable
                    || can_complete_from_stereo_state_memo(
                        context.runtime,
                        context.graph,
                        &successor,
                        context.completion_cache,
                    )
                {
                    out.entry(token).or_default().push(successor);
                }
            }
            Ok(out)
        }
    }
}

fn resolved_selected_neighbors_from_assignment_state(
    runtime: &StereoWalkerRuntimeData,
    selected_neighbors: &[isize],
) -> Vec<isize> {
    let mut resolved_neighbors = selected_neighbors.to_vec();
    let facts_by_component = selected_neighbor_facts_by_component(runtime, selected_neighbors);
    let assignment_state = StereoAssignmentState::from_facts_by_component(
        &runtime.constraint_model,
        StereoConstraintLayer::Semantic,
        &facts_by_component,
    );
    for component in &runtime.constraint_model.components {
        for &side_idx in &component.side_ids {
            if resolved_neighbors[side_idx] >= 0 {
                continue;
            }
            if let Some(neighbor_idx) = assignment_state.forced_neighbor(
                &runtime.constraint_model,
                component.component_idx,
                side_idx,
            ) {
                resolved_neighbors[side_idx] = neighbor_idx as isize;
            }
        }
    }

    resolved_neighbors
}

// Suspicious current model:
// Resolves ambiguous shared-edge selections after the fact by forcing both
// sides to the shared edge. This remains runtime behavior until carrier-choice
// row state can cover the coupled diene witnesses without reducing support.
fn resolved_selected_neighbors_from_fields(
    runtime: &StereoWalkerRuntimeData,
    selected_neighbors: &[isize],
    first_emitted_candidates: &[isize],
) -> Vec<isize> {
    let mut selected_neighbors = selected_neighbors.to_vec();

    for group in &runtime.ambiguous_shared_edge_groups {
        let left_saw_shared_first =
            first_emitted_candidates[group.left_side_idx] == group.left_shared_neighbor as isize;
        let right_saw_shared_first =
            first_emitted_candidates[group.right_side_idx] == group.right_shared_neighbor as isize;

        if left_saw_shared_first && right_saw_shared_first {
            selected_neighbors[group.left_side_idx] = group.left_shared_neighbor as isize;
            selected_neighbors[group.right_side_idx] = group.right_shared_neighbor as isize;
            continue;
        }

        selected_neighbors[group.left_side_idx] = group.left_shared_neighbor as isize;
        selected_neighbors[group.right_side_idx] = group.right_shared_neighbor as isize;
    }

    selected_neighbors
}

fn resolved_selected_neighbors(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
) -> Vec<isize> {
    let mut selected_neighbors = resolved_selected_neighbors_from_fields(
        runtime,
        &state.stereo_selected_neighbors,
        &state.stereo_first_emitted_candidates,
    );
    for constraint in state.deferred_carrier_choice_constraints.iter() {
        if selected_neighbors
            .get(constraint.side_idx)
            .copied()
            .is_some_and(|neighbor_idx| neighbor_idx == constraint.deferred_neighbor_idx as isize)
        {
            selected_neighbors[constraint.side_idx] = -1;
        }
    }
    selected_neighbors
}

fn support_boundary_facts_from_token_constraints(
    carrier_facts_by_component: Vec<Vec<StereoConstraintFact>>,
    deferred_carrier_choice_constraints: Vec<DeferredCarrierChoiceConstraint>,
    token_constraints: &[ComponentTokenConstraint],
    marker_event_facts_by_component: Vec<Vec<StereoMarkerEventFact>>,
) -> PyResult<StereoSupportBoundaryFacts> {
    let marker_obligation_domains_by_component =
        rdkit_writer_marker_obligation_domains_by_component(&marker_event_facts_by_component);
    let marker_obligation_event_facts_by_component =
        rdkit_writer_slot_coalesced_marker_event_facts_by_component(
            &marker_event_facts_by_component,
        );
    Ok(StereoSupportBoundaryFacts {
        carrier_facts_by_component,
        deferred_carrier_choice_constraints,
        known_token_flip_facts: known_token_flip_facts_from_constraints(token_constraints),
        inferred_token_observation_facts: inferred_token_observation_facts_from_constraints(
            token_constraints,
        ),
        marker_event_facts_by_component,
        marker_obligation_domains_by_component,
        marker_obligation_event_facts_by_component,
    })
}

fn support_boundary_facts_from_walker_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    selected_neighbors: &[isize],
    use_partial_token_constraints: bool,
) -> PyResult<StereoSupportBoundaryFacts> {
    let carrier_facts_by_component = carrier_facts_by_component_from_state(
        runtime,
        selected_neighbors,
        &state.deferred_carrier_choice_constraints,
    )?;
    let token_constraints = if use_partial_token_constraints {
        partial_component_token_constraints_from_state(runtime, graph, state, selected_neighbors)?
    } else {
        component_token_constraints_from_state(runtime, graph, state, selected_neighbors)?
    };
    support_boundary_facts_from_token_constraints(
        carrier_facts_by_component,
        state.deferred_carrier_choice_constraints.to_vec(),
        &token_constraints,
        rdkit_writer_marker_event_facts_by_component(runtime, state)?,
    )
}

fn support_boundary_facts_from_edge_state(
    context: &StereoEdgeEmissionContext<'_>,
    selected_neighbors: &[isize],
    deferred_carrier_choice_constraints: &[DeferredCarrierChoiceConstraint],
    marker_event_traces: &[MarkerEventTrace],
    committed_component_token_flips: &[Option<StereoTokenFlip>],
) -> PyResult<StereoSupportBoundaryFacts> {
    let carrier_facts_by_component = carrier_facts_by_component_from_edge_state(
        context,
        selected_neighbors,
        deferred_carrier_choice_constraints,
    )?;
    let marker_event_facts_by_component = rdkit_writer_marker_event_facts_by_component_from_traces(
        context.constraint_model,
        context.constraint_model.component_count(),
        marker_event_traces,
    )?;
    let marker_obligation_domains_by_component =
        rdkit_writer_marker_obligation_domains_by_component(&marker_event_facts_by_component);
    let marker_obligation_event_facts_by_component =
        rdkit_writer_slot_coalesced_marker_event_facts_by_component(
            &marker_event_facts_by_component,
        );
    Ok(StereoSupportBoundaryFacts {
        carrier_facts_by_component,
        deferred_carrier_choice_constraints: deferred_carrier_choice_constraints.to_vec(),
        known_token_flip_facts: known_token_flip_facts_from_committed_component_token_flips(
            committed_component_token_flips,
        ),
        inferred_token_observation_facts: Vec::new(),
        marker_event_facts_by_component,
        marker_obligation_domains_by_component,
        marker_obligation_event_facts_by_component,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SupportStateSelectedNeighborComponentSummary {
    component_idx: usize,
    carrier_assignment_count: usize,
    token_phase_assignment_count: usize,
    marker_row_count_before_events: Option<usize>,
    marker_row_count_after_events: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SupportStateSelectedNeighborQuery {
    selected_neighbors: Vec<isize>,
    forced_side_neighbors: Vec<(usize, usize)>,
    unresolved_side_ids: Vec<usize>,
    component_summaries: Vec<SupportStateSelectedNeighborComponentSummary>,
}

fn support_state_selected_neighbor_query(
    runtime: &StereoWalkerRuntimeData,
    raw_selected_neighbors: &[isize],
    boundary_facts: &StereoSupportBoundaryFacts,
    include_marker_survivor_counts: bool,
) -> PyResult<SupportStateSelectedNeighborQuery> {
    let constraint_state =
        boundary_facts.constraint_state(runtime, StereoConstraintLayer::Semantic)?;

    let mut selected_neighbors = raw_selected_neighbors.to_vec();
    let mut forced_side_neighbors = Vec::<(usize, usize)>::new();
    let mut unresolved_side_ids = BTreeSet::<usize>::new();
    let mut component_summaries = Vec::new();
    for component in &runtime.constraint_model.components {
        let component_idx = component.component_idx;
        let carrier_assignment_count = constraint_state
            .carrier_assignment_state
            .remaining_by_component
            .get(component_idx)
            .map(Vec::len)
            .unwrap_or(0);
        let token_phase_assignment_ids = constraint_state
            .token_phase_remaining_by_component
            .get(component_idx)
            .cloned()
            .unwrap_or_default();
        let (marker_row_count_before_events, marker_row_count_after_events) =
            if include_marker_survivor_counts {
                let Some(marker_events) = boundary_facts
                    .marker_event_facts_by_component
                    .get(component_idx)
                else {
                    return Err(PyValueError::new_err(
                        "support-state selected-neighbor query missing marker events",
                    ));
                };
                let survivor_state = rdkit_marker_row_survivor_component_state(
                    runtime,
                    component_idx,
                    &token_phase_assignment_ids,
                    marker_events,
                )?;
                (
                    Some(survivor_state.row_ids_before_marker_events.len()),
                    Some(survivor_state.row_ids_after_marker_events.len()),
                )
            } else {
                (None, None)
            };

        for &side_idx in &component.side_ids {
            if selected_neighbors[side_idx] >= 0 {
                continue;
            }
            let available_neighbors = available_carrier_neighbors_for_support_boundary_model(
                &runtime.constraint_model,
                boundary_facts,
                component_idx,
                side_idx,
            )?;
            if let [neighbor_idx] = available_neighbors.as_slice() {
                selected_neighbors[side_idx] = *neighbor_idx as isize;
                forced_side_neighbors.push((side_idx, *neighbor_idx));
            } else if let Some(neighbor_idx) =
                constraint_state.forced_neighbor(&runtime.constraint_model, component_idx, side_idx)
            {
                selected_neighbors[side_idx] = neighbor_idx as isize;
                forced_side_neighbors.push((side_idx, neighbor_idx));
            } else {
                unresolved_side_ids.insert(side_idx);
            }
        }
        component_summaries.push(SupportStateSelectedNeighborComponentSummary {
            component_idx,
            carrier_assignment_count,
            token_phase_assignment_count: token_phase_assignment_ids.len(),
            marker_row_count_before_events,
            marker_row_count_after_events,
        });
    }
    Ok(SupportStateSelectedNeighborQuery {
        selected_neighbors,
        forced_side_neighbors,
        unresolved_side_ids: unresolved_side_ids.into_iter().collect(),
        component_summaries,
    })
}

fn selected_neighbors_from_support_boundary_constraints(
    runtime: &StereoWalkerRuntimeData,
    raw_selected_neighbors: &[isize],
    boundary_facts: &StereoSupportBoundaryFacts,
) -> PyResult<Vec<isize>> {
    Ok(support_state_selected_neighbor_query(
        runtime,
        raw_selected_neighbors,
        boundary_facts,
        false,
    )?
    .selected_neighbors)
}

fn support_state_selected_neighbor_query_to_py(
    py: Python<'_>,
    query: &SupportStateSelectedNeighborQuery,
) -> PyResult<Py<PyDict>> {
    let row = PyDict::new(py);
    row.set_item("selected_neighbors", query.selected_neighbors.clone())?;
    row.set_item("forced_side_neighbors", query.forced_side_neighbors.clone())?;
    row.set_item("unresolved_side_ids", query.unresolved_side_ids.clone())?;
    row.set_item(
        "component_summaries",
        query
            .component_summaries
            .iter()
            .map(|summary| {
                let component_row = PyDict::new(py);
                component_row.set_item("component_idx", summary.component_idx)?;
                component_row
                    .set_item("carrier_assignment_count", summary.carrier_assignment_count)?;
                component_row.set_item(
                    "token_phase_assignment_count",
                    summary.token_phase_assignment_count,
                )?;
                component_row.set_item(
                    "marker_row_count_before_events",
                    summary.marker_row_count_before_events,
                )?;
                component_row.set_item(
                    "marker_row_count_after_events",
                    summary.marker_row_count_after_events,
                )?;
                Ok(component_row.unbind())
            })
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    Ok(row.unbind())
}

fn joined_support_boundary_selected_neighbors(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<Vec<isize>> {
    let raw_selected_neighbors = state.stereo_selected_neighbors.as_ref();
    let boundary_facts = support_boundary_facts_from_walker_state(
        runtime,
        graph,
        state,
        raw_selected_neighbors,
        false,
    )?;
    selected_neighbors_from_support_boundary_constraints(
        runtime,
        raw_selected_neighbors,
        &boundary_facts,
    )
}

fn partial_joined_support_boundary_selected_neighbors(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<Vec<isize>> {
    let raw_selected_neighbors = state.stereo_selected_neighbors.as_ref();
    let boundary_facts = support_boundary_facts_from_walker_state(
        runtime,
        graph,
        state,
        raw_selected_neighbors,
        true,
    )?;
    selected_neighbors_from_support_boundary_constraints(
        runtime,
        raw_selected_neighbors,
        &boundary_facts,
    )
}

pub(crate) fn enumerate_rooted_connected_stereo_smiles_support(
    graph: &PreparedSmilesGraphData,
    root_idx: isize,
) -> PyResult<Vec<String>> {
    let root_idx = validate_root_idx(graph, root_idx)?;
    if graph.atom_count() == 0 {
        return Ok(vec![String::new()]);
    }

    let runtime = build_walker_runtime(graph, root_idx)?;
    let mut support = BTreeSet::new();
    let initial_state = initial_stereo_state_for_root(&runtime, graph, root_idx);
    if runtime.side_infos.is_empty() {
        enumerate_support_from_atom_stereo_exact_state(
            &runtime,
            graph,
            stereo_exact_state_from_full(initial_state),
            &mut support,
        )?;
    } else {
        enumerate_support_from_stereo_state(&runtime, graph, initial_state, &mut support)?;
    }
    Ok(support.into_iter().collect())
}

fn build_walker_runtime(
    graph: &PreparedSmilesGraphData,
    root_idx: usize,
) -> PyResult<StereoWalkerRuntimeData> {
    let stereo_component_ids = stereo_component_ids(graph);
    let component_count = stereo_component_ids.iter().copied().max().unwrap_or(-1) + 1;
    let isolated_components = component_sizes(&stereo_component_ids)
        .into_iter()
        .map(|size| size == 1)
        .collect::<Vec<_>>();
    let StereoSideInfoBuild {
        side_infos,
        edge_to_side_ids,
    } = stereo_side_infos(graph, &stereo_component_ids)?;
    let mut side_ids_by_component = vec![Vec::new(); component_count as usize];
    for (side_idx, side_info) in side_infos.iter().enumerate() {
        if side_info.component_idx >= side_ids_by_component.len() {
            return Err(PyValueError::new_err(
                "stereo side component index out of range",
            ));
        }
        side_ids_by_component[side_info.component_idx].push(side_idx);
    }
    let ambiguous_shared_edge_groups =
        ambiguous_shared_edge_groups(&side_infos, &edge_to_side_ids, &isolated_components);
    let local_hazards = rdkit_local_writer_hazards(graph, &side_infos);
    let constraint_model =
        stereo_constraint_model(&side_infos, &side_ids_by_component, &local_hazards)?;
    Ok(StereoWalkerRuntimeData {
        root_idx,
        stereo_component_ids,
        isolated_components,
        side_infos,
        edge_to_side_ids,
        side_ids_by_component,
        ambiguous_shared_edge_groups,
        constraint_model,
    })
}

fn stereo_constraint_layer_name(layer: StereoConstraintLayer) -> &'static str {
    match layer {
        StereoConstraintLayer::Semantic => "semantic",
        StereoConstraintLayer::RdkitLocalWriter => "rdkit_local_writer",
        StereoConstraintLayer::RdkitTraversalWriter => "rdkit_traversal_writer",
    }
}

fn component_phase_name(value: i8) -> &'static str {
    match value {
        UNKNOWN_COMPONENT_PHASE => "unknown",
        STORED_COMPONENT_PHASE => "stored",
        FLIPPED_COMPONENT_PHASE => "flipped",
        _ => "invalid",
    }
}

fn component_token_flip_name(value: i8) -> &'static str {
    match value {
        UNKNOWN_COMPONENT_TOKEN_FLIP => "unknown",
        STORED_COMPONENT_TOKEN_FLIP => "stored",
        FLIPPED_COMPONENT_TOKEN_FLIP => "flipped",
        _ => "invalid",
    }
}

fn model_token_flip_from_component_value(value: i8) -> Option<StereoTokenFlip> {
    match value {
        STORED_COMPONENT_TOKEN_FLIP => Some(StereoTokenFlip::Stored),
        FLIPPED_COMPONENT_TOKEN_FLIP => Some(StereoTokenFlip::Flipped),
        _ => None,
    }
}

fn model_token_flip_name(value: StereoTokenFlip) -> &'static str {
    match value {
        StereoTokenFlip::Stored => "stored",
        StereoTokenFlip::Flipped => "flipped",
    }
}

fn model_component_phase_from_value(value: i8) -> Option<StereoComponentPhase> {
    match value {
        STORED_COMPONENT_PHASE => Some(StereoComponentPhase::Stored),
        FLIPPED_COMPONENT_PHASE => Some(StereoComponentPhase::Flipped),
        _ => None,
    }
}

fn model_component_phase_name(value: StereoComponentPhase) -> &'static str {
    match value {
        StereoComponentPhase::Stored => "stored",
        StereoComponentPhase::Flipped => "flipped",
    }
}

fn stereo_direction_token_name(value: StereoDirectionToken) -> &'static str {
    match value {
        StereoDirectionToken::Slash => "/",
        StereoDirectionToken::Backslash => "\\",
    }
}

fn rdkit_marker_placement_row_to_py(
    py: Python<'_>,
    component: &StereoComponentConstraintModel,
    row_idx: usize,
) -> PyResult<Py<PyDict>> {
    let Some(row) = component.all_marker_placement_rows.get(row_idx) else {
        return Err(PyValueError::new_err(
            "marker placement diagnostic row index out of range",
        ));
    };
    let row_dict = PyDict::new(py);
    row_dict.set_item("row_idx", row_idx)?;
    row_dict.set_item("token_phase_assignment_id", row.token_phase_assignment_id)?;
    row_dict.set_item("marker_neighbor_sets", row.marker_neighbor_sets.clone())?;
    if let Some(token_phase_assignment) = component
        .all_token_phase_assignments
        .get(row.token_phase_assignment_id)
    {
        row_dict.set_item(
            "neighbor_assignment_id",
            token_phase_assignment.neighbor_assignment_id,
        )?;
        row_dict.set_item(
            "token_flips",
            token_phase_assignment
                .token_flips
                .iter()
                .copied()
                .map(model_token_flip_name)
                .collect::<Vec<_>>(),
        )?;
        if let Some(carrier_neighbors) = component
            .all_neighbor_assignments
            .get(token_phase_assignment.neighbor_assignment_id)
        {
            row_dict.set_item("carrier_neighbors", carrier_neighbors.clone())?;
        }
    }
    Ok(row_dict.unbind())
}

#[pyfunction(name = "_stereo_constraint_model_summary")]
pub fn internal_stereo_constraint_model_summary(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
) -> PyResult<Py<PyDict>> {
    let graph = PreparedSmilesGraphData::from_any(graph)?;
    let runtime = build_walker_runtime(&graph, 0)?;
    let model = &runtime.constraint_model;

    let summary = PyDict::new(py);
    summary.set_item("component_count", model.component_count())?;
    summary.set_item("side_count", runtime.side_infos.len())?;
    summary.set_item(
        "component_sizes",
        component_sizes(&runtime.stereo_component_ids),
    )?;
    let shared_carrier_groups = runtime
        .ambiguous_shared_edge_groups
        .iter()
        .map(|group| {
            let row = PyDict::new(py);
            row.set_item("left_side_idx", group.left_side_idx)?;
            row.set_item("right_side_idx", group.right_side_idx)?;
            row.set_item("left_shared_neighbor", group.left_shared_neighbor)?;
            row.set_item("right_shared_neighbor", group.right_shared_neighbor)?;
            Ok(row.unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    summary.set_item("shared_carrier_groups", shared_carrier_groups)?;

    let components = model
        .components
        .iter()
        .map(|component| {
            let component_dict = PyDict::new(py);
            component_dict.set_item("component_idx", component.component_idx)?;
            component_dict.set_item("side_ids", component.side_ids.clone())?;
            component_dict.set_item(
                "side_domain_sizes",
                component
                    .side_domains
                    .iter()
                    .map(|domain| domain.choices.len())
                    .collect::<Vec<_>>(),
            )?;
            component_dict.set_item(
                "domain_assignment_count",
                component
                    .side_domains
                    .iter()
                    .map(|domain| domain.choices.len())
                    .product::<usize>(),
            )?;
            component_dict.set_item(
                "marker_placement_domain_sizes",
                component
                    .side_domains
                    .iter()
                    .map(|domain| {
                        let unique_neighbors = domain
                            .choices
                            .iter()
                            .map(|choice| choice.neighbor_idx)
                            .collect::<BTreeSet<_>>()
                            .len();
                        (1usize << unique_neighbors) - 1
                    })
                    .collect::<Vec<_>>(),
            )?;
            component_dict.set_item(
                "token_phase_assignment_count",
                component.all_token_phase_assignments.len(),
            )?;
            component_dict.set_item(
                "marker_placement_row_count",
                component.all_marker_placement_rows.len(),
            )?;

            let side_domains = component
                .side_domains
                .iter()
                .map(|domain| {
                    let domain_dict = PyDict::new(py);
                    domain_dict.set_item("side_idx", domain.side_idx)?;
                    domain_dict.set_item("component_idx", domain.component_idx)?;
                    domain_dict.set_item("endpoint_atom_idx", domain.endpoint_atom_idx)?;
                    let choices = domain
                        .choices
                        .iter()
                        .map(|choice| {
                            let choice_dict = PyDict::new(py);
                            choice_dict.set_item("neighbor_idx", choice.neighbor_idx)?;
                            choice_dict.set_item("base_token", &choice.base_token)?;
                            Ok(choice_dict)
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    domain_dict.set_item("choices", choices)?;
                    Ok(domain_dict)
                })
                .collect::<PyResult<Vec<_>>>()?;
            component_dict.set_item("side_domains", side_domains)?;

            let marker_placement_rows = component
                .all_marker_placement_rows
                .iter()
                .enumerate()
                .map(|(row_idx, _)| rdkit_marker_placement_row_to_py(py, component, row_idx))
                .collect::<PyResult<Vec<_>>>()?;
            component_dict.set_item("marker_placement_rows", marker_placement_rows)?;

            let layers = component
                .layer_assignments
                .iter()
                .map(|assignments| {
                    let layer_dict = PyDict::new(py);
                    layer_dict
                        .set_item("layer", stereo_constraint_layer_name(assignments.layer))?;
                    layer_dict.set_item(
                        "assignment_count",
                        assignments
                            .allowed_neighbor_assignments
                            .as_ref()
                            .map(Vec::len),
                    )?;
                    layer_dict.set_item(
                        "is_unrestricted",
                        assignments.allowed_neighbor_assignments.is_none(),
                    )?;
                    Ok(layer_dict)
                })
                .collect::<PyResult<Vec<_>>>()?;
            component_dict.set_item("layers", layers)?;

            Ok(component_dict)
        })
        .collect::<PyResult<Vec<_>>>()?;
    summary.set_item("components", components)?;

    Ok(summary.unbind())
}

fn selected_neighbor_fact_rows(
    runtime: &StereoWalkerRuntimeData,
    selected_neighbors: &[isize],
) -> Vec<(usize, usize, usize)> {
    let mut rows = Vec::new();
    for (side_idx, &neighbor_idx) in selected_neighbors.iter().enumerate() {
        if neighbor_idx < 0 {
            continue;
        }
        let Some(component_idx) = runtime.constraint_model.component_for_side(side_idx) else {
            continue;
        };
        rows.push((component_idx, side_idx, neighbor_idx as usize));
    }
    rows
}

fn selected_neighbor_facts_to_py(
    py: Python<'_>,
    runtime: &StereoWalkerRuntimeData,
    selected_neighbors: &[isize],
) -> PyResult<Vec<Py<PyDict>>> {
    selected_neighbor_fact_rows(runtime, selected_neighbors)
        .into_iter()
        .map(|(component_idx, side_idx, neighbor_idx)| {
            let fact = PyDict::new(py);
            fact.set_item("component_idx", component_idx)?;
            fact.set_item("side_idx", side_idx)?;
            fact.set_item("neighbor_idx", neighbor_idx)?;
            Ok(fact.unbind())
        })
        .collect()
}

fn rdkit_traversal_writer_facts_to_py(
    py: Python<'_>,
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    selected_neighbors: &[isize],
) -> PyResult<Vec<Py<PyDict>>> {
    let mut rows = Vec::new();
    for (component_idx, facts) in
        rdkit_traversal_writer_facts_by_component(runtime, state, selected_neighbors)
            .into_iter()
            .enumerate()
    {
        for fact in facts {
            let row = PyDict::new(py);
            row.set_item("component_idx", component_idx)?;
            match fact {
                StereoConstraintFact::CarrierSelected {
                    side_idx,
                    neighbor_idx,
                } => {
                    row.set_item("fact", "carrier_selected")?;
                    row.set_item("side_idx", side_idx)?;
                    row.set_item("neighbor_idx", neighbor_idx)?;
                }
                StereoConstraintFact::CarrierSelectionBlocked {
                    side_idx,
                    neighbor_idx,
                } => {
                    row.set_item("fact", "carrier_selection_blocked")?;
                    row.set_item("side_idx", side_idx)?;
                    row.set_item("neighbor_idx", neighbor_idx)?;
                }
                StereoConstraintFact::CarrierEdgeEmitted {
                    side_idx,
                    begin_idx,
                    end_idx,
                    role,
                } => {
                    row.set_item("fact", "carrier_edge_emitted")?;
                    row.set_item("side_idx", side_idx)?;
                    row.set_item("begin_idx", begin_idx)?;
                    row.set_item("end_idx", end_idx)?;
                    row.set_item("role", stereo_traversal_role_name(role))?;
                }
                StereoConstraintFact::DirectionalMarkerPlaced {
                    side_idx,
                    slot,
                    marker,
                    role,
                } => {
                    row.set_item("fact", "directional_marker_placed")?;
                    row.set_item("side_idx", side_idx)?;
                    row.set_item("slot", slot)?;
                    row.set_item("marker", marker.to_string())?;
                    row.set_item("role", stereo_traversal_role_name(role))?;
                }
            }
            rows.push(row.unbind());
        }
    }
    Ok(rows)
}

fn rdkit_writer_selected_marker_event_facts_by_component(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<Vec<Vec<StereoMarkerEventFact>>> {
    let mut facts_by_component = vec![Vec::new(); runtime.constraint_model.component_count()];
    for trace in state.directional_marker_traces.iter() {
        if trace.side_idx < 0 || trace.edge_begin_idx < 0 || trace.edge_end_idx < 0 {
            continue;
        }
        let side_idx = trace.side_idx as usize;
        let Some(component_idx) = runtime.constraint_model.component_for_side(side_idx) else {
            continue;
        };
        let mut marker_buf = [0; 4];
        let marker = StereoDirectionToken::from_str(trace.marker.encode_utf8(&mut marker_buf))?;
        facts_by_component[component_idx].push(StereoMarkerEventFact::MarkerPlaced {
            side_idx,
            slot: trace.slot,
            begin_idx: trace.edge_begin_idx as usize,
            end_idx: trace.edge_end_idx as usize,
            marker,
            role: trace.role,
        });
    }
    Ok(facts_by_component)
}

fn rdkit_writer_marker_event_facts_by_component_from_traces(
    model: &StereoConstraintModel,
    component_count: usize,
    marker_event_traces: &[MarkerEventTrace],
) -> PyResult<Vec<Vec<StereoMarkerEventFact>>> {
    let mut facts_by_component = vec![Vec::new(); component_count];
    for trace in marker_event_traces {
        if trace.side_idx < 0 || trace.edge_begin_idx < 0 || trace.edge_end_idx < 0 {
            continue;
        }
        let side_idx = trace.side_idx as usize;
        let Some(component_idx) = model.component_for_side(side_idx) else {
            continue;
        };
        let fact = match trace.marker {
            Some(marker) => {
                let mut marker_buf = [0; 4];
                let marker = StereoDirectionToken::from_str(marker.encode_utf8(&mut marker_buf))?;
                StereoMarkerEventFact::MarkerPlaced {
                    side_idx,
                    slot: trace.slot,
                    begin_idx: trace.edge_begin_idx as usize,
                    end_idx: trace.edge_end_idx as usize,
                    marker,
                    role: trace.role,
                }
            }
            None => StereoMarkerEventFact::NoMarker {
                side_idx,
                slot: trace.slot,
                begin_idx: trace.edge_begin_idx as usize,
                end_idx: trace.edge_end_idx as usize,
                role: trace.role,
            },
        };
        facts_by_component[component_idx].push(fact);
    }
    Ok(facts_by_component)
}

fn rdkit_writer_marker_event_facts_by_component(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<Vec<Vec<StereoMarkerEventFact>>> {
    rdkit_writer_marker_event_facts_by_component_from_traces(
        &runtime.constraint_model,
        runtime.constraint_model.component_count(),
        state.marker_event_traces.as_ref(),
    )
}

fn marker_event_facts_to_py(
    py: Python<'_>,
    rdkit_writer_marker_events_by_component: &[Vec<StereoMarkerEventFact>],
) -> PyResult<Vec<Py<PyDict>>> {
    let mut rows = Vec::new();
    for (component_idx, facts) in rdkit_writer_marker_events_by_component.iter().enumerate() {
        for &fact in facts {
            let row = PyDict::new(py);
            row.set_item("component_idx", component_idx)?;
            match fact {
                StereoMarkerEventFact::MarkerPlaced {
                    side_idx,
                    slot,
                    begin_idx,
                    end_idx,
                    marker,
                    role,
                } => {
                    row.set_item("event", "marker_placed")?;
                    row.set_item("side_idx", side_idx)?;
                    row.set_item("slot", slot)?;
                    row.set_item("begin_idx", begin_idx)?;
                    row.set_item("end_idx", end_idx)?;
                    row.set_item("marker", stereo_direction_token_name(marker))?;
                    row.set_item("role", stereo_traversal_role_name(role))?;
                }
                StereoMarkerEventFact::NoMarker {
                    side_idx,
                    slot,
                    begin_idx,
                    end_idx,
                    role,
                } => {
                    row.set_item("event", "no_marker")?;
                    row.set_item("side_idx", side_idx)?;
                    row.set_item("slot", slot)?;
                    row.set_item("begin_idx", begin_idx)?;
                    row.set_item("end_idx", end_idx)?;
                    row.set_item("marker", Option::<&str>::None)?;
                    row.set_item("role", stereo_traversal_role_name(role))?;
                }
            }
            rows.push(row.unbind());
        }
    }
    Ok(rows)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RdkitWriterMarkerObligationDomain {
    component_idx: usize,
    no_marker_event: StereoMarkerEventFact,
    same_edge_future_marker_slots: Vec<usize>,
    same_side_other_edge_future_markers: Vec<(usize, (usize, usize))>,
}

impl RdkitWriterMarkerObligationDomain {
    fn is_deferred(&self) -> bool {
        !self.same_edge_future_marker_slots.is_empty()
    }

    fn no_marker_key(&self) -> Option<(usize, usize, (usize, usize))> {
        match self.no_marker_event {
            StereoMarkerEventFact::NoMarker {
                side_idx,
                slot,
                begin_idx,
                end_idx,
                ..
            } => Some((side_idx, slot, canonical_edge(begin_idx, end_idx))),
            StereoMarkerEventFact::MarkerPlaced { .. } => None,
        }
    }
}

fn rdkit_writer_marker_obligation_domains_for_component(
    component_idx: usize,
    facts: &[StereoMarkerEventFact],
) -> Vec<RdkitWriterMarkerObligationDomain> {
    let future_markers_by_side = facts
        .iter()
        .filter_map(|&fact| match fact {
            StereoMarkerEventFact::MarkerPlaced {
                side_idx,
                slot,
                begin_idx,
                end_idx,
                ..
            } => Some((side_idx, slot, canonical_edge(begin_idx, end_idx))),
            StereoMarkerEventFact::NoMarker { .. } => None,
        })
        .fold(
            BTreeMap::<usize, Vec<(usize, (usize, usize))>>::new(),
            |mut acc, (side_idx, slot, edge)| {
                acc.entry(side_idx).or_default().push((slot, edge));
                acc
            },
        );

    facts
        .iter()
        .filter_map(|&fact| match fact {
            StereoMarkerEventFact::MarkerPlaced { .. } => None,
            StereoMarkerEventFact::NoMarker {
                side_idx,
                slot,
                begin_idx,
                end_idx,
                ..
            } => {
                let edge = canonical_edge(begin_idx, end_idx);
                let future_markers = future_markers_by_side
                    .get(&side_idx)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                let same_edge_future_marker_slots = future_markers
                    .iter()
                    .filter_map(|&(future_slot, future_edge)| {
                        (future_slot > slot && future_edge == edge).then_some(future_slot)
                    })
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                let same_side_other_edge_future_markers = future_markers
                    .iter()
                    .filter_map(|&(future_slot, future_edge)| {
                        (future_slot > slot && future_edge != edge)
                            .then_some((future_slot, future_edge))
                    })
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                Some(RdkitWriterMarkerObligationDomain {
                    component_idx,
                    no_marker_event: fact,
                    same_edge_future_marker_slots,
                    same_side_other_edge_future_markers,
                })
            }
        })
        .collect()
}

fn rdkit_writer_marker_obligation_domains_by_component(
    rdkit_writer_marker_events_by_component: &[Vec<StereoMarkerEventFact>],
) -> Vec<Vec<RdkitWriterMarkerObligationDomain>> {
    rdkit_writer_marker_events_by_component
        .iter()
        .enumerate()
        .map(|(component_idx, facts)| {
            rdkit_writer_marker_obligation_domains_for_component(component_idx, facts)
        })
        .collect()
}

fn rdkit_writer_marker_obligation_domains_to_py(
    py: Python<'_>,
    domains_by_component: &[Vec<RdkitWriterMarkerObligationDomain>],
) -> PyResult<Vec<Py<PyDict>>> {
    let mut rows = Vec::new();
    for domains in domains_by_component {
        for domain in domains {
            let row = PyDict::new(py);
            row.set_item("component_idx", domain.component_idx)?;
            if let StereoMarkerEventFact::NoMarker {
                side_idx,
                slot,
                begin_idx,
                end_idx,
                role,
            } = domain.no_marker_event
            {
                let edge = canonical_edge(begin_idx, end_idx);
                row.set_item("side_idx", side_idx)?;
                row.set_item("slot", slot)?;
                row.set_item("begin_idx", begin_idx)?;
                row.set_item("end_idx", end_idx)?;
                row.set_item("canonical_edge", edge)?;
                row.set_item("role", stereo_traversal_role_name(role))?;
                row.set_item("is_deferred", domain.is_deferred())?;
                row.set_item(
                    "same_edge_future_marker_slots",
                    domain.same_edge_future_marker_slots.clone(),
                )?;
                let other_edge_markers = domain
                    .same_side_other_edge_future_markers
                    .iter()
                    .map(|&(future_slot, future_edge)| {
                        let marker = PyDict::new(py);
                        marker.set_item("slot", future_slot)?;
                        marker.set_item("canonical_edge", future_edge)?;
                        Ok(marker.unbind())
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                row.set_item("same_side_other_edge_future_markers", other_edge_markers)?;
            }
            rows.push(row.unbind());
        }
    }
    Ok(rows)
}

fn rdkit_writer_slot_coalesced_marker_event_facts(
    component_idx: usize,
    marker_event_facts: &[StereoMarkerEventFact],
) -> Vec<StereoMarkerEventFact> {
    let deferred_no_marker_keys =
        rdkit_writer_marker_obligation_domains_for_component(component_idx, marker_event_facts)
            .into_iter()
            .filter(|domain| domain.is_deferred())
            .filter_map(|domain| domain.no_marker_key())
            .collect::<BTreeSet<_>>();

    marker_event_facts
        .iter()
        .copied()
        .filter(|&fact| match fact {
            StereoMarkerEventFact::MarkerPlaced { .. } => true,
            StereoMarkerEventFact::NoMarker {
                side_idx,
                slot,
                begin_idx,
                end_idx,
                ..
            } => !deferred_no_marker_keys.contains(&(
                side_idx,
                slot,
                canonical_edge(begin_idx, end_idx),
            )),
        })
        .collect()
}

fn rdkit_writer_slot_coalesced_marker_event_facts_by_component(
    rdkit_writer_marker_events_by_component: &[Vec<StereoMarkerEventFact>],
) -> Vec<Vec<StereoMarkerEventFact>> {
    rdkit_writer_marker_events_by_component
        .iter()
        .enumerate()
        .map(|(component_idx, facts)| {
            rdkit_writer_slot_coalesced_marker_event_facts(component_idx, facts)
        })
        .collect()
}

fn selected_neighbors_layer_completions_to_py(
    py: Python<'_>,
    runtime: &StereoWalkerRuntimeData,
    selected_neighbors: &[isize],
) -> PyResult<Py<PyDict>> {
    let completions = PyDict::new(py);
    for layer in StereoConstraintLayer::ALL {
        completions.set_item(
            stereo_constraint_layer_name(layer),
            selected_neighbors_have_constraint_completion(runtime, selected_neighbors, layer),
        )?;
    }
    Ok(completions.unbind())
}

fn rdkit_traversal_writer_layer_completions_to_py(
    py: Python<'_>,
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    selected_neighbors: &[isize],
) -> PyResult<Py<PyDict>> {
    let completions = PyDict::new(py);
    for layer in StereoConstraintLayer::ALL {
        completions.set_item(
            stereo_constraint_layer_name(layer),
            rdkit_traversal_writer_has_completion(runtime, state, selected_neighbors, layer),
        )?;
    }
    Ok(completions.unbind())
}

fn assignment_state_to_py(
    py: Python<'_>,
    runtime: &StereoWalkerRuntimeData,
    facts_by_component: &[Vec<StereoConstraintFact>],
) -> PyResult<Py<PyDict>> {
    let state_by_layer = PyDict::new(py);
    for layer in StereoConstraintLayer::ALL {
        let assignment_state = StereoAssignmentState::from_facts_by_component(
            &runtime.constraint_model,
            layer,
            facts_by_component,
        );
        let components = runtime
            .constraint_model
            .components
            .iter()
            .map(|component| {
                let component_idx = component.component_idx;
                let remaining_assignment_ids = assignment_state
                    .remaining_by_component
                    .get(component_idx)
                    .cloned()
                    .unwrap_or_default();
                let forced_neighbors = component
                    .side_ids
                    .iter()
                    .filter_map(|&side_idx| {
                        let neighbor_idx = assignment_state.forced_neighbor(
                            &runtime.constraint_model,
                            component_idx,
                            side_idx,
                        )?;
                        Some((side_idx, neighbor_idx))
                    })
                    .map(|(side_idx, neighbor_idx)| {
                        let row = PyDict::new(py);
                        row.set_item("side_idx", side_idx)?;
                        row.set_item("neighbor_idx", neighbor_idx)?;
                        Ok(row.unbind())
                    })
                    .collect::<PyResult<Vec<_>>>()?;

                let row = PyDict::new(py);
                row.set_item("component_idx", component_idx)?;
                row.set_item("side_ids", component.side_ids.clone())?;
                row.set_item("remaining_assignment_ids", remaining_assignment_ids.clone())?;
                row.set_item("remaining_count", remaining_assignment_ids.len())?;
                row.set_item("forced_neighbors", forced_neighbors)?;
                Ok(row.unbind())
            })
            .collect::<PyResult<Vec<_>>>()?;
        state_by_layer.set_item(stereo_constraint_layer_name(layer), components)?;
    }
    Ok(state_by_layer.unbind())
}

enum ComponentTokenConstraintFact {
    KnownTokenFlip(StereoTokenFlipFact),
    InferredTokenObservation(StereoTokenObservationFact),
    NoTokenConstraint,
}

impl ComponentTokenConstraintFact {
    fn kind_name(&self) -> &'static str {
        match self {
            Self::KnownTokenFlip(_) => "known_token_flip",
            Self::InferredTokenObservation(_) => "inferred_token_observation",
            Self::NoTokenConstraint => "no_token_constraint",
        }
    }
}

struct ComponentTokenConstraint {
    fact: ComponentTokenConstraintFact,
    inputs: ComponentTokenInferenceInputs,
}

fn committed_component_token_flip_fact_from_state(
    state: &RootedConnectedStereoWalkerStateData,
    component_idx: usize,
) -> Option<CommittedComponentTokenFlipFact> {
    state.committed_component_token_flips[component_idx].map(|token_flip| {
        CommittedComponentTokenFlipFact {
            runtime_component_idx: component_idx,
            token_flip,
        }
    })
}

fn known_token_flip_facts_from_committed_component_token_flips(
    committed_component_token_flips: &[Option<StereoTokenFlip>],
) -> Vec<StereoTokenFlipFact> {
    committed_component_token_flips
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(runtime_component_idx, token_flip)| {
            Some(StereoTokenFlipFact {
                runtime_component_idx,
                token_flip: token_flip?,
            })
        })
        .collect()
}

fn component_token_constraint_from_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    resolved_selected_neighbors: &[isize],
    component_idx: usize,
) -> PyResult<ComponentTokenConstraint> {
    let inputs = component_token_inference_inputs(
        runtime,
        graph,
        state,
        resolved_selected_neighbors,
        component_idx,
    )?;
    if let Some(committed_fact) =
        committed_component_token_flip_fact_from_state(state, component_idx)
    {
        return Ok(ComponentTokenConstraint {
            fact: ComponentTokenConstraintFact::KnownTokenFlip(committed_fact.token_flip_fact()),
            inputs,
        });
    }
    let fact = match inputs.supported_token_observation()? {
        Some(observation) => ComponentTokenConstraintFact::InferredTokenObservation(observation),
        None if inputs.inferred.is_some() => {
            return Err(PyValueError::new_err(format!(
                "Inferred token flip has no supported observation fact for branch {}",
                inputs.inference_branch
            )));
        }
        None => ComponentTokenConstraintFact::NoTokenConstraint,
    };
    Ok(ComponentTokenConstraint { fact, inputs })
}

fn component_token_constraints_from_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    resolved_selected_neighbors: &[isize],
) -> PyResult<Vec<ComponentTokenConstraint>> {
    (0..runtime.isolated_components.len())
        .map(|component_idx| {
            component_token_constraint_from_state(
                runtime,
                graph,
                state,
                resolved_selected_neighbors,
                component_idx,
            )
        })
        .collect()
}

fn partial_component_token_constraints_from_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    resolved_selected_neighbors: &[isize],
) -> PyResult<Vec<ComponentTokenConstraint>> {
    (0..runtime.isolated_components.len())
        .map(|component_idx| {
            component_token_constraint_from_state(
                runtime,
                graph,
                state,
                resolved_selected_neighbors,
                component_idx,
            )
        })
        .collect()
}

fn known_token_flip_facts_from_constraints(
    constraints: &[ComponentTokenConstraint],
) -> Vec<StereoTokenFlipFact> {
    constraints
        .iter()
        .filter_map(|constraint| match constraint.fact {
            ComponentTokenConstraintFact::KnownTokenFlip(fact) => Some(fact),
            ComponentTokenConstraintFact::InferredTokenObservation(_)
            | ComponentTokenConstraintFact::NoTokenConstraint => None,
        })
        .collect()
}

fn inferred_token_observation_facts_from_constraints(
    constraints: &[ComponentTokenConstraint],
) -> Vec<StereoTokenObservationFact> {
    constraints
        .iter()
        .filter_map(|constraint| match constraint.fact {
            ComponentTokenConstraintFact::InferredTokenObservation(fact) => Some(fact),
            ComponentTokenConstraintFact::KnownTokenFlip(_)
            | ComponentTokenConstraintFact::NoTokenConstraint => None,
        })
        .collect()
}

fn supported_token_observation_facts_from_constraints(
    constraints: &[ComponentTokenConstraint],
) -> PyResult<Vec<StereoTokenObservationFact>> {
    let mut facts = Vec::new();
    for constraint in constraints {
        match constraint.inputs.supported_token_observation()? {
            Some(fact) => facts.push(fact),
            None if matches!(
                constraint.fact,
                ComponentTokenConstraintFact::InferredTokenObservation(_)
            ) =>
            {
                return Err(PyValueError::new_err(format!(
                    "Inferred token flip has no supported observation fact for branch {}",
                    constraint.inputs.inference_branch
                )));
            }
            None => {}
        }
    }
    Ok(facts)
}

fn runtime_token_constraint_facts_to_py(
    py: Python<'_>,
    known_token_flip_facts: &[StereoTokenFlipFact],
    inferred_token_observation_facts: &[StereoTokenObservationFact],
) -> PyResult<Py<PyDict>> {
    let row = PyDict::new(py);
    row.set_item(
        "known_token_flip_facts",
        token_flip_facts_to_py(py, known_token_flip_facts)?,
    )?;
    row.set_item(
        "inferred_token_observation_facts",
        token_observation_facts_to_py(py, inferred_token_observation_facts)?,
    )?;
    row.set_item("known_token_flip_count", known_token_flip_facts.len())?;
    row.set_item(
        "inferred_token_observation_count",
        inferred_token_observation_facts.len(),
    )?;
    Ok(row.unbind())
}

fn resolved_constraint_state_from_walker_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    layer: StereoConstraintLayer,
) -> PyResult<StereoConstraintState> {
    let raw_selected_neighbors = state.stereo_selected_neighbors.as_ref();
    let boundary_facts = support_boundary_facts_from_walker_state(
        runtime,
        graph,
        state,
        raw_selected_neighbors,
        true,
    )?;
    boundary_facts.constraint_state(runtime, layer)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TerminalStereoSupportBoundarySummary {
    deferred_marker_obligation_domain_count: usize,
}

fn terminal_stereo_state_support_boundary_summary(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<Option<TerminalStereoSupportBoundarySummary>> {
    let raw_selected_neighbors = state.stereo_selected_neighbors.as_ref();
    let boundary_facts = support_boundary_facts_from_walker_state(
        runtime,
        graph,
        state,
        raw_selected_neighbors,
        true,
    )?;
    let constraint_state =
        boundary_facts.constraint_state(runtime, StereoConstraintLayer::Semantic)?;
    for component in &runtime.constraint_model.components {
        let component_idx = component.component_idx;
        if constraint_state.is_empty(component_idx) {
            return Ok(None);
        }
        let Some(marker_events) = boundary_facts
            .marker_event_facts_by_component
            .get(component_idx)
        else {
            return Err(PyValueError::new_err(
                "support-boundary marker facts do not match the constraint model",
            ));
        };
        let token_phase_assignment_ids = constraint_state
            .token_phase_remaining_by_component
            .get(component_idx)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let survivor_state = rdkit_marker_row_survivor_component_state(
            runtime,
            component_idx,
            token_phase_assignment_ids,
            marker_events,
        )?;
        if survivor_state.row_ids_after_marker_events.is_empty() {
            return Ok(None);
        }
    }
    let deferred_marker_obligation_domain_count = boundary_facts
        .marker_obligation_domains_by_component
        .iter()
        .flatten()
        .filter(|domain| domain.is_deferred())
        .count();
    Ok(Some(TerminalStereoSupportBoundarySummary {
        deferred_marker_obligation_domain_count,
    }))
}

fn terminal_stereo_state_has_support_boundary_marker_placement(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<bool> {
    Ok(terminal_stereo_state_support_boundary_summary(runtime, graph, state)?.is_some())
}

fn assert_token_flips_explained_by_constraint_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<()> {
    let constraint_state = resolved_constraint_state_from_walker_state(
        runtime,
        graph,
        state,
        StereoConstraintLayer::Semantic,
    )?;
    for component_idx in 0..state.committed_component_token_flips.len() {
        let Some(known) = state.committed_component_token_flips[component_idx] else {
            continue;
        };
        let Some(model_component_idx) = runtime
            .constraint_model
            .component_for_runtime_component(component_idx)
        else {
            return Err(PyValueError::new_err(
                "Known token flip references unknown runtime component",
            ));
        };
        if constraint_state.is_empty(model_component_idx) {
            return Err(PyValueError::new_err(
                "Known token flip leaves no compatible stereo assignment",
            ));
        }
        let Some(forced) = constraint_state.forced_token_flip(
            &runtime.constraint_model,
            model_component_idx,
            component_idx,
        ) else {
            return Err(PyValueError::new_err(
                "Known token flip is not forced by stereo constraint state",
            ));
        };
        if forced != known {
            return Err(PyValueError::new_err(
                "Known token flip disagrees with stereo constraint state",
            ));
        }
    }
    Ok(())
}

fn mixed_constraint_state_to_py(
    py: Python<'_>,
    runtime: &StereoWalkerRuntimeData,
    facts_by_component: &[Vec<StereoConstraintFact>],
    token_flip_facts: &[StereoTokenFlipFact],
    token_observation_facts: &[StereoTokenObservationFact],
) -> PyResult<Py<PyDict>> {
    let state_by_layer = PyDict::new(py);
    for layer in StereoConstraintLayer::ALL {
        let constraint_state = StereoConstraintState::from_facts_and_token_observations(
            &runtime.constraint_model,
            layer,
            facts_by_component,
            token_flip_facts,
            token_observation_facts,
        )?;
        let components = constraint_state_components_to_py(py, runtime, &constraint_state)?;
        state_by_layer.set_item(stereo_constraint_layer_name(layer), components)?;
    }
    Ok(state_by_layer.unbind())
}

struct RdkitMarkerRowSurvivorSideDomain {
    side_idx: usize,
    carrier_neighbors: Vec<usize>,
    marker_neighbor_sets: Vec<Vec<usize>>,
}

struct RdkitMarkerRowSurvivorComponentState {
    component_idx: usize,
    token_phase_assignment_ids: Vec<usize>,
    row_ids_before_marker_events: Vec<usize>,
    row_ids_after_marker_events: Vec<usize>,
    token_phase_assignment_ids_after_marker_events: Vec<usize>,
    neighbor_assignment_ids_after_marker_events: Vec<usize>,
    side_domains: Vec<RdkitMarkerRowSurvivorSideDomain>,
}

fn rdkit_marker_row_survivor_component_state(
    runtime: &StereoWalkerRuntimeData,
    component_idx: usize,
    token_phase_assignment_ids: &[usize],
    marker_event_facts: &[StereoMarkerEventFact],
) -> PyResult<RdkitMarkerRowSurvivorComponentState> {
    let Some(component) = runtime.constraint_model.components.get(component_idx) else {
        return Err(PyValueError::new_err(
            "marker row survivor component index out of range",
        ));
    };
    let row_ids_before_marker_events = runtime
        .constraint_model
        .marker_placement_row_ids_for_token_phase_assignment_ids(
            component_idx,
            token_phase_assignment_ids,
        )?;
    let row_ids_after_marker_events = runtime
        .constraint_model
        .filter_marker_placement_row_ids_for_marker_event_facts(
            component_idx,
            &row_ids_before_marker_events,
            marker_event_facts,
        )?;
    let token_phase_assignment_ids_after_marker_events = runtime
        .constraint_model
        .token_phase_assignment_ids_for_marker_placement_row_ids(
            component_idx,
            &row_ids_after_marker_events,
        )?;
    let neighbor_assignment_ids_after_marker_events = runtime
        .constraint_model
        .neighbor_assignment_ids_for_marker_placement_row_ids(
            component_idx,
            &row_ids_after_marker_events,
        )?;
    let side_domains = component
        .side_ids
        .iter()
        .map(|&side_idx| {
            Ok(RdkitMarkerRowSurvivorSideDomain {
                side_idx,
                carrier_neighbors: runtime
                    .constraint_model
                    .available_neighbors_for_assignment_ids(
                        component_idx,
                        side_idx,
                        &neighbor_assignment_ids_after_marker_events,
                    ),
                marker_neighbor_sets: runtime
                    .constraint_model
                    .marker_neighbor_sets_for_marker_placement_row_ids(
                        component_idx,
                        side_idx,
                        &row_ids_after_marker_events,
                    )?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    Ok(RdkitMarkerRowSurvivorComponentState {
        component_idx,
        token_phase_assignment_ids: token_phase_assignment_ids.to_vec(),
        row_ids_before_marker_events,
        row_ids_after_marker_events,
        token_phase_assignment_ids_after_marker_events,
        neighbor_assignment_ids_after_marker_events,
        side_domains,
    })
}

fn rdkit_marker_placement_state_to_py(
    py: Python<'_>,
    runtime: &StereoWalkerRuntimeData,
    facts_by_component: &[Vec<StereoConstraintFact>],
    token_flip_facts: &[StereoTokenFlipFact],
    token_observation_facts: &[StereoTokenObservationFact],
    rdkit_writer_marker_events_by_component: &[Vec<StereoMarkerEventFact>],
) -> PyResult<Py<PyDict>> {
    let state_by_layer = PyDict::new(py);
    for layer in StereoConstraintLayer::ALL {
        let constraint_state = StereoConstraintState::from_facts_and_token_observations(
            &runtime.constraint_model,
            layer,
            facts_by_component,
            token_flip_facts,
            token_observation_facts,
        )?;
        let components = runtime
            .constraint_model
            .components
            .iter()
            .map(|component| {
                let component_idx = component.component_idx;
                let token_phase_assignment_ids = constraint_state
                    .token_phase_remaining_by_component
                    .get(component_idx)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                let marker_event_facts = rdkit_writer_marker_events_by_component
                    .get(component_idx)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                let survivor_state = rdkit_marker_row_survivor_component_state(
                    runtime,
                    component_idx,
                    token_phase_assignment_ids,
                    marker_event_facts,
                )?;
                let rows_after_marker_events = survivor_state
                    .row_ids_after_marker_events
                    .iter()
                    .copied()
                    .map(|row_idx| rdkit_marker_placement_row_to_py(py, component, row_idx))
                    .collect::<PyResult<Vec<_>>>()?;
                let survivor_side_domains = survivor_state
                    .side_domains
                    .iter()
                    .map(|domain| {
                        let row = PyDict::new(py);
                        row.set_item("side_idx", domain.side_idx)?;
                        row.set_item("carrier_neighbors", domain.carrier_neighbors.clone())?;
                        row.set_item("marker_neighbor_sets", domain.marker_neighbor_sets.clone())?;
                        Ok(row.unbind())
                    })
                    .collect::<PyResult<Vec<_>>>()?;

                let row = PyDict::new(py);
                row.set_item("component_idx", survivor_state.component_idx)?;
                row.set_item(
                    "runtime_component_ids",
                    component.runtime_component_ids.clone(),
                )?;
                row.set_item("side_ids", component.side_ids.clone())?;
                row.set_item(
                    "token_phase_assignment_ids",
                    survivor_state.token_phase_assignment_ids.clone(),
                )?;
                row.set_item(
                    "token_phase_assignment_count",
                    survivor_state.token_phase_assignment_ids.len(),
                )?;
                row.set_item(
                    "row_ids_before_marker_events",
                    survivor_state.row_ids_before_marker_events.clone(),
                )?;
                row.set_item(
                    "row_count_before_marker_events",
                    survivor_state.row_ids_before_marker_events.len(),
                )?;
                row.set_item("marker_event_count", marker_event_facts.len())?;
                row.set_item(
                    "row_ids_after_marker_events",
                    survivor_state.row_ids_after_marker_events.clone(),
                )?;
                row.set_item(
                    "token_phase_assignment_ids_after_marker_events",
                    survivor_state
                        .token_phase_assignment_ids_after_marker_events
                        .clone(),
                )?;
                row.set_item(
                    "token_phase_assignment_count_after_marker_events",
                    survivor_state
                        .token_phase_assignment_ids_after_marker_events
                        .len(),
                )?;
                row.set_item(
                    "neighbor_assignment_ids_after_marker_events",
                    survivor_state
                        .neighbor_assignment_ids_after_marker_events
                        .clone(),
                )?;
                row.set_item(
                    "neighbor_assignment_count_after_marker_events",
                    survivor_state
                        .neighbor_assignment_ids_after_marker_events
                        .len(),
                )?;
                row.set_item(
                    "row_count_after_marker_events",
                    survivor_state.row_ids_after_marker_events.len(),
                )?;
                row.set_item(
                    "is_empty_after_marker_events",
                    survivor_state.row_ids_after_marker_events.is_empty(),
                )?;
                row.set_item("survivor_side_domains", survivor_side_domains)?;
                row.set_item("rows_after_marker_events", rows_after_marker_events)?;
                Ok(row.unbind())
            })
            .collect::<PyResult<Vec<_>>>()?;
        state_by_layer.set_item(stereo_constraint_layer_name(layer), components)?;
    }
    Ok(state_by_layer.unbind())
}

fn observation_constraint_state_to_py(
    py: Python<'_>,
    runtime: &StereoWalkerRuntimeData,
    facts_by_component: &[Vec<StereoConstraintFact>],
    token_observation_facts: &[StereoTokenObservationFact],
) -> PyResult<Py<PyDict>> {
    let state_by_layer = PyDict::new(py);
    for layer in StereoConstraintLayer::ALL {
        let constraint_state = StereoConstraintState::from_token_observation_facts(
            &runtime.constraint_model,
            layer,
            facts_by_component,
            token_observation_facts,
        )?;
        let components = constraint_state_components_to_py(py, runtime, &constraint_state)?;
        state_by_layer.set_item(stereo_constraint_layer_name(layer), components)?;
    }
    Ok(state_by_layer.unbind())
}

fn constraint_state_components_to_py(
    py: Python<'_>,
    runtime: &StereoWalkerRuntimeData,
    constraint_state: &StereoConstraintState,
) -> PyResult<Vec<Py<PyDict>>> {
    runtime
        .constraint_model
        .components
        .iter()
        .map(|component| {
            let component_idx = component.component_idx;
            let carrier_assignment_ids = constraint_state
                .carrier_assignment_state
                .remaining_by_component
                .get(component_idx)
                .cloned()
                .unwrap_or_default();
            let token_phase_assignment_ids = constraint_state
                .token_phase_remaining_by_component
                .get(component_idx)
                .cloned()
                .unwrap_or_default();
            let forced_neighbors = component
                .side_ids
                .iter()
                .filter_map(|&side_idx| {
                    let neighbor_idx = constraint_state.forced_neighbor(
                        &runtime.constraint_model,
                        component_idx,
                        side_idx,
                    )?;
                    Some((side_idx, neighbor_idx))
                })
                .map(|(side_idx, neighbor_idx)| {
                    let row = PyDict::new(py);
                    row.set_item("side_idx", side_idx)?;
                    row.set_item("neighbor_idx", neighbor_idx)?;
                    Ok(row.unbind())
                })
                .collect::<PyResult<Vec<_>>>()?;
            let forced_token_flips = component
                .runtime_component_ids
                .iter()
                .filter_map(|&runtime_component_idx| {
                    let token_flip = constraint_state.forced_token_flip(
                        &runtime.constraint_model,
                        component_idx,
                        runtime_component_idx,
                    )?;
                    Some((runtime_component_idx, token_flip))
                })
                .map(|(runtime_component_idx, token_flip)| {
                    let row = PyDict::new(py);
                    row.set_item("runtime_component_idx", runtime_component_idx)?;
                    row.set_item("token_flip", model_token_flip_name(token_flip))?;
                    Ok(row.unbind())
                })
                .collect::<PyResult<Vec<_>>>()?;

            let row = PyDict::new(py);
            row.set_item("component_idx", component_idx)?;
            row.set_item(
                "runtime_component_ids",
                component.runtime_component_ids.clone(),
            )?;
            row.set_item("side_ids", component.side_ids.clone())?;
            row.set_item("is_empty", constraint_state.is_empty(component_idx))?;
            row.set_item("carrier_assignment_ids", carrier_assignment_ids.clone())?;
            row.set_item("carrier_assignment_count", carrier_assignment_ids.len())?;
            row.set_item(
                "token_phase_assignment_ids",
                token_phase_assignment_ids.clone(),
            )?;
            row.set_item(
                "token_phase_assignment_count",
                token_phase_assignment_ids.len(),
            )?;
            row.set_item("forced_neighbors", forced_neighbors)?;
            row.set_item("forced_token_flips", forced_token_flips)?;
            Ok(row.unbind())
        })
        .collect()
}

fn shared_carrier_resolution_to_py(
    py: Python<'_>,
    runtime: &StereoWalkerRuntimeData,
    raw_selected_neighbors: &[isize],
    resolved_selected_neighbors: &[isize],
    assignment_state: &StereoAssignmentState,
) -> PyResult<Vec<Py<PyDict>>> {
    runtime
        .ambiguous_shared_edge_groups
        .iter()
        .map(|group| {
            let left_component_idx = runtime
                .constraint_model
                .component_for_side(group.left_side_idx);
            let right_component_idx = runtime
                .constraint_model
                .component_for_side(group.right_side_idx);
            let left_forced_neighbor = left_component_idx.and_then(|component_idx| {
                assignment_state.forced_neighbor(
                    &runtime.constraint_model,
                    component_idx,
                    group.left_side_idx,
                )
            });
            let right_forced_neighbor = right_component_idx.and_then(|component_idx| {
                assignment_state.forced_neighbor(
                    &runtime.constraint_model,
                    component_idx,
                    group.right_side_idx,
                )
            });
            let left_changed = raw_selected_neighbors[group.left_side_idx]
                != resolved_selected_neighbors[group.left_side_idx];
            let right_changed = raw_selected_neighbors[group.right_side_idx]
                != resolved_selected_neighbors[group.right_side_idx];

            let row = PyDict::new(py);
            row.set_item("left_side_idx", group.left_side_idx)?;
            row.set_item("right_side_idx", group.right_side_idx)?;
            row.set_item("left_component_idx", left_component_idx)?;
            row.set_item("right_component_idx", right_component_idx)?;
            row.set_item("left_shared_neighbor", group.left_shared_neighbor)?;
            row.set_item("right_shared_neighbor", group.right_shared_neighbor)?;
            row.set_item(
                "left_raw_neighbor",
                raw_selected_neighbors[group.left_side_idx],
            )?;
            row.set_item(
                "right_raw_neighbor",
                raw_selected_neighbors[group.right_side_idx],
            )?;
            row.set_item(
                "left_resolved_neighbor",
                resolved_selected_neighbors[group.left_side_idx],
            )?;
            row.set_item(
                "right_resolved_neighbor",
                resolved_selected_neighbors[group.right_side_idx],
            )?;
            row.set_item("left_changed", left_changed)?;
            row.set_item("right_changed", right_changed)?;
            row.set_item("left_forced_neighbor", left_forced_neighbor)?;
            row.set_item("right_forced_neighbor", right_forced_neighbor)?;
            row.set_item(
                "left_change_explained_by_assignment_state",
                !left_changed
                    || left_forced_neighbor == Some(group.left_shared_neighbor)
                        && resolved_selected_neighbors[group.left_side_idx]
                            == group.left_shared_neighbor as isize,
            )?;
            row.set_item(
                "right_change_explained_by_assignment_state",
                !right_changed
                    || right_forced_neighbor == Some(group.right_shared_neighbor)
                        && resolved_selected_neighbors[group.right_side_idx]
                            == group.right_shared_neighbor as isize,
            )?;
            Ok(row.unbind())
        })
        .collect()
}

fn component_token_phase_diagnostics_to_py(
    py: Python<'_>,
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    resolved_selected_neighbors: &[isize],
    assignment_state: &StereoAssignmentState,
    token_constraints: &[ComponentTokenConstraint],
) -> PyResult<Vec<Py<PyDict>>> {
    (0..runtime.isolated_components.len())
        .map(|component_idx| {
            let side_ids = &runtime.side_ids_by_component[component_idx];
            let selected_side_count = side_ids
                .iter()
                .filter(|&&side_idx| resolved_selected_neighbors[side_idx] >= 0)
                .count();
            let mut model_component_idx = None;
            let mut model_component_is_consistent = true;
            for &side_idx in side_ids {
                let Some(current_component_idx) =
                    runtime.constraint_model.component_for_side(side_idx)
                else {
                    model_component_is_consistent = false;
                    break;
                };
                match model_component_idx {
                    None => model_component_idx = Some(current_component_idx),
                    Some(existing_component_idx)
                        if existing_component_idx == current_component_idx => {}
                    Some(_) => {
                        model_component_is_consistent = false;
                        break;
                    }
                }
            }
            if !model_component_is_consistent {
                model_component_idx = None;
            }
            let remaining_assignment_count = assignment_state
                .remaining_by_component
                .get(model_component_idx.unwrap_or(usize::MAX))
                .map(Vec::len)
                .unwrap_or(0);
            let token_inference_inputs = &token_constraints[component_idx].inputs;
            let inferred = token_inference_inputs.inferred;
            let inferred_model_token_flip =
                inferred.and_then(model_token_flip_from_component_value);
            let state_token_flip = state.committed_component_token_flips[component_idx]
                .map(component_token_flip_value_from_model)
                .unwrap_or(UNKNOWN_COMPONENT_TOKEN_FLIP);
            let state_known = state.committed_component_token_flips[component_idx].is_some();
            let inferred_matches_state =
                inferred.is_none_or(|inferred| !state_known || inferred == state_token_flip);
            let remaining_neighbor_assignment_ids = model_component_idx
                .and_then(|idx| assignment_state.remaining_by_component.get(idx))
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let token_phase_assignment_ids_before_token = if let Some(idx) = model_component_idx {
                runtime
                    .constraint_model
                    .token_phase_assignment_ids_for_neighbor_assignment_ids(
                        idx,
                        remaining_neighbor_assignment_ids,
                        &[],
                    )?
            } else {
                Vec::new()
            };
            let shadow_inferred_token_flip_constraints = inferred_model_token_flip
                .map(|token_flip| {
                    vec![StereoTokenFlipFact {
                        runtime_component_idx: component_idx,
                        token_flip,
                    }]
                })
                .unwrap_or_default();
            let shadow_token_flip_assignment_ids_after_token =
                if let Some(idx) = model_component_idx {
                    runtime
                        .constraint_model
                        .token_phase_assignment_ids_for_neighbor_assignment_ids(
                            idx,
                            remaining_neighbor_assignment_ids,
                            &shadow_inferred_token_flip_constraints,
                        )?
                } else {
                    Vec::new()
                };
            let shadow_token_flip_forced_model_token_flip = model_component_idx.and_then(|idx| {
                runtime
                    .constraint_model
                    .forced_token_flip_for_token_phase_assignment_ids(
                        idx,
                        component_idx,
                        &shadow_token_flip_assignment_ids_after_token,
                    )
            });
            let model_token_phase_component_count = model_component_idx
                .and_then(|idx| runtime.constraint_model.components.get(idx))
                .map(|component| component.runtime_component_ids.len())
                .unwrap_or(0);
            let token_observation_fact = token_inference_inputs.supported_token_observation()?;
            let token_observation_facts = token_observation_fact
                .into_iter()
                .collect::<Vec<StereoTokenObservationFact>>();
            let token_observation_assignment_ids_after_token = match model_component_idx {
                Some(idx) => runtime
                    .constraint_model
                    .token_phase_assignment_ids_for_token_observation_facts(
                        idx,
                        remaining_neighbor_assignment_ids,
                        &token_observation_facts,
                    )?,
                None => Vec::new(),
            };
            let token_observation_forced_model_token_flip = model_component_idx.and_then(|idx| {
                runtime
                    .constraint_model
                    .forced_token_flip_for_token_phase_assignment_ids(
                        idx,
                        component_idx,
                        &token_observation_assignment_ids_after_token,
                    )
            });
            let token_observation_unsupported_reason = if token_observation_facts.is_empty() {
                Some(match token_inference_inputs.inference_branch {
                    "isolated_all_single_candidate" => "missing_required_observation_inputs",
                    "isolated_selected_begin_side" => "missing_required_observation_inputs",
                    "coupled_one_candidate_begin_side" => "missing_required_observation_inputs",
                    "coupled_two_candidate_begin_side" => "missing_required_observation_inputs",
                    _ => "unsupported_observation_branch",
                })
            } else {
                None
            };
            let token_observation_matches_inferred_flip = !token_observation_facts.is_empty()
                && token_observation_forced_model_token_flip == inferred_model_token_flip;

            let row = PyDict::new(py);
            row.set_item("component_idx", component_idx)?;
            row.set_item("model_component_idx", model_component_idx)?;
            row.set_item(
                "model_component_is_consistent",
                model_component_is_consistent,
            )?;
            row.set_item("side_ids", side_ids.clone())?;
            row.set_item("side_count", side_ids.len())?;
            row.set_item("selected_side_count", selected_side_count)?;
            row.set_item("is_isolated", runtime.isolated_components[component_idx])?;
            row.set_item(
                "component_phase",
                component_phase_name(state.stereo_component_phases[component_idx]),
            )?;
            row.set_item(
                "component_phase_value",
                state.stereo_component_phases[component_idx],
            )?;
            row.set_item(
                "component_begin_atom_idx",
                state.stereo_component_begin_atoms[component_idx],
            )?;
            row.set_item(
                "rdkit_token_flip_adjustment",
                token_inference_inputs
                    .observations
                    .rdkit_token_flip_adjustment
                    .value(),
            )?;
            row.set_item(
                "state_token_flip",
                component_token_flip_name(state_token_flip),
            )?;
            row.set_item("state_token_flip_value", state_token_flip)?;
            row.set_item(
                "inferred_token_flip",
                inferred.map(component_token_flip_name),
            )?;
            row.set_item("inferred_token_flip_value", inferred)?;
            row.set_item("inferred_matches_state", inferred_matches_state)?;
            row.set_item(
                "token_constraint_kind",
                token_constraints[component_idx].fact.kind_name(),
            )?;
            row.set_item("remaining_assignment_count", remaining_assignment_count)?;
            row.set_item(
                "model_token_phase_component_count",
                model_token_phase_component_count,
            )?;
            row.set_item(
                "token_phase_assignment_count_before_token",
                token_phase_assignment_ids_before_token.len(),
            )?;
            row.set_item(
                "token_observation_facts",
                token_observation_facts_to_py(py, &token_observation_facts)?,
            )?;
            row.set_item(
                "token_observation_supported_branch",
                !token_observation_facts.is_empty(),
            )?;
            row.set_item(
                "token_observation_unsupported_reason",
                token_observation_unsupported_reason,
            )?;
            row.set_item(
                "token_observation_assignment_count_before",
                token_phase_assignment_ids_before_token.len(),
            )?;
            row.set_item(
                "token_observation_assignment_count_after",
                token_observation_assignment_ids_after_token.len(),
            )?;
            row.set_item(
                "token_observation_forced_flip",
                token_observation_forced_model_token_flip.map(model_token_flip_name),
            )?;
            row.set_item(
                "token_observation_matches_inferred_flip",
                token_observation_matches_inferred_flip,
            )?;
            row.set_item(
                "carrier_assignment_singleton",
                remaining_assignment_count == 1,
            )?;
            row.set_item(
                "needs_token_phase_assignment_dimension",
                inferred.is_some()
                    && model_component_idx.is_some()
                    && remaining_assignment_count == 1,
            )?;
            row.set_item(
                "token_phase_dimension_explains_inferred_flip",
                inferred_model_token_flip.is_some()
                    && model_token_phase_component_count > 0
                    && token_observation_assignment_ids_after_token.len() * 2
                        == token_phase_assignment_ids_before_token.len()
                    && token_observation_forced_model_token_flip == inferred_model_token_flip,
            )?;
            let shadow_debug = PyDict::new(py);
            shadow_debug.set_item(
                "token_flip_assignment_count_after_token",
                shadow_token_flip_assignment_ids_after_token.len(),
            )?;
            shadow_debug.set_item(
                "token_flip_forced_flip",
                shadow_token_flip_forced_model_token_flip.map(model_token_flip_name),
            )?;
            shadow_debug.set_item(
                "token_flip_matches_observation_backed_state",
                shadow_token_flip_assignment_ids_after_token
                    == token_observation_assignment_ids_after_token
                    && shadow_token_flip_forced_model_token_flip
                        == token_observation_forced_model_token_flip,
            )?;
            row.set_item("shadow_debug", shadow_debug)?;
            row.set_item(
                "token_flip_inference_inputs",
                component_token_inference_inputs_to_py(py, token_inference_inputs)?,
            )?;
            Ok(row.unbind())
        })
        .collect()
}

fn component_phase_boundary_facts_to_py(
    py: Python<'_>,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<Py<PyDict>> {
    let phase_facts = PyList::empty(py);
    let begin_atom_facts = PyList::empty(py);
    for component_idx in 0..state.stereo_component_phases.len() {
        let phase = state.stereo_component_phases[component_idx];
        if phase != UNKNOWN_COMPONENT_PHASE {
            let fact = ComponentPhaseCommitFact {
                component_idx,
                phase,
            };
            let row = PyDict::new(py);
            row.set_item("fact", "component_phase")?;
            row.set_item("component_idx", fact.component_idx)?;
            row.set_item("phase", component_phase_name(fact.phase))?;
            row.set_item("phase_value", fact.phase)?;
            row.set_item("source", "state")?;
            phase_facts.append(row)?;
        }

        let begin_atom_idx = state.stereo_component_begin_atoms[component_idx];
        if begin_atom_idx >= 0 {
            let fact = ComponentBeginAtomFact {
                component_idx,
                begin_atom_idx: begin_atom_idx as usize,
            };
            let row = PyDict::new(py);
            row.set_item("fact", "component_begin_atom")?;
            row.set_item("component_idx", fact.component_idx)?;
            row.set_item("begin_atom_idx", fact.begin_atom_idx)?;
            row.set_item("source", "state")?;
            begin_atom_facts.append(row)?;
        }
    }

    let out = PyDict::new(py);
    out.set_item("phase_facts", phase_facts)?;
    out.set_item("begin_atom_facts", begin_atom_facts)?;
    out.set_item(
        "phase_fact_count",
        state
            .stereo_component_phases
            .iter()
            .filter(|&&phase| phase != UNKNOWN_COMPONENT_PHASE)
            .count(),
    )?;
    out.set_item(
        "begin_atom_fact_count",
        state
            .stereo_component_begin_atoms
            .iter()
            .filter(|&&begin_atom_idx| begin_atom_idx >= 0)
            .count(),
    )?;
    Ok(out.unbind())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DirectionalMarkerSlot {
    slot: usize,
    marker: char,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RingLabelSpan {
    label: String,
    start_slot: usize,
    end_slot: usize,
}

fn direction_marker_slots(smiles: &str) -> (String, Vec<DirectionalMarkerSlot>) {
    let mut skeleton = String::with_capacity(smiles.len());
    let mut markers = Vec::new();
    for ch in smiles.chars() {
        if ch == '/' || ch == '\\' {
            markers.push(DirectionalMarkerSlot {
                slot: skeleton.chars().count(),
                marker: ch,
            });
        } else {
            skeleton.push(ch);
        }
    }
    (skeleton, markers)
}

fn ring_label_spans(skeleton: &str) -> Vec<RingLabelSpan> {
    let chars = skeleton.chars().collect::<Vec<_>>();
    let mut spans = Vec::new();
    let mut idx = 0usize;
    while idx < chars.len() {
        let ch = chars[idx];
        if ch.is_ascii_digit() {
            spans.push(RingLabelSpan {
                label: ch.to_string(),
                start_slot: idx,
                end_slot: idx + 1,
            });
            idx += 1;
            continue;
        }
        if ch == '%'
            && idx + 2 < chars.len()
            && chars[idx + 1].is_ascii_digit()
            && chars[idx + 2].is_ascii_digit()
        {
            spans.push(RingLabelSpan {
                label: chars[idx..idx + 3].iter().collect(),
                start_slot: idx,
                end_slot: idx + 3,
            });
            idx += 3;
            continue;
        }
        idx += 1;
    }
    spans
}

fn smiles_from_direction_marker_slots(skeleton: &str, markers: &[DirectionalMarkerSlot]) -> String {
    let mut markers_by_slot = BTreeMap::<usize, Vec<char>>::new();
    for marker in markers {
        markers_by_slot
            .entry(marker.slot)
            .or_default()
            .push(marker.marker);
    }

    let chars = skeleton.chars().collect::<Vec<_>>();
    let mut out = String::with_capacity(skeleton.len() + markers.len());
    for slot in 0..=chars.len() {
        for marker in markers_by_slot.get(&slot).into_iter().flatten() {
            out.push(*marker);
        }
        if let Some(ch) = chars.get(slot) {
            out.push(*ch);
        }
    }
    out
}

fn rdkit_ring_closure_projected_marker_slots(
    state: &RootedConnectedStereoWalkerStateData,
) -> (String, Vec<DirectionalMarkerSlot>) {
    let (skeleton, markers) = direction_marker_slots(state.prefix.as_ref());
    let markers_by_slot = markers
        .iter()
        .map(|marker| (marker.slot, marker.marker))
        .collect::<BTreeMap<_, _>>();
    let trace_roles_by_slot = state
        .directional_marker_traces
        .iter()
        .map(|trace| (trace.slot, trace.role))
        .collect::<BTreeMap<_, _>>();
    let skeleton_chars = skeleton.chars().collect::<Vec<_>>();
    let mut moved_slots = BTreeSet::<usize>::new();
    let mut rewritten = Vec::<DirectionalMarkerSlot>::new();
    let mut spans_by_label = BTreeMap::<String, Vec<RingLabelSpan>>::new();
    for span in ring_label_spans(&skeleton) {
        spans_by_label
            .entry(span.label.clone())
            .or_default()
            .push(span);
    }

    for spans in spans_by_label.values() {
        for pair in spans.windows(2) {
            let left = &pair[0];
            let right = &pair[1];
            let Some(&marker) = markers_by_slot.get(&left.end_slot) else {
                continue;
            };
            if trace_roles_by_slot.get(&left.end_slot) != Some(&StereoTraversalRole::RingOpen) {
                continue;
            }
            let closure_is_bracket_atom = right
                .start_slot
                .checked_sub(1)
                .and_then(|slot| skeleton_chars.get(slot))
                == Some(&']');
            if !closure_is_bracket_atom || markers_by_slot.contains_key(&right.start_slot) {
                continue;
            }
            moved_slots.insert(left.end_slot);
            rewritten.push(DirectionalMarkerSlot {
                slot: right.start_slot,
                marker,
            });
        }
    }

    rewritten.extend(
        markers
            .into_iter()
            .filter(|marker| !moved_slots.contains(&marker.slot)),
    );
    rewritten.sort_by_key(|marker| marker.slot);
    (skeleton, rewritten)
}

fn directional_marker_slots_to_py(
    py: Python<'_>,
    markers: &[DirectionalMarkerSlot],
) -> PyResult<Vec<Py<PyDict>>> {
    markers
        .iter()
        .map(|marker| {
            let row = PyDict::new(py);
            row.set_item("slot", marker.slot)?;
            row.set_item("marker", marker.marker.to_string())?;
            Ok(row.unbind())
        })
        .collect()
}

fn ring_closure_marker_projection_to_py(
    py: Python<'_>,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<Py<PyDict>> {
    let (skeleton, marker_slots) = rdkit_ring_closure_projected_marker_slots(state);
    let projection = PyDict::new(py);
    projection.set_item("direction_erased_skeleton", &skeleton)?;
    projection.set_item(
        "marker_slots",
        directional_marker_slots_to_py(py, &marker_slots)?,
    )?;
    projection.set_item(
        "smiles",
        smiles_from_direction_marker_slots(&skeleton, &marker_slots),
    )?;
    Ok(projection.unbind())
}

fn directional_spelling_summary_to_py(py: Python<'_>, smiles: &str) -> PyResult<Py<PyDict>> {
    let mut total_count = 0usize;
    let mut ring_digit_adjacent_count = 0usize;
    let mut direction_erased_skeleton = String::with_capacity(smiles.len());
    let mut ordered_markers = Vec::<String>::new();
    let mut marker_slots = Vec::<Py<PyDict>>::new();

    let chars = smiles.chars().collect::<Vec<_>>();
    for (idx, &ch) in chars.iter().enumerate() {
        if ch == '/' || ch == '\\' {
            let slot = direction_erased_skeleton.chars().count();
            total_count += 1;
            ordered_markers.push(ch.to_string());

            let previous_char = idx.checked_sub(1).and_then(|offset| chars.get(offset));
            let next_char = chars.get(idx + 1);
            let ring_digit_adjacent = previous_char
                .is_some_and(|value| value.is_ascii_digit() || *value == '%')
                || next_char.is_some_and(|value| value.is_ascii_digit() || *value == '%');
            if ring_digit_adjacent {
                ring_digit_adjacent_count += 1;
            }

            let marker_slot = PyDict::new(py);
            marker_slot.set_item("slot", slot)?;
            marker_slot.set_item("marker", ch.to_string())?;
            marker_slot.set_item(
                "after_ring_label",
                previous_char.is_some_and(|value| value.is_ascii_digit() || *value == '%'),
            )?;
            marker_slot.set_item(
                "before_ring_label",
                next_char.is_some_and(|value| value.is_ascii_digit() || *value == '%'),
            )?;
            marker_slot.set_item("before_bracket_atom", next_char == Some(&'['))?;
            marker_slot.set_item("after_branch_open", previous_char == Some(&'('))?;
            marker_slots.push(marker_slot.unbind());
            continue;
        }
        direction_erased_skeleton.push(ch);
    }

    let summary = PyDict::new(py);
    summary.set_item("total", total_count)?;
    summary.set_item("ring_digit_adjacent", ring_digit_adjacent_count)?;
    summary.set_item("non_ring", total_count - ring_digit_adjacent_count)?;
    summary.set_item("direction_erased_skeleton", direction_erased_skeleton)?;
    summary.set_item("ordered_markers", ordered_markers)?;
    summary.set_item("marker_slots", marker_slots)?;
    Ok(summary.unbind())
}

fn marker_local_role(previous_char: Option<&char>, next_char: Option<&char>) -> &'static str {
    if previous_char.is_some_and(|value| value.is_ascii_digit() || *value == '%') {
        "after_ring_label"
    } else if next_char.is_some_and(|value| value.is_ascii_digit() || *value == '%') {
        "before_ring_label"
    } else if previous_char == Some(&'(') {
        "branch_edge"
    } else if next_char == Some(&'[') {
        "before_bracket_atom"
    } else {
        "tree_or_chain_edge"
    }
}

fn directional_marker_provenance_to_py(
    py: Python<'_>,
    graph: &PreparedSmilesGraphData,
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<Vec<Py<PyDict>>> {
    let smiles = state.prefix.as_ref();
    let chars = smiles.chars().collect::<Vec<_>>();
    let mut skeleton_slot = 0usize;
    let mut local_roles = BTreeMap::<usize, &'static str>::new();

    for (smiles_offset, &ch) in chars.iter().enumerate() {
        if ch != '/' && ch != '\\' {
            skeleton_slot += 1;
            continue;
        }

        let previous_char = smiles_offset
            .checked_sub(1)
            .and_then(|offset| chars.get(offset));
        let next_char = chars.get(smiles_offset + 1);
        local_roles.insert(skeleton_slot, marker_local_role(previous_char, next_char));
    }

    let mut provenance = Vec::<Py<PyDict>>::new();
    for (marker_idx, trace) in state.directional_marker_traces.iter().enumerate() {
        let row = PyDict::new(py);
        row.set_item("marker_idx", marker_idx)?;
        row.set_item("slot", trace.slot)?;
        row.set_item("marker", trace.marker.to_string())?;
        row.set_item(
            "local_role",
            local_roles
                .get(&trace.slot)
                .copied()
                .unwrap_or("unknown_marker_slot"),
        )?;
        row.set_item("trace_role", stereo_traversal_role_name(trace.role))?;
        row.set_item("component_idx", trace.component_idx)?;
        row.set_item("side_idx", trace.side_idx)?;
        row.set_item("endpoint_atom_idx", trace.endpoint_atom_idx)?;
        row.set_item("selected_neighbor_idx", trace.selected_neighbor_idx)?;
        row.set_item(
            "support_boundary_selected_neighbor_idx",
            trace.support_boundary_selected_neighbor_idx,
        )?;
        row.set_item("edge_begin_idx", trace.edge_begin_idx)?;
        row.set_item("edge_end_idx", trace.edge_end_idx)?;

        if trace.edge_begin_idx >= 0 && trace.edge_end_idx >= 0 {
            let begin_idx = trace.edge_begin_idx as usize;
            let end_idx = trace.edge_end_idx as usize;
            let canonical = canonical_edge(begin_idx, end_idx);
            row.set_item("canonical_edge", canonical)?;
            row.set_item("bond_idx", graph.bond_index(begin_idx, end_idx))?;
            row.set_item(
                "edge_side_ids",
                runtime
                    .edge_to_side_ids
                    .get(&canonical)
                    .cloned()
                    .unwrap_or_default(),
            )?;
        }
        provenance.push(row.unbind());
    }

    Ok(provenance)
}

fn stereo_output_fact_row_to_py(
    py: Python<'_>,
    graph: &PreparedSmilesGraphData,
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<Py<PyDict>> {
    let raw_selected_neighbors = state.stereo_selected_neighbors.as_ref();
    let boundary_facts = support_boundary_facts_from_walker_state(
        runtime,
        graph,
        state,
        raw_selected_neighbors,
        true,
    )?;
    let legacy_field_resolved_selected_neighbors = resolved_selected_neighbors(runtime, state);
    let assignment_state_resolved_selected_neighbors =
        resolved_selected_neighbors_from_assignment_state(runtime, raw_selected_neighbors);
    let joined_support_boundary_selected_neighbors =
        joined_support_boundary_selected_neighbors(runtime, graph, state)?;
    let support_state_selected_neighbor_query = support_state_selected_neighbor_query(
        runtime,
        raw_selected_neighbors,
        &boundary_facts,
        true,
    )?;
    let resolved_selected_neighbors = joined_support_boundary_selected_neighbors.clone();
    let raw_facts_by_component =
        selected_neighbor_facts_by_component(runtime, raw_selected_neighbors);
    let resolved_facts_by_component =
        selected_neighbor_facts_by_component(runtime, &resolved_selected_neighbors);
    let traversal_facts_by_component =
        rdkit_traversal_writer_facts_by_component(runtime, state, &resolved_selected_neighbors);
    let marker_events_by_component =
        rdkit_writer_selected_marker_event_facts_by_component(runtime, state)?;
    let marker_obligation_domains =
        rdkit_writer_marker_obligation_domains_by_component(&marker_events_by_component);
    let marker_obligation_events_by_component =
        rdkit_writer_slot_coalesced_marker_event_facts_by_component(&marker_events_by_component);
    let support_boundary_marker_events_by_component =
        &boundary_facts.marker_event_facts_by_component;
    let support_boundary_marker_obligation_domains =
        &boundary_facts.marker_obligation_domains_by_component;
    let support_boundary_marker_obligation_events_by_component =
        &boundary_facts.marker_obligation_event_facts_by_component;
    let raw_semantic_assignment_state = StereoAssignmentState::from_facts_by_component(
        &runtime.constraint_model,
        StereoConstraintLayer::Semantic,
        &raw_facts_by_component,
    );
    let resolved_semantic_assignment_state = StereoAssignmentState::from_facts_by_component(
        &runtime.constraint_model,
        StereoConstraintLayer::Semantic,
        &resolved_facts_by_component,
    );
    let token_constraints = component_token_constraints_from_state(
        runtime,
        graph,
        state,
        &resolved_selected_neighbors,
    )?;
    let known_token_flip_facts = boundary_facts.known_token_flip_facts.clone();
    let inferred_token_observation_facts = boundary_facts.inferred_token_observation_facts.clone();
    let supported_token_observation_facts =
        supported_token_observation_facts_from_constraints(&token_constraints)?;

    let row = PyDict::new(py);
    row.set_item("root_idx", runtime.root_idx)?;
    row.set_item("smiles", state.prefix.as_ref())?;
    row.set_item(
        "directional_spelling",
        directional_spelling_summary_to_py(py, state.prefix.as_ref())?,
    )?;
    row.set_item(
        "directional_marker_provenance",
        directional_marker_provenance_to_py(py, graph, runtime, state)?,
    )?;
    row.set_item(
        "ring_closure_marker_projection",
        ring_closure_marker_projection_to_py(py, state)?,
    )?;
    row.set_item(
        "raw_facts",
        selected_neighbor_facts_to_py(py, runtime, raw_selected_neighbors)?,
    )?;
    row.set_item(
        "resolved_facts",
        selected_neighbor_facts_to_py(py, runtime, &resolved_selected_neighbors)?,
    )?;
    row.set_item(
        "traversal_facts",
        rdkit_traversal_writer_facts_to_py(py, runtime, state, &resolved_selected_neighbors)?,
    )?;
    row.set_item(
        "marker_event_facts",
        marker_event_facts_to_py(py, &marker_events_by_component)?,
    )?;
    row.set_item(
        "marker_obligation_facts",
        marker_event_facts_to_py(py, &marker_obligation_events_by_component)?,
    )?;
    row.set_item(
        "marker_obligation_domains",
        rdkit_writer_marker_obligation_domains_to_py(py, &marker_obligation_domains)?,
    )?;
    row.set_item(
        "raw_layer_completions",
        selected_neighbors_layer_completions_to_py(py, runtime, raw_selected_neighbors)?,
    )?;
    row.set_item(
        "resolved_layer_completions",
        selected_neighbors_layer_completions_to_py(py, runtime, &resolved_selected_neighbors)?,
    )?;
    row.set_item(
        "traversal_layer_completions",
        rdkit_traversal_writer_layer_completions_to_py(
            py,
            runtime,
            state,
            &resolved_selected_neighbors,
        )?,
    )?;
    row.set_item(
        "raw_assignment_state",
        assignment_state_to_py(py, runtime, &raw_facts_by_component)?,
    )?;
    row.set_item(
        "resolved_assignment_state",
        assignment_state_to_py(py, runtime, &resolved_facts_by_component)?,
    )?;
    row.set_item(
        "traversal_assignment_state",
        assignment_state_to_py(py, runtime, &traversal_facts_by_component)?,
    )?;
    row.set_item(
        "resolved_constraint_state",
        mixed_constraint_state_to_py(
            py,
            runtime,
            &resolved_facts_by_component,
            &known_token_flip_facts,
            &inferred_token_observation_facts,
        )?,
    )?;
    let support_boundary = PyDict::new(py);
    support_boundary.set_item(
        "marker_event_facts",
        marker_event_facts_to_py(py, support_boundary_marker_events_by_component)?,
    )?;
    support_boundary.set_item(
        "marker_obligation_facts",
        marker_event_facts_to_py(py, support_boundary_marker_obligation_events_by_component)?,
    )?;
    support_boundary.set_item(
        "marker_obligation_domains",
        rdkit_writer_marker_obligation_domains_to_py(
            py,
            support_boundary_marker_obligation_domains,
        )?,
    )?;
    support_boundary.set_item(
        "marker_placement_state",
        rdkit_marker_placement_state_to_py(
            py,
            runtime,
            &resolved_facts_by_component,
            &known_token_flip_facts,
            &inferred_token_observation_facts,
            support_boundary_marker_events_by_component,
        )?,
    )?;
    support_boundary.set_item(
        "marker_obligation_state",
        rdkit_marker_placement_state_to_py(
            py,
            runtime,
            &resolved_facts_by_component,
            &known_token_flip_facts,
            &inferred_token_observation_facts,
            support_boundary_marker_obligation_events_by_component,
        )?,
    )?;
    support_boundary.set_item(
        "selected_neighbor_query",
        support_state_selected_neighbor_query_to_py(py, &support_state_selected_neighbor_query)?,
    )?;
    row.set_item("support_boundary", support_boundary)?;

    let shadow_debug = PyDict::new(py);
    shadow_debug.set_item(
        "support_state_selected_neighbors",
        support_state_selected_neighbor_query
            .selected_neighbors
            .clone(),
    )?;
    shadow_debug.set_item(
        "support_state_unresolved_side_ids",
        support_state_selected_neighbor_query
            .unresolved_side_ids
            .clone(),
    )?;
    shadow_debug.set_item(
        "support_state_forced_side_neighbors",
        support_state_selected_neighbor_query
            .forced_side_neighbors
            .clone(),
    )?;
    shadow_debug.set_item(
        "support_state_selected_neighbors_match_runtime",
        support_state_selected_neighbor_query.selected_neighbors
            == legacy_field_resolved_selected_neighbors,
    )?;
    shadow_debug.set_item(
        "support_state_selected_neighbors_match_joined_support_boundary",
        support_state_selected_neighbor_query.selected_neighbors
            == joined_support_boundary_selected_neighbors,
    )?;
    shadow_debug.set_item(
        "resolved_selected_neighbors_from_assignment_state",
        assignment_state_resolved_selected_neighbors.clone(),
    )?;
    shadow_debug.set_item(
        "assignment_state_resolution_matches_runtime",
        assignment_state_resolved_selected_neighbors == legacy_field_resolved_selected_neighbors,
    )?;
    shadow_debug.set_item(
        "assignment_state_resolution_matches_support_boundary",
        assignment_state_resolved_selected_neighbors == resolved_selected_neighbors,
    )?;
    shadow_debug.set_item(
        "legacy_field_resolved_selected_neighbors",
        legacy_field_resolved_selected_neighbors.clone(),
    )?;
    shadow_debug.set_item(
        "legacy_field_resolution_matches_support_boundary",
        legacy_field_resolved_selected_neighbors == resolved_selected_neighbors,
    )?;
    shadow_debug.set_item(
        "joined_support_boundary_selected_neighbors",
        joined_support_boundary_selected_neighbors.clone(),
    )?;
    shadow_debug.set_item(
        "joined_support_boundary_matches_runtime",
        joined_support_boundary_selected_neighbors == legacy_field_resolved_selected_neighbors,
    )?;
    row.set_item("shadow_debug", shadow_debug)?;
    row.set_item(
        "resolved_constraint_state_from_supported_token_observations",
        observation_constraint_state_to_py(
            py,
            runtime,
            &resolved_facts_by_component,
            &supported_token_observation_facts,
        )?,
    )?;
    row.set_item(
        "resolved_constraint_state_from_known_token_flips_and_inferred_token_observations",
        mixed_constraint_state_to_py(
            py,
            runtime,
            &resolved_facts_by_component,
            &known_token_flip_facts,
            &inferred_token_observation_facts,
        )?,
    )?;
    row.set_item(
        "runtime_token_constraint_facts",
        runtime_token_constraint_facts_to_py(
            py,
            &known_token_flip_facts,
            &inferred_token_observation_facts,
        )?,
    )?;
    row.set_item(
        "component_phase_boundary_facts",
        component_phase_boundary_facts_to_py(py, state)?,
    )?;
    row.set_item(
        "marker_placement_state",
        rdkit_marker_placement_state_to_py(
            py,
            runtime,
            &resolved_facts_by_component,
            &known_token_flip_facts,
            &inferred_token_observation_facts,
            &marker_events_by_component,
        )?,
    )?;
    row.set_item(
        "marker_obligation_state",
        rdkit_marker_placement_state_to_py(
            py,
            runtime,
            &resolved_facts_by_component,
            &known_token_flip_facts,
            &inferred_token_observation_facts,
            &marker_obligation_events_by_component,
        )?,
    )?;
    row.set_item(
        "shared_carrier_resolution",
        shared_carrier_resolution_to_py(
            py,
            runtime,
            raw_selected_neighbors,
            &resolved_selected_neighbors,
            &raw_semantic_assignment_state,
        )?,
    )?;
    row.set_item(
        "component_token_phase",
        component_token_phase_diagnostics_to_py(
            py,
            runtime,
            state,
            &resolved_selected_neighbors,
            &resolved_semantic_assignment_state,
            &token_constraints,
        )?,
    )?;
    Ok(row.unbind())
}

fn collect_stereo_output_fact_rows(
    py: Python<'_>,
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    mut state: RootedConnectedStereoWalkerStateData,
    rows: &Bound<'_, PyList>,
) -> PyResult<()> {
    drain_exact_linear_stereo_actions(&mut state);
    if state.action_stack.is_empty() {
        if is_complete_terminal_stereo_state(graph, &state)
            && terminal_stereo_state_has_support_boundary_marker_placement(runtime, graph, &state)?
        {
            rows.append(stereo_output_fact_row_to_py(py, graph, runtime, &state)?)?;
        }
        return Ok(());
    }

    let successors = flatten_exact_stereo_successor_groups(successors_by_token_stereo_raw(
        runtime, graph, &state,
    )?);
    for successor in successors {
        collect_stereo_output_fact_rows(py, runtime, graph, successor, rows)?;
    }
    Ok(())
}

struct DeferredMarkerObligationWitnessScan {
    terminal_state_count: usize,
    visited_state_count: usize,
    truncated: bool,
}

fn collect_deferred_marker_obligation_witness_rows(
    py: Python<'_>,
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    mut state: RootedConnectedStereoWalkerStateData,
    rows: &Bound<'_, PyList>,
    limit: usize,
    max_terminal_states: usize,
    max_states: usize,
    scan: &mut DeferredMarkerObligationWitnessScan,
) -> PyResult<()> {
    if rows.len() >= limit || scan.truncated {
        return Ok(());
    }
    if scan.visited_state_count >= max_states {
        scan.truncated = true;
        return Ok(());
    }
    scan.visited_state_count += 1;

    drain_exact_linear_stereo_actions(&mut state);
    if state.action_stack.is_empty() {
        if is_complete_terminal_stereo_state(graph, &state) {
            if let Some(summary) =
                terminal_stereo_state_support_boundary_summary(runtime, graph, &state)?
            {
                scan.terminal_state_count += 1;
                if summary.deferred_marker_obligation_domain_count > 0 {
                    let row = PyDict::new(py);
                    row.set_item("smiles", state.prefix.to_string())?;
                    row.set_item(
                        "deferred_marker_obligation_domain_count",
                        summary.deferred_marker_obligation_domain_count,
                    )?;
                    rows.append(row)?;
                }
                if scan.terminal_state_count >= max_terminal_states {
                    scan.truncated = true;
                }
            }
        }
        return Ok(());
    }

    let successors = flatten_exact_stereo_successor_groups(successors_by_token_stereo_raw(
        runtime, graph, &state,
    )?);
    for successor in successors {
        collect_deferred_marker_obligation_witness_rows(
            py,
            runtime,
            graph,
            successor,
            rows,
            limit,
            max_terminal_states,
            max_states,
            scan,
        )?;
        if rows.len() >= limit || scan.truncated {
            break;
        }
    }
    Ok(())
}

struct DeferredMarkerBasisDiagnosticScan {
    visited_state_count: usize,
    truncated: bool,
}

fn deferred_edge_to_py(
    py: Python<'_>,
    graph: &PreparedSmilesGraphData,
    deferred: &DeferredDirectionalToken,
) -> PyResult<Py<PyDict>> {
    let row = PyDict::new(py);
    row.set_item("begin_idx", deferred.begin_idx)?;
    row.set_item("end_idx", deferred.end_idx)?;
    if deferred.begin_idx >= 0 && deferred.end_idx >= 0 {
        let begin_idx = deferred.begin_idx as usize;
        let end_idx = deferred.end_idx as usize;
        row.set_item("canonical_edge", canonical_edge(begin_idx, end_idx))?;
        row.set_item("bond_token", graph.bond_token(begin_idx, end_idx))?;
    } else {
        row.set_item("canonical_edge", Option::<(usize, usize)>::None)?;
        row.set_item("bond_token", Option::<&str>::None)?;
    }
    Ok(row.unbind())
}

fn deferred_marker_basis_candidate_rows_to_py(
    py: Python<'_>,
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    deferred: &DeferredDirectionalToken,
) -> PyResult<Vec<Py<PyDict>>> {
    if deferred.component_tokens.is_empty() {
        return Ok(Vec::new());
    }

    let boundary_facts = support_boundary_facts_from_walker_state(
        runtime,
        graph,
        state,
        state.stereo_selected_neighbors.as_ref(),
        true,
    )?;
    let constraint_state =
        boundary_facts.constraint_state(runtime, StereoConstraintLayer::Semantic)?;
    let raw_tokens = raw_tokens_for_deferred_edge(runtime, graph, state, deferred)?;
    let current_support =
        deferred_token_support_from_constraint_state(runtime, graph, state, deferred)?;
    let candidate_tokens = if deferred.component_tokens.len() == 1 {
        if raw_tokens.is_empty() {
            Vec::new()
        } else {
            candidate_tokens_from_raw_deferred_tokens(&raw_tokens)?
        }
    } else {
        vec!["/".to_owned(), "\\".to_owned()]
    };

    let mut rows = Vec::<Py<PyDict>>::new();
    for candidate_token in candidate_tokens {
        let component_diagnostics = deferred
            .component_tokens
            .iter()
            .map(|component_token| {
                let model_component_idx = runtime
                    .constraint_model
                    .component_for_runtime_component(component_token.component_idx)
                    .ok_or_else(|| {
                        PyValueError::new_err("Deferred token references unknown runtime component")
                    })?;
                let accepted_token_flips = if deferred.component_tokens.len() == 1 {
                    accepted_single_component_deferred_token_flips_from_raw_tokens(
                        runtime,
                        state,
                        deferred,
                        component_token,
                        &raw_tokens,
                        &candidate_token,
                        &constraint_state,
                        &boundary_facts,
                    )?
                } else {
                    accepted_deferred_token_flips(
                        runtime,
                        state,
                        deferred,
                        component_token,
                        &candidate_token,
                        &constraint_state,
                        &boundary_facts,
                    )?
                };
                let raw_selected_carrier_token_flips =
                    accepted_token_flips_from_raw_selected_carrier_basis(
                        runtime,
                        state,
                        deferred,
                        component_token,
                        &raw_tokens,
                        &candidate_token,
                        &constraint_state,
                        &boundary_facts,
                    )?;
                let basis_diagnostics = visible_marker_basis_diagnostics_for_component_token(
                    runtime,
                    state,
                    deferred,
                    component_token,
                    &candidate_token,
                    &constraint_state,
                    &boundary_facts,
                )?;
                let basis_classes = basis_diagnostics
                    .iter()
                    .map(|diagnostic| diagnostic.basis_class)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                let visible_edge_token_flips = basis_diagnostics
                    .iter()
                    .filter(|diagnostic| {
                        diagnostic.basis_class == "non_selected_visible_edge"
                            || diagnostic.basis_class == "shared_visible_edge"
                    })
                    .map(|diagnostic| diagnostic.implied_token_flip)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                let remaining_basis_summary = remaining_component_visible_marker_basis_summary(
                    runtime,
                    component_token.component_idx,
                    &constraint_state,
                )?;
                let has_legacy_remaining_visible_marker_basis = remaining_basis_summary
                    .has_shared_visible_edge_basis
                    || (remaining_basis_summary.has_non_selected_visible_edge_basis
                        && legacy_topology_guard_component_has_cross_component_carrier_edge(
                            runtime,
                            component_token.component_idx,
                        ));
                let all_marker_row_token_flips = accepted_deferred_token_flips(
                    runtime,
                    state,
                    deferred,
                    component_token,
                    &candidate_token,
                    &constraint_state,
                    &boundary_facts,
                )?;
                let legacy_remaining_visible_token_flips =
                    if has_legacy_remaining_visible_marker_basis {
                        all_marker_row_token_flips.clone()
                    } else {
                        Vec::new()
                    };
                let remaining_shared_visible_token_flips =
                    if remaining_basis_summary.has_shared_visible_edge_basis {
                        all_marker_row_token_flips.clone()
                    } else {
                        Vec::new()
                    };
                let frontier_shared_nonselected_visible_token_flips =
                    if remaining_component_has_shared_nonselected_visible_marker_basis(
                        &remaining_basis_summary,
                    ) {
                        all_marker_row_token_flips.clone()
                    } else {
                        Vec::new()
                    };
                let legacy_guard_applies = deferred_token_legacy_topology_guard_applies(
                    runtime,
                    graph,
                    component_token.component_idx,
                );
                let legacy_topology_gated_token_flips = if legacy_guard_applies
                    && (!visible_edge_token_flips.is_empty()
                        || has_legacy_remaining_visible_marker_basis)
                {
                    all_marker_row_token_flips.clone()
                } else {
                    Vec::new()
                };

                let component_row = PyDict::new(py);
                component_row.set_item("component_idx", component_token.component_idx)?;
                component_row.set_item("model_component_idx", model_component_idx)?;
                component_row
                    .set_item("reference_tokens", component_token.reference_tokens.clone())?;
                let shadow_debug = PyDict::new(py);
                shadow_debug.set_item("legacy_topology_guard_applies", legacy_guard_applies)?;
                shadow_debug.set_item(
                    "remaining_has_non_selected_visible_edge_basis",
                    remaining_basis_summary.has_non_selected_visible_edge_basis,
                )?;
                shadow_debug.set_item(
                    "remaining_has_shared_visible_edge_basis",
                    remaining_basis_summary.has_shared_visible_edge_basis,
                )?;
                shadow_debug.set_item(
                    "legacy_remaining_visible_basis_applies",
                    has_legacy_remaining_visible_marker_basis,
                )?;
                shadow_debug.set_item(
                    "frontier_shared_nonselected_visible_basis_applies",
                    remaining_component_has_shared_nonselected_visible_marker_basis(
                        &remaining_basis_summary,
                    ),
                )?;
                component_row.set_item("shadow_debug", shadow_debug)?;
                component_row.set_item(
                    "accepted_token_flips",
                    token_flips_to_py_names(&accepted_token_flips),
                )?;
                component_row.set_item(
                    "raw_selected_carrier_token_flips",
                    token_flips_to_py_names(&raw_selected_carrier_token_flips),
                )?;
                component_row.set_item(
                    "visible_edge_token_flips",
                    token_flips_to_py_names(&visible_edge_token_flips),
                )?;
                let policy_variant_token_flips = PyDict::new(py);
                policy_variant_token_flips.set_item(
                    "raw_selected_carrier",
                    token_flips_to_py_names(&raw_selected_carrier_token_flips),
                )?;
                policy_variant_token_flips.set_item(
                    "current_visible_basis",
                    token_flips_to_py_names(&visible_edge_token_flips),
                )?;
                policy_variant_token_flips.set_item(
                    "legacy_remaining_visible_basis",
                    token_flips_to_py_names(&legacy_remaining_visible_token_flips),
                )?;
                policy_variant_token_flips.set_item(
                    "remaining_shared_visible_basis",
                    token_flips_to_py_names(&remaining_shared_visible_token_flips),
                )?;
                policy_variant_token_flips.set_item(
                    "legacy_topology_gated_visible_basis",
                    token_flips_to_py_names(&legacy_topology_gated_token_flips),
                )?;
                policy_variant_token_flips.set_item(
                    "frontier_shared_nonselected_visible_basis",
                    token_flips_to_py_names(&frontier_shared_nonselected_visible_token_flips),
                )?;
                component_row.set_item("policy_variant_token_flips", policy_variant_token_flips)?;
                let policy_variant_accepts = PyDict::new(py);
                policy_variant_accepts.set_item(
                    "raw_selected_carrier",
                    !raw_selected_carrier_token_flips.is_empty(),
                )?;
                policy_variant_accepts.set_item(
                    "current_visible_basis",
                    !visible_edge_token_flips.is_empty(),
                )?;
                policy_variant_accepts.set_item(
                    "legacy_remaining_visible_basis",
                    !legacy_remaining_visible_token_flips.is_empty(),
                )?;
                policy_variant_accepts.set_item(
                    "remaining_shared_visible_basis",
                    !remaining_shared_visible_token_flips.is_empty(),
                )?;
                policy_variant_accepts.set_item(
                    "legacy_topology_gated_visible_basis",
                    !legacy_topology_gated_token_flips.is_empty(),
                )?;
                policy_variant_accepts.set_item(
                    "frontier_shared_nonselected_visible_basis",
                    !frontier_shared_nonselected_visible_token_flips.is_empty(),
                )?;
                component_row.set_item("policy_variant_accepts", policy_variant_accepts)?;
                component_row.set_item(
                    "raw_selected_carrier_explains_chosen_token",
                    !raw_selected_carrier_token_flips.is_empty(),
                )?;
                component_row.set_item(
                    "visible_edge_basis_explains_chosen_token",
                    !visible_edge_token_flips.is_empty(),
                )?;
                component_row.set_item("basis_classes_considered", basis_classes)?;
                component_row.set_item(
                    "basis_candidates",
                    basis_diagnostics
                        .iter()
                        .map(|diagnostic| {
                            visible_marker_basis_diagnostic_to_py(
                                py,
                                component_token.component_idx,
                                model_component_idx,
                                diagnostic,
                            )
                        })
                        .collect::<PyResult<Vec<_>>>()?,
                )?;
                Ok(component_row.unbind())
            })
            .collect::<PyResult<Vec<_>>>()?;

        let row = PyDict::new(py);
        row.set_item("root_idx", runtime.root_idx)?;
        row.set_item("prefix", state.prefix.as_ref())?;
        row.set_item("candidate_token", &candidate_token)?;
        row.set_item(
            "current_support_accepts_candidate",
            current_support.contains(&candidate_token),
        )?;
        row.set_item("deferred_edge", deferred_edge_to_py(py, graph, deferred)?)?;
        row.set_item("raw_tokens", raw_tokens.clone())?;
        row.set_item("component_count", deferred.component_tokens.len())?;
        row.set_item("components", component_diagnostics)?;
        rows.push(row.unbind());
    }
    Ok(rows)
}

fn collect_deferred_marker_basis_diagnostic_rows(
    py: Python<'_>,
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    mut state: RootedConnectedStereoWalkerStateData,
    rows: &Bound<'_, PyList>,
    limit: usize,
    max_states: usize,
    scan: &mut DeferredMarkerBasisDiagnosticScan,
) -> PyResult<()> {
    if rows.len() >= limit || scan.truncated {
        return Ok(());
    }
    if scan.visited_state_count >= max_states {
        scan.truncated = true;
        return Ok(());
    }
    scan.visited_state_count += 1;

    drain_exact_linear_stereo_actions(&mut state);
    if let Some(WalkerAction::EmitDeferred(deferred)) = state.action_stack.last().cloned() {
        for row in
            deferred_marker_basis_candidate_rows_to_py(py, runtime, graph, &state, &deferred)?
        {
            rows.append(row)?;
            if rows.len() >= limit {
                scan.truncated = true;
                return Ok(());
            }
        }
    }
    if state.action_stack.is_empty() {
        return Ok(());
    }

    let successors = flatten_exact_stereo_successor_groups(successors_by_token_stereo_raw(
        runtime, graph, &state,
    )?);
    for successor in successors {
        collect_deferred_marker_basis_diagnostic_rows(
            py, runtime, graph, successor, rows, limit, max_states, scan,
        )?;
        if rows.len() >= limit || scan.truncated {
            break;
        }
    }
    Ok(())
}

#[pyfunction(name = "_stereo_constraint_output_facts", signature = (graph, root_idx=-1))]
pub fn internal_stereo_constraint_output_facts(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
    root_idx: isize,
) -> PyResult<Py<PyList>> {
    let graph = PreparedSmilesGraphData::from_any(graph)?;
    let root_indices = if root_idx == -1 {
        (0..graph.atom_count()).collect::<Vec<_>>()
    } else {
        vec![validate_root_idx(&graph, root_idx)?]
    };

    let rows = PyList::empty(py);
    for root_idx in root_indices {
        let runtime = build_walker_runtime(&graph, root_idx)?;
        if runtime.side_infos.is_empty() {
            return Err(PyValueError::new_err(
                "stereo constraint output facts require bond-stereo side metadata",
            ));
        }
        collect_stereo_output_fact_rows(
            py,
            &runtime,
            &graph,
            initial_stereo_state_for_root(&runtime, &graph, runtime.root_idx),
            &rows,
        )?;
    }
    Ok(rows.unbind())
}

#[pyfunction(
    name = "_stereo_deferred_marker_obligation_witnesses",
    signature = (graph, root_idx=-1, limit=16, max_terminal_states=100000, max_states=1000000)
)]
pub fn internal_stereo_deferred_marker_obligation_witnesses(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
    root_idx: isize,
    limit: usize,
    max_terminal_states: usize,
    max_states: usize,
) -> PyResult<Py<PyDict>> {
    if limit == 0 {
        return Err(PyValueError::new_err("limit must be positive"));
    }
    if max_terminal_states == 0 {
        return Err(PyValueError::new_err(
            "max_terminal_states must be positive",
        ));
    }
    if max_states == 0 {
        return Err(PyValueError::new_err("max_states must be positive"));
    }

    let graph = PreparedSmilesGraphData::from_any(graph)?;
    let root_indices = if root_idx == -1 {
        (0..graph.atom_count()).collect::<Vec<_>>()
    } else {
        vec![validate_root_idx(&graph, root_idx)?]
    };

    let rows = PyList::empty(py);
    let mut total_terminal_state_count = 0usize;
    let mut total_visited_state_count = 0usize;
    let mut truncated = false;
    for root_idx in root_indices {
        let runtime = build_walker_runtime(&graph, root_idx)?;
        if runtime.side_infos.is_empty() {
            return Err(PyValueError::new_err(
                "stereo deferred marker-obligation witnesses require bond-stereo side metadata",
            ));
        }
        let mut scan = DeferredMarkerObligationWitnessScan {
            terminal_state_count: 0,
            visited_state_count: 0,
            truncated: false,
        };
        collect_deferred_marker_obligation_witness_rows(
            py,
            &runtime,
            &graph,
            initial_stereo_state_for_root(&runtime, &graph, runtime.root_idx),
            &rows,
            limit,
            max_terminal_states.saturating_sub(total_terminal_state_count),
            max_states.saturating_sub(total_visited_state_count),
            &mut scan,
        )?;
        total_terminal_state_count += scan.terminal_state_count;
        total_visited_state_count += scan.visited_state_count;
        truncated |= scan.truncated;
        if rows.len() >= limit
            || total_terminal_state_count >= max_terminal_states
            || total_visited_state_count >= max_states
        {
            truncated |= total_terminal_state_count >= max_terminal_states;
            truncated |= total_visited_state_count >= max_states;
            break;
        }
    }

    let result = PyDict::new(py);
    result.set_item("witnesses", rows)?;
    result.set_item("terminal_state_count", total_terminal_state_count)?;
    result.set_item("visited_state_count", total_visited_state_count)?;
    result.set_item("truncated", truncated)?;
    Ok(result.unbind())
}

#[pyfunction(
    name = "_stereo_deferred_marker_basis_diagnostics",
    signature = (graph, root_idx=-1, limit=64, max_states=100000)
)]
pub fn internal_stereo_deferred_marker_basis_diagnostics(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
    root_idx: isize,
    limit: usize,
    max_states: usize,
) -> PyResult<Py<PyDict>> {
    if limit == 0 {
        return Err(PyValueError::new_err("limit must be positive"));
    }
    if max_states == 0 {
        return Err(PyValueError::new_err("max_states must be positive"));
    }

    let graph = PreparedSmilesGraphData::from_any(graph)?;
    let root_indices = if root_idx == -1 {
        (0..graph.atom_count()).collect::<Vec<_>>()
    } else {
        vec![validate_root_idx(&graph, root_idx)?]
    };

    let rows = PyList::empty(py);
    let mut total_visited_state_count = 0usize;
    let mut truncated = false;
    for root_idx in root_indices {
        let runtime = build_walker_runtime(&graph, root_idx)?;
        if runtime.side_infos.is_empty() {
            return Err(PyValueError::new_err(
                "stereo deferred marker-basis diagnostics require bond-stereo side metadata",
            ));
        }
        let mut scan = DeferredMarkerBasisDiagnosticScan {
            visited_state_count: 0,
            truncated: false,
        };
        collect_deferred_marker_basis_diagnostic_rows(
            py,
            &runtime,
            &graph,
            initial_stereo_state_for_root(&runtime, &graph, runtime.root_idx),
            &rows,
            limit,
            max_states.saturating_sub(total_visited_state_count),
            &mut scan,
        )?;
        total_visited_state_count += scan.visited_state_count;
        truncated |= scan.truncated;
        if rows.len() >= limit || total_visited_state_count >= max_states {
            truncated = true;
            break;
        }
    }

    let result = PyDict::new(py);
    result.set_item("rows", rows)?;
    result.set_item("visited_state_count", total_visited_state_count)?;
    result.set_item("truncated", truncated)?;
    Ok(result.unbind())
}

fn validate_stereo_state_shape(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<()> {
    if state.visited.len() != graph.atom_count()
        || state
            .pending
            .iter()
            .any(|(atom_idx, _rings)| *atom_idx >= graph.atom_count())
        || state.stereo_component_phases.len() != runtime.isolated_components.len()
        || state.stereo_component_begin_atoms.len() != runtime.isolated_components.len()
        || state.committed_component_token_flips.len() != runtime.isolated_components.len()
        || state.stereo_selected_neighbors.len() != runtime.side_infos.len()
        || state.stereo_selected_orientations.len() != runtime.side_infos.len()
        || state.stereo_first_emitted_candidates.len() != runtime.side_infos.len()
        || state
            .deferred_carrier_choice_constraints
            .iter()
            .any(|constraint| {
                constraint.side_idx >= runtime.side_infos.len()
                    || !constraint
                        .available_neighbors
                        .contains(&constraint.deferred_neighbor_idx)
            })
    {
        return Err(PyValueError::new_err(
            "walker state is not compatible with this PreparedSmilesGraph",
        ));
    }
    #[cfg(debug_assertions)]
    validate_stereo_state_against_constraint_model(runtime, state)?;
    Ok(())
}

fn selected_neighbor_facts_by_component(
    runtime: &StereoWalkerRuntimeData,
    selected_neighbors: &[isize],
) -> Vec<Vec<StereoConstraintFact>> {
    let mut facts_by_component = vec![Vec::new(); runtime.constraint_model.component_count()];
    for (component_idx, side_idx, neighbor_idx) in
        selected_neighbor_fact_rows(runtime, selected_neighbors)
    {
        facts_by_component[component_idx].push(StereoConstraintFact::CarrierSelected {
            side_idx,
            neighbor_idx,
        });
    }
    facts_by_component
}

fn carrier_facts_by_component_from_state(
    runtime: &StereoWalkerRuntimeData,
    selected_neighbors: &[isize],
    deferred_carrier_choice_constraints: &[DeferredCarrierChoiceConstraint],
) -> PyResult<Vec<Vec<StereoConstraintFact>>> {
    let mut facts_by_component = selected_neighbor_facts_by_component(runtime, selected_neighbors);
    for constraint in deferred_carrier_choice_constraints {
        let Some(component_idx) = runtime
            .constraint_model
            .component_for_side(constraint.side_idx)
        else {
            return Err(PyValueError::new_err(
                "deferred carrier-choice constraint references unknown side",
            ));
        };
        facts_by_component[component_idx].push(StereoConstraintFact::CarrierSelectionBlocked {
            side_idx: constraint.side_idx,
            neighbor_idx: constraint.deferred_neighbor_idx,
        });
    }
    Ok(facts_by_component)
}

fn rdkit_traversal_writer_facts_by_component(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    selected_neighbors: &[isize],
) -> Vec<Vec<StereoConstraintFact>> {
    let mut facts_by_component = selected_neighbor_facts_by_component(runtime, selected_neighbors);
    for trace in state.directional_marker_traces.iter() {
        if trace.side_idx < 0 || trace.edge_begin_idx < 0 || trace.edge_end_idx < 0 {
            continue;
        }
        let side_idx = trace.side_idx as usize;
        let Some(component_idx) = runtime.constraint_model.component_for_side(side_idx) else {
            continue;
        };
        facts_by_component[component_idx].push(StereoConstraintFact::CarrierEdgeEmitted {
            side_idx,
            begin_idx: trace.edge_begin_idx as usize,
            end_idx: trace.edge_end_idx as usize,
            role: trace.role,
        });
        facts_by_component[component_idx].push(StereoConstraintFact::DirectionalMarkerPlaced {
            side_idx,
            slot: trace.slot,
            marker: trace.marker,
            role: trace.role,
        });
    }
    facts_by_component
}

fn selected_neighbors_have_constraint_completion(
    runtime: &StereoWalkerRuntimeData,
    selected_neighbors: &[isize],
    layer: StereoConstraintLayer,
) -> bool {
    selected_neighbor_facts_by_component(runtime, selected_neighbors)
        .iter()
        .enumerate()
        .all(|(component_idx, facts)| {
            runtime
                .constraint_model
                .has_completion(component_idx, layer, facts)
        })
}

fn rdkit_traversal_writer_has_completion(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    selected_neighbors: &[isize],
    layer: StereoConstraintLayer,
) -> bool {
    rdkit_traversal_writer_facts_by_component(runtime, state, selected_neighbors)
        .iter()
        .enumerate()
        .all(|(component_idx, facts)| {
            runtime
                .constraint_model
                .has_completion(component_idx, layer, facts)
        })
}

#[cfg(debug_assertions)]
fn validate_selected_neighbors_against_constraint_model(
    runtime: &StereoWalkerRuntimeData,
    selected_neighbors: &[isize],
    view_name: &str,
) -> PyResult<()> {
    // Current walker carrier facts are only guaranteed to satisfy the semantic
    // layer. RDKit writer layers can be stricter than the present bookkeeping
    // until pruning is moved to a derived writer-policy application point.
    for layer in [StereoConstraintLayer::Semantic] {
        if !selected_neighbors_have_constraint_completion(runtime, selected_neighbors, layer) {
            let facts_by_component =
                selected_neighbor_facts_by_component(runtime, selected_neighbors);
            for (component_idx, facts) in facts_by_component.iter().enumerate() {
                if runtime
                    .constraint_model
                    .has_completion(component_idx, layer, facts)
                {
                    continue;
                }
                return Err(PyValueError::new_err(format!(
                    "walker stereo state violates {view_name} {} constraint model for component {component_idx}",
                    stereo_constraint_layer_name(layer),
                )));
            }
        }
    }
    Ok(())
}

#[cfg(debug_assertions)]
fn validate_stereo_state_against_constraint_model(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<()> {
    validate_selected_neighbors_against_constraint_model(
        runtime,
        &state.stereo_selected_neighbors,
        "raw carrier-selection",
    )?;
    validate_selected_neighbors_against_constraint_model(
        runtime,
        &resolved_selected_neighbors(runtime, state),
        "resolved carrier-selection",
    )
}

#[cfg(debug_assertions)]
fn validate_stereo_successors_against_constraint_model(
    runtime: &StereoWalkerRuntimeData,
    successors: &BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>,
) -> PyResult<()> {
    for states in successors.values() {
        for state in states {
            validate_stereo_state_against_constraint_model(runtime, state)?;
        }
    }
    Ok(())
}

fn initial_stereo_state_for_root(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    root_idx: usize,
) -> RootedConnectedStereoWalkerStateData {
    let mut action_stack = Vec::new();
    if graph.atom_count() > 0 {
        action_stack.push(WalkerAction::EnterAtom {
            atom_idx: root_idx,
            parent_idx: None,
        });
    }
    RootedConnectedStereoWalkerStateData {
        prefix: Arc::<str>::from(""),
        visited: Arc::<[bool]>::from(vec![false; graph.atom_count()]),
        visited_count: 0,
        pending: Arc::new(Vec::new()),
        free_labels: Arc::new(Vec::new()),
        next_label: 1,
        stereo_component_phases: Arc::new(vec![
            UNKNOWN_COMPONENT_PHASE;
            runtime.isolated_components.len()
        ]),
        stereo_selected_neighbors: Arc::new(vec![-1; runtime.side_infos.len()]),
        stereo_selected_orientations: Arc::new(vec![
            UNKNOWN_EDGE_ORIENTATION;
            runtime.side_infos.len()
        ]),
        stereo_first_emitted_candidates: Arc::new(vec![-1; runtime.side_infos.len()]),
        stereo_component_begin_atoms: Arc::new(vec![-1; runtime.isolated_components.len()]),
        committed_component_token_flips: Arc::new(vec![None; runtime.isolated_components.len()]),
        deferred_carrier_choice_constraints: Arc::new(Vec::new()),
        stereo_token_basis_facts: Arc::new(vec![None; runtime.isolated_components.len()]),
        directional_marker_traces: Arc::new(Vec::new()),
        marker_event_traces: Arc::new(Vec::new()),
        action_stack,
    }
}

fn is_terminal_stereo_state(state: &RootedConnectedStereoWalkerStateData) -> bool {
    state.action_stack.is_empty()
}

fn is_complete_terminal_stereo_state(
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> bool {
    is_terminal_stereo_state(state)
        && state.visited_count == graph.atom_count()
        && state.pending.is_empty()
}

fn part_to_action(part: Part) -> Option<WalkerAction> {
    match part {
        Part::Literal(token) => {
            if token.is_empty() {
                None
            } else {
                Some(WalkerAction::EmitLiteral(token))
            }
        }
        Part::Deferred(token) => Some(WalkerAction::EmitDeferred(token)),
    }
}

// RDKit-writer observation, not generic stereo semantics. This builder keeps
// the root-side and adjacent-side reasons named before they become compact
// support-state facts.
fn rdkit_token_flip_adjustment_observation_from_state(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    resolved_selected_neighbors: &[isize],
    component_idx: usize,
) -> RdkitTokenFlipAdjustmentObservation {
    let mut observation = RdkitTokenFlipAdjustmentObservation::default();
    let begin_atom_idx = state.stereo_component_begin_atoms[component_idx];
    if begin_atom_idx < 0 {
        return observation;
    }
    let begin_atom_idx = begin_atom_idx as usize;
    observation.begin_atom_is_root = begin_atom_idx == runtime.root_idx;

    let Some(begin_side_idx) = runtime.side_ids_by_component[component_idx]
        .iter()
        .copied()
        .find(|&side_idx| runtime.side_infos[side_idx].endpoint_atom_idx == begin_atom_idx)
    else {
        return observation;
    };

    let begin_side = &runtime.side_infos[begin_side_idx];
    let begin_side_orientation = state.stereo_selected_orientations[begin_side_idx];
    observation.begin_side_orientation = Some(edge_orientation_name(begin_side_orientation));
    observation.root_begin_side_adjustment =
        observation.begin_atom_is_root && begin_side_orientation == AFTER_ATOM_EDGE_ORIENTATION;

    if runtime.isolated_components[component_idx] || begin_side.candidate_neighbors.len() != 1 {
        return observation;
    }

    let selected_neighbor_idx = resolved_selected_neighbors[begin_side_idx];
    if selected_neighbor_idx < 0 {
        return observation;
    }
    let selected_neighbor_idx = selected_neighbor_idx as usize;
    observation.selected_neighbor_is_root = Some(selected_neighbor_idx == runtime.root_idx);

    let adjacent_two_side_idx = runtime.side_ids_by_component[component_idx]
        .iter()
        .copied()
        .find(|&side_idx| {
            let side_info = &runtime.side_infos[side_idx];
            side_info.candidate_neighbors.len() == 2
                && side_info.endpoint_atom_idx == selected_neighbor_idx
                && side_info.candidate_neighbors.contains(&begin_atom_idx)
        });

    let Some(adjacent_two_side_idx) = adjacent_two_side_idx else {
        return observation;
    };
    observation.adjacent_two_candidate_side_idx = Some(adjacent_two_side_idx);

    let first_neighbor_idx = state.stereo_first_emitted_candidates[adjacent_two_side_idx];
    if first_neighbor_idx >= 0 {
        let first_neighbor_idx = first_neighbor_idx as usize;
        observation.adjacent_first_emitted_candidate_idx = Some(first_neighbor_idx);
        observation.adjacent_first_emitted_is_not_begin =
            Some(first_neighbor_idx != begin_atom_idx);
    }
    observation.adjacent_two_candidate_adjustment = selected_neighbor_idx == runtime.root_idx
        && observation
            .adjacent_first_emitted_is_not_begin
            .unwrap_or(false);
    observation
}

fn edge_orientation_name(value: i8) -> &'static str {
    match value {
        BEFORE_ATOM_EDGE_ORIENTATION => "before_atom",
        AFTER_ATOM_EDGE_ORIENTATION => "after_atom",
        _ => "unknown",
    }
}

fn provisional_phase_from_selected_side(
    graph: &PreparedSmilesGraphData,
    side_info: &StereoSideInfo,
) -> PyResult<i8> {
    let Some(bond_idx) = graph.bond_index(
        side_info.endpoint_atom_idx,
        side_info.other_endpoint_atom_idx,
    ) else {
        return Err(PyKeyError::new_err(format!(
            "No bond between atoms {} and {}",
            side_info.endpoint_atom_idx, side_info.other_endpoint_atom_idx
        )));
    };
    let stored_begin_idx = graph.bond_begin_atom_indices[bond_idx];
    let stored_end_idx = graph.bond_end_atom_indices[bond_idx];
    let stereo_kind = graph.bond_stereo_kinds[bond_idx].as_str();
    if (
        side_info.endpoint_atom_idx,
        side_info.other_endpoint_atom_idx,
    ) == (stored_begin_idx, stored_end_idx)
    {
        return Ok(STORED_COMPONENT_PHASE);
    }
    if CIS_STEREO_BOND_KINDS.contains(&stereo_kind) {
        return Ok(STORED_COMPONENT_PHASE);
    }
    if TRANS_STEREO_BOND_KINDS.contains(&stereo_kind) {
        return Ok(FLIPPED_COMPONENT_PHASE);
    }
    Err(PyValueError::new_err(format!(
        "Unsupported stereo bond kind: {stereo_kind}"
    )))
}

fn component_token_flip_value_from_model(value: StereoTokenFlip) -> i8 {
    match value {
        StereoTokenFlip::Stored => STORED_COMPONENT_TOKEN_FLIP,
        StereoTokenFlip::Flipped => FLIPPED_COMPONENT_TOKEN_FLIP,
    }
}

fn inferred_component_token_flip(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    graph: &PreparedSmilesGraphData,
    component_idx: usize,
) -> PyResult<Option<i8>> {
    let resolved_selected_neighbors = resolved_selected_neighbors(runtime, state);
    Ok(component_token_inference_inputs(
        runtime,
        graph,
        state,
        &resolved_selected_neighbors,
        component_idx,
    )?
    .inferred)
}

struct ComponentTokenInferenceInputs {
    component_idx: usize,
    inference_branch: &'static str,
    has_required_inputs: bool,
    required_input_facts: Vec<String>,
    missing_input_facts: Vec<String>,
    side_ids: Vec<usize>,
    selected_side_ids: Vec<usize>,
    is_isolated: bool,
    all_single_candidate: bool,
    all_two_candidate: bool,
    observations: ComponentTokenObservationInputs,
    inferred: Option<i8>,
}

struct ComponentTokenObservationInputs {
    component_phase: ComponentPhaseObservation,
    component_begin_atom: ComponentBeginAtomObservation,
    inferred_selected_side_idx: Option<usize>,
    begin_side: BeginSideObservation,
    selected_begin: SelectedBeginObservation,
    first_emitted_candidate: FirstEmittedCandidateObservation,
    rdkit_token_flip_adjustment: RdkitTokenFlipAdjustmentObservation,
}

struct ComponentPhaseObservation {
    input: i8,
    effective: i8,
    source: &'static str,
}

struct ComponentBeginAtomObservation {
    input_atom_idx: isize,
    effective_atom_idx: isize,
    source: &'static str,
}

struct BeginSideObservation {
    side_idx: Option<usize>,
    candidate_count: usize,
}

struct SelectedBeginObservation {
    neighbor_idx: Option<usize>,
    token: Option<String>,
}

struct FirstEmittedCandidateObservation {
    neighbor_idx: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RdkitTokenFlipAdjustmentObservation {
    root_begin_side_adjustment: bool,
    begin_atom_is_root: bool,
    begin_side_orientation: Option<&'static str>,
    adjacent_two_candidate_adjustment: bool,
    adjacent_two_candidate_side_idx: Option<usize>,
    selected_neighbor_is_root: Option<bool>,
    adjacent_first_emitted_candidate_idx: Option<usize>,
    adjacent_first_emitted_is_not_begin: Option<bool>,
}

impl RdkitTokenFlipAdjustmentObservation {
    fn value(&self) -> bool {
        self.support_observations().value()
    }

    fn support_observations(&self) -> RdkitTokenFlipAdjustmentObservations {
        RdkitTokenFlipAdjustmentObservations {
            root_begin_side_orientation: self.root_begin_side_adjustment,
            adjacent_two_candidate_first_emitted: self.adjacent_two_candidate_adjustment,
        }
    }
}

impl ComponentTokenInferenceInputs {
    fn supported_token_observation(&self) -> PyResult<Option<StereoTokenObservationFact>> {
        if !self.has_required_inputs {
            return Ok(None);
        }
        let observations = &self.observations;
        let Some(component_phase) =
            model_component_phase_from_value(observations.component_phase.effective)
        else {
            return Ok(None);
        };
        match self.inference_branch {
            "isolated_all_single_candidate" => {
                Ok(Some(StereoTokenObservationFact::AllSingleCandidate {
                    runtime_component_idx: self.component_idx,
                    component_phase,
                    rdkit_token_flip_adjustment: observations
                        .rdkit_token_flip_adjustment
                        .support_observations(),
                }))
            }
            "isolated_selected_begin_side" | "coupled_one_candidate_begin_side" => {
                let Some(selected_begin_token) = observations.selected_begin.token.as_deref()
                else {
                    return Ok(None);
                };
                Ok(Some(StereoTokenObservationFact::SelectedBeginSide {
                    runtime_component_idx: self.component_idx,
                    component_phase,
                    selected_begin_token: StereoDirectionToken::from_str(selected_begin_token)?,
                    rdkit_token_flip_adjustment: observations
                        .rdkit_token_flip_adjustment
                        .support_observations(),
                }))
            }
            "isolated_two_candidate_begin_side" => {
                let Some(selected_begin_token) = observations.selected_begin.token.as_deref()
                else {
                    return Ok(None);
                };
                let Some(selected_begin_neighbor_idx) = observations.selected_begin.neighbor_idx
                else {
                    return Ok(None);
                };
                Ok(Some(
                    StereoTokenObservationFact::IsolatedTwoCandidateBeginSide {
                        runtime_component_idx: self.component_idx,
                        component_phase,
                        selected_begin_token: StereoDirectionToken::from_str(selected_begin_token)?,
                        selected_begin_neighbor_is_first_emitted: self
                            .observations
                            .first_emitted_candidate
                            .neighbor_idx
                            .map(|first_idx| first_idx == selected_begin_neighbor_idx),
                        rdkit_token_flip_adjustment: observations
                            .rdkit_token_flip_adjustment
                            .support_observations(),
                    },
                ))
            }
            "coupled_two_candidate_begin_side" => {
                let Some(selected_begin_token) = observations.selected_begin.token.as_deref()
                else {
                    return Ok(None);
                };
                let Some(selected_begin_neighbor_idx) = observations.selected_begin.neighbor_idx
                else {
                    return Ok(None);
                };
                Ok(Some(StereoTokenObservationFact::TwoCandidateBeginSide {
                    runtime_component_idx: self.component_idx,
                    component_phase,
                    selected_begin_token: StereoDirectionToken::from_str(selected_begin_token)?,
                    selected_begin_neighbor_is_first_emitted: self
                        .observations
                        .first_emitted_candidate
                        .neighbor_idx
                        .map(|first_idx| first_idx == selected_begin_neighbor_idx),
                    rdkit_token_flip_adjustment: observations
                        .rdkit_token_flip_adjustment
                        .support_observations(),
                }))
            }
            _ => Ok(None),
        }
    }
}

fn component_token_observation_inputs(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    resolved_selected_neighbors: &[isize],
    component_idx: usize,
) -> PyResult<ComponentTokenObservationInputs> {
    let side_ids = &runtime.side_ids_by_component[component_idx];
    let selected_side_ids = side_ids
        .iter()
        .copied()
        .filter(|&side_idx| resolved_selected_neighbors[side_idx] >= 0)
        .collect::<Vec<_>>();

    let input_phase = state.stereo_component_phases[component_idx];
    let input_begin_atom_idx = state.stereo_component_begin_atoms[component_idx];
    let mut effective_phase = input_phase;
    let mut effective_begin_atom_idx = input_begin_atom_idx;
    let mut phase_source = if input_phase == UNKNOWN_COMPONENT_PHASE {
        "missing"
    } else {
        "state"
    };
    let mut begin_atom_source = if input_begin_atom_idx < 0 {
        "missing"
    } else {
        "state"
    };
    let mut inferred_selected_side_idx = None;
    if effective_phase == UNKNOWN_COMPONENT_PHASE || effective_begin_atom_idx < 0 {
        if selected_side_ids.len() == 1 {
            let selected_side_idx = selected_side_ids[0];
            let selected_side = &runtime.side_infos[selected_side_idx];
            inferred_selected_side_idx = Some(selected_side_idx);
            if effective_phase == UNKNOWN_COMPONENT_PHASE {
                effective_phase = provisional_phase_from_selected_side(graph, selected_side)?;
                phase_source = "provisional_selected_side";
            }
            if effective_begin_atom_idx < 0 {
                effective_begin_atom_idx = selected_side.endpoint_atom_idx as isize;
                begin_atom_source = "selected_side";
            }
        }
    }

    let begin_side_idx = if effective_begin_atom_idx >= 0 {
        side_ids.iter().copied().find(|&side_idx| {
            runtime.side_infos[side_idx].endpoint_atom_idx == effective_begin_atom_idx as usize
        })
    } else {
        None
    };
    let begin_side = begin_side_idx.map(|side_idx| &runtime.side_infos[side_idx]);
    let selected_begin_neighbor_idx = begin_side_idx.and_then(|side_idx| {
        let neighbor_idx = resolved_selected_neighbors[side_idx];
        (neighbor_idx >= 0).then_some(neighbor_idx as usize)
    });
    let selected_begin_token = begin_side
        .zip(selected_begin_neighbor_idx)
        .map(|(side_info, neighbor_idx)| candidate_base_token(side_info, neighbor_idx))
        .transpose()?;
    let selected_begin_token = state
        .stereo_token_basis_facts
        .get(component_idx)
        .and_then(|fact| *fact)
        .map(|fact| stereo_direction_token_name(fact.selected_begin_token).to_owned())
        .or(selected_begin_token);
    let first_emitted_candidate_idx = begin_side_idx
        .map(|side_idx| state.stereo_first_emitted_candidates[side_idx])
        .filter(|&neighbor_idx| neighbor_idx >= 0)
        .map(|neighbor_idx| neighbor_idx as usize);
    let rdkit_adjustment_observation = rdkit_token_flip_adjustment_observation_from_state(
        runtime,
        state,
        resolved_selected_neighbors,
        component_idx,
    );
    Ok(ComponentTokenObservationInputs {
        component_phase: ComponentPhaseObservation {
            input: input_phase,
            effective: effective_phase,
            source: phase_source,
        },
        component_begin_atom: ComponentBeginAtomObservation {
            input_atom_idx: input_begin_atom_idx,
            effective_atom_idx: effective_begin_atom_idx,
            source: begin_atom_source,
        },
        inferred_selected_side_idx,
        begin_side: BeginSideObservation {
            side_idx: begin_side_idx,
            candidate_count: begin_side
                .map(|side_info| side_info.candidate_neighbors.len())
                .unwrap_or(0),
        },
        selected_begin: SelectedBeginObservation {
            neighbor_idx: selected_begin_neighbor_idx,
            token: selected_begin_token,
        },
        first_emitted_candidate: FirstEmittedCandidateObservation {
            neighbor_idx: first_emitted_candidate_idx,
        },
        rdkit_token_flip_adjustment: rdkit_adjustment_observation,
    })
}

fn component_token_inference_inputs(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    resolved_selected_neighbors: &[isize],
    component_idx: usize,
) -> PyResult<ComponentTokenInferenceInputs> {
    let side_ids = &runtime.side_ids_by_component[component_idx];
    let selected_side_ids = side_ids
        .iter()
        .copied()
        .filter(|&side_idx| resolved_selected_neighbors[side_idx] >= 0)
        .collect::<Vec<_>>();
    let all_single_candidate = side_ids
        .iter()
        .all(|&side_idx| runtime.side_infos[side_idx].candidate_neighbors.len() == 1);
    let all_two_candidate = side_ids
        .iter()
        .all(|&side_idx| runtime.side_infos[side_idx].candidate_neighbors.len() == 2);
    let observations = component_token_observation_inputs(
        runtime,
        graph,
        state,
        resolved_selected_neighbors,
        component_idx,
    )?;
    let effective_phase = observations.component_phase.effective;
    let effective_begin_atom_idx = observations.component_begin_atom.effective_atom_idx;
    let begin_side_idx = observations.begin_side.side_idx;
    let begin_side_candidate_count = observations.begin_side.candidate_count;
    let inference_branch = if side_ids.is_empty() {
        "no_sides"
    } else if !runtime.isolated_components[component_idx] && selected_side_ids.len() < 2 {
        "insufficient_coupled_selection"
    } else if effective_phase == UNKNOWN_COMPONENT_PHASE {
        "missing_phase"
    } else if effective_begin_atom_idx < 0 || begin_side_idx.is_none() {
        "missing_begin_side"
    } else if runtime.isolated_components[component_idx] && all_single_candidate {
        "isolated_all_single_candidate"
    } else if runtime.isolated_components[component_idx]
        && all_two_candidate
        && begin_side_candidate_count == 2
    {
        "isolated_two_candidate_begin_side"
    } else if runtime.isolated_components[component_idx] {
        "isolated_selected_begin_side"
    } else if begin_side_candidate_count == 1 {
        "coupled_one_candidate_begin_side"
    } else if begin_side_candidate_count == 2 {
        "coupled_two_candidate_begin_side"
    } else {
        "unsupported_begin_side_domain"
    };
    let value_branch = matches!(
        inference_branch,
        "isolated_all_single_candidate"
            | "isolated_selected_begin_side"
            | "isolated_two_candidate_begin_side"
            | "coupled_one_candidate_begin_side"
            | "coupled_two_candidate_begin_side"
    );
    let mut required_input_facts = Vec::<String>::new();
    let mut missing_input_facts = Vec::<String>::new();
    if value_branch {
        required_input_facts.push("component_phase".to_owned());
        if effective_phase == UNKNOWN_COMPONENT_PHASE {
            missing_input_facts.push("component_phase".to_owned());
        }
        required_input_facts.push("component_begin_atom".to_owned());
        if effective_begin_atom_idx < 0 {
            missing_input_facts.push("component_begin_atom".to_owned());
        }
        required_input_facts.push("begin_side".to_owned());
        if begin_side_idx.is_none() {
            missing_input_facts.push("begin_side".to_owned());
        }
        if inference_branch != "isolated_all_single_candidate" {
            required_input_facts.push("selected_begin_neighbor".to_owned());
            if observations.selected_begin.neighbor_idx.is_none() {
                missing_input_facts.push("selected_begin_neighbor".to_owned());
            }
            required_input_facts.push("selected_begin_token".to_owned());
            if observations.selected_begin.token.is_none() {
                missing_input_facts.push("selected_begin_token".to_owned());
            }
        }
        if matches!(
            inference_branch,
            "isolated_two_candidate_begin_side" | "coupled_two_candidate_begin_side"
        ) {
            required_input_facts.push("first_emitted_candidate_or_adjustment_fallback".to_owned());
        }
        required_input_facts.push("rdkit_token_flip_adjustment".to_owned());
    }
    let has_required_inputs = value_branch && missing_input_facts.is_empty();

    let mut inputs = ComponentTokenInferenceInputs {
        component_idx,
        inference_branch,
        has_required_inputs,
        required_input_facts,
        missing_input_facts,
        side_ids: side_ids.clone(),
        selected_side_ids,
        is_isolated: runtime.isolated_components[component_idx],
        all_single_candidate,
        all_two_candidate,
        observations,
        inferred: None,
    };
    let observation_inferred = inputs
        .supported_token_observation()?
        .map(|fact| component_token_flip_value_from_model(fact.implied_token_flip()));
    inputs.inferred = observation_inferred;
    Ok(inputs)
}

fn component_token_inference_inputs_to_py(
    py: Python<'_>,
    inputs: &ComponentTokenInferenceInputs,
) -> PyResult<Py<PyDict>> {
    let observations = &inputs.observations;
    let row = PyDict::new(py);
    row.set_item("component_idx", inputs.component_idx)?;
    row.set_item("inference_branch", inputs.inference_branch)?;
    row.set_item("has_required_inputs", inputs.has_required_inputs)?;
    row.set_item("required_input_facts", inputs.required_input_facts.clone())?;
    row.set_item("missing_input_facts", inputs.missing_input_facts.clone())?;
    row.set_item("side_ids", inputs.side_ids.clone())?;
    row.set_item("selected_side_ids", inputs.selected_side_ids.clone())?;
    row.set_item("is_isolated", inputs.is_isolated)?;
    row.set_item("all_single_candidate", inputs.all_single_candidate)?;
    row.set_item("all_two_candidate", inputs.all_two_candidate)?;
    row.set_item(
        "input_observation_facts",
        component_token_observation_inputs_to_py(py, observations)?,
    )?;
    row.set_item(
        "input_phase",
        component_phase_name(observations.component_phase.input),
    )?;
    row.set_item("input_phase_value", observations.component_phase.input)?;
    row.set_item(
        "effective_phase",
        component_phase_name(observations.component_phase.effective),
    )?;
    row.set_item(
        "effective_phase_value",
        observations.component_phase.effective,
    )?;
    row.set_item("phase_source", observations.component_phase.source)?;
    row.set_item(
        "input_begin_atom_idx",
        observations.component_begin_atom.input_atom_idx,
    )?;
    row.set_item(
        "effective_begin_atom_idx",
        observations.component_begin_atom.effective_atom_idx,
    )?;
    row.set_item(
        "begin_atom_source",
        observations.component_begin_atom.source,
    )?;
    row.set_item(
        "inferred_selected_side_idx",
        observations.inferred_selected_side_idx,
    )?;
    row.set_item("begin_side_idx", observations.begin_side.side_idx)?;
    row.set_item(
        "begin_side_candidate_count",
        observations.begin_side.candidate_count,
    )?;
    row.set_item(
        "selected_begin_neighbor_idx",
        observations.selected_begin.neighbor_idx,
    )?;
    row.set_item(
        "selected_begin_token",
        observations.selected_begin.token.clone(),
    )?;
    row.set_item(
        "first_emitted_candidate_idx",
        observations.first_emitted_candidate.neighbor_idx,
    )?;
    row.set_item(
        "first_emitted_candidate_known",
        observations.first_emitted_candidate.neighbor_idx.is_some(),
    )?;
    row.set_item(
        "rdkit_token_flip_adjustment",
        observations.rdkit_token_flip_adjustment.value(),
    )?;
    row.set_item(
        "inferred_token_flip",
        inputs.inferred.map(component_token_flip_name),
    )?;
    row.set_item("inferred_token_flip_value", inputs.inferred)?;
    Ok(row.unbind())
}

fn component_token_observation_inputs_to_py(
    py: Python<'_>,
    observations: &ComponentTokenObservationInputs,
) -> PyResult<Vec<Py<PyDict>>> {
    let component_phase = PyDict::new(py);
    component_phase.set_item("fact", "component_phase")?;
    component_phase.set_item(
        "input_phase",
        component_phase_name(observations.component_phase.input),
    )?;
    component_phase.set_item("input_phase_value", observations.component_phase.input)?;
    component_phase.set_item(
        "effective_phase",
        component_phase_name(observations.component_phase.effective),
    )?;
    component_phase.set_item(
        "effective_phase_value",
        observations.component_phase.effective,
    )?;
    component_phase.set_item("source", observations.component_phase.source)?;

    let component_begin_atom = PyDict::new(py);
    component_begin_atom.set_item("fact", "component_begin_atom")?;
    component_begin_atom.set_item(
        "input_atom_idx",
        observations.component_begin_atom.input_atom_idx,
    )?;
    component_begin_atom.set_item(
        "effective_atom_idx",
        observations.component_begin_atom.effective_atom_idx,
    )?;
    component_begin_atom.set_item("source", observations.component_begin_atom.source)?;

    let begin_side = PyDict::new(py);
    begin_side.set_item("fact", "begin_side")?;
    begin_side.set_item("side_idx", observations.begin_side.side_idx)?;
    begin_side.set_item("candidate_count", observations.begin_side.candidate_count)?;

    let selected_begin = PyDict::new(py);
    selected_begin.set_item("fact", "selected_begin_token")?;
    selected_begin.set_item("neighbor_idx", observations.selected_begin.neighbor_idx)?;
    selected_begin.set_item("token", observations.selected_begin.token.clone())?;

    let first_emitted_candidate = PyDict::new(py);
    first_emitted_candidate.set_item("fact", "first_emitted_candidate")?;
    first_emitted_candidate.set_item(
        "neighbor_idx",
        observations.first_emitted_candidate.neighbor_idx,
    )?;

    let rdkit_adjustment = PyDict::new(py);
    rdkit_adjustment.set_item("fact", "rdkit_token_flip_adjustment")?;
    rdkit_adjustment.set_item("value", observations.rdkit_token_flip_adjustment.value())?;
    rdkit_adjustment.set_item(
        "root_begin_side_adjustment",
        observations
            .rdkit_token_flip_adjustment
            .root_begin_side_adjustment,
    )?;
    rdkit_adjustment.set_item(
        "begin_atom_is_root",
        observations.rdkit_token_flip_adjustment.begin_atom_is_root,
    )?;
    rdkit_adjustment.set_item(
        "begin_side_orientation",
        observations
            .rdkit_token_flip_adjustment
            .begin_side_orientation,
    )?;
    rdkit_adjustment.set_item(
        "adjacent_two_candidate_adjustment",
        observations
            .rdkit_token_flip_adjustment
            .adjacent_two_candidate_adjustment,
    )?;
    rdkit_adjustment.set_item(
        "adjacent_two_candidate_side_idx",
        observations
            .rdkit_token_flip_adjustment
            .adjacent_two_candidate_side_idx,
    )?;
    rdkit_adjustment.set_item(
        "selected_neighbor_is_root",
        observations
            .rdkit_token_flip_adjustment
            .selected_neighbor_is_root,
    )?;
    rdkit_adjustment.set_item(
        "adjacent_first_emitted_candidate_idx",
        observations
            .rdkit_token_flip_adjustment
            .adjacent_first_emitted_candidate_idx,
    )?;
    rdkit_adjustment.set_item(
        "adjacent_first_emitted_is_not_begin",
        observations
            .rdkit_token_flip_adjustment
            .adjacent_first_emitted_is_not_begin,
    )?;

    Ok(vec![
        component_phase.unbind(),
        component_begin_atom.unbind(),
        begin_side.unbind(),
        selected_begin.unbind(),
        first_emitted_candidate.unbind(),
        rdkit_adjustment.unbind(),
    ])
}

fn token_flip_facts_to_py(
    py: Python<'_>,
    facts: &[StereoTokenFlipFact],
) -> PyResult<Vec<Py<PyDict>>> {
    facts
        .iter()
        .map(|fact| {
            let row = PyDict::new(py);
            row.set_item("runtime_component_idx", fact.runtime_component_idx)?;
            row.set_item("token_flip", model_token_flip_name(fact.token_flip))?;
            Ok(row.unbind())
        })
        .collect()
}

fn token_observation_facts_to_py(
    py: Python<'_>,
    facts: &[StereoTokenObservationFact],
) -> PyResult<Vec<Py<PyDict>>> {
    facts
        .iter()
        .map(|fact| {
            let row = PyDict::new(py);
            row.set_item("observation_kind", fact.observation_kind())?;
            row.set_item("runtime_component_idx", fact.runtime_component_idx())?;
            row.set_item(
                "component_phase",
                model_component_phase_name(fact.component_phase()),
            )?;
            row.set_item(
                "selected_begin_token",
                fact.selected_begin_token().map(stereo_direction_token_name),
            )?;
            row.set_item(
                "selected_begin_neighbor_is_first_emitted",
                fact.selected_begin_neighbor_is_first_emitted(),
            )?;
            let rdkit_adjustment = fact.rdkit_token_flip_adjustment();
            row.set_item("rdkit_token_flip_adjustment", rdkit_adjustment.value())?;
            row.set_item(
                "rdkit_token_flip_adjustment_observations",
                vec![
                    (
                        "root_begin_side_orientation",
                        rdkit_adjustment.root_begin_side_orientation,
                    ),
                    (
                        "adjacent_two_candidate_first_emitted",
                        rdkit_adjustment.adjacent_two_candidate_first_emitted,
                    ),
                ]
                .into_iter()
                .filter_map(|(name, present)| present.then_some(name))
                .collect::<Vec<_>>(),
            )?;
            row.set_item(
                "implied_token_flip",
                model_token_flip_name(fact.implied_token_flip()),
            )?;
            Ok(row.unbind())
        })
        .collect()
}

fn assert_component_token_flip_boundary_invariants(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<()> {
    assert_token_flips_explained_by_constraint_state(runtime, graph, state)?;
    let resolved_selected_neighbors = resolved_selected_neighbors(runtime, state);
    for component_idx in 0..state.committed_component_token_flips.len() {
        let committed = state.committed_component_token_flips[component_idx]
            .map(component_token_flip_value_from_model)
            .unwrap_or(UNKNOWN_COMPONENT_TOKEN_FLIP);
        let inferred = inferred_component_token_flip(runtime, state, graph, component_idx)?;
        if let Some(inferred) = inferred {
            if committed != UNKNOWN_COMPONENT_TOKEN_FLIP && committed != inferred {
                let component_selection_complete = runtime.side_ids_by_component[component_idx]
                    .iter()
                    .all(|&side_idx| resolved_selected_neighbors[side_idx] >= 0);
                if !runtime.isolated_components[component_idx] && !component_selection_complete {
                    continue;
                }
                return Err(PyValueError::new_err(
                    "Stereo component token flip was committed inconsistently",
                ));
            }
        }
    }
    Ok(())
}

fn commit_component_token_basis_fact(
    state: &mut RootedConnectedStereoWalkerStateData,
    fact: StereoTokenBasisFact,
) -> PyResult<()> {
    let fact_slot =
        &mut Arc::make_mut(&mut state.stereo_token_basis_facts)[fact.runtime_component_idx];
    if let Some(existing_fact) = fact_slot {
        if *existing_fact != fact {
            return Err(PyValueError::new_err(format!(
                "Conflicting token-basis facts for runtime component {}",
                fact.runtime_component_idx
            )));
        }
    } else {
        *fact_slot = Some(fact);
    }
    Ok(())
}

fn model_token_flip_for_chosen_token(
    stored_token: &str,
    chosen_token: &str,
) -> PyResult<Option<StereoTokenFlip>> {
    if chosen_token == stored_token {
        return Ok(Some(StereoTokenFlip::Stored));
    }
    if chosen_token == flip_direction_token(stored_token)? {
        return Ok(Some(StereoTokenFlip::Flipped));
    }
    Ok(None)
}

fn commit_component_token_flip_fact(
    state: &mut RootedConnectedStereoWalkerStateData,
    fact: CommittedComponentTokenFlipFact,
) -> PyResult<()> {
    let existing = state.committed_component_token_flips[fact.runtime_component_idx];
    if let Some(existing) = existing {
        if existing != fact.token_flip {
            return Err(PyValueError::new_err(
                "Stereo deferred token was committed inconsistently",
            ));
        }
    } else {
        Arc::make_mut(&mut state.committed_component_token_flips)[fact.runtime_component_idx] =
            Some(fact.token_flip);
    }
    Ok(())
}

fn raw_tokens_for_deferred_edge(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    deferred: &DeferredDirectionalToken,
) -> PyResult<Vec<String>> {
    if deferred.begin_idx < 0 || deferred.end_idx < 0 {
        return Ok(deferred
            .component_tokens
            .first()
            .and_then(|component_token| component_token.reference_tokens.first().cloned())
            .into_iter()
            .collect());
    }

    let begin_idx = deferred.begin_idx as usize;
    let end_idx = deferred.end_idx as usize;
    let edge = canonical_edge(begin_idx, end_idx);
    let resolved_selected_neighbors =
        partial_joined_support_boundary_selected_neighbors(runtime, graph, state)?;
    let mut active_tokens = Vec::<String>::new();

    for &side_idx in runtime.edge_to_side_ids.get(&edge).into_iter().flatten() {
        let side_info = &runtime.side_infos[side_idx];
        let edge_neighbor_idx = if begin_idx == side_info.endpoint_atom_idx {
            end_idx
        } else if end_idx == side_info.endpoint_atom_idx {
            begin_idx
        } else {
            continue;
        };

        let selected_neighbor_idx = resolved_selected_neighbors[side_idx];
        if selected_neighbor_idx == edge_neighbor_idx as isize {
            active_tokens.push(emitted_candidate_token(side_info, begin_idx, end_idx)?);
        }
    }

    active_tokens.sort();
    active_tokens.dedup();
    Ok(active_tokens)
}

fn marker_events_for_deferred_component_token(
    runtime: &StereoWalkerRuntimeData,
    prefix: &str,
    deferred: &DeferredDirectionalToken,
    component_token: &DeferredDirectionalComponentToken,
    chosen_token: &str,
) -> PyResult<Vec<StereoMarkerEventFact>> {
    if deferred.begin_idx < 0 || deferred.end_idx < 0 {
        return Ok(Vec::new());
    }
    let marker = match chosen_token {
        "/" => StereoDirectionToken::Slash,
        "\\" => StereoDirectionToken::Backslash,
        _ => return Ok(Vec::new()),
    };
    let begin_idx = deferred.begin_idx as usize;
    let end_idx = deferred.end_idx as usize;
    let edge = canonical_edge(begin_idx, end_idx);
    let mut events = Vec::new();
    for &side_idx in runtime.edge_to_side_ids.get(&edge).into_iter().flatten() {
        let side_info = &runtime.side_infos[side_idx];
        if side_info.component_idx != component_token.component_idx {
            continue;
        }
        let edge_neighbor_idx = if begin_idx == side_info.endpoint_atom_idx {
            end_idx
        } else if end_idx == side_info.endpoint_atom_idx {
            begin_idx
        } else {
            continue;
        };
        if !side_info.candidate_neighbors.contains(&edge_neighbor_idx) {
            continue;
        }
        events.push(StereoMarkerEventFact::MarkerPlaced {
            side_idx,
            slot: direction_erased_slot(prefix),
            begin_idx,
            end_idx,
            marker,
            role: directional_token_role(prefix, None),
        });
    }
    Ok(events)
}

fn rdkit_marker_rows_accept_deferred_token(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    deferred: &DeferredDirectionalToken,
    component_token: &DeferredDirectionalComponentToken,
    implied_token_flip: StereoTokenFlip,
    chosen_token: &str,
    constraint_state: &StereoConstraintState,
    boundary_facts: &StereoSupportBoundaryFacts,
) -> PyResult<bool> {
    let Some(model_component_idx) = runtime
        .constraint_model
        .component_for_runtime_component(component_token.component_idx)
    else {
        return Err(PyValueError::new_err(
            "Deferred token references unknown runtime component",
        ));
    };
    if constraint_state.is_empty(model_component_idx) {
        return Ok(false);
    }
    let token_phase_assignment_ids = constraint_state
        .token_phase_remaining_by_component
        .get(model_component_idx)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let candidate_token_phase_assignment_ids = runtime
        .constraint_model
        .filter_token_phase_assignment_ids_for_token_flip(
            model_component_idx,
            component_token.component_idx,
            token_phase_assignment_ids,
            implied_token_flip,
        )?;
    if candidate_token_phase_assignment_ids.is_empty() {
        return Ok(false);
    }

    let mut marker_events = boundary_facts
        .marker_obligation_event_facts_by_component
        .get(model_component_idx)
        .cloned()
        .unwrap_or_default();
    marker_events.extend(marker_events_for_deferred_component_token(
        runtime,
        state.prefix.as_ref(),
        deferred,
        component_token,
        chosen_token,
    )?);
    let marker_events =
        rdkit_writer_slot_coalesced_marker_event_facts(model_component_idx, &marker_events);
    let survivor_state = rdkit_marker_row_survivor_component_state(
        runtime,
        model_component_idx,
        &candidate_token_phase_assignment_ids,
        &marker_events,
    )?;
    Ok(!survivor_state.row_ids_after_marker_events.is_empty())
}

fn accepted_deferred_token_flips(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    deferred: &DeferredDirectionalToken,
    component_token: &DeferredDirectionalComponentToken,
    chosen_token: &str,
    constraint_state: &StereoConstraintState,
    boundary_facts: &StereoSupportBoundaryFacts,
) -> PyResult<Vec<StereoTokenFlip>> {
    let mut accepted = BTreeSet::<StereoTokenFlip>::new();
    for reference_token in &component_token.reference_tokens {
        let Some(implied_token_flip) =
            model_token_flip_for_chosen_token(reference_token, chosen_token)?
        else {
            continue;
        };
        if rdkit_marker_rows_accept_deferred_token(
            runtime,
            state,
            deferred,
            component_token,
            implied_token_flip,
            chosen_token,
            constraint_state,
            boundary_facts,
        )? {
            accepted.insert(implied_token_flip);
        }
    }
    Ok(accepted.into_iter().collect())
}

fn deferred_token_legacy_topology_guard_applies(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    component_idx: usize,
) -> bool {
    graph.bond_count + 1 == graph.atom_count
        && legacy_topology_guard_component_has_multi_candidate_side(runtime, component_idx)
        && legacy_topology_guard_component_has_cross_component_carrier_edge(runtime, component_idx)
}

fn marker_event_fact_to_py(
    py: Python<'_>,
    component_idx: usize,
    fact: StereoMarkerEventFact,
) -> PyResult<Py<PyDict>> {
    let row = PyDict::new(py);
    row.set_item("component_idx", component_idx)?;
    match fact {
        StereoMarkerEventFact::MarkerPlaced {
            side_idx,
            slot,
            begin_idx,
            end_idx,
            marker,
            role,
        } => {
            row.set_item("event", "marker_placed")?;
            row.set_item("side_idx", side_idx)?;
            row.set_item("slot", slot)?;
            row.set_item("begin_idx", begin_idx)?;
            row.set_item("end_idx", end_idx)?;
            row.set_item("canonical_edge", canonical_edge(begin_idx, end_idx))?;
            row.set_item("marker", stereo_direction_token_name(marker))?;
            row.set_item("role", stereo_traversal_role_name(role))?;
        }
        StereoMarkerEventFact::NoMarker {
            side_idx,
            slot,
            begin_idx,
            end_idx,
            role,
        } => {
            row.set_item("event", "no_marker")?;
            row.set_item("side_idx", side_idx)?;
            row.set_item("slot", slot)?;
            row.set_item("begin_idx", begin_idx)?;
            row.set_item("end_idx", end_idx)?;
            row.set_item("canonical_edge", canonical_edge(begin_idx, end_idx))?;
            row.set_item("marker", Option::<&str>::None)?;
            row.set_item("role", stereo_traversal_role_name(role))?;
        }
    }
    Ok(row.unbind())
}

fn component_ids_for_edge(runtime: &StereoWalkerRuntimeData, edge: (usize, usize)) -> Vec<usize> {
    runtime
        .edge_to_side_ids
        .get(&edge)
        .into_iter()
        .flatten()
        .filter_map(|&side_idx| runtime.side_infos.get(side_idx))
        .map(|side_info| side_info.component_idx)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

struct RemainingVisibleMarkerBasisSummary {
    has_non_selected_visible_edge_basis: bool,
    has_shared_visible_edge_basis: bool,
}

fn remaining_component_visible_marker_basis_summary(
    runtime: &StereoWalkerRuntimeData,
    component_idx: usize,
    constraint_state: &StereoConstraintState,
) -> PyResult<RemainingVisibleMarkerBasisSummary> {
    let Some(model_component_idx) = runtime
        .constraint_model
        .component_for_runtime_component(component_idx)
    else {
        return Err(PyValueError::new_err(
            "Deferred token references unknown runtime component",
        ));
    };
    if constraint_state.is_empty(model_component_idx) {
        return Ok(RemainingVisibleMarkerBasisSummary {
            has_non_selected_visible_edge_basis: false,
            has_shared_visible_edge_basis: false,
        });
    }
    let Some(component) = runtime.constraint_model.components.get(model_component_idx) else {
        return Err(PyValueError::new_err(
            "visible-marker basis component index out of range",
        ));
    };
    let remaining_token_phase_assignment_ids = constraint_state
        .token_phase_remaining_by_component
        .get(model_component_idx)
        .map(|ids| ids.iter().copied().collect::<BTreeSet<_>>())
        .unwrap_or_default();
    if remaining_token_phase_assignment_ids.is_empty() {
        return Ok(RemainingVisibleMarkerBasisSummary {
            has_non_selected_visible_edge_basis: false,
            has_shared_visible_edge_basis: false,
        });
    }

    let mut has_non_selected_visible_edge_basis = false;
    let mut has_shared_visible_edge_basis = false;
    for row in &component.all_marker_placement_rows {
        if !remaining_token_phase_assignment_ids.contains(&row.token_phase_assignment_id) {
            continue;
        }
        let Some(token_phase_assignment) = component
            .all_token_phase_assignments
            .get(row.token_phase_assignment_id)
        else {
            return Err(PyValueError::new_err(
                "visible-marker basis token-phase index out of range",
            ));
        };
        let Some(carrier_neighbors) = component
            .all_neighbor_assignments
            .get(token_phase_assignment.neighbor_assignment_id)
        else {
            return Err(PyValueError::new_err(
                "visible-marker basis carrier assignment index out of range",
            ));
        };
        for (side_position, marker_neighbors) in row.marker_neighbor_sets.iter().enumerate() {
            let Some(&side_idx) = component.side_ids.get(side_position) else {
                return Err(PyValueError::new_err(
                    "visible-marker basis side index out of range",
                ));
            };
            let Some(side_info) = runtime.side_infos.get(side_idx) else {
                return Err(PyValueError::new_err(
                    "visible-marker basis runtime side index out of range",
                ));
            };
            let Some(&selected_neighbor_idx) = carrier_neighbors.get(side_position) else {
                return Err(PyValueError::new_err(
                    "visible-marker basis carrier side index out of range",
                ));
            };
            for &marker_neighbor_idx in marker_neighbors {
                if marker_neighbor_idx != selected_neighbor_idx {
                    has_non_selected_visible_edge_basis = true;
                }
                if component_ids_for_edge(
                    runtime,
                    canonical_edge(side_info.endpoint_atom_idx, marker_neighbor_idx),
                )
                .len()
                    > 1
                {
                    has_shared_visible_edge_basis = true;
                }
            }
        }
    }
    Ok(RemainingVisibleMarkerBasisSummary {
        has_non_selected_visible_edge_basis,
        has_shared_visible_edge_basis,
    })
}

fn remaining_component_has_shared_nonselected_visible_marker_basis(
    summary: &RemainingVisibleMarkerBasisSummary,
) -> bool {
    summary.has_non_selected_visible_edge_basis && summary.has_shared_visible_edge_basis
}

fn edge_neighbor_for_marker_event(
    runtime: &StereoWalkerRuntimeData,
    fact: StereoMarkerEventFact,
) -> Option<(usize, usize, (usize, usize))> {
    let StereoMarkerEventFact::MarkerPlaced {
        side_idx,
        begin_idx,
        end_idx,
        ..
    } = fact
    else {
        return None;
    };
    let side_info = runtime.side_infos.get(side_idx)?;
    let edge_neighbor_idx = if begin_idx == side_info.endpoint_atom_idx {
        end_idx
    } else if end_idx == side_info.endpoint_atom_idx {
        begin_idx
    } else {
        return None;
    };
    Some((
        side_idx,
        edge_neighbor_idx,
        canonical_edge(begin_idx, end_idx),
    ))
}

fn marker_basis_row_ids_for_event(
    runtime: &StereoWalkerRuntimeData,
    model_component_idx: usize,
    event: StereoMarkerEventFact,
    survivor_state: &RdkitMarkerRowSurvivorComponentState,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let Some((side_idx, edge_neighbor_idx, _)) = edge_neighbor_for_marker_event(runtime, event)
    else {
        return Ok((Vec::new(), Vec::new()));
    };
    let Some(component) = runtime.constraint_model.components.get(model_component_idx) else {
        return Err(PyValueError::new_err(
            "visible-marker basis diagnostic component index out of range",
        ));
    };
    let Some(side_position) = component
        .side_ids
        .iter()
        .position(|&candidate_side_idx| candidate_side_idx == side_idx)
    else {
        return Ok((Vec::new(), Vec::new()));
    };

    let mut selected_carrier_rows = BTreeSet::<usize>::new();
    let mut non_selected_visible_edge_rows = BTreeSet::<usize>::new();
    for &row_idx in &survivor_state.row_ids_after_marker_events {
        let Some(row) = component.all_marker_placement_rows.get(row_idx) else {
            return Err(PyValueError::new_err(
                "visible-marker basis diagnostic row index out of range",
            ));
        };
        let Some(token_phase_assignment) = component
            .all_token_phase_assignments
            .get(row.token_phase_assignment_id)
        else {
            return Err(PyValueError::new_err(
                "visible-marker basis diagnostic token-phase index out of range",
            ));
        };
        let Some(carrier_neighbors) = component
            .all_neighbor_assignments
            .get(token_phase_assignment.neighbor_assignment_id)
        else {
            return Err(PyValueError::new_err(
                "visible-marker basis diagnostic carrier assignment index out of range",
            ));
        };
        let Some(&selected_neighbor_idx) = carrier_neighbors.get(side_position) else {
            return Err(PyValueError::new_err(
                "visible-marker basis diagnostic carrier side index out of range",
            ));
        };
        let Some(marker_neighbors) = row.marker_neighbor_sets.get(side_position) else {
            return Err(PyValueError::new_err(
                "visible-marker basis diagnostic marker side index out of range",
            ));
        };
        if !marker_neighbors.contains(&edge_neighbor_idx) {
            continue;
        }
        if selected_neighbor_idx == edge_neighbor_idx {
            selected_carrier_rows.insert(row_idx);
        } else {
            non_selected_visible_edge_rows.insert(row_idx);
        }
    }
    Ok((
        selected_carrier_rows.into_iter().collect(),
        non_selected_visible_edge_rows.into_iter().collect(),
    ))
}

struct VisibleMarkerBasisDiagnostic {
    basis_class: &'static str,
    reference_token: String,
    implied_token_flip: StereoTokenFlip,
    row_ids_before_marker_events: Vec<usize>,
    row_ids_after_marker_events: Vec<usize>,
    row_ids_supporting_basis: Vec<usize>,
    marker_events: Vec<StereoMarkerEventFact>,
}

fn visible_marker_basis_diagnostic_to_py(
    py: Python<'_>,
    component_idx: usize,
    model_component_idx: usize,
    diagnostic: &VisibleMarkerBasisDiagnostic,
) -> PyResult<Py<PyDict>> {
    let row = PyDict::new(py);
    row.set_item("basis_class", diagnostic.basis_class)?;
    row.set_item("component_idx", component_idx)?;
    row.set_item("model_component_idx", model_component_idx)?;
    row.set_item("reference_token", &diagnostic.reference_token)?;
    row.set_item(
        "implied_token_flip",
        model_token_flip_name(diagnostic.implied_token_flip),
    )?;
    row.set_item(
        "row_ids_before_marker_events",
        diagnostic.row_ids_before_marker_events.clone(),
    )?;
    row.set_item(
        "row_ids_after_marker_events",
        diagnostic.row_ids_after_marker_events.clone(),
    )?;
    row.set_item(
        "row_ids_supporting_basis",
        diagnostic.row_ids_supporting_basis.clone(),
    )?;
    row.set_item(
        "row_count_supporting_basis",
        diagnostic.row_ids_supporting_basis.len(),
    )?;
    row.set_item(
        "marker_events",
        diagnostic
            .marker_events
            .iter()
            .copied()
            .map(|event| marker_event_fact_to_py(py, model_component_idx, event))
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    Ok(row.unbind())
}

fn visible_marker_basis_diagnostics_for_component_token(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    deferred: &DeferredDirectionalToken,
    component_token: &DeferredDirectionalComponentToken,
    chosen_token: &str,
    constraint_state: &StereoConstraintState,
    boundary_facts: &StereoSupportBoundaryFacts,
) -> PyResult<Vec<VisibleMarkerBasisDiagnostic>> {
    let Some(model_component_idx) = runtime
        .constraint_model
        .component_for_runtime_component(component_token.component_idx)
    else {
        return Err(PyValueError::new_err(
            "Deferred token references unknown runtime component",
        ));
    };
    if constraint_state.is_empty(model_component_idx) {
        return Ok(Vec::new());
    }
    let token_phase_assignment_ids = constraint_state
        .token_phase_remaining_by_component
        .get(model_component_idx)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let marker_events = marker_events_for_deferred_component_token(
        runtime,
        state.prefix.as_ref(),
        deferred,
        component_token,
        chosen_token,
    )?;
    if marker_events.is_empty() {
        return Ok(Vec::new());
    }

    let mut diagnostics = Vec::<VisibleMarkerBasisDiagnostic>::new();
    for reference_token in &component_token.reference_tokens {
        let Some(implied_token_flip) =
            model_token_flip_for_chosen_token(reference_token, chosen_token)?
        else {
            continue;
        };
        let candidate_token_phase_assignment_ids = runtime
            .constraint_model
            .filter_token_phase_assignment_ids_for_token_flip(
                model_component_idx,
                component_token.component_idx,
                token_phase_assignment_ids,
                implied_token_flip,
            )?;
        if candidate_token_phase_assignment_ids.is_empty() {
            continue;
        }

        let mut all_marker_events = boundary_facts
            .marker_obligation_event_facts_by_component
            .get(model_component_idx)
            .cloned()
            .unwrap_or_default();
        all_marker_events.extend(marker_events.iter().copied());
        let all_marker_events =
            rdkit_writer_slot_coalesced_marker_event_facts(model_component_idx, &all_marker_events);
        let survivor_state = rdkit_marker_row_survivor_component_state(
            runtime,
            model_component_idx,
            &candidate_token_phase_assignment_ids,
            &all_marker_events,
        )?;
        if survivor_state.row_ids_after_marker_events.is_empty() {
            continue;
        }

        let mut selected_carrier_rows = BTreeSet::<usize>::new();
        let mut non_selected_visible_edge_rows = BTreeSet::<usize>::new();
        let mut shared_visible_edge_rows = BTreeSet::<usize>::new();
        for &event in &marker_events {
            let Some((_, _, edge)) = edge_neighbor_for_marker_event(runtime, event) else {
                continue;
            };
            let (selected_rows, non_selected_rows) = marker_basis_row_ids_for_event(
                runtime,
                model_component_idx,
                event,
                &survivor_state,
            )?;
            selected_carrier_rows.extend(selected_rows);
            non_selected_visible_edge_rows.extend(non_selected_rows);
            if component_ids_for_edge(runtime, edge).len() > 1 {
                shared_visible_edge_rows
                    .extend(survivor_state.row_ids_after_marker_events.iter().copied());
            }
        }

        for (basis_class, row_ids_supporting_basis) in [
            (
                "selected_carrier",
                selected_carrier_rows.into_iter().collect::<Vec<_>>(),
            ),
            (
                "non_selected_visible_edge",
                non_selected_visible_edge_rows
                    .into_iter()
                    .collect::<Vec<_>>(),
            ),
            (
                "shared_visible_edge",
                shared_visible_edge_rows.into_iter().collect::<Vec<_>>(),
            ),
        ] {
            if row_ids_supporting_basis.is_empty() {
                continue;
            }
            diagnostics.push(VisibleMarkerBasisDiagnostic {
                basis_class,
                reference_token: reference_token.clone(),
                implied_token_flip,
                row_ids_before_marker_events: survivor_state.row_ids_before_marker_events.clone(),
                row_ids_after_marker_events: survivor_state.row_ids_after_marker_events.clone(),
                row_ids_supporting_basis,
                marker_events: marker_events.clone(),
            });
        }
    }
    Ok(diagnostics)
}

fn accepted_token_flips_from_raw_selected_carrier_basis(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    deferred: &DeferredDirectionalToken,
    component_token: &DeferredDirectionalComponentToken,
    raw_tokens: &[String],
    chosen_token: &str,
    constraint_state: &StereoConstraintState,
    boundary_facts: &StereoSupportBoundaryFacts,
) -> PyResult<Vec<StereoTokenFlip>> {
    let mut accepted = BTreeSet::<StereoTokenFlip>::new();
    for raw_token in raw_tokens {
        let Some(implied_token_flip) = model_token_flip_for_chosen_token(raw_token, chosen_token)?
        else {
            continue;
        };
        if rdkit_marker_rows_accept_deferred_token(
            runtime,
            state,
            deferred,
            component_token,
            implied_token_flip,
            chosen_token,
            constraint_state,
            boundary_facts,
        )? {
            accepted.insert(implied_token_flip);
        }
    }
    Ok(accepted.into_iter().collect())
}

fn token_flips_to_py_names(token_flips: &[StereoTokenFlip]) -> Vec<&'static str> {
    token_flips
        .iter()
        .copied()
        .map(model_token_flip_name)
        .collect()
}

fn legacy_topology_guard_component_has_multi_candidate_side(
    runtime: &StereoWalkerRuntimeData,
    runtime_component_idx: usize,
) -> bool {
    runtime
        .side_ids_by_component
        .get(runtime_component_idx)
        .into_iter()
        .flatten()
        .any(|&side_idx| runtime.side_infos[side_idx].candidate_neighbors.len() > 1)
}

fn legacy_topology_guard_component_has_cross_component_carrier_edge(
    runtime: &StereoWalkerRuntimeData,
    runtime_component_idx: usize,
) -> bool {
    runtime
        .side_ids_by_component
        .get(runtime_component_idx)
        .into_iter()
        .flatten()
        .any(|&side_idx| {
            let side_info = &runtime.side_infos[side_idx];
            side_info.candidate_neighbors.iter().any(|&neighbor_idx| {
                runtime
                    .edge_to_side_ids
                    .get(&canonical_edge(side_info.endpoint_atom_idx, neighbor_idx))
                    .into_iter()
                    .flatten()
                    .any(|&other_side_idx| {
                        runtime.side_infos[other_side_idx].component_idx != runtime_component_idx
                    })
            })
        })
}

fn accepted_single_component_deferred_token_flips(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    deferred: &DeferredDirectionalToken,
    component_token: &DeferredDirectionalComponentToken,
    raw_token: &str,
    chosen_token: &str,
    constraint_state: &StereoConstraintState,
    boundary_facts: &StereoSupportBoundaryFacts,
) -> PyResult<Vec<StereoTokenFlip>> {
    let raw_accepted = if let Some(implied_token_flip) =
        model_token_flip_for_chosen_token(raw_token, chosen_token)?
    {
        if rdkit_marker_rows_accept_deferred_token(
            runtime,
            state,
            deferred,
            component_token,
            implied_token_flip,
            chosen_token,
            constraint_state,
            boundary_facts,
        )? {
            vec![implied_token_flip]
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    let remaining_basis_summary = remaining_component_visible_marker_basis_summary(
        runtime,
        component_token.component_idx,
        constraint_state,
    )?;
    let has_frontier_visible_marker_basis =
        remaining_component_has_shared_nonselected_visible_marker_basis(&remaining_basis_summary);
    if has_frontier_visible_marker_basis {
        return accepted_deferred_token_flips(
            runtime,
            state,
            deferred,
            component_token,
            chosen_token,
            constraint_state,
            boundary_facts,
        );
    }

    Ok(raw_accepted)
}

fn candidate_tokens_from_raw_deferred_tokens(raw_tokens: &[String]) -> PyResult<Vec<String>> {
    let mut candidates = BTreeSet::<String>::new();
    for raw_token in raw_tokens {
        candidates.insert(raw_token.clone());
        candidates.insert(flip_direction_token(raw_token)?);
    }
    Ok(candidates.into_iter().collect())
}

fn accepted_single_component_deferred_token_flips_from_raw_tokens(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    deferred: &DeferredDirectionalToken,
    component_token: &DeferredDirectionalComponentToken,
    raw_tokens: &[String],
    chosen_token: &str,
    constraint_state: &StereoConstraintState,
    boundary_facts: &StereoSupportBoundaryFacts,
) -> PyResult<Vec<StereoTokenFlip>> {
    let [raw_token] = raw_tokens else {
        return accepted_deferred_token_flips(
            runtime,
            state,
            deferred,
            component_token,
            chosen_token,
            constraint_state,
            boundary_facts,
        );
    };
    accepted_single_component_deferred_token_flips(
        runtime,
        state,
        deferred,
        component_token,
        raw_token,
        chosen_token,
        constraint_state,
        boundary_facts,
    )
}

fn deferred_token_support_from_constraint_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    deferred: &DeferredDirectionalToken,
) -> PyResult<Vec<String>> {
    let boundary_facts = support_boundary_facts_from_walker_state(
        runtime,
        graph,
        state,
        state.stereo_selected_neighbors.as_ref(),
        true,
    )?;
    let constraint_state =
        boundary_facts.constraint_state(runtime, StereoConstraintLayer::Semantic)?;
    if deferred.component_tokens.len() == 1 {
        let literal_token = if deferred.begin_idx >= 0 && deferred.end_idx >= 0 {
            Some(
                graph
                    .bond_token(deferred.begin_idx as usize, deferred.end_idx as usize)
                    .ok_or_else(|| {
                        PyKeyError::new_err(format!(
                            "No bond between atoms {} and {}",
                            deferred.begin_idx, deferred.end_idx
                        ))
                    })?
                    .to_owned(),
            )
        } else {
            None
        };
        let raw_tokens = raw_tokens_for_deferred_edge(runtime, graph, state, deferred)?;
        if raw_tokens.is_empty() {
            return Ok(vec![literal_token.unwrap_or_default()]);
        }
        let component_token = &deferred.component_tokens[0];
        let mut out = Vec::new();
        for candidate_token in candidate_tokens_from_raw_deferred_tokens(&raw_tokens)? {
            if !accepted_single_component_deferred_token_flips_from_raw_tokens(
                runtime,
                state,
                deferred,
                component_token,
                &raw_tokens,
                candidate_token.as_str(),
                &constraint_state,
                &boundary_facts,
            )?
            .is_empty()
            {
                out.push(candidate_token);
            }
        }
        out.sort();
        out.dedup();
        return Ok(out);
    }
    let mut out = Vec::new();
    for candidate_token in ["/", "\\"] {
        let mut compatible = true;
        for component_token in deferred.component_tokens.iter() {
            let accepted_flips = accepted_deferred_token_flips(
                runtime,
                state,
                deferred,
                component_token,
                candidate_token,
                &constraint_state,
                &boundary_facts,
            )?;
            if accepted_flips.len() != 1 {
                compatible = false;
                break;
            }
        }
        if compatible {
            out.push(candidate_token.to_owned());
        }
    }
    Ok(out)
}

fn deferred_token_support(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    deferred: &DeferredDirectionalToken,
) -> PyResult<Vec<String>> {
    if !deferred.component_tokens.is_empty() {
        return deferred_token_support_from_constraint_state(runtime, graph, state, deferred);
    }
    let literal_token = if deferred.begin_idx >= 0 && deferred.end_idx >= 0 {
        Some(
            graph
                .bond_token(deferred.begin_idx as usize, deferred.end_idx as usize)
                .ok_or_else(|| {
                    PyKeyError::new_err(format!(
                        "No bond between atoms {} and {}",
                        deferred.begin_idx, deferred.end_idx
                    ))
                })?
                .to_owned(),
        )
    } else {
        None
    };
    Ok(vec![literal_token.unwrap_or_default()])
}

fn commit_deferred_token_choice(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &mut RootedConnectedStereoWalkerStateData,
    deferred: &DeferredDirectionalToken,
    chosen_token: &str,
) -> PyResult<()> {
    if deferred.component_tokens.is_empty() {
        let literal_token = if deferred.begin_idx >= 0 && deferred.end_idx >= 0 {
            graph
                .bond_token(deferred.begin_idx as usize, deferred.end_idx as usize)
                .ok_or_else(|| {
                    PyKeyError::new_err(format!(
                        "No bond between atoms {} and {}",
                        deferred.begin_idx, deferred.end_idx
                    ))
                })?
        } else {
            ""
        };
        if chosen_token != literal_token {
            return Err(PyKeyError::new_err(format!(
                "Token {chosen_token:?} is not available for deferred stereo token"
            )));
        }
        return assert_component_token_flip_boundary_invariants(runtime, graph, state);
    }

    if deferred.component_tokens.len() == 1 {
        let raw_tokens = raw_tokens_for_deferred_edge(runtime, graph, state, deferred)?;
        if raw_tokens.is_empty() {
            let literal_token = if deferred.begin_idx >= 0 && deferred.end_idx >= 0 {
                graph
                    .bond_token(deferred.begin_idx as usize, deferred.end_idx as usize)
                    .ok_or_else(|| {
                        PyKeyError::new_err(format!(
                            "No bond between atoms {} and {}",
                            deferred.begin_idx, deferred.end_idx
                        ))
                    })?
            } else {
                ""
            };
            if chosen_token != literal_token {
                return Err(PyKeyError::new_err(format!(
                    "Token {chosen_token:?} is not available for deferred stereo token"
                )));
            }
            return assert_component_token_flip_boundary_invariants(runtime, graph, state);
        };
        let boundary_facts = support_boundary_facts_from_walker_state(
            runtime,
            graph,
            state,
            state.stereo_selected_neighbors.as_ref(),
            true,
        )?;
        let constraint_state =
            boundary_facts.constraint_state(runtime, StereoConstraintLayer::Semantic)?;
        let component_token = &deferred.component_tokens[0];
        let accepted_flips = accepted_single_component_deferred_token_flips_from_raw_tokens(
            runtime,
            state,
            deferred,
            component_token,
            &raw_tokens,
            chosen_token,
            &constraint_state,
            &boundary_facts,
        )?;
        if accepted_flips.is_empty() {
            return Err(PyKeyError::new_err(format!(
                "Token {chosen_token:?} is not available for deferred stereo token"
            )));
        };
        if let [chosen_model_flip] = accepted_flips.as_slice() {
            commit_component_token_flip_fact(
                state,
                CommittedComponentTokenFlipFact {
                    runtime_component_idx: component_token.component_idx,
                    token_flip: *chosen_model_flip,
                },
            )?;
        } else {
            commit_component_token_basis_fact(
                state,
                StereoTokenBasisFact {
                    runtime_component_idx: component_token.component_idx,
                    selected_begin_token: StereoDirectionToken::from_str(chosen_token)?,
                },
            )?;
            commit_component_token_flip_fact(
                state,
                CommittedComponentTokenFlipFact {
                    runtime_component_idx: component_token.component_idx,
                    token_flip: StereoTokenFlip::Stored,
                },
            )?;
        }
        return assert_component_token_flip_boundary_invariants(runtime, graph, state);
    }

    let boundary_facts = support_boundary_facts_from_walker_state(
        runtime,
        graph,
        state,
        state.stereo_selected_neighbors.as_ref(),
        true,
    )?;
    let constraint_state =
        boundary_facts.constraint_state(runtime, StereoConstraintLayer::Semantic)?;
    for component_token in deferred.component_tokens.iter() {
        let accepted_flips = accepted_deferred_token_flips(
            runtime,
            state,
            deferred,
            component_token,
            chosen_token,
            &constraint_state,
            &boundary_facts,
        )?;
        let [chosen_model_flip] = accepted_flips.as_slice() else {
            return Err(PyKeyError::new_err(format!(
                "Token {chosen_token:?} is not available for deferred stereo token"
            )));
        };
        commit_component_token_flip_fact(
            state,
            CommittedComponentTokenFlipFact {
                runtime_component_idx: component_token.component_idx,
                token_flip: *chosen_model_flip,
            },
        )?;
    }
    assert_component_token_flip_boundary_invariants(runtime, graph, state)
}

fn enter_atom_successors_by_token(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    atom_idx: usize,
    parent_idx: Option<usize>,
) -> PyResult<BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>> {
    let mut base_state = state.clone();
    base_state.action_stack.pop();

    let visited_now = visited_with_marked(&base_state.visited, atom_idx);
    let visited_count_now = base_state.visited_count + 1;
    let mut pending_now = Arc::unwrap_or_clone(base_state.pending.clone());
    let closures_here = take_pending_for_atom(&mut pending_now, atom_idx);
    let ordered_groups = ordered_neighbor_groups(graph, atom_idx, visited_now.as_ref());
    let is_chiral_atom = graph.atom_chiral_tags[atom_idx] != "CHI_UNSPECIFIED";

    if closures_here.is_empty() && ordered_groups.iter().all(|group| group.len() == 1) {
        let chosen_children = ordered_groups
            .iter()
            .map(|group| group[0])
            .collect::<Vec<_>>();
        let mut successors = Vec::<(String, Vec<RootedConnectedStereoWalkerStateData>)>::new();
        let mut status = Ok(());
        permutations_copy_distinct(&chosen_children, &mut |child_order| {
            if status.is_err() {
                return;
            }
            let outcome: PyResult<()> = (|| {
                let atom_token = if !is_chiral_atom {
                    graph.atom_tokens[atom_idx].clone()
                } else {
                    let emitted_neighbor_order =
                        stereo_neighbor_order(graph, atom_idx, parent_idx, &[], child_order)?;
                    stereo_atom_token(graph, atom_idx, &emitted_neighbor_order)?
                };
                let mut successor = base_state.clone();
                successor.visited = visited_now.clone();
                successor.visited_count = visited_count_now;
                successor.pending = Arc::new(pending_now.clone());
                if !child_order.is_empty() {
                    successor.action_stack.push(WalkerAction::ProcessChildren {
                        parent_idx: atom_idx,
                        child_order: Arc::<[usize]>::from(child_order.to_vec()),
                        next_branch_index: 0,
                    });
                }
                push_literal_token(&mut successor.prefix, &atom_token);
                assert_component_token_flip_boundary_invariants(runtime, graph, &successor)?;
                push_successor_bucket(&mut successors, atom_token, successor);
                Ok(())
            })();
            if let Err(err) = outcome {
                status = Err(err);
            }
        });
        status?;
        return Ok(finalize_linear_structural_transitions(successors));
    }

    if runtime.side_infos.is_empty() {
        return enter_atom_successors_without_bond_stereo(AtomStereoExpansionInput {
            graph,
            base_state: &base_state,
            atom_idx,
            parent_idx,
            visited_now,
            visited_count_now,
            pending_now,
            closures_here,
            ordered_groups,
            is_chiral_atom,
        });
    }

    let edge_context = StereoEdgeEmissionContext {
        graph,
        constraint_model: &runtime.constraint_model,
        side_infos: &runtime.side_infos,
        side_ids_by_component: &runtime.side_ids_by_component,
        edge_to_side_ids: &runtime.edge_to_side_ids,
        isolated_components: &runtime.isolated_components,
    };
    let mut successors = Vec::<(String, Vec<RootedConnectedStereoWalkerStateData>)>::new();
    let mut status = Ok(());
    for_each_cartesian_choice(&ordered_groups, &mut |chosen_children| {
        if status.is_err() {
            return;
        }

        let ring_actions =
            ring_actions_for_choice(closures_here.len(), &ordered_groups, chosen_children);
        let opening_target_count = ring_actions
            .iter()
            .filter(|action| matches!(action, RingAction::Open(_)))
            .count();

        permutations_copy_distinct(&ring_actions, &mut |ring_action_order| {
            if status.is_err() {
                return;
            }

            let outcome: PyResult<()> = (|| {
                let mut current_pending = pending_now.clone();
                let mut current_free = Arc::unwrap_or_clone(base_state.free_labels.clone());
                let mut current_next = base_state.next_label;
                let mut current_component_phases =
                    Arc::unwrap_or_clone(base_state.stereo_component_phases.clone());
                let mut current_selected_neighbors =
                    Arc::unwrap_or_clone(base_state.stereo_selected_neighbors.clone());
                let mut current_selected_orientations =
                    Arc::unwrap_or_clone(base_state.stereo_selected_orientations.clone());
                let mut current_first_emitted_candidates =
                    Arc::unwrap_or_clone(base_state.stereo_first_emitted_candidates.clone());
                let mut current_component_begin_atoms =
                    Arc::unwrap_or_clone(base_state.stereo_component_begin_atoms.clone());
                let mut current_deferred_carrier_choice_constraints =
                    Arc::unwrap_or_clone(base_state.deferred_carrier_choice_constraints.clone());
                let mut current_token_basis_facts =
                    Arc::unwrap_or_clone(base_state.stereo_token_basis_facts.clone());
                let mut current_marker_event_traces =
                    Arc::unwrap_or_clone(base_state.marker_event_traces.clone());
                let mut current_ring_actions = Vec::<WalkerAction>::with_capacity(
                    closures_here.len() * 2 + opening_target_count,
                );
                let mut labels_freed_after_atom = Vec::<usize>::with_capacity(closures_here.len());
                let mut ring_neighbor_order = is_chiral_atom.then(Vec::<usize>::new);

                for ring_action in ring_action_order {
                    match *ring_action {
                        RingAction::Close(closure_idx) => {
                            let closure = &closures_here[closure_idx];
                            let EmittedEdgePartResult {
                                part: bond_part,
                                selected_neighbors: updated_neighbors,
                                selected_orientations: updated_orientations,
                                first_emitted_candidates: updated_first_candidates,
                                deferred_carrier_choice_constraints:
                                    updated_deferred_carrier_choice_constraints,
                                token_basis_facts: updated_token_basis_facts,
                            } = emitted_edge_part(
                                &edge_context,
                                &StereoEdgeEmissionState {
                                    component_phases: &current_component_phases,
                                    selected_neighbors: &current_selected_neighbors,
                                    selected_orientations: &current_selected_orientations,
                                    first_emitted_candidates: &current_first_emitted_candidates,
                                    component_begin_atoms: &current_component_begin_atoms,
                                    deferred_carrier_choice_constraints:
                                        &current_deferred_carrier_choice_constraints,
                                    token_basis_facts: &current_token_basis_facts,
                                    marker_event_traces: &current_marker_event_traces,
                                    committed_component_token_flips: &base_state
                                        .committed_component_token_flips,
                                },
                                atom_idx,
                                closure.other_atom_idx,
                            )?;
                            if matches!(bond_part, Part::Literal(_)) {
                                append_rdkit_marker_event_traces_for_edge(
                                    runtime,
                                    base_state.prefix.as_ref(),
                                    &mut current_marker_event_traces,
                                    atom_idx as isize,
                                    closure.other_atom_idx as isize,
                                    marker_event_marker_from_literal_part(&bond_part),
                                    StereoTraversalRole::RingClose,
                                );
                            }
                            current_selected_neighbors = updated_neighbors;
                            current_selected_orientations = updated_orientations;
                            current_first_emitted_candidates = updated_first_candidates;
                            current_deferred_carrier_choice_constraints =
                                updated_deferred_carrier_choice_constraints;
                            current_token_basis_facts = updated_token_basis_facts;
                            if let Some(action) = part_to_action(bond_part) {
                                current_ring_actions.push(action);
                            }
                            current_ring_actions.push(WalkerAction::EmitRingLabel(closure.label));
                            labels_freed_after_atom.push(closure.label);
                            if let Some(order) = &mut ring_neighbor_order {
                                order.push(closure.other_atom_idx);
                            }
                        }
                        RingAction::Open(target_idx) => {
                            let label = allocate_label(&mut current_free, &mut current_next);
                            current_ring_actions.push(WalkerAction::EmitRingLabel(label));
                            append_rdkit_marker_event_traces_for_edge(
                                runtime,
                                base_state.prefix.as_ref(),
                                &mut current_marker_event_traces,
                                atom_idx as isize,
                                target_idx as isize,
                                None,
                                StereoTraversalRole::RingOpen,
                            );
                            let (updated_phases, updated_begin_atoms) =
                                component_phases_after_edge(
                                    graph,
                                    &runtime.stereo_component_ids,
                                    &runtime.isolated_components,
                                    &current_component_phases,
                                    &current_component_begin_atoms,
                                    atom_idx,
                                    target_idx,
                                )?;
                            let (updated_neighbors, updated_orientations) =
                                force_known_begin_side_selection(
                                    runtime,
                                    &updated_phases,
                                    &updated_begin_atoms,
                                    &current_selected_neighbors,
                                    &current_selected_orientations,
                                );
                            current_selected_neighbors = updated_neighbors;
                            current_selected_orientations = updated_orientations;
                            let (updated_phases, updated_begin_atoms) =
                                defer_component_phase_for_unresolved_begin_side(
                                    runtime,
                                    graph,
                                    &updated_phases,
                                    &updated_begin_atoms,
                                    &current_selected_neighbors,
                                    atom_idx,
                                    target_idx,
                                )?;
                            current_component_phases = updated_phases;
                            current_component_begin_atoms = updated_begin_atoms;
                            add_pending(
                                &mut current_pending,
                                target_idx,
                                PendingRing {
                                    label,
                                    other_atom_idx: atom_idx,
                                },
                            );
                            if let Some(order) = &mut ring_neighbor_order {
                                order.push(target_idx);
                            }
                        }
                    }
                }

                for label in labels_freed_after_atom {
                    insert_sorted(&mut current_free, label);
                }
                permutations_copy_distinct(chosen_children, &mut |child_order| {
                    if status.is_err() {
                        return;
                    }

                    let inner: PyResult<()> = (|| {
                        let atom_token = if !is_chiral_atom {
                            graph.atom_tokens[atom_idx].clone()
                        } else {
                            let emitted_neighbor_order = stereo_neighbor_order(
                                graph,
                                atom_idx,
                                parent_idx,
                                ring_neighbor_order.as_deref().unwrap_or(&[]),
                                child_order,
                            )?;
                            stereo_atom_token(graph, atom_idx, &emitted_neighbor_order)?
                        };
                        let mut successor = RootedConnectedStereoWalkerStateData {
                            prefix: base_state.prefix.clone(),
                            visited: visited_now.clone(),
                            visited_count: visited_count_now,
                            pending: Arc::new(current_pending.clone()),
                            free_labels: Arc::new(current_free.clone()),
                            next_label: current_next,
                            stereo_component_phases: Arc::new(current_component_phases.clone()),
                            stereo_selected_neighbors: Arc::new(current_selected_neighbors.clone()),
                            stereo_selected_orientations: Arc::new(
                                current_selected_orientations.clone(),
                            ),
                            stereo_first_emitted_candidates: Arc::new(
                                current_first_emitted_candidates.clone(),
                            ),
                            stereo_component_begin_atoms: Arc::new(
                                current_component_begin_atoms.clone(),
                            ),
                            committed_component_token_flips: base_state
                                .committed_component_token_flips
                                .clone(),
                            deferred_carrier_choice_constraints: Arc::new(
                                current_deferred_carrier_choice_constraints.clone(),
                            ),
                            stereo_token_basis_facts: Arc::new(current_token_basis_facts.clone()),
                            directional_marker_traces: base_state.directional_marker_traces.clone(),
                            marker_event_traces: Arc::new(current_marker_event_traces.clone()),
                            action_stack: base_state.action_stack.clone(),
                        };
                        if !child_order.is_empty() {
                            successor.action_stack.push(WalkerAction::ProcessChildren {
                                parent_idx: atom_idx,
                                child_order: Arc::<[usize]>::from(child_order.to_vec()),
                                next_branch_index: 0,
                            });
                        }
                        for action in current_ring_actions.iter().rev() {
                            successor.action_stack.push(action.clone());
                        }
                        push_literal_token(&mut successor.prefix, &atom_token);
                        assert_component_token_flip_boundary_invariants(
                            runtime, graph, &successor,
                        )?;
                        push_successor_bucket(&mut successors, atom_token, successor);
                        Ok(())
                    })();
                    if let Err(err) = inner {
                        status = Err(err);
                    }
                });

                Ok(())
            })();

            if let Err(err) = outcome {
                status = Err(err);
            }
        });
    });
    status?;
    Ok(finalize_linear_structural_transitions(successors))
}

fn enter_atom_successors_without_bond_stereo(
    input: AtomStereoExpansionInput<'_>,
) -> PyResult<BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>> {
    let AtomStereoExpansionInput {
        graph,
        base_state,
        atom_idx,
        parent_idx,
        visited_now,
        visited_count_now,
        pending_now,
        closures_here,
        ordered_groups,
        is_chiral_atom,
    } = input;
    let mut successors = Vec::<(String, Vec<RootedConnectedStereoWalkerStateData>)>::new();
    let mut status = Ok(());

    for_each_cartesian_choice(&ordered_groups, &mut |chosen_children| {
        if status.is_err() {
            return;
        }

        let ring_actions =
            ring_actions_for_choice(closures_here.len(), &ordered_groups, chosen_children);
        let opening_target_count = ring_actions
            .iter()
            .filter(|action| matches!(action, RingAction::Open(_)))
            .count();

        permutations_copy_distinct(&ring_actions, &mut |ring_action_order| {
            if status.is_err() {
                return;
            }

            let outcome: PyResult<()> = (|| {
                let mut current_pending = pending_now.clone();
                let mut current_free = Arc::unwrap_or_clone(base_state.free_labels.clone());
                let mut current_next = base_state.next_label;
                let mut current_ring_actions = Vec::<WalkerAction>::with_capacity(
                    closures_here.len() * 2 + opening_target_count,
                );
                let mut labels_freed_after_atom = Vec::<usize>::with_capacity(closures_here.len());
                let mut ring_neighbor_order = is_chiral_atom.then(Vec::<usize>::new);

                for ring_action in ring_action_order {
                    match *ring_action {
                        RingAction::Close(closure_idx) => {
                            let closure = &closures_here[closure_idx];
                            let bond_token = graph
                                .bond_token(atom_idx, closure.other_atom_idx)
                                .ok_or_else(|| {
                                    PyKeyError::new_err(format!(
                                        "No bond between atoms {atom_idx} and {}",
                                        closure.other_atom_idx
                                    ))
                                })?
                                .to_owned();
                            if !bond_token.is_empty() {
                                current_ring_actions.push(WalkerAction::EmitLiteral(bond_token));
                            }
                            current_ring_actions.push(WalkerAction::EmitRingLabel(closure.label));
                            labels_freed_after_atom.push(closure.label);
                            if let Some(order) = &mut ring_neighbor_order {
                                order.push(closure.other_atom_idx);
                            }
                        }
                        RingAction::Open(target_idx) => {
                            let label = allocate_label(&mut current_free, &mut current_next);
                            current_ring_actions.push(WalkerAction::EmitRingLabel(label));
                            add_pending(
                                &mut current_pending,
                                target_idx,
                                PendingRing {
                                    label,
                                    other_atom_idx: atom_idx,
                                },
                            );
                            if let Some(order) = &mut ring_neighbor_order {
                                order.push(target_idx);
                            }
                        }
                    }
                }

                for label in labels_freed_after_atom {
                    insert_sorted(&mut current_free, label);
                }
                permutations_copy_distinct(chosen_children, &mut |child_order| {
                    if status.is_err() {
                        return;
                    }

                    let inner: PyResult<()> = (|| {
                        let atom_token = if !is_chiral_atom {
                            graph.atom_tokens[atom_idx].clone()
                        } else {
                            let emitted_neighbor_order = stereo_neighbor_order(
                                graph,
                                atom_idx,
                                parent_idx,
                                ring_neighbor_order.as_deref().unwrap_or(&[]),
                                child_order,
                            )?;
                            stereo_atom_token(graph, atom_idx, &emitted_neighbor_order)?
                        };
                        let mut successor = RootedConnectedStereoWalkerStateData {
                            prefix: base_state.prefix.clone(),
                            visited: visited_now.clone(),
                            visited_count: visited_count_now,
                            pending: Arc::new(current_pending.clone()),
                            free_labels: Arc::new(current_free.clone()),
                            next_label: current_next,
                            stereo_component_phases: base_state.stereo_component_phases.clone(),
                            stereo_selected_neighbors: base_state.stereo_selected_neighbors.clone(),
                            stereo_selected_orientations: base_state
                                .stereo_selected_orientations
                                .clone(),
                            stereo_first_emitted_candidates: base_state
                                .stereo_first_emitted_candidates
                                .clone(),
                            stereo_component_begin_atoms: base_state
                                .stereo_component_begin_atoms
                                .clone(),
                            committed_component_token_flips: base_state
                                .committed_component_token_flips
                                .clone(),
                            deferred_carrier_choice_constraints: base_state
                                .deferred_carrier_choice_constraints
                                .clone(),
                            stereo_token_basis_facts: base_state.stereo_token_basis_facts.clone(),
                            directional_marker_traces: base_state.directional_marker_traces.clone(),
                            marker_event_traces: base_state.marker_event_traces.clone(),
                            action_stack: base_state.action_stack.clone(),
                        };
                        if !child_order.is_empty() {
                            successor.action_stack.push(WalkerAction::ProcessChildren {
                                parent_idx: atom_idx,
                                child_order: Arc::<[usize]>::from(child_order.to_vec()),
                                next_branch_index: 0,
                            });
                        }
                        for action in current_ring_actions.iter().rev() {
                            successor.action_stack.push(action.clone());
                        }
                        push_literal_token(&mut successor.prefix, &atom_token);
                        push_successor_bucket(&mut successors, atom_token, successor);
                        Ok(())
                    })();
                    if let Err(err) = inner {
                        status = Err(err);
                    }
                });

                Ok(())
            })();

            if let Err(err) = outcome {
                status = Err(err);
            }
        });
    });

    status?;
    Ok(finalize_linear_structural_transitions(successors))
}

fn enter_atom_successors_without_bond_stereo_exact(
    input: ExactAtomStereoExpansionInput<'_>,
) -> PyResult<Vec<RootedConnectedStereoExactStateData>> {
    let ExactAtomStereoExpansionInput {
        graph,
        state,
        base_action_stack,
        atom_idx,
        parent_idx,
        visited_now,
        visited_count_now,
        pending_base,
        closures_here,
        ordered_groups,
        is_chiral_atom,
    } = input;
    let mut successors = Vec::<RootedConnectedStereoExactStateData>::new();
    let mut status = Ok(());

    for_each_cartesian_choice(&ordered_groups, &mut |chosen_children| {
        if status.is_err() {
            return;
        }

        let total_group_members = ordered_groups
            .iter()
            .map(|group| group.len())
            .sum::<usize>();
        let opening_target_count = total_group_members.saturating_sub(chosen_children.len());
        let mut ring_actions = Vec::with_capacity(closures_here.len() + opening_target_count);
        for closure_idx in 0..closures_here.len() {
            ring_actions.push(RingAction::Close(closure_idx));
        }
        for group in &ordered_groups {
            for &target_idx in group {
                if !chosen_children.contains(&target_idx) {
                    ring_actions.push(RingAction::Open(target_idx));
                }
            }
        }
        if ring_actions.len() <= 1 && chosen_children.len() <= 1 {
            let direct: PyResult<()> = (|| {
                let mut current_pending: Option<Vec<(usize, Vec<PendingRing>)>> = None;
                let mut current_free = Arc::unwrap_or_clone(state.dynamic.free_labels.clone());
                let mut current_next = state.dynamic.next_label;
                let mut current_ring_actions = Vec::<ExactWalkerAction>::with_capacity(
                    closures_here.len() * 2 + opening_target_count,
                );
                let mut ring_neighbor_order = is_chiral_atom.then(Vec::<usize>::new);
                if let Some(ring_action) = ring_actions.first() {
                    match *ring_action {
                        RingAction::Close(closure_idx) => {
                            let closure = &closures_here[closure_idx];
                            let bond_token = graph
                                .bond_token(atom_idx, closure.other_atom_idx)
                                .ok_or_else(|| {
                                    PyKeyError::new_err(format!(
                                        "No bond between atoms {atom_idx} and {}",
                                        closure.other_atom_idx
                                    ))
                                })?
                                .to_owned();
                            if let Some(token) = ExactBondToken::from_owned(bond_token) {
                                current_ring_actions.push(ExactWalkerAction::EmitBondToken(token));
                            }
                            current_ring_actions
                                .push(ExactWalkerAction::EmitRingLabel(closure.label));
                            insert_sorted(&mut current_free, closure.label);
                            if let Some(order) = &mut ring_neighbor_order {
                                order.push(closure.other_atom_idx);
                            }
                        }
                        RingAction::Open(target_idx) => {
                            let label = allocate_label(&mut current_free, &mut current_next);
                            current_ring_actions.push(ExactWalkerAction::EmitRingLabel(label));
                            add_pending(
                                current_pending.get_or_insert_with(|| {
                                    Arc::unwrap_or_clone(pending_base.clone())
                                }),
                                target_idx,
                                PendingRing {
                                    label,
                                    other_atom_idx: atom_idx,
                                },
                            );
                            if let Some(order) = &mut ring_neighbor_order {
                                order.push(target_idx);
                            }
                        }
                    }
                }
                let child_order = chosen_children;
                let atom_token = if !is_chiral_atom {
                    Cow::Borrowed(graph.atom_tokens[atom_idx].as_str())
                } else {
                    let emitted_neighbor_order = stereo_neighbor_order(
                        graph,
                        atom_idx,
                        parent_idx,
                        ring_neighbor_order.as_deref().unwrap_or(&[]),
                        child_order,
                    )?;
                    Cow::Owned(stereo_atom_token(graph, atom_idx, &emitted_neighbor_order)?)
                };
                let mut successor = RootedConnectedStereoExactStateData {
                    prefix: if !is_chiral_atom {
                        let mut prefix = state.prefix.clone();
                        push_literal_token(&mut prefix, graph.atom_tokens[atom_idx].as_str());
                        prefix
                    } else {
                        state.prefix.clone()
                    },
                    dynamic: Arc::new(RootedConnectedStereoExactDynamicData {
                        visited: visited_now.clone(),
                        visited_count: visited_count_now,
                        pending: current_pending
                            .map(Arc::new)
                            .unwrap_or_else(|| pending_base.clone()),
                        free_labels: Arc::new(current_free),
                        next_label: current_next,
                    }),
                    action_stack: {
                        let mut stack = Vec::with_capacity(
                            base_action_stack.len()
                                + current_ring_actions.len()
                                + usize::from(!child_order.is_empty()),
                        );
                        stack.extend_from_slice(base_action_stack);
                        stack
                    },
                };
                if let Some(&child_idx) = child_order.first() {
                    push_single_exact_child_action(
                        graph,
                        &mut successor.action_stack,
                        atom_idx,
                        child_idx,
                    )?;
                }
                for action in current_ring_actions.iter().rev() {
                    successor.action_stack.push(action.clone());
                }
                if is_chiral_atom {
                    push_literal_token(&mut successor.prefix, atom_token.as_ref());
                }
                successors.push(successor);
                Ok(())
            })();
            if let Err(err) = direct {
                status = Err(err);
            }
            return;
        }
        successors.reserve(
            small_permutation_count(ring_actions.len())
                .saturating_mul(small_permutation_count(chosen_children.len())),
        );

        permutations_copy_distinct(&ring_actions, &mut |ring_action_order| {
            if status.is_err() {
                return;
            }

            let outcome: PyResult<()> = (|| {
                let mut current_pending: Option<Vec<(usize, Vec<PendingRing>)>> = None;
                let mut current_free = Arc::unwrap_or_clone(state.dynamic.free_labels.clone());
                let mut current_next = state.dynamic.next_label;
                let mut current_ring_actions = Vec::<ExactWalkerAction>::with_capacity(
                    closures_here.len() * 2 + opening_target_count,
                );
                let mut labels_freed_after_atom = Vec::<usize>::with_capacity(closures_here.len());
                let mut ring_neighbor_order = is_chiral_atom.then(Vec::<usize>::new);

                for ring_action in ring_action_order {
                    match *ring_action {
                        RingAction::Close(closure_idx) => {
                            let closure = &closures_here[closure_idx];
                            let bond_token = graph
                                .bond_token(atom_idx, closure.other_atom_idx)
                                .ok_or_else(|| {
                                    PyKeyError::new_err(format!(
                                        "No bond between atoms {atom_idx} and {}",
                                        closure.other_atom_idx
                                    ))
                                })?
                                .to_owned();
                            if let Some(token) = ExactBondToken::from_owned(bond_token) {
                                current_ring_actions.push(ExactWalkerAction::EmitBondToken(token));
                            }
                            current_ring_actions
                                .push(ExactWalkerAction::EmitRingLabel(closure.label));
                            labels_freed_after_atom.push(closure.label);
                            if let Some(order) = &mut ring_neighbor_order {
                                order.push(closure.other_atom_idx);
                            }
                        }
                        RingAction::Open(target_idx) => {
                            let label = allocate_label(&mut current_free, &mut current_next);
                            current_ring_actions.push(ExactWalkerAction::EmitRingLabel(label));
                            add_pending(
                                current_pending.get_or_insert_with(|| {
                                    Arc::unwrap_or_clone(pending_base.clone())
                                }),
                                target_idx,
                                PendingRing {
                                    label,
                                    other_atom_idx: atom_idx,
                                },
                            );
                            if let Some(order) = &mut ring_neighbor_order {
                                order.push(target_idx);
                            }
                        }
                    }
                }

                for label in labels_freed_after_atom {
                    insert_sorted(&mut current_free, label);
                }
                let free_labels_shared = Arc::new(current_free.clone());
                let dynamic_shared = Arc::new(RootedConnectedStereoExactDynamicData {
                    visited: visited_now.clone(),
                    visited_count: visited_count_now,
                    pending: current_pending
                        .as_ref()
                        .map(|pending| Arc::new(pending.clone()))
                        .unwrap_or_else(|| pending_base.clone()),
                    free_labels: free_labels_shared.clone(),
                    next_label: current_next,
                });
                let nonchiral_prefix = (!is_chiral_atom).then(|| {
                    let mut prefix = state.prefix.clone();
                    push_literal_token(&mut prefix, graph.atom_tokens[atom_idx].as_str());
                    prefix
                });
                if chosen_children.len() <= 1 {
                    let child_order = chosen_children;
                    let atom_token = if !is_chiral_atom {
                        Cow::Borrowed(graph.atom_tokens[atom_idx].as_str())
                    } else {
                        let emitted_neighbor_order = stereo_neighbor_order(
                            graph,
                            atom_idx,
                            parent_idx,
                            ring_neighbor_order.as_deref().unwrap_or(&[]),
                            child_order,
                        )?;
                        Cow::Owned(stereo_atom_token(graph, atom_idx, &emitted_neighbor_order)?)
                    };
                    let mut successor = RootedConnectedStereoExactStateData {
                        prefix: nonchiral_prefix
                            .as_ref()
                            .cloned()
                            .unwrap_or_else(|| state.prefix.clone()),
                        dynamic: dynamic_shared,
                        action_stack: {
                            let mut stack = Vec::with_capacity(
                                base_action_stack.len()
                                    + current_ring_actions.len()
                                    + usize::from(!child_order.is_empty()),
                            );
                            stack.extend_from_slice(base_action_stack);
                            stack
                        },
                    };
                    if let Some(&child_idx) = child_order.first() {
                        push_single_exact_child_action(
                            graph,
                            &mut successor.action_stack,
                            atom_idx,
                            child_idx,
                        )?;
                    }
                    for action in current_ring_actions.iter().rev() {
                        successor.action_stack.push(action.clone());
                    }
                    if nonchiral_prefix.is_none() {
                        push_literal_token(&mut successor.prefix, atom_token.as_ref());
                    }
                    successors.push(successor);
                    return Ok(());
                }
                permutations_copy_distinct(chosen_children, &mut |child_order| {
                    if status.is_err() {
                        return;
                    }

                    let inner: PyResult<()> = (|| {
                        let atom_token = if !is_chiral_atom {
                            Cow::Borrowed(graph.atom_tokens[atom_idx].as_str())
                        } else {
                            let emitted_neighbor_order = stereo_neighbor_order(
                                graph,
                                atom_idx,
                                parent_idx,
                                ring_neighbor_order.as_deref().unwrap_or(&[]),
                                child_order,
                            )?;
                            Cow::Owned(stereo_atom_token(graph, atom_idx, &emitted_neighbor_order)?)
                        };
                        let mut successor = RootedConnectedStereoExactStateData {
                            prefix: nonchiral_prefix
                                .as_ref()
                                .cloned()
                                .unwrap_or_else(|| state.prefix.clone()),
                            dynamic: dynamic_shared.clone(),
                            action_stack: {
                                let mut stack = Vec::with_capacity(
                                    base_action_stack.len()
                                        + current_ring_actions.len()
                                        + usize::from(!child_order.is_empty()),
                                );
                                stack.extend_from_slice(base_action_stack);
                                stack
                            },
                        };
                        if child_order.len() == 1 {
                            push_single_exact_child_action(
                                graph,
                                &mut successor.action_stack,
                                atom_idx,
                                child_order[0],
                            )?;
                        } else if !child_order.is_empty() {
                            successor
                                .action_stack
                                .push(ExactWalkerAction::ProcessChildren {
                                    parent_idx: atom_idx,
                                    child_order: Arc::<[usize]>::from(child_order.to_vec()),
                                    next_branch_index: 0,
                                });
                        }
                        for action in current_ring_actions.iter().rev() {
                            successor.action_stack.push(action.clone());
                        }
                        if nonchiral_prefix.is_none() {
                            push_literal_token(&mut successor.prefix, atom_token.as_ref());
                        }
                        successors.push(successor);
                        Ok(())
                    })();
                    if let Err(err) = inner {
                        status = Err(err);
                    }
                });

                Ok(())
            })();

            if let Err(err) = outcome {
                status = Err(err);
            }
        });
    });

    status?;
    Ok(finalize_exact_atom_stereo_successors(successors))
}

fn process_children_successors_by_token(
    context: &mut StereoProcessChildrenContext<'_>,
    state: &RootedConnectedStereoWalkerStateData,
    parent_idx: usize,
    child_order: Arc<[usize]>,
    next_branch_index: usize,
) -> PyResult<BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>> {
    if child_order.is_empty() {
        return Ok(BTreeMap::new());
    }
    if context.runtime.side_infos.is_empty() {
        return process_children_successors_without_bond_stereo(
            context,
            state,
            parent_idx,
            child_order,
            next_branch_index,
        );
    }
    let branch_count = child_order.len().saturating_sub(1);

    if next_branch_index < branch_count {
        let child_idx = child_order[next_branch_index];
        let mut successor = state.clone();
        successor.action_stack.pop();
        push_char_token(&mut successor.prefix, '(');
        let ProcessChildrenEdgeUpdate {
            edge_part,
            selected_neighbors,
            selected_orientations,
            first_emitted_candidates,
            deferred_carrier_choice_constraints,
            token_basis_facts,
            component_phases,
            component_begin_atoms,
        } = process_children_edge_update(
            context.runtime,
            context.graph,
            state,
            parent_idx,
            child_order.as_ref(),
            child_idx,
        )?;
        successor.stereo_selected_neighbors = Arc::new(selected_neighbors);
        successor.stereo_selected_orientations = Arc::new(selected_orientations);
        successor.stereo_first_emitted_candidates = Arc::new(first_emitted_candidates);
        successor.stereo_component_phases = Arc::new(component_phases);
        successor.stereo_component_begin_atoms = Arc::new(component_begin_atoms);
        successor.deferred_carrier_choice_constraints =
            Arc::new(deferred_carrier_choice_constraints);
        successor.stereo_token_basis_facts = Arc::new(token_basis_facts);
        if let Part::Literal(token) = &edge_part {
            let role = directional_token_role(successor.prefix.as_ref(), None);
            record_rdkit_literal_edge_marker_trace(
                context.runtime,
                context.graph,
                &mut successor,
                parent_idx as isize,
                child_idx as isize,
                token.as_str(),
                role,
            );
        }
        push_process_children_branch_actions(
            &mut successor.action_stack,
            parent_idx,
            child_order.clone(),
            next_branch_index,
            child_idx,
            part_to_action(edge_part),
        );
        assert_component_token_flip_boundary_invariants(
            context.runtime,
            context.graph,
            &successor,
        )?;
        return Ok(BTreeMap::from([("(".to_owned(), vec![successor])]));
    }

    let child_idx = child_order[child_order.len() - 1];
    let ProcessChildrenEdgeUpdate {
        edge_part,
        selected_neighbors,
        selected_orientations,
        first_emitted_candidates,
        deferred_carrier_choice_constraints,
        token_basis_facts,
        component_phases,
        component_begin_atoms,
    } = process_children_edge_update(
        context.runtime,
        context.graph,
        state,
        parent_idx,
        child_order.as_ref(),
        child_idx,
    )?;
    let mut base_state = state.clone();
    base_state.action_stack.pop();
    base_state.stereo_selected_neighbors = Arc::new(selected_neighbors);
    base_state.stereo_selected_orientations = Arc::new(selected_orientations);
    base_state.stereo_first_emitted_candidates = Arc::new(first_emitted_candidates);
    base_state.stereo_component_phases = Arc::new(component_phases);
    base_state.stereo_component_begin_atoms = Arc::new(component_begin_atoms);
    base_state.deferred_carrier_choice_constraints = Arc::new(deferred_carrier_choice_constraints);
    base_state.stereo_token_basis_facts = Arc::new(token_basis_facts);
    assert_component_token_flip_boundary_invariants(context.runtime, context.graph, &base_state)?;
    process_children_terminal_successors(
        context,
        base_state,
        ProcessChildrenTerminalStep {
            parent_idx,
            child_idx,
            edge_part,
        },
    )
}

fn process_children_successors_without_bond_stereo(
    context: &mut StereoProcessChildrenContext<'_>,
    state: &RootedConnectedStereoWalkerStateData,
    parent_idx: usize,
    child_order: Arc<[usize]>,
    next_branch_index: usize,
) -> PyResult<BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>> {
    let branch_count = child_order.len().saturating_sub(1);

    if next_branch_index < branch_count {
        let child_idx = child_order[next_branch_index];
        let bond_token = context
            .graph
            .bond_token(parent_idx, child_idx)
            .ok_or_else(|| {
                PyKeyError::new_err(format!(
                    "No bond between atoms {parent_idx} and {child_idx}"
                ))
            })?
            .to_owned();
        let mut base_action_stack = state.action_stack.clone();
        base_action_stack.pop();
        let mut successor = RootedConnectedStereoWalkerStateData {
            prefix: state.prefix.clone(),
            visited: state.visited.clone(),
            visited_count: state.visited_count,
            pending: state.pending.clone(),
            free_labels: state.free_labels.clone(),
            next_label: state.next_label,
            stereo_component_phases: state.stereo_component_phases.clone(),
            stereo_selected_neighbors: state.stereo_selected_neighbors.clone(),
            stereo_selected_orientations: state.stereo_selected_orientations.clone(),
            stereo_first_emitted_candidates: state.stereo_first_emitted_candidates.clone(),
            stereo_component_begin_atoms: state.stereo_component_begin_atoms.clone(),
            committed_component_token_flips: state.committed_component_token_flips.clone(),
            deferred_carrier_choice_constraints: state.deferred_carrier_choice_constraints.clone(),
            stereo_token_basis_facts: state.stereo_token_basis_facts.clone(),
            directional_marker_traces: state.directional_marker_traces.clone(),
            marker_event_traces: state.marker_event_traces.clone(),
            action_stack: {
                let extra = 2
                    + usize::from(next_branch_index + 1 < child_order.len())
                    + usize::from(!bond_token.is_empty());
                let mut stack = Vec::with_capacity(base_action_stack.len() + extra);
                stack.extend_from_slice(&base_action_stack);
                stack
            },
        };
        push_char_token(&mut successor.prefix, '(');
        push_process_children_branch_actions(
            &mut successor.action_stack,
            parent_idx,
            child_order.clone(),
            next_branch_index,
            child_idx,
            (!bond_token.is_empty()).then_some(WalkerAction::EmitLiteral(bond_token)),
        );
        return Ok(BTreeMap::from([("(".to_owned(), vec![successor])]));
    }

    let child_idx = child_order[child_order.len() - 1];
    let bond_token = context
        .graph
        .bond_token(parent_idx, child_idx)
        .ok_or_else(|| {
            PyKeyError::new_err(format!(
                "No bond between atoms {parent_idx} and {child_idx}"
            ))
        })?
        .to_owned();
    let mut base_state = state.clone();
    base_state.action_stack.pop();
    process_children_terminal_successors(
        context,
        base_state,
        ProcessChildrenTerminalStep {
            parent_idx,
            child_idx,
            edge_part: Part::Literal(bond_token),
        },
    )
}

fn process_children_successors_without_bond_stereo_exact(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoExactStateData,
    parent_idx: usize,
    child_order: Arc<[usize]>,
    next_branch_index: usize,
) -> PyResult<Vec<RootedConnectedStereoExactStateData>> {
    let branch_count = child_order.len().saturating_sub(1);
    let base_action_stack = &state.action_stack[..state.action_stack.len() - 1];

    if next_branch_index < branch_count {
        let child_idx = child_order[next_branch_index];
        let bond_token = graph
            .bond_token(parent_idx, child_idx)
            .ok_or_else(|| {
                PyKeyError::new_err(format!(
                    "No bond between atoms {parent_idx} and {child_idx}"
                ))
            })?
            .to_owned();
        let mut successor = RootedConnectedStereoExactStateData {
            prefix: state.prefix.clone(),
            dynamic: state.dynamic.clone(),
            action_stack: {
                let extra = 2
                    + usize::from(next_branch_index + 1 < child_order.len())
                    + usize::from(!bond_token.is_empty());
                let mut stack = Vec::with_capacity(base_action_stack.len() + extra);
                stack.extend_from_slice(base_action_stack);
                stack
            },
        };
        push_char_token(&mut successor.prefix, '(');
        if next_branch_index + 1 < child_order.len() {
            successor
                .action_stack
                .push(ExactWalkerAction::ProcessChildren {
                    parent_idx,
                    child_order: child_order.clone(),
                    next_branch_index: next_branch_index + 1,
                });
        }
        successor
            .action_stack
            .push(ExactWalkerAction::EmitCloseParen);
        successor.action_stack.push(ExactWalkerAction::EnterAtom {
            atom_idx: child_idx,
            parent_idx: Some(parent_idx),
        });
        if let Some(token) = ExactBondToken::from_owned(bond_token) {
            successor
                .action_stack
                .push(ExactWalkerAction::EmitBondToken(token));
        }
        return Ok(vec![successor]);
    }

    let child_idx = child_order[child_order.len() - 1];
    let bond_token = graph
        .bond_token(parent_idx, child_idx)
        .ok_or_else(|| {
            PyKeyError::new_err(format!(
                "No bond between atoms {parent_idx} and {child_idx}"
            ))
        })?
        .to_owned();

    if bond_token.is_empty() {
        let mut base_state = RootedConnectedStereoExactStateData {
            prefix: state.prefix.clone(),
            dynamic: state.dynamic.clone(),
            action_stack: {
                let mut stack = Vec::with_capacity(base_action_stack.len() + 1);
                stack.extend_from_slice(base_action_stack);
                stack
            },
        };
        base_state.action_stack.push(ExactWalkerAction::EnterAtom {
            atom_idx: child_idx,
            parent_idx: Some(parent_idx),
        });
        return exact_successors_from_atom_stereo_state(runtime, graph, base_state);
    }

    let mut base_state = RootedConnectedStereoExactStateData {
        prefix: state.prefix.clone(),
        dynamic: state.dynamic.clone(),
        action_stack: {
            let mut stack = Vec::with_capacity(base_action_stack.len() + 1);
            stack.extend_from_slice(base_action_stack);
            stack
        },
    };
    push_literal_token(&mut base_state.prefix, &bond_token);
    base_state.action_stack.push(ExactWalkerAction::EnterAtom {
        atom_idx: child_idx,
        parent_idx: Some(parent_idx),
    });
    Ok(vec![base_state])
}

fn push_single_exact_child_action(
    graph: &PreparedSmilesGraphData,
    stack: &mut Vec<ExactWalkerAction>,
    parent_idx: usize,
    child_idx: usize,
) -> PyResult<()> {
    let bond_token = graph
        .bond_token(parent_idx, child_idx)
        .ok_or_else(|| {
            PyKeyError::new_err(format!(
                "No bond between atoms {parent_idx} and {child_idx}"
            ))
        })?
        .to_owned();
    stack.push(ExactWalkerAction::EnterAtom {
        atom_idx: child_idx,
        parent_idx: Some(parent_idx),
    });
    if let Some(token) = ExactBondToken::from_owned(bond_token) {
        stack.push(ExactWalkerAction::EmitBondToken(token));
    }
    Ok(())
}

fn can_complete_from_stereo_state_memo(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    cache: &mut StereoCompletionCache,
) -> bool {
    match state.action_stack.last() {
        None => return is_complete_terminal_stereo_state(graph, state),
        Some(WalkerAction::EmitLiteral(_))
        | Some(WalkerAction::EmitRingLabel(_))
        | Some(WalkerAction::EmitCloseParen) => {
            let mut successor = state.clone();
            successor.action_stack.pop();
            return can_complete_from_stereo_state_memo(runtime, graph, &successor, cache);
        }
        _ => {}
    }

    let key = StereoCompletionKey::from(state);
    if let Some(&cached) = cache.get(&key) {
        return cached;
    }

    let successors = match stereo_next_token_successors_from_boundary(
        runtime,
        graph,
        state,
        StereoNextTokenCompleteness::Raw,
        cache,
    ) {
        Ok(successors) => successors,
        Err(_) => {
            cache.insert(key, false);
            return false;
        }
    };
    let result = if successors.is_empty() {
        is_complete_terminal_stereo_state(graph, state)
    } else {
        successors.into_values().any(|successor_group| {
            successor_group.into_iter().any(|successor| {
                can_complete_from_stereo_state_memo(runtime, graph, &successor, cache)
            })
        })
    };
    cache.insert(key, result);
    result
}

fn filter_complete_successors(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    successors: BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>,
    completion_cache: &mut StereoCompletionCache,
) -> BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>> {
    let mut filtered = BTreeMap::new();
    for (token, successor_group) in successors {
        let kept = successor_group
            .into_iter()
            .filter(|successor| {
                can_complete_from_stereo_state_memo(runtime, graph, successor, completion_cache)
            })
            .collect::<Vec<_>>();
        if !kept.is_empty() {
            filtered.insert(token, kept);
        }
    }
    filtered
}

#[cfg(test)]
fn next_token_support_for_stereo_state_impl(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    completion_cache: &mut StereoCompletionCache,
) -> PyResult<Vec<String>> {
    Ok(stereo_next_token_successors_from_boundary(
        runtime,
        graph,
        state,
        StereoNextTokenCompleteness::Completable,
        completion_cache,
    )?
    .into_keys()
    .collect())
}

#[cfg(test)]
fn next_token_support_for_stereo_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<Vec<String>> {
    let mut completion_cache = FxHashMap::default();
    next_token_support_for_stereo_state_impl(runtime, graph, state, &mut completion_cache)
}

/// Single internal boundary for online stereo support queries.
///
/// The caller supplies the current walker prefix/state and chooses whether to
/// filter to completable continuations. Row filtering, propagation, and
/// RDKit-writer policy details stay behind this boundary.
fn stereo_next_token_successors_from_boundary(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    completeness: StereoNextTokenCompleteness,
    completion_cache: &mut StereoCompletionCache,
) -> PyResult<BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>> {
    let require_completable = completeness == StereoNextTokenCompleteness::Completable;
    let action = match state.action_stack.last() {
        Some(action) => action.clone(),
        None => return Ok(BTreeMap::new()),
    };
    let raw_successors = match match action {
        WalkerAction::EmitLiteral(token) => {
            let mut successor = state.clone();
            successor.action_stack.pop();
            push_literal_token(&mut successor.prefix, &token);
            Ok(BTreeMap::from([(token, vec![successor])]))
        }
        WalkerAction::EmitRingLabel(label) => {
            let mut successor = state.clone();
            successor.action_stack.pop();
            push_ring_label(&mut successor.prefix, label);
            let token = ring_label_text(label);
            Ok(BTreeMap::from([(token, vec![successor])]))
        }
        WalkerAction::EmitCloseParen => {
            let mut successor = state.clone();
            successor.action_stack.pop();
            push_char_token(&mut successor.prefix, ')');
            Ok(BTreeMap::from([(")".to_owned(), vec![successor])]))
        }
        WalkerAction::EmitDeferred(deferred) => {
            let mut out = BTreeMap::<String, Vec<RootedConnectedStereoWalkerStateData>>::new();
            for token in deferred_token_support(runtime, graph, state, &deferred)? {
                let mut successor = state.clone();
                successor.action_stack.pop();
                if let Err(err) =
                    commit_deferred_token_choice(runtime, graph, &mut successor, &deferred, &token)
                {
                    if require_completable {
                        return Err(err);
                    }
                    continue;
                }
                let role = directional_token_role(
                    successor.prefix.as_ref(),
                    successor.action_stack.last(),
                );
                record_rdkit_directional_marker_trace(
                    runtime,
                    graph,
                    &mut successor,
                    deferred.begin_idx,
                    deferred.end_idx,
                    &token,
                    role,
                );
                push_literal_token(&mut successor.prefix, &token);
                out.entry(token).or_default().push(successor);
            }
            Ok(out)
        }
        WalkerAction::EnterAtom {
            atom_idx,
            parent_idx,
        } => enter_atom_successors_by_token(runtime, graph, state, atom_idx, parent_idx),
        WalkerAction::ProcessChildren {
            parent_idx,
            child_order,
            next_branch_index,
        } => process_children_successors_by_token(
            &mut StereoProcessChildrenContext {
                runtime,
                graph,
                require_completable,
                completion_cache,
            },
            state,
            parent_idx,
            child_order,
            next_branch_index,
        ),
    } {
        Ok(successors) => successors,
        Err(err) => {
            if require_completable {
                return Err(err);
            }
            return Ok(BTreeMap::new());
        }
    };
    let mut expanded = Vec::<(String, Vec<RootedConnectedStereoWalkerStateData>)>::new();
    for (token, successor_group) in raw_successors {
        if token.is_empty() {
            for successor in successor_group {
                let nested = stereo_next_token_successors_from_boundary(
                    runtime,
                    graph,
                    &successor,
                    completeness,
                    completion_cache,
                );
                match nested {
                    Ok(successors) => {
                        extend_linear_structural_transitions(&mut expanded, successors)
                    }
                    Err(err) => {
                        if require_completable {
                            return Err(err);
                        }
                    }
                }
            }
            continue;
        }
        if let Some((_, states)) = expanded
            .iter_mut()
            .find(|(existing_token, _)| *existing_token == token)
        {
            states.extend(successor_group);
        } else {
            expanded.push((token, successor_group));
        }
    }
    let successors = finalize_linear_structural_transitions(expanded);
    #[cfg(debug_assertions)]
    validate_stereo_successors_against_constraint_model(runtime, &successors)?;
    if require_completable && runtime.side_infos.is_empty() {
        Ok(successors)
    } else if require_completable {
        Ok(filter_complete_successors(
            runtime,
            graph,
            successors,
            completion_cache,
        ))
    } else {
        Ok(successors)
    }
}

fn successors_by_token_stereo(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>> {
    let mut completion_cache = FxHashMap::default();
    stereo_next_token_successors_from_boundary(
        runtime,
        graph,
        state,
        StereoNextTokenCompleteness::Completable,
        &mut completion_cache,
    )
}

fn successors_by_token_stereo_raw(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>> {
    let mut completion_cache = FxHashMap::default();
    stereo_next_token_successors_from_boundary(
        runtime,
        graph,
        state,
        StereoNextTokenCompleteness::Raw,
        &mut completion_cache,
    )
}

fn exact_successors_from_atom_stereo_state_drained(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: RootedConnectedStereoExactStateData,
) -> PyResult<Vec<RootedConnectedStereoExactStateData>> {
    debug_assert!(runtime.side_infos.is_empty());
    let action = match state.action_stack.last() {
        Some(action) => action,
        None => return Ok(Vec::new()),
    };

    match action {
        ExactWalkerAction::EnterAtom {
            atom_idx,
            parent_idx,
        } => {
            let base_action_stack = &state.action_stack[..state.action_stack.len() - 1];
            let atom_idx = *atom_idx;
            let parent_idx = *parent_idx;
            let visited_now = visited_with_marked(&state.dynamic.visited, atom_idx);
            let visited_count_now = state.dynamic.visited_count + 1;
            let TakenPendingRings {
                pending: pending_after_closures,
                rings: closures_here,
            } = take_pending_for_atom_arc(&state.dynamic.pending, atom_idx);
            let ordered_groups = ordered_neighbor_groups(graph, atom_idx, visited_now.as_ref());
            let is_chiral_atom = graph.atom_chiral_tags[atom_idx] != "CHI_UNSPECIFIED";

            if closures_here.is_empty() && ordered_groups.iter().all(|group| group.len() == 1) {
                let chosen_children = ordered_groups
                    .iter()
                    .map(|group| group[0])
                    .collect::<Vec<_>>();
                let dynamic_shared = Arc::new(RootedConnectedStereoExactDynamicData {
                    visited: visited_now.clone(),
                    visited_count: visited_count_now,
                    pending: pending_after_closures.clone(),
                    free_labels: state.dynamic.free_labels.clone(),
                    next_label: state.dynamic.next_label,
                });
                let nonchiral_prefix = (!is_chiral_atom).then(|| {
                    let mut prefix = state.prefix.clone();
                    push_literal_token(&mut prefix, graph.atom_tokens[atom_idx].as_str());
                    prefix
                });
                if chosen_children.len() <= 1 {
                    let child_order = chosen_children.as_slice();
                    let atom_token = if !is_chiral_atom {
                        Cow::Borrowed(graph.atom_tokens[atom_idx].as_str())
                    } else {
                        let emitted_neighbor_order =
                            stereo_neighbor_order(graph, atom_idx, parent_idx, &[], child_order)?;
                        Cow::Owned(stereo_atom_token(graph, atom_idx, &emitted_neighbor_order)?)
                    };
                    let mut successor = RootedConnectedStereoExactStateData {
                        prefix: nonchiral_prefix
                            .as_ref()
                            .cloned()
                            .unwrap_or_else(|| state.prefix.clone()),
                        dynamic: dynamic_shared,
                        action_stack: {
                            let mut stack = Vec::with_capacity(
                                base_action_stack.len() + usize::from(!child_order.is_empty()),
                            );
                            stack.extend_from_slice(base_action_stack);
                            stack
                        },
                    };
                    if let Some(&child_idx) = child_order.first() {
                        push_single_exact_child_action(
                            graph,
                            &mut successor.action_stack,
                            atom_idx,
                            child_idx,
                        )?;
                    }
                    if nonchiral_prefix.is_none() {
                        push_literal_token(&mut successor.prefix, atom_token.as_ref());
                    }
                    return Ok(vec![successor]);
                }
                let mut successors = Vec::<RootedConnectedStereoExactStateData>::with_capacity(
                    small_permutation_count(chosen_children.len()),
                );
                let mut status = Ok(());
                permutations_copy_distinct(&chosen_children, &mut |child_order| {
                    if status.is_err() {
                        return;
                    }
                    let outcome: PyResult<()> = (|| {
                        let atom_token = if !is_chiral_atom {
                            Cow::Borrowed(graph.atom_tokens[atom_idx].as_str())
                        } else {
                            let emitted_neighbor_order = stereo_neighbor_order(
                                graph,
                                atom_idx,
                                parent_idx,
                                &[],
                                child_order,
                            )?;
                            Cow::Owned(stereo_atom_token(graph, atom_idx, &emitted_neighbor_order)?)
                        };
                        let mut successor = RootedConnectedStereoExactStateData {
                            prefix: nonchiral_prefix
                                .as_ref()
                                .cloned()
                                .unwrap_or_else(|| state.prefix.clone()),
                            dynamic: dynamic_shared.clone(),
                            action_stack: {
                                let mut stack = Vec::with_capacity(
                                    base_action_stack.len() + usize::from(!child_order.is_empty()),
                                );
                                stack.extend_from_slice(base_action_stack);
                                stack
                            },
                        };
                        if child_order.len() == 1 {
                            push_single_exact_child_action(
                                graph,
                                &mut successor.action_stack,
                                atom_idx,
                                child_order[0],
                            )?;
                        } else if !child_order.is_empty() {
                            successor
                                .action_stack
                                .push(ExactWalkerAction::ProcessChildren {
                                    parent_idx: atom_idx,
                                    child_order: Arc::<[usize]>::from(child_order.to_vec()),
                                    next_branch_index: 0,
                                });
                        }
                        if nonchiral_prefix.is_none() {
                            push_literal_token(&mut successor.prefix, atom_token.as_ref());
                        }
                        push_exact_atom_stereo_successor(&mut successors, successor);
                        Ok(())
                    })();
                    if let Err(err) = outcome {
                        status = Err(err);
                    }
                });
                status?;
                return Ok(finalize_exact_atom_stereo_successors(successors));
            }

            enter_atom_successors_without_bond_stereo_exact(ExactAtomStereoExpansionInput {
                graph,
                state: &state,
                base_action_stack,
                atom_idx,
                parent_idx,
                visited_now,
                visited_count_now,
                pending_base: pending_after_closures,
                closures_here,
                ordered_groups,
                is_chiral_atom,
            })
        }
        ExactWalkerAction::ProcessChildren {
            parent_idx,
            child_order,
            next_branch_index,
        } => process_children_successors_without_bond_stereo_exact(
            runtime,
            graph,
            &state,
            *parent_idx,
            child_order.clone(),
            *next_branch_index,
        ),
        ExactWalkerAction::EmitBondToken(_)
        | ExactWalkerAction::EmitRingLabel(_)
        | ExactWalkerAction::EmitCloseParen => unreachable!(),
    }
}

fn exact_successors_from_atom_stereo_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    mut state: RootedConnectedStereoExactStateData,
) -> PyResult<Vec<RootedConnectedStereoExactStateData>> {
    drain_exact_linear_atom_stereo_actions(&mut state);
    exact_successors_from_atom_stereo_state_drained(runtime, graph, state)
}

#[cfg(test)]
fn advance_stereo_token_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    chosen_token: &str,
) -> PyResult<RootedConnectedStereoWalkerStateData> {
    let mut successors = successors_by_token_stereo(runtime, graph, state)?;
    let candidates = successors.remove(chosen_token).ok_or_else(|| {
        let available = successors.keys().cloned().collect::<Vec<_>>();
        PyKeyError::new_err(format!(
            "Token {chosen_token:?} is not available; choices={available:?}"
        ))
    })?;
    take_first_stereo_successor_state(candidates, "token advance")
}

#[cfg(test)]
fn choices_for_stereo_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<Vec<DecoderChoice<RootedConnectedStereoWalkerStateData>>> {
    let mut choices = Vec::new();
    for (token, successors) in successors_by_token_stereo(runtime, graph, state)? {
        for successor in successors {
            choices.push(DecoderChoice {
                text: token.clone(),
                next_frontier: vec![successor],
            });
        }
    }
    Ok(choices)
}

#[cfg(test)]
fn advance_stereo_choice_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    chosen_idx: usize,
) -> PyResult<RootedConnectedStereoWalkerStateData> {
    let mut choices = choices_for_stereo_state(runtime, graph, state)?;
    take_only_stereo_successor_state(
        take_choice_or_err(&mut choices, chosen_idx)?,
        "choice advance",
    )
}

fn take_only_stereo_successor_state(
    mut successors: Vec<RootedConnectedStereoWalkerStateData>,
    context: &str,
) -> PyResult<RootedConnectedStereoWalkerStateData> {
    if successors.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "Expected exactly one stereo successor state for {context}, got {}",
            successors.len()
        )));
    }
    match successors.pop() {
        Some(successor) => Ok(successor),
        None => Err(PyValueError::new_err(format!(
            "Expected exactly one stereo successor state for {context}, got 0"
        ))),
    }
}

fn take_first_stereo_successor_state(
    mut successors: Vec<RootedConnectedStereoWalkerStateData>,
    context: &str,
) -> PyResult<RootedConnectedStereoWalkerStateData> {
    match successors.drain(..).next() {
        Some(successor) => Ok(successor),
        None => Err(PyValueError::new_err(format!(
            "Expected at least one stereo successor state for {context}, got 0"
        ))),
    }
}

fn frontier_next_token_support_for_stereo(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    frontier: &[RootedConnectedStereoWalkerStateData],
) -> PyResult<Vec<String>> {
    Ok(frontier_transitions_for_stereo(runtime, graph, frontier)?
        .into_keys()
        .collect())
}

fn frontier_transitions_for_stereo(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    frontier: &[RootedConnectedStereoWalkerStateData],
) -> PyResult<BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>> {
    let mut transitions = Vec::<(String, Vec<RootedConnectedStereoWalkerStateData>)>::new();
    for state in frontier {
        extend_linear_structural_transitions(
            &mut transitions,
            successors_by_token_stereo(runtime, graph, state)?,
        );
    }
    Ok(finalize_linear_structural_transitions(transitions))
}

fn frontier_transitions_for_stereo_linear(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    frontier: &[RootedConnectedStereoWalkerStateData],
) -> PyResult<Vec<(String, Vec<RootedConnectedStereoWalkerStateData>)>> {
    let mut transitions = Vec::<(String, Vec<RootedConnectedStereoWalkerStateData>)>::new();
    for state in frontier {
        extend_linear_structural_transitions(
            &mut transitions,
            successors_by_token_stereo(runtime, graph, state)?,
        );
    }
    Ok(finalize_linear_structural_transitions_vec(transitions))
}

fn frontier_choices_for_stereo(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    frontier: &[RootedConnectedStereoWalkerStateData],
) -> PyResult<Vec<DecoderChoice<RootedConnectedStereoWalkerStateData>>> {
    let mut choices = Vec::new();
    for state in frontier {
        for (token, successors) in successors_by_token_stereo(runtime, graph, state)? {
            for successor in successors {
                choices.push(DecoderChoice {
                    text: token.clone(),
                    next_frontier: vec![successor],
                });
            }
        }
    }
    Ok(choices)
}

fn frontier_choice_successors_for_stereo(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    frontier: &[RootedConnectedStereoWalkerStateData],
) -> PyResult<Vec<(String, RootedConnectedStereoWalkerStateData)>> {
    let mut choices = Vec::new();
    for state in frontier {
        for (token, successors) in successors_by_token_stereo(runtime, graph, state)? {
            for successor in successors {
                choices.push((token.clone(), successor));
            }
        }
    }
    Ok(choices)
}

fn stereo_frontier_prefix(frontier: &[RootedConnectedStereoWalkerStateData]) -> String {
    shared_frontier_prefix(frontier, |state| state.prefix.as_ref())
}

fn stereo_frontier_is_terminal(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    frontier: &[RootedConnectedStereoWalkerStateData],
) -> PyResult<bool> {
    Ok(frontier_next_token_support_for_stereo(runtime, graph, frontier)?.is_empty())
}

fn enumerate_support_from_stereo_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    mut state: RootedConnectedStereoWalkerStateData,
    out: &mut BTreeSet<String>,
) -> PyResult<()> {
    if runtime.side_infos.is_empty() {
        return enumerate_support_from_atom_stereo_exact_state(
            runtime,
            graph,
            stereo_exact_state_from_full(state),
            out,
        );
    }
    drain_exact_linear_stereo_actions(&mut state);
    if state.action_stack.is_empty() {
        if is_complete_terminal_stereo_state(graph, &state)
            && terminal_stereo_state_has_support_boundary_marker_placement(runtime, graph, &state)?
        {
            out.insert(state.prefix.to_string());
        }
        return Ok(());
    }
    let successors = flatten_exact_stereo_successor_groups(successors_by_token_stereo_raw(
        runtime, graph, &state,
    )?);

    for successor in successors {
        enumerate_support_from_stereo_state(runtime, graph, successor, out)?;
    }
    Ok(())
}

fn enumerate_support_from_atom_stereo_exact_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    mut state: RootedConnectedStereoExactStateData,
    out: &mut BTreeSet<String>,
) -> PyResult<()> {
    debug_assert!(runtime.side_infos.is_empty());
    drain_exact_linear_atom_stereo_actions(&mut state);
    if state.action_stack.is_empty() {
        if state.dynamic.visited_count == graph.atom_count() && state.dynamic.pending.is_empty() {
            out.insert(state.prefix.to_string());
        }
        return Ok(());
    }
    let successors = exact_successors_from_atom_stereo_state_drained(runtime, graph, state)?;

    for successor in successors {
        enumerate_support_from_atom_stereo_exact_state(runtime, graph, successor, out)?;
    }
    Ok(())
}

#[pyclass(
    name = "RootedConnectedStereoWalkerState",
    module = "grimace._core",
    frozen
)]
pub struct PyRootedConnectedStereoWalkerState {
    data: RootedConnectedStereoWalkerStateData,
}

#[pymethods]
impl PyRootedConnectedStereoWalkerState {
    #[getter]
    fn prefix(&self) -> String {
        self.data.prefix.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "RootedConnectedStereoWalkerState(prefix={:?}, visited_count={}, pending_sites={}, next_label={}, stack_depth={})",
            self.data.prefix,
            self.data.visited_count,
            self.data.pending.len(),
            self.data.next_label,
            self.data.action_stack.len(),
        )
    }
}

#[pyclass(name = "RootedConnectedStereoWalker", module = "grimace._core", frozen)]
pub struct PyRootedConnectedStereoWalker {
    graph: PreparedSmilesGraphData,
    runtime: StereoWalkerRuntimeData,
    root_idx: usize,
}

#[pymethods]
impl PyRootedConnectedStereoWalker {
    #[new]
    fn new(graph: &Bound<'_, PyAny>, root_idx: isize) -> PyResult<Self> {
        let graph = PreparedSmilesGraphData::from_any(graph)?;
        check_supported_stereo_writer_surface(&graph)?;
        let root_idx = validate_root_idx(&graph, root_idx)?;
        let runtime = build_walker_runtime(&graph, root_idx)?;
        Ok(Self {
            graph,
            runtime,
            root_idx,
        })
    }

    #[getter]
    fn root_idx(&self) -> usize {
        self.root_idx
    }

    fn initial_state(&self) -> PyRootedConnectedStereoWalkerState {
        PyRootedConnectedStereoWalkerState {
            data: initial_stereo_state_for_root(&self.runtime, &self.graph, self.root_idx),
        }
    }

    fn next_token_support(
        &self,
        state: &PyRootedConnectedStereoWalkerState,
    ) -> PyResult<Vec<String>> {
        validate_stereo_state_shape(&self.runtime, &self.graph, &state.data)?;
        Ok(
            successors_by_token_stereo(&self.runtime, &self.graph, &state.data)?
                .into_keys()
                .collect(),
        )
    }

    fn next_choice_texts(
        &self,
        state: &PyRootedConnectedStereoWalkerState,
    ) -> PyResult<Vec<String>> {
        validate_stereo_state_shape(&self.runtime, &self.graph, &state.data)?;
        let mut choices = Vec::new();
        for (token, successors) in
            successors_by_token_stereo(&self.runtime, &self.graph, &state.data)?
        {
            for successor in successors {
                choices.push(DecoderChoice {
                    text: token.clone(),
                    next_frontier: vec![successor],
                });
            }
        }
        Ok(choice_texts(&choices))
    }

    fn advance_token(
        &self,
        state: &PyRootedConnectedStereoWalkerState,
        chosen_token: &str,
    ) -> PyResult<PyRootedConnectedStereoWalkerState> {
        validate_stereo_state_shape(&self.runtime, &self.graph, &state.data)?;
        let mut choices = successors_by_token_stereo(&self.runtime, &self.graph, &state.data)?;
        let successors = take_first_stereo_successor_state(
            take_transition_or_err(&mut choices, chosen_token)?,
            "walker token advance",
        )?;
        Ok(PyRootedConnectedStereoWalkerState { data: successors })
    }

    fn advance_choice(
        &self,
        state: &PyRootedConnectedStereoWalkerState,
        chosen_idx: usize,
    ) -> PyResult<PyRootedConnectedStereoWalkerState> {
        validate_stereo_state_shape(&self.runtime, &self.graph, &state.data)?;
        let mut choices = Vec::new();
        for (token, successors) in
            successors_by_token_stereo(&self.runtime, &self.graph, &state.data)?
        {
            for successor in successors {
                choices.push(DecoderChoice {
                    text: token.clone(),
                    next_frontier: vec![successor],
                });
            }
        }
        Ok(PyRootedConnectedStereoWalkerState {
            data: take_only_stereo_successor_state(
                take_choice_or_err(&mut choices, chosen_idx)?,
                "walker choice advance",
            )?,
        })
    }

    fn is_terminal(&self, state: &PyRootedConnectedStereoWalkerState) -> PyResult<bool> {
        validate_stereo_state_shape(&self.runtime, &self.graph, &state.data)?;
        Ok(is_terminal_stereo_state(&state.data))
    }

    fn enumerate_support(&self) -> PyResult<Vec<String>> {
        enumerate_rooted_connected_stereo_smiles_support(&self.graph, self.root_idx as isize)
    }

    fn __repr__(&self) -> String {
        format!(
            "RootedConnectedStereoWalker(root_idx={}, atom_count={}, policy_name={:?}, policy_digest={:?})",
            self.root_idx,
            self.graph.atom_count(),
            self.graph.policy_name,
            self.graph.policy_digest,
        )
    }
}

#[pyclass(
    skip_from_py_object,
    name = "RootedConnectedStereoDecoder",
    module = "grimace._core"
)]
#[derive(Clone)]
pub struct PyRootedConnectedStereoDecoder {
    graph: Arc<PreparedSmilesGraphData>,
    mode: StereoDecoderMode,
}

impl PyRootedConnectedStereoDecoder {
    fn from_mode(graph: Arc<PreparedSmilesGraphData>, mode: StereoDecoderMode) -> Self {
        Self { graph, mode }
    }

    fn from_single(
        graph: Arc<PreparedSmilesGraphData>,
        runtime: Arc<StereoWalkerRuntimeData>,
        frontier: Vec<RootedConnectedStereoWalkerStateData>,
    ) -> Self {
        Self::from_mode(graph, StereoDecoderMode::single(runtime, frontier))
    }

    fn from_merged(
        graph: Arc<PreparedSmilesGraphData>,
        branches: Vec<StereoDecoderBranch>,
    ) -> Self {
        if let [branch] = branches.as_slice() {
            return Self::from_single(graph, branch.runtime.clone(), branch.frontier.clone());
        }
        Self::from_mode(graph, StereoDecoderMode::merged(branches))
    }
}

fn merged_stereo_prefix(branches: &[StereoDecoderBranch]) -> String {
    let prefix = branches
        .first()
        .map(|branch| stereo_frontier_prefix(&branch.frontier))
        .unwrap_or_default();
    debug_assert!(branches
        .iter()
        .all(|branch| stereo_frontier_prefix(&branch.frontier) == prefix));
    prefix
}

fn merged_stereo_cache_key(branches: &[StereoDecoderBranch]) -> String {
    let mut keys = branches
        .iter()
        .map(|branch| format!("{:?}", branch.frontier))
        .collect::<Vec<_>>();
    keys.sort();
    format!("{:?}", ("merged", keys))
}

fn merged_stereo_is_terminal(
    graph: &PreparedSmilesGraphData,
    branches: &[StereoDecoderBranch],
) -> PyResult<bool> {
    for branch in branches {
        if !stereo_frontier_is_terminal(branch.runtime.as_ref(), graph, &branch.frontier)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn merged_stereo_choice_successors(
    graph: Arc<PreparedSmilesGraphData>,
    branches: &[StereoDecoderBranch],
) -> PyResult<Vec<(String, PyRootedConnectedStereoDecoder)>> {
    let mut out = Vec::new();
    for branch in branches {
        if stereo_frontier_is_terminal(branch.runtime.as_ref(), graph.as_ref(), &branch.frontier)? {
            continue;
        }
        for (token, successor) in frontier_choice_successors_for_stereo(
            branch.runtime.as_ref(),
            graph.as_ref(),
            &branch.frontier,
        )? {
            out.push((
                token,
                PyRootedConnectedStereoDecoder::from_single(
                    graph.clone(),
                    branch.runtime.clone(),
                    vec![successor],
                ),
            ));
        }
    }
    Ok(out)
}

fn merged_stereo_grouped_successors(
    graph: Arc<PreparedSmilesGraphData>,
    branches: &[StereoDecoderBranch],
) -> PyResult<Vec<(String, PyRootedConnectedStereoDecoder)>> {
    let mut buckets = Vec::<(String, Vec<StereoDecoderBranch>)>::new();
    for branch in branches {
        for (token, frontier) in frontier_transitions_for_stereo_linear(
            branch.runtime.as_ref(),
            graph.as_ref(),
            &branch.frontier,
        )? {
            let successor_branch = StereoDecoderBranch {
                runtime: branch.runtime.clone(),
                frontier,
            };
            if let Some((_, grouped)) = buckets.iter_mut().find(|(existing, _)| *existing == token)
            {
                grouped.push(successor_branch);
            } else {
                buckets.push((token, vec![successor_branch]));
            }
        }
    }
    Ok(buckets
        .into_iter()
        .map(|(token, grouped)| {
            (
                token,
                PyRootedConnectedStereoDecoder::from_merged(graph.clone(), grouped),
            )
        })
        .collect())
}

#[pymethods]
impl PyRootedConnectedStereoDecoder {
    #[new]
    fn new(graph: &Bound<'_, PyAny>, root_idx: isize) -> PyResult<Self> {
        let graph = Arc::new(PreparedSmilesGraphData::from_any(graph)?);
        check_supported_stereo_writer_surface(graph.as_ref())?;
        if graph.atom_count() == 0 {
            let root_idx = validate_root_idx(graph.as_ref(), 0)?;
            let runtime = Arc::new(build_walker_runtime(graph.as_ref(), root_idx)?);
            return Ok(Self::from_single(
                graph.clone(),
                runtime.clone(),
                vec![initial_stereo_state_for_root(
                    runtime.as_ref(),
                    graph.as_ref(),
                    root_idx,
                )],
            ));
        }
        if root_idx < 0 {
            let mut branches = Vec::with_capacity(graph.atom_count());
            for atom_idx in 0..graph.atom_count() {
                let runtime = Arc::new(build_walker_runtime(graph.as_ref(), atom_idx)?);
                branches.push(StereoDecoderBranch {
                    runtime: runtime.clone(),
                    frontier: vec![initial_stereo_state_for_root(
                        runtime.as_ref(),
                        graph.as_ref(),
                        atom_idx,
                    )],
                });
            }
            return Ok(Self::from_merged(graph, branches));
        }
        let root_idx = validate_root_idx(graph.as_ref(), root_idx)?;
        let runtime = Arc::new(build_walker_runtime(graph.as_ref(), root_idx)?);
        Ok(Self::from_single(
            graph.clone(),
            runtime.clone(),
            vec![initial_stereo_state_for_root(
                runtime.as_ref(),
                graph.as_ref(),
                root_idx,
            )],
        ))
    }

    fn next_token_support(&mut self) -> PyResult<Vec<String>> {
        self.mode.next_token_support(&self.graph)
    }

    fn advance_token(&mut self, chosen_token: &str) -> PyResult<()> {
        self.mode.advance_token(&self.graph, chosen_token)
    }

    fn next_choice_texts(&mut self) -> PyResult<Vec<String>> {
        self.mode.next_choice_texts(&self.graph)
    }

    fn advance_choice(&mut self, chosen_idx: usize) -> PyResult<()> {
        self.mode.advance_choice(&self.graph, chosen_idx)
    }

    fn choice_successors(&self) -> PyResult<Vec<(String, Self)>> {
        Ok(self
            .mode
            .choice_successor_modes(&self.graph)?
            .into_iter()
            .map(|(token, mode)| (token, Self::from_mode(self.graph.clone(), mode)))
            .collect())
    }

    fn grouped_successors(&self) -> PyResult<Vec<(String, Self)>> {
        Ok(self
            .mode
            .grouped_successor_modes(&self.graph)?
            .into_iter()
            .map(|(token, mode)| (token, Self::from_mode(self.graph.clone(), mode)))
            .collect())
    }

    fn prefix(&self) -> String {
        self.mode.prefix()
    }

    fn cache_key(&self) -> String {
        self.mode.cache_key()
    }

    fn is_terminal(&self) -> PyResult<bool> {
        self.mode.is_terminal(self.graph.as_ref())
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RootedConnectedStereoDecoder(prefix={:?}, frontier_size={}, atom_count={})",
            self.prefix(),
            self.mode.frontier_size(),
            self.graph.atom_count(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use pyo3::types::{PyAnyMethods, PyDictMethods};
    use pyo3::Python;

    use super::{
        advance_stereo_choice_state, advance_stereo_token_state, apply_component_begin_atom_fact,
        apply_component_phase_commit_fact, apply_deferred_component_phase_fact,
        available_carrier_neighbors_from_support_boundary, build_walker_runtime,
        carrier_commitment_boundary_query_from_edge_state,
        carrier_commitment_decision_from_boundary_query, check_supported_stereo_writer_surface,
        choices_for_stereo_state, component_token_constraints_from_state,
        drain_exact_linear_stereo_actions, enumerate_rooted_connected_stereo_smiles_support,
        enumerate_support_from_stereo_state, flatten_exact_stereo_successor_groups,
        inferred_token_observation_facts_from_constraints, initial_stereo_state_for_root,
        is_complete_terminal_stereo_state, is_terminal_stereo_state,
        joined_support_boundary_selected_neighbors, known_token_flip_facts_from_constraints,
        marker_events_for_deferred_component_token, next_token_support_for_stereo_state,
        partial_joined_support_boundary_selected_neighbors,
        rdkit_marker_rows_accept_deferred_token, rdkit_ring_closure_projected_marker_slots,
        rdkit_traversal_writer_facts_by_component, rdkit_traversal_writer_has_completion,
        rdkit_writer_marker_event_facts_by_component,
        rdkit_writer_marker_obligation_domains_by_component,
        rdkit_writer_slot_coalesced_marker_event_facts_by_component,
        record_rdkit_literal_edge_marker_trace, resolved_constraint_state_from_walker_state,
        resolved_selected_neighbors, resolved_selected_neighbors_from_assignment_state,
        selected_neighbor_facts_by_component, smiles_from_direction_marker_slots,
        successors_by_token_stereo_raw, support_boundary_facts_from_edge_state,
        support_boundary_facts_from_walker_state, support_state_selected_neighbor_query,
        supported_token_observation_facts_from_constraints, validate_root_idx,
        validate_stereo_state_shape, CarrierCommitmentBoundaryQuery, CarrierCommitmentDecision,
        ComponentBeginAtomFact, ComponentPhaseCommitFact, ComponentTokenConstraintFact,
        DeferredCarrierChoiceConstraint, DeferredComponentPhaseFact,
        DeferredDirectionalComponentToken, DeferredDirectionalToken, MarkerEventTrace,
        RootedConnectedStereoWalkerStateData, StereoCompletionKey, StereoEdgeEmissionContext,
        StereoEdgeEmissionState, FLIPPED_COMPONENT_PHASE, STORED_COMPONENT_PHASE,
        UNKNOWN_COMPONENT_PHASE,
    };
    use crate::bond_stereo_constraints::{
        StereoAssignmentState, StereoConstraintFact, StereoConstraintLayer, StereoConstraintState,
        StereoDirectionToken, StereoMarkerEventFact, StereoTokenFlip, StereoTraversalRole,
    };
    use crate::prepared_graph::{
        PreparedSmilesGraphData, CONNECTED_STEREO_SURFACE, PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
    };

    fn stereo_support_set(graph: &PreparedSmilesGraphData, root_idx: isize) -> BTreeSet<String> {
        enumerate_rooted_connected_stereo_smiles_support(graph, root_idx)
            .expect("stereo enumeration should succeed")
            .into_iter()
            .collect()
    }

    fn stereo_runtime_and_state(
        graph: &PreparedSmilesGraphData,
        root_idx: usize,
    ) -> (
        super::StereoWalkerRuntimeData,
        super::RootedConnectedStereoWalkerStateData,
    ) {
        let runtime = build_walker_runtime(graph, root_idx).expect("stereo runtime should build");
        let state = initial_stereo_state_for_root(&runtime, graph, root_idx);
        (runtime, state)
    }

    fn observed_choice_support(
        graph: &PreparedSmilesGraphData,
        root_idx: usize,
    ) -> BTreeSet<String> {
        let (runtime, initial_state) = stereo_runtime_and_state(graph, root_idx);
        let mut stack = vec![initial_state];
        let mut observed = BTreeSet::new();
        while let Some(state) = stack.pop() {
            if is_terminal_stereo_state(&state) {
                observed.insert(state.prefix.to_string());
                continue;
            }
            let mut choices = choices_for_stereo_state(&runtime, graph, &state)
                .expect("stereo choices should enumerate");
            while let Some(choice) = choices.pop() {
                stack.extend(choice.next_frontier);
            }
        }
        observed
    }

    fn first_assignment_state_resolution_gap(
        graph: &PreparedSmilesGraphData,
        root_idx: usize,
    ) -> Option<(usize, isize, isize)> {
        let (runtime, initial_state) = stereo_runtime_and_state(graph, root_idx);
        let mut stack = vec![initial_state];
        while let Some(state) = stack.pop() {
            let assignment_resolved = resolved_selected_neighbors_from_assignment_state(
                &runtime,
                &state.stereo_selected_neighbors,
            );
            let runtime_resolved = resolved_selected_neighbors(&runtime, &state);
            for (side_idx, &assignment_neighbor_idx) in assignment_resolved.iter().enumerate() {
                if assignment_neighbor_idx < 0 || state.stereo_selected_neighbors[side_idx] >= 0 {
                    continue;
                }
                if runtime_resolved[side_idx] != assignment_neighbor_idx {
                    return Some((
                        side_idx,
                        assignment_neighbor_idx,
                        runtime_resolved[side_idx],
                    ));
                }
            }

            if is_terminal_stereo_state(&state) {
                continue;
            }
            let successors = flatten_exact_stereo_successor_groups(
                successors_by_token_stereo_raw(&runtime, graph, &state)
                    .expect("successors should enumerate"),
            );
            stack.extend(successors);
        }
        None
    }

    fn terminal_stereo_states(
        runtime: &super::StereoWalkerRuntimeData,
        graph: &PreparedSmilesGraphData,
        mut state: super::RootedConnectedStereoWalkerStateData,
        out: &mut Vec<super::RootedConnectedStereoWalkerStateData>,
    ) {
        drain_exact_linear_stereo_actions(&mut state);
        if state.action_stack.is_empty() {
            if is_complete_terminal_stereo_state(graph, &state) {
                out.push(state);
            }
            return;
        }

        let successors = flatten_exact_stereo_successor_groups(
            successors_by_token_stereo_raw(runtime, graph, &state)
                .expect("successors should enumerate"),
        );
        for successor in successors {
            terminal_stereo_states(runtime, graph, successor, out);
        }
    }

    fn first_terminal_stereo_state(
        runtime: &super::StereoWalkerRuntimeData,
        graph: &PreparedSmilesGraphData,
        state: super::RootedConnectedStereoWalkerStateData,
    ) -> super::RootedConnectedStereoWalkerStateData {
        let mut states = Vec::new();
        terminal_stereo_states(runtime, graph, state, &mut states);
        states
            .into_iter()
            .next()
            .expect("at least one terminal stereo state should be reachable")
    }

    fn sample_stereo_graph() -> PreparedSmilesGraphData {
        PreparedSmilesGraphData {
            schema_version: PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
            surface_kind: CONNECTED_STEREO_SURFACE.to_owned(),
            policy_name: "test_policy".to_owned(),
            policy_digest: "deadbeef".to_owned(),
            rdkit_version: "2025.09.6".to_owned(),
            identity_smiles: "F/C=C\\Cl".to_owned(),
            atom_count: 4,
            bond_count: 3,
            atom_atomic_numbers: vec![9, 6, 6, 17],
            atom_is_aromatic: vec![false, false, false, false],
            atom_isotopes: vec![0, 0, 0, 0],
            atom_formal_charges: vec![0, 0, 0, 0],
            atom_total_hs: vec![0, 1, 1, 0],
            atom_radical_electrons: vec![0, 0, 0, 0],
            atom_map_numbers: vec![0, 0, 0, 0],
            atom_tokens: vec![
                "F".to_owned(),
                "[CH]".to_owned(),
                "[CH]".to_owned(),
                "Cl".to_owned(),
            ],
            neighbors: vec![vec![1], vec![0, 2], vec![1, 3], vec![2]],
            neighbor_bond_tokens: vec![
                vec!["".to_owned()],
                vec!["".to_owned(), "=".to_owned()],
                vec!["=".to_owned(), "".to_owned()],
                vec!["".to_owned()],
            ],
            bond_pairs: vec![(0, 1), (1, 2), (2, 3)],
            bond_kinds: vec![
                "SINGLE".to_owned(),
                "DOUBLE".to_owned(),
                "SINGLE".to_owned(),
            ],
            writer_do_isomeric_smiles: true,
            writer_kekule_smiles: false,
            writer_all_bonds_explicit: false,
            writer_all_hs_explicit: false,
            writer_ignore_atom_map_numbers: false,
            identity_parse_with_rdkit: true,
            identity_canonical: true,
            identity_do_isomeric_smiles: true,
            identity_kekule_smiles: false,
            identity_rooted_at_atom: -1,
            identity_all_bonds_explicit: false,
            identity_all_hs_explicit: false,
            identity_do_random: false,
            identity_ignore_atom_map_numbers: false,
            atom_chiral_tags: vec![
                "CHI_UNSPECIFIED".to_owned(),
                "CHI_UNSPECIFIED".to_owned(),
                "CHI_UNSPECIFIED".to_owned(),
                "CHI_UNSPECIFIED".to_owned(),
            ],
            atom_stereo_neighbor_orders: vec![vec![1], vec![0, 2], vec![1, 3], vec![2]],
            atom_explicit_h_counts: vec![0, 1, 1, 0],
            atom_implicit_h_counts: vec![0, 0, 0, 0],
            bond_stereo_kinds: vec![
                "STEREONONE".to_owned(),
                "STEREOZ".to_owned(),
                "STEREONONE".to_owned(),
            ],
            bond_stereo_atoms: vec![(-1, -1), (0, 3), (-1, -1)],
            bond_dirs: vec![
                "ENDUPRIGHT".to_owned(),
                "NONE".to_owned(),
                "ENDDOWNRIGHT".to_owned(),
            ],
            bond_begin_atom_indices: vec![0, 1, 2],
            bond_end_atom_indices: vec![1, 2, 3],
        }
    }

    fn atom_stereo_graph() -> PreparedSmilesGraphData {
        PreparedSmilesGraphData {
            schema_version: PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
            surface_kind: CONNECTED_STEREO_SURFACE.to_owned(),
            policy_name: "test_policy".to_owned(),
            policy_digest: "deadbeef".to_owned(),
            rdkit_version: "2025.09.6".to_owned(),
            identity_smiles: "F[C@H](Cl)Br".to_owned(),
            atom_count: 4,
            bond_count: 3,
            atom_atomic_numbers: vec![9, 6, 17, 35],
            atom_is_aromatic: vec![false, false, false, false],
            atom_isotopes: vec![0, 0, 0, 0],
            atom_formal_charges: vec![0, 0, 0, 0],
            atom_total_hs: vec![0, 1, 0, 0],
            atom_radical_electrons: vec![0, 0, 0, 0],
            atom_map_numbers: vec![0, 0, 0, 0],
            atom_tokens: vec![
                "F".to_owned(),
                "C".to_owned(),
                "Cl".to_owned(),
                "Br".to_owned(),
            ],
            neighbors: vec![vec![1], vec![0, 2, 3], vec![1], vec![1]],
            neighbor_bond_tokens: vec![
                vec!["".to_owned()],
                vec!["".to_owned(), "".to_owned(), "".to_owned()],
                vec!["".to_owned()],
                vec!["".to_owned()],
            ],
            bond_pairs: vec![(0, 1), (1, 2), (1, 3)],
            bond_kinds: vec![
                "SINGLE".to_owned(),
                "SINGLE".to_owned(),
                "SINGLE".to_owned(),
            ],
            writer_do_isomeric_smiles: true,
            writer_kekule_smiles: false,
            writer_all_bonds_explicit: false,
            writer_all_hs_explicit: false,
            writer_ignore_atom_map_numbers: false,
            identity_parse_with_rdkit: true,
            identity_canonical: true,
            identity_do_isomeric_smiles: true,
            identity_kekule_smiles: false,
            identity_rooted_at_atom: -1,
            identity_all_bonds_explicit: false,
            identity_all_hs_explicit: false,
            identity_do_random: false,
            identity_ignore_atom_map_numbers: false,
            atom_chiral_tags: vec![
                "CHI_UNSPECIFIED".to_owned(),
                "CHI_TETRAHEDRAL_CCW".to_owned(),
                "CHI_UNSPECIFIED".to_owned(),
                "CHI_UNSPECIFIED".to_owned(),
            ],
            atom_stereo_neighbor_orders: vec![vec![1], vec![0, 2, 3], vec![1], vec![1]],
            atom_explicit_h_counts: vec![0, 1, 0, 0],
            atom_implicit_h_counts: vec![0, 0, 0, 0],
            bond_stereo_kinds: vec![
                "STEREONONE".to_owned(),
                "STEREONONE".to_owned(),
                "STEREONONE".to_owned(),
            ],
            bond_stereo_atoms: vec![(-1, -1), (-1, -1), (-1, -1)],
            bond_dirs: vec!["NONE".to_owned(), "NONE".to_owned(), "NONE".to_owned()],
            bond_begin_atom_indices: vec![0, 1, 1],
            bond_end_atom_indices: vec![1, 2, 3],
        }
    }

    fn prepared_graph_from_smiles(smiles: &str) -> Option<PreparedSmilesGraphData> {
        Python::initialize();
        Python::attach(|py| {
            let sys = py.import("sys").ok()?;
            let path = sys.getattr("path").ok()?;
            let version_info = sys.getattr("version_info").ok()?;
            let major: usize = version_info.get_item(0).ok()?.extract().ok()?;
            let minor: usize = version_info.get_item(1).ok()?.extract().ok()?;
            let repo_python = format!("{}/python", env!("CARGO_MANIFEST_DIR"));
            let venv_site_packages = format!(
                "{}/.venv/lib/python{}.{}/site-packages",
                env!("CARGO_MANIFEST_DIR"),
                major,
                minor
            );
            let _ = path.call_method1("insert", (0, repo_python));
            let _ = path.call_method1("insert", (0, venv_site_packages));
            let Ok(chem) = py.import("rdkit.Chem") else {
                return None;
            };
            let runtime = py
                .import("grimace._runtime")
                .expect("grimace._runtime import should succeed");
            let mol = chem
                .getattr("MolFromSmiles")
                .expect("MolFromSmiles should exist")
                .call1((smiles,))
                .expect("SMILES should parse");
            let flags = runtime
                .getattr("_make_flags")
                .expect("_make_flags should exist")
                .call0()
                .expect("_make_flags should build default flags");
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs
                .set_item("flags", flags)
                .expect("kwargs population should succeed");
            let prepared = runtime
                .getattr("prepare_smiles_graph")
                .expect("prepare_smiles_graph should exist")
                .call((mol,), Some(&kwargs))
                .expect("prepare_smiles_graph should succeed");
            Some(
                PreparedSmilesGraphData::from_any(&prepared)
                    .expect("prepared graph extraction should work"),
            )
        })
    }

    #[test]
    fn stereo_root_validation_rejects_out_of_range_indices() {
        let graph = sample_stereo_graph();
        assert!(validate_root_idx(&graph, -1).is_err());
        assert!(validate_root_idx(&graph, 4).is_err());
    }

    #[test]
    fn stereo_surface_support_matches_expected() {
        let graph = sample_stereo_graph();
        let support = stereo_support_set(&graph, 0);
        assert_eq!(BTreeSet::from(["F/[CH]=[CH]\\Cl".to_owned()]), support);
    }

    #[test]
    fn stereo_runtime_builds_noop_constraint_model() {
        let graph = sample_stereo_graph();
        let (runtime, _state) = stereo_runtime_and_state(&graph, 0);

        assert_eq!(
            runtime.side_ids_by_component.len(),
            runtime.constraint_model.component_count(),
        );
        let side_info = &runtime.side_infos[0];
        assert!(runtime.constraint_model.has_completion(
            side_info.component_idx,
            StereoConstraintLayer::Semantic,
            &[StereoConstraintFact::CarrierSelected {
                side_idx: 0,
                neighbor_idx: side_info.candidate_neighbors[0],
            }],
        ));
    }

    #[test]
    fn deferred_phase_facts_apply_explicitly() {
        let phases = vec![STORED_COMPONENT_PHASE, UNKNOWN_COMPONENT_PHASE];
        let begin_atoms = vec![-1, -1];
        let deferred_fact = DeferredComponentPhaseFact {
            component_idx: 1,
            begin_atom_idx: 2,
        };

        let (phases, begin_atoms) =
            apply_deferred_component_phase_fact(&phases, &begin_atoms, deferred_fact)
                .expect("deferred phase fact should apply");

        assert_eq!(STORED_COMPONENT_PHASE, phases[0]);
        assert_eq!(UNKNOWN_COMPONENT_PHASE, phases[1]);
        assert_eq!(-1, begin_atoms[0]);
        assert_eq!(2, begin_atoms[1]);
        apply_component_begin_atom_fact(
            &begin_atoms,
            ComponentBeginAtomFact {
                component_idx: 1,
                begin_atom_idx: 3,
            },
        )
        .expect_err("conflicting begin-atom fact should fail");

        let committed = apply_component_phase_commit_fact(
            &phases,
            ComponentPhaseCommitFact {
                component_idx: 1,
                phase: FLIPPED_COMPONENT_PHASE,
            },
        )
        .expect("phase commit fact should apply");

        assert_eq!(FLIPPED_COMPONENT_PHASE, committed[1]);
        apply_component_phase_commit_fact(
            &committed,
            ComponentPhaseCommitFact {
                component_idx: 1,
                phase: STORED_COMPONENT_PHASE,
            },
        )
        .expect_err("conflicting phase commit fact should fail");
    }

    #[test]
    fn rdkit_traversal_writer_facts_include_marker_emissions() {
        let graph = sample_stereo_graph();
        let (runtime, initial_state) = stereo_runtime_and_state(&graph, 0);
        let mut states = Vec::new();
        terminal_stereo_states(&runtime, &graph, initial_state, &mut states);

        assert_eq!(1, states.len());
        let state = &states[0];
        assert_eq!("F/[CH]=[CH]\\Cl", state.prefix.as_ref());

        let selected_neighbors = resolved_selected_neighbors(&runtime, state);
        let facts_by_component =
            rdkit_traversal_writer_facts_by_component(&runtime, state, &selected_neighbors);
        let facts = &facts_by_component[0];

        assert_eq!(
            2,
            facts
                .iter()
                .filter(|fact| matches!(fact, StereoConstraintFact::DirectionalMarkerPlaced { .. }))
                .count()
        );
        assert!(facts.iter().any(|fact| matches!(
            fact,
            StereoConstraintFact::CarrierEdgeEmitted {
                side_idx: 0,
                begin_idx: 0,
                end_idx: 1,
                role: StereoTraversalRole::TreeOrChain,
            }
        )));
        assert!(rdkit_traversal_writer_has_completion(
            &runtime,
            state,
            &selected_neighbors,
            StereoConstraintLayer::RdkitTraversalWriter,
        ));
    }

    #[test]
    fn literal_directional_edge_records_marker_placed_event() {
        let graph = sample_stereo_graph();
        let (runtime, mut state) = stereo_runtime_and_state(&graph, 0);
        record_rdkit_literal_edge_marker_trace(
            &runtime,
            &graph,
            &mut state,
            0,
            1,
            "/",
            StereoTraversalRole::TreeOrChain,
        );

        let marker_events = rdkit_writer_marker_event_facts_by_component(&runtime, &state)
            .expect("marker events should build");
        assert!(marker_events[0].iter().any(|event| matches!(
            event,
            StereoMarkerEventFact::MarkerPlaced {
                side_idx: 0,
                begin_idx: 0,
                end_idx: 1,
                marker: StereoDirectionToken::Slash,
                role: StereoTraversalRole::TreeOrChain,
                ..
            }
        )));
        assert!(!marker_events[0].iter().any(|event| matches!(
            event,
            StereoMarkerEventFact::NoMarker {
                side_idx: 0,
                begin_idx: 0,
                end_idx: 1,
                ..
            }
        )));
    }

    #[test]
    fn rdkit_traversal_writer_facts_classify_minimal_witness() {
        let Some(graph) = prepared_graph_from_smiles("C/N=C1C=C/C(=N/C)[N-]/1") else {
            return;
        };
        let mut accepted = 0usize;
        let mut rejected = 0usize;

        for root_idx in 0..graph.atom_count {
            let (runtime, initial_state) = stereo_runtime_and_state(&graph, root_idx);
            let mut states = Vec::new();
            terminal_stereo_states(&runtime, &graph, initial_state, &mut states);
            for state in states {
                let selected_neighbors = resolved_selected_neighbors(&runtime, &state);
                if rdkit_traversal_writer_has_completion(
                    &runtime,
                    &state,
                    &selected_neighbors,
                    StereoConstraintLayer::RdkitTraversalWriter,
                ) {
                    accepted += 1;
                } else {
                    rejected += 1;
                }
            }
        }

        assert_eq!(32, accepted);
        assert_eq!(12, rejected);
    }

    #[test]
    fn shared_component_deferred_edge_uses_component_local_reference_tokens() {
        let Some(graph) = prepared_graph_from_smiles("CC/C=C\\C(CO)=C(/C)CC") else {
            return;
        };
        let (runtime, mut state) = stereo_runtime_and_state(&graph, 0);
        for token in ["C", "C", "/", "C", "=", "C", "\\"] {
            state = advance_stereo_token_state(&runtime, &graph, &state, token)
                .expect("RDKit shared-edge prefix should stay completable");
        }
        assert_eq!("CC/C=C\\", state.prefix.as_ref());
    }

    #[test]
    fn isolated_two_candidate_first_emitted_observation_keeps_manual_difficult_surface() {
        let expected = "CC/C=C\\C(CO)=C(/C)CC";
        let Some(graph) = prepared_graph_from_smiles(expected) else {
            return;
        };
        let (runtime, mut state) = stereo_runtime_and_state(&graph, 0);
        for token in expected.chars().map(|token| token.to_string()) {
            state = advance_stereo_token_state(&runtime, &graph, &state, &token)
                .expect("manual difficult RDKit surface should stay completable");
        }
        assert_eq!(expected, state.prefix.as_ref());
        assert!(is_complete_terminal_stereo_state(&graph, &state));
    }

    #[test]
    fn acyclic_deferred_marker_basis_keeps_manual_stereo_atoms_surfaces() {
        for expected in [
            "CC\\C=C/C(/C=C/CC)=C(/CC)CO",
            "CC\\C=C/C(/C=C/CC)=C(\\CC)CO",
        ] {
            let Some(graph) = prepared_graph_from_smiles(expected) else {
                return;
            };
            let (runtime, mut state) = stereo_runtime_and_state(&graph, 0);
            for token in expected.chars().map(|token| token.to_string()) {
                state = advance_stereo_token_state(&runtime, &graph, &state, &token)
                    .expect("manual stereo-atoms RDKit surface should stay completable");
            }
            assert_eq!(expected, state.prefix.as_ref());
            assert!(is_complete_terminal_stereo_state(&graph, &state));
        }
    }

    #[test]
    fn deferred_marker_events_route_through_candidate_side_rows() {
        let Some(graph) = prepared_graph_from_smiles("C/C=C/C(C)=C/C") else {
            return;
        };
        let (runtime, state) = stereo_runtime_and_state(&graph, 0);
        let side_idx = runtime
            .side_infos
            .iter()
            .position(|side_info| side_info.candidate_neighbors.len() == 2)
            .expect("witness should have a two-candidate side");
        let side_info = &runtime.side_infos[side_idx];
        let candidate_neighbor = side_info.candidate_neighbors[0];
        let component_token = DeferredDirectionalComponentToken {
            component_idx: side_info.component_idx,
            reference_tokens: vec![side_info.candidate_base_tokens[0].clone()],
        };
        let deferred = DeferredDirectionalToken {
            component_tokens: Arc::from(vec![component_token.clone()]),
            begin_idx: side_info.endpoint_atom_idx as isize,
            end_idx: candidate_neighbor as isize,
        };

        assert_eq!(-1, resolved_selected_neighbors(&runtime, &state)[side_idx]);
        let events = marker_events_for_deferred_component_token(
            &runtime,
            "",
            &deferred,
            &component_token,
            "/",
        )
        .expect("candidate marker events should build");

        assert!(events.iter().any(|event| matches!(
            event,
            StereoMarkerEventFact::MarkerPlaced {
                side_idx: event_side_idx,
                ..
            } if *event_side_idx == side_idx
        )));
    }

    #[test]
    fn support_boundary_facts_expose_marker_obligation_inputs() {
        let expected = "C/C=C/C(C)=C/C";
        let Some(graph) = prepared_graph_from_smiles(expected) else {
            return;
        };
        let (runtime, mut state) = stereo_runtime_and_state(&graph, 0);
        for token in expected.chars().map(|token| token.to_string()) {
            state = advance_stereo_token_state(&runtime, &graph, &state, &token)
                .expect("witness should stay completable");
        }

        let boundary_facts = support_boundary_facts_from_walker_state(
            &runtime,
            &graph,
            &state,
            state.stereo_selected_neighbors.as_ref(),
            true,
        )
        .expect("support-boundary facts should build");
        let marker_events = rdkit_writer_marker_event_facts_by_component(&runtime, &state)
            .expect("marker event facts should build");
        assert!(marker_events.iter().any(|facts| !facts.is_empty()));
        assert_eq!(
            marker_events,
            boundary_facts.marker_event_facts_by_component
        );
        assert_eq!(
            rdkit_writer_marker_obligation_domains_by_component(&marker_events),
            boundary_facts.marker_obligation_domains_by_component,
        );
        assert_eq!(
            rdkit_writer_slot_coalesced_marker_event_facts_by_component(&marker_events),
            boundary_facts.marker_obligation_event_facts_by_component,
        );
    }

    #[test]
    fn completion_key_includes_marker_events_consumed_by_support_boundary() {
        let Some(graph) = prepared_graph_from_smiles("C/C=C/C") else {
            return;
        };
        let (runtime, state) = stereo_runtime_and_state(&graph, 0);
        let (side_idx, candidate_idx) = runtime
            .side_infos
            .iter()
            .enumerate()
            .find_map(|(side_idx, side_info)| {
                side_info
                    .candidate_base_tokens
                    .iter()
                    .position(|token| token == "/" || token == "\\")
                    .map(|candidate_idx| (side_idx, candidate_idx))
            })
            .expect("witness should have a directional side");
        let side_info = &runtime.side_infos[side_idx];
        let marked_neighbor = side_info.candidate_neighbors[candidate_idx];
        let marker = side_info.candidate_base_tokens[candidate_idx]
            .chars()
            .next()
            .expect("candidate token should be directional");

        let mut traced_state = state.clone();
        traced_state.marker_event_traces = Arc::new(vec![MarkerEventTrace {
            slot: 0,
            marker: Some(marker),
            component_idx: side_info.component_idx as isize,
            side_idx: side_idx as isize,
            endpoint_atom_idx: side_info.endpoint_atom_idx as isize,
            edge_neighbor_idx: marked_neighbor as isize,
            edge_begin_idx: side_info.endpoint_atom_idx as isize,
            edge_end_idx: marked_neighbor as isize,
            role: StereoTraversalRole::TreeOrChain,
        }]);

        assert_ne!(
            StereoCompletionKey::from(&state),
            StereoCompletionKey::from(&traced_state)
        );

        let empty_boundary_facts = support_boundary_facts_from_walker_state(
            &runtime,
            &graph,
            &state,
            state.stereo_selected_neighbors.as_ref(),
            true,
        )
        .expect("empty support-boundary facts should build");
        let traced_boundary_facts = support_boundary_facts_from_walker_state(
            &runtime,
            &graph,
            &traced_state,
            traced_state.stereo_selected_neighbors.as_ref(),
            true,
        )
        .expect("traced support-boundary facts should build");

        assert_ne!(
            empty_boundary_facts.marker_event_facts_by_component,
            traced_boundary_facts.marker_event_facts_by_component
        );
    }

    #[derive(Default, Debug, PartialEq, Eq)]
    struct SharedCarrierBoundaryDeltaCounts {
        visited_state_count: usize,
        full_joined_mismatch_count: usize,
        partial_joined_mismatch_count: usize,
        terminal_full_joined_mismatch_count: usize,
        terminal_partial_joined_mismatch_count: usize,
    }

    fn shared_carrier_boundary_delta_counts(
        runtime: &super::StereoWalkerRuntimeData,
        graph: &PreparedSmilesGraphData,
        state: RootedConnectedStereoWalkerStateData,
    ) -> SharedCarrierBoundaryDeltaCounts {
        fn visit(
            runtime: &super::StereoWalkerRuntimeData,
            graph: &PreparedSmilesGraphData,
            mut state: RootedConnectedStereoWalkerStateData,
            seen: &mut BTreeSet<RootedConnectedStereoWalkerStateData>,
            counts: &mut SharedCarrierBoundaryDeltaCounts,
        ) {
            drain_exact_linear_stereo_actions(&mut state);
            if !seen.insert(state.clone()) {
                return;
            }
            counts.visited_state_count += 1;

            let legacy = resolved_selected_neighbors(runtime, &state);
            let is_terminal = is_complete_terminal_stereo_state(graph, &state);
            let full_matches = joined_support_boundary_selected_neighbors(runtime, graph, &state)
                .map(|neighbors| neighbors == legacy)
                .unwrap_or(false);
            let partial_matches =
                partial_joined_support_boundary_selected_neighbors(runtime, graph, &state)
                    .map(|neighbors| neighbors == legacy)
                    .unwrap_or(false);

            if !full_matches {
                counts.full_joined_mismatch_count += 1;
                counts.terminal_full_joined_mismatch_count += usize::from(is_terminal);
            }
            if !partial_matches {
                counts.partial_joined_mismatch_count += 1;
                counts.terminal_partial_joined_mismatch_count += usize::from(is_terminal);
            }

            let successors = flatten_exact_stereo_successor_groups(
                successors_by_token_stereo_raw(runtime, graph, &state)
                    .expect("shared-carrier witness successors should build"),
            );
            for successor in successors {
                visit(runtime, graph, successor, seen, counts);
            }
        }

        let mut seen = BTreeSet::new();
        let mut counts = SharedCarrierBoundaryDeltaCounts::default();
        visit(runtime, graph, state, &mut seen, &mut counts);
        counts
    }

    #[test]
    fn shared_carrier_boundary_resolution_delta_counts_are_pinned() {
        let Some(graph) = prepared_graph_from_smiles("CC/C(C)=C(\\C)C(/C)=C/C") else {
            return;
        };
        let (runtime, state) = stereo_runtime_and_state(&graph, 0);
        let counts = shared_carrier_boundary_delta_counts(&runtime, &graph, state);
        assert_eq!(
            SharedCarrierBoundaryDeltaCounts {
                visited_state_count: 137,
                full_joined_mismatch_count: 109,
                partial_joined_mismatch_count: 109,
                terminal_full_joined_mismatch_count: 0,
                terminal_partial_joined_mismatch_count: 0,
            },
            counts,
        );
    }

    #[test]
    fn support_state_selected_neighbor_query_exposes_shadow_state() {
        let Some(graph) = prepared_graph_from_smiles("CC/C(C)=C(\\C)C(/C)=C/C") else {
            return;
        };
        let (runtime, state) = stereo_runtime_and_state(&graph, 0);
        let boundary_facts = support_boundary_facts_from_walker_state(
            &runtime,
            &graph,
            &state,
            state.stereo_selected_neighbors.as_ref(),
            true,
        )
        .expect("support-state selected-neighbor facts should build");
        let query = support_state_selected_neighbor_query(
            &runtime,
            state.stereo_selected_neighbors.as_ref(),
            &boundary_facts,
            true,
        )
        .expect("support-state selected-neighbor query should build");

        assert_eq!(
            query.selected_neighbors,
            partial_joined_support_boundary_selected_neighbors(&runtime, &graph, &state)
                .expect("partial support-boundary selected neighbors should build"),
        );
        assert!(query
            .component_summaries
            .iter()
            .all(|summary| summary.marker_row_count_before_events.is_some()
                && summary.marker_row_count_after_events.is_some()));
    }

    #[test]
    fn directional_marker_traces_expose_support_boundary_shadow_lookup() {
        let Some(graph) = prepared_graph_from_smiles("CC/C(C)=C(\\C)C(/C)=C/C") else {
            return;
        };
        let (runtime, initial_state) = stereo_runtime_and_state(&graph, 0);
        let mut seen = BTreeSet::new();
        let mut stack = vec![initial_state];
        let mut total_marker_trace_count = 0usize;
        let mut support_boundary_shadow_count = 0usize;
        let mut support_boundary_shadow_mismatch_count = 0usize;
        while let Some(mut state) = stack.pop() {
            drain_exact_linear_stereo_actions(&mut state);
            if !seen.insert(state.clone()) {
                continue;
            }
            for trace in state.directional_marker_traces.iter() {
                total_marker_trace_count += 1;
                if trace.support_boundary_selected_neighbor_idx >= 0 {
                    support_boundary_shadow_count += 1;
                    if trace.support_boundary_selected_neighbor_idx != trace.selected_neighbor_idx {
                        support_boundary_shadow_mismatch_count += 1;
                    }
                }
            }
            stack.extend(flatten_exact_stereo_successor_groups(
                successors_by_token_stereo_raw(&runtime, &graph, &state)
                    .expect("shared-carrier witness successors should build"),
            ));
        }
        assert_eq!(262, total_marker_trace_count);
        assert_eq!(262, support_boundary_shadow_count);
        assert_eq!(
            0,
            support_boundary_shadow_mismatch_count,
            "selected marker traces should expose a support-boundary shadow without changing the legacy trace value"
        );
    }

    #[test]
    fn edge_state_marker_rows_do_not_conflate_marker_placement_with_carrier_selection() {
        let Some(graph) = prepared_graph_from_smiles("C/C=C/C(C)=C/C") else {
            return;
        };
        let (runtime, state) = stereo_runtime_and_state(&graph, 0);
        let side_idx = runtime
            .side_infos
            .iter()
            .position(|side_info| side_info.candidate_neighbors.len() == 2)
            .expect("witness should have a two-candidate side");
        let side_info = &runtime.side_infos[side_idx];
        let blocked_neighbor = side_info.candidate_neighbors[0];
        let marked_neighbor = side_info.candidate_neighbors[1];
        let marker = side_info.candidate_base_tokens[1]
            .chars()
            .next()
            .expect("candidate token should be directional");
        let marker_event_traces = vec![
            MarkerEventTrace {
                slot: 0,
                marker: Some(marker),
                component_idx: side_info.component_idx as isize,
                side_idx: side_idx as isize,
                endpoint_atom_idx: side_info.endpoint_atom_idx as isize,
                edge_neighbor_idx: marked_neighbor as isize,
                edge_begin_idx: side_info.endpoint_atom_idx as isize,
                edge_end_idx: marked_neighbor as isize,
                role: StereoTraversalRole::TreeOrChain,
            },
            MarkerEventTrace {
                slot: 1,
                marker: None,
                component_idx: side_info.component_idx as isize,
                side_idx: side_idx as isize,
                endpoint_atom_idx: side_info.endpoint_atom_idx as isize,
                edge_neighbor_idx: blocked_neighbor as isize,
                edge_begin_idx: side_info.endpoint_atom_idx as isize,
                edge_end_idx: blocked_neighbor as isize,
                role: StereoTraversalRole::TreeOrChain,
            },
        ];

        let context = StereoEdgeEmissionContext {
            graph: &graph,
            constraint_model: &runtime.constraint_model,
            side_infos: &runtime.side_infos,
            side_ids_by_component: &runtime.side_ids_by_component,
            edge_to_side_ids: &runtime.edge_to_side_ids,
            isolated_components: &runtime.isolated_components,
        };
        let boundary_facts = support_boundary_facts_from_edge_state(
            &context,
            &state.stereo_selected_neighbors,
            &state.deferred_carrier_choice_constraints,
            &marker_event_traces,
            &state.committed_component_token_flips,
        )
        .expect("support-boundary facts should preserve edge marker events");
        let available_neighbors =
            available_carrier_neighbors_from_support_boundary(&context, &boundary_facts, side_idx)
                .expect("boundary availability should build")
                .expect("side should belong to a model component");

        // A visible marker on one candidate and a no-marker event on the
        // other constrains token-phase rows, but it does not by itself prove
        // which side candidate RDKit selected as the semantic carrier.
        assert_eq!(vec![blocked_neighbor, marked_neighbor], available_neighbors);
    }

    #[test]
    fn deferred_marker_row_filtering_coalesces_prior_no_marker_on_same_edge() {
        let Some(graph) = prepared_graph_from_smiles("C/C=C/C(C)=C/C") else {
            return;
        };
        let (runtime, mut state) = stereo_runtime_and_state(&graph, 0);
        state.prefix = Arc::from("C");
        let side_idx = runtime
            .side_infos
            .iter()
            .position(|side_info| side_info.candidate_neighbors.len() == 2)
            .expect("witness should have a two-candidate side");
        let side_info = &runtime.side_infos[side_idx];
        let candidate_neighbor = side_info.candidate_neighbors[0];
        let component_token = DeferredDirectionalComponentToken {
            component_idx: side_info.component_idx,
            reference_tokens: vec![side_info.candidate_base_tokens[0].clone()],
        };
        let deferred = DeferredDirectionalToken {
            component_tokens: Arc::from(vec![component_token.clone()]),
            begin_idx: side_info.endpoint_atom_idx as isize,
            end_idx: candidate_neighbor as isize,
        };
        let model_component_idx = runtime
            .constraint_model
            .component_for_runtime_component(component_token.component_idx)
            .expect("runtime component should be modeled");
        let constraint_state = resolved_constraint_state_from_walker_state(
            &runtime,
            &graph,
            &state,
            StereoConstraintLayer::Semantic,
        )
        .expect("initial constraint state should resolve");
        let mut marker_events_by_component =
            vec![Vec::new(); runtime.constraint_model.component_count()];
        marker_events_by_component[model_component_idx].push(StereoMarkerEventFact::NoMarker {
            side_idx,
            slot: 0,
            begin_idx: side_info.endpoint_atom_idx,
            end_idx: candidate_neighbor,
            role: StereoTraversalRole::TreeOrChain,
        });
        let mut boundary_facts = support_boundary_facts_from_walker_state(
            &runtime,
            &graph,
            &state,
            state.stereo_selected_neighbors.as_ref(),
            true,
        )
        .expect("support-boundary facts should build");
        boundary_facts.marker_event_facts_by_component = marker_events_by_component;
        boundary_facts.marker_obligation_domains_by_component =
            rdkit_writer_marker_obligation_domains_by_component(
                &boundary_facts.marker_event_facts_by_component,
            );
        boundary_facts.marker_obligation_event_facts_by_component =
            rdkit_writer_slot_coalesced_marker_event_facts_by_component(
                &boundary_facts.marker_event_facts_by_component,
            );

        assert!(rdkit_marker_rows_accept_deferred_token(
            &runtime,
            &state,
            &deferred,
            &component_token,
            StereoTokenFlip::Stored,
            &component_token.reference_tokens[0],
            &constraint_state,
            &boundary_facts,
        )
        .expect("candidate row filtering should run"));
    }

    #[test]
    fn ring_closure_projection_moves_minimal_witness_marker() {
        let Some(graph) = prepared_graph_from_smiles("C/N=C1C=C/C(=N/C)[N-]/1") else {
            return;
        };

        for root_idx in 0..graph.atom_count {
            let (runtime, initial_state) = stereo_runtime_and_state(&graph, root_idx);
            let mut states = Vec::new();
            terminal_stereo_states(&runtime, &graph, initial_state, &mut states);
            for state in states {
                if state.prefix.as_ref() != "C/N=C1/C=C/C(=N/C)[N-]1" {
                    continue;
                }
                let (skeleton, projected_slots) = rdkit_ring_closure_projected_marker_slots(&state);
                assert_eq!(
                    "C/N=C1C=C/C(=N/C)[N-]/1",
                    smiles_from_direction_marker_slots(&skeleton, &projected_slots),
                );
                return;
            }
        }

        panic!("minimal witness state with ring-open marker was not enumerated");
    }

    #[test]
    fn stereo_state_validation_checks_constraint_model_domain() {
        Python::initialize();
        let graph = sample_stereo_graph();
        let (runtime, mut state) = stereo_runtime_and_state(&graph, 0);
        let side_info = &runtime.side_infos[0];

        state.stereo_selected_neighbors =
            Arc::new(vec![side_info.other_endpoint_atom_idx as isize]);

        validate_stereo_state_shape(&runtime, &graph, &state)
            .expect_err("invalid selected carrier should fail validation");
    }

    #[test]
    fn unknown_token_flips_route_as_observations_not_token_flip_facts() {
        let graph = sample_stereo_graph();
        let (runtime, initial_state) = stereo_runtime_and_state(&graph, 0);
        let mut state = first_terminal_stereo_state(&runtime, &graph, initial_state);
        Arc::make_mut(&mut state.committed_component_token_flips).fill(None);

        let selected_neighbors = resolved_selected_neighbors(&runtime, &state);
        let constraints =
            component_token_constraints_from_state(&runtime, &graph, &state, &selected_neighbors)
                .expect("token constraints should classify");
        assert!(constraints.iter().all(|constraint| matches!(
            constraint.fact,
            ComponentTokenConstraintFact::InferredTokenObservation(_)
        )));

        let known_token_flip_facts = known_token_flip_facts_from_constraints(&constraints);
        let inferred_token_observation_facts =
            inferred_token_observation_facts_from_constraints(&constraints);
        let supported_token_observation_facts =
            supported_token_observation_facts_from_constraints(&constraints)
                .expect("supported observations should build");
        assert!(known_token_flip_facts.is_empty());
        assert_eq!(
            runtime.isolated_components.len(),
            inferred_token_observation_facts.len()
        );
        assert_eq!(
            supported_token_observation_facts,
            inferred_token_observation_facts
        );

        let facts_by_component =
            selected_neighbor_facts_by_component(&runtime, &selected_neighbors);
        let boundary_facts = support_boundary_facts_from_walker_state(
            &runtime,
            &graph,
            &state,
            &state.stereo_selected_neighbors,
            false,
        )
        .expect("support-boundary facts should build");
        assert_eq!(
            inferred_token_observation_facts,
            boundary_facts.inferred_token_observation_facts,
        );
        let runtime_state = resolved_constraint_state_from_walker_state(
            &runtime,
            &graph,
            &state,
            StereoConstraintLayer::Semantic,
        )
        .expect("runtime constraint state should build");
        let observation_state = StereoConstraintState::from_facts_and_token_observations(
            &runtime.constraint_model,
            StereoConstraintLayer::Semantic,
            &facts_by_component,
            &[],
            &inferred_token_observation_facts,
        )
        .expect("observation-backed constraint state should build");
        assert_eq!(observation_state, runtime_state);
        assert_eq!(
            runtime_state,
            boundary_facts
                .constraint_state(&runtime, StereoConstraintLayer::Semantic)
                .expect("boundary facts should build the same constraint state"),
        );
    }

    #[test]
    fn transition_carrier_resolution_still_has_assignment_state_gap() {
        let graph = sample_stereo_graph();
        let Some((side_idx, assignment_neighbor_idx, runtime_neighbor_idx)) =
            first_assignment_state_resolution_gap(&graph, 0)
        else {
            panic!("expected transition-time carrier resolution gap");
        };
        assert_eq!(0, side_idx);
        assert_ne!(assignment_neighbor_idx, runtime_neighbor_idx);
    }

    #[test]
    fn deferred_carrier_choice_constraints_filter_resolved_model_state() {
        let Some(graph) = prepared_graph_from_smiles("C/C=C(/C(=C/C)/c1ccccc1)\\c1ccccc1") else {
            return;
        };
        let (runtime, mut state) = stereo_runtime_and_state(&graph, 5);
        let unconstrained = StereoAssignmentState::from_model(
            &runtime.constraint_model,
            StereoConstraintLayer::Semantic,
        );
        let (component_idx, side_idx, unconstrained_neighbors) = runtime
            .side_infos
            .iter()
            .enumerate()
            .filter_map(|(side_idx, _side_info)| {
                let component_idx = runtime.constraint_model.component_for_side(side_idx)?;
                let remaining_ids = &unconstrained.remaining_by_component[component_idx];
                let available_neighbors = runtime
                    .constraint_model
                    .available_neighbors_for_assignment_ids(component_idx, side_idx, remaining_ids);
                (available_neighbors.len() > 1).then_some((
                    component_idx,
                    side_idx,
                    available_neighbors,
                ))
            })
            .next()
            .expect("fixture should have a multi-candidate carrier side");
        let blocked_neighbor_idx = unconstrained_neighbors[0];
        state.deferred_carrier_choice_constraints =
            Arc::new(vec![DeferredCarrierChoiceConstraint {
                side_idx,
                deferred_neighbor_idx: blocked_neighbor_idx,
                available_neighbors: unconstrained_neighbors.clone(),
            }]);

        let constraint_state = resolved_constraint_state_from_walker_state(
            &runtime,
            &graph,
            &state,
            StereoConstraintLayer::Semantic,
        )
        .expect("deferred carrier constraint should be a model fact");
        let remaining_ids = &constraint_state
            .carrier_assignment_state
            .remaining_by_component[component_idx];
        let available_neighbors = runtime
            .constraint_model
            .available_neighbors_for_assignment_ids(component_idx, side_idx, remaining_ids);

        assert!(!available_neighbors.contains(&blocked_neighbor_idx));
        assert!(!available_neighbors.is_empty());

        let context = StereoEdgeEmissionContext {
            graph: &graph,
            constraint_model: &runtime.constraint_model,
            side_infos: &runtime.side_infos,
            side_ids_by_component: &runtime.side_ids_by_component,
            edge_to_side_ids: &runtime.edge_to_side_ids,
            isolated_components: &runtime.isolated_components,
        };
        let boundary_facts = support_boundary_facts_from_edge_state(
            &context,
            &state.stereo_selected_neighbors,
            &state.deferred_carrier_choice_constraints,
            &state.marker_event_traces,
            &state.committed_component_token_flips,
        )
        .expect("support-boundary facts should build");
        let boundary_available_neighbors =
            available_carrier_neighbors_from_support_boundary(&context, &boundary_facts, side_idx)
                .expect("support-boundary availability query should build")
                .expect("side should belong to a modeled component");
        assert_eq!(available_neighbors, boundary_available_neighbors);

        let edge_state = StereoEdgeEmissionState {
            component_phases: &state.stereo_component_phases,
            selected_neighbors: &state.stereo_selected_neighbors,
            selected_orientations: &state.stereo_selected_orientations,
            first_emitted_candidates: &state.stereo_first_emitted_candidates,
            component_begin_atoms: &state.stereo_component_begin_atoms,
            deferred_carrier_choice_constraints: &state.deferred_carrier_choice_constraints,
            token_basis_facts: &state.stereo_token_basis_facts,
            marker_event_traces: &state.marker_event_traces,
            committed_component_token_flips: &state.committed_component_token_flips,
        };
        let query = carrier_commitment_boundary_query_from_edge_state(
            &context,
            &edge_state,
            &boundary_facts,
            side_idx,
            blocked_neighbor_idx,
        )
        .expect("carrier commitment query should build from boundary facts");
        assert_eq!(
            CarrierCommitmentDecision::Skip,
            carrier_commitment_decision_from_boundary_query(query)
        );
    }

    #[test]
    fn carrier_commitment_boundary_query_decision_order_is_explicit() {
        let constraint = DeferredCarrierChoiceConstraint {
            side_idx: 3,
            deferred_neighbor_idx: 5,
            available_neighbors: vec![5, 7],
        };

        assert_eq!(
            CarrierCommitmentDecision::Skip,
            carrier_commitment_decision_from_boundary_query(CarrierCommitmentBoundaryQuery {
                candidate_neighbor_idx: 5,
                forced_neighbor: None,
                blocked_by_deferred_constraint: true,
                deferrable_constraint: Some(constraint.clone()),
            })
        );
        assert_eq!(
            CarrierCommitmentDecision::Skip,
            carrier_commitment_decision_from_boundary_query(CarrierCommitmentBoundaryQuery {
                candidate_neighbor_idx: 5,
                forced_neighbor: Some(7),
                blocked_by_deferred_constraint: false,
                deferrable_constraint: Some(constraint.clone()),
            })
        );
        assert_eq!(
            CarrierCommitmentDecision::Defer(constraint),
            carrier_commitment_decision_from_boundary_query(CarrierCommitmentBoundaryQuery {
                candidate_neighbor_idx: 5,
                forced_neighbor: Some(5),
                blocked_by_deferred_constraint: false,
                deferrable_constraint: Some(DeferredCarrierChoiceConstraint {
                    side_idx: 3,
                    deferred_neighbor_idx: 5,
                    available_neighbors: vec![5, 7],
                }),
            })
        );
        assert_eq!(
            CarrierCommitmentDecision::Commit,
            carrier_commitment_decision_from_boundary_query(CarrierCommitmentBoundaryQuery {
                candidate_neighbor_idx: 5,
                forced_neighbor: Some(5),
                blocked_by_deferred_constraint: false,
                deferrable_constraint: None,
            })
        );
    }

    #[test]
    fn known_token_flips_override_observations_without_runtime_duplication() {
        let graph = sample_stereo_graph();
        let (runtime, initial_state) = stereo_runtime_and_state(&graph, 0);
        let state = first_terminal_stereo_state(&runtime, &graph, initial_state);

        let selected_neighbors = resolved_selected_neighbors(&runtime, &state);
        let constraints =
            component_token_constraints_from_state(&runtime, &graph, &state, &selected_neighbors)
                .expect("token constraints should classify");
        assert!(constraints.iter().all(|constraint| matches!(
            constraint.fact,
            ComponentTokenConstraintFact::KnownTokenFlip(_)
        )));

        let known_token_flip_facts = known_token_flip_facts_from_constraints(&constraints);
        let inferred_token_observation_facts =
            inferred_token_observation_facts_from_constraints(&constraints);
        let supported_token_observation_facts =
            supported_token_observation_facts_from_constraints(&constraints)
                .expect("supported observations should still be derivable");
        assert_eq!(
            runtime.isolated_components.len(),
            known_token_flip_facts.len()
        );
        assert!(inferred_token_observation_facts.is_empty());
        assert_eq!(
            runtime.isolated_components.len(),
            supported_token_observation_facts.len()
        );

        let facts_by_component =
            selected_neighbor_facts_by_component(&runtime, &selected_neighbors);
        let runtime_state = resolved_constraint_state_from_walker_state(
            &runtime,
            &graph,
            &state,
            StereoConstraintLayer::Semantic,
        )
        .expect("runtime constraint state should build");
        let known_only_state = StereoConstraintState::from_facts_and_token_observations(
            &runtime.constraint_model,
            StereoConstraintLayer::Semantic,
            &facts_by_component,
            &known_token_flip_facts,
            &[],
        )
        .expect("known-token constraint state should build");
        assert_eq!(known_only_state, runtime_state);
    }

    #[test]
    fn atom_stereo_support_matches_expected_curated_outputs() {
        let graph = atom_stereo_graph();
        let root_0 = stereo_support_set(&graph, 0);
        let root_1 = stereo_support_set(&graph, 1);

        assert_eq!(
            BTreeSet::from(["F[C@H](Cl)Br".to_owned(), "F[C@@H](Br)Cl".to_owned(),]),
            root_0,
        );
        assert_eq!(
            BTreeSet::from([
                "[C@H](Cl)(F)Br".to_owned(),
                "[C@@H](F)(Cl)Br".to_owned(),
                "[C@@H](Br)(F)Cl".to_owned(),
                "[C@H](F)(Br)Cl".to_owned(),
                "[C@H](Br)(Cl)F".to_owned(),
                "[C@@H](Cl)(Br)F".to_owned(),
            ]),
            root_1,
        );
    }

    #[test]
    fn stereo_direct_enumerator_matches_walker_state_machine_support() {
        for (graph, root_idx) in [
            (sample_stereo_graph(), 0usize),
            (atom_stereo_graph(), 1usize),
        ] {
            let (runtime, initial_state) = stereo_runtime_and_state(&graph, root_idx);
            let mut walker_support = BTreeSet::new();
            enumerate_support_from_stereo_state(
                &runtime,
                &graph,
                initial_state,
                &mut walker_support,
            )
            .expect("walker state-machine support should enumerate");

            let direct_support = stereo_support_set(&graph, root_idx as isize);

            assert_eq!(direct_support, walker_support);
        }
    }

    #[test]
    fn native_online_walker_matches_reference_for_coupled_diene_root_5() {
        let Some(graph) = prepared_graph_from_smiles("C/C=C(/C(=C/C)/c1ccccc1)\\c1ccccc1") else {
            return;
        };
        let observed = observed_choice_support(&graph, 5);
        let expected = stereo_support_set(&graph, 5);

        assert_eq!(expected, observed);
    }

    #[test]
    fn native_online_walker_matches_reference_for_polyene_root_11() {
        let Some(graph) =
            prepared_graph_from_smiles("CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C(=O)O)/C)/C")
        else {
            return;
        };
        let observed = observed_choice_support(&graph, 11);
        let expected = stereo_support_set(&graph, 11);

        assert_eq!(expected, observed);
    }

    #[test]
    fn stereo_empty_graph_enumerates_empty_string() {
        let mut graph = sample_stereo_graph();
        graph.atom_count = 0;
        graph.bond_count = 0;
        graph.atom_atomic_numbers.clear();
        graph.atom_is_aromatic.clear();
        graph.atom_isotopes.clear();
        graph.atom_formal_charges.clear();
        graph.atom_total_hs.clear();
        graph.atom_radical_electrons.clear();
        graph.atom_map_numbers.clear();
        graph.atom_tokens.clear();
        graph.neighbors.clear();
        graph.neighbor_bond_tokens.clear();
        graph.bond_pairs.clear();
        graph.bond_kinds.clear();
        graph.atom_chiral_tags.clear();
        graph.atom_stereo_neighbor_orders.clear();
        graph.atom_explicit_h_counts.clear();
        graph.atom_implicit_h_counts.clear();
        graph.bond_stereo_kinds.clear();
        graph.bond_stereo_atoms.clear();
        graph.bond_dirs.clear();
        graph.bond_begin_atom_indices.clear();
        graph.bond_end_atom_indices.clear();
        graph.identity_smiles.clear();

        let support = enumerate_rooted_connected_stereo_smiles_support(&graph, 0)
            .expect("empty stereo graph should enumerate");
        assert_eq!(vec![String::new()], support);
    }

    #[test]
    fn stereo_initial_state_support_is_root_atom_token() {
        let graph = sample_stereo_graph();
        let (runtime, state) = stereo_runtime_and_state(&graph, 0);
        assert_eq!(
            vec!["F".to_owned()],
            next_token_support_for_stereo_state(&runtime, &graph, &state)
                .expect("support should be available"),
        );
    }

    #[test]
    fn stereo_walker_can_reach_expected_terminal_prefix() {
        let graph = sample_stereo_graph();
        let (runtime, mut state) = stereo_runtime_and_state(&graph, 0);
        let support = stereo_support_set(&graph, 0);

        while !is_terminal_stereo_state(&state) {
            let options = next_token_support_for_stereo_state(&runtime, &graph, &state)
                .expect("support should be available");
            assert!(
                !options.is_empty(),
                "non-terminal stereo state should expose at least one token"
            );
            let chosen = options[0].clone();
            state = advance_stereo_token_state(&runtime, &graph, &state, &chosen)
                .expect("advancing along an available path should succeed");
        }

        assert!(
            support.contains(state.prefix.as_ref()),
            "unexpected terminal prefix: {}",
            state.prefix
        );
    }

    #[test]
    fn stereo_walker_rejects_invalid_token_with_choices() {
        let graph = sample_stereo_graph();
        let (runtime, state) = stereo_runtime_and_state(&graph, 0);
        Python::initialize();
        let err = advance_stereo_token_state(&runtime, &graph, &state, "(")
            .expect_err("invalid token should be rejected");
        assert!(
            err.to_string().contains("choices=[\"F\"]"),
            "unexpected error: {:?}",
            err
        );
    }

    #[test]
    fn stereo_surface_rejects_unsupported_chiral_tag() {
        let mut graph = sample_stereo_graph();
        graph.atom_chiral_tags[1] = "CHI_SQUAREPLANAR".to_owned();
        Python::initialize();
        let err = check_supported_stereo_writer_surface(&graph)
            .expect_err("unsupported chiral tags should be rejected");
        assert!(
            err.to_string().contains("Unsupported chiral tag"),
            "unexpected error: {:?}",
            err
        );
    }

    #[test]
    fn internal_root_branch_first_state_uses_forward_direction_token() {
        let graph = sample_stereo_graph();
        let (runtime, mut state) = stereo_runtime_and_state(&graph, 1);
        state = advance_stereo_choice_state(&runtime, &graph, &state, 1)
            .expect("second initial choice should be available");
        state = advance_stereo_token_state(&runtime, &graph, &state, "(")
            .expect("branch token should be available");
        for token in ["=", "[CH]"] {
            state = advance_stereo_token_state(&runtime, &graph, &state, token)
                .expect("token should be available");
        }
        assert_eq!(
            vec!["/".to_owned()],
            next_token_support_for_stereo_state(&runtime, &graph, &state)
                .expect("forward token support should be available"),
        );
    }
}
