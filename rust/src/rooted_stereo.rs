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
    stereo_constraint_model, stereo_side_infos, AmbiguousSharedEdgeGroup, StereoConstraintFact,
    StereoConstraintLayer, StereoConstraintModel, StereoSideInfo, StereoSideInfoBuild,
    CIS_STEREO_BOND_KINDS, TRANS_STEREO_BOND_KINDS,
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
struct DeferredDirectionalToken {
    component_idx: usize,
    stored_token: String,
    begin_idx: isize,
    end_idx: isize,
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
}

struct ProcessChildrenEdgeUpdate {
    edge_part: Part,
    selected_neighbors: Vec<isize>,
    selected_orientations: Vec<i8>,
    first_emitted_candidates: Vec<isize>,
    component_phases: Vec<i8>,
    component_begin_atoms: Vec<isize>,
}

struct StereoEdgeEmissionContext<'a> {
    graph: &'a PreparedSmilesGraphData,
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
    stereo_component_token_flips: Arc<Vec<i8>>,
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
    stereo_component_token_flips: Arc<Vec<i8>>,
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
            stereo_component_token_flips: state.stereo_component_token_flips.clone(),
            action_stack: state.action_stack.clone(),
        }
    }
}

type StereoCompletionCache = FxHashMap<StereoCompletionKey, bool>;

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
        stereo_component_token_flips,
        action_stack,
    } = state;
    debug_assert!(stereo_component_phases.is_empty());
    debug_assert!(stereo_selected_neighbors.is_empty());
    debug_assert!(stereo_selected_orientations.is_empty());
    debug_assert!(stereo_first_emitted_candidates.is_empty());
    debug_assert!(stereo_component_begin_atoms.is_empty());
    debug_assert!(stereo_component_token_flips.is_empty());
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
                &left.stereo_component_token_flips,
                &right.stereo_component_token_flips,
            ) {
                Ordering::Equal
            } else {
                left.stereo_component_token_flips
                    .cmp(&right.stereo_component_token_flips)
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
        let (next_phases, next_begin_atoms) =
            defer_coupled_component_phase_if_begin_side_is_unresolved(
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

// Suspicious current model:
// Defers a coupled-component phase from a ring-opening edge before the
// traversal has resolved which side should carry the visible token.
fn defer_coupled_component_phase_if_begin_side_is_unresolved(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    component_phases: &[i8],
    component_begin_atoms: &[isize],
    selected_neighbors: &[isize],
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<(Vec<i8>, Vec<isize>)> {
    let Some(bond_idx) = graph.bond_index(begin_idx, end_idx) else {
        return Ok((component_phases.to_vec(), component_begin_atoms.to_vec()));
    };
    let component_idx = runtime.stereo_component_ids[bond_idx];
    if component_idx < 0 || !is_stereo_double_bond(graph, bond_idx) {
        return Ok((component_phases.to_vec(), component_begin_atoms.to_vec()));
    }
    let component_idx = component_idx as usize;
    if runtime.isolated_components[component_idx] {
        return Ok((component_phases.to_vec(), component_begin_atoms.to_vec()));
    }

    let Some(begin_side_idx) = runtime.side_ids_by_component[component_idx]
        .iter()
        .copied()
        .find(|&side_idx| runtime.side_infos[side_idx].endpoint_atom_idx == begin_idx)
    else {
        return Ok((component_phases.to_vec(), component_begin_atoms.to_vec()));
    };
    let begin_side = &runtime.side_infos[begin_side_idx];
    if begin_side.candidate_neighbors.len() <= 1 || selected_neighbors[begin_side_idx] >= 0 {
        return Ok((component_phases.to_vec(), component_begin_atoms.to_vec()));
    }
    if component_begin_atoms[component_idx] >= 0
        && component_begin_atoms[component_idx] != begin_idx as isize
    {
        return Ok((component_phases.to_vec(), component_begin_atoms.to_vec()));
    }

    let mut updated_phases = component_phases.to_vec();
    updated_phases[component_idx] = UNKNOWN_COMPONENT_PHASE;
    let updated_begin_atoms =
        with_component_begin_atom(component_begin_atoms, component_idx, begin_idx)?;
    Ok((updated_phases, updated_begin_atoms))
}

// Suspicious current model:
// Commits the deferred coupled-component phase from the first later token that
// happens to resolve it. This should become an explicit deferred constraint.
fn commit_coupled_component_phase_from_deferred_part(
    runtime: &StereoWalkerRuntimeData,
    component_phases: &[i8],
    component_begin_atoms: &[isize],
    begin_idx: usize,
    part: &Part,
) -> PyResult<Vec<i8>> {
    let Part::Deferred(deferred) = part else {
        return Ok(component_phases.to_vec());
    };
    let component_idx = deferred.component_idx;
    if runtime.isolated_components[component_idx]
        || component_phases[component_idx] != UNKNOWN_COMPONENT_PHASE
        || component_begin_atoms
            .get(component_idx)
            .copied()
            .unwrap_or(-1)
            != begin_idx as isize
    {
        return Ok(component_phases.to_vec());
    }

    let phase = match deferred.stored_token.as_str() {
        "/" => STORED_COMPONENT_PHASE,
        "\\" => FLIPPED_COMPONENT_PHASE,
        token => {
            return Err(PyValueError::new_err(format!(
                "Unsupported deferred directional token: {token:?}"
            )));
        }
    };
    with_component_phase(component_phases, component_idx, phase)
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

// Suspicious current model:
// Hard-codes one terminal-neighbor ambiguity shape instead of representing the
// unresolved carrier choice as part of the online constraint state.
fn should_defer_unknown_two_candidate_side_commit(
    graph: &PreparedSmilesGraphData,
    side_info: &StereoSideInfo,
    component_phases: &[i8],
    neighbor_idx: usize,
) -> bool {
    if side_info.candidate_neighbors.len() != 2
        || component_phases
            .get(side_info.component_idx)
            .copied()
            .unwrap_or(UNKNOWN_COMPONENT_PHASE)
            != UNKNOWN_COMPONENT_PHASE
    {
        return false;
    }
    let terminal_candidates = side_info
        .candidate_neighbors
        .iter()
        .copied()
        .filter(|&candidate_neighbor| graph.neighbors[candidate_neighbor].len() == 1)
        .collect::<Vec<_>>();
    terminal_candidates.len() == 1 && neighbor_idx == terminal_candidates[0]
}

// Suspicious current model:
// Forces a shared candidate edge through local heuristics. This is one of the
// main places to replace with a principled deferred carrier-choice constraint.
fn forced_shared_candidate_neighbor(
    side_infos: &[StereoSideInfo],
    edge_to_side_ids: &BTreeMap<(usize, usize), Vec<usize>>,
    component_phases: &[i8],
    side_idx: usize,
) -> Option<usize> {
    let side_info = &side_infos[side_idx];
    if side_info.candidate_neighbors.len() != 2 {
        return None;
    }

    // When exactly one candidate edge is shared with another side of the same
    // coupled component, the unresolved phase often has to flow through that
    // shared edge first so both sides stay phase-compatible. Once the component
    // phase is known, stop forcing if that edge is still contested by another
    // two-candidate side; that later side may need to own the visible token.
    let shared_neighbors = side_info
        .candidate_neighbors
        .iter()
        .copied()
        .filter(|&neighbor_idx| {
            edge_to_side_ids
                .get(&canonical_edge(side_info.endpoint_atom_idx, neighbor_idx))
                .into_iter()
                .flatten()
                .copied()
                .any(|other_side_idx| {
                    other_side_idx != side_idx
                        && side_infos[other_side_idx].component_idx == side_info.component_idx
                })
        })
        .collect::<Vec<_>>();
    if shared_neighbors.len() != 1 {
        return None;
    }
    let shared_neighbor = shared_neighbors[0];
    if component_phases[side_info.component_idx] == UNKNOWN_COMPONENT_PHASE {
        return Some(shared_neighbor);
    }
    Some(shared_neighbor)
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
) -> PyResult<EmittedEdgePartResult> {
    Ok(EmittedEdgePartResult {
        part: literal_bond_part(graph, begin_idx, end_idx)?,
        selected_neighbors: selected_neighbors.to_vec(),
        selected_orientations: selected_orientations.to_vec(),
        first_emitted_candidates: first_emitted_candidates.to_vec(),
    })
}

fn edge_result_with_part(
    part: Part,
    selected_neighbors: Vec<isize>,
    selected_orientations: Vec<i8>,
    first_emitted_candidates: Vec<isize>,
) -> EmittedEdgePartResult {
    EmittedEdgePartResult {
        part,
        selected_neighbors,
        selected_orientations,
        first_emitted_candidates,
    }
}

fn deferred_edge_part(
    stored_tokens: &[(usize, String)],
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<Part> {
    let component_idx = stored_tokens[0].0;
    let stored_token = stored_tokens[0].1.clone();
    for (other_component_idx, other_stored_token) in &stored_tokens[1..] {
        if *other_component_idx != component_idx {
            return Err(PyValueError::new_err(
                "Carrier edge unexpectedly spans multiple stereo components",
            ));
        }
        if *other_stored_token != stored_token {
            return Err(PyValueError::new_err(
                "Carrier edge received conflicting stereo token assignments",
            ));
        }
    }

    Ok(Part::Deferred(DeferredDirectionalToken {
        component_idx,
        stored_token,
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
        );
    };

    let mut updated_neighbors = state.selected_neighbors.to_vec();
    let mut updated_orientations = state.selected_orientations.to_vec();
    let mut updated_first_candidates = state.first_emitted_candidates.to_vec();
    let mut stored_tokens = Vec::<(usize, String)>::new();

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
            // The ordering matters:
            // 1. If the unresolved component must flow through one shared
            //    candidate to stay coherent, force that candidate.
            // 2. Otherwise, if this is the unique terminal candidate on an
            //    unresolved two-candidate side, defer until the more
            //    informative non-terminal candidate appears.
            let forced_neighbor = forced_shared_candidate_neighbor(
                context.side_infos,
                context.edge_to_side_ids,
                state.component_phases,
                side_idx,
            );
            if forced_neighbor.is_some() && forced_neighbor != Some(neighbor_idx) {
                continue;
            }
            if should_defer_unknown_two_candidate_side_commit(
                context.graph,
                side_info,
                state.component_phases,
                neighbor_idx,
            ) {
                continue;
            }
            updated_neighbors[side_idx] = neighbor_idx as isize;
            updated_orientations[side_idx] = edge_orientation;
        }
        let selected_neighbor = updated_neighbors[side_idx];
        if selected_neighbor != neighbor_idx as isize {
            continue;
        }
        stored_tokens.push((
            side_info.component_idx,
            emitted_candidate_token(side_info, begin_idx, end_idx)?,
        ));
    }

    if stored_tokens.is_empty() {
        return Ok(edge_result_with_part(
            literal_bond_part(context.graph, begin_idx, end_idx)?,
            updated_neighbors,
            updated_orientations,
            updated_first_candidates,
        ));
    }

    Ok(edge_result_with_part(
        deferred_edge_part(&stored_tokens, begin_idx, end_idx)?,
        updated_neighbors,
        updated_orientations,
        updated_first_candidates,
    ))
}

// Suspicious current model:
// Contains isolated-component token repair logic for aromatic begin sides. Keep
// this isolated until it can be replaced by deferred token-choice constraints.
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
        );
    };

    let mut updated_neighbors = state.selected_neighbors.to_vec();
    let mut updated_orientations = state.selected_orientations.to_vec();
    let mut updated_first_candidates = state.first_emitted_candidates.to_vec();
    let mut stored_tokens = Vec::<(usize, String)>::new();

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
        let mut stored_token = emitted_candidate_token(side_info, begin_idx, end_idx)?;
        let component_idx = side_info.component_idx;
        let begin_atom_idx = state
            .component_begin_atoms
            .get(component_idx)
            .copied()
            .unwrap_or(-1);
        if side_info.candidate_neighbors.len() == 1
            && begin_atom_idx >= 0
            && begin_atom_idx as usize != side_info.endpoint_atom_idx
        {
            if let Some(begin_side_idx) = context
                .side_ids_by_component
                .get(component_idx)
                .into_iter()
                .flatten()
                .copied()
                .find(|&other_side_idx| {
                    let other_side = &context.side_infos[other_side_idx];
                    other_side.endpoint_atom_idx == begin_atom_idx as usize
                        && other_side.candidate_neighbors.len() == 2
                })
            {
                let begin_side = &context.side_infos[begin_side_idx];
                let begin_selected_neighbor = updated_neighbors[begin_side_idx];
                if begin_selected_neighbor >= 0
                    && begin_side
                        .candidate_neighbors
                        .iter()
                        .all(|&candidate_neighbor| {
                            context
                                .graph
                                .bond_index(begin_side.endpoint_atom_idx, candidate_neighbor)
                                .map(|bond_idx| context.graph.bond_kinds[bond_idx] == "AROMATIC")
                                .unwrap_or(false)
                        })
                {
                    let begin_selected_token =
                        candidate_base_token(begin_side, begin_selected_neighbor as usize)?;
                    stored_token = if begin_idx == side_info.endpoint_atom_idx {
                        begin_selected_token
                    } else {
                        flip_direction_token(&begin_selected_token)?
                    };
                }
            }
        }
        stored_tokens.push((component_idx, stored_token));
    }

    if stored_tokens.is_empty() {
        return Ok(edge_result_with_part(
            literal_bond_part(context.graph, begin_idx, end_idx)?,
            updated_neighbors,
            updated_orientations,
            updated_first_candidates,
        ));
    }

    Ok(edge_result_with_part(
        deferred_edge_part(&stored_tokens, begin_idx, end_idx)?,
        updated_neighbors,
        updated_orientations,
        updated_first_candidates,
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
        let EmittedEdgePartResult {
            selected_neighbors: updated_neighbors,
            selected_orientations: updated_orientations,
            first_emitted_candidates: updated_first_candidates,
            ..
        } = emitted_edge_part_generic(context, state, begin_idx, end_idx)?;
        if side_ids.is_empty() {
            return Ok(edge_result_with_part(
                literal_bond_part(context.graph, begin_idx, end_idx)?,
                updated_neighbors,
                updated_orientations,
                updated_first_candidates,
            ));
        }
        let component_idx = context.side_infos[side_ids[0]].component_idx;
        let stored_token =
            emitted_candidate_token(&context.side_infos[side_ids[0]], begin_idx, end_idx)?;
        for &side_idx in &side_ids[1..] {
            let side_info = &context.side_infos[side_idx];
            if side_info.component_idx != component_idx {
                return Err(PyValueError::new_err(
                    "Carrier edge unexpectedly spans multiple stereo components",
                ));
            }
            let side_token = emitted_candidate_token(side_info, begin_idx, end_idx)?;
            if side_token != stored_token {
                return Err(PyValueError::new_err(
                    "Carrier edge received conflicting stereo token assignments",
                ));
            }
        }
        Ok(edge_result_with_part(
            Part::Deferred(DeferredDirectionalToken {
                component_idx,
                stored_token,
                begin_idx: begin_idx as isize,
                end_idx: end_idx as isize,
            }),
            updated_neighbors,
            updated_orientations,
            updated_first_candidates,
        ))
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
    } = emitted_edge_part(
        &edge_context,
        &StereoEdgeEmissionState {
            component_phases: &current_phases,
            selected_neighbors: &current_selected_neighbors,
            selected_orientations: &current_selected_orientations,
            first_emitted_candidates: &current_first_candidates,
            component_begin_atoms: &current_begin_atoms,
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
    let (updated_phases, updated_begin_atoms) =
        defer_coupled_component_phase_if_begin_side_is_unresolved(
            runtime,
            graph,
            &updated_phases,
            &updated_begin_atoms,
            &updated_neighbors,
            parent_idx,
            child_idx,
        )?;
    let updated_phases = commit_coupled_component_phase_from_deferred_part(
        runtime,
        &updated_phases,
        &updated_begin_atoms,
        parent_idx,
        &edge_part,
    )?;
    Ok(ProcessChildrenEdgeUpdate {
        edge_part,
        selected_neighbors: updated_neighbors,
        selected_orientations: updated_orientations,
        first_emitted_candidates: updated_first_candidates,
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
            base_state.action_stack.push(WalkerAction::EnterAtom {
                atom_idx: step.child_idx,
                parent_idx: Some(step.parent_idx),
            });
            successors_by_token_stereo_impl(
                context.runtime,
                context.graph,
                &base_state,
                context.require_completable,
                context.completion_cache,
            )
        }
        Part::Literal(token) => {
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

// Suspicious current model:
// Resolves ambiguous shared-edge selections after the fact by forcing both
// sides to the shared edge. This is the clearest candidate for replacement by
// a first-class deferred shared-edge constraint.
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
    resolved_selected_neighbors_from_fields(
        runtime,
        &state.stereo_selected_neighbors,
        &state.stereo_first_emitted_candidates,
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

fn stereo_output_fact_row_to_py(
    py: Python<'_>,
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<Py<PyDict>> {
    let raw_selected_neighbors = state.stereo_selected_neighbors.as_ref();
    let resolved_selected_neighbors = resolved_selected_neighbors(runtime, state);

    let row = PyDict::new(py);
    row.set_item("root_idx", runtime.root_idx)?;
    row.set_item("smiles", state.prefix.as_ref())?;
    row.set_item(
        "raw_facts",
        selected_neighbor_facts_to_py(py, runtime, raw_selected_neighbors)?,
    )?;
    row.set_item(
        "resolved_facts",
        selected_neighbor_facts_to_py(py, runtime, &resolved_selected_neighbors)?,
    )?;
    row.set_item(
        "raw_layer_completions",
        selected_neighbors_layer_completions_to_py(py, runtime, raw_selected_neighbors)?,
    )?;
    row.set_item(
        "resolved_layer_completions",
        selected_neighbors_layer_completions_to_py(py, runtime, &resolved_selected_neighbors)?,
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
        if is_complete_terminal_stereo_state(graph, &state) {
            rows.append(stereo_output_fact_row_to_py(py, runtime, &state)?)?;
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
        || state.stereo_component_token_flips.len() != runtime.isolated_components.len()
        || state.stereo_selected_neighbors.len() != runtime.side_infos.len()
        || state.stereo_selected_orientations.len() != runtime.side_infos.len()
        || state.stereo_first_emitted_candidates.len() != runtime.side_infos.len()
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
        stereo_component_token_flips: Arc::new(vec![
            UNKNOWN_COMPONENT_TOKEN_FLIP;
            runtime.isolated_components.len()
        ]),
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

// Suspicious current model:
// Encodes RDKit-observed root/first-emission token flip adjustments as a local
// post-hoc correction instead of deriving them from the active constraints.
fn rdkit_component_token_flip_adjustment(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    resolved_selected_neighbors: &[isize],
    component_idx: usize,
) -> bool {
    let begin_atom_idx = state.stereo_component_begin_atoms[component_idx];
    if begin_atom_idx < 0 {
        return false;
    }
    let begin_atom_idx = begin_atom_idx as usize;

    let Some(begin_side_idx) = runtime.side_ids_by_component[component_idx]
        .iter()
        .copied()
        .find(|&side_idx| runtime.side_infos[side_idx].endpoint_atom_idx == begin_atom_idx)
    else {
        return false;
    };

    let begin_side = &runtime.side_infos[begin_side_idx];
    let mut adjustment = begin_atom_idx == runtime.root_idx
        && state.stereo_selected_orientations[begin_side_idx] == AFTER_ATOM_EDGE_ORIENTATION;

    if runtime.isolated_components[component_idx] || begin_side.candidate_neighbors.len() != 1 {
        return adjustment;
    }

    let selected_neighbor_idx = resolved_selected_neighbors[begin_side_idx];
    if selected_neighbor_idx < 0 {
        return adjustment;
    }
    let selected_neighbor_idx = selected_neighbor_idx as usize;

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
        return adjustment;
    };

    let first_neighbor_idx = state.stereo_first_emitted_candidates[adjacent_two_side_idx];
    if selected_neighbor_idx == runtime.root_idx
        && first_neighbor_idx >= 0
        && first_neighbor_idx as usize != begin_atom_idx
    {
        adjustment = !adjustment;
    }
    adjustment
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

// Suspicious current model:
// Infers one component-wide token flip from partially resolved side selections.
// This mixes state normalization with semantic constraint solving.
fn inferred_component_token_flip(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    graph: &PreparedSmilesGraphData,
    component_idx: usize,
) -> PyResult<Option<i8>> {
    let isolated = runtime.isolated_components[component_idx];
    let side_ids = &runtime.side_ids_by_component[component_idx];
    if side_ids.is_empty() {
        return Ok(None);
    }
    let resolved_selected_neighbors = resolved_selected_neighbors(runtime, state);
    let selected_side_ids = side_ids
        .iter()
        .copied()
        .filter(|&side_idx| resolved_selected_neighbors[side_idx] >= 0)
        .collect::<Vec<_>>();
    if !isolated && selected_side_ids.len() < 2 {
        return Ok(None);
    }
    let mut phase = state.stereo_component_phases[component_idx];
    let mut begin_atom_idx = state.stereo_component_begin_atoms[component_idx];
    if phase == UNKNOWN_COMPONENT_PHASE || begin_atom_idx < 0 {
        if selected_side_ids.len() != 1 {
            return Ok(None);
        }
        let selected_side_idx = selected_side_ids[0];
        let selected_side = &runtime.side_infos[selected_side_idx];
        if phase == UNKNOWN_COMPONENT_PHASE {
            phase = provisional_phase_from_selected_side(graph, selected_side)?;
        }
        if begin_atom_idx < 0 {
            begin_atom_idx = selected_side.endpoint_atom_idx as isize;
        }
    }
    let adjustment = rdkit_component_token_flip_adjustment(
        runtime,
        state,
        &resolved_selected_neighbors,
        component_idx,
    );
    if begin_atom_idx < 0 {
        return Ok(None);
    }
    let begin_atom_idx = begin_atom_idx as usize;
    let Some(begin_side_idx) = side_ids
        .iter()
        .copied()
        .find(|&side_idx| runtime.side_infos[side_idx].endpoint_atom_idx == begin_atom_idx)
    else {
        return Ok(None);
    };
    let begin_side = &runtime.side_infos[begin_side_idx];

    let component_flip = if isolated {
        let all_single_candidate = side_ids
            .iter()
            .all(|&side_idx| runtime.side_infos[side_idx].candidate_neighbors.len() == 1);
        let isolated_flip = if all_single_candidate {
            false
        } else {
            let selected_neighbor_idx = resolved_selected_neighbors[begin_side_idx];
            if selected_neighbor_idx < 0 {
                return Ok(None);
            }
            let selected_neighbor_idx = selected_neighbor_idx as usize;
            let selected_token = candidate_base_token(begin_side, selected_neighbor_idx)?;
            selected_token
                == if phase == STORED_COMPONENT_PHASE {
                    "/"
                } else {
                    "\\"
                }
        };
        isolated_flip ^ adjustment
    } else if begin_side.candidate_neighbors.len() == 1 {
        let selected_neighbor_idx = resolved_selected_neighbors[begin_side_idx];
        if selected_neighbor_idx < 0 {
            return Ok(None);
        }
        let selected_neighbor_idx = selected_neighbor_idx as usize;
        let selected_token = candidate_base_token(begin_side, selected_neighbor_idx)?;
        let coupled_flip = selected_token
            == if phase == STORED_COMPONENT_PHASE {
                "/"
            } else {
                "\\"
            };
        coupled_flip ^ adjustment
    } else if begin_side.candidate_neighbors.len() == 2 {
        let selected_neighbor_idx = resolved_selected_neighbors[begin_side_idx];
        if selected_neighbor_idx < 0 {
            return Ok(None);
        }
        let selected_neighbor_idx = selected_neighbor_idx as usize;
        let stored_token = candidate_base_token(begin_side, selected_neighbor_idx)?;
        let first_neighbor_idx = state.stereo_first_emitted_candidates[begin_side_idx];
        if first_neighbor_idx < 0 {
            adjustment
        } else {
            let resolved_token = if phase == STORED_COMPONENT_PHASE {
                stored_token.clone()
            } else {
                flip_direction_token(&stored_token)?
            };
            let invert_selected_first = stored_token == "/";
            let mut desired_token = stored_token.clone();
            if (first_neighbor_idx == resolved_selected_neighbors[begin_side_idx])
                == invert_selected_first
            {
                desired_token = flip_direction_token(&stored_token)?;
            }
            (desired_token != resolved_token) ^ adjustment
        }
    } else {
        return Ok(None);
    };

    let final_flip = (phase == FLIPPED_COMPONENT_PHASE) ^ component_flip;
    Ok(Some(if final_flip {
        FLIPPED_COMPONENT_TOKEN_FLIP
    } else {
        STORED_COMPONENT_TOKEN_FLIP
    }))
}

// Suspicious current model:
// Performs consistency repair/checking after successor construction. A cleaner
// design should make token-flip commitments explicit transition constraints.
fn normalize_component_token_flips(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &mut RootedConnectedStereoWalkerStateData,
) -> PyResult<()> {
    let resolved_selected_neighbors = resolved_selected_neighbors(runtime, state);
    for component_idx in 0..state.stereo_component_token_flips.len() {
        let inferred = inferred_component_token_flip(runtime, state, graph, component_idx)?;
        if let Some(inferred) = inferred {
            let existing = state.stereo_component_token_flips[component_idx];
            if existing != UNKNOWN_COMPONENT_TOKEN_FLIP && existing != inferred {
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

fn token_from_stored_with_flip(stored_token: &str, token_flip: i8) -> PyResult<String> {
    match token_flip {
        STORED_COMPONENT_TOKEN_FLIP => Ok(stored_token.to_owned()),
        FLIPPED_COMPONENT_TOKEN_FLIP => flip_direction_token(stored_token),
        _ => Err(PyValueError::new_err(
            "Unsupported component token flip value",
        )),
    }
}

fn raw_token_for_deferred_edge(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    deferred: &DeferredDirectionalToken,
) -> PyResult<Option<String>> {
    if deferred.begin_idx < 0 || deferred.end_idx < 0 {
        return Ok(Some(deferred.stored_token.clone()));
    }

    let begin_idx = deferred.begin_idx as usize;
    let end_idx = deferred.end_idx as usize;
    let edge = canonical_edge(begin_idx, end_idx);
    let resolved_selected_neighbors = resolved_selected_neighbors(runtime, state);
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

    if active_tokens.is_empty() {
        return Ok(None);
    }

    let raw_token = active_tokens[0].clone();
    if active_tokens[1..].iter().any(|token| token != &raw_token) {
        return Err(PyValueError::new_err(
            "Carrier edge received conflicting stereo token assignments",
        ));
    }
    Ok(Some(raw_token))
}

fn deferred_token_support(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    deferred: &DeferredDirectionalToken,
) -> PyResult<Vec<String>> {
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
    let Some(raw_token) = raw_token_for_deferred_edge(runtime, state, deferred)? else {
        return Ok(vec![literal_token.unwrap_or_default()]);
    };

    let known_flip = if state.stereo_component_token_flips[deferred.component_idx]
        != UNKNOWN_COMPONENT_TOKEN_FLIP
    {
        Some(state.stereo_component_token_flips[deferred.component_idx])
    } else {
        inferred_component_token_flip(runtime, state, graph, deferred.component_idx)?
    };
    if let Some(token_flip) = known_flip {
        return Ok(vec![token_from_stored_with_flip(&raw_token, token_flip)?]);
    }
    let flipped = flip_direction_token(&raw_token)?;
    if flipped == raw_token {
        Ok(vec![raw_token])
    } else {
        let mut out = vec![raw_token, flipped];
        out.sort();
        out.dedup();
        Ok(out)
    }
}

fn commit_deferred_token_choice(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &mut RootedConnectedStereoWalkerStateData,
    deferred: &DeferredDirectionalToken,
    chosen_token: &str,
) -> PyResult<()> {
    let raw_token = raw_token_for_deferred_edge(runtime, state, deferred)?;
    let Some(raw_token) = raw_token else {
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
        return normalize_component_token_flips(runtime, graph, state);
    };

    let flipped = flip_direction_token(&raw_token)?;
    let chosen_flip = if chosen_token == raw_token {
        STORED_COMPONENT_TOKEN_FLIP
    } else if chosen_token == flipped {
        FLIPPED_COMPONENT_TOKEN_FLIP
    } else {
        return Err(PyKeyError::new_err(format!(
            "Token {chosen_token:?} is not available for deferred stereo token"
        )));
    };
    let existing = state.stereo_component_token_flips[deferred.component_idx];
    if existing == UNKNOWN_COMPONENT_TOKEN_FLIP {
        Arc::make_mut(&mut state.stereo_component_token_flips)[deferred.component_idx] =
            chosen_flip;
    } else if existing != chosen_flip {
        return Err(PyValueError::new_err(
            "Stereo deferred token was committed inconsistently",
        ));
    }
    normalize_component_token_flips(runtime, graph, state)
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
                normalize_component_token_flips(runtime, graph, &mut successor)?;
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
                            } = emitted_edge_part(
                                &edge_context,
                                &StereoEdgeEmissionState {
                                    component_phases: &current_component_phases,
                                    selected_neighbors: &current_selected_neighbors,
                                    selected_orientations: &current_selected_orientations,
                                    first_emitted_candidates: &current_first_emitted_candidates,
                                    component_begin_atoms: &current_component_begin_atoms,
                                },
                                atom_idx,
                                closure.other_atom_idx,
                            )?;
                            current_selected_neighbors = updated_neighbors;
                            current_selected_orientations = updated_orientations;
                            current_first_emitted_candidates = updated_first_candidates;
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
                                defer_coupled_component_phase_if_begin_side_is_unresolved(
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
                            stereo_component_token_flips: base_state
                                .stereo_component_token_flips
                                .clone(),
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
                        normalize_component_token_flips(runtime, graph, &mut successor)?;
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
                            stereo_component_token_flips: base_state
                                .stereo_component_token_flips
                                .clone(),
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
        push_process_children_branch_actions(
            &mut successor.action_stack,
            parent_idx,
            child_order.clone(),
            next_branch_index,
            child_idx,
            part_to_action(edge_part),
        );
        normalize_component_token_flips(context.runtime, context.graph, &mut successor)?;
        return Ok(BTreeMap::from([("(".to_owned(), vec![successor])]));
    }

    let child_idx = child_order[child_order.len() - 1];
    let ProcessChildrenEdgeUpdate {
        edge_part,
        selected_neighbors,
        selected_orientations,
        first_emitted_candidates,
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
    normalize_component_token_flips(context.runtime, context.graph, &mut base_state)?;
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
            stereo_component_token_flips: state.stereo_component_token_flips.clone(),
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

    let successors = match successors_by_token_stereo_impl(runtime, graph, state, false, cache) {
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
    Ok(
        successors_by_token_stereo_impl(runtime, graph, state, true, completion_cache)?
            .into_keys()
            .collect(),
    )
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

fn successors_by_token_stereo_impl(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    require_completable: bool,
    completion_cache: &mut StereoCompletionCache,
) -> PyResult<BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>> {
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
                let nested = successors_by_token_stereo_impl(
                    runtime,
                    graph,
                    &successor,
                    require_completable,
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
    successors_by_token_stereo_impl(runtime, graph, state, true, &mut completion_cache)
}

fn successors_by_token_stereo_raw(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>> {
    let mut completion_cache = FxHashMap::default();
    successors_by_token_stereo_impl(runtime, graph, state, false, &mut completion_cache)
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
        if is_complete_terminal_stereo_state(graph, &state) {
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
        advance_stereo_choice_state, advance_stereo_token_state, build_walker_runtime,
        check_supported_stereo_writer_surface, choices_for_stereo_state,
        enumerate_rooted_connected_stereo_smiles_support, enumerate_support_from_stereo_state,
        initial_stereo_state_for_root, is_terminal_stereo_state,
        next_token_support_for_stereo_state, validate_root_idx, validate_stereo_state_shape,
    };
    use crate::bond_stereo_constraints::{StereoConstraintFact, StereoConstraintLayer};
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
