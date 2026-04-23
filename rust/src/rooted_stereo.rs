use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::cmp::Ordering;
use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rustc_hash::FxHashMap;

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
const CIS_STEREO_BOND_KINDS: &[&str] = &["STEREOCIS", "STEREOZ"];
const TRANS_STEREO_BOND_KINDS: &[&str] = &["STEREOE", "STEREOTRANS"];

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
    RingLabel(usize),
    OpenParen,
    CloseParen,
    Deferred(DeferredDirectionalToken),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct StereoSideInfo {
    component_idx: usize,
    endpoint_atom_idx: usize,
    other_endpoint_atom_idx: usize,
    candidate_neighbors: Vec<usize>,
    candidate_base_tokens: Vec<String>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum RingAction {
    Close(usize),
    Open(usize),
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
pub(crate) struct RootedConnectedStereoWalkerStateData {
    prefix: Arc<str>,
    visited: Arc<[bool]>,
    visited_count: usize,
    pending: Vec<(usize, Vec<PendingRing>)>,
    free_labels: Vec<usize>,
    next_label: usize,
    stereo_component_phases: Vec<i8>,
    stereo_selected_neighbors: Vec<isize>,
    stereo_selected_orientations: Vec<i8>,
    stereo_first_emitted_candidates: Vec<isize>,
    stereo_component_begin_atoms: Vec<isize>,
    stereo_component_token_flips: Vec<i8>,
    action_stack: Vec<WalkerAction>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StereoCompletionKey {
    visited: Arc<[bool]>,
    visited_count: usize,
    pending: Vec<(usize, Vec<PendingRing>)>,
    free_labels: Vec<usize>,
    next_label: usize,
    stereo_component_phases: Vec<i8>,
    stereo_selected_neighbors: Vec<isize>,
    stereo_selected_orientations: Vec<i8>,
    stereo_first_emitted_candidates: Vec<isize>,
    stereo_component_begin_atoms: Vec<isize>,
    stereo_component_token_flips: Vec<i8>,
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
#[allow(dead_code)]
struct SupportSearchResult {
    parts: Vec<Part>,
    visited: Arc<[bool]>,
    visited_count: usize,
    pending: Vec<(usize, Vec<PendingRing>)>,
    free_labels: Vec<usize>,
    next_label: usize,
    stereo_component_phases: Vec<i8>,
    stereo_selected_neighbors: Vec<isize>,
    stereo_selected_orientations: Vec<i8>,
    stereo_first_emitted_candidates: Vec<isize>,
    stereo_component_begin_atoms: Vec<isize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[allow(dead_code)]
struct SupportSubproblemKey {
    atom_idx: usize,
    parent_idx: Option<usize>,
    visited: Arc<[bool]>,
    visited_count: usize,
    pending: Vec<(usize, Vec<PendingRing>)>,
    free_labels: Vec<usize>,
    next_label: usize,
    stereo_component_phases: Vec<i8>,
    stereo_selected_neighbors: Vec<isize>,
    stereo_selected_orientations: Vec<i8>,
    stereo_first_emitted_candidates: Vec<isize>,
    stereo_component_begin_atoms: Vec<isize>,
}

#[derive(Clone, Debug)]
struct AmbiguousSharedEdgeGroup {
    left_side_idx: usize,
    right_side_idx: usize,
    left_shared_neighbor: usize,
    right_shared_neighbor: usize,
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
}

#[derive(Clone)]
struct StereoDecoderBranch {
    runtime: Arc<StereoWalkerRuntimeData>,
    frontier: Vec<RootedConnectedStereoWalkerStateData>,
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

fn cmp_stereo_state_structure(
    left: &RootedConnectedStereoWalkerStateData,
    right: &RootedConnectedStereoWalkerStateData,
) -> Ordering {
    left.action_stack
        .len()
        .cmp(&right.action_stack.len())
        .then(left.action_stack.cmp(&right.action_stack))
        .then(left.pending.len().cmp(&right.pending.len()))
        .then(left.pending.cmp(&right.pending))
        .then(left.free_labels.cmp(&right.free_labels))
        .then(left.next_label.cmp(&right.next_label))
        .then(
            left.stereo_component_phases
                .cmp(&right.stereo_component_phases),
        )
        .then(
            left.stereo_selected_neighbors
                .cmp(&right.stereo_selected_neighbors),
        )
        .then(
            left.stereo_selected_orientations
                .cmp(&right.stereo_selected_orientations),
        )
        .then(
            left.stereo_first_emitted_candidates
                .cmp(&right.stereo_first_emitted_candidates),
        )
        .then(
            left.stereo_component_begin_atoms
                .cmp(&right.stereo_component_begin_atoms),
        )
        .then(
            left.stereo_component_token_flips
                .cmp(&right.stereo_component_token_flips),
        )
        .then(left.visited_count.cmp(&right.visited_count))
        .then(left.visited.cmp(&right.visited))
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


fn flip_direction_token(token: &str) -> PyResult<String> {
    match token {
        "/" => Ok("\\".to_owned()),
        "\\" => Ok("/".to_owned()),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported directional token: {token:?}"
        ))),
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
    let mut remaining_neighbors = BTreeSet::new();
    for &neighbor_idx in graph.neighbors_of(atom_idx) {
        if !visited[neighbor_idx] {
            remaining_neighbors.insert(neighbor_idx);
        }
    }
    if remaining_neighbors.is_empty() {
        return Vec::new();
    }

    let mut groups_with_mins = Vec::<(usize, Vec<usize>)>::new();
    while let Some(&seed) = remaining_neighbors.iter().next() {
        remaining_neighbors.remove(&seed);
        let mut queue = VecDeque::from([seed]);
        let mut seen = vec![false; graph.atom_count()];
        seen[seed] = true;
        let mut component_min = seed;
        let mut group = vec![seed];

        while let Some(current) = queue.pop_front() {
            if current < component_min {
                component_min = current;
            }
            for &neighbor_idx in graph.neighbors_of(current) {
                if neighbor_idx == atom_idx || visited[neighbor_idx] || seen[neighbor_idx] {
                    continue;
                }
                seen[neighbor_idx] = true;
                if remaining_neighbors.remove(&neighbor_idx) {
                    group.push(neighbor_idx);
                }
                queue.push_back(neighbor_idx);
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

    let mut current = Vec::new();
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

fn canonical_edge(begin_idx: usize, end_idx: usize) -> (usize, usize) {
    if begin_idx < end_idx {
        (begin_idx, end_idx)
    } else {
        (end_idx, begin_idx)
    }
}

fn is_stereo_double_bond(graph: &PreparedSmilesGraphData, bond_idx: usize) -> bool {
    if graph.bond_kinds[bond_idx] != "DOUBLE" {
        return false;
    }
    let stereo_kind = graph.bond_stereo_kinds[bond_idx].as_str();
    CIS_STEREO_BOND_KINDS.contains(&stereo_kind) || TRANS_STEREO_BOND_KINDS.contains(&stereo_kind)
}

fn stereo_component_ids(graph: &PreparedSmilesGraphData) -> Vec<isize> {
    let stereo_bond_indices = (0..graph.bond_count)
        .filter(|&bond_idx| is_stereo_double_bond(graph, bond_idx))
        .collect::<Vec<_>>();
    if stereo_bond_indices.is_empty() {
        return vec![-1; graph.bond_count];
    }

    let mut parents = stereo_bond_indices
        .iter()
        .map(|&bond_idx| (bond_idx, bond_idx))
        .collect::<BTreeMap<_, _>>();

    fn find(parents: &mut BTreeMap<usize, usize>, bond_idx: usize) -> usize {
        let mut root = bond_idx;
        while parents[&root] != root {
            root = parents[&root];
        }
        let mut current = bond_idx;
        while parents[&current] != current {
            let next_idx = parents[&current];
            parents.insert(current, root);
            current = next_idx;
        }
        root
    }

    fn union(parents: &mut BTreeMap<usize, usize>, left_idx: usize, right_idx: usize) {
        let left_root = find(parents, left_idx);
        let right_root = find(parents, right_idx);
        if left_root != right_root {
            parents.insert(right_root, left_root);
        }
    }

    let mut edge_to_bonds = BTreeMap::<(usize, usize), Vec<usize>>::new();
    for &bond_idx in &stereo_bond_indices {
        let stored_begin_idx = graph.bond_begin_atom_indices[bond_idx];
        let stored_end_idx = graph.bond_end_atom_indices[bond_idx];
        let (stereo_begin_atom, stereo_end_atom) = graph.bond_stereo_atoms[bond_idx];
        if stereo_begin_atom >= 0 {
            edge_to_bonds
                .entry(canonical_edge(stored_begin_idx, stereo_begin_atom as usize))
                .or_default()
                .push(bond_idx);
        }
        if stereo_end_atom >= 0 {
            edge_to_bonds
                .entry(canonical_edge(stored_end_idx, stereo_end_atom as usize))
                .or_default()
                .push(bond_idx);
        }
    }

    for connected_bonds in edge_to_bonds.values() {
        let head_idx = connected_bonds[0];
        for &other_idx in &connected_bonds[1..] {
            union(&mut parents, head_idx, other_idx);
        }
    }

    let mut component_lookup = BTreeMap::<usize, isize>::new();
    let mut component_ids = vec![-1; graph.bond_count];
    let mut next_component_id = 0isize;
    for &bond_idx in &stereo_bond_indices {
        let root_idx = find(&mut parents, bond_idx);
        let component_id = *component_lookup.entry(root_idx).or_insert_with(|| {
            let current = next_component_id;
            next_component_id += 1;
            current
        });
        component_ids[bond_idx] = component_id;
    }

    component_ids
}

fn component_sizes(stereo_component_ids: &[isize]) -> Vec<usize> {
    let component_count = stereo_component_ids.iter().copied().max().unwrap_or(-1) + 1;
    let mut counts = vec![0usize; component_count as usize];
    for &component_idx in stereo_component_ids {
        if component_idx >= 0 {
            counts[component_idx as usize] += 1;
        }
    }
    counts
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
    let phase = if (begin_idx, end_idx) == (stored_begin_idx, stored_end_idx) {
        STORED_COMPONENT_PHASE
    } else if CIS_STEREO_BOND_KINDS.contains(&stereo_kind) {
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
            .find(|&side_idx| runtime.side_infos[side_idx].endpoint_atom_idx == begin_atom_idx as usize)
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

    for component_idx in 0..runtime.isolated_components.len() {
        if !runtime.isolated_components[component_idx] {
            continue;
        }
        if component_begin_atoms[component_idx] != parent_idx as isize {
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

fn stereo_side_infos(
    graph: &PreparedSmilesGraphData,
    stereo_component_ids: &[isize],
) -> PyResult<(Vec<StereoSideInfo>, BTreeMap<(usize, usize), Vec<usize>>)> {
    let mut side_candidates = Vec::<(usize, usize, usize, Vec<usize>)>::new();
    let mut oriented_nodes = BTreeSet::<(usize, usize)>::new();
    let mut parity_edges = BTreeMap::<(usize, usize), Vec<((usize, usize), bool)>>::new();
    let mut seed_tokens = BTreeMap::<(usize, usize), String>::new();

    for bond_idx in 0..graph.bond_count {
        let component_idx = stereo_component_ids[bond_idx];
        if component_idx < 0 || !is_stereo_double_bond(graph, bond_idx) {
            continue;
        }

        let begin_idx = graph.bond_begin_atom_indices[bond_idx];
        let end_idx = graph.bond_end_atom_indices[bond_idx];
        for (endpoint_idx, other_idx) in [(begin_idx, end_idx), (end_idx, begin_idx)] {
            let candidate_neighbors = graph
                .neighbors_of(endpoint_idx)
                .iter()
                .copied()
                .filter(|&neighbor_idx| {
                    neighbor_idx != other_idx
                        && graph
                            .bond_index(endpoint_idx, neighbor_idx)
                            .map(|bond_idx| {
                                matches!(graph.bond_kinds[bond_idx].as_str(), "SINGLE" | "AROMATIC")
                            })
                            .unwrap_or(false)
                })
                .collect::<Vec<_>>();
            if candidate_neighbors.is_empty() {
                continue;
            }
            if candidate_neighbors.len() > 2 {
                return Err(PyValueError::new_err(
                    "Unsupported stereo endpoint with more than two eligible carrier edges",
                ));
            }

            side_candidates.push((
                component_idx as usize,
                endpoint_idx,
                other_idx,
                candidate_neighbors.clone(),
            ));

            let oriented_for_side = candidate_neighbors
                .iter()
                .map(|&neighbor_idx| (endpoint_idx, neighbor_idx))
                .collect::<Vec<_>>();
            for node in &oriented_for_side {
                oriented_nodes.insert(*node);
            }

            if oriented_for_side.len() == 2 {
                let left_node = oriented_for_side[0];
                let right_node = oriented_for_side[1];
                parity_edges
                    .entry(left_node)
                    .or_default()
                    .push((right_node, true));
                parity_edges
                    .entry(right_node)
                    .or_default()
                    .push((left_node, true));
            }

            for node in oriented_for_side {
                let reverse_node = (node.1, node.0);
                if oriented_nodes.contains(&reverse_node) {
                    parity_edges
                        .entry(node)
                        .or_default()
                        .push((reverse_node, true));
                    parity_edges
                        .entry(reverse_node)
                        .or_default()
                        .push((node, true));
                }

                let stored_token = graph.directed_bond_token(node.0, node.1)?;
                if stored_token == "/" || stored_token == "\\" {
                    if let Some(existing) = seed_tokens.get(&node) {
                        if existing != &stored_token {
                            return Err(PyValueError::new_err(
                                "Inconsistent stored directional token assignment",
                            ));
                        }
                    } else {
                        seed_tokens.insert(node, stored_token);
                    }
                }
            }
        }
    }

    for (_, endpoint_idx, _other_idx, candidate_neighbors) in &side_candidates {
        if candidate_neighbors.len() != 2 {
            continue;
        }
        let known_neighbors = candidate_neighbors
            .iter()
            .copied()
            .filter(|neighbor_idx| seed_tokens.contains_key(&(*endpoint_idx, *neighbor_idx)))
            .collect::<Vec<_>>();
        if known_neighbors.len() == 1 {
            let known_neighbor = known_neighbors[0];
            let other_neighbor = if candidate_neighbors[1] == known_neighbor {
                candidate_neighbors[0]
            } else {
                candidate_neighbors[1]
            };
            let known_token = seed_tokens
                .get(&(*endpoint_idx, known_neighbor))
                .cloned()
                .ok_or_else(|| PyKeyError::new_err("Missing stereo carrier seed token"))?;
            seed_tokens.insert(
                (*endpoint_idx, other_neighbor),
                flip_direction_token(&known_token)?,
            );
        }
    }

    let mut assignments = BTreeMap::<(usize, usize), String>::new();
    let mut queue = seed_tokens.keys().copied().collect::<VecDeque<_>>();
    while let Some(node) = queue.pop_front() {
        let token = seed_tokens
            .get(&node)
            .cloned()
            .ok_or_else(|| PyKeyError::new_err("Missing stereo carrier token"))?;
        if let Some(assigned) = assignments.get(&node) {
            if assigned != &token {
                return Err(PyValueError::new_err(
                    "Conflicting stereo carrier token constraints",
                ));
            }
            continue;
        }
        assignments.insert(node, token.clone());
        if let Some(entries) = parity_edges.get(&node) {
            for &(other_node, flipped) in entries {
                let other_token = if flipped {
                    flip_direction_token(&token)?
                } else {
                    token.clone()
                };
                if let Some(existing) = seed_tokens.get(&other_node) {
                    if existing != &other_token {
                        return Err(PyValueError::new_err(
                            "Conflicting stereo carrier token propagation",
                        ));
                    }
                } else {
                    seed_tokens.insert(other_node, other_token);
                    queue.push_back(other_node);
                }
            }
        }
    }

    for &node in &oriented_nodes {
        if assignments.contains_key(&node) {
            continue;
        }
        seed_tokens.insert(node, "/".to_owned());
        queue.push_back(node);
        while let Some(current_node) = queue.pop_front() {
            let current_token = seed_tokens
                .get(&current_node)
                .cloned()
                .ok_or_else(|| PyKeyError::new_err("Missing stereo carrier token"))?;
            if let Some(assigned) = assignments.get(&current_node) {
                if assigned != &current_token {
                    return Err(PyValueError::new_err(
                        "Conflicting stereo carrier token constraints",
                    ));
                }
                continue;
            }
            assignments.insert(current_node, current_token.clone());
            if let Some(entries) = parity_edges.get(&current_node) {
                for &(other_node, flipped) in entries {
                    let other_token = if flipped {
                        flip_direction_token(&current_token)?
                    } else {
                        current_token.clone()
                    };
                    if let Some(existing) = seed_tokens.get(&other_node) {
                        if existing != &other_token {
                            return Err(PyValueError::new_err(
                                "Conflicting stereo carrier token propagation",
                            ));
                        }
                    } else {
                        seed_tokens.insert(other_node, other_token);
                        queue.push_back(other_node);
                    }
                }
            }
        }
    }

    let side_infos = side_candidates
        .into_iter()
        .map(
            |(component_idx, endpoint_idx, other_idx, candidate_neighbors)| {
                let candidate_base_tokens = candidate_neighbors
                    .iter()
                    .map(|&neighbor_idx| {
                        assignments
                            .get(&(endpoint_idx, neighbor_idx))
                            .cloned()
                            .ok_or_else(|| PyKeyError::new_err("Missing stereo carrier assignment"))
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(StereoSideInfo {
                    component_idx,
                    endpoint_atom_idx: endpoint_idx,
                    other_endpoint_atom_idx: other_idx,
                    candidate_neighbors,
                    candidate_base_tokens,
                })
            },
        )
        .collect::<PyResult<Vec<_>>>()?;

    let mut edge_to_side_ids = BTreeMap::<(usize, usize), Vec<usize>>::new();
    for (side_idx, side_info) in side_infos.iter().enumerate() {
        for &neighbor_idx in &side_info.candidate_neighbors {
            edge_to_side_ids
                .entry(canonical_edge(side_info.endpoint_atom_idx, neighbor_idx))
                .or_default()
                .push(side_idx);
        }
    }

    Ok((side_infos, edge_to_side_ids))
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

fn ambiguous_shared_edge_groups(
    side_infos: &[StereoSideInfo],
    edge_to_side_ids: &BTreeMap<(usize, usize), Vec<usize>>,
    isolated_components: &[bool],
) -> Vec<AmbiguousSharedEdgeGroup> {
    let mut seen_edges = BTreeSet::new();
    let mut groups = Vec::new();

    for side_info in side_infos {
        if isolated_components[side_info.component_idx] || side_info.candidate_neighbors.len() != 2 {
            continue;
        }
        for &neighbor_idx in &side_info.candidate_neighbors {
            let edge = canonical_edge(side_info.endpoint_atom_idx, neighbor_idx);
            if seen_edges.contains(&edge) {
                continue;
            }
            let two_candidate_side_ids = edge_to_side_ids
                .get(&edge)
                .into_iter()
                .flatten()
                .copied()
                .filter(|&other_side_idx| {
                    side_infos[other_side_idx].component_idx == side_info.component_idx
                        && side_infos[other_side_idx].candidate_neighbors.len() == 2
                })
                .collect::<Vec<_>>();
            if two_candidate_side_ids.len() != 2 {
                continue;
            }
            seen_edges.insert(edge);
            let left_side_idx = two_candidate_side_ids[0];
            let right_side_idx = two_candidate_side_ids[1];
            let left_shared_neighbor = if side_infos[left_side_idx].endpoint_atom_idx == edge.0 {
                edge.1
            } else {
                edge.0
            };
            let right_shared_neighbor = if side_infos[right_side_idx].endpoint_atom_idx == edge.0 {
                edge.1
            } else {
                edge.0
            };
            groups.push(AmbiguousSharedEdgeGroup {
                left_side_idx,
                right_side_idx,
                left_shared_neighbor,
                right_shared_neighbor,
            });
        }
    }

    groups
}

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

fn emitted_edge_part_generic(
    graph: &PreparedSmilesGraphData,
    side_infos: &[StereoSideInfo],
    edge_to_side_ids: &BTreeMap<(usize, usize), Vec<usize>>,
    component_phases: &[i8],
    selected_neighbors: &[isize],
    selected_orientations: &[i8],
    first_emitted_candidates: &[isize],
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<(Part, Vec<isize>, Vec<i8>, Vec<isize>)> {
    let Some(side_ids) = edge_to_side_ids.get(&canonical_edge(begin_idx, end_idx)) else {
        return Ok((
            Part::Literal(
                graph
                    .bond_token(begin_idx, end_idx)
                    .ok_or_else(|| {
                        PyKeyError::new_err(format!(
                            "No bond between atoms {begin_idx} and {end_idx}"
                        ))
                    })?
                    .to_owned(),
            ),
            selected_neighbors.to_vec(),
            selected_orientations.to_vec(),
            first_emitted_candidates.to_vec(),
        ));
    };

    let mut updated_neighbors = selected_neighbors.to_vec();
    let mut updated_orientations = selected_orientations.to_vec();
    let mut updated_first_candidates = first_emitted_candidates.to_vec();
    let mut stored_tokens = Vec::<(usize, String)>::new();

    for &side_idx in side_ids {
        let side_info = &side_infos[side_idx];
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
                side_infos,
                edge_to_side_ids,
                component_phases,
                side_idx,
            );
            if forced_neighbor.is_some() && forced_neighbor != Some(neighbor_idx) {
                continue;
            }
            if should_defer_unknown_two_candidate_side_commit(
                graph,
                side_info,
                component_phases,
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
        return Ok((
            Part::Literal(
                graph
                    .bond_token(begin_idx, end_idx)
                    .ok_or_else(|| {
                        PyKeyError::new_err(format!(
                            "No bond between atoms {begin_idx} and {end_idx}"
                        ))
                    })?
                    .to_owned(),
            ),
            updated_neighbors,
            updated_orientations,
            updated_first_candidates,
        ));
    }

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

    Ok((
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

fn emitted_isolated_edge_part(
    graph: &PreparedSmilesGraphData,
    side_infos: &[StereoSideInfo],
    side_ids_by_component: &[Vec<usize>],
    edge_to_side_ids: &BTreeMap<(usize, usize), Vec<usize>>,
    selected_neighbors: &[isize],
    selected_orientations: &[i8],
    first_emitted_candidates: &[isize],
    component_begin_atoms: &[isize],
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<(Part, Vec<isize>, Vec<i8>, Vec<isize>)> {
    let Some(side_ids) = edge_to_side_ids.get(&canonical_edge(begin_idx, end_idx)) else {
        return Ok((
            Part::Literal(
                graph
                    .bond_token(begin_idx, end_idx)
                    .ok_or_else(|| {
                        PyKeyError::new_err(format!(
                            "No bond between atoms {begin_idx} and {end_idx}"
                        ))
                    })?
                    .to_owned(),
            ),
            selected_neighbors.to_vec(),
            selected_orientations.to_vec(),
            first_emitted_candidates.to_vec(),
        ));
    };

    let mut updated_neighbors = selected_neighbors.to_vec();
    let mut updated_orientations = selected_orientations.to_vec();
    let mut updated_first_candidates = first_emitted_candidates.to_vec();
    let mut stored_tokens = Vec::<(usize, String)>::new();

    for &side_idx in side_ids {
        let side_info = &side_infos[side_idx];
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
        let begin_atom_idx = component_begin_atoms
            .get(component_idx)
            .copied()
            .unwrap_or(-1);
        if side_info.candidate_neighbors.len() == 1
            && begin_atom_idx >= 0
            && begin_atom_idx as usize != side_info.endpoint_atom_idx
        {
            if let Some(begin_side_idx) = side_ids_by_component
                .get(component_idx)
                .into_iter()
                .flatten()
                .copied()
                .find(|&other_side_idx| {
                    let other_side = &side_infos[other_side_idx];
                    other_side.endpoint_atom_idx == begin_atom_idx as usize
                        && other_side.candidate_neighbors.len() == 2
                })
            {
                let begin_side = &side_infos[begin_side_idx];
                let begin_selected_neighbor = updated_neighbors[begin_side_idx];
                if begin_selected_neighbor >= 0
                    && begin_side
                        .candidate_neighbors
                        .iter()
                        .all(|&candidate_neighbor| {
                            graph
                                .bond_index(begin_side.endpoint_atom_idx, candidate_neighbor)
                                .map(|bond_idx| graph.bond_kinds[bond_idx] == "AROMATIC")
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
        return Ok((
            Part::Literal(
                graph
                    .bond_token(begin_idx, end_idx)
                    .ok_or_else(|| {
                        PyKeyError::new_err(format!(
                            "No bond between atoms {begin_idx} and {end_idx}"
                        ))
                    })?
                    .to_owned(),
            ),
            updated_neighbors,
            updated_orientations,
            updated_first_candidates,
        ));
    }

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

    Ok((
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

fn emitted_edge_part(
    graph: &PreparedSmilesGraphData,
    side_infos: &[StereoSideInfo],
    side_ids_by_component: &[Vec<usize>],
    edge_to_side_ids: &BTreeMap<(usize, usize), Vec<usize>>,
    component_phases: &[i8],
    selected_neighbors: &[isize],
    selected_orientations: &[i8],
    first_emitted_candidates: &[isize],
    component_begin_atoms: &[isize],
    isolated_components: &[bool],
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<(Part, Vec<isize>, Vec<i8>, Vec<isize>)> {
    let edge = canonical_edge(begin_idx, end_idx);
    let Some(side_ids) = edge_to_side_ids.get(&edge) else {
        return Ok((
            Part::Literal(
                graph
                    .bond_token(begin_idx, end_idx)
                    .ok_or_else(|| {
                        PyKeyError::new_err(format!(
                            "No bond between atoms {begin_idx} and {end_idx}"
                        ))
                    })?
                    .to_owned(),
            ),
            selected_neighbors.to_vec(),
            selected_orientations.to_vec(),
            first_emitted_candidates.to_vec(),
        ));
    };

    let uses_isolated_component = side_ids.iter().any(|&side_idx| {
        let component_idx = side_infos[side_idx].component_idx;
        isolated_components
            .get(component_idx)
            .copied()
            .unwrap_or(false)
    });
    if uses_isolated_component {
        emitted_isolated_edge_part(
            graph,
            side_infos,
            side_ids_by_component,
            edge_to_side_ids,
            selected_neighbors,
            selected_orientations,
            first_emitted_candidates,
            component_begin_atoms,
            begin_idx,
            end_idx,
        )
    } else {
        let (part, updated_neighbors, updated_orientations, updated_first_candidates) =
            emitted_edge_part_generic(
                graph,
                side_infos,
                edge_to_side_ids,
                component_phases,
                selected_neighbors,
                selected_orientations,
                first_emitted_candidates,
                begin_idx,
                end_idx,
            )?;
        if side_ids.is_empty() {
            return Ok((
                part,
                updated_neighbors,
                updated_orientations,
                updated_first_candidates,
            ));
        }
        let component_idx = side_infos[side_ids[0]].component_idx;
        let stored_token = emitted_candidate_token(&side_infos[side_ids[0]], begin_idx, end_idx)?;
        for &side_idx in &side_ids[1..] {
            let side_info = &side_infos[side_idx];
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
        Ok((
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
    enumerate_support_from_stereo_state(
        &runtime,
        graph,
        initial_stereo_state_for_root(&runtime, graph, root_idx),
        &mut support,
    )?;
    Ok(support.into_iter().collect())
}

#[cfg(test)]
fn enumerate_rooted_connected_stereo_smiles_support_native(
    graph: &PreparedSmilesGraphData,
    root_idx: isize,
) -> PyResult<Vec<String>> {
    let root_idx = validate_root_idx(graph, root_idx)?;
    if graph.atom_count() == 0 {
        return Ok(vec![String::new()]);
    }

    let runtime = build_walker_runtime(graph, root_idx)?;
    let component_count = runtime.isolated_components.len();
    let mut support = BTreeSet::new();
    let mut cache = FxHashMap::default();
    enumerate_support_results_from_atom(
        &runtime,
        graph,
        root_idx,
        None,
        SupportSearchResult {
            parts: Vec::new(),
            visited: Arc::<[bool]>::from(vec![false; graph.atom_count()]),
            visited_count: 0,
            pending: Vec::new(),
            free_labels: Vec::new(),
            next_label: 1,
            stereo_component_phases: vec![UNKNOWN_COMPONENT_PHASE; component_count],
            stereo_selected_neighbors: vec![-1; runtime.side_infos.len()],
            stereo_selected_orientations: vec![UNKNOWN_EDGE_ORIENTATION; runtime.side_infos.len()],
            stereo_first_emitted_candidates: vec![-1; runtime.side_infos.len()],
            stereo_component_begin_atoms: vec![-1; component_count],
        },
        &mut cache,
        &mut |result| {
            if result.visited_count != graph.atom_count() {
                return Ok(());
            }
            if !result.pending.is_empty() {
                return Ok(());
            }
            support.insert(resolve_support_result_smiles(&runtime, graph, &result)?);
            Ok(())
        },
    )?;
    Ok(support.into_iter().collect())
}

#[allow(dead_code)]
fn enumerate_support_results_from_atom(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    atom_idx: usize,
    parent_idx: Option<usize>,
    result: SupportSearchResult,
    cache: &mut FxHashMap<SupportSubproblemKey, Vec<SupportSearchResult>>,
    emit: &mut dyn FnMut(SupportSearchResult) -> PyResult<()>,
) -> PyResult<()> {
    let cache_key = SupportSubproblemKey {
        atom_idx,
        parent_idx,
        visited: result.visited.clone(),
        visited_count: result.visited_count,
        pending: result.pending.clone(),
        free_labels: result.free_labels.clone(),
        next_label: result.next_label,
        stereo_component_phases: result.stereo_component_phases.clone(),
        stereo_selected_neighbors: result.stereo_selected_neighbors.clone(),
        stereo_selected_orientations: result.stereo_selected_orientations.clone(),
        stereo_first_emitted_candidates: result.stereo_first_emitted_candidates.clone(),
        stereo_component_begin_atoms: result.stereo_component_begin_atoms.clone(),
    };
    if let Some(cached_results) = cache.get(&cache_key) {
        for cached_result in cached_results {
            emit(cached_result.clone())?;
        }
        return Ok(());
    }

    let visited_now = visited_with_marked(&result.visited, atom_idx);
    let visited_count_now = result.visited_count + 1;
    let mut pending_now = result.pending.clone();
    let closures_here = take_pending_for_atom(&mut pending_now, atom_idx);
    let ordered_groups = ordered_neighbor_groups(graph, atom_idx, visited_now.as_ref());
    let is_chiral_atom = graph.atom_chiral_tags[atom_idx] != "CHI_UNSPECIFIED";

    let mut local_results = Vec::new();
    let mut emit_and_cache = |generated: SupportSearchResult| -> PyResult<()> {
        local_results.push(generated.clone());
        emit(generated)
    };
    let mut status = Ok(());
    for_each_cartesian_choice(&ordered_groups, &mut |chosen_children| {
        if status.is_err() {
            return;
        }

        let total_group_members = ordered_groups.iter().map(|group| group.len()).sum::<usize>();
        let opening_target_count = total_group_members.saturating_sub(chosen_children.len());
        let mut ring_actions =
            Vec::with_capacity(closures_here.len() + opening_target_count);
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

        permutations_copy_distinct(&ring_actions, &mut |ring_action_order| {
            if status.is_err() {
                return;
            }

            let outcome: PyResult<()> = (|| {
                let mut current_pending = pending_now.clone();
                let mut current_free = result.free_labels.clone();
                let mut current_next = result.next_label;
                let mut current_component_phases = result.stereo_component_phases.clone();
                let mut current_selected_neighbors = result.stereo_selected_neighbors.clone();
                let mut current_selected_orientations = result.stereo_selected_orientations.clone();
                let mut current_first_emitted_candidates =
                    result.stereo_first_emitted_candidates.clone();
                let mut current_component_begin_atoms = result.stereo_component_begin_atoms.clone();
                let mut current_ring_parts =
                    Vec::<Part>::with_capacity(closures_here.len() * 2 + opening_target_count);
                let mut labels_freed_after_atom = Vec::<usize>::with_capacity(closures_here.len());
                let mut ring_neighbor_order = is_chiral_atom.then(Vec::<usize>::new);

                for ring_action in ring_action_order {
                    match *ring_action {
                        RingAction::Close(closure_idx) => {
                            let closure = &closures_here[closure_idx];
                            let (
                                bond_part,
                                updated_neighbors,
                                updated_orientations,
                                updated_first_candidates,
                            ) = emitted_edge_part(
                                graph,
                                &runtime.side_infos,
                                &runtime.side_ids_by_component,
                                &runtime.edge_to_side_ids,
                                &current_component_phases,
                                &current_selected_neighbors,
                                &current_selected_orientations,
                                &current_first_emitted_candidates,
                                &current_component_begin_atoms,
                                &runtime.isolated_components,
                                atom_idx,
                                closure.other_atom_idx,
                            )?;
                            current_selected_neighbors = updated_neighbors;
                            current_selected_orientations = updated_orientations;
                            current_first_emitted_candidates = updated_first_candidates;
                            push_part(&mut current_ring_parts, bond_part);
                            current_ring_parts.push(Part::RingLabel(closure.label));
                            labels_freed_after_atom.push(closure.label);
                            if let Some(order) = &mut ring_neighbor_order {
                                order.push(closure.other_atom_idx);
                            }
                        }
                        RingAction::Open(target_idx) => {
                            let label = allocate_label(&mut current_free, &mut current_next);
                            current_ring_parts.push(Part::RingLabel(label));
                            let (updated_phases, updated_begin_atoms) = component_phases_after_edge(
                                graph,
                                &runtime.stereo_component_ids,
                                &runtime.isolated_components,
                                &current_component_phases,
                                &current_component_begin_atoms,
                                atom_idx,
                                target_idx,
                            )?;
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

                        let mut prefix_parts = Vec::with_capacity(1 + current_ring_parts.len());
                        prefix_parts.push(Part::Literal(atom_token));
                        prefix_parts.extend(current_ring_parts.iter().cloned());

                        let partial = SupportSearchResult {
                            parts: prefix_parts,
                            visited: visited_now.clone(),
                            visited_count: visited_count_now,
                            pending: current_pending.clone(),
                            free_labels: current_free.clone(),
                            next_label: current_next,
                            stereo_component_phases: current_component_phases.clone(),
                            stereo_selected_neighbors: current_selected_neighbors.clone(),
                            stereo_selected_orientations: current_selected_orientations.clone(),
                            stereo_first_emitted_candidates: current_first_emitted_candidates.clone(),
                            stereo_component_begin_atoms: current_component_begin_atoms.clone(),
                        };
                        expand_support_children(
                            runtime,
                            graph,
                            atom_idx,
                            child_order,
                            partial,
                            cache,
                            &mut emit_and_cache,
                        )
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

    if status.is_ok() {
        cache.insert(cache_key, local_results);
    }
    status
}

#[allow(dead_code)]
fn expand_support_children(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    parent_idx: usize,
    child_order: &[usize],
    partial: SupportSearchResult,
    cache: &mut FxHashMap<SupportSubproblemKey, Vec<SupportSearchResult>>,
    emit: &mut dyn FnMut(SupportSearchResult) -> PyResult<()>,
) -> PyResult<()> {
    if child_order.is_empty() {
        return emit(partial);
    }

    let branch_count = child_order.len().saturating_sub(1);
    for branch_index in 0..=branch_count {
        if branch_index == branch_count {
            let main_child = child_order[child_order.len() - 1];
            let (
                edge_part,
                main_selected_neighbors,
                main_selected_orientations,
                main_first_emitted_candidates,
            ) = emitted_edge_part(
                graph,
                &runtime.side_infos,
                &runtime.side_ids_by_component,
                &runtime.edge_to_side_ids,
                &partial.stereo_component_phases,
                &partial.stereo_selected_neighbors,
                &partial.stereo_selected_orientations,
                &partial.stereo_first_emitted_candidates,
                &partial.stereo_component_begin_atoms,
                &runtime.isolated_components,
                parent_idx,
                main_child,
            )?;
            let (main_component_phases, main_component_begin_atoms) = component_phases_after_edge(
                graph,
                &runtime.stereo_component_ids,
                &runtime.isolated_components,
                &partial.stereo_component_phases,
                &partial.stereo_component_begin_atoms,
                parent_idx,
                main_child,
            )?;
            let (main_component_phases, main_component_begin_atoms) =
                defer_coupled_component_phase_if_begin_side_is_unresolved(
                    runtime,
                    graph,
                    &main_component_phases,
                    &main_component_begin_atoms,
                    &main_selected_neighbors,
                    parent_idx,
                    main_child,
                )?;
            let main_component_phases = commit_coupled_component_phase_from_deferred_part(
                runtime,
                &main_component_phases,
                &main_component_begin_atoms,
                parent_idx,
                &edge_part,
            )?;
            enumerate_support_results_from_atom(
                runtime,
                graph,
                main_child,
                Some(parent_idx),
                SupportSearchResult {
                    parts: Vec::new(),
                    visited: partial.visited.clone(),
                    visited_count: partial.visited_count,
                    pending: partial.pending.clone(),
                    free_labels: partial.free_labels.clone(),
                    next_label: partial.next_label,
                    stereo_component_phases: main_component_phases,
                    stereo_selected_neighbors: main_selected_neighbors,
                    stereo_selected_orientations: main_selected_orientations,
                    stereo_first_emitted_candidates: main_first_emitted_candidates,
                    stereo_component_begin_atoms: main_component_begin_atoms,
                },
                cache,
                &mut |main_result| {
                    let mut parts = partial.parts.clone();
                    push_part(&mut parts, edge_part.clone());
                    parts.extend(main_result.parts);
                    emit(SupportSearchResult {
                        parts,
                        visited: main_result.visited,
                        visited_count: main_result.visited_count,
                        pending: main_result.pending,
                        free_labels: main_result.free_labels,
                        next_label: main_result.next_label,
                        stereo_component_phases: main_result.stereo_component_phases,
                        stereo_selected_neighbors: main_result.stereo_selected_neighbors,
                        stereo_selected_orientations: main_result.stereo_selected_orientations,
                        stereo_first_emitted_candidates: main_result.stereo_first_emitted_candidates,
                        stereo_component_begin_atoms: main_result.stereo_component_begin_atoms,
                    })
                },
            )?;
            return Ok(());
        }

        let child_idx = child_order[branch_index];
        let (
            branch_part,
            branch_selected_neighbors,
            branch_selected_orientations,
            branch_first_emitted_candidates,
        ) = emitted_edge_part(
            graph,
            &runtime.side_infos,
            &runtime.side_ids_by_component,
            &runtime.edge_to_side_ids,
            &partial.stereo_component_phases,
            &partial.stereo_selected_neighbors,
            &partial.stereo_selected_orientations,
            &partial.stereo_first_emitted_candidates,
            &partial.stereo_component_begin_atoms,
            &runtime.isolated_components,
            parent_idx,
            child_idx,
        )?;
        let (child_component_phases, child_component_begin_atoms) = component_phases_after_edge(
            graph,
            &runtime.stereo_component_ids,
            &runtime.isolated_components,
            &partial.stereo_component_phases,
            &partial.stereo_component_begin_atoms,
            parent_idx,
            child_idx,
        )?;
        let (child_component_phases, child_component_begin_atoms) =
            defer_coupled_component_phase_if_begin_side_is_unresolved(
                runtime,
                graph,
                &child_component_phases,
                &child_component_begin_atoms,
                &branch_selected_neighbors,
                parent_idx,
                child_idx,
            )?;
        let child_component_phases = commit_coupled_component_phase_from_deferred_part(
            runtime,
            &child_component_phases,
            &child_component_begin_atoms,
            parent_idx,
            &branch_part,
        )?;
        let mut branch_results = Vec::new();
        enumerate_support_results_from_atom(
            runtime,
            graph,
            child_idx,
            Some(parent_idx),
            SupportSearchResult {
                parts: Vec::new(),
                visited: partial.visited.clone(),
                visited_count: partial.visited_count,
                pending: partial.pending.clone(),
                free_labels: partial.free_labels.clone(),
                next_label: partial.next_label,
                stereo_component_phases: child_component_phases,
                stereo_selected_neighbors: branch_selected_neighbors,
                stereo_selected_orientations: branch_selected_orientations,
                stereo_first_emitted_candidates: branch_first_emitted_candidates,
                stereo_component_begin_atoms: child_component_begin_atoms,
            },
            cache,
            &mut |branch_result| {
                branch_results.push(branch_result);
                Ok(())
            },
        )?;
        for mut branch_result in branch_results {
            let mut parts = partial.parts.clone();
            parts.push(Part::OpenParen);
            push_part(&mut parts, branch_part.clone());
            parts.extend(branch_result.parts);
            parts.push(Part::CloseParen);
            branch_result.parts = parts;
            expand_support_children(
                runtime,
                graph,
                parent_idx,
                &child_order[branch_index + 1..],
                branch_result,
                cache,
                emit,
            )?;
        }
        return Ok(());
    }

    Ok(())
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
    let (side_infos, edge_to_side_ids) = stereo_side_infos(graph, &stereo_component_ids)?;
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
    Ok(StereoWalkerRuntimeData {
        root_idx,
        stereo_component_ids,
        isolated_components,
        side_infos,
        edge_to_side_ids,
        side_ids_by_component,
        ambiguous_shared_edge_groups,
    })
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
        pending: Vec::new(),
        free_labels: Vec::new(),
        next_label: 1,
        stereo_component_phases: vec![
            UNKNOWN_COMPONENT_PHASE;
            runtime.isolated_components.len()
        ],
        stereo_selected_neighbors: vec![-1; runtime.side_infos.len()],
        stereo_selected_orientations: vec![
            UNKNOWN_EDGE_ORIENTATION;
            runtime.side_infos.len()
        ],
        stereo_first_emitted_candidates: vec![-1; runtime.side_infos.len()],
        stereo_component_begin_atoms: vec![
            -1;
            runtime.isolated_components.len()
        ],
        stereo_component_token_flips: vec![
            UNKNOWN_COMPONENT_TOKEN_FLIP;
            runtime.isolated_components.len()
        ],
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
        Part::RingLabel(label) => Some(WalkerAction::EmitRingLabel(label)),
        Part::OpenParen => Some(WalkerAction::EmitLiteral("(".to_owned())),
        Part::CloseParen => Some(WalkerAction::EmitCloseParen),
        Part::Deferred(token) => Some(WalkerAction::EmitDeferred(token)),
    }
}

#[allow(dead_code)]
fn push_part(parts: &mut Vec<Part>, part: Part) {
    match &part {
        Part::Literal(token) if token.is_empty() => {}
        _ => parts.push(part),
    }
}

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
    let Some(bond_idx) = graph.bond_index(side_info.endpoint_atom_idx, side_info.other_endpoint_atom_idx)
    else {
        return Err(PyKeyError::new_err(format!(
            "No bond between atoms {} and {}",
            side_info.endpoint_atom_idx, side_info.other_endpoint_atom_idx
        )));
    };
    let stored_begin_idx = graph.bond_begin_atom_indices[bond_idx];
    let stored_end_idx = graph.bond_end_atom_indices[bond_idx];
    let stereo_kind = graph.bond_stereo_kinds[bond_idx].as_str();
    if (side_info.endpoint_atom_idx, side_info.other_endpoint_atom_idx)
        == (stored_begin_idx, stored_end_idx)
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

#[allow(dead_code)]
fn resolve_directional_token(
    token: &str,
    component_idx: usize,
    component_phases: &[i8],
    component_flips: &[bool],
) -> PyResult<String> {
    let mut resolved = token.to_owned();
    if component_phases[component_idx] == FLIPPED_COMPONENT_PHASE {
        resolved = flip_direction_token(&resolved)?;
    }
    if component_flips[component_idx] {
        resolved = flip_direction_token(&resolved)?;
    }
    Ok(resolved)
}

#[allow(dead_code)]
fn isolated_component_flips_for_result(
    runtime: &StereoWalkerRuntimeData,
    _graph: &PreparedSmilesGraphData,
    result: &SupportSearchResult,
) -> PyResult<Vec<bool>> {
    if runtime.isolated_components.is_empty() {
        return Ok(Vec::new());
    }

    let mut flips = vec![false; runtime.isolated_components.len()];
    for (component_idx, isolated) in runtime.isolated_components.iter().copied().enumerate() {
        if !isolated {
            continue;
        }

        let side_ids = &runtime.side_ids_by_component[component_idx];
        if side_ids.is_empty() {
            continue;
        }
        if side_ids
            .iter()
            .all(|&side_idx| runtime.side_infos[side_idx].candidate_neighbors.len() == 1)
        {
            continue;
        }

        let begin_atom_idx = result.stereo_component_begin_atoms[component_idx];
        if begin_atom_idx < 0 {
            continue;
        }
        let Some(begin_side_idx) = side_ids.iter().copied().find(|&side_idx| {
            runtime.side_infos[side_idx].endpoint_atom_idx == begin_atom_idx as usize
        }) else {
            continue;
        };

        let selected_neighbor_idx = result.stereo_selected_neighbors[begin_side_idx];
        if selected_neighbor_idx < 0 {
            continue;
        }

        let selected_token = candidate_base_token(
            &runtime.side_infos[begin_side_idx],
            selected_neighbor_idx as usize,
        )?;
        let phase = result.stereo_component_phases[component_idx];
        if phase == UNKNOWN_COMPONENT_PHASE {
            continue;
        }
        flips[component_idx] = selected_token
            == if phase == STORED_COMPONENT_PHASE { "/" } else { "\\" };
    }

    Ok(flips)
}

#[allow(dead_code)]
fn coupled_begin_side_flips_for_result(
    runtime: &StereoWalkerRuntimeData,
    result: &SupportSearchResult,
) -> PyResult<Vec<bool>> {
    let component_count = runtime.isolated_components.len();
    let mut flips = vec![false; component_count];
    for component_idx in 0..component_count {
        if runtime.isolated_components[component_idx] {
            continue;
        }

        let begin_atom_idx = result.stereo_component_begin_atoms[component_idx];
        if begin_atom_idx < 0 {
            continue;
        }
        let Some(begin_side_idx) = runtime.side_ids_by_component[component_idx]
            .iter()
            .copied()
            .find(|&side_idx| runtime.side_infos[side_idx].endpoint_atom_idx == begin_atom_idx as usize)
        else {
            continue;
        };
        let begin_side = &runtime.side_infos[begin_side_idx];
        if begin_side.candidate_neighbors.len() > 2 {
            continue;
        }

        let selected_neighbor_idx = result.stereo_selected_neighbors[begin_side_idx];
        if selected_neighbor_idx < 0 {
            continue;
        }

        let stored_token = candidate_base_token(begin_side, selected_neighbor_idx as usize)?;
        let phase = result.stereo_component_phases[component_idx];
        if begin_side.candidate_neighbors.len() == 1 {
            flips[component_idx] = stored_token
                == if phase == STORED_COMPONENT_PHASE { "/" } else { "\\" };
            continue;
        }

        let first_neighbor_idx = result.stereo_first_emitted_candidates[begin_side_idx];
        if first_neighbor_idx < 0 {
            continue;
        }

        let resolved_token = if matches!(phase, UNKNOWN_COMPONENT_PHASE | STORED_COMPONENT_PHASE) {
            stored_token.clone()
        } else {
            flip_direction_token(&stored_token)?
        };
        let invert_selected_first = stored_token == "/";
        let mut desired_token = stored_token.clone();
        if (first_neighbor_idx == selected_neighbor_idx) == invert_selected_first {
            desired_token = flip_direction_token(&stored_token)?;
        }
        flips[component_idx] = desired_token != resolved_token;
    }
    Ok(flips)
}

#[allow(dead_code)]
fn rdkit_component_token_flip_adjustments_for_result(
    runtime: &StereoWalkerRuntimeData,
    result: &SupportSearchResult,
) -> Vec<bool> {
    let component_count = runtime.isolated_components.len();
    let mut adjustments = vec![false; component_count];
    for component_idx in 0..component_count {
        let begin_atom_idx = result.stereo_component_begin_atoms[component_idx];
        if begin_atom_idx < 0 {
            continue;
        }
        let Some(begin_side_idx) = runtime.side_ids_by_component[component_idx]
            .iter()
            .copied()
            .find(|&side_idx| runtime.side_infos[side_idx].endpoint_atom_idx == begin_atom_idx as usize)
        else {
            continue;
        };

        if begin_atom_idx as usize == runtime.root_idx
            && result.stereo_selected_orientations[begin_side_idx] == AFTER_ATOM_EDGE_ORIENTATION
        {
            adjustments[component_idx] = !adjustments[component_idx];
        }

        if runtime.isolated_components[component_idx] {
            continue;
        }

        let begin_side = &runtime.side_infos[begin_side_idx];
        if begin_side.candidate_neighbors.len() != 1 {
            continue;
        }

        let selected_neighbor_idx = result.stereo_selected_neighbors[begin_side_idx];
        if selected_neighbor_idx < 0 {
            continue;
        }

        let adjacent_two_side_idx = runtime.side_ids_by_component[component_idx]
            .iter()
            .copied()
            .find(|&side_idx| {
                let side_info = &runtime.side_infos[side_idx];
                side_info.candidate_neighbors.len() == 2
                    && side_info.endpoint_atom_idx == selected_neighbor_idx as usize
                    && side_info
                        .candidate_neighbors
                        .contains(&(begin_atom_idx as usize))
            });
        let Some(adjacent_two_side_idx) = adjacent_two_side_idx else {
            continue;
        };
        if selected_neighbor_idx as usize != runtime.root_idx {
            continue;
        }

        let first_neighbor_idx = result.stereo_first_emitted_candidates[adjacent_two_side_idx];
        if first_neighbor_idx >= 0 && first_neighbor_idx as usize != begin_atom_idx as usize {
            adjustments[component_idx] = !adjustments[component_idx];
        }
    }
    adjustments
}

#[allow(dead_code)]
fn resolve_support_result_smiles(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    result: &SupportSearchResult,
) -> PyResult<String> {
    let selected_neighbors = resolved_selected_neighbors_from_fields(
        runtime,
        &result.stereo_selected_neighbors,
        &result.stereo_first_emitted_candidates,
    );
    let resolved_result = SupportSearchResult {
        parts: result.parts.clone(),
        visited: result.visited.clone(),
        visited_count: result.visited_count,
        pending: result.pending.clone(),
        free_labels: result.free_labels.clone(),
        next_label: result.next_label,
        stereo_component_phases: result.stereo_component_phases.clone(),
        stereo_selected_neighbors: selected_neighbors,
        stereo_selected_orientations: result.stereo_selected_orientations.clone(),
        stereo_first_emitted_candidates: result.stereo_first_emitted_candidates.clone(),
        stereo_component_begin_atoms: result.stereo_component_begin_atoms.clone(),
    };

    let isolated_flips = isolated_component_flips_for_result(runtime, graph, &resolved_result)?;
    let coupled_begin_side_flips = coupled_begin_side_flips_for_result(runtime, &resolved_result)?;
    let rdkit_adjustments = rdkit_component_token_flip_adjustments_for_result(runtime, &resolved_result);
    let component_flips = isolated_flips
        .into_iter()
        .zip(coupled_begin_side_flips)
        .zip(rdkit_adjustments)
        .map(|((isolated_flip, coupled_flip), rdkit_adjustment)| {
            isolated_flip ^ coupled_flip ^ rdkit_adjustment
        })
        .collect::<Vec<_>>();

    let mut resolved_parts = Vec::new();
    for part in &resolved_result.parts {
        match part {
            Part::Deferred(deferred) if deferred.begin_idx >= 0 && deferred.end_idx >= 0 => {
                let begin_idx = deferred.begin_idx as usize;
                let end_idx = deferred.end_idx as usize;
                let edge = canonical_edge(begin_idx, end_idx);
                let mut active_tokens = Vec::<String>::new();
                for &side_idx in runtime.edge_to_side_ids.get(&edge).into_iter().flatten() {
                    let side_info = &runtime.side_infos[side_idx];
                    let selected_neighbor_idx = resolved_result.stereo_selected_neighbors[side_idx];
                    let edge_neighbor_idx = if begin_idx == side_info.endpoint_atom_idx {
                        end_idx
                    } else {
                        begin_idx
                    };
                    if selected_neighbor_idx == edge_neighbor_idx as isize {
                        active_tokens.push(emitted_candidate_token(side_info, begin_idx, end_idx)?);
                    }
                }
                if active_tokens.is_empty() {
                    resolved_parts.push(
                        graph
                            .bond_token(begin_idx, end_idx)
                            .ok_or_else(|| {
                                PyKeyError::new_err(format!(
                                    "No bond between atoms {} and {}",
                                    begin_idx, end_idx
                                ))
                            })?
                            .to_owned(),
                    );
                    continue;
                }
                let raw_token = active_tokens[0].clone();
                if active_tokens[1..].iter().any(|token| token != &raw_token) {
                    return Err(PyValueError::new_err(
                        "Carrier edge received conflicting stereo token assignments",
                    ));
                }
                resolved_parts.push(resolve_directional_token(
                    &raw_token,
                    deferred.component_idx,
                    &resolved_result.stereo_component_phases,
                    &component_flips,
                )?);
            }
            Part::Deferred(deferred) => {
                resolved_parts.push(resolve_directional_token(
                    &deferred.stored_token,
                    deferred.component_idx,
                    &resolved_result.stereo_component_phases,
                    &component_flips,
                )?);
            }
            Part::Literal(token) => resolved_parts.push(token.clone()),
            Part::RingLabel(label) => resolved_parts.push(ring_label_text(*label)),
            Part::OpenParen => resolved_parts.push("(".to_owned()),
            Part::CloseParen => resolved_parts.push(")".to_owned()),
        }
    }
    Ok(resolved_parts.concat())
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

    for &side_idx in runtime
        .edge_to_side_ids
        .get(&edge)
        .into_iter()
        .flatten()
    {
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
        state.stereo_component_token_flips[deferred.component_idx] = chosen_flip;
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
    let mut pending_now = base_state.pending.clone();
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
                successor.pending = pending_now.clone();
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
        return enter_atom_successors_without_bond_stereo(
            graph,
            &base_state,
            atom_idx,
            parent_idx,
            visited_now,
            visited_count_now,
            pending_now,
            closures_here,
            ordered_groups,
            is_chiral_atom,
        );
    }

    let mut successors = Vec::<(String, Vec<RootedConnectedStereoWalkerStateData>)>::new();
    let mut status = Ok(());
    for_each_cartesian_choice(&ordered_groups, &mut |chosen_children| {
        if status.is_err() {
            return;
        }

        let total_group_members = ordered_groups.iter().map(|group| group.len()).sum::<usize>();
        let opening_target_count = total_group_members.saturating_sub(chosen_children.len());
        let mut ring_actions =
            Vec::with_capacity(closures_here.len() + opening_target_count);
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

        permutations_copy_distinct(&ring_actions, &mut |ring_action_order| {
            if status.is_err() {
                return;
            }

            let outcome: PyResult<()> = (|| {
                let mut current_pending = pending_now.clone();
                let mut current_free = base_state.free_labels.clone();
                let mut current_next = base_state.next_label;
                let mut current_component_phases = base_state.stereo_component_phases.to_vec();
                let mut current_selected_neighbors =
                    base_state.stereo_selected_neighbors.to_vec();
                let mut current_selected_orientations =
                    base_state.stereo_selected_orientations.to_vec();
                let mut current_first_emitted_candidates =
                    base_state.stereo_first_emitted_candidates.to_vec();
                let mut current_component_begin_atoms =
                    base_state.stereo_component_begin_atoms.to_vec();
                let mut current_ring_actions = Vec::<WalkerAction>::with_capacity(
                    closures_here.len() * 2 + opening_target_count,
                );
                let mut labels_freed_after_atom = Vec::<usize>::with_capacity(closures_here.len());
                let mut ring_neighbor_order = is_chiral_atom.then(Vec::<usize>::new);

                for ring_action in ring_action_order {
                    match *ring_action {
                        RingAction::Close(closure_idx) => {
                            let closure = &closures_here[closure_idx];
                            let (
                                bond_part,
                                updated_neighbors,
                                updated_orientations,
                                updated_first_candidates,
                            ) = emitted_edge_part(
                                graph,
                                &runtime.side_infos,
                                &runtime.side_ids_by_component,
                                &runtime.edge_to_side_ids,
                                &current_component_phases,
                                &current_selected_neighbors,
                                &current_selected_orientations,
                                &current_first_emitted_candidates,
                                &current_component_begin_atoms,
                                &runtime.isolated_components,
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
                            let (updated_phases, updated_begin_atoms) = component_phases_after_edge(
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
                            pending: current_pending.clone(),
                            free_labels: current_free.clone(),
                            next_label: current_next,
                            stereo_component_phases: current_component_phases.clone(),
                            stereo_selected_neighbors: current_selected_neighbors.clone(),
                            stereo_selected_orientations: current_selected_orientations.clone(),
                            stereo_first_emitted_candidates: current_first_emitted_candidates
                                .clone(),
                            stereo_component_begin_atoms: current_component_begin_atoms.clone(),
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

#[allow(clippy::too_many_arguments)]
fn enter_atom_successors_without_bond_stereo(
    graph: &PreparedSmilesGraphData,
    base_state: &RootedConnectedStereoWalkerStateData,
    atom_idx: usize,
    parent_idx: Option<usize>,
    visited_now: Arc<[bool]>,
    visited_count_now: usize,
    pending_now: Vec<(usize, Vec<PendingRing>)>,
    closures_here: Vec<PendingRing>,
    ordered_groups: Vec<Vec<usize>>,
    is_chiral_atom: bool,
) -> PyResult<BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>> {
    let mut successors = Vec::<(String, Vec<RootedConnectedStereoWalkerStateData>)>::new();
    let mut status = Ok(());

    for_each_cartesian_choice(&ordered_groups, &mut |chosen_children| {
        if status.is_err() {
            return;
        }

        let total_group_members = ordered_groups.iter().map(|group| group.len()).sum::<usize>();
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

        permutations_copy_distinct(&ring_actions, &mut |ring_action_order| {
            if status.is_err() {
                return;
            }

            let outcome: PyResult<()> = (|| {
                let mut current_pending = pending_now.clone();
                let mut current_free = base_state.free_labels.clone();
                let mut current_next = base_state.next_label;
                let mut current_ring_actions =
                    Vec::<WalkerAction>::with_capacity(closures_here.len() * 2 + opening_target_count);
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
                            pending: current_pending.clone(),
                            free_labels: current_free.clone(),
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

fn process_children_successors_by_token(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    parent_idx: usize,
    child_order: Arc<[usize]>,
    next_branch_index: usize,
    require_completable: bool,
    completion_cache: &mut StereoCompletionCache,
) -> PyResult<BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>> {
    if child_order.is_empty() {
        return Ok(BTreeMap::new());
    }
    if runtime.side_infos.is_empty() {
        return process_children_successors_without_bond_stereo(
            runtime,
            graph,
            state,
            parent_idx,
            child_order,
            next_branch_index,
            require_completable,
            completion_cache,
        );
    }
    let branch_count = child_order.len().saturating_sub(1);

    if next_branch_index < branch_count {
        let child_idx = child_order[next_branch_index];
        let mut successor = state.clone();
        successor.action_stack.pop();
        push_char_token(&mut successor.prefix, '(');
        let (current_phases, current_begin_atoms) = eager_component_phases_for_child_order(
            runtime,
            graph,
            &successor.stereo_component_phases,
            &successor.stereo_component_begin_atoms,
            &successor.stereo_selected_neighbors,
            parent_idx,
            child_order.as_ref(),
        )?;
        let (
            current_selected_neighbors,
            current_selected_orientations,
            current_first_candidates,
        ) = eager_begin_side_child_order_state(
            runtime,
            &current_begin_atoms,
            &successor.stereo_selected_neighbors,
            &successor.stereo_selected_orientations,
            &successor.stereo_first_emitted_candidates,
            parent_idx,
            child_order.as_ref(),
        );
        let (edge_part, updated_neighbors, updated_orientations, updated_first_candidates) =
            emitted_edge_part(
                graph,
                &runtime.side_infos,
                &runtime.side_ids_by_component,
                &runtime.edge_to_side_ids,
                &current_phases,
                &current_selected_neighbors,
                &current_selected_orientations,
                &current_first_candidates,
                &current_begin_atoms,
                &runtime.isolated_components,
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
        successor.stereo_selected_neighbors = updated_neighbors;
        successor.stereo_selected_orientations = updated_orientations;
        successor.stereo_first_emitted_candidates = updated_first_candidates;
        successor.stereo_component_phases = updated_phases;
        successor.stereo_component_begin_atoms = updated_begin_atoms;
        if next_branch_index + 1 < child_order.len() {
            successor.action_stack.push(WalkerAction::ProcessChildren {
                parent_idx,
                child_order: child_order.clone(),
                next_branch_index: next_branch_index + 1,
            });
        }
        successor.action_stack.push(WalkerAction::EmitCloseParen);
        successor.action_stack.push(WalkerAction::EnterAtom {
            atom_idx: child_idx,
            parent_idx: Some(parent_idx),
        });
        if let Some(action) = part_to_action(edge_part) {
            successor.action_stack.push(action);
        }
        normalize_component_token_flips(runtime, graph, &mut successor)?;
        return Ok(BTreeMap::from([("(".to_owned(), vec![successor])]));
    }

    let child_idx = child_order[child_order.len() - 1];
    let (current_phases, current_begin_atoms) = eager_component_phases_for_child_order(
        runtime,
        graph,
        &state.stereo_component_phases,
        &state.stereo_component_begin_atoms,
        &state.stereo_selected_neighbors,
        parent_idx,
        child_order.as_ref(),
    )?;
    let (
        current_selected_neighbors,
        current_selected_orientations,
        current_first_candidates,
    ) = eager_begin_side_child_order_state(
        runtime,
        &current_begin_atoms,
        &state.stereo_selected_neighbors,
        &state.stereo_selected_orientations,
        &state.stereo_first_emitted_candidates,
        parent_idx,
        child_order.as_ref(),
    );
    let (edge_part, updated_neighbors, updated_orientations, updated_first_candidates) =
        emitted_edge_part(
            graph,
            &runtime.side_infos,
            &runtime.side_ids_by_component,
            &runtime.edge_to_side_ids,
            &current_phases,
            &current_selected_neighbors,
            &current_selected_orientations,
            &current_first_candidates,
            &current_begin_atoms,
            &runtime.isolated_components,
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
    let mut base_state = state.clone();
    base_state.action_stack.pop();
    base_state.stereo_selected_neighbors = updated_neighbors;
    base_state.stereo_selected_orientations = updated_orientations;
    base_state.stereo_first_emitted_candidates = updated_first_candidates;
    base_state.stereo_component_phases = updated_phases;
    base_state.stereo_component_begin_atoms = updated_begin_atoms;
    normalize_component_token_flips(runtime, graph, &mut base_state)?;

    match edge_part {
        Part::Literal(token) if token.is_empty() => {
            base_state.action_stack.push(WalkerAction::EnterAtom {
                atom_idx: child_idx,
                parent_idx: Some(parent_idx),
            });
            successors_by_token_stereo_impl(
                runtime,
                graph,
                &base_state,
                require_completable,
                completion_cache,
            )
        }
        Part::Literal(token) => {
            push_literal_token(&mut base_state.prefix, &token);
            base_state.action_stack.push(WalkerAction::EnterAtom {
                atom_idx: child_idx,
                parent_idx: Some(parent_idx),
            });
            Ok(BTreeMap::from([(token, vec![base_state])]))
        }
        Part::RingLabel(label) => {
            push_ring_label(&mut base_state.prefix, label);
            base_state.action_stack.push(WalkerAction::EnterAtom {
                atom_idx: child_idx,
                parent_idx: Some(parent_idx),
            });
            let token = ring_label_text(label);
            Ok(BTreeMap::from([(token, vec![base_state])]))
        }
        Part::OpenParen => {
            push_char_token(&mut base_state.prefix, '(');
            base_state.action_stack.push(WalkerAction::EnterAtom {
                atom_idx: child_idx,
                parent_idx: Some(parent_idx),
            });
            Ok(BTreeMap::from([("(".to_owned(), vec![base_state])]))
        }
        Part::CloseParen => {
            push_char_token(&mut base_state.prefix, ')');
            base_state.action_stack.push(WalkerAction::EnterAtom {
                atom_idx: child_idx,
                parent_idx: Some(parent_idx),
            });
            Ok(BTreeMap::from([(")".to_owned(), vec![base_state])]))
        }
        Part::Deferred(deferred) => {
            let mut out = BTreeMap::<String, Vec<RootedConnectedStereoWalkerStateData>>::new();
            for token in deferred_token_support(runtime, graph, &base_state, &deferred)? {
                let mut successor = base_state.clone();
                if let Err(err) =
                    commit_deferred_token_choice(runtime, graph, &mut successor, &deferred, &token)
                {
                    if require_completable {
                        return Err(err);
                    }
                    continue;
                }
                push_literal_token(&mut successor.prefix, &token);
                successor.action_stack.push(WalkerAction::EnterAtom {
                    atom_idx: child_idx,
                    parent_idx: Some(parent_idx),
                });
                if !require_completable
                    || can_complete_from_stereo_state_memo(
                        runtime,
                        graph,
                        &successor,
                        completion_cache,
                    )
                {
                    out.entry(token).or_default().push(successor);
                }
            }
            Ok(out)
        }
    }
}

fn process_children_successors_without_bond_stereo(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    parent_idx: usize,
    child_order: Arc<[usize]>,
    next_branch_index: usize,
    require_completable: bool,
    completion_cache: &mut StereoCompletionCache,
) -> PyResult<BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>> {
    let branch_count = child_order.len().saturating_sub(1);

    if next_branch_index < branch_count {
        let child_idx = child_order[next_branch_index];
        let mut successor = state.clone();
        successor.action_stack.pop();
        push_char_token(&mut successor.prefix, '(');
        if next_branch_index + 1 < child_order.len() {
            successor.action_stack.push(WalkerAction::ProcessChildren {
                parent_idx,
                child_order: child_order.clone(),
                next_branch_index: next_branch_index + 1,
            });
        }
        successor.action_stack.push(WalkerAction::EmitCloseParen);
        successor.action_stack.push(WalkerAction::EnterAtom {
            atom_idx: child_idx,
            parent_idx: Some(parent_idx),
        });
        let bond_token = graph
            .bond_token(parent_idx, child_idx)
            .ok_or_else(|| {
                PyKeyError::new_err(format!(
                    "No bond between atoms {parent_idx} and {child_idx}"
                ))
            })?
            .to_owned();
        if !bond_token.is_empty() {
            successor
                .action_stack
                .push(WalkerAction::EmitLiteral(bond_token));
        }
        return Ok(BTreeMap::from([("(".to_owned(), vec![successor])]));
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
    let mut base_state = state.clone();
    base_state.action_stack.pop();

    if bond_token.is_empty() {
        base_state.action_stack.push(WalkerAction::EnterAtom {
            atom_idx: child_idx,
            parent_idx: Some(parent_idx),
        });
        return successors_by_token_stereo_impl(
            runtime,
            graph,
            &base_state,
            require_completable,
            completion_cache,
        );
    }

    push_literal_token(&mut base_state.prefix, &bond_token);
    base_state.action_stack.push(WalkerAction::EnterAtom {
        atom_idx: child_idx,
        parent_idx: Some(parent_idx),
    });
    Ok(BTreeMap::from([(bond_token, vec![base_state])]))
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
            runtime,
            graph,
            state,
            parent_idx,
            child_order,
            next_branch_index,
            require_completable,
            completion_cache,
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
                    Ok(successors) => extend_linear_structural_transitions(&mut expanded, successors),
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
    Ok(candidates
        .into_iter()
        .next()
        .expect("chosen token should have at least one successor"))
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
    Ok(take_choice_or_err(&mut choices, chosen_idx)?
        .into_iter()
        .next()
        .expect("choice should advance to exactly one successor state"))
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

#[allow(dead_code)]
fn enumerate_support_from_stereo_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: RootedConnectedStereoWalkerStateData,
    out: &mut BTreeSet<String>,
) -> PyResult<()> {
    let successors = successors_by_token_stereo_raw(runtime, graph, &state)?;
    if successors.is_empty() {
        if is_complete_terminal_stereo_state(graph, &state) {
            out.insert(state.prefix.to_string());
        }
        return Ok(());
    }

    for successor_group in successors.into_values() {
        for successor in successor_group {
            enumerate_support_from_stereo_state(runtime, graph, successor, out)?;
        }
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
        Ok(successors_by_token_stereo(&self.runtime, &self.graph, &state.data)?
            .into_keys()
            .collect())
    }

    fn next_choice_texts(
        &self,
        state: &PyRootedConnectedStereoWalkerState,
    ) -> PyResult<Vec<String>> {
        validate_stereo_state_shape(&self.runtime, &self.graph, &state.data)?;
        let mut choices = Vec::new();
        for (token, successors) in successors_by_token_stereo(&self.runtime, &self.graph, &state.data)?
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
        let successors = take_transition_or_err(&mut choices, chosen_token)?
            .into_iter()
            .next()
            .expect("chosen token should have at least one successor state");
        Ok(PyRootedConnectedStereoWalkerState { data: successors })
    }

    fn advance_choice(
        &self,
        state: &PyRootedConnectedStereoWalkerState,
        chosen_idx: usize,
    ) -> PyResult<PyRootedConnectedStereoWalkerState> {
        validate_stereo_state_shape(&self.runtime, &self.graph, &state.data)?;
        let mut choices = Vec::new();
        for (token, successors) in successors_by_token_stereo(&self.runtime, &self.graph, &state.data)?
        {
            for successor in successors {
                choices.push(DecoderChoice {
                    text: token.clone(),
                    next_frontier: vec![successor],
                });
            }
        }
        Ok(PyRootedConnectedStereoWalkerState {
            data: take_choice_or_err(&mut choices, chosen_idx)?
                .into_iter()
                .next()
                .expect("choice should advance to exactly one successor state"),
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
    runtime: Option<Arc<StereoWalkerRuntimeData>>,
    frontier: Vec<RootedConnectedStereoWalkerStateData>,
    merged_branches: Option<Vec<StereoDecoderBranch>>,
    cached_choices: Option<Vec<DecoderChoice<RootedConnectedStereoWalkerStateData>>>,
}

impl PyRootedConnectedStereoDecoder {
    fn from_single(
        graph: Arc<PreparedSmilesGraphData>,
        runtime: Arc<StereoWalkerRuntimeData>,
        frontier: Vec<RootedConnectedStereoWalkerStateData>,
    ) -> Self {
        Self {
            graph,
            runtime: Some(runtime),
            frontier,
            merged_branches: None,
            cached_choices: None,
        }
    }

    fn from_merged(graph: Arc<PreparedSmilesGraphData>, branches: Vec<StereoDecoderBranch>) -> Self {
        if let [branch] = branches.as_slice() {
            return Self::from_single(graph, branch.runtime.clone(), branch.frontier.clone());
        }
        Self {
            graph,
            runtime: None,
            frontier: Vec::new(),
            merged_branches: Some(branches),
            cached_choices: None,
        }
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
            if let Some((_, grouped)) = buckets.iter_mut().find(|(existing, _)| *existing == token) {
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
        if let Some(branches) = &self.merged_branches {
            return Ok(merged_stereo_grouped_successors(self.graph.clone(), branches)?
                .into_iter()
                .map(|(token, _)| token)
                .collect());
        }
        if self.cached_choices.is_none() {
            self.cached_choices = Some(frontier_choices_for_stereo(
                self.runtime
                    .as_ref()
                    .expect("single decoder runtime should be present")
                    .as_ref(),
                self.graph.as_ref(),
                &self.frontier,
            )?);
        }
        Ok(grouped_choice_texts(
            self.cached_choices
                .as_ref()
                .expect("cache should be populated"),
        ))
    }

    fn advance_token(&mut self, chosen_token: &str) -> PyResult<()> {
        if let Some(branches) = &self.merged_branches {
            let successors = merged_stereo_grouped_successors(self.graph.clone(), branches)?;
            let (_, successor) = successors
                .into_iter()
                .find(|(token, _)| token == chosen_token)
                .ok_or_else(|| PyKeyError::new_err(format!("Token {chosen_token:?} is not available")))?;
            *self = successor;
            return Ok(());
        }
        let choices = match self.cached_choices.take() {
            Some(choices) => choices,
            None => frontier_choices_for_stereo(
                self.runtime
                    .as_ref()
                    .expect("single decoder runtime should be present")
                    .as_ref(),
                self.graph.as_ref(),
                &self.frontier,
            )?,
        };
        self.frontier = take_grouped_choices_or_err(choices, chosen_token)?;
        Ok(())
    }

    fn next_choice_texts(&mut self) -> PyResult<Vec<String>> {
        if let Some(branches) = &self.merged_branches {
            return Ok(merged_stereo_choice_successors(self.graph.clone(), branches)?
                .into_iter()
                .map(|(token, _)| token)
                .collect());
        }
        if self.cached_choices.is_none() {
            self.cached_choices = Some(frontier_choices_for_stereo(
                self.runtime
                    .as_ref()
                    .expect("single decoder runtime should be present")
                    .as_ref(),
                self.graph.as_ref(),
                &self.frontier,
            )?);
        }
        Ok(choice_texts(
            self.cached_choices
                .as_ref()
                .expect("cache should be populated"),
        ))
    }

    fn advance_choice(&mut self, chosen_idx: usize) -> PyResult<()> {
        if let Some(branches) = &self.merged_branches {
            let mut successors = merged_stereo_choice_successors(self.graph.clone(), branches)?;
            if chosen_idx >= successors.len() {
                return Err(PyKeyError::new_err(format!(
                    "Choice index {chosen_idx} is not available; choice_count={}",
                    successors.len()
                )));
            }
            *self = successors.swap_remove(chosen_idx).1;
            return Ok(());
        }
        let mut choices = match self.cached_choices.take() {
            Some(choices) => choices,
            None => frontier_choices_for_stereo(
                self.runtime
                    .as_ref()
                    .expect("single decoder runtime should be present")
                    .as_ref(),
                self.graph.as_ref(),
                &self.frontier,
            )?,
        };
        self.frontier = take_choice_or_err(&mut choices, chosen_idx)?;
        Ok(())
    }

    fn choice_successors(&self) -> PyResult<Vec<(String, Self)>> {
        if let Some(branches) = &self.merged_branches {
            return merged_stereo_choice_successors(self.graph.clone(), branches);
        }
        Ok(frontier_choice_successors_for_stereo(
            self.runtime
                .as_ref()
                .expect("single decoder runtime should be present")
                .as_ref(),
            self.graph.as_ref(),
            &self.frontier,
        )?
        .into_iter()
        .map(|(token, successor)| {
            (
                token,
                Self::from_single(
                    self.graph.clone(),
                    self.runtime
                        .as_ref()
                        .expect("single decoder runtime should be present")
                        .clone(),
                    vec![successor],
                ),
            )
        })
        .collect())
    }

    fn grouped_successors(&self) -> PyResult<Vec<(String, Self)>> {
        if let Some(branches) = &self.merged_branches {
            return merged_stereo_grouped_successors(self.graph.clone(), branches);
        }
        Ok(frontier_transitions_for_stereo_linear(
            self.runtime
                .as_ref()
                .expect("single decoder runtime should be present")
                .as_ref(),
            self.graph.as_ref(),
            &self.frontier,
        )?
        .into_iter()
        .map(|(token, frontier)| {
            (
                token,
                Self::from_single(
                    self.graph.clone(),
                    self.runtime
                        .as_ref()
                        .expect("single decoder runtime should be present")
                        .clone(),
                    frontier,
                ),
            )
        })
        .collect())
    }

    fn prefix(&self) -> String {
        if let Some(branches) = &self.merged_branches {
            merged_stereo_prefix(branches)
        } else {
            stereo_frontier_prefix(&self.frontier)
        }
    }

    fn cache_key(&self) -> String {
        if let Some(branches) = &self.merged_branches {
            merged_stereo_cache_key(branches)
        } else {
            format!("{:?}", self.frontier)
        }
    }

    fn is_terminal(&self) -> PyResult<bool> {
        if let Some(branches) = &self.merged_branches {
            return merged_stereo_is_terminal(self.graph.as_ref(), branches);
        }
        if let Some(choices) = &self.cached_choices {
            Ok(choices.is_empty())
        } else {
            stereo_frontier_is_terminal(
                self.runtime
                    .as_ref()
                    .expect("single decoder runtime should be present")
                    .as_ref(),
                self.graph.as_ref(),
                &self.frontier,
            )
        }
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RootedConnectedStereoDecoder(prefix={:?}, frontier_size={}, atom_count={})",
            self.prefix(),
            self.frontier.len(),
            self.graph.atom_count(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use pyo3::types::{PyAnyMethods, PyDictMethods};
    use pyo3::Python;

    use super::{
        advance_stereo_choice_state, advance_stereo_token_state, build_walker_runtime,
        check_supported_stereo_writer_surface, choices_for_stereo_state,
        enumerate_rooted_connected_stereo_smiles_support,
        enumerate_rooted_connected_stereo_smiles_support_native, enumerate_support_from_stereo_state,
        initial_stereo_state_for_root, is_terminal_stereo_state, next_token_support_for_stereo_state,
        validate_root_idx,
    };
    use crate::prepared_graph::{
        PreparedSmilesGraphData, CONNECTED_STEREO_SURFACE, PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
    };

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct StereoDecoderStats {
        state_count: usize,
        terminal_state_count: usize,
        enter_atom_state_count: usize,
        chiral_enter_atom_state_count: usize,
        process_children_state_count: usize,
        emit_deferred_state_count: usize,
        emit_literal_state_count: usize,
        emit_ring_label_state_count: usize,
        emit_close_paren_state_count: usize,
        total_prefix_len: usize,
        max_prefix_len: usize,
        total_action_stack_len: usize,
        max_action_stack_len: usize,
        total_action_stack_payload_bytes: usize,
        max_action_stack_payload_bytes: usize,
        total_pending_sites: usize,
        max_pending_sites: usize,
        total_pending_rings: usize,
        max_pending_rings: usize,
        total_enter_atom_bucket_count: usize,
        max_enter_atom_bucket_count: usize,
        total_enter_atom_successor_count: usize,
        max_enter_atom_successor_count: usize,
        total_enter_atom_added_action_count: usize,
        max_enter_atom_added_action_count: usize,
        total_process_children_successor_count: usize,
        max_process_children_successor_count: usize,
        total_process_children_added_action_count: usize,
        max_process_children_added_action_count: usize,
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct StereoFieldDiffStats {
        equal_pairs: usize,
        visited: usize,
        visited_count: usize,
        pending: usize,
        free_labels: usize,
        next_label: usize,
        component_phases: usize,
        selected_neighbors: usize,
        selected_orientations: usize,
        first_emitted_candidates: usize,
        component_begin_atoms: usize,
        component_token_flips: usize,
        action_stack: usize,
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct StereoProcessChildrenShapeStats {
        state_count: usize,
        branch_state_count: usize,
        final_state_count: usize,
        edge_without_side_ids: usize,
        empty_literal_edge_part: usize,
        nonempty_literal_edge_part: usize,
        deferred_edge_part: usize,
        ring_label_edge_part: usize,
        open_paren_edge_part: usize,
        close_paren_edge_part: usize,
        eager_state_changed_count: usize,
        unchanged_state_update_count: usize,
    }

    fn first_stereo_structure_difference(
        left: &super::RootedConnectedStereoWalkerStateData,
        right: &super::RootedConnectedStereoWalkerStateData,
    ) -> &'static str {
        if left.visited != right.visited {
            return "visited";
        }
        if left.visited_count != right.visited_count {
            return "visited_count";
        }
        if left.pending != right.pending {
            return "pending";
        }
        if left.free_labels != right.free_labels {
            return "free_labels";
        }
        if left.next_label != right.next_label {
            return "next_label";
        }
        if left.stereo_component_phases != right.stereo_component_phases {
            return "component_phases";
        }
        if left.stereo_selected_neighbors != right.stereo_selected_neighbors {
            return "selected_neighbors";
        }
        if left.stereo_selected_orientations != right.stereo_selected_orientations {
            return "selected_orientations";
        }
        if left.stereo_first_emitted_candidates != right.stereo_first_emitted_candidates {
            return "first_emitted_candidates";
        }
        if left.stereo_component_begin_atoms != right.stereo_component_begin_atoms {
            return "component_begin_atoms";
        }
        if left.stereo_component_token_flips != right.stereo_component_token_flips {
            return "component_token_flips";
        }
        if left.action_stack != right.action_stack {
            return "action_stack";
        }
        "equal"
    }

    fn collect_enter_atom_field_diff_stats(
        runtime: &super::StereoWalkerRuntimeData,
        graph: &PreparedSmilesGraphData,
        state: &super::RootedConnectedStereoWalkerStateData,
        stats: &mut StereoFieldDiffStats,
    ) -> pyo3::PyResult<()> {
        let Some(super::WalkerAction::EnterAtom { atom_idx, parent_idx }) = state.action_stack.last()
        else {
            return Ok(());
        };
        let raw_successors =
            super::enter_atom_successors_by_token(runtime, graph, state, *atom_idx, *parent_idx)?;
        for successors in raw_successors.into_values() {
            for left_idx in 0..successors.len() {
                for right_idx in left_idx + 1..successors.len() {
                    match first_stereo_structure_difference(
                        &successors[left_idx],
                        &successors[right_idx],
                    ) {
                        "visited" => stats.visited += 1,
                        "visited_count" => stats.visited_count += 1,
                        "pending" => stats.pending += 1,
                        "free_labels" => stats.free_labels += 1,
                        "next_label" => stats.next_label += 1,
                        "component_phases" => stats.component_phases += 1,
                        "selected_neighbors" => stats.selected_neighbors += 1,
                        "selected_orientations" => stats.selected_orientations += 1,
                        "first_emitted_candidates" => stats.first_emitted_candidates += 1,
                        "component_begin_atoms" => stats.component_begin_atoms += 1,
                        "component_token_flips" => stats.component_token_flips += 1,
                        "action_stack" => stats.action_stack += 1,
                        "equal" => stats.equal_pairs += 1,
                        _ => unreachable!(),
                    }
                }
            }
        }
        Ok(())
    }

    fn collect_process_children_shape_stats(
        runtime: &super::StereoWalkerRuntimeData,
        graph: &PreparedSmilesGraphData,
        state: &super::RootedConnectedStereoWalkerStateData,
        stats: &mut StereoProcessChildrenShapeStats,
    ) -> pyo3::PyResult<()> {
        let Some(
            super::WalkerAction::ProcessChildren {
                parent_idx,
                child_order,
                next_branch_index,
            },
        ) = state.action_stack.last()
        else {
            return Ok(());
        };
        stats.state_count += 1;
        let branch_count = child_order.len().saturating_sub(1);
        let child_idx = if *next_branch_index < branch_count {
            stats.branch_state_count += 1;
            child_order[*next_branch_index]
        } else {
            stats.final_state_count += 1;
            child_order[child_order.len() - 1]
        };

        if !runtime
            .edge_to_side_ids
            .contains_key(&super::canonical_edge(*parent_idx, child_idx))
        {
            stats.edge_without_side_ids += 1;
        }

        let (current_phases, current_begin_atoms) = super::eager_component_phases_for_child_order(
            runtime,
            graph,
            &state.stereo_component_phases,
            &state.stereo_component_begin_atoms,
            &state.stereo_selected_neighbors,
            *parent_idx,
            child_order.as_ref(),
        )?;
        let (
            current_selected_neighbors,
            current_selected_orientations,
            current_first_candidates,
        ) = super::eager_begin_side_child_order_state(
            runtime,
            &current_begin_atoms,
            &state.stereo_selected_neighbors,
            &state.stereo_selected_orientations,
            &state.stereo_first_emitted_candidates,
            *parent_idx,
            child_order.as_ref(),
        );
        if current_phases != state.stereo_component_phases
            || current_begin_atoms != state.stereo_component_begin_atoms
            || current_selected_neighbors != state.stereo_selected_neighbors
            || current_selected_orientations != state.stereo_selected_orientations
            || current_first_candidates != state.stereo_first_emitted_candidates
        {
            stats.eager_state_changed_count += 1;
        }
        let (edge_part, updated_neighbors, updated_orientations, updated_first_candidates) =
            super::emitted_edge_part(
                graph,
                &runtime.side_infos,
                &runtime.side_ids_by_component,
                &runtime.edge_to_side_ids,
                &current_phases,
                &current_selected_neighbors,
                &current_selected_orientations,
                &current_first_candidates,
                &current_begin_atoms,
                &runtime.isolated_components,
                *parent_idx,
                child_idx,
            )?;
        let (updated_phases, updated_begin_atoms) = super::component_phases_after_edge(
            graph,
            &runtime.stereo_component_ids,
            &runtime.isolated_components,
            &current_phases,
            &current_begin_atoms,
            *parent_idx,
            child_idx,
        )?;
        let (updated_neighbors, updated_orientations) = super::force_known_begin_side_selection(
            runtime,
            &updated_phases,
            &updated_begin_atoms,
            &updated_neighbors,
            &updated_orientations,
        );
        let (updated_phases, updated_begin_atoms) =
            super::defer_coupled_component_phase_if_begin_side_is_unresolved(
                runtime,
                graph,
                &updated_phases,
                &updated_begin_atoms,
                &updated_neighbors,
                *parent_idx,
                child_idx,
            )?;

        match &edge_part {
            super::Part::Literal(token) if token.is_empty() => {
                stats.empty_literal_edge_part += 1;
            }
            super::Part::Literal(_) => {
                stats.nonempty_literal_edge_part += 1;
            }
            super::Part::Deferred(_) => {
                stats.deferred_edge_part += 1;
            }
            super::Part::RingLabel(_) => {
                stats.ring_label_edge_part += 1;
            }
            super::Part::OpenParen => {
                stats.open_paren_edge_part += 1;
            }
            super::Part::CloseParen => {
                stats.close_paren_edge_part += 1;
            }
        }

        if updated_neighbors == current_selected_neighbors
            && updated_orientations == current_selected_orientations
            && updated_first_candidates == current_first_candidates
            && updated_phases == current_phases
            && updated_begin_atoms == current_begin_atoms
        {
            stats.unchanged_state_update_count += 1;
        }

        Ok(())
    }

    impl StereoDecoderStats {
        fn average_action_stack_len(&self) -> f64 {
            if self.state_count == 0 {
                0.0
            } else {
                self.total_action_stack_len as f64 / self.state_count as f64
            }
        }

        fn average_prefix_len(&self) -> f64 {
            if self.state_count == 0 {
                0.0
            } else {
                self.total_prefix_len as f64 / self.state_count as f64
            }
        }

        fn average_action_stack_payload_bytes(&self) -> f64 {
            if self.state_count == 0 {
                0.0
            } else {
                self.total_action_stack_payload_bytes as f64 / self.state_count as f64
            }
        }

        fn average_pending_sites(&self) -> f64 {
            if self.state_count == 0 {
                0.0
            } else {
                self.total_pending_sites as f64 / self.state_count as f64
            }
        }

        fn average_pending_rings(&self) -> f64 {
            if self.state_count == 0 {
                0.0
            } else {
                self.total_pending_rings as f64 / self.state_count as f64
            }
        }

        fn average_enter_atom_successor_count(&self) -> f64 {
            if self.enter_atom_state_count == 0 {
                0.0
            } else {
                self.total_enter_atom_successor_count as f64 / self.enter_atom_state_count as f64
            }
        }

        fn average_enter_atom_added_action_count(&self) -> f64 {
            if self.total_enter_atom_successor_count == 0 {
                0.0
            } else {
                self.total_enter_atom_added_action_count as f64
                    / self.total_enter_atom_successor_count as f64
            }
        }

        fn average_process_children_successor_count(&self) -> f64 {
            if self.process_children_state_count == 0 {
                0.0
            } else {
                self.total_process_children_successor_count as f64
                    / self.process_children_state_count as f64
            }
        }

        fn average_process_children_added_action_count(&self) -> f64 {
            if self.total_process_children_successor_count == 0 {
                0.0
            } else {
                self.total_process_children_added_action_count as f64
                    / self.total_process_children_successor_count as f64
            }
        }

    }

    fn collect_stereo_decoder_stats(
        runtime: &super::StereoWalkerRuntimeData,
        graph: &PreparedSmilesGraphData,
        state: super::RootedConnectedStereoWalkerStateData,
        completion_cache: &mut super::StereoCompletionCache,
        stats: &mut StereoDecoderStats,
    ) -> pyo3::PyResult<()> {
        stats.state_count += 1;
        stats.total_prefix_len += state.prefix.len();
        stats.max_prefix_len = stats.max_prefix_len.max(state.prefix.len());
        stats.total_action_stack_len += state.action_stack.len();
        stats.max_action_stack_len = stats.max_action_stack_len.max(state.action_stack.len());
        let action_stack_payload_bytes = state
            .action_stack
            .iter()
            .map(|action| match action {
                super::WalkerAction::EmitLiteral(token) => token.len(),
                super::WalkerAction::EmitDeferred(deferred) => deferred.stored_token.len(),
                _ => 0,
            })
            .sum::<usize>();
        stats.total_action_stack_payload_bytes += action_stack_payload_bytes;
        stats.max_action_stack_payload_bytes = stats
            .max_action_stack_payload_bytes
            .max(action_stack_payload_bytes);
        stats.total_pending_sites += state.pending.len();
        stats.max_pending_sites = stats.max_pending_sites.max(state.pending.len());
        let pending_ring_count = state.pending.iter().map(|(_, rings)| rings.len()).sum::<usize>();
        stats.total_pending_rings += pending_ring_count;
        stats.max_pending_rings = stats.max_pending_rings.max(pending_ring_count);

        match state.action_stack.last() {
            None => {
                stats.terminal_state_count += 1;
                return Ok(());
            }
            Some(super::WalkerAction::EnterAtom { atom_idx, parent_idx }) => {
                stats.enter_atom_state_count += 1;
                if graph.atom_chiral_tags[*atom_idx] != "CHI_UNSPECIFIED" {
                    stats.chiral_enter_atom_state_count += 1;
                }
                let raw_successors = super::enter_atom_successors_by_token(
                    runtime,
                    graph,
                    &state,
                    *atom_idx,
                    *parent_idx,
                )?;
                stats.total_enter_atom_bucket_count += raw_successors.len();
                stats.max_enter_atom_bucket_count = stats
                    .max_enter_atom_bucket_count
                    .max(raw_successors.len());
                let raw_successor_count = raw_successors.values().map(Vec::len).sum::<usize>();
                stats.total_enter_atom_successor_count += raw_successor_count;
                stats.max_enter_atom_successor_count = stats
                    .max_enter_atom_successor_count
                    .max(raw_successor_count);
                let base_len = state.action_stack.len().saturating_sub(1);
                for successors in raw_successors.values() {
                    for successor in successors {
                        let added = successor.action_stack.len().saturating_sub(base_len);
                        stats.total_enter_atom_added_action_count += added;
                        stats.max_enter_atom_added_action_count =
                            stats.max_enter_atom_added_action_count.max(added);
                    }
                }
            }
            Some(super::WalkerAction::ProcessChildren {
                parent_idx,
                child_order,
                next_branch_index,
            }) => {
                stats.process_children_state_count += 1;
                let raw_successors = super::process_children_successors_by_token(
                    runtime,
                    graph,
                    &state,
                    *parent_idx,
                    child_order.clone(),
                    *next_branch_index,
                    true,
                    completion_cache,
                )?;
                let raw_successor_count = raw_successors.values().map(Vec::len).sum::<usize>();
                stats.total_process_children_successor_count += raw_successor_count;
                stats.max_process_children_successor_count = stats
                    .max_process_children_successor_count
                    .max(raw_successor_count);
                let base_len = state.action_stack.len().saturating_sub(1);
                for successors in raw_successors.values() {
                    for successor in successors {
                        let added = successor.action_stack.len().saturating_sub(base_len);
                        stats.total_process_children_added_action_count += added;
                        stats.max_process_children_added_action_count =
                            stats.max_process_children_added_action_count.max(added);
                    }
                }
            }
            Some(super::WalkerAction::EmitDeferred(_)) => {
                stats.emit_deferred_state_count += 1;
            }
            Some(super::WalkerAction::EmitLiteral(_)) => {
                stats.emit_literal_state_count += 1;
            }
            Some(super::WalkerAction::EmitRingLabel(_)) => {
                stats.emit_ring_label_state_count += 1;
            }
            Some(super::WalkerAction::EmitCloseParen) => {
                stats.emit_close_paren_state_count += 1;
            }
        }

        for successors in super::successors_by_token_stereo_impl(
            runtime,
            graph,
            &state,
            true,
            completion_cache,
        )?
        .into_values()
        {
            for successor in successors {
                collect_stereo_decoder_stats(runtime, graph, successor, completion_cache, stats)?;
            }
        }
        Ok(())
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
        let support = enumerate_rooted_connected_stereo_smiles_support(&graph, 0)
            .expect("stereo enumeration should succeed")
            .into_iter()
            .collect::<BTreeSet<_>>();
        assert_eq!(BTreeSet::from(["F/[CH]=[CH]\\Cl".to_owned()]), support);
    }

    #[test]
    fn atom_stereo_support_matches_expected_curated_outputs() {
        let graph = atom_stereo_graph();
        let root_0 = enumerate_rooted_connected_stereo_smiles_support(&graph, 0)
            .expect("stereo enumeration should succeed")
            .into_iter()
            .collect::<BTreeSet<_>>();
        let root_1 = enumerate_rooted_connected_stereo_smiles_support(&graph, 1)
            .expect("stereo enumeration should succeed")
            .into_iter()
            .collect::<BTreeSet<_>>();

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
            let runtime = build_walker_runtime(&graph, root_idx).expect("runtime should build");
            let initial_state = initial_stereo_state_for_root(&runtime, &graph, root_idx);
            let mut walker_support = BTreeSet::new();
            enumerate_support_from_stereo_state(
                &runtime,
                &graph,
                initial_state,
                &mut walker_support,
            )
            .expect("walker state-machine support should enumerate");

            let direct_support =
                enumerate_rooted_connected_stereo_smiles_support(&graph, root_idx as isize)
                    .expect("direct stereo enumeration should succeed")
                    .into_iter()
                    .collect::<BTreeSet<_>>();

            assert_eq!(direct_support, walker_support);
        }
    }

    #[test]
    fn native_exact_support_matches_reference_for_small_isolated_bond_stereo_roots() {
        let graph = sample_stereo_graph();
        for root_idx in [0isize, 1, 2, 3] {
            let native = enumerate_rooted_connected_stereo_smiles_support_native(&graph, root_idx)
                .expect("native exact support should enumerate")
                .into_iter()
                .collect::<BTreeSet<_>>();
            let reference = enumerate_rooted_connected_stereo_smiles_support(&graph, root_idx)
                .expect("reference-backed exact support should enumerate")
                .into_iter()
                .collect::<BTreeSet<_>>();
            assert_eq!(reference, native, "root_idx={root_idx}");
        }
    }

    #[test]
    fn native_exact_support_matches_reference_for_branch_first_small_bond_stereo_roots() {
        let Some(graph) = prepared_graph_from_smiles("C(/C=C/Cl)Cl") else {
            return;
        };
        for root_idx in [0isize, 1, 2, 3] {
            let native = enumerate_rooted_connected_stereo_smiles_support_native(&graph, root_idx)
                .expect("native exact support should enumerate")
                .into_iter()
                .collect::<BTreeSet<_>>();
            let reference = enumerate_rooted_connected_stereo_smiles_support(&graph, root_idx)
                .expect("reference-backed exact support should enumerate")
                .into_iter()
                .collect::<BTreeSet<_>>();
            assert_eq!(reference, native, "root_idx={root_idx}");
        }
    }

    #[test]
    fn native_online_walker_matches_reference_for_coupled_diene_root_5() {
        let Some(graph) = prepared_graph_from_smiles("C/C=C(/C(=C/C)/c1ccccc1)\\c1ccccc1") else {
            return;
        };
        let runtime = build_walker_runtime(&graph, 5).expect("runtime should build");
        let initial_state = initial_stereo_state_for_root(&runtime, &graph, 5);
        let mut stack = vec![initial_state];
        let mut observed = BTreeSet::new();
        while let Some(state) = stack.pop() {
            if is_terminal_stereo_state(&state) {
                observed.insert(state.prefix.to_string());
                continue;
            }
            let mut choices =
                choices_for_stereo_state(&runtime, &graph, &state).expect("choices should enumerate");
            while let Some(choice) = choices.pop() {
                stack.extend(choice.next_frontier);
            }
        }

        let expected = enumerate_rooted_connected_stereo_smiles_support(&graph, 5)
            .expect("reference-backed exact support should enumerate")
            .into_iter()
            .collect::<BTreeSet<_>>();

        assert_eq!(expected, observed);
    }

    #[test]
    fn native_online_walker_matches_reference_for_polyene_root_11() {
        let Some(graph) =
            prepared_graph_from_smiles("CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C(=O)O)/C)/C")
        else {
            return;
        };
        let runtime = build_walker_runtime(&graph, 11).expect("runtime should build");
        let initial_state = initial_stereo_state_for_root(&runtime, &graph, 11);
        let mut stack = vec![initial_state];
        let mut observed = BTreeSet::new();
        while let Some(state) = stack.pop() {
            if is_terminal_stereo_state(&state) {
                observed.insert(state.prefix.to_string());
                continue;
            }
            let mut choices =
                choices_for_stereo_state(&runtime, &graph, &state).expect("choices should enumerate");
            while let Some(choice) = choices.pop() {
                stack.extend(choice.next_frontier);
            }
        }

        let expected = enumerate_rooted_connected_stereo_smiles_support(&graph, 11)
            .expect("reference-backed exact support should enumerate")
            .into_iter()
            .collect::<BTreeSet<_>>();

        assert_eq!(expected, observed);
    }

    #[test]
    #[ignore = "diagnostic-only stereo decoder shape report for profiler-guided optimization"]
    fn stereo_decoder_stats_report_large_all_roots_case() {
        let Some(graph) = prepared_graph_from_smiles("COc1ccc2cc([C@H](C)C(=O)O)ccc2c1") else {
            return;
        };
        let mut stats = StereoDecoderStats::default();
        for root_idx in 0..graph.atom_count() {
            let runtime = build_walker_runtime(&graph, root_idx).expect("runtime should build");
            let initial_state = initial_stereo_state_for_root(&runtime, &graph, root_idx);
            let mut completion_cache = rustc_hash::FxHashMap::default();
            collect_stereo_decoder_stats(
                &runtime,
                &graph,
                initial_state,
                &mut completion_cache,
                &mut stats,
            )
            .expect("stereo decoder stats should collect");
        }

        println!(
            "large stereo decoder stats: states={} terminal={} avg_prefix={:.2} max_prefix={} avg_stack={:.2} max_stack={} avg_stack_payload={:.2} max_stack_payload={} avg_pending_sites={:.2} max_pending_sites={} avg_pending_rings={:.2} max_pending_rings={} enter_states={} chiral_enter_states={} avg_enter_succ={:.2} max_enter_succ={} avg_enter_added={:.2} max_enter_added={} process_states={} avg_process_succ={:.2} max_process_succ={} avg_process_added={:.2} max_process_added={} emit_deferred={} emit_literal={} emit_ring={} emit_close={}",
            stats.state_count,
            stats.terminal_state_count,
            stats.average_prefix_len(),
            stats.max_prefix_len,
            stats.average_action_stack_len(),
            stats.max_action_stack_len,
            stats.average_action_stack_payload_bytes(),
            stats.max_action_stack_payload_bytes,
            stats.average_pending_sites(),
            stats.max_pending_sites,
            stats.average_pending_rings(),
            stats.max_pending_rings,
            stats.enter_atom_state_count,
            stats.chiral_enter_atom_state_count,
            stats.average_enter_atom_successor_count(),
            stats.max_enter_atom_successor_count,
            stats.average_enter_atom_added_action_count(),
            stats.max_enter_atom_added_action_count,
            stats.process_children_state_count,
            stats.average_process_children_successor_count(),
            stats.max_process_children_successor_count,
            stats.average_process_children_added_action_count(),
            stats.max_process_children_added_action_count,
            stats.emit_deferred_state_count,
            stats.emit_literal_state_count,
            stats.emit_ring_label_state_count,
            stats.emit_close_paren_state_count,
        );
    }

    #[test]
    #[ignore = "diagnostic-only stereo enter-atom field-diff report"]
    fn stereo_enter_atom_field_diff_report_large_all_roots_case() {
        let Some(graph) = prepared_graph_from_smiles("COc1ccc2cc([C@H](C)C(=O)O)ccc2c1") else {
            return;
        };
        let mut stats = StereoFieldDiffStats::default();
        for root_idx in 0..graph.atom_count() {
            let runtime = build_walker_runtime(&graph, root_idx).expect("runtime should build");
            let initial_state = initial_stereo_state_for_root(&runtime, &graph, root_idx);
            let mut stack = vec![initial_state];
            let mut completion_cache = rustc_hash::FxHashMap::default();
            while let Some(state) = stack.pop() {
                collect_enter_atom_field_diff_stats(&runtime, &graph, &state, &mut stats)
                    .expect("field diff stats should collect");
                for successors in super::successors_by_token_stereo_impl(
                    &runtime,
                    &graph,
                    &state,
                    true,
                    &mut completion_cache,
                )
                .expect("successors should enumerate")
                .into_values()
                {
                    stack.extend(successors);
                }
            }
        }

        println!(
            "large stereo enter-atom field diffs: equal={} visited={} visited_count={} pending={} free_labels={} next_label={} phases={} neighbors={} orientations={} first_candidates={} begin_atoms={} token_flips={} action_stack={}",
            stats.equal_pairs,
            stats.visited,
            stats.visited_count,
            stats.pending,
            stats.free_labels,
            stats.next_label,
            stats.component_phases,
            stats.selected_neighbors,
            stats.selected_orientations,
            stats.first_emitted_candidates,
            stats.component_begin_atoms,
            stats.component_token_flips,
            stats.action_stack,
        );
    }

    #[test]
    #[ignore = "diagnostic-only process-children shape report"]
    fn stereo_process_children_shape_report_large_all_roots_case() {
        let Some(graph) = prepared_graph_from_smiles("COc1ccc2cc([C@H](C)C(=O)O)ccc2c1") else {
            return;
        };
        let mut stats = StereoProcessChildrenShapeStats::default();
        for root_idx in 0..graph.atom_count() {
            let runtime = build_walker_runtime(&graph, root_idx).expect("runtime should build");
            let initial_state = initial_stereo_state_for_root(&runtime, &graph, root_idx);
            let mut stack = vec![initial_state];
            let mut completion_cache = rustc_hash::FxHashMap::default();
            while let Some(state) = stack.pop() {
                collect_process_children_shape_stats(&runtime, &graph, &state, &mut stats)
                    .expect("process-children stats should collect");
                for successors in super::successors_by_token_stereo_impl(
                    &runtime,
                    &graph,
                    &state,
                    true,
                    &mut completion_cache,
                )
                .expect("successors should enumerate")
                .into_values()
                {
                    stack.extend(successors);
                }
            }
        }

        println!(
            "large stereo process-children shapes: states={} branch={} final={} no_side_edge={} empty_literal={} nonempty_literal={} deferred={} ring_label={} open_paren={} close_paren={} eager_changed={} unchanged_updates={}",
            stats.state_count,
            stats.branch_state_count,
            stats.final_state_count,
            stats.edge_without_side_ids,
            stats.empty_literal_edge_part,
            stats.nonempty_literal_edge_part,
            stats.deferred_edge_part,
            stats.ring_label_edge_part,
            stats.open_paren_edge_part,
            stats.close_paren_edge_part,
            stats.eager_state_changed_count,
            stats.unchanged_state_update_count,
        );
    }

    #[test]
    #[ignore = "medium native exact-support parity probe"]
    fn native_exact_support_matches_reference_for_coupled_diene_root_5() {
        let Some(graph) = prepared_graph_from_smiles("C/C=C(/C(=C/C)/c1ccccc1)\\c1ccccc1") else {
            return;
        };
        let native = enumerate_rooted_connected_stereo_smiles_support_native(&graph, 5)
            .expect("native exact support should enumerate")
            .into_iter()
            .collect::<BTreeSet<_>>();
        let reference = enumerate_rooted_connected_stereo_smiles_support(&graph, 5)
            .expect("reference-backed exact support should enumerate")
            .into_iter()
            .collect::<BTreeSet<_>>();
        assert_eq!(reference, native);
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
        let runtime = build_walker_runtime(&graph, 0).expect("runtime should build");
        let state = initial_stereo_state_for_root(&runtime, &graph, 0);
        assert_eq!(
            vec!["F".to_owned()],
            next_token_support_for_stereo_state(&runtime, &graph, &state)
                .expect("support should be available"),
        );
    }

    #[test]
    fn stereo_walker_can_reach_expected_terminal_prefix() {
        let graph = sample_stereo_graph();
        let runtime = build_walker_runtime(&graph, 0).expect("runtime should build");
        let support = enumerate_rooted_connected_stereo_smiles_support(&graph, 0)
            .expect("stereo enumeration should succeed")
            .into_iter()
            .collect::<BTreeSet<_>>();
        let mut state = initial_stereo_state_for_root(&runtime, &graph, 0);

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
        let runtime = build_walker_runtime(&graph, 0).expect("runtime should build");
        let state = initial_stereo_state_for_root(&runtime, &graph, 0);
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
        let runtime = build_walker_runtime(&graph, 1).expect("runtime should build");
        let mut state = initial_stereo_state_for_root(&runtime, &graph, 1);
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
