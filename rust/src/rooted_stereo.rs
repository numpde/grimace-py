use std::collections::{BTreeMap, BTreeSet, VecDeque};

use pyo3::exceptions::{PyIndexError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::prepared_graph::{PreparedSmilesGraphData, CONNECTED_STEREO_SURFACE};

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

#[derive(Clone, Debug, PartialEq, Eq)]
struct DeferredDirectionalToken {
    component_idx: usize,
    stored_token: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Part {
    Literal(String),
    Deferred(DeferredDirectionalToken),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SearchState {
    visited: Vec<bool>,
    visited_count: usize,
    pending: Vec<Vec<PendingRing>>,
    free_labels: Vec<usize>,
    next_label: usize,
    stereo_component_phases: Vec<i8>,
    stereo_selected_neighbors: Vec<isize>,
    stereo_selected_orientations: Vec<i8>,
    stereo_component_begin_atoms: Vec<isize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SearchResult {
    parts: Vec<Part>,
    state: SearchState,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct StereoSideInfo {
    component_idx: usize,
    endpoint_atom_idx: usize,
    other_endpoint_atom_idx: usize,
    candidate_neighbors: Vec<usize>,
    candidate_base_tokens: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum RingAction {
    Close(usize),
    Open(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum WalkerAction {
    EmitLiteral(String),
    EmitDeferred(DeferredDirectionalToken),
    EnterAtom { atom_idx: usize, parent_idx: Option<usize> },
    ProcessChildren {
        parent_idx: usize,
        child_order: Vec<usize>,
        next_branch_index: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RootedConnectedStereoWalkerStateData {
    prefix: String,
    visited: Vec<bool>,
    visited_count: usize,
    pending: Vec<Vec<PendingRing>>,
    free_labels: Vec<usize>,
    next_label: usize,
    stereo_component_phases: Vec<i8>,
    stereo_selected_neighbors: Vec<isize>,
    stereo_selected_orientations: Vec<i8>,
    stereo_component_begin_atoms: Vec<isize>,
    stereo_component_token_flips: Vec<i8>,
    action_stack: Vec<WalkerAction>,
}

#[derive(Clone, Debug)]
struct StereoWalkerRuntimeData {
    stereo_component_ids: Vec<isize>,
    isolated_components: Vec<bool>,
    side_infos: Vec<StereoSideInfo>,
    edge_to_side_ids: BTreeMap<(usize, usize), Vec<usize>>,
    side_ids_by_component: Vec<Vec<usize>>,
    component_bond_indices: Vec<Option<usize>>,
}

fn ring_label_text(label: usize) -> String {
    if label < 10 {
        label.to_string()
    } else if label < 100 {
        format!("%{label}")
    } else {
        format!("%({label})")
    }
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
        return Err(PyValueError::new_err("Stereo neighbor order length mismatch"));
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
            reference_index.get(neighbor).copied().ok_or_else(|| {
                PyValueError::new_err("Stereo neighbor order membership mismatch")
            })
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
    let hydrogen_count = graph.atom_explicit_h_counts[atom_idx] + graph.atom_implicit_h_counts[atom_idx];
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
    let chiral_tag = graph.atom_chiral_tags[atom_idx].as_str();
    if chiral_tag == "CHI_UNSPECIFIED" {
        return Ok(graph.atom_tokens[atom_idx].clone());
    }
    if !SUPPORTED_CHIRAL_TAGS.contains(&chiral_tag) {
        return Err(PyValueError::new_err(format!(
            "Unsupported chiral tag for rooted stereo emission: {chiral_tag}"
        )));
    }

    let hydrogen_count = graph.atom_explicit_h_counts[atom_idx] + graph.atom_implicit_h_counts[atom_idx];
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

fn collect_cartesian_choices(groups: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    for_each_cartesian_choice(groups, &mut |choice| out.push(choice.to_vec()));
    out
}

fn unique_permutations<T, F>(items: &[T], f: &mut F)
where
    T: Clone + Eq,
    F: FnMut(&[T]),
{
    fn recurse<T, F>(items: &[T], used: &mut [bool], current: &mut Vec<T>, f: &mut F)
    where
        T: Clone + Eq,
        F: FnMut(&[T]),
    {
        if current.len() == items.len() {
            f(current);
            return;
        }

        let mut seen_at_depth = Vec::<T>::new();
        for (index, value) in items.iter().enumerate() {
            if used[index] || seen_at_depth.iter().any(|seen| seen == value) {
                continue;
            }
            seen_at_depth.push(value.clone());
            used[index] = true;
            current.push(value.clone());
            recurse(items, used, current, f);
            current.pop();
            used[index] = false;
        }
    }

    let mut used = vec![false; items.len()];
    let mut current = Vec::with_capacity(items.len());
    recurse(items, &mut used, &mut current, f);
    if items.is_empty() {
        f(&[]);
    }
}

fn collect_unique_permutations<T>(items: &[T]) -> Vec<Vec<T>>
where
    T: Clone + Eq,
{
    let mut out = Vec::new();
    unique_permutations(items, &mut |perm| out.push(perm.to_vec()));
    out
}

fn add_pending(pending: &mut [Vec<PendingRing>], target_atom: usize, ring: PendingRing) {
    let current = &mut pending[target_atom];
    let insert_at = current
        .binary_search_by(|candidate| {
            (candidate.label, candidate.other_atom_idx).cmp(&(ring.label, ring.other_atom_idx))
        })
        .unwrap_or_else(|offset| offset);
    current.insert(insert_at, ring);
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

fn with_component_phase(component_phases: &[i8], component_idx: usize, phase: i8) -> PyResult<Vec<i8>> {
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
    Ok((
        with_component_phase(component_phases, component_idx, phase)?,
        with_component_begin_atom(component_begin_atoms, component_idx, begin_idx)?,
    ))
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
                            .map(|bond_idx| graph.bond_kinds[bond_idx] == "SINGLE")
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
                parity_edges.entry(left_node).or_default().push((right_node, true));
                parity_edges.entry(right_node).or_default().push((left_node, true));
            }

            for node in oriented_for_side {
                let reverse_node = (node.1, node.0);
                if oriented_nodes.contains(&reverse_node) {
                    parity_edges.entry(node).or_default().push((reverse_node, true));
                    parity_edges.entry(reverse_node).or_default().push((node, true));
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

    for (_, endpoint_idx, _, candidate_neighbors) in &side_candidates {
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
        .map(|(component_idx, endpoint_idx, other_idx, candidate_neighbors)| {
            let candidate_base_tokens = candidate_neighbors
                .iter()
                .map(|&neighbor_idx| {
                    assignments.get(&(endpoint_idx, neighbor_idx)).cloned().ok_or_else(|| {
                        PyKeyError::new_err("Missing stereo carrier assignment")
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(StereoSideInfo {
                component_idx,
                endpoint_atom_idx: endpoint_idx,
                other_endpoint_atom_idx: other_idx,
                candidate_neighbors,
                candidate_base_tokens,
            })
        })
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
    Err(PyKeyError::new_err("Emitted edge does not match the stereo side"))
}

fn emitted_edge_part_generic(
    graph: &PreparedSmilesGraphData,
    side_infos: &[StereoSideInfo],
    edge_to_side_ids: &BTreeMap<(usize, usize), Vec<usize>>,
    component_phases: &[i8],
    selected_neighbors: &[isize],
    selected_orientations: &[i8],
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<(Part, Vec<isize>, Vec<i8>)> {
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
        ));
    };

    let mut updated_neighbors = selected_neighbors.to_vec();
    let mut updated_orientations = selected_orientations.to_vec();
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

        let selected_neighbor = updated_neighbors[side_idx];
        if selected_neighbor < 0 {
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

    let phase = component_phases[component_idx];
    let part = if phase == UNKNOWN_COMPONENT_PHASE {
        Part::Deferred(DeferredDirectionalToken {
            component_idx,
            stored_token,
        })
    } else if phase == STORED_COMPONENT_PHASE {
        Part::Literal(stored_token)
    } else {
        Part::Literal(flip_direction_token(&stored_token)?)
    };
    Ok((part, updated_neighbors, updated_orientations))
}

fn emitted_isolated_edge_part(
    graph: &PreparedSmilesGraphData,
    side_infos: &[StereoSideInfo],
    edge_to_side_ids: &BTreeMap<(usize, usize), Vec<usize>>,
    selected_neighbors: &[isize],
    selected_orientations: &[i8],
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<(Part, Vec<isize>, Vec<i8>)> {
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
        ));
    };

    let mut updated_neighbors = selected_neighbors.to_vec();
    let mut updated_orientations = selected_orientations.to_vec();
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

        let selected_neighbor = updated_neighbors[side_idx];
        if selected_neighbor < 0 {
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
        }),
        updated_neighbors,
        updated_orientations,
    ))
}

fn emitted_edge_part(
    graph: &PreparedSmilesGraphData,
    side_infos: &[StereoSideInfo],
    edge_to_side_ids: &BTreeMap<(usize, usize), Vec<usize>>,
    component_phases: &[i8],
    selected_neighbors: &[isize],
    selected_orientations: &[i8],
    isolated_components: &[bool],
    begin_idx: usize,
    end_idx: usize,
) -> PyResult<(Part, Vec<isize>, Vec<i8>)> {
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
        ));
    };

    let uses_isolated_component = side_ids.iter().any(|&side_idx| {
        let component_idx = side_infos[side_idx].component_idx;
        isolated_components.get(component_idx).copied().unwrap_or(false)
    });
    if uses_isolated_component {
        emitted_isolated_edge_part(
            graph,
            side_infos,
            edge_to_side_ids,
            selected_neighbors,
            selected_orientations,
            begin_idx,
            end_idx,
        )
    } else {
        emitted_edge_part_generic(
            graph,
            side_infos,
            edge_to_side_ids,
            component_phases,
            selected_neighbors,
            selected_orientations,
            begin_idx,
            end_idx,
        )
    }
}

fn resolve_part(
    part: &Part,
    component_phases: &[i8],
    component_flips: Option<&[bool]>,
) -> PyResult<String> {
    match part {
        Part::Literal(value) => Ok(value.clone()),
        Part::Deferred(deferred) => {
            let mut resolved = if component_phases[deferred.component_idx] == FLIPPED_COMPONENT_PHASE {
                flip_direction_token(&deferred.stored_token)?
            } else {
                deferred.stored_token.clone()
            };
            if component_flips
                .and_then(|flips| flips.get(deferred.component_idx))
                .copied()
                .unwrap_or(false)
            {
                resolved = flip_direction_token(&resolved)?;
            }
            Ok(resolved)
        }
    }
}

fn resolve_parts(
    parts: &[Part],
    component_phases: &[i8],
    component_flips: Option<&[bool]>,
) -> PyResult<String> {
    let mut resolved = String::new();
    for part in parts {
        resolved.push_str(&resolve_part(part, component_phases, component_flips)?);
    }
    Ok(resolved)
}

fn isolated_component_flips(
    graph: &PreparedSmilesGraphData,
    stereo_component_ids: &[isize],
    side_infos: &[StereoSideInfo],
    isolated_components: &[bool],
    state: &SearchState,
) -> Vec<bool> {
    if isolated_components.is_empty() {
        return Vec::new();
    }

    let mut component_bond_indices = BTreeMap::<usize, usize>::new();
    for (bond_idx, &component_idx) in stereo_component_ids.iter().enumerate() {
        if component_idx >= 0 {
            component_bond_indices.entry(component_idx as usize).or_insert(bond_idx);
        }
    }

    let mut side_ids_by_component = BTreeMap::<usize, Vec<usize>>::new();
    for (side_idx, side_info) in side_infos.iter().enumerate() {
        side_ids_by_component
            .entry(side_info.component_idx)
            .or_default()
            .push(side_idx);
    }

    let mut flips = vec![false; isolated_components.len()];
    for (component_idx, &isolated) in isolated_components.iter().enumerate() {
        if !isolated {
            continue;
        }

        let Some(side_ids) = side_ids_by_component.get(&component_idx) else {
            continue;
        };
        if side_ids
            .iter()
            .all(|&side_idx| side_infos[side_idx].candidate_neighbors.len() == 1)
        {
            continue;
        }

        let begin_atom_idx = state.stereo_component_begin_atoms[component_idx];
        if begin_atom_idx < 0 {
            continue;
        }
        let begin_atom_idx = begin_atom_idx as usize;
        let Some(begin_side_idx) = side_ids
            .iter()
            .copied()
            .find(|&side_idx| side_infos[side_idx].endpoint_atom_idx == begin_atom_idx)
        else {
            continue;
        };

        let selected_neighbor_idx = state.stereo_selected_neighbors[begin_side_idx];
        if selected_neighbor_idx < 0 {
            continue;
        }
        let selected_neighbor_idx = selected_neighbor_idx as usize;
        let Some(selected_rank) = side_infos[begin_side_idx]
            .candidate_neighbors
            .iter()
            .position(|&neighbor_idx| neighbor_idx == selected_neighbor_idx)
        else {
            continue;
        };

        let Some(&bond_idx) = component_bond_indices.get(&component_idx) else {
            continue;
        };
        let begin_is_stored_begin = begin_atom_idx == graph.bond_begin_atom_indices[bond_idx];
        flips[component_idx] = selected_rank == if begin_is_stored_begin { 1 } else { 0 };
    }
    flips
}

fn expand_children(
    graph: &PreparedSmilesGraphData,
    stereo_component_ids: &[isize],
    side_infos: &[StereoSideInfo],
    edge_to_side_ids: &BTreeMap<(usize, usize), Vec<usize>>,
    isolated_components: &[bool],
    parent_idx: usize,
    child_order: &[usize],
    prefix_parts: &[Part],
    state: &SearchState,
) -> PyResult<Vec<SearchResult>> {
    if child_order.is_empty() {
        return Ok(vec![SearchResult {
            parts: prefix_parts.to_vec(),
            state: state.clone(),
        }]);
    }

    let branch_children = &child_order[..child_order.len() - 1];
    let main_child = child_order[child_order.len() - 1];

    fn recurse_branch_children(
        graph: &PreparedSmilesGraphData,
        stereo_component_ids: &[isize],
        side_infos: &[StereoSideInfo],
        edge_to_side_ids: &BTreeMap<(usize, usize), Vec<usize>>,
        isolated_components: &[bool],
        parent_idx: usize,
        branch_children: &[usize],
        main_child: usize,
        branch_index: usize,
        partial: &SearchResult,
    ) -> PyResult<Vec<SearchResult>> {
        if branch_index == branch_children.len() {
            let (edge_part, main_selected_neighbors, main_selected_orientations) = emitted_edge_part(
                graph,
                side_infos,
                edge_to_side_ids,
                &partial.state.stereo_component_phases,
                &partial.state.stereo_selected_neighbors,
                &partial.state.stereo_selected_orientations,
                isolated_components,
                parent_idx,
                main_child,
            )?;
            let (main_component_phases, main_component_begin_atoms) = component_phases_after_edge(
                graph,
                stereo_component_ids,
                &partial.state.stereo_component_phases,
                &partial.state.stereo_component_begin_atoms,
                parent_idx,
                main_child,
            )?;
            let main_state = SearchState {
                visited: partial.state.visited.clone(),
                visited_count: partial.state.visited_count,
                pending: partial.state.pending.clone(),
                free_labels: partial.state.free_labels.clone(),
                next_label: partial.state.next_label,
                stereo_component_phases: main_component_phases,
                stereo_selected_neighbors: main_selected_neighbors,
                stereo_selected_orientations: main_selected_orientations,
                stereo_component_begin_atoms: main_component_begin_atoms,
            };
            let main_results = enumerate_from_atom(
                graph,
                stereo_component_ids,
                side_infos,
                edge_to_side_ids,
                isolated_components,
                main_child,
                Some(parent_idx),
                &main_state,
            )?;
            return Ok(main_results
                .into_iter()
                .map(|main_result| {
                    let mut parts = partial.parts.clone();
                    parts.push(edge_part.clone());
                    parts.extend(main_result.parts);
                    SearchResult {
                        parts,
                        state: main_result.state,
                    }
                })
                .collect());
        }

        let child_idx = branch_children[branch_index];
        let (branch_part, branch_selected_neighbors, branch_selected_orientations) = emitted_edge_part(
            graph,
            side_infos,
            edge_to_side_ids,
            &partial.state.stereo_component_phases,
            &partial.state.stereo_selected_neighbors,
            &partial.state.stereo_selected_orientations,
            isolated_components,
            parent_idx,
            child_idx,
        )?;
        let (child_component_phases, child_component_begin_atoms) = component_phases_after_edge(
            graph,
            stereo_component_ids,
            &partial.state.stereo_component_phases,
            &partial.state.stereo_component_begin_atoms,
            parent_idx,
            child_idx,
        )?;
        let branch_state = SearchState {
            visited: partial.state.visited.clone(),
            visited_count: partial.state.visited_count,
            pending: partial.state.pending.clone(),
            free_labels: partial.state.free_labels.clone(),
            next_label: partial.state.next_label,
            stereo_component_phases: child_component_phases,
            stereo_selected_neighbors: branch_selected_neighbors,
            stereo_selected_orientations: branch_selected_orientations,
            stereo_component_begin_atoms: child_component_begin_atoms,
        };
        let branch_results = enumerate_from_atom(
            graph,
            stereo_component_ids,
            side_infos,
            edge_to_side_ids,
            isolated_components,
            child_idx,
            Some(parent_idx),
            &branch_state,
        )?;

        let mut results = Vec::new();
        for branch_result in branch_results {
            let mut parts = partial.parts.clone();
            parts.push(Part::Literal("(".to_owned()));
            parts.push(branch_part.clone());
            parts.extend(branch_result.parts.clone());
            parts.push(Part::Literal(")".to_owned()));
            let next_partial = SearchResult {
                parts,
                state: branch_result.state,
            };
            results.extend(recurse_branch_children(
                graph,
                stereo_component_ids,
                side_infos,
                edge_to_side_ids,
                isolated_components,
                parent_idx,
                branch_children,
                main_child,
                branch_index + 1,
                &next_partial,
            )?);
        }
        Ok(results)
    }

    recurse_branch_children(
        graph,
        stereo_component_ids,
        side_infos,
        edge_to_side_ids,
        isolated_components,
        parent_idx,
        branch_children,
        main_child,
        0,
        &SearchResult {
            parts: prefix_parts.to_vec(),
            state: state.clone(),
        },
    )
}

fn enumerate_from_atom(
    graph: &PreparedSmilesGraphData,
    stereo_component_ids: &[isize],
    side_infos: &[StereoSideInfo],
    edge_to_side_ids: &BTreeMap<(usize, usize), Vec<usize>>,
    isolated_components: &[bool],
    atom_idx: usize,
    parent_idx: Option<usize>,
    state: &SearchState,
) -> PyResult<Vec<SearchResult>> {
    let mut visited_now = state.visited.clone();
    visited_now[atom_idx] = true;
    let visited_count_now = state.visited_count + 1;

    let mut pending = state.pending.clone();
    let closures_here = std::mem::take(&mut pending[atom_idx]);
    let ordered_groups = ordered_neighbor_groups(graph, atom_idx, &visited_now);

    let mut results = Vec::new();
    for chosen_children in collect_cartesian_choices(&ordered_groups) {
        let child_order_seed = chosen_children.clone();
        let child_set = chosen_children.iter().copied().collect::<BTreeSet<_>>();
        let opening_targets = ordered_groups
            .iter()
            .flat_map(|group| group.iter().copied())
            .filter(|neighbor_idx| !child_set.contains(neighbor_idx))
            .collect::<Vec<_>>();

        let mut ring_actions = Vec::new();
        for closure_idx in 0..closures_here.len() {
            ring_actions.push(RingAction::Close(closure_idx));
        }
        for target_idx in &opening_targets {
            ring_actions.push(RingAction::Open(*target_idx));
        }

        for ring_action_order in collect_unique_permutations(&ring_actions) {
            let mut current_pending = pending.clone();
            let mut current_free = state.free_labels.clone();
            let mut current_next = state.next_label;
            let mut current_component_phases = state.stereo_component_phases.clone();
            let mut current_selected_neighbors = state.stereo_selected_neighbors.clone();
            let mut current_selected_orientations = state.stereo_selected_orientations.clone();
            let mut current_component_begin_atoms = state.stereo_component_begin_atoms.clone();
            let mut current_ring_parts = Vec::<Part>::new();
            let mut labels_freed_after_atom = Vec::<usize>::new();
            let mut ring_neighbor_order = Vec::<usize>::new();

            for ring_action in &ring_action_order {
                match *ring_action {
                    RingAction::Close(closure_idx) => {
                        let closure = &closures_here[closure_idx];
                        let (bond_part, updated_neighbors, updated_orientations) = emitted_edge_part(
                            graph,
                            side_infos,
                            edge_to_side_ids,
                            &current_component_phases,
                            &current_selected_neighbors,
                            &current_selected_orientations,
                            isolated_components,
                            atom_idx,
                            closure.other_atom_idx,
                        )?;
                        current_selected_neighbors = updated_neighbors;
                        current_selected_orientations = updated_orientations;
                        current_ring_parts.push(bond_part);
                        current_ring_parts.push(Part::Literal(ring_label_text(closure.label)));
                        labels_freed_after_atom.push(closure.label);
                        ring_neighbor_order.push(closure.other_atom_idx);
                    }
                    RingAction::Open(target_idx) => {
                        let label = allocate_label(&mut current_free, &mut current_next);
                        current_ring_parts.push(Part::Literal(ring_label_text(label)));
                        let (updated_phases, updated_begin_atoms) = component_phases_after_edge(
                            graph,
                            stereo_component_ids,
                            &current_component_phases,
                            &current_component_begin_atoms,
                            atom_idx,
                            target_idx,
                        )?;
                        current_component_phases = updated_phases;
                        current_component_begin_atoms = updated_begin_atoms;
                        let (_bond_part, updated_neighbors, updated_orientations) = emitted_edge_part(
                            graph,
                            side_infos,
                            edge_to_side_ids,
                            &current_component_phases,
                            &current_selected_neighbors,
                            &current_selected_orientations,
                            isolated_components,
                            atom_idx,
                            target_idx,
                        )?;
                        current_selected_neighbors = updated_neighbors;
                        current_selected_orientations = updated_orientations;
                        add_pending(
                            &mut current_pending,
                            target_idx,
                            PendingRing {
                                label,
                                other_atom_idx: atom_idx,
                            },
                        );
                        ring_neighbor_order.push(target_idx);
                    }
                }
            }

            for label in labels_freed_after_atom {
                insert_sorted(&mut current_free, label);
            }

            for child_order in collect_unique_permutations(&child_order_seed) {
                let atom_token = if graph.atom_chiral_tags[atom_idx] == "CHI_UNSPECIFIED" {
                    graph.atom_tokens[atom_idx].clone()
                } else {
                    let emitted_neighbor_order = stereo_neighbor_order(
                        graph,
                        atom_idx,
                        parent_idx,
                        &ring_neighbor_order,
                        &child_order,
                    )?;
                    stereo_atom_token(graph, atom_idx, &emitted_neighbor_order)?
                };
                let mut prefix_parts = vec![Part::Literal(atom_token)];
                prefix_parts.extend(current_ring_parts.clone());
                let current_state = SearchState {
                    visited: visited_now.clone(),
                    visited_count: visited_count_now,
                    pending: current_pending.clone(),
                    free_labels: current_free.clone(),
                    next_label: current_next,
                    stereo_component_phases: current_component_phases.clone(),
                    stereo_selected_neighbors: current_selected_neighbors.clone(),
                    stereo_selected_orientations: current_selected_orientations.clone(),
                    stereo_component_begin_atoms: current_component_begin_atoms.clone(),
                };
                let mut expanded = expand_children(
                    graph,
                    stereo_component_ids,
                    side_infos,
                    edge_to_side_ids,
                    isolated_components,
                    atom_idx,
                    &child_order,
                    &prefix_parts,
                    &current_state,
                )?;
                results.append(&mut expanded);
            }
        }
    }

    Ok(results)
}

pub(crate) fn enumerate_rooted_connected_stereo_smiles_support(
    graph: &PreparedSmilesGraphData,
    root_idx: isize,
) -> PyResult<Vec<String>> {
    check_supported_stereo_writer_surface(graph)?;
    if graph.atom_count() == 0 {
        return Ok(vec![String::new()]);
    }
    let root_idx = validate_root_idx(graph, root_idx)?;

    let stereo_component_ids = stereo_component_ids(graph);
    let component_count = stereo_component_ids.iter().copied().max().unwrap_or(-1) + 1;
    let isolated_components = component_sizes(&stereo_component_ids)
        .into_iter()
        .map(|size| size == 1)
        .collect::<Vec<_>>();
    let side_infos;
    let edge_to_side_ids;
    (side_infos, edge_to_side_ids) = stereo_side_infos(graph, &stereo_component_ids)?;

    let initial_state = SearchState {
        visited: vec![false; graph.atom_count()],
        visited_count: 0,
        pending: vec![Vec::new(); graph.atom_count()],
        free_labels: Vec::new(),
        next_label: 1,
        stereo_component_phases: vec![UNKNOWN_COMPONENT_PHASE; component_count as usize],
        stereo_selected_neighbors: vec![-1; side_infos.len()],
        stereo_selected_orientations: vec![UNKNOWN_EDGE_ORIENTATION; side_infos.len()],
        stereo_component_begin_atoms: vec![-1; component_count as usize],
    };

    let results = enumerate_from_atom(
        graph,
        &stereo_component_ids,
        &side_infos,
        &edge_to_side_ids,
        &isolated_components,
        root_idx,
        None,
        &initial_state,
    )?;
    let mut support = BTreeSet::new();
    for result in results {
        if result.state.visited_count != graph.atom_count() {
            continue;
        }
        if result.state.pending.iter().any(|rings| !rings.is_empty()) {
            continue;
        }
        let component_flips = isolated_component_flips(
            graph,
            &stereo_component_ids,
            &side_infos,
            &isolated_components,
            &result.state,
        );
        support.insert(resolve_parts(
            &result.parts,
            &result.state.stereo_component_phases,
            Some(&component_flips),
        )?);
    }
    Ok(support.into_iter().collect())
}

fn build_walker_runtime(graph: &PreparedSmilesGraphData) -> PyResult<StereoWalkerRuntimeData> {
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
            return Err(PyValueError::new_err("stereo side component index out of range"));
        }
        side_ids_by_component[side_info.component_idx].push(side_idx);
    }
    let mut component_bond_indices = vec![None; component_count as usize];
    for (bond_idx, &component_idx) in stereo_component_ids.iter().enumerate() {
        if component_idx >= 0 && component_bond_indices[component_idx as usize].is_none() {
            component_bond_indices[component_idx as usize] = Some(bond_idx);
        }
    }
    Ok(StereoWalkerRuntimeData {
        stereo_component_ids,
        isolated_components,
        side_infos,
        edge_to_side_ids,
        side_ids_by_component,
        component_bond_indices,
    })
}

fn validate_stereo_state_shape(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<()> {
    if state.visited.len() != graph.atom_count()
        || state.pending.len() != graph.atom_count()
        || state.stereo_component_phases.len() != runtime.isolated_components.len()
        || state.stereo_component_begin_atoms.len() != runtime.isolated_components.len()
        || state.stereo_component_token_flips.len() != runtime.isolated_components.len()
        || state.stereo_selected_neighbors.len() != runtime.side_infos.len()
        || state.stereo_selected_orientations.len() != runtime.side_infos.len()
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
        prefix: String::new(),
        visited: vec![false; graph.atom_count()],
        visited_count: 0,
        pending: vec![Vec::new(); graph.atom_count()],
        free_labels: Vec::new(),
        next_label: 1,
        stereo_component_phases: vec![UNKNOWN_COMPONENT_PHASE; runtime.isolated_components.len()],
        stereo_selected_neighbors: vec![-1; runtime.side_infos.len()],
        stereo_selected_orientations: vec![UNKNOWN_EDGE_ORIENTATION; runtime.side_infos.len()],
        stereo_component_begin_atoms: vec![-1; runtime.isolated_components.len()],
        stereo_component_token_flips: vec![UNKNOWN_COMPONENT_TOKEN_FLIP; runtime.isolated_components.len()],
        action_stack,
    }
}

fn is_terminal_stereo_state(state: &RootedConnectedStereoWalkerStateData) -> bool {
    state.action_stack.is_empty()
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

fn inferred_component_token_flip(
    runtime: &StereoWalkerRuntimeData,
    state: &RootedConnectedStereoWalkerStateData,
    component_idx: usize,
    graph: &PreparedSmilesGraphData,
) -> PyResult<Option<i8>> {
    let phase = state.stereo_component_phases[component_idx];
    let isolated = runtime.isolated_components[component_idx];
    if !isolated {
        return Ok(match phase {
            STORED_COMPONENT_PHASE => Some(STORED_COMPONENT_TOKEN_FLIP),
            FLIPPED_COMPONENT_PHASE => Some(FLIPPED_COMPONENT_TOKEN_FLIP),
            _ => None,
        });
    }

    let side_ids = &runtime.side_ids_by_component[component_idx];
    if side_ids.is_empty() {
        return Ok(None);
    }
    if side_ids
        .iter()
        .all(|&side_idx| runtime.side_infos[side_idx].candidate_neighbors.len() == 1)
    {
        return Ok(match phase {
            STORED_COMPONENT_PHASE => Some(STORED_COMPONENT_TOKEN_FLIP),
            FLIPPED_COMPONENT_PHASE => Some(FLIPPED_COMPONENT_TOKEN_FLIP),
            _ => None,
        });
    }

    let begin_atom_idx = state.stereo_component_begin_atoms[component_idx];
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
    let selected_neighbor_idx = state.stereo_selected_neighbors[begin_side_idx];
    if selected_neighbor_idx < 0 {
        return Ok(None);
    }
    let selected_neighbor_idx = selected_neighbor_idx as usize;
    let Some(selected_rank) = runtime.side_infos[begin_side_idx]
        .candidate_neighbors
        .iter()
        .position(|&neighbor_idx| neighbor_idx == selected_neighbor_idx)
    else {
        return Ok(None);
    };
    let Some(bond_idx) = runtime.component_bond_indices[component_idx] else {
        return Ok(None);
    };
    let begin_is_stored_begin = begin_atom_idx == graph.bond_begin_atom_indices[bond_idx];
    let component_flip = if selected_rank == if begin_is_stored_begin { 1 } else { 0 } {
        FLIPPED_COMPONENT_TOKEN_FLIP
    } else {
        STORED_COMPONENT_TOKEN_FLIP
    };
    Ok(match phase {
        STORED_COMPONENT_PHASE => Some(component_flip),
        FLIPPED_COMPONENT_PHASE => Some(if component_flip == STORED_COMPONENT_TOKEN_FLIP {
            FLIPPED_COMPONENT_TOKEN_FLIP
        } else {
            STORED_COMPONENT_TOKEN_FLIP
        }),
        _ => None,
    })
}

fn normalize_component_token_flips(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &mut RootedConnectedStereoWalkerStateData,
) -> PyResult<()> {
    for component_idx in 0..state.stereo_component_token_flips.len() {
        let inferred = inferred_component_token_flip(runtime, state, component_idx, graph)?;
        if let Some(inferred) = inferred {
            let existing = state.stereo_component_token_flips[component_idx];
            if existing == UNKNOWN_COMPONENT_TOKEN_FLIP {
                state.stereo_component_token_flips[component_idx] = inferred;
            } else if existing != inferred {
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
        _ => Err(PyValueError::new_err("Unsupported component token flip value")),
    }
}

fn deferred_token_support(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    deferred: &DeferredDirectionalToken,
) -> PyResult<Vec<String>> {
    let known_flip = if state.stereo_component_token_flips[deferred.component_idx]
        != UNKNOWN_COMPONENT_TOKEN_FLIP
    {
        Some(state.stereo_component_token_flips[deferred.component_idx])
    } else {
        inferred_component_token_flip(runtime, state, deferred.component_idx, graph)?
    };
    if let Some(token_flip) = known_flip {
        return Ok(vec![token_from_stored_with_flip(
            &deferred.stored_token,
            token_flip,
        )?]);
    }
    let flipped = flip_direction_token(&deferred.stored_token)?;
    if flipped == deferred.stored_token {
        Ok(vec![deferred.stored_token.clone()])
    } else {
        let mut out = vec![deferred.stored_token.clone(), flipped];
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
    let stored = deferred.stored_token.as_str();
    let flipped = flip_direction_token(stored)?;
    let chosen_flip = if chosen_token == stored {
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

    let mut visited_now = base_state.visited.clone();
    debug_assert!(!visited_now[atom_idx]);
    visited_now[atom_idx] = true;
    let visited_count_now = base_state.visited_count + 1;

    let mut pending_now = base_state.pending.clone();
    let closures_here = std::mem::take(&mut pending_now[atom_idx]);
    let ordered_groups = ordered_neighbor_groups(graph, atom_idx, &visited_now);

    let mut successors = BTreeMap::<String, Vec<RootedConnectedStereoWalkerStateData>>::new();
    for chosen_children in collect_cartesian_choices(&ordered_groups) {
        let child_order_seed = chosen_children.clone();
        let child_set = chosen_children.iter().copied().collect::<BTreeSet<_>>();
        let opening_targets = ordered_groups
            .iter()
            .flat_map(|group| group.iter().copied())
            .filter(|neighbor_idx| !child_set.contains(neighbor_idx))
            .collect::<Vec<_>>();

        let mut ring_actions = Vec::new();
        for closure_idx in 0..closures_here.len() {
            ring_actions.push(RingAction::Close(closure_idx));
        }
        for &target_idx in &opening_targets {
            ring_actions.push(RingAction::Open(target_idx));
        }

        for ring_action_order in collect_unique_permutations(&ring_actions) {
            let mut current_pending = pending_now.clone();
            let mut current_free = base_state.free_labels.clone();
            let mut current_next = base_state.next_label;
            let mut current_component_phases = base_state.stereo_component_phases.clone();
            let mut current_selected_neighbors = base_state.stereo_selected_neighbors.clone();
            let mut current_selected_orientations = base_state.stereo_selected_orientations.clone();
            let mut current_component_begin_atoms = base_state.stereo_component_begin_atoms.clone();
            let mut current_ring_actions = Vec::<WalkerAction>::new();
            let mut labels_freed_after_atom = Vec::<usize>::new();
            let mut ring_neighbor_order = Vec::<usize>::new();

            for ring_action in &ring_action_order {
                match *ring_action {
                    RingAction::Close(closure_idx) => {
                        let closure = &closures_here[closure_idx];
                        let (bond_part, updated_neighbors, updated_orientations) = emitted_edge_part(
                            graph,
                            &runtime.side_infos,
                            &runtime.edge_to_side_ids,
                            &current_component_phases,
                            &current_selected_neighbors,
                            &current_selected_orientations,
                            &runtime.isolated_components,
                            atom_idx,
                            closure.other_atom_idx,
                        )?;
                        current_selected_neighbors = updated_neighbors;
                        current_selected_orientations = updated_orientations;
                        if let Some(action) = part_to_action(bond_part) {
                            current_ring_actions.push(action);
                        }
                        current_ring_actions.push(WalkerAction::EmitLiteral(ring_label_text(closure.label)));
                        labels_freed_after_atom.push(closure.label);
                        ring_neighbor_order.push(closure.other_atom_idx);
                    }
                    RingAction::Open(target_idx) => {
                        let label = allocate_label(&mut current_free, &mut current_next);
                        current_ring_actions.push(WalkerAction::EmitLiteral(ring_label_text(label)));
                        let (updated_phases, updated_begin_atoms) = component_phases_after_edge(
                            graph,
                            &runtime.stereo_component_ids,
                            &current_component_phases,
                            &current_component_begin_atoms,
                            atom_idx,
                            target_idx,
                        )?;
                        current_component_phases = updated_phases;
                        current_component_begin_atoms = updated_begin_atoms;
                        let (_ignored_part, updated_neighbors, updated_orientations) =
                            emitted_edge_part(
                                graph,
                                &runtime.side_infos,
                                &runtime.edge_to_side_ids,
                                &current_component_phases,
                                &current_selected_neighbors,
                                &current_selected_orientations,
                                &runtime.isolated_components,
                                atom_idx,
                                target_idx,
                            )?;
                        current_selected_neighbors = updated_neighbors;
                        current_selected_orientations = updated_orientations;
                        add_pending(
                            &mut current_pending,
                            target_idx,
                            PendingRing {
                                label,
                                other_atom_idx: atom_idx,
                            },
                        );
                        ring_neighbor_order.push(target_idx);
                    }
                }
            }

            for label in labels_freed_after_atom {
                insert_sorted(&mut current_free, label);
            }

            for child_order in collect_unique_permutations(&child_order_seed) {
                let atom_token = if graph.atom_chiral_tags[atom_idx] == "CHI_UNSPECIFIED" {
                    graph.atom_tokens[atom_idx].clone()
                } else {
                    let emitted_neighbor_order = stereo_neighbor_order(
                        graph,
                        atom_idx,
                        parent_idx,
                        &ring_neighbor_order,
                        &child_order,
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
                    stereo_component_begin_atoms: current_component_begin_atoms.clone(),
                    stereo_component_token_flips: base_state.stereo_component_token_flips.clone(),
                    action_stack: base_state.action_stack.clone(),
                };
                if !child_order.is_empty() {
                    successor.action_stack.push(WalkerAction::ProcessChildren {
                        parent_idx: atom_idx,
                        child_order: child_order.clone(),
                        next_branch_index: 0,
                    });
                }
                for action in current_ring_actions.iter().rev() {
                    successor.action_stack.push(action.clone());
                }
                successor.prefix.push_str(&atom_token);
                normalize_component_token_flips(runtime, graph, &mut successor)?;
                successors.entry(atom_token).or_default().push(successor);
            }
        }
    }
    Ok(successors)
}

fn process_children_successors_by_token(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    parent_idx: usize,
    child_order: &[usize],
    next_branch_index: usize,
) -> PyResult<BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>> {
    if child_order.is_empty() {
        return Ok(BTreeMap::new());
    }
    let branch_count = child_order.len().saturating_sub(1);

    if next_branch_index < branch_count {
        let child_idx = child_order[next_branch_index];
        let mut successor = state.clone();
        successor.action_stack.pop();
        successor.prefix.push('(');
        let (edge_part, updated_neighbors, updated_orientations) = emitted_edge_part(
            graph,
            &runtime.side_infos,
            &runtime.edge_to_side_ids,
            &successor.stereo_component_phases,
            &successor.stereo_selected_neighbors,
            &successor.stereo_selected_orientations,
            &runtime.isolated_components,
            parent_idx,
            child_idx,
        )?;
        let (updated_phases, updated_begin_atoms) = component_phases_after_edge(
            graph,
            &runtime.stereo_component_ids,
            &successor.stereo_component_phases,
            &successor.stereo_component_begin_atoms,
            parent_idx,
            child_idx,
        )?;
        successor.stereo_selected_neighbors = updated_neighbors;
        successor.stereo_selected_orientations = updated_orientations;
        successor.stereo_component_phases = updated_phases;
        successor.stereo_component_begin_atoms = updated_begin_atoms;
        if next_branch_index + 1 < child_order.len() {
            successor.action_stack.push(WalkerAction::ProcessChildren {
                parent_idx,
                child_order: child_order.to_vec(),
                next_branch_index: next_branch_index + 1,
            });
        }
        successor
            .action_stack
            .push(WalkerAction::EmitLiteral(")".to_owned()));
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
    let (edge_part, updated_neighbors, updated_orientations) = emitted_edge_part(
        graph,
        &runtime.side_infos,
        &runtime.edge_to_side_ids,
        &state.stereo_component_phases,
        &state.stereo_selected_neighbors,
        &state.stereo_selected_orientations,
        &runtime.isolated_components,
        parent_idx,
        child_idx,
    )?;
    let (updated_phases, updated_begin_atoms) = component_phases_after_edge(
        graph,
        &runtime.stereo_component_ids,
        &state.stereo_component_phases,
        &state.stereo_component_begin_atoms,
        parent_idx,
        child_idx,
    )?;

    let mut base_state = state.clone();
    base_state.action_stack.pop();
    base_state.stereo_selected_neighbors = updated_neighbors;
    base_state.stereo_selected_orientations = updated_orientations;
    base_state.stereo_component_phases = updated_phases;
    base_state.stereo_component_begin_atoms = updated_begin_atoms;
    normalize_component_token_flips(runtime, graph, &mut base_state)?;

    match edge_part {
        Part::Literal(token) if token.is_empty() => {
            base_state.action_stack.push(WalkerAction::EnterAtom {
                atom_idx: child_idx,
                parent_idx: Some(parent_idx),
            });
            successors_by_token_stereo(runtime, graph, &base_state)
        }
        Part::Literal(token) => {
            base_state.prefix.push_str(&token);
            base_state.action_stack.push(WalkerAction::EnterAtom {
                atom_idx: child_idx,
                parent_idx: Some(parent_idx),
            });
            Ok(BTreeMap::from([(token, vec![base_state])]))
        }
        Part::Deferred(deferred) => {
            let mut out = BTreeMap::<String, Vec<RootedConnectedStereoWalkerStateData>>::new();
            for token in deferred_token_support(runtime, graph, &base_state, &deferred)? {
                let mut successor = base_state.clone();
                commit_deferred_token_choice(runtime, graph, &mut successor, &deferred, &token)?;
                successor.prefix.push_str(&token);
                successor.action_stack.push(WalkerAction::EnterAtom {
                    atom_idx: child_idx,
                    parent_idx: Some(parent_idx),
                });
                if can_complete_from_stereo_state(runtime, graph, &successor) {
                    out.entry(token).or_default().push(successor);
                }
            }
            Ok(out)
        }
    }
}

fn next_token_support_for_stereo_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<Vec<String>> {
    Ok(successors_by_token_stereo(runtime, graph, state)?
        .into_keys()
        .collect())
}

fn can_complete_from_stereo_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> bool {
    let successors = match successors_by_token_stereo(runtime, graph, state) {
        Ok(successors) => successors,
        Err(_) => return false,
    };
    if successors.is_empty() {
        return state.action_stack.is_empty()
            && state.visited_count == graph.atom_count()
            && state.pending.iter().all(|rings| rings.is_empty());
    }
    for successor_group in successors.into_values() {
        for successor in successor_group {
            if can_complete_from_stereo_state(runtime, graph, &successor) {
                return true;
            }
        }
    }
    false
}

fn successors_by_token_stereo(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
) -> PyResult<BTreeMap<String, Vec<RootedConnectedStereoWalkerStateData>>> {
    let action = match state.action_stack.last() {
        Some(action) => action.clone(),
        None => return Ok(BTreeMap::new()),
    };

    match action {
        WalkerAction::EmitLiteral(token) => {
            let mut successor = state.clone();
            successor.action_stack.pop();
            successor.prefix.push_str(&token);
            Ok(BTreeMap::from([(token, vec![successor])]))
        }
        WalkerAction::EmitDeferred(deferred) => {
            let mut out = BTreeMap::<String, Vec<RootedConnectedStereoWalkerStateData>>::new();
            for token in deferred_token_support(runtime, graph, state, &deferred)? {
                let mut successor = state.clone();
                successor.action_stack.pop();
                commit_deferred_token_choice(runtime, graph, &mut successor, &deferred, &token)?;
                successor.prefix.push_str(&token);
                if can_complete_from_stereo_state(runtime, graph, &successor) {
                    out.entry(token).or_default().push(successor);
                }
            }
            Ok(out)
        }
        WalkerAction::EnterAtom { atom_idx, parent_idx } => {
            enter_atom_successors_by_token(runtime, graph, state, atom_idx, parent_idx)
        }
        WalkerAction::ProcessChildren {
            parent_idx,
            child_order,
            next_branch_index,
        } => process_children_successors_by_token(
            runtime,
            graph,
            state,
            parent_idx,
            &child_order,
            next_branch_index,
        ),
    }
}

fn advance_stereo_token_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: &RootedConnectedStereoWalkerStateData,
    chosen_token: &str,
) -> PyResult<RootedConnectedStereoWalkerStateData> {
    let mut successors = successors_by_token_stereo(runtime, graph, state)?;
    let candidates = successors.remove(chosen_token).ok_or_else(|| {
        let available = next_token_support_for_stereo_state(runtime, graph, state)
            .unwrap_or_default();
        PyKeyError::new_err(format!(
            "Token {chosen_token:?} is not available; choices={available:?}"
        ))
    })?;
    Ok(candidates
        .into_iter()
        .next()
        .expect("chosen token should have at least one successor"))
}

fn enumerate_support_from_stereo_state(
    runtime: &StereoWalkerRuntimeData,
    graph: &PreparedSmilesGraphData,
    state: RootedConnectedStereoWalkerStateData,
    out: &mut BTreeSet<String>,
) -> PyResult<()> {
    let successors = successors_by_token_stereo(runtime, graph, &state)?;
    if successors.is_empty() {
        if state.action_stack.is_empty()
            && state.visited_count == graph.atom_count()
            && state.pending.iter().all(|rings| rings.is_empty())
        {
            out.insert(state.prefix);
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

#[pyclass(name = "RootedConnectedStereoWalkerState", module = "smiles_next_token._core", frozen)]
pub struct PyRootedConnectedStereoWalkerState {
    data: RootedConnectedStereoWalkerStateData,
}

#[pymethods]
impl PyRootedConnectedStereoWalkerState {
    #[getter]
    fn prefix(&self) -> String {
        self.data.prefix.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RootedConnectedStereoWalkerState(prefix={:?}, visited_count={}, pending_sites={}, next_label={}, stack_depth={})",
            self.data.prefix,
            self.data.visited_count,
            self.data.pending.iter().filter(|rings| !rings.is_empty()).count(),
            self.data.next_label,
            self.data.action_stack.len(),
        )
    }
}

#[pyclass(name = "RootedConnectedStereoWalker", module = "smiles_next_token._core", frozen)]
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
        let runtime = build_walker_runtime(&graph)?;
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
        next_token_support_for_stereo_state(&self.runtime, &self.graph, &state.data)
    }

    fn advance_token(
        &self,
        state: &PyRootedConnectedStereoWalkerState,
        chosen_token: &str,
    ) -> PyResult<PyRootedConnectedStereoWalkerState> {
        validate_stereo_state_shape(&self.runtime, &self.graph, &state.data)?;
        Ok(PyRootedConnectedStereoWalkerState {
            data: advance_stereo_token_state(&self.runtime, &self.graph, &state.data, chosen_token)?,
        })
    }

    fn is_terminal(&self, state: &PyRootedConnectedStereoWalkerState) -> PyResult<bool> {
        validate_stereo_state_shape(&self.runtime, &self.graph, &state.data)?;
        Ok(is_terminal_stereo_state(&state.data))
    }

    fn enumerate_support(&self) -> PyResult<Vec<String>> {
        let initial_state = initial_stereo_state_for_root(&self.runtime, &self.graph, self.root_idx);
        let mut out = BTreeSet::new();
        enumerate_support_from_stereo_state(&self.runtime, &self.graph, initial_state, &mut out)?;
        Ok(out.into_iter().collect())
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
