//! Current bond-stereo constraint preprocessing.
//!
//! This module is deliberately a mechanical extraction from `rooted_stereo`.
//! It gathers the static carrier-side/component model so the later semantic
//! replacement can happen behind a narrow boundary.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

use crate::prepared_graph::PreparedSmilesGraphData;

pub(crate) const CIS_STEREO_BOND_KINDS: &[&str] = &["STEREOCIS", "STEREOZ"];
pub(crate) const TRANS_STEREO_BOND_KINDS: &[&str] = &["STEREOE", "STEREOTRANS"];

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct StereoSideInfo {
    pub(crate) component_idx: usize,
    pub(crate) endpoint_atom_idx: usize,
    pub(crate) other_endpoint_atom_idx: usize,
    pub(crate) candidate_neighbors: Vec<usize>,
    pub(crate) candidate_base_tokens: Vec<String>,
}

pub(crate) struct StereoSideInfoBuild {
    pub(crate) side_infos: Vec<StereoSideInfo>,
    pub(crate) edge_to_side_ids: BTreeMap<(usize, usize), Vec<usize>>,
}

#[derive(Clone, Debug)]
pub(crate) struct AmbiguousSharedEdgeGroup {
    pub(crate) left_side_idx: usize,
    pub(crate) right_side_idx: usize,
    pub(crate) left_shared_neighbor: usize,
    pub(crate) right_shared_neighbor: usize,
}

pub(crate) fn canonical_edge(begin_idx: usize, end_idx: usize) -> (usize, usize) {
    if begin_idx < end_idx {
        (begin_idx, end_idx)
    } else {
        (end_idx, begin_idx)
    }
}

pub(crate) fn flip_direction_token(token: &str) -> PyResult<String> {
    match token {
        "/" => Ok("\\".to_owned()),
        "\\" => Ok("/".to_owned()),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported directional token: {token:?}"
        ))),
    }
}

pub(crate) fn is_stereo_double_bond(graph: &PreparedSmilesGraphData, bond_idx: usize) -> bool {
    if graph.bond_kinds[bond_idx] != "DOUBLE" {
        return false;
    }
    let stereo_kind = graph.bond_stereo_kinds[bond_idx].as_str();
    CIS_STEREO_BOND_KINDS.contains(&stereo_kind) || TRANS_STEREO_BOND_KINDS.contains(&stereo_kind)
}

fn rdkit_selected_stereo_seed_token(
    graph: &PreparedSmilesGraphData,
    bond_idx: usize,
    component_idx: usize,
    isolated_components: &[bool],
    all_single_candidate_components: &[bool],
    endpoint_idx: usize,
    neighbor_idx: usize,
) -> PyResult<Option<String>> {
    if !isolated_components
        .get(component_idx)
        .copied()
        .unwrap_or(false)
        || !all_single_candidate_components
            .get(component_idx)
            .copied()
            .unwrap_or(false)
        || !is_stereo_double_bond(graph, bond_idx)
    {
        return Ok(None);
    }

    let stored_begin_idx = graph.bond_begin_atom_indices[bond_idx];
    let stored_end_idx = graph.bond_end_atom_indices[bond_idx];
    let (stereo_begin_atom, stereo_end_atom) = graph.bond_stereo_atoms[bond_idx];
    if endpoint_idx == stored_begin_idx
        && stereo_begin_atom >= 0
        && neighbor_idx == stereo_begin_atom as usize
    {
        return Ok(Some("\\".to_owned()));
    }
    if endpoint_idx == stored_end_idx
        && stereo_end_atom >= 0
        && neighbor_idx == stereo_end_atom as usize
    {
        let stereo_kind = graph.bond_stereo_kinds[bond_idx].as_str();
        if CIS_STEREO_BOND_KINDS.contains(&stereo_kind) {
            return Ok(Some("\\".to_owned()));
        }
        if TRANS_STEREO_BOND_KINDS.contains(&stereo_kind) {
            return Ok(Some("/".to_owned()));
        }
        return Err(PyValueError::new_err(format!(
            "Unsupported stereo bond kind: {stereo_kind}"
        )));
    }
    Ok(None)
}

pub(crate) fn stereo_component_ids(graph: &PreparedSmilesGraphData) -> Vec<isize> {
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

pub(crate) fn component_sizes(stereo_component_ids: &[isize]) -> Vec<usize> {
    let component_count = stereo_component_ids.iter().copied().max().unwrap_or(-1) + 1;
    let mut counts = vec![0usize; component_count as usize];
    for &component_idx in stereo_component_ids {
        if component_idx >= 0 {
            counts[component_idx as usize] += 1;
        }
    }
    counts
}

// Suspicious current model:
// This precomputes component-wide carrier-token phases before the online
// traversal has established visit order, ring-closure status, and selected
// carrier ownership. It is intentionally isolated so the next implementation
// can replace it with explicit online constraints without touching traversal.
pub(crate) fn stereo_side_infos(
    graph: &PreparedSmilesGraphData,
    stereo_component_ids: &[isize],
) -> PyResult<StereoSideInfoBuild> {
    let isolated_components = component_sizes(stereo_component_ids)
        .into_iter()
        .map(|size| size == 1)
        .collect::<Vec<_>>();
    let mut side_candidates = Vec::<(usize, usize, usize, Vec<usize>)>::new();
    let mut oriented_nodes = BTreeSet::<(usize, usize)>::new();
    let mut parity_edges = BTreeMap::<(usize, usize), Vec<((usize, usize), bool)>>::new();
    let mut seed_tokens = BTreeMap::<(usize, usize), String>::new();
    let mut seed_nodes = Vec::<(usize, usize, (usize, usize))>::new();

    for (bond_idx, &component_idx) in stereo_component_ids
        .iter()
        .enumerate()
        .take(graph.bond_count)
    {
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

                seed_nodes.push((bond_idx, component_idx as usize, node));
            }
        }
    }

    let mut all_single_candidate_components = vec![true; isolated_components.len()];
    for (component_idx, _endpoint_idx, _other_idx, candidate_neighbors) in &side_candidates {
        if candidate_neighbors.len() != 1 {
            all_single_candidate_components[*component_idx] = false;
        }
    }

    for (bond_idx, component_idx, node) in seed_nodes {
        let stored_token = rdkit_selected_stereo_seed_token(
            graph,
            bond_idx,
            component_idx,
            &isolated_components,
            &all_single_candidate_components,
            node.0,
            node.1,
        )?
        .unwrap_or(graph.directed_bond_token(node.0, node.1)?);
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

    Ok(StereoSideInfoBuild {
        side_infos,
        edge_to_side_ids,
    })
}

// Suspicious current model:
// This detects contested shared carrier edges after static carrier selection.
// A future online constraint engine should derive this from emitted edge facts
// rather than from preselected component sides.
pub(crate) fn ambiguous_shared_edge_groups(
    side_infos: &[StereoSideInfo],
    edge_to_side_ids: &BTreeMap<(usize, usize), Vec<usize>>,
    isolated_components: &[bool],
) -> Vec<AmbiguousSharedEdgeGroup> {
    let mut seen_edges = BTreeSet::new();
    let mut groups = Vec::new();

    for side_info in side_infos {
        if isolated_components[side_info.component_idx] || side_info.candidate_neighbors.len() != 2
        {
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
