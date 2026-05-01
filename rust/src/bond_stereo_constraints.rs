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

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum StereoConstraintLayer {
    Semantic,
    RdkitLocalWriter,
    RdkitTraversalWriter,
}

impl StereoConstraintLayer {
    pub(crate) const ALL: [Self; 3] = [
        Self::Semantic,
        Self::RdkitLocalWriter,
        Self::RdkitTraversalWriter,
    ];
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StereoCarrierChoice {
    pub(crate) neighbor_idx: usize,
    pub(crate) base_token: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StereoSideChoiceDomain {
    pub(crate) side_idx: usize,
    pub(crate) component_idx: usize,
    pub(crate) endpoint_atom_idx: usize,
    pub(crate) choices: Vec<StereoCarrierChoice>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StereoLayerAssignments {
    pub(crate) layer: StereoConstraintLayer,
    // `None` currently means "all domain-valid assignments are allowed". Later
    // phases will fill this with generated allowed tuples per component.
    pub(crate) allowed_neighbor_assignments: Option<Vec<Vec<usize>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StereoComponentConstraintModel {
    pub(crate) component_idx: usize,
    pub(crate) side_ids: Vec<usize>,
    pub(crate) side_domains: Vec<StereoSideChoiceDomain>,
    pub(crate) layer_assignments: Vec<StereoLayerAssignments>,
}

#[allow(dead_code)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum StereoTraversalRole {
    TreeOrChain,
    Branch,
    RingOpen,
    RingClose,
    Deferred,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum StereoConstraintFact {
    CarrierSelected {
        side_idx: usize,
        neighbor_idx: usize,
    },
    CarrierEdgeEmitted {
        side_idx: usize,
        begin_idx: usize,
        end_idx: usize,
        role: StereoTraversalRole,
    },
    DirectionalMarkerPlaced {
        side_idx: usize,
        slot: usize,
        marker: char,
        role: StereoTraversalRole,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct StereoConstraintModel {
    pub(crate) components: Vec<StereoComponentConstraintModel>,
    side_to_component: Vec<Option<usize>>,
}

#[derive(Clone, Debug)]
pub(crate) struct AmbiguousSharedEdgeGroup {
    pub(crate) left_side_idx: usize,
    pub(crate) right_side_idx: usize,
    pub(crate) left_shared_neighbor: usize,
    pub(crate) right_shared_neighbor: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct StereoLocalHazard {
    pub(crate) left_side_idx: usize,
    pub(crate) left_neighbor_idx: usize,
    pub(crate) right_side_idx: usize,
    pub(crate) right_neighbor_idx: usize,
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

fn find_component(parents: &mut [usize], component_idx: usize) -> usize {
    let mut root = component_idx;
    while parents[root] != root {
        root = parents[root];
    }
    let mut current = component_idx;
    while parents[current] != current {
        let next_idx = parents[current];
        parents[current] = root;
        current = next_idx;
    }
    root
}

fn union_components(parents: &mut [usize], left_idx: usize, right_idx: usize) {
    let left_root = find_component(parents, left_idx);
    let right_root = find_component(parents, right_idx);
    if left_root != right_root {
        parents[right_root] = left_root;
    }
}

fn enumerate_domain_assignments(side_domains: &[StereoSideChoiceDomain]) -> Vec<Vec<usize>> {
    fn rec(
        side_domains: &[StereoSideChoiceDomain],
        offset: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if offset == side_domains.len() {
            out.push(current.clone());
            return;
        }
        for choice in &side_domains[offset].choices {
            current.push(choice.neighbor_idx);
            rec(side_domains, offset + 1, current, out);
            current.pop();
        }
    }

    let mut out = Vec::new();
    rec(side_domains, 0, &mut Vec::new(), &mut out);
    out
}

fn side_has_neighbor(side_infos: &[StereoSideInfo], side_idx: usize, neighbor_idx: usize) -> bool {
    side_infos
        .get(side_idx)
        .map(|side_info| side_info.candidate_neighbors.contains(&neighbor_idx))
        .unwrap_or(false)
}

fn local_writer_allowed_assignments(
    side_ids: &[usize],
    side_domains: &[StereoSideChoiceDomain],
    local_hazards: &[StereoLocalHazard],
) -> Option<Vec<Vec<usize>>> {
    let side_positions = side_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(position, side_idx)| (side_idx, position))
        .collect::<BTreeMap<_, _>>();
    let component_hazards = local_hazards
        .iter()
        .copied()
        .filter_map(|hazard| {
            let left_position = side_positions.get(&hazard.left_side_idx).copied()?;
            let right_position = side_positions.get(&hazard.right_side_idx).copied()?;
            Some((hazard, left_position, right_position))
        })
        .collect::<Vec<_>>();
    if component_hazards.is_empty() {
        return None;
    }

    let assignments = enumerate_domain_assignments(side_domains);
    let allowed = assignments
        .iter()
        .filter(|assignment| {
            component_hazards
                .iter()
                .all(|(hazard, left_position, right_position)| {
                    assignment[*left_position] != hazard.left_neighbor_idx
                        || assignment[*right_position] != hazard.right_neighbor_idx
                })
        })
        .cloned()
        .collect::<Vec<_>>();
    if allowed.len() == assignments.len() {
        None
    } else {
        Some(allowed)
    }
}

pub(crate) fn stereo_constraint_model(
    side_infos: &[StereoSideInfo],
    side_ids_by_component: &[Vec<usize>],
    local_hazards: &[StereoLocalHazard],
) -> PyResult<StereoConstraintModel> {
    let mut input_side_to_component = vec![None; side_infos.len()];

    for (component_idx, side_ids) in side_ids_by_component.iter().enumerate() {
        for &side_idx in side_ids {
            let side_info = side_infos.get(side_idx).ok_or_else(|| {
                PyValueError::new_err("stereo constraint side index out of range")
            })?;
            if side_info.component_idx != component_idx {
                return Err(PyValueError::new_err(
                    "stereo constraint side component mismatch",
                ));
            }
            if input_side_to_component[side_idx]
                .replace(component_idx)
                .is_some()
            {
                return Err(PyValueError::new_err(
                    "stereo constraint side assigned to multiple components",
                ));
            }
            if side_info.candidate_neighbors.len() != side_info.candidate_base_tokens.len() {
                return Err(PyValueError::new_err(
                    "stereo constraint side candidate/token length mismatch",
                ));
            }
        }
    }

    if input_side_to_component.iter().any(Option::is_none) {
        return Err(PyValueError::new_err(
            "stereo constraint side missing from component mapping",
        ));
    }

    let mut component_parents = (0..side_ids_by_component.len()).collect::<Vec<_>>();
    for hazard in local_hazards {
        if hazard.left_side_idx == hazard.right_side_idx {
            return Err(PyValueError::new_err(
                "local hazard cannot use the same side twice",
            ));
        }
        if !side_has_neighbor(side_infos, hazard.left_side_idx, hazard.left_neighbor_idx)
            || !side_has_neighbor(side_infos, hazard.right_side_idx, hazard.right_neighbor_idx)
        {
            return Err(PyValueError::new_err(
                "local hazard references a neighbor outside the side domain",
            ));
        }
        let Some(left_component_idx) = input_side_to_component
            .get(hazard.left_side_idx)
            .and_then(|value| *value)
        else {
            return Err(PyValueError::new_err(
                "local hazard side index out of range",
            ));
        };
        let Some(right_component_idx) = input_side_to_component
            .get(hazard.right_side_idx)
            .and_then(|value| *value)
        else {
            return Err(PyValueError::new_err(
                "local hazard side index out of range",
            ));
        };
        union_components(
            &mut component_parents,
            left_component_idx,
            right_component_idx,
        );
    }

    let mut component_lookup = BTreeMap::<usize, usize>::new();
    let mut next_component_idx = 0usize;
    let mut input_component_to_model_component = vec![0usize; component_parents.len()];
    for input_component_idx in 0..component_parents.len() {
        let root_idx = find_component(&mut component_parents, input_component_idx);
        let model_component_idx = *component_lookup.entry(root_idx).or_insert_with(|| {
            let current = next_component_idx;
            next_component_idx += 1;
            current
        });
        input_component_to_model_component[input_component_idx] = model_component_idx;
    }

    let mut side_to_component = vec![None; side_infos.len()];
    let mut side_ids_by_model_component = vec![Vec::new(); next_component_idx];
    for side_idx in 0..side_infos.len() {
        let input_component_idx = input_side_to_component[side_idx]
            .ok_or_else(|| PyValueError::new_err("stereo constraint side missing"))?;
        let model_component_idx = input_component_to_model_component[input_component_idx];
        side_to_component[side_idx] = Some(model_component_idx);
        side_ids_by_model_component[model_component_idx].push(side_idx);
    }

    let mut components = Vec::with_capacity(side_ids_by_model_component.len());
    for (component_idx, side_ids) in side_ids_by_model_component.iter().enumerate() {
        let side_domains = side_ids
            .iter()
            .map(|&side_idx| {
                let side_info = &side_infos[side_idx];
                let choices = side_info
                    .candidate_neighbors
                    .iter()
                    .copied()
                    .zip(side_info.candidate_base_tokens.iter().cloned())
                    .map(|(neighbor_idx, base_token)| StereoCarrierChoice {
                        neighbor_idx,
                        base_token,
                    })
                    .collect::<Vec<_>>();
                StereoSideChoiceDomain {
                    side_idx,
                    component_idx,
                    endpoint_atom_idx: side_info.endpoint_atom_idx,
                    choices,
                }
            })
            .collect::<Vec<_>>();
        let local_assignments =
            local_writer_allowed_assignments(side_ids, &side_domains, local_hazards);
        let layer_assignments = StereoConstraintLayer::ALL
            .into_iter()
            .map(|layer| StereoLayerAssignments {
                layer,
                allowed_neighbor_assignments: match layer {
                    StereoConstraintLayer::Semantic => None,
                    StereoConstraintLayer::RdkitLocalWriter
                    | StereoConstraintLayer::RdkitTraversalWriter => local_assignments.clone(),
                },
            })
            .collect::<Vec<_>>();

        components.push(StereoComponentConstraintModel {
            component_idx,
            side_ids: side_ids.clone(),
            side_domains,
            layer_assignments,
        });
    }

    Ok(StereoConstraintModel {
        components,
        side_to_component,
    })
}

impl StereoConstraintModel {
    pub(crate) fn component_count(&self) -> usize {
        self.components.len()
    }

    pub(crate) fn component_for_side(&self, side_idx: usize) -> Option<usize> {
        self.side_to_component
            .get(side_idx)
            .and_then(|value| *value)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn has_completion(
        &self,
        component_idx: usize,
        layer: StereoConstraintLayer,
        facts: &[StereoConstraintFact],
    ) -> bool {
        let Some(component) = self.components.get(component_idx) else {
            return false;
        };
        let Some(layer_assignments) = component
            .layer_assignments
            .iter()
            .find(|assignments| assignments.layer == layer)
        else {
            return false;
        };

        let mut selected_neighbors = BTreeMap::<usize, usize>::new();
        for fact in facts {
            let (side_idx, selected_neighbor) = match *fact {
                StereoConstraintFact::CarrierSelected {
                    side_idx,
                    neighbor_idx,
                } => (side_idx, Some(neighbor_idx)),
                StereoConstraintFact::CarrierEdgeEmitted {
                    side_idx,
                    begin_idx,
                    end_idx,
                    role: _,
                } => {
                    let Some(domain) = component
                        .side_domains
                        .iter()
                        .find(|domain| domain.side_idx == side_idx)
                    else {
                        return false;
                    };
                    if begin_idx != domain.endpoint_atom_idx && end_idx != domain.endpoint_atom_idx
                    {
                        return false;
                    }
                    (side_idx, None)
                }
                StereoConstraintFact::DirectionalMarkerPlaced {
                    side_idx,
                    slot: _,
                    marker,
                    role: _,
                } => {
                    if marker != '/' && marker != '\\' {
                        return false;
                    }
                    (side_idx, None)
                }
            };
            if self
                .side_to_component
                .get(side_idx)
                .and_then(|value| *value)
                != Some(component_idx)
            {
                return false;
            }
            let Some(domain) = component
                .side_domains
                .iter()
                .find(|domain| domain.side_idx == side_idx)
            else {
                return false;
            };
            if !domain.choices.iter().any(|choice| {
                selected_neighbor.is_none_or(|neighbor_idx| choice.neighbor_idx == neighbor_idx)
            }) {
                return false;
            }
            if let Some(neighbor_idx) = selected_neighbor {
                if selected_neighbors
                    .insert(side_idx, neighbor_idx)
                    .is_some_and(|existing_neighbor_idx| existing_neighbor_idx != neighbor_idx)
                {
                    return false;
                }
            }
        }

        match &layer_assignments.allowed_neighbor_assignments {
            None => true,
            Some(allowed_neighbor_assignments) => {
                allowed_neighbor_assignments.iter().any(|assignment| {
                    selected_neighbors.iter().all(|(&side_idx, &neighbor_idx)| {
                        let Some(position) = component
                            .side_ids
                            .iter()
                            .position(|&component_side_idx| component_side_idx == side_idx)
                        else {
                            return false;
                        };
                        assignment.get(position).copied() == Some(neighbor_idx)
                    })
                })
            }
        }
    }
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

pub(crate) fn rdkit_local_writer_hazards(
    graph: &PreparedSmilesGraphData,
    side_infos: &[StereoSideInfo],
) -> Vec<StereoLocalHazard> {
    let mut hazards = Vec::new();
    for bond_idx in 0..graph.bond_count {
        if graph.bond_kinds[bond_idx] != "DOUBLE" || is_stereo_double_bond(graph, bond_idx) {
            continue;
        }
        let begin_idx = graph.bond_begin_atom_indices[bond_idx];
        let end_idx = graph.bond_end_atom_indices[bond_idx];
        let begin_candidates = side_candidates_touching_atom(side_infos, begin_idx, Some(end_idx));
        let end_candidates = side_candidates_touching_atom(side_infos, end_idx, Some(begin_idx));
        for &(left_side_idx, left_neighbor_idx) in &begin_candidates {
            for &(right_side_idx, right_neighbor_idx) in &end_candidates {
                if left_side_idx == right_side_idx {
                    continue;
                }
                hazards.push(StereoLocalHazard {
                    left_side_idx,
                    left_neighbor_idx,
                    right_side_idx,
                    right_neighbor_idx,
                });
            }
        }
    }
    hazards.sort_by_key(|hazard| {
        (
            hazard.left_side_idx,
            hazard.left_neighbor_idx,
            hazard.right_side_idx,
            hazard.right_neighbor_idx,
        )
    });
    hazards.dedup();
    hazards
}

fn side_candidates_touching_atom(
    side_infos: &[StereoSideInfo],
    atom_idx: usize,
    excluded_neighbor_idx: Option<usize>,
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (side_idx, side_info) in side_infos.iter().enumerate() {
        for &neighbor_idx in &side_info.candidate_neighbors {
            let touches_atom = side_info.endpoint_atom_idx == atom_idx || neighbor_idx == atom_idx;
            let touches_excluded = excluded_neighbor_idx.is_some_and(|excluded_idx| {
                side_info.endpoint_atom_idx == excluded_idx || neighbor_idx == excluded_idx
            });
            if touches_atom && !touches_excluded {
                out.push((side_idx, neighbor_idx));
            }
        }
    }
    out
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

#[cfg(test)]
mod tests {
    use super::{
        stereo_constraint_model, StereoConstraintFact, StereoConstraintLayer, StereoLocalHazard,
        StereoSideInfo, StereoTraversalRole,
    };

    #[test]
    fn constraint_model_accepts_empty_no_stereo_shape() {
        let model = stereo_constraint_model(&[], &[], &[]).expect("empty model should build");

        assert_eq!(0, model.component_count());
        assert!(!model.has_completion(0, StereoConstraintLayer::Semantic, &[]));
    }

    #[test]
    fn constraint_model_builds_component_side_domains() {
        let side_infos = vec![
            StereoSideInfo {
                component_idx: 0,
                endpoint_atom_idx: 1,
                other_endpoint_atom_idx: 2,
                candidate_neighbors: vec![0, 3],
                candidate_base_tokens: vec!["/".to_owned(), "\\".to_owned()],
            },
            StereoSideInfo {
                component_idx: 0,
                endpoint_atom_idx: 2,
                other_endpoint_atom_idx: 1,
                candidate_neighbors: vec![4],
                candidate_base_tokens: vec!["/".to_owned()],
            },
            StereoSideInfo {
                component_idx: 1,
                endpoint_atom_idx: 7,
                other_endpoint_atom_idx: 8,
                candidate_neighbors: vec![6],
                candidate_base_tokens: vec!["\\".to_owned()],
            },
        ];

        let model = stereo_constraint_model(&side_infos, &[vec![0, 1], vec![2]], &[])
            .expect("model should build");

        assert_eq!(2, model.component_count());
        assert_eq!(vec![0, 1], model.components[0].side_ids);
        assert_eq!(vec![2], model.components[1].side_ids);
        assert_eq!(2, model.components[0].side_domains[0].choices.len());
        assert_eq!(1, model.components[0].side_domains[1].choices.len());
        assert_eq!(1, model.components[1].side_domains[0].choices.len());
    }

    #[test]
    fn constraint_model_completion_query_validates_domain_facts() {
        let side_infos = vec![StereoSideInfo {
            component_idx: 0,
            endpoint_atom_idx: 1,
            other_endpoint_atom_idx: 2,
            candidate_neighbors: vec![0, 3],
            candidate_base_tokens: vec!["/".to_owned(), "\\".to_owned()],
        }];
        let model =
            stereo_constraint_model(&side_infos, &[vec![0]], &[]).expect("model should build");

        assert!(model.has_completion(
            0,
            StereoConstraintLayer::RdkitLocalWriter,
            &[StereoConstraintFact::CarrierSelected {
                side_idx: 0,
                neighbor_idx: 3,
            }],
        ));
        assert!(model.has_completion(
            0,
            StereoConstraintLayer::RdkitTraversalWriter,
            &[
                StereoConstraintFact::CarrierSelected {
                    side_idx: 0,
                    neighbor_idx: 3,
                },
                StereoConstraintFact::CarrierSelected {
                    side_idx: 0,
                    neighbor_idx: 3,
                },
            ],
        ));
        assert!(!model.has_completion(
            0,
            StereoConstraintLayer::Semantic,
            &[StereoConstraintFact::CarrierSelected {
                side_idx: 0,
                neighbor_idx: 99,
            }],
        ));
        assert!(!model.has_completion(
            1,
            StereoConstraintLayer::Semantic,
            &[StereoConstraintFact::CarrierSelected {
                side_idx: 0,
                neighbor_idx: 3,
            }],
        ));
        assert!(!model.has_completion(
            0,
            StereoConstraintLayer::Semantic,
            &[
                StereoConstraintFact::CarrierSelected {
                    side_idx: 0,
                    neighbor_idx: 0,
                },
                StereoConstraintFact::CarrierSelected {
                    side_idx: 0,
                    neighbor_idx: 3,
                },
            ],
        ));
    }

    #[test]
    fn constraint_model_accepts_traversal_writer_facts() {
        let side_infos = vec![StereoSideInfo {
            component_idx: 0,
            endpoint_atom_idx: 1,
            other_endpoint_atom_idx: 2,
            candidate_neighbors: vec![0, 3],
            candidate_base_tokens: vec!["/".to_owned(), "\\".to_owned()],
        }];
        let model =
            stereo_constraint_model(&side_infos, &[vec![0]], &[]).expect("model should build");

        assert!(model.has_completion(
            0,
            StereoConstraintLayer::RdkitTraversalWriter,
            &[
                StereoConstraintFact::CarrierSelected {
                    side_idx: 0,
                    neighbor_idx: 3,
                },
                StereoConstraintFact::CarrierEdgeEmitted {
                    side_idx: 0,
                    begin_idx: 1,
                    end_idx: 3,
                    role: StereoTraversalRole::RingClose,
                },
                StereoConstraintFact::DirectionalMarkerPlaced {
                    side_idx: 0,
                    slot: 12,
                    marker: '/',
                    role: StereoTraversalRole::RingClose,
                },
            ],
        ));
        assert!(!model.has_completion(
            0,
            StereoConstraintLayer::RdkitTraversalWriter,
            &[StereoConstraintFact::CarrierEdgeEmitted {
                side_idx: 0,
                begin_idx: 7,
                end_idx: 8,
                role: StereoTraversalRole::TreeOrChain,
            }],
        ));
        assert!(!model.has_completion(
            0,
            StereoConstraintLayer::RdkitTraversalWriter,
            &[StereoConstraintFact::DirectionalMarkerPlaced {
                side_idx: 0,
                slot: 12,
                marker: '-',
                role: StereoTraversalRole::TreeOrChain,
            }],
        ));
    }

    #[test]
    fn constraint_model_merges_local_hazard_components_and_filters_writer_layer() {
        let side_infos = vec![
            StereoSideInfo {
                component_idx: 0,
                endpoint_atom_idx: 2,
                other_endpoint_atom_idx: 1,
                candidate_neighbors: vec![3, 8],
                candidate_base_tokens: vec!["/".to_owned(), "\\".to_owned()],
            },
            StereoSideInfo {
                component_idx: 1,
                endpoint_atom_idx: 5,
                other_endpoint_atom_idx: 6,
                candidate_neighbors: vec![4, 8],
                candidate_base_tokens: vec!["\\".to_owned(), "/".to_owned()],
            },
        ];
        let model = stereo_constraint_model(
            &side_infos,
            &[vec![0], vec![1]],
            &[StereoLocalHazard {
                left_side_idx: 0,
                left_neighbor_idx: 3,
                right_side_idx: 1,
                right_neighbor_idx: 4,
            }],
        )
        .expect("model should build");

        assert_eq!(1, model.component_count());
        assert_eq!(vec![0, 1], model.components[0].side_ids);
        assert!(model.has_completion(
            0,
            StereoConstraintLayer::Semantic,
            &[
                StereoConstraintFact::CarrierSelected {
                    side_idx: 0,
                    neighbor_idx: 3,
                },
                StereoConstraintFact::CarrierSelected {
                    side_idx: 1,
                    neighbor_idx: 4,
                },
            ],
        ));
        assert!(!model.has_completion(
            0,
            StereoConstraintLayer::RdkitLocalWriter,
            &[
                StereoConstraintFact::CarrierSelected {
                    side_idx: 0,
                    neighbor_idx: 3,
                },
                StereoConstraintFact::CarrierSelected {
                    side_idx: 1,
                    neighbor_idx: 4,
                },
            ],
        ));
        assert!(model.has_completion(
            0,
            StereoConstraintLayer::RdkitLocalWriter,
            &[
                StereoConstraintFact::CarrierSelected {
                    side_idx: 0,
                    neighbor_idx: 8,
                },
                StereoConstraintFact::CarrierSelected {
                    side_idx: 1,
                    neighbor_idx: 4,
                },
            ],
        ));
    }

    #[test]
    fn constraint_model_rejects_malformed_local_hazards() {
        let side_infos = vec![
            StereoSideInfo {
                component_idx: 0,
                endpoint_atom_idx: 2,
                other_endpoint_atom_idx: 1,
                candidate_neighbors: vec![3],
                candidate_base_tokens: vec!["/".to_owned()],
            },
            StereoSideInfo {
                component_idx: 1,
                endpoint_atom_idx: 5,
                other_endpoint_atom_idx: 6,
                candidate_neighbors: vec![4],
                candidate_base_tokens: vec!["\\".to_owned()],
            },
        ];

        assert!(stereo_constraint_model(
            &side_infos,
            &[vec![0], vec![1]],
            &[StereoLocalHazard {
                left_side_idx: 0,
                left_neighbor_idx: 99,
                right_side_idx: 1,
                right_neighbor_idx: 4,
            }],
        )
        .is_err());
        assert!(stereo_constraint_model(
            &side_infos,
            &[vec![0], vec![1]],
            &[StereoLocalHazard {
                left_side_idx: 0,
                left_neighbor_idx: 3,
                right_side_idx: 0,
                right_neighbor_idx: 3,
            }],
        )
        .is_err());
    }

    #[test]
    fn constraint_model_rejects_inconsistent_component_mapping() {
        let side_infos = vec![StereoSideInfo {
            component_idx: 1,
            endpoint_atom_idx: 1,
            other_endpoint_atom_idx: 2,
            candidate_neighbors: vec![0],
            candidate_base_tokens: vec!["/".to_owned()],
        }];

        assert!(stereo_constraint_model(&side_infos, &[vec![0]], &[]).is_err());
    }

    #[test]
    fn constraint_model_rejects_missing_side_mapping() {
        let side_infos = vec![StereoSideInfo {
            component_idx: 0,
            endpoint_atom_idx: 1,
            other_endpoint_atom_idx: 2,
            candidate_neighbors: vec![0],
            candidate_base_tokens: vec!["/".to_owned()],
        }];

        assert!(stereo_constraint_model(&side_infos, &[vec![]], &[]).is_err());
    }
}
