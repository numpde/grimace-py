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
    // Molecule-level stereo assignments that should remain valid independent
    // of RDKit's particular writer spelling.
    Semantic,
    // RDKit writer exclusions that are local to the stereo component shape.
    RdkitLocalWriter,
    // RDKit writer exclusions that depend on traversal/emission observations.
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum StereoTokenFlip {
    Stored,
    Flipped,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum StereoComponentPhase {
    Stored,
    Flipped,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum StereoDirectionToken {
    Slash,
    Backslash,
}

#[cfg_attr(not(test), allow(dead_code))]
impl StereoDirectionToken {
    pub(crate) fn from_str(token: &str) -> PyResult<Self> {
        match token {
            "/" => Ok(Self::Slash),
            "\\" => Ok(Self::Backslash),
            _ => Err(PyValueError::new_err(format!(
                "Unsupported directional token observation: {token:?}"
            ))),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct StereoTokenBasisFact {
    pub(crate) runtime_component_idx: usize,
    pub(crate) selected_begin_token: StereoDirectionToken,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StereoTokenPhaseAssignment {
    pub(crate) neighbor_assignment_id: usize,
    pub(crate) token_flips: Vec<StereoTokenFlip>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StereoMarkerPlacementRow {
    pub(crate) token_phase_assignment_id: usize,
    pub(crate) marker_neighbor_sets: Vec<Vec<usize>>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct StereoTokenFlipFact {
    pub(crate) runtime_component_idx: usize,
    pub(crate) token_flip: StereoTokenFlip,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RdkitTokenFlipAdjustmentObservations {
    pub(crate) root_begin_side_orientation: bool,
    pub(crate) adjacent_two_candidate_first_emitted: bool,
}

impl RdkitTokenFlipAdjustmentObservations {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) const NONE: Self = Self {
        root_begin_side_orientation: false,
        adjacent_two_candidate_first_emitted: false,
    };

    pub(crate) fn value(self) -> bool {
        self.root_begin_side_orientation ^ self.adjacent_two_candidate_first_emitted
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum StereoTokenObservationFact {
    AllSingleCandidate {
        runtime_component_idx: usize,
        component_phase: StereoComponentPhase,
        rdkit_token_flip_adjustment: RdkitTokenFlipAdjustmentObservations,
    },
    SelectedBeginSide {
        runtime_component_idx: usize,
        component_phase: StereoComponentPhase,
        selected_begin_token: StereoDirectionToken,
        rdkit_token_flip_adjustment: RdkitTokenFlipAdjustmentObservations,
    },
    TwoCandidateBeginSide {
        runtime_component_idx: usize,
        component_phase: StereoComponentPhase,
        selected_begin_token: StereoDirectionToken,
        selected_begin_neighbor_is_first_emitted: Option<bool>,
        rdkit_token_flip_adjustment: RdkitTokenFlipAdjustmentObservations,
    },
}

#[cfg_attr(not(test), allow(dead_code))]
impl StereoTokenObservationFact {
    pub(crate) fn runtime_component_idx(self) -> usize {
        match self {
            Self::AllSingleCandidate {
                runtime_component_idx,
                ..
            }
            | Self::SelectedBeginSide {
                runtime_component_idx,
                ..
            }
            | Self::TwoCandidateBeginSide {
                runtime_component_idx,
                ..
            } => runtime_component_idx,
        }
    }

    pub(crate) fn component_phase(self) -> StereoComponentPhase {
        match self {
            Self::AllSingleCandidate {
                component_phase, ..
            }
            | Self::SelectedBeginSide {
                component_phase, ..
            }
            | Self::TwoCandidateBeginSide {
                component_phase, ..
            } => component_phase,
        }
    }

    pub(crate) fn selected_begin_token(self) -> Option<StereoDirectionToken> {
        match self {
            Self::AllSingleCandidate { .. } => None,
            Self::SelectedBeginSide {
                selected_begin_token,
                ..
            } => Some(selected_begin_token),
            Self::TwoCandidateBeginSide {
                selected_begin_token,
                ..
            } => Some(selected_begin_token),
        }
    }

    pub(crate) fn selected_begin_neighbor_is_first_emitted(self) -> Option<bool> {
        match self {
            Self::AllSingleCandidate { .. } | Self::SelectedBeginSide { .. } => None,
            Self::TwoCandidateBeginSide {
                selected_begin_neighbor_is_first_emitted,
                ..
            } => selected_begin_neighbor_is_first_emitted,
        }
    }

    pub(crate) fn rdkit_token_flip_adjustment(self) -> RdkitTokenFlipAdjustmentObservations {
        match self {
            Self::AllSingleCandidate {
                rdkit_token_flip_adjustment,
                ..
            }
            | Self::SelectedBeginSide {
                rdkit_token_flip_adjustment,
                ..
            }
            | Self::TwoCandidateBeginSide {
                rdkit_token_flip_adjustment,
                ..
            } => rdkit_token_flip_adjustment,
        }
    }

    pub(crate) fn observation_kind(self) -> &'static str {
        match self {
            Self::AllSingleCandidate { .. } => "all_single_candidate",
            Self::SelectedBeginSide { .. } => "selected_begin_side",
            Self::TwoCandidateBeginSide { .. } => "two_candidate_begin_side",
        }
    }

    pub(crate) fn implied_token_flip(self) -> StereoTokenFlip {
        let phase_is_flipped = self.component_phase() == StereoComponentPhase::Flipped;
        let rdkit_token_flip_adjustment = self.rdkit_token_flip_adjustment().value();
        let final_flip = match self {
            Self::AllSingleCandidate { .. } => phase_is_flipped ^ rdkit_token_flip_adjustment,
            Self::SelectedBeginSide {
                component_phase,
                selected_begin_token,
                ..
            } => {
                let observation_flip = match component_phase {
                    StereoComponentPhase::Stored => {
                        selected_begin_token == StereoDirectionToken::Slash
                    }
                    StereoComponentPhase::Flipped => {
                        selected_begin_token == StereoDirectionToken::Backslash
                    }
                };
                phase_is_flipped ^ observation_flip ^ rdkit_token_flip_adjustment
            }
            Self::TwoCandidateBeginSide {
                selected_begin_token,
                selected_begin_neighbor_is_first_emitted,
                ..
            } => {
                if let Some(selected_is_first) = selected_begin_neighbor_is_first_emitted {
                    (selected_is_first == (selected_begin_token == StereoDirectionToken::Slash))
                        ^ rdkit_token_flip_adjustment
                } else {
                    phase_is_flipped ^ rdkit_token_flip_adjustment
                }
            }
        };
        if final_flip {
            StereoTokenFlip::Flipped
        } else {
            StereoTokenFlip::Stored
        }
    }

    pub(crate) fn token_flip_fact(self) -> StereoTokenFlipFact {
        StereoTokenFlipFact {
            runtime_component_idx: self.runtime_component_idx(),
            token_flip: self.implied_token_flip(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StereoComponentConstraintModel {
    pub(crate) component_idx: usize,
    pub(crate) runtime_component_ids: Vec<usize>,
    pub(crate) side_ids: Vec<usize>,
    pub(crate) side_domains: Vec<StereoSideChoiceDomain>,
    pub(crate) all_neighbor_assignments: Vec<Vec<usize>>,
    pub(crate) all_token_phase_assignments: Vec<StereoTokenPhaseAssignment>,
    pub(crate) all_marker_placement_rows: Vec<StereoMarkerPlacementRow>,
    pub(crate) layer_assignments: Vec<StereoLayerAssignments>,
}

#[allow(dead_code)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    CarrierSelectionBlocked {
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

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum StereoMarkerEventFact {
    MarkerPlaced {
        side_idx: usize,
        slot: usize,
        begin_idx: usize,
        end_idx: usize,
        marker: StereoDirectionToken,
        role: StereoTraversalRole,
    },
    NoMarker {
        side_idx: usize,
        slot: usize,
        begin_idx: usize,
        end_idx: usize,
        role: StereoTraversalRole,
    },
}

#[cfg_attr(not(test), allow(dead_code))]
impl StereoMarkerEventFact {
    pub(crate) fn side_idx(self) -> usize {
        match self {
            Self::MarkerPlaced { side_idx, .. } | Self::NoMarker { side_idx, .. } => side_idx,
        }
    }

    pub(crate) fn edge(self) -> (usize, usize) {
        match self {
            Self::MarkerPlaced {
                begin_idx, end_idx, ..
            }
            | Self::NoMarker {
                begin_idx, end_idx, ..
            } => (begin_idx, end_idx),
        }
    }

    pub(crate) fn is_marker_placed(self) -> bool {
        matches!(self, Self::MarkerPlaced { .. })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct StereoConstraintModel {
    pub(crate) components: Vec<StereoComponentConstraintModel>,
    side_to_component: Vec<Option<usize>>,
    runtime_component_to_component: Vec<Option<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StereoAssignmentState {
    pub(crate) remaining_by_component: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StereoConstraintState {
    pub(crate) carrier_assignment_state: StereoAssignmentState,
    pub(crate) token_phase_remaining_by_component: Vec<Vec<usize>>,
}

struct StereoCarrierFactConstraints {
    selected_neighbors: BTreeMap<usize, usize>,
    blocked_neighbors: BTreeSet<(usize, usize)>,
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

fn enumerate_token_phase_assignments(
    neighbor_assignments: &[Vec<usize>],
    runtime_component_count: usize,
) -> Vec<StereoTokenPhaseAssignment> {
    fn rec(
        neighbor_assignment_id: usize,
        runtime_component_count: usize,
        offset: usize,
        current: &mut Vec<StereoTokenFlip>,
        out: &mut Vec<StereoTokenPhaseAssignment>,
    ) {
        if offset == runtime_component_count {
            out.push(StereoTokenPhaseAssignment {
                neighbor_assignment_id,
                token_flips: current.clone(),
            });
            return;
        }
        for token_flip in [StereoTokenFlip::Stored, StereoTokenFlip::Flipped] {
            current.push(token_flip);
            rec(
                neighbor_assignment_id,
                runtime_component_count,
                offset + 1,
                current,
                out,
            );
            current.pop();
        }
    }

    let mut out = Vec::new();
    for (neighbor_assignment_id, _) in neighbor_assignments.iter().enumerate() {
        rec(
            neighbor_assignment_id,
            runtime_component_count,
            0,
            &mut Vec::new(),
            &mut out,
        );
    }
    out
}

fn marker_neighbor_domain(domain: &StereoSideChoiceDomain) -> Vec<usize> {
    let mut neighbors = Vec::new();
    for choice in &domain.choices {
        if !neighbors.contains(&choice.neighbor_idx) {
            neighbors.push(choice.neighbor_idx);
        }
    }
    neighbors
}

fn marker_neighbor_set_domain(domain: &StereoSideChoiceDomain) -> Vec<Vec<usize>> {
    let neighbors = marker_neighbor_domain(domain);
    if neighbors.is_empty() {
        return Vec::new();
    }
    (1usize..(1usize << neighbors.len()))
        .map(|mask| {
            neighbors
                .iter()
                .enumerate()
                .filter_map(|(offset, &neighbor_idx)| {
                    ((mask & (1usize << offset)) != 0).then_some(neighbor_idx)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn enumerate_marker_placement_rows(
    side_domains: &[StereoSideChoiceDomain],
    token_phase_assignments: &[StereoTokenPhaseAssignment],
) -> Vec<StereoMarkerPlacementRow> {
    fn rec(
        marker_domains: &[Vec<Vec<usize>>],
        token_phase_assignment_id: usize,
        offset: usize,
        current: &mut Vec<Vec<usize>>,
        out: &mut Vec<StereoMarkerPlacementRow>,
    ) {
        if offset == marker_domains.len() {
            out.push(StereoMarkerPlacementRow {
                token_phase_assignment_id,
                marker_neighbor_sets: current.clone(),
            });
            return;
        }
        for neighbor_set in &marker_domains[offset] {
            current.push(neighbor_set.clone());
            rec(
                marker_domains,
                token_phase_assignment_id,
                offset + 1,
                current,
                out,
            );
            current.pop();
        }
    }

    let marker_domains = side_domains
        .iter()
        .map(marker_neighbor_set_domain)
        .collect::<Vec<_>>();
    if marker_domains.iter().any(Vec::is_empty) {
        return Vec::new();
    }

    let mut out = Vec::new();
    for token_phase_assignment_id in 0..token_phase_assignments.len() {
        rec(
            &marker_domains,
            token_phase_assignment_id,
            0,
            &mut Vec::<Vec<usize>>::new(),
            &mut out,
        );
    }
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

    let mut runtime_component_to_component = vec![
        None;
        side_infos
            .iter()
            .map(|side_info| side_info.component_idx)
            .max()
            .map(|max_idx| max_idx + 1)
            .unwrap_or(0)
    ];
    let mut components = Vec::with_capacity(side_ids_by_model_component.len());
    for (component_idx, side_ids) in side_ids_by_model_component.iter().enumerate() {
        let runtime_component_ids = side_ids
            .iter()
            .map(|&side_idx| side_infos[side_idx].component_idx)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        for &runtime_component_idx in &runtime_component_ids {
            if runtime_component_to_component[runtime_component_idx]
                .replace(component_idx)
                .is_some()
            {
                return Err(PyValueError::new_err(
                    "runtime stereo component assigned to multiple model components",
                ));
            }
        }
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
        let all_neighbor_assignments = enumerate_domain_assignments(&side_domains);
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

        let all_token_phase_assignments = enumerate_token_phase_assignments(
            &all_neighbor_assignments,
            runtime_component_ids.len(),
        );
        let all_marker_placement_rows =
            enumerate_marker_placement_rows(&side_domains, &all_token_phase_assignments);

        components.push(StereoComponentConstraintModel {
            component_idx,
            runtime_component_ids: runtime_component_ids.clone(),
            side_ids: side_ids.clone(),
            side_domains,
            all_token_phase_assignments,
            all_marker_placement_rows,
            all_neighbor_assignments,
            layer_assignments,
        });
    }

    Ok(StereoConstraintModel {
        components,
        side_to_component,
        runtime_component_to_component,
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

    pub(crate) fn component_for_runtime_component(
        &self,
        runtime_component_idx: usize,
    ) -> Option<usize> {
        self.runtime_component_to_component
            .get(runtime_component_idx)
            .and_then(|value| *value)
    }

    fn carrier_constraints_from_facts(
        &self,
        component: &StereoComponentConstraintModel,
        component_idx: usize,
        facts: &[StereoConstraintFact],
    ) -> Option<StereoCarrierFactConstraints> {
        let mut selected_neighbors = BTreeMap::<usize, usize>::new();
        let mut blocked_neighbors = BTreeSet::<(usize, usize)>::new();
        for fact in facts {
            let (side_idx, selected_neighbor, blocked_neighbor) = match *fact {
                StereoConstraintFact::CarrierSelected {
                    side_idx,
                    neighbor_idx,
                } => (side_idx, Some(neighbor_idx), None),
                StereoConstraintFact::CarrierSelectionBlocked {
                    side_idx,
                    neighbor_idx,
                } => (side_idx, None, Some(neighbor_idx)),
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
                        return None;
                    };
                    if begin_idx != domain.endpoint_atom_idx && end_idx != domain.endpoint_atom_idx
                    {
                        return None;
                    }
                    (side_idx, None, None)
                }
                StereoConstraintFact::DirectionalMarkerPlaced {
                    side_idx,
                    slot: _,
                    marker,
                    role: _,
                } => {
                    if marker != '/' && marker != '\\' {
                        return None;
                    }
                    (side_idx, None, None)
                }
            };
            if self
                .side_to_component
                .get(side_idx)
                .and_then(|value| *value)
                != Some(component_idx)
            {
                return None;
            }
            let Some(domain) = component
                .side_domains
                .iter()
                .find(|domain| domain.side_idx == side_idx)
            else {
                return None;
            };
            let constrained_neighbor = selected_neighbor.or(blocked_neighbor);
            if !domain.choices.iter().any(|choice| {
                constrained_neighbor.is_none_or(|neighbor_idx| choice.neighbor_idx == neighbor_idx)
            }) {
                return None;
            }
            if let Some(neighbor_idx) = selected_neighbor {
                if selected_neighbors
                    .insert(side_idx, neighbor_idx)
                    .is_some_and(|existing_neighbor_idx| existing_neighbor_idx != neighbor_idx)
                {
                    return None;
                }
            }
            if let Some(neighbor_idx) = blocked_neighbor {
                blocked_neighbors.insert((side_idx, neighbor_idx));
            }
        }
        Some(StereoCarrierFactConstraints {
            selected_neighbors,
            blocked_neighbors,
        })
    }

    fn layer_allows_assignment(
        layer_assignments: &StereoLayerAssignments,
        assignment: &[usize],
    ) -> bool {
        match &layer_assignments.allowed_neighbor_assignments {
            None => true,
            Some(allowed_neighbor_assignments) => allowed_neighbor_assignments
                .iter()
                .any(|allowed_assignment| allowed_assignment == assignment),
        }
    }

    fn assignment_matches_carrier_constraints(
        component: &StereoComponentConstraintModel,
        assignment: &[usize],
        constraints: &StereoCarrierFactConstraints,
    ) -> bool {
        let selected_match =
            constraints
                .selected_neighbors
                .iter()
                .all(|(&side_idx, &neighbor_idx)| {
                    let Some(position) = component
                        .side_ids
                        .iter()
                        .position(|&component_side_idx| component_side_idx == side_idx)
                    else {
                        return false;
                    };
                    assignment.get(position).copied() == Some(neighbor_idx)
                });
        if !selected_match {
            return false;
        }
        constraints
            .blocked_neighbors
            .iter()
            .all(|&(side_idx, neighbor_idx)| {
                let Some(position) = component
                    .side_ids
                    .iter()
                    .position(|&component_side_idx| component_side_idx == side_idx)
                else {
                    return false;
                };
                assignment.get(position).copied() != Some(neighbor_idx)
            })
    }

    pub(crate) fn remaining_assignment_ids(
        &self,
        component_idx: usize,
        layer: StereoConstraintLayer,
        facts: &[StereoConstraintFact],
    ) -> Vec<usize> {
        let Some(component) = self.components.get(component_idx) else {
            return Vec::new();
        };
        let Some(layer_assignments) = component
            .layer_assignments
            .iter()
            .find(|assignments| assignments.layer == layer)
        else {
            return Vec::new();
        };
        let Some(carrier_constraints) =
            self.carrier_constraints_from_facts(component, component_idx, facts)
        else {
            return Vec::new();
        };

        component
            .all_neighbor_assignments
            .iter()
            .enumerate()
            .filter_map(|(assignment_id, assignment)| {
                (Self::layer_allows_assignment(layer_assignments, assignment)
                    && Self::assignment_matches_carrier_constraints(
                        component,
                        assignment,
                        &carrier_constraints,
                    ))
                .then_some(assignment_id)
            })
            .collect()
    }

    pub(crate) fn forced_neighbor_for_assignment_ids(
        &self,
        component_idx: usize,
        side_idx: usize,
        assignment_ids: &[usize],
    ) -> Option<usize> {
        let component = self.components.get(component_idx)?;
        let position = component
            .side_ids
            .iter()
            .position(|&component_side_idx| component_side_idx == side_idx)?;
        let mut values = assignment_ids.iter().filter_map(|&assignment_id| {
            component
                .all_neighbor_assignments
                .get(assignment_id)
                .and_then(|assignment| assignment.get(position))
                .copied()
        });
        let first = values.next()?;
        values.all(|value| value == first).then_some(first)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn available_neighbors_for_assignment_ids(
        &self,
        component_idx: usize,
        side_idx: usize,
        assignment_ids: &[usize],
    ) -> Vec<usize> {
        let Some(component) = self.components.get(component_idx) else {
            return Vec::new();
        };
        let Some(position) = component
            .side_ids
            .iter()
            .position(|&component_side_idx| component_side_idx == side_idx)
        else {
            return Vec::new();
        };
        assignment_ids
            .iter()
            .filter_map(|&assignment_id| {
                component
                    .all_neighbor_assignments
                    .get(assignment_id)
                    .and_then(|assignment| assignment.get(position))
                    .copied()
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub(crate) fn token_phase_assignment_ids_for_neighbor_assignment_ids(
        &self,
        component_idx: usize,
        neighbor_assignment_ids: &[usize],
        token_flip_facts: &[StereoTokenFlipFact],
    ) -> PyResult<Vec<usize>> {
        let Some(component) = self.components.get(component_idx) else {
            return Err(PyValueError::new_err(
                "token phase query component index out of range",
            ));
        };
        let neighbor_assignment_ids = neighbor_assignment_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let mut token_flip_positions = BTreeMap::<usize, StereoTokenFlip>::new();
        for fact in token_flip_facts {
            let Some(position) = component
                .runtime_component_ids
                .iter()
                .position(|&idx| idx == fact.runtime_component_idx)
            else {
                return Err(PyValueError::new_err(
                    "token flip fact runtime component outside model component",
                ));
            };
            if token_flip_positions
                .insert(position, fact.token_flip)
                .is_some_and(|existing| existing != fact.token_flip)
            {
                return Ok(Vec::new());
            }
        }
        let assignment_ids = component
            .all_token_phase_assignments
            .iter()
            .enumerate()
            .filter_map(|(assignment_id, assignment)| {
                (neighbor_assignment_ids.contains(&assignment.neighbor_assignment_id)
                    && token_flip_positions.iter().all(|(&position, &token_flip)| {
                        assignment.token_flips.get(position).copied() == Some(token_flip)
                    }))
                .then_some(assignment_id)
            })
            .collect();
        Ok(assignment_ids)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn neighbor_assignment_ids_for_token_phase_assignment_ids(
        &self,
        component_idx: usize,
        token_phase_assignment_ids: &[usize],
    ) -> PyResult<Vec<usize>> {
        let Some(component) = self.components.get(component_idx) else {
            return Err(PyValueError::new_err(
                "token phase query component index out of range",
            ));
        };

        let mut neighbor_assignment_ids = BTreeSet::new();
        for &assignment_id in token_phase_assignment_ids {
            let Some(assignment) = component.all_token_phase_assignments.get(assignment_id) else {
                return Err(PyValueError::new_err(
                    "token phase assignment index out of range",
                ));
            };
            neighbor_assignment_ids.insert(assignment.neighbor_assignment_id);
        }
        Ok(neighbor_assignment_ids.into_iter().collect())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn token_phase_assignment_ids_for_token_observation_facts(
        &self,
        component_idx: usize,
        neighbor_assignment_ids: &[usize],
        observation_facts: &[StereoTokenObservationFact],
    ) -> PyResult<Vec<usize>> {
        let token_flip_facts = observation_facts
            .iter()
            .map(|fact| fact.token_flip_fact())
            .collect::<Vec<_>>();
        self.token_phase_assignment_ids_for_neighbor_assignment_ids(
            component_idx,
            neighbor_assignment_ids,
            &token_flip_facts,
        )
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn filter_token_phase_assignment_ids_for_token_flip(
        &self,
        component_idx: usize,
        runtime_component_idx: usize,
        token_phase_assignment_ids: &[usize],
        token_flip: StereoTokenFlip,
    ) -> PyResult<Vec<usize>> {
        let Some(component) = self.components.get(component_idx) else {
            return Err(PyValueError::new_err(
                "token phase filter component index out of range",
            ));
        };
        let Some(runtime_component_position) = component
            .runtime_component_ids
            .iter()
            .position(|&idx| idx == runtime_component_idx)
        else {
            return Err(PyValueError::new_err(
                "token phase filter runtime component outside model component",
            ));
        };
        let mut filtered = Vec::new();
        for &assignment_id in token_phase_assignment_ids {
            let Some(assignment) = component.all_token_phase_assignments.get(assignment_id) else {
                return Err(PyValueError::new_err(
                    "token phase filter assignment index out of range",
                ));
            };
            if assignment
                .token_flips
                .get(runtime_component_position)
                .copied()
                == Some(token_flip)
            {
                filtered.push(assignment_id);
            }
        }
        Ok(filtered)
    }

    pub(crate) fn forced_token_flip_for_token_phase_assignment_ids(
        &self,
        component_idx: usize,
        runtime_component_idx: usize,
        assignment_ids: &[usize],
    ) -> Option<StereoTokenFlip> {
        let component = self.components.get(component_idx)?;
        let position = component
            .runtime_component_ids
            .iter()
            .position(|&idx| idx == runtime_component_idx)?;
        let mut values = assignment_ids.iter().filter_map(|&assignment_id| {
            component
                .all_token_phase_assignments
                .get(assignment_id)
                .and_then(|assignment| assignment.token_flips.get(position))
                .copied()
        });
        let first = values.next()?;
        values.all(|value| value == first).then_some(first)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn marker_placement_row_ids_for_token_phase_assignment_ids(
        &self,
        component_idx: usize,
        token_phase_assignment_ids: &[usize],
    ) -> PyResult<Vec<usize>> {
        let Some(component) = self.components.get(component_idx) else {
            return Err(PyValueError::new_err(
                "marker placement query component index out of range",
            ));
        };
        for &assignment_id in token_phase_assignment_ids {
            if assignment_id >= component.all_token_phase_assignments.len() {
                return Err(PyValueError::new_err(
                    "marker placement query token phase assignment index out of range",
                ));
            }
        }
        let token_phase_assignment_ids = token_phase_assignment_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();

        Ok(component
            .all_marker_placement_rows
            .iter()
            .enumerate()
            .filter_map(|(row_id, row)| {
                token_phase_assignment_ids
                    .contains(&row.token_phase_assignment_id)
                    .then_some(row_id)
            })
            .collect())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn token_phase_assignment_ids_for_marker_placement_row_ids(
        &self,
        component_idx: usize,
        row_ids: &[usize],
    ) -> PyResult<Vec<usize>> {
        let Some(component) = self.components.get(component_idx) else {
            return Err(PyValueError::new_err(
                "marker placement query component index out of range",
            ));
        };

        let mut token_phase_assignment_ids = BTreeSet::new();
        for &row_id in row_ids {
            let Some(row) = component.all_marker_placement_rows.get(row_id) else {
                return Err(PyValueError::new_err(
                    "marker placement row index out of range",
                ));
            };
            token_phase_assignment_ids.insert(row.token_phase_assignment_id);
        }
        Ok(token_phase_assignment_ids.into_iter().collect())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn neighbor_assignment_ids_for_marker_placement_row_ids(
        &self,
        component_idx: usize,
        row_ids: &[usize],
    ) -> PyResult<Vec<usize>> {
        let token_phase_assignment_ids =
            self.token_phase_assignment_ids_for_marker_placement_row_ids(component_idx, row_ids)?;
        self.neighbor_assignment_ids_for_token_phase_assignment_ids(
            component_idx,
            &token_phase_assignment_ids,
        )
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn marker_neighbor_sets_for_marker_placement_row_ids(
        &self,
        component_idx: usize,
        side_idx: usize,
        row_ids: &[usize],
    ) -> PyResult<Vec<Vec<usize>>> {
        let Some(component) = self.components.get(component_idx) else {
            return Err(PyValueError::new_err(
                "marker placement query component index out of range",
            ));
        };
        let Some(side_position) = component
            .side_ids
            .iter()
            .position(|&component_side_idx| component_side_idx == side_idx)
        else {
            return Err(PyValueError::new_err(
                "marker placement query side outside model component",
            ));
        };

        let mut marker_neighbor_sets = BTreeSet::new();
        for &row_id in row_ids {
            let Some(row) = component.all_marker_placement_rows.get(row_id) else {
                return Err(PyValueError::new_err(
                    "marker placement row index out of range",
                ));
            };
            let Some(neighbor_set) = row.marker_neighbor_sets.get(side_position) else {
                return Err(PyValueError::new_err(
                    "marker placement row side index out of range",
                ));
            };
            marker_neighbor_sets.insert(neighbor_set.clone());
        }
        Ok(marker_neighbor_sets.into_iter().collect())
    }

    fn marker_event_target(
        component: &StereoComponentConstraintModel,
        event: StereoMarkerEventFact,
    ) -> PyResult<Option<(usize, usize)>> {
        let side_idx = event.side_idx();
        let Some(side_position) = component
            .side_ids
            .iter()
            .position(|&component_side_idx| component_side_idx == side_idx)
        else {
            return Err(PyValueError::new_err(
                "marker event side outside model component",
            ));
        };
        let domain = &component.side_domains[side_position];
        let (begin_idx, end_idx) = event.edge();
        let neighbor_idx = if begin_idx == domain.endpoint_atom_idx {
            end_idx
        } else if end_idx == domain.endpoint_atom_idx {
            begin_idx
        } else {
            return Ok(None);
        };
        if domain
            .choices
            .iter()
            .any(|choice| choice.neighbor_idx == neighbor_idx)
        {
            Ok(Some((side_position, neighbor_idx)))
        } else {
            Ok(None)
        }
    }

    fn marker_placement_row_matches_event(
        component: &StereoComponentConstraintModel,
        row: &StereoMarkerPlacementRow,
        event: StereoMarkerEventFact,
    ) -> PyResult<bool> {
        let Some((side_position, event_neighbor_idx)) =
            Self::marker_event_target(component, event)?
        else {
            return Ok(!event.is_marker_placed());
        };
        let row_neighbor_set = row.marker_neighbor_sets.get(side_position);
        if event.is_marker_placed() {
            Ok(row_neighbor_set.is_some_and(|neighbors| neighbors.contains(&event_neighbor_idx)))
        } else {
            Ok(!row_neighbor_set.is_some_and(|neighbors| neighbors.contains(&event_neighbor_idx)))
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn filter_marker_placement_row_ids_for_marker_event_facts(
        &self,
        component_idx: usize,
        row_ids: &[usize],
        marker_event_facts: &[StereoMarkerEventFact],
    ) -> PyResult<Vec<usize>> {
        let Some(component) = self.components.get(component_idx) else {
            return Err(PyValueError::new_err(
                "marker placement query component index out of range",
            ));
        };

        let mut filtered_row_ids = Vec::new();
        for &row_id in row_ids {
            let Some(row) = component.all_marker_placement_rows.get(row_id) else {
                return Err(PyValueError::new_err(
                    "marker placement row index out of range",
                ));
            };
            let mut keep = true;
            for &event in marker_event_facts {
                if !Self::marker_placement_row_matches_event(component, row, event)? {
                    keep = false;
                    break;
                }
            }
            if keep {
                filtered_row_ids.push(row_id);
            }
        }
        Ok(filtered_row_ids)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn has_completion(
        &self,
        component_idx: usize,
        layer: StereoConstraintLayer,
        facts: &[StereoConstraintFact],
    ) -> bool {
        !self
            .remaining_assignment_ids(component_idx, layer, facts)
            .is_empty()
    }
}

impl StereoAssignmentState {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_model(model: &StereoConstraintModel, layer: StereoConstraintLayer) -> Self {
        Self::from_facts_by_component(model, layer, &[])
    }

    pub(crate) fn from_facts_by_component(
        model: &StereoConstraintModel,
        layer: StereoConstraintLayer,
        facts_by_component: &[Vec<StereoConstraintFact>],
    ) -> Self {
        let empty_facts = Vec::new();
        let remaining_by_component = model
            .components
            .iter()
            .map(|component| {
                let facts = facts_by_component
                    .get(component.component_idx)
                    .unwrap_or(&empty_facts);
                model.remaining_assignment_ids(component.component_idx, layer, facts)
            })
            .collect();
        Self {
            remaining_by_component,
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn filter_facts_by_component(
        &self,
        model: &StereoConstraintModel,
        layer: StereoConstraintLayer,
        facts_by_component: &[Vec<StereoConstraintFact>],
    ) -> Self {
        let filtered = Self::from_facts_by_component(model, layer, facts_by_component);
        let remaining_by_component = self
            .remaining_by_component
            .iter()
            .zip(filtered.remaining_by_component.iter())
            .map(|(current_ids, filtered_ids)| {
                current_ids
                    .iter()
                    .copied()
                    .filter(|assignment_id| filtered_ids.contains(assignment_id))
                    .collect()
            })
            .collect();
        Self {
            remaining_by_component,
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn is_empty(&self, component_idx: usize) -> bool {
        self.remaining_by_component
            .get(component_idx)
            .is_none_or(Vec::is_empty)
    }

    pub(crate) fn forced_neighbor(
        &self,
        model: &StereoConstraintModel,
        component_idx: usize,
        side_idx: usize,
    ) -> Option<usize> {
        let assignment_ids = self.remaining_by_component.get(component_idx)?;
        model.forced_neighbor_for_assignment_ids(component_idx, side_idx, assignment_ids)
    }
}

impl StereoConstraintState {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_facts(
        model: &StereoConstraintModel,
        layer: StereoConstraintLayer,
        facts_by_component: &[Vec<StereoConstraintFact>],
        token_flip_facts: &[StereoTokenFlipFact],
    ) -> PyResult<Self> {
        for fact in token_flip_facts {
            if model
                .component_for_runtime_component(fact.runtime_component_idx)
                .is_none()
            {
                return Err(PyValueError::new_err(
                    "token flip fact references unknown runtime component",
                ));
            }
        }

        let carrier_assignment_state =
            StereoAssignmentState::from_facts_by_component(model, layer, facts_by_component);
        let token_phase_remaining_by_component = model
            .components
            .iter()
            .map(|component| {
                let component_token_flip_facts = token_flip_facts
                    .iter()
                    .copied()
                    .filter(|fact| {
                        model.component_for_runtime_component(fact.runtime_component_idx)
                            == Some(component.component_idx)
                    })
                    .collect::<Vec<_>>();
                let remaining_neighbor_assignment_ids = carrier_assignment_state
                    .remaining_by_component
                    .get(component.component_idx)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                model.token_phase_assignment_ids_for_neighbor_assignment_ids(
                    component.component_idx,
                    remaining_neighbor_assignment_ids,
                    &component_token_flip_facts,
                )
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self {
            carrier_assignment_state,
            token_phase_remaining_by_component,
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_token_observation_facts(
        model: &StereoConstraintModel,
        layer: StereoConstraintLayer,
        facts_by_component: &[Vec<StereoConstraintFact>],
        token_observation_facts: &[StereoTokenObservationFact],
    ) -> PyResult<Self> {
        for fact in token_observation_facts {
            if model
                .component_for_runtime_component(fact.runtime_component_idx())
                .is_none()
            {
                return Err(PyValueError::new_err(
                    "token observation fact references unknown runtime component",
                ));
            }
        }

        let carrier_assignment_state =
            StereoAssignmentState::from_facts_by_component(model, layer, facts_by_component);
        let token_phase_remaining_by_component = model
            .components
            .iter()
            .map(|component| {
                let component_token_observation_facts = token_observation_facts
                    .iter()
                    .copied()
                    .filter(|fact| {
                        model.component_for_runtime_component(fact.runtime_component_idx())
                            == Some(component.component_idx)
                    })
                    .collect::<Vec<_>>();
                let remaining_neighbor_assignment_ids = carrier_assignment_state
                    .remaining_by_component
                    .get(component.component_idx)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                model.token_phase_assignment_ids_for_token_observation_facts(
                    component.component_idx,
                    remaining_neighbor_assignment_ids,
                    &component_token_observation_facts,
                )
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self {
            carrier_assignment_state,
            token_phase_remaining_by_component,
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_facts_and_token_observations(
        model: &StereoConstraintModel,
        layer: StereoConstraintLayer,
        facts_by_component: &[Vec<StereoConstraintFact>],
        token_flip_facts: &[StereoTokenFlipFact],
        token_observation_facts: &[StereoTokenObservationFact],
    ) -> PyResult<Self> {
        for fact in token_flip_facts {
            if model
                .component_for_runtime_component(fact.runtime_component_idx)
                .is_none()
            {
                return Err(PyValueError::new_err(
                    "token flip fact references unknown runtime component",
                ));
            }
        }
        for fact in token_observation_facts {
            if model
                .component_for_runtime_component(fact.runtime_component_idx())
                .is_none()
            {
                return Err(PyValueError::new_err(
                    "token observation fact references unknown runtime component",
                ));
            }
        }

        let mut combined_token_flip_facts = token_flip_facts.to_vec();
        combined_token_flip_facts.extend(
            token_observation_facts
                .iter()
                .map(|fact| fact.token_flip_fact()),
        );
        Self::from_facts(model, layer, facts_by_component, &combined_token_flip_facts)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn is_empty(&self, component_idx: usize) -> bool {
        self.carrier_assignment_state.is_empty(component_idx)
            || self
                .token_phase_remaining_by_component
                .get(component_idx)
                .is_none_or(Vec::is_empty)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn forced_neighbor(
        &self,
        model: &StereoConstraintModel,
        component_idx: usize,
        side_idx: usize,
    ) -> Option<usize> {
        self.carrier_assignment_state
            .forced_neighbor(model, component_idx, side_idx)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn forced_token_flip(
        &self,
        model: &StereoConstraintModel,
        component_idx: usize,
        runtime_component_idx: usize,
    ) -> Option<StereoTokenFlip> {
        let assignment_ids = self.token_phase_remaining_by_component.get(component_idx)?;
        model.forced_token_flip_for_token_phase_assignment_ids(
            component_idx,
            runtime_component_idx,
            assignment_ids,
        )
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn available_token_flips(
        &self,
        model: &StereoConstraintModel,
        component_idx: usize,
        runtime_component_idx: usize,
    ) -> Vec<StereoTokenFlip> {
        let Some(component) = model.components.get(component_idx) else {
            return Vec::new();
        };
        let Some(position) = component
            .runtime_component_ids
            .iter()
            .position(|&idx| idx == runtime_component_idx)
        else {
            return Vec::new();
        };
        let mut flips = self
            .token_phase_remaining_by_component
            .get(component_idx)
            .into_iter()
            .flatten()
            .filter_map(|&assignment_id| {
                component
                    .all_token_phase_assignments
                    .get(assignment_id)
                    .and_then(|assignment| assignment.token_flips.get(position))
                    .copied()
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        flips.sort();
        flips
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
        stereo_constraint_model, RdkitTokenFlipAdjustmentObservations, StereoComponentPhase,
        StereoConstraintFact, StereoConstraintLayer, StereoDirectionToken, StereoLocalHazard,
        StereoMarkerEventFact, StereoSideInfo, StereoTokenFlip, StereoTokenFlipFact,
        StereoTokenObservationFact, StereoTraversalRole,
    };

    fn rdkit_adjustment(value: bool) -> RdkitTokenFlipAdjustmentObservations {
        if value {
            RdkitTokenFlipAdjustmentObservations {
                root_begin_side_orientation: true,
                adjacent_two_candidate_first_emitted: false,
            }
        } else {
            RdkitTokenFlipAdjustmentObservations::NONE
        }
    }

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
    fn marker_placement_rows_extend_token_phase_rows_with_visible_marker_choices() {
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
        ];
        let model =
            stereo_constraint_model(&side_infos, &[vec![0, 1]], &[]).expect("model should build");
        let component = &model.components[0];

        assert_eq!(2, component.all_neighbor_assignments.len());
        assert_eq!(4, component.all_token_phase_assignments.len());
        assert_eq!(12, component.all_marker_placement_rows.len());

        let first_token_phase_rows = component
            .all_marker_placement_rows
            .iter()
            .filter(|row| row.token_phase_assignment_id == 0)
            .map(|row| row.marker_neighbor_sets.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            vec![
                vec![vec![0], vec![4]],
                vec![vec![3], vec![4]],
                vec![vec![0, 3], vec![4]]
            ],
            first_token_phase_rows,
        );

        for row in &component.all_marker_placement_rows {
            let token_phase_assignment =
                &component.all_token_phase_assignments[row.token_phase_assignment_id];
            assert!(
                token_phase_assignment.neighbor_assignment_id
                    < component.all_neighbor_assignments.len()
            );
            assert_eq!(component.side_ids.len(), row.marker_neighbor_sets.len());
            for (side_position, marker_neighbor_set) in row.marker_neighbor_sets.iter().enumerate()
            {
                assert!(!marker_neighbor_set.is_empty());
                for &marker_neighbor_idx in marker_neighbor_set {
                    assert!(component.side_domains[side_position]
                        .choices
                        .iter()
                        .any(|choice| choice.neighbor_idx == marker_neighbor_idx));
                }
            }
        }
    }

    #[test]
    fn marker_placement_rows_preserve_merged_component_token_phase_dimension() {
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
        let component = &model.components[0];

        assert_eq!(vec![0, 1], component.runtime_component_ids);
        assert_eq!(4, component.all_neighbor_assignments.len());
        assert_eq!(16, component.all_token_phase_assignments.len());
        assert_eq!(144, component.all_marker_placement_rows.len());

        let first_token_phase_rows = component
            .all_marker_placement_rows
            .iter()
            .filter(|row| row.token_phase_assignment_id == 0)
            .map(|row| row.marker_neighbor_sets.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            vec![
                vec![vec![3], vec![4]],
                vec![vec![3], vec![8]],
                vec![vec![3], vec![4, 8]],
                vec![vec![8], vec![4]],
                vec![vec![8], vec![8]],
                vec![vec![8], vec![4, 8]],
                vec![vec![3, 8], vec![4]],
                vec![vec![3, 8], vec![8]],
                vec![vec![3, 8], vec![4, 8]]
            ],
            first_token_phase_rows,
        );
    }

    #[test]
    fn marker_placement_row_query_filters_by_token_phase_assignment_ids() {
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
        ];
        let model =
            stereo_constraint_model(&side_infos, &[vec![0, 1]], &[]).expect("model should build");

        assert_eq!(
            vec![0, 1, 2],
            model
                .marker_placement_row_ids_for_token_phase_assignment_ids(0, &[0])
                .expect("marker placement query should be valid"),
        );
        assert_eq!(
            vec![3, 4, 5, 9, 10, 11],
            model
                .marker_placement_row_ids_for_token_phase_assignment_ids(0, &[1, 3])
                .expect("marker placement query should be valid"),
        );
        assert!(model
            .marker_placement_row_ids_for_token_phase_assignment_ids(1, &[0])
            .is_err());
        assert!(model
            .marker_placement_row_ids_for_token_phase_assignment_ids(0, &[4])
            .is_err());
    }

    #[test]
    fn marker_placement_event_filter_handles_positive_and_negative_events() {
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
        ];
        let model =
            stereo_constraint_model(&side_infos, &[vec![0, 1]], &[]).expect("model should build");
        let component = &model.components[0];
        let all_row_ids = (0..component.all_marker_placement_rows.len()).collect::<Vec<_>>();

        let marker_on_side_zero_neighbor_three = model
            .filter_marker_placement_row_ids_for_marker_event_facts(
                0,
                &all_row_ids,
                &[StereoMarkerEventFact::MarkerPlaced {
                    side_idx: 0,
                    slot: 12,
                    begin_idx: 1,
                    end_idx: 3,
                    marker: StereoDirectionToken::Slash,
                    role: StereoTraversalRole::TreeOrChain,
                }],
            )
            .expect("marker event query should be valid");
        assert_eq!(
            vec![1, 2, 4, 5, 7, 8, 10, 11],
            marker_on_side_zero_neighbor_three
        );

        let only_neighbor_three = model
            .filter_marker_placement_row_ids_for_marker_event_facts(
                0,
                &marker_on_side_zero_neighbor_three,
                &[StereoMarkerEventFact::NoMarker {
                    side_idx: 0,
                    slot: 13,
                    begin_idx: 0,
                    end_idx: 1,
                    role: StereoTraversalRole::Branch,
                }],
            )
            .expect("marker event query should be valid");
        assert_eq!(vec![1, 4, 7, 10], only_neighbor_three);
        assert_eq!(
            vec![0, 1, 2, 3],
            model
                .token_phase_assignment_ids_for_marker_placement_row_ids(0, &only_neighbor_three)
                .expect("marker row token phase query should be valid"),
        );
        assert_eq!(
            vec![0, 1],
            model
                .neighbor_assignment_ids_for_marker_placement_row_ids(0, &only_neighbor_three)
                .expect("marker row neighbor assignment query should be valid"),
        );
        assert_eq!(
            vec![0, 3],
            model.available_neighbors_for_assignment_ids(0, 0, &[0, 1]),
        );
        assert_eq!(
            vec![vec![3]],
            model
                .marker_neighbor_sets_for_marker_placement_row_ids(0, 0, &only_neighbor_three)
                .expect("marker row marker-neighbor query should be valid"),
        );
        assert_eq!(
            vec![vec![4]],
            model
                .marker_neighbor_sets_for_marker_placement_row_ids(0, 1, &only_neighbor_three)
                .expect("marker row marker-neighbor query should be valid"),
        );

        let contradicted = model
            .filter_marker_placement_row_ids_for_marker_event_facts(
                0,
                &marker_on_side_zero_neighbor_three,
                &[StereoMarkerEventFact::NoMarker {
                    side_idx: 0,
                    slot: 14,
                    begin_idx: 1,
                    end_idx: 3,
                    role: StereoTraversalRole::Branch,
                }],
            )
            .expect("marker event query should be valid");
        assert!(contradicted.is_empty());

        let no_marker_on_single_candidate_side = model
            .filter_marker_placement_row_ids_for_marker_event_facts(
                0,
                &all_row_ids,
                &[StereoMarkerEventFact::NoMarker {
                    side_idx: 1,
                    slot: 15,
                    begin_idx: 2,
                    end_idx: 4,
                    role: StereoTraversalRole::RingClose,
                }],
            )
            .expect("marker event query should be valid");
        assert!(no_marker_on_single_candidate_side.is_empty());
    }

    #[test]
    fn marker_placement_event_filter_validates_routing_without_cloning_rows() {
        let side_infos = vec![StereoSideInfo {
            component_idx: 0,
            endpoint_atom_idx: 1,
            other_endpoint_atom_idx: 2,
            candidate_neighbors: vec![0, 3],
            candidate_base_tokens: vec!["/".to_owned(), "\\".to_owned()],
        }];
        let model =
            stereo_constraint_model(&side_infos, &[vec![0]], &[]).expect("model should build");
        let all_row_ids =
            (0..model.components[0].all_marker_placement_rows.len()).collect::<Vec<_>>();

        assert_eq!(
            all_row_ids,
            model
                .filter_marker_placement_row_ids_for_marker_event_facts(
                    0,
                    &all_row_ids,
                    &[StereoMarkerEventFact::NoMarker {
                        side_idx: 0,
                        slot: 12,
                        begin_idx: 7,
                        end_idx: 8,
                        role: StereoTraversalRole::Deferred,
                    }],
                )
                .expect("noncandidate no-marker event should be a no-op"),
        );
        assert!(model
            .filter_marker_placement_row_ids_for_marker_event_facts(
                0,
                &all_row_ids,
                &[StereoMarkerEventFact::MarkerPlaced {
                    side_idx: 0,
                    slot: 12,
                    begin_idx: 7,
                    end_idx: 8,
                    marker: StereoDirectionToken::Backslash,
                    role: StereoTraversalRole::Deferred,
                }],
            )
            .expect("noncandidate marker event should be contradictory")
            .is_empty());
        assert!(model
            .filter_marker_placement_row_ids_for_marker_event_facts(
                0,
                &[model.components[0].all_marker_placement_rows.len()],
                &[],
            )
            .is_err());
        assert!(model
            .filter_marker_placement_row_ids_for_marker_event_facts(
                0,
                &all_row_ids,
                &[StereoMarkerEventFact::NoMarker {
                    side_idx: 99,
                    slot: 12,
                    begin_idx: 1,
                    end_idx: 0,
                    role: StereoTraversalRole::TreeOrChain,
                }],
            )
            .is_err());
    }

    #[test]
    fn token_observation_fact_derives_token_flips() {
        for (component_phase, adjustment, expected) in [
            (StereoComponentPhase::Stored, false, StereoTokenFlip::Stored),
            (
                StereoComponentPhase::Flipped,
                false,
                StereoTokenFlip::Flipped,
            ),
            (StereoComponentPhase::Stored, true, StereoTokenFlip::Flipped),
            (StereoComponentPhase::Flipped, true, StereoTokenFlip::Stored),
        ] {
            let observation = StereoTokenObservationFact::AllSingleCandidate {
                runtime_component_idx: 0,
                component_phase,
                rdkit_token_flip_adjustment: rdkit_adjustment(adjustment),
            };
            assert_eq!(expected, observation.implied_token_flip());
            assert_eq!("all_single_candidate", observation.observation_kind());
            assert_eq!(None, observation.selected_begin_token());
        }

        let cases = [
            (
                StereoComponentPhase::Stored,
                StereoDirectionToken::Slash,
                false,
                StereoTokenFlip::Flipped,
            ),
            (
                StereoComponentPhase::Stored,
                StereoDirectionToken::Backslash,
                false,
                StereoTokenFlip::Stored,
            ),
            (
                StereoComponentPhase::Flipped,
                StereoDirectionToken::Slash,
                false,
                StereoTokenFlip::Flipped,
            ),
            (
                StereoComponentPhase::Flipped,
                StereoDirectionToken::Backslash,
                false,
                StereoTokenFlip::Stored,
            ),
            (
                StereoComponentPhase::Stored,
                StereoDirectionToken::Slash,
                true,
                StereoTokenFlip::Stored,
            ),
            (
                StereoComponentPhase::Stored,
                StereoDirectionToken::Backslash,
                true,
                StereoTokenFlip::Flipped,
            ),
            (
                StereoComponentPhase::Flipped,
                StereoDirectionToken::Slash,
                true,
                StereoTokenFlip::Stored,
            ),
            (
                StereoComponentPhase::Flipped,
                StereoDirectionToken::Backslash,
                true,
                StereoTokenFlip::Flipped,
            ),
        ];

        for (component_phase, selected_begin_token, adjustment, expected) in cases {
            let observation = StereoTokenObservationFact::SelectedBeginSide {
                runtime_component_idx: 0,
                component_phase,
                selected_begin_token,
                rdkit_token_flip_adjustment: rdkit_adjustment(adjustment),
            };
            assert_eq!(expected, observation.implied_token_flip(),);
        }

        let two_candidate_known_order_cases = [
            (
                StereoDirectionToken::Slash,
                Some(true),
                false,
                StereoTokenFlip::Flipped,
            ),
            (
                StereoDirectionToken::Slash,
                Some(false),
                false,
                StereoTokenFlip::Stored,
            ),
            (
                StereoDirectionToken::Backslash,
                Some(true),
                false,
                StereoTokenFlip::Stored,
            ),
            (
                StereoDirectionToken::Backslash,
                Some(false),
                false,
                StereoTokenFlip::Flipped,
            ),
            (
                StereoDirectionToken::Slash,
                Some(true),
                true,
                StereoTokenFlip::Stored,
            ),
            (
                StereoDirectionToken::Backslash,
                Some(false),
                true,
                StereoTokenFlip::Stored,
            ),
        ];

        for (selected_begin_token, selected_is_first, adjustment, expected) in
            two_candidate_known_order_cases
        {
            for component_phase in [StereoComponentPhase::Stored, StereoComponentPhase::Flipped] {
                let observation = StereoTokenObservationFact::TwoCandidateBeginSide {
                    runtime_component_idx: 0,
                    component_phase,
                    selected_begin_token,
                    selected_begin_neighbor_is_first_emitted: selected_is_first,
                    rdkit_token_flip_adjustment: rdkit_adjustment(adjustment),
                };
                assert_eq!(expected, observation.implied_token_flip());
                assert_eq!("two_candidate_begin_side", observation.observation_kind());
                assert_eq!(
                    selected_is_first,
                    observation.selected_begin_neighbor_is_first_emitted()
                );
            }
        }

        for (component_phase, adjustment, expected) in [
            (StereoComponentPhase::Stored, false, StereoTokenFlip::Stored),
            (
                StereoComponentPhase::Flipped,
                false,
                StereoTokenFlip::Flipped,
            ),
            (StereoComponentPhase::Stored, true, StereoTokenFlip::Flipped),
            (StereoComponentPhase::Flipped, true, StereoTokenFlip::Stored),
        ] {
            let observation = StereoTokenObservationFact::TwoCandidateBeginSide {
                runtime_component_idx: 0,
                component_phase,
                selected_begin_token: StereoDirectionToken::Slash,
                selected_begin_neighbor_is_first_emitted: None,
                rdkit_token_flip_adjustment: rdkit_adjustment(adjustment),
            };
            assert_eq!(expected, observation.implied_token_flip());
        }
    }

    #[test]
    fn stereo_direction_token_parser_rejects_invalid_tokens() {
        assert_eq!(
            StereoDirectionToken::Slash,
            StereoDirectionToken::from_str("/").expect("slash token should parse"),
        );
        assert_eq!(
            StereoDirectionToken::Backslash,
            StereoDirectionToken::from_str("\\").expect("backslash token should parse"),
        );
        assert!(StereoDirectionToken::from_str("").is_err());
        assert!(StereoDirectionToken::from_str("-").is_err());
        assert!(StereoDirectionToken::from_str("=").is_err());
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

        assert_eq!(
            vec![1],
            model.remaining_assignment_ids(
                0,
                StereoConstraintLayer::Semantic,
                &[StereoConstraintFact::CarrierSelected {
                    side_idx: 0,
                    neighbor_idx: 3,
                }],
            ),
        );
        assert_eq!(
            vec![1],
            model.remaining_assignment_ids(
                0,
                StereoConstraintLayer::Semantic,
                &[StereoConstraintFact::CarrierSelectionBlocked {
                    side_idx: 0,
                    neighbor_idx: 0,
                }],
            ),
        );
        assert_eq!(
            vec![0],
            model.remaining_assignment_ids(
                0,
                StereoConstraintLayer::Semantic,
                &[StereoConstraintFact::CarrierSelectionBlocked {
                    side_idx: 0,
                    neighbor_idx: 3,
                }],
            ),
        );
        assert!(model
            .remaining_assignment_ids(
                0,
                StereoConstraintLayer::Semantic,
                &[
                    StereoConstraintFact::CarrierSelected {
                        side_idx: 0,
                        neighbor_idx: 3,
                    },
                    StereoConstraintFact::CarrierSelectionBlocked {
                        side_idx: 0,
                        neighbor_idx: 3,
                    },
                ],
            )
            .is_empty());
        assert!(!model.has_completion(
            0,
            StereoConstraintLayer::Semantic,
            &[StereoConstraintFact::CarrierSelectionBlocked {
                side_idx: 0,
                neighbor_idx: 99,
            }],
        ));
        assert_eq!(
            Some(3),
            model.forced_neighbor_for_assignment_ids(0, 0, &[1]),
        );
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
    fn assignment_state_tracks_remaining_ids_and_forced_neighbors() {
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
                candidate_neighbors: vec![4, 5],
                candidate_base_tokens: vec!["/".to_owned(), "\\".to_owned()],
            },
        ];
        let model =
            stereo_constraint_model(&side_infos, &[vec![0, 1]], &[]).expect("model should build");
        let unconstrained =
            super::StereoAssignmentState::from_model(&model, StereoConstraintLayer::Semantic);
        assert_eq!(vec![vec![0, 1, 2, 3]], unconstrained.remaining_by_component);
        assert_eq!(None, unconstrained.forced_neighbor(&model, 0, 0));
        assert_eq!(
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            model
                .token_phase_assignment_ids_for_neighbor_assignment_ids(
                    0,
                    &unconstrained.remaining_by_component[0],
                    &[],
                )
                .expect("token phase query should be valid"),
        );

        let facts_by_component = vec![vec![StereoConstraintFact::CarrierSelected {
            side_idx: 0,
            neighbor_idx: 3,
        }]];
        let constrained = super::StereoAssignmentState::from_facts_by_component(
            &model,
            StereoConstraintLayer::Semantic,
            &facts_by_component,
        );
        assert_eq!(vec![vec![2, 3]], constrained.remaining_by_component);
        assert_eq!(false, constrained.is_empty(0));
        assert_eq!(Some(3), constrained.forced_neighbor(&model, 0, 0));
        assert_eq!(None, constrained.forced_neighbor(&model, 0, 1));
        assert_eq!(
            vec![4, 6],
            model
                .token_phase_assignment_ids_for_neighbor_assignment_ids(
                    0,
                    &constrained.remaining_by_component[0],
                    &[StereoTokenFlipFact {
                        runtime_component_idx: 0,
                        token_flip: StereoTokenFlip::Stored,
                    }],
                )
                .expect("token phase query should be valid"),
        );
        assert_eq!(
            Some(StereoTokenFlip::Stored),
            model.forced_token_flip_for_token_phase_assignment_ids(0, 0, &[4, 6]),
        );
        assert_eq!(
            vec![4, 6],
            model
                .filter_token_phase_assignment_ids_for_token_flip(
                    0,
                    0,
                    &[4, 5, 6, 7],
                    StereoTokenFlip::Stored,
                )
                .expect("token phase filter should be valid"),
        );
        assert_eq!(
            vec![5, 7],
            model
                .filter_token_phase_assignment_ids_for_token_flip(
                    0,
                    0,
                    &[4, 5, 6, 7],
                    StereoTokenFlip::Flipped,
                )
                .expect("token phase filter should be valid"),
        );
        assert_eq!(
            None,
            model.forced_token_flip_for_token_phase_assignment_ids(0, 0, &[4, 5]),
        );

        let narrowed_again = constrained.filter_facts_by_component(
            &model,
            StereoConstraintLayer::Semantic,
            &[vec![StereoConstraintFact::CarrierSelected {
                side_idx: 1,
                neighbor_idx: 5,
            }]],
        );
        assert_eq!(vec![vec![3]], narrowed_again.remaining_by_component);
        assert_eq!(Some(3), narrowed_again.forced_neighbor(&model, 0, 0));
        assert_eq!(Some(5), narrowed_again.forced_neighbor(&model, 0, 1));
        assert_eq!(
            vec![6, 7],
            model
                .token_phase_assignment_ids_for_neighbor_assignment_ids(
                    0,
                    &narrowed_again.remaining_by_component[0],
                    &[],
                )
                .expect("token phase query should be valid"),
        );
        assert_eq!(
            vec![7],
            model
                .token_phase_assignment_ids_for_neighbor_assignment_ids(
                    0,
                    &narrowed_again.remaining_by_component[0],
                    &[StereoTokenFlipFact {
                        runtime_component_idx: 0,
                        token_flip: StereoTokenFlip::Flipped,
                    }],
                )
                .expect("token phase query should be valid"),
        );

        let contradictory = constrained.filter_facts_by_component(
            &model,
            StereoConstraintLayer::Semantic,
            &[vec![StereoConstraintFact::CarrierSelected {
                side_idx: 0,
                neighbor_idx: 0,
            }]],
        );
        assert!(contradictory.is_empty(0));
    }

    #[test]
    fn constraint_state_filters_carrier_and_token_phase_facts() {
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

        let state = super::StereoConstraintState::from_facts(
            &model,
            StereoConstraintLayer::RdkitLocalWriter,
            &[vec![StereoConstraintFact::CarrierSelected {
                side_idx: 0,
                neighbor_idx: 8,
            }]],
            &[StereoTokenFlipFact {
                runtime_component_idx: 0,
                token_flip: StereoTokenFlip::Stored,
            }],
        )
        .expect("constraint state should build");

        assert_eq!(
            vec![vec![2, 3]],
            state.carrier_assignment_state.remaining_by_component,
        );
        assert_eq!(
            vec![vec![8, 9, 12, 13]],
            state.token_phase_remaining_by_component
        );

        let observation_state = super::StereoConstraintState::from_token_observation_facts(
            &model,
            StereoConstraintLayer::RdkitLocalWriter,
            &[vec![StereoConstraintFact::CarrierSelected {
                side_idx: 0,
                neighbor_idx: 8,
            }]],
            &[StereoTokenObservationFact::SelectedBeginSide {
                runtime_component_idx: 0,
                component_phase: StereoComponentPhase::Stored,
                selected_begin_token: StereoDirectionToken::Backslash,
                rdkit_token_flip_adjustment: RdkitTokenFlipAdjustmentObservations::NONE,
            }],
        )
        .expect("observation constraint state should build");
        assert_eq!(state, observation_state);

        let mixed_state = super::StereoConstraintState::from_facts_and_token_observations(
            &model,
            StereoConstraintLayer::RdkitLocalWriter,
            &[vec![StereoConstraintFact::CarrierSelected {
                side_idx: 0,
                neighbor_idx: 8,
            }]],
            &[],
            &[StereoTokenObservationFact::SelectedBeginSide {
                runtime_component_idx: 0,
                component_phase: StereoComponentPhase::Stored,
                selected_begin_token: StereoDirectionToken::Backslash,
                rdkit_token_flip_adjustment: RdkitTokenFlipAdjustmentObservations::NONE,
            }],
        )
        .expect("mixed observation constraint state should build");
        assert_eq!(state, mixed_state);

        assert_eq!(Some(8), state.forced_neighbor(&model, 0, 0));
        assert_eq!(
            Some(StereoTokenFlip::Stored),
            state.forced_token_flip(&model, 0, 0),
        );
        assert_eq!(
            vec![StereoTokenFlip::Stored],
            state.available_token_flips(&model, 0, 0),
        );
        assert_eq!(
            vec![StereoTokenFlip::Stored, StereoTokenFlip::Flipped],
            state.available_token_flips(&model, 0, 1),
        );
        assert_eq!(None, state.forced_token_flip(&model, 0, 1));
        assert!(!state.is_empty(0));

        let contradictory = super::StereoConstraintState::from_facts(
            &model,
            StereoConstraintLayer::RdkitLocalWriter,
            &[vec![StereoConstraintFact::CarrierSelected {
                side_idx: 0,
                neighbor_idx: 8,
            }]],
            &[
                StereoTokenFlipFact {
                    runtime_component_idx: 0,
                    token_flip: StereoTokenFlip::Stored,
                },
                StereoTokenFlipFact {
                    runtime_component_idx: 0,
                    token_flip: StereoTokenFlip::Flipped,
                },
            ],
        )
        .expect("contradictory token facts should create an empty state");
        assert!(contradictory.is_empty(0));
        assert_eq!(
            Vec::<StereoTokenFlip>::new(),
            contradictory.available_token_flips(&model, 0, 0),
        );
        assert_eq!(
            vec![Vec::<usize>::new()],
            contradictory.token_phase_remaining_by_component
        );

        let mixed_contradictory = super::StereoConstraintState::from_facts_and_token_observations(
            &model,
            StereoConstraintLayer::RdkitLocalWriter,
            &[vec![StereoConstraintFact::CarrierSelected {
                side_idx: 0,
                neighbor_idx: 8,
            }]],
            &[StereoTokenFlipFact {
                runtime_component_idx: 0,
                token_flip: StereoTokenFlip::Flipped,
            }],
            &[StereoTokenObservationFact::SelectedBeginSide {
                runtime_component_idx: 0,
                component_phase: StereoComponentPhase::Stored,
                selected_begin_token: StereoDirectionToken::Backslash,
                rdkit_token_flip_adjustment: RdkitTokenFlipAdjustmentObservations::NONE,
            }],
        )
        .expect("contradictory mixed facts should create an empty state");
        assert!(mixed_contradictory.is_empty(0));
    }

    #[test]
    fn token_phase_queries_reject_invalid_runtime_component_facts() {
        let side_infos = vec![
            StereoSideInfo {
                component_idx: 0,
                endpoint_atom_idx: 1,
                other_endpoint_atom_idx: 2,
                candidate_neighbors: vec![0],
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
        let model = stereo_constraint_model(&side_infos, &[vec![0], vec![1]], &[])
            .expect("model should build");

        assert_eq!(Some(0), model.component_for_runtime_component(0));
        assert_eq!(Some(1), model.component_for_runtime_component(1));
        assert_eq!(None, model.component_for_runtime_component(2));
        assert!(model
            .token_phase_assignment_ids_for_neighbor_assignment_ids(
                0,
                &[0],
                &[StereoTokenFlipFact {
                    runtime_component_idx: 1,
                    token_flip: StereoTokenFlip::Stored,
                }],
            )
            .is_err());
        assert!(model
            .token_phase_assignment_ids_for_token_observation_facts(
                0,
                &[0],
                &[StereoTokenObservationFact::SelectedBeginSide {
                    runtime_component_idx: 1,
                    component_phase: StereoComponentPhase::Stored,
                    selected_begin_token: StereoDirectionToken::Backslash,
                    rdkit_token_flip_adjustment: RdkitTokenFlipAdjustmentObservations::NONE,
                }],
            )
            .is_err());
        assert!(model
            .token_phase_assignment_ids_for_token_observation_facts(
                0,
                &[0],
                &[StereoTokenObservationFact::SelectedBeginSide {
                    runtime_component_idx: 2,
                    component_phase: StereoComponentPhase::Stored,
                    selected_begin_token: StereoDirectionToken::Backslash,
                    rdkit_token_flip_adjustment: RdkitTokenFlipAdjustmentObservations::NONE,
                }],
            )
            .is_err());
        assert!(super::StereoConstraintState::from_token_observation_facts(
            &model,
            StereoConstraintLayer::Semantic,
            &[],
            &[StereoTokenObservationFact::SelectedBeginSide {
                runtime_component_idx: 2,
                component_phase: StereoComponentPhase::Stored,
                selected_begin_token: StereoDirectionToken::Backslash,
                rdkit_token_flip_adjustment: RdkitTokenFlipAdjustmentObservations::NONE,
            }],
        )
        .is_err());
        assert!(super::StereoConstraintState::from_facts(
            &model,
            StereoConstraintLayer::Semantic,
            &[],
            &[StereoTokenFlipFact {
                runtime_component_idx: 2,
                token_flip: StereoTokenFlip::Stored,
            }],
        )
        .is_err());
        assert!(
            super::StereoConstraintState::from_facts_and_token_observations(
                &model,
                StereoConstraintLayer::Semantic,
                &[],
                &[],
                &[StereoTokenObservationFact::SelectedBeginSide {
                    runtime_component_idx: 2,
                    component_phase: StereoComponentPhase::Stored,
                    selected_begin_token: StereoDirectionToken::Backslash,
                    rdkit_token_flip_adjustment: RdkitTokenFlipAdjustmentObservations::NONE,
                }],
            )
            .is_err()
        );
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
        assert_eq!(vec![0, 1], model.components[0].runtime_component_ids);
        assert_eq!(vec![0, 1], model.components[0].side_ids);
        assert_eq!(
            16,
            model
                .token_phase_assignment_ids_for_neighbor_assignment_ids(0, &[0, 1, 2, 3], &[],)
                .expect("token phase query should be valid")
                .len(),
        );
        let runtime_zero_stored = model
            .token_phase_assignment_ids_for_neighbor_assignment_ids(
                0,
                &[0, 1, 2, 3],
                &[StereoTokenFlipFact {
                    runtime_component_idx: 0,
                    token_flip: StereoTokenFlip::Stored,
                }],
            )
            .expect("token phase query should be valid");
        assert_eq!(8, runtime_zero_stored.len());
        assert_eq!(
            Some(StereoTokenFlip::Stored),
            model.forced_token_flip_for_token_phase_assignment_ids(0, 0, &runtime_zero_stored),
        );
        assert_eq!(
            None,
            model.forced_token_flip_for_token_phase_assignment_ids(0, 1, &runtime_zero_stored),
        );
        assert_eq!(
            runtime_zero_stored,
            model
                .token_phase_assignment_ids_for_token_observation_facts(
                    0,
                    &[0, 1, 2, 3],
                    &[StereoTokenObservationFact::SelectedBeginSide {
                        runtime_component_idx: 0,
                        component_phase: StereoComponentPhase::Stored,
                        selected_begin_token: StereoDirectionToken::Backslash,
                        rdkit_token_flip_adjustment: RdkitTokenFlipAdjustmentObservations::NONE,
                    }],
                )
                .expect("observation query should be valid"),
        );
        assert_eq!(
            4,
            model
                .token_phase_assignment_ids_for_neighbor_assignment_ids(
                    0,
                    &[0, 1, 2, 3],
                    &[
                        StereoTokenFlipFact {
                            runtime_component_idx: 0,
                            token_flip: StereoTokenFlip::Stored,
                        },
                        StereoTokenFlipFact {
                            runtime_component_idx: 1,
                            token_flip: StereoTokenFlip::Flipped,
                        },
                    ],
                )
                .expect("token phase query should be valid")
                .len(),
        );
        let both_observations = model
            .token_phase_assignment_ids_for_token_observation_facts(
                0,
                &[0, 1, 2, 3],
                &[
                    StereoTokenObservationFact::SelectedBeginSide {
                        runtime_component_idx: 0,
                        component_phase: StereoComponentPhase::Stored,
                        selected_begin_token: StereoDirectionToken::Backslash,
                        rdkit_token_flip_adjustment: RdkitTokenFlipAdjustmentObservations::NONE,
                    },
                    StereoTokenObservationFact::SelectedBeginSide {
                        runtime_component_idx: 1,
                        component_phase: StereoComponentPhase::Stored,
                        selected_begin_token: StereoDirectionToken::Slash,
                        rdkit_token_flip_adjustment: RdkitTokenFlipAdjustmentObservations::NONE,
                    },
                ],
            )
            .expect("observation query should be valid");
        assert_eq!(4, both_observations.len());
        assert_eq!(
            Some(StereoTokenFlip::Stored),
            model.forced_token_flip_for_token_phase_assignment_ids(0, 0, &both_observations),
        );
        assert_eq!(
            Some(StereoTokenFlip::Flipped),
            model.forced_token_flip_for_token_phase_assignment_ids(0, 1, &both_observations),
        );
        assert_eq!(
            Vec::<usize>::new(),
            model
                .token_phase_assignment_ids_for_token_observation_facts(
                    0,
                    &[0, 1, 2, 3],
                    &[
                        StereoTokenObservationFact::SelectedBeginSide {
                            runtime_component_idx: 0,
                            component_phase: StereoComponentPhase::Stored,
                            selected_begin_token: StereoDirectionToken::Backslash,
                            rdkit_token_flip_adjustment: RdkitTokenFlipAdjustmentObservations::NONE,
                        },
                        StereoTokenObservationFact::SelectedBeginSide {
                            runtime_component_idx: 0,
                            component_phase: StereoComponentPhase::Stored,
                            selected_begin_token: StereoDirectionToken::Slash,
                            rdkit_token_flip_adjustment: RdkitTokenFlipAdjustmentObservations::NONE,
                        },
                    ],
                )
                .expect("conflicting observations should empty the query"),
        );
        assert_eq!(
            vec![0, 1, 2, 3],
            model.remaining_assignment_ids(0, StereoConstraintLayer::Semantic, &[]),
        );
        assert_eq!(
            vec![1, 2, 3],
            model.remaining_assignment_ids(0, StereoConstraintLayer::RdkitLocalWriter, &[]),
        );
        assert_eq!(
            None,
            model.forced_neighbor_for_assignment_ids(0, 0, &[1, 2, 3]),
        );
        assert_eq!(
            vec![2, 3],
            model.remaining_assignment_ids(
                0,
                StereoConstraintLayer::RdkitLocalWriter,
                &[StereoConstraintFact::CarrierSelected {
                    side_idx: 0,
                    neighbor_idx: 8,
                }],
            ),
        );
        assert_eq!(
            None,
            model.forced_neighbor_for_assignment_ids(0, 1, &[2, 3]),
        );
        assert_eq!(
            vec![2],
            model.remaining_assignment_ids(
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
            ),
        );
        assert_eq!(
            Some(4),
            model.forced_neighbor_for_assignment_ids(0, 1, &[2]),
        );
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
