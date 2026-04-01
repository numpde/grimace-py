use pyo3::exceptions::{PyIndexError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

pub const PREPARED_SMILES_GRAPH_SCHEMA_VERSION: usize = 1;
pub const CONNECTED_NONSTEREO_SURFACE: &str = "connected_nonstereo";
pub const CONNECTED_STEREO_SURFACE: &str = "connected_stereo";

pub(crate) fn mol_to_smiles_support_data(
    graph: &PreparedSmilesGraphData,
    root_idx: isize,
    isomeric_smiles: bool,
) -> PyResult<Vec<String>> {
    if isomeric_smiles {
        if graph.surface_kind != CONNECTED_STEREO_SURFACE {
            return Err(PyValueError::new_err(format!(
                "PreparedSmilesGraph surface_kind={} is not compatible with isomeric_smiles=True",
                graph.surface_kind
            )));
        }
        return crate::rooted_stereo::enumerate_rooted_connected_stereo_smiles_support(
            graph, root_idx,
        );
    }
    if graph.surface_kind != CONNECTED_NONSTEREO_SURFACE {
        return Err(PyValueError::new_err(format!(
            "PreparedSmilesGraph surface_kind={} is not compatible with isomeric_smiles=False",
            graph.surface_kind
        )));
    }
    crate::rooted_nonstereo::enumerate_rooted_connected_nonstereo_smiles_support(graph, root_idx)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PreparedSmilesGraphData {
    pub(crate) schema_version: usize,
    pub(crate) surface_kind: String,
    pub(crate) policy_name: String,
    pub(crate) policy_digest: String,
    pub(crate) rdkit_version: String,
    pub(crate) identity_smiles: String,
    pub(crate) atom_count: usize,
    pub(crate) bond_count: usize,
    pub(crate) atom_atomic_numbers: Vec<usize>,
    pub(crate) atom_is_aromatic: Vec<bool>,
    pub(crate) atom_isotopes: Vec<usize>,
    pub(crate) atom_formal_charges: Vec<i32>,
    pub(crate) atom_total_hs: Vec<usize>,
    pub(crate) atom_radical_electrons: Vec<usize>,
    pub(crate) atom_map_numbers: Vec<usize>,
    pub(crate) atom_tokens: Vec<String>,
    pub(crate) neighbors: Vec<Vec<usize>>,
    pub(crate) neighbor_bond_tokens: Vec<Vec<String>>,
    pub(crate) bond_pairs: Vec<(usize, usize)>,
    pub(crate) bond_kinds: Vec<String>,
    pub(crate) writer_do_isomeric_smiles: bool,
    pub(crate) writer_kekule_smiles: bool,
    pub(crate) writer_all_bonds_explicit: bool,
    pub(crate) writer_all_hs_explicit: bool,
    pub(crate) writer_ignore_atom_map_numbers: bool,
    pub(crate) identity_parse_with_rdkit: bool,
    pub(crate) identity_canonical: bool,
    pub(crate) identity_do_isomeric_smiles: bool,
    pub(crate) identity_kekule_smiles: bool,
    pub(crate) identity_rooted_at_atom: i32,
    pub(crate) identity_all_bonds_explicit: bool,
    pub(crate) identity_all_hs_explicit: bool,
    pub(crate) identity_do_random: bool,
    pub(crate) identity_ignore_atom_map_numbers: bool,
    pub(crate) atom_chiral_tags: Vec<String>,
    pub(crate) atom_stereo_neighbor_orders: Vec<Vec<usize>>,
    pub(crate) atom_explicit_h_counts: Vec<usize>,
    pub(crate) atom_implicit_h_counts: Vec<usize>,
    pub(crate) bond_stereo_kinds: Vec<String>,
    pub(crate) bond_stereo_atoms: Vec<(isize, isize)>,
    pub(crate) bond_dirs: Vec<String>,
    pub(crate) bond_begin_atom_indices: Vec<usize>,
    pub(crate) bond_end_atom_indices: Vec<usize>,
}

fn required_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(format!("missing PreparedSmilesGraph field: {key}")))
}

fn extract_vec_vec_usize(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<Vec<usize>>> {
    required_item(dict, key)?.extract()
}

fn extract_vec_vec_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<Vec<String>>> {
    required_item(dict, key)?.extract()
}

fn extract_bond_pairs(dict: &Bound<'_, PyDict>) -> PyResult<Vec<(usize, usize)>> {
    let raw_pairs: Vec<Vec<usize>> = required_item(dict, "bond_pairs")?.extract()?;
    let mut pairs = Vec::with_capacity(raw_pairs.len());
    for pair in raw_pairs {
        if pair.len() != 2 {
            return Err(PyValueError::new_err(
                "PreparedSmilesGraph bond_pairs items must have length 2",
            ));
        }
        pairs.push((pair[0], pair[1]));
    }
    Ok(pairs)
}

fn optional_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    dict.get_item(key)
}

fn extract_optional_vec_usize(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<usize>> {
    match optional_item(dict, key)? {
        Some(value) => value.extract(),
        None => Ok(Vec::new()),
    }
}

fn extract_optional_vec_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<String>> {
    match optional_item(dict, key)? {
        Some(value) => value.extract(),
        None => Ok(Vec::new()),
    }
}

fn extract_optional_vec_vec_usize(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Vec<Vec<usize>>> {
    match optional_item(dict, key)? {
        Some(value) => value.extract(),
        None => Ok(Vec::new()),
    }
}

fn extract_optional_int_pairs(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Vec<(isize, isize)>> {
    let Some(value) = optional_item(dict, key)? else {
        return Ok(Vec::new());
    };
    let raw_pairs: Vec<Vec<isize>> = value.extract()?;
    let mut pairs = Vec::with_capacity(raw_pairs.len());
    for pair in raw_pairs {
        if pair.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "PreparedSmilesGraph {key} items must have length 2"
            )));
        }
        pairs.push((pair[0], pair[1]));
    }
    Ok(pairs)
}

impl PreparedSmilesGraphData {
    pub(crate) fn from_any(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(dict) = obj.cast::<PyDict>() {
            return Self::from_dict(&dict);
        }

        let dict_any = obj.call_method0("to_dict")?;
        let dict = dict_any.cast::<PyDict>()?;
        Self::from_dict(&dict)
    }

    fn from_dict(dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let data = Self {
            schema_version: required_item(dict, "schema_version")?.extract()?,
            surface_kind: required_item(dict, "surface_kind")?.extract()?,
            policy_name: required_item(dict, "policy_name")?.extract()?,
            policy_digest: required_item(dict, "policy_digest")?.extract()?,
            rdkit_version: required_item(dict, "rdkit_version")?.extract()?,
            identity_smiles: required_item(dict, "identity_smiles")?.extract()?,
            atom_count: required_item(dict, "atom_count")?.extract()?,
            bond_count: required_item(dict, "bond_count")?.extract()?,
            atom_atomic_numbers: required_item(dict, "atom_atomic_numbers")?.extract()?,
            atom_is_aromatic: required_item(dict, "atom_is_aromatic")?.extract()?,
            atom_isotopes: required_item(dict, "atom_isotopes")?.extract()?,
            atom_formal_charges: required_item(dict, "atom_formal_charges")?.extract()?,
            atom_total_hs: required_item(dict, "atom_total_hs")?.extract()?,
            atom_radical_electrons: required_item(dict, "atom_radical_electrons")?.extract()?,
            atom_map_numbers: required_item(dict, "atom_map_numbers")?.extract()?,
            atom_tokens: required_item(dict, "atom_tokens")?.extract()?,
            neighbors: extract_vec_vec_usize(dict, "neighbors")?,
            neighbor_bond_tokens: extract_vec_vec_string(dict, "neighbor_bond_tokens")?,
            bond_pairs: extract_bond_pairs(dict)?,
            bond_kinds: required_item(dict, "bond_kinds")?.extract()?,
            writer_do_isomeric_smiles: required_item(dict, "writer_do_isomeric_smiles")?
                .extract()?,
            writer_kekule_smiles: required_item(dict, "writer_kekule_smiles")?.extract()?,
            writer_all_bonds_explicit: required_item(dict, "writer_all_bonds_explicit")?
                .extract()?,
            writer_all_hs_explicit: required_item(dict, "writer_all_hs_explicit")?.extract()?,
            writer_ignore_atom_map_numbers: required_item(dict, "writer_ignore_atom_map_numbers")?
                .extract()?,
            identity_parse_with_rdkit: required_item(dict, "identity_parse_with_rdkit")?
                .extract()?,
            identity_canonical: required_item(dict, "identity_canonical")?.extract()?,
            identity_do_isomeric_smiles: required_item(dict, "identity_do_isomeric_smiles")?
                .extract()?,
            identity_kekule_smiles: required_item(dict, "identity_kekule_smiles")?.extract()?,
            identity_rooted_at_atom: required_item(dict, "identity_rooted_at_atom")?.extract()?,
            identity_all_bonds_explicit: required_item(dict, "identity_all_bonds_explicit")?
                .extract()?,
            identity_all_hs_explicit: required_item(dict, "identity_all_hs_explicit")?.extract()?,
            identity_do_random: required_item(dict, "identity_do_random")?.extract()?,
            identity_ignore_atom_map_numbers: required_item(
                dict,
                "identity_ignore_atom_map_numbers",
            )?
            .extract()?,
            atom_chiral_tags: extract_optional_vec_string(dict, "atom_chiral_tags")?,
            atom_stereo_neighbor_orders: extract_optional_vec_vec_usize(
                dict,
                "atom_stereo_neighbor_orders",
            )?,
            atom_explicit_h_counts: extract_optional_vec_usize(dict, "atom_explicit_h_counts")?,
            atom_implicit_h_counts: extract_optional_vec_usize(dict, "atom_implicit_h_counts")?,
            bond_stereo_kinds: extract_optional_vec_string(dict, "bond_stereo_kinds")?,
            bond_stereo_atoms: extract_optional_int_pairs(dict, "bond_stereo_atoms")?,
            bond_dirs: extract_optional_vec_string(dict, "bond_dirs")?,
            bond_begin_atom_indices: extract_optional_vec_usize(dict, "bond_begin_atom_indices")?,
            bond_end_atom_indices: extract_optional_vec_usize(dict, "bond_end_atom_indices")?,
        };
        data.validate()?;
        Ok(data)
    }

    fn validate(&self) -> PyResult<()> {
        if self.schema_version != PREPARED_SMILES_GRAPH_SCHEMA_VERSION {
            return Err(PyValueError::new_err(format!(
                "Unexpected PreparedSmilesGraph schema version: {}",
                self.schema_version
            )));
        }
        if self.surface_kind != CONNECTED_NONSTEREO_SURFACE
            && self.surface_kind != CONNECTED_STEREO_SURFACE
        {
            return Err(PyValueError::new_err(format!(
                "Unsupported PreparedSmilesGraph surface: {}",
                self.surface_kind
            )));
        }

        let atom_field_lengths = [
            self.atom_atomic_numbers.len(),
            self.atom_is_aromatic.len(),
            self.atom_isotopes.len(),
            self.atom_formal_charges.len(),
            self.atom_total_hs.len(),
            self.atom_radical_electrons.len(),
            self.atom_map_numbers.len(),
            self.atom_tokens.len(),
            self.neighbors.len(),
            self.neighbor_bond_tokens.len(),
        ];
        if atom_field_lengths
            .into_iter()
            .any(|field_len| field_len != self.atom_count)
        {
            return Err(PyValueError::new_err(
                "PreparedSmilesGraph atom field length mismatch",
            ));
        }

        if self.bond_pairs.len() != self.bond_count || self.bond_kinds.len() != self.bond_count {
            return Err(PyValueError::new_err(
                "PreparedSmilesGraph bond field length mismatch",
            ));
        }

        let stereo_atom_fields = [
            self.atom_chiral_tags.len(),
            self.atom_stereo_neighbor_orders.len(),
            self.atom_explicit_h_counts.len(),
            self.atom_implicit_h_counts.len(),
        ];
        for field_len in stereo_atom_fields {
            if field_len != 0 && field_len != self.atom_count {
                return Err(PyValueError::new_err(
                    "PreparedSmilesGraph stereo atom field length mismatch",
                ));
            }
        }

        let stereo_bond_fields = [
            self.bond_stereo_kinds.len(),
            self.bond_stereo_atoms.len(),
            self.bond_dirs.len(),
            self.bond_begin_atom_indices.len(),
            self.bond_end_atom_indices.len(),
        ];
        for field_len in stereo_bond_fields {
            if field_len != 0 && field_len != self.bond_count {
                return Err(PyValueError::new_err(
                    "PreparedSmilesGraph stereo bond field length mismatch",
                ));
            }
        }

        let mut bond_pairs_seen = std::collections::BTreeSet::new();
        let mut neighbor_pairs_seen = std::collections::BTreeSet::new();

        for (atom_idx, (neighbors, bond_tokens)) in self
            .neighbors
            .iter()
            .zip(self.neighbor_bond_tokens.iter())
            .enumerate()
        {
            if neighbors.len() != bond_tokens.len() {
                return Err(PyValueError::new_err(
                    "PreparedSmilesGraph neighbor token row length mismatch",
                ));
            }

            let mut sorted_neighbors = neighbors.clone();
            sorted_neighbors.sort_unstable();
            if *neighbors != sorted_neighbors {
                return Err(PyValueError::new_err(
                    "PreparedSmilesGraph neighbor rows must be sorted",
                ));
            }

            let unique_neighbors = neighbors
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>();
            if unique_neighbors.len() != neighbors.len() {
                return Err(PyValueError::new_err(
                    "PreparedSmilesGraph neighbor rows must be unique",
                ));
            }

            for &neighbor_idx in neighbors {
                if neighbor_idx >= self.atom_count {
                    return Err(PyValueError::new_err(
                        "PreparedSmilesGraph neighbor index out of range",
                    ));
                }
                if neighbor_idx == atom_idx {
                    return Err(PyValueError::new_err(
                        "PreparedSmilesGraph cannot contain self-loops",
                    ));
                }
                neighbor_pairs_seen
                    .insert((atom_idx.min(neighbor_idx), atom_idx.max(neighbor_idx)));
            }
        }

        for &(begin_idx, end_idx) in &self.bond_pairs {
            if begin_idx >= self.atom_count || end_idx >= self.atom_count {
                return Err(PyValueError::new_err(
                    "PreparedSmilesGraph bond index out of range",
                ));
            }
            if begin_idx >= end_idx {
                return Err(PyValueError::new_err(
                    "PreparedSmilesGraph bond_pairs must be canonicalized",
                ));
            }
            if !bond_pairs_seen.insert((begin_idx, end_idx)) {
                return Err(PyValueError::new_err(
                    "PreparedSmilesGraph bond_pairs must be unique",
                ));
            }

            let begin_neighbors = &self.neighbors[begin_idx];
            let end_neighbors = &self.neighbors[end_idx];
            if !begin_neighbors.contains(&end_idx) || !end_neighbors.contains(&begin_idx) {
                return Err(PyValueError::new_err(
                    "PreparedSmilesGraph bond_pairs must agree with neighbors",
                ));
            }
        }

        if bond_pairs_seen != neighbor_pairs_seen {
            return Err(PyValueError::new_err(
                "PreparedSmilesGraph neighbor graph and bond_pairs disagree",
            ));
        }

        for (begin_idx, neighbors) in self.neighbors.iter().enumerate() {
            for (offset, &end_idx) in neighbors.iter().enumerate() {
                let reverse_offset = self.neighbors[end_idx]
                    .iter()
                    .position(|&candidate| candidate == begin_idx)
                    .ok_or_else(|| {
                        PyValueError::new_err("PreparedSmilesGraph neighbors must be symmetric")
                    })?;
                let left_token = &self.neighbor_bond_tokens[begin_idx][offset];
                let right_token = &self.neighbor_bond_tokens[end_idx][reverse_offset];
                if left_token == right_token {
                    continue;
                }
                let bond_idx = self.bond_index(begin_idx, end_idx).ok_or_else(|| {
                    PyValueError::new_err(
                        "PreparedSmilesGraph bond_pairs must agree with neighbors",
                    )
                })?;
                let is_dative_pair = self.bond_kinds[bond_idx] == "DATIVE"
                    && ((left_token == "->" && right_token == "<-")
                        || (left_token == "<-" && right_token == "->"));
                if !is_dative_pair {
                    return Err(PyValueError::new_err(
                        "PreparedSmilesGraph bond tokens must be symmetric",
                    ));
                }
            }
        }

        if !self.atom_stereo_neighbor_orders.is_empty() {
            for (atom_idx, stereo_neighbor_order) in
                self.atom_stereo_neighbor_orders.iter().enumerate()
            {
                if stereo_neighbor_order.len() != self.neighbors[atom_idx].len() {
                    return Err(PyValueError::new_err(
                        "PreparedSmilesGraph stereo neighbor order length mismatch",
                    ));
                }
                let order_set = stereo_neighbor_order
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>();
                let neighbor_set = self.neighbors[atom_idx]
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>();
                if order_set != neighbor_set {
                    return Err(PyValueError::new_err(
                        "PreparedSmilesGraph stereo neighbor order must match neighbors",
                    ));
                }
            }
        }

        if !self.bond_stereo_atoms.is_empty() {
            for &(begin_idx, end_idx) in &self.bond_stereo_atoms {
                if (begin_idx, end_idx) == (-1, -1) {
                    continue;
                }
                if begin_idx < 0
                    || end_idx < 0
                    || begin_idx as usize >= self.atom_count
                    || end_idx as usize >= self.atom_count
                {
                    return Err(PyValueError::new_err(
                        "PreparedSmilesGraph stereo atom index out of range",
                    ));
                }
            }
        }

        if self.surface_kind == CONNECTED_STEREO_SURFACE {
            if self.atom_chiral_tags.len() != self.atom_count
                || self.atom_stereo_neighbor_orders.len() != self.atom_count
                || self.atom_explicit_h_counts.len() != self.atom_count
                || self.atom_implicit_h_counts.len() != self.atom_count
            {
                return Err(PyValueError::new_err(
                    "PreparedSmilesGraph connected_stereo surface requires stereo atom metadata",
                ));
            }
            if self.bond_stereo_kinds.len() != self.bond_count
                || self.bond_stereo_atoms.len() != self.bond_count
                || self.bond_dirs.len() != self.bond_count
                || self.bond_begin_atom_indices.len() != self.bond_count
                || self.bond_end_atom_indices.len() != self.bond_count
            {
                return Err(PyValueError::new_err(
                    "PreparedSmilesGraph connected_stereo surface requires stereo bond metadata",
                ));
            }
        }

        Ok(())
    }

    pub(crate) fn atom_count(&self) -> usize {
        self.atom_count
    }

    pub(crate) fn atom_token(&self, atom_idx: usize) -> &str {
        &self.atom_tokens[atom_idx]
    }

    pub(crate) fn neighbors_of(&self, atom_idx: usize) -> &[usize] {
        &self.neighbors[atom_idx]
    }

    pub(crate) fn bond_token(&self, begin_idx: usize, end_idx: usize) -> Option<&str> {
        self.neighbors[begin_idx]
            .iter()
            .position(|&candidate| candidate == end_idx)
            .map(|offset| self.neighbor_bond_tokens[begin_idx][offset].as_str())
    }

    pub(crate) fn bond_index(&self, begin_idx: usize, end_idx: usize) -> Option<usize> {
        let low_idx = begin_idx.min(end_idx);
        let high_idx = begin_idx.max(end_idx);
        self.bond_pairs
            .iter()
            .position(|&(left_idx, right_idx)| left_idx == low_idx && right_idx == high_idx)
    }

    pub(crate) fn directed_bond_token(&self, begin_idx: usize, end_idx: usize) -> PyResult<String> {
        let bond_idx = self.bond_index(begin_idx, end_idx).ok_or_else(|| {
            PyKeyError::new_err(format!("No bond between atoms {begin_idx} and {end_idx}"))
        })?;
        let base_token = self.bond_token(begin_idx, end_idx).ok_or_else(|| {
            PyKeyError::new_err(format!("No bond between atoms {begin_idx} and {end_idx}"))
        })?;
        if self.bond_dirs.is_empty() {
            return Ok(base_token.to_owned());
        }

        let bond_dir = self.bond_dirs[bond_idx].as_str();
        if bond_dir == "NONE" {
            return Ok(base_token.to_owned());
        }
        if bond_dir != "ENDUPRIGHT" && bond_dir != "ENDDOWNRIGHT" {
            return Err(PyValueError::new_err(format!(
                "Unsupported directional bond token: {bond_dir}"
            )));
        }
        if self.bond_begin_atom_indices.is_empty() || self.bond_end_atom_indices.is_empty() {
            return Err(PyValueError::new_err(
                "PreparedSmilesGraph is missing bond begin/end orientation metadata",
            ));
        }

        let stored_begin_idx = self.bond_begin_atom_indices[bond_idx];
        let stored_end_idx = self.bond_end_atom_indices[bond_idx];
        let reverse = if (begin_idx, end_idx) == (stored_begin_idx, stored_end_idx) {
            false
        } else if (begin_idx, end_idx) == (stored_end_idx, stored_begin_idx) {
            true
        } else {
            return Err(PyKeyError::new_err(format!(
                "No directed bond between atoms {begin_idx} and {end_idx}"
            )));
        };

        let token = if bond_dir == "ENDUPRIGHT" { "/" } else { "\\" };
        if reverse {
            Ok(if token == "/" { "\\" } else { "/" }.to_owned())
        } else {
            Ok(token.to_owned())
        }
    }

    fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("schema_version", self.schema_version)?;
        dict.set_item("surface_kind", &self.surface_kind)?;
        dict.set_item("policy_name", &self.policy_name)?;
        dict.set_item("policy_digest", &self.policy_digest)?;
        dict.set_item("rdkit_version", &self.rdkit_version)?;
        dict.set_item("identity_smiles", &self.identity_smiles)?;
        dict.set_item("atom_count", self.atom_count)?;
        dict.set_item("bond_count", self.bond_count)?;
        dict.set_item("atom_atomic_numbers", self.atom_atomic_numbers.clone())?;
        dict.set_item("atom_is_aromatic", self.atom_is_aromatic.clone())?;
        dict.set_item("atom_isotopes", self.atom_isotopes.clone())?;
        dict.set_item("atom_formal_charges", self.atom_formal_charges.clone())?;
        dict.set_item("atom_total_hs", self.atom_total_hs.clone())?;
        dict.set_item(
            "atom_radical_electrons",
            self.atom_radical_electrons.clone(),
        )?;
        dict.set_item("atom_map_numbers", self.atom_map_numbers.clone())?;
        dict.set_item("atom_tokens", self.atom_tokens.clone())?;
        dict.set_item("neighbors", self.neighbors.clone())?;
        dict.set_item("neighbor_bond_tokens", self.neighbor_bond_tokens.clone())?;
        dict.set_item(
            "bond_pairs",
            self.bond_pairs
                .iter()
                .map(|&(begin_idx, end_idx)| vec![begin_idx, end_idx])
                .collect::<Vec<_>>(),
        )?;
        dict.set_item("bond_kinds", self.bond_kinds.clone())?;
        dict.set_item("writer_do_isomeric_smiles", self.writer_do_isomeric_smiles)?;
        dict.set_item("writer_kekule_smiles", self.writer_kekule_smiles)?;
        dict.set_item("writer_all_bonds_explicit", self.writer_all_bonds_explicit)?;
        dict.set_item("writer_all_hs_explicit", self.writer_all_hs_explicit)?;
        dict.set_item(
            "writer_ignore_atom_map_numbers",
            self.writer_ignore_atom_map_numbers,
        )?;
        dict.set_item("identity_parse_with_rdkit", self.identity_parse_with_rdkit)?;
        dict.set_item("identity_canonical", self.identity_canonical)?;
        dict.set_item(
            "identity_do_isomeric_smiles",
            self.identity_do_isomeric_smiles,
        )?;
        dict.set_item("identity_kekule_smiles", self.identity_kekule_smiles)?;
        dict.set_item("identity_rooted_at_atom", self.identity_rooted_at_atom)?;
        dict.set_item(
            "identity_all_bonds_explicit",
            self.identity_all_bonds_explicit,
        )?;
        dict.set_item("identity_all_hs_explicit", self.identity_all_hs_explicit)?;
        dict.set_item("identity_do_random", self.identity_do_random)?;
        dict.set_item(
            "identity_ignore_atom_map_numbers",
            self.identity_ignore_atom_map_numbers,
        )?;
        if !self.atom_chiral_tags.is_empty() {
            dict.set_item("atom_chiral_tags", self.atom_chiral_tags.clone())?;
            dict.set_item(
                "atom_stereo_neighbor_orders",
                self.atom_stereo_neighbor_orders.clone(),
            )?;
            dict.set_item(
                "atom_explicit_h_counts",
                self.atom_explicit_h_counts.clone(),
            )?;
            dict.set_item(
                "atom_implicit_h_counts",
                self.atom_implicit_h_counts.clone(),
            )?;
        }
        if !self.bond_stereo_kinds.is_empty() {
            dict.set_item("bond_stereo_kinds", self.bond_stereo_kinds.clone())?;
            dict.set_item(
                "bond_stereo_atoms",
                self.bond_stereo_atoms
                    .iter()
                    .map(|&(begin_idx, end_idx)| vec![begin_idx, end_idx])
                    .collect::<Vec<_>>(),
            )?;
            dict.set_item("bond_dirs", self.bond_dirs.clone())?;
            dict.set_item(
                "bond_begin_atom_indices",
                self.bond_begin_atom_indices.clone(),
            )?;
            dict.set_item("bond_end_atom_indices", self.bond_end_atom_indices.clone())?;
        }
        Ok(dict)
    }
}

#[pyclass(name = "PreparedSmilesGraph", module = "grimace._core", frozen)]
pub struct PyPreparedSmilesGraph {
    data: PreparedSmilesGraphData,
}

#[pymethods]
impl PyPreparedSmilesGraph {
    #[new]
    fn new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            data: PreparedSmilesGraphData::from_any(data)?,
        })
    }

    #[getter]
    fn schema_version(&self) -> usize {
        self.data.schema_version
    }

    #[getter]
    fn surface_kind(&self) -> String {
        self.data.surface_kind.clone()
    }

    #[getter]
    fn policy_name(&self) -> String {
        self.data.policy_name.clone()
    }

    #[getter]
    fn policy_digest(&self) -> String {
        self.data.policy_digest.clone()
    }

    #[getter]
    fn rdkit_version(&self) -> String {
        self.data.rdkit_version.clone()
    }

    #[getter]
    fn identity_smiles(&self) -> String {
        self.data.identity_smiles.clone()
    }

    #[getter]
    fn atom_count(&self) -> usize {
        self.data.atom_count
    }

    #[getter]
    fn bond_count(&self) -> usize {
        self.data.bond_count
    }

    #[getter]
    fn atom_tokens(&self) -> Vec<String> {
        self.data.atom_tokens.clone()
    }

    fn neighbors_of(&self, atom_idx: usize) -> PyResult<Vec<usize>> {
        self.data
            .neighbors
            .get(atom_idx)
            .cloned()
            .ok_or_else(|| PyIndexError::new_err("atom_idx out of range"))
    }

    fn bond_token(&self, begin_idx: usize, end_idx: usize) -> PyResult<String> {
        if begin_idx >= self.data.atom_count || end_idx >= self.data.atom_count {
            return Err(PyIndexError::new_err("atom index out of range"));
        }
        self.data
            .bond_token(begin_idx, end_idx)
            .map(str::to_owned)
            .ok_or_else(|| {
                PyKeyError::new_err(format!("No bond between atoms {begin_idx} and {end_idx}"))
            })
    }

    fn validate_policy(&self, policy_name: &str, policy_digest: &str) -> PyResult<()> {
        if self.data.policy_name != policy_name || self.data.policy_digest != policy_digest {
            return Err(PyValueError::new_err(
                "PreparedSmilesGraph does not match the provided policy",
            ));
        }
        Ok(())
    }

    fn enumerate_rooted_connected_nonstereo_support(
        &self,
        root_idx: isize,
    ) -> PyResult<Vec<String>> {
        crate::rooted_nonstereo::enumerate_rooted_connected_nonstereo_smiles_support(
            &self.data, root_idx,
        )
    }

    fn enumerate_rooted_connected_stereo_support(&self, root_idx: isize) -> PyResult<Vec<String>> {
        crate::rooted_stereo::enumerate_rooted_connected_stereo_smiles_support(&self.data, root_idx)
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        Ok(self.data.to_pydict(py)?.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "PreparedSmilesGraph(surface_kind={:?}, atom_count={}, bond_count={}, policy_name={:?}, policy_digest={:?})",
            self.data.surface_kind,
            self.data.atom_count,
            self.data.bond_count,
            self.data.policy_name,
            self.data.policy_digest,
        )
    }
}

#[pyfunction]
pub fn prepared_smiles_graph_schema_version() -> usize {
    PREPARED_SMILES_GRAPH_SCHEMA_VERSION
}

#[pyfunction]
pub fn mol_to_smiles_support(
    graph: &Bound<'_, PyAny>,
    root_idx: isize,
    isomeric_smiles: bool,
) -> PyResult<Vec<String>> {
    let graph = PreparedSmilesGraphData::from_any(graph)?;
    mol_to_smiles_support_data(&graph, root_idx, isomeric_smiles)
}

#[cfg(test)]
mod tests {
    use super::{
        mol_to_smiles_support_data, PreparedSmilesGraphData, CONNECTED_NONSTEREO_SURFACE,
        CONNECTED_STEREO_SURFACE, PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
    };
    use pyo3::Python;

    fn assert_validation_error(graph: PreparedSmilesGraphData, expected: &str) {
        Python::initialize();
        let err = graph.validate().expect_err("graph should be rejected");
        let message = err.to_string();
        assert!(
            message.contains(expected),
            "expected error containing {expected:?}, got {message:?}",
        );
    }

    fn sample_graph() -> PreparedSmilesGraphData {
        PreparedSmilesGraphData {
            schema_version: PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
            surface_kind: CONNECTED_NONSTEREO_SURFACE.to_owned(),
            policy_name: "test_policy".to_owned(),
            policy_digest: "deadbeef".to_owned(),
            rdkit_version: "2025.09.6".to_owned(),
            identity_smiles: "CC".to_owned(),
            atom_count: 2,
            bond_count: 1,
            atom_atomic_numbers: vec![6, 6],
            atom_is_aromatic: vec![false, false],
            atom_isotopes: vec![0, 0],
            atom_formal_charges: vec![0, 0],
            atom_total_hs: vec![3, 3],
            atom_radical_electrons: vec![0, 0],
            atom_map_numbers: vec![0, 0],
            atom_tokens: vec!["C".to_owned(), "C".to_owned()],
            neighbors: vec![vec![1], vec![0]],
            neighbor_bond_tokens: vec![vec!["".to_owned()], vec!["".to_owned()]],
            bond_pairs: vec![(0, 1)],
            bond_kinds: vec!["SINGLE".to_owned()],
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
            atom_chiral_tags: Vec::new(),
            atom_stereo_neighbor_orders: Vec::new(),
            atom_explicit_h_counts: Vec::new(),
            atom_implicit_h_counts: Vec::new(),
            bond_stereo_kinds: Vec::new(),
            bond_stereo_atoms: Vec::new(),
            bond_dirs: Vec::new(),
            bond_begin_atom_indices: Vec::new(),
            bond_end_atom_indices: Vec::new(),
        }
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

    #[test]
    fn validate_accepts_consistent_graph() {
        sample_graph()
            .validate()
            .expect("sample graph should validate");
    }

    #[test]
    fn validate_rejects_neighbor_token_length_mismatch() {
        let mut broken = sample_graph();
        broken.neighbor_bond_tokens[0].clear();

        assert_validation_error(broken, "neighbor token row length mismatch");
    }

    #[test]
    fn validate_rejects_schema_version_mismatch() {
        let mut broken = sample_graph();
        broken.schema_version = 999;

        assert_validation_error(broken, "Unexpected PreparedSmilesGraph schema version");
    }

    #[test]
    fn validate_rejects_unsorted_neighbors() {
        let mut broken = sample_graph();
        broken.atom_count = 3;
        broken.bond_count = 2;
        broken.identity_smiles = "CCO".to_owned();
        broken.atom_atomic_numbers = vec![6, 6, 8];
        broken.atom_is_aromatic = vec![false, false, false];
        broken.atom_isotopes = vec![0, 0, 0];
        broken.atom_formal_charges = vec![0, 0, 0];
        broken.atom_total_hs = vec![3, 2, 1];
        broken.atom_radical_electrons = vec![0, 0, 0];
        broken.atom_map_numbers = vec![0, 0, 0];
        broken.atom_tokens = vec!["C".to_owned(), "C".to_owned(), "O".to_owned()];
        broken.neighbors = vec![vec![1], vec![2, 0], vec![1]];
        broken.neighbor_bond_tokens = vec![
            vec!["".to_owned()],
            vec!["".to_owned(), "".to_owned()],
            vec!["".to_owned()],
        ];
        broken.bond_pairs = vec![(0, 1), (1, 2)];
        broken.bond_kinds = vec!["SINGLE".to_owned(), "SINGLE".to_owned()];

        assert_validation_error(broken, "neighbor rows must be sorted");
    }

    #[test]
    fn validate_rejects_neighbor_graph_bond_pair_disagreement() {
        let mut broken = sample_graph();
        broken.bond_pairs.clear();
        broken.bond_count = 0;
        broken.bond_kinds.clear();

        assert_validation_error(broken, "neighbor graph and bond_pairs disagree");
    }

    #[test]
    fn validate_rejects_asymmetric_bond_tokens() {
        let mut broken = sample_graph();
        broken.neighbor_bond_tokens[1][0] = "=".to_owned();

        assert_validation_error(broken, "bond tokens must be symmetric");
    }

    #[test]
    fn validate_accepts_consistent_stereo_graph() {
        sample_stereo_graph()
            .validate()
            .expect("sample stereo graph should validate");
    }

    #[test]
    fn validate_rejects_connected_stereo_without_stereo_atom_metadata() {
        let mut broken = sample_stereo_graph();
        broken.atom_chiral_tags.clear();

        assert_validation_error(
            broken,
            "connected_stereo surface requires stereo atom metadata",
        );
    }

    #[test]
    fn validate_rejects_connected_stereo_with_inconsistent_neighbor_order() {
        let mut broken = sample_stereo_graph();
        broken.atom_stereo_neighbor_orders[1] = vec![0, 3];

        assert_validation_error(broken, "stereo neighbor order must match neighbors");
    }

    #[test]
    fn bond_token_finds_existing_bond() {
        let graph = sample_graph();
        assert_eq!(Some(""), graph.bond_token(0, 1));
        assert_eq!(None, graph.bond_token(0, 0));
    }

    #[test]
    fn mol_to_smiles_support_dispatches_nonstereo() {
        let graph = sample_graph();
        let support = mol_to_smiles_support_data(&graph, 0, false).expect("nonstereo support");
        let expected =
            crate::rooted_nonstereo::enumerate_rooted_connected_nonstereo_smiles_support(&graph, 0)
                .expect("direct nonstereo support");
        assert_eq!(expected, support);
    }

    #[test]
    fn mol_to_smiles_support_dispatches_stereo() {
        let graph = sample_stereo_graph();
        let support = mol_to_smiles_support_data(&graph, 0, true).expect("stereo support");
        let expected =
            crate::rooted_stereo::enumerate_rooted_connected_stereo_smiles_support(&graph, 0)
                .expect("direct stereo support");
        assert_eq!(expected, support);
    }

    #[test]
    fn mol_to_smiles_support_rejects_surface_mismatch() {
        Python::initialize();
        let err = mol_to_smiles_support_data(&sample_graph(), 0, true).expect_err("should fail");
        assert!(
            err.to_string().contains("isomeric_smiles=True"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn mol_to_smiles_support_preserves_root_validation() {
        Python::initialize();
        let err = mol_to_smiles_support_data(&sample_graph(), -1, false).expect_err("should fail");
        assert!(
            err.to_string().contains("root_idx out of range"),
            "unexpected error: {err}",
        );
    }
}
