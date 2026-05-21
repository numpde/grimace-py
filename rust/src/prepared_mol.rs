use std::collections::BTreeSet;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyDict, PyList};

use crate::prepared_graph::{PreparedSmilesGraphData, PyPreparedSmilesGraph};

const PREPARED_MOL_SCHEMA_VERSION: usize = 1;
const PREPARED_MOL_BINARY_MAGIC: &[u8] = b"GRIMACEPM\0";
const PREPARED_MOL_BINARY_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
struct PreparedMolWriterFlags {
    isomeric_smiles: bool,
    kekule_smiles: bool,
    all_bonds_explicit: bool,
    all_hs_explicit: bool,
    ignore_atom_map_numbers: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PreparedMolFragmentData {
    atom_indices: Vec<usize>,
    prepared_graph: PreparedSmilesGraphData,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PreparedMolData {
    schema_version: usize,
    writer_flags: PreparedMolWriterFlags,
    fragments: Vec<PreparedMolFragmentData>,
}

fn required_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing PreparedMol field: {key}")))
}

fn required_dict_item<'py>(
    dict: &Bound<'py, PyDict>,
    key: &str,
    message: &str,
) -> PyResult<Bound<'py, PyDict>> {
    required_item(dict, key)?
        .cast::<PyDict>()
        .cloned()
        .map_err(|_| PyValueError::new_err(message.to_owned()))
}

fn required_bool(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    let value = required_item(dict, key)?;
    if !value.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(format!(
            "PreparedMol writer flag {key:?} must be a bool"
        )));
    }
    value.extract()
}

fn required_usize(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<usize> {
    let value = required_item(dict, key)?;
    if value.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(format!(
            "PreparedMol field {key:?} must be an integer"
        )));
    }
    value
        .extract()
        .map_err(|_| PyValueError::new_err(format!("PreparedMol field {key:?} must be an integer")))
}

fn extract_usize_list(value: &Bound<'_, PyAny>, message: &str) -> PyResult<Vec<usize>> {
    let items = value
        .cast::<PyList>()
        .map_err(|_| PyValueError::new_err(message.to_owned()))?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if item.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(
                "PreparedMol fragment atom indices must be integers",
            ));
        }
        out.push(item.extract().map_err(|_| {
            PyValueError::new_err("PreparedMol fragment atom indices must be integers")
        })?);
    }
    Ok(out)
}

impl PreparedMolWriterFlags {
    fn from_dict(dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            isomeric_smiles: required_bool(dict, "isomeric_smiles")?,
            kekule_smiles: required_bool(dict, "kekule_smiles")?,
            all_bonds_explicit: required_bool(dict, "all_bonds_explicit")?,
            all_hs_explicit: required_bool(dict, "all_hs_explicit")?,
            ignore_atom_map_numbers: required_bool(dict, "ignore_atom_map_numbers")?,
        })
    }

    fn validate_graph(&self, graph: &PreparedSmilesGraphData) -> PyResult<()> {
        let actual = (
            graph.writer_do_isomeric_smiles,
            graph.writer_kekule_smiles,
            graph.writer_all_bonds_explicit,
            graph.writer_all_hs_explicit,
            graph.writer_ignore_atom_map_numbers,
        );
        let expected = (
            self.isomeric_smiles,
            self.kekule_smiles,
            self.all_bonds_explicit,
            self.all_hs_explicit,
            self.ignore_atom_map_numbers,
        );
        if actual != expected {
            return Err(PyValueError::new_err(
                "PreparedMol writer flags do not match fragment prepared graph",
            ));
        }
        Ok(())
    }
}

impl PreparedMolFragmentData {
    fn from_any(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dict = obj
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("PreparedMol fragment must be an object"))?;
        let atom_indices = extract_usize_list(
            &required_item(dict, "atom_indices")?,
            "PreparedMol fragment atom_indices must be an array",
        )?;
        let prepared_graph =
            PreparedSmilesGraphData::from_any(&required_item(dict, "prepared_graph")?)?;
        Ok(Self {
            atom_indices,
            prepared_graph,
        })
    }
}

impl PreparedMolData {
    fn from_any(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dict = obj
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("PreparedMol payload must be an object"))?;
        let schema_version = required_usize(dict, "schema_version")?;
        let writer_flags = PreparedMolWriterFlags::from_dict(&required_dict_item(
            dict,
            "writer_flags",
            "PreparedMol writer_flags must be an object",
        )?)?;
        let raw_fragments = required_item(dict, "fragments")?;
        let fragments_seq = raw_fragments
            .cast::<PyList>()
            .map_err(|_| PyValueError::new_err("PreparedMol fragments must be an array"))?;
        let mut fragments = Vec::new();
        for item in fragments_seq {
            fragments.push(PreparedMolFragmentData::from_any(&item)?);
        }

        let data = Self {
            schema_version,
            writer_flags,
            fragments,
        };
        data.validate()?;
        Ok(data)
    }

    fn validate(&self) -> PyResult<()> {
        if self.schema_version != PREPARED_MOL_SCHEMA_VERSION {
            return Err(PyValueError::new_err(format!(
                "Unsupported PreparedMol schema version: {}",
                self.schema_version
            )));
        }
        if self.fragments.is_empty() {
            return Err(PyValueError::new_err(
                "PreparedMol must contain at least one fragment",
            ));
        }

        let mut atom_indices_seen = BTreeSet::new();
        let mut previous_fragment_first_atom_idx: Option<usize> = None;
        for fragment in &self.fragments {
            self.writer_flags.validate_graph(&fragment.prepared_graph)?;
            if fragment.atom_indices.len() != fragment.prepared_graph.atom_count() {
                return Err(PyValueError::new_err(
                    "PreparedMol fragment atom_indices length does not match graph atom_count",
                ));
            }
            if fragment.atom_indices.is_empty() && self.fragments.len() != 1 {
                return Err(PyValueError::new_err(
                    "PreparedMol empty fragment is valid only for an empty molecule",
                ));
            }
            if !fragment
                .atom_indices
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            {
                return Err(PyValueError::new_err(
                    "PreparedMol fragment atom indices must be sorted",
                ));
            }
            if let Some(first_atom_idx) = fragment.atom_indices.first().copied() {
                if let Some(previous_atom_idx) = previous_fragment_first_atom_idx {
                    if first_atom_idx <= previous_atom_idx {
                        return Err(PyValueError::new_err(
                            "PreparedMol fragments must be ordered by atom index",
                        ));
                    }
                }
                previous_fragment_first_atom_idx = Some(first_atom_idx);
            }
            for atom_idx in &fragment.atom_indices {
                if !atom_indices_seen.insert(*atom_idx) {
                    return Err(PyValueError::new_err(
                        "PreparedMol fragment atom indices overlap",
                    ));
                }
            }
        }
        for (expected_idx, actual_idx) in atom_indices_seen.iter().copied().enumerate() {
            if expected_idx != actual_idx {
                return Err(PyValueError::new_err(
                    "PreparedMol fragment atom indices must cover all molecule atoms",
                ));
            }
        }
        Ok(())
    }

    fn to_binary(&self) -> Vec<u8> {
        let mut writer = BinaryWriter::new();
        writer.write_raw(PREPARED_MOL_BINARY_MAGIC);
        writer.write_u32(PREPARED_MOL_BINARY_VERSION);
        writer.write_usize(self.schema_version);
        writer.write_bool(self.writer_flags.isomeric_smiles);
        writer.write_bool(self.writer_flags.kekule_smiles);
        writer.write_bool(self.writer_flags.all_bonds_explicit);
        writer.write_bool(self.writer_flags.all_hs_explicit);
        writer.write_bool(self.writer_flags.ignore_atom_map_numbers);
        writer.write_usize(self.fragments.len());
        for fragment in &self.fragments {
            writer.write_vec_usize(&fragment.atom_indices);
            writer.write_graph(&fragment.prepared_graph);
        }
        writer.into_inner()
    }

    fn from_binary(data: &[u8]) -> PyResult<Self> {
        let mut reader = BinaryReader::new(data);
        reader.read_magic(PREPARED_MOL_BINARY_MAGIC)?;
        let format_version = reader.read_u32()?;
        if format_version != PREPARED_MOL_BINARY_VERSION {
            return Err(PyValueError::new_err(format!(
                "Unsupported PreparedMol binary format version: {format_version}"
            )));
        }
        let schema_version = reader.read_usize()?;
        let writer_flags = PreparedMolWriterFlags {
            isomeric_smiles: reader.read_bool()?,
            kekule_smiles: reader.read_bool()?,
            all_bonds_explicit: reader.read_bool()?,
            all_hs_explicit: reader.read_bool()?,
            ignore_atom_map_numbers: reader.read_bool()?,
        };
        let fragment_count = reader.read_len(8)?;
        let mut fragments = Vec::with_capacity(fragment_count);
        for _ in 0..fragment_count {
            let atom_indices = reader.read_vec_usize()?;
            let prepared_graph = reader.read_graph()?;
            fragments.push(PreparedMolFragmentData {
                atom_indices,
                prepared_graph,
            });
        }
        reader.finish()?;

        let data = Self {
            schema_version,
            writer_flags,
            fragments,
        };
        data.validate()?;
        Ok(data)
    }
}

struct BinaryWriter {
    data: Vec<u8>,
}

impl BinaryWriter {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn into_inner(self) -> Vec<u8> {
        self.data
    }

    fn write_raw(&mut self, value: &[u8]) {
        self.data.extend_from_slice(value);
    }

    fn write_u8(&mut self, value: u8) {
        self.data.push(value);
    }

    fn write_bool(&mut self, value: bool) {
        self.write_u8(if value { 1 } else { 0 });
    }

    fn write_u32(&mut self, value: u32) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    fn write_i32(&mut self, value: i32) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u64(&mut self, value: u64) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    fn write_i64(&mut self, value: i64) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    fn write_isize(&mut self, value: isize) {
        self.write_i64(value as i64);
    }

    fn write_string(&mut self, value: &str) {
        self.write_usize(value.len());
        self.write_raw(value.as_bytes());
    }

    fn write_vec_bool(&mut self, values: &[bool]) {
        self.write_usize(values.len());
        for value in values {
            self.write_bool(*value);
        }
    }

    fn write_vec_i32(&mut self, values: &[i32]) {
        self.write_usize(values.len());
        for value in values {
            self.write_i32(*value);
        }
    }

    fn write_vec_usize(&mut self, values: &[usize]) {
        self.write_usize(values.len());
        for value in values {
            self.write_usize(*value);
        }
    }

    fn write_vec_string(&mut self, values: &[String]) {
        self.write_usize(values.len());
        for value in values {
            self.write_string(value);
        }
    }

    fn write_vec_vec_usize(&mut self, values: &[Vec<usize>]) {
        self.write_usize(values.len());
        for value in values {
            self.write_vec_usize(value);
        }
    }

    fn write_vec_vec_string(&mut self, values: &[Vec<String>]) {
        self.write_usize(values.len());
        for value in values {
            self.write_vec_string(value);
        }
    }

    fn write_vec_usize_pairs(&mut self, values: &[(usize, usize)]) {
        self.write_usize(values.len());
        for &(left, right) in values {
            self.write_usize(left);
            self.write_usize(right);
        }
    }

    fn write_vec_isize_pairs(&mut self, values: &[(isize, isize)]) {
        self.write_usize(values.len());
        for &(left, right) in values {
            self.write_isize(left);
            self.write_isize(right);
        }
    }

    fn write_graph(&mut self, graph: &PreparedSmilesGraphData) {
        self.write_usize(graph.schema_version);
        self.write_string(&graph.surface_kind);
        self.write_string(&graph.policy_name);
        self.write_string(&graph.policy_digest);
        self.write_string(&graph.rdkit_version);
        self.write_string(&graph.identity_smiles);
        self.write_usize(graph.atom_count);
        self.write_usize(graph.bond_count);
        self.write_vec_usize(&graph.atom_atomic_numbers);
        self.write_vec_bool(&graph.atom_is_aromatic);
        self.write_vec_usize(&graph.atom_isotopes);
        self.write_vec_i32(&graph.atom_formal_charges);
        self.write_vec_usize(&graph.atom_total_hs);
        self.write_vec_usize(&graph.atom_radical_electrons);
        self.write_vec_usize(&graph.atom_map_numbers);
        self.write_vec_string(&graph.atom_tokens);
        self.write_vec_vec_usize(&graph.neighbors);
        self.write_vec_vec_string(&graph.neighbor_bond_tokens);
        self.write_vec_usize_pairs(&graph.bond_pairs);
        self.write_vec_string(&graph.bond_kinds);
        self.write_bool(graph.writer_do_isomeric_smiles);
        self.write_bool(graph.writer_kekule_smiles);
        self.write_bool(graph.writer_all_bonds_explicit);
        self.write_bool(graph.writer_all_hs_explicit);
        self.write_bool(graph.writer_ignore_atom_map_numbers);
        self.write_bool(graph.identity_parse_with_rdkit);
        self.write_bool(graph.identity_canonical);
        self.write_bool(graph.identity_do_isomeric_smiles);
        self.write_bool(graph.identity_kekule_smiles);
        self.write_i32(graph.identity_rooted_at_atom);
        self.write_bool(graph.identity_all_bonds_explicit);
        self.write_bool(graph.identity_all_hs_explicit);
        self.write_bool(graph.identity_do_random);
        self.write_bool(graph.identity_ignore_atom_map_numbers);
        self.write_vec_string(&graph.atom_chiral_tags);
        self.write_vec_vec_usize(&graph.atom_stereo_neighbor_orders);
        self.write_vec_usize(&graph.atom_explicit_h_counts);
        self.write_vec_usize(&graph.atom_implicit_h_counts);
        self.write_vec_string(&graph.bond_stereo_kinds);
        self.write_vec_isize_pairs(&graph.bond_stereo_atoms);
        self.write_vec_string(&graph.bond_dirs);
        self.write_vec_usize(&graph.bond_begin_atom_indices);
        self.write_vec_usize(&graph.bond_end_atom_indices);
    }
}

struct BinaryReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> BinaryReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn read_exact(&mut self, len: usize) -> PyResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| PyValueError::new_err("Malformed PreparedMol binary payload"))?;
        if end > self.data.len() {
            return Err(PyValueError::new_err(
                "Malformed PreparedMol binary payload is truncated",
            ));
        }
        let out = &self.data[self.offset..end];
        self.offset = end;
        Ok(out)
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.offset
    }

    fn read_len(&mut self, min_item_bytes: usize) -> PyResult<usize> {
        let len = self.read_usize()?;
        if len > self.remaining() / min_item_bytes {
            return Err(PyValueError::new_err(
                "Malformed PreparedMol binary length is out of range",
            ));
        }
        Ok(len)
    }

    fn finish(&self) -> PyResult<()> {
        if self.offset != self.data.len() {
            return Err(PyValueError::new_err(
                "Malformed PreparedMol binary payload has trailing bytes",
            ));
        }
        Ok(())
    }

    fn read_magic(&mut self, expected: &[u8]) -> PyResult<()> {
        let actual = self.read_exact(expected.len())?;
        if actual != expected {
            return Err(PyValueError::new_err("Malformed PreparedMol binary magic"));
        }
        Ok(())
    }

    fn read_u8(&mut self) -> PyResult<u8> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_bool(&mut self) -> PyResult<bool> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(PyValueError::new_err(
                "Malformed PreparedMol binary boolean value",
            )),
        }
    }

    fn read_u32(&mut self) -> PyResult<u32> {
        let mut bytes = [0_u8; 4];
        bytes.copy_from_slice(self.read_exact(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_i32(&mut self) -> PyResult<i32> {
        let mut bytes = [0_u8; 4];
        bytes.copy_from_slice(self.read_exact(4)?);
        Ok(i32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> PyResult<u64> {
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(self.read_exact(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i64(&mut self) -> PyResult<i64> {
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(self.read_exact(8)?);
        Ok(i64::from_le_bytes(bytes))
    }

    fn read_usize(&mut self) -> PyResult<usize> {
        usize::try_from(self.read_u64()?).map_err(|_| {
            PyValueError::new_err("Malformed PreparedMol binary integer is out of range")
        })
    }

    fn read_isize(&mut self) -> PyResult<isize> {
        isize::try_from(self.read_i64()?).map_err(|_| {
            PyValueError::new_err("Malformed PreparedMol binary integer is out of range")
        })
    }

    fn read_string(&mut self) -> PyResult<String> {
        let len = self.read_len(1)?;
        let bytes = self.read_exact(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| PyValueError::new_err("Malformed PreparedMol binary string"))
    }

    fn read_vec_bool(&mut self) -> PyResult<Vec<bool>> {
        let len = self.read_len(1)?;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.read_bool()?);
        }
        Ok(out)
    }

    fn read_vec_i32(&mut self) -> PyResult<Vec<i32>> {
        let len = self.read_len(4)?;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.read_i32()?);
        }
        Ok(out)
    }

    fn read_vec_usize(&mut self) -> PyResult<Vec<usize>> {
        let len = self.read_len(8)?;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.read_usize()?);
        }
        Ok(out)
    }

    fn read_vec_string(&mut self) -> PyResult<Vec<String>> {
        let len = self.read_len(8)?;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.read_string()?);
        }
        Ok(out)
    }

    fn read_vec_vec_usize(&mut self) -> PyResult<Vec<Vec<usize>>> {
        let len = self.read_len(8)?;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.read_vec_usize()?);
        }
        Ok(out)
    }

    fn read_vec_vec_string(&mut self) -> PyResult<Vec<Vec<String>>> {
        let len = self.read_len(8)?;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.read_vec_string()?);
        }
        Ok(out)
    }

    fn read_vec_usize_pairs(&mut self) -> PyResult<Vec<(usize, usize)>> {
        let len = self.read_len(16)?;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push((self.read_usize()?, self.read_usize()?));
        }
        Ok(out)
    }

    fn read_vec_isize_pairs(&mut self) -> PyResult<Vec<(isize, isize)>> {
        let len = self.read_len(16)?;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push((self.read_isize()?, self.read_isize()?));
        }
        Ok(out)
    }

    fn read_graph(&mut self) -> PyResult<PreparedSmilesGraphData> {
        let graph = PreparedSmilesGraphData {
            schema_version: self.read_usize()?,
            surface_kind: self.read_string()?,
            policy_name: self.read_string()?,
            policy_digest: self.read_string()?,
            rdkit_version: self.read_string()?,
            identity_smiles: self.read_string()?,
            atom_count: self.read_usize()?,
            bond_count: self.read_usize()?,
            atom_atomic_numbers: self.read_vec_usize()?,
            atom_is_aromatic: self.read_vec_bool()?,
            atom_isotopes: self.read_vec_usize()?,
            atom_formal_charges: self.read_vec_i32()?,
            atom_total_hs: self.read_vec_usize()?,
            atom_radical_electrons: self.read_vec_usize()?,
            atom_map_numbers: self.read_vec_usize()?,
            atom_tokens: self.read_vec_string()?,
            neighbors: self.read_vec_vec_usize()?,
            neighbor_bond_tokens: self.read_vec_vec_string()?,
            bond_pairs: self.read_vec_usize_pairs()?,
            bond_kinds: self.read_vec_string()?,
            writer_do_isomeric_smiles: self.read_bool()?,
            writer_kekule_smiles: self.read_bool()?,
            writer_all_bonds_explicit: self.read_bool()?,
            writer_all_hs_explicit: self.read_bool()?,
            writer_ignore_atom_map_numbers: self.read_bool()?,
            identity_parse_with_rdkit: self.read_bool()?,
            identity_canonical: self.read_bool()?,
            identity_do_isomeric_smiles: self.read_bool()?,
            identity_kekule_smiles: self.read_bool()?,
            identity_rooted_at_atom: self.read_i32()?,
            identity_all_bonds_explicit: self.read_bool()?,
            identity_all_hs_explicit: self.read_bool()?,
            identity_do_random: self.read_bool()?,
            identity_ignore_atom_map_numbers: self.read_bool()?,
            atom_chiral_tags: self.read_vec_string()?,
            atom_stereo_neighbor_orders: self.read_vec_vec_usize()?,
            atom_explicit_h_counts: self.read_vec_usize()?,
            atom_implicit_h_counts: self.read_vec_usize()?,
            bond_stereo_kinds: self.read_vec_string()?,
            bond_stereo_atoms: self.read_vec_isize_pairs()?,
            bond_dirs: self.read_vec_string()?,
            bond_begin_atom_indices: self.read_vec_usize()?,
            bond_end_atom_indices: self.read_vec_usize()?,
        };
        graph.validate()?;
        Ok(graph)
    }
}

#[pyclass(name = "PreparedMol", module = "grimace._core", frozen)]
pub struct PyPreparedMol {
    data: PreparedMolData,
}

#[pymethods]
impl PyPreparedMol {
    #[new]
    fn new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            data: PreparedMolData::from_any(data)?,
        })
    }

    #[staticmethod]
    fn from_bytes(data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let data = PreparedMolData::from_binary(data.as_bytes()).map_err(|err| {
            PyValueError::new_err(format!("Malformed PreparedMol payload: {err}"))
        })?;
        Ok(Self { data })
    }

    fn to_bytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        Ok(PyBytes::new(py, &self.data.to_binary()).unbind())
    }

    fn writer_flag_values(&self) -> (bool, bool, bool, bool, bool) {
        (
            self.data.writer_flags.isomeric_smiles,
            self.data.writer_flags.kekule_smiles,
            self.data.writer_flags.all_bonds_explicit,
            self.data.writer_flags.all_hs_explicit,
            self.data.writer_flags.ignore_atom_map_numbers,
        )
    }

    fn fragment_count(&self) -> usize {
        self.data.fragments.len()
    }

    fn fragment_atom_indices(&self, fragment_idx: usize) -> PyResult<Vec<usize>> {
        self.data
            .fragments
            .get(fragment_idx)
            .map(|fragment| fragment.atom_indices.clone())
            .ok_or_else(|| PyIndexError::new_err("PreparedMol fragment index out of range"))
    }

    fn fragment_prepared_graph(&self, fragment_idx: usize) -> PyResult<PyPreparedSmilesGraph> {
        self.data
            .fragments
            .get(fragment_idx)
            .map(|fragment| PyPreparedSmilesGraph::from_data(fragment.prepared_graph.clone()))
            .ok_or_else(|| PyIndexError::new_err("PreparedMol fragment index out of range"))
    }

    fn __repr__(&self) -> String {
        format!(
            "PreparedMol(schema_version={}, fragment_count={})",
            self.data.schema_version,
            self.data.fragments.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BinaryReader, PreparedMolData, PreparedMolFragmentData, PreparedMolWriterFlags,
        PREPARED_MOL_SCHEMA_VERSION,
    };
    use crate::prepared_graph::{
        PreparedSmilesGraphData, CONNECTED_NONSTEREO_SURFACE, PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
    };

    fn writer_flags() -> PreparedMolWriterFlags {
        PreparedMolWriterFlags {
            isomeric_smiles: false,
            kekule_smiles: false,
            all_bonds_explicit: false,
            all_hs_explicit: false,
            ignore_atom_map_numbers: false,
        }
    }

    fn single_atom_graph() -> PreparedSmilesGraphData {
        PreparedSmilesGraphData {
            schema_version: PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
            surface_kind: CONNECTED_NONSTEREO_SURFACE.to_owned(),
            policy_name: "test_policy".to_owned(),
            policy_digest: "deadbeef".to_owned(),
            rdkit_version: "2026.03.1".to_owned(),
            identity_smiles: "C".to_owned(),
            atom_count: 1,
            bond_count: 0,
            atom_atomic_numbers: vec![6],
            atom_is_aromatic: vec![false],
            atom_isotopes: vec![0],
            atom_formal_charges: vec![0],
            atom_total_hs: vec![4],
            atom_radical_electrons: vec![0],
            atom_map_numbers: vec![0],
            atom_tokens: vec!["C".to_owned()],
            neighbors: vec![Vec::new()],
            neighbor_bond_tokens: vec![Vec::new()],
            bond_pairs: Vec::new(),
            bond_kinds: Vec::new(),
            writer_do_isomeric_smiles: false,
            writer_kekule_smiles: false,
            writer_all_bonds_explicit: false,
            writer_all_hs_explicit: false,
            writer_ignore_atom_map_numbers: false,
            identity_parse_with_rdkit: true,
            identity_canonical: true,
            identity_do_isomeric_smiles: false,
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

    fn empty_graph() -> PreparedSmilesGraphData {
        let mut graph = single_atom_graph();
        graph.identity_smiles = String::new();
        graph.atom_count = 0;
        graph.atom_atomic_numbers = Vec::new();
        graph.atom_is_aromatic = Vec::new();
        graph.atom_isotopes = Vec::new();
        graph.atom_formal_charges = Vec::new();
        graph.atom_total_hs = Vec::new();
        graph.atom_radical_electrons = Vec::new();
        graph.atom_map_numbers = Vec::new();
        graph.atom_tokens = Vec::new();
        graph.neighbors = Vec::new();
        graph.neighbor_bond_tokens = Vec::new();
        graph
    }

    fn prepared_mol_with_fragments(fragments: Vec<PreparedMolFragmentData>) -> PreparedMolData {
        PreparedMolData {
            schema_version: PREPARED_MOL_SCHEMA_VERSION,
            writer_flags: writer_flags(),
            fragments,
        }
    }

    #[test]
    fn validates_distinct_fragment_atom_indices() {
        let prepared = prepared_mol_with_fragments(vec![
            PreparedMolFragmentData {
                atom_indices: vec![0],
                prepared_graph: single_atom_graph(),
            },
            PreparedMolFragmentData {
                atom_indices: vec![1],
                prepared_graph: single_atom_graph(),
            },
        ]);

        prepared.validate().expect("prepared mol should validate");
    }

    #[test]
    fn rejects_empty_fragment_list() {
        let prepared = prepared_mol_with_fragments(Vec::new());

        assert!(prepared.validate().is_err());
    }

    #[test]
    fn rejects_fragment_atom_count_mismatch() {
        let prepared = prepared_mol_with_fragments(vec![PreparedMolFragmentData {
            atom_indices: Vec::new(),
            prepared_graph: single_atom_graph(),
        }]);

        assert!(prepared.validate().is_err());
    }

    #[test]
    fn rejects_empty_fragment_inside_nonempty_molecule() {
        let prepared = prepared_mol_with_fragments(vec![
            PreparedMolFragmentData {
                atom_indices: Vec::new(),
                prepared_graph: empty_graph(),
            },
            PreparedMolFragmentData {
                atom_indices: vec![0],
                prepared_graph: single_atom_graph(),
            },
        ]);

        assert!(prepared.validate().is_err());
    }

    #[test]
    fn rejects_overlapping_fragment_atom_indices() {
        let prepared = prepared_mol_with_fragments(vec![
            PreparedMolFragmentData {
                atom_indices: vec![0],
                prepared_graph: single_atom_graph(),
            },
            PreparedMolFragmentData {
                atom_indices: vec![0],
                prepared_graph: single_atom_graph(),
            },
        ]);

        assert!(prepared.validate().is_err());
    }

    #[test]
    fn rejects_out_of_order_fragments() {
        let prepared = prepared_mol_with_fragments(vec![
            PreparedMolFragmentData {
                atom_indices: vec![1],
                prepared_graph: single_atom_graph(),
            },
            PreparedMolFragmentData {
                atom_indices: vec![0],
                prepared_graph: single_atom_graph(),
            },
        ]);

        assert!(prepared.validate().is_err());
    }

    #[test]
    fn rejects_unsorted_fragment_atom_indices() {
        let mut graph = single_atom_graph();
        graph.atom_count = 2;
        graph.atom_atomic_numbers = vec![6, 6];
        graph.atom_is_aromatic = vec![false, false];
        graph.atom_isotopes = vec![0, 0];
        graph.atom_formal_charges = vec![0, 0];
        graph.atom_total_hs = vec![4, 4];
        graph.atom_radical_electrons = vec![0, 0];
        graph.atom_map_numbers = vec![0, 0];
        graph.atom_tokens = vec!["C".to_owned(), "C".to_owned()];
        graph.neighbors = vec![Vec::new(), Vec::new()];
        graph.neighbor_bond_tokens = vec![Vec::new(), Vec::new()];
        let prepared = prepared_mol_with_fragments(vec![PreparedMolFragmentData {
            atom_indices: vec![1, 0],
            prepared_graph: graph,
        }]);

        assert!(prepared.validate().is_err());
    }

    #[test]
    fn rejects_missing_global_atom_indices() {
        let prepared = prepared_mol_with_fragments(vec![PreparedMolFragmentData {
            atom_indices: vec![1],
            prepared_graph: single_atom_graph(),
        }]);

        assert!(prepared.validate().is_err());
    }

    #[test]
    fn rejects_writer_flag_mismatch() {
        let mut graph = single_atom_graph();
        graph.writer_all_hs_explicit = true;
        let prepared = prepared_mol_with_fragments(vec![PreparedMolFragmentData {
            atom_indices: vec![0],
            prepared_graph: graph,
        }]);

        assert!(prepared.validate().is_err());
    }

    #[test]
    fn binary_roundtrip_preserves_prepared_mol() {
        let prepared = prepared_mol_with_fragments(vec![
            PreparedMolFragmentData {
                atom_indices: vec![0],
                prepared_graph: single_atom_graph(),
            },
            PreparedMolFragmentData {
                atom_indices: vec![1],
                prepared_graph: single_atom_graph(),
            },
        ]);

        let encoded = prepared.to_binary();
        assert!(encoded.starts_with(super::PREPARED_MOL_BINARY_MAGIC));
        assert_eq!(
            prepared,
            PreparedMolData::from_binary(&encoded).expect("binary roundtrip should decode"),
        );

        assert!(PreparedMolData::from_binary(&encoded[..encoded.len() - 1]).is_err());

        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(PreparedMolData::from_binary(&trailing).is_err());

        let mut bad_magic = encoded;
        bad_magic[0] = b'X';
        assert!(PreparedMolData::from_binary(&bad_magic).is_err());
    }

    #[test]
    fn binary_reader_rejects_length_larger_than_remaining_payload() {
        let data = u64::MAX.to_le_bytes();

        assert!(BinaryReader::new(&data).read_string().is_err());
        assert!(BinaryReader::new(&data).read_vec_usize().is_err());
    }
}
