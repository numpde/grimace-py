use std::collections::BTreeSet;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyDict, PyList};

use crate::prepared_graph::{PreparedSmilesGraphData, PyPreparedSmilesGraph};

const PREPARED_MOL_SCHEMA_VERSION: usize = 1;

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
    value.extract()
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
        if atom_indices.len() != prepared_graph.atom_count() {
            return Err(PyValueError::new_err(
                "PreparedMol fragment atom_indices length does not match graph atom_count",
            ));
        }
        let unique_indices = atom_indices.iter().copied().collect::<BTreeSet<_>>();
        if unique_indices.len() != atom_indices.len() {
            return Err(PyValueError::new_err(
                "PreparedMol fragment atom indices must be unique",
            ));
        }
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

        let mut atom_indices_seen = BTreeSet::new();
        for fragment in &self.fragments {
            self.writer_flags.validate_graph(&fragment.prepared_graph)?;
            for atom_idx in &fragment.atom_indices {
                if !atom_indices_seen.insert(*atom_idx) {
                    return Err(PyValueError::new_err(
                        "PreparedMol fragment atom indices overlap",
                    ));
                }
            }
        }
        Ok(())
    }

    fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("schema_version", self.schema_version)?;

        let writer_flags = PyDict::new(py);
        writer_flags.set_item("isomeric_smiles", self.writer_flags.isomeric_smiles)?;
        writer_flags.set_item("kekule_smiles", self.writer_flags.kekule_smiles)?;
        writer_flags.set_item("all_bonds_explicit", self.writer_flags.all_bonds_explicit)?;
        writer_flags.set_item("all_hs_explicit", self.writer_flags.all_hs_explicit)?;
        writer_flags.set_item(
            "ignore_atom_map_numbers",
            self.writer_flags.ignore_atom_map_numbers,
        )?;
        dict.set_item("writer_flags", writer_flags)?;

        let fragments = PyList::empty(py);
        for fragment in &self.fragments {
            let fragment_dict = PyDict::new(py);
            fragment_dict.set_item("atom_indices", fragment.atom_indices.clone())?;
            fragment_dict.set_item("prepared_graph", fragment.prepared_graph.to_pydict(py)?)?;
            fragments.append(fragment_dict)?;
        }
        dict.set_item("fragments", fragments)?;
        Ok(dict)
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
        let text = std::str::from_utf8(data.as_bytes())
            .map_err(|_| PyValueError::new_err("Malformed PreparedMol payload"))?;
        let py = data.py();
        let json = py.import("json")?;
        let payload = json
            .call_method1("loads", (text,))
            .map_err(|_| PyValueError::new_err("Malformed PreparedMol payload"))?;
        Self::new(&payload).map_err(|err| {
            PyValueError::new_err(format!("Malformed PreparedMol payload: {err}"))
        })
    }

    fn to_bytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let payload = self.data.to_pydict(py)?;
        let json = py.import("json")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("separators", (",", ":"))?;
        let text: String = json
            .getattr("dumps")?
            .call((payload,), Some(&kwargs))?
            .extract()?;
        Ok(PyBytes::new(py, text.as_bytes()).unbind())
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
        PreparedMolData, PreparedMolFragmentData, PreparedMolWriterFlags,
        PREPARED_MOL_SCHEMA_VERSION,
    };
    use crate::prepared_graph::{
        PreparedSmilesGraphData, CONNECTED_NONSTEREO_SURFACE,
        PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
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

    fn prepared_mol_with_fragments(
        fragments: Vec<PreparedMolFragmentData>,
    ) -> PreparedMolData {
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
    fn rejects_writer_flag_mismatch() {
        let mut graph = single_atom_graph();
        graph.writer_all_hs_explicit = true;
        let prepared = prepared_mol_with_fragments(vec![PreparedMolFragmentData {
            atom_indices: vec![0],
            prepared_graph: graph,
        }]);

        assert!(prepared.validate().is_err());
    }
}
