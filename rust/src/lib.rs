mod prepared_graph;
mod rooted_nonstereo;
mod rooted_stereo;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::prepared_graph::{
    mol_to_smiles_support, prepared_smiles_graph_schema_version, PyPreparedSmilesGraph,
};
use crate::rooted_nonstereo::{
    PyRootedConnectedNonStereoWalker, PyRootedConnectedNonStereoWalkerState,
};
use crate::rooted_stereo::{PyRootedConnectedStereoWalker, PyRootedConnectedStereoWalkerState};

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPreparedSmilesGraph>()?;
    m.add_class::<PyRootedConnectedNonStereoWalker>()?;
    m.add_class::<PyRootedConnectedNonStereoWalkerState>()?;
    m.add_class::<PyRootedConnectedStereoWalker>()?;
    m.add_class::<PyRootedConnectedStereoWalkerState>()?;
    m.add_function(wrap_pyfunction!(prepared_smiles_graph_schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(mol_to_smiles_support, m)?)?;
    Ok(())
}
