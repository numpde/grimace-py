mod bond_stereo_constraints;
mod frontier;
mod prepared_graph;
mod prepared_mol;
// Promote to an unconditional module when the runtime walk consumes it.
#[cfg(test)]
mod rng;
mod rooted_nonstereo;
mod rooted_stereo;
mod smiles_shared;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::prepared_graph::{
    mol_to_smiles_support, prepared_smiles_graph_schema_version, PyPreparedSmilesGraph,
};
use crate::prepared_mol::PyPreparedMol;
use crate::rooted_nonstereo::{
    PyRootedConnectedNonStereoDecoder, PyRootedConnectedNonStereoWalker,
    PyRootedConnectedNonStereoWalkerState,
};
use crate::rooted_stereo::{
    PyRootedConnectedStereoDecoder, PyRootedConnectedStereoWalker,
    PyRootedConnectedStereoWalkerState,
};

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPreparedSmilesGraph>()?;
    m.add_class::<PyPreparedMol>()?;
    m.add_class::<PyRootedConnectedNonStereoDecoder>()?;
    m.add_class::<PyRootedConnectedNonStereoWalker>()?;
    m.add_class::<PyRootedConnectedNonStereoWalkerState>()?;
    m.add_class::<PyRootedConnectedStereoDecoder>()?;
    m.add_class::<PyRootedConnectedStereoWalker>()?;
    m.add_class::<PyRootedConnectedStereoWalkerState>()?;
    m.add_function(wrap_pyfunction!(prepared_smiles_graph_schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(mol_to_smiles_support, m)?)?;
    Ok(())
}
