//! Private Python transport for the independent South Star 2 native walker.

use grimace_south_star2::{
    AtomId, AtomToken, BondToken, ComponentOrder, PreparedGraphBuilder, PreparedWriter, TextPrefix,
    Walker, WalkerFailure, WalkerFailureKind,
};
use pyo3::exceptions::{PyAssertionError, PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyclass(name = "_SouthStar2PreparedWriter", module = "grimace._core", frozen)]
pub struct PySouthStar2PreparedWriter {
    prepared: PreparedWriter,
}

#[pymethods]
impl PySouthStar2PreparedWriter {
    #[new]
    fn new(atoms: Vec<String>, bonds: Vec<(u32, u32, String)>) -> PyResult<Self> {
        let mut graph = PreparedGraphBuilder::new();
        for _ in &atoms {
            graph
                .add_atom()
                .map_err(|error| PyValueError::new_err(format!("Cannot prepare atoms: {error}")))?;
        }
        let mut tokens = Vec::with_capacity(bonds.len());
        for (index, (a, b, spelling)) in bonds.into_iter().enumerate() {
            graph
                .add_bond(AtomId::new(a), AtomId::new(b))
                .map_err(|error| {
                    PyValueError::new_err(format!("Cannot prepare bond {index}: {error}"))
                })?;
            tokens.push(bond_token(index, &spelling)?);
        }
        let prepared = PreparedWriter::new(
            graph.build(),
            atoms.into_iter().map(AtomToken::Fixed).collect(),
            tokens,
            Vec::new(),
            ComponentOrder::AnyOrder,
        )
        .map_err(python_failure)?;
        Ok(Self { prepared })
    }

    fn initial_state(&self) -> PyResult<PySouthStar2Walker> {
        self.prepared
            .initial()
            .map(|state| PySouthStar2Walker { state })
            .map_err(python_failure)
    }

    fn initial_prefix(&self) -> PyResult<PySouthStar2TextPrefix> {
        TextPrefix::initial(&self.prepared)
            .map(|prefix| PySouthStar2TextPrefix { prefix })
            .map_err(python_failure)
    }
}

#[pyclass(name = "_SouthStar2TextPrefix", module = "grimace._core", frozen)]
pub struct PySouthStar2TextPrefix {
    prefix: TextPrefix,
}

#[pymethods]
impl PySouthStar2TextPrefix {
    #[getter]
    fn is_accepted(&self) -> bool {
        self.prefix.is_accepted()
    }

    #[getter]
    fn frontier_size(&self) -> usize {
        self.prefix.frontier_size()
    }

    fn choices(&self) -> PyResult<Vec<(String, Self)>> {
        self.prefix
            .choices()
            .map(|choices| {
                choices
                    .into_iter()
                    .map(|choice| {
                        (
                            choice.text,
                            Self {
                                prefix: choice.successor,
                            },
                        )
                    })
                    .collect()
            })
            .map_err(python_failure)
    }
}

#[pyclass(name = "_SouthStar2Walker", module = "grimace._core", frozen)]
pub struct PySouthStar2Walker {
    state: Walker,
}

#[pymethods]
impl PySouthStar2Walker {
    #[getter]
    fn is_accepted(&self) -> bool {
        self.state.is_accepted()
    }

    fn choices(&self) -> PyResult<Vec<(String, Self)>> {
        self.state
            .choices()
            .map(|choices| {
                choices
                    .into_iter()
                    .map(|choice| {
                        (
                            choice.text,
                            Self {
                                state: choice.successor,
                            },
                        )
                    })
                    .collect()
            })
            .map_err(python_failure)
    }
}

fn bond_token(index: usize, spelling: &str) -> PyResult<BondToken> {
    match spelling {
        "" => Ok(BondToken::Elided),
        "-" => Ok(BondToken::Single),
        ":" => Ok(BondToken::Aromatic),
        "=" => Ok(BondToken::Double),
        "#" => Ok(BondToken::Triple),
        "->" => Ok(BondToken::DativeAToB),
        "<-" => Ok(BondToken::DativeBToA),
        _ => Err(PyValueError::new_err(format!(
            "Cannot prepare bond {index} with spelling {spelling:?}; use an elided, single, \
             aromatic, double, triple, or directed dative bond token"
        ))),
    }
}

fn python_failure(error: WalkerFailure) -> PyErr {
    match error.kind() {
        WalkerFailureKind::InvalidInput | WalkerFailureKind::Spelling => {
            PyValueError::new_err(error.to_string())
        }
        WalkerFailureKind::Backend => PyRuntimeError::new_err(error.to_string()),
        WalkerFailureKind::Incomplete => PyNotImplementedError::new_err(error.to_string()),
        WalkerFailureKind::Invariant => PyAssertionError::new_err(error.to_string()),
    }
}
