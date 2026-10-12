//! Private Python transport for the independent South Star 2 native walker.

use grimace_south_star2::{
    AtomFacts, AtomId, AtomToken, BondId, BondToken, ComponentOrder, DirectionalCarrier,
    DirectionalRelation, ElementSymbol, PreparedGraphBuilder, PreparedWriter, TetrahedralLigand,
    TextPrefix, Walker, WalkerFailure, WalkerFailureKind, WriterRegime,
};
use pyo3::exceptions::{PyAssertionError, PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt, PyString, PyTuple};

#[pyclass(name = "_SouthStar2PreparedWriter", module = "grimace._core", frozen)]
pub struct PySouthStar2PreparedWriter {
    prepared: PreparedWriter,
}

#[pymethods]
impl PySouthStar2PreparedWriter {
    #[new]
    #[pyo3(signature = (atoms, bonds, writer_regime, directional=Vec::new()))]
    fn new(
        py: Python<'_>,
        atoms: Vec<Py<PyAny>>,
        bonds: Vec<(u32, u32, String)>,
        writer_regime: &str,
        directional: Vec<Py<PyAny>>,
    ) -> PyResult<Self> {
        let regime = match writer_regime {
            "generic_any_order" => WriterRegime::Generic {
                component_order: ComponentOrder::AnyOrder,
            },
            "generic_input_order" => WriterRegime::Generic {
                component_order: ComponentOrder::InputOrder,
            },
            "rdkit_random_noncanonical" => WriterRegime::RdkitRandomNoncanonical,
            _ => {
                return Err(PyValueError::new_err(
                    "writer_regime must be 'generic_any_order', 'generic_input_order', or 'rdkit_random_noncanonical'",
                ));
            }
        };
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
        let atom_tokens = atoms
            .iter()
            .enumerate()
            .map(|(index, atom)| atom_token(atom.bind(py), index))
            .collect::<PyResult<Vec<_>>>()?;
        let relations = directional
            .iter()
            .enumerate()
            .map(|(index, fact)| directional_relation(fact.bind(py), index))
            .collect::<PyResult<Vec<_>>>()?;
        let prepared = PreparedWriter::new(graph.build(), atom_tokens, tokens, relations, regime)
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

fn directional_relation(fact: &Bound<'_, PyAny>, index: usize) -> PyResult<DirectionalRelation> {
    let values = fact.cast::<PyTuple>().map_err(|_| {
        PyValueError::new_err(format!(
            "Cannot prepare directional relation {index}: expected a six-item tuple"
        ))
    })?;
    if values.len() != 6 {
        return Err(PyValueError::new_err(format!(
            "Cannot prepare directional relation {index}: expected six items, got {}",
            values.len()
        )));
    }
    let mut ids = [0_u32; 5];
    for (position, id) in ids.iter_mut().enumerate() {
        let value = values.get_item(position)?;
        if !value.is_instance_of::<PyInt>() || value.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(format!(
                "Cannot prepare directional relation {index}: item {position} must be an unsigned 32-bit integer ID"
            )));
        }
        *id = value.extract::<u32>().map_err(|_| {
            PyValueError::new_err(format!(
                "Cannot prepare directional relation {index}: item {position} must be an unsigned 32-bit integer ID"
            ))
        })?;
    }
    let phase = values.get_item(5)?;
    if !phase.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(format!(
            "Cannot prepare directional relation {index}: item 5 must be a boolean phase"
        )));
    }
    let side_phase_xor = phase.extract::<bool>()?;
    Ok(DirectionalRelation {
        double_bond: BondId::new(ids[0]),
        left_endpoint: AtomId::new(ids[1]),
        left_carriers: Box::new([DirectionalCarrier {
            bond: BondId::new(ids[2]),
            side_flip: false,
        }]),
        right_endpoint: AtomId::new(ids[3]),
        right_carriers: Box::new([DirectionalCarrier {
            bond: BondId::new(ids[4]),
            side_flip: false,
        }]),
        side_phase_xor,
    })
}

fn atom_token(atom: &Bound<'_, PyAny>, atom_index: usize) -> PyResult<AtomToken> {
    let fact = atom.cast::<PyTuple>().map_err(|_| {
        PyValueError::new_err(format!(
            "Cannot prepare atom {atom_index}: expected an atom-fact tuple"
        ))
    })?;
    if fact.len() != 10 {
        return Err(PyValueError::new_err(format!(
            "Cannot prepare atom {atom_index}: expected ten atom-fact fields, got {}",
            fact.len()
        )));
    }
    let symbol = fact.get_item(0)?.extract::<String>().map_err(|_| {
        PyValueError::new_err(format!("Cannot prepare atom {atom_index}: element must be text"))
    })?;
    let element = ElementSymbol::new(&symbol)
        .map_err(|error| PyValueError::new_err(format!("Cannot prepare atom {atom_index}: {error}")))?;
    let aromatic_value = fact.get_item(1)?;
    if !aromatic_value.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(format!("Cannot prepare atom {atom_index}: aromatic flag must be boolean")));
    }
    let aromatic = aromatic_value.extract::<bool>()?;
    let isotope_value = fact.get_item(2)?;
    let isotope = if isotope_value.is_none() {
        None
    } else {
        Some(isotope_value.extract::<u32>().map_err(|_| PyValueError::new_err(format!(
            "Cannot prepare atom {atom_index}: isotope must be an unsigned integer or None"
        )))?)
    };
    let formal_charge = fact.get_item(3)?.extract::<i8>().map_err(|_| PyValueError::new_err(format!(
        "Cannot prepare atom {atom_index}: formal charge must be an 8-bit integer"
    )))?;
    let non_graph_hydrogens = fact.get_item(4)?.extract::<u8>().map_err(|_| PyValueError::new_err(format!(
        "Cannot prepare atom {atom_index}: non-graph hydrogen count must be an 8-bit integer"
    )))?;
    let atom_class = fact.get_item(5)?.extract::<u64>().map_err(|_| PyValueError::new_err(format!(
        "Cannot prepare atom {atom_index}: atom class must be an unsigned 64-bit integer"
    )))?;
    let class_property_present = fact.get_item(8)?;
    if !class_property_present.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(format!(
            "Cannot prepare atom {atom_index}: atom-class property presence must be boolean"
        )));
    }
    let source_context = grimace_south_star2::AtomSourceContext {
        atom_class_property_present: class_property_present.extract::<bool>()?,
    };
    let valence_value = fact.get_item(9)?;
    let aromatic_total_valence = if valence_value.is_none() {
        None
    } else {
        if valence_value.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(format!(
                "Cannot prepare atom {atom_index}: aromatic total valence must be an unsigned integer or None"
            )));
        }
        Some(valence_value.extract::<u32>().map_err(|_| PyValueError::new_err(format!(
            "Cannot prepare atom {atom_index}: aromatic total valence must be an unsigned integer or None"
        )))?)
    };
    let facts = AtomFacts::new(element, aromatic, isotope, formal_charge, non_graph_hydrogens, atom_class, aromatic_total_valence)
        .map_err(|error| PyValueError::new_err(format!("Cannot prepare atom {atom_index}: {error}")))?;
    let parity_context = fact.get_item(6)?;
    let at_for_even_parity = if parity_context.is_none() {
        None
    } else {
        if !parity_context.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(format!("Cannot prepare atom {atom_index}: tetrahedral parity context must be boolean or None")));
        }
        Some(parity_context.extract::<bool>()?)
    };
    let reference_order = fact.get_item(7)?;
    let references = reference_order.cast::<PyTuple>().map_err(|_| {
        PyValueError::new_err(format!(
            "Cannot prepare tetrahedral atom {atom_index}: reference order must be a tuple"
        ))
    })?;
    let mut ligands = Vec::with_capacity(references.len());
    for position in 0..references.len() {
        let reference = references.get_item(position)?;
        if reference.is_instance_of::<PyInt>() && !reference.is_instance_of::<PyBool>() {
            let bond = reference.extract::<u32>().map_err(|_| {
                PyValueError::new_err(format!(
                    "Cannot prepare tetrahedral atom {atom_index}: reference at position \
                     {position} must be an unsigned 32-bit bond ID"
                ))
            })?;
            ligands.push(TetrahedralLigand::Bond(BondId::new(bond)));
            continue;
        }
        if reference.is_instance_of::<PyString>() {
            let ligand = reference.extract::<String>()?;
            if ligand == "virtual_hydrogen" {
                ligands.push(TetrahedralLigand::VirtualHydrogen);
                continue;
            }
        }
        return Err(PyValueError::new_err(format!(
            "Cannot prepare tetrahedral atom {atom_index}: reference {position} \
             must be a bond ID or VirtualHydrogen"
        )));
    }
    let tetrahedral = if let Some(at_for_even_parity) = at_for_even_parity {
        let reference_order = ligands.try_into().map_err(|ligands: Vec<TetrahedralLigand>| {
            PyValueError::new_err(format!(
                "Cannot prepare tetrahedral atom {atom_index}: expected four ligand references, got {}",
                ligands.len()
            ))
        })?;
        Some((reference_order, at_for_even_parity))
    } else if ligands.is_empty() {
        None
    } else {
        return Err(PyValueError::new_err(format!(
            "Cannot prepare atom {atom_index}: reference ligands require tetrahedral parity context"
        )));
    };
    let (tetrahedral_reference_order, at_for_even_parity) = match tetrahedral {
        Some((reference_order, flip)) => (Some(reference_order), flip),
        None => (None, false),
    };
    Ok(AtomToken::Facts { facts, source_context, tetrahedral_reference_order, at_for_even_parity })
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
