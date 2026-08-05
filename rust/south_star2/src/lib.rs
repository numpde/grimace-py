#![forbid(unsafe_code)]

//! South Star 2: a pure-Rust incremental constraint-driven SMILES walker.
//!
//! This crate intentionally starts below the Python and RDKit boundaries.
//! Provisional implementation stages remain private rather than becoming
//! capability labels in the semantic API.

mod domain;
mod ids;
mod model;
mod native;
mod prepared;

pub use domain::{Domain, DomainError, DOMAIN_VALUE_CAPACITY};
pub use ids::{AtomId, BondId, FactorId, TokenId, VariableId};
pub use model::{
    BinaryRelationFactor, ConstraintModel, ConstraintModelBuilder, ConstraintModelError,
    FactorDefinition, VariableDefinition,
};
pub use prepared::{
    AdjacentBond, PreparedAtom, PreparedBond, PreparedGraph, PreparedGraphBuilder,
    PreparedGraphError, PreparedMolecule,
};
