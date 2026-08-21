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
#[cfg(test)]
mod native_oracle;
mod native_solver;
mod nonstereo_writer;
mod persistent;
mod prepared;
mod solver;
mod traversal;
mod writer_state;

pub use domain::{Domain, DomainError, DOMAIN_VALUE_CAPACITY};
pub use ids::{AtomId, BondId, FactorId, VariableId};
pub use model::{
    BinaryRelationFactor, BondRole, ConstraintModel, ConstraintModelBuilder, ConstraintModelError,
    FactorDefinition, SpanningTreeEdge, SpanningTreeFactor, VariableDefinition,
};
pub use prepared::{
    AdjacentBond, PreparedBond, PreparedGraph, PreparedGraphBuilder, PreparedGraphError,
    PreparedMolecule,
};
