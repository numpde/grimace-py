#![forbid(unsafe_code)]

//! South Star 2: a pure-Rust incremental constraint-driven SMILES walker.
//!
//! This crate intentionally starts below the Python and RDKit boundaries.
//! Provisional implementation stages remain private rather than becoming
//! capability labels in the semantic API.

mod aromatic_matching;
mod atom_spelling;
mod domain;
mod event_schedule;
mod ids;
mod model;
mod native;
#[cfg(test)]
mod native_oracle;
mod native_solver;
mod nonstereo_writer;
mod persistent;
mod prefix;
mod prepared;
mod solver;
mod tetrahedral;
mod traversal;
mod walker;
mod writer_state;

pub use atom_spelling::{
    AtomBondContext, AtomFacts, AtomFactsError, AtomSpellingChoice, AtomSpellingContext,
    AtomSpellingField, AtomSpellingState, ElementSymbol, TetrahedralMarker,
};
pub use domain::{Domain, DomainError, DOMAIN_VALUE_CAPACITY};
pub use ids::{AtomId, BondId, FactorId, VariableId};
pub use model::{
    AromaticMatchingEdge, AromaticMatchingFactor, BinaryRelationFactor, BondRole, ConstraintModel,
    ConstraintModelBuilder, ConstraintModelError, DirectionalRingPlacementFactor,
    EdgeRolePartition, FactorDefinition, SpanningTreeEdge, TraversalTopologyFactor,
    TraversalTopologyRequirement, VariableDefinition,
};
pub use nonstereo_writer::{
    AtomSourceContext, NonStereoBondToken as BondToken, PreparedAtomToken as AtomToken,
    PreparedDirectionalCarrier as DirectionalCarrier,
    PreparedDirectionalRelation as DirectionalRelation,
};
pub use prefix::{PrefixSelectionError, TextPrefix, TextPrefixChoice};
pub use prepared::{
    AdjacentBond, PreparedBond, PreparedGraph, PreparedGraphBuilder, PreparedGraphError,
    PreparedMolecule,
};
pub use tetrahedral::TetrahedralLigand;
pub use walker::{
    ComponentOrder, PreparedWriter, Walker, WalkerChoice, WalkerFailure, WalkerFailureKind,
    WriterRegime,
};
