"""Declared default ordinary writer capability ledger for South Star tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions


_GRAPH_EXTRACTION = RdkitOrdinaryExtractionOptions(include_potential_sites=False)
_POTENTIAL_STEREO_EXTRACTION = RdkitOrdinaryExtractionOptions(
    include_potential_sites=True,
)


@dataclass(frozen=True, slots=True)
class DefaultWriterCapabilityCase:
    name: str
    smiles: str
    extraction_profile: Literal[
        "graph_no_potential_sites",
        "with_potential_sites",
    ]
    extraction_options: RdkitOrdinaryExtractionOptions
    expected: Literal["accepted", "blocked"]
    support_surface: str
    expected_support_count: int | None = None
    expected_completion_count: int | None = None
    blocker_phase: Literal["preparation", "frontier"] | None = None
    blocker_kind: str | None = None
    blocker_operation: str | None = None
    blocker_error_kind: SouthStarErrorKind | None = None


DEFAULT_WRITER_CAPABILITY_CASES = (
    DefaultWriterCapabilityCase(
        name="ethanol",
        smiles="CCO",
        extraction_profile="graph_no_potential_sites",
        extraction_options=_GRAPH_EXTRACTION,
        expected="accepted",
        support_surface="acyclic_graph",
        expected_support_count=1,
        expected_completion_count=1,
    ),
    DefaultWriterCapabilityCase(
        name="branched_alcohol",
        smiles="CC(C)O",
        extraction_profile="graph_no_potential_sites",
        extraction_options=_GRAPH_EXTRACTION,
        expected="accepted",
        support_surface="branched_graph",
        expected_support_count=2,
        expected_completion_count=2,
    ),
    DefaultWriterCapabilityCase(
        name="cyclopropane",
        smiles="C1CC1",
        extraction_profile="graph_no_potential_sites",
        extraction_options=_GRAPH_EXTRACTION,
        expected="accepted",
        support_surface="single_ring_closure",
        expected_support_count=1,
        expected_completion_count=2,
    ),
    DefaultWriterCapabilityCase(
        name="cyclobutane",
        smiles="C1CCC1",
        extraction_profile="graph_no_potential_sites",
        extraction_options=_GRAPH_EXTRACTION,
        expected="accepted",
        support_surface="single_ring_closure",
        expected_support_count=1,
        expected_completion_count=2,
    ),
    DefaultWriterCapabilityCase(
        name="cyclopropene_double_closure",
        smiles="C1=CC1",
        extraction_profile="graph_no_potential_sites",
        extraction_options=_GRAPH_EXTRACTION,
        expected="accepted",
        support_surface="non_single_ring_closure_double",
        expected_support_count=3,
        expected_completion_count=3,
    ),
    DefaultWriterCapabilityCase(
        name="cyclopropyne_triple_closure",
        smiles="C1#CC1",
        extraction_profile="graph_no_potential_sites",
        extraction_options=_GRAPH_EXTRACTION,
        expected="accepted",
        support_surface="non_single_ring_closure_triple",
        expected_support_count=3,
        expected_completion_count=3,
    ),
    DefaultWriterCapabilityCase(
        name="branched_cyclobutane",
        smiles="C1CC(C)C1",
        extraction_profile="graph_no_potential_sites",
        extraction_options=_GRAPH_EXTRACTION,
        expected="accepted",
        support_surface="branched_ring",
        expected_support_count=2,
        expected_completion_count=4,
    ),
    DefaultWriterCapabilityCase(
        name="ammonium_charge",
        smiles="[NH4+]",
        extraction_profile="graph_no_potential_sites",
        extraction_options=_GRAPH_EXTRACTION,
        expected="accepted",
        support_surface="simple_bracket_charge",
        expected_support_count=1,
        expected_completion_count=1,
    ),
    DefaultWriterCapabilityCase(
        name="isotopic_methane",
        smiles="[13CH4]",
        extraction_profile="graph_no_potential_sites",
        extraction_options=_GRAPH_EXTRACTION,
        expected="blocked",
        support_surface="unsupported_isotope",
        blocker_phase="preparation",
        blocker_kind="unsupported_atom",
        blocker_error_kind=SouthStarErrorKind.UNSUPPORTED_ATOM,
    ),
    DefaultWriterCapabilityCase(
        name="cyclopropene_potential_directional_boundary",
        smiles="C1=CC1",
        extraction_profile="with_potential_sites",
        extraction_options=_POTENTIAL_STEREO_EXTRACTION,
        expected="blocked",
        support_surface="unsupported_potential_directional_non_neighbor",
        blocker_phase="frontier",
        blocker_kind="unsupported_directional_non_neighbor_ligand",
        blocker_operation="directional carrier-mark restriction",
    ),
)


ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES = tuple(
    item for item in DEFAULT_WRITER_CAPABILITY_CASES if item.expected == "accepted"
)
BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES = tuple(
    item for item in DEFAULT_WRITER_CAPABILITY_CASES if item.expected == "blocked"
)


__all__ = (
    "ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES",
    "BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES",
    "DEFAULT_WRITER_CAPABILITY_CASES",
    "DefaultWriterCapabilityCase",
)
