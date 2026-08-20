"""Typed contracts used by the default South Star writer ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.ordinary_stereo_sites import OrdinaryStereoSiteOptions
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions

QualificationAuthority = Literal[
    "materialized_support_artifact",
    "continuation_proof_complete",
]
ExtractionContractId = Literal[
    "ordinary_graph",
    "ordinary_potential_stereo",
    "ordinary_specified_stereo",
    "ordinary_coupled_tetrahedral_stereo",
]
ExtractionProfile = Literal[
    "graph_no_potential_sites",
    "specified_stereo_closure",
    "with_potential_sites",
]
RdkitAuditFamily = Literal["aromatic", "bracket", "disconnected", "stereo"]
AcceptedEvidenceProfileId = Literal["default", "disconnected"]


@dataclass(frozen=True, slots=True)
class ExtractionContractDefinition:
    contract_id: ExtractionContractId
    public_profile: ExtractionProfile
    options: RdkitOrdinaryExtractionOptions
    rdkit_audit_families: tuple[RdkitAuditFamily, ...] = ()


EXTRACTION_CONTRACT_DEFINITIONS = (
    ExtractionContractDefinition(
        "ordinary_graph",
        "graph_no_potential_sites",
        RdkitOrdinaryExtractionOptions(include_potential_sites=False),
    ),
    ExtractionContractDefinition(
        "ordinary_potential_stereo",
        "with_potential_sites",
        RdkitOrdinaryExtractionOptions(include_potential_sites=True),
    ),
    ExtractionContractDefinition(
        "ordinary_specified_stereo",
        "specified_stereo_closure",
        RdkitOrdinaryExtractionOptions(
            include_potential_sites=True,
            stereo_site_discovery_mode="specified_closure",
        ),
        ("stereo",),
    ),
    ExtractionContractDefinition(
        "ordinary_coupled_tetrahedral_stereo",
        "specified_stereo_closure",
        RdkitOrdinaryExtractionOptions(
            include_potential_sites=True,
            stereo_site_discovery_mode="specified_closure",
            stereo_site_options=OrdinaryStereoSiteOptions(
                ligand_equivalence="exact_stereochemical_graph_automorphism",
            ),
        ),
        ("stereo",),
    ),
)
_EXTRACTION_BY_ID = {item.contract_id: item for item in EXTRACTION_CONTRACT_DEFINITIONS}

DEFAULT_OFFLINE_RELATION_FAMILIES = (
    "branch_projection_identity", "count_dag_arithmetic", "graph_ring_branch_delta",
    "local_branch_successor_evidence", "residual_stereo_obligation_classification",
    "support_image_coverage", "support_string_replay_path", "terminal_support_identity",
)
DISCONNECTED_OFFLINE_RELATION_FAMILIES = (*DEFAULT_OFFLINE_RELATION_FAMILIES, "component_boundary_transition")
DEFAULT_OFFLINE_OBJECT_KINDS = (
    "branch_support", "count_dag", "count_envelope", "frontier_product", "replay_path",
    "source_snapshot", "support_image", "support_image_coverage", "support_string",
    "terminal_projection", "terminal_support", "text_projection",
)
DEFAULT_OFFLINE_UNCHECKED_OBJECT_KINDS: tuple[str, ...] = ()
DEFAULT_OFFLINE_UNCHECKED_OBLIGATION_FAMILIES: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AcceptedEvidenceExpectation:
    structural_artifact: bool
    live_artifact_verifier: bool
    facts_bound_verifier: bool
    offline_replay_complete: bool
    live_frontier_agreement_complete: bool
    live_count_agreement_complete: bool
    snapshot_resume_agreement_complete: bool
    continuation_asset_complete: bool
    rust_runtime_agreement_complete: bool
    continuation_snapshot_resume_complete: bool
    lazy_branch_proof_complete: bool
    lazy_terminal_proof_complete: bool
    offline_object_kinds: tuple[str, ...]
    offline_unchecked_object_kinds: tuple[str, ...]
    offline_relation_families: tuple[str, ...]
    offline_unchecked_obligation_families: tuple[str, ...]


def _evidence(relations: tuple[str, ...]) -> AcceptedEvidenceExpectation:
    return AcceptedEvidenceExpectation(
        True, True, True, True, True, True, True, True, True, True, True, True,
        DEFAULT_OFFLINE_OBJECT_KINDS, DEFAULT_OFFLINE_UNCHECKED_OBJECT_KINDS,
        relations, DEFAULT_OFFLINE_UNCHECKED_OBLIGATION_FAMILIES,
    )


ACCEPTED_EVIDENCE_PROFILES: Mapping[AcceptedEvidenceProfileId, AcceptedEvidenceExpectation] = {
    "default": _evidence(DEFAULT_OFFLINE_RELATION_FAMILIES),
    "disconnected": _evidence(DISCONNECTED_OFFLINE_RELATION_FAMILIES),
}


@dataclass(frozen=True, slots=True)
class SupportSurfaceDefinition:
    name: str
    expected_disposition: Literal["accepted", "blocked"]
    accepted_evidence_profile: AcceptedEvidenceProfileId | None
    rdkit_audit_families: tuple[RdkitAuditFamily, ...] = ()


_BRACKET_SURFACES = {
    "simple_bracket_charge", "simple_isotope_bracket_atom", "unsupported_charged_isotope",
    "unsupported_charged_oxygen_isotope", "unsupported_negative_nitrogen_charge",
    "unsupported_positive_oxygen_charge",
}
_AROMATIC_SURFACES = {
    "aromatic_homocycle", "aromatic_heterocycle", "fused_aromatic", "aromatic_substitution",
    "aromatic_single_bridge",
}
_BLOCKED_SURFACES = {
    "unsupported_charged_isotope", "unsupported_charged_oxygen_isotope",
    "unsupported_negative_nitrogen_charge", "unsupported_positive_oxygen_charge",
    "unsupported_potential_directional_non_neighbor", "unsupported_aromatic_bracketed_hydrogen",
    "unsupported_aromatic_boron", "unsupported_aromatic_phosphorus", "unsupported_aromatic_atom_map",
}
_SURFACE_NAMES = (
    "acyclic_graph", "branched_graph", "single_ring_closure", "non_single_ring_closure_double",
    "non_single_ring_closure_triple", "branched_ring", "simple_bracket_charge",
    "simple_isotope_bracket_atom", "specified_tetrahedral", "specified_tetrahedral_zero_h",
    "specified_tetrahedral_adjacent", "specified_tetrahedral_coupled",
    "specified_directional_acyclic_implicit_h", "disconnected_fixed_order", "unsupported_charged_isotope",
    "unsupported_charged_oxygen_isotope", "unsupported_negative_nitrogen_charge",
    "unsupported_positive_oxygen_charge", "unsupported_potential_directional_non_neighbor",
    "aromatic_homocycle", "aromatic_heterocycle", "fused_aromatic", "aromatic_substitution",
    "aromatic_single_bridge", "disconnected_aromatic", "unsupported_aromatic_bracketed_hydrogen",
    "unsupported_aromatic_boron", "unsupported_aromatic_phosphorus", "unsupported_aromatic_atom_map",
)


def _surface(name: str) -> SupportSurfaceDefinition:
    blocked = name in _BLOCKED_SURFACES
    families: list[RdkitAuditFamily] = []
    if name in _BRACKET_SURFACES:
        families.append("bracket")
    if name in _AROMATIC_SURFACES:
        families.append("aromatic")
    if name == "disconnected_fixed_order":
        families.append("disconnected")
    if name == "disconnected_aromatic":
        families.extend(("disconnected", "aromatic"))
    return SupportSurfaceDefinition(
        name, "blocked" if blocked else "accepted",
        None if blocked else ("disconnected" if name in {"disconnected_fixed_order", "disconnected_aromatic"} else "default"),
        tuple(families),
    )


SUPPORT_SURFACE_DEFINITIONS = tuple(_surface(name) for name in _SURFACE_NAMES)
_SURFACES_BY_NAME = {item.name: item for item in SUPPORT_SURFACE_DEFINITIONS}


@dataclass(frozen=True, slots=True)
class ContinuationProofExpectation:
    raw_cursor_count: int
    edge_locator_count: int
    branch_locator_count: int
    terminal_record_count: int
    terminal_locator_count: int
    replayed_operations: tuple[str, ...]
    checked_relation_families: tuple[str, ...]
    checked_obligation_families: tuple[str, ...]
    unchecked_obligation_families: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AcceptedCaseExpectation:
    support_count: int
    completion_count: int
    support_digest: str
    qualification_authority: QualificationAuthority
    continuation: ContinuationProofExpectation | None = None


@dataclass(frozen=True, slots=True)
class PreparationBlockerExpectation:
    kind: str
    error_kind: SouthStarErrorKind
    message_contains: str


@dataclass(frozen=True, slots=True)
class FrontierBlockerExpectation:
    kind: str
    operation: str


CaseExpectation = AcceptedCaseExpectation | PreparationBlockerExpectation | FrontierBlockerExpectation


@dataclass(frozen=True, slots=True)
class DefaultWriterCapabilityCase:
    name: str
    smiles: str
    extraction_contract_id: ExtractionContractId
    support_surface: str
    expectation: CaseExpectation
    rooted_at_atom: int = 0

    @property
    def extraction_profile(self) -> ExtractionProfile:
        return _EXTRACTION_BY_ID[self.extraction_contract_id].public_profile

    @property
    def extraction_options(self) -> RdkitOrdinaryExtractionOptions:
        return _EXTRACTION_BY_ID[self.extraction_contract_id].options

    @property
    def expected(self) -> Literal["accepted", "blocked"]:
        return "accepted" if isinstance(self.expectation, AcceptedCaseExpectation) else "blocked"

    @property
    def qualification_authority(self) -> QualificationAuthority | None:
        return self.expectation.qualification_authority if isinstance(self.expectation, AcceptedCaseExpectation) else None

    @property
    def expected_support_count(self) -> int | None:
        return self.expectation.support_count if isinstance(self.expectation, AcceptedCaseExpectation) else None

    @property
    def expected_completion_count(self) -> int | None:
        return self.expectation.completion_count if isinstance(self.expectation, AcceptedCaseExpectation) else None

    @property
    def expected_support_digest(self) -> str | None:
        return self.expectation.support_digest if isinstance(self.expectation, AcceptedCaseExpectation) else None

    @property
    def blocker_phase(self) -> Literal["preparation", "frontier"] | None:
        if isinstance(self.expectation, PreparationBlockerExpectation): return "preparation"
        if isinstance(self.expectation, FrontierBlockerExpectation): return "frontier"
        return None

    @property
    def blocker_kind(self) -> str | None:
        return self.expectation.kind if isinstance(self.expectation, (PreparationBlockerExpectation, FrontierBlockerExpectation)) else None

    @property
    def blocker_operation(self) -> str | None:
        return self.expectation.operation if isinstance(self.expectation, FrontierBlockerExpectation) else None

    @property
    def blocker_error_kind(self) -> SouthStarErrorKind | None:
        return self.expectation.error_kind if isinstance(self.expectation, PreparationBlockerExpectation) else None

    @property
    def blocker_message_contains(self) -> str | None:
        return self.expectation.message_contains if isinstance(self.expectation, PreparationBlockerExpectation) else None

    @property
    def accepted_evidence(self) -> AcceptedEvidenceExpectation | None:
        if not isinstance(self.expectation, AcceptedCaseExpectation): return None
        return ACCEPTED_EVIDENCE_PROFILES[_SURFACES_BY_NAME[self.support_surface].accepted_evidence_profile or "default"]

    def _evidence_value(self, name: str, default):
        evidence = self.accepted_evidence
        return getattr(evidence, name) if evidence is not None else default

    @property
    def expected_structural_artifact(self) -> bool: return self._evidence_value("structural_artifact", False)
    @property
    def expected_live_artifact_verifier(self) -> bool: return self._evidence_value("live_artifact_verifier", False)
    @property
    def expected_facts_bound_verifier(self) -> bool: return self._evidence_value("facts_bound_verifier", False)
    @property
    def expected_offline_replay_complete(self) -> bool: return self._evidence_value("offline_replay_complete", False)
    @property
    def expected_live_frontier_agreement_complete(self) -> bool: return self._evidence_value("live_frontier_agreement_complete", False)
    @property
    def expected_live_count_agreement_complete(self) -> bool: return self._evidence_value("live_count_agreement_complete", False)
    @property
    def expected_snapshot_resume_agreement_complete(self) -> bool: return self._evidence_value("snapshot_resume_agreement_complete", False)
    @property
    def expected_continuation_asset_complete(self) -> bool: return self._evidence_value("continuation_asset_complete", False)
    @property
    def expected_rust_runtime_agreement_complete(self) -> bool: return self._evidence_value("rust_runtime_agreement_complete", False)
    @property
    def expected_continuation_snapshot_resume_complete(self) -> bool: return self._evidence_value("continuation_snapshot_resume_complete", False)
    @property
    def expected_lazy_branch_proof_complete(self) -> bool: return self._evidence_value("lazy_branch_proof_complete", False)
    @property
    def expected_lazy_terminal_proof_complete(self) -> bool: return self._evidence_value("lazy_terminal_proof_complete", False)
    @property
    def expected_offline_object_kinds(self) -> tuple[str, ...]: return self._evidence_value("offline_object_kinds", ())
    @property
    def expected_offline_unchecked_object_kinds(self) -> tuple[str, ...]: return self._evidence_value("offline_unchecked_object_kinds", ())
    @property
    def expected_offline_relation_families(self) -> tuple[str, ...]: return self._evidence_value("offline_relation_families", ())
    @property
    def expected_offline_unchecked_obligation_families(self) -> tuple[str, ...]: return self._evidence_value("offline_unchecked_obligation_families", ())

    @property
    def expected_rdkit_audit_version_pinned(self) -> bool:
        return bool(self.rdkit_audit_families)

    @property
    def rdkit_audit_families(self) -> tuple[RdkitAuditFamily, ...]:
        contract = _EXTRACTION_BY_ID.get(self.extraction_contract_id)
        families = list(contract.rdkit_audit_families if contract else ())
        for family in _SURFACES_BY_NAME[self.support_surface].rdkit_audit_families:
            if family not in families:
                families.append(family)
        return tuple(families)

    def _continuation_value(self, name: str, default=None):
        continuation = self.expectation.continuation if isinstance(self.expectation, AcceptedCaseExpectation) else None
        return getattr(continuation, name) if continuation is not None else default

    @property
    def expected_continuation_raw_cursor_count(self): return self._continuation_value("raw_cursor_count")
    @property
    def expected_continuation_edge_locator_count(self): return self._continuation_value("edge_locator_count")
    @property
    def expected_continuation_branch_locator_count(self): return self._continuation_value("branch_locator_count")
    @property
    def expected_continuation_terminal_record_count(self): return self._continuation_value("terminal_record_count")
    @property
    def expected_continuation_terminal_locator_count(self): return self._continuation_value("terminal_locator_count")
    @property
    def expected_continuation_replayed_operations(self): return self._continuation_value("replayed_operations", ())
    @property
    def expected_continuation_checked_relation_families(self): return self._continuation_value("checked_relation_families", ())
    @property
    def expected_continuation_checked_obligation_families(self): return self._continuation_value("checked_obligation_families", ())
    @property
    def expected_continuation_unchecked_obligation_families(self): return self._continuation_value("unchecked_obligation_families", ())


def _contract_id_for_options(options: RdkitOrdinaryExtractionOptions) -> ExtractionContractId:
    for item in EXTRACTION_CONTRACT_DEFINITIONS:
        if item.options == options:
            return item.contract_id
    raise ValueError("options do not identify a registered extraction contract")


def _validate_common(name: str, smiles: str, extraction_contract_id: ExtractionContractId, support_surface: str) -> None:
    if extraction_contract_id not in _EXTRACTION_BY_ID:
        raise ValueError(f"unknown extraction contract: {extraction_contract_id!r}")
    if support_surface not in _SURFACES_BY_NAME:
        raise ValueError(f"unknown support surface: {support_surface!r}")
    if not name or not smiles:
        raise ValueError("case name and SMILES are required")


def accepted_writer_case(*, name: str, smiles: str, extraction_contract_id: ExtractionContractId, support_surface: str, support_count: int, completion_count: int, support_digest: str, qualification_authority: QualificationAuthority = "materialized_support_artifact", continuation: ContinuationProofExpectation | None = None, rooted_at_atom: int = 0) -> DefaultWriterCapabilityCase:
    _validate_common(name, smiles, extraction_contract_id, support_surface)
    if support_count <= 0 or completion_count <= 0 or len(support_digest) != 64:
        raise ValueError("accepted cases require positive counts and a 64-character digest")
    if qualification_authority == "continuation_proof_complete" and continuation is None:
        raise ValueError("continuation-qualified cases require continuation expectations")
    if qualification_authority == "materialized_support_artifact" and continuation is not None:
        raise ValueError("materialized cases cannot carry continuation expectations")
    evidence = ACCEPTED_EVIDENCE_PROFILES[_SURFACES_BY_NAME[support_surface].accepted_evidence_profile or "default"]
    contract = _EXTRACTION_BY_ID[extraction_contract_id]
    expectation = AcceptedCaseExpectation(
        support_count, completion_count, support_digest,
        qualification_authority, continuation,
    )
    return DefaultWriterCapabilityCase(
        name, smiles, extraction_contract_id, support_surface, expectation, rooted_at_atom
    )


def preparation_blocked_writer_case(*, name: str, smiles: str, extraction_contract_id: ExtractionContractId, support_surface: str, kind: str, error_kind: SouthStarErrorKind, message_contains: str, rooted_at_atom: int = 0) -> DefaultWriterCapabilityCase:
    _validate_common(name, smiles, extraction_contract_id, support_surface)
    if _SURFACES_BY_NAME[support_surface].expected_disposition != "blocked":
        raise ValueError("preparation blockers require blocked support surfaces")
    contract = _EXTRACTION_BY_ID[extraction_contract_id]
    return DefaultWriterCapabilityCase(
        name, smiles, extraction_contract_id, support_surface,
        PreparationBlockerExpectation(kind, error_kind, message_contains), rooted_at_atom,
    )


def frontier_blocked_writer_case(*, name: str, smiles: str, extraction_contract_id: ExtractionContractId, support_surface: str, kind: str, operation: str, rooted_at_atom: int = 0) -> DefaultWriterCapabilityCase:
    _validate_common(name, smiles, extraction_contract_id, support_surface)
    if _SURFACES_BY_NAME[support_surface].expected_disposition != "blocked":
        raise ValueError("frontier blockers require blocked support surfaces")
    contract = _EXTRACTION_BY_ID[extraction_contract_id]
    return DefaultWriterCapabilityCase(
        name, smiles, extraction_contract_id, support_surface,
        FrontierBlockerExpectation(kind, operation), rooted_at_atom,
    )


def validate_default_writer_capability_contracts() -> None:
    if len({item.contract_id for item in EXTRACTION_CONTRACT_DEFINITIONS}) != len(EXTRACTION_CONTRACT_DEFINITIONS):
        raise ValueError("duplicate extraction contract")
    if len({item.name for item in SUPPORT_SURFACE_DEFINITIONS}) != len(SUPPORT_SURFACE_DEFINITIONS):
        raise ValueError("duplicate support surface")
    known = {"aromatic", "bracket", "disconnected", "stereo"}
    for surface in SUPPORT_SURFACE_DEFINITIONS:
        if surface.expected_disposition == "accepted" and surface.accepted_evidence_profile not in ACCEPTED_EVIDENCE_PROFILES:
            raise ValueError("accepted surface lacks evidence profile")
        if surface.expected_disposition == "blocked" and surface.accepted_evidence_profile is not None:
            raise ValueError("blocked surface has evidence profile")
        if not surface.name or any(not family or family not in known for family in surface.rdkit_audit_families):
            raise ValueError("invalid audit family")
        if len(set(surface.rdkit_audit_families)) != len(surface.rdkit_audit_families):
            raise ValueError("duplicate audit family")


__all__ = tuple(name for name in globals() if not name.startswith("_") or name in {"_EXTRACTION_BY_ID"})
