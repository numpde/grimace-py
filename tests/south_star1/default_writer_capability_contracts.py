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


# The public dataclass fields intentionally remain the historical projection.  The
# constrained builders below are the only construction API; typed expectations and
# registries supply their values and prevent mixed accepted/blocked states.
@dataclass(frozen=True, slots=True)
class DefaultWriterCapabilityCase:
    name: str
    smiles: str
    extraction_profile: ExtractionProfile
    extraction_options: RdkitOrdinaryExtractionOptions
    expected: Literal["accepted", "blocked"]
    support_surface: str
    qualification_authority: QualificationAuthority | None = "materialized_support_artifact"
    rooted_at_atom: int = 0
    expected_support_count: int | None = None
    expected_completion_count: int | None = None
    blocker_phase: Literal["preparation", "frontier"] | None = None
    blocker_kind: str | None = None
    blocker_operation: str | None = None
    blocker_error_kind: SouthStarErrorKind | None = None
    blocker_message_contains: str | None = None
    expected_structural_artifact: bool = False
    expected_live_artifact_verifier: bool = False
    expected_facts_bound_verifier: bool = False
    expected_offline_replay_complete: bool = False
    expected_live_frontier_agreement_complete: bool = False
    expected_live_count_agreement_complete: bool = False
    expected_snapshot_resume_agreement_complete: bool = False
    expected_continuation_asset_complete: bool = False
    expected_rust_runtime_agreement_complete: bool = False
    expected_continuation_snapshot_resume_complete: bool = False
    expected_lazy_branch_proof_complete: bool = False
    expected_lazy_terminal_proof_complete: bool = False
    expected_support_digest: str | None = None
    expected_rdkit_audit_version_pinned: bool = False
    expected_offline_object_kinds: tuple[str, ...] = ()
    expected_offline_unchecked_object_kinds: tuple[str, ...] = ()
    expected_offline_relation_families: tuple[str, ...] = ()
    expected_offline_unchecked_obligation_families: tuple[str, ...] = ()
    expected_continuation_raw_cursor_count: int | None = None
    expected_continuation_edge_locator_count: int | None = None
    expected_continuation_branch_locator_count: int | None = None
    expected_continuation_terminal_record_count: int | None = None
    expected_continuation_terminal_locator_count: int | None = None
    expected_continuation_replayed_operations: tuple[str, ...] = ()
    expected_continuation_checked_relation_families: tuple[str, ...] = ()
    expected_continuation_checked_obligation_families: tuple[str, ...] = ()
    expected_continuation_unchecked_obligation_families: tuple[str, ...] = ()

    @property
    def rdkit_audit_families(self) -> tuple[RdkitAuditFamily, ...]:
        contract = _EXTRACTION_BY_ID.get(_contract_id_for_options(self.extraction_options))
        families = list(contract.rdkit_audit_families if contract else ())
        for family in _SURFACES_BY_NAME[self.support_surface].rdkit_audit_families:
            if family not in families:
                families.append(family)
        return tuple(families)

    @property
    def expectation(self) -> CaseExpectation:
        if self.expected == "accepted":
            continuation = None
            if self.expected_continuation_raw_cursor_count is not None:
                continuation = ContinuationProofExpectation(
                    self.expected_continuation_raw_cursor_count,
                    self.expected_continuation_edge_locator_count or 0,
                    self.expected_continuation_branch_locator_count or 0,
                    self.expected_continuation_terminal_record_count or 0,
                    self.expected_continuation_terminal_locator_count or 0,
                    self.expected_continuation_replayed_operations,
                    self.expected_continuation_checked_relation_families,
                    self.expected_continuation_checked_obligation_families,
                    self.expected_continuation_unchecked_obligation_families,
                )
            return AcceptedCaseExpectation(self.expected_support_count or 0, self.expected_completion_count or 0, self.expected_support_digest or "", self.qualification_authority or "materialized_support_artifact", continuation)
        if self.blocker_phase == "preparation":
            return PreparationBlockerExpectation(self.blocker_kind or "", self.blocker_error_kind or SouthStarErrorKind.INTERNAL_INVARIANT, self.blocker_message_contains or "")
        return FrontierBlockerExpectation(self.blocker_kind or "", self.blocker_operation or "")


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
    continuation_values = continuation or ContinuationProofExpectation(0, 0, 0, 0, 0, (), (), (), ())
    return DefaultWriterCapabilityCase(
        name, smiles, contract.public_profile, contract.options, "accepted", support_surface,
        qualification_authority, rooted_at_atom, support_count, completion_count, None, None,
        None, None, None, evidence.structural_artifact, evidence.live_artifact_verifier,
        evidence.facts_bound_verifier, evidence.offline_replay_complete, evidence.live_frontier_agreement_complete,
        evidence.live_count_agreement_complete, evidence.snapshot_resume_agreement_complete,
        evidence.continuation_asset_complete, evidence.rust_runtime_agreement_complete,
        evidence.continuation_snapshot_resume_complete, evidence.lazy_branch_proof_complete,
        evidence.lazy_terminal_proof_complete, support_digest,
        bool(contract.rdkit_audit_families or _SURFACES_BY_NAME[support_surface].rdkit_audit_families),
        evidence.offline_object_kinds, evidence.offline_unchecked_object_kinds, evidence.offline_relation_families,
        evidence.offline_unchecked_obligation_families,
        (continuation_values.raw_cursor_count or None), (continuation_values.edge_locator_count or None),
        (continuation_values.branch_locator_count or None), (continuation_values.terminal_record_count or None),
        (continuation_values.terminal_locator_count or None), continuation_values.replayed_operations,
        continuation_values.checked_relation_families, continuation_values.checked_obligation_families,
        continuation_values.unchecked_obligation_families,
    )


def preparation_blocked_writer_case(*, name: str, smiles: str, extraction_contract_id: ExtractionContractId, support_surface: str, kind: str, error_kind: SouthStarErrorKind, message_contains: str, rooted_at_atom: int = 0) -> DefaultWriterCapabilityCase:
    _validate_common(name, smiles, extraction_contract_id, support_surface)
    if _SURFACES_BY_NAME[support_surface].expected_disposition != "blocked":
        raise ValueError("preparation blockers require blocked support surfaces")
    contract = _EXTRACTION_BY_ID[extraction_contract_id]
    return DefaultWriterCapabilityCase(
        name, smiles, contract.public_profile, contract.options, "blocked", support_surface,
        None, rooted_at_atom, None, None, "preparation", kind, None, error_kind,
        message_contains,
        expected_rdkit_audit_version_pinned=bool(
            contract.rdkit_audit_families
            or _SURFACES_BY_NAME[support_surface].rdkit_audit_families
        ),
    )


def frontier_blocked_writer_case(*, name: str, smiles: str, extraction_contract_id: ExtractionContractId, support_surface: str, kind: str, operation: str, rooted_at_atom: int = 0) -> DefaultWriterCapabilityCase:
    _validate_common(name, smiles, extraction_contract_id, support_surface)
    if _SURFACES_BY_NAME[support_surface].expected_disposition != "blocked":
        raise ValueError("frontier blockers require blocked support surfaces")
    contract = _EXTRACTION_BY_ID[extraction_contract_id]
    return DefaultWriterCapabilityCase(
        name, smiles, contract.public_profile, contract.options, "blocked", support_surface,
        None, rooted_at_atom, None, None, "frontier", kind, operation,
        expected_rdkit_audit_version_pinned=bool(
            contract.rdkit_audit_families
            or _SURFACES_BY_NAME[support_surface].rdkit_audit_families
        ),
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
