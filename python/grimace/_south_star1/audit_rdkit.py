"""Optional RDKit audit helpers for the private proof kernel.

Audit code may compare generated strings against external parser behavior in
tests, but it is not part of support definition or enumeration.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from enum import Enum
from typing import Literal

from rdkit import Chem

from .errors import SouthStarErrorKind
from .fact_isomorphism import FactIsomorphismResult
from .fact_isomorphism import facts_are_isomorphic
from .facts import LigandKind
from .facts import MoleculeFacts
from .facts import SiteStatus
from .facts import StereoFacts
from .ids import AtomId
from .ids import BondId
from .ordinary_stereo_closure import RawDirectionalStereoRecord
from .ordinary_stereo_closure import RawSpecifiedStereo
from .ordinary_stereo_closure import RawStereoRecordId
from .ordinary_stereo_closure import RawTetraStereoRecord
from .ordinary_stereo_closure import StereoClosureCertificate
from .ordinary_stereo_closure import certify_ordinary_stereo_specified_closure
from .ordinary_stereo_closure import raw_record_id
from .ordinary_policy import OrdinaryPolicyOptions
from .ordinary_policy import ordinary_policy_for_facts
from .ordinary_semantics import OrdinarySmilesSemantics
from .ordinary_stereo_sites import OrdinaryStereoSiteOptions
from .rdkit_adapter import RdkitOrdinaryExtractionOptions
from .rdkit_adapter import ordinary_molecule_facts_from_rdkit
from .semantics import ParserSemantics
from .skeleton import TraversalSkeleton
from .support_enumeration import enumerate_stereo_support
from .support_enumeration import enumerate_stereo_witnesses


@dataclass(frozen=True, slots=True)
class RdkitAuditCase:
    name: str
    smiles: str
    kind: Literal["supported", "unsupported"]
    tags: tuple[str, ...]
    policy_options: OrdinaryPolicyOptions = field(default_factory=OrdinaryPolicyOptions)
    adapter_options: RdkitOrdinaryExtractionOptions = field(
        default_factory=RdkitOrdinaryExtractionOptions,
    )
    max_support_size: int | None = None
    expected_contains: tuple[str, ...] = ()
    expected_error_kind: SouthStarErrorKind | None = None


SOUTH_STAR1_SUPPORTED_V0_AUDIT_CASES = (
    RdkitAuditCase(
        name="nonstereo_tree",
        smiles="CCO",
        kind="supported",
        tags=("nonstereo",),
        max_support_size=12,
    ),
    RdkitAuditCase(
        name="simple_directional",
        smiles="C(/F)=C(\\Cl)",
        kind="supported",
        tags=("directional",),
        max_support_size=64,
    ),
    RdkitAuditCase(
        name="ring_tetra",
        smiles="[C@H]1(F)CO1",
        kind="supported",
        tags=("ring", "tetra"),
        max_support_size=168,
    ),
    RdkitAuditCase(
        name="disconnected_stereo",
        smiles="CCO.[C@H](F)(Cl)Br",
        kind="supported",
        tags=("disconnected", "tetra"),
        max_support_size=432,
    ),
    RdkitAuditCase(
        name="mixed_tetra_directional",
        smiles="[C@H](F)(Cl)C(/F)=C(\\Cl)",
        kind="supported",
        tags=("mixed", "tetra", "directional"),
        max_support_size=1024,
    ),
    RdkitAuditCase(
        name="joint_double_ring_closure",
        smiles="C1=CC1",
        kind="supported",
        tags=("ring", "double"),
        policy_options=OrdinaryPolicyOptions(non_single_ring_closures="joint"),
        max_support_size=32,
    ),
    RdkitAuditCase(
        name="joint_triple_ring_closure",
        smiles="C1#CC1",
        kind="supported",
        tags=("ring", "triple"),
        policy_options=OrdinaryPolicyOptions(non_single_ring_closures="joint"),
        max_support_size=32,
    ),
)


SOUTH_STAR1_UNSUPPORTED_V0_AUDIT_CASES = (
    RdkitAuditCase(
        name="enhanced_stereo_group",
        smiles="F[C@H](Cl)Br |&1:1|",
        kind="unsupported",
        tags=("enhanced", "tetra"),
        expected_error_kind=SouthStarErrorKind.UNSUPPORTED_STEREO,
    ),
)


_EXACT_EQUIVALENCE_ADAPTER_OPTIONS = RdkitOrdinaryExtractionOptions(
    stereo_site_options=OrdinaryStereoSiteOptions(
        ligand_equivalence="exact_graph_automorphism",
    ),
)


SOUTH_STAR1_EXPERIMENTAL_EXACT_EQUIVALENCE_SUPPORTED_CASES = (
    RdkitAuditCase(
        name="exact_deep_distinct_tetra",
        smiles="[C@H](F)(CBr)CCl",
        kind="supported",
        tags=("experimental", "exact_ligand_equivalence", "tetra"),
        adapter_options=_EXACT_EQUIVALENCE_ADAPTER_OPTIONS,
        max_support_size=256,
    ),
    RdkitAuditCase(
        name="exact_deep_distinct_directional",
        smiles="BrC/C(CCl)=C/F",
        kind="supported",
        tags=("experimental", "exact_ligand_equivalence", "directional"),
        adapter_options=_EXACT_EQUIVALENCE_ADAPTER_OPTIONS,
        max_support_size=1024,
    ),
)


@dataclass(frozen=True, slots=True)
class RdkitAuditResult:
    text: str
    parsed: bool
    comparison: FactIsomorphismResult | None
    parse_error: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.parsed
            and self.comparison is not None
            and self.comparison.isomorphic
        )


@dataclass(frozen=True, slots=True)
class RdkitWitnessAuditResult:
    witness_id: str
    text: str
    annotation_count: int
    constraints: tuple[str, ...]
    parsed: bool
    comparison: FactIsomorphismResult | None
    parse_error: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.parsed
            and self.comparison is not None
            and self.comparison.isomorphic
        )


RdkitAnyAuditResult = RdkitAuditResult | RdkitWitnessAuditResult


@dataclass(frozen=True, slots=True)
class RdkitAuditSummary:
    case_name: str
    input_smiles: str
    support_count: int
    witness_count: int | None
    parsed_count: int
    failed_parse_count: int
    failed_isomorphism_count: int
    first_failure: RdkitAnyAuditResult | None

    @property
    def ok(self) -> bool:
        return self.failed_parse_count == 0 and self.failed_isomorphism_count == 0


@dataclass(frozen=True, slots=True)
class TetraStatusSummary:
    specified: int
    unspecified: int


@dataclass(frozen=True, slots=True)
class RawStereoSummary:
    tetrahedral: tuple[RawStereoRecordId, ...]
    directional: tuple[RawStereoRecordId, ...]


class RoundTripFailureKind(Enum):
    PARSE_FAILURE = "parse_failure"
    SPECIFIED_TETRA_RECORD_LOSS = "specified_tetra_record_loss"
    SPECIFIED_TETRA_RECORD_GAIN = "specified_tetra_record_gain"
    UNSPECIFIED_TETRA_POTENTIAL_MISMATCH = "unspecified_tetra_potential_mismatch"
    CLOSURE_CERTIFICATE_MISMATCH = "closure_certificate_mismatch"
    FACT_PROMOTION_MISMATCH = "fact_promotion_mismatch"
    OTHER_ISOMORPHISM_MISMATCH = "other_isomorphism_mismatch"


@dataclass(frozen=True, slots=True)
class SpecifiedClosureRoundTripTrace:
    text: str
    parsed: bool
    original_tetra_status: TetraStatusSummary
    reparsed_tetra_status: TetraStatusSummary
    original_raw: RawStereoSummary
    reparsed_raw_sanitized: RawStereoSummary
    reparsed_raw_unsanitized: RawStereoSummary | None
    original_certificates: tuple[StereoClosureCertificate, ...]
    reparsed_certificates: tuple[StereoClosureCertificate, ...]
    isomorphic_with_potential_sites: bool
    isomorphic_without_potential_sites: bool
    failure_kind: RoundTripFailureKind | None
    failure_reason: str | None


def audit_generated_support_with_rdkit(
    mol: Chem.Mol,
    *,
    policy_options: OrdinaryPolicyOptions = OrdinaryPolicyOptions(),
    adapter_options: RdkitOrdinaryExtractionOptions = (
        RdkitOrdinaryExtractionOptions()
    ),
    semantics: ParserSemantics | None = None,
    skeletons: Iterable[TraversalSkeleton] | None = None,
) -> tuple[RdkitAuditResult, ...]:
    """Audit generated support by reparsing each string with RDKit.

    This function is an external falsifier for the declared South Star model.
    It never filters, repairs, or redefines the generated support.
    """

    original = ordinary_molecule_facts_from_rdkit(mol, adapter_options)
    policy = ordinary_policy_for_facts(original, policy_options)
    image = enumerate_stereo_support(
        facts=original,
        policy=policy,
        semantics=semantics or OrdinarySmilesSemantics(),
        skeletons=None if skeletons is None else tuple(skeletons),
    )

    results: list[RdkitAuditResult] = []
    for text in image.strings:
        parsed = Chem.MolFromSmiles(text)
        if parsed is None:
            results.append(
                RdkitAuditResult(
                    text=text,
                    parsed=False,
                    comparison=None,
                    parse_error="RDKit MolFromSmiles returned None",
                )
            )
            continue

        reparsed = ordinary_molecule_facts_from_rdkit(parsed, adapter_options)
        results.append(
            RdkitAuditResult(
                text=text,
                parsed=True,
                comparison=facts_are_isomorphic(original, reparsed),
            )
        )

    return tuple(results)


def trace_specified_closure_round_trip(
    text: str,
    *,
    original_facts: MoleculeFacts,
    extraction_options: RdkitOrdinaryExtractionOptions,
    policy_options: OrdinaryPolicyOptions,
) -> SpecifiedClosureRoundTripTrace:
    """Trace a specified-closure support string through RDKit reparse."""

    del policy_options
    original_raw = _raw_specified_from_facts(original_facts)
    original_certificates = _certificates_for_raw_context(
        original_facts,
        original_raw,
        extraction_options,
    )
    parsed = Chem.MolFromSmiles(text)
    if parsed is None:
        empty_raw = RawStereoSummary(tetrahedral=(), directional=())
        return SpecifiedClosureRoundTripTrace(
            text=text,
            parsed=False,
            original_tetra_status=_tetra_status_summary(original_facts),
            reparsed_tetra_status=TetraStatusSummary(0, 0),
            original_raw=_raw_summary_from_raw(original_raw),
            reparsed_raw_sanitized=empty_raw,
            reparsed_raw_unsanitized=None,
            original_certificates=original_certificates,
            reparsed_certificates=(),
            isomorphic_with_potential_sites=False,
            isomorphic_without_potential_sites=False,
            failure_kind=RoundTripFailureKind.PARSE_FAILURE,
            failure_reason="RDKit MolFromSmiles returned None",
        )

    reparsed_facts = ordinary_molecule_facts_from_rdkit(parsed, extraction_options)
    sanitized_raw = _raw_summary_from_rdkit_mol(parsed)
    unsanitized = Chem.MolFromSmiles(text, sanitize=False)
    unsanitized_raw = (
        None if unsanitized is None else _raw_summary_from_rdkit_mol(unsanitized)
    )
    reparsed_raw = _raw_specified_from_facts(reparsed_facts)
    reparsed_certificates = _certificates_for_raw_context(
        reparsed_facts,
        reparsed_raw,
        extraction_options,
    )
    with_potential = facts_are_isomorphic(
        original_facts,
        reparsed_facts,
        compare_stereo=True,
        compare_potential_sites=True,
    )
    without_potential = facts_are_isomorphic(
        original_facts,
        reparsed_facts,
        compare_stereo=True,
        compare_potential_sites=False,
    )
    failure_kind = _round_trip_failure_kind(
        original_raw=_raw_summary_from_raw(original_raw),
        sanitized_raw=sanitized_raw,
        original_certificates=original_certificates,
        reparsed_certificates=reparsed_certificates,
        with_potential=with_potential,
        without_potential=without_potential,
    )
    return SpecifiedClosureRoundTripTrace(
        text=text,
        parsed=True,
        original_tetra_status=_tetra_status_summary(original_facts),
        reparsed_tetra_status=_tetra_status_summary(reparsed_facts),
        original_raw=_raw_summary_from_raw(original_raw),
        reparsed_raw_sanitized=sanitized_raw,
        reparsed_raw_unsanitized=unsanitized_raw,
        original_certificates=original_certificates,
        reparsed_certificates=reparsed_certificates,
        isomorphic_with_potential_sites=with_potential.isomorphic,
        isomorphic_without_potential_sites=without_potential.isomorphic,
        failure_kind=failure_kind,
        failure_reason=with_potential.reason,
    )


def classify_specified_closure_round_trips(
    traces: Iterable[SpecifiedClosureRoundTripTrace],
) -> dict[RoundTripFailureKind, int]:
    counts: dict[RoundTripFailureKind, int] = {}
    for trace in traces:
        if trace.failure_kind is None:
            continue
        counts[trace.failure_kind] = counts.get(trace.failure_kind, 0) + 1
    return counts


def trace_specified_closure_support_round_trips(
    original_facts: MoleculeFacts,
    *,
    extraction_options: RdkitOrdinaryExtractionOptions,
    policy_options: OrdinaryPolicyOptions = OrdinaryPolicyOptions(),
    semantics: ParserSemantics | None = None,
    skeletons: Iterable[TraversalSkeleton] | None = None,
) -> tuple[SpecifiedClosureRoundTripTrace, ...]:
    """Trace every rendered support string from a specified-closure fact object."""

    policy = ordinary_policy_for_facts(original_facts, policy_options)
    image = enumerate_stereo_support(
        facts=original_facts,
        policy=policy,
        semantics=semantics or OrdinarySmilesSemantics(),
        skeletons=None if skeletons is None else tuple(skeletons),
    )
    return tuple(
        trace_specified_closure_round_trip(
            text,
            original_facts=original_facts,
            extraction_options=extraction_options,
            policy_options=policy_options,
        )
        for text in image.strings
    )


def _tetra_status_summary(facts: MoleculeFacts) -> TetraStatusSummary:
    specified = sum(
        site.status is SiteStatus.SPECIFIED
        for site in facts.stereo.tetrahedral
    )
    unspecified = sum(
        site.status is SiteStatus.UNSPECIFIED
        for site in facts.stereo.tetrahedral
    )
    return TetraStatusSummary(specified=specified, unspecified=unspecified)


def _raw_summary_from_rdkit_mol(mol: Chem.Mol) -> RawStereoSummary:
    tetrahedral = tuple(
        RawStereoRecordId("tetrahedral", (AtomId(atom.GetIdx()),))
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    )
    directional = tuple(
        RawStereoRecordId("directional", (BondId(bond.GetIdx()),))
        for bond in mol.GetBonds()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE
    )
    return RawStereoSummary(tetrahedral=tetrahedral, directional=directional)


def _raw_summary_from_raw(raw: RawSpecifiedStereo) -> RawStereoSummary:
    return RawStereoSummary(
        tetrahedral=tuple(raw_record_id(record) for record in raw.tetrahedral),
        directional=tuple(raw_record_id(record) for record in raw.directional),
    )


def _raw_specified_from_facts(facts: MoleculeFacts) -> RawSpecifiedStereo:
    occurrence_by_id = {
        occurrence.id: occurrence
        for occurrence in facts.ligand_occurrences
    }
    tetrahedral = tuple(
        RawTetraStereoRecord(
            center=site.center,
            target=site.target,
            reference_atoms=_tetra_reference_atoms_from_site(
                site.reference_order,
                occurrence_by_id,
            ),
        )
        for site in facts.stereo.tetrahedral
        if site.status is SiteStatus.SPECIFIED
    )
    directional = tuple(
        RawDirectionalStereoRecord(
            center_bond=site.center_bond,
            target=site.target,
            reference_atoms=_directional_reference_atoms_from_site(
                site,
                occurrence_by_id,
            ),
        )
        for site in facts.stereo.directional
        if site.status is SiteStatus.SPECIFIED
    )
    return RawSpecifiedStereo(tetrahedral=tetrahedral, directional=directional)


def _tetra_reference_atoms_from_site(
    reference_order: tuple[object, ...],
    occurrence_by_id: dict[object, object],
) -> tuple[AtomId | None, ...]:
    out: list[AtomId | None] = []
    for occurrence_id in reference_order:
        occurrence = occurrence_by_id[occurrence_id]
        if occurrence.kind is LigandKind.IMPLICIT_H:
            out.append(None)
            continue
        if occurrence.atom is None:
            raise ValueError(
                "specified tetrahedral reference occurrence lacks atom: "
                f"{occurrence.id!r}",
            )
        out.append(occurrence.atom)
    return tuple(out)


def _directional_reference_atoms_from_site(
    site,
    occurrence_by_id: dict[object, object],
) -> tuple[AtomId, AtomId]:
    if site.reference_pair is None:
        raise ValueError(
            f"specified directional site lacks reference pair: {site.id!r}",
        )
    out: list[AtomId] = []
    for occurrence_id in site.reference_pair:
        occurrence = occurrence_by_id[occurrence_id]
        if occurrence.kind is not LigandKind.NEIGHBOR_ATOM or occurrence.atom is None:
            raise ValueError(
                "specified directional reference occurrence is not a neighbor "
                f"atom: {occurrence.id!r}",
            )
        out.append(occurrence.atom)
    return (out[0], out[1])


def _certificates_for_raw_context(
    facts: MoleculeFacts,
    raw: RawSpecifiedStereo,
    extraction_options: RdkitOrdinaryExtractionOptions,
) -> tuple[StereoClosureCertificate, ...]:
    if not raw.tetrahedral and not raw.directional:
        return ()
    return certify_ordinary_stereo_specified_closure(
        replace(facts, stereo=StereoFacts(), ligand_occurrences=()),
        raw_specified=raw,
        site_options=extraction_options.stereo_site_options,
    )


def _round_trip_failure_kind(
    *,
    original_raw: RawStereoSummary,
    sanitized_raw: RawStereoSummary,
    original_certificates: tuple[StereoClosureCertificate, ...],
    reparsed_certificates: tuple[StereoClosureCertificate, ...],
    with_potential: FactIsomorphismResult,
    without_potential: FactIsomorphismResult,
) -> RoundTripFailureKind | None:
    if with_potential.isomorphic:
        return None
    if len(sanitized_raw.tetrahedral) < len(original_raw.tetrahedral):
        return RoundTripFailureKind.SPECIFIED_TETRA_RECORD_LOSS
    if len(sanitized_raw.tetrahedral) > len(original_raw.tetrahedral):
        return RoundTripFailureKind.SPECIFIED_TETRA_RECORD_GAIN
    if without_potential.isomorphic:
        return RoundTripFailureKind.UNSPECIFIED_TETRA_POTENTIAL_MISMATCH
    if _certificate_signature(original_certificates) != _certificate_signature(
        reparsed_certificates,
    ):
        return RoundTripFailureKind.CLOSURE_CERTIFICATE_MISMATCH
    if "site status count mismatch" in (with_potential.reason or ""):
        return RoundTripFailureKind.FACT_PROMOTION_MISMATCH
    return RoundTripFailureKind.OTHER_ISOMORPHISM_MISMATCH


def _certificate_signature(
    certificates: tuple[StereoClosureCertificate, ...],
) -> frozenset[tuple[RawStereoRecordId, str, SiteStatus | None, str | None]]:
    return frozenset(
        (
            certificate.raw_record,
            certificate.status,
            None
            if certificate.matched_site is None
            else SiteStatus.SPECIFIED,
            certificate.reason,
        )
        for certificate in certificates
    )


def summarize_rdkit_audit(
    *,
    case_name: str,
    input_smiles: str,
    results: Iterable[RdkitAnyAuditResult],
    witness_count: int | None = None,
) -> RdkitAuditSummary:
    result_tuple = tuple(results)
    first_failure = next(
        (result for result in result_tuple if not result.ok),
        None,
    )
    failed_parse_count = sum(not result.parsed for result in result_tuple)
    failed_isomorphism_count = sum(
        result.parsed
        and result.comparison is not None
        and not result.comparison.isomorphic
        for result in result_tuple
    )
    return RdkitAuditSummary(
        case_name=case_name,
        input_smiles=input_smiles,
        support_count=len(result_tuple),
        witness_count=witness_count,
        parsed_count=sum(result.parsed for result in result_tuple),
        failed_parse_count=failed_parse_count,
        failed_isomorphism_count=failed_isomorphism_count,
        first_failure=first_failure,
    )


def audit_generated_witnesses_with_rdkit(
    mol: Chem.Mol,
    *,
    policy_options: OrdinaryPolicyOptions = OrdinaryPolicyOptions(),
    adapter_options: RdkitOrdinaryExtractionOptions = (
        RdkitOrdinaryExtractionOptions()
    ),
    semantics: ParserSemantics | None = None,
    skeletons: Iterable[TraversalSkeleton] | None = None,
) -> tuple[RdkitWitnessAuditResult, ...]:
    """Audit generated witnesses while preserving diagnostic witness context."""

    original = ordinary_molecule_facts_from_rdkit(mol, adapter_options)
    policy = ordinary_policy_for_facts(original, policy_options)
    witness_iter = enumerate_stereo_witnesses(
        facts=original,
        policy=policy,
        semantics=semantics or OrdinarySmilesSemantics(),
        skeletons=None if skeletons is None else tuple(skeletons),
    )

    results: list[RdkitWitnessAuditResult] = []
    for witness in witness_iter:
        parsed = Chem.MolFromSmiles(witness.rendered)
        if parsed is None:
            results.append(
                RdkitWitnessAuditResult(
                    witness_id=witness.id,
                    text=witness.rendered,
                    annotation_count=witness.annotation_count,
                    constraints=_constraint_names(witness.constraints),
                    parsed=False,
                    comparison=None,
                    parse_error="RDKit MolFromSmiles returned None",
                )
            )
            continue

        reparsed = ordinary_molecule_facts_from_rdkit(parsed, adapter_options)
        results.append(
            RdkitWitnessAuditResult(
                witness_id=witness.id,
                text=witness.rendered,
                annotation_count=witness.annotation_count,
                constraints=_constraint_names(witness.constraints),
                parsed=True,
                comparison=facts_are_isomorphic(original, reparsed),
            )
        )

    return tuple(results)


def _constraint_names(constraints) -> tuple[str, ...]:
    return tuple(constraint.name for constraint in constraints)


__all__ = (
    "RdkitAuditCase",
    "RdkitAuditResult",
    "RdkitAuditSummary",
    "RdkitWitnessAuditResult",
    "RawStereoSummary",
    "RoundTripFailureKind",
    "SOUTH_STAR1_EXPERIMENTAL_EXACT_EQUIVALENCE_SUPPORTED_CASES",
    "SOUTH_STAR1_SUPPORTED_V0_AUDIT_CASES",
    "SOUTH_STAR1_UNSUPPORTED_V0_AUDIT_CASES",
    "SpecifiedClosureRoundTripTrace",
    "TetraStatusSummary",
    "audit_generated_support_with_rdkit",
    "audit_generated_witnesses_with_rdkit",
    "classify_specified_closure_round_trips",
    "summarize_rdkit_audit",
    "trace_specified_closure_support_round_trips",
    "trace_specified_closure_round_trip",
)
