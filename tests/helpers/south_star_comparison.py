from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

import grimace

from tests.helpers.south_star_semantic_oracle import (
    SouthStarConformanceReport,
    parse_smiles,
    semantic_oracle_accepts,
    south_star_conformance_report,
)
from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native_for_case
from tests.helpers.south_star_exact_support import SouthStarExpandedSupportCase
from tests.helpers.south_star_semantics import SouthStarSemanticCase


SOUTH_STAR_INTERSECTION = "intersection"
SOUTH_STAR_ONLY = "SouthStarOnly"
RDKIT_PARITY_ONLY = "RDKitParityOnly"


@dataclass(frozen=True, slots=True)
class SouthStarComparisonLabel:
    case_id: str
    candidate_smiles: str
    expected_semantic_acceptance: bool
    semantic_oracle_accepts: bool
    rdkit_writer_observed: bool
    grimace_parity_support_accepts: bool


@dataclass(frozen=True, slots=True)
class SouthStarSupportMembershipClassification:
    candidate_smiles: str
    membership: str
    conformance_report: SouthStarConformanceReport


@dataclass(frozen=True, slots=True)
class SouthStarParityComparisonReport:
    case_id: str
    rdkit_parity_available: bool
    rdkit_parity_unavailable_reason: str
    south_star_support_size: int
    rdkit_parity_support_size: int
    intersection_size: int
    south_star_only: tuple[str, ...]
    rdkit_parity_only: tuple[str, ...]
    intersection: tuple[str, ...]
    classifications: tuple[SouthStarSupportMembershipClassification, ...]


@dataclass(frozen=True, slots=True)
class SouthStarPublicParitySupport:
    support: frozenset[str]
    available: bool
    unavailable_reason: str


def south_star_comparison_labels(
    case: SouthStarSemanticCase,
    *,
    rdkit_sample_count: int = 128,
) -> tuple[SouthStarComparisonLabel, ...]:
    positive_labels = tuple(
        _comparison_label(
            case,
            candidate_smiles=candidate,
            expected_semantic_acceptance=True,
            rdkit_sample_count=rdkit_sample_count,
        )
        for candidate in case.positive_semantic_smiles
    )
    negative_labels = tuple(
        _comparison_label(
            case,
            candidate_smiles=negative.smiles,
            expected_semantic_acceptance=False,
            rdkit_sample_count=rdkit_sample_count,
        )
        for negative in case.negative_semantic_smiles
    )
    return positive_labels + negative_labels


def south_star_parity_comparison_report(
    case: SouthStarSemanticCase,
) -> SouthStarParityComparisonReport:
    return _parity_comparison_report(
        case_id=case.case_id,
        source_smiles=case.source_smiles,
        south_star_outputs=mol_to_smiles_enum_s_graph_native_for_case(case).outputs,
    )


def south_star_expanded_parity_comparison_report(
    case: SouthStarExpandedSupportCase,
) -> SouthStarParityComparisonReport:
    return _parity_comparison_report(
        case_id=case.case_id,
        source_smiles=case.source_smiles,
        south_star_outputs=mol_to_smiles_enum_s_graph_native(
            case.source_smiles,
            case_id=case.case_id,
        ).outputs,
    )


def _parity_comparison_report(
    *,
    case_id: str,
    source_smiles: str,
    south_star_outputs: tuple[str, ...],
) -> SouthStarParityComparisonReport:
    south_star_support = frozenset(south_star_outputs)
    public_parity = _grimace_public_parity_support_or_unavailable(source_smiles)
    rdkit_parity_support = public_parity.support
    intersection = south_star_support.intersection(rdkit_parity_support)
    south_star_only = south_star_support.difference(rdkit_parity_support)
    rdkit_parity_only = rdkit_parity_support.difference(south_star_support)

    return SouthStarParityComparisonReport(
        case_id=case_id,
        rdkit_parity_available=public_parity.available,
        rdkit_parity_unavailable_reason=public_parity.unavailable_reason,
        south_star_support_size=len(south_star_support),
        rdkit_parity_support_size=len(rdkit_parity_support),
        intersection_size=len(intersection),
        south_star_only=tuple(sorted(south_star_only)),
        rdkit_parity_only=tuple(sorted(rdkit_parity_only)),
        intersection=tuple(sorted(intersection)),
        classifications=tuple(
            _membership_classification(
                source_smiles=source_smiles,
                candidate_smiles=candidate_smiles,
                membership=membership,
            )
            for membership, candidates in (
                (SOUTH_STAR_INTERSECTION, intersection),
                (SOUTH_STAR_ONLY, south_star_only),
                (RDKIT_PARITY_ONLY, rdkit_parity_only),
            )
            for candidate_smiles in sorted(candidates)
        ),
    )


def _comparison_label(
    case: SouthStarSemanticCase,
    *,
    candidate_smiles: str,
    expected_semantic_acceptance: bool,
    rdkit_sample_count: int,
) -> SouthStarComparisonLabel:
    return SouthStarComparisonLabel(
        case_id=case.case_id,
        candidate_smiles=candidate_smiles,
        expected_semantic_acceptance=expected_semantic_acceptance,
        semantic_oracle_accepts=semantic_oracle_accepts(
            source_smiles=case.source_smiles,
            candidate_smiles=candidate_smiles,
        ),
        rdkit_writer_observed=candidate_smiles
        in _sample_rdkit_writer_outputs(case.source_smiles, sample_count=rdkit_sample_count),
        grimace_parity_support_accepts=candidate_smiles
        in _grimace_public_parity_support(case.source_smiles),
    )


def _membership_classification(
    *,
    source_smiles: str,
    candidate_smiles: str,
    membership: str,
) -> SouthStarSupportMembershipClassification:
    return SouthStarSupportMembershipClassification(
        candidate_smiles=candidate_smiles,
        membership=membership,
        conformance_report=south_star_conformance_report(
            source_smiles=source_smiles,
            candidate_smiles=candidate_smiles,
        ),
    )


def _sample_rdkit_writer_outputs(
    source_smiles: str,
    *,
    sample_count: int,
) -> frozenset[str]:
    mol = parse_smiles(source_smiles)
    return frozenset(
        Chem.MolToSmiles(
            mol,
            canonical=False,
            doRandom=True,
            isomericSmiles=True,
        )
        for _ in range(sample_count)
    )


def _grimace_public_parity_support(source_smiles: str) -> frozenset[str]:
    return frozenset(
        grimace.MolToSmilesEnum(
            parse_smiles(source_smiles),
            canonical=False,
            doRandom=True,
            isomericSmiles=True,
        )
    )


def _grimace_public_parity_support_or_unavailable(
    source_smiles: str,
) -> SouthStarPublicParitySupport:
    try:
        support = _grimace_public_parity_support(source_smiles)
    except (NotImplementedError, ValueError) as exc:
        return SouthStarPublicParitySupport(
            support=frozenset(),
            available=False,
            unavailable_reason=f"{type(exc).__name__}: {exc}",
        )
    return SouthStarPublicParitySupport(
        support=support,
        available=True,
        unavailable_reason="",
    )
