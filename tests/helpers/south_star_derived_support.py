from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from grimace._south_star.fragments import (
    SouthStarFragmentSupport,
    compose_disconnected_fragment_supports,
)
from tests.helpers.south_star_exact_support import (
    load_south_star_exact_first_domain_cases,
    load_south_star_expanded_support_cases,
)


DERIVED_SUPPORT_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "south_star_derived_support"
    / "derived_support_v1.json"
)
DERIVED_SUPPORT_POLICY = "south_star_derived_support_v1"


@dataclass(frozen=True, slots=True)
class SouthStarDerivedFragmentRef:
    fixture_family: str
    case_id: str
    fragment_id: str
    expected_output_count: int


@dataclass(frozen=True, slots=True)
class SouthStarDerivedSupportCase:
    case_id: str
    source_smiles: str
    support_authority: str
    evidence_notes: str
    fragment_refs: tuple[SouthStarDerivedFragmentRef, ...]
    fragment_order_policy: str
    output_order_policy: str
    expected_product_count: int
    expected_digest_sha256: str
    sentinel_outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SouthStarDerivedSupportProof:
    case: SouthStarDerivedSupportCase
    outputs: tuple[str, ...]
    digest_sha256: str
    fragment_output_counts: tuple[int, ...]
    fragment_order_policy: str
    output_order_policy: str
    estimated_product_size: int


def load_south_star_derived_support_cases(
    path: Path = DERIVED_SUPPORT_FIXTURE,
) -> tuple[SouthStarDerivedSupportCase, ...]:
    raw = json.loads(path.read_text())
    if raw["schema_version"] != 1:
        raise ValueError(f"unsupported South Star derived-support schema: {raw!r}")
    if raw["policy"] != DERIVED_SUPPORT_POLICY:
        raise ValueError(f"unsupported South Star derived-support policy: {raw!r}")
    cases = tuple(_derived_support_case(case) for case in raw["cases"])
    case_ids = tuple(case.case_id for case in cases)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("duplicate South Star derived-support case ids")
    return cases


def derived_support_proof_for_case(
    case: SouthStarDerivedSupportCase,
) -> SouthStarDerivedSupportProof:
    fragment_supports = tuple(
        SouthStarFragmentSupport(
            fragment_id=fragment_ref.fragment_id,
            outputs=_support_for_fragment_ref(fragment_ref),
        )
        for fragment_ref in case.fragment_refs
    )
    composition = compose_disconnected_fragment_supports(fragment_supports)
    digest = support_digest_sha256(composition.outputs)
    return SouthStarDerivedSupportProof(
        case=case,
        outputs=composition.outputs,
        digest_sha256=digest,
        fragment_output_counts=composition.fragment_output_counts,
        fragment_order_policy=composition.fragment_order_policy,
        output_order_policy=composition.output_order_policy,
        estimated_product_size=composition.estimated_product_size,
    )


def support_digest_sha256(outputs: tuple[str, ...]) -> str:
    payload = json.dumps(list(outputs), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _derived_support_case(raw_case: object) -> SouthStarDerivedSupportCase:
    if not isinstance(raw_case, dict):
        raise ValueError(
            f"South Star derived-support case must be a dict: {raw_case!r}"
        )
    fragment_refs = tuple(
        SouthStarDerivedFragmentRef(
            fixture_family=raw_ref["fixture_family"],
            case_id=raw_ref["case_id"],
            fragment_id=raw_ref["fragment_id"],
            expected_output_count=raw_ref["expected_output_count"],
        )
        for raw_ref in raw_case["fragment_refs"]
    )
    if not fragment_refs:
        raise ValueError(
            f"derived-support case {raw_case['case_id']!r} needs fragments"
        )
    sentinel_outputs = tuple(raw_case["sentinel_outputs"])
    if len(set(sentinel_outputs)) != len(sentinel_outputs):
        raise ValueError(
            f"duplicate sentinel output in case {raw_case['case_id']!r}"
        )
    return SouthStarDerivedSupportCase(
        case_id=raw_case["case_id"],
        source_smiles=raw_case["source_smiles"],
        support_authority=raw_case["support_authority"],
        evidence_notes=raw_case["evidence_notes"],
        fragment_refs=fragment_refs,
        fragment_order_policy=raw_case["fragment_order_policy"],
        output_order_policy=raw_case["output_order_policy"],
        expected_product_count=raw_case["expected_product_count"],
        expected_digest_sha256=raw_case["expected_digest_sha256"],
        sentinel_outputs=sentinel_outputs,
    )


def _support_for_fragment_ref(
    fragment_ref: SouthStarDerivedFragmentRef,
) -> tuple[str, ...]:
    if fragment_ref.fixture_family == "south_star_expanded_support":
        cases = load_south_star_expanded_support_cases()
    elif fragment_ref.fixture_family == "south_star_exact_first_domain":
        cases = load_south_star_exact_first_domain_cases()
    else:
        raise ValueError(
            f"unsupported fragment fixture family {fragment_ref.fixture_family!r}"
        )

    matches = tuple(case for case in cases if case.case_id == fragment_ref.case_id)
    if len(matches) != 1:
        raise ValueError(
            f"expected one fragment case {fragment_ref.case_id!r}, "
            f"found {len(matches)}"
        )
    outputs = matches[0].expected_support
    if len(outputs) != fragment_ref.expected_output_count:
        raise ValueError(
            f"fragment {fragment_ref.case_id!r} expected "
            f"{fragment_ref.expected_output_count} outputs, found {len(outputs)}"
        )
    return outputs
