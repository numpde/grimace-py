"""Optional RDKit audit helpers for the private proof kernel.

Audit code may compare generated strings against external parser behavior in
tests, but it is not part of support definition or enumeration.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from typing import Literal

from rdkit import Chem

from .errors import SouthStarErrorKind
from .fact_isomorphism import FactIsomorphismResult
from .fact_isomorphism import facts_are_isomorphic
from .ordinary_policy import OrdinaryPolicyOptions
from .ordinary_policy import ordinary_policy_for_facts
from .ordinary_semantics import OrdinarySmilesSemantics
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
    "SOUTH_STAR1_SUPPORTED_V0_AUDIT_CASES",
    "SOUTH_STAR1_UNSUPPORTED_V0_AUDIT_CASES",
    "audit_generated_support_with_rdkit",
    "audit_generated_witnesses_with_rdkit",
    "summarize_rdkit_audit",
)
