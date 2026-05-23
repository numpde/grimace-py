"""Optional RDKit audit helpers for the private proof kernel.

Audit code may compare generated strings against external parser behavior in
tests, but it is not part of support definition or enumeration.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rdkit import Chem

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
    "RdkitAuditResult",
    "RdkitWitnessAuditResult",
    "audit_generated_support_with_rdkit",
    "audit_generated_witnesses_with_rdkit",
)
