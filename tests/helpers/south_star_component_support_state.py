from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul

from rdkit import Chem

from tests.helpers.south_star_annotation_policy import (
    DIRECTIONAL_MARKERS,
    AnnotationPolicy,
    Edge,
    EmittedEdgeBasis,
    MaximalEligibleCarrierAnnotationPolicy,
    SemanticCarrierOpportunity,
    SurvivingSemanticAssignment,
    normalized_edge,
)
from tests.helpers.south_star_components import (
    SouthStarSemanticStereoComponent,
    extract_south_star_components,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import SouthStarSemanticCase


@dataclass(frozen=True, slots=True)
class SouthStarAffectedComponentSupport:
    component_id: str
    local_survivor_count: int


@dataclass(frozen=True, slots=True)
class SouthStarComponentMarkerSupport:
    edge: Edge
    marker: str
    token_allowed: bool
    reason: str
    affected_components: tuple[SouthStarAffectedComponentSupport, ...]


@dataclass(frozen=True, slots=True)
class SouthStarComponentAssignmentEstimate:
    component_id: str
    source_feature_count: int
    eligible_carrier_count: int
    coupling_cause_count: int
    estimated_local_assignment_count: int


@dataclass(frozen=True, slots=True)
class SouthStarComponentComplexitySnapshot:
    component_count: int
    local_assignment_estimates: tuple[SouthStarComponentAssignmentEstimate, ...]
    estimated_product_size: int


@dataclass(frozen=True, slots=True)
class SouthStarComponentSupportState:
    components: tuple[SouthStarSemanticStereoComponent, ...]
    annotation_policy: AnnotationPolicy

    @classmethod
    def from_case(
        cls,
        case: SouthStarSemanticCase,
        *,
        annotation_policy: AnnotationPolicy | None = None,
    ) -> SouthStarComponentSupportState:
        return cls.from_mol(
            parse_smiles(case.source_smiles),
            annotation_policy=annotation_policy,
        )

    @classmethod
    def from_mol(
        cls,
        mol: Chem.Mol,
        *,
        annotation_policy: AnnotationPolicy | None = None,
    ) -> SouthStarComponentSupportState:
        extraction = extract_south_star_components(mol)
        extraction.fail_if_unsupported()
        return cls(
            components=extraction.components,
            annotation_policy=annotation_policy
            or MaximalEligibleCarrierAnnotationPolicy(),
        )

    def explain_directional_marker(
        self,
        *,
        edge: Edge,
        marker: str,
    ) -> SouthStarComponentMarkerSupport:
        if marker not in DIRECTIONAL_MARKERS:
            raise ValueError(f"directional marker must be one of {DIRECTIONAL_MARKERS}")

        normalized = normalized_edge(edge)
        affected = tuple(
            component
            for component in self.components
            if normalized in component.eligible_carrier_edges
        )
        if not affected:
            return SouthStarComponentMarkerSupport(
                edge=normalized,
                marker=marker,
                token_allowed=False,
                reason="edge_affects_no_semantic_component",
                affected_components=(),
            )

        policy_decision = self.annotation_policy.decision(
            carrier_opportunities=_carrier_opportunities_for_components(affected),
            emitted_edge=EmittedEdgeBasis(edge=normalized),
            surviving_assignments=_prototype_surviving_assignments_for_components(
                affected
            ),
        )
        if not policy_decision.marker_required:
            return SouthStarComponentMarkerSupport(
                edge=normalized,
                marker=marker,
                token_allowed=False,
                reason="marker_not_required_by_annotation_policy",
                affected_components=(),
            )
        if marker not in policy_decision.allowed_markers:
            return SouthStarComponentMarkerSupport(
                edge=normalized,
                marker=marker,
                token_allowed=False,
                reason="marker_rejected_by_annotation_policy",
                affected_components=(),
            )

        affected_support = tuple(
            SouthStarAffectedComponentSupport(
                component_id=component.component_id,
                local_survivor_count=_prototype_local_survivor_count(
                    component,
                    edge=normalized,
                    marker=marker,
                ),
            )
            for component in affected
        )
        if any(support.local_survivor_count == 0 for support in affected_support):
            return SouthStarComponentMarkerSupport(
                edge=normalized,
                marker=marker,
                token_allowed=False,
                reason="affected_component_has_no_semantic_survivor",
                affected_components=affected_support,
            )

        return SouthStarComponentMarkerSupport(
            edge=normalized,
            marker=marker,
            token_allowed=True,
            reason="all_affected_components_have_semantic_survivors",
            affected_components=affected_support,
        )

    def allowed_directional_markers(self, *, edge: Edge) -> tuple[str, ...]:
        return tuple(
            marker
            for marker in DIRECTIONAL_MARKERS
            if self.explain_directional_marker(edge=edge, marker=marker).token_allowed
        )

    def complexity_snapshot(self) -> SouthStarComponentComplexitySnapshot:
        estimates = tuple(
            SouthStarComponentAssignmentEstimate(
                component_id=component.component_id,
                source_feature_count=len(component.source_features),
                eligible_carrier_count=len(component.eligible_carrier_edges),
                coupling_cause_count=len(component.coupling_causes),
                estimated_local_assignment_count=(
                    _prototype_component_assignment_estimate(component)
                ),
            )
            for component in self.components
        )
        return SouthStarComponentComplexitySnapshot(
            component_count=len(self.components),
            local_assignment_estimates=estimates,
            estimated_product_size=reduce(
                mul,
                (
                    estimate.estimated_local_assignment_count
                    for estimate in estimates
                ),
                1,
            ),
        )


def _prototype_local_survivor_count(
    component: SouthStarSemanticStereoComponent,
    *,
    edge: Edge,
    marker: str,
) -> int:
    if edge not in component.eligible_carrier_edges:
        return 0
    if marker not in DIRECTIONAL_MARKERS:
        return 0
    return 1


def _prototype_component_assignment_estimate(
    component: SouthStarSemanticStereoComponent,
) -> int:
    return 2 ** len(component.source_features)


def _carrier_opportunities_for_components(
    components: tuple[SouthStarSemanticStereoComponent, ...],
) -> tuple[SemanticCarrierOpportunity, ...]:
    return tuple(
        SemanticCarrierOpportunity(edge=edge)
        for edge in dict.fromkeys(
            edge
            for component in components
            for edge in component.eligible_carrier_edges
        )
    )


def _prototype_surviving_assignments_for_components(
    components: tuple[SouthStarSemanticStereoComponent, ...],
) -> tuple[SurvivingSemanticAssignment, ...]:
    return tuple(
        SurvivingSemanticAssignment(
            assignment_id=component.component_id,
            marker_options_by_edge={
                edge: DIRECTIONAL_MARKERS
                for edge in component.eligible_carrier_edges
            },
        )
        for component in components
    )
