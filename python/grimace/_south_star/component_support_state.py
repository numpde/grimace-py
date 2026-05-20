from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from operator import mul

from rdkit import Chem

from grimace._south_star.annotation_policy import (
    DIRECTIONAL_MARKERS,
    AnnotationPolicy,
    Edge,
    EmittedEdgeBasis,
    MaximalEligibleCarrierAnnotationPolicy,
    SemanticCarrierOpportunity,
    SurvivingSemanticAssignment,
    normalized_edge,
)
from grimace._south_star.components import (
    SouthStarSemanticStereoComponent,
)
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts


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
class SouthStarComponentMarkerAssignment:
    component_id: str
    assignment_id: str
    marker_by_edge: tuple[tuple[Edge, str], ...]


@dataclass(frozen=True, slots=True)
class SouthStarComponentSupportState:
    molecule_facts: SouthStarMoleculeFacts
    annotation_policy: AnnotationPolicy

    @classmethod
    def from_case(
        cls,
        case: object,
        *,
        annotation_policy: AnnotationPolicy | None = None,
    ) -> SouthStarComponentSupportState:
        return cls.from_mol(
            _parse_smiles(case.source_smiles),
            annotation_policy=annotation_policy,
        )

    @classmethod
    def from_mol(
        cls,
        mol: Chem.Mol,
        *,
        annotation_policy: AnnotationPolicy | None = None,
    ) -> SouthStarComponentSupportState:
        return cls.from_molecule_facts(
            SouthStarMoleculeFacts.from_mol(mol),
            annotation_policy=annotation_policy,
        )

    @classmethod
    def from_molecule_facts(
        cls,
        molecule_facts: SouthStarMoleculeFacts,
        *,
        annotation_policy: AnnotationPolicy | None = None,
    ) -> SouthStarComponentSupportState:
        molecule_facts.fail_if_unsupported()
        return cls(
            molecule_facts=molecule_facts,
            annotation_policy=annotation_policy
            or MaximalEligibleCarrierAnnotationPolicy(),
        )

    @property
    def components(self) -> tuple[SouthStarSemanticStereoComponent, ...]:
        return self.molecule_facts.components

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
            surviving_assignments=_surviving_assignments_for_components(affected),
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
                local_survivor_count=_local_survivor_count(
                    component,
                    observations={normalized: marker},
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

    def explain_directional_marker_observations(
        self,
        observations: Mapping[Edge, str],
    ) -> tuple[SouthStarAffectedComponentSupport, ...]:
        normalized_observations = _normalized_marker_observations(observations)
        affected = tuple(
            component
            for component in self.components
            if any(
                edge in component.eligible_carrier_edges
                for edge in normalized_observations
            )
        )
        return tuple(
            SouthStarAffectedComponentSupport(
                component_id=component.component_id,
                local_survivor_count=_local_survivor_count(
                    component,
                    observations=normalized_observations,
                ),
            )
            for component in affected
        )

    def component_marker_assignments(
        self,
    ) -> tuple[tuple[SouthStarComponentMarkerAssignment, ...], ...]:
        return tuple(
            tuple(
                SouthStarComponentMarkerAssignment(
                    component_id=component.component_id,
                    assignment_id=assignment.assignment_id,
                    marker_by_edge=tuple(
                        (edge, marker_options[0])
                        for edge, marker_options in sorted(
                            assignment.marker_options_by_edge.items()
                        )
                    ),
                )
                for assignment in _component_marker_assignments(component)
            )
            for component in self.components
        )

    def complexity_snapshot(self) -> SouthStarComponentComplexitySnapshot:
        estimates = tuple(
            SouthStarComponentAssignmentEstimate(
                component_id=component.component_id,
                source_feature_count=len(component.source_features),
                eligible_carrier_count=len(component.eligible_carrier_edges),
                coupling_cause_count=len(component.coupling_causes),
                estimated_local_assignment_count=(
                    len(_component_marker_assignments(component))
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


def _local_survivor_count(
    component: SouthStarSemanticStereoComponent,
    *,
    observations: Mapping[Edge, str],
) -> int:
    return sum(
        1
        for assignment in _component_marker_assignments(component)
        if all(
            assignment.marker_options_by_edge.get(edge) == (marker,)
            for edge, marker in observations.items()
            if edge in component.eligible_carrier_edges
        )
    )


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


def _surviving_assignments_for_components(
    components: tuple[SouthStarSemanticStereoComponent, ...],
) -> tuple[SurvivingSemanticAssignment, ...]:
    return tuple(
        assignment
        for component in components
        for assignment in _component_marker_assignments(component)
    )


def _component_marker_assignments(
    component: SouthStarSemanticStereoComponent,
) -> tuple[SurvivingSemanticAssignment, ...]:
    source_markers = _component_source_markers(component)
    return (
        SurvivingSemanticAssignment(
            assignment_id=f"{component.component_id}:source",
            marker_options_by_edge={
                edge: (marker,) for edge, marker in source_markers.items()
            },
        ),
        SurvivingSemanticAssignment(
            assignment_id=f"{component.component_id}:global_flip",
            marker_options_by_edge={
                edge: (_flipped_marker(marker),)
                for edge, marker in source_markers.items()
            },
        ),
    )


def _component_source_markers(
    component: SouthStarSemanticStereoComponent,
) -> dict[Edge, str]:
    source_markers: dict[Edge, str] = {}
    for feature in component.source_features:
        for edge, marker in feature.source_marker_by_edge:
            existing = source_markers.setdefault(edge, marker)
            if existing != marker:
                raise ValueError(
                    f"component {component.component_id!r} has conflicting source "
                    f"markers for carrier edge {edge!r}: {existing!r} vs {marker!r}"
                )
    missing_edges = tuple(
        edge for edge in component.eligible_carrier_edges if edge not in source_markers
    )
    if missing_edges:
        raise ValueError(
            f"component {component.component_id!r} has no source marker for "
            f"carrier edges {missing_edges!r}"
        )
    return source_markers


def _flipped_marker(marker: str) -> str:
    if marker == "/":
        return "\\"
    if marker == "\\":
        return "/"
    raise ValueError(f"directional marker must be one of {DIRECTIONAL_MARKERS}")


def _normalized_marker_observations(
    observations: Mapping[Edge, str],
) -> dict[Edge, str]:
    normalized_observations: dict[Edge, str] = {}
    for edge, marker in observations.items():
        if marker not in DIRECTIONAL_MARKERS:
            raise ValueError(f"directional marker must be one of {DIRECTIONAL_MARKERS}")
        normalized = normalized_edge(edge)
        existing = normalized_observations.setdefault(normalized, marker)
        if existing != marker:
            raise ValueError(
                f"conflicting marker observations for carrier edge {normalized!r}"
            )
    return normalized_observations


def _parse_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"failed to parse SMILES {smiles!r}")
    return mol
