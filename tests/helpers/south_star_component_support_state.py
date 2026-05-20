from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from tests.helpers.south_star_annotation_policy import (
    DIRECTIONAL_MARKERS,
    Edge,
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
class SouthStarComponentSupportState:
    components: tuple[SouthStarSemanticStereoComponent, ...]

    @classmethod
    def from_case(
        cls,
        case: SouthStarSemanticCase,
    ) -> SouthStarComponentSupportState:
        return cls.from_mol(parse_smiles(case.source_smiles))

    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> SouthStarComponentSupportState:
        extraction = extract_south_star_components(mol)
        extraction.fail_if_unsupported()
        return cls(components=extraction.components)

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
