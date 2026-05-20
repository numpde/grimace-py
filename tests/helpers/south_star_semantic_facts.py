from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.annotation_policy import (
    DIRECTIONAL_MARKERS,
    SemanticCarrierOpportunity,
    SurvivingSemanticAssignment,
    normalized_edge,
)
from grimace._south_star.components import SouthStarSemanticStereoComponent
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.support_gates import SouthStarSupportGateReport
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import SouthStarSemanticCase


@dataclass(frozen=True, slots=True)
class SouthStarSemanticFacts:
    case_id: str
    source_smiles: str
    molecule_facts: SouthStarMoleculeFacts

    @property
    def components(self) -> tuple[SouthStarSemanticStereoComponent, ...]:
        return self.molecule_facts.components

    @property
    def carrier_opportunities(self) -> tuple[SemanticCarrierOpportunity, ...]:
        return self.molecule_facts.carrier_opportunities

    @property
    def support_gate_report(self) -> SouthStarSupportGateReport:
        return self.molecule_facts.support_gate_report


def south_star_semantic_facts_from_case(
    case: SouthStarSemanticCase,
) -> SouthStarSemanticFacts:
    molecule_facts = SouthStarMoleculeFacts.from_mol(parse_smiles(case.source_smiles))
    molecule_facts.fail_if_unsupported()
    return SouthStarSemanticFacts(
        case_id=case.case_id,
        source_smiles=case.source_smiles,
        molecule_facts=molecule_facts,
    )


def prototype_surviving_assignments_from_case(
    case: SouthStarSemanticCase,
) -> tuple[SurvivingSemanticAssignment, ...]:
    return (
        SurvivingSemanticAssignment(
            assignment_id=f"{case.case_id}:prototype-all-marker-survivor",
            marker_options_by_edge={
                normalized_edge(edge): DIRECTIONAL_MARKERS
                for edge in case.eligible_carrier_edges
            },
        ),
    )
