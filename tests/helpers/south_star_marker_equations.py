from __future__ import annotations

from grimace._south_star.component_support_state import (
    SouthStarComponentSupportState,
)
from grimace._south_star.enum_s import mol_to_smiles_enum_s_tree_traversals_for_case
from grimace._south_star.marker_equations import SouthStarMarkerSlotParityEquation
from grimace._south_star.marker_equations import (
    marker_slot_parity_equations_for_traversal,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles


def marker_slot_parity_equations_for_case(
    case: object,
) -> tuple[tuple[SouthStarMarkerSlotParityEquation, ...], ...]:
    mol = parse_smiles(case.source_smiles)
    state = SouthStarComponentSupportState.from_mol(mol)
    return tuple(
        marker_slot_parity_equations_for_traversal(state, traversal)
        for traversal in mol_to_smiles_enum_s_tree_traversals_for_case(case)
    )
