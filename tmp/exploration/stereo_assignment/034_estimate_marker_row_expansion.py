from __future__ import annotations

"""Estimate explicit marker-placement row growth on pinned stereo cases."""

from collections import Counter
from dataclasses import dataclass

from rdkit import Chem, rdBase

from grimace import _core, _runtime
from tests.helpers.rdkit_writer_membership import load_pinned_writer_membership_cases
from tests.rdkit_serialization._support import mol_from_pinned_source


SUPPORTED_STEREO_FLAGS = _runtime.MolToSmilesFlags(
    isomeric_smiles=True,
    kekule_smiles=False,
    rooted_at_atom=-1,
    canonical=False,
    all_bonds_explicit=False,
    all_hs_explicit=False,
    do_random=True,
    ignore_atom_map_numbers=False,
)


@dataclass(frozen=True, slots=True)
class ComponentEstimate:
    case_id: str
    component_idx: int
    side_domain_product: int
    runtime_component_count: int
    current_token_phase_rows: int
    two_candidate_side_count: int
    marker_placement_rows: int


MANUAL_DIFFICULT_SMILES = {
    "known_gap_manual_bond_stereo_difficult_cis_cis": "CC/C=C\\C(CO)=C(/C)CC",
    "known_gap_manual_bond_stereo_difficult_cis_trans": "CC/C=C\\C(CO)=C(\\C)CC",
    "known_gap_manual_bond_stereo_difficult_trans_cis": "CC/C=C/C(CO)=C(\\C)CC",
    "known_gap_manual_bond_stereo_difficult_trans_trans": "CC/C=C/C(CO)=C(/C)CC",
}


def component_estimates(case_id: str, mol: Chem.Mol) -> tuple[ComponentEstimate, ...]:
    prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
    summary = _core._stereo_constraint_model_summary(prepared)
    estimates = []
    for component in summary["components"]:
        component_idx = int(component["component_idx"])
        side_domain_sizes = tuple(int(value) for value in component["side_domain_sizes"])
        side_domain_product = 1
        for size in side_domain_sizes:
            side_domain_product *= size
        runtime_component_count = int(summary["component_sizes"][component_idx])
        current_token_phase_rows = side_domain_product * (2**runtime_component_count)
        two_candidate_side_count = sum(1 for size in side_domain_sizes if size == 2)
        marker_placement_rows = current_token_phase_rows * (2**two_candidate_side_count)
        estimates.append(
            ComponentEstimate(
                case_id=case_id,
                component_idx=component_idx,
                side_domain_product=side_domain_product,
                runtime_component_count=runtime_component_count,
                current_token_phase_rows=current_token_phase_rows,
                two_candidate_side_count=two_candidate_side_count,
                marker_placement_rows=marker_placement_rows,
            )
        )
    return tuple(estimates)


def main() -> None:
    print(f"RDKit version: {rdBase.rdkitVersion}")
    estimates: list[ComponentEstimate] = []
    skipped = Counter[str]()
    for case in load_pinned_writer_membership_cases(rdBase.rdkitVersion):
        if not case.isomeric_smiles:
            continue
        try:
            estimates.extend(component_estimates(case.case_id, mol_from_pinned_source(case)))
        except (AssertionError, NotImplementedError, ValueError) as exc:
            skipped[type(exc).__name__ + ": " + str(exc)] += 1
    for case_id, smiles in MANUAL_DIFFICULT_SMILES.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped[f"parse failed: {case_id}"] += 1
            continue
        try:
            estimates.extend(component_estimates(case_id, mol))
        except (AssertionError, NotImplementedError, ValueError) as exc:
            skipped[type(exc).__name__ + ": " + str(exc)] += 1

    estimates.sort(key=lambda row: row.marker_placement_rows, reverse=True)
    print(f"components estimated: {len(estimates)}")
    if skipped:
        print("skipped:")
        for reason, count in skipped.most_common():
            print(f"  {count} x {reason}")
    print()
    print("largest estimated marker-placement row counts:")
    for row in estimates[:20]:
        print(
            "  {case_id} component={component} side_product={side_product} "
            "runtime_components={runtime_components} current_rows={current_rows} "
            "two_candidate_sides={two_sides} marker_rows={marker_rows}".format(
                case_id=row.case_id,
                component=row.component_idx,
                side_product=row.side_domain_product,
                runtime_components=row.runtime_component_count,
                current_rows=row.current_token_phase_rows,
                two_sides=row.two_candidate_side_count,
                marker_rows=row.marker_placement_rows,
            )
        )


if __name__ == "__main__":
    main()
