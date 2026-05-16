from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, rdBase

from grimace import _core, _runtime
from tests.rdkit_serialization._support import grimace_support


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
class ManualDifficultCase:
    case_id: str
    smiles: str
    expected_bridge_marker: str


CASES = (
    ManualDifficultCase(
        case_id="manual_bond_stereo_difficult_cis_cis",
        smiles="CC/C=C\\C(CO)=C(/C)CC",
        expected_bridge_marker="\\",
    ),
    ManualDifficultCase(
        case_id="manual_bond_stereo_difficult_cis_trans",
        smiles="CC/C=C\\C(CO)=C(\\C)CC",
        expected_bridge_marker="\\",
    ),
    ManualDifficultCase(
        case_id="manual_bond_stereo_difficult_trans_cis",
        smiles="CC/C=C/C(CO)=C(\\C)CC",
        expected_bridge_marker="/",
    ),
    ManualDifficultCase(
        case_id="manual_bond_stereo_difficult_trans_trans",
        smiles="CC/C=C/C(CO)=C(/C)CC",
        expected_bridge_marker="/",
    ),
)


def flip_marker(marker: str) -> str:
    if marker == "/":
        return "\\"
    if marker == "\\":
        return "/"
    raise ValueError(f"not a directional marker: {marker!r}")


def marker_flip_name(chosen: str, basis: str) -> str:
    if chosen == basis:
        return "stored"
    if chosen == flip_marker(basis):
        return "flipped"
    return "incompatible"


def physical_token_for_edge(
    *,
    endpoint_atom_idx: int,
    edge_begin_idx: int,
    edge_end_idx: int,
    model_basis_token: str,
) -> str:
    if endpoint_atom_idx == edge_begin_idx:
        return model_basis_token
    if endpoint_atom_idx == edge_end_idx:
        return flip_marker(model_basis_token)
    raise ValueError("side endpoint is not on the emitted edge")


def bridge_side_rows(mol: Chem.Mol, edge_begin_idx: int, edge_end_idx: int) -> list[dict[str, object]]:
    prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
    summary = _core._stereo_constraint_model_summary(prepared)
    rows: list[dict[str, object]] = []
    edge_atoms = {edge_begin_idx, edge_end_idx}
    for component in summary["components"]:
        for side in component["side_domains"]:
            endpoint = int(side["endpoint_atom_idx"])
            for choice in side["choices"]:
                neighbor = int(choice["neighbor_idx"])
                if {endpoint, neighbor} != edge_atoms:
                    continue
                model_basis = str(choice["base_token"])
                physical = physical_token_for_edge(
                    endpoint_atom_idx=endpoint,
                    edge_begin_idx=edge_begin_idx,
                    edge_end_idx=edge_end_idx,
                    model_basis_token=model_basis,
                )
                rows.append(
                    {
                        "component_idx": int(component["component_idx"]),
                        "side_idx": int(side["side_idx"]),
                        "endpoint_atom_idx": endpoint,
                        "neighbor_idx": neighbor,
                        "model_basis_token": model_basis,
                        "physical_emitted_token": physical,
                    }
                )
    return rows


def main() -> None:
    print(f"RDKit version: {rdBase.rdkitVersion}")
    print("Bridge edge is emitted as atom 3 -> atom 4 along the canonical witness path.")
    for case in CASES:
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None:
            raise ValueError(f"RDKit failed to parse {case.case_id}")
        expected = Chem.MolToSmiles(
            Chem.Mol(mol),
            isomericSmiles=True,
            canonical=True,
            doRandom=False,
        )
        support = grimace_support(mol, rooted_at_atom=None, isomeric_smiles=True)
        print()
        print(case.case_id)
        print(f"  source:   {case.smiles}")
        print(f"  expected: {expected}")
        print(
            "  grimace:  "
            f"{'contains' if expected in support else 'missing'} "
            f"(support size {len(support)})"
        )
        print(f"  expected bridge marker: {case.expected_bridge_marker}")
        for row in bridge_side_rows(mol, 3, 4):
            physical_flip = marker_flip_name(
                case.expected_bridge_marker,
                str(row["physical_emitted_token"]),
            )
            basis_flip = marker_flip_name(
                case.expected_bridge_marker,
                str(row["model_basis_token"]),
            )
            print(
                "  bridge side: "
                f"component={row['component_idx']} side={row['side_idx']} "
                f"endpoint={row['endpoint_atom_idx']} neighbor={row['neighbor_idx']} "
                f"basis={row['model_basis_token']} physical={row['physical_emitted_token']} "
                f"marker_vs_physical={physical_flip} marker_vs_basis={basis_flip}"
            )


if __name__ == "__main__":
    main()
