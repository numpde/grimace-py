from __future__ import annotations

"""Classify RDKit sampled marker placements for manual diene witnesses."""

from collections import Counter
from dataclasses import dataclass

from rdkit import Chem, rdBase

from grimace import _core, _runtime


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


@dataclass(frozen=True, slots=True)
class SideChoice:
    component_idx: int
    side_idx: int
    endpoint_atom_idx: int
    neighbor_idx: int
    base_token: str


@dataclass(frozen=True, slots=True)
class VisibleMarker:
    edge: tuple[int, int]
    marker: str


CASES = (
    ManualDifficultCase(
        case_id="manual_bond_stereo_difficult_cis_cis",
        smiles="CC/C=C\\C(CO)=C(/C)CC",
    ),
    ManualDifficultCase(
        case_id="manual_bond_stereo_difficult_cis_trans",
        smiles="CC/C=C\\C(CO)=C(\\C)CC",
    ),
    ManualDifficultCase(
        case_id="manual_bond_stereo_difficult_trans_cis",
        smiles="CC/C=C/C(CO)=C(\\C)CC",
    ),
    ManualDifficultCase(
        case_id="manual_bond_stereo_difficult_trans_trans",
        smiles="CC/C=C/C(CO)=C(/C)CC",
    ),
)


def flip_marker(marker: str) -> str:
    if marker == "/":
        return "\\"
    if marker == "\\":
        return "/"
    raise ValueError(f"not a directional marker: {marker!r}")


def marker_from_bond_dir(bond_dir: Chem.BondDir) -> str | None:
    if bond_dir == Chem.BondDir.ENDUPRIGHT:
        return "/"
    if bond_dir == Chem.BondDir.ENDDOWNRIGHT:
        return "\\"
    return None


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


def side_choices(mol: Chem.Mol) -> tuple[SideChoice, ...]:
    prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
    summary = _core._stereo_constraint_model_summary(prepared)
    choices: list[SideChoice] = []
    for component in summary["components"]:
        for side in component["side_domains"]:
            for choice in side["choices"]:
                choices.append(
                    SideChoice(
                        component_idx=int(component["component_idx"]),
                        side_idx=int(side["side_idx"]),
                        endpoint_atom_idx=int(side["endpoint_atom_idx"]),
                        neighbor_idx=int(choice["neighbor_idx"]),
                        base_token=str(choice["base_token"]),
                    )
                )
    return tuple(choices)


def visible_markers(mol: Chem.Mol) -> tuple[VisibleMarker, ...]:
    markers = []
    for bond in mol.GetBonds():
        marker = marker_from_bond_dir(bond.GetBondDir())
        if marker is None:
            continue
        markers.append(
            VisibleMarker(
                edge=tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))),
                marker=marker,
            )
        )
    return tuple(markers)


def selected_edges_by_component(
    mol: Chem.Mol,
    choices: tuple[SideChoice, ...],
) -> dict[int, set[tuple[int, int]]]:
    out: dict[int, set[tuple[int, int]]] = {}
    for bond in mol.GetBonds():
        stereo_atoms = tuple(bond.GetStereoAtoms())
        if len(stereo_atoms) != 2:
            continue
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        for endpoint_idx, neighbor_idx in (
            (begin_idx, stereo_atoms[0]),
            (end_idx, stereo_atoms[1]),
        ):
            matching = [
                choice
                for choice in choices
                if choice.endpoint_atom_idx == endpoint_idx
                and choice.neighbor_idx == neighbor_idx
            ]
            if len(matching) != 1:
                continue
            out.setdefault(matching[0].component_idx, set()).add(
                tuple(sorted((endpoint_idx, neighbor_idx)))
            )
    return out


def side_choice_by_edge(
    choices: tuple[SideChoice, ...],
) -> dict[tuple[int, int], tuple[SideChoice, ...]]:
    out: dict[tuple[int, int], list[SideChoice]] = {}
    for choice in choices:
        out.setdefault(
            tuple(sorted((choice.endpoint_atom_idx, choice.neighbor_idx))),
            [],
        ).append(choice)
    return {edge: tuple(edge_choices) for edge, edge_choices in out.items()}


def classify_marker(
    marker: VisibleMarker,
    choices_for_edge: tuple[SideChoice, ...],
    selected_by_component: dict[int, set[tuple[int, int]]],
) -> tuple[str, ...]:
    rows = []
    for choice in choices_for_edge:
        selected = marker.edge in selected_by_component.get(choice.component_idx, set())
        relation = "selected" if selected else "complement_or_unselected"
        edge_begin_idx, edge_end_idx = marker.edge
        physical_basis = physical_token_for_edge(
            endpoint_atom_idx=choice.endpoint_atom_idx,
            edge_begin_idx=edge_begin_idx,
            edge_end_idx=edge_end_idx,
            model_basis_token=choice.base_token,
        )
        rows.append(
            ":".join(
                (
                    f"component{choice.component_idx}",
                    f"side{choice.side_idx}",
                    relation,
                    f"physical_{marker_flip_name(marker.marker, physical_basis)}",
                    f"basis_{marker_flip_name(marker.marker, choice.base_token)}",
                )
            )
        )
    if not rows:
        return ("not_candidate",)
    return tuple(rows)


def sampled_outputs(mol: Chem.Mol, *, count: int, seed: int) -> tuple[str, ...]:
    outputs = Chem.MolToRandomSmilesVect(
        Chem.Mol(mol),
        count,
        randomSeed=seed,
        isomericSmiles=True,
    )
    return tuple(dict.fromkeys(outputs))


def main() -> None:
    print(f"RDKit version: {rdBase.rdkitVersion}")
    for case in CASES:
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None:
            raise ValueError(f"RDKit failed to parse {case.case_id}")

        outputs = (
            Chem.MolToSmiles(
                Chem.Mol(mol),
                isomericSmiles=True,
                canonical=True,
                doRandom=False,
            ),
            *sampled_outputs(mol, count=64, seed=1),
            *sampled_outputs(mol, count=64, seed=17),
        )
        output_counts = Counter[str]()
        marker_counts = Counter[str]()
        output_marker_profiles = Counter[tuple[str, ...]]()
        model_errors = Counter[str]()

        for smiles in dict.fromkeys(outputs):
            parsed = Chem.MolFromSmiles(smiles)
            if parsed is None:
                raise ValueError(f"RDKit failed to reparse sampled output: {smiles}")
            try:
                choices = side_choices(parsed)
            except ValueError as exc:
                model_errors[str(exc)] += 1
                continue
            selected = selected_edges_by_component(parsed, choices)
            choices_by_edge = side_choice_by_edge(choices)
            marker_profile: list[str] = []
            for marker in visible_markers(parsed):
                classifications = classify_marker(
                    marker,
                    choices_by_edge.get(marker.edge, ()),
                    selected,
                )
                for classification in classifications:
                    marker_counts[classification] += 1
                marker_profile.extend(classifications)
            output_counts[smiles] += 1
            output_marker_profiles[tuple(sorted(marker_profile))] += 1

        print()
        print(case.case_id)
        print(f"  unique sampled outputs: {len(output_counts)}")
        if model_errors:
            print("  Grimace model-construction errors:")
            for error, count in sorted(model_errors.items()):
                print(f"    {count} x {error}")
        print("  marker classification counts:")
        for classification, count in sorted(marker_counts.items()):
            print(f"    {classification}: {count}")
        print("  output marker profiles:")
        for profile, count in output_marker_profiles.most_common(8):
            print(f"    {count} x {profile}")


if __name__ == "__main__":
    main()
