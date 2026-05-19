from __future__ import annotations

"""Compare shared visible-marker interpretations on manual diene witnesses."""

from dataclasses import dataclass
from enum import Enum

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


class MarkerPolicy(str, Enum):
    VISIBLE_EDGE_SELECTS_CARRIER = "visible-edge-selects-carrier"
    VISIBLE_EDGE_IS_TOKEN_FLIP_FACT = "visible-edge-is-token-flip-fact"
    VISIBLE_EDGE_IS_MARKER_OBLIGATION = "visible-edge-is-marker-obligation"


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
    bond_dir: str


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
                bond_dir=str(bond.GetBondDir()).removeprefix("BondDir."),
            )
        )
    return tuple(markers)


def stereo_selected_edges_by_component(
    mol: Chem.Mol,
    choices: tuple[SideChoice, ...],
) -> dict[int, set[tuple[int, int]]]:
    edges_by_component: dict[int, set[tuple[int, int]]] = {}
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
            matching_choices = [
                choice
                for choice in choices
                if choice.endpoint_atom_idx == endpoint_idx
                and choice.neighbor_idx == neighbor_idx
            ]
            if len(matching_choices) != 1:
                raise ValueError(
                    "could not map RDKit selected stereo edge to exactly one "
                    f"Grimace side choice: {endpoint_idx}-{neighbor_idx}"
                )
            choice = matching_choices[0]
            edges_by_component.setdefault(choice.component_idx, set()).add(
                tuple(sorted((endpoint_idx, neighbor_idx)))
            )
    return edges_by_component


def matching_choices_for_edge(
    choices: tuple[SideChoice, ...],
    edge: tuple[int, int],
) -> tuple[SideChoice, ...]:
    edge_set = set(edge)
    return tuple(
        choice
        for choice in choices
        if {choice.endpoint_atom_idx, choice.neighbor_idx} == edge_set
    )


def describe_policy_implications(
    policy: MarkerPolicy,
    marker: VisibleMarker,
    choices: tuple[SideChoice, ...],
    selected_edges_by_component: dict[int, set[tuple[int, int]]],
) -> tuple[str, ...]:
    matching = matching_choices_for_edge(choices, marker.edge)
    if not matching:
        return ("not a stereo carrier candidate edge",)

    rows: list[str] = []
    for choice in matching:
        edge_begin_idx, edge_end_idx = marker.edge
        physical_basis = physical_token_for_edge(
            endpoint_atom_idx=choice.endpoint_atom_idx,
            edge_begin_idx=edge_begin_idx,
            edge_end_idx=edge_end_idx,
            model_basis_token=choice.base_token,
        )
        selected_by_rdkit = (
            marker.edge in selected_edges_by_component.get(choice.component_idx, set())
        )
        physical_flip = marker_flip_name(marker.marker, physical_basis)
        model_basis_flip = marker_flip_name(marker.marker, choice.base_token)
        if policy == MarkerPolicy.VISIBLE_EDGE_SELECTS_CARRIER:
            implication = (
                "would force selected carrier edge"
                if selected_by_rdkit
                else "would contradict RDKit selected stereo atoms"
            )
        elif policy == MarkerPolicy.VISIBLE_EDGE_IS_TOKEN_FLIP_FACT:
            implication = f"would emit immediate physical-basis token flip={physical_flip}"
        else:
            implication = (
                "would record marker obligation "
                f"(physical_flip={physical_flip}, model_basis_flip={model_basis_flip})"
            )
        rows.append(
            "component={component} side={side} endpoint={endpoint} neighbor={neighbor} "
            "base={base} physical_basis={physical_basis} marker={marker} "
            "rdkit_selected_edge={selected}: {implication}".format(
                component=choice.component_idx,
                side=choice.side_idx,
                endpoint=choice.endpoint_atom_idx,
                neighbor=choice.neighbor_idx,
                base=choice.base_token,
                physical_basis=physical_basis,
                marker=marker.marker,
                selected=selected_by_rdkit,
                implication=implication,
            )
        )
    return tuple(rows)


def main() -> None:
    print(f"RDKit version: {rdBase.rdkitVersion}")
    print()
    print("Alternative meanings compared:")
    for policy in MarkerPolicy:
        print(f"  - {policy.value}")

    for case in CASES:
        source_mol = Chem.MolFromSmiles(case.smiles)
        if source_mol is None:
            raise ValueError(f"RDKit failed to parse source {case.case_id}")
        expected = Chem.MolToSmiles(
            Chem.Mol(source_mol),
            isomericSmiles=True,
            canonical=True,
            doRandom=False,
        )
        expected_mol = Chem.MolFromSmiles(expected)
        if expected_mol is None:
            raise ValueError(f"RDKit failed to parse expected {case.case_id}")
        support = grimace_support(source_mol, rooted_at_atom=None, isomeric_smiles=True)
        choices = side_choices(source_mol)
        selected_edges_by_component = stereo_selected_edges_by_component(
            source_mol,
            choices,
        )

        print()
        print(case.case_id)
        print(f"  expected: {expected}")
        print(
            "  current Grimace: "
            f"{'contains' if expected in support else 'missing'} "
            f"(support size {len(support)})"
        )
        print(
            "  RDKit selected stereo-atom edges by component: "
            f"{dict(sorted((component, sorted(edges)) for component, edges in selected_edges_by_component.items()))}"
        )
        for marker in visible_markers(expected_mol):
            print(
                f"  visible marker edge={marker.edge} marker={marker.marker} "
                f"dir={marker.bond_dir}"
            )
            for policy in MarkerPolicy:
                for row in describe_policy_implications(
                    policy,
                    marker,
                    choices,
                    selected_edges_by_component,
                ):
                    print(f"    {policy.value}: {row}")


if __name__ == "__main__":
    main()
