"""Probe modified aromatic atom text as a separate South Star policy slice.

This script does not define runtime behavior. It inspects sanitized RDKit
molecule facts for bracketed aromatic atoms such as ``[nH]`` and asks whether
the existing ring traversal spine would be enough if the atom-text and grammar
policies admitted the bracket aromatic token.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

import grimace._south_star.enum_s as enum_s
from grimace._south_star.annotation_policy import (
    MaximalEligibleCarrierAnnotationPolicy,
)
from grimace._south_star.atom_text import (
    SOUTH_STAR_AROMATIC_ATOM_TEXT_TOKENS,
    SouthStarAtomTextFields,
    atom_text_obligation_for_supported_fields,
    south_star_atom_text_fields,
)
from grimace._south_star.component_support_state import SouthStarComponentSupportState
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_semantic_oracle import south_star_conformance_report


@dataclass(frozen=True, slots=True)
class ProbeCase:
    case_id: str
    smiles: str


CASES = (
    ProbeCase("pyrrole_explicit_h", "c1cc[nH]c1"),
    ProbeCase("pyrrole_isotope_h", "c1cc[15nH]c1"),
    ProbeCase("pyrrole_mapped_h", "[nH:7]1cccc1"),
    ProbeCase("pyridine_mapped_n", "c1cc[n:7]cc1"),
    ProbeCase("pyridinium_h", "c1cc[nH+]cc1"),
    ProbeCase("pyridine_n_oxide", "c1cc[n+]([O-])cc1"),
    ProbeCase("selenophene", "[se]1cccc1"),
)


def main() -> None:
    for case in CASES:
        mol = Chem.MolFromSmiles(case.smiles)
        print(f"{case.case_id}: {case.smiles}")
        if mol is None:
            print("  parsed=False")
            continue

        facts = SouthStarMoleculeFacts.from_mol(mol)
        report = south_star_support_gate_report(mol)
        modified_atoms = tuple(_modified_aromatic_atom_rows(mol))
        print(f"  parsed=True canonical={Chem.MolToSmiles(mol, canonical=True)}")
        print(f"  gate_categories={sorted(report.categories)}")
        print(f"  fact_categories={sorted(facts.unsupported_categories)}")
        print(
            "  topology="
            f"atoms:{facts.graph_topology.atom_count} "
            f"bonds:{facts.graph_topology.bond_count} "
            f"rings:{facts.graph_topology.ring_count}"
        )
        for row in modified_atoms:
            print(f"  modified_atom={row}")

        if _can_probe_with_hypothetical_atom_text(mol):
            support = _hypothetical_support(mol)
            conformance = tuple(
                south_star_conformance_report(
                    source_smiles=case.smiles,
                    candidate_smiles=output,
                )
                for output in support
            )
            print(f"  hypothetical_support={len(support)}")
            print(
                "  conformance_rejections="
                f"parse:{_failed_count(conformance, 'rdkit_parseability')} "
                f"graph:{_failed_count(conformance, 'graph_equivalence')} "
                f"stereo:{_failed_count(conformance, 'stereo_equivalence')} "
                f"grammar:{_failed_count(conformance, 'grammar_conformance')}"
            )
            print(f"  first_outputs={support[:8]}")
        print()


def _modified_aromatic_atom_rows(mol: Chem.Mol) -> tuple[str, ...]:
    rows: list[str] = []
    for atom in mol.GetAtoms():
        fields = south_star_atom_text_fields(atom)
        if not fields.is_aromatic:
            continue
        try:
            obligation = atom_text_obligation_for_supported_fields(fields)
            current = f"current:{obligation.emitted_text}"
        except NotImplementedError as exc:
            current = f"current_error:{exc}"
        token = _hypothetical_aromatic_atom_text(fields)
        rows.append(
            "idx:{idx} symbol:{symbol} isotope:{isotope} charge:{charge} "
            "h:{hydrogen} map:{atom_map} token:{token} {current}".format(
                idx=fields.atom_idx,
                symbol=fields.symbol,
                isotope=fields.isotope,
                charge=fields.formal_charge,
                hydrogen=fields.explicit_hydrogen_count,
                atom_map=fields.atom_map_number,
                token=token,
                current=current,
            )
        )
    return tuple(rows)


def _can_probe_with_hypothetical_atom_text(mol: Chem.Mol) -> bool:
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    if mol.GetRingInfo().NumRings() != 1:
        return False
    return all(
        _hypothetical_aromatic_atom_text(south_star_atom_text_fields(atom))
        is not None
        for atom in mol.GetAtoms()
        if atom.GetIsAromatic()
    )


def _hypothetical_support(mol: Chem.Mol) -> tuple[str, ...]:
    old_atom_text = enum_s.atom_text_for_supported_atom
    enum_s.atom_text_for_supported_atom = _hypothetical_atom_text_for_atom
    try:
        facts = SouthStarMoleculeFacts.from_mol(mol)
        state = SouthStarComponentSupportState(
            molecule_facts=facts,
            annotation_policy=MaximalEligibleCarrierAnnotationPolicy(),
        )
        traversals = enum_s._ring_system_traversals(
            mol,
            molecule_facts=facts,
            state=state,
            closure_edge_sets=tuple(
                (edge,) for edge in enum_s._supported_single_ring_edges(mol)
            ),
            marker_by_edge={},
            component_marker_assignments=(),
        )
        return tuple(
            dict.fromkeys(
                enum_s.render_south_star_tree_traversal(traversal)
                for traversal in traversals
            )
        )
    finally:
        enum_s.atom_text_for_supported_atom = old_atom_text


def _hypothetical_atom_text_for_atom(atom: Chem.Atom) -> str:
    fields = south_star_atom_text_fields(atom)
    token = _hypothetical_aromatic_atom_text(fields)
    if token is not None:
        return token
    return atom_text_obligation_for_supported_fields(fields).emitted_text


def _failed_count(conformance: tuple[object, ...], attribute: str) -> int:
    return sum(
        1
        for report in conformance
        if not getattr(report, attribute).passed
    )


def _hypothetical_aromatic_atom_text(
    fields: SouthStarAtomTextFields,
) -> str | None:
    if not fields.is_aromatic:
        return None
    token = fields.symbol.lower()
    if token not in SOUTH_STAR_AROMATIC_ATOM_TEXT_TOKENS:
        return None
    if (
        fields.isotope == 0
        and fields.formal_charge == 0
        and fields.explicit_hydrogen_count == 0
        and fields.atom_map_number == 0
        and fields.radical_electron_count == 0
    ):
        return token
    return (
        "["
        f"{'' if fields.isotope == 0 else fields.isotope}"
        f"{token}"
        f"{_hydrogen_text(fields.explicit_hydrogen_count)}"
        f"{_charge_text(fields.formal_charge)}"
        f"{'' if fields.atom_map_number == 0 else ':' + str(fields.atom_map_number)}"
        "]"
    )


def _hydrogen_text(count: int) -> str:
    if count == 0:
        return ""
    if count == 1:
        return "H"
    return f"H{count}"


def _charge_text(charge: int) -> str:
    if charge == 0:
        return ""
    sign = "+" if charge > 0 else "-"
    magnitude = abs(charge)
    if magnitude == 1:
        return sign
    return f"{sign}{magnitude}"


if __name__ == "__main__":
    main()
