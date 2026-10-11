"""Capture fresh 2026.03.4 witnesses for the bounded South Star 2 fixtures."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from rdkit import Chem, rdBase


ROOT = Path(__file__).resolve().parents[3]
VERSION = "2026.03.4"
SOURCE_COMMIT = "8afba32ec539dcb2369bc84549d802aca3f7eb39"
SOURCE_ROOT = ROOT / "tests/fixtures/rdkit_upstream_serializer_sources" / VERSION
STRUCTURAL_FIXTURES = ROOT / "tests/fixtures"
LIMIT = 100_000

EXACT_BOUNDS = {
    "ss2_c_o_linear_ethanol": (
        "CCO",
        None,
        4,
        [
            "Either terminal root forces one orientation, yielding CCO or OCC.",
            "The central root has two child orders, yielding C(C)O or C(O)C.",
        ],
    ),
    "ss2_c_o_isopropanol": (
        "CC(C)O",
        None,
        6,
        [
            "The tertiary carbon root has three distinct child orders because its two carbon children are identical.",
            "Either equivalent methyl root allows two orders for the remaining carbon and oxygen children at the tertiary carbon.",
            "The oxygen root has one order for the two equivalent carbon children at the tertiary carbon.",
        ],
    ),
    "ss2_c_cycle_triangle": (
        "C1CC1",
        None,
        1,
        ["Every root and either direction around the cycle yields the same deterministic ring-label spelling."],
    ),
    "ss2_disconnected_carbon_oxygen": (
        "C.O",
        None,
        1,
        ["Each atom component has one spelling; fixed input-order fragment construction yields C.O."],
    ),
    "ss2_disconnected_cc_o_ethane_component_first": (
        "CC.O",
        [0, 2, 1],
        1,
        ["The ethane and oxygen components each have one spelling; this input atom order assembles CC.O."],
    ),
    "ss2_disconnected_cc_o_oxygen_component_first": (
        "CC.O",
        [2, 1, 0],
        1,
        ["The ethane and oxygen components each have one spelling; this input atom order assembles O.CC."],
    ),
}

SOURCE_CLAIMS = [
    {
        "path": "Code/GraphMol/Canon.cpp",
        "lines": "841-965, 991-1025, 1033-1124, 1146-1154",
        "claim": "The cycle-detection and writer DFS phases define finite root and continuation-child order choices; the random writer orders branch children within those choices.",
    },
    {
        "path": "Code/GraphMol/SmilesParse/SmilesWrite.cpp",
        "lines": "105-240, 279-311, 435-462, 503-530, 620-624, 681-696, 704-745, 834-849",
        "claim": "These lines define atom and bond spellings, deterministic ring labels, random root selection when rootedAtAtom is -1, consumption of the concrete fragment atom-index mapping, restoration of source atom indices, noncanonical component assembly in vector order, and seeded random-vector replay.",
    },
    {
        "path": "Code/GraphMol/MolOps.cpp",
        "lines": "723-755, 857-899",
        "claim": "Fragment membership is grouped by component and source atom indices are appended in input order. For each fixed disconnected input, the recorded GetMolFrags atom-index tuples provide its concrete component order; the writer consumes this mapping directly.",
    },
]

STEREO_SOURCE_CLAIMS = [
    {
        "path": "Code/GraphMol/Atom.cpp",
        "lines": "748-760",
        "claim": "Atom::getPerturbationOrder compares the proposed traversal bond order with the incoming-bond reference frame.",
    },
    {
        "path": "Code/GraphMol/Canon.cpp",
        "lines": "338-345, 356-417, 553-563, 1198-1304, 1312-1344, 1460-1508",
        "claim": "RDKit maps tetrahedral tags and double-bond stereo through traversal order, selects and directs carrier bonds, and removes only redundant directional marks.",
    },
    {
        "path": "Code/GraphMol/SmilesParse/SmilesWrite.cpp",
        "lines": "279-292",
        "claim": "The writer maps up and down bond directions to slash and backslash in isomeric SMILES.",
    },
    *SOURCE_CLAIMS,
]


def _manifest() -> dict[str, object]:
    source = SOURCE_ROOT / "source"
    files = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        content = path.read_bytes()
        files.append(
            {
                "bytes": len(content),
                "lines": content.count(b"\n"),
                "path": path.relative_to(source).as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    license_path = SOURCE_ROOT / "LICENSE.rdkit.txt"
    return {
        "files": files,
        "license": {
            "path": "LICENSE.rdkit.txt",
            "sha256": hashlib.sha256(license_path.read_bytes()).hexdigest(),
            "spdx": "BSD-3-Clause",
        },
        "rdkit_version": VERSION,
        "source_commit": SOURCE_COMMIT,
        "source_describe": "Release_2026_03_4",
        "source_repository": "https://github.com/rdkit/rdkit",
    }


def _fresh_mol(smiles: str, atom_order: list[int] | None = None) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise AssertionError(f"RDKit could not parse {smiles!r}")
    return Chem.RenumberAtoms(mol, atom_order) if atom_order is not None else mol


def _write(mol: Chem.Mol, seed: int) -> tuple[str, list[int], list[int]]:
    fresh = Chem.Mol(mol)
    output = Chem.MolToRandomSmilesVect(fresh, 1, seed, True, False, False, False)[0]
    atom_order = json.loads(fresh.GetProp("_smilesAtomOutputOrder"))
    bond_order = json.loads(fresh.GetProp("_smilesBondOutputOrder"))
    return output, atom_order, bond_order


def _capture_exact_support() -> dict[str, object]:
    cases = []
    for case_id, (smiles, atom_order, upper, bound_cases) in EXACT_BOUNDS.items():
        mol = _fresh_mol(smiles, atom_order)
        fragments = [list(fragment) for fragment in Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)]
        observed: dict[str, int] = {}
        for seed in range(1, LIMIT + 1):
            output, _, _ = _write(mol, seed)
            observed.setdefault(output, seed)
            if len(observed) == upper:
                break
        else:
            raise AssertionError(f"{case_id} only observed {len(observed)} of {upper} bounded spellings")
        expected = sorted(observed)
        cases.append(
            {
                "id": case_id,
                "source": "Fresh RDKit 2026.03.4 writer witnesses attain the finite source-derived root and child-order bound.",
                "input_smiles": smiles,
                **({"atom_order": atom_order} if atom_order is not None else {}),
                **({"fragment_atom_indices": fragments} if len(fragments) > 1 else {}),
                "upper_bound": upper,
                "bound_cases": bound_cases,
                "expected_support": expected,
                "witness_seeds": [observed[item] for item in expected],
            }
        )
    return {
        "rdkit_version": VERSION,
        "writer_policy": {
            "input": "Fresh RDKit MolFromSmiles input with atom indexing preserved; renumbered cases apply the listed Chem.RenumberAtoms order to a fresh parse",
            "rooted_at_atom": -1,
            "canonical": False,
            "do_random": True,
            "isomeric_smiles": True,
            "kekule": False,
            "all_bonds_explicit": False,
            "all_hs_explicit": False,
            "component_order": "For disconnected cases, the fixture records the actual .4 GetMolFrags atom-index tuples for this fixed parsed input; the pinned source shows those tuples are passed to the noncanonical writer and assembled in vector order. This is per-input evidence, not a general component-label theorem.",
        },
        "completeness_basis": {
            "kind": "source_bounded_writer_permutations",
            "source_manifest": "tests/fixtures/rdkit_upstream_serializer_sources/2026.03.4/manifest.json",
            "source_commit": SOURCE_COMMIT,
            "argument": "For these fixed parsed graphs and writer parameters, the tagged writer has a finite set of root and child-order traversals. The complete output spelling set is enumerated structurally per input, including deterministic cycle labels and the actual .4 GetMolFrags atom-index tuples for each disconnected input. The pinned writer source consumes those tuples and assembles fragments in vector order. Fresh .4 positive seed replays attain each listed spelling, so the finite bound is attained.",
            "source_claims": SOURCE_CLAIMS,
            "witnesses": "Each case records one positive seeded MolToRandomSmilesVect replay per expected spelling, captured under RDKit 2026.03.4. The source-derived upper bound establishes completeness; sampled or seeded membership alone does not.",
        },
        "cases": cases,
    }


def _tree_schedule_bound(atoms: list[dict[str, object]]) -> int:
    degrees = [int(atom["degree"]) for atom in atoms]
    edge_count = sum(degrees) // 2
    bound = sum(
        math.factorial(root_degree)
        * math.prod(math.factorial(degree - 1) for index, degree in enumerate(degrees) if index != root)
        for root, root_degree in enumerate(degrees)
    )
    simplified = 2 * edge_count * math.prod(math.factorial(degree - 1) for degree in degrees)
    if bound != simplified:
        raise AssertionError("tree schedule bound does not reduce to the degree formula")
    return bound


def _stereo_structures() -> dict[str, dict[str, object]]:
    path = STRUCTURAL_FIXTURES / "rdkit_south_star2_stereo_evidence/2026.03.1.json"
    source = json.loads(path.read_text())
    # Only graph and stereo input facts are carried forward; no .1 output, seed,
    # source authority, or completeness claim is used as .4 output evidence.
    keep = {
        "id", "input_smiles", "graph_facts", "configured_bonds",
        "clear_all_bond_directions_before_write", "unspecified_stereo", "alkenes",
    }
    structures = {case["id"]: {key: case[key] for key in keep if key in case} for case in source["cases"]}
    for case in structures.values():
        mol = _fresh_mol(case["input_smiles"])
        case["graph_facts"] = _graph_facts(mol)
    return structures


def _graph_facts(mol: Chem.Mol) -> dict[str, object]:
    return {
        "atoms": [
            {
                "index": atom.GetIdx(),
                "element": atom.GetSymbol(),
                "degree": atom.GetDegree(),
                "non_graph_hydrogens": atom.GetTotalNumHs(),
            }
            for atom in mol.GetAtoms()
        ],
        "bonds": [
            {
                "index": bond.GetIdx(),
                "begin": bond.GetBeginAtomIdx(),
                "end": bond.GetEndAtomIdx(),
                "type": str(bond.GetBondType()),
                "stereo": str(bond.GetStereo()),
                "direction": str(bond.GetBondDir()),
            }
            for bond in mol.GetBonds()
        ],
        "tetrahedral_centers": [
            {
                "atom": atom.GetIdx(),
                "incident_bonds": sorted(bond.GetIdx() for bond in atom.GetBonds()),
                "non_graph_hydrogens": atom.GetTotalNumHs(),
                "input_chiral_tag": str(atom.GetChiralTag()),
            }
            for atom in mol.GetAtoms()
            if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        ],
    }


def _rooted_tree_schedules(mol: Chem.Mol) -> list[tuple[int, tuple[int, ...], tuple[int, ...]]]:
    adjacency = {
        atom.GetIdx(): tuple(sorted(neighbor.GetIdx() for neighbor in atom.GetNeighbors()))
        for atom in mol.GetAtoms()
    }

    def expand(atom: int, parent: int | None):
        children = tuple(neighbor for neighbor in adjacency[atom] if neighbor != parent)
        outcomes = []
        from itertools import permutations

        for child_order in permutations(children):
            partial = [((atom,), {atom: child_order})]
            for child in child_order:
                combined = []
                for prefix, decisions in partial:
                    for suffix, child_decisions in expand(child, atom):
                        combined.append((prefix + suffix, decisions | child_decisions))
                partial = combined
            outcomes.extend(partial)
        return outcomes

    schedules = {
        (root, tuple(decisions[root]), tuple(order))
        for root in adjacency
        for order, decisions in expand(root, None)
    }
    return sorted(schedules)


def _assert_graph_facts(mol: Chem.Mol, case: dict[str, object]) -> None:
    facts = case["graph_facts"]
    if len(facts["atoms"]) != mol.GetNumAtoms():
        raise AssertionError(f".4 atom count differs for {case['id']}")
    for row in facts["atoms"]:
        atom = mol.GetAtomWithIdx(row["index"])
        if (atom.GetSymbol(), atom.GetDegree(), atom.GetTotalNumHs()) != (
            row["element"], row["degree"], row["non_graph_hydrogens"]
        ):
            raise AssertionError(f".4 parsed atom fact differs in {case['id']}: {row!r}")
    for row in facts["bonds"]:
        bond = mol.GetBondWithIdx(row["index"])
        if (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), str(bond.GetBondType())) != (
            row["begin"], row["end"], row["type"]
        ):
            raise AssertionError(f".4 parsed bond fact differs in {case['id']}: {row!r}")
        if "stereo" in row and str(bond.GetStereo()) != row["stereo"]:
            raise AssertionError(f".4 parsed bond stereo differs in {case['id']}: {row!r}")
        if "direction" in row and str(bond.GetBondDir()) != row["direction"]:
            raise AssertionError(f".4 parsed bond direction differs in {case['id']}: {row!r}")


def _configured_mol(case: dict[str, object]) -> Chem.Mol:
    mol = _fresh_mol(case["input_smiles"])
    for fact in case.get("configured_bonds", []):
        bond = mol.GetBondWithIdx(fact["index"])
        if (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) != (fact["begin"], fact["end"]):
            raise AssertionError(f"configured bond endpoints differ for {case['id']}")
        if fact["stereo"] != "STEREONONE":
            bond.SetStereoAtoms(*fact["stereo_atoms"])
            bond.SetStereo(getattr(Chem.BondStereo, fact["stereo"]))
        elif bond.GetStereo() != Chem.BondStereo.STEREONONE:
            raise AssertionError(f"middle alkene unexpectedly configured for {case['id']}")
    if case.get("clear_all_bond_directions_before_write"):
        for bond in mol.GetBonds():
            bond.SetBondDir(Chem.BondDir.NONE)
    return mol


def _capture_support(case: dict[str, object], mol: Chem.Mol, bound: int) -> tuple[list[str], list[int]]:
    observed: dict[str, int] = {}
    for seed in range(1, LIMIT + 1):
        output, _, _ = _write(mol, seed)
        observed.setdefault(output, seed)
        if len(observed) == bound:
            expected = sorted(observed)
            return expected, [observed[item] for item in expected]
    raise AssertionError(f"{case['id']} observed fewer than its source bound {bound}")


def _capture_schedule_rows(case: dict[str, object], mol: Chem.Mol) -> list[dict[str, object]]:
    schedules = _rooted_tree_schedules(mol)
    rows = []
    for root, children, visit_order in schedules:
        for seed in range(1, LIMIT + 1):
            output, atom_order, bond_order = _write(mol, seed)
            if tuple(atom_order) != visit_order:
                continue
            endpoints = tuple(
                bond["begin"]
                for bond in case["graph_facts"]["bonds"]
                if bond["type"] == "DOUBLE"
            )
            double_bond = next(
                bond for bond in case["graph_facts"]["bonds"] if bond["type"] == "DOUBLE"
            )
            endpoints = (double_bond["begin"], double_bond["end"])
            carriers = {
                endpoint: next(
                    bond for bond in case["graph_facts"]["bonds"]
                    if endpoint in (bond["begin"], bond["end"]) and bond["index"] != double_bond["index"]
                )
                for endpoint in endpoints
            }
            first_endpoint = min(endpoints, key=visit_order.index)
            second_endpoint = next(endpoint for endpoint in endpoints if endpoint != first_endpoint)
            control_bond = carriers[first_endpoint]
            other_bond = carriers[second_endpoint]
            parsed = Chem.MolFromSmiles(output)
            if parsed is None:
                raise AssertionError(f".4 emitted unparsable schedule output {output!r}")
            parsed_atom_for_source = {source: index for index, source in enumerate(atom_order)}

            def carrier_row(fact: dict[str, object], endpoint: int) -> dict[str, object]:
                other = fact["end"] if fact["begin"] == endpoint else fact["begin"]
                source_from, source_to = sorted((endpoint, other), key=visit_order.index)
                parsed_bond = parsed.GetBondBetweenAtoms(
                    parsed_atom_for_source[source_from], parsed_atom_for_source[source_to]
                )
                if parsed_bond is None:
                    raise AssertionError("carrier bond missing after .4 parseback")
                glyph = {
                    Chem.BondDir.ENDUPRIGHT: "/",
                    Chem.BondDir.ENDDOWNRIGHT: "\\",
                }.get(parsed_bond.GetBondDir())
                if glyph is None:
                    raise AssertionError(f"directional carrier has no slash glyph: {output!r}")
                return {
                    "bond_index": fact["index"],
                    "from_atom": source_from,
                    "to_atom": source_to,
                    "glyph": glyph,
                }

            positions = {atom: index for index, atom in enumerate(visit_order)}
            control_anchor = control_bond["end"] if control_bond["begin"] == first_endpoint else control_bond["begin"]
            other_anchor = other_bond["end"] if other_bond["begin"] == second_endpoint else other_bond["begin"]
            rows.append(
                {
                    "root": root,
                    "root_children": list(children),
                    "visit_order": list(visit_order),
                    "first_endpoint": first_endpoint,
                    "control_carrier": carrier_row(control_bond, first_endpoint),
                    "other_carrier": carrier_row(other_bond, second_endpoint),
                    "rendered_smiles": output,
                    "seed": seed,
                    "atom_output_order": atom_order,
                    "bond_output_order": bond_order,
                    "control_flipped": positions[first_endpoint] < positions[control_anchor],
                    "other_flipped": positions[other_anchor] < positions[second_endpoint],
                }
            )
            break
        else:
            raise AssertionError(f"could not capture .4 schedule {(root, children, visit_order)!r}")
    if len(rows) != len(schedules):
        raise AssertionError(f"{case['id']} did not capture each source schedule")
    return rows


def _parseback_preserves_unspecified(case: dict[str, object], source: Chem.Mol, output: str, atom_order: list[int]) -> bool:
    parsed = Chem.MolFromSmiles(output)
    if parsed is None:
        return False
    parsed_atom_for_source = {source_index: parsed_index for parsed_index, source_index in enumerate(atom_order)}
    for fact in case["unspecified_stereo"]:
        if fact["kind"] == "alkene":
            source_bond = source.GetBondWithIdx(fact["bond_index"])
            parsed_bond = parsed.GetBondBetweenAtoms(
                parsed_atom_for_source[source_bond.GetBeginAtomIdx()],
                parsed_atom_for_source[source_bond.GetEndAtomIdx()],
            )
            if parsed_bond is None or str(parsed_bond.GetStereo()) != fact["expected"]:
                return False
        else:
            parsed_atom = parsed.GetAtomWithIdx(parsed_atom_for_source[fact["atom_index"]])
            if str(parsed_atom.GetChiralTag()) != fact["expected"]:
                return False
    return True


def _capture_controls(structures: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    cases = []
    for case_id in ("unspecified_alkene_with_tetrahedral_stereo", "unspecified_tetrahedron_with_e_alkene"):
        case = structures[case_id]
        mol = _configured_mol(case)
        _assert_graph_facts(mol, case)
        witnesses = []
        for seed in range(1, LIMIT + 1):
            output, atom_order, _ = _write(mol, seed)
            if not _parseback_preserves_unspecified(case, mol, output, atom_order):
                continue
            witnesses.append({"seed": seed, "smiles": output})
            if len(witnesses) == 2:
                break
        if len(witnesses) != 2:
            raise AssertionError(f"could not capture two .4 semantic controls for {case_id}")
        cases.append(
            {
                **case,
                "source": "Fresh RDKit 2026.03.4 seeded writer membership witnesses with semantic parse-back controls; no support completeness is claimed.",
                "witnesses": witnesses,
                "support_authority": "seeded_writer_membership_and_semantic_parseback",
                "evidence_kind": "unspecified_stereo_semantic_control",
            }
        )
    return cases


def _capture_triene(structures: dict[str, dict[str, object]]) -> dict[str, object]:
    case = structures["unspecified_middle_alkene_receives_writer_directional_marks"]
    mol = _configured_mol(case)
    witnesses: dict[str, dict[str, object]] = {}
    for seed in range(1, LIMIT + 1):
        output, _, _ = _write(mol, seed)
        parsed = Chem.MolFromSmiles(output)
        if parsed is None:
            continue
        # The source input has a one-to-one chain order; the middle double bond
        # is the second double bond in the parsed triene.
        doubles = [bond for bond in parsed.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE]
        if len(doubles) != 3 or doubles[1].GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        stereo = str(doubles[1].GetStereo())
        witnesses.setdefault(stereo, {"seed": seed, "smiles": output, "parsed_middle_stereo": stereo})
        if {"STEREOE", "STEREOZ"} <= witnesses.keys():
            break
    if {"STEREOE", "STEREOZ"} - witnesses.keys():
        raise AssertionError(".4 did not reproduce both middle-bond directional parseback outcomes")
    return {
        **case,
        "source": "Fresh RDKit 2026.03.4 seeded membership witnesses record the directional-mark parse-back discrepancy; no support completeness is claimed.",
        "source_lines": "Code/GraphMol/Canon.cpp:1198-1304",
        "evidence_scope": "Positive writer membership and parse-back discrepancy only; no support completeness or negative parity claim.",
        "witnesses": [witnesses[stereo] for stereo in ("STEREOE", "STEREOZ")],
        "support_authority": "seeded_writer_membership_and_semantic_parseback",
        "evidence_kind": "known_writer_quirk",
    }


def _capture_stereo_evidence() -> dict[str, object]:
    structures = _stereo_structures()
    complete = []
    for case_id in (
        "stereo_tetra_ccw_tree", "stereo_tetra_cw_tree", "stereo_mixed_alkene_tetra_tree",
    ):
        case = structures[case_id]
        mol = _configured_mol(case)
        _assert_graph_facts(mol, case)
        bound = _tree_schedule_bound(case["graph_facts"]["atoms"])
        support, seeds = _capture_support(case, mol, bound)
        complete.append(
            {
                **case,
                "schedule_upper_bound": bound,
                "expected_distinct_outputs": bound,
                "distinct_output_upper_bound": bound,
                "bound_argument": "The fixed graph is a tree. The source-derived sum over roots and continuation orders bounds complete writer support; .4 seed witnesses attain every distinct spelling up to this bound.",
                "source": "RDKit 2026.03.4 seeded witnesses attain the finite source-derived tree traversal bound.",
                "support_authority": "source_bounded_writer_schedule_and_seeded_witnesses",
                "evidence_kind": "complete_bounded_writer_support",
                "expected_support": support,
                "witness_seeds": seeds,
            }
        )
    for case_id in ("stereo_e_butene_tree", "stereo_z_butene_tree"):
        case = structures[case_id]
        mol = _configured_mol(case)
        _assert_graph_facts(mol, case)
        rows = _capture_schedule_rows(case, mol)
        bound = _tree_schedule_bound(case["graph_facts"]["atoms"])
        if bound != len(rows):
            raise AssertionError(f"{case_id} has {len(rows)} schedules but source bound {bound}")
        complete.append(
            {
                **case,
                "schedule_upper_bound": bound,
                "expected_distinct_outputs": len({row["rendered_smiles"] for row in rows}),
                "distinct_output_upper_bound": bound,
                "bound_argument": "The graph is a fixed four-atom tree. The source-derived root and continuation-order schedules enumerate the complete bound; each schedule has a fresh .4 seed replay. Some schedule reversals serialize identically, so the number of distinct outputs can be below the schedule upper bound.",
                "source": "Every root/child-order schedule is witnessed and replayed with RDKit 2026.03.4.",
                "support_authority": "source_bounded_writer_schedule_and_seeded_witnesses",
                "evidence_kind": "complete_bounded_writer_support",
                "source_schedule_rows": rows,
            }
        )
    cases = [*complete, *_capture_controls(structures), _capture_triene(structures)]
    if len(cases) != 8:
        raise AssertionError(f"expected eight stereo evidence cases, found {len(cases)}")
    return {
        "rdkit_version": VERSION,
        "source_commit": SOURCE_COMMIT,
        "source_manifest": "tests/fixtures/rdkit_upstream_serializer_sources/2026.03.4/manifest.json",
        "tree_bound_formula": "sum_r[d(r)! product(v!=r)(degree(v)-1)!] = 2|E| product_v(degree(v)-1)!",
        "source_claims": STEREO_SOURCE_CLAIMS,
        "cases": cases,
    }


def _write_fixture(name: str, payload: dict[str, object]) -> None:
    path = STRUCTURAL_FIXTURES / name / f"{VERSION}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)} cases={len(payload['cases'])}")


def main() -> None:
    if rdBase.rdkitVersion != VERSION:
        raise RuntimeError(f"capture requires RDKit {VERSION}, got {rdBase.rdkitVersion}")
    source_manifest_path = SOURCE_ROOT / "manifest.json"
    source_manifest_path.write_text(json.dumps(_manifest(), indent=2) + "\n")
    _write_fixture("rdkit_south_star2_exact_support", _capture_exact_support())
    _write_fixture("rdkit_south_star2_stereo_evidence", _capture_stereo_evidence())


if __name__ == "__main__":
    main()
