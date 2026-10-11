"""Capture bounded SS2 admission and output deltas against the published package.

The published API's enumeration is recorded as a package baseline, not as an
independently source-proven complete RDKit support set. Seeded RDKit strings are
membership witnesses only. This one-off probe is pinned to the artifacts below.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import zipfile

BASE_REVISION = "b8e2ec21def681ec03000f21df1fa5746e4633bb"
RDKIT_VERSION = "2026.03.4"
PUBLISHED_VERSION = "0.1.18"
PUBLISHED_WHEEL_SHA256 = "14589d18295e418b5570566953250b5b039a4c20c79ef7815eb221e98ff239ee"
PUBLISHED_CORE_SHA256 = "998c36017b541c9c438979de895bf4ac15a64fc06813427b83b449d416b4bdb1"
CURRENT_CORE_SHA256 = "8a26e652ad7a2efd26ea3bd00b0d009fcfec66c089d3bd291d8cce759ed1f24c"
CURRENT_PREPARATION_SHA256 = "3598850b51a817703d1b4009f082e1bbb5428c5e006fab99341164de62023053"
MAX_PREFIX_STATES = 20_000
TRIENE_SMILES = "CC=CC=CC=CC"
TRIENE_EXPLICIT_H = r"C/C=C(\[H])C=CC(/[H])=C/C"

CASES = ({'id': 'control-C', 'smiles': 'C', 'exact': True, 'has_witness_path': True}, {'id': 'control-O', 'smiles': 'O', 'exact': True, 'has_witness_path': True}, {'id': 'control-CO', 'smiles': 'CO', 'exact': True, 'has_witness_path': True}, {'id': 'control-CCO', 'smiles': 'CCO', 'exact': True, 'has_witness_path': True}, {'id': 'control-isopropanol', 'smiles': 'CC(C)O', 'exact': True, 'has_witness_path': True}, {'id': 'control-triangle', 'smiles': 'C1CC1', 'exact': True, 'has_witness_path': True}, {'id': 'control-disconnected', 'smiles': 'C.O', 'exact': True, 'has_witness_path': True}, {'id': 'element-N', 'smiles': 'CCN', 'exact': True, 'has_witness_path': True}, {'id': 'element-S', 'smiles': 'CCS', 'exact': True, 'has_witness_path': True}, {'id': 'element-P', 'smiles': 'CCP', 'exact': True, 'has_witness_path': True}, {'id': 'element-B', 'smiles': 'CCB', 'exact': True, 'has_witness_path': True}, {'id': 'element-F', 'smiles': 'CCF', 'exact': True, 'has_witness_path': True}, {'id': 'element-Cl', 'smiles': 'CCCl', 'exact': True, 'has_witness_path': True}, {'id': 'charged-ammonium', 'smiles': '[NH4+]', 'exact': True, 'has_witness_path': True}, {'id': 'charged-methylammonium', 'smiles': 'C[NH3+]', 'exact': True, 'has_witness_path': True}, {'id': 'explicit-H', 'smiles': '[H]C', 'exact': True, 'has_witness_path': True}, {'id': 'isotope', 'smiles': '[13CH4]', 'exact': True, 'has_witness_path': True}, {'id': 'atom-map', 'smiles': '[CH3:7]C', 'exact': True, 'has_witness_path': True}, {'id': 'radical', 'smiles': '[CH3]', 'exact': True, 'has_witness_path': True}, {'id': 'aromatic-carbon', 'smiles': 'c1ccccc1', 'exact': True, 'has_witness_path': True}, {'id': 'aromatic-hetero', 'smiles': 'c1ncccc1', 'exact': True, 'has_witness_path': True}, {'id': 'triple', 'smiles': 'C#N', 'exact': True, 'has_witness_path': True}, {'id': 'tetrahedral', 'smiles': 'C[C@H](O)CO', 'exact': True, 'has_witness_path': True}, {'id': 'directional-E', 'smiles': 'F/C=C/F', 'exact': True, 'has_witness_path': True}, {'id': 'directional-chain', 'smiles': 'C/C=C/C=C/C', 'exact': True, 'has_witness_path': True}, {'id': 'cyclic-tetra', 'smiles': 'C1CC[C@H]1O', 'exact': True, 'has_witness_path': True}, {'id': 'published-01', 'smiles': 'CC(=O)Oc1ccccc1C(=O)O', 'exact': False, 'has_witness_path': False}, {'id': 'published-02', 'smiles': 'C1COCCC12CO2', 'exact': False, 'has_witness_path': False}, {'id': 'published-03', 'smiles': 'CN1CCC[C@H]1C2=CN=CC=C2', 'exact': False, 'has_witness_path': False}, {'id': 'published-04', 'smiles': 'C/C(=N\\\\OC(=O)NC)/SC', 'exact': False, 'has_witness_path': False}, {'id': 'published-05', 'smiles': 'C1=CC(=C(C=C1C[C@@H](C(=O)O)N)O)O', 'exact': False, 'has_witness_path': False}, {'id': 'published-06', 'smiles': 'C[C@@H](C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O', 'exact': False, 'has_witness_path': False}, {'id': 'published-07', 'smiles': 'C1[C@H]([C@@H]2[C@H](O1)[C@H](CO2)O[N+](=O)[O-])O[N+](=O)[O-]', 'exact': False, 'has_witness_path': False}, {'id': 'published-08', 'smiles': 'C[C@@H]([C@@H]1[C@@]2([C@@H](CC1)/C(=C/C=C1/C[C@@H](O)CCC1=C)CCC2)C)', 'exact': False, 'has_witness_path': False}, {'id': 'published-09', 'smiles': 'CC\\\\1=C(C2=C(/C1=C\\\\C3=CC=C(C=C3)S(=O)C)C=CC(=C2)F)CC(=O)O', 'exact': False, 'has_witness_path': False}, {'id': 'directional-CC-E', 'smiles': 'C/C=C/C', 'exact': True, 'has_witness_path': True}, {'id': 'directional-CC-Z', 'smiles': 'C/C=C\\C', 'exact': True, 'has_witness_path': True}, {'id': 'directional-CO-E', 'smiles': 'C/C=C/O', 'exact': True, 'has_witness_path': True}, {'id': 'directional-CO-Z', 'smiles': 'C/C=C\\O', 'exact': True, 'has_witness_path': True}, {'id': 'shared-CC-double-E', 'smiles': 'C/C=C/C=C', 'exact': True, 'has_witness_path': True}, {'id': 'cyclic-olefin', 'smiles': 'C1=CCCCC1', 'exact': True, 'has_witness_path': True}, {'id': 'cyclic-5-one-double', 'smiles': 'C1=CCCC1', 'exact': True, 'has_witness_path': True}, {'id': 'cyclic-7-one-double', 'smiles': 'C1=CCCCCC1', 'exact': True, 'has_witness_path': True}, {'id': 'cyclic-6-two-double', 'smiles': 'C1=CC=CCC1', 'exact': True, 'has_witness_path': True}, {'id': 'cyclic-substituted-one-double', 'smiles': 'C1=CCCCC1O', 'exact': True, 'has_witness_path': True})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def set_process_limit(memory_mib: int) -> None:
    limit = memory_mib * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def fixture_triene(Chem):
    mol = Chem.MolFromSmiles(TRIENE_SMILES)
    if mol is None:
        raise ValueError("pinned triene input did not parse")
    for index, references, stereo in (
        (1, (0, 3), Chem.BondStereo.STEREOE),
        (3, None, Chem.BondStereo.STEREONONE),
        (5, (4, 7), Chem.BondStereo.STEREOE),
    ):
        bond = mol.GetBondWithIdx(index)
        if references is not None:
            bond.SetStereoAtoms(*references)
        bond.SetStereo(stereo)
    for bond in mol.GetBonds():
        bond.SetBondDir(Chem.BondDir.NONE)
    return mol


def mapped_double_bond_facts(source, output, source_bond_indices):
    """Map output stereo labels onto source bonds without canonical text."""
    matches = output.GetSubstructMatches(
        source, uniquify=False, useChirality=False, maxMatches=10_000
    )
    reports = []
    for match in matches:
        inverse = {output_idx: source_idx for source_idx, output_idx in enumerate(match)}
        mapped = []
        for source_bond_index in source_bond_indices:
            source_bond = source.GetBondWithIdx(source_bond_index)
            source_begin, source_end = source_bond.GetBeginAtomIdx(), source_bond.GetEndAtomIdx()
            output_bond = output.GetBondBetweenAtoms(match[source_begin], match[source_end])
            output_stereo = str(output_bond.GetStereo())
            output_references = list(output_bond.GetStereoAtoms())
            if output_bond.GetBeginAtomIdx() != match[source_begin]:
                output_references.reverse()
            mapped_references = [inverse[index] for index in output_references]
            source_references = list(source_bond.GetStereoAtoms())
            effective = output_stereo
            if output_stereo in ("STEREOE", "STEREOZ") and len(source_references) == 2:
                one_reference_flipped = (
                    mapped_references[0] != source_references[0]
                ) ^ (mapped_references[1] != source_references[1])
                if one_reference_flipped:
                    effective = "STEREOZ" if output_stereo == "STEREOE" else "STEREOE"
            mapped.append({
                "source_bond": source_bond_index,
                "source_expected": str(source_bond.GetStereo()),
                "output_raw": output_stereo,
                "output_references_in_source_order": mapped_references,
                "output_effective_in_source_reference_frame": effective,
            })
        reports.append(mapped)
    unique = {json.dumps(report, sort_keys=True) for report in reports}
    return {
        "graph_isomorphism_count": len(matches),
        "all_mapped_stereo_facts": [json.loads(value) for value in sorted(unique)],
    }


def classified_mapped_smiles(Chem, source, strings):
    classified = []
    for text in strings:
        output = Chem.MolFromSmiles(text)
        facts = mapped_double_bond_facts(source, output, (1, 3, 5)) if output else None
        signatures = set()
        if facts is not None:
            for report in facts["all_mapped_stereo_facts"]:
                signatures.add((
                    report[0]["output_effective_in_source_reference_frame"],
                    report[1]["output_raw"],
                    report[2]["output_effective_in_source_reference_frame"],
                ))
        classified.append({
            "smiles": text,
            "parse": output is not None,
            "outer_middle_outer_stereo": [list(value) for value in sorted(signatures)],
            "mapped_bond_facts": facts,
        })
    return classified


def collect_prefix_outputs(prefix, state_cap):
    pending = [(prefix, "")]
    outputs = set()
    visited = 0
    accepted_expandable = 0
    while pending:
        state, text = pending.pop()
        visited += 1
        if visited > state_cap:
            raise RuntimeError(f"prefix walk exceeded {state_cap} states")
        accepted = state.is_accepted
        choices = state.choices()
        if accepted:
            outputs.add(text)
            if choices:
                accepted_expandable += 1
        pending.extend((successor, text + token) for token, successor in choices)
    return outputs, visited, accepted_expandable


def check_prefix_walk_contract():
    class Prefix:
        def __init__(self, accepted, choices=()):
            self.is_accepted = accepted
            self._choices = choices

        def choices(self):
            return self._choices

    longer = Prefix(True)
    start = Prefix(True, (("x", longer),))
    outputs, visited, accepted_expandable = collect_prefix_outputs(start, 2)
    assert outputs == {"", "x"}
    assert visited == 2
    assert accepted_expandable == 1


def chain_facts(mol):
    """Return atom-order-independent facts for a connected carbon chain."""
    ends = [
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetAtomicNum() != 1
        and sum(n.GetAtomicNum() != 1 for n in atom.GetNeighbors()) == 1
    ]
    candidates = []
    for end in ends:
        order, previous = [end], None
        while True:
            next_atoms = [
                atom.GetIdx() for atom in mol.GetAtomWithIdx(order[-1]).GetNeighbors()
                if atom.GetAtomicNum() != 1 and atom.GetIdx() != previous
            ]
            if not next_atoms:
                break
            previous, next_atom = order[-1], next_atoms[0]
            order.append(next_atom)
        if len(order) != mol.GetNumHeavyAtoms():
            continue
        atoms = []
        for index in order:
            atom = mol.GetAtomWithIdx(index)
            atoms.append((
                atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope(),
                atom.GetNumRadicalElectrons(), atom.GetNumExplicitHs(), atom.GetNumImplicitHs(),
            ))
        bonds = []
        for left, right in zip(order, order[1:]):
            bond = mol.GetBondBetweenAtoms(left, right)
            bonds.append((str(bond.GetBondType()), str(bond.GetStereo())))
        candidates.append((tuple(atoms), tuple(bonds)))
    return min(candidates) if candidates else None


def molecule_for_case(Chem, case):
    if case["id"] == "fixture-unspecified-middle-triene":
        return fixture_triene(Chem)
    return Chem.MolFromSmiles(case["smiles"])


def worker(case, role, published_root, wheel_path, source_revision):
    from rdkit import Chem, rdBase

    if rdBase.rdkitVersion != RDKIT_VERSION:
        raise RuntimeError(f"expected RDKit {RDKIT_VERSION}, got {rdBase.rdkitVersion}")
    import grimace
    import grimace._core

    grimace_path = Path(grimace.__file__).resolve()
    core_path = Path(grimace._core.__file__).resolve()
    repo_root = Path(__file__).resolve().parents[3]
    if role == "published":
        if importlib.metadata.version("grimace-py") != PUBLISHED_VERSION:
            raise RuntimeError("published lane did not import grimace-py 0.1.18 metadata")
        expected_root = Path(published_root).resolve()
        if expected_root not in grimace_path.parents or expected_root not in core_path.parents:
            raise RuntimeError("published imports escaped the extracted pinned wheel")
        if sha256(core_path) != PUBLISHED_CORE_SHA256:
            raise RuntimeError("published extension hash differs from the verified wheel")
    else:
        if repo_root == grimace_path or repo_root in grimace_path.parents:
            raise RuntimeError("current lane imported from source tree rather than installed package")
        if sha256(core_path) != CURRENT_CORE_SHA256:
            raise RuntimeError("current extension hash is not the qualified 8a26 artifact")

    mol = molecule_for_case(Chem, case)
    record = {
        "id": case["id"], "smiles": case["smiles"], "lane": role,
        "rdkit_version": rdBase.rdkitVersion,
        "grimace_distribution_version": importlib.metadata.version("grimace-py"),
        "module_origins": {
            "grimace": str(grimace_path), "core": str(core_path),
            "rdkit": str(Path(Chem.__file__).resolve()),
        },
        "core_sha256": sha256(core_path),
        "current_source_revision_claim": source_revision if role == "current" else None,
    }
    if mol is None:
        record["parse"] = "rejected"
        return record
    record["parse"] = "accepted"
    record["atom_count"] = mol.GetNumAtoms()
    record["bond_count"] = mol.GetNumBonds()
    record["input_graph_facts"] = {
        "atoms": [
            {"index": a.GetIdx(), "atomic_number": a.GetAtomicNum(),
             "charge": a.GetFormalCharge(), "isotope": a.GetIsotope(),
             "radical_electrons": a.GetNumRadicalElectrons(),
             "explicit_hs": a.GetNumExplicitHs(), "implicit_hs": a.GetNumImplicitHs()}
            for a in mol.GetAtoms()
        ],
        "bonds": [
            {"index": b.GetIdx(), "begin": b.GetBeginAtomIdx(), "end": b.GetEndAtomIdx(),
             "type": str(b.GetBondType()), "stereo": str(b.GetStereo()),
             "stereo_atoms": list(b.GetStereoAtoms()), "direction": str(b.GetBondDir())}
            for b in mol.GetBonds()
        ],
    }
    if case["id"] == "fixture-unspecified-middle-triene":
        try:
            record["triene_input_stereo_analysis"] = {
                "status": "complete",
                "mapped_double_bonds": mapped_double_bond_facts(mol, mol, (1, 3, 5)),
            }
        except Exception as exc:
            record["triene_input_stereo_analysis"] = {
                "status": "error", "error": f"{type(exc).__name__}: {exc}"
            }

    prefix = None
    start = None
    try:
        from grimace.decoding import start
    except ModuleNotFoundError as exc:
        if role == "published" and exc.name == "grimace.decoding":
            record["south_star2_start"] = {"status": "not available in published release"}
        else:
            record["south_star2_start"] = {
                "status": "import_error", "module": exc.name,
                "error": f"{type(exc).__name__}: {exc}",
            }
    except Exception as exc:
        record["south_star2_start"] = {
            "status": "import_error", "error": f"{type(exc).__name__}: {exc}"
        }
    else:
        if role == "published":
            record["south_star2_start"] = {"status": "present_not_exercised_in_published_lane"}

    if role == "current":
        import grimace._south_star2.preparation as preparation_module
        import grimace.decoding as decoding_module
        preparation_path = Path(preparation_module.__file__).resolve()
        record["module_origins"]["preparation"] = str(preparation_path)
        record["module_origins"]["decoding"] = str(Path(decoding_module.__file__).resolve())
        record["preparation_sha256"] = sha256(preparation_path)
        if record["preparation_sha256"] != CURRENT_PREPARATION_SHA256:
            raise RuntimeError("current preparation source hash differs from the qualified candidate")

    try:
        random_smiles = Chem.MolToRandomSmilesVect(mol, 1, 271828, True, False, False, False)[0]
        parsed = Chem.MolFromSmiles(random_smiles)
        witness = {
            "seed": 271828, "smiles": random_smiles,
            "parse_back": parsed is not None,
            "canonical_graph_roundtrip": None,
        }
        if parsed is not None and case["id"] != "fixture-unspecified-middle-triene":
            original_key = Chem.MolToSmiles(Chem.Mol(mol), canonical=True, isomericSmiles=True)
            parsed_key = Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=True)
            witness["canonical_graph_roundtrip"] = parsed_key == original_key
        record["rdkit_seeded_witness"] = witness
    except Exception as exc:
        record["rdkit_seeded_witness"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    if case["id"] == "fixture-unspecified-middle-triene":
        try:
            record["fixture_seeded_witnesses"] = {}
            for seed in (1, 2):
                text = Chem.MolToRandomSmilesVect(mol, 1, seed, True, False, False, False)[0]
                parsed_seed = Chem.MolFromSmiles(text)
                record["fixture_seeded_witnesses"][str(seed)] = {
                    "smiles": text,
                    "mapped_double_bonds": mapped_double_bond_facts(mol, parsed_seed, (1, 3, 5)) if parsed_seed else None,
                    "outer_middle_outer_stereo": classified_mapped_smiles(Chem, mol, [text])[0]["outer_middle_outer_stereo"],
                }
        except Exception as exc:
            record["fixture_seeded_witnesses"] = {
                "status": "error", "error": f"{type(exc).__name__}: {exc}"
            }

    try:
        decoder = grimace.MolToSmilesDeterminizedDecoder(
            mol, rootedAtAtom=-1, isomericSmiles=True, canonical=False, doRandom=True
        )
        record["published_style_decoder"] = {
            "status": "accepted", "initial_choices": sorted({c.text for c in decoder.next_choices})
        }
    except Exception as exc:
        decoder = None
        record["published_style_decoder"] = {"status": f"{type(exc).__name__}: {exc}"}

    if role == "current":
        try:
            from grimace._south_star2.preparation import prepare_molecule
            prepared = prepare_molecule(mol)
            record["south_star2_preparation"] = {"status": "accepted", "prepared_atoms": len(prepared.atoms)}
        except Exception as exc:
            record["south_star2_preparation"] = {"status": f"{type(exc).__name__}: {exc}"}

        if start is not None:
            try:
                prefix = start(mol)
                record["south_star2_start"] = {"status": "accepted"}
            except Exception as exc:
                record["south_star2_start"] = {
                    "status": "error", "error": f"{type(exc).__name__}: {exc}"
                }

        if prefix is not None:
            try:
                choices = prefix.choices()
                record["south_star2_initial_choices"] = {
                    "status": "complete", "tokens": sorted({token for token, _ in choices})
                }
            except Exception as exc:
                record["south_star2_initial_choices"] = {
                    "status": "error", "error": f"{type(exc).__name__}: {exc}"
                }

            if case["exact"]:
                try:
                    outputs, visited, accepted_expandable = collect_prefix_outputs(
                        prefix, MAX_PREFIX_STATES
                    )
                    record["south_star2_prefix_walk"] = {
                        "status": "complete", "visited_states": visited,
                        "accepted_expandable_states": accepted_expandable,
                        "output_count": len(outputs), "outputs": sorted(outputs),
                    }
                except Exception as exc:
                    record["south_star2_prefix_walk"] = {
                        "status": "error", "error": f"{type(exc).__name__}: {exc}",
                        "state_cap": MAX_PREFIX_STATES,
                    }
                if case["id"] == "fixture-unspecified-middle-triene" and record.get(
                    "south_star2_prefix_walk", {}
                ).get("status") == "complete":
                    try:
                        outputs = record["south_star2_prefix_walk"]["outputs"]
                        record["south_star2_triene_output_analysis"] = {
                            "status": "complete",
                            "mapped_output_stereo": classified_mapped_smiles(Chem, mol, outputs),
                        }
                    except Exception as exc:
                        record["south_star2_triene_output_analysis"] = {
                            "status": "error", "error": f"{type(exc).__name__}: {exc}"
                        }

            if decoder is not None and case["has_witness_path"]:
                try:
                    target = record["rdkit_seeded_witness"]["smiles"]
                    rd_state, ss_state, offset = decoder, prefix, 0
                    deltas, steps, unique = [], 0, True
                    while offset < len(target):
                        rd_choices, ss_choices = tuple(rd_state.next_choices), tuple(ss_state.choices())
                        rd_tokens = sorted({choice.text for choice in rd_choices})
                        ss_tokens = sorted({token for token, _ in ss_choices})
                        if rd_tokens != ss_tokens:
                            deltas.append({"offset": offset, "published_only": sorted(set(rd_tokens)-set(ss_tokens)),
                                           "current_only": sorted(set(ss_tokens)-set(rd_tokens))})
                        rd_matches = [choice for choice in rd_choices if target.startswith(choice.text, offset)]
                        ss_matches = [(token, successor) for token, successor in ss_choices if target.startswith(token, offset)]
                        if len(rd_matches) != 1 or len(ss_matches) != 1 or rd_matches[0].text != ss_matches[0][0]:
                            unique = False
                            break
                        token = ss_matches[0][0]
                        rd_state, ss_state = rd_matches[0].next_state, ss_matches[0][1]
                        offset += len(token)
                        steps += 1
                    record["selected_witness_path"] = {
                        "status": "complete" if unique and offset == len(target) else "not-uniquely-consumable",
                        "steps": steps, "choice_deltas": deltas, "fully_consumed": offset == len(target),
                        "current_accepted_at_end": bool(ss_state.is_accepted) if offset == len(target) else None,
                        "published_terminal_at_end": bool(rd_state.is_terminal) if offset == len(target) else None,
                    }
                except Exception as exc:
                    record["selected_witness_path"] = {
                        "status": "error", "error": f"{type(exc).__name__}: {exc}"
                    }

    if case["exact"]:
        try:
            outputs = sorted(set(grimace.MolToSmilesEnum(
                mol, rootedAtAtom=-1, isomericSmiles=True, canonical=False, doRandom=True
            )))
            record["package_enum_output"] = {
                "status": "enumerated", "count": len(outputs), "strings": outputs,
                "authority": "package output only; not independently source-proven RDKit completeness",
            }
            if case["id"] == "fixture-unspecified-middle-triene":
                try:
                    record["package_enum_triene_analysis"] = {
                        "status": "complete",
                        "mapped_output_stereo": classified_mapped_smiles(Chem, mol, outputs),
                    }
                except Exception as exc:
                    record["package_enum_triene_analysis"] = {
                        "status": "error", "error": f"{type(exc).__name__}: {exc}"
                    }
        except Exception as exc:
            record["package_enum_output"] = {"status": f"{type(exc).__name__}: {exc}"}
    else:
        record["package_enum_output"] = {"status": "not enumerated: broad-corpus admission/witness only"}

    if role == "current" and case["id"] == "fixture-unspecified-middle-triene":
        try:
            params = Chem.SmilesParserParams()
            params.removeHs = False
            explicit = Chem.MolFromSmiles(TRIENE_EXPLICIT_H, params)
            normalized = Chem.RemoveHs(explicit) if explicit is not None else None
            default_parsed = Chem.MolFromSmiles(TRIENE_EXPLICIT_H)
            record["explicit_h_normalization_probe"] = {
                "status": "complete",
                "candidate": TRIENE_EXPLICIT_H,
                "retained_h_parse_atom_count": explicit.GetNumAtoms() if explicit else None,
                "retained_h_double_bond_stereo": [str(b.GetStereo()) for b in explicit.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE] if explicit else None,
                "after_Chem_RemoveHs_atom_count": normalized.GetNumAtoms() if normalized else None,
                "after_remove_h_chain_facts_equal_input": normalized is not None and chain_facts(normalized) == chain_facts(mol),
                "after_remove_h_mapped_stereo": mapped_double_bond_facts(mol, normalized, (1, 3, 5)) if normalized else None,
                "after_remove_h_outer_middle_outer_stereo": classified_mapped_smiles(Chem, mol, [TRIENE_EXPLICIT_H])[0]["outer_middle_outer_stereo"] if normalized else None,
                "default_parser_double_bond_stereo": [str(b.GetStereo()) for b in default_parsed.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE] if default_parsed else None,
            }
        except Exception as exc:
            record["explicit_h_normalization_probe"] = {
                "status": "error", "error": f"{type(exc).__name__}: {exc}"
            }
    return record


def verify_published_extraction(wheel: Path, root: Path) -> None:
    if sha256(wheel) != PUBLISHED_WHEEL_SHA256:
        raise RuntimeError("published wheel SHA-256 mismatch")
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("grimace/") and not name.endswith("/"):
                extracted = root / name
                if not extracted.is_file() or extracted.read_bytes() != archive.read(name):
                    raise RuntimeError(f"published extraction differs from wheel member {name}")


def worker_process(case, role, args):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    if role == "published":
        env["PYTHONPATH"] = str(Path(args.published_root).resolve())
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        json.dumps(case, separators=(",", ":")), role,
        str(Path(args.published_root).resolve()), str(Path(args.published_wheel).resolve()),
        args.current_source_revision,
    ]
    try:
        completed = subprocess.run(
            command, env=env, capture_output=True, text=True,
            timeout=args.timeout_seconds,
            preexec_fn=lambda: set_process_limit(args.memory_limit_mib),
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "timeout_seconds": args.timeout_seconds,
                "memory_limit_mib": args.memory_limit_mib}
    if completed.returncode != 0:
        return {"status": "worker_failure", "returncode": completed.returncode,
                "stderr": completed.stderr[-2000:], "stdout": completed.stdout[-1000:]}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {"status": "invalid_worker_json", "error": str(exc), "stdout": completed.stdout[-2000:]}


def run(args):
    check_prefix_walk_contract()
    wheel = Path(args.published_wheel).resolve()
    root = Path(args.published_root).resolve()
    verify_published_extraction(wheel, root)
    cases = [dict(case) for case in CASES]
    cases.append({
        "id": "fixture-unspecified-middle-triene", "smiles": TRIENE_SMILES,
        "exact": True, "has_witness_path": True,
    })
    records = []
    for case in cases:
        lane_records = {
            role: worker_process(case, role, args) for role in ("published", "current")
        }
        record = {"id": case["id"], "smiles": case["smiles"], "exact_case": case["exact"],
                  "lanes": lane_records}
        published = lane_records["published"]
        current = lane_records["current"]
        pub_outputs = published.get("package_enum_output", {}).get("strings")
        cur_outputs = current.get("south_star2_prefix_walk", {}).get("outputs")
        if isinstance(pub_outputs, list) and isinstance(cur_outputs, list):
            record["published_package_vs_ss2"] = {
                "published_only": sorted(set(pub_outputs) - set(cur_outputs)),
                "ss2_only": sorted(set(cur_outputs) - set(pub_outputs)),
                "published_package_output_count": len(pub_outputs),
                "ss2_output_count": len(cur_outputs),
            }
        records.append(record)
    exact_records = [r for r in records if r["exact_case"] and "published_package_vs_ss2" in r]
    matrix_records = records[:-1]
    matrix_equal = [r for r in matrix_records if r["exact_case"]
                    and "published_package_vs_ss2" in r
                    and not r["published_package_vs_ss2"]["published_only"]
                    and not r["published_package_vs_ss2"]["ss2_only"]]
    report = {
        "evidence_boundary": {
            "published_enumeration": "versioned PyPI package output; no independent source proof of completeness",
            "ss2_enumeration": "complete bounded walk of public Prefix choices for cases marked exact",
            "rdkit_strings": "seeded positive membership witnesses only",
            "partial_stereo": "mapped configured bond facts and H-normalization behavior; not a normative chemistry claim",
        },
        "provenance": {
            "current_source_revision_claim": args.current_source_revision,
            "expected_base_revision": BASE_REVISION,
            "rdkit_version": RDKIT_VERSION,
            "published_distribution_version": PUBLISHED_VERSION,
            "published_wheel_path": str(wheel), "published_wheel_sha256": sha256(wheel),
            "published_extracted_root": str(root),
            "published_core_sha256": PUBLISHED_CORE_SHA256,
            "current_core_sha256": CURRENT_CORE_SHA256,
            "current_preparation_sha256": CURRENT_PREPARATION_SHA256,
            "python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "timeout_seconds_per_lane_case": args.timeout_seconds,
            "address_space_limit_mib_per_lane_case": args.memory_limit_mib,
            "prefix_state_cap": MAX_PREFIX_STATES,
            "seed": 271828, "special_triene_seeds": [1, 2],
        },
        "counts": {
            "matrix_admission_cases": len(CASES),
            "matrix_complete_exact_support_cases": len([r for r in matrix_records if r["exact_case"] and "published_package_vs_ss2" in r]),
            "matrix_complete_exact_equal_cases": len(matrix_equal),
            "matrix_complete_exact_delta_cases": len([r for r in matrix_records if r["exact_case"] and "published_package_vs_ss2" in r and (r["published_package_vs_ss2"]["published_only"] or r["published_package_vs_ss2"]["ss2_only"])]),
            "separate_triene_complete_exact_comparison": "published-package and SS2 outputs retained as a separate partial-stereo audit",
        },
        "cases": records,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-wheel")
    parser.add_argument("--published-root")
    parser.add_argument("--current-source-revision", default=BASE_REVISION)
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    parser.add_argument("--memory-limit-mib", type=int, default=2048)
    parser.add_argument("--output")
    parser.add_argument("--worker", nargs=5, metavar=("CASE_JSON", "LANE", "PUBLISHED_ROOT", "PUBLISHED_WHEEL", "SOURCE_REVISION"))
    args = parser.parse_args()
    if args.worker:
        case_text, role, published_root, published_wheel, source_revision = args.worker
        result = worker(json.loads(case_text), role, published_root, published_wheel, source_revision)
        print(json.dumps(result, sort_keys=True))
        return
    if not args.published_wheel or not args.published_root:
        parser.error("--published-wheel and --published-root are required")
    run(args)


if __name__ == "__main__":
    main()
