from __future__ import annotations

import importlib.util
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, rdBase


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "stereo_z3",
    ROOT / "tmp/exploration/stereo_assignment/023_investigate_stereo_constraint_system_z3.py",
)
stereo_z3 = importlib.util.module_from_spec(SPEC)
sys.modules["stereo_z3"] = stereo_z3
assert SPEC.loader is not None
SPEC.loader.exec_module(stereo_z3)


REDUCED_PORPHYRIN = (
    "c1cc2[n-]c1/N=c1/cc/c([n-]1)=N/c1ccc([n-]1)/N=c1/cc/c([n-]1)=N/2"
)


@dataclass(frozen=True)
class TraversalObservation:
    mode: str
    smiles: str
    shape: tuple[tuple[int, int], ...]
    bits: tuple[int, ...]
    atom_order: tuple[int, ...]
    bond_order: tuple[int, ...]
    final_bond_dirs: tuple[tuple[int, int, int, str], ...]
    inferred_tree_bonds: tuple[int, ...]
    inferred_closure_bonds: tuple[int, ...]


def two_choice_endpoints(system: object) -> tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...]:
    out = []
    for endpoint_id, endpoint in enumerate(system.endpoints):
        if len(endpoint.candidate_neighbors) != 2:
            continue
        edges = tuple(
            stereo_z3.canonical_edge(endpoint.endpoint, neighbor)
            for neighbor in endpoint.candidate_neighbors
        )
        out.append((endpoint_id, endpoint.endpoint, edges))
    return tuple(out)


def bit_pattern(
    shape: tuple[tuple[int, int], ...],
    two_choice: tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...],
) -> tuple[int, ...]:
    directed = set(shape)
    return tuple(1 if edges[1] in directed else 0 for _endpoint, _atom, edges in two_choice)


def collect_output_mols(source: Chem.Mol, *, samples: int) -> list[tuple[str, str, Chem.Mol]]:
    out: dict[str, tuple[str, Chem.Mol]] = {}

    def add(mode: str, mol: Chem.Mol) -> None:
        smiles = Chem.MolToSmiles(
            mol,
            canonical=False,
            doRandom=False,
            isomericSmiles=True,
        )
        out.setdefault(smiles, (mode, mol))

    for root in range(source.GetNumAtoms()):
        mol = Chem.Mol(source)
        smiles = Chem.MolToSmiles(
            mol,
            canonical=False,
            doRandom=False,
            rootedAtAtom=root,
            isomericSmiles=True,
        )
        out.setdefault(smiles, (f"root:{root}", mol))

    for seed in range(samples):
        mol = Chem.Mol(source)
        Chem.rdBase.SeedRandomNumberGenerator(seed)
        smiles = Chem.MolToSmiles(
            mol,
            canonical=False,
            doRandom=True,
            isomericSmiles=True,
        )
        out.setdefault(smiles, (f"random:{seed}", mol))

    return [(mode, smiles, mol) for smiles, (mode, mol) in sorted(out.items())]


def order_prop(prop_text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in prop_text.strip("[]").split(",") if part.strip())


def final_bond_dirs(mol: Chem.Mol) -> tuple[tuple[int, int, int, str], ...]:
    out = []
    for bond in mol.GetBonds():
        direction = str(bond.GetBondDir())
        if direction != "NONE":
            out.append(
                (
                    bond.GetIdx(),
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    direction,
                )
            )
    return tuple(out)


def bond_atoms(mol: Chem.Mol, bond_idx: int) -> tuple[int, int]:
    bond = mol.GetBondWithIdx(bond_idx)
    return bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()


def bond_idx_by_edge(mol: Chem.Mol) -> dict[tuple[int, int], int]:
    return {stereo_z3.canonical_edge(*bond_atoms(mol, idx)): idx for idx in range(mol.GetNumBonds())}


def ring_choice_pairs(
    mol: Chem.Mol,
    two_choice: tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...],
) -> list[tuple[int, int, tuple[int, ...], tuple[int, ...]]]:
    atom_rings = [tuple(ring) for ring in mol.GetRingInfo().AtomRings()]
    out = []
    for left_idx, (_left_endpoint_id, left_atom, left_edges) in enumerate(two_choice):
        left_neighbors = {edge[0] ^ edge[1] ^ left_atom for edge in left_edges}
        for right_idx, (_right_endpoint_id, right_atom, right_edges) in enumerate(two_choice):
            if left_idx >= right_idx:
                continue
            right_neighbors = {edge[0] ^ edge[1] ^ right_atom for edge in right_edges}
            for ring in atom_rings:
                ring_set = set(ring)
                if left_atom not in ring_set or right_atom not in ring_set:
                    continue
                if not left_neighbors & ring_set or not right_neighbors & ring_set:
                    continue
                out.append((left_idx, right_idx, ring, tuple(sorted(ring_set))))
    return out


def infer_tree_and_closure_bonds(
    mol: Chem.Mol,
    atom_order: tuple[int, ...],
    bond_order: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Infer DFS tree vs ring-closure bonds from RDKit output metadata.

    RDKit does not expose the molStack used by `canonicalizeFragment()`.
    `_smilesAtomOutputOrder` and `_smilesBondOutputOrder` are enough to
    recover a useful approximation: each new atom needs one incoming bond from
    the already-emitted atom set; bonds encountered before that which connect
    two already-emitted atoms are ring closures.
    """

    seen = {atom_order[0]}
    tree: list[int] = []
    closures: list[int] = []
    cursor = 0
    used: set[int] = set()

    for atom in atom_order[1:]:
        incoming = None
        while cursor < len(bond_order):
            bond_idx = bond_order[cursor]
            cursor += 1
            if bond_idx in used:
                continue
            begin, end = bond_atoms(mol, bond_idx)
            if begin in seen and end in seen:
                closures.append(bond_idx)
                used.add(bond_idx)
                continue
            if atom in (begin, end) and (begin in seen or end in seen):
                incoming = bond_idx
                used.add(bond_idx)
                break
        if incoming is None:
            raise AssertionError(f"could not infer incoming bond for atom {atom}")
        tree.append(incoming)
        seen.add(atom)

    for bond_idx in bond_order[cursor:]:
        if bond_idx not in used:
            closures.append(bond_idx)
            used.add(bond_idx)

    return tuple(tree), tuple(closures)


def observations(source_smiles: str, *, samples: int) -> tuple[object, tuple[TraversalObservation, ...]]:
    source = Chem.MolFromSmiles(source_smiles)
    assert source is not None
    system = stereo_z3.stereo_system_from_mol(source_smiles, source)
    two_choice = two_choice_endpoints(system)
    output_mols = collect_output_mols(source, samples=samples)
    out = []
    for mode, smiles, output_mol in output_mols:
        parsed = Chem.MolFromSmiles(smiles)
        if parsed is None:
            continue
        matches = stereo_z3.preserving_matches(source, parsed, system.stereo_bonds)
        if not matches:
            continue
        match = matches[0]
        shape = stereo_z3.mapped_directed_edges(parsed, match)
        atom_order = order_prop(output_mol.GetProp("_smilesAtomOutputOrder"))
        bond_order = order_prop(output_mol.GetProp("_smilesBondOutputOrder"))
        tree_bonds, closure_bonds = infer_tree_and_closure_bonds(
            source,
            atom_order,
            bond_order,
        )
        out.append(
            TraversalObservation(
                mode=mode,
                smiles=smiles,
                shape=shape,
                bits=bit_pattern(shape, two_choice),
                atom_order=atom_order,
                bond_order=bond_order,
                final_bond_dirs=final_bond_dirs(output_mol),
                inferred_tree_bonds=tree_bonds,
                inferred_closure_bonds=closure_bonds,
            )
        )
    return system, tuple(out)


def model_bits(system: object) -> dict[tuple[int, ...], tuple[tuple[int, int], ...]]:
    hypothesis = next(
        hypothesis
        for hypothesis in stereo_z3.HYPOTHESES
        if hypothesis.name == "observed-edge-shared-pair"
    )
    shapes = stereo_z3.Z3CarrierModel(system, hypothesis).projected_directed_edge_shapes()
    two_choice = two_choice_endpoints(system)
    return {bit_pattern(shape, two_choice): shape for shape in shapes}


def print_report() -> None:
    source = Chem.MolFromSmiles(REDUCED_PORPHYRIN)
    assert source is not None
    edge_to_bond = bond_idx_by_edge(source)
    system, obs = observations(REDUCED_PORPHYRIN, samples=32768)
    bits_to_shape = model_bits(system)
    observed_bits = {item.bits for item in obs}
    missing_bits = sorted(set(bits_to_shape) - observed_bits)
    by_bits: dict[tuple[int, ...], list[TraversalObservation]] = defaultdict(list)
    for item in obs:
        by_bits[item.bits].append(item)

    print("rdkit:", rdBase.rdkitVersion)
    print("source:", REDUCED_PORPHYRIN)
    print("observed outputs:", len(obs))
    print("model bit patterns:", len(bits_to_shape))
    print("observed bit patterns:", len(observed_bits))
    print("missing bit patterns:", missing_bits)
    print()
    print("two-choice endpoint legend:")
    for bit_idx, (endpoint_id, atom_idx, edges) in enumerate(two_choice_endpoints(system)):
        print(f"  b{bit_idx}: endpoint={endpoint_id} atom={atom_idx} 0={edges[0]} 1={edges[1]}")

    print()
    for bits in sorted(bits_to_shape):
        status = "OBSERVED" if bits in observed_bits else "MISSING"
        members = tuple(by_bits.get(bits, ()))
        print("=" * 100)
        print(bits, status, "shape=", bits_to_shape[bits], "outputs=", len(members))
        if members:
            print("  example:", members[0].mode, members[0].smiles)
            print("  atom_order:", members[0].atom_order)
            print("  bond_order:", members[0].bond_order)
            print("  inferred_tree_bonds:", members[0].inferred_tree_bonds)
            print("  inferred_closure_bonds:", members[0].inferred_closure_bonds)
            print("  directed:", members[0].shape)
            print("  final_bond_dirs:", members[0].final_bond_dirs)
            start_counts = Counter(member.atom_order[0] for member in members)
            first_two_counts = Counter(member.atom_order[:2] for member in members)
            print("  root/start counts:", start_counts.most_common(8))
            print("  first-two counts:", first_two_counts.most_common(8))

    print()
    print("missing relation probes:")
    all_bits = sorted(bits_to_shape)
    for left in range(4):
        for right in range(left + 1, 4):
            missing_with = [bits for bits in missing_bits if bits[left] != bits[right]]
            observed_with = [bits for bits in observed_bits if bits[left] != bits[right]]
            print(
                f"  b{left}!=b{right}:",
                f"missing={len(missing_with)}/{len(missing_bits)}",
                f"observed={len(observed_with)}/{len(observed_bits)}",
            )
    print("  missing all:", missing_bits)

    print()
    print("ring-local two-choice pairs:")
    for left_idx, right_idx, ring, _ring_set in ring_choice_pairs(source, two_choice_endpoints(system)):
        print(f"  b{left_idx}/b{right_idx}: ring={ring}")

    print()
    print("chosen-edge tree/closure roles:")
    two_choice = two_choice_endpoints(system)
    for bits in sorted(observed_bits):
        role_counts = Counter()
        for member in by_bits[bits]:
            roles = []
            for bit, (_endpoint_id, _atom_idx, edges) in zip(bits, two_choice, strict=True):
                bond_idx = edge_to_bond[edges[bit]]
                if bond_idx in member.inferred_closure_bonds:
                    roles.append("C")
                elif bond_idx in member.inferred_tree_bonds:
                    roles.append("T")
                else:
                    roles.append("?")
            role_counts[tuple(roles)] += 1
        print(f"  {bits}: {role_counts.most_common(6)}")

    print()
    print("cycle-pair policy probe:")
    pair_edges = []
    two_choice = two_choice_endpoints(system)
    for left_idx, (_left_endpoint_id, left_atom, left_edges) in enumerate(two_choice):
        for right_idx, (_right_endpoint_id, right_atom, right_edges) in enumerate(two_choice):
            if left_idx >= right_idx:
                continue
            shared = set(left_edges) & set(right_edges)
            if shared:
                pair_edges.append((left_idx, right_idx, tuple(sorted(shared))[0]))
            elif source_path_exists_between_choice_atoms(system, left_atom, right_atom):
                pass
    print("  shared-edge pairs:", pair_edges)
    if len(two_choice) == 4:
        candidate_forbidden = [
            bits
            for bits in sorted(bits_to_shape)
            if (bits[0] != bits[1]) and (bits[2] != bits[3])
        ]
        print("  forbid both opposing pair disagreements:", candidate_forbidden)
        print("  equals missing:", candidate_forbidden == missing_bits)


def source_path_exists_between_choice_atoms(
    system: object,
    left_atom: int,
    right_atom: int,
) -> bool:
    # Placeholder hook for follow-up cycle-basis experiments. Kept deliberately
    # side-effect free so this tmp script documents that the current result is
    # still a bit-pattern observation, not yet a derived graph theorem.
    return bool(system and left_atom >= 0 and right_atom >= 0)


if __name__ == "__main__":
    print_report()
