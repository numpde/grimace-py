"""Mine dataset-backed candidates for RDKit writer support-count fixtures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re

from rdkit_writer_support_count_surfaces import (
    CANDIDATE_MINING_SURFACE_FLAGS,
    surface_flags,
)


@dataclass(frozen=True)
class Candidate:
    case_id: str
    cid: str
    name: str
    smiles: str
    canonical_smiles: str
    rooted_at_atom: int
    atom_count: int
    support_count: int
    feature_labels: tuple[str, ...]

    @property
    def score(self) -> tuple[int, int, str]:
        return (self.support_count, len(self.feature_labels), self.cid)


def _slug(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return normalized or "molecule"


def case_id_for(*, cid: str, surface: str, rooted_at_atom: int) -> str:
    root = "unrooted" if rooted_at_atom == -1 else f"root{rooted_at_atom}"
    return f"pubchem_{cid}_{_slug(surface)}_{root}"


def support_bucket(support_count: int) -> str:
    if support_count <= 100:
        return "small"
    if support_count <= 2_000:
        return "medium"
    if support_count <= 10_000:
        return "large"
    return "too_large"


def feature_labels(mol: object, *, support_count: int) -> tuple[str, ...]:
    from rdkit import Chem

    labels: list[str] = [f"support_{support_bucket(support_count)}"]
    fragment_count = len(Chem.GetMolFrags(mol))
    labels.append("connected" if fragment_count <= 1 else "disconnected")
    if any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()):
        labels.append("atom_stereo")
    if any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE or bond.GetBondDir() != Chem.BondDir.NONE
        for bond in mol.GetBonds()
    ):
        labels.append("bond_stereo")
    if mol.GetRingInfo().NumRings() > 0:
        labels.append("ring")
    if mol.GetRingInfo().NumRings() > 1:
        labels.append("multiple_rings")
    if any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
        labels.append("aromatic")
    if any(atom.GetAtomicNum() not in (1, 6) for atom in mol.GetAtoms()):
        labels.append("hetero_atoms")
    if any(atom.GetAtomicNum() in (9, 17, 35, 53) for atom in mol.GetAtoms()):
        labels.append("halogen")
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        labels.append("charged")
    if any(atom.GetIsotope() != 0 for atom in mol.GetAtoms()):
        labels.append("isotope")
    if any(atom.GetAtomMapNum() != 0 for atom in mol.GetAtoms()):
        labels.append("atom_map")
    return tuple(labels)


def _matches_molecule_filter(mol: object, molecule_filter: str) -> bool:
    from grimace._reference.dataset import (
        molecule_has_stereochemistry,
        molecule_is_connected,
    )

    if molecule_filter == "any":
        return True
    if molecule_filter == "connected":
        return molecule_is_connected(mol)
    if molecule_filter == "disconnected":
        return not molecule_is_connected(mol)
    if molecule_filter == "stereo":
        return molecule_has_stereochemistry(mol)
    if molecule_filter == "nonstereo":
        return not molecule_has_stereochemistry(mol)
    raise ValueError(f"unknown molecule filter: {molecule_filter!r}")


def _support_count(mol: object, *, flags: dict[str, bool], rooted_at_atom: int) -> int:
    import grimace

    return sum(
        1
        for _ in grimace.MolToSmilesEnum(
            mol,
            rootedAtAtom=rooted_at_atom,
            **flags,
        )
    )


def mine_candidates(args: argparse.Namespace) -> dict[str, object]:
    from rdkit import Chem, rdBase

    from grimace._reference.dataset import iter_default_molecule_cases

    flags = surface_flags(args.surface)
    candidates: list[Candidate] = []
    scanned_count = 0
    parsed_count = 0
    eligible_count = 0
    skipped_too_small = 0
    skipped_too_large = 0

    for case in iter_default_molecule_cases(
        limit=args.limit,
        max_smiles_length=args.max_smiles_length,
    ):
        scanned_count += 1
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None:
            continue
        parsed_count += 1
        atom_count = mol.GetNumAtoms()
        if atom_count > args.max_atoms:
            continue
        if not _matches_molecule_filter(mol, args.molecule_filter):
            continue
        eligible_count += 1

        count = _support_count(mol, flags=flags, rooted_at_atom=args.rooted_at_atom)
        if count < args.min_support_count:
            skipped_too_small += 1
            continue
        if count > args.max_support_count:
            skipped_too_large += 1
            continue

        candidates.append(
            Candidate(
                case_id=case_id_for(
                    cid=case.cid,
                    surface=args.surface,
                    rooted_at_atom=args.rooted_at_atom,
                ),
                cid=case.cid,
                name=case.name,
                smiles=case.smiles,
                canonical_smiles=Chem.MolToSmiles(mol, canonical=True),
                rooted_at_atom=args.rooted_at_atom,
                atom_count=atom_count,
                support_count=count,
                feature_labels=feature_labels(mol, support_count=count),
            )
        )

    ranked = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    selected = ranked[: args.max_candidates]
    selected_cases = [
        {
            "id": candidate.case_id,
            "source": (
                "Dataset-derived PubChem top_100000 candidate "
                f"(CID {candidate.cid}, Grimace exact support pre-screen)"
            ),
            "smiles": candidate.smiles,
            "rooted_at_atom": candidate.rooted_at_atom,
        }
        for candidate in selected
    ]
    return {
        "rdkit_version": rdBase.rdkitVersion,
        "surface": args.surface,
        "flags": flags,
        "rooted_at_atom": args.rooted_at_atom,
        "selection": {
            "limit": args.limit,
            "max_smiles_length": args.max_smiles_length,
            "max_atoms": args.max_atoms,
            "molecule_filter": args.molecule_filter,
            "min_support_count": args.min_support_count,
            "max_support_count": args.max_support_count,
            "max_candidates": args.max_candidates,
        },
        "summary": {
            "scanned_count": scanned_count,
            "parsed_count": parsed_count,
            "eligible_count": eligible_count,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "skipped_too_small_count": skipped_too_small,
            "skipped_too_large_count": skipped_too_large,
        },
        "candidates": [
            {
                "id": candidate.case_id,
                "cid": candidate.cid,
                "name": candidate.name,
                "smiles": candidate.smiles,
                "canonical_smiles": candidate.canonical_smiles,
                "rooted_at_atom": candidate.rooted_at_atom,
                "atom_count": candidate.atom_count,
                "support_count": candidate.support_count,
                "feature_labels": list(candidate.feature_labels),
            }
            for candidate in selected
        ],
        "generator_input": {"cases": selected_cases},
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mine dataset-backed candidates for RDKit writer support-count "
            "saturation probes."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--surface",
        choices=sorted(CANDIDATE_MINING_SURFACE_FLAGS),
        default="nonisomeric__random",
    )
    parser.add_argument("--rooted-at-atom", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=1_000)
    parser.add_argument("--max-smiles-length", type=int, default=140)
    parser.add_argument("--max-atoms", type=int, default=45)
    parser.add_argument("--min-support-count", type=int, default=100)
    parser.add_argument("--max-support-count", type=int, default=2_000)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument(
        "--molecule-filter",
        choices=("any", "connected", "disconnected", "stereo", "nonstereo"),
        default="any",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be nonnegative")
    if args.max_smiles_length < 0:
        raise SystemExit("--max-smiles-length must be nonnegative")
    if args.max_atoms <= 0:
        raise SystemExit("--max-atoms must be positive")
    if args.min_support_count <= 0:
        raise SystemExit("--min-support-count must be positive")
    if args.max_support_count < args.min_support_count:
        raise SystemExit("--max-support-count must be at least --min-support-count")
    if args.max_candidates <= 0:
        raise SystemExit("--max-candidates must be positive")
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists; pass --force to overwrite it")

    payload = mine_candidates(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
