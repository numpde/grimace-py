from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

from rdkit import Chem, RDLogger


MarkerSlots = tuple[tuple[int, str], ...]
MarkerSlotSet = tuple[MarkerSlots, ...]


@dataclass(frozen=True, slots=True)
class WriterMarkerSlotQuotientDiagnostic:
    emitted_marker_slots: MarkerSlots
    semantic_row_accepts: bool
    marker_slot_quotient_candidate: bool
    rdkit_writer_target_slots: bool


def direction_erased_skeleton(smiles: str) -> str:
    return "".join(char for char in smiles if char not in {"/", "\\"})


def direction_marker_slots(smiles: str) -> MarkerSlots:
    slots = []
    skeleton_slot = 0
    for char in smiles:
        if char in {"/", "\\"}:
            slots.append((skeleton_slot, char))
        else:
            skeleton_slot += 1
    return tuple(slots)


def emitted_marker_slots_from_attempt(attempt: dict[str, object]) -> MarkerSlots:
    raw_slots = attempt["emitted_marker_slots"]
    if type(raw_slots) is not list:
        raise AssertionError("emitted_marker_slots diagnostic must be a list")
    return tuple((slot, marker) for slot, marker in raw_slots)


def writer_marker_slot_quotient_diagnostic(
    *,
    emitted_marker_slots: MarkerSlots,
    semantic_row_accepts: bool,
    parse_equivalent_marker_slots: MarkerSlotSet,
    rdkit_expected_smiles: str,
) -> WriterMarkerSlotQuotientDiagnostic:
    return WriterMarkerSlotQuotientDiagnostic(
        emitted_marker_slots=emitted_marker_slots,
        semantic_row_accepts=semantic_row_accepts,
        marker_slot_quotient_candidate=(
            not semantic_row_accepts
            and emitted_marker_slots in parse_equivalent_marker_slots
        ),
        rdkit_writer_target_slots=(
            emitted_marker_slots == direction_marker_slots(rdkit_expected_smiles)
        ),
    )


def smiles_from_direction_marker_slots(
    skeleton: str,
    marker_slots: MarkerSlots,
) -> str:
    markers_by_slot = dict(marker_slots)
    parts = []
    for slot, char in enumerate(skeleton):
        if slot in markers_by_slot:
            parts.append(markers_by_slot[slot])
        parts.append(char)
    if len(skeleton) in markers_by_slot:
        parts.append(markers_by_slot[len(skeleton)])
    return "".join(parts)


def directional_bond_signature(mol: Chem.Mol) -> tuple[tuple[int, int, str], ...]:
    return tuple(
        sorted(
            (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                str(bond.GetBondDir()),
            )
            for bond in mol.GetBonds()
            if bond.GetBondDir() != Chem.BondDir.NONE
        )
    )


def double_bond_stereo_signature(
    mol: Chem.Mol,
) -> tuple[tuple[int, int, str, tuple[int, ...]], ...]:
    return tuple(
        sorted(
            (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                str(bond.GetStereo()),
                tuple(bond.GetStereoAtoms()),
            )
            for bond in mol.GetBonds()
            if bond.GetStereo() != Chem.BondStereo.STEREONONE
        )
    )


def canonical_isomeric_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(
        Chem.Mol(mol),
        canonical=True,
        isomericSmiles=True,
    )


def _single_marker_candidate_slots(skeleton: str) -> tuple[int, ...]:
    candidate_slots = []
    RDLogger.DisableLog("rdApp.*")
    try:
        for slot in range(len(skeleton) + 1):
            if any(
                Chem.MolFromSmiles(
                    smiles_from_direction_marker_slots(skeleton, ((slot, marker),))
                )
                is not None
                for marker in ("/", "\\")
            ):
                candidate_slots.append(slot)
    finally:
        RDLogger.EnableLog("rdApp.*")
    return tuple(candidate_slots)


def parse_equivalent_minimal_marker_slot_sets(
    skeleton: str,
    reference_mol: Chem.Mol,
) -> MarkerSlotSet:
    # A minimal directional-bond basis needs at most two marker-bearing
    # neighboring bonds per double bond; shared markers can reduce that count.
    max_marker_count = 2 * len(double_bond_stereo_signature(reference_mol))
    reference_canonical_smiles = canonical_isomeric_smiles(reference_mol)
    candidate_slots = _single_marker_candidate_slots(skeleton)
    valid_marker_slot_sets = []

    RDLogger.DisableLog("rdApp.*")
    try:
        for marker_count in range(max_marker_count + 1):
            for slots in combinations(candidate_slots, marker_count):
                for markers in product(("/", "\\"), repeat=marker_count):
                    marker_slots = tuple(zip(slots, markers))
                    mol = Chem.MolFromSmiles(
                        smiles_from_direction_marker_slots(skeleton, marker_slots)
                    )
                    if (
                        mol is not None
                        and canonical_isomeric_smiles(mol)
                        == reference_canonical_smiles
                    ):
                        valid_marker_slot_sets.append(marker_slots)

        minimal_marker_slot_sets = []
        for marker_slots in valid_marker_slot_sets:
            is_minimal = True
            for marker_idx in range(len(marker_slots)):
                reduced_marker_slots = tuple(
                    marker_slot
                    for idx, marker_slot in enumerate(marker_slots)
                    if idx != marker_idx
                )
                reduced_mol = Chem.MolFromSmiles(
                    smiles_from_direction_marker_slots(
                        skeleton,
                        reduced_marker_slots,
                    )
                )
                if (
                    reduced_mol is not None
                    and canonical_isomeric_smiles(reduced_mol)
                    == reference_canonical_smiles
                ):
                    is_minimal = False
                    break
            if is_minimal:
                minimal_marker_slot_sets.append(marker_slots)
    finally:
        RDLogger.EnableLog("rdApp.*")

    return tuple(sorted(minimal_marker_slot_sets))
