"""Specified-stereo closure for ordinary stereo-site discovery."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import AtomFacts
from .facts import BondFacts
from .facts import DirectionalSiteFacts
from .facts import DirectionalValue
from .facts import LigandKind
from .facts import LigandOccurrence
from .facts import MoleculeFacts
from .facts import SiteStatus
from .facts import StereoFacts
from .facts import TetraValue
from .facts import TetrahedralSiteFacts
from .ids import AtomId
from .ids import BondId
from .ids import OccurrenceId
from .ids import SiteId
from .ordinary_stereo_sites import OrdinaryStereoSiteOptions
from .ordinary_stereo_sites import add_ordinary_potential_sites


@dataclass(frozen=True, slots=True)
class RawTetraStereoRecord:
    center: AtomId
    target: TetraValue
    reference_atoms: tuple[AtomId | None, ...]


@dataclass(frozen=True, slots=True)
class RawDirectionalStereoRecord:
    center_bond: BondId
    target: DirectionalValue
    reference_atoms: tuple[AtomId, AtomId]


@dataclass(frozen=True, slots=True)
class RawSpecifiedStereo:
    tetrahedral: tuple[RawTetraStereoRecord, ...] = ()
    directional: tuple[RawDirectionalStereoRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class StereoDiscoveryResult:
    facts: MoleculeFacts
    promoted_tetrahedral: int
    promoted_directional: int
    potential_tetrahedral: int
    potential_directional: int


def build_ordinary_stereo_specified_closure(
    base_facts: MoleculeFacts,
    *,
    raw_specified: RawSpecifiedStereo,
    site_options: OrdinaryStereoSiteOptions,
) -> StereoDiscoveryResult:
    """Build ordinary sites using raw specified stereo as the closure context.

    Every raw specified record must be eligible with all raw specified labels
    available except its own site id. The final ordinary potential-site universe
    is then rebuilt under the accepted specified context.
    """

    context = _materialize_raw_specified_stereo(base_facts, raw_specified)
    probe = add_ordinary_potential_sites(
        context,
        preserve_specified=False,
        options=site_options,
    )
    _require_raw_records_eligible(context, probe, raw_specified)

    out = add_ordinary_potential_sites(
        context,
        preserve_specified=True,
        options=site_options,
    )
    return StereoDiscoveryResult(
        facts=out,
        promoted_tetrahedral=len(raw_specified.tetrahedral),
        promoted_directional=len(raw_specified.directional),
        potential_tetrahedral=len(out.stereo.tetrahedral),
        potential_directional=len(out.stereo.directional),
    )


def _materialize_raw_specified_stereo(
    facts: MoleculeFacts,
    raw: RawSpecifiedStereo,
) -> MoleculeFacts:
    facts.validate()
    _validate_raw_records(raw)

    tetrahedral = list(facts.stereo.tetrahedral)
    directional = list(facts.stereo.directional)
    occurrences = list(facts.ligand_occurrences)
    next_site = _next_site_id(facts)
    next_occurrence = _next_occurrence_id(facts)

    for record in raw.tetrahedral:
        site_id = SiteId(next_site)
        next_site += 1
        site, new_occurrences = _materialize_raw_tetra_record(
            facts,
            record,
            site_id=site_id,
            first_occurrence=next_occurrence,
        )
        next_occurrence += len(new_occurrences)
        tetrahedral.append(site)
        occurrences.extend(new_occurrences)

    for record in raw.directional:
        site_id = SiteId(next_site)
        next_site += 1
        site, new_occurrences = _materialize_raw_directional_record(
            facts,
            record,
            site_id=site_id,
            first_occurrence=next_occurrence,
        )
        next_occurrence += len(new_occurrences)
        directional.append(site)
        occurrences.extend(new_occurrences)

    out = replace(
        facts,
        stereo=StereoFacts(
            tetrahedral=tuple(tetrahedral),
            directional=tuple(directional),
        ),
        ligand_occurrences=tuple(occurrences),
    )
    out.validate()
    return out


def _validate_raw_records(raw: RawSpecifiedStereo) -> None:
    tetra_centers = tuple(record.center for record in raw.tetrahedral)
    if len(set(tetra_centers)) != len(tetra_centers):
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "duplicate raw tetrahedral stereo centers",
        )
    directional_bonds = tuple(record.center_bond for record in raw.directional)
    if len(set(directional_bonds)) != len(directional_bonds):
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "duplicate raw directional stereo center bonds",
        )


def _materialize_raw_tetra_record(
    facts: MoleculeFacts,
    record: RawTetraStereoRecord,
    *,
    site_id: SiteId,
    first_occurrence: int,
) -> tuple[TetrahedralSiteFacts, tuple[LigandOccurrence, ...]]:
    if record.target is TetraValue.NONE:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            f"raw tetrahedral record has no target: {record.center!r}",
        )
    atom = _atom_by_id(facts).get(record.center)
    if atom is None:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            f"raw tetrahedral record has unknown center: {record.center!r}",
        )

    raw_occurrences = _atom_ligands(facts, atom)
    occurrence_ids = tuple(
        OccurrenceId(first_occurrence + index)
        for index in range(len(raw_occurrences))
    )
    occurrences = tuple(
        replace(
            occurrence,
            id=occurrence_id,
            site=site_id,
            ordinal=index,
        )
        for index, (occurrence, occurrence_id) in enumerate(
            zip(raw_occurrences, occurrence_ids, strict=True)
        )
    )
    reference_order = _tetra_reference_order_from_atoms(
        occurrences,
        record.reference_atoms,
    )
    return (
        TetrahedralSiteFacts(
            id=site_id,
            center=record.center,
            status=SiteStatus.SPECIFIED,
            target=record.target,
            ligand_occurrences=occurrence_ids,
            reference_order=reference_order,
        ),
        occurrences,
    )


def _materialize_raw_directional_record(
    facts: MoleculeFacts,
    record: RawDirectionalStereoRecord,
    *,
    site_id: SiteId,
    first_occurrence: int,
) -> tuple[DirectionalSiteFacts, tuple[LigandOccurrence, ...]]:
    if record.target is DirectionalValue.NONE:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            f"raw directional record has no target: {record.center_bond!r}",
        )
    bond = _bond_by_id(facts).get(record.center_bond)
    if bond is None:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            f"raw directional record has unknown center bond: {record.center_bond!r}",
        )

    left_raw = _endpoint_ligands(facts, endpoint=bond.a, center_bond=bond.id)
    right_raw = _endpoint_ligands(facts, endpoint=bond.b, center_bond=bond.id)
    raw_occurrences = left_raw + right_raw
    occurrence_ids = tuple(
        OccurrenceId(first_occurrence + index)
        for index in range(len(raw_occurrences))
    )
    occurrences = tuple(
        replace(
            occurrence,
            id=occurrence_id,
            site=site_id,
            ordinal=index,
        )
        for index, (occurrence, occurrence_id) in enumerate(
            zip(raw_occurrences, occurrence_ids, strict=True)
        )
    )
    left_ids = occurrence_ids[: len(left_raw)]
    right_ids = occurrence_ids[len(left_raw) :]
    reference_pair = _directional_reference_pair_from_atoms(
        occurrences,
        left_ids=left_ids,
        right_ids=right_ids,
        reference_atoms=record.reference_atoms,
    )
    return (
        DirectionalSiteFacts(
            id=site_id,
            center_bond=record.center_bond,
            left_endpoint=bond.a,
            right_endpoint=bond.b,
            status=SiteStatus.SPECIFIED,
            target=record.target,
            left_ligands=left_ids,
            right_ligands=right_ids,
            reference_pair=reference_pair,
        ),
        occurrences,
    )


def _require_raw_records_eligible(
    context: MoleculeFacts,
    probe: MoleculeFacts,
    raw: RawSpecifiedStereo,
) -> None:
    context_tetra_count = len(context.stereo.tetrahedral)
    context_directional_count = len(context.stereo.directional)
    eligible_tetra_centers = {
        site.center
        for site in probe.stereo.tetrahedral[context_tetra_count:]
    }
    eligible_directional_bonds = {
        site.center_bond
        for site in probe.stereo.directional[context_directional_count:]
    }

    for record in raw.tetrahedral:
        if record.center not in eligible_tetra_centers:
            raise SouthStarError(
                SouthStarErrorKind.UNSUPPORTED_STEREO,
                "raw tetrahedral stereo has no ordinary potential site under "
                f"specified closure: {record.center!r}",
            )
    for record in raw.directional:
        if record.center_bond not in eligible_directional_bonds:
            raise SouthStarError(
                SouthStarErrorKind.UNSUPPORTED_STEREO,
                "raw directional stereo has no ordinary potential site under "
                f"specified closure: {record.center_bond!r}",
            )


def _tetra_reference_order_from_atoms(
    occurrences: tuple[LigandOccurrence, ...],
    reference_atoms: tuple[AtomId | None, ...],
) -> tuple[OccurrenceId, ...]:
    if len(reference_atoms) != len(occurrences):
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "raw tetrahedral reference order does not match ligand count",
        )
    by_atom = _neighbor_occurrences_by_atom(occurrences)
    implicit_h = tuple(
        occurrence.id
        for occurrence in occurrences
        if occurrence.kind is LigandKind.IMPLICIT_H
    )
    out: list[OccurrenceId] = []
    used: set[OccurrenceId] = set()
    for reference in reference_atoms:
        if reference is None:
            if len(implicit_h) != 1:
                raise SouthStarError(
                    SouthStarErrorKind.UNSUPPORTED_STEREO,
                    "raw tetrahedral implicit-H reference is ambiguous",
                )
            occurrence_id = implicit_h[0]
        else:
            occurrence_id = by_atom.get(reference)
            if occurrence_id is None:
                raise SouthStarError(
                    SouthStarErrorKind.UNSUPPORTED_STEREO,
                    "raw tetrahedral reference atom is not a ligand: "
                    f"{reference!r}",
                )
        if occurrence_id in used:
            raise SouthStarError(
                SouthStarErrorKind.UNSUPPORTED_STEREO,
                "raw tetrahedral reference order repeats a ligand",
            )
        used.add(occurrence_id)
        out.append(occurrence_id)
    return tuple(out)


def _directional_reference_pair_from_atoms(
    occurrences: tuple[LigandOccurrence, ...],
    *,
    left_ids: tuple[OccurrenceId, ...],
    right_ids: tuple[OccurrenceId, ...],
    reference_atoms: tuple[AtomId, AtomId],
) -> tuple[OccurrenceId, OccurrenceId]:
    if len(reference_atoms) != 2:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "raw directional reference must contain two atoms",
        )
    by_atom = _neighbor_occurrences_by_atom(occurrences)
    first = by_atom.get(reference_atoms[0])
    second = by_atom.get(reference_atoms[1])
    if first is None or second is None:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "raw directional reference atom is not a ligand",
        )
    if first in left_ids and second in right_ids:
        return (first, second)
    if second in left_ids and first in right_ids:
        return (second, first)
    raise SouthStarError(
        SouthStarErrorKind.UNSUPPORTED_STEREO,
        "raw directional references do not span the center bond",
    )


def _atom_ligands(facts: MoleculeFacts, atom: AtomFacts) -> tuple[LigandOccurrence, ...]:
    occurrences: list[LigandOccurrence] = []
    for bond in sorted(
        _bonds_by_atom(facts).get(atom.id, ()),
        key=lambda candidate: (int(_other_atom(candidate, atom.id)), int(candidate.id)),
    ):
        occurrences.append(
            LigandOccurrence(
                id=OccurrenceId(-1),
                site=SiteId(-1),
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=_other_atom(bond, atom.id),
                bond=bond.id,
            )
        )
    for _ in range(atom.implicit_h_count):
        occurrences.append(
            LigandOccurrence(
                id=OccurrenceId(-1),
                site=SiteId(-1),
                kind=LigandKind.IMPLICIT_H,
                atom=atom.id,
                bond=None,
            )
        )
    return tuple(occurrences)


def _endpoint_ligands(
    facts: MoleculeFacts,
    *,
    endpoint: AtomId,
    center_bond: BondId,
) -> tuple[LigandOccurrence, ...]:
    occurrences: list[LigandOccurrence] = []
    for bond in sorted(
        (
            bond
            for bond in _bonds_by_atom(facts).get(endpoint, ())
            if bond.id != center_bond
        ),
        key=lambda candidate: (int(_other_atom(candidate, endpoint)), int(candidate.id)),
    ):
        occurrences.append(
            LigandOccurrence(
                id=OccurrenceId(-1),
                site=SiteId(-1),
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=_other_atom(bond, endpoint),
                bond=bond.id,
            )
        )

    atom = _atom_by_id(facts)[endpoint]
    for _ in range(atom.implicit_h_count):
        occurrences.append(
            LigandOccurrence(
                id=OccurrenceId(-1),
                site=SiteId(-1),
                kind=LigandKind.IMPLICIT_H,
                atom=endpoint,
                bond=None,
            )
        )
    return tuple(occurrences)


def _neighbor_occurrences_by_atom(
    occurrences: tuple[LigandOccurrence, ...],
) -> dict[AtomId, OccurrenceId]:
    out: dict[AtomId, OccurrenceId] = {}
    for occurrence in occurrences:
        if occurrence.kind is not LigandKind.NEIGHBOR_ATOM:
            continue
        if occurrence.atom is None:
            raise ValueError(f"neighbor occurrence lacks atom: {occurrence.id!r}")
        out[occurrence.atom] = occurrence.id
    return out


def _bonds_by_atom(facts: MoleculeFacts) -> dict[AtomId, tuple[BondFacts, ...]]:
    out: dict[AtomId, list[BondFacts]] = {atom.id: [] for atom in facts.atoms}
    for bond in facts.bonds:
        out[bond.a].append(bond)
        out[bond.b].append(bond)
    return {atom: tuple(bonds) for atom, bonds in out.items()}


def _other_atom(bond: BondFacts, atom: AtomId) -> AtomId:
    if bond.a == atom:
        return bond.b
    if bond.b == atom:
        return bond.a
    raise ValueError(f"bond {bond.id!r} is not incident to atom {atom!r}")


def _atom_by_id(facts: MoleculeFacts) -> dict[AtomId, AtomFacts]:
    return {atom.id: atom for atom in facts.atoms}


def _bond_by_id(facts: MoleculeFacts) -> dict[BondId, BondFacts]:
    return {bond.id: bond for bond in facts.bonds}


def _next_site_id(facts: MoleculeFacts) -> int:
    ids = [int(site.id) for site in facts.stereo.tetrahedral]
    ids.extend(int(site.id) for site in facts.stereo.directional)
    return max(ids, default=-1) + 1


def _next_occurrence_id(facts: MoleculeFacts) -> int:
    return (
        max(
            (int(occurrence.id) for occurrence in facts.ligand_occurrences),
            default=-1,
        )
        + 1
    )


__all__ = (
    "RawDirectionalStereoRecord",
    "RawSpecifiedStereo",
    "RawTetraStereoRecord",
    "StereoDiscoveryResult",
    "build_ordinary_stereo_specified_closure",
)
