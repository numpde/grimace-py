"""Bounded writer support for directional non-neighbor ligands.

Implicit hydrogens are fixed ligand references, not emitted carriers.  This
module admits the bounded ordinary acyclic surface while leaving the existing
carrier model and residual relation owned by ``writer_stereo``.
"""

from __future__ import annotations

from .facts import LigandKind
from .facts import SiteStatus
from .ids import BondId
from .stereo_templates import DirectionalTemplate
from .writer_stereo import WriterStereoPolicyBlocker
from . import writer_stereo as _writer_stereo

if False:  # pragma: no cover - typing only without runtime import cycles
    from .prepared_runtime import SouthStarPreparedMol


def _directional_non_neighbor_ligand_relation_supported_for_template(
    prepared: "SouthStarPreparedMol",
    template: DirectionalTemplate,
) -> bool:
    occurrence_by_id = _writer_stereo._occurrence_by_id(prepared)
    if template.status is not SiteStatus.SPECIFIED or template.reference_pair is None:
        return False

    for ligand_ids in (template.left_ligands, template.right_ligands):
        occurrences = tuple(occurrence_by_id[item] for item in ligand_ids)
        if not 1 <= len(occurrences) <= 2:
            return False
        if sum(item.kind is LigandKind.NEIGHBOR_ATOM for item in occurrences) != 1:
            return False
        if sum(item.kind is LigandKind.IMPLICIT_H for item in occurrences) > 1:
            return False
        if any(
            item.kind not in (LigandKind.NEIGHBOR_ATOM, LigandKind.IMPLICIT_H)
            for item in occurrences
        ):
            return False

    carrier_bonds = _writer_stereo._directional_template_substituent_bonds(
        prepared,
        template,
    )
    if not carrier_bonds:
        return False

    bridge_bonds = prepared.writer_graph_metadata.block_cut.bridge_bonds
    local_bonds = frozenset((*carrier_bonds, template.center_bond))
    return local_bonds.issubset(bridge_bonds)


def _unsupported_directional_non_neighbor_ligand_blocker_for_bond(
    prepared: "SouthStarPreparedMol",
    bond: BondId,
    *,
    operation: str,
) -> WriterStereoPolicyBlocker | None:
    occurrence_by_id = _writer_stereo._occurrence_by_id(prepared)
    for template in sorted(
        prepared.directional_templates,
        key=lambda item: int(item.site),
    ):
        ligand_ids = template.left_ligands + template.right_ligands
        if not any(
            occurrence_by_id[item].kind is not LigandKind.NEIGHBOR_ATOM
            for item in ligand_ids
        ):
            continue
        if bond not in _writer_stereo._directional_template_substituent_bonds(
            prepared,
            template,
        ):
            continue
        if _directional_non_neighbor_ligand_relation_supported_for_template(
            prepared,
            template,
        ):
            continue
        return WriterStereoPolicyBlocker(
            kind="unsupported_directional_non_neighbor_ligand",
            site=template.site,
            operation=operation,
        )

    return None


def install() -> None:
    _writer_stereo._unsupported_directional_non_neighbor_ligand_blocker_for_bond = (
        _unsupported_directional_non_neighbor_ligand_blocker_for_bond
    )


__all__ = ("install",)
