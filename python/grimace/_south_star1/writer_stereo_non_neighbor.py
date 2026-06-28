"""Bounded writer support for directional non-neighbor ligands.

This module installs a small extension over ``writer_stereo`` so the first
non-neighbor directional slice can be developed without reopening the broader
stereo factor implementation.  The supported case is deliberately narrow:
exactly one implicit-hydrogen non-neighbor ligand on an otherwise neighbor-atom
directional site whose center/carrier bonds are acyclic bridge bonds.
"""

from __future__ import annotations

from typing import Literal

from .facts import LigandKind
from .ids import BondId
from .ids import OccurrenceId
from .ids import SiteId
from .residual_constraints import DirectionalBondEmissionFactor
from .residual_constraints import DirectionalNormalizedSign
from .residual_constraints import DirectionalSiteCarrierModel
from .residual_constraints import DirectionalSiteFactor
from .residual_constraints import TetraLocalParity
from .residual_constraints import TetraTokenParityFactor
from .residual_constraints import VarId
from .stereo_templates import DirectionalTemplate
from .writer_stereo import WriterStereoPolicyBlocker
from . import writer_stereo as _writer_stereo

if False:  # pragma: no cover - typing only without runtime import cycles
    from .prepared_runtime import SouthStarPreparedMol


def _directional_non_neighbor_ligand_var(
    site: SiteId,
    occurrence: OccurrenceId,
) -> VarId:
    return VarId(
        "directional_non_neighbor_ligand",
        (int(site), int(occurrence)),
    )


def _directional_non_neighbor_occurrences(
    prepared: "SouthStarPreparedMol",
    template: DirectionalTemplate,
) -> tuple[OccurrenceId, ...]:
    occurrence_by_id = _writer_stereo._occurrence_by_id(prepared)
    return tuple(
        occurrence_id
        for occurrence_id in template.left_ligands + template.right_ligands
        if occurrence_by_id[occurrence_id].kind is not LigandKind.NEIGHBOR_ATOM
    )


def _directional_non_neighbor_ligand_relation_supported_for_template(
    prepared: "SouthStarPreparedMol",
    template: DirectionalTemplate,
) -> bool:
    occurrence_by_id = _writer_stereo._occurrence_by_id(prepared)
    non_neighbor_occurrences = _directional_non_neighbor_occurrences(
        prepared,
        template,
    )
    if len(non_neighbor_occurrences) != 1:
        return False

    non_neighbor = occurrence_by_id[non_neighbor_occurrences[0]]
    if non_neighbor.kind is not LigandKind.IMPLICIT_H:
        return False

    if any(
        occurrence_by_id[occurrence_id].kind
        not in (LigandKind.NEIGHBOR_ATOM, LigandKind.IMPLICIT_H)
        for occurrence_id in template.left_ligands + template.right_ligands
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


def _directional_non_neighbor_ligand_fixed_value(
    occurrence: OccurrenceId,
    *,
    reference: OccurrenceId,
    side_ligands: tuple[OccurrenceId, ...],
) -> DirectionalNormalizedSign:
    ligand_factor = _writer_stereo._ligand_factor(
        occurrence,
        reference=reference,
        side_ligands=side_ligands,
    )
    return (
        DirectionalNormalizedSign.POSITIVE
        if ligand_factor == 1
        else DirectionalNormalizedSign.NEGATIVE
    )


def _directional_non_neighbor_ligand_side_entries(
    prepared: "SouthStarPreparedMol",
    template: DirectionalTemplate,
) -> tuple[tuple[VarId, Literal["left", "right"], DirectionalNormalizedSign], ...]:
    if not _directional_non_neighbor_ligand_relation_supported_for_template(
        prepared,
        template,
    ):
        return ()

    left_reference, right_reference = _writer_stereo._directional_reference_pair(
        template,
    )
    occurrence_by_id = _writer_stereo._occurrence_by_id(prepared)
    entries: list[tuple[VarId, Literal["left", "right"], DirectionalNormalizedSign]] = []

    for occurrence_id in template.left_ligands:
        occurrence = occurrence_by_id[occurrence_id]
        if occurrence.kind is LigandKind.NEIGHBOR_ATOM:
            continue
        entries.append(
            (
                _directional_non_neighbor_ligand_var(
                    template.site,
                    occurrence_id,
                ),
                "left",
                _directional_non_neighbor_ligand_fixed_value(
                    occurrence_id,
                    reference=left_reference,
                    side_ligands=template.left_ligands,
                ),
            )
        )

    for occurrence_id in template.right_ligands:
        occurrence = occurrence_by_id[occurrence_id]
        if occurrence.kind is LigandKind.NEIGHBOR_ATOM:
            continue
        entries.append(
            (
                _directional_non_neighbor_ligand_var(
                    template.site,
                    occurrence_id,
                ),
                "right",
                _directional_non_neighbor_ligand_fixed_value(
                    occurrence_id,
                    reference=right_reference,
                    side_ligands=template.right_ligands,
                ),
            )
        )

    return tuple(sorted(entries, key=lambda item: _writer_stereo._var_sort_tuple(item[0])))


def _writer_stereo_relation_definitions(
    prepared: "SouthStarPreparedMol",
) -> tuple[tuple[tuple[VarId, tuple[object, ...]], ...], tuple[object, ...]]:
    domains: list[tuple[VarId, tuple[object, ...]]] = []
    factors: list[object] = []
    seen_vars: set[VarId] = set()

    def add_var(var: VarId, domain: tuple[object, ...]) -> None:
        if var in seen_vars:
            return
        seen_vars.add(var)
        domains.append((var, domain))

    for template in prepared.tetra_templates:
        token = _writer_stereo.tetra_token_var(template.site)
        parity = _writer_stereo.tetra_parity_var(template.site)
        add_var(token, _writer_stereo._tetra_domain(template))
        add_var(parity, (TetraLocalParity.EVEN, TetraLocalParity.ODD))
        factors.append(
            TetraTokenParityFactor(
                key=_writer_stereo._tetra_factor_key(template.site),
                scope=(token, parity),
                status=template.status,
                target=template.target,
            )
        )

    bond_models: dict[BondId, list[tuple[VarId, DirectionalSiteCarrierModel]]] = {}
    for template in prepared.directional_templates:
        site_models = _writer_stereo._directional_site_carrier_models(
            prepared,
            template,
        )
        non_neighbor_entries = _directional_non_neighbor_ligand_side_entries(
            prepared,
            template,
        )
        scope = tuple(
            var
            for var in (
                *(var for var, _model in site_models),
                *(var for var, _side, _value in non_neighbor_entries),
            )
        )
        scope = tuple(sorted(scope, key=_writer_stereo._var_sort_tuple))
        sides = tuple(
            sorted(
                (
                    *((var, model.side) for var, model in site_models),
                    *((var, side) for var, side, _value in non_neighbor_entries),
                ),
                key=lambda item: _writer_stereo._var_sort_tuple(item[0]),
            )
        )

        for var, _model in site_models:
            add_var(var, _writer_stereo._directional_normalized_domain())
        for var, _side, value in non_neighbor_entries:
            add_var(var, (value,))

        factors.append(
            DirectionalSiteFactor(
                key=_writer_stereo._directional_site_factor_key(template.site),
                scope=scope,
                sides=sides,
                status=template.status,
                target=template.target,
            )
        )
        for var, model in site_models:
            bond_models.setdefault(model.bond, []).append((var, model))

    for bond, entries in bond_models.items():
        factors.append(
            DirectionalBondEmissionFactor(
                key=_writer_stereo._directional_bond_factor_key(bond),
                scope=tuple(var for var, _ in entries),
                models=tuple(model for _, model in entries),
                allowed_marks=_writer_stereo._allowed_direction_marks(
                    prepared,
                    bond,
                ),
            )
        )

    return tuple(domains), tuple(factors)


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
    _writer_stereo._writer_stereo_relation_definitions = (
        _writer_stereo_relation_definitions
    )
    _writer_stereo._unsupported_directional_non_neighbor_ligand_blocker_for_bond = (
        _unsupported_directional_non_neighbor_ligand_blocker_for_bond
    )


__all__ = ("install",)
