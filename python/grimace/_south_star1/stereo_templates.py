"""Static stereo templates for online residual constraints."""

from __future__ import annotations

from dataclasses import dataclass

from .facts import DirectionalValue
from .facts import MoleculeFacts
from .facts import SiteStatus
from .facts import TetraValue
from .ids import AtomId
from .ids import BondId
from .ids import OccurrenceId
from .ids import SiteId


@dataclass(frozen=True, slots=True)
class TetraTemplate:
    site: SiteId
    center: AtomId
    status: SiteStatus
    target: TetraValue
    reference_order: tuple[OccurrenceId, ...]
    ligand_occurrences: tuple[OccurrenceId, ...]


@dataclass(frozen=True, slots=True)
class DirectionalTemplate:
    site: SiteId
    center_bond: BondId
    left_endpoint: AtomId
    right_endpoint: AtomId
    status: SiteStatus
    target: DirectionalValue
    left_ligands: tuple[OccurrenceId, ...]
    right_ligands: tuple[OccurrenceId, ...]
    reference_pair: tuple[OccurrenceId, OccurrenceId] | None


@dataclass(frozen=True, slots=True)
class StereoTemplateBundle:
    tetrahedral: tuple[TetraTemplate, ...]
    directional: tuple[DirectionalTemplate, ...]


def build_stereo_templates(facts: MoleculeFacts) -> StereoTemplateBundle:
    """Extract static stereo templates from validated molecule facts."""

    facts.validate()
    return StereoTemplateBundle(
        tetrahedral=tuple(
            TetraTemplate(
                site=site.id,
                center=site.center,
                status=site.status,
                target=site.target,
                reference_order=site.reference_order,
                ligand_occurrences=site.ligand_occurrences,
            )
            for site in facts.stereo.tetrahedral
        ),
        directional=tuple(
            DirectionalTemplate(
                site=site.id,
                center_bond=site.center_bond,
                left_endpoint=site.left_endpoint,
                right_endpoint=site.right_endpoint,
                status=site.status,
                target=site.target,
                left_ligands=site.left_ligands,
                right_ligands=site.right_ligands,
                reference_pair=site.reference_pair,
            )
            for site in facts.stereo.directional
        ),
    )


__all__ = (
    "DirectionalTemplate",
    "StereoTemplateBundle",
    "TetraTemplate",
    "build_stereo_templates",
)
