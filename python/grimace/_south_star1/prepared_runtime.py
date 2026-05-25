"""Prepared South Star runtime boundary.

Preparation owns input validation, writer-surface selection, and static
per-molecule structures. Online enumeration and decoding consume the prepared
state plus query-time runtime options.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import MoleculeFacts
from .online_decoder import online_decode_tokens
from .ordinary_policy import ordinary_policy_for_facts
from .ordinary_semantics import OrdinarySmilesSemantics
from .policy import SmilesPolicy
from .semantics import ParserSemantics
from .stereo_templates import DirectionalTemplate
from .stereo_templates import StereoTemplateBundle
from .stereo_templates import TetraTemplate
from .stereo_templates import build_stereo_templates


@dataclass(frozen=True, slots=True)
class SouthStarWriterSurface:
    isomeric_smiles: bool = True
    kekule_smiles: bool = False
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    ignore_atom_map_numbers: bool = False


@dataclass(frozen=True, slots=True)
class SouthStarRuntimeOptions:
    rooted_at_atom: int = -1
    canonical: bool = False
    do_random: bool = True


@dataclass(frozen=True, slots=True)
class SouthStarPreparedMol:
    facts: MoleculeFacts
    policy: SmilesPolicy
    semantics: ParserSemantics
    writer_surface: SouthStarWriterSurface
    tetra_templates: tuple[TetraTemplate, ...]
    directional_templates: tuple[DirectionalTemplate, ...]
    token_inventory_superset: tuple[str, ...]
    atom_count: int
    component_count: int

    def stereo_template_bundle(self) -> StereoTemplateBundle:
        return StereoTemplateBundle(
            tetrahedral=self.tetra_templates,
            directional=self.directional_templates,
        )


def prepare_south_star_mol_from_facts(
    facts: MoleculeFacts,
    *,
    writer_surface: SouthStarWriterSurface,
    policy: SmilesPolicy | None = None,
    semantics: ParserSemantics | None = None,
) -> SouthStarPreparedMol:
    facts.validate()
    _validate_writer_surface(facts, writer_surface)
    resolved_policy = policy if policy is not None else ordinary_policy_for_facts(facts)
    resolved_policy.validate_for_facts(facts)
    resolved_semantics = semantics if semantics is not None else OrdinarySmilesSemantics()
    templates = build_stereo_templates(facts)
    return SouthStarPreparedMol(
        facts=facts,
        policy=resolved_policy,
        semantics=resolved_semantics,
        writer_surface=writer_surface,
        tetra_templates=templates.tetrahedral,
        directional_templates=templates.directional,
        token_inventory_superset=_token_inventory_superset(resolved_policy),
        atom_count=len(facts.atoms),
        component_count=len(facts.components),
    )


def prepare_south_star_mol_from_rdkit(
    mol: object,
    *,
    writer_surface: SouthStarWriterSurface,
    extraction_options: object | None = None,
    policy: SmilesPolicy | None = None,
    semantics: ParserSemantics | None = None,
) -> SouthStarPreparedMol:
    from .rdkit_adapter import RdkitOrdinaryExtractionOptions
    from .rdkit_adapter import ordinary_molecule_facts_from_rdkit

    options = (
        extraction_options
        if extraction_options is not None
        else RdkitOrdinaryExtractionOptions()
    )
    if not isinstance(options, RdkitOrdinaryExtractionOptions):
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            f"unsupported South Star RDKit extraction options: {options!r}",
        )
    return prepare_south_star_mol_from_facts(
        ordinary_molecule_facts_from_rdkit(mol, options),
        writer_surface=writer_surface,
        policy=policy,
        semantics=semantics,
    )


def validate_south_star_runtime_options(options: SouthStarRuntimeOptions) -> None:
    if options.canonical:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "South Star online runtime currently supports canonical=False",
        )
    if not options.do_random:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "South Star online runtime currently supports do_random=True",
        )
    if options.rooted_at_atom != -1:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "South Star online runtime does not yet implement rooted traversal queries",
        )


def _validate_writer_surface(
    facts: MoleculeFacts,
    writer_surface: SouthStarWriterSurface,
) -> None:
    unsupported: list[str] = []
    if writer_surface.kekule_smiles:
        unsupported.append("kekule_smiles=True")
    if writer_surface.all_bonds_explicit:
        unsupported.append("all_bonds_explicit=True")
    if writer_surface.all_hs_explicit:
        unsupported.append("all_hs_explicit=True")
    if unsupported:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "unsupported South Star writer surface flags: " + ", ".join(unsupported),
        )
    if not writer_surface.isomeric_smiles and _has_stereo_facts(facts):
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "isomeric_smiles=False conflicts with prepared South Star stereo facts",
        )


def _has_stereo_facts(facts: MoleculeFacts) -> bool:
    return bool(facts.stereo.tetrahedral or facts.stereo.directional)


def _token_inventory_superset(policy: SmilesPolicy) -> tuple[str, ...]:
    return tuple(sorted({token.text for token in online_decode_tokens(policy)}))


__all__ = (
    "SouthStarPreparedMol",
    "SouthStarRuntimeOptions",
    "SouthStarWriterSurface",
    "prepare_south_star_mol_from_facts",
    "prepare_south_star_mol_from_rdkit",
    "validate_south_star_runtime_options",
)
