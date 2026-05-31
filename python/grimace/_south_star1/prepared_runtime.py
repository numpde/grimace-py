"""Prepared South Star runtime boundary.

Preparation owns input validation, writer-surface selection, and static
per-molecule structures. Online enumeration and decoding consume the prepared
state plus query-time runtime options.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import MoleculeFacts
from .graph_index import GraphIndex
from .graph_index import build_graph_index
from .ids import AtomId
from .ids import ComponentId
from .writer_graph_obligations import WriterGraphPreparedMetadata
from .writer_graph_obligations import build_writer_graph_prepared_metadata_from_facts
from .online_decoder import online_decode_tokens
from .online_traversal_graph import OnlineTraversalGraph
from .online_traversal_graph import build_online_traversal_graph_from_index
from .ordinary_policy import ordinary_policy_for_facts
from .ordinary_semantics import OrdinarySmilesSemantics
from .policy import SerializationLanguageMode
from .policy import SmilesPolicy
from .root_domains import component_root_domains_for_facts
from .root_domains import component_root_domains_from_metadata
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
    serialization_language: SerializationLanguageMode = (
        SerializationLanguageMode.EXHAUSTIVE
    )


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
    atom_ids: tuple[AtomId, ...]
    component_ids: tuple[ComponentId, ...]
    component_atom_ids: tuple[tuple[AtomId, ...], ...]
    atom_component: tuple[tuple[AtomId, ComponentId], ...]
    graph_index: GraphIndex
    writer_graph_metadata: WriterGraphPreparedMetadata
    online_traversal_graph: OnlineTraversalGraph
    all_root_domains: tuple[tuple[ComponentId, tuple[AtomId, ...]], ...]
    atom_component_map: Mapping[AtomId, ComponentId]
    component_root_domains_by_explicit_root: Mapping[
        AtomId,
        tuple[tuple[ComponentId, tuple[AtomId, ...]], ...],
    ]

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
    graph_index = build_graph_index(facts)
    writer_graph_metadata = build_writer_graph_prepared_metadata_from_facts(
        facts,
        graph_index,
        tuple(atom.id for atom in facts.atoms),
    )
    online_traversal_graph = build_online_traversal_graph_from_index(
        facts,
        graph_index,
    )
    atom_component = tuple(
        (atom, component.id)
        for component in facts.components
        for atom in component.atoms
    )
    atom_component_map = MappingProxyType(dict(atom_component))
    all_root_domains = component_root_domains_for_facts(facts, None)
    explicit_root_domains = MappingProxyType(
        {
            atom_id: component_root_domains_from_metadata(
                all_root_domains=all_root_domains,
                atom_component_map=atom_component_map,
                rooted_at_atom=atom_id,
            )
            for atom_id in (atom.id for atom in facts.atoms)
        }
    )
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
        atom_ids=tuple(atom.id for atom in facts.atoms),
        component_ids=tuple(component.id for component in facts.components),
        component_atom_ids=tuple(component.atoms for component in facts.components),
        atom_component=atom_component,
        graph_index=graph_index,
        writer_graph_metadata=writer_graph_metadata,
        online_traversal_graph=online_traversal_graph,
        all_root_domains=all_root_domains,
        atom_component_map=atom_component_map,
        component_root_domains_by_explicit_root=explicit_root_domains,
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


def enumerate_prepared_stereo_support(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions(),
):
    validate_south_star_runtime_options_common(runtime_options)
    match runtime_options.serialization_language:
        case SerializationLanguageMode.EXHAUSTIVE:
            return enumerate_prepared_exhaustive_stereo_support(
                prepared=prepared,
                runtime_options=runtime_options,
            )
        case SerializationLanguageMode.WRITER_SHAPED:
            return enumerate_prepared_writer_shaped_support(
                prepared=prepared,
                runtime_options=runtime_options,
            )
        case SerializationLanguageMode.RDKIT_PARITY:
            raise SouthStarError(
                SouthStarErrorKind.UNSUPPORTED_POLICY,
                "RDKIT_PARITY writer-state runtime is not wired yet",
            )
        case _:
            raise ValueError(
                "unsupported South Star serialization language: "
                f"{runtime_options.serialization_language!r}"
            )


def enumerate_prepared_exhaustive_stereo_support(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions(),
):
    from .skeleton import enumerate_traversal_skeletons
    from .support_enumeration import enumerate_exhaustive_stereo_support

    require_exhaustive_runtime_options(runtime_options)

    rooted_at_atom = runtime_root_atom_for_prepared(
        runtime_options,
        prepared=prepared,
    )
    root_domains = component_root_domains_for_prepared(
        prepared=prepared,
        rooted_at_atom=rooted_at_atom,
    )
    skeletons = enumerate_traversal_skeletons(
        facts=prepared.facts,
        index=prepared.graph_index,
        policy=prepared.policy,
        rooted_at_atom=rooted_at_atom,
        component_root_domains=tuple(atoms for _, atoms in root_domains),
        validate_inputs=False,
    )
    return enumerate_exhaustive_stereo_support(
        facts=prepared.facts,
        policy=prepared.policy,
        semantics=prepared.semantics,
        skeletons=skeletons,
        validate_inputs=False,
    )


def enumerate_prepared_writer_shaped_support(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
):
    from .writer_support import (
        enumerate_prepared_writer_shaped_support as enumerate_writer_support,
    )

    return enumerate_writer_support(
        prepared=prepared,
        runtime_options=runtime_options,
    )

def validate_south_star_runtime_options_common(
    options: SouthStarRuntimeOptions,
    *,
    facts: MoleculeFacts | None = None,
) -> None:
    if not isinstance(options.serialization_language, SerializationLanguageMode):
        raise ValueError(
            "serialization_language must be a SerializationLanguageMode"
        )
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
    if facts is not None and options.rooted_at_atom >= 0:
        atom = AtomId(options.rooted_at_atom)
        if atom not in {item.id for item in facts.atoms}:
            raise SouthStarError(
                SouthStarErrorKind.INVALID_FACTS,
                f"rooted_at_atom is not present in molecule facts: {options.rooted_at_atom}",
            )


def require_exhaustive_runtime_options(
    options: SouthStarRuntimeOptions,
    *,
    facts: MoleculeFacts | None = None,
) -> None:
    validate_south_star_runtime_options_common(options, facts=facts)
    if options.serialization_language is not SerializationLanguageMode.EXHAUSTIVE:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "South Star exhaustive runtime requires "
            "serialization_language=EXHAUSTIVE",
        )


def require_writer_shaped_runtime_options(
    options: SouthStarRuntimeOptions,
    *,
    facts: MoleculeFacts | None = None,
) -> None:
    validate_south_star_runtime_options_common(options, facts=facts)
    if options.serialization_language is not SerializationLanguageMode.WRITER_SHAPED:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "South Star writer-shaped runtime requires "
            "serialization_language=WRITER_SHAPED",
        )


def runtime_root_atom(
    options: SouthStarRuntimeOptions,
    *,
    facts: MoleculeFacts,
) -> AtomId | None:
    validate_south_star_runtime_options_common(options, facts=facts)
    if options.rooted_at_atom < 0:
        return None
    return AtomId(options.rooted_at_atom)


def runtime_root_atom_for_prepared(
    options: SouthStarRuntimeOptions,
    *,
    prepared: SouthStarPreparedMol,
) -> AtomId | None:
    validate_south_star_runtime_options_common(options)
    if options.rooted_at_atom < 0:
        return None
    atom = AtomId(options.rooted_at_atom)
    if atom not in prepared.atom_component_map:
        raise SouthStarError(
            SouthStarErrorKind.INVALID_FACTS,
            f"rooted_at_atom is not present in prepared molecule: {options.rooted_at_atom}",
        )
    return atom


def component_root_domains_for_prepared(
    *,
    prepared: SouthStarPreparedMol,
    rooted_at_atom: AtomId | None,
) -> tuple[tuple[ComponentId, tuple[AtomId, ...]], ...]:
    if rooted_at_atom is None:
        return prepared.all_root_domains
    try:
        return prepared.component_root_domains_by_explicit_root[rooted_at_atom]
    except KeyError as exc:
        raise ValueError(
            f"rooted atom is not present in prepared molecule: {rooted_at_atom!r}"
        ) from exc


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
    "OnlineTraversalGraph",
    "component_root_domains_for_facts",
    "component_root_domains_for_prepared",
    "enumerate_prepared_exhaustive_stereo_support",
    "enumerate_prepared_stereo_support",
    "enumerate_prepared_writer_shaped_support",
    "prepare_south_star_mol_from_facts",
    "prepare_south_star_mol_from_rdkit",
    "require_exhaustive_runtime_options",
    "require_writer_shaped_runtime_options",
    "runtime_root_atom",
    "runtime_root_atom_for_prepared",
    "validate_south_star_runtime_options_common",
)
