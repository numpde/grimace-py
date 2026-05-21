from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.enum_s import SouthStarEnumSGenerationDiagnostics
from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_domain_manifest import SOUTH_STAR_PRIVATE_DOMAIN
from tests.helpers.south_star_semantic_oracle import parse_smiles


@dataclass(frozen=True, slots=True)
class SouthStarAdversarialVariant:
    variant_id: str
    source_smiles: str
    axes: tuple[str, ...]
    mutation_path: tuple[str, ...]
    boundary_targets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SouthStarAdversarialSeed:
    seed_id: str
    variants: tuple[SouthStarAdversarialVariant, ...]


@dataclass(frozen=True, slots=True)
class SouthStarAdversarialCandidate:
    candidate_id: str
    seed_id: str
    source_smiles: str
    axes: tuple[str, ...]
    mutation_path: tuple[str, ...]
    boundary_targets: tuple[str, ...]
    shrink_key: tuple[int, int, int, str]


@dataclass(frozen=True, slots=True)
class SouthStarAdversarialTriage:
    candidate: SouthStarAdversarialCandidate
    supported_by_gate: bool
    unsupported_categories: tuple[str, ...]
    generated_output_count: int | None
    generation_diagnostics: SouthStarEnumSGenerationDiagnostics | None


SOUTH_STAR_ADVERSARIAL_AXES: frozenset[str] = frozenset(
    {
        "root_choice",
        "branch_main_order",
        "carrier_placement",
        "shared_carrier",
        "ring_closure_choice",
        "tetrahedral_ligand_order",
        "disconnected_fragment_order",
        "atom_modifier",
        "unsupported_feature_trigger",
    }
)
SOUTH_STAR_ADVERSARIAL_EXTRA_BOUNDARY_TARGETS: frozenset[str] = frozenset(
    {
        "first_domain_directional_stereo",
    }
)
SOUTH_STAR_ADVERSARIAL_BOUNDARY_TARGETS: frozenset[str] = frozenset(
    {
        *SOUTH_STAR_PRIVATE_DOMAIN.expanded_feature_areas,
        *SOUTH_STAR_PRIVATE_DOMAIN.support_gate_blocker_categories,
        *SOUTH_STAR_PRIVATE_DOMAIN.annotation_policies,
        *SOUTH_STAR_PRIVATE_DOMAIN.fragment_order_policies,
        *SOUTH_STAR_PRIVATE_DOMAIN.output_order_policies,
        *SOUTH_STAR_ADVERSARIAL_EXTRA_BOUNDARY_TARGETS,
    }
)
SOUTH_STAR_ADVERSARIAL_REQUIRED_SUPPORTED_FEATURE_TARGETS: frozenset[str] = (
    frozenset(
        {
            "charged_atom_text",
            "disconnected_stereo_fragments",
            "explicit_bracket_hydrogen",
            "radical_atom_text",
            "ring_stereo_monocycle",
            "tetrahedral_atom_stereo",
        }
    )
)
SOUTH_STAR_ADVERSARIAL_REQUIRED_UNSUPPORTED_CATEGORY_TARGETS: frozenset[str] = (
    frozenset(
        {
            "aromatic_ring_surface",
            "dative_bond",
            "unsupported_bond_type",
        }
    )
)


def generate_south_star_adversarial_candidates() -> tuple[
    SouthStarAdversarialCandidate,
    ...
]:
    """Generate deterministic triage candidates, not expected support sets."""

    candidates = tuple(
        _candidate_from_variant(seed, variant)
        for seed in _ADVERSARIAL_SEEDS
        for variant in seed.variants
    )
    return tuple(sorted(candidates, key=lambda candidate: candidate.shrink_key))


def south_star_adversarial_triage(
    candidate: SouthStarAdversarialCandidate,
) -> SouthStarAdversarialTriage:
    mol = parse_smiles(candidate.source_smiles)
    gate_report = south_star_support_gate_report(mol)
    unsupported_categories = tuple(sorted(gate_report.categories))
    if not gate_report.supported:
        return SouthStarAdversarialTriage(
            candidate=candidate,
            supported_by_gate=False,
            unsupported_categories=unsupported_categories,
            generated_output_count=None,
            generation_diagnostics=None,
        )

    result = mol_to_smiles_enum_s_graph_native(
        candidate.source_smiles,
        case_id=candidate.candidate_id,
    )
    if result.generation_diagnostics is None:
        raise AssertionError("adversarial triage requires generation diagnostics")
    return SouthStarAdversarialTriage(
        candidate=candidate,
        supported_by_gate=True,
        unsupported_categories=(),
        generated_output_count=len(result.outputs),
        generation_diagnostics=result.generation_diagnostics,
    )


def _candidate_from_variant(
    seed: SouthStarAdversarialSeed,
    variant: SouthStarAdversarialVariant,
) -> SouthStarAdversarialCandidate:
    if not variant.axes:
        raise ValueError(f"adversarial variant {variant.variant_id!r} needs axes")
    unknown_axes = tuple(
        axis for axis in variant.axes if axis not in SOUTH_STAR_ADVERSARIAL_AXES
    )
    if unknown_axes:
        raise ValueError(
            f"adversarial variant {variant.variant_id!r} uses unknown axes "
            f"{unknown_axes!r}"
        )
    if not variant.boundary_targets:
        raise ValueError(
            f"adversarial variant {variant.variant_id!r} needs boundary targets"
        )
    unknown_targets = tuple(
        target
        for target in variant.boundary_targets
        if target not in SOUTH_STAR_ADVERSARIAL_BOUNDARY_TARGETS
    )
    if unknown_targets:
        raise ValueError(
            f"adversarial variant {variant.variant_id!r} uses unknown boundary "
            f"targets {unknown_targets!r}"
        )
    candidate_id = f"{seed.seed_id}:{variant.variant_id}"
    return SouthStarAdversarialCandidate(
        candidate_id=candidate_id,
        seed_id=seed.seed_id,
        source_smiles=variant.source_smiles,
        axes=variant.axes,
        mutation_path=variant.mutation_path,
        boundary_targets=variant.boundary_targets,
        shrink_key=_shrink_key(candidate_id, variant),
    )


def _shrink_key(
    candidate_id: str,
    variant: SouthStarAdversarialVariant,
) -> tuple[int, int, int, str]:
    mol = Chem.MolFromSmiles(variant.source_smiles)
    atom_count = mol.GetNumAtoms() if mol is not None else 10**9
    return (
        atom_count,
        len(variant.mutation_path),
        len(variant.source_smiles),
        candidate_id,
    )


_ADVERSARIAL_SEEDS: tuple[SouthStarAdversarialSeed, ...] = (
    SouthStarAdversarialSeed(
        seed_id="acyclic_directional_root_branch",
        variants=(
            SouthStarAdversarialVariant(
                variant_id="canonical_root",
                source_smiles="F/C=C\\Cl",
                axes=("root_choice", "carrier_placement"),
                mutation_path=("seed",),
                boundary_targets=(
                    "first_domain_directional_stereo",
                    "maximal_eligible_carrier",
                ),
            ),
            SouthStarAdversarialVariant(
                variant_id="branch_first_carrier",
                source_smiles="C(\\F)=C\\Cl",
                axes=("branch_main_order", "carrier_placement"),
                mutation_path=("seed", "branch_first"),
                boundary_targets=(
                    "first_domain_directional_stereo",
                    "maximal_eligible_carrier",
                ),
            ),
            SouthStarAdversarialVariant(
                variant_id="opposite_halogen_root",
                source_smiles="Cl\\C=C/F",
                axes=("root_choice", "carrier_placement"),
                mutation_path=("seed", "reroot_halogen"),
                boundary_targets=(
                    "first_domain_directional_stereo",
                    "maximal_eligible_carrier",
                ),
            ),
        ),
    ),
    SouthStarAdversarialSeed(
        seed_id="shared_carrier_chain",
        variants=(
            SouthStarAdversarialVariant(
                variant_id="two_double_bonds",
                source_smiles="F/C=C/C=C/F",
                axes=("shared_carrier", "carrier_placement"),
                mutation_path=("seed", "add_coupled_double_bond"),
                boundary_targets=(
                    "first_domain_directional_stereo",
                    "maximal_eligible_carrier",
                ),
            ),
            SouthStarAdversarialVariant(
                variant_id="branch_exposes_shared_side",
                source_smiles="C(/F)=C/C=C/F",
                axes=("shared_carrier", "branch_main_order", "carrier_placement"),
                mutation_path=("seed", "add_coupled_double_bond", "branch_first"),
                boundary_targets=(
                    "first_domain_directional_stereo",
                    "maximal_eligible_carrier",
                ),
            ),
        ),
    ),
    SouthStarAdversarialSeed(
        seed_id="ring_closure_stereo",
        variants=(
            SouthStarAdversarialVariant(
                variant_id="ring_root",
                source_smiles="C1/C=C\\CCCCC1",
                axes=("ring_closure_choice", "carrier_placement"),
                mutation_path=("seed", "ring_closure"),
                boundary_targets=("ring_stereo_monocycle",),
            ),
            SouthStarAdversarialVariant(
                variant_id="branch_ring_marker",
                source_smiles="C(/C=C\\1)CCCCC1",
                axes=("ring_closure_choice", "branch_main_order", "carrier_placement"),
                mutation_path=("seed", "ring_closure", "branch_first"),
                boundary_targets=("ring_stereo_monocycle",),
            ),
        ),
    ),
    SouthStarAdversarialSeed(
        seed_id="tetrahedral_ligand_order",
        variants=(
            SouthStarAdversarialVariant(
                variant_id="implicit_h",
                source_smiles="C[C@H](F)Cl",
                axes=("tetrahedral_ligand_order", "branch_main_order"),
                mutation_path=("seed", "implicit_h"),
                boundary_targets=("tetrahedral_atom_stereo",),
            ),
            SouthStarAdversarialVariant(
                variant_id="quaternary",
                source_smiles="[C@@](C)(F)(Cl)Br",
                axes=("tetrahedral_ligand_order", "branch_main_order"),
                mutation_path=("seed", "quaternary"),
                boundary_targets=("tetrahedral_atom_stereo",),
            ),
        ),
    ),
    SouthStarAdversarialSeed(
        seed_id="disconnected_fragment_order",
        variants=(
            SouthStarAdversarialVariant(
                variant_id="atom_then_stereo",
                source_smiles="O.F/C=C\\Cl",
                axes=("disconnected_fragment_order", "carrier_placement"),
                mutation_path=("seed", "prepend_fragment"),
                boundary_targets=(
                    "disconnected_stereo_fragments",
                    "all_fragment_orders",
                ),
            ),
            SouthStarAdversarialVariant(
                variant_id="stereo_then_atom",
                source_smiles="F/C=C\\Cl.O",
                axes=("disconnected_fragment_order", "carrier_placement"),
                mutation_path=("seed", "append_fragment"),
                boundary_targets=(
                    "disconnected_stereo_fragments",
                    "all_fragment_orders",
                ),
            ),
        ),
    ),
    SouthStarAdversarialSeed(
        seed_id="atom_text_modifiers",
        variants=(
            SouthStarAdversarialVariant(
                variant_id="explicit_hydrogen",
                source_smiles="[H][H]",
                axes=("atom_modifier", "root_choice"),
                mutation_path=("seed", "explicit_hydrogen"),
                boundary_targets=("explicit_bracket_hydrogen",),
            ),
            SouthStarAdversarialVariant(
                variant_id="charged_connected",
                source_smiles="[NH3+]C",
                axes=("atom_modifier", "root_choice"),
                mutation_path=("seed", "formal_charge", "explicit_hydrogen"),
                boundary_targets=("charged_atom_text",),
            ),
            SouthStarAdversarialVariant(
                variant_id="radical_atom",
                source_smiles="[CH3]",
                axes=("atom_modifier",),
                mutation_path=("seed", "radical_valence"),
                boundary_targets=("radical_atom_text",),
            ),
        ),
    ),
    SouthStarAdversarialSeed(
        seed_id="unsupported_feature_triggers",
        variants=(
            SouthStarAdversarialVariant(
                variant_id="quadruple_bond",
                source_smiles="C$C",
                axes=("unsupported_feature_trigger",),
                mutation_path=("seed", "unsupported_bond_type"),
                boundary_targets=("unsupported_bond_type",),
            ),
            SouthStarAdversarialVariant(
                variant_id="dative_bond",
                source_smiles="N->[O]",
                axes=("unsupported_feature_trigger",),
                mutation_path=("seed", "dative_bond"),
                boundary_targets=("dative_bond",),
            ),
            SouthStarAdversarialVariant(
                variant_id="aromatic_ring",
                source_smiles="c1ccccc1",
                axes=("unsupported_feature_trigger", "ring_closure_choice"),
                mutation_path=("seed", "aromatic_ring"),
                boundary_targets=("aromatic_ring_surface",),
            ),
        ),
    ),
)
